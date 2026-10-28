#include "orbital_snapshot.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/SVD>

#include "core/memory.hpp"
#include "core/pair_index.hpp"
#include "input/loading/loader.hpp"
#include "mcscf/ci/similarity.hpp"
#include "mcscf/ci/sigma.hpp"
#include "vbscf/derivatives/gradient/orbital/evaluator.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"
#include "vbscf/structures/overlap/metric.hpp"

namespace {

long double pairing(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    long double result = 0;
    for (Eigen::Index i = 0; i < a.size(); ++i)
        result += static_cast<long double>(a[i]) * b[i];
    return result;
}

void integrals(const xmvb::vb::StructureAction& action) {
    const auto& physical = action.orthogonal_integrals();
    const auto& r = action.orthogonal_view().alpha.orbital_transform();
    const xmvb::mcscf::SimilarityHamiltonian canonical(physical, r);
    const Eigen::Index n = r.rows(), d = n * n;
    const Eigen::MatrixXd inverse = r.inverse();
    Eigen::MatrixXd raw_h(n, n), raw_g(d, d), physical_g(d, d), back(d, d);
    for (Eigen::Index i = 0; i < n; ++i)
        for (Eigen::Index j = 0; j < n; ++j) {
            raw_h(i, j) = canonical.element(std::uint64_t{1} << i, 0, std::uint64_t{1} << j, 0);
            for (Eigen::Index k = 0; k < n; ++k)
                for (Eigen::Index l = 0; l < n; ++l) {
                    raw_g(i * n + j, k * n + l) = canonical.g(i, j, k, l);
                    physical_g(i * n + j, k * n + l) =
                        physical.pair_kernel(xmvb::core::PairIndex::packed_pair_index(i, j),
                                             xmvb::core::PairIndex::packed_pair_index(k, l));
                    back(i * n + j, k * n + l) = r(i, k) * inverse(l, j);
                }
        }
    const auto gamma = [](Eigen::Index operations) {
        const double product = operations * std::numeric_limits<double>::epsilon();
        return product / (1.0 - product);
    };
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(r);
    std::cout << "INTEGRALS,condition_r="
              << svd.singularValues()[0] / svd.singularValues().tail(1)[0]
              << ",one_roundtrip_error=" << (r * raw_h * inverse - physical.one_electron).norm()
              << ",one_data_norm=" << physical.one_electron.norm() << ",one_gamma=" << gamma(4 * n)
              << ",two_roundtrip_error=" << (back * raw_g * back.transpose() - physical_g).norm()
              << ",two_data_norm=" << physical_g.norm() << ",two_gamma=" << gamma(4 * d) << '\n';
}

void state(const char* name,
           const Eigen::MatrixXd& c,
           const xmvb::vb::StructureActionResult& images) {
    for (Eigen::Index i = 0; i < c.cols(); ++i) {
        const auto norm = pairing(c.col(i), images.overlap.col(i));
        const auto energy = pairing(c.col(i), images.hamiltonian.col(i)) / norm;
        std::cout << "STATE,order=" << name << ",root=" << i << ",norm=" << norm
                  << ",energy=" << energy << '\n';
    }
}

/** @brief Bounded independent QR oracle; never used by a production action. */
void physical_reference(const xmvb::vb::StructureAction& action,
                        const xmvb::vb::SelectedMetric* metric,
                        double accepted_energy) {
    const auto& view = action.orthogonal_view();
    const Eigen::Index n = action.n_structures(), na = view.alpha.dimension(),
                       nb = view.beta.dimension();
    if (n > 8 || na > 4096 / nb)
        return;
    Eigen::MatrixXd carrier = action.expand_structure_block(Eigen::MatrixXd::Identity(n, n));
    view.alpha.apply_left(&carrier);
    for (Eigen::Index i = 0; i < n; ++i) {
        auto block = carrier.middleCols(i * nb, nb);
        view.beta.apply_right_block(block);
    }
    const Eigen::Map<const Eigen::MatrixXd> b(carrier.data(), na * nb, n);
    const Eigen::HouseholderQR<Eigen::MatrixXd> qr(b);
    const Eigen::MatrixXd r = qr.matrixQR().topLeftCorner(n, n).triangularView<Eigen::Upper>();
    const Eigen::MatrixXd q = qr.householderQ() * Eigen::MatrixXd::Identity(na * nb, n);
    const Eigen::Map<const Eigen::MatrixXd> panel(q.data(), na, nb * n);
    const Eigen::MatrixXd image = view.sigma.apply(panel);
    const Eigen::Map<const Eigen::MatrixXd> hq(image.data(), na * nb, n);
    const Eigen::MatrixXd h = q.transpose() * hq;
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> roots(h);
    if (roots.info() != Eigen::Success)
        throw std::runtime_error("physical QR reference diagonalization failed");
    std::cout << "PHYSICAL_QR,energy=" << roots.eigenvalues()[0]
              << ",accepted_energy=" << accepted_energy
              << ",energy_error=" << accepted_energy - roots.eigenvalues()[0]
              << ",orthogonality=" << (q.transpose() * q - Eigen::MatrixXd::Identity(n, n)).norm();
    if (metric) {
        const Eigen::MatrixXd inverse =
            r.triangularView<Eigen::Upper>().solve(Eigen::MatrixXd::Identity(n, n));
        const Eigen::MatrixXd factor = metric->factor.matrixU() * inverse;
        std::cout << ",metric_error="
                  << (factor.transpose() * factor - Eigen::MatrixXd::Identity(n, n)).norm();
    }
    std::cout << '\n';
}

} // namespace

/** @brief Audit equivalent contractions at recorded orbitals, without SCF optimization or HVP. */
int main(int argc, char** argv) {
    try {
        if (argc != 4)
            throw std::invalid_argument(
                "usage: check_structure_accuracy INPUT.xmi RAW.bin STATE.bin");
        xmvb::memory::configure({});
        auto loaded = xmvb::vb::load_vbscf_input_with_timings(argv[1]);
        auto& input = loaded.input;
        auto& o = input.orbital_preparation_input;
        xmvb::test::set_orbitals(o, xmvb::test::read_orbitals(argv[2], o));
        const Eigen::MatrixXd c =
            xmvb::test::read_coefficients(argv[3], o, input.structure_data.n_structures);
        const auto accepted =
            xmvb::vb::OrbitalGradientEvaluator{}.evaluate_without_reference_energy_gradient(
                input,
                {0},
                {1.0},
                loaded.nuclear_repulsion_energy,
                xmvb::vb::StructureEigensolver::Davidson,
                xmvb::vb::StructureSolveAccuracy{},
                Eigen::MatrixXd{});
        const auto& context = *accepted.second_order_context;
        if (!context.structure_action || !context.structure_action->storage().orthogonal_direct_ci)
            throw std::runtime_error("accuracy audit requires the orthogonal-CI structure action");
        const auto& action = *context.structure_action;
        const Eigen::Index count = std::min<Eigen::Index>(c.rows(), o.n_active_orbitals);
        Eigen::MatrixXd probes = Eigen::MatrixXd::Zero(c.rows(), count + c.cols());
        probes.topLeftCorner(count, count).setIdentity();
        probes.rightCols(c.cols()) = c;
        const auto normal = action.apply(probes);
        const auto reference = action.apply_sources(probes, xmvb::workspace::available());
        const Eigen::MatrixXd h0 = normal.hamiltonian.topLeftCorner(count, count);
        const Eigen::MatrixXd h1 = reference.hamiltonian.topLeftCorner(count, count);
        const Eigen::MatrixXd m0 = normal.overlap.topLeftCorner(count, count);
        const Eigen::MatrixXd m1 = reference.overlap.topLeftCorner(count, count);
        const Eigen::MatrixXd metric = 0.5 * (m1 + m1.transpose());
        const Eigen::LLT<Eigen::MatrixXd> factor(metric);
        if (factor.info() != Eigen::Success)
            throw std::runtime_error("reference probe metric is not positive definite");
        const auto whiten = [&](const Eigen::MatrixXd& matrix) -> Eigen::MatrixXd {
            const Eigen::MatrixXd first = factor.matrixL().solve(matrix);
            return factor.matrixL().solve(first.transpose()).transpose();
        };
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(metric);
        std::cout << std::setprecision(17) << "ACTION,n_structures=" << c.rows()
                  << ",probe_rank=" << count << ",condition_probe_metric="
                  << spectrum.eigenvalues().tail(1)[0] / spectrum.eigenvalues()[0]
                  << ",h_difference=" << (h0 - h1).norm() << ",m_difference=" << (m0 - m1).norm()
                  << ",physical_h_difference=" << whiten(h0 - h1).norm()
                  << ",physical_m_difference=" << whiten(m0 - m1).norm()
                  << ",physical_h_skew=" << whiten(h0 - h0.transpose()).norm()
                  << ",reference_h_skew=" << whiten(h1 - h1.transpose()).norm() << '\n';
        if (context.selected_metric) {
            const Eigen::MatrixXd cached =
                context.selected_metric->overlap.topLeftCorner(count, count);
            std::cout << "METRIC,cache_physical_difference=" << whiten(cached - m1).norm() << '\n';
            const Eigen::MatrixXd coordinates = context.selected_metric->factor.matrixU() * c;
            const Eigen::MatrixXd factored_image =
                context.selected_metric->factor.matrixL() * coordinates;
            std::cout << "NORMALIZATION,factored=" << coordinates.squaredNorm()
                      << ",factored_image=" << pairing(c.col(0), factored_image.col(0))
                      << ",cached_image="
                      << pairing(c.col(0), (context.selected_metric->overlap * c).col(0)) << '\n';
        }
        const auto single = action.apply(c);
        const auto source = action.apply_sources(c, xmvb::workspace::available());
        state("default_single", c, single);
        state("orthogonal_single", c, source);
        state("default_block",
              c,
              {normal.hamiltonian.rightCols(c.cols()), normal.overlap.rightCols(c.cols())});
        state("orthogonal_block",
              c,
              {reference.hamiltonian.rightCols(c.cols()), reference.overlap.rightCols(c.cols())});
        integrals(action);
        physical_reference(
            action, context.selected_metric.get(), context.selected_state_energies[0]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
