#include <Eigen/QR>
#include <Eigen/Eigenvalues>
#include <chrono>
#include <cmath>
#include <string>
#include "core/index.hpp"
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "core/memory.hpp"
#include "input/loading/loader.hpp"
#include "vbscf/integrals/active/space.hpp"
#include "vbscf/determinants/pairs/evaluator.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"
#include "linalg/eigensolver.hpp"

/**
 * @brief Independent double-precision cofactor check in a stable selected basis.
 *
 * Only the original VB structure columns are transformed. QR changes their
 * coordinates, never their span. Pair matrix elements use actual frame overlaps
 * and deleted cofactors, not DirectCiSigmaAction or Slater--Condon links.
 *
 * Full physical panels and ordered pair data are deliberate small-system oracle
 * storage, bounded to eight active orbitals. This is not a production H/S action
 * or an alternative orbital optimizer.
 */
int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 3)
            throw std::invalid_argument("usage: test_cofactor_point INPUT.xmi [REFERENCE_ENERGY]");
        const auto start = std::chrono::steady_clock::now();
        xmvb::memory::configure({});
        auto loaded = xmvb::vb::load_vbscf_input_with_timings(argv[1]);
        const auto& in = loaded.input;
        const auto& data = in.structure_data;
        const int n = in.orbital_preparation_input.n_active_orbitals;
        if (n > 8)
            throw std::invalid_argument("independent cofactor diagnostic is bounded to CAS <= 8");
        const auto ctx = xmvb::vb::prepare_timed_active_space_context(in).prepared_active_space;
        const auto& g = ctx.active_space_two_electron_result;
        const auto& h = ctx.active_space_one_electron_result.h1e_act;
        const xmvb::mcscf::SpinStrings sa(n, data.alpha().electron_count()),
            sb(n, data.beta().electron_count());
        const auto aa = sa.decode(), bb = sb.decode();
        const auto na = sa.size(), nb = sb.size();
        const auto ns = data.n_structures;
        xmvb::vb::ExteriorOrbitalTransform ta(sa, g.orbital_transform), tb(sb, g.orbital_transform);
        const auto rows = xmvb::core::checked_multiply(na, nb);
        if (ns <= 0 || static_cast<std::size_t>(ns) > rows)
            throw std::invalid_argument("selected structure dimension exceeds the carrier");
        Eigen::MatrixXd f(xmvb::core::checked_index<Eigen::Index>(rows), ns);
        for (Eigen::Index k = 0; k < ns; ++k) {
            Eigen::MatrixXd x = Eigen::MatrixXd::Zero(na, nb);
            const auto incidence = data.incidence();
            for (std::size_t d = 0; d < data.alpha().size(); ++d)
                for (const auto& t : incidence[d])
                    if (t.structure_index == k)
                        x(sa.index(data.alpha().mask(d)), sb.index(data.beta().mask(d))) +=
                            t.coefficient;
            ta.apply_left(&x);
            tb.apply_right(&x);
            // Fix beta-major row order; only the span of the original selected columns is used.
            f.col(k) = Eigen::Map<const Eigen::VectorXd>(x.data(), x.size());
        }
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(f);
        const Eigen::MatrixXd q = qr.householderQ() * Eigen::MatrixXd::Identity(f.rows(), ns);
        const Eigen::MatrixXd physical_metric = g.dense_active_coefficients.transpose() *
                                                in.orbital_preparation_input.ao_overlap_matrix *
                                                g.dense_active_coefficients;
        const std::vector<double> overlap(physical_metric.data(),
                                          physical_metric.data() + physical_metric.size());
        const xmvb::vb::DeterminantPairEvaluator evaluator;
        std::vector<xmvb::vb::SpinDeterminantPairEvaluation> ap, bp;
        for (std::size_t i = 0; i < na; ++i)
            for (std::size_t j = 0; j < na; ++j)
                ap.push_back(
                    evaluator.evaluate_same_spin_pair(aa[i], aa[j], overlap, h, n, g, false));
        for (std::size_t i = 0; i < nb; ++i)
            for (std::size_t j = 0; j < nb; ++j)
                bp.push_back(
                    evaluator.evaluate_same_spin_pair(bb[i], bb[j], overlap, h, n, g, false));
        Eigen::MatrixXd hq = Eigen::MatrixXd::Zero(q.rows(), ns), mq = hq;
        for (std::size_t al = 0; al < na; ++al)
            for (std::size_t bl = 0; bl < nb; ++bl)
                for (std::size_t ar = 0; ar < na; ++ar)
                    for (std::size_t br = 0; br < nb; ++br) {
                        const auto pair = evaluator.combine_spin_pair_evaluations(aa[al],
                                                                                  aa[ar],
                                                                                  ap[al * na + ar],
                                                                                  bb[bl],
                                                                                  bb[br],
                                                                                  bp[bl * nb + br],
                                                                                  n,
                                                                                  g,
                                                                                  false);
                        hq.row(al + na * bl).noalias() +=
                            pair.total_hamiltonian * q.row(ar + na * br);
                        mq.row(al + na * bl).noalias() +=
                            pair.overlap_determinant * q.row(ar + na * br);
                    }
        const Eigen::MatrixXd hs = q.transpose() * hq, ms = q.transpose() * mq;
        const xmvb::linalg::GeneralizedEigensolver solver;
        const auto dense = solver.solve_dense(
            {hs.data(), std::size_t(hs.size())}, {ms.data(), std::size_t(ms.size())}, ns);
        auto options = xmvb::linalg::make_davidson_options(ns, 1, 1e-7, 1e-7, 1e-7);
        const auto dav = solver.solve_davidson(
            [&](const Eigen::Ref<const Eigen::MatrixXd>& v) {
                return xmvb::linalg::GeneralizedEigenActionResult{hs * v, ms * v};
            },
            hs.diagonal(),
            ms.diagonal(),
            options);
        const double core = ctx.one_electron_reference_energy + loaded.nuclear_repulsion_energy;
        if (std::abs(dense.eigenvalues[0] - dav.eigenpairs.eigenvalues[0]) > 1e-9)
            throw std::runtime_error("stable cofactor dense/Davidson energies disagree");
        if (argc == 3 && std::abs(dense.eigenvalues[0] + core - std::stod(argv[2])) > 1e-9)
            throw std::runtime_error(
                "stable cofactor energy disagrees with the supplied reference");
        std::cout << std::setprecision(17) << "STABLE_COFACTOR_DENSE,"
                  << dense.eigenvalues[0] + core << '\n'
                  << "STABLE_COFACTOR_DAVIDSON," << dav.eigenpairs.eigenvalues[0] + core << '\n'
                  << "METRIC_ERROR," << (ms - Eigen::MatrixXd::Identity(ns, ns)).norm() << '\n'
                  << "H_SYMMETRY," << (hs - hs.transpose()).norm() << '\n'
                  << "CI_ITERATIONS," << dav.iterations << '\n'
                  << "SECONDS,"
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                  << '\n';
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
