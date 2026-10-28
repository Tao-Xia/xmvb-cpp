#include <cstdint>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "core/pair_index.hpp"
#include "cuda/memory/buffer.hpp"
#include "cuda/vb/point.hpp"
#include "cuda/vb/chart.hpp"
#include "integrals/active/two_electron.hpp"
#include "integrals/ao/graph.hpp"
#include "mcscf/ci/strings.hpp"
#include "vbscf/core/contracts/orbital_type.hpp"
#include "vbscf/orbitals/preparation/preparer.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"

namespace {

using Matrix = Eigen::MatrixXd;
using Input = xmvb::vb::OrbitalPreparationInput;
using DeviceMatrix = xmvb::cuda::Matrix<double>;
using Terms = std::vector<std::vector<xmvb::vb::StructureExpansionTerm>>;
using Strings = std::vector<std::vector<int>>;

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

void near(double actual, double expected, double tolerance, const std::string& label) {
    require(std::isfinite(actual) && std::isfinite(expected) &&
                std::abs(actual - expected) <= tolerance * std::max(1.0, std::abs(expected)),
            label + " mismatch");
}

void near(const Matrix& actual,
          const Matrix& expected,
          double tolerance,
          const std::string& label) {
    require(actual.rows() == expected.rows() && actual.cols() == expected.cols() &&
                actual.allFinite() && expected.allFinite() &&
                (actual - expected).norm() <= tolerance * std::max(1.0, expected.norm()),
            label + " mismatch");
}

DeviceMatrix upload(xmvb::cuda::Context& context, const Matrix& source) {
    DeviceMatrix result(context, source.rows(), source.cols());
    xmvb::cuda::copy_to_device(context, result.view(), source);
    return result;
}

Matrix download(xmvb::cuda::Context& context, xmvb::linalg::ConstMatrixView source) {
    Matrix result(source.rows, source.cols);
    xmvb::cuda::copy_to_host(context, result, source);
    return result;
}

int pair(int i, int j) {
    return xmvb::core::PairIndex::packed_pair_index(i, j);
}

struct AoFixture {
    static constexpr int basis = 4;
    Matrix core = Matrix::Zero(basis, basis);
    Matrix eri = Matrix::Zero(10, 10);
    std::vector<std::int64_t> left, right;
    std::vector<double> values;

    AoFixture() {
        core << -1.13, 0.07, -0.02, 0.03, 0.07, -0.82, 0.05, -0.01, -0.02, 0.05, -0.51, 0.04, 0.03,
            -0.01, 0.04, -0.25;
        for (int p = 0; p < 10; ++p)
            for (int q = 0; q <= p; ++q) {
                const double value =
                    0.035 * std::cos(0.19 * (p + 1) * (q + 2)) + 0.006 / (p + q + 1.0);
                eri(p, q) = eri(q, p) = value;
                left.push_back(p);
                right.push_back(q);
                values.push_back(value);
            }
    }

    xmvb::integrals::AoPairGraph graph() const {
        return xmvb::integrals::build_ao_pair_graph(left, right, values, basis);
    }

    double integral(int i, int j, int k, int l) const {
        return eri(pair(i, j), pair(k, l));
    }

    Matrix jk(const Matrix& density) const {
        Matrix result = Matrix::Zero(basis, basis);
        for (int i = 0; i < basis; ++i)
            for (int j = 0; j < basis; ++j)
                for (int k = 0; k < basis; ++k)
                    for (int l = 0; l < basis; ++l)
                        result(i, j) +=
                            (2.0 * integral(i, j, k, l) - integral(i, k, j, l)) * density(k, l);
        return result;
    }

    Matrix pair_features(const Matrix& orbitals) const {
        const int active = orbitals.cols();
        const int pairs = active * (active + 1) / 2;
        Matrix result(10, pairs);
        int row = 0;
        for (int i = 0; i < basis; ++i)
            for (int j = 0; j <= i; ++j, ++row)
                for (int a = 0; a < active; ++a)
                    for (int b = 0; b <= a; ++b)
                        result(row, pair(a, b)) = orbitals(i, a) * orbitals(j, b) +
                                                  (i == j ? 0.0 : orbitals(j, a) * orbitals(i, b));
        return result;
    }

    Matrix active_pair_kernel(const Matrix& orbitals) const {
        const Matrix features = pair_features(orbitals);
        return features.transpose() * eri * features;
    }

    xmvb::integrals::ActiveSpaceTwoElectronResult packed(const Matrix& kernel) const {
        xmvb::integrals::ActiveSpaceTwoElectronResult result;
        auto& output = result.packed_active_two_electron_integrals;
        output.resize(kernel.rows() * (kernel.rows() + 1) / 2);
        for (int p = 0; p < kernel.rows(); ++p)
            for (int q = 0; q <= p; ++q)
                output[xmvb::core::PairIndex::packed_pair_of_pairs_index(p, q)] = kernel(p, q);
        return result;
    }
};

Input orbital_input(bool inactive) {
    Input input;
    input.n_basis_functions = 4;
    input.n_active_orbitals = 2;
    input.n_orbitals = inactive ? 3 : 2;
    input.n_active_electrons = 2;
    input.n_total_electrons = inactive ? 4 : 2;
    input.orbital_type = xmvb::vb::kOrbitalTypeHao;
    input.ao_overlap_matrix = Matrix::Identity(4, 4);
    input.ao_overlap_matrix.array() += 0.03;
    input.ao_overlap_matrix(0, 2) += 0.018;
    input.ao_overlap_matrix(2, 0) += 0.018;
    const std::vector<std::vector<int>> supports =
        inactive ? std::vector<std::vector<int>>{{1, 2, 3}, {2, 3, 4}, {1, 3, 4}}
                 : std::vector<std::vector<int>>{{1, 2, 3}, {2, 3, 4}};
    const int orbitals = static_cast<int>(input.n_orbitals);
    input.orbital_basis_index_table.assign(4 * orbitals, 0);
    input.orbital_value_table.assign(4 * orbitals, 0.0);
    input.orbital_basis_counts.assign(orbitals, 3);
    input.original_orbital_basis_counts.assign(orbitals, 3);
    for (int o = 0; o < orbitals; ++o)
        for (int j = 0; j < 3; ++j) {
            input.orbital_basis_index_table[4 * o + j] = supports[o][j];
            input.orbital_value_table[4 * o + j] = (j == o % 3 ? 0.83 : 0.12 + 0.07 * j + 0.02 * o);
        }
    input.original_orbital_basis_counts.back() = 2;
    return input;
}

struct SelectedSpace {
    Terms terms;
    xmvb::vb::SameSpinPairCacheContext topology;
};

SelectedSpace selected_space(int structures) {
    const Strings alpha = xmvb::mcscf::SpinStrings(2, 1).decode();
    const Strings beta = alpha;
    const double covalent = 1.0 / std::sqrt(2.0);
    Terms terms(4);
    // Product order: 00, 01, 10, 11. Structure 0 is covalent, followed
    // by the two ionic structures. Keep the complete spin carrier in all cases.
    terms[1].push_back({0, covalent});
    terms[2].push_back({0, covalent});
    if (structures >= 2)
        terms[0].push_back({1, 1.0});
    if (structures >= 3)
        terms[3].push_back({2, 1.0});
    const Strings product_alpha{alpha[0], alpha[0], alpha[1], alpha[1]};
    const Strings product_beta{beta[0], beta[1], beta[0], beta[1]};
    return {std::move(terms),
            xmvb::vb::build_same_spin_pair_topology(product_alpha, product_beta, 2)};
}

xmvb::vb::StructureAction
source_action(const SelectedSpace& space,
              int structures,
              const Matrix& overlap,
              const Matrix& one,
              const xmvb::integrals::ActiveSpaceTwoElectronResult& pair_kernel) {
    const std::vector<double> overlap_data(overlap.data(), overlap.data() + overlap.size());
    return xmvb::vb::StructureAction(space.terms,
                                     structures,
                                     space.topology,
                                     overlap_data,
                                     one,
                                     pair_kernel,
                                     overlap.rows(),
                                     nullptr,
                                     xmvb::vb::StructureActionMode::ActionOnly);
}

double independent_energy(const Input& input,
                          const AoFixture& ao,
                          const SelectedSpace& space,
                          int structures,
                          const Matrix& coefficients,
                          double nuclear) {
    const auto prepared = xmvb::vb::ActiveSpaceOrbitalPreparer{}.prepare(input);
    const int inactive = static_cast<int>((input.n_total_electrons - input.n_active_electrons) / 2);
    const Matrix p = prepared.inactive_density_matrix;
    const Matrix effective = ao.core + ao.jk(p);
    const double core = (p.array() * (ao.core + effective).array()).sum();
    const int active = input.n_active_orbitals;
    const Matrix t = prepared.auxiliary_orbital_matrix.middleCols(inactive, active);
    const Matrix overlap =
        Eigen::Map<const Matrix>(prepared.active_orbital_overlap_matrix.data(), active, active);
    const Matrix h = t.transpose() * effective * t;
    const Matrix g = ao.active_pair_kernel(t);
    const auto action = source_action(space, structures, overlap, h, ao.packed(g));
    const auto result = action.apply(coefficients);
    const double numerator = (coefficients.array() * result.hamiltonian.array()).sum();
    const double denominator = (coefficients.array() * result.overlap.array()).sum();
    require(denominator > 0.0, "CPU selected overlap is nonpositive");
    return nuclear + core + numerator / denominator;
}

struct Derivatives {
    Matrix raw, ci;
};

Derivatives gradient(xmvb::cuda::Context& context,
                     xmvb::cuda::VbPoint& point,
                     const Matrix& raw,
                     const Matrix& ci) {
    auto draw = upload(context, raw), dci = upload(context, ci);
    point.evaluate(draw.view(), dci.view());
    DeviceMatrix gout(context, raw.rows(), raw.cols()), cout(context, ci.rows(), 1);
    point.gradient(gout.view(), cout.view());
    Derivatives result{download(context, gout.view()), download(context, cout.view())};
    // Omitting a component must not change either covector, including at
    // finite CI residual. The following independent finite differences use
    // the same result for both raw-orbital and structure directions.
    point.gradient(gout.view());
    near(download(context, gout.view()), result.raw, 2e-11, "orbital-only gradient");
    point.gradient({}, cout.view());
    near(download(context, cout.view()), result.ci, 2e-11, "structure-only gradient");
    return result;
}

double energy(xmvb::cuda::Context& context,
              xmvb::cuda::VbPoint& point,
              const Matrix& raw,
              const Matrix& ci) {
    auto draw = upload(context, raw), dci = upload(context, ci);
    return point.evaluate(draw.view(), dci.view());
}

double bilinear(const Derivatives& left, const Derivatives& right) {
    return (left.raw.array() * right.raw.array()).sum() +
           (left.ci.array() * right.ci.array()).sum();
}

void check_case(bool inactive, int structures, bool source_order = false, bool source_hao = false) {
    constexpr double nuclear = 0.37;
    const AoFixture ao;
    Input input = orbital_input(inactive);
    auto space = selected_space(structures);
    if (source_order) {
        input.n_active_orbitals = input.n_orbitals = 4;
        input.n_total_electrons = input.n_active_electrons = 4;
        input.orbital_type = source_hao ? xmvb::vb::kOrbitalTypeHao : xmvb::vb::kOrbitalTypeOeo;
        const int count = source_hao ? 3 : 4;
        input.orbital_basis_counts.assign(4, count);
        input.original_orbital_basis_counts.assign(4, count);
        input.orbital_basis_index_table.resize(16);
        input.orbital_value_table.resize(16);
        for (int o = 0; o < 4; ++o)
            for (int j = 0; j < 4; ++j) {
                const int ao = source_hao ? (o + j) % 4 : j;
                input.orbital_basis_index_table[4 * o + j] = j < count ? ao + 1 : 0;
                input.orbital_value_table[4 * o + j] =
                    j < count ? (o == ao ? 1.0 : 0.04 * std::sin(o + 2 * ao + 1.)) : 0.0;
            }
        const Strings alpha{{0, 1}, {0, 2}, {1, 3}}, beta{{0, 1}, {0, 3}, {2, 3}};
        space.terms = {{{0, 0.9}}, {{1, 0.7}}, {{2, -0.8}}};
        space.topology = xmvb::vb::build_orthogonal_spin_topology(alpha, beta, 4);
    }
    const int active = input.n_active_orbitals, pairs = active * (active + 1) / 2;
    const Matrix identity = Matrix::Identity(active, active);
    const auto source = source_action(space,
                                      structures,
                                      identity,
                                      Matrix::Zero(active, active),
                                      ao.packed(Matrix::Zero(pairs, pairs)));
    require(source.storage().orthogonal_direct_ci,
            "selected fixture does not retain a complete orthogonal carrier");
    xmvb::cuda::Context context(256ULL * 1024ULL * 1024ULL);
    const xmvb::cuda::AoMetric ao_metric(context, input.ao_overlap_matrix);
    xmvb::cuda::StructureAction action(context, source);
    if (source_order)
        require(action.uses_operator_first(),
                "source-order point test did not enter production executor");
    xmvb::cuda::ExactAoPairOperator integrals(context, ao.graph(), AoFixture::basis);
    auto dcore = upload(context, ao.core);
    xmvb::cuda::VbPoint point(
        context, input, action, integrals, dcore.view(), ao_metric.overlap(), nuclear);
    Matrix raw = Eigen::Map<const Matrix>(input.orbital_value_table.data(), 4, input.n_orbitals);
    Matrix ci(structures, 1);
    for (int s = 0; s < structures; ++s)
        ci(s, 0) = 0.83 - 0.27 * s;
    near(energy(context, point, raw, ci),
         independent_energy(input, ao, space, structures, ci, nuclear),
         3e-10,
         "joint VB energy versus independent CPU source");
    const auto accepted = gradient(context, point, raw, ci);
    {
        auto orbital = upload(context, raw), coefficients = upload(context, ci);
        const auto original = point.evaluate(orbital.view(), coefficients.view());
        DeviceMatrix mc(context, structures, 1), go(context, raw.rows(), raw.cols()),
            gc(context, structures, 1);
        action.apply_overlap(coefficients.view(), mc.view());
        near(point.evaluate_coefficients(coefficients.view(), mc.view()),
             original,
             2e-11,
             "prepared overlap primal energy at finite CI residual");
        point.gradient(go.view(), gc.view());
        near(
            download(context, go.view()), accepted.raw, 2e-11, "prepared overlap orbital gradient");
        near(download(context, gc.view()), accepted.ci, 2e-11, "prepared overlap CI gradient");
    }
    const double epsilon = 1e-5;
    for (int o = 0; o < raw.cols(); ++o) {
        const int count = xmvb::vb::differentiable_sparse_orbital_parameter_count(input, o);
        for (int j = 0; j < count; ++j) {
            Matrix plus = raw, minus = raw;
            plus(j, o) += epsilon;
            minus(j, o) -= epsilon;
            const double reference =
                (energy(context, point, plus, ci) - energy(context, point, minus, ci)) /
                (2 * epsilon);
            near(accepted.raw(j, o), reference, 8e-6, "joint raw gradient");
        }
        for (int j = count; j < raw.rows(); ++j)
            require(accepted.raw(j, o) == 0.0, "fixed HAO slot gradient is nonzero");
    }
    for (int s = 0; s < structures; ++s) {
        Matrix plus = ci, minus = ci;
        plus(s, 0) += epsilon;
        minus(s, 0) -= epsilon;
        const double reference =
            (energy(context, point, raw, plus) - energy(context, point, raw, minus)) /
            (2 * epsilon);
        near(accepted.ci(s, 0), reference, 8e-6, "finite CI residual gradient");
    }

    Matrix draw = Matrix::Zero(raw.rows(), raw.cols());
    Matrix draw2 = draw;
    for (int o = 0; o < raw.cols(); ++o)
        for (int j = 0; j < xmvb::vb::differentiable_sparse_orbital_parameter_count(input, o);
             ++j) {
            draw(j, o) = 0.18 * std::sin(0.3 + 2 * j + o);
            draw2(j, o) = 0.16 * std::cos(0.7 + j + 2 * o);
        }
    Matrix dci(structures, 1), dci2(structures, 1);
    for (int s = 0; s < structures; ++s) {
        dci(s, 0) = 0.14 * std::cos(0.2 + s);
        dci2(s, 0) = 0.12 * std::sin(0.6 + 2 * s);
    }
    const auto hvp = [&](const Matrix& orbital, const Matrix& structure) {
        auto base_raw = upload(context, raw), base_ci = upload(context, ci);
        point.evaluate(base_raw.view(), base_ci.view());
        DeviceMatrix prepared_mc(context, structures, 1);
        action.apply_overlap(base_ci.view(), prepared_mc.view());
        point.evaluate_coefficients(base_ci.view(), prepared_mc.view());
        xmvb::cuda::AcceptedOrbitalChart chart(context, input, false, ao_metric);
        chart.rebuild(point.geometry());
        auto diagonal = point.hessian_diagonal(chart.accepted_mo_frame());
        const auto version = action.version();
        point.activate();
        require(action.version() == version, "unchanged point rebuilt shared action");
        const auto before_hvp = action.apply(ci);
        auto dorbital = upload(context, orbital), dstructure = upload(context, structure);
        DeviceMatrix out_raw(context, raw.rows(), raw.cols());
        DeviceMatrix out_ci(context, structures, 1);
        DeviceMatrix ci_raw(context, raw.rows(), raw.cols()), ci_image(context, structures, 1);
        DeviceMatrix pure_ci_overlap(context, structures, 1);
        point.hessian(dorbital.view(),
                      dstructure.view(),
                      out_raw.view(),
                      out_ci.view(),
                      ci_raw.view(),
                      ci_image.view(),
                      pure_ci_overlap.view());
        cublasPointerMode_t pointer_mode;
        xmvb::cuda::check_cublas(cublasGetPointerMode(context.blas(), &pointer_mode),
                                 "point test scalar mode");
        require(pointer_mode == CUBLAS_POINTER_MODE_HOST,
                "device normalization leaked cuBLAS scalar mode");
        near(download(context, pure_ci_overlap.view()),
             action.apply(structure).overlap,
             2e-10,
             "joint pure CI overlap image");
        const auto direct_raw = download(context, out_raw.view());
        const auto direct_ci = download(context, out_ci.view());
        const auto split_raw = download(context, ci_raw.view()),
                   split_ci = download(context, ci_image.view());
        auto zero_raw = upload(context, Matrix::Zero(raw.rows(), raw.cols()).eval());
        auto zero_ci = upload(context, Matrix::Zero(structures, 1).eval());
        // Independent full actions are test oracles only. Production obtains both
        // columns in one sigma/adjoint sweep and never makes these extra calls.
        point.hessian(zero_raw.view(), dstructure.view(), out_raw.view(), out_ci.view());
        near(split_raw, download(context, out_raw.view()), 2e-10, "split CI orbital image");
        near(split_ci, download(context, out_ci.view()), 2e-10, "split CI structure image");
        point.hessian(zero_raw.view(),
                      dstructure.view(),
                      out_raw.view(),
                      out_ci.view(),
                      ci_raw.view(),
                      ci_image.view(),
                      pure_ci_overlap.view());
        near(download(context, pure_ci_overlap.view()),
             action.apply(structure).overlap,
             2e-10,
             "zero-orbital pure CI overlap image");
        point.hessian(dorbital.view(), zero_ci.view(), out_raw.view(), out_ci.view());
        near(direct_raw - split_raw,
             download(context, out_raw.view()),
             2e-10,
             "split orbital column");
        near(direct_ci - split_ci,
             download(context, out_ci.view()),
             2e-10,
             "split orbital structure forcing");
        xmvb::cuda::copy_to_device(context, dorbital.view(), (-0.7 * orbital).eval());
        xmvb::cuda::copy_to_device(context, dstructure.view(), (-0.7 * structure).eval());
        point.hessian(dorbital.view(), dstructure.view(), out_raw.view(), out_ci.view());
        near(download(context, out_raw.view()),
             -0.7 * direct_raw,
             2e-10,
             "device-scalar raw HVP linearity");
        near(download(context, out_ci.view()),
             -0.7 * direct_ci,
             2e-10,
             "device-scalar CI HVP linearity");
        xmvb::cuda::copy_to_device(context, dorbital.view(), orbital);
        xmvb::cuda::copy_to_device(context, dstructure.view(), structure);
        point.hessian(dorbital.view(), dstructure.view(), out_raw.view(), out_ci.view());
        near(download(context, out_raw.view()), direct_raw, 2e-10, "replayed raw HVP");
        near(download(context, out_ci.view()), direct_ci, 2e-10, "replayed CI HVP");
        // Without accepted AO response maps, the two density adjoints must share
        // one batched JK sweep and agree with the prepared projection path.
        point.evaluate(base_raw.view(), base_ci.view());
        point.hessian(dorbital.view(),
                      dstructure.view(),
                      out_raw.view(),
                      out_ci.view(),
                      ci_raw.view(),
                      ci_image.view());
        near(download(context, out_raw.view()), direct_raw, 2e-10, "unprepared split raw HVP");
        near(download(context, out_ci.view()), direct_ci, 2e-10, "unprepared split CI HVP");
        near(download(context, ci_raw.view()), split_raw, 2e-10, "unprepared CI orbital column");
        near(download(context, ci_image.view()), split_ci, 2e-10, "unprepared CI structure column");
        const auto after_hvp = action.apply(ci);
        near(after_hvp.hamiltonian,
             before_hvp.hamiltonian,
             2e-10,
             "borrowed HVP carriers corrupted the next H action");
        near(after_hvp.overlap,
             before_hvp.overlap,
             2e-10,
             "borrowed HVP carriers corrupted the next overlap action");
        return Derivatives{download(context, out_raw.view()), download(context, out_ci.view())};
    };
    const auto image = hvp(draw, dci);
    const auto reference_plus = gradient(context, point, raw + epsilon * draw, ci + epsilon * dci);
    const auto reference_minus = gradient(context, point, raw - epsilon * draw, ci - epsilon * dci);
    near(image.raw,
         (reference_plus.raw - reference_minus.raw) / (2 * epsilon),
         4e-5,
         "joint orbital HVP");
    near(image.ci, (reference_plus.ci - reference_minus.ci) / (2 * epsilon), 4e-5, "joint CI HVP");
    const auto image2 = hvp(draw2, dci2);
    near(bilinear({draw2, dci2}, image),
         bilinear({draw, dci}, image2),
         4e-5,
         "joint Hessian bilinear symmetry");
}

} // namespace

int main() {
    try {
        for (bool inactive : {false, true})
            for (int structures : {1, 2, 3})
                check_case(inactive, structures);
        check_case(false, 3, true);
        check_case(false, 3, true, true);
        std::cout << "CUDA VB point energy, gradient, and HVP checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
