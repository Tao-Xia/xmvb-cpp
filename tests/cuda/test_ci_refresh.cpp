#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "core/pair_index.hpp"
#include "cuda/memory/buffer.hpp"
#include "cuda/vb/action.hpp"
#include "cuda/vb/ci.hpp"
#include "cuda/vb/metric.hpp"
#include "integrals/active/two_electron.hpp"
#include "integrals/active/transform.hpp"
#include "mcscf/ci/strings.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/structures/assembly/action.hpp"

namespace {

using Matrix = Eigen::MatrixXd;
using Terms = std::vector<std::vector<xmvb::vb::StructureExpansionTerm>>;
using Strings = std::vector<std::vector<int>>;
using DeviceMatrix = xmvb::cuda::Matrix<double>;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void near(double actual, double expected, double tolerance, const char* message) {
    require(std::isfinite(actual) && std::isfinite(expected) &&
                std::abs(actual - expected) <= tolerance * std::max(1.0, std::abs(expected)),
            message);
}

void near(const Matrix& actual, const Matrix& expected, double tolerance, const char* message) {
    require(actual.rows() == expected.rows() && actual.cols() == expected.cols() &&
                actual.allFinite() && expected.allFinite() &&
                (actual - expected).norm() <= tolerance * std::max(1.0, expected.norm()),
            message);
}

DeviceMatrix upload(xmvb::cuda::Context& context, const Matrix& host) {
    DeviceMatrix result(context, host.rows(), host.cols());
    xmvb::cuda::copy_to_device(context, result.view(), host);
    return result;
}

Matrix download(xmvb::cuda::Context& context, xmvb::linalg::ConstMatrixView source) {
    Matrix result(source.rows, source.cols);
    xmvb::cuda::copy_to_host(context, result, source);
    return result;
}

Matrix upper_frame(bool changed) {
    Matrix upper = Matrix::Identity(2, 2);
    upper(0, 0) = changed ? 1.17 : 1.06;
    upper(0, 1) = changed ? -0.13 : 0.08;
    upper(1, 1) = changed ? 0.89 : 0.96;
    return upper;
}

Matrix one_electron(bool changed) {
    Matrix h(2, 2);
    h << (changed ? -0.70 : -0.91), (changed ? 0.12 : 0.09), (changed ? 0.12 : 0.09),
        (changed ? -0.22 : -0.31);
    return h;
}

Matrix pair_kernel(bool changed) {
    Matrix g(3, 3);
    for (int p = 0; p < 3; ++p)
        for (int q = 0; q < 3; ++q)
            g(p, q) = 0.035 * std::cos(0.37 * (p + 1) * (q + 1)) +
                      (changed ? 0.012 : 0.0) * std::sin(0.4 * (p + q + 1));
    return 0.5 * (g + g.transpose()).eval();
}

xmvb::integrals::ActiveSpaceTwoElectronResult packed(const Matrix& g) {
    xmvb::integrals::ActiveSpaceTwoElectronResult eri;
    eri.packed_active_two_electron_integrals.resize(
        xmvb::integrals::packed_active_two_electron_integral_count(2));
    for (int p = 0; p < 3; ++p)
        for (int q = 0; q <= p; ++q)
            eri.packed_active_two_electron_integrals
                [xmvb::core::PairIndex::packed_pair_of_pairs_index(p, q)] = g(p, q);
    return eri;
}

Terms terms(int count) {
    Terms result(4);
    result[0].push_back({0, 1.0});
    if (count == 3) {
        constexpr double mixed = 0.7071067811865475244;
        result[1].push_back({1, mixed});
        result[2].push_back({1, mixed});
        result[3].push_back({2, 1.0});
    } else {
        constexpr double mixed = 0.7071067811865475244;
        result[1].push_back({1, mixed});
        result[2].push_back({1, mixed});
    }
    return result;
}

xmvb::vb::StructureAction source(int count, bool changed_frame, bool changed_integrals) {
    const Strings strings = xmvb::mcscf::SpinStrings(2, 1).decode();
    const Strings product_alpha{strings[0], strings[0], strings[1], strings[1]};
    const Strings product_beta{strings[0], strings[1], strings[0], strings[1]};
    const auto topology = xmvb::vb::build_same_spin_pair_topology(product_alpha, product_beta, 2);
    const Matrix upper = upper_frame(changed_frame);
    const Matrix overlap = upper.transpose() * upper;
    const std::vector<double> overlap_data(overlap.data(), overlap.data() + overlap.size());
    // StructureAction accepts raw nonorthogonal active integrals and internally
    // changes to the upper-frame orthogonal carrier. The device update accepts
    // that orthogonal carrier directly, so reconstruct the raw CPU oracle here.
    const Matrix raw_h = upper.transpose() * one_electron(changed_integrals) * upper;
    const Matrix raw_g =
        xmvb::integrals::transform_pair_kernel(pair_kernel(changed_integrals), upper);
    return xmvb::vb::StructureAction(terms(count),
                                     count,
                                     topology,
                                     overlap_data,
                                     raw_h,
                                     packed(raw_g),
                                     2,
                                     nullptr,
                                     xmvb::vb::StructureActionMode::ActionOnly);
}

void check_case(bool full_space) {
    const int count = full_space ? 3 : 2;
    const Strings strings = xmvb::mcscf::SpinStrings(2, 1).decode();
    auto initial = source(count, false, false);
    require(initial.storage().orthogonal_direct_ci, "CI refresh fixture has no exact carrier");
    xmvb::cuda::Context context(128ULL * 1024 * 1024);
    xmvb::cuda::StructureAction action(context, initial);
    xmvb::cuda::SpinMetric spin(context, initial, strings, strings, 4096);
    xmvb::cuda::StructureCi ci(context, action, spin, full_space);
    auto options = xmvb::linalg::make_davidson_options(count, 1, 1e-10, 1e-9, 1e-10);

    DeviceMatrix model_h(context, count, 1), model_m(context, count, 1);
    action.diagonal_model(model_h.view(), model_m.view());
    const Matrix old_h = download(context, model_h.view());
    const Matrix old_m = download(context, model_m.view());

    auto first = ci.solve_device(options);
    const auto origin_point = ci.point();
    const auto first_preparations = ci.prepared_points();
    require(ci.point() == origin_point && ci.prepared_points() == first_preparations,
            "unchanged CI point was rebuilt");
    require(first.eigenpairs.eigenvector_matrix.rows() == count &&
                first.hamiltonian_eigenvectors.rows() == count &&
                first.overlap_view().rows == count,
            "device CI did not retain full-dimension roots and images");
    auto old_roots = first.eigenpairs.eigenvector_matrix.view();
    DeviceMatrix frozen_before(context, count, 1), frozen_after(context, count, 1);
    if (!full_space)
        ci.inverse_augmented_hessian_model(old_roots,
                                           first.eigenpairs.eigenvalues.front(),
                                           -0.01,
                                           frozen_before.view(),
                                           origin_point.get());

    auto d_upper = upload(context, upper_frame(true));
    const auto frame_identity = xmvb::cuda::OrbitalFrameIdentity::fresh();
    action.update_frame(d_upper.view(), frame_identity);
    require(action.version() != 0, "frame-only update did not invalidate CI model");
    const auto frame_version = action.version();
    action.update_frame(d_upper.view(), frame_identity);
    require(action.version() == frame_version, "unchanged frame identity invalidated the CI model");
    ci.refresh();
    action.diagonal_model(model_h.view(), model_m.view());
    require((download(context, model_m.view()) - old_m).norm() > 1e-3,
            "frame-only update left carrier overlap model stale");
    auto frame_only = ci.solve_device(old_roots, options);
    auto cpu_frame = source(count, true, false);
    const auto frame_matrices = cpu_frame.apply(Matrix::Identity(count, count));
    near(download(context, frame_only.overlap_view()),
         frame_matrices.overlap *
             download(context, frame_only.eigenpairs.eigenvector_matrix.view()),
         2e-9,
         "frame-only CI overlap image differs from CPU source");

    auto d_h = upload(context, one_electron(true));
    auto d_g = upload(context, pair_kernel(true));
    action.update_integrals(d_h.view(), d_g.view(), d_upper.view(), frame_identity);
    require(action.version() > frame_version,
            "integral refresh did not invalidate the CI model at an unchanged frame");
    action.diagonal_model(model_h.view(), model_m.view());
    require((download(context, model_h.view()) - old_h).norm() > 1e-3,
            "integral update left carrier Hamiltonian model stale");
    const auto device = ci.solve_device(frame_only.eigenpairs.eigenvector_matrix.view(), options);
    const Matrix root = download(context, device.eigenpairs.eigenvector_matrix.view());
    const Matrix h_root = download(context, device.hamiltonian_eigenvectors.view());
    const Matrix m_root = download(context, device.overlap_view());
    auto updated = source(count, true, true);
    const auto dense = updated.apply(Matrix::Identity(count, count));
    const Matrix h = dense.hamiltonian, m = dense.overlap;
    const auto gpu_dense = action.apply(Matrix::Identity(count, count));
    near(gpu_dense.hamiltonian,
         h,
         2e-9,
         "dynamic GPU H action differs from CPU orthogonal-frame oracle");
    near(gpu_dense.overlap, m, 2e-9, "dynamic GPU M action differs from CPU current frame");
    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> reference(h, m);
    require(reference.info() == Eigen::Success, "CPU CI dense oracle did not converge");
    if (std::abs(device.eigenpairs.eigenvalues.at(0) - reference.eigenvalues()[0]) >
        2e-8 * std::max(1.0, std::abs(reference.eigenvalues()[0])))
        std::cerr << "CI diagnostic full=" << full_space
                  << " device_energy=" << device.eigenpairs.eigenvalues.at(0)
                  << " cpu_energy=" << reference.eigenvalues()[0]
                  << " residual=" << device.stationarity_residual_norms.at(0) << '\n';
    near(device.eigenpairs.eigenvalues.at(0),
         reference.eigenvalues()[0],
         2e-8,
         "dynamic device CI Ritz energy differs from CPU exact H/M");
    near(h_root, h * root, 2e-9, "device CI retained H image differs");
    near(m_root, m * root, 2e-9, "device CI retained M image differs");
    near(
        (root.transpose() * m * root)(0, 0), 1.0, 2e-9, "device CI root lost metric normalization");

    DeviceMatrix energy(context, 1, 1), norm(context, 1, 1);
    ci.energy_norm_device(
        device.eigenpairs.eigenvector_matrix.view().column(0), energy.view(), norm.view());
    near(download(context, energy.view())(0, 0),
         device.eigenpairs.eigenvalues.at(0),
         2e-8,
         "device CI Rayleigh energy differs from Ritz value");
    near(download(context, norm.view())(0, 0),
         1.0,
         2e-9,
         "device CI metric norm differs from normalization");

    require(ci.has_exact_metric_inverse(),
            "small selected/FULL fixture lost its exact metric inverse");
    Matrix rhs = Matrix::Zero(count, 2);
    for (int j = 0; j < rhs.cols(); ++j)
        for (int i = 0; i < rhs.rows(); ++i)
            rhs(i, j) = 0.18 * std::sin(0.4 + (i + 1) * (j + 2));
    auto d_rhs = upload(context, rhs);
    DeviceMatrix inverse(context, count, 2), check(context, count, 2);
    ci.inverse_metric(d_rhs.view(), inverse.view());
    action.apply_overlap(inverse.view(), check.view());
    near(
        download(context, check.view()), rhs, 3e-9, "CI current-frame metric inverse is not exact");

    if (full_space) {
        xmvb::cuda::FullMetric metric(context, spin, action);
        metric.inverse_metric(d_rhs.view(), inverse.view());
        action.apply_overlap(inverse.view(), check.view());
        near(download(context, check.view()),
             rhs,
             3e-9,
             "FULL current-frame metric inverse is not exact");
    }
    if (!full_space) {
        const auto preparations = ci.prepared_points();
        ci.inverse_augmented_hessian_model(old_roots,
                                           first.eigenpairs.eigenvalues.front(),
                                           -0.01,
                                           frozen_after.view(),
                                           origin_point.get());
        near(download(context, frozen_after.view()),
             download(context, frozen_before.view()),
             2e-10,
             "accepted AH model changed with trial action cursor");
        require(ci.prepared_points() == preparations, "frozen AH inverse rebuilt trial model");
    }
    auto original_upper = upload(context, upper_frame(false));
    auto original_h = upload(context, one_electron(false));
    auto original_g = upload(context, pair_kernel(false));
    action.update_integrals(original_h.view(), original_g.view(), original_upper.view());
    const auto builds = ci.metric_builds(), preparations = ci.prepared_points();
    ci.bind_point(origin_point);
    auto restored = ci.solve_device(old_roots, options);
    near(restored.eigenpairs.eigenvalues.front(),
         first.eigenpairs.eigenvalues.front(),
         2e-8,
         "retained origin CI point did not restore its energy");
    require(ci.metric_builds() == builds && ci.prepared_points() == preparations,
            "binding a retained point rebuilt its selected metric");
}

} // namespace

int main() {
    try {
        check_case(false);
        check_case(true);
        std::cout << "CUDA dynamic structure CI refresh: PASS\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
