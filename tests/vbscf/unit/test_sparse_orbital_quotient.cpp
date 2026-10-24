#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>

#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/whitening.hpp"
#include "vbscf/orbitals/charts/raw.hpp"
#include "vbscf/orbitals/charts/physical_metric.hpp"
#include "vbscf/core/contracts/orbital_type.hpp"
#include "vbscf/optimization/preconditioners/transported_lbfgs.hpp"
#include "vbscf/optimization/preconditioners/hessian_diagonal.hpp"
#include <Eigen/QR>
#include "vbscf/optimization/trust_region/retraction.hpp"
#include "vbscf/optimization/trust_region/truncated_newton.hpp"
#include "vbscf/diagnostics/orbitals/chart_audit.hpp"
#include "vbscf/derivatives/hessian/exact/operator.hpp"

namespace {
using namespace xmvb::vb;

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}
constexpr double kFixtureForcing = 5.0e-2;

/** @brief Compare shared radial whitening with independent dense geometry. */
void check_whitened_tangent_inverse() {
    constexpr int n = 9, d = 5;
    Eigen::MatrixXd seed(n, d), mixing(n, n);
    Eigen::VectorXd raw(n);
    for (int i = 0; i < n; ++i) {
        raw[i] = std::cos(0.37 * (i + 1));
        for (int j = 0; j < d; ++j)
            seed(i, j) = std::sin(0.23 * (i + 1) * (j + 2));
        for (int j = 0; j < n; ++j)
            mixing(i, j) = std::cos(0.19 * (i + 2) * (j + 1));
    }
    const Eigen::HouseholderQR<Eigen::MatrixXd> qr(seed);
    const Eigen::MatrixXd q = qr.householderQ() * Eigen::MatrixXd::Identity(n, d);
    for (const double conditioning : {1.0, 1.0e-4}) {
        const Eigen::MatrixXd s =
            mixing.transpose() * mixing + conditioning * Eigen::MatrixXd::Identity(n, n);
        const TangentWhitening whitening(q, s);
        const RawTangentMetric raw_metric(q, s);
        for (const double scale : {0.2, 1.0, 7.0}) {
            const Eigen::VectorXd coefficients = scale * raw;
            const double norm = std::sqrt(coefficients.dot(s * coefficients));
            const Eigen::VectorXd c = coefficients / norm;
            const Eigen::MatrixXd jacobian =
                (Eigen::MatrixXd::Identity(n, n) - c * (c.transpose() * s)) / norm;
            const Eigen::MatrixXd physical = jacobian * q;
            const Eigen::MatrixXd g = physical.transpose() * s * physical;
            const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(g);
            const Eigen::MatrixXd inverse_root =
                spectrum.eigenvectors() *
                spectrum.eigenvalues().cwiseSqrt().cwiseInverse().asDiagonal() *
                spectrum.eigenvectors().transpose();
            for (const bool sphere : {false, true}) {
                const WhitenedTangent result = whitening.apply(coefficients, s, sphere);
                const RawTangent raw_result = raw_metric.apply(q, coefficients, s, sphere);
                const Eigen::MatrixXd conversion =
                    raw_result.inverse_gram * raw_result.basis.transpose() * s * result.basis;
                const Eigen::MatrixXd raw_coordinate_metric =
                    q.transpose() * s * q -
                    raw_result.projection * raw_result.projection.transpose();
                const Eigen::MatrixXd raw_coordinate_inverse =
                    raw_metric.inverse() + raw_result.inverse_projection *
                                               raw_result.inverse_projection.transpose() /
                                               raw_result.radial_gap;
                require((raw_result.basis * conversion - result.basis).norm() <
                            2e-9 * result.basis.norm(),
                        "unwhitened chart changed the horizontal representative");
                require((conversion.transpose() * raw_coordinate_metric * conversion -
                         Eigen::MatrixXd::Identity(d, d))
                                .norm() < 2e-8,
                        "raw coordinate metric is inconsistent with whitening");
                require((raw_coordinate_metric * raw_coordinate_inverse -
                         Eigen::MatrixXd::Identity(d, d))
                                .norm() < 2e-8,
                        "raw coordinate inverse failed its residual");
                const Eigen::MatrixXd raw_gram =
                    raw_result.basis.transpose() * s * raw_result.basis;
                require(
                    (raw_gram * raw_result.inverse_gram - Eigen::MatrixXd::Identity(d, d)).norm() <
                        2e-8,
                    "unwhitened representative has an incorrect left inverse");
                const Eigen::VectorXd raw_covector = Eigen::VectorXd::LinSpaced(d, -.4, .7);
                require((raw_metric.reference_covector(raw_covector, raw_result.projection) -
                         conversion.transpose() * raw_covector)
                                .norm() < 2e-8,
                        "implicit reference covector differs from an explicitly whitened chart");
                const Eigen::MatrixXd expected =
                    sphere ? (norm * physical * inverse_root).eval() : (q * inverse_root).eval();
                const Eigen::MatrixXd rotation = expected.colPivHouseholderQr().solve(result.basis);
                require(
                    (expected * rotation - result.basis).norm() < 2e-10 * expected.norm() &&
                        (rotation.transpose() * rotation - Eigen::MatrixXd::Identity(d, d)).norm() <
                            2e-9,
                    "shared whitening changed the physical horizontal lift");
                const Eigen::MatrixXd gram = result.basis.transpose() * s * result.basis;
                const Eigen::MatrixXd inverse = gram.ldlt().solve(Eigen::MatrixXd::Identity(d, d));
                require((result.inverse_gram - inverse).norm() < 2e-10 * inverse.norm(),
                        "raw AO-metric inverse differs from a dense reference");
                require((gram * result.inverse_gram - Eigen::MatrixXd::Identity(d, d)).norm() <
                            2e-9,
                        "raw tangent left inverse failed its dense residual");
                const Eigen::MatrixXd normalized_tangent = jacobian * result.basis;
                require((normalized_tangent.transpose() * s * normalized_tangent -
                         Eigen::MatrixXd::Identity(d, d))
                                .norm() < 2e-9,
                        "optimized tangent changed the normalized physical metric");
            }
        }
    }
    // A conservative congruence bound is insufficient here, but the actual
    // updated metric is well conditioned. Test its secular rank certificate.
    Eigen::Matrix3d s = Eigen::Matrix3d::Identity();
    s(0, 0) = 1e8;
    const Eigen::Matrix<double, 3, 2> q2 = Eigen::Matrix3d::Identity().leftCols<2>();
    const TangentWhitening whitening(q2, s);
    const Eigen::Vector3d regular(std::sqrt(1.0 - 1e-8) / 1e4, 0.0, 1e-4);
    const auto recovered = whitening.apply(regular, s, false);
    require(recovered.basis.allFinite() && recovered.inverse_gram.allFinite(),
            "secular rank certificate rejected a regular radial downdate");
    bool rejected = false;
    try {
        whitening.apply(Eigen::Vector3d(1e-4, 0.0, 1e-12), s, false);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "shared radial whitening admitted a rank-deficient tangent");
    std::cout << "shared radial whitening and raw Gram inverse: passed\n";
}

void check_response_spectrum_summary() {
    StructureResponseSchurModel model;
    model.orbital_couplings = Eigen::Matrix3d::Identity();
    model.projected_inverse = (Eigen::Vector3d() << 4.0, -1.0, 0.25).finished().asDiagonal();
    const ResponseSpectrumSummary reference = summarize_response_spectrum(model);
    require(reference.model_rank == 3 && reference.rank_90 == 2 && reference.rank_99 == 3,
            "response spectrum summary returned incorrect capture ranks");
    require(std::abs(reference.top_mode_fraction - 4.0 / 5.25) < 1.0e-14 &&
                std::abs(reference.effective_rank - (5.25 * 5.25) / (16.0 + 1.0 + 0.0625)) <
                    1.0e-14,
            "response spectrum summary returned incorrect weights");

    Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(0.37, Eigen::Vector3d(1.0, 2.0, -1.0).normalized()).toRotationMatrix();
    model.orbital_couplings = rotation;
    model.projected_inverse = rotation.transpose() *
                              (Eigen::Vector3d() << 4.0, -1.0, 0.25).finished().asDiagonal() *
                              rotation;
    const ResponseSpectrumSummary rotated = summarize_response_spectrum(model);
    require(rotated.rank_90 == reference.rank_90 && rotated.rank_99 == reference.rank_99 &&
                std::abs(rotated.effective_rank - reference.effective_rank) < 1.0e-13,
            "response spectrum summary depends on response-basis rotation");
}

void check_accuracy_aware_forcing() {
    constexpr double initial_norm = 4.0;
    constexpr double gradient_tolerance = 1.0e-3;
    const double accuracy_floor = std::sqrt(gradient_tolerance / initial_norm);
    require(inexact_newton_forcing_term(
                initial_norm, initial_norm, initial_norm, gradient_tolerance) == 0.5,
            "accuracy-aware forcing lost its contraction limit");
    require(std::abs(inexact_newton_forcing_term(1.0, 1.0, initial_norm, gradient_tolerance) -
                     0.25) < 1.0e-15,
            "accuracy-aware forcing does not track gradient progress");
    require(std::abs(inexact_newton_forcing_term(1.0e-2, 1.0e-2, initial_norm, gradient_tolerance) -
                     accuracy_floor) < 1.0e-15,
            "accuracy-aware forcing did not stop at the outer accuracy floor");
    require(std::abs(inexact_newton_forcing_term(1.0e-5, 1.0e-5, initial_norm, gradient_tolerance) -
                     accuracy_floor) < 1.0e-15,
            "outer target over-solved beyond the requested accuracy");
    require(std::abs(inexact_newton_forcing_term(6.0e-3, 1.7e-3, 6.0e-3, gradient_tolerance) -
                     (gradient_tolerance / 6.0e-3)) < 1.0e-15,
            "one-contraction outer target was not enforced");
}

OrbitalPreparationInput make_input(const Eigen::MatrixXd& coefficients,
                                   const std::vector<std::vector<int>>& supports,
                                   int inactive_count) {
    OrbitalPreparationInput input;
    input.n_basis_functions = coefficients.rows();
    input.n_orbitals = coefficients.cols();
    input.n_active_orbitals = coefficients.cols() - inactive_count;
    input.n_active_electrons = input.n_active_orbitals;
    input.n_total_electrons = 2 * inactive_count + input.n_active_electrons;
    input.orbital_type = 1;
    input.ao_overlap_matrix = Eigen::MatrixXd::Identity(coefficients.rows(), coefficients.rows());
    // Nontrivial SPD AO metric; the admissible gauge span is algebraic, while
    // its horizontal complement uses this metric.
    input.ao_overlap_matrix.array() += 0.07;
    input.orbital_value_table.assign(coefficients.size(), 0.0);
    input.orbital_basis_index_table.assign(coefficients.size(), 0);
    for (int p = 0; p < coefficients.cols(); ++p) {
        input.orbital_basis_counts.push_back(supports[p].size());
        for (int j = 0; j < static_cast<int>(supports[p].size()); ++j) {
            const int slot = p * coefficients.rows() + j;
            input.orbital_basis_index_table[slot] = supports[p][j] + 1;
            input.orbital_value_table[slot] = coefficients(supports[p][j], p);
        }
    }
    return input;
}

Eigen::MatrixXd dense(const OrbitalPreparationInput& input) {
    Eigen::MatrixXd c = Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_orbitals);
    for (int p = 0; p < c.cols(); ++p) {
        for (int j = 0; j < stored_sparse_orbital_coefficient_count(input, p); ++j) {
            const int slot = p * c.rows() + j;
            c(input.orbital_basis_index_table[slot] - 1, p) = input.orbital_value_table[slot];
        }
    }
    return c;
}

Eigen::MatrixXd packed_ao_metric(const OrbitalPreparationInput& input,
                                 const SparseParameterLayout& view) {
    Eigen::MatrixXd metric = Eigen::MatrixXd::Zero(view.size(), view.size());
    for (int orbital = 0; orbital < input.n_orbitals; ++orbital) {
        const int count = view.orbital_coefficient_count(orbital);
        for (int row = 0; row < count; ++row) {
            const int ao_row =
                input.orbital_basis_index_table[orbital * input.n_basis_functions + row] - 1;
            for (int column = 0; column < count; ++column) {
                const int ao_column =
                    input.orbital_basis_index_table[orbital * input.n_basis_functions + column] - 1;
                metric(view.packed_index(orbital, row), view.packed_index(orbital, column)) =
                    input.ao_overlap_matrix(ao_row, ao_column);
            }
        }
    }
    return metric;
}

/** @brief Curvature surrogates must not change quotient maps or stationarity. */
void check_preconditioner_geometry(const OrbitalPreparationInput& input,
                                   bool complete_active_space = false) {
    const SparseParameterLayout layout(input);
    const Eigen::MatrixXd orbitals = dense(input);
    const Eigen::MatrixXd f =
        Eigen::VectorXd::LinSpaced(input.n_basis_functions, -2., 3.).asDiagonal();
    const OrbitalChart geometry(
        input, layout, orbitals, orbitals, nullptr, false, complete_active_space);
    const OrbitalChart preconditioned(
        input, layout, orbitals, orbitals, &f, false, complete_active_space);
    require(geometry.reduced_size() == preconditioned.reduced_size() &&
                geometry.rank_signature() == preconditioned.rank_signature(),
            "one-electron preconditioner changed quotient dimensions");
    const Eigen::VectorXd gradient = Eigen::VectorXd::LinSpaced(layout.pack(input).size(), -.4, .7);
    require((geometry.project_reduced_gradient(gradient) -
             preconditioned.project_reduced_gradient(gradient))
                    .norm() < 1.e-12,
            "one-electron preconditioner changed gradient coordinates");
    const Eigen::VectorXd step = Eigen::VectorXd::LinSpaced(geometry.reduced_size(), -.2, .3);
    require((geometry.expand_step(step) - preconditioned.expand_step(step)).norm() < 1.e-12,
            "one-electron preconditioner changed the physical orbital lift");
    const NonredundantRetractionMetric metric(geometry, layout, input);
    const NonredundantRetractionMetric preconditioned_metric(preconditioned, layout, input);
    require((metric.apply(step) - preconditioned_metric.apply(step)).norm() < 1.e-12,
            "one-electron preconditioner changed the trust-region metric");
}

// Finite physical map, independent of both analytic gauge constructions.
// Ray projectors remove scale/sign without differentiating a normalization.
Eigen::VectorXd physical_map(const OrbitalPreparationInput& input) {
    const Eigen::MatrixXd c = dense(input);
    const int ni = (input.n_total_electrons - input.n_active_electrons) / 2;
    const int n = c.rows();
    Eigen::MatrixXd p = Eigen::MatrixXd::Zero(n, n);
    if (ni > 0) {
        const Eigen::MatrixXd metric =
            c.leftCols(ni).transpose() * input.ao_overlap_matrix * c.leftCols(ni);
        p = c.leftCols(ni) * metric.ldlt().solve(c.leftCols(ni).transpose());
    }
    Eigen::VectorXd result(n * n * (1 + input.n_active_orbitals));
    result.head(n * n) = Eigen::Map<const Eigen::VectorXd>(p.data(), n * n);
    for (int a = 0; a < input.n_active_orbitals; ++a) {
        const Eigen::VectorXd b = c.col(ni + a) - p * input.ao_overlap_matrix * c.col(ni + a);
        require(b.norm() > 1e-12, "degenerate synthetic active ray");
        const Eigen::MatrixXd ray = b * b.transpose() / b.squaredNorm();
        result.segment((a + 1) * n * n, n * n) =
            Eigen::Map<const Eigen::VectorXd>(ray.data(), n * n);
    }
    return result;
}

void check_finite_step_gradient_pullback(const OrbitalPreparationInput& input) {
    const SparseParameterLayout view(input);
    const Eigen::MatrixXd orbitals = dense(input);
    const OrbitalChart source(input, view, orbitals, orbitals);
    require(source.reduced_size() > 0, "keyframe pullback fixture has no tangent space");
    const Eigen::VectorXd step =
        0.15 * Eigen::VectorXd::LinSpaced(source.reduced_size(), -0.4, 0.7);
    const auto trial = source.retract_step(input, step);
    const auto energy = [](const OrbitalPreparationInput& point) {
        const Eigen::VectorXd features = physical_map(point);
        return features.dot(Eigen::VectorXd::LinSpaced(features.size(), -0.4, 0.8));
    };
    // Independently differentiate the finite normalized/projected physical map
    // in raw trial coordinates; its gradient already includes these pullbacks.
    constexpr double h = 1e-6;
    const Eigen::VectorXd parameters = view.pack(trial);
    Eigen::VectorXd raw_gradient(parameters.size());
    for (Eigen::Index i = 0; i < parameters.size(); ++i) {
        auto plus = trial, minus = trial;
        const Eigen::VectorXd delta = h * Eigen::VectorXd::Unit(parameters.size(), i);
        view.unpack(parameters + delta, &plus);
        view.unpack(parameters - delta, &minus);
        raw_gradient[i] = (energy(plus) - energy(minus)) / (2 * h);
    }
    const Eigen::VectorXd pulled = source.project_reduced_gradient(raw_gradient);
    const Eigen::VectorXd direction =
        Eigen::VectorXd::LinSpaced(source.reduced_size(), 0.3, -0.8).normalized();
    const double finite = (energy(source.retract_step(input, step + h * direction)) -
                           energy(source.retract_step(input, step - h * direction))) /
                          (2 * h);
    require(std::abs(pulled.dot(direction) - finite) < 1e-7 * std::max(1.0, std::abs(finite)),
            "source-chart keyframe gradient differs from finite retraction derivative");
}

double finite_physical_metric_squared_norm(const OrbitalPreparationInput& input,
                                           const SparseParameterLayout& view,
                                           const Eigen::VectorXd& direction) {
    constexpr double step = 1.0e-5;
    auto plus = input;
    auto minus = input;
    view.unpack(view.pack(input) + step * direction, &plus);
    view.unpack(view.pack(input) - step * direction, &minus);

    const Eigen::MatrixXd current = dense(input);
    const Eigen::MatrixXd displaced_plus = dense(plus);
    const Eigen::MatrixXd displaced_minus = dense(minus);
    const int n_inactive = (input.n_total_electrons - input.n_active_electrons) / 2;
    const int n_active = input.n_active_orbitals;
    const int n_bf = input.n_basis_functions;
    const Eigen::MatrixXd& overlap = input.ao_overlap_matrix;
    Eigen::MatrixXd complement = Eigen::MatrixXd::Identity(n_bf, n_bf);
    Eigen::MatrixXd inverse = Eigen::MatrixXd::Zero(n_inactive, n_inactive);
    if (n_inactive > 0) {
        const auto inactive = current.leftCols(n_inactive);
        inverse = (inactive.transpose() * overlap * inactive)
                      .ldlt()
                      .solve(Eigen::MatrixXd::Identity(n_inactive, n_inactive));
        complement.noalias() -= inactive * inverse * inactive.transpose() * overlap;
    }
    auto projected_active = [&](const Eigen::MatrixXd& orbitals) {
        Eigen::MatrixXd projector = Eigen::MatrixXd::Identity(n_bf, n_bf);
        if (n_inactive > 0) {
            const auto inactive = orbitals.leftCols(n_inactive);
            const Eigen::MatrixXd local_inverse =
                (inactive.transpose() * overlap * inactive)
                    .ldlt()
                    .solve(Eigen::MatrixXd::Identity(n_inactive, n_inactive));
            projector.noalias() -= inactive * local_inverse * inactive.transpose() * overlap;
        }
        return (projector * orbitals.middleCols(n_inactive, n_active)).eval();
    };

    double squared_norm = 0.0;
    if (n_inactive > 0) {
        const Eigen::MatrixXd delta_inactive =
            (displaced_plus.leftCols(n_inactive) - displaced_minus.leftCols(n_inactive)) /
            (2.0 * step);
        const Eigen::MatrixXd horizontal = complement * delta_inactive;
        squared_norm += (inverse * horizontal.transpose() * overlap * horizontal).trace();
    }
    const Eigen::MatrixXd base_active = projected_active(current);
    const Eigen::MatrixXd delta_active =
        (projected_active(displaced_plus) - projected_active(displaced_minus)) / (2.0 * step);
    for (int active = 0; active < n_active; ++active) {
        const Eigen::VectorXd ray = base_active.col(active);
        const double norm_squared = ray.dot(overlap * ray);
        Eigen::VectorXd horizontal = delta_active.col(active);
        horizontal.noalias() -= ray * (ray.dot(overlap * horizontal) / norm_squared);
        squared_norm += horizontal.dot(overlap * horizontal) / norm_squared;
    }
    return squared_norm;
}

void check(const std::string& name, const OrbitalPreparationInput& input, int expected_dimension) {
    const SparseParameterLayout view(input);
    const Eigen::MatrixXd c = dense(input);
    OrbitalChart space(input, view, c, c, nullptr, true);
    require(space.reduced_size() == expected_dimension,
            name + ": expected dimension " + std::to_string(expected_dimension) + ", got " +
                std::to_string(space.reduced_size()));
    const auto chart_diagnostics = space.structural_diagnostics();
    require(chart_diagnostics.maximum_sphere_tangency_residual < 1.0e-12,
            name + ": product-sphere representative is not S-tangent");
    Eigen::MatrixXd u(view.size(), space.reduced_size());
    for (int j = 0; j < u.cols(); ++j) {
        u.col(j) = space.expand_step(Eigen::VectorXd::Unit(u.cols(), j));
    }
    const auto audit = audit_orbital_chart(input, view, &u);
    require(audit.gauge_rank == audit.physical_jacobian_nullity,
            name + ": independent kernel mismatch");
    require(audit.current_retained_gauge_dimension == 0 &&
                audit.current_missing_physical_dimension == 0,
            name + ": incorrect physical image");
    // The admissible gauge is unchanged; each raw lift must use its AO-metric
    // horizontal complement, including restricted and frozen sparse supports.
    const Eigen::MatrixXd sparse_metric = packed_ao_metric(input, view);
    const Eigen::MatrixXd gauge_horizontal =
        audit.packed_gauge_basis.transpose() * sparse_metric * u;
    require(gauge_horizontal.norm() < 5.0e-10 * std::max(1.0, (sparse_metric * u).norm()),
            name + ": raw quotient lift is not S-horizontal to admissible gauge");
    require(audit.quotient_dimension == space.reduced_size(),
            name + ": AO-metric horizontal lift lost a physical dimension");
    const OrbitalPhysicalMetric coupled_metric(input, space.uses_active_subspace_gauge());
    for (int gauge = 0; gauge < audit.packed_gauge_basis.cols(); ++gauge) {
        require(space.project_vector(audit.packed_gauge_basis.col(gauge)).reduced_gradient.norm() <
                    1.0e-10,
                name + ": quotient vector projection retained a gauge direction");
        require(coupled_metric.squared_norm(view, audit.packed_gauge_basis.col(gauge)) < 1.0e-18,
                name + ": coupled metric did not annihilate gauge");
        require(coupled_metric.apply(view, audit.packed_gauge_basis.col(gauge)).norm() < 1.0e-10,
                name + ": matrix-free metric action retained gauge");
    }
    if (u.cols() > 0) {
        const Eigen::VectorXd direction =
            u * Eigen::VectorXd::LinSpaced(u.cols(), -0.7, 0.9).normalized();
        const double analytic = coupled_metric.squared_norm(view, direction);
        const double finite = finite_physical_metric_squared_norm(input, view, direction);
        require(std::abs(analytic - finite) < 1.0e-7 * std::max(1.0, finite),
                name + ": coupled metric differs from finite physical map");
        const Eigen::VectorXd action = coupled_metric.apply(view, direction);
        require(std::abs(direction.dot(action) - analytic) < 1.0e-10 * std::max(1.0, analytic),
                name + ": matrix-free metric action is not the feature adjoint");
    }
    for (int column = 0; column < u.cols(); ++column) {
        const Eigen::VectorXd recovered = space.project_vector(u.col(column)).reduced_gradient;
        require((recovered - Eigen::VectorXd::Unit(u.cols(), column)).norm() < 1e-10,
                name + ": reduced vector coordinates are not invertible");
    }
    Eigen::MatrixXd physical_actions(view.size(), u.cols());
    for (int column = 0; column < u.cols(); ++column) {
        physical_actions.col(column) = coupled_metric.apply(view, u.col(column));
    }
    const Eigen::MatrixXd physical_gram = u.transpose() * physical_actions;
    require((physical_gram - physical_gram.transpose()).norm() < 1.0e-12 &&
                (physical_gram.diagonal().array() > 0.0).all(),
            name + ": coupled quotient physical metric is not positive");
    if (u.cols() > 0) {
        const Eigen::VectorXd first = u.col(0);
        const Eigen::VectorXd last = u.col(u.cols() - 1);
        const double polarization = 0.5 * (coupled_metric.squared_norm(view, first + last) -
                                           coupled_metric.squared_norm(view, first) -
                                           coupled_metric.squared_norm(view, last));
        require(std::abs(polarization - physical_gram(0, u.cols() - 1)) < 1.0e-10,
                name + ": matrix-free metric action disagrees with polarization");
        if (u.cols() <= 14) {
            const NonredundantRetractionMetric metric(space, view, input);
            const Eigen::VectorXd covector = Eigen::VectorXd::LinSpaced(u.cols(), -0.8, 1.3);
            const Eigen::VectorXd riesz_vector = metric.solve(covector);
            const Eigen::VectorXd dense_riesz_vector = physical_gram.ldlt().solve(covector);
            require((riesz_vector - dense_riesz_vector).norm() <
                        5.0e-8 * std::max(1.0, dense_riesz_vector.norm()),
                    name + ": matrix-free Riesz solve differs from dense metric");
            require((metric.apply(riesz_vector) - covector).norm() < 2.0e-8 * covector.norm(),
                    name + ": matrix-free Riesz solve has a large true residual");
            require(
                std::abs(metric.dual_norm(covector) - std::sqrt(covector.dot(dense_riesz_vector))) <
                    5.0e-8 * std::max(1.0, dense_riesz_vector.norm()),
                name + ": physical dual norm differs from the dense metric");
            const Eigen::VectorXd test_vector = Eigen::VectorXd::LinSpaced(u.cols(), 0.9, -0.4);
            require((metric.solve(metric.apply(test_vector)) - test_vector).norm() <
                        5.0e-8 * std::max(1.0, test_vector.norm()),
                    name + ": Riesz solve does not recover reduced coordinates");
            require(
                std::abs(riesz_vector.dot(metric.apply(test_vector)) - covector.dot(test_vector)) <
                    2.0e-8 * std::max(1.0, std::abs(covector.dot(test_vector))),
                name + ": Riesz solve violates metric-covector duality");

            const Eigen::MatrixXd model_hessian =
                physical_gram + Eigen::MatrixXd::Identity(u.cols(), u.cols());
            const Eigen::VectorXd gradient = Eigen::VectorXd::LinSpaced(u.cols(), 0.2, 1.1);
            TruncatedNewtonSubspace subspace;
            subspace.orthonormal_basis = Eigen::MatrixXd::Identity(u.cols(), u.cols());
            subspace.tangent_basis = subspace.orthonormal_basis;
            subspace.hessian_basis = model_hessian;
            subspace.metric_basis = physical_gram;
            subspace.reduced_hessian = model_hessian;
            subspace.reduced_metric = physical_gram;
            subspace.projected_gradient = gradient;
            OrbitalChart::ProjectionResult projection;
            projection.reduced_gradient = gradient;
            const Eigen::VectorXd exact_step = -model_hessian.ldlt().solve(gradient);
            const double exact_norm = metric.norm(exact_step);
            for (double radius : {2.0 * exact_norm, 0.5 * exact_norm}) {
                const auto step = solve_trust_region_in_subspace(
                    projection, radius, metric, subspace, kFixtureForcing);
                require(step.predicted_decrease > 0.0 &&
                            metric.norm(step.reduced_step) <= radius * (1.0 + 1.0e-7),
                        name + ": generalized trust-region radius mismatch");
                require((gradient + step.reduced_hessian_times_step +
                         step.trust_region_shift * step.reduced_metric_times_step)
                                .norm() < 1.0e-7 * gradient.norm(),
                        name + ": generalized trust-region KKT mismatch");
                if (radius > exact_norm) {
                    require((step.reduced_step - exact_step).norm() < 1.0e-8,
                            name + ": generalized interior Newton step mismatch");
                }
            }

            const double affine_radius = 2.0 * exact_norm;
            const Eigen::VectorXd baseline_step = 0.35 * exact_step;
            const auto affine_step =
                solve_affine_trust_region_in_subspace(projection,
                                                      affine_radius,
                                                      metric,
                                                      subspace,
                                                      baseline_step,
                                                      model_hessian * baseline_step,
                                                      physical_gram * baseline_step,
                                                      kFixtureForcing);
            require(affine_step.predicted_decrease > 0.0 &&
                        (affine_step.reduced_step - exact_step).norm() < 1.0e-8,
                    name + ": affine correction did not recover the Newton step");
            require((gradient + affine_step.reduced_hessian_times_step +
                     affine_step.trust_region_shift * affine_step.reduced_metric_times_step)
                            .norm() < 1.0e-7 * gradient.norm(),
                    name + ": affine correction KKT mismatch");

            const auto zero_defect_step =
                solve_affine_trust_region_in_subspace(projection,
                                                      affine_radius,
                                                      metric,
                                                      subspace,
                                                      exact_step,
                                                      model_hessian * exact_step,
                                                      physical_gram * exact_step,
                                                      kFixtureForcing);
            require(zero_defect_step.predicted_decrease > 0.0 &&
                        (zero_defect_step.reduced_step - exact_step).norm() < 1.0e-8,
                    name + ": affine solver changed an exact baseline step");
        }
    }
    const Eigen::VectorXd x = view.pack(input);
    Eigen::MatrixXd fd(physical_map(input).size(), view.size());
    const double h = 1e-4;
    for (int j = 0; j < view.size(); ++j) {
        auto plus = input, minus = input;
        view.unpack(x + h * Eigen::VectorXd::Unit(x.size(), j), &plus);
        view.unpack(x - h * Eigen::VectorXd::Unit(x.size(), j), &minus);
        fd.col(j) = (physical_map(plus) - physical_map(minus)) / (2 * h);
    }
    require((fd * audit.packed_gauge_basis).norm() / std::max(1.0, fd.norm()) < 1e-7,
            name + ": finite physical-map gauge derivative");
    if (u.cols() > 0) {
        const Eigen::MatrixXd image = fd * u;
        const Eigen::JacobiSVD<Eigen::MatrixXd> svd(image);
        require(svd.singularValues().tail(1)[0] > 1e-7, name + ": finite physical-map rank loss");
        const Eigen::VectorXd d = Eigen::VectorXd::LinSpaced(u.cols(), -0.7, 0.9);
        const Eigen::VectorXd g = Eigen::VectorXd::LinSpaced(view.size(), 0.2, 1.1);
        require(std::abs(g.dot(space.expand_step(d)) - space.project_reduced_gradient(g).dot(d)) <
                    1e-12,
                name + ": adjoint projection");
        const Eigen::VectorXd reduced_covector = Eigen::VectorXd::LinSpaced(u.cols(), -0.4, 0.6);
        require((space.project_reduced_gradient(space.expand_gradient(reduced_covector)) -
                 reduced_covector)
                        .norm() < 1e-10,
                name + ": reduced covector lift is not a right inverse");
        require((audit.packed_gauge_basis.transpose() * space.expand_gradient(reduced_covector))
                        .norm() < 1.0e-10,
                name + ": lifted covector does not annihilate admissible gauge");
        Eigen::VectorXd secant_step = Eigen::VectorXd::LinSpaced(u.cols(), 0.3, 1.2).normalized();
        Eigen::VectorXd secant_gradient_change = (2.0 * secant_step).eval();
        std::vector<PackedSecantPair> secant_history;
        append_projected_secant_pair(space,
                                     space.expand_step(secant_step),
                                     space.expand_gradient(secant_gradient_change),
                                     1,
                                     &secant_history);
        auto inverse_hessian = build_transported_reduced_lbfgs_preconditioner(
            space, secant_history, 1, LbfgsInitialInverse::ScaledIdentity);
        require(inverse_hessian.size() == 1, name + ": valid quotient secant was rejected");
        require((inverse_hessian.apply(secant_gradient_change) - secant_step).norm() < 1.0e-10,
                name + ": quotient L-BFGS inverse violates its secant equation");
        const auto trial = space.retract_step(input, d, 0.01);
        require((view.pack(trial) - x - 0.01 * u * d).norm() < 1e-12,
                name + ": additive retraction");
        const auto opposite = space.retract_step(input, -d, 0.01);
        require((view.pack(trial) + view.pack(opposite) - 2.0 * x).norm() < 1e-12 &&
                    trial.orbital_basis_index_table == input.orbital_basis_index_table &&
                    opposite.orbital_basis_index_table == input.orbital_basis_index_table,
                name + ": opposite raw lifts changed the sparse graph");
        const Eigen::MatrixXd original_orbitals = dense(input);
        const Eigen::MatrixXd trial_orbitals = dense(trial);
        const Eigen::MatrixXd opposite_orbitals = dense(opposite);
        const int n_inactive = (input.n_total_electrons - input.n_active_electrons) / 2;
        for (int orbital = 0; orbital < n_inactive; ++orbital) {
            const Eigen::VectorXd metric_origin =
                input.ao_overlap_matrix * original_orbitals.col(orbital);
            require(metric_origin.dot(trial_orbitals.col(orbital)) > 0.0 &&
                        metric_origin.dot(opposite_orbitals.col(orbital)) > 0.0,
                    name + ": fixed-core chart flipped an inactive frame column");
        }
        const Eigen::VectorXd raw_coupling = Eigen::VectorXd::LinSpaced(view.size(), -0.6, 0.8);
        const Eigen::MatrixXd raw_hessian =
            sparse_metric + 0.03 * raw_coupling * raw_coupling.transpose();
        const Eigen::VectorXd raw_linear = Eigen::VectorXd::LinSpaced(view.size(), 0.4, -0.3);
        const auto pulled_gradient = [&](const OrbitalPreparationInput& point) {
            const Eigen::VectorXd packed = view.pack(point);
            return space.project_reduced_gradient(raw_hessian * packed + raw_linear);
        };
        constexpr double derivative_step = 1.0e-5;
        const Eigen::VectorXd finite_hvp =
            (pulled_gradient(space.retract_step(input, derivative_step * d)) -
             pulled_gradient(space.retract_step(input, -derivative_step * d))) /
            (2.0 * derivative_step);
        const Eigen::VectorXd analytic_hvp = space.project_reduced_gradient(raw_hessian * u * d);
        require((finite_hvp - analytic_hvp).norm() < 2.0e-9 * std::max(1.0, analytic_hvp.norm()),
                name + ": reduced HVP lost the raw retraction pullback");
        // Stored but frozen slots must not change.
        std::vector<bool> differentiable(input.orbital_value_table.size(), false);
        for (int slot : view.differentiable_parameter_indices())
            differentiable[slot] = true;
        for (int slot = 0; slot < static_cast<int>(differentiable.size()); ++slot) {
            require(differentiable[slot] ||
                        trial.orbital_value_table[slot] == input.orbital_value_table[slot],
                    name + ": modified fixed coefficient");
        }
    }
    std::cout << name << ": passed (" << view.size() << " -> " << space.reduced_size() << ")\n";
}

void check_complete_oeo_active_subspace_gauge() {
    Eigen::MatrixXd coefficients = Eigen::MatrixXd::Zero(5, 3);
    coefficients(0, 0) = 1.0;
    coefficients(1, 1) = 1.0;
    coefficients(2, 2) = 1.0;
    coefficients(3, 1) = 0.2;
    coefficients(4, 2) = -0.3;
    OrbitalPreparationInput input =
        make_input(coefficients, {{0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}}, 1);
    input.orbital_type = kOrbitalTypeOeo;
    const SparseParameterLayout view(input);
    const Eigen::MatrixXd orbitals = dense(input);

    const OrbitalChart structure_subset_chart(
        input, view, orbitals, orbitals, nullptr, false, false);
    require(structure_subset_chart.reduced_size() == 10,
            "OEO structure subset did not retain independent active rays");
    const auto subset_diagnostics = structure_subset_chart.structural_diagnostics();
    require(subset_diagnostics.product_sphere_orbital_count == 3 &&
                subset_diagnostics.maximum_sphere_tangency_residual < 1.0e-12,
            "OEO structure subset did not use S-tangent orbital rays");

    const OrbitalChart complete_cas_chart(input, view, orbitals, orbitals, nullptr, false, true);
    require(complete_cas_chart.reduced_size() == 8,
            "complete-CAS OEO chart did not remove active--active gauge");
    const auto complete_diagnostics = complete_cas_chart.structural_diagnostics();
    require(complete_diagnostics.product_sphere_orbital_count == 0,
            "complete-CAS OEO chart was split into independent orbital rays");

    for (const auto coordinates :
         {OrbitalChart::Coordinates::Whitened, OrbitalChart::Coordinates::Unwhitened}) {
        const OrbitalChart chart(
            input, view, orbitals, orbitals, nullptr, false, true, nullptr, coordinates);
        Eigen::VectorXd raw =
            chart.expand_step(Eigen::VectorXd::LinSpaced(chart.reduced_size(), -0.9, 0.7));
        for (int slot = 0; slot < 5; ++slot)
            raw[view.packed_index(0, slot)] = 0.0;
        const Eigen::VectorXd step = chart.project_vector(raw).reduced_gradient;
        const Eigen::MatrixXd core = orbitals.leftCols(1);
        const Eigen::MatrixXd active =
            orbitals.rightCols(2) -
            core * (core.transpose() * input.ao_overlap_matrix * core)
                       .ldlt()
                       .solve(core.transpose() * input.ao_overlap_matrix * orbitals.rightCols(2));
        const Eigen::MatrixXd origin_gram = active.transpose() * input.ao_overlap_matrix * active;
        for (const double scale : {1.0, 4.0, 20.0}) {
            const Eigen::MatrixXd moved = dense(chart.retract_step(input, step, scale));
            require((moved.leftCols(1) - core).norm() < 1.0e-12,
                    "fixed-core OEO test moved the core");
            const Eigen::MatrixXd cross =
                active.transpose() * input.ao_overlap_matrix * moved.rightCols(2);
            require((cross - origin_gram).norm() < 2.0e-11,
                    "S-horizontal OEO finite step changed its active graph orientation");
            require(cross.determinant() > 0,
                    "fixed-core OEO finite step reversed the active frame");
        }
    }

    Eigen::VectorXd active_mixing = Eigen::VectorXd::Zero(view.size());
    for (int slot = 0; slot < 5; ++slot) {
        active_mixing[view.packed_index(1, slot)] = coefficients(slot, 2);
    }
    require(complete_cas_chart.project_vector(active_mixing).reduced_gradient.norm() < 1.0e-12,
            "complete-CAS OEO chart retained an active--active gauge direction");
    const OrbitalPhysicalMetric metric(input, true);
    const OrbitalPhysicalMetric subset_metric(input, false);
    require(metric.squared_norm(view, active_mixing) < 1.0e-22 &&
                metric.apply(view, active_mixing).norm() < 1.0e-12,
            "complete-CAS metric retained active GL gauge");
    require(subset_metric.squared_norm(view, active_mixing) > 0.1,
            "structure-subset metric incorrectly removed physical active mixing");

    const Eigen::VectorXd direction = Eigen::VectorXd::LinSpaced(view.size(), -0.3, 0.7);
    const Eigen::VectorXd other = Eigen::VectorXd::LinSpaced(view.size(), 0.8, -0.2);
    const double reference = metric.squared_norm(view, direction);
    const auto action = metric.apply(view, direction);
    require(std::abs(direction.dot(action) - reference) < 1.0e-12 &&
                std::abs(other.dot(action) - direction.dot(metric.apply(view, other))) < 1.0e-12,
            "complete-CAS metric action is not the feature adjoint");

    // Independent finite differences of the two AO-whitened span projectors.
    const auto projectors = [&](const Eigen::VectorXd& parameters) {
        auto displaced = input;
        view.unpack(parameters, &displaced);
        Eigen::LLT<Eigen::MatrixXd> factor(input.ao_overlap_matrix);
        const Eigen::MatrixXd white = factor.matrixU() * dense(displaced);
        Eigen::MatrixXd core = white.leftCols(1);
        Eigen::MatrixXd pi = core * (core.transpose() * core).ldlt().solve(core.transpose());
        const Eigen::MatrixXd active = white.rightCols(2) - pi * white.rightCols(2);
        Eigen::MatrixXd pa =
            active * (active.transpose() * active).ldlt().solve(active.transpose());
        return std::make_pair(pi, pa);
    };
    constexpr double h = 1.0e-5;
    const auto plus = projectors(view.pack(input) + h * direction);
    const auto minus = projectors(view.pack(input) - h * direction);
    const double finite =
        ((plus.first - minus.first).squaredNorm() + (plus.second - minus.second).squaredNorm()) /
        (8 * h * h);
    require(std::abs(finite - reference) < 1.0e-8,
            "complete-CAS metric differs from projector derivative norm");

    Eigen::Matrix3d gauge = Eigen::Matrix3d::Identity();
    gauge(0, 0) = 1.7;
    gauge(0, 1) = 0.4;
    gauge(0, 2) = -0.2;
    gauge(1, 2) = 2.0;
    gauge(2, 1) = -0.3;
    gauge(2, 2) = 1.1;
    const Eigen::MatrixXd changed = coefficients * gauge;
    auto changed_input =
        make_input(changed, {{0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}}, 1);
    changed_input.orbital_type = kOrbitalTypeOeo;
    const SparseParameterLayout changed_view(changed_input);
    const OrbitalPhysicalMetric changed_metric(changed_input, true);
    const Eigen::MatrixXd moved_direction =
        Eigen::Map<const Eigen::MatrixXd>(direction.data(), 5, 3) * gauge;
    const Eigen::VectorXd changed_direction =
        Eigen::Map<const Eigen::VectorXd>(moved_direction.data(), 15);
    require(std::abs(changed_metric.squared_norm(changed_view, changed_direction) - reference) <
                1.0e-11,
            "complete-CAS metric is not GL-representative invariant");
    const OrbitalChart changed_chart(
        changed_input, changed_view, changed, changed, nullptr, false, true);
    const NonredundantRetractionMetric original_reduced(complete_cas_chart, view, input);
    const NonredundantRetractionMetric changed_reduced(changed_chart, changed_view, changed_input);
    const Eigen::VectorXd reduced_covector =
        complete_cas_chart.project_gradient(action).reduced_gradient;
    const double cg_dual =
        std::sqrt(reduced_covector.dot(original_reduced.solve(reduced_covector)));
    require(std::abs(original_reduced.dual_norm(reduced_covector) - cg_dual) < 1.0e-10,
            "analytic complete-CAS dual norm differs from the matrix-free Riesz solve");
    const Eigen::MatrixXd changed_covector =
        Eigen::Map<const Eigen::MatrixXd>(action.data(), 5, 3) * gauge.inverse().transpose();
    const Eigen::VectorXd packed_covector =
        Eigen::Map<const Eigen::VectorXd>(changed_covector.data(), 15);
    require(std::abs(original_reduced.dual_norm(
                         complete_cas_chart.project_gradient(action).reduced_gradient) -
                     changed_reduced.dual_norm(
                         changed_chart.project_gradient(packed_covector).reduced_gradient)) <
                1.0e-10,
            "complete-CAS stationarity norm depends on GL representative");
    std::cout << "complete-CAS metric gauge, adjoint, projector and dual invariance: passed\n";
}

void check_curvature_spectral_regularization() {
    Eigen::MatrixXd c = Eigen::MatrixXd::Zero(3, 1);
    c(0, 0) = 1.0;
    auto input = make_input(c, {{0, 1, 2}}, 0);
    input.ao_overlap_matrix.setIdentity();
    const SparseParameterLayout view(input);
    Eigen::Matrix2d rotation;
    rotation << std::cos(0.37), -std::sin(0.37), std::sin(0.37), std::cos(0.37);
    for (const Eigen::Vector2d spectrum : {Eigen::Vector2d(0.3, 2.0),
                                           Eigen::Vector2d(-1.0, 3.0),
                                           Eigen::Vector2d(1.0e-14, 2.0),
                                           Eigen::Vector2d(2.0, 2.0)}) {
        Eigen::MatrixXd f = -2.0 * Eigen::Matrix3d::Identity();
        f.bottomRightCorner<2, 2>().noalias() +=
            0.5 * rotation * spectrum.asDiagonal() * rotation.transpose();
        const OrbitalChart space(input, view, c, c, &f);
        Eigen::MatrixXd u(3, 2);
        for (int column = 0; column < 2; ++column)
            u.col(column) = space.expand_step(Eigen::Vector2d::Unit(column));
        const Eigen::Matrix2d signed_block =
            2.0 * u.transpose() * (f - f(0, 0) * Eigen::Matrix3d::Identity()) * u;
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> reference(signed_block);
        Eigen::Vector2d safe = reference.eigenvalues().cwiseAbs();
        safe = safe.cwiseMax(std::sqrt(std::numeric_limits<double>::epsilon()) * safe.maxCoeff());
        const Eigen::Matrix2d positive =
            reference.eigenvectors() * safe.asDiagonal() * reference.eigenvectors().transpose();
        const Eigen::Matrix2d inverse = reference.eigenvectors() *
                                        safe.cwiseInverse().asDiagonal() *
                                        reference.eigenvectors().transpose();
        const Eigen::Vector2d probe(0.3, -0.7);
        require((space.apply_reduced_curvature(probe) - positive * probe).norm() < 1.0e-12,
                "curvature spectrum regularization differs from the independent reference");
        require((space.apply_inverse_reduced_block_preconditioner(probe) - inverse * probe).norm() <
                    2.0e-8 * std::max(1.0, (inverse * probe).norm()),
                "regularized curvature inverse differs from the independent reference");
    }
}

/** @brief Compare analytic pullback with the independent nested sum. */
void check_analytic_block_pullback(const OrbitalPreparationInput& input,
                                   bool complete_active_space) {
    const SparseParameterLayout layout(input);
    const Eigen::MatrixXd orbitals = dense(input);
    const int n_bf = input.n_basis_functions;
    const int n_active = input.n_active_orbitals;
    const int n_core = input.n_orbitals - n_active;
    AnalyticOrbitalDiagonal model;
    const Eigen::LLT<Eigen::MatrixXd> metric(input.ao_overlap_matrix);
    model.mo_coefficients = metric.matrixU().solve(Eigen::MatrixXd::Identity(n_bf, n_bf));
    if (complete_active_space && input.orbital_type == kOrbitalTypeOeo) {
        const Eigen::MatrixXd white = metric.matrixU() * orbitals;
        const Eigen::HouseholderQR<Eigen::MatrixXd> qr(white);
        const Eigen::MatrixXd frame = qr.householderQ() * Eigen::MatrixXd::Identity(n_bf, n_bf);
        model.mo_coefficients = metric.matrixU().solve(frame);
    }
    model.rotation_diagonal.resize(n_bf, n_bf);
    for (int p = 0; p < n_bf; ++p)
        for (int q = 0; q <= p; ++q)
            model.rotation_diagonal(p, q) = model.rotation_diagonal(q, p) =
                ((p + q) % 3 == 0 ? -1.0 : 1.0) * (0.8 + 0.13 * p + 0.07 * q);
    model.inactive_inverse_transform = Eigen::MatrixXd::Identity(n_core, n_core);
    model.active_inverse_transform = Eigen::MatrixXd::Identity(n_active, n_active);
    if (n_core > 1)
        model.inactive_inverse_transform(0, 1) = 0.23;
    if (n_active > 1)
        model.active_inverse_transform(0, 1) = -0.31;
    const OrbitalChart chart(input,
                             layout,
                             orbitals,
                             orbitals,
                             nullptr,
                             false,
                             complete_active_space,
                             &model,
                             OrbitalChart::Coordinates::Whitened);
    const OrbitalChart raw_chart(input,
                                 layout,
                                 orbitals,
                                 orbitals,
                                 nullptr,
                                 false,
                                 complete_active_space,
                                 &model,
                                 OrbitalChart::Coordinates::Unwhitened);
    const int size = chart.reduced_size();
    require(raw_chart.reduced_size() == size &&
                raw_chart.rank_signature() == chart.rank_signature(),
            "unwhitened quotient removed different physical directions");
    Eigen::MatrixXd conversion(size, size);
    for (int j = 0; j < size; ++j)
        conversion.col(j) =
            raw_chart.project_vector(chart.expand_step(Eigen::VectorXd::Unit(size, j)))
                .reduced_gradient;
    const Eigen::VectorXd packed = Eigen::VectorXd::LinSpaced(layout.size(), -.4, .7);
    const Eigen::VectorXd raw_gradient = raw_chart.project_reduced_gradient(packed);
    const Eigen::VectorXd white_gradient = chart.project_reduced_gradient(packed);
    require((conversion.transpose() * raw_gradient - white_gradient).norm() < 2e-11,
            "unwhitened gradient violates covector pullback");
    require(std::abs(raw_chart.coordinate_dual_norm(raw_gradient) - white_gradient.norm()) < 2e-10,
            "unwhitened progress norm changed its physical units");
    require(std::abs(raw_chart.reference_gradient_inf_norm(raw_gradient) -
                     white_gradient.cwiseAbs().maxCoeff()) < 2e-10,
            "reference-only covector conversion changed the stopping criterion");
    require(
        (raw_chart.inverse_coordinate_metric(raw_gradient) - conversion * white_gradient).norm() <
            2e-9,
        "unwhitened residual seed is not the coordinate Riesz vector");
    const NonredundantRetractionMetric raw_physical(raw_chart, layout, input);
    const NonredundantRetractionMetric white_physical(chart, layout, input);
    const Eigen::VectorXd coordinate_probe = Eigen::VectorXd::LinSpaced(size, -.2, .4);
    require((conversion.transpose() * raw_physical.apply(conversion * coordinate_probe) -
             white_physical.apply(coordinate_probe))
                    .norm() < 2e-9,
            "unwhitened physical metric violates congruence");
    require(
        (raw_chart.expand_step(conversion * coordinate_probe) - chart.expand_step(coordinate_probe))
                    .norm() < 2e-10 &&
            std::abs(raw_chart.maximum_rotation_component(conversion * coordinate_probe) -
                     chart.maximum_rotation_component(coordinate_probe)) < 2e-10,
        "unwhitened orbital update or clipping changed");
    for (const double shift : {-0.7, 0.2})
        require((raw_chart.apply_inverse_augmented_hessian_diagonal(raw_gradient, shift) -
                 conversion * chart.apply_inverse_augmented_hessian_diagonal(white_gradient, shift))
                        .norm() < 2e-8,
                "unwhitened shifted preconditioner changed its physical action");
    for (const auto coordinates :
         {OrbitalChart::Coordinates::Whitened, OrbitalChart::Coordinates::Unwhitened}) {
        const auto& reference =
            coordinates == OrbitalChart::Coordinates::Whitened ? chart : raw_chart;
        OrbitalChart ah(input,
                        layout,
                        orbitals,
                        orbitals,
                        nullptr,
                        false,
                        complete_active_space,
                        nullptr,
                        coordinates);
        ah.set_curvature(input, nullptr, &model, OrbitalChart::CurvatureUse::AugmentedHessian);
        require((ah.augmented_hessian_diagonal() - reference.augmented_hessian_diagonal()).norm() <
                        2e-11 &&
                    (ah.expand_step(coordinate_probe) - reference.expand_step(coordinate_probe))
                            .norm() < 2e-13 &&
                    std::abs(ah.maximum_rotation_component(coordinate_probe) -
                             reference.maximum_rotation_component(coordinate_probe)) < 2e-12,
                "AH-only preparation changed diagonal, coordinates or step clipping");
        for (const double shift : {-0.7, 0.2})
            require((ah.apply_inverse_augmented_hessian_diagonal(coordinate_probe, shift) -
                     reference.apply_inverse_augmented_hessian_diagonal(coordinate_probe, shift))
                            .norm() < 2e-11,
                    "AH-only preparation changed the signed preconditioner");
        if (complete_active_space && input.orbital_type == kOrbitalTypeOeo) {
            bool refused_positive_action = false;
            try {
                ah.apply_reduced_curvature(coordinate_probe);
            } catch (const std::logic_error&) {
                refused_positive_action = true;
            }
            require(refused_positive_action,
                    "AH-only frame unexpectedly retained positive curvature blocks");
        } else {
            require((ah.apply_inverse_reduced_block_preconditioner(coordinate_probe) -
                     reference.apply_inverse_reduced_block_preconditioner(coordinate_probe))
                            .norm() < 2e-10,
                    "constrained AH preparation lost its local spectral block");
        }
    }
    const double positive_shift = 0.7;
    require((raw_chart.shifted_block_inverse(positive_shift)(raw_gradient) -
             conversion * chart.shifted_block_inverse(positive_shift)(white_gradient))
                    .norm() < 2e-9,
            "unwhitened positive shifted preconditioner omitted its coordinate metric");
    const Eigen::MatrixXd one_electron = Eigen::VectorXd::LinSpaced(n_bf, -2.0, 3.0).asDiagonal();
    const OrbitalChart raw_one(input,
                               layout,
                               orbitals,
                               orbitals,
                               &one_electron,
                               false,
                               complete_active_space,
                               nullptr,
                               OrbitalChart::Coordinates::Unwhitened);
    const OrbitalChart white_one(input,
                                 layout,
                                 orbitals,
                                 orbitals,
                                 &one_electron,
                                 false,
                                 complete_active_space,
                                 nullptr,
                                 OrbitalChart::Coordinates::Whitened);
    const Eigen::VectorXd white_one_step =
        white_one.apply_inverse_reduced_block_preconditioner(white_gradient);
    require((raw_one.apply_inverse_reduced_block_preconditioner(raw_gradient) -
             conversion * white_one_step)
                    .norm() < 2e-8 * std::max(1.0, white_one_step.norm()),
            "one-electron inverse changed under unwhitened coordinates");
    require((raw_one.shifted_block_inverse(positive_shift)(raw_gradient) -
             conversion * white_one.shifted_block_inverse(positive_shift)(white_gradient))
                    .norm() < 2e-8,
            "shifted one-electron inverse omitted its AO coordinate metric");
    OrbitalChart deferred(input, layout, orbitals, orbitals, nullptr, false, complete_active_space);
    const Eigen::VectorXd geometry_probe = Eigen::VectorXd::LinSpaced(size, -0.3, 0.6);
    const Eigen::VectorXd geometry_lift = deferred.expand_step(geometry_probe);
    deferred.set_curvature(input, nullptr, &model);
    require((deferred.expand_step(geometry_probe) - geometry_lift).norm() < 1.0e-14 &&
                deferred.rank_signature() == chart.rank_signature(),
            "deferred curvature changed quotient geometry");
    require((deferred.augmented_hessian_diagonal() - chart.augmented_hessian_diagonal()).norm() <
                    1.0e-13 &&
                (deferred.apply_inverse_augmented_hessian_diagonal(geometry_probe, -0.7) -
                 chart.apply_inverse_augmented_hessian_diagonal(geometry_probe, -0.7))
                        .norm() < 1.0e-12 &&
                std::abs(deferred.maximum_rotation_component(geometry_probe) -
                         chart.maximum_rotation_component(geometry_probe)) < 1.0e-13,
            "deferred analytic curvature or AH step mapping differs");
    std::vector<Eigen::VectorXd> lifts;
    for (int j = 0; j < size; ++j)
        lifts.push_back(chart.expand_retract_input_tangent(input, Eigen::VectorXd::Unit(size, j)));
    Eigen::MatrixXd signed_reference = Eigen::MatrixXd::Zero(size, size);
    const int n_core_rotations = (n_bf - n_core) * n_core;
    Eigen::MatrixXd rotation_map =
        Eigen::MatrixXd::Zero(n_core_rotations + (n_bf - input.n_orbitals) * n_active, size);
    Eigen::VectorXd rotation_weights = Eigen::VectorXd::Zero(rotation_map.rows());
    const Eigen::VectorXd probe = Eigen::VectorXd::LinSpaced(size, -0.4, 0.7);
    double maximum_rotation = 0.0;
    for (int target = 0; target < input.n_orbitals; ++target) {
        Eigen::MatrixXd raw = Eigen::MatrixXd::Zero(n_bf, size);
        for (int row = 0; row < stored_sparse_orbital_coefficient_count(input, target); ++row) {
            const int slot = target * n_bf + row;
            const int ao = input.orbital_basis_index_table[slot] - 1;
            for (int j = 0; j < size; ++j)
                raw(ao, j) = lifts[j][slot];
        }
        const double norm =
            std::sqrt(orbitals.col(target).dot(input.ao_overlap_matrix * orbitals.col(target)));
        const Eigen::VectorXd normalized = orbitals.col(target) / norm;
        Eigen::MatrixXd physical = raw / norm;
        physical.noalias() -=
            normalized * (normalized.transpose() * input.ao_overlap_matrix * raw / norm);
        if (target >= n_core && n_core > 0) {
            const auto core = model.mo_coefficients.leftCols(n_core);
            physical.noalias() -= core * (core.transpose() * input.ao_overlap_matrix * physical);
        }
        const Eigen::VectorXd source =
            target < n_core
                ? Eigen::VectorXd(model.inactive_inverse_transform.row(target).transpose())
                : Eigen::VectorXd(model.active_inverse_transform.row(target - n_core).transpose());
        const int source_offset = target < n_core ? 0 : n_core;
        for (int p = n_core; p < n_bf; ++p) {
            const Eigen::VectorXd component =
                physical.transpose() * input.ao_overlap_matrix * model.mo_coefficients.col(p);
            for (int q = 0; q < source.size(); ++q) {
                signed_reference.noalias() += model.rotation_diagonal(p, source_offset + q) *
                                              source[q] * source[q] *
                                              (component * component.transpose());
                maximum_rotation =
                    std::max(maximum_rotation, std::abs(source[q] * component.dot(probe)));
                if (target < n_core || p >= input.n_orbitals) {
                    const int row = target < n_core ? p - n_core + q * (n_bf - n_core)
                                                    : n_core_rotations + p - input.n_orbitals +
                                                          q * (n_bf - input.n_orbitals);
                    rotation_map.row(row).noalias() += source[q] * component.transpose();
                    rotation_weights[row] = model.rotation_diagonal(p, source_offset + q);
                }
            }
        }
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(signed_reference);
    require(spectrum.eigenvalues().cwiseAbs().minCoeff() > 1.0e-6,
            "analytic pullback fixture has an unintended null direction");
    const Eigen::MatrixXd positive = spectrum.eigenvectors() *
                                     spectrum.eigenvalues().cwiseAbs().asDiagonal() *
                                     spectrum.eigenvectors().transpose();
    require((chart.apply_reduced_curvature(probe) - positive * probe).norm() < 1.0e-10,
            "fused analytic curvature differs from nested contraction");
    require((chart.apply_inverse_reduced_block_preconditioner(probe) - positive.ldlt().solve(probe))
                    .norm() < 1.0e-10,
            "fused analytic positive inverse differs from reference");
    const double shift = spectrum.eigenvalues().minCoeff() - 1.0;
    const bool full_frame = complete_active_space && input.orbital_type == kOrbitalTypeOeo;
    const Eigen::MatrixXd shifted_reference =
        full_frame
            ? Eigen::MatrixXd(rotation_map.transpose() *
                              (rotation_weights.array() - shift).matrix().asDiagonal() *
                              rotation_map)
            : Eigen::MatrixXd(signed_reference - shift * Eigen::MatrixXd::Identity(size, size));
    require((chart.apply_inverse_augmented_hessian_diagonal(probe, shift) -
             shifted_reference.ldlt().solve(probe))
                    .norm() < 1.0e-10,
            "factorized AH inverse differs from its dense pullback reference");
    require((chart.augmented_hessian_diagonal() - signed_reference.diagonal()).norm() < 1.0e-10,
            "fused analytic pullback changed signed diagonal");
    const double expected_maximum =
        full_frame ? (rotation_map * probe).cwiseAbs().maxCoeff() : maximum_rotation;
    require(std::abs(chart.maximum_rotation_component(probe) - expected_maximum) < 1.0e-10,
            "fused projection changed maximum rotation component");
}

void check_occupation_curvature() {
    Eigen::MatrixXd c = Eigen::MatrixXd::Zero(3, 1);
    c(0, 0) = 1.0;
    Eigen::MatrixXd f = Eigen::MatrixXd::Zero(3, 3);
    f.diagonal() << -2.0, 1.0, 3.0;
    for (const int inactive_count : {0, 1}) {
        auto input = make_input(c, {{0, 1, 2}}, inactive_count);
        input.ao_overlap_matrix.setIdentity();
        const SparseParameterLayout view(input);
        const OrbitalChart space(input, view, c, c, &f);
        const OrbitalChart geometry(input, view, c, c);
        OrbitalChart deferred(input, view, c, c);
        deferred.set_curvature(input, &f, nullptr);
        require(geometry.rank_signature() == space.rank_signature() &&
                    geometry.reduced_size() == space.reduced_size(),
                "geometry-only chart changed the orbital quotient");
        Eigen::MatrixXd u(3, space.reduced_size()), action(2, 2), inverse(2, 2);
        Eigen::MatrixXd shifted_inverse(2, 2);
        const auto cached_shifted_inverse = space.shifted_block_inverse(0.4);
        require(space.reduced_size() == 2, "invalid occupation fixture rank");
        for (int j = 0; j < 2; ++j) {
            const Eigen::VectorXd unit = Eigen::VectorXd::Unit(2, j);
            u.col(j) = space.expand_step(unit);
            require((geometry.expand_step(unit) - u.col(j)).norm() < 1.0e-14,
                    "geometry-only chart changed the tangent lift");
            require((geometry.project_reduced_gradient(u.col(j)) -
                     space.project_reduced_gradient(u.col(j)))
                            .norm() < 1.0e-14,
                    "geometry-only chart changed the gradient pullback");
            action.col(j) = space.apply_reduced_curvature(unit);
            inverse.col(j) = space.apply_inverse_reduced_block_preconditioner(unit);
            require((deferred.apply_inverse_reduced_block_preconditioner(unit) - inverse.col(j))
                            .norm() < 1.0e-13,
                    "deferred one-electron preconditioner differs");
            shifted_inverse.col(j) =
                space.apply_inverse_reduced_shifted_block_preconditioner(unit, 0.4);
            require((cached_shifted_inverse(unit) - shifted_inverse.col(j)).norm() < 1.0e-12,
                    "cached shifted block inverse differs from direct action");
        }
        // Independent diagonal one-electron gap formula at a stationary ray.
        const double occupation = inactive_count == 1 ? 2.0 : 1.0;
        const Eigen::MatrixXd expected =
            2.0 * occupation * u.transpose() * (f - f(0, 0) * Eigen::MatrixXd::Identity(3, 3)) * u;
        require((action - expected).norm() < 1e-12,
                "production block lost inactive double occupancy or changed active unit model");
        require((inverse * expected - Eigen::MatrixXd::Identity(2, 2)).norm() < 1e-12,
                "inverse occupation curvature is inconsistent");
        require((shifted_inverse * (expected + 0.4 * Eigen::Matrix2d::Identity()) -
                 Eigen::Matrix2d::Identity())
                        .norm() < 1e-12,
                "shifted inverse occupation curvature is inconsistent");
        const std::vector<PackedSecantPair> empty_history;
        const auto standard_lbfgs = build_transported_reduced_lbfgs_preconditioner(
            space, empty_history, 1, LbfgsInitialInverse::ScaledIdentity);
        const auto block_preconditioner = build_transported_reduced_lbfgs_preconditioner(
            space, empty_history, 1, LbfgsInitialInverse::OrbitalBlock);
        const Eigen::Vector2d probe(0.3, -0.7);
        const Eigen::VectorXd pairings = space.orbital_pairings(probe, action * probe);
        require(std::abs(pairings.sum() - probe.dot(action * probe)) < 1.0e-12,
                "orbital diagnostic does not conserve the vector-covector pairing");
        require((standard_lbfgs.apply(probe) - probe).norm() < 1.0e-14,
                "standard L-BFGS did not start from the identity inverse");
        require((block_preconditioner.apply(probe) - inverse * probe).norm() < 1.0e-14,
                "TNHVP L-BFGS did not use the orbital-block inverse");
        // An independent dense inverse-BFGS reference checks cross-orbital
        // secants. The production recursion must not alter the chart curvature.
        auto secant_inverse = block_preconditioner;
        Eigen::Matrix2d reference_inverse = inverse;
        Eigen::Matrix2d coupled_curvature;
        coupled_curvature << 3.0, 0.7, 0.7, 2.0;
        for (const Eigen::Vector2d step : {Eigen::Vector2d(0.2, 0.3), Eigen::Vector2d(-0.4, 0.1)}) {
            const Eigen::Vector2d change = coupled_curvature * step;
            require(secant_inverse.try_add_pair(step, change),
                    "positive cross-orbital secant was rejected");
            const double rho = 1.0 / step.dot(change);
            const Eigen::Matrix2d update =
                Eigen::Matrix2d::Identity() - rho * step * change.transpose();
            reference_inverse =
                (update * reference_inverse * update.transpose() + rho * step * step.transpose())
                    .eval();
            require((secant_inverse.apply(probe) - reference_inverse * probe).norm() < 1.0e-12,
                    "block two-loop recursion differs from inverse BFGS");
            require((secant_inverse.apply(change) - step).norm() < 1.0e-12,
                    "block inverse lost its latest secant equation");
        }
        require((space.apply_reduced_curvature(probe) - action * probe).norm() < 1.0e-14,
                "secant preconditioning modified the physical chart model");
    }
    std::cout << "Inactive occupation and active unit-model production blocks: passed\n";
}

Eigen::VectorXd normalized_rayleigh_gradient(const Eigen::VectorXd& coefficients,
                                             const Eigen::MatrixXd& overlap,
                                             const Eigen::MatrixXd& one_electron) {
    const double norm_squared = coefficients.dot(overlap * coefficients);
    const double energy = coefficients.dot(one_electron * coefficients) / norm_squared;
    return 2.0 * (one_electron * coefficients - energy * overlap * coefficients) / norm_squared;
}

void check_pullback_secant_transport() {
    Eigen::MatrixXd coefficients(3, 1);
    coefficients << 0.8, -0.35, 0.55;
    auto input = make_input(coefficients, {{0, 1, 2}}, 0);
    Eigen::Matrix3d overlap;
    overlap << 1.2, 0.08, -0.03, 0.08, 0.9, 0.04, -0.03, 0.04, 1.1;
    input.ao_overlap_matrix = overlap;
    Eigen::Matrix3d one_electron;
    one_electron << -1.4, 0.22, -0.17, 0.22, 0.6, 0.31, -0.17, 0.31, 1.3;

    const SparseParameterLayout view(input);
    const OrbitalChart old_space(input, view, dense(input), dense(input), nullptr, true);
    require(old_space.reduced_size() == 2,
            "Rayleigh pullback fixture has the wrong quotient dimension");
    const Eigen::Vector2d direction = Eigen::Vector2d(0.7, -0.4).normalized();
    const Eigen::VectorXd packed_direction = old_space.expand_step(direction);
    const Eigen::VectorXd x0 = view.pack(input);
    const Eigen::VectorXd gradient0 =
        normalized_rayleigh_gradient(dense(input).col(0), overlap, one_electron);

    constexpr double finite_difference_step = 1.0e-6;
    auto plus = input;
    auto minus = input;
    view.unpack(x0 + finite_difference_step * packed_direction, &plus);
    view.unpack(x0 - finite_difference_step * packed_direction, &minus);
    const Eigen::VectorXd reference_action =
        old_space.project_reduced_gradient(
            normalized_rayleigh_gradient(dense(plus).col(0), overlap, one_electron) -
            normalized_rayleigh_gradient(dense(minus).col(0), overlap, one_electron)) /
        (2.0 * finite_difference_step);

    constexpr double accepted_step = 2.0e-6;
    const auto next_input = old_space.retract_step(input, direction, accepted_step);
    const OrbitalChart next_space(
        next_input, view, dense(next_input), dense(next_input), nullptr, true);
    const Eigen::VectorXd gradient1 =
        normalized_rayleigh_gradient(dense(next_input).col(0), overlap, one_electron);
    const Eigen::VectorXd packed_secant =
        next_space.project_reduced_gradient(gradient1 - gradient0) / accepted_step;

    const Eigen::VectorXd untransported_gradient_secant =
        (next_space.project_reduced_gradient(gradient1) -
         old_space.project_reduced_gradient(gradient0)) /
        accepted_step;

    const double packed_error = (packed_secant - reference_action).norm();
    const double untransported_gradient_error =
        (untransported_gradient_secant - reference_action).norm();
    require(packed_error < 2.0e-4 * std::max(1.0, reference_action.norm()),
            "ambient gradient-difference secant lost the polar pullback Hessian");
    require(untransported_gradient_error > 1000.0 * packed_error,
            "fixture did not expose the untransported chart-drift term");
    std::cout << "polar pullback ambient secant: passed (packed error " << packed_error
              << ", untransported error " << untransported_gradient_error << ")\n";
}

void check_ill_conditioned_representative(const OrbitalPreparationInput& input,
                                          int expected_dimension) {
    const SparseParameterLayout view(input);
    const Eigen::MatrixXd orbitals = dense(input);
    const OrbitalChart space(input, view, orbitals, orbitals, nullptr, true);
    require(space.reduced_size() == expected_dimension,
            "ill-conditioned representative changed the quotient dimension");
    const auto diagnostics = space.structural_diagnostics();
    require(diagnostics.total_gauge_rank == view.size() - expected_dimension,
            "ill-conditioned representative changed the stable gauge rank");
    for (int column = 0; column < space.reduced_size(); ++column) {
        const Eigen::VectorXd unit = Eigen::VectorXd::Unit(space.reduced_size(), column);
        require((space.project_vector(space.expand_step(unit)).reduced_gradient - unit).norm() <
                    1.0e-9,
                "ill-conditioned quotient coordinates are not invertible");
    }
    std::cout << "ill-conditioned inactive representative: passed (" << view.size() << " -> "
              << space.reduced_size() << ")\n";
}

void check_metric_blocks(const OrbitalPreparationInput& input, bool active_span) {
    const SparseParameterLayout layout(input);
    const OrbitalPhysicalMetric metric(input, active_span);
    const int n_occupied =
        (input.n_total_electrons - input.n_active_electrons) / 2 + input.n_active_orbitals;
    for (int orbital = 0; orbital < n_occupied; ++orbital) {
        const int count = layout.orbital_coefficient_count(orbital);
        std::vector<int> support(count);
        Eigen::MatrixXd lift(count, 3);
        for (int row = 0; row < count; ++row) {
            support[row] =
                input.orbital_basis_index_table[orbital * input.n_basis_functions + row] - 1;
            for (int column = 0; column < 3; ++column)
                lift(row, column) = std::sin((row + 0.3) * (column + 0.7));
        }
        Eigen::MatrixXd reference(3, 3);
        for (int column = 0; column < 3; ++column) {
            Eigen::VectorXd direction = Eigen::VectorXd::Zero(layout.size());
            for (int row = 0; row < count; ++row)
                direction[layout.packed_index(orbital, row)] = lift(row, column);
            const Eigen::VectorXd image = metric.apply(layout, direction);
            Eigen::VectorXd local_image(count);
            for (int row = 0; row < count; ++row)
                local_image[row] = image[layout.packed_index(orbital, row)];
            reference.col(column) = lift.transpose() * local_image;
        }
        const Eigen::MatrixXd block = metric.orbital_block(orbital, support, lift);
        require((block - reference).norm() < 1.0e-11 * std::max(1.0, reference.norm()),
                "analytic metric block disagrees with full coupled action");
        for (int column = 0; column < 3; ++column) {
            Eigen::VectorXd direction = Eigen::VectorXd::Zero(layout.size());
            for (int row = 0; row < count; ++row)
                direction[layout.packed_index(orbital, row)] = lift(row, column);
            require(std::abs(block(column, column) - metric.squared_norm(layout, direction)) <
                        1.0e-11 * std::max(1.0, std::abs(block(column, column))),
                    "analytic metric block disagrees with independent feature norm");
        }
    }
    std::cout << "analytic physical metric orbital blocks: passed\n";
}

/** @brief The norm and action share an adjoint even with cancelling representatives. */
void check_metric_feature_adjoint() {
    Eigen::MatrixXd coefficients = Eigen::MatrixXd::Zero(6, 4);
    coefficients(0, 0) = coefficients(0, 1) = 1.0;
    coefficients(1, 1) = 1e-5;
    coefficients(0, 2) = 1000.0;
    coefficients(2, 2) = 1.0;
    coefficients(1, 3) = -2000.0;
    coefficients(3, 3) = 1.0;
    const auto input =
        make_input(coefficients,
                   {{0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}},
                   2);
    const SparseParameterLayout layout(input);
    for (const bool span : {false, true}) {
        const OrbitalPhysicalMetric metric(input, span);
        for (int probe = 0; probe < 8; ++probe) {
            Eigen::VectorXd x(layout.size()), y(layout.size());
            for (int i = 0; i < layout.size(); ++i) {
                x[i] = std::sin(0.31 * (i + 1) * (probe + 2));
                y[i] = std::cos(0.19 * (i + 2) * (probe + 1));
            }
            const auto mx = metric.apply(layout, x), my = metric.apply(layout, y);
            const double nx = metric.squared_norm(layout, x), ny = metric.squared_norm(layout, y);
            const double bilinear =
                0.25 * (metric.squared_norm(layout, x + y) - metric.squared_norm(layout, x - y));
            const double scale = std::sqrt(nx * ny);
            require(std::abs(x.dot(mx) - nx) < 1e-10 * nx &&
                        std::abs(x.dot(my) - bilinear) < 1e-10 * scale &&
                        std::abs(x.dot(my) - y.dot(mx)) < 1e-10 * scale,
                    "metric action is not the adjoint of its physical feature map");
        }
    }
}

void check_metric_inverse_certificates(const OrbitalPreparationInput& input,
                                       bool active_span = false) {
    const SparseParameterLayout view(input);
    const Eigen::MatrixXd orbitals = dense(input);
    const OrbitalChart space(input, view, orbitals, orbitals, nullptr, true, active_span);
    const NonredundantRetractionMetric metric(space, view, input);
    const Eigen::Index size = space.reduced_size();
    Eigen::MatrixXd gram(size, size);
    for (Eigen::Index column = 0; column < size; ++column)
        gram.col(column) = metric.apply(Eigen::VectorXd::Unit(size, column));
    const Eigen::LDLT<Eigen::MatrixXd> factor(gram);
    require(factor.info() == Eigen::Success && factor.isPositive(),
            "metric inverse fixture is not positive definite");
    for (int sample = 0; sample < 6; ++sample) {
        Eigen::VectorXd rhs(size);
        for (Eigen::Index i = 0; i < size; ++i)
            rhs[i] = std::sin((i + 1.0) * (sample + 0.7));
        for (const double scale : {1.0e-8, 1.0, 1.0e8}) {
            const Eigen::VectorXd covector = scale * rhs;
            const Eigen::VectorXd solution = metric.solve(covector);
            require((covector - metric.apply(solution)).norm() <=
                        std::sqrt(std::numeric_limits<double>::epsilon()) * covector.norm(),
                    "metric inverse accepted without its original true-residual certificate");
            const Eigen::VectorXd reference = factor.solve(covector);
            require((solution - reference).norm() <= 1.0e-7 * reference.norm(),
                    "certified metric inverse disagrees with dense reference");
        }
    }
    std::cout << "metric inverse true-residual certificates and RHS scaling: passed\n";
}

class DenseTestHvp final : public ReducedHvp {
  public:
    explicit DenseTestHvp(Eigen::MatrixXd hessian) : hessian_(std::move(hessian)) {}

    Eigen::VectorXd apply(const Eigen::VectorXd& direction) override {
        ++applies;
        return hessian_ * direction;
    }

    int applies = 0;

  private:
    Eigen::MatrixXd hessian_;
};

/** @brief Two response-space enrichments followed by a fixed symmetric model. */
class ChangingModelHvp final : public ReducedHvp {
  public:
    ChangingModelHvp(Eigen::MatrixXd initial, Eigen::MatrixXd enriched)
        : hessian_(std::move(initial)), enriched_(std::move(enriched)) {}

    Eigen::VectorXd apply(const Eigen::VectorXd& direction) override {
        const Eigen::MatrixXd block = direction;
        return apply_batch(block).col(0);
    }

    Eigen::MatrixXd apply_batch(const Eigen::Ref<const Eigen::MatrixXd>& directions) override {
        ++batch_applies;
        if (batch_applies == 1) {
            ++revision_;
        } else if (batch_applies == 2) {
            hessian_ = enriched_;
            ++revision_;
        }
        prepared_directions.emplace_back(directions);
        return hessian_ * directions;
    }

    Eigen::MatrixXd
    apply_frozen_batch(const Eigen::Ref<const Eigen::MatrixXd>& directions) override {
        ++frozen_applies;
        frozen_directions.emplace_back(directions);
        return hessian_ * directions;
    }

    std::uint64_t model_revision() const noexcept override {
        return revision_;
    }

    /** @brief Simulates response enrichment outside a retained radius trial. */
    void replace_model(Eigen::MatrixXd hessian) {
        hessian_ = std::move(hessian);
        ++revision_;
    }

    const Eigen::MatrixXd& hessian() const noexcept {
        return hessian_;
    }

    int batch_applies = 0;
    int frozen_applies = 0;
    std::vector<Eigen::MatrixXd> prepared_directions;
    std::vector<Eigen::MatrixXd> frozen_directions;

  private:
    Eigen::MatrixXd hessian_;
    Eigen::MatrixXd enriched_;
    std::uint64_t revision_ = 0;
};

/** @brief Checks cached Newton data against an independently applied model. */
void check_cached_newton_model(const TruncatedNewtonStepResult& step,
                               const ChangingModelHvp& hvp,
                               const Eigen::VectorXd& gradient,
                               const NonredundantRetractionMetric& metric) {
    const auto& subspace = step.subspace;
    require(subspace.model_revision == hvp.model_revision(),
            "Newton subspace retained an obsolete response-model revision");
    const Eigen::MatrixXd expected_images = hvp.hessian() * subspace.orthonormal_basis;
    require((subspace.hessian_basis - expected_images).norm() <
                1.0e-11 * std::max(1.0, expected_images.norm()),
            "Newton subspace mixed Hessian images from different models");
    const Eigen::MatrixXd expected_projection =
        subspace.orthonormal_basis.transpose() * expected_images;
    require((subspace.reduced_hessian - expected_projection).norm() <
                1.0e-11 * std::max(1.0, expected_projection.norm()),
            "projected Newton Hessian was not rebuilt after response enrichment");
    require((expected_projection - expected_projection.transpose()).norm() <
                1.0e-11 * std::max(1.0, expected_projection.norm()),
            "changing-model fixture lost symmetric curvature");

    const Eigen::VectorXd expected_hvp = hvp.hessian() * step.reduced_step;
    const Eigen::VectorXd expected_metric = metric.apply(step.reduced_step);
    require((step.reduced_hessian_times_step - expected_hvp).norm() <
                1.0e-11 * std::max(1.0, expected_hvp.norm()),
            "Newton step retained a stale Hessian image");
    require((step.reduced_metric_times_step - expected_metric).norm() <
                1.0e-11 * std::max(1.0, expected_metric.norm()),
            "Newton step lost the physical metric image");
    const double expected_prediction =
        -gradient.dot(step.reduced_step) - 0.5 * step.reduced_step.dot(expected_hvp);
    require(std::abs(step.predicted_decrease - expected_prediction) <
                1.0e-12 * std::max(1.0, std::abs(expected_prediction)),
            "Newton predicted decrease refers to an obsolete response model");
    const double expected_kkt =
        (gradient + expected_hvp + step.trust_region_shift * expected_metric).norm() /
        gradient.norm();
    require(std::abs(step.model_kkt_relative_residual - expected_kkt) < 1.0e-10,
            "Newton KKT certificate refers to an obsolete response model");
}

void check_truncated_newton_certificates() {
    Eigen::VectorXd gradient(2);
    gradient << 1.0, 2.0;
    TruncatedNewtonStepResult step;
    step.reduced_step = -gradient;
    step.reduced_hessian_times_step = -gradient;
    step.reduced_metric_times_step = -gradient;
    step.target_kkt_relative_residual = kFixtureForcing;
    refresh_truncated_newton_step_certificate(gradient, &step);
    require(step.model_kkt_converged && step.newton_forcing_converged,
            "exact interior Newton model lacks a forcing certificate");

    step.reduced_hessian_times_step = -0.9 * gradient;
    step.target_kkt_relative_residual = 0.2;
    refresh_truncated_newton_step_certificate(gradient, &step);
    require(step.model_kkt_converged, "explicit loose KKT target rejected a valid step");
    step.target_kkt_relative_residual = 0.05;
    refresh_truncated_newton_step_certificate(gradient, &step);
    require(!step.model_kkt_converged, "explicit tight KKT target accepted an unresolved step");

    step.reduced_hessian_times_step.setZero();
    step.trust_region_shift = 1.0;
    step.reached_boundary = true;
    refresh_truncated_newton_step_certificate(gradient, &step);
    require(step.model_kkt_converged && !step.newton_forcing_converged,
            "shifted boundary KKT was misreported as Newton convergence");

    OrbitalChart::ProjectionResult ray_projection;
    ray_projection.reduced_gradient = Eigen::Vector2d(1.0, 0.0);
    TruncatedNewtonStepResult ray_step;
    ray_step.reduced_step = Eigen::Vector2d(-1.0, 0.0);
    ray_step.reduced_hessian_times_step = Eigen::Vector2d(-3.0, 0.0);
    ray_step.reduced_metric_times_step = ray_step.reduced_step;
    ray_step.retract_tangent_norm = 1.0;
    ray_step.reached_boundary = true;
    ray_step.trust_region_shift = 2.0;
    ray_step.target_kkt_relative_residual = 0.5;
    require(minimize_truncated_newton_step_on_ray(ray_projection, &ray_step) &&
                std::abs(ray_step.reduced_step[0] + 1.0 / 3.0) < 1.0e-14 &&
                std::abs(ray_step.predicted_decrease - 1.0 / 6.0) < 1.0e-14 &&
                !ray_step.reached_boundary && ray_step.trust_region_shift == 0.0,
            "exact directional quadratic was not minimized on the sampled ray");

    require(!observed_contraction_requires_newton_correction(2.0, 0.5, 0.25) &&
                observed_contraction_requires_newton_correction(2.0, 0.5000000001, 0.25) &&
                observed_contraction_requires_newton_correction(0.0, 0.0, 0.25),
            "observed outer contraction did not enforce the Newton target");
}

void check_residual_driven_subspace(const OrbitalPreparationInput& input) {
    const SparseParameterLayout view(input);
    const Eigen::MatrixXd c = dense(input);
    const OrbitalChart space(input, view, c, c, nullptr, true);
    const NonredundantRetractionMetric metric(space, view, input);
    const int dimension = space.reduced_size();
    require(dimension > 4, "residual-driven fixture is too small");
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(dimension, dimension);
    for (int i = 0; i < dimension; ++i)
        hessian(i, i) = 1.0 + i;
    const Eigen::VectorXd gradient = Eigen::VectorXd::LinSpaced(dimension, 1.0e-3, 2.0e-3);
    OrbitalChart::ProjectionResult projection;
    projection.reduced_gradient = gradient;

    DenseTestHvp unrestricted_hvp(hessian);
    const auto unrestricted = solve_nonredundant_truncated_newton_step(
        metric, space, projection, 10.0, 1.0e-12, 1.0e-8, 1.0e-12, &unrestricted_hvp, nullptr);
    require(unrestricted.subspace_dimension > 4 && unrestricted.subspace_dimension <= dimension &&
                unrestricted.stop_reason == TruncatedNewtonStopReason::ModelKktConverged &&
                unrestricted.newton_forcing_converged,
            "residual-driven Newton solve stopped before its certificate");

    DenseTestHvp reused_hvp(hessian);
    const auto reused = solve_nonredundant_truncated_newton_step(metric,
                                                                 space,
                                                                 projection,
                                                                 10.0,
                                                                 1.0e-12,
                                                                 1.0e-8,
                                                                 1.0e-12,
                                                                 &reused_hvp,
                                                                 nullptr,
                                                                 nullptr,
                                                                 &unrestricted.subspace);
    require(reused.model_kkt_converged && reused_hvp.applies == 0,
            "certified same-point subspace recomputed exact HVP samples");

    Eigen::MatrixXd clustered = Eigen::MatrixXd::Zero(dimension, dimension);
    for (int i = 0; i < dimension; ++i) {
        clustered(i, i) = (i % 3 == 0) ? 1.0 : (i % 3 == 1 ? 4.0 : 16.0);
    }
    DenseTestHvp clustered_hvp(clustered);
    const auto certified = solve_nonredundant_truncated_newton_step(
        metric, space, projection, 10.0, 1.0e-7, 1.0e-3, kFixtureForcing, &clustered_hvp, nullptr);
    require(certified.subspace_dimension <= 4 && clustered_hvp.applies <= 4 &&
                certified.stop_reason == TruncatedNewtonStopReason::ModelKktConverged &&
                certified.newton_forcing_converged &&
                certified.model_kkt_relative_residual < kFixtureForcing,
            "clustered Newton subspace did not certify the full-space model");

    hessian(0, 0) = -10.0;
    DenseTestHvp indefinite(hessian);
    const auto boundary = solve_nonredundant_truncated_newton_step(
        metric, space, projection, 1.0e-3, 1.0e-7, 1.0e-3, kFixtureForcing, &indefinite, nullptr);
    require(boundary.reached_boundary && boundary.encountered_negative_curvature &&
                !boundary.newton_forcing_converged,
            "negative curvature did not produce a certified boundary model");

    const Eigen::VectorXd exact_predictor = -clustered.ldlt().solve(gradient);
    DenseTestHvp certified_predictor_hvp(clustered);
    const auto certified_predictor =
        solve_nonredundant_truncated_newton_step(metric,
                                                 space,
                                                 projection,
                                                 10.0,
                                                 1.0e-7,
                                                 1.0e-3,
                                                 kFixtureForcing,
                                                 &certified_predictor_hvp,
                                                 nullptr,
                                                 &exact_predictor);
    require(certified_predictor_hvp.applies == 1 && certified_predictor.subspace_dimension == 0 &&
                certified_predictor.newton_forcing_converged,
            "exact block-L-BFGS predictor was not certified by one HVP");

    Eigen::MatrixXd two_cluster = Eigen::MatrixXd::Zero(dimension, dimension);
    for (int i = 0; i < dimension; ++i) {
        two_cluster(i, i) = i % 2 == 0 ? 1.0 : 4.0;
    }
    const Eigen::VectorXd identity_predictor = -gradient;
    DenseTestHvp defect_hvp(two_cluster);
    const auto corrected_predictor = solve_nonredundant_truncated_newton_step(metric,
                                                                              space,
                                                                              projection,
                                                                              10.0,
                                                                              1.0e-7,
                                                                              1.0e-3,
                                                                              kFixtureForcing,
                                                                              &defect_hvp,
                                                                              nullptr,
                                                                              &identity_predictor);
    require(defect_hvp.applies == 3 && corrected_predictor.subspace_dimension == 2 &&
                corrected_predictor.newton_forcing_converged,
            "predictor Newton defect was not resolved in its correction space");
    std::cout << "residual-driven Newton subspace: passed\n";
}

void check_response_model_refresh(const OrbitalPreparationInput& input) {
    const SparseParameterLayout view(input);
    const Eigen::MatrixXd c = dense(input);
    const OrbitalChart space(input, view, c, c, nullptr, true);
    const NonredundantRetractionMetric metric(space, view, input);
    const int dimension = space.reduced_size();
    require(dimension > 4, "response-model refresh fixture is too small");
    Eigen::MatrixXd initial = Eigen::MatrixXd::Zero(dimension, dimension);
    for (int i = 0; i < dimension; ++i)
        initial(i, i) = 1.0 + i;
    const Eigen::VectorXd coupling = Eigen::VectorXd::LinSpaced(dimension, 0.25, 1.0).normalized();
    const Eigen::MatrixXd enriched = 1.3 * initial + 0.2 * coupling * coupling.transpose();
    ChangingModelHvp hvp(initial, enriched);
    const Eigen::VectorXd gradient = Eigen::VectorXd::LinSpaced(dimension, 1.0e-3, 2.0e-3);
    OrbitalChart::ProjectionResult projection;
    projection.reduced_gradient = gradient;
    constexpr double forcing = 1.0e-10;

    const auto step = solve_nonredundant_truncated_newton_step(
        metric, space, projection, 10.0, 1.0e-16, 1.0e-12, forcing, &hvp, nullptr);
    require(hvp.model_revision() == 2 && hvp.batch_applies >= 2 && hvp.frozen_applies > 0 &&
                step.newton_forcing_converged,
            "Newton solve did not refresh its evolving response model");
    require(
        hvp.prepared_directions.front().cols() == 1 && hvp.frozen_directions.front().cols() == 1 &&
            (hvp.frozen_directions.front().col(0) - hvp.prepared_directions.front().col(0)).norm() <
                1.0e-14,
        "response enrichment did not refresh the stale baseline image");
    require(((enriched - initial) * hvp.prepared_directions.front().col(0)).norm() > 1.0e-2,
            "refresh fixture did not change the original Hessian sample");
    check_cached_newton_model(step, hvp, gradient, metric);

    const int prepared_before_reuse = hvp.batch_applies;
    const int frozen_before_reuse = hvp.frozen_applies;
    const auto reused = solve_nonredundant_truncated_newton_step(metric,
                                                                 space,
                                                                 projection,
                                                                 5.0,
                                                                 1.0e-16,
                                                                 1.0e-12,
                                                                 forcing,
                                                                 &hvp,
                                                                 nullptr,
                                                                 nullptr,
                                                                 &step.subspace);
    require(reused.newton_forcing_converged && hvp.batch_applies == prepared_before_reuse &&
                hvp.frozen_applies == frozen_before_reuse,
            "unchanged response-model reuse unnecessarily recomputed HVPs");
    check_cached_newton_model(reused, hvp, gradient, metric);

    hvp.replace_model(enriched + 2.0 * Eigen::MatrixXd::Identity(dimension, dimension));
    const Eigen::VectorXd exact_retry_step = -hvp.hessian().ldlt().solve(gradient);
    const double retry_radius = 0.4 * metric.norm(exact_retry_step);
    const auto retry = solve_nonredundant_truncated_newton_step(metric,
                                                                space,
                                                                projection,
                                                                retry_radius,
                                                                1.0e-16,
                                                                1.0e-12,
                                                                forcing,
                                                                &hvp,
                                                                nullptr,
                                                                nullptr,
                                                                &step.subspace);
    require(hvp.frozen_applies > frozen_before_reuse && retry.reached_boundary &&
                retry.trust_region_shift > 0.0,
            "radius retry did not invalidate a stale response-model cache");
    check_cached_newton_model(retry, hvp, gradient, metric);
    require(metric.norm(retry.reduced_step) <= retry_radius * (1.0 + 1.0e-8),
            "refreshed Newton step exceeded its revised trust radius");
    std::cout << "response-model image refresh and radius reuse: passed\n";
}
} // namespace

int main() {
    try {
        check_whitened_tangent_inverse();
        check_response_spectrum_summary();
        check_accuracy_aware_forcing();
        check_curvature_spectral_regularization();
        check_occupation_curvature();
        check_pullback_secant_transport();
        Eigen::MatrixXd c(6, 4);
        c << 1, 0, 0.3, 0, 0, 1, 0.4, 0, 1, 1, 0, 0.2, 1, 1, 0, 0.3, 0, 0, 1, 0, 0, 0, 0, 1;
        auto partial = make_input(c, {{0, 2, 3}, {1, 2, 3}, {0, 1, 4}, {2, 3, 5}}, 2);
        check("unequal supports with off-support cancellation", partial, 7);
        auto frozen = partial;
        frozen.original_orbital_basis_counts = {3, 3, 2, 3};
        check("stored but frozen active tail", frozen, 7);
        auto full = make_input(
            c, {{0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}}, 2);
        check("full support", full, 14);
        auto reordered = make_input(
            c, {{5, 4, 3, 2, 1, 0}, {0, 1, 2, 3, 4, 5}, {5, 4, 3, 2, 1, 0}, {0, 1, 2, 3, 4, 5}}, 2);
        check("mixed full-support AO ordering", reordered, 14);
        reordered.orbital_type = kOrbitalTypeOeo;
        check_preconditioner_geometry(reordered, true);
        check_metric_inverse_certificates(reordered, true);
        check_finite_step_gradient_pullback(partial);
        check_finite_step_gradient_pullback(full);
        check_metric_inverse_certificates(partial);
        check_metric_inverse_certificates(full);
        auto oeo = full;
        oeo.orbital_type = kOrbitalTypeOeo;
        check_analytic_block_pullback(partial, false);
        check_analytic_block_pullback(frozen, false);
        check_analytic_block_pullback(oeo, true);
        check_analytic_block_pullback(oeo, false);
        check_preconditioner_geometry(partial);
        check_preconditioner_geometry(full);
        check_preconditioner_geometry(oeo, true);
        check_metric_inverse_certificates(oeo, true);
        check_metric_feature_adjoint();
        check_metric_blocks(partial, false);
        check_metric_blocks(frozen, false);
        check_metric_blocks(full, false);
        check_metric_blocks(full, true);
        check_complete_oeo_active_subspace_gauge();
        check_truncated_newton_certificates();
        check_residual_driven_subspace(full);
        check_response_model_refresh(full);
        Eigen::MatrixXd rotated = c;
        Eigen::Matrix2d a;
        a << 1, 0.3, -0.2, 1.1;
        rotated.leftCols(2) = (c.leftCols(2) * a).eval();
        check("inactive gauge rotation",
              make_input(
                  rotated,
                  {{0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}},
                  2),
              14);
        Eigen::MatrixXd ill_conditioned = c;
        Eigen::Matrix2d ill_conditioned_gauge;
        ill_conditioned_gauge << 1.0, 1.0, 0.0, 1.0e-4;
        ill_conditioned.leftCols(2) = (c.leftCols(2) * ill_conditioned_gauge).eval();
        check_ill_conditioned_representative(
            make_input(
                ill_conditioned,
                {{0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}},
                2),
            14);
        const Eigen::MatrixXd active = c.rightCols(2);
        check("no inactive orbitals", make_input(active, {{0, 1, 4}, {2, 3, 5}}, 0), 4);
        check_metric_inverse_certificates(make_input(active, {{0, 1, 4}, {2, 3, 5}}, 0));
        check("inactive only", make_input(c.leftCols(2), {{0, 2, 3}, {1, 2, 3}}, 2), 4);
        check_metric_inverse_certificates(make_input(c.leftCols(2), {{0, 2, 3}, {1, 2, 3}}, 2));
        Eigen::MatrixXd single = Eigen::MatrixXd::Identity(2, 2);
        check("all coordinates are scale gauge", make_input(single, {{0}, {1}}, 0), 0);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
