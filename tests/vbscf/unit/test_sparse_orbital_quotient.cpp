#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/physical_metric.hpp"
#include "vbscf/core/contracts/orbital_type.hpp"
#include "vbscf/optimization/preconditioners/transported_lbfgs.hpp"
#include "vbscf/optimization/trust_region/retraction.hpp"
#include "vbscf/optimization/trust_region/truncated_newton.hpp"
#include "vbscf/diagnostics/orbitals/chart_audit.hpp"
#include "vbscf/derivatives/hessian/exact/operator.hpp"

namespace {
using namespace xmvb::vb;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
constexpr double kFixtureForcing = 5.0e-2;

void check_response_spectrum_summary() {
  StructureResponseSchurModel model;
  model.orbital_couplings = Eigen::Matrix3d::Identity();
  model.projected_inverse =
      (Eigen::Vector3d() << 4.0, -1.0, 0.25).finished().asDiagonal();
  const ResponseSpectrumSummary reference =
      summarize_response_spectrum(model);
  require(reference.model_rank == 3 && reference.rank_90 == 2 &&
              reference.rank_99 == 3,
          "response spectrum summary returned incorrect capture ranks");
  require(std::abs(reference.top_mode_fraction - 4.0 / 5.25) < 1.0e-14 &&
              std::abs(reference.effective_rank -
                       (5.25 * 5.25) / (16.0 + 1.0 + 0.0625)) < 1.0e-14,
          "response spectrum summary returned incorrect weights");

  Eigen::Matrix3d rotation =
      Eigen::AngleAxisd(0.37, Eigen::Vector3d(1.0, 2.0, -1.0).normalized())
          .toRotationMatrix();
  model.orbital_couplings = rotation;
  model.projected_inverse =
      rotation.transpose() *
      (Eigen::Vector3d() << 4.0, -1.0, 0.25).finished().asDiagonal() *
      rotation;
  const ResponseSpectrumSummary rotated =
      summarize_response_spectrum(model);
  require(rotated.rank_90 == reference.rank_90 &&
              rotated.rank_99 == reference.rank_99 &&
              std::abs(rotated.effective_rank - reference.effective_rank) <
                  1.0e-13,
          "response spectrum summary depends on response-basis rotation");
}

void check_accuracy_aware_forcing() {
  constexpr double initial_norm = 4.0;
  constexpr double gradient_tolerance = 1.0e-3;
  const double accuracy_floor =
      std::sqrt(gradient_tolerance / initial_norm);
  require(
      inexact_newton_forcing_term(
          initial_norm, initial_norm, initial_norm,
          gradient_tolerance) == 0.5,
      "accuracy-aware forcing lost its contraction limit");
  require(
      std::abs(inexact_newton_forcing_term(
                   1.0, 1.0, initial_norm,
                   gradient_tolerance) -
               0.25) < 1.0e-15,
      "accuracy-aware forcing does not track gradient progress");
  require(
      std::abs(inexact_newton_forcing_term(
                   1.0e-2, 1.0e-2, initial_norm,
                   gradient_tolerance) -
               accuracy_floor) < 1.0e-15,
      "accuracy-aware forcing did not stop at the outer accuracy floor");
  require(
      std::abs(inexact_newton_forcing_term(
                   1.0e-5, 1.0e-5, initial_norm,
                   gradient_tolerance) -
               accuracy_floor) < 1.0e-15,
      "outer target over-solved beyond the requested accuracy");
  require(
      std::abs(inexact_newton_forcing_term(
                   6.0e-3, 1.7e-3, 6.0e-3,
                   gradient_tolerance) -
               (gradient_tolerance / 6.0e-3)) < 1.0e-15,
      "one-contraction outer target was not enforced");
}

OrbitalPreparationInput make_input(
    const Eigen::MatrixXd& coefficients,
    const std::vector<std::vector<int>>& supports, int inactive_count) {
  OrbitalPreparationInput input;
  input.n_basis_functions = coefficients.rows();
  input.n_orbitals = coefficients.cols();
  input.n_active_orbitals = coefficients.cols() - inactive_count;
  input.n_active_electrons = input.n_active_orbitals;
  input.n_total_electrons = 2 * inactive_count + input.n_active_electrons;
  input.orbital_type = 1;
  input.ao_overlap_matrix = Eigen::MatrixXd::Identity(
      coefficients.rows(), coefficients.rows());
  // Nontrivial SPD AO metric; the algebraic gauge must not depend on it.
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
  Eigen::MatrixXd c = Eigen::MatrixXd::Zero(
      input.n_basis_functions, input.n_orbitals);
  for (int p = 0; p < c.cols(); ++p) {
    for (int j = 0; j < stored_sparse_orbital_coefficient_count(input, p); ++j) {
      const int slot = p * c.rows() + j;
      c(input.orbital_basis_index_table[slot] - 1, p) =
          input.orbital_value_table[slot];
    }
  }
  return c;
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
    const Eigen::VectorXd b =
        c.col(ni + a) - p * input.ao_overlap_matrix * c.col(ni + a);
    require(b.norm() > 1e-12, "degenerate synthetic active ray");
    const Eigen::MatrixXd ray = b * b.transpose() / b.squaredNorm();
    result.segment((a + 1) * n * n, n * n) =
        Eigen::Map<const Eigen::VectorXd>(ray.data(), n * n);
  }
  return result;
}

double finite_physical_metric_squared_norm(
    const OrbitalPreparationInput& input,
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
  const int n_inactive =
      (input.n_total_electrons - input.n_active_electrons) / 2;
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
    complement.noalias() -=
        inactive * inverse * inactive.transpose() * overlap;
  }
  auto projected_active = [&](const Eigen::MatrixXd& orbitals) {
    Eigen::MatrixXd projector = Eigen::MatrixXd::Identity(n_bf, n_bf);
    if (n_inactive > 0) {
      const auto inactive = orbitals.leftCols(n_inactive);
      const Eigen::MatrixXd local_inverse =
          (inactive.transpose() * overlap * inactive)
              .ldlt()
              .solve(Eigen::MatrixXd::Identity(n_inactive, n_inactive));
      projector.noalias() -=
          inactive * local_inverse * inactive.transpose() * overlap;
    }
    return (projector * orbitals.middleCols(n_inactive, n_active)).eval();
  };

  double squared_norm = 0.0;
  if (n_inactive > 0) {
    const Eigen::MatrixXd delta_inactive =
        (displaced_plus.leftCols(n_inactive) -
         displaced_minus.leftCols(n_inactive)) /
        (2.0 * step);
    const Eigen::MatrixXd horizontal = complement * delta_inactive;
    squared_norm +=
        (inverse * horizontal.transpose() * overlap * horizontal).trace();
  }
  const Eigen::MatrixXd base_active =
      projected_active(current);
  const Eigen::MatrixXd delta_active =
      (projected_active(displaced_plus) -
       projected_active(displaced_minus)) /
      (2.0 * step);
  for (int active = 0; active < n_active; ++active) {
    const Eigen::VectorXd ray = base_active.col(active);
    const double norm_squared = ray.dot(overlap * ray);
    Eigen::VectorXd horizontal = delta_active.col(active);
    horizontal.noalias() -=
        ray * (ray.dot(overlap * horizontal) / norm_squared);
    squared_norm += horizontal.dot(overlap * horizontal) / norm_squared;
  }
  return squared_norm;
}

void check(const std::string& name, const OrbitalPreparationInput& input,
           int expected_dimension) {
  const SparseParameterLayout view(input);
  const Eigen::MatrixXd c = dense(input);
  OrbitalChart space(input, view, c, c, nullptr, true);
  require(space.reduced_size() == expected_dimension,
          name + ": expected dimension " + std::to_string(expected_dimension) +
          ", got " + std::to_string(space.reduced_size()));
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
  const OrbitalPhysicalMetric coupled_metric(input);
  for (int gauge = 0; gauge < audit.packed_gauge_basis.cols(); ++gauge) {
    require(coupled_metric.squared_norm(
                view, audit.packed_gauge_basis.col(gauge)) < 1.0e-18,
            name + ": coupled metric did not annihilate gauge");
    require(coupled_metric.apply(view, audit.packed_gauge_basis.col(gauge))
                .norm() < 1.0e-10,
            name + ": matrix-free metric action retained gauge");
  }
  if (u.cols() > 0) {
    const Eigen::VectorXd direction =
        u * Eigen::VectorXd::LinSpaced(u.cols(), -0.7, 0.9).normalized();
    const double analytic = coupled_metric.squared_norm(view, direction);
    const double finite =
        finite_physical_metric_squared_norm(input, view, direction);
    require(std::abs(analytic - finite) <
                1.0e-7 * std::max(1.0, finite),
            name + ": coupled metric differs from finite physical map");
    const Eigen::VectorXd action = coupled_metric.apply(view, direction);
    require(std::abs(direction.dot(action) - analytic) <
                1.0e-10 * std::max(1.0, analytic),
            name + ": matrix-free metric action is not the feature adjoint");
  }
  for (int column = 0; column < u.cols(); ++column) {
    const Eigen::VectorXd recovered =
        space.project_vector(u.col(column)).reduced_gradient;
    require(
        (recovered - Eigen::VectorXd::Unit(u.cols(), column)).norm() < 1e-10,
        name + ": reduced vector coordinates are not invertible");
  }
  Eigen::MatrixXd physical_actions(view.size(), u.cols());
  for (int column = 0; column < u.cols(); ++column) {
    physical_actions.col(column) = coupled_metric.apply(view, u.col(column));
  }
  const Eigen::MatrixXd physical_gram = u.transpose() * physical_actions;
  require(
      (physical_gram - physical_gram.transpose()).norm() < 1.0e-12 &&
      (physical_gram.diagonal().array() > 0.0).all(),
      name + ": coupled quotient physical metric is not positive");
  if (u.cols() > 0) {
    const Eigen::VectorXd first = u.col(0);
    const Eigen::VectorXd last = u.col(u.cols() - 1);
    const double polarization = 0.5 *
        (coupled_metric.squared_norm(view, first + last) -
         coupled_metric.squared_norm(view, first) -
         coupled_metric.squared_norm(view, last));
    require(std::abs(polarization - physical_gram(0, u.cols() - 1)) < 1.0e-10,
            name + ": matrix-free metric action disagrees with polarization");
    if (u.cols() <= 14) {
      const NonredundantRetractionMetric metric(space, view, input);
      const Eigen::VectorXd covector =
          Eigen::VectorXd::LinSpaced(u.cols(), -0.8, 1.3);
      const Eigen::VectorXd riesz_vector = metric.solve(covector);
      const Eigen::VectorXd dense_riesz_vector =
          physical_gram.ldlt().solve(covector);
      require((riesz_vector - dense_riesz_vector).norm() <
                  5.0e-8 * std::max(1.0, dense_riesz_vector.norm()),
              name + ": matrix-free Riesz solve differs from dense metric");
      require((metric.apply(riesz_vector) - covector).norm() <
                  2.0e-8 * covector.norm(),
              name + ": matrix-free Riesz solve has a large true residual");
      const Eigen::VectorXd test_vector =
          Eigen::VectorXd::LinSpaced(u.cols(), 0.9, -0.4);
      require((metric.solve(metric.apply(test_vector)) - test_vector).norm() <
                  5.0e-8 * std::max(1.0, test_vector.norm()),
              name + ": Riesz solve does not recover reduced coordinates");
      require(std::abs(riesz_vector.dot(metric.apply(test_vector)) -
                       covector.dot(test_vector)) <
                  2.0e-8 * std::max(1.0, std::abs(covector.dot(test_vector))),
              name + ": Riesz solve violates metric-covector duality");

      const Eigen::MatrixXd model_hessian = physical_gram +
          Eigen::MatrixXd::Identity(u.cols(), u.cols());
      const Eigen::VectorXd gradient =
          Eigen::VectorXd::LinSpaced(u.cols(), 0.2, 1.1);
      TruncatedNewtonSubspace subspace;
      subspace.orthonormal_basis =
          Eigen::MatrixXd::Identity(u.cols(), u.cols());
      subspace.tangent_basis = subspace.orthonormal_basis;
      subspace.hessian_basis = model_hessian;
      subspace.metric_basis = physical_gram;
      subspace.reduced_hessian = model_hessian;
      subspace.reduced_metric = physical_gram;
      subspace.projected_gradient = gradient;
      OrbitalChart::ProjectionResult projection;
      projection.reduced_gradient = gradient;
      const Eigen::VectorXd exact_step =
          -model_hessian.ldlt().solve(gradient);
      const double exact_norm = metric.norm(exact_step);
      for (double radius : {2.0 * exact_norm, 0.5 * exact_norm}) {
        const auto step = solve_trust_region_in_subspace(
            projection, radius, metric, subspace,
            kFixtureForcing);
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
      const auto affine_step = solve_affine_trust_region_in_subspace(
          projection,
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
               affine_step.trust_region_shift *
                   affine_step.reduced_metric_times_step)
                  .norm() < 1.0e-7 * gradient.norm(),
              name + ": affine correction KKT mismatch");

      const auto zero_defect_step = solve_affine_trust_region_in_subspace(
          projection,
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
  require((fd * audit.packed_gauge_basis).norm() / std::max(1.0, fd.norm())
              < 1e-7,
          name + ": finite physical-map gauge derivative");
  if (u.cols() > 0) {
    const Eigen::MatrixXd image = fd * u;
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(image);
    require(svd.singularValues().tail(1)[0] > 1e-7,
            name + ": finite physical-map rank loss");
    const Eigen::VectorXd d = Eigen::VectorXd::LinSpaced(u.cols(), -0.7, 0.9);
    const Eigen::VectorXd g = Eigen::VectorXd::LinSpaced(view.size(), 0.2, 1.1);
    require(std::abs(g.dot(space.expand_step(d)) -
                     space.project_reduced_gradient(g).dot(d)) < 1e-12,
            name + ": adjoint projection");
    const Eigen::VectorXd reduced_covector =
        Eigen::VectorXd::LinSpaced(u.cols(), -0.4, 0.6);
    require(
        (space.project_reduced_gradient(
             space.expand_gradient(reduced_covector)) -
         reduced_covector)
                .norm() < 1e-10,
        name + ": reduced covector lift is not a right inverse");
    Eigen::VectorXd secant_step =
        Eigen::VectorXd::LinSpaced(u.cols(), 0.3, 1.2).normalized();
    Eigen::VectorXd secant_gradient_change =
        (2.0 * secant_step).eval();
    std::vector<PackedSecantPair> secant_history;
    append_projected_secant_pair(
        space,
        space.expand_step(secant_step),
        space.expand_gradient(secant_gradient_change),
        1,
        &secant_history);
    auto inverse_hessian = build_transported_reduced_lbfgs_preconditioner(
        space, secant_history, 1, LbfgsInitialInverse::ScaledIdentity);
    require(inverse_hessian.size() == 1,
            name + ": valid quotient secant was rejected");
    require((inverse_hessian.apply(secant_gradient_change) - secant_step)
                    .norm() < 1.0e-10,
            name + ": quotient L-BFGS inverse violates its secant equation");
    const auto trial = space.retract_step(input, d, 0.01);
    require((view.pack(trial) - x - 0.01 * u * d).norm() < 1e-12,
            name + ": additive retraction");
    // Stored but frozen slots must not change.
    std::vector<bool> differentiable(input.orbital_value_table.size(), false);
    for (int slot : view.differentiable_parameter_indices()) differentiable[slot] = true;
    for (int slot = 0; slot < static_cast<int>(differentiable.size()); ++slot) {
      require(differentiable[slot] ||
                  trial.orbital_value_table[slot] == input.orbital_value_table[slot],
              name + ": modified fixed coefficient");
    }
  }
  std::cout << name << ": passed (" << view.size() << " -> "
            << space.reduced_size() << ")\n";
}

void check_complete_oeo_active_subspace_gauge() {
  Eigen::MatrixXd coefficients = Eigen::MatrixXd::Zero(5, 3);
  coefficients(0, 0) = 1.0;
  coefficients(1, 1) = 1.0;
  coefficients(2, 2) = 1.0;
  coefficients(3, 1) = 0.2;
  coefficients(4, 2) = -0.3;
  OrbitalPreparationInput input = make_input(
      coefficients,
      {{0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}},
      1);
  input.orbital_type = kOrbitalTypeOeo;
  const SparseParameterLayout view(input);
  const Eigen::MatrixXd orbitals = dense(input);

  const OrbitalChart structure_subset_chart(
      input, view, orbitals, orbitals, nullptr, false, false);
  require(structure_subset_chart.reduced_size() == 10,
          "OEO structure subset did not retain independent active rays");
  const auto subset_diagnostics =
      structure_subset_chart.structural_diagnostics();
  require(subset_diagnostics.product_sphere_orbital_count == 3 &&
              subset_diagnostics.maximum_sphere_tangency_residual < 1.0e-12,
          "OEO structure subset did not use S-tangent orbital rays");

  const OrbitalChart complete_cas_chart(
      input, view, orbitals, orbitals, nullptr, false, true);
  require(complete_cas_chart.reduced_size() == 8,
          "complete-CAS OEO chart did not remove active--active gauge");
  const auto complete_diagnostics =
      complete_cas_chart.structural_diagnostics();
  require(complete_diagnostics.product_sphere_orbital_count == 0,
          "complete-CAS OEO chart was split into independent orbital rays");

  Eigen::VectorXd active_mixing = Eigen::VectorXd::Zero(view.size());
  for (int slot = 0; slot < 5; ++slot) {
    active_mixing[view.packed_index(1, slot)] = coefficients(slot, 2);
  }
  require(complete_cas_chart.project_vector(active_mixing)
              .reduced_gradient.norm() < 1.0e-12,
          "complete-CAS OEO chart retained an active--active gauge direction");
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
    Eigen::MatrixXd u(3, space.reduced_size()), action(2, 2), inverse(2, 2);
    Eigen::MatrixXd shifted_inverse(2, 2);
    require(space.reduced_size() == 2, "invalid occupation fixture rank");
    for (int j = 0; j < 2; ++j) {
      const Eigen::VectorXd unit = Eigen::VectorXd::Unit(2, j);
      u.col(j) = space.expand_step(unit);
      action.col(j) = space.apply_reduced_curvature(unit);
      inverse.col(j) = space.apply_inverse_reduced_block_preconditioner(unit);
      shifted_inverse.col(j) =
          space.apply_inverse_reduced_shifted_block_preconditioner(unit, 0.4);
    }
    // Independent diagonal one-electron gap formula at a stationary ray.
    const double occupation = inactive_count == 1 ? 2.0 : 1.0;
    const Eigen::MatrixXd expected = 2.0 * occupation * u.transpose() *
        (f - f(0, 0) * Eigen::MatrixXd::Identity(3, 3)) * u;
    require((action - expected).norm() < 1e-12,
            "production block lost inactive double occupancy or changed active unit model");
    require((inverse * expected - Eigen::MatrixXd::Identity(2, 2)).norm() < 1e-12,
            "inverse occupation curvature is inconsistent");
    require((shifted_inverse *
                 (expected + 0.4 * Eigen::Matrix2d::Identity()) -
             Eigen::Matrix2d::Identity()).norm() < 1e-12,
            "shifted inverse occupation curvature is inconsistent");
    const std::vector<PackedSecantPair> empty_history;
    const auto standard_lbfgs = build_transported_reduced_lbfgs_preconditioner(
        space, empty_history, 1, LbfgsInitialInverse::ScaledIdentity);
    const auto block_preconditioner =
        build_transported_reduced_lbfgs_preconditioner(
            space, empty_history, 1, LbfgsInitialInverse::OrbitalBlock);
    const Eigen::Vector2d probe(0.3, -0.7);
    require((standard_lbfgs.apply(probe) - probe).norm() < 1.0e-14,
            "standard L-BFGS did not start from the identity inverse");
    require((block_preconditioner.apply(probe) - inverse * probe).norm() <
                1.0e-14,
            "TNHVP L-BFGS did not use the orbital-block inverse");
  }
  std::cout << "Inactive occupation and active unit-model production blocks: passed\n";
}

Eigen::VectorXd normalized_rayleigh_gradient(
    const Eigen::VectorXd& coefficients,
    const Eigen::MatrixXd& overlap,
    const Eigen::MatrixXd& one_electron) {
  const double norm_squared = coefficients.dot(overlap * coefficients);
  const double energy =
      coefficients.dot(one_electron * coefficients) / norm_squared;
  return 2.0 * (one_electron * coefficients -
                energy * overlap * coefficients) /
         norm_squared;
}

void check_pullback_secant_transport() {
  Eigen::MatrixXd coefficients(3, 1);
  coefficients << 0.8, -0.35, 0.55;
  auto input = make_input(coefficients, {{0, 1, 2}}, 0);
  Eigen::Matrix3d overlap;
  overlap << 1.2, 0.08, -0.03,
             0.08, 0.9, 0.04,
             -0.03, 0.04, 1.1;
  input.ao_overlap_matrix = overlap;
  Eigen::Matrix3d one_electron;
  one_electron << -1.4, 0.22, -0.17,
                   0.22, 0.6, 0.31,
                  -0.17, 0.31, 1.3;

  const SparseParameterLayout view(input);
  const OrbitalChart old_space(
      input, view, dense(input), dense(input), nullptr, true);
  require(old_space.reduced_size() == 2,
          "Rayleigh pullback fixture has the wrong quotient dimension");
  const Eigen::Vector2d direction =
      Eigen::Vector2d(0.7, -0.4).normalized();
  const Eigen::VectorXd packed_direction = old_space.expand_step(direction);
  const Eigen::VectorXd x0 = view.pack(input);
  const Eigen::VectorXd gradient0 = normalized_rayleigh_gradient(
      dense(input).col(0), overlap, one_electron);

  constexpr double finite_difference_step = 1.0e-6;
  auto plus = input;
  auto minus = input;
  view.unpack(x0 + finite_difference_step * packed_direction, &plus);
  view.unpack(x0 - finite_difference_step * packed_direction, &minus);
  const Eigen::VectorXd reference_action =
      old_space.project_reduced_gradient(
          normalized_rayleigh_gradient(
              dense(plus).col(0), overlap, one_electron) -
          normalized_rayleigh_gradient(
              dense(minus).col(0), overlap, one_electron)) /
      (2.0 * finite_difference_step);

  constexpr double accepted_step = 2.0e-6;
  const auto next_input =
      old_space.retract_step(input, direction, accepted_step);
  const OrbitalChart next_space(
      next_input, view, dense(next_input), dense(next_input), nullptr, true);
  const Eigen::VectorXd gradient1 = normalized_rayleigh_gradient(
      dense(next_input).col(0), overlap, one_electron);
  const Eigen::VectorXd packed_secant =
      next_space.project_reduced_gradient(gradient1 - gradient0) /
      accepted_step;

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
  std::cout << "polar pullback ambient secant: passed (packed error "
            << packed_error << ", untransported error "
            << untransported_gradient_error << ")\n";
}

void check_ill_conditioned_representative(
    const OrbitalPreparationInput& input,
    int expected_dimension) {
  const SparseParameterLayout view(input);
  const Eigen::MatrixXd orbitals = dense(input);
  const OrbitalChart space(input, view, orbitals, orbitals, nullptr, true);
  require(space.reduced_size() == expected_dimension,
          "ill-conditioned representative changed the quotient dimension");
  const auto diagnostics = space.structural_diagnostics();
  require(diagnostics.total_gauge_rank ==
              view.size() - expected_dimension,
          "ill-conditioned representative changed the stable gauge rank");
  for (int column = 0; column < space.reduced_size(); ++column) {
    const Eigen::VectorXd unit =
        Eigen::VectorXd::Unit(space.reduced_size(), column);
    require((space.project_vector(space.expand_step(unit)).reduced_gradient -
             unit).norm() < 1.0e-9,
            "ill-conditioned quotient coordinates are not invertible");
  }
  std::cout << "ill-conditioned inactive representative: passed ("
            << view.size() << " -> " << space.reduced_size() << ")\n";
}

class DenseTestHvp final : public ReducedHvp {
 public:
  explicit DenseTestHvp(Eigen::MatrixXd hessian)
      : hessian_(std::move(hessian)) {}

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

  Eigen::MatrixXd apply_batch(
      const Eigen::Ref<const Eigen::MatrixXd>& directions) override {
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

  Eigen::MatrixXd apply_frozen_batch(
      const Eigen::Ref<const Eigen::MatrixXd>& directions) override {
    ++frozen_applies;
    frozen_directions.emplace_back(directions);
    return hessian_ * directions;
  }

  std::uint64_t model_revision() const noexcept override { return revision_; }

  /** @brief Simulates response enrichment outside a retained radius trial. */
  void replace_model(Eigen::MatrixXd hessian) {
    hessian_ = std::move(hessian);
    ++revision_;
  }

  const Eigen::MatrixXd& hessian() const noexcept { return hessian_; }

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
void check_cached_newton_model(
    const TruncatedNewtonStepResult& step,
    const ChangingModelHvp& hvp,
    const Eigen::VectorXd& gradient,
    const NonredundantRetractionMetric& metric) {
  const auto& subspace = step.subspace;
  require(subspace.model_revision == hvp.model_revision(),
          "Newton subspace retained an obsolete response-model revision");
  const Eigen::MatrixXd expected_images =
      hvp.hessian() * subspace.orthonormal_basis;
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
      -gradient.dot(step.reduced_step) -
      0.5 * step.reduced_step.dot(expected_hvp);
  require(std::abs(step.predicted_decrease - expected_prediction) <
              1.0e-12 * std::max(1.0, std::abs(expected_prediction)),
          "Newton predicted decrease refers to an obsolete response model");
  const double expected_kkt =
      (gradient + expected_hvp +
       step.trust_region_shift * expected_metric).norm() / gradient.norm();
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
  step.target_kkt_relative_residual =
      kFixtureForcing;
  refresh_truncated_newton_step_certificate(gradient, &step);
  require(step.model_kkt_converged && step.newton_forcing_converged,
          "exact interior Newton model lacks a forcing certificate");

  step.reduced_hessian_times_step = -0.9 * gradient;
  step.target_kkt_relative_residual = 0.2;
  refresh_truncated_newton_step_certificate(gradient, &step);
  require(step.model_kkt_converged,
          "explicit loose KKT target rejected a valid step");
  step.target_kkt_relative_residual = 0.05;
  refresh_truncated_newton_step_certificate(gradient, &step);
  require(!step.model_kkt_converged,
          "explicit tight KKT target accepted an unresolved step");

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
  require(
      minimize_truncated_newton_step_on_ray(ray_projection, &ray_step) &&
      std::abs(ray_step.reduced_step[0] + 1.0 / 3.0) < 1.0e-14 &&
      std::abs(ray_step.predicted_decrease - 1.0 / 6.0) < 1.0e-14 &&
      !ray_step.reached_boundary && ray_step.trust_region_shift == 0.0,
      "exact directional quadratic was not minimized on the sampled ray");

  require(!observed_contraction_requires_newton_correction(2.0, 0.5, 0.25) &&
              observed_contraction_requires_newton_correction(
                  2.0, 0.5000000001, 0.25) &&
              observed_contraction_requires_newton_correction(
                  0.0, 0.0, 0.25),
          "observed outer contraction did not enforce the Newton target");
}

void check_residual_driven_subspace(
    const OrbitalPreparationInput& input) {
  const SparseParameterLayout view(input);
  const Eigen::MatrixXd c = dense(input);
  const OrbitalChart space(input, view, c, c, nullptr, true);
  const NonredundantRetractionMetric metric(space, view, input);
  const int dimension = space.reduced_size();
  require(dimension > 4, "residual-driven fixture is too small");
  Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(dimension, dimension);
  for (int i = 0; i < dimension; ++i) hessian(i, i) = 1.0 + i;
  const Eigen::VectorXd gradient =
      Eigen::VectorXd::LinSpaced(dimension, 1.0e-3, 2.0e-3);
  OrbitalChart::ProjectionResult projection;
  projection.reduced_gradient = gradient;

  DenseTestHvp unrestricted_hvp(hessian);
  const auto unrestricted = solve_nonredundant_truncated_newton_step(
      metric, space, projection, 10.0, 1.0e-12, 1.0e-8,
      1.0e-12, &unrestricted_hvp, nullptr);
  require(unrestricted.subspace_dimension > 4 &&
              unrestricted.subspace_dimension <= dimension &&
              unrestricted.stop_reason ==
                  TruncatedNewtonStopReason::ModelKktConverged &&
              unrestricted.newton_forcing_converged,
          "residual-driven Newton solve stopped before its certificate");

  DenseTestHvp reused_hvp(hessian);
  const auto reused = solve_nonredundant_truncated_newton_step(
      metric, space, projection, 10.0, 1.0e-12, 1.0e-8,
      1.0e-12, &reused_hvp, nullptr, nullptr, &unrestricted.subspace);
  require(reused.model_kkt_converged && reused_hvp.applies == 0,
          "certified same-point subspace recomputed exact HVP samples");

  Eigen::MatrixXd clustered = Eigen::MatrixXd::Zero(dimension, dimension);
  for (int i = 0; i < dimension; ++i) {
    clustered(i, i) = (i % 3 == 0) ? 1.0 : (i % 3 == 1 ? 4.0 : 16.0);
  }
  DenseTestHvp clustered_hvp(clustered);
  const auto certified = solve_nonredundant_truncated_newton_step(
      metric, space, projection, 10.0, 1.0e-7, 1.0e-3,
      kFixtureForcing, &clustered_hvp, nullptr);
  require(certified.subspace_dimension <= 4 &&
              clustered_hvp.applies <= 4 &&
              certified.stop_reason ==
                  TruncatedNewtonStopReason::ModelKktConverged &&
              certified.newton_forcing_converged &&
              certified.model_kkt_relative_residual < kFixtureForcing,
          "clustered Newton subspace did not certify the full-space model");

  hessian(0, 0) = -10.0;
  DenseTestHvp indefinite(hessian);
  const auto boundary = solve_nonredundant_truncated_newton_step(
      metric, space, projection, 1.0e-3, 1.0e-7, 1.0e-3,
      kFixtureForcing, &indefinite, nullptr);
  require(boundary.reached_boundary &&
              boundary.encountered_negative_curvature &&
              !boundary.newton_forcing_converged,
          "negative curvature did not produce a certified boundary model");

  const Eigen::VectorXd exact_predictor =
      -clustered.ldlt().solve(gradient);
  DenseTestHvp certified_predictor_hvp(clustered);
  const auto certified_predictor = solve_nonredundant_truncated_newton_step(
      metric, space, projection, 10.0, 1.0e-7, 1.0e-3,
      kFixtureForcing, &certified_predictor_hvp, nullptr, &exact_predictor);
  require(certified_predictor_hvp.applies == 1 &&
              certified_predictor.subspace_dimension == 0 &&
              certified_predictor.newton_forcing_converged,
          "exact block-L-BFGS predictor was not certified by one HVP");

  Eigen::MatrixXd two_cluster =
      Eigen::MatrixXd::Zero(dimension, dimension);
  for (int i = 0; i < dimension; ++i) {
    two_cluster(i, i) = i % 2 == 0 ? 1.0 : 4.0;
  }
  const Eigen::VectorXd identity_predictor = -gradient;
  DenseTestHvp defect_hvp(two_cluster);
  const auto corrected_predictor = solve_nonredundant_truncated_newton_step(
      metric, space, projection, 10.0, 1.0e-7, 1.0e-3,
      kFixtureForcing, &defect_hvp, nullptr, &identity_predictor);
  require(defect_hvp.applies == 3 &&
              corrected_predictor.subspace_dimension == 2 &&
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
  for (int i = 0; i < dimension; ++i) initial(i, i) = 1.0 + i;
  const Eigen::VectorXd coupling =
      Eigen::VectorXd::LinSpaced(dimension, 0.25, 1.0).normalized();
  const Eigen::MatrixXd enriched =
      1.3 * initial + 0.2 * coupling * coupling.transpose();
  ChangingModelHvp hvp(initial, enriched);
  const Eigen::VectorXd gradient =
      Eigen::VectorXd::LinSpaced(dimension, 1.0e-3, 2.0e-3);
  OrbitalChart::ProjectionResult projection;
  projection.reduced_gradient = gradient;
  constexpr double forcing = 1.0e-10;

  const auto step = solve_nonredundant_truncated_newton_step(
      metric, space, projection, 10.0, 1.0e-16, 1.0e-12,
      forcing, &hvp, nullptr);
  require(hvp.model_revision() == 2 && hvp.batch_applies >= 2 &&
              hvp.frozen_applies > 0 && step.newton_forcing_converged,
          "Newton solve did not refresh its evolving response model");
  require(hvp.prepared_directions.front().cols() == 1 &&
              hvp.frozen_directions.front().cols() == 1 &&
              (hvp.frozen_directions.front().col(0) -
               hvp.prepared_directions.front().col(0)).norm() < 1.0e-14,
          "response enrichment did not refresh the stale baseline image");
  require(((enriched - initial) *
           hvp.prepared_directions.front().col(0)).norm() > 1.0e-2,
          "refresh fixture did not change the original Hessian sample");
  check_cached_newton_model(step, hvp, gradient, metric);

  const int prepared_before_reuse = hvp.batch_applies;
  const int frozen_before_reuse = hvp.frozen_applies;
  const auto reused = solve_nonredundant_truncated_newton_step(
      metric, space, projection, 5.0, 1.0e-16, 1.0e-12,
      forcing, &hvp, nullptr, nullptr, &step.subspace);
  require(reused.newton_forcing_converged &&
              hvp.batch_applies == prepared_before_reuse &&
              hvp.frozen_applies == frozen_before_reuse,
          "unchanged response-model reuse unnecessarily recomputed HVPs");
  check_cached_newton_model(reused, hvp, gradient, metric);

  hvp.replace_model(enriched +
                    2.0 * Eigen::MatrixXd::Identity(dimension, dimension));
  const Eigen::VectorXd exact_retry_step =
      -hvp.hessian().ldlt().solve(gradient);
  const double retry_radius = 0.4 * metric.norm(exact_retry_step);
  const auto retry = solve_nonredundant_truncated_newton_step(
      metric, space, projection, retry_radius, 1.0e-16, 1.0e-12,
      forcing, &hvp, nullptr, nullptr, &step.subspace);
  require(hvp.frozen_applies > frozen_before_reuse &&
              retry.reached_boundary && retry.trust_region_shift > 0.0,
          "radius retry did not invalidate a stale response-model cache");
  check_cached_newton_model(retry, hvp, gradient, metric);
  require(metric.norm(retry.reduced_step) <= retry_radius * (1.0 + 1.0e-8),
          "refreshed Newton step exceeded its revised trust radius");
  std::cout << "response-model image refresh and radius reuse: passed\n";
}
}  // namespace

int main() {
  try {
    check_response_spectrum_summary();
    check_accuracy_aware_forcing();
    check_occupation_curvature();
    check_pullback_secant_transport();
    Eigen::MatrixXd c(6, 4);
    c << 1, 0, 0.3, 0,
         0, 1, 0.4, 0,
         1, 1, 0, 0.2,
         1, 1, 0, 0.3,
         0, 0, 1, 0,
         0, 0, 0, 1;
    auto partial = make_input(c, {{0,2,3}, {1,2,3}, {0,1,4}, {2,3,5}}, 2);
    check("unequal supports with off-support cancellation", partial, 7);
    auto frozen = partial;
    frozen.original_orbital_basis_counts = {3,3,2,3};
    check("stored but frozen active tail", frozen, 7);
    auto full = make_input(c, {{0,1,2,3,4,5}, {0,1,2,3,4,5},
                               {0,1,2,3,4,5}, {0,1,2,3,4,5}}, 2);
    check("full support", full, 14);
    check_complete_oeo_active_subspace_gauge();
    check_truncated_newton_certificates();
    check_residual_driven_subspace(full);
    check_response_model_refresh(full);
    Eigen::MatrixXd rotated = c;
    Eigen::Matrix2d a;
    a << 1, 0.3, -0.2, 1.1;
    rotated.leftCols(2) = (c.leftCols(2) * a).eval();
    check("inactive gauge rotation", make_input(rotated,
          {{0,1,2,3,4,5}, {0,1,2,3,4,5},
           {0,1,2,3,4,5}, {0,1,2,3,4,5}}, 2), 14);
    Eigen::MatrixXd ill_conditioned = c;
    Eigen::Matrix2d ill_conditioned_gauge;
    ill_conditioned_gauge << 1.0, 1.0, 0.0, 1.0e-4;
    ill_conditioned.leftCols(2) =
        (c.leftCols(2) * ill_conditioned_gauge).eval();
    check_ill_conditioned_representative(make_input(
        ill_conditioned,
        {{0,1,2,3,4,5}, {0,1,2,3,4,5},
         {0,1,2,3,4,5}, {0,1,2,3,4,5}}, 2), 14);
    const Eigen::MatrixXd active = c.rightCols(2);
    check("no inactive orbitals", make_input(active, {{0,1,4}, {2,3,5}}, 0), 4);
    check("inactive only", make_input(c.leftCols(2), {{0,2,3}, {1,2,3}}, 2), 4);
    Eigen::MatrixXd single = Eigen::MatrixXd::Identity(2, 2);
    check("all coordinates are scale gauge", make_input(single, {{0}, {1}}, 0), 0);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
