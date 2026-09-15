#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/SVD>

#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/physical_metric.hpp"
#include "vbscf/optimization/trust_region/retraction.hpp"
#include "vbscf/optimization/trust_region/truncated_newton.hpp"
#include "vbscf/diagnostics/orbitals/chart_audit.hpp"

namespace {
using namespace xmvb::vb;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
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
            projection, radius, metric, subspace);
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
    }
  }
  require((audit.packed_gauge_basis.transpose() * u).norm() < 1e-11,
          name + ": nonzero gauge overlap");

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
    require(space.reduced_size() == 2, "invalid occupation fixture rank");
    for (int j = 0; j < 2; ++j) {
      const Eigen::VectorXd unit = Eigen::VectorXd::Unit(2, j);
      u.col(j) = space.expand_step(unit);
      action.col(j) = space.apply_reduced_curvature(unit);
      inverse.col(j) = space.apply_inverse_reduced_block_preconditioner(unit);
    }
    // Independent diagonal one-electron gap formula at a stationary ray.
    const double occupation = inactive_count == 1 ? 2.0 : 1.0;
    const Eigen::MatrixXd expected = 2.0 * occupation * u.transpose() *
        (f - f(0, 0) * Eigen::MatrixXd::Identity(3, 3)) * u;
    require((action - expected).norm() < 1e-12,
            "production block lost inactive double occupancy or changed active unit model");
    require((inverse * expected - Eigen::MatrixXd::Identity(2, 2)).norm() < 1e-12,
            "inverse occupation curvature is inconsistent");
  }
  std::cout << "Inactive occupation and active unit-model production blocks: passed\n";
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
}  // namespace

int main() {
  try {
    check_occupation_curvature();
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
#include <algorithm>
