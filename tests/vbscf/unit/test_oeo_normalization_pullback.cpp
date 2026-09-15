#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/orbital/preparation.hpp"
#include "vbscf/orbitals/charts/layout.hpp"
#include "vbscf/orbitals/preparation/preparer.hpp"
#include "vbscf/orbitals/pullback/operator.hpp"

namespace {

using namespace xmvb::vb;

Eigen::VectorXd fixed_upstream_raw_gradient(
    const OrbitalPreparationInput& input,
    const Eigen::Ref<const Eigen::MatrixXd>& active_gradient,
    const std::vector<double>& inactive_gradient) {
  const auto prepared = ActiveSpaceOrbitalPreparer().prepare(input);
  const auto cache = build_accepted_orbital_preparation_cache(
      input, active_gradient, inactive_gradient);
  const Eigen::Map<const Eigen::MatrixXd> inactive_gradient_matrix(
      inactive_gradient.data(),
      input.n_basis_functions,
      input.n_basis_functions);
  const std::vector<double> raw = backpropagate_active_space_orbital_gradient(
      active_gradient, inactive_gradient_matrix, input, prepared, cache);
  return Eigen::Map<const Eigen::VectorXd>(raw.data(), raw.size());
}

void check_fixed_upstream_pullback_curvature(
    const OrbitalPreparationInput& input,
    const Eigen::Ref<const Eigen::MatrixXd>& active_gradient,
    const std::vector<double>& inactive_gradient) {
  const SparseParameterLayout layout(input);
  Eigen::VectorXd packed_direction =
      Eigen::VectorXd::LinSpaced(layout.size(), -0.8, 1.1);
  packed_direction.normalize();

  Eigen::VectorXd raw_direction = Eigen::VectorXd::Zero(
      static_cast<Eigen::Index>(input.orbital_value_table.size()));
  const auto& differentiable = layout.differentiable_parameter_indices();
  for (Eigen::Index index = 0; index < packed_direction.size(); ++index) {
    raw_direction[differentiable[index]] = packed_direction[index];
  }

  const auto cache = build_accepted_orbital_preparation_cache(
      input, active_gradient, inactive_gradient);
  const auto tangent = build_dense_orbital_tangent_context(
      input, layout, packed_direction, cache);
  const int n_inactive =
      (input.n_total_electrons - input.n_active_electrons) / 2;
  const auto preparation_direction = build_orbital_preparation_directional_result(
      input, tangent, n_inactive, input.n_active_orbitals, cache);
  const std::vector<double> analytic_raw =
      apply_fixed_upstream_orbital_pullback_direction(
          input,
          tangent,
          active_gradient,
          preparation_direction.basis_overlap_times_delta_active_orbitals,
          inactive_gradient,
          raw_direction,
          cache);
  const Eigen::Map<const Eigen::VectorXd> analytic(
      analytic_raw.data(), analytic_raw.size());

  constexpr double step = 1.0e-5;
  OrbitalPreparationInput plus = input;
  OrbitalPreparationInput minus = input;
  layout.unpack(layout.pack(input) + step * packed_direction, &plus);
  layout.unpack(layout.pack(input) - step * packed_direction, &minus);
  const Eigen::VectorXd finite_difference =
      (fixed_upstream_raw_gradient(plus, active_gradient, inactive_gradient) -
       fixed_upstream_raw_gradient(minus, active_gradient, inactive_gradient)) /
      (2.0 * step);
  const double relative_error =
      (analytic - finite_difference).norm() /
      std::max(1.0, finite_difference.norm());
  if (relative_error > 2.0e-8) {
    throw std::runtime_error(
        "orbital-preparation second-order pullback fails gradient differences");
  }
}

}  // namespace

int main() {
  try {
    OrbitalPreparationInput input;
    input.n_basis_functions = 5;
    input.n_orbitals = 3;
    input.n_active_orbitals = 2;
    input.n_active_electrons = 2;
    input.n_total_electrons = 4;
    input.orbital_type = 3;
    input.ao_overlap_matrix = Eigen::MatrixXd::Identity(5, 5);
    input.ao_overlap_matrix.array() += 0.07;
    input.orbital_basis_counts = {5, 5, 5};
    input.orbital_value_table = {
        1.4, 0.2, -0.1, 0.3, 0.1,
        0.1, 0.8, 0.3, -0.2, 0.1,
        -0.2, 0.1, 0.1, 1.3, 0.4};
    input.orbital_basis_index_table = {
        1, 2, 3, 4, 5,
        1, 2, 3, 4, 5,
        1, 2, 3, 4, 5};
    const Eigen::MatrixXd active = Eigen::MatrixXd::Random(5, 2);
    const Eigen::MatrixXd density = Eigen::MatrixXd::Random(5, 5);
    ActiveSpaceOrbitalPreparer prepare;
    ActiveSpaceOrbitalBackpropagator backward;
    auto energy = [&](const OrbitalPreparationInput& x) {
      const auto result = prepare.prepare(x);
      return (active.array() *
              result.auxiliary_orbital_matrix.middleCols(1, 2).array())
                 .sum() +
          (density.array() * result.inactive_density_matrix.array()).sum();
    };
    const auto result = prepare.prepare(input);
    const auto gradient =
        backward.backpropagate(active, density, input, result)
            .orbital_value_gradient;
    constexpr double step = 1e-5;
    for (int j = 0; j < 15; ++j) {
      auto plus = input;
      auto minus = input;
      plus.orbital_value_table[j] += step;
      minus.orbital_value_table[j] -= step;
      const double finite_difference =
          (energy(plus) - energy(minus)) / (2.0 * step);
      if (std::abs(finite_difference - gradient[j]) > 1e-8) {
        throw std::runtime_error(
            "OEO raw-coefficient gradient fails energy difference");
      }
    }
    auto generic = input;
    // Identical explicit full-AO support, not a different orbital manifold.
    generic.orbital_type = 1;
    const auto reference =
        backward.backpropagate(
            active, density, generic, prepare.prepare(generic))
            .orbital_value_gradient;
    for (int j = 0; j < 15; ++j) {
      if (std::abs(gradient[j] - reference[j]) > 1e-13) {
        throw std::runtime_error(
            "full-AO pullback depends on the orbital-type label");
      }
    }
    for (int p = 0; p < 3; ++p) {
      double radial = 0.0;
      for (int j = 0; j < 5; ++j) {
        radial +=
            input.orbital_value_table[5 * p + j] * gradient[5 * p + j];
      }
      if (std::abs(radial) > 1e-12) {
        throw std::runtime_error(
            "normalization scale gauge not annihilated");
      }
    }
    const std::vector<double> density_gradient(
        density.data(), density.data() + density.size());
    check_fixed_upstream_pullback_curvature(input, active, density_gradient);

    OrbitalPreparationInput sparse = input;
    sparse.orbital_type = 1;
    sparse.orbital_basis_counts = {3, 4, 3};
    sparse.orbital_basis_index_table = {1, 2, 4, 0, 0,
                                        1, 2, 3, 5, 0,
                                        2, 4, 5, 0, 0};
    sparse.orbital_value_table = {1.4, 0.2, 0.3, 0.0, 0.0,
                                  0.1, 0.8, 0.3, 0.1, 0.0,
                                  0.1, 1.3, 0.4, 0.0, 0.0};
    check_fixed_upstream_pullback_curvature(sparse, active, density_gradient);

    std::cout
        << "HAO/OEO normalization, projector pullback, and pullback curvature: passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
