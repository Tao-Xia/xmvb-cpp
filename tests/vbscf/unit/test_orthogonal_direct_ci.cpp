#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"
#include "vbscf/structures/orthogonal_ci/integrals.hpp"
#include "vbscf/structures/orthogonal_ci/planner.hpp"
#include "vbscf/structures/orthogonal_ci/sigma.hpp"
#include "vbscf/determinants/pairs/evaluator.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/assembly/hamiltonian_overlap.hpp"

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void append_combinations(
    int n_orbitals,
    int n_electrons,
    int first,
    std::vector<int>* occupied,
    std::vector<std::vector<int>>* determinants) {
  if (static_cast<int>(occupied->size()) == n_electrons) {
    determinants->push_back(*occupied);
    return;
  }
  const int remaining = n_electrons - static_cast<int>(occupied->size());
  for (int orbital = first;
       orbital <= n_orbitals - remaining;
       ++orbital) {
    occupied->push_back(orbital);
    append_combinations(
        n_orbitals,
        n_electrons,
        orbital + 1,
        occupied,
        determinants);
    occupied->pop_back();
  }
}

std::vector<std::vector<int>> complete_space(
    int n_orbitals,
    int n_electrons) {
  std::vector<std::vector<int>> determinants;
  std::vector<int> occupied;
  append_combinations(
      n_orbitals,
      n_electrons,
      0,
      &occupied,
      &determinants);
  return determinants;
}

Eigen::MatrixXd explicit_exterior(
    const std::vector<std::vector<int>>& determinants,
    const Eigen::MatrixXd& orbital_transform) {
  const int dimension = static_cast<int>(determinants.size());
  const int n_electrons = static_cast<int>(determinants.front().size());
  Eigen::MatrixXd exterior(dimension, dimension);
  for (int target = 0; target < dimension; ++target) {
    for (int source = 0; source < dimension; ++source) {
      Eigen::MatrixXd minor(n_electrons, n_electrons);
      for (int row = 0; row < n_electrons; ++row) {
        for (int column = 0; column < n_electrons; ++column) {
          minor(row, column) = orbital_transform(
              determinants[target][row],
              determinants[source][column]);
        }
      }
      exterior(target, source) = minor.determinant();
    }
  }
  return exterior;
}

void check_exterior_transform() {
  const auto determinants = complete_space(6, 3);
  Eigen::MatrixXd orbital = Eigen::MatrixXd::Zero(6, 6);
  for (int row = 0; row < 6; ++row) {
    orbital(row, row) = 0.8 + 0.07 * static_cast<double>(row + 1);
    for (int column = row + 1; column < 6; ++column) {
      orbital(row, column) =
          0.03 * std::sin(static_cast<double>((row + 2) * (column + 1)));
    }
  }
  const Eigen::MatrixXd exterior = explicit_exterior(determinants, orbital);
  const xmvb::vb::ExteriorOrbitalTransform transform(determinants, orbital);

  Eigen::MatrixXd coefficients(exterior.rows(), 7);
  for (int column = 0; column < coefficients.cols(); ++column) {
    for (int row = 0; row < coefficients.rows(); ++row) {
      coefficients(row, column) =
          std::sin(0.17 * static_cast<double>((row + 1) * (column + 2)));
    }
  }
  Eigen::MatrixXd left = coefficients;
  transform.apply_left(&left);
  require(
      (left - exterior * coefficients).cwiseAbs().maxCoeff() < 2.0e-13,
      "left exterior action differs from explicit compound matrix");

  Eigen::MatrixXd right(5, exterior.rows());
  for (int column = 0; column < right.cols(); ++column) {
    for (int row = 0; row < right.rows(); ++row) {
      right(row, column) =
          std::cos(0.11 * static_cast<double>((row + 3) * (column + 1)));
    }
  }
  const Eigen::MatrixXd original_right = right;
  transform.apply_right(&right);
  require(
      (right - original_right * exterior.transpose())
              .cwiseAbs()
              .maxCoeff() < 2.0e-13,
      "right exterior action differs from explicit compound matrix");

  Eigen::MatrixXd block_storage = Eigen::MatrixXd::Zero(
      original_right.rows(), original_right.cols() + 4);
  block_storage.middleCols(2, original_right.cols()) = original_right;
  auto right_block =
      block_storage.middleCols(2, original_right.cols());
  transform.apply_right_block(right_block);
  require(
      (right_block - right).cwiseAbs().maxCoeff() < 2.0e-13 &&
          block_storage.leftCols(2).isZero(0.0) &&
          block_storage.rightCols(2).isZero(0.0),
      "right exterior block action differs from the owning-matrix path");

  transform.apply_adjoint_right(&right);
  require(
      (right - original_right * exterior.transpose() * exterior)
              .cwiseAbs()
              .maxCoeff() < 4.0e-13,
      "right exterior adjoint is inconsistent");
  transform.apply_adjoint_right_block(right_block);
  require(
      (right_block - right).cwiseAbs().maxCoeff() < 4.0e-13 &&
          block_storage.leftCols(2).isZero(0.0) &&
          block_storage.rightCols(2).isZero(0.0),
      "right exterior block adjoint differs from the owning-matrix path");

  Eigen::MatrixXd adjoint = coefficients;
  transform.apply_adjoint_left(&adjoint);
  require(
      (adjoint - exterior.transpose() * coefficients)
              .cwiseAbs()
              .maxCoeff() < 2.0e-13,
      "left exterior adjoint differs from explicit compound matrix");
  require(transform.shear_pair_count() > 0, "test transform has no shears");
}

void check_planner() {
  const auto cerras_alpha = complete_space(11, 5);
  const auto cerras_beta = complete_space(11, 6);
  const auto cerras = xmvb::vb::plan_orthogonal_direct_ci_action(
      cerras_alpha,
      cerras_beta,
      11,
      1);
  require(cerras.complete(), "CERRAS-size fixed-spin space was not recognized");
  require(cerras.n_alpha_strings == 462 && cerras.n_beta_strings == 462,
          "CERRAS-size determinant count is wrong");
  require(cerras.alpha_same_spin_connections == 181 &&
              cerras.beta_same_spin_connections == 181 &&
              cerras.opposite_spin_connections == 900,
          "CERRAS-size connection count is wrong");
  require(cerras.favors_direct_ci(), "CERRAS-size FLOP plan rejected direct CI");

  const auto loflea_space = complete_space(12, 6);
  const auto loflea = xmvb::vb::plan_orthogonal_direct_ci_action(
      loflea_space,
      loflea_space,
      12,
      1);
  require(loflea.complete(), "LOFLEA-size fixed-spin space was not recognized");
  require(loflea.n_alpha_strings == 924 && loflea.n_beta_strings == 924,
          "LOFLEA-size determinant count is wrong");
  require(loflea.alpha_same_spin_connections == 262 &&
              loflea.opposite_spin_connections == 1296,
          "LOFLEA-size connection count is wrong");
  require(loflea.favors_direct_ci(), "LOFLEA-size FLOP plan rejected direct CI");

  auto incomplete = cerras_alpha;
  incomplete.pop_back();
  const auto rejected = xmvb::vb::plan_orthogonal_direct_ci_action(
      incomplete,
      cerras_beta,
      11,
      1);
  require(!rejected.complete() && !rejected.favors_direct_ci(),
          "incomplete fixed-spin space was admitted");

}

void check_sigma_action() {
  constexpr int n_orbitals = 4;
  const auto alpha = complete_space(n_orbitals, 2);
  const auto beta = complete_space(n_orbitals, 2);
  Eigen::MatrixXd orbital = Eigen::MatrixXd::Zero(n_orbitals, n_orbitals);
  orbital <<
      1.10, 0.08, -0.03, 0.05,
      0.00, 0.91,  0.06, 0.02,
      0.00, 0.00,  1.07, 0.04,
      0.00, 0.00,  0.00, 0.96;
  const Eigen::MatrixXd overlap = orbital.transpose() * orbital;
  std::vector<double> packed_overlap(
      overlap.data(), overlap.data() + overlap.size());

  Eigen::MatrixXd one_electron(n_orbitals, n_orbitals);
  for (int column = 0; column < n_orbitals; ++column) {
    for (int row = 0; row <= column; ++row) {
      const double value =
          0.13 * std::cos(static_cast<double>((row + 2) * (column + 1)));
      one_electron(row, column) = value;
      one_electron(column, row) = value;
    }
  }
  const int n_pairs = xmvb::vb::packed_active_pair_count(n_orbitals);
  Eigen::MatrixXd pair_factor(7, n_pairs);
  for (int column = 0; column < n_pairs; ++column) {
    for (int row = 0; row < pair_factor.rows(); ++row) {
      pair_factor(row, column) =
          0.07 * std::sin(static_cast<double>((row + 1) * (column + 3)));
    }
  }
  const Eigen::MatrixXd pair_kernel = pair_factor.transpose() * pair_factor;
  xmvb::vb::ActiveSpaceTwoElectronResult two_electron;
  two_electron.packed_active_two_electron_integrals.resize(
      xmvb::vb::packed_active_two_electron_integral_count(n_orbitals));
  for (int column = 0; column < n_pairs; ++column) {
    for (int row = 0; row <= column; ++row) {
      two_electron.packed_active_two_electron_integrals[
          xmvb::vb::TwoElectronIndexer::packed_pair_of_pairs_index(
              row, column)] = pair_kernel(row, column);
    }
  }

  const auto orthogonal = xmvb::vb::orthogonalize_active_integrals(
      packed_overlap,
      one_electron,
      two_electron,
      n_orbitals);

  Eigen::MatrixXd overlap_direction(n_orbitals, n_orbitals);
  Eigen::MatrixXd one_electron_direction(n_orbitals, n_orbitals);
  for (int column = 0; column < n_orbitals; ++column) {
    for (int row = 0; row <= column; ++row) {
      const double overlap_value =
          0.02 * std::sin(static_cast<double>((row + 1) * (column + 3)));
      const double one_electron_value =
          0.03 * std::cos(static_cast<double>((row + 2) * (column + 4)));
      overlap_direction(row, column) = overlap_value;
      overlap_direction(column, row) = overlap_value;
      one_electron_direction(row, column) = one_electron_value;
      one_electron_direction(column, row) = one_electron_value;
    }
  }
  std::vector<double> packed_overlap_direction(
      overlap_direction.data(),
      overlap_direction.data() + overlap_direction.size());
  std::vector<double> packed_two_electron_direction(
      two_electron.packed_active_two_electron_integrals.size());
  for (std::size_t index = 0;
       index < packed_two_electron_direction.size();
       ++index) {
    packed_two_electron_direction[index] =
        0.01 * std::sin(0.13 * static_cast<double>(index + 1));
  }
  const auto orthogonal_direction =
      xmvb::vb::orthogonalize_active_integral_direction(
          orthogonal,
          packed_overlap_direction,
          one_electron_direction,
          packed_two_electron_direction,
          n_orbitals);
  constexpr double direction_step = 1.0e-6;
  std::vector<double> plus_overlap = packed_overlap;
  std::vector<double> minus_overlap = packed_overlap;
  auto plus_two_electron = two_electron;
  auto minus_two_electron = two_electron;
  for (std::size_t index = 0; index < plus_overlap.size(); ++index) {
    plus_overlap[index] += direction_step * packed_overlap_direction[index];
    minus_overlap[index] -= direction_step * packed_overlap_direction[index];
  }
  for (std::size_t index = 0;
       index < packed_two_electron_direction.size();
       ++index) {
    plus_two_electron.packed_active_two_electron_integrals[index] +=
        direction_step * packed_two_electron_direction[index];
    minus_two_electron.packed_active_two_electron_integrals[index] -=
        direction_step * packed_two_electron_direction[index];
  }
  const auto plus_orthogonal = xmvb::vb::orthogonalize_active_integrals(
      plus_overlap,
      one_electron + direction_step * one_electron_direction,
      plus_two_electron,
      n_orbitals);
  const auto minus_orthogonal = xmvb::vb::orthogonalize_active_integrals(
      minus_overlap,
      one_electron - direction_step * one_electron_direction,
      minus_two_electron,
      n_orbitals);
  require(
      (((plus_orthogonal.orbital_transform -
         minus_orthogonal.orbital_transform) /
        (2.0 * direction_step)) -
       orthogonal_direction.orbital_transform).cwiseAbs().maxCoeff() < 2.0e-9,
      "orthogonal orbital-transform direction fails finite differences");
  require(
      (((plus_orthogonal.one_electron - minus_orthogonal.one_electron) /
        (2.0 * direction_step)) -
       orthogonal_direction.one_electron).cwiseAbs().maxCoeff() < 2.0e-9,
      "orthogonal one-electron direction fails finite differences");
  require(
      (((plus_orthogonal.pair_kernel - minus_orthogonal.pair_kernel) /
        (2.0 * direction_step)) -
       orthogonal_direction.pair_kernel).cwiseAbs().maxCoeff() < 2.0e-9,
      "orthogonal two-electron direction fails finite differences");

  const xmvb::vb::ExteriorOrbitalTransform alpha_transform(
      alpha, orthogonal.orbital_transform);
  const xmvb::vb::ExteriorOrbitalTransform beta_transform(
      beta, orthogonal.orbital_transform);
  const xmvb::vb::DirectCiSigmaAction sigma_action(
      alpha, beta, orthogonal);
  const xmvb::vb::DirectCiSigmaAction sigma_direction(
      alpha, beta, orthogonal_direction);

  constexpr int block_width = 2;
  Eigen::MatrixXd coefficients(
      static_cast<int>(alpha.size()),
      block_width * static_cast<int>(beta.size()));
  for (int column = 0; column < coefficients.cols(); ++column) {
    for (int row = 0; row < coefficients.rows(); ++row) {
      coefficients(row, column) =
          std::sin(0.19 * static_cast<double>((row + 1) * (column + 2)));
    }
  }

  Eigen::MatrixXd left_coefficients(coefficients.rows(), coefficients.cols());
  for (int column = 0; column < left_coefficients.cols(); ++column) {
    for (int row = 0; row < left_coefficients.rows(); ++row) {
      left_coefficients(row, column) =
          std::cos(0.23 * static_cast<double>((row + 2) * (column + 1)));
    }
  }
  const xmvb::vb::DirectCiIntegralAdjoint integral_adjoint =
      sigma_action.integral_adjoint(left_coefficients, coefficients);
  const xmvb::vb::DirectCiIntegralAdjoint reverse_integral_adjoint =
      sigma_action.integral_adjoint(coefficients, left_coefficients);
  require(
      (integral_adjoint.one_electron -
       reverse_integral_adjoint.one_electron).cwiseAbs().maxCoeff() < 2.0e-12,
      "direct-CI one-electron adjoint violates bilinear reciprocity");
  require(
      (integral_adjoint.pair_kernel -
       reverse_integral_adjoint.pair_kernel).cwiseAbs().maxCoeff() < 2.0e-12,
      "direct-CI pair-kernel adjoint violates bilinear reciprocity");
  const double predicted_integral_direction =
      (integral_adjoint.one_electron.cwiseProduct(
           orthogonal_direction.one_electron)).sum() +
      (integral_adjoint.pair_kernel.cwiseProduct(
           orthogonal_direction.pair_kernel)).sum();
  const double applied_integral_direction =
      (left_coefficients.cwiseProduct(
           sigma_direction.apply(coefficients))).sum();
  require(
      std::abs(predicted_integral_direction - applied_integral_direction) <
          3.0e-10 * std::max(1.0, std::abs(applied_integral_direction)),
      "direct-CI integral adjoint fails the bilinear directional identity");
  const auto pack_pair_kernel = [n_orbitals](
      const Eigen::MatrixXd& kernel,
      xmvb::vb::ActiveSpaceTwoElectronResult* result) {
    result->packed_active_two_electron_integrals.resize(
        xmvb::vb::packed_active_two_electron_integral_count(n_orbitals));
    for (int column = 0; column < kernel.cols(); ++column) {
      for (int row = 0; row <= column; ++row) {
        result->packed_active_two_electron_integrals[
            xmvb::vb::TwoElectronIndexer::packed_pair_of_pairs_index(
                row, column)] = kernel(row, column);
      }
    }
  };
  constexpr double adjoint_step = 1.0e-6;
  auto plus_sigma_integrals = orthogonal;
  auto minus_sigma_integrals = orthogonal;
  plus_sigma_integrals.one_electron +=
      adjoint_step * orthogonal_direction.one_electron;
  minus_sigma_integrals.one_electron -=
      adjoint_step * orthogonal_direction.one_electron;
  plus_sigma_integrals.pair_kernel +=
      adjoint_step * orthogonal_direction.pair_kernel;
  minus_sigma_integrals.pair_kernel -=
      adjoint_step * orthogonal_direction.pair_kernel;
  pack_pair_kernel(
      plus_sigma_integrals.pair_kernel,
      &plus_sigma_integrals.two_electron);
  pack_pair_kernel(
      minus_sigma_integrals.pair_kernel,
      &minus_sigma_integrals.two_electron);
  const double plus_bilinear =
      (left_coefficients.cwiseProduct(
           xmvb::vb::DirectCiSigmaAction(
               alpha, beta, plus_sigma_integrals).apply(coefficients))).sum();
  const double minus_bilinear =
      (left_coefficients.cwiseProduct(
           xmvb::vb::DirectCiSigmaAction(
               alpha, beta, minus_sigma_integrals).apply(coefficients))).sum();
  const double finite_difference_bilinear =
      (plus_bilinear - minus_bilinear) / (2.0 * adjoint_step);
  require(
      std::abs(predicted_integral_direction - finite_difference_bilinear) <
          2.0e-8 * std::max(1.0, std::abs(finite_difference_bilinear)),
      "direct-CI integral adjoint fails central finite differences");

  const auto alpha_transform_direction = alpha_transform.direction(
      orthogonal_direction.orbital_transform);
  Eigen::MatrixXd transformed_coefficients = coefficients;
  Eigen::MatrixXd transformed_direction = Eigen::MatrixXd::Zero(
      coefficients.rows(), coefficients.cols());
  alpha_transform.apply_directional_left(
      alpha_transform_direction,
      &transformed_coefficients,
      &transformed_direction);
  Eigen::MatrixXd plus_transformed = coefficients;
  Eigen::MatrixXd minus_transformed = coefficients;
  xmvb::vb::ExteriorOrbitalTransform(
      alpha, plus_orthogonal.orbital_transform).apply_left(&plus_transformed);
  xmvb::vb::ExteriorOrbitalTransform(
      alpha, minus_orthogonal.orbital_transform).apply_left(&minus_transformed);
  require(
      (((plus_transformed - minus_transformed) /
        (2.0 * direction_step)) -
       transformed_direction).cwiseAbs().maxCoeff() < 2.0e-9,
      "exterior-transform direction fails finite differences");

  Eigen::MatrixXd right_probe(5, static_cast<int>(alpha.size()));
  for (int column = 0; column < right_probe.cols(); ++column) {
    for (int row = 0; row < right_probe.rows(); ++row) {
      right_probe(row, column) =
          std::cos(0.09 * static_cast<double>((row + 2) * (column + 1)));
    }
  }
  Eigen::MatrixXd directional_right = right_probe;
  Eigen::MatrixXd delta_directional_right = Eigen::MatrixXd::Zero(
      right_probe.rows(), right_probe.cols());
  alpha_transform.apply_directional_right(
      alpha_transform_direction,
      &directional_right,
      &delta_directional_right);
  Eigen::MatrixXd plus_right = right_probe;
  Eigen::MatrixXd minus_right = right_probe;
  xmvb::vb::ExteriorOrbitalTransform(
      alpha, plus_orthogonal.orbital_transform).apply_right(&plus_right);
  xmvb::vb::ExteriorOrbitalTransform(
      alpha, minus_orthogonal.orbital_transform).apply_right(&minus_right);
  require(
      (((plus_right - minus_right) / (2.0 * direction_step)) -
       delta_directional_right).cwiseAbs().maxCoeff() < 2.0e-9,
      "right exterior-transform direction fails finite differences");

  Eigen::MatrixXd directional_adjoint_left = coefficients;
  Eigen::MatrixXd delta_directional_adjoint_left = Eigen::MatrixXd::Zero(
      coefficients.rows(), coefficients.cols());
  alpha_transform.apply_directional_adjoint_left(
      alpha_transform_direction,
      &directional_adjoint_left,
      &delta_directional_adjoint_left);
  Eigen::MatrixXd plus_adjoint_left = coefficients;
  Eigen::MatrixXd minus_adjoint_left = coefficients;
  xmvb::vb::ExteriorOrbitalTransform(
      alpha,
      plus_orthogonal.orbital_transform).apply_adjoint_left(
          &plus_adjoint_left);
  xmvb::vb::ExteriorOrbitalTransform(
      alpha,
      minus_orthogonal.orbital_transform).apply_adjoint_left(
          &minus_adjoint_left);
  require(
      (((plus_adjoint_left - minus_adjoint_left) /
        (2.0 * direction_step)) -
       delta_directional_adjoint_left).cwiseAbs().maxCoeff() < 2.0e-9,
      "adjoint-left exterior direction fails finite differences");

  Eigen::MatrixXd directional_adjoint_right = right_probe;
  Eigen::MatrixXd delta_directional_adjoint_right = Eigen::MatrixXd::Zero(
      right_probe.rows(), right_probe.cols());
  alpha_transform.apply_directional_adjoint_right(
      alpha_transform_direction,
      &directional_adjoint_right,
      &delta_directional_adjoint_right);
  Eigen::MatrixXd plus_adjoint_right = right_probe;
  Eigen::MatrixXd minus_adjoint_right = right_probe;
  xmvb::vb::ExteriorOrbitalTransform(
      alpha,
      plus_orthogonal.orbital_transform).apply_adjoint_right(
          &plus_adjoint_right);
  xmvb::vb::ExteriorOrbitalTransform(
      alpha,
      minus_orthogonal.orbital_transform).apply_adjoint_right(
          &minus_adjoint_right);
  require(
      (((plus_adjoint_right - minus_adjoint_right) /
        (2.0 * direction_step)) -
       delta_directional_adjoint_right).cwiseAbs().maxCoeff() < 2.0e-9,
      "adjoint-right exterior direction fails finite differences");

  Eigen::MatrixXd sigma_probe = coefficients;
  const Eigen::MatrixXd plus_sigma =
      xmvb::vb::DirectCiSigmaAction(
          alpha, beta, plus_orthogonal).apply(sigma_probe);
  const Eigen::MatrixXd minus_sigma =
      xmvb::vb::DirectCiSigmaAction(
          alpha, beta, minus_orthogonal).apply(sigma_probe);
  require(
      (((plus_sigma - minus_sigma) / (2.0 * direction_step)) -
       sigma_direction.apply(sigma_probe)).cwiseAbs().maxCoeff() < 2.0e-9,
      "direct-CI sigma direction fails finite differences");

  const auto apply_orthogonal_action = [&alpha, &beta, block_width](
      const xmvb::vb::OrthogonalActiveIntegrals& integrals,
      const Eigen::MatrixXd& input) {
    const xmvb::vb::ExteriorOrbitalTransform alpha_action_transform(
        alpha, integrals.orbital_transform);
    const xmvb::vb::ExteriorOrbitalTransform beta_action_transform(
        beta, integrals.orbital_transform);
    const xmvb::vb::DirectCiSigmaAction action_sigma(
        alpha, beta, integrals);
    Eigen::MatrixXd values = input;
    alpha_action_transform.apply_left(&values);
    for (int block = 0; block < block_width; ++block) {
      Eigen::MatrixXd value_block = values.middleCols(
          block * static_cast<int>(beta.size()),
          static_cast<int>(beta.size()));
      beta_action_transform.apply_right(&value_block);
      values.middleCols(
          block * static_cast<int>(beta.size()),
          static_cast<int>(beta.size())) = value_block;
    }
    Eigen::MatrixXd hamiltonian = action_sigma.apply(values);
    Eigen::MatrixXd overlap_image = values;
    for (int block = 0; block < block_width; ++block) {
      Eigen::MatrixXd hamiltonian_block = hamiltonian.middleCols(
          block * static_cast<int>(beta.size()),
          static_cast<int>(beta.size()));
      Eigen::MatrixXd overlap_block = overlap_image.middleCols(
          block * static_cast<int>(beta.size()),
          static_cast<int>(beta.size()));
      beta_action_transform.apply_adjoint_right(&hamiltonian_block);
      beta_action_transform.apply_adjoint_right(&overlap_block);
      hamiltonian.middleCols(
          block * static_cast<int>(beta.size()),
          static_cast<int>(beta.size())) = hamiltonian_block;
      overlap_image.middleCols(
          block * static_cast<int>(beta.size()),
          static_cast<int>(beta.size())) = overlap_block;
    }
    alpha_action_transform.apply_adjoint_left(&hamiltonian);
    alpha_action_transform.apply_adjoint_left(&overlap_image);
    return std::make_pair(hamiltonian, overlap_image);
  };

  Eigen::MatrixXd action_values = coefficients;
  Eigen::MatrixXd delta_action_values = Eigen::MatrixXd::Zero(
      coefficients.rows(), coefficients.cols());
  alpha_transform.apply_directional_left(
      alpha_transform_direction,
      &action_values,
      &delta_action_values);
  const auto beta_transform_direction = beta_transform.direction(
      orthogonal_direction.orbital_transform);
  for (int block = 0; block < block_width; ++block) {
    Eigen::MatrixXd value_block = action_values.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size()));
    Eigen::MatrixXd delta_value_block = delta_action_values.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size()));
    beta_transform.apply_directional_right(
        beta_transform_direction,
        &value_block,
        &delta_value_block);
    action_values.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size())) = value_block;
    delta_action_values.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size())) = delta_value_block;
  }
  const Eigen::MatrixXd inverse_orbital_transform =
      orthogonal.orbital_transform
          .template triangularView<Eigen::Upper>()
          .solve(Eigen::MatrixXd::Identity(n_orbitals, n_orbitals));
  const Eigen::MatrixXd relative_orbital_direction =
      orthogonal_direction.orbital_transform * inverse_orbital_transform;
  const Eigen::MatrixXd generator_adjoint =
      sigma_action.one_body_generator_adjoint(
          left_coefficients,
          action_values);
  const double predicted_exterior_direction =
      (generator_adjoint.cwiseProduct(relative_orbital_direction)).sum();
  const double applied_exterior_direction =
      (left_coefficients.cwiseProduct(delta_action_values)).sum();
  require(
      std::abs(predicted_exterior_direction - applied_exterior_direction) <
          3.0e-10 * std::max(1.0, std::abs(applied_exterior_direction)),
      "direct-CI orbital-generator adjoint is inconsistent: predicted=" +
          std::to_string(predicted_exterior_direction) +
          ", applied=" + std::to_string(applied_exterior_direction));
  const Eigen::MatrixXd generator_action =
      sigma_action.apply_one_body_generator(
          action_values,
          relative_orbital_direction);
  require(
      (generator_action - delta_action_values).cwiseAbs().maxCoeff() < 2.0e-11,
      "direct-CI one-body generator action is inconsistent with the "
      "exterior derivative");
  Eigen::MatrixXd action_hamiltonian = sigma_action.apply(action_values);
  Eigen::MatrixXd delta_action_hamiltonian =
      sigma_action.apply(delta_action_values) +
      sigma_direction.apply(action_values);
  Eigen::MatrixXd action_overlap = action_values;
  Eigen::MatrixXd delta_action_overlap = delta_action_values;
  for (int block = 0; block < block_width; ++block) {
    Eigen::MatrixXd hamiltonian_block = action_hamiltonian.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size()));
    Eigen::MatrixXd delta_hamiltonian_block =
        delta_action_hamiltonian.middleCols(
            block * static_cast<int>(beta.size()),
            static_cast<int>(beta.size()));
    Eigen::MatrixXd overlap_block = action_overlap.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size()));
    Eigen::MatrixXd delta_overlap_block =
        delta_action_overlap.middleCols(
            block * static_cast<int>(beta.size()),
            static_cast<int>(beta.size()));
    beta_transform.apply_directional_adjoint_right(
        beta_transform_direction,
        &hamiltonian_block,
        &delta_hamiltonian_block);
    beta_transform.apply_directional_adjoint_right(
        beta_transform_direction,
        &overlap_block,
        &delta_overlap_block);
    action_hamiltonian.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size())) = hamiltonian_block;
    delta_action_hamiltonian.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size())) = delta_hamiltonian_block;
    action_overlap.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size())) = overlap_block;
    delta_action_overlap.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size())) = delta_overlap_block;
  }
  alpha_transform.apply_directional_adjoint_left(
      alpha_transform_direction,
      &action_hamiltonian,
      &delta_action_hamiltonian);
  alpha_transform.apply_directional_adjoint_left(
      alpha_transform_direction,
      &action_overlap,
      &delta_action_overlap);
  const auto plus_action = apply_orthogonal_action(
      plus_orthogonal, coefficients);
  const auto minus_action = apply_orthogonal_action(
      minus_orthogonal, coefficients);
  require(
      ((((plus_action.first - minus_action.first) /
         (2.0 * direction_step)) -
        delta_action_hamiltonian).cwiseAbs().maxCoeff()) < 5.0e-9,
      "complete direct-CI Hamiltonian direction fails finite differences");
  require(
      ((((plus_action.second - minus_action.second) /
         (2.0 * direction_step)) -
        delta_action_overlap).cwiseAbs().maxCoeff()) < 5.0e-9,
      "complete direct-CI overlap direction fails finite differences");

  constexpr double test_energy = 0.37;
  const Eigen::MatrixXd state_coefficients = coefficients.leftCols(
      static_cast<int>(beta.size()));
  const Eigen::MatrixXd orthogonal_state = action_values.leftCols(
      static_cast<int>(beta.size()));
  const Eigen::MatrixXd orthogonal_sigma = sigma_action.apply(
      orthogonal_state);
  const Eigen::MatrixXd orthogonal_residual =
      orthogonal_sigma - test_energy * orthogonal_state;
  const xmvb::vb::DirectCiIntegralAdjoint state_integral_adjoint =
      sigma_action.integral_adjoint(
          orthogonal_state,
          orthogonal_state);
  const Eigen::MatrixXd state_generator_adjoint =
      2.0 * sigma_action.one_body_generator_adjoint(
          orthogonal_residual,
          orthogonal_state);
  const xmvb::vb::NonorthogonalActiveIntegralAdjoint state_adjoint =
      xmvb::vb::backpropagate_orthogonal_active_integral_adjoint(
          orthogonal,
          state_integral_adjoint.one_electron,
          state_integral_adjoint.pair_kernel,
          state_generator_adjoint);
  Eigen::MatrixXd original_pair_direction(n_pairs, n_pairs);
  for (int column = 0; column < n_pairs; ++column) {
    for (int row = 0; row < n_pairs; ++row) {
      original_pair_direction(row, column) =
          packed_two_electron_direction[
              xmvb::vb::TwoElectronIndexer::packed_pair_of_pairs_index(
                  row, column)];
    }
  }
  const double predicted_state_direction =
      (state_adjoint.overlap.cwiseProduct(overlap_direction)).sum() +
      (state_adjoint.one_electron.cwiseProduct(
           one_electron_direction)).sum() +
      (state_adjoint.pair_kernel.cwiseProduct(
           original_pair_direction)).sum();
  const auto state_lagrangian = [test_energy, &state_coefficients](
      const std::pair<Eigen::MatrixXd, Eigen::MatrixXd>& images) {
    return (state_coefficients.cwiseProduct(
        images.first.leftCols(state_coefficients.cols()) -
        test_energy * images.second.leftCols(
            state_coefficients.cols()))).sum();
  };
  const double finite_difference_state_direction =
      (state_lagrangian(plus_action) - state_lagrangian(minus_action)) /
      (2.0 * direction_step);
  require(
      std::abs(predicted_state_direction -
               finite_difference_state_direction) <
          3.0e-8 *
          std::max(1.0, std::abs(finite_difference_state_direction)),
      "direct-CI active-integral adjoint fails finite differences");

  const Eigen::MatrixXd directional_orthogonal_state =
      delta_action_values.leftCols(static_cast<int>(beta.size()));
  const Eigen::MatrixXd directional_orthogonal_sigma =
      sigma_action.apply(directional_orthogonal_state) +
      sigma_direction.apply(orthogonal_state);
  const Eigen::MatrixXd directional_orthogonal_residual =
      directional_orthogonal_sigma -
      test_energy * directional_orthogonal_state;
  const xmvb::vb::DirectCiIntegralAdjoint left_state_adjoint_direction =
      sigma_action.integral_adjoint(
          directional_orthogonal_state,
          orthogonal_state);
  const xmvb::vb::DirectCiIntegralAdjoint right_state_adjoint_direction =
      sigma_action.integral_adjoint(
          orthogonal_state,
          directional_orthogonal_state);
  const Eigen::MatrixXd one_gradient_direction =
      left_state_adjoint_direction.one_electron +
      right_state_adjoint_direction.one_electron;
  const Eigen::MatrixXd pair_gradient_direction =
      left_state_adjoint_direction.pair_kernel +
      right_state_adjoint_direction.pair_kernel;
  const Eigen::MatrixXd generator_gradient_direction = 2.0 *
      (sigma_action.one_body_generator_adjoint(
           directional_orthogonal_residual,
           orthogonal_state) +
       sigma_action.one_body_generator_adjoint(
           orthogonal_residual,
           directional_orthogonal_state));
  const xmvb::vb::NonorthogonalActiveIntegralAdjoint
      analytic_state_adjoint_direction =
          xmvb::vb::backpropagate_orthogonal_active_integral_adjoint_direction(
              orthogonal,
              orthogonal_direction,
              state_integral_adjoint.one_electron,
              state_integral_adjoint.pair_kernel,
              state_generator_adjoint,
              one_gradient_direction,
              pair_gradient_direction,
              generator_gradient_direction);
  const auto build_state_adjoint = [
      &alpha,
      &beta,
      &state_coefficients,
      test_energy](const xmvb::vb::OrthogonalActiveIntegrals& integrals) {
    const xmvb::vb::ExteriorOrbitalTransform alpha_state_transform(
        alpha, integrals.orbital_transform);
    const xmvb::vb::ExteriorOrbitalTransform beta_state_transform(
        beta, integrals.orbital_transform);
    const xmvb::vb::DirectCiSigmaAction state_sigma(alpha, beta, integrals);
    Eigen::MatrixXd state = state_coefficients;
    alpha_state_transform.apply_left(&state);
    beta_state_transform.apply_right(&state);
    const Eigen::MatrixXd sigma = state_sigma.apply(state);
    const Eigen::MatrixXd residual = sigma - test_energy * state;
    const xmvb::vb::DirectCiIntegralAdjoint integral_gradient =
        state_sigma.integral_adjoint(state, state);
    const Eigen::MatrixXd generator_gradient =
        2.0 * state_sigma.one_body_generator_adjoint(residual, state);
    return xmvb::vb::backpropagate_orthogonal_active_integral_adjoint(
        integrals,
        integral_gradient.one_electron,
        integral_gradient.pair_kernel,
        generator_gradient);
  };
  const xmvb::vb::NonorthogonalActiveIntegralAdjoint plus_state_adjoint =
      build_state_adjoint(plus_orthogonal);
  const xmvb::vb::NonorthogonalActiveIntegralAdjoint minus_state_adjoint =
      build_state_adjoint(minus_orthogonal);
  require(
      ((((plus_state_adjoint.overlap - minus_state_adjoint.overlap) /
         (2.0 * direction_step)) -
        analytic_state_adjoint_direction.overlap).cwiseAbs().maxCoeff()) <
          2.0e-7,
      "direct-CI overlap-adjoint direction fails finite differences");
  require(
      ((((plus_state_adjoint.one_electron -
          minus_state_adjoint.one_electron) /
         (2.0 * direction_step)) -
        analytic_state_adjoint_direction.one_electron)
           .cwiseAbs()
           .maxCoeff()) < 2.0e-7,
      "direct-CI one-electron-adjoint direction fails finite differences");
  require(
      ((((plus_state_adjoint.pair_kernel -
          minus_state_adjoint.pair_kernel) /
         (2.0 * direction_step)) -
        analytic_state_adjoint_direction.pair_kernel)
           .cwiseAbs()
           .maxCoeff()) < 2.0e-7,
      "direct-CI pair-kernel-adjoint direction fails finite differences");

  Eigen::MatrixXd orthogonal_coefficients = coefficients;
  for (int block = 0; block < block_width; ++block) {
    Eigen::MatrixXd block_coefficients =
        orthogonal_coefficients.middleCols(
            block * static_cast<int>(beta.size()),
            static_cast<int>(beta.size()));
    alpha_transform.apply_left(&block_coefficients);
    beta_transform.apply_right(&block_coefficients);
    orthogonal_coefficients.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size())) = block_coefficients;
  }
  Eigen::MatrixXd direct_hamiltonian =
      sigma_action.apply(orthogonal_coefficients);
  Eigen::MatrixXd direct_overlap = orthogonal_coefficients;
  for (int block = 0; block < block_width; ++block) {
    Eigen::MatrixXd hamiltonian_block = direct_hamiltonian.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size()));
    Eigen::MatrixXd overlap_block = direct_overlap.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size()));
    beta_transform.apply_adjoint_right(&hamiltonian_block);
    alpha_transform.apply_adjoint_left(&hamiltonian_block);
    beta_transform.apply_adjoint_right(&overlap_block);
    alpha_transform.apply_adjoint_left(&overlap_block);
    direct_hamiltonian.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size())) = hamiltonian_block;
    direct_overlap.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size())) = overlap_block;
  }

  const int product_dimension =
      static_cast<int>(alpha.size() * beta.size());
  Eigen::MatrixXd dense_hamiltonian(product_dimension, product_dimension);
  Eigen::MatrixXd dense_overlap(product_dimension, product_dimension);
  const xmvb::vb::DeterminantPairEvaluator evaluator;
  for (int target_alpha = 0;
       target_alpha < static_cast<int>(alpha.size());
       ++target_alpha) {
    for (int target_beta = 0;
         target_beta < static_cast<int>(beta.size());
         ++target_beta) {
      const int target =
          target_alpha * static_cast<int>(beta.size()) + target_beta;
      for (int source_alpha = 0;
           source_alpha < static_cast<int>(alpha.size());
           ++source_alpha) {
        for (int source_beta = 0;
             source_beta < static_cast<int>(beta.size());
             ++source_beta) {
          const int source =
              source_alpha * static_cast<int>(beta.size()) + source_beta;
          const auto pair = evaluator.evaluate(
              alpha[target_alpha],
              alpha[source_alpha],
              beta[target_beta],
              beta[source_beta],
              packed_overlap,
              one_electron,
              n_orbitals,
              two_electron,
              false);
          dense_hamiltonian(target, source) = pair.total_hamiltonian;
          dense_overlap(target, source) = pair.overlap_determinant;
        }
      }
    }
  }

  Eigen::MatrixXd coefficient_columns(product_dimension, block_width);
  for (int block = 0; block < block_width; ++block) {
    for (int alpha_index = 0;
         alpha_index < static_cast<int>(alpha.size());
         ++alpha_index) {
      for (int beta_index = 0;
           beta_index < static_cast<int>(beta.size());
           ++beta_index) {
        coefficient_columns(
            alpha_index * static_cast<int>(beta.size()) + beta_index,
            block) = coefficients(
                alpha_index,
                block * static_cast<int>(beta.size()) + beta_index);
      }
    }
  }
  const Eigen::MatrixXd reference_hamiltonian =
      dense_hamiltonian * coefficient_columns;
  const Eigen::MatrixXd reference_overlap =
      dense_overlap * coefficient_columns;
  for (int block = 0; block < block_width; ++block) {
    for (int alpha_index = 0;
         alpha_index < static_cast<int>(alpha.size());
         ++alpha_index) {
      for (int beta_index = 0;
           beta_index < static_cast<int>(beta.size());
           ++beta_index) {
        const int product =
            alpha_index * static_cast<int>(beta.size()) + beta_index;
        const int column =
            block * static_cast<int>(beta.size()) + beta_index;
        require(
            std::abs(
                direct_hamiltonian(alpha_index, column) -
                reference_hamiltonian(product, block)) < 2.0e-10,
            "orthogonal direct-CI Hamiltonian action is inconsistent");
        require(
            std::abs(
                direct_overlap(alpha_index, column) -
                reference_overlap(product, block)) < 2.0e-10,
            "orthogonal direct-CI overlap action is inconsistent");
      }
    }
  }

  // Project the exact determinant-product action onto three arbitrary sparse
  // structure vectors. The structure subspace is deliberately tiny and is
  // not invariant under H; exactness therefore cannot rely on a full-CI
  // structure basis.
  constexpr int n_test_structures = 3;
  Eigen::MatrixXd structure_expansion = Eigen::MatrixXd::Zero(
      product_dimension, n_test_structures);
  for (int structure = 0; structure < n_test_structures; ++structure) {
    for (int term = 0; term < 7; ++term) {
      const int product =
          (5 * term + 11 * structure + structure * term) % product_dimension;
      structure_expansion(product, structure) +=
          std::sin(0.31 * static_cast<double>((term + 1) * (structure + 2)));
    }
  }
  Eigen::MatrixXd structure_vectors(n_test_structures, block_width);
  for (int column = 0; column < block_width; ++column) {
    for (int row = 0; row < n_test_structures; ++row) {
      structure_vectors(row, column) =
          std::cos(0.27 * static_cast<double>((row + 1) * (column + 3)));
    }
  }
  const Eigen::MatrixXd product_vectors =
      structure_expansion * structure_vectors;
  Eigen::MatrixXd packed_product_vectors(
      static_cast<int>(alpha.size()),
      block_width * static_cast<int>(beta.size()));
  for (int block = 0; block < block_width; ++block) {
    for (int alpha_index = 0;
         alpha_index < static_cast<int>(alpha.size());
         ++alpha_index) {
      for (int beta_index = 0;
           beta_index < static_cast<int>(beta.size());
           ++beta_index) {
        packed_product_vectors(
            alpha_index,
            block * static_cast<int>(beta.size()) + beta_index) =
            product_vectors(
                alpha_index * static_cast<int>(beta.size()) + beta_index,
                block);
      }
    }
  }
  const auto projected_action = apply_orthogonal_action(
      orthogonal, packed_product_vectors);
  Eigen::MatrixXd projected_hamiltonian_products(
      product_dimension, block_width);
  Eigen::MatrixXd projected_overlap_products(
      product_dimension, block_width);
  for (int block = 0; block < block_width; ++block) {
    for (int alpha_index = 0;
         alpha_index < static_cast<int>(alpha.size());
         ++alpha_index) {
      for (int beta_index = 0;
           beta_index < static_cast<int>(beta.size());
           ++beta_index) {
        const int product =
            alpha_index * static_cast<int>(beta.size()) + beta_index;
        const int packed_column =
            block * static_cast<int>(beta.size()) + beta_index;
        projected_hamiltonian_products(product, block) =
            projected_action.first(alpha_index, packed_column);
        projected_overlap_products(product, block) =
            projected_action.second(alpha_index, packed_column);
      }
    }
  }
  const Eigen::MatrixXd direct_three_structure_hamiltonian =
      structure_expansion.transpose() * projected_hamiltonian_products;
  const Eigen::MatrixXd direct_three_structure_overlap =
      structure_expansion.transpose() * projected_overlap_products;
  const Eigen::MatrixXd reference_three_structure_hamiltonian =
      structure_expansion.transpose() * dense_hamiltonian *
      structure_expansion * structure_vectors;
  const Eigen::MatrixXd reference_three_structure_overlap =
      structure_expansion.transpose() * dense_overlap *
      structure_expansion * structure_vectors;
  require(
      (direct_three_structure_hamiltonian -
       reference_three_structure_hamiltonian).cwiseAbs().maxCoeff() < 2.0e-10,
      "direct CI is inexact on an arbitrary three-structure subspace");
  require(
      (direct_three_structure_overlap -
       reference_three_structure_overlap).cwiseAbs().maxCoeff() < 2.0e-10,
      "direct-CI overlap is inexact on an arbitrary three-structure subspace");

  std::vector<std::vector<int>> product_alpha;
  std::vector<std::vector<int>> product_beta;
  std::vector<std::vector<xmvb::vb::StructureExpansionTerm>>
      determinant_to_structure_terms(product_dimension);
  product_alpha.reserve(product_dimension);
  product_beta.reserve(product_dimension);
  for (int alpha_index = 0;
       alpha_index < static_cast<int>(alpha.size());
       ++alpha_index) {
    for (int beta_index = 0;
         beta_index < static_cast<int>(beta.size());
         ++beta_index) {
      const int product =
          alpha_index * static_cast<int>(beta.size()) + beta_index;
      product_alpha.push_back(alpha[alpha_index]);
      product_beta.push_back(beta[beta_index]);
      for (int structure = 0; structure < n_test_structures; ++structure) {
        const double coefficient = structure_expansion(product, structure);
        if (coefficient != 0.0) {
          determinant_to_structure_terms[product].push_back(
              xmvb::vb::StructureExpansionTerm{structure, coefficient});
        }
      }
    }
  }
  const auto topology = xmvb::vb::build_same_spin_pair_topology(
      product_alpha, product_beta, n_orbitals);
  require(!topology.enabled(), "topology-only context materialized pair data");
  require(topology.alpha_pair_cache.empty() && topology.beta_pair_cache.empty(),
          "topology-only context retained ordered pair arrays");

  xmvb::vb::StructureDiagonal exact_diagonal;
  exact_diagonal.hamiltonian =
      (structure_expansion.transpose() * dense_hamiltonian *
       structure_expansion).diagonal();
  exact_diagonal.overlap =
      (structure_expansion.transpose() * dense_overlap *
       structure_expansion).diagonal();
  const xmvb::vb::FullDeterminantStructureHamiltonianOverlapBuilder builder;
  const xmvb::vb::StructureDiagonal streamed_diagonal =
      builder.build_diagonal(
          product_alpha,
          product_beta,
          determinant_to_structure_terms,
          packed_overlap,
          one_electron,
          n_orbitals,
          two_electron,
          n_test_structures,
          topology);
  require(
      (streamed_diagonal.hamiltonian - exact_diagonal.hamiltonian)
              .cwiseAbs()
              .maxCoeff() < 2.0e-10,
      "streamed structure Hamiltonian diagonal is inexact");
  require(
      (streamed_diagonal.overlap - exact_diagonal.overlap)
              .cwiseAbs()
              .maxCoeff() < 2.0e-10,
      "streamed structure overlap diagonal is inexact");
  const Eigen::VectorXd overlap_only_diagonal =
      builder.build_exact_overlap_diagonal(
          product_alpha,
          product_beta,
          determinant_to_structure_terms,
          packed_overlap,
          n_orbitals,
          n_test_structures,
          topology);
  require(
      (overlap_only_diagonal - exact_diagonal.overlap)
              .cwiseAbs()
              .maxCoeff() < 2.0e-10,
      "overlap-only structure diagonal is inexact");

  xmvb::vb::StructureDiagonal expected_jacobi;
  expected_jacobi.hamiltonian =
      Eigen::VectorXd::Zero(n_test_structures);
  expected_jacobi.overlap = Eigen::VectorXd::Zero(n_test_structures);
  for (int determinant = 0; determinant < product_dimension; ++determinant) {
    for (int structure = 0; structure < n_test_structures; ++structure) {
      const double coefficient =
          structure_expansion(determinant, structure);
      expected_jacobi.hamiltonian[structure] +=
          coefficient * coefficient *
          dense_hamiltonian(determinant, determinant);
      expected_jacobi.overlap[structure] +=
          coefficient * coefficient * dense_overlap(determinant, determinant);
    }
  }
  const xmvb::vb::StructureDiagonal jacobi =
      builder.build_davidson_jacobi_preconditioner(
          product_alpha,
          product_beta,
          determinant_to_structure_terms,
          packed_overlap,
          one_electron,
          n_orbitals,
          two_electron,
          n_test_structures,
          topology);
  require(
      (jacobi.hamiltonian - expected_jacobi.hamiltonian)
              .cwiseAbs()
              .maxCoeff() < 2.0e-10,
      "determinant-diagonal Hamiltonian Jacobi model is inexact");
  require(
      (jacobi.overlap - expected_jacobi.overlap)
              .cwiseAbs()
              .maxCoeff() < 2.0e-10,
      "determinant-diagonal overlap Jacobi model is inexact");
  require(
      (jacobi.hamiltonian - exact_diagonal.hamiltonian)
              .cwiseAbs()
              .maxCoeff() > 1.0e-8,
      "Jacobi model unexpectedly retained determinant cross terms");
  xmvb::vb::StructureAction topology_action(
      determinant_to_structure_terms,
      n_test_structures,
      topology,
      packed_overlap,
      one_electron,
      two_electron,
      n_orbitals,
      &exact_diagonal);
  const auto topology_images = topology_action.apply(structure_vectors);
  require(topology_action.supports_integral_direction(),
          "topology-only complete space did not select direct CI");
  require(
      (topology_images.hamiltonian -
       reference_three_structure_hamiltonian).cwiseAbs().maxCoeff() < 2.0e-10,
      "topology-only direct-CI Hamiltonian differs from the dense reference");
  require(
      (topology_images.overlap -
       reference_three_structure_overlap).cwiseAbs().maxCoeff() < 2.0e-10,
      "topology-only direct-CI overlap differs from the dense reference");
}

void check_demand_driven_structure_diagonal() {
  // Put one demanded pair in the first cell of a unique-spin space larger
  // than the forward matrix tile. A tile-backed diagonal lookup would evaluate
  // the entire 256x256 tile even though all other determinants have zero
  // structure coefficient.
  constexpr int n_orbitals = 11;
  constexpr int n_unique_alpha = 257;
  auto alpha = complete_space(n_orbitals, 5);
  alpha.resize(n_unique_alpha);
  std::vector<std::vector<int>> beta(n_unique_alpha, std::vector<int>{0});
  std::vector<std::vector<xmvb::vb::StructureExpansionTerm>> expansion(
      n_unique_alpha);
  expansion.front().push_back({0, 1.0});

  const Eigen::MatrixXd overlap =
      Eigen::MatrixXd::Identity(n_orbitals, n_orbitals);
  const std::vector<double> packed_overlap(
      overlap.data(),
      overlap.data() + overlap.size());
  Eigen::MatrixXd one_electron(n_orbitals, n_orbitals);
  for (int column = 0; column < n_orbitals; ++column) {
    for (int row = 0; row <= column; ++row) {
      const double value =
          0.03 * std::cos(static_cast<double>((row + 1) * (column + 2)));
      one_electron(row, column) = value;
      one_electron(column, row) = value;
    }
  }
  xmvb::vb::ActiveSpaceTwoElectronResult two_electron;
  two_electron.packed_active_two_electron_integrals.assign(
      xmvb::vb::packed_active_two_electron_integral_count(n_orbitals),
      0.0);

  const auto large_topology = xmvb::vb::build_same_spin_pair_topology(
      alpha,
      beta,
      n_orbitals);
  require(!large_topology.enabled(),
          "demand-driven diagonal regression unexpectedly built pair caches");
  require(large_topology.alpha_reuse_table.unique_determinants.size() ==
              n_unique_alpha,
          "demand-driven diagonal regression did not span multiple tiles");

  const xmvb::vb::FullDeterminantStructureHamiltonianOverlapBuilder builder;
  const xmvb::vb::StructureDiagonal large_diagonal = builder.build_diagonal(
      alpha,
      beta,
      expansion,
      packed_overlap,
      one_electron,
      n_orbitals,
      two_electron,
      1,
      large_topology);

  const std::vector<std::vector<int>> reference_alpha{alpha.front()};
  const std::vector<std::vector<int>> reference_beta{beta.front()};
  const std::vector<std::vector<xmvb::vb::StructureExpansionTerm>>
      reference_expansion{{{0, 1.0}}};
  const auto reference_topology = xmvb::vb::build_same_spin_pair_topology(
      reference_alpha,
      reference_beta,
      n_orbitals);
  const xmvb::vb::StructureDiagonal reference_diagonal =
      builder.build_diagonal(
          reference_alpha,
          reference_beta,
          reference_expansion,
          packed_overlap,
          one_electron,
          n_orbitals,
          two_electron,
          1,
          reference_topology);

  require(
      std::abs(
          large_diagonal.hamiltonian[0] -
          reference_diagonal.hamiltonian[0]) < 2.0e-12,
      "unused unique determinants changed the demand-driven Hamiltonian diagonal");
  require(
      std::abs(
          large_diagonal.overlap[0] -
          reference_diagonal.overlap[0]) < 2.0e-12,
      "unused unique determinants changed the demand-driven overlap diagonal");

  std::vector<std::vector<xmvb::vb::StructureExpansionTerm>>
      jacobi_expansion(n_unique_alpha);
  double expected_jacobi_hamiltonian = 0.0;
  double expected_jacobi_overlap = 0.0;
  const xmvb::vb::DeterminantPairEvaluator evaluator;
  for (int determinant = 0; determinant < n_unique_alpha; ++determinant) {
    const double coefficient =
        std::sin(0.013 * static_cast<double>(determinant + 1));
    if (determinant == 0) {
      // Duplicate determinant/structure terms must be combined before the
      // coefficient is squared.
      jacobi_expansion[determinant].push_back({0, 0.4 * coefficient});
      jacobi_expansion[determinant].push_back({0, 0.6 * coefficient});
    } else {
      jacobi_expansion[determinant].push_back({0, coefficient});
    }
    const auto self_pair = evaluator.evaluate(
        alpha[determinant],
        alpha[determinant],
        beta[determinant],
        beta[determinant],
        packed_overlap,
        one_electron,
        n_orbitals,
        two_electron,
        false);
    expected_jacobi_hamiltonian +=
        coefficient * coefficient * self_pair.total_hamiltonian;
    expected_jacobi_overlap +=
        coefficient * coefficient * self_pair.overlap_determinant;
  }
  const xmvb::vb::StructureDiagonal jacobi =
      builder.build_davidson_jacobi_preconditioner(
          alpha,
          beta,
          jacobi_expansion,
          packed_overlap,
          one_electron,
          n_orbitals,
          two_electron,
          1,
          large_topology);
  require(
      std::abs(jacobi.hamiltonian[0] - expected_jacobi_hamiltonian) < 2.0e-10,
      "large determinant-diagonal Hamiltonian Jacobi model is inexact");
  require(
      std::abs(jacobi.overlap[0] - expected_jacobi_overlap) < 2.0e-10,
      "large determinant-diagonal overlap Jacobi model is inexact");

  const std::vector<std::vector<xmvb::vb::StructureExpansionTerm>>
      cancelling_expansion{{{0, 1.0}, {0, -1.0}}};
  bool rejected_zero_overlap = false;
  try {
    static_cast<void>(builder.build_davidson_jacobi_preconditioner(
        reference_alpha,
        reference_beta,
        cancelling_expansion,
        packed_overlap,
        one_electron,
        n_orbitals,
        two_electron,
        1,
        reference_topology));
  } catch (const std::domain_error&) {
    rejected_zero_overlap = true;
  }
  require(
      rejected_zero_overlap,
      "zero Davidson Jacobi overlap was not rejected explicitly");
}

}  // namespace

int main() {
  try {
    check_exterior_transform();
    check_planner();
    check_sigma_action();
    check_demand_driven_structure_diagonal();
    std::cout << "orthogonal direct-CI planner and exterior transform: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "orthogonal direct-CI test failed: " << error.what() << '\n';
    return 1;
  }
}
