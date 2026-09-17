#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"
#include "vbscf/structures/orthogonal_ci/integrals.hpp"
#include "vbscf/structures/orthogonal_ci/planner.hpp"
#include "vbscf/structures/orthogonal_ci/sigma.hpp"
#include "vbscf/determinants/pairs/evaluator.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"

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

  transform.apply_adjoint_right(&right);
  require(
      (right - original_right * exterior.transpose() * exterior)
              .cwiseAbs()
              .maxCoeff() < 4.0e-13,
      "right exterior adjoint is inconsistent");

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
  incomplete.back() = incomplete.front();
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
  const xmvb::vb::ExteriorOrbitalTransform alpha_transform(
      alpha, orthogonal.orbital_transform);
  const xmvb::vb::ExteriorOrbitalTransform beta_transform(
      beta, orthogonal.orbital_transform);
  const xmvb::vb::DirectCiSigmaAction sigma_action(
      alpha, beta, orthogonal);

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
}

}  // namespace

int main() {
  try {
    check_exterior_transform();
    check_planner();
    check_sigma_action();
    std::cout << "orthogonal direct-CI planner and exterior transform: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "orthogonal direct-CI test failed: " << error.what() << '\n';
    return 1;
  }
}
