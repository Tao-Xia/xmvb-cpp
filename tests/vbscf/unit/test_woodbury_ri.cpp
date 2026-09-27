#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/determinants/algebra/cofactor_differential.hpp"
#include "vbscf/determinants/algebra/overlap.hpp"
#include "vbscf/determinants/pairs/contractions.hpp"
#include "vbscf/determinants/pairs/accepted_tile.hpp"
#include "vbscf/determinants/pairs/evaluator.hpp"
#include "vbscf/determinants/pairs/woodbury_ri.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"

namespace {

void require_close(
    double value,
    double reference,
    double tolerance,
    const char* label) {
  const double scale = std::max({1.0, std::abs(value), std::abs(reference)});
  if (std::abs(value - reference) > tolerance * scale) {
    throw std::runtime_error(std::string(label) + " mismatch");
  }
}

void require_matrix_close(
    const Eigen::MatrixXd& value,
    const Eigen::MatrixXd& reference,
    double tolerance,
    const char* label) {
  const double scale = std::max({1.0, value.norm(), reference.norm()});
  if (value.rows() != reference.rows() || value.cols() != reference.cols() ||
      (value - reference).norm() > tolerance * scale) {
    throw std::runtime_error(std::string(label) + " mismatch");
  }
}

Eigen::MatrixXd overlap_block(
    const Eigen::MatrixXd& active_overlap,
    const std::vector<int>& left,
    const std::vector<int>& right) {
  Eigen::MatrixXd result(right.size(), left.size());
  for (int column = 0; column < static_cast<int>(left.size()); ++column) {
    for (int row = 0; row < static_cast<int>(right.size()); ++row) {
      result(row, column) = active_overlap(right[row], left[column]);
    }
  }
  return result;
}

Eigen::MatrixXd transition(
    const Eigen::MatrixXd& factors,
    Eigen::Index auxiliary,
    const std::vector<int>& left,
    const std::vector<int>& right) {
  Eigen::MatrixXd result(right.size(), left.size());
  for (int column = 0; column < static_cast<int>(left.size()); ++column) {
    for (int row = 0; row < static_cast<int>(right.size()); ++row) {
      result(row, column) = factors(
          auxiliary,
          xmvb::vb::TwoElectronIndexer::packed_pair_index(
              right[row], left[column]));
    }
  }
  return result;
}

Eigen::MatrixXd exterior_square(const Eigen::MatrixXd& matrix) {
  const int n = matrix.rows();
  Eigen::MatrixXd result(
      n * (n - 1) / 2,
      n * (n - 1) / 2);
  for (int j = 1; j < n; ++j) {
    for (int i = 0; i < j; ++i) {
      const int row = j * (j - 1) / 2 + i;
      for (int l = 1; l < n; ++l) {
        for (int k = 0; k < l; ++k) {
          const int column = l * (l - 1) / 2 + k;
          result(row, column) =
              matrix(i, k) * matrix(j, l) -
              matrix(i, l) * matrix(j, k);
        }
      }
    }
  }
  return result;
}

void check_state(
    const xmvb::vb::WoodburyRiState& state,
    const Eigen::MatrixXd& active_overlap,
    const Eigen::MatrixXd& factors,
    const std::vector<int>& left,
    const std::vector<int>& right,
    const char* label) {
  xmvb::vb::DeterminantOverlapResolver resolver;
  const auto resolved = resolver.resolve_matrix(
      overlap_block(active_overlap, left, right));
  const xmvb::vb::CofactorDifferential cofactor(resolved);
  require_close(
      state.overlap_determinant(),
      resolved.overlap_determinant,
      2.0e-9,
      label);
  require_matrix_close(
      state.first_cofactor(), cofactor.value(), 2.0e-8, label);

  double two_electron = 0.0;
  Eigen::VectorXd auxiliary(factors.rows());
  for (Eigen::Index q = 0; q < factors.rows(); ++q) {
    const Eigen::MatrixXd factor = transition(factors, q, left, right);
    auxiliary(q) = (cofactor.value().cwiseProduct(factor)).sum();
    two_electron += cofactor.second_contraction(exterior_square(factor));
  }
  require_close(
      state.two_electron_contraction(), two_electron, 2.0e-8, label);
  require_matrix_close(
      state.first_cofactor_auxiliary(), auxiliary, 2.0e-8, label);

  Eigen::MatrixXd one_electron(left.size(), left.size());
  Eigen::MatrixXd direction(left.size(), left.size());
  for (int column = 0; column < one_electron.cols(); ++column) {
    for (int row = 0; row < one_electron.rows(); ++row) {
      one_electron(row, column) =
          0.031 * (row + 1) - 0.014 * (column + 2);
      direction(row, column) =
          0.008 * (column + 1) + 0.003 * (row - column);
    }
  }
  const Eigen::MatrixXd gradient = state.hamiltonian_overlap_gradient(
      one_electron, factors);
  constexpr double epsilon = 2.0e-6;
  const Eigen::MatrixXd center = overlap_block(active_overlap, left, right);
  xmvb::vb::WoodburyRiState plus;
  xmvb::vb::WoodburyRiState minus;
  if (!plus.initialize(left, right, center + epsilon * direction, factors) ||
      !minus.initialize(left, right, center - epsilon * direction, factors)) {
    throw std::runtime_error("finite-difference RI state initialization failed");
  }
  const double plus_value = plus.one_electron_contraction(one_electron) +
      plus.two_electron_contraction();
  const double minus_value = minus.one_electron_contraction(one_electron) +
      minus.two_electron_contraction();
  require_close(
      (gradient.cwiseProduct(direction)).sum(),
      (plus_value - minus_value) / (2.0 * epsilon),
      3.0e-6,
      "RI Hamiltonian overlap gradient");
}

}  // namespace

int main() {
  try {
    constexpr int n_active = 7;
    constexpr int n_auxiliary = 9;
    Eigen::MatrixXd orbital_vectors(5, n_active);
    for (int column = 0; column < n_active; ++column) {
      for (int row = 0; row < orbital_vectors.rows(); ++row) {
        orbital_vectors(row, column) =
            std::sin(0.19 * (row + 1) * (column + 2)) +
            0.2 * std::cos(0.31 * (row + 2) * (column + 1));
      }
    }
    orbital_vectors.col(5) = orbital_vectors.col(4);
    orbital_vectors.col(6) =
        orbital_vectors.col(3) + 1.0e-8 * orbital_vectors.col(2);
    const Eigen::MatrixXd active_overlap =
        orbital_vectors.transpose() * orbital_vectors;

    Eigen::MatrixXd factors(
        n_auxiliary,
        xmvb::vb::packed_active_pair_count(n_active));
    for (Eigen::Index column = 0; column < factors.cols(); ++column) {
      for (Eigen::Index row = 0; row < factors.rows(); ++row) {
        factors(row, column) =
            0.013 * (row + 1) - 0.007 * (column + 2) +
            0.002 * ((row + 3) * (column + 1) % 7);
      }
    }

    std::vector<int> left{0, 1, 4, 6};
    std::vector<int> right{0, 2, 4, 6};
    xmvb::vb::WoodburyRiState state;
    if (!state.initialize(
            left,
            right,
            overlap_block(active_overlap, left, right),
            factors)) {
      throw std::runtime_error("failed to initialize Woodbury RI state");
    }
    check_state(state, active_overlap, factors, left, right, "initial pair");

    right[2] = 5;
    if (!state.update_right(
            right,
            overlap_block(active_overlap, left, right),
            factors)) {
      throw std::runtime_error("failed right Woodbury RI update");
    }
    check_state(state, active_overlap, factors, left, right, "right update");

    left[3] = 3;
    if (!state.update_left(
            left,
            overlap_block(active_overlap, left, right),
            factors)) {
      throw std::runtime_error("failed left Woodbury RI update");
    }
    check_state(state, active_overlap, factors, left, right, "left update");

    right[3] = 3;
    if (!state.update_right(
            right,
            overlap_block(active_overlap, left, right),
            factors)) {
      throw std::runtime_error("failed singular Woodbury RI update");
    }
    check_state(state, active_overlap, factors, left, right, "singular update");

    const std::vector<std::vector<int>> strings{
        {0, 1, 4, 6},
        {0, 2, 4, 6},
        {0, 2, 5, 6},
        {0, 2, 3, 5}};
    std::vector<double> overlap_storage(
        active_overlap.data(),
        active_overlap.data() + active_overlap.size());
    Eigen::MatrixXd one_electron(n_active, n_active);
    for (int column = 0; column < n_active; ++column) {
      for (int row = 0; row < n_active; ++row) {
        one_electron(row, column) =
            0.021 * (row + 1) - 0.009 * (column + 2);
      }
    }
    xmvb::vb::ActiveSpaceTwoElectronResult two_electron;
    two_electron.representation =
        xmvb::vb::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity;
    two_electron.n_auxiliary_functions = n_auxiliary;
    two_electron.ri_active_pair_factors = factors;
    const xmvb::vb::AcceptedPairTileProvider provider(strings, n_active);
    const xmvb::vb::DeterminantPairEvaluator evaluator;
    for (const bool materialize : {false, true}) {
      const xmvb::vb::AcceptedSpinPairTile tile = provider.build(
          0,
          strings.size(),
          0,
          strings.size(),
          overlap_storage,
          one_electron,
          two_electron,
          xmvb::vb::AcceptedPairTileBuildOptions{
              .materialize_projected_pair_values = materialize,
              .populate_response_payload = false,
              .populate_opposite_spin_projection = true});
      for (int left_index = 0;
           left_index < static_cast<int>(strings.size());
           ++left_index) {
        for (int right_index = 0;
             right_index < static_cast<int>(strings.size());
             ++right_index) {
          const auto reference = evaluator.evaluate_same_spin_pair(
              strings[left_index],
              strings[right_index],
              overlap_storage,
              one_electron,
              n_active,
              two_electron,
              false);
          const auto& value = tile.pair(left_index, right_index);
          require_close(
              value.overlap_result.overlap_determinant,
              reference.overlap_result.overlap_determinant,
              2.0e-8,
              "tile overlap");
          require_close(
              value.total_hamiltonian,
              reference.total_hamiltonian,
              3.0e-8,
              "tile Hamiltonian");
          require_matrix_close(
              xmvb::vb::calc_cofactor_1st(value.overlap_result),
              xmvb::vb::calc_cofactor_1st(reference.overlap_result),
              3.0e-8,
              "tile first cofactor");
          if (materialize) {
            const auto& projected = value.opposite_spin_pair_cache
                .first_order_cofactor_projection.projected_pair_values;
            if (projected.size() !=
                static_cast<std::size_t>(
                    xmvb::vb::packed_active_pair_count(n_active))) {
              throw std::runtime_error(
                  "tile projected RI channel size mismatch");
            }
          }
        }
      }
    }

    std::cout << "Woodbury RI contraction tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
