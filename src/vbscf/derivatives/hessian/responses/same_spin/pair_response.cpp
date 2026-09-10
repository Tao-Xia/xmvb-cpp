#include "vbscf/derivatives/hessian/responses/same_spin/pair_response_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/matrix_weights_internal.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

#include "vbscf/determinants/cofactor_differential.hpp"
#include "vbscf/determinants/pair_storage.hpp"
#include "vbscf/determinants/spin_pair_contractions.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"

namespace xmvb::vb::detail {

namespace {

constexpr double kContributionTolerance = 1.0e-15;

std::size_t square_storage_size(int dimension) {
  return static_cast<std::size_t>(dimension) *
      static_cast<std::size_t>(dimension);
}

void set_symmetric_entry(
    Eigen::MatrixXd* matrix,
    int row,
    int column,
    double value) {
  (*matrix)(row, column) = value;
  (*matrix)(column, row) = value;
}

}  // namespace

void accumulate_one_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& cofactor_1st,
    double weight,
    Eigen::MatrixXd* active_one_electron_gradient) {
  if (active_one_electron_gradient == nullptr) {
    throw std::invalid_argument("active_one_electron_gradient must not be null");
  }
  if (std::abs(weight) <= kContributionTolerance) {
    return;
  }

  for (int left_column = 0; left_column < static_cast<int>(occ_L.size()); ++left_column) {
    const int orbital_index_left = occ_L[left_column];
    for (int right_row = 0; right_row < static_cast<int>(occ_R.size()); ++right_row) {
      const int orbital_index_right = occ_R[right_row];
      (*active_one_electron_gradient)(orbital_index_right, orbital_index_left) +=
          weight * cofactor_1st(right_row, left_column);
    }
  }
}

void accumulate_overlap_block_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& overlap_block_gradient,
    double weight,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (active_orbital_overlap_gradient == nullptr) {
    throw std::invalid_argument("active_orbital_overlap_gradient must not be null");
  }
  if (std::abs(weight) <= kContributionTolerance) {
    return;
  }
  if (overlap_block_gradient.rows() != static_cast<int>(occ_R.size()) ||
      overlap_block_gradient.cols() != static_cast<int>(occ_L.size())) {
    throw std::invalid_argument(
        "overlap_block_gradient dimensions do not match occupied lists");
  }

  for (int left_column = 0; left_column < static_cast<int>(occ_L.size()); ++left_column) {
    const int orbital_index_left = occ_L[left_column];
    for (int right_row = 0; right_row < static_cast<int>(occ_R.size()); ++right_row) {
      const int orbital_index_right = occ_R[right_row];
      (*active_orbital_overlap_gradient)[orbital_index_left *
                                             n_active_orbitals +
                                         orbital_index_right] +=
          weight * overlap_block_gradient(right_row, left_column);
    }
  }
}

void accumulate_deleted_minor_same_spin_two_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const CofactorDifferential& cofactor,
    double weight,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (packed_active_two_electron_gradient == nullptr) {
    throw std::invalid_argument("packed_active_two_electron_gradient must not be null");
  }
  if (std::abs(weight) <= kContributionTolerance) {
    return;
  }

  const int n_electrons = static_cast<int>(occ_L.size());
  if (n_electrons < 2) {
    return;
  }

  const Eigen::MatrixXd second = cofactor.second();
  for (int left_first = 0; left_first < n_electrons - 1; ++left_first) {
    const int orbital_index_left_first = occ_L[left_first];
    for (int right_first = 0; right_first < n_electrons - 1; ++right_first) {
      const int orbital_index_right_first = occ_R[right_first];
      for (int left_second = left_first + 1; left_second < n_electrons; ++left_second) {
        const int orbital_index_left_second = occ_L[left_second];
        for (int right_second = right_first + 1; right_second < n_electrons; ++right_second) {
          const int orbital_index_right_second = occ_R[right_second];
          const double second_order_cofactor =
              weight * second(right_second*(right_second-1)/2+right_first,
                     left_second*(left_second-1)/2+left_first);
          if (std::abs(second_order_cofactor) <= kContributionTolerance) {
            continue;
          }

          const int direct_index = TwoElectronIndexer::two_electron_storage_index(
              orbital_index_right_first,
              orbital_index_left_first,
              orbital_index_right_second,
              orbital_index_left_second);
          const int exchange_index = TwoElectronIndexer::two_electron_storage_index(
              orbital_index_right_first,
              orbital_index_left_second,
              orbital_index_right_second,
              orbital_index_left_first);
          (*packed_active_two_electron_gradient)[direct_index] +=
              second_order_cofactor;
          (*packed_active_two_electron_gradient)[exchange_index] -=
              second_order_cofactor;
        }
      }
    }
  }
}

Eigen::MatrixXd build_spin_one_electron_block_matrix_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act) {
  const int n_electrons = static_cast<int>(occ_L.size());
  Eigen::MatrixXd one_electron_block(n_electrons, n_electrons);
  for (int left_column = 0; left_column < n_electrons; ++left_column) {
    const int orbital_index_left = occ_L[left_column];
    for (int right_row = 0; right_row < n_electrons; ++right_row) {
      const int orbital_index_right = occ_R[right_row];
      one_electron_block(right_row, left_column) =
          h1e_act(orbital_index_right, orbital_index_left);
    }
  }
  return one_electron_block;
}

Eigen::MatrixXd build_local_overlap_direction_matrix(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& delta_ao_overlap_matrix,
    int n_active_orbitals) {
  return build_overlap_submatrix(
      occ_L,
      occ_R,
      delta_ao_overlap_matrix,
      n_active_orbitals);
}

SameSpinPolynomialDirectionalPairData build_polynomial_spin_directional_data(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const SpinDeterminantPairEvaluation& pair_evaluation,
    int n_active_orbitals,
    const ActiveSpaceIntegralDirectionView& direction,
    bool need_overlap_gradient) {
  SameSpinPolynomialDirectionalPairData result;
  const CofactorDifferential& cofactor =
      cached_cofactor_differential(pair_evaluation);
  const Eigen::MatrixXd ds = build_local_overlap_direction_matrix(
      occ_L, occ_R, direction.overlap, n_active_orbitals);
  const Eigen::MatrixXd dh = build_spin_one_electron_block_matrix_local(
      occ_L, occ_R, Eigen::Map<const Eigen::MatrixXd>(
          direction.one_electron.data(), n_active_orbitals, n_active_orbitals));
  const Eigen::MatrixXd dg = build_spin_antisymmetrized_interaction_direction(
      occ_L, occ_R, direction.packed_two_electron);
  result.cofactor_1st = cofactor.value();
  result.delta_cofactor_1st = cofactor.first(ds);
  result.delta_overlap_determinant = (result.cofactor_1st.cwiseProduct(ds)).sum();
  result.delta_total_hamiltonian =
      (dh.cwiseProduct(result.cofactor_1st)).sum() +
      (dg.cwiseProduct(cofactor.second())).sum() +
      (pair_evaluation.same_spin_overlap_hamiltonian_gradient.cwiseProduct(ds)).sum();
  if (need_overlap_gradient)
    result.delta_same_spin_overlap_hamiltonian_gradient =
        cofactor.mixed(ds, pair_evaluation.same_spin_one_electron_block) +
        cofactor.first(dh) +
        cofactor.second_contraction_gradient_direction(
            ds, pair_evaluation.same_spin_antisymmetrized_interaction, dg);
  return result;
}

SameSpinDirectionalScalarMatrices build_directional_pair_scalar_matrices(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int n_active_orbitals,
    const ActiveSpaceIntegralDirectionView& direction) {
  const std::size_t expected_size = square_storage_size(n_unique_determinants);
  if (ordered_pair_cache.size() != expected_size ||
      unique_determinants.size() !=
          static_cast<std::size_t>(n_unique_determinants)) {
    throw std::invalid_argument(
        "same-spin pair data do not match unique determinant dimensions");
  }

  SameSpinDirectionalScalarMatrices scalars;
  scalars.delta_overlap_determinant_matrix =
      Eigen::MatrixXd::Zero(n_unique_determinants, n_unique_determinants);
  scalars.delta_regular_total_hamiltonian_matrix =
      Eigen::MatrixXd::Zero(n_unique_determinants, n_unique_determinants);
  scalars.delta_singular_total_hamiltonian_matrix =
      Eigen::MatrixXd::Zero(n_unique_determinants, n_unique_determinants);
  scalars.ordered_pair_data.resize(expected_size);

  for (int left_id = 0; left_id < n_unique_determinants; ++left_id) {
    for (int right_id = left_id; right_id < n_unique_determinants; ++right_id) {
      const std::size_t forward_index = ordered_spin_pair_storage_index(
          left_id,
          right_id,
          n_unique_determinants);
      const auto& pair_evaluation = ordered_pair_cache[forward_index];
      SameSpinPolynomialDirectionalPairData directional_data =
          build_polynomial_spin_directional_data(
              unique_determinants[left_id],
              unique_determinants[right_id],
              pair_evaluation,
              n_active_orbitals,
              direction);
      scalars.ordered_pair_data[forward_index] = directional_data;
      if (left_id != right_id) {
        SameSpinPolynomialDirectionalPairData transposed = directional_data;
        transposed.cofactor_1st.transposeInPlace();
        transposed.delta_cofactor_1st.transposeInPlace();
        transposed.delta_same_spin_overlap_hamiltonian_gradient
            .transposeInPlace();
        scalars.ordered_pair_data[ordered_spin_pair_storage_index(
            right_id,
            left_id,
            n_unique_determinants)] = std::move(transposed);
      }

      set_symmetric_entry(
          &scalars.delta_overlap_determinant_matrix,
          left_id,
          right_id,
          directional_data.delta_overlap_determinant);
      Eigen::MatrixXd* directional_hamiltonian =
          pair_evaluation.overlap_result.nullity == 0 &&
                  pair_evaluation.overlap_result.overlap_determinant != 0.0
              ? &scalars.delta_regular_total_hamiltonian_matrix
              : &scalars.delta_singular_total_hamiltonian_matrix;
      set_symmetric_entry(
          directional_hamiltonian,
          left_id,
          right_id,
          directional_data.delta_total_hamiltonian);
    }
  }
  return scalars;
}

void accumulate_directional_one_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& cofactor_1st,
    const Eigen::MatrixXd& delta_cofactor_1st,
    double weight,
    double delta_weight,
    Eigen::MatrixXd* active_one_electron_gradient) {
  if (active_one_electron_gradient == nullptr) {
    throw std::invalid_argument("active_one_electron_gradient must not be null");
  }
  if (std::abs(weight) <= kContributionTolerance &&
      std::abs(delta_weight) <= kContributionTolerance) {
    return;
  }

  for (int left_column = 0; left_column < static_cast<int>(occ_L.size()); ++left_column) {
    const int orbital_index_left = occ_L[left_column];
    for (int right_row = 0; right_row < static_cast<int>(occ_R.size()); ++right_row) {
      const int orbital_index_right = occ_R[right_row];
      (*active_one_electron_gradient)(orbital_index_right, orbital_index_left) +=
          delta_weight * cofactor_1st(right_row, left_column) +
          weight * delta_cofactor_1st(right_row, left_column);
    }
  }
}

void accumulate_directional_deleted_minor_same_spin_two_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const CofactorDifferential& cofactor,
    const std::vector<double>& delta_ao_overlap_matrix,
    int n_active_orbitals,
    double weight,
    double delta_weight,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (packed_active_two_electron_gradient == nullptr) {
    throw std::invalid_argument("packed_active_two_electron_gradient must not be null");
  }
  if (std::abs(weight) <= kContributionTolerance &&
      std::abs(delta_weight) <= kContributionTolerance) {
    return;
  }

  const int n_electrons = static_cast<int>(occ_L.size());
  if (n_electrons < 2) {
    return;
  }

  const Eigen::MatrixXd delta_overlap_submatrix =
      build_local_overlap_direction_matrix(
          occ_L,
          occ_R,
          delta_ao_overlap_matrix,
          n_active_orbitals);
  const Eigen::MatrixXd second = cofactor.second();
  const Eigen::MatrixXd delta_second = cofactor.second_first(delta_overlap_submatrix);
  for (int left_first = 0; left_first < n_electrons - 1; ++left_first) {
    const int orbital_index_left_first = occ_L[left_first];
    for (int right_first = 0; right_first < n_electrons - 1; ++right_first) {
      const int orbital_index_right_first = occ_R[right_first];
      for (int left_second = left_first + 1; left_second < n_electrons; ++left_second) {
        const int orbital_index_left_second = occ_L[left_second];
        for (int right_second = right_first + 1; right_second < n_electrons; ++right_second) {
          const int orbital_index_right_second = occ_R[right_second];
          const double second_order_cofactor =
              second(right_second*(right_second-1)/2+right_first,
                     left_second*(left_second-1)/2+left_first);
          const double directional_second_order_cofactor =
              delta_second(right_second*(right_second-1)/2+right_first,
                           left_second*(left_second-1)/2+left_first);
          const double total_directional_second_order_cofactor =
              delta_weight * second_order_cofactor +
              weight * directional_second_order_cofactor;
          if (std::abs(total_directional_second_order_cofactor) <=
              kContributionTolerance) {
            continue;
          }

          const int direct_index = TwoElectronIndexer::two_electron_storage_index(
              orbital_index_right_first,
              orbital_index_left_first,
              orbital_index_right_second,
              orbital_index_left_second);
          const int exchange_index = TwoElectronIndexer::two_electron_storage_index(
              orbital_index_right_first,
              orbital_index_left_second,
              orbital_index_right_second,
              orbital_index_left_first);
          (*packed_active_two_electron_gradient)[direct_index] +=
              total_directional_second_order_cofactor;
          (*packed_active_two_electron_gradient)[exchange_index] -=
              total_directional_second_order_cofactor;
        }
      }
    }
  }
}


}  // namespace xmvb::vb::detail
