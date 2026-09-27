#include "vbscf/derivatives/hessian/responses/same_spin/pair_response_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/matrix_weights_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "core/openmp.hpp"
#include "vbscf/determinants/algebra/cofactor_differential.hpp"
#include "vbscf/determinants/pairs/storage.hpp"
#include "vbscf/determinants/pairs/contractions.hpp"
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

const SameSpinPolynomialDirectionalPairData& SameSpinDirectionalPairTile::pair(
    int left_local,
    int right_local) const {
  if (left_local < 0 || left_local >= left_size() ||
      right_local < 0 || right_local >= right_size()) {
    throw std::out_of_range("directional same-spin tile index out of range");
  }
  return pairs[static_cast<std::size_t>(left_local) * right_size() +
      right_local];
}

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
#pragma omp atomic update
          (*packed_active_two_electron_gradient)[direct_index] +=
              second_order_cofactor;
#pragma omp atomic update
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
  if (!pair_evaluation.same_spin_polynomial_response) {
    throw std::logic_error(
        "polynomial same-spin response requires its explicit payload");
  }
  const auto& polynomial = *pair_evaluation.same_spin_polynomial_response;
  const CofactorDifferential& cofactor =
      cached_cofactor_differential(pair_evaluation);
  const Eigen::MatrixXd ds = build_local_overlap_direction_matrix(
      occ_L, occ_R, direction.overlap, n_active_orbitals);
  const Eigen::MatrixXd dh = build_spin_one_electron_block_matrix_local(
      occ_L, occ_R, Eigen::Map<const Eigen::MatrixXd>(
          direction.one_electron.data(), n_active_orbitals, n_active_orbitals));
  const Eigen::MatrixXd dg = build_spin_antisymmetrized_interaction_direction(
      occ_L, occ_R, direction.packed_two_electron);
  result.delta_cofactor_1st = cofactor.first(ds);
  result.delta_overlap_determinant = (cofactor.value().cwiseProduct(ds)).sum();
  result.delta_total_hamiltonian =
      (dh.cwiseProduct(cofactor.value())).sum() +
      cofactor.second_contraction(dg) +
      (pair_evaluation.same_spin_overlap_hamiltonian_gradient.cwiseProduct(ds)).sum();
  if (need_overlap_gradient)
    result.delta_same_spin_overlap_hamiltonian_gradient =
        cofactor.mixed(ds, polynomial.one_electron_block) +
        cofactor.first(dh) +
        cofactor.second_contraction_gradient_direction(
            ds, polynomial.antisymmetrized_interaction, dg);
  return result;
}

SameSpinDirectionalPairTile build_directional_pair_tile(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int n_active_orbitals,
    const ActiveSpaceIntegralDirectionView& direction,
    int left_begin,
    int left_end,
    int right_begin,
    int right_end,
    const Eigen::MatrixXd* accepted_active_one_electron,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors,
    const Eigen::MatrixXd* directional_ri_active_pair_factors) {
  const std::size_t expected_size = square_storage_size(n_unique_determinants);
  if (ordered_pair_cache.size() != expected_size ||
      unique_determinants.size() !=
          static_cast<std::size_t>(n_unique_determinants) ||
      left_begin < 0 || left_end <= left_begin ||
      left_end > n_unique_determinants || right_begin < 0 ||
      right_end <= right_begin || right_end > n_unique_determinants) {
    throw std::invalid_argument(
        "same-spin directional tile dimensions are inconsistent");
  }
  const bool use_ri = accepted_active_one_electron != nullptr &&
      accepted_ri_active_pair_factors != nullptr &&
      directional_ri_active_pair_factors != nullptr;
  if ((accepted_active_one_electron != nullptr ||
       accepted_ri_active_pair_factors != nullptr ||
       directional_ri_active_pair_factors != nullptr) &&
      !use_ri) {
    throw std::invalid_argument(
        "same-spin RI directional tile requires all accepted/directional matrices");
  }

  SameSpinDirectionalPairTile tile;
  tile.left_begin = left_begin;
  tile.right_begin = right_begin;
  const int left_size = left_end - left_begin;
  const int right_size = right_end - right_begin;
  tile.delta_overlap = Eigen::MatrixXd::Zero(left_size, right_size);
  tile.delta_regular_hamiltonian =
      Eigen::MatrixXd::Zero(left_size, right_size);
  tile.delta_singular_hamiltonian =
      Eigen::MatrixXd::Zero(left_size, right_size);
  tile.pairs.resize(static_cast<std::size_t>(left_size) * right_size);

  Eigen::Map<const Eigen::MatrixXd> delta_h1e(
      direction.one_electron.data(),
      n_active_orbitals,
      n_active_orbitals);
  const int work_items = left_size * right_size;
  const int n_threads = std::max(
      1,
      std::min(xmvb::effective_openmp_thread_count(), work_items));
#pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
  for (int work = 0; work < work_items; ++work) {
    const int left_local = work / right_size;
    const int right_local = work % right_size;
    const int left_id = left_begin + left_local;
    const int right_id = right_begin + right_local;
    const int canonical_left = std::min(left_id, right_id);
    const int canonical_right = std::max(left_id, right_id);
    const auto& pair_evaluation = ordered_pair_cache[
        ordered_spin_pair_storage_index(
            canonical_left, canonical_right, n_unique_determinants)];
    SameSpinPolynomialDirectionalPairData pair_direction;
    const bool regular_ri_pair =
        pair_evaluation.has_same_spin_phi_cache && use_ri;
    if (regular_ri_pair) {
      const Eigen::MatrixXd delta_overlap =
          build_local_overlap_direction_matrix(
              unique_determinants[canonical_left],
              unique_determinants[canonical_right],
              direction.overlap,
              n_active_orbitals);
      const RegularRiSameSpinDirection ri_direction =
          evaluate_regular_ri_same_spin_direction(
              unique_determinants[canonical_left],
              unique_determinants[canonical_right],
              *accepted_active_one_electron,
              delta_h1e,
              n_active_orbitals,
              *accepted_ri_active_pair_factors,
              *directional_ri_active_pair_factors,
              pair_evaluation.overlap_result,
              delta_overlap,
              pair_evaluation.same_spin_total_phi,
              pair_evaluation.same_spin_inverse_overlap_gradient);
      pair_direction.delta_overlap_determinant =
          ri_direction.delta_overlap_determinant;
      pair_direction.delta_total_hamiltonian =
          ri_direction.delta_total_hamiltonian;
      pair_direction.delta_cofactor_1st =
          ri_direction.delta_first_cofactor;
      pair_direction.delta_same_spin_overlap_hamiltonian_gradient =
          ri_direction.delta_overlap_hamiltonian_gradient;
    } else {
      pair_direction = build_polynomial_spin_directional_data(
          unique_determinants[canonical_left],
          unique_determinants[canonical_right],
          pair_evaluation,
          n_active_orbitals,
          direction);
    }

    if (left_id > right_id) {
      pair_direction.delta_cofactor_1st.transposeInPlace();
      pair_direction.delta_same_spin_overlap_hamiltonian_gradient
          .transposeInPlace();
    }

    tile.pairs[static_cast<std::size_t>(work)] = std::move(pair_direction);
    const auto& stored = tile.pairs[static_cast<std::size_t>(work)];
    tile.delta_overlap(left_local, right_local) =
        stored.delta_overlap_determinant;
    Eigen::MatrixXd& hamiltonian =
        pair_evaluation.overlap_result.nullity == 0 &&
                pair_evaluation.overlap_result.overlap_determinant != 0.0
            ? tile.delta_regular_hamiltonian
            : tile.delta_singular_hamiltonian;
    hamiltonian(left_local, right_local) =
        stored.delta_total_hamiltonian;
  }
  return tile;
}

// Obsolete full-grid builders are removed after the tile migration; this
// private declaration keeps the remaining implementation isolated meanwhile.
struct SameSpinDirectionalScalarMatrices {
  Eigen::MatrixXd delta_overlap_determinant_matrix;
  Eigen::MatrixXd delta_regular_total_hamiltonian_matrix;
  Eigen::MatrixXd delta_singular_total_hamiltonian_matrix;
  std::vector<SameSpinPolynomialDirectionalPairData> ordered_pair_data;
};

SameSpinDirectionalScalarMatrices build_directional_pair_scalar_matrices(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int n_active_orbitals,
    const ActiveSpaceIntegralDirectionView& direction,
    const Eigen::MatrixXd* accepted_active_one_electron,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors,
    const Eigen::MatrixXd* directional_ri_active_pair_factors) {
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

  const int n_threads = xmvb::effective_openmp_thread_count();
#pragma omp parallel for schedule(dynamic, 1) if(n_threads > 1) num_threads(n_threads)
  for (int left_id = 0; left_id < n_unique_determinants; ++left_id) {
    for (int right_id = left_id; right_id < n_unique_determinants; ++right_id) {
      const std::size_t forward_index = ordered_spin_pair_storage_index(
          left_id,
          right_id,
          n_unique_determinants);
      const auto& pair_evaluation = ordered_pair_cache[forward_index];
      SameSpinPolynomialDirectionalPairData directional_data;
      const bool regular_ri_pair =
          pair_evaluation.has_same_spin_phi_cache &&
          accepted_active_one_electron != nullptr &&
          accepted_ri_active_pair_factors != nullptr &&
          directional_ri_active_pair_factors != nullptr;
      if (regular_ri_pair) {
        const Eigen::Map<const Eigen::MatrixXd> delta_h1e(
            direction.one_electron.data(),
            n_active_orbitals,
            n_active_orbitals);
        const Eigen::MatrixXd delta_overlap =
            build_local_overlap_direction_matrix(
                unique_determinants[left_id],
                unique_determinants[right_id],
                direction.overlap,
                n_active_orbitals);
        const RegularRiSameSpinDirection ri_direction =
            evaluate_regular_ri_same_spin_direction(
                unique_determinants[left_id],
                unique_determinants[right_id],
                *accepted_active_one_electron,
                delta_h1e,
                n_active_orbitals,
                *accepted_ri_active_pair_factors,
                *directional_ri_active_pair_factors,
                pair_evaluation.overlap_result,
                delta_overlap,
                pair_evaluation.same_spin_total_phi,
                pair_evaluation.same_spin_inverse_overlap_gradient);
        directional_data.delta_overlap_determinant =
            ri_direction.delta_overlap_determinant;
        directional_data.delta_total_hamiltonian =
            ri_direction.delta_total_hamiltonian;
        directional_data.delta_cofactor_1st =
            ri_direction.delta_first_cofactor;
        directional_data.delta_same_spin_overlap_hamiltonian_gradient =
            ri_direction.delta_overlap_hamiltonian_gradient;
      } else {
        directional_data = build_polynomial_spin_directional_data(
            unique_determinants[left_id],
            unique_determinants[right_id],
            pair_evaluation,
            n_active_orbitals,
            direction);
      }
      scalars.ordered_pair_data[forward_index] = directional_data;
      if (left_id != right_id) {
        SameSpinPolynomialDirectionalPairData transposed = directional_data;
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

std::vector<SameSpinDirectionalScalarMatrices>
build_directional_pair_scalar_matrices_batch(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int n_active_orbitals,
    const std::vector<ActiveSpaceIntegralDirectionView>& directions,
    const Eigen::MatrixXd* accepted_active_one_electron,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors,
    const std::vector<Eigen::MatrixXd>* directional_ri_active_pair_factors) {
  const std::size_t expected_size = square_storage_size(n_unique_determinants);
  if (ordered_pair_cache.size() != expected_size ||
      unique_determinants.size() !=
          static_cast<std::size_t>(n_unique_determinants)) {
    throw std::invalid_argument(
        "same-spin pair data do not match unique determinant dimensions");
  }
  const bool use_ri_block = accepted_active_one_electron != nullptr &&
      accepted_ri_active_pair_factors != nullptr &&
      directional_ri_active_pair_factors != nullptr;
  if ((accepted_active_one_electron != nullptr ||
       accepted_ri_active_pair_factors != nullptr ||
       directional_ri_active_pair_factors != nullptr) &&
      !use_ri_block) {
    throw std::invalid_argument(
        "same-spin RI direction block requires accepted matrices and all factor directions");
  }
  if (use_ri_block &&
      directional_ri_active_pair_factors->size() != directions.size()) {
    throw std::invalid_argument(
        "same-spin RI factor direction block has inconsistent width");
  }

  std::vector<SameSpinDirectionalScalarMatrices> blocks(directions.size());
  std::vector<Eigen::MatrixXd> delta_h1e;
  delta_h1e.reserve(directions.size());
  for (std::size_t direction = 0; direction < directions.size(); ++direction) {
    auto& scalars = blocks[direction];
    scalars.delta_overlap_determinant_matrix =
        Eigen::MatrixXd::Zero(n_unique_determinants, n_unique_determinants);
    scalars.delta_regular_total_hamiltonian_matrix =
        Eigen::MatrixXd::Zero(n_unique_determinants, n_unique_determinants);
    scalars.delta_singular_total_hamiltonian_matrix =
        Eigen::MatrixXd::Zero(n_unique_determinants, n_unique_determinants);
    scalars.ordered_pair_data.resize(expected_size);
    if (directions[direction].one_electron.size() !=
        square_storage_size(n_active_orbitals)) {
      throw std::invalid_argument(
          "same-spin direction block one-electron matrix has inconsistent dimensions");
    }
    delta_h1e.emplace_back(Eigen::Map<const Eigen::MatrixXd>(
        directions[direction].one_electron.data(),
        n_active_orbitals,
        n_active_orbitals));
  }

  const int n_threads = xmvb::effective_openmp_thread_count();
#pragma omp parallel for schedule(dynamic, 1) if(n_threads > 1) num_threads(n_threads)
  for (int left_id = 0; left_id < n_unique_determinants; ++left_id) {
    for (int right_id = left_id; right_id < n_unique_determinants; ++right_id) {
      const std::size_t forward_index = ordered_spin_pair_storage_index(
          left_id,
          right_id,
          n_unique_determinants);
      const auto& pair_evaluation = ordered_pair_cache[forward_index];
      const bool regular_ri_pair =
          pair_evaluation.has_same_spin_phi_cache && use_ri_block;
      std::vector<SameSpinPolynomialDirectionalPairData> pair_directions(
          directions.size());
      if (regular_ri_pair) {
        std::vector<Eigen::MatrixXd> delta_overlap;
        delta_overlap.reserve(directions.size());
        for (const auto& direction : directions) {
          delta_overlap.emplace_back(build_local_overlap_direction_matrix(
              unique_determinants[left_id],
              unique_determinants[right_id],
              direction.overlap,
              n_active_orbitals));
        }
        const auto ri_directions =
            evaluate_regular_ri_same_spin_direction_batch(
                unique_determinants[left_id],
                unique_determinants[right_id],
                *accepted_active_one_electron,
                delta_h1e,
                n_active_orbitals,
                *accepted_ri_active_pair_factors,
                *directional_ri_active_pair_factors,
                pair_evaluation.overlap_result,
                delta_overlap,
                pair_evaluation.same_spin_total_phi,
                pair_evaluation.same_spin_inverse_overlap_gradient);
        for (std::size_t direction = 0;
             direction < directions.size();
             ++direction) {
          pair_directions[direction].delta_overlap_determinant =
              ri_directions[direction].delta_overlap_determinant;
          pair_directions[direction].delta_total_hamiltonian =
              ri_directions[direction].delta_total_hamiltonian;
          pair_directions[direction].delta_cofactor_1st =
              ri_directions[direction].delta_first_cofactor;
          pair_directions[direction]
              .delta_same_spin_overlap_hamiltonian_gradient =
                  ri_directions[direction]
                      .delta_overlap_hamiltonian_gradient;
        }
      } else {
        for (std::size_t direction = 0;
             direction < directions.size();
             ++direction) {
          pair_directions[direction] =
              build_polynomial_spin_directional_data(
                  unique_determinants[left_id],
                  unique_determinants[right_id],
                  pair_evaluation,
                  n_active_orbitals,
                  directions[direction]);
        }
      }

      for (std::size_t direction = 0;
           direction < directions.size();
           ++direction) {
        auto& scalars = blocks[direction];
        const auto& pair_direction = pair_directions[direction];
        scalars.ordered_pair_data[forward_index] = pair_direction;
        if (left_id != right_id) {
          SameSpinPolynomialDirectionalPairData transposed = pair_direction;
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
            pair_direction.delta_overlap_determinant);
        Eigen::MatrixXd* directional_hamiltonian =
            pair_evaluation.overlap_result.nullity == 0 &&
                    pair_evaluation.overlap_result.overlap_determinant != 0.0
                ? &scalars.delta_regular_total_hamiltonian_matrix
                : &scalars.delta_singular_total_hamiltonian_matrix;
        set_symmetric_entry(
            directional_hamiltonian,
            left_id,
            right_id,
            pair_direction.delta_total_hamiltonian);
      }
    }
  }
  return blocks;
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
#pragma omp atomic update
          (*packed_active_two_electron_gradient)[direct_index] +=
              total_directional_second_order_cofactor;
#pragma omp atomic update
          (*packed_active_two_electron_gradient)[exchange_index] -=
              total_directional_second_order_cofactor;
        }
      }
    }
  }
}


}  // namespace xmvb::vb::detail
