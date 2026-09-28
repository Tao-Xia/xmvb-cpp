#include "vbscf/derivatives/hessian/responses/same_spin/pair_response_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <Eigen/LU>

#include "core/openmp.hpp"
#include "vbscf/determinants/algebra/cofactor_differential.hpp"
#include "vbscf/determinants/algebra/ri_directional_graph.hpp"
#include "vbscf/determinants/pairs/storage.hpp"
#include "vbscf/determinants/pairs/contractions.hpp"
#include "vbscf/determinants/pairs/traversal.hpp"
#include "vbscf/determinants/pairs/woodbury_ri.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"

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

int directional_auxiliary_tile_width(
    int n_auxiliary,
    int n_electrons,
    int n_workers) {
  constexpr std::size_t workspace_bytes = 256ULL * 1024ULL * 1024ULL;
  // ContractedDensityJet owns accepted and directional power/moment tables.
  // Sixteen occupied matrices per auxiliary is a conservative live bound.
  const std::size_t bytes_per_auxiliary = std::max<std::size_t>(
      1,
      16ULL * static_cast<std::size_t>(n_electrons) * n_electrons *
          sizeof(double));
  const std::size_t bytes_per_worker =
      workspace_bytes / static_cast<std::size_t>(std::max(1, n_workers));
  return std::max(
      1,
      std::min(
          n_auxiliary,
          static_cast<int>(bytes_per_worker / bytes_per_auxiliary)));
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

SameSpinDirectionalPairTileView SameSpinDirectionalPairTile::view(
    bool transposed) const {
  return SameSpinDirectionalPairTileView(*this, transposed);
}

int SameSpinDirectionalPairTileView::left_begin() const noexcept {
  return transposed_ ? storage_->right_begin : storage_->left_begin;
}

int SameSpinDirectionalPairTileView::right_begin() const noexcept {
  return transposed_ ? storage_->left_begin : storage_->right_begin;
}

int SameSpinDirectionalPairTileView::left_size() const noexcept {
  return transposed_ ? storage_->right_size() : storage_->left_size();
}

int SameSpinDirectionalPairTileView::right_size() const noexcept {
  return transposed_ ? storage_->left_size() : storage_->right_size();
}

SameSpinDirectionalPairTileView::ConstMatrixMap
SameSpinDirectionalPairTileView::matrix_view(
    const Eigen::MatrixXd& matrix) const {
  if (!transposed_) {
    return ConstMatrixMap(
        matrix.data(),
        matrix.rows(),
        matrix.cols(),
        Stride(matrix.outerStride(), matrix.innerStride()));
  }
  return ConstMatrixMap(
      matrix.data(),
      matrix.cols(),
      matrix.rows(),
      Stride(matrix.innerStride(), matrix.outerStride()));
}

SameSpinDirectionalPairTileView::ConstMatrixMap
SameSpinDirectionalPairTileView::delta_overlap() const {
  return matrix_view(storage_->delta_overlap);
}

SameSpinDirectionalPairTileView::ConstMatrixMap
SameSpinDirectionalPairTileView::delta_regular_hamiltonian() const {
  return matrix_view(storage_->delta_regular_hamiltonian);
}

SameSpinDirectionalPairTileView::ConstMatrixMap
SameSpinDirectionalPairTileView::delta_singular_hamiltonian() const {
  return matrix_view(storage_->delta_singular_hamiltonian);
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

SameSpinPolynomialDirectionalPairData regular_one_electron_direction(
    const std::vector<int>& occupied_left,
    const std::vector<int>& occupied_right,
    const SpinDeterminantPairEvaluation& accepted,
    const Eigen::Ref<const Eigen::MatrixXd>& one_electron,
    const Eigen::Ref<const Eigen::MatrixXd>& one_electron_direction,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_direction) {
  Eigen::MatrixXd inverse =
      accepted.overlap_result.inverse_overlap_submatrix;
  if (inverse.rows() != overlap_direction.rows() ||
      inverse.cols() != overlap_direction.cols()) {
    inverse = accepted.overlap_result.overlap_submatrix.inverse();
  }
  const Eigen::MatrixXd inverse_direction =
      -inverse * overlap_direction * inverse;
  const double determinant = accepted.overlap_result.overlap_determinant;
  const double determinant_direction =
      determinant * (inverse * overlap_direction).trace();
  const Eigen::MatrixXd h = build_spin_one_electron_block_matrix_local(
      occupied_left, occupied_right, one_electron);
  const Eigen::MatrixXd dh = build_spin_one_electron_block_matrix_local(
      occupied_left, occupied_right, one_electron_direction);
  const double phi = (inverse * h).trace();
  const double phi_direction =
      (inverse_direction * h).trace() + (inverse * dh).trace();
  const Eigen::MatrixXd response = inverse * h * inverse;
  const Eigen::MatrixXd response_direction =
      inverse_direction * h * inverse + inverse * dh * inverse +
      inverse * h * inverse_direction;
  const Eigen::MatrixXd bracket =
      phi * inverse.transpose() - response.transpose();
  const Eigen::MatrixXd bracket_direction =
      phi_direction * inverse.transpose() +
      phi * inverse_direction.transpose() -
      response_direction.transpose();

  SameSpinPolynomialDirectionalPairData result;
  result.delta_overlap_determinant = determinant_direction;
  result.delta_total_hamiltonian =
      determinant_direction * phi + determinant * phi_direction;
  result.delta_cofactor_1st =
      determinant_direction * inverse.transpose() +
      determinant * inverse_direction.transpose();
  result.delta_same_spin_overlap_hamiltonian_gradient =
      determinant_direction * bracket + determinant * bracket_direction;
  return result;
}

static SameSpinPolynomialDirectionalPairData
build_polynomial_spin_directional_data_impl(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const SpinDeterminantPairEvaluation& pair_evaluation,
    int n_active_orbitals,
    const ActiveSpaceIntegralDirectionView& direction,
    const Eigen::Ref<const Eigen::MatrixXd>& interaction_direction,
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
  result.delta_cofactor_1st = cofactor.first(ds);
  result.delta_overlap_determinant = (cofactor.value().cwiseProduct(ds)).sum();
  result.delta_total_hamiltonian =
      (dh.cwiseProduct(cofactor.value())).sum() +
      cofactor.second_contraction(interaction_direction) +
      (pair_evaluation.same_spin_overlap_hamiltonian_gradient.cwiseProduct(ds)).sum();
  if (need_overlap_gradient)
    result.delta_same_spin_overlap_hamiltonian_gradient =
        cofactor.mixed(ds, polynomial.one_electron_block) +
        cofactor.first(dh) +
        cofactor.second_contraction_gradient_direction(
            ds,
            polynomial.antisymmetrized_interaction,
            interaction_direction);
  return result;
}

static Eigen::MatrixXd build_ri_spin_antisymmetrized_interaction_direction(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    int n_active_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& accepted_factors,
    const Eigen::Ref<const Eigen::MatrixXd>& directional_factors) {
  const int n_electrons = static_cast<int>(occ_L.size());
  const int n_electron_pairs = n_electrons * (n_electrons - 1) / 2;
  Eigen::MatrixXd result(n_electron_pairs, n_electron_pairs);
  const auto pair = TwoElectronIndexer::packed_pair_index;
  const auto kernel_direction = [&](int p, int q) {
    return directional_factors.col(p).dot(accepted_factors.col(q)) +
        accepted_factors.col(p).dot(directional_factors.col(q));
  };
  for (int j = 1; j < n_electrons; ++j) {
    for (int i = 0; i < j; ++i) {
      for (int l = 1; l < n_electrons; ++l) {
        for (int k = 0; k < l; ++k) {
          result(j * (j - 1) / 2 + i, l * (l - 1) / 2 + k) =
              kernel_direction(
                  pair(occ_R[i], occ_L[k]),
                  pair(occ_R[j], occ_L[l])) -
              kernel_direction(
                  pair(occ_R[i], occ_L[l]),
                  pair(occ_R[j], occ_L[k]));
        }
      }
    }
  }
  return result;
}

SameSpinPolynomialDirectionalPairData build_polynomial_spin_directional_data(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const SpinDeterminantPairEvaluation& pair_evaluation,
    int n_active_orbitals,
    const ActiveSpaceIntegralDirectionView& direction,
    bool need_overlap_gradient) {
  const Eigen::MatrixXd interaction_direction =
      build_spin_antisymmetrized_interaction_direction(
          occ_L, occ_R, direction.packed_two_electron);
  return build_polynomial_spin_directional_data_impl(
      occ_L,
      occ_R,
      pair_evaluation,
      n_active_orbitals,
      direction,
      interaction_direction,
      need_overlap_gradient);
}

namespace {

SameSpinDirectionalPairTile build_directional_pair_tile_impl(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>* ordered_pair_cache,
    const AcceptedSpinPairTile* accepted_pair_tile,
    int n_unique_determinants,
    int n_active_orbitals,
    const ActiveSpaceIntegralDirectionView& direction,
    int left_begin,
    int left_end,
    int right_begin,
    int right_end,
    const Eigen::MatrixXd* accepted_active_one_electron,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors,
    const Eigen::MatrixXd* directional_ri_active_pair_factors,
    bool build_ri_projected_channels) {
  const std::size_t expected_size = square_storage_size(n_unique_determinants);
  const bool full_cache_valid = ordered_pair_cache != nullptr &&
      ordered_pair_cache->size() == expected_size;
  const bool tile_cache_valid = accepted_pair_tile != nullptr &&
      accepted_pair_tile->left_begin == left_begin &&
      accepted_pair_tile->right_begin == right_begin &&
      accepted_pair_tile->left_size == left_end - left_begin &&
      accepted_pair_tile->right_size == right_end - right_begin;
  if ((!full_cache_valid && !tile_cache_valid) ||
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
  if (use_ri && build_ri_projected_channels) {
    tile.ri_projected_channels = Eigen::MatrixXd::Zero(
        left_size * right_size,
        packed_active_pair_count(n_active_orbitals));
    tile.ri_projected_ready.assign(
        static_cast<std::size_t>(left_size) * right_size, 0);
  }
  tile.pairs.resize(static_cast<std::size_t>(left_size) * right_size);

  Eigen::Map<const Eigen::MatrixXd> delta_h1e(
      direction.one_electron.data(),
      n_active_orbitals,
      n_active_orbitals);
  const bool stream_woodbury_ri = use_ri && accepted_pair_tile != nullptr &&
      std::all_of(
          accepted_pair_tile->pairs.begin(),
          accepted_pair_tile->pairs.end(),
          [](const SpinDeterminantPairEvaluation& pair) {
            return pair.has_woodbury_ri_response ||
                pair.has_same_spin_phi_cache;
          });
  const auto regular_graph_pair =
      [](const SpinDeterminantPairEvaluation& pair) {
        return pair.has_same_spin_phi_cache &&
            pair.overlap_result.nullity == 0 &&
            pair.overlap_result.overlap_determinant != 0.0;
      };
  const bool has_regular_directional_graph =
      stream_woodbury_ri && std::any_of(
          accepted_pair_tile->pairs.begin(),
          accepted_pair_tile->pairs.end(),
          regular_graph_pair);
  const bool all_regular_directional_graph =
      has_regular_directional_graph && std::all_of(
          accepted_pair_tile->pairs.begin(),
          accepted_pair_tile->pairs.end(),
          regular_graph_pair);
  std::vector<char> regular_graph_ready(tile.pairs.size(), 0);
  if (has_regular_directional_graph) {
    const int n_threads = std::max(
        1,
        std::min(xmvb::effective_openmp_thread_count(), left_size));
    const int auxiliary_tile = directional_auxiliary_tile_width(
        accepted_ri_active_pair_factors->rows(),
        static_cast<int>(unique_determinants[left_begin].size()),
        n_threads);
#pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
    for (int left_local = 0; left_local < left_size; ++left_local) {
      const int left_id = left_begin + left_local;
      const auto& occupied_left = unique_determinants[left_id];
      const std::vector<int> right_traversal = build_pair_update_traversal(
          unique_determinants, left_id, right_begin, right_end);

      for (int right_local = 0; right_local < right_size; ++right_local) {
        const int right_id = right_begin + right_local;
        const auto& occupied_right = unique_determinants[right_id];
        const auto& accepted =
            accepted_pair_tile->pair(left_local, right_local);
        if (!regular_graph_pair(accepted)) {
          continue;
        }
        const Eigen::MatrixXd overlap_direction =
            build_local_overlap_direction_matrix(
                occupied_left,
                occupied_right,
                direction.overlap,
                n_active_orbitals);
        SameSpinPolynomialDirectionalPairData pair_direction =
            regular_one_electron_direction(
                occupied_left,
                occupied_right,
                accepted,
                *accepted_active_one_electron,
                delta_h1e,
                overlap_direction);
        const std::size_t pair_index =
            static_cast<std::size_t>(left_local) * right_size + right_local;
        tile.delta_overlap(left_local, right_local) =
            pair_direction.delta_overlap_determinant;
        tile.delta_regular_hamiltonian(left_local, right_local) =
            pair_direction.delta_total_hamiltonian;
        tile.pairs[pair_index] = std::move(pair_direction);
        regular_graph_ready[pair_index] = 1;
      }

      for (int auxiliary_begin = 0;
           auxiliary_begin < accepted_ri_active_pair_factors->rows();
           auxiliary_begin += auxiliary_tile) {
        const int auxiliary_size = std::min(
            auxiliary_tile,
            static_cast<int>(accepted_ri_active_pair_factors->rows()) -
                auxiliary_begin);
        const auto accepted_factors =
            accepted_ri_active_pair_factors->middleRows(
                auxiliary_begin, auxiliary_size);
        const auto directional_factors =
            directional_ri_active_pair_factors->middleRows(
                auxiliary_begin, auxiliary_size);
        RiDirectionalGraph graph;
        bool initialized = false;
        for (const int right_id : right_traversal) {
          const int right_local = right_id - right_begin;
          const auto& occupied_right = unique_determinants[right_id];
          const auto& accepted =
              accepted_pair_tile->pair(left_local, right_local);
          if (!regular_graph_pair(accepted)) {
            initialized = false;
            continue;
          }
          const Eigen::MatrixXd overlap_direction =
              build_local_overlap_direction_matrix(
                  occupied_left,
                  occupied_right,
                  direction.overlap,
                  n_active_orbitals);
          const bool updated = initialized && graph.update_right(
              occupied_right,
              accepted.overlap_result.overlap_submatrix,
              overlap_direction,
              accepted_factors,
              directional_factors);
          if (!updated && !graph.initialize(
                  occupied_left,
                  occupied_right,
                  accepted.overlap_result.overlap_submatrix,
                  overlap_direction,
                  accepted_factors,
                  directional_factors)) {
            throw std::runtime_error(
                "failed to initialize regular directional RI graph");
          }
          initialized = true;
          const RiDirectionalTileValue value = graph.value();
          const std::size_t pair_index =
              static_cast<std::size_t>(left_local) * right_size + right_local;
          auto& pair_direction = tile.pairs[pair_index];
          pair_direction.delta_total_hamiltonian +=
              value.two_electron_direction;
          pair_direction.delta_same_spin_overlap_hamiltonian_gradient
              .noalias() += value.overlap_gradient_direction;
          tile.delta_regular_hamiltonian(left_local, right_local) =
              pair_direction.delta_total_hamiltonian;
          if (build_ri_projected_channels) {
            const int channel_work =
                left_local + left_size * right_local;
            tile.ri_projected_channels.row(channel_work).noalias() +=
                (accepted_factors.transpose() *
                     value.directional_first_contractions +
                 directional_factors.transpose() *
                     value.accepted_first_contractions)
                    .transpose();
            tile.ri_projected_ready[
                static_cast<std::size_t>(channel_work)] = 1;
          }
        }
      }
    }
    if (all_regular_directional_graph) {
      return tile;
    }
  }
  if (stream_woodbury_ri) {
    const int n_threads = std::max(
        1,
        std::min(xmvb::effective_openmp_thread_count(), left_size));
#pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
    for (int left_local = 0; left_local < left_size; ++left_local) {
      const int left_id = left_begin + left_local;
      const auto& occupied_left = unique_determinants[left_id];
      const std::vector<int> right_traversal = build_pair_update_traversal(
          unique_determinants, left_id, right_begin, right_end);
      WoodburyRiState state;
      bool initialized = false;
      for (const int right_id : right_traversal) {
        const int right_local = right_id - right_begin;
        const auto& occupied_right = unique_determinants[right_id];
        const auto& accepted =
            accepted_pair_tile->pair(left_local, right_local);
        const bool updated = initialized && state.update_right(
            occupied_right,
            accepted.overlap_result.overlap_submatrix,
            *accepted_ri_active_pair_factors);
        if (!updated && !state.initialize(
                occupied_left,
                occupied_right,
                accepted.overlap_result.overlap_submatrix,
                *accepted_ri_active_pair_factors)) {
          throw std::runtime_error(
              "failed to initialize streamed directional Woodbury RI state");
        }
        initialized = true;

        const std::size_t pair_index =
            static_cast<std::size_t>(left_local) * right_size + right_local;
        if (regular_graph_ready[pair_index]) {
          continue;
        }

        const Eigen::MatrixXd overlap_direction =
            build_local_overlap_direction_matrix(
                occupied_left,
                occupied_right,
                direction.overlap,
                n_active_orbitals);
        SameSpinPolynomialDirectionalPairData pair_direction;
        if (accepted.has_woodbury_ri_response) {
          const Eigen::MatrixXd one_electron =
              build_spin_one_electron_block_matrix_local(
                  occupied_left,
                  occupied_right,
                  *accepted_active_one_electron);
          const Eigen::MatrixXd one_electron_direction =
              build_spin_one_electron_block_matrix_local(
                  occupied_left,
                  occupied_right,
                  delta_h1e);
          const WoodburyRiDirection woodbury_direction =
              state.hamiltonian_direction(
                  one_electron,
                  one_electron_direction,
                  overlap_direction,
                  *accepted_ri_active_pair_factors,
                  *directional_ri_active_pair_factors,
                  build_ri_projected_channels);
          pair_direction.delta_overlap_determinant =
              woodbury_direction.overlap_determinant;
          pair_direction.delta_total_hamiltonian =
              woodbury_direction.hamiltonian;
          pair_direction.delta_cofactor_1st =
              woodbury_direction.first_cofactor;
          pair_direction.delta_same_spin_overlap_hamiltonian_gradient =
              woodbury_direction.hamiltonian_overlap_gradient;
          if (build_ri_projected_channels) {
            const int channel_work = left_local + left_size * right_local;
            tile.ri_projected_channels.row(channel_work).noalias() =
                (accepted_ri_active_pair_factors->transpose() *
                     woodbury_direction.directional_auxiliary +
                 directional_ri_active_pair_factors->transpose() *
                     woodbury_direction.accepted_auxiliary)
                    .transpose();
            tile.ri_projected_ready[
                static_cast<std::size_t>(channel_work)] = 1;
          }
        } else {
          const RegularRiSameSpinDirection regular_direction =
              evaluate_regular_ri_same_spin_direction(
                  occupied_left,
                  occupied_right,
                  *accepted_active_one_electron,
                  delta_h1e,
                  n_active_orbitals,
                  *accepted_ri_active_pair_factors,
                  *directional_ri_active_pair_factors,
                  accepted.overlap_result,
                  overlap_direction,
                  accepted.same_spin_total_phi,
                  accepted.same_spin_inverse_overlap_gradient,
                  build_ri_projected_channels);
          pair_direction.delta_overlap_determinant =
              regular_direction.delta_overlap_determinant;
          pair_direction.delta_total_hamiltonian =
              regular_direction.delta_total_hamiltonian;
          pair_direction.delta_cofactor_1st =
              regular_direction.delta_first_cofactor;
          pair_direction.delta_same_spin_overlap_hamiltonian_gradient =
              regular_direction.delta_overlap_hamiltonian_gradient;
          if (build_ri_projected_channels) {
            const int channel_work = left_local + left_size * right_local;
            tile.ri_projected_channels.row(channel_work) =
                regular_direction.delta_projected_first_cofactor.transpose();
            tile.ri_projected_ready[
                static_cast<std::size_t>(channel_work)] = 1;
          }
        }
        const double overlap_value = pair_direction.delta_overlap_determinant;
        const double hamiltonian_value = pair_direction.delta_total_hamiltonian;
        tile.pairs[pair_index] = std::move(pair_direction);
        tile.delta_overlap(left_local, right_local) =
            overlap_value;
        Eigen::MatrixXd& hamiltonian =
            accepted.overlap_result.nullity == 0 &&
                    accepted.overlap_result.overlap_determinant != 0.0
                ? tile.delta_regular_hamiltonian
                : tile.delta_singular_hamiltonian;
        hamiltonian(left_local, right_local) =
            hamiltonian_value;
      }
    }
    return tile;
  }
  const bool diagonal_tile = left_begin == right_begin &&
      left_size == right_size;
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
    if (diagonal_tile && left_id > right_id) {
      continue;
    }
    const int canonical_left = std::min(left_id, right_id);
    const int canonical_right = std::max(left_id, right_id);
    const auto& pair_evaluation = accepted_pair_tile != nullptr
        ? accepted_pair_tile->pair(
              canonical_left - left_begin,
              canonical_right - right_begin)
        : (*ordered_pair_cache)[ordered_spin_pair_storage_index(
              canonical_left, canonical_right, n_unique_determinants)];
    SameSpinPolynomialDirectionalPairData pair_direction;
    const bool woodbury_ri_pair =
        pair_evaluation.has_woodbury_ri_response && use_ri;
    const bool regular_ri_pair =
        pair_evaluation.has_same_spin_phi_cache && use_ri;
    if (woodbury_ri_pair) {
      const auto& occupied_left = unique_determinants[canonical_left];
      const auto& occupied_right = unique_determinants[canonical_right];
      const Eigen::MatrixXd delta_overlap =
          build_local_overlap_direction_matrix(
              occupied_left,
              occupied_right,
              direction.overlap,
              n_active_orbitals);
      const Eigen::MatrixXd one_electron =
          build_spin_one_electron_block_matrix_local(
              occupied_left,
              occupied_right,
              *accepted_active_one_electron);
      const Eigen::MatrixXd one_electron_direction =
          build_spin_one_electron_block_matrix_local(
              occupied_left,
              occupied_right,
              delta_h1e);
      WoodburyRiState state;
      if (!state.initialize(
              occupied_left,
              occupied_right,
              pair_evaluation.overlap_result.overlap_submatrix,
              *accepted_ri_active_pair_factors)) {
        throw std::runtime_error(
            "failed to initialize directional Woodbury RI state");
      }
      const WoodburyRiDirection woodbury_direction =
          state.hamiltonian_direction(
              one_electron,
              one_electron_direction,
              delta_overlap,
              *accepted_ri_active_pair_factors,
              *directional_ri_active_pair_factors,
              build_ri_projected_channels);
      pair_direction.delta_overlap_determinant =
          woodbury_direction.overlap_determinant;
      pair_direction.delta_total_hamiltonian =
          woodbury_direction.hamiltonian;
      pair_direction.delta_cofactor_1st =
          woodbury_direction.first_cofactor;
      pair_direction.delta_same_spin_overlap_hamiltonian_gradient =
          woodbury_direction.hamiltonian_overlap_gradient;
      if (build_ri_projected_channels) {
        const int channel_work = left_local + left_size * right_local;
        tile.ri_projected_channels.row(channel_work).noalias() =
            (accepted_ri_active_pair_factors->transpose() *
                 woodbury_direction.directional_auxiliary +
             directional_ri_active_pair_factors->transpose() *
                 woodbury_direction.accepted_auxiliary)
                .transpose();
        tile.ri_projected_ready[static_cast<std::size_t>(channel_work)] = 1;
      }
    } else if (regular_ri_pair) {
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
              pair_evaluation.same_spin_inverse_overlap_gradient,
              build_ri_projected_channels);
      pair_direction.delta_overlap_determinant =
          ri_direction.delta_overlap_determinant;
      pair_direction.delta_total_hamiltonian =
          ri_direction.delta_total_hamiltonian;
      pair_direction.delta_cofactor_1st =
          ri_direction.delta_first_cofactor;
      pair_direction.delta_same_spin_overlap_hamiltonian_gradient =
          ri_direction.delta_overlap_hamiltonian_gradient;
      if (build_ri_projected_channels) {
        const int channel_work = left_local + left_size * right_local;
        tile.ri_projected_channels.row(channel_work) =
            ri_direction.delta_projected_first_cofactor.transpose();
        tile.ri_projected_ready[static_cast<std::size_t>(channel_work)] = 1;
      }
    } else {
      if (use_ri) {
        const Eigen::MatrixXd interaction_direction =
            build_ri_spin_antisymmetrized_interaction_direction(
                unique_determinants[canonical_left],
                unique_determinants[canonical_right],
                n_active_orbitals,
                *accepted_ri_active_pair_factors,
                *directional_ri_active_pair_factors);
        pair_direction = build_polynomial_spin_directional_data_impl(
            unique_determinants[canonical_left],
            unique_determinants[canonical_right],
            pair_evaluation,
            n_active_orbitals,
            direction,
            interaction_direction,
            true);
      } else {
        pair_direction = build_polynomial_spin_directional_data(
            unique_determinants[canonical_left],
            unique_determinants[canonical_right],
            pair_evaluation,
            n_active_orbitals,
            direction);
      }
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
  if (diagonal_tile) {
    for (int left_local = 1; left_local < left_size; ++left_local) {
      for (int right_local = 0; right_local < left_local; ++right_local) {
        const auto& source = tile.pair(right_local, left_local);
        auto reverse = source;
        reverse.delta_cofactor_1st.transposeInPlace();
        reverse.delta_same_spin_overlap_hamiltonian_gradient
            .transposeInPlace();
        tile.pairs[static_cast<std::size_t>(left_local) * right_size +
            right_local] = std::move(reverse);
        tile.delta_overlap(left_local, right_local) =
            tile.delta_overlap(right_local, left_local);
        tile.delta_regular_hamiltonian(left_local, right_local) =
            tile.delta_regular_hamiltonian(right_local, left_local);
        tile.delta_singular_hamiltonian(left_local, right_local) =
            tile.delta_singular_hamiltonian(right_local, left_local);
        if (!tile.ri_projected_ready.empty()) {
          const int source_work = right_local + left_size * left_local;
          const int target_work = left_local + left_size * right_local;
          tile.ri_projected_channels.row(target_work) =
              tile.ri_projected_channels.row(source_work);
          tile.ri_projected_ready[static_cast<std::size_t>(target_work)] =
              tile.ri_projected_ready[static_cast<std::size_t>(source_work)];
        }
      }
    }
  }
  return tile;
}

}  // namespace

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
    const Eigen::MatrixXd* directional_ri_active_pair_factors,
    bool build_ri_projected_channels) {
  return build_directional_pair_tile_impl(
      unique_determinants,
      &ordered_pair_cache,
      nullptr,
      n_unique_determinants,
      n_active_orbitals,
      direction,
      left_begin,
      left_end,
      right_begin,
      right_end,
      accepted_active_one_electron,
      accepted_ri_active_pair_factors,
      directional_ri_active_pair_factors,
      build_ri_projected_channels);
}

SameSpinDirectionalPairTile build_directional_pair_tile(
    const std::vector<std::vector<int>>& unique_determinants,
    const AcceptedSpinPairTile& accepted_pair_tile,
    int n_active_orbitals,
    const ActiveSpaceIntegralDirectionView& direction,
    const Eigen::MatrixXd* accepted_active_one_electron,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors,
    const Eigen::MatrixXd* directional_ri_active_pair_factors,
    bool build_ri_projected_channels) {
  return build_directional_pair_tile_impl(
      unique_determinants,
      nullptr,
      &accepted_pair_tile,
      static_cast<int>(unique_determinants.size()),
      n_active_orbitals,
      direction,
      accepted_pair_tile.left_begin,
      accepted_pair_tile.left_begin + accepted_pair_tile.left_size,
      accepted_pair_tile.right_begin,
      accepted_pair_tile.right_begin + accepted_pair_tile.right_size,
      accepted_active_one_electron,
      accepted_ri_active_pair_factors,
      directional_ri_active_pair_factors,
      build_ri_projected_channels);
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
