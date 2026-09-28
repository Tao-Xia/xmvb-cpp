#include "vbscf/determinants/pairs/accepted_tile.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include "core/openmp.hpp"
#include "vbscf/determinants/algebra/cofactor_differential.hpp"
#include "vbscf/determinants/pairs/contractions.hpp"
#include "vbscf/determinants/pairs/traversal.hpp"
#include "vbscf/determinants/pairs/woodbury_overlap.hpp"
#include "vbscf/determinants/pairs/woodbury_ri.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"

namespace xmvb::vb {

namespace {

Eigen::MatrixXd occupied_one_electron_block(
    const std::vector<int>& occupied_left,
    const std::vector<int>& occupied_right,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron) {
  Eigen::MatrixXd result(occupied_right.size(), occupied_left.size());
  for (int left = 0; left < static_cast<int>(occupied_left.size()); ++left) {
    for (int right = 0;
         right < static_cast<int>(occupied_right.size());
         ++right) {
      result(right, left) = active_one_electron(
          occupied_right[right], occupied_left[left]);
    }
  }
  return result;
}

SpinDeterminantPairEvaluation evaluate_woodbury_ri_pair(
    const std::vector<int>& occupied_left,
    const std::vector<int>& occupied_right,
    Eigen::MatrixXd overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron,
    const Eigen::Ref<const Eigen::MatrixXd>& ri_factors,
    const WoodburyRiState& state,
    bool populate_response_payload) {
  SpinDeterminantPairEvaluation result;
  if (populate_response_payload && state.core_rank() == 0) {
    const DeterminantOverlapResolver resolver;
    result.overlap_result = resolver.resolve_matrix(overlap);
    result.overlap_result.first_order_cofactor_matrix =
        result.overlap_result.overlap_determinant *
        result.overlap_result.inverse_overlap_submatrix.transpose();
  } else {
    result.overlap_result.n_electrons = overlap.rows();
    result.overlap_result.overlap_submatrix = std::move(overlap);
    result.overlap_result.overlap_determinant = state.overlap_determinant();
    result.overlap_result.nullity = state.overlap_nullity();
    result.overlap_result.first_order_cofactor_matrix =
        state.first_cofactor();
    if (state.core_rank() == 0) {
      result.overlap_result.inverse_overlap_submatrix =
          state.regular_inverse();
    }
    const double determinant = result.overlap_result.overlap_determinant;
    if (determinant == 0.0) {
      result.overlap_result.determinant_sign = 0.0;
      result.overlap_result.log_abs_determinant =
          -std::numeric_limits<double>::infinity();
    } else {
      result.overlap_result.determinant_sign =
          std::signbit(determinant) ? -1.0 : 1.0;
      result.overlap_result.log_abs_determinant =
          std::log(std::abs(determinant));
    }
  }
  const Eigen::MatrixXd one_electron = occupied_one_electron_block(
      occupied_left, occupied_right, active_one_electron);
  result.one_electron_hamiltonian =
      (result.overlap_result.first_order_cofactor_matrix
           .cwiseProduct(one_electron))
          .sum();
  const double state_determinant = state.overlap_determinant();
  const double two_electron =
      populate_response_payload && state.core_rank() == 0 &&
              state_determinant != 0.0
          ? result.overlap_result.overlap_determinant *
                state.two_electron_contraction() / state_determinant
          : state.two_electron_contraction();
  result.total_hamiltonian =
      result.one_electron_hamiltonian +
      two_electron;
  if (populate_response_payload && state.core_rank() != 0) {
    result.cofactor_differential =
        std::make_shared<const CofactorDifferential>(result.overlap_result);
    result.has_woodbury_ri_response = true;
    result.same_spin_overlap_hamiltonian_gradient =
        state.hamiltonian_overlap_gradient(one_electron, ri_factors);
  }
  return result;
}

std::optional<RegularRiPairResponseData> regular_ri_response(
    const WoodburyRiState& state,
    bool requested) {
  if (!requested || state.core_rank() != 0 ||
      state.overlap_determinant() == 0.0) {
    return std::nullopt;
  }
  RegularRiPairResponseData result;
  result.two_electron_phi =
      state.two_electron_contraction() / state.overlap_determinant();
  result.two_electron_inverse_overlap_gradient =
      state.regular_two_electron_inverse_gradient();
  return result;
}

}  // namespace

const SpinDeterminantPairEvaluation& AcceptedSpinPairTile::pair(
    int left_local,
    int right_local) const {
  if (left_local < 0 || left_local >= left_size ||
      right_local < 0 || right_local >= right_size) {
    throw std::out_of_range("accepted pair tile index is out of range");
  }
  return pairs[static_cast<std::size_t>(left_local) * right_size +
      right_local];
}

AcceptedPairTileProvider::AcceptedPairTileProvider(
    std::vector<std::vector<int>> unique_spin_strings,
    int n_active_orbitals,
    double linear_dependence_threshold)
    : unique_spin_strings_(std::move(unique_spin_strings)),
      n_active_orbitals_(n_active_orbitals),
      overlap_resolver_(linear_dependence_threshold),
      pair_evaluator_(overlap_resolver_,
                      DeterminantHamiltonianResolver(overlap_resolver_)) {}

AcceptedSpinPairTile AcceptedPairTileProvider::build(
    int left_begin,
    int left_end,
    int right_begin,
    int right_end,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_electron,
    const ActiveSpaceTwoElectronResult& active_two_electron,
    AcceptedPairTileBuildOptions options) const {
  if (left_begin < 0 || left_end <= left_begin || left_end > size() ||
      right_begin < 0 || right_end <= right_begin || right_end > size()) {
    throw std::invalid_argument("accepted pair tile bounds are invalid");
  }
  if (active_one_electron.rows() != n_active_orbitals_ ||
      active_one_electron.cols() != n_active_orbitals_ ||
      active_overlap.size() != static_cast<std::size_t>(n_active_orbitals_) *
          n_active_orbitals_) {
    throw std::invalid_argument(
        "accepted pair tile active-space dimensions differ");
  }

  AcceptedSpinPairTile tile;
  tile.left_begin = left_begin;
  tile.right_begin = right_begin;
  tile.left_size = left_end - left_begin;
  tile.right_size = right_end - right_begin;
  tile.pairs.resize(
      static_cast<std::size_t>(tile.left_size) * tile.right_size);

  const Eigen::Map<const Eigen::MatrixXd> overlap_map(
      active_overlap.data(), n_active_orbitals_, n_active_orbitals_);
  const bool direct_ri = uses_direct_ri_pair_factors(active_two_electron);
  if (direct_ri &&
      active_two_electron.ri_active_pair_factors.cols() !=
          packed_active_pair_count(n_active_orbitals_)) {
    throw std::invalid_argument(
        "RI active-pair factor dimensions are inconsistent");
  }
  const int n_threads = std::max(
      1, std::min(xmvb::effective_openmp_thread_count(), tile.left_size));

  // Accepted-point consumers that do not retain projected RI images traverse
  // their full rectangular tile as a product graph: right edges change
  // overlap rows and left edges change columns.  Scalar and compact response
  // payloads share the same certified path.  One exact anchor per worker
  // replaces one anchor per tile row.
  if (direct_ri && !options.materialize_projected_pair_values) {
#pragma omp parallel if(n_threads > 1) num_threads(n_threads)
    {
      int thread = 0;
#ifdef _OPENMP
      thread = omp_get_thread_num();
#endif
      const int local_left_begin =
          tile.left_size * thread / n_threads;
      const int local_left_end =
          tile.left_size * (thread + 1) / n_threads;
      if (local_left_begin < local_left_end) {
        const int global_left_begin = left_begin + local_left_begin;
        const int global_left_end = left_begin + local_left_end;
        const std::vector<int> left_traversal = build_pair_update_traversal(
            unique_spin_strings_,
            global_left_begin,
            global_left_begin,
            global_left_end);
        const std::vector<int> right_traversal = build_pair_update_traversal(
            unique_spin_strings_, right_begin, right_begin, right_end);

        WoodburyRiState ri_state;
        bool have_previous = false;

        const auto evaluate_pair = [&](
            int left_index,
            int right_index,
            bool left_edge) {
          const auto& occupied_left = unique_spin_strings_[left_index];
          const auto& occupied_right = unique_spin_strings_[right_index];
          Eigen::MatrixXd overlap = build_overlap_submatrix(
              occupied_left, occupied_right, overlap_map);
          bool updated = false;
          if (have_previous && ri_state.core_rank() == 0) {
            updated = left_edge
                ? ri_state.update_left(
                      occupied_left,
                      overlap,
                      active_two_electron.ri_active_pair_factors)
                : ri_state.update_right(
                      occupied_right,
                      overlap,
                      active_two_electron.ri_active_pair_factors);
          }
          if (!updated && !ri_state.initialize(
                  occupied_left,
                  occupied_right,
                  overlap,
                  active_two_electron.ri_active_pair_factors)) {
            throw std::runtime_error(
                "failed to initialize Woodbury RI pair state");
          }

          SpinDeterminantPairEvaluation evaluation =
              evaluate_woodbury_ri_pair(
                  occupied_left,
                  occupied_right,
                  std::move(overlap),
                  active_one_electron,
                  active_two_electron.ri_active_pair_factors,
                  ri_state,
                  options.populate_response_payload);
          if (options.populate_opposite_spin_projection ||
              options.populate_response_payload) {
            const auto ri_response = regular_ri_response(
                ri_state, options.populate_response_payload);
            complete_same_spin_pair_evaluation(
                occupied_left,
                occupied_right,
                active_one_electron,
                n_active_orbitals_,
                active_two_electron,
                options.populate_opposite_spin_projection,
                false,
                options.populate_response_payload,
                &evaluation,
                ri_response ? &*ri_response : nullptr);
          }
          tile.pairs[
              static_cast<std::size_t>(left_index - left_begin) *
                  tile.right_size +
              right_index - right_begin] = std::move(evaluation);
          have_previous = true;
        };

        for (int left_position = 0;
             left_position < static_cast<int>(left_traversal.size());
             ++left_position) {
          const int left_index = left_traversal[left_position];
          const bool reverse = left_position % 2 != 0;
          if (!reverse) {
            for (int right_position = 0;
                 right_position < static_cast<int>(right_traversal.size());
                 ++right_position) {
              evaluate_pair(
                  left_index,
                  right_traversal[right_position],
                  right_position == 0 && left_position != 0);
            }
          } else {
            for (int right_position =
                     static_cast<int>(right_traversal.size()) - 1;
                 right_position >= 0;
                 --right_position) {
              evaluate_pair(
                  left_index,
                  right_traversal[right_position],
                  right_position ==
                      static_cast<int>(right_traversal.size()) - 1);
            }
          }
        }
      }
    }
    return tile;
  }

  if (direct_ri && options.materialize_projected_pair_values) {
#pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
    for (int left_local = 0; left_local < tile.left_size; ++left_local) {
      const int left_index = left_begin + left_local;
      const auto& occupied_left = unique_spin_strings_[left_index];
      const std::vector<int> right_traversal = build_pair_update_traversal(
          unique_spin_strings_, left_index, right_begin, right_end);
      WoodburyRiState ri_state;
      Eigen::MatrixXd auxiliary_panel(
          active_two_electron.ri_active_pair_factors.rows(),
          tile.right_size);

      for (int traversal_index = 0;
           traversal_index < tile.right_size;
           ++traversal_index) {
        const int right_index = right_traversal[traversal_index];
        const int right_local = right_index - right_begin;
        const auto& occupied_right = unique_spin_strings_[right_index];
        Eigen::MatrixXd overlap = build_overlap_submatrix(
            occupied_left, occupied_right, overlap_map);
        const bool updated = traversal_index > 0 &&
            ri_state.core_rank() == 0 && ri_state.update_right(
            occupied_right,
            overlap,
            active_two_electron.ri_active_pair_factors);
        if (!updated && !ri_state.initialize(
                occupied_left,
                occupied_right,
                overlap,
                active_two_electron.ri_active_pair_factors)) {
          throw std::runtime_error(
              "failed to initialize projected Woodbury RI pair state");
        }

        SpinDeterminantPairEvaluation evaluation =
            evaluate_woodbury_ri_pair(
                occupied_left,
                occupied_right,
                std::move(overlap),
                active_one_electron,
                active_two_electron.ri_active_pair_factors,
                ri_state,
                options.populate_response_payload);
        const auto ri_response = regular_ri_response(
            ri_state, options.populate_response_payload);
        complete_same_spin_pair_evaluation(
            occupied_left,
            occupied_right,
            active_one_electron,
            n_active_orbitals_,
            active_two_electron,
            true,
            false,
            options.populate_response_payload,
            &evaluation,
            ri_response ? &*ri_response : nullptr);
        auxiliary_panel.col(right_local) =
            ri_state.first_cofactor_auxiliary();
        tile.pairs[
            static_cast<std::size_t>(left_local) * tile.right_size +
            right_local] = std::move(evaluation);
      }

      const Eigen::MatrixXd projected_panel =
          active_two_electron.ri_active_pair_factors.transpose() *
          auxiliary_panel;
      for (int right_local = 0;
           right_local < tile.right_size;
           ++right_local) {
        auto& projected = tile.pairs[
            static_cast<std::size_t>(left_local) * tile.right_size +
            right_local]
                              .opposite_spin_pair_cache
                              .first_order_cofactor_projection
                              .projected_pair_values;
        projected.assign(
            projected_panel.col(right_local).data(),
            projected_panel.col(right_local).data() +
                projected_panel.rows());
      }
    }
    return tile;
  }

#pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
  for (int left_local = 0; left_local < tile.left_size; ++left_local) {
    const int left_index = left_begin + left_local;
    const std::vector<int> right_traversal = build_pair_update_traversal(
        unique_spin_strings_, left_index, right_begin, right_end);
    for (int traversal_index = 0;
         traversal_index < tile.right_size;
         ++traversal_index) {
      const int right_index = right_traversal[traversal_index];
      const int right_local = right_index - right_begin;
      const auto& occupied_left =
          unique_spin_strings_[left_index];
      const auto& occupied_right =
          unique_spin_strings_[right_index];
      const std::size_t pair_index =
          static_cast<std::size_t>(left_local) * tile.right_size + right_local;
      DeterminantOverlapResult overlap_result;
      int previous_right = -1;
      std::size_t previous_index = 0;
      if (traversal_index == 0) {
        overlap_result = overlap_resolver_.resolve_matrix(
            build_overlap_submatrix(
                occupied_left, occupied_right, overlap_map));
      } else {
        previous_right = right_traversal[traversal_index - 1];
        previous_index =
            static_cast<std::size_t>(left_local) * tile.right_size +
            previous_right - right_begin;
        auto updated_overlap = try_woodbury_right_overlap_update(
            occupied_left,
            unique_spin_strings_[previous_right],
            occupied_right,
            overlap_map,
            tile.pairs[previous_index].overlap_result);
        if (updated_overlap.has_value()) {
          overlap_result = std::move(*updated_overlap);
        } else {
          overlap_result = overlap_resolver_.resolve_matrix(
              build_overlap_submatrix(
                  occupied_left, occupied_right, overlap_map));
        }
      }

      SpinDeterminantPairEvaluation evaluation =
          pair_evaluator_.evaluate_same_spin_pair(
              occupied_left,
              occupied_right,
              std::move(overlap_result),
              active_one_electron,
              n_active_orbitals_,
              active_two_electron,
              options.populate_response_payload);
      if (options.populate_opposite_spin_projection ||
          options.populate_response_payload) {
        complete_same_spin_pair_evaluation(
            occupied_left,
            occupied_right,
            active_one_electron,
            n_active_orbitals_,
            active_two_electron,
            options.populate_opposite_spin_projection,
            options.materialize_projected_pair_values,
            options.populate_response_payload,
            &evaluation);
      }
      tile.pairs[pair_index] = std::move(evaluation);
    }
  }
  return tile;
}

}  // namespace xmvb::vb
