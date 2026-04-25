#include "vb/scf/selected_state_determinant_matrices.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "vb/matrices/determinant_pair_storage_utils.hpp"

namespace xmvb::vb {

namespace {

constexpr double kNormalizedWeightTolerance = 1e-10;
constexpr double kSupportSparseContractionSavingsThreshold = 0.8;

enum class SupportSparseSelectedStateMode {
  kAuto,
  kOn,
  kOff,
};

SupportSparseSelectedStateMode selected_state_support_sparse_mode() {
  const char* env_value = std::getenv("XMVB_CPP_SELECTED_STATE_SUPPORT_SPARSE");
  if (env_value == nullptr || env_value[0] == '\0') {
    return SupportSparseSelectedStateMode::kAuto;
  }
  std::string mode(env_value);
  std::transform(
      mode.begin(),
      mode.end(),
      mode.begin(),
      [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
  if (mode == "auto") {
    return SupportSparseSelectedStateMode::kAuto;
  }
  if (mode == "on" || mode == "true" || mode == "1") {
    return SupportSparseSelectedStateMode::kOn;
  }
  if (mode == "off" || mode == "false" || mode == "0") {
    return SupportSparseSelectedStateMode::kOff;
  }
  throw std::invalid_argument(
      "XMVB_CPP_SELECTED_STATE_SUPPORT_SPARSE must be one of auto/on/off");
}

double estimate_dense_selected_state_contraction_work(
    const SelectedStateDeterminantMatrices& selected_state_matrices) {
  const double n_unique_alpha =
      static_cast<double>(selected_state_matrices.n_unique_alpha);
  const double n_unique_beta =
      static_cast<double>(selected_state_matrices.n_unique_beta);
  const double per_state_work =
      n_unique_alpha * n_unique_beta * (n_unique_alpha + n_unique_beta);
  return per_state_work *
      static_cast<double>(selected_state_matrices.states.size());
}

double estimate_support_sparse_selected_state_contraction_work(
    const SelectedStateDeterminantMatrices& selected_state_matrices) {
  double total_work = 0.0;
  for (const auto& state_coefficients : selected_state_matrices.states) {
    const double alpha_support_size =
        static_cast<double>(state_coefficients.alpha_support.size());
    const double beta_support_size =
        static_cast<double>(state_coefficients.beta_support.size());
    if (alpha_support_size == 0.0 || beta_support_size == 0.0) {
      continue;
    }

    // The trimmed contraction still needs local kernel gather/scatter, but it
    // avoids the dense BLAS sweep on the full unique-spin product space.
    total_work +=
        alpha_support_size * beta_support_size *
            (alpha_support_size + beta_support_size) +
        alpha_support_size * alpha_support_size +
        beta_support_size * beta_support_size;
  }
  return total_work;
}

bool selected_state_support_is_actually_trimmed(
    const SelectedStateDeterminantMatrices& selected_state_matrices) {
  for (const auto& state_coefficients : selected_state_matrices.states) {
    if (static_cast<int>(state_coefficients.alpha_support.size()) <
            selected_state_matrices.n_unique_alpha ||
        static_cast<int>(state_coefficients.beta_support.size()) <
            selected_state_matrices.n_unique_beta) {
      return true;
    }
  }
  return false;
}

void validate_selected_state_indices(
    const std::vector<int>& selected_state_indices,
    int n_structures) {
  if (selected_state_indices.empty()) {
    throw std::invalid_argument("selected_state_indices must not be empty");
  }
  for (const int state_index : selected_state_indices) {
    if (state_index < 0 || state_index >= n_structures) {
      throw std::out_of_range("selected state index is out of range");
    }
  }
}

void validate_nonnegative_weights(
    const std::vector<double>& state_average_weights) {
  if (state_average_weights.empty()) {
    throw std::invalid_argument("state_average_weights must not be empty");
  }
  for (const double state_weight : state_average_weights) {
    if (state_weight < 0.0) {
      throw std::invalid_argument("state_average_weights must be non-negative");
    }
  }
}

void validate_input_shapes(
    const FullDeterminantStructureData& full_determinant_data,
    const std::vector<double>& eigenvector_matrix,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    const SameSpinPairCacheContext& same_spin_pair_cache) {
  const int n_structures = full_determinant_data.n_structures;
  if (n_structures <= 0) {
    throw std::invalid_argument("full_determinant_data.n_structures must be positive");
  }

  const std::size_t n_determinants = full_determinant_data.alpha_det.size();
  if (n_determinants == 0) {
    throw std::invalid_argument("full_determinant_data must contain at least one determinant");
  }
  if (full_determinant_data.beta_det.size() != n_determinants ||
      full_determinant_data.determinant_to_structure_terms.size() != n_determinants) {
    throw std::invalid_argument(
        "alpha_det, beta_det, and determinant_to_structure_terms sizes must match");
  }

  const std::size_t expected_eigenvector_size =
      n_structures * n_structures;
  if (eigenvector_matrix.size() != expected_eigenvector_size) {
    throw std::invalid_argument("eigenvector_matrix size does not match n_structures^2");
  }

  if (selected_state_indices.size() != state_average_weights.size()) {
    throw std::invalid_argument(
        "selected_state_indices and state_average_weights must have the same length");
  }
  validate_selected_state_indices(selected_state_indices, n_structures);
  validate_nonnegative_weights(state_average_weights);

  if (same_spin_pair_cache.alpha_reuse_table.determinant_to_unique_id.size() != n_determinants ||
      same_spin_pair_cache.beta_reuse_table.determinant_to_unique_id.size() != n_determinants) {
    throw std::invalid_argument(
        "same_spin_pair_cache determinant_to_unique_id sizes must match n_determinants");
  }
  if (same_spin_pair_cache.alpha_reuse_table.unique_determinants.empty() ||
      same_spin_pair_cache.beta_reuse_table.unique_determinants.empty()) {
    throw std::invalid_argument(
        "same_spin_pair_cache unique_determinants must not be empty");
  }
}

void validate_selected_state_column_matrix_input_shapes(
    const FullDeterminantStructureData& full_determinant_data,
    const Eigen::MatrixXd& selected_state_columns,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    const SameSpinPairCacheContext& same_spin_pair_cache) {
  const int n_structures = full_determinant_data.n_structures;
  if (n_structures <= 0) {
    throw std::invalid_argument("full_determinant_data.n_structures must be positive");
  }
  const std::size_t n_determinants = full_determinant_data.alpha_det.size();
  if (n_determinants == 0) {
    throw std::invalid_argument("full_determinant_data must contain at least one determinant");
  }
  if (full_determinant_data.beta_det.size() != n_determinants ||
      full_determinant_data.determinant_to_structure_terms.size() != n_determinants) {
    throw std::invalid_argument(
        "alpha_det, beta_det, and determinant_to_structure_terms sizes must match");
  }
  if (selected_state_columns.rows() != n_structures ||
      selected_state_columns.cols() !=
          static_cast<int>(selected_state_indices.size())) {
    throw std::invalid_argument(
        "selected_state_columns shape must match "
        "(n_structures, selected_state_indices.size())");
  }
  if (selected_state_indices.size() != state_average_weights.size()) {
    throw std::invalid_argument(
        "selected_state_indices and state_average_weights must have the same length");
  }
  validate_selected_state_indices(selected_state_indices, n_structures);
  validate_nonnegative_weights(state_average_weights);

  if (same_spin_pair_cache.alpha_reuse_table.determinant_to_unique_id.size() !=
          n_determinants ||
      same_spin_pair_cache.beta_reuse_table.determinant_to_unique_id.size() !=
          n_determinants) {
    throw std::invalid_argument(
        "same_spin_pair_cache determinant_to_unique_id sizes must match n_determinants");
  }
  if (same_spin_pair_cache.alpha_reuse_table.unique_determinants.empty() ||
      same_spin_pair_cache.beta_reuse_table.unique_determinants.empty()) {
    throw std::invalid_argument(
        "same_spin_pair_cache unique_determinants must not be empty");
  }
}

void validate_normalized_weights(
    const std::vector<double>& normalized_state_weights) {
  validate_nonnegative_weights(normalized_state_weights);
  double weight_sum = 0.0;
  for (const double state_weight : normalized_state_weights) {
    weight_sum += state_weight;
  }
  if (weight_sum <= 0.0) {
    throw std::invalid_argument("normalized_state_weights must sum to a positive value");
  }
  if (std::fabs(weight_sum - 1.0) > kNormalizedWeightTolerance) {
    throw std::invalid_argument(
        "normalized_state_weights must sum to 1 within tolerance");
  }
}

template <typename ColumnProvider>
SelectedStateDeterminantMatrices
build_selected_state_determinant_matrices_from_column_provider_impl(
    const FullDeterminantStructureData& full_determinant_data,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& normalized_state_weights,
    const SameSpinPairCacheContext& same_spin_pair_cache,
    ColumnProvider&& column_provider) {
  const int n_structures = full_determinant_data.n_structures;
  const int n_determinants = static_cast<int>(full_determinant_data.alpha_det.size());
  const int n_unique_alpha = static_cast<int>(
      same_spin_pair_cache.alpha_reuse_table.unique_determinants.size());
  const int n_unique_beta = static_cast<int>(
      same_spin_pair_cache.beta_reuse_table.unique_determinants.size());
  const bool close_shell_diagonal =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();

  SelectedStateDeterminantMatrices result;
  result.n_structures = n_structures;
  result.n_determinants = n_determinants;
  result.n_unique_alpha = n_unique_alpha;
  result.n_unique_beta = n_unique_beta;
  result.selected_state_indices = selected_state_indices;
  result.normalized_state_weights = normalized_state_weights;
  result.determinant_to_unique_alpha_id =
      same_spin_pair_cache.alpha_reuse_table.determinant_to_unique_id;
  result.determinant_to_unique_beta_id =
      same_spin_pair_cache.beta_reuse_table.determinant_to_unique_id;
  result.states.reserve(selected_state_indices.size());

  // Build one determinant coefficient vector c_d^(n) per selected state and
  // then scatter it into the unique-spin matrix C^(n)[alpha_id, beta_id].
  for (std::size_t selected_state_offset = 0;
       selected_state_offset < selected_state_indices.size();
       ++selected_state_offset) {
    const int state_index = selected_state_indices[selected_state_offset];
    const double* selected_state_column =
        column_provider(selected_state_offset, state_index);
    if (selected_state_column == nullptr) {
      throw std::invalid_argument(
          "selected-state column provider returned a null column");
    }

    SelectedStateDeterminantCoefficients state_coefficients;
    state_coefficients.state_index = state_index;
    state_coefficients.normalized_state_weight =
        normalized_state_weights[selected_state_offset];
    state_coefficients.determinant_coefficients.assign(
        n_determinants,
        0.0);
    state_coefficients.close_shell_diagonal = close_shell_diagonal;
    state_coefficients.coefficient_matrix =
        Eigen::MatrixXd::Zero(n_unique_alpha, n_unique_beta);
    std::vector<std::size_t> touched_pair_indices;
    touched_pair_indices.reserve(n_determinants);
    std::vector<int> touched_diagonal_indices;
    if (close_shell_diagonal) {
      state_coefficients.diagonal_coefficients.assign(
          n_unique_alpha,
          0.0);
      touched_diagonal_indices.reserve(n_determinants);
    }

    for (int determinant_index = 0;
         determinant_index < n_determinants;
         ++determinant_index) {
      const auto& structure_terms =
          full_determinant_data.determinant_to_structure_terms[
              determinant_index];
      double coefficient = 0.0;
      for (const auto& term : structure_terms) {
        if (term.structure_index < 0 || term.structure_index >= n_structures) {
          throw std::out_of_range("determinant_to_structure_terms structure index out of range");
        }
        coefficient +=
            term.coefficient *
            selected_state_column[term.structure_index];
      }
      state_coefficients.determinant_coefficients[determinant_index] =
          coefficient;

      const int unique_alpha_id =
          result.determinant_to_unique_alpha_id[determinant_index];
      const int unique_beta_id =
          result.determinant_to_unique_beta_id[determinant_index];
      if (unique_alpha_id < 0 || unique_alpha_id >= n_unique_alpha ||
          unique_beta_id < 0 || unique_beta_id >= n_unique_beta) {
        throw std::out_of_range("determinant-to-unique spin id is out of range");
      }
      // In the expected full-determinant space each (alpha,beta) pair is unique.
      // We still accumulate defensively in case an upstream caller keeps
      // duplicate determinant rows mapped to the same unique spin pair.
      if (close_shell_diagonal) {
        if (unique_alpha_id != unique_beta_id) {
          throw std::runtime_error(
              "close-shell selected-state compression requires alpha and beta unique ids to match");
        }
        state_coefficients.diagonal_coefficients[unique_alpha_id] +=
            coefficient;
        touched_diagonal_indices.push_back(unique_alpha_id);
      } else {
        state_coefficients.coefficient_matrix(unique_alpha_id, unique_beta_id) +=
            coefficient;
        touched_pair_indices.push_back(
            unique_beta_id * n_unique_alpha +
            unique_alpha_id);
      }
    }

    // Build the exact trimmed support block for this selected state after the
    // full determinant-to-unique accumulation has completed, so any upstream
    // cancellations are removed before the local alpha/beta supports are fixed.
    // Iterate only the touched unique-spin pairs instead of rescanning the full
    // `n_unique_alpha x n_unique_beta` matrix on every selected-state rebuild.
    std::vector<unsigned char> alpha_has_support(
        n_unique_alpha,
        0u);
    std::vector<unsigned char> beta_has_support(
        n_unique_beta,
        0u);
    if (close_shell_diagonal) {
      std::sort(
          touched_diagonal_indices.begin(),
          touched_diagonal_indices.end());
      touched_diagonal_indices.erase(
          std::unique(
              touched_diagonal_indices.begin(),
              touched_diagonal_indices.end()),
          touched_diagonal_indices.end());
      for (const int unique_id : touched_diagonal_indices) {
        const double coefficient =
            state_coefficients.diagonal_coefficients[unique_id];
        if (coefficient == 0.0) {
          continue;
        }
        alpha_has_support[unique_id] = 1u;
        beta_has_support[unique_id] = 1u;
        state_coefficients.coefficient_matrix(unique_id, unique_id) =
            coefficient;
        ++state_coefficients.nonzero_coefficient_count;
      }
    } else {
      std::sort(touched_pair_indices.begin(), touched_pair_indices.end());
      touched_pair_indices.erase(
          std::unique(touched_pair_indices.begin(), touched_pair_indices.end()),
          touched_pair_indices.end());
      for (const std::size_t touched_pair_index : touched_pair_indices) {
        const int unique_alpha_id = static_cast<int>(
            touched_pair_index % n_unique_alpha);
        const int unique_beta_id = static_cast<int>(
            touched_pair_index / n_unique_alpha);
        const double coefficient =
            state_coefficients.coefficient_matrix(unique_alpha_id, unique_beta_id);
        if (coefficient == 0.0) {
          continue;
        }
        alpha_has_support[unique_alpha_id] = 1u;
        beta_has_support[unique_beta_id] = 1u;
        ++state_coefficients.nonzero_coefficient_count;
      }
    }

    std::vector<int> alpha_global_to_local(
        n_unique_alpha,
        -1);
    std::vector<int> beta_global_to_local(
        n_unique_beta,
        -1);
    for (int unique_alpha_id = 0;
         unique_alpha_id < n_unique_alpha;
         ++unique_alpha_id) {
      if (alpha_has_support[unique_alpha_id] == 0u) {
        continue;
      }
      alpha_global_to_local[unique_alpha_id] =
          static_cast<int>(state_coefficients.alpha_support.size());
      state_coefficients.alpha_support.push_back(unique_alpha_id);
    }
    for (int unique_beta_id = 0;
         unique_beta_id < n_unique_beta;
         ++unique_beta_id) {
      if (beta_has_support[unique_beta_id] == 0u) {
        continue;
      }
      beta_global_to_local[unique_beta_id] =
          static_cast<int>(state_coefficients.beta_support.size());
      state_coefficients.beta_support.push_back(unique_beta_id);
    }

    state_coefficients.local_coefficient_matrix =
        Eigen::MatrixXd::Zero(
            static_cast<int>(state_coefficients.alpha_support.size()),
            static_cast<int>(state_coefficients.beta_support.size()));
    if (close_shell_diagonal) {
      state_coefficients.local_diagonal_coefficients.assign(
          state_coefficients.alpha_support.size(),
          0.0);
      for (int local_index = 0;
           local_index < static_cast<int>(state_coefficients.alpha_support.size());
           ++local_index) {
        const int unique_id =
            state_coefficients.alpha_support[local_index];
        const double coefficient =
            state_coefficients.diagonal_coefficients[unique_id];
        state_coefficients.local_diagonal_coefficients[local_index] =
            coefficient;
        state_coefficients.local_coefficient_matrix(local_index, local_index) =
            coefficient;
      }
    } else {
      for (const std::size_t touched_pair_index : touched_pair_indices) {
        const int unique_alpha_id = static_cast<int>(
            touched_pair_index % n_unique_alpha);
        const int unique_beta_id = static_cast<int>(
            touched_pair_index / n_unique_alpha);
        const int alpha_local =
            alpha_global_to_local[unique_alpha_id];
        const int beta_local =
            beta_global_to_local[unique_beta_id];
        if (alpha_local < 0 || beta_local < 0) {
          continue;
        }
        state_coefficients.local_coefficient_matrix(alpha_local, beta_local) =
            state_coefficients.coefficient_matrix(unique_alpha_id, unique_beta_id);
      }
    }

    result.states.push_back(std::move(state_coefficients));
  }

  return result;
}

SelectedStateDeterminantMatrices build_selected_state_determinant_matrices_impl(
    const FullDeterminantStructureData& full_determinant_data,
    const std::vector<double>& eigenvector_matrix,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& normalized_state_weights,
    const SameSpinPairCacheContext& same_spin_pair_cache) {
  const int n_structures = full_determinant_data.n_structures;
  return build_selected_state_determinant_matrices_from_column_provider_impl(
      full_determinant_data,
      selected_state_indices,
      normalized_state_weights,
      same_spin_pair_cache,
      [&eigenvector_matrix, n_structures](
          std::size_t selected_state_offset,
          int state_index) -> const double* {
        (void) selected_state_offset;
        return eigenvector_matrix.data() +
            state_index * n_structures;
      });
}

}  // namespace

std::vector<double> normalize_state_average_weights(
    const std::vector<double>& state_average_weights) {
  validate_nonnegative_weights(state_average_weights);
  double weight_sum = 0.0;
  for (const double state_weight : state_average_weights) {
    weight_sum += state_weight;
  }
  if (weight_sum <= 0.0) {
    throw std::invalid_argument("state_average_weights must sum to a positive value");
  }

  std::vector<double> normalized_weights = state_average_weights;
  for (double& state_weight : normalized_weights) {
    state_weight /= weight_sum;
  }
  return normalized_weights;
}

SelectedStateDeterminantMatrices build_selected_state_determinant_matrices(
    const FullDeterminantStructureData& full_determinant_data,
    const std::vector<double>& eigenvector_matrix,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    const SameSpinPairCacheContext& same_spin_pair_cache) {
  validate_input_shapes(
      full_determinant_data,
      eigenvector_matrix,
      selected_state_indices,
      state_average_weights,
      same_spin_pair_cache);
  const std::vector<double> normalized_state_weights =
      normalize_state_average_weights(state_average_weights);
  return build_selected_state_determinant_matrices_impl(
      full_determinant_data,
      eigenvector_matrix,
      selected_state_indices,
      normalized_state_weights,
      same_spin_pair_cache);
}

SelectedStateDeterminantMatrices
build_selected_state_determinant_matrices_from_normalized_weights(
    const FullDeterminantStructureData& full_determinant_data,
    const std::vector<double>& eigenvector_matrix,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& normalized_state_weights,
    const SameSpinPairCacheContext& same_spin_pair_cache) {
  validate_input_shapes(
      full_determinant_data,
      eigenvector_matrix,
      selected_state_indices,
      normalized_state_weights,
      same_spin_pair_cache);
  validate_normalized_weights(normalized_state_weights);
  return build_selected_state_determinant_matrices_impl(
      full_determinant_data,
      eigenvector_matrix,
      selected_state_indices,
      normalized_state_weights,
      same_spin_pair_cache);
}

SelectedStateDeterminantMatrices
build_selected_state_determinant_matrices_from_selected_columns(
    const FullDeterminantStructureData& full_determinant_data,
    const Eigen::MatrixXd& selected_state_columns,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& normalized_state_weights,
    const SameSpinPairCacheContext& same_spin_pair_cache) {
  validate_selected_state_column_matrix_input_shapes(
      full_determinant_data,
      selected_state_columns,
      selected_state_indices,
      normalized_state_weights,
      same_spin_pair_cache);
  validate_normalized_weights(normalized_state_weights);
  const int n_structures = full_determinant_data.n_structures;
  return build_selected_state_determinant_matrices_from_column_provider_impl(
      full_determinant_data,
      selected_state_indices,
      normalized_state_weights,
      same_spin_pair_cache,
      [&selected_state_columns, n_structures](
          std::size_t selected_state_offset,
          int state_index) -> const double* {
        (void) state_index;
        return selected_state_columns.data() +
            selected_state_offset * n_structures;
      });
}

bool should_use_support_sparse_selected_state_contractions(
    const SelectedStateDeterminantMatrices& selected_state_matrices) {
  const SupportSparseSelectedStateMode mode =
      selected_state_support_sparse_mode();
  if (mode == SupportSparseSelectedStateMode::kOn) {
    return true;
  }
  if (mode == SupportSparseSelectedStateMode::kOff) {
    return false;
  }
  if (!selected_state_support_is_actually_trimmed(selected_state_matrices)) {
    return false;
  }

  const double dense_work =
      estimate_dense_selected_state_contraction_work(selected_state_matrices);
  const double sparse_work =
      estimate_support_sparse_selected_state_contraction_work(
          selected_state_matrices);
  if (dense_work == 0.0) {
    return false;
  }

  // Keep the dense BLAS path when the trimmed supports are not materially
  // smaller. The sparse path pays extra gather/scatter overhead and only wins
  // when the per-state supports cut a meaningful fraction of the contraction.
  return sparse_work <=
      kSupportSparseContractionSavingsThreshold * dense_work;
}

std::vector<double> gather_selected_state_energies(
    const std::vector<double>& eigenvalues,
    const std::vector<int>& selected_state_indices) {
  std::vector<double> selected_state_energies;
  selected_state_energies.reserve(selected_state_indices.size());
  for (const int state_index : selected_state_indices) {
    if (state_index < 0 || state_index >= static_cast<int>(eigenvalues.size())) {
      throw std::out_of_range("selected state index is out of range for eigenvalues");
    }
    selected_state_energies.push_back(eigenvalues[state_index]);
  }
  return selected_state_energies;
}

DeterminantPairWeightTablesFromCoefficients
build_exact_determinant_pair_weight_tables_from_coefficients(
    const SelectedStateDeterminantMatrices& selected_state_matrices,
    const std::vector<double>& selected_state_energies) {
  const int n_determinants = selected_state_matrices.n_determinants;
  if (n_determinants <= 0) {
    throw std::invalid_argument("selected_state_matrices.n_determinants must be positive");
  }
  if (selected_state_matrices.states.size() !=
      selected_state_matrices.selected_state_indices.size()) {
    throw std::invalid_argument(
        "selected_state_matrices states and selected_state_indices size mismatch");
  }
  if (selected_state_matrices.states.size() !=
      selected_state_matrices.normalized_state_weights.size()) {
    throw std::invalid_argument(
        "selected_state_matrices states and normalized_state_weights size mismatch");
  }
  if (selected_state_energies.size() != selected_state_matrices.states.size()) {
    throw std::invalid_argument(
        "selected_state_energies must align with selected_state_matrices.states");
  }

  DeterminantPairWeightTablesFromCoefficients pair_weights;
  pair_weights.n_determinants = n_determinants;
  pair_weights.ordered_hamiltonian_weights.assign(
      n_determinants * n_determinants,
      0.0);
  pair_weights.ordered_overlap_weights.assign(
      n_determinants * n_determinants,
      0.0);

  // Exact selected-state bilinear form:
  //   H-weight(dL,dR) = sum_n w_n c_dL^(n) c_dR^(n)
  //   S-weight(dL,dR) = -sum_n w_n E_n c_dL^(n) c_dR^(n)
  for (std::size_t state_offset = 0;
       state_offset < selected_state_matrices.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_state_matrices.states[state_offset];
    if (state_coefficients.determinant_coefficients.size() !=
        n_determinants) {
      throw std::invalid_argument(
          "state determinant_coefficients size does not match n_determinants");
    }
    const double state_weight =
        selected_state_matrices.normalized_state_weights[state_offset];
    const double state_energy = selected_state_energies[state_offset];
    const double hamiltonian_prefactor = state_weight;
    const double overlap_prefactor = -state_weight * state_energy;

    for (int determinant_index_left = 0;
         determinant_index_left < n_determinants;
         ++determinant_index_left) {
      const double coefficient_left =
          state_coefficients.determinant_coefficients[
              determinant_index_left];
      const std::size_t row_offset =
          determinant_index_left * n_determinants;
      for (int determinant_index_right = 0;
           determinant_index_right < n_determinants;
           ++determinant_index_right) {
        const double pair_product =
            coefficient_left *
            state_coefficients.determinant_coefficients[
                determinant_index_right];
        const std::size_t ordered_index =
            row_offset + determinant_index_right;
        pair_weights.ordered_hamiltonian_weights[ordered_index] +=
            hamiltonian_prefactor * pair_product;
        pair_weights.ordered_overlap_weights[ordered_index] +=
            overlap_prefactor * pair_product;
      }
    }
  }

  const std::size_t n_unordered_pairs =
      unordered_determinant_pair_count(n_determinants);
  pair_weights.unordered_combined_hamiltonian_weights.assign(n_unordered_pairs, 0.0);
  pair_weights.unordered_combined_overlap_weights.assign(n_unordered_pairs, 0.0);

  for (int determinant_index_left = 0;
       determinant_index_left < n_determinants;
       ++determinant_index_left) {
    const std::size_t left_row_offset =
        determinant_index_left * n_determinants;
    for (int determinant_index_right = 0;
         determinant_index_right <= determinant_index_left;
         ++determinant_index_right) {
      const std::size_t unordered_index =
          canonical_determinant_pair_storage_index(
              determinant_index_left,
              determinant_index_right);
      const std::size_t direct_ordered_index =
          left_row_offset + determinant_index_right;
      const double hamiltonian_direct =
          pair_weights.ordered_hamiltonian_weights[direct_ordered_index];
      const double overlap_direct =
          pair_weights.ordered_overlap_weights[direct_ordered_index];
      if (determinant_index_left == determinant_index_right) {
        pair_weights.unordered_combined_hamiltonian_weights[unordered_index] =
            hamiltonian_direct;
        pair_weights.unordered_combined_overlap_weights[unordered_index] =
            overlap_direct;
      } else {
        const std::size_t swapped_ordered_index =
            determinant_index_right * n_determinants +
            determinant_index_left;
        pair_weights.unordered_combined_hamiltonian_weights[unordered_index] =
            hamiltonian_direct +
            pair_weights.ordered_hamiltonian_weights[swapped_ordered_index];
        pair_weights.unordered_combined_overlap_weights[unordered_index] =
            overlap_direct +
            pair_weights.ordered_overlap_weights[swapped_ordered_index];
      }
    }
  }

  return pair_weights;
}

DeterminantPairWeightTablesFromCoefficients
build_directional_determinant_pair_weight_tables_from_coefficients(
    const SelectedStateDeterminantMatrices& selected_state_matrices,
    const SelectedStateDeterminantMatrices& directional_selected_state_matrices,
    const std::vector<double>& selected_state_energies,
    const std::vector<double>& directional_selected_state_energies) {
  const int n_determinants = selected_state_matrices.n_determinants;
  if (n_determinants <= 0) {
    throw std::invalid_argument("selected_state_matrices.n_determinants must be positive");
  }
  if (selected_state_matrices.n_determinants !=
          directional_selected_state_matrices.n_determinants ||
      selected_state_matrices.selected_state_indices !=
          directional_selected_state_matrices.selected_state_indices ||
      selected_state_matrices.states.size() !=
          directional_selected_state_matrices.states.size()) {
    throw std::invalid_argument(
        "directional selected-state matrices must align with accepted-point states");
  }
  if (selected_state_energies.size() != selected_state_matrices.states.size() ||
      directional_selected_state_energies.size() != selected_state_matrices.states.size()) {
    throw std::invalid_argument(
        "selected-state energies must align with selected_state_matrices.states");
  }

  DeterminantPairWeightTablesFromCoefficients pair_weights;
  pair_weights.n_determinants = n_determinants;
  pair_weights.ordered_hamiltonian_weights.assign(
      n_determinants * n_determinants,
      0.0);
  pair_weights.ordered_overlap_weights.assign(
      n_determinants * n_determinants,
      0.0);

  for (std::size_t state_offset = 0;
       state_offset < selected_state_matrices.states.size();
       ++state_offset) {
    const auto& state_coefficients = selected_state_matrices.states[state_offset];
    const auto& directional_state_coefficients =
        directional_selected_state_matrices.states[state_offset];
    if (state_coefficients.determinant_coefficients.size() !=
            n_determinants ||
        directional_state_coefficients.determinant_coefficients.size() !=
            n_determinants) {
      throw std::invalid_argument(
          "state determinant_coefficients size does not match n_determinants");
    }

    const double state_weight =
        selected_state_matrices.normalized_state_weights[state_offset];
    const double state_energy = selected_state_energies[state_offset];
    const double directional_state_energy =
        directional_selected_state_energies[state_offset];

    for (int determinant_index_left = 0;
         determinant_index_left < n_determinants;
         ++determinant_index_left) {
      const double coefficient_left =
          state_coefficients.determinant_coefficients[
              determinant_index_left];
      const double directional_coefficient_left =
          directional_state_coefficients.determinant_coefficients[
              determinant_index_left];
      const std::size_t row_offset =
          determinant_index_left * n_determinants;
      for (int determinant_index_right = 0;
           determinant_index_right < n_determinants;
           ++determinant_index_right) {
        const double coefficient_right =
            state_coefficients.determinant_coefficients[
                determinant_index_right];
        const double directional_coefficient_right =
            directional_state_coefficients.determinant_coefficients[
                determinant_index_right];
        const double directional_pair_product =
            directional_coefficient_left * coefficient_right +
            coefficient_left * directional_coefficient_right;
        const std::size_t ordered_index =
            row_offset + determinant_index_right;
        pair_weights.ordered_hamiltonian_weights[ordered_index] +=
            state_weight * directional_pair_product;
        pair_weights.ordered_overlap_weights[ordered_index] -=
            state_weight *
            (directional_state_energy * coefficient_left * coefficient_right +
             state_energy * directional_pair_product);
      }
    }
  }

  const std::size_t n_unordered_pairs =
      unordered_determinant_pair_count(n_determinants);
  pair_weights.unordered_combined_hamiltonian_weights.assign(
      n_unordered_pairs,
      0.0);
  pair_weights.unordered_combined_overlap_weights.assign(
      n_unordered_pairs,
      0.0);

  for (int determinant_index_left = 0;
       determinant_index_left < n_determinants;
       ++determinant_index_left) {
    const std::size_t left_row_offset =
        determinant_index_left * n_determinants;
    for (int determinant_index_right = 0;
         determinant_index_right <= determinant_index_left;
         ++determinant_index_right) {
      const std::size_t unordered_index =
          canonical_determinant_pair_storage_index(
              determinant_index_left,
              determinant_index_right);
      const std::size_t direct_ordered_index =
          left_row_offset + determinant_index_right;
      const double hamiltonian_direct =
          pair_weights.ordered_hamiltonian_weights[direct_ordered_index];
      const double overlap_direct =
          pair_weights.ordered_overlap_weights[direct_ordered_index];
      if (determinant_index_left == determinant_index_right) {
        pair_weights.unordered_combined_hamiltonian_weights[unordered_index] =
            hamiltonian_direct;
        pair_weights.unordered_combined_overlap_weights[unordered_index] =
            overlap_direct;
      } else {
        const std::size_t swapped_ordered_index =
            determinant_index_right * n_determinants +
            determinant_index_left;
        pair_weights.unordered_combined_hamiltonian_weights[unordered_index] =
            hamiltonian_direct +
            pair_weights.ordered_hamiltonian_weights[swapped_ordered_index];
        pair_weights.unordered_combined_overlap_weights[unordered_index] =
            overlap_direct +
            pair_weights.ordered_overlap_weights[swapped_ordered_index];
      }
    }
  }

  return pair_weights;
}

DeterminantPairWeightTablesFromCoefficients
build_exact_determinant_pair_weight_tables_from_eigenvalues(
    const SelectedStateDeterminantMatrices& selected_state_matrices,
    const std::vector<double>& eigenvalues) {
  const std::vector<double> selected_state_energies =
      gather_selected_state_energies(
          eigenvalues,
          selected_state_matrices.selected_state_indices);
  return build_exact_determinant_pair_weight_tables_from_coefficients(
      selected_state_matrices,
      selected_state_energies);
}

}  // namespace xmvb::vb
