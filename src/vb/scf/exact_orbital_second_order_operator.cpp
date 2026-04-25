#include "vb/scf/exact_orbital_second_order_operator.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <cblas.h>

#include "vb/matrices/structure_coefficient_blocks.hpp"
#include "vb/matrices/structure_block_kernels.hpp"
#include "vb/matrices/full_structure_builder.hpp"
#include "vb/matrices/determinant_pair_storage_utils.hpp"
#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/orbital/active_space_orbital_backpropagator.hpp"
#include "vb/orbital/active_space_matrix_backpropagator.hpp"
#include "vb/orbital/active_space_two_electron_backpropagator.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"
#include "vb/orbital/ao_effective_one_electron_backpropagator.hpp"
#include "vb/orbital/ao_effective_one_electron_builder.hpp"
#include "vb/orbital/ao_effective_one_electron_graph_operator.hpp"
#include "vb/scf/exact_ctx_memory_accounting.hpp"
#include "vb/scf/exact_ctx_strategy_profile.hpp"
#include "vb/scf/opposite_spin_matrix_backward.hpp"
#include "vb/scf/same_spin_matrix_backward.hpp"
#include "vb/scf/selected_state_determinant_matrices.hpp"

namespace xmvb::vb {

/**
 * @brief Caches direction-independent orbital preparation quantities.
 *
 * All fields depend only on the accepted-point orbital coefficients and the
 * AO basis overlap, not on the HVP direction vector. Building them once in
 * the operator constructor avoids redundant LDLT solves, projector rebuilds,
 * and sparse-to-dense gathers across the ~3-4 HVP calls per outer step.
 */
struct AcceptedOrbitalPreparationCache {
  Eigen::MatrixXd normalized_orbitals;
  std::vector<double> inverse_norms;
  Eigen::MatrixXd basis_overlap_times_normalized;
  Eigen::MatrixXd internal_inactive_orbitals;
  Eigen::MatrixXd selector_inactive_right_inverse_transform;
  Eigen::MatrixXd selector_inactive_inverse_transpose_right_transform;
  Eigen::MatrixXd selector_active_inactive_coefficients;

  Eigen::MatrixXd inactive_overlap_inverse;
  Eigen::MatrixXd inactive_auxiliary;
  Eigen::MatrixXd inactive_density;

  std::vector<std::vector<int>> orbital_basis_function_indices;
  std::vector<int> orbital_coefficient_counts;
  std::vector<Eigen::MatrixXd> orbital_overlap_submatrices;

  Eigen::MatrixXd original_orbital_gradient;
  Eigen::MatrixXd inactive_density_gradient_symmetric;

  bool has_inactive_orbitals = false;
  bool uses_internal_inactive_chart = false;
  bool has_orthonormal_inactive_chart = false;
  bool has_pullback_cache = false;
};

namespace {

constexpr int kLegacyOrbitalTypeOeo = 3;

struct ExactCtxInternalInactiveChart {
  Eigen::MatrixXd inactive_orbitals;
  Eigen::MatrixXd inactive_right_inverse_transform;
  Eigen::MatrixXd inactive_inverse_transpose_right_transform;
  Eigen::MatrixXd active_inactive_coefficients;
  bool enabled = false;
};

std::vector<double> scatter_dense_orbital_gradient_to_sparse_slots_local(
    const Eigen::Ref<const Eigen::MatrixXd>& original_orbital_gradient,
    const OrbitalPreparationInput& input) {
  if (original_orbital_gradient.rows() != input.n_basis_functions ||
      original_orbital_gradient.cols() != input.n_orbitals) {
    throw std::invalid_argument(
        "dense orbital gradient shape mismatch while scattering to sparse slots");
  }

  std::vector<double> orbital_value_gradient(input.orbital_value_table.size(), 0.0);
  for (int orbital_index = 0; orbital_index < input.n_orbitals; ++orbital_index) {
    const int coefficient_count =
        stored_sparse_orbital_coefficient_count(input, orbital_index);
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      const int basis_function_index =
          input.orbital_basis_index_table[orbital_index *
                                              input.n_basis_functions +
                                          coefficient_index] -
          1;
      orbital_value_gradient[orbital_index * input.n_basis_functions +
                             coefficient_index] =
          original_orbital_gradient(basis_function_index, orbital_index);
    }
  }
  return orbital_value_gradient;
}

struct PhysicalOrbitalGradientBlocks {
  Eigen::MatrixXd inactive_gradient;
  Eigen::MatrixXd active_gradient;
};


struct FusedAoEffectiveOneElectronDirectionalResult {
  std::vector<double> delta_ao_effective_h1e;
  std::vector<double> inactive_density_gradient;
};

/**
 * @brief Low-rank AO matrix represented as `left_factors * right_factors^T`.
 *
 * The exact-context HVP repeatedly differentiates the inactive-space
 * projector.  That derivative has rank at most twice the number of inactive
 * orbitals even though its AO representation is dense.  Keeping the factor
 * form lets orbital-preparation and fixed-upstream chain-rule terms multiply
 * by the projector derivative without paying for avoidable AO-by-AO products.
 */
struct LowRankAoMatrix {
  Eigen::MatrixXd left_factors;
  Eigen::MatrixXd right_factors;
};

struct InactiveAuxiliaryDirectionResult {
  Eigen::MatrixXd delta_inactive_auxiliary;
  Eigen::MatrixXd delta_inactive_overlap_inverse;
};

LowRankAoMatrix make_empty_low_rank_ao_matrix(int n_basis_functions) {
  LowRankAoMatrix matrix;
  matrix.left_factors.resize(n_basis_functions, 0);
  matrix.right_factors.resize(n_basis_functions, 0);
  return matrix;
}

LowRankAoMatrix make_low_rank_ao_matrix(
    const Eigen::Ref<const Eigen::MatrixXd>& left_factors,
    const Eigen::Ref<const Eigen::MatrixXd>& right_factors,
    const char* label) {
  if (left_factors.rows() != right_factors.rows() ||
      left_factors.cols() != right_factors.cols()) {
    throw std::invalid_argument(
        std::string(label) + " low-rank AO factors have inconsistent shapes");
  }
  LowRankAoMatrix matrix;
  matrix.left_factors = left_factors;
  matrix.right_factors = right_factors;
  return matrix;
}

LowRankAoMatrix build_inactive_density_direction_low_rank(
    int n_basis_functions,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_auxiliary,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_inactive_auxiliary) {
  const int n_inactive_orbitals = inactive_orbitals.cols();
  if (inactive_orbitals.rows() != n_basis_functions ||
      delta_inactive_orbitals.rows() != n_basis_functions ||
      inactive_auxiliary.rows() != n_basis_functions ||
      delta_inactive_auxiliary.rows() != n_basis_functions ||
      delta_inactive_orbitals.cols() != n_inactive_orbitals ||
      inactive_auxiliary.cols() != n_inactive_orbitals ||
      delta_inactive_auxiliary.cols() != n_inactive_orbitals) {
    throw std::invalid_argument(
        "inactive-density directional low-rank factors have inconsistent shapes");
  }
  if (n_inactive_orbitals == 0) {
    return make_empty_low_rank_ao_matrix(n_basis_functions);
  }

  // dP_i = dA_i C_i^T + A_i dC_i^T, where A_i = C_i (C_i^T S C_i)^{-1}.
  // The two rank-n_i terms are concatenated into one AO low-rank product.
  LowRankAoMatrix matrix;
  matrix.left_factors.resize(n_basis_functions, 2 * n_inactive_orbitals);
  matrix.right_factors.resize(n_basis_functions, 2 * n_inactive_orbitals);
  matrix.left_factors.leftCols(n_inactive_orbitals) =
      delta_inactive_auxiliary;
  matrix.left_factors.rightCols(n_inactive_orbitals) =
      inactive_auxiliary;
  matrix.right_factors.leftCols(n_inactive_orbitals) =
      inactive_orbitals;
  matrix.right_factors.rightCols(n_inactive_orbitals) =
      delta_inactive_orbitals;
  return matrix;
}

bool matrix_is_effectively_identity(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    double tolerance = 1.0e-10) {
  if (matrix.rows() != matrix.cols()) {
    return false;
  }
  for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
      const double target = row == column ? 1.0 : 0.0;
      if (std::abs(matrix(row, column) - target) > tolerance) {
        return false;
      }
    }
  }
  return true;
}

bool matrix_is_effectively_zero(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    double tolerance = 1.0e-10) {
  for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
      if (std::abs(matrix(row, column)) > tolerance) {
        return false;
      }
    }
  }
  return true;
}

bool orbitals_use_canonical_full_support(
    const OrbitalPreparationInput& input,
    int first_orbital,
    int orbital_count) {
  for (int orbital_index = first_orbital;
       orbital_index < first_orbital + orbital_count;
       ++orbital_index) {
    const int coefficient_count =
        stored_sparse_orbital_coefficient_count(input, orbital_index);
    if (coefficient_count != input.n_basis_functions) {
      return false;
    }
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      const int basis_function_index =
          input.orbital_basis_index_table[orbital_index *
                                              input.n_basis_functions +
                                          coefficient_index] -
          1;
      if (basis_function_index != coefficient_index) {
        return false;
      }
    }
  }
  return true;
}

bool exact_ctx_supports_internal_inactive_chart(
    const OrbitalPreparationInput& input,
    const OrbitalPreparationResult& orbital_result,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) {
  constexpr int kLegacyOrbitalTypeOeo = 3;
  if (n_inactive_doubly_occupied_orbitals <= 0 ||
      input.orbital_type != kLegacyOrbitalTypeOeo) {
    return false;
  }
  if (!orbitals_use_canonical_full_support(
          input,
          0,
          n_inactive_doubly_occupied_orbitals + n_active_orbitals)) {
    return false;
  }

  const auto& physical_orbital_frame =
      orbital_result.physical_orbital_frame;
  const auto& selector =
      physical_orbital_frame.localized_representative_selector;
  return physical_orbital_frame.inactive_orthonormal_orbital_matrix.rows() ==
          input.n_basis_functions &&
      physical_orbital_frame.inactive_orthonormal_orbital_matrix.cols() ==
          n_inactive_doubly_occupied_orbitals &&
      orbital_result.auxiliary_orbital_matrix.rows() == input.n_basis_functions &&
      orbital_result.auxiliary_orbital_matrix.cols() >=
          n_inactive_doubly_occupied_orbitals + n_active_orbitals &&
      selector.inactive_inverse_transpose_right_transform.rows() ==
          n_inactive_doubly_occupied_orbitals &&
      selector.inactive_inverse_transpose_right_transform.cols() ==
          n_inactive_doubly_occupied_orbitals &&
      selector.active_inactive_coefficients.rows() ==
          n_inactive_doubly_occupied_orbitals &&
      selector.active_inactive_coefficients.cols() == n_active_orbitals;
}

bool exact_ctx_internal_inactive_chart_runtime_enabled() {
  const char* disable_flag =
      std::getenv("XMVB_CPP_DISABLE_EXACT_CTX_INTERNAL_INACTIVE_CHART");
  if (disable_flag != nullptr &&
      disable_flag[0] != '\0' &&
      std::strcmp(disable_flag, "0") != 0 &&
      std::strcmp(disable_flag, "false") != 0 &&
      std::strcmp(disable_flag, "FALSE") != 0) {
    return false;
  }

  const char* enable_flag =
      std::getenv("XMVB_CPP_ENABLE_EXACT_CTX_INTERNAL_INACTIVE_CHART");
  if (enable_flag == nullptr || enable_flag[0] == '\0') {
    // Leave the runtime gate open by default.  The actual default policy is
    // chosen later from the molecule/spin context in
    // `exact_ctx_prefers_internal_inactive_chart()`: closed-shell HAO cases
    // such as 241/10698 benefit substantially from the cheaper `(Q_i, T_a)`
    // chart, while MnF2-class open-shell production cases need the physical
    // occupied chart for robust trust-region progress.
    return true;
  }
  return std::strcmp(enable_flag, "0") != 0 &&
      std::strcmp(enable_flag, "false") != 0 &&
      std::strcmp(enable_flag, "FALSE") != 0;
}

bool exact_ctx_prefers_internal_inactive_chart(
    const OrbitalPreparationInput& input,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) {
  const char* enable_flag =
      std::getenv("XMVB_CPP_ENABLE_EXACT_CTX_INTERNAL_INACTIVE_CHART");
  const bool force_enable =
      enable_flag != nullptr &&
      enable_flag[0] != '\0' &&
      std::strcmp(enable_flag, "0") != 0 &&
      std::strcmp(enable_flag, "false") != 0 &&
      std::strcmp(enable_flag, "FALSE") != 0;
  if (force_enable) {
    return true;
  }
  (void)n_inactive_doubly_occupied_orbitals;
  (void)n_active_orbitals;
  const ExactCtxDefaultStrategy strategy =
      choose_exact_ctx_default_strategy(
          build_exact_ctx_system_profile(input));
  return strategy.prefer_internal_inactive_chart;
}

ExactCtxInternalInactiveChart build_exact_ctx_internal_inactive_chart(
    const OrbitalPreparationInput& input,
    const OrbitalPreparationResult& orbital_result,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) {
  ExactCtxInternalInactiveChart chart;
  if (!exact_ctx_internal_inactive_chart_runtime_enabled()) {
    return chart;
  }
  if (!exact_ctx_prefers_internal_inactive_chart(
          input,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals)) {
    return chart;
  }
  if (!exact_ctx_supports_internal_inactive_chart(
          input,
          orbital_result,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals)) {
    return chart;
  }

  const auto& physical_orbital_frame =
      orbital_result.physical_orbital_frame;
  const auto& selector =
      physical_orbital_frame.localized_representative_selector;

  // exact_ctx works internally on the accepted-point chart `(Q_i, T_a)` while
  // the optimizer still stores the localized physical representative
  // `(C_i, C_a)`.  The fixed selector `(U_i^{-1}, U_i^{-T}, K_a)` is the
  // small accepted-point Jacobian that lifts packed physical directions into
  // the internal chart and later scatters internal HVP gradients back.
  chart.inactive_orbitals =
      physical_orbital_frame.inactive_orthonormal_orbital_matrix;
  chart.inactive_inverse_transpose_right_transform =
      selector.inactive_inverse_transpose_right_transform;
  chart.inactive_right_inverse_transform =
      selector.inactive_inverse_transpose_right_transform.transpose();
  chart.active_inactive_coefficients =
      selector.active_inactive_coefficients;
  chart.enabled = true;
  return chart;
}

bool exact_ctx_uses_orthonormal_inactive_chart(
    const OrbitalPreparationInput& input,
    int n_inactive_doubly_occupied_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_overlap_inverse) {
  constexpr int kLegacyOrbitalTypeOeo = 3;
  return n_inactive_doubly_occupied_orbitals > 0 &&
      input.orbital_type == kLegacyOrbitalTypeOeo &&
      orbitals_use_canonical_full_support(
          input,
          0,
          n_inactive_doubly_occupied_orbitals) &&
      matrix_is_effectively_identity(inactive_overlap_inverse);
}

Eigen::MatrixXd build_inactive_auxiliary(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_overlap_inverse,
    bool use_orthonormal_inactive_chart) {
  return use_orthonormal_inactive_chart
      ? Eigen::MatrixXd(inactive_orbitals)
      : inactive_orbitals * inactive_overlap_inverse;
}

Eigen::MatrixXd apply_occupied_projector_to_orbitals(
    const Eigen::Ref<const Eigen::MatrixXd>& orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_auxiliary,
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap_times_orbitals) {
  if (basis_overlap_times_orbitals.rows() != orbitals.rows() ||
      basis_overlap_times_orbitals.cols() != orbitals.cols() ||
      inactive_orbitals.rows() != orbitals.rows() ||
      inactive_auxiliary.rows() != orbitals.rows() ||
      inactive_orbitals.cols() != inactive_auxiliary.cols()) {
    throw std::invalid_argument(
        "occupied projector application has inconsistent dimensions");
  }
  if (inactive_orbitals.cols() == 0) {
    return orbitals;
  }

  // Apply `O X = (I - A_i C_i^T S) X` through inactive factors instead of
  // materializing the AO-by-AO occupied projector.  In the orthonormal
  // inactive gauge `A_i = Q_i`, but the same contraction order remains valid.
  return orbitals -
      inactive_auxiliary *
          (inactive_orbitals.transpose() * basis_overlap_times_orbitals);
}

Eigen::MatrixXd apply_occupied_projector_transpose_to_gradient(
    const Eigen::Ref<const Eigen::MatrixXd>& gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_auxiliary,
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap_times_inactive) {
  if (inactive_auxiliary.rows() != gradient.rows() ||
      basis_overlap_times_inactive.rows() != gradient.rows() ||
      inactive_auxiliary.cols() != basis_overlap_times_inactive.cols()) {
    throw std::invalid_argument(
        "occupied projector transpose application has inconsistent dimensions");
  }
  if (inactive_auxiliary.cols() == 0) {
    return gradient;
  }

  // `O^T G = G - S C_i A_i^T G`; this is the adjoint of the projector above
  // and avoids the old dense AO projector multiply in the cached pullback.
  return gradient -
      basis_overlap_times_inactive *
          (inactive_auxiliary.transpose() * gradient);
}

Eigen::MatrixXd apply_inactive_density_direction_to_basis_overlap_times_active(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_auxiliary,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_inactive_auxiliary,
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap_times_active_orbitals,
    bool use_orthonormal_inactive_chart) {
  if (inactive_orbitals.rows() != basis_overlap_times_active_orbitals.rows() ||
      delta_inactive_orbitals.rows() != basis_overlap_times_active_orbitals.rows() ||
      inactive_auxiliary.rows() != basis_overlap_times_active_orbitals.rows() ||
      delta_inactive_auxiliary.rows() != basis_overlap_times_active_orbitals.rows() ||
      inactive_orbitals.cols() != delta_inactive_orbitals.cols() ||
      inactive_orbitals.cols() != inactive_auxiliary.cols() ||
      inactive_orbitals.cols() != delta_inactive_auxiliary.cols()) {
    throw std::invalid_argument(
        "inactive-density direction application has inconsistent dimensions");
  }
  if (inactive_orbitals.cols() == 0) {
    return Eigen::MatrixXd::Zero(
        basis_overlap_times_active_orbitals.rows(),
        basis_overlap_times_active_orbitals.cols());
  }

  const Eigen::MatrixXd inactive_active_overlap =
      inactive_orbitals.transpose() * basis_overlap_times_active_orbitals;
  const Eigen::MatrixXd delta_inactive_active_overlap =
      delta_inactive_orbitals.transpose() * basis_overlap_times_active_orbitals;
  if (use_orthonormal_inactive_chart &&
      matrix_is_effectively_zero(inactive_active_overlap)) {
    // With `Q_i^T S C_a = 0`, the `delta Q_i (Q_i^T S C_a)` half of
    // `d(Q_i Q_i^T) S C_a` vanishes.  Keeping only
    // `Q_i (delta Q_i^T S C_a)` removes one inactive-rank multiply and avoids
    // numerical noise from an overlap that should be exactly zero in this chart.
  return inactive_auxiliary * delta_inactive_active_overlap;
  }
  return delta_inactive_auxiliary * inactive_active_overlap +
      inactive_auxiliary * delta_inactive_active_overlap;
}

Eigen::MatrixXd build_internal_inactive_density_pullback_gradient(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_gradient_symmetric,
    const Eigen::Ref<const Eigen::MatrixXd>& internal_inactive_orbitals) {
  if (inactive_density_gradient_symmetric.rows() != internal_inactive_orbitals.rows() ||
      inactive_density_gradient_symmetric.cols() != internal_inactive_orbitals.rows()) {
    throw std::invalid_argument(
        "internal inactive-density pullback gradient has inconsistent dimensions");
  }
  return inactive_density_gradient_symmetric * internal_inactive_orbitals;
}

Eigen::MatrixXd build_internal_inactive_density_pullback_gradient_direction(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_gradient_symmetric,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_internal_inactive_orbitals) {
  if (inactive_density_gradient_symmetric.rows() != delta_internal_inactive_orbitals.rows() ||
      inactive_density_gradient_symmetric.cols() != delta_internal_inactive_orbitals.rows()) {
    throw std::invalid_argument(
        "internal inactive-density pullback gradient direction has inconsistent dimensions");
  }
  return inactive_density_gradient_symmetric * delta_internal_inactive_orbitals;
}

PhysicalOrbitalGradientBlocks transport_internal_chart_gradient_to_physical(
    const Eigen::Ref<const Eigen::MatrixXd>& internal_inactive_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& internal_active_gradient,
    const ExactCtxInternalInactiveChart& chart) {
  if (!chart.enabled) {
    throw std::invalid_argument(
        "internal inactive chart transport requires an enabled chart");
  }
  if (internal_inactive_gradient.rows() != chart.inactive_orbitals.rows() ||
      internal_inactive_gradient.cols() != chart.inactive_orbitals.cols() ||
      internal_active_gradient.rows() != chart.inactive_orbitals.rows() ||
      internal_active_gradient.cols() != chart.active_inactive_coefficients.cols()) {
    throw std::invalid_argument(
        "internal chart gradient transport has inconsistent dimensions");
  }

  PhysicalOrbitalGradientBlocks physical_gradient;
  physical_gradient.active_gradient = internal_active_gradient;
  physical_gradient.inactive_gradient =
      (internal_inactive_gradient -
       internal_active_gradient * chart.active_inactive_coefficients.transpose()) *
      chart.inactive_inverse_transpose_right_transform;
  return physical_gradient;
}

InactiveAuxiliaryDirectionResult build_delta_inactive_auxiliary_direction(
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_overlap_inverse,
    bool use_orthonormal_inactive_chart) {
  InactiveAuxiliaryDirectionResult result;
  result.delta_inactive_auxiliary =
      Eigen::MatrixXd::Zero(
          inactive_orbitals.rows(),
          inactive_orbitals.cols());
  result.delta_inactive_overlap_inverse =
      Eigen::MatrixXd::Zero(
          inactive_overlap_inverse.rows(),
          inactive_overlap_inverse.cols());
  if (use_orthonormal_inactive_chart) {
    result.delta_inactive_auxiliary = delta_inactive_orbitals;
    return result;
  }

  const Eigen::MatrixXd delta_inactive_overlap =
      delta_inactive_orbitals.transpose() * basis_overlap * inactive_orbitals +
      inactive_orbitals.transpose() * basis_overlap * delta_inactive_orbitals;
  result.delta_inactive_overlap_inverse =
      -inactive_overlap_inverse *
      delta_inactive_overlap *
      inactive_overlap_inverse;
  result.delta_inactive_auxiliary =
      delta_inactive_orbitals * inactive_overlap_inverse +
      inactive_orbitals * result.delta_inactive_overlap_inverse;
  return result;
}

Eigen::MatrixXd build_inactive_overlap_gradient(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_overlap_inverse,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& total_inactive_gradient) {
  // The mixed-gauge pullback depends on the derivative of
  // `M^{-1}` with `M = C_i^T S C_i`.  In the orthonormal inactive gauge
  // `M = I`, so this entire contribution vanishes.
  const Eigen::MatrixXd inactive_overlap_inverse_gradient =
      inactive_orbitals.transpose() *
      total_inactive_gradient *
      inactive_orbitals;
  return -inactive_overlap_inverse *
      inactive_overlap_inverse_gradient *
      inactive_overlap_inverse;
}

Eigen::MatrixXd build_inactive_overlap_gradient_direction(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_overlap_inverse,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_inactive_overlap_inverse,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& total_inactive_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_total_inactive_gradient) {
  const Eigen::MatrixXd inactive_overlap_inverse_gradient =
      inactive_orbitals.transpose() *
      total_inactive_gradient *
      inactive_orbitals;
  const Eigen::MatrixXd delta_inactive_overlap_inverse_gradient =
      delta_inactive_orbitals.transpose() *
          total_inactive_gradient *
          inactive_orbitals +
      inactive_orbitals.transpose() *
          delta_total_inactive_gradient *
          inactive_orbitals +
      inactive_orbitals.transpose() *
          total_inactive_gradient *
          delta_inactive_orbitals;
  return -delta_inactive_overlap_inverse *
          inactive_overlap_inverse_gradient *
          inactive_overlap_inverse -
      inactive_overlap_inverse *
          delta_inactive_overlap_inverse_gradient *
          inactive_overlap_inverse -
      inactive_overlap_inverse *
          inactive_overlap_inverse_gradient *
          delta_inactive_overlap_inverse;
}

Eigen::MatrixXd build_inactive_projector_pullback_gradient(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_gradient_symmetric,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_auxiliary,
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap_times_inactive,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_overlap_gradient) {
  return inactive_density_gradient_symmetric * inactive_auxiliary +
      basis_overlap_times_inactive *
          (inactive_overlap_gradient + inactive_overlap_gradient.transpose());
}

Eigen::MatrixXd build_inactive_projector_pullback_gradient_direction(
    const Eigen::Ref<const Eigen::MatrixXd>& delta_inactive_density_gradient_symmetric,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_gradient_symmetric,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_auxiliary,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_inactive_auxiliary,
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap_times_inactive,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_basis_overlap_times_inactive,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_overlap_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_inactive_overlap_gradient) {
  return delta_inactive_density_gradient_symmetric *
          inactive_auxiliary +
      inactive_density_gradient_symmetric *
          delta_inactive_auxiliary +
      delta_basis_overlap_times_inactive *
          (inactive_overlap_gradient + inactive_overlap_gradient.transpose()) +
      basis_overlap_times_inactive *
          (delta_inactive_overlap_gradient +
           delta_inactive_overlap_gradient.transpose());
}

Eigen::MatrixXd build_identity_metric_inactive_projector_pullback_gradient(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_gradient_symmetric,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap_times_inactive) {
  if (inactive_density_gradient_symmetric.rows() != inactive_orbitals.rows() ||
      inactive_density_gradient_symmetric.cols() != inactive_orbitals.rows() ||
      basis_overlap_times_inactive.rows() != inactive_orbitals.rows() ||
      basis_overlap_times_inactive.cols() != inactive_orbitals.cols()) {
    throw std::invalid_argument(
        "identity-metric fixed-upstream inactive gradient has inconsistent dimensions");
  }
  if (inactive_orbitals.cols() == 0) {
    return Eigen::MatrixXd::Zero(
        inactive_orbitals.rows(),
        inactive_orbitals.cols());
  }

  // At an orthonormalized accepted point we still differentiate with respect
  // to the raw inactive coefficients `C_i`, not a generalized-Stiefel tangent
  // variable.  With `M = C_i^T S C_i = I`, the exact mixed-gauge pullback
  // reduces to `G_sym C_i - S C_i (C_i^T G_sym C_i)` and therefore keeps the
  // raw-chart correction without rebuilding any inactive inverse factors.
  const Eigen::MatrixXd reduced_metric_gradient =
      inactive_orbitals.transpose() *
      inactive_density_gradient_symmetric *
      inactive_orbitals;
  return inactive_density_gradient_symmetric * inactive_orbitals -
      basis_overlap_times_inactive * reduced_metric_gradient;
}

Eigen::MatrixXd build_identity_metric_inactive_projector_pullback_gradient_direction(
    const Eigen::Ref<const Eigen::MatrixXd>& delta_inactive_density_gradient_symmetric,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_gradient_symmetric,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap_times_inactive,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_basis_overlap_times_inactive) {
  if (delta_inactive_density_gradient_symmetric.rows() != inactive_orbitals.rows() ||
      delta_inactive_density_gradient_symmetric.cols() != inactive_orbitals.rows() ||
      inactive_density_gradient_symmetric.rows() != inactive_orbitals.rows() ||
      inactive_density_gradient_symmetric.cols() != inactive_orbitals.rows() ||
      delta_inactive_orbitals.rows() != inactive_orbitals.rows() ||
      delta_inactive_orbitals.cols() != inactive_orbitals.cols() ||
      basis_overlap_times_inactive.rows() != inactive_orbitals.rows() ||
      basis_overlap_times_inactive.cols() != inactive_orbitals.cols() ||
      delta_basis_overlap_times_inactive.rows() != inactive_orbitals.rows() ||
      delta_basis_overlap_times_inactive.cols() != inactive_orbitals.cols()) {
    throw std::invalid_argument(
        "identity-metric fixed-upstream inactive gradient direction has inconsistent dimensions");
  }
  if (inactive_orbitals.cols() == 0) {
    return Eigen::MatrixXd::Zero(
        inactive_orbitals.rows(),
        inactive_orbitals.cols());
  }

  // Differentiate
  // `g_C = G_sym C_i - S C_i (C_i^T G_sym C_i)`
  // in the raw inactive chart at an accepted point with `C_i^T S C_i = I`.
  const Eigen::MatrixXd reduced_metric_gradient =
      inactive_orbitals.transpose() *
      inactive_density_gradient_symmetric *
      inactive_orbitals;
  const Eigen::MatrixXd delta_reduced_metric_gradient =
      delta_inactive_orbitals.transpose() *
          inactive_density_gradient_symmetric *
          inactive_orbitals +
      inactive_orbitals.transpose() *
          delta_inactive_density_gradient_symmetric *
          inactive_orbitals +
      inactive_orbitals.transpose() *
          inactive_density_gradient_symmetric *
          delta_inactive_orbitals;
  return delta_inactive_density_gradient_symmetric *
          inactive_orbitals +
      inactive_density_gradient_symmetric *
          delta_inactive_orbitals -
      delta_basis_overlap_times_inactive *
          reduced_metric_gradient -
      basis_overlap_times_inactive *
          delta_reduced_metric_gradient;
}

void validate_low_rank_ao_matrix(
    const LowRankAoMatrix& matrix,
    const char* label) {
  if (matrix.left_factors.rows() != matrix.right_factors.rows() ||
      matrix.left_factors.cols() != matrix.right_factors.cols()) {
    throw std::invalid_argument(
        std::string(label) + " low-rank AO matrix has inconsistent factors");
  }
}

Eigen::MatrixXd materialize_low_rank_ao_matrix(
    const LowRankAoMatrix& matrix,
    const char* label) {
  validate_low_rank_ao_matrix(matrix, label);
  if (matrix.left_factors.cols() == 0) {
    return Eigen::MatrixXd::Zero(
        matrix.left_factors.rows(),
        matrix.right_factors.rows());
  }
  return matrix.left_factors * matrix.right_factors.transpose();
}

Eigen::MatrixXd apply_low_rank_ao_matrix(
    const LowRankAoMatrix& matrix,
    const Eigen::Ref<const Eigen::MatrixXd>& rhs,
    const char* label) {
  validate_low_rank_ao_matrix(matrix, label);
  if (matrix.right_factors.rows() != rhs.rows()) {
    throw std::invalid_argument(
        std::string(label) + " low-rank AO multiply dimension mismatch");
  }
  if (matrix.left_factors.cols() == 0) {
    return Eigen::MatrixXd::Zero(matrix.left_factors.rows(), rhs.cols());
  }
  return matrix.left_factors * (matrix.right_factors.transpose() * rhs);
}

Eigen::MatrixXd apply_metric_times_low_rank_ao_matrix(
    const Eigen::Ref<const Eigen::MatrixXd>& metric,
    const LowRankAoMatrix& matrix,
    const Eigen::Ref<const Eigen::MatrixXd>& rhs,
    const char* label) {
  validate_low_rank_ao_matrix(matrix, label);
  if (metric.rows() != matrix.left_factors.rows() ||
      metric.cols() != matrix.left_factors.rows() ||
      matrix.right_factors.rows() != rhs.rows()) {
    throw std::invalid_argument(
        std::string(label) + " metric low-rank AO multiply dimension mismatch");
  }
  if (matrix.left_factors.cols() == 0) {
    return Eigen::MatrixXd::Zero(metric.rows(), rhs.cols());
  }

  // Contract through the smaller outer dimension.  This avoids Eigen parsing
  // `S * dP * G` as a dense AO-by-AO product while keeping the same algebra.
  const Eigen::MatrixXd compressed_rhs =
      matrix.right_factors.transpose() * rhs;
  if (matrix.left_factors.cols() <= rhs.cols()) {
    return (metric * matrix.left_factors) * compressed_rhs;
  }
  return metric * (matrix.left_factors * compressed_rhs);
}

void add_orbital_value_gradient_in_place(
    std::vector<double>* target,
    const std::vector<double>& contribution,
    const char* /*label*/) {
  if (target->empty()) {
    *target = contribution;
    return;
  }
  for (std::size_t index = 0; index < target->size(); ++index) {
    (*target)[index] += contribution[index];
  }
}

void resize_for_overwrite(
    std::vector<double>* values,
    std::size_t size) {
  if (values->size() != size) {
    values->resize(size);
  }
}

std::vector<std::size_t> build_ao_matrix_column_offsets(
    int n_basis_functions) {
  std::vector<std::size_t> column_offsets(n_basis_functions, 0);
  for (int column = 0; column < n_basis_functions; ++column) {
    column_offsets[column] =
        column * n_basis_functions;
  }
  return column_offsets;
}

int positive_env_override_local(
    const char* env_name,
    int default_value) {
  const char* env_value = std::getenv(env_name);
  if (env_value == nullptr || env_value[0] == '\0') {
    return default_value;
  }
  const int parsed_value = std::stoi(env_value);
  if (parsed_value <= 0) {
    throw std::invalid_argument(
        std::string(env_name) + " must be positive");
  }
  return parsed_value;
}

int exact_ctx_openmp_thread_cap() {
  return positive_env_override_local(
      "XMVB_CPP_EXACT_CTX_MAX_OMP_THREADS",
      8);
}

int exact_ctx_ao_h1e_thread_divisor() {
  // The AO-H1E fused graph sweep is memory-bandwidth bound, so the practical
  // OpenMP width should scale with the AO dimension instead of blindly
  // matching `OMP_NUM_THREADS`. Keep the historical `n_basis / 32` heuristic
  // as the repository default, but expose the divisor for same-binary tuning
  // on production systems where the memory subsystem or AO graph density shifts
  // the optimal point.
  return positive_env_override_local(
      "XMVB_CPP_EXACT_CTX_AO_H1E_THREAD_DIVISOR",
      32);
}

int exact_ctx_ao_h1e_thread_divisor_default(
    const ExactCtxSystemProfile& system_profile,
    bool compute_outer_response) {
  // The accepted-point AO-H1E graph is one of the dominant exact_ctx kernels.
  // Same-node HVP benchmarks show three distinct regimes:
  //
  // 1. Open-shell sparse charts such as MnF2 lose time if the AO-H1E graph is
  //    widened aggressively. Keep the conservative historical divisor there.
  // 2. Closed-shell sparse charts spend many more HVPs in the cheap core-only
  //    path than in the occasional outer-response path. Those systems benefit
  //    from a slightly wider AO-H1E sweep during cheap solves.
  // 3. The outer-response/full-model steps on the same closed-shell sparse
  //    systems still prefer a more conservative width to avoid overshooting the
  //    memory-bandwidth knee.
  //
  // This heuristic only changes the default thread count. Users can still
  // override it globally with `XMVB_CPP_EXACT_CTX_AO_H1E_THREAD_DIVISOR`.
  if (system_profile.open_shell) {
    return 32;
  }
  if (!system_profile.sparse_orbital_chart) {
    return 32;
  }
  return compute_outer_response ? 24 : 16;
}

int exact_ctx_effective_openmp_thread_limit(
    int workload_limited_threads) {
  int omp_max_threads = 1;
#ifdef _OPENMP
  omp_max_threads = omp_get_max_threads();
#endif
  return std::max(
      1,
      std::min(
          std::min(omp_max_threads, exact_ctx_openmp_thread_cap()),
          workload_limited_threads));
}

int choose_exact_ctx_ao_h1e_threads(
    const OrbitalPreparationInput& orbital_preparation_input,
    bool compute_outer_response) {
  // The fused AO-H1E sweep is memory-bandwidth bound for medium AO spaces.
  // Scaling threads with the AO dimension avoids the severe wide-team slowdown
  // seen in exact_ctx HVP benchmarks, while still allowing moderate speedup on
  // the closed-shell sparse charts where the core-only path dominates total
  // wall time.
  const ExactCtxSystemProfile system_profile =
      build_exact_ctx_system_profile(orbital_preparation_input);
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  const int thread_divisor =
      positive_env_override_local(
          "XMVB_CPP_EXACT_CTX_AO_H1E_THREAD_DIVISOR",
          exact_ctx_ao_h1e_thread_divisor_default(
              system_profile,
              compute_outer_response));
  const int workload_limited_threads =
      std::max(1, n_basis_functions / thread_divisor);
  return exact_ctx_effective_openmp_thread_limit(workload_limited_threads);
}

int choose_exact_ctx_directional_structure_threads(
    int n_structures) {
  // The directional structure builder allocates large thread-local tile
  // providers. For a few hundred structures, oversubscribing this loop with
  // dozens of threads mostly amplifies cache and allocation overhead.
  // Keep the conservative default, but expose the divisor so cluster
  // benchmarks can tune this stage without patching the source again.
  const int thread_divisor =
      positive_env_override_local(
          "XMVB_CPP_EXACT_CTX_DIRECTIONAL_STRUCTURE_THREAD_DIVISOR",
          48);
  const int workload_limited_threads =
      std::max(1, n_structures / thread_divisor);
  return exact_ctx_effective_openmp_thread_limit(workload_limited_threads);
}

std::size_t ao_pair_index_packed(int first, int second) {
  if (first >= second) {
    return first * (first + 1) / 2 + second;
  }
  return second * (second + 1) / 2 + first;
}

std::vector<double> pack_symmetric_matrix(
    const Eigen::MatrixXd& matrix) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("symmetric AO matrix pack requires a square matrix");
  }
  const int n_basis_functions = static_cast<int>(matrix.rows());
  const std::size_t n_ao_pairs =
      n_basis_functions * (n_basis_functions + 1) / 2;
  std::vector<double> packed(n_ao_pairs, 0.0);
  for (int first = 0; first < n_basis_functions; ++first) {
    for (int second = 0; second <= first; ++second) {
      packed[ao_pair_index_packed(first, second)] =
          first == second
              ? matrix(first, second)
              : 0.5 * (matrix(first, second) + matrix(second, first));
    }
  }
  return packed;
}

std::vector<double> unpack_symmetric_matrix_from_pairs(
    const std::vector<double>& packed_matrix,
    int n_basis_functions) {
  const std::size_t n_ao_pairs =
      n_basis_functions * (n_basis_functions + 1) / 2;
  if (packed_matrix.size() != n_ao_pairs) {
    throw std::invalid_argument("packed AO-pair matrix size mismatch");
  }
  std::vector<double> full_matrix(
      n_basis_functions * n_basis_functions,
      0.0);
  for (int first = 0; first < n_basis_functions; ++first) {
    for (int second = 0; second <= first; ++second) {
      const double value =
          packed_matrix[ao_pair_index_packed(first, second)];
      full_matrix[second * n_basis_functions + first] = value;
      full_matrix[first * n_basis_functions + second] = value;
    }
  }
  return full_matrix;
}

template <bool ValidateIntegralIndices, bool UsePrecomputedSymmetryShifts, bool UseLinearIndexCache>
inline void accumulate_fused_ao_effective_one_electron_integral(
    std::size_t integral_index,
    const double* inactive_density_matrix,
    const double* unsymmetrized_gradient_storage,
    const double* ao_two_electron_integral_values,
    const int* ao_two_electron_integral_indices,
    const std::uint8_t* ao_two_electron_integral_symmetry_shifts,
    const int* ao_effective_one_electron_linear_indices,
    const std::size_t* column_offsets,
    int n_basis_functions,
    std::atomic<int>& invalid_integral_index,
    double* local_delta_ao_effective_h1e,
    double* local_inactive_density_gradient) {
  const int* integral_indices =
      ao_two_electron_integral_indices + integral_index * 4;
  const int i = integral_indices[0];
  const int j = integral_indices[1];
  const int k = integral_indices[2];
  const int l = integral_indices[3];

  if constexpr (ValidateIntegralIndices) {
    if (i < 0 || i >= n_basis_functions ||
        j < 0 || j >= n_basis_functions ||
        k < 0 || k >= n_basis_functions ||
        l < 0 || l >= n_basis_functions) {
      int expected = -1;
      invalid_integral_index.compare_exchange_strong(
          expected,
          static_cast<int>(integral_index));
      return;
    }
  }

  double two_electron_value = ao_two_electron_integral_values[integral_index];
  if constexpr (UsePrecomputedSymmetryShifts) {
    constexpr double kIntegralSymmetryMultipliers[] = {
        1.0,
        0.5,
        0.25,
        0.125,
    };
    two_electron_value *= kIntegralSymmetryMultipliers[
        ao_two_electron_integral_symmetry_shifts[integral_index]];
  } else {
    if (i == j) {
      two_electron_value *= 0.5;
    }
    if (k == l) {
      two_electron_value *= 0.5;
    }
    if (i == k && j == l) {
      two_electron_value *= 0.5;
    }
  }

  std::size_t ij_index = 0;
  std::size_t kl_index = 0;
  std::size_t ik_index = 0;
  std::size_t jl_index = 0;
  std::size_t il_index = 0;
  std::size_t jk_index = 0;
  std::size_t lj_index = 0;
  std::size_t ki_index = 0;
  std::size_t kj_index = 0;
  std::size_t li_index = 0;
  if constexpr (UseLinearIndexCache) {
    const int* linear_indices =
        ao_effective_one_electron_linear_indices + integral_index * 10;
    ij_index = linear_indices[0];
    kl_index = linear_indices[1];
    ik_index = linear_indices[2];
    jl_index = linear_indices[3];
    il_index = linear_indices[4];
    jk_index = linear_indices[5];
    lj_index = linear_indices[6];
    ki_index = linear_indices[7];
    kj_index = linear_indices[8];
    li_index = linear_indices[9];
  } else {
    const std::size_t col_i = column_offsets[i];
    const std::size_t col_j = column_offsets[j];
    const std::size_t col_k = column_offsets[k];
    const std::size_t col_l = column_offsets[l];

    ij_index = col_j + i;
    kl_index = col_l + k;
    ik_index = col_k + i;
    jl_index = col_l + j;
    il_index = col_l + i;
    jk_index = col_k + j;
    lj_index = col_j + l;
    ki_index = col_i + k;
    kj_index = col_j + k;
    li_index = col_i + l;
  }

  const double density_ij = inactive_density_matrix[ij_index];
  const double density_kl = inactive_density_matrix[kl_index];
  const double density_lj = inactive_density_matrix[lj_index];
  const double density_ki = inactive_density_matrix[ki_index];
  const double density_kj = inactive_density_matrix[kj_index];
  const double density_li = inactive_density_matrix[li_index];

  const double gradient_ij = unsymmetrized_gradient_storage[ij_index];
  const double gradient_kl = unsymmetrized_gradient_storage[kl_index];
  const double gradient_ik = unsymmetrized_gradient_storage[ik_index];
  const double gradient_jl = unsymmetrized_gradient_storage[jl_index];
  const double gradient_il = unsymmetrized_gradient_storage[il_index];
  const double gradient_jk = unsymmetrized_gradient_storage[jk_index];
  const double scaled_two_electron_value = two_electron_value * 4.0;

  local_delta_ao_effective_h1e[ij_index] +=
      density_kl * scaled_two_electron_value;
  local_delta_ao_effective_h1e[kl_index] +=
      density_ij * scaled_two_electron_value;
  local_delta_ao_effective_h1e[ik_index] -= density_lj * two_electron_value;
  local_delta_ao_effective_h1e[jl_index] -= density_ki * two_electron_value;
  local_delta_ao_effective_h1e[il_index] -= density_kj * two_electron_value;
  local_delta_ao_effective_h1e[jk_index] -= density_li * two_electron_value;

  local_inactive_density_gradient[ij_index] +=
      gradient_kl * scaled_two_electron_value;
  local_inactive_density_gradient[kl_index] +=
      gradient_ij * scaled_two_electron_value;
  local_inactive_density_gradient[lj_index] -=
      gradient_ik * two_electron_value;
  local_inactive_density_gradient[ki_index] -=
      gradient_jl * two_electron_value;
  local_inactive_density_gradient[kj_index] -=
      gradient_il * two_electron_value;
  local_inactive_density_gradient[li_index] -=
      gradient_jk * two_electron_value;
}

template <bool ValidateIntegralIndices, bool UsePrecomputedSymmetryShifts, bool UseLinearIndexCache>
inline void apply_fused_ao_effective_one_electron_integral_loop_single_thread(
    const double* inactive_density_matrix,
    const double* unsymmetrized_gradient_storage,
    const AoIntegralInput& ao_integral_input,
    const std::size_t* column_offsets,
    int n_basis_functions,
    std::atomic<int>& invalid_integral_index,
    double* delta_ao_effective_h1e,
    double* inactive_density_gradient) {
  const std::uint8_t* symmetry_shifts =
      UsePrecomputedSymmetryShifts
          ? ao_integral_input.ao_two_electron_integral_symmetry_shifts.data()
          : nullptr;
  const int* linear_indices =
      UseLinearIndexCache
          ? ao_integral_input.ao_effective_one_electron_linear_indices.data()
          : nullptr;
  for (std::size_t integral_index = 0;
       integral_index < ao_integral_input.ao_two_electron_integral_values.size();
       ++integral_index) {
    accumulate_fused_ao_effective_one_electron_integral<
        ValidateIntegralIndices,
        UsePrecomputedSymmetryShifts,
        UseLinearIndexCache>(
        integral_index,
        inactive_density_matrix,
        unsymmetrized_gradient_storage,
        ao_integral_input.ao_two_electron_integral_values.data(),
        ao_integral_input.ao_two_electron_integral_indices.data(),
        symmetry_shifts,
        linear_indices,
        column_offsets,
        n_basis_functions,
        invalid_integral_index,
        delta_ao_effective_h1e,
        inactive_density_gradient);
  }
}

template <bool ValidateIntegralIndices, bool UsePrecomputedSymmetryShifts, bool UseLinearIndexCache>
inline void apply_fused_ao_effective_one_electron_integral_loop_parallel(
    const double* inactive_density_matrix,
    const double* unsymmetrized_gradient_storage,
    const AoIntegralInput& ao_integral_input,
    const std::size_t* column_offsets,
    int n_basis_functions,
    int n_threads,
    std::atomic<int>& invalid_integral_index,
    std::vector<Eigen::MatrixXd>* partial_delta_h1e,
    std::vector<Eigen::MatrixXd>* partial_density_gradients) {
  const std::uint8_t* symmetry_shifts =
      UsePrecomputedSymmetryShifts
          ? ao_integral_input.ao_two_electron_integral_symmetry_shifts.data()
          : nullptr;
  const int* linear_indices =
      UseLinearIndexCache
          ? ao_integral_input.ao_effective_one_electron_linear_indices.data()
          : nullptr;

#pragma omp parallel num_threads(n_threads)
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    Eigen::MatrixXd& local_delta_h1e =
        (*partial_delta_h1e)[thread_index];
    Eigen::MatrixXd& local_density_gradient =
        (*partial_density_gradients)[thread_index];

#pragma omp for schedule(static)
    for (std::ptrdiff_t integral_offset = 0;
         integral_offset <
             static_cast<std::ptrdiff_t>(
                 ao_integral_input.ao_two_electron_integral_values.size());
         ++integral_offset) {
      const std::size_t integral_index = integral_offset;
      accumulate_fused_ao_effective_one_electron_integral<
          ValidateIntegralIndices,
          UsePrecomputedSymmetryShifts,
          UseLinearIndexCache>(
          integral_index,
          inactive_density_matrix,
          unsymmetrized_gradient_storage,
          ao_integral_input.ao_two_electron_integral_values.data(),
          ao_integral_input.ao_two_electron_integral_indices.data(),
          symmetry_shifts,
          linear_indices,
          column_offsets,
          n_basis_functions,
          invalid_integral_index,
          local_delta_h1e.data(),
          local_density_gradient.data());
    }
  }
}

void resize_zero_fused_ao_h1e_partial_workspaces(
    int n_threads,
    int n_basis_functions,
    std::vector<Eigen::MatrixXd>* partial_delta_h1e,
    std::vector<Eigen::MatrixXd>* partial_density_gradients) {
  const Eigen::Index basis_dim =
      static_cast<Eigen::Index>(n_basis_functions);
  const std::size_t thread_count =
      static_cast<std::size_t>(n_threads);
  partial_delta_h1e->resize(thread_count);
  partial_density_gradients->resize(thread_count);

  // Each Krylov matvec touches the same AO square shape. Keep one dense AO
  // accumulator per OpenMP worker and just zero/reuse it instead of rebuilding
  // `n_threads * n_basis^2` heap buffers on every apply.
  for (std::size_t thread_index = 0;
       thread_index < thread_count;
       ++thread_index) {
    Eigen::MatrixXd& local_delta_h1e =
        (*partial_delta_h1e)[thread_index];
    if (local_delta_h1e.rows() != basis_dim ||
        local_delta_h1e.cols() != basis_dim) {
      local_delta_h1e.resize(basis_dim, basis_dim);
    }
    local_delta_h1e.setZero();

    Eigen::MatrixXd& local_density_gradient =
        (*partial_density_gradients)[thread_index];
    if (local_density_gradient.rows() != basis_dim ||
        local_density_gradient.cols() != basis_dim) {
      local_density_gradient.resize(basis_dim, basis_dim);
    }
    local_density_gradient.setZero();
  }
}

/**
 * The exact-context HVP needs both `delta G11[delta P11]` and the transpose
 * pullback `G11^T[delta Lambda]` against the same accepted AO ERI list.
 * Sweeping the materialized AO integrals once and accumulating both outputs
 * removes one full exact AO-H1E pass per matvec without introducing any new
 * spin-string-sized cache.
 */
void apply_fused_exact_ao_effective_one_electron_directional_operator(
    const Eigen::MatrixXd& inactive_density_matrix,
    const Eigen::MatrixXd& ao_effective_one_electron_gradient,
    const AoIntegralInput& ao_integral_input,
    const OrbitalPreparationInput& orbital_preparation_input,
    bool compute_outer_response,
    Eigen::MatrixXd* symmetrized_pullback_source,
    std::vector<double>* delta_ao_effective_h1e_storage,
    std::vector<double>* inactive_density_gradient_storage,
    std::vector<Eigen::MatrixXd>* partial_delta_h1e_workspaces = nullptr,
    std::vector<Eigen::MatrixXd>* partial_density_gradient_workspaces = nullptr) {
  const int n_basis_functions = ao_integral_input.n_basis_functions;
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("AO-H1E fused exact operator requires positive dimensions");
  }
  const std::size_t matrix_size =
      n_basis_functions * n_basis_functions;
  if (inactive_density_matrix.size() != matrix_size ||
      ao_effective_one_electron_gradient.size() != matrix_size) {
    throw std::invalid_argument("AO-H1E fused exact operator matrix size mismatch");
  }
  if (ao_integral_input.ao_two_electron_integral_indices.size() !=
      ao_integral_input.ao_two_electron_integral_values.size() * 4) {
    throw std::invalid_argument("AO two-electron index/value sizes are inconsistent");
  }
  if (!ao_integral_input.ao_two_electron_integral_symmetry_shifts.empty() &&
      ao_integral_input.ao_two_electron_integral_symmetry_shifts.size() !=
          ao_integral_input.ao_two_electron_integral_values.size()) {
    throw std::invalid_argument("AO two-electron symmetry-shift/value sizes are inconsistent");
  }
  if (!ao_integral_input.ao_effective_one_electron_linear_indices.empty() &&
      ao_integral_input.ao_effective_one_electron_linear_indices.size() !=
          ao_integral_input.ao_two_electron_integral_values.size() * 10) {
    throw std::invalid_argument(
        "AO effective one-electron linear-index cache size is inconsistent");
  }
  symmetrized_pullback_source->resizeLike(ao_effective_one_electron_gradient);
  *symmetrized_pullback_source = ao_effective_one_electron_gradient;
  *symmetrized_pullback_source += ao_effective_one_electron_gradient.transpose();
  delta_ao_effective_h1e_storage->assign(matrix_size, 0.0);
  inactive_density_gradient_storage->assign(matrix_size, 0.0);

  std::atomic<int> invalid_integral_index(-1);
  const bool validate_integral_indices =
      ao_integral_input.ao_two_electron_pair_indices.empty() &&
      ao_integral_input.ao_two_electron_pair_graph_row_offsets.empty();
  std::vector<std::size_t> column_offsets;
  if (ao_integral_input.ao_effective_one_electron_linear_indices.empty()) {
    column_offsets = build_ao_matrix_column_offsets(n_basis_functions);
  }
  const std::size_t* column_offsets_data =
      ao_integral_input.ao_effective_one_electron_linear_indices.empty()
          ? column_offsets.data()
          : nullptr;
  const std::uint8_t* symmetry_shifts =
      ao_integral_input.ao_two_electron_integral_symmetry_shifts.empty()
          ? nullptr
          : ao_integral_input.ao_two_electron_integral_symmetry_shifts.data();
  const int* linear_indices =
      ao_integral_input.ao_effective_one_electron_linear_indices.empty()
          ? nullptr
          : ao_integral_input.ao_effective_one_electron_linear_indices.data();

  const int n_threads =
      choose_exact_ctx_ao_h1e_threads(
          orbital_preparation_input,
          compute_outer_response);
  if (ao_effective_one_electron_graph_available(ao_integral_input)) {
    apply_fused_ao_effective_one_electron_graph(
        inactive_density_matrix.data(),
        symmetrized_pullback_source->data(),
        ao_integral_input,
        n_threads,
        delta_ao_effective_h1e_storage,
        inactive_density_gradient_storage);
  } else if (n_threads <= 1) {
    auto* delta_h1e = delta_ao_effective_h1e_storage->data();
    auto* density_gradient = inactive_density_gradient_storage->data();
    if (linear_indices != nullptr) {
      if (validate_integral_indices) {
        if (symmetry_shifts != nullptr) {
          apply_fused_ao_effective_one_electron_integral_loop_single_thread<
              true,
              true,
              true>(
              inactive_density_matrix.data(),
              symmetrized_pullback_source->data(),
              ao_integral_input,
              column_offsets_data,
              n_basis_functions,
              invalid_integral_index,
              delta_h1e,
              density_gradient);
        } else {
          apply_fused_ao_effective_one_electron_integral_loop_single_thread<
              true,
              false,
              true>(
              inactive_density_matrix.data(),
              symmetrized_pullback_source->data(),
              ao_integral_input,
              column_offsets_data,
              n_basis_functions,
              invalid_integral_index,
              delta_h1e,
              density_gradient);
        }
      } else if (symmetry_shifts != nullptr) {
        apply_fused_ao_effective_one_electron_integral_loop_single_thread<
            false,
            true,
            true>(
            inactive_density_matrix.data(),
            symmetrized_pullback_source->data(),
            ao_integral_input,
            column_offsets_data,
            n_basis_functions,
            invalid_integral_index,
            delta_h1e,
            density_gradient);
      } else {
        apply_fused_ao_effective_one_electron_integral_loop_single_thread<
            false,
            false,
            true>(
            inactive_density_matrix.data(),
            symmetrized_pullback_source->data(),
            ao_integral_input,
            column_offsets_data,
            n_basis_functions,
            invalid_integral_index,
            delta_h1e,
            density_gradient);
      }
    } else if (validate_integral_indices) {
      if (symmetry_shifts != nullptr) {
        apply_fused_ao_effective_one_electron_integral_loop_single_thread<
            true,
            true,
            false>(
            inactive_density_matrix.data(),
            symmetrized_pullback_source->data(),
            ao_integral_input,
            column_offsets_data,
            n_basis_functions,
            invalid_integral_index,
            delta_h1e,
            density_gradient);
      } else {
        apply_fused_ao_effective_one_electron_integral_loop_single_thread<
            true,
            false,
            false>(
            inactive_density_matrix.data(),
            symmetrized_pullback_source->data(),
            ao_integral_input,
            column_offsets_data,
            n_basis_functions,
            invalid_integral_index,
            delta_h1e,
            density_gradient);
      }
    } else if (symmetry_shifts != nullptr) {
      apply_fused_ao_effective_one_electron_integral_loop_single_thread<
          false,
          true,
          false>(
          inactive_density_matrix.data(),
          symmetrized_pullback_source->data(),
          ao_integral_input,
          column_offsets_data,
          n_basis_functions,
          invalid_integral_index,
          delta_h1e,
          density_gradient);
    } else {
      apply_fused_ao_effective_one_electron_integral_loop_single_thread<
          false,
          false,
          false>(
          inactive_density_matrix.data(),
          symmetrized_pullback_source->data(),
          ao_integral_input,
          column_offsets_data,
          n_basis_functions,
          invalid_integral_index,
          delta_h1e,
          density_gradient);
    }
  } else {
    std::vector<Eigen::MatrixXd> owned_partial_delta_h1e_workspaces;
    std::vector<Eigen::MatrixXd> owned_partial_density_gradient_workspaces;
    if (partial_delta_h1e_workspaces == nullptr) {
      partial_delta_h1e_workspaces =
          &owned_partial_delta_h1e_workspaces;
    }
    if (partial_density_gradient_workspaces == nullptr) {
      partial_density_gradient_workspaces =
          &owned_partial_density_gradient_workspaces;
    }
    resize_zero_fused_ao_h1e_partial_workspaces(
        n_threads,
        n_basis_functions,
        partial_delta_h1e_workspaces,
        partial_density_gradient_workspaces);

    if (linear_indices != nullptr) {
      if (validate_integral_indices) {
        if (symmetry_shifts != nullptr) {
          apply_fused_ao_effective_one_electron_integral_loop_parallel<
              true,
              true,
              true>(
              inactive_density_matrix.data(),
              symmetrized_pullback_source->data(),
              ao_integral_input,
              column_offsets_data,
              n_basis_functions,
              n_threads,
              invalid_integral_index,
              partial_delta_h1e_workspaces,
              partial_density_gradient_workspaces);
        } else {
          apply_fused_ao_effective_one_electron_integral_loop_parallel<
              true,
              false,
              true>(
              inactive_density_matrix.data(),
              symmetrized_pullback_source->data(),
              ao_integral_input,
              column_offsets_data,
              n_basis_functions,
              n_threads,
              invalid_integral_index,
              partial_delta_h1e_workspaces,
              partial_density_gradient_workspaces);
        }
      } else if (symmetry_shifts != nullptr) {
        apply_fused_ao_effective_one_electron_integral_loop_parallel<
            false,
            true,
            true>(
            inactive_density_matrix.data(),
            symmetrized_pullback_source->data(),
            ao_integral_input,
            column_offsets_data,
            n_basis_functions,
            n_threads,
            invalid_integral_index,
            partial_delta_h1e_workspaces,
            partial_density_gradient_workspaces);
      } else {
        apply_fused_ao_effective_one_electron_integral_loop_parallel<
            false,
            false,
            true>(
            inactive_density_matrix.data(),
            symmetrized_pullback_source->data(),
            ao_integral_input,
            column_offsets_data,
            n_basis_functions,
            n_threads,
            invalid_integral_index,
            partial_delta_h1e_workspaces,
            partial_density_gradient_workspaces);
      }
    } else if (validate_integral_indices) {
      if (symmetry_shifts != nullptr) {
        apply_fused_ao_effective_one_electron_integral_loop_parallel<
            true,
            true,
            false>(
            inactive_density_matrix.data(),
            symmetrized_pullback_source->data(),
            ao_integral_input,
            column_offsets_data,
            n_basis_functions,
            n_threads,
            invalid_integral_index,
            partial_delta_h1e_workspaces,
            partial_density_gradient_workspaces);
      } else {
        apply_fused_ao_effective_one_electron_integral_loop_parallel<
            true,
            false,
            false>(
            inactive_density_matrix.data(),
            symmetrized_pullback_source->data(),
            ao_integral_input,
            column_offsets_data,
            n_basis_functions,
            n_threads,
            invalid_integral_index,
            partial_delta_h1e_workspaces,
            partial_density_gradient_workspaces);
      }
    } else if (symmetry_shifts != nullptr) {
      apply_fused_ao_effective_one_electron_integral_loop_parallel<
          false,
          true,
          false>(
          inactive_density_matrix.data(),
          symmetrized_pullback_source->data(),
          ao_integral_input,
          column_offsets_data,
          n_basis_functions,
          n_threads,
          invalid_integral_index,
          partial_delta_h1e_workspaces,
          partial_density_gradient_workspaces);
    } else {
      apply_fused_ao_effective_one_electron_integral_loop_parallel<
          false,
          false,
          false>(
          inactive_density_matrix.data(),
          symmetrized_pullback_source->data(),
          ao_integral_input,
          column_offsets_data,
          n_basis_functions,
          n_threads,
          invalid_integral_index,
          partial_delta_h1e_workspaces,
          partial_density_gradient_workspaces);
    }

    Eigen::Map<Eigen::MatrixXd> delta_ao_effective_h1e(
        delta_ao_effective_h1e_storage->data(),
        n_basis_functions,
        n_basis_functions);
    Eigen::Map<Eigen::MatrixXd> inactive_density_gradient(
        inactive_density_gradient_storage->data(),
        n_basis_functions,
        n_basis_functions);
    for (const Eigen::MatrixXd& partial_delta_h1e_matrix :
         *partial_delta_h1e_workspaces) {
      delta_ao_effective_h1e += partial_delta_h1e_matrix;
    }
    for (const Eigen::MatrixXd& partial_density_gradient_matrix :
         *partial_density_gradient_workspaces) {
      inactive_density_gradient += partial_density_gradient_matrix;
    }
  }

  if (validate_integral_indices && invalid_integral_index.load() >= 0) {
    throw std::invalid_argument("AO two-electron index out of range");
  }
  Eigen::Map<Eigen::MatrixXd> delta_ao_effective_h1e(
      delta_ao_effective_h1e_storage->data(),
      n_basis_functions,
      n_basis_functions);
  for (int row = 0; row < n_basis_functions; ++row) {
    for (int column = 0; column <= row; ++column) {
      delta_ao_effective_h1e(row, column) =
          delta_ao_effective_h1e(row, column) +
          delta_ao_effective_h1e(column, row);
      delta_ao_effective_h1e(column, row) =
          delta_ao_effective_h1e(row, column);
    }
  }
}

FusedAoEffectiveOneElectronDirectionalResult
apply_fused_exact_ao_effective_one_electron_directional_operator(
    const Eigen::MatrixXd& inactive_density_matrix,
    const Eigen::MatrixXd& ao_effective_one_electron_gradient,
    const AoIntegralInput& ao_integral_input,
    const OrbitalPreparationInput& orbital_preparation_input,
    bool compute_outer_response) {
  FusedAoEffectiveOneElectronDirectionalResult result;
  Eigen::MatrixXd symmetrized_pullback_source;
  apply_fused_exact_ao_effective_one_electron_directional_operator(
      inactive_density_matrix,
      ao_effective_one_electron_gradient,
      ao_integral_input,
      orbital_preparation_input,
      compute_outer_response,
      &symmetrized_pullback_source,
      &result.delta_ao_effective_h1e,
      &result.inactive_density_gradient);
  return result;
}

double elapsed_wall_time_seconds(
    const std::chrono::steady_clock::time_point& start_time) {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now() - start_time)
      .count();
}

struct ProcessResidentSetSnapshot {
  std::size_t vmrss_bytes = 0;
  std::size_t vmhwm_bytes = 0;
};

bool exact_ctx_apply_rss_logging_enabled() {
  const char* flag = std::getenv("XMVB_CPP_LOG_EXACT_CTX_APPLY_RSS");
  if (flag == nullptr || flag[0] == '\0') {
    return false;
  }
  return std::strcmp(flag, "0") != 0 &&
      std::strcmp(flag, "false") != 0 &&
      std::strcmp(flag, "FALSE") != 0;
}

std::size_t exact_ctx_apply_rss_logging_max_applies() {
  const char* value = std::getenv("XMVB_CPP_LOG_EXACT_CTX_APPLY_RSS_MAX_APPLIES");
  if (value == nullptr || value[0] == '\0') {
    return 1;
  }
  const long long parsed = std::atoll(value);
  if (parsed <= 0) {
    throw std::invalid_argument(
        "XMVB_CPP_LOG_EXACT_CTX_APPLY_RSS_MAX_APPLIES must be positive");
  }
  return static_cast<std::size_t>(parsed);
}

bool exact_ctx_structure_stage_logging_enabled() {
  const char* flag = std::getenv("XMVB_CPP_LOG_EXACT_CTX_STRUCTURE_STAGE");
  if (flag == nullptr || flag[0] == '\0') {
    return false;
  }
  return std::strcmp(flag, "0") != 0 &&
      std::strcmp(flag, "false") != 0 &&
      std::strcmp(flag, "FALSE") != 0;
}

ProcessResidentSetSnapshot read_process_resident_set_snapshot() {
  ProcessResidentSetSnapshot snapshot;
  std::FILE* status = std::fopen("/proc/self/status", "r");
  if (status == nullptr) {
    return snapshot;
  }
  char line[512];
  while (std::fgets(line, sizeof(line), status) != nullptr) {
    long long kibibytes = 0;
    if (std::sscanf(line, "VmRSS: %lld kB", &kibibytes) == 1) {
      snapshot.vmrss_bytes =
          static_cast<std::size_t>(kibibytes) * static_cast<std::size_t>(1024);
    } else if (std::sscanf(line, "VmHWM: %lld kB", &kibibytes) == 1) {
      snapshot.vmhwm_bytes =
          static_cast<std::size_t>(kibibytes) * static_cast<std::size_t>(1024);
    }
  }
  std::fclose(status);
  return snapshot;
}

void maybe_log_exact_ctx_apply_rss_stage(
    const char* stage,
    std::size_t apply_index,
    double elapsed_seconds) {
  if (!exact_ctx_apply_rss_logging_enabled()) {
    return;
  }
  if (stage == nullptr || stage[0] == '\0') {
    stage = "unknown";
  }
  const ProcessResidentSetSnapshot snapshot =
      read_process_resident_set_snapshot();
  const double vmrss_mib =
      static_cast<double>(snapshot.vmrss_bytes) / (1024.0 * 1024.0);
  const double vmhwm_mib =
      static_cast<double>(snapshot.vmhwm_bytes) / (1024.0 * 1024.0);
  std::fprintf(
      stderr,
      "[exact_ctx_apply_rss] apply=%zu stage=%s elapsed=%.6f rss=%.2f MiB hwm=%.2f MiB\n",
      apply_index,
      stage,
      elapsed_seconds,
      vmrss_mib,
      vmhwm_mib);
  std::fflush(stderr);
}

Eigen::MatrixXd invert_self_adjoint_positive_definite(
    const Eigen::MatrixXd& matrix,
    const char* label) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument(std::string(label) + " must be square");
  }
  Eigen::LDLT<Eigen::MatrixXd> ldlt(matrix);
  if (ldlt.info() != Eigen::Success) {
    throw std::runtime_error(std::string("failed LDLT factorization for ") + label);
  }
  Eigen::MatrixXd inverse =
      ldlt.solve(Eigen::MatrixXd::Identity(matrix.rows(), matrix.cols()));
  if (ldlt.info() != Eigen::Success) {
    throw std::runtime_error(std::string("failed inverse solve for ") + label);
  }
  return inverse;
}

void throw_if_nonfinite(
    const std::vector<double>& values,
    const char* label) {
  const auto nonfinite_it =
      std::find_if(
          values.begin(),
          values.end(),
          [](double value) { return !std::isfinite(value); });
  if (nonfinite_it == values.end()) {
    return;
  }
  throw std::runtime_error(
      std::string(label) + " contains non-finite values");
}

void throw_if_nonfinite(
    const Eigen::MatrixXd& values,
    const char* label) {
  if (values.allFinite()) {
    return;
  }
  throw std::runtime_error(
      std::string(label) + " contains non-finite values");
}

void throw_if_nonfinite(
    const Eigen::VectorXd& values,
    const char* label) {
  if (values.allFinite()) {
    return;
  }
  throw std::runtime_error(
      std::string(label) + " contains non-finite values");
}

struct DenseOrbitalTangentContext {
  Eigen::MatrixXd normalized_orbitals;
  Eigen::MatrixXd delta_normalized_orbitals;
  Eigen::MatrixXd internal_inactive_orbitals;
  Eigen::MatrixXd delta_internal_inactive_orbitals;
  Eigen::MatrixXd delta_internal_active_auxiliary_orbitals;
  std::vector<double> inverse_norms;
  std::vector<double> normalization_direction_projections;
  bool uses_internal_inactive_chart = false;
};


AcceptedOrbitalPreparationCache build_accepted_orbital_preparation_cache(
    const OrbitalPreparationInput& input,
    const OrbitalPreparationResult& orbital_result,
    const Eigen::Ref<const Eigen::MatrixXd>& total_active_auxiliary_gradient,
    const std::vector<double>& total_inactive_density_gradient) {
  const int n_basis_functions = input.n_basis_functions;
  const int n_orbitals = input.n_orbitals;
  const int n_active_orbitals = input.n_active_orbitals;
  const int n_inactive_doubly_occupied_orbitals =
      (input.n_total_electrons - input.n_active_electrons) / 2;

  const Eigen::Map<const Eigen::MatrixXd> basis_overlap(
      input.active_orbital_overlap_matrix.data(),
      n_basis_functions,
      n_basis_functions);

  AcceptedOrbitalPreparationCache cache;
  cache.normalized_orbitals =
      Eigen::MatrixXd::Zero(n_basis_functions, n_orbitals);
  cache.inverse_norms.assign(n_orbitals, 0.0);
  cache.orbital_basis_function_indices.resize(n_orbitals);
  cache.orbital_coefficient_counts.resize(n_orbitals, 0);
  cache.orbital_overlap_submatrices.resize(n_orbitals);

  for (int orbital_index = 0; orbital_index < n_orbitals; ++orbital_index) {
    Eigen::VectorXd dense_raw =
        Eigen::VectorXd::Zero(n_basis_functions);
    const int coefficient_count =
        stored_sparse_orbital_coefficient_count(input, orbital_index);
    cache.orbital_coefficient_counts[orbital_index] =
        coefficient_count;
    auto& basis_indices =
        cache.orbital_basis_function_indices[orbital_index];
    basis_indices.resize(coefficient_count);
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      const int basis_function_index =
          input.orbital_basis_index_table[orbital_index *
                                              n_basis_functions +
                                          coefficient_index] -
          1;
      if (basis_function_index < 0 || basis_function_index >= n_basis_functions) {
        throw std::runtime_error("invalid sparse orbital basis index in cache build");
      }
      basis_indices[coefficient_index] = basis_function_index;
      const int flat_index =
          orbital_index * n_basis_functions + coefficient_index;
      dense_raw[basis_function_index] =
          input.orbital_value_table[flat_index];
    }

    const double squared_norm = dense_raw.dot(basis_overlap * dense_raw);
    if (!(squared_norm > 0.0) || !std::isfinite(squared_norm)) {
      throw std::runtime_error("orbital normalization failed in cache build");
    }
    const double norm = std::sqrt(squared_norm);
    cache.normalized_orbitals.col(orbital_index) = dense_raw / norm;
    cache.inverse_norms[orbital_index] = 1.0 / norm;

    auto& overlap_sub =
        cache.orbital_overlap_submatrices[orbital_index];
    overlap_sub = Eigen::MatrixXd::Zero(coefficient_count, coefficient_count);
    for (int row = 0; row < coefficient_count; ++row) {
      for (int column = 0; column < coefficient_count; ++column) {
        overlap_sub(row, column) = basis_overlap(
            basis_indices[row],
            basis_indices[column]);
      }
    }
  }

  cache.basis_overlap_times_normalized =
      basis_overlap * cache.normalized_orbitals;
  cache.has_inactive_orbitals = n_inactive_doubly_occupied_orbitals > 0;

  if (cache.has_inactive_orbitals) {
    const ExactCtxInternalInactiveChart internal_chart =
        build_exact_ctx_internal_inactive_chart(
            input,
            orbital_result,
            n_inactive_doubly_occupied_orbitals,
            n_active_orbitals);
    cache.uses_internal_inactive_chart = internal_chart.enabled;
    if (cache.uses_internal_inactive_chart) {
      // In the internal chart the inactive block is represented directly by
      // the accepted orthonormal frame `Q_i`, so the inactive density is
      // `P_i = Q_i Q_i^T` and the active occupied variable is the already
      // projected auxiliary block `T_a`.
      cache.internal_inactive_orbitals = internal_chart.inactive_orbitals;
      cache.selector_inactive_right_inverse_transform =
          internal_chart.inactive_right_inverse_transform;
      cache.selector_inactive_inverse_transpose_right_transform =
          internal_chart.inactive_inverse_transpose_right_transform;
      cache.selector_active_inactive_coefficients =
          internal_chart.active_inactive_coefficients;
      cache.inactive_overlap_inverse =
          Eigen::MatrixXd::Identity(
              n_inactive_doubly_occupied_orbitals,
              n_inactive_doubly_occupied_orbitals);
      cache.has_orthonormal_inactive_chart = true;
      cache.inactive_auxiliary = internal_chart.inactive_orbitals;
      cache.inactive_density =
          internal_chart.inactive_orbitals *
          internal_chart.inactive_orbitals.transpose();
    } else {
      const auto inactive_orbitals =
          cache.normalized_orbitals.leftCols(
              n_inactive_doubly_occupied_orbitals);
      const Eigen::MatrixXd inactive_overlap =
          inactive_orbitals.transpose() * basis_overlap * inactive_orbitals;
      cache.inactive_overlap_inverse =
          invert_self_adjoint_positive_definite(
              inactive_overlap,
              "cached_inactive_overlap");
      cache.has_orthonormal_inactive_chart =
          exact_ctx_uses_orthonormal_inactive_chart(
              input,
              n_inactive_doubly_occupied_orbitals,
              cache.inactive_overlap_inverse);
      cache.inactive_auxiliary =
          build_inactive_auxiliary(
              inactive_orbitals,
              cache.inactive_overlap_inverse,
              cache.has_orthonormal_inactive_chart);
      cache.inactive_density =
          cache.inactive_auxiliary * inactive_orbitals.transpose();
      cache.internal_inactive_orbitals = inactive_orbitals;
      cache.selector_inactive_right_inverse_transform =
          Eigen::MatrixXd::Zero(
              n_inactive_doubly_occupied_orbitals,
              n_inactive_doubly_occupied_orbitals);
      cache.selector_inactive_inverse_transpose_right_transform =
          Eigen::MatrixXd::Zero(
              n_inactive_doubly_occupied_orbitals,
              n_inactive_doubly_occupied_orbitals);
      cache.selector_active_inactive_coefficients =
          Eigen::MatrixXd::Zero(
              n_inactive_doubly_occupied_orbitals,
              n_active_orbitals);
    }
  } else {
    cache.internal_inactive_orbitals =
        Eigen::MatrixXd::Zero(n_basis_functions, 0);
    cache.selector_inactive_right_inverse_transform =
        Eigen::MatrixXd::Zero(0, 0);
    cache.selector_inactive_inverse_transpose_right_transform =
        Eigen::MatrixXd::Zero(0, 0);
    cache.selector_active_inactive_coefficients =
        Eigen::MatrixXd::Zero(0, n_active_orbitals);
    cache.inactive_density =
        Eigen::MatrixXd::Zero(n_basis_functions, n_basis_functions);
  }

  const std::size_t ao_matrix_size =
      n_basis_functions * n_basis_functions;
  cache.has_pullback_cache =
      total_active_auxiliary_gradient.rows() == n_basis_functions &&
      total_active_auxiliary_gradient.cols() == n_active_orbitals &&
      total_inactive_density_gradient.size() == ao_matrix_size;
  if (cache.has_pullback_cache) {
    const Eigen::Map<const Eigen::MatrixXd> inactive_density_gradient_matrix(
        total_inactive_density_gradient.data(),
        n_basis_functions,
        n_basis_functions);
    cache.original_orbital_gradient =
        Eigen::MatrixXd::Zero(n_basis_functions, n_orbitals);

    if (!cache.has_inactive_orbitals) {
      cache.original_orbital_gradient.middleCols(
          0,
          n_active_orbitals) = total_active_auxiliary_gradient;
    } else if (cache.uses_internal_inactive_chart) {
      cache.inactive_density_gradient_symmetric =
          inactive_density_gradient_matrix +
          inactive_density_gradient_matrix.transpose();
      const Eigen::MatrixXd internal_inactive_gradient =
          build_internal_inactive_density_pullback_gradient(
              cache.inactive_density_gradient_symmetric,
              cache.internal_inactive_orbitals);
      const ExactCtxInternalInactiveChart internal_chart = {
          cache.internal_inactive_orbitals,
          cache.selector_inactive_right_inverse_transform,
          cache.selector_inactive_inverse_transpose_right_transform,
          cache.selector_active_inactive_coefficients,
          true};
      const PhysicalOrbitalGradientBlocks physical_gradient =
          transport_internal_chart_gradient_to_physical(
              internal_inactive_gradient,
              total_active_auxiliary_gradient,
              internal_chart);
      cache.original_orbital_gradient.leftCols(
          n_inactive_doubly_occupied_orbitals) =
          physical_gradient.inactive_gradient;
      cache.original_orbital_gradient.middleCols(
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals) =
          physical_gradient.active_gradient;
    } else {
      const auto inactive_orbitals =
          cache.normalized_orbitals.leftCols(
              n_inactive_doubly_occupied_orbitals);
      const auto active_orbitals =
          cache.normalized_orbitals.middleCols(
              n_inactive_doubly_occupied_orbitals,
              n_active_orbitals);
      const Eigen::MatrixXd original_active_gradient =
          total_active_auxiliary_gradient -
          basis_overlap * cache.inactive_density * total_active_auxiliary_gradient;
      const Eigen::MatrixXd bs_active = basis_overlap * active_orbitals;
      const Eigen::MatrixXd total_inactive_gradient =
          inactive_density_gradient_matrix -
          total_active_auxiliary_gradient * bs_active.transpose();
      cache.inactive_density_gradient_symmetric =
          total_inactive_gradient + total_inactive_gradient.transpose();
      const Eigen::MatrixXd basis_overlap_times_inactive =
          basis_overlap * inactive_orbitals;
      Eigen::MatrixXd original_inactive_gradient =
          Eigen::MatrixXd::Zero(
              n_basis_functions,
              n_inactive_doubly_occupied_orbitals);
      if (cache.has_orthonormal_inactive_chart) {
        // The accepted inactive block is orthonormalized, but fixed-upstream
        // still differentiates with respect to the raw inactive coefficients.
        // Reuse the exact `M = I` raw-chart formula instead of the overly
        // aggressive generalized-Stiefel shortcut.
        original_inactive_gradient =
            build_identity_metric_inactive_projector_pullback_gradient(
                cache.inactive_density_gradient_symmetric,
                inactive_orbitals,
                basis_overlap_times_inactive);
      } else {
        const Eigen::MatrixXd inactive_overlap_gradient =
            build_inactive_overlap_gradient(
                cache.inactive_overlap_inverse,
                inactive_orbitals,
                total_inactive_gradient);
        original_inactive_gradient =
            build_inactive_projector_pullback_gradient(
                cache.inactive_density_gradient_symmetric,
                inactive_orbitals,
                cache.inactive_auxiliary,
                basis_overlap_times_inactive,
                inactive_overlap_gradient);
      }
      cache.original_orbital_gradient.leftCols(
          n_inactive_doubly_occupied_orbitals) = original_inactive_gradient;
      cache.original_orbital_gradient.middleCols(
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals) = original_active_gradient;
    }
  }

  return cache;
}

std::vector<double> backpropagate_normalization_to_raw_slots_cached(
    const Eigen::Ref<const Eigen::MatrixXd>& original_orbital_gradient,
    const OrbitalPreparationInput& input,
    const AcceptedOrbitalPreparationCache& cache) {
  std::vector<double> orbital_value_gradient(input.orbital_value_table.size(), 0.0);

#pragma omp parallel for schedule(static)
  for (int orbital_index = 0; orbital_index < input.n_orbitals; ++orbital_index) {
    const auto& basis_indices =
        cache.orbital_basis_function_indices[orbital_index];
    const int coefficient_count =
        cache.orbital_coefficient_counts[orbital_index];
    Eigen::VectorXd local_normalized =
        Eigen::VectorXd::Zero(coefficient_count);
    Eigen::VectorXd local_bs_normalized =
        Eigen::VectorXd::Zero(coefficient_count);
    Eigen::VectorXd local_gradient =
        Eigen::VectorXd::Zero(coefficient_count);

    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      const int basis_function_index =
          basis_indices[coefficient_index];
      local_normalized(coefficient_index) =
          cache.normalized_orbitals(basis_function_index, orbital_index);
      local_bs_normalized(coefficient_index) =
          cache.basis_overlap_times_normalized(basis_function_index, orbital_index);
      local_gradient(coefficient_index) =
          original_orbital_gradient(basis_function_index, orbital_index);
    }

    const double inverse_norm =
        cache.inverse_norms[orbital_index];
    const double scalar_term =
        local_gradient.dot(local_normalized);
    const Eigen::VectorXd raw_gradient =
        inverse_norm *
        (local_gradient - scalar_term * local_bs_normalized);

    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      orbital_value_gradient[orbital_index *
                                 input.n_basis_functions +
                             coefficient_index] =
          raw_gradient(coefficient_index);
    }
  }

  return orbital_value_gradient;
}

std::vector<double> backpropagate_normalization_to_raw_slots_uncached(
    const Eigen::Ref<const Eigen::MatrixXd>& original_orbital_gradient,
    const OrbitalPreparationInput& input,
    const Eigen::Ref<const Eigen::MatrixXd>& normalized_orbitals) {
  if (original_orbital_gradient.rows() != input.n_basis_functions ||
      original_orbital_gradient.cols() != input.n_orbitals ||
      normalized_orbitals.rows() != input.n_basis_functions ||
      normalized_orbitals.cols() != input.n_orbitals) {
    throw std::invalid_argument(
        "uncached orbital normalization pullback has inconsistent dimensions");
  }

  const Eigen::Map<const Eigen::MatrixXd> basis_overlap(
      input.active_orbital_overlap_matrix.data(),
      input.n_basis_functions,
      input.n_basis_functions);
  const Eigen::MatrixXd basis_overlap_times_normalized =
      basis_overlap * normalized_orbitals;
  std::vector<double> orbital_value_gradient(input.orbital_value_table.size(), 0.0);

  for (int orbital_index = 0; orbital_index < input.n_orbitals; ++orbital_index) {
    const int coefficient_count =
        stored_sparse_orbital_coefficient_count(input, orbital_index);
    Eigen::VectorXd local_normalized =
        Eigen::VectorXd::Zero(coefficient_count);
    Eigen::VectorXd local_bs_normalized =
        Eigen::VectorXd::Zero(coefficient_count);
    Eigen::VectorXd local_gradient =
        Eigen::VectorXd::Zero(coefficient_count);
    Eigen::MatrixXd overlap_submatrix =
        Eigen::MatrixXd::Zero(coefficient_count, coefficient_count);
    double squared_norm = 0.0;

    for (int row = 0; row < coefficient_count; ++row) {
      const int row_basis_function_index =
          input.orbital_basis_index_table[orbital_index * input.n_basis_functions + row] -1;
      if (row_basis_function_index < 0 ||
          row_basis_function_index >= input.n_basis_functions) {
        throw std::runtime_error(
            "invalid sparse orbital basis index while uncached normalization pullback");
      }
      const double row_raw =
          input.orbital_value_table[orbital_index *
                                        input.n_basis_functions +
                                    row];
      local_normalized(row) =
          normalized_orbitals(row_basis_function_index, orbital_index);
      local_bs_normalized(row) =
          basis_overlap_times_normalized(row_basis_function_index, orbital_index);
      local_gradient(row) =
          original_orbital_gradient(row_basis_function_index, orbital_index);
      for (int column = 0; column < coefficient_count; ++column) {
        const int column_basis_function_index =
            input.orbital_basis_index_table[orbital_index *
                                                input.n_basis_functions +
                                            column] -
            1;
        overlap_submatrix(row, column) =
            basis_overlap(
                row_basis_function_index,
                column_basis_function_index);
        const double column_raw =
            input.orbital_value_table[orbital_index *
                                          input.n_basis_functions +
                                      column];
        squared_norm +=
            row_raw * overlap_submatrix(row, column) * column_raw;
      }
    }

    if (!(squared_norm > 0.0) || !std::isfinite(squared_norm)) {
      throw std::runtime_error(
          "orbital normalization failed in uncached exact_ctx pullback");
    }
    const double inverse_norm = 1.0 / std::sqrt(squared_norm);
    const double scalar_term = local_gradient.dot(local_normalized);
    const Eigen::VectorXd raw_gradient =
        inverse_norm *
        (local_gradient - scalar_term * local_bs_normalized);
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      orbital_value_gradient[orbital_index *
                                 input.n_basis_functions +
                             coefficient_index] =
          raw_gradient(coefficient_index);
    }
  }

  return orbital_value_gradient;
}

std::vector<double> backpropagate_active_space_orbital_gradient_internal_chart(
    const Eigen::Ref<const Eigen::MatrixXd>& active_auxiliary_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_gradient,
    const OrbitalPreparationInput& input,
    const Eigen::Ref<const Eigen::MatrixXd>& physical_normalized_orbitals,
    const ExactCtxInternalInactiveChart& internal_chart) {
  const int n_inactive_doubly_occupied_orbitals =
      (input.n_total_electrons - input.n_active_electrons) / 2;
  Eigen::MatrixXd original_orbital_gradient =
      Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_orbitals);
  if (n_inactive_doubly_occupied_orbitals == 0) {
    original_orbital_gradient.middleCols(0, input.n_active_orbitals) =
        active_auxiliary_gradient;
  } else {
    const Eigen::MatrixXd internal_inactive_density_gradient_symmetric =
        inactive_density_gradient + inactive_density_gradient.transpose();
    const Eigen::MatrixXd internal_inactive_gradient =
        build_internal_inactive_density_pullback_gradient(
            internal_inactive_density_gradient_symmetric,
            internal_chart.inactive_orbitals);
    const PhysicalOrbitalGradientBlocks physical_gradient =
        transport_internal_chart_gradient_to_physical(
            internal_inactive_gradient,
            active_auxiliary_gradient,
            internal_chart);
    original_orbital_gradient.leftCols(n_inactive_doubly_occupied_orbitals) =
        physical_gradient.inactive_gradient;
    original_orbital_gradient.middleCols(
        n_inactive_doubly_occupied_orbitals,
        input.n_active_orbitals) =
        physical_gradient.active_gradient;
  }

  if (input.orbital_type == kLegacyOrbitalTypeOeo) {
    return scatter_dense_orbital_gradient_to_sparse_slots_local(
        original_orbital_gradient,
        input);
  }

  return backpropagate_normalization_to_raw_slots_uncached(
      original_orbital_gradient,
      input,
      physical_normalized_orbitals);
}

std::vector<double> backpropagate_active_space_orbital_gradient_cached(
    const Eigen::Ref<const Eigen::MatrixXd>& active_auxiliary_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_gradient,
    const OrbitalPreparationInput& input,
    const OrbitalPreparationResult& /*orbital_preparation_result*/,
    const AcceptedOrbitalPreparationCache& cache) {
  if (input.n_basis_functions <= 0 || input.n_orbitals <= 0 ||
      input.n_active_orbitals <= 0) {
    throw std::invalid_argument(
        "cached orbital backprop input dimensions must be positive");
  }
  const int n_inactive_doubly_occupied_orbitals =
      (input.n_total_electrons - input.n_active_electrons) / 2;
  Eigen::MatrixXd original_orbital_gradient =
      Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_orbitals);
  if (n_inactive_doubly_occupied_orbitals == 0) {
    original_orbital_gradient.middleCols(0, input.n_active_orbitals) =
        active_auxiliary_gradient;
  } else if (cache.uses_internal_inactive_chart) {
    const Eigen::MatrixXd internal_inactive_density_gradient_symmetric =
        inactive_density_gradient + inactive_density_gradient.transpose();
    const Eigen::MatrixXd internal_inactive_gradient =
        build_internal_inactive_density_pullback_gradient(
            internal_inactive_density_gradient_symmetric,
            cache.internal_inactive_orbitals);
    const ExactCtxInternalInactiveChart internal_chart = {
        cache.internal_inactive_orbitals,
        cache.selector_inactive_right_inverse_transform,
        cache.selector_inactive_inverse_transpose_right_transform,
        cache.selector_active_inactive_coefficients,
        true};
    const PhysicalOrbitalGradientBlocks physical_gradient =
        transport_internal_chart_gradient_to_physical(
            internal_inactive_gradient,
            active_auxiliary_gradient,
            internal_chart);
    original_orbital_gradient.leftCols(n_inactive_doubly_occupied_orbitals) =
        physical_gradient.inactive_gradient;
    original_orbital_gradient.middleCols(
        n_inactive_doubly_occupied_orbitals,
        input.n_active_orbitals) =
        physical_gradient.active_gradient;
  } else {
    const auto inactive_orbitals =
        cache.normalized_orbitals.leftCols(n_inactive_doubly_occupied_orbitals);
    const Eigen::MatrixXd basis_overlap_times_inactive =
        cache.basis_overlap_times_normalized.leftCols(
            n_inactive_doubly_occupied_orbitals);
    const Eigen::MatrixXd basis_overlap_times_active =
        cache.basis_overlap_times_normalized.middleCols(
            n_inactive_doubly_occupied_orbitals,
            input.n_active_orbitals);

    // The accepted-point cache already owns the mixed-gauge inactive duals and
    // the accepted physical active-orbital overlap source `S * C_active`.
    // Reusing them removes the repeated sparse norm and accepted-point overlap
    // contractions that previously ran inside every exact_ctx matvec.
    const Eigen::MatrixXd original_active_gradient =
        apply_occupied_projector_transpose_to_gradient(
            active_auxiliary_gradient,
            cache.inactive_auxiliary,
            basis_overlap_times_inactive);
    const Eigen::MatrixXd effective_inactive_density_gradient =
        inactive_density_gradient -
        active_auxiliary_gradient * basis_overlap_times_active.transpose();
    const Eigen::MatrixXd effective_inactive_density_gradient_symmetric =
        effective_inactive_density_gradient +
        effective_inactive_density_gradient.transpose();
    Eigen::MatrixXd original_inactive_gradient =
        Eigen::MatrixXd::Zero(
            input.n_basis_functions,
            n_inactive_doubly_occupied_orbitals);
    if (cache.has_orthonormal_inactive_chart) {
      original_inactive_gradient =
          build_identity_metric_inactive_projector_pullback_gradient(
              effective_inactive_density_gradient_symmetric,
              inactive_orbitals,
              basis_overlap_times_inactive);
    } else {
      const Eigen::MatrixXd inactive_overlap_gradient =
          build_inactive_overlap_gradient(
              cache.inactive_overlap_inverse,
              inactive_orbitals,
              effective_inactive_density_gradient);
      original_inactive_gradient =
          build_inactive_projector_pullback_gradient(
              effective_inactive_density_gradient_symmetric,
              inactive_orbitals,
              cache.inactive_auxiliary,
              basis_overlap_times_inactive,
              inactive_overlap_gradient);
    }

    original_orbital_gradient.leftCols(n_inactive_doubly_occupied_orbitals) =
        original_inactive_gradient;
    original_orbital_gradient.middleCols(
        n_inactive_doubly_occupied_orbitals,
        input.n_active_orbitals) = original_active_gradient;
  }

  if (input.orbital_type == kLegacyOrbitalTypeOeo) {
    return scatter_dense_orbital_gradient_to_sparse_slots_local(
        original_orbital_gradient,
        input);
  }

  return backpropagate_normalization_to_raw_slots_cached(
      original_orbital_gradient,
      input,
      cache);
}

DenseOrbitalTangentContext build_dense_orbital_tangent_context(
    const OrbitalPreparationInput& input,
    const SparseOrbitalParameterView& parameter_view,
    const Eigen::VectorXd& packed_direction,
    const ExactCtxInternalInactiveChart* internal_chart) {
  if (packed_direction.size() != parameter_view.size()) {
    throw std::invalid_argument(
        "packed_direction size does not match sparse orbital parameter view");
  }

  const Eigen::Map<const Eigen::MatrixXd> basis_overlap(
      input.active_orbital_overlap_matrix.data(),
      input.n_basis_functions,
      input.n_basis_functions);

  std::vector<double> full_direction(
      input.n_orbitals * input.n_basis_functions,
      0.0);
  const auto& differentiable_indices =
      parameter_view.differentiable_parameter_indices();
  for (Eigen::Index packed_index = 0;
       packed_index < packed_direction.size();
       ++packed_index) {
    full_direction[
        differentiable_indices[packed_index]] =
        packed_direction[packed_index];
  }

  DenseOrbitalTangentContext result;
  result.normalized_orbitals =
      Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_orbitals);
  result.delta_normalized_orbitals =
      Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_orbitals);
  if (input.n_total_electrons < input.n_active_electrons) {
    throw std::invalid_argument(
        "n_total_electrons must be >= n_active_electrons");
  }
  const Eigen::Index n_internal_inactive_orbitals =
      static_cast<Eigen::Index>(
          (input.n_total_electrons - input.n_active_electrons) / 2);
  result.internal_inactive_orbitals =
      Eigen::MatrixXd::Zero(
          input.n_basis_functions,
          n_internal_inactive_orbitals);
  result.delta_internal_inactive_orbitals =
      Eigen::MatrixXd::Zero(
          input.n_basis_functions,
          n_internal_inactive_orbitals);
  result.delta_internal_active_auxiliary_orbitals =
      Eigen::MatrixXd::Zero(
          input.n_basis_functions,
          input.n_active_orbitals);
  result.inverse_norms.assign(input.n_orbitals, 0.0);
  result.normalization_direction_projections.assign(
      input.n_orbitals,
      0.0);

  for (int orbital_index = 0; orbital_index < input.n_orbitals; ++orbital_index) {
    Eigen::VectorXd dense_raw =
        Eigen::VectorXd::Zero(input.n_basis_functions);
    Eigen::VectorXd dense_direction =
        Eigen::VectorXd::Zero(input.n_basis_functions);
    const int coefficient_count =
        stored_sparse_orbital_coefficient_count(input, orbital_index);
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      const int basis_function_index =
          input.orbital_basis_index_table[orbital_index *
                                              input.n_basis_functions +
                                          coefficient_index] -
          1;
      if (basis_function_index < 0 || basis_function_index >= input.n_basis_functions) {
        throw std::runtime_error("invalid sparse orbital basis index");
      }
      const int flat_index =
          orbital_index * input.n_basis_functions + coefficient_index;
      dense_raw[basis_function_index] =
          input.orbital_value_table[flat_index];
      dense_direction[basis_function_index] =
          full_direction[flat_index];
    }

    const double squared_norm = dense_raw.dot(basis_overlap * dense_raw);
    if (!(squared_norm > 0.0) || !std::isfinite(squared_norm)) {
      throw std::runtime_error("orbital normalization failed in exact core HVP");
    }
    const double norm = std::sqrt(squared_norm);
    const Eigen::VectorXd normalized = dense_raw / norm;
    const double direction_projection =
        dense_raw.dot(basis_overlap * dense_direction) / squared_norm;
    const Eigen::VectorXd delta_normalized =
        dense_direction / norm - normalized * direction_projection;

    result.normalized_orbitals.col(orbital_index) = normalized;
    result.delta_normalized_orbitals.col(orbital_index) = delta_normalized;
    result.inverse_norms[orbital_index] = 1.0 / norm;
    result.normalization_direction_projections[orbital_index] =
        direction_projection;
  }

  const int n_inactive_doubly_occupied_orbitals =
      (input.n_total_electrons - input.n_active_electrons) / 2;
  if (internal_chart != nullptr && internal_chart->enabled) {
    result.uses_internal_inactive_chart = true;
    result.internal_inactive_orbitals = internal_chart->inactive_orbitals;
    if (n_inactive_doubly_occupied_orbitals > 0) {
      const Eigen::MatrixXd physical_delta_inactive =
          result.delta_normalized_orbitals.leftCols(
              n_inactive_doubly_occupied_orbitals);
      result.delta_internal_inactive_orbitals =
          physical_delta_inactive *
          internal_chart->inactive_right_inverse_transform;
    }
    if (input.n_active_orbitals > 0) {
      result.delta_internal_active_auxiliary_orbitals =
          result.delta_normalized_orbitals.middleCols(
              n_inactive_doubly_occupied_orbitals,
              input.n_active_orbitals) -
          result.delta_internal_inactive_orbitals *
              internal_chart->active_inactive_coefficients;
    }
  } else {
    result.internal_inactive_orbitals =
        result.normalized_orbitals.leftCols(
            n_inactive_doubly_occupied_orbitals);
    result.delta_internal_inactive_orbitals =
        result.delta_normalized_orbitals.leftCols(
            n_inactive_doubly_occupied_orbitals);
    if (input.n_active_orbitals > 0) {
      result.delta_internal_active_auxiliary_orbitals =
          result.delta_normalized_orbitals.middleCols(
              n_inactive_doubly_occupied_orbitals,
              input.n_active_orbitals);
    }
  }

  return result;
}

DenseOrbitalTangentContext build_dense_orbital_tangent_context_cached(
    const OrbitalPreparationInput& input,
    const SparseOrbitalParameterView& parameter_view,
    const Eigen::VectorXd& packed_direction,
    const AcceptedOrbitalPreparationCache& cache) {
  DenseOrbitalTangentContext result;
  result.normalized_orbitals = cache.normalized_orbitals;
  result.inverse_norms = cache.inverse_norms;
  result.delta_normalized_orbitals =
      Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_orbitals);
  result.internal_inactive_orbitals = cache.internal_inactive_orbitals;
  result.delta_internal_inactive_orbitals =
      Eigen::MatrixXd::Zero(
          cache.internal_inactive_orbitals.rows(),
          cache.internal_inactive_orbitals.cols());
  result.delta_internal_active_auxiliary_orbitals =
      Eigen::MatrixXd::Zero(
          input.n_basis_functions,
          input.n_active_orbitals);
  result.uses_internal_inactive_chart = cache.uses_internal_inactive_chart;
  result.normalization_direction_projections.assign(
      input.n_orbitals,
      0.0);

  Eigen::Index packed_offset = 0;
  for (int orbital_index = 0; orbital_index < input.n_orbitals; ++orbital_index) {
    const auto& basis_indices =
        cache.orbital_basis_function_indices[orbital_index];
    const int differentiable_coefficient_count =
        parameter_view.orbital_coefficient_count(orbital_index);
    const double inverse_norm =
        cache.inverse_norms[orbital_index];
    const Eigen::Index orbital_packed_offset = packed_offset;
    double direction_projection = 0.0;
    for (int coefficient_index = 0;
         coefficient_index < differentiable_coefficient_count;
         ++coefficient_index, ++packed_offset) {
      const int basis_function_index =
          basis_indices[coefficient_index];
      direction_projection +=
          packed_direction[packed_offset] *
          cache.basis_overlap_times_normalized(
              basis_function_index,
              orbital_index);
    }
    // `delta_normalized = dc / ||c|| - normalized * (c^T S dc) / (c^T S c)`.
    // With `normalized = c / ||c||`, the scalar projection is
    // `inverse_norm * normalized^T S dc`, not `inverse_norm^2 * ...`.
    direction_projection *= inverse_norm;

    result.delta_normalized_orbitals.col(orbital_index).noalias() =
        -direction_projection * cache.normalized_orbitals.col(orbital_index);
    for (int coefficient_index = 0;
         coefficient_index < differentiable_coefficient_count;
         ++coefficient_index) {
      const int basis_function_index =
          basis_indices[coefficient_index];
      result.delta_normalized_orbitals(
          basis_function_index,
          orbital_index) +=
          inverse_norm *
          packed_direction[orbital_packed_offset + coefficient_index];
    }
    result.normalization_direction_projections[orbital_index] =
        direction_projection;
  }
  if (packed_offset != packed_direction.size()) {
    throw std::invalid_argument(
        "packed direction size does not match cached orbital tangent traversal");
  }

  const int n_inactive_doubly_occupied_orbitals =
      (input.n_total_electrons - input.n_active_electrons) / 2;
  if (cache.uses_internal_inactive_chart) {
    if (n_inactive_doubly_occupied_orbitals > 0) {
      const Eigen::MatrixXd physical_delta_inactive =
          result.delta_normalized_orbitals.leftCols(
              n_inactive_doubly_occupied_orbitals);
      result.delta_internal_inactive_orbitals =
          physical_delta_inactive *
          cache.selector_inactive_right_inverse_transform;
    }
    if (input.n_active_orbitals > 0) {
      result.delta_internal_active_auxiliary_orbitals =
          result.delta_normalized_orbitals.middleCols(
              n_inactive_doubly_occupied_orbitals,
              input.n_active_orbitals) -
          result.delta_internal_inactive_orbitals *
              cache.selector_active_inactive_coefficients;
    }
  } else {
    result.delta_internal_inactive_orbitals =
        result.delta_normalized_orbitals.leftCols(
            n_inactive_doubly_occupied_orbitals);
    if (input.n_active_orbitals > 0) {
      result.delta_internal_active_auxiliary_orbitals =
          result.delta_normalized_orbitals.middleCols(
              n_inactive_doubly_occupied_orbitals,
              input.n_active_orbitals);
    }
  }

  return result;
}

struct OrbitalPreparationDirectionalResult {
  Eigen::MatrixXd delta_inactive_density;
  LowRankAoMatrix delta_inactive_density_low_rank;
  Eigen::MatrixXd delta_active_auxiliary_orbitals;
  Eigen::MatrixXd basis_overlap_times_delta_active_orbitals;
};

OrbitalPreparationDirectionalResult build_orbital_preparation_directional_result(
    const OrbitalPreparationInput& input,
    const DenseOrbitalTangentContext& orbital_tangent_context,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) {
  if (input.n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }
  const Eigen::Map<const Eigen::MatrixXd> basis_overlap(
      input.active_orbital_overlap_matrix.data(),
      input.n_basis_functions,
      input.n_basis_functions);

  Eigen::MatrixXd inactive_orbitals =
      Eigen::MatrixXd::Zero(input.n_basis_functions, n_inactive_doubly_occupied_orbitals);
  Eigen::MatrixXd delta_inactive_orbitals =
      Eigen::MatrixXd::Zero(input.n_basis_functions, n_inactive_doubly_occupied_orbitals);
  if (n_inactive_doubly_occupied_orbitals > 0) {
    inactive_orbitals =
        orbital_tangent_context.normalized_orbitals.leftCols(
            n_inactive_doubly_occupied_orbitals);
    delta_inactive_orbitals =
        orbital_tangent_context.delta_normalized_orbitals.leftCols(
            n_inactive_doubly_occupied_orbitals);
  }

  Eigen::MatrixXd active_orbitals =
      Eigen::MatrixXd::Zero(input.n_basis_functions, n_active_orbitals);
  Eigen::MatrixXd delta_active_orbitals =
      Eigen::MatrixXd::Zero(input.n_basis_functions, n_active_orbitals);
  if (n_active_orbitals > 0) {
    active_orbitals =
        orbital_tangent_context.normalized_orbitals.middleCols(
            n_inactive_doubly_occupied_orbitals,
            n_active_orbitals);
    delta_active_orbitals =
        orbital_tangent_context.delta_normalized_orbitals.middleCols(
            n_inactive_doubly_occupied_orbitals,
            n_active_orbitals);
  }

  LowRankAoMatrix delta_inactive_density_low_rank =
      make_empty_low_rank_ao_matrix(input.n_basis_functions);
  OrbitalPreparationDirectionalResult result;
  Eigen::MatrixXd basis_overlap_times_delta_active_orbitals =
      Eigen::MatrixXd::Zero(input.n_basis_functions, n_active_orbitals);
  if (orbital_tangent_context.uses_internal_inactive_chart) {
    if (n_active_orbitals > 0) {
      // Once the packed direction is lifted to `(delta Q_i, delta T_a)`,
      // orbital preparation is trivial: the active auxiliary block is exactly
      // `T_a`, so its direction is just `delta T_a`.
      result.delta_active_auxiliary_orbitals =
          orbital_tangent_context.delta_internal_active_auxiliary_orbitals;
      result.basis_overlap_times_delta_active_orbitals =
          basis_overlap * result.delta_active_auxiliary_orbitals;
    } else {
      result.delta_active_auxiliary_orbitals =
          Eigen::MatrixXd::Zero(input.n_basis_functions, 0);
      result.basis_overlap_times_delta_active_orbitals =
          Eigen::MatrixXd::Zero(input.n_basis_functions, 0);
    }
    if (n_inactive_doubly_occupied_orbitals > 0) {
      // The inactive density in the internal chart is `Q_i Q_i^T`, so the
      // directional density remains the rank-2n_i product
      // `delta Q_i Q_i^T + Q_i delta Q_i^T`.
      delta_inactive_density_low_rank =
          build_inactive_density_direction_low_rank(
              input.n_basis_functions,
              orbital_tangent_context.internal_inactive_orbitals,
              orbital_tangent_context.delta_internal_inactive_orbitals,
              orbital_tangent_context.internal_inactive_orbitals,
              orbital_tangent_context.delta_internal_inactive_orbitals);
    }
    result.delta_inactive_density_low_rank = delta_inactive_density_low_rank;
    result.delta_inactive_density =
        materialize_low_rank_ao_matrix(
            result.delta_inactive_density_low_rank,
            "orbital-preparation internal inactive-density direction");
    return result;
  }
  if (n_active_orbitals > 0) {
    basis_overlap_times_delta_active_orbitals =
        basis_overlap * delta_active_orbitals;
  }

  if (n_inactive_doubly_occupied_orbitals > 0) {
    const Eigen::MatrixXd inactive_overlap =
        inactive_orbitals.transpose() * basis_overlap * inactive_orbitals;
    const Eigen::MatrixXd inactive_overlap_inverse =
        invert_self_adjoint_positive_definite(
            inactive_overlap,
            "inactive_overlap");
    const bool use_orthonormal_inactive_chart =
        exact_ctx_uses_orthonormal_inactive_chart(
            input,
            n_inactive_doubly_occupied_orbitals,
            inactive_overlap_inverse);
    const Eigen::MatrixXd inactive_auxiliary =
        build_inactive_auxiliary(
            inactive_orbitals,
            inactive_overlap_inverse,
            use_orthonormal_inactive_chart);
    const InactiveAuxiliaryDirectionResult inactive_auxiliary_direction =
        build_delta_inactive_auxiliary_direction(
            basis_overlap,
            inactive_orbitals,
            delta_inactive_orbitals,
            inactive_overlap_inverse,
            false);
    delta_inactive_density_low_rank =
        build_inactive_density_direction_low_rank(
            input.n_basis_functions,
            inactive_orbitals,
            delta_inactive_orbitals,
            inactive_auxiliary,
            inactive_auxiliary_direction.delta_inactive_auxiliary);
    const Eigen::MatrixXd basis_overlap_times_active_orbitals =
        basis_overlap * active_orbitals;
    result.basis_overlap_times_delta_active_orbitals =
        std::move(basis_overlap_times_delta_active_orbitals);
    result.delta_active_auxiliary_orbitals =
        apply_occupied_projector_to_orbitals(
            delta_active_orbitals,
            inactive_orbitals,
            inactive_auxiliary,
            result.basis_overlap_times_delta_active_orbitals) -
        apply_inactive_density_direction_to_basis_overlap_times_active(
            inactive_orbitals,
            delta_inactive_orbitals,
            inactive_auxiliary,
            inactive_auxiliary_direction.delta_inactive_auxiliary,
            basis_overlap_times_active_orbitals,
            use_orthonormal_inactive_chart);
  } else {
    result.basis_overlap_times_delta_active_orbitals =
        std::move(basis_overlap_times_delta_active_orbitals);
    result.delta_active_auxiliary_orbitals = delta_active_orbitals;
  }

  result.delta_inactive_density_low_rank = delta_inactive_density_low_rank;
  result.delta_inactive_density =
      materialize_low_rank_ao_matrix(
          result.delta_inactive_density_low_rank,
          "orbital-preparation inactive-density direction");
  return result;
}

OrbitalPreparationDirectionalResult build_orbital_preparation_directional_result_cached(
    const OrbitalPreparationInput& input,
    const DenseOrbitalTangentContext& orbital_tangent_context,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals,
    const AcceptedOrbitalPreparationCache& cache) {
  if (input.n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }
  const Eigen::Map<const Eigen::MatrixXd> basis_overlap(
      input.active_orbital_overlap_matrix.data(),
      input.n_basis_functions,
      input.n_basis_functions);

  LowRankAoMatrix delta_inactive_density_low_rank =
      make_empty_low_rank_ao_matrix(input.n_basis_functions);
  OrbitalPreparationDirectionalResult result;

  Eigen::MatrixXd delta_active_orbitals =
      Eigen::MatrixXd::Zero(input.n_basis_functions, n_active_orbitals);
  result.basis_overlap_times_delta_active_orbitals =
      Eigen::MatrixXd::Zero(input.n_basis_functions, n_active_orbitals);
  if (cache.uses_internal_inactive_chart) {
    if (n_active_orbitals > 0) {
      result.delta_active_auxiliary_orbitals =
          orbital_tangent_context.delta_internal_active_auxiliary_orbitals;
      result.basis_overlap_times_delta_active_orbitals =
          basis_overlap * result.delta_active_auxiliary_orbitals;
    } else {
      result.delta_active_auxiliary_orbitals =
          Eigen::MatrixXd::Zero(input.n_basis_functions, 0);
      result.basis_overlap_times_delta_active_orbitals =
          Eigen::MatrixXd::Zero(input.n_basis_functions, 0);
    }
    if (cache.has_inactive_orbitals && n_inactive_doubly_occupied_orbitals > 0) {
      delta_inactive_density_low_rank =
          build_inactive_density_direction_low_rank(
              input.n_basis_functions,
              cache.internal_inactive_orbitals,
              orbital_tangent_context.delta_internal_inactive_orbitals,
              cache.internal_inactive_orbitals,
              orbital_tangent_context.delta_internal_inactive_orbitals);
    }
    result.delta_inactive_density_low_rank = delta_inactive_density_low_rank;
    result.delta_inactive_density =
        materialize_low_rank_ao_matrix(
            result.delta_inactive_density_low_rank,
            "cached orbital-preparation internal inactive-density direction");
    return result;
  }
  if (n_active_orbitals > 0) {
    delta_active_orbitals =
        orbital_tangent_context.delta_normalized_orbitals.middleCols(
            n_inactive_doubly_occupied_orbitals,
            n_active_orbitals);
    result.basis_overlap_times_delta_active_orbitals =
        basis_overlap * delta_active_orbitals;
  }

  if (cache.has_inactive_orbitals && n_inactive_doubly_occupied_orbitals > 0) {
    const auto inactive_orbitals =
        orbital_tangent_context.normalized_orbitals.leftCols(
            n_inactive_doubly_occupied_orbitals);
    const auto delta_inactive_orbitals =
        orbital_tangent_context.delta_normalized_orbitals.leftCols(
            n_inactive_doubly_occupied_orbitals);
    const InactiveAuxiliaryDirectionResult inactive_auxiliary_direction =
        build_delta_inactive_auxiliary_direction(
            basis_overlap,
            inactive_orbitals,
            delta_inactive_orbitals,
            cache.inactive_overlap_inverse,
            cache.has_orthonormal_inactive_chart);
    delta_inactive_density_low_rank =
        build_inactive_density_direction_low_rank(
            input.n_basis_functions,
            inactive_orbitals,
            delta_inactive_orbitals,
            cache.inactive_auxiliary,
            inactive_auxiliary_direction.delta_inactive_auxiliary);
    const Eigen::MatrixXd basis_overlap_times_active_orbitals =
        cache.basis_overlap_times_normalized.middleCols(
            n_inactive_doubly_occupied_orbitals,
            n_active_orbitals);
    result.delta_active_auxiliary_orbitals =
        apply_occupied_projector_to_orbitals(
            delta_active_orbitals,
            inactive_orbitals,
            cache.inactive_auxiliary,
            result.basis_overlap_times_delta_active_orbitals) -
        apply_inactive_density_direction_to_basis_overlap_times_active(
            inactive_orbitals,
            delta_inactive_orbitals,
            cache.inactive_auxiliary,
            inactive_auxiliary_direction.delta_inactive_auxiliary,
            basis_overlap_times_active_orbitals,
            cache.has_orthonormal_inactive_chart);
  } else {
    result.delta_active_auxiliary_orbitals = delta_active_orbitals;
  }

  result.delta_inactive_density_low_rank = delta_inactive_density_low_rank;
  result.delta_inactive_density =
      materialize_low_rank_ao_matrix(
          result.delta_inactive_density_low_rank,
          "cached orbital-preparation inactive-density direction");
  return result;
}

std::vector<double> apply_fixed_upstream_orbital_pullback_direction(
    const OrbitalPreparationInput& input,
    const DenseOrbitalTangentContext& orbital_tangent_context,
    const Eigen::Ref<const Eigen::MatrixXd>& total_active_auxiliary_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap_times_delta_active_orbitals,
    const std::vector<double>& total_inactive_density_gradient,
    const Eigen::Ref<const Eigen::VectorXd>& input_retract_tangent,
    const ExactCtxInternalInactiveChart* internal_chart) {
  if (input.n_basis_functions <= 0 || input.n_orbitals <= 0 ||
      input.n_active_orbitals <= 0) {
    throw std::invalid_argument(
        "orbital pullback directional input dimensions must be positive");
  }
  if (input_retract_tangent.size() !=
      static_cast<Eigen::Index>(input.orbital_value_table.size())) {
    throw std::invalid_argument(
        "fixed-upstream input tangent size does not match orbital_value_table");
  }

  const int n_inactive_doubly_occupied_orbitals =
      (input.n_total_electrons - input.n_active_electrons) / 2;
  const Eigen::Map<const Eigen::MatrixXd> basis_overlap(
      input.active_orbital_overlap_matrix.data(),
      input.n_basis_functions,
      input.n_basis_functions);
  const Eigen::Map<const Eigen::MatrixXd> inactive_density_gradient_matrix(
      total_inactive_density_gradient.data(),
      input.n_basis_functions,
      input.n_basis_functions);

  const auto inactive_orbitals =
      orbital_tangent_context.normalized_orbitals.leftCols(
          n_inactive_doubly_occupied_orbitals);
  const auto delta_inactive_orbitals =
      orbital_tangent_context.delta_normalized_orbitals.leftCols(
          n_inactive_doubly_occupied_orbitals);
  const auto active_orbitals =
      orbital_tangent_context.normalized_orbitals.middleCols(
          n_inactive_doubly_occupied_orbitals,
          input.n_active_orbitals);

  Eigen::MatrixXd original_orbital_gradient =
      Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_orbitals);
  Eigen::MatrixXd delta_original_orbital_gradient =
      Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_orbitals);

  if (n_inactive_doubly_occupied_orbitals == 0) {
    original_orbital_gradient.middleCols(
        0,
        input.n_active_orbitals) = total_active_auxiliary_gradient;
  } else if (orbital_tangent_context.uses_internal_inactive_chart) {
    if (internal_chart == nullptr || !internal_chart->enabled) {
      throw std::invalid_argument(
          "internal exact_ctx fixed-upstream pullback requires an enabled chart");
    }
    // The fixed-upstream term collapses in the internal chart.  `T_a` is a
    // direct state variable, so only the inactive-density pullback contributes
    // and its directional derivative is simply `G_P,sym * delta Q_i`.
    const Eigen::MatrixXd internal_inactive_density_gradient_symmetric =
        inactive_density_gradient_matrix +
        inactive_density_gradient_matrix.transpose();
    const Eigen::MatrixXd internal_inactive_gradient =
        build_internal_inactive_density_pullback_gradient(
            internal_inactive_density_gradient_symmetric,
            internal_chart->inactive_orbitals);
    const Eigen::MatrixXd delta_internal_inactive_gradient =
        build_internal_inactive_density_pullback_gradient_direction(
            internal_inactive_density_gradient_symmetric,
            orbital_tangent_context.delta_internal_inactive_orbitals);
    const PhysicalOrbitalGradientBlocks physical_gradient =
        transport_internal_chart_gradient_to_physical(
            internal_inactive_gradient,
            total_active_auxiliary_gradient,
            *internal_chart);
    const PhysicalOrbitalGradientBlocks delta_physical_gradient =
        transport_internal_chart_gradient_to_physical(
            delta_internal_inactive_gradient,
            Eigen::MatrixXd::Zero(
                input.n_basis_functions,
                input.n_active_orbitals),
            *internal_chart);
    original_orbital_gradient.leftCols(n_inactive_doubly_occupied_orbitals) =
        physical_gradient.inactive_gradient;
    original_orbital_gradient.middleCols(
        n_inactive_doubly_occupied_orbitals,
        input.n_active_orbitals) =
        physical_gradient.active_gradient;
    delta_original_orbital_gradient.leftCols(
        n_inactive_doubly_occupied_orbitals) =
        delta_physical_gradient.inactive_gradient;
  } else {
    const Eigen::MatrixXd inactive_overlap =
        inactive_orbitals.transpose() * basis_overlap * inactive_orbitals;
    const Eigen::MatrixXd inactive_overlap_inverse =
        invert_self_adjoint_positive_definite(
            inactive_overlap,
            "fixed_upstream_inactive_overlap");
    const bool use_orthonormal_inactive_chart =
        exact_ctx_uses_orthonormal_inactive_chart(
            input,
            n_inactive_doubly_occupied_orbitals,
            inactive_overlap_inverse);
    const Eigen::MatrixXd inactive_auxiliary =
        build_inactive_auxiliary(
            inactive_orbitals,
            inactive_overlap_inverse,
            use_orthonormal_inactive_chart);
    const InactiveAuxiliaryDirectionResult inactive_auxiliary_direction =
        build_delta_inactive_auxiliary_direction(
            basis_overlap,
            inactive_orbitals,
            delta_inactive_orbitals,
            inactive_overlap_inverse,
            false);
    const LowRankAoMatrix inactive_density_low_rank =
        make_low_rank_ao_matrix(
            inactive_auxiliary,
            inactive_orbitals,
            "fixed-upstream inactive density");
    const LowRankAoMatrix delta_inactive_density_low_rank =
        build_inactive_density_direction_low_rank(
            input.n_basis_functions,
            inactive_orbitals,
            delta_inactive_orbitals,
            inactive_auxiliary,
            inactive_auxiliary_direction.delta_inactive_auxiliary);

    const Eigen::MatrixXd original_active_gradient =
        total_active_auxiliary_gradient -
        apply_metric_times_low_rank_ao_matrix(
            basis_overlap,
            inactive_density_low_rank,
            total_active_auxiliary_gradient,
            "fixed-upstream inactive density");
    const Eigen::MatrixXd delta_original_active_gradient =
        -apply_metric_times_low_rank_ao_matrix(
            basis_overlap,
            delta_inactive_density_low_rank,
            total_active_auxiliary_gradient,
            "fixed-upstream inactive-density direction");

    const Eigen::MatrixXd bs_active = basis_overlap * active_orbitals;
    const Eigen::MatrixXd& delta_bs_active =
        basis_overlap_times_delta_active_orbitals;
    const Eigen::MatrixXd total_inactive_gradient =
        inactive_density_gradient_matrix -
        total_active_auxiliary_gradient * bs_active.transpose();
    const Eigen::MatrixXd delta_total_inactive_gradient =
        -total_active_auxiliary_gradient * delta_bs_active.transpose();
    const Eigen::MatrixXd inactive_density_gradient_symmetric =
        total_inactive_gradient + total_inactive_gradient.transpose();
    const Eigen::MatrixXd delta_inactive_density_gradient_symmetric =
        delta_total_inactive_gradient + delta_total_inactive_gradient.transpose();
    const Eigen::MatrixXd basis_overlap_times_inactive =
        basis_overlap * inactive_orbitals;
    const Eigen::MatrixXd delta_basis_overlap_times_inactive =
        basis_overlap * delta_inactive_orbitals;
    Eigen::MatrixXd original_inactive_gradient =
        Eigen::MatrixXd::Zero(
            input.n_basis_functions,
            n_inactive_doubly_occupied_orbitals);
    Eigen::MatrixXd delta_original_inactive_gradient =
        Eigen::MatrixXd::Zero(
            input.n_basis_functions,
            n_inactive_doubly_occupied_orbitals);
    if (use_orthonormal_inactive_chart) {
      original_inactive_gradient =
          build_identity_metric_inactive_projector_pullback_gradient(
              inactive_density_gradient_symmetric,
              inactive_orbitals,
              basis_overlap_times_inactive);
      delta_original_inactive_gradient =
          build_identity_metric_inactive_projector_pullback_gradient_direction(
              delta_inactive_density_gradient_symmetric,
              inactive_density_gradient_symmetric,
              inactive_orbitals,
              delta_inactive_orbitals,
              basis_overlap_times_inactive,
              delta_basis_overlap_times_inactive);
    } else {
      const Eigen::MatrixXd inactive_overlap_gradient =
          build_inactive_overlap_gradient(
              inactive_overlap_inverse,
              inactive_orbitals,
              total_inactive_gradient);
      const Eigen::MatrixXd delta_inactive_overlap_gradient =
          build_inactive_overlap_gradient_direction(
              inactive_overlap_inverse,
              inactive_auxiliary_direction.delta_inactive_overlap_inverse,
              inactive_orbitals,
              delta_inactive_orbitals,
              total_inactive_gradient,
              delta_total_inactive_gradient);
      original_inactive_gradient =
          build_inactive_projector_pullback_gradient(
              inactive_density_gradient_symmetric,
              inactive_orbitals,
              inactive_auxiliary,
              basis_overlap_times_inactive,
              inactive_overlap_gradient);
      delta_original_inactive_gradient =
          build_inactive_projector_pullback_gradient_direction(
              delta_inactive_density_gradient_symmetric,
              inactive_density_gradient_symmetric,
              inactive_orbitals,
              delta_inactive_orbitals,
              inactive_auxiliary,
              inactive_auxiliary_direction.delta_inactive_auxiliary,
              basis_overlap_times_inactive,
              delta_basis_overlap_times_inactive,
              inactive_overlap_gradient,
              delta_inactive_overlap_gradient);
    }

    original_orbital_gradient.leftCols(
        n_inactive_doubly_occupied_orbitals) = original_inactive_gradient;
    original_orbital_gradient.middleCols(
        n_inactive_doubly_occupied_orbitals,
        input.n_active_orbitals) = original_active_gradient;
    delta_original_orbital_gradient.leftCols(
        n_inactive_doubly_occupied_orbitals) = delta_original_inactive_gradient;
    delta_original_orbital_gradient.middleCols(
        n_inactive_doubly_occupied_orbitals,
        input.n_active_orbitals) = delta_original_active_gradient;
  }

  if (input.orbital_type == kLegacyOrbitalTypeOeo) {
    return scatter_dense_orbital_gradient_to_sparse_slots_local(
        delta_original_orbital_gradient,
        input);
  }

  std::vector<double> orbital_value_gradient_direction(
      input.orbital_value_table.size(),
      0.0);
  for (int orbital_index = 0; orbital_index < input.n_orbitals; ++orbital_index) {
    const int coefficient_count =
        stored_sparse_orbital_coefficient_count(input, orbital_index);
    std::vector<int> basis_function_indices(coefficient_count, 0);
    Eigen::VectorXd normalized_vector =
        Eigen::VectorXd::Zero(coefficient_count);
    Eigen::VectorXd delta_normalized_vector =
        Eigen::VectorXd::Zero(coefficient_count);
    Eigen::VectorXd input_direction =
        Eigen::VectorXd::Zero(coefficient_count);
    Eigen::VectorXd dense_gradient =
        Eigen::VectorXd::Zero(coefficient_count);
    Eigen::VectorXd delta_dense_gradient =
        Eigen::VectorXd::Zero(coefficient_count);

    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      const int basis_function_index =
          input.orbital_basis_index_table[orbital_index *
                                              input.n_basis_functions +
                                          coefficient_index] -
          1;
      basis_function_indices[coefficient_index] =
          basis_function_index;
      normalized_vector(coefficient_index) =
          orbital_tangent_context.normalized_orbitals(
              basis_function_index,
              orbital_index);
      delta_normalized_vector(coefficient_index) =
          orbital_tangent_context.delta_normalized_orbitals(
              basis_function_index,
              orbital_index);
      input_direction(coefficient_index) =
          input_retract_tangent[
              orbital_index * input.n_basis_functions + coefficient_index];
      dense_gradient(coefficient_index) =
          original_orbital_gradient(
              basis_function_index,
              orbital_index);
      delta_dense_gradient(coefficient_index) =
          delta_original_orbital_gradient(
              basis_function_index,
              orbital_index);
    }

    Eigen::MatrixXd overlap_submatrix =
        Eigen::MatrixXd::Zero(coefficient_count, coefficient_count);
    for (int row = 0; row < coefficient_count; ++row) {
      for (int column = 0; column < coefficient_count; ++column) {
        overlap_submatrix(row, column) = basis_overlap(
            basis_function_indices[row],
            basis_function_indices[column]);
      }
    }

    const double inverse_norm =
        orbital_tangent_context.inverse_norms[orbital_index];
    const Eigen::VectorXd overlap_times_normalized =
        overlap_submatrix * normalized_vector;
    const Eigen::VectorXd delta_overlap_times_normalized =
        overlap_submatrix * delta_normalized_vector;
    const double input_direction_projection =
        inverse_norm *
        input_direction.dot(overlap_times_normalized);
    const double delta_inverse_norm =
        -inverse_norm * input_direction_projection;
    const double scalar_term =
        dense_gradient.dot(normalized_vector);
    const double delta_scalar_term =
        delta_dense_gradient.dot(normalized_vector) +
        dense_gradient.dot(delta_normalized_vector);
    const Eigen::VectorXd directional_raw_gradient =
        delta_inverse_norm *
            (dense_gradient - scalar_term * overlap_times_normalized) +
        inverse_norm *
            (delta_dense_gradient -
             delta_scalar_term * overlap_times_normalized -
             scalar_term * delta_overlap_times_normalized);

    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      orbital_value_gradient_direction[orbital_index *
                                           input.n_basis_functions +
                                       coefficient_index] =
          directional_raw_gradient(coefficient_index);
    }
  }

  return orbital_value_gradient_direction;
}

std::vector<double> apply_fixed_upstream_orbital_pullback_direction_cached(
    const OrbitalPreparationInput& input,
    const DenseOrbitalTangentContext& orbital_tangent_context,
    const Eigen::Ref<const Eigen::MatrixXd>& total_active_auxiliary_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap_times_delta_active_orbitals,
    const std::vector<double>& total_inactive_density_gradient,
    const Eigen::Ref<const Eigen::VectorXd>& input_retract_tangent,
    const AcceptedOrbitalPreparationCache& cache) {
  if (input.n_basis_functions <= 0 || input.n_orbitals <= 0 ||
      input.n_active_orbitals <= 0) {
    throw std::invalid_argument(
        "orbital pullback directional input dimensions must be positive");
  }
  if (input_retract_tangent.size() !=
      static_cast<Eigen::Index>(input.orbital_value_table.size())) {
    throw std::invalid_argument(
        "fixed-upstream input tangent size does not match orbital_value_table");
  }
  if (!cache.has_pullback_cache) {
    const ExactCtxInternalInactiveChart internal_chart = {
        cache.internal_inactive_orbitals,
        cache.selector_inactive_right_inverse_transform,
        cache.selector_inactive_inverse_transpose_right_transform,
        cache.selector_active_inactive_coefficients,
        cache.uses_internal_inactive_chart};
    return apply_fixed_upstream_orbital_pullback_direction(
        input,
        orbital_tangent_context,
        total_active_auxiliary_gradient,
        basis_overlap_times_delta_active_orbitals,
        total_inactive_density_gradient,
        input_retract_tangent,
        &internal_chart);
  }

  const int n_inactive_doubly_occupied_orbitals =
      (input.n_total_electrons - input.n_active_electrons) / 2;
  const std::size_t ao_matrix_size =
      input.n_basis_functions * input.n_basis_functions;
  const Eigen::Map<const Eigen::MatrixXd> basis_overlap(
      input.active_orbital_overlap_matrix.data(),
      input.n_basis_functions,
      input.n_basis_functions);

  Eigen::MatrixXd delta_original_orbital_gradient =
      Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_orbitals);

  if (n_inactive_doubly_occupied_orbitals == 0) {
    // original_orbital_gradient is just active_auxiliary_gradient,
    // delta is zero since the gradient doesn't depend on the direction.
    // Nothing to do.
  } else if (cache.uses_internal_inactive_chart) {
    const ExactCtxInternalInactiveChart internal_chart = {
        cache.internal_inactive_orbitals,
        cache.selector_inactive_right_inverse_transform,
        cache.selector_inactive_inverse_transpose_right_transform,
        cache.selector_active_inactive_coefficients,
        true};
    // Cached fixed-upstream on `(Q_i, T_a)` reuses the accepted symmetric
    // inactive-density adjoint and only applies the selector Jacobian back to
    // the physical occupied chart.
    const Eigen::MatrixXd delta_internal_inactive_gradient =
        build_internal_inactive_density_pullback_gradient_direction(
            cache.inactive_density_gradient_symmetric,
            orbital_tangent_context.delta_internal_inactive_orbitals);
    const PhysicalOrbitalGradientBlocks delta_physical_gradient =
        transport_internal_chart_gradient_to_physical(
            delta_internal_inactive_gradient,
            Eigen::MatrixXd::Zero(
                input.n_basis_functions,
                input.n_active_orbitals),
            internal_chart);
    delta_original_orbital_gradient.leftCols(
        n_inactive_doubly_occupied_orbitals) =
        delta_physical_gradient.inactive_gradient;
  } else {
    const auto inactive_orbitals =
        orbital_tangent_context.normalized_orbitals.leftCols(
            n_inactive_doubly_occupied_orbitals);
    const auto delta_inactive_orbitals =
        orbital_tangent_context.delta_normalized_orbitals.leftCols(
            n_inactive_doubly_occupied_orbitals);

    const InactiveAuxiliaryDirectionResult inactive_auxiliary_direction =
        build_delta_inactive_auxiliary_direction(
            basis_overlap,
            inactive_orbitals,
            delta_inactive_orbitals,
            cache.inactive_overlap_inverse,
            false);
    const LowRankAoMatrix delta_inactive_density_low_rank =
        build_inactive_density_direction_low_rank(
            input.n_basis_functions,
            inactive_orbitals,
            delta_inactive_orbitals,
            cache.inactive_auxiliary,
            inactive_auxiliary_direction.delta_inactive_auxiliary);

    const Eigen::MatrixXd delta_original_active_gradient =
        -apply_metric_times_low_rank_ao_matrix(
            basis_overlap,
            delta_inactive_density_low_rank,
            total_active_auxiliary_gradient,
            "cached fixed-upstream inactive-density direction");
    const Eigen::MatrixXd& delta_bs_active =
        basis_overlap_times_delta_active_orbitals;
    const Eigen::MatrixXd delta_total_inactive_gradient =
        -total_active_auxiliary_gradient * delta_bs_active.transpose();
    const Eigen::MatrixXd delta_inactive_density_gradient_symmetric =
        delta_total_inactive_gradient + delta_total_inactive_gradient.transpose();
    const Eigen::MatrixXd bs_active =
        cache.basis_overlap_times_normalized.middleCols(
            n_inactive_doubly_occupied_orbitals,
            input.n_active_orbitals);
    const Eigen::MatrixXd basis_overlap_times_inactive =
        cache.basis_overlap_times_normalized.leftCols(
            n_inactive_doubly_occupied_orbitals);
    const Eigen::MatrixXd delta_basis_overlap_times_inactive =
        basis_overlap * delta_inactive_orbitals;
    const Eigen::MatrixXd total_inactive_gradient =
        Eigen::Map<const Eigen::MatrixXd>(
            total_inactive_density_gradient.data(),
            input.n_basis_functions,
            input.n_basis_functions) -
        total_active_auxiliary_gradient * bs_active.transpose();
    Eigen::MatrixXd delta_original_inactive_gradient =
        Eigen::MatrixXd::Zero(
            input.n_basis_functions,
            n_inactive_doubly_occupied_orbitals);
    if (cache.has_orthonormal_inactive_chart) {
      delta_original_inactive_gradient =
          build_identity_metric_inactive_projector_pullback_gradient_direction(
              delta_inactive_density_gradient_symmetric,
              cache.inactive_density_gradient_symmetric,
              inactive_orbitals,
              delta_inactive_orbitals,
              basis_overlap_times_inactive,
              delta_basis_overlap_times_inactive);
    } else {
      const Eigen::MatrixXd delta_inactive_overlap_gradient =
          build_inactive_overlap_gradient_direction(
              cache.inactive_overlap_inverse,
              inactive_auxiliary_direction.delta_inactive_overlap_inverse,
              inactive_orbitals,
              delta_inactive_orbitals,
              total_inactive_gradient,
              delta_total_inactive_gradient);
      const Eigen::MatrixXd inactive_overlap_gradient =
          build_inactive_overlap_gradient(
              cache.inactive_overlap_inverse,
              inactive_orbitals,
              total_inactive_gradient);
      delta_original_inactive_gradient =
          build_inactive_projector_pullback_gradient_direction(
              delta_inactive_density_gradient_symmetric,
              cache.inactive_density_gradient_symmetric,
              inactive_orbitals,
              delta_inactive_orbitals,
              cache.inactive_auxiliary,
              inactive_auxiliary_direction.delta_inactive_auxiliary,
              basis_overlap_times_inactive,
              delta_basis_overlap_times_inactive,
              inactive_overlap_gradient,
              delta_inactive_overlap_gradient);
    }

    delta_original_orbital_gradient.leftCols(
        n_inactive_doubly_occupied_orbitals) = delta_original_inactive_gradient;
    delta_original_orbital_gradient.middleCols(
        n_inactive_doubly_occupied_orbitals,
        input.n_active_orbitals) = delta_original_active_gradient;
  }

  if (input.orbital_type == kLegacyOrbitalTypeOeo) {
    return scatter_dense_orbital_gradient_to_sparse_slots_local(
        delta_original_orbital_gradient,
        input);
  }

  std::vector<double> orbital_value_gradient_direction(
      input.orbital_value_table.size(),
      0.0);
  for (int orbital_index = 0; orbital_index < input.n_orbitals; ++orbital_index) {
    const auto& basis_indices =
        cache.orbital_basis_function_indices[orbital_index];
    const int coefficient_count =
        cache.orbital_coefficient_counts[orbital_index];
    Eigen::VectorXd normalized_vector =
        Eigen::VectorXd::Zero(coefficient_count);
    Eigen::VectorXd delta_normalized_vector =
        Eigen::VectorXd::Zero(coefficient_count);
    Eigen::VectorXd input_direction =
        Eigen::VectorXd::Zero(coefficient_count);
    Eigen::VectorXd dense_gradient =
        Eigen::VectorXd::Zero(coefficient_count);
    Eigen::VectorXd delta_dense_gradient =
        Eigen::VectorXd::Zero(coefficient_count);

    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      const int basis_function_index =
          basis_indices[coefficient_index];
      normalized_vector(coefficient_index) =
          orbital_tangent_context.normalized_orbitals(
              basis_function_index,
              orbital_index);
      delta_normalized_vector(coefficient_index) =
          orbital_tangent_context.delta_normalized_orbitals(
              basis_function_index,
              orbital_index);
      input_direction(coefficient_index) =
          input_retract_tangent[
              orbital_index * input.n_basis_functions + coefficient_index];
      dense_gradient(coefficient_index) =
          cache.original_orbital_gradient(
              basis_function_index,
              orbital_index);
      delta_dense_gradient(coefficient_index) =
          delta_original_orbital_gradient(
              basis_function_index,
              orbital_index);
    }

    const auto& overlap_submatrix =
        cache.orbital_overlap_submatrices[orbital_index];
    const double inverse_norm =
        orbital_tangent_context.inverse_norms[orbital_index];
    const Eigen::VectorXd overlap_times_normalized =
        overlap_submatrix * normalized_vector;
    const Eigen::VectorXd delta_overlap_times_normalized =
        overlap_submatrix * delta_normalized_vector;
    const double input_direction_projection =
        inverse_norm *
        input_direction.dot(overlap_times_normalized);
    const double delta_inverse_norm =
        -inverse_norm * input_direction_projection;
    const double scalar_term =
        dense_gradient.dot(normalized_vector);
    const double delta_scalar_term =
        delta_dense_gradient.dot(normalized_vector) +
        dense_gradient.dot(delta_normalized_vector);
    const Eigen::VectorXd directional_raw_gradient =
        delta_inverse_norm *
            (dense_gradient - scalar_term * overlap_times_normalized) +
        inverse_norm *
            (delta_dense_gradient -
             delta_scalar_term * overlap_times_normalized -
             scalar_term * delta_overlap_times_normalized);

    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      orbital_value_gradient_direction[orbital_index *
                                           input.n_basis_functions +
                                       coefficient_index] =
          directional_raw_gradient(coefficient_index);
    }
  }

  return orbital_value_gradient_direction;
}

void build_active_space_directional_integrals(
    const CppVbInput& input,
    const ActiveSpaceTwoElectronResult& accepted_active_space_two_electron_result,
    const Eigen::MatrixXd& accepted_active_auxiliary_orbitals,
    const Eigen::MatrixXd& accepted_basis_overlap_times_active_auxiliary_orbitals,
    const Eigen::MatrixXd& accepted_ao_effective_one_electron_times_active_auxiliary_orbitals,
    const Eigen::MatrixXd&
        accepted_ao_effective_one_electron_transpose_times_active_auxiliary_orbitals,
    const Eigen::MatrixXd& delta_active_auxiliary_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& accepted_dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_dense_active_coefficients,
    const Eigen::MatrixXd& delta_ao_effective_h1e_times_active_auxiliary_orbitals,
    std::vector<double>* delta_active_orbital_overlap_matrix_storage,
    std::vector<double>* delta_active_one_electron_matrix_storage,
    ExactPackedActiveTwoElectronDirectionalDerivativeWorkspace* exact_2e_workspace,
    std::vector<double>* delta_packed_active_two_electron_integrals) {
  const int n_basis_functions =
      input.orbital_preparation_input.n_basis_functions;
  const int n_active_orbitals =
      input.orbital_preparation_input.n_active_orbitals;
  const std::size_t active_matrix_size =
      n_active_orbitals * n_active_orbitals;
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument(
        "active-space directional integrals require positive dimensions");
  }
  resize_for_overwrite(
      delta_active_orbital_overlap_matrix_storage,
      active_matrix_size);
  resize_for_overwrite(
      delta_active_one_electron_matrix_storage,
      active_matrix_size);
  Eigen::Map<Eigen::MatrixXd> delta_active_orbital_overlap_matrix(
      delta_active_orbital_overlap_matrix_storage->data(),
      n_active_orbitals,
      n_active_orbitals);
  Eigen::Map<Eigen::MatrixXd> delta_active_one_electron_matrix(
      delta_active_one_electron_matrix_storage->data(),
      n_active_orbitals,
      n_active_orbitals);

  // The accepted-point active auxiliary block A is fixed during one Newton
  // linear solve.  Reusing cached `S * A`, `F * A`, and `F^T * A` contractions
  // leaves only active-sized temporaries here while writing the directional
  // `delta SSO` and `delta HHO` directly into caller-owned column-major buffers.
  const Eigen::MatrixXd delta_active_overlap_left =
      delta_active_auxiliary_orbitals.transpose() *
      accepted_basis_overlap_times_active_auxiliary_orbitals;
  delta_active_orbital_overlap_matrix.noalias() =
      delta_active_overlap_left;
  delta_active_orbital_overlap_matrix.noalias() +=
      delta_active_overlap_left.transpose();

  const Eigen::MatrixXd delta_active_one_electron_left =
      delta_active_auxiliary_orbitals.transpose() *
      accepted_ao_effective_one_electron_times_active_auxiliary_orbitals;
  delta_active_one_electron_matrix.noalias() =
      delta_active_one_electron_left;
  delta_active_one_electron_matrix.noalias() +=
      accepted_active_auxiliary_orbitals.transpose() *
      delta_ao_effective_h1e_times_active_auxiliary_orbitals;
  delta_active_one_electron_matrix.noalias() +=
      accepted_ao_effective_one_electron_transpose_times_active_auxiliary_orbitals
          .transpose() *
      delta_active_auxiliary_orbitals;
  if (accepted_dense_active_coefficients.rows() != n_basis_functions ||
      accepted_dense_active_coefficients.cols() != n_active_orbitals ||
      delta_dense_active_coefficients.rows() != n_basis_functions ||
      delta_dense_active_coefficients.cols() != n_active_orbitals) {
    throw std::invalid_argument(
        "dense active coefficient shapes are inconsistent in active-space directional integrals");
  }

  compute_exact_packed_active_two_electron_integral_directional_derivative(
      accepted_dense_active_coefficients,
      delta_dense_active_coefficients,
      input.ao_integral_input,
      n_active_orbitals,
      exact_2e_workspace,
      delta_packed_active_two_electron_integrals,
      &accepted_active_space_two_electron_result);
}

struct StructurePairAdjoints {
  double hamiltonian_weight = 0.0;
  double overlap_weight = 0.0;
};

struct StructurePairWeightTables {
  std::vector<double> hamiltonian_upper_weights;
  std::vector<double> overlap_upper_weights;
};

struct ActiveSpaceGradientDirection {
  std::vector<double> active_orbital_overlap_gradient;
  std::vector<double> active_one_electron_gradient;
  std::vector<double> packed_active_two_electron_gradient;
};

void validate_outer_response_structure_matrices(
    const StructureAccumulationResult& directional_structure_matrices) {
  throw_if_nonfinite(
      directional_structure_matrices.overlap_matrix,
      "exact outer-response directional overlap matrix");
  throw_if_nonfinite(
      directional_structure_matrices.hamiltonian_matrix,
      "exact outer-response directional Hamiltonian matrix");
}

void validate_outer_response_pair_weights(
    const StructurePairWeightTables& structure_pair_weights) {
  throw_if_nonfinite(
      structure_pair_weights.hamiltonian_upper_weights,
      "exact outer-response directional Hamiltonian pair weights");
  throw_if_nonfinite(
      structure_pair_weights.overlap_upper_weights,
      "exact outer-response directional overlap pair weights");
}

void validate_outer_response_active_gradient(
    const ActiveSpaceGradientDirection& active_space_gradient) {
  throw_if_nonfinite(
      active_space_gradient.active_orbital_overlap_gradient,
      "exact outer-response active overlap gradient");
  throw_if_nonfinite(
      active_space_gradient.active_one_electron_gradient,
      "exact outer-response active one-electron gradient");
  throw_if_nonfinite(
      active_space_gradient.packed_active_two_electron_gradient,
      "exact outer-response active two-electron gradient");
}

void validate_selected_state_determinant_matrices(
    const SelectedStateDeterminantMatrices& selected_state_matrices,
    const char* label) {
  for (const auto& state : selected_state_matrices.states) {
    throw_if_nonfinite(
        state.determinant_coefficients,
        label);
    throw_if_nonfinite(
        state.coefficient_matrix,
        label);
    throw_if_nonfinite(
        state.local_coefficient_matrix,
        label);
  }
}

void validate_directional_determinant_pair_weights(
    const DeterminantPairWeightTablesFromCoefficients& pair_weights,
    const char* label) {
  throw_if_nonfinite(
      pair_weights.ordered_hamiltonian_weights,
      label);
  throw_if_nonfinite(
      pair_weights.ordered_overlap_weights,
      label);
  throw_if_nonfinite(
      pair_weights.unordered_combined_hamiltonian_weights,
      label);
  throw_if_nonfinite(
      pair_weights.unordered_combined_overlap_weights,
      label);
}

void validate_same_spin_matrix_backward_contribution(
    const SameSpinMatrixBackwardContribution& contribution,
    const char* label) {
  throw_if_nonfinite(
      contribution.active_orbital_overlap_gradient,
      label);
  throw_if_nonfinite(
      contribution.active_one_electron_gradient,
      label);
  throw_if_nonfinite(
      contribution.packed_active_two_electron_gradient,
      label);
}

void validate_opposite_spin_matrix_backward_contribution(
    const OppositeSpinMatrixBackwardContribution& contribution,
    const char* label) {
  throw_if_nonfinite(
      contribution.active_orbital_overlap_gradient,
      label);
  throw_if_nonfinite(
      contribution.packed_active_two_electron_gradient,
      label);
}

struct DeterminantPairDirectionalScratch {
  Eigen::MatrixXd active_one_electron_gradient;
  std::vector<double> active_orbital_overlap_gradient;
  std::vector<double> packed_active_two_electron_gradient;

  void resize(int n_active_orbitals) {
    if (n_active_orbitals <= 0) {
      throw std::invalid_argument(
          "n_active_orbitals must be positive in pair directional scratch");
    }
    active_one_electron_gradient =
        Eigen::MatrixXd::Zero(n_active_orbitals, n_active_orbitals);
    active_orbital_overlap_gradient.assign(
        n_active_orbitals * n_active_orbitals,
        0.0);
    packed_active_two_electron_gradient.assign(
        packed_active_two_electron_integral_count(n_active_orbitals),
        0.0);
  }

  void set_zero() {
    active_one_electron_gradient.setZero();
    std::fill(
        active_orbital_overlap_gradient.begin(),
        active_orbital_overlap_gradient.end(),
        0.0);
    std::fill(
        packed_active_two_electron_gradient.begin(),
        packed_active_two_electron_gradient.end(),
        0.0);
  }
};

Eigen::MatrixXd build_spin_one_electron_block_matrix_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_active_orbitals) {
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

SameSpinPhiResult evaluate_same_spin_phi_with_optional_cache_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const SpinDeterminantPairEvaluation& pair_evaluation,
    Eigen::MatrixXd* inverse_overlap_gradient);

struct RegularSpinDirectionalData {
  double overlap_determinant = 0.0;
  double delta_overlap_determinant = 0.0;
  Eigen::MatrixXd inverse_overlap_submatrix;
  Eigen::MatrixXd delta_inverse_overlap_submatrix;
  Eigen::MatrixXd cofactor_1st;
  Eigen::MatrixXd delta_cofactor_1st;
  double same_spin_total_phi = 0.0;
  double delta_same_spin_total_phi = 0.0;
  Eigen::MatrixXd same_spin_inverse_overlap_gradient;
  Eigen::MatrixXd delta_same_spin_inverse_overlap_gradient;
};

struct SingularSpinDirectionalData {
  double delta_overlap_determinant = 0.0;
  double delta_total_hamiltonian = 0.0;
  Eigen::MatrixXd cofactor_1st;
  Eigen::MatrixXd delta_cofactor_1st;
};

struct OppositeSpinDirectionalData {
  double phi = 0.0;
  double delta_phi = 0.0;
  Eigen::MatrixXd alpha_inverse_overlap_gradient;
  Eigen::MatrixXd delta_alpha_inverse_overlap_gradient;
  Eigen::MatrixXd beta_inverse_overlap_gradient;
  Eigen::MatrixXd delta_beta_inverse_overlap_gradient;
};

Eigen::MatrixXd build_local_overlap_direction_matrix(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    int n_active_orbitals) {
  return build_overlap_submatrix(
      occ_L,
      occ_R,
      delta_active_orbital_overlap_matrix,
      n_active_orbitals);
}

double lookup_directional_active_two_electron_kernel_value(
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    int row_packed_pair_index,
    int column_packed_pair_index) {
  if (delta_packed_active_two_electron_integrals.empty()) {
    return 0.0;
  }
  const int packed_pair_of_pairs_index =
      TwoElectronIndexer::packed_pair_of_pairs_index(
          row_packed_pair_index,
          column_packed_pair_index);
  return delta_packed_active_two_electron_integrals[
      packed_pair_of_pairs_index];
}

SingularSpinDirectionalData build_singular_spin_directional_data(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const ActiveSpaceOneElectronResult& active_space_one_electron_result,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const SpinDeterminantPairEvaluation& pair_evaluation,
    int n_active_orbitals,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  const auto& overlap_result = pair_evaluation.overlap_result;
  if (overlap_result.nullity == 0 &&
      overlap_result.overlap_determinant != 0.0) {
    throw std::runtime_error(
        "singular spin directional data requires a singular determinant pair");
  }

  SingularSpinDirectionalData result;
  const Eigen::MatrixXd overlap_block =
      build_overlap_submatrix_from_result(overlap_result);
  const Eigen::MatrixXd delta_overlap_submatrix =
      build_local_overlap_direction_matrix(
          occ_L,
          occ_R,
          delta_active_orbital_overlap_matrix,
          n_active_orbitals);
  const DeterminantOverlapResolver overlap_resolver;

  result.cofactor_1st = calc_cofactor_1st(overlap_result);
  result.delta_overlap_determinant =
      (result.cofactor_1st.cwiseProduct(delta_overlap_submatrix)).sum();
  result.delta_cofactor_1st =
      build_directional_first_cofactor_matrix(
          overlap_block,
          delta_overlap_submatrix,
          overlap_result,
          overlap_resolver);

  const Eigen::MatrixXd one_electron_block =
      build_spin_one_electron_block_matrix_local(
          occ_L,
          occ_R,
          active_space_one_electron_result.h1e_act,
          n_active_orbitals);
  const Eigen::MatrixXd delta_one_electron_block =
      build_spin_one_electron_block_matrix_local(
          occ_L,
          occ_R,
          Eigen::Map<const Eigen::MatrixXd>(
              delta_active_one_electron_matrix.data(),
              n_active_orbitals,
              n_active_orbitals),
          n_active_orbitals);
  result.delta_total_hamiltonian =
      (delta_one_electron_block.cwiseProduct(result.cofactor_1st)).sum() +
      (one_electron_block.cwiseProduct(result.delta_cofactor_1st)).sum();

  const ActiveSpaceTwoElectronView two_electron_view =
      make_active_space_two_electron_view(active_space_two_electron_result);
  const int n_electrons = static_cast<int>(occ_L.size());
  for (int left_first = 0; left_first < n_electrons - 1; ++left_first) {
    const int orbital_index_left_first = occ_L[left_first];
    for (int right_first = 0; right_first < n_electrons - 1; ++right_first) {
      const int orbital_index_right_first = occ_R[right_first];
      const int direct_left_pair_index = TwoElectronIndexer::packed_pair_index(
          orbital_index_right_first,
          orbital_index_left_first);
      for (int left_second = left_first + 1;
           left_second < n_electrons;
           ++left_second) {
        const int orbital_index_left_second = occ_L[left_second];
        const int exchange_left_pair_index = TwoElectronIndexer::packed_pair_index(
            orbital_index_right_first,
            orbital_index_left_second);
        for (int right_second = right_first + 1;
             right_second < n_electrons;
             ++right_second) {
          const int orbital_index_right_second =
              occ_R[right_second];
          const int direct_right_pair_index = TwoElectronIndexer::packed_pair_index(
              orbital_index_right_second,
              orbital_index_left_second);
          const int exchange_right_pair_index = TwoElectronIndexer::packed_pair_index(
              orbital_index_right_second,
              orbital_index_left_first);
          const double interaction_value =
              lookup_active_space_two_electron_kernel_value(
                  two_electron_view,
                  direct_left_pair_index,
                  direct_right_pair_index,
                  n_active_orbitals) -
              lookup_active_space_two_electron_kernel_value(
                  two_electron_view,
                  exchange_left_pair_index,
                  exchange_right_pair_index,
                  n_active_orbitals);
          const double delta_interaction_value =
              lookup_directional_active_two_electron_kernel_value(
                  delta_packed_active_two_electron_integrals,
                  direct_left_pair_index,
                  direct_right_pair_index) -
              lookup_directional_active_two_electron_kernel_value(
                  delta_packed_active_two_electron_integrals,
                  exchange_left_pair_index,
                  exchange_right_pair_index);
          const double second_order_cofactor =
              calc_second_order_cofactor(
                  overlap_result,
                  right_first,
                  right_second,
                  left_first,
                  left_second);
          const double directional_second_order_cofactor =
              calc_directional_second_order_cofactor(
                  overlap_block,
                  delta_overlap_submatrix,
                  overlap_result,
                  right_first,
                  right_second,
                  left_first,
                  left_second,
                  overlap_resolver);
          result.delta_total_hamiltonian +=
              delta_interaction_value * second_order_cofactor +
              interaction_value * directional_second_order_cofactor;
        }
      }
    }
  }

  return result;
}

RegularSpinDirectionalData build_regular_spin_directional_data(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const ActiveSpaceOneElectronResult& active_space_one_electron_result,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const SpinDeterminantPairEvaluation& pair_evaluation,
    int n_active_orbitals,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  // Build the local regular-pair directional payloads
  //   δdet, δX^{-1}, δC, δφ, δ(∂φ/∂X^{-1})
  // once per ordered same-spin determinant pair. The outer-response local term
  // then reuses this data across the one-electron, overlap, and 2e pullbacks.
  const auto& overlap_result = pair_evaluation.overlap_result;
  if (overlap_result.nullity != 0 || overlap_result.overlap_determinant == 0.0) {
    throw std::runtime_error(
        "regular spin directional data requires a non-singular determinant pair");
  }

  RegularSpinDirectionalData result;
  result.overlap_determinant = overlap_result.overlap_determinant;
  result.inverse_overlap_submatrix =
      build_inverse_overlap_submatrix_from_result(overlap_result);
  const Eigen::MatrixXd delta_overlap_submatrix =
      build_local_overlap_direction_matrix(
          occ_L,
          occ_R,
          delta_active_orbital_overlap_matrix,
          n_active_orbitals);
  result.delta_overlap_determinant =
      result.overlap_determinant *
      (result.inverse_overlap_submatrix * delta_overlap_submatrix).trace();
  result.delta_inverse_overlap_submatrix =
      -result.inverse_overlap_submatrix *
      delta_overlap_submatrix *
      result.inverse_overlap_submatrix;

  result.cofactor_1st = calc_cofactor_1st(overlap_result);
  result.delta_cofactor_1st =
      result.delta_overlap_determinant *
      result.inverse_overlap_submatrix.transpose();
  result.delta_cofactor_1st.noalias() +=
      result.overlap_determinant *
      result.delta_inverse_overlap_submatrix.transpose();

  const Eigen::MatrixXd one_electron_block =
      build_spin_one_electron_block_matrix_local(
          occ_L,
          occ_R,
          active_space_one_electron_result.h1e_act,
          n_active_orbitals);
  const Eigen::MatrixXd delta_one_electron_block =
      build_spin_one_electron_block_matrix_local(
          occ_L,
          occ_R,
          Eigen::Map<const Eigen::MatrixXd>(
              delta_active_one_electron_matrix.data(),
              n_active_orbitals,
              n_active_orbitals),
          n_active_orbitals);
  const SameSpinPhiResult same_spin_phi_result =
      evaluate_same_spin_phi_with_optional_cache_local(
          occ_L,
          occ_R,
          active_space_one_electron_result.h1e_act,
          n_active_orbitals,
          active_space_two_electron_result,
          pair_evaluation,
          &result.same_spin_inverse_overlap_gradient);
  result.same_spin_total_phi = same_spin_phi_result.total_phi;
  result.delta_same_spin_inverse_overlap_gradient =
      delta_one_electron_block.transpose();
  result.delta_same_spin_total_phi =
      (delta_one_electron_block.cwiseProduct(
           result.inverse_overlap_submatrix.transpose()))
          .sum();
  result.delta_same_spin_total_phi +=
      (one_electron_block.cwiseProduct(
           result.delta_inverse_overlap_submatrix.transpose()))
          .sum();

  const ActiveSpaceTwoElectronView two_electron_view =
      make_active_space_two_electron_view(active_space_two_electron_result);
  const int n_electrons = static_cast<int>(occ_L.size());
  for (int left_first = 0; left_first < n_electrons - 1; ++left_first) {
    const int orbital_index_left_first = occ_L[left_first];
    for (int right_first = 0; right_first < n_electrons - 1; ++right_first) {
      const int orbital_index_right_first = occ_R[right_first];
      const int direct_left_pair_index = TwoElectronIndexer::packed_pair_index(
          orbital_index_right_first,
          orbital_index_left_first);
      for (int left_second = left_first + 1;
           left_second < n_electrons;
           ++left_second) {
        const int orbital_index_left_second = occ_L[left_second];
        const int exchange_left_pair_index = TwoElectronIndexer::packed_pair_index(
            orbital_index_right_first,
            orbital_index_left_second);
        for (int right_second = right_first + 1;
             right_second < n_electrons;
             ++right_second) {
          const int orbital_index_right_second =
              occ_R[right_second];
          const int direct_right_pair_index = TwoElectronIndexer::packed_pair_index(
              orbital_index_right_second,
              orbital_index_left_second);
          const int exchange_right_pair_index = TwoElectronIndexer::packed_pair_index(
              orbital_index_right_second,
              orbital_index_left_first);
          const double interaction_value =
              lookup_active_space_two_electron_kernel_value(
                  two_electron_view,
                  direct_left_pair_index,
                  direct_right_pair_index,
                  n_active_orbitals) -
              lookup_active_space_two_electron_kernel_value(
                  two_electron_view,
                  exchange_left_pair_index,
                  exchange_right_pair_index,
                  n_active_orbitals);
          const double delta_interaction_value =
              lookup_directional_active_two_electron_kernel_value(
                  delta_packed_active_two_electron_integrals,
                  direct_left_pair_index,
                  direct_right_pair_index) -
              lookup_directional_active_two_electron_kernel_value(
                  delta_packed_active_two_electron_integrals,
                  exchange_left_pair_index,
                  exchange_right_pair_index);

          const double x11 =
              result.inverse_overlap_submatrix(left_first, right_first);
          const double x22 =
              result.inverse_overlap_submatrix(left_second, right_second);
          const double x12 =
              result.inverse_overlap_submatrix(left_second, right_first);
          const double x21 =
              result.inverse_overlap_submatrix(left_first, right_second);
          const double dx11 =
              result.delta_inverse_overlap_submatrix(left_first, right_first);
          const double dx22 =
              result.delta_inverse_overlap_submatrix(left_second, right_second);
          const double dx12 =
              result.delta_inverse_overlap_submatrix(left_second, right_first);
          const double dx21 =
              result.delta_inverse_overlap_submatrix(left_first, right_second);
          const double wedge = x11 * x22 - x12 * x21;
          const double delta_wedge =
              dx11 * x22 + x11 * dx22 - dx12 * x21 - x12 * dx21;
          result.delta_same_spin_total_phi +=
              delta_interaction_value * wedge +
              interaction_value * delta_wedge;

          result.delta_same_spin_inverse_overlap_gradient(
              left_first,
              right_first) +=
              delta_interaction_value * x22 +
              interaction_value * dx22;
          result.delta_same_spin_inverse_overlap_gradient(
              left_second,
              right_second) +=
              delta_interaction_value * x11 +
              interaction_value * dx11;
          result.delta_same_spin_inverse_overlap_gradient(
              left_second,
              right_first) -=
              delta_interaction_value * x21 +
              interaction_value * dx21;
          result.delta_same_spin_inverse_overlap_gradient(
              left_first,
              right_second) -=
              delta_interaction_value * x12 +
              interaction_value * dx12;
        }
      }
    }
  }

  return result;
}

OppositeSpinDirectionalData build_opposite_spin_directional_data(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const RegularSpinDirectionalData& alpha_data,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const RegularSpinDirectionalData& beta_data,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    int n_active_orbitals,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  OppositeSpinDirectionalData result;
  result.alpha_inverse_overlap_gradient =
      Eigen::MatrixXd::Zero(
          static_cast<int>(alpha_occ_L.size()),
          static_cast<int>(alpha_occ_L.size()));
  result.delta_alpha_inverse_overlap_gradient =
      Eigen::MatrixXd::Zero(
          static_cast<int>(alpha_occ_L.size()),
          static_cast<int>(alpha_occ_L.size()));
  result.beta_inverse_overlap_gradient =
      Eigen::MatrixXd::Zero(
          static_cast<int>(beta_occ_L.size()),
          static_cast<int>(beta_occ_L.size()));
  result.delta_beta_inverse_overlap_gradient =
      Eigen::MatrixXd::Zero(
          static_cast<int>(beta_occ_L.size()),
          static_cast<int>(beta_occ_L.size()));

  if (alpha_occ_L.empty() || beta_occ_L.empty()) {
    return result;
  }

  const ActiveSpaceTwoElectronView two_electron_view =
      make_active_space_two_electron_view(active_space_two_electron_result);
  for (int alpha_left_column = 0;
       alpha_left_column < static_cast<int>(alpha_occ_L.size());
       ++alpha_left_column) {
    const int alpha_orbital_left =
        alpha_occ_L[alpha_left_column];
    for (int alpha_right_row = 0;
         alpha_right_row < static_cast<int>(alpha_occ_R.size());
         ++alpha_right_row) {
      const int alpha_orbital_right =
          alpha_occ_R[alpha_right_row];
      const int alpha_packed_pair_index = TwoElectronIndexer::packed_pair_index(
          alpha_orbital_right,
          alpha_orbital_left);
      const double alpha_inverse_value =
          alpha_data.inverse_overlap_submatrix(
              alpha_left_column,
              alpha_right_row);
      const double delta_alpha_inverse_value =
          alpha_data.delta_inverse_overlap_submatrix(
              alpha_left_column,
              alpha_right_row);
      for (int beta_left_column = 0;
           beta_left_column < static_cast<int>(beta_occ_L.size());
           ++beta_left_column) {
        const int beta_orbital_left =
            beta_occ_L[beta_left_column];
        for (int beta_right_row = 0;
             beta_right_row < static_cast<int>(beta_occ_R.size());
             ++beta_right_row) {
          const int beta_orbital_right =
              beta_occ_R[beta_right_row];
          const int beta_packed_pair_index = TwoElectronIndexer::packed_pair_index(
              beta_orbital_right,
              beta_orbital_left);
          const double beta_inverse_value =
              beta_data.inverse_overlap_submatrix(
                  beta_left_column,
                  beta_right_row);
          const double delta_beta_inverse_value =
              beta_data.delta_inverse_overlap_submatrix(
                  beta_left_column,
                  beta_right_row);
          const double interaction_value =
              lookup_active_space_two_electron_kernel_value(
                  two_electron_view,
                  beta_packed_pair_index,
                  alpha_packed_pair_index,
                  n_active_orbitals);
          const double delta_interaction_value =
              lookup_directional_active_two_electron_kernel_value(
                  delta_packed_active_two_electron_integrals,
                  beta_packed_pair_index,
                  alpha_packed_pair_index);
          result.phi +=
              interaction_value *
              alpha_inverse_value *
              beta_inverse_value;
          result.delta_phi +=
              delta_interaction_value *
              alpha_inverse_value *
              beta_inverse_value;
          result.delta_phi +=
              interaction_value *
              delta_alpha_inverse_value *
              beta_inverse_value;
          result.delta_phi +=
              interaction_value *
              alpha_inverse_value *
              delta_beta_inverse_value;

          result.alpha_inverse_overlap_gradient(
              alpha_left_column,
              alpha_right_row) +=
              interaction_value * beta_inverse_value;
          result.delta_alpha_inverse_overlap_gradient(
              alpha_left_column,
              alpha_right_row) +=
              delta_interaction_value * beta_inverse_value +
              interaction_value * delta_beta_inverse_value;
          result.beta_inverse_overlap_gradient(
              beta_left_column,
              beta_right_row) +=
              interaction_value * alpha_inverse_value;
          result.delta_beta_inverse_overlap_gradient(
              beta_left_column,
              beta_right_row) +=
              delta_interaction_value * alpha_inverse_value +
              interaction_value * delta_alpha_inverse_value;
        }
      }
    }
  }

  return result;
}

void accumulate_directional_one_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& cofactor_1st,
    const Eigen::MatrixXd& delta_cofactor_1st,
    double weight,
    double delta_weight,
    Eigen::MatrixXd* active_one_electron_gradient) {
  if (weight == 0.0 && delta_weight == 0.0) {
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

void accumulate_directional_same_spin_two_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& cofactor_1st,
    const Eigen::MatrixXd& delta_cofactor_1st,
    double overlap_determinant,
    double delta_overlap_determinant,
    double weight,
    double delta_weight,
    std::vector<double>* packed_active_two_electron_gradient) {
  if ((weight == 0.0 && delta_weight == 0.0) ||
      overlap_determinant == 0.0) {
    return;
  }

  const int n_electrons = static_cast<int>(occ_L.size());
  const double inv_overlap_determinant = 1.0 / overlap_determinant;
  const double delta_prefactor =
      delta_weight * inv_overlap_determinant -
      weight * delta_overlap_determinant *
          inv_overlap_determinant * inv_overlap_determinant;
  const double prefactor = weight * inv_overlap_determinant;
  for (int left_first = 0; left_first < n_electrons - 1; ++left_first) {
    const int orbital_index_left_first = occ_L[left_first];
    for (int right_first = 0; right_first < n_electrons - 1; ++right_first) {
      const int orbital_index_right_first = occ_R[right_first];
      const double cofactor_11 = cofactor_1st(right_first, left_first);
      const double delta_cofactor_11 =
          delta_cofactor_1st(right_first, left_first);
      for (int left_second = left_first + 1; left_second < n_electrons; ++left_second) {
        const int orbital_index_left_second = occ_L[left_second];
        const double cofactor_12 = cofactor_1st(right_first, left_second);
        const double delta_cofactor_12 =
            delta_cofactor_1st(right_first, left_second);
        for (int right_second = right_first + 1;
             right_second < n_electrons;
             ++right_second) {
          const int orbital_index_right_second =
              occ_R[right_second];
          const double cofactor_22 = cofactor_1st(right_second, left_second);
          const double cofactor_21 = cofactor_1st(right_second, left_first);
          const double delta_cofactor_22 =
              delta_cofactor_1st(right_second, left_second);
          const double delta_cofactor_21 =
              delta_cofactor_1st(right_second, left_first);
          const double wedge =
              cofactor_11 * cofactor_22 - cofactor_12 * cofactor_21;
          const double delta_wedge =
              delta_cofactor_11 * cofactor_22 +
              cofactor_11 * delta_cofactor_22 -
              delta_cofactor_12 * cofactor_21 -
              cofactor_12 * delta_cofactor_21;
          const double directional_second_order_cofactor =
              delta_prefactor * wedge +
              prefactor * delta_wedge;

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
              directional_second_order_cofactor;
          (*packed_active_two_electron_gradient)[exchange_index] -=
              directional_second_order_cofactor;
        }
      }
    }
  }
}

void accumulate_directional_opposite_spin_two_electron_gradient_contribution_local(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const Eigen::MatrixXd& alpha_cofactor_1st,
    const Eigen::MatrixXd& delta_alpha_cofactor_1st,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const Eigen::MatrixXd& beta_cofactor_1st,
    const Eigen::MatrixXd& delta_beta_cofactor_1st,
    double weight,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (weight == 0.0) {
    return;
  }

  for (int alpha_left_column = 0;
       alpha_left_column < static_cast<int>(alpha_occ_L.size());
       ++alpha_left_column) {
    const int alpha_orbital_left =
        alpha_occ_L[alpha_left_column];
    for (int alpha_right_row = 0;
         alpha_right_row < static_cast<int>(alpha_occ_R.size());
         ++alpha_right_row) {
      const int alpha_orbital_right =
          alpha_occ_R[alpha_right_row];
      const double alpha_cofactor =
          alpha_cofactor_1st(alpha_right_row, alpha_left_column);
      const double delta_alpha_cofactor =
          delta_alpha_cofactor_1st(alpha_right_row, alpha_left_column);
      for (int beta_left_column = 0;
           beta_left_column < static_cast<int>(beta_occ_L.size());
           ++beta_left_column) {
        const int beta_orbital_left =
            beta_occ_L[beta_left_column];
        for (int beta_right_row = 0;
             beta_right_row < static_cast<int>(beta_occ_R.size());
             ++beta_right_row) {
          const int beta_orbital_right =
              beta_occ_R[beta_right_row];
          const double beta_cofactor =
              beta_cofactor_1st(beta_right_row, beta_left_column);
          const double delta_beta_cofactor =
              delta_beta_cofactor_1st(beta_right_row, beta_left_column);
          const int two_electron_index = TwoElectronIndexer::two_electron_storage_index(
              beta_orbital_right,
              beta_orbital_left,
              alpha_orbital_right,
              alpha_orbital_left);
          (*packed_active_two_electron_gradient)[two_electron_index] +=
              weight *
              (delta_alpha_cofactor * beta_cofactor +
               alpha_cofactor * delta_beta_cofactor);
        }
      }
    }
  }
}

void accumulate_regular_spin_overlap_gradient_direction_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    double overlap_determinant,
    double delta_overlap_determinant,
    const Eigen::MatrixXd& inverse_overlap_submatrix,
    const Eigen::MatrixXd& delta_inverse_overlap_submatrix,
    double determinant_overlap_weight,
    double delta_determinant_overlap_weight,
    const Eigen::MatrixXd& inverse_overlap_gradient,
    const Eigen::MatrixXd& delta_inverse_overlap_gradient,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient) {
  if (overlap_determinant == 0.0) {
    throw std::invalid_argument(
        "directional overlap gradient requires non-zero overlap_determinant");
  }

  const Eigen::MatrixXd inverse_overlap_transpose = inverse_overlap_submatrix.transpose();
  const Eigen::MatrixXd delta_inverse_overlap_transpose =
      delta_inverse_overlap_submatrix.transpose();
  Eigen::MatrixXd overlap_submatrix_gradient_direction =
      (delta_determinant_overlap_weight * overlap_determinant +
       determinant_overlap_weight * delta_overlap_determinant) *
      inverse_overlap_transpose;
  overlap_submatrix_gradient_direction.noalias() +=
      determinant_overlap_weight * overlap_determinant *
      delta_inverse_overlap_transpose;
  overlap_submatrix_gradient_direction.noalias() -=
      delta_overlap_determinant *
      inverse_overlap_transpose *
      inverse_overlap_gradient *
      inverse_overlap_transpose;
  overlap_submatrix_gradient_direction.noalias() -=
      overlap_determinant *
      delta_inverse_overlap_transpose *
      inverse_overlap_gradient *
      inverse_overlap_transpose;
  overlap_submatrix_gradient_direction.noalias() -=
      overlap_determinant *
      inverse_overlap_transpose *
      delta_inverse_overlap_gradient *
      inverse_overlap_transpose;
  overlap_submatrix_gradient_direction.noalias() -=
      overlap_determinant *
      inverse_overlap_transpose *
      inverse_overlap_gradient *
      delta_inverse_overlap_transpose;

  for (int left_column = 0; left_column < static_cast<int>(occ_L.size()); ++left_column) {
    const int orbital_index_left = occ_L[left_column];
    for (int right_row = 0; right_row < static_cast<int>(occ_R.size()); ++right_row) {
      const int orbital_index_right = occ_R[right_row];
      (*active_orbital_overlap_gradient)[orbital_index_left *
                                             n_active_orbitals +
                                         orbital_index_right] +=
          overlap_submatrix_gradient_direction(right_row, left_column);
    }
  }
}

bool has_nonzero_structure_pair_adjoints(
    const StructurePairAdjoints& adjoints) {
  return adjoints.hamiltonian_weight != 0.0 ||
      adjoints.overlap_weight != 0.0;
}

std::size_t structure_upper_storage_index(
    int structure_row,
    int structure_column) {
  if (structure_row < 0 || structure_column < 0 ||
      structure_row > structure_column) {
    throw std::invalid_argument("structure upper-triangular index is out of range");
  }
  return structure_column * (structure_column + 1) / 2 +
      structure_row;
}

Eigen::MatrixXd unpack_symmetric_structure_matrix(
    const std::vector<double>& matrix_storage,
    int dimension) {
  const std::size_t expected_size =
      dimension * dimension;
  if (dimension <= 0 || matrix_storage.size() != expected_size) {
    throw std::invalid_argument(
        "structure matrix storage does not match the requested dimension");
  }

  const Eigen::Map<const Eigen::MatrixXd> stored_matrix(
      matrix_storage.data(),
      dimension,
      dimension);
  Eigen::MatrixXd symmetric_matrix = Eigen::MatrixXd::Zero(dimension, dimension);
  for (int column = 0; column < dimension; ++column) {
    for (int row = 0; row <= column; ++row) {
      const double value = stored_matrix(row, column);
      symmetric_matrix(row, column) = value;
      symmetric_matrix(column, row) = value;
    }
  }
  return symmetric_matrix;
}

std::vector<double> pack_symmetric_structure_weight_matrix(
    const Eigen::MatrixXd& weight_matrix) {
  if (weight_matrix.rows() != weight_matrix.cols()) {
    throw std::invalid_argument("structure weight matrix must be square");
  }

  const int dimension = static_cast<int>(weight_matrix.rows());
  std::vector<double> upper_weights(
      dimension * (dimension + 1) / 2,
      0.0);
  for (int structure_column = 0;
       structure_column < dimension;
       ++structure_column) {
    for (int structure_row = 0;
         structure_row <= structure_column;
         ++structure_row) {
      const double symmetry =
          structure_row == structure_column ? 1.0 : 2.0;
      upper_weights[structure_upper_storage_index(
          structure_row,
          structure_column)] =
          symmetry * weight_matrix(structure_row, structure_column);
    }
  }
  return upper_weights;
}

constexpr double kDirectionalStructureContributionTolerance = 1.0e-15;
struct LocalProjectionBlockLocal {
  struct ProjectionSlot {
    const OppositeSpinPackedPairProjection* borrowed_projection = nullptr;
    OppositeSpinPackedPairProjection owned_projection;

    const OppositeSpinPackedPairProjection& projection() const {
      return (borrowed_projection != nullptr)
          ? *borrowed_projection
          : owned_projection;
    }
  };

  int n_rows = 0;
  int n_cols = 0;
  std::vector<ProjectionSlot> projections;

  const OppositeSpinPackedPairProjection& projection(
      int row_local,
      int column_local) const {
    return projections[(column_local) * (n_rows) + (row_local)]
        .projection();
  }
};

struct LocalOppositeSpinChannelFamilyLocal {
  std::vector<int> packed_pair_indices;
  std::vector<Eigen::MatrixXd> alpha_channel_matrices;
};

int count_distinct_projection_block_pairs_local(
    const LocalProjectionBlockLocal& projection_block,
    int n_packed_active_pairs) {
  if (projection_block.n_rows <= 0 ||
      projection_block.n_cols <= 0 ||
      n_packed_active_pairs <= 0) {
    return 0;
  }

  std::vector<unsigned char> touched_mask(
      n_packed_active_pairs,
      0u);
  int distinct_pair_count = 0;
  for (int column_local = 0;
       column_local < projection_block.n_cols;
       ++column_local) {
    for (int row_local = 0;
         row_local < projection_block.n_rows;
         ++row_local) {
      const auto& projection =
          projection_block.projection(row_local, column_local);
      for (const int packed_pair_index : projection.packed_pair_indices) {
        if (packed_pair_index < 0 ||
            packed_pair_index >= n_packed_active_pairs) {
          throw std::out_of_range(
              "packed_pair_index outside local channel range");
        }
        if (touched_mask[packed_pair_index] == 0u) {
          touched_mask[packed_pair_index] = 1u;
          ++distinct_pair_count;
        }
      }
    }
  }
  return distinct_pair_count;
}

int count_distinct_projection_block_pairs_union_local(
    const LocalProjectionBlockLocal& first_projection_block,
    const LocalProjectionBlockLocal& second_projection_block,
    int n_packed_active_pairs) {
  if (first_projection_block.n_rows != second_projection_block.n_rows ||
      first_projection_block.n_cols != second_projection_block.n_cols) {
    throw std::invalid_argument(
        "projection blocks must share the same local shape");
  }
  if (first_projection_block.n_rows <= 0 ||
      first_projection_block.n_cols <= 0 ||
      n_packed_active_pairs <= 0) {
    return 0;
  }

  std::vector<unsigned char> touched_mask(
      n_packed_active_pairs,
      0u);
  int distinct_pair_count = 0;
  for (int column_local = 0;
       column_local < first_projection_block.n_cols;
       ++column_local) {
    for (int row_local = 0;
         row_local < first_projection_block.n_rows;
         ++row_local) {
      const auto& first_projection =
          first_projection_block.projection(row_local, column_local);
      for (const int packed_pair_index : first_projection.packed_pair_indices) {
        if (packed_pair_index < 0 ||
            packed_pair_index >= n_packed_active_pairs) {
          throw std::out_of_range(
              "packed_pair_index outside local channel range");
        }
        if (touched_mask[packed_pair_index] == 0u) {
          touched_mask[packed_pair_index] = 1u;
          ++distinct_pair_count;
        }
      }
      const auto& second_projection =
          second_projection_block.projection(row_local, column_local);
      for (const int packed_pair_index : second_projection.packed_pair_indices) {
        if (packed_pair_index < 0 ||
            packed_pair_index >= n_packed_active_pairs) {
          throw std::out_of_range(
              "packed_pair_index outside local channel range");
        }
        if (touched_mask[packed_pair_index] == 0u) {
          touched_mask[packed_pair_index] = 1u;
          ++distinct_pair_count;
        }
      }
    }
  }
  return distinct_pair_count;
}

LocalOppositeSpinChannelFamilyLocal build_local_channel_family_local(
    const LocalProjectionBlockLocal& projection_block,
    int n_packed_active_pairs) {
  LocalOppositeSpinChannelFamilyLocal channel_family;
  if (projection_block.n_rows <= 0 ||
      projection_block.n_cols <= 0 ||
      n_packed_active_pairs <= 0) {
    return channel_family;
  }

  std::vector<int> channel_index_by_packed_pair(
      n_packed_active_pairs,
      -1);
  for (int column_local = 0;
       column_local < projection_block.n_cols;
       ++column_local) {
    for (int row_local = 0;
         row_local < projection_block.n_rows;
         ++row_local) {
      const auto& projection =
          projection_block.projection(row_local, column_local);
      for (std::size_t entry_index = 0;
           entry_index < projection.packed_pair_indices.size();
           ++entry_index) {
        const int packed_pair_index = projection.packed_pair_indices[entry_index];
        if (packed_pair_index < 0 ||
            packed_pair_index >= n_packed_active_pairs) {
          throw std::out_of_range(
              "packed_pair_index outside local channel range");
        }
        const double packed_pair_value =
            projection.packed_pair_values[entry_index];
        if (std::abs(packed_pair_value) <=
            kDirectionalStructureContributionTolerance) {
          continue;
        }
        int& channel_index =
            channel_index_by_packed_pair[packed_pair_index];
        if (channel_index < 0) {
          channel_index =
              static_cast<int>(channel_family.packed_pair_indices.size());
          channel_family.packed_pair_indices.push_back(packed_pair_index);
          channel_family.alpha_channel_matrices.emplace_back(
              Eigen::MatrixXd::Zero(
                  projection_block.n_rows,
                  projection_block.n_cols));
        }
        channel_family.alpha_channel_matrices[channel_index](row_local, column_local) =
            packed_pair_value;
      }
    }
  }
  return channel_family;
}

/**
 * @brief Reuses the packed-pair to local-channel map across many block gathers.
 *
 * Each gathered structure block only touches a small subset of packed pairs.
 * Reinitializing an `n_packed_active_pairs` sized lookup table for every block
 * would add avoidable O(n_pairs) work. This builder tracks only the touched
 * packed pairs and resets those entries between block gathers.
 */
struct LocalOppositeSpinChannelFamilyBuilderLocal {
  explicit LocalOppositeSpinChannelFamilyBuilderLocal(
      int n_packed_active_pairs)
      : channel_index_by_packed_pair(
            n_packed_active_pairs,
            -1) {
    touched_packed_pair_indices.reserve(
        n_packed_active_pairs);
  }

  void reset(LocalOppositeSpinChannelFamilyLocal* channel_family) {
    for (const int packed_pair_index : touched_packed_pair_indices) {
      channel_index_by_packed_pair[packed_pair_index] = -1;
    }
    touched_packed_pair_indices.clear();
    channel_family->packed_pair_indices.clear();
  }

  void accumulate(
      const OppositeSpinPackedPairProjection& projection,
      int n_rows,
      int n_cols,
      int row_local,
      int column_local,
      LocalOppositeSpinChannelFamilyLocal* channel_family) {
    for (std::size_t entry_index = 0;
         entry_index < projection.packed_pair_indices.size();
         ++entry_index) {
      const int packed_pair_index = projection.packed_pair_indices[entry_index];
      if (packed_pair_index < 0 ||
          packed_pair_index >=
              static_cast<int>(channel_index_by_packed_pair.size())) {
        throw std::out_of_range(
            "packed_pair_index outside local channel range");
      }
      const double packed_pair_value =
          projection.packed_pair_values[entry_index];
      if (std::abs(packed_pair_value) <=
          kDirectionalStructureContributionTolerance) {
        continue;
      }
      int& channel_index =
          channel_index_by_packed_pair[packed_pair_index];
      if (channel_index < 0) {
        channel_index =
            static_cast<int>(channel_family->packed_pair_indices.size());
        touched_packed_pair_indices.push_back(packed_pair_index);
        channel_family->packed_pair_indices.push_back(packed_pair_index);
        if (channel_index <
            static_cast<int>(channel_family->alpha_channel_matrices.size())) {
          Eigen::MatrixXd& reused_channel_matrix =
              channel_family->alpha_channel_matrices[channel_index];
          reused_channel_matrix.resize(n_rows, n_cols);
          reused_channel_matrix.setZero();
        } else {
          channel_family->alpha_channel_matrices.emplace_back(
              Eigen::MatrixXd::Zero(n_rows, n_cols));
        }
      }
      channel_family->alpha_channel_matrices[channel_index](row_local, column_local) = packed_pair_value;
    }
  }

private:
  std::vector<int> channel_index_by_packed_pair;
  std::vector<int> touched_packed_pair_indices;
};

struct DirectionalSpinPairEntry {
  double overlap_determinant = 0.0;
  double total_hamiltonian = 0.0;
  double delta_overlap_determinant = 0.0;
  double delta_total_hamiltonian = 0.0;
  OppositeSpinPackedPairProjection delta_first_order_projection;
};

struct DirectionalSpinPairTile {
  int row_tile = 0;
  int column_tile = 0;
  int row_begin = 0;
  int row_end = 0;
  int column_begin = 0;
  int column_end = 0;
  std::uint64_t last_access_stamp = 0;
  std::vector<DirectionalSpinPairEntry> entries;

  const DirectionalSpinPairEntry& entry(
      int global_row,
      int global_column) const {
    const int local_row = global_row - row_begin;
    const int local_column = global_column - column_begin;
    return entries[(local_column) * (row_end - row_begin) + (local_row)];
  }
};

std::size_t square_storage_size_local(int dimension) {
  return dimension * dimension;
}

int count_distinct_touched_tiles_local(
    const std::vector<int>& indices,
    int tile_size) {
  if (tile_size <= 0) {
    throw std::invalid_argument("tile_size must be positive");
  }
  if (indices.empty()) {
    return 0;
  }

  int distinct_tile_count = 0;
  int previous_tile = std::numeric_limits<int>::min();
  for (const int index : indices) {
    const int tile_index = index / tile_size;
    if (tile_index != previous_tile) {
      ++distinct_tile_count;
      previous_tile = tile_index;
    }
  }
  return distinct_tile_count;
}

int directional_structure_matrix_tile_size() {
  const char* env_value = std::getenv("XMVB_CPP_STRUCTURE_TILE_SIZE");
  if (env_value == nullptr || env_value[0] == '\0') {
    return 256;
  }
  const int tile_size = std::atoi(env_value);
  if (tile_size <= 0) {
    throw std::invalid_argument(
        "XMVB_CPP_STRUCTURE_TILE_SIZE must be a positive integer");
  }
  return tile_size;
}

int directional_structure_matrix_tile_cache_tiles() {
  const char* env_value = std::getenv("XMVB_CPP_STRUCTURE_TILE_CACHE_TILES");
  if (env_value == nullptr || env_value[0] == '\0') {
    return 4;
  }
  const int cache_tiles = std::atoi(env_value);
  if (cache_tiles <= 0) {
    throw std::invalid_argument(
        "XMVB_CPP_STRUCTURE_TILE_CACHE_TILES must be a positive integer");
  }
  return cache_tiles;
}

void reset_local_projection_block_local(
    int n_rows,
    int n_cols,
    LocalProjectionBlockLocal* local_projection_block) {
  if (local_projection_block == nullptr) {
    return;
  }
  local_projection_block->n_rows = n_rows;
  local_projection_block->n_cols = n_cols;
  // Every slot is overwritten before it is consumed, so avoid an extra full
  // memory pass that default-initializes the whole block on every gather.
  local_projection_block->projections.resize(
      n_rows * n_cols);
}

void symmetrize_structure_matrices_local(
    StructureAccumulationResult* accumulation_result) {
  const int n_structures = accumulation_result->n_structures;
  for (int row = 0; row < n_structures; ++row) {
    for (int column = 0; column < row; ++column) {
      const std::size_t lower_index =
          (column) * (n_structures) + (row);
      const std::size_t upper_index =
          (row) * (n_structures) + (column);
      accumulation_result->overlap_matrix[lower_index] =
          accumulation_result->overlap_matrix[upper_index];
      accumulation_result->hamiltonian_matrix[lower_index] =
          accumulation_result->hamiltonian_matrix[upper_index];
    }
  }
}

double frobenius_inner_product_local(
    const Eigen::MatrixXd& left_matrix,
    const Eigen::MatrixXd& right_matrix) {
  if (left_matrix.size() == 0) {
    return 0.0;
  }
  return cblas_ddot(
      static_cast<int>(left_matrix.size()),
      left_matrix.data(),
      1,
      right_matrix.data(),
      1);
}

double contract_structure_pair_kernel_local(
    const Eigen::MatrixXd& left_coefficients,
    const Eigen::MatrixXd& right_coefficients,
    const Eigen::MatrixXd& alpha_kernel,
    const Eigen::MatrixXd& beta_kernel,
    Eigen::MatrixXd* beta_push,
    Eigen::MatrixXd* image) {
  if (left_coefficients.size() == 0 || right_coefficients.size() == 0) {
    return 0.0;
  }

  beta_push->resize(right_coefficients.rows(), beta_kernel.rows());
  beta_push->noalias() = right_coefficients * beta_kernel.transpose();
  image->resize(alpha_kernel.rows(), beta_kernel.rows());
  image->noalias() = alpha_kernel * (*beta_push);
  return frobenius_inner_product_local(left_coefficients, *image);
}

struct DirectionalSameSpinContractionScratchLocal {
  Eigen::MatrixXd alpha_overlap_push;
  Eigen::MatrixXd alpha_total_push;
  Eigen::MatrixXd alpha_delta_overlap_push;
  Eigen::MatrixXd alpha_delta_total_push;
  Eigen::MatrixXd beta_overlap_push;
  Eigen::MatrixXd beta_total_push;
  Eigen::MatrixXd beta_delta_overlap_push;
  Eigen::MatrixXd beta_delta_total_push;
};

struct DirectionalSameSpinContractionResultLocal {
  double overlap = 0.0;
  double hamiltonian = 0.0;
};

DirectionalSameSpinContractionResultLocal
contract_close_shell_diagonal_directional_same_spin_structure_kernels_local(
    const std::vector<double>& left_diagonal_coefficients,
    const std::vector<double>& right_diagonal_coefficients,
    const Eigen::MatrixXd& overlap_subblock,
    const Eigen::MatrixXd& total_subblock,
    const Eigen::MatrixXd& delta_overlap_subblock,
    const Eigen::MatrixXd& delta_total_subblock) {
  // Close-shell diagonal structure blocks satisfy
  //   C_L = diag(l), C_R = diag(r), alpha == beta,
  // so the directional same-spin contraction reduces to one gathered spin
  // channel and a factor of two for the duplicated alpha/beta contribution.
  DirectionalSameSpinContractionResultLocal result;
  result.overlap =
      2.0 *
      contract_diagonal_structure_pair_kernel(
          left_diagonal_coefficients,
          right_diagonal_coefficients,
          delta_overlap_subblock,
          overlap_subblock);
  result.hamiltonian =
      2.0 *
      (contract_diagonal_structure_pair_kernel(
           left_diagonal_coefficients,
           right_diagonal_coefficients,
           delta_total_subblock,
           overlap_subblock) +
       contract_diagonal_structure_pair_kernel(
           left_diagonal_coefficients,
           right_diagonal_coefficients,
           total_subblock,
           delta_overlap_subblock));
  return result;
}

void build_left_alpha_push_local(
    const Eigen::MatrixXd& left_coefficients,
    const Eigen::MatrixXd& right_coefficients,
    const Eigen::MatrixXd& alpha_kernel,
    Eigen::MatrixXd* alpha_push) {
  alpha_push->resize(alpha_kernel.cols(), left_coefficients.cols());
  alpha_push->noalias() = alpha_kernel.transpose() * left_coefficients;
}

void build_right_beta_push_local(
    const Eigen::MatrixXd& left_coefficients,
    const Eigen::MatrixXd& right_coefficients,
    const Eigen::MatrixXd& beta_kernel,
    Eigen::MatrixXd* beta_push) {
  beta_push->resize(right_coefficients.rows(), beta_kernel.rows());
  beta_push->noalias() = right_coefficients * beta_kernel.transpose();
}

DirectionalSameSpinContractionResultLocal
contract_directional_same_spin_structure_kernels_local(
    const Eigen::MatrixXd& left_coefficients,
    const Eigen::MatrixXd& right_coefficients,
    const Eigen::MatrixXd& alpha_overlap_subblock,
    const Eigen::MatrixXd& alpha_total_subblock,
    const Eigen::MatrixXd& alpha_delta_overlap_subblock,
    const Eigen::MatrixXd& alpha_delta_total_subblock,
    const Eigen::MatrixXd& beta_overlap_subblock,
    const Eigen::MatrixXd& beta_total_subblock,
    const Eigen::MatrixXd& beta_delta_overlap_subblock,
    const Eigen::MatrixXd& beta_delta_total_subblock,
    DirectionalSameSpinContractionScratchLocal* scratch) {
  if (left_coefficients.size() == 0 || right_coefficients.size() == 0) {
    return {};
  }

  // Each same-spin term has the form
  //   <L, A * (R * B^T)> = <A^T * L, R * B^T>.
  // The directional overlap/Hamiltonian need six such pairings, with repeated
  // alpha and beta kernels. Building the four unique left pushes and four
  // unique right pushes once per structure pair avoids repeated dense GEMMs
  // while preserving the mathematical contraction and matrix dimensions:
  //   L: n_alpha_left x n_beta_left
  //   R: n_alpha_right x n_beta_right
  //   A^T L and R B^T: n_alpha_right x n_beta_left.
  build_left_alpha_push_local(
      left_coefficients,
      right_coefficients,
      alpha_overlap_subblock,
      &scratch->alpha_overlap_push);
  build_left_alpha_push_local(
      left_coefficients,
      right_coefficients,
      alpha_total_subblock,
      &scratch->alpha_total_push);
  build_left_alpha_push_local(
      left_coefficients,
      right_coefficients,
      alpha_delta_overlap_subblock,
      &scratch->alpha_delta_overlap_push);
  build_left_alpha_push_local(
      left_coefficients,
      right_coefficients,
      alpha_delta_total_subblock,
      &scratch->alpha_delta_total_push);
  build_right_beta_push_local(
      left_coefficients,
      right_coefficients,
      beta_overlap_subblock,
      &scratch->beta_overlap_push);
  build_right_beta_push_local(
      left_coefficients,
      right_coefficients,
      beta_total_subblock,
      &scratch->beta_total_push);
  build_right_beta_push_local(
      left_coefficients,
      right_coefficients,
      beta_delta_overlap_subblock,
      &scratch->beta_delta_overlap_push);
  build_right_beta_push_local(
      left_coefficients,
      right_coefficients,
      beta_delta_total_subblock,
      &scratch->beta_delta_total_push);

  DirectionalSameSpinContractionResultLocal result;
  result.overlap =
      frobenius_inner_product_local(
          scratch->alpha_delta_overlap_push,
          scratch->beta_overlap_push) +
      frobenius_inner_product_local(
          scratch->alpha_overlap_push,
          scratch->beta_delta_overlap_push);
  result.hamiltonian =
      frobenius_inner_product_local(
          scratch->alpha_delta_total_push,
          scratch->beta_overlap_push) +
      frobenius_inner_product_local(
          scratch->alpha_total_push,
          scratch->beta_delta_overlap_push) +
      frobenius_inner_product_local(
          scratch->alpha_delta_overlap_push,
          scratch->beta_total_push) +
      frobenius_inner_product_local(
          scratch->alpha_overlap_push,
          scratch->beta_delta_total_push);
  return result;
}

std::vector<double> apply_directional_active_two_electron_kernel_to_sparse_projection_local(
    int n_active_orbitals,
    const std::vector<int>& packed_pair_indices,
    const std::vector<double>& packed_pair_values,
    const std::vector<double>& delta_packed_active_two_electron_integrals);

void materialize_directional_projected_pair_values_local(
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    int n_orbitals,
    const OppositeSpinPackedPairProjection& accepted_projection,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    OppositeSpinPackedPairProjection* directional_projection) {
  const ActiveSpaceTwoElectronView two_electron_view =
      make_active_space_two_electron_view(active_space_two_electron_result);
  directional_projection->projected_pair_values =
      apply_active_space_two_electron_kernel_to_sparse_projection(
          two_electron_view,
          n_orbitals,
          directional_projection->packed_pair_indices,
          directional_projection->packed_pair_values);
  const std::vector<double> delta_kernel_times_accepted =
      apply_directional_active_two_electron_kernel_to_sparse_projection_local(
          n_orbitals,
          accepted_projection.packed_pair_indices,
          accepted_projection.packed_pair_values,
          delta_packed_active_two_electron_integrals);
  if (directional_projection->projected_pair_values.size() <
      delta_kernel_times_accepted.size()) {
    directional_projection->projected_pair_values.resize(
        delta_kernel_times_accepted.size(),
        0.0);
  }
  for (std::size_t pair_index = 0;
       pair_index < delta_kernel_times_accepted.size();
       ++pair_index) {
    directional_projection->projected_pair_values[pair_index] +=
        delta_kernel_times_accepted[pair_index];
  }
}

OppositeSpinPackedPairProjection
build_sparse_packed_pair_projection_from_coefficients_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& coefficient_matrix,
    bool coefficient_matrix_is_right_by_left,
    int n_orbitals) {
  OppositeSpinPackedPairProjection projection;
  if (occ_L.empty()) {
    return projection;
  }

  const int n_packed_active_pairs = packed_active_pair_count(n_orbitals);
  std::vector<double> dense_pair_values(
      n_packed_active_pairs,
      0.0);
  std::vector<unsigned char> touched_mask(
      n_packed_active_pairs,
      0u);
  std::vector<int> touched_indices;
  touched_indices.reserve(occ_L.size() * occ_R.size());

  for (int left_column = 0;
       left_column < static_cast<int>(occ_L.size());
       ++left_column) {
    const int orbital_index_left = occ_L[left_column];
    for (int right_row = 0;
         right_row < static_cast<int>(occ_R.size());
         ++right_row) {
      const int orbital_index_right = occ_R[right_row];
      const int packed_pair_index = TwoElectronIndexer::packed_pair_index(
          orbital_index_right,
          orbital_index_left);
      if (touched_mask[packed_pair_index] == 0u) {
        touched_mask[packed_pair_index] = 1u;
        touched_indices.push_back(packed_pair_index);
      }
      const double coefficient =
          coefficient_matrix_is_right_by_left
              ? coefficient_matrix(right_row, left_column)
              : coefficient_matrix(left_column, right_row);
      dense_pair_values[packed_pair_index] += coefficient;
    }
  }

  projection.packed_pair_indices.reserve(touched_indices.size());
  projection.packed_pair_values.reserve(touched_indices.size());
  for (const int packed_pair_index : touched_indices) {
    const double packed_pair_value =
        dense_pair_values[packed_pair_index];
    if (std::abs(packed_pair_value) <=
        kDirectionalStructureContributionTolerance) {
      continue;
    }
    projection.packed_pair_indices.push_back(packed_pair_index);
    projection.packed_pair_values.push_back(packed_pair_value);
  }
  return projection;
}

std::vector<double> apply_directional_active_two_electron_kernel_to_sparse_projection_local(
    int n_active_orbitals,
    const std::vector<int>& packed_pair_indices,
    const std::vector<double>& packed_pair_values,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  const int n_packed_active_pairs = packed_active_pair_count(n_active_orbitals);
  std::vector<double> projected_pair_values(
      n_packed_active_pairs,
      0.0);
  if (delta_packed_active_two_electron_integrals.empty() ||
      packed_pair_indices.empty()) {
    return projected_pair_values;
  }

  for (std::size_t entry_index = 0;
       entry_index < packed_pair_indices.size();
       ++entry_index) {
    const int packed_pair_index = packed_pair_indices[entry_index];
    const double packed_pair_value = packed_pair_values[entry_index];
    for (int row_pair = 0; row_pair < n_packed_active_pairs; ++row_pair) {
      const int packed_pair_of_pairs_index =
          TwoElectronIndexer::packed_pair_of_pairs_index(
              row_pair,
              packed_pair_index);
      projected_pair_values[row_pair] +=
          delta_packed_active_two_electron_integrals[
              packed_pair_of_pairs_index] *
          packed_pair_value;
    }
  }
  return projected_pair_values;
}

double project_sparse_projection_onto_packed_pair_local(
    const OppositeSpinPackedPairProjection& sparse_projection,
    int target_packed_pair_index,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_orbitals) {
  if (target_packed_pair_index >= 0 &&
      target_packed_pair_index <
          static_cast<int>(sparse_projection.projected_pair_values.size())) {
    return sparse_projection.projected_pair_values[
        target_packed_pair_index];
  }

  double projected_value = 0.0;
  for (std::size_t entry_index = 0;
       entry_index < sparse_projection.packed_pair_indices.size();
       ++entry_index) {
    projected_value +=
        lookup_active_space_two_electron_kernel_value(
            two_electron_view,
            target_packed_pair_index,
            sparse_projection.packed_pair_indices[entry_index],
            n_orbitals) *
        sparse_projection.packed_pair_values[entry_index];
  }
  return projected_value;
}

double project_directional_sparse_projection_onto_packed_pair_local(
    const OppositeSpinPackedPairProjection& accepted_projection,
    const OppositeSpinPackedPairProjection& directional_projection,
    int target_packed_pair_index,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_orbitals,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  if (target_packed_pair_index >= 0 &&
      target_packed_pair_index <
          static_cast<int>(directional_projection.projected_pair_values.size())) {
    return directional_projection.projected_pair_values[
        target_packed_pair_index];
  }
  double projected_value =
      project_sparse_projection_onto_packed_pair_local(
          directional_projection,
          target_packed_pair_index,
          two_electron_view,
          n_orbitals);
  for (std::size_t entry_index = 0;
       entry_index < accepted_projection.packed_pair_indices.size();
       ++entry_index) {
    projected_value +=
        lookup_directional_active_two_electron_kernel_value(
            delta_packed_active_two_electron_integrals,
            target_packed_pair_index,
            accepted_projection.packed_pair_indices[entry_index]) *
        accepted_projection.packed_pair_values[entry_index];
  }
  return projected_value;
}

void build_local_projected_channel_block_local(
    const LocalProjectionBlockLocal& local_projection_block,
    int target_packed_pair_index,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_orbitals,
    Eigen::MatrixXd* projected_channel_block) {
  projected_channel_block->resize(
      local_projection_block.n_rows,
      local_projection_block.n_cols);
  for (int column_local = 0;
       column_local < local_projection_block.n_cols;
       ++column_local) {
    for (int row_local = 0;
         row_local < local_projection_block.n_rows;
         ++row_local) {
      const auto& projection =
          local_projection_block.projection(
              row_local,
              column_local);
      (*projected_channel_block)(row_local, column_local) =
          project_sparse_projection_onto_packed_pair_local(
              projection,
              target_packed_pair_index,
              two_electron_view,
              n_orbitals);
    }
  }
}

void build_local_directional_projected_channel_block_local(
    const LocalProjectionBlockLocal& accepted_projection_block,
    const LocalProjectionBlockLocal& directional_projection_block,
    int target_packed_pair_index,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_orbitals,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    Eigen::MatrixXd* projected_channel_block) {
  projected_channel_block->resize(
      accepted_projection_block.n_rows,
      accepted_projection_block.n_cols);
  for (int column_local = 0;
       column_local < accepted_projection_block.n_cols;
       ++column_local) {
    for (int row_local = 0;
         row_local < accepted_projection_block.n_rows;
         ++row_local) {
      const auto& accepted_projection =
          accepted_projection_block.projection(
              row_local,
              column_local);
      const auto& directional_projection =
          directional_projection_block.projection(
              row_local,
              column_local);
      (*projected_channel_block)(row_local, column_local) =
          project_directional_sparse_projection_onto_packed_pair_local(
              accepted_projection,
              directional_projection,
              target_packed_pair_index,
              two_electron_view,
              n_orbitals,
              delta_packed_active_two_electron_integrals);
    }
  }
}

double contract_local_directional_opposite_spin_block_local(
    const Eigen::MatrixXd& left_coefficients,
    const Eigen::MatrixXd& right_coefficients,
    const LocalProjectionBlockLocal& accepted_alpha_projection_block,
    const LocalProjectionBlockLocal& directional_alpha_projection_block,
    const LocalOppositeSpinChannelFamilyLocal& accepted_alpha_channels,
    const LocalOppositeSpinChannelFamilyLocal& directional_alpha_channels,
    const LocalProjectionBlockLocal& accepted_beta_projection_block,
    const LocalProjectionBlockLocal& directional_beta_projection_block,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_orbitals,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    Eigen::MatrixXd* beta_projected_channel_block,
    Eigen::MatrixXd* beta_push,
    Eigen::MatrixXd* image) {
  const int n_packed_active_pairs = packed_active_pair_count(n_orbitals);
  const int alpha_channel_count =
      count_distinct_projection_block_pairs_union_local(
          accepted_alpha_projection_block,
          directional_alpha_projection_block,
          n_packed_active_pairs);
  const int beta_channel_count =
      count_distinct_projection_block_pairs_union_local(
          accepted_beta_projection_block,
          directional_beta_projection_block,
          n_packed_active_pairs);
  double contraction = 0.0;
  if (alpha_channel_count <= beta_channel_count) {
    for (std::size_t channel_index = 0;
         channel_index < accepted_alpha_channels.packed_pair_indices.size();
         ++channel_index) {
      build_local_directional_projected_channel_block_local(
          accepted_beta_projection_block,
          directional_beta_projection_block,
          accepted_alpha_channels.packed_pair_indices[channel_index],
          two_electron_view,
          n_orbitals,
          delta_packed_active_two_electron_integrals,
          beta_projected_channel_block);
      contraction +=
          contract_structure_pair_kernel_local(
              left_coefficients,
              right_coefficients,
              accepted_alpha_channels.alpha_channel_matrices[channel_index],
              *beta_projected_channel_block,
              beta_push,
              image);
    }
    for (std::size_t channel_index = 0;
         channel_index < directional_alpha_channels.packed_pair_indices.size();
         ++channel_index) {
      build_local_projected_channel_block_local(
          accepted_beta_projection_block,
          directional_alpha_channels.packed_pair_indices[channel_index],
          two_electron_view,
          n_orbitals,
          beta_projected_channel_block);
      contraction +=
          contract_structure_pair_kernel_local(
              left_coefficients,
              right_coefficients,
              directional_alpha_channels.alpha_channel_matrices[channel_index],
              *beta_projected_channel_block,
              beta_push,
              image);
    }
  } else {
    const LocalOppositeSpinChannelFamilyLocal accepted_beta_channels =
        build_local_channel_family_local(
            accepted_beta_projection_block,
            n_packed_active_pairs);
    const LocalOppositeSpinChannelFamilyLocal directional_beta_channels =
        build_local_channel_family_local(
            directional_beta_projection_block,
            n_packed_active_pairs);
    for (std::size_t channel_index = 0;
         channel_index < accepted_beta_channels.packed_pair_indices.size();
         ++channel_index) {
      build_local_directional_projected_channel_block_local(
          accepted_alpha_projection_block,
          directional_alpha_projection_block,
          accepted_beta_channels.packed_pair_indices[channel_index],
          two_electron_view,
          n_orbitals,
          delta_packed_active_two_electron_integrals,
          beta_projected_channel_block);
      contraction +=
          contract_structure_pair_kernel_local(
              left_coefficients,
              right_coefficients,
              *beta_projected_channel_block,
              accepted_beta_channels.alpha_channel_matrices[channel_index],
              beta_push,
              image);
    }
    for (std::size_t channel_index = 0;
         channel_index < directional_beta_channels.packed_pair_indices.size();
         ++channel_index) {
      build_local_projected_channel_block_local(
          accepted_alpha_projection_block,
          directional_beta_channels.packed_pair_indices[channel_index],
          two_electron_view,
          n_orbitals,
          beta_projected_channel_block);
      contraction +=
          contract_structure_pair_kernel_local(
              left_coefficients,
              right_coefficients,
              *beta_projected_channel_block,
              directional_beta_channels.alpha_channel_matrices[channel_index],
              beta_push,
              image);
    }
  }
  return contraction;
}

double contract_close_shell_diagonal_local_directional_opposite_spin_block_local(
    const std::vector<double>& left_diagonal_coefficients,
    const std::vector<double>& right_diagonal_coefficients,
    const LocalOppositeSpinChannelFamilyLocal& accepted_alpha_channels,
    const LocalOppositeSpinChannelFamilyLocal& directional_alpha_channels,
    const LocalProjectionBlockLocal& accepted_projection_block,
    const LocalProjectionBlockLocal& directional_projection_block,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_orbitals,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    Eigen::MatrixXd* projected_channel_block) {
  double contraction = 0.0;
  for (std::size_t channel_index = 0;
       channel_index < accepted_alpha_channels.packed_pair_indices.size();
       ++channel_index) {
    build_local_directional_projected_channel_block_local(
        accepted_projection_block,
        directional_projection_block,
        accepted_alpha_channels.packed_pair_indices[channel_index],
        two_electron_view,
        n_orbitals,
        delta_packed_active_two_electron_integrals,
        projected_channel_block);
    contraction +=
        contract_diagonal_structure_pair_kernel(
            left_diagonal_coefficients,
            right_diagonal_coefficients,
            accepted_alpha_channels.alpha_channel_matrices[channel_index],
            *projected_channel_block);
  }
  for (std::size_t channel_index = 0;
       channel_index < directional_alpha_channels.packed_pair_indices.size();
       ++channel_index) {
    build_local_projected_channel_block_local(
        accepted_projection_block,
        directional_alpha_channels.packed_pair_indices[channel_index],
        two_electron_view,
        n_orbitals,
        projected_channel_block);
    contraction +=
        contract_diagonal_structure_pair_kernel(
            left_diagonal_coefficients,
            right_diagonal_coefficients,
            directional_alpha_channels.alpha_channel_matrices[channel_index],
            *projected_channel_block);
  }
  return contraction;
}

DirectionalSpinPairEntry build_directional_spin_pair_entry_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const ActiveSpaceOneElectronResult& active_space_one_electron_result,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const SpinDeterminantPairEvaluation& pair_evaluation,
    int n_active_orbitals,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  DirectionalSpinPairEntry result;
  result.overlap_determinant = pair_evaluation.overlap_result.overlap_determinant;
  result.total_hamiltonian = pair_evaluation.total_hamiltonian;
  if (occ_L.empty()) {
    return result;
  }
  // The tiled directional builder must support both regular and singular
  // same-spin determinant pairs. C2H2 and high-spin cases hit nullity>0
  // pairs inside the OpenMP tile build, so throwing here turns the first HVP
  // into a recursive terminate. The singular formulas already live in this
  // file; reuse them instead of falling back to the determinant-pair path.
  const bool is_regular_pair =
      pair_evaluation.overlap_result.nullity == 0 &&
      pair_evaluation.overlap_result.overlap_determinant != 0.0;
  if (is_regular_pair) {
    // Each tile entry stores only the directional same-spin scalars and the
    // sparse `delta U` first-order cofactor projection. The expensive dense
    // projected channel `G delta U + delta G U` is rebuilt only for the local
    // packed-pair channels touched by one structure-pair block.
    const RegularSpinDirectionalData directional_data =
        build_regular_spin_directional_data(
            occ_L,
            occ_R,
            active_space_one_electron_result,
            active_space_two_electron_result,
            pair_evaluation,
            n_active_orbitals,
            delta_active_orbital_overlap_matrix,
            delta_active_one_electron_matrix,
            delta_packed_active_two_electron_integrals);
    result.delta_overlap_determinant =
        directional_data.delta_overlap_determinant;
    result.delta_total_hamiltonian =
        directional_data.delta_overlap_determinant *
            directional_data.same_spin_total_phi +
        directional_data.overlap_determinant *
            directional_data.delta_same_spin_total_phi;
    result.delta_first_order_projection =
        build_sparse_packed_pair_projection_from_coefficients_local(
            occ_L,
            occ_R,
            directional_data.delta_cofactor_1st,
            true,
            n_active_orbitals);
    materialize_directional_projected_pair_values_local(
        active_space_two_electron_result,
        n_active_orbitals,
        pair_evaluation.opposite_spin_pair_cache.first_order_cofactor_projection,
        delta_packed_active_two_electron_integrals,
        &result.delta_first_order_projection);
    return result;
  }

  const SingularSpinDirectionalData singular_directional_data =
      build_singular_spin_directional_data(
          occ_L,
          occ_R,
          active_space_one_electron_result,
          active_space_two_electron_result,
          pair_evaluation,
          n_active_orbitals,
          delta_active_orbital_overlap_matrix,
          delta_active_one_electron_matrix,
          delta_packed_active_two_electron_integrals);
  result.delta_overlap_determinant =
      singular_directional_data.delta_overlap_determinant;
  result.delta_total_hamiltonian =
      singular_directional_data.delta_total_hamiltonian;
  result.delta_first_order_projection =
      build_sparse_packed_pair_projection_from_coefficients_local(
          occ_L,
          occ_R,
          singular_directional_data.delta_cofactor_1st,
          true,
          n_active_orbitals);
  materialize_directional_projected_pair_values_local(
      active_space_two_electron_result,
      n_active_orbitals,
      pair_evaluation.opposite_spin_pair_cache.first_order_cofactor_projection,
      delta_packed_active_two_electron_integrals,
      &result.delta_first_order_projection);
  return result;
}

class DirectionalSpinPairTileProvider {
public:
  DirectionalSpinPairTileProvider(
      const std::vector<std::vector<int>>& unique_spin_determinants,
      const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
      const ActiveSpaceOneElectronResult& active_space_one_electron_result,
      const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
      int n_active_orbitals,
      const std::vector<double>& delta_active_orbital_overlap_matrix,
      const std::vector<double>& delta_active_one_electron_matrix,
      const std::vector<double>& delta_packed_active_two_electron_integrals,
      int tile_size,
      int max_cached_tiles)
      : unique_spin_determinants_(unique_spin_determinants),
        ordered_pair_cache_(ordered_pair_cache),
        active_space_one_electron_result_(active_space_one_electron_result),
        active_space_two_electron_result_(active_space_two_electron_result),
        n_active_orbitals_(n_active_orbitals),
        delta_active_orbital_overlap_matrix_(delta_active_orbital_overlap_matrix),
        delta_active_one_electron_matrix_(delta_active_one_electron_matrix),
        delta_packed_active_two_electron_integrals_(
            delta_packed_active_two_electron_integrals),
        tile_size_(tile_size),
        max_cached_tiles_(max_cached_tiles) {
    if (tile_size_ <= 0 || max_cached_tiles_ <= 0) {
      throw std::invalid_argument(
          "directional spin tile provider requires positive cache settings");
    }
    const std::size_t expected_cache_entries =
        square_storage_size_local(static_cast<int>(unique_spin_determinants_.size()));
    if (ordered_pair_cache_.size() != expected_cache_entries) {
      throw std::invalid_argument(
          "ordered same-spin pair cache size does not match unique determinant count");
    }
    cached_tiles_.reserve(max_cached_tiles_);
  }

  const SpinDeterminantPairEvaluation& pair_evaluation(
      int left_unique_index,
      int right_unique_index) const {
    if (left_unique_index < 0 ||
        left_unique_index >= static_cast<int>(unique_spin_determinants_.size()) ||
        right_unique_index < 0 ||
        right_unique_index >= static_cast<int>(unique_spin_determinants_.size())) {
      throw std::out_of_range("directional spin pair lookup index out of range");
    }
    return ordered_pair_cache_[ordered_spin_pair_storage_index(
        left_unique_index,
        right_unique_index,
        static_cast<int>(unique_spin_determinants_.size()))];
  }

  const DirectionalSpinPairEntry& entry(
      int left_unique_index,
      int right_unique_index) {
    if (left_unique_index < 0 ||
        left_unique_index >= static_cast<int>(unique_spin_determinants_.size()) ||
        right_unique_index < 0 ||
        right_unique_index >= static_cast<int>(unique_spin_determinants_.size())) {
      throw std::out_of_range("directional spin tile lookup index out of range");
    }

    const int row_tile = left_unique_index / tile_size_;
    const int column_tile = right_unique_index / tile_size_;
    DirectionalSpinPairTile* tile = find_or_build_tile(row_tile, column_tile);
    tile->last_access_stamp = ++access_stamp_;
    return tile->entry(left_unique_index, right_unique_index);
  }

  bool can_borrow_directional_projections_for_block(
      const std::vector<int>& row_indices,
      const std::vector<int>& column_indices) const {
    const std::size_t required_tiles =
        count_distinct_touched_tiles_local(
            row_indices,
            tile_size_) *
        count_distinct_touched_tiles_local(
            column_indices,
            tile_size_);
    return required_tiles <= max_cached_tiles_;
  }

private:
  DirectionalSpinPairTile* find_or_build_tile(
      int row_tile,
      int column_tile) {
    for (auto& tile : cached_tiles_) {
      if (tile.row_tile == row_tile && tile.column_tile == column_tile) {
        return &tile;
      }
    }

    const int row_begin = row_tile * tile_size_;
    const int column_begin = column_tile * tile_size_;
    const int row_end = std::min(
        row_begin + tile_size_,
        static_cast<int>(unique_spin_determinants_.size()));
    const int column_end = std::min(
        column_begin + tile_size_,
        static_cast<int>(unique_spin_determinants_.size()));

    DirectionalSpinPairTile built_tile;
    built_tile.row_tile = row_tile;
    built_tile.column_tile = column_tile;
    built_tile.row_begin = row_begin;
    built_tile.row_end = row_end;
    built_tile.column_begin = column_begin;
    built_tile.column_end = column_end;
    built_tile.last_access_stamp = ++access_stamp_;
    built_tile.entries.resize(
        (row_end - row_begin) * (column_end - column_begin));

    for (int column_index = column_begin;
         column_index < column_end;
         ++column_index) {
      for (int row_index = row_begin;
           row_index < row_end;
           ++row_index) {
        built_tile.entries[(column_index - column_begin) * (row_end - row_begin) + (row_index - row_begin)] =
            build_directional_spin_pair_entry_local(
                unique_spin_determinants_[row_index],
                unique_spin_determinants_[column_index],
                active_space_one_electron_result_,
                active_space_two_electron_result_,
                pair_evaluation(row_index, column_index),
                n_active_orbitals_,
                delta_active_orbital_overlap_matrix_,
                delta_active_one_electron_matrix_,
                delta_packed_active_two_electron_integrals_);
      }
    }

    if (static_cast<int>(cached_tiles_.size()) == max_cached_tiles_) {
      auto victim_iterator = cached_tiles_.begin();
      for (auto iterator = cached_tiles_.begin();
           iterator != cached_tiles_.end();
           ++iterator) {
        if (iterator->last_access_stamp < victim_iterator->last_access_stamp) {
          victim_iterator = iterator;
        }
      }
      *victim_iterator = std::move(built_tile);
      return &(*victim_iterator);
    }

    cached_tiles_.push_back(std::move(built_tile));
    return &cached_tiles_.back();
  }

  const std::vector<std::vector<int>>& unique_spin_determinants_;
  const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache_;
  const ActiveSpaceOneElectronResult& active_space_one_electron_result_;
  const ActiveSpaceTwoElectronResult& active_space_two_electron_result_;
  int n_active_orbitals_ = 0;
  const std::vector<double>& delta_active_orbital_overlap_matrix_;
  const std::vector<double>& delta_active_one_electron_matrix_;
  const std::vector<double>& delta_packed_active_two_electron_integrals_;
  int tile_size_ = 0;
  int max_cached_tiles_ = 0;
  std::uint64_t access_stamp_ = 0;
  std::vector<DirectionalSpinPairTile> cached_tiles_;
};

void gather_directional_spin_block_local(
    DirectionalSpinPairTileProvider* tile_provider,
    const std::vector<int>& row_indices,
    const std::vector<int>& column_indices,
    Eigen::MatrixXd* overlap_block,
    Eigen::MatrixXd* total_block,
    Eigen::MatrixXd* delta_overlap_block,
    Eigen::MatrixXd* delta_total_block,
    LocalProjectionBlockLocal* accepted_projection_block,
    LocalProjectionBlockLocal* directional_projection_block,
    LocalOppositeSpinChannelFamilyBuilderLocal* accepted_channel_builder,
    LocalOppositeSpinChannelFamilyLocal* accepted_channel_family,
    LocalOppositeSpinChannelFamilyBuilderLocal* directional_channel_builder,
    LocalOppositeSpinChannelFamilyLocal* directional_channel_family) {
  const bool gather_accepted_channels =
      accepted_channel_builder != nullptr ||
      accepted_channel_family != nullptr;
  const bool gather_directional_channels =
      directional_channel_builder != nullptr ||
      directional_channel_family != nullptr;

  overlap_block->resize(
      static_cast<int>(row_indices.size()),
      static_cast<int>(column_indices.size()));
  total_block->resize(overlap_block->rows(), overlap_block->cols());
  delta_overlap_block->resize(overlap_block->rows(), overlap_block->cols());
  delta_total_block->resize(overlap_block->rows(), overlap_block->cols());
  reset_local_projection_block_local(
      overlap_block->rows(),
      overlap_block->cols(),
      accepted_projection_block);
  reset_local_projection_block_local(
      overlap_block->rows(),
      overlap_block->cols(),
      directional_projection_block);
  if (gather_accepted_channels) {
    accepted_channel_builder->reset(accepted_channel_family);
  }
  if (gather_directional_channels) {
    directional_channel_builder->reset(directional_channel_family);
  }
  // Borrowing directional tile payloads is safe only when the whole block fits
  // inside the provider's tile cache. Otherwise later tile builds could evict
  // an earlier tile before the local contraction consumes its projection.
  const bool can_borrow_directional_projections =
      directional_projection_block != nullptr &&
      tile_provider->can_borrow_directional_projections_for_block(
          row_indices,
          column_indices);

  for (int column_local = 0;
       column_local < static_cast<int>(column_indices.size());
       ++column_local) {
    const int column_global = column_indices[column_local];
    for (int row_local = 0;
         row_local < static_cast<int>(row_indices.size());
         ++row_local) {
      const int row_global = row_indices[row_local];
      const auto& pair_evaluation =
          tile_provider->pair_evaluation(row_global, column_global);
      const auto& directional_entry =
          tile_provider->entry(row_global, column_global);
      (*overlap_block)(row_local, column_local) =
          pair_evaluation.overlap_result.overlap_determinant;
      (*total_block)(row_local, column_local) =
          pair_evaluation.total_hamiltonian;
      (*delta_overlap_block)(row_local, column_local) =
          directional_entry.delta_overlap_determinant;
      (*delta_total_block)(row_local, column_local) =
          directional_entry.delta_total_hamiltonian;
      const auto& accepted_projection =
          pair_evaluation
              .opposite_spin_pair_cache
              .first_order_cofactor_projection;

      if (accepted_projection_block != nullptr) {
        auto& accepted_projection_slot = accepted_projection_block->projections[(column_local) * (overlap_block->rows()) + (row_local)];
        accepted_projection_slot.borrowed_projection = &accepted_projection;
        accepted_projection_slot.owned_projection =
            OppositeSpinPackedPairProjection{};
      }
      if (gather_accepted_channels) {
        accepted_channel_builder->accumulate(
            accepted_projection,
            overlap_block->rows(),
            overlap_block->cols(),
            row_local,
            column_local,
            accepted_channel_family);
      }

      if (directional_projection_block != nullptr) {
        auto& directional_projection_slot = directional_projection_block->projections[(column_local) * (overlap_block->rows()) + (row_local)];
        if (can_borrow_directional_projections) {
          directional_projection_slot.borrowed_projection =
              &directional_entry.delta_first_order_projection;
          directional_projection_slot.owned_projection =
              OppositeSpinPackedPairProjection{};
        } else {
          directional_projection_slot.borrowed_projection = nullptr;
          directional_projection_slot.owned_projection =
              directional_entry.delta_first_order_projection;
        }
      }
      if (gather_directional_channels) {
        directional_channel_builder->accumulate(
            directional_entry.delta_first_order_projection,
            overlap_block->rows(),
            overlap_block->cols(),
            row_local,
            column_local,
            directional_channel_family);
      }
    }
  }
}

StructureAccumulationResult build_tiled_directional_structure_matrices(
    const CppVbInput& input,
    const CppActiveSpaceSecondOrderContext& accepted_point_context,
    const std::vector<StructureCoefficientBlock>& coefficient_blocks,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  const int n_determinants =
      static_cast<int>(input.structure_data.alpha_det.size());
  const int n_structures = input.structure_data.n_structures;
  const int n_active_orbitals =
      input.orbital_preparation_input.n_active_orbitals;
  if (n_determinants <= 0 || n_structures <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument(
        "directional structure matrices require positive dimensions");
  }

  const auto& same_spin_pair_cache = accepted_point_context.same_spin_pair_cache;
  if (!same_spin_pair_cache.enabled()) {
    throw std::invalid_argument(
        "tiled directional structure builder requires same-spin cache");
  }
  if (static_cast<int>(coefficient_blocks.size()) != n_structures) {
    throw std::invalid_argument(
        "tiled directional structure coefficient blocks do not match n_structures");
  }
  const bool shared_same_spin_pair_cache =
      same_spin_pair_cache.shares_same_spin_pair_cache_between_spins();
  const bool close_shell_same_spin =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();
  const int tile_size = directional_structure_matrix_tile_size();
  const int max_cached_tiles = directional_structure_matrix_tile_cache_tiles();
  const int n_packed_active_pairs =
      packed_active_pair_count(n_active_orbitals);
  const auto& accepted_prepared_active_space =
      accepted_point_context.prepared_active_space;
  const ActiveSpaceTwoElectronView two_electron_view =
      make_active_space_two_electron_view(
          accepted_prepared_active_space.active_space_two_electron_result);

  StructureAccumulationResult result;
  result.n_structures = n_structures;
  result.overlap_matrix.assign(
      square_storage_size_local(n_structures),
      0.0);
  result.hamiltonian_matrix.assign(
      square_storage_size_local(n_structures),
      0.0);
  result.determinant_overlap_cache.assign(
      n_determinants,
      0.0);
  std::exception_ptr parallel_exception;
  std::atomic<bool> parallel_failed(false);
  const int n_parallel_threads =
      choose_exact_ctx_directional_structure_threads(n_structures);
  const bool log_structure_stage = exact_ctx_structure_stage_logging_enabled();
  std::vector<double> thread_alpha_gather_wall_seconds(
      std::max(1, n_parallel_threads),
      0.0);
  std::vector<double> thread_beta_gather_wall_seconds(
      std::max(1, n_parallel_threads),
      0.0);
  std::vector<double> thread_same_spin_wall_seconds(
      std::max(1, n_parallel_threads),
      0.0);
  std::vector<double> thread_opposite_spin_wall_seconds(
      std::max(1, n_parallel_threads),
      0.0);
  std::vector<std::size_t> thread_structure_pair_count(
      std::max(1, n_parallel_threads),
      0);
  std::vector<std::size_t> thread_diagonal_structure_pair_count(
      std::max(1, n_parallel_threads),
      0);

// The matrix-form directional builder mirrors the tiled forward structure path:
// it visits only the unique alpha/beta support blocks touched by each
// structure pair and keeps the opposite-spin directional response sparse until
// one local packed-pair channel is actually needed.
#pragma omp parallel if(n_parallel_threads > 1 && n_structures > 2) num_threads(n_parallel_threads)
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    DirectionalSpinPairTileProvider thread_alpha_provider(
        same_spin_pair_cache.alpha_reuse_table.unique_determinants,
        same_spin_pair_cache.alpha_pair_cache_ref(),
        accepted_prepared_active_space.active_space_one_electron_result,
        accepted_prepared_active_space.active_space_two_electron_result,
        n_active_orbitals,
        delta_active_orbital_overlap_matrix,
        delta_active_one_electron_matrix,
        delta_packed_active_two_electron_integrals,
        tile_size,
        max_cached_tiles);
    DirectionalSpinPairTileProvider thread_beta_provider(
        same_spin_pair_cache.beta_reuse_table.unique_determinants,
        same_spin_pair_cache.beta_pair_cache_ref(),
        accepted_prepared_active_space.active_space_one_electron_result,
        accepted_prepared_active_space.active_space_two_electron_result,
        n_active_orbitals,
        delta_active_orbital_overlap_matrix,
        delta_active_one_electron_matrix,
        delta_packed_active_two_electron_integrals,
        tile_size,
        max_cached_tiles);
    Eigen::MatrixXd alpha_overlap_subblock;
    Eigen::MatrixXd alpha_total_subblock;
    Eigen::MatrixXd alpha_delta_overlap_subblock;
    Eigen::MatrixXd alpha_delta_total_subblock;
    Eigen::MatrixXd beta_overlap_subblock;
    Eigen::MatrixXd beta_total_subblock;
    Eigen::MatrixXd beta_delta_overlap_subblock;
    Eigen::MatrixXd beta_delta_total_subblock;
    Eigen::MatrixXd beta_projected_channel_block;
    Eigen::MatrixXd beta_push;
    Eigen::MatrixXd image;
    DirectionalSameSpinContractionScratchLocal same_spin_scratch;
    LocalProjectionBlockLocal accepted_alpha_projection_block;
    LocalProjectionBlockLocal directional_alpha_projection_block;
    LocalProjectionBlockLocal accepted_beta_projection_block;
    LocalProjectionBlockLocal directional_beta_projection_block;
    LocalOppositeSpinChannelFamilyBuilderLocal accepted_alpha_channel_builder(
        n_packed_active_pairs);
    LocalOppositeSpinChannelFamilyBuilderLocal directional_alpha_channel_builder(
        n_packed_active_pairs);
    LocalOppositeSpinChannelFamilyLocal accepted_alpha_channels;
    LocalOppositeSpinChannelFamilyLocal directional_alpha_channels;
    double local_alpha_gather_wall_seconds = 0.0;
    double local_beta_gather_wall_seconds = 0.0;
    double local_same_spin_wall_seconds = 0.0;
    double local_opposite_spin_wall_seconds = 0.0;
    std::size_t local_structure_pair_count = 0;
    std::size_t local_diagonal_structure_pair_count = 0;

#pragma omp for schedule(dynamic)
    for (int right_structure = 0;
         right_structure < n_structures;
         ++right_structure) {
      if (parallel_failed.load(std::memory_order_relaxed)) {
        continue;
      }
      const auto& right_block =
          coefficient_blocks[right_structure];
      try {
        for (int left_structure = 0;
             left_structure <= right_structure;
             ++left_structure) {
          if (parallel_failed.load(std::memory_order_relaxed)) {
            continue;
          }
          const auto& left_block =
              coefficient_blocks[left_structure];
          const std::size_t linear_index =
              (right_structure) * (n_structures) + (left_structure);
          if (left_block.local_coefficients.size() == 0 ||
              right_block.local_coefficients.size() == 0) {
            result.overlap_matrix[linear_index] = 0.0;
            result.hamiltonian_matrix[linear_index] = 0.0;
            continue;
          }

          const bool structure_pair_close_shell_diagonal =
              close_shell_same_spin &&
              left_block.close_shell_diagonal &&
              right_block.close_shell_diagonal;
          ++local_structure_pair_count;
          if (structure_pair_close_shell_diagonal) {
            ++local_diagonal_structure_pair_count;
          }
          if (structure_pair_close_shell_diagonal) {
            const auto alpha_gather_start_time =
                log_structure_stage
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point();
            gather_directional_spin_block_local(
                &thread_alpha_provider,
                left_block.alpha_support,
                right_block.alpha_support,
                &alpha_overlap_subblock,
                &alpha_total_subblock,
                &alpha_delta_overlap_subblock,
                &alpha_delta_total_subblock,
                &accepted_alpha_projection_block,
                &directional_alpha_projection_block,
                &accepted_alpha_channel_builder,
                &accepted_alpha_channels,
                &directional_alpha_channel_builder,
                &directional_alpha_channels);
            if (log_structure_stage) {
              local_alpha_gather_wall_seconds +=
                  elapsed_wall_time_seconds(alpha_gather_start_time);
            }
          } else {
            const auto alpha_gather_start_time =
                log_structure_stage
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point();
            gather_directional_spin_block_local(
                &thread_alpha_provider,
                left_block.alpha_support,
                right_block.alpha_support,
                &alpha_overlap_subblock,
                &alpha_total_subblock,
                &alpha_delta_overlap_subblock,
                &alpha_delta_total_subblock,
                &accepted_alpha_projection_block,
                &directional_alpha_projection_block,
                &accepted_alpha_channel_builder,
                &accepted_alpha_channels,
                &directional_alpha_channel_builder,
                &directional_alpha_channels);
            if (log_structure_stage) {
              local_alpha_gather_wall_seconds +=
                  elapsed_wall_time_seconds(alpha_gather_start_time);
            }
            const auto beta_gather_start_time =
                log_structure_stage
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point();
            gather_directional_spin_block_local(
                shared_same_spin_pair_cache
                    ? &thread_alpha_provider
                    : &thread_beta_provider,
                left_block.beta_support,
                right_block.beta_support,
                &beta_overlap_subblock,
                &beta_total_subblock,
                &beta_delta_overlap_subblock,
                &beta_delta_total_subblock,
                &accepted_beta_projection_block,
                &directional_beta_projection_block,
                nullptr,
                nullptr,
                nullptr,
                nullptr);
            if (log_structure_stage) {
              local_beta_gather_wall_seconds +=
                  elapsed_wall_time_seconds(beta_gather_start_time);
            }
          }

          double directional_overlap = 0.0;
          double directional_hamiltonian = 0.0;
          if (structure_pair_close_shell_diagonal) {
            const auto same_spin_start_time =
                log_structure_stage
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point();
            const auto same_spin_contraction =
                contract_close_shell_diagonal_directional_same_spin_structure_kernels_local(
                    left_block.local_diagonal_coefficients,
                    right_block.local_diagonal_coefficients,
                    alpha_overlap_subblock,
                    alpha_total_subblock,
                    alpha_delta_overlap_subblock,
                    alpha_delta_total_subblock);
            if (log_structure_stage) {
              local_same_spin_wall_seconds +=
                  elapsed_wall_time_seconds(same_spin_start_time);
            }
            directional_overlap = same_spin_contraction.overlap;
            directional_hamiltonian = same_spin_contraction.hamiltonian;
            const auto opposite_spin_start_time =
                log_structure_stage
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point();
            directional_hamiltonian +=
                contract_close_shell_diagonal_local_directional_opposite_spin_block_local(
                    left_block.local_diagonal_coefficients,
                    right_block.local_diagonal_coefficients,
                    accepted_alpha_channels,
                    directional_alpha_channels,
                    accepted_alpha_projection_block,
                    directional_alpha_projection_block,
                    two_electron_view,
                    n_active_orbitals,
                    delta_packed_active_two_electron_integrals,
                    &beta_projected_channel_block);
            if (log_structure_stage) {
              local_opposite_spin_wall_seconds +=
                  elapsed_wall_time_seconds(opposite_spin_start_time);
            }
          } else {
            const auto same_spin_start_time =
                log_structure_stage
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point();
            const auto same_spin_contraction =
                contract_directional_same_spin_structure_kernels_local(
                    left_block.local_coefficients,
                    right_block.local_coefficients,
                    alpha_overlap_subblock,
                    alpha_total_subblock,
                    alpha_delta_overlap_subblock,
                    alpha_delta_total_subblock,
                    beta_overlap_subblock,
                    beta_total_subblock,
                    beta_delta_overlap_subblock,
                    beta_delta_total_subblock,
                    &same_spin_scratch);
            if (log_structure_stage) {
              local_same_spin_wall_seconds +=
                  elapsed_wall_time_seconds(same_spin_start_time);
            }
            directional_overlap = same_spin_contraction.overlap;
            directional_hamiltonian = same_spin_contraction.hamiltonian;
            const auto opposite_spin_start_time =
                log_structure_stage
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point();
            directional_hamiltonian +=
                contract_local_directional_opposite_spin_block_local(
                    left_block.local_coefficients,
                    right_block.local_coefficients,
                    accepted_alpha_projection_block,
                    directional_alpha_projection_block,
                    accepted_alpha_channels,
                    directional_alpha_channels,
                    accepted_beta_projection_block,
                    directional_beta_projection_block,
                    two_electron_view,
                    n_active_orbitals,
                    delta_packed_active_two_electron_integrals,
                    &beta_projected_channel_block,
                    &beta_push,
                    &image);
            if (log_structure_stage) {
              local_opposite_spin_wall_seconds +=
                  elapsed_wall_time_seconds(opposite_spin_start_time);
            }
          }

          result.overlap_matrix[linear_index] = directional_overlap;
          result.hamiltonian_matrix[linear_index] = directional_hamiltonian;
        }
      } catch (...) {
        // OpenMP cannot propagate exceptions across the parallel region.
        // Capture the first failure so exact_ctx reports the real cause
        // instead of collapsing into a recursive terminate.
        parallel_failed.store(true, std::memory_order_relaxed);
#pragma omp critical(exact_ctx_directional_structure_exception)
        {
          if (!parallel_exception) {
            parallel_exception = std::current_exception();
          }
        }
      }
    }

    thread_alpha_gather_wall_seconds[thread_index] =
        local_alpha_gather_wall_seconds;
    thread_beta_gather_wall_seconds[thread_index] =
        local_beta_gather_wall_seconds;
    thread_same_spin_wall_seconds[thread_index] =
        local_same_spin_wall_seconds;
    thread_opposite_spin_wall_seconds[thread_index] =
        local_opposite_spin_wall_seconds;
    thread_structure_pair_count[thread_index] =
        local_structure_pair_count;
    thread_diagonal_structure_pair_count[thread_index] =
        local_diagonal_structure_pair_count;
  }

  if (parallel_exception) {
    try {
      std::rethrow_exception(parallel_exception);
    } catch (const std::exception& error) {
      throw std::runtime_error(
          std::string("exact outer-response directional structure build failed: ") +
          error.what());
    } catch (...) {
      throw std::runtime_error(
          "exact outer-response directional structure build failed with an unknown exception");
    }
  }

  symmetrize_structure_matrices_local(&result);
  if (log_structure_stage) {
    double alpha_gather_wall_seconds = 0.0;
    double beta_gather_wall_seconds = 0.0;
    double same_spin_wall_seconds = 0.0;
    double opposite_spin_wall_seconds = 0.0;
    std::size_t structure_pair_count = 0;
    std::size_t diagonal_structure_pair_count = 0;
    for (std::size_t thread_index = 0;
         thread_index < thread_alpha_gather_wall_seconds.size();
         ++thread_index) {
      alpha_gather_wall_seconds +=
          thread_alpha_gather_wall_seconds[thread_index];
      beta_gather_wall_seconds +=
          thread_beta_gather_wall_seconds[thread_index];
      same_spin_wall_seconds +=
          thread_same_spin_wall_seconds[thread_index];
      opposite_spin_wall_seconds +=
          thread_opposite_spin_wall_seconds[thread_index];
      structure_pair_count +=
          thread_structure_pair_count[thread_index];
      diagonal_structure_pair_count +=
          thread_diagonal_structure_pair_count[thread_index];
    }
    std::fprintf(
        stderr,
        "exact_ctx_structure_stage"
        " mode=tiled threads=%d n_structures=%d"
        " pairs=%zu diagonal_pairs=%zu"
        " alpha_gather_s=%.6f beta_gather_s=%.6f"
        " same_spin_s=%.6f opposite_spin_s=%.6f\n",
        n_parallel_threads,
        n_structures,
        structure_pair_count,
        diagonal_structure_pair_count,
        alpha_gather_wall_seconds,
        beta_gather_wall_seconds,
        same_spin_wall_seconds,
        opposite_spin_wall_seconds);
    std::fflush(stderr);
  }
  return result;
}

SameSpinPhiResult evaluate_same_spin_phi_with_optional_cache_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const SpinDeterminantPairEvaluation& pair_evaluation,
    Eigen::MatrixXd* inverse_overlap_gradient) {
  if (pair_evaluation.has_same_spin_phi_cache) {
    if (inverse_overlap_gradient != nullptr) {
      *inverse_overlap_gradient = pair_evaluation.same_spin_inverse_overlap_gradient;
    }
    return {
        pair_evaluation.same_spin_one_electron_phi,
        pair_evaluation.same_spin_total_phi,
    };
  }

  return compute_same_spin_original_phi(
      occ_L,
      occ_R,
      h1e_act,
      n_active_orbitals,
      active_space_two_electron_result,
      pair_evaluation.overlap_result,
      inverse_overlap_gradient);
}

void accumulate_one_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& cofactor_1st,
    double weight,
    Eigen::MatrixXd* active_one_electron_gradient) {
  if (weight == 0.0) {
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

void accumulate_same_spin_two_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& cofactor_1st,
    double overlap_determinant,
    double weight,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (weight == 0.0) {
    return;
  }

  const int n_electrons = static_cast<int>(occ_L.size());
  const double cofactor_scale = weight / overlap_determinant;
  for (int left_first = 0; left_first < n_electrons - 1; ++left_first) {
    const int orbital_index_left_first = occ_L[left_first];
    for (int right_first = 0; right_first < n_electrons - 1; ++right_first) {
      const int orbital_index_right_first = occ_R[right_first];
      const double cofactor_11 = cofactor_1st(right_first, left_first);
      for (int left_second = left_first + 1; left_second < n_electrons; ++left_second) {
        const int orbital_index_left_second = occ_L[left_second];
        const double cofactor_12 = cofactor_1st(right_first, left_second);
        for (int right_second = right_first + 1; right_second < n_electrons; ++right_second) {
          const int orbital_index_right_second = occ_R[right_second];
          const double cofactor_22 = cofactor_1st(right_second, left_second);
          const double cofactor_21 = cofactor_1st(right_second, left_first);
          const double second_order_cofactor =
              cofactor_scale * (cofactor_11 * cofactor_22 - cofactor_12 * cofactor_21);

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

void accumulate_opposite_spin_two_electron_gradient_contribution_local(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const Eigen::MatrixXd& alpha_cofactor_1st,
    const OppositeSpinPairCache* alpha_pair_cache,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const Eigen::MatrixXd& beta_cofactor_1st,
    const OppositeSpinPairCache* beta_pair_cache,
    double weight,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (weight == 0.0) {
    return;
  }

  if (alpha_pair_cache != nullptr && beta_pair_cache != nullptr &&
      has_opposite_spin_first_order_projection(*alpha_pair_cache) &&
      has_opposite_spin_first_order_projection(*beta_pair_cache) &&
      alpha_pair_cache->n_packed_active_pairs ==
          beta_pair_cache->n_packed_active_pairs) {
    const auto& alpha_projection = alpha_pair_cache->first_order_cofactor_projection;
    const auto& beta_projection = beta_pair_cache->first_order_cofactor_projection;
    for (std::size_t alpha_entry = 0;
         alpha_entry < alpha_projection.packed_pair_indices.size();
         ++alpha_entry) {
      const int alpha_packed_pair_index =
          alpha_projection.packed_pair_indices[alpha_entry];
      const double weighted_alpha_value =
          weight * alpha_projection.packed_pair_values[alpha_entry];
      for (std::size_t beta_entry = 0;
           beta_entry < beta_projection.packed_pair_indices.size();
           ++beta_entry) {
        const int beta_packed_pair_index =
            beta_projection.packed_pair_indices[beta_entry];
        const int packed_pair_of_pairs_index =
            TwoElectronIndexer::packed_pair_of_pairs_index(
                beta_packed_pair_index,
                alpha_packed_pair_index);
        (*packed_active_two_electron_gradient)[
            packed_pair_of_pairs_index] +=
            weighted_alpha_value * beta_projection.packed_pair_values[beta_entry];
      }
    }
    return;
  }

  for (int alpha_left_column = 0;
       alpha_left_column < static_cast<int>(alpha_occ_L.size());
       ++alpha_left_column) {
    const int alpha_orbital_left =
        alpha_occ_L[alpha_left_column];
    for (int alpha_right_row = 0;
         alpha_right_row < static_cast<int>(alpha_occ_R.size());
         ++alpha_right_row) {
      const int alpha_orbital_right =
          alpha_occ_R[alpha_right_row];
      const double weighted_alpha_cofactor =
          weight * alpha_cofactor_1st(alpha_right_row, alpha_left_column);
      for (int beta_left_column = 0;
           beta_left_column < static_cast<int>(beta_occ_L.size());
           ++beta_left_column) {
        const int beta_orbital_left =
            beta_occ_L[beta_left_column];
        for (int beta_right_row = 0;
             beta_right_row < static_cast<int>(beta_occ_R.size());
             ++beta_right_row) {
          const int beta_orbital_right =
              beta_occ_R[beta_right_row];
          const int two_electron_index = TwoElectronIndexer::two_electron_storage_index(
              beta_orbital_right,
              beta_orbital_left,
              alpha_orbital_right,
              alpha_orbital_left);
          (*packed_active_two_electron_gradient)[two_electron_index] +=
              weighted_alpha_cofactor *
              beta_cofactor_1st(beta_right_row, beta_left_column);
        }
      }
    }
  }
}

void accumulate_active_space_gradient_pair_with_adjoints_local(
    const StructurePairAdjoints& pair_adjoints,
    const CppVbInput& input,
    const ActiveSpaceOneElectronResult& active_space_one_electron_result,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const FullDeterminantPairEvaluation& determinant_pair_evaluation,
    int determinant_index_left,
    int determinant_index_right,
    int n_active_orbitals,
    bool skip_opposite_spin,
    Eigen::MatrixXd* active_one_electron_gradient,
    std::vector<double>* active_orbital_overlap_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (!has_nonzero_structure_pair_adjoints(pair_adjoints)) {
    return;
  }

  const auto& alpha_occ_L =
      input.structure_data.alpha_det[determinant_index_left];
  const auto& alpha_occ_R =
      input.structure_data.alpha_det[determinant_index_right];
  const auto& beta_occ_L =
      input.structure_data.beta_det[determinant_index_left];
  const auto& beta_occ_R =
      input.structure_data.beta_det[determinant_index_right];
  const auto& alpha_result = determinant_pair_evaluation.alpha.overlap_result;
  const auto& beta_result = determinant_pair_evaluation.beta.overlap_result;
  const Eigen::MatrixXd alpha_cofactor_1st =
      calc_cofactor_1st(alpha_result);
  const Eigen::MatrixXd beta_cofactor_1st =
      calc_cofactor_1st(beta_result);

  const double alpha_weight =
      pair_adjoints.hamiltonian_weight * beta_result.overlap_determinant;
  const double beta_weight =
      pair_adjoints.hamiltonian_weight * alpha_result.overlap_determinant;

  accumulate_one_electron_gradient_contribution_local(
      alpha_occ_L,
      alpha_occ_R,
      alpha_cofactor_1st,
      alpha_weight,
      active_one_electron_gradient);
  accumulate_one_electron_gradient_contribution_local(
      beta_occ_L,
      beta_occ_R,
      beta_cofactor_1st,
      beta_weight,
      active_one_electron_gradient);

  if (alpha_result.nullity != 0 || beta_result.nullity != 0) {
    throw std::runtime_error(
        "exact outer-response pair directional requires nullity == 0");
  }
  Eigen::MatrixXd alpha_same_spin_inverse_overlap_gradient =
      Eigen::MatrixXd::Zero(
          static_cast<int>(alpha_occ_L.size()),
          static_cast<int>(alpha_occ_L.size()));
  Eigen::MatrixXd beta_same_spin_inverse_overlap_gradient =
      Eigen::MatrixXd::Zero(
          static_cast<int>(beta_occ_L.size()),
          static_cast<int>(beta_occ_L.size()));
  Eigen::MatrixXd alpha_opposite_spin_inverse_overlap_gradient =
      Eigen::MatrixXd::Zero(
          static_cast<int>(alpha_occ_L.size()),
          static_cast<int>(alpha_occ_L.size()));
  Eigen::MatrixXd beta_opposite_spin_inverse_overlap_gradient =
      Eigen::MatrixXd::Zero(
          static_cast<int>(beta_occ_L.size()),
          static_cast<int>(beta_occ_L.size()));

  double opposite_spin_phi = 0.0;
  const SameSpinPhiResult alpha_phi_result =
      evaluate_same_spin_phi_with_optional_cache_local(
          alpha_occ_L,
          alpha_occ_R,
          active_space_one_electron_result.h1e_act,
          n_active_orbitals,
          active_space_two_electron_result,
          determinant_pair_evaluation.alpha,
          &alpha_same_spin_inverse_overlap_gradient);
  const SameSpinPhiResult beta_phi_result =
      evaluate_same_spin_phi_with_optional_cache_local(
          beta_occ_L,
          beta_occ_R,
          active_space_one_electron_result.h1e_act,
          n_active_orbitals,
          active_space_two_electron_result,
          determinant_pair_evaluation.beta,
          &beta_same_spin_inverse_overlap_gradient);
  if (!skip_opposite_spin &&
      !alpha_occ_L.empty() && !beta_occ_L.empty()) {
    opposite_spin_phi = compute_opposite_spin_original_phi(
        alpha_occ_L,
        alpha_occ_R,
        alpha_result,
        &determinant_pair_evaluation.alpha.opposite_spin_pair_cache,
        beta_occ_L,
        beta_occ_R,
        beta_result,
        &determinant_pair_evaluation.beta.opposite_spin_pair_cache,
        active_space_two_electron_result,
        &alpha_opposite_spin_inverse_overlap_gradient,
        &beta_opposite_spin_inverse_overlap_gradient);
  }
  const Eigen::MatrixXd alpha_inverse_overlap_gradient =
      alpha_same_spin_inverse_overlap_gradient +
      alpha_opposite_spin_inverse_overlap_gradient;
  const Eigen::MatrixXd beta_inverse_overlap_gradient =
      beta_same_spin_inverse_overlap_gradient +
      beta_opposite_spin_inverse_overlap_gradient;
  const double alpha_phi = alpha_phi_result.total_phi;
  const double beta_phi = beta_phi_result.total_phi;
  const double phi_sum = alpha_phi + beta_phi + opposite_spin_phi;
  const double alpha_determinant_weight =
      pair_adjoints.overlap_weight * beta_result.overlap_determinant +
      pair_adjoints.hamiltonian_weight * beta_result.overlap_determinant *
          phi_sum;
  const double beta_determinant_weight =
      pair_adjoints.overlap_weight * alpha_result.overlap_determinant +
      pair_adjoints.hamiltonian_weight * alpha_result.overlap_determinant *
          phi_sum;
  accumulate_spin_overlap_gradient(
      alpha_occ_L,
      alpha_occ_R,
      alpha_result,
      alpha_determinant_weight,
      pair_adjoints.hamiltonian_weight * beta_result.overlap_determinant *
          alpha_inverse_overlap_gradient,
      n_active_orbitals,
      active_orbital_overlap_gradient);
  accumulate_spin_overlap_gradient(
      beta_occ_L,
      beta_occ_R,
      beta_result,
      beta_determinant_weight,
      pair_adjoints.hamiltonian_weight * alpha_result.overlap_determinant *
          beta_inverse_overlap_gradient,
      n_active_orbitals,
      active_orbital_overlap_gradient);

  if (alpha_result.nullity == 0) {
    accumulate_same_spin_two_electron_gradient_contribution_local(
        alpha_occ_L,
        alpha_occ_R,
        alpha_cofactor_1st,
        alpha_result.overlap_determinant,
        alpha_weight,
        packed_active_two_electron_gradient);
  }

  if (beta_result.nullity == 0) {
    accumulate_same_spin_two_electron_gradient_contribution_local(
        beta_occ_L,
        beta_occ_R,
        beta_cofactor_1st,
        beta_result.overlap_determinant,
        beta_weight,
        packed_active_two_electron_gradient);
  }

  if (alpha_result.nullity < 2 && beta_result.nullity < 2 &&
      !skip_opposite_spin &&
      !alpha_occ_L.empty() && !beta_occ_L.empty()) {
    accumulate_opposite_spin_two_electron_gradient_contribution_local(
        alpha_occ_L,
        alpha_occ_R,
        alpha_cofactor_1st,
        &determinant_pair_evaluation.alpha.opposite_spin_pair_cache,
        beta_occ_L,
        beta_occ_R,
        beta_cofactor_1st,
        &determinant_pair_evaluation.beta.opposite_spin_pair_cache,
        pair_adjoints.hamiltonian_weight,
        packed_active_two_electron_gradient);
  }
}

void accumulate_active_space_gradient_pair_local_response_with_adjoints_local(
    const StructurePairAdjoints& pair_adjoints,
    const CppVbInput& input,
    const ActiveSpaceOneElectronResult& active_space_one_electron_result,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const FullDeterminantPairEvaluation& determinant_pair_evaluation,
    int determinant_index_left,
    int determinant_index_right,
    int n_active_orbitals,
    bool skip_opposite_spin,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    Eigen::MatrixXd* active_one_electron_gradient,
    std::vector<double>* active_orbital_overlap_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  // This is the missing `δJ_pair^T λ_pair` term in the exact outer-response:
  // accepted determinant-pair adjoints stay fixed, while the local same-spin /
  // opposite-spin pair kernels respond analytically to `(δS_act, δh_act, δg_act)`.
  if (!has_nonzero_structure_pair_adjoints(pair_adjoints)) {
    return;
  }

  const auto& alpha_occ_L =
      input.structure_data.alpha_det[determinant_index_left];
  const auto& alpha_occ_R =
      input.structure_data.alpha_det[determinant_index_right];
  const auto& beta_occ_L =
      input.structure_data.beta_det[determinant_index_left];
  const auto& beta_occ_R =
      input.structure_data.beta_det[determinant_index_right];

  const RegularSpinDirectionalData alpha_data =
      build_regular_spin_directional_data(
          alpha_occ_L,
          alpha_occ_R,
          active_space_one_electron_result,
          active_space_two_electron_result,
          determinant_pair_evaluation.alpha,
          n_active_orbitals,
          delta_active_orbital_overlap_matrix,
          delta_active_one_electron_matrix,
          delta_packed_active_two_electron_integrals);
  const RegularSpinDirectionalData beta_data =
      build_regular_spin_directional_data(
          beta_occ_L,
          beta_occ_R,
          active_space_one_electron_result,
          active_space_two_electron_result,
          determinant_pair_evaluation.beta,
          n_active_orbitals,
          delta_active_orbital_overlap_matrix,
          delta_active_one_electron_matrix,
          delta_packed_active_two_electron_integrals);

  OppositeSpinDirectionalData opposite_spin_data;
  if (!skip_opposite_spin &&
      !alpha_occ_L.empty() &&
      !beta_occ_L.empty()) {
    opposite_spin_data = build_opposite_spin_directional_data(
        alpha_occ_L,
        alpha_occ_R,
        alpha_data,
        beta_occ_L,
        beta_occ_R,
        beta_data,
        active_space_two_electron_result,
        n_active_orbitals,
        delta_packed_active_two_electron_integrals);
  } else {
    opposite_spin_data.alpha_inverse_overlap_gradient =
        Eigen::MatrixXd::Zero(
            static_cast<int>(alpha_occ_L.size()),
            static_cast<int>(alpha_occ_L.size()));
    opposite_spin_data.delta_alpha_inverse_overlap_gradient =
        Eigen::MatrixXd::Zero(
            static_cast<int>(alpha_occ_L.size()),
            static_cast<int>(alpha_occ_L.size()));
    opposite_spin_data.beta_inverse_overlap_gradient =
        Eigen::MatrixXd::Zero(
            static_cast<int>(beta_occ_L.size()),
            static_cast<int>(beta_occ_L.size()));
    opposite_spin_data.delta_beta_inverse_overlap_gradient =
        Eigen::MatrixXd::Zero(
            static_cast<int>(beta_occ_L.size()),
            static_cast<int>(beta_occ_L.size()));
  }

  const double alpha_weight =
      pair_adjoints.hamiltonian_weight * beta_data.overlap_determinant;
  const double beta_weight =
      pair_adjoints.hamiltonian_weight * alpha_data.overlap_determinant;
  const double delta_alpha_weight =
      pair_adjoints.hamiltonian_weight * beta_data.delta_overlap_determinant;
  const double delta_beta_weight =
      pair_adjoints.hamiltonian_weight * alpha_data.delta_overlap_determinant;

  accumulate_directional_one_electron_gradient_contribution_local(
      alpha_occ_L,
      alpha_occ_R,
      alpha_data.cofactor_1st,
      alpha_data.delta_cofactor_1st,
      alpha_weight,
      delta_alpha_weight,
      active_one_electron_gradient);
  accumulate_directional_one_electron_gradient_contribution_local(
      beta_occ_L,
      beta_occ_R,
      beta_data.cofactor_1st,
      beta_data.delta_cofactor_1st,
      beta_weight,
      delta_beta_weight,
      active_one_electron_gradient);

  const Eigen::MatrixXd alpha_inverse_overlap_gradient =
      alpha_data.same_spin_inverse_overlap_gradient +
      opposite_spin_data.alpha_inverse_overlap_gradient;
  const Eigen::MatrixXd beta_inverse_overlap_gradient =
      beta_data.same_spin_inverse_overlap_gradient +
      opposite_spin_data.beta_inverse_overlap_gradient;
  const Eigen::MatrixXd delta_alpha_inverse_overlap_gradient =
      alpha_data.delta_same_spin_inverse_overlap_gradient +
      opposite_spin_data.delta_alpha_inverse_overlap_gradient;
  const Eigen::MatrixXd delta_beta_inverse_overlap_gradient =
      beta_data.delta_same_spin_inverse_overlap_gradient +
      opposite_spin_data.delta_beta_inverse_overlap_gradient;
  const double phi_sum =
      alpha_data.same_spin_total_phi +
      beta_data.same_spin_total_phi +
      opposite_spin_data.phi;
  const double delta_phi_sum =
      alpha_data.delta_same_spin_total_phi +
      beta_data.delta_same_spin_total_phi +
      opposite_spin_data.delta_phi;
  const double alpha_determinant_weight =
      pair_adjoints.overlap_weight * beta_data.overlap_determinant +
      pair_adjoints.hamiltonian_weight * beta_data.overlap_determinant * phi_sum;
  const double beta_determinant_weight =
      pair_adjoints.overlap_weight * alpha_data.overlap_determinant +
      pair_adjoints.hamiltonian_weight * alpha_data.overlap_determinant * phi_sum;
  const double delta_alpha_determinant_weight =
      pair_adjoints.overlap_weight * beta_data.delta_overlap_determinant +
      pair_adjoints.hamiltonian_weight *
          (beta_data.delta_overlap_determinant * phi_sum +
           beta_data.overlap_determinant * delta_phi_sum);
  const double delta_beta_determinant_weight =
      pair_adjoints.overlap_weight * alpha_data.delta_overlap_determinant +
      pair_adjoints.hamiltonian_weight *
          (alpha_data.delta_overlap_determinant * phi_sum +
           alpha_data.overlap_determinant * delta_phi_sum);
  const Eigen::MatrixXd alpha_overlap_pullback =
      pair_adjoints.hamiltonian_weight *
      beta_data.overlap_determinant *
      alpha_inverse_overlap_gradient;
  const Eigen::MatrixXd beta_overlap_pullback =
      pair_adjoints.hamiltonian_weight *
      alpha_data.overlap_determinant *
      beta_inverse_overlap_gradient;
  const Eigen::MatrixXd delta_alpha_overlap_pullback =
      pair_adjoints.hamiltonian_weight *
      (beta_data.delta_overlap_determinant * alpha_inverse_overlap_gradient +
       beta_data.overlap_determinant * delta_alpha_inverse_overlap_gradient);
  const Eigen::MatrixXd delta_beta_overlap_pullback =
      pair_adjoints.hamiltonian_weight *
      (alpha_data.delta_overlap_determinant * beta_inverse_overlap_gradient +
       alpha_data.overlap_determinant * delta_beta_inverse_overlap_gradient);

  accumulate_regular_spin_overlap_gradient_direction_local(
      alpha_occ_L,
      alpha_occ_R,
      alpha_data.overlap_determinant,
      alpha_data.delta_overlap_determinant,
      alpha_data.inverse_overlap_submatrix,
      alpha_data.delta_inverse_overlap_submatrix,
      alpha_determinant_weight,
      delta_alpha_determinant_weight,
      alpha_overlap_pullback,
      delta_alpha_overlap_pullback,
      n_active_orbitals,
      active_orbital_overlap_gradient);
  accumulate_regular_spin_overlap_gradient_direction_local(
      beta_occ_L,
      beta_occ_R,
      beta_data.overlap_determinant,
      beta_data.delta_overlap_determinant,
      beta_data.inverse_overlap_submatrix,
      beta_data.delta_inverse_overlap_submatrix,
      beta_determinant_weight,
      delta_beta_determinant_weight,
      beta_overlap_pullback,
      delta_beta_overlap_pullback,
      n_active_orbitals,
      active_orbital_overlap_gradient);

  accumulate_directional_same_spin_two_electron_gradient_contribution_local(
      alpha_occ_L,
      alpha_occ_R,
      alpha_data.cofactor_1st,
      alpha_data.delta_cofactor_1st,
      alpha_data.overlap_determinant,
      alpha_data.delta_overlap_determinant,
      alpha_weight,
      delta_alpha_weight,
      packed_active_two_electron_gradient);
  accumulate_directional_same_spin_two_electron_gradient_contribution_local(
      beta_occ_L,
      beta_occ_R,
      beta_data.cofactor_1st,
      beta_data.delta_cofactor_1st,
      beta_data.overlap_determinant,
      beta_data.delta_overlap_determinant,
      beta_weight,
      delta_beta_weight,
      packed_active_two_electron_gradient);

  if (!skip_opposite_spin &&
      !alpha_occ_L.empty() &&
      !beta_occ_L.empty()) {
    accumulate_directional_opposite_spin_two_electron_gradient_contribution_local(
        alpha_occ_L,
        alpha_occ_R,
        alpha_data.cofactor_1st,
        alpha_data.delta_cofactor_1st,
        beta_occ_L,
        beta_occ_R,
        beta_data.cofactor_1st,
        beta_data.delta_cofactor_1st,
        pair_adjoints.hamiltonian_weight,
        packed_active_two_electron_gradient);
  }
}

void accumulate_active_space_gradient_pair_opposite_spin_local_response_with_adjoints_local(
    const StructurePairAdjoints& pair_adjoints,
    const CppVbInput& input,
    const ActiveSpaceOneElectronResult& active_space_one_electron_result,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const FullDeterminantPairEvaluation& determinant_pair_evaluation,
    int determinant_index_left,
    int determinant_index_right,
    int n_active_orbitals,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    std::vector<double>* active_orbital_overlap_gradient,
    std::vector<double>* packed_active_two_electron_gradient) {
  if (pair_adjoints.hamiltonian_weight == 0.0) {
    return;
  }

  const auto& alpha_occ_L =
      input.structure_data.alpha_det[determinant_index_left];
  const auto& alpha_occ_R =
      input.structure_data.alpha_det[determinant_index_right];
  const auto& beta_occ_L =
      input.structure_data.beta_det[determinant_index_left];
  const auto& beta_occ_R =
      input.structure_data.beta_det[determinant_index_right];
  if (alpha_occ_L.empty() || beta_occ_L.empty()) {
    return;
  }

  const RegularSpinDirectionalData alpha_data =
      build_regular_spin_directional_data(
          alpha_occ_L,
          alpha_occ_R,
          active_space_one_electron_result,
          active_space_two_electron_result,
          determinant_pair_evaluation.alpha,
          n_active_orbitals,
          delta_active_orbital_overlap_matrix,
          delta_active_one_electron_matrix,
          delta_packed_active_two_electron_integrals);
  const RegularSpinDirectionalData beta_data =
      build_regular_spin_directional_data(
          beta_occ_L,
          beta_occ_R,
          active_space_one_electron_result,
          active_space_two_electron_result,
          determinant_pair_evaluation.beta,
          n_active_orbitals,
          delta_active_orbital_overlap_matrix,
          delta_active_one_electron_matrix,
          delta_packed_active_two_electron_integrals);
  const OppositeSpinDirectionalData opposite_spin_data =
      build_opposite_spin_directional_data(
          alpha_occ_L,
          alpha_occ_R,
          alpha_data,
          beta_occ_L,
          beta_occ_R,
          beta_data,
          active_space_two_electron_result,
          n_active_orbitals,
          delta_packed_active_two_electron_integrals);

  const double alpha_determinant_weight =
      pair_adjoints.hamiltonian_weight *
      beta_data.overlap_determinant *
      opposite_spin_data.phi;
  const double beta_determinant_weight =
      pair_adjoints.hamiltonian_weight *
      alpha_data.overlap_determinant *
      opposite_spin_data.phi;
  const double delta_alpha_determinant_weight =
      pair_adjoints.hamiltonian_weight *
      beta_data.overlap_determinant *
      opposite_spin_data.delta_phi;
  const double delta_beta_determinant_weight =
      pair_adjoints.hamiltonian_weight *
      alpha_data.overlap_determinant *
      opposite_spin_data.delta_phi;
  const Eigen::MatrixXd alpha_overlap_pullback =
      pair_adjoints.hamiltonian_weight *
      beta_data.overlap_determinant *
      opposite_spin_data.alpha_inverse_overlap_gradient;
  const Eigen::MatrixXd beta_overlap_pullback =
      pair_adjoints.hamiltonian_weight *
      alpha_data.overlap_determinant *
      opposite_spin_data.beta_inverse_overlap_gradient;
  const Eigen::MatrixXd delta_alpha_overlap_pullback =
      pair_adjoints.hamiltonian_weight *
      beta_data.overlap_determinant *
      opposite_spin_data.delta_alpha_inverse_overlap_gradient;
  const Eigen::MatrixXd delta_beta_overlap_pullback =
      pair_adjoints.hamiltonian_weight *
      alpha_data.overlap_determinant *
      opposite_spin_data.delta_beta_inverse_overlap_gradient;

  accumulate_regular_spin_overlap_gradient_direction_local(
      alpha_occ_L,
      alpha_occ_R,
      alpha_data.overlap_determinant,
      alpha_data.delta_overlap_determinant,
      alpha_data.inverse_overlap_submatrix,
      alpha_data.delta_inverse_overlap_submatrix,
      alpha_determinant_weight,
      delta_alpha_determinant_weight,
      alpha_overlap_pullback,
      delta_alpha_overlap_pullback,
      n_active_orbitals,
      active_orbital_overlap_gradient);
  accumulate_regular_spin_overlap_gradient_direction_local(
      beta_occ_L,
      beta_occ_R,
      beta_data.overlap_determinant,
      beta_data.delta_overlap_determinant,
      beta_data.inverse_overlap_submatrix,
      beta_data.delta_inverse_overlap_submatrix,
      beta_determinant_weight,
      delta_beta_determinant_weight,
      beta_overlap_pullback,
      delta_beta_overlap_pullback,
      n_active_orbitals,
      active_orbital_overlap_gradient);
  accumulate_directional_opposite_spin_two_electron_gradient_contribution_local(
      alpha_occ_L,
      alpha_occ_R,
      alpha_data.cofactor_1st,
      alpha_data.delta_cofactor_1st,
      beta_occ_L,
      beta_occ_R,
      beta_data.cofactor_1st,
      beta_data.delta_cofactor_1st,
      pair_adjoints.hamiltonian_weight,
      packed_active_two_electron_gradient);
}

FullDeterminantPairEvaluation evaluate_active_space_determinant_pair_local(
    const SameSpinPairCacheContext* same_spin_pair_cache,
    const FullDeterminantPairEvaluator& pair_evaluator,
    const CppVbInput& input,
    const std::vector<double>& active_orbital_overlap_matrix,
    const ActiveSpaceOneElectronResult& active_space_one_electron_result,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    int determinant_index_left,
    int determinant_index_right,
    int n_active_orbitals) {
  return evaluate_full_determinant_pair_with_optional_same_spin_cache(
      same_spin_pair_cache,
      pair_evaluator,
      input.structure_data.alpha_det,
      input.structure_data.beta_det,
      determinant_index_left,
      determinant_index_right,
      active_orbital_overlap_matrix,
      active_space_one_electron_result.h1e_act,
      n_active_orbitals,
      active_space_two_electron_result);
}

double contract_active_space_direction(
    const DeterminantPairDirectionalScratch& scratch,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  double directional_scalar = 0.0;
  for (std::size_t index = 0;
       index < delta_active_orbital_overlap_matrix.size();
       ++index) {
    directional_scalar +=
        scratch.active_orbital_overlap_gradient[index] *
        delta_active_orbital_overlap_matrix[index];
  }
  const double* one_electron_gradient_data =
      scratch.active_one_electron_gradient.data();
  for (std::size_t index = 0;
       index < delta_active_one_electron_matrix.size();
       ++index) {
    directional_scalar +=
        one_electron_gradient_data[index] *
        delta_active_one_electron_matrix[index];
  }
  for (std::size_t index = 0;
       index < delta_packed_active_two_electron_integrals.size();
       ++index) {
    directional_scalar +=
        scratch.packed_active_two_electron_gradient[index] *
        delta_packed_active_two_electron_integrals[index];
  }
  return directional_scalar;
}

double compute_determinant_pair_directional_scalar(
    const StructurePairAdjoints& pair_adjoints,
    const CppVbInput& input,
    const ActiveSpaceOneElectronResult& active_space_one_electron_result,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const FullDeterminantPairEvaluation& determinant_pair_evaluation,
    int determinant_index_left,
    int determinant_index_right,
    int n_active_orbitals,
    bool skip_opposite_spin,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    DeterminantPairDirectionalScratch* scratch) {
  scratch->set_zero();
  accumulate_active_space_gradient_pair_with_adjoints_local(
      pair_adjoints,
      input,
      active_space_one_electron_result,
      active_space_two_electron_result,
      determinant_pair_evaluation,
      determinant_index_left,
      determinant_index_right,
      n_active_orbitals,
      skip_opposite_spin,
      &scratch->active_one_electron_gradient,
      &scratch->active_orbital_overlap_gradient,
      &scratch->packed_active_two_electron_gradient);
  return contract_active_space_direction(
      *scratch,
      delta_active_orbital_overlap_matrix,
      delta_active_one_electron_matrix,
      delta_packed_active_two_electron_integrals);
}

StructureAccumulationResult build_directional_structure_matrices(
    const CppVbInput& input,
    const CppActiveSpaceSecondOrderContext& accepted_point_context,
    const std::vector<StructureCoefficientBlock>& coefficient_blocks,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  const auto& same_spin_pair_cache = accepted_point_context.same_spin_pair_cache;
  const int n_determinants =
      static_cast<int>(input.structure_data.alpha_det.size());
  const bool has_valid_matrix_form_cache =
      same_spin_pair_cache.enabled() &&
      static_cast<int>(
          same_spin_pair_cache.alpha_reuse_table.determinant_to_unique_id.size()) ==
          n_determinants &&
      static_cast<int>(
          same_spin_pair_cache.beta_reuse_table.determinant_to_unique_id.size()) ==
          n_determinants &&
      !same_spin_pair_cache.alpha_reuse_table.unique_determinants.empty() &&
      !same_spin_pair_cache.beta_reuse_table.unique_determinants.empty();
  if (!has_valid_matrix_form_cache) {
    throw std::runtime_error(
        "exact outer-response directional structure builder requires the "
        "unique-spin tiled matrix path");
  }
  if (static_cast<int>(coefficient_blocks.size()) !=
      input.structure_data.n_structures) {
    throw std::invalid_argument(
        "directional structure coefficient blocks do not match n_structures");
  }

  return build_tiled_directional_structure_matrices(
      input,
      accepted_point_context,
      coefficient_blocks,
      delta_active_orbital_overlap_matrix,
      delta_active_one_electron_matrix,
      delta_packed_active_two_electron_integrals);
}

SelectedStateProjectedDirectionalMatrices
build_selected_state_projected_directional_structure_matrices(
    const CppVbInput& input,
    const CppActiveSpaceSecondOrderContext& accepted_point_context,
    const std::vector<StructureCoefficientBlock>& coefficient_blocks,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  const int n_determinants =
      static_cast<int>(input.structure_data.alpha_det.size());
  const int n_structures = input.structure_data.n_structures;
  const int n_active_orbitals =
      input.orbital_preparation_input.n_active_orbitals;
  const int n_selected_states =
      static_cast<int>(accepted_point_context.selected_state_indices.size());
  if (n_determinants <= 0 || n_structures <= 0 || n_active_orbitals <= 0 ||
      n_selected_states <= 0) {
    throw std::invalid_argument(
        "projected directional structure response requires positive dimensions");
  }

  const auto& same_spin_pair_cache = accepted_point_context.same_spin_pair_cache;
  if (!same_spin_pair_cache.enabled()) {
    throw std::invalid_argument(
        "projected directional structure response requires same-spin cache");
  }
  const std::size_t expected_eigenvector_size =
      n_structures * n_structures;
  if (accepted_point_context.eigen_result.eigenvector_matrix.size() !=
      expected_eigenvector_size) {
    throw std::invalid_argument(
        "accepted-point generalized eigensystem dimensions are inconsistent");
  }

  const Eigen::Map<const Eigen::MatrixXd> eigenvector_matrix(
      accepted_point_context.eigen_result.eigenvector_matrix.data(),
      n_structures,
      n_structures);
  const auto& selected_state_indices =
      accepted_point_context.selected_state_indices;
  const bool single_selected_state = n_selected_states == 1;
  const int selected_state_index =
      single_selected_state ? selected_state_indices.front() : -1;
  for (int selected_state_offset = 0;
       selected_state_offset < n_selected_states;
       ++selected_state_offset) {
    const int state_index =
        selected_state_indices[selected_state_offset];
    if (state_index < 0 || state_index >= n_structures) {
      throw std::out_of_range("selected state index is out of range");
    }
  }

  if (static_cast<int>(coefficient_blocks.size()) != n_structures) {
    throw std::invalid_argument(
        "projected directional structure coefficient blocks do not match n_structures");
  }
  const bool shared_same_spin_pair_cache =
      same_spin_pair_cache.shares_same_spin_pair_cache_between_spins();
  const bool close_shell_same_spin =
      same_spin_pair_cache.close_shell_reuses_same_spin_pair_cache();
  const int tile_size = directional_structure_matrix_tile_size();
  const int max_cached_tiles = directional_structure_matrix_tile_cache_tiles();
  const int n_packed_active_pairs =
      packed_active_pair_count(n_active_orbitals);
  const auto& accepted_prepared_active_space =
      accepted_point_context.prepared_active_space;
  const ActiveSpaceTwoElectronView two_electron_view =
      make_active_space_two_electron_view(
          accepted_prepared_active_space.active_space_two_electron_result);

  SelectedStateProjectedDirectionalMatrices result;
  result.transformed_delta_hamiltonian_selected =
      Eigen::MatrixXd::Zero(n_structures, n_selected_states);
  result.transformed_delta_overlap_selected =
      Eigen::MatrixXd::Zero(n_structures, n_selected_states);
  std::exception_ptr parallel_exception;
  std::atomic<bool> parallel_failed(false);
  const int n_parallel_threads =
      choose_exact_ctx_directional_structure_threads(n_structures);
  const bool log_structure_stage = exact_ctx_structure_stage_logging_enabled();
  std::vector<Eigen::MatrixXd> thread_transformed_delta_hamiltonian_selected(
      std::max(1, n_parallel_threads),
      Eigen::MatrixXd::Zero(n_structures, n_selected_states));
  std::vector<Eigen::MatrixXd> thread_transformed_delta_overlap_selected(
      std::max(1, n_parallel_threads),
      Eigen::MatrixXd::Zero(n_structures, n_selected_states));
  std::vector<double> thread_gather_wall_seconds(
      std::max(1, n_parallel_threads),
      0.0);
  std::vector<double> thread_pair_kernel_wall_seconds(
      std::max(1, n_parallel_threads),
      0.0);
  std::vector<double> thread_projection_wall_seconds(
      std::max(1, n_parallel_threads),
      0.0);
  std::vector<double> thread_alpha_gather_wall_seconds(
      std::max(1, n_parallel_threads),
      0.0);
  std::vector<double> thread_beta_gather_wall_seconds(
      std::max(1, n_parallel_threads),
      0.0);
  std::vector<double> thread_same_spin_wall_seconds(
      std::max(1, n_parallel_threads),
      0.0);
  std::vector<double> thread_opposite_spin_wall_seconds(
      std::max(1, n_parallel_threads),
      0.0);
  std::vector<std::size_t> thread_structure_pair_count(
      std::max(1, n_parallel_threads),
      0);
  std::vector<std::size_t> thread_diagonal_structure_pair_count(
      std::max(1, n_parallel_threads),
      0);

// The projected outer-response builder fuses the old
//   structure-pair scalar accumulation + U^T (delta M) U_sel
// pipeline into one block contraction. The tile sweeps over determinant
// supports stay unchanged, but the H/S outputs now live directly in the
// minimal `(all_states, selected_states)` transformed basis required by the
// selected-state directional response.
#pragma omp parallel if(n_parallel_threads > 1 && n_structures > 2) num_threads(n_parallel_threads)
  {
    int thread_index = 0;
#ifdef _OPENMP
    thread_index = omp_get_thread_num();
#endif
    DirectionalSpinPairTileProvider thread_alpha_provider(
        same_spin_pair_cache.alpha_reuse_table.unique_determinants,
        same_spin_pair_cache.alpha_pair_cache_ref(),
        accepted_prepared_active_space.active_space_one_electron_result,
        accepted_prepared_active_space.active_space_two_electron_result,
        n_active_orbitals,
        delta_active_orbital_overlap_matrix,
        delta_active_one_electron_matrix,
        delta_packed_active_two_electron_integrals,
        tile_size,
        max_cached_tiles);
    DirectionalSpinPairTileProvider thread_beta_provider(
        same_spin_pair_cache.beta_reuse_table.unique_determinants,
        same_spin_pair_cache.beta_pair_cache_ref(),
        accepted_prepared_active_space.active_space_one_electron_result,
        accepted_prepared_active_space.active_space_two_electron_result,
        n_active_orbitals,
        delta_active_orbital_overlap_matrix,
        delta_active_one_electron_matrix,
        delta_packed_active_two_electron_integrals,
        tile_size,
        max_cached_tiles);
    Eigen::MatrixXd alpha_overlap_subblock;
    Eigen::MatrixXd alpha_total_subblock;
    Eigen::MatrixXd alpha_delta_overlap_subblock;
    Eigen::MatrixXd alpha_delta_total_subblock;
    Eigen::MatrixXd beta_overlap_subblock;
    Eigen::MatrixXd beta_total_subblock;
    Eigen::MatrixXd beta_delta_overlap_subblock;
    Eigen::MatrixXd beta_delta_total_subblock;
    Eigen::MatrixXd beta_projected_channel_block;
    Eigen::MatrixXd beta_push;
    Eigen::MatrixXd image;
    DirectionalSameSpinContractionScratchLocal same_spin_scratch;
    LocalProjectionBlockLocal accepted_alpha_projection_block;
    LocalProjectionBlockLocal directional_alpha_projection_block;
    LocalProjectionBlockLocal accepted_beta_projection_block;
    LocalProjectionBlockLocal directional_beta_projection_block;
    LocalOppositeSpinChannelFamilyBuilderLocal accepted_alpha_channel_builder(
        n_packed_active_pairs);
    LocalOppositeSpinChannelFamilyBuilderLocal directional_alpha_channel_builder(
        n_packed_active_pairs);
    LocalOppositeSpinChannelFamilyLocal accepted_alpha_channels;
    LocalOppositeSpinChannelFamilyLocal directional_alpha_channels;
    Eigen::VectorXd directional_overlap_column =
        Eigen::VectorXd::Zero(n_structures);
    Eigen::VectorXd directional_hamiltonian_column =
        Eigen::VectorXd::Zero(n_structures);
    Eigen::VectorXd transformed_left_hamiltonian =
        Eigen::VectorXd::Zero(n_structures);
    Eigen::VectorXd transformed_left_overlap =
        Eigen::VectorXd::Zero(n_structures);
    Eigen::MatrixXd& local_transformed_delta_hamiltonian_selected =
        thread_transformed_delta_hamiltonian_selected[thread_index];
    Eigen::MatrixXd& local_transformed_delta_overlap_selected =
        thread_transformed_delta_overlap_selected[thread_index];
    double local_gather_wall_seconds = 0.0;
    double local_pair_kernel_wall_seconds = 0.0;
    double local_projection_wall_seconds = 0.0;
    double local_alpha_gather_wall_seconds = 0.0;
    double local_beta_gather_wall_seconds = 0.0;
    double local_same_spin_wall_seconds = 0.0;
    double local_opposite_spin_wall_seconds = 0.0;
    std::size_t local_structure_pair_count = 0;
    std::size_t local_diagonal_structure_pair_count = 0;

// The projected outer-response reduction used to add each thread-local matrix
// through one OpenMP critical section. That made the exact_ctx HVP depend on
// thread arrival order and reproduced the user's 5-step / 16-step / 32-step
// trajectory drift on the same 32-thread input. Keep the per-thread work local
// here, but assign `right_structure` deterministically and reduce the thread
// buffers in a fixed order after the parallel region.
#pragma omp for schedule(static, 1)
    for (int right_structure = 0;
         right_structure < n_structures;
         ++right_structure) {
      if (parallel_failed.load(std::memory_order_relaxed)) {
        continue;
      }
      const auto& right_block =
          coefficient_blocks[right_structure];
      try {
        directional_overlap_column.head(right_structure + 1).setZero();
        directional_hamiltonian_column.head(right_structure + 1).setZero();
        for (int left_structure = 0;
             left_structure <= right_structure;
             ++left_structure) {
          if (parallel_failed.load(std::memory_order_relaxed)) {
            continue;
          }
          const auto& left_block =
              coefficient_blocks[left_structure];
          if (left_block.local_coefficients.size() == 0 ||
              right_block.local_coefficients.size() == 0) {
            directional_overlap_column[left_structure] = 0.0;
            directional_hamiltonian_column[left_structure] = 0.0;
            continue;
          }

          const auto gather_start_time =
              log_structure_stage
                  ? std::chrono::steady_clock::now()
                  : std::chrono::steady_clock::time_point();
          const bool structure_pair_close_shell_diagonal =
              close_shell_same_spin &&
              left_block.close_shell_diagonal &&
              right_block.close_shell_diagonal;
          ++local_structure_pair_count;
          if (structure_pair_close_shell_diagonal) {
            ++local_diagonal_structure_pair_count;
          }
          if (structure_pair_close_shell_diagonal) {
            const auto alpha_gather_start_time =
                log_structure_stage
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point();
            gather_directional_spin_block_local(
                &thread_alpha_provider,
                left_block.alpha_support,
                right_block.alpha_support,
                &alpha_overlap_subblock,
                &alpha_total_subblock,
                &alpha_delta_overlap_subblock,
                &alpha_delta_total_subblock,
                &accepted_alpha_projection_block,
                &directional_alpha_projection_block,
                &accepted_alpha_channel_builder,
                &accepted_alpha_channels,
                &directional_alpha_channel_builder,
                &directional_alpha_channels);
            if (log_structure_stage) {
              local_alpha_gather_wall_seconds +=
                  elapsed_wall_time_seconds(alpha_gather_start_time);
            }
          } else {
            const auto alpha_gather_start_time =
                log_structure_stage
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point();
            gather_directional_spin_block_local(
                &thread_alpha_provider,
                left_block.alpha_support,
                right_block.alpha_support,
                &alpha_overlap_subblock,
                &alpha_total_subblock,
                &alpha_delta_overlap_subblock,
                &alpha_delta_total_subblock,
                &accepted_alpha_projection_block,
                &directional_alpha_projection_block,
                &accepted_alpha_channel_builder,
                &accepted_alpha_channels,
                &directional_alpha_channel_builder,
                &directional_alpha_channels);
            if (log_structure_stage) {
              local_alpha_gather_wall_seconds +=
                  elapsed_wall_time_seconds(alpha_gather_start_time);
            }
            const auto beta_gather_start_time =
                log_structure_stage
                    ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point();
            gather_directional_spin_block_local(
                shared_same_spin_pair_cache
                    ? &thread_alpha_provider
                    : &thread_beta_provider,
                left_block.beta_support,
                right_block.beta_support,
                &beta_overlap_subblock,
                &beta_total_subblock,
                &beta_delta_overlap_subblock,
                &beta_delta_total_subblock,
                &accepted_beta_projection_block,
                &directional_beta_projection_block,
                nullptr,
                nullptr,
                nullptr,
                nullptr);
            if (log_structure_stage) {
              local_beta_gather_wall_seconds +=
                  elapsed_wall_time_seconds(beta_gather_start_time);
            }
          }
          if (log_structure_stage) {
            local_gather_wall_seconds +=
                elapsed_wall_time_seconds(gather_start_time);
          }

          const auto pair_kernel_start_time =
              log_structure_stage
                  ? std::chrono::steady_clock::now()
                  : std::chrono::steady_clock::time_point();
          const auto same_spin_contraction =
              structure_pair_close_shell_diagonal
                  ? [&]() {
                      const auto same_spin_start_time =
                          log_structure_stage
                              ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point();
                      const auto contraction =
                          contract_close_shell_diagonal_directional_same_spin_structure_kernels_local(
                              left_block.local_diagonal_coefficients,
                              right_block.local_diagonal_coefficients,
                              alpha_overlap_subblock,
                              alpha_total_subblock,
                              alpha_delta_overlap_subblock,
                              alpha_delta_total_subblock);
                      if (log_structure_stage) {
                        local_same_spin_wall_seconds +=
                            elapsed_wall_time_seconds(same_spin_start_time);
                      }
                      return contraction;
                    }()
                  : [&]() {
                      const auto same_spin_start_time =
                          log_structure_stage
                              ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point();
                      const auto contraction =
                          contract_directional_same_spin_structure_kernels_local(
                              left_block.local_coefficients,
                              right_block.local_coefficients,
                              alpha_overlap_subblock,
                              alpha_total_subblock,
                              alpha_delta_overlap_subblock,
                              alpha_delta_total_subblock,
                              beta_overlap_subblock,
                              beta_total_subblock,
                              beta_delta_overlap_subblock,
                              beta_delta_total_subblock,
                              &same_spin_scratch);
                      if (log_structure_stage) {
                        local_same_spin_wall_seconds +=
                            elapsed_wall_time_seconds(same_spin_start_time);
                      }
                      return contraction;
                    }();
          const double directional_overlap =
              same_spin_contraction.overlap;
          double directional_hamiltonian =
              same_spin_contraction.hamiltonian;
          directional_hamiltonian +=
              structure_pair_close_shell_diagonal
                  ? [&]() {
                      const auto opposite_spin_start_time =
                          log_structure_stage
                              ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point();
                      const double contraction =
                          contract_close_shell_diagonal_local_directional_opposite_spin_block_local(
                              left_block.local_diagonal_coefficients,
                              right_block.local_diagonal_coefficients,
                              accepted_alpha_channels,
                              directional_alpha_channels,
                              accepted_alpha_projection_block,
                              directional_alpha_projection_block,
                              two_electron_view,
                              n_active_orbitals,
                              delta_packed_active_two_electron_integrals,
                              &beta_projected_channel_block);
                      if (log_structure_stage) {
                        local_opposite_spin_wall_seconds +=
                            elapsed_wall_time_seconds(opposite_spin_start_time);
                      }
                      return contraction;
                    }()
                  : [&]() {
                      const auto opposite_spin_start_time =
                          log_structure_stage
                              ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point();
                      const double contraction =
                          contract_local_directional_opposite_spin_block_local(
                              left_block.local_coefficients,
                              right_block.local_coefficients,
                              accepted_alpha_projection_block,
                              directional_alpha_projection_block,
                              accepted_alpha_channels,
                              directional_alpha_channels,
                              accepted_beta_projection_block,
                              directional_beta_projection_block,
                              two_electron_view,
                              n_active_orbitals,
                              delta_packed_active_two_electron_integrals,
                              &beta_projected_channel_block,
                              &beta_push,
                              &image);
                      if (log_structure_stage) {
                        local_opposite_spin_wall_seconds +=
                            elapsed_wall_time_seconds(opposite_spin_start_time);
                      }
                      return contraction;
                    }();
          if (log_structure_stage) {
            local_pair_kernel_wall_seconds +=
                elapsed_wall_time_seconds(pair_kernel_start_time);
          }

          directional_overlap_column[left_structure] = directional_overlap;
          directional_hamiltonian_column[left_structure] =
              directional_hamiltonian;
        }

        const auto projection_start_time =
            log_structure_stage
                ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point();
        const auto directional_hamiltonian_column_head =
            directional_hamiltonian_column.head(right_structure + 1);
        const auto directional_overlap_column_head =
            directional_overlap_column.head(right_structure + 1);
        transformed_left_hamiltonian.noalias() =
            eigenvector_matrix.topRows(right_structure + 1).transpose() *
            directional_hamiltonian_column_head;
        transformed_left_overlap.noalias() =
            eigenvector_matrix.topRows(right_structure + 1).transpose() *
            directional_overlap_column_head;
        const auto eigenvector_row = eigenvector_matrix.row(right_structure);
        if (single_selected_state) {
          const double selected_right_value =
              eigenvector_row[selected_state_index];
          local_transformed_delta_hamiltonian_selected.col(0).noalias() +=
              selected_right_value * transformed_left_hamiltonian;
          local_transformed_delta_overlap_selected.col(0).noalias() +=
              selected_right_value * transformed_left_overlap;
        } else {
          for (int selected_state_offset = 0;
               selected_state_offset < n_selected_states;
               ++selected_state_offset) {
            const int state_index =
                selected_state_indices[selected_state_offset];
            const double selected_right_value =
                eigenvector_row[state_index];
            local_transformed_delta_hamiltonian_selected
                .col(selected_state_offset)
                .noalias() +=
                selected_right_value * transformed_left_hamiltonian;
            local_transformed_delta_overlap_selected
                .col(selected_state_offset)
                .noalias() +=
                selected_right_value * transformed_left_overlap;
          }
        }

        if (right_structure > 0) {
          const double directional_hamiltonian_diagonal =
              directional_hamiltonian_column[right_structure];
          const double directional_overlap_diagonal =
              directional_overlap_column[right_structure];
          const auto eigenvector_right_column =
              eigenvector_matrix.row(right_structure).transpose();
          if (single_selected_state) {
            const double transformed_selected_hamiltonian_head =
                transformed_left_hamiltonian[selected_state_index] -
                directional_hamiltonian_diagonal *
                    eigenvector_row[selected_state_index];
            const double transformed_selected_overlap_head =
                transformed_left_overlap[selected_state_index] -
                directional_overlap_diagonal *
                    eigenvector_row[selected_state_index];
            local_transformed_delta_hamiltonian_selected.col(0).noalias() +=
                transformed_selected_hamiltonian_head *
                eigenvector_right_column;
            local_transformed_delta_overlap_selected.col(0).noalias() +=
                transformed_selected_overlap_head *
                eigenvector_right_column;
          } else {
            for (int selected_state_offset = 0;
                 selected_state_offset < n_selected_states;
                 ++selected_state_offset) {
              const int state_index =
                  selected_state_indices[selected_state_offset];
              const double transformed_selected_hamiltonian_head =
                  transformed_left_hamiltonian[state_index] -
                  directional_hamiltonian_diagonal *
                      eigenvector_row[state_index];
              const double transformed_selected_overlap_head =
                  transformed_left_overlap[state_index] -
                  directional_overlap_diagonal *
                      eigenvector_row[state_index];
              local_transformed_delta_hamiltonian_selected
                  .col(selected_state_offset)
                  .noalias() +=
                  transformed_selected_hamiltonian_head *
                  eigenvector_right_column;
              local_transformed_delta_overlap_selected
                  .col(selected_state_offset)
                  .noalias() +=
                  transformed_selected_overlap_head *
                  eigenvector_right_column;
            }
          }
        }
        if (log_structure_stage) {
          local_projection_wall_seconds +=
              elapsed_wall_time_seconds(projection_start_time);
        }
      } catch (...) {
        parallel_failed.store(true, std::memory_order_relaxed);
#pragma omp critical(exact_ctx_projected_directional_structure_exception)
        {
          if (!parallel_exception) {
            parallel_exception = std::current_exception();
          }
        }
      }
    }

    thread_gather_wall_seconds[thread_index] =
        local_gather_wall_seconds;
    thread_pair_kernel_wall_seconds[thread_index] =
        local_pair_kernel_wall_seconds;
    thread_projection_wall_seconds[thread_index] =
        local_projection_wall_seconds;
    thread_alpha_gather_wall_seconds[thread_index] =
        local_alpha_gather_wall_seconds;
    thread_beta_gather_wall_seconds[thread_index] =
        local_beta_gather_wall_seconds;
    thread_same_spin_wall_seconds[thread_index] =
        local_same_spin_wall_seconds;
    thread_opposite_spin_wall_seconds[thread_index] =
        local_opposite_spin_wall_seconds;
    thread_structure_pair_count[thread_index] =
        local_structure_pair_count;
    thread_diagonal_structure_pair_count[thread_index] =
        local_diagonal_structure_pair_count;
  }

  const auto reduction_start_time =
      log_structure_stage
          ? std::chrono::steady_clock::now()
          : std::chrono::steady_clock::time_point();
  for (int reduction_thread = 0;
       reduction_thread < static_cast<int>(
           thread_transformed_delta_hamiltonian_selected.size());
       ++reduction_thread) {
    result.transformed_delta_hamiltonian_selected +=
        thread_transformed_delta_hamiltonian_selected[reduction_thread];
    result.transformed_delta_overlap_selected +=
        thread_transformed_delta_overlap_selected[reduction_thread];
  }
  const double reduction_wall_seconds =
      log_structure_stage ? elapsed_wall_time_seconds(reduction_start_time) : 0.0;

  if (parallel_exception) {
    try {
      std::rethrow_exception(parallel_exception);
    } catch (const std::exception& error) {
      throw std::runtime_error(
          std::string(
              "exact outer-response projected directional structure build failed: ") +
          error.what());
    } catch (...) {
      throw std::runtime_error(
          "exact outer-response projected directional structure build failed "
          "with an unknown exception");
    }
  }

  throw_if_nonfinite(
      result.transformed_delta_hamiltonian_selected,
      "exact outer-response projected directional Hamiltonian");
  throw_if_nonfinite(
      result.transformed_delta_overlap_selected,
      "exact outer-response projected directional overlap");
  if (log_structure_stage) {
    double gather_wall_seconds = 0.0;
    double pair_kernel_wall_seconds = 0.0;
    double projection_wall_seconds = 0.0;
    double alpha_gather_wall_seconds = 0.0;
    double beta_gather_wall_seconds = 0.0;
    double same_spin_wall_seconds = 0.0;
    double opposite_spin_wall_seconds = 0.0;
    std::size_t structure_pair_count = 0;
    std::size_t diagonal_structure_pair_count = 0;
    for (std::size_t thread_index = 0;
         thread_index < thread_gather_wall_seconds.size();
         ++thread_index) {
      gather_wall_seconds += thread_gather_wall_seconds[thread_index];
      pair_kernel_wall_seconds += thread_pair_kernel_wall_seconds[thread_index];
      projection_wall_seconds += thread_projection_wall_seconds[thread_index];
      alpha_gather_wall_seconds +=
          thread_alpha_gather_wall_seconds[thread_index];
      beta_gather_wall_seconds +=
          thread_beta_gather_wall_seconds[thread_index];
      same_spin_wall_seconds +=
          thread_same_spin_wall_seconds[thread_index];
      opposite_spin_wall_seconds +=
          thread_opposite_spin_wall_seconds[thread_index];
      structure_pair_count +=
          thread_structure_pair_count[thread_index];
      diagonal_structure_pair_count +=
          thread_diagonal_structure_pair_count[thread_index];
    }
    std::fprintf(
        stderr,
        "exact_ctx_structure_stage"
        " threads=%d n_structures=%d n_selected=%d"
        " pairs=%zu diagonal_pairs=%zu"
        " gather_s=%.6f pair_kernel_s=%.6f projection_s=%.6f reduction_s=%.6f"
        " alpha_gather_s=%.6f beta_gather_s=%.6f"
        " same_spin_s=%.6f opposite_spin_s=%.6f\n",
        n_parallel_threads,
        n_structures,
        n_selected_states,
        structure_pair_count,
        diagonal_structure_pair_count,
        gather_wall_seconds,
        pair_kernel_wall_seconds,
        projection_wall_seconds,
        reduction_wall_seconds,
        alpha_gather_wall_seconds,
        beta_gather_wall_seconds,
        same_spin_wall_seconds,
        opposite_spin_wall_seconds);
    std::fflush(stderr);
  }
  return result;
}

struct GeneralizedEigenDirectionalResponse {
  Eigen::MatrixXd delta_eigenvector_matrix;
  std::vector<double> delta_eigenvalues;
};

void validate_outer_response_eigensystem(
    const GeneralizedEigenDirectionalResponse& directional_eigensystem) {
  throw_if_nonfinite(
      directional_eigensystem.delta_eigenvector_matrix,
      "exact outer-response directional eigenvectors");
  throw_if_nonfinite(
      directional_eigensystem.delta_eigenvalues,
      "exact outer-response directional eigenvalues");
}

GeneralizedEigenDirectionalResponse build_generalized_eigen_directional_response(
    const CppActiveSpaceSecondOrderContext& accepted_point_context,
    const StructureAccumulationResult& directional_structure_matrices) {
  const int n_structures = accepted_point_context.structure_matrices.n_structures;
  if (n_structures <= 0) {
    throw std::invalid_argument("accepted-point structure dimension must be positive");
  }
  const std::size_t expected_matrix_size =
      n_structures * n_structures;
  if (accepted_point_context.eigen_result.eigenvalues.size() !=
          n_structures ||
      accepted_point_context.eigen_result.eigenvector_matrix.size() !=
          expected_matrix_size) {
    throw std::invalid_argument(
        "accepted-point generalized eigensystem dimensions are inconsistent");
  }
  if (directional_structure_matrices.hamiltonian_matrix.size() != expected_matrix_size ||
      directional_structure_matrices.overlap_matrix.size() != expected_matrix_size) {
    throw std::invalid_argument(
        "directional structure matrix dimensions do not match the accepted point");
  }

  const Eigen::MatrixXd eigenvector_matrix =
      Eigen::Map<const Eigen::MatrixXd>(
          accepted_point_context.eigen_result.eigenvector_matrix.data(),
          n_structures,
          n_structures);
  const Eigen::MatrixXd delta_hamiltonian =
      unpack_symmetric_structure_matrix(
          directional_structure_matrices.hamiltonian_matrix,
          n_structures);
  const Eigen::MatrixXd delta_overlap =
      unpack_symmetric_structure_matrix(
          directional_structure_matrices.overlap_matrix,
          n_structures);
  const Eigen::MatrixXd transformed_delta_hamiltonian =
      eigenvector_matrix.transpose() * delta_hamiltonian * eigenvector_matrix;
  const Eigen::MatrixXd transformed_delta_overlap =
      eigenvector_matrix.transpose() * delta_overlap * eigenvector_matrix;

  std::vector<double> selected_state_weights(
      n_structures,
      0.0);
  for (std::size_t selected_state_offset = 0;
       selected_state_offset < accepted_point_context.selected_state_indices.size();
       ++selected_state_offset) {
    const int state_index =
        accepted_point_context.selected_state_indices[selected_state_offset];
    if (state_index < 0 || state_index >= n_structures) {
      throw std::out_of_range("selected state index is out of range");
    }
    selected_state_weights[state_index] =
        accepted_point_context.normalized_state_weights[selected_state_offset];
  }

  Eigen::MatrixXd eigenvector_rotation = Eigen::MatrixXd::Zero(n_structures, n_structures);
  std::vector<double> delta_eigenvalues(n_structures, 0.0);
  for (int state_index = 0; state_index < n_structures; ++state_index) {
    const double state_energy =
        accepted_point_context.eigen_result.eigenvalues[state_index];
    const double transformed_overlap_diagonal =
        transformed_delta_overlap(state_index, state_index);
    delta_eigenvalues[state_index] =
        transformed_delta_hamiltonian(state_index, state_index) -
        state_energy * transformed_overlap_diagonal;
    eigenvector_rotation(state_index, state_index) =
        -0.5 * transformed_overlap_diagonal;
  }

  // Exact first-order eigenvector response scales like 1 / gap. In the VBSCF
  // structure problem quasi-degenerate roots are common enough that the raw
  // formula can explode numerically and poison the whole outer-response HVP.
  // Treat very small gaps as gauge-ambiguous blocks and regularize unequal
  // weight pairs with a bounded denominator instead of a naked division.
  constexpr double kDegeneracyToleranceScale = 1.0e6;
  constexpr double kRelativeGapTolerance = 1.0e-8;
  // The first-order generalized-eigenvector response is required for every
  // off-diagonal state pair; large gaps make the division well-conditioned,
  // not negligible. Keep the cached path algebraically identical to the
  // reference implementation and only reuse accepted-point storage.
  for (int column_state = 0; column_state < n_structures; ++column_state) {
    const double column_energy =
        accepted_point_context.eigen_result.eigenvalues[column_state];
    for (int row_state = 0; row_state < n_structures; ++row_state) {
      if (row_state == column_state) {
        continue;
      }
      const double row_energy =
          accepted_point_context.eigen_result.eigenvalues[row_state];
      const double gap = column_energy - row_energy;
      const double numerator =
          transformed_delta_hamiltonian(row_state, column_state) -
          column_energy * transformed_delta_overlap(row_state, column_state);
      const double overlap_gauge_rotation =
          -0.5 * transformed_delta_overlap(row_state, column_state);
      const double gap_tolerance =
          std::max(
              kRelativeGapTolerance *
                  std::max({1.0, std::abs(column_energy), std::abs(row_energy)}),
              kDegeneracyToleranceScale *
                  std::numeric_limits<double>::epsilon() *
                  std::max({1.0, std::abs(column_energy), std::abs(row_energy)}));
      if (std::abs(gap) <= gap_tolerance) {
        const double row_weight =
            selected_state_weights[row_state];
        const double column_weight =
            selected_state_weights[column_state];
        if (std::abs(row_weight - column_weight) <= 1.0e-12) {
          eigenvector_rotation(row_state, column_state) =
              overlap_gauge_rotation;
          continue;
        }
        const double safe_gap =
            std::copysign(
                gap_tolerance,
                gap != 0.0 ? gap : (numerator != 0.0 ? numerator : 1.0));
        const double regularized_rotation = numerator / safe_gap;
        eigenvector_rotation(row_state, column_state) =
            std::isfinite(regularized_rotation)
                ? regularized_rotation
                : overlap_gauge_rotation;
        continue;
      }
      eigenvector_rotation(row_state, column_state) =
          numerator / gap;
    }
  }
  throw_if_nonfinite(
      eigenvector_rotation,
      "exact outer-response eigenvector rotation");
  throw_if_nonfinite(
      delta_eigenvalues,
      "exact outer-response directional eigenvalues");

  GeneralizedEigenDirectionalResponse result;
  result.delta_eigenvector_matrix =
      eigenvector_matrix * eigenvector_rotation;
  throw_if_nonfinite(
      result.delta_eigenvector_matrix,
      "exact outer-response directional eigenvectors");
  result.delta_eigenvalues = std::move(delta_eigenvalues);
  return result;
}

SelectedStateGeneralizedEigenDirectionalResponse
build_selected_state_generalized_eigen_directional_response(
    const CppActiveSpaceSecondOrderContext& accepted_point_context,
    const SelectedStateProjectedDirectionalMatrices&
        projected_directional_structure_matrices) {
  const int n_structures = accepted_point_context.structure_matrices.n_structures;
  const int n_selected_states =
      static_cast<int>(accepted_point_context.selected_state_indices.size());
  if (n_structures <= 0 || n_selected_states <= 0) {
    throw std::invalid_argument(
        "selected-state directional eigensystem requires positive dimensions");
  }
  if (projected_directional_structure_matrices
              .transformed_delta_hamiltonian_selected.rows() != n_structures ||
      projected_directional_structure_matrices
              .transformed_delta_hamiltonian_selected.cols() !=
          n_selected_states ||
      projected_directional_structure_matrices
              .transformed_delta_overlap_selected.rows() != n_structures ||
      projected_directional_structure_matrices
              .transformed_delta_overlap_selected.cols() !=
          n_selected_states) {
    throw std::invalid_argument(
        "projected directional structure dimensions do not match the accepted point");
  }
  const std::size_t expected_eigenvector_size =
      n_structures * n_structures;
  if (accepted_point_context.eigen_result.eigenvalues.size() !=
          n_structures ||
      accepted_point_context.eigen_result.eigenvector_matrix.size() !=
          expected_eigenvector_size) {
    throw std::invalid_argument(
        "accepted-point generalized eigensystem dimensions are inconsistent");
  }

  const Eigen::Map<const Eigen::MatrixXd> eigenvector_matrix(
      accepted_point_context.eigen_result.eigenvector_matrix.data(),
      n_structures,
      n_structures);
  SelectedStateGeneralizedEigenDirectionalResponse result;
  result.delta_selected_eigenvector_matrix =
      Eigen::MatrixXd::Zero(n_structures, n_selected_states);
  result.delta_selected_eigenvalues.assign(n_selected_states, 0.0);
  constexpr double kDegeneracyToleranceScale = 1.0e6;
  constexpr double kRelativeGapTolerance = 1.0e-8;
  if (n_selected_states == 1) {
    const int column_state =
        accepted_point_context.selected_state_indices.front();
    if (column_state < 0 || column_state >= n_structures) {
      throw std::out_of_range("selected state index is out of range");
    }
    const double column_energy =
        accepted_point_context.eigen_result.eigenvalues[
            column_state];
    const auto transformed_delta_hamiltonian_column =
        projected_directional_structure_matrices
            .transformed_delta_hamiltonian_selected.col(0);
    const auto transformed_delta_overlap_column =
        projected_directional_structure_matrices
            .transformed_delta_overlap_selected.col(0);
    Eigen::VectorXd eigenvector_rotation_column =
        Eigen::VectorXd::Zero(n_structures);
    const double transformed_overlap_diagonal =
        transformed_delta_overlap_column[column_state];
    result.delta_selected_eigenvalues[0] =
        transformed_delta_hamiltonian_column[column_state] -
        column_energy * transformed_overlap_diagonal;
    eigenvector_rotation_column[column_state] =
        -0.5 * transformed_overlap_diagonal;

    for (int row_state = 0; row_state < n_structures; ++row_state) {
      if (row_state == column_state) {
        continue;
      }
      const double row_energy =
          accepted_point_context.eigen_result.eigenvalues[
              row_state];
      const double gap = column_energy - row_energy;
      const double numerator =
          transformed_delta_hamiltonian_column[row_state] -
          column_energy * transformed_delta_overlap_column[row_state];
      const double overlap_gauge_rotation =
          -0.5 * transformed_delta_overlap_column[row_state];
      const double gap_tolerance =
          std::max(
              kRelativeGapTolerance *
                  std::max({1.0, std::abs(column_energy), std::abs(row_energy)}),
              kDegeneracyToleranceScale *
                  std::numeric_limits<double>::epsilon() *
                  std::max({1.0, std::abs(column_energy), std::abs(row_energy)}));
      if (std::abs(gap) <= gap_tolerance) {
        const double safe_gap =
            std::copysign(
                gap_tolerance,
                gap != 0.0 ? gap : (numerator != 0.0 ? numerator : 1.0));
        const double regularized_rotation = numerator / safe_gap;
        eigenvector_rotation_column[row_state] =
            std::isfinite(regularized_rotation)
                ? regularized_rotation
                : overlap_gauge_rotation;
        continue;
      }
      eigenvector_rotation_column[row_state] = numerator / gap;
    }

    result.delta_selected_eigenvector_matrix.col(0).noalias() =
        eigenvector_matrix * eigenvector_rotation_column;
    throw_if_nonfinite(
        result.delta_selected_eigenvector_matrix,
        "exact outer-response directional selected-state eigenvectors");
    throw_if_nonfinite(
        result.delta_selected_eigenvalues,
        "exact outer-response directional selected-state energies");
    return result;
  }

  std::vector<double> selected_state_weights(
      n_structures,
      0.0);
  for (std::size_t selected_state_offset = 0;
       selected_state_offset < accepted_point_context.selected_state_indices.size();
       ++selected_state_offset) {
    const int state_index =
        accepted_point_context.selected_state_indices[selected_state_offset];
    if (state_index < 0 || state_index >= n_structures) {
      throw std::out_of_range("selected state index is out of range");
    }
    selected_state_weights[state_index] =
        accepted_point_context.normalized_state_weights[selected_state_offset];
  }

  for (int selected_state_offset = 0;
       selected_state_offset < n_selected_states;
       ++selected_state_offset) {
    const int column_state =
        accepted_point_context.selected_state_indices[
            selected_state_offset];
    const double column_energy =
        accepted_point_context.eigen_result.eigenvalues[
            column_state];
    const Eigen::VectorXd transformed_delta_hamiltonian_column =
        projected_directional_structure_matrices
            .transformed_delta_hamiltonian_selected.col(selected_state_offset);
    const Eigen::VectorXd transformed_delta_overlap_column =
        projected_directional_structure_matrices
            .transformed_delta_overlap_selected.col(selected_state_offset);
    Eigen::VectorXd eigenvector_rotation_column =
        Eigen::VectorXd::Zero(n_structures);
    const double transformed_overlap_diagonal =
        transformed_delta_overlap_column[column_state];
    result.delta_selected_eigenvalues[selected_state_offset] =
        transformed_delta_hamiltonian_column[column_state] -
        column_energy * transformed_overlap_diagonal;
    eigenvector_rotation_column[column_state] =
        -0.5 * transformed_overlap_diagonal;

    for (int row_state = 0; row_state < n_structures; ++row_state) {
      if (row_state == column_state) {
        continue;
      }
      const double row_energy =
          accepted_point_context.eigen_result.eigenvalues[
              row_state];
      const double gap = column_energy - row_energy;
      const double numerator =
          transformed_delta_hamiltonian_column[row_state] -
          column_energy * transformed_delta_overlap_column[row_state];
      const double overlap_gauge_rotation =
          -0.5 * transformed_delta_overlap_column[row_state];
      const double gap_tolerance =
          std::max(
              kRelativeGapTolerance *
                  std::max({1.0, std::abs(column_energy), std::abs(row_energy)}),
              kDegeneracyToleranceScale *
                  std::numeric_limits<double>::epsilon() *
                  std::max({1.0, std::abs(column_energy), std::abs(row_energy)}));
      if (std::abs(gap) <= gap_tolerance) {
        const double row_weight =
            selected_state_weights[row_state];
        const double column_weight =
            selected_state_weights[column_state];
        if (std::abs(row_weight - column_weight) <= 1.0e-12) {
          eigenvector_rotation_column[row_state] =
              overlap_gauge_rotation;
          continue;
        }
        const double safe_gap =
            std::copysign(
                gap_tolerance,
                gap != 0.0 ? gap : (numerator != 0.0 ? numerator : 1.0));
        const double regularized_rotation = numerator / safe_gap;
        eigenvector_rotation_column[row_state] =
            std::isfinite(regularized_rotation)
                ? regularized_rotation
                : overlap_gauge_rotation;
        continue;
      }
      eigenvector_rotation_column[row_state] = numerator / gap;
    }

    result.delta_selected_eigenvector_matrix.col(selected_state_offset) =
        eigenvector_matrix * eigenvector_rotation_column;
  }

  throw_if_nonfinite(
      result.delta_selected_eigenvector_matrix,
      "exact outer-response directional selected-state eigenvectors");
  throw_if_nonfinite(
      result.delta_selected_eigenvalues,
      "exact outer-response directional selected-state energies");
  return result;
}


StructurePairWeightTables build_directional_structure_pair_weight_tables(
    const CppActiveSpaceSecondOrderContext& accepted_point_context,
    const GeneralizedEigenDirectionalResponse& directional_eigensystem) {
  const int n_structures = accepted_point_context.structure_matrices.n_structures;
  if (directional_eigensystem.delta_eigenvector_matrix.rows() != n_structures ||
      directional_eigensystem.delta_eigenvector_matrix.cols() != n_structures ||
      directional_eigensystem.delta_eigenvalues.size() != n_structures) {
    throw std::invalid_argument(
        "directional eigensystem dimensions do not match the accepted point");
  }

  const Eigen::MatrixXd eigenvector_matrix =
      Eigen::Map<const Eigen::MatrixXd>(
          accepted_point_context.eigen_result.eigenvector_matrix.data(),
          n_structures,
          n_structures);
  Eigen::MatrixXd hamiltonian_weight_matrix =
      Eigen::MatrixXd::Zero(n_structures, n_structures);
  Eigen::MatrixXd overlap_weight_matrix =
      Eigen::MatrixXd::Zero(n_structures, n_structures);
  for (std::size_t selected_state_offset = 0;
       selected_state_offset < accepted_point_context.selected_state_indices.size();
       ++selected_state_offset) {
    const int state_index =
        accepted_point_context.selected_state_indices[selected_state_offset];
    const double state_weight =
        accepted_point_context.normalized_state_weights[selected_state_offset];
    const double state_energy =
        accepted_point_context.eigen_result.eigenvalues[state_index];
    const double delta_state_energy =
        directional_eigensystem.delta_eigenvalues[state_index];
    const Eigen::VectorXd eigenvector =
        eigenvector_matrix.col(state_index);
    const Eigen::VectorXd delta_eigenvector =
        directional_eigensystem.delta_eigenvector_matrix.col(state_index);
    const Eigen::MatrixXd directional_projector =
        delta_eigenvector * eigenvector.transpose() +
        eigenvector * delta_eigenvector.transpose();
    hamiltonian_weight_matrix.noalias() +=
        state_weight * directional_projector;
    overlap_weight_matrix.noalias() -=
        state_weight *
        (delta_state_energy * (eigenvector * eigenvector.transpose()) +
         state_energy * directional_projector);
  }

  StructurePairWeightTables result;
  result.hamiltonian_upper_weights =
      pack_symmetric_structure_weight_matrix(
          hamiltonian_weight_matrix);
  result.overlap_upper_weights =
      pack_symmetric_structure_weight_matrix(
          overlap_weight_matrix);
  return result;
}

StructurePairAdjoints determinant_pair_structure_adjoints(
    const std::vector<StructureExpansionTerm>& determinant_to_structures_left,
    const std::vector<StructureExpansionTerm>& determinant_to_structures_right,
    const StructurePairWeightTables& structure_pair_weights) {
  StructurePairAdjoints adjoints;
  for (const auto& left_term : determinant_to_structures_left) {
    for (const auto& right_term : determinant_to_structures_right) {
      if (left_term.structure_index > right_term.structure_index) {
        continue;
      }
      const double coefficient_product =
          left_term.coefficient * right_term.coefficient;
      const std::size_t storage_index = structure_upper_storage_index(
          left_term.structure_index,
          right_term.structure_index);
      adjoints.hamiltonian_weight +=
          coefficient_product *
          structure_pair_weights.hamiltonian_upper_weights[storage_index];
      adjoints.overlap_weight +=
          coefficient_product *
          structure_pair_weights.overlap_upper_weights[storage_index];
    }
  }
  return adjoints;
}

ActiveSpaceGradientDirection build_pairwise_active_space_gradient_direction_from_structure_weights(
    const CppVbInput& input,
    const CppActiveSpaceSecondOrderContext& accepted_point_context,
    const StructurePairWeightTables& structure_pair_weights,
    bool skip_opposite_spin) {
  const int n_active_orbitals =
      input.orbital_preparation_input.n_active_orbitals;
  const int n_determinants =
      static_cast<int>(input.structure_data.alpha_det.size());
  if (n_active_orbitals <= 0 || n_determinants <= 0) {
    throw std::invalid_argument(
        "active-space gradient direction requires positive dimensions");
  }

  ActiveSpaceGradientDirection direction;
  direction.active_orbital_overlap_gradient.assign(
      n_active_orbitals * n_active_orbitals,
      0.0);
  direction.active_one_electron_gradient.assign(
      n_active_orbitals * n_active_orbitals,
      0.0);
  direction.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  Eigen::MatrixXd active_one_electron_gradient_matrix =
      Eigen::MatrixXd::Zero(n_active_orbitals, n_active_orbitals);
  FullDeterminantStructureHamiltonianOverlapBuilder structure_builder;
  const FullDeterminantPairEvaluator pair_evaluator =
      structure_builder.make_pair_evaluator();
  const auto& accepted_prepared_active_space =
      accepted_point_context.prepared_active_space;
  for (int determinant_index_left = 0;
       determinant_index_left < n_determinants;
       ++determinant_index_left) {
    for (int determinant_index_right = 0;
         determinant_index_right <= determinant_index_left;
         ++determinant_index_right) {
      const auto direct_pair_adjoints = determinant_pair_structure_adjoints(
          input.structure_data.determinant_to_structure_terms[
              determinant_index_left],
          input.structure_data.determinant_to_structure_terms[
              determinant_index_right],
          structure_pair_weights);
      StructurePairAdjoints combined_pair_adjoints = direct_pair_adjoints;
      if (determinant_index_left != determinant_index_right) {
        const auto swapped_pair_adjoints = determinant_pair_structure_adjoints(
            input.structure_data.determinant_to_structure_terms[
                determinant_index_right],
            input.structure_data.determinant_to_structure_terms[
                determinant_index_left],
            structure_pair_weights);
        combined_pair_adjoints.hamiltonian_weight +=
            swapped_pair_adjoints.hamiltonian_weight;
        combined_pair_adjoints.overlap_weight +=
            swapped_pair_adjoints.overlap_weight;
      }
      if (!has_nonzero_structure_pair_adjoints(combined_pair_adjoints)) {
        continue;
      }

      const auto determinant_pair_evaluation =
          evaluate_active_space_determinant_pair_local(
              &accepted_point_context.same_spin_pair_cache,
              pair_evaluator,
              input,
              accepted_prepared_active_space.orbital_result
                  .active_orbital_overlap_matrix,
              accepted_prepared_active_space.active_space_one_electron_result,
              accepted_prepared_active_space.active_space_two_electron_result,
              determinant_index_left,
              determinant_index_right,
              n_active_orbitals);
      accumulate_active_space_gradient_pair_with_adjoints_local(
          combined_pair_adjoints,
          input,
          accepted_prepared_active_space.active_space_one_electron_result,
          accepted_prepared_active_space.active_space_two_electron_result,
          determinant_pair_evaluation,
          determinant_index_left,
          determinant_index_right,
          n_active_orbitals,
          skip_opposite_spin,
          &active_one_electron_gradient_matrix,
          &direction.active_orbital_overlap_gradient,
          &direction.packed_active_two_electron_gradient);
    }
  }

  direction.active_one_electron_gradient.assign(
      active_one_electron_gradient_matrix.data(),
      active_one_electron_gradient_matrix.data() +
          active_one_electron_gradient_matrix.size());
  return direction;
}

ActiveSpaceGradientDirection
build_pairwise_active_space_gradient_direction_from_determinant_pair_weights(
    const CppVbInput& input,
    const CppActiveSpaceSecondOrderContext& accepted_point_context,
    const DeterminantPairWeightTablesFromCoefficients& determinant_pair_weights,
    bool skip_opposite_spin) {
  const int n_active_orbitals =
      input.orbital_preparation_input.n_active_orbitals;
  const int n_determinants =
      static_cast<int>(input.structure_data.alpha_det.size());
  if (n_active_orbitals <= 0 || n_determinants <= 0) {
    throw std::invalid_argument(
        "active-space gradient direction requires positive dimensions");
  }
  if (determinant_pair_weights.n_determinants != n_determinants) {
    throw std::invalid_argument(
        "determinant-pair directional weights do not match n_determinants");
  }
  if (determinant_pair_weights.unordered_combined_hamiltonian_weights.size() !=
          unordered_determinant_pair_count(n_determinants) ||
      determinant_pair_weights.unordered_combined_overlap_weights.size() !=
          unordered_determinant_pair_count(n_determinants)) {
    throw std::invalid_argument(
        "determinant-pair directional unordered weights size mismatch");
  }

  ActiveSpaceGradientDirection direction;
  direction.active_orbital_overlap_gradient.assign(
      n_active_orbitals * n_active_orbitals,
      0.0);
  direction.active_one_electron_gradient.assign(
      n_active_orbitals * n_active_orbitals,
      0.0);
  direction.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  Eigen::MatrixXd active_one_electron_gradient_matrix =
      Eigen::MatrixXd::Zero(n_active_orbitals, n_active_orbitals);
  FullDeterminantStructureHamiltonianOverlapBuilder structure_builder;
  const FullDeterminantPairEvaluator pair_evaluator =
      structure_builder.make_pair_evaluator();
  const auto& accepted_prepared_active_space =
      accepted_point_context.prepared_active_space;
  for (int determinant_index_left = 0;
       determinant_index_left < n_determinants;
       ++determinant_index_left) {
    for (int determinant_index_right = 0;
         determinant_index_right <= determinant_index_left;
         ++determinant_index_right) {
      const std::size_t unordered_index =
          canonical_determinant_pair_storage_index(
              determinant_index_left,
              determinant_index_right);
      const StructurePairAdjoints pair_adjoints = {
          determinant_pair_weights
              .unordered_combined_hamiltonian_weights[unordered_index],
          determinant_pair_weights
              .unordered_combined_overlap_weights[unordered_index],
      };
      if (!has_nonzero_structure_pair_adjoints(pair_adjoints)) {
        continue;
      }

      const auto determinant_pair_evaluation =
          evaluate_active_space_determinant_pair_local(
              &accepted_point_context.same_spin_pair_cache,
              pair_evaluator,
              input,
              accepted_prepared_active_space.orbital_result
                  .active_orbital_overlap_matrix,
              accepted_prepared_active_space.active_space_one_electron_result,
              accepted_prepared_active_space.active_space_two_electron_result,
              determinant_index_left,
              determinant_index_right,
              n_active_orbitals);
      accumulate_active_space_gradient_pair_with_adjoints_local(
          pair_adjoints,
          input,
          accepted_prepared_active_space.active_space_one_electron_result,
          accepted_prepared_active_space.active_space_two_electron_result,
          determinant_pair_evaluation,
          determinant_index_left,
          determinant_index_right,
          n_active_orbitals,
          skip_opposite_spin,
          &active_one_electron_gradient_matrix,
          &direction.active_orbital_overlap_gradient,
          &direction.packed_active_two_electron_gradient);
    }
  }

  direction.active_one_electron_gradient.assign(
      active_one_electron_gradient_matrix.data(),
      active_one_electron_gradient_matrix.data() +
          active_one_electron_gradient_matrix.size());
  return direction;
}

ActiveSpaceGradientDirection
build_pairwise_local_active_space_gradient_direction_from_determinant_pair_weights(
    const CppVbInput& input,
    const CppActiveSpaceSecondOrderContext& accepted_point_context,
    const DeterminantPairWeightTablesFromCoefficients& determinant_pair_weights,
    bool skip_opposite_spin,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  // Pairwise analytic fallback for the outer-response local term. This keeps
  // the accepted selected-state weights fixed and differentiates the exact
  // determinant-pair backward formulas themselves with respect to the active
  // overlap / one-electron / two-electron direction.
  const int n_active_orbitals =
      input.orbital_preparation_input.n_active_orbitals;
  const int n_determinants =
      static_cast<int>(input.structure_data.alpha_det.size());
  if (n_active_orbitals <= 0 || n_determinants <= 0) {
    throw std::invalid_argument(
        "local active-space gradient direction requires positive dimensions");
  }
  if (determinant_pair_weights.n_determinants != n_determinants ||
      determinant_pair_weights.unordered_combined_hamiltonian_weights.size() !=
          unordered_determinant_pair_count(n_determinants) ||
      determinant_pair_weights.unordered_combined_overlap_weights.size() !=
          unordered_determinant_pair_count(n_determinants)) {
    throw std::invalid_argument(
        "determinant_pair_weights dimensions do not match determinant count");
  }

  ActiveSpaceGradientDirection direction;
  direction.active_orbital_overlap_gradient.assign(
      n_active_orbitals * n_active_orbitals,
      0.0);
  direction.active_one_electron_gradient.assign(
      n_active_orbitals * n_active_orbitals,
      0.0);
  direction.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  Eigen::MatrixXd active_one_electron_gradient_matrix =
      Eigen::MatrixXd::Zero(n_active_orbitals, n_active_orbitals);
  FullDeterminantStructureHamiltonianOverlapBuilder structure_builder;
  const FullDeterminantPairEvaluator pair_evaluator =
      structure_builder.make_pair_evaluator();
  const auto& accepted_prepared_active_space =
      accepted_point_context.prepared_active_space;
  for (int determinant_index_left = 0;
       determinant_index_left < n_determinants;
       ++determinant_index_left) {
    for (int determinant_index_right = 0;
         determinant_index_right <= determinant_index_left;
         ++determinant_index_right) {
      const std::size_t unordered_index =
          canonical_determinant_pair_storage_index(
              determinant_index_left,
              determinant_index_right);
      const StructurePairAdjoints pair_adjoints{
          determinant_pair_weights
              .unordered_combined_hamiltonian_weights[unordered_index],
          determinant_pair_weights
              .unordered_combined_overlap_weights[unordered_index],
      };
      if (!has_nonzero_structure_pair_adjoints(pair_adjoints)) {
        continue;
      }

      const auto determinant_pair_evaluation =
          evaluate_active_space_determinant_pair_local(
              &accepted_point_context.same_spin_pair_cache,
              pair_evaluator,
              input,
              accepted_prepared_active_space.orbital_result
                  .active_orbital_overlap_matrix,
              accepted_prepared_active_space.active_space_one_electron_result,
              accepted_prepared_active_space.active_space_two_electron_result,
              determinant_index_left,
              determinant_index_right,
              n_active_orbitals);
      accumulate_active_space_gradient_pair_local_response_with_adjoints_local(
          pair_adjoints,
          input,
          accepted_prepared_active_space.active_space_one_electron_result,
          accepted_prepared_active_space.active_space_two_electron_result,
          determinant_pair_evaluation,
          determinant_index_left,
          determinant_index_right,
          n_active_orbitals,
          skip_opposite_spin,
          delta_active_orbital_overlap_matrix,
          delta_active_one_electron_matrix,
          delta_packed_active_two_electron_integrals,
          &active_one_electron_gradient_matrix,
          &direction.active_orbital_overlap_gradient,
          &direction.packed_active_two_electron_gradient);
    }
  }

  direction.active_one_electron_gradient.assign(
      active_one_electron_gradient_matrix.data(),
      active_one_electron_gradient_matrix.data() +
          active_one_electron_gradient_matrix.size());
  return direction;
}

ActiveSpaceGradientDirection
build_pairwise_opposite_spin_local_active_space_gradient_direction_from_determinant_pair_weights(
    const CppVbInput& input,
    const CppActiveSpaceSecondOrderContext& accepted_point_context,
    const DeterminantPairWeightTablesFromCoefficients& determinant_pair_weights,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  const int n_active_orbitals =
      input.orbital_preparation_input.n_active_orbitals;
  const int n_determinants =
      static_cast<int>(input.structure_data.alpha_det.size());
  if (n_active_orbitals <= 0 || n_determinants <= 0) {
    throw std::invalid_argument(
        "local active-space gradient direction requires positive dimensions");
  }
  if (determinant_pair_weights.n_determinants != n_determinants ||
      determinant_pair_weights.unordered_combined_hamiltonian_weights.size() !=
          unordered_determinant_pair_count(n_determinants) ||
      determinant_pair_weights.unordered_combined_overlap_weights.size() !=
          unordered_determinant_pair_count(n_determinants)) {
    throw std::invalid_argument(
        "determinant_pair_weights dimensions do not match determinant count");
  }

  ActiveSpaceGradientDirection direction;
  direction.active_orbital_overlap_gradient.assign(
      n_active_orbitals * n_active_orbitals,
      0.0);
  direction.active_one_electron_gradient.assign(
      n_active_orbitals * n_active_orbitals,
      0.0);
  direction.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(n_active_orbitals),
      0.0);

  FullDeterminantStructureHamiltonianOverlapBuilder structure_builder;
  const FullDeterminantPairEvaluator pair_evaluator =
      structure_builder.make_pair_evaluator();
  const auto& accepted_prepared_active_space =
      accepted_point_context.prepared_active_space;
  for (int determinant_index_left = 0;
       determinant_index_left < n_determinants;
       ++determinant_index_left) {
    for (int determinant_index_right = 0;
         determinant_index_right <= determinant_index_left;
         ++determinant_index_right) {
      const std::size_t unordered_index =
          canonical_determinant_pair_storage_index(
              determinant_index_left,
              determinant_index_right);
      const StructurePairAdjoints pair_adjoints{
          determinant_pair_weights
              .unordered_combined_hamiltonian_weights[unordered_index],
          determinant_pair_weights
              .unordered_combined_overlap_weights[unordered_index],
      };
      if (pair_adjoints.hamiltonian_weight == 0.0) {
        continue;
      }

      const auto determinant_pair_evaluation =
          evaluate_active_space_determinant_pair_local(
              &accepted_point_context.same_spin_pair_cache,
              pair_evaluator,
              input,
              accepted_prepared_active_space.orbital_result
                  .active_orbital_overlap_matrix,
              accepted_prepared_active_space.active_space_one_electron_result,
              accepted_prepared_active_space.active_space_two_electron_result,
              determinant_index_left,
              determinant_index_right,
              n_active_orbitals);
      accumulate_active_space_gradient_pair_opposite_spin_local_response_with_adjoints_local(
          pair_adjoints,
          input,
          accepted_prepared_active_space.active_space_one_electron_result,
          accepted_prepared_active_space.active_space_two_electron_result,
          determinant_pair_evaluation,
          determinant_index_left,
          determinant_index_right,
          n_active_orbitals,
          delta_active_orbital_overlap_matrix,
          delta_active_one_electron_matrix,
          delta_packed_active_two_electron_integrals,
          &direction.active_orbital_overlap_gradient,
          &direction.packed_active_two_electron_gradient);
    }
  }

  return direction;
}

std::vector<double> gather_directional_selected_state_energies(
    const std::vector<double>& directional_eigenvalues,
    const std::vector<int>& selected_state_indices) {
  std::vector<double> selected_state_directional_energies;
  selected_state_directional_energies.reserve(selected_state_indices.size());
  for (const int state_index : selected_state_indices) {
    if (state_index < 0 ||
        state_index >= directional_eigenvalues.size()) {
      throw std::out_of_range(
          "selected state index is out of range for directional eigenvalues");
    }
    selected_state_directional_energies.push_back(
        directional_eigenvalues[state_index]);
  }
  return selected_state_directional_energies;
}

void accumulate_scaled_vector(
    const std::vector<double>& source,
    double scale,
    std::vector<double>* target) {
  if (scale == 0.0) {
    return;
  }
  for (std::size_t index = 0; index < source.size(); ++index) {
    (*target)[index] += scale * source[index];
  }
}

void accumulate_scaled_same_spin_contribution(
    const SameSpinMatrixBackwardContribution& contribution,
    double scale,
    ActiveSpaceGradientDirection* target) {
  accumulate_scaled_vector(
      contribution.active_orbital_overlap_gradient,
      scale,
      &target->active_orbital_overlap_gradient);
  accumulate_scaled_vector(
      contribution.active_one_electron_gradient,
      scale,
      &target->active_one_electron_gradient);
  accumulate_scaled_vector(
      contribution.packed_active_two_electron_gradient,
      scale,
      &target->packed_active_two_electron_gradient);
}

ActiveSpaceGradientDirection build_active_space_gradient_direction_from_outer_response(
    const CppVbInput& input,
    const CppActiveSpaceSecondOrderContext& accepted_point_context,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals,
    const SelectedStateDeterminantMatrices& directional_selected_states,
    const std::vector<double>& directional_selected_state_energies) {
  if (!accepted_point_context.use_matrix_form_opposite_spin) {
    throw std::runtime_error(
        "outer-response active-gradient direction requires selected-state matrices");
  }
  validate_selected_state_determinant_matrices(
      directional_selected_states,
      "exact outer-response directional selected-state coefficients");
  ActiveSpaceGradientDirection direction =
      {};
  direction.active_orbital_overlap_gradient.assign(
      input.orbital_preparation_input.n_active_orbitals *
          input.orbital_preparation_input.n_active_orbitals,
      0.0);
  direction.active_one_electron_gradient.assign(
      input.orbital_preparation_input.n_active_orbitals *
          input.orbital_preparation_input.n_active_orbitals,
      0.0);
  direction.packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(
          input.orbital_preparation_input.n_active_orbitals),
      0.0);
  // The local outer-response is fully matrix-form again: same-spin uses the
  // repaired canonical half-pair contraction and opposite-spin stays on the
  // already-validated matrix-form block contraction. HHO/SSO are symmetrized
  // later before the orbital pullback, so matching the pairwise canonical
  // storage convention here removes the previous diagnostic mismatch without
  // changing the physical HVP.
  const SameSpinMatrixBackwardContribution matrix_form_local_same_spin_response =
      build_local_same_spin_matrix_backward_contribution(
          accepted_point_context.same_spin_pair_cache,
          accepted_point_context.selected_state_matrices,
          accepted_point_context.selected_state_energies,
          input.orbital_preparation_input.n_active_orbitals,
          accepted_point_context.prepared_active_space
              .active_space_one_electron_result.h1e_act,
          accepted_point_context.prepared_active_space.active_space_two_electron_result,
          delta_active_orbital_overlap_matrix,
          delta_active_one_electron_matrix,
          delta_packed_active_two_electron_integrals);
  validate_same_spin_matrix_backward_contribution(
      matrix_form_local_same_spin_response,
      "exact outer-response local same-spin backward contribution");
  const OppositeSpinMatrixBackwardContribution matrix_form_local_opposite_spin_response =
      build_local_opposite_spin_matrix_backward_contribution(
          accepted_point_context.same_spin_pair_cache,
          accepted_point_context.selected_state_matrices,
          input.orbital_preparation_input.n_active_orbitals,
          accepted_point_context.prepared_active_space.active_space_two_electron_result,
          delta_active_orbital_overlap_matrix,
          delta_packed_active_two_electron_integrals);
  validate_opposite_spin_matrix_backward_contribution(
      matrix_form_local_opposite_spin_response,
      "exact outer-response local opposite-spin backward contribution");
  accumulate_scaled_same_spin_contribution(
      matrix_form_local_same_spin_response,
      1.0,
      &direction);
  accumulate_scaled_vector(
      matrix_form_local_opposite_spin_response.active_orbital_overlap_gradient,
      1.0,
      &direction.active_orbital_overlap_gradient);
  accumulate_scaled_vector(
      matrix_form_local_opposite_spin_response.packed_active_two_electron_gradient,
      1.0,
      &direction.packed_active_two_electron_gradient);
  if (!accepted_point_context.use_full_matrix_form_adjoint) {
    throw std::runtime_error(
        "exact outer-response active-gradient direction requires the "
        "same-spin matrix-form adjoint path");
  }

  const SameSpinMatrixBackwardContribution matrix_form_same_spin_direction =
      build_directional_same_spin_matrix_backward_contribution(
          accepted_point_context.same_spin_pair_cache,
          accepted_point_context.selected_state_matrices,
          directional_selected_states,
          accepted_point_context.selected_state_energies,
          directional_selected_state_energies,
          input.orbital_preparation_input.n_active_orbitals);
  validate_same_spin_matrix_backward_contribution(
      matrix_form_same_spin_direction,
      "exact outer-response directional same-spin backward contribution");
  const OppositeSpinMatrixBackwardContribution matrix_form_opposite_spin_direction =
      build_directional_opposite_spin_matrix_backward_contribution(
          accepted_point_context.same_spin_pair_cache,
          accepted_point_context.selected_state_matrices,
          directional_selected_states,
          input.orbital_preparation_input.n_active_orbitals);
  validate_opposite_spin_matrix_backward_contribution(
      matrix_form_opposite_spin_direction,
      "exact outer-response directional opposite-spin backward contribution");
  accumulate_scaled_same_spin_contribution(
      matrix_form_same_spin_direction,
      1.0,
      &direction);
  accumulate_scaled_vector(
      matrix_form_opposite_spin_direction.active_orbital_overlap_gradient,
      1.0,
      &direction.active_orbital_overlap_gradient);
  accumulate_scaled_vector(
      matrix_form_opposite_spin_direction.packed_active_two_electron_gradient,
      1.0,
      &direction.packed_active_two_electron_gradient);
  return direction;
}

void write_symmetric_active_matrix_average_local(
    const std::vector<double>& matrix_storage,
    int dimension,
    std::vector<double>* symmetric_storage) {
  const std::size_t expected_size = dimension * dimension;
  resize_for_overwrite(symmetric_storage, expected_size);
  for (int column = 0; column < dimension; ++column) {
    for (int row = 0; row < dimension; ++row) {
      (*symmetric_storage)[(column) * (dimension) + (row)] =
          0.5 *
          (matrix_storage[(column) * (dimension) + (row)] +
           matrix_storage[(row) * (dimension) + (column)]);
    }
  }
}

std::vector<double> build_orbital_value_gradient_from_active_space_gradient_direction(
    const CppVbInput& input,
    const CppActiveSpaceSecondOrderContext& accepted_point_context,
    const ActiveSpaceGradientDirection& active_space_gradient_direction,
    const AcceptedOrbitalPreparationCache* orbital_preparation_cache,
    std::vector<double>* symmetric_active_overlap_gradient_workspace,
    std::vector<double>* symmetric_active_one_electron_gradient_workspace) {
  const int n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) / 2;
  const int n_active_orbitals =
      input.orbital_preparation_input.n_active_orbitals;
  const auto& prepared_active_space =
      accepted_point_context.prepared_active_space;
  const auto& orbital_result = prepared_active_space.orbital_result;
  const auto& ao_effective_one_electron_result =
      prepared_active_space.ao_effective_one_electron_result;
  const auto& active_space_two_electron_result =
      prepared_active_space.active_space_two_electron_result;
  // The outer-response adjoint is assembled in canonical pairwise storage, so
  // the orbital pullback must consume the symmetric active-space gradients
  // through the same vector-storage overload used by the fixed-adjoint path.
  // The matrix overload applies a different SSO normalization and was the
  // source of the large exact-ctx outer-response HVP mismatch on 241_VBSCF.
  std::vector<double> local_symmetric_active_overlap_gradient;
  std::vector<double> local_symmetric_active_one_electron_gradient;
  std::vector<double>& symmetric_active_overlap_gradient =
      symmetric_active_overlap_gradient_workspace != nullptr
          ? *symmetric_active_overlap_gradient_workspace
          : local_symmetric_active_overlap_gradient;
  std::vector<double>& symmetric_active_one_electron_gradient =
      symmetric_active_one_electron_gradient_workspace != nullptr
          ? *symmetric_active_one_electron_gradient_workspace
          : local_symmetric_active_one_electron_gradient;
  write_symmetric_active_matrix_average_local(
      active_space_gradient_direction.active_orbital_overlap_gradient,
      n_active_orbitals,
      &symmetric_active_overlap_gradient);
  write_symmetric_active_matrix_average_local(
      active_space_gradient_direction.active_one_electron_gradient,
      n_active_orbitals,
      &symmetric_active_one_electron_gradient);

  ActiveSpaceMatrixBackpropagator matrix_backpropagator;
  const auto active_space_matrix_backpropagation_result =
      matrix_backpropagator.backpropagate(
          symmetric_active_overlap_gradient,
          symmetric_active_one_electron_gradient,
          input.orbital_preparation_input.active_orbital_overlap_matrix,
          ao_effective_one_electron_result.ao_effective_h1e,
          orbital_result.auxiliary_orbital_matrix,
          input.orbital_preparation_input.n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);

  ActiveSpaceTwoElectronBackpropagator active_space_two_electron_backpropagator;
  const auto active_space_two_electron_backpropagation_result =
      active_space_two_electron_backpropagator.backpropagate(
          active_space_gradient_direction.packed_active_two_electron_gradient,
          input.ao_integral_input.ao_two_electron_integral_values,
          input.ao_integral_input.ao_two_electron_integral_indices,
          orbital_result,
          active_space_two_electron_result,
          input.orbital_preparation_input.n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);

  AoEffectiveOneElectronBackpropagator ao_effective_one_electron_backpropagator;
  const auto ao_effective_one_electron_backpropagation_result =
      ao_effective_one_electron_backpropagator.backpropagate(
          active_space_matrix_backpropagation_result.ao_effective_one_electron_gradient,
          input.ao_integral_input);

  std::vector<double> total_inactive_density_gradient =
      ao_effective_one_electron_backpropagation_result.inactive_density_gradient;
  Eigen::MatrixXd total_active_auxiliary_gradient =
      active_space_matrix_backpropagation_result.active_auxiliary_orbital_gradient;
  total_active_auxiliary_gradient.noalias() +=
      active_space_two_electron_backpropagation_result
          .active_auxiliary_orbital_gradient;
  const Eigen::Map<const Eigen::MatrixXd> total_inactive_density_gradient_matrix(
      total_inactive_density_gradient.data(),
      input.orbital_preparation_input.n_basis_functions,
      input.orbital_preparation_input.n_basis_functions);
  const ExactCtxInternalInactiveChart internal_chart =
      build_exact_ctx_internal_inactive_chart(
          input.orbital_preparation_input,
          orbital_result,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);

  const std::vector<double> orbital_value_gradient =
      orbital_preparation_cache != nullptr
          ? backpropagate_active_space_orbital_gradient_cached(
                total_active_auxiliary_gradient,
                total_inactive_density_gradient_matrix,
                input.orbital_preparation_input,
                orbital_result,
                *orbital_preparation_cache)
          : internal_chart.enabled
              ? backpropagate_active_space_orbital_gradient_internal_chart(
                    total_active_auxiliary_gradient,
                    total_inactive_density_gradient_matrix,
                    input.orbital_preparation_input,
                    orbital_result
                        .physical_orbital_frame
                        .normalized_orbital_matrix,
                    internal_chart)
              : ActiveSpaceOrbitalBackpropagator().backpropagate(
                    total_active_auxiliary_gradient,
                    total_inactive_density_gradient_matrix,
                    input.orbital_preparation_input,
                    orbital_result)
                    .orbital_value_gradient;
  throw_if_nonfinite(
      orbital_value_gradient,
      "exact outer-response orbital-value gradient");
  return orbital_value_gradient;
}

bool has_active_matrix_gradient(
    const CppActiveSpaceSecondOrderContext& context,
    int n_active_orbitals) {
  const std::size_t active_matrix_size =
      n_active_orbitals * n_active_orbitals;
  return context.active_orbital_overlap_gradient.size() == active_matrix_size &&
      context.active_one_electron_gradient.size() == active_matrix_size &&
      context.packed_active_two_electron_gradient.size() ==
      packed_active_two_electron_integral_count(n_active_orbitals);
}

bool exact_ctx_stage1_analytic_core_enabled() {
  return true;
}

bool exact_ctx_outer_response_enabled() {
  const char* flag = std::getenv("XMVB_CPP_DISABLE_EXACT_CTX_OUTER_RESPONSE");
  if (flag == nullptr || flag[0] == '\0') {
    return true;
  }
  return std::strcmp(flag, "0") == 0 ||
      std::strcmp(flag, "false") == 0 ||
      std::strcmp(flag, "FALSE") == 0;
}

bool exact_ctx_orbital_preparation_cache_enabled() {
  const char* flag =
      std::getenv("XMVB_CPP_DISABLE_EXACT_CTX_ORBITAL_PREP_CACHE");
  if (flag == nullptr || flag[0] == '\0') {
    return true;
  }
  return std::strcmp(flag, "0") == 0 ||
      std::strcmp(flag, "false") == 0 ||
      std::strcmp(flag, "FALSE") == 0;
}

bool exact_ctx_workspace_exact_2e_enabled() {
  const char* flag = std::getenv("XMVB_CPP_DISABLE_EXACT_2E_WORKSPACE_HVP");
  if (flag == nullptr || flag[0] == '\0') {
    return true;
  }
  return std::strcmp(flag, "0") == 0 ||
      std::strcmp(flag, "false") == 0 ||
      std::strcmp(flag, "FALSE") == 0;
}

struct AcceptedOrbitalBackpropInputs {
  Eigen::MatrixXd total_active_auxiliary_gradient;
  std::vector<double> total_inactive_density_gradient;
};

AcceptedOrbitalBackpropInputs build_accepted_orbital_backprop_inputs(
    const CppActiveSpaceSecondOrderContext& accepted_point_context,
    const CppVbInput& input) {
  const int n_basis_functions = input.orbital_preparation_input.n_basis_functions;
  const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
  const int n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) /
      2;
  const std::size_t ao_matrix_size =
      n_basis_functions * n_basis_functions;

  const auto& orbital_result =
      accepted_point_context.prepared_active_space.orbital_result;
  const auto& ao_effective_one_electron_result =
      accepted_point_context.prepared_active_space.ao_effective_one_electron_result;
  if (orbital_result.auxiliary_orbital_matrix.size() != ao_matrix_size ||
      orbital_result.inactive_density_matrix.size() != ao_matrix_size ||
      ao_effective_one_electron_result.ao_effective_h1e.size() != ao_matrix_size ||
      input.ao_integral_input.ao_core_hamiltonian_matrix.size() != ao_matrix_size) {
    throw std::invalid_argument(
        "accepted-point orbital backprop inputs have inconsistent AO matrix sizes");
  }

  ActiveSpaceMatrixBackpropagator matrix_backpropagator;
  const auto matrix_backpropagation_result =
      matrix_backpropagator.backpropagate(
          accepted_point_context.active_orbital_overlap_gradient,
          accepted_point_context.active_one_electron_gradient,
          input.orbital_preparation_input.active_orbital_overlap_matrix,
          ao_effective_one_electron_result.ao_effective_h1e,
          orbital_result.auxiliary_orbital_matrix,
          n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);

  ActiveSpaceTwoElectronBackpropagator active_space_two_electron_backpropagator;
  const auto active_space_two_electron_backpropagation_result =
      active_space_two_electron_backpropagator.backpropagate(
          accepted_point_context.packed_active_two_electron_gradient,
          input.ao_integral_input.ao_two_electron_integral_values,
          input.ao_integral_input.ao_two_electron_integral_indices,
          orbital_result,
          accepted_point_context.prepared_active_space.active_space_two_electron_result,
          n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);

  std::vector<double> total_inactive_density_gradient =
      std::vector<double>(
          ao_effective_one_electron_result.ao_effective_h1e.data(),
          ao_effective_one_electron_result.ao_effective_h1e.data() +
              ao_effective_one_electron_result.ao_effective_h1e.size());
  const double* ao_core_hamiltonian_data =
      input.ao_integral_input.ao_core_hamiltonian_matrix.data();
  for (std::size_t index = 0;
       index < total_inactive_density_gradient.size();
       ++index) {
    total_inactive_density_gradient[index] +=
        ao_core_hamiltonian_data[index];
  }

  std::vector<double> total_ao_effective_one_electron_gradient =
      matrix_backpropagation_result.ao_effective_one_electron_gradient;
  for (std::size_t index = 0;
       index < total_ao_effective_one_electron_gradient.size();
       ++index) {
    total_ao_effective_one_electron_gradient[index] +=
        orbital_result.inactive_density_matrix.data()[index];
  }

  AoEffectiveOneElectronBackpropagator ao_backpropagator;
  const auto ao_effective_one_electron_backpropagation_result =
      ao_backpropagator.backpropagate(
          total_ao_effective_one_electron_gradient,
          input.ao_integral_input);
  for (std::size_t index = 0;
       index < total_inactive_density_gradient.size();
       ++index) {
    total_inactive_density_gradient[index] +=
        ao_effective_one_electron_backpropagation_result
            .inactive_density_gradient[index];
  }

  Eigen::MatrixXd total_active_auxiliary_gradient =
      matrix_backpropagation_result.active_auxiliary_orbital_gradient;
  total_active_auxiliary_gradient.noalias() +=
      active_space_two_electron_backpropagation_result
          .active_auxiliary_orbital_gradient;

  AcceptedOrbitalBackpropInputs result;
  result.total_active_auxiliary_gradient =
      std::move(total_active_auxiliary_gradient);
  result.total_inactive_density_gradient = std::move(total_inactive_density_gradient);
  return result;
}

std::vector<double> symmetrize_square_storage_average_local(
    const std::vector<double>& matrix_storage,
    int dimension) {
  const std::size_t expected_size = dimension * dimension;
  if (matrix_storage.size() != expected_size) {
    throw std::invalid_argument(
        "square matrix symmetrization requires dimension-aligned storage");
  }

  std::vector<double> symmetrized(matrix_storage.size(), 0.0);
  for (int column = 0; column < dimension; ++column) {
    for (int row = 0; row < dimension; ++row) {
      const double value =
          0.5 *
          (matrix_storage[column * dimension + row] +
           matrix_storage[row * dimension + column]);
      symmetrized[column * dimension + row] = value;
    }
  }
  return symmetrized;
}

}  // namespace

OppositeSpinMatrixBackwardContribution
build_pairwise_local_opposite_spin_matrix_backward_reference(
    const CppVbInput& input,
    const CppActiveSpaceSecondOrderContext& accepted_point_context,
    const DeterminantPairWeightTablesFromCoefficients& determinant_pair_weights,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  const ActiveSpaceGradientDirection pairwise_direction =
      build_pairwise_opposite_spin_local_active_space_gradient_direction_from_determinant_pair_weights(
          input,
          accepted_point_context,
          determinant_pair_weights,
          delta_active_orbital_overlap_matrix,
          delta_active_one_electron_matrix,
          delta_packed_active_two_electron_integrals);
  OppositeSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient =
      pairwise_direction.active_orbital_overlap_gradient;
  result.packed_active_two_electron_gradient =
      pairwise_direction.packed_active_two_electron_gradient;
  return result;
}

SameSpinMatrixBackwardContribution
build_pairwise_local_same_spin_matrix_backward_reference(
    const CppVbInput& input,
    const CppActiveSpaceSecondOrderContext& accepted_point_context,
    const DeterminantPairWeightTablesFromCoefficients& determinant_pair_weights,
    const std::vector<double>& delta_active_orbital_overlap_matrix,
    const std::vector<double>& delta_active_one_electron_matrix,
    const std::vector<double>& delta_packed_active_two_electron_integrals) {
  const ActiveSpaceGradientDirection pairwise_direction =
      build_pairwise_local_active_space_gradient_direction_from_determinant_pair_weights(
          input,
          accepted_point_context,
          determinant_pair_weights,
          true,
          delta_active_orbital_overlap_matrix,
          delta_active_one_electron_matrix,
          delta_packed_active_two_electron_integrals);
  const int n_active_orbitals =
      input.orbital_preparation_input.n_active_orbitals;
  SameSpinMatrixBackwardContribution result;
  result.active_orbital_overlap_gradient =
      symmetrize_square_storage_average_local(
          pairwise_direction.active_orbital_overlap_gradient,
          n_active_orbitals);
  result.active_one_electron_gradient =
      symmetrize_square_storage_average_local(
          pairwise_direction.active_one_electron_gradient,
          n_active_orbitals);
  result.packed_active_two_electron_gradient =
      pairwise_direction.packed_active_two_electron_gradient;
  return result;
}

ExactOrbitalSecondOrderOperator::ExactOrbitalSecondOrderOperator(
    std::shared_ptr<const CppActiveSpaceSecondOrderContext> accepted_point_context,
    const CppVbInput* current_input,
    SparseOrbitalParameterView parameter_view,
    const NonredundantOrbitalSpace* nonredundant_space)
    : accepted_point_context_(std::move(accepted_point_context)),
      current_input_(current_input),
      parameter_view_(std::move(parameter_view)),
      nonredundant_space_(nonredundant_space) {
  if (accepted_point_context_ == nullptr) {
    throw std::invalid_argument(
        "accepted-point second-order context must not be null");
  }
  if (current_input_ == nullptr) {
    throw std::invalid_argument("current_input must not be null");
  }
  if (nonredundant_space_ == nullptr) {
    throw std::invalid_argument("nonredundant_space must not be null");
  }
  if (supports_analytic_core_model()) {
    const int n_basis_functions =
        current_input_->orbital_preparation_input.n_basis_functions;
    const int n_active_orbitals =
        current_input_->orbital_preparation_input.n_active_orbitals;
    const int n_inactive_doubly_occupied_orbitals =
        (current_input_->orbital_preparation_input.n_total_electrons -
         current_input_->orbital_preparation_input.n_active_electrons) /
        2;
    const std::size_t ao_matrix_size =
        n_basis_functions * n_basis_functions;
    const std::size_t active_matrix_size =
        n_active_orbitals * n_active_orbitals;
    if (accepted_point_context_->prepared_active_space.orbital_result
                .auxiliary_orbital_matrix.size() == ao_matrix_size &&
        accepted_point_context_->active_orbital_overlap_gradient.size() ==
            active_matrix_size &&
        accepted_point_context_->active_one_electron_gradient.size() ==
            active_matrix_size) {
      const Eigen::Map<const Eigen::MatrixXd> basis_overlap(
          current_input_->orbital_preparation_input.active_orbital_overlap_matrix.data(),
          n_basis_functions,
          n_basis_functions);
      const Eigen::Map<const Eigen::MatrixXd> accepted_auxiliary_matrix(
          accepted_point_context_->prepared_active_space.orbital_result
              .auxiliary_orbital_matrix.data(),
          n_basis_functions,
          n_basis_functions);
      const Eigen::Map<const Eigen::MatrixXd> accepted_ao_effective_h1e(
          accepted_point_context_->prepared_active_space.ao_effective_one_electron_result
              .ao_effective_h1e.data(),
          n_basis_functions,
          n_basis_functions);
      accepted_active_auxiliary_orbitals_ =
          accepted_auxiliary_matrix.middleCols(
              n_inactive_doubly_occupied_orbitals,
              n_active_orbitals);
      accepted_basis_overlap_times_active_auxiliary_orbitals_ =
          basis_overlap * accepted_active_auxiliary_orbitals_;
      accepted_ao_effective_one_electron_times_active_auxiliary_orbitals_ =
          accepted_ao_effective_h1e * accepted_active_auxiliary_orbitals_;
      accepted_ao_effective_one_electron_transpose_times_active_auxiliary_orbitals_ =
          accepted_ao_effective_h1e.transpose() *
          accepted_active_auxiliary_orbitals_;
      const Eigen::Map<const Eigen::MatrixXd> accepted_sso_gradient(
          accepted_point_context_->active_orbital_overlap_gradient.data(),
          n_active_orbitals,
          n_active_orbitals);
      accepted_sso_gradient_symmetric_ =
          accepted_sso_gradient + accepted_sso_gradient.transpose();
      const Eigen::Map<const Eigen::MatrixXd> accepted_hho_gradient(
          accepted_point_context_->active_one_electron_gradient.data(),
          n_active_orbitals,
          n_active_orbitals);
      accepted_hho_gradient_symmetric_ =
          accepted_hho_gradient + accepted_hho_gradient.transpose();
      accepted_active_auxiliary_orbitals_times_hho_gradient_symmetric_ =
          accepted_active_auxiliary_orbitals_ *
          accepted_hho_gradient_symmetric_;
      const auto& accepted_active_space_two_electron_result =
          accepted_point_context_->prepared_active_space.active_space_two_electron_result;
      if (accepted_active_space_two_electron_result.dense_active_coefficients.size() != 0) {
        accepted_dense_active_coefficients_ =
            accepted_active_space_two_electron_result.dense_active_coefficients;
      } else {
        accepted_dense_active_coefficients_ = accepted_active_auxiliary_orbitals_;
      }
      zero_core_hamiltonian_ =
          Eigen::MatrixXd::Zero(n_basis_functions, n_basis_functions);
      accepted_exact_two_electron_cache_ =
          build_exact_packed_active_two_electron_adjoint_cache(
              accepted_point_context_->packed_active_two_electron_gradient,
              accepted_dense_active_coefficients_,
              current_input_->ao_integral_input,
              n_active_orbitals,
              &accepted_point_context_->prepared_active_space
                   .active_space_two_electron_result);
      has_accepted_exact_two_electron_cache_ = true;
    }

    const auto accepted_orbital_backprop_inputs =
        build_accepted_orbital_backprop_inputs(
            *accepted_point_context_,
            *current_input_);
    accepted_total_active_auxiliary_gradient_ =
        accepted_orbital_backprop_inputs.total_active_auxiliary_gradient;
    accepted_total_inactive_density_gradient_ =
        accepted_orbital_backprop_inputs.total_inactive_density_gradient;

    accepted_orbital_preparation_cache_ =
        std::make_unique<AcceptedOrbitalPreparationCache>(
            build_accepted_orbital_preparation_cache(
                current_input_->orbital_preparation_input,
                accepted_point_context_->prepared_active_space.orbital_result,
                accepted_total_active_auxiliary_gradient_,
                accepted_total_inactive_density_gradient_));

      if (accepted_point_context_->same_spin_pair_cache.enabled()) {
        structure_coefficient_blocks_ =
            build_structure_coefficient_blocks(
                current_input_->structure_data.determinant_to_structure_terms,
                current_input_->structure_data.n_structures,
                accepted_point_context_->same_spin_pair_cache.alpha_reuse_table,
                accepted_point_context_->same_spin_pair_cache.beta_reuse_table,
                true);
        accepted_outer_response_cache_ =
            build_accepted_outer_response_linear_response_cache(
                current_input_,
                accepted_point_context_.get(),
                &structure_coefficient_blocks_);
      }

    ExactCtxMemoryBreakdown memory_breakdown;
    memory_breakdown.add(
        "exact_operator.accepted_active_auxiliary_orbitals",
        exact_ctx_matrix_bytes(accepted_active_auxiliary_orbitals_));
    memory_breakdown.add(
        "exact_operator.accepted_basis_overlap_times_active_auxiliary_orbitals",
        exact_ctx_matrix_bytes(
            accepted_basis_overlap_times_active_auxiliary_orbitals_));
    memory_breakdown.add(
        "exact_operator.accepted_ao_effective_one_electron_times_active_auxiliary_orbitals",
        exact_ctx_matrix_bytes(
            accepted_ao_effective_one_electron_times_active_auxiliary_orbitals_));
    memory_breakdown.add(
        "exact_operator.accepted_ao_effective_one_electron_transpose_times_active_auxiliary_orbitals",
        exact_ctx_matrix_bytes(
            accepted_ao_effective_one_electron_transpose_times_active_auxiliary_orbitals_));
    memory_breakdown.add(
        "exact_operator.accepted_dense_active_coefficients",
        exact_ctx_matrix_bytes(accepted_dense_active_coefficients_));
    memory_breakdown.add(
        "exact_operator.accepted_sso_gradient_symmetric",
        exact_ctx_matrix_bytes(accepted_sso_gradient_symmetric_));
    memory_breakdown.add(
        "exact_operator.accepted_hho_gradient_symmetric",
        exact_ctx_matrix_bytes(accepted_hho_gradient_symmetric_));
    memory_breakdown.add(
        "exact_operator.accepted_active_auxiliary_orbitals_times_hho_gradient_symmetric",
        exact_ctx_matrix_bytes(
            accepted_active_auxiliary_orbitals_times_hho_gradient_symmetric_));
    memory_breakdown.add(
        "exact_operator.accepted_total_active_auxiliary_gradient",
        exact_ctx_matrix_bytes(accepted_total_active_auxiliary_gradient_));
    memory_breakdown.add(
        "exact_operator.accepted_total_inactive_density_gradient",
        exact_ctx_vector_capacity_bytes(
            accepted_total_inactive_density_gradient_));
    memory_breakdown.add(
        "exact_operator.zero_core_hamiltonian",
        exact_ctx_matrix_bytes(zero_core_hamiltonian_));
    append_exact_ctx_memory_breakdown(
        "exact_operator.accepted_exact_two_electron_cache",
        accepted_exact_two_electron_cache_,
        &memory_breakdown);
    append_exact_ctx_memory_breakdown(
        "exact_operator.accepted_exact_two_electron_apply_workspace",
        accepted_exact_two_electron_apply_workspace_,
        &memory_breakdown);
    append_exact_ctx_memory_breakdown(
        "exact_operator.outer_response_exact_two_electron_directional_workspace",
        outer_response_exact_two_electron_directional_workspace_,
        &memory_breakdown);
    append_exact_ctx_memory_breakdown(
        "exact_operator.structure_coefficient_blocks",
        structure_coefficient_blocks_,
        &memory_breakdown);
    append_exact_ctx_memory_breakdown(
        "exact_operator.accepted_outer_response_cache",
        accepted_outer_response_cache_,
        &memory_breakdown);
    maybe_log_exact_ctx_memory_breakdown(
        "exact_operator",
        memory_breakdown);
  }
}

ExactOrbitalSecondOrderOperator::~ExactOrbitalSecondOrderOperator() = default;

Eigen::VectorXd ExactOrbitalSecondOrderOperator::apply_reduced(
    const Eigen::VectorXd& reduced_direction) const {
  return apply_reduced_impl(
      reduced_direction,
      true,
      true,
      true,
      exact_ctx_orbital_preparation_cache_enabled());
}

Eigen::VectorXd ExactOrbitalSecondOrderOperator::apply_reduced_uncached(
    const Eigen::VectorXd& reduced_direction) const {
  return apply_reduced_impl(reduced_direction, true, true, true, false);
}

Eigen::VectorXd ExactOrbitalSecondOrderOperator::apply_reduced_cached(
    const Eigen::VectorXd& reduced_direction) const {
  return apply_reduced_impl(reduced_direction, true, true, true, true);
}

Eigen::VectorXd ExactOrbitalSecondOrderOperator::apply_reduced_without_outer_response(
    const Eigen::VectorXd& reduced_direction) const {
  return apply_reduced_impl(
      reduced_direction,
      true,
      true,
      false,
      exact_ctx_orbital_preparation_cache_enabled());
}

Eigen::VectorXd
ExactOrbitalSecondOrderOperator::apply_reduced_without_outer_response_uncached(
    const Eigen::VectorXd& reduced_direction) const {
  return apply_reduced_impl(reduced_direction, true, true, false, false);
}

Eigen::VectorXd
ExactOrbitalSecondOrderOperator::apply_reduced_without_outer_response_cached(
    const Eigen::VectorXd& reduced_direction) const {
  return apply_reduced_impl(reduced_direction, true, true, false, true);
}

Eigen::VectorXd ExactOrbitalSecondOrderOperator::apply_reduced_outer_response_only(
    const Eigen::VectorXd& reduced_direction) const {
  return apply_reduced_impl(
      reduced_direction,
      false,
      false,
      true,
      exact_ctx_orbital_preparation_cache_enabled());
}

Eigen::VectorXd
ExactOrbitalSecondOrderOperator::apply_reduced_outer_response_only_uncached(
    const Eigen::VectorXd& reduced_direction) const {
  return apply_reduced_impl(reduced_direction, false, false, true, false);
}

Eigen::VectorXd
ExactOrbitalSecondOrderOperator::apply_reduced_outer_response_only_cached(
    const Eigen::VectorXd& reduced_direction) const {
  return apply_reduced_impl(reduced_direction, false, false, true, true);
}

Eigen::VectorXd ExactOrbitalSecondOrderOperator::apply_reduced_core_direct_only(
    const Eigen::VectorXd& reduced_direction) const {
  return apply_reduced_impl(reduced_direction, true, false, false, false);
}

Eigen::VectorXd
ExactOrbitalSecondOrderOperator::apply_reduced_core_direct_only_cached(
    const Eigen::VectorXd& reduced_direction) const {
  return apply_reduced_impl(reduced_direction, true, false, false, true);
}

Eigen::VectorXd ExactOrbitalSecondOrderOperator::apply_reduced_fixed_upstream_only(
    const Eigen::VectorXd& reduced_direction) const {
  return apply_reduced_impl(reduced_direction, false, true, false, false);
}

Eigen::VectorXd
ExactOrbitalSecondOrderOperator::apply_reduced_fixed_upstream_only_cached(
    const Eigen::VectorXd& reduced_direction) const {
  return apply_reduced_impl(reduced_direction, false, true, false, true);
}

Eigen::VectorXd
ExactOrbitalSecondOrderOperator::apply_reduced_core_direct_only_uncached(
    const Eigen::VectorXd& reduced_direction) const {
  return apply_reduced_impl(reduced_direction, true, false, false, false);
}

Eigen::VectorXd
ExactOrbitalSecondOrderOperator::apply_reduced_fixed_upstream_only_uncached(
    const Eigen::VectorXd& reduced_direction) const {
  return apply_reduced_impl(reduced_direction, false, true, false, false);
}

Eigen::VectorXd ExactOrbitalSecondOrderOperator::apply_reduced_impl(
    const Eigen::VectorXd& reduced_direction,
    bool include_direct_core_response,
    bool include_fixed_upstream_pullback,
    bool include_outer_response,
    bool use_cached_orbital_preparation_cache) const {
  const auto apply_start_time = std::chrono::steady_clock::now();
  auto record_apply_wall_time = [&]() {
    ++apply_timing_totals_.apply_count;
    apply_timing_totals_.total_apply_wall_time_seconds +=
        elapsed_wall_time_seconds(apply_start_time);
  };
  const std::size_t apply_index = apply_timing_totals_.apply_count + 1;
  const bool log_apply_rss =
      exact_ctx_apply_rss_logging_enabled() &&
      apply_index <= exact_ctx_apply_rss_logging_max_applies();
  auto log_apply_rss_stage = [&](const char* stage) {
    if (!log_apply_rss) {
      return;
    }
    maybe_log_exact_ctx_apply_rss_stage(
        stage,
        apply_index,
        elapsed_wall_time_seconds(apply_start_time));
  };
  log_apply_rss_stage("enter");

  Eigen::VectorXd response =
      Eigen::VectorXd::Zero(reduced_direction.size());

  if (!supports_analytic_core_model()) {
    throw std::runtime_error(
        "exact_ctx analytic core HVP is unavailable for the current accepted point");
  }

  const int n_basis_functions =
      current_input_->orbital_preparation_input.n_basis_functions;
  const int n_active_orbitals =
      current_input_->orbital_preparation_input.n_active_orbitals;
  const int n_inactive_doubly_occupied_orbitals =
      (current_input_->orbital_preparation_input.n_total_electrons -
       current_input_->orbital_preparation_input.n_active_electrons) /
      2;
  const std::size_t ao_matrix_size =
      n_basis_functions * n_basis_functions;

  const auto core_setup_start_time = std::chrono::steady_clock::now();
  const Eigen::VectorXd packed_direction =
      nonredundant_space_->expand_step(reduced_direction);
  if (packed_direction.norm() == 0.0) {
    record_apply_wall_time();
    return Eigen::VectorXd::Zero(reduced_direction.size());
  }

  const AcceptedOrbitalPreparationCache* orbital_preparation_cache =
      use_cached_orbital_preparation_cache && accepted_orbital_preparation_cache_
          ? accepted_orbital_preparation_cache_.get()
          : nullptr;
  const ExactCtxInternalInactiveChart internal_chart =
      orbital_preparation_cache == nullptr
          ? build_exact_ctx_internal_inactive_chart(
                current_input_->orbital_preparation_input,
                accepted_point_context_->prepared_active_space.orbital_result,
                n_inactive_doubly_occupied_orbitals,
                n_active_orbitals)
          : ExactCtxInternalInactiveChart{};
  const auto dense_orbital_tangent_context =
      [&]() {
        if (orbital_preparation_cache != nullptr) {
          return build_dense_orbital_tangent_context_cached(
              current_input_->orbital_preparation_input,
              parameter_view_,
              packed_direction,
              *orbital_preparation_cache);
        }
        return build_dense_orbital_tangent_context(
            current_input_->orbital_preparation_input,
            parameter_view_,
            packed_direction,
            &internal_chart);
      }();
  const Eigen::VectorXd input_retract_tangent =
      include_fixed_upstream_pullback
          ? nonredundant_space_->expand_retract_input_tangent(
                current_input_->orbital_preparation_input,
                reduced_direction)
          : Eigen::VectorXd();
  const auto orbital_preparation_directional_result =
      orbital_preparation_cache
          ? build_orbital_preparation_directional_result_cached(
                current_input_->orbital_preparation_input,
                dense_orbital_tangent_context,
                n_inactive_doubly_occupied_orbitals,
                n_active_orbitals,
                *orbital_preparation_cache)
          : build_orbital_preparation_directional_result(
                current_input_->orbital_preparation_input,
                dense_orbital_tangent_context,
                n_inactive_doubly_occupied_orbitals,
                n_active_orbitals);

  const Eigen::Map<const Eigen::MatrixXd> basis_overlap(
      current_input_->orbital_preparation_input.active_orbital_overlap_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const Eigen::MatrixXd> ao_effective_h1e(
      accepted_point_context_->prepared_active_space.ao_effective_one_electron_result
          .ao_effective_h1e.data(),
      n_basis_functions,
      n_basis_functions);
  const auto& delta_dense_active_coefficients =
      orbital_preparation_directional_result.delta_active_auxiliary_orbitals;
  apply_timing_totals_.core_setup_wall_time_seconds +=
      elapsed_wall_time_seconds(core_setup_start_time);
  log_apply_rss_stage("after_core_setup");
  // `F11` and `delta F11` are explicitly symmetrized in the AO-H1E builder, so
  // the directional active-space matrix gradient only needs the two distinct
  // left contractions `delta F11 * T_active` and `F11 * delta T_active`.
  const Eigen::MatrixXd delta_active_times_hho_symmetric =
      orbital_preparation_directional_result.delta_active_auxiliary_orbitals *
      accepted_hho_gradient_symmetric_;
  Eigen::MatrixXd total_ao_effective_one_electron_direction =
      orbital_preparation_directional_result.delta_inactive_density;
  total_ao_effective_one_electron_direction.noalias() +=
      delta_active_times_hho_symmetric *
      accepted_active_auxiliary_orbitals_.transpose();
  const auto ao_effective_one_electron_fused_start_time =
      std::chrono::steady_clock::now();
  apply_fused_exact_ao_effective_one_electron_directional_operator(
      orbital_preparation_directional_result.delta_inactive_density,
      total_ao_effective_one_electron_direction,
      current_input_->ao_integral_input,
      current_input_->orbital_preparation_input,
      include_outer_response,
      &ao_h1e_symmetrized_gradient_workspace_,
      &ao_h1e_delta_h1e_workspace_,
      &ao_h1e_inactive_density_gradient_workspace_,
      &ao_h1e_partial_delta_h1e_workspaces_,
      &ao_h1e_partial_inactive_density_gradient_workspaces_);
  const Eigen::Map<const Eigen::MatrixXd> delta_ao_effective_h1e(
      ao_h1e_delta_h1e_workspace_.data(),
      n_basis_functions,
      n_basis_functions);
  apply_timing_totals_.ao_effective_one_electron_fused_wall_time_seconds +=
      elapsed_wall_time_seconds(ao_effective_one_electron_fused_start_time);
  log_apply_rss_stage("after_ao_h1e_fused");

  const bool compute_outer_response =
      include_outer_response && exact_ctx_outer_response_enabled();
  std::vector<double> combined_core_orbital_value_gradient;
  bool has_combined_core_orbital_value_gradient = false;
  Eigen::MatrixXd delta_ao_effective_h1e_times_active_auxiliary_orbitals;
  if (compute_outer_response) {
    // Full HVPs need `delta F11 * A_active` both for the direct-core
    // `delta F11 * A_active * G_hho` pullback and for outer-response
    // `delta HHO = A_active^T * delta F11 * A_active`.  Materializing it once
    // avoids repeating the same AO-by-active contraction in the two stages.
    delta_ao_effective_h1e_times_active_auxiliary_orbitals.noalias() =
        delta_ao_effective_h1e * accepted_active_auxiliary_orbitals_;
  }

  if (include_direct_core_response) {
    Eigen::MatrixXd delta_auxiliary_active_gradient =
        basis_overlap *
        orbital_preparation_directional_result.delta_active_auxiliary_orbitals *
        accepted_sso_gradient_symmetric_;
    if (compute_outer_response) {
      delta_auxiliary_active_gradient.noalias() +=
          delta_ao_effective_h1e_times_active_auxiliary_orbitals *
          accepted_hho_gradient_symmetric_;
    } else {
      delta_auxiliary_active_gradient.noalias() +=
          delta_ao_effective_h1e *
          accepted_active_auxiliary_orbitals_times_hho_gradient_symmetric_;
    }
    delta_auxiliary_active_gradient.noalias() +=
        ao_effective_h1e *
        delta_active_times_hho_symmetric;
    const auto active_two_electron_start_time =
        std::chrono::steady_clock::now();
    Eigen::MatrixXd dense_active_two_electron_gradient_direction_storage;
    const Eigen::MatrixXd* dense_active_two_electron_gradient_direction =
        nullptr;
    if (has_accepted_exact_two_electron_cache_ &&
        exact_ctx_workspace_exact_2e_enabled()) {
      apply_exact_packed_active_two_electron_adjoint_hessian_vector(
          accepted_exact_two_electron_cache_,
          delta_dense_active_coefficients,
          current_input_->ao_integral_input,
          &accepted_exact_two_electron_apply_workspace_,
          &accepted_exact_two_electron_apply_workspace_
               .dense_active_gradient_direction);
      dense_active_two_electron_gradient_direction =
          &accepted_exact_two_electron_apply_workspace_
               .dense_active_gradient_direction;
    } else {
      dense_active_two_electron_gradient_direction_storage =
          apply_exact_packed_active_two_electron_adjoint_hessian_vector(
              accepted_point_context_->packed_active_two_electron_gradient,
              accepted_dense_active_coefficients_,
              delta_dense_active_coefficients,
              current_input_->ao_integral_input,
              n_active_orbitals,
              &accepted_point_context_->prepared_active_space
                   .active_space_two_electron_result);
      dense_active_two_electron_gradient_direction =
          &dense_active_two_electron_gradient_direction_storage;
    }
    apply_timing_totals_.active_two_electron_wall_time_seconds +=
        elapsed_wall_time_seconds(active_two_electron_start_time);
    log_apply_rss_stage("after_active_two_electron");
    if (dense_active_two_electron_gradient_direction != nullptr) {
      delta_auxiliary_active_gradient.noalias() +=
          *dense_active_two_electron_gradient_direction;
    }

    Eigen::MatrixXd total_inactive_density_direction =
        delta_ao_effective_h1e;
    const Eigen::Map<const Eigen::MatrixXd> ao_backpropagated_inactive_density(
        ao_h1e_inactive_density_gradient_workspace_.data(),
        n_basis_functions,
        n_basis_functions);
    total_inactive_density_direction.noalias() +=
        ao_backpropagated_inactive_density;

    const auto orbital_backprop_start_time =
        std::chrono::steady_clock::now();
    const std::vector<double> orbital_value_gradient =
        orbital_preparation_cache != nullptr
            ? backpropagate_active_space_orbital_gradient_cached(
                  delta_auxiliary_active_gradient,
                  total_inactive_density_direction,
                  current_input_->orbital_preparation_input,
                  accepted_point_context_->prepared_active_space.orbital_result,
                  *orbital_preparation_cache)
            : internal_chart.enabled
                ? backpropagate_active_space_orbital_gradient_internal_chart(
                      delta_auxiliary_active_gradient,
                      total_inactive_density_direction,
                      current_input_->orbital_preparation_input,
                      accepted_point_context_
                          ->prepared_active_space
                          .orbital_result
                          .physical_orbital_frame
                          .normalized_orbital_matrix,
                      internal_chart)
                : ActiveSpaceOrbitalBackpropagator().backpropagate(
                      delta_auxiliary_active_gradient,
                      total_inactive_density_direction,
                      current_input_->orbital_preparation_input,
                      accepted_point_context_->prepared_active_space.orbital_result)
                      .orbital_value_gradient;
    add_orbital_value_gradient_in_place(
        &combined_core_orbital_value_gradient,
        orbital_value_gradient,
        "direct-core");
    has_combined_core_orbital_value_gradient = true;
    apply_timing_totals_.orbital_backprop_wall_time_seconds +=
        elapsed_wall_time_seconds(orbital_backprop_start_time);
    log_apply_rss_stage("after_direct_core_orbital_backprop");

  }

  if (include_fixed_upstream_pullback &&
      accepted_total_active_auxiliary_gradient_.rows() == n_basis_functions &&
      accepted_total_active_auxiliary_gradient_.cols() == n_active_orbitals &&
      accepted_total_inactive_density_gradient_.size() == ao_matrix_size) {
      const auto fixed_upstream_pullback_start_time =
          std::chrono::steady_clock::now();
      const std::vector<double> fixed_upstream_orbital_value_gradient =
              orbital_preparation_cache
                  ? apply_fixed_upstream_orbital_pullback_direction_cached(
                        current_input_->orbital_preparation_input,
                        dense_orbital_tangent_context,
                        accepted_total_active_auxiliary_gradient_,
                        orbital_preparation_directional_result
                            .basis_overlap_times_delta_active_orbitals,
                        accepted_total_inactive_density_gradient_,
                        input_retract_tangent,
                        *orbital_preparation_cache)
                  : apply_fixed_upstream_orbital_pullback_direction(
                        current_input_->orbital_preparation_input,
                        dense_orbital_tangent_context,
                        accepted_total_active_auxiliary_gradient_,
                        orbital_preparation_directional_result
                            .basis_overlap_times_delta_active_orbitals,
                        accepted_total_inactive_density_gradient_,
                        input_retract_tangent,
                        &internal_chart);
      add_orbital_value_gradient_in_place(
          &combined_core_orbital_value_gradient,
          fixed_upstream_orbital_value_gradient,
          "fixed-upstream");
      has_combined_core_orbital_value_gradient = true;
      apply_timing_totals_.fixed_upstream_pullback_wall_time_seconds +=
          elapsed_wall_time_seconds(fixed_upstream_pullback_start_time);
      log_apply_rss_stage("after_fixed_upstream_pullback");
  }

  if (compute_outer_response) {
    const auto outer_response_start_time =
        std::chrono::steady_clock::now();

    const auto active_space_integrals_start_time =
        std::chrono::steady_clock::now();
    build_active_space_directional_integrals(
        *current_input_,
        accepted_point_context_->prepared_active_space.active_space_two_electron_result,
        accepted_active_auxiliary_orbitals_,
        accepted_basis_overlap_times_active_auxiliary_orbitals_,
        accepted_ao_effective_one_electron_times_active_auxiliary_orbitals_,
        accepted_ao_effective_one_electron_transpose_times_active_auxiliary_orbitals_,
        orbital_preparation_directional_result.delta_active_auxiliary_orbitals,
        accepted_dense_active_coefficients_,
        delta_dense_active_coefficients,
        delta_ao_effective_h1e_times_active_auxiliary_orbitals,
        &outer_response_delta_active_orbital_overlap_matrix_workspace_,
        &outer_response_delta_active_one_electron_matrix_workspace_,
        &outer_response_exact_two_electron_directional_workspace_,
        &outer_response_delta_packed_active_two_electron_workspace_);
    apply_timing_totals_.outer_response_active_space_integrals_wall_time_seconds +=
        elapsed_wall_time_seconds(active_space_integrals_start_time);
    log_apply_rss_stage("after_outer_active_space_integrals");

    const auto structure_matrices_start_time =
        std::chrono::steady_clock::now();
    const auto projected_directional_structure_matrices =
        build_selected_state_projected_directional_structure_matrices(
            *current_input_,
            *accepted_point_context_,
            structure_coefficient_blocks_,
            outer_response_delta_active_orbital_overlap_matrix_workspace_,
            outer_response_delta_active_one_electron_matrix_workspace_,
            outer_response_delta_packed_active_two_electron_workspace_);
    apply_timing_totals_.outer_response_structure_matrices_wall_time_seconds +=
        elapsed_wall_time_seconds(structure_matrices_start_time);
    log_apply_rss_stage("after_outer_structure_matrices");

    const auto eigensystem_start_time =
        std::chrono::steady_clock::now();
    const auto directional_selected_state_response =
        accepted_outer_response_cache_
                .selected_state_eigen_response_operator
                .accepted_eigenvector_matrix_storage != nullptr
            ? accepted_outer_response_cache_
                  .selected_state_eigen_response_operator
                  .apply(projected_directional_structure_matrices)
            : build_selected_state_generalized_eigen_directional_response(
                  *accepted_point_context_,
                  projected_directional_structure_matrices);
    apply_timing_totals_.outer_response_eigensystem_wall_time_seconds +=
        elapsed_wall_time_seconds(eigensystem_start_time);
    log_apply_rss_stage("after_outer_eigensystem");

    const auto active_gradient_start_time =
        std::chrono::steady_clock::now();
    const SelectedStateDeterminantMatrices directional_selected_states =
        build_selected_state_determinant_matrices_from_selected_columns(
            current_input_->structure_data,
            directional_selected_state_response.delta_selected_eigenvector_matrix,
            accepted_point_context_->selected_state_indices,
            accepted_point_context_->normalized_state_weights,
            accepted_point_context_->same_spin_pair_cache);
    const auto directional_active_space_gradient =
        build_active_space_gradient_direction_from_outer_response(
            *current_input_,
            *accepted_point_context_,
            outer_response_delta_active_orbital_overlap_matrix_workspace_,
            outer_response_delta_active_one_electron_matrix_workspace_,
            outer_response_delta_packed_active_two_electron_workspace_,
            directional_selected_states,
            directional_selected_state_response.delta_selected_eigenvalues);
    validate_outer_response_active_gradient(
        directional_active_space_gradient);
    apply_timing_totals_.outer_response_active_gradient_wall_time_seconds +=
        elapsed_wall_time_seconds(active_gradient_start_time);
    log_apply_rss_stage("after_outer_active_gradient");

    const auto orbital_pullback_start_time =
        std::chrono::steady_clock::now();
    const std::vector<double> outer_response_orbital_value_gradient =
        build_orbital_value_gradient_from_active_space_gradient_direction(
            *current_input_,
            *accepted_point_context_,
            directional_active_space_gradient,
            orbital_preparation_cache,
            &outer_response_symmetric_active_overlap_gradient_workspace_,
            &outer_response_symmetric_active_one_electron_gradient_workspace_);
    add_orbital_value_gradient_in_place(
        &combined_core_orbital_value_gradient,
        outer_response_orbital_value_gradient,
        "outer-response");
    has_combined_core_orbital_value_gradient = true;
    apply_timing_totals_.outer_response_orbital_pullback_wall_time_seconds +=
        elapsed_wall_time_seconds(orbital_pullback_start_time);
    log_apply_rss_stage("after_outer_orbital_pullback");

    apply_timing_totals_.outer_response_wall_time_seconds +=
        elapsed_wall_time_seconds(outer_response_start_time);
  }

  if (has_combined_core_orbital_value_gradient) {
    // Full exact_ctx matvecs used to project the direct-core/fixed-upstream
    // pullback and the outer-response pullback separately.  Both contributions
    // live in the same raw sparse-orbital coefficient chart, so combining them
    // before `gather_from_full + project_reduced_gradient` removes one full
    // packed/reduced projection from every HVP apply without changing the
    // accepted-point nonredundant semantics.
    const Eigen::VectorXd packed_response =
        parameter_view_.gather_from_full(combined_core_orbital_value_gradient);
    response += nonredundant_space_->project_reduced_gradient(packed_response);
  }
  log_apply_rss_stage("before_return");

  record_apply_wall_time();
  return response;
}

ExactOrbitalSecondOrderOperator::DirectionalStructureDiagnostics
ExactOrbitalSecondOrderOperator::compute_directional_structure_diagnostics(
    const Eigen::VectorXd& reduced_direction) const {
  if (!supports_analytic_core_model()) {
    throw std::runtime_error(
        "analytic exact_ctx model is unavailable for structure diagnostics");
  }
  const int n_basis_functions =
      current_input_->orbital_preparation_input.n_basis_functions;
  const int n_active_orbitals =
      current_input_->orbital_preparation_input.n_active_orbitals;
  const int n_inactive_doubly_occupied_orbitals =
      (current_input_->orbital_preparation_input.n_total_electrons -
       current_input_->orbital_preparation_input.n_active_electrons) /
      2;

  const Eigen::VectorXd packed_direction =
      nonredundant_space_->expand_step(reduced_direction);
  const Eigen::VectorXd input_retract_tangent =
      nonredundant_space_->expand_retract_input_tangent(
          current_input_->orbital_preparation_input,
          reduced_direction);
  const ExactCtxInternalInactiveChart internal_chart =
      build_exact_ctx_internal_inactive_chart(
          current_input_->orbital_preparation_input,
          accepted_point_context_->prepared_active_space.orbital_result,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);
  const auto dense_orbital_tangent_context =
      build_dense_orbital_tangent_context(
          current_input_->orbital_preparation_input,
          parameter_view_,
          packed_direction,
          &internal_chart);
  const auto orbital_preparation_directional_result =
      build_orbital_preparation_directional_result(
          current_input_->orbital_preparation_input,
          dense_orbital_tangent_context,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);
  std::vector<double> delta_inactive_density_storage(
      orbital_preparation_directional_result.delta_inactive_density.data(),
      orbital_preparation_directional_result.delta_inactive_density.data() +
          orbital_preparation_directional_result.delta_inactive_density.size());
  AoEffectiveOneElectronBuilder ao_effective_one_electron_builder;
  const auto delta_ao_effective_result =
      ao_effective_one_electron_builder.build(
          delta_inactive_density_storage,
          zero_core_hamiltonian_,
          current_input_->ao_integral_input);
  const Eigen::Map<const Eigen::MatrixXd> delta_ao_effective_h1e(
      delta_ao_effective_result.ao_effective_h1e.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::MatrixXd delta_ao_effective_h1e_times_active_auxiliary_orbitals =
      delta_ao_effective_h1e * accepted_active_auxiliary_orbitals_;
  const auto& delta_dense_active_coefficients =
      orbital_preparation_directional_result.delta_active_auxiliary_orbitals;
  std::vector<double> directional_active_orbital_overlap_matrix;
  std::vector<double> directional_active_one_electron_matrix;
  std::vector<double> directional_packed_active_two_electron_integrals;
  ExactPackedActiveTwoElectronDirectionalDerivativeWorkspace
      directional_exact_two_electron_workspace;
  build_active_space_directional_integrals(
      *current_input_,
      accepted_point_context_->prepared_active_space.active_space_two_electron_result,
      accepted_active_auxiliary_orbitals_,
      accepted_basis_overlap_times_active_auxiliary_orbitals_,
      accepted_ao_effective_one_electron_times_active_auxiliary_orbitals_,
      accepted_ao_effective_one_electron_transpose_times_active_auxiliary_orbitals_,
      orbital_preparation_directional_result.delta_active_auxiliary_orbitals,
      accepted_dense_active_coefficients_,
      delta_dense_active_coefficients,
      delta_ao_effective_h1e_times_active_auxiliary_orbitals,
      &directional_active_orbital_overlap_matrix,
      &directional_active_one_electron_matrix,
      &directional_exact_two_electron_workspace,
      &directional_packed_active_two_electron_integrals);
  const auto directional_structure_matrices =
      build_directional_structure_matrices(
          *current_input_,
          *accepted_point_context_,
          structure_coefficient_blocks_,
          directional_active_orbital_overlap_matrix,
          directional_active_one_electron_matrix,
          directional_packed_active_two_electron_integrals);
  validate_outer_response_structure_matrices(
      directional_structure_matrices);
  const auto projected_directional_structure_matrices =
      build_selected_state_projected_directional_structure_matrices(
          *current_input_,
          *accepted_point_context_,
          structure_coefficient_blocks_,
          directional_active_orbital_overlap_matrix,
          directional_active_one_electron_matrix,
          directional_packed_active_two_electron_integrals);
  const auto directional_selected_state_response =
      build_selected_state_generalized_eigen_directional_response(
          *accepted_point_context_,
          projected_directional_structure_matrices);
  throw_if_nonfinite(
      directional_selected_state_response.delta_selected_eigenvector_matrix,
      "exact outer-response directional selected-state eigenvectors");
  throw_if_nonfinite(
      directional_selected_state_response.delta_selected_eigenvalues,
      "exact outer-response directional selected-state energies");
  const std::vector<double> directional_selected_state_energies =
      directional_selected_state_response.delta_selected_eigenvalues;
  const SelectedStateDeterminantMatrices directional_selected_states =
      build_selected_state_determinant_matrices_from_selected_columns(
          current_input_->structure_data,
          directional_selected_state_response.delta_selected_eigenvector_matrix,
          accepted_point_context_->selected_state_indices,
          accepted_point_context_->normalized_state_weights,
          accepted_point_context_->same_spin_pair_cache);
  const auto directional_active_space_gradient =
      build_active_space_gradient_direction_from_outer_response(
          *current_input_,
          *accepted_point_context_,
          directional_active_orbital_overlap_matrix,
          directional_active_one_electron_matrix,
          directional_packed_active_two_electron_integrals,
          directional_selected_states,
          directional_selected_state_energies);
  validate_outer_response_active_gradient(
      directional_active_space_gradient);

  DirectionalStructureDiagnostics diagnostics;
  diagnostics.active_orbital_overlap_matrix =
      directional_active_orbital_overlap_matrix;
  diagnostics.active_one_electron_matrix =
      directional_active_one_electron_matrix;
  diagnostics.packed_active_two_electron_integrals =
      directional_packed_active_two_electron_integrals;
  diagnostics.active_orbital_overlap_gradient =
      directional_active_space_gradient.active_orbital_overlap_gradient;
  diagnostics.active_one_electron_gradient =
      directional_active_space_gradient.active_one_electron_gradient;
  diagnostics.packed_active_two_electron_gradient =
      directional_active_space_gradient.packed_active_two_electron_gradient;
  diagnostics.overlap_matrix =
      directional_structure_matrices.overlap_matrix;
  diagnostics.hamiltonian_matrix =
      directional_structure_matrices.hamiltonian_matrix;
  return diagnostics;
}

ExactOrbitalSecondOrderOperator::DirectCoreDiagnostics
ExactOrbitalSecondOrderOperator::compute_direct_core_diagnostics(
    const Eigen::VectorXd& reduced_direction) const {
  if (!supports_analytic_core_model()) {
    throw std::runtime_error(
        "analytic exact_ctx model is unavailable for direct-core diagnostics");
  }

  const int n_basis_functions =
      current_input_->orbital_preparation_input.n_basis_functions;
  const int n_active_orbitals =
      current_input_->orbital_preparation_input.n_active_orbitals;
  const int n_inactive_doubly_occupied_orbitals =
      (current_input_->orbital_preparation_input.n_total_electrons -
       current_input_->orbital_preparation_input.n_active_electrons) /
      2;

  const Eigen::VectorXd packed_direction =
      nonredundant_space_->expand_step(reduced_direction);
  const Eigen::VectorXd input_retract_tangent =
      nonredundant_space_->expand_retract_input_tangent(
          current_input_->orbital_preparation_input,
          reduced_direction);
  const ExactCtxInternalInactiveChart internal_chart =
      build_exact_ctx_internal_inactive_chart(
          current_input_->orbital_preparation_input,
          accepted_point_context_->prepared_active_space.orbital_result,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);
  const auto dense_orbital_tangent_context =
      build_dense_orbital_tangent_context(
          current_input_->orbital_preparation_input,
          parameter_view_,
          packed_direction,
          &internal_chart);
  const auto orbital_preparation_directional_result =
      build_orbital_preparation_directional_result(
          current_input_->orbital_preparation_input,
          dense_orbital_tangent_context,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);

  const Eigen::Map<const Eigen::MatrixXd> basis_overlap(
      current_input_->orbital_preparation_input.active_orbital_overlap_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const Eigen::MatrixXd> ao_effective_h1e(
      accepted_point_context_->prepared_active_space.ao_effective_one_electron_result
          .ao_effective_h1e.data(),
      n_basis_functions,
      n_basis_functions);

  const Eigen::MatrixXd delta_active_times_hho_symmetric =
      orbital_preparation_directional_result.delta_active_auxiliary_orbitals *
      accepted_hho_gradient_symmetric_;
  Eigen::MatrixXd total_ao_effective_one_electron_direction =
      orbital_preparation_directional_result.delta_inactive_density;
  total_ao_effective_one_electron_direction.noalias() +=
      delta_active_times_hho_symmetric *
      accepted_active_auxiliary_orbitals_.transpose();

  Eigen::MatrixXd ao_h1e_symmetrized_gradient_workspace;
  std::vector<double> delta_ao_effective_h1e_storage;
  std::vector<double> ao_h1e_inactive_density_gradient_storage;
  apply_fused_exact_ao_effective_one_electron_directional_operator(
      orbital_preparation_directional_result.delta_inactive_density,
      total_ao_effective_one_electron_direction,
      current_input_->ao_integral_input,
      current_input_->orbital_preparation_input,
      false,
      &ao_h1e_symmetrized_gradient_workspace,
      &delta_ao_effective_h1e_storage,
      &ao_h1e_inactive_density_gradient_storage);
  const Eigen::Map<const Eigen::MatrixXd> delta_ao_effective_h1e(
      delta_ao_effective_h1e_storage.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const Eigen::MatrixXd> ao_backpropagated_inactive_density(
      ao_h1e_inactive_density_gradient_storage.data(),
      n_basis_functions,
      n_basis_functions);

  // These two dense AO/orbital blocks are the direct-core directional
  // derivative of `build_accepted_orbital_backprop_inputs()`. Comparing them
  // against finite differences isolates whether the mismatch lives in the
  // active-auxiliary pullback or in the inactive-density/AO-H1E chain.
  DirectCoreDiagnostics diagnostics;
  diagnostics.delta_matrix_active_auxiliary_gradient =
      basis_overlap *
      orbital_preparation_directional_result.delta_active_auxiliary_orbitals *
      accepted_sso_gradient_symmetric_;
  diagnostics.delta_matrix_active_auxiliary_gradient.noalias() +=
      delta_ao_effective_h1e *
      accepted_active_auxiliary_orbitals_times_hho_gradient_symmetric_;
  diagnostics.delta_matrix_active_auxiliary_gradient.noalias() +=
      ao_effective_h1e *
      delta_active_times_hho_symmetric;

  if (has_accepted_exact_two_electron_cache_ &&
      exact_ctx_workspace_exact_2e_enabled()) {
    apply_exact_packed_active_two_electron_adjoint_hessian_vector(
        accepted_exact_two_electron_cache_,
        orbital_preparation_directional_result.delta_active_auxiliary_orbitals,
        current_input_->ao_integral_input,
        &accepted_exact_two_electron_apply_workspace_,
        &accepted_exact_two_electron_apply_workspace_
             .dense_active_gradient_direction);
    diagnostics.delta_two_electron_active_auxiliary_gradient =
        accepted_exact_two_electron_apply_workspace_.dense_active_gradient_direction;
  } else {
    diagnostics.delta_two_electron_active_auxiliary_gradient =
        apply_exact_packed_active_two_electron_adjoint_hessian_vector(
            accepted_point_context_->packed_active_two_electron_gradient,
            accepted_dense_active_coefficients_,
            orbital_preparation_directional_result.delta_active_auxiliary_orbitals,
            current_input_->ao_integral_input,
            n_active_orbitals,
            &accepted_point_context_->prepared_active_space
                 .active_space_two_electron_result);
  }

  diagnostics.delta_total_active_auxiliary_gradient =
      diagnostics.delta_matrix_active_auxiliary_gradient;
  diagnostics.delta_total_active_auxiliary_gradient.noalias() +=
      diagnostics.delta_two_electron_active_auxiliary_gradient;
  diagnostics.delta_ao_effective_one_electron_matrix =
      delta_ao_effective_h1e;
  diagnostics.delta_ao_backpropagated_inactive_density_gradient =
      ao_backpropagated_inactive_density;
  diagnostics.delta_total_inactive_density_gradient =
      delta_ao_effective_h1e;
  diagnostics.delta_total_inactive_density_gradient.noalias() +=
      ao_backpropagated_inactive_density;

  const std::vector<double> orbital_value_gradient =
      internal_chart.enabled
          ? backpropagate_active_space_orbital_gradient_internal_chart(
                diagnostics.delta_total_active_auxiliary_gradient,
                diagnostics.delta_total_inactive_density_gradient,
                current_input_->orbital_preparation_input,
                accepted_point_context_
                    ->prepared_active_space
                    .orbital_result
                    .physical_orbital_frame
                    .normalized_orbital_matrix,
                internal_chart)
          : ActiveSpaceOrbitalBackpropagator().backpropagate(
                diagnostics.delta_total_active_auxiliary_gradient,
                diagnostics.delta_total_inactive_density_gradient,
                current_input_->orbital_preparation_input,
                accepted_point_context_->prepared_active_space.orbital_result)
                .orbital_value_gradient;
  const Eigen::VectorXd packed_response =
      parameter_view_.gather_from_full(orbital_value_gradient);
  diagnostics.reduced_response =
      nonredundant_space_->project_reduced_gradient(packed_response);
  return diagnostics;
}

ExactOrbitalSecondOrderOperator::FixedUpstreamDiagnostics
ExactOrbitalSecondOrderOperator::compute_fixed_upstream_diagnostics(
    const Eigen::VectorXd& reduced_direction) const {
  if (!supports_analytic_core_model()) {
    throw std::runtime_error(
        "analytic exact_ctx model is unavailable for fixed-upstream diagnostics");
  }

  const int n_active_orbitals =
      current_input_->orbital_preparation_input.n_active_orbitals;
  const int n_inactive_doubly_occupied_orbitals =
      (current_input_->orbital_preparation_input.n_total_electrons -
       current_input_->orbital_preparation_input.n_active_electrons) /
      2;

  const Eigen::VectorXd packed_direction =
      nonredundant_space_->expand_step(reduced_direction);
  const Eigen::VectorXd input_retract_tangent =
      nonredundant_space_->expand_retract_input_tangent(
          current_input_->orbital_preparation_input,
          reduced_direction);
  const ExactCtxInternalInactiveChart internal_chart =
      build_exact_ctx_internal_inactive_chart(
          current_input_->orbital_preparation_input,
          accepted_point_context_->prepared_active_space.orbital_result,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);
  const auto dense_orbital_tangent_context =
      build_dense_orbital_tangent_context(
          current_input_->orbital_preparation_input,
          parameter_view_,
          packed_direction,
          &internal_chart);
  const auto orbital_preparation_directional_result =
      build_orbital_preparation_directional_result(
          current_input_->orbital_preparation_input,
          dense_orbital_tangent_context,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);

  FixedUpstreamDiagnostics diagnostics;
  if (accepted_orbital_preparation_cache_ != nullptr &&
      accepted_orbital_preparation_cache_->has_pullback_cache) {
    diagnostics.original_orbital_gradient =
        accepted_orbital_preparation_cache_->original_orbital_gradient;
  } else {
    diagnostics.original_orbital_gradient =
        build_accepted_orbital_preparation_cache(
            current_input_->orbital_preparation_input,
            accepted_point_context_->prepared_active_space.orbital_result,
            accepted_total_active_auxiliary_gradient_,
            accepted_total_inactive_density_gradient_)
            .original_orbital_gradient;
  }
  diagnostics.orbital_value_gradient =
      apply_fixed_upstream_orbital_pullback_direction(
          current_input_->orbital_preparation_input,
          dense_orbital_tangent_context,
          accepted_total_active_auxiliary_gradient_,
          orbital_preparation_directional_result
              .basis_overlap_times_delta_active_orbitals,
          accepted_total_inactive_density_gradient_,
          input_retract_tangent,
          &internal_chart);
  diagnostics.packed_response =
      parameter_view_.gather_from_full(diagnostics.orbital_value_gradient);
  diagnostics.reduced_response =
      nonredundant_space_->project_reduced_gradient(
          diagnostics.packed_response);
  return diagnostics;
}

std::vector<NonredundantOrbitalSpace::BlockRotationDirection>
ExactOrbitalSecondOrderOperator::expand_block_rotation_directions(
    const Eigen::VectorXd& reduced_direction) const {
  return nonredundant_space_->expand_block_rotation_directions(
      reduced_direction);
}

bool ExactOrbitalSecondOrderOperator::supports_analytic_core_model() const noexcept {
  if (accepted_point_context_ == nullptr ||
      current_input_ == nullptr ||
      nonredundant_space_ == nullptr) {
    return false;
  }
  if (!exact_ctx_stage1_analytic_core_enabled()) {
    return false;
  }
  if (current_input_->standard_two_electron_mode ==
      StandardTwoElectronMode::ResolutionOfIdentity) {
    return false;
  }
  if (current_input_->ao_integral_input.ao_two_electron_integral_values.empty() ||
      current_input_->ao_integral_input.ao_two_electron_integral_indices.empty()) {
    return false;
  }
  const int n_active_orbitals =
      current_input_->orbital_preparation_input.n_active_orbitals;
  return n_active_orbitals > 0 &&
      has_active_matrix_gradient(*accepted_point_context_, n_active_orbitals);
}

ExactOrbitalSecondOrderOperator::Diagnostics
ExactOrbitalSecondOrderOperator::diagnostics() const {
  Diagnostics info;
  if (accepted_point_context_ == nullptr || nonredundant_space_ == nullptr) {
    return info;
  }
  info.supports_analytic_core_model = supports_analytic_core_model();
  info.outer_response_enabled = exact_ctx_outer_response_enabled();
  info.internal_inactive_chart_runtime_enabled =
      exact_ctx_internal_inactive_chart_runtime_enabled();
  info.used_reduced_curvature_diagonal =
      nonredundant_space_->has_reduced_curvature_diagonal();
  info.has_same_spin_matrix_form =
      accepted_point_context_->use_full_matrix_form_adjoint;
  info.has_opposite_spin_matrix_form =
      accepted_point_context_->use_matrix_form_opposite_spin;
  info.n_selected_states =
      static_cast<int>(accepted_point_context_->selected_state_indices.size());
  info.n_active_orbitals = accepted_point_context_->n_active_orbitals;
  info.n_blocks =
      static_cast<int>(
          nonredundant_space_->expand_block_rotation_directions(
              Eigen::VectorXd::Zero(nonredundant_space_->reduced_size()))
              .size());
  info.uses_internal_inactive_chart =
      build_exact_ctx_internal_inactive_chart(
          current_input_->orbital_preparation_input,
          accepted_point_context_->prepared_active_space.orbital_result,
          (current_input_->orbital_preparation_input.n_total_electrons -
           current_input_->orbital_preparation_input.n_active_electrons) /
              2,
          current_input_->orbital_preparation_input.n_active_orbitals)
          .enabled;
  info.apply_count = apply_timing_totals_.apply_count;
  info.total_apply_wall_time_seconds =
      apply_timing_totals_.total_apply_wall_time_seconds;
  info.core_setup_wall_time_seconds =
      apply_timing_totals_.core_setup_wall_time_seconds;
  info.ao_effective_one_electron_build_wall_time_seconds =
      apply_timing_totals_.ao_effective_one_electron_build_wall_time_seconds;
  info.ao_effective_one_electron_fused_wall_time_seconds =
      apply_timing_totals_.ao_effective_one_electron_fused_wall_time_seconds;
  info.active_two_electron_wall_time_seconds =
      apply_timing_totals_.active_two_electron_wall_time_seconds;
  info.ao_effective_one_electron_backprop_wall_time_seconds =
      apply_timing_totals_.ao_effective_one_electron_backprop_wall_time_seconds;
  info.orbital_backprop_wall_time_seconds =
      apply_timing_totals_.orbital_backprop_wall_time_seconds;
  info.fixed_upstream_pullback_wall_time_seconds =
      apply_timing_totals_.fixed_upstream_pullback_wall_time_seconds;
  info.outer_response_wall_time_seconds =
      apply_timing_totals_.outer_response_wall_time_seconds;
  info.outer_response_active_space_integrals_wall_time_seconds =
      apply_timing_totals_.outer_response_active_space_integrals_wall_time_seconds;
  info.outer_response_structure_matrices_wall_time_seconds =
      apply_timing_totals_.outer_response_structure_matrices_wall_time_seconds;
  info.outer_response_eigensystem_wall_time_seconds =
      apply_timing_totals_.outer_response_eigensystem_wall_time_seconds;
  info.outer_response_pair_weights_wall_time_seconds =
      apply_timing_totals_.outer_response_pair_weights_wall_time_seconds;
  info.outer_response_active_gradient_wall_time_seconds =
      apply_timing_totals_.outer_response_active_gradient_wall_time_seconds;
  info.outer_response_orbital_pullback_wall_time_seconds =
      apply_timing_totals_.outer_response_orbital_pullback_wall_time_seconds;
  return info;
}

}  // namespace xmvb::vb
