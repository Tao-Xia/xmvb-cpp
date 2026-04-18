#include "vb/pdft/selected_state_exact_physical_on_top_pair_density_builder.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vb/matrices/determinant_overlap_resolver.hpp"
#include "vb/matrices/same_spin_pair_cache.hpp"
#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/scf/cpp_active_space_second_order_context.hpp"
#include "vb/scf/selected_state_determinant_matrices.hpp"

namespace xmvb::vb {

namespace {


constexpr double kUnitWeightTolerance = 1.0e-10;
constexpr double kNormalizationTolerance = 1.0e-12;

struct SelectedStateCoefficientContext {
  SelectedStateDeterminantMatrices selected_state_matrices;
  SpinDeterminantReuseTable alpha_reuse_table;
  SpinDeterminantReuseTable beta_reuse_table;
};

struct PhysicalSpinSupportDeterminant {
  std::vector<int> full_occupied_orbitals;
  Eigen::MatrixXd occupied_physical_orbitals;
};

struct PhysicalSpinBuffers {
  Eigen::MatrixXd overlap_matrix;
  int n_support_determinants = 0;
  int n_occupied_orbitals = 0;
  std::vector<Eigen::MatrixXd> support_occupied_physical_orbitals;
  std::vector<Eigen::MatrixXd> ordered_pair_first_order_cofactors;
};

void validate_single_state_gradient_result(
    const CppActiveSpaceGradientResult& gradient_result) {
  if (gradient_result.scf_result.selected_state_indices.size() != 1 ||
      gradient_result.scf_result.state_average_weights.size() != 1) {
    throw std::invalid_argument(
        "exact physical on-top pair-density build requires exactly one selected state");
  }
  const double state_weight =
      gradient_result.scf_result.state_average_weights.front();
  if (std::abs(state_weight - 1.0) > kUnitWeightTolerance) {
    throw std::invalid_argument(
        "exact physical on-top pair-density build requires a unit selected-state weight");
  }
}

double frobenius_inner_product(
    const Eigen::MatrixXd& left,
    const Eigen::MatrixXd& right) {
  if (left.rows() != right.rows() || left.cols() != right.cols()) {
    throw std::invalid_argument("Frobenius inner product size mismatch");
  }
  return left.cwiseProduct(right).sum();
}

std::size_t ordered_pair_storage_index(
    int left_index,
    int right_index,
    int n_support_determinants) {
  return xmvb::to_size(left_index) * n_support_determinants + right_index;
}

SelectedStateCoefficientContext materialize_selected_state_coefficients(
    const CppVbInput& input,
    const CppActiveSpaceGradientResult& gradient_result) {
  SelectedStateCoefficientContext context;
  if (gradient_result.second_order_context != nullptr) {
    context.alpha_reuse_table =
        gradient_result.second_order_context->same_spin_pair_cache.alpha_reuse_table;
    context.beta_reuse_table =
        gradient_result.second_order_context->same_spin_pair_cache.beta_reuse_table;
    if (!gradient_result.second_order_context->selected_state_matrices.states.empty()) {
      context.selected_state_matrices =
          gradient_result.second_order_context->selected_state_matrices;
      return context;
    }
  }

  context.alpha_reuse_table =
      build_spin_determinant_reuse_table(input.structure_data.alpha_det);
  context.beta_reuse_table =
      build_spin_determinant_reuse_table(input.structure_data.beta_det);

  SameSpinPairCacheContext reuse_context;
  reuse_context.alpha_reuse_table = context.alpha_reuse_table;
  reuse_context.beta_reuse_table = context.beta_reuse_table;
  context.selected_state_matrices =
      build_selected_state_determinant_matrices_from_normalized_weights(
          input.structure_data,
          gradient_result.scf_result.eigenvector_matrix,
          gradient_result.scf_result.selected_state_indices,
          gradient_result.scf_result.state_average_weights,
          reuse_context);
  return context;
}

std::vector<int> build_full_spin_occupied_orbitals(
    const std::vector<int>& active_spin_occupied_orbitals,
    int n_inactive_doubly_occupied_orbitals) {
  if (n_inactive_doubly_occupied_orbitals < 0) {
    throw std::invalid_argument(
        "n_inactive_doubly_occupied_orbitals must be non-negative");
  }

  std::vector<int> full_occupied_orbitals;
  full_occupied_orbitals.reserve(
      xmvb::to_size(n_inactive_doubly_occupied_orbitals) +
      active_spin_occupied_orbitals.size());
  for (int inactive_orbital_index = 0;
       inactive_orbital_index < n_inactive_doubly_occupied_orbitals;
       ++inactive_orbital_index) {
    full_occupied_orbitals.push_back(inactive_orbital_index);
  }
  for (const int active_orbital_index : active_spin_occupied_orbitals) {
    if (active_orbital_index < 0) {
      throw std::invalid_argument("active occupied orbital index must be non-negative");
    }
    full_occupied_orbitals.push_back(
        n_inactive_doubly_occupied_orbitals + active_orbital_index);
  }
  return full_occupied_orbitals;
}

Eigen::MatrixXd extract_occupied_physical_orbitals(
    const Eigen::Ref<const Eigen::MatrixXd>& full_physical_orbital_matrix,
    const std::vector<int>& full_occupied_orbitals) {
  Eigen::MatrixXd occupied_physical_orbitals(
      full_physical_orbital_matrix.rows(),
      static_cast<int>(full_occupied_orbitals.size()));
  for (int occupied_index = 0;
       occupied_index < static_cast<int>(full_occupied_orbitals.size());
       ++occupied_index) {
    const int orbital_index = full_occupied_orbitals[xmvb::to_size(occupied_index)];
    if (orbital_index < 0 || orbital_index >= full_physical_orbital_matrix.cols()) {
      throw std::out_of_range("physical occupied orbital index is out of range");
    }
    occupied_physical_orbitals.col(occupied_index) =
        full_physical_orbital_matrix.col(orbital_index);
  }
  return occupied_physical_orbitals;
}

PhysicalSpinBuffers build_physical_spin_buffers(
    const std::vector<std::vector<int>>& unique_spin_determinants,
    const std::vector<int>& support_indices,
    const Eigen::Ref<const Eigen::MatrixXd>& full_physical_orbital_matrix,
    const std::vector<double>& full_physical_orbital_overlap_matrix,
    int n_total_orbitals,
    int n_inactive_doubly_occupied_orbitals,
    const DeterminantOverlapResolver& overlap_resolver) {
  PhysicalSpinBuffers buffers;
  buffers.n_support_determinants = static_cast<int>(support_indices.size());
  if (buffers.n_support_determinants <= 0) {
    throw std::invalid_argument("selected-state spin support must not be empty");
  }

  std::vector<PhysicalSpinSupportDeterminant> support_determinants;
  support_determinants.reserve(support_indices.size());
  for (const int support_index : support_indices) {
    if (support_index < 0 ||
        support_index >= static_cast<int>(unique_spin_determinants.size())) {
      throw std::out_of_range("support determinant index is out of range");
    }
    const auto full_occupied_orbitals =
        build_full_spin_occupied_orbitals(
            unique_spin_determinants[xmvb::to_size(support_index)],
            n_inactive_doubly_occupied_orbitals);
    PhysicalSpinSupportDeterminant support_determinant;
    support_determinant.full_occupied_orbitals = full_occupied_orbitals;
    support_determinant.occupied_physical_orbitals =
        extract_occupied_physical_orbitals(
            full_physical_orbital_matrix,
            full_occupied_orbitals);
    support_determinants.push_back(std::move(support_determinant));
  }

  buffers.n_occupied_orbitals =
      support_determinants.front().occupied_physical_orbitals.cols();
  buffers.support_occupied_physical_orbitals.reserve(
      xmvb::to_size(buffers.n_support_determinants));
  buffers.ordered_pair_first_order_cofactors.assign(
      xmvb::to_size(buffers.n_support_determinants) *
          buffers.n_support_determinants,
      Eigen::MatrixXd::Zero(
          buffers.n_occupied_orbitals,
          buffers.n_occupied_orbitals));
  buffers.overlap_matrix =
      Eigen::MatrixXd::Zero(buffers.n_support_determinants, buffers.n_support_determinants);

  for (int support_local_index = 0;
       support_local_index < buffers.n_support_determinants;
       ++support_local_index) {
    buffers.support_occupied_physical_orbitals.push_back(
        support_determinants[xmvb::to_size(support_local_index)]
            .occupied_physical_orbitals);
  }

  // Store each ordered determinant-pair cofactor in the physical full-occupied
  // frame. Later on-top evaluation only needs AO values at one point, the
  // support coefficient matrix, and these first cofactors to reconstruct the
  // exact opposite-spin transition-density product.
  for (int left_local = 0;
       left_local < buffers.n_support_determinants;
       ++left_local) {
    for (int right_local = left_local;
         right_local < buffers.n_support_determinants;
         ++right_local) {
      const auto& left_determinant =
          support_determinants[xmvb::to_size(left_local)];
      const auto& right_determinant =
          support_determinants[xmvb::to_size(right_local)];
      const std::vector<double> overlap_submatrix =
          build_overlap_submatrix(
              left_determinant.full_occupied_orbitals,
              right_determinant.full_occupied_orbitals,
              full_physical_orbital_overlap_matrix,
              n_total_orbitals);
      const DeterminantOverlapResult overlap_result =
          overlap_resolver.resolve(
              overlap_submatrix,
              static_cast<int>(left_determinant.full_occupied_orbitals.size()));
      buffers.overlap_matrix(right_local, left_local) =
          overlap_result.overlap_determinant;
      buffers.overlap_matrix(left_local, right_local) =
          overlap_result.overlap_determinant;
      if (overlap_result.nullity >= 2) {
        continue;
      }

      const Eigen::MatrixXd first_order_cofactor =
          calc_cofactor_1st(overlap_result);
      buffers.ordered_pair_first_order_cofactors[ordered_pair_storage_index(
          left_local,
          right_local,
          buffers.n_support_determinants)] = first_order_cofactor;

      if (right_local == left_local) {
        continue;
      }
      buffers.ordered_pair_first_order_cofactors[ordered_pair_storage_index(
          right_local,
          left_local,
          buffers.n_support_determinants)] = first_order_cofactor.transpose();
    }
  }

  return buffers;
}

Eigen::MatrixXd build_transition_density_scalar_matrix(
    const std::vector<Eigen::MatrixXd>& support_occupied_physical_orbitals,
    const std::vector<Eigen::MatrixXd>& ordered_pair_first_order_cofactors,
    const Eigen::Ref<const Eigen::VectorXd>& ao_values) {
  const int n_support_determinants =
      static_cast<int>(support_occupied_physical_orbitals.size());
  const int n_occupied_orbitals =
      support_occupied_physical_orbitals.front().cols();

  Eigen::MatrixXd occupied_value_projections =
      Eigen::MatrixXd::Zero(n_occupied_orbitals, n_support_determinants);
  for (int support_index = 0;
       support_index < n_support_determinants;
       ++support_index) {
    occupied_value_projections.col(support_index).noalias() =
        support_occupied_physical_orbitals[xmvb::to_size(support_index)].transpose() *
        ao_values;
  }

  Eigen::MatrixXd transition_density_scalars =
      Eigen::MatrixXd::Zero(n_support_determinants, n_support_determinants);
  for (int left_index = 0;
       left_index < n_support_determinants;
       ++left_index) {
    for (int right_index = 0;
         right_index < n_support_determinants;
         ++right_index) {
      const Eigen::Map<const Eigen::MatrixXd> first_order_cofactor(
          ordered_pair_first_order_cofactors[ordered_pair_storage_index(
              left_index,
              right_index,
              n_support_determinants)].data(),
          n_occupied_orbitals,
          n_occupied_orbitals);
      transition_density_scalars(right_index, left_index) =
          occupied_value_projections.col(right_index).dot(
              first_order_cofactor * occupied_value_projections.col(left_index));
    }
  }
  return transition_density_scalars;
}

Eigen::MatrixXd build_transition_density_scalar_block(
    const std::vector<Eigen::MatrixXd>& support_occupied_physical_orbitals,
    const std::vector<Eigen::MatrixXd>& ordered_pair_first_order_cofactors,
    const Eigen::Ref<const Eigen::MatrixXd>& ao_probe_matrix) {
  const int n_support_determinants =
      static_cast<int>(support_occupied_physical_orbitals.size());
  const int n_occupied_orbitals =
      support_occupied_physical_orbitals.front().cols();
  std::vector<Eigen::MatrixXd> occupied_value_projections;
  occupied_value_projections.reserve(xmvb::to_size(n_support_determinants));
  for (int support_index = 0;
       support_index < n_support_determinants;
       ++support_index) {
    Eigen::MatrixXd support_projection(n_occupied_orbitals, ao_probe_matrix.cols());
    support_projection.noalias() =
        support_occupied_physical_orbitals[xmvb::to_size(support_index)].transpose() *
        ao_probe_matrix;
    occupied_value_projections.push_back(std::move(support_projection));
  }

  // For each ordered determinant pair `(L, R)` and each probe column `g`,
  // this block routine evaluates the transition-density scalar
  //
  // `D_{R,L}(g) = p_R(g)^T C_{L,R}^{(1)} p_L(g)`,
  //
  // where `p_L(g)` is the occupied-orbital value projection at probe `g`. The
  // flattened storage preserves the existing ordered-pair convention so that
  // one probe column can be reinterpreted as the same matrix used by the
  // scalar evaluator.
  Eigen::MatrixXd flattened_transition_density_scalars =
      Eigen::MatrixXd::Zero(n_support_determinants * n_support_determinants,
                   ao_probe_matrix.cols());
  for (int left_index = 0;
       left_index < n_support_determinants;
       ++left_index) {
    const Eigen::MatrixXd& left_projection =
        occupied_value_projections[xmvb::to_size(left_index)];
    for (int right_index = 0;
         right_index < n_support_determinants;
         ++right_index) {
      const Eigen::Map<const Eigen::MatrixXd> first_order_cofactor(
          ordered_pair_first_order_cofactors[ordered_pair_storage_index(
              left_index,
              right_index,
              n_support_determinants)].data(),
          n_occupied_orbitals,
          n_occupied_orbitals);
      const Eigen::MatrixXd transformed_left_projection =
          first_order_cofactor * left_projection;
      const Eigen::Index pair_index =
          static_cast<Eigen::Index>(ordered_pair_storage_index(
              left_index,
              right_index,
              n_support_determinants));
      flattened_transition_density_scalars.row(pair_index) =
          occupied_value_projections[xmvb::to_size(right_index)]
              .cwiseProduct(transformed_left_projection)
              .colwise()
              .sum();
    }
  }
  return flattened_transition_density_scalars;
}

SelectedStateExactPhysicalOnTopPairDensityContext
assemble_selected_state_exact_physical_on_top_pair_density_context(
    const CppVbInput& input,
    const CppActiveSpaceGradientResult& gradient_result) {
  validate_single_state_gradient_result(gradient_result);

  const int n_basis_functions =
      input.orbital_preparation_input.n_basis_functions;
  const int n_total_orbitals =
      input.orbital_preparation_input.n_orbitals;
  const int n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) / 2;
  if (n_basis_functions <= 0 || n_total_orbitals <= 0) {
    throw std::invalid_argument(
        "exact physical on-top pair-density build requires positive basis and orbital dimensions");
  }
  if (n_inactive_doubly_occupied_orbitals < 0 ||
      n_inactive_doubly_occupied_orbitals > n_total_orbitals) {
    throw std::invalid_argument("invalid inactive orbital count");
  }

  const std::size_t ao_matrix_size =
      xmvb::to_size(n_basis_functions) * n_basis_functions;
  const std::size_t physical_orbital_matrix_size =
      xmvb::to_size(n_basis_functions) * n_total_orbitals;
  if (input.orbital_preparation_input.active_orbital_overlap_matrix.size() !=
          ao_matrix_size ||
      gradient_result.orbital_preparation_result.physical_orbital_frame
              .normalized_orbital_matrix.size() !=
          physical_orbital_matrix_size) {
    throw std::invalid_argument(
        "exact physical on-top pair-density build received inconsistent AO or physical-orbital dimensions");
  }

  const SelectedStateCoefficientContext coefficient_context =
      materialize_selected_state_coefficients(input, gradient_result);
  if (coefficient_context.selected_state_matrices.states.size() != 1) {
    throw std::invalid_argument(
        "exact physical on-top pair-density build requires one selected-state coefficient bundle");
  }
  const auto& selected_state =
      coefficient_context.selected_state_matrices.states.front();

  const Eigen::Map<const Eigen::MatrixXd> ao_overlap_matrix(
      input.orbital_preparation_input.active_orbital_overlap_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const Eigen::MatrixXd> full_physical_orbital_matrix(
      gradient_result.orbital_preparation_result.physical_orbital_frame
          .normalized_orbital_matrix.data(),
      n_basis_functions,
      n_total_orbitals);
  const Eigen::MatrixXd full_physical_orbital_overlap =
      full_physical_orbital_matrix.transpose() *
      ao_overlap_matrix *
      full_physical_orbital_matrix;
  std::vector<double> full_physical_orbital_overlap_storage(
      full_physical_orbital_overlap.data(),
      full_physical_orbital_overlap.data() +
          full_physical_orbital_overlap.size());

  const DeterminantOverlapResolver overlap_resolver;
  const PhysicalSpinBuffers alpha_buffers =
      build_physical_spin_buffers(
          coefficient_context.alpha_reuse_table.unique_determinants,
          selected_state.alpha_support,
          full_physical_orbital_matrix,
          full_physical_orbital_overlap_storage,
          n_total_orbitals,
          n_inactive_doubly_occupied_orbitals,
          overlap_resolver);
  const PhysicalSpinBuffers beta_buffers =
      build_physical_spin_buffers(
          coefficient_context.beta_reuse_table.unique_determinants,
          selected_state.beta_support,
          full_physical_orbital_matrix,
          full_physical_orbital_overlap_storage,
          n_total_orbitals,
          n_inactive_doubly_occupied_orbitals,
          overlap_resolver);

  const Eigen::MatrixXd& coefficient_matrix = selected_state.local_coefficient_matrix;
  const Eigen::MatrixXd alpha_normalization_weights =
      coefficient_matrix *
      beta_buffers.overlap_matrix *
      coefficient_matrix.transpose();
  const Eigen::MatrixXd beta_normalization_weights =
      coefficient_matrix.transpose() *
      alpha_buffers.overlap_matrix *
      coefficient_matrix;
  const double alpha_normalization =
      frobenius_inner_product(
          alpha_normalization_weights,
          alpha_buffers.overlap_matrix);
  const double beta_normalization =
      frobenius_inner_product(
          beta_normalization_weights,
          beta_buffers.overlap_matrix);
  const double normalization =
      0.5 * (alpha_normalization + beta_normalization);
  if (!std::isfinite(normalization) ||
      std::abs(normalization) <= kNormalizationTolerance) {
    throw std::runtime_error(
        "exact physical on-top pair-density normalization is too small or non-finite");
  }

  SelectedStateExactPhysicalOnTopPairDensityContext result;
  result.state_index = gradient_result.scf_result.selected_state_indices.front();
  result.n_basis_functions = n_basis_functions;
  result.n_alpha_support_determinants = alpha_buffers.n_support_determinants;
  result.n_beta_support_determinants = beta_buffers.n_support_determinants;
  result.n_alpha_occupied_orbitals = alpha_buffers.n_occupied_orbitals;
  result.n_beta_occupied_orbitals = beta_buffers.n_occupied_orbitals;
  result.selected_state_overlap_normalization = normalization;
  result.alpha_beta_normalization_abs_error =
      std::abs(alpha_normalization - beta_normalization);
  result.local_coefficient_matrix = coefficient_matrix;
  result.alpha_support_occupied_physical_orbitals =
      alpha_buffers.support_occupied_physical_orbitals;
  result.beta_support_occupied_physical_orbitals =
      beta_buffers.support_occupied_physical_orbitals;
  result.alpha_pair_first_order_cofactors =
      alpha_buffers.ordered_pair_first_order_cofactors;
  result.beta_pair_first_order_cofactors =
      beta_buffers.ordered_pair_first_order_cofactors;
  return result;
}

void validate_exact_physical_on_top_pair_density_context(
    const SelectedStateExactPhysicalOnTopPairDensityContext& context) {
  if (context.n_basis_functions <= 0 ||
      context.n_alpha_support_determinants <= 0 ||
      context.n_beta_support_determinants <= 0 ||
      context.n_alpha_occupied_orbitals <= 0 ||
      context.n_beta_occupied_orbitals <= 0) {
    throw std::invalid_argument(
        "exact physical on-top pair-density context dimensions must be positive");
  }
  const std::size_t expected_coefficient_matrix_size =
      xmvb::to_size(context.n_alpha_support_determinants) *
      context.n_beta_support_determinants;
  if (context.local_coefficient_matrix.rows() != context.n_alpha_support_determinants ||
      context.local_coefficient_matrix.cols() != context.n_beta_support_determinants ||
      xmvb::to_size(context.local_coefficient_matrix.size()) !=
          expected_coefficient_matrix_size) {
    throw std::invalid_argument("selected-state coefficient matrix size mismatch");
  }
  if (static_cast<int>(context.alpha_support_occupied_physical_orbitals.size()) !=
      context.n_alpha_support_determinants) {
    throw std::invalid_argument(
        "alpha support occupied physical orbital storage size mismatch");
  }
  if (static_cast<int>(context.beta_support_occupied_physical_orbitals.size()) !=
      context.n_beta_support_determinants) {
    throw std::invalid_argument(
        "beta support occupied physical orbital storage size mismatch");
  }
  if (static_cast<int>(context.alpha_pair_first_order_cofactors.size()) !=
      context.n_alpha_support_determinants * context.n_alpha_support_determinants) {
    throw std::invalid_argument("alpha pair first-cofactor storage size mismatch");
  }
  if (static_cast<int>(context.beta_pair_first_order_cofactors.size()) !=
      context.n_beta_support_determinants * context.n_beta_support_determinants) {
    throw std::invalid_argument("beta pair first-cofactor storage size mismatch");
  }
  for (const Eigen::MatrixXd& block : context.alpha_support_occupied_physical_orbitals) {
    if (block.rows() != context.n_basis_functions ||
        block.cols() != context.n_alpha_occupied_orbitals) {
      throw std::invalid_argument(
          "alpha support occupied physical orbital block shape mismatch");
    }
  }
  for (const Eigen::MatrixXd& block : context.beta_support_occupied_physical_orbitals) {
    if (block.rows() != context.n_basis_functions ||
        block.cols() != context.n_beta_occupied_orbitals) {
      throw std::invalid_argument(
          "beta support occupied physical orbital block shape mismatch");
    }
  }
  for (const Eigen::MatrixXd& block : context.alpha_pair_first_order_cofactors) {
    if (block.rows() != context.n_alpha_occupied_orbitals ||
        block.cols() != context.n_alpha_occupied_orbitals) {
      throw std::invalid_argument("alpha pair first-cofactor block shape mismatch");
    }
  }
  for (const Eigen::MatrixXd& block : context.beta_pair_first_order_cofactors) {
    if (block.rows() != context.n_beta_occupied_orbitals ||
        block.cols() != context.n_beta_occupied_orbitals) {
      throw std::invalid_argument("beta pair first-cofactor block shape mismatch");
    }
  }
  if (!std::isfinite(context.selected_state_overlap_normalization) ||
      std::abs(context.selected_state_overlap_normalization) <=
          kNormalizationTolerance) {
    throw std::invalid_argument("selected-state overlap normalization is invalid");
  }
}

}  // namespace

SelectedStateExactPhysicalOnTopPairDensityBuilder::
SelectedStateExactPhysicalOnTopPairDensityBuilder(
    VBSCFAlgorithm algorithm)
    : gradient_evaluator_(algorithm) {}

SelectedStateExactPhysicalOnTopPairDensityBuilder::
SelectedStateExactPhysicalOnTopPairDensityBuilder(
    CppActiveSpaceGradientEvaluator gradient_evaluator)
    : gradient_evaluator_(std::move(gradient_evaluator)) {}

SelectedStateExactPhysicalOnTopPairDensityContext
SelectedStateExactPhysicalOnTopPairDensityBuilder::build(
    const CppVbInput& input,
    int state_index,
    double nuclear_repulsion_energy) const {
  const CppActiveSpaceGradientResult gradient_result =
      gradient_evaluator_.evaluate(
          input,
          std::vector<int>{state_index},
          std::vector<double>{1.0},
          nuclear_repulsion_energy);
  return build_from_state_specific_gradient(input, gradient_result);
}

SelectedStateExactPhysicalOnTopPairDensityContext
SelectedStateExactPhysicalOnTopPairDensityBuilder::build_from_state_specific_gradient(
    const CppVbInput& input,
    const CppActiveSpaceGradientResult& gradient_result) const {
  return assemble_selected_state_exact_physical_on_top_pair_density_context(
      input,
      gradient_result);
}

double evaluate_selected_state_exact_physical_on_top_pair_density(
    const SelectedStateExactPhysicalOnTopPairDensityContext& context,
    const Eigen::Ref<const Eigen::VectorXd>& ao_values) {
  const Eigen::MatrixXd probe_matrix = ao_values;
  return evaluate_selected_state_exact_physical_on_top_pair_density_batch(
      context,
      probe_matrix)(0);
}

Eigen::VectorXd evaluate_selected_state_exact_physical_on_top_pair_density_batch(
    const SelectedStateExactPhysicalOnTopPairDensityContext& context,
    const Eigen::Ref<const Eigen::MatrixXd>& ao_value_matrix) {
  validate_exact_physical_on_top_pair_density_context(context);
  if (ao_value_matrix.rows() != context.n_basis_functions) {
    throw std::invalid_argument("AO probe matrix row count mismatch");
  }
  if (ao_value_matrix.cols() == 0) {
    return Eigen::VectorXd::Zero(0);
  }

  const Eigen::MatrixXd alpha_flattened_transition_density_scalars =
      build_transition_density_scalar_block(
          context.alpha_support_occupied_physical_orbitals,
          context.alpha_pair_first_order_cofactors,
          ao_value_matrix);
  const Eigen::MatrixXd beta_flattened_transition_density_scalars =
      build_transition_density_scalar_block(
          context.beta_support_occupied_physical_orbitals,
          context.beta_pair_first_order_cofactors,
          ao_value_matrix);
  const Eigen::MatrixXd& coefficient_matrix = context.local_coefficient_matrix;

  // Each probe column reuses the same support coefficient matrix and pair
  // cofactors. The only probe-dependent objects are the same-spin transition
  // density scalars. This block path therefore exposes the exact future grid
  // contraction without introducing an explicit physical four-index `2-RDM`.
  Eigen::VectorXd on_top_pair_density_values =
      Eigen::VectorXd::Zero(ao_value_matrix.cols());
  for (int probe_index = 0; probe_index < ao_value_matrix.cols(); ++probe_index) {
    const Eigen::Map<const Eigen::MatrixXd> alpha_transition_density_scalars(
        alpha_flattened_transition_density_scalars.col(probe_index).data(),
        context.n_alpha_support_determinants,
        context.n_alpha_support_determinants);
    const Eigen::Map<const Eigen::MatrixXd> beta_transition_density_scalars(
        beta_flattened_transition_density_scalars.col(probe_index).data(),
        context.n_beta_support_determinants,
        context.n_beta_support_determinants);
    const Eigen::MatrixXd contracted_beta =
        alpha_transition_density_scalars *
        coefficient_matrix *
        beta_transition_density_scalars.transpose();
    on_top_pair_density_values(probe_index) =
        contracted_beta.cwiseProduct(coefficient_matrix).sum() /
        context.selected_state_overlap_normalization;
  }
  return on_top_pair_density_values;
}

}  // namespace xmvb::vb
