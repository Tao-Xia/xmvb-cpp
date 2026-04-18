#include "vb/pdft/selected_state_exact_physical_one_rdm_builder.hpp"

#include <algorithm>
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
constexpr double kWeightTolerance = 1.0e-16;

struct SelectedStateCoefficientContext {
  SelectedStateDeterminantMatrices selected_state_matrices;
  SpinDeterminantReuseTable alpha_reuse_table;
  SpinDeterminantReuseTable beta_reuse_table;
};

struct PhysicalSpinSupportDeterminant {
  std::vector<int> full_occupied_orbitals;
  Eigen::MatrixXd occupied_physical_orbitals;
};

struct PhysicalSpinPairEntry {
  double overlap_determinant = 0.0;
  Eigen::MatrixXd first_order_cofactor;
};

struct PhysicalSpinPairTables {
  Eigen::MatrixXd overlap_matrix;
  std::vector<PhysicalSpinSupportDeterminant> support_determinants;
  std::vector<PhysicalSpinPairEntry> ordered_pair_entries;
};

double max_abs_skew_part(const Eigen::MatrixXd& matrix) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("skew-part check requires a square matrix");
  }

  double max_abs_value = 0.0;
  for (int column = 0; column < matrix.cols(); ++column) {
    for (int row = 0; row < matrix.rows(); ++row) {
      max_abs_value = std::max(
          max_abs_value,
          std::abs(matrix(row, column) - matrix(column, row)));
    }
  }
  return max_abs_value;
}

double frobenius_inner_product(
    const Eigen::MatrixXd& left,
    const Eigen::MatrixXd& right) {
  if (left.rows() != right.rows() || left.cols() != right.cols()) {
    throw std::invalid_argument("Frobenius inner product size mismatch");
  }
  return left.cwiseProduct(right).sum();
}

void validate_single_state_gradient_result(
    const CppActiveSpaceGradientResult& gradient_result) {
  if (gradient_result.scf_result.selected_state_indices.size() != 1 ||
      gradient_result.scf_result.state_average_weights.size() != 1) {
    throw std::invalid_argument(
        "exact physical one-RDM build requires exactly one selected state");
  }
  const double state_weight =
      gradient_result.scf_result.state_average_weights.front();
  if (std::abs(state_weight - 1.0) > kUnitWeightTolerance) {
    throw std::invalid_argument(
        "exact physical one-RDM build requires a unit selected-state weight");
  }
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

PhysicalSpinPairTables build_physical_spin_pair_tables(
    const std::vector<std::vector<int>>& unique_spin_determinants,
    const std::vector<int>& support_indices,
    const Eigen::Ref<const Eigen::MatrixXd>& full_physical_orbital_matrix,
    const std::vector<double>& full_physical_orbital_overlap_matrix,
    int n_total_orbitals,
    int n_inactive_doubly_occupied_orbitals,
    const DeterminantOverlapResolver& overlap_resolver) {
  const int n_support_determinants = static_cast<int>(support_indices.size());
  PhysicalSpinPairTables tables;
  tables.overlap_matrix =
      Eigen::MatrixXd::Zero(n_support_determinants, n_support_determinants);
  tables.support_determinants.reserve(support_indices.size());
  tables.ordered_pair_entries.resize(
      xmvb::to_size(n_support_determinants) * n_support_determinants);

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
    tables.support_determinants.push_back(std::move(support_determinant));
  }

  // Each ordered same-spin pair is defined in the physical full-occupied frame:
  // inactive orbitals remain explicit, and the active occupied strings are
  // lifted from local active indices to the global physical orbital ordering.
  // The first-order cofactor is exactly the determinant-level transition
  // density kernel needed for one-body observables, so this cache is the clean
  // route to an exact physical AO density.
  for (int left_local = 0;
       left_local < n_support_determinants;
       ++left_local) {
    for (int right_local = left_local;
         right_local < n_support_determinants;
         ++right_local) {
      const auto& left_determinant =
          tables.support_determinants[xmvb::to_size(left_local)];
      const auto& right_determinant =
          tables.support_determinants[xmvb::to_size(right_local)];
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

      PhysicalSpinPairEntry forward_entry;
      forward_entry.overlap_determinant = overlap_result.overlap_determinant;
      if (overlap_result.nullity < 2) {
        forward_entry.first_order_cofactor =
            calc_cofactor_1st(overlap_result);
      }

      tables.overlap_matrix(right_local, left_local) =
          forward_entry.overlap_determinant;
      tables.ordered_pair_entries[ordered_pair_storage_index(
          left_local,
          right_local,
          n_support_determinants)] = forward_entry;

      if (right_local == left_local) {
        continue;
      }

      PhysicalSpinPairEntry reverse_entry;
      reverse_entry.overlap_determinant = forward_entry.overlap_determinant;
      if (forward_entry.first_order_cofactor.size() != 0) {
        reverse_entry.first_order_cofactor =
            forward_entry.first_order_cofactor.transpose();
      }
      tables.overlap_matrix(left_local, right_local) =
          reverse_entry.overlap_determinant;
      tables.ordered_pair_entries[ordered_pair_storage_index(
          right_local,
          left_local,
          n_support_determinants)] = std::move(reverse_entry);
    }
  }

  return tables;
}

SelectedStateExactPhysicalOneRdmResult assemble_selected_state_exact_physical_one_rdm(
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
        "exact physical one-RDM build requires positive basis and orbital dimensions");
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
        "exact physical one-RDM build received inconsistent AO or physical-orbital dimensions");
  }

  const SelectedStateCoefficientContext coefficient_context =
      materialize_selected_state_coefficients(input, gradient_result);
  if (coefficient_context.selected_state_matrices.states.size() != 1) {
    throw std::invalid_argument(
        "exact physical one-RDM build requires one selected-state coefficient bundle");
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
  const PhysicalSpinPairTables alpha_tables =
      build_physical_spin_pair_tables(
          coefficient_context.alpha_reuse_table.unique_determinants,
          selected_state.alpha_support,
          full_physical_orbital_matrix,
          full_physical_orbital_overlap_storage,
          n_total_orbitals,
          n_inactive_doubly_occupied_orbitals,
          overlap_resolver);
  const PhysicalSpinPairTables beta_tables =
      build_physical_spin_pair_tables(
          coefficient_context.beta_reuse_table.unique_determinants,
          selected_state.beta_support,
          full_physical_orbital_matrix,
          full_physical_orbital_overlap_storage,
          n_total_orbitals,
          n_inactive_doubly_occupied_orbitals,
          overlap_resolver);

  const Eigen::MatrixXd& coefficient_matrix = selected_state.local_coefficient_matrix;
  const Eigen::MatrixXd alpha_weights =
      coefficient_matrix *
      beta_tables.overlap_matrix *
      coefficient_matrix.transpose();
  const Eigen::MatrixXd beta_weights =
      coefficient_matrix.transpose() *
      alpha_tables.overlap_matrix *
      coefficient_matrix;
  const double alpha_normalization =
      frobenius_inner_product(alpha_weights, alpha_tables.overlap_matrix);
  const double beta_normalization =
      frobenius_inner_product(beta_weights, beta_tables.overlap_matrix);
  const double normalization =
      0.5 * (alpha_normalization + beta_normalization);
  if (!std::isfinite(normalization) ||
      std::abs(normalization) <= kNormalizationTolerance) {
    throw std::runtime_error(
        "exact physical one-RDM normalization is too small or non-finite");
  }

  Eigen::MatrixXd raw_ao_density =
      Eigen::MatrixXd::Zero(n_basis_functions, n_basis_functions);
  for (int left_local = 0;
       left_local < static_cast<int>(selected_state.alpha_support.size());
       ++left_local) {
    for (int right_local = 0;
         right_local < static_cast<int>(selected_state.alpha_support.size());
         ++right_local) {
      const double weight = alpha_weights(right_local, left_local);
      if (std::abs(weight) <= kWeightTolerance) {
        continue;
      }
      const auto& pair_entry =
          alpha_tables.ordered_pair_entries[ordered_pair_storage_index(
              left_local,
              right_local,
              static_cast<int>(selected_state.alpha_support.size()))];
      if (pair_entry.first_order_cofactor.size() == 0) {
        continue;
      }
      const auto& left_orbitals =
          alpha_tables.support_determinants[xmvb::to_size(left_local)]
              .occupied_physical_orbitals;
      const auto& right_orbitals =
          alpha_tables.support_determinants[xmvb::to_size(right_local)]
              .occupied_physical_orbitals;
      raw_ao_density.noalias() +=
          weight *
          right_orbitals *
          pair_entry.first_order_cofactor *
          left_orbitals.transpose();
    }
  }
  for (int left_local = 0;
       left_local < static_cast<int>(selected_state.beta_support.size());
       ++left_local) {
    for (int right_local = 0;
         right_local < static_cast<int>(selected_state.beta_support.size());
         ++right_local) {
      const double weight = beta_weights(right_local, left_local);
      if (std::abs(weight) <= kWeightTolerance) {
        continue;
      }
      const auto& pair_entry =
          beta_tables.ordered_pair_entries[ordered_pair_storage_index(
              left_local,
              right_local,
              static_cast<int>(selected_state.beta_support.size()))];
      if (pair_entry.first_order_cofactor.size() == 0) {
        continue;
      }
      const auto& left_orbitals =
          beta_tables.support_determinants[xmvb::to_size(left_local)]
              .occupied_physical_orbitals;
      const auto& right_orbitals =
          beta_tables.support_determinants[xmvb::to_size(right_local)]
              .occupied_physical_orbitals;
      raw_ao_density.noalias() +=
          weight *
          right_orbitals *
          pair_entry.first_order_cofactor *
          left_orbitals.transpose();
    }
  }

  // The exact determinant-pair sum above is the physical spin-summed
  // transition-density accumulation:
  //
  // `P = sum_abab C_L C_R [ S_beta * P_alpha + S_alpha * P_beta ]`.
  //
  // Dividing by the selected-state normalization and symmetrizing the final AO
  // matrix gives the real physical density that should enter future grid-space
  // `rho(r)` evaluations.
  const Eigen::MatrixXd normalized_raw_ao_density =
      raw_ao_density / normalization;
  const Eigen::MatrixXd symmetrized_ao_density =
      0.5 * (normalized_raw_ao_density + normalized_raw_ao_density.transpose());

  SelectedStateExactPhysicalOneRdmResult result;
  result.state_index = gradient_result.scf_result.selected_state_indices.front();
  result.n_basis_functions = n_basis_functions;
  result.physical_ao_total_density_matrix = symmetrized_ao_density;
  result.ao_density_asymmetry_max_abs =
      max_abs_skew_part(normalized_raw_ao_density);
  result.selected_state_overlap_normalization = normalization;
  result.alpha_beta_normalization_abs_error =
      std::abs(alpha_normalization - beta_normalization);
  result.ao_total_metric_trace =
      frobenius_inner_product(symmetrized_ao_density, ao_overlap_matrix);
  return result;
}

void validate_exact_physical_density_result(
    const SelectedStateExactPhysicalOneRdmResult& density_result) {
  if (density_result.n_basis_functions <= 0) {
    throw std::invalid_argument(
        "exact physical density result requires a positive basis dimension");
  }
  if (density_result.physical_ao_total_density_matrix.rows() !=
          density_result.n_basis_functions ||
      density_result.physical_ao_total_density_matrix.cols() !=
          density_result.n_basis_functions) {
    throw std::invalid_argument(
        "exact physical density matrix storage size mismatch");
  }
}

}  // namespace

SelectedStateExactPhysicalOneRdmBuilder::SelectedStateExactPhysicalOneRdmBuilder(
    VBSCFAlgorithm algorithm)
    : gradient_evaluator_(algorithm) {}

SelectedStateExactPhysicalOneRdmBuilder::SelectedStateExactPhysicalOneRdmBuilder(
    CppActiveSpaceGradientEvaluator gradient_evaluator)
    : gradient_evaluator_(std::move(gradient_evaluator)) {}

SelectedStateExactPhysicalOneRdmResult
SelectedStateExactPhysicalOneRdmBuilder::build(
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

SelectedStateExactPhysicalOneRdmResult
SelectedStateExactPhysicalOneRdmBuilder::build_from_state_specific_gradient(
    const CppVbInput& input,
    const CppActiveSpaceGradientResult& gradient_result) const {
  return assemble_selected_state_exact_physical_one_rdm(input, gradient_result);
}

double evaluate_selected_state_exact_physical_density(
    const SelectedStateExactPhysicalOneRdmResult& density_result,
    const Eigen::Ref<const Eigen::VectorXd>& ao_values) {
  const Eigen::MatrixXd probe_matrix = ao_values;
  return evaluate_selected_state_exact_physical_density_batch(
      density_result,
      probe_matrix)(0);
}

Eigen::VectorXd evaluate_selected_state_exact_physical_density_batch(
    const SelectedStateExactPhysicalOneRdmResult& density_result,
    const Eigen::Ref<const Eigen::MatrixXd>& ao_value_matrix) {
  validate_exact_physical_density_result(density_result);
  if (ao_value_matrix.rows() != density_result.n_basis_functions) {
    throw std::invalid_argument("AO probe matrix row count mismatch");
  }
  if (ao_value_matrix.cols() == 0) {
    return Eigen::VectorXd::Zero(0);
  }

  // Each column of `probe_matrix` is one AO value vector `chi(r_g)`. The
  // physical density at that probe is the quadratic form
  //
  // `rho(r_g) = chi(r_g)^T P_AO chi(r_g)`.
  //
  // The block evaluation below shares the matrix-vector product `P_AO * chi`
  // across all probes so the future grid layer can evaluate many points with a
  // single dense-matrix multiply.
  const Eigen::MatrixXd projected_probe_matrix =
      density_result.physical_ao_total_density_matrix * ao_value_matrix;
  return (ao_value_matrix.cwiseProduct(projected_probe_matrix)).colwise().sum().transpose();
}

}  // namespace xmvb::vb
