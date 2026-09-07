#include "vb/scf/orbital_objective.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "vb/orbital/localized_representative_selector.hpp"
#include "vb/orbital/orbital_preparation_input.hpp"
#include "vb/orbital/support_aware_mo_gauge_fix.hpp"
#include "vb/runtime_utils.hpp"
#include "vb/scf/cpp_active_space_second_order_context.hpp"
#include "vb/scf/scf_vector_utilities.hpp"

namespace xmvb::vb {

// ===========================================================================
// Sparse-orbital block manipulation helpers.
//
// These helpers move between the packed differentiable parameter view and the
// flat `OrbitalPreparationInput` sparse-orbital table (`orbital_value_table`,
// `orbital_basis_index_table`). They are kept here because the canonicalization
// path below is the only consumer that needs dense block views of the active
// and inactive occupied orbitals at the accepted point.
// ===========================================================================

bool orbital_uses_full_ao_support(
    const OrbitalPreparationInput& orbital_preparation_input,
    int orbital_index) {
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  if (stored_sparse_orbital_coefficient_count(
          orbital_preparation_input,
          orbital_index) !=
      n_basis_functions) {
    return false;
  }

  std::vector<unsigned char> seen_basis_rows(
      n_basis_functions,
      static_cast<unsigned char>(0));
  for (int coefficient_index = 0;
       coefficient_index < n_basis_functions;
       ++coefficient_index) {
    const int basis_function_index =
        orbital_preparation_input.orbital_basis_index_table
            [orbital_index * n_basis_functions +
             coefficient_index] -
        1;
    if (basis_function_index < 0 ||
        basis_function_index >= n_basis_functions ||
        seen_basis_rows[basis_function_index] != 0) {
      return false;
    }
    seen_basis_rows[basis_function_index] = 1;
  }
  return true;
}

void require_full_ao_orbital_block(
    const OrbitalPreparationInput& orbital_preparation_input,
    int first_orbital,
    int orbital_count,
    const char* label) {
  for (int orbital_offset = 0; orbital_offset < orbital_count; ++orbital_offset) {
    const int orbital_index = first_orbital + orbital_offset;
    if (!orbital_uses_full_ao_support(orbital_preparation_input, orbital_index)) {
      throw std::runtime_error(
          std::string(label) +
          " requires a full-AO occupied chart for exact representative transport");
    }
  }
}

Eigen::MatrixXd build_dense_orbital_block_from_sparse_input(
    const OrbitalPreparationInput& orbital_preparation_input,
    int first_orbital,
    int orbital_count) {
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  Eigen::MatrixXd dense_block =
      Eigen::MatrixXd::Zero(n_basis_functions, std::max(0, orbital_count));
  for (int orbital_offset = 0; orbital_offset < orbital_count; ++orbital_offset) {
    const int orbital_index = first_orbital + orbital_offset;
    const int coefficient_count =
        stored_sparse_orbital_coefficient_count(
            orbital_preparation_input,
            orbital_index);
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      const int basis_function_index =
          orbital_preparation_input.orbital_basis_index_table
              [orbital_index * n_basis_functions +
               coefficient_index] -
              1;
      if (basis_function_index < 0 ||
          basis_function_index >= n_basis_functions) {
        throw std::runtime_error(
            "invalid sparse orbital basis index while rebuilding the accepted-point frame");
      }
      dense_block(basis_function_index, orbital_offset) =
          orbital_preparation_input.orbital_value_table
              [orbital_index * n_basis_functions +
               coefficient_index];
    }
  }
  return dense_block;
}

Eigen::MatrixXd build_dense_orbital_block_from_full_vector(
    const OrbitalPreparationInput& orbital_preparation_input,
    const std::vector<double>& full_vector,
    int first_orbital,
    int orbital_count) {
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  Eigen::MatrixXd dense_block =
      Eigen::MatrixXd::Zero(n_basis_functions, std::max(0, orbital_count));
  for (int orbital_offset = 0; orbital_offset < orbital_count; ++orbital_offset) {
    const int orbital_index = first_orbital + orbital_offset;
    const int coefficient_count =
        stored_sparse_orbital_coefficient_count(
            orbital_preparation_input,
            orbital_index);
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      const int basis_function_index =
          orbital_preparation_input.orbital_basis_index_table
              [orbital_index * n_basis_functions +
               coefficient_index] -
              1;
      if (basis_function_index < 0 ||
          basis_function_index >= n_basis_functions) {
        throw std::runtime_error(
            "invalid sparse orbital basis index while rebuilding a dense gradient block");
      }
      dense_block(basis_function_index, orbital_offset) =
          full_vector[orbital_index * n_basis_functions +
                      coefficient_index];
    }
  }
  return dense_block;
}

void scatter_dense_orbital_block_to_full_vector(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_block,
    const OrbitalPreparationInput& orbital_preparation_input,
    int first_orbital,
    std::vector<double>* full_vector) {
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  for (int orbital_offset = 0; orbital_offset < dense_block.cols(); ++orbital_offset) {
    const int orbital_index = first_orbital + orbital_offset;
    const int coefficient_count =
        stored_sparse_orbital_coefficient_count(
            orbital_preparation_input,
            orbital_index);
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      const int basis_function_index =
          orbital_preparation_input.orbital_basis_index_table
              [orbital_index * n_basis_functions +
               coefficient_index] -
              1;
      if (basis_function_index < 0 ||
          basis_function_index >= n_basis_functions) {
        throw std::runtime_error(
            "invalid sparse orbital basis index while scattering a dense gradient block");
      }
      (*full_vector)[orbital_index * n_basis_functions +
                     coefficient_index] =
          dense_block(basis_function_index, orbital_offset);
    }
  }
}

// ===========================================================================
// OEO active-representative transport and secant-history transport.
// ===========================================================================

void transform_sparse_oeo_active_representative_gradient(
    const LocalizedRepresentativeSelector& source_selector,
    const LocalizedRepresentativeSelector& target_selector,
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* sparse_orbital_gradient) {
  if (orbital_preparation_input.orbital_type != kLegacyOrbitalTypeOeo) {
    throw std::invalid_argument(
        "active representative transport is only defined on the current OEO chart");
  }

  const int n_inactive_doubly_occupied_orbitals =
      (orbital_preparation_input.n_total_electrons -
       orbital_preparation_input.n_active_electrons) / 2;
  const int n_active_orbitals =
      orbital_preparation_input.n_active_orbitals;
  if (n_inactive_doubly_occupied_orbitals <= 0 || n_active_orbitals <= 0) {
    return;
  }

  if (source_selector.inactive_inverse_transpose_right_transform.rows() !=
          n_inactive_doubly_occupied_orbitals ||
      source_selector.inactive_inverse_transpose_right_transform.cols() !=
          n_inactive_doubly_occupied_orbitals ||
      source_selector.active_inactive_coefficients.rows() !=
          n_inactive_doubly_occupied_orbitals ||
      source_selector.active_inactive_coefficients.cols() != n_active_orbitals ||
      target_selector.active_inactive_coefficients.rows() !=
          n_inactive_doubly_occupied_orbitals ||
      target_selector.active_inactive_coefficients.cols() != n_active_orbitals) {
    throw std::runtime_error(
        "localized representative selector dimensions do not match the current occupied chart");
  }

  require_full_ao_orbital_block(
      orbital_preparation_input,
      0,
      n_inactive_doubly_occupied_orbitals + n_active_orbitals,
      "OEO active representative reset");

  // The accepted-point repair keeps the internal frame `(Q_i, T_a)` fixed and
  // changes only the inactive-null active coefficients:
  // `C_a = T_a + C_i U_i^{-1} K_a`.
  //
  // If the representative changes from `K_a` to `K̂_a`, then in the new chart
  // the old physical active block satisfies
  // `C_a = Ĉ_a - Ĉ_i Λ_rep`, where
  // `Λ_rep = U_i^{-1} (K̂_a - K_a)`.
  // Therefore the sparse full-AO orbital gradient transforms as
  // `Ĝ_Ci = G_Ci - G_Ca Λ_rep^T` and `Ĝ_Ca = G_Ca`.
  const Eigen::MatrixXd inactive_right_inverse =
      source_selector.inactive_inverse_transpose_right_transform.transpose();
  const Eigen::MatrixXd representative_shift =
      inactive_right_inverse *
      (target_selector.active_inactive_coefficients -
       source_selector.active_inactive_coefficients);

  constexpr double kRepresentativeShiftTolerance = 1.0e-13;
  const double representative_shift_max_abs =
      representative_shift.size() == 0
          ? 0.0
          : representative_shift.cwiseAbs().maxCoeff();
  if (!(representative_shift_max_abs > kRepresentativeShiftTolerance)) {
    return;
  }

  const Eigen::MatrixXd inactive_gradient =
      build_dense_orbital_block_from_full_vector(
          orbital_preparation_input,
          *sparse_orbital_gradient,
          0,
          n_inactive_doubly_occupied_orbitals);
  const Eigen::MatrixXd active_gradient =
      build_dense_orbital_block_from_full_vector(
          orbital_preparation_input,
          *sparse_orbital_gradient,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);
  const Eigen::MatrixXd transported_inactive_gradient =
      inactive_gradient - active_gradient * representative_shift.transpose();
  scatter_dense_orbital_block_to_full_vector(
      transported_inactive_gradient,
      orbital_preparation_input,
      0,
      sparse_orbital_gradient);
}

void transform_sparse_inactive_orbital_step(
    const SupportAwareInactiveMoGaugeTransform& transform,
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* sparse_orbital_step) {
  if (!transform.chart_changed || transform.n_inactive_orbitals <= 1) {
    return;
  }

  const int n_inactive_orbitals = transform.n_inactive_orbitals;
  const Eigen::Map<const Eigen::MatrixXd> right_transform(
      transform.right_transform.data(),
      n_inactive_orbitals,
      n_inactive_orbitals);
  const Eigen::MatrixXd inactive_step =
      build_dense_orbital_block_from_full_vector(
          orbital_preparation_input,
          *sparse_orbital_step,
          0,
          n_inactive_orbitals);
  const Eigen::MatrixXd transported_inactive_step =
      inactive_step * right_transform;
  scatter_dense_orbital_block_to_full_vector(
      transported_inactive_step,
      orbital_preparation_input,
      0,
      sparse_orbital_step);
}

void transform_sparse_oeo_active_representative_step(
    const LocalizedRepresentativeSelector& source_selector,
    const LocalizedRepresentativeSelector& target_selector,
    const OrbitalPreparationInput& orbital_preparation_input,
    std::vector<double>* sparse_orbital_step) {
  if (orbital_preparation_input.orbital_type != kLegacyOrbitalTypeOeo) {
    throw std::invalid_argument(
        "active representative step transport is only defined on the current OEO chart");
  }

  const int n_inactive_doubly_occupied_orbitals =
      (orbital_preparation_input.n_total_electrons -
       orbital_preparation_input.n_active_electrons) / 2;
  const int n_active_orbitals =
      orbital_preparation_input.n_active_orbitals;
  if (n_inactive_doubly_occupied_orbitals <= 0 || n_active_orbitals <= 0) {
    return;
  }

  const Eigen::MatrixXd inactive_right_inverse =
      source_selector.inactive_inverse_transpose_right_transform.transpose();
  const Eigen::MatrixXd representative_shift =
      inactive_right_inverse *
      (target_selector.active_inactive_coefficients -
       source_selector.active_inactive_coefficients);
  constexpr double kRepresentativeShiftTolerance = 1.0e-13;
  const double representative_shift_max_abs =
      representative_shift.size() == 0
          ? 0.0
          : representative_shift.cwiseAbs().maxCoeff();
  if (!(representative_shift_max_abs > kRepresentativeShiftTolerance)) {
    return;
  }

  require_full_ao_orbital_block(
      orbital_preparation_input,
      0,
      n_inactive_doubly_occupied_orbitals + n_active_orbitals,
      "OEO active representative reset");
  // The affine active representative reset is
  // `C_a(new) = C_a(old) + C_i(old) Λ_rep`, so tangent vectors transport with
  // the direct Jacobian:
  // `dC_i(new) = dC_i(old)` and `dC_a(new) = dC_a(old) + dC_i(old) Λ_rep`.
  const Eigen::MatrixXd inactive_step =
      build_dense_orbital_block_from_full_vector(
          orbital_preparation_input,
          *sparse_orbital_step,
          0,
          n_inactive_doubly_occupied_orbitals);
  Eigen::MatrixXd active_step =
      build_dense_orbital_block_from_full_vector(
          orbital_preparation_input,
          *sparse_orbital_step,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);
  active_step.noalias() += inactive_step * representative_shift;
  scatter_dense_orbital_block_to_full_vector(
      active_step,
      orbital_preparation_input,
      n_inactive_doubly_occupied_orbitals,
      sparse_orbital_step);
}

std::vector<double> build_full_sparse_vector_from_packed(
    const SparseOrbitalParameterView& parameter_view,
    const OrbitalPreparationInput& orbital_preparation_input,
    const Eigen::VectorXd& packed_vector) {
  std::vector<double> full_sparse_vector(
      orbital_preparation_input.orbital_value_table.size(),
      0.0);
  const auto& differentiable_parameter_indices =
      parameter_view.differentiable_parameter_indices();
  for (Eigen::Index packed_index = 0;
       packed_index < packed_vector.size();
       ++packed_index) {
    full_sparse_vector[differentiable_parameter_indices[packed_index]] =
        packed_vector[packed_index];
  }
  return full_sparse_vector;
}

void overwrite_packed_vector_from_full_sparse(
    const SparseOrbitalParameterView& parameter_view,
    const std::vector<double>& full_sparse_vector,
    Eigen::VectorXd* packed_vector) {
  *packed_vector =
      parameter_view.gather_from_full(full_sparse_vector);
}

void transport_packed_secant_history_with_support_aware_inactive_gauge(
    const SupportAwareInactiveMoGaugeTransform& transform,
    const OrbitalPreparationInput& orbital_preparation_input,
    const SparseOrbitalParameterView& parameter_view,
    std::vector<PackedSecantPair>* packed_secant_history) {
  if (packed_secant_history == nullptr || !transform.chart_changed) {
    return;
  }

  for (PackedSecantPair& packed_pair : *packed_secant_history) {
    std::vector<double> full_sparse_step =
        build_full_sparse_vector_from_packed(
            parameter_view,
            orbital_preparation_input,
            packed_pair.packed_step);
    transform_sparse_inactive_orbital_step(
        transform,
        orbital_preparation_input,
        &full_sparse_step);
    overwrite_packed_vector_from_full_sparse(
        parameter_view,
        full_sparse_step,
        &packed_pair.packed_step);

    std::vector<double> full_sparse_gradient_change =
        build_full_sparse_vector_from_packed(
            parameter_view,
            orbital_preparation_input,
            packed_pair.packed_projected_gradient_change);
    transform_sparse_inactive_orbital_gradient(
        transform,
        orbital_preparation_input,
        &full_sparse_gradient_change);
    overwrite_packed_vector_from_full_sparse(
        parameter_view,
        full_sparse_gradient_change,
        &packed_pair.packed_projected_gradient_change);
  }
}

void transport_packed_secant_history_with_oeo_active_representative_reset(
    const LocalizedRepresentativeSelector& source_selector,
    const LocalizedRepresentativeSelector& target_selector,
    const OrbitalPreparationInput& orbital_preparation_input,
    const SparseOrbitalParameterView& parameter_view,
    std::vector<PackedSecantPair>* packed_secant_history) {
  if (packed_secant_history == nullptr || packed_secant_history->empty()) {
    return;
  }

  for (PackedSecantPair& packed_pair : *packed_secant_history) {
    std::vector<double> full_sparse_step =
        build_full_sparse_vector_from_packed(
            parameter_view,
            orbital_preparation_input,
            packed_pair.packed_step);
    transform_sparse_oeo_active_representative_step(
        source_selector,
        target_selector,
        orbital_preparation_input,
        &full_sparse_step);
    overwrite_packed_vector_from_full_sparse(
        parameter_view,
        full_sparse_step,
        &packed_pair.packed_step);

    std::vector<double> full_sparse_gradient_change =
        build_full_sparse_vector_from_packed(
            parameter_view,
            orbital_preparation_input,
            packed_pair.packed_projected_gradient_change);
    transform_sparse_oeo_active_representative_gradient(
        source_selector,
        target_selector,
        orbital_preparation_input,
        &full_sparse_gradient_change);
    overwrite_packed_vector_from_full_sparse(
        parameter_view,
        full_sparse_gradient_change,
        &packed_pair.packed_projected_gradient_change);
  }
}

// ===========================================================================
// Accepted-point cache refresh and dense matrix math.
// ===========================================================================

void refresh_cached_localized_representative_selector(
    const OrbitalPreparationInput& orbital_preparation_input,
    OrbitalPreparationResult* orbital_result) {
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  const int n_inactive_doubly_occupied_orbitals =
      (orbital_preparation_input.n_total_electrons -
       orbital_preparation_input.n_active_electrons) / 2;
  const int n_active_orbitals =
      orbital_preparation_input.n_active_orbitals;
  const int n_occupied_orbitals =
      n_inactive_doubly_occupied_orbitals + n_active_orbitals;
  const int n_virtual_orbitals =
      n_basis_functions - n_occupied_orbitals;
  auto& physical_orbital_frame = orbital_result->physical_orbital_frame;
  if (physical_orbital_frame.normalized_orbital_matrix.rows() != n_basis_functions ||
      physical_orbital_frame.normalized_orbital_matrix.cols() !=
          orbital_preparation_input.n_orbitals ||
      physical_orbital_frame.active_physical_orbital_matrix.rows() !=
          n_basis_functions ||
      physical_orbital_frame.active_physical_orbital_matrix.cols() !=
          n_active_orbitals ||
      orbital_result->active_orbital_overlap_matrix.size() !=
          n_active_orbitals * n_active_orbitals ||
      orbital_result->inactive_auxiliary_transform.size() !=
          n_basis_functions * n_basis_functions ||
      orbital_result->auxiliary_orbital_inverse_matrix.size() !=
          n_basis_functions * n_basis_functions ||
      orbital_result->auxiliary_orbital_matrix.rows() != n_basis_functions ||
      orbital_result->auxiliary_orbital_matrix.cols() != n_basis_functions) {
    throw std::runtime_error(
        "cached orbital result does not match the current accepted-point dimensions");
  }

  const Eigen::Map<const Eigen::MatrixXd> basis_overlap_matrix(
      orbital_preparation_input.ao_overlap_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  physical_orbital_frame.inactive_physical_orbital_matrix =
      build_dense_orbital_block_from_sparse_input(
          orbital_preparation_input,
          0,
          n_inactive_doubly_occupied_orbitals);
  if (n_inactive_doubly_occupied_orbitals > 0) {
    const Eigen::MatrixXd inactive_overlap =
        physical_orbital_frame.inactive_physical_orbital_matrix.transpose() *
        basis_overlap_matrix *
        physical_orbital_frame.inactive_physical_orbital_matrix;
    physical_orbital_frame.inactive_orthonormal_gauge_transform =
        build_self_adjoint_matrix_power(
            inactive_overlap,
            -0.5,
            "accepted_point_inactive_overlap");
    physical_orbital_frame.inactive_orthonormal_orbital_matrix =
        physical_orbital_frame.inactive_physical_orbital_matrix *
        physical_orbital_frame.inactive_orthonormal_gauge_transform;
  } else {
    physical_orbital_frame.inactive_orthonormal_gauge_transform =
        Eigen::MatrixXd::Zero(0, 0);
    physical_orbital_frame.inactive_orthonormal_orbital_matrix =
        Eigen::MatrixXd::Zero(n_basis_functions, 0);
  }

  orbital_result->auxiliary_orbital_matrix.leftCols(
      n_inactive_doubly_occupied_orbitals) =
      physical_orbital_frame.inactive_physical_orbital_matrix;
  physical_orbital_frame.normalized_orbital_matrix.leftCols(
      n_inactive_doubly_occupied_orbitals) =
      physical_orbital_frame.inactive_physical_orbital_matrix;

  Eigen::MatrixXd inactive_auxiliary_transform =
      Eigen::MatrixXd::Zero(n_basis_functions, n_basis_functions);
  if (n_inactive_doubly_occupied_orbitals > 0) {
    const Eigen::MatrixXd inactive_overlap_inverse =
        build_inactive_metric_inverse(
            physical_orbital_frame.inactive_physical_orbital_matrix,
            basis_overlap_matrix);
    inactive_auxiliary_transform.leftCols(n_inactive_doubly_occupied_orbitals) =
        physical_orbital_frame.inactive_physical_orbital_matrix *
        inactive_overlap_inverse;
  }
  orbital_result->inactive_auxiliary_transform.assign(
      inactive_auxiliary_transform.data(),
      inactive_auxiliary_transform.data() + inactive_auxiliary_transform.size());

  orbital_result->inactive_density_matrix =
      physical_orbital_frame.inactive_orthonormal_orbital_matrix *
      physical_orbital_frame.inactive_orthonormal_orbital_matrix.transpose();
  orbital_result->inactive_orthonormal_projector_matrix =
      orbital_result->inactive_density_matrix;
  orbital_result->inactive_density_low_rank_factors =
      physical_orbital_frame.inactive_orthonormal_orbital_matrix;

  Eigen::MatrixXd occupied_space_projector =
      Eigen::MatrixXd::Identity(n_basis_functions, n_basis_functions);
  occupied_space_projector.noalias() -=
      orbital_result->inactive_density_matrix * basis_overlap_matrix;
  orbital_result->occupied_space_projector.assign(
      occupied_space_projector.data(),
      occupied_space_projector.data() + occupied_space_projector.size());

  const Eigen::MatrixXd active_overlap_source =
      basis_overlap_matrix * physical_orbital_frame.active_physical_orbital_matrix;
  const Eigen::MatrixXd inactive_active_overlap_matrix =
      inactive_auxiliary_transform.leftCols(n_inactive_doubly_occupied_orbitals)
          .transpose() *
      active_overlap_source;
  orbital_result->inactive_active_overlap_matrix.assign(
      inactive_active_overlap_matrix.data(),
      inactive_active_overlap_matrix.data() +
          inactive_active_overlap_matrix.size());
  const Eigen::MatrixXd projected_active_overlap_matrix =
      occupied_space_projector.transpose() * active_overlap_source;
  orbital_result->projected_active_overlap_matrix.assign(
      projected_active_overlap_matrix.data(),
      projected_active_overlap_matrix.data() +
          projected_active_overlap_matrix.size());

  Eigen::MatrixXd occupied_overlap_inverse =
      Eigen::MatrixXd::Zero(n_occupied_orbitals, n_occupied_orbitals);
  if (n_inactive_doubly_occupied_orbitals > 0) {
    const Eigen::MatrixXd inactive_overlap_inverse =
        build_inactive_metric_inverse(
            physical_orbital_frame.inactive_physical_orbital_matrix,
            basis_overlap_matrix);
    occupied_overlap_inverse.topLeftCorner(
        n_inactive_doubly_occupied_orbitals,
        n_inactive_doubly_occupied_orbitals) = inactive_overlap_inverse;
  }
  if (n_active_orbitals > 0) {
    const Eigen::Map<const Eigen::MatrixXd> active_overlap_matrix(
        orbital_result->active_orbital_overlap_matrix.data(),
        n_active_orbitals,
        n_active_orbitals);
    Eigen::LDLT<Eigen::MatrixXd> active_overlap_ldlt(active_overlap_matrix);
    if (active_overlap_ldlt.info() != Eigen::Success) {
      throw std::runtime_error(
          "failed to factor accepted-point active auxiliary overlap while refreshing cached selector");
    }
    occupied_overlap_inverse.bottomRightCorner(
        n_active_orbitals,
        n_active_orbitals) =
        active_overlap_ldlt.solve(
            Eigen::MatrixXd::Identity(n_active_orbitals, n_active_orbitals));
    if (active_overlap_ldlt.info() != Eigen::Success) {
      throw std::runtime_error(
          "failed to invert accepted-point active auxiliary overlap while refreshing cached selector");
    }
  }

  const Eigen::MatrixXd s_times_auxiliary =
      basis_overlap_matrix * orbital_result->auxiliary_orbital_matrix;
  Eigen::MatrixXd auxiliary_orbital_inverse =
      Eigen::MatrixXd::Zero(n_basis_functions, n_basis_functions);
  if (n_occupied_orbitals > 0) {
    auxiliary_orbital_inverse.topRows(n_occupied_orbitals).noalias() =
        occupied_overlap_inverse *
        s_times_auxiliary.leftCols(n_occupied_orbitals).transpose();
  }
  if (n_virtual_orbitals > 0) {
    auxiliary_orbital_inverse.bottomRows(n_virtual_orbitals) =
        s_times_auxiliary.rightCols(n_virtual_orbitals).transpose();
  }
  orbital_result->auxiliary_orbital_inverse_matrix.assign(
      auxiliary_orbital_inverse.data(),
      auxiliary_orbital_inverse.data() + auxiliary_orbital_inverse.size());

  physical_orbital_frame.localized_representative_selector =
      build_localized_representative_selector(
          physical_orbital_frame.inactive_physical_orbital_matrix,
          physical_orbital_frame.inactive_orthonormal_orbital_matrix,
          physical_orbital_frame.active_physical_orbital_matrix,
          orbital_result->auxiliary_orbital_matrix.middleCols(
              n_inactive_doubly_occupied_orbitals,
              n_active_orbitals),
          basis_overlap_matrix);
}

void overwrite_sparse_orbitals_from_dense_physical_frame(
    const Eigen::MatrixXd& dense_orbitals,
    OrbitalPreparationInput* orbital_preparation_input) {
  std::vector<double> updated_orbital_values =
      orbital_preparation_input->orbital_value_table;
  const int n_basis_functions = orbital_preparation_input->n_basis_functions;
  for (int orbital_index = 0;
       orbital_index < orbital_preparation_input->n_orbitals;
       ++orbital_index) {
    const int coefficient_count =
        stored_sparse_orbital_coefficient_count(
            *orbital_preparation_input,
            orbital_index);
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      const int basis_function_index =
          orbital_preparation_input->orbital_basis_index_table
              [orbital_index * n_basis_functions +
               coefficient_index] -
              1;
      if (basis_function_index < 0 ||
          basis_function_index >= orbital_preparation_input->n_basis_functions) {
        throw std::runtime_error(
            "invalid sparse orbital basis index while overwriting the final physical frame");
      }
      updated_orbital_values[orbital_index * n_basis_functions +
                             coefficient_index] =
          dense_orbitals(basis_function_index, orbital_index);
    }
  }
  orbital_preparation_input->orbital_value_table = std::move(updated_orbital_values);
}

Eigen::MatrixXd build_self_adjoint_matrix_power(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    double exponent,
    const char* label) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument(std::string(label) + " must be square");
  }
  if (matrix.rows() == 0) {
    return Eigen::MatrixXd::Zero(0, 0);
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(matrix);
  if (eigen_solver.info() != Eigen::Success) {
    throw std::runtime_error(
        std::string("failed eigendecomposition for ") + label);
  }

  Eigen::VectorXd powered_eigenvalues(matrix.rows());
  for (Eigen::Index index = 0; index < matrix.rows(); ++index) {
    const double eigenvalue = eigen_solver.eigenvalues()[index];
    if (!std::isfinite(eigenvalue) ||
        eigenvalue <= std::numeric_limits<double>::epsilon()) {
      throw std::runtime_error(
          std::string(label) + " is not numerically positive definite");
    }
    powered_eigenvalues[index] = std::pow(eigenvalue, exponent);
  }

  return eigen_solver.eigenvectors() *
      powered_eigenvalues.asDiagonal() *
      eigen_solver.eigenvectors().transpose();
}

Eigen::MatrixXd build_inactive_metric_inverse(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_physical_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap_matrix) {
  if (inactive_physical_orbitals.cols() == 0) {
    return Eigen::MatrixXd::Zero(0, 0);
  }

  // Dimensions:
  // `inactive_physical_orbitals` is `(n_basis, n_inactive)` and
  // `basis_overlap_matrix` is `(n_basis, n_basis)`. The resulting metric is
  // the small occupied-space overlap `(n_inactive, n_inactive)`.
  const Eigen::MatrixXd inactive_metric =
      inactive_physical_orbitals.transpose() *
      basis_overlap_matrix *
      inactive_physical_orbitals;
  Eigen::LDLT<Eigen::MatrixXd> inactive_metric_ldlt(inactive_metric);
  if (inactive_metric_ldlt.info() != Eigen::Success) {
    throw std::runtime_error(
        "failed to factor inactive occupied overlap while repairing OEO representative");
  }

  const Eigen::MatrixXd inactive_metric_inverse =
      inactive_metric_ldlt.solve(
          Eigen::MatrixXd::Identity(
              inactive_metric.rows(),
              inactive_metric.cols()));
  if (inactive_metric_ldlt.info() != Eigen::Success) {
    throw std::runtime_error(
        "failed to invert inactive occupied overlap while repairing OEO representative");
  }
  return inactive_metric_inverse;
}

Eigen::MatrixXd build_metric_preserving_inactive_repaired_active_physical_orbitals(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_physical_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& current_active_auxiliary,
    const Eigen::Ref<const Eigen::MatrixXd>& current_active_physical_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& reference_active_physical_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap_matrix) {
  if (current_active_auxiliary.cols() != current_active_physical_orbitals.cols() ||
      current_active_auxiliary.cols() != reference_active_physical_orbitals.cols() ||
      current_active_auxiliary.rows() != inactive_physical_orbitals.rows() ||
      current_active_physical_orbitals.rows() != inactive_physical_orbitals.rows() ||
      reference_active_physical_orbitals.rows() != inactive_physical_orbitals.rows() ||
      basis_overlap_matrix.rows() != inactive_physical_orbitals.rows() ||
      basis_overlap_matrix.cols() != inactive_physical_orbitals.rows()) {
    throw std::invalid_argument(
        "metric-preserving inactive representative repair has inconsistent dimensions");
  }
  if (current_active_auxiliary.cols() == 0) {
    return Eigen::MatrixXd::Zero(current_active_auxiliary.rows(), 0);
  }
  if (inactive_physical_orbitals.cols() == 0) {
    return current_active_physical_orbitals;
  }

  const Eigen::MatrixXd inactive_metric_inverse =
      build_inactive_metric_inverse(
          inactive_physical_orbitals,
          basis_overlap_matrix);
  const Eigen::MatrixXd inactive_dual_orbitals =
      inactive_physical_orbitals * inactive_metric_inverse;
  Eigen::MatrixXd repaired_active_physical_orbitals =
      current_active_physical_orbitals;

  // `orbtyp=oeo` stores the physical active orbitals `C_a`, while the energy
  // depends on the projected auxiliaries `T_a = (I - P_i S) C_a`.  Adding an
  // inactive component `C_i K` leaves `T_a` unchanged because
  // `(I - P_i S) C_i = 0`.  The representative reset below therefore keeps the
  // current projected active orbitals and their metric norm exactly fixed, and
  // only refreshes the inactive coefficients so the physical occupied chart
  // stays close to the initial localized reference instead of drifting along
  // the inactive-null gauge.
  for (int active_index = 0;
       active_index < current_active_auxiliary.cols();
       ++active_index) {
    const Eigen::VectorXd current_auxiliary =
        current_active_auxiliary.col(active_index);
    const Eigen::VectorXd current_physical =
        current_active_physical_orbitals.col(active_index);
    const Eigen::VectorXd reference_physical =
        reference_active_physical_orbitals.col(active_index);

    const double auxiliary_norm_squared =
        current_auxiliary.transpose() *
        basis_overlap_matrix *
        current_auxiliary;
    if (!std::isfinite(auxiliary_norm_squared) ||
        auxiliary_norm_squared <= std::numeric_limits<double>::epsilon()) {
      throw std::runtime_error(
          "encountered non-positive active auxiliary norm while resetting the OEO chart");
    }

    const double target_inactive_metric_norm_squared =
        std::max(0.0, 1.0 - auxiliary_norm_squared);
    if (target_inactive_metric_norm_squared <=
        64.0 * std::numeric_limits<double>::epsilon()) {
      repaired_active_physical_orbitals.col(active_index) = current_auxiliary;
      continue;
    }

    const Eigen::VectorXd current_inactive_coefficients =
        inactive_dual_orbitals.transpose() *
        basis_overlap_matrix *
        current_physical;
    const Eigen::VectorXd reference_direction =
        inactive_dual_orbitals.transpose() *
        basis_overlap_matrix *
        (reference_physical - current_auxiliary);
    const double reference_direction_metric_squared =
        reference_direction.dot(inactive_metric_inverse * reference_direction);

    Eigen::VectorXd repaired_inactive_coefficients =
        current_inactive_coefficients;
    if (std::isfinite(reference_direction_metric_squared) &&
        reference_direction_metric_squared >
            64.0 * std::numeric_limits<double>::epsilon()) {
      repaired_inactive_coefficients =
          std::sqrt(
              target_inactive_metric_norm_squared /
              reference_direction_metric_squared) *
          (inactive_metric_inverse * reference_direction);
    }

    Eigen::VectorXd repaired_orbital =
        current_auxiliary +
        inactive_physical_orbitals * repaired_inactive_coefficients;
    const double repaired_norm_squared =
        repaired_orbital.transpose() *
        basis_overlap_matrix *
        repaired_orbital;
    if (!std::isfinite(repaired_norm_squared) ||
        std::abs(repaired_norm_squared - 1.0) > 1.0e-8) {
      throw std::runtime_error(
          "metric-preserving OEO active representative reset changed the physical orbital norm");
    }
    repaired_active_physical_orbitals.col(active_index) = repaired_orbital;
  }

  return repaired_active_physical_orbitals;
}

Eigen::MatrixXd build_metric_preserving_oeo_repaired_normalized_orbital_matrix(
    const OrbitalPreparationInput& orbital_preparation_input,
    const OrbitalPreparationResult& orbital_result,
    const Eigen::Ref<const Eigen::MatrixXd>& reference_normalized_orbital_matrix) {
  const auto& physical_orbital_frame =
      orbital_result.physical_orbital_frame;
  if (reference_normalized_orbital_matrix.size() == 0 ||
      orbital_preparation_input.orbital_type != kLegacyOrbitalTypeOeo) {
    return physical_orbital_frame.normalized_orbital_matrix;
  }

  const int n_basis_functions =
      orbital_preparation_input.n_basis_functions;
  const int n_inactive_doubly_occupied_orbitals =
      (orbital_preparation_input.n_total_electrons -
       orbital_preparation_input.n_active_electrons) / 2;
  const int n_active_orbitals =
      orbital_preparation_input.n_active_orbitals;
  if (n_inactive_doubly_occupied_orbitals <= 0 || n_active_orbitals <= 0) {
    return physical_orbital_frame.normalized_orbital_matrix;
  }

  if (reference_normalized_orbital_matrix.rows() != n_basis_functions ||
      reference_normalized_orbital_matrix.cols() !=
          orbital_preparation_input.n_orbitals ||
      physical_orbital_frame.normalized_orbital_matrix.rows() !=
          n_basis_functions ||
      physical_orbital_frame.normalized_orbital_matrix.cols() !=
          orbital_preparation_input.n_orbitals ||
      physical_orbital_frame.inactive_physical_orbital_matrix.rows() !=
          n_basis_functions ||
      physical_orbital_frame.inactive_physical_orbital_matrix.cols() !=
          n_inactive_doubly_occupied_orbitals ||
      physical_orbital_frame.active_physical_orbital_matrix.rows() !=
          n_basis_functions ||
      physical_orbital_frame.active_physical_orbital_matrix.cols() !=
          n_active_orbitals ||
      orbital_result.auxiliary_orbital_matrix.rows() != n_basis_functions ||
      orbital_result.auxiliary_orbital_matrix.cols() <
          n_inactive_doubly_occupied_orbitals + n_active_orbitals) {
    throw std::runtime_error(
        "cached OEO orbital preparation result is incomplete while repairing the output representative");
  }

  // Dimensions:
  // `reference_normalized_orbital_matrix` and the cached physical frame are
  // `(n_basis, n_orbitals)`, while the repaired active block is
  // `(n_basis, n_active)`. Only the occupied active columns change; inactive
  // and virtual orbitals are copied through unchanged for final export.
  const Eigen::Map<const Eigen::MatrixXd> basis_overlap_matrix(
      orbital_preparation_input.ao_overlap_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::MatrixXd repaired_active_physical_orbitals =
      build_metric_preserving_inactive_repaired_active_physical_orbitals(
          physical_orbital_frame.inactive_physical_orbital_matrix,
          orbital_result.auxiliary_orbital_matrix.middleCols(
              n_inactive_doubly_occupied_orbitals,
              n_active_orbitals),
          physical_orbital_frame.active_physical_orbital_matrix,
          reference_normalized_orbital_matrix.middleCols(
              n_inactive_doubly_occupied_orbitals,
              n_active_orbitals),
          basis_overlap_matrix);

  Eigen::MatrixXd repaired_normalized_orbital_matrix =
      physical_orbital_frame.normalized_orbital_matrix;
  repaired_normalized_orbital_matrix.middleCols(
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals) = repaired_active_physical_orbitals;
  return repaired_normalized_orbital_matrix;
}

OrbitalObjective::OrbitalObjective(
    const CppVbInput& input,
    SparseOrbitalParameterView parameter_view,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy,
    const CppOrbitalGradientEvaluator* orbital_gradient_evaluator,
    const CppVbScfEvaluator* scf_evaluator)
    : working_input_(input),
      probe_input_buffer_(input),
      parameter_view_(std::move(parameter_view)),
      selected_state_indices_(selected_state_indices),
      state_average_weights_(state_average_weights),
      nuclear_repulsion_energy_(nuclear_repulsion_energy),
      orbital_gradient_evaluator_(orbital_gradient_evaluator),
      scf_evaluator_(scf_evaluator) {}

double OrbitalObjective::operator()(
    const Eigen::VectorXd& parameter_vector,
    Eigen::VectorXd& gradient) {
  TrialEvaluation evaluation =
      evaluate_trial_without_committing(parameter_vector);
  gradient = std::move(evaluation.gradient);
  const double energy = evaluation.energy;
  commit_trial_evaluation(std::move(evaluation));
  return energy;
}

void OrbitalObjective::ensure_last_reference_energy_gradient() {
  orbital_gradient_evaluator_->populate_reference_energy_gradient(
      working_input_,
      &last_gradient_result_);
}

OrbitalObjective::TrialEvaluation
OrbitalObjective::evaluate_trial_without_committing(
    const Eigen::VectorXd& parameter_vector) const {
  const auto iteration_start_time = std::chrono::steady_clock::now();
  // TN trial acceptance only needs a scratch orbital point.  Do not copy the
  // whole objective state here: the accepted-point second-order context can
  // be large, and rejected trust-region trials must leave it untouched.
  probe_input_buffer_.orbital_preparation_input =
      working_input_.orbital_preparation_input;
  parameter_view_.unpack(
      parameter_vector,
      &probe_input_buffer_.orbital_preparation_input);

  TrialEvaluation evaluation;
  evaluation.orbital_preparation_input =
      probe_input_buffer_.orbital_preparation_input;
  evaluation.gradient_result =
      orbital_gradient_evaluator_->evaluate_without_reference_energy_gradient(
          probe_input_buffer_,
          selected_state_indices_,
          state_average_weights_,
          nuclear_repulsion_energy_);
  if (evaluation.gradient_result.second_order_context == nullptr) {
    throw std::runtime_error(
        "relaxed orbital gradient did not populate the accepted-point second-order context");
  }

  evaluation.gradient = parameter_view_.gather_from_full(
      evaluation.gradient_result.sparse_orbital_energy_gradient);
  evaluation.energy = evaluation.gradient_result.scf_result.total_energy;
  evaluation.gradient_inf_norm =
      gradient_infinity_norm(evaluation.gradient);
  evaluation.wall_time_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - iteration_start_time)
          .count();
  evaluation.valid = true;
  return evaluation;
}

void OrbitalObjective::commit_trial_evaluation(TrialEvaluation evaluation) {
  if (!evaluation.valid) {
    throw std::invalid_argument("cannot commit an invalid orbital trial evaluation");
  }
  // A committed trial becomes the new accepted point used by later gradients,
  // exact-ctx HVPs, chart canonicalization, and accepted-iteration snapshots.
  working_input_.orbital_preparation_input =
      std::move(evaluation.orbital_preparation_input);
  last_gradient_result_ = std::move(evaluation.gradient_result);
  probe_input_buffer_.orbital_preparation_input =
      working_input_.orbital_preparation_input;
  energy_history_.push_back(evaluation.energy);
  gradient_inf_norm_history_.push_back(evaluation.gradient_inf_norm);
  iteration_time_history_seconds_.push_back(evaluation.wall_time_seconds);
  objective_wall_time_seconds_ += evaluation.wall_time_seconds;
}

double OrbitalObjective::evaluate_energy_only(
    const Eigen::VectorXd& parameter_vector) const {
  if (scf_evaluator_ == nullptr) {
    throw std::runtime_error("energy-only objective evaluation requires a live SCF evaluator");
  }
  // Trust-region trial rejection only needs the relaxed energy. Rebuild the
  // current orbital point in a scratch input buffer so rejected steps do not
  // pay for full gradient, adjoint, and second-order-context construction.
  probe_input_buffer_.orbital_preparation_input =
      working_input_.orbital_preparation_input;
  parameter_view_.unpack(
      parameter_vector,
      &probe_input_buffer_.orbital_preparation_input);
  const auto evaluation_start_time = std::chrono::steady_clock::now();
  const double energy = scf_evaluator_->evaluate_energy_only(
      probe_input_buffer_,
      selected_state_indices_,
      state_average_weights_,
      nuclear_repulsion_energy_);
  const double elapsed_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - evaluation_start_time)
          .count();
  ++energy_only_call_count_;
  energy_only_wall_time_seconds_ += elapsed_seconds;
  last_energy_only_wall_time_seconds_ = elapsed_seconds;
  return energy;
}

bool OrbitalObjective::canonicalize_orbital_chart_at_current_point(
    Eigen::VectorXd* parameter_vector,
    Eigen::VectorXd* gradient,
    std::vector<PackedSecantPair>* packed_secant_history) {
  bool chart_changed = false;
  const auto transform =
      apply_support_aware_inactive_mo_gauge_fix(
          &working_input_.orbital_preparation_input);
  if (transform.chart_changed) {
    transform_sparse_inactive_orbital_gradient(
        transform,
        working_input_.orbital_preparation_input,
        &last_gradient_result_.sparse_orbital_energy_gradient);

    if (!last_gradient_result_.sparse_orbital_reference_energy_gradient.empty()) {
      transform_sparse_inactive_orbital_gradient(
          transform,
          working_input_.orbital_preparation_input,
          &last_gradient_result_.sparse_orbital_reference_energy_gradient);
    }
    transport_packed_secant_history_with_support_aware_inactive_gauge(
        transform,
        working_input_.orbital_preparation_input,
        parameter_view_,
        packed_secant_history);
    refresh_cached_localized_representative_selector(
        working_input_.orbital_preparation_input,
        &last_gradient_result_.orbital_preparation_result);
    if (last_gradient_result_.second_order_context != nullptr) {
      refresh_cached_localized_representative_selector(
          working_input_.orbital_preparation_input,
          &last_gradient_result_
               .second_order_context
               ->prepared_active_space
               .orbital_result);
    }
    chart_changed = true;
  }

  if (!chart_changed) {
    return false;
  }
  probe_input_buffer_.orbital_preparation_input =
      working_input_.orbital_preparation_input;
  if (parameter_vector != nullptr) {
    *parameter_vector =
        parameter_view_.pack(working_input_.orbital_preparation_input);
  }
  if (gradient != nullptr) {
    *gradient = parameter_view_.gather_from_full(
        last_gradient_result_.sparse_orbital_energy_gradient);
    if (!gradient_inf_norm_history_.empty()) {
      gradient_inf_norm_history_.back() =
          gradient_infinity_norm(*gradient);
    }
  }
  return true;
}

OrbitalObjective OrbitalObjective::make_probe_copy() const {
  // HVP finite-difference probes and TN trial evaluations only need the
  // current immutable inputs plus the evaluator handles. Reconstruct a fresh
  // objective instead of copying the last accepted gradient result and
  // second-order context into another large object.
  OrbitalObjective copy(
      working_input_,
      parameter_view_,
      selected_state_indices_,
      state_average_weights_,
      nuclear_repulsion_energy_,
      orbital_gradient_evaluator_,
      scf_evaluator_);
  copy.probe_input_buffer_.orbital_preparation_input =
      working_input_.orbital_preparation_input;
  return copy;
}

}  // namespace xmvb::vb
