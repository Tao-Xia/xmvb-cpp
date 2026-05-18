#include "vb/scf/cpp_vb_scf_optimizer.hpp"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <LBFGS.h>

#include "vb/orbital/localized_representative_selector.hpp"
#include "vb/orbital/nonredundant_optimizer_input_adapter.hpp"
#include "vb/orbital/nonredundant_orbital_space.hpp"
#include "vb/orbital/sparse_orbital_parameter_view.hpp"
#include "vb/orbital/support_aware_mo_gauge_fix.hpp"
#include "vb/scf/exact_ctx_strategy_profile.hpp"
#include "vb/scf/exact_orbital_second_order_operator.hpp"

#ifdef XMVB_CPP_ENABLE_LEGACY_FORTRAN_BACKEND
extern "C" void lbfgs_driver_(
    int* n,
    int* m,
    double* x,
    double* energy,
    double* gradient,
    int* diagco,
    double* diag,
    int* iprint,
    double* eps,
    double* xtol,
    double* workspace,
    int* iflag,
    double* gxn);
#endif

namespace xmvb::vb {

namespace {

constexpr double kNonredundantPolishMinIterationSeconds = 5.0e-2;
constexpr double kLineSearchExpansionFactor = 10.0;
constexpr int kLegacyOrbitalTypeHao = 1;
constexpr int kLegacyOrbitalTypeBdo = 2;
constexpr int kLegacyOrbitalTypeOeo = 3;

Eigen::MatrixXd build_self_adjoint_matrix_power(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    double exponent,
    const char* label);

Eigen::MatrixXd build_inactive_metric_inverse(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_physical_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap_matrix);

struct PackedSecantPair {
  Eigen::VectorXd packed_step;
  Eigen::VectorXd packed_projected_gradient_change;
};

bool optimizer_backend_uses_nonredundant_space(
    CppVbScfOptimizerBackend backend) {
  switch (backend) {
    case CppVbScfOptimizerBackend::NonredundantProjectedGradient:
    case CppVbScfOptimizerBackend::NonredundantLbfgspp:
    case CppVbScfOptimizerBackend::NonredundantTruncatedNewton:
      return true;
    case CppVbScfOptimizerBackend::LegacyFortran:
    case CppVbScfOptimizerBackend::Lbfgspp:
    case CppVbScfOptimizerBackend::DeepVBHOnnx:
    case CppVbScfOptimizerBackend::DeepVBHOnnxDirectFinal:
      return false;
  }
  return false;
}

double gradient_infinity_norm(const Eigen::VectorXd& gradient) {
  double norm = 0.0;
  for (Eigen::Index index = 0; index < gradient.size(); ++index) {
    if (!std::isfinite(gradient[index])) {
      return std::numeric_limits<double>::infinity();
    }
    norm = std::max(norm, std::abs(gradient[index]));
  }
  return norm;
}

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
      orbital_preparation_input.active_orbital_overlap_matrix.data(),
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
      inactive_active_overlap_matrix.data() + inactive_active_overlap_matrix.size());
  const Eigen::MatrixXd projected_active_overlap_matrix =
      occupied_space_projector.transpose() * active_overlap_source;
  orbital_result->projected_active_overlap_matrix.assign(
      projected_active_overlap_matrix.data(),
      projected_active_overlap_matrix.data() + projected_active_overlap_matrix.size());

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
      orbital_preparation_input.active_orbital_overlap_matrix.data(),
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

bool is_effectively_zero_step(
    const Eigen::VectorXd& parameter_step,
    const Eigen::VectorXd& reference_parameters) {
  // Moré-Thuente returns the best trial point seen so far. When none of the
  // trial evaluations satisfy the Wolfe conditions, that "best" point may be
  // the original iterate itself, which appears as a zero update and can trap
  // the outer L-BFGS loop in repeated null iterations.
  constexpr double kRelativeStepTolerance =
      128.0 * std::numeric_limits<double>::epsilon();
  const double step_inf_norm = gradient_infinity_norm(parameter_step);
  const double reference_inf_norm =
      std::max(1.0, gradient_infinity_norm(reference_parameters));
  return step_inf_norm <= kRelativeStepTolerance * reference_inf_norm;
}

bool line_search_made_no_meaningful_progress(
    double reference_energy,
    double trial_energy,
    double previous_gradient_inf_norm,
    double current_gradient_inf_norm,
    double gradient_tolerance) {
  if (!std::isfinite(reference_energy) || !std::isfinite(trial_energy) ||
      !std::isfinite(previous_gradient_inf_norm) ||
      !std::isfinite(current_gradient_inf_norm)) {
    return false;
  }

  const double energy_scale = std::max(1.0, std::abs(reference_energy));
  const double energy_change =
      std::abs(trial_energy - reference_energy);
  constexpr double kRelativeEnergyProgressTolerance =
      4096.0 * std::numeric_limits<double>::epsilon();
  const double effective_energy_tolerance =
      kRelativeEnergyProgressTolerance * energy_scale;
  if (energy_change > effective_energy_tolerance) {
    return false;
  }
  if (previous_gradient_inf_norm < gradient_tolerance) {
    return false;
  }

  // A Wolfe line search that returns nearly the same energy while leaving the
  // gradient essentially unchanged is not a real accepted iteration for this
  // noisy parallel objective. Treat it as a line-search stall and fall back to
  // Armijo steepest descent from the previous iterate.
  return current_gradient_inf_norm >= 0.9 * previous_gradient_inf_norm;
}

bool finite_vector_matches_size(
    const Eigen::VectorXd& vector,
    Eigen::Index expected_size) {
  return vector.size() == expected_size &&
      vector.allFinite();
}

bool finite_nonzero_vector_matches_size(
    const Eigen::VectorXd& vector,
    Eigen::Index expected_size) {
  return finite_vector_matches_size(vector, expected_size) &&
      vector.squaredNorm() > 0.0;
}

const char* bool_name(bool value) {
  return value ? "true" : "false";
}

bool parse_env_flag_with_default(
    const char* variable_name,
    bool default_value) {
  const char* value = std::getenv(variable_name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  return std::strcmp(value, "0") != 0 &&
      std::strcmp(value, "false") != 0 &&
      std::strcmp(value, "FALSE") != 0;
}

std::optional<bool> parse_env_optional_flag(
    const char* variable_name) {
  const char* value = std::getenv(variable_name);
  if (value == nullptr || value[0] == '\0') {
    return std::nullopt;
  }
  return std::strcmp(value, "0") != 0 &&
      std::strcmp(value, "false") != 0 &&
      std::strcmp(value, "FALSE") != 0;
}

bool oeo_active_representative_accepted_point_canonicalization_enabled() {
  const auto enable_override =
      parse_env_optional_flag(
          "XMVB_CPP_ENABLE_OEO_ACTIVE_REPRESENTATIVE_CANONICALIZATION");
  if (enable_override.has_value()) {
    return *enable_override;
  }
  const auto disable_override =
      parse_env_optional_flag(
          "XMVB_CPP_DISABLE_OEO_ACTIVE_REPRESENTATIVE_CANONICALIZATION");
  if (disable_override.has_value()) {
    return !*disable_override;
  }
  // The OEO representative choice is a gauge/output convention. Applying that
  // reset at every accepted optimization point perturbs the common OEO chart,
  // transported secant pairs, and exact_ctx/TiCl convergence. Keep the
  // accepted-point reset off by default and reserve it for targeted debugging.
  return false;
}

int parse_env_int_with_default(
    const char* variable_name,
    int default_value) {
  const char* value = std::getenv(variable_name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }

  errno = 0;
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (errno != 0 || end == value || (end != nullptr && end[0] != '\0') ||
      parsed < static_cast<long>(std::numeric_limits<int>::min()) ||
      parsed > static_cast<long>(std::numeric_limits<int>::max())) {
    return default_value;
  }
  return static_cast<int>(parsed);
}

double parse_env_double_with_default(
    const char* variable_name,
    double default_value) {
  const char* value = std::getenv(variable_name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }

  errno = 0;
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (errno != 0 || end == value || (end != nullptr && end[0] != '\0') ||
      !std::isfinite(parsed)) {
    return default_value;
  }
  return parsed;
}

std::optional<bool> exact_ctx_inner_solve_outer_response_override() {
  return parse_env_optional_flag(
      "XMVB_CPP_EXACT_CTX_INNER_SOLVE_USE_OUTER_RESPONSE");
}

bool exact_ctx_sr1_outer_response_approximation_enabled() {
  return parse_env_flag_with_default(
      "XMVB_CPP_EXACT_CTX_SR1_OUTER_APPROX",
      false);
}

int exact_ctx_sr1_outer_response_history_size() {
  return std::max(
      0,
      parse_env_int_with_default(
          "XMVB_CPP_EXACT_CTX_SR1_OUTER_HISTORY_SIZE",
          6));
}

double exact_ctx_sr1_outer_response_min_alignment() {
  return std::max(
      0.0,
      parse_env_double_with_default(
          "XMVB_CPP_EXACT_CTX_SR1_OUTER_MIN_ALIGNMENT",
          1.0e-8));
}

double exact_ctx_sr1_outer_response_max_correction_ratio() {
  return std::max(
      0.0,
      parse_env_double_with_default(
          "XMVB_CPP_EXACT_CTX_SR1_OUTER_MAX_CORRECTION_RATIO",
          0.5));
}

double exact_ctx_sr1_outer_response_initial_scale() {
  return parse_env_double_with_default(
      "XMVB_CPP_EXACT_CTX_SR1_OUTER_INITIAL_SCALE",
      0.0);
}

double exact_ctx_sr1_outer_response_clear_trust_ratio() {
  return std::clamp(
      parse_env_double_with_default(
          "XMVB_CPP_EXACT_CTX_SR1_OUTER_CLEAR_TRUST_RATIO",
          0.35),
      0.0,
      1.0);
}

int exact_ctx_startup_full_inner_solve_begin() {
  return std::max(
      0,
      parse_env_int_with_default(
          "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_BEGIN",
          0));
}

int exact_ctx_startup_full_inner_solve_count() {
  return std::max(
      0,
      parse_env_int_with_default(
          "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_COUNT",
          2));
}

int exact_ctx_startup_full_inner_solve_max_extra_count() {
  return std::max(
      0,
      parse_env_int_with_default(
          "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_MAX_EXTRA_COUNT",
          0));
}

double exact_ctx_startup_full_inner_solve_gradient_ratio_threshold() {
  return std::max(
      0.0,
      parse_env_double_with_default(
          "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_GRAD_RATIO_THRESHOLD",
          0.2));
}

int exact_ctx_startup_full_inner_solve_base_max_cg_iterations() {
  return std::max(
      1,
      parse_env_int_with_default(
          "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_BASE_MAX_CG_ITERATIONS",
          6));
}

int exact_ctx_startup_full_inner_solve_tail_max_cg_iterations() {
  return std::max(
      1,
      parse_env_int_with_default(
          "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_TAIL_MAX_CG_ITERATIONS",
          8));
}

int exact_ctx_startup_full_inner_solve_multi_step_max_active_orbitals() {
  return std::max(
      0,
      parse_env_int_with_default(
          "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_MULTI_STEP_MAX_ACTIVE_ORBITALS",
          8));
}

int exact_ctx_startup_full_inner_solve_enable_max_active_orbitals() {
  return std::max(
      0,
      parse_env_int_with_default(
          "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_ENABLE_MAX_ACTIVE_ORBITALS",
          6));
}

int exact_ctx_full_model_correction_enable_max_active_orbitals() {
  return std::max(
      0,
      parse_env_int_with_default(
          "XMVB_CPP_EXACT_CTX_FULL_MODEL_CORRECTION_ENABLE_MAX_ACTIVE_ORBITALS",
          6));
}

int exact_ctx_hybrid_followup_full_inner_solve_enable_max_active_orbitals() {
  return std::max(
      0,
      parse_env_int_with_default(
          "XMVB_CPP_EXACT_CTX_HYBRID_FOLLOWUP_FULL_ENABLE_MAX_ACTIVE_ORBITALS",
          6));
}

double exact_ctx_hybrid_followup_full_inner_solve_max_outer_response_cost_fraction() {
  return std::clamp(
      parse_env_double_with_default(
          "XMVB_CPP_EXACT_CTX_HYBRID_FOLLOWUP_FULL_MAX_OUTER_RESPONSE_COST_FRACTION",
          0.4),
      0.0,
      1.0);
}

double exact_ctx_hybrid_followup_full_inner_solve_max_outer_response_to_objective_time_ratio() {
  return std::max(
      0.0,
      parse_env_double_with_default(
          "XMVB_CPP_EXACT_CTX_HYBRID_FOLLOWUP_FULL_MAX_OUTER_RESPONSE_TO_OBJECTIVE_TIME_RATIO",
          0.1));
}

double exact_ctx_hybrid_bridge_low_trust_ratio_threshold() {
  return std::clamp(
      parse_env_double_with_default(
          "XMVB_CPP_EXACT_CTX_HYBRID_BRIDGE_LOW_TRUST_RATIO_THRESHOLD",
          0.35),
      0.0,
      1.0);
}

int exact_ctx_stall_tail_boost_min_consecutive_projected_stalls() {
  return std::max(
      0,
      parse_env_int_with_default(
          "XMVB_CPP_EXACT_CTX_STALL_TAIL_BOOST_MIN_CONSECUTIVE_PROJECTED_STALLS",
          2));
}

int exact_ctx_stall_tail_boost_max_cg_iterations() {
  return std::max(
      1,
      parse_env_int_with_default(
          "XMVB_CPP_EXACT_CTX_STALL_TAIL_BOOST_MAX_CG_ITERATIONS",
          12));
}

int exact_ctx_tail_full_inner_solve_enable_max_active_orbitals() {
  return std::max(
      0,
      parse_env_int_with_default(
          "XMVB_CPP_EXACT_CTX_TAIL_FULL_INNER_SOLVE_ENABLE_MAX_ACTIVE_ORBITALS",
          0));
}

int exact_ctx_tail_full_inner_solve_min_consecutive_projected_stalls() {
  return std::max(
      0,
      parse_env_int_with_default(
          "XMVB_CPP_EXACT_CTX_TAIL_FULL_INNER_SOLVE_MIN_CONSECUTIVE_PROJECTED_STALLS",
          24));
}

int exact_ctx_tail_full_inner_solve_cooldown_accepted_iterations() {
  return std::max(
      0,
      parse_env_int_with_default(
          "XMVB_CPP_EXACT_CTX_TAIL_FULL_INNER_SOLVE_COOLDOWN",
          24));
}

double exact_ctx_tail_full_inner_solve_gradient_tolerance_multiple() {
  return std::max(
      0.0,
      parse_env_double_with_default(
          "XMVB_CPP_EXACT_CTX_TAIL_FULL_INNER_SOLVE_GRADIENT_TOLERANCE_MULTIPLE",
          16.0));
}

double nonredundant_truncated_newton_relative_residual_target_fraction() {
  return std::clamp(
      parse_env_double_with_default(
          "XMVB_CPP_TN_RELATIVE_RESIDUAL_TARGET_FRACTION",
          0.02),
      0.0,
      1.0);
}

double nonredundant_truncated_newton_absolute_residual_target_gradient_multiple() {
  return std::max(
      0.0,
      parse_env_double_with_default(
          "XMVB_CPP_TN_ABSOLUTE_RESIDUAL_TARGET_GRAD_MULTIPLE",
          0.1));
}

bool truncated_newton_disable_curvature_preconditioner() {
  return parse_env_flag_with_default(
      "XMVB_CPP_DISABLE_TN_CURVATURE_PRECONDITIONER",
      false);
}

bool truncated_newton_use_block_preconditioner(
    const NonredundantOrbitalSpace& current_space) {
  const auto override = parse_env_optional_flag(
      "XMVB_CPP_TN_USE_BLOCK_PRECONDITIONER");
  if (override.has_value()) {
    return *override;
  }
  return current_space.use_block_preconditioner_by_default();
}

bool exact_ctx_tiny_full_fd_fallback_enabled() {
  return parse_env_flag_with_default(
      "XMVB_CPP_EXACT_CTX_TINY_FULL_FD_FALLBACK",
      true);
}

bool exact_ctx_use_tiny_full_fd_fallback(
    const ExactCtxSystemProfile& system_profile) {
  return
      exact_ctx_tiny_full_fd_fallback_enabled() &&
      !system_profile.open_shell &&
      system_profile.sparse_orbital_chart &&
      system_profile.n_active_orbitals > 0 &&
      system_profile.n_active_orbitals <= 2 &&
      system_profile.n_basis_functions <= 64;
}

int exact_ctx_stall_full_model_correction_enable_max_active_orbitals() {
  return std::max(
      0,
      parse_env_int_with_default(
          "XMVB_CPP_EXACT_CTX_STALL_FULL_MODEL_CORRECTION_ENABLE_MAX_ACTIVE_ORBITALS",
          6));
}

bool exact_ctx_stall_tail_boost_enabled(
    int consecutive_projected_stall_count) {
  return consecutive_projected_stall_count >=
      exact_ctx_stall_tail_boost_min_consecutive_projected_stalls();
}

bool exact_ctx_full_model_correction_allowed(
    const ExactCtxSystemProfile& system_profile,
    int consecutive_projected_stall_count) {
  if (system_profile.n_active_orbitals <=
      exact_ctx_full_model_correction_enable_max_active_orbitals()) {
    return true;
  }

  // Open-shell sparse charts are the regime where the cheap core-only Hessian
  // most often gives the right descent direction but poor trust-ratio
  // calibration.  Allow a rejected-step full-model bridge through active=8
  // without promoting every accepted-point solve to the full outer response.
  if (system_profile.open_shell &&
      system_profile.sparse_orbital_chart &&
      system_profile.n_active_orbitals <= 8) {
    return true;
  }

  // Larger active spaces should keep the cheap exact-ctx path during the early
  // high-gradient regime. The stall gate keeps full-model corrections as a
  // tail rescue instead of a steady-state cost.
  return exact_ctx_stall_tail_boost_enabled(
             consecutive_projected_stall_count) &&
      system_profile.n_active_orbitals <=
          exact_ctx_stall_full_model_correction_enable_max_active_orbitals();
}

struct ExactCtxInnerSolvePolicy {
  bool use_outer_response = false;
  bool forced_override = false;
  bool used_gradient_tail_extension = false;
  bool used_hybrid_followup_full_solve = false;
  bool used_stall_tail_full_solve = false;
};

struct ExactCtxHybridStrategyState {
  bool request_followup = false;
  int hybrid_followup_cooldown_remaining = 0;
  int tail_full_solve_cooldown_remaining = 0;
  bool has_full_operator_cost_sample = false;
  double last_outer_response_cost_fraction =
      std::numeric_limits<double>::infinity();
  double last_outer_response_average_wall_time_seconds =
      std::numeric_limits<double>::infinity();
};

struct HvpModelQualityState {
  int consecutive_low_trust = 0;
  int consecutive_cheap_rejects = 0;
  int consecutive_projected_stalls = 0;
  int force_full_retry_remaining = 0;
  int force_full_inner_solve_remaining = 0;

  bool wants_full_retry() const noexcept {
    return force_full_retry_remaining > 0 ||
        consecutive_low_trust > 0 ||
        consecutive_cheap_rejects > 0 ||
        consecutive_projected_stalls >= 2;
  }

  bool wants_full_inner_solve() const noexcept {
    return force_full_inner_solve_remaining > 0;
  }

  void decay_request_counters() noexcept {
    if (force_full_retry_remaining > 0) --force_full_retry_remaining;
    if (force_full_inner_solve_remaining > 0) {
      --force_full_inner_solve_remaining;
    }
  }

  void observe_rejected_cheap_trial() noexcept {
    ++consecutive_cheap_rejects;
    force_full_retry_remaining =
        std::max(force_full_retry_remaining, 2);
    if (consecutive_cheap_rejects >= 2) {
      force_full_inner_solve_remaining =
          std::max(force_full_inner_solve_remaining, 1);
    }
  }

  void observe_accepted_step(
      double trust_ratio,
      bool cheap_trial_rejected,
      bool used_full_model_probe,
      bool projected_stall,
      bool used_krylov_rescue_step) noexcept {
    const bool low_trust =
        !std::isfinite(trust_ratio) || trust_ratio < 0.35;
    consecutive_low_trust =
        low_trust ? consecutive_low_trust + 1 : 0;
    consecutive_cheap_rejects =
        cheap_trial_rejected ? consecutive_cheap_rejects + 1 : 0;
    consecutive_projected_stalls =
        projected_stall ? consecutive_projected_stalls + 1 : 0;

    if (low_trust || cheap_trial_rejected || used_krylov_rescue_step ||
        consecutive_projected_stalls >= 2) {
      force_full_retry_remaining =
          std::max(force_full_retry_remaining, 2);
    }
    if (used_krylov_rescue_step || consecutive_cheap_rejects >= 2 ||
        consecutive_low_trust >= 2) {
      force_full_inner_solve_remaining =
          std::max(force_full_inner_solve_remaining, 1);
    }
    if (used_full_model_probe && !low_trust && !cheap_trial_rejected &&
        !projected_stall && trust_ratio >= 0.75) {
      consecutive_low_trust = 0;
      consecutive_cheap_rejects = 0;
      consecutive_projected_stalls = 0;
    }
  }

  void reset_after_chart_change() noexcept {
    consecutive_low_trust = 0;
    consecutive_cheap_rejects = 0;
    consecutive_projected_stalls = 0;
    force_full_retry_remaining = 0;
    force_full_inner_solve_remaining = 0;
  }
};

bool exact_ctx_log_tnhvp_policy() {
  return parse_env_flag_with_default(
      "XMVB_CPP_LOG_TNHVP_POLICY",
      false);
}

bool exact_ctx_log_hvp_diagnostics() {
  return parse_env_flag_with_default(
      "XMVB_CPP_LOG_EXACT_CTX_HVP_DIAGNOSTICS",
      false);
}

double exact_ctx_average_stage_wall_time_seconds(
    double total_wall_time_seconds,
    std::size_t apply_count) {
  if (apply_count == 0 || !std::isfinite(total_wall_time_seconds) ||
      total_wall_time_seconds < 0.0) {
    return 0.0;
  }
  return total_wall_time_seconds / static_cast<double>(apply_count);
}

void maybe_log_exact_ctx_policy_decision(
    int accepted_iteration_index,
    bool sparse_orbital_chart,
    int n_active_orbitals,
    double reduced_gradient_inf_norm,
    double latest_objective_seconds,
    bool full_model_correction_allowed,
    bool full_retry_available,
    const ExactCtxHybridStrategyState& hybrid_strategy_state,
    const ExactCtxInnerSolvePolicy& policy) {
  if (!exact_ctx_log_tnhvp_policy()) {
    return;
  }
  std::ostringstream stream;
  stream << "tnhvp_policy"
         << " iter=" << accepted_iteration_index
         << " sparse=" << bool_name(sparse_orbital_chart)
         << " n_active=" << n_active_orbitals
         << " grad_inf=" << std::scientific << std::setprecision(6)
         << reduced_gradient_inf_norm
         << " latest_obj_s=" << std::fixed << std::setprecision(6)
         << latest_objective_seconds
         << " outer=" << bool_name(policy.use_outer_response)
         << " forced=" << bool_name(policy.forced_override)
         << " tail_ext=" << bool_name(policy.used_gradient_tail_extension)
         << " followup_req=" << bool_name(hybrid_strategy_state.request_followup)
         << " followup_cd="
         << hybrid_strategy_state.hybrid_followup_cooldown_remaining
         << " tail_full_cd="
         << hybrid_strategy_state.tail_full_solve_cooldown_remaining
         << " followup_full="
         << bool_name(policy.used_hybrid_followup_full_solve)
         << " stall_full="
         << bool_name(policy.used_stall_tail_full_solve)
         << " full_corr_allowed="
         << bool_name(full_model_correction_allowed)
         << " full_retry_ready=" << bool_name(full_retry_available)
         << '\n';
  std::cerr << stream.str();
  std::cerr.flush();
}

void maybe_log_exact_ctx_policy_outcome(
    int accepted_iteration_index,
    bool accepted_trial,
    double trust_ratio,
    double trust_radius,
    bool cheap_trial_rejected,
    bool full_retry_attempted,
    bool full_operator_step_refined,
    int transport_history_limit,
    int packed_secant_history_size,
    int transported_preconditioner_size,
    bool reused_cheap_krylov_subspace,
    bool reused_full_krylov_subspace,
    bool transported_warm_start_admitted,
    bool used_initial_step,
    bool warm_start_hvp_performed,
    int cg_iterations,
    bool projected_stall,
    int consecutive_projected_stall_count,
    bool request_followup_next_iteration,
    bool used_krylov_rescue_step,
    bool reached_boundary,
    bool encountered_negative_curvature,
    double reduced_step_norm,
    double predicted_decrease) {
  if (!exact_ctx_log_tnhvp_policy()) {
    return;
  }
  std::ostringstream stream;
  stream << "tnhvp_outcome"
         << " iter=" << accepted_iteration_index
         << " accepted=" << bool_name(accepted_trial)
         << " trust=" << std::fixed << std::setprecision(6)
         << trust_ratio
         << " radius=" << trust_radius
         << " cheap_reject=" << bool_name(cheap_trial_rejected)
         << " full_retry=" << bool_name(full_retry_attempted)
         << " refined=" << bool_name(full_operator_step_refined)
         << " hist_cap=" << transport_history_limit
         << " hist_stored=" << packed_secant_history_size
         << " hist_used=" << transported_preconditioner_size
         << " cheap_krylov_reuse=" << bool_name(reused_cheap_krylov_subspace)
         << " full_krylov_reuse=" << bool_name(reused_full_krylov_subspace)
         << " warm_admit=" << bool_name(transported_warm_start_admitted)
         << " warm_used=" << bool_name(used_initial_step)
         << " warm_hvp=" << bool_name(warm_start_hvp_performed)
         << " cg_iters=" << cg_iterations
         << " krylov_rescue=" << bool_name(used_krylov_rescue_step)
         << " boundary=" << bool_name(reached_boundary)
         << " neg_curv=" << bool_name(encountered_negative_curvature)
         << " step_norm=" << reduced_step_norm
         << " pred_dec=" << predicted_decrease
         << " projected_stall=" << bool_name(projected_stall)
         << " stall_count=" << consecutive_projected_stall_count
         << " followup_next=" << bool_name(request_followup_next_iteration)
         << '\n';
  std::cerr << stream.str();
  std::cerr.flush();
}

bool exact_ctx_hybrid_followup_full_inner_solve_allowed(
    const ExactCtxSystemProfile& system_profile,
    double latest_objective_seconds,
    const ExactCtxHybridStrategyState& hybrid_strategy_state) {
  const ExactCtxDefaultStrategy strategy =
      choose_exact_ctx_default_strategy(system_profile);
  const auto force_followup_full_solve =
      parse_env_optional_flag(
          "XMVB_CPP_EXACT_CTX_HYBRID_FOLLOWUP_FULL_SOLVE");
  if (force_followup_full_solve.has_value() &&
      !*force_followup_full_solve) {
    return false;
  }
  const bool followup_full_solve_forced =
      force_followup_full_solve.value_or(false);
  const bool open_shell_sparse_active8_followup =
      system_profile.open_shell &&
      system_profile.sparse_orbital_chart &&
      system_profile.n_active_orbitals == 8;
  if (!followup_full_solve_forced &&
      !strategy.allow_hybrid_followup_full_solve) {
    return false;
  }
  if (!hybrid_strategy_state.request_followup ||
      hybrid_strategy_state.hybrid_followup_cooldown_remaining > 0 ||
      !hybrid_strategy_state.has_full_operator_cost_sample) {
    return false;
  }
  if (!followup_full_solve_forced &&
      !open_shell_sparse_active8_followup &&
      system_profile.n_active_orbitals >
          exact_ctx_hybrid_followup_full_inner_solve_enable_max_active_orbitals()) {
    return false;
  }

  // Small full-AO `orbtyp=oeo` radicals are the exact regime where the
  // accepted-point cheap step plus one full-model probe/refinement already
  // captures the useful outer-response correction. Promoting the *next*
  // iteration to a full accepted-point TN solve over-corrects TiCl/FeCl-class
  // active orbitals and pushes them away from the legacy XMVB localized
  // solution, even though the same full probe is still valuable as a
  // single-step model-mismatch diagnostic. Keep that cheap-step probe path,
  // but do not arm the followup full solve by default on the dense OEO chart.
  if (!system_profile.sparse_orbital_chart &&
      system_profile.n_active_orbitals <=
          exact_ctx_hybrid_followup_full_inner_solve_enable_max_active_orbitals() &&
      !followup_full_solve_forced) {
    return false;
  }

  if (followup_full_solve_forced || open_shell_sparse_active8_followup) {
    return true;
  }

  const bool cost_fraction_is_affordable =
      std::isfinite(hybrid_strategy_state.last_outer_response_cost_fraction) &&
      hybrid_strategy_state.last_outer_response_cost_fraction <=
          exact_ctx_hybrid_followup_full_inner_solve_max_outer_response_cost_fraction();
  const bool outer_response_time_is_affordable =
      std::isfinite(
          hybrid_strategy_state.last_outer_response_average_wall_time_seconds) &&
      latest_objective_seconds > 0.0 &&
      hybrid_strategy_state.last_outer_response_average_wall_time_seconds <=
          exact_ctx_hybrid_followup_full_inner_solve_max_outer_response_to_objective_time_ratio() *
              latest_objective_seconds;
  return cost_fraction_is_affordable || outer_response_time_is_affordable;
}

bool exact_ctx_stall_tail_full_inner_solve_allowed(
    const ExactCtxSystemProfile& system_profile,
    double current_projected_gradient_inf_norm,
    double gradient_tolerance,
    int consecutive_projected_stall_count,
    const ExactCtxHybridStrategyState& hybrid_strategy_state) {
  const int max_active_orbitals =
      exact_ctx_tail_full_inner_solve_enable_max_active_orbitals();
  if (max_active_orbitals <= 0 ||
      system_profile.n_active_orbitals > max_active_orbitals ||
      hybrid_strategy_state.tail_full_solve_cooldown_remaining > 0) {
    return false;
  }
  if (consecutive_projected_stall_count <
      exact_ctx_tail_full_inner_solve_min_consecutive_projected_stalls()) {
    return false;
  }
  const double gradient_multiple =
      exact_ctx_tail_full_inner_solve_gradient_tolerance_multiple();
  if (gradient_multiple > 0.0 &&
      (!std::isfinite(current_projected_gradient_inf_norm) ||
       current_projected_gradient_inf_norm >
           gradient_multiple * std::max(gradient_tolerance, 0.0))) {
    return false;
  }

  // Full outer-response HVPs are too expensive to use as the default steady-
  // state model, but the dense OEO TiCl-class tail shows that a sparse pulse of
  // the relaxed model can break long projected-gradient plateaus.  This gate
  // therefore waits for sustained projected stalls, limits the active-space
  // size, and then arms only one full accepted-point solve before cooldown.
  return true;
}

ExactCtxInnerSolvePolicy choose_exact_ctx_inner_solve_policy(
    int accepted_iteration_index,
    const ExactCtxSystemProfile& system_profile,
    double current_projected_gradient_inf_norm,
    double initial_projected_gradient_inf_norm,
    double gradient_tolerance,
    double latest_objective_seconds,
    int consecutive_projected_stall_count,
    const ExactCtxHybridStrategyState& hybrid_strategy_state) {
  // Keep the exact-ctx TN inner solve on the cheap core-only model by default,
  // and only spend full outer-response work in regimes where it materially
  // improves the accepted-point Newton model. Even very small full-AO active
  // spaces should stay on the cheaper core-only model by default because on
  // FeCl2-class systems the extra relaxed-response cost buys fewer outer
  // iterations but more total wall time. Full solves are therefore reserved
  // for explicit overrides, startup calibration windows, and later
  // cheap-step/full-model corrections that have already proven worthwhile.
  ExactCtxInnerSolvePolicy policy;
  const auto override = exact_ctx_inner_solve_outer_response_override();
  if (override.has_value()) {
    policy.use_outer_response = *override;
    policy.forced_override = true;
    return policy;
  }
  const ExactCtxDefaultStrategy strategy =
      choose_exact_ctx_default_strategy(system_profile);
  if (exact_ctx_hybrid_followup_full_inner_solve_allowed(
          system_profile,
          latest_objective_seconds,
          hybrid_strategy_state)) {
    // A prior cheap-step/full-model bridge already established that the full
    // operator is both useful and not too expensive on this system. Promote the
    // next accepted-point solve to one full-model TN step instead of repeating
    // the cheaper solve plus another diagnostic bridge.
    policy.use_outer_response = true;
    policy.used_hybrid_followup_full_solve = true;
    return policy;
  }
  if (exact_ctx_stall_tail_full_inner_solve_allowed(
          system_profile,
          current_projected_gradient_inf_norm,
          gradient_tolerance,
          consecutive_projected_stall_count,
          hybrid_strategy_state)) {
    policy.use_outer_response = true;
    policy.used_stall_tail_full_solve = true;
    return policy;
  }
  const int startup_full_enable_max_active_orbitals =
      std::max(
          0,
          parse_env_int_with_default(
              "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_ENABLE_MAX_ACTIVE_ORBITALS",
              strategy.startup_full_inner_solve_enable_max_active_orbitals));
  if (system_profile.n_active_orbitals >
      startup_full_enable_max_active_orbitals) {
    return policy;
  }
  const int full_inner_solve_begin =
      std::max(
          0,
          parse_env_int_with_default(
              "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_BEGIN",
              strategy.startup_full_inner_solve_begin));
  int full_inner_solve_count =
      std::max(
          0,
          parse_env_int_with_default(
              "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_COUNT",
              strategy.startup_full_inner_solve_count));
  int max_extra_count =
      std::max(
          0,
          parse_env_int_with_default(
              "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_MAX_EXTRA_COUNT",
              strategy.startup_full_inner_solve_max_extra_count));
  const int startup_multi_step_max_active_orbitals =
      std::max(
          0,
          parse_env_int_with_default(
              "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_MULTI_STEP_MAX_ACTIVE_ORBITALS",
              strategy.startup_full_inner_solve_multi_step_max_active_orbitals));
  if (system_profile.n_active_orbitals >
      startup_multi_step_max_active_orbitals) {
    full_inner_solve_count = std::min(full_inner_solve_count, 1);
    max_extra_count = 0;
  }
  if (accepted_iteration_index < full_inner_solve_begin) {
    return policy;
  }

  const int startup_window_offset =
      accepted_iteration_index - full_inner_solve_begin;
  if (startup_window_offset < full_inner_solve_count) {
    policy.use_outer_response = true;
    return policy;
  }

  const int startup_extra_offset =
      startup_window_offset - full_inner_solve_count;
  if (startup_extra_offset >= max_extra_count) {
    return policy;
  }

  constexpr double kGradientToleranceFloorMultiple = 8.0;
  const double reference_projected_gradient_inf_norm =
      std::max(initial_projected_gradient_inf_norm, gradient_tolerance);
  const double gradient_threshold =
      std::max(
          kGradientToleranceFloorMultiple * gradient_tolerance,
          exact_ctx_startup_full_inner_solve_gradient_ratio_threshold() *
              reference_projected_gradient_inf_norm);
  if (std::isfinite(current_projected_gradient_inf_norm) &&
      current_projected_gradient_inf_norm > gradient_threshold) {
    policy.use_outer_response = true;
    policy.used_gradient_tail_extension = true;
  }
  return policy;
}

bool exact_ctx_retry_rejected_step_with_full_operator(
    const ExactCtxDefaultStrategy& strategy) {
  const auto env_override = parse_env_optional_flag(
      "XMVB_CPP_EXACT_CTX_RETRY_REJECTED_WITH_FULL_OPERATOR");
  if (env_override.has_value()) {
    return *env_override;
  }
  return strategy.retry_rejected_step_with_full_operator;
}

class OrbitalObjective;

int exact_ctx_hybrid_refinement_max_cg_iterations() {
  return std::max(
      0,
      parse_env_int_with_default(
          "XMVB_CPP_EXACT_CTX_HYBRID_REFINE_MAX_CG_ITERATIONS",
          4));
}

int exact_ctx_sparse_hybrid_followup_probe_cooldown_accepted_iterations() {
  return std::max(
      0,
      parse_env_int_with_default(
          "XMVB_CPP_EXACT_CTX_SPARSE_FOLLOWUP_PROBE_COOLDOWN",
          2));
}

int exact_ctx_dense_hybrid_followup_probe_cooldown_accepted_iterations() {
  return std::max(
      0,
      parse_env_int_with_default(
          "XMVB_CPP_EXACT_CTX_DENSE_FOLLOWUP_PROBE_COOLDOWN",
          8));
}

int exact_ctx_hybrid_followup_probe_cooldown_accepted_iterations(
    bool sparse_orbital_chart) {
  return sparse_orbital_chart
      ? exact_ctx_sparse_hybrid_followup_probe_cooldown_accepted_iterations()
      : exact_ctx_dense_hybrid_followup_probe_cooldown_accepted_iterations();
}

class OrbitalObjective {
public:
  struct TrialEvaluation {
    OrbitalPreparationInput orbital_preparation_input;
    CppOrbitalGradientResult gradient_result;
    Eigen::VectorXd gradient;
    double energy = 0.0;
    double gradient_inf_norm = 0.0;
    double wall_time_seconds = 0.0;
    bool valid = false;
  };

  OrbitalObjective(
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

  double operator()(const Eigen::VectorXd& parameter_vector, Eigen::VectorXd& gradient) {
    TrialEvaluation evaluation =
        evaluate_trial_without_committing(parameter_vector);
    gradient = std::move(evaluation.gradient);
    const double energy = evaluation.energy;
    commit_trial_evaluation(std::move(evaluation));
    return energy;
  }

  const CppVbInput& last_input() const { return working_input_; }
  const CppOrbitalGradientResult& last_gradient_result() const { return last_gradient_result_; }
  const std::shared_ptr<CppActiveSpaceSecondOrderContext>&
  last_second_order_context() const {
    return last_gradient_result_.second_order_context;
  }
  void set_oeo_active_reference_orbitals(
      const Eigen::Ref<const Eigen::MatrixXd>& normalized_orbital_matrix) {
    if (normalized_orbital_matrix.size() == 0) {
      initial_oeo_reference_orbital_matrix_ = Eigen::MatrixXd();
      return;
    }
    if (normalized_orbital_matrix.rows() !=
            working_input_.orbital_preparation_input.n_basis_functions ||
        normalized_orbital_matrix.cols() !=
            working_input_.orbital_preparation_input.n_orbitals) {
      throw std::invalid_argument(
          "initial OEO reference orbital matrix dimensions do not match the optimizer chart");
    }
    initial_oeo_reference_orbital_matrix_ = normalized_orbital_matrix;
  }
  void ensure_last_reference_energy_gradient() {
    orbital_gradient_evaluator_->populate_reference_energy_gradient(
        working_input_,
        &last_gradient_result_);
  }
  const std::vector<double>& energy_history() const { return energy_history_; }
  const std::vector<double>& gradient_inf_norm_history() const { return gradient_inf_norm_history_; }
  const std::vector<double>& iteration_time_history_seconds() const {
    return iteration_time_history_seconds_;
  }
  std::size_t call_count() const { return energy_history_.size(); }
  double objective_wall_time_seconds() const { return objective_wall_time_seconds_; }
  std::size_t energy_only_call_count() const { return energy_only_call_count_; }
  double energy_only_wall_time_seconds() const {
    return energy_only_wall_time_seconds_;
  }
  double last_energy_only_wall_time_seconds() const {
    return last_energy_only_wall_time_seconds_;
  }

  double last_gradient_inf_norm() const {
    if (gradient_inf_norm_history_.empty()) {
      return 0.0;
    }
    return gradient_inf_norm_history_.back();
  }

  TrialEvaluation evaluate_trial_without_committing(
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

  void commit_trial_evaluation(TrialEvaluation evaluation) {
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

  double evaluate_energy_only(const Eigen::VectorXd& parameter_vector) const {
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

  bool canonicalize_orbital_chart_at_current_point(
      Eigen::VectorXd* parameter_vector,
      Eigen::VectorXd* gradient,
      std::vector<PackedSecantPair>* packed_secant_history = nullptr) {
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

    if (canonicalize_oeo_active_representative_at_current_point(
            packed_secant_history)) {
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

  OrbitalObjective make_probe_copy() const {
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
    copy.initial_oeo_reference_orbital_matrix_ =
        initial_oeo_reference_orbital_matrix_;
    return copy;
  }

private:
  bool canonicalize_oeo_active_representative_at_current_point(
      std::vector<PackedSecantPair>* packed_secant_history) {
    // This accepted-point chart repair keeps the OEO auxiliary active block
    // fixed while refreshing the inactive-null representative of the physical
    // active orbitals. That representative choice is handled as an export gauge
    // by default; enabling the accepted-point reset here restores the older
    // behavior for targeted debugging.
    if (!oeo_active_representative_accepted_point_canonicalization_enabled()) {
      return false;
    }
    const auto& orbital_preparation_input =
        working_input_.orbital_preparation_input;
    if (orbital_preparation_input.orbital_type != kLegacyOrbitalTypeOeo) {
      return false;
    }
    if (initial_oeo_reference_orbital_matrix_.size() == 0) {
      return false;
    }

    const int n_basis_functions =
        orbital_preparation_input.n_basis_functions;
    const int n_inactive_doubly_occupied_orbitals =
        (orbital_preparation_input.n_total_electrons -
         orbital_preparation_input.n_active_electrons) / 2;
    const int n_active_orbitals =
        orbital_preparation_input.n_active_orbitals;
    if (n_inactive_doubly_occupied_orbitals <= 0 || n_active_orbitals <= 0) {
      return false;
    }
    if (initial_oeo_reference_orbital_matrix_.rows() != n_basis_functions ||
        initial_oeo_reference_orbital_matrix_.cols() !=
            orbital_preparation_input.n_orbitals) {
      throw std::runtime_error(
          "stored OEO active reference does not match the current orbital chart");
    }

    auto& orbital_result = last_gradient_result_.orbital_preparation_result;
    const auto& physical_orbital_frame =
        orbital_result.physical_orbital_frame;
    if (physical_orbital_frame.normalized_orbital_matrix.rows() !=
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
          "cached OEO orbital preparation result is incomplete at the accepted point");
    }

    const Eigen::MatrixXd repaired_normalized_orbital_matrix =
        build_metric_preserving_oeo_repaired_normalized_orbital_matrix(
            orbital_preparation_input,
            orbital_result,
            initial_oeo_reference_orbital_matrix_);
    const Eigen::MatrixXd repaired_active_physical_orbitals =
        repaired_normalized_orbital_matrix.middleCols(
            n_inactive_doubly_occupied_orbitals,
            n_active_orbitals);
    const Eigen::Map<const Eigen::MatrixXd> basis_overlap_matrix(
        orbital_preparation_input.active_orbital_overlap_matrix.data(),
        n_basis_functions,
        n_basis_functions);

    constexpr double kRepresentativeChartTolerance = 1.0e-12;
    const double representative_change =
        (repaired_active_physical_orbitals -
         physical_orbital_frame.active_physical_orbital_matrix)
            .cwiseAbs()
            .maxCoeff();
    if (!(representative_change > kRepresentativeChartTolerance)) {
      return false;
    }

    const LocalizedRepresentativeSelector repaired_selector =
        build_localized_representative_selector(
            orbital_result.physical_orbital_frame.inactive_physical_orbital_matrix,
            orbital_result.physical_orbital_frame.inactive_orthonormal_orbital_matrix,
            repaired_active_physical_orbitals,
            orbital_result.auxiliary_orbital_matrix.middleCols(
                n_inactive_doubly_occupied_orbitals,
                n_active_orbitals),
            basis_overlap_matrix);
    transform_sparse_oeo_active_representative_gradient(
        physical_orbital_frame.localized_representative_selector,
        repaired_selector,
        orbital_preparation_input,
        &last_gradient_result_.sparse_orbital_energy_gradient);
    if (!last_gradient_result_.sparse_orbital_reference_energy_gradient.empty()) {
      transform_sparse_oeo_active_representative_gradient(
          physical_orbital_frame.localized_representative_selector,
          repaired_selector,
          orbital_preparation_input,
          &last_gradient_result_.sparse_orbital_reference_energy_gradient);
    }
    transport_packed_secant_history_with_oeo_active_representative_reset(
        physical_orbital_frame.localized_representative_selector,
        repaired_selector,
        orbital_preparation_input,
        parameter_view_,
        packed_secant_history);

    overwrite_sparse_orbitals_from_dense_physical_frame(
        repaired_normalized_orbital_matrix,
        &working_input_.orbital_preparation_input);

    const Eigen::MatrixXd active_overlap_source =
        basis_overlap_matrix * repaired_active_physical_orbitals;
    const Eigen::Map<const Eigen::MatrixXd> inactive_auxiliary_transform(
        orbital_result.inactive_auxiliary_transform.data(),
        n_basis_functions,
        n_basis_functions);
    const Eigen::MatrixXd inactive_active_overlap_matrix =
        inactive_auxiliary_transform.leftCols(n_inactive_doubly_occupied_orbitals)
            .transpose() *
        active_overlap_source;
    orbital_result.inactive_active_overlap_matrix.assign(
        inactive_active_overlap_matrix.data(),
        inactive_active_overlap_matrix.data() +
            inactive_active_overlap_matrix.size());
    orbital_result.physical_orbital_frame.normalized_orbital_matrix =
        repaired_normalized_orbital_matrix;
    orbital_result.physical_orbital_frame.active_physical_orbital_matrix =
        repaired_active_physical_orbitals;
    orbital_result.physical_orbital_frame.localized_representative_selector =
        repaired_selector;

    if (last_gradient_result_.second_order_context != nullptr) {
      auto& cached_orbital_result =
          last_gradient_result_
              .second_order_context
              ->prepared_active_space
              .orbital_result;
      cached_orbital_result.inactive_active_overlap_matrix =
          orbital_result.inactive_active_overlap_matrix;
      cached_orbital_result.physical_orbital_frame.normalized_orbital_matrix =
          repaired_normalized_orbital_matrix;
      cached_orbital_result.physical_orbital_frame.active_physical_orbital_matrix =
          repaired_active_physical_orbitals;
      cached_orbital_result.physical_orbital_frame.localized_representative_selector =
          repaired_selector;
    }

    return true;
  }

  CppVbInput working_input_;
  mutable CppVbInput probe_input_buffer_;
  SparseOrbitalParameterView parameter_view_;
  std::vector<int> selected_state_indices_;
  std::vector<double> state_average_weights_;
  double nuclear_repulsion_energy_ = 0.0;
  const CppOrbitalGradientEvaluator* orbital_gradient_evaluator_ = nullptr;
  const CppVbScfEvaluator* scf_evaluator_ = nullptr;

  CppOrbitalGradientResult last_gradient_result_;
  Eigen::MatrixXd initial_oeo_reference_orbital_matrix_;
  std::vector<double> energy_history_;
  std::vector<double> gradient_inf_norm_history_;
  std::vector<double> iteration_time_history_seconds_;
  double objective_wall_time_seconds_ = 0.0;
  mutable std::size_t energy_only_call_count_ = 0;
  mutable double energy_only_wall_time_seconds_ = 0.0;
  mutable double last_energy_only_wall_time_seconds_ = 0.0;
};

NonredundantOrbitalSpace build_nonredundant_space(
    const OrbitalObjective& objective,
    const SparseOrbitalParameterView& parameter_view) {
  const auto& orbital_preparation_input =
      objective.last_input().orbital_preparation_input;
  const auto& orbital_preparation_result =
      objective.last_gradient_result().orbital_preparation_result;
  const auto& normalized_orbital_matrix =
      orbital_preparation_result.physical_orbital_frame.normalized_orbital_matrix;
  if (normalized_orbital_matrix.size() == 0) {
    throw std::runtime_error(
        "nonredundant space requires the cached physical orbital frame");
  }
  const int n_inactive_doubly_occupied_orbitals =
      (orbital_preparation_input.n_total_electrons -
       orbital_preparation_input.n_active_electrons) / 2;
  const int n_occupied_orbitals =
      n_inactive_doubly_occupied_orbitals +
      orbital_preparation_input.n_active_orbitals;
  return NonredundantOrbitalSpace(
      orbital_preparation_input,
      parameter_view,
      orbital_preparation_result.auxiliary_orbital_matrix.leftCols(n_occupied_orbitals),
      normalized_orbital_matrix,
      &objective.last_gradient_result().ao_effective_one_electron_result.ao_effective_h1e);
}

Eigen::VectorXd build_nonredundant_preconditioned_gradient_direction(
    const NonredundantOrbitalSpace& space,
    const NonredundantOrbitalSpace::ProjectionResult& projection) {
  // The reduced coordinates are either the orthogonalized tangent chart or the
  // direct nonredundant orbital-replacement amplitudes used by the full-AO
  // fast path. Use the full block preconditioner rather than only the
  // curvature diagonal so large reduced sparse blocks keep the accepted-point
  // block metric in the search direction.
  return -space.expand_step(
      space.apply_inverse_reduced_block_preconditioner(
          projection.reduced_gradient));
}

// The optimizer objective lives in packed differentiable sparse coefficients,
// while `expand_retract_input_tangent()` returns the full stored-orbital
// tangent. Gather the actual finite-retraction tangent back to packed
// coordinates whenever a directional derivative or finite-difference scale must
// match `retract_step()` rather than the raw additive chart.
Eigen::VectorXd gather_nonredundant_retract_tangent(
    const OrbitalPreparationInput& orbital_preparation_input,
    const NonredundantOrbitalSpace& space,
    const SparseOrbitalParameterView& parameter_view,
    const Eigen::VectorXd& reduced_step) {
  const Eigen::VectorXd full_tangent =
      space.expand_retract_input_tangent(
          orbital_preparation_input,
          reduced_step);
  Eigen::VectorXd packed_tangent =
      Eigen::VectorXd::Zero(
          static_cast<Eigen::Index>(parameter_view.size()));
  const auto& differentiable_indices =
      parameter_view.differentiable_parameter_indices();
  for (Eigen::Index packed_index = 0;
       packed_index < packed_tangent.size();
       ++packed_index) {
    packed_tangent[packed_index] =
        full_tangent[differentiable_indices[packed_index]];
  }
  return packed_tangent;
}

double compute_nonredundant_retract_tangent_norm(
    const OrbitalPreparationInput& orbital_preparation_input,
    const NonredundantOrbitalSpace& space,
    const SparseOrbitalParameterView& parameter_view,
    const Eigen::VectorXd& reduced_step) {
  if (reduced_step.size() == 0) {
    return 0.0;
  }
  const double tangent_norm =
      gather_nonredundant_retract_tangent(
          orbital_preparation_input,
          space,
          parameter_view,
          reduced_step)
          .norm();
  return std::isfinite(tangent_norm) ? tangent_norm : 0.0;
}

class NonredundantRetractionMetric {
public:
  NonredundantRetractionMetric(
      const OrbitalPreparationInput& orbital_preparation_input,
      const NonredundantOrbitalSpace& space,
      const SparseOrbitalParameterView& parameter_view)
      : orbital_preparation_input_(orbital_preparation_input),
        space_(space),
        parameter_view_(parameter_view) {}

  Eigen::VectorXd tangent(const Eigen::VectorXd& reduced_step) const {
    return gather_nonredundant_retract_tangent(
        orbital_preparation_input_,
        space_,
        parameter_view_,
        reduced_step);
  }

  double norm(const Eigen::VectorXd& reduced_step) const {
    const double tangent_norm = tangent(reduced_step).norm();
    return std::isfinite(tangent_norm) ? tangent_norm : 0.0;
  }

  Eigen::VectorXd clip_to_radius(
      const Eigen::VectorXd& reduced_step,
      double trust_radius) const {
    if (!(trust_radius > 0.0) || !std::isfinite(trust_radius)) {
      return Eigen::VectorXd::Zero(reduced_step.size());
    }
    const double tangent_norm = norm(reduced_step);
    if (!(tangent_norm > 0.0) || !std::isfinite(tangent_norm)) {
      return Eigen::VectorXd::Zero(reduced_step.size());
    }
    if (tangent_norm <= trust_radius) {
      return reduced_step;
    }
    return (trust_radius / tangent_norm) * reduced_step;
  }

private:
  const OrbitalPreparationInput& orbital_preparation_input_;
  const NonredundantOrbitalSpace& space_;
  const SparseOrbitalParameterView& parameter_view_;
};

Eigen::VectorXd clip_nonredundant_reduced_step_to_retract_tangent_radius(
    const OrbitalPreparationInput& orbital_preparation_input,
    const NonredundantOrbitalSpace& space,
    const SparseOrbitalParameterView& parameter_view,
    const Eigen::VectorXd& reduced_step,
    double trust_radius) {
  if (!(trust_radius > 0.0) || !std::isfinite(trust_radius)) {
    return Eigen::VectorXd::Zero(reduced_step.size());
  }
  const double tangent_norm =
      compute_nonredundant_retract_tangent_norm(
          orbital_preparation_input,
          space,
          parameter_view,
          reduced_step);
  if (!(tangent_norm > 0.0) || !std::isfinite(tangent_norm)) {
    return Eigen::VectorXd::Zero(reduced_step.size());
  }
  if (tangent_norm <= trust_radius) {
    return reduced_step;
  }
  return (trust_radius / tangent_norm) * reduced_step;
}

Eigen::VectorXd shrink_nonredundant_reduced_step_inside_retract_tangent_radius(
    const OrbitalPreparationInput& orbital_preparation_input,
    const NonredundantOrbitalSpace& space,
    const SparseOrbitalParameterView& parameter_view,
    const Eigen::VectorXd& reduced_step,
    double trust_radius) {
  if (!(trust_radius > 0.0) || !std::isfinite(trust_radius)) {
    return Eigen::VectorXd::Zero(reduced_step.size());
  }
  constexpr double kInitialStepSafetyFraction = 0.95;
  const double target_radius = kInitialStepSafetyFraction * trust_radius;
  if (!(target_radius > 0.0) || !std::isfinite(target_radius)) {
    return Eigen::VectorXd::Zero(reduced_step.size());
  }
  const double tangent_norm =
      compute_nonredundant_retract_tangent_norm(
          orbital_preparation_input,
          space,
          parameter_view,
          reduced_step);
  if (!(tangent_norm > 0.0) || !std::isfinite(tangent_norm)) {
    return Eigen::VectorXd::Zero(reduced_step.size());
  }
  if (tangent_norm < target_radius) {
    return reduced_step;
  }
  return (target_radius / tangent_norm) * reduced_step;
}

bool optimizer_chart_uses_sparse_orbital_support(
    const OrbitalPreparationInput& orbital_preparation_input) {
  // Legacy `orbtyp` defines the physical orbital manifold. `OEO` means every
  // orbital lives on the full AO chart, while `HAO/BDO` keep a sparse
  // atom/bond-local support. The localized charts are materially more ill-
  // conditioned in reduced TNHVP coordinates, so they need a larger inner
  // Krylov budget than the corresponding full-AO `OEO` cases.
  if (orbital_preparation_input.orbital_type == kLegacyOrbitalTypeOeo) {
    return false;
  }
  if (orbital_preparation_input.orbital_type == kLegacyOrbitalTypeHao ||
      orbital_preparation_input.orbital_type == kLegacyOrbitalTypeBdo) {
    return true;
  }

  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  if (n_basis_functions <= 0) {
    return false;
  }
  for (std::size_t orbital_index = 0;
       orbital_index < orbital_preparation_input.orbital_basis_counts.size();
       ++orbital_index) {
    const int basis_count =
        orbital_preparation_input.orbital_basis_counts[orbital_index];
    if (basis_count > 0 && basis_count < n_basis_functions) {
      return true;
    }
  }
  return false;
}

int choose_nonredundant_truncated_newton_max_cg_iterations(
    const CppVbScfOptimizerOptions& options,
    const OrbitalObjective& objective,
    int reduced_size,
    const ExactCtxInnerSolvePolicy& exact_ctx_inner_solve_policy,
    int consecutive_projected_stall_count) {
  if (options.nonredundant_truncated_newton_max_cg_iterations > 0) {
    return options.nonredundant_truncated_newton_max_cg_iterations;
  }
  const int bounded_reduced_size = std::max(1, reduced_size);

  const double latest_objective_seconds =
      objective.iteration_time_history_seconds().empty()
          ? 0.0
          : objective.iteration_time_history_seconds().back();
  const bool sparse_orbital_chart =
      optimizer_chart_uses_sparse_orbital_support(
          objective.last_input().orbital_preparation_input);
  const ExactCtxSystemProfile system_profile =
      build_exact_ctx_system_profile(
          objective.last_input().orbital_preparation_input);
  const ExactCtxDefaultStrategy strategy =
      choose_exact_ctx_default_strategy(system_profile);
  if (options.nonredundant_truncated_newton_hvp_mode ==
          NonredundantTruncatedNewtonHvpMode::ExactContextDirectAction &&
      exact_ctx_use_tiny_full_fd_fallback(system_profile)) {
    // The tiny exact_ctx fallback uses the full finite-difference operator, so
    // give it the same Krylov budget as explicit full_fd instead of the shorter
    // full-outer-response calibration budget.
    if (latest_objective_seconds <= 2.0e-2) {
      return std::min(bounded_reduced_size, 10);
    }
    if (latest_objective_seconds <= 1.0e-1) {
      return std::min(bounded_reduced_size, 8);
    }
    return std::min(bounded_reduced_size, 6);
  }
  if (!system_profile.open_shell &&
      system_profile.sparse_orbital_chart &&
      system_profile.n_active_orbitals > 0 &&
      system_profile.n_active_orbitals <= 6 &&
      strategy.kind == ExactCtxDefaultStrategyKind::CheapCoreOnly) {
    // 241-class closed-shell sparse charts spend less wall time on a tighter
    // cheap-core Krylov solve than on extra HVPs or same-iteration full retry.
    return std::min(bounded_reduced_size, 8);
  }
  // Cheap objectives can afford a more accurate Newton solve, while expensive
  // relaxed VBSCF evaluations should spend the HVP budget conservatively.
  if (options.nonredundant_truncated_newton_hvp_mode ==
      NonredundantTruncatedNewtonHvpMode::ExactContextDirectAction) {
    if (exact_ctx_inner_solve_policy.use_outer_response) {
      // Full-model accepted-point solves are calibration or correction steps,
      // not the default steady-state path. Keep their Krylov budget
      // conservative because the optimizer should return to the cheaper
      // core-only model once the full operator has served its purpose.
      if (exact_ctx_inner_solve_policy.used_hybrid_followup_full_solve) {
        return std::min(
            bounded_reduced_size,
            exact_ctx_startup_full_inner_solve_base_max_cg_iterations());
      }
      if (exact_ctx_inner_solve_policy.forced_override) {
        return std::min(
            bounded_reduced_size,
            exact_ctx_startup_full_inner_solve_base_max_cg_iterations());
      }
      const int tail_max_cg_iterations =
          std::max(
              1,
              parse_env_int_with_default(
                  "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_TAIL_MAX_CG_ITERATIONS",
                  strategy.startup_full_inner_solve_tail_max_cg_iterations));
      return exact_ctx_inner_solve_policy.used_gradient_tail_extension
          ? tail_max_cg_iterations
          : exact_ctx_startup_full_inner_solve_base_max_cg_iterations();
    }
    // The accepted-point HVP only pays off if the inner solve is accurate
    // enough to avoid repeated trust-region backtracking. Keep the budget
    // moderate, but stop starving exact_ctx after just two or three probes.
    //
    // Sparse HAO/BDO charts are more ill-conditioned than the corresponding
    // full-AO OEO manifold, so the cheap exact-ctx solve needs a slightly
    // larger Krylov budget before paying for any outer-response correction.
    //
    // Scale the budget with reduced dimension: high-dimensional reduced spaces
    // (>500) need a larger Krylov subspace to approximate the Newton direction.
    const int dim_scale = bounded_reduced_size > 800 ? 6 : 0;
    if (latest_objective_seconds <= 2.0e-2) {
      return std::min(
          bounded_reduced_size,
          sparse_orbital_chart ? 14 + dim_scale : 12);
    }
    if (latest_objective_seconds <= 1.0e-1) {
      return std::min(
          bounded_reduced_size,
          sparse_orbital_chart ? 12 + dim_scale : 10);
    }
    const int default_budget =
        std::min(
            bounded_reduced_size,
            sparse_orbital_chart ? 10 + dim_scale : 8);
    if (exact_ctx_stall_tail_boost_enabled(
            consecutive_projected_stall_count)) {
      // When the projected gradient stops contracting, the cheap exact-ctx
      // solve is usually under-solving the accepted-point trust-region model.
      // Spending a couple more H*s here is far cheaper than hundreds of
      // accepted outer iterations with essentially linear tail behavior.
      const int stall_budget =
          sparse_orbital_chart
              ? std::max(default_budget, 14)
              : std::max(
                    default_budget,
                    exact_ctx_stall_tail_boost_max_cg_iterations());
      return std::min(
          bounded_reduced_size,
          stall_budget);
    }
    return default_budget;
  }
  if (latest_objective_seconds <= 2.0e-2) {
    return std::min(bounded_reduced_size, 10);
  }
  if (latest_objective_seconds <= 1.0e-1) {
    return std::min(bounded_reduced_size, 8);
  }
  return std::min(bounded_reduced_size, 6);
}

int choose_nonredundant_truncated_newton_transport_history_size(
    const CppVbScfOptimizerOptions& options,
    const OrbitalObjective& objective) {
  if (options.nonredundant_truncated_newton_transport_history_size <= 0) {
    return 0;
  }

  const double latest_objective_seconds =
      objective.iteration_time_history_seconds().empty()
          ? 0.0
          : objective.iteration_time_history_seconds().back();
  const bool sparse_orbital_chart =
      optimizer_chart_uses_sparse_orbital_support(
          objective.last_input().orbital_preparation_input);
  const int n_active_orbitals =
      objective.last_input().orbital_preparation_input.n_active_orbitals;
  // exact_ctx benefits from transported secant history on localized sparse
  // charts where the reduced tangent basis changes smoothly and the accepted-
  // point HVP still lacks some cheap curvature information. Full-AO OEO charts
  // already use a much denser accepted-point model, and TiCl/FeCl-class small-
  // active open-shell radicals show that replaying packed secant pairs across
  // accepted points can over-steer the late-stage TN search and distort the
  // active orbitals relative to XMVB. Keep history transport off by default for
  // those small full-AO exact_ctx cases, but leave larger FeCl2-class active
  // spaces on the existing path until they show the same failure mode.
  if (options.nonredundant_truncated_newton_hvp_mode ==
      NonredundantTruncatedNewtonHvpMode::ExactContextDirectAction) {
    if (!sparse_orbital_chart && n_active_orbitals <= 7) {
      const auto full_ao_history_override = parse_env_optional_flag(
          "XMVB_CPP_EXACT_CTX_FULL_AO_TRANSPORT_HISTORY");
      if (full_ao_history_override.has_value()) {
        return *full_ao_history_override
            ? options.nonredundant_truncated_newton_transport_history_size
            : 0;
      }
      return 0;
    }
    if (latest_objective_seconds <= 5.0e-3) {
      return 0;
    }
    return options.nonredundant_truncated_newton_transport_history_size;
  }
  if (latest_objective_seconds <= 2.0e-2) {
    return 0;
  }
  return options.nonredundant_truncated_newton_transport_history_size;
}

class TransportedReducedLbfgsPreconditioner {
public:
  explicit TransportedReducedLbfgsPreconditioner(
      const NonredundantOrbitalSpace* space)
      : space_(space) {}

  bool try_add_pair(
      Eigen::VectorXd reduced_step,
      Eigen::VectorXd reduced_gradient_change) {
    const double step_norm = reduced_step.norm();
    const double gradient_change_norm = reduced_gradient_change.norm();
    const double secant_curvature =
        reduced_step.dot(reduced_gradient_change);
    constexpr double kMinimumSecantAlignment = 1.0e-8;
    if (!(step_norm > 0.0) ||
        !(gradient_change_norm > 0.0) ||
        !std::isfinite(step_norm) ||
        !std::isfinite(gradient_change_norm) ||
        !std::isfinite(secant_curvature) ||
        secant_curvature <=
            kMinimumSecantAlignment * step_norm * gradient_change_norm) {
      return false;
    }

    Pair pair;
    pair.reduced_step = std::move(reduced_step);
    pair.reduced_gradient_change = std::move(reduced_gradient_change);
    pair.inverse_curvature = 1.0 / secant_curvature;
    pairs_.push_back(std::move(pair));
    return true;
  }

  bool empty() const noexcept {
    return pairs_.empty();
  }

  int size() const noexcept {
    return static_cast<int>(pairs_.size());
  }

  Eigen::VectorXd apply(const Eigen::VectorXd& reduced_vector) const {
    if (pairs_.empty()) {
      return space_->apply_inverse_reduced_block_preconditioner(reduced_vector);
    }

    Eigen::VectorXd q = reduced_vector;
    std::vector<double> alphas(pairs_.size(), 0.0);
    for (std::size_t pair_index = pairs_.size(); pair_index-- > 0;) {
      const auto& pair = pairs_[pair_index];
      const double alpha =
          pair.inverse_curvature * pair.reduced_step.dot(q);
      if (!std::isfinite(alpha)) {
        return space_->apply_inverse_reduced_block_preconditioner(reduced_vector);
      }
      alphas[pair_index] = alpha;
      q.noalias() -= alpha * pair.reduced_gradient_change;
    }

    Eigen::VectorXd z =
        space_->apply_inverse_reduced_block_preconditioner(q);
    for (std::size_t pair_index = 0;
         pair_index < pairs_.size();
         ++pair_index) {
      const auto& pair = pairs_[pair_index];
      const double beta =
          pair.inverse_curvature * pair.reduced_gradient_change.dot(z);
      if (!std::isfinite(beta)) {
        return space_->apply_inverse_reduced_block_preconditioner(reduced_vector);
      }
      z.noalias() += pair.reduced_step * (alphas[pair_index] - beta);
    }
    return z;
  }

private:
  struct Pair {
    Eigen::VectorXd reduced_step;
    Eigen::VectorXd reduced_gradient_change;
    double inverse_curvature = 0.0;
  };

  const NonredundantOrbitalSpace* space_ = nullptr;
  std::vector<Pair> pairs_;
};

TransportedReducedLbfgsPreconditioner
build_nonredundant_truncated_newton_preconditioner(
    const NonredundantOrbitalSpace& current_space,
    const std::vector<PackedSecantPair>& packed_secant_history,
    int max_history_size) {
  TransportedReducedLbfgsPreconditioner preconditioner(&current_space);
  if (max_history_size <= 0 || packed_secant_history.empty()) {
    return preconditioner;
  }

  const std::size_t history_begin =
      packed_secant_history.size() >
              static_cast<std::size_t>(max_history_size)
          ? packed_secant_history.size() -
                static_cast<std::size_t>(max_history_size)
          : 0;
  for (std::size_t pair_index = history_begin;
       pair_index < packed_secant_history.size();
       ++pair_index) {
    const auto& packed_pair = packed_secant_history[pair_index];
    // The reduced basis changes after every accepted orbital update. Reproject
    // each ambient packed secant pair into the current tangent space so the
    // L-BFGS recursion only uses directions that still survive the latest
    // nonredundant parameterization.
    preconditioner.try_add_pair(
        current_space.project_vector(packed_pair.packed_step).reduced_gradient,
        current_space
            .project_vector(packed_pair.packed_projected_gradient_change)
            .reduced_gradient);
  }
  return preconditioner;
}

Eigen::VectorXd apply_nonredundant_truncated_newton_preconditioner(
    const NonredundantOrbitalSpace& current_space,
    const TransportedReducedLbfgsPreconditioner* transported_preconditioner,
    const Eigen::VectorXd& reduced_vector) {
  if (truncated_newton_disable_curvature_preconditioner()) {
    // Diagnostic escape hatch for exact_ctx regressions: bypass both the
    // reduced-curvature diagonal and transported secant history so TNHVP can
    // be tested as plain PCG on the analytic HVP alone.
    return reduced_vector;
  }
  const bool use_block_preconditioner =
      truncated_newton_use_block_preconditioner(current_space);
  if (transported_preconditioner == nullptr ||
      transported_preconditioner->empty()) {
    return use_block_preconditioner
        ? current_space.apply_inverse_reduced_block_preconditioner(
              reduced_vector)
        : current_space.apply_inverse_reduced_curvature(reduced_vector);
  }

  const Eigen::VectorXd preconditioned =
      transported_preconditioner->apply(reduced_vector);
  const double curvature = reduced_vector.dot(preconditioned);
  if (!std::isfinite(curvature) || curvature <= 0.0) {
    return use_block_preconditioner
        ? current_space.apply_inverse_reduced_block_preconditioner(
              reduced_vector)
        : current_space.apply_inverse_reduced_curvature(reduced_vector);
  }
  return preconditioned;
}

void append_nonredundant_truncated_newton_secant_pair(
    Eigen::VectorXd packed_step,
    Eigen::VectorXd packed_projected_gradient_change,
    int max_history_size,
    std::vector<PackedSecantPair>* packed_secant_history) {
  if (max_history_size <= 0) {
    return;
  }
  const double step_norm = packed_step.norm();
  const double gradient_change_norm = packed_projected_gradient_change.norm();
  const double secant_curvature =
      packed_step.dot(packed_projected_gradient_change);
  constexpr double kMinimumSecantAlignment = 1.0e-10;
  if (!(step_norm > 0.0) ||
      !(gradient_change_norm > 0.0) ||
      !std::isfinite(step_norm) ||
      !std::isfinite(gradient_change_norm) ||
      !std::isfinite(secant_curvature) ||
      secant_curvature <=
          kMinimumSecantAlignment * step_norm * gradient_change_norm) {
    return;
  }

  packed_secant_history->push_back(
      PackedSecantPair{
          std::move(packed_step),
          std::move(packed_projected_gradient_change)});
  if (packed_secant_history->size() >
      static_cast<std::size_t>(max_history_size)) {
    packed_secant_history->erase(packed_secant_history->begin());
  }
}

double solve_trust_region_metric_boundary_tau(
    const Eigen::VectorXd& current_step_tangent,
    const Eigen::VectorXd& search_direction_tangent,
    double trust_radius) {
  if (current_step_tangent.size() != search_direction_tangent.size()) {
    return 0.0;
  }
  const double a = search_direction_tangent.squaredNorm();
  if (!(a > 0.0) || !std::isfinite(a) ||
      !(trust_radius > 0.0) || !std::isfinite(trust_radius)) {
    return 0.0;
  }
  // Solve ||J(s + tau p)||^2 = Delta^2 without materializing G = J^T J.
  const double b = current_step_tangent.dot(search_direction_tangent);
  const double c =
      current_step_tangent.squaredNorm() - trust_radius * trust_radius;
  const double discriminant = std::max(0.0, b * b - a * c);
  const double tau =
      (-b + std::sqrt(discriminant)) / a;
  if (!std::isfinite(tau)) {
    return 0.0;
  }
  return std::max(0.0, tau);
}

class ReducedHvpOperator {
public:
  virtual ~ReducedHvpOperator() = default;
  virtual Eigen::VectorXd apply(const Eigen::VectorXd& reduced_direction) = 0;
};

struct MissingCurvatureSr1OuterResponsePair {
  Eigen::VectorXd packed_step;
  Eigen::VectorXd packed_missing_response;
};

class MissingCurvatureSr1OuterResponseModel {
public:
  explicit MissingCurvatureSr1OuterResponseModel(bool enabled)
      : enabled_(enabled),
        max_history_size_(exact_ctx_sr1_outer_response_history_size()),
        min_alignment_(exact_ctx_sr1_outer_response_min_alignment()),
        initial_scale_(exact_ctx_sr1_outer_response_initial_scale()) {}

  bool enabled() const noexcept {
    return enabled_ && max_history_size_ > 0;
  }

  std::size_t size() const noexcept {
    return pairs_.size();
  }

  void clear() {
    pairs_.clear();
  }

  bool try_append_pair(
      const NonredundantOrbitalSpace& current_space,
      const Eigen::VectorXd& packed_step,
      const Eigen::VectorXd& reduced_step,
      const Eigen::VectorXd& reduced_gradient_change,
      const Eigen::VectorXd& core_hessian_times_step) {
    if (!enabled() ||
        !finite_nonzero_vector_matches_size(
            reduced_step,
            reduced_gradient_change.size()) ||
        !finite_vector_matches_size(
            core_hessian_times_step,
            reduced_gradient_change.size()) ||
        !finite_nonzero_vector_matches_size(
            packed_step,
            packed_step.size())) {
      return false;
    }

    const Eigen::VectorXd missing_response =
        reduced_gradient_change - core_hessian_times_step;
    const double step_norm = reduced_step.norm();
    const double missing_norm = missing_response.norm();
    const double secant_denominator = reduced_step.dot(missing_response);
    if (!(step_norm > 0.0) ||
        !(missing_norm > 0.0) ||
        !std::isfinite(step_norm) ||
        !std::isfinite(missing_norm) ||
        !std::isfinite(secant_denominator) ||
        std::abs(secant_denominator) <
            min_alignment_ * step_norm * missing_norm) {
      return false;
    }

    // The outer-response secant residual is a reduced covector, while the
    // TNHVP history is stored in ambient packed coordinates so it can be
    // reprojected into the next accepted-point chart.  Expanding the reduced
    // residual through the same block-local chart used by the transported
    // preconditioner gives an approximate but cheap accepted-point transport.
    Eigen::VectorXd packed_missing_response =
        current_space.expand_step(missing_response);
    if (!finite_nonzero_vector_matches_size(
            packed_missing_response,
            packed_step.size())) {
      return false;
    }

    pairs_.push_back(
        MissingCurvatureSr1OuterResponsePair{
            packed_step,
            std::move(packed_missing_response)});
    while (pairs_.size() > static_cast<std::size_t>(max_history_size_)) {
      pairs_.erase(pairs_.begin());
    }
    return true;
  }

  Eigen::VectorXd apply(
      const NonredundantOrbitalSpace& current_space,
      const Eigen::VectorXd& reduced_direction) const {
    Eigen::VectorXd response =
        Eigen::VectorXd::Zero(reduced_direction.size());
    if (!enabled() ||
        pairs_.empty() ||
        reduced_direction.size() == 0 ||
        !reduced_direction.allFinite()) {
      return response;
    }

    const double initial_scale =
        std::isfinite(initial_scale_) ? initial_scale_ : 0.0;
    if (initial_scale != 0.0) {
      response.noalias() += initial_scale * reduced_direction;
    }

    std::vector<Eigen::VectorXd> update_vectors;
    std::vector<double> inverse_denominators;
    update_vectors.reserve(pairs_.size());
    inverse_denominators.reserve(pairs_.size());

    for (const auto& pair : pairs_) {
      if (!finite_nonzero_vector_matches_size(
              pair.packed_step,
              pair.packed_missing_response.size()) ||
          pair.packed_step.size() != pair.packed_missing_response.size()) {
        continue;
      }

      const Eigen::VectorXd reduced_step =
          current_space.project_vector(pair.packed_step).reduced_gradient;
      const Eigen::VectorXd missing_response =
          current_space
              .project_vector(pair.packed_missing_response)
              .reduced_gradient;
      if (!finite_nonzero_vector_matches_size(
              reduced_step,
              reduced_direction.size()) ||
          !finite_nonzero_vector_matches_size(
              missing_response,
              reduced_direction.size())) {
        continue;
      }

      Eigen::VectorXd sr1_residual =
          missing_response - initial_scale * reduced_step;
      for (std::size_t update_index = 0;
           update_index < update_vectors.size();
           ++update_index) {
        sr1_residual.noalias() -=
            update_vectors[update_index] *
            (inverse_denominators[update_index] *
             update_vectors[update_index].dot(reduced_step));
      }

      const double residual_norm = sr1_residual.norm();
      const double step_norm = reduced_step.norm();
      const double denominator = sr1_residual.dot(reduced_step);
      if (!(residual_norm > 0.0) ||
          !(step_norm > 0.0) ||
          !std::isfinite(residual_norm) ||
          !std::isfinite(step_norm) ||
          !std::isfinite(denominator) ||
          std::abs(denominator) <
              min_alignment_ * residual_norm * step_norm) {
        continue;
      }

      update_vectors.push_back(std::move(sr1_residual));
      inverse_denominators.push_back(1.0 / denominator);
    }

    for (std::size_t update_index = 0;
         update_index < update_vectors.size();
         ++update_index) {
      response.noalias() +=
          update_vectors[update_index] *
          (inverse_denominators[update_index] *
           update_vectors[update_index].dot(reduced_direction));
    }
    if (!response.allFinite()) {
      return Eigen::VectorXd::Zero(reduced_direction.size());
    }
    return response;
  }

private:
  bool enabled_ = false;
  int max_history_size_ = 0;
  double min_alignment_ = 0.0;
  double initial_scale_ = 0.0;
  std::vector<MissingCurvatureSr1OuterResponsePair> pairs_;
};

class FullFiniteDifferenceReducedHvpOperator final : public ReducedHvpOperator {
public:
  FullFiniteDifferenceReducedHvpOperator(
      const OrbitalObjective& objective,
      const NonredundantOrbitalSpace& current_space,
      const NonredundantOrbitalSpace::ProjectionResult& current_projection,
      const OrbitalPreparationInput& current_orbital_input,
      const SparseOrbitalParameterView& parameter_view,
      const Eigen::VectorXd& current_parameters,
      double hvp_step_size)
      : probe_objective_(objective.make_probe_copy()),
        current_space_(current_space),
        current_reduced_gradient_(current_projection.reduced_gradient),
        current_orbital_input_(current_orbital_input),
        parameter_view_(parameter_view),
        current_parameters_(current_parameters),
        hvp_step_size_(hvp_step_size) {}

  Eigen::VectorXd apply(const Eigen::VectorXd& reduced_direction) override {
    if (reduced_direction.size() == 0) {
      return Eigen::VectorXd::Zero(0);
    }

    const Eigen::VectorXd packed_direction =
        gather_nonredundant_retract_tangent(
            current_orbital_input_,
            current_space_,
            parameter_view_,
            reduced_direction);
    const double packed_direction_norm = packed_direction.norm();
    if (!(packed_direction_norm > 0.0) || !std::isfinite(packed_direction_norm)) {
      return Eigen::VectorXd::Zero(reduced_direction.size());
    }

    const double epsilon =
        hvp_step_size_ / std::max(1.0, packed_direction_norm);
    if (!(epsilon > 0.0) || !std::isfinite(epsilon)) {
      throw std::runtime_error("invalid finite-difference step for reduced HVP");
    }

    const OrbitalPreparationInput trial_orbital_input =
        current_space_.retract_step(
            current_orbital_input_,
            reduced_direction,
            epsilon);
    const Eigen::VectorXd trial_parameters =
        parameter_view_.pack(trial_orbital_input);
    const OrbitalObjective::TrialEvaluation trial_evaluation =
        probe_objective_.evaluate_trial_without_committing(trial_parameters);
    return
        (current_space_.project_reduced_gradient(trial_evaluation.gradient) -
         current_reduced_gradient_) /
        epsilon;
  }

private:
  OrbitalObjective probe_objective_;
  const NonredundantOrbitalSpace& current_space_;
  Eigen::VectorXd current_reduced_gradient_;
  OrbitalPreparationInput current_orbital_input_;
  SparseOrbitalParameterView parameter_view_;
  Eigen::VectorXd current_parameters_;
  double hvp_step_size_ = 0.0;
};

class ExactContextReducedHvpOperator final : public ReducedHvpOperator {
public:
  ExactContextReducedHvpOperator(
      const OrbitalObjective& objective,
      const NonredundantOrbitalSpace& current_space,
      const NonredundantOrbitalSpace::ProjectionResult& current_projection,
      double hvp_step_size,
      bool include_outer_response,
      const MissingCurvatureSr1OuterResponseModel* sr1_outer_response_model =
          nullptr)
      : exact_operator_(
            objective.last_second_order_context(),
            &objective.last_input(),
            SparseOrbitalParameterView(
                objective.last_input().orbital_preparation_input),
            &current_space),
        current_space_(current_space),
        expected_reduced_size_(current_projection.reduced_gradient.size()),
        include_outer_response_(include_outer_response),
        sr1_outer_response_model_(
            include_outer_response ? nullptr : sr1_outer_response_model) {
    (void) hvp_step_size;
  }

  Eigen::VectorXd apply(const Eigen::VectorXd& reduced_direction) override {
    return apply_with_mode(reduced_direction, true);
  }

  Eigen::VectorXd apply_outer_response_only(
      const Eigen::VectorXd& reduced_direction) {
    return apply_with_mode(reduced_direction, false);
  }

  bool supports_analytic_core_model() const noexcept {
    return exact_operator_.supports_analytic_core_model();
  }

  Eigen::VectorXd apply_core_only(
      const Eigen::VectorXd& reduced_direction) {
    return exact_operator_.apply_reduced(
        reduced_direction,
        {.direct_core_response = true,
         .fixed_upstream_pullback = true,
         .outer_response = false});
  }

  ExactOrbitalSecondOrderOperator::Diagnostics diagnostics() const {
    return exact_operator_.diagnostics();
  }

  bool includes_outer_response() const noexcept {
    return include_outer_response_;
  }

  bool uses_sr1_outer_response_approximation() const noexcept {
    return
        !include_outer_response_ &&
        sr1_outer_response_model_ != nullptr &&
        sr1_outer_response_model_->enabled() &&
        sr1_outer_response_model_->size() > 0;
  }

  std::size_t sr1_outer_response_history_size() const noexcept {
    return sr1_outer_response_model_ != nullptr
        ? sr1_outer_response_model_->size()
        : 0;
  }

private:
  Eigen::VectorXd apply_with_mode(
      const Eigen::VectorXd& reduced_direction,
      bool include_core_response) {
    // The accepted-point exact operator is constructed only after the caller
    // has validated the current chart and analytic-core availability.
    Eigen::VectorXd response =
        include_outer_response_
            ? (include_core_response
                   ? exact_operator_.apply_reduced(reduced_direction)
                   : exact_operator_.apply_reduced(
                         reduced_direction,
                         {.direct_core_response = false,
                          .fixed_upstream_pullback = false,
                          .outer_response = true}))
            : exact_operator_.apply_reduced(
                  reduced_direction,
                  {.direct_core_response = true,
                   .fixed_upstream_pullback = true,
                   .outer_response = false});
    if (include_core_response &&
        !include_outer_response_ &&
        sr1_outer_response_model_ != nullptr &&
        sr1_outer_response_model_->enabled() &&
        sr1_outer_response_model_->size() > 0) {
      Eigen::VectorXd correction =
          sr1_outer_response_model_->apply(
              current_space_,
              reduced_direction);
      const double correction_norm = correction.norm();
      if (correction.size() == response.size() &&
          correction.allFinite() &&
          correction_norm > 0.0 &&
          std::isfinite(correction_norm)) {
        const double response_norm = response.norm();
        const double direction_norm = reduced_direction.norm();
        const double correction_scale =
            std::max(response_norm, direction_norm);
        const double max_correction_norm =
            exact_ctx_sr1_outer_response_max_correction_ratio() *
            correction_scale;
        if (max_correction_norm > 0.0 &&
            std::isfinite(max_correction_norm) &&
            correction_norm > max_correction_norm) {
          correction *= max_correction_norm / correction_norm;
        }
        response.noalias() += correction;
      }
    }
    return response;
  }

  ExactOrbitalSecondOrderOperator exact_operator_;
  const NonredundantOrbitalSpace& current_space_;
  Eigen::Index expected_reduced_size_ = 0;
  bool include_outer_response_ = true;
  const MissingCurvatureSr1OuterResponseModel* sr1_outer_response_model_ =
      nullptr;
};

void maybe_log_exact_ctx_hvp_diagnostics(
    int accepted_iteration_index,
    const char* role,
    const ReducedHvpOperator* hvp_operator) {
  if (!exact_ctx_log_hvp_diagnostics() || hvp_operator == nullptr) {
    return;
  }
  const auto* exact_ctx_hvp_operator =
      dynamic_cast<const ExactContextReducedHvpOperator*>(hvp_operator);
  if (exact_ctx_hvp_operator == nullptr) {
    return;
  }

  const auto diagnostics = exact_ctx_hvp_operator->diagnostics();
  if (diagnostics.apply_count == 0 ||
      !std::isfinite(diagnostics.total_apply_wall_time_seconds) ||
      diagnostics.total_apply_wall_time_seconds < 0.0) {
    return;
  }

  std::ostringstream stream;
  stream << "tnhvp_hvp"
         << " iter=" << accepted_iteration_index
         << " role=" << role
         << " outer_included="
         << bool_name(exact_ctx_hvp_operator->includes_outer_response())
         << " sr1_outer_approx="
         << bool_name(
                exact_ctx_hvp_operator
                    ->uses_sr1_outer_response_approximation())
         << " sr1_hist="
         << exact_ctx_hvp_operator->sr1_outer_response_history_size()
         << " outer_runtime="
         << bool_name(diagnostics.outer_response_enabled)
         << " outer_local_only_approx="
         << bool_name(diagnostics.outer_response_local_only_approximation)
         << " outer_energy_only_approx="
         << bool_name(diagnostics.outer_response_energy_only_approximation)
         << " internal_chart_runtime="
         << bool_name(diagnostics.internal_inactive_chart_runtime_enabled)
         << " internal_chart="
         << bool_name(diagnostics.uses_internal_inactive_chart)
         << " apply_count=" << diagnostics.apply_count
         << " avg_apply_s=" << std::fixed << std::setprecision(6)
         << exact_ctx_average_stage_wall_time_seconds(
                diagnostics.total_apply_wall_time_seconds,
                diagnostics.apply_count)
         << " avg_core_setup_s="
         << exact_ctx_average_stage_wall_time_seconds(
                diagnostics.core_setup_wall_time_seconds,
                diagnostics.apply_count)
         << " avg_h1e_build_s="
         << exact_ctx_average_stage_wall_time_seconds(
                diagnostics.ao_effective_one_electron_build_wall_time_seconds,
                diagnostics.apply_count)
         << " avg_h1e_fused_s="
         << exact_ctx_average_stage_wall_time_seconds(
                diagnostics.ao_effective_one_electron_fused_wall_time_seconds,
                diagnostics.apply_count)
         << " avg_active_2e_s="
         << exact_ctx_average_stage_wall_time_seconds(
                diagnostics.active_two_electron_wall_time_seconds,
                diagnostics.apply_count)
         << " avg_h1e_backprop_s="
         << exact_ctx_average_stage_wall_time_seconds(
                diagnostics.ao_effective_one_electron_backprop_wall_time_seconds,
                diagnostics.apply_count)
         << " avg_orb_backprop_s="
         << exact_ctx_average_stage_wall_time_seconds(
                diagnostics.orbital_backprop_wall_time_seconds,
                diagnostics.apply_count)
         << " avg_fixed_upstream_s="
         << exact_ctx_average_stage_wall_time_seconds(
                diagnostics.fixed_upstream_pullback_wall_time_seconds,
                diagnostics.apply_count)
         << " avg_outer_s="
         << exact_ctx_average_stage_wall_time_seconds(
                diagnostics.outer_response_wall_time_seconds,
                diagnostics.apply_count)
         << '\n';
  std::cerr << stream.str();
  std::cerr.flush();
}

std::string build_exact_ctx_unavailable_message(
    const ExactContextReducedHvpOperator& hvp_operator) {
  const auto info = hvp_operator.diagnostics();
  std::ostringstream message;
  message << "exact_ctx HVP is unavailable"
          << ": supports_analytic_core_model="
          << bool_name(info.supports_analytic_core_model)
          << " outer_response_enabled="
          << bool_name(info.outer_response_enabled)
          << " has_same_spin_matrix_form="
          << bool_name(info.has_same_spin_matrix_form)
          << " has_opposite_spin_matrix_form="
          << bool_name(info.has_opposite_spin_matrix_form)
          << " n_selected_states=" << info.n_selected_states
          << " n_active_orbitals=" << info.n_active_orbitals
          << " n_blocks=" << info.n_blocks;
  return message.str();
}

void maybe_update_exact_ctx_hybrid_strategy_state_from_hvp_operator(
    const ReducedHvpOperator* hvp_operator,
    ExactCtxHybridStrategyState* hybrid_strategy_state) {
  if (hvp_operator == nullptr || hybrid_strategy_state == nullptr) {
    return;
  }
  const auto* exact_ctx_hvp_operator =
      dynamic_cast<const ExactContextReducedHvpOperator*>(hvp_operator);
  if (exact_ctx_hvp_operator == nullptr ||
      !exact_ctx_hvp_operator->includes_outer_response()) {
    return;
  }

  const auto diagnostics = exact_ctx_hvp_operator->diagnostics();
  if (!diagnostics.outer_response_enabled ||
      diagnostics.apply_count == 0 ||
      !std::isfinite(diagnostics.total_apply_wall_time_seconds) ||
      !(diagnostics.total_apply_wall_time_seconds > 0.0) ||
      !std::isfinite(diagnostics.outer_response_wall_time_seconds) ||
      diagnostics.outer_response_wall_time_seconds < 0.0) {
    return;
  }

  hybrid_strategy_state->has_full_operator_cost_sample = true;
  hybrid_strategy_state->last_outer_response_cost_fraction =
      std::clamp(
          diagnostics.outer_response_wall_time_seconds /
              diagnostics.total_apply_wall_time_seconds,
          0.0,
          1.0);
  hybrid_strategy_state->last_outer_response_average_wall_time_seconds =
      diagnostics.outer_response_wall_time_seconds /
      static_cast<double>(diagnostics.apply_count);
}

struct TruncatedNewtonKrylovSubspace {
  Eigen::MatrixXd orthonormal_basis;
  Eigen::MatrixXd tangent_basis;
  Eigen::MatrixXd hessian_basis;
  Eigen::MatrixXd reduced_hessian;
  Eigen::VectorXd projected_gradient;
};

struct TruncatedNewtonStepResult {
  Eigen::VectorXd reduced_step;
  Eigen::VectorXd reduced_hessian_times_step;
  TruncatedNewtonKrylovSubspace krylov_subspace;
  double retract_tangent_norm = 0.0;
  bool reached_boundary = false;
  bool encountered_negative_curvature = false;
  bool used_initial_step = false;
  bool warm_start_hvp_performed = false;
  bool used_krylov_rescue = false;
  int cg_iterations = 0;
  double predicted_decrease = 0.0;
};

double truncated_newton_step_effective_norm(
    const TruncatedNewtonStepResult& step) {
  if (std::isfinite(step.retract_tangent_norm) &&
      step.retract_tangent_norm > 0.0) {
    return step.retract_tangent_norm;
  }
  const double reduced_norm = step.reduced_step.norm();
  return std::isfinite(reduced_norm) ? reduced_norm : 0.0;
}

void clamp_nonredundant_step_result_to_retract_tangent_radius(
    const OrbitalPreparationInput& orbital_preparation_input,
    const NonredundantOrbitalSpace& current_space,
    const SparseOrbitalParameterView& parameter_view,
    const NonredundantOrbitalSpace::ProjectionResult& current_projection,
    double trust_radius,
    TruncatedNewtonStepResult* step) {
  if (step == nullptr ||
      step->reduced_step.size() != current_projection.reduced_gradient.size() ||
      step->reduced_step.size() == 0 ||
      !step->reduced_step.allFinite()) {
    return;
  }

  const double tangent_norm =
      compute_nonredundant_retract_tangent_norm(
          orbital_preparation_input,
          current_space,
          parameter_view,
          step->reduced_step);
  step->retract_tangent_norm = tangent_norm;
  if (!(trust_radius > 0.0) ||
      !std::isfinite(trust_radius) ||
      !(tangent_norm > 0.0) ||
      !std::isfinite(tangent_norm)) {
    return;
  }

  if (tangent_norm > trust_radius) {
    const double scale = trust_radius / tangent_norm;
    step->reduced_step *= scale;
    if (step->reduced_hessian_times_step.size() ==
            current_projection.reduced_gradient.size() &&
        step->reduced_hessian_times_step.allFinite()) {
      step->reduced_hessian_times_step *= scale;
      step->predicted_decrease =
          -current_projection.reduced_gradient.dot(step->reduced_step) -
          0.5 * step->reduced_step.dot(step->reduced_hessian_times_step);
    } else {
      step->predicted_decrease = 0.0;
    }
    step->retract_tangent_norm = trust_radius;
    step->reached_boundary = true;
    return;
  }

  step->reached_boundary =
      step->reached_boundary ||
      tangent_norm >= (1.0 - 1.0e-8) * trust_radius;
}

struct ExactCtxHybridRefinementResult {
  TruncatedNewtonStepResult step;
  bool used_full_model_probe = false;
  bool used_refinement = false;
  double full_step_residual_ratio = 0.0;
  double full_model_gap_ratio = 0.0;
};

struct TruncatedNewtonTrialEvaluation {
  double actual_decrease = 0.0;
  double trust_ratio = -std::numeric_limits<double>::infinity();
};

struct FullRetryWarmStart {
  bool used_cached_rejected_step = false;
  bool used_cheap_step_warm_start = false;
  Eigen::VectorXd initial_reduced_step;
  Eigen::VectorXd initial_hessian_times_step;

  const Eigen::VectorXd* initial_reduced_step_ptr(
      Eigen::Index expected_size) const {
    return finite_nonzero_vector_matches_size(
               initial_reduced_step,
               expected_size)
        ? &initial_reduced_step
        : nullptr;
  }

  const Eigen::VectorXd* initial_hessian_times_step_ptr(
      Eigen::Index expected_size) const {
    return finite_vector_matches_size(
               initial_hessian_times_step,
               expected_size)
        ? &initial_hessian_times_step
        : nullptr;
  }
};

struct RejectedTruncatedNewtonStepCache {
  Eigen::VectorXd cheap_step;
  Eigen::VectorXd full_step;

  bool has_cheap_step(Eigen::Index expected_size) const {
    return finite_nonzero_vector_matches_size(cheap_step, expected_size);
  }

  bool has_full_step(Eigen::Index expected_size) const {
    return finite_nonzero_vector_matches_size(full_step, expected_size);
  }

  void clear() {
    cheap_step.resize(0);
    full_step.resize(0);
  }

  void update(
      const OrbitalPreparationInput& orbital_preparation_input,
      const NonredundantOrbitalSpace& current_space,
      const SparseOrbitalParameterView& parameter_view,
      const TruncatedNewtonStepResult& cheap_model_step,
      const TruncatedNewtonStepResult& current_trial_step,
      Eigen::Index expected_size,
      double trust_radius) {
    cheap_step =
        finite_nonzero_vector_matches_size(
                cheap_model_step.reduced_step,
                expected_size)
            ? shrink_nonredundant_reduced_step_inside_retract_tangent_radius(
                  orbital_preparation_input,
                  current_space,
                  parameter_view,
                  cheap_model_step.reduced_step,
                  trust_radius)
            : Eigen::VectorXd();
    full_step =
        finite_nonzero_vector_matches_size(
                current_trial_step.reduced_step,
                expected_size)
            ? shrink_nonredundant_reduced_step_inside_retract_tangent_radius(
                  orbital_preparation_input,
                  current_space,
                  parameter_view,
                  current_trial_step.reduced_step,
                  trust_radius)
            : Eigen::VectorXd();
  }
};

struct AcceptedTruncatedNewtonStepControl {
  double trust_radius = 0.0;
  bool stalled_projected_convergence = false;
  int consecutive_projected_stall_count = 0;
  int hybrid_followup_cooldown_remaining = 0;
  bool previous_iteration_reliable_for_transport = false;
  bool enable_hybrid_bridge_next_iteration = false;
};

bool truncated_newton_krylov_subspace_is_usable(
    const TruncatedNewtonKrylovSubspace& krylov_subspace,
    Eigen::Index reduced_size) {
  return
      reduced_size >= 0 &&
      krylov_subspace.orthonormal_basis.rows() == reduced_size &&
      krylov_subspace.orthonormal_basis.cols() > 0 &&
      krylov_subspace.orthonormal_basis.allFinite() &&
      krylov_subspace.tangent_basis.cols() ==
          krylov_subspace.orthonormal_basis.cols() &&
      krylov_subspace.tangent_basis.allFinite() &&
      krylov_subspace.hessian_basis.rows() == reduced_size &&
      krylov_subspace.hessian_basis.cols() ==
          krylov_subspace.orthonormal_basis.cols() &&
      krylov_subspace.hessian_basis.allFinite() &&
      krylov_subspace.reduced_hessian.rows() ==
          krylov_subspace.orthonormal_basis.cols() &&
      krylov_subspace.reduced_hessian.cols() ==
          krylov_subspace.orthonormal_basis.cols() &&
      krylov_subspace.reduced_hessian.allFinite() &&
      krylov_subspace.projected_gradient.size() ==
          krylov_subspace.orthonormal_basis.cols() &&
      krylov_subspace.projected_gradient.allFinite();
}

bool truncated_newton_step_is_usable(
    const TruncatedNewtonStepResult& step,
    const Eigen::VectorXd& reduced_gradient) {
  return
      step.reduced_step.size() == reduced_gradient.size() &&
      step.reduced_step.allFinite() &&
      step.reduced_step.squaredNorm() > 0.0 &&
      std::isfinite(step.predicted_decrease) &&
      step.predicted_decrease > 0.0 &&
      reduced_gradient.dot(step.reduced_step) < 0.0;
}

bool append_truncated_newton_krylov_basis_vector(
    const NonredundantRetractionMetric& retraction_metric,
    const Eigen::VectorXd& candidate_vector,
    const Eigen::VectorXd& hessian_times_candidate,
    std::vector<Eigen::VectorXd>* basis_vectors,
    std::vector<Eigen::VectorXd>* tangent_basis_vectors,
    std::vector<Eigen::VectorXd>* hessian_basis_vectors) {
  if (basis_vectors == nullptr ||
      tangent_basis_vectors == nullptr ||
      hessian_basis_vectors == nullptr ||
      candidate_vector.size() != hessian_times_candidate.size() ||
      candidate_vector.size() == 0 ||
      !candidate_vector.allFinite() ||
      !hessian_times_candidate.allFinite()) {
    return false;
  }

  Eigen::VectorXd orthogonal_vector = candidate_vector;
  Eigen::VectorXd orthogonal_hessian_vector = hessian_times_candidate;
  Eigen::VectorXd orthogonal_tangent =
      retraction_metric.tangent(candidate_vector);
  const double candidate_norm = orthogonal_tangent.norm();
  if (!(candidate_norm > 0.0) || !std::isfinite(candidate_norm)) {
    return false;
  }

  // Orthonormalize in the accepted-point retraction metric
  //   <u,v>_G = (J u)^T (J v).
  // The reduced basis columns are G-orthonormal, while the cached tangent
  // columns keep reorthogonalization and cached Ritz solves matrix-free.
  for (int orthogonalization_pass = 0;
       orthogonalization_pass < 2;
       ++orthogonalization_pass) {
    for (std::size_t basis_index = 0;
         basis_index < basis_vectors->size();
         ++basis_index) {
      const double coefficient =
          (*tangent_basis_vectors)[basis_index].dot(orthogonal_tangent);
      if (!std::isfinite(coefficient)) {
        return false;
      }
      orthogonal_vector.noalias() -=
          coefficient * (*basis_vectors)[basis_index];
      orthogonal_tangent.noalias() -=
          coefficient * (*tangent_basis_vectors)[basis_index];
      orthogonal_hessian_vector.noalias() -=
          coefficient * (*hessian_basis_vectors)[basis_index];
    }
  }

  constexpr double kLinearDependenceTolerance = 1.0e-10;
  const double orthogonal_norm = orthogonal_tangent.norm();
  if (!(orthogonal_norm >
        kLinearDependenceTolerance * std::max(1.0, candidate_norm)) ||
      !std::isfinite(orthogonal_norm)) {
    return false;
  }

  basis_vectors->push_back(orthogonal_vector / orthogonal_norm);
  tangent_basis_vectors->push_back(orthogonal_tangent / orthogonal_norm);
  hessian_basis_vectors->push_back(orthogonal_hessian_vector / orthogonal_norm);
  return true;
}

TruncatedNewtonKrylovSubspace build_truncated_newton_krylov_subspace(
    const Eigen::VectorXd& reduced_gradient,
    const std::vector<Eigen::VectorXd>& basis_vectors,
    const std::vector<Eigen::VectorXd>& tangent_basis_vectors,
    const std::vector<Eigen::VectorXd>& hessian_basis_vectors) {
  TruncatedNewtonKrylovSubspace krylov_subspace;
  if (basis_vectors.empty() ||
      basis_vectors.size() != tangent_basis_vectors.size() ||
      basis_vectors.size() != hessian_basis_vectors.size()) {
    return krylov_subspace;
  }

  const Eigen::Index reduced_size = reduced_gradient.size();
  const Eigen::Index basis_size =
      static_cast<Eigen::Index>(basis_vectors.size());
  krylov_subspace.orthonormal_basis.resize(reduced_size, basis_size);
  krylov_subspace.tangent_basis.resize(
      tangent_basis_vectors.front().size(),
      basis_size);
  krylov_subspace.hessian_basis.resize(reduced_size, basis_size);
  krylov_subspace.reduced_hessian.resize(basis_size, basis_size);
  krylov_subspace.projected_gradient.resize(basis_size);

  for (Eigen::Index column = 0; column < basis_size; ++column) {
    krylov_subspace.orthonormal_basis.col(column) =
        basis_vectors[column];
    krylov_subspace.tangent_basis.col(column) =
        tangent_basis_vectors[column];
    krylov_subspace.hessian_basis.col(column) =
        hessian_basis_vectors[column];
    krylov_subspace.projected_gradient[column] =
        basis_vectors[column].dot(reduced_gradient);
  }

  for (Eigen::Index row = 0; row < basis_size; ++row) {
    for (Eigen::Index column = 0; column < basis_size; ++column) {
      krylov_subspace.reduced_hessian(row, column) =
          basis_vectors[row].dot(
              hessian_basis_vectors[column]);
    }
  }
  krylov_subspace.reduced_hessian =
      0.5 *
      (krylov_subspace.reduced_hessian +
       krylov_subspace.reduced_hessian.transpose());
  if (!truncated_newton_krylov_subspace_is_usable(
          krylov_subspace,
          reduced_size)) {
    return TruncatedNewtonKrylovSubspace();
  }
  return krylov_subspace;
}

TruncatedNewtonStepResult solve_trust_region_in_krylov_subspace(
    const NonredundantOrbitalSpace::ProjectionResult& current_projection,
    double trust_radius,
    const TruncatedNewtonKrylovSubspace& krylov_subspace) {
  TruncatedNewtonStepResult result;
  result.reduced_step =
      Eigen::VectorXd::Zero(current_projection.reduced_gradient.size());
  result.reduced_hessian_times_step =
      Eigen::VectorXd::Zero(current_projection.reduced_gradient.size());
  if (!(trust_radius > 0.0) ||
      !std::isfinite(trust_radius) ||
      !truncated_newton_krylov_subspace_is_usable(
          krylov_subspace,
          current_projection.reduced_gradient.size())) {
    return result;
  }

  const Eigen::MatrixXd reduced_hessian =
      0.5 *
      (krylov_subspace.reduced_hessian +
       krylov_subspace.reduced_hessian.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(reduced_hessian);
  if (eigensolver.info() != Eigen::Success) {
    return result;
  }

  const Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
  const Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();
  if (eigenvalues.size() == 0 ||
      !eigenvalues.allFinite() ||
      !eigenvectors.allFinite()) {
    return result;
  }
  const Eigen::VectorXd projected_gradient_in_eigenbasis =
      eigenvectors.transpose() * krylov_subspace.projected_gradient;
  if (!projected_gradient_in_eigenbasis.allFinite()) {
    return result;
  }

  const double radius_squared = trust_radius * trust_radius;
  const double spectral_scale =
      std::max(1.0, reduced_hessian.cwiseAbs().maxCoeff());
  constexpr double kShiftToleranceFactor =
      64.0 * std::numeric_limits<double>::epsilon();
  constexpr double kRelativeRadiusTolerance = 1.0e-10;
  auto solve_shifted_subspace_system =
      [&](double lambda,
          Eigen::VectorXd* eigen_coordinates,
          double* squared_norm) -> bool {
        eigen_coordinates->resize(eigenvalues.size());
        *squared_norm = 0.0;
        for (Eigen::Index index = 0; index < eigenvalues.size(); ++index) {
          const double denominator = eigenvalues[index] + lambda;
          const double denominator_floor =
              kShiftToleranceFactor *
              std::max(1.0, std::abs(eigenvalues[index]) + std::abs(lambda));
          if (!(denominator > denominator_floor) ||
              !std::isfinite(denominator)) {
            return false;
          }
          (*eigen_coordinates)[index] =
              -projected_gradient_in_eigenbasis[index] / denominator;
          *squared_norm +=
              (*eigen_coordinates)[index] * (*eigen_coordinates)[index];
        }
        return std::isfinite(*squared_norm) &&
            eigen_coordinates->allFinite();
      };

  Eigen::Index minimum_eigenvalue_index = 0;
  const double minimum_eigenvalue =
      eigenvalues.minCoeff(&minimum_eigenvalue_index);
  Eigen::VectorXd eigen_coordinates;
  double coordinate_squared_norm = 0.0;
  bool solved_subproblem = false;
  if (minimum_eigenvalue > kShiftToleranceFactor * spectral_scale &&
      solve_shifted_subspace_system(
          0.0,
          &eigen_coordinates,
          &coordinate_squared_norm) &&
      coordinate_squared_norm <=
          radius_squared * (1.0 + kRelativeRadiusTolerance)) {
    solved_subproblem = true;
  } else {
    double lower_shift = std::max(0.0, -minimum_eigenvalue);
    if (lower_shift > 0.0 || minimum_eigenvalue <= 0.0) {
      lower_shift +=
          kShiftToleranceFactor *
          std::max(1.0, std::abs(minimum_eigenvalue));
    }
    if (!solve_shifted_subspace_system(
            lower_shift,
            &eigen_coordinates,
            &coordinate_squared_norm)) {
      return result;
    }

    if (coordinate_squared_norm <=
        radius_squared * (1.0 + kRelativeRadiusTolerance)) {
      solved_subproblem = true;
      if (minimum_eigenvalue < 0.0) {
        const double remaining_squared_radius =
            radius_squared - coordinate_squared_norm;
        if (remaining_squared_radius >
            radius_squared * kRelativeRadiusTolerance) {
          const double augmentation_sign =
              projected_gradient_in_eigenbasis[minimum_eigenvalue_index] > 0.0
                  ? -1.0
                  : 1.0;
          eigen_coordinates[minimum_eigenvalue_index] +=
              augmentation_sign * std::sqrt(remaining_squared_radius);
        }
      }
    } else {
      double upper_shift =
          std::max(1.0, std::max(2.0 * lower_shift, lower_shift + 1.0));
      Eigen::VectorXd upper_coordinates;
      double upper_squared_norm = 0.0;
      bool bracketed = false;
      for (int expansion_iteration = 0;
           expansion_iteration < 64;
           ++expansion_iteration) {
        if (!solve_shifted_subspace_system(
                upper_shift,
                &upper_coordinates,
                &upper_squared_norm)) {
          return result;
        }
        if (upper_squared_norm <= radius_squared) {
          bracketed = true;
          break;
        }
        upper_shift = std::max(2.0 * upper_shift, upper_shift + 1.0);
      }
      if (!bracketed) {
        return result;
      }

      double bisection_lower_shift = lower_shift;
      double bisection_upper_shift = upper_shift;
      eigen_coordinates = upper_coordinates;
      coordinate_squared_norm = upper_squared_norm;
      for (int bisection_iteration = 0;
           bisection_iteration < 64;
           ++bisection_iteration) {
        const double mid_shift =
            0.5 * (bisection_lower_shift + bisection_upper_shift);
        Eigen::VectorXd mid_coordinates;
        double mid_squared_norm = 0.0;
        if (!solve_shifted_subspace_system(
                mid_shift,
                &mid_coordinates,
                &mid_squared_norm)) {
          return result;
        }
        if (mid_squared_norm > radius_squared) {
          bisection_lower_shift = mid_shift;
        } else {
          bisection_upper_shift = mid_shift;
          eigen_coordinates = std::move(mid_coordinates);
          coordinate_squared_norm = mid_squared_norm;
        }
      }
      solved_subproblem = true;
    }
  }

  if (!solved_subproblem || !eigen_coordinates.allFinite()) {
    return result;
  }

  Eigen::VectorXd subspace_coordinates =
      eigenvectors * eigen_coordinates;
  Eigen::VectorXd reduced_step =
      krylov_subspace.orthonormal_basis * subspace_coordinates;
  double step_metric_norm = subspace_coordinates.norm();
  if (!(step_metric_norm > 0.0) || !std::isfinite(step_metric_norm)) {
    return result;
  }
  if (step_metric_norm >
      trust_radius * (1.0 + 1.0e-8)) {
    const double scale = trust_radius / step_metric_norm;
    subspace_coordinates *= scale;
    reduced_step *= scale;
    step_metric_norm = subspace_coordinates.norm();
  }
  const Eigen::VectorXd reduced_hessian_times_step =
      krylov_subspace.hessian_basis * subspace_coordinates;
  if (reduced_hessian_times_step.size() !=
          current_projection.reduced_gradient.size() ||
      !reduced_hessian_times_step.allFinite()) {
    return result;
  }

  const Eigen::VectorXd reduced_model_hessian_step =
      reduced_hessian * subspace_coordinates;
  const double predicted_decrease =
      -krylov_subspace.projected_gradient.dot(subspace_coordinates) -
      0.5 * subspace_coordinates.dot(reduced_model_hessian_step);
  if (!std::isfinite(predicted_decrease) ||
      predicted_decrease <= 0.0 ||
      current_projection.reduced_gradient.dot(reduced_step) >= 0.0) {
    return result;
  }

  result.reduced_step = std::move(reduced_step);
  result.reduced_hessian_times_step = reduced_hessian_times_step;
  result.krylov_subspace = krylov_subspace;
  result.retract_tangent_norm = step_metric_norm;
  result.reached_boundary =
      step_metric_norm >= (1.0 - 1.0e-8) * trust_radius;
  result.encountered_negative_curvature =
      minimum_eigenvalue <= -kShiftToleranceFactor * spectral_scale;
  result.predicted_decrease = predicted_decrease;
  return result;
}

Eigen::VectorXd build_nonredundant_preconditioned_reduced_gradient_step(
    const NonredundantRetractionMetric& retraction_metric,
    const NonredundantOrbitalSpace& space,
    const NonredundantOrbitalSpace::ProjectionResult& projection,
    double trust_radius,
    const TransportedReducedLbfgsPreconditioner* transported_preconditioner) {
  const Eigen::VectorXd reduced_preconditioned_gradient =
      apply_nonredundant_truncated_newton_preconditioner(
          space,
          transported_preconditioner,
          projection.reduced_gradient);
  return retraction_metric.clip_to_radius(
      -reduced_preconditioned_gradient,
      trust_radius);
}

bool
assess_nonredundant_truncated_newton_transported_initial_step(
    const NonredundantRetractionMetric& retraction_metric,
    const NonredundantOrbitalSpace& current_space,
    const NonredundantOrbitalSpace::ProjectionResult& current_projection,
    double trust_radius,
    const Eigen::VectorXd& transported_initial_reduced_step,
    bool previous_iteration_reliable) {
  if (!previous_iteration_reliable) {
    return false;
  }
  if (transported_initial_reduced_step.size() !=
          current_projection.reduced_gradient.size() ||
      transported_initial_reduced_step.size() == 0 ||
      !transported_initial_reduced_step.allFinite()) {
    return false;
  }

  const double step_norm = transported_initial_reduced_step.norm();
  const double gradient_norm = current_projection.reduced_gradient.norm();
  const double gradient_inf_norm =
      gradient_infinity_norm(current_projection.reduced_gradient);
  if (!(step_norm > 0.0) ||
      !(gradient_norm > 0.0) ||
      !(gradient_inf_norm > 0.0) ||
      !std::isfinite(step_norm) ||
      !std::isfinite(gradient_norm) ||
      !std::isfinite(gradient_inf_norm)) {
    return false;
  }

  const double descent_measure =
      -current_projection.reduced_gradient.dot(
          transported_initial_reduced_step);
  const double directional_cosine =
      descent_measure / std::max(
          std::numeric_limits<double>::min(),
          gradient_norm * step_norm);
  if (!std::isfinite(directional_cosine) ||
      directional_cosine <= 0.0) {
    return false;
  }

  const Eigen::VectorXd transported_curvature =
      current_space.apply_reduced_curvature(
          transported_initial_reduced_step);
  if (transported_curvature.size() !=
          current_projection.reduced_gradient.size() ||
      !transported_curvature.allFinite()) {
    return false;
  }

  const double transported_predicted_decrease =
      descent_measure -
      0.5 * transported_initial_reduced_step.dot(transported_curvature);
  const Eigen::VectorXd transported_surrogate_residual =
      transported_curvature + current_projection.reduced_gradient;
  const double surrogate_residual_ratio =
      gradient_infinity_norm(transported_surrogate_residual) /
      std::max(gradient_inf_norm, std::numeric_limits<double>::min());
  if (!std::isfinite(surrogate_residual_ratio) ||
      !std::isfinite(transported_predicted_decrease) ||
      transported_predicted_decrease <= 0.0) {
    return false;
  }

  const Eigen::VectorXd diagonal_reference_step =
      build_nonredundant_preconditioned_reduced_gradient_step(
          retraction_metric,
          current_space,
          current_projection,
          trust_radius,
          nullptr);
  const Eigen::VectorXd diagonal_reference_curvature =
      current_space.apply_reduced_curvature(diagonal_reference_step);
  const double diagonal_reference_predicted_decrease =
      -current_projection.reduced_gradient.dot(diagonal_reference_step) -
      0.5 * diagonal_reference_step.dot(diagonal_reference_curvature);
  double surrogate_predicted_decrease_ratio = 0.0;
  if (std::isfinite(diagonal_reference_predicted_decrease) &&
      diagonal_reference_predicted_decrease > 0.0) {
    surrogate_predicted_decrease_ratio =
        transported_predicted_decrease /
        diagonal_reference_predicted_decrease;
  }

  // A transported step is only worth paying one extra H*s for if it still
  // looks Newton-like in the current accepted-point diagonal model. Otherwise
  // the warm start tends to poison the cheap solve and trigger the expensive
  // same-iteration full retry path.
  constexpr double kMinimumDirectionalCosine = 5.0e-2;
  constexpr double kMaximumResidualRatio = 5.0e-1;
  constexpr double kMinimumPredictedDecreaseRatio = 5.0e-1;
  return
      directional_cosine >= kMinimumDirectionalCosine &&
      surrogate_residual_ratio <= kMaximumResidualRatio &&
      surrogate_predicted_decrease_ratio >=
          kMinimumPredictedDecreaseRatio;
}

bool
assess_nonredundant_truncated_newton_full_retry_after_cheap_reject(
    const TruncatedNewtonStepResult& cheap_step,
    bool cheap_predicted_decrease_fallback,
    const TruncatedNewtonTrialEvaluation& cheap_trial_evaluation,
    bool model_quality_retry_requested) {
  const bool has_positive_actual_decrease =
      std::isfinite(cheap_trial_evaluation.actual_decrease) &&
      cheap_trial_evaluation.actual_decrease > 0.0;
  const bool has_positive_trust_ratio =
      std::isfinite(cheap_trial_evaluation.trust_ratio) &&
      cheap_trial_evaluation.trust_ratio > 0.0;

  // Same-iteration full retry is worthwhile when the cheap step already moves
  // downhill in the full objective and the rejection mainly looks like a
  // trust-ratio calibration issue. Boundary steps are deliberately allowed:
  // sparse open-shell charts often produce boundary-limited cheap directions
  // whose full relaxed-response model fixes the radius calibration and avoids
  // many tiny follow-up steps. Fallback, negative curvature, and Krylov rescue
  // still indicate a genuinely unreliable cheap subproblem.
  const bool cheap_step_has_model_descent =
      cheap_step.reduced_step.size() > 0 &&
      cheap_step.reduced_step.allFinite() &&
      std::isfinite(cheap_step.predicted_decrease) &&
      cheap_step.predicted_decrease > 0.0;
  return
      !cheap_predicted_decrease_fallback &&
      !cheap_step.encountered_negative_curvature &&
      !cheap_step.used_krylov_rescue &&
      cheap_step_has_model_descent &&
      ((has_positive_actual_decrease && has_positive_trust_ratio) ||
       model_quality_retry_requested);
}

bool
assess_exact_ctx_hybrid_bridge_for_next_iteration(
    bool full_operator_available,
    bool sparse_orbital_chart,
    bool projected_stall,
    int consecutive_projected_stall_count,
    int hybrid_followup_cooldown_remaining,
    double trust_ratio,
    bool used_krylov_rescue_step) {
  // Sparse HAO/BDO charts frequently show single-step projected-gradient
  // plateaus, and dense OEO radicals can otherwise request the same expensive
  // outer-response probe on every tail iteration. Require either sustained
  // stall or an outright Krylov rescue, then honor a chart-specific cooldown
  // before spending the next full-model correction.
  const bool due_to_projected_stall =
      projected_stall &&
      (!sparse_orbital_chart || consecutive_projected_stall_count >= 2);
  const bool due_to_low_trust_ratio =
      !sparse_orbital_chart &&
      std::isfinite(trust_ratio) &&
      trust_ratio < exact_ctx_hybrid_bridge_low_trust_ratio_threshold();
  const bool due_to_krylov_rescue = used_krylov_rescue_step;
  const bool followup_is_cooling_down =
      hybrid_followup_cooldown_remaining > 0;
  return
      full_operator_available &&
      (!followup_is_cooling_down || due_to_krylov_rescue) &&
      (due_to_projected_stall ||
       due_to_low_trust_ratio ||
       due_to_krylov_rescue);
}

FullRetryWarmStart
build_nonredundant_truncated_newton_full_retry_warm_start(
    const NonredundantOrbitalSpace::ProjectionResult& current_projection,
    double trust_radius,
    const RejectedTruncatedNewtonStepCache& rejected_step_cache,
    const TruncatedNewtonStepResult& cheap_step,
    ReducedHvpOperator* full_hvp_operator) {
  FullRetryWarmStart warm_start;
  const Eigen::Index reduced_size =
      current_projection.reduced_gradient.size();
  warm_start.used_cached_rejected_step =
      rejected_step_cache.has_full_step(reduced_size);
  if (warm_start.used_cached_rejected_step) {
    warm_start.initial_reduced_step = rejected_step_cache.full_step;
    return warm_start;
  }

  const Eigen::VectorXd& cheap_reduced_step = cheap_step.reduced_step;
  const bool can_warm_start =
      finite_nonzero_vector_matches_size(
          cheap_reduced_step,
          reduced_size) &&
      !cheap_step.reached_boundary;
  if (!can_warm_start) {
    return warm_start;
  }

  warm_start.used_cheap_step_warm_start = true;
  warm_start.initial_reduced_step = cheap_reduced_step;
  {
    auto* exact_ctx_full_hvp_operator =
        static_cast<ExactContextReducedHvpOperator*>(full_hvp_operator);
    assert(exact_ctx_full_hvp_operator != nullptr);
    if (finite_vector_matches_size(
            cheap_step.reduced_hessian_times_step,
            reduced_size)) {
      // The rejected cheap exact_ctx solve already formed the accepted-point
      // core H*s. The full retry warm start only needs the missing relaxed
      // outer-response correction for the same reduced direction.
      warm_start.initial_hessian_times_step =
          cheap_step.reduced_hessian_times_step +
          exact_ctx_full_hvp_operator->apply_outer_response_only(
              warm_start.initial_reduced_step);
    } else {
      warm_start.initial_hessian_times_step =
          full_hvp_operator->apply(warm_start.initial_reduced_step);
    }
  }
  return warm_start;
}

AcceptedTruncatedNewtonStepControl
assess_accepted_nonredundant_truncated_newton_step(
    bool full_operator_available,
    bool sparse_orbital_chart,
    double trust_radius,
    double minimum_step_size,
    double max_trust_radius,
    double reject_shrink,
    double expand_ratio,
    double boundary_fraction,
    double previous_projected_gradient_inf_norm,
    double next_projected_gradient_inf_norm,
    double gradient_tolerance,
    int previous_consecutive_projected_stall_count,
    int previous_hybrid_followup_cooldown_remaining,
    double trust_ratio,
    bool cheap_trial_rejected,
    bool full_retry_attempted,
    bool full_operator_step_refined,
    bool used_full_model_probe,
    const TruncatedNewtonStepResult& accepted_step) {
  AcceptedTruncatedNewtonStepControl control;
  const bool used_krylov_rescue_step =
      accepted_step.used_krylov_rescue;

  if (used_krylov_rescue_step) {
    // A Krylov rescue step means the raw PCG model was numerically unreliable
    // at the current radius. Keep the rescued step, but preserve the regularized
    // trust-region behavior by contracting the next solve radius.
    control.trust_radius =
        std::max(minimum_step_size, reject_shrink * trust_radius);
  } else if (trust_ratio < 0.25) {
    control.trust_radius =
        std::max(minimum_step_size, reject_shrink * trust_radius);
  } else if (
      trust_ratio > expand_ratio &&
      (accepted_step.reached_boundary ||
       truncated_newton_step_effective_norm(accepted_step) >=
           boundary_fraction * trust_radius)) {
    control.trust_radius =
        std::min(max_trust_radius, 2.0 * trust_radius);
  } else {
    control.trust_radius = trust_radius;
  }

  control.stalled_projected_convergence =
      next_projected_gradient_inf_norm >=
      0.9 * std::max(
                previous_projected_gradient_inf_norm,
                gradient_tolerance);
  control.consecutive_projected_stall_count =
      control.stalled_projected_convergence
          ? previous_consecutive_projected_stall_count + 1
          : 0;
  const int hybrid_followup_cooldown_after_decay =
      std::max(0, previous_hybrid_followup_cooldown_remaining - 1);
  control.hybrid_followup_cooldown_remaining =
      used_full_model_probe
          ? exact_ctx_hybrid_followup_probe_cooldown_accepted_iterations(
                sparse_orbital_chart)
          : hybrid_followup_cooldown_after_decay;
  control.previous_iteration_reliable_for_transport =
      !cheap_trial_rejected &&
      !full_retry_attempted &&
      !full_operator_step_refined &&
      !accepted_step.reached_boundary &&
      !accepted_step.encountered_negative_curvature &&
      !control.stalled_projected_convergence &&
      trust_ratio >= expand_ratio &&
      !used_krylov_rescue_step;
  control.enable_hybrid_bridge_next_iteration =
      assess_exact_ctx_hybrid_bridge_for_next_iteration(
          full_operator_available,
          sparse_orbital_chart,
          control.stalled_projected_convergence,
          control.consecutive_projected_stall_count,
          control.hybrid_followup_cooldown_remaining,
          trust_ratio,
          used_krylov_rescue_step);
  return control;
}

double estimate_nonredundant_reduced_model_decrease(
    const NonredundantOrbitalSpace::ProjectionResult& projection,
    const Eigen::VectorXd& reduced_step,
    ReducedHvpOperator* hvp_operator) {
  const Eigen::VectorXd reduced_hessian_step =
      hvp_operator->apply(reduced_step);
  return
      -projection.reduced_gradient.dot(reduced_step) -
      0.5 * reduced_step.dot(reduced_hessian_step);
}

TruncatedNewtonStepResult solve_nonredundant_truncated_newton_step(
    const NonredundantRetractionMetric& retraction_metric,
    const NonredundantOrbitalSpace& current_space,
    const NonredundantOrbitalSpace::ProjectionResult& current_projection,
    double trust_radius,
    int max_cg_iterations,
    double gradient_tolerance,
    ReducedHvpOperator* hvp_operator,
    const TransportedReducedLbfgsPreconditioner* transported_preconditioner,
    const Eigen::VectorXd* initial_reduced_step = nullptr,
    const Eigen::VectorXd* initial_hessian_times_step = nullptr) {
  TruncatedNewtonStepResult result;
  const Eigen::VectorXd fallback_step =
      build_nonredundant_preconditioned_reduced_gradient_step(
          retraction_metric,
          current_space,
          current_projection,
          trust_radius,
          transported_preconditioner);
  result.reduced_step =
      Eigen::VectorXd::Zero(current_projection.reduced_gradient.size());
  result.reduced_hessian_times_step =
      Eigen::VectorXd::Zero(current_projection.reduced_gradient.size());
  std::vector<Eigen::VectorXd> krylov_basis_vectors;
  std::vector<Eigen::VectorXd> krylov_tangent_basis_vectors;
  std::vector<Eigen::VectorXd> krylov_hessian_basis_vectors;
  krylov_basis_vectors.reserve(
      std::max(0, max_cg_iterations) + (initial_reduced_step != nullptr ? 1 : 0));
  krylov_tangent_basis_vectors.reserve(krylov_basis_vectors.capacity());
  krylov_hessian_basis_vectors.reserve(krylov_basis_vectors.capacity());
  auto finalize_result = [&]() -> TruncatedNewtonStepResult {
    result.krylov_subspace =
        build_truncated_newton_krylov_subspace(
            current_projection.reduced_gradient,
            krylov_basis_vectors,
            krylov_tangent_basis_vectors,
            krylov_hessian_basis_vectors);
    return result;
  };
  if (current_projection.reduced_gradient.size() == 0 ||
      max_cg_iterations <= 0) {
    result.reduced_step = fallback_step;
    result.reduced_hessian_times_step.resize(0);
    return finalize_result();
  }

  const Eigen::VectorXd rhs = -current_projection.reduced_gradient;
  Eigen::VectorXd residual = rhs;
  if (initial_reduced_step != nullptr) {
    const double initial_step_norm = retraction_metric.norm(*initial_reduced_step);
    if (std::isfinite(initial_step_norm) &&
        initial_step_norm > 0.0 &&
        initial_step_norm < trust_radius) {
      result.used_initial_step = true;
      Eigen::VectorXd hessian_times_initial_step;
      if (initial_hessian_times_step != nullptr) {
        hessian_times_initial_step = *initial_hessian_times_step;
      } else {
        result.warm_start_hvp_performed = true;
        hessian_times_initial_step =
            hvp_operator->apply(*initial_reduced_step);
      }

      if (hessian_times_initial_step.allFinite()) {
        result.reduced_step = *initial_reduced_step;
        result.reduced_hessian_times_step = hessian_times_initial_step;
        result.predicted_decrease =
            rhs.dot(result.reduced_step) -
            0.5 * result.reduced_step.dot(hessian_times_initial_step);
        residual.noalias() -= hessian_times_initial_step;
        append_truncated_newton_krylov_basis_vector(
            retraction_metric,
            *initial_reduced_step,
            hessian_times_initial_step,
            &krylov_basis_vectors,
            &krylov_tangent_basis_vectors,
            &krylov_hessian_basis_vectors);
      }
    }
  }

  Eigen::VectorXd preconditioned_residual =
      apply_nonredundant_truncated_newton_preconditioner(
          current_space,
          transported_preconditioner,
          residual);
  Eigen::VectorXd search_direction = preconditioned_residual;
  double residual_dot_preconditioned =
      residual.dot(preconditioned_residual);
  if (!std::isfinite(residual_dot_preconditioned) ||
      residual_dot_preconditioned <= 0.0) {
    if (result.reduced_step.squaredNorm() > 0.0 &&
        current_projection.reduced_gradient.dot(result.reduced_step) < 0.0 &&
        std::isfinite(result.predicted_decrease) &&
        result.predicted_decrease > 0.0) {
      return finalize_result();
    }
    result.reduced_step = fallback_step;
    result.reduced_hessian_times_step.resize(0);
    result.predicted_decrease = 0.0;
    return finalize_result();
  }

  const double initial_residual_inf_norm =
      gradient_infinity_norm(residual);
  // The accepted-point HVP is only worthwhile when the inner CG solve drives
  // the trust-region residual well below the outer convergence threshold.
  // Stopping at 5% of the initial residual left FeCl2/MnF2-scale problems in
  // a long linear tail, so tighten the default target while keeping env hooks
  // for cheaper exploratory runs.
  const double residual_inf_target =
      std::max(
          nonredundant_truncated_newton_relative_residual_target_fraction() *
              initial_residual_inf_norm,
          nonredundant_truncated_newton_absolute_residual_target_gradient_multiple() *
              gradient_tolerance);
  if (initial_residual_inf_norm <= residual_inf_target) {
    if (result.reduced_step.squaredNorm() > 0.0 &&
        current_projection.reduced_gradient.dot(result.reduced_step) < 0.0 &&
        std::isfinite(result.predicted_decrease) &&
        result.predicted_decrease > 0.0) {
      return finalize_result();
    }
    result.reduced_step = fallback_step;
    result.reduced_hessian_times_step.resize(0);
    result.predicted_decrease = 0.0;
    return finalize_result();
  }

  constexpr double kCurvatureTolerance =
      64.0 * std::numeric_limits<double>::epsilon();
  for (int cg_iteration = 0;
       cg_iteration < max_cg_iterations;
       ++cg_iteration) {
    const Eigen::VectorXd hessian_times_direction =
        hvp_operator->apply(search_direction);
    append_truncated_newton_krylov_basis_vector(
        retraction_metric,
        search_direction,
        hessian_times_direction,
        &krylov_basis_vectors,
        &krylov_tangent_basis_vectors,
        &krylov_hessian_basis_vectors);
    const double curvature =
        search_direction.dot(hessian_times_direction);
    const Eigen::VectorXd current_step_tangent =
        retraction_metric.tangent(result.reduced_step);
    const Eigen::VectorXd search_direction_tangent =
        retraction_metric.tangent(search_direction);
    const double search_direction_metric_norm_squared =
        search_direction_tangent.squaredNorm();
    if (!std::isfinite(curvature) ||
        curvature <=
            kCurvatureTolerance *
            std::max(search_direction.squaredNorm(),
                     search_direction_metric_norm_squared)) {
      result.encountered_negative_curvature = true;
      result.reached_boundary = true;
      if (search_direction_metric_norm_squared > 0.0 &&
          std::isfinite(search_direction_metric_norm_squared)) {
        const double tau =
            solve_trust_region_metric_boundary_tau(
                current_step_tangent,
                search_direction_tangent,
                trust_radius);
        result.predicted_decrease +=
            tau * residual.dot(search_direction) -
            0.5 * tau * tau * curvature;
        result.reduced_step.noalias() +=
            tau *
            search_direction;
        result.reduced_hessian_times_step.noalias() +=
            tau *
            hessian_times_direction;
      }
      break;
    }

    const double alpha =
        residual_dot_preconditioned / curvature;
    if (!std::isfinite(alpha) || alpha <= 0.0) {
      result.reduced_step = fallback_step;
      result.reduced_hessian_times_step.resize(0);
      return finalize_result();
    }

    const Eigen::VectorXd candidate_step =
        result.reduced_step + alpha * search_direction;
    const double candidate_step_metric_norm =
        retraction_metric.norm(candidate_step);
    if (candidate_step_metric_norm >= trust_radius) {
      result.reached_boundary = true;
      const double tau =
          solve_trust_region_metric_boundary_tau(
              current_step_tangent,
              search_direction_tangent,
              trust_radius);
      result.predicted_decrease +=
          tau * residual.dot(search_direction) -
          0.5 * tau * tau * curvature;
      result.reduced_step.noalias() +=
          tau *
          search_direction;
      result.reduced_hessian_times_step.noalias() +=
          tau *
          hessian_times_direction;
      break;
    }

    result.predicted_decrease +=
        alpha * residual.dot(search_direction) -
        0.5 * alpha * alpha * curvature;
    result.reduced_step = candidate_step;
    result.retract_tangent_norm = candidate_step_metric_norm;
    result.reduced_hessian_times_step.noalias() +=
        alpha *
        hessian_times_direction;
    residual.noalias() -= alpha * hessian_times_direction;
    result.cg_iterations = cg_iteration + 1;
    if (gradient_infinity_norm(residual) <= residual_inf_target) {
      break;
    }

    preconditioned_residual =
        apply_nonredundant_truncated_newton_preconditioner(
            current_space,
            transported_preconditioner,
            residual);
    const double next_residual_dot_preconditioned =
        residual.dot(preconditioned_residual);
    if (!std::isfinite(next_residual_dot_preconditioned) ||
        next_residual_dot_preconditioned <= 0.0) {
      break;
    }
    const double beta =
        next_residual_dot_preconditioned / residual_dot_preconditioned;
    if (!std::isfinite(beta) || beta < 0.0) {
      break;
    }
    search_direction =
        preconditioned_residual + beta * search_direction;
    residual_dot_preconditioned = next_residual_dot_preconditioned;
  }

  // CG is used here to collect a local HVP subspace. The final candidate must be
  // the trust-region minimizer in the accepted-point retraction metric
  // ||J d||, not the raw Euclidean PCG accumulation, otherwise sparse charts
  // solve the wrong spherical subproblem and tend to exhaust the radius.
  const TruncatedNewtonKrylovSubspace krylov_subspace =
      build_truncated_newton_krylov_subspace(
          current_projection.reduced_gradient,
          krylov_basis_vectors,
          krylov_tangent_basis_vectors,
          krylov_hessian_basis_vectors);
  auto metric_trust_region_step =
      solve_trust_region_in_krylov_subspace(
          current_projection,
          trust_radius,
          krylov_subspace);
  if (truncated_newton_step_is_usable(
          metric_trust_region_step,
          current_projection.reduced_gradient)) {
    metric_trust_region_step.used_initial_step = result.used_initial_step;
    metric_trust_region_step.warm_start_hvp_performed =
        result.warm_start_hvp_performed;
    metric_trust_region_step.encountered_negative_curvature =
        metric_trust_region_step.encountered_negative_curvature ||
        result.encountered_negative_curvature;
    metric_trust_region_step.cg_iterations = result.cg_iterations;
    return metric_trust_region_step;
  }

  const bool result_step_is_usable =
      std::isfinite(result.reduced_step.norm()) &&
      result.reduced_step.squaredNorm() > 0.0 &&
      current_projection.reduced_gradient.dot(result.reduced_step) < 0.0 &&
      std::isfinite(result.predicted_decrease) &&
      result.predicted_decrease > 0.0;
  if (!result_step_is_usable) {
    result.reduced_step = fallback_step;
    result.reduced_hessian_times_step.resize(0);
    result.predicted_decrease = 0.0;
  }
  return finalize_result();
}

ExactCtxHybridRefinementResult maybe_refine_exact_ctx_step_with_full_operator(
    const NonredundantRetractionMetric& retraction_metric,
    const NonredundantOrbitalSpace& current_space,
    const NonredundantOrbitalSpace::ProjectionResult& current_projection,
    double trust_radius,
    int max_refinement_cg_iterations,
    double gradient_tolerance,
    ReducedHvpOperator* full_hvp_operator,
    const TransportedReducedLbfgsPreconditioner* transported_preconditioner,
    const TruncatedNewtonStepResult& cheap_step,
    const Eigen::VectorXd* cheap_core_hessian_times_step) {
  ExactCtxHybridRefinementResult result;
  result.step = cheap_step;
  if (full_hvp_operator == nullptr ||
      cheap_step.reduced_step.size() != current_projection.reduced_gradient.size()) {
    return result;
  }

  const double step_norm = cheap_step.reduced_step.norm();
  if (!(step_norm > 0.0) || !std::isfinite(step_norm)) {
    return result;
  }

  Eigen::VectorXd full_hessian_times_step;
  const Eigen::VectorXd* reference_cheap_core_hessian_times_step =
      (cheap_core_hessian_times_step != nullptr &&
       cheap_core_hessian_times_step->size() ==
           current_projection.reduced_gradient.size() &&
       cheap_core_hessian_times_step->allFinite())
          ? cheap_core_hessian_times_step
          : (cheap_step.reduced_hessian_times_step.size() ==
                     current_projection.reduced_gradient.size() &&
                 cheap_step.reduced_hessian_times_step.allFinite())
          ? &cheap_step.reduced_hessian_times_step
          : nullptr;
  {
    auto* exact_ctx_full_hvp_operator =
        static_cast<ExactContextReducedHvpOperator*>(full_hvp_operator);
    assert(exact_ctx_full_hvp_operator != nullptr);
    if (reference_cheap_core_hessian_times_step != nullptr) {
      // The cheap exact_ctx solve already knows the accepted-point core H*s. When
      // probing the full model on the same step, only the relaxed outer-response
      // correction is missing.
      full_hessian_times_step =
          *reference_cheap_core_hessian_times_step +
          exact_ctx_full_hvp_operator->apply_outer_response_only(
              cheap_step.reduced_step);
    } else {
      full_hessian_times_step =
          full_hvp_operator->apply(cheap_step.reduced_step);
    }
  }
  result.used_full_model_probe = true;
  if (full_hessian_times_step.size() != current_projection.reduced_gradient.size() ||
      !full_hessian_times_step.allFinite()) {
    return result;
  }
  result.step.reduced_hessian_times_step = full_hessian_times_step;

  const Eigen::VectorXd rhs = -current_projection.reduced_gradient;
  const double full_predicted_decrease =
      rhs.dot(cheap_step.reduced_step) -
      0.5 * cheap_step.reduced_step.dot(full_hessian_times_step);
  if (std::isfinite(full_predicted_decrease) &&
      full_predicted_decrease > 0.0) {
    result.step.predicted_decrease = full_predicted_decrease;
  }
  {
    std::vector<Eigen::VectorXd> full_probe_basis_vectors;
    std::vector<Eigen::VectorXd> full_probe_tangent_basis_vectors;
    std::vector<Eigen::VectorXd> full_probe_hessian_basis_vectors;
    append_truncated_newton_krylov_basis_vector(
        retraction_metric,
        cheap_step.reduced_step,
        full_hessian_times_step,
        &full_probe_basis_vectors,
        &full_probe_tangent_basis_vectors,
        &full_probe_hessian_basis_vectors);
    result.step.krylov_subspace =
        build_truncated_newton_krylov_subspace(
            current_projection.reduced_gradient,
            full_probe_basis_vectors,
            full_probe_tangent_basis_vectors,
            full_probe_hessian_basis_vectors);
  }

  const Eigen::VectorXd full_residual =
      rhs - full_hessian_times_step;
  const double reference_residual_inf_norm =
      std::max(gradient_infinity_norm(rhs), std::numeric_limits<double>::min());
  result.full_step_residual_ratio =
      gradient_infinity_norm(full_residual) / reference_residual_inf_norm;

  if (std::isfinite(full_predicted_decrease) &&
      full_predicted_decrease > 0.0 &&
      std::isfinite(cheap_step.predicted_decrease) &&
      cheap_step.predicted_decrease > 0.0) {
    const double predicted_decrease_scale =
        std::max(
            std::abs(full_predicted_decrease),
            std::abs(cheap_step.predicted_decrease));
    if (predicted_decrease_scale > 0.0) {
      result.full_model_gap_ratio =
          std::abs(full_predicted_decrease - cheap_step.predicted_decrease) /
          predicted_decrease_scale;
    }
  }

  // The cheap exact_ctx inner solve omits the relaxed outer response. One full
  // HVP on the cheap step quantifies that model mismatch. When the cheap step
  // looks inaccurate, fall back directly to a few full-model CG iterations
  // instead of trying to patch the cheap Krylov model in place. Do not exclude
  // boundary-limited steps here: sparse open-shell runs can need the relaxed
  // model exactly when cheap boundary steps start pushing into a bad basin.
  constexpr double kResidualRatioThreshold = 0.35;
  constexpr double kModelGapRatioThreshold = 0.25;
  constexpr double kRefinementBoundaryFraction = 0.9;
  const double boundary_refinement_gradient_tolerance_multiple =
      std::max(
          0.0,
          parse_env_double_with_default(
              "XMVB_CPP_EXACT_CTX_HYBRID_BOUNDARY_REFINE_GRAD_MULTIPLE",
              1.0e12));
  const bool boundary_limited_step =
      cheap_step.reached_boundary ||
      step_norm >= kRefinementBoundaryFraction * trust_radius;
  const double projected_gradient_inf_norm =
      gradient_infinity_norm(current_projection.reduced_gradient);
  const bool boundary_refinement_allowed =
      !boundary_limited_step ||
      (std::isfinite(projected_gradient_inf_norm) &&
       projected_gradient_inf_norm <=
           boundary_refinement_gradient_tolerance_multiple *
               std::max(gradient_tolerance, 0.0));
  const bool full_model_disagrees =
      !std::isfinite(full_predicted_decrease) ||
      full_predicted_decrease <= 0.0 ||
      result.full_step_residual_ratio > kResidualRatioThreshold ||
      result.full_model_gap_ratio > kModelGapRatioThreshold;
  const bool detected_model_mismatch =
      !cheap_step.encountered_negative_curvature &&
      boundary_refinement_allowed &&
      full_model_disagrees;
  if (!detected_model_mismatch) {
    return result;
  }

  const bool should_refine =
      max_refinement_cg_iterations > 0;
  if (!should_refine) {
    return result;
  }

  TruncatedNewtonStepResult refined_step =
      solve_nonredundant_truncated_newton_step(
          retraction_metric,
          current_space,
          current_projection,
          trust_radius,
          max_refinement_cg_iterations,
          gradient_tolerance,
          full_hvp_operator,
          transported_preconditioner,
          &cheap_step.reduced_step,
          &full_hessian_times_step);
  if (refined_step.reduced_step.size() !=
          current_projection.reduced_gradient.size() ||
      !refined_step.reduced_step.allFinite() ||
      !std::isfinite(refined_step.predicted_decrease) ||
      refined_step.predicted_decrease <= 0.0 ||
      current_projection.reduced_gradient.dot(refined_step.reduced_step) >= 0.0) {
    return result;
  }

  result.step = std::move(refined_step);
  result.used_refinement = true;
  return result;
}

bool try_armijo_backtracking_direction(
    OrbitalObjective* objective,
    const Eigen::VectorXd& current_parameters,
    double current_energy,
    const Eigen::VectorXd& current_gradient,
    const Eigen::VectorXd& search_direction,
    double initial_step,
    double minimum_step,
    double armijo_constant,
    Eigen::VectorXd* accepted_parameters,
    Eigen::VectorXd* accepted_gradient,
    double* accepted_energy) {
  const double directional_derivative =
      current_gradient.dot(search_direction);
  if (!std::isfinite(directional_derivative) ||
      directional_derivative >= 0.0) {
    return false;
  }

  double step = std::max(minimum_step, initial_step);
  Eigen::VectorXd trial_parameters(current_parameters.size());
  // This helper is shared by the L-BFGS stall fallback and the nonredundant
  // projected-gradient backend. Both use the same sufficient-decrease test on
  // the full relaxed orbital objective.
  while (step >= minimum_step) {
    trial_parameters.noalias() =
        current_parameters + step * search_direction;
    auto trial_evaluation =
        objective->evaluate_trial_without_committing(trial_parameters);
    const double trial_energy = trial_evaluation.energy;
    const double armijo_upper_bound =
        current_energy + armijo_constant * step * directional_derivative;
    if (std::isfinite(trial_energy) && trial_energy <= armijo_upper_bound) {
      *accepted_parameters = trial_parameters;
      *accepted_gradient = trial_evaluation.gradient;
      *accepted_energy = trial_energy;
      objective->commit_trial_evaluation(std::move(trial_evaluation));
      return true;
    }
    step *= 0.5;
  }

  return false;
}

bool try_build_nonredundant_lifted_trial_parameters(
    const OrbitalPreparationInput& current_orbital_input,
    const NonredundantOrbitalSpace& current_space,
    const SparseOrbitalParameterView& parameter_view,
    const Eigen::VectorXd& reduced_step,
    Eigen::VectorXd* trial_parameters) {
  const Eigen::VectorXd current_parameters =
      parameter_view.pack(current_orbital_input);
  // The reduced coordinates parameterize a local tangent vector on the
  // accepted sparse-orbital chart. Build finite trial points with the same
  // retraction used by the reduced HVP finite-difference operator, then pack
  // the resulting orbital table back into the optimizer coordinate vector.
  try {
    const OrbitalPreparationInput trial_orbital_input =
        current_space.retract_step(
            current_orbital_input,
            reduced_step);
    *trial_parameters =
        parameter_view.pack(trial_orbital_input);
  } catch (const std::exception&) {
    // In a line search or trust-radius retry, leaving the local retraction
    // chart means this trial is too large; callers can shrink and retry.
    return false;
  }
  if (trial_parameters->size() != current_parameters.size() ||
      !trial_parameters->allFinite()) {
    return false;
  }
  return true;
}

bool try_armijo_backtracking_nonredundant_direction(
    OrbitalObjective* objective,
    const OrbitalPreparationInput& current_orbital_input,
    const NonredundantOrbitalSpace& current_space,
    const SparseOrbitalParameterView& parameter_view,
    const Eigen::VectorXd& current_parameters,
    double current_energy,
    const Eigen::VectorXd& current_gradient,
    const Eigen::VectorXd& reduced_search_direction,
    const Eigen::VectorXd& packed_tangent_search_direction,
    double initial_step,
    double minimum_step,
    double armijo_constant,
    Eigen::VectorXd* accepted_parameters,
    Eigen::VectorXd* accepted_gradient,
    double* accepted_energy) {
  const double directional_derivative =
      current_gradient.dot(packed_tangent_search_direction);
  if (!std::isfinite(directional_derivative) ||
      directional_derivative >= 0.0) {
    return false;
  }

  double step = std::max(minimum_step, initial_step);
  Eigen::VectorXd trial_parameters(current_parameters.size());
  // Keep the accepted-point state in the original sparse-coefficient chart,
  // but generate finite trial points with the nonredundant orbital-increment
  // lift rather than by adding the reduced direction directly in packed sparse
  // coordinates.
  while (step >= minimum_step) {
    if (!try_build_nonredundant_lifted_trial_parameters(
            current_orbital_input,
            current_space,
            parameter_view,
            step * reduced_search_direction,
            &trial_parameters)) {
      step *= 0.5;
      continue;
    }
    if (is_effectively_zero_step(
            trial_parameters - current_parameters,
            current_parameters)) {
      step *= 0.5;
      continue;
    }

    auto trial_evaluation =
        objective->evaluate_trial_without_committing(trial_parameters);
    const double trial_energy = trial_evaluation.energy;
    const double armijo_upper_bound =
        current_energy + armijo_constant * step * directional_derivative;
    if (std::isfinite(trial_energy) && trial_energy <= armijo_upper_bound) {
      *accepted_parameters = trial_parameters;
      *accepted_gradient = trial_evaluation.gradient;
      *accepted_energy = trial_energy;
      objective->commit_trial_evaluation(std::move(trial_evaluation));
      return true;
    }
    step *= 0.5;
  }

  return false;
}

bool try_armijo_backtracking_step(
    OrbitalObjective* objective,
    const Eigen::VectorXd& current_parameters,
    double current_energy,
    const Eigen::VectorXd& current_gradient,
    double initial_step,
    double minimum_step,
    double armijo_constant,
    Eigen::VectorXd* accepted_parameters,
    Eigen::VectorXd* accepted_gradient,
    double* accepted_energy) {
  // This fallback intentionally switches to steepest descent with Armijo-only
  // decrease. It is more tolerant to small gradient inconsistencies from
  // highly parallel reductions than a failed quasi-Newton line search.
  const Eigen::VectorXd search_direction = -current_gradient;
  return try_armijo_backtracking_direction(
      objective,
      current_parameters,
      current_energy,
      current_gradient,
      search_direction,
      initial_step,
      minimum_step,
      armijo_constant,
      accepted_parameters,
      accepted_gradient,
      accepted_energy);
}

void sync_result_from_objective(
    const OrbitalObjective& objective,
    CppVbScfOptimizerResult* result);

void record_accepted_iteration_snapshot(
    OrbitalObjective* objective,
    int iteration,
    const CppVbScfOptimizerOptions& options,
    CppVbScfOptimizerResult* result);

struct NonredundantFullSpacePolishResult {
  int accepted_iterations = 0;
  bool reached_dual_tolerance = false;
};

NonredundantFullSpacePolishResult run_nonredundant_full_space_polish(
    OrbitalObjective* objective,
    const CppVbScfOptimizerOptions& options,
    const char* dual_tolerance_reason,
    const char* budget_reason,
    Eigen::VectorXd* current_parameters,
    Eigen::VectorXd* current_gradient,
    double* energy,
    double* previous_energy,
    int* n_iterations,
    CppVbScfOptimizerResult* result,
    double* final_gradient_l2_norm) {
  NonredundantFullSpacePolishResult polish_result;
  if (objective == nullptr ||
      current_parameters == nullptr ||
      current_gradient == nullptr ||
      energy == nullptr ||
      previous_energy == nullptr ||
      n_iterations == nullptr ||
      result == nullptr ||
      final_gradient_l2_norm == nullptr) {
    return polish_result;
  }

  const int n = static_cast<int>(current_parameters->size());
  const int polish_iteration_budget =
      std::min(
          options.nonredundant_polish_max_iterations,
          std::max(0, options.max_iterations - *n_iterations));
  const double polish_gradient_tolerance =
      std::max(
          std::numeric_limits<double>::epsilon(),
          std::min(
              options.gradient_tolerance,
              options.gradient_tolerance *
                  options.nonredundant_polish_gradient_scale));
  if (n <= 0 ||
      polish_iteration_budget <= 0 ||
      gradient_infinity_norm(*current_gradient) <=
          polish_gradient_tolerance) {
    return polish_result;
  }

  LBFGSpp::BFGSMat<double> polish_inverse_hessian;
  polish_inverse_hessian.reset(n, options.history_size);
  Eigen::VectorXd search_direction = -*current_gradient;
  Eigen::VectorXd previous_parameters(n);
  Eigen::VectorXd previous_gradient(n);

  // The reduced nonredundant phase removes the expensive tangent-space error.
  // This short full-coordinate polish recovers any remaining support-local
  // gradient component that is invisible to the reduced projector, with all
  // vectors in the packed sparse-orbital coefficient convention.
  for (int polish_iteration = 0;
       polish_iteration < polish_iteration_budget;
       ++polish_iteration) {
    if (search_direction.dot(*current_gradient) >= 0.0 ||
        is_effectively_zero_step(search_direction, *current_parameters)) {
      search_direction = -*current_gradient;
    }
    const double directional_derivative =
        current_gradient->dot(search_direction);
    if (!std::isfinite(directional_derivative) ||
        directional_derivative >= 0.0) {
      break;
    }

    previous_parameters = *current_parameters;
    previous_gradient = *current_gradient;
    Eigen::VectorXd accepted_parameters(current_parameters->size());
    Eigen::VectorXd accepted_gradient(current_gradient->size());
    double accepted_energy = *energy;
    if (!try_armijo_backtracking_direction(
            objective,
            *current_parameters,
            *energy,
            *current_gradient,
            search_direction,
            std::min(1.0, options.initial_step_size),
            options.minimum_step_size,
            options.armijo_constant,
            &accepted_parameters,
            &accepted_gradient,
            &accepted_energy)) {
      break;
    }

    *current_parameters = std::move(accepted_parameters);
    *current_gradient = std::move(accepted_gradient);
    *energy = accepted_energy;
    ++(*n_iterations);
    ++polish_result.accepted_iterations;
    sync_result_from_objective(*objective, result);
    record_accepted_iteration_snapshot(
        objective,
        *n_iterations,
        options,
        result);
    *final_gradient_l2_norm = current_gradient->norm();

    const double de = *energy - *previous_energy;
    *previous_energy = *energy;
    if (std::abs(de) < options.energy_tolerance &&
        result->gradient_inf_norm_history.back() <
            polish_gradient_tolerance) {
      result->converged = true;
      result->termination_reason = dual_tolerance_reason;
      polish_result.reached_dual_tolerance = true;
      break;
    }

    const Eigen::VectorXd parameter_step =
        *current_parameters - previous_parameters;
    const Eigen::VectorXd gradient_step =
        *current_gradient - previous_gradient;
    if (parameter_step.dot(gradient_step) >
        std::numeric_limits<double>::epsilon() *
            gradient_step.squaredNorm()) {
      polish_inverse_hessian.add_correction(parameter_step, gradient_step);
    }
    polish_inverse_hessian.apply_Hv(
        *current_gradient,
        -1.0,
        search_direction);
  }

  if (polish_result.accepted_iterations > 0 &&
      !polish_result.reached_dual_tolerance) {
    result->converged = false;
    result->termination_reason = budget_reason;
  }
  return polish_result;
}

bool try_steepest_descent_armijo_fallback(
    OrbitalObjective* objective,
    const LBFGSpp::LBFGSParam<double>& param,
    const Eigen::VectorXd& start_parameters,
    const Eigen::VectorXd& start_gradient,
    double start_energy,
    double initial_step,
    Eigen::VectorXd* accepted_parameters,
    Eigen::VectorXd* accepted_gradient,
    double* accepted_energy,
    double* accepted_step) {
  const Eigen::VectorXd direction = -start_gradient;
  const double directional_derivative = start_gradient.dot(direction);
  if (!(directional_derivative < 0.0)) {
    return false;
  }

  double reference_step =
      std::max(param.min_step, std::min(initial_step, param.max_step));
  std::vector<double> trial_steps;
  trial_steps.push_back(reference_step);

  constexpr int kMaxExpansionTrials = 4;
  double expanded_step = reference_step;
  for (int trial = 0; trial < kMaxExpansionTrials; ++trial) {
    if (expanded_step >= param.max_step) {
      break;
    }
    const double next_step =
        std::min(param.max_step, expanded_step * 2.0);
    if (next_step <= expanded_step) {
      break;
    }
    trial_steps.push_back(next_step);
    expanded_step = next_step;
  }

  double contracted_step = reference_step;
  while (contracted_step > param.min_step) {
    contracted_step *= 0.5;
    if (contracted_step < param.min_step) {
      contracted_step = param.min_step;
    }
    if (contracted_step < trial_steps.back()) {
      trial_steps.push_back(contracted_step);
    }
    if (contracted_step <= param.min_step) {
      break;
    }
  }

  bool has_best_descent = false;
  Eigen::VectorXd best_parameters;
  Eigen::VectorXd best_gradient;
  double best_energy = start_energy;
  double best_step = reference_step;
  for (double step : trial_steps) {
    Eigen::VectorXd trial_parameters =
        (start_parameters + step * direction).eval();
    if (is_effectively_zero_step(
            trial_parameters - start_parameters,
            start_parameters)) {
      continue;
    }
    Eigen::VectorXd trial_gradient;
    const double trial_energy =
        (*objective)(trial_parameters, trial_gradient);
    if (std::isfinite(trial_energy) &&
        trial_gradient.allFinite() &&
        trial_energy < best_energy) {
      has_best_descent = true;
      best_parameters = trial_parameters;
      best_gradient = trial_gradient;
      best_energy = trial_energy;
      best_step = step;
    }
    if (std::isfinite(trial_energy) &&
        trial_gradient.allFinite() &&
        trial_energy <= start_energy + param.ftol * step * directional_derivative) {
      *accepted_parameters = std::move(trial_parameters);
      *accepted_gradient = std::move(trial_gradient);
      *accepted_energy = trial_energy;
      *accepted_step = step;
      return true;
    }
  }

  if (has_best_descent) {
    *accepted_parameters = std::move(best_parameters);
    *accepted_gradient = std::move(best_gradient);
    *accepted_energy = best_energy;
    *accepted_step = best_step;
    return true;
  }

  return false;
}

void sync_result_from_objective(
    const OrbitalObjective& objective,
    CppVbScfOptimizerResult* result) {
  result->total_energy_history = objective.energy_history();
  result->gradient_inf_norm_history = objective.gradient_inf_norm_history();
  result->iteration_time_history_seconds = objective.iteration_time_history_seconds();
  result->scf_result = objective.last_gradient_result().scf_result;
}

void record_accepted_iteration_snapshot(
    OrbitalObjective* objective,
    int accepted_iteration_index,
    const CppVbScfOptimizerOptions& options,
    CppVbScfOptimizerResult* result) {
  if (!options.retain_accepted_iteration_trace && !options.accepted_iteration_callback) {
    return;
  }
  const bool include_reference_energy_gradient =
      options.retain_accepted_iteration_trace ||
      options.accepted_iteration_callback_requires_reference_gradient;
  const bool include_full_payload =
      options.retain_accepted_iteration_trace ||
      options.accepted_iteration_callback_requires_full_snapshot;
  // Lightweight callback snapshots only need scalar iteration summaries. Skip
  // the heavyweight matrix/integral deep copies unless the retained trace or a
  // callback explicitly requested the full payload.
  if (include_reference_energy_gradient) {
    objective->ensure_last_reference_energy_gradient();
  }
  const auto& gradient_result = objective->last_gradient_result();
  CppVbScfAcceptedIterationSnapshot snapshot;
  snapshot.accepted_iteration_index = accepted_iteration_index;
  snapshot.has_full_payload = include_full_payload;
  for (const double value : gradient_result.sparse_orbital_energy_gradient) {
    snapshot.sparse_orbital_energy_gradient_inf_norm =
        std::max(snapshot.sparse_orbital_energy_gradient_inf_norm, std::abs(value));
    snapshot.sparse_orbital_energy_gradient_l2_norm += value * value;
  }
  snapshot.sparse_orbital_energy_gradient_l2_norm =
      std::sqrt(snapshot.sparse_orbital_energy_gradient_l2_norm);
  if (include_full_payload) {
    snapshot.orbital_value_table.assign(
        objective->last_input().orbital_preparation_input.orbital_value_table.begin(),
        objective->last_input().orbital_preparation_input.orbital_value_table.end());
    snapshot.structure_matrices = gradient_result.scf_result.structure_matrices;
    snapshot.active_orbital_overlap_matrix =
        gradient_result.active_orbital_overlap_matrix;
    snapshot.active_one_electron_integrals =
        gradient_result.active_one_electron_integrals;
    snapshot.packed_active_two_electron_integrals =
        gradient_result.packed_active_two_electron_integrals;
    snapshot.sparse_orbital_energy_gradient =
        gradient_result.sparse_orbital_energy_gradient;
    if (include_reference_energy_gradient) {
      snapshot.sparse_orbital_reference_energy_gradient =
          gradient_result.sparse_orbital_reference_energy_gradient;
    }
  }
  snapshot.total_energy = gradient_result.scf_result.total_energy;
  snapshot.one_electron_reference_energy =
      gradient_result.scf_result.one_electron_reference_energy;
  snapshot.average_structure_overlap =
      gradient_result.scf_result.average_structure_overlap;
  if (options.retain_accepted_iteration_trace) {
    result->accepted_iteration_trace.push_back(snapshot);
  }
  if (options.accepted_iteration_callback) {
    options.accepted_iteration_callback(snapshot);
  }
}

}  // namespace

CppVbScfOptimizer::CppVbScfOptimizer(
    CppVbScfOptimizerOptions options)
    : orbital_gradient_evaluator_(options.algorithm),
      scf_evaluator_(options.algorithm),
      options_(options) {}

CppVbScfOptimizer::CppVbScfOptimizer(
    CppOrbitalGradientEvaluator orbital_gradient_evaluator,
    CppVbScfEvaluator scf_evaluator,
    CppVbScfOptimizerOptions options)
    : orbital_gradient_evaluator_(std::move(orbital_gradient_evaluator)),
      scf_evaluator_(std::move(scf_evaluator)),
      options_(options) {}

CppVbScfOptimizerResult CppVbScfOptimizer::optimize(
    const CppVbInput& input,
    double nuclear_repulsion_energy) const {
  return optimize(input, {0}, {1.0}, nuclear_repulsion_energy);
}

CppVbScfOptimizerResult CppVbScfOptimizer::optimize(
    const CppVbInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy) const {
  if (options_.max_iterations <= 0) {
    throw std::invalid_argument("max_iterations must be positive");
  }
  if (options_.gradient_tolerance <= 0.0 ||
      options_.energy_tolerance <= 0.0 ||
      options_.initial_step_size <= 0.0 ||
      options_.minimum_step_size <= 0.0) {
    throw std::invalid_argument("optimizer tolerances and step sizes must be positive");
  }
  if (options_.minimum_step_size > options_.initial_step_size) {
    throw std::invalid_argument("minimum_step_size must not exceed initial_step_size");
  }
  if (options_.history_size <= 0) {
    throw std::invalid_argument("history_size must be positive");
  }
  if (options_.nonredundant_polish_max_iterations < 0) {
    throw std::invalid_argument("nonredundant_polish_max_iterations must not be negative");
  }
  if (options_.nonredundant_polish_gradient_scale <= 0.0) {
    throw std::invalid_argument("nonredundant_polish_gradient_scale must be positive");
  }
  if (options_.nonredundant_truncated_newton_max_cg_iterations < 0) {
    throw std::invalid_argument(
        "nonredundant_truncated_newton_max_cg_iterations must be nonnegative");
  }
  if (options_.nonredundant_truncated_newton_hvp_step_size <= 0.0) {
    throw std::invalid_argument(
        "nonredundant_truncated_newton_hvp_step_size must be positive");
  }
  if (options_.nonredundant_truncated_newton_transport_history_size < 0) {
    throw std::invalid_argument(
        "nonredundant_truncated_newton_transport_history_size must be nonnegative");
  }
  if (!cpp_vb_scf_optimizer_backend_supported(options_.backend)) {
    throw std::invalid_argument(
        "requested optimizer backend is not enabled in this build");
  }
  if (options_.backend == CppVbScfOptimizerBackend::DeepVBHOnnx) {
    throw std::invalid_argument(
        "deepvbh_onnx requires DeepVBHOnnxHybridOptimizer and runtime metadata");
  }

  CppVbScfOptimizerResult result;
  const auto optimization_start_time = std::chrono::steady_clock::now();

  // For `guess=mo`, numerical parity with the legacy VBSCF implementation is
  // more important than any temporary convergence-speed heuristic. Keep the
  // nonredundant optimizer on the original legacy sparse chart and exact-
  // support block partition so the reduced coordinates, projected gradients,
  // and exact-context orbital derivatives all live on the same variational
  // manifold as the reference `.xmo` calculation.
  std::optional<CppVbInput> adapted_optimizer_input;
  const CppVbInput* optimizer_input = &input;
  if (optimizer_backend_uses_nonredundant_space(options_.backend)) {
    adapted_optimizer_input = build_nonredundant_optimizer_input(input);
    optimizer_input = &adapted_optimizer_input.value();
  }
  const SparseOrbitalParameterView parameter_view(
      optimizer_input->orbital_preparation_input);
  Eigen::VectorXd parameter_vector =
      parameter_view.pack(optimizer_input->orbital_preparation_input);
  Eigen::MatrixXd initial_normalized_orbital_matrix;

  OrbitalObjective objective(
      *optimizer_input,
      parameter_view,
      selected_state_indices,
      state_average_weights,
      nuclear_repulsion_energy,
      &orbital_gradient_evaluator_,
      &scf_evaluator_);
  const int n = static_cast<int>(parameter_vector.size());
  int n_iterations = 0;
  double final_gradient_l2_norm = 0.0;
  bool final_projected_gradient_ready = false;

  try {
    Eigen::VectorXd gradient(parameter_vector.size());
    double energy = objective(parameter_vector, gradient);
    initial_normalized_orbital_matrix =
        objective.last_gradient_result()
            .orbital_preparation_result
            .physical_orbital_frame
            .normalized_orbital_matrix;
    objective.set_oeo_active_reference_orbitals(
        initial_normalized_orbital_matrix);
    sync_result_from_objective(objective, &result);
    record_accepted_iteration_snapshot(&objective, 0, options_, &result);
    result.initial_total_energy = energy;
    result.initial_one_electron_reference_energy =
        objective.last_gradient_result().scf_result.one_electron_reference_energy;
    double previous_energy = energy;
    final_gradient_l2_norm = gradient.norm();
    switch (options_.backend) {
      case CppVbScfOptimizerBackend::LegacyFortran: {
#ifdef XMVB_CPP_ENABLE_LEGACY_FORTRAN_BACKEND
        const int m = options_.history_size;
        std::vector<double> x(n);
        std::memcpy(x.data(), parameter_vector.data(), sizeof(double) * n);
        std::vector<double> gradient_buffer(n, 0.0);
        std::memcpy(
            gradient_buffer.data(),
            gradient.data(),
            sizeof(double) * n);
        std::vector<double> workspace(
            n * 2 * m + 1 +
                2 * m,
            0.0);
        std::vector<double> diag(n, 1.0);

        int diagco = 0;
        int iprint[2] = {-1, 0};
        int iflag = 0;
        double eps = 1.0e-5;
        double xtol = 1.0e-16;
        double gxn = final_gradient_l2_norm;

        auto evaluate_current_point = [&]() {
          Eigen::Map<Eigen::VectorXd> parameter_map(x.data(), n);
          Eigen::VectorXd gradient_map;
          energy = objective(parameter_map, gradient_map);
          std::memcpy(
              gradient_buffer.data(),
              gradient_map.data(),
              sizeof(double) * n);
          sync_result_from_objective(objective, &result);
          final_gradient_l2_norm = gradient_map.norm();
        };

        while (true) {
          lbfgs_driver_(
              const_cast<int*>(&n),
              const_cast<int*>(&m),
              x.data(),
              &energy,
              gradient_buffer.data(),
              &diagco,
              diag.data(),
              iprint,
              &eps,
              &xtol,
              workspace.data(),
              &iflag,
              &gxn);

          if (iflag == 1) {
            evaluate_current_point();
            ++n_iterations;
            record_accepted_iteration_snapshot(&objective, n_iterations, options_, &result);

            const double de = energy - previous_energy;
            previous_energy = energy;
            if (std::abs(de) < options_.energy_tolerance &&
                gxn < options_.gradient_tolerance) {
              result.converged = true;
              result.termination_reason = "legacy_dual_tolerance";
              iflag = 0;
              break;
            }

            if (n_iterations >= options_.max_iterations) {
              result.termination_reason = "max_iterations";
              break;
            }

            continue;
          }

          if (iflag == 0) {
            result.converged = true;
            if (result.termination_reason.empty()) {
              result.termination_reason = "lbfgs_driver_finished";
            }
            break;
          }

          if (iflag == 2) {
            result.termination_reason = "unexpected_diag_request";
            break;
          }

          result.termination_reason = "lbfgs_driver_error";
          break;
        }
        final_gradient_l2_norm = gxn;
        break;
#else
        result.termination_reason = "legacy_fortran_backend_disabled";
        break;
#endif
      }

      case CppVbScfOptimizerBackend::Lbfgspp: {
        LBFGSpp::LBFGSParam<double> param;
        param.m = options_.history_size;
        param.epsilon = 0.0;
        param.epsilon_rel = 0.0;
        param.past = 0;
        param.delta = 0.0;
        param.max_iterations = 0;
        param.max_linesearch = 20;
        param.min_step = options_.minimum_step_size;
        // Let the primary line search expand beyond the nominal unit trial
        // step, while still clamping it to a finite orbital-parameter radius.
        param.max_step =
            std::max(
                options_.initial_step_size,
                options_.initial_step_size * kLineSearchExpansionFactor);
        param.ftol = options_.armijo_constant;
        param.wolfe = 0.9;
        param.linesearch = LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;
        param.check_param();

        LBFGSpp::BFGSMat<double> inverse_hessian;
        inverse_hessian.reset(n, param.m);

        Eigen::VectorXd current_parameters = parameter_vector;
        Eigen::VectorXd current_gradient = gradient;
        Eigen::VectorXd previous_parameters(n);
        Eigen::VectorXd previous_gradient(n);
        Eigen::VectorXd search_direction = -current_gradient;
        double last_robust_step = std::min(1.0, options_.initial_step_size);
        bool has_robust_step_history = false;
        bool last_iteration_used_fallback = false;
        constexpr double kCurvatureEpsilon = std::numeric_limits<double>::epsilon();
        constexpr double kTinyStepFactor = 10.0;
        constexpr double kRobustStepShrinkRatio = 0.1;
        constexpr double kSuspiciousPrimaryStepRatio = 0.1;

        for (int iteration = 0; iteration < options_.max_iterations; ++iteration) {
          if (search_direction.dot(current_gradient) >= 0.0) {
            search_direction = -current_gradient;
          }
          double directional_derivative = current_gradient.dot(search_direction);
          if (directional_derivative >= 0.0) {
            result.termination_reason = "lbfgspp_non_descent_direction";
            break;
          }

          previous_parameters = current_parameters;
          previous_gradient = current_gradient;
          const double reference_energy = energy;
          const double previous_gradient_inf_norm =
              gradient_infinity_norm(previous_gradient);
          double step = std::min(1.0, options_.initial_step_size);
          if (last_iteration_used_fallback && has_robust_step_history) {
            step = std::max(
                param.min_step,
                std::min(last_robust_step, param.max_step));
          }
          bool used_fallback = false;
          bool reset_inverse_hessian = false;
          std::string primary_line_search_error;

          try {
            LBFGSpp::LineSearchMoreThuente<double>::LineSearch(
                objective,
                param,
                previous_parameters,
                search_direction,
                param.max_step,
                step,
                energy,
                current_gradient,
                directional_derivative,
                current_parameters);
          } catch (const std::exception& error) {
            primary_line_search_error = error.what();
          }

          Eigen::VectorXd parameter_step = current_parameters - previous_parameters;
          const double primary_gradient_inf_norm =
              gradient_infinity_norm(current_gradient);
          const bool stalled_line_search =
              primary_line_search_error.empty() &&
              (is_effectively_zero_step(parameter_step, previous_parameters) ||
               line_search_made_no_meaningful_progress(
                   reference_energy,
                   energy,
                   previous_gradient_inf_norm,
                   primary_gradient_inf_norm,
                   options_.gradient_tolerance));
          const bool suspicious_primary_step =
              primary_line_search_error.empty() &&
              has_robust_step_history &&
              step < kSuspiciousPrimaryStepRatio * last_robust_step;
          const bool should_try_fallback =
              !primary_line_search_error.empty() ||
              (previous_gradient_inf_norm >= options_.gradient_tolerance &&
               (stalled_line_search ||
                step <= kTinyStepFactor * param.min_step ||
                suspicious_primary_step));
          if (should_try_fallback) {
            const Eigen::VectorXd primary_parameters = current_parameters;
            const Eigen::VectorXd primary_gradient = current_gradient;
            const double primary_energy = energy;
            double fallback_step =
                has_robust_step_history
                    ? last_robust_step
                    : std::max(
                          param.min_step,
                          std::min(
                              std::min(1.0, options_.initial_step_size),
                              param.max_step));
            Eigen::VectorXd fallback_parameters;
            Eigen::VectorXd fallback_gradient;
            double fallback_energy = reference_energy;
            if (try_steepest_descent_armijo_fallback(
                    &objective,
                    param,
                    previous_parameters,
                    previous_gradient,
                    reference_energy,
                    fallback_step,
                    &fallback_parameters,
                    &fallback_gradient,
                    &fallback_energy,
                    &fallback_step)) {
              const bool should_replace_primary =
                  !primary_line_search_error.empty() ||
                  stalled_line_search ||
                  step <= kTinyStepFactor * param.min_step ||
                  suspicious_primary_step ||
                  fallback_energy < primary_energy;
              if (should_replace_primary) {
                current_parameters = std::move(fallback_parameters);
                current_gradient = std::move(fallback_gradient);
                energy = fallback_energy;
                step = fallback_step;
                parameter_step = current_parameters - previous_parameters;
                used_fallback = true;
                reset_inverse_hessian = true;
              } else {
                current_parameters = primary_parameters;
                current_gradient = primary_gradient;
                energy = primary_energy;
              }
            } else if (!primary_line_search_error.empty() ||
                       stalled_line_search) {
              energy = objective(previous_parameters, current_gradient);
              current_parameters = previous_parameters;
              sync_result_from_objective(objective, &result);
              final_gradient_l2_norm = current_gradient.norm();
              result.termination_reason =
                  !primary_line_search_error.empty()
                      ? std::string("lbfgspp_line_search: ") +
                            primary_line_search_error
                      : "lbfgspp_line_search_stalled";
              break;
            }
          }

          if (step > kTinyStepFactor * param.min_step &&
              (!has_robust_step_history ||
               step >= kRobustStepShrinkRatio * last_robust_step)) {
            last_robust_step = step;
            has_robust_step_history = true;
          }
          last_iteration_used_fallback = used_fallback;

          const bool accepted_point_chart_reset =
              objective.canonicalize_orbital_chart_at_current_point(
                  &current_parameters,
                  &current_gradient);
          if (accepted_point_chart_reset) {
            reset_inverse_hessian = true;
          }

          ++n_iterations;
          sync_result_from_objective(objective, &result);
          record_accepted_iteration_snapshot(&objective, n_iterations, options_, &result);
          final_gradient_l2_norm = current_gradient.norm();
          const double de = energy - previous_energy;
          previous_energy = energy;
          if (std::abs(de) < options_.energy_tolerance &&
              final_gradient_l2_norm < options_.gradient_tolerance) {
            result.converged = true;
            result.termination_reason = "lbfgspp_dual_tolerance";
            break;
          }

          Eigen::VectorXd gradient_step = current_gradient - previous_gradient;
          if (reset_inverse_hessian) {
            inverse_hessian.reset(n, param.m);
          }
          if (!reset_inverse_hessian &&
              parameter_step.dot(gradient_step) >
              kCurvatureEpsilon * gradient_step.squaredNorm()) {
            inverse_hessian.add_correction(parameter_step, gradient_step);
          }
          inverse_hessian.apply_Hv(current_gradient, -1.0, search_direction);
          if (used_fallback &&
              search_direction.dot(current_gradient) >= 0.0) {
            search_direction = -current_gradient;
          }
        }
        break;
      }

      case CppVbScfOptimizerBackend::NonredundantProjectedGradient: {
        Eigen::VectorXd current_parameters = parameter_vector;
        Eigen::VectorXd current_gradient = gradient;
        NonredundantOrbitalSpace current_space =
            build_nonredundant_space(objective, parameter_view);
        auto current_projection =
            current_space.project_gradient(current_gradient);

        for (int iteration = 0; iteration < options_.max_iterations; ++iteration) {
          if (current_space.reduced_size() == 0) {
            result.termination_reason = "nonredundant_space_empty";
            break;
          }

          const double reduced_gradient_inf_norm =
              gradient_infinity_norm(current_projection.reduced_gradient);
          if (iteration == 0 &&
              reduced_gradient_inf_norm < options_.gradient_tolerance) {
            result.converged = true;
            result.termination_reason =
                "nonredundant_projected_gradient_initial_tolerance";
            final_gradient_l2_norm = current_projection.reduced_gradient.norm();
            break;
          }

          const Eigen::VectorXd reduced_search_direction =
              -current_space.apply_inverse_reduced_block_preconditioner(
                  current_projection.reduced_gradient);
          const Eigen::VectorXd search_direction =
              gather_nonredundant_retract_tangent(
                  objective.last_input().orbital_preparation_input,
                  current_space,
                  parameter_view,
                  reduced_search_direction);
          // The accepted point remains in packed sparse coefficients, so the
          // projected-gradient backend must use the actual retraction tangent
          // rather than the raw additive chart when testing descent and Armijo.
          const double directional_derivative =
              current_gradient.dot(search_direction);
          if (!std::isfinite(directional_derivative) ||
              directional_derivative >= 0.0) {
            result.termination_reason =
                "nonredundant_projected_gradient_non_descent_direction";
            final_gradient_l2_norm = current_projection.reduced_gradient.norm();
            break;
          }

          const OrbitalPreparationInput current_orbital_input =
              objective.last_input().orbital_preparation_input;
          Eigen::VectorXd accepted_parameters(current_parameters.size());
          Eigen::VectorXd accepted_gradient(current_gradient.size());
          double accepted_energy = energy;
          if (!try_armijo_backtracking_nonredundant_direction(
                  &objective,
                  current_orbital_input,
                  current_space,
                  parameter_view,
                  current_parameters,
                  energy,
                  current_gradient,
                  reduced_search_direction,
                  search_direction,
                  std::min(1.0, options_.initial_step_size),
                  options_.minimum_step_size,
                  options_.armijo_constant,
                  &accepted_parameters,
                  &accepted_gradient,
                  &accepted_energy)) {
            result.termination_reason =
                "nonredundant_projected_gradient_line_search_failed";
            final_gradient_l2_norm = current_projection.reduced_gradient.norm();
            break;
          }

          current_parameters = std::move(accepted_parameters);
          current_gradient = std::move(accepted_gradient);
          energy = accepted_energy;
          objective.canonicalize_orbital_chart_at_current_point(
              &current_parameters,
              &current_gradient);
          ++n_iterations;
          sync_result_from_objective(objective, &result);
          record_accepted_iteration_snapshot(&objective, n_iterations, options_, &result);
          final_gradient_l2_norm = current_gradient.norm();
          const double de = energy - previous_energy;
          previous_energy = energy;
          NonredundantOrbitalSpace next_space =
              build_nonredundant_space(objective, parameter_view);
          auto next_projection =
              next_space.project_gradient(current_gradient);
          if (std::abs(de) < options_.energy_tolerance &&
              gradient_infinity_norm(next_projection.reduced_gradient) <
                  options_.gradient_tolerance) {
            result.converged = true;
            result.termination_reason =
                "nonredundant_projected_gradient_dual_tolerance";
            final_gradient_l2_norm = next_projection.reduced_gradient.norm();
            break;
          }

          current_space = std::move(next_space);
          current_projection = std::move(next_projection);
        }
        break;
      }

      case CppVbScfOptimizerBackend::NonredundantLbfgspp: {
        const int history_size = options_.history_size;
        const double polish_gradient_tolerance =
            std::max(
                std::numeric_limits<double>::epsilon(),
                std::min(
                    options_.gradient_tolerance,
                    options_.gradient_tolerance *
                        options_.nonredundant_polish_gradient_scale));
        LBFGSpp::BFGSMat<double> inverse_hessian;
        inverse_hessian.reset(n, history_size);

        Eigen::VectorXd current_parameters = parameter_vector;
        Eigen::VectorXd current_gradient = gradient;
        NonredundantOrbitalSpace current_space =
            build_nonredundant_space(objective, parameter_view);
        auto current_projection =
            current_space.project_gradient(current_gradient);
        Eigen::VectorXd previous_parameters(n);
        Eigen::VectorXd previous_gradient(n);
        Eigen::VectorXd previous_projected_gradient(n);
        constexpr double kCurvatureEpsilon = std::numeric_limits<double>::epsilon();

        for (int iteration = 0; iteration < options_.max_iterations; ++iteration) {
          if (current_space.reduced_size() == 0) {
            result.termination_reason = "nonredundant_space_empty";
            break;
          }

          const double reduced_gradient_inf_norm =
              gradient_infinity_norm(current_projection.reduced_gradient);
          if (iteration == 0 &&
              reduced_gradient_inf_norm < options_.gradient_tolerance) {
            result.converged = true;
            result.termination_reason =
                "nonredundant_lbfgspp_initial_tolerance";
            final_gradient_l2_norm = current_projection.reduced_gradient.norm();
            break;
          }

          const Eigen::VectorXd fallback_reduced_direction =
              -current_projection.reduced_gradient;
          const OrbitalPreparationInput previous_orbital_input =
              objective.last_input().orbital_preparation_input;
          const Eigen::VectorXd fallback_direction =
              gather_nonredundant_retract_tangent(
                  previous_orbital_input,
                  current_space,
                  parameter_view,
                  fallback_reduced_direction);
          Eigen::VectorXd search_direction;
          Eigen::VectorXd reduced_search_direction;
          inverse_hessian.apply_Hv(
              current_projection.packed_projected_gradient,
              -1.0,
              search_direction);
          {
            const auto search_projection =
                current_space.project_vector(search_direction);
            reduced_search_direction = search_projection.reduced_gradient;
            search_direction =
                gather_nonredundant_retract_tangent(
                    previous_orbital_input,
                    current_space,
                    parameter_view,
                    reduced_search_direction);
          }
          double directional_derivative =
              current_gradient.dot(search_direction);
          if (!std::isfinite(directional_derivative) ||
              directional_derivative >= 0.0 ||
              is_effectively_zero_step(search_direction, current_parameters)) {
            inverse_hessian.reset(n, history_size);
            search_direction = fallback_direction;
            reduced_search_direction = fallback_reduced_direction;
            directional_derivative = current_gradient.dot(search_direction);
          }
          if (!std::isfinite(directional_derivative) ||
              directional_derivative >= 0.0) {
            result.termination_reason =
                "nonredundant_lbfgspp_non_descent_direction";
            final_gradient_l2_norm = current_projection.reduced_gradient.norm();
            break;
          }

          previous_parameters = current_parameters;
          previous_gradient = current_gradient;
          previous_projected_gradient = current_projection.packed_projected_gradient;
          const Eigen::VectorXd previous_reduced_gradient =
              current_projection.reduced_gradient;
          const double reference_energy = energy;
          Eigen::VectorXd accepted_parameters(current_parameters.size());
          Eigen::VectorXd accepted_gradient(current_gradient.size());
          double accepted_energy = energy;
          if (!try_armijo_backtracking_nonredundant_direction(
                  &objective,
                  previous_orbital_input,
                  current_space,
                  parameter_view,
                  current_parameters,
                  energy,
                  current_gradient,
                  reduced_search_direction,
                  search_direction,
                  std::min(1.0, options_.initial_step_size),
                  options_.minimum_step_size,
                  options_.armijo_constant,
                  &accepted_parameters,
                  &accepted_gradient,
                  &accepted_energy)) {
            result.termination_reason =
                "nonredundant_lbfgspp_line_search_failed";
            final_gradient_l2_norm = current_projection.reduced_gradient.norm();
            break;
          }

          current_parameters = std::move(accepted_parameters);
          current_gradient = std::move(accepted_gradient);
          energy = accepted_energy;

          NonredundantOrbitalSpace next_space =
              build_nonredundant_space(objective, parameter_view);
          auto next_projection =
              next_space.project_gradient(current_gradient);
          double next_reduced_gradient_inf_norm =
              gradient_infinity_norm(next_projection.reduced_gradient);

          Eigen::VectorXd parameter_step = current_parameters - previous_parameters;
          bool recovered_from_stall = false;
          const bool stalled_line_search =
              is_effectively_zero_step(parameter_step, previous_parameters) ||
              line_search_made_no_meaningful_progress(
                  reference_energy,
                  energy,
                  reduced_gradient_inf_norm,
                  next_reduced_gradient_inf_norm,
                  options_.gradient_tolerance);
          if (stalled_line_search &&
              reduced_gradient_inf_norm >= options_.gradient_tolerance) {
            const double fallback_initial_step =
                std::max(options_.minimum_step_size,
                         std::min(
                             std::min(1.0, options_.initial_step_size),
                             1.0 / std::max(1.0, reduced_gradient_inf_norm)));
            if (!try_armijo_backtracking_nonredundant_direction(
                    &objective,
                    previous_orbital_input,
                    current_space,
                    parameter_view,
                    previous_parameters,
                    reference_energy,
                    previous_gradient,
                    -previous_reduced_gradient,
                    gather_nonredundant_retract_tangent(
                        previous_orbital_input,
                        current_space,
                        parameter_view,
                        -previous_reduced_gradient),
                    fallback_initial_step,
                    options_.minimum_step_size,
                    options_.armijo_constant,
                    &current_parameters,
                    &current_gradient,
                    &energy)) {
              energy = objective(previous_parameters, current_gradient);
              current_parameters = previous_parameters;
              sync_result_from_objective(objective, &result);
              final_gradient_l2_norm = current_gradient.norm();
              result.termination_reason = "nonredundant_lbfgspp_line_search_stalled";
              break;
            }
            next_space = build_nonredundant_space(objective, parameter_view);
            next_projection =
                next_space.project_gradient(current_gradient);
            next_reduced_gradient_inf_norm =
                gradient_infinity_norm(next_projection.reduced_gradient);
            parameter_step = current_parameters - previous_parameters;
            inverse_hessian.reset(n, history_size);
            recovered_from_stall = true;
          }

          const bool accepted_point_chart_reset =
              objective.canonicalize_orbital_chart_at_current_point(
                  &current_parameters,
                  &current_gradient);
          if (accepted_point_chart_reset) {
            next_space = build_nonredundant_space(objective, parameter_view);
            next_projection =
                next_space.project_gradient(current_gradient);
            next_reduced_gradient_inf_norm =
                gradient_infinity_norm(next_projection.reduced_gradient);
          }

          ++n_iterations;
          sync_result_from_objective(objective, &result);
          record_accepted_iteration_snapshot(&objective, n_iterations, options_, &result);
          final_gradient_l2_norm = current_gradient.norm();
          const double de = energy - previous_energy;
          previous_energy = energy;
          if (std::abs(de) < options_.energy_tolerance &&
              next_reduced_gradient_inf_norm <
                  options_.gradient_tolerance) {
            result.converged = true;
            result.termination_reason =
                "nonredundant_lbfgspp_dual_tolerance";
            final_gradient_l2_norm = next_projection.reduced_gradient.norm();
            break;
          }

          if (accepted_point_chart_reset) {
            inverse_hessian.reset(n, history_size);
          } else {
            parameter_step = current_parameters - previous_parameters;
            const Eigen::VectorXd projected_gradient_step =
                next_projection.packed_projected_gradient -
                previous_projected_gradient;
            if (parameter_step.dot(projected_gradient_step) >
                kCurvatureEpsilon * projected_gradient_step.squaredNorm()) {
              inverse_hessian.add_correction(
                  parameter_step,
                  projected_gradient_step);
            } else if (recovered_from_stall) {
              inverse_hessian.reset(n, history_size);
            }
          }

          current_space = std::move(next_space);
          current_projection = std::move(next_projection);
        }

        const int polish_iteration_budget =
            std::min(
                options_.nonredundant_polish_max_iterations,
                std::max(0, options_.max_iterations - n_iterations));
        const double latest_objective_seconds =
            result.iteration_time_history_seconds.empty()
                ? 0.0
                : result.iteration_time_history_seconds.back();
        const bool should_run_polish =
            polish_iteration_budget > 0 &&
            result.converged &&
            (result.termination_reason == "nonredundant_lbfgspp_initial_tolerance" ||
             result.termination_reason == "nonredundant_lbfgspp_tolerance") &&
            latest_objective_seconds >= kNonredundantPolishMinIterationSeconds &&
            gradient_infinity_norm(current_gradient) > polish_gradient_tolerance;
        if (should_run_polish) {
          LBFGSpp::BFGSMat<double> polish_inverse_hessian;
          polish_inverse_hessian.reset(n, history_size);
          Eigen::VectorXd search_direction = -current_gradient;
          int accepted_polish_iterations = 0;
          bool polish_reached_dual_tolerance = false;

          // The nonredundant stage removes the expensive near-null directions
          // quickly. A short full-space polish then recovers the remaining
          // energy in the original coordinates without paying for dozens of
          // full-space iterations from the beginning.
          for (int polish_iteration = 0;
               polish_iteration < polish_iteration_budget;
               ++polish_iteration) {
            if (search_direction.dot(current_gradient) >= 0.0 ||
                is_effectively_zero_step(search_direction, current_parameters)) {
              search_direction = -current_gradient;
            }
            const double directional_derivative =
                current_gradient.dot(search_direction);
            if (!std::isfinite(directional_derivative) ||
                directional_derivative >= 0.0) {
              break;
            }

            previous_parameters = current_parameters;
            previous_gradient = current_gradient;
            Eigen::VectorXd accepted_parameters(current_parameters.size());
            Eigen::VectorXd accepted_gradient(current_gradient.size());
            double accepted_energy = energy;
            if (!try_armijo_backtracking_direction(
                    &objective,
                    current_parameters,
                    energy,
                    current_gradient,
                    search_direction,
                    std::min(1.0, options_.initial_step_size),
                    options_.minimum_step_size,
                    options_.armijo_constant,
                    &accepted_parameters,
                    &accepted_gradient,
                    &accepted_energy)) {
              break;
            }

            current_parameters = std::move(accepted_parameters);
            current_gradient = std::move(accepted_gradient);
            energy = accepted_energy;
            ++n_iterations;
            ++accepted_polish_iterations;
            sync_result_from_objective(objective, &result);
            record_accepted_iteration_snapshot(&objective, n_iterations, options_, &result);
            final_gradient_l2_norm = current_gradient.norm();

            const double de = energy - previous_energy;
            previous_energy = energy;
            if (std::abs(de) < options_.energy_tolerance &&
                result.gradient_inf_norm_history.back() <
                    polish_gradient_tolerance) {
              result.converged = true;
              result.termination_reason =
                  "nonredundant_lbfgspp_polish_dual_tolerance";
              polish_reached_dual_tolerance = true;
              break;
            }

            const Eigen::VectorXd parameter_step =
                current_parameters - previous_parameters;
            const Eigen::VectorXd gradient_step =
                current_gradient - previous_gradient;
            if (parameter_step.dot(gradient_step) >
                kCurvatureEpsilon * gradient_step.squaredNorm()) {
              polish_inverse_hessian.add_correction(parameter_step, gradient_step);
            }
            polish_inverse_hessian.apply_Hv(
                current_gradient,
                -1.0,
                search_direction);
          }

          if (accepted_polish_iterations > 0 && !polish_reached_dual_tolerance) {
            result.converged = true;
            result.termination_reason = "nonredundant_lbfgspp_polish_budget";
          }
        }
        break;
      }

      case CppVbScfOptimizerBackend::NonredundantTruncatedNewton: {
        Eigen::VectorXd current_parameters = parameter_vector;
        Eigen::VectorXd current_gradient = gradient;
        NonredundantOrbitalSpace current_space =
            build_nonredundant_space(objective, parameter_view);
        auto current_projection =
            current_space.project_gradient(current_gradient);
        double final_projected_gradient_inf_norm =
            gradient_infinity_norm(current_projection.reduced_gradient);
        const double initial_projected_gradient_inf_norm =
            final_projected_gradient_inf_norm;
        double final_projected_gradient_l2_norm =
            current_projection.reduced_gradient.norm();
        std::vector<PackedSecantPair> packed_secant_history;
        packed_secant_history.reserve(
            std::max(
                0,
                options_.nonredundant_truncated_newton_transport_history_size));
        // The trust-region subproblem still lives in reduced block-local
        // nonredundant coordinates even though accepted trial points are
        // lifted back to sparse coefficients. In that reduced chart,
        // amplitudes much larger than O(1) no longer define meaningful local
        // trust regions, so clamp the initial and maximum radii to a
        // chart-scale value instead of the much looser raw-parameter default.
        const double chart_scale_initial_step =
            std::max(
                options_.minimum_step_size,
                std::min(1.0, options_.initial_step_size));
        double trust_radius = chart_scale_initial_step;
        const double max_trust_radius =
            std::max(trust_radius, 8.0 * chart_scale_initial_step);
        constexpr double kAcceptRatio = 0.1;
        constexpr double kRejectShrink = 0.25;
        constexpr double kExpandRatio = 0.75;
        constexpr double kBoundaryFraction = 0.8;
        int rejected_trial_step_count_for_current_point = 0;
        RejectedTruncatedNewtonStepCache rejected_step_cache;
        TruncatedNewtonKrylovSubspace cached_cheap_krylov_subspace;
        TruncatedNewtonKrylovSubspace cached_full_krylov_subspace;
        Eigen::VectorXd previous_accepted_packed_step;
        bool previous_accepted_iteration_reliable_for_transport = false;
        int consecutive_projected_stall_count = 0;
        ExactCtxHybridStrategyState exact_ctx_hybrid_strategy_state;
        HvpModelQualityState hvp_model_quality_state;
        MissingCurvatureSr1OuterResponseModel sr1_outer_response_model(
            exact_ctx_sr1_outer_response_approximation_enabled());
        const auto reset_tn_state_after_full_space_update = [&]() {
          // Full-coordinate polish moves on the packed sparse coefficient
          // chart, outside the accepted-point reduced TN model.  Any Krylov
          // basis, transported secant pair, or trust-radius calibration from
          // before that move no longer belongs to the rebuilt NROS chart.
          packed_secant_history.clear();
          rejected_step_cache.clear();
          cached_cheap_krylov_subspace = TruncatedNewtonKrylovSubspace();
          cached_full_krylov_subspace = TruncatedNewtonKrylovSubspace();
          previous_accepted_packed_step = Eigen::VectorXd();
          previous_accepted_iteration_reliable_for_transport = false;
          consecutive_projected_stall_count = 0;
          rejected_trial_step_count_for_current_point = 0;
          exact_ctx_hybrid_strategy_state = ExactCtxHybridStrategyState();
          hvp_model_quality_state.reset_after_chart_change();
          sr1_outer_response_model.clear();
          trust_radius = chart_scale_initial_step;
        };

        while (n_iterations < options_.max_iterations) {
          if (current_space.reduced_size() == 0) {
            if (gradient_infinity_norm(current_gradient) <=
                options_.gradient_tolerance) {
              result.converged = true;
              result.termination_reason =
                  "nonredundant_truncated_newton_full_gradient_tolerance";
              final_gradient_l2_norm = current_gradient.norm();
              break;
            }
            const NonredundantFullSpacePolishResult polish_result =
                run_nonredundant_full_space_polish(
                    &objective,
                    options_,
                    "nonredundant_truncated_newton_polish_dual_tolerance",
                    "nonredundant_truncated_newton_polish_budget",
                    &current_parameters,
                    &current_gradient,
                    &energy,
                    &previous_energy,
                    &n_iterations,
                    &result,
                    &final_gradient_l2_norm);
            if (polish_result.accepted_iterations > 0) {
              reset_tn_state_after_full_space_update();
              current_space = build_nonredundant_space(objective, parameter_view);
              current_projection =
                  current_space.project_gradient(current_gradient);
              final_projected_gradient_inf_norm =
                  gradient_infinity_norm(current_projection.reduced_gradient);
              final_projected_gradient_l2_norm =
                  current_projection.reduced_gradient.norm();
              if (result.converged) {
                break;
              }
              if (current_space.reduced_size() == 0) {
                final_gradient_l2_norm = current_gradient.norm();
                break;
              }
              result.converged = false;
              result.termination_reason.clear();
              continue;
            }
            result.termination_reason = "nonredundant_space_empty";
            break;
          }

          const double reduced_gradient_inf_norm =
              gradient_infinity_norm(current_projection.reduced_gradient);
          final_projected_gradient_inf_norm = reduced_gradient_inf_norm;
          final_projected_gradient_l2_norm =
              current_projection.reduced_gradient.norm();
          if (n_iterations == 0 &&
              reduced_gradient_inf_norm < options_.gradient_tolerance) {
            result.converged = true;
            result.termination_reason =
                "nonredundant_truncated_newton_initial_tolerance";
            final_gradient_l2_norm = current_projection.reduced_gradient.norm();
            break;
          }

          if (!(trust_radius > 0.0) || !std::isfinite(trust_radius)) {
            result.termination_reason =
                "nonredundant_truncated_newton_invalid_trust_radius";
            final_gradient_l2_norm = current_projection.reduced_gradient.norm();
            break;
          }

          std::unique_ptr<ReducedHvpOperator> hvp_operator;
          std::unique_ptr<ReducedHvpOperator> retry_with_full_hvp_operator;
          ExactContextReducedHvpOperator* cheap_exact_ctx_hvp_operator = nullptr;
          const int n_active_orbitals =
              objective.last_input().orbital_preparation_input
                  .n_active_orbitals;
          const ExactCtxSystemProfile system_profile =
              build_exact_ctx_system_profile(
                  objective.last_input().orbital_preparation_input);
          const ExactCtxDefaultStrategy strategy =
              choose_exact_ctx_default_strategy(system_profile);
          const double latest_objective_seconds =
              objective.iteration_time_history_seconds().empty()
                  ? 0.0
                  : objective.iteration_time_history_seconds().back();
          ExactCtxInnerSolvePolicy exact_ctx_inner_solve_policy =
              options_.nonredundant_truncated_newton_hvp_mode ==
                      NonredundantTruncatedNewtonHvpMode::ExactContextDirectAction
                  ? choose_exact_ctx_inner_solve_policy(
                        n_iterations,
                        system_profile,
                        reduced_gradient_inf_norm,
                        initial_projected_gradient_inf_norm,
                        options_.gradient_tolerance,
                        latest_objective_seconds,
                        consecutive_projected_stall_count,
                        exact_ctx_hybrid_strategy_state)
                  : ExactCtxInnerSolvePolicy();
          const bool model_quality_full_inner_requested =
              options_.nonredundant_truncated_newton_hvp_mode ==
                      NonredundantTruncatedNewtonHvpMode::ExactContextDirectAction &&
              hvp_model_quality_state.wants_full_inner_solve();
          if (model_quality_full_inner_requested &&
              !exact_ctx_inner_solve_policy.use_outer_response) {
            exact_ctx_inner_solve_policy.use_outer_response = true;
            exact_ctx_inner_solve_policy.used_hybrid_followup_full_solve = true;
          }
          const bool inner_solve_uses_outer_response =
              exact_ctx_inner_solve_policy.use_outer_response;
          const bool model_quality_full_retry_requested =
              options_.nonredundant_truncated_newton_hvp_mode ==
                      NonredundantTruncatedNewtonHvpMode::ExactContextDirectAction &&
              hvp_model_quality_state.wants_full_retry();
          const bool hybrid_followup_requested =
              exact_ctx_hybrid_strategy_state.request_followup ||
              model_quality_full_retry_requested;
          switch (options_.nonredundant_truncated_newton_hvp_mode) {
            case NonredundantTruncatedNewtonHvpMode::FullFiniteDifference:
              hvp_operator = std::make_unique<FullFiniteDifferenceReducedHvpOperator>(
                  objective,
                  current_space,
                  current_projection,
                  objective.last_input().orbital_preparation_input,
                  parameter_view,
                  current_parameters,
                  options_.nonredundant_truncated_newton_hvp_step_size);
              break;
            case NonredundantTruncatedNewtonHvpMode::ExactContextDirectAction: {
              if (exact_ctx_use_tiny_full_fd_fallback(system_profile)) {
                // On F2-scale sparse charts the full finite-difference HVP is
                // cheaper than the exact-context setup and avoids letting a
                // small analytic/FD model mismatch dominate smoke-test
                // convergence.  Larger systems still use the exact_ctx path.
                hvp_operator =
                    std::make_unique<FullFiniteDifferenceReducedHvpOperator>(
                        objective,
                        current_space,
                        current_projection,
                        objective.last_input().orbital_preparation_input,
                        parameter_view,
                        current_parameters,
                        options_.nonredundant_truncated_newton_hvp_step_size);
                maybe_log_exact_ctx_policy_decision(
                    n_iterations,
                    system_profile.sparse_orbital_chart,
                    n_active_orbitals,
                    reduced_gradient_inf_norm,
                    latest_objective_seconds,
                    false,
                    false,
                    exact_ctx_hybrid_strategy_state,
                    exact_ctx_inner_solve_policy);
                break;
              }
              auto exact_ctx_hvp_operator =
                  std::make_unique<ExactContextReducedHvpOperator>(
                      objective,
                      current_space,
                      current_projection,
                      options_.nonredundant_truncated_newton_hvp_step_size,
                      inner_solve_uses_outer_response,
                      !inner_solve_uses_outer_response &&
                              sr1_outer_response_model.enabled()
                          ? &sr1_outer_response_model
                          : nullptr);
              cheap_exact_ctx_hvp_operator = exact_ctx_hvp_operator.get();
              if (!exact_ctx_hvp_operator->supports_analytic_core_model()) {
                throw std::runtime_error(
                    build_exact_ctx_unavailable_message(*exact_ctx_hvp_operator));
              }
              const auto exact_ctx_info = exact_ctx_hvp_operator->diagnostics();
              const bool full_model_correction_allowed =
                  exact_ctx_full_model_correction_allowed(
                      system_profile,
                      consecutive_projected_stall_count);
              if ((exact_ctx_retry_rejected_step_with_full_operator(strategy) ||
                   model_quality_full_retry_requested ||
                   exact_ctx_hybrid_strategy_state.request_followup) &&
                  full_model_correction_allowed &&
                  exact_ctx_info.outer_response_enabled &&
                  !inner_solve_uses_outer_response) {
                auto retry_with_full_exact_ctx_hvp_operator =
                    std::make_unique<ExactContextReducedHvpOperator>(
                        objective,
                        current_space,
                        current_projection,
                        options_.nonredundant_truncated_newton_hvp_step_size,
                        true);
                retry_with_full_hvp_operator =
                    std::move(retry_with_full_exact_ctx_hvp_operator);
              }
              maybe_log_exact_ctx_policy_decision(
                  n_iterations,
                  system_profile.sparse_orbital_chart,
                  n_active_orbitals,
                  reduced_gradient_inf_norm,
                  latest_objective_seconds,
                  full_model_correction_allowed,
                  retry_with_full_hvp_operator != nullptr,
                  exact_ctx_hybrid_strategy_state,
                  exact_ctx_inner_solve_policy);
              hvp_operator = std::move(exact_ctx_hvp_operator);
              break;
            }
          }
          const auto update_exact_ctx_hybrid_cost_sample = [&]() {
            maybe_update_exact_ctx_hybrid_strategy_state_from_hvp_operator(
                hvp_operator.get(),
                &exact_ctx_hybrid_strategy_state);
            maybe_update_exact_ctx_hybrid_strategy_state_from_hvp_operator(
                retry_with_full_hvp_operator.get(),
                &exact_ctx_hybrid_strategy_state);
          };
          const auto log_exact_ctx_hvp_diagnostics = [&]() {
            maybe_log_exact_ctx_hvp_diagnostics(
                n_iterations,
                "cheap",
                hvp_operator.get());
            maybe_log_exact_ctx_hvp_diagnostics(
                n_iterations,
                "full_retry",
                retry_with_full_hvp_operator.get());
          };
          const int transport_history_size =
              choose_nonredundant_truncated_newton_transport_history_size(
                  options_,
                  objective);
          const auto transported_preconditioner =
              build_nonredundant_truncated_newton_preconditioner(
                  current_space,
                  packed_secant_history,
                  transport_history_size);
          const int max_cg_iterations =
              choose_nonredundant_truncated_newton_max_cg_iterations(
                  options_,
                  objective,
                  current_projection.reduced_gradient.size(),
                  exact_ctx_inner_solve_policy,
                  consecutive_projected_stall_count);
          const int hybrid_refinement_max_cg_iterations =
              std::min(
                  max_cg_iterations,
                  exact_ctx_hybrid_refinement_max_cg_iterations());
          const OrbitalPreparationInput current_orbital_input =
              objective.last_input().orbital_preparation_input;
          const NonredundantRetractionMetric retraction_metric(
              current_orbital_input,
              current_space,
              parameter_view);
          const auto admit_energy_only_trial_screen = [&]() {
            const auto& objective_time_history =
                objective.iteration_time_history_seconds();
            const double last_objective_seconds =
                objective_time_history.empty()
                    ? 0.0
                    : objective_time_history.back();
            // An energy-only probe is only worthwhile when a rejected trial
            // would otherwise trigger a materially more expensive full
            // objective evaluation. On FeCl2-scale systems the current
            // energy-only path can cost more than a full energy+gradient
            // objective, so blindly screening every rejected trial just turns
            // trust-region tail cleanup into extra work.
            constexpr double kMinimumObjectiveSecondsForScreening = 0.75;
            if (!(last_objective_seconds >=
                  kMinimumObjectiveSecondsForScreening)) {
              return false;
            }
            if (objective.energy_only_call_count() == 0) {
              return true;
            }
            const double last_energy_only_seconds =
                objective.last_energy_only_wall_time_seconds();
            constexpr double kMaximumEnergyOnlyToObjectiveRatio = 0.85;
            return
                std::isfinite(last_energy_only_seconds) &&
                last_energy_only_seconds > 0.0 &&
                last_energy_only_seconds <
                    kMaximumEnergyOnlyToObjectiveRatio *
                        last_objective_seconds;
          };
          auto try_truncated_newton_trial_step =
              [&](const Eigen::VectorXd& candidate_reduced_step,
                  double candidate_predicted_decrease,
                  bool screen_with_energy_only,
                  TruncatedNewtonTrialEvaluation* trial_evaluation,
                  Eigen::VectorXd* accepted_packed_step,
                  OrbitalObjective::TrialEvaluation* accepted_trial_evaluation,
                  Eigen::VectorXd* accepted_trial_parameters,
                  Eigen::VectorXd* accepted_trial_gradient,
                  double* accepted_trial_energy,
                  double* accepted_trust_ratio) -> bool {
                if (trial_evaluation != nullptr) {
                  *trial_evaluation = TruncatedNewtonTrialEvaluation();
                }
                if (!std::isfinite(candidate_predicted_decrease) ||
                    candidate_predicted_decrease <= 0.0) {
                  return false;
                }

                Eigen::VectorXd candidate_trial_parameters(
                    current_parameters.size());
                if (!try_build_nonredundant_lifted_trial_parameters(
                        current_orbital_input,
                        current_space,
                        parameter_view,
                        candidate_reduced_step,
                        &candidate_trial_parameters)) {
                  return false;
                }
                const Eigen::VectorXd candidate_packed_step =
                    candidate_trial_parameters - current_parameters;
                if (is_effectively_zero_step(
                        candidate_packed_step,
                        current_parameters)) {
                  return false;
                }

                double effective_predicted_decrease =
                    candidate_predicted_decrease;
                if (candidate_reduced_step.size() ==
                        current_projection.reduced_gradient.size()) {
                  const double reduced_linear_decrease =
                      -current_projection.reduced_gradient.dot(
                          candidate_reduced_step);
                  const double packed_retraction_linear_decrease =
                      -current_gradient.dot(candidate_packed_step);
                  if (std::isfinite(reduced_linear_decrease) &&
                      std::isfinite(packed_retraction_linear_decrease)) {
                    const double curvature_decrease =
                        candidate_predicted_decrease -
                        reduced_linear_decrease;
                    const double retraction_predicted_decrease =
                        packed_retraction_linear_decrease +
                        curvature_decrease;
                    if (std::isfinite(retraction_predicted_decrease) &&
                        retraction_predicted_decrease > 0.0) {
                      effective_predicted_decrease =
                          retraction_predicted_decrease;
                    }
                  }
                }
                if (!std::isfinite(effective_predicted_decrease) ||
                    effective_predicted_decrease <= 0.0) {
                  return false;
                }

                if (screen_with_energy_only) {
                  const double candidate_trial_energy =
                      objective.evaluate_energy_only(
                          candidate_trial_parameters);
                  const double actual_decrease =
                      energy - candidate_trial_energy;
                  const double candidate_trust_ratio =
                      actual_decrease / effective_predicted_decrease;
                  if (trial_evaluation != nullptr) {
                    trial_evaluation->actual_decrease = actual_decrease;
                    trial_evaluation->trust_ratio = candidate_trust_ratio;
                  }
                  if (!std::isfinite(candidate_trial_energy) ||
                      !std::isfinite(candidate_trust_ratio) ||
                      actual_decrease <= 0.0 ||
                      candidate_trust_ratio < kAcceptRatio) {
                    return false;
                  }
                }
                OrbitalObjective::TrialEvaluation candidate_trial_evaluation =
                    objective.evaluate_trial_without_committing(
                        candidate_trial_parameters);
                const double candidate_trial_energy =
                    candidate_trial_evaluation.energy;
                const double actual_decrease =
                    energy - candidate_trial_energy;
                const double candidate_trust_ratio =
                    actual_decrease / effective_predicted_decrease;
                if (trial_evaluation != nullptr) {
                  trial_evaluation->actual_decrease = actual_decrease;
                  trial_evaluation->trust_ratio = candidate_trust_ratio;
                }
                if (!std::isfinite(candidate_trial_energy) ||
                    !std::isfinite(candidate_trust_ratio) ||
                    actual_decrease <= 0.0 ||
                    candidate_trust_ratio < kAcceptRatio) {
                  return false;
                }

                *accepted_packed_step = candidate_packed_step;
                *accepted_trial_evaluation =
                    std::move(candidate_trial_evaluation);
                *accepted_trial_parameters = candidate_trial_parameters;
                *accepted_trial_gradient =
                    std::move(accepted_trial_evaluation->gradient);
                *accepted_trial_energy = candidate_trial_energy;
                *accepted_trust_ratio = candidate_trust_ratio;
                return true;
              };
          auto try_nonredundant_descent_fallback_step =
              [&](Eigen::VectorXd* accepted_packed_step,
                  OrbitalObjective* accepted_trial_objective,
                  Eigen::VectorXd* accepted_trial_parameters,
                  Eigen::VectorXd* accepted_trial_gradient,
                  double* accepted_trial_energy) -> bool {
                Eigen::VectorXd fallback_reduced_direction =
                    -apply_nonredundant_truncated_newton_preconditioner(
                        current_space,
                        &transported_preconditioner,
                        current_projection.reduced_gradient);
                Eigen::VectorXd search_direction =
                    gather_nonredundant_retract_tangent(
                        current_orbital_input,
                        current_space,
                        parameter_view,
                        fallback_reduced_direction);
                double directional_derivative =
                    current_gradient.dot(search_direction);
                if (!std::isfinite(directional_derivative) ||
                    directional_derivative >= 0.0 ||
                    is_effectively_zero_step(
                        search_direction,
                        current_parameters)) {
                  fallback_reduced_direction =
                      -current_projection.reduced_gradient;
                  search_direction =
                      gather_nonredundant_retract_tangent(
                          current_orbital_input,
                          current_space,
                          parameter_view,
                          fallback_reduced_direction);
                  directional_derivative =
                      current_gradient.dot(search_direction);
                }
                if (!std::isfinite(directional_derivative) ||
                    directional_derivative >= 0.0 ||
                    is_effectively_zero_step(
                        search_direction,
                        current_parameters)) {
                  return false;
                }

                const double reduced_search_direction_norm =
                    search_direction.norm();
                const double trust_radius_limited_initial_step =
                    std::isfinite(reduced_search_direction_norm) &&
                            reduced_search_direction_norm > 0.0
                        ? trust_radius / reduced_search_direction_norm
                        : options_.minimum_step_size;
                const double initial_fallback_step =
                    std::max(
                        options_.minimum_step_size,
                        std::min(
                            std::min(
                                std::min(1.0, options_.initial_step_size),
                                1.0 / std::max(1.0, reduced_gradient_inf_norm)),
                            trust_radius_limited_initial_step));
                // The descent fallback is entered only after the current
                // accepted-point Newton model already failed to produce an
                // acceptable trust-region step. Starting the Armijo backtrack
                // from a reduced step that already fits inside the current
                // trust radius avoids burning many full objective evaluations
                // just to rediscover the same radius contraction.
                OrbitalObjective fallback_objective =
                    objective.make_probe_copy();
                Eigen::VectorXd fallback_parameters(current_parameters.size());
                Eigen::VectorXd fallback_gradient(current_gradient.size());
                double fallback_energy = energy;
                if (!try_armijo_backtracking_nonredundant_direction(
                        &fallback_objective,
                        current_orbital_input,
                        current_space,
                        parameter_view,
                        current_parameters,
                        energy,
                        current_gradient,
                        fallback_reduced_direction,
                        search_direction,
                        initial_fallback_step,
                        options_.minimum_step_size,
                        options_.armijo_constant,
                        &fallback_parameters,
                        &fallback_gradient,
                        &fallback_energy)) {
                  return false;
                }

                *accepted_packed_step =
                    fallback_parameters - current_parameters;
                *accepted_trial_objective =
                    std::move(fallback_objective);
                *accepted_trial_parameters = std::move(fallback_parameters);
                *accepted_trial_gradient = std::move(fallback_gradient);
                *accepted_trial_energy = fallback_energy;
                return true;
              };
          const Eigen::Index reduced_size =
              current_projection.reduced_gradient.size();
          bool transported_warm_start_admitted = false;
          bool cheap_predicted_decrease_fallback = false;
          bool cheap_trial_rejected = false;
          TruncatedNewtonTrialEvaluation cheap_trial_evaluation;
          bool full_model_probe_used = inner_solve_uses_outer_response;
          bool full_retry_admitted = false;
          bool full_retry_attempted = false;
          bool reused_full_krylov_subspace = false;
          Eigen::VectorXd transported_initial_reduced_step;
          const Eigen::VectorXd* initial_reduced_step_for_current_solve = nullptr;
          if (rejected_step_cache.has_cheap_step(reduced_size)) {
            initial_reduced_step_for_current_solve =
                &rejected_step_cache.cheap_step;
          }
          if (initial_reduced_step_for_current_solve == nullptr &&
              finite_nonzero_vector_matches_size(
                  previous_accepted_packed_step,
                  current_parameters.size())) {
            transported_initial_reduced_step =
                shrink_nonredundant_reduced_step_inside_retract_tangent_radius(
                    current_orbital_input,
                    current_space,
                    parameter_view,
                    current_space
                        .project_vector(previous_accepted_packed_step)
                        .reduced_gradient,
                    trust_radius);
            if (finite_nonzero_vector_matches_size(
                    transported_initial_reduced_step,
                    reduced_size)) {
              transported_warm_start_admitted =
                  assess_nonredundant_truncated_newton_transported_initial_step(
                      retraction_metric,
                      current_space,
                      current_projection,
                      trust_radius,
                      transported_initial_reduced_step,
                      previous_accepted_iteration_reliable_for_transport);
              if (transported_warm_start_admitted) {
                initial_reduced_step_for_current_solve =
                    &transported_initial_reduced_step;
              }
            }
          }

          auto truncated_newton_step =
              solve_trust_region_in_krylov_subspace(
                  current_projection,
                  trust_radius,
                  cached_cheap_krylov_subspace);
          const bool reused_cheap_krylov_subspace =
              truncated_newton_step_is_usable(
                  truncated_newton_step,
                  current_projection.reduced_gradient);
          if (!reused_cheap_krylov_subspace) {
            cached_cheap_krylov_subspace = TruncatedNewtonKrylovSubspace();
            truncated_newton_step =
                solve_nonredundant_truncated_newton_step(
                    retraction_metric,
                    current_space,
                    current_projection,
                    trust_radius,
                    max_cg_iterations,
                    options_.gradient_tolerance,
                    hvp_operator.get(),
                    &transported_preconditioner,
                    initial_reduced_step_for_current_solve);
          }
          clamp_nonredundant_step_result_to_retract_tangent_radius(
              current_orbital_input,
              current_space,
              parameter_view,
              current_projection,
              trust_radius,
              &truncated_newton_step);
          if (truncated_newton_krylov_subspace_is_usable(
                  truncated_newton_step.krylov_subspace,
                  current_projection.reduced_gradient.size())) {
            cached_cheap_krylov_subspace =
                truncated_newton_step.krylov_subspace;
          }
          const TruncatedNewtonStepResult cheap_model_step =
              truncated_newton_step;
          Eigen::VectorXd reduced_step = truncated_newton_step.reduced_step;
          if (reduced_step.size() != current_projection.reduced_gradient.size()) {
            result.termination_reason =
                "nonredundant_truncated_newton_invalid_step_dimension";
            final_gradient_l2_norm = current_projection.reduced_gradient.norm();
            break;
          }
          double predicted_decrease =
              truncated_newton_step.predicted_decrease;
          if (!std::isfinite(predicted_decrease) ||
              predicted_decrease <= 0.0) {
            reduced_step =
                build_nonredundant_preconditioned_reduced_gradient_step(
                    retraction_metric,
                    current_space,
                    current_projection,
                    trust_radius,
                    &transported_preconditioner);
            predicted_decrease =
                estimate_nonredundant_reduced_model_decrease(
                    current_projection,
                    reduced_step,
                    hvp_operator.get());
            truncated_newton_step.reduced_step = reduced_step;
            truncated_newton_step.reduced_hessian_times_step.resize(0);
            truncated_newton_step.retract_tangent_norm =
                compute_nonredundant_retract_tangent_norm(
                    current_orbital_input,
                    current_space,
                    parameter_view,
                    reduced_step);
            truncated_newton_step.reached_boundary =
                truncated_newton_step.retract_tangent_norm >=
                (1.0 - 1.0e-8) * trust_radius;
            truncated_newton_step.predicted_decrease = predicted_decrease;
            cheap_predicted_decrease_fallback = true;
          }
          bool full_operator_step_refined = false;
          TruncatedNewtonStepResult trial_step_for_current_trial =
              truncated_newton_step;
          if (retry_with_full_hvp_operator != nullptr &&
              hybrid_followup_requested &&
              !exact_ctx_inner_solve_policy.used_hybrid_followup_full_solve) {
            Eigen::VectorXd cheap_core_hessian_times_step_for_full_probe;
            const Eigen::VectorXd* cheap_core_hessian_times_step_ptr = nullptr;
            if (cheap_exact_ctx_hvp_operator != nullptr &&
                !cheap_exact_ctx_hvp_operator
                     ->uses_sr1_outer_response_approximation() &&
                truncated_newton_step.reduced_hessian_times_step.size() ==
                    current_projection.reduced_gradient.size() &&
                truncated_newton_step.reduced_hessian_times_step.allFinite()) {
              cheap_core_hessian_times_step_for_full_probe =
                  truncated_newton_step.reduced_hessian_times_step;
              cheap_core_hessian_times_step_ptr =
                  &cheap_core_hessian_times_step_for_full_probe;
            }
            const auto hybrid_refinement =
                maybe_refine_exact_ctx_step_with_full_operator(
                    retraction_metric,
                    current_space,
                    current_projection,
                    trust_radius,
                    hybrid_refinement_max_cg_iterations,
                    options_.gradient_tolerance,
                    retry_with_full_hvp_operator.get(),
                    &transported_preconditioner,
                    truncated_newton_step,
                    cheap_core_hessian_times_step_ptr);
            if (hybrid_refinement.used_full_model_probe) {
              full_model_probe_used = true;
              trial_step_for_current_trial = hybrid_refinement.step;
              clamp_nonredundant_step_result_to_retract_tangent_radius(
                  current_orbital_input,
                  current_space,
                  parameter_view,
                  current_projection,
                  trust_radius,
                  &trial_step_for_current_trial);
              if (truncated_newton_krylov_subspace_is_usable(
                      hybrid_refinement.step.krylov_subspace,
                      current_projection.reduced_gradient.size())) {
                cached_full_krylov_subspace =
                    hybrid_refinement.step.krylov_subspace;
              }
              reduced_step = trial_step_for_current_trial.reduced_step;
              predicted_decrease =
                  trial_step_for_current_trial.predicted_decrease;
              truncated_newton_step = trial_step_for_current_trial;
            }
            if (hybrid_refinement.used_refinement) {
              full_operator_step_refined = true;
              truncated_newton_step = hybrid_refinement.step;
              clamp_nonredundant_step_result_to_retract_tangent_radius(
                  current_orbital_input,
                  current_space,
                  parameter_view,
                  current_projection,
                  trust_radius,
                  &truncated_newton_step);
              reduced_step = truncated_newton_step.reduced_step;
              predicted_decrease = truncated_newton_step.predicted_decrease;
            }
          }

          Eigen::VectorXd packed_step(current_parameters.size());
          OrbitalObjective::TrialEvaluation accepted_trial_evaluation;
          Eigen::VectorXd trial_parameters(current_parameters.size());
          Eigen::VectorXd trial_gradient(current_gradient.size());
          double trial_energy = energy;
          double trust_ratio = 0.0;
          const bool screen_rejected_trials_with_energy_only =
              rejected_trial_step_count_for_current_point > 0 &&
              admit_energy_only_trial_screen();
          bool accepted_trial =
              try_truncated_newton_trial_step(
                  reduced_step,
                  predicted_decrease,
                  screen_rejected_trials_with_energy_only,
                  &cheap_trial_evaluation,
                  &packed_step,
                  &accepted_trial_evaluation,
                  &trial_parameters,
                  &trial_gradient,
                  &trial_energy,
                  &trust_ratio);
          cheap_trial_rejected = !accepted_trial;
          if (!accepted_trial &&
              retry_with_full_hvp_operator != nullptr &&
              !full_operator_step_refined) {
            full_retry_admitted =
                assess_nonredundant_truncated_newton_full_retry_after_cheap_reject(
                    trial_step_for_current_trial,
                    cheap_predicted_decrease_fallback,
                    cheap_trial_evaluation,
                    model_quality_full_retry_requested ||
                        exact_ctx_hybrid_strategy_state.request_followup);
            if (full_retry_admitted) {
              full_retry_attempted = true;
              full_model_probe_used = true;
              auto retry_step =
                  solve_trust_region_in_krylov_subspace(
                      current_projection,
                      trust_radius,
                      cached_full_krylov_subspace);
              reused_full_krylov_subspace =
                  truncated_newton_step_is_usable(
                      retry_step,
                      current_projection.reduced_gradient);
              if (!reused_full_krylov_subspace) {
                cached_full_krylov_subspace = TruncatedNewtonKrylovSubspace();
                TruncatedNewtonStepResult full_retry_seed_step =
                    truncated_newton_step;
                if (cheap_exact_ctx_hvp_operator != nullptr &&
                    cheap_exact_ctx_hvp_operator
                        ->uses_sr1_outer_response_approximation()) {
                  full_retry_seed_step.reduced_hessian_times_step.resize(0);
                }
                const FullRetryWarmStart full_retry_warm_start =
                    build_nonredundant_truncated_newton_full_retry_warm_start(
                        current_projection,
                        trust_radius,
                        rejected_step_cache,
                        full_retry_seed_step,
                        retry_with_full_hvp_operator.get());
                retry_step =
                    solve_nonredundant_truncated_newton_step(
                        retraction_metric,
                        current_space,
                        current_projection,
                        trust_radius,
                        max_cg_iterations,
                        options_.gradient_tolerance,
                        retry_with_full_hvp_operator.get(),
                        &transported_preconditioner,
                        full_retry_warm_start.initial_reduced_step_ptr(
                            reduced_size),
                        full_retry_warm_start.initial_hessian_times_step_ptr(
                            reduced_size));
              }
              clamp_nonredundant_step_result_to_retract_tangent_radius(
                  current_orbital_input,
                  current_space,
                  parameter_view,
                  current_projection,
                  trust_radius,
                  &retry_step);
              if (truncated_newton_krylov_subspace_is_usable(
                      retry_step.krylov_subspace,
                      current_projection.reduced_gradient.size())) {
                cached_full_krylov_subspace =
                    retry_step.krylov_subspace;
              }
              reduced_step = retry_step.reduced_step;
              if (reduced_step.size() != current_projection.reduced_gradient.size()) {
                result.termination_reason =
                    "nonredundant_truncated_newton_invalid_step_dimension";
                final_gradient_l2_norm = current_projection.reduced_gradient.norm();
                break;
              }
              predicted_decrease = retry_step.predicted_decrease;
              if (!std::isfinite(predicted_decrease) ||
                  predicted_decrease <= 0.0) {
                reduced_step =
                    build_nonredundant_preconditioned_reduced_gradient_step(
                        retraction_metric,
                        current_space,
                        current_projection,
                        trust_radius,
                        &transported_preconditioner);
                predicted_decrease =
                    estimate_nonredundant_reduced_model_decrease(
                        current_projection,
                        reduced_step,
                        retry_with_full_hvp_operator.get());
                retry_step.reduced_step = reduced_step;
                retry_step.reduced_hessian_times_step.resize(0);
                retry_step.retract_tangent_norm =
                    compute_nonredundant_retract_tangent_norm(
                        current_orbital_input,
                        current_space,
                        parameter_view,
                        reduced_step);
                retry_step.reached_boundary =
                    retry_step.retract_tangent_norm >=
                    (1.0 - 1.0e-8) * trust_radius;
                retry_step.predicted_decrease = predicted_decrease;
              }
              truncated_newton_step = std::move(retry_step);
              accepted_trial =
                  try_truncated_newton_trial_step(
                      reduced_step,
                      predicted_decrease,
                      screen_rejected_trials_with_energy_only,
                      nullptr,
                      &packed_step,
                      &accepted_trial_evaluation,
                      &trial_parameters,
                      &trial_gradient,
                      &trial_energy,
                      &trust_ratio);
            }
          }
          if (!accepted_trial &&
              rejected_trial_step_count_for_current_point == 0 &&
              reduced_gradient_inf_norm >=
                  8.0 * options_.gradient_tolerance &&
              (cheap_model_step.encountered_negative_curvature ||
               full_retry_attempted)) {
            if (try_nonredundant_descent_fallback_step(
                    &packed_step,
                    &objective,
                    &trial_parameters,
                    &trial_gradient,
                    &trial_energy)) {
              accepted_trial = true;
              trust_ratio = kAcceptRatio;
              truncated_newton_step.used_krylov_rescue = true;
              truncated_newton_step.reached_boundary = false;
              truncated_newton_step.encountered_negative_curvature = false;
              truncated_newton_step.cg_iterations = 0;
              truncated_newton_step.reduced_step =
                  current_space.project_vector(packed_step).reduced_gradient;
              truncated_newton_step.retract_tangent_norm = packed_step.norm();
              truncated_newton_step.predicted_decrease =
                  std::max(options_.energy_tolerance, energy - trial_energy);
            }
          }
          if (!accepted_trial) {
            update_exact_ctx_hybrid_cost_sample();
            log_exact_ctx_hvp_diagnostics();
            ++rejected_trial_step_count_for_current_point;
            if (cheap_trial_rejected) {
              hvp_model_quality_state.observe_rejected_cheap_trial();
            }
            exact_ctx_hybrid_strategy_state.request_followup =
                (retry_with_full_hvp_operator != nullptr);
            maybe_log_exact_ctx_policy_outcome(
                n_iterations,
                false,
                trust_ratio,
                trust_radius,
                cheap_trial_rejected,
                full_retry_attempted,
                full_operator_step_refined,
                transport_history_size,
                static_cast<int>(packed_secant_history.size()),
                transported_preconditioner.size(),
                reused_cheap_krylov_subspace,
                reused_full_krylov_subspace,
                transported_warm_start_admitted,
                truncated_newton_step.used_initial_step,
                truncated_newton_step.warm_start_hvp_performed,
                truncated_newton_step.cg_iterations,
                false,
                consecutive_projected_stall_count,
                exact_ctx_hybrid_strategy_state.request_followup,
                truncated_newton_step.used_krylov_rescue,
                truncated_newton_step.reached_boundary,
                truncated_newton_step.encountered_negative_curvature,
                truncated_newton_step_effective_norm(truncated_newton_step),
                truncated_newton_step.predicted_decrease);
            trust_radius *= kRejectShrink;
            if (trust_radius <= options_.minimum_step_size) {
              result.termination_reason =
                  "nonredundant_truncated_newton_trust_radius_exhausted";
              final_gradient_l2_norm = current_projection.reduced_gradient.norm();
              break;
            }
            if (sr1_outer_response_model.size() > 0) {
              // A rejected trial is direct evidence that the accepted-point
              // SR1 missing-curvature model is stale or over-aggressive. Clear
              // both the model and same-point cheap caches so the retry uses
              // the raw core HVP rather than recycling a poisoned subspace.
              sr1_outer_response_model.clear();
              cached_cheap_krylov_subspace = TruncatedNewtonKrylovSubspace();
              rejected_step_cache.clear();
            } else {
              rejected_step_cache.update(
                  current_orbital_input,
                  current_space,
                  parameter_view,
                  cheap_model_step,
                  truncated_newton_step,
                  reduced_size,
                  trust_radius);
            }
            continue;
          }

          bool sr1_outer_pair_candidate_ready = false;
          Eigen::VectorXd sr1_outer_reduced_step;
          Eigen::VectorXd sr1_outer_reduced_gradient_change;
          Eigen::VectorXd sr1_outer_core_hessian_times_step;
          if (sr1_outer_response_model.enabled() &&
              options_.nonredundant_truncated_newton_hvp_mode ==
                  NonredundantTruncatedNewtonHvpMode::ExactContextDirectAction &&
              cheap_exact_ctx_hvp_operator != nullptr) {
            sr1_outer_reduced_step =
                current_space.project_vector(packed_step).reduced_gradient;
            const Eigen::VectorXd packed_gradient_change =
                trial_gradient - current_gradient;
            sr1_outer_reduced_gradient_change =
                current_space.project_reduced_gradient(packed_gradient_change);
            if (finite_nonzero_vector_matches_size(
                    sr1_outer_reduced_step,
                    current_projection.reduced_gradient.size()) &&
                finite_vector_matches_size(
                    sr1_outer_reduced_gradient_change,
                    current_projection.reduced_gradient.size())) {
              sr1_outer_core_hessian_times_step =
                  cheap_exact_ctx_hvp_operator->apply_core_only(
                      sr1_outer_reduced_step);
              sr1_outer_pair_candidate_ready =
                  finite_vector_matches_size(
                      sr1_outer_core_hessian_times_step,
                      current_projection.reduced_gradient.size());
            }
          }

          update_exact_ctx_hybrid_cost_sample();
          log_exact_ctx_hvp_diagnostics();
          if (accepted_trial_evaluation.valid) {
            objective.commit_trial_evaluation(
                std::move(accepted_trial_evaluation));
          }
          current_parameters = trial_parameters;
          current_gradient = std::move(trial_gradient);
          energy = trial_energy;
          const bool accepted_point_chart_reset =
              objective.canonicalize_orbital_chart_at_current_point(
                  &current_parameters,
                  &current_gradient,
                  &packed_secant_history);
          ++n_iterations;
          rejected_trial_step_count_for_current_point = 0;
          previous_accepted_packed_step =
              accepted_point_chart_reset ? Eigen::VectorXd() : packed_step;
          rejected_step_cache.clear();
          cached_cheap_krylov_subspace = TruncatedNewtonKrylovSubspace();
          cached_full_krylov_subspace = TruncatedNewtonKrylovSubspace();
          if (accepted_point_chart_reset) {
            sr1_outer_response_model.clear();
            previous_accepted_iteration_reliable_for_transport = false;
            consecutive_projected_stall_count = 0;
            exact_ctx_hybrid_strategy_state.request_followup = false;
            exact_ctx_hybrid_strategy_state.hybrid_followup_cooldown_remaining = 0;
            exact_ctx_hybrid_strategy_state.tail_full_solve_cooldown_remaining = 0;
            hvp_model_quality_state.reset_after_chart_change();
          } else if (trust_ratio <
                         exact_ctx_sr1_outer_response_clear_trust_ratio()) {
            sr1_outer_response_model.clear();
          } else if (sr1_outer_pair_candidate_ready) {
            sr1_outer_response_model.try_append_pair(
                current_space,
                packed_step,
                sr1_outer_reduced_step,
                sr1_outer_reduced_gradient_change,
                sr1_outer_core_hessian_times_step);
          }
          sync_result_from_objective(objective, &result);
          record_accepted_iteration_snapshot(&objective, n_iterations, options_, &result);
          final_gradient_l2_norm = current_gradient.norm();
          const double de = energy - previous_energy;
          previous_energy = energy;
          NonredundantOrbitalSpace next_space =
              build_nonredundant_space(objective, parameter_view);
          auto next_projection =
              next_space.project_gradient(current_gradient);
          const bool nonredundant_rank_changed =
              current_space.reduced_size() != next_space.reduced_size() ||
              current_space.rank_signature() != next_space.rank_signature();
          final_projected_gradient_inf_norm =
              gradient_infinity_norm(next_projection.reduced_gradient);
          final_projected_gradient_l2_norm =
              next_projection.reduced_gradient.norm();
          const AcceptedTruncatedNewtonStepControl accepted_step_control =
              assess_accepted_nonredundant_truncated_newton_step(
                  retry_with_full_hvp_operator != nullptr,
                  system_profile.sparse_orbital_chart,
                  trust_radius,
                  options_.minimum_step_size,
                  max_trust_radius,
                  kRejectShrink,
                  kExpandRatio,
                  kBoundaryFraction,
                  reduced_gradient_inf_norm,
                  final_projected_gradient_inf_norm,
                  options_.gradient_tolerance,
                  consecutive_projected_stall_count,
                  exact_ctx_hybrid_strategy_state.hybrid_followup_cooldown_remaining,
                  trust_ratio,
                  cheap_trial_rejected,
                  full_retry_attempted,
                  full_operator_step_refined,
                  full_model_probe_used,
                  truncated_newton_step);
          if (nonredundant_rank_changed) {
            packed_secant_history.clear();
            previous_accepted_packed_step = Eigen::VectorXd();
            sr1_outer_response_model.clear();
            previous_accepted_iteration_reliable_for_transport = false;
            consecutive_projected_stall_count = 0;
            hvp_model_quality_state.reset_after_chart_change();
          }
          if (!accepted_point_chart_reset && !nonredundant_rank_changed) {
            hvp_model_quality_state.observe_accepted_step(
                trust_ratio,
                cheap_trial_rejected,
                full_model_probe_used,
                accepted_step_control.stalled_projected_convergence,
                truncated_newton_step.used_krylov_rescue);
            hvp_model_quality_state.decay_request_counters();
          }
          if (!accepted_point_chart_reset && !nonredundant_rank_changed) {
            const Eigen::VectorXd packed_projected_gradient_change =
                next_projection.packed_projected_gradient -
                current_projection.packed_projected_gradient;
            append_nonredundant_truncated_newton_secant_pair(
                packed_step,
                packed_projected_gradient_change,
                transport_history_size,
                &packed_secant_history);
          }
          trust_radius = accepted_step_control.trust_radius;
          consecutive_projected_stall_count =
              accepted_step_control.consecutive_projected_stall_count;
          exact_ctx_hybrid_strategy_state.hybrid_followup_cooldown_remaining =
              accepted_step_control.hybrid_followup_cooldown_remaining;
          const int tail_full_solve_cooldown_after_decay =
              std::max(
                  0,
                  exact_ctx_hybrid_strategy_state
                          .tail_full_solve_cooldown_remaining -
                      1);
          exact_ctx_hybrid_strategy_state.tail_full_solve_cooldown_remaining =
              exact_ctx_inner_solve_policy.used_stall_tail_full_solve
                  ? exact_ctx_tail_full_inner_solve_cooldown_accepted_iterations()
                  : tail_full_solve_cooldown_after_decay;
          previous_accepted_iteration_reliable_for_transport =
              accepted_step_control.previous_iteration_reliable_for_transport;
          exact_ctx_hybrid_strategy_state.request_followup =
              accepted_step_control.enable_hybrid_bridge_next_iteration;
          maybe_log_exact_ctx_policy_outcome(
              n_iterations,
              true,
              trust_ratio,
              trust_radius,
              cheap_trial_rejected,
              full_retry_attempted,
              full_operator_step_refined,
              transport_history_size,
              static_cast<int>(packed_secant_history.size()),
              transported_preconditioner.size(),
              reused_cheap_krylov_subspace,
              reused_full_krylov_subspace,
              transported_warm_start_admitted,
              truncated_newton_step.used_initial_step,
              truncated_newton_step.warm_start_hvp_performed,
              truncated_newton_step.cg_iterations,
              accepted_step_control.stalled_projected_convergence,
	              accepted_step_control.consecutive_projected_stall_count,
	              exact_ctx_hybrid_strategy_state.request_followup,
	              truncated_newton_step.used_krylov_rescue,
	              truncated_newton_step.reached_boundary,
	              truncated_newton_step.encountered_negative_curvature,
	              truncated_newton_step_effective_norm(truncated_newton_step),
	              truncated_newton_step.predicted_decrease);
          if (std::abs(de) < options_.energy_tolerance &&
              gradient_infinity_norm(next_projection.reduced_gradient) <
                  options_.gradient_tolerance) {
            result.converged = true;
            result.termination_reason =
                "nonredundant_truncated_newton_dual_tolerance";
            final_gradient_l2_norm = next_projection.reduced_gradient.norm();
            const double full_gradient_inf_norm =
                gradient_infinity_norm(current_gradient);
            if (full_gradient_inf_norm > options_.gradient_tolerance) {
              const NonredundantFullSpacePolishResult polish_result =
                  run_nonredundant_full_space_polish(
                      &objective,
                      options_,
                      "nonredundant_truncated_newton_polish_dual_tolerance",
                      "nonredundant_truncated_newton_polish_budget",
                      &current_parameters,
                      &current_gradient,
                      &energy,
                      &previous_energy,
                      &n_iterations,
                      &result,
                      &final_gradient_l2_norm);
              if (polish_result.accepted_iterations > 0) {
                reset_tn_state_after_full_space_update();
                next_space = build_nonredundant_space(objective, parameter_view);
                next_projection = next_space.project_gradient(current_gradient);
                final_projected_gradient_inf_norm =
                    gradient_infinity_norm(next_projection.reduced_gradient);
                final_projected_gradient_l2_norm =
                    next_projection.reduced_gradient.norm();
                final_gradient_l2_norm = current_gradient.norm();
                if (!polish_result.reached_dual_tolerance) {
                  result.converged = false;
                  result.termination_reason.clear();
                  current_space = std::move(next_space);
                  current_projection = std::move(next_projection);
                  continue;
                }
              } else if (!polish_result.reached_dual_tolerance) {
                result.converged = false;
                result.termination_reason =
                    "nonredundant_truncated_newton_polish_failed";
              }
            }
            break;
          }

          current_space = std::move(next_space);
          current_projection = std::move(next_projection);
        }
        result.final_projected_gradient_inf_norm =
            final_projected_gradient_inf_norm;
        result.final_projected_gradient_l2_norm =
            final_projected_gradient_l2_norm;
        final_projected_gradient_ready = true;
        break;
      }

      case CppVbScfOptimizerBackend::DeepVBHOnnx:
        result.termination_reason =
            "deepvbh_onnx_requires_hybrid_optimizer";
        break;
    }
  } catch (const std::exception& error) {
    result.termination_reason = error.what();
  }

  if (result.total_energy_history.empty()) {
    if (!result.termination_reason.empty()) {
      throw std::runtime_error(
          std::string("optimizer did not evaluate the objective: ") +
          result.termination_reason);
    }
    throw std::runtime_error("optimizer did not evaluate the objective");
  }

  result.n_iterations = n_iterations;
  result.final_total_energy = result.total_energy_history.back();
  result.final_one_electron_reference_energy =
      result.scf_result.one_electron_reference_energy;
  result.final_gradient_inf_norm = result.gradient_inf_norm_history.back();
  result.final_gradient_l2_norm = final_gradient_l2_norm;
  if (optimizer_backend_uses_nonredundant_space(options_.backend) &&
      !final_projected_gradient_ready) {
    const NonredundantOrbitalSpace final_space =
        build_nonredundant_space(objective, parameter_view);
    const Eigen::VectorXd final_packed_gradient =
        parameter_view.gather_from_full(
            objective.last_gradient_result().sparse_orbital_energy_gradient);
    const auto final_projection =
        final_space.project_gradient(final_packed_gradient);
    result.final_projected_gradient_inf_norm =
        gradient_infinity_norm(final_projection.reduced_gradient);
    result.final_projected_gradient_l2_norm =
        final_projection.reduced_gradient.norm();
  }
  result.optimized_input = objective.last_input();
  Eigen::MatrixXd final_normalized_orbital_matrix =
      objective.last_gradient_result()
          .orbital_preparation_result
          .physical_orbital_frame
          .normalized_orbital_matrix;
  if (final_normalized_orbital_matrix.size() != 0) {
    // The evaluator always works with the normalized physical orbital frame,
    // mirroring legacy `normalize(...)`. Store that same frame in the final
    // sparse slots before exporting so Molden / restart artifacts see the
    // actual accepted physical orbitals rather than a pre-normalization raw
    // parameter vector.
    //
    // The OEO active representative reset is now treated as an export/gauge
    // choice by default rather than an accepted-point optimizer mutation. When
    // the accepted-point reset is disabled, rebuild the final physical active
    // representative here from the converged auxiliary block so Molden / restart
    // artifacts still use the localized occupied representative tied to the
    // initial reference.
    if (!oeo_active_representative_accepted_point_canonicalization_enabled()) {
      final_normalized_orbital_matrix =
          build_metric_preserving_oeo_repaired_normalized_orbital_matrix(
              result.optimized_input.orbital_preparation_input,
              objective.last_gradient_result().orbital_preparation_result,
              initial_normalized_orbital_matrix);
    }
    overwrite_sparse_orbitals_from_dense_physical_frame(
        final_normalized_orbital_matrix,
        &result.optimized_input.orbital_preparation_input);
  }
  enforce_strict_sparse_orbital_support(
      &result.optimized_input.orbital_preparation_input);

  if (result.termination_reason.empty()) {
    if (result.n_iterations >= options_.max_iterations) {
      result.termination_reason = "max_iterations";
    } else {
      result.termination_reason = "stopped";
    }
  }

  const auto optimization_end_time = std::chrono::steady_clock::now();
  const std::chrono::duration<double> total_elapsed_seconds =
      optimization_end_time - optimization_start_time;
  result.total_wall_time_seconds = total_elapsed_seconds.count();

  return result;
}

}  // namespace xmvb::vb
