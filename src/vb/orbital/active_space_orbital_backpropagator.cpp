#include "vb/orbital/active_space_orbital_backpropagator.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xmvb::vb {

namespace {

constexpr int kLegacyOrbitalTypeOeo = 3;

std::vector<double> compute_sparse_orbital_squared_norms(
    const OrbitalPreparationInput& input,
    const Eigen::Map<const Eigen::MatrixXd>& basis_overlap_matrix) {
  std::vector<double> squared_norms(input.n_orbitals, 0.0);

#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t orbital_offset = 0;
       orbital_offset < static_cast<std::ptrdiff_t>(input.n_orbitals);
       ++orbital_offset) {
    const std::size_t orbital_index = orbital_offset;
    const int coefficient_count =
        stored_sparse_orbital_coefficient_count(input, orbital_index);
    double squared_norm = 0.0;
    for (int left_index = 0; left_index < coefficient_count; ++left_index) {
      const int left_basis_function =
          input.orbital_basis_index_table[orbital_index *
                                              input.n_basis_functions +
                                          left_index] -
          1;
      if (left_basis_function < 0 || left_basis_function >= input.n_basis_functions) {
        throw std::runtime_error("invalid sparse orbital basis index while computing norms");
      }
      const double left_value =
          input.orbital_value_table[orbital_index * input.n_basis_functions +
                                    left_index];
      for (int right_index = 0; right_index < coefficient_count; ++right_index) {
        const int right_basis_function =
            input.orbital_basis_index_table[orbital_index *
                                                input.n_basis_functions +
                                            right_index] -
            1;
        if (right_basis_function < 0 || right_basis_function >= input.n_basis_functions) {
          throw std::runtime_error("invalid sparse orbital basis index while computing norms");
        }
        const double right_value =
            input.orbital_value_table[orbital_index * input.n_basis_functions +
                                      right_index];
        squared_norm +=
            left_value * right_value *
            basis_overlap_matrix(left_basis_function, right_basis_function);
      }
    }

    if (!std::isfinite(squared_norm) ||
        squared_norm <= std::numeric_limits<double>::epsilon()) {
      throw std::runtime_error("orbital normalization failed during cached backpropagation");
    }
    squared_norms[orbital_index] = squared_norm;
  }

  return squared_norms;
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

Eigen::MatrixXd build_identity_metric_inactive_projector_pullback_gradient(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_gradient_symmetric,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap_times_inactive) {
  if (inactive_density_gradient_symmetric.rows() != inactive_orbitals.rows() ||
      inactive_density_gradient_symmetric.cols() != inactive_orbitals.rows() ||
      basis_overlap_times_inactive.rows() != inactive_orbitals.rows() ||
      basis_overlap_times_inactive.cols() != inactive_orbitals.cols()) {
    throw std::invalid_argument(
        "identity-metric inactive pullback gradient has inconsistent dimensions");
  }
  if (inactive_orbitals.cols() == 0) {
    return Eigen::MatrixXd::Zero(
        inactive_orbitals.rows(),
        inactive_orbitals.cols());
  }

  // The accepted point may already satisfy `C_i^T S C_i = I` after the
  // orthonormal-inactive gauge rewrite, while the pullback is still taken with
  // respect to the raw inactive coefficients. In that chart the exact accepted
  // pullback simplifies to `G_sym C_i - S C_i (C_i^T G_sym C_i)` without any
  // inactive inverse construction.
  const Eigen::MatrixXd reduced_metric_gradient =
      inactive_orbitals.transpose() *
      inactive_density_gradient_symmetric *
      inactive_orbitals;
  return inactive_density_gradient_symmetric * inactive_orbitals -
      basis_overlap_times_inactive * reduced_metric_gradient;
}

void validate_orbital_preparation_result(
    const OrbitalPreparationInput& input,
    const OrbitalPreparationResult& orbital_preparation_result,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) {
  const std::size_t ao_matrix_size =
      input.n_basis_functions * input.n_basis_functions;
  const std::size_t normalized_matrix_size =
      input.n_basis_functions * input.n_orbitals;
  const std::size_t projected_active_size =
      input.n_basis_functions * static_cast<std::size_t>(n_active_orbitals);
  const std::size_t inactive_active_size =
      static_cast<std::size_t>(n_inactive_doubly_occupied_orbitals) *
      static_cast<std::size_t>(n_active_orbitals);

  if (orbital_preparation_result.occupied_space_projector.size() != ao_matrix_size ||
      orbital_preparation_result.inactive_auxiliary_transform.size() != ao_matrix_size ||
      static_cast<std::size_t>(
          orbital_preparation_result.physical_orbital_frame.normalized_orbital_matrix.size()) !=
          normalized_matrix_size ||
      orbital_preparation_result.projected_active_overlap_matrix.size() !=
          projected_active_size) {
    throw std::invalid_argument(
        "orbital preparation cache dimensions do not match orbital input");
  }
  if (n_inactive_doubly_occupied_orbitals > 0 &&
      orbital_preparation_result.inactive_active_overlap_matrix.size() !=
          inactive_active_size) {
    throw std::invalid_argument(
        "inactive-active overlap cache dimensions do not match orbital input");
  }
}

ActiveSpaceOrbitalBackpropagationResult scatter_dense_orbital_gradient_to_sparse_slots(
    const Eigen::Ref<const Eigen::MatrixXd>& original_orbital_gradient,
    const OrbitalPreparationInput& input) {
  if (original_orbital_gradient.rows() != input.n_basis_functions ||
      original_orbital_gradient.cols() != input.n_orbitals) {
    throw std::invalid_argument("original orbital gradient shape mismatch");
  }

  std::vector<double> orbital_value_gradient(input.orbital_value_table.size(), 0.0);
  for (std::size_t orbital_index = 0; orbital_index < input.n_orbitals; ++orbital_index) {
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
        throw std::runtime_error(
            "invalid sparse orbital basis index while scattering dense orbital gradient");
      }
      orbital_value_gradient[orbital_index * input.n_basis_functions +
                             coefficient_index] =
          original_orbital_gradient(basis_function_index, orbital_index);
    }
  }

  ActiveSpaceOrbitalBackpropagationResult result;
  result.orbital_value_gradient = std::move(orbital_value_gradient);
  return result;
}

ActiveSpaceOrbitalBackpropagationResult backpropagate_normalization_to_raw_slots(
    const Eigen::Ref<const Eigen::MatrixXd>& original_orbital_gradient,
    const OrbitalPreparationInput& input,
    const Eigen::Map<const Eigen::MatrixXd>& basis_overlap_matrix,
    const Eigen::Ref<const Eigen::MatrixXd>& normalized_orbital_matrix,
    const std::vector<double>& squared_norms) {
  if (original_orbital_gradient.rows() != input.n_basis_functions ||
      original_orbital_gradient.cols() != input.n_orbitals) {
    throw std::invalid_argument("original orbital gradient shape mismatch");
  }
  if (normalized_orbital_matrix.rows() != input.n_basis_functions ||
      normalized_orbital_matrix.cols() != input.n_orbitals) {
    throw std::invalid_argument("normalized orbital matrix shape mismatch");
  }
  if (squared_norms.size() != input.n_orbitals) {
    throw std::invalid_argument("squared orbital norms size mismatch");
  }

  std::vector<double> orbital_value_gradient(input.orbital_value_table.size(), 0.0);

#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t orbital_offset = 0;
       orbital_offset < static_cast<std::ptrdiff_t>(input.n_orbitals);
       ++orbital_offset) {
    const std::size_t orbital_index = orbital_offset;
    const int coefficient_count =
        stored_sparse_orbital_coefficient_count(input, orbital_index);
    const std::size_t n_coefficients = coefficient_count;
    std::vector<int> basis_function_indices(n_coefficients, 0);
    Eigen::VectorXd normalized_vector = Eigen::VectorXd::Zero(coefficient_count);
    Eigen::VectorXd dense_gradient = Eigen::VectorXd::Zero(coefficient_count);

    for (int coefficient_index = 0; coefficient_index < coefficient_count; ++coefficient_index) {
      const int basis_function_index =
          input.orbital_basis_index_table[orbital_index *
                                              input.n_basis_functions +
                                          coefficient_index] -
          1;
      if (basis_function_index < 0 || basis_function_index >= input.n_basis_functions) {
        throw std::runtime_error(
            "invalid sparse orbital basis index while pulling gradients back");
      }
      basis_function_indices[coefficient_index] = basis_function_index;
      normalized_vector(coefficient_index) =
          normalized_orbital_matrix(basis_function_index, orbital_index);
      dense_gradient(coefficient_index) =
          original_orbital_gradient(basis_function_index, orbital_index);
    }

    Eigen::MatrixXd overlap_submatrix =
        Eigen::MatrixXd::Zero(coefficient_count, coefficient_count);
    for (int row = 0; row < coefficient_count; ++row) {
      for (int column = 0; column < coefficient_count; ++column) {
        overlap_submatrix(row, column) =
                basis_overlap_matrix(
                basis_function_indices[row],
                basis_function_indices[column]);
      }
    }

    const double normalization_factor =
        std::sqrt(1.0 / squared_norms[orbital_index]);
    const double scalar_term = dense_gradient.dot(normalized_vector);
    const Eigen::VectorXd raw_gradient =
        normalization_factor * dense_gradient -
        normalization_factor * scalar_term * (overlap_submatrix * normalized_vector);

    for (int coefficient_index = 0; coefficient_index < coefficient_count; ++coefficient_index) {
      orbital_value_gradient[orbital_index * input.n_basis_functions +
                             coefficient_index] = raw_gradient(coefficient_index);
    }
  }

  ActiveSpaceOrbitalBackpropagationResult result;
  result.orbital_value_gradient = std::move(orbital_value_gradient);
  return result;
}

}  // namespace

ActiveSpaceOrbitalBackpropagationDiagnostics
ActiveSpaceOrbitalBackpropagator::compute_diagnostics(
    const Eigen::Ref<const Eigen::MatrixXd>& active_auxiliary_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_gradient,
    const OrbitalPreparationInput& input,
    const OrbitalPreparationResult& orbital_preparation_result) const {
  if (input.n_basis_functions <= 0 || input.n_orbitals <= 0 || input.n_active_orbitals <= 0) {
    throw std::invalid_argument("orbital preparation input dimensions must be positive");
  }

  const int n_inactive_doubly_occupied_orbitals =
      (input.n_total_electrons - input.n_active_electrons) / 2;
  if (n_inactive_doubly_occupied_orbitals < 0 ||
      n_inactive_doubly_occupied_orbitals + input.n_active_orbitals > input.n_orbitals) {
    throw std::invalid_argument("invalid occupied-space partition in cached orbital backprop");
  }
  if (active_auxiliary_gradient.rows() != input.n_basis_functions ||
      active_auxiliary_gradient.cols() != input.n_active_orbitals) {
    throw std::invalid_argument("active auxiliary gradient shape mismatch");
  }
  if (inactive_density_gradient.rows() != input.n_basis_functions ||
      inactive_density_gradient.cols() != input.n_basis_functions) {
    throw std::invalid_argument("inactive density gradient shape mismatch");
  }

  validate_orbital_preparation_result(
      input,
      orbital_preparation_result,
      n_inactive_doubly_occupied_orbitals,
      input.n_active_orbitals);

  const Eigen::Map<const Eigen::MatrixXd> basis_overlap_matrix(
      input.ao_overlap_matrix.data(),
      input.n_basis_functions,
      input.n_basis_functions);
  const Eigen::Ref<const Eigen::MatrixXd> normalized_orbital_matrix =
      orbital_preparation_result.physical_orbital_frame.normalized_orbital_matrix;

  Eigen::MatrixXd original_orbital_gradient =
      Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_orbitals);
  if (n_inactive_doubly_occupied_orbitals == 0) {
    original_orbital_gradient.middleCols(0, input.n_active_orbitals) =
        active_auxiliary_gradient;
  } else {
    const Eigen::Map<const Eigen::MatrixXd> occupied_space_projector(
        orbital_preparation_result.occupied_space_projector.data(),
        input.n_basis_functions,
        input.n_basis_functions);
    const Eigen::Map<const Eigen::MatrixXd> inactive_auxiliary_transform(
        orbital_preparation_result.inactive_auxiliary_transform.data(),
        input.n_basis_functions,
        input.n_basis_functions);
    const Eigen::Map<const Eigen::MatrixXd> inactive_active_overlap_matrix(
        orbital_preparation_result.inactive_active_overlap_matrix.data(),
        n_inactive_doubly_occupied_orbitals,
        input.n_active_orbitals);
    const Eigen::Map<const Eigen::MatrixXd> projected_active_overlap_matrix(
        orbital_preparation_result.projected_active_overlap_matrix.data(),
        input.n_basis_functions,
        input.n_active_orbitals);

    const auto inactive_orbitals =
        normalized_orbital_matrix.leftCols(n_inactive_doubly_occupied_orbitals);
    const auto inactive_auxiliary =
        inactive_auxiliary_transform.leftCols(n_inactive_doubly_occupied_orbitals);

    // The mixed-gauge preparer already built the occupied-space objects
    // `A1 = I - P11 S`, `A3 = C_i (C_i^T S C_i)^{-1}`, `A2 = A3^T S C_a`,
    // and `A4 = A1^T S C_a`. Reusing them lets the orbital pullback stay in
    // the inactive-dual / active-auxiliary gauge instead of rebuilding the full
    // inactive density and occupied projector from scratch.
    const Eigen::MatrixXd basis_overlap_times_inactive =
        basis_overlap_matrix * inactive_orbitals;
    const Eigen::MatrixXd basis_overlap_times_inactive_auxiliary =
        basis_overlap_matrix * inactive_auxiliary;
    const Eigen::MatrixXd inactive_overlap_inverse_unsym =
        inactive_auxiliary.transpose() * basis_overlap_times_inactive_auxiliary;
    const Eigen::MatrixXd inactive_overlap_inverse =
        0.5 *
        (inactive_overlap_inverse_unsym + inactive_overlap_inverse_unsym.transpose());
    const bool has_orthonormal_inactive_chart =
        matrix_is_effectively_identity(inactive_overlap_inverse);

    const Eigen::MatrixXd original_active_gradient =
        occupied_space_projector.transpose() * active_auxiliary_gradient;
    const Eigen::MatrixXd basis_overlap_times_active =
        projected_active_overlap_matrix +
        basis_overlap_times_inactive * inactive_active_overlap_matrix;
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
    if (has_orthonormal_inactive_chart) {
      original_inactive_gradient =
          build_identity_metric_inactive_projector_pullback_gradient(
              effective_inactive_density_gradient_symmetric,
              inactive_orbitals,
              basis_overlap_times_inactive);
    } else {
      const Eigen::MatrixXd inactive_overlap_inverse_gradient =
          inactive_orbitals.transpose() *
          effective_inactive_density_gradient *
          inactive_orbitals;
      const Eigen::MatrixXd inactive_overlap_gradient =
          -inactive_overlap_inverse *
          inactive_overlap_inverse_gradient *
          inactive_overlap_inverse;
      original_inactive_gradient =
          effective_inactive_density_gradient_symmetric * inactive_auxiliary +
          basis_overlap_times_inactive *
              (inactive_overlap_gradient + inactive_overlap_gradient.transpose());
    }

    original_orbital_gradient.leftCols(n_inactive_doubly_occupied_orbitals) =
        original_inactive_gradient;
    original_orbital_gradient.middleCols(
        n_inactive_doubly_occupied_orbitals,
        input.n_active_orbitals) = original_active_gradient;
  }

  // Legacy `orbtyp=oeo` normalizes the occupied physical orbitals inside
  // `Orbprep` and then accumulates `Grdbas(J, I)` directly on that normalized
  // coefficient chart. The extra per-orbital normalization pullback used by
  // the generic sparse-orbital path changes the OEO representative semantics
  // and is the main suspect for the open-shell active-orbital drift relative
  // to legacy XMVB. Preserve the legacy OEO chart here.
  if (input.orbital_type == kLegacyOrbitalTypeOeo) {
    ActiveSpaceOrbitalBackpropagationDiagnostics diagnostics;
    diagnostics.original_orbital_gradient = std::move(original_orbital_gradient);
    diagnostics.orbital_value_gradient =
        scatter_dense_orbital_gradient_to_sparse_slots(
            diagnostics.original_orbital_gradient,
            input)
            .orbital_value_gradient;
    return diagnostics;
  }

  const std::vector<double> squared_norms =
      compute_sparse_orbital_squared_norms(input, basis_overlap_matrix);
  ActiveSpaceOrbitalBackpropagationDiagnostics diagnostics;
  diagnostics.original_orbital_gradient = std::move(original_orbital_gradient);
  diagnostics.orbital_value_gradient =
      backpropagate_normalization_to_raw_slots(
          diagnostics.original_orbital_gradient,
          input,
          basis_overlap_matrix,
          normalized_orbital_matrix,
          squared_norms)
          .orbital_value_gradient;
  return diagnostics;
}

ActiveSpaceOrbitalBackpropagationResult ActiveSpaceOrbitalBackpropagator::backpropagate(
    const Eigen::Ref<const Eigen::MatrixXd>& active_auxiliary_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_gradient,
    const OrbitalPreparationInput& input,
    const OrbitalPreparationResult& orbital_preparation_result) const {
  const auto diagnostics =
      compute_diagnostics(
          active_auxiliary_gradient,
          inactive_density_gradient,
          input,
          orbital_preparation_result);
  ActiveSpaceOrbitalBackpropagationResult result;
  result.orbital_value_gradient = diagnostics.orbital_value_gradient;
  return result;
}

ActiveSpaceOrbitalBackpropagationResult ActiveSpaceOrbitalBackpropagator::backpropagate(
    const std::vector<double>& active_auxiliary_gradient,
    const std::vector<double>& inactive_density_gradient,
    const OrbitalPreparationInput& input,
    const OrbitalPreparationResult& orbital_preparation_result) const {
  const std::size_t active_auxiliary_size =
      input.n_basis_functions * input.n_active_orbitals;
  const std::size_t inactive_density_size =
      input.n_basis_functions * input.n_basis_functions;
  if (active_auxiliary_gradient.size() != active_auxiliary_size) {
    throw std::invalid_argument("active auxiliary gradient size mismatch");
  }
  if (inactive_density_gradient.size() != inactive_density_size) {
    throw std::invalid_argument("inactive density gradient size mismatch");
  }

  const Eigen::Map<const Eigen::MatrixXd> active_auxiliary_gradient_matrix(
      active_auxiliary_gradient.data(),
      input.n_basis_functions,
      input.n_active_orbitals);
  const Eigen::Map<const Eigen::MatrixXd> inactive_density_gradient_matrix(
      inactive_density_gradient.data(),
      input.n_basis_functions,
      input.n_basis_functions);
  return backpropagate(
      active_auxiliary_gradient_matrix,
      inactive_density_gradient_matrix,
      input,
      orbital_preparation_result);
}

}  // namespace xmvb::vb
