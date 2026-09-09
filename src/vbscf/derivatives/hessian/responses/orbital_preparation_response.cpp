#include "vbscf/derivatives/hessian/responses/orbital_preparation_response.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

namespace xmvb::vb {
namespace {

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

Eigen::MatrixXd build_inactive_auxiliary(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_overlap_inverse) {
  return inactive_orbitals * inactive_overlap_inverse;
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
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap_times_active_orbitals) {
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

  return delta_inactive_auxiliary * inactive_active_overlap +
      inactive_auxiliary * delta_inactive_active_overlap;
}

InactiveAuxiliaryDirectionResult build_delta_inactive_auxiliary_direction(
    const Eigen::Ref<const Eigen::MatrixXd>& basis_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_overlap_inverse) {
  InactiveAuxiliaryDirectionResult result;
  result.delta_inactive_auxiliary =
      Eigen::MatrixXd::Zero(
          inactive_orbitals.rows(),
          inactive_orbitals.cols());
  result.delta_inactive_overlap_inverse =
      Eigen::MatrixXd::Zero(
          inactive_overlap_inverse.rows(),
          inactive_overlap_inverse.cols());

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


}  // namespace

AcceptedOrbitalPreparationCache build_accepted_orbital_preparation_cache(
    const OrbitalPreparationInput& input,
    const Eigen::Ref<const Eigen::MatrixXd>& total_active_auxiliary_gradient,
    const std::vector<double>& total_inactive_density_gradient) {
  const int n_basis_functions = input.n_basis_functions;
  const int n_orbitals = input.n_orbitals;
  const int n_active_orbitals = input.n_active_orbitals;
  const int n_inactive_doubly_occupied_orbitals =
      (input.n_total_electrons - input.n_active_electrons) / 2;

  const Eigen::Map<const Eigen::MatrixXd> basis_overlap(
      input.ao_overlap_matrix.data(),
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
    const auto inactive_orbitals =
        cache.normalized_orbitals.leftCols(
            n_inactive_doubly_occupied_orbitals);
    const Eigen::MatrixXd inactive_overlap =
        inactive_orbitals.transpose() * basis_overlap * inactive_orbitals;
    cache.inactive_overlap_inverse =
        invert_self_adjoint_positive_definite(
            inactive_overlap,
            "cached_inactive_overlap");

    cache.inactive_auxiliary =
        build_inactive_auxiliary(
            inactive_orbitals,
            cache.inactive_overlap_inverse);
    cache.inactive_density =
        cache.inactive_auxiliary * inactive_orbitals.transpose();
  } else {
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
      {
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

namespace {

std::vector<double> backpropagate_normalization_to_raw_slots(
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

}  // namespace

std::vector<double> backpropagate_active_space_orbital_gradient(
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
    {
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

  return backpropagate_normalization_to_raw_slots(
      original_orbital_gradient,
      input,
      cache);
}

DenseOrbitalTangentContext build_dense_orbital_tangent_context(
    const OrbitalPreparationInput& input,
    const SparseParameterLayout& parameter_view,
    const Eigen::VectorXd& packed_direction,
    const AcceptedOrbitalPreparationCache& cache) {
  DenseOrbitalTangentContext result;
  result.normalized_orbitals = cache.normalized_orbitals;
  result.inverse_norms = cache.inverse_norms;
  result.delta_normalized_orbitals =
      Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_orbitals);
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

  return result;
}

OrbitalPreparationDirectionalResult build_orbital_preparation_directional_result(
    const OrbitalPreparationInput& input,
    const DenseOrbitalTangentContext& orbital_tangent_context,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals,
    const AcceptedOrbitalPreparationCache& cache) {
  if (input.n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }
  const Eigen::Map<const Eigen::MatrixXd> basis_overlap(
      input.ao_overlap_matrix.data(),
      input.n_basis_functions,
      input.n_basis_functions);

  LowRankAoMatrix delta_inactive_density_low_rank =
      make_empty_low_rank_ao_matrix(input.n_basis_functions);
  OrbitalPreparationDirectionalResult result;

  Eigen::MatrixXd delta_active_orbitals =
      Eigen::MatrixXd::Zero(input.n_basis_functions, n_active_orbitals);
  result.basis_overlap_times_delta_active_orbitals =
      Eigen::MatrixXd::Zero(input.n_basis_functions, n_active_orbitals);
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
            cache.inactive_overlap_inverse);
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
            basis_overlap_times_active_orbitals);
  } else {
    result.delta_active_auxiliary_orbitals = delta_active_orbitals;
  }

  result.delta_inactive_density =
      materialize_low_rank_ao_matrix(
          delta_inactive_density_low_rank,
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
  const int n_inactive_doubly_occupied_orbitals =
      (input.n_total_electrons - input.n_active_electrons) / 2;
  const std::size_t ao_matrix_size =
      input.n_basis_functions * input.n_basis_functions;
  const Eigen::Map<const Eigen::MatrixXd> basis_overlap(
      input.ao_overlap_matrix.data(),
      input.n_basis_functions,
      input.n_basis_functions);

  Eigen::MatrixXd delta_original_orbital_gradient =
      Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_orbitals);

  if (n_inactive_doubly_occupied_orbitals == 0) {
    // original_orbital_gradient is just active_auxiliary_gradient,
    // delta is zero since the gradient doesn't depend on the direction.
    // Nothing to do.
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
            cache.inactive_overlap_inverse);
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
    {
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


}  // namespace xmvb::vb
