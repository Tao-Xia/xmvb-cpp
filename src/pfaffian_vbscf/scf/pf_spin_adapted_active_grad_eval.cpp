#include "pfaffian_vbscf/scf/pf_spin_adapted_active_grad_eval.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "pfaffian_vbscf/kernel/pf_adjoint_kernel.hpp"
#include "pfaffian_vbscf/kernel/pf_closed_shell_ri_kernel.hpp"
#include "pfaffian_vbscf/kernel/pf_fixed_ms_open_shell_kernel.hpp"
#include "pfaffian_vbscf/kernel/pf_forward_kernel.hpp"
#include "pfaffian_vbscf/kernel/pf_high_spin_open_shell_hamiltonian.hpp"
#include "pfaffian_vbscf/math/antisymm_codec.hpp"
#include "pfaffian_vbscf/math/dense_utils.hpp"
#include "pfaffian_vbscf/matrices/pf_matrix_builder.hpp"
#include "pfaffian_vbscf/scf/pf_overlap_metric_gradient.hpp"
#include "pfaffian_vbscf/scf/pf_prepared_active_space.hpp"
#include "pfaffian_vbscf/matrices/pf_spin_adapted_matrix_projector.hpp"
#include "pfaffian_vbscf/types/pf_active_space_utils.hpp"
#include "vb/matrices/prepared_active_space_context.hpp"

namespace xmvb::pfaffian_vbscf {

namespace {

using Clock = std::chrono::steady_clock;

struct PairWeight {
  int row = 0;
  int col = 0;
  std::size_t cache_index = 0;
  double hamiltonian_weight = 0.0;
  double overlap_weight = 0.0;
};

struct RegularizedOverlapData {
  Matrix matrix;
  double shift = 0.0;
  Eigen::VectorXd min_eigenvector;
};

struct PairGradAccum {
  ScalarBuffer sso_grad;
  ScalarBuffer hho_grad;
  ScalarBuffer ggo_grad;
  int grad_pair_calls = 0;
  int grad_sample_calls = 0;
  double overlap_density_dt = 0.0;
  double one_electron_sigma_dt = 0.0;
  double two_electron_sigma_dt = 0.0;
};

double seconds_since(const Clock::time_point& start_time) {
  return std::chrono::duration<double>(Clock::now() - start_time).count();
}

bool is_effectively_zero(double value) {
  return std::abs(value) < 1.0e-15;
}

std::size_t lower_triangle_index(
    int row,
    int col) {
  return row * (row + 1) / 2 + col;
}

double compute_average_diagonal(
    const ScalarBuffer& matrix,
    int dimension) {
  if (dimension <= 0) {
    throw std::invalid_argument("dimension must be positive");
  }
  double diagonal_sum = 0.0;
  for (int index = 0; index < dimension; ++index) {
    diagonal_sum += matrix[index * dimension + index];
  }
  return diagonal_sum / static_cast<double>(dimension);
}

void accumulate_scaled_buffer(
    const ScalarBuffer& source,
    double scale,
    ScalarBuffer* destination) {
  if (destination == nullptr) {
    throw std::invalid_argument("destination must not be null");
  }
  if (source.empty() || scale == 0.0) {
    return;
  }
  if (destination->empty()) {
    destination->assign(source.size(), 0.0);
  } else if (destination->size() != source.size()) {
    throw std::invalid_argument("buffer size mismatch in accumulate_scaled_buffer");
  }
  for (std::size_t index = 0; index < source.size(); ++index) {
    (*destination)[index] += scale * source[index];
  }
}

void accumulate_scaled_matrix(
    const ConstMatrixRef& source,
    double scale,
    ScalarBuffer* destination) {
  if (destination == nullptr) {
    throw std::invalid_argument("destination must not be null");
  }
  if (scale == 0.0) {
    return;
  }

  const std::size_t expected_size =
      source.rows() * source.cols();
  if (destination->empty()) {
    destination->assign(expected_size, 0.0);
  } else if (destination->size() != expected_size) {
    throw std::invalid_argument("buffer size mismatch in accumulate_scaled_matrix");
  }

  for (Eigen::Index col = 0; col < source.cols(); ++col) {
    for (Eigen::Index row = 0; row < source.rows(); ++row) {
      (*destination)[col * source.rows() + row] +=
          scale * source(row, col);
    }
  }
}

PairGradAccum make_pair_grad_accum(
    std::size_t sso_size,
    std::size_t hho_size,
    std::size_t ggo_size,
    bool store_ggo_grad) {
  PairGradAccum accum;
  accum.sso_grad.assign(sso_size, 0.0);
  accum.hho_grad.assign(hho_size, 0.0);
  if (store_ggo_grad && ggo_size > 0) {
    accum.ggo_grad.assign(ggo_size, 0.0);
  }
  return accum;
}

/**
 * @brief Builds the analytic closed-shell overlap derivative with respect to
 * the spatial overlap matrix for one cached primitive Pfaffian state pair.
 *
 * The primitive pair cache stores the left spin-orbital matrix in bra
 * orientation, so we transpose it back before feeding it to the generic
 * overlap-metric gradient helper.
 */
ScalarBuffer build_closed_shell_overlap_metric_gradient(
    const PfKernelCache& cache,
    const ConstMatrixRef& spatial_overlap_matrix,
    const PfBasisData& primitive_basis) {
  if (cache.left.rows() != cache.n_spin_orbitals ||
      cache.left.cols() != cache.n_spin_orbitals ||
      cache.right.rows() != cache.n_spin_orbitals ||
      cache.right.cols() != cache.n_spin_orbitals) {
    throw std::invalid_argument(
        "closed-shell overlap-metric gradient requires full spin-orbital left/right matrices");
  }

  const Matrix left_pairing_matrix = cache.left.transpose().eval();
  return evaluate_spin_resolved_overlap_metric_gradient(
      left_pairing_matrix,
      cache.right,
      spatial_overlap_matrix,
      primitive_basis.n_alpha,
      primitive_basis.n_beta,
      primitive_basis.n_singlet_pairs);
}

void merge_pair_grad_accum(
    const PairGradAccum& local_accum,
    PfActiveGradResult* result,
    double* overlap_density_dt,
    double* one_electron_sigma_dt,
    double* two_electron_sigma_dt) {
  if (result == nullptr ||
      overlap_density_dt == nullptr ||
      one_electron_sigma_dt == nullptr ||
      two_electron_sigma_dt == nullptr) {
    throw std::invalid_argument("pair gradient merge outputs must not be null");
  }

  accumulate_scaled_buffer(local_accum.sso_grad, 1.0, &result->sso_grad);
  accumulate_scaled_buffer(local_accum.hho_grad, 1.0, &result->hho_grad);
  accumulate_scaled_buffer(local_accum.ggo_grad, 1.0, &result->ggo_grad);
  result->pair_grad_profile.grad_pair_calls += local_accum.grad_pair_calls;
  result->pair_grad_profile.grad_sample_calls += local_accum.grad_sample_calls;
  *overlap_density_dt += local_accum.overlap_density_dt;
  *one_electron_sigma_dt += local_accum.one_electron_sigma_dt;
  *two_electron_sigma_dt += local_accum.two_electron_sigma_dt;
}

Matrix dense_mat(
    const ScalarBuffer& data,
    int dimension,
    const char* label) {
  const std::size_t expected_size =
      dimension * dimension;
  if (data.size() != expected_size) {
    throw std::invalid_argument(
        std::string(label) + " size does not match the matrix dimension");
  }

  Matrix matrix = Matrix::Zero(dimension, dimension);
  for (int col = 0; col < dimension; ++col) {
    for (int row = 0; row < dimension; ++row) {
      matrix(row, col) = data[col * dimension + row];
    }
  }
  return matrix;
}

ScalarBuffer column_major_storage(const ConstMatrixRef& matrix) {
  ScalarBuffer data(matrix.rows() * matrix.cols(), 0.0);
  for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
      data[col * matrix.rows() + row] = matrix(row, col);
    }
  }
  return data;
}

Matrix symmetrize(const ConstMatrixRef& matrix) {
  return 0.5 * (matrix + matrix.transpose());
}

RegularizedOverlapData regularize_overlap(const ConstMatrixRef& overlap_matrix) {
  RegularizedOverlapData regularized;
  regularized.matrix = symmetrize(overlap_matrix);
  Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(regularized.matrix);
  if (eigensolver.info() != Eigen::Success) {
    throw std::runtime_error(
        "failed to diagonalize the spin-adapted Pfaffian overlap matrix");
  }

  Eigen::Index min_eigenvalue_index = 0;
  const double min_eigenvalue =
      eigensolver.eigenvalues().minCoeff(&min_eigenvalue_index);
  regularized.shift = std::max(0.0, 1.0e-8 - min_eigenvalue);
  if (regularized.shift > 0.0) {
    regularized.matrix.diagonal().array() += regularized.shift;
    regularized.min_eigenvector =
        eigensolver.eigenvectors().col(min_eigenvalue_index);
  }

  Eigen::LLT<Matrix> llt(regularized.matrix);
  if (llt.info() != Eigen::Success) {
    throw std::runtime_error("failed to regularize the spin-adapted overlap matrix");
  }
  return regularized;
}

Eigen::VectorXd ground_state_coefficients(
    const std::vector<double>& eigenvector_matrix,
    int dimension) {
  Eigen::VectorXd coefficients = Eigen::VectorXd::Zero(dimension);
  for (int index = 0; index < dimension; ++index) {
    coefficients[index] = eigenvector_matrix[index];
  }
  return coefficients;
}

std::vector<PairWeight> build_pair_weights(
    const ConstMatrixRef& hamiltonian_gradient,
    const ConstMatrixRef& overlap_gradient) {
  if (hamiltonian_gradient.rows() != hamiltonian_gradient.cols() ||
      overlap_gradient.rows() != overlap_gradient.cols() ||
      hamiltonian_gradient.rows() != overlap_gradient.rows()) {
    throw std::invalid_argument("pair-weight gradients must be square and aligned");
  }

  const int dimension = static_cast<int>(hamiltonian_gradient.rows());
  std::vector<PairWeight> weights;
  weights.reserve(dimension * (dimension + 1) / 2);
  for (int row = 0; row < dimension; ++row) {
    for (int col = 0; col <= row; ++col) {
      const double multiplicity = (row == col) ? 1.0 : 2.0;
      const double hamiltonian_weight =
          multiplicity * hamiltonian_gradient(row, col);
      const double overlap_weight =
          multiplicity * overlap_gradient(row, col);
      if (is_effectively_zero(hamiltonian_weight) &&
          is_effectively_zero(overlap_weight)) {
        continue;
      }
      weights.push_back({
          row,
          col,
          lower_triangle_index(row, col),
          hamiltonian_weight,
          overlap_weight,
      });
    }
  }
  return weights;
}

bool basis_contains_open_shell_states(const PfBasisData& basis) {
  for (const PfState& state : basis.states) {
    if (state.has_blocked_open_shell()) {
      return true;
    }
  }
  return basis.has_blocked_open_shell();
}

bool supports_high_spin_blocked_alpha_basis(const PfBasisData& basis) {
  if (!basis_contains_open_shell_states(basis)) {
    return false;
  }
  if (basis.n_blocked_beta != 0 || basis.n_beta > basis.n_alpha) {
    return false;
  }
  if (basis.n_blocked_alpha != basis.n_alpha - basis.n_beta) {
    return false;
  }
  for (const PfState& state : basis.states) {
    if (!state.blocked_beta_orbitals.empty() ||
        state.n_blocked_alpha() != basis.n_blocked_alpha ||
        state.n_singlet_pairs != basis.n_singlet_pairs) {
      return false;
    }
  }
  return true;
}

bool supports_general_fixed_ms_open_shell_basis(const PfBasisData& basis) {
  if (!basis_contains_open_shell_states(basis)) {
    return false;
  }
  if (basis.n_singlet_pairs < 0 ||
      basis.n_blocked_alpha < 0 ||
      basis.n_blocked_beta < 0) {
    return false;
  }
  if (basis.n_alpha != basis.n_singlet_pairs + basis.n_blocked_alpha ||
      basis.n_beta != basis.n_singlet_pairs + basis.n_blocked_beta) {
    return false;
  }
  for (const PfState& state : basis.states) {
    if (state.n_blocked_alpha() != basis.n_blocked_alpha ||
        state.n_blocked_beta() != basis.n_blocked_beta ||
        state.n_singlet_pairs != basis.n_singlet_pairs) {
      return false;
    }
  }
  return true;
}

}  // namespace

PfActiveGradResult PfSpinAdaptedActiveGradEval::eval(
    const xmvb::vb::CppVbInput& input,
    const PfSpinAdaptedBasisData& basis,
    double nuclear_repulsion_energy) const {
  const Clock::time_point total_start_time = Clock::now();
  const PfBasisData& primitive_basis = basis.primitive_basis;

  if (basis.n_states <= 0 || primitive_basis.n_states <= 0) {
    throw std::invalid_argument("basis dimensions must be positive");
  }
  const bool use_closed_shell_cache_path = !basis_contains_open_shell_states(primitive_basis);
  const bool use_high_spin_open_shell_path =
      !use_closed_shell_cache_path &&
      supports_high_spin_blocked_alpha_basis(primitive_basis);
  const bool use_general_fixed_ms_open_shell_path =
      !use_closed_shell_cache_path &&
      !use_high_spin_open_shell_path &&
      supports_general_fixed_ms_open_shell_basis(primitive_basis);
  if (!use_closed_shell_cache_path &&
      !use_high_spin_open_shell_path &&
      !use_general_fixed_ms_open_shell_path) {
    throw std::invalid_argument(
        "spin-adapted Pf active-space gradients require a primitive basis that is "
        "closed-shell or fixed-M_s open-shell with consistent blocked alpha/beta counts");
  }

  const auto timed_prepared_active_space =
      prepare_pf_timed_active_space_context(input);
  const auto& prepared_active_space =
      timed_prepared_active_space.prepared_active_space;

  PfActiveSpaceData active_space;
  active_space.n_active_orbitals =
      input.orbital_preparation_input.n_active_orbitals;
  active_space.n_alpha = primitive_basis.n_alpha;
  active_space.n_beta = primitive_basis.n_beta;
  active_space.sso.assign(
      prepared_active_space.orbital_result.active_orbital_overlap_matrix.begin(),
      prepared_active_space.orbital_result.active_orbital_overlap_matrix.end());
  active_space.hho.assign(
      prepared_active_space.active_space_one_electron_result.h1e_act.data(),
      prepared_active_space.active_space_one_electron_result.h1e_act.data() +
          prepared_active_space.active_space_one_electron_result.h1e_act.size());
  active_space.two_electron_representation =
      prepared_active_space.active_space_two_electron_result.representation;
  active_space.n_auxiliary_functions =
      prepared_active_space.active_space_two_electron_result.n_auxiliary_functions;
  active_space.ggo.assign(
      prepared_active_space.active_space_two_electron_result
          .packed_active_two_electron_integrals.begin(),
      prepared_active_space.active_space_two_electron_result
          .packed_active_two_electron_integrals.end());
  active_space.ri_active_pair_factors =
      flatten_column_major_matrix(
          prepared_active_space.active_space_two_electron_result
              .ri_active_pair_factors);
  if (!use_closed_shell_cache_path &&
      active_space.two_electron_representation ==
          xmvb::vb::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity) {
    throw std::invalid_argument(
        "Pf RI active-space preparation currently supports only the closed-shell path");
  }

  const Clock::time_point matrix_start_time = Clock::now();
  PfMatrixBuilder matrix_builder;
  PfMatrixBuildResult primitive_mats = matrix_builder.build(
      primitive_basis,
      active_space,
      use_closed_shell_cache_path);
  const double matrix_dt = seconds_since(matrix_start_time);
  if (use_closed_shell_cache_path &&
      primitive_mats.lower_triangle_pair_caches.size() !=
          primitive_basis.n_states *
              (primitive_basis.n_states + 1) / 2) {
    throw std::runtime_error(
        "PfMatrixBuilder did not return the expected number of pair caches");
  }

  PfMatrixBuildResult projected_mats =
      project_spin_adapted_matrices(
          primitive_mats,
          basis.primitive_to_adapted_coefficients);
  const RegularizedOverlapData regularized_overlap =
      regularize_overlap(dense_mat(projected_mats.s, basis.n_states, "pf_overlap_matrix"));
  Matrix overlap_matrix = regularized_overlap.matrix;
  Matrix hamiltonian_matrix =
      symmetrize(dense_mat(projected_mats.h, basis.n_states, "pf_hamiltonian_matrix"));

  xmvb::core::GeneralizedEigensolver eigensolver;
  const auto eigen_result = eigensolver.solve(
      column_major_storage(hamiltonian_matrix),
      column_major_storage(overlap_matrix),
      basis.n_states);
  if (eigen_result.eigenvalues.empty()) {
    throw std::runtime_error("generalized eigensolver returned no eigenvalues");
  }

  PfActiveGradResult result;
  result.matrix_dt = matrix_dt;
  result.scf_result.e_ref = prepared_active_space.one_electron_reference_energy;
  result.scf_result.evals = eigen_result.eigenvalues;
  result.scf_result.e_ele =
      result.scf_result.e_ref + eigen_result.eigenvalues.front();
  result.scf_result.e_tot =
      result.scf_result.e_ele + nuclear_repulsion_energy;
  result.scf_result.avg_diag_s =
      compute_average_diagonal(column_major_storage(overlap_matrix), basis.n_states);
  result.scf_result.matrix_dt = matrix_dt;
  result.scf_result.total_dt = seconds_since(total_start_time);

  result.orbital_preparation_result = prepared_active_space.orbital_result;
  result.ao_effective_one_electron_result =
      prepared_active_space.ao_effective_one_electron_result;
  result.act_h1e_result = prepared_active_space.active_space_one_electron_result;
  result.act_eri_result = prepared_active_space.active_space_two_electron_result;
  result.sso = active_space.sso;
  result.sso_grad.assign(active_space.sso.size(), 0.0);
  result.hho_grad.assign(active_space.hho.size(), 0.0);
  result.ggo_grad.assign(active_space.ggo.size(), 0.0);
  result.ri_active_pair_factor_grad.clear();

  const int n_active_orbitals = primitive_basis.n_active_orbitals;
  const Matrix spatial_overlap_matrix =
      dense_mat(active_space.sso, n_active_orbitals, "active_spatial_overlap_matrix");
  const Matrix one_electron_matrix =
      dense_mat(active_space.hho, n_active_orbitals, "active_one_electron_matrix");
  const Eigen::VectorXd coefficients =
      ground_state_coefficients(
          eigen_result.eigenvector_matrix,
          basis.n_states);
  const Matrix spin_adapted_hamiltonian_gradient =
      coefficients * coefficients.transpose();
  Matrix spin_adapted_overlap_gradient =
      -eigen_result.eigenvalues.front() * spin_adapted_hamiltonian_gradient;
  if (regularized_overlap.shift > 0.0) {
    spin_adapted_overlap_gradient.noalias() -=
        spin_adapted_overlap_gradient.trace() *
        (regularized_overlap.min_eigenvector *
         regularized_overlap.min_eigenvector.transpose());
  }

  const Matrix primitive_hamiltonian_gradient =
      lift_spin_adapted_matrix_gradient(
          spin_adapted_hamiltonian_gradient,
          basis.primitive_to_adapted_coefficients);
  const Matrix primitive_overlap_gradient =
      lift_spin_adapted_matrix_gradient(
          spin_adapted_overlap_gradient,
          basis.primitive_to_adapted_coefficients);
  const std::vector<PairWeight> pair_weights =
      build_pair_weights(
          primitive_hamiltonian_gradient,
          primitive_overlap_gradient);

  const Clock::time_point adjoint_start_time = Clock::now();
  double overlap_density_dt = 0.0;
  double one_electron_sigma_dt = 0.0;
  double two_electron_sigma_dt = 0.0;
  std::string parallel_error_message;
  const bool store_ggo_grad =
      !active_space.ggo.empty();
  const long long n_weighted_pairs =
      static_cast<long long>(pair_weights.size());

#ifdef _OPENMP
#pragma omp parallel if(n_weighted_pairs > 1)
#endif
  {
    PairGradAccum local_accum =
        make_pair_grad_accum(
            result.sso_grad.size(),
            result.hho_grad.size(),
            result.ggo_grad.size(),
            store_ggo_grad);
    Matrix spatial_density_workspace;
    ScalarBuffer ggo_grad_workspace;
    bool local_failed = false;
    std::string local_error_message;

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for (long long pair_index = 0; pair_index < n_weighted_pairs; ++pair_index) {
      if (local_failed) {
        continue;
      }

      try {
        const PairWeight& weight =
            pair_weights[pair_index];
        ++local_accum.grad_pair_calls;

        if (use_closed_shell_cache_path) {
          ++local_accum.grad_sample_calls;
          const PfKernelCache& cache =
              primitive_mats.lower_triangle_pair_caches[weight.cache_index];

          if (!is_effectively_zero(weight.overlap_weight)) {
            const Clock::time_point overlap_density_start_time = Clock::now();
            const ScalarBuffer overlap_metric_gradient =
                build_closed_shell_overlap_metric_gradient(
                    cache,
                    spatial_overlap_matrix,
                    basis.primitive_basis);
            accumulate_scaled_buffer(
                overlap_metric_gradient,
                weight.overlap_weight,
                &local_accum.sso_grad);
            local_accum.overlap_density_dt +=
                seconds_since(overlap_density_start_time);
          }

          if (!is_effectively_zero(weight.hamiltonian_weight)) {
            const Clock::time_point two_electron_sigma_start_time = Clock::now();
            const Matrix one_electron_sigma_adjoint =
                PfAdjointKernel::build_one_rdm_source_sigma_adjoint(
                    cache,
                    one_electron_matrix);
            const PfAdjointResult two_electron_adjoint =
                PfAdjointKernel::evaluate_closed_shell_two_electron(
                    cache,
                    active_space.ggo);
            spatial_density_workspace =
                collapse_spin_diagonal_blocks(one_electron_sigma_adjoint);
            spatial_density_workspace.noalias() +=
                two_electron_adjoint.spatial_density;
            ggo_grad_workspace = two_electron_adjoint.ggo_grad;
            local_accum.two_electron_sigma_dt +=
                seconds_since(two_electron_sigma_start_time);
            accumulate_scaled_matrix(
                spatial_density_workspace,
                weight.hamiltonian_weight,
                &local_accum.sso_grad);
            accumulate_scaled_matrix(
                cache.one_rdm,
                weight.hamiltonian_weight,
                &local_accum.hho_grad);
            accumulate_scaled_buffer(
                ggo_grad_workspace,
                weight.hamiltonian_weight,
                &local_accum.ggo_grad);
          }
        } else if (use_high_spin_open_shell_path) {
          const Clock::time_point open_shell_pair_start_time = Clock::now();
          const PfHighSpinOpenShellHamiltonianResult pair_result =
              evaluate_high_spin_open_shell_pf_state_pair_hamiltonian(
                  primitive_basis.states[weight.row],
                  primitive_basis.states[weight.col],
                  spatial_overlap_matrix,
                  one_electron_matrix,
                  active_space.ggo,
                  true);
          if (!pair_result.gradients_computed) {
            throw std::runtime_error(
                "high-spin open-shell pair evaluation did not return gradients");
          }
          local_accum.grad_sample_calls +=
              pair_result.interpolation_degree + 1;
          accumulate_scaled_matrix(
              pair_result.overlap_spatial_gradient,
              weight.overlap_weight,
              &local_accum.sso_grad);
          accumulate_scaled_matrix(
              pair_result.total_hamiltonian_spatial_gradient,
              weight.hamiltonian_weight,
              &local_accum.sso_grad);
          accumulate_scaled_matrix(
              pair_result.one_electron_matrix_gradient,
              weight.hamiltonian_weight,
              &local_accum.hho_grad);
          accumulate_scaled_buffer(
              pair_result.packed_two_electron_gradient,
              weight.hamiltonian_weight,
              &local_accum.ggo_grad);
          local_accum.two_electron_sigma_dt +=
              seconds_since(open_shell_pair_start_time);
        } else {
          const Clock::time_point open_shell_pair_start_time = Clock::now();
          const PfFixedMsOpenShellHamiltonianResult pair_result =
              evaluate_fixed_ms_open_shell_pf_state_pair_hamiltonian(
                  primitive_basis.states[weight.row],
                  primitive_basis.states[weight.col],
                  spatial_overlap_matrix,
                  one_electron_matrix,
                  active_space.ggo,
                  true);
          if (!pair_result.gradients_computed) {
            throw std::runtime_error(
                "fixed-M_s open-shell pair evaluation did not return gradients");
          }
          local_accum.grad_sample_calls +=
              pair_result.interpolation_degree + 1;
          accumulate_scaled_matrix(
              pair_result.overlap_spatial_gradient,
              weight.overlap_weight,
              &local_accum.sso_grad);
          accumulate_scaled_matrix(
              pair_result.total_hamiltonian_spatial_gradient,
              weight.hamiltonian_weight,
              &local_accum.sso_grad);
          accumulate_scaled_matrix(
              pair_result.one_electron_matrix_gradient,
              weight.hamiltonian_weight,
              &local_accum.hho_grad);
          accumulate_scaled_buffer(
              pair_result.packed_two_electron_gradient,
              weight.hamiltonian_weight,
              &local_accum.ggo_grad);
          local_accum.two_electron_sigma_dt +=
              seconds_since(open_shell_pair_start_time);
        }
      } catch (const std::exception& error) {
        local_failed = true;
        local_error_message = error.what();
      }
    }

#ifdef _OPENMP
#pragma omp critical
#endif
    {
      if (!local_error_message.empty() && parallel_error_message.empty()) {
        parallel_error_message = local_error_message;
      }
      merge_pair_grad_accum(
          local_accum,
          &result,
          &overlap_density_dt,
          &one_electron_sigma_dt,
          &two_electron_sigma_dt);
    }
  }

  if (!parallel_error_message.empty()) {
    throw std::runtime_error(parallel_error_message);
  }

  if (active_space.two_electron_representation ==
      xmvb::vb::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity) {
    backpropagate_closed_shell_ri_factor_gram(
        active_space.n_active_orbitals,
        active_space.n_auxiliary_functions,
        active_space.ri_active_pair_factors,
        result.ggo_grad,
        &result.ri_active_pair_factor_grad);
  }

  result.pair_grad_profile.grad_source_dt = overlap_density_dt;
  result.pair_grad_profile.grad_same_spin_dt = one_electron_sigma_dt;
  result.pair_grad_profile.grad_mix_dt = two_electron_sigma_dt;
  result.pair_grad_profile.grad_sso_dt =
      overlap_density_dt + one_electron_sigma_dt + two_electron_sigma_dt;
  result.adj_dt = seconds_since(adjoint_start_time);
  result.pair_grad_profile.grad_pair_total_dt = result.adj_dt;
  primitive_mats.lower_triangle_pair_caches.clear();
  result.scf_result.mats = std::move(projected_mats);
  result.scf_result.mats.s = column_major_storage(overlap_matrix);
  result.scf_result.mats.h = column_major_storage(hamiltonian_matrix);
  result.total_dt = seconds_since(total_start_time);
  result.scf_result.total_dt = result.total_dt;
  return result;
}

}  // namespace xmvb::pfaffian_vbscf
