#include "pfaffian_vbscf/scf/pf_spin_adapted_scf_eval.hpp"

#include <chrono>
#include <stdexcept>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "pfaffian_vbscf/matrices/pf_spin_adapted_matrix_projector.hpp"
#include "pfaffian_vbscf/types/eigen_types.hpp"

namespace xmvb::pfaffian_vbscf {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(const Clock::time_point& start_time) {
  return std::chrono::duration<double>(Clock::now() - start_time).count();
}

double compute_average_diagonal(
    const ScalarBuffer& matrix,
    int dimension) {
  if (dimension <= 0) {
    throw std::invalid_argument("dimension must be positive");
  }
  double diagonal_sum = 0.0;
  for (int index = 0; index < dimension; ++index) {
    diagonal_sum += matrix[xmvb::to_size(index) * dimension + index];
  }
  return diagonal_sum / static_cast<double>(dimension);
}

Matrix dense_mat(
    const ScalarBuffer& data,
    int dimension,
    const char* label) {
  const std::size_t expected_size =
      xmvb::to_size(dimension) * xmvb::to_size(dimension);
  if (data.size() != expected_size) {
    throw std::invalid_argument(std::string(label) + " size does not match the matrix dimension");
  }

  Matrix matrix = Matrix::Zero(dimension, dimension);
  for (int col = 0; col < dimension; ++col) {
    for (int row = 0; row < dimension; ++row) {
      matrix(row, col) = data[xmvb::to_size(col) * dimension + row];
    }
  }
  return matrix;
}

ScalarBuffer column_major_storage(const ConstMatrixRef& matrix) {
  ScalarBuffer data(xmvb::to_size(matrix.rows()) * matrix.cols(), 0.0);
  for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
      data[xmvb::to_size(col) * matrix.rows() + row] = matrix(row, col);
    }
  }
  return data;
}

Matrix symmetrize(const ConstMatrixRef& matrix) {
  return 0.5 * (matrix + matrix.transpose());
}

Matrix regularize_overlap(const ConstMatrixRef& overlap_matrix) {
  Matrix regularized_overlap = symmetrize(overlap_matrix);
  Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(regularized_overlap);
  if (eigensolver.info() != Eigen::Success) {
    throw std::runtime_error(
        "failed to diagonalize the spin-adapted Pfaffian overlap matrix");
  }

  const double min_eigenvalue = eigensolver.eigenvalues().minCoeff();
  const double shift = std::max(0.0, 1.0e-8 - min_eigenvalue);
  if (shift > 0.0) {
    regularized_overlap.diagonal().array() += shift;
  }

  Eigen::LLT<Matrix> llt(regularized_overlap);
  if (llt.info() != Eigen::Success) {
    throw std::runtime_error("failed to regularize the spin-adapted overlap matrix");
  }
  return regularized_overlap;
}

}  // namespace

PfSpinAdaptedScfEval::PfSpinAdaptedScfEval()
    : active_space_builder_(), matrix_builder_() {}

PfSpinAdaptedScfEval::PfSpinAdaptedScfEval(
    PfActBuilder active_space_builder,
    PfMatrixBuilder matrix_builder)
    : active_space_builder_(std::move(active_space_builder)),
      matrix_builder_(std::move(matrix_builder)) {}

PfScfResult PfSpinAdaptedScfEval::eval(
    const xmvb::vb::CppVbInput& input,
    const PfSpinAdaptedBasisData& basis,
    double nuclear_repulsion_energy) const {
  PfPreparedActiveSpaceData prepared_active_space =
      active_space_builder_.prepare(input);
  PfActiveSpaceData active_space = std::move(prepared_active_space.active_space);
  active_space.n_alpha = basis.primitive_basis.n_alpha;
  active_space.n_beta = basis.primitive_basis.n_beta;
  return eval_active_space(
      active_space,
      basis,
      nuclear_repulsion_energy,
      prepared_active_space.reference_energy);
}

PfScfResult PfSpinAdaptedScfEval::eval_active_space(
    const PfActiveSpaceData& active_space,
    const PfSpinAdaptedBasisData& basis,
    double nuclear_repulsion_energy,
    double reference_energy) const {
  const Clock::time_point start_time = Clock::now();
  PfScfResult result;
  PfActiveSpaceData primitive_active_space = active_space;
  primitive_active_space.n_alpha = basis.primitive_basis.n_alpha;
  primitive_active_space.n_beta = basis.primitive_basis.n_beta;

  const Clock::time_point matrix_start_time = Clock::now();
  const PfMatrixBuildResult primitive_mats =
      matrix_builder_.build(basis.primitive_basis, primitive_active_space);
  result.mats = project_spin_adapted_matrices(
      primitive_mats,
      basis.primitive_to_adapted_coefficients);
  result.matrix_dt = seconds_since(matrix_start_time);

  Matrix overlap_matrix = regularize_overlap(
      dense_mat(result.mats.s, basis.n_states, "result.mats.s"));
  Matrix hamiltonian_matrix =
      symmetrize(dense_mat(result.mats.h, basis.n_states, "result.mats.h"));
  result.mats.s = column_major_storage(overlap_matrix);
  result.mats.h = column_major_storage(hamiltonian_matrix);

  xmvb::core::GeneralizedEigensolver eigensolver;
  const auto eig = eigensolver.solve(result.mats.h, result.mats.s, basis.n_states);
  result.evals = eig.eigenvalues;
  if (result.evals.empty()) {
    throw std::runtime_error("generalized eigensolver returned no eigenvalues");
  }

  result.e_ref = reference_energy;
  result.e_ele = result.e_ref + result.evals.front();
  result.e_tot = result.e_ele + nuclear_repulsion_energy;
  result.avg_diag_s = compute_average_diagonal(result.mats.s, basis.n_states);
  result.total_dt = seconds_since(start_time);
  return result;
}

}  // namespace xmvb::pfaffian_vbscf
