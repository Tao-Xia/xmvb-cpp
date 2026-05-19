#include "core/linear_algebra/generalized_eigensolver.hpp"

#include <string>
#include <stdexcept>
#include <numeric>

#include <Eigen/Core>
#include "lapacke.h"

namespace xmvb::core {

namespace {

void validate_generalized_eigenproblem_inputs(
    const std::vector<double>& hamiltonian_matrix,
    const std::vector<double>& overlap_matrix,
    int dimension) {
  if (dimension <= 0) {
    throw std::invalid_argument("dimension must be positive");
  }

  const std::size_t matrix_dimension = static_cast<std::size_t>(dimension);
  const std::size_t expected_size =
      matrix_dimension * matrix_dimension;
  if (hamiltonian_matrix.size() != expected_size ||
      overlap_matrix.size() != expected_size) {
    throw std::invalid_argument(
        "hamiltonian_matrix and overlap_matrix must be square matrices");
  }
}

void check_generalized_eigensolver_info(
    lapack_int info) {
  if (info < 0) {
    throw std::runtime_error(
        "LAPACKE_dsygvd failed: illegal argument " + std::to_string(-info));
  }
  if (info > 0) {
    throw std::runtime_error(
        "LAPACKE_dsygvd failed: overlap_matrix is not positive definite or the "
        "eigensolver did not converge (info=" + std::to_string(info) + ")");
  }
}

}  // namespace

GeneralizedEigenResult GeneralizedEigensolver::solve(
    const std::vector<double>& hamiltonian_matrix,
    const std::vector<double>& overlap_matrix,
    int dimension) const {
  validate_generalized_eigenproblem_inputs(
      hamiltonian_matrix,
      overlap_matrix,
      dimension);

  GeneralizedEigenResult result;
  std::vector<double> hamiltonian_matrix_copy = hamiltonian_matrix;
  std::vector<double> overlap_matrix_copy = overlap_matrix;
  result.eigenvalues.resize(static_cast<std::size_t>(dimension));
  const lapack_int info = LAPACKE_dsygvd(
      LAPACK_COL_MAJOR,
      1,
      'V',
      'U',
      dimension,
      hamiltonian_matrix_copy.data(),
      dimension,
      overlap_matrix_copy.data(),
      dimension,
      result.eigenvalues.data());
  check_generalized_eigensolver_info(info);
  result.eigenvector_matrix = std::move(hamiltonian_matrix_copy);
  return result;
}

std::vector<double> GeneralizedEigensolver::solve_eigenvalues_only(
    const std::vector<double>& hamiltonian_matrix,
    const std::vector<double>& overlap_matrix,
    int dimension) const {
  validate_generalized_eigenproblem_inputs(
      hamiltonian_matrix,
      overlap_matrix,
      dimension);

  std::vector<double> hamiltonian_matrix_copy = hamiltonian_matrix;
  std::vector<double> overlap_matrix_copy = overlap_matrix;
  std::vector<double> eigenvalues(static_cast<std::size_t>(dimension));
  const lapack_int info = LAPACKE_dsygvd(
      LAPACK_COL_MAJOR,
      1,
      'N',
      'U',
      dimension,
      hamiltonian_matrix_copy.data(),
      dimension,
      overlap_matrix_copy.data(),
      dimension,
      eigenvalues.data());
  check_generalized_eigensolver_info(info);
  return eigenvalues;
}

}  // namespace xmvb::core
namespace xmvb::core {

namespace {

// --- Davidson generalized eigensolver helpers ---
//
// The Davidson algorithm solves H x = lambda S x for the lowest n_roots
// eigenpairs of large sparse-ish symmetric matrices.  It builds an expanding
// subspace V (n x m with m << n) and solves the small dense projected
// eigenproblem V^T H V y = lambda V^T S V y at each iteration.
// Convergence is tested on the Ritz residual r = H x - lambda S x.
//
// All dense linear algebra uses Eigen with column-major storage.

Eigen::VectorXd davidson_precondition(
    const Eigen::Ref<const Eigen::VectorXd>& diag_H,
    const Eigen::Ref<const Eigen::VectorXd>& diag_S,
    double lambda,
    const Eigen::Ref<const Eigen::VectorXd>& residual) {
  Eigen::VectorXd t(residual.size());
  for (Eigen::Index i = 0; i < residual.size(); ++i) {
    double denom = diag_H[i] - lambda * diag_S[i];
    if (std::abs(denom) < 1e-12) denom = (denom >= 0 ? 1e-12 : -1e-12);
    t[i] = residual[i] / denom;
  }
  return t;
}

bool davidson_s_orthogonalize(
    const Eigen::Ref<const Eigen::MatrixXd>& S,
    const Eigen::Ref<const Eigen::MatrixXd>& V,
    Eigen::Ref<Eigen::VectorXd> w) {
  const Eigen::VectorXd VtSw = V.transpose() * (S * w);
  w.noalias() -= V * VtSw;
  const double wSw = w.dot(S * w);
  if (wSw < 1e-24) return false;
  w /= std::sqrt(wSw);
  return true;
}

Eigen::MatrixXd solve_davidson_subspace(
    Eigen::Ref<Eigen::MatrixXd> H_proj,
    Eigen::Ref<Eigen::MatrixXd> S_proj,
    Eigen::Ref<Eigen::VectorXd> eigenvalues) {
  const int m = static_cast<int>(H_proj.rows());
  const lapack_int info = LAPACKE_dsygvd(
      LAPACK_COL_MAJOR, 1, 'V', 'U', m,
      H_proj.data(), m, S_proj.data(), m, eigenvalues.data());
  if (info != 0) {
    LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', m,
                   H_proj.data(), m, eigenvalues.data());
  }
  return H_proj;
}

}  // namespace

GeneralizedEigenResult GeneralizedEigensolver::solve_davidson(
    const std::vector<double>& hamiltonian_matrix,
    const std::vector<double>& overlap_matrix,
    int dimension,
    int n_roots) const {
  validate_generalized_eigenproblem_inputs(
      hamiltonian_matrix, overlap_matrix, dimension);

  if (dimension <= n_roots || dimension < 50) {
    return solve(hamiltonian_matrix, overlap_matrix, dimension);
  }

  const Eigen::Map<const Eigen::MatrixXd> H(
      hamiltonian_matrix.data(), dimension, dimension);
  const Eigen::Map<const Eigen::MatrixXd> S(
      overlap_matrix.data(), dimension, dimension);

  constexpr int kMaxIter = 60;
  constexpr int kMaxSubspace = 300;
  constexpr double kTol = 1e-7;
  const int max_subspace = std::min(dimension, kMaxSubspace);
  const int n_guess = std::min(max_subspace, 2 * n_roots + 8);

  const Eigen::VectorXd diag_H = H.diagonal();
  const Eigen::VectorXd diag_S = S.diagonal();
  const double shift = (diag_H.array() / diag_S.array().max(1e-14)).mean();

  Eigen::VectorXd diag_dist =
      (diag_H.array() - shift * diag_S.array()).abs();
  std::vector<int> guess_indices(dimension);
  std::iota(guess_indices.begin(), guess_indices.end(), 0);
  std::partial_sort(
      guess_indices.begin(), guess_indices.begin() + n_guess,
      guess_indices.end(),
      [&](int a, int b) { return diag_dist[a] < diag_dist[b]; });

  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(dimension, max_subspace);
  int subspace_dim = 0;
  Eigen::MatrixXd HV(dimension, max_subspace);
  Eigen::MatrixXd SV(dimension, max_subspace);

  for (int k = 0; k < n_guess; ++k) {
    Eigen::VectorXd v = Eigen::VectorXd::Unit(dimension, guess_indices[k]);
    if (!davidson_s_orthogonalize(S, V.leftCols(subspace_dim), v))
      continue;
    V.col(subspace_dim) = v;
    HV.col(subspace_dim).noalias() = H * v;
    SV.col(subspace_dim).noalias() = S * v;
    ++subspace_dim;
    if (subspace_dim >= max_subspace) break;
  }

  Eigen::VectorXd eigenvalues(subspace_dim);
  Eigen::MatrixXd subspace_eigenvectors;
  Eigen::VectorXd x(dimension), r(dimension);

  for (int iter = 0; iter < kMaxIter; ++iter) {
    const int m = subspace_dim;
    const Eigen::MatrixXd Vm = V.leftCols(m);
    const Eigen::MatrixXd HVm = HV.leftCols(m);
    const Eigen::MatrixXd SVm = SV.leftCols(m);

    Eigen::MatrixXd H_proj = Vm.transpose() * HVm;
    Eigen::MatrixXd S_proj = Vm.transpose() * SVm;
    subspace_eigenvectors =
        solve_davidson_subspace(H_proj, S_proj, eigenvalues.head(m));

    bool all_converged = true;
    for (int root = 0; root < n_roots; ++root) {
      const double lambda = eigenvalues[root];
      x.noalias() = Vm * subspace_eigenvectors.col(root);
      r.noalias() = H * x - lambda * (S * x);
      if (r.norm() <= kTol) continue;
      all_converged = false;

      Eigen::VectorXd t = davidson_precondition(diag_H, diag_S, lambda, r);
      if (davidson_s_orthogonalize(S, Vm, t)) {
        if (subspace_dim < max_subspace) {
          V.col(subspace_dim) = t;
          HV.col(subspace_dim).noalias() = H * t;
          SV.col(subspace_dim).noalias() = S * t;
          ++subspace_dim;
          eigenvalues.conservativeResize(subspace_dim);
        }
      }
    }
    if (all_converged || subspace_dim >= max_subspace) break;
  }

  const Eigen::MatrixXd Vm = V.leftCols(subspace_dim);
  GeneralizedEigenResult result;
  result.eigenvalues.assign(eigenvalues.data(), eigenvalues.data() + n_roots);
  result.eigenvector_matrix.resize(
      static_cast<std::size_t>(dimension) * n_roots);
  Eigen::Map<Eigen::MatrixXd> result_vecs(
      result.eigenvector_matrix.data(), dimension, n_roots);
  for (int root = 0; root < n_roots; ++root) {
    result_vecs.col(root).noalias() =
        Vm * subspace_eigenvectors.col(root);
  }
  return result;
}
}  // namespace xmvb::core
