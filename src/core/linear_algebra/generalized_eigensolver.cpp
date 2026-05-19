#include "core/linear_algebra/generalized_eigensolver.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

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
  const std::size_t expected_size =
      static_cast<std::size_t>(dimension) * dimension;
  if (hamiltonian_matrix.size() != expected_size ||
      overlap_matrix.size() != expected_size) {
    throw std::invalid_argument(
        "hamiltonian_matrix and overlap_matrix must be square matrices");
  }
}

void check_generalized_eigensolver_info(lapack_int info) {
  if (info < 0) {
    throw std::runtime_error(
        "LAPACKE_dsygvd failed: illegal argument " + std::to_string(-info));
  }
  if (info > 0) {
    throw std::runtime_error(
        "LAPACKE_dsygvd failed: overlap_matrix is not positive definite"
        " (info=" + std::to_string(info) + ")");
  }
}

// --- Davidson helpers -------------------------------------------------------
//
// The Davidson algorithm builds an expanding subspace V (n x m, m << n) and
// solves the small dense projected eigenproblem V^T H V y = lambda V^T S V y.
// Subspace restart (compression to the best Ritz vectors) prevents loss of
// orthogonality when the active subspace grows too large.

/// S-orthogonalize w against the columns of V (two-pass for numerical stability).
/// V columns must be S-orthonormal.  Returns false if w becomes linearly dependent.
bool davidson_s_orthogonalize(
    const Eigen::Ref<const Eigen::MatrixXd>& S,
    const Eigen::Ref<const Eigen::MatrixXd>& V,
    Eigen::Ref<Eigen::VectorXd> w) {
  for (int pass = 0; pass < 2; ++pass) {
    const Eigen::VectorXd coeffs = V.transpose() * (S * w);
    w.noalias() -= V * coeffs;
  }
  const double wSw = w.dot(S * w);
  if (wSw < 1e-24) return false;
  w /= std::sqrt(wSw);
  return true;
}

/// Diagonal preconditioner: t_i = r_i / (H_ii - lambda * S_ii).
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

/// Solve the projected eigenproblem H_proj y = lambda S_proj y using LAPACK.
Eigen::MatrixXd solve_subspace_eigenproblem(
    const Eigen::Ref<const Eigen::MatrixXd>& H_proj,
    const Eigen::Ref<const Eigen::MatrixXd>& S_proj,
    Eigen::Ref<Eigen::VectorXd> eigenvalues) {
  const int m = static_cast<int>(H_proj.rows());
  Eigen::MatrixXd H_copy = H_proj;
  Eigen::MatrixXd S_copy = S_proj;
  const lapack_int info = LAPACKE_dsygvd(
      LAPACK_COL_MAJOR, 1, 'V', 'U', m,
      H_copy.data(), m, S_copy.data(), m, eigenvalues.data());
  if (info != 0) {
    // Fallback: standard eigenproblem
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_copy);
    eigenvalues = solver.eigenvalues();
    return solver.eigenvectors();
  }
  return H_copy;
}

}  // namespace

GeneralizedEigenResult GeneralizedEigensolver::solve(
    const std::vector<double>& hamiltonian_matrix,
    const std::vector<double>& overlap_matrix,
    int dimension) const {
  validate_generalized_eigenproblem_inputs(
      hamiltonian_matrix, overlap_matrix, dimension);

  GeneralizedEigenResult result;
  std::vector<double> H_copy = hamiltonian_matrix;
  std::vector<double> S_copy = overlap_matrix;
  result.eigenvalues.resize(static_cast<std::size_t>(dimension));
  const lapack_int info = LAPACKE_dsygvd(
      LAPACK_COL_MAJOR, 1, 'V', 'U', dimension,
      H_copy.data(), dimension, S_copy.data(), dimension,
      result.eigenvalues.data());
  check_generalized_eigensolver_info(info);
  result.eigenvector_matrix = std::move(H_copy);
  return result;
}

std::vector<double> GeneralizedEigensolver::solve_eigenvalues_only(
    const std::vector<double>& hamiltonian_matrix,
    const std::vector<double>& overlap_matrix,
    int dimension) const {
  validate_generalized_eigenproblem_inputs(
      hamiltonian_matrix, overlap_matrix, dimension);

  std::vector<double> H_copy = hamiltonian_matrix;
  std::vector<double> S_copy = overlap_matrix;
  std::vector<double> eigenvalues(static_cast<std::size_t>(dimension));
  const lapack_int info = LAPACKE_dsygvd(
      LAPACK_COL_MAJOR, 1, 'N', 'U', dimension,
      H_copy.data(), dimension, S_copy.data(), dimension,
      eigenvalues.data());
  check_generalized_eigensolver_info(info);
  return eigenvalues;
}

GeneralizedEigenResult GeneralizedEigensolver::solve_davidson(
    const std::vector<double>& hamiltonian_matrix,
    const std::vector<double>& overlap_matrix,
    int dimension,
    int n_roots) const {
  validate_generalized_eigenproblem_inputs(
      hamiltonian_matrix, overlap_matrix, dimension);

  if (dimension <= n_roots || dimension < 20) {
    return solve(hamiltonian_matrix, overlap_matrix, dimension);
  }

  // Map flat column-major arrays to Eigen matrices (read-only).
  const Eigen::Map<const Eigen::MatrixXd> H(
      hamiltonian_matrix.data(), dimension, dimension);
  const Eigen::Map<const Eigen::MatrixXd> S(
      overlap_matrix.data(), dimension, dimension);

  constexpr int kMaxOuterIter = 300;
  const int kMaxSubspace  = std::min(120, dimension / 2);
  constexpr double kTol       = 1e-7;
  const int n_guess = std::min(kMaxSubspace, 2 * n_roots + 5);

  const Eigen::VectorXd diag_H = H.diagonal();
  const Eigen::VectorXd diag_S = S.diagonal();

  // Initial guess: unit vectors for diagonal entries nearest to the spectral
  // shift estimate, which approximates the ground-state energy.
  const double shift = (diag_H.array() / diag_S.array().max(1e-14)).mean();
  Eigen::VectorXd diag_dist =
      (diag_H.array() - shift * diag_S.array()).abs();
  std::vector<int> guess_idx(dimension);
  for (int i = 0; i < dimension; ++i) guess_idx[i] = i;
  std::partial_sort(
      guess_idx.begin(), guess_idx.begin() + n_guess, guess_idx.end(),
      [&](int a, int b) { return diag_dist[a] < diag_dist[b]; });

  // Subspace basis V (dimension x kMaxSubspace), S-orthonormal columns.
  Eigen::MatrixXd V(dimension, kMaxSubspace);
  Eigen::MatrixXd HV(dimension, kMaxSubspace);
  Eigen::MatrixXd SV(dimension, kMaxSubspace);
  int m = 0;  // active subspace dimension

  for (int k = 0; k < n_guess; ++k) {
    Eigen::VectorXd v = Eigen::VectorXd::Unit(dimension, guess_idx[k]);
    if (!davidson_s_orthogonalize(S, V.leftCols(m), v)) continue;
    V.col(m) = v;
    HV.col(m).noalias() = H * v;
    SV.col(m).noalias() = S * v;
    ++m;
    if (m >= kMaxSubspace) break;
  }

  // --- Subspace restart helper ---
  // Compress the subspace to the best n_keep Ritz vectors.
  auto subspace_restart = [&](int n_keep) {
    const Eigen::MatrixXd Vm = V.leftCols(m);
    const Eigen::MatrixXd HVm = HV.leftCols(m);
    const Eigen::MatrixXd SVm = SV.leftCols(m);
    Eigen::MatrixXd H_proj = Vm.transpose() * HVm;
    Eigen::MatrixXd S_proj = Vm.transpose() * SVm;
    Eigen::VectorXd ev(m);
    Eigen::MatrixXd evecs = solve_subspace_eigenproblem(H_proj, S_proj, ev);
    // Keep the best n_keep Ritz vectors
    const int nk = std::min(n_keep, m);
    Eigen::MatrixXd V_new(dimension, kMaxSubspace);
    Eigen::MatrixXd HV_new(dimension, kMaxSubspace);
    Eigen::MatrixXd SV_new(dimension, kMaxSubspace);
    for (int j = 0; j < nk; ++j) {
      V_new.col(j).noalias() = Vm * evecs.col(j);
      HV_new.col(j).noalias() = HVm * evecs.col(j);
      SV_new.col(j).noalias() = SVm * evecs.col(j);
    }
    V.swap(V_new);
    HV.swap(HV_new);
    SV.swap(SV_new);
    return nk;
  };

  Eigen::VectorXd eigenvalues(m);
  Eigen::MatrixXd subspace_eigenvectors;
  Eigen::VectorXd x(dimension), r(dimension);

  for (int iter = 0; iter < kMaxOuterIter; ++iter) {
    // Build projected matrices
    const Eigen::MatrixXd Vm = V.leftCols(m);
    const Eigen::MatrixXd HVm = HV.leftCols(m);
    const Eigen::MatrixXd SVm = SV.leftCols(m);
    Eigen::MatrixXd H_proj = Vm.transpose() * HVm;
    Eigen::MatrixXd S_proj = Vm.transpose() * SVm;

    // Solve subspace generalized eigenproblem
    eigenvalues.conservativeResize(m);
    subspace_eigenvectors = solve_subspace_eigenproblem(H_proj, S_proj, eigenvalues);

    // Check convergence and collect correction vectors for all unconverged roots
    std::vector<Eigen::VectorXd> corrections;
    bool all_converged = true;
    for (int root = 0; root < n_roots; ++root) {
      const double lambda = eigenvalues[root];
      x.noalias() = Vm * subspace_eigenvectors.col(root);
      r.noalias() = H * x - lambda * (S * x);
      if (r.norm() <= kTol) continue;
      all_converged = false;
      corrections.push_back(davidson_precondition(diag_H, diag_S, lambda, r));
    }

    if (all_converged) break;

    // Restart if subspace is full, keeping enough Ritz vectors to maintain
    // diversity for all requested roots plus correction directions.
    if (m + static_cast<int>(corrections.size()) > kMaxSubspace) {
      const int n_keep = std::min(m,
          std::max(n_roots + static_cast<int>(corrections.size()) + 4,
                   2 * n_roots + 4));
      m = subspace_restart(n_keep);
    }

    // Add correction vectors to subspace (S-orthogonalize against all active cols)
    for (auto& t : corrections) {
      if (davidson_s_orthogonalize(S, V.leftCols(m), t)) {
        V.col(m) = t;
        HV.col(m).noalias() = H * t;
        SV.col(m).noalias() = S * t;
        ++m;
        if (m >= kMaxSubspace) break;
      }
    }
  }

  // --- Build result ---
  const Eigen::MatrixXd Vm = V.leftCols(m);
  const Eigen::MatrixXd HVm = HV.leftCols(m);
  const Eigen::MatrixXd SVm = SV.leftCols(m);
  Eigen::MatrixXd H_proj = Vm.transpose() * HVm;
  Eigen::MatrixXd S_proj = Vm.transpose() * SVm;
  eigenvalues.resize(m);
  subspace_eigenvectors = solve_subspace_eigenproblem(H_proj, S_proj, eigenvalues);

  GeneralizedEigenResult result;
  result.eigenvalues.assign(eigenvalues.data(), eigenvalues.data() + n_roots);
  result.eigenvector_matrix.resize(
      static_cast<std::size_t>(dimension) * n_roots, 0.0);
  Eigen::Map<Eigen::MatrixXd> result_vecs(
      result.eigenvector_matrix.data(), dimension, n_roots);
  for (int root = 0; root < n_roots; ++root)
    result_vecs.col(root).noalias() = Vm * subspace_eigenvectors.col(root);

  return result;
}

}  // namespace xmvb::core
