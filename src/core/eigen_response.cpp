#include "core/eigen_response.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace xmvb::core {
namespace {

void validate_inputs(
    const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected,
    const EigenResponseOptions& options) {
  const Eigen::Index n = hamiltonian_diagonal.size();
  const Eigen::Index n_selected = selected_eigenvalues.size();
  if (n <= 0 || n_selected <= 0 || overlap_diagonal.size() != n ||
      selected_eigenvectors.rows() != n ||
      selected_eigenvectors.cols() != n_selected ||
      delta_hamiltonian_selected.rows() != n ||
      delta_hamiltonian_selected.cols() != n_selected ||
      delta_overlap_selected.rows() != n ||
      delta_overlap_selected.cols() != n_selected) {
    throw std::invalid_argument(
        "generalized-eigen response dimensions are inconsistent");
  }
  if (!hamiltonian_diagonal.allFinite() ||
      !overlap_diagonal.allFinite() ||
      !selected_eigenvalues.allFinite() ||
      !selected_eigenvectors.allFinite() ||
      !delta_hamiltonian_selected.allFinite() ||
      !delta_overlap_selected.allFinite() ||
      (overlap_diagonal.array() <= 0.0).any()) {
    throw std::invalid_argument(
        "generalized-eigen response inputs must be finite with a positive overlap diagonal");
  }
  if (options.max_iterations <= 0 ||
      !std::isfinite(options.relative_residual_tolerance) ||
      options.relative_residual_tolerance <= 0.0) {
    throw std::invalid_argument(
        "generalized-eigen response solver options are invalid");
  }
}

GeneralizedEigenActionResult apply_checked(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::MatrixXd>& vectors,
    int* block_actions) {
  if (!action) {
    throw std::invalid_argument(
        "generalized-eigen response requires a nonempty action");
  }
  GeneralizedEigenActionResult images = action(vectors);
  if (images.hamiltonian.rows() != vectors.rows() ||
      images.hamiltonian.cols() != vectors.cols() ||
      images.overlap.rows() != vectors.rows() ||
      images.overlap.cols() != vectors.cols() ||
      !images.hamiltonian.allFinite() || !images.overlap.allFinite()) {
    throw std::runtime_error(
        "generalized-eigen response action returned invalid images");
  }
  ++(*block_actions);
  return images;
}

Eigen::MatrixXd apply_bordered_operators(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& vectors,
    int* block_actions) {
  const Eigen::Index n = overlap_selected.rows();
  const Eigen::Index n_selected = overlap_selected.cols();
  GeneralizedEigenActionResult images = apply_checked(
      action, vectors.topRows(n), block_actions);
  Eigen::MatrixXd result(n + 1, n_selected);
  for (Eigen::Index state = 0; state < n_selected; ++state) {
    result.col(state).head(n).noalias() =
        images.hamiltonian.col(state) -
        selected_eigenvalues[state] * images.overlap.col(state) +
        vectors(n, state) * overlap_selected.col(state);
    result(n, state) = overlap_selected.col(state).dot(
        vectors.col(state).head(n));
  }
  return result;
}

Eigen::MatrixXd build_inverse_preconditioner(
    const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_selected) {
  const Eigen::Index n = hamiltonian_diagonal.size();
  const Eigen::Index n_selected = selected_eigenvalues.size();
  Eigen::MatrixXd inverse(n + 1, n_selected);
  for (Eigen::Index state = 0; state < n_selected; ++state) {
    const Eigen::ArrayXd shifted_diagonal =
        hamiltonian_diagonal.array() -
        selected_eigenvalues[state] * overlap_diagonal.array();
    const double scale = std::max(1.0, shifted_diagonal.abs().maxCoeff());
    const double numerical_floor =
        std::numeric_limits<double>::epsilon() * scale;
    inverse.col(state).head(n) =
        shifted_diagonal.abs().max(numerical_floor).inverse().matrix();
    inverse(n, state) = 1.0 / std::max(
        overlap_selected.col(state).norm(), numerical_floor);
  }
  return inverse;
}

}  // namespace

EigenResponseResult solve_generalized_eigen_response(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected,
    const EigenResponseOptions& options) {
  validate_inputs(
      hamiltonian_diagonal,
      overlap_diagonal,
      selected_eigenvalues,
      selected_eigenvectors,
      delta_hamiltonian_selected,
      delta_overlap_selected,
      options);

  const Eigen::Index n = hamiltonian_diagonal.size();
  const Eigen::Index n_selected = selected_eigenvalues.size();
  EigenResponseResult result;
  result.eigenvalue_response.resize(n_selected);

  const GeneralizedEigenActionResult accepted_images = apply_checked(
      action, selected_eigenvectors, &result.block_actions);
  const Eigen::MatrixXd& overlap_selected = accepted_images.overlap;
  Eigen::MatrixXd rhs(n + 1, n_selected);
  for (Eigen::Index state = 0; state < n_selected; ++state) {
    const Eigen::VectorXd forcing =
        delta_hamiltonian_selected.col(state) -
        selected_eigenvalues[state] * delta_overlap_selected.col(state);
    result.eigenvalue_response[state] =
        selected_eigenvectors.col(state).dot(forcing);
    rhs.col(state).head(n) = -forcing;
    rhs(n, state) = -0.5 * selected_eigenvectors.col(state).dot(
        delta_overlap_selected.col(state));
  }

  const Eigen::MatrixXd inverse_preconditioner = build_inverse_preconditioner(
      hamiltonian_diagonal,
      overlap_diagonal,
      selected_eigenvalues,
      overlap_selected);
  Eigen::MatrixXd solution = Eigen::MatrixXd::Zero(n + 1, n_selected);
  Eigen::MatrixXd v_old = Eigen::MatrixXd::Zero(n + 1, n_selected);
  Eigen::MatrixXd v = Eigen::MatrixXd::Zero(n + 1, n_selected);
  Eigen::MatrixXd v_new = rhs;
  Eigen::MatrixXd w = Eigen::MatrixXd::Zero(n + 1, n_selected);
  Eigen::MatrixXd w_new = inverse_preconditioner.array() * v_new.array();
  Eigen::MatrixXd p_older(n + 1, n_selected);
  Eigen::MatrixXd p_old = Eigen::MatrixXd::Zero(n + 1, n_selected);
  Eigen::MatrixXd p = Eigen::MatrixXd::Zero(n + 1, n_selected);

  Eigen::VectorXd rhs_norms(n_selected);
  Eigen::VectorXd residual_norms(n_selected);
  Eigen::VectorXd beta_new(n_selected);
  Eigen::VectorXd beta_first(n_selected);
  Eigen::VectorXd cosine = Eigen::VectorXd::Ones(n_selected);
  Eigen::VectorXd old_cosine = Eigen::VectorXd::Ones(n_selected);
  Eigen::VectorXd sine = Eigen::VectorXd::Zero(n_selected);
  Eigen::VectorXd old_sine = Eigen::VectorXd::Zero(n_selected);
  Eigen::VectorXd eta = Eigen::VectorXd::Ones(n_selected);
  result.iterations.assign(static_cast<std::size_t>(n_selected), 0);
  std::vector<bool> converged(static_cast<std::size_t>(n_selected), false);
  for (Eigen::Index state = 0; state < n_selected; ++state) {
    rhs_norms[state] = rhs.col(state).norm();
    residual_norms[state] = rhs_norms[state];
    if (rhs_norms[state] == 0.0) {
      converged[static_cast<std::size_t>(state)] = true;
      beta_new[state] = 0.0;
      beta_first[state] = 0.0;
      continue;
    }
    const double beta_squared = v_new.col(state).dot(w_new.col(state));
    if (!(beta_squared > 0.0) || !std::isfinite(beta_squared)) {
      throw std::runtime_error(
          "generalized-eigen response preconditioner is not positive definite");
    }
    beta_new[state] = std::sqrt(beta_squared);
    beta_first[state] = beta_new[state];
  }

  for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
    for (Eigen::Index state = 0; state < n_selected; ++state) {
      if (converged[static_cast<std::size_t>(state)]) {
        w_new.col(state).setZero();
        continue;
      }
      const double beta = beta_new[state];
      if (!(beta > 0.0) || !std::isfinite(beta)) {
        throw std::runtime_error(
            "generalized-eigen response MINRES encountered a Lanczos breakdown");
      }
      v_old.col(state) = v.col(state);
      v_new.col(state) /= beta;
      w_new.col(state) /= beta;
      v.col(state) = v_new.col(state);
      w.col(state) = w_new.col(state);
    }

    const Eigen::MatrixXd operator_images = apply_bordered_operators(
        action,
        selected_eigenvalues,
        overlap_selected,
        w,
        &result.block_actions);
    bool all_converged = true;
    for (Eigen::Index state = 0; state < n_selected; ++state) {
      if (converged[static_cast<std::size_t>(state)]) {
        continue;
      }
      const double beta = beta_new[state];
      v_new.col(state).noalias() =
          operator_images.col(state) - beta * v_old.col(state);
      const double alpha = v_new.col(state).dot(w_new.col(state));
      v_new.col(state).noalias() -= alpha * v.col(state);
      w_new.col(state) =
          inverse_preconditioner.col(state).array() *
          v_new.col(state).array();
      const double beta_squared = v_new.col(state).dot(w_new.col(state));
      if (beta_squared < 0.0 || !std::isfinite(beta_squared)) {
        throw std::runtime_error(
            "generalized-eigen response MINRES lost positive preconditioner curvature");
      }
      beta_new[state] = std::sqrt(std::max(0.0, beta_squared));

      const double r2 =
          sine[state] * alpha +
          cosine[state] * old_cosine[state] * beta;
      const double r3 = old_sine[state] * beta;
      const double r1_head =
          cosine[state] * alpha -
          old_cosine[state] * sine[state] * beta;
      const double r1 = std::hypot(r1_head, beta_new[state]);
      if (!(r1 > 0.0) || !std::isfinite(r1)) {
        throw std::runtime_error(
            "generalized-eigen response MINRES encountered a singular rotation");
      }
      old_cosine[state] = cosine[state];
      old_sine[state] = sine[state];
      cosine[state] = r1_head / r1;
      sine[state] = beta_new[state] / r1;

      p_older.col(state) = p_old.col(state);
      p_old.col(state) = p.col(state);
      p.col(state).noalias() =
          (w.col(state) - r2 * p_old.col(state) -
           r3 * p_older.col(state)) /
          r1;
      solution.col(state).noalias() +=
          beta_first[state] * cosine[state] * eta[state] * p.col(state);
      residual_norms[state] *= std::abs(sine[state]);
      result.iterations[static_cast<std::size_t>(state)] = iteration + 1;
      converged[static_cast<std::size_t>(state)] =
          residual_norms[state] <=
          options.relative_residual_tolerance * rhs_norms[state];
      if (!converged[static_cast<std::size_t>(state)]) {
        eta[state] = -sine[state] * eta[state];
        all_converged = false;
      }
    }
    if (all_converged) {
      break;
    }
  }

  const Eigen::MatrixXd final_images = apply_bordered_operators(
      action,
      selected_eigenvalues,
      overlap_selected,
      solution,
      &result.block_actions);
  result.relative_residual_norms.resize(n_selected);
  for (Eigen::Index state = 0; state < n_selected; ++state) {
    result.relative_residual_norms[state] = rhs_norms[state] == 0.0
        ? 0.0
        : (rhs.col(state) - final_images.col(state)).norm() /
            rhs_norms[state];
    if (!std::isfinite(result.relative_residual_norms[state]) ||
        result.relative_residual_norms[state] >
            options.relative_residual_tolerance) {
      std::ostringstream message;
      message << "generalized-eigen response MINRES did not reach the requested residual: state="
              << state << " residual="
              << result.relative_residual_norms[state] << " iterations="
              << result.iterations[static_cast<std::size_t>(state)];
      throw std::runtime_error(
          message.str());
    }
  }
  result.eigenvector_response = solution.topRows(n);
  return result;
}

}  // namespace xmvb::core
