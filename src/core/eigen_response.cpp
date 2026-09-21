#include "core/eigen_response.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <Eigen/QR>
#include <Eigen/LU>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

namespace xmvb::core {
namespace {

void validate_response_columns(
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected) {
  const Eigen::Index n = selected_eigenvectors.rows();
  const Eigen::Index n_selected = selected_eigenvalues.size();
  if (n <= 0 || n_selected <= 0 ||
      selected_eigenvectors.cols() != n_selected ||
      overlap_selected.rows() != n ||
      overlap_selected.cols() != n_selected ||
      delta_hamiltonian_selected.rows() != n ||
      delta_hamiltonian_selected.cols() != n_selected ||
      delta_overlap_selected.rows() != n ||
      delta_overlap_selected.cols() != n_selected) {
    throw std::invalid_argument(
        "generalized-eigen response dimensions are inconsistent");
  }
  if (!selected_eigenvalues.allFinite() ||
      !selected_eigenvectors.allFinite() ||
      !overlap_selected.allFinite() ||
      !delta_hamiltonian_selected.allFinite() ||
      !delta_overlap_selected.allFinite()) {
    throw std::invalid_argument(
        "generalized-eigen response inputs must be finite");
  }
}

void validate_inputs(
    const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected,
    const EigenResponseOptions& options) {
  validate_response_columns(
      selected_eigenvalues, selected_eigenvectors, overlap_selected,
      delta_hamiltonian_selected, delta_overlap_selected);
  const Eigen::Index n = hamiltonian_diagonal.size();
  if (n != selected_eigenvectors.rows() || overlap_diagonal.size() != n) {
    throw std::invalid_argument(
        "generalized-eigen response dimensions are inconsistent");
  }
  if (!hamiltonian_diagonal.allFinite() ||
      !overlap_diagonal.allFinite() ||
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
    const Eigen::Ref<const Eigen::MatrixXd>& normal_projector_diagonal) {
  const Eigen::Index n = hamiltonian_diagonal.size();
  const Eigen::Index n_selected = selected_eigenvalues.size();
  if (normal_projector_diagonal.rows() != n ||
      normal_projector_diagonal.cols() != n_selected ||
      !normal_projector_diagonal.allFinite() ||
      (normal_projector_diagonal.array() < 0.0).any()) {
    throw std::invalid_argument(
        "generalized-eigen response normal-projector diagonal is invalid");
  }
  Eigen::MatrixXd inverse(n, n_selected);
  for (Eigen::Index state = 0; state < n_selected; ++state) {
    const Eigen::ArrayXd shifted_diagonal =
        hamiltonian_diagonal.array() -
        selected_eigenvalues[state] * overlap_diagonal.array();
    const double sigma = shifted_diagonal.abs().maxCoeff();
    // Adding sigma Q Q^T changes only the Jacobi model: P Q=0 implies
    // P(A+sigma Q Q^T)P=PAP. Its positive diagonal lift prevents the known
    // gauge null direction from producing an inverse-epsilon preconditioner.
    const Eigen::ArrayXd lifted_diagonal = shifted_diagonal.abs() +
        sigma * normal_projector_diagonal.col(state).array();
    const double scale = std::max(1.0, sigma);
    const double numerical_floor =
        std::numeric_limits<double>::epsilon() * scale;
    inverse.col(state) =
        lifted_diagonal.max(numerical_floor).inverse().matrix();
  }
  return inverse;
}

Eigen::VectorXd apply_response_preconditioner_column(
    const Eigen::Ref<const Eigen::VectorXd>& residual,
    const Eigen::Ref<const Eigen::VectorXd>& jacobi_inverse,
    EigenResponseRecycleSpace* recycle) {
  return recycle != nullptr && recycle->size() > 0
      ? recycle->apply_absolute_spectral_preconditioner(
            residual, jacobi_inverse)
      : (jacobi_inverse.array() * residual.array()).matrix();
}

Eigen::MatrixXd apply_response_preconditioner(
    const Eigen::Ref<const Eigen::MatrixXd>& residuals,
    const Eigen::Ref<const Eigen::MatrixXd>& jacobi_inverse,
    const std::vector<EigenResponseRecycleSpace*>& recycle_spaces) {
  if (residuals.rows() != jacobi_inverse.rows() ||
      residuals.cols() != jacobi_inverse.cols() ||
      (!recycle_spaces.empty() &&
       recycle_spaces.size() != static_cast<std::size_t>(residuals.cols()))) {
    throw std::invalid_argument(
        "response preconditioner dimensions are inconsistent");
  }
  Eigen::MatrixXd result(residuals.rows(), residuals.cols());
  for (Eigen::Index state = 0; state < residuals.cols(); ++state) {
    EigenResponseRecycleSpace* recycle = recycle_spaces.empty()
        ? nullptr
        : recycle_spaces[static_cast<std::size_t>(state)];
    result.col(state) = apply_response_preconditioner_column(
        residuals.col(state), jacobi_inverse.col(state), recycle);
  }
  return result;
}

void project_selected_roots(
    Eigen::MatrixXd* vectors,
    const Eigen::Ref<const Eigen::MatrixXd>& root_units) {
  for (Eigen::Index state = 0; state < vectors->cols(); ++state) {
    vectors->col(state).noalias() -= root_units.col(state) *
        root_units.col(state).dot(vectors->col(state));
  }
}

Eigen::MatrixXd apply_projected_operators(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& root_units,
    const Eigen::Ref<const Eigen::MatrixXd>& vectors,
    int* block_actions) {
  Eigen::MatrixXd projected = vectors;
  project_selected_roots(&projected, root_units);
  GeneralizedEigenActionResult images = apply_checked(
      action, projected, block_actions);
  Eigen::MatrixXd result(vectors.rows(), vectors.cols());
  for (Eigen::Index state = 0; state < vectors.cols(); ++state) {
    result.col(state).noalias() = images.hamiltonian.col(state) -
        selected_eigenvalues[state] * images.overlap.col(state);
  }
  project_selected_roots(&result, root_units);
  return result;
}

Eigen::MatrixXd selected_ritz_residuals(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::MatrixXd* cached_residuals,
    int* block_actions) {
  if (cached_residuals != nullptr) {
    if (cached_residuals->rows() != selected_eigenvectors.rows() ||
        cached_residuals->cols() != selected_eigenvectors.cols() ||
        !cached_residuals->allFinite()) {
      throw std::invalid_argument(
          "selected generalized-eigen residual cache is invalid");
    }
    return *cached_residuals;
  }
  const GeneralizedEigenActionResult images = apply_checked(
      action, selected_eigenvectors, block_actions);
  return images.hamiltonian -
      images.overlap * selected_eigenvalues.asDiagonal();
}

}  // namespace

int EigenResponseRecycleSpace::dimension() const noexcept {
  return static_cast<int>(basis_.rows());
}

int EigenResponseRecycleSpace::size() const noexcept {
  return static_cast<int>(basis_.cols());
}

std::uint64_t EigenResponseRecycleSpace::revision() const noexcept {
  return revision_;
}

const Eigen::MatrixXd& EigenResponseRecycleSpace::basis() const noexcept {
  return basis_;
}

const Eigen::MatrixXd& EigenResponseRecycleSpace::projected_inverse() const {
  if (basis_.cols() == 0) {
    projected_inverse_.resize(0, 0);
    projected_inverse_revision_ = revision_;
  } else {
    prepare_projected_inverse();
  }
  return projected_inverse_;
}

Eigen::VectorXd
EigenResponseRecycleSpace::apply_absolute_spectral_preconditioner(
    const Eigen::Ref<const Eigen::VectorXd>& right_hand_side,
    const Eigen::Ref<const Eigen::VectorXd>& jacobi_inverse) const {
  if (right_hand_side.size() != dimension() ||
      jacobi_inverse.size() != dimension() ||
      !right_hand_side.allFinite() || !jacobi_inverse.allFinite() ||
      (jacobi_inverse.array() <= 0.0).any()) {
    throw std::invalid_argument(
        "response spectral preconditioner inputs are invalid");
  }
  if (size() == 0) {
    return jacobi_inverse.array() * right_hand_side.array();
  }
  prepare_projected_inverse();
  if (spectral_preconditioner_basis_.cols() == 0) {
    return jacobi_inverse.array() * right_hand_side.array();
  }
  const Eigen::VectorXd spectral_coordinates =
      spectral_preconditioner_basis_.transpose() * right_hand_side;
  Eigen::VectorXd complement = right_hand_side -
      spectral_preconditioner_basis_ * spectral_coordinates;
  Eigen::VectorXd result = jacobi_inverse.array() * complement.array();
  result.noalias() -= spectral_preconditioner_basis_ *
      (spectral_preconditioner_basis_.transpose() * result);
  result.noalias() += spectral_preconditioner_basis_ *
      (spectral_preconditioner_inverse_eigenvalues_.array() *
       spectral_coordinates.array()).matrix();
  return result;
}

void EigenResponseRecycleSpace::clear() {
  basis_.resize(0, 0);
  operator_images_.resize(0, 0);
  projected_inverse_.resize(0, 0);
  spectral_preconditioner_basis_.resize(0, 0);
  spectral_preconditioner_inverse_eigenvalues_.resize(0);
  ++revision_;
}

void EigenResponseRecycleSpace::prepare_projected_inverse() const {
  if (projected_inverse_revision_ == revision_) return;

  const Eigen::MatrixXd raw_projected_operator =
      basis_.transpose() * operator_images_;
  const double symmetry_error =
      (raw_projected_operator - raw_projected_operator.transpose()).norm();
  const double symmetry_tolerance =
      std::sqrt(std::numeric_limits<double>::epsilon()) *
      std::max(1.0, raw_projected_operator.norm());
  if (!std::isfinite(symmetry_error) ||
      symmetry_error > symmetry_tolerance) {
    throw std::runtime_error(
        "response recycle projected operator is not symmetric");
  }
  const Eigen::MatrixXd projected_operator = 0.5 *
      (raw_projected_operator + raw_projected_operator.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(
      projected_operator);
  if (eigensolver.info() != Eigen::Success) {
    throw std::runtime_error(
        "response recycle projected operator diagonalization failed");
  }
  const Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
  const double spectral_scale = std::max(
      1.0, eigenvalues.cwiseAbs().maxCoeff());
  const double threshold = std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max<Eigen::Index>(1, basis_.cols())) *
      spectral_scale;
  Eigen::VectorXd inverse_eigenvalues = Eigen::VectorXd::Zero(
      eigenvalues.size());
  int numerical_rank = 0;
  for (Eigen::Index index = 0; index < eigenvalues.size(); ++index) {
    if (std::abs(eigenvalues[index]) > threshold) {
      inverse_eigenvalues[index] = 1.0 / eigenvalues[index];
      ++numerical_rank;
    }
  }
  projected_inverse_.noalias() = eigensolver.eigenvectors() *
      inverse_eigenvalues.asDiagonal() *
      eigensolver.eigenvectors().transpose();
  spectral_preconditioner_basis_.resize(dimension(), numerical_rank);
  spectral_preconditioner_inverse_eigenvalues_.resize(numerical_rank);
  int retained = 0;
  for (Eigen::Index index = 0; index < eigenvalues.size(); ++index) {
    if (inverse_eigenvalues[index] == 0.0) continue;
    spectral_preconditioner_basis_.col(retained).noalias() =
        basis_ * eigensolver.eigenvectors().col(index);
    spectral_preconditioner_inverse_eigenvalues_[retained] =
        std::abs(inverse_eigenvalues[index]);
    ++retained;
  }
  projected_inverse_revision_ = revision_;
}

EigenResponseRecycleApplication EigenResponseRecycleSpace::galerkin_apply(
    const Eigen::Ref<const Eigen::VectorXd>& right_hand_side) const {
  if (!right_hand_side.allFinite()) {
    throw std::invalid_argument(
        "response recycle right-hand side must be finite");
  }
  EigenResponseRecycleApplication result;
  result.solution = Eigen::VectorXd::Zero(right_hand_side.size());
  result.operator_image = Eigen::VectorXd::Zero(right_hand_side.size());
  if (basis_.cols() == 0) return result;
  if (basis_.rows() != right_hand_side.size() ||
      operator_images_.rows() != right_hand_side.size() ||
      operator_images_.cols() != basis_.cols()) {
    throw std::invalid_argument(
        "response recycle space dimension does not match the right-hand side");
  }

  prepare_projected_inverse();
  const Eigen::VectorXd coefficients =
      projected_inverse_ * (basis_.transpose() * right_hand_side);
  result.solution.noalias() = basis_ * coefficients;
  result.operator_image.noalias() = operator_images_ * coefficients;
  if (!result.solution.allFinite() || !result.operator_image.allFinite()) {
    throw std::runtime_error(
        "response recycle Galerkin application is not finite");
  }
  result.available = true;
  return result;
}

bool EigenResponseRecycleSpace::append(
    const Eigen::Ref<const Eigen::VectorXd>& solution,
    const Eigen::Ref<const Eigen::VectorXd>& operator_image) {
  if (solution.size() <= 0 || operator_image.size() != solution.size() ||
      !solution.allFinite() || !operator_image.allFinite()) {
    throw std::invalid_argument(
        "response recycle pair dimensions or values are invalid");
  }
  if (basis_.cols() == 0) {
    basis_.resize(solution.size(), 0);
    operator_images_.resize(solution.size(), 0);
  } else if (basis_.rows() != solution.size() ||
             operator_images_.rows() != solution.size() ||
             operator_images_.cols() != basis_.cols()) {
    throw std::invalid_argument(
        "response recycle pair dimension does not match the stored space");
  }

  Eigen::VectorXd direction = solution;
  Eigen::VectorXd image = operator_image;
  for (int pass = 0; pass < 2; ++pass) {
    const Eigen::VectorXd coefficients = basis_.transpose() * direction;
    direction.noalias() -= basis_ * coefficients;
    image.noalias() -= operator_images_ * coefficients;
  }
  const double original_norm = solution.norm();
  const double direction_norm = direction.norm();
  const double dependence_threshold =
      std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max<Eigen::Index>(1, solution.size())) *
      original_norm;
  if (!(direction_norm > dependence_threshold)) return false;
  direction /= direction_norm;
  image /= direction_norm;
  const Eigen::Index old_size = basis_.cols();
  basis_.conservativeResize(Eigen::NoChange, old_size + 1);
  operator_images_.conservativeResize(Eigen::NoChange, old_size + 1);
  basis_.col(old_size) = direction;
  operator_images_.col(old_size) = image;
  ++revision_;
  return true;
}

EigenResponseResult solve_generalized_eigen_response_from_full_spectrum(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& full_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& full_eigenvectors,
    const std::vector<int>& selected_root_indices,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected,
    double relative_residual_tolerance) {
  const Eigen::Index n = full_eigenvalues.size();
  const Eigen::Index n_rhs = selected_eigenvalues.size();
  if (n <= 0 || n_rhs <= 0 || full_eigenvectors.rows() != n ||
      full_eigenvectors.cols() != n ||
      selected_root_indices.size() != static_cast<std::size_t>(n_rhs) ||
      selected_eigenvectors.rows() != n ||
      selected_eigenvectors.cols() != n_rhs ||
      overlap_selected.rows() != n || overlap_selected.cols() != n_rhs ||
      delta_hamiltonian_selected.rows() != n ||
      delta_hamiltonian_selected.cols() != n_rhs ||
      delta_overlap_selected.rows() != n ||
      delta_overlap_selected.cols() != n_rhs ||
      !full_eigenvalues.allFinite() || !full_eigenvectors.allFinite() ||
      !selected_eigenvalues.allFinite() ||
      !selected_eigenvectors.allFinite() || !overlap_selected.allFinite() ||
      !delta_hamiltonian_selected.allFinite() ||
      !delta_overlap_selected.allFinite() ||
      !std::isfinite(relative_residual_tolerance) ||
      !(relative_residual_tolerance > 0.0)) {
    throw std::invalid_argument(
        "complete generalized-eigen response inputs are inconsistent");
  }

  Eigen::MatrixXd forcing(n, n_rhs);
  Eigen::MatrixXd rhs(n + 1, n_rhs);
  EigenResponseResult result;
  result.eigenvalue_response.resize(n_rhs);
  result.iterations.assign(static_cast<std::size_t>(n_rhs), 0);
  for (Eigen::Index column = 0; column < n_rhs; ++column) {
    const int root = selected_root_indices[static_cast<std::size_t>(column)];
    if (root < 0 || root >= n ||
        selected_eigenvalues[column] != full_eigenvalues[root]) {
      throw std::invalid_argument(
          "selected root does not match the complete eigenspectrum");
    }
    forcing.col(column) =
        delta_hamiltonian_selected.col(column) -
        selected_eigenvalues[column] *
            delta_overlap_selected.col(column);
    result.eigenvalue_response[column] =
        selected_eigenvectors.col(column).dot(forcing.col(column));
    rhs.col(column).head(n) = -forcing.col(column);
    rhs(n, column) = -0.5 *
        selected_eigenvectors.col(column).dot(
            delta_overlap_selected.col(column));
  }

  Eigen::MatrixXd coefficients =
      full_eigenvectors.transpose() * forcing;
  for (Eigen::Index column = 0; column < n_rhs; ++column) {
    const int root = selected_root_indices[static_cast<std::size_t>(column)];
    for (Eigen::Index other = 0; other < n; ++other) {
      if (other == root) {
        coefficients(other, column) = rhs(n, column);
        continue;
      }
      const double gap =
          full_eigenvalues[other] - selected_eigenvalues[column];
      if (gap == 0.0) {
        throw std::runtime_error(
            "selected structure root is degenerate; isolated-state response is undefined");
      }
      coefficients(other, column) /= -gap;
    }
  }
  result.eigenvector_response = full_eigenvectors * coefficients;
  Eigen::MatrixXd solution(n + 1, n_rhs);
  solution.topRows(n) = result.eigenvector_response;
  // The bordered multiplier is -delta E because its top-right block is +S c.
  solution.bottomRows(1) = -result.eigenvalue_response.transpose();
  const Eigen::MatrixXd images = apply_bordered_operators(
      action,
      selected_eigenvalues,
      overlap_selected,
      solution,
      &result.block_actions);
  result.relative_residual_norms.resize(n_rhs);
  for (Eigen::Index column = 0; column < n_rhs; ++column) {
    const double rhs_norm = rhs.col(column).norm();
    result.relative_residual_norms[column] = rhs_norm == 0.0
        ? (rhs.col(column) - images.col(column)).norm()
        : (rhs.col(column) - images.col(column)).norm() / rhs_norm;
    if (!std::isfinite(result.relative_residual_norms[column]) ||
        result.relative_residual_norms[column] >
            relative_residual_tolerance) {
      std::ostringstream message;
      message << "complete-spectrum structure response disagrees with the accepted H/S action: state="
              << column << " relative residual="
              << result.relative_residual_norms[column];
      throw std::runtime_error(message.str());
    }
  }
  return result;
}

EigenResponseResult solve_generalized_eigen_response(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected,
    const EigenResponseOptions& options,
    const std::vector<EigenResponseRecycleSpace*>& recycle_spaces,
    const Eigen::MatrixXd* selected_residuals) {
  validate_inputs(
      hamiltonian_diagonal,
      overlap_diagonal,
      selected_eigenvalues,
      selected_eigenvectors,
      overlap_selected,
      delta_hamiltonian_selected,
      delta_overlap_selected,
      options);

  const Eigen::Index n = hamiltonian_diagonal.size();
  const Eigen::Index n_selected = selected_eigenvalues.size();
  if (!recycle_spaces.empty() &&
      recycle_spaces.size() != static_cast<std::size_t>(n_selected)) {
    throw std::invalid_argument(
        "generalized-eigen response recycle-space count is inconsistent");
  }
  EigenResponseResult result;
  result.eigenvalue_response.resize(n_selected);
  const Eigen::MatrixXd ritz_residuals = selected_ritz_residuals(
      action, selected_eigenvalues, selected_eigenvectors,
      selected_residuals, &result.block_actions);

  Eigen::MatrixXd full_rhs(n + 1, n_selected);
  Eigen::MatrixXd rhs(n, n_selected);
  Eigen::MatrixXd root_units(n, n_selected);
  Eigen::MatrixXd particular_solution(n, n_selected);
  Eigen::VectorXd root_metric_norms(n_selected);
  for (Eigen::Index state = 0; state < n_selected; ++state) {
    const Eigen::VectorXd forcing =
        delta_hamiltonian_selected.col(state) -
        selected_eigenvalues[state] * delta_overlap_selected.col(state);
    root_metric_norms[state] = selected_eigenvectors.col(state).dot(
        overlap_selected.col(state));
    if (!(root_metric_norms[state] > 0.0) ||
        !std::isfinite(root_metric_norms[state])) {
      throw std::invalid_argument(
          "selected generalized-eigen root has no positive overlap norm");
    }
    // The caller supplies S-normalized Ritz vectors. Dividing by the measured
    // metric norm also removes harmless normalization drift from their images.
    result.eigenvalue_response[state] =
        selected_eigenvectors.col(state).dot(forcing) /
        root_metric_norms[state];
    full_rhs.col(state).head(n) = -forcing;
    full_rhs(n, state) = -0.5 * selected_eigenvectors.col(state).dot(
        delta_overlap_selected.col(state));
    const double constraint_norm = overlap_selected.col(state).norm();
    if (!(constraint_norm > 0.0) || !std::isfinite(constraint_norm)) {
      throw std::invalid_argument(
          "selected generalized-eigen overlap image is zero");
    }
    root_units.col(state) =
        overlap_selected.col(state) / constraint_norm;
    particular_solution.col(state) = selected_eigenvectors.col(state) *
        (full_rhs(n, state) / root_metric_norms[state]);
    rhs.col(state) = -forcing - ritz_residuals.col(state) *
        (full_rhs(n, state) / root_metric_norms[state]);
  }
  project_selected_roots(&rhs, root_units);

  const Eigen::MatrixXd inverse_preconditioner = build_inverse_preconditioner(
      hamiltonian_diagonal, overlap_diagonal, selected_eigenvalues,
      root_units.array().square().matrix());
  Eigen::MatrixXd solution = Eigen::MatrixXd::Zero(n, n_selected);
  Eigen::MatrixXd v_old = Eigen::MatrixXd::Zero(n, n_selected);
  Eigen::MatrixXd v = Eigen::MatrixXd::Zero(n, n_selected);
  Eigen::MatrixXd v_new = rhs;
  Eigen::MatrixXd w = Eigen::MatrixXd::Zero(n, n_selected);
  Eigen::MatrixXd w_new = Eigen::MatrixXd::Zero(n, n_selected);
  Eigen::MatrixXd p_older = Eigen::MatrixXd::Zero(n, n_selected);
  Eigen::MatrixXd p_old = Eigen::MatrixXd::Zero(n, n_selected);
  Eigen::MatrixXd p = Eigen::MatrixXd::Zero(n, n_selected);

  Eigen::VectorXd rhs_norms(n_selected);
  Eigen::VectorXd original_rhs_norms(n_selected);
  Eigen::VectorXd bordered_absolute_targets(n_selected);
  Eigen::VectorXd correctable_absolute_targets(n_selected);
  Eigen::VectorXd preconditioned_absolute_targets(n_selected);
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
  std::vector<int> candidate_checks(static_cast<std::size_t>(n_selected), 0);
  std::vector<int> reliable_restarts(static_cast<std::size_t>(n_selected), 0);
  Eigen::VectorXd least_candidate_projected_residual =
      Eigen::VectorXd::Constant(n_selected,
          std::numeric_limits<double>::infinity());
  Eigen::VectorXd last_candidate_projected_residual =
      Eigen::VectorXd::Zero(n_selected);
  Eigen::VectorXd least_candidate_bordered_residual =
      Eigen::VectorXd::Constant(n_selected,
          std::numeric_limits<double>::infinity());
  Eigen::VectorXd last_candidate_bordered_residual =
      Eigen::VectorXd::Zero(n_selected);
  for (Eigen::Index state = 0; state < n_selected; ++state) {
    rhs_norms[state] = rhs.col(state).norm();
    original_rhs_norms[state] = full_rhs.col(state).norm();
    bordered_absolute_targets[state] =
        options.relative_residual_tolerance * original_rhs_norms[state];
    // The multiplier recovered from c^T maps a complement residual e through
    // the oblique projector I - (S c)c^T/(c^T S c).  On (S c)^perp its exact
    // Euclidean norm is ||S c|| ||c||/(c^T S c), so certify the requested
    // bordered tolerance before treating the projected equation as solved.
    const double bordered_amplification =
        overlap_selected.col(state).norm() *
        selected_eigenvectors.col(state).norm() /
        root_metric_norms[state];
    correctable_absolute_targets[state] =
        bordered_absolute_targets[state] / bordered_amplification;
    // MINRES estimates ||r||_{P}, while the candidate contract is Euclidean.
    // Since ||r||_2 <= ||r||_{P}/sqrt(min_i P_ii), this sufficient trigger
    // cannot claim a Euclidean tolerance before a true-residual check.
    preconditioned_absolute_targets[state] =
        correctable_absolute_targets[state] *
        std::sqrt(inverse_preconditioner.col(state).minCoeff());
  }

  for (Eigen::Index state = 0; state < n_selected; ++state) {
    EigenResponseRecycleSpace* recycle = recycle_spaces.empty()
        ? nullptr
        : recycle_spaces[static_cast<std::size_t>(state)];
    bool has_recycled_candidate = false;
    if (recycle != nullptr && recycle->size() > 0) {
      const EigenResponseRecycleApplication application =
          recycle->galerkin_apply(rhs.col(state));
      if (application.available) {
        solution.col(state) = application.solution;
        v_new.col(state) = rhs.col(state) - application.operator_image;
        has_recycled_candidate = true;
      }
    }

    // Reconstruct the initial bordered residual from its complement component.
    // With the finite-Ritz multiplier, the top residual is exactly
    // [I-(S c)c^T/(c^T S c)] e for e in (S c)^perp; the metric gauge is
    // already satisfied by x0 and the projected recycle basis. This screens a
    // cold zero correction and every unconditional Galerkin candidate before
    // MINRES without another H/S action. The independently evaluated final
    // bordered residual remains the authoritative certificate because recycle
    // images carry finite orthogonalization error.
    Eigen::VectorXd bordered_top_residual = v_new.col(state) -
        overlap_selected.col(state) *
            (selected_eigenvectors.col(state).dot(v_new.col(state)) /
             root_metric_norms[state]);
    double gauge_residual = full_rhs(n, state) -
        overlap_selected.col(state).dot(
            solution.col(state) + particular_solution.col(state));
    double bordered_residual_norm = std::hypot(
        bordered_top_residual.norm(), gauge_residual);
    if (bordered_residual_norm <= bordered_absolute_targets[state]) {
      converged[static_cast<std::size_t>(state)] = true;
      v_new.col(state).setZero();
      continue;
    }
    if (has_recycled_candidate &&
        v_new.col(state).norm() >= rhs.col(state).norm()) {
      solution.col(state).setZero();
      v_new.col(state) = rhs.col(state);
      bordered_top_residual = v_new.col(state) -
          overlap_selected.col(state) *
              (selected_eigenvectors.col(state).dot(v_new.col(state)) /
               root_metric_norms[state]);
      gauge_residual = full_rhs(n, state) -
          overlap_selected.col(state).dot(particular_solution.col(state));
      bordered_residual_norm = std::hypot(
          bordered_top_residual.norm(), gauge_residual);
      if (bordered_residual_norm <= bordered_absolute_targets[state]) {
        converged[static_cast<std::size_t>(state)] = true;
        v_new.col(state).setZero();
        continue;
      }
    }
    if (v_new.col(state).norm() == 0.0) {
      std::ostringstream message;
      message << "initial selected-root response has a nonzero bordered residual but zero complement: state="
              << state << " full_residual=" << bordered_residual_norm
              << " full_target=" << bordered_absolute_targets[state]
              << " projected_target=" << correctable_absolute_targets[state]
              << " ritz_residual=" << ritz_residuals.col(state).norm();
      throw std::runtime_error(message.str());
    }
  }

  w_new = apply_response_preconditioner(
      v_new, inverse_preconditioner, recycle_spaces);
  project_selected_roots(&w_new, root_units);
  for (Eigen::Index state = 0; state < n_selected; ++state) {
    if (rhs_norms[state] == 0.0) {
      converged[static_cast<std::size_t>(state)] = true;
      beta_new[state] = 0.0;
      beta_first[state] = 0.0;
      continue;
    }
    if (converged[static_cast<std::size_t>(state)]) {
      beta_new[state] = 0.0;
      beta_first[state] = 0.0;
      residual_norms[state] = 0.0;
      continue;
    }
    const double beta_squared = v_new.col(state).dot(w_new.col(state));
    if (!(beta_squared > 0.0) || !std::isfinite(beta_squared)) {
      throw std::runtime_error(
          "generalized-eigen response preconditioner is not positive definite");
    }
    beta_new[state] = std::sqrt(beta_squared);
    beta_first[state] = beta_new[state];
    // The Lanczos/Givens residual is measured in the inverse-preconditioner
    // metric; an Euclidean RHS norm gives an invalid trigger scale.
    residual_norms[state] = beta_first[state];
  }

  for (int iteration = 0;
       iteration < options.max_iterations &&
       !std::all_of(converged.begin(), converged.end(),
                    [](bool value) { return value; });
       ++iteration) {
    for (Eigen::Index state = 0; state < n_selected; ++state) {
      if (converged[static_cast<std::size_t>(state)]) {
        w_new.col(state).setZero();
        w.col(state).setZero();
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

    const Eigen::MatrixXd operator_images = apply_projected_operators(
        action, selected_eigenvalues, root_units, w,
        &result.block_actions);
    std::vector<Eigen::Index> estimated_converged_states;
    for (Eigen::Index state = 0; state < n_selected; ++state) {
      if (converged[static_cast<std::size_t>(state)]) {
        continue;
      }
      const double beta = beta_new[state];
      v_new.col(state).noalias() =
          operator_images.col(state) - beta * v_old.col(state);
      const double alpha = v_new.col(state).dot(w_new.col(state));
      v_new.col(state).noalias() -= alpha * v.col(state);
      v_new.col(state).noalias() -= root_units.col(state) *
          root_units.col(state).dot(v_new.col(state));
      w_new.col(state) = apply_response_preconditioner_column(
          v_new.col(state), inverse_preconditioner.col(state),
          recycle_spaces.empty()
              ? nullptr
              : recycle_spaces[static_cast<std::size_t>(state)]);
      w_new.col(state).noalias() -= root_units.col(state) *
          root_units.col(state).dot(w_new.col(state));
      const double beta_squared = v_new.col(state).dot(w_new.col(state));
      const double absolute_dot =
          v_new.col(state).cwiseAbs().dot(w_new.col(state).cwiseAbs());
      const double dot_operations = 4.0 * static_cast<double>(n);
      const double unit_roundoff = std::numeric_limits<double>::epsilon();
      const double roundoff_factor =
          dot_operations * unit_roundoff /
          (1.0 - dot_operations * unit_roundoff);
      const double negative_roundoff_bound =
          roundoff_factor * absolute_dot;
      if (!std::isfinite(beta_squared) ||
          beta_squared < -negative_roundoff_bound) {
        std::ostringstream message;
        message << "generalized-eigen response MINRES lost positive "
                   "preconditioner curvature: value=" << beta_squared
                << " roundoff_bound=" << negative_roundoff_bound;
        throw std::runtime_error(message.str());
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
      const bool estimated_converged =
          residual_norms[state] <= preconditioned_absolute_targets[state];
      converged[static_cast<std::size_t>(state)] = estimated_converged;
      if (estimated_converged) {
        estimated_converged_states.push_back(state);
      } else {
        eta[state] = -sine[state] * eta[state];
      }
    }

    if (!estimated_converged_states.empty()) {
      const Eigen::Index n_candidates =
          static_cast<Eigen::Index>(estimated_converged_states.size());
      Eigen::VectorXd candidate_eigenvalues(n_candidates);
      Eigen::MatrixXd candidate_overlap_selected(n, n_candidates);
      Eigen::MatrixXd candidate_solutions(n + 1, n_candidates);
      for (Eigen::Index candidate = 0;
           candidate < n_candidates;
           ++candidate) {
        const Eigen::Index state =
            estimated_converged_states[static_cast<std::size_t>(candidate)];
        const Eigen::VectorXd forcing =
            delta_hamiltonian_selected.col(state) -
            selected_eigenvalues[state] * delta_overlap_selected.col(state);
        const Eigen::VectorXd response =
            solution.col(state) + particular_solution.col(state);
        result.eigenvalue_response[state] =
            (selected_eigenvectors.col(state).dot(forcing) +
             ritz_residuals.col(state).dot(response)) /
            root_metric_norms[state];
        candidate_eigenvalues[candidate] = selected_eigenvalues[state];
        candidate_overlap_selected.col(candidate) =
            overlap_selected.col(state);
        candidate_solutions.col(candidate).head(n) = response;
        candidate_solutions(n, candidate) =
            -result.eigenvalue_response[state];
      }
      const Eigen::MatrixXd checked_images = apply_bordered_operators(
          action, candidate_eigenvalues, candidate_overlap_selected,
          candidate_solutions, &result.block_actions);
      for (Eigen::Index candidate = 0;
           candidate < n_candidates;
           ++candidate) {
        const Eigen::Index state =
            estimated_converged_states[static_cast<std::size_t>(candidate)];
        const Eigen::VectorXd true_residual =
            full_rhs.col(state) - checked_images.col(candidate);
        const Eigen::VectorXd correctable_residual =
            true_residual.head(n) - root_units.col(state) *
                root_units.col(state).dot(true_residual.head(n));
        ++candidate_checks[static_cast<std::size_t>(state)];
        last_candidate_projected_residual[state] =
            correctable_residual.norm();
        least_candidate_projected_residual[state] = std::min(
            least_candidate_projected_residual[state],
            last_candidate_projected_residual[state]);
        last_candidate_bordered_residual[state] = true_residual.norm();
        least_candidate_bordered_residual[state] = std::min(
            least_candidate_bordered_residual[state],
            last_candidate_bordered_residual[state]);
        if (true_residual.norm() <= bordered_absolute_targets[state]) {
          continue;
        }
        if (correctable_residual.norm() == 0.0) {
          std::ostringstream message;
          message << "selected-root response has a nonzero bordered residual "
                     "but zero complement: state=" << state
                  << " full_residual=" << true_residual.norm()
                  << " full_target=" << bordered_absolute_targets[state]
                  << " projected_residual=" << correctable_residual.norm()
                  << " projected_target="
                  << correctable_absolute_targets[state]
                  << " ritz_residual=" << ritz_residuals.col(state).norm()
                  << " ritz_response_coupling="
                  << ritz_residuals.col(state).dot(
                         candidate_solutions.col(candidate).head(n));
          throw std::runtime_error(message.str());
        }

        // Only the original bordered residual certifies a response. Restart
        // its complement component without another H/S action; do not freeze
        // a column merely because MINRES estimated convergence.
        converged[static_cast<std::size_t>(state)] = false;
        ++reliable_restarts[static_cast<std::size_t>(state)];
        v_old.col(state).setZero();
        v.col(state).setZero();
        v_new.col(state) = correctable_residual;
        if (v_new.col(state).norm() == 0.0) {
          throw std::runtime_error(
              "accepted selected-root response residual has no correctable complement component");
        }
        w_new.col(state) = apply_response_preconditioner_column(
            v_new.col(state), inverse_preconditioner.col(state),
            recycle_spaces.empty()
                ? nullptr
                : recycle_spaces[static_cast<std::size_t>(state)]);
        w_new.col(state).noalias() -= root_units.col(state) *
            root_units.col(state).dot(w_new.col(state));
        p_older.col(state).setZero();
        p_old.col(state).setZero();
        p.col(state).setZero();
        const double beta_squared = v_new.col(state).dot(w_new.col(state));
        if (!(beta_squared > 0.0) || !std::isfinite(beta_squared)) {
          throw std::runtime_error(
              "generalized-eigen response reliable restart failed");
        }
        beta_new[state] = std::sqrt(beta_squared);
        beta_first[state] = beta_new[state];
        residual_norms[state] = beta_first[state];
        cosine[state] = 1.0;
        old_cosine[state] = 1.0;
        sine[state] = 0.0;
        old_sine[state] = 0.0;
        eta[state] = 1.0;
      }
    }
    if (std::all_of(converged.begin(), converged.end(), [](bool value) {
          return value;
        })) {
      break;
    }
  }

  result.eigenvector_response.resize(n, n_selected);
  Eigen::MatrixXd bordered_solution(n + 1, n_selected);
  for (Eigen::Index state = 0; state < n_selected; ++state) {
    const Eigen::VectorXd forcing =
        delta_hamiltonian_selected.col(state) -
        selected_eigenvalues[state] * delta_overlap_selected.col(state);
    result.eigenvector_response.col(state) =
        solution.col(state) + particular_solution.col(state);
    result.eigenvalue_response[state] =
        (selected_eigenvectors.col(state).dot(forcing) +
         ritz_residuals.col(state).dot(
             result.eigenvector_response.col(state))) /
        root_metric_norms[state];
    bordered_solution.col(state).head(n) =
        result.eigenvector_response.col(state);
    bordered_solution(n, state) = -result.eigenvalue_response[state];
  }
  const Eigen::MatrixXd final_images = apply_bordered_operators(
      action, selected_eigenvalues, overlap_selected, bordered_solution,
      &result.block_actions);
  result.relative_residual_norms.resize(n_selected);
  for (Eigen::Index state = 0; state < n_selected; ++state) {
    const double original_rhs_norm = full_rhs.col(state).norm();
    const Eigen::VectorXd bordered_residual =
        full_rhs.col(state) - final_images.col(state);
    const Eigen::VectorXd projected_residual =
        bordered_residual.head(n) - root_units.col(state) *
            root_units.col(state).dot(bordered_residual.head(n));
    result.relative_residual_norms[state] = original_rhs_norm == 0.0
        ? bordered_residual.norm()
        : bordered_residual.norm() / original_rhs_norm;
    if (!std::isfinite(result.relative_residual_norms[state]) ||
        bordered_residual.norm() > bordered_absolute_targets[state]) {
      const double root_component = std::abs(
          root_units.col(state).dot(bordered_residual.head(n)));
      std::ostringstream message;
      message << "projected generalized-eigen response MINRES did not reach the requested bordered residual: state="
              << state << " residual="
              << result.relative_residual_norms[state] << " iterations="
              << result.iterations[static_cast<std::size_t>(state)]
              << " full_residual=" << bordered_residual.norm()
              << " full_target=" << bordered_absolute_targets[state]
              << " projected_component=" << projected_residual.norm()
              << " projected_target="
              << correctable_absolute_targets[state]
              << " selected_root_component=" << root_component
              << " response_norm=" <<
                     result.eigenvector_response.col(state).norm()
              << " ritz_residual=" << ritz_residuals.col(state).norm()
              << " ritz_response_coupling=" <<
                     ritz_residuals.col(state).dot(
                         result.eigenvector_response.col(state))
              << " projected_rhs_norm=" << rhs_norms[state]
              << " original_rhs_norm=" << original_rhs_norms[state]
              << " projected_absolute_target=" <<
                     correctable_absolute_targets[state]
              << " candidate_checks=" <<
                     candidate_checks[static_cast<std::size_t>(state)]
              << " reliable_restarts=" <<
                     reliable_restarts[static_cast<std::size_t>(state)]
              << " least_candidate_projected=" <<
                     least_candidate_projected_residual[state]
              << " last_candidate_projected=" <<
                     last_candidate_projected_residual[state]
              << " least_candidate_bordered=" <<
                     least_candidate_bordered_residual[state]
              << " last_candidate_bordered=" <<
                     last_candidate_bordered_residual[state];
      throw std::runtime_error(
          message.str());
    }
  }

  std::vector<Eigen::Index> new_recycle_states;
  for (Eigen::Index state = 0; state < n_selected; ++state) {
    EigenResponseRecycleSpace* recycle = recycle_spaces.empty()
        ? nullptr
        : recycle_spaces[static_cast<std::size_t>(state)];
    if (recycle != nullptr && solution.col(state).norm() > 0.0 &&
        (recycle->size() == 0 ||
         result.iterations[static_cast<std::size_t>(state)] > 0)) {
      new_recycle_states.push_back(state);
    }
  }
  if (!new_recycle_states.empty()) {
    const Eigen::Index count =
        static_cast<Eigen::Index>(new_recycle_states.size());
    Eigen::VectorXd recycle_eigenvalues(count);
    Eigen::MatrixXd recycle_root_units(n, count);
    Eigen::MatrixXd recycle_solutions(n, count);
    for (Eigen::Index column = 0; column < count; ++column) {
      const Eigen::Index state =
          new_recycle_states[static_cast<std::size_t>(column)];
      recycle_eigenvalues[column] = selected_eigenvalues[state];
      recycle_root_units.col(column) = root_units.col(state);
      recycle_solutions.col(column) = solution.col(state);
    }
    const Eigen::MatrixXd recycle_images = apply_projected_operators(
        action, recycle_eigenvalues, recycle_root_units,
        recycle_solutions, &result.block_actions);
    for (Eigen::Index column = 0; column < count; ++column) {
      const Eigen::Index state =
          new_recycle_states[static_cast<std::size_t>(column)];
      recycle_spaces[static_cast<std::size_t>(state)]->append(
          recycle_solutions.col(column), recycle_images.col(column));
    }
  }
  return result;
}

EigenResponseResult evaluate_frozen_generalized_eigen_response(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected,
    const std::vector<const EigenResponseRecycleSpace*>& recycle_spaces,
    const Eigen::MatrixXd* selected_residuals) {
  validate_response_columns(
      selected_eigenvalues, selected_eigenvectors, overlap_selected,
      delta_hamiltonian_selected, delta_overlap_selected);
  const Eigen::Index n = selected_eigenvectors.rows();
  const Eigen::Index n_selected = selected_eigenvalues.size();
  if (recycle_spaces.size() != static_cast<std::size_t>(n_selected)) {
    throw std::invalid_argument(
        "frozen generalized-eigen response recycle-space count is inconsistent");
  }

  EigenResponseResult result;
  result.eigenvector_response = Eigen::MatrixXd::Zero(n, n_selected);
  result.eigenvalue_response.resize(n_selected);
  result.relative_residual_norms.resize(n_selected);
  result.iterations.assign(static_cast<std::size_t>(n_selected), 0);
  const Eigen::MatrixXd ritz_residuals = selected_ritz_residuals(
      action, selected_eigenvalues, selected_eigenvectors,
      selected_residuals, &result.block_actions);
  Eigen::MatrixXd full_rhs(n + 1, n_selected);
  Eigen::MatrixXd bordered_solution(n + 1, n_selected);
  Eigen::VectorXd root_metric_norms(n_selected);
  for (Eigen::Index state = 0; state < n_selected; ++state) {
    const Eigen::VectorXd forcing =
        delta_hamiltonian_selected.col(state) -
        selected_eigenvalues[state] * delta_overlap_selected.col(state);
    root_metric_norms[state] = selected_eigenvectors.col(state).dot(
        overlap_selected.col(state));
    if (!(root_metric_norms[state] > 0.0) ||
        !std::isfinite(root_metric_norms[state])) {
      throw std::invalid_argument(
          "selected generalized-eigen root has no positive overlap norm");
    }
    const double constraint_norm = overlap_selected.col(state).norm();
    if (!(constraint_norm > 0.0) || !std::isfinite(constraint_norm)) {
      throw std::invalid_argument(
          "selected generalized-eigen overlap image is zero");
    }
    const Eigen::VectorXd constraint_unit =
        overlap_selected.col(state) / constraint_norm;
    full_rhs.col(state).head(n) = -forcing;
    full_rhs(n, state) = -0.5 * selected_eigenvectors.col(state).dot(
        delta_overlap_selected.col(state));
    const Eigen::VectorXd particular_solution =
        selected_eigenvectors.col(state) *
        (full_rhs(n, state) / root_metric_norms[state]);
    Eigen::VectorXd projected_rhs = -forcing -
        ritz_residuals.col(state) *
            (full_rhs(n, state) / root_metric_norms[state]);
    projected_rhs.noalias() -=
        constraint_unit * constraint_unit.dot(projected_rhs);

    Eigen::VectorXd external_response = Eigen::VectorXd::Zero(n);
    const EigenResponseRecycleSpace* recycle =
        recycle_spaces[static_cast<std::size_t>(state)];
    if (recycle != nullptr) {
      const EigenResponseRecycleApplication application =
          recycle->galerkin_apply(projected_rhs);
      if (application.available) {
        external_response = application.solution;
      }
    }
    result.eigenvector_response.col(state) =
        external_response + particular_solution;
    result.eigenvalue_response[state] =
        (selected_eigenvectors.col(state).dot(forcing) +
         ritz_residuals.col(state).dot(
             result.eigenvector_response.col(state))) /
        root_metric_norms[state];
    bordered_solution.col(state).head(n) =
        result.eigenvector_response.col(state);
    bordered_solution(n, state) = -result.eigenvalue_response[state];
  }

  const Eigen::MatrixXd images = apply_bordered_operators(
      action, selected_eigenvalues, overlap_selected, bordered_solution,
      &result.block_actions);
  for (Eigen::Index state = 0; state < n_selected; ++state) {
    const double rhs_norm = full_rhs.col(state).norm();
    const double residual_norm =
        (full_rhs.col(state) - images.col(state)).norm();
    result.relative_residual_norms[state] = rhs_norm == 0.0
        ? residual_norm
        : residual_norm / rhs_norm;
    if (!std::isfinite(result.relative_residual_norms[state])) {
      throw std::runtime_error(
          "frozen generalized-eigen response residual is not finite");
    }
  }
  return result;
}

namespace {

struct EqualWeightResponseData {
  Eigen::MatrixXd metric;
  Eigen::MatrixXd forcing;
  Eigen::MatrixXd gauge_target;
};

EqualWeightResponseData prepare_equal_weight_response(
    const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected,
    const EigenResponseOptions& options) {
  validate_inputs(
      hamiltonian_diagonal,
      overlap_diagonal,
      selected_eigenvalues,
      selected_eigenvectors,
      overlap_selected,
      delta_hamiltonian_selected,
      delta_overlap_selected,
      options);
  EqualWeightResponseData data;
  data.metric = selected_eigenvectors.transpose() * overlap_selected;
  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(
      data.metric.rows(), data.metric.cols());
  const double metric_error = (data.metric - identity).norm();
  const double metric_tolerance = 1.0e3 *
      std::numeric_limits<double>::epsilon() *
      std::max(1.0, static_cast<double>(selected_eigenvectors.rows()));
  if (metric_error > metric_tolerance) {
    std::ostringstream message;
    message << "equal-weight response requires S-orthonormal selected roots: error="
            << metric_error;
    throw std::invalid_argument(message.str());
  }

  data.forcing = delta_hamiltonian_selected -
      delta_overlap_selected * selected_eigenvalues.asDiagonal();
  Eigen::MatrixXd metric_derivative =
      selected_eigenvectors.transpose() * delta_overlap_selected;
  const double antisymmetric_error =
      (metric_derivative - metric_derivative.transpose()).norm();
  const double derivative_scale = std::max(1.0, metric_derivative.norm());
  if (antisymmetric_error > 1.0e-10 * derivative_scale) {
    throw std::invalid_argument(
        "equal-weight response requires one symmetric overlap derivative for the complete selected cluster");
  }
  metric_derivative =
      0.5 * (metric_derivative + metric_derivative.transpose()).eval();
  data.gauge_target = -0.5 * metric_derivative;
  return data;
}

Eigen::MatrixXd selected_span_units(
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors) {
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(selected_eigenvectors);
  qr.setThreshold(std::numeric_limits<double>::epsilon() *
      std::max(selected_eigenvectors.rows(), selected_eigenvectors.cols()));
  if (qr.rank() != selected_eigenvectors.cols()) {
    throw std::invalid_argument(
        "equal-weight response selected roots are linearly dependent");
  }
  return qr.householderQ() * Eigen::MatrixXd::Identity(
      selected_eigenvectors.rows(), selected_eigenvectors.cols());
}

double selected_oblique_amplification(
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& metric_image_units) {
  const Eigen::MatrixXd root_units =
      selected_span_units(selected_eigenvectors);
  const Eigen::MatrixXd cross =
      metric_image_units.transpose() * root_units;
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(cross);
  const double sigma_min = svd.singularValues().minCoeff();
  if (!(sigma_min > 0.0) || !std::isfinite(sigma_min)) {
    throw std::invalid_argument(
        "equal-weight response selected and metric-image spans are orthogonal");
  }
  return 1.0 / sigma_min;
}

void project_selected_span(
    Eigen::MatrixXd* vectors,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_units) {
  vectors->noalias() -= selected_units *
      (selected_units.transpose() * (*vectors));
}

Eigen::MatrixXd apply_cluster_projected_operators(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_units,
    const Eigen::Ref<const Eigen::MatrixXd>& vectors,
    int* block_actions) {
  Eigen::MatrixXd projected = vectors;
  project_selected_span(&projected, selected_units);
  const GeneralizedEigenActionResult images = apply_checked(
      action, projected, block_actions);
  Eigen::MatrixXd result(vectors.rows(), vectors.cols());
  for (Eigen::Index state = 0; state < vectors.cols(); ++state) {
    result.col(state).noalias() = images.hamiltonian.col(state) -
        eigenvalues[state] * images.overlap.col(state);
  }
  project_selected_span(&result, selected_units);
  return result;
}

EigenSubspaceResponseResult finish_equal_weight_response(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& forcing,
    const Eigen::Ref<const Eigen::MatrixXd>& gauge_target,
    const Eigen::Ref<const Eigen::MatrixXd>& external_response,
    double relative_residual_tolerance,
    bool enforce_residual_tolerance,
    std::vector<int> iterations,
    int block_actions,
    const std::vector<EigenResponseRecycleSpace*>& recycle_spaces = {}) {
  const GeneralizedEigenActionResult external_images = apply_checked(
      action, external_response, &block_actions);
  if (!recycle_spaces.empty()) {
    Eigen::MatrixXd projected_images = external_images.hamiltonian -
        external_images.overlap * selected_eigenvalues.asDiagonal();
    const Eigen::MatrixXd selected_units =
        selected_span_units(overlap_selected);
    project_selected_span(&projected_images, selected_units);
    for (Eigen::Index state = 0;
         state < selected_eigenvalues.size(); ++state) {
      if (recycle_spaces[static_cast<std::size_t>(state)] != nullptr &&
          external_response.col(state).norm() > 0.0) {
        recycle_spaces[static_cast<std::size_t>(state)]->append(
            external_response.col(state), projected_images.col(state));
      }
    }
  }
  const Eigen::MatrixXd metric =
      selected_eigenvectors.transpose() * overlap_selected;
  const Eigen::MatrixXd internal_coefficients = metric.partialPivLu().solve(
      gauge_target -
      selected_eigenvectors.transpose() * external_images.overlap);

  EigenSubspaceResponseResult result;
  result.eigenvector_response = external_response +
      selected_eigenvectors * internal_coefficients;
  const GeneralizedEigenActionResult response_images = apply_checked(
      action, result.eigenvector_response, &block_actions);
  Eigen::MatrixXd shifted_response = response_images.hamiltonian -
      response_images.overlap * selected_eigenvalues.asDiagonal();
  result.selected_matrix_response = metric.partialPivLu().solve(
      selected_eigenvectors.transpose() * (forcing + shifted_response));

  const Eigen::MatrixXd equation_residual = forcing + shifted_response -
      overlap_selected * result.selected_matrix_response;
  const Eigen::MatrixXd gauge_residual =
      selected_eigenvectors.transpose() * response_images.overlap -
      gauge_target;
  result.relative_residual_norms.resize(selected_eigenvalues.size());
  result.iterations = std::move(iterations);
  result.block_actions = block_actions;
  for (Eigen::Index state = 0;
       state < selected_eigenvalues.size();
       ++state) {
    const double rhs_norm = std::hypot(
        forcing.col(state).norm(), gauge_target.col(state).norm());
    const double residual_norm = std::hypot(
        equation_residual.col(state).norm(),
        gauge_residual.col(state).norm());
    result.relative_residual_norms[state] = rhs_norm == 0.0
        ? residual_norm
        : residual_norm / rhs_norm;
    if (!std::isfinite(result.relative_residual_norms[state])) {
      throw std::runtime_error(
          "equal-weight generalized-eigen subspace response residual is not finite");
    }
    if (enforce_residual_tolerance &&
        result.relative_residual_norms[state] > relative_residual_tolerance) {
      std::ostringstream message;
      message << "equal-weight generalized-eigen subspace response did not reach the requested residual: state="
              << state << " relative_residual="
              << result.relative_residual_norms[state];
      throw std::runtime_error(message.str());
    }
  }
  return result;
}

}  // namespace

EigenSubspaceResponseResult
solve_equal_weight_generalized_eigen_subspace_response_from_full_spectrum(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& full_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& full_eigenvectors,
    const std::vector<int>& selected_root_indices,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected,
    double relative_residual_tolerance) {
  const Eigen::Index n = full_eigenvalues.size();
  const Eigen::Index n_selected = selected_eigenvalues.size();
  if (n <= 0 || n_selected <= 0 || full_eigenvectors.rows() != n ||
      full_eigenvectors.cols() != n ||
      selected_root_indices.size() !=
          static_cast<std::size_t>(n_selected) ||
      selected_eigenvectors.rows() != n ||
      selected_eigenvectors.cols() != n_selected ||
      overlap_selected.rows() != n ||
      overlap_selected.cols() != n_selected ||
      delta_hamiltonian_selected.rows() != n ||
      delta_hamiltonian_selected.cols() != n_selected ||
      delta_overlap_selected.rows() != n ||
      delta_overlap_selected.cols() != n_selected ||
      !full_eigenvalues.allFinite() || !full_eigenvectors.allFinite() ||
      !selected_eigenvalues.allFinite() ||
      !selected_eigenvectors.allFinite() ||
      !overlap_selected.allFinite() ||
      !delta_hamiltonian_selected.allFinite() ||
      !delta_overlap_selected.allFinite() ||
      !std::isfinite(relative_residual_tolerance) ||
      !(relative_residual_tolerance > 0.0)) {
    throw std::invalid_argument(
        "complete equal-weight subspace response inputs are inconsistent");
  }

  const EigenResponseOptions validation_options{
      1, relative_residual_tolerance};
  const EqualWeightResponseData data = prepare_equal_weight_response(
      Eigen::VectorXd::Ones(n), Eigen::VectorXd::Ones(n),
      selected_eigenvalues, selected_eigenvectors, overlap_selected,
      delta_hamiltonian_selected, delta_overlap_selected,
      validation_options);

  std::vector<bool> selected(static_cast<std::size_t>(n), false);
  for (Eigen::Index state = 0; state < n_selected; ++state) {
    const int root = selected_root_indices[static_cast<std::size_t>(state)];
    if (root < 0 || root >= n || selected[static_cast<std::size_t>(root)] ||
        selected_eigenvalues[state] != full_eigenvalues[root]) {
      throw std::invalid_argument(
          "selected cluster does not match the complete eigenspectrum");
    }
    selected[static_cast<std::size_t>(root)] = true;
  }

  Eigen::MatrixXd coefficients =
      full_eigenvectors.transpose() * data.forcing;
  for (Eigen::Index state = 0; state < n_selected; ++state) {
    for (Eigen::Index other = 0; other < n; ++other) {
      if (selected[static_cast<std::size_t>(other)]) {
        coefficients(other, state) = 0.0;
        continue;
      }
      const double gap =
          selected_eigenvalues[state] - full_eigenvalues[other];
      const double scale = std::max({
          1.0, std::abs(selected_eigenvalues[state]),
          std::abs(full_eigenvalues[other])});
      if (std::abs(gap) <=
          std::numeric_limits<double>::epsilon() * scale) {
        throw std::runtime_error(
            "a degenerate root omitted from the equal-weight cluster makes its response undefined");
      }
      coefficients(other, state) /= gap;
    }
  }
  const Eigen::MatrixXd external_response =
      full_eigenvectors * coefficients;
  return finish_equal_weight_response(
      action, selected_eigenvalues, selected_eigenvectors,
      overlap_selected, data.forcing, data.gauge_target,
      external_response, relative_residual_tolerance, true,
      std::vector<int>(static_cast<std::size_t>(n_selected), 0), 0);
}

EigenSubspaceResponseResult
solve_equal_weight_generalized_eigen_subspace_response(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected,
    const EigenResponseOptions& options,
    const std::vector<EigenResponseRecycleSpace*>& recycle_spaces,
    const Eigen::MatrixXd* selected_residuals) {
  const EqualWeightResponseData data = prepare_equal_weight_response(
      hamiltonian_diagonal, overlap_diagonal, selected_eigenvalues,
      selected_eigenvectors, overlap_selected,
      delta_hamiltonian_selected, delta_overlap_selected, options);
  const Eigen::Index n = hamiltonian_diagonal.size();
  const Eigen::Index n_selected = selected_eigenvalues.size();
  if (!recycle_spaces.empty() &&
      recycle_spaces.size() != static_cast<std::size_t>(n_selected)) {
    throw std::invalid_argument(
        "equal-weight response recycle-space count is inconsistent");
  }
  const Eigen::MatrixXd selected_units =
      selected_span_units(overlap_selected);
  const double bordered_amplification = selected_oblique_amplification(
      selected_eigenvectors, selected_units);

  int block_actions = 0;
  const Eigen::MatrixXd ritz_residuals = selected_ritz_residuals(
      action, selected_eigenvalues, selected_eigenvectors,
      selected_residuals, &block_actions);
  const Eigen::MatrixXd internal_particular =
      data.metric.partialPivLu().solve(data.gauge_target);
  Eigen::MatrixXd rhs =
      -data.forcing - ritz_residuals * internal_particular;
  project_selected_span(&rhs, selected_units);
  const Eigen::VectorXd selected_projector_diagonal =
      selected_units.array().square().rowwise().sum().matrix();
  const Eigen::MatrixXd inverse_preconditioner = build_inverse_preconditioner(
      hamiltonian_diagonal, overlap_diagonal, selected_eigenvalues,
      selected_projector_diagonal.replicate(1, n_selected));
  Eigen::MatrixXd solution = Eigen::MatrixXd::Zero(n, n_selected);
  Eigen::MatrixXd v_old = Eigen::MatrixXd::Zero(n, n_selected);
  Eigen::MatrixXd v = Eigen::MatrixXd::Zero(n, n_selected);
  Eigen::MatrixXd v_new = rhs;
  Eigen::MatrixXd w = Eigen::MatrixXd::Zero(n, n_selected);
  std::vector<bool> initial_converged(
      static_cast<std::size_t>(n_selected), false);
  for (Eigen::Index state = 0; state < n_selected; ++state) {
    EigenResponseRecycleSpace* recycle = recycle_spaces.empty()
        ? nullptr
        : recycle_spaces[static_cast<std::size_t>(state)];
    bool has_recycled_candidate = false;
    if (recycle != nullptr && recycle->size() > 0) {
      const EigenResponseRecycleApplication application =
          recycle->galerkin_apply(rhs.col(state));
      if (application.available) {
        solution.col(state) = application.solution;
        v_new.col(state) = rhs.col(state) - application.operator_image;
        has_recycled_candidate = true;
      }
    }
    // This fixed-space reconstruction is an initial scheduling check. The
    // fresh bordered residual evaluated by finish_equal_weight_response is the
    // authoritative certificate.
    auto full_residual_norm = [&]() {
      const Eigen::VectorXd multiplier_component =
          data.metric.partialPivLu().solve(
              selected_eigenvectors.transpose() * v_new.col(state));
      return (v_new.col(state) -
              overlap_selected * multiplier_component).norm();
    };
    const double full_rhs_norm = std::hypot(
        data.forcing.col(state).norm(),
        data.gauge_target.col(state).norm());
    const double full_target =
        options.relative_residual_tolerance * full_rhs_norm;
    if (full_residual_norm() <= full_target) {
      initial_converged[static_cast<std::size_t>(state)] = true;
      v_new.col(state).setZero();
      continue;
    }
    if (has_recycled_candidate &&
        v_new.col(state).norm() >= rhs.col(state).norm()) {
      solution.col(state).setZero();
      v_new.col(state) = rhs.col(state);
      if (full_residual_norm() <= full_target) {
        initial_converged[static_cast<std::size_t>(state)] = true;
        v_new.col(state).setZero();
        continue;
      }
    }
    if (v_new.col(state).norm() == 0.0) {
      throw std::runtime_error(
          "initial equal-weight response has a nonzero bordered residual but zero external complement");
    }
  }
  Eigen::MatrixXd w_new = apply_response_preconditioner(
      v_new, inverse_preconditioner, recycle_spaces);
  project_selected_span(&w_new, selected_units);
  Eigen::MatrixXd p_older = Eigen::MatrixXd::Zero(n, n_selected);
  Eigen::MatrixXd p_old = Eigen::MatrixXd::Zero(n, n_selected);
  Eigen::MatrixXd p = Eigen::MatrixXd::Zero(n, n_selected);

  const Eigen::VectorXd rhs_norms = rhs.colwise().norm();
  Eigen::VectorXd absolute_targets(n_selected);
  Eigen::VectorXd estimated_targets(n_selected);
  Eigen::VectorXd residual_norms(n_selected);
  Eigen::VectorXd beta_new(n_selected);
  Eigen::VectorXd beta_first(n_selected);
  Eigen::VectorXd cosine = Eigen::VectorXd::Ones(n_selected);
  Eigen::VectorXd old_cosine = Eigen::VectorXd::Ones(n_selected);
  Eigen::VectorXd sine = Eigen::VectorXd::Zero(n_selected);
  Eigen::VectorXd old_sine = Eigen::VectorXd::Zero(n_selected);
  Eigen::VectorXd eta = Eigen::VectorXd::Ones(n_selected);
  std::vector<int> iterations(static_cast<std::size_t>(n_selected), 0);
  std::vector<bool> converged = std::move(initial_converged);
  for (Eigen::Index state = 0; state < n_selected; ++state) {
    const double full_rhs_norm = std::hypot(
        data.forcing.col(state).norm(),
        data.gauge_target.col(state).norm());
    const double full_target =
        options.relative_residual_tolerance * full_rhs_norm;
    // The contract is relative to the original bordered RHS. Cancellation
    // under projection must not impose a second, arbitrarily tighter target.
    absolute_targets[state] = full_target / bordered_amplification;
    estimated_targets[state] = absolute_targets[state] *
        std::sqrt(inverse_preconditioner.col(state).minCoeff());
    if (rhs_norms[state] == 0.0) {
      converged[static_cast<std::size_t>(state)] = true;
      beta_new[state] = 0.0;
      beta_first[state] = 0.0;
      continue;
    }
    if (v_new.col(state).norm() <= absolute_targets[state]) {
      converged[static_cast<std::size_t>(state)] = true;
      beta_new[state] = 0.0;
      beta_first[state] = 0.0;
      residual_norms[state] = 0.0;
      continue;
    }
    const double beta_squared = v_new.col(state).dot(w_new.col(state));
    if (!(beta_squared > 0.0) || !std::isfinite(beta_squared)) {
      throw std::runtime_error(
          "equal-weight response preconditioner is not positive definite");
    }
    beta_new[state] = std::sqrt(beta_squared);
    beta_first[state] = beta_new[state];
    residual_norms[state] = beta_first[state];
  }

  for (int iteration = 0;
       iteration < options.max_iterations &&
       !std::all_of(converged.begin(), converged.end(),
                    [](bool value) { return value; });
       ++iteration) {
    for (Eigen::Index state = 0; state < n_selected; ++state) {
      if (converged[static_cast<std::size_t>(state)]) {
        w.col(state).setZero();
        w_new.col(state).setZero();
        continue;
      }
      if (!(beta_new[state] > 0.0) || !std::isfinite(beta_new[state])) {
        throw std::runtime_error(
            "equal-weight response MINRES encountered a Lanczos breakdown");
      }
      v_old.col(state) = v.col(state);
      v_new.col(state) /= beta_new[state];
      w_new.col(state) /= beta_new[state];
      v.col(state) = v_new.col(state);
      w.col(state) = w_new.col(state);
    }
    const Eigen::MatrixXd images = apply_cluster_projected_operators(
        action, selected_eigenvalues, selected_units, w, &block_actions);
    std::vector<Eigen::Index> candidates;
    for (Eigen::Index state = 0; state < n_selected; ++state) {
      if (converged[static_cast<std::size_t>(state)]) continue;
      const double beta = beta_new[state];
      v_new.col(state).noalias() =
          images.col(state) - beta * v_old.col(state);
      const double alpha = v_new.col(state).dot(w_new.col(state));
      v_new.col(state).noalias() -= alpha * v.col(state);
      project_selected_span(&v_new, selected_units);
      w_new.col(state) = apply_response_preconditioner_column(
          v_new.col(state), inverse_preconditioner.col(state),
          recycle_spaces.empty()
              ? nullptr
              : recycle_spaces[static_cast<std::size_t>(state)]);
      Eigen::MatrixXd projected_preconditioned = w_new.middleCols(state, 1);
      project_selected_span(&projected_preconditioned, selected_units);
      w_new.col(state) = projected_preconditioned.col(0);
      const double beta_squared = v_new.col(state).dot(w_new.col(state));
      const double roundoff_bound = 16.0 * static_cast<double>(n) *
          std::numeric_limits<double>::epsilon() *
          v_new.col(state).cwiseAbs().dot(w_new.col(state).cwiseAbs());
      if (!std::isfinite(beta_squared) || beta_squared < -roundoff_bound) {
        throw std::runtime_error(
            "equal-weight response MINRES lost preconditioner curvature");
      }
      beta_new[state] = std::sqrt(std::max(0.0, beta_squared));

      const double r2 = sine[state] * alpha +
          cosine[state] * old_cosine[state] * beta;
      const double r3 = old_sine[state] * beta;
      const double r1_head = cosine[state] * alpha -
          old_cosine[state] * sine[state] * beta;
      const double r1 = std::hypot(r1_head, beta_new[state]);
      if (!(r1 > 0.0) || !std::isfinite(r1)) {
        throw std::runtime_error(
            "equal-weight response MINRES encountered a singular rotation");
      }
      old_cosine[state] = cosine[state];
      old_sine[state] = sine[state];
      cosine[state] = r1_head / r1;
      sine[state] = beta_new[state] / r1;
      p_older.col(state) = p_old.col(state);
      p_old.col(state) = p.col(state);
      p.col(state).noalias() =
          (w.col(state) - r2 * p_old.col(state) -
           r3 * p_older.col(state)) / r1;
      solution.col(state).noalias() +=
          beta_first[state] * cosine[state] * eta[state] * p.col(state);
      residual_norms[state] *= std::abs(sine[state]);
      iterations[static_cast<std::size_t>(state)] = iteration + 1;
      if (residual_norms[state] <= estimated_targets[state]) {
        candidates.push_back(state);
        converged[static_cast<std::size_t>(state)] = true;
      } else {
        eta[state] = -sine[state] * eta[state];
      }
    }

    if (!candidates.empty()) {
      Eigen::VectorXd candidate_energies(candidates.size());
      Eigen::MatrixXd candidate_solutions(n, candidates.size());
      for (Eigen::Index candidate = 0;
           candidate < static_cast<Eigen::Index>(candidates.size());
           ++candidate) {
        candidate_energies[candidate] =
            selected_eigenvalues[candidates[static_cast<std::size_t>(candidate)]];
        candidate_solutions.col(candidate) =
            solution.col(candidates[static_cast<std::size_t>(candidate)]);
      }
      const Eigen::MatrixXd candidate_images =
          apply_cluster_projected_operators(
              action, candidate_energies, selected_units,
              candidate_solutions, &block_actions);
      for (Eigen::Index candidate = 0;
           candidate < static_cast<Eigen::Index>(candidates.size());
           ++candidate) {
        const Eigen::Index state =
            candidates[static_cast<std::size_t>(candidate)];
        Eigen::VectorXd true_residual =
            rhs.col(state) - candidate_images.col(candidate);
        if (true_residual.norm() <= absolute_targets[state]) continue;
        converged[static_cast<std::size_t>(state)] = false;
        v_old.col(state).setZero();
        v.col(state).setZero();
        v_new.col(state) = true_residual;
        w_new.col(state) = apply_response_preconditioner_column(
            true_residual, inverse_preconditioner.col(state),
            recycle_spaces.empty()
                ? nullptr
                : recycle_spaces[static_cast<std::size_t>(state)]);
        Eigen::MatrixXd projected_preconditioned =
            w_new.middleCols(state, 1);
        project_selected_span(&projected_preconditioned, selected_units);
        w_new.col(state) = projected_preconditioned.col(0);
        p_older.col(state).setZero();
        p_old.col(state).setZero();
        p.col(state).setZero();
        const double beta_squared = v_new.col(state).dot(w_new.col(state));
        if (!(beta_squared > 0.0) || !std::isfinite(beta_squared)) {
          std::ostringstream message;
          message << "equal-weight response reliable restart failed: state="
                  << state << " projected_residual=" << true_residual.norm()
                  << " projected_target=" << absolute_targets[state]
                  << " projected_rhs_norm=" << rhs_norms[state]
                  << " preconditioned_norm_squared=" << beta_squared;
          throw std::runtime_error(message.str());
        }
        beta_new[state] = std::sqrt(beta_squared);
        beta_first[state] = beta_new[state];
        residual_norms[state] = beta_first[state];
        cosine[state] = 1.0;
        old_cosine[state] = 1.0;
        sine[state] = 0.0;
        old_sine[state] = 0.0;
        eta[state] = 1.0;
      }
    }
    if (std::all_of(converged.begin(), converged.end(), [](bool value) {
          return value;
        })) {
      break;
    }
  }

  if (!std::all_of(converged.begin(), converged.end(), [](bool value) {
        return value;
      })) {
    throw std::runtime_error(
        "equal-weight generalized-eigen subspace response exhausted its iteration budget");
  }

  return finish_equal_weight_response(
      action, selected_eigenvalues, selected_eigenvectors,
      overlap_selected, data.forcing, data.gauge_target, solution,
      options.relative_residual_tolerance, true, std::move(iterations),
      block_actions, recycle_spaces);
}

EigenSubspaceResponseResult
evaluate_frozen_equal_weight_generalized_eigen_subspace_response(
    const GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& selected_eigenvalues,
    const Eigen::Ref<const Eigen::MatrixXd>& selected_eigenvectors,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_hamiltonian_selected,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_overlap_selected,
    const std::vector<const EigenResponseRecycleSpace*>& recycle_spaces,
    const Eigen::MatrixXd* selected_residuals) {
  validate_response_columns(
      selected_eigenvalues, selected_eigenvectors, overlap_selected,
      delta_hamiltonian_selected, delta_overlap_selected);
  const Eigen::Index n = selected_eigenvectors.rows();
  const Eigen::Index n_selected = selected_eigenvalues.size();
  if (recycle_spaces.size() != static_cast<std::size_t>(n_selected)) {
    throw std::invalid_argument(
        "frozen equal-weight response recycle-space count is inconsistent");
  }
  const EigenResponseOptions validation_options{1, 1.0};
  const EqualWeightResponseData data = prepare_equal_weight_response(
      Eigen::VectorXd::Ones(n), Eigen::VectorXd::Ones(n),
      selected_eigenvalues, selected_eigenvectors, overlap_selected,
      delta_hamiltonian_selected, delta_overlap_selected,
      validation_options);
  const Eigen::MatrixXd selected_units =
      selected_span_units(overlap_selected);
  int block_actions = 0;
  const Eigen::MatrixXd ritz_residuals = selected_ritz_residuals(
      action, selected_eigenvalues, selected_eigenvectors,
      selected_residuals, &block_actions);
  const Eigen::MatrixXd internal_particular =
      data.metric.partialPivLu().solve(data.gauge_target);
  Eigen::MatrixXd rhs =
      -data.forcing - ritz_residuals * internal_particular;
  project_selected_span(&rhs, selected_units);
  Eigen::MatrixXd external_response = Eigen::MatrixXd::Zero(n, n_selected);
  for (Eigen::Index state = 0; state < n_selected; ++state) {
    const EigenResponseRecycleSpace* recycle =
        recycle_spaces[static_cast<std::size_t>(state)];
    if (recycle == nullptr) continue;
    const EigenResponseRecycleApplication application =
        recycle->galerkin_apply(rhs.col(state));
    if (application.available) {
      external_response.col(state) = application.solution;
    }
  }
  return finish_equal_weight_response(
      action, selected_eigenvalues, selected_eigenvectors,
      overlap_selected, data.forcing, data.gauge_target,
      external_response, 1.0, false,
      std::vector<int>(static_cast<std::size_t>(n_selected), 0),
      block_actions);
}

}  // namespace xmvb::core
