#include "vb/biorthogonal_vbscf/biorthogonal_orbital_frame.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <Eigen/Cholesky>

#include "vb/biorthogonal_vbscf/biorthogonal_structure_local_utils.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

namespace {

}  // namespace

BiorthogonalOrbitalFrame build_biorthogonal_orbital_frame(
    const Eigen::Ref<const Eigen::MatrixXd>& ao_overlap_matrix,
    const Eigen::Ref<const Eigen::MatrixXd>& right_orbitals) {
  if (ao_overlap_matrix.rows() <= 0 || ao_overlap_matrix.cols() <= 0) {
    throw std::invalid_argument("ao_overlap_matrix must be non-empty");
  }
  if (ao_overlap_matrix.rows() != ao_overlap_matrix.cols()) {
    throw std::invalid_argument("ao_overlap_matrix must be square");
  }
  if (right_orbitals.rows() != ao_overlap_matrix.rows() ||
      right_orbitals.cols() <= 0) {
    throw std::invalid_argument(
        "right_orbitals shape must be (n_ao, n_orbitals) with matching n_ao");
  }
  throw_if_nonfinite(ao_overlap_matrix, "ao overlap matrix");
  throw_if_nonfinite(right_orbitals, "right orbitals");

  BiorthogonalOrbitalFrame frame;
  frame.right_orbitals = right_orbitals;
  frame.right_orbital_overlap =
      right_orbitals.transpose() * ao_overlap_matrix * right_orbitals;
  throw_if_nonfinite(frame.right_orbital_overlap, "right orbital overlap");

  // The dual-orbital map uses X^{-1} with X = C^T S C. The prototype expects
  // the chosen right orbitals to remain linearly independent in the AO metric,
  // so this factorization is the first hard numerical gate of the method.
  Eigen::LDLT<Eigen::MatrixXd> overlap_factor(frame.right_orbital_overlap);
  if (overlap_factor.info() != Eigen::Success) {
    throw std::runtime_error("right orbital overlap factorization failed");
  }
  const Eigen::VectorXd overlap_diagonal = overlap_factor.vectorD();
  if (overlap_diagonal.size() != frame.right_orbital_overlap.rows()) {
    throw std::runtime_error("right orbital overlap factorization diagonal mismatch");
  }
  if ((overlap_diagonal.array() <= 0.0).any()) {
    throw std::runtime_error(
        "right orbital overlap is not positive definite in the AO metric");
  }

  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(
      right_orbitals.cols(),
      right_orbitals.cols());
      
  frame.right_orbital_overlap_inverse = overlap_factor.solve(identity);
  throw_if_nonfinite(
      frame.right_orbital_overlap_inverse,
      "right orbital overlap inverse");

  frame.left_dual_orbitals =
      right_orbitals * frame.right_orbital_overlap_inverse;
  throw_if_nonfinite(frame.left_dual_orbitals, "left dual orbitals");

  const Eigen::MatrixXd residual =
      frame.left_dual_orbitals.transpose() * ao_overlap_matrix * right_orbitals -
      identity;
  frame.biorthogonality_residual_frobenius_norm = residual.norm();
  if (!std::isfinite(frame.biorthogonality_residual_frobenius_norm)) {
    throw std::runtime_error("biorthogonality residual is not finite");
  }

  return frame;
}

void validate_biorthogonal_orbital_frame(
    const BiorthogonalOrbitalFrame& orbital_frame,
    double residual_tolerance) {
  if (residual_tolerance < 0.0) {
    throw std::invalid_argument("residual_tolerance must be non-negative");
  }
  if (orbital_frame.right_orbitals.rows() <= 0 ||
      orbital_frame.right_orbitals.cols() <= 0) {
    throw std::invalid_argument("right_orbitals must be non-empty");
  }
  if (orbital_frame.right_orbital_overlap.rows() !=
          orbital_frame.right_orbital_overlap.cols() ||
      orbital_frame.right_orbital_overlap.rows() !=
          orbital_frame.right_orbitals.cols()) {
    throw std::invalid_argument("right_orbital_overlap dimensions are inconsistent");
  }
  if (orbital_frame.right_orbital_overlap_inverse.rows() !=
          orbital_frame.right_orbital_overlap.rows() ||
      orbital_frame.right_orbital_overlap_inverse.cols() !=
          orbital_frame.right_orbital_overlap.cols()) {
    throw std::invalid_argument(
        "right_orbital_overlap_inverse dimensions are inconsistent");
  }
  if (orbital_frame.left_dual_orbitals.rows() != orbital_frame.right_orbitals.rows() ||
      orbital_frame.left_dual_orbitals.cols() != orbital_frame.right_orbitals.cols()) {
    throw std::invalid_argument("left_dual_orbitals dimensions are inconsistent");
  }
  throw_if_nonfinite(orbital_frame.right_orbitals, "right_orbitals");
  throw_if_nonfinite(orbital_frame.right_orbital_overlap, "right_orbital_overlap");
  throw_if_nonfinite(
      orbital_frame.right_orbital_overlap_inverse,
      "right_orbital_overlap_inverse");
  throw_if_nonfinite(orbital_frame.left_dual_orbitals, "left_dual_orbitals");
  if (!std::isfinite(orbital_frame.biorthogonality_residual_frobenius_norm)) {
    throw std::invalid_argument("biorthogonality residual must be finite");
  }
  if (orbital_frame.biorthogonality_residual_frobenius_norm > residual_tolerance) {
    throw std::runtime_error("biorthogonality residual exceeds tolerance");
  }
}

}  // namespace xmvb::vb::biorthogonal_vbscf
