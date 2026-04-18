#include "vb/biorthogonal_vbscf/biorthogonal_orbital_integrals.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include <Eigen/Cholesky>

#include "vb/biorthogonal_vbscf/biorthogonal_structure_local_utils.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

namespace {

void validate_orbital_index(
    int orbital_index,
    int n_orbitals,
    const char* label) {
  if (orbital_index < 0 || orbital_index >= n_orbitals) {
    throw std::out_of_range(std::string(label) + " is out of range");
  }
}

BiorthogonalOrbitalIntegrals build_from_overlap_and_one_electron(
    const Eigen::Ref<const Eigen::MatrixXd>& right_orbital_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& right_right_one_electron) {
  if (right_orbital_overlap.rows() <= 0 || right_orbital_overlap.cols() <= 0) {
    throw std::invalid_argument("right_orbital_overlap must be non-empty");
  }
  if (right_orbital_overlap.rows() != right_orbital_overlap.cols()) {
    throw std::invalid_argument("right_orbital_overlap must be square");
  }
  if (right_right_one_electron.rows() != right_orbital_overlap.rows() ||
      right_right_one_electron.cols() != right_orbital_overlap.cols()) {
    throw std::invalid_argument(
        "right_right_one_electron must match right_orbital_overlap dimensions");
  }
  throw_if_nonfinite(right_orbital_overlap, "right_orbital_overlap");
  throw_if_nonfinite(right_right_one_electron, "right_right_one_electron");

  // In the one-sided biorthogonal prototype the entire dual transformation is
  // governed by the active-space orbital overlap `X = C^T S C`. Once `X^{-1}`
  // is available, both the left/right one-electron matrix and all mixed 2e
  // integrals can be assembled without revisiting the AO-side orbital table.
  Eigen::LLT<Eigen::MatrixXd> overlap_factor(right_orbital_overlap);
  if (overlap_factor.info() != Eigen::Success) {
    throw std::runtime_error("right_orbital_overlap is not positive definite");
  }

  BiorthogonalOrbitalIntegrals orbital_integrals;
  orbital_integrals.n_orbitals = right_orbital_overlap.rows();
  orbital_integrals.left_dual_from_right_transform =
      overlap_factor.solve(Eigen::MatrixXd::Identity(
          orbital_integrals.n_orbitals,
          orbital_integrals.n_orbitals));
  orbital_integrals.left_right_one_electron =
      orbital_integrals.left_dual_from_right_transform *
      right_right_one_electron;
  validate_biorthogonal_orbital_integrals(orbital_integrals);
  return orbital_integrals;
}

}  // namespace

BiorthogonalOrbitalIntegrals build_biorthogonal_orbital_integrals(
    const BiorthogonalOrbitalFrame& orbital_frame,
    const std::vector<double>& right_right_one_electron) {
  validate_biorthogonal_orbital_frame(orbital_frame, 1.0e-10);

  const int n_orbitals = orbital_frame.right_orbitals.cols();
  const std::size_t expected_size =
      static_cast<std::size_t>(n_orbitals) * static_cast<std::size_t>(n_orbitals);
  if (right_right_one_electron.size() != expected_size) {
    throw std::invalid_argument(
        "right_right_one_electron size does not match orbital_frame dimensions");
  }

  const Eigen::Map<const Eigen::MatrixXd> right_right_one_electron_matrix(
      right_right_one_electron.data(),
      n_orbitals,
      n_orbitals);

  return build_from_overlap_and_one_electron(
      orbital_frame.right_orbital_overlap,
      right_right_one_electron_matrix);
}

BiorthogonalOrbitalIntegrals build_biorthogonal_orbital_integrals(
    int n_orbitals,
    const std::vector<double>& right_orbital_overlap,
    const std::vector<double>& right_right_one_electron) {
  if (n_orbitals <= 0) {
    throw std::invalid_argument("n_orbitals must be positive");
  }
  const std::size_t expected_size =
      static_cast<std::size_t>(n_orbitals) * static_cast<std::size_t>(n_orbitals);
  if (right_orbital_overlap.size() != expected_size) {
    throw std::invalid_argument(
        "right_orbital_overlap size does not match n_orbitals");
  }
  if (right_right_one_electron.size() != expected_size) {
    throw std::invalid_argument(
        "right_right_one_electron size does not match n_orbitals");
  }

  const Eigen::Map<const Eigen::MatrixXd> right_orbital_overlap_matrix(
      right_orbital_overlap.data(),
      n_orbitals,
      n_orbitals);
  const Eigen::Map<const Eigen::MatrixXd> right_right_one_electron_matrix(
      right_right_one_electron.data(),
      n_orbitals,
      n_orbitals);
  return build_from_overlap_and_one_electron(
      right_orbital_overlap_matrix,
      right_right_one_electron_matrix);
}

BiorthogonalOrbitalIntegrals build_biorthogonal_orbital_integrals(
    const BiorthogonalOrbitalFrame& orbital_frame,
    const ActiveSpaceOneElectronResult& right_right_one_electron_result) {
  return build_biorthogonal_orbital_integrals(
      orbital_frame,
      right_right_one_electron_result.h1e_act);
}

void validate_biorthogonal_orbital_integrals(
    const BiorthogonalOrbitalIntegrals& orbital_integrals) {
  if (orbital_integrals.n_orbitals <= 0) {
    throw std::invalid_argument("n_orbitals must be positive");
  }
  if (orbital_integrals.left_dual_from_right_transform.rows() !=
          orbital_integrals.n_orbitals ||
      orbital_integrals.left_dual_from_right_transform.cols() !=
          orbital_integrals.n_orbitals) {
    throw std::invalid_argument(
        "left_dual_from_right_transform dimensions are inconsistent");
  }
  if (orbital_integrals.left_right_one_electron.rows() !=
          orbital_integrals.n_orbitals ||
      orbital_integrals.left_right_one_electron.cols() !=
          orbital_integrals.n_orbitals) {
    throw std::invalid_argument("left_right_one_electron dimensions are inconsistent");
  }
  throw_if_nonfinite(
      orbital_integrals.left_dual_from_right_transform,
      "left_dual_from_right_transform");
  throw_if_nonfinite(
      orbital_integrals.left_right_one_electron,
      "left_right_one_electron");
}

double evaluate_biorthogonal_two_electron_integral(
    int right_first_orbital,
    int left_first_orbital,
    int right_second_orbital,
    int left_second_orbital,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view) {
  validate_biorthogonal_orbital_integrals(orbital_integrals);
  validate_orbital_index(
      right_first_orbital,
      orbital_integrals.n_orbitals,
      "right_first_orbital");
  validate_orbital_index(
      left_first_orbital,
      orbital_integrals.n_orbitals,
      "left_first_orbital");
  validate_orbital_index(
      right_second_orbital,
      orbital_integrals.n_orbitals,
      "right_second_orbital");
  validate_orbital_index(
      left_second_orbital,
      orbital_integrals.n_orbitals,
      "left_second_orbital");

  // The packed active-space `GGO` kernel groups indices as two mixed
  // `(right, left)` pairs. The biorthogonalization therefore acts only on the
  // left leg of each mixed pair, while the right legs stay in the accepted
  // active-orbital basis.
  double integral_value = 0.0;
  for (int right_basis_left = 0; right_basis_left < orbital_integrals.n_orbitals;
       ++right_basis_left) {
    const double left_first_scale =
        orbital_integrals.left_dual_from_right_transform(
            right_basis_left,
            left_first_orbital);
    if (std::abs(left_first_scale) <= 1.0e-15) {
      continue;
    }
    for (int right_basis_right = 0;
         right_basis_right < orbital_integrals.n_orbitals;
         ++right_basis_right) {
      const double left_second_scale =
          orbital_integrals.left_dual_from_right_transform(
              right_basis_right,
              left_second_orbital);
      if (std::abs(left_second_scale) <= 1.0e-15) {
        continue;
      }
      const int left_pair_index = TwoElectronIndexer::packed_pair_index(
          right_first_orbital,
          right_basis_left);
      const int right_pair_index = TwoElectronIndexer::packed_pair_index(
          right_second_orbital,
          right_basis_right);
      integral_value +=
          left_first_scale * left_second_scale *
          lookup_active_space_two_electron_kernel_value(
              right_right_two_electron_view,
              left_pair_index,
              right_pair_index,
              orbital_integrals.n_orbitals);
    }
  }

  return integral_value;
}

double evaluate_biorthogonal_two_electron_integral(
    int right_first_orbital,
    int left_first_orbital,
    int right_second_orbital,
    int left_second_orbital,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronResult& right_right_two_electron_result) {
  return evaluate_biorthogonal_two_electron_integral(
      right_first_orbital,
      left_first_orbital,
      right_second_orbital,
      left_second_orbital,
      orbital_integrals,
      make_active_space_two_electron_view(right_right_two_electron_result));
}

}  // namespace xmvb::vb::biorthogonal_vbscf
