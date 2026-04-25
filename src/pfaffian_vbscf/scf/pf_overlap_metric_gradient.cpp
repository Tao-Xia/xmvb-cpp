#include "pfaffian_vbscf/scf/pf_overlap_metric_gradient.hpp"

#include <stdexcept>

#include "pfaffian_vbscf/matrices/pf_pair_kernel_common.hpp"

namespace xmvb::pfaffian_vbscf {

namespace {

Matrix build_spin_overlap_matrix(const ConstMatrixRef& spatial_overlap_matrix) {
  Matrix spin_orbital_overlap_matrix = Matrix::Zero(
      2 * spatial_overlap_matrix.rows(),
      2 * spatial_overlap_matrix.cols());
  spin_orbital_overlap_matrix.topLeftCorner(
      spatial_overlap_matrix.rows(),
      spatial_overlap_matrix.cols()) = spatial_overlap_matrix;
  spin_orbital_overlap_matrix.bottomRightCorner(
      spatial_overlap_matrix.rows(),
      spatial_overlap_matrix.cols()) = spatial_overlap_matrix;
  return spin_orbital_overlap_matrix;
}

}  // namespace

ScalarBuffer evaluate_spin_resolved_overlap_metric_gradient(
    const ConstMatrixRef& left_pairing_matrix,
    const ConstMatrixRef& right_pairing_matrix,
    const ConstMatrixRef& spatial_overlap_matrix,
    int n_alpha_electrons,
    int n_beta_electrons,
    int n_pairs) {
  using Scalar = double;
  using GenericMatrix = detail::GenericMatrix<Scalar>;

  if (n_alpha_electrons < 0 || n_beta_electrons < 0) {
    throw std::invalid_argument("spin electron counts must be non-negative");
  }
  if (n_pairs < 0) {
    throw std::invalid_argument("n_pairs must be non-negative");
  }
  if (left_pairing_matrix.rows() != left_pairing_matrix.cols() ||
      right_pairing_matrix.rows() != right_pairing_matrix.cols()) {
    throw std::invalid_argument("pairing matrices must be square");
  }
  if (left_pairing_matrix.rows() != right_pairing_matrix.rows()) {
    throw std::invalid_argument("left/right pairing matrix dimensions must match");
  }
  if (spatial_overlap_matrix.rows() != spatial_overlap_matrix.cols()) {
    throw std::invalid_argument("spatial_overlap_matrix must be square");
  }
  if (left_pairing_matrix.rows() != 2 * spatial_overlap_matrix.rows()) {
    throw std::invalid_argument(
        "pairing matrix dimension does not match the active spin space");
  }

  const int n_active_orbitals = static_cast<int>(spatial_overlap_matrix.rows());
  const int n_spatial_entries = n_active_orbitals * n_active_orbitals;
  const int max_degree = n_active_orbitals;
  const Eigen::MatrixXd& inverse_vandermonde =
      detail::get_vandermonde_inverse(max_degree);

  ScalarBuffer gradient(n_spatial_entries, 0.0);
  const GenericMatrix spin_orbital_overlap_matrix =
      build_spin_overlap_matrix(spatial_overlap_matrix);

  for (int alpha_degree = 0; alpha_degree <= max_degree; ++alpha_degree) {
    for (int beta_degree = 0; beta_degree <= max_degree; ++beta_degree) {
      const double sample_weight =
          inverse_vandermonde(n_alpha_electrons, alpha_degree) *
          inverse_vandermonde(n_beta_electrons, beta_degree);
      if (sample_weight == 0.0) {
        continue;
      }

      const GenericMatrix scaling_matrix =
          detail::build_spin_sector_scaling_matrix_generic<Scalar>(
              n_active_orbitals,
              static_cast<double>(alpha_degree),
              static_cast<double>(beta_degree));
      const GenericMatrix scaled_left_pairing_matrix =
          scaling_matrix * left_pairing_matrix * scaling_matrix;
      const GenericMatrix left_transpose = scaled_left_pairing_matrix.transpose();
      const GenericMatrix kernel_matrix =
          detail::build_kernel_generic<Scalar>(
              left_transpose,
              spin_orbital_overlap_matrix,
              right_pairing_matrix);
      const std::vector<GenericMatrix> matrix_powers =
          detail::build_matrix_powers_including_identity_generic(
              kernel_matrix,
              n_pairs);
      const std::vector<Scalar> traces =
          detail::build_power_traces_generic(kernel_matrix, n_pairs);
      const GenericMatrix right_times_source_transpose =
          right_pairing_matrix * spin_orbital_overlap_matrix.transpose();
      const GenericMatrix source_times_right =
          spin_orbital_overlap_matrix * right_pairing_matrix;

      for (int left_column = 0; left_column < n_active_orbitals; ++left_column) {
        for (int right_row = 0; right_row < n_active_orbitals; ++right_row) {
          const int entry_index =
              detail::spatial_source_entry_index(
                  right_row,
                  left_column,
                  n_active_orbitals);
          const GenericMatrix alpha_source_direction =
              detail::build_spin_source_direction_matrix_generic<Scalar>(
                  n_active_orbitals,
                  true,
                  right_row,
                  left_column);
          const GenericMatrix beta_source_direction =
              detail::build_spin_source_direction_matrix_generic<Scalar>(
                  n_active_orbitals,
                  false,
                  right_row,
                  left_column);
          const GenericMatrix alpha_first_direction_matrix =
              left_transpose *
                  alpha_source_direction *
                  right_times_source_transpose +
              left_transpose *
                  source_times_right *
                  alpha_source_direction.transpose();
          const GenericMatrix beta_first_direction_matrix =
              left_transpose *
                  beta_source_direction *
                  right_times_source_transpose +
              left_transpose *
                  source_times_right *
                  beta_source_direction.transpose();
          const std::vector<Scalar> alpha_first_derivative_traces =
              detail::build_first_derivative_power_traces_generic(
                  matrix_powers,
                  alpha_first_direction_matrix,
                  n_pairs);
          const std::vector<Scalar> beta_first_derivative_traces =
              detail::build_first_derivative_power_traces_generic(
                  matrix_powers,
                  beta_first_direction_matrix,
                  n_pairs);
          const double sample_gradient =
              detail::projected_overlap_first_derivative_coefficient_from_traces_generic(
                  traces,
                  alpha_first_derivative_traces,
                  n_pairs) +
              detail::projected_overlap_first_derivative_coefficient_from_traces_generic(
                  traces,
                  beta_first_derivative_traces,
                  n_pairs);
          gradient[entry_index] +=
              sample_weight * sample_gradient;
        }
      }
    }
  }

  return gradient;
}

}  // namespace xmvb::pfaffian_vbscf
