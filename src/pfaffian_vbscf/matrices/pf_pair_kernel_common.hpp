#pragma once

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/QR>

#include "pfaffian_vbscf/types/eigen_types.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace xmvb::pfaffian_vbscf::detail {

template <typename Scalar>
using GenericMatrix =
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

template <typename Scalar>
struct PfPairKernelResultGeneric {
  Scalar overlap = Scalar(0.0);
  Scalar one_electron_hamiltonian = Scalar(0.0);
  Scalar total_hamiltonian = Scalar(0.0);
};

inline double projected_trace_weight(int power) {
  const double alternating_sign = ((power % 2) == 1) ? 1.0 : -1.0;
  return 0.5 * alternating_sign;
}

inline Eigen::MatrixXd build_vandermonde_inverse(int max_degree) {
  Eigen::MatrixXd vandermonde =
      Eigen::MatrixXd::Zero(max_degree + 1, max_degree + 1);
  for (int node = 0; node <= max_degree; ++node) {
    double power_value = 1.0;
    for (int degree = 0; degree <= max_degree; ++degree) {
      vandermonde(node, degree) = power_value;
      power_value *= static_cast<double>(node);
    }
  }
  return Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>(vandermonde)
      .solve(Eigen::MatrixXd::Identity(max_degree + 1, max_degree + 1));
}

inline const Eigen::MatrixXd& get_vandermonde_inverse(int max_degree) {
  thread_local std::vector<std::unique_ptr<Eigen::MatrixXd>> cache;
  if (max_degree < 0) {
    throw std::invalid_argument("max_degree must be non-negative");
  }

  // Closed-shell gradient evaluation queries this helper inside OpenMP loops.
  // Each worker therefore keeps its own cache to avoid lock contention in the
  // hot path while still using heap-stable matrix storage across vector growth.
  if (static_cast<int>(cache.size()) <= max_degree) {
    cache.resize(max_degree + 1);
  }
  std::unique_ptr<Eigen::MatrixXd>& cached_inverse =
      cache[max_degree];
  if (!cached_inverse) {
    cached_inverse =
        std::make_unique<Eigen::MatrixXd>(build_vandermonde_inverse(max_degree));
  }
  return *cached_inverse;
}

template <typename Scalar>
inline Scalar zero_value() {
  return Scalar(0.0);
}

template <typename Scalar>
inline Scalar scalar_from_double(double value) {
  return Scalar(value);
}

template <typename Scalar>
inline GenericMatrix<Scalar> build_kernel_generic(
    const GenericMatrix<Scalar>& left,
    const GenericMatrix<Scalar>& sigma,
    const GenericMatrix<Scalar>& right) {
  if (left.rows() != left.cols() ||
      sigma.rows() != sigma.cols() ||
      right.rows() != right.cols()) {
    throw std::invalid_argument("left/sigma/right must be square");
  }
  if (left.rows() != sigma.rows() || sigma.rows() != right.rows()) {
    throw std::invalid_argument("left/sigma/right dimensions must match");
  }
  return left * sigma * right * sigma.transpose();
}

template <typename Scalar>
inline std::vector<Scalar> build_projected_overlap_coefficients_from_traces_generic(
    const std::vector<Scalar>& traces,
    int coefficient_order) {
  if (coefficient_order < 0) {
    throw std::invalid_argument("coefficient_order must be non-negative");
  }
  if (static_cast<int>(traces.size()) < coefficient_order) {
    throw std::invalid_argument("traces size is smaller than coefficient_order");
  }

  std::vector<Scalar> coefficients(
      coefficient_order + 1,
      zero_value<Scalar>());
  coefficients[0] = scalar_from_double<Scalar>(1.0);
  for (int order = 1; order <= coefficient_order; ++order) {
    Scalar scaled_sum = zero_value<Scalar>();
    for (int power = 1; power <= order; ++power) {
      scaled_sum +=
          scalar_from_double<Scalar>(projected_trace_weight(power)) *
          traces[power - 1] *
          coefficients[order - power];
    }
    coefficients[order] =
        scaled_sum / scalar_from_double<Scalar>(static_cast<double>(order));
  }
  return coefficients;
}

template <typename Scalar>
inline Scalar projected_overlap_first_derivative_coefficient_from_traces_generic(
    const std::vector<Scalar>& traces,
    const std::vector<Scalar>& first_derivative_traces,
    int coefficient_order) {
  if (static_cast<int>(first_derivative_traces.size()) < coefficient_order) {
    throw std::invalid_argument(
        "first_derivative_traces size is smaller than coefficient_order");
  }

  const std::vector<Scalar> coefficients =
      build_projected_overlap_coefficients_from_traces_generic(
          traces,
          coefficient_order);
  std::vector<Scalar> derivative_coefficients(
      coefficient_order + 1,
      zero_value<Scalar>());
  for (int order = 1; order <= coefficient_order; ++order) {
    Scalar scaled_sum = zero_value<Scalar>();
    for (int power = 1; power <= order; ++power) {
      const Scalar weighted_trace =
          scalar_from_double<Scalar>(projected_trace_weight(power));
      scaled_sum +=
          weighted_trace *
          first_derivative_traces[power - 1] *
          coefficients[order - power];
      scaled_sum +=
          weighted_trace *
          traces[power - 1] *
          derivative_coefficients[order - power];
    }
    derivative_coefficients[order] =
        scaled_sum / scalar_from_double<Scalar>(static_cast<double>(order));
  }
  return derivative_coefficients[coefficient_order];
}

template <typename Scalar>
inline Scalar projected_overlap_second_derivative_coefficient_from_traces_generic(
    const std::vector<Scalar>& traces,
    const std::vector<Scalar>& first_derivative_traces_left,
    const std::vector<Scalar>& first_derivative_traces_right,
    const std::vector<Scalar>& second_derivative_traces,
    int coefficient_order) {
  if (static_cast<int>(first_derivative_traces_left.size()) < coefficient_order ||
      static_cast<int>(first_derivative_traces_right.size()) < coefficient_order ||
      static_cast<int>(second_derivative_traces.size()) < coefficient_order) {
    throw std::invalid_argument(
        "second_derivative trace arrays are smaller than coefficient_order");
  }

  const std::vector<Scalar> coefficients =
      build_projected_overlap_coefficients_from_traces_generic(
          traces,
          coefficient_order);
  std::vector<Scalar> left_derivative_coefficients(
      coefficient_order + 1,
      zero_value<Scalar>());
  std::vector<Scalar> right_derivative_coefficients(
      coefficient_order + 1,
      zero_value<Scalar>());
  std::vector<Scalar> second_derivative_coefficients(
      coefficient_order + 1,
      zero_value<Scalar>());

  for (int order = 1; order <= coefficient_order; ++order) {
    Scalar left_scaled_sum = zero_value<Scalar>();
    Scalar right_scaled_sum = zero_value<Scalar>();
    Scalar second_scaled_sum = zero_value<Scalar>();
    for (int power = 1; power <= order; ++power) {
      const Scalar weighted_trace =
          scalar_from_double<Scalar>(projected_trace_weight(power));
      left_scaled_sum +=
          weighted_trace *
          first_derivative_traces_left[power - 1] *
          coefficients[order - power];
      left_scaled_sum +=
          weighted_trace *
          traces[power - 1] *
          left_derivative_coefficients[order - power];
      right_scaled_sum +=
          weighted_trace *
          first_derivative_traces_right[power - 1] *
          coefficients[order - power];
      right_scaled_sum +=
          weighted_trace *
          traces[power - 1] *
          right_derivative_coefficients[order - power];
      second_scaled_sum +=
          weighted_trace *
          second_derivative_traces[power - 1] *
          coefficients[order - power];
      second_scaled_sum +=
          weighted_trace *
          first_derivative_traces_left[power - 1] *
          right_derivative_coefficients[order - power];
      second_scaled_sum +=
          weighted_trace *
          first_derivative_traces_right[power - 1] *
          left_derivative_coefficients[order - power];
      second_scaled_sum +=
          weighted_trace *
          traces[power - 1] *
          second_derivative_coefficients[order - power];
    }
    left_derivative_coefficients[order] =
        left_scaled_sum / scalar_from_double<Scalar>(static_cast<double>(order));
    right_derivative_coefficients[order] =
        right_scaled_sum / scalar_from_double<Scalar>(static_cast<double>(order));
    second_derivative_coefficients[order] =
        second_scaled_sum / scalar_from_double<Scalar>(static_cast<double>(order));
  }
  return second_derivative_coefficients[coefficient_order];
}

inline int spatial_source_entry_index(
    int right_row,
    int left_column,
    int n_active_orbitals) {
  return left_column * n_active_orbitals + right_row;
}

template <typename Scalar>
inline GenericMatrix<Scalar> build_spin_source_direction_matrix_generic(
    int n_active_orbitals,
    bool alpha_channel,
    int right_row,
    int left_column) {
  GenericMatrix<Scalar> direction_matrix =
      GenericMatrix<Scalar>::Zero(2 * n_active_orbitals, 2 * n_active_orbitals);
  const int block_offset = alpha_channel ? 0 : n_active_orbitals;
  direction_matrix(block_offset + right_row, block_offset + left_column) =
      scalar_from_double<Scalar>(1.0);
  return direction_matrix;
}

template <typename Scalar>
inline GenericMatrix<Scalar> build_spin_sector_scaling_matrix_generic(
    int n_active_orbitals,
    double alpha_scale,
    double beta_scale) {
  GenericMatrix<Scalar> scaling_matrix =
      GenericMatrix<Scalar>::Zero(2 * n_active_orbitals, 2 * n_active_orbitals);
  for (int orbital = 0; orbital < n_active_orbitals; ++orbital) {
    scaling_matrix(orbital, orbital) =
        scalar_from_double<Scalar>(alpha_scale);
    scaling_matrix(n_active_orbitals + orbital, n_active_orbitals + orbital) =
        scalar_from_double<Scalar>(beta_scale);
  }
  return scaling_matrix;
}

template <typename Scalar>
inline std::vector<GenericMatrix<Scalar>> build_matrix_powers_including_identity_generic(
    const GenericMatrix<Scalar>& matrix,
    int max_power) {
  if (max_power < 0) {
    throw std::invalid_argument("max_power must be non-negative");
  }

  std::vector<GenericMatrix<Scalar>> powers(
      max_power + 1);
  powers[0] = GenericMatrix<Scalar>::Identity(matrix.rows(), matrix.cols());
  for (int power = 1; power <= max_power; ++power) {
    powers[power] =
        powers[power - 1] * matrix;
  }
  return powers;
}

template <typename Scalar>
inline std::vector<Scalar> build_power_traces_generic(
    const GenericMatrix<Scalar>& matrix,
    int max_power) {
  if (max_power < 0) {
    throw std::invalid_argument("max_power must be non-negative");
  }

  std::vector<Scalar> traces(max_power, zero_value<Scalar>());
  if (max_power == 0) {
    return traces;
  }

  GenericMatrix<Scalar> power_matrix = matrix;
  for (int power = 1; power <= max_power; ++power) {
    traces[power - 1] = power_matrix.trace();
    if (power < max_power) {
      power_matrix *= matrix;
    }
  }
  return traces;
}

template <typename Scalar>
inline std::vector<Scalar> build_first_derivative_power_traces_generic(
    const std::vector<GenericMatrix<Scalar>>& matrix_powers,
    const GenericMatrix<Scalar>& first_direction_matrix,
    int max_power) {
  std::vector<Scalar> traces(max_power, zero_value<Scalar>());
  for (int power = 1; power <= max_power; ++power) {
    traces[power - 1] =
        scalar_from_double<Scalar>(static_cast<double>(power)) *
        (matrix_powers[power - 1] *
         first_direction_matrix)
            .trace();
  }
  return traces;
}

template <typename Scalar>
inline std::vector<Scalar> build_second_derivative_power_traces_generic(
    const std::vector<GenericMatrix<Scalar>>& matrix_powers,
    const GenericMatrix<Scalar>& first_direction_matrix_left,
    const GenericMatrix<Scalar>& first_direction_matrix_right,
    const GenericMatrix<Scalar>& second_direction_matrix,
    int max_power) {
  std::vector<Scalar> traces(max_power, zero_value<Scalar>());
  for (int power = 1; power <= max_power; ++power) {
    Scalar trace_value =
        scalar_from_double<Scalar>(static_cast<double>(power)) *
        (matrix_powers[power - 1] *
         second_direction_matrix)
            .trace();
    for (int split_power = 0; split_power <= power - 2; ++split_power) {
      trace_value +=
          scalar_from_double<Scalar>(static_cast<double>(power)) *
          (matrix_powers[split_power] *
           first_direction_matrix_right *
           matrix_powers[power - 2 - split_power] *
           first_direction_matrix_left)
              .trace();
    }
    traces[power - 1] = trace_value;
  }
  return traces;
}

template <typename Scalar>
inline Scalar spin_resolved_projected_overlap_generic(
    const GenericMatrix<Scalar>& left_pairing_matrix,
    const GenericMatrix<Scalar>& right_pairing_matrix,
    const GenericMatrix<Scalar>& spin_orbital_overlap_matrix,
    int n_pairs,
    int n_active_orbitals,
    int n_alpha_electrons,
    int n_beta_electrons) {
  if (n_alpha_electrons < 0 || n_beta_electrons < 0) {
    throw std::invalid_argument("spin electron counts must be non-negative");
  }
  const int max_degree = n_active_orbitals;
  if (n_alpha_electrons > max_degree || n_beta_electrons > max_degree) {
    throw std::invalid_argument("spin-sector degree exceeds interpolation range");
  }

  const Eigen::MatrixXd& inverse_vandermonde_double =
      get_vandermonde_inverse(max_degree);
  const GenericMatrix<Scalar> inverse_vandermonde =
      inverse_vandermonde_double.template cast<Scalar>();
  GenericMatrix<Scalar> sampled_values =
      GenericMatrix<Scalar>::Zero(max_degree + 1, max_degree + 1);
  for (int alpha_degree = 0; alpha_degree <= max_degree; ++alpha_degree) {
    for (int beta_degree = 0; beta_degree <= max_degree; ++beta_degree) {
      const GenericMatrix<Scalar> scaling_matrix =
          build_spin_sector_scaling_matrix_generic<Scalar>(
              n_active_orbitals,
              static_cast<double>(alpha_degree),
              static_cast<double>(beta_degree));
      const GenericMatrix<Scalar> scaled_left_pairing_matrix =
          scaling_matrix *
          left_pairing_matrix *
          scaling_matrix;
      const GenericMatrix<Scalar> scaled_left_pairing_matrix_transpose =
          scaled_left_pairing_matrix.transpose();
      const GenericMatrix<Scalar> kernel_matrix =
          build_kernel_generic(
              scaled_left_pairing_matrix_transpose,
              spin_orbital_overlap_matrix,
              right_pairing_matrix);
      const std::vector<Scalar> traces =
          build_power_traces_generic(kernel_matrix, n_pairs);
      sampled_values(alpha_degree, beta_degree) =
          build_projected_overlap_coefficients_from_traces_generic(
              traces,
              n_pairs)[n_pairs];
    }
  }

  const GenericMatrix<Scalar> coefficient_matrix =
      inverse_vandermonde *
      sampled_values *
      inverse_vandermonde.transpose();
  return coefficient_matrix(n_alpha_electrons, n_beta_electrons);
}

template <typename Scalar>
inline PfPairKernelResultGeneric<Scalar> evaluate_pf_pair_kernel_generic(
    const GenericMatrix<Scalar>& left_pairing_matrix,
    const GenericMatrix<Scalar>& right_pairing_matrix,
    const GenericMatrix<Scalar>& spatial_overlap_matrix,
    const GenericMatrix<Scalar>& one_electron_matrix,
    const std::vector<Scalar>& packed_active_two_electron_integrals,
    int n_alpha_electrons,
    int n_beta_electrons,
    int n_pairs) {
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
  if (spatial_overlap_matrix.rows() != spatial_overlap_matrix.cols() ||
      one_electron_matrix.rows() != one_electron_matrix.cols()) {
    throw std::invalid_argument("spatial matrices must be square");
  }
  if (spatial_overlap_matrix.rows() != one_electron_matrix.rows()) {
    throw std::invalid_argument("spatial overlap and one-electron dimensions must match");
  }
  if (left_pairing_matrix.rows() != 2 * spatial_overlap_matrix.rows()) {
    throw std::invalid_argument(
        "pairing matrix dimension does not match the active spin space");
  }

  const int n_active_orbitals = static_cast<int>(spatial_overlap_matrix.rows());
  const int n_spatial_entries = n_active_orbitals * n_active_orbitals;

  GenericMatrix<Scalar> spin_orbital_overlap_matrix =
      GenericMatrix<Scalar>::Zero(2 * n_active_orbitals, 2 * n_active_orbitals);
  spin_orbital_overlap_matrix.topLeftCorner(
      n_active_orbitals,
      n_active_orbitals) = spatial_overlap_matrix;
  spin_orbital_overlap_matrix.bottomRightCorner(
      n_active_orbitals,
      n_active_orbitals) = spatial_overlap_matrix;
  const GenericMatrix<Scalar> left_pairing_transpose =
      left_pairing_matrix.transpose();
  const GenericMatrix<Scalar> kernel_matrix =
      build_kernel_generic(
          left_pairing_transpose,
          spin_orbital_overlap_matrix,
          right_pairing_matrix);
  const std::vector<GenericMatrix<Scalar>> matrix_powers =
      build_matrix_powers_including_identity_generic(kernel_matrix, n_pairs);
  const std::vector<Scalar> traces =
      build_power_traces_generic(kernel_matrix, n_pairs);

  PfPairKernelResultGeneric<Scalar> result;
  result.overlap =
      spin_resolved_projected_overlap_generic(
          left_pairing_matrix,
          right_pairing_matrix,
          spin_orbital_overlap_matrix,
          n_pairs,
          n_active_orbitals,
          n_alpha_electrons,
          n_beta_electrons);

  std::vector<GenericMatrix<Scalar>> alpha_source_directions(
      n_spatial_entries);
  std::vector<GenericMatrix<Scalar>> beta_source_directions(
      n_spatial_entries);
  std::vector<GenericMatrix<Scalar>> alpha_first_direction_matrices(
      n_spatial_entries);
  std::vector<GenericMatrix<Scalar>> beta_first_direction_matrices(
      n_spatial_entries);
  std::vector<std::vector<Scalar>> alpha_first_derivative_traces(
      n_spatial_entries);
  std::vector<std::vector<Scalar>> beta_first_derivative_traces(
      n_spatial_entries);
  std::vector<Scalar> alpha_first_derivative_coefficients(
      n_spatial_entries,
      zero_value<Scalar>());
  std::vector<Scalar> beta_first_derivative_coefficients(
      n_spatial_entries,
      zero_value<Scalar>());

  const GenericMatrix<Scalar> left_transpose = left_pairing_matrix.transpose();
  const GenericMatrix<Scalar> right_times_source_transpose =
      right_pairing_matrix * spin_orbital_overlap_matrix.transpose();
  const GenericMatrix<Scalar> source_times_right =
      spin_orbital_overlap_matrix * right_pairing_matrix;

  for (int left_column = 0; left_column < n_active_orbitals; ++left_column) {
    for (int right_row = 0; right_row < n_active_orbitals; ++right_row) {
      const int entry_index =
          spatial_source_entry_index(
              right_row,
              left_column,
              n_active_orbitals);
      alpha_source_directions[entry_index] =
          build_spin_source_direction_matrix_generic<Scalar>(
              n_active_orbitals,
              true,
              right_row,
              left_column);
      beta_source_directions[entry_index] =
          build_spin_source_direction_matrix_generic<Scalar>(
              n_active_orbitals,
              false,
              right_row,
              left_column);
      alpha_first_direction_matrices[entry_index] =
          left_transpose *
              alpha_source_directions[entry_index] *
              right_times_source_transpose +
          left_transpose *
              source_times_right *
              alpha_source_directions[entry_index].transpose();
      beta_first_direction_matrices[entry_index] =
          left_transpose *
              beta_source_directions[entry_index] *
              right_times_source_transpose +
          left_transpose *
              source_times_right *
              beta_source_directions[entry_index].transpose();
      alpha_first_derivative_traces[entry_index] =
          build_first_derivative_power_traces_generic(
              matrix_powers,
              alpha_first_direction_matrices[entry_index],
              n_pairs);
      beta_first_derivative_traces[entry_index] =
          build_first_derivative_power_traces_generic(
              matrix_powers,
              beta_first_direction_matrices[entry_index],
              n_pairs);
      alpha_first_derivative_coefficients[entry_index] =
          projected_overlap_first_derivative_coefficient_from_traces_generic(
              traces,
              alpha_first_derivative_traces[entry_index],
              n_pairs);
      beta_first_derivative_coefficients[entry_index] =
          projected_overlap_first_derivative_coefficient_from_traces_generic(
              traces,
              beta_first_derivative_traces[entry_index],
              n_pairs);
    }
  }

  for (int left_column = 0; left_column < n_active_orbitals; ++left_column) {
    for (int right_row = 0; right_row < n_active_orbitals; ++right_row) {
      const int entry_index =
          spatial_source_entry_index(
              right_row,
              left_column,
              n_active_orbitals);
      const Scalar h_qp = one_electron_matrix(right_row, left_column);
      result.one_electron_hamiltonian +=
          h_qp * alpha_first_derivative_coefficients[entry_index];
      result.one_electron_hamiltonian +=
          h_qp * beta_first_derivative_coefficients[entry_index];
    }
  }
  result.total_hamiltonian = result.one_electron_hamiltonian;

  for (int left_first = 0; left_first < n_active_orbitals - 1; ++left_first) {
    for (int right_first = 0; right_first < n_active_orbitals - 1; ++right_first) {
      const int first_entry_index =
          spatial_source_entry_index(
              right_first,
              left_first,
              n_active_orbitals);
      for (int left_second = left_first + 1;
           left_second < n_active_orbitals;
           ++left_second) {
        for (int right_second = right_first + 1;
             right_second < n_active_orbitals;
             ++right_second) {
          const int second_entry_index =
              spatial_source_entry_index(
                  right_second,
                  left_second,
                  n_active_orbitals);
          const int direct_index =
              xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
                  right_first,
                  left_first,
                  right_second,
                  left_second);
          const int exchange_index =
              xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
                  right_first,
                  left_second,
                  right_second,
                  left_first);
          const Scalar same_spin_interaction =
              packed_active_two_electron_integrals[direct_index] -
              packed_active_two_electron_integrals[exchange_index];

          const GenericMatrix<Scalar> alpha_second_direction_matrix =
              left_transpose *
                  alpha_source_directions[first_entry_index] *
                  right_pairing_matrix *
                  alpha_source_directions[second_entry_index].transpose() +
              left_transpose *
                  alpha_source_directions[second_entry_index] *
                  right_pairing_matrix *
                  alpha_source_directions[first_entry_index].transpose();
          const std::vector<Scalar> alpha_second_derivative_traces =
              build_second_derivative_power_traces_generic(
                  matrix_powers,
                  alpha_first_direction_matrices[first_entry_index],
                  alpha_first_direction_matrices[second_entry_index],
                  alpha_second_direction_matrix,
                  n_pairs);
          result.total_hamiltonian +=
              same_spin_interaction *
              projected_overlap_second_derivative_coefficient_from_traces_generic(
                  traces,
                  alpha_first_derivative_traces[first_entry_index],
                  alpha_first_derivative_traces[second_entry_index],
                  alpha_second_derivative_traces,
                  n_pairs);

          const GenericMatrix<Scalar> beta_second_direction_matrix =
              left_transpose *
                  beta_source_directions[first_entry_index] *
                  right_pairing_matrix *
                  beta_source_directions[second_entry_index].transpose() +
              left_transpose *
                  beta_source_directions[second_entry_index] *
                  right_pairing_matrix *
                  beta_source_directions[first_entry_index].transpose();
          const std::vector<Scalar> beta_second_derivative_traces =
              build_second_derivative_power_traces_generic(
                  matrix_powers,
                  beta_first_direction_matrices[first_entry_index],
                  beta_first_direction_matrices[second_entry_index],
                  beta_second_direction_matrix,
                  n_pairs);
          result.total_hamiltonian +=
              same_spin_interaction *
              projected_overlap_second_derivative_coefficient_from_traces_generic(
                  traces,
                  beta_first_derivative_traces[first_entry_index],
                  beta_first_derivative_traces[second_entry_index],
                  beta_second_derivative_traces,
                  n_pairs);
        }
      }
    }
  }

  for (int alpha_left_column = 0;
       alpha_left_column < n_active_orbitals;
       ++alpha_left_column) {
    for (int alpha_right_row = 0;
         alpha_right_row < n_active_orbitals;
         ++alpha_right_row) {
      const int alpha_entry_index =
          spatial_source_entry_index(
              alpha_right_row,
              alpha_left_column,
              n_active_orbitals);
      for (int beta_left_column = 0;
           beta_left_column < n_active_orbitals;
           ++beta_left_column) {
        for (int beta_right_row = 0;
             beta_right_row < n_active_orbitals;
             ++beta_right_row) {
          const int beta_entry_index =
              spatial_source_entry_index(
                  beta_right_row,
                  beta_left_column,
                  n_active_orbitals);
          const int opposite_spin_index =
              xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
                  beta_right_row,
                  beta_left_column,
                  alpha_right_row,
                  alpha_left_column);
          const GenericMatrix<Scalar> mixed_second_direction_matrix =
              left_transpose *
                  alpha_source_directions[alpha_entry_index] *
                  right_pairing_matrix *
                  beta_source_directions[beta_entry_index].transpose() +
              left_transpose *
                  beta_source_directions[beta_entry_index] *
                  right_pairing_matrix *
                  alpha_source_directions[alpha_entry_index].transpose();
          const std::vector<Scalar> mixed_second_derivative_traces =
              build_second_derivative_power_traces_generic(
                  matrix_powers,
                  alpha_first_direction_matrices[alpha_entry_index],
                  beta_first_direction_matrices[beta_entry_index],
                  mixed_second_direction_matrix,
                  n_pairs);
          result.total_hamiltonian +=
              packed_active_two_electron_integrals[opposite_spin_index] *
              projected_overlap_second_derivative_coefficient_from_traces_generic(
                  traces,
                  alpha_first_derivative_traces[alpha_entry_index],
                  beta_first_derivative_traces[beta_entry_index],
                  mixed_second_derivative_traces,
                  n_pairs);
        }
      }
    }
  }

  return result;
}

template <typename Scalar>
inline GenericMatrix<Scalar> cast_matrix_generic(const Matrix& matrix) {
  return matrix.template cast<Scalar>();
}

template <typename Scalar>
inline std::vector<Scalar> cast_buffer_generic(const ScalarBuffer& buffer) {
  std::vector<Scalar> out(buffer.size(), zero_value<Scalar>());
  for (std::size_t index = 0; index < buffer.size(); ++index) {
    out[index] = scalar_from_double<Scalar>(buffer[index]);
  }
  return out;
}

}  // namespace xmvb::pfaffian_vbscf::detail
