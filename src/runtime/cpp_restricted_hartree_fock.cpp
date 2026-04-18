#include "runtime/cpp_restricted_hartree_fock.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "runtime/cpp_closed_shell_fock_builder.hpp"

namespace xmvb::vb {

namespace {

std::vector<double> symmetrize_matrix(
    const std::vector<double>& matrix,
    int dimension) {
  const std::size_t expected_size =
      xmvb::to_size(dimension) * xmvb::to_size(dimension);
  if (matrix.size() != expected_size) {
    throw std::invalid_argument("matrix size does not match dimension");
  }

  std::vector<double> symmetric_matrix = matrix;
  for (int column = 0; column < dimension; ++column) {
    for (int row = 0; row < column; ++row) {
      const std::size_t upper_index =
          xmvb::to_size(column) * dimension + row;
      const std::size_t lower_index =
          xmvb::to_size(row) * dimension + column;
      const double average =
          0.5 * (symmetric_matrix[upper_index] + symmetric_matrix[lower_index]);
      symmetric_matrix[upper_index] = average;
      symmetric_matrix[lower_index] = average;
    }
  }
  return symmetric_matrix;
}

std::vector<double> build_density_projector(
    const std::vector<double>& molecular_orbital_matrix,
    int n_basis_functions,
    int n_occupied_orbitals) {
  std::vector<double> density_projector(
      xmvb::to_size(n_basis_functions) * n_basis_functions,
      0.0);
  for (int occupied_index = 0; occupied_index < n_occupied_orbitals; ++occupied_index) {
    const double* orbital_column =
        molecular_orbital_matrix.data() +
        xmvb::to_size(occupied_index) * n_basis_functions;
    for (int column = 0; column < n_basis_functions; ++column) {
      const double column_value = orbital_column[column];
      for (int row = 0; row < n_basis_functions; ++row) {
        density_projector[xmvb::to_size(column) * n_basis_functions + row] +=
            orbital_column[row] * column_value;
      }
    }
  }
  return density_projector;
}

double max_abs_difference(
    const std::vector<double>& left,
    const std::vector<double>& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("density matrices must have the same size");
  }

  double max_difference = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    max_difference =
        std::max(max_difference, std::abs(left[index] - right[index]));
  }
  return max_difference;
}

double max_abs_value(const std::vector<double>& values) {
  double max_value = 0.0;
  for (double value : values) {
    max_value = std::max(max_value, std::abs(value));
  }
  return max_value;
}

std::vector<double> mix_density_projector(
    const std::vector<double>& old_density_projector,
    const std::vector<double>& new_density_projector,
    double old_density_weight) {
  if (old_density_projector.size() != new_density_projector.size()) {
    throw std::invalid_argument("density matrices must have the same size");
  }

  std::vector<double> mixed_density_projector(old_density_projector.size(), 0.0);
  const double new_density_weight = 1.0 - old_density_weight;
  for (std::size_t index = 0; index < old_density_projector.size(); ++index) {
    mixed_density_projector[index] =
        old_density_weight * old_density_projector[index] +
        new_density_weight * new_density_projector[index];
  }
  return mixed_density_projector;
}

std::vector<double> multiply_square_matrices(
    const std::vector<double>& left,
    const std::vector<double>& right,
    int dimension) {
  const std::size_t expected_size =
      xmvb::to_size(dimension) * xmvb::to_size(dimension);
  if (left.size() != expected_size || right.size() != expected_size) {
    throw std::invalid_argument("matrix size does not match dimension");
  }

  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
      left_matrix(left.data(), dimension, dimension);
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
      right_matrix(right.data(), dimension, dimension);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> product =
      left_matrix * right_matrix;
  return std::vector<double>(product.data(), product.data() + product.size());
}

std::vector<double> build_cdiis_error_matrix(
    const std::vector<double>& fock_matrix,
    const std::vector<double>& density_projector,
    const std::vector<double>& overlap_matrix,
    int dimension) {
  const auto fock_density = multiply_square_matrices(
      fock_matrix,
      density_projector,
      dimension);
  const auto fock_density_overlap = multiply_square_matrices(
      fock_density,
      overlap_matrix,
      dimension);
  std::vector<double> error_matrix = fock_density_overlap;
  for (int column = 0; column < dimension; ++column) {
    for (int row = 0; row < dimension; ++row) {
      const std::size_t index =
          xmvb::to_size(column) * dimension + row;
      const std::size_t transpose_index =
          xmvb::to_size(row) * dimension + column;
      error_matrix[index] -= fock_density_overlap[transpose_index];
    }
  }
  return error_matrix;
}

double inner_product(
    const std::vector<double>& left,
    const std::vector<double>& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("vectors must have the same size");
  }
  double result = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    result += left[index] * right[index];
  }
  return result;
}

struct CdiisEntry {
  std::vector<double> fock_matrix;
  std::vector<double> error_matrix;
};

void append_cdiis_entry(
    const CdiisEntry& entry,
    int max_history,
    std::vector<CdiisEntry>* history) {
  if (history == nullptr) {
    throw std::invalid_argument("history must not be null");
  }
  if (max_history <= 0) {
    return;
  }
  if (static_cast<int>(history->size()) == max_history) {
    history->erase(history->begin());
  }
  history->push_back(entry);
}

bool try_build_cdiis_fock(
    const std::vector<CdiisEntry>& history,
    std::vector<double>* mixed_fock_matrix) {
  if (mixed_fock_matrix == nullptr) {
    throw std::invalid_argument("mixed_fock_matrix must not be null");
  }
  const int n_entries = static_cast<int>(history.size());
  if (n_entries < 2) {
    return false;
  }

  Eigen::MatrixXd augmented_matrix =
      Eigen::MatrixXd::Zero(n_entries + 1, n_entries + 1);
  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n_entries + 1);
  rhs(n_entries) = -1.0;
  for (int row = 0; row < n_entries; ++row) {
    for (int column = 0; column < n_entries; ++column) {
      augmented_matrix(row, column) = inner_product(
          history[xmvb::to_size(row)].error_matrix,
          history[xmvb::to_size(column)].error_matrix);
    }
    augmented_matrix(row, n_entries) = -1.0;
    augmented_matrix(n_entries, row) = -1.0;
  }

  Eigen::FullPivLU<Eigen::MatrixXd> lu(augmented_matrix);
  if (!lu.isInvertible()) {
    return false;
  }

  const Eigen::VectorXd solution = lu.solve(rhs);
  mixed_fock_matrix->assign(
      history.front().fock_matrix.size(),
      0.0);
  for (int entry_index = 0; entry_index < n_entries; ++entry_index) {
    const double coefficient = solution(entry_index);
    const auto& entry_fock_matrix =
        history[xmvb::to_size(entry_index)].fock_matrix;
    for (std::size_t value_index = 0; value_index < entry_fock_matrix.size(); ++value_index) {
      (*mixed_fock_matrix)[value_index] += coefficient * entry_fock_matrix[value_index];
    }
  }
  return true;
}

double compute_electronic_energy(
    const std::vector<double>& density_projector,
    const std::vector<double>& core_hamiltonian_matrix,
    const std::vector<double>& fock_matrix) {
  if (density_projector.size() != core_hamiltonian_matrix.size() ||
      density_projector.size() != fock_matrix.size()) {
    throw std::invalid_argument("RHF energy inputs must have matching sizes");
  }

  double electronic_energy = 0.0;
  for (std::size_t index = 0; index < density_projector.size(); ++index) {
    electronic_energy +=
        density_projector[index] *
        (core_hamiltonian_matrix[index] + fock_matrix[index]);
  }
  return electronic_energy;
}

}  // namespace

CppRestrictedHartreeFockSolver::CppRestrictedHartreeFockSolver(
    CppRestrictedHartreeFockOptions options)
    : options_(options) {}

CppRestrictedHartreeFockResult CppRestrictedHartreeFockSolver::solve(
    int n_total_electrons,
    const std::vector<double>& ao_overlap_matrix,
    const AoIntegralInput& ao_integral_input) const {
  return solve(
      n_total_electrons,
      ao_overlap_matrix,
      ao_integral_input,
      {});
}

CppRestrictedHartreeFockResult CppRestrictedHartreeFockSolver::solve(
    int n_total_electrons,
    const std::vector<double>& ao_overlap_matrix,
    const AoIntegralInput& ao_integral_input,
    const std::vector<double>& initial_density_projector) const {
  const int n_basis_functions = ao_integral_input.n_basis_functions;
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }
  if (n_total_electrons < 0 || n_total_electrons % 2 != 0) {
    throw std::invalid_argument("RHF solver requires an even non-negative electron count");
  }
  if (n_total_electrons > n_basis_functions * 2) {
    throw std::invalid_argument("electron count exceeds RHF basis capacity");
  }
  if (options_.max_iterations <= 0) {
    throw std::invalid_argument("RHF max_iterations must be positive");
  }
  if (!(options_.density_tolerance > 0.0)) {
    throw std::invalid_argument("RHF density_tolerance must be positive");
  }
  if (options_.old_density_weight < 0.0 || options_.old_density_weight >= 1.0) {
    throw std::invalid_argument("RHF old_density_weight must be in [0, 1)");
  }
  if (options_.diis_history < 0) {
    throw std::invalid_argument("RHF diis_history must be non-negative");
  }
  if (options_.diis_start_iteration < 0) {
    throw std::invalid_argument("RHF diis_start_iteration must be non-negative");
  }

  const int n_occupied_orbitals = n_total_electrons / 2;
  const auto symmetric_overlap_matrix =
      symmetrize_matrix(ao_overlap_matrix, n_basis_functions);
  const auto symmetric_core_hamiltonian =
      symmetrize_matrix(
          ao_integral_input.ao_core_hamiltonian_matrix.vector(),
          n_basis_functions);

  core::GeneralizedEigensolver eigensolver;
  CppClosedShellFockBuilder fock_builder;

  auto core_eigen_result = eigensolver.solve(
      symmetric_core_hamiltonian,
      symmetric_overlap_matrix,
      n_basis_functions);
  std::vector<double> density_projector;
  if (!initial_density_projector.empty()) {
    if (initial_density_projector.size() !=
        xmvb::to_size(n_basis_functions) * n_basis_functions) {
      throw std::invalid_argument(
          "initial_density_projector size does not match n_basis_functions");
    }
    density_projector = symmetrize_matrix(initial_density_projector, n_basis_functions);
  } else {
    density_projector = build_density_projector(
        core_eigen_result.eigenvector_matrix,
        n_basis_functions,
        n_occupied_orbitals);
  }

  CppRestrictedHartreeFockResult result;
  result.orbital_energies = std::move(core_eigen_result.eigenvalues);
  result.molecular_orbital_matrix = std::move(core_eigen_result.eigenvector_matrix);
  std::vector<CdiisEntry> diis_history;
  diis_history.reserve(xmvb::to_size(std::max(0, options_.diis_history)));

  for (int iteration = 0; iteration < options_.max_iterations; ++iteration) {
    auto symmetric_fock_matrix =
        symmetrize_matrix(fock_builder.build(density_projector, ao_integral_input), n_basis_functions);
    const auto cdiis_error_matrix = build_cdiis_error_matrix(
        symmetric_fock_matrix,
        density_projector,
        symmetric_overlap_matrix,
        n_basis_functions);
    append_cdiis_entry(
        {symmetric_fock_matrix, cdiis_error_matrix},
        options_.diis_history,
        &diis_history);
    if (iteration + 1 >= options_.diis_start_iteration) {
      std::vector<double> cdiis_fock_matrix;
      if (try_build_cdiis_fock(diis_history, &cdiis_fock_matrix)) {
        symmetric_fock_matrix = std::move(cdiis_fock_matrix);
      }
    }
    auto fock_eigen_result = eigensolver.solve(
        symmetric_fock_matrix,
        symmetric_overlap_matrix,
        n_basis_functions);
    const auto new_density_projector = build_density_projector(
        fock_eigen_result.eigenvector_matrix,
        n_basis_functions,
        n_occupied_orbitals);

    result.iterations = iteration + 1;
    result.fock_matrix = symmetric_fock_matrix;
    result.orbital_energies = std::move(fock_eigen_result.eigenvalues);
    result.molecular_orbital_matrix = std::move(fock_eigen_result.eigenvector_matrix);
    result.electronic_energy = compute_electronic_energy(
        new_density_projector,
        symmetric_core_hamiltonian,
        symmetric_fock_matrix);
    const double cdiis_error_inf_norm = max_abs_value(cdiis_error_matrix);

    if (max_abs_difference(new_density_projector, density_projector) <
            options_.density_tolerance &&
        cdiis_error_inf_norm < options_.density_tolerance) {
      result.converged = true;
      result.density_projector = new_density_projector;
      break;
    }

    density_projector = mix_density_projector(
        density_projector,
        new_density_projector,
        options_.old_density_weight);
    result.density_projector = density_projector;
  }

  result.fock_matrix = symmetrize_matrix(
      fock_builder.build(result.density_projector, ao_integral_input),
      n_basis_functions);
  auto final_eigen_result = eigensolver.solve(
      result.fock_matrix,
      symmetric_overlap_matrix,
      n_basis_functions);
  result.orbital_energies = std::move(final_eigen_result.eigenvalues);
  result.molecular_orbital_matrix = std::move(final_eigen_result.eigenvector_matrix);
  result.density_projector = build_density_projector(
      result.molecular_orbital_matrix,
      n_basis_functions,
      n_occupied_orbitals);
  result.electronic_energy = compute_electronic_energy(
      result.density_projector,
      symmetric_core_hamiltonian,
      result.fock_matrix);
  return result;
}

}  // namespace xmvb::vb
