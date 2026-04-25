#include "runtime/cpp_restricted_hartree_fock.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "runtime/cpp_closed_shell_fock_builder.hpp"
#include "vb/matrices/eigen_matrix_storage_utils.hpp"

namespace xmvb::vb {

namespace {

Eigen::MatrixXd symmetrize_matrix(
    const std::vector<double>& matrix_buffer,
    int dimension) {
  const std::size_t expected_size =
      dimension * dimension;
  if (matrix_buffer.size() != expected_size) {
    throw std::invalid_argument("matrix size does not match dimension");
  }

  Eigen::Map<const Eigen::MatrixXd> matrix(
      matrix_buffer.data(),
      dimension,
      dimension);
  Eigen::MatrixXd symmetric_matrix = matrix;
  for (int column = 0; column < dimension; ++column) {
    for (int row = 0; row < column; ++row) {
      const double average =
          0.5 * (symmetric_matrix(row, column) + symmetric_matrix(column, row));
      symmetric_matrix(row, column) = average;
      symmetric_matrix(column, row) = average;
    }
  }
  return symmetric_matrix;
}

Eigen::MatrixXd symmetrize_matrix(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("matrix must be square");
  }

  Eigen::MatrixXd symmetric_matrix = matrix;
  for (int column = 0; column < symmetric_matrix.cols(); ++column) {
    for (int row = 0; row < column; ++row) {
      const double average =
          0.5 * (symmetric_matrix(row, column) + symmetric_matrix(column, row));
      symmetric_matrix(row, column) = average;
      symmetric_matrix(column, row) = average;
    }
  }
  return symmetric_matrix;
}

Eigen::MatrixXd build_density_projector(
    const Eigen::Ref<const Eigen::MatrixXd>& molecular_orbital_matrix,
    int n_basis_functions,
    int n_occupied_orbitals) {
  return molecular_orbital_matrix.leftCols(n_occupied_orbitals) *
      molecular_orbital_matrix.leftCols(n_occupied_orbitals).transpose();
}

double max_abs_difference(
    const Eigen::Ref<const Eigen::MatrixXd>& left,
    const Eigen::Ref<const Eigen::MatrixXd>& right) {
  if (left.rows() != right.rows() || left.cols() != right.cols()) {
    throw std::invalid_argument("density matrices must have the same shape");
  }
  return (left - right).cwiseAbs().maxCoeff();
}

double max_abs_value(const Eigen::Ref<const Eigen::MatrixXd>& values) {
  return values.cwiseAbs().maxCoeff();
}

Eigen::MatrixXd mix_density_projector(
    const Eigen::Ref<const Eigen::MatrixXd>& old_density_projector,
    const Eigen::Ref<const Eigen::MatrixXd>& new_density_projector,
    double old_density_weight) {
  if (old_density_projector.rows() != new_density_projector.rows() ||
      old_density_projector.cols() != new_density_projector.cols()) {
    throw std::invalid_argument("density matrices must have the same shape");
  }
  const double new_density_weight = 1.0 - old_density_weight;
  return old_density_weight * old_density_projector +
      new_density_weight * new_density_projector;
}

Eigen::MatrixXd build_cdiis_error_matrix(
    const Eigen::Ref<const Eigen::MatrixXd>& fock_matrix,
    const Eigen::Ref<const Eigen::MatrixXd>& density_projector,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_matrix) {
  const Eigen::MatrixXd fock_density_overlap =
      fock_matrix * density_projector * overlap_matrix;
  return fock_density_overlap - fock_density_overlap.transpose();
}

double inner_product(
    const Eigen::Ref<const Eigen::MatrixXd>& left,
    const Eigen::Ref<const Eigen::MatrixXd>& right) {
  if (left.rows() != right.rows() || left.cols() != right.cols()) {
    throw std::invalid_argument("matrices must have the same shape");
  }
  return (left.array() * right.array()).sum();
}

struct CdiisEntry {
  Eigen::MatrixXd fock_matrix;
  Eigen::MatrixXd error_matrix;
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
    Eigen::MatrixXd* mixed_fock_matrix) {
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
          history[row].error_matrix,
          history[column].error_matrix);
    }
    augmented_matrix(row, n_entries) = -1.0;
    augmented_matrix(n_entries, row) = -1.0;
  }

  Eigen::FullPivLU<Eigen::MatrixXd> lu(augmented_matrix);
  if (!lu.isInvertible()) {
    return false;
  }

  const Eigen::VectorXd solution = lu.solve(rhs);
  *mixed_fock_matrix = Eigen::MatrixXd::Zero(
      history.front().fock_matrix.rows(),
      history.front().fock_matrix.cols());
  for (int entry_index = 0; entry_index < n_entries; ++entry_index) {
    const double coefficient = solution(entry_index);
    *mixed_fock_matrix += coefficient * history[entry_index].fock_matrix;
  }
  return true;
}

double compute_electronic_energy(
    const Eigen::Ref<const Eigen::MatrixXd>& density_projector,
    const Eigen::Ref<const Eigen::MatrixXd>& core_hamiltonian_matrix,
    const Eigen::Ref<const Eigen::MatrixXd>& fock_matrix) {
  if (density_projector.rows() != core_hamiltonian_matrix.rows() ||
      density_projector.cols() != core_hamiltonian_matrix.cols() ||
      density_projector.rows() != fock_matrix.rows() ||
      density_projector.cols() != fock_matrix.cols()) {
    throw std::invalid_argument("RHF energy inputs must have matching shapes");
  }
  return (density_projector.array() *
          (core_hamiltonian_matrix.array() + fock_matrix.array())).sum();
}

}  // namespace

CppRestrictedHartreeFockSolver::CppRestrictedHartreeFockSolver(
    CppRestrictedHartreeFockOptions options)
    : options_(options) {}

CppRestrictedHartreeFockResult CppRestrictedHartreeFockSolver::solve(
    int n_total_electrons,
    const Eigen::Ref<const Eigen::MatrixXd>& ao_overlap_matrix,
    const AoIntegralInput& ao_integral_input) const {
  return solve(
      n_total_electrons,
      ao_overlap_matrix,
      ao_integral_input,
      {});
}

CppRestrictedHartreeFockResult CppRestrictedHartreeFockSolver::solve(
    int n_total_electrons,
    const Eigen::Ref<const Eigen::MatrixXd>& ao_overlap_matrix,
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
      symmetrize_matrix(ao_overlap_matrix);
  const auto symmetric_core_hamiltonian =
      symmetrize_matrix(ao_integral_input.ao_core_hamiltonian_matrix);

  core::GeneralizedEigensolver eigensolver;
  CppClosedShellFockBuilder fock_builder;

  auto core_eigen_result = eigensolver.solve(
      flatten_matrix_column_major(symmetric_core_hamiltonian),
      flatten_matrix_column_major(symmetric_overlap_matrix),
      n_basis_functions);
  Eigen::MatrixXd density_projector;
  if (!initial_density_projector.empty()) {
    if (initial_density_projector.size() !=
        n_basis_functions * n_basis_functions) {
      throw std::invalid_argument(
          "initial_density_projector size does not match n_basis_functions");
    }
    density_projector = symmetrize_matrix(initial_density_projector, n_basis_functions);
  } else {
    const Eigen::Map<const Eigen::MatrixXd> core_eigenvectors(
        core_eigen_result.eigenvector_matrix.data(),
        n_basis_functions,
        n_basis_functions);
    density_projector = build_density_projector(
        core_eigenvectors,
        n_basis_functions,
        n_occupied_orbitals);
  }

  CppRestrictedHartreeFockResult result;
  result.orbital_energies = Eigen::Map<const Eigen::VectorXd>(
      core_eigen_result.eigenvalues.data(),
      static_cast<Eigen::Index>(core_eigen_result.eigenvalues.size()));
  result.molecular_orbital_matrix = Eigen::Map<const Eigen::MatrixXd>(
      core_eigen_result.eigenvector_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  std::vector<CdiisEntry> diis_history;
  diis_history.reserve(std::max(0, options_.diis_history));

  for (int iteration = 0; iteration < options_.max_iterations; ++iteration) {
    auto symmetric_fock_matrix =
        fock_builder.build(density_projector, ao_integral_input);
    const auto cdiis_error_matrix = build_cdiis_error_matrix(
        symmetric_fock_matrix,
        density_projector,
        symmetric_overlap_matrix);
    append_cdiis_entry(
        {symmetric_fock_matrix, cdiis_error_matrix},
        options_.diis_history,
        &diis_history);
    if (iteration + 1 >= options_.diis_start_iteration) {
      Eigen::MatrixXd cdiis_fock_matrix;
      if (try_build_cdiis_fock(diis_history, &cdiis_fock_matrix)) {
        symmetric_fock_matrix = std::move(cdiis_fock_matrix);
      }
    }
    auto fock_eigen_result = eigensolver.solve(
        flatten_matrix_column_major(symmetric_fock_matrix),
        flatten_matrix_column_major(symmetric_overlap_matrix),
        n_basis_functions);
    const Eigen::Map<const Eigen::MatrixXd> fock_eigenvectors(
        fock_eigen_result.eigenvector_matrix.data(),
        n_basis_functions,
        n_basis_functions);
    const auto new_density_projector = build_density_projector(
        fock_eigenvectors,
        n_basis_functions,
        n_occupied_orbitals);

    result.iterations = iteration + 1;
    result.fock_matrix = symmetric_fock_matrix;
    result.orbital_energies = Eigen::Map<const Eigen::VectorXd>(
        fock_eigen_result.eigenvalues.data(),
        static_cast<Eigen::Index>(fock_eigen_result.eigenvalues.size()));
    result.molecular_orbital_matrix = fock_eigenvectors;
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

  result.fock_matrix =
      fock_builder.build(result.density_projector, ao_integral_input);
  auto final_eigen_result = eigensolver.solve(
      flatten_matrix_column_major(result.fock_matrix),
      flatten_matrix_column_major(symmetric_overlap_matrix),
      n_basis_functions);
  result.orbital_energies = Eigen::Map<const Eigen::VectorXd>(
      final_eigen_result.eigenvalues.data(),
      static_cast<Eigen::Index>(final_eigen_result.eigenvalues.size()));
  result.molecular_orbital_matrix = Eigen::Map<const Eigen::MatrixXd>(
      final_eigen_result.eigenvector_matrix.data(),
      n_basis_functions,
      n_basis_functions);
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
