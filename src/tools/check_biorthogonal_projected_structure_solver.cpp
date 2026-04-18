#include <cmath>
#include <iostream>
#include <stdexcept>

#include "vb/biorthogonal_vbscf/biorthogonal_projected_solver.hpp"

namespace {

using xmvb::vb::biorthogonal_vbscf::BiorthogonalProjectedStructureProblem;
using xmvb::vb::biorthogonal_vbscf::BiorthogonalProjectedStructureSolveResult;
using xmvb::vb::biorthogonal_vbscf::BiorthogonalSelectedStructureSpace;
using xmvb::vb::biorthogonal_vbscf::DenseMatrix;
using xmvb::vb::biorthogonal_vbscf::build_biorthogonal_projected_structure_problem;
using xmvb::vb::biorthogonal_vbscf::build_biorthogonal_selected_structure_space;
using xmvb::vb::biorthogonal_vbscf::solve_biorthogonal_projected_structure_problem;
using xmvb::vb::biorthogonal_vbscf::validate_biorthogonal_projected_structure_solve_result;

void require_close(
    double actual,
    double expected,
    double tolerance,
    const char* label) {
  if (std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(
        std::string(label) + " mismatch: actual=" + std::to_string(actual) +
        " expected=" + std::to_string(expected));
  }
}

void test_projected_solver_on_fixed_metric_nonhermitian_problem() {
  DenseMatrix determinant_hamiltonian(2, 2);
  determinant_hamiltonian << 1.0, 2.0,
                             0.0, 3.0;

  DenseMatrix structure_to_determinant(2, 2);
  structure_to_determinant << 1.0, 1.0,
                              0.0, 1.0;

  const BiorthogonalSelectedStructureSpace structure_space =
      build_biorthogonal_selected_structure_space(structure_to_determinant);
  const BiorthogonalProjectedStructureProblem projected_problem =
      build_biorthogonal_projected_structure_problem(
          determinant_hamiltonian,
          structure_space);
  const BiorthogonalProjectedStructureSolveResult solve_result =
      solve_biorthogonal_projected_structure_problem(
          projected_problem,
          structure_space,
          1.0e-10);

  validate_biorthogonal_projected_structure_solve_result(
      solve_result,
      2,
      1.0e-10,
      1.0e-10);
  require_close(solve_result.eigenvalues[0], 1.0, 1.0e-10, "eigenvalue 0");
  require_close(solve_result.eigenvalues[1], 3.0, 1.0e-10, "eigenvalue 1");
}

}  // namespace

int main() {
  try {
    test_projected_solver_on_fixed_metric_nonhermitian_problem();
  } catch (const std::exception& error) {
    std::cerr << "check_biorthogonal_projected_structure_solver failed: "
              << error.what() << '\n';
    return 1;
  }

  std::cout << "check_biorthogonal_projected_structure_solver passed\n";
  return 0;
}
