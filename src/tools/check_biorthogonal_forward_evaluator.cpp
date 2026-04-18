#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "vb/biorthogonal_vbscf/biorthogonal_forward_evaluator.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_determinant_hamiltonian.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_orbital_integrals.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_structure_expansion.hpp"
#include "vb/matrices/structure_types.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"

namespace {

using xmvb::vb::FullDeterminantStructureData;
using xmvb::vb::StructureExpansionTerm;
using xmvb::vb::biorthogonal_vbscf::BiorthogonalForwardEvaluationResult;
using xmvb::vb::biorthogonal_vbscf::BiorthogonalOrbitalIntegrals;
using xmvb::vb::biorthogonal_vbscf::BiorthogonalStructureExpansion;
using xmvb::vb::biorthogonal_vbscf::build_biorthogonal_determinant_hamiltonian_matrix;
using xmvb::vb::biorthogonal_vbscf::build_biorthogonal_orbital_integrals;
using xmvb::vb::biorthogonal_vbscf::build_biorthogonal_structure_expansion;
using xmvb::vb::biorthogonal_vbscf::evaluate_biorthogonal_forward;
using xmvb::vb::biorthogonal_vbscf::validate_biorthogonal_forward_evaluation_result;

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

std::vector<double> make_zero_packed_two_electron_integrals(int n_orbitals) {
  const int last_pair_index =
      xmvb::vb::TwoElectronIndexer::packed_pair_index(
          n_orbitals - 1,
          n_orbitals - 1);
  const int last_storage_index =
      xmvb::vb::TwoElectronIndexer::packed_pair_of_pairs_index(
          last_pair_index,
          last_pair_index);
  return std::vector<double>(static_cast<std::size_t>(last_storage_index + 1), 0.0);
}

void test_forward_evaluator_matches_fixed_metric_reference_problem() {
  FullDeterminantStructureData structure_data;
  structure_data.n_structures = 2;
  structure_data.n_active_orbitals = 2;
  structure_data.alpha_det = {
      std::vector<int>{0},
      std::vector<int>{1},
  };
  structure_data.beta_det = {
      std::vector<int>{},
      std::vector<int>{},
  };
  structure_data.determinant_to_structure_terms = {
      std::vector<StructureExpansionTerm>{
          {0, 1.0},
          {1, 1.0},
      },
      std::vector<StructureExpansionTerm>{
          {1, 1.0},
      },
  };
  structure_data.ovlp_act = {
      1.0, 0.0,
      0.0, 1.0,
  };
  structure_data.h1e_act = {
      1.0, 0.0,
      2.0, 3.0,
  };
  structure_data.eri_act = make_zero_packed_two_electron_integrals(2);

  const BiorthogonalForwardEvaluationResult result =
      evaluate_biorthogonal_forward(structure_data, 1.0e-10);
  const BiorthogonalStructureExpansion structure_expansion =
      build_biorthogonal_structure_expansion(structure_data);
  const BiorthogonalOrbitalIntegrals orbital_integrals =
      build_biorthogonal_orbital_integrals(
          structure_data.n_active_orbitals,
          structure_data.ovlp_act,
          structure_data.h1e_act);
  const auto determinant_hamiltonian =
      build_biorthogonal_determinant_hamiltonian_matrix(
          structure_expansion.determinants,
          orbital_integrals,
          xmvb::vb::make_active_space_two_electron_view(structure_data.eri_act));
  const auto explicit_selected_hamiltonian =
      structure_expansion.structure_to_determinant.transpose() *
      determinant_hamiltonian *
      structure_expansion.structure_to_determinant;

  validate_biorthogonal_forward_evaluation_result(
      result,
      1.0e-12,
      1.0e-10,
      1.0e-10,
      1.0e-10);
  require_close(
      result.structure_expansion.structure_to_determinant(0, 0),
      1.0,
      1.0e-12,
      "T(0,0)");
  require_close(
      result.structure_expansion.structure_to_determinant(0, 1),
      1.0,
      1.0e-12,
      "T(0,1)");
  require_close(
      result.structure_expansion.structure_to_determinant(1, 0),
      0.0,
      1.0e-12,
      "T(1,0)");
  require_close(
      result.structure_expansion.structure_to_determinant(1, 1),
      1.0,
      1.0e-12,
      "T(1,1)");
  require_close(
      determinant_hamiltonian(0, 0),
      1.0,
      1.0e-12,
      "h(0,0)");
  require_close(
      determinant_hamiltonian(0, 1),
      2.0,
      1.0e-12,
      "h(0,1)");
  require_close(
      determinant_hamiltonian(1, 0),
      0.0,
      1.0e-12,
      "h(1,0)");
  require_close(
      determinant_hamiltonian(1, 1),
      3.0,
      1.0e-12,
      "h(1,1)");
  require_close(
      result.structure_hamiltonian_build_result.selected_structure_hamiltonian(0, 0),
      explicit_selected_hamiltonian(0, 0),
      1.0e-12,
      "H(0,0)");
  require_close(
      result.structure_hamiltonian_build_result.selected_structure_hamiltonian(0, 1),
      explicit_selected_hamiltonian(0, 1),
      1.0e-12,
      "H(0,1)");
  require_close(
      result.structure_hamiltonian_build_result.selected_structure_hamiltonian(1, 0),
      explicit_selected_hamiltonian(1, 0),
      1.0e-12,
      "H(1,0)");
  require_close(
      result.structure_hamiltonian_build_result.selected_structure_hamiltonian(1, 1),
      explicit_selected_hamiltonian(1, 1),
      1.0e-12,
      "H(1,1)");
  require_close(
      result.structure_hamiltonian_build_result.selected_structure_hamiltonian(0, 0),
      1.0,
      1.0e-12,
      "projected H(0,0) reference");
  require_close(
      result.structure_hamiltonian_build_result.selected_structure_hamiltonian(0, 1),
      3.0,
      1.0e-12,
      "projected H(0,1) reference");
  require_close(
      result.structure_hamiltonian_build_result.selected_structure_hamiltonian(1, 0),
      1.0,
      1.0e-12,
      "projected H(1,0) reference");
  require_close(
      result.structure_hamiltonian_build_result.selected_structure_hamiltonian(1, 1),
      6.0,
      1.0e-12,
      "projected H(1,1) reference");
  require_close(result.solve_result.eigenvalues[0], 1.0, 1.0e-10, "eigenvalue 0");
  require_close(result.solve_result.eigenvalues[1], 3.0, 1.0e-10, "eigenvalue 1");
}

}  // namespace

int main() {
  try {
    test_forward_evaluator_matches_fixed_metric_reference_problem();
  } catch (const std::exception& error) {
    std::cerr << "check_biorthogonal_forward_evaluator failed: "
              << error.what() << '\n';
    return 1;
  }

  std::cout << "check_biorthogonal_forward_evaluator passed\n";
  return 0;
}
