#include "vb/biorthogonal_vbscf/biorthogonal_structure_expansion.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "vb/biorthogonal_vbscf/biorthogonal_structure_local_utils.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

namespace {

}  // namespace

BiorthogonalStructureExpansion build_biorthogonal_structure_expansion(
    const xmvb::vb::FullDeterminantStructureData& structure_data) {
  if (structure_data.n_structures <= 0) {
    throw std::invalid_argument("structure_data.n_structures must be positive");
  }
  if (structure_data.n_active_orbitals <= 0) {
    throw std::invalid_argument("structure_data.n_active_orbitals must be positive");
  }
  if (structure_data.alpha_det.size() != structure_data.beta_det.size() ||
      structure_data.alpha_det.size() !=
          structure_data.determinant_to_structure_terms.size()) {
    throw std::invalid_argument("full determinant structure arrays are inconsistent");
  }
  if (structure_data.alpha_det.empty()) {
    throw std::invalid_argument("at least one determinant is required");
  }

  const int n_determinants = static_cast<int>(structure_data.alpha_det.size());
  const int n_structures = structure_data.n_structures;

  BiorthogonalStructureExpansion structure_expansion;
  structure_expansion.determinants.reserve(xmvb::to_size(n_determinants));
  structure_expansion.structure_to_determinant =
      Eigen::MatrixXd::Zero(n_determinants, n_structures);

  for (int determinant_index = 0; determinant_index < n_determinants; ++determinant_index) {
    BiorthogonalDeterminant determinant{
        structure_data.alpha_det[xmvb::to_size(determinant_index)],
        structure_data.beta_det[xmvb::to_size(determinant_index)],
    };
    validate_biorthogonal_determinant(
        determinant,
        structure_data.n_active_orbitals);
    structure_expansion.determinants.push_back(std::move(determinant));

    // Each determinant row stores the signed expansion coefficients that map
    // selected-structure coefficients into determinant coefficients. Repeated
    // terms for the same `(determinant, structure)` pair are summed here so
    // the returned matrix is the exact algebraic `T`, not a sparse term list.
    for (const auto& term :
         structure_data.determinant_to_structure_terms[xmvb::to_size(determinant_index)]) {
      if (term.structure_index < 0 || term.structure_index >= n_structures) {
        throw std::out_of_range("structure expansion term index is out of range");
      }
      if (!std::isfinite(term.coefficient)) {
        throw std::runtime_error("structure expansion coefficient is not finite");
      }
      structure_expansion.structure_to_determinant(
          determinant_index,
          term.structure_index) += term.coefficient;
    }
  }

  validate_biorthogonal_structure_expansion(
      structure_expansion,
      structure_data.n_structures,
      structure_data.n_active_orbitals);
  return structure_expansion;
}

void validate_biorthogonal_structure_expansion(
    const BiorthogonalStructureExpansion& structure_expansion,
    int expected_structure_count,
    int expected_active_orbital_count) {
  if (expected_structure_count <= 0) {
    throw std::invalid_argument("expected_structure_count must be positive");
  }
  if (expected_active_orbital_count <= 0) {
    throw std::invalid_argument("expected_active_orbital_count must be positive");
  }
  if (structure_expansion.determinants.empty()) {
    throw std::invalid_argument("determinants must not be empty");
  }
  if (structure_expansion.structure_to_determinant.rows() !=
          static_cast<int>(structure_expansion.determinants.size()) ||
      structure_expansion.structure_to_determinant.cols() !=
          expected_structure_count) {
    throw std::invalid_argument(
        "structure_to_determinant dimensions are inconsistent");
  }
  throw_if_nonfinite(
      structure_expansion.structure_to_determinant,
      "structure_to_determinant");

  for (const auto& determinant : structure_expansion.determinants) {
    validate_biorthogonal_determinant(
        determinant,
        expected_active_orbital_count);
  }
}

}  // namespace xmvb::vb::biorthogonal_vbscf
