#include "vb/matrices/full_structure_builder.hpp"

#include <stdexcept>
#include <vector>

#include "vb/matrices/biorthogonal_spin_pair.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace xmvb::vb {

namespace {

struct SpinDeterminantPairResult {
  DeterminantOverlapResult det_ovlp_result;
  double one_electron_hamiltonian = 0.0;
  double total_hamiltonian = 0.0;
};

struct FullDeterminantPairResult {
  double overlap_determinant = 0.0;
  double one_electron_hamiltonian = 0.0;
  double total_hamiltonian = 0.0;
};

struct DeterminantContribution {
  int determinant_index = -1;
  std::vector<StructureExpansionTerm> structure_terms;
};

void validate_full_determinant_input(
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
    int n_orbitals,
    int n_structures) {
  const int n_determinants = static_cast<int>(alpha_det.size());
  if (n_determinants <= 0) {
    throw std::invalid_argument("at least one determinant is required");
  }
  if (static_cast<int>(beta_det.size()) != n_determinants ||
      static_cast<int>(determinant_to_structure_terms.size()) != n_determinants) {
    throw std::invalid_argument("all determinant-indexed inputs must have the same size");
  }
  if (n_orbitals <= 0 || n_structures <= 0) {
    throw std::invalid_argument("n_orbitals and n_structures must be positive");
  }
}

std::vector<double> build_overlap_submatrix(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& ovlp_act,
    int n_orbitals) {
  const int n_electrons = static_cast<int>(occ_L.size());
  if (static_cast<int>(occ_R.size()) != n_electrons) {
    throw std::invalid_argument("left and right occupation sizes must match");
  }

  std::vector<double> overlap_submatrix(
      static_cast<std::size_t>(n_electrons) * static_cast<std::size_t>(n_electrons),
      0.0);

  for (int left_column = 0; left_column < n_electrons; ++left_column) {
    const int orbital_index_left = occ_L[static_cast<std::size_t>(left_column)];
    for (int right_row = 0; right_row < n_electrons; ++right_row) {
      const int orbital_index_right = occ_R[static_cast<std::size_t>(right_row)];
      overlap_submatrix[static_cast<std::size_t>(left_column) * n_electrons + right_row] =
          ovlp_act[static_cast<std::size_t>(orbital_index_left) * n_orbitals +
                   orbital_index_right];
    }
  }

  return overlap_submatrix;
}

SpinDeterminantPairResult evaluate_spin_determinant_pair(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& ovlp_act,
    const std::vector<double>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act,
    const DeterminantOverlapResolver& determinant_overlap_resolver,
    const DeterminantHamiltonianResolver& determinant_hamiltonian_resolver) {

  SpinDeterminantPairResult result;
  if (occ_L.empty()) {
    result.det_ovlp_result.overlap_determinant = 1.0;
    return result;
  }

  const auto overlap_submatrix = build_overlap_submatrix(
      occ_L,
      occ_R,
      ovlp_act,
      n_orbitals);

  const auto det_ovlp_result = determinant_overlap_resolver.resolve(
      overlap_submatrix,
      static_cast<int>(occ_L.size()));

  const auto hamiltonian_result = determinant_hamiltonian_resolver.resolve(
      occ_L,
      occ_R,
      overlap_submatrix,
      det_ovlp_result,
      h1e_act,
      n_orbitals,
      eri_act);

  result.det_ovlp_result = det_ovlp_result;
  result.one_electron_hamiltonian = hamiltonian_result.one_electron_hamiltonian;
  result.total_hamiltonian = hamiltonian_result.total_hamiltonian;

  return result;
}

double evaluate_opposite_spin_coulomb_coupling(
    VBSCFAlgorithm algorithm,
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const SpinDeterminantPairResult& alpha_result,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const SpinDeterminantPairResult& beta_result,
    const std::vector<double>& eri_act) {
  if (algorithm == VBSCFAlgorithm::Biorthogonal) {
    return compute_opposite_spin_biorthogonal_hamiltonian(
        alpha_occ_L,
        alpha_occ_R,
        alpha_result.det_ovlp_result,
        beta_occ_L,
        beta_occ_R,
        beta_result.det_ovlp_result,
        eri_act);
  }

  if (alpha_occ_L.empty() || beta_occ_L.empty()) {
    return 0.0;
  }
  if (alpha_result.det_ovlp_result.nullity >= 2 || beta_result.det_ovlp_result.nullity >= 2) {
    return 0.0;
  }

  const Matrix alpha_cofactor_1st =
      calc_cofactor_1st(alpha_result.det_ovlp_result);

  const Matrix beta_cofactor_1st =
      calc_cofactor_1st(beta_result.det_ovlp_result);
  const int n_alpha_electrons = static_cast<int>(alpha_occ_L.size());
  const int n_beta_electrons = static_cast<int>(beta_occ_L.size());
  double opposite_spin_coulomb_coupling = 0.0;

  for (int alpha_left_column = 0; alpha_left_column < n_alpha_electrons; ++alpha_left_column) {
    const int alpha_orbital_left =
        alpha_occ_L[static_cast<std::size_t>(alpha_left_column)];
    for (int alpha_right_row = 0; alpha_right_row < n_alpha_electrons; ++alpha_right_row) {
      const int alpha_orbital_right =
          alpha_occ_R[static_cast<std::size_t>(alpha_right_row)];
      const double alpha_cofactor =
          alpha_cofactor_1st(alpha_right_row, alpha_left_column);

      for (int beta_left_column = 0; beta_left_column < n_beta_electrons; ++beta_left_column) {
        const int beta_orbital_left =
            beta_occ_L[static_cast<std::size_t>(beta_left_column)];
        for (int beta_right_row = 0; beta_right_row < n_beta_electrons; ++beta_right_row) {
          const int beta_orbital_right =
              beta_occ_R[static_cast<std::size_t>(beta_right_row)];
          const double beta_cofactor =
              beta_cofactor_1st(beta_right_row, beta_left_column);

          const int two_electron_index = TwoElectronIndexer::two_electron_storage_index(
              beta_orbital_right,
              beta_orbital_left,
              alpha_orbital_right,
              alpha_orbital_left);
          opposite_spin_coulomb_coupling +=
              alpha_cofactor * beta_cofactor *
              eri_act[static_cast<std::size_t>(two_electron_index)];
        }
      }
    }
  }

  return opposite_spin_coulomb_coupling;
}

FullDeterminantPairResult evaluate_full_determinant_pair(
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    const std::vector<double>& ovlp_act,
    const std::vector<double>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act,
    int left_det_idx,
    int right_det_idx,
    const DeterminantOverlapResolver& determinant_overlap_resolver,
    const DeterminantHamiltonianResolver& determinant_hamiltonian_resolver,
    VBSCFAlgorithm algorithm) {
  const auto alpha_result = evaluate_spin_determinant_pair(
      alpha_det[static_cast<std::size_t>(left_det_idx)],
      alpha_det[static_cast<std::size_t>(right_det_idx)],
      ovlp_act,
      h1e_act,
      n_orbitals,
      eri_act,
      determinant_overlap_resolver,
      determinant_hamiltonian_resolver);

  const auto beta_result = evaluate_spin_determinant_pair(
      beta_det[static_cast<std::size_t>(left_det_idx)],
      beta_det[static_cast<std::size_t>(right_det_idx)],
      ovlp_act,
      h1e_act,
      n_orbitals,
      eri_act,
      determinant_overlap_resolver,
      determinant_hamiltonian_resolver);

  FullDeterminantPairResult result;

  result.overlap_determinant =
      alpha_result.det_ovlp_result.overlap_determinant *
      beta_result.det_ovlp_result.overlap_determinant;
    
  result.one_electron_hamiltonian =
      alpha_result.one_electron_hamiltonian * beta_result.det_ovlp_result.overlap_determinant +
      beta_result.one_electron_hamiltonian * alpha_result.det_ovlp_result.overlap_determinant;
      
  result.total_hamiltonian =
      alpha_result.total_hamiltonian * beta_result.det_ovlp_result.overlap_determinant +
      beta_result.total_hamiltonian * alpha_result.det_ovlp_result.overlap_determinant +
      evaluate_opposite_spin_coulomb_coupling(
          algorithm,
          alpha_det[static_cast<std::size_t>(left_det_idx)],
          alpha_det[static_cast<std::size_t>(right_det_idx)],
          alpha_result,
          beta_det[static_cast<std::size_t>(left_det_idx)],
          beta_det[static_cast<std::size_t>(right_det_idx)],
          beta_result,
          eri_act);
  return result;
}

}  // namespace

FullDeterminantStructureHamiltonianOverlapBuilder::FullDeterminantStructureHamiltonianOverlapBuilder(
    VBSCFAlgorithm algorithm)
    : determinant_overlap_resolver_(),
      determinant_hamiltonian_resolver_(algorithm),
      structure_pair_accumulator_(),
      algorithm_(algorithm) {}

FullDeterminantStructureHamiltonianOverlapBuilder::FullDeterminantStructureHamiltonianOverlapBuilder(
    DeterminantOverlapResolver determinant_overlap_resolver,
    DeterminantHamiltonianResolver determinant_hamiltonian_resolver,
    StructurePairAccumulator structure_pair_accumulator,
    VBSCFAlgorithm algorithm)
    : determinant_overlap_resolver_(std::move(determinant_overlap_resolver)),
      determinant_hamiltonian_resolver_(std::move(determinant_hamiltonian_resolver)),
      structure_pair_accumulator_(std::move(structure_pair_accumulator)),
      algorithm_(algorithm) {}


StructureAccumulationResult FullDeterminantStructureHamiltonianOverlapBuilder::build(
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
    const std::vector<double>& ovlp_act,
    const std::vector<double>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act,
    int n_structures) const 
{
  validate_full_determinant_input(
      alpha_det,
      beta_det,
      determinant_to_structure_terms,
      n_orbitals,
      n_structures);

  const int n_determinants = static_cast<int>(alpha_det.size());
  StructureAccumulationResult accumulation_result =
      structure_pair_accumulator_.create_result(n_structures, n_determinants);

  for (int left_det_idx = 0; left_det_idx < n_determinants; ++left_det_idx) 
  {
    for (int right_det_idx = 0; right_det_idx < n_determinants; ++right_det_idx) 
    {
      const FullDeterminantPairResult determinant_pair_result =
          evaluate_full_determinant_pair(
              alpha_det,
              beta_det,
              ovlp_act,
              h1e_act,
              n_orbitals,
              eri_act,
              left_det_idx,
              right_det_idx,
              determinant_overlap_resolver_,
              determinant_hamiltonian_resolver_,
              algorithm_);

      structure_pair_accumulator_.accumulate(
          left_det_idx,
          right_det_idx,
          determinant_to_structure_terms[static_cast<std::size_t>(left_det_idx)],
          determinant_to_structure_terms[static_cast<std::size_t>(right_det_idx)],
          determinant_pair_result.overlap_determinant,
          determinant_pair_result.total_hamiltonian,
          determinant_pair_result.one_electron_hamiltonian,
          accumulation_result);
    }
  }

  // symmetric
  for (int row = 0; row < n_structures; ++row) {
    for (int column = 0; column < row; ++column) {
      const std::size_t upper_index =
          static_cast<std::size_t>(row) * n_structures + column;
      const std::size_t lower_index =
          static_cast<std::size_t>(column) * n_structures + row;

      accumulation_result.overlap_matrix[lower_index] =
          accumulation_result.overlap_matrix[upper_index];

      accumulation_result.hamiltonian_matrix[lower_index] =
          accumulation_result.hamiltonian_matrix[upper_index];

      accumulation_result.one_electron_hamiltonian_matrix[lower_index] =
          accumulation_result.one_electron_hamiltonian_matrix[upper_index];
    }
  }

  return accumulation_result;
}



StructureAccumulationResult FullDeterminantStructureHamiltonianOverlapBuilder::build(
    const FullDeterminantStructureData& input) const 
{
  return build(
      input.alpha_det,
      input.beta_det,
      input.determinant_to_structure_terms,
      input.ovlp_act,
      input.h1e_act,
      input.n_active_orbitals,
      input.eri_act,
      input.n_structures);
}


}  // namespace xmvb::vb
