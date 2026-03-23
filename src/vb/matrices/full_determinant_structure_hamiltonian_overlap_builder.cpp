#include "vb/matrices/full_determinant_structure_hamiltonian_overlap_builder.hpp"

#include <stdexcept>
#include <vector>

#include "vb/matrices/biorthogonal_spin_pair.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace xmvb::vb {

namespace {

struct SpinDeterminantPairResult {
  DeterminantOverlapResult overlap_result;
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
    const std::vector<std::vector<int>>& alpha_occupied_orbitals_by_determinant,
    const std::vector<std::vector<int>>& beta_occupied_orbitals_by_determinant,
    const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
    int n_orbitals,
    int n_structures) {
  const int n_determinants = static_cast<int>(alpha_occupied_orbitals_by_determinant.size());
  if (n_determinants <= 0) {
    throw std::invalid_argument("at least one determinant is required");
  }
  if (static_cast<int>(beta_occupied_orbitals_by_determinant.size()) != n_determinants ||
      static_cast<int>(determinant_to_structure_terms.size()) != n_determinants) {
    throw std::invalid_argument("all determinant-indexed inputs must have the same size");
  }
  if (n_orbitals <= 0 || n_structures <= 0) {
    throw std::invalid_argument("n_orbitals and n_structures must be positive");
  }
}

std::vector<double> build_overlap_submatrix(
    const std::vector<int>& occupied_orbitals_left,
    const std::vector<int>& occupied_orbitals_right,
    const std::vector<double>& basis_overlap_matrix,
    int n_orbitals) {
  const int n_electrons = static_cast<int>(occupied_orbitals_left.size());
  if (static_cast<int>(occupied_orbitals_right.size()) != n_electrons) {
    throw std::invalid_argument("left and right occupation sizes must match");
  }

  std::vector<double> overlap_submatrix(
      static_cast<std::size_t>(n_electrons) * static_cast<std::size_t>(n_electrons),
      0.0);

  for (int left_column = 0; left_column < n_electrons; ++left_column) {
    const int orbital_index_left = occupied_orbitals_left[static_cast<std::size_t>(left_column)];
    for (int right_row = 0; right_row < n_electrons; ++right_row) {
      const int orbital_index_right = occupied_orbitals_right[static_cast<std::size_t>(right_row)];
      overlap_submatrix[static_cast<std::size_t>(left_column) * n_electrons + right_row] =
          basis_overlap_matrix[static_cast<std::size_t>(orbital_index_left) * n_orbitals +
                               orbital_index_right];
    }
  }

  return overlap_submatrix;
}

SpinDeterminantPairResult evaluate_spin_determinant_pair(
    const std::vector<int>& occupied_orbitals_left,
    const std::vector<int>& occupied_orbitals_right,
    const std::vector<double>& basis_overlap_matrix,
    const std::vector<double>& one_electron_matrix,
    int n_orbitals,
    const std::vector<double>& packed_two_electron_integrals,
    const DeterminantOverlapResolver& determinant_overlap_resolver,
    const DeterminantHamiltonianResolver& determinant_hamiltonian_resolver) {
  SpinDeterminantPairResult result;
  if (occupied_orbitals_left.empty()) {
    result.overlap_result.overlap_determinant = 1.0;
    return result;
  }

  const auto overlap_submatrix = build_overlap_submatrix(
      occupied_orbitals_left,
      occupied_orbitals_right,
      basis_overlap_matrix,
      n_orbitals);

  const auto overlap_result = determinant_overlap_resolver.resolve(
      overlap_submatrix,
      static_cast<int>(occupied_orbitals_left.size()));
  const auto hamiltonian_result = determinant_hamiltonian_resolver.resolve(
      occupied_orbitals_left,
      occupied_orbitals_right,
      overlap_submatrix,
      overlap_result,
      one_electron_matrix,
      n_orbitals,
      packed_two_electron_integrals);

  result.overlap_result = overlap_result;
  result.one_electron_hamiltonian = hamiltonian_result.one_electron_hamiltonian;
  result.total_hamiltonian = hamiltonian_result.total_hamiltonian;
  return result;
}

double evaluate_opposite_spin_coulomb_coupling(
    VbScfAlgorithm algorithm,
    const std::vector<int>& alpha_occupied_orbitals_left,
    const std::vector<int>& alpha_occupied_orbitals_right,
    const SpinDeterminantPairResult& alpha_result,
    const std::vector<int>& beta_occupied_orbitals_left,
    const std::vector<int>& beta_occupied_orbitals_right,
    const SpinDeterminantPairResult& beta_result,
    const std::vector<double>& packed_two_electron_integrals) {
  if (algorithm == VbScfAlgorithm::Biorthogonal) {
    return compute_opposite_spin_biorthogonal_hamiltonian(
        alpha_occupied_orbitals_left,
        alpha_occupied_orbitals_right,
        alpha_result.overlap_result,
        beta_occupied_orbitals_left,
        beta_occupied_orbitals_right,
        beta_result.overlap_result,
        packed_two_electron_integrals);
  }

  if (alpha_occupied_orbitals_left.empty() || beta_occupied_orbitals_left.empty()) {
    return 0.0;
  }
  if (alpha_result.overlap_result.nullity >= 2 || beta_result.overlap_result.nullity >= 2) {
    return 0.0;
  }

  const ColumnMajorMatrixXd alpha_first_order_cofactor_matrix =
      build_first_order_cofactor_matrix_from_result(alpha_result.overlap_result);
  const ColumnMajorMatrixXd beta_first_order_cofactor_matrix =
      build_first_order_cofactor_matrix_from_result(beta_result.overlap_result);
  const int n_alpha_electrons = static_cast<int>(alpha_occupied_orbitals_left.size());
  const int n_beta_electrons = static_cast<int>(beta_occupied_orbitals_left.size());
  double opposite_spin_coulomb_coupling = 0.0;

  for (int alpha_left_column = 0; alpha_left_column < n_alpha_electrons; ++alpha_left_column) {
    const int alpha_orbital_left =
        alpha_occupied_orbitals_left[static_cast<std::size_t>(alpha_left_column)];
    for (int alpha_right_row = 0; alpha_right_row < n_alpha_electrons; ++alpha_right_row) {
      const int alpha_orbital_right =
          alpha_occupied_orbitals_right[static_cast<std::size_t>(alpha_right_row)];
      const double alpha_cofactor =
          alpha_first_order_cofactor_matrix(alpha_right_row, alpha_left_column);

      for (int beta_left_column = 0; beta_left_column < n_beta_electrons; ++beta_left_column) {
        const int beta_orbital_left =
            beta_occupied_orbitals_left[static_cast<std::size_t>(beta_left_column)];
        for (int beta_right_row = 0; beta_right_row < n_beta_electrons; ++beta_right_row) {
          const int beta_orbital_right =
              beta_occupied_orbitals_right[static_cast<std::size_t>(beta_right_row)];
          const double beta_cofactor =
              beta_first_order_cofactor_matrix(beta_right_row, beta_left_column);

          const int two_electron_index = TwoElectronIndexer::two_electron_storage_index(
              beta_orbital_right,
              beta_orbital_left,
              alpha_orbital_right,
              alpha_orbital_left);
          opposite_spin_coulomb_coupling +=
              alpha_cofactor * beta_cofactor *
              packed_two_electron_integrals[static_cast<std::size_t>(two_electron_index)];
        }
      }
    }
  }

  return opposite_spin_coulomb_coupling;
}

FullDeterminantPairResult evaluate_full_determinant_pair(
    const std::vector<std::vector<int>>& alpha_occupied_orbitals_by_determinant,
    const std::vector<std::vector<int>>& beta_occupied_orbitals_by_determinant,
    const std::vector<double>& basis_overlap_matrix,
    const std::vector<double>& one_electron_matrix,
    int n_orbitals,
    const std::vector<double>& packed_two_electron_integrals,
    int determinant_index_left,
    int determinant_index_right,
    const DeterminantOverlapResolver& determinant_overlap_resolver,
    const DeterminantHamiltonianResolver& determinant_hamiltonian_resolver,
    VbScfAlgorithm algorithm) {
  const auto alpha_result = evaluate_spin_determinant_pair(
      alpha_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_left)],
      alpha_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_right)],
      basis_overlap_matrix,
      one_electron_matrix,
      n_orbitals,
      packed_two_electron_integrals,
      determinant_overlap_resolver,
      determinant_hamiltonian_resolver);
  const auto beta_result = evaluate_spin_determinant_pair(
      beta_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_left)],
      beta_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_right)],
      basis_overlap_matrix,
      one_electron_matrix,
      n_orbitals,
      packed_two_electron_integrals,
      determinant_overlap_resolver,
      determinant_hamiltonian_resolver);

  FullDeterminantPairResult result;
  result.overlap_determinant =
      alpha_result.overlap_result.overlap_determinant *
      beta_result.overlap_result.overlap_determinant;
  result.one_electron_hamiltonian =
      alpha_result.one_electron_hamiltonian * beta_result.overlap_result.overlap_determinant +
      beta_result.one_electron_hamiltonian * alpha_result.overlap_result.overlap_determinant;
  result.total_hamiltonian =
      alpha_result.total_hamiltonian * beta_result.overlap_result.overlap_determinant +
      beta_result.total_hamiltonian * alpha_result.overlap_result.overlap_determinant +
      evaluate_opposite_spin_coulomb_coupling(
          algorithm,
          alpha_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_left)],
          alpha_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_right)],
          alpha_result,
          beta_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_left)],
          beta_occupied_orbitals_by_determinant[static_cast<std::size_t>(determinant_index_right)],
          beta_result,
          packed_two_electron_integrals);
  return result;
}

std::vector<int> build_selected_structure_remap(
    int n_structures,
    const std::vector<int>& selected_structure_indices,
    int candidate_structure_index) {
  if (selected_structure_indices.empty()) {
    throw std::invalid_argument("selected_structure_indices must not be empty");
  }
  if (candidate_structure_index < 0 || candidate_structure_index >= n_structures) {
    throw std::out_of_range("candidate structure index is out of range");
  }

  std::vector<int> selected_structure_remap(static_cast<std::size_t>(n_structures), -1);
  for (std::size_t selected_offset = 0;
       selected_offset < selected_structure_indices.size();
       ++selected_offset) {
    const int structure_index = selected_structure_indices[selected_offset];
    if (structure_index < 0 || structure_index >= n_structures) {
      throw std::out_of_range("selected structure index is out of range");
    }
    if (selected_structure_remap[static_cast<std::size_t>(structure_index)] >= 0) {
      throw std::invalid_argument("selected_structure_indices must be unique");
    }
    if (structure_index == candidate_structure_index) {
      throw std::invalid_argument("candidate structure must not already be selected");
    }
    selected_structure_remap[static_cast<std::size_t>(structure_index)] =
        static_cast<int>(selected_offset);
  }
  return selected_structure_remap;
}

std::vector<DeterminantContribution> collect_determinant_contributions(
    const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
    const std::vector<int>& selected_structure_remap,
    int candidate_structure_index,
    int candidate_local_index,
    bool collect_selected_terms) {
  std::vector<DeterminantContribution> determinant_contributions;
  for (std::size_t determinant_index = 0;
       determinant_index < determinant_to_structure_terms.size();
       ++determinant_index) {
    DeterminantContribution contribution;
    contribution.determinant_index = static_cast<int>(determinant_index);
    for (const auto& structure_term :
         determinant_to_structure_terms[determinant_index]) {
      if (structure_term.structure_index < 0 ||
          static_cast<std::size_t>(structure_term.structure_index) >=
              selected_structure_remap.size()) {
        throw std::invalid_argument("structure expansion term index out of range");
      }

      if (collect_selected_terms) {
        const int remapped_index =
            selected_structure_remap[static_cast<std::size_t>(structure_term.structure_index)];
        if (remapped_index < 0) {
          continue;
        }
        contribution.structure_terms.push_back(
            StructureExpansionTerm{remapped_index, structure_term.coefficient});
        continue;
      }

      if (structure_term.structure_index == candidate_structure_index) {
        contribution.structure_terms.push_back(
            StructureExpansionTerm{candidate_local_index, structure_term.coefficient});
      }
    }

    if (!contribution.structure_terms.empty()) {
      determinant_contributions.push_back(std::move(contribution));
    }
  }
  return determinant_contributions;
}

}  // namespace

FullDeterminantStructureHamiltonianOverlapBuilder::FullDeterminantStructureHamiltonianOverlapBuilder(
    VbScfAlgorithm algorithm)
    : determinant_overlap_resolver_(),
      determinant_hamiltonian_resolver_(algorithm),
      structure_pair_accumulator_(),
      algorithm_(algorithm) {}

FullDeterminantStructureHamiltonianOverlapBuilder::FullDeterminantStructureHamiltonianOverlapBuilder(
    DeterminantOverlapResolver determinant_overlap_resolver,
    DeterminantHamiltonianResolver determinant_hamiltonian_resolver,
    StructurePairAccumulator structure_pair_accumulator,
    VbScfAlgorithm algorithm)
    : determinant_overlap_resolver_(std::move(determinant_overlap_resolver)),
      determinant_hamiltonian_resolver_(std::move(determinant_hamiltonian_resolver)),
      structure_pair_accumulator_(std::move(structure_pair_accumulator)),
      algorithm_(algorithm) {}

StructureAccumulationResult FullDeterminantStructureHamiltonianOverlapBuilder::build(
    const std::vector<std::vector<int>>& alpha_occupied_orbitals_by_determinant,
    const std::vector<std::vector<int>>& beta_occupied_orbitals_by_determinant,
    const std::vector<std::vector<StructureExpansionTerm>>& determinant_to_structure_terms,
    const std::vector<double>& basis_overlap_matrix,
    const std::vector<double>& one_electron_matrix,
    int n_orbitals,
    const std::vector<double>& packed_two_electron_integrals,
    int n_structures) const {
  validate_full_determinant_input(
      alpha_occupied_orbitals_by_determinant,
      beta_occupied_orbitals_by_determinant,
      determinant_to_structure_terms,
      n_orbitals,
      n_structures);

  const int n_determinants = static_cast<int>(alpha_occupied_orbitals_by_determinant.size());
  StructureAccumulationResult accumulation_result =
      structure_pair_accumulator_.create_result(n_structures, n_determinants);

  for (int determinant_index_left = 0; determinant_index_left < n_determinants; ++determinant_index_left) {
    for (int determinant_index_right = 0;
         determinant_index_right < n_determinants;
         ++determinant_index_right) {
      const FullDeterminantPairResult determinant_pair_result =
          evaluate_full_determinant_pair(
              alpha_occupied_orbitals_by_determinant,
              beta_occupied_orbitals_by_determinant,
              basis_overlap_matrix,
              one_electron_matrix,
              n_orbitals,
              packed_two_electron_integrals,
              determinant_index_left,
              determinant_index_right,
              determinant_overlap_resolver_,
              determinant_hamiltonian_resolver_,
              algorithm_);

      structure_pair_accumulator_.accumulate(
          determinant_index_left,
          determinant_index_right,
          determinant_to_structure_terms[static_cast<std::size_t>(determinant_index_left)],
          determinant_to_structure_terms[static_cast<std::size_t>(determinant_index_right)],
          determinant_pair_result.overlap_determinant,
          determinant_pair_result.total_hamiltonian,
          determinant_pair_result.one_electron_hamiltonian,
          accumulation_result);
    }
  }

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
    const FullDeterminantStructureData& input) const {
  return build(
      input.alpha_occupied_orbitals_by_determinant,
      input.beta_occupied_orbitals_by_determinant,
      input.determinant_to_structure_terms,
      input.basis_overlap_matrix,
      input.one_electron_matrix,
      input.n_active_orbitals,
      input.packed_two_electron_integrals,
      input.n_structures);
}

StructureCandidateCouplingResult
FullDeterminantStructureHamiltonianOverlapBuilder::build_candidate_coupling(
    const FullDeterminantStructureData& input,
    const std::vector<int>& selected_structure_indices,
    int candidate_structure_index) const {
  validate_full_determinant_input(
      input.alpha_occupied_orbitals_by_determinant,
      input.beta_occupied_orbitals_by_determinant,
      input.determinant_to_structure_terms,
      input.n_active_orbitals,
      input.n_structures);

  const std::vector<int> selected_structure_remap =
      build_selected_structure_remap(
          input.n_structures,
          selected_structure_indices,
          candidate_structure_index);
  const int candidate_local_index = static_cast<int>(selected_structure_indices.size());

  const std::vector<DeterminantContribution> selected_determinants =
      collect_determinant_contributions(
          input.determinant_to_structure_terms,
          selected_structure_remap,
          candidate_structure_index,
          candidate_local_index,
          true);
  const std::vector<DeterminantContribution> candidate_determinants =
      collect_determinant_contributions(
          input.determinant_to_structure_terms,
          selected_structure_remap,
          candidate_structure_index,
          candidate_local_index,
          false);
  if (candidate_determinants.empty()) {
    throw std::invalid_argument("candidate structure must contribute at least one determinant");
  }

  StructureCandidateCouplingResult result;
  result.candidate_structure_index = candidate_structure_index;
  result.h_column.assign(static_cast<std::size_t>(candidate_local_index), 0.0);
  result.s_column.assign(static_cast<std::size_t>(candidate_local_index), 0.0);

  for (const auto& selected_determinant : selected_determinants) {
    for (const auto& candidate_determinant : candidate_determinants) {
      const FullDeterminantPairResult determinant_pair_result =
          evaluate_full_determinant_pair(
              input.alpha_occupied_orbitals_by_determinant,
              input.beta_occupied_orbitals_by_determinant,
              input.basis_overlap_matrix,
              input.one_electron_matrix,
              input.n_active_orbitals,
              input.packed_two_electron_integrals,
              selected_determinant.determinant_index,
              candidate_determinant.determinant_index,
              determinant_overlap_resolver_,
              determinant_hamiltonian_resolver_,
              algorithm_);
      for (const auto& selected_term : selected_determinant.structure_terms) {
        for (const auto& candidate_term : candidate_determinant.structure_terms) {
          const double combined_coefficient =
              selected_term.coefficient * candidate_term.coefficient;
          result.s_column[static_cast<std::size_t>(selected_term.structure_index)] +=
              determinant_pair_result.overlap_determinant * combined_coefficient;
          result.h_column[static_cast<std::size_t>(selected_term.structure_index)] +=
              determinant_pair_result.total_hamiltonian * combined_coefficient;
        }
      }
    }
  }

  for (const auto& left_candidate_determinant : candidate_determinants) {
    for (const auto& right_candidate_determinant : candidate_determinants) {
      const FullDeterminantPairResult determinant_pair_result =
          evaluate_full_determinant_pair(
              input.alpha_occupied_orbitals_by_determinant,
              input.beta_occupied_orbitals_by_determinant,
              input.basis_overlap_matrix,
              input.one_electron_matrix,
              input.n_active_orbitals,
              input.packed_two_electron_integrals,
              left_candidate_determinant.determinant_index,
              right_candidate_determinant.determinant_index,
              determinant_overlap_resolver_,
              determinant_hamiltonian_resolver_,
              algorithm_);
      for (const auto& left_candidate_term : left_candidate_determinant.structure_terms) {
        for (const auto& right_candidate_term : right_candidate_determinant.structure_terms) {
          const double combined_coefficient =
              left_candidate_term.coefficient * right_candidate_term.coefficient;
          result.diagonal_s +=
              determinant_pair_result.overlap_determinant * combined_coefficient;
          result.diagonal_h +=
              determinant_pair_result.total_hamiltonian * combined_coefficient;
        }
      }
    }
  }

  return result;
}

}  // namespace xmvb::vb
