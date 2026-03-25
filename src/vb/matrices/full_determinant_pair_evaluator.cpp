#include "vb/matrices/full_determinant_pair_evaluator.hpp"

#include <utility>

#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace xmvb::vb {

namespace {

SpinDeterminantPairEvaluation evaluate_spin_determinant_pair(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& ovlp_act,
    const std::vector<double>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act,
    const DeterminantOverlapResolver& determinant_overlap_resolver,
    const DeterminantHamiltonianResolver& determinant_hamiltonian_resolver) {
  SpinDeterminantPairEvaluation result;
  if (occ_L.empty()) {
    result.overlap_result.overlap_determinant = 1.0;
    return result;
  }

  const auto overlap_submatrix = build_overlap_submatrix(
      occ_L,
      occ_R,
      ovlp_act,
      n_orbitals);
  result.overlap_result = determinant_overlap_resolver.resolve(
      overlap_submatrix,
      static_cast<int>(occ_L.size()));
  const auto hamiltonian_result = determinant_hamiltonian_resolver.resolve(
      occ_L,
      occ_R,
      overlap_submatrix,
      result.overlap_result,
      h1e_act,
      n_orbitals,
      eri_act);
  result.one_electron_hamiltonian = hamiltonian_result.one_electron_hamiltonian;
  result.total_hamiltonian = hamiltonian_result.total_hamiltonian;
  return result;
}

double evaluate_opposite_spin_coulomb_coupling(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const SpinDeterminantPairEvaluation& alpha_result,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const SpinDeterminantPairEvaluation& beta_result,
    const std::vector<double>& eri_act) {
  if (alpha_occ_L.empty() || beta_occ_L.empty()) {
    return 0.0;
  }
  if (alpha_result.overlap_result.nullity >= 2 || beta_result.overlap_result.nullity >= 2) {
    return 0.0;
  }

  const Matrix alpha_cofactor_1st =
      calc_cofactor_1st(alpha_result.overlap_result);
  const Matrix beta_cofactor_1st =
      calc_cofactor_1st(beta_result.overlap_result);
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

}  // namespace

FullDeterminantPairEvaluator::FullDeterminantPairEvaluator()
    : determinant_overlap_resolver_(),
      determinant_hamiltonian_resolver_() {}

FullDeterminantPairEvaluator::FullDeterminantPairEvaluator(
    DeterminantOverlapResolver determinant_overlap_resolver,
    DeterminantHamiltonianResolver determinant_hamiltonian_resolver)
    : determinant_overlap_resolver_(std::move(determinant_overlap_resolver)),
      determinant_hamiltonian_resolver_(std::move(determinant_hamiltonian_resolver)) {}

FullDeterminantPairEvaluation FullDeterminantPairEvaluator::evaluate(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const std::vector<double>& ovlp_act,
    const std::vector<double>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act) const {
// 
// function
  FullDeterminantPairEvaluation result;
  // compute alpha-alpha
  result.alpha = evaluate_spin_determinant_pair(
      alpha_occ_L,
      alpha_occ_R,
      ovlp_act,
      h1e_act,
      n_orbitals,
      eri_act,
      determinant_overlap_resolver_,
      determinant_hamiltonian_resolver_);
    
  // compute beta-beta
  result.beta = evaluate_spin_determinant_pair(
      beta_occ_L,
      beta_occ_R,
      ovlp_act,
      h1e_act,
      n_orbitals,
      eri_act,
      determinant_overlap_resolver_,
      determinant_hamiltonian_resolver_);
    
  result.overlap_determinant =
      result.alpha.overlap_result.overlap_determinant *
      result.beta.overlap_result.overlap_determinant;
    
  result.one_electron_hamiltonian =
      result.alpha.one_electron_hamiltonian *
          result.beta.overlap_result.overlap_determinant +
      result.beta.one_electron_hamiltonian *
          result.alpha.overlap_result.overlap_determinant;

    // alpha-beta
  result.total_hamiltonian =
      result.alpha.total_hamiltonian *
          result.beta.overlap_result.overlap_determinant +
      result.beta.total_hamiltonian *
          result.alpha.overlap_result.overlap_determinant +
      evaluate_opposite_spin_coulomb_coupling(
          alpha_occ_L,
          alpha_occ_R,
          result.alpha,
          beta_occ_L,
          beta_occ_R,
          result.beta,
          eri_act);

  return result;

}

}  // namespace xmvb::vb
