#include "vb/matrices/full_determinant_pair_evaluator.hpp"

#include <utility>

#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"

namespace xmvb::vb {

namespace {

struct PreparedSpinDeterminantPair {
  SpinDeterminantPairEvaluation evaluation;
  std::vector<double> overlap_submatrix;
};

PreparedSpinDeterminantPair prepare_spin_determinant_pair(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& ovlp_act,
    int n_orbitals,
    const DeterminantOverlapResolver& determinant_overlap_resolver) {
  PreparedSpinDeterminantPair result;
  if (occ_L.empty()) {
    result.evaluation.overlap_result.overlap_determinant = 1.0;
    return result;
  }

  result.overlap_submatrix = build_overlap_submatrix(
      occ_L,
      occ_R,
      ovlp_act,
      n_orbitals);

  result.evaluation.overlap_result = determinant_overlap_resolver.resolve(
      result.overlap_submatrix,
      static_cast<int>(occ_L.size()));
  cache_first_order_cofactor(&result.evaluation.overlap_result);

  return result;
}

void evaluate_spin_determinant_hamiltonian(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act,
    const DeterminantHamiltonianResolver& determinant_hamiltonian_resolver,
    PreparedSpinDeterminantPair* prepared_result) {
  if (occ_L.empty()) {
    return;
  }

  const auto hamiltonian_result = determinant_hamiltonian_resolver.resolve(
      occ_L,
      occ_R,
      prepared_result->overlap_submatrix,
      prepared_result->evaluation.overlap_result,
      h1e_act,
      n_orbitals,
      eri_act);

  prepared_result->evaluation.one_electron_hamiltonian =
      hamiltonian_result.one_electron_hamiltonian;
  prepared_result->evaluation.total_hamiltonian =
      hamiltonian_result.total_hamiltonian;
}

void evaluate_spin_determinant_hamiltonian(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& h1e_act,
    int n_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const DeterminantHamiltonianResolver& determinant_hamiltonian_resolver,
    PreparedSpinDeterminantPair* prepared_result) {
  if (occ_L.empty()) {
    return;
  }

  const auto hamiltonian_result = determinant_hamiltonian_resolver.resolve(
      occ_L,
      occ_R,
      prepared_result->overlap_submatrix,
      prepared_result->evaluation.overlap_result,
      h1e_act,
      n_orbitals,
      active_space_two_electron_result);

  prepared_result->evaluation.one_electron_hamiltonian =
      hamiltonian_result.one_electron_hamiltonian;

  prepared_result->evaluation.total_hamiltonian =
      hamiltonian_result.total_hamiltonian;
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

  if (has_opposite_spin_first_order_projection(alpha_result.opposite_spin_pair_cache) &&
      has_opposite_spin_first_order_projection(beta_result.opposite_spin_pair_cache) &&
      alpha_result.opposite_spin_pair_cache.n_packed_active_pairs ==
          beta_result.opposite_spin_pair_cache.n_packed_active_pairs) {
    return contract_opposite_spin_first_order_projections(
        alpha_result.opposite_spin_pair_cache,
        beta_result.opposite_spin_pair_cache);
  }

  const Eigen::MatrixXd alpha_cofactor_1st =
      calc_cofactor_1st(alpha_result.overlap_result);
  const Eigen::MatrixXd beta_cofactor_1st =
      calc_cofactor_1st(beta_result.overlap_result);
  const int n_alpha_electrons = static_cast<int>(alpha_occ_L.size());
  const int n_beta_electrons = static_cast<int>(beta_occ_L.size());
  double opposite_spin_coulomb_coupling = 0.0;

  for (int alpha_left_column = 0; alpha_left_column < n_alpha_electrons; ++alpha_left_column) {
    const int alpha_orbital_left =
        alpha_occ_L[xmvb::to_size(alpha_left_column)];
    for (int alpha_right_row = 0; alpha_right_row < n_alpha_electrons; ++alpha_right_row) {
      const int alpha_orbital_right =
          alpha_occ_R[xmvb::to_size(alpha_right_row)];
      const double alpha_cofactor =
          alpha_cofactor_1st(alpha_right_row, alpha_left_column);
      for (int beta_left_column = 0; beta_left_column < n_beta_electrons; ++beta_left_column) {
        const int beta_orbital_left =
            beta_occ_L[xmvb::to_size(beta_left_column)];
        for (int beta_right_row = 0; beta_right_row < n_beta_electrons; ++beta_right_row) {
          const int beta_orbital_right =
              beta_occ_R[xmvb::to_size(beta_right_row)];
          const double beta_cofactor =
              beta_cofactor_1st(beta_right_row, beta_left_column);
          const int two_electron_index = TwoElectronIndexer::two_electron_storage_index(
              beta_orbital_right,
              beta_orbital_left,
              alpha_orbital_right,
              alpha_orbital_left);
          opposite_spin_coulomb_coupling +=
              alpha_cofactor * beta_cofactor *
              eri_act[xmvb::to_size(two_electron_index)];
        }
      }
    }
  }

  return opposite_spin_coulomb_coupling;
}

double evaluate_opposite_spin_coulomb_coupling(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const SpinDeterminantPairEvaluation& alpha_result,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const SpinDeterminantPairEvaluation& beta_result,
    const ActiveSpaceTwoElectronView& active_space_two_electron_view,
    int n_orbitals) {
  if (alpha_occ_L.empty() || beta_occ_L.empty()) {
    return 0.0;
  }
  if (alpha_result.overlap_result.nullity >= 2 || beta_result.overlap_result.nullity >= 2) {
    return 0.0;
  }

  if (has_opposite_spin_first_order_projection(alpha_result.opposite_spin_pair_cache) &&
      has_opposite_spin_first_order_projection(beta_result.opposite_spin_pair_cache) &&
      alpha_result.opposite_spin_pair_cache.n_packed_active_pairs ==
          beta_result.opposite_spin_pair_cache.n_packed_active_pairs) {
    return contract_opposite_spin_first_order_projections(
        alpha_result.opposite_spin_pair_cache,
        beta_result.opposite_spin_pair_cache);
  }

  const Eigen::MatrixXd alpha_cofactor_1st =
      calc_cofactor_1st(alpha_result.overlap_result);
  const Eigen::MatrixXd beta_cofactor_1st =
      calc_cofactor_1st(beta_result.overlap_result);
  const int n_alpha_electrons = static_cast<int>(alpha_occ_L.size());
  const int n_beta_electrons = static_cast<int>(beta_occ_L.size());
  double opposite_spin_coulomb_coupling = 0.0;

  for (int alpha_left_column = 0; alpha_left_column < n_alpha_electrons; ++alpha_left_column) {
    const int alpha_orbital_left =
        alpha_occ_L[xmvb::to_size(alpha_left_column)];
    for (int alpha_right_row = 0; alpha_right_row < n_alpha_electrons; ++alpha_right_row) {
      const int alpha_orbital_right =
          alpha_occ_R[xmvb::to_size(alpha_right_row)];
      const double alpha_cofactor =
          alpha_cofactor_1st(alpha_right_row, alpha_left_column);
      const int alpha_packed_pair_index = TwoElectronIndexer::packed_pair_index(
          alpha_orbital_right,
          alpha_orbital_left);
      for (int beta_left_column = 0; beta_left_column < n_beta_electrons; ++beta_left_column) {
        const int beta_orbital_left =
            beta_occ_L[xmvb::to_size(beta_left_column)];
        for (int beta_right_row = 0; beta_right_row < n_beta_electrons; ++beta_right_row) {
          const int beta_orbital_right =
              beta_occ_R[xmvb::to_size(beta_right_row)];
          const double beta_cofactor =
              beta_cofactor_1st(beta_right_row, beta_left_column);
          const int beta_packed_pair_index = TwoElectronIndexer::packed_pair_index(
              beta_orbital_right,
              beta_orbital_left);
          opposite_spin_coulomb_coupling +=
              alpha_cofactor * beta_cofactor *
              lookup_active_space_two_electron_kernel_value(
                  active_space_two_electron_view,
                  beta_packed_pair_index,
                  alpha_packed_pair_index,
                  n_orbitals);
        }
      }
    }
  }

  return opposite_spin_coulomb_coupling;
}

}  // namespace

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

  if (has_opposite_spin_first_order_projection(alpha_result.opposite_spin_pair_cache) &&
      has_opposite_spin_first_order_projection(beta_result.opposite_spin_pair_cache) &&
      alpha_result.opposite_spin_pair_cache.n_packed_active_pairs ==
          beta_result.opposite_spin_pair_cache.n_packed_active_pairs) {
    return contract_opposite_spin_first_order_projections(
        alpha_result.opposite_spin_pair_cache,
        beta_result.opposite_spin_pair_cache);
  }

  const Eigen::MatrixXd alpha_cofactor_1st =
      calc_cofactor_1st(alpha_result.overlap_result);
  const Eigen::MatrixXd beta_cofactor_1st =
      calc_cofactor_1st(beta_result.overlap_result);
  const int n_alpha_electrons = static_cast<int>(alpha_occ_L.size());
  const int n_beta_electrons = static_cast<int>(beta_occ_L.size());
  double opposite_spin_coulomb_coupling = 0.0;

  for (int alpha_left_column = 0; alpha_left_column < n_alpha_electrons; ++alpha_left_column) {
    const int alpha_orbital_left =
        alpha_occ_L[xmvb::to_size(alpha_left_column)];
    for (int alpha_right_row = 0; alpha_right_row < n_alpha_electrons; ++alpha_right_row) {
      const int alpha_orbital_right =
          alpha_occ_R[xmvb::to_size(alpha_right_row)];
      const double alpha_cofactor =
          alpha_cofactor_1st(alpha_right_row, alpha_left_column);
      for (int beta_left_column = 0; beta_left_column < n_beta_electrons; ++beta_left_column) {
        const int beta_orbital_left =
            beta_occ_L[xmvb::to_size(beta_left_column)];
        for (int beta_right_row = 0; beta_right_row < n_beta_electrons; ++beta_right_row) {
          const int beta_orbital_right =
              beta_occ_R[xmvb::to_size(beta_right_row)];
          const double beta_cofactor =
              beta_cofactor_1st(beta_right_row, beta_left_column);
          const int two_electron_index = TwoElectronIndexer::two_electron_storage_index(
              beta_orbital_right,
              beta_orbital_left,
              alpha_orbital_right,
              alpha_orbital_left);
          opposite_spin_coulomb_coupling +=
              alpha_cofactor * beta_cofactor *
              eri_act[xmvb::to_size(two_electron_index)];
        }
      }
    }
  }

  return opposite_spin_coulomb_coupling;
}

FullDeterminantPairEvaluator::FullDeterminantPairEvaluator()
    : determinant_overlap_resolver_(),
      determinant_hamiltonian_resolver_() {}

FullDeterminantPairEvaluator::FullDeterminantPairEvaluator(
    DeterminantOverlapResolver determinant_overlap_resolver,
    DeterminantHamiltonianResolver determinant_hamiltonian_resolver)
    : determinant_overlap_resolver_(std::move(determinant_overlap_resolver)),
      determinant_hamiltonian_resolver_(std::move(determinant_hamiltonian_resolver)) {}

SpinDeterminantPairEvaluation FullDeterminantPairEvaluator::evaluate_same_spin_pair(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& ovlp_act,
    const std::vector<double>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act) const {
  // This is the cacheable same-spin kernel: it resolves the occupied-overlap
  // submatrix once, then evaluates the same-spin Hamiltonian for that spin
  // block without reference to the opposite spin. The structure builder reuses
  // these results across many full determinant pairs that share the same alpha
  // or beta occupied strings.
  auto prepared_result = prepare_spin_determinant_pair(
      occ_L,
      occ_R,
      ovlp_act,
      n_orbitals,
      determinant_overlap_resolver_);
  if (!occ_L.empty()) {
    evaluate_spin_determinant_hamiltonian(
        occ_L,
        occ_R,
        h1e_act,
        n_orbitals,
        eri_act,
        determinant_hamiltonian_resolver_,
        &prepared_result);
  }
  return std::move(prepared_result.evaluation);
}

SpinDeterminantPairEvaluation FullDeterminantPairEvaluator::evaluate_same_spin_pair(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& ovlp_act,
    const std::vector<double>& h1e_act,
    int n_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result) const {
  auto prepared_result = prepare_spin_determinant_pair(
      occ_L,
      occ_R,
      ovlp_act,
      n_orbitals,
      determinant_overlap_resolver_);
  if (!occ_L.empty()) {
    evaluate_spin_determinant_hamiltonian(
        occ_L,
        occ_R,
        h1e_act,
        n_orbitals,
        active_space_two_electron_result,
        determinant_hamiltonian_resolver_,
        &prepared_result);
  }
  return std::move(prepared_result.evaluation);
}

FullDeterminantPairEvaluation FullDeterminantPairEvaluator::combine_spin_pair_evaluations(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const SpinDeterminantPairEvaluation& alpha_result,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const SpinDeterminantPairEvaluation& beta_result,
    const std::vector<double>& eri_act,
    bool retain_spin_pair_evaluations) const {
  // Once the reusable alpha and beta same-spin kernels are available, the full
  // determinant pair is a pure algebraic combination of:
  // 1. same-spin Hamiltonians scaled by the opposite-spin overlap determinant
  // 2. the opposite-spin Coulomb coupling reconstructed from the cached
  //    first-cofactor data contained in `alpha_result` and `beta_result`
  FullDeterminantPairEvaluation result;
  if (retain_spin_pair_evaluations) {
    result.alpha = alpha_result;
    result.beta = beta_result;
  }
  result.overlap_determinant =
      alpha_result.overlap_result.overlap_determinant *
      beta_result.overlap_result.overlap_determinant;
  result.one_electron_hamiltonian =
      alpha_result.one_electron_hamiltonian *
          beta_result.overlap_result.overlap_determinant +
      beta_result.one_electron_hamiltonian *
          alpha_result.overlap_result.overlap_determinant;
  result.total_hamiltonian =
      alpha_result.total_hamiltonian *
          beta_result.overlap_result.overlap_determinant +
      beta_result.total_hamiltonian *
          alpha_result.overlap_result.overlap_determinant +
      ::xmvb::vb::evaluate_opposite_spin_coulomb_coupling(
          alpha_occ_L,
          alpha_occ_R,
          alpha_result,
          beta_occ_L,
          beta_occ_R,
          beta_result,
          eri_act);
  return result;
}

FullDeterminantPairEvaluation FullDeterminantPairEvaluator::combine_spin_pair_evaluations(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const SpinDeterminantPairEvaluation& alpha_result,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const SpinDeterminantPairEvaluation& beta_result,
    int n_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    bool retain_spin_pair_evaluations) const {
  FullDeterminantPairEvaluation result;
  if (retain_spin_pair_evaluations) {
    result.alpha = alpha_result;
    result.beta = beta_result;
  }
  result.overlap_determinant =
      alpha_result.overlap_result.overlap_determinant *
      beta_result.overlap_result.overlap_determinant;
  result.one_electron_hamiltonian =
      alpha_result.one_electron_hamiltonian *
          beta_result.overlap_result.overlap_determinant +
      beta_result.one_electron_hamiltonian *
          alpha_result.overlap_result.overlap_determinant;
  result.total_hamiltonian =
      alpha_result.total_hamiltonian *
          beta_result.overlap_result.overlap_determinant +
      beta_result.total_hamiltonian *
          alpha_result.overlap_result.overlap_determinant +
      evaluate_opposite_spin_coulomb_coupling(
          alpha_occ_L,
          alpha_occ_R,
          alpha_result,
          beta_occ_L,
          beta_occ_R,
          beta_result,
          make_active_space_two_electron_view(active_space_two_electron_result),
          n_orbitals);
  return result;
}

FullDeterminantPairEvaluation FullDeterminantPairEvaluator::evaluate(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const std::vector<double>& ovlp_act,
    const std::vector<double>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act,
    bool retain_spin_pair_evaluations) const {
  // The standalone full-pair path still evaluates both spin blocks locally,
  // then hands the two same-spin kernels to the shared combination routine.
  auto alpha_prepared = prepare_spin_determinant_pair(
      alpha_occ_L,
      alpha_occ_R,
      ovlp_act,
      n_orbitals,
      determinant_overlap_resolver_);
  auto beta_prepared = prepare_spin_determinant_pair(
      beta_occ_L,
      beta_occ_R,
      ovlp_act,
      n_orbitals,
      determinant_overlap_resolver_);

  const int alpha_nullity = alpha_prepared.evaluation.overlap_result.nullity;
  const int beta_nullity = beta_prepared.evaluation.overlap_result.nullity;
  if (!alpha_occ_L.empty() && alpha_nullity < 3 && beta_nullity == 0) {
    evaluate_spin_determinant_hamiltonian(
        alpha_occ_L,
        alpha_occ_R,
        h1e_act,
        n_orbitals,
        eri_act,
        determinant_hamiltonian_resolver_,
        &alpha_prepared);
  }
  if (!beta_occ_L.empty() && beta_nullity < 3 && alpha_nullity == 0) {
    evaluate_spin_determinant_hamiltonian(
        beta_occ_L,
        beta_occ_R,
        h1e_act,
        n_orbitals,
        eri_act,
        determinant_hamiltonian_resolver_,
        &beta_prepared);
  }
  return combine_spin_pair_evaluations(
      alpha_occ_L,
      alpha_occ_R,
      alpha_prepared.evaluation,
      beta_occ_L,
      beta_occ_R,
      beta_prepared.evaluation,
      eri_act,
      retain_spin_pair_evaluations);
}

FullDeterminantPairEvaluation FullDeterminantPairEvaluator::evaluate(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const std::vector<double>& ovlp_act,
    const std::vector<double>& h1e_act,
    int n_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    bool retain_spin_pair_evaluations) const {
  auto alpha_prepared = prepare_spin_determinant_pair(
      alpha_occ_L,
      alpha_occ_R,
      ovlp_act,
      n_orbitals,
      determinant_overlap_resolver_);
  auto beta_prepared = prepare_spin_determinant_pair(
      beta_occ_L,
      beta_occ_R,
      ovlp_act,
      n_orbitals,
      determinant_overlap_resolver_);

  const int alpha_nullity = alpha_prepared.evaluation.overlap_result.nullity;
  const int beta_nullity = beta_prepared.evaluation.overlap_result.nullity;
  if (!alpha_occ_L.empty() && alpha_nullity < 3 && beta_nullity == 0) {
    evaluate_spin_determinant_hamiltonian(
        alpha_occ_L,
        alpha_occ_R,
        h1e_act,
        n_orbitals,
        active_space_two_electron_result,
        determinant_hamiltonian_resolver_,
        &alpha_prepared);
  }
  if (!beta_occ_L.empty() && beta_nullity < 3 && alpha_nullity == 0) {
    evaluate_spin_determinant_hamiltonian(
        beta_occ_L,
        beta_occ_R,
        h1e_act,
        n_orbitals,
        active_space_two_electron_result,
        determinant_hamiltonian_resolver_,
        &beta_prepared);
  }
  return combine_spin_pair_evaluations(
      alpha_occ_L,
      alpha_occ_R,
      alpha_prepared.evaluation,
      beta_occ_L,
      beta_occ_R,
      beta_prepared.evaluation,
      n_orbitals,
      active_space_two_electron_result,
      retain_spin_pair_evaluations);
}

}  // namespace xmvb::vb
