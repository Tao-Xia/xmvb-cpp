#pragma once

#include <Eigen/Core>

#include <vector>
#include <memory>

#include "vbscf/determinants/algebra/hamiltonian.hpp"
#include "vbscf/determinants/algebra/overlap.hpp"
#include "vbscf/determinants/contracts/types.hpp"

namespace xmvb::vb {

class CofactorDifferential;

/**
 * @brief High-order occupied-block data restricted to polynomial responses.
 *
 * Regular RI pairs deliberately leave this payload absent and use their
 * two-index inverse/factor representation. Singular/interpolated pairs, and
 * the current packed-exact response path, retain it only where an inverse-free
 * polynomial derivative is actually required.
 */
struct SameSpinPolynomialResponsePayload {
  Eigen::MatrixXd one_electron_block;
  Eigen::MatrixXd antisymmetrized_interaction;
};

struct SpinDeterminantPairEvaluation {
  // Accepted-point polynomial cofactor factorization. Shared ownership keeps
  // copies of pair evaluations cheap while all HVP consumers reuse one SVD.
  std::shared_ptr<const CofactorDifferential> cofactor_differential;
  /**
   * @brief Cached same-spin `\phi` payload for one ordered determinant pair.
   *
   * The active-space gradient reuses the same ordered alpha/beta determinant
   * pair many times. When available, `same_spin_inverse_overlap_gradient`
   * stores `\partial \phi / \partial X` for the pair's inverse overlap matrix,
   * and the scalar fields store the corresponding one- and two-electron
   * contributions to `\phi`.
   */
  bool has_same_spin_phi_cache = false;
  /** RI response is carried by the inverse-free Woodbury dangerous core. */
  bool has_woodbury_ri_response = false;
  double same_spin_one_electron_phi = 0.0;
  double same_spin_total_phi = 0.0;
  Eigen::MatrixXd same_spin_inverse_overlap_gradient;
  // Singular same-spin pairs cannot use the regular `det * phi(X^{-1})`
  // factorization, so we cache the exact occupied-block overlap gradient of
  // the same-spin Hamiltonian directly in `(right, left)` matrix form.
  Eigen::MatrixXd same_spin_overlap_hamiltonian_gradient;
  std::shared_ptr<const SameSpinPolynomialResponsePayload>
      same_spin_polynomial_response;
  DeterminantOverlapResult overlap_result;
  double one_electron_hamiltonian = 0.0;
  double total_hamiltonian = 0.0;
  // Exact packed active-pair regroupings used by the production opposite-spin
  // cache path. Direct small-space callers may leave this payload empty.
  OppositeSpinPairCache opposite_spin_pair_cache;
};

struct DeterminantPairEvaluation {
  SpinDeterminantPairEvaluation alpha;
  SpinDeterminantPairEvaluation beta;
  double overlap_determinant = 0.0;
  double one_electron_hamiltonian = 0.0;
  double total_hamiltonian = 0.0;
};

class DeterminantPairEvaluator {
public:
  DeterminantPairEvaluator();

  DeterminantPairEvaluator(
      DeterminantOverlapResolver determinant_overlap_resolver,
      DeterminantHamiltonianResolver determinant_hamiltonian_resolver);

  /**
   * @brief Evaluates one same-spin determinant pair exactly.
   *
   * This returns the reusable alpha-alpha or beta-beta kernel for one occupied
   * determinant pair, including the overlap-factorization data needed later by
   * the opposite-spin Coulomb term.
   */
  SpinDeterminantPairEvaluation evaluate_same_spin_pair(
      const std::vector<int>& occ_L,
      const std::vector<int>& occ_R,
      const std::vector<double>& ovlp_act,
      const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
      int n_orbitals,
      const std::vector<double>& eri_act,
      bool retain_derivative_payload = true) const;

  /**
   * @brief Builds a regular pair from an already contracted two-electron phi.
   *
   * Low-rank RI traversals use this entry point after updating the contracted
   * auxiliary channels. It avoids repeating the cubic RI contraction while
   * preserving the canonical pair payload.
   */
  SpinDeterminantPairEvaluation evaluate_regular_same_spin_pair(
      const std::vector<int>& occ_L,
      const std::vector<int>& occ_R,
      DeterminantOverlapResult overlap_result,
      const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
      double normalized_two_electron_phi,
      bool retain_derivative_payload = true) const;

  /**
   * @brief Evaluates one same-spin determinant pair from packed or RI active ERIs.
   */
  SpinDeterminantPairEvaluation evaluate_same_spin_pair(
      const std::vector<int>& occ_L,
      const std::vector<int>& occ_R,
      const std::vector<double>& ovlp_act,
      const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
      int n_orbitals,
      const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
      bool retain_derivative_payload = true) const;

  /**
   * @brief Evaluates a same-spin kernel from a certified overlap result.
   *
   * Pair-tile generators use this overload after a Woodbury update has
   * produced the determinant, inverse, and canonical occupied overlap block.
   * It avoids factorizing that block again while retaining the same
   * Hamiltonian and cofactor contracts as independent pair evaluation.
   */
  SpinDeterminantPairEvaluation evaluate_same_spin_pair(
      const std::vector<int>& occ_L,
      const std::vector<int>& occ_R,
      DeterminantOverlapResult overlap_result,
      const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
      int n_orbitals,
      const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
      bool retain_derivative_payload = true) const;

  /**
   * @brief Combines cached alpha/beta same-spin kernels into one full pair.
   *
   * This overload is used by the structure builder once the reusable alpha and
   * beta determinant-pair kernels have already been computed and cached.
   */
  DeterminantPairEvaluation combine_spin_pair_evaluations(
      const std::vector<int>& alpha_occ_L,
      const std::vector<int>& alpha_occ_R,
      const SpinDeterminantPairEvaluation& alpha_result,
      const std::vector<int>& beta_occ_L,
      const std::vector<int>& beta_occ_R,
      const SpinDeterminantPairEvaluation& beta_result,
      const std::vector<double>& eri_act,
      bool retain_spin_pair_evaluations = true) const;

  /**
   * @brief Combines cached spin kernels using packed or RI active ERIs.
   */
  DeterminantPairEvaluation combine_spin_pair_evaluations(
      const std::vector<int>& alpha_occ_L,
      const std::vector<int>& alpha_occ_R,
      const SpinDeterminantPairEvaluation& alpha_result,
      const std::vector<int>& beta_occ_L,
      const std::vector<int>& beta_occ_R,
      const SpinDeterminantPairEvaluation& beta_result,
      int n_orbitals,
      const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
      bool retain_spin_pair_evaluations = true) const;

  DeterminantPairEvaluation evaluate(
      const std::vector<int>& alpha_occ_L,
      const std::vector<int>& alpha_occ_R,
      const std::vector<int>& beta_occ_L,
      const std::vector<int>& beta_occ_R,
      const std::vector<double>& ovlp_act,
      const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
      int n_orbitals,
      const std::vector<double>& eri_act,
      bool retain_spin_pair_evaluations = true) const;

  /**
   * @brief Evaluates one full determinant pair from packed or RI active ERIs.
   */
  DeterminantPairEvaluation evaluate(
      const std::vector<int>& alpha_occ_L,
      const std::vector<int>& alpha_occ_R,
      const std::vector<int>& beta_occ_L,
      const std::vector<int>& beta_occ_R,
      const std::vector<double>& ovlp_act,
      const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
      int n_orbitals,
      const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
      bool retain_spin_pair_evaluations = true) const;

private:
  DeterminantOverlapResolver determinant_overlap_resolver_;
  DeterminantHamiltonianResolver determinant_hamiltonian_resolver_;
};

/**
 * @brief Evaluates the opposite-spin Coulomb coupling between alpha and beta electrons.
 */
double evaluate_opposite_spin_coulomb_coupling(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const SpinDeterminantPairEvaluation& alpha_result,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const SpinDeterminantPairEvaluation& beta_result,
    const std::vector<double>& eri_act);

/**
 * @brief Evaluates opposite-spin coupling without materializing packed RI integrals.
 */
double evaluate_opposite_spin_coulomb_coupling(
    const std::vector<int>& alpha_occ_L,
    const std::vector<int>& alpha_occ_R,
    const SpinDeterminantPairEvaluation& alpha_result,
    const std::vector<int>& beta_occ_L,
    const std::vector<int>& beta_occ_R,
    const SpinDeterminantPairEvaluation& beta_result,
    int n_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result);

}  // namespace xmvb::vb
