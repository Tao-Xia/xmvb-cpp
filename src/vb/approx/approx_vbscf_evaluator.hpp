#pragma once

#include <Eigen/Core>

#include <vector>

#include "vb/approx/approx_vbscf_cluster_quotient.hpp"
#include "vb/approx/approx_vbscf_metric.hpp"
#include "vb/approx/approx_vbscf_resonance_functional.hpp"
#include "vb/matrices/structure_types.hpp"
#include "vb/orbital/active_space_two_electron_result.hpp"

namespace xmvb::vb {

struct ApproxVbScfModelInput {
  int n_active_orbitals = 0;
  double one_electron_reference_energy = 0.0;
  double nuclear_repulsion_energy = 0.0;
  const Eigen::MatrixXd* active_one_electron_integrals = nullptr;
  const std::vector<double>* active_orbital_overlap_matrix = nullptr;
  const ActiveSpaceTwoElectronResult* active_two_electron_result = nullptr;
};

struct ApproxVbScfStateMappingInput {
  int n_active_orbitals = 0;
  const RawStructureData* raw_structure_data = nullptr;
  const std::vector<double>* structure_overlap_matrix = nullptr;
  const std::vector<double>* eigenvector_matrix = nullptr;
  int state_index = 0;
};

struct ApproxVbScfPairTerm {
  int first_orbital = 0;
  int second_orbital = 0;
  double local_energy = 0.0;
  double cluster_quotient_energy = 0.0;
  double overlap_response = 0.0;
  double one_pair_metric_log = 0.0;
};

/**
 * @brief Pair-space approximate VBSCF model built from active-space tensors.
 *
 * The model is independent of the VB structure list. Exact VBSCF data can be
 * mapped into pair occupations for small-system labels, but production
 * approximate evaluations only need this model and a pair-occupation vector.
 */
struct ApproxVbScfModel {
  int n_active_orbitals = 0;
  int n_active_pairs = 0;
  double one_electron_reference_energy = 0.0;
  double nuclear_repulsion_energy = 0.0;
  std::vector<ApproxVbScfPairTerm> pair_terms;
  Eigen::MatrixXd pair_interactions;
  ApproxVbScfMetricModel metric_model;
  ApproxVbScfClusterQuotientModel cluster_quotient_model;
  ApproxVbScfResonanceFunctionalModel resonance_functional_model;
};

/**
 * @brief Pair observables obtained by projecting one exact VBSCF state.
 *
 * This object is a validation bridge. It depends on exact structure weights
 * and should not be required by the production approximate evaluator.  The
 * pair-pair occupation stores Gamma(p,q) = sum_I W_I b_I(p) b_I(q).
 */
struct ApproxVbScfState {
  int n_structures = 0;
  double mapped_weight_sum = 0.0;
  std::vector<double> pair_occupations;
  Eigen::MatrixXd pair_pair_occupations;
  std::vector<double> structure_weights;
};

struct ApproxVbScfResult {
  int n_active_pairs = 0;
  double exact_active_energy = 0.0;
  double exact_total_energy = 0.0;
  double approximate_active_energy = 0.0;
  double approximate_total_energy = 0.0;
  double diagonal_pair_energy = 0.0;
  double offdiagonal_pair_energy = 0.0;
  double pair_interaction_energy = 0.0;
  double pair_energy_kernel = 0.0;
  double cluster_quotient_pair_energy = 0.0;
  double cluster_quotient_pair_pair_energy = 0.0;
  double cluster_quotient_active_energy = 0.0;
  double cluster_quotient_total_energy = 0.0;
  double resonance_functional_pair_energy = 0.0;
  double resonance_functional_pair_pair_energy = 0.0;
  double resonance_functional_active_energy = 0.0;
  double resonance_functional_total_energy = 0.0;
  double overlap_response_energy = 0.0;
  double one_pair_metric_log = 0.0;
  double pair_pair_metric_log = 0.0;
  double total_metric_log = 0.0;
  double metric_denominator = 1.0;
  double diagnostic_metric_quotient_active_energy = 0.0;
  double diagnostic_metric_quotient_total_energy = 0.0;
  int local_term_count = 0;
  int pair_interaction_term_count = 0;
  int resonance_functional_one_pair_term_count = 0;
  int resonance_functional_dense_pair_pair_term_count = 0;
  int resonance_functional_pair_pair_term_count = 0;
  std::vector<double> pair_occupations;
  std::vector<double> pair_local_energy_values;
};

struct ApproxVbScfPairScfOptions {
  int max_iterations = 200;
  double step_size = 0.05;
  double gradient_tolerance = 1.0e-8;
  double minimum_occupation = 1.0e-14;
};

struct ApproxVbScfPairScfResult {
  int iterations = 0;
  bool converged = false;
  double occupied_pair_count = 0.0;
  double initial_total_energy = 0.0;
  double final_projected_gradient_norm = 0.0;
  std::vector<double> pair_occupations;
  ApproxVbScfResult result;
};

struct ApproxVbScfStructureProjectionOptions {
  int max_iterations = 5000;
  double step_size = 0.1;
  double gradient_tolerance = 1.0e-10;
  double minimum_weight = 1.0e-16;
};

struct ApproxVbScfStructureProjectionResult {
  int n_structures = 0;
  int n_active_pairs = 0;
  int iterations = 0;
  bool converged = false;
  double residual_norm = 0.0;
  double projected_gradient_norm = 0.0;
  std::vector<double> structure_weights;
  std::vector<double> reconstructed_pair_occupations;
};

ApproxVbScfModel build_approx_vbscf_model(
    const ApproxVbScfModelInput& input);

ApproxVbScfState map_selected_state_to_pair_state(
    const ApproxVbScfStateMappingInput& input);

ApproxVbScfResult evaluate_approx_vbscf(
    const ApproxVbScfModel& model,
    const std::vector<double>& pair_occupations);

/**
 * @brief Evaluates the aVB-2 energy with explicit pair-pair occupations.
 */
ApproxVbScfResult evaluate_approx_vbscf(
    const ApproxVbScfModel& model,
    const std::vector<double>& pair_occupations,
    const Eigen::MatrixXd& pair_pair_occupations);

/**
 * @brief Builds the uniform starting point on the pair-occupation simplex.
 */
std::vector<double> build_uniform_pair_occupations(
    int n_active_pairs,
    double occupied_pair_count);

/**
 * @brief Builds the mean-field pair-pair occupation Gamma(p,q)=n_p n_q.
 */
Eigen::MatrixXd build_mean_field_pair_pair_occupations(
    const std::vector<double>& pair_occupations);

/**
 * @brief Computes dE/dxi for the mean-field aVB-2 pair energy.
 */
std::vector<double> compute_approx_vbscf_pair_gradient(
    const ApproxVbScfModel& model,
    const std::vector<double>& pair_occupations);

/**
 * @brief Optimizes continuous pair occupations with fixed total pair count.
 *
 * This is the xi-only pair-SCF prototype. It keeps the mean-field
 * factorization Gamma(p,q)=n_p n_q while using multiplicative mirror descent,
 * so the occupations stay non-negative and keep
 * `sum_p n_p = occupied_pair_count`.
 */
ApproxVbScfPairScfResult optimize_approx_vbscf_pair_occupations(
    const ApproxVbScfModel& model,
    double occupied_pair_count,
    const ApproxVbScfPairScfOptions& options = {});

/**
 * @brief Projects pair occupations back to non-negative candidate structure weights.
 *
 * The projection solves `min_W ||A W - xi||^2` on the simplex
 * `sum_I W_I = 1, W_I >= 0`, where `A_{pI}` is the pair-incidence pattern of
 * the provided structure list. The result is a pseudo-structure importance
 * measure for diagnostics, not an exact VBSCF structure weight.
 */
ApproxVbScfStructureProjectionResult
project_pair_occupations_to_structure_weights(
    const RawStructureData& raw_structure_data,
    int n_active_orbitals,
    const std::vector<double>& target_pair_occupations,
    const ApproxVbScfStructureProjectionOptions& options = {});

}  // namespace xmvb::vb
