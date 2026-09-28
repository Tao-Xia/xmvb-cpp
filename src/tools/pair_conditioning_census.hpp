#pragma once

#include <iosfwd>
#include <vector>

#include <Eigen/Core>

namespace xmvb::tools {

/** Primary numerical origin of an occupied-overlap conditioning failure. */
enum class PairConditioningClass {
  Regular,
  SelfRankDeficient,
  SelfConditionLimited,
  IntrinsicRankDeficient,
  IntrinsicConditionLimited,
};

/** Conditioning observation for one ordered unique-spin-string pair. */
struct PairConditioningObservation {
  PairConditioningClass classification = PairConditioningClass::Regular;
  int numerical_nullity = 0;
  int dangerous_dimension = 0;
  double smallest_principal_singular_value = 1.0;
  double largest_principal_singular_value = 1.0;
  double principal_condition_number = 1.0;
};

/** Most ill-conditioned pair retained for human inspection. */
struct PairConditioningExample {
  int left_string = -1;
  int right_string = -1;
  PairConditioningObservation observation;
};

/**
 * @brief Streaming census that separates orbital self-conditioning from
 * inter-string principal-angle conditioning.
 */
struct PairConditioningCensus {
  long long pair_population = 0;
  long long visited_pairs = 0;
  long long regular_pairs = 0;
  long long self_rank_deficient_pairs = 0;
  long long self_condition_limited_pairs = 0;
  long long intrinsic_rank_deficient_pairs = 0;
  long long intrinsic_condition_limited_pairs = 0;
  long long hamiltonian_zero_by_nullity_pairs = 0;
  long long gradient_zero_by_nullity_pairs = 0;
  long long hvp_zero_by_nullity_pairs = 0;
  long long self_rank_deficient_strings = 0;
  long long self_condition_limited_strings = 0;
  std::vector<long long> nullity_counts;
  std::vector<long long> dangerous_dimension_counts;
  double condition_limit = 0.0;
  double smallest_self_relative_eigenvalue = 1.0;
  double smallest_principal_singular_value = 1.0;
  double largest_principal_singular_value_excess = 0.0;
  std::vector<PairConditioningExample> worst_pairs;
};

/** Selected-state coefficient exposure carried by one spin-string pair. */
struct PairExposureExample {
  int left_string = -1;
  int right_string = -1;
  double exposure = 0.0;
  PairConditioningObservation observation;
};

/**
 * @brief Parameter-free census of wavefunction support on conditioned pairs.
 *
 * Exposure is accumulated without a screening threshold. `zero_exposure_pairs`
 * therefore counts only exact zeros in the selected-state coefficient support.
 */
struct PairExposureCensus {
  long long pair_population = 0;
  long long zero_exposure_pairs = 0;
  double total_exposure = 0.0;
  double dangerous_exposure = 0.0;
  double total_squared_exposure = 0.0;
  double dangerous_squared_exposure = 0.0;
  double maximum_regular_exposure = 0.0;
  double maximum_dangerous_exposure = 0.0;
  std::vector<double> exposure_by_dangerous_dimension;
  std::vector<double> squared_exposure_by_dangerous_dimension;
  std::vector<PairExposureExample> largest_dangerous_pairs;
};

/**
 * @brief Diagnoses one pair after removing each determinant's self metric.
 *
 * `active_overlap` is the active-orbital overlap matrix in column-major
 * storage.  The classification threshold follows floating-point precision and
 * the fourth-order HVP accuracy contract; it contains no molecular parameter.
 */
PairConditioningObservation diagnose_pair_conditioning(
    const std::vector<int>& left_string,
    const std::vector<int>& right_string,
    const std::vector<double>& active_overlap,
    int n_active_orbitals);

/**
 * @brief Diagnoses all pairs or a deterministic uniform sample.
 *
 * `max_pairs == 0` requests the full ordered pair population.  Positive
 * values bound diagnostic work without changing any production calculation.
 */
PairConditioningCensus run_pair_conditioning_census(
    const std::vector<std::vector<int>>& unique_strings,
    const std::vector<double>& active_overlap,
    int n_active_orbitals,
    long long max_pairs,
    int retained_worst_pairs = 8);

/**
 * @brief Measures selected-state coefficient exposure on every ordered pair.
 *
 * For alpha pairs, the exposure is
 * `sum_s w_s ||C_s(i,:)||_1 ||C_s(j,:)||_1`; beta pairs use column norms.
 * This is a dimensionless upper-bound factor for all partner-spin
 * contractions and contains no empirical screening threshold.
 */
PairExposureCensus run_pair_exposure_census(
    const std::vector<std::vector<int>>& unique_strings,
    const std::vector<double>& active_overlap,
    int n_active_orbitals,
    const std::vector<Eigen::MatrixXd>& selected_state_coefficients,
    const std::vector<double>& normalized_state_weights,
    bool alpha_pairs,
    int retained_largest_pairs = 8);

void print_pair_conditioning_census(
    const char* label,
    const std::vector<std::vector<int>>& unique_strings,
    const PairConditioningCensus& census,
    std::ostream& output);

void print_pair_exposure_census(
    const char* label,
    const std::vector<std::vector<int>>& unique_strings,
    const PairExposureCensus& census,
    std::ostream& output);

}  // namespace xmvb::tools
