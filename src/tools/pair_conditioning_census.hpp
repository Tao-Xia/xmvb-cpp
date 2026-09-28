#pragma once

#include <iosfwd>
#include <vector>

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

void print_pair_conditioning_census(
    const char* label,
    const std::vector<std::vector<int>>& unique_strings,
    const PairConditioningCensus& census,
    std::ostream& output);

}  // namespace xmvb::tools
