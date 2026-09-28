#include "tools/pair_conditioning_census.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include "vbscf/determinants/pairs/contractions.hpp"
#include "vbscf/determinants/pairs/woodbury_overlap.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xmvb::tools {
namespace {

struct StringMetric {
  bool rank_deficient = false;
  bool condition_limited = false;
  double smallest_relative_eigenvalue = 1.0;
  Eigen::MatrixXd inverse_square_root;
};

struct CensusAccumulator {
  long long regular_pairs = 0;
  long long self_rank_deficient_pairs = 0;
  long long self_condition_limited_pairs = 0;
  long long intrinsic_rank_deficient_pairs = 0;
  long long intrinsic_condition_limited_pairs = 0;
  long long hamiltonian_zero_by_nullity_pairs = 0;
  long long gradient_zero_by_nullity_pairs = 0;
  long long hvp_zero_by_nullity_pairs = 0;
  std::vector<long long> nullity_counts;
  std::vector<long long> dangerous_dimension_counts;
  double smallest_principal_singular_value = 1.0;
  double largest_principal_singular_value_excess = 0.0;
  std::vector<PairConditioningExample> worst_pairs;
};

void require_valid_inputs(
    const std::vector<std::vector<int>>& unique_strings,
    const std::vector<double>& active_overlap,
    int n_active_orbitals) {
  if (n_active_orbitals < 0 ||
      active_overlap.size() !=
          static_cast<std::size_t>(n_active_orbitals) * n_active_orbitals) {
    throw std::invalid_argument("active-overlap dimensions are inconsistent");
  }
  if (unique_strings.empty()) {
    return;
  }
  const std::size_t n_electrons = unique_strings.front().size();
  for (const auto& string : unique_strings) {
    if (string.size() != n_electrons) {
      throw std::invalid_argument("unique spin strings have inconsistent sizes");
    }
    for (const int orbital : string) {
      if (orbital < 0 || orbital >= n_active_orbitals) {
        throw std::invalid_argument("unique spin string contains an invalid orbital");
      }
    }
  }
}

StringMetric build_string_metric(
    const std::vector<int>& string,
    const Eigen::Ref<const Eigen::MatrixXd>& active_overlap) {
  StringMetric metric;
  const int n = static_cast<int>(string.size());
  if (n == 0) {
    metric.inverse_square_root.resize(0, 0);
    return metric;
  }

  Eigen::MatrixXd gram = xmvb::vb::build_overlap_submatrix(
      string, string, active_overlap);
  gram = 0.5 * (gram + gram.transpose()).eval();
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(gram);
  if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) {
    throw std::runtime_error("failed to diagonalize a unique-string self Gram matrix");
  }

  const Eigen::VectorXd eigenvalues = solver.eigenvalues();
  const double largest = eigenvalues(n - 1);
  const double rank_tolerance =
      std::numeric_limits<double>::epsilon() * static_cast<double>(n) *
      std::max(1.0, std::abs(largest));
  const double smallest = eigenvalues(0);
  metric.smallest_relative_eigenvalue = largest > 0.0
      ? smallest / largest
      : 0.0;
  metric.rank_deficient = largest <= 0.0 || smallest <= rank_tolerance;
  if (metric.rank_deficient) {
    return metric;
  }

  const double condition_number = largest / smallest;
  metric.condition_limited =
      !std::isfinite(condition_number) ||
      condition_number > xmvb::vb::regular_overlap_condition_limit();
  metric.inverse_square_root.noalias() =
      solver.eigenvectors() * eigenvalues.cwiseSqrt().cwiseInverse().asDiagonal() *
      solver.eigenvectors().transpose();
  return metric;
}

PairConditioningObservation diagnose_with_metrics(
    const std::vector<int>& left_string,
    const std::vector<int>& right_string,
    const Eigen::Ref<const Eigen::MatrixXd>& active_overlap,
    const StringMetric& left_metric,
    const StringMetric& right_metric) {
  PairConditioningObservation result;
  if (left_metric.rank_deficient || right_metric.rank_deficient) {
    result.classification = PairConditioningClass::SelfRankDeficient;
    return result;
  }

  const Eigen::MatrixXd overlap = xmvb::vb::build_overlap_submatrix(
      left_string, right_string, active_overlap);
  const Eigen::MatrixXd whitened =
      right_metric.inverse_square_root * overlap *
      left_metric.inverse_square_root;
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(whitened);
  if (svd.info() != Eigen::Success || !svd.singularValues().allFinite()) {
    throw std::runtime_error("failed to diagonalize a whitened pair overlap");
  }

  const Eigen::VectorXd singular_values = svd.singularValues();
  if (singular_values.size() == 0) {
    return result;
  }
  result.largest_principal_singular_value = singular_values(0);
  result.smallest_principal_singular_value =
      singular_values(singular_values.size() - 1);
  const double numerical_zero =
      std::numeric_limits<double>::epsilon() *
      static_cast<double>(singular_values.size()) *
      std::max(1.0, result.largest_principal_singular_value);
  result.numerical_nullity = static_cast<int>(
      (singular_values.array() <= numerical_zero).count());

  const double dangerous_threshold =
      result.largest_principal_singular_value /
      xmvb::vb::regular_overlap_condition_limit();
  result.dangerous_dimension = static_cast<int>(
      (singular_values.array() <= dangerous_threshold).count());
  if (result.smallest_principal_singular_value > 0.0) {
    result.principal_condition_number =
        result.largest_principal_singular_value /
        result.smallest_principal_singular_value;
  } else {
    result.principal_condition_number =
        std::numeric_limits<double>::infinity();
  }

  if (left_metric.condition_limited || right_metric.condition_limited) {
    result.classification = PairConditioningClass::SelfConditionLimited;
  } else if (result.numerical_nullity > 0) {
    result.classification = PairConditioningClass::IntrinsicRankDeficient;
  } else if (result.dangerous_dimension > 0) {
    result.classification = PairConditioningClass::IntrinsicConditionLimited;
  }
  return result;
}

void increment(std::vector<long long>* counts, int index) {
  if (index >= static_cast<int>(counts->size())) {
    counts->resize(index + 1, 0);
  }
  ++(*counts)[index];
}

bool worse_than(
    const PairConditioningExample& left,
    const PairConditioningExample& right) {
  return left.observation.smallest_principal_singular_value <
      right.observation.smallest_principal_singular_value;
}

void retain_worst(
    PairConditioningExample example,
    int retained_count,
    std::vector<PairConditioningExample>* worst) {
  if (retained_count <= 0 ||
      example.observation.classification ==
          PairConditioningClass::SelfRankDeficient) {
    return;
  }
  worst->push_back(std::move(example));
  std::sort(worst->begin(), worst->end(), worse_than);
  if (static_cast<int>(worst->size()) > retained_count) {
    worst->resize(retained_count);
  }
}

void retain_largest_exposure(
    PairExposureExample example,
    int retained_count,
    std::vector<PairExposureExample>* largest) {
  if (retained_count <= 0 || example.exposure == 0.0 ||
      example.observation.dangerous_dimension == 0) {
    return;
  }
  largest->push_back(std::move(example));
  std::sort(
      largest->begin(),
      largest->end(),
      [](const PairExposureExample& left, const PairExposureExample& right) {
        if (left.exposure != right.exposure) {
          return left.exposure > right.exposure;
        }
        if (left.left_string != right.left_string) {
          return left.left_string < right.left_string;
        }
        return left.right_string < right.right_string;
      });
  if (static_cast<int>(largest->size()) > retained_count) {
    largest->resize(retained_count);
  }
}

void accumulate_observation(
    int left,
    int right,
    const PairConditioningObservation& observation,
    int retained_worst_pairs,
    CensusAccumulator* accumulator) {
  switch (observation.classification) {
    case PairConditioningClass::Regular:
      ++accumulator->regular_pairs;
      break;
    case PairConditioningClass::SelfRankDeficient:
      ++accumulator->self_rank_deficient_pairs;
      break;
    case PairConditioningClass::SelfConditionLimited:
      ++accumulator->self_condition_limited_pairs;
      break;
    case PairConditioningClass::IntrinsicRankDeficient:
      ++accumulator->intrinsic_rank_deficient_pairs;
      break;
    case PairConditioningClass::IntrinsicConditionLimited:
      ++accumulator->intrinsic_condition_limited_pairs;
      break;
  }

  if (observation.classification != PairConditioningClass::SelfRankDeficient) {
    increment(&accumulator->nullity_counts, observation.numerical_nullity);
    increment(
        &accumulator->dangerous_dimension_counts,
        observation.dangerous_dimension);
    if (observation.numerical_nullity > 2) {
      ++accumulator->hamiltonian_zero_by_nullity_pairs;
    }
    if (observation.numerical_nullity > 3) {
      ++accumulator->gradient_zero_by_nullity_pairs;
    }
    if (observation.numerical_nullity > 4) {
      ++accumulator->hvp_zero_by_nullity_pairs;
    }
    accumulator->smallest_principal_singular_value = std::min(
        accumulator->smallest_principal_singular_value,
        observation.smallest_principal_singular_value);
    accumulator->largest_principal_singular_value_excess = std::max(
        accumulator->largest_principal_singular_value_excess,
        std::max(0.0, observation.largest_principal_singular_value - 1.0));
    retain_worst(
        PairConditioningExample{left, right, observation},
        retained_worst_pairs,
        &accumulator->worst_pairs);
  }
}

void merge_counts(
    const std::vector<long long>& source,
    std::vector<long long>* target) {
  if (target->size() < source.size()) {
    target->resize(source.size(), 0);
  }
  for (int index = 0; index < static_cast<int>(source.size()); ++index) {
    (*target)[index] += source[index];
  }
}

void merge_accumulator(
    const CensusAccumulator& source,
    int retained_worst_pairs,
    PairConditioningCensus* target) {
  target->regular_pairs += source.regular_pairs;
  target->self_rank_deficient_pairs += source.self_rank_deficient_pairs;
  target->self_condition_limited_pairs += source.self_condition_limited_pairs;
  target->intrinsic_rank_deficient_pairs += source.intrinsic_rank_deficient_pairs;
  target->intrinsic_condition_limited_pairs +=
      source.intrinsic_condition_limited_pairs;
  target->hamiltonian_zero_by_nullity_pairs +=
      source.hamiltonian_zero_by_nullity_pairs;
  target->gradient_zero_by_nullity_pairs +=
      source.gradient_zero_by_nullity_pairs;
  target->hvp_zero_by_nullity_pairs += source.hvp_zero_by_nullity_pairs;
  merge_counts(source.nullity_counts, &target->nullity_counts);
  merge_counts(
      source.dangerous_dimension_counts,
      &target->dangerous_dimension_counts);
  target->smallest_principal_singular_value = std::min(
      target->smallest_principal_singular_value,
      source.smallest_principal_singular_value);
  target->largest_principal_singular_value_excess = std::max(
      target->largest_principal_singular_value_excess,
      source.largest_principal_singular_value_excess);
  for (const auto& example : source.worst_pairs) {
    retain_worst(example, retained_worst_pairs, &target->worst_pairs);
  }
}

long long sampled_flat_index(
    long long sample,
    long long sample_count,
    long long population) {
  if (sample_count == population) {
    return sample;
  }
  const long double midpoint =
      (static_cast<long double>(sample) + 0.5L) *
      static_cast<long double>(population) /
      static_cast<long double>(sample_count);
  return std::min(
      population - 1,
      static_cast<long long>(midpoint));
}

std::string format_string(const std::vector<int>& occupied) {
  std::ostringstream stream;
  stream << '[';
  for (int index = 0; index < static_cast<int>(occupied.size()); ++index) {
    if (index != 0) {
      stream << ',';
    }
    stream << occupied[index];
  }
  stream << ']';
  return stream.str();
}

const char* class_name(PairConditioningClass classification) {
  switch (classification) {
    case PairConditioningClass::Regular:
      return "regular";
    case PairConditioningClass::SelfRankDeficient:
      return "self_rank_deficient";
    case PairConditioningClass::SelfConditionLimited:
      return "self_condition_limited";
    case PairConditioningClass::IntrinsicRankDeficient:
      return "intrinsic_rank_deficient";
    case PairConditioningClass::IntrinsicConditionLimited:
      return "intrinsic_condition_limited";
  }
  return "unknown";
}

}  // namespace

PairConditioningObservation diagnose_pair_conditioning(
    const std::vector<int>& left_string,
    const std::vector<int>& right_string,
    const std::vector<double>& active_overlap,
    int n_active_orbitals) {
  require_valid_inputs(
      std::vector<std::vector<int>>{left_string, right_string},
      active_overlap,
      n_active_orbitals);
  const Eigen::Map<const Eigen::MatrixXd> overlap(
      active_overlap.data(), n_active_orbitals, n_active_orbitals);
  const StringMetric left_metric = build_string_metric(left_string, overlap);
  const StringMetric right_metric = build_string_metric(right_string, overlap);
  return diagnose_with_metrics(
      left_string, right_string, overlap, left_metric, right_metric);
}

PairConditioningCensus run_pair_conditioning_census(
    const std::vector<std::vector<int>>& unique_strings,
    const std::vector<double>& active_overlap,
    int n_active_orbitals,
    long long max_pairs,
    int retained_worst_pairs) {
  require_valid_inputs(unique_strings, active_overlap, n_active_orbitals);
  if (max_pairs < 0 || retained_worst_pairs < 0) {
    throw std::invalid_argument("pair-conditioning census limits must be non-negative");
  }

  PairConditioningCensus census;
  census.condition_limit = xmvb::vb::regular_overlap_condition_limit();
  if (unique_strings.empty()) {
    return census;
  }
  const long long n_strings = static_cast<long long>(unique_strings.size());
  if (n_strings > std::numeric_limits<long long>::max() / n_strings) {
    throw std::overflow_error("ordered unique-string pair population overflow");
  }
  census.pair_population = n_strings * n_strings;
  census.visited_pairs = max_pairs == 0
      ? census.pair_population
      : std::min(max_pairs, census.pair_population);

  const Eigen::Map<const Eigen::MatrixXd> overlap(
      active_overlap.data(), n_active_orbitals, n_active_orbitals);
  std::vector<StringMetric> metrics;
  metrics.reserve(unique_strings.size());
  for (const auto& string : unique_strings) {
    metrics.push_back(build_string_metric(string, overlap));
    census.smallest_self_relative_eigenvalue = std::min(
        census.smallest_self_relative_eigenvalue,
        metrics.back().smallest_relative_eigenvalue);
    if (metrics.back().rank_deficient) {
      ++census.self_rank_deficient_strings;
    } else if (metrics.back().condition_limited) {
      ++census.self_condition_limited_strings;
    }
  }

  int thread_count = 1;
#ifdef _OPENMP
  thread_count = omp_get_max_threads();
#endif
  std::vector<CensusAccumulator> accumulators(thread_count);

#pragma omp parallel for schedule(static)
  for (long long sample = 0; sample < census.visited_pairs; ++sample) {
    const long long flat = sampled_flat_index(
        sample, census.visited_pairs, census.pair_population);
    const int left = static_cast<int>(flat / n_strings);
    const int right = static_cast<int>(flat % n_strings);
    const PairConditioningObservation observation = diagnose_with_metrics(
        unique_strings[left],
        unique_strings[right],
        overlap,
        metrics[left],
        metrics[right]);
    int thread = 0;
#ifdef _OPENMP
    thread = omp_get_thread_num();
#endif
    accumulate_observation(
        left,
        right,
        observation,
        retained_worst_pairs,
        &accumulators[thread]);
  }

  for (const auto& accumulator : accumulators) {
    merge_accumulator(accumulator, retained_worst_pairs, &census);
  }
  return census;
}

PairExposureCensus run_pair_exposure_census(
    const std::vector<std::vector<int>>& unique_strings,
    const std::vector<double>& active_overlap,
    int n_active_orbitals,
    const std::vector<Eigen::MatrixXd>& selected_state_coefficients,
    const std::vector<double>& normalized_state_weights,
    bool alpha_pairs,
    int retained_largest_pairs) {
  require_valid_inputs(unique_strings, active_overlap, n_active_orbitals);
  if (selected_state_coefficients.empty() ||
      selected_state_coefficients.size() != normalized_state_weights.size()) {
    throw std::invalid_argument(
        "selected-state coefficients and weights must have equal nonzero sizes");
  }
  if (retained_largest_pairs < 0) {
    throw std::invalid_argument("retained exposure-pair count must be non-negative");
  }

  const int n_strings = static_cast<int>(unique_strings.size());
  std::vector<Eigen::VectorXd> marginal_norms;
  marginal_norms.reserve(selected_state_coefficients.size());
  double weight_sum = 0.0;
  for (std::size_t state = 0;
       state < selected_state_coefficients.size();
       ++state) {
    const auto& coefficients = selected_state_coefficients[state];
    const double weight = normalized_state_weights[state];
    if (weight < 0.0 || !std::isfinite(weight)) {
      throw std::invalid_argument(
          "selected-state weights must be finite and non-negative");
    }
    const int target_size = alpha_pairs
        ? static_cast<int>(coefficients.rows())
        : static_cast<int>(coefficients.cols());
    if (target_size != n_strings) {
      throw std::invalid_argument(
          "selected-state coefficient dimensions do not match spin strings");
    }
    Eigen::VectorXd marginal_norm(target_size);
    if (alpha_pairs) {
      marginal_norm = coefficients.cwiseAbs().rowwise().sum();
    } else {
      marginal_norm = coefficients.cwiseAbs().colwise().sum().transpose();
    }
    marginal_norms.push_back(std::move(marginal_norm));
    weight_sum += weight;
  }
  if (!(weight_sum > 0.0) ||
      std::abs(weight_sum - 1.0) >
          64.0 * std::numeric_limits<double>::epsilon()) {
    throw std::invalid_argument(
        "selected-state weights must sum to one within roundoff");
  }

  PairExposureCensus census;
  census.pair_population =
      static_cast<long long>(n_strings) * n_strings;
  if (n_strings == 0) {
    return census;
  }

  const Eigen::Map<const Eigen::MatrixXd> overlap(
      active_overlap.data(), n_active_orbitals, n_active_orbitals);
  std::vector<StringMetric> metrics;
  metrics.reserve(unique_strings.size());
  for (const auto& string : unique_strings) {
    metrics.push_back(build_string_metric(string, overlap));
  }

  for (int left = 0; left < n_strings; ++left) {
    for (int right = 0; right < n_strings; ++right) {
      double exposure = 0.0;
      for (std::size_t state = 0; state < marginal_norms.size(); ++state) {
        exposure += normalized_state_weights[state] *
            marginal_norms[state](left) * marginal_norms[state](right);
      }
      const PairConditioningObservation observation = diagnose_with_metrics(
          unique_strings[left],
          unique_strings[right],
          overlap,
          metrics[left],
          metrics[right]);
      if (exposure == 0.0) {
        ++census.zero_exposure_pairs;
      }
      const double squared_exposure = exposure * exposure;
      census.total_exposure += exposure;
      census.total_squared_exposure += squared_exposure;
      const int dangerous_dimension = observation.dangerous_dimension;
      if (dangerous_dimension >= static_cast<int>(
              census.exposure_by_dangerous_dimension.size())) {
        census.exposure_by_dangerous_dimension.resize(
            dangerous_dimension + 1, 0.0);
        census.squared_exposure_by_dangerous_dimension.resize(
            dangerous_dimension + 1, 0.0);
      }
      census.exposure_by_dangerous_dimension[dangerous_dimension] += exposure;
      census.squared_exposure_by_dangerous_dimension[dangerous_dimension] +=
          squared_exposure;
      if (dangerous_dimension == 0) {
        census.maximum_regular_exposure = std::max(
            census.maximum_regular_exposure, exposure);
      } else {
        census.dangerous_exposure += exposure;
        census.dangerous_squared_exposure += squared_exposure;
        census.maximum_dangerous_exposure = std::max(
            census.maximum_dangerous_exposure, exposure);
        retain_largest_exposure(
            PairExposureExample{left, right, exposure, observation},
            retained_largest_pairs,
            &census.largest_dangerous_pairs);
      }
    }
  }
  return census;
}

void print_pair_conditioning_census(
    const char* label,
    const std::vector<std::vector<int>>& unique_strings,
    const PairConditioningCensus& census,
    std::ostream& output) {
  const auto fraction = [&](long long count) {
    return census.visited_pairs > 0
        ? static_cast<double>(count) /
              static_cast<double>(census.visited_pairs)
        : 0.0;
  };
  output << std::setprecision(15);
  output << label << "_string_count = " << unique_strings.size() << '\n';
  output << label << "_pair_population = " << census.pair_population << '\n';
  output << label << "_visited_pairs = " << census.visited_pairs << '\n';
  output << label << "_sample_fraction = "
         << (census.pair_population > 0
                 ? static_cast<double>(census.visited_pairs) /
                       static_cast<double>(census.pair_population)
                 : 0.0)
         << '\n';
  output << label << "_condition_limit = " << census.condition_limit << '\n';
  output << label << "_self_rank_deficient_strings = "
         << census.self_rank_deficient_strings << '\n';
  output << label << "_self_condition_limited_strings = "
         << census.self_condition_limited_strings << '\n';
  output << label << "_smallest_self_relative_eigenvalue = "
         << census.smallest_self_relative_eigenvalue << '\n';

  const std::pair<const char*, long long> categories[] = {
      {"regular", census.regular_pairs},
      {"self_rank_deficient", census.self_rank_deficient_pairs},
      {"self_condition_limited", census.self_condition_limited_pairs},
      {"intrinsic_rank_deficient", census.intrinsic_rank_deficient_pairs},
      {"intrinsic_condition_limited", census.intrinsic_condition_limited_pairs},
  };
  for (const auto& [name, count] : categories) {
    output << label << '_' << name << "_pairs = " << count << '\n';
    output << label << '_' << name << "_fraction = " << fraction(count) << '\n';
  }
  output << label << "_hamiltonian_zero_by_nullity_pairs = "
         << census.hamiltonian_zero_by_nullity_pairs << '\n';
  output << label << "_gradient_zero_by_nullity_pairs = "
         << census.gradient_zero_by_nullity_pairs << '\n';
  output << label << "_hvp_zero_by_nullity_pairs = "
         << census.hvp_zero_by_nullity_pairs << '\n';
  output << label << "_smallest_principal_singular_value = "
         << census.smallest_principal_singular_value << '\n';
  output << label << "_largest_principal_singular_value_excess = "
         << census.largest_principal_singular_value_excess << '\n';

  for (int nullity = 0;
       nullity < static_cast<int>(census.nullity_counts.size());
       ++nullity) {
    output << label << "_numerical_nullity_" << nullity << "_pairs = "
           << census.nullity_counts[nullity] << '\n';
  }
  for (int dimension = 0;
       dimension < static_cast<int>(census.dangerous_dimension_counts.size());
       ++dimension) {
    output << label << "_dangerous_dimension_" << dimension << "_pairs = "
           << census.dangerous_dimension_counts[dimension] << '\n';
  }
  for (int index = 0;
       index < static_cast<int>(census.worst_pairs.size());
       ++index) {
    const auto& example = census.worst_pairs[index];
    output << label << "_worst_pair_" << index << " = left:"
           << example.left_string << format_string(
                  unique_strings[example.left_string])
           << " right:" << example.right_string << format_string(
                  unique_strings[example.right_string])
           << " class:" << class_name(example.observation.classification)
           << " nullity:" << example.observation.numerical_nullity
           << " dangerous_dimension:"
           << example.observation.dangerous_dimension
           << " sigma_min:"
           << example.observation.smallest_principal_singular_value
           << " condition:"
           << example.observation.principal_condition_number << '\n';
  }
}

void print_pair_exposure_census(
    const char* label,
    const std::vector<std::vector<int>>& unique_strings,
    const PairExposureCensus& census,
    std::ostream& output) {
  const auto fraction = [](double numerator, double denominator) {
    return denominator > 0.0 ? numerator / denominator : 0.0;
  };
  output << std::setprecision(15);
  output << label << "_exposure_pair_population = "
         << census.pair_population << '\n';
  output << label << "_zero_exposure_pairs = "
         << census.zero_exposure_pairs << '\n';
  output << label << "_zero_exposure_fraction = "
         << (census.pair_population > 0
                 ? static_cast<double>(census.zero_exposure_pairs) /
                       static_cast<double>(census.pair_population)
                 : 0.0)
         << '\n';
  output << label << "_total_exposure = " << census.total_exposure << '\n';
  output << label << "_dangerous_exposure = "
         << census.dangerous_exposure << '\n';
  output << label << "_dangerous_exposure_fraction = "
         << fraction(census.dangerous_exposure, census.total_exposure) << '\n';
  output << label << "_total_squared_exposure = "
         << census.total_squared_exposure << '\n';
  output << label << "_dangerous_squared_exposure = "
         << census.dangerous_squared_exposure << '\n';
  output << label << "_dangerous_squared_exposure_fraction = "
         << fraction(
                census.dangerous_squared_exposure,
                census.total_squared_exposure)
         << '\n';
  output << label << "_maximum_regular_exposure = "
         << census.maximum_regular_exposure << '\n';
  output << label << "_maximum_dangerous_exposure = "
         << census.maximum_dangerous_exposure << '\n';
  for (int dimension = 0;
       dimension < static_cast<int>(
           census.exposure_by_dangerous_dimension.size());
       ++dimension) {
    output << label << "_dangerous_dimension_" << dimension
           << "_exposure_fraction = "
           << fraction(
                  census.exposure_by_dangerous_dimension[dimension],
                  census.total_exposure)
           << '\n';
    output << label << "_dangerous_dimension_" << dimension
           << "_squared_exposure_fraction = "
           << fraction(
                  census.squared_exposure_by_dangerous_dimension[dimension],
                  census.total_squared_exposure)
           << '\n';
  }
  for (int index = 0;
       index < static_cast<int>(census.largest_dangerous_pairs.size());
       ++index) {
    const auto& example = census.largest_dangerous_pairs[index];
    output << label << "_largest_dangerous_exposure_pair_" << index
           << " = left:" << example.left_string
           << format_string(unique_strings[example.left_string])
           << " right:" << example.right_string
           << format_string(unique_strings[example.right_string])
           << " exposure:" << example.exposure
           << " dangerous_dimension:"
           << example.observation.dangerous_dimension
           << " sigma_min:"
           << example.observation.smallest_principal_singular_value
           << '\n';
  }
}

}  // namespace xmvb::tools
