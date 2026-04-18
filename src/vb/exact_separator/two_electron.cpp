#include "vb/exact_separator/two_electron.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "vb/exact_separator/component_terms.hpp"
#include "vb/matrices/determinant_hamiltonian_resolver.hpp"
#include "vb/matrices/full_determinant_pair_evaluator.hpp"

namespace xmvb::vb::exact_separator {

namespace {

struct DeterminantPairTwoElectronChannels {
  double overlap = 0.0;
  double same_spin_alpha = 0.0;
  double same_spin_beta = 0.0;
  double opposite_spin = 0.0;
  double total = 0.0;
};

DeterminantPairTwoElectronChannels evaluate_determinant_pair_two_electron_channels(
    const GlobalOrientationTerm& left_term,
    const GlobalOrientationTerm& right_term,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& zero_one_electron_storage,
    int support_size,
    const std::vector<double>& packed_active_two_electron_integrals,
    const FullDeterminantPairEvaluator& determinant_pair_evaluator) {
  // Evaluate one exact determinant pair and split the pure two-electron
  // matrix element into the natural same-spin alpha, same-spin beta, and
  // opposite-spin channels. The zero one-electron matrix ensures that
  // `total_hamiltonian` returned by the determinant evaluator is already a
  // pure two-electron quantity.
  const FullDeterminantPairEvaluation determinant_pair =
      determinant_pair_evaluator.evaluate(
          left_term.alpha_occ,
          right_term.alpha_occ,
          left_term.beta_occ,
          right_term.beta_occ,
          support_overlap_storage,
          zero_one_electron_storage,
          support_size,
          packed_active_two_electron_integrals);

  DeterminantPairTwoElectronChannels channels;
  channels.overlap = determinant_pair.overlap_determinant;
  channels.same_spin_alpha =
      determinant_pair.alpha.total_hamiltonian *
      determinant_pair.beta.overlap_result.overlap_determinant;
  channels.same_spin_beta =
      determinant_pair.beta.total_hamiltonian *
      determinant_pair.alpha.overlap_result.overlap_determinant;
  channels.opposite_spin =
      determinant_pair.total_hamiltonian -
      channels.same_spin_alpha -
      channels.same_spin_beta;
  channels.total = determinant_pair.total_hamiltonian;
  return channels;
}

}  // namespace

ExactTwoElectronStarPairStats
evaluate_component_ordered_open_state_star_pair_two_electron_exact(
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    const std::vector<ComponentData>& ordered_components,
    const DeterminantOverlapResolver& overlap_resolver) {
  // This is the clean exact determinant-space baseline for the separator
  // two-electron work. The evaluator first merges component-local orientation
  // terms into canonical global determinant terms on each side, then sums the
  // determinant-pair overlap and pure two-electron matrix elements over that
  // merged basis.
  if (support_size <= 0) {
    throw std::invalid_argument("support_size must be positive");
  }
  if (ordered_components.empty()) {
    throw std::invalid_argument("ordered_components must not be empty");
  }
  if (support_overlap_storage.size() !=
      xmvb::to_size(support_size * support_size)) {
    throw std::invalid_argument(
        "support_overlap_storage must hold a square support-space matrix");
  }

  ExactTwoElectronStarPairStats stats;
  const std::vector<GlobalOrientationTerm> left_global_terms =
      build_global_orientation_terms(ordered_components, true);
  const std::vector<GlobalOrientationTerm> right_global_terms =
      build_global_orientation_terms(ordered_components, false);
  stats.left_global_term_count = static_cast<int>(left_global_terms.size());
  stats.right_global_term_count = static_cast<int>(right_global_terms.size());

  const DeterminantHamiltonianResolver hamiltonian_resolver(overlap_resolver);
  const FullDeterminantPairEvaluator determinant_pair_evaluator(
      overlap_resolver,
      hamiltonian_resolver);
  const std::vector<double> zero_one_electron_storage(
      xmvb::to_size(support_size * support_size),
      0.0);

  for (const auto& left_term : left_global_terms) {
    for (const auto& right_term : right_global_terms) {
      const double pair_weight = left_term.coefficient * right_term.coefficient;
      if (std::abs(pair_weight) <= 1.0e-15) {
        ++stats.skipped_zero_weight_pair_count;
        continue;
      }

      const DeterminantPairTwoElectronChannels channels =
          evaluate_determinant_pair_two_electron_channels(
              left_term,
              right_term,
              support_overlap_storage,
              zero_one_electron_storage,
              support_size,
              packed_active_two_electron_integrals,
              determinant_pair_evaluator);
      ++stats.evaluated_determinant_pair_count;
      stats.exact_overlap += pair_weight * channels.overlap;
      stats.exact_same_spin_alpha_two_electron += pair_weight * channels.same_spin_alpha;
      stats.exact_same_spin_beta_two_electron += pair_weight * channels.same_spin_beta;
      stats.exact_opposite_spin_two_electron += pair_weight * channels.opposite_spin;
      stats.exact_two_electron += pair_weight * channels.total;
    }
  }

  return stats;
}

}  // namespace xmvb::vb::exact_separator
