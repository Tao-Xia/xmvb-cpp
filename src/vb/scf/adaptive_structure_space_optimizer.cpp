#include "vb/scf/adaptive_structure_space_optimizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vb/matrices/determinant_hamiltonian_resolver.hpp"
#include "vb/matrices/determinant_overlap_resolver.hpp"
#include "vb/matrices/full_determinant_pair_evaluator.hpp"
#include "vb/matrices/full_structure_expander.hpp"

namespace xmvb::vb {

namespace {

struct RawStructureTopologySignature {
  std::vector<std::pair<int, int>> paired_orbitals;
  std::vector<int> open_shell_orbitals;
};

struct ProposedCandidate {
  int raw_structure_index = 0;
  int topology_distance = 0;
};

struct CandidateScore {
  int raw_structure_index = 0;
  int topology_distance = 0;
  int determinant_count = 0;
  double score = 0.0;
};

struct DeterminantKey {
  std::vector<int> alpha_orbitals;
  std::vector<int> beta_orbitals;

  bool operator==(const DeterminantKey& other) const {
    return alpha_orbitals == other.alpha_orbitals &&
           beta_orbitals == other.beta_orbitals;
  }
};

struct DeterminantKeyHasher {
  std::size_t operator()(const DeterminantKey& determinant_key) const {
    std::size_t hash_value = 0;
    for (const int orbital_index : determinant_key.alpha_orbitals) {
      hash_value = hash_value * 1315423911u +
          xmvb::to_size(orbital_index + 257);
    }
    hash_value = hash_value * 2654435761u + 17u;
    for (const int orbital_index : determinant_key.beta_orbitals) {
      hash_value = hash_value * 1315423911u +
          xmvb::to_size(orbital_index + 257);
    }
    return hash_value;
  }
};

struct ProposalDeterminantContribution {
  int candidate_offset = 0;
  double coefficient = 0.0;
};

struct ProposalDeterminantAggregate {
  std::vector<int> alpha_orbitals;
  std::vector<int> beta_orbitals;
  bool is_outside = false;
  std::vector<ProposalDeterminantContribution> contributions;
};

struct CandidateScoreBatch {
  std::vector<CandidateScore> scores;
  int unique_proposal_determinant_count = 0;
  int unique_outside_determinant_count = 0;
  int unique_shared_determinant_count = 0;
  std::uint64_t boundary_pair_evaluation_count = 0;
  double scoring_wall_time_seconds = 0.0;
};

void validate_options(
    const AdaptiveStructureSpaceOptimizerOptions& options) {
  if (options.max_outer_iterations <= 0) {
    throw std::invalid_argument("adaptive max_outer_iterations must be positive");
  }
  if (options.max_topology_distance < 0) {
    throw std::invalid_argument("adaptive max_topology_distance must be non-negative");
  }
  if (options.max_neighbors_per_structure <= 0) {
    throw std::invalid_argument("adaptive max_neighbors_per_structure must be positive");
  }
  if (options.max_candidate_pool_size <= 0) {
    throw std::invalid_argument("adaptive max_candidate_pool_size must be positive");
  }
  if (options.batch_size <= 0) {
    throw std::invalid_argument("adaptive batch_size must be positive");
  }
  if (options.max_total_structures <= 0) {
    throw std::invalid_argument("adaptive max_total_structures must be positive");
  }
  if (options.minimum_candidate_score < 0.0) {
    throw std::invalid_argument("adaptive minimum_candidate_score must be non-negative");
  }
}

std::vector<int> build_seed_raw_structure_indices(
    const RawStructureData& raw_structure_data,
    RawStructureSelectionMode seed_selection) {
  auto selected_indices = select_raw_structure_indices(
      raw_structure_data,
      seed_selection);
  std::sort(selected_indices.begin(), selected_indices.end());
  return selected_indices;
}

CppVbInput build_input_for_selected_raw_structures(
    const CppVbInput& orbital_template,
    const RawStructureData& raw_structure_data,
    const std::vector<int>& selected_raw_structure_indices,
    const FullDeterminantStructureExpander& expander) {
  if (selected_raw_structure_indices.empty()) {
    throw std::invalid_argument("selected_raw_structure_indices must not be empty");
  }

  CppVbInput input;
  input.orbital_preparation_input = orbital_template.orbital_preparation_input;
  input.ao_integral_input = orbital_template.ao_integral_input;
  input.libcint_input = orbital_template.libcint_input;
  input.structure_data = expander.expand_subset(
      raw_structure_data,
      selected_raw_structure_indices);
  return input;
}

std::vector<RawStructureTopologySignature> build_topology_signatures(
    const RawStructureData& raw_structure_data) {
  const int n_inactive_doubly_occupied_orbitals =
      (raw_structure_data.n_total_electrons - raw_structure_data.n_active_electrons) / 2;
  const int n_open_shell_electrons = raw_structure_data.spin_multiplicity - 1;
  const int n_active_beta_electrons =
      (raw_structure_data.n_active_electrons - n_open_shell_electrons) / 2;
  const int active_start = 2 * n_inactive_doubly_occupied_orbitals;
  const int active_stop = active_start + raw_structure_data.n_active_electrons;
  if (active_start < 0 || active_stop > raw_structure_data.n_total_electrons) {
    throw std::invalid_argument("active-electron window is out of range for raw structures");
  }

  std::vector<RawStructureTopologySignature> signatures(
      xmvb::to_size(raw_structure_data.n_structures));
  for (int structure_index = 0;
       structure_index < raw_structure_data.n_structures;
       ++structure_index) {
    const int* structure_orbitals =
        raw_structure_data.structure_orbitals_data(structure_index);
    auto& signature = signatures[xmvb::to_size(structure_index)];
    signature.paired_orbitals.reserve(xmvb::to_size(n_active_beta_electrons));
    for (int pair_index = 0; pair_index < n_active_beta_electrons; ++pair_index) {
      int left_orbital = structure_orbitals[active_start + 2 * pair_index];
      int right_orbital = structure_orbitals[active_start + 2 * pair_index + 1];
      if (left_orbital > right_orbital) {
        std::swap(left_orbital, right_orbital);
      }
      signature.paired_orbitals.emplace_back(left_orbital, right_orbital);
    }
    std::sort(
        signature.paired_orbitals.begin(),
        signature.paired_orbitals.end());

    signature.open_shell_orbitals.reserve(xmvb::to_size(n_open_shell_electrons));
    for (int open_shell_index = 0;
         open_shell_index < n_open_shell_electrons;
         ++open_shell_index) {
      signature.open_shell_orbitals.push_back(
          structure_orbitals[active_start + 2 * n_active_beta_electrons + open_shell_index]);
    }
    std::sort(
        signature.open_shell_orbitals.begin(),
        signature.open_shell_orbitals.end());
  }

  return signatures;
}

int topology_distance(
    const RawStructureTopologySignature& left,
    const RawStructureTopologySignature& right) {
  if (left.paired_orbitals.size() != right.paired_orbitals.size() ||
      left.open_shell_orbitals.size() != right.open_shell_orbitals.size()) {
    throw std::invalid_argument("raw structure topology sizes must match");
  }

  int distance = 0;
  for (std::size_t index = 0; index < left.paired_orbitals.size(); ++index) {
    if (left.paired_orbitals[index] != right.paired_orbitals[index]) {
      ++distance;
    }
  }
  for (std::size_t index = 0; index < left.open_shell_orbitals.size(); ++index) {
    if (left.open_shell_orbitals[index] != right.open_shell_orbitals[index]) {
      ++distance;
    }
  }
  return distance;
}

std::vector<ProposedCandidate> build_candidate_pool(
    const std::vector<int>& selected_raw_structure_indices,
    const std::vector<RawStructureTopologySignature>& topology_signatures,
    const AdaptiveStructureSpaceOptimizerOptions& options) {
  std::vector<bool> selected_mask(topology_signatures.size(), false);
  for (const int structure_index : selected_raw_structure_indices) {
    selected_mask[xmvb::to_size(structure_index)] = true;
  }

  std::vector<int> best_distances(topology_signatures.size(), -1);
  for (const int selected_index : selected_raw_structure_indices) {
    std::vector<ProposedCandidate> local_candidates;
    const auto& selected_signature =
        topology_signatures[xmvb::to_size(selected_index)];
    for (int candidate_index = 0;
         candidate_index < static_cast<int>(topology_signatures.size());
         ++candidate_index) {
      if (selected_mask[xmvb::to_size(candidate_index)]) {
        continue;
      }
      const int distance = topology_distance(
          selected_signature,
          topology_signatures[xmvb::to_size(candidate_index)]);
      if (distance > options.max_topology_distance) {
        continue;
      }
      local_candidates.push_back({candidate_index, distance});
    }

    std::sort(
        local_candidates.begin(),
        local_candidates.end(),
        [](const ProposedCandidate& left, const ProposedCandidate& right) {
          if (left.topology_distance != right.topology_distance) {
            return left.topology_distance < right.topology_distance;
          }
          return left.raw_structure_index < right.raw_structure_index;
        });
    if (static_cast<int>(local_candidates.size()) >
        options.max_neighbors_per_structure) {
      local_candidates.resize(
          xmvb::to_size(options.max_neighbors_per_structure));
    }

    for (const auto& candidate : local_candidates) {
      int& best_distance =
          best_distances[xmvb::to_size(candidate.raw_structure_index)];
      if (best_distance < 0 || candidate.topology_distance < best_distance) {
        best_distance = candidate.topology_distance;
      }
    }
  }

  std::vector<ProposedCandidate> merged_candidates;
  for (int candidate_index = 0;
       candidate_index < static_cast<int>(best_distances.size());
       ++candidate_index) {
    const int best_distance = best_distances[xmvb::to_size(candidate_index)];
    if (best_distance >= 0) {
      merged_candidates.push_back({candidate_index, best_distance});
    }
  }
  std::sort(
      merged_candidates.begin(),
      merged_candidates.end(),
      [](const ProposedCandidate& left, const ProposedCandidate& right) {
        if (left.topology_distance != right.topology_distance) {
          return left.topology_distance < right.topology_distance;
        }
        return left.raw_structure_index < right.raw_structure_index;
      });
  if (static_cast<int>(merged_candidates.size()) > options.max_candidate_pool_size) {
    merged_candidates.resize(
        xmvb::to_size(options.max_candidate_pool_size));
  }
  return merged_candidates;
}

const FullDeterminantStructureData& get_single_structure_expansion(
    int raw_structure_index,
    const RawStructureData& raw_structure_data,
    const FullDeterminantStructureExpander& expander,
    std::vector<std::optional<FullDeterminantStructureData>>* single_structure_cache) {
  auto& cached_value =
      (*single_structure_cache)[xmvb::to_size(raw_structure_index)];
  if (!cached_value.has_value()) {
    cached_value = expander.expand_subset(raw_structure_data, {raw_structure_index});
  }
  return *cached_value;
}

CandidateScore score_candidate_structure(
    const ProposedCandidate& candidate,
    int determinant_count,
    const std::vector<double>& candidate_state_residuals,
    const std::vector<double>& state_average_weights) {
  if (candidate_state_residuals.size() != state_average_weights.size()) {
    throw std::invalid_argument(
        "candidate_state_residuals and state_average_weights size mismatch");
  }

  double weighted_residual_norm_sq = 0.0;
  for (std::size_t selected_state_offset = 0;
       selected_state_offset < state_average_weights.size();
       ++selected_state_offset) {
    const double residual =
        candidate_state_residuals[selected_state_offset];
    weighted_residual_norm_sq +=
        state_average_weights[selected_state_offset] * residual * residual;
  }

  CandidateScore score;
  score.raw_structure_index = candidate.raw_structure_index;
  score.topology_distance = candidate.topology_distance;
  score.determinant_count = determinant_count;
  score.score = std::sqrt(std::max(0.0, weighted_residual_norm_sq));
  return score;
}

double summed_single_structure_coefficient(
    const std::vector<StructureExpansionTerm>& determinant_terms) {
  double coefficient_sum = 0.0;
  for (const auto& determinant_term : determinant_terms) {
    if (determinant_term.structure_index != 0) {
      throw std::invalid_argument(
          "single-structure expansion must use local structure index 0");
    }
    coefficient_sum += determinant_term.coefficient;
  }
  return coefficient_sum;
}

CandidateScoreBatch score_candidate_pool_with_aggregated_determinants(
    const std::vector<ProposedCandidate>& candidate_pool,
    const RawStructureData& raw_structure_data,
    const FullDeterminantStructureExpander& expander,
    std::vector<std::optional<FullDeterminantStructureData>>* single_structure_cache,
    const CppVbInput& current_input,
    const CppVbScfOptimizerResult& current_result,
    const CppVbScfAcceptedIterationSnapshot& current_snapshot,
    VBSCFAlgorithm algorithm,
    AdaptiveDeterminantScoreMode score_mode) {
  const auto scoring_start_time = std::chrono::steady_clock::now();
  CandidateScoreBatch score_batch;
  const int n_current_structures = current_input.structure_data.n_structures;
  if (n_current_structures <= 0) {
    throw std::invalid_argument("current input must contain at least one structure");
  }
  const auto& scf_result = current_result.scf_result;
  const int n_selected_states =
      static_cast<int>(scf_result.selected_state_indices.size());
  if (n_selected_states <= 0) {
    throw std::invalid_argument("current result must contain at least one selected state");
  }

  std::unordered_map<DeterminantKey, int, DeterminantKeyHasher> determinant_to_index;
  std::unordered_map<DeterminantKey, int, DeterminantKeyHasher> current_determinant_to_index;
  current_determinant_to_index.reserve(current_input.structure_data.alpha_det.size());
  for (std::size_t determinant_index = 0;
       determinant_index < current_input.structure_data.alpha_det.size();
       ++determinant_index) {
    current_determinant_to_index.emplace(
        DeterminantKey{
            current_input.structure_data.alpha_det[determinant_index],
            current_input.structure_data.beta_det[determinant_index],
        },
        static_cast<int>(determinant_index));
  }
  std::vector<ProposalDeterminantAggregate> proposal_determinants;
  std::vector<int> candidate_determinant_counts(
      candidate_pool.size(),
      0);
  for (std::size_t candidate_offset = 0;
       candidate_offset < candidate_pool.size();
       ++candidate_offset) {
    const auto& candidate = candidate_pool[candidate_offset];
    const auto& candidate_structure_data = get_single_structure_expansion(
        candidate.raw_structure_index,
        raw_structure_data,
        expander,
        single_structure_cache);
    if (candidate_structure_data.n_structures != 1) {
      throw std::invalid_argument(
          "candidate_structure_data must contain exactly one structure");
    }
    candidate_determinant_counts[candidate_offset] =
        static_cast<int>(candidate_structure_data.alpha_det.size());
    for (std::size_t determinant_index = 0;
         determinant_index < candidate_structure_data.alpha_det.size();
         ++determinant_index) {
      const double determinant_coefficient = summed_single_structure_coefficient(
          candidate_structure_data.determinant_to_structure_terms[determinant_index]);
      if (determinant_coefficient == 0.0) {
        continue;
      }

      DeterminantKey determinant_key{
          candidate_structure_data.alpha_det[determinant_index],
          candidate_structure_data.beta_det[determinant_index],
      };
      const auto [iterator, inserted] = determinant_to_index.emplace(
          determinant_key,
          static_cast<int>(proposal_determinants.size()));
      if (inserted) {
        ProposalDeterminantAggregate aggregate;
        aggregate.alpha_orbitals =
            candidate_structure_data.alpha_det[determinant_index];
        aggregate.beta_orbitals =
            candidate_structure_data.beta_det[determinant_index];
        aggregate.is_outside =
            current_determinant_to_index.find(determinant_key) ==
            current_determinant_to_index.end();
        proposal_determinants.push_back(std::move(aggregate));
        ++score_batch.unique_proposal_determinant_count;
        if (proposal_determinants.back().is_outside) {
          ++score_batch.unique_outside_determinant_count;
        } else {
          ++score_batch.unique_shared_determinant_count;
        }
      }
      proposal_determinants[xmvb::to_size(iterator->second)]
          .contributions.push_back({
              static_cast<int>(candidate_offset),
              determinant_coefficient,
          });
    }
  }

  std::vector<double> candidate_state_residuals(
      candidate_pool.size() * xmvb::to_size(n_selected_states),
      0.0);
  std::vector<double> state_structure_coefficients(
      xmvb::to_size(n_selected_states) *
          xmvb::to_size(n_current_structures),
      0.0);
  for (int selected_state_offset = 0;
       selected_state_offset < n_selected_states;
       ++selected_state_offset) {
    const int state_index =
        scf_result.selected_state_indices[xmvb::to_size(selected_state_offset)];
    for (int structure_index = 0;
         structure_index < n_current_structures;
         ++structure_index) {
      state_structure_coefficients[xmvb::to_size(selected_state_offset) *
                                       n_current_structures +
                                   xmvb::to_size(structure_index)] =
          scf_result.eigenvector_matrix[xmvb::to_size(state_index) *
                                            n_current_structures +
                                        xmvb::to_size(structure_index)];
    }
  }

  const FullDeterminantPairEvaluator pair_evaluator{
      DeterminantOverlapResolver(),
      DeterminantHamiltonianResolver(algorithm)};
  const int n_active_orbitals =
      current_input.orbital_preparation_input.n_active_orbitals;
  std::vector<double> row_hamiltonian(
      xmvb::to_size(n_current_structures),
      0.0);
  std::vector<double> row_overlap(
      xmvb::to_size(n_current_structures),
      0.0);
  std::vector<double> determinant_state_residuals(
      xmvb::to_size(n_selected_states),
      0.0);

  for (const auto& proposal_determinant : proposal_determinants) {
    if (score_mode == AdaptiveDeterminantScoreMode::OutsideOnly &&
        !proposal_determinant.is_outside) {
      continue;
    }
    std::fill(row_hamiltonian.begin(), row_hamiltonian.end(), 0.0);
    std::fill(row_overlap.begin(), row_overlap.end(), 0.0);
    score_batch.boundary_pair_evaluation_count +=
        static_cast<std::uint64_t>(current_input.structure_data.alpha_det.size());
    for (std::size_t current_det_index = 0;
         current_det_index < current_input.structure_data.alpha_det.size();
         ++current_det_index) {
      const auto determinant_pair_evaluation = pair_evaluator.evaluate(
          proposal_determinant.alpha_orbitals,
          current_input.structure_data.alpha_det[current_det_index],
          proposal_determinant.beta_orbitals,
          current_input.structure_data.beta_det[current_det_index],
          current_snapshot.active_orbital_overlap_matrix,
          current_snapshot.active_one_electron_integrals,
          n_active_orbitals,
          current_snapshot.packed_active_two_electron_integrals);
      const auto& current_terms =
          current_input.structure_data.determinant_to_structure_terms[current_det_index];
      for (const auto& current_term : current_terms) {
        row_hamiltonian[xmvb::to_size(current_term.structure_index)] +=
            current_term.coefficient *
            determinant_pair_evaluation.total_hamiltonian;
        row_overlap[xmvb::to_size(current_term.structure_index)] +=
            current_term.coefficient *
            determinant_pair_evaluation.overlap_determinant;
      }
    }

    for (int selected_state_offset = 0;
         selected_state_offset < n_selected_states;
         ++selected_state_offset) {
      const int state_index =
          scf_result.selected_state_indices[xmvb::to_size(selected_state_offset)];
      const double state_energy =
          scf_result.electronic_state_energies[xmvb::to_size(state_index)];
      double residual = 0.0;
      for (int structure_index = 0;
           structure_index < n_current_structures;
           ++structure_index) {
        residual +=
            (row_hamiltonian[xmvb::to_size(structure_index)] -
             state_energy * row_overlap[xmvb::to_size(structure_index)]) *
            state_structure_coefficients[xmvb::to_size(selected_state_offset) *
                                             n_current_structures +
                                         xmvb::to_size(structure_index)];
      }
      determinant_state_residuals[xmvb::to_size(selected_state_offset)] =
          residual;
    }

    for (const auto& contribution : proposal_determinant.contributions) {
      for (int selected_state_offset = 0;
           selected_state_offset < n_selected_states;
           ++selected_state_offset) {
        candidate_state_residuals[xmvb::to_size(contribution.candidate_offset) *
                                      n_selected_states +
                                  xmvb::to_size(selected_state_offset)] +=
            contribution.coefficient *
            determinant_state_residuals[xmvb::to_size(selected_state_offset)];
      }
    }
  }

  score_batch.scores.reserve(candidate_pool.size());
  for (std::size_t candidate_offset = 0;
       candidate_offset < candidate_pool.size();
       ++candidate_offset) {
    const auto residual_begin =
        candidate_state_residuals.begin() +
        static_cast<std::ptrdiff_t>(candidate_offset * xmvb::to_size(n_selected_states));
    const auto residual_end =
        residual_begin + n_selected_states;
    score_batch.scores.push_back(score_candidate_structure(
        candidate_pool[candidate_offset],
        candidate_determinant_counts[candidate_offset],
        std::vector<double>(residual_begin, residual_end),
        scf_result.state_average_weights));
  }
  score_batch.scoring_wall_time_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - scoring_start_time)
          .count();
  return score_batch;
}

std::vector<int> select_candidates_to_add(
    const std::vector<CandidateScore>& candidate_scores,
    int batch_size,
    int remaining_capacity,
    double minimum_candidate_score) {
  if (remaining_capacity <= 0) {
    return {};
  }
  const int n_to_add = std::min(batch_size, remaining_capacity);
  std::vector<int> selected_candidates;
  selected_candidates.reserve(xmvb::to_size(n_to_add));
  for (const auto& candidate_score : candidate_scores) {
    if (candidate_score.score < minimum_candidate_score) {
      break;
    }
    selected_candidates.push_back(candidate_score.raw_structure_index);
    if (static_cast<int>(selected_candidates.size()) >= n_to_add) {
      break;
    }
  }
  return selected_candidates;
}

CppVbScfOptimizerOptions build_inner_optimizer_options(
    const CppVbScfOptimizerOptions& base_options) {
  CppVbScfOptimizerOptions inner_options = base_options;
  inner_options.retain_accepted_iteration_trace = true;
  inner_options.accepted_iteration_callback = nullptr;
  return inner_options;
}

}  // namespace

AdaptiveStructureSpaceOptimizer::AdaptiveStructureSpaceOptimizer(
    CppVbScfOptimizerOptions optimizer_options,
    AdaptiveStructureSpaceOptimizerOptions options)
    : optimizer_options_(std::move(optimizer_options)),
      options_(std::move(options)) {
  validate_options(options_);
}

AdaptiveStructureSpaceOptimizerResult AdaptiveStructureSpaceOptimizer::optimize(
    const CppVbInputLoadResult& load_result) const {
  if (load_result.raw_structure_data.n_structures <= 0) {
    throw std::invalid_argument("adaptive optimizer requires non-empty raw structure data");
  }

  AdaptiveStructureSpaceOptimizerResult result;
  const auto optimization_start_time = std::chrono::steady_clock::now();
  const auto topology_signatures =
      build_topology_signatures(load_result.raw_structure_data);
  const FullDeterminantStructureExpander expander;
  auto current_selected_raw_structure_indices = build_seed_raw_structure_indices(
      load_result.raw_structure_data,
      options_.seed_selection);
  std::vector<std::optional<FullDeterminantStructureData>> single_structure_cache(
      xmvb::to_size(load_result.raw_structure_data.n_structures));

  CppVbInput current_input = build_input_for_selected_raw_structures(
      load_result.input,
      load_result.raw_structure_data,
      current_selected_raw_structure_indices,
      expander);
  const CppVbScfOptimizerOptions inner_options =
      build_inner_optimizer_options(optimizer_options_);

  for (int outer_iteration_index = 0;
       outer_iteration_index < options_.max_outer_iterations;
       ++outer_iteration_index) {
    const auto outer_iteration_start_time = std::chrono::steady_clock::now();
    CppVbScfOptimizer optimizer(inner_options);
    auto current_inner_result =
        optimizer.optimize(current_input, load_result.nuclear_repulsion_energy);
    result.inner_result = current_inner_result;
    if (!current_inner_result.converged) {
      AdaptiveStructureSpaceIterationSummary summary;
      summary.outer_iteration_index = outer_iteration_index;
      summary.selected_raw_structure_count =
          static_cast<int>(current_selected_raw_structure_indices.size());
      summary.expanded_determinant_count =
          static_cast<int>(current_inner_result.optimized_input.structure_data.alpha_det.size());
      summary.inner_iterations = current_inner_result.n_iterations;
      summary.total_energy = current_inner_result.final_total_energy;
      summary.outer_iteration_wall_time_seconds =
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - outer_iteration_start_time)
              .count();
      result.iteration_summaries.push_back(summary);
      result.termination_reason =
          std::string("inner_optimizer: ") + current_inner_result.termination_reason;
      break;
    }

    current_input = current_inner_result.optimized_input;
    if (current_inner_result.accepted_iteration_trace.empty()) {
      throw std::runtime_error(
          "adaptive optimizer requires retained accepted iteration trace");
    }
    const auto& current_snapshot =
        current_inner_result.accepted_iteration_trace.back();

    AdaptiveStructureSpaceIterationSummary summary;
    summary.outer_iteration_index = outer_iteration_index;
    summary.selected_raw_structure_count =
        static_cast<int>(current_selected_raw_structure_indices.size());
    summary.expanded_determinant_count =
        static_cast<int>(current_input.structure_data.alpha_det.size());
    summary.inner_iterations = current_inner_result.n_iterations;
    summary.total_energy = current_inner_result.final_total_energy;

    if (static_cast<int>(current_selected_raw_structure_indices.size()) >=
        load_result.raw_structure_data.n_structures) {
      summary.outer_iteration_wall_time_seconds =
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - outer_iteration_start_time)
              .count();
      result.iteration_summaries.push_back(summary);
      result.termination_reason = "all_raw_structures_selected";
      break;
    }
    if (static_cast<int>(current_selected_raw_structure_indices.size()) >=
        options_.max_total_structures) {
      summary.outer_iteration_wall_time_seconds =
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - outer_iteration_start_time)
              .count();
      result.iteration_summaries.push_back(summary);
      result.termination_reason = "max_total_structures";
      break;
    }

    const auto candidate_pool = build_candidate_pool(
        current_selected_raw_structure_indices,
        topology_signatures,
        options_);
    summary.proposal_count = static_cast<int>(candidate_pool.size());
    if (candidate_pool.empty()) {
      summary.outer_iteration_wall_time_seconds =
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - outer_iteration_start_time)
              .count();
      result.iteration_summaries.push_back(summary);
      result.termination_reason = "proposal_pool_empty";
      break;
    }

    auto candidate_score_batch = score_candidate_pool_with_aggregated_determinants(
        candidate_pool,
        load_result.raw_structure_data,
        expander,
        &single_structure_cache,
        current_input,
        current_inner_result,
        current_snapshot,
        optimizer_options_.algorithm,
        options_.determinant_score_mode);
    summary.unique_proposal_determinant_count =
        candidate_score_batch.unique_proposal_determinant_count;
    summary.unique_outside_determinant_count =
        candidate_score_batch.unique_outside_determinant_count;
    summary.unique_shared_determinant_count =
        candidate_score_batch.unique_shared_determinant_count;
    summary.boundary_pair_evaluation_count =
        candidate_score_batch.boundary_pair_evaluation_count;
    summary.scoring_wall_time_seconds =
        candidate_score_batch.scoring_wall_time_seconds;
    auto candidate_scores = std::move(candidate_score_batch.scores);
    std::sort(
        candidate_scores.begin(),
        candidate_scores.end(),
        [](const CandidateScore& left, const CandidateScore& right) {
          if (left.score != right.score) {
            return left.score > right.score;
          }
          if (left.topology_distance != right.topology_distance) {
            return left.topology_distance < right.topology_distance;
          }
          return left.raw_structure_index < right.raw_structure_index;
        });

    if (!candidate_scores.empty()) {
      summary.best_candidate_score = candidate_scores.front().score;
    }
    if (summary.best_candidate_score < options_.minimum_candidate_score) {
      summary.outer_iteration_wall_time_seconds =
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - outer_iteration_start_time)
              .count();
      result.iteration_summaries.push_back(summary);
      result.termination_reason = "candidate_score_below_threshold";
      break;
    }

    const int remaining_capacity =
        options_.max_total_structures -
        static_cast<int>(current_selected_raw_structure_indices.size());
    const auto added_candidates = select_candidates_to_add(
        candidate_scores,
        options_.batch_size,
        remaining_capacity,
        options_.minimum_candidate_score);
    summary.added_structure_count =
        static_cast<int>(added_candidates.size());
    summary.outer_iteration_wall_time_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - outer_iteration_start_time)
            .count();
    result.iteration_summaries.push_back(summary);
    if (added_candidates.empty()) {
      result.termination_reason = "no_candidate_selected";
      break;
    }

    current_selected_raw_structure_indices.insert(
        current_selected_raw_structure_indices.end(),
        added_candidates.begin(),
        added_candidates.end());
    std::sort(
        current_selected_raw_structure_indices.begin(),
        current_selected_raw_structure_indices.end());
    current_selected_raw_structure_indices.erase(
        std::unique(
            current_selected_raw_structure_indices.begin(),
            current_selected_raw_structure_indices.end()),
        current_selected_raw_structure_indices.end());
    current_input = build_input_for_selected_raw_structures(
        current_inner_result.optimized_input,
        load_result.raw_structure_data,
        current_selected_raw_structure_indices,
        expander);
  }

  if (result.termination_reason.empty()) {
    result.termination_reason = "max_outer_iterations";
  }
  result.selected_raw_structure_indices = std::move(current_selected_raw_structure_indices);
  result.converged =
      result.inner_result.converged &&
      result.termination_reason != "max_outer_iterations";
  result.total_wall_time_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - optimization_start_time)
          .count();
  return result;
}

}  // namespace xmvb::vb
