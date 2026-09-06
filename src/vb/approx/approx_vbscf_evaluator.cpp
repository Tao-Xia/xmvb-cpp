#include "vb/approx/approx_vbscf_evaluator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"

namespace xmvb::vb {

namespace {

double active_overlap_value(
    const std::vector<double>& active_overlap,
    int n_active_orbitals,
    int row,
    int column) {
  return active_overlap[column * n_active_orbitals + row];
}

double pair_one_electron_energy(
    const Eigen::MatrixXd& h1e_act,
    int first_orbital,
    int second_orbital) {
  if (first_orbital == second_orbital) {
    return 2.0 * h1e_act(first_orbital, first_orbital);
  }
  return h1e_act(first_orbital, first_orbital) +
      h1e_act(second_orbital, second_orbital);
}

double pair_self_two_electron_energy(
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals,
    int first_orbital,
    int second_orbital) {
  const int direct_pair =
      TwoElectronIndexer::packed_pair_index(first_orbital, second_orbital);
  const double direct =
      lookup_active_space_two_electron_kernel_value(
          two_electron_view,
          direct_pair,
          direct_pair,
          n_active_orbitals);
  if (first_orbital == second_orbital) {
    return direct;
  }
  const int first_first_pair =
      TwoElectronIndexer::packed_pair_index(first_orbital, first_orbital);
  const int second_second_pair =
      TwoElectronIndexer::packed_pair_index(second_orbital, second_orbital);
  const double coulomb =
      lookup_active_space_two_electron_kernel_value(
          two_electron_view,
          first_first_pair,
          second_second_pair,
          n_active_orbitals);
  return coulomb - direct;
}

double pair_interaction(
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals,
    int first_pair,
    int second_pair) {
  return lookup_active_space_two_electron_kernel_value(
      two_electron_view,
      first_pair,
      second_pair,
      n_active_orbitals);
}

double overlap_response(
    const std::vector<double>& active_overlap,
    int n_active_orbitals,
    int first_orbital,
    int second_orbital) {
  if (first_orbital == second_orbital) {
    return 0.0;
  }
  const double overlap =
      active_overlap_value(
          active_overlap,
          n_active_orbitals,
          first_orbital,
          second_orbital);
  return overlap * overlap;
}

std::vector<double> map_selected_state_structure_weights(
    const RawStructureData& raw_structure_data,
    const std::vector<double>& structure_overlap_matrix,
    const std::vector<double>& eigenvector_matrix,
    int state_index,
    double* weight_sum) {
  const int n_structures = raw_structure_data.n_structures;
  const double* coefficients =
      eigenvector_matrix.data() + state_index * n_structures;
  std::vector<double> weights(n_structures, 0.0);
  double sum = 0.0;
  for (int structure_index = 0;
       structure_index < n_structures;
       ++structure_index) {
    double overlap_projected_coefficient = 0.0;
    for (int column = 0; column < n_structures; ++column) {
      overlap_projected_coefficient +=
          structure_overlap_matrix[column * n_structures + structure_index] *
          coefficients[column];
    }
    weights[structure_index] =
        coefficients[structure_index] * overlap_projected_coefficient;
    sum += weights[structure_index];
  }
  if (weight_sum != nullptr) {
    *weight_sum = sum;
  }
  return weights;
}

void accumulate_pair_observables_from_structure_indices(
    const std::vector<std::vector<int>>& structure_pair_indices,
    const std::vector<double>& structure_weights,
    std::vector<double>* pair_occupations,
    Eigen::MatrixXd* pair_pair_occupations) {
  for (int structure_index = 0;
       structure_index < static_cast<int>(structure_pair_indices.size());
       ++structure_index) {
    const double weight = structure_weights[structure_index];
    const std::vector<int>& pair_indices =
        structure_pair_indices[structure_index];
    for (int pair_index : pair_indices) {
      (*pair_occupations)[pair_index] += weight;
    }
    for (int column_pair : pair_indices) {
      for (int row_pair : pair_indices) {
        (*pair_pair_occupations)(row_pair, column_pair) += weight;
      }
    }
  }
}

std::vector<std::vector<int>> build_structure_pair_indices(
    const RawStructureData& raw_structure_data,
    int n_active_orbitals) {
  const int active_start =
      raw_structure_data.n_total_electrons -
      raw_structure_data.n_active_electrons;
  const int n_inactive_doubly_occupied_orbitals = active_start / 2;
  const int n_open_shell_electrons = raw_structure_data.spin_multiplicity - 1;
  const int n_active_pairs_in_structure =
      (raw_structure_data.n_active_electrons - n_open_shell_electrons) / 2;
  std::vector<std::vector<int>> structure_pair_indices(
      raw_structure_data.n_structures);
  for (int structure_index = 0;
       structure_index < raw_structure_data.n_structures;
       ++structure_index) {
    const int* orbitals =
        raw_structure_data.structure_orbitals_data(structure_index);
    std::vector<int>& pair_indices =
        structure_pair_indices[structure_index];
    pair_indices.reserve(n_active_pairs_in_structure);
    for (int pair_index = 0;
         pair_index < n_active_pairs_in_structure;
         ++pair_index) {
      const int first_orbital =
          orbitals[active_start + 2 * pair_index] -
          n_inactive_doubly_occupied_orbitals - 1;
      const int second_orbital =
          orbitals[active_start + 2 * pair_index + 1] -
          n_inactive_doubly_occupied_orbitals - 1;
      const int packed_pair =
          TwoElectronIndexer::packed_pair_index(first_orbital, second_orbital);
      if (packed_pair >= 0 &&
          packed_pair < approx_vbscf_active_pair_count(n_active_orbitals)) {
        pair_indices.push_back(packed_pair);
      }
    }
  }
  return structure_pair_indices;
}

std::vector<double> reconstruct_pair_occupations_from_structure_weights(
    const std::vector<std::vector<int>>& structure_pair_indices,
    const std::vector<double>& structure_weights,
    int n_active_pairs) {
  std::vector<double> reconstructed_pair_occupations(n_active_pairs, 0.0);
  for (int structure_index = 0;
       structure_index < static_cast<int>(structure_pair_indices.size());
       ++structure_index) {
    const double structure_weight = structure_weights[structure_index];
    for (int pair_index : structure_pair_indices[structure_index]) {
      reconstructed_pair_occupations[pair_index] += structure_weight;
    }
  }
  return reconstructed_pair_occupations;
}

double residual_norm(
    const std::vector<double>& residual) {
  double squared_norm = 0.0;
  for (double value : residual) {
    squared_norm += value * value;
  }
  return std::sqrt(squared_norm);
}

}  // namespace

ApproxVbScfModel build_approx_vbscf_model(
    const ApproxVbScfModelInput& input) {
  const int n_active_orbitals = input.n_active_orbitals;
  const int n_active_pairs =
      approx_vbscf_active_pair_count(n_active_orbitals);
  const auto& h1e_act = *input.active_one_electron_integrals;
  const auto& active_overlap = *input.active_orbital_overlap_matrix;
  const auto two_electron_view =
      make_active_space_two_electron_view(*input.active_two_electron_result);

  ApproxVbScfModel model;
  model.n_active_orbitals = n_active_orbitals;
  model.n_active_pairs = n_active_pairs;
  model.one_electron_reference_energy =
      input.one_electron_reference_energy;
  model.nuclear_repulsion_energy = input.nuclear_repulsion_energy;
  model.pair_terms.resize(n_active_pairs);
  model.pair_interactions =
      Eigen::MatrixXd::Zero(n_active_pairs, n_active_pairs);
  ApproxVbScfMetricInput metric_input;
  metric_input.n_active_orbitals = n_active_orbitals;
  metric_input.active_orbital_overlap_matrix =
      input.active_orbital_overlap_matrix;
  model.metric_model =
      build_approx_vbscf_metric_model(metric_input);
  ApproxVbScfPairClusterHamiltonianInput cluster_input;
  cluster_input.n_active_orbitals = n_active_orbitals;
  cluster_input.active_orbital_overlap_matrix =
      input.active_orbital_overlap_matrix;
  cluster_input.active_one_electron_integrals =
      input.active_one_electron_integrals;
  cluster_input.active_two_electron_result =
      input.active_two_electron_result;
  model.cluster_quotient_model =
      build_approx_vbscf_cluster_quotient_model(cluster_input);
  ApproxVbScfResonanceFunctionalInput resonance_input;
  resonance_input.n_active_orbitals = n_active_orbitals;
  resonance_input.cluster_quotient_model =
      &model.cluster_quotient_model;
  model.resonance_functional_model =
      build_approx_vbscf_resonance_functional_model(resonance_input);

  for (int pair_index = 0; pair_index < n_active_pairs; ++pair_index) {
    const auto [first, second] =
        unpack_approx_vbscf_active_pair(pair_index);
    ApproxVbScfPairTerm term;
    term.first_orbital = first;
    term.second_orbital = second;
    term.local_energy =
        pair_one_electron_energy(h1e_act, first, second) +
        pair_self_two_electron_energy(
            two_electron_view,
            n_active_orbitals,
            first,
            second);
    term.overlap_response =
        overlap_response(active_overlap, n_active_orbitals, first, second);
    term.cluster_quotient_energy =
        model.cluster_quotient_model.one_pair_energy[pair_index];
    term.one_pair_metric_log =
        model.metric_model.one_pair_metric_log[pair_index];
    model.pair_terms[pair_index] = term;
  }

  for (int column_pair = 0; column_pair < n_active_pairs; ++column_pair) {
    for (int row_pair = 0; row_pair < n_active_pairs; ++row_pair) {
      model.pair_interactions(row_pair, column_pair) =
          pair_interaction(
              two_electron_view,
              n_active_orbitals,
              row_pair,
              column_pair);
    }
  }

  return model;
}

ApproxVbScfState map_selected_state_to_pair_state(
    const ApproxVbScfStateMappingInput& input) {
  const auto& raw_structure_data = *input.raw_structure_data;
  const int n_active_pairs =
      approx_vbscf_active_pair_count(input.n_active_orbitals);

  double weight_sum = 0.0;
  std::vector<double> structure_weights =
      map_selected_state_structure_weights(
          raw_structure_data,
          *input.structure_overlap_matrix,
          *input.eigenvector_matrix,
          input.state_index,
          &weight_sum);

  ApproxVbScfState state;
  state.n_structures = raw_structure_data.n_structures;
  state.mapped_weight_sum = weight_sum;
  state.structure_weights = std::move(structure_weights);
  state.pair_occupations.assign(n_active_pairs, 0.0);
  state.pair_pair_occupations =
      Eigen::MatrixXd::Zero(n_active_pairs, n_active_pairs);
  const std::vector<std::vector<int>> structure_pair_indices =
      build_structure_pair_indices(
          raw_structure_data,
          input.n_active_orbitals);
  accumulate_pair_observables_from_structure_indices(
      structure_pair_indices,
      state.structure_weights,
      &state.pair_occupations,
      &state.pair_pair_occupations);
  return state;
}

ApproxVbScfResult evaluate_approx_vbscf(
    const ApproxVbScfModel& model,
    const std::vector<double>& pair_occupations) {
  return evaluate_approx_vbscf(
      model,
      pair_occupations,
      build_mean_field_pair_pair_occupations(pair_occupations));
}

ApproxVbScfResult evaluate_approx_vbscf(
    const ApproxVbScfModel& model,
    const std::vector<double>& pair_occupations,
    const Eigen::MatrixXd& pair_pair_occupations) {
  ApproxVbScfResult result;
  result.n_active_pairs = model.n_active_pairs;
  result.local_term_count = model.n_active_pairs;
  result.pair_interaction_term_count =
      model.n_active_pairs * (model.n_active_pairs - 1) / 2;
  result.resonance_functional_one_pair_term_count =
      model.resonance_functional_model.one_pair_term_count;
  result.resonance_functional_dense_pair_pair_term_count =
      model.resonance_functional_model.dense_pair_pair_term_count;
  result.resonance_functional_pair_pair_term_count =
      model.resonance_functional_model.compatible_pair_pair_term_count;
  result.pair_occupations = pair_occupations;
  result.pair_local_energy_values.resize(model.n_active_pairs, 0.0);

  for (int pair_index = 0; pair_index < model.n_active_pairs; ++pair_index) {
    const ApproxVbScfPairTerm& term = model.pair_terms[pair_index];
    const double occupation = pair_occupations[pair_index];
    const double local_energy = occupation * term.local_energy;
    result.pair_local_energy_values[pair_index] = term.local_energy;
    if (term.first_orbital == term.second_orbital) {
      result.diagonal_pair_energy += local_energy;
    } else {
      result.offdiagonal_pair_energy += local_energy;
    }
    result.overlap_response_energy +=
        occupation * term.overlap_response;
    result.cluster_quotient_pair_energy +=
        occupation * term.cluster_quotient_energy;
    result.resonance_functional_pair_energy +=
        occupation *
        model.resonance_functional_model.one_pair_energy[pair_index];
    result.one_pair_metric_log +=
        occupation * term.one_pair_metric_log;
  }

  for (int pair_q = 0; pair_q < model.n_active_pairs; ++pair_q) {
    for (int pair_p = 0; pair_p < pair_q; ++pair_p) {
      result.pair_interaction_energy +=
          pair_pair_occupations(pair_p, pair_q) *
          model.pair_interactions(pair_p, pair_q);
      result.pair_pair_metric_log +=
          pair_pair_occupations(pair_p, pair_q) *
          model.metric_model.pair_pair_metric_log(pair_p, pair_q);
      result.cluster_quotient_pair_pair_energy +=
          pair_pair_occupations(pair_p, pair_q) *
          model.cluster_quotient_model.pair_pair_energy(pair_p, pair_q);
    }
  }

  for (const ApproxVbScfResonancePairPairTerm& term :
       model.resonance_functional_model.pair_pair_terms) {
    result.resonance_functional_pair_pair_energy +=
        pair_pair_occupations(
            term.first_pair_index,
            term.second_pair_index) *
        term.connected_energy;
  }

  result.pair_energy_kernel =
      result.diagonal_pair_energy +
      result.offdiagonal_pair_energy +
      result.pair_interaction_energy;
  result.total_metric_log =
      result.one_pair_metric_log +
      result.pair_pair_metric_log;
  const double kMaxExpArgument =
      std::log(std::numeric_limits<double>::max());
  const double kMinExpArgument =
      std::log(std::numeric_limits<double>::min());
  const double bounded_metric_log =
      std::clamp(
          result.total_metric_log,
          kMinExpArgument,
          kMaxExpArgument);
  result.metric_denominator = std::exp(bounded_metric_log);
  result.diagnostic_metric_quotient_active_energy =
      result.pair_energy_kernel / result.metric_denominator;
  result.diagnostic_metric_quotient_total_energy =
      model.one_electron_reference_energy +
      result.diagnostic_metric_quotient_active_energy +
      model.nuclear_repulsion_energy;
  result.cluster_quotient_active_energy =
      result.cluster_quotient_pair_energy +
      result.cluster_quotient_pair_pair_energy;
  result.cluster_quotient_total_energy =
      model.one_electron_reference_energy +
      result.cluster_quotient_active_energy +
      model.nuclear_repulsion_energy;
  result.resonance_functional_active_energy =
      result.resonance_functional_pair_energy +
      result.resonance_functional_pair_pair_energy;
  result.resonance_functional_total_energy =
      model.one_electron_reference_energy +
      result.resonance_functional_active_energy +
      model.nuclear_repulsion_energy;
  result.approximate_active_energy =
      result.resonance_functional_active_energy;
  result.approximate_total_energy =
      model.one_electron_reference_energy +
      result.approximate_active_energy +
      model.nuclear_repulsion_energy;
  return result;
}

std::vector<double> build_uniform_pair_occupations(
    int n_active_pairs,
    double occupied_pair_count) {
  std::vector<double> pair_occupations(n_active_pairs, 0.0);
  if (n_active_pairs > 0) {
    const double uniform_occupation =
        occupied_pair_count / static_cast<double>(n_active_pairs);
    std::fill(
        pair_occupations.begin(),
        pair_occupations.end(),
        uniform_occupation);
  }
  return pair_occupations;
}

Eigen::MatrixXd build_mean_field_pair_pair_occupations(
    const std::vector<double>& pair_occupations) {
  const int n_active_pairs =
      static_cast<int>(pair_occupations.size());
  Eigen::MatrixXd pair_pair_occupations =
      Eigen::MatrixXd::Zero(n_active_pairs, n_active_pairs);
  for (int column_pair = 0; column_pair < n_active_pairs; ++column_pair) {
    for (int row_pair = 0; row_pair < n_active_pairs; ++row_pair) {
      pair_pair_occupations(row_pair, column_pair) =
          pair_occupations[row_pair] * pair_occupations[column_pair];
    }
  }
  return pair_pair_occupations;
}

std::vector<double> compute_approx_vbscf_pair_gradient(
    const ApproxVbScfModel& model,
    const std::vector<double>& pair_occupations) {
  std::vector<double> gradient(model.n_active_pairs, 0.0);
  for (int pair_index = 0; pair_index < model.n_active_pairs; ++pair_index) {
    gradient[pair_index] =
        model.resonance_functional_model.one_pair_energy[pair_index];
  }
  for (const ApproxVbScfResonancePairPairTerm& term :
       model.resonance_functional_model.pair_pair_terms) {
    gradient[term.first_pair_index] +=
        pair_occupations[term.second_pair_index] *
        term.connected_energy;
    gradient[term.second_pair_index] +=
        pair_occupations[term.first_pair_index] *
        term.connected_energy;
  }
  return gradient;
}

ApproxVbScfPairScfResult optimize_approx_vbscf_pair_occupations(
    const ApproxVbScfModel& model,
    double occupied_pair_count,
    const ApproxVbScfPairScfOptions& options) {
  ApproxVbScfPairScfResult scf_result;
  scf_result.occupied_pair_count = occupied_pair_count;
  scf_result.pair_occupations =
      build_uniform_pair_occupations(
          model.n_active_pairs,
          occupied_pair_count);
  scf_result.result =
      evaluate_approx_vbscf(model, scf_result.pair_occupations);
  scf_result.initial_total_energy =
      scf_result.result.approximate_total_energy;

  for (int iteration = 0;
       iteration < options.max_iterations;
       ++iteration) {
    const std::vector<double> gradient =
        compute_approx_vbscf_pair_gradient(
            model,
            scf_result.pair_occupations);
    const double gradient_mean =
        std::accumulate(gradient.begin(), gradient.end(), 0.0) /
        static_cast<double>(gradient.size());
    double projected_gradient_norm = 0.0;
    for (double gradient_value : gradient) {
      projected_gradient_norm = std::max(
          projected_gradient_norm,
          std::abs(gradient_value - gradient_mean));
    }
    scf_result.final_projected_gradient_norm =
        projected_gradient_norm;
    if (projected_gradient_norm <= options.gradient_tolerance) {
      scf_result.converged = true;
      scf_result.iterations = iteration;
      break;
    }

    double normalization = 0.0;
    for (int pair_index = 0; pair_index < model.n_active_pairs; ++pair_index) {
      const double current_occupation = std::max(
          scf_result.pair_occupations[pair_index],
          options.minimum_occupation);
      const double shifted_gradient =
          gradient[pair_index] - gradient_mean;
      scf_result.pair_occupations[pair_index] =
          current_occupation *
          std::exp(-options.step_size * shifted_gradient);
      normalization += scf_result.pair_occupations[pair_index];
    }
    const double scale =
        occupied_pair_count / normalization;
    for (double& occupation : scf_result.pair_occupations) {
      occupation *= scale;
    }

    scf_result.result =
        evaluate_approx_vbscf(model, scf_result.pair_occupations);
    scf_result.iterations = iteration + 1;
  }

  const std::vector<double> final_gradient =
      compute_approx_vbscf_pair_gradient(
          model,
          scf_result.pair_occupations);
  const double final_gradient_mean =
      std::accumulate(final_gradient.begin(), final_gradient.end(), 0.0) /
      static_cast<double>(final_gradient.size());
  double final_projected_gradient_norm = 0.0;
  for (double gradient_value : final_gradient) {
    final_projected_gradient_norm = std::max(
        final_projected_gradient_norm,
        std::abs(gradient_value - final_gradient_mean));
  }
  scf_result.final_projected_gradient_norm =
      final_projected_gradient_norm;
  if (final_projected_gradient_norm <= options.gradient_tolerance) {
    scf_result.converged = true;
  }
  scf_result.result =
      evaluate_approx_vbscf(model, scf_result.pair_occupations);
  return scf_result;
}

ApproxVbScfStructureProjectionResult
project_pair_occupations_to_structure_weights(
    const RawStructureData& raw_structure_data,
    int n_active_orbitals,
    const std::vector<double>& target_pair_occupations,
    const ApproxVbScfStructureProjectionOptions& options) {
  ApproxVbScfStructureProjectionResult projection_result;
  projection_result.n_structures = raw_structure_data.n_structures;
  projection_result.n_active_pairs =
      approx_vbscf_active_pair_count(n_active_orbitals);
  projection_result.structure_weights.assign(
      raw_structure_data.n_structures,
      1.0 / static_cast<double>(raw_structure_data.n_structures));

  const std::vector<std::vector<int>> structure_pair_indices =
      build_structure_pair_indices(
          raw_structure_data,
          n_active_orbitals);

  for (int iteration = 0;
       iteration < options.max_iterations;
       ++iteration) {
    projection_result.reconstructed_pair_occupations =
        reconstruct_pair_occupations_from_structure_weights(
            structure_pair_indices,
            projection_result.structure_weights,
            projection_result.n_active_pairs);

    std::vector<double> residual(projection_result.n_active_pairs, 0.0);
    for (int pair_index = 0;
         pair_index < projection_result.n_active_pairs;
         ++pair_index) {
      residual[pair_index] =
          projection_result.reconstructed_pair_occupations[pair_index] -
          target_pair_occupations[pair_index];
    }

    std::vector<double> gradient(raw_structure_data.n_structures, 0.0);
    for (int structure_index = 0;
         structure_index < raw_structure_data.n_structures;
         ++structure_index) {
      for (int pair_index : structure_pair_indices[structure_index]) {
        gradient[structure_index] += residual[pair_index];
      }
    }

    const double gradient_mean =
        std::accumulate(gradient.begin(), gradient.end(), 0.0) /
        static_cast<double>(gradient.size());
    double projected_gradient_norm = 0.0;
    for (double gradient_value : gradient) {
      projected_gradient_norm = std::max(
          projected_gradient_norm,
          std::abs(gradient_value - gradient_mean));
    }
    projection_result.projected_gradient_norm =
        projected_gradient_norm;
    projection_result.residual_norm = residual_norm(residual);
    if (projected_gradient_norm <= options.gradient_tolerance) {
      projection_result.converged = true;
      projection_result.iterations = iteration;
      break;
    }

    double normalization = 0.0;
    for (int structure_index = 0;
         structure_index < raw_structure_data.n_structures;
         ++structure_index) {
      const double current_weight = std::max(
          projection_result.structure_weights[structure_index],
          options.minimum_weight);
      const double shifted_gradient =
          gradient[structure_index] - gradient_mean;
      projection_result.structure_weights[structure_index] =
          current_weight *
          std::exp(-options.step_size * shifted_gradient);
      normalization += projection_result.structure_weights[structure_index];
    }
    for (double& structure_weight : projection_result.structure_weights) {
      structure_weight /= normalization;
    }
    projection_result.iterations = iteration + 1;
  }

  projection_result.reconstructed_pair_occupations =
      reconstruct_pair_occupations_from_structure_weights(
          structure_pair_indices,
          projection_result.structure_weights,
          projection_result.n_active_pairs);
  std::vector<double> final_residual(projection_result.n_active_pairs, 0.0);
  for (int pair_index = 0;
       pair_index < projection_result.n_active_pairs;
       ++pair_index) {
    final_residual[pair_index] =
        projection_result.reconstructed_pair_occupations[pair_index] -
        target_pair_occupations[pair_index];
  }
  std::vector<double> final_gradient(raw_structure_data.n_structures, 0.0);
  for (int structure_index = 0;
       structure_index < raw_structure_data.n_structures;
       ++structure_index) {
    for (int pair_index : structure_pair_indices[structure_index]) {
      final_gradient[structure_index] += final_residual[pair_index];
    }
  }
  const double final_gradient_mean =
      std::accumulate(final_gradient.begin(), final_gradient.end(), 0.0) /
      static_cast<double>(final_gradient.size());
  double final_projected_gradient_norm = 0.0;
  for (double gradient_value : final_gradient) {
    final_projected_gradient_norm = std::max(
        final_projected_gradient_norm,
        std::abs(gradient_value - final_gradient_mean));
  }
  projection_result.residual_norm = residual_norm(final_residual);
  projection_result.projected_gradient_norm =
      final_projected_gradient_norm;
  if (final_projected_gradient_norm <= options.gradient_tolerance) {
    projection_result.converged = true;
  }
  return projection_result;
}

}  // namespace xmvb::vb
