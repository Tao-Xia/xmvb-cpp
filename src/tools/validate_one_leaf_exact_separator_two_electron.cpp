#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/exact_separator/component_data.hpp"
#include "vb/exact_separator/two_electron.hpp"
#include "vb/matrices/legacy_structure_overlap.hpp"
#include "vb/matrices/prepared_active_space_context.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/matrices/union_graph_screening.hpp"

namespace {

using Matrix = xmvb::vb::Matrix;
using OrbitalPair = xmvb::vb::OrbitalPair;
using xmvb::vb::TwoElectronIndexer;
using xmvb::vb::exact_separator::CollapsedTwoElectronOneLeafStarPairStats;
using xmvb::vb::exact_separator::ComponentData;
using xmvb::vb::exact_separator::OrientationTerm;

struct Options {
  std::string input_path;
  int filter_left_structure = -1;
  int filter_right_structure = -1;
  int max_pairs = 0;
  int top_examples = 8;
  double tolerance = 1.0e-10;
};

struct Example {
  int left_structure = 0;
  int right_structure = 0;
  int root_choice = 0;
  int support_size = 0;
  double overlap_abs_error = 0.0;
  double alpha_abs_error = 0.0;
  double beta_abs_error = 0.0;
  double opposite_abs_error = 0.0;
  double total_abs_error = 0.0;
  std::uint64_t processed_term_quadruple_count = 0;
  std::uint64_t alpha_mask_state_count = 0;
  std::uint64_t beta_mask_state_count = 0;
  std::uint64_t subdeterminant_evaluations = 0;
};

void print_usage() {
  std::cerr << "usage: validate_one_leaf_exact_separator_two_electron <input.xmi>"
               " [--left-structure I --right-structure J]"
               " [--max-pairs N]"
               " [--top-examples N]"
               " [--tolerance F]\n";
}

Options parse_arguments(int argc, char** argv) {
  if (argc < 2 || ((argc - 2) % 2 != 0)) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }

  Options options;
  options.input_path = argv[1];
  for (int argument_index = 2; argument_index < argc; argument_index += 2) {
    const std::string argument_name = argv[argument_index];
    const std::string argument_value = argv[argument_index + 1];
    if (argument_name == "--left-structure") {
      options.filter_left_structure = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--right-structure") {
      options.filter_right_structure = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--max-pairs") {
      options.max_pairs = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--top-examples") {
      options.top_examples = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--tolerance") {
      options.tolerance = std::stod(argument_value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument_name);
  }

  if (options.max_pairs < 0) {
    throw std::invalid_argument("--max-pairs must be >= 0");
  }
  if ((options.filter_left_structure < 0) != (options.filter_right_structure < 0)) {
    throw std::invalid_argument(
        "--left-structure and --right-structure must be provided together");
  }
  if (options.top_examples <= 0) {
    throw std::invalid_argument("--top-examples must be positive");
  }
  if (options.tolerance <= 0.0) {
    throw std::invalid_argument("--tolerance must be positive");
  }
  return options;
}

std::vector<std::pair<int, int>> build_pair_list(
    int structure_count,
    int filter_left_structure,
    int filter_right_structure,
    int max_pairs) {
  if (filter_left_structure >= 0 && filter_right_structure >= 0) {
    if (filter_left_structure >= structure_count ||
        filter_right_structure >= structure_count ||
        filter_left_structure <= filter_right_structure) {
      throw std::invalid_argument("requested structure filter is out of range");
    }
    return {{filter_left_structure, filter_right_structure}};
  }

  std::vector<std::pair<int, int>> pairs;
  for (int left_structure = 0; left_structure < structure_count; ++left_structure) {
    for (int right_structure = 0; right_structure < left_structure; ++right_structure) {
      pairs.emplace_back(left_structure, right_structure);
    }
  }
  if (max_pairs > 0 && static_cast<int>(pairs.size()) > max_pairs) {
    pairs.resize(xmvb::to_size(max_pairs));
  }
  return pairs;
}

std::vector<double> flatten_column_major_matrix(const Matrix& matrix) {
  return std::vector<double>(matrix.data(), matrix.data() + matrix.size());
}

std::vector<int> build_component_ordered_support_orbitals(
    const std::vector<int>& support_orbitals,
    const std::vector<xmvb::vb::UnionGraphComponent>& components,
    int root_component_index) {
  // The recurrence assumes that the root component occupies the leading
  // contiguous support block and the leaf component occupies the trailing block.
  // Reordering the support orbitals does not change the exact raw-VB value as
  // long as the same relabeling is applied consistently to overlaps, ERIs, and
  // component-local determinant terms.
  if (components.size() != 2U) {
    throw std::invalid_argument(
        "build_component_ordered_support_orbitals expects exactly two components");
  }
  const int leaf_component_index = 1 - root_component_index;
  std::vector<int> ordered_support_orbitals;
  ordered_support_orbitals.reserve(support_orbitals.size());

  const auto append_component = [&](int component_index) {
    std::vector<int> local_vertices =
        components[xmvb::to_size(component_index)].local_vertices;
    std::sort(local_vertices.begin(), local_vertices.end());
    for (const int local_vertex : local_vertices) {
      ordered_support_orbitals.push_back(
          support_orbitals[xmvb::to_size(local_vertex)]);
    }
  };

  append_component(root_component_index);
  append_component(leaf_component_index);
  return ordered_support_orbitals;
}

std::vector<xmvb::vb::UnionGraphComponent> sort_components_by_min_vertex(
    std::vector<xmvb::vb::UnionGraphComponent> components) {
  std::sort(
      components.begin(),
      components.end(),
      [](const auto& left, const auto& right) {
        const int left_min =
            *std::min_element(left.local_vertices.begin(), left.local_vertices.end());
        const int right_min =
            *std::min_element(right.local_vertices.begin(), right.local_vertices.end());
        return left_min < right_min;
      });
  return components;
}

std::vector<OrientationTerm> enumerate_orientation_terms(
    const std::vector<OrbitalPair>& pairs) {
  // The validator must use the exact same component-local determinant expansion
  // as the determinant-space baseline. Reusing the legacy enumerator guarantees
  // that local coefficients and canonicalization signs match the exact raw-VB
  // determinant basis.
  const auto legacy_terms = xmvb::vb::enumerate_legacy_determinant_terms(pairs);
  std::vector<OrientationTerm> terms;
  terms.reserve(legacy_terms.size());
  for (const auto& legacy_term : legacy_terms) {
    terms.push_back(OrientationTerm{
        .alpha_occ = legacy_term.alpha_occ,
        .beta_occ = legacy_term.beta_occ,
        .coefficient = legacy_term.coefficient,
    });
  }
  return terms;
}

ComponentData build_local_component_data(
    int graph_node,
    const xmvb::vb::UnionGraphComponent& component) {
  ComponentData result;
  result.graph_node = graph_node;
  result.left_pairs = component.left_pairs;
  result.right_pairs = component.right_pairs;
  result.left_orientation_terms = enumerate_orientation_terms(component.left_pairs);
  result.right_orientation_terms = enumerate_orientation_terms(component.right_pairs);
  return result;
}

std::vector<double> build_support_local_packed_two_electron_integrals(
    const std::vector<int>& ordered_support_orbitals,
    const std::vector<double>& packed_active_two_electron_integrals) {
  // The exact-separator kernels use support-local orbital labels
  // `0..support_size-1`. This helper slices the full active-space packed ERI
  // tensor into the exact same packed convention after applying the support
  // relabeling.
  const int support_size = static_cast<int>(ordered_support_orbitals.size());
  const int packed_size =
      TwoElectronIndexer::two_electron_storage_index(
          support_size - 1,
          support_size - 1,
          support_size - 1,
          support_size - 1) +
      1;
  std::vector<double> support_local_packed(
      xmvb::to_size(packed_size),
      0.0);

  for (int p = 0; p < support_size; ++p) {
    const int full_p = ordered_support_orbitals[xmvb::to_size(p)];
    for (int q = 0; q < support_size; ++q) {
      const int full_q = ordered_support_orbitals[xmvb::to_size(q)];
      for (int r = 0; r < support_size; ++r) {
        const int full_r = ordered_support_orbitals[xmvb::to_size(r)];
        for (int s = 0; s < support_size; ++s) {
          const int full_s = ordered_support_orbitals[xmvb::to_size(s)];
          support_local_packed[xmvb::to_size(
              TwoElectronIndexer::two_electron_storage_index(p, q, r, s))] =
              packed_active_two_electron_integrals[xmvb::to_size(
                  TwoElectronIndexer::two_electron_storage_index(
                      full_p,
                      full_q,
                      full_r,
                      full_s))];
        }
      }
    }
  }

  return support_local_packed;
}

void maybe_record_example(
    const Options& options,
    Example example,
    std::vector<Example>* examples) {
  if (examples == nullptr) {
    return;
  }
  examples->push_back(std::move(example));
  std::sort(
      examples->begin(),
      examples->end(),
      [](const Example& left, const Example& right) {
        if (left.total_abs_error != right.total_abs_error) {
          return left.total_abs_error > right.total_abs_error;
        }
        return left.overlap_abs_error > right.overlap_abs_error;
      });
  if (static_cast<int>(examples->size()) > options.top_examples) {
    examples->resize(xmvb::to_size(options.top_examples));
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
    const auto& raw_structure_data = load_result.raw_structure_data;

    if (raw_structure_data.spin_multiplicity != 1) {
      throw std::runtime_error(
          "validate_one_leaf_exact_separator_two_electron currently supports only singlet closed-shell structures");
    }

    xmvb::vb::ActiveSpaceOrbitalPreparer orbital_preparer;
    xmvb::vb::AoEffectiveOneElectronBuilder ao_effective_one_electron_builder;
    xmvb::vb::ActiveSpaceOneElectronBuilder active_space_one_electron_builder;
    xmvb::vb::ActiveSpaceTwoElectronBuilder active_space_two_electron_builder;
    const auto timed_active_space = xmvb::vb::prepare_timed_active_space_context(
        load_result.input,
        orbital_preparer,
        ao_effective_one_electron_builder,
        active_space_one_electron_builder,
        active_space_two_electron_builder);
    const auto& prepared_active_space = timed_active_space.prepared_active_space;
    if (prepared_active_space.active_space_two_electron_result.representation !=
        xmvb::vb::ActiveSpaceTwoElectronRepresentation::PackedExact) {
      throw std::runtime_error(
          "validate_one_leaf_exact_separator_two_electron requires exact packed active-space ERIs");
    }

    const auto& active_overlap_storage =
        prepared_active_space.orbital_result.active_orbital_overlap_matrix;
    const auto& packed_active_two_electron_integrals =
        prepared_active_space.active_space_two_electron_result
            .packed_active_two_electron_integrals;
    const int n_active_orbitals =
        load_result.input.orbital_preparation_input.n_active_orbitals;

    std::vector<std::vector<OrbitalPair>> structure_pairs(
        xmvb::to_size(raw_structure_data.n_structures));
    for (int structure_index = 0;
         structure_index < raw_structure_data.n_structures;
         ++structure_index) {
      structure_pairs[xmvb::to_size(structure_index)] =
          xmvb::vb::extract_active_pairs(raw_structure_data, structure_index);
    }

    const auto pair_list = build_pair_list(
        raw_structure_data.n_structures,
        options.filter_left_structure,
        options.filter_right_structure,
        options.max_pairs);
    if (pair_list.empty()) {
      throw std::runtime_error("no structure pairs selected");
    }

    xmvb::vb::DeterminantOverlapResolver overlap_resolver;
    int total_pairs = 0;
    int one_leaf_pairs = 0;
    int tested_root_choices = 0;
    int exact_match_count = 0;
    int mismatch_count = 0;
    double max_overlap_abs_error = 0.0;
    double max_alpha_abs_error = 0.0;
    double max_beta_abs_error = 0.0;
    double max_opposite_abs_error = 0.0;
    double max_total_abs_error = 0.0;
    std::vector<Example> examples;

    for (const auto& [left_structure, right_structure] : pair_list) {
      ++total_pairs;
      const auto& left_pairs = structure_pairs[xmvb::to_size(left_structure)];
      const auto& right_pairs = structure_pairs[xmvb::to_size(right_structure)];

      const auto support_orbitals =
          xmvb::vb::build_support_orbitals(left_pairs, right_pairs);
      const auto support_index = xmvb::vb::build_support_index(support_orbitals);
      const auto left_pairs_local =
          xmvb::vb::remap_pairs_to_support(left_pairs, support_index);
      const auto right_pairs_local =
          xmvb::vb::remap_pairs_to_support(right_pairs, support_index);
      const auto union_components = xmvb::vb::build_union_graph_components(
          left_pairs_local,
          right_pairs_local,
          support_orbitals);
      if (union_components.size() != 2U) {
        continue;
      }
      ++one_leaf_pairs;

      for (int root_choice = 0; root_choice < 2; ++root_choice) {
        ++tested_root_choices;
        const auto ordered_support_orbitals = build_component_ordered_support_orbitals(
            support_orbitals,
            union_components,
            root_choice);
        const auto ordered_support_index =
            xmvb::vb::build_support_index(ordered_support_orbitals);
        const auto ordered_left_pairs =
            xmvb::vb::remap_pairs_to_support(left_pairs, ordered_support_index);
        const auto ordered_right_pairs =
            xmvb::vb::remap_pairs_to_support(right_pairs, ordered_support_index);
        const auto ordered_union_components = sort_components_by_min_vertex(
            xmvb::vb::build_union_graph_components(
                ordered_left_pairs,
                ordered_right_pairs,
                ordered_support_orbitals));
        if (ordered_union_components.size() != 2U) {
          throw std::runtime_error(
              "component ordering changed the number of union components");
        }

        std::vector<ComponentData> ordered_components;
        ordered_components.reserve(2);
        for (int component_index = 0; component_index < 2; ++component_index) {
          ordered_components.push_back(build_local_component_data(
              component_index,
              ordered_union_components[xmvb::to_size(component_index)]));
        }

        const std::vector<double> support_overlap = flatten_column_major_matrix(
            xmvb::vb::build_support_overlap_matrix(
                ordered_support_orbitals,
                active_overlap_storage,
                n_active_orbitals));
        const std::vector<double> support_packed_two_electron =
            build_support_local_packed_two_electron_integrals(
                ordered_support_orbitals,
                packed_active_two_electron_integrals);

        const CollapsedTwoElectronOneLeafStarPairStats stats =
            xmvb::vb::exact_separator::
                evaluate_component_ordered_open_state_star_pair_two_electron_one_leaf(
                    support_overlap,
                    support_packed_two_electron,
                    static_cast<int>(ordered_support_orbitals.size()),
                    ordered_components,
                    overlap_resolver);

        max_overlap_abs_error =
            std::max(max_overlap_abs_error, stats.overlap_absolute_error);
        max_alpha_abs_error = std::max(
            max_alpha_abs_error,
            stats.same_spin_alpha_absolute_error);
        max_beta_abs_error = std::max(
            max_beta_abs_error,
            stats.same_spin_beta_absolute_error);
        max_opposite_abs_error = std::max(
            max_opposite_abs_error,
            stats.opposite_spin_absolute_error);
        max_total_abs_error = std::max(
            max_total_abs_error,
            stats.total_two_electron_absolute_error);

        const bool matches =
            stats.overlap_absolute_error <= options.tolerance &&
            stats.same_spin_alpha_absolute_error <= options.tolerance &&
            stats.same_spin_beta_absolute_error <= options.tolerance &&
            stats.opposite_spin_absolute_error <= options.tolerance &&
            stats.total_two_electron_absolute_error <= options.tolerance;
        if (matches) {
          ++exact_match_count;
        } else {
          ++mismatch_count;
          maybe_record_example(
              options,
              Example{
                  .left_structure = left_structure,
                  .right_structure = right_structure,
                  .root_choice = root_choice,
                  .support_size = static_cast<int>(ordered_support_orbitals.size()),
                  .overlap_abs_error = stats.overlap_absolute_error,
                  .alpha_abs_error = stats.same_spin_alpha_absolute_error,
                  .beta_abs_error = stats.same_spin_beta_absolute_error,
                  .opposite_abs_error = stats.opposite_spin_absolute_error,
                  .total_abs_error = stats.total_two_electron_absolute_error,
                  .processed_term_quadruple_count =
                      stats.processed_term_quadruple_count,
                  .alpha_mask_state_count = stats.alpha_mask_state_count,
                  .beta_mask_state_count = stats.beta_mask_state_count,
                  .subdeterminant_evaluations = stats.subdeterminant_evaluations,
              },
              &examples);
        }
      }
    }

    if (tested_root_choices == 0) {
      throw std::runtime_error("no one-leaf structure pairs were validated");
    }

    std::cout << std::setprecision(15);
    std::cout << "total_pairs = " << total_pairs << '\n';
    std::cout << "one_leaf_pairs = " << one_leaf_pairs << '\n';
    std::cout << "tested_root_choices = " << tested_root_choices << '\n';
    std::cout << "exact_match_count = " << exact_match_count << '\n';
    std::cout << "mismatch_count = " << mismatch_count << '\n';
    std::cout << "max_overlap_abs_error = " << max_overlap_abs_error << '\n';
    std::cout << "max_same_spin_alpha_abs_error = " << max_alpha_abs_error << '\n';
    std::cout << "max_same_spin_beta_abs_error = " << max_beta_abs_error << '\n';
    std::cout << "max_opposite_spin_abs_error = " << max_opposite_abs_error << '\n';
    std::cout << "max_total_two_electron_abs_error = " << max_total_abs_error << '\n';

    if (!examples.empty()) {
      std::cout << "top_mismatches:\n";
      for (const auto& example : examples) {
        std::cout << "  left=" << example.left_structure
                  << " right=" << example.right_structure
                  << " root=" << example.root_choice
                  << " support=" << example.support_size
                  << " overlap_err=" << example.overlap_abs_error
                  << " alpha_err=" << example.alpha_abs_error
                  << " beta_err=" << example.beta_abs_error
                  << " opposite_err=" << example.opposite_abs_error
                  << " total_err=" << example.total_abs_error
                  << " term_quadruples=" << example.processed_term_quadruple_count
                  << " alpha_masks=" << example.alpha_mask_state_count
                  << " beta_masks=" << example.beta_mask_state_count
                  << " subdets=" << example.subdeterminant_evaluations
                  << '\n';
      }
    }

    return (mismatch_count == 0) ? 0 : 1;
  } catch (const std::exception& exception) {
    std::cerr << "error: " << exception.what() << '\n';
    return 1;
  }
}
