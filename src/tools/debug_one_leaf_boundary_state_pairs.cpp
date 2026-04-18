#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/exact_separator/component_data.hpp"
#include "vb/exact_separator/leaf_boundary_message.hpp"
#include "vb/exact_separator/leaf_coefficient_operator.hpp"
#include "vb/exact_separator/spin_state_aggregate.hpp"
#include "vb/matrices/determinant_hamiltonian_resolver.hpp"
#include "vb/matrices/determinant_overlap_resolver.hpp"
#include "vb/matrices/legacy_structure_overlap.hpp"
#include "vb/matrices/prepared_active_space_context.hpp"
#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/matrices/union_graph_screening.hpp"

namespace {

using Matrix = xmvb::vb::Matrix;
using OrbitalPair = xmvb::vb::OrbitalPair;
using xmvb::vb::DeterminantHamiltonianResolver;
using xmvb::vb::DeterminantOverlapResolver;
using xmvb::vb::exact_separator::ComponentData;
using xmvb::vb::exact_separator::ComponentSpinCoefficientOperator;
using xmvb::vb::exact_separator::DirectSpinStateAggregate;
using xmvb::vb::exact_separator::OneLeafBoundarySpinAggregate;
using xmvb::vb::exact_separator::OneLeafBoundarySpinMessage;
using xmvb::vb::exact_separator::OrientationTerm;
using xmvb::vb::exact_separator::SpinPairStateKey;

struct Options {
  std::string input_path;
  int left_structure = -1;
  int right_structure = -1;
  int root_choice = 0;
  int top_states = 10;
};

struct StateMismatch {
  int root_state_index = -1;
  int leaf_state_index = -1;
  int nullity = -1;
  std::uint64_t sector_count = 0;
  double overlap_abs_error = 0.0;
  double first_cofactor_abs_error = 0.0;
  double same_spin_abs_error = 0.0;
  SpinPairStateKey root_state;
  SpinPairStateKey leaf_state;
};

void print_usage() {
  std::cerr << "usage: debug_one_leaf_boundary_state_pairs <input.xmi>"
               " --left-structure I --right-structure J --root-choice R"
               " [--top-states N]\n";
}

Options parse_arguments(int argc, char** argv) {
  if (argc < 8 || ((argc - 2) % 2 != 0)) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }

  Options options;
  options.input_path = argv[1];
  for (int argument_index = 2; argument_index < argc; argument_index += 2) {
    const std::string argument_name = argv[argument_index];
    const std::string argument_value = argv[argument_index + 1];
    if (argument_name == "--left-structure") {
      options.left_structure = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--right-structure") {
      options.right_structure = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--root-choice") {
      options.root_choice = std::stoi(argument_value);
      continue;
    }
    if (argument_name == "--top-states") {
      options.top_states = std::stoi(argument_value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument_name);
  }

  if (options.left_structure < 0 || options.right_structure < 0) {
    throw std::invalid_argument("--left-structure and --right-structure are required");
  }
  if (options.left_structure <= options.right_structure) {
    throw std::invalid_argument("left_structure must be greater than right_structure");
  }
  if (options.root_choice < 0 || options.root_choice > 1) {
    throw std::invalid_argument("--root-choice must be 0 or 1");
  }
  if (options.top_states <= 0) {
    throw std::invalid_argument("--top-states must be positive");
  }
  return options;
}

std::vector<double> flatten_column_major_matrix(const Matrix& matrix) {
  return std::vector<double>(matrix.data(), matrix.data() + matrix.size());
}

std::vector<int> build_component_ordered_support_orbitals(
    const std::vector<int>& support_orbitals,
    const std::vector<xmvb::vb::UnionGraphComponent>& components,
    int root_component_index) {
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
  const int support_size = static_cast<int>(ordered_support_orbitals.size());
  const int packed_size =
      xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
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
              xmvb::vb::TwoElectronIndexer::two_electron_storage_index(p, q, r, s))] =
              packed_active_two_electron_integrals[xmvb::to_size(
                  xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
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

double max_abs_matrix_difference(const Matrix& left, const Matrix& right) {
  if (left.rows() != right.rows() || left.cols() != right.cols()) {
    throw std::invalid_argument("matrix dimension mismatch");
  }
  double max_abs_error = 0.0;
  for (int col = 0; col < left.cols(); ++col) {
    for (int row = 0; row < left.rows(); ++row) {
      max_abs_error = std::max(
          max_abs_error,
          std::abs(left(row, col) - right(row, col)));
    }
  }
  return max_abs_error;
}

std::uint64_t count_boundary_sectors(const OneLeafBoundarySpinMessage& message) {
  std::uint64_t total = 0;
  for (const auto& family : message.families) {
    total += static_cast<std::uint64_t>(family.sectors.size());
  }
  return total;
}

std::string occ_to_string(const std::vector<int>& occ) {
  std::ostringstream out;
  out << '[';
  for (std::size_t index = 0; index < occ.size(); ++index) {
    if (index > 0U) {
      out << ',';
    }
    out << occ[index];
  }
  out << ']';
  return out.str();
}

std::vector<StateMismatch> analyze_spin_channel(
    const std::string& spin_label,
    const std::vector<SpinPairStateKey>& root_states,
    const std::vector<SpinPairStateKey>& leaf_states,
    const std::vector<double>& support_overlap,
    const std::vector<double>& support_packed_two_electron,
    int support_size,
    const DeterminantOverlapResolver& overlap_resolver,
    int top_states) {
  const std::vector<double> zero_one_electron_storage(
      xmvb::to_size(support_size) * xmvb::to_size(support_size),
      0.0);
  const DeterminantHamiltonianResolver hamiltonian_resolver(overlap_resolver);

  std::vector<StateMismatch> mismatches;
  std::uint64_t total_boundary_subdets = 0;
  std::uint64_t total_direct_subdets = 0;
  double max_overlap_abs_error = 0.0;
  double max_first_abs_error = 0.0;
  double max_same_spin_abs_error = 0.0;

  for (int root_state_index = 0;
       root_state_index < static_cast<int>(root_states.size());
       ++root_state_index) {
    for (int leaf_state_index = 0;
         leaf_state_index < static_cast<int>(leaf_states.size());
         ++leaf_state_index) {
      const auto& root_state = root_states[xmvb::to_size(root_state_index)];
      const auto& leaf_state = leaf_states[xmvb::to_size(leaf_state_index)];

      std::uint64_t boundary_subdets = 0;
      const OneLeafBoundarySpinMessage message =
          xmvb::vb::exact_separator::build_one_leaf_spin_boundary_message(
              root_state.left_occ,
              leaf_state.left_occ,
              root_state.right_occ,
              leaf_state.right_occ,
              support_overlap,
              support_size,
              overlap_resolver,
              &boundary_subdets);
      const OneLeafBoundarySpinAggregate boundary =
          xmvb::vb::exact_separator::contract_one_leaf_spin_boundary_message(
              message,
              support_packed_two_electron);
      total_boundary_subdets += boundary_subdets;

      std::vector<int> full_left_occ = root_state.left_occ;
      full_left_occ.insert(full_left_occ.end(), leaf_state.left_occ.begin(), leaf_state.left_occ.end());
      std::vector<int> full_right_occ = root_state.right_occ;
      full_right_occ.insert(full_right_occ.end(), leaf_state.right_occ.begin(), leaf_state.right_occ.end());
      std::uint64_t direct_subdets = 0;
      const DirectSpinStateAggregate direct =
          xmvb::vb::exact_separator::build_direct_spin_state_aggregate(
              full_left_occ,
              full_right_occ,
              support_overlap,
              zero_one_electron_storage,
              support_packed_two_electron,
              support_size,
              true,
              overlap_resolver,
              hamiltonian_resolver,
              &direct_subdets);
      total_direct_subdets += direct_subdets;

      const std::vector<double> overlap_submatrix =
          xmvb::vb::build_overlap_submatrix(
              full_left_occ,
              full_right_occ,
              support_overlap,
              support_size);
      const auto overlap_result =
          overlap_resolver.resolve(overlap_submatrix, static_cast<int>(full_left_occ.size()));

      const double overlap_abs_error = std::abs(boundary.overlap - direct.overlap);
      const double first_abs_error =
          max_abs_matrix_difference(boundary.first_cofactor, direct.first_cofactor);
      const double same_spin_abs_error =
          std::abs(boundary.same_spin_two_electron - direct.same_spin_two_electron);
      max_overlap_abs_error = std::max(max_overlap_abs_error, overlap_abs_error);
      max_first_abs_error = std::max(max_first_abs_error, first_abs_error);
      max_same_spin_abs_error = std::max(max_same_spin_abs_error, same_spin_abs_error);

      if (same_spin_abs_error <= 1.0e-12 &&
          first_abs_error <= 1.0e-12 &&
          overlap_abs_error <= 1.0e-12) {
        continue;
      }

      mismatches.push_back(StateMismatch{
          .root_state_index = root_state_index,
          .leaf_state_index = leaf_state_index,
          .nullity = overlap_result.nullity,
          .sector_count = count_boundary_sectors(message),
          .overlap_abs_error = overlap_abs_error,
          .first_cofactor_abs_error = first_abs_error,
          .same_spin_abs_error = same_spin_abs_error,
          .root_state = root_state,
          .leaf_state = leaf_state,
      });
    }
  }

  std::sort(
      mismatches.begin(),
      mismatches.end(),
      [](const StateMismatch& left, const StateMismatch& right) {
        if (left.same_spin_abs_error != right.same_spin_abs_error) {
          return left.same_spin_abs_error > right.same_spin_abs_error;
        }
        if (left.first_cofactor_abs_error != right.first_cofactor_abs_error) {
          return left.first_cofactor_abs_error > right.first_cofactor_abs_error;
        }
        return left.overlap_abs_error > right.overlap_abs_error;
      });
  if (static_cast<int>(mismatches.size()) > top_states) {
    mismatches.resize(xmvb::to_size(top_states));
  }

  std::cout << "spin = " << spin_label << '\n';
  std::cout << "  root_state_count = " << root_states.size() << '\n';
  std::cout << "  leaf_state_count = " << leaf_states.size() << '\n';
  std::cout << "  total_state_pairs = "
            << static_cast<std::uint64_t>(root_states.size()) *
                   static_cast<std::uint64_t>(leaf_states.size())
            << '\n';
  std::cout << "  total_boundary_subdets = " << total_boundary_subdets << '\n';
  std::cout << "  total_direct_subdets = " << total_direct_subdets << '\n';
  std::cout << "  max_overlap_abs_error = " << max_overlap_abs_error << '\n';
  std::cout << "  max_first_cofactor_abs_error = " << max_first_abs_error << '\n';
  std::cout << "  max_same_spin_abs_error = " << max_same_spin_abs_error << '\n';
  std::cout << "  mismatch_count = " << mismatches.size() << '\n';
  for (const auto& mismatch : mismatches) {
    std::cout << "  state root=" << mismatch.root_state_index
              << " leaf=" << mismatch.leaf_state_index
              << " nullity=" << mismatch.nullity
              << " sectors=" << mismatch.sector_count
              << " overlap_err=" << mismatch.overlap_abs_error
              << " first_err=" << mismatch.first_cofactor_abs_error
              << " same_spin_err=" << mismatch.same_spin_abs_error << '\n';
    std::cout << "    root_left=" << occ_to_string(mismatch.root_state.left_occ)
              << " root_right=" << occ_to_string(mismatch.root_state.right_occ) << '\n';
    std::cout << "    leaf_left=" << occ_to_string(mismatch.leaf_state.left_occ)
              << " leaf_right=" << occ_to_string(mismatch.leaf_state.right_occ) << '\n';
  }

  return mismatches;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(options.input_path);
    const auto& raw_structure_data = load_result.raw_structure_data;

    if (raw_structure_data.spin_multiplicity != 1) {
      throw std::runtime_error(
          "debug_one_leaf_boundary_state_pairs currently supports only singlet closed-shell structures");
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
          "debug_one_leaf_boundary_state_pairs requires exact packed active-space ERIs");
    }

    const auto& active_overlap_storage =
        prepared_active_space.orbital_result.active_orbital_overlap_matrix;
    const auto& packed_active_two_electron_integrals =
        prepared_active_space.active_space_two_electron_result
            .packed_active_two_electron_integrals;
    const int n_active_orbitals =
        load_result.input.orbital_preparation_input.n_active_orbitals;

    if (options.left_structure >= raw_structure_data.n_structures ||
        options.right_structure >= raw_structure_data.n_structures) {
      throw std::invalid_argument("requested structure index is out of range");
    }

    const auto left_pairs = xmvb::vb::extract_active_pairs(raw_structure_data, options.left_structure);
    const auto right_pairs =
        xmvb::vb::extract_active_pairs(raw_structure_data, options.right_structure);
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
      throw std::runtime_error("selected structure pair is not a one-leaf pair");
    }

    const auto ordered_support_orbitals = build_component_ordered_support_orbitals(
        support_orbitals,
        union_components,
        options.root_choice);
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
      throw std::runtime_error("component ordering changed the number of union components");
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
    const int support_size = static_cast<int>(ordered_support_orbitals.size());

    const ComponentSpinCoefficientOperator root_operator =
        xmvb::vb::exact_separator::build_component_spin_coefficient_operator(
            ordered_components.front());
    const ComponentSpinCoefficientOperator leaf_operator =
        xmvb::vb::exact_separator::build_component_spin_coefficient_operator(
            ordered_components.back());

    std::cout << std::setprecision(15);
    std::cout << "left_structure = " << options.left_structure << '\n';
    std::cout << "right_structure = " << options.right_structure << '\n';
    std::cout << "root_choice = " << options.root_choice << '\n';
    std::cout << "support_size = " << support_size << '\n';
    std::cout << "root_raw_nonzero_pair_count = " << root_operator.raw_nonzero_pair_count << '\n';
    std::cout << "leaf_raw_nonzero_pair_count = " << leaf_operator.raw_nonzero_pair_count << '\n';

    DeterminantOverlapResolver overlap_resolver;
    analyze_spin_channel(
        "alpha",
        root_operator.alpha_states,
        leaf_operator.alpha_states,
        support_overlap,
        support_packed_two_electron,
        support_size,
        overlap_resolver,
        options.top_states);
    analyze_spin_channel(
        "beta",
        root_operator.beta_states,
        leaf_operator.beta_states,
        support_overlap,
        support_packed_two_electron,
        support_size,
        overlap_resolver,
        options.top_states);
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    std::cerr << "error: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
}
