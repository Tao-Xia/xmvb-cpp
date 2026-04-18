#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime_c/cpp_runtime_extractor.h"
#include "vb/matrices/full_structure_expander.hpp"
#include "vb/matrices/raw_structure_subspace_selector.hpp"
#include "vb/matrices/same_spin_pair_cache.hpp"

namespace {

struct RuntimeSnapshotOwner {
  RuntimeSnapshotOwner() {
    init_cpp_runtime_snapshot(&snapshot);
  }

  ~RuntimeSnapshotOwner() {
    free_cpp_runtime_snapshot(&snapshot);
  }

  CppRuntimeSnapshot snapshot{};
};

struct SpinStringOccurrence {
  int unique_id = -1;
  std::vector<int> determinant_indices;
};

void print_usage() {
  std::cerr
      << "usage: inspect_same_spin_reuse <input.xmi>"
      << " [--raw-structure-selection full|covalent]"
      << " [--top-k <count>]"
      << " [--examples-per-string <count>]"
      << " [--show-det-indices true|false]\n";
}

bool parse_bool_argument(const std::string& value) {
  if (value == "true" || value == "1") {
    return true;
  }
  if (value == "false" || value == "0") {
    return false;
  }
  throw std::invalid_argument("invalid boolean value: " + value);
}

void apply_raw_structure_selection_argument(
    const std::string& value,
    xmvb::vb::RawStructureSelectionMode* mode) {
  if (mode == nullptr) {
    throw std::invalid_argument("mode must not be null");
  }
  if (value == "full") {
    *mode = xmvb::vb::RawStructureSelectionMode::Full;
    return;
  }
  if (value == "covalent") {
    *mode = xmvb::vb::RawStructureSelectionMode::Covalent;
    return;
  }
  throw std::invalid_argument("invalid raw structure selection: " + value);
}

std::string format_spin_string(
    const std::vector<int>& occupied_orbitals) {
  std::string text = "[";
  for (std::size_t index = 0; index < occupied_orbitals.size(); ++index) {
    if (index > 0) {
      text += ", ";
    }
    text += std::to_string(occupied_orbitals[index] + 1);
  }
  text += "]";
  return text;
}

std::string format_histogram(
    const std::map<int, int>& histogram) {
  std::string text = "{";
  bool first_entry = true;
  for (const auto& [multiplicity, unique_string_count] : histogram) {
    if (!first_entry) {
      text += ", ";
    }
    first_entry = false;
    text += std::to_string(multiplicity);
    text += ": ";
    text += std::to_string(unique_string_count);
  }
  text += "}";
  return text;
}

std::vector<SpinStringOccurrence> build_occurrences(
    const xmvb::vb::SpinDeterminantReuseTable& reuse_table) {
  std::vector<SpinStringOccurrence> occurrences(
      reuse_table.unique_determinants.size());
  for (int unique_id = 0;
       unique_id < static_cast<int>(reuse_table.unique_determinants.size());
       ++unique_id) {
    occurrences[xmvb::to_size(unique_id)].unique_id = unique_id;
  }

  for (int determinant_index = 0;
       determinant_index < static_cast<int>(reuse_table.determinant_to_unique_id.size());
       ++determinant_index) {
    const int unique_id =
        reuse_table.determinant_to_unique_id[xmvb::to_size(determinant_index)];
    occurrences[xmvb::to_size(unique_id)].determinant_indices.push_back(
        determinant_index);
  }
  return occurrences;
}

double average_multiplicity(
    const std::vector<SpinStringOccurrence>& occurrences) {
  if (occurrences.empty()) {
    return 0.0;
  }
  std::size_t total_count = 0;
  for (const auto& occurrence : occurrences) {
    total_count += occurrence.determinant_indices.size();
  }
  return static_cast<double>(total_count) /
      static_cast<double>(occurrences.size());
}

std::map<int, int> build_multiplicity_histogram(
    const std::vector<SpinStringOccurrence>& occurrences) {
  std::map<int, int> histogram;
  for (const auto& occurrence : occurrences) {
    ++histogram[static_cast<int>(occurrence.determinant_indices.size())];
  }
  return histogram;
}

void print_spin_section(
    const char* spin_label,
    const xmvb::vb::SpinDeterminantReuseTable& reuse_table,
    const std::vector<std::vector<int>>& other_spin_determinants,
    int top_k,
    int examples_per_string,
    bool show_det_indices) {
  auto occurrences = build_occurrences(reuse_table);
  const auto multiplicity_histogram = build_multiplicity_histogram(occurrences);
  const auto repeated_unique_count = static_cast<int>(std::count_if(
      occurrences.begin(),
      occurrences.end(),
      [](const SpinStringOccurrence& occurrence) {
        return occurrence.determinant_indices.size() > 1;
      }));

  std::sort(
      occurrences.begin(),
      occurrences.end(),
      [&reuse_table](const SpinStringOccurrence& left, const SpinStringOccurrence& right) {
        if (left.determinant_indices.size() != right.determinant_indices.size()) {
          return left.determinant_indices.size() > right.determinant_indices.size();
        }
        return reuse_table.unique_determinants[xmvb::to_size(left.unique_id)] <
            reuse_table.unique_determinants[xmvb::to_size(right.unique_id)];
      });

  std::cout << spin_label << "_summary\n";
  std::cout << "  unique_string_count = " << reuse_table.unique_determinants.size() << '\n';
  std::cout << "  repeated_unique_string_count = " << repeated_unique_count << '\n';
  std::cout << "  average_multiplicity = " << std::fixed << std::setprecision(6)
            << average_multiplicity(occurrences) << '\n';
  std::cout << "  multiplicity_histogram = "
            << format_histogram(multiplicity_histogram) << '\n';

  int printed = 0;
  for (const auto& occurrence : occurrences) {
    if (occurrence.determinant_indices.size() <= 1) {
      break;
    }
    if (printed >= top_k) {
      break;
    }

    const auto& occupied_orbitals =
        reuse_table.unique_determinants[xmvb::to_size(occurrence.unique_id)];
    std::cout << "  repeated_string rank=" << (printed + 1)
              << " unique_id=" << occurrence.unique_id
              << " multiplicity=" << occurrence.determinant_indices.size()
              << " occ=" << format_spin_string(occupied_orbitals) << '\n';
    if (show_det_indices) {
      std::cout << "    determinant_indices_zero_based = [";
      for (std::size_t index = 0; index < occurrence.determinant_indices.size(); ++index) {
        if (index > 0) {
          std::cout << ", ";
        }
        std::cout << occurrence.determinant_indices[index];
      }
      std::cout << "]\n";
    }

    const int example_count = std::min(
        examples_per_string,
        static_cast<int>(occurrence.determinant_indices.size()));
    for (int example_index = 0; example_index < example_count; ++example_index) {
      const int determinant_index =
          occurrence.determinant_indices[xmvb::to_size(example_index)];
      std::cout << "    det=" << determinant_index
                << " paired_other_spin="
                << format_spin_string(
                       other_spin_determinants[xmvb::to_size(determinant_index)])
                << '\n';
    }
    ++printed;
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || ((argc - 2) % 2 != 0)) {
      print_usage();
      return 1;
    }

    const std::string input_path = argv[1];
    xmvb::vb::RawStructureSelectionMode raw_structure_selection =
        xmvb::vb::RawStructureSelectionMode::Full;
    int top_k = 6;
    int examples_per_string = 4;
    bool show_det_indices = false;

    for (int argument_index = 2; argument_index < argc; argument_index += 2) {
      const std::string argument_name = argv[argument_index];
      const std::string argument_value = argv[argument_index + 1];
      if (argument_name == "--raw-structure-selection") {
        apply_raw_structure_selection_argument(argument_value, &raw_structure_selection);
      } else if (argument_name == "--top-k") {
        top_k = std::stoi(argument_value);
      } else if (argument_name == "--examples-per-string") {
        examples_per_string = std::stoi(argument_value);
      } else if (argument_name == "--show-det-indices") {
        show_det_indices = parse_bool_argument(argument_value);
      } else {
        throw std::invalid_argument("unknown argument: " + argument_name);
      }
    }

    CppRuntimeExtractionOptions extraction_options{};
    init_cpp_runtime_extraction_options(&extraction_options);
    extraction_options.skip_legacy_two_electron_integrals = 1;
    extraction_options.skip_legacy_vbguess = 1;
    extraction_options.skip_legacy_hf_setup = 1;

    RuntimeSnapshotOwner runtime_snapshot_owner;
    CppRuntimeExtractionTimings runtime_timings{};
    char error_message[1024] = {0};
    const int status = extract_cpp_runtime_snapshot_with_options(
        input_path.c_str(),
        &extraction_options,
        &runtime_snapshot_owner.snapshot,
        &runtime_timings,
        error_message,
        sizeof(error_message));
    if (status != 0) {
      throw std::runtime_error(
          error_message[0] != '\0' ? error_message : "C++ runtime extraction failed");
    }

    const auto& snapshot = runtime_snapshot_owner.snapshot;
    xmvb::vb::RawStructureData raw_structure_data;
    raw_structure_data.n_structures = snapshot.n_structures;
    raw_structure_data.n_total_electrons = snapshot.n_total_electrons;
    raw_structure_data.n_active_electrons = snapshot.n_active_electrons;
    raw_structure_data.spin_multiplicity = snapshot.spin_multiplicity;
    raw_structure_data.wavefunction_type = snapshot.wavefunction_type;
    raw_structure_data.vb_function_type = snapshot.vb_function_type;
    raw_structure_data.raw_structure_orbitals.assign(
        snapshot.raw_structure_orbitals,
        snapshot.raw_structure_orbitals +
            xmvb::to_size(snapshot.n_structures) *
                xmvb::to_size(snapshot.n_total_electrons));

    const int source_raw_structure_count = raw_structure_data.n_structures;
    const auto selected_raw_structure_indices =
        xmvb::vb::select_raw_structure_indices(raw_structure_data, raw_structure_selection);
    xmvb::vb::RawStructureData selected_raw_structure_data;
    if (static_cast<int>(selected_raw_structure_indices.size()) == raw_structure_data.n_structures) {
      selected_raw_structure_data = std::move(raw_structure_data);
    } else {
      selected_raw_structure_data = xmvb::vb::build_raw_structure_subset(
          raw_structure_data,
          selected_raw_structure_indices);
    }

    // This tool intentionally reuses the exact production determinant expansion
    // path so the repeated alpha/beta strings match the same-spin cache inputs.
    xmvb::vb::FullDeterminantStructureExpander expander;
    const xmvb::vb::FullDeterminantStructureData structure_data =
        expander.expand(selected_raw_structure_data);

    const auto alpha_reuse_table =
        xmvb::vb::build_spin_determinant_reuse_table(structure_data.alpha_det);
    const auto beta_reuse_table =
        xmvb::vb::build_spin_determinant_reuse_table(structure_data.beta_det);

    std::cout << std::setprecision(12);
    std::cout << "input_file = " << input_path << '\n';
    std::cout << "raw_structure_selection = "
              << xmvb::vb::raw_structure_selection_mode_name(raw_structure_selection) << '\n';
    std::cout << "source_raw_structure_count = " << source_raw_structure_count << '\n';
    std::cout << "selected_raw_structure_count = "
              << selected_raw_structure_data.n_structures << '\n';
    std::cout << "expanded_determinant_count = "
              << structure_data.alpha_det.size() << '\n';
    std::cout << "runtime_extraction_wall_time_seconds = "
              << runtime_timings.total_seconds << '\n';
    std::cout << "alpha_unique_string_count = "
              << alpha_reuse_table.unique_determinants.size() << '\n';
    std::cout << "beta_unique_string_count = "
              << beta_reuse_table.unique_determinants.size() << '\n';
    std::cout << "alpha_reuse_factor = "
              << std::fixed << std::setprecision(6)
              << static_cast<double>(structure_data.alpha_det.size()) /
                     static_cast<double>(alpha_reuse_table.unique_determinants.size())
              << '\n';
    std::cout << "beta_reuse_factor = "
              << std::fixed << std::setprecision(6)
              << static_cast<double>(structure_data.beta_det.size()) /
                     static_cast<double>(beta_reuse_table.unique_determinants.size())
              << '\n';

    print_spin_section(
        "alpha",
        alpha_reuse_table,
        structure_data.beta_det,
        top_k,
        examples_per_string,
        show_det_indices);
    print_spin_section(
        "beta",
        beta_reuse_table,
        structure_data.alpha_det,
        top_k,
        examples_per_string,
        show_det_indices);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
