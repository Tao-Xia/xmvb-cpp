#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/full_structure_expander.hpp"
#include "xmvb/indexing.hpp"

namespace {

extern "C" int str2det(
    int* str,
    int* det,
    int* ndet,
    int nel,
    int nmul,
    int nd,
    int wfntyp);

extern "C" int detect_ndb(
    int* ntstr,
    int nstr,
    int nel,
    int nmul,
    int wfntyp);

using OccupationKey = std::pair<std::vector<int>, std::vector<int>>;
using DeterminantTermMap = std::map<OccupationKey, double>;

struct Options {
  std::string input_path;
};

void print_usage() {
  std::cerr << "usage: check_full_structure_expander_legacy <input.xmi>\n";
}

Options parse_arguments(int argc, char** argv) {
  if (argc != 2) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }
  return {.input_path = argv[1]};
}

int canonicalize_spin_string(std::vector<int>* occupied_orbitals) {
  if (occupied_orbitals == nullptr) {
    throw std::invalid_argument("occupied_orbitals must not be null");
  }
  int permutation_sign = 1;
  for (std::size_t left_index = 0;
       left_index + 1 < occupied_orbitals->size();
       ++left_index) {
    for (std::size_t right_index = left_index + 1;
         right_index < occupied_orbitals->size();
         ++right_index) {
      if ((*occupied_orbitals)[left_index] > (*occupied_orbitals)[right_index]) {
        std::swap(
            (*occupied_orbitals)[left_index],
            (*occupied_orbitals)[right_index]);
        permutation_sign = -permutation_sign;
      }
    }
  }
  return permutation_sign;
}

void erase_near_zero_terms(DeterminantTermMap* term_map) {
  if (term_map == nullptr) {
    throw std::invalid_argument("term_map must not be null");
  }
  for (auto iterator = term_map->begin(); iterator != term_map->end();) {
    if (std::abs(iterator->second) <= 1.0e-12) {
      iterator = term_map->erase(iterator);
    } else {
      ++iterator;
    }
  }
}

std::vector<int> collect_unique_active_labels(
    const xmvb::vb::RawStructureData& raw_structure_data,
    int n_inactive_doubly_occupied_orbitals) {
  std::vector<int> labels;
  const int active_start = 2 * n_inactive_doubly_occupied_orbitals;
  for (int structure_index = 0;
       structure_index < raw_structure_data.n_structures;
       ++structure_index) {
    const int* structure =
        raw_structure_data.structure_orbitals_data(structure_index);
    for (int electron_index = active_start;
         electron_index < raw_structure_data.n_total_electrons;
         ++electron_index) {
      labels.push_back(structure[electron_index]);
    }
  }
  std::sort(labels.begin(), labels.end());
  labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
  return labels;
}

std::vector<DeterminantTermMap> build_cpp_terms_by_structure(
    const xmvb::vb::FullDeterminantStructureData& expanded_data,
    int n_inactive_doubly_occupied_orbitals) {
  std::vector<DeterminantTermMap> terms_by_structure(
      xmvb::to_size(expanded_data.n_structures));
  for (std::size_t determinant_index = 0;
       determinant_index < expanded_data.alpha_det.size();
       ++determinant_index) {
    std::vector<int> alpha_absolute =
        expanded_data.alpha_det[determinant_index];
    std::vector<int> beta_absolute =
        expanded_data.beta_det[determinant_index];
    for (int& orbital_index : alpha_absolute) {
      orbital_index += n_inactive_doubly_occupied_orbitals + 1;
    }
    for (int& orbital_index : beta_absolute) {
      orbital_index += n_inactive_doubly_occupied_orbitals + 1;
    }

    const OccupationKey key{
        std::move(alpha_absolute),
        std::move(beta_absolute)};
    for (const auto& term :
         expanded_data.determinant_to_structure_terms[determinant_index]) {
      terms_by_structure[xmvb::to_size(term.structure_index)][key] +=
          term.coefficient;
    }
  }

  for (auto& structure_terms : terms_by_structure) {
    erase_near_zero_terms(&structure_terms);
  }
  return terms_by_structure;
}

std::vector<DeterminantTermMap> build_legacy_terms_by_structure(
    const xmvb::vb::RawStructureData& raw_structure_data) {
  const int n_total_electrons = raw_structure_data.n_total_electrons;
  const int spin_multiplicity = raw_structure_data.spin_multiplicity;
  const int n_beta_electrons =
      (n_total_electrons + 1 - spin_multiplicity) / 2;
  const int n_inactive_doubly_occupied_orbitals =
      (raw_structure_data.n_total_electrons -
       raw_structure_data.n_active_electrons) /
      2;
  const int n_open_shell_electrons = spin_multiplicity - 1;
  const int n_active_beta_electrons =
      (raw_structure_data.n_active_electrons - n_open_shell_electrons) / 2;
  const int n_active_alpha_electrons =
      raw_structure_data.n_active_electrons - n_active_beta_electrons;
  int max_structure_determinants = 1;
  for (int electron_index = 0;
       electron_index < n_active_beta_electrons;
       ++electron_index) {
    if (max_structure_determinants > (1 << 20)) {
      throw std::runtime_error(
          "diagnostic refuses to allocate more than 2^20 determinants per structure");
    }
    max_structure_determinants *= 2;
  }
  if (max_structure_determinants <= 0) {
    throw std::runtime_error("invalid max_structure_determinants");
  }

  // `str2det` returns one legacy full determinant as a flat `nel` vector in
  // determinant-wavefunction ordering:
  // `[paired_alpha, paired_beta, open_shell_alpha]`.
  // Recover the active alpha/beta strings, canonicalize them, and accumulate
  // the permutation sign exactly as the old determinant summation does.
  std::vector<int> determinant_storage(
      xmvb::to_size(max_structure_determinants) * n_total_electrons,
      0);
  std::vector<DeterminantTermMap> terms_by_structure(
      xmvb::to_size(raw_structure_data.n_structures));
  for (int structure_index = 0;
       structure_index < raw_structure_data.n_structures;
       ++structure_index) {
    if (structure_index < 4) {
      std::cerr << "stage=legacy_str2det structure=" << structure_index << '\n';
      std::cerr.flush();
    }
    int n_determinants = max_structure_determinants;
    const int status = str2det(
        const_cast<int*>(raw_structure_data.structure_orbitals_data(structure_index)),
        determinant_storage.data(),
        &n_determinants,
        raw_structure_data.n_total_electrons,
        raw_structure_data.spin_multiplicity,
        n_inactive_doubly_occupied_orbitals,
        raw_structure_data.wavefunction_type);
    if (status != 0) {
      throw std::runtime_error(
          "legacy str2det failed for structure " +
          std::to_string(structure_index) +
          " with status=" + std::to_string(status));
    }

    auto& structure_terms = terms_by_structure[xmvb::to_size(structure_index)];
    for (int determinant_index = 0;
         determinant_index < n_determinants;
         ++determinant_index) {
      const int* determinant =
          determinant_storage.data() +
          xmvb::to_size(determinant_index) * n_total_electrons;

      std::vector<int> alpha_active;
      std::vector<int> beta_active;
      alpha_active.reserve(n_active_alpha_electrons);
      beta_active.reserve(n_active_beta_electrons);

      for (int electron_index = n_inactive_doubly_occupied_orbitals;
           electron_index < n_beta_electrons;
           ++electron_index) {
        alpha_active.push_back(determinant[electron_index]);
        beta_active.push_back(determinant[electron_index + n_beta_electrons]);
      }
      for (int open_shell_index = 0;
           open_shell_index < n_open_shell_electrons;
           ++open_shell_index) {
        alpha_active.push_back(
            determinant[2 * n_beta_electrons + open_shell_index]);
      }

      const int alpha_sign = canonicalize_spin_string(&alpha_active);
      const int beta_sign = canonicalize_spin_string(&beta_active);
      structure_terms[{alpha_active, beta_active}] +=
          static_cast<double>(alpha_sign * beta_sign);
    }
    erase_near_zero_terms(&structure_terms);
  }

  return terms_by_structure;
}

void print_term(
    const char* label,
    const OccupationKey& key,
    double coefficient) {
  std::cout << label << " coeff=" << coefficient << " alpha=[";
  for (std::size_t index = 0; index < key.first.size(); ++index) {
    if (index > 0) {
      std::cout << ',';
    }
    std::cout << key.first[index];
  }
  std::cout << "] beta=[";
  for (std::size_t index = 0; index < key.second.size(); ++index) {
    if (index > 0) {
      std::cout << ',';
    }
    std::cout << key.second[index];
  }
  std::cout << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    std::cerr << "stage=load\n";
    std::cerr.flush();
    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.expand_selected_raw_structures = false;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(options.input_path, load_options);
    const auto& raw_structure_data = load_result.raw_structure_data;

    const int n_inactive_doubly_occupied_orbitals =
        (raw_structure_data.n_total_electrons -
         raw_structure_data.n_active_electrons) /
        2;
    std::cerr << "stage=detect_ndb\n";
    std::cerr.flush();
    const int detected_ndb = detect_ndb(
        const_cast<int*>(raw_structure_data.raw_structure_orbitals.data()),
        raw_structure_data.n_structures,
        raw_structure_data.n_total_electrons,
        raw_structure_data.spin_multiplicity,
        raw_structure_data.wavefunction_type);
    std::cerr << "stage=collect_labels\n";
    std::cerr.flush();
    const auto active_labels = collect_unique_active_labels(
        raw_structure_data,
        n_inactive_doubly_occupied_orbitals);

    std::cerr << "stage=expand_cpp\n";
    std::cerr.flush();
    xmvb::vb::FullDeterminantStructureExpander expander;
    const auto expanded_data = expander.expand(raw_structure_data);
    std::cerr << "stage=cpp_maps\n";
    std::cerr.flush();
    const auto cpp_terms_by_structure = build_cpp_terms_by_structure(
        expanded_data,
        n_inactive_doubly_occupied_orbitals);
    std::cerr << "stage=legacy_maps\n";
    std::cerr.flush();
    const auto legacy_terms_by_structure =
        build_legacy_terms_by_structure(raw_structure_data);
    std::cerr << "stage=compare\n";
    std::cerr.flush();

    int mismatched_structure_count = 0;
    int mismatched_term_count = 0;
    for (int structure_index = 0;
         structure_index < raw_structure_data.n_structures;
         ++structure_index) {
      const auto& cpp_terms =
          cpp_terms_by_structure[xmvb::to_size(structure_index)];
      const auto& legacy_terms =
          legacy_terms_by_structure[xmvb::to_size(structure_index)];
      bool structure_mismatch = false;

      for (const auto& [key, cpp_coefficient] : cpp_terms) {
        const auto legacy_iterator = legacy_terms.find(key);
        const double legacy_coefficient =
            legacy_iterator == legacy_terms.end() ? 0.0 : legacy_iterator->second;
        if (std::abs(cpp_coefficient - legacy_coefficient) > 1.0e-12) {
          if (!structure_mismatch) {
            ++mismatched_structure_count;
            structure_mismatch = true;
          }
          ++mismatched_term_count;
          if (mismatched_term_count <= 12) {
            std::cout << "structure_mismatch structure=" << structure_index << '\n';
            print_term("cpp", key, cpp_coefficient);
            print_term("legacy", key, legacy_coefficient);
          }
        }
      }

      for (const auto& [key, legacy_coefficient] : legacy_terms) {
        if (cpp_terms.find(key) != cpp_terms.end()) {
          continue;
        }
        if (!structure_mismatch) {
          ++mismatched_structure_count;
          structure_mismatch = true;
        }
        ++mismatched_term_count;
        if (mismatched_term_count <= 12) {
          std::cout << "structure_mismatch structure=" << structure_index << '\n';
          print_term("cpp", key, 0.0);
          print_term("legacy", key, legacy_coefficient);
        }
      }
    }

    std::cout << std::setprecision(15);
    std::cout << "n_structures = " << raw_structure_data.n_structures << '\n';
    std::cout << "n_total_electrons = " << raw_structure_data.n_total_electrons << '\n';
    std::cout << "n_active_electrons = " << raw_structure_data.n_active_electrons << '\n';
    std::cout << "n_active_orbitals = "
              << load_result.input.orbital_preparation_input.n_active_orbitals
              << '\n';
    std::cout << "spin_multiplicity = " << raw_structure_data.spin_multiplicity << '\n';
    std::cout << "wavefunction_type = " << raw_structure_data.wavefunction_type << '\n';
    std::cout << "vb_function_type = " << raw_structure_data.vb_function_type << '\n';
    std::cout << "derived_n_inactive_doubly_occupied_orbitals = "
              << n_inactive_doubly_occupied_orbitals << '\n';
    std::cout << "legacy_detect_ndb = " << detected_ndb << '\n';
    std::cout << "n_unique_active_labels = " << active_labels.size() << '\n';
    std::cout << "active_labels =";
    for (const int label : active_labels) {
      std::cout << ' ' << label;
    }
    std::cout << '\n';
    std::cout << "expanded_unique_determinants = "
              << expanded_data.alpha_det.size() << '\n';
    std::cout << "mismatched_structure_count = "
              << mismatched_structure_count << '\n';
    std::cout << "mismatched_term_count = "
              << mismatched_term_count << '\n';
    return mismatched_term_count == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
