#include "runtime/cpp_vb_input_loader.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "runtime/cpp_block_guess_builder.hpp"
#include "runtime/generated_raw_structure_builder.hpp"
#include "runtime/input_deck_keywords.hpp"
#include "runtime/input_deck_model.hpp"
#include "runtime/input_deck_orbital_support_builder.hpp"
#include "runtime/input_deck_primary_basis_builder.hpp"
#include "runtime/libcint_auxiliary_basis_builder.hpp"
#include "runtime/libcint_compat.hpp"
#include "runtime/libcint_direct_shell_evaluator.hpp"
#include "runtime/libcint_materialized_integral_provider.hpp"
#include "runtime/materialized_ao_integral_input_builder.hpp"
#include "vb/matrices/full_structure_expander.hpp"
#include "vb/matrices/raw_structure_subspace_selector.hpp"
#include "vb/orbital/libcint_input_utils.hpp"
#include "vb/orbital/sparse_orbital_parameter_view.hpp"
#include "vb/orbital/support_aware_mo_gauge_fix.hpp"

#include <Eigen/Core>

namespace xmvb::vb {

namespace {

bool load_progress_logging_enabled() {
  const char* value = std::getenv("XMVB_CPP_LOG_LOAD_PROGRESS");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool ao_effective_one_electron_graph_enabled() {
  const char* disable_flag = std::getenv("XMVB_CPP_DISABLE_AO_H1E_GRAPH");
  return disable_flag == nullptr ||
      disable_flag[0] == '\0' ||
      disable_flag[0] == '0';
}

void log_load_stage(
    const char* stage_name,
    double seconds) {
  if (!load_progress_logging_enabled()) {
    return;
  }
  std::fprintf(stderr, "load_stage %s %.12f\n", stage_name, seconds);
  std::fflush(stderr);
}

constexpr int kLibcintBasSlots = 8;
constexpr int kLibcintPtrExpSlot = 5;
constexpr int kLibcintPtrCoeffSlot = 6;
constexpr double kBohrPerAngstrom = 1.0 / 0.529177249;

struct StaticMoleculeTopology {
  int n_atoms = 0;
  int n_shells = 0;
  int n_basis_functions = 0;
  std::vector<int> shell_to_atom;
  std::vector<int> shell_angular_momenta;
  std::vector<int> shell_n_primitives;
  std::vector<int> shell_ao_starts;
  std::vector<int> shell_ao_counts;
  std::vector<int> ao_to_atom;
  std::vector<int> ao_to_shell;
  std::vector<int> ao_angular_momenta;
  std::vector<int> ao_shell_local_indices;
  std::vector<int> ao_cartesian_exponents;
};

Eigen::MatrixXd build_full_ao_overlap_matrix(
    const LibcintInput& libcint_input) {
  LibcintDirectShellEvaluator evaluator(libcint_input);
  const int n_basis_functions = evaluator.n_basis_functions();
  Eigen::MatrixXd overlap_matrix =
      Eigen::MatrixXd::Zero(n_basis_functions, n_basis_functions);

  // Assemble the full AO overlap metric shell-by-shell in the repository's
  // column-major dense layout, then flatten only when filling packed runtime
  // fields that still store column-major buffers.
  for (int right_shell = 0; right_shell < libcint_input.n_shells; ++right_shell) {
    for (int left_shell = 0; left_shell <= right_shell; ++left_shell) {
      const LibcintShellBlock shell_block =
          evaluator.evaluate_overlap_shell_pair(left_shell, right_shell);
      for (int local_column = 0;
           local_column < shell_block.right_ao_count;
           ++local_column) {
        const int global_column = shell_block.right_ao_offset + local_column;
        for (int local_row = 0;
             local_row < shell_block.left_ao_count;
             ++local_row) {
          const int global_row = shell_block.left_ao_offset + local_row;
          const double value = shell_block.values[
              local_row +
              local_column * shell_block.left_ao_count];
          overlap_matrix(global_row, global_column) = value;
          overlap_matrix(global_column, global_row) = value;
        }
      }
    }
  }
  return overlap_matrix;
}

Eigen::MatrixXd build_full_ao_core_hamiltonian_matrix(
    const LibcintInput& libcint_input) {
  LibcintDirectShellEvaluator evaluator(libcint_input);
  const int n_basis_functions = evaluator.n_basis_functions();
  Eigen::MatrixXd core_hamiltonian_matrix =
      Eigen::MatrixXd::Zero(n_basis_functions, n_basis_functions);

  // Materialize the AO one-electron Hamiltonian directly from libcint so the
  // standalone loader can build RHF guesses and RI inputs from the same pure
  // C++ integral source.
  for (int right_shell = 0; right_shell < libcint_input.n_shells; ++right_shell) {
    for (int left_shell = 0; left_shell <= right_shell; ++left_shell) {
      const LibcintShellBlock shell_block =
          evaluator.evaluate_core_hamiltonian_shell_pair(
              left_shell,
              right_shell);
      for (int local_column = 0;
           local_column < shell_block.right_ao_count;
           ++local_column) {
        const int global_column = shell_block.right_ao_offset + local_column;
        for (int local_row = 0;
             local_row < shell_block.left_ao_count;
             ++local_row) {
          const int global_row = shell_block.left_ao_offset + local_row;
          const double value = shell_block.values[
              local_row +
              local_column * shell_block.left_ao_count];
          core_hamiltonian_matrix(global_row, global_column) = value;
          core_hamiltonian_matrix(global_column, global_row) = value;
        }
      }
    }
  }
  return core_hamiltonian_matrix;
}

std::string trim_ascii_whitespace(std::string value) {
  const auto first = std::find_if_not(
      value.begin(),
      value.end(),
      [](unsigned char c) { return std::isspace(c) != 0; });
  const auto last = std::find_if_not(
      value.rbegin(),
      value.rend(),
      [](unsigned char c) { return std::isspace(c) != 0; })
                        .base();
  if (first >= last) {
    return {};
  }
  return std::string(first, last);
}

std::string to_ascii_lower(std::string value) {
  std::transform(
      value.begin(),
      value.end(),
      value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::string canonicalize_element_symbol(std::string symbol) {
  if (symbol.empty()) {
    return {};
  }
  symbol = trim_ascii_whitespace(std::move(symbol));
  if (symbol.empty()) {
    return {};
  }
  symbol[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(symbol[0])));
  for (std::size_t character_index = 1;
       character_index < symbol.size();
       ++character_index) {
    symbol[character_index] = static_cast<char>(
        std::tolower(static_cast<unsigned char>(symbol[character_index])));
  }
  return symbol;
}

int atomic_number_from_element_symbol(const std::string& element_symbol) {
  static const char* const kElementSymbols[] = {
      "", "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
      "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc",
      "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
      "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc",
      "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe",
      "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
      "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os",
      "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"};
  constexpr int kElementSymbolCount =
      static_cast<int>(sizeof(kElementSymbols) / sizeof(kElementSymbols[0]));
  const std::string canonical_symbol = canonicalize_element_symbol(element_symbol);
  for (int atomic_number = 1;
       atomic_number < kElementSymbolCount;
       ++atomic_number) {
    if (canonical_symbol == kElementSymbols[atomic_number]) {
      return atomic_number;
    }
  }
  throw std::runtime_error("unsupported element symbol in input deck geometry: " + element_symbol);
}

std::vector<int> build_atomic_numbers_from_geometry(
    const InputDeck& input_deck) {
  std::vector<int> atomic_numbers;
  atomic_numbers.reserve(input_deck.geometry_atoms.size());
  for (const InputDeckGeometryAtom& atom : input_deck.geometry_atoms) {
    atomic_numbers.push_back(atomic_number_from_element_symbol(atom.element_symbol));
  }
  return atomic_numbers;
}

std::vector<double> build_atomic_coordinates_bohr_from_geometry(
    const InputDeck& input_deck) {
  std::vector<double> coordinates_bohr;
  coordinates_bohr.reserve(input_deck.geometry_atoms.size() * 3);
  const double coordinate_scale =
      input_deck.metadata.geometry_coordinates_in_bohr ? 1.0 : kBohrPerAngstrom;
  for (const InputDeckGeometryAtom& atom : input_deck.geometry_atoms) {
    for (double coordinate : atom.coordinates) {
      coordinates_bohr.push_back(coordinate * coordinate_scale);
    }
  }
  return coordinates_bohr;
}

double compute_nuclear_repulsion_energy_from_geometry(
    const std::vector<int>& atomic_numbers,
    const std::vector<double>& atomic_coordinates_bohr) {
  const std::size_t n_atoms = atomic_numbers.size();
  if (atomic_coordinates_bohr.size() != n_atoms * 3) {
    throw std::invalid_argument(
        "atomic coordinate array must contain exactly 3 values per atom");
  }

  double nuclear_repulsion_energy = 0.0;
  for (std::size_t left_atom = 0; left_atom < n_atoms; ++left_atom) {
    const double left_x = atomic_coordinates_bohr[left_atom * 3];
    const double left_y = atomic_coordinates_bohr[left_atom * 3 + 1];
    const double left_z = atomic_coordinates_bohr[left_atom * 3 + 2];
    for (std::size_t right_atom = left_atom + 1; right_atom < n_atoms; ++right_atom) {
      const double dx = left_x - atomic_coordinates_bohr[right_atom * 3];
      const double dy = left_y - atomic_coordinates_bohr[right_atom * 3 + 1];
      const double dz = left_z - atomic_coordinates_bohr[right_atom * 3 + 2];
      const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (!(distance > 0.0)) {
        throw std::runtime_error("input deck geometry contains overlapping nuclei");
      }
      nuclear_repulsion_energy +=
          static_cast<double>(atomic_numbers[left_atom]) *
          static_cast<double>(atomic_numbers[right_atom]) / distance;
    }
  }
  return nuclear_repulsion_energy;
}

int resolve_total_electron_count(const InputDeck& input_deck) {
  if (input_deck.has_explicit_raw_structures &&
      input_deck.explicit_raw_structures.n_total_electrons > 0) {
    return input_deck.explicit_raw_structures.n_total_electrons;
  }
  if (!input_deck.geometry_atoms.empty()) {
    const std::vector<int> atomic_numbers =
        build_atomic_numbers_from_geometry(input_deck);
    int total_electron_count = -input_deck.metadata.total_charge;
    for (int atomic_number : atomic_numbers) {
      total_electron_count += atomic_number;
    }
    if (total_electron_count > 0) {
      return total_electron_count;
    }
  }
  throw std::runtime_error(
      "failed to resolve the total electron count from the pure C++ input deck");
}

int resolve_active_orbital_count(const InputDeckMetadata& input_deck_metadata) {
  if (input_deck_metadata.declared_active_orbitals > 0) {
    return input_deck_metadata.declared_active_orbitals;
  }
  throw std::runtime_error("pure C++ standalone input loader requires NAO");
}

int resolve_active_electron_count(
    const InputDeck& input_deck,
    const InputDeckMetadata& input_deck_metadata) {
  if (input_deck_metadata.declared_active_electrons > 0) {
    return input_deck_metadata.declared_active_electrons;
  }
  if (input_deck.has_explicit_raw_structures &&
      input_deck.explicit_raw_structures.n_active_electrons > 0) {
    return input_deck.explicit_raw_structures.n_active_electrons;
  }
  throw std::runtime_error("pure C++ standalone input loader requires NAE");
}

int resolve_spin_multiplicity(
    const InputDeck& input_deck,
    const InputDeckMetadata& input_deck_metadata) {
  if (input_deck_metadata.declared_spin_multiplicity > 0) {
    return input_deck_metadata.declared_spin_multiplicity;
  }
  if (input_deck.has_explicit_raw_structures &&
      input_deck.explicit_raw_structures.spin_multiplicity > 0) {
    return input_deck.explicit_raw_structures.spin_multiplicity;
  }
  throw std::runtime_error("pure C++ standalone input loader requires NMUL");
}

int infer_total_orbital_count_from_raw_structures(
    const RawStructureData& raw_structure_data) {
  int max_orbital_label = 0;
  for (int orbital_label : raw_structure_data.raw_structure_orbitals) {
    max_orbital_label = std::max(max_orbital_label, orbital_label);
  }
  if (max_orbital_label <= 0) {
    throw std::runtime_error("failed to infer the total orbital count from raw structures");
  }
  return max_orbital_label;
}

void populate_detected_orbital_blocks(
    OrbitalPreparationInput* orbital_preparation_input) {
  if (orbital_preparation_input == nullptr) {
    throw std::invalid_argument("orbital_preparation_input must not be null");
  }

  const auto blocks = detect_orbital_blocks(*orbital_preparation_input);
  orbital_preparation_input->n_blocks = blocks.size();
  orbital_preparation_input->block_storage_dimension = 0;
  for (const auto& block : blocks) {
    orbital_preparation_input->block_storage_dimension = std::max(
        orbital_preparation_input->block_storage_dimension,
        block.size());
  }
  orbital_preparation_input->block_partial_overlap = 0;
  const std::size_t block_storage_stride =
      std::max<std::size_t>(1, orbital_preparation_input->block_storage_dimension);
  orbital_preparation_input->block_members.assign(
      orbital_preparation_input->n_blocks * block_storage_stride,
      0);
  orbital_preparation_input->block_orbital_counts.assign(
      orbital_preparation_input->n_blocks,
      0);
  orbital_preparation_input->block_basis_counts.clear();
  orbital_preparation_input->block_basis_counts.reserve(blocks.size());

  for (std::size_t block_index = 0;
       block_index < orbital_preparation_input->n_blocks;
       ++block_index) {
    const auto& block = blocks[block_index];
    orbital_preparation_input->block_orbital_counts[block_index] =
        static_cast<int>(block.size());
    for (std::size_t orbital_offset = 0;
         orbital_offset < block.size();
         ++orbital_offset) {
      orbital_preparation_input->block_members[
          block_index * block_storage_stride +
          orbital_offset] = block[orbital_offset];
    }
    orbital_preparation_input->block_basis_counts.push_back(
        block.empty()
            ? 0
            : stored_sparse_orbital_coefficient_count(
                  *orbital_preparation_input,
                  block.front()));
  }
}

std::optional<StandardTwoElectronMode> sniff_standard_two_electron_mode_from_input(
    const std::string& input_file_path) {
  std::ifstream input_stream(input_file_path);
  if (!input_stream) {
    return std::nullopt;
  }

  bool inside_ctrl_block = false;
  std::string line;
  while (std::getline(input_stream, line)) {
    const std::size_t comment_pos = line.find('#');
    if (comment_pos != std::string::npos) {
      line.erase(comment_pos);
    }
    std::string normalized_line = to_ascii_lower(trim_ascii_whitespace(std::move(line)));
    if (normalized_line.empty()) {
      continue;
    }
    if (!inside_ctrl_block) {
      if (normalized_line == "$ctrl") {
        inside_ctrl_block = true;
      }
      continue;
    }
    if (normalized_line == "$end") {
      break;
    }
    if (normalized_line.rfind("int", 0) != 0) {
      continue;
    }
    std::size_t position = 3;
    while (position < normalized_line.size() &&
           std::isspace(static_cast<unsigned char>(normalized_line[position])) != 0) {
      ++position;
    }
    if (position >= normalized_line.size() || normalized_line[position] != '=') {
      continue;
    }
    ++position;
    while (position < normalized_line.size() &&
           std::isspace(static_cast<unsigned char>(normalized_line[position])) != 0) {
      ++position;
    }
    const std::string value =
        trim_ascii_whitespace(normalized_line.substr(position));
    if (value == "ri") {
      return StandardTwoElectronMode::ResolutionOfIdentity;
    }
    if (value == "libcint") {
      return StandardTwoElectronMode::Exact;
    }
  }

  return std::nullopt;
}

StaticMoleculeTopology build_static_molecule_topology(
    const LibcintInput& libcint_input) {
  StaticMoleculeTopology topology;
  validate_libcint_input_shape(libcint_input);
  topology.n_atoms = libcint_input.n_atoms;
  topology.n_shells = libcint_input.n_shells;
  topology.n_basis_functions = infer_n_basis_functions(libcint_input);
  topology.shell_to_atom.reserve(topology.n_shells);
  topology.shell_angular_momenta.reserve(topology.n_shells);
  topology.shell_n_primitives.reserve(topology.n_shells);
  topology.shell_ao_starts.reserve(topology.n_shells);
  topology.shell_ao_counts.reserve(topology.n_shells);
  topology.ao_to_atom.assign(topology.n_basis_functions, -1);
  topology.ao_to_shell.assign(topology.n_basis_functions, -1);
  topology.ao_angular_momenta.assign(topology.n_basis_functions, 0);
  topology.ao_shell_local_indices.assign(topology.n_basis_functions, 0);
  topology.ao_cartesian_exponents.assign(
      topology.n_basis_functions * 3,
      0);

  for (int shell_index = 0; shell_index < topology.n_shells; ++shell_index) {
    const std::size_t shell_offset = shell_index * kLibcintBasSlots;
    const int atom_index = libcint_input.bas[shell_offset + ATOM_OF];
    const int angular_momentum = libcint_input.bas[shell_offset + ANG_OF];
    const int n_primitives = libcint_input.bas[shell_offset + NPRIM_OF];
    const int ao_start = shell_ao_offset(libcint_input, shell_index);
    const int ao_count = shell_ao_count(libcint_input, shell_index);
    topology.shell_to_atom.push_back(atom_index);
    topology.shell_angular_momenta.push_back(angular_momentum);
    topology.shell_n_primitives.push_back(n_primitives);
    topology.shell_ao_starts.push_back(ao_start);
    topology.shell_ao_counts.push_back(ao_count);

    int local_ao_index = 0;
    for (int lx = angular_momentum; lx >= 0; --lx) {
      for (int ly = angular_momentum - lx; ly >= 0; --ly) {
        const int lz = angular_momentum - lx - ly;
        const int ao_index = ao_start + local_ao_index;
        topology.ao_to_atom[ao_index] = atom_index;
        topology.ao_to_shell[ao_index] = shell_index;
        topology.ao_angular_momenta[ao_index] = angular_momentum;
        topology.ao_shell_local_indices[ao_index] = local_ao_index;
        const std::size_t exponent_offset = ao_index * 3;
        topology.ao_cartesian_exponents[exponent_offset] = lx;
        topology.ao_cartesian_exponents[exponent_offset + 1] = ly;
        topology.ao_cartesian_exponents[exponent_offset + 2] = lz;
        ++local_ao_index;
      }
    }
  }

  return topology;
}

}  // namespace

const char* ao_integral_source_name(AoIntegralSource source) {
  switch (source) {
    case AoIntegralSource::Auto:
      return "auto";
    case AoIntegralSource::LibcintMaterializedCpp:
      return "libcint_cpp";
    case AoIntegralSource::RuntimeCoreHamiltonianOnly:
      return "runtime_hcore";
  }
  return "unknown";
}

const char* standard_two_electron_mode_name(StandardTwoElectronMode mode) {
  switch (mode) {
    case StandardTwoElectronMode::Auto:
      return "auto";
    case StandardTwoElectronMode::Exact:
      return "exact";
    case StandardTwoElectronMode::ResolutionOfIdentity:
      return "ri";
  }
  return "unknown";
}

namespace {

bool should_use_standard_ri_two_electron_mode(
    bool input_requests_ri_two_electron_mode,
    const CppVbInputLoadOptions& options) {
  switch (options.standard_two_electron_mode) {
    case StandardTwoElectronMode::Exact:
      return false;
    case StandardTwoElectronMode::ResolutionOfIdentity:
      return true;
    case StandardTwoElectronMode::Auto:
      break;
  }

  // In auto mode, follow the input deck semantics rather than switching on a
  // size heuristic. `INT=LIBCINT` requests exact AO integrals, while `INT=RI`
  // requests the standard RI path. The legacy parser already handles
  // case-insensitive keywords before populating this runtime flag.
  return input_requests_ri_two_electron_mode;
}

int configured_openmp_thread_count();

AoIntegralSource resolve_ao_integral_source(
    bool use_standard_ri_two_electron_mode,
    const CppVbInputLoadOptions& options) {
  switch (options.ao_integral_source) {
    case AoIntegralSource::Auto:
      return use_standard_ri_two_electron_mode
          ? AoIntegralSource::RuntimeCoreHamiltonianOnly
          : AoIntegralSource::LibcintMaterializedCpp;
    case AoIntegralSource::LibcintMaterializedCpp:
    case AoIntegralSource::RuntimeCoreHamiltonianOnly:
      return options.ao_integral_source;
  }
  throw std::invalid_argument("invalid AO integral source");
}

std::size_t active_pair_count(int n_active_orbitals) {
  if (n_active_orbitals <= 0) {
    return 0;
  }
  return n_active_orbitals *
      (n_active_orbitals + 1) / 2;
}

int configured_openmp_thread_count() {
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  return n_threads;
}

}  // namespace

CppVbInputLoadResult load_cpp_vb_input_with_timings(
    const std::string& input_file_path,
    const CppVbInputLoadOptions& options) {
  const auto total_start_time = std::chrono::steady_clock::now();
  CppVbInputLoadResult load_result;
  CppVbInput result;
  const InputDeck input_deck =
      parse_input_deck_model(input_file_path);
  const InputDeckMetadata& input_deck_metadata = input_deck.metadata;
  const InputDeckPrimaryBasisBuildResult primary_basis_build_result =
      build_input_deck_primary_basis(input_deck);
  const LibcintInput& primary_libcint_input =
      primary_basis_build_result.libcint_input;
  const StaticMoleculeTopology static_topology =
      build_static_molecule_topology(primary_libcint_input);
  const Eigen::MatrixXd active_ao_overlap_matrix =
      build_full_ao_overlap_matrix(primary_libcint_input);
  const bool use_standard_ri_two_electron_mode =
      should_use_standard_ri_two_electron_mode(
          input_deck_metadata.request_ri_two_electron_mode,
          options);
  const AoIntegralSource resolved_ao_integral_source =
      resolve_ao_integral_source(
          use_standard_ri_two_electron_mode,
          options);
  RuntimeExtractionTimings runtime_timings;

  const int resolved_n_active_orbitals =
      resolve_active_orbital_count(input_deck_metadata);
  const int resolved_n_active_electrons =
      resolve_active_electron_count(
          input_deck,
          input_deck_metadata);
  const int resolved_spin_multiplicity =
      resolve_spin_multiplicity(
          input_deck,
          input_deck_metadata);
  const int resolved_n_total_electrons = resolve_total_electron_count(input_deck);
  RawStructureData raw_structure_data;
  if (input_deck.has_explicit_raw_structures) {
    load_result.raw_structure_source = RawStructureSource::ExplicitInputBlock;
    raw_structure_data = input_deck.explicit_raw_structures;
    if (raw_structure_data.n_total_electrons <= 0) {
      raw_structure_data.n_total_electrons = resolved_n_total_electrons;
    }
    if (raw_structure_data.n_active_electrons <= 0) {
      raw_structure_data.n_active_electrons = resolved_n_active_electrons;
    }
    if (raw_structure_data.spin_multiplicity <= 0) {
      raw_structure_data.spin_multiplicity = resolved_spin_multiplicity;
    }
  } else if (can_build_generated_raw_structures_from_structure_class(
                 input_deck_metadata,
                 resolved_n_total_electrons)) {
    load_result.raw_structure_source = RawStructureSource::GeneratedFromStructureClassCpp;
    raw_structure_data =
        build_generated_raw_structures_from_structure_class(
            input_deck_metadata,
            resolved_n_total_electrons);
  } else {
    throw std::runtime_error(
        "standalone raw-structure generation requires either an explicit $STR block "
        "or a supported pure C++ STR=... structure class");
  }
  const int resolved_n_orbitals =
      infer_total_orbital_count_from_raw_structures(raw_structure_data);
  const int resolved_n_atoms =
      static_cast<int>(input_deck.geometry_atoms.size());
  const std::vector<int> atomic_numbers =
      build_atomic_numbers_from_geometry(input_deck);
  const std::vector<double> atomic_coordinates_bohr =
      build_atomic_coordinates_bohr_from_geometry(input_deck);
  const double nuclear_repulsion_energy =
      compute_nuclear_repulsion_energy_from_geometry(
          atomic_numbers,
          atomic_coordinates_bohr);
  load_result.standard_two_electron_mode =
      use_standard_ri_two_electron_mode
          ? StandardTwoElectronMode::ResolutionOfIdentity
          : StandardTwoElectronMode::Exact;
  load_result.requested_scf_max_iterations =
      input_deck_metadata.requested_scf_max_iterations > 0
          ? input_deck_metadata.requested_scf_max_iterations
          : 2000;
  load_result.request_molden_output = input_deck_metadata.request_molden_output;
  result.standard_two_electron_mode = load_result.standard_two_electron_mode;
  load_result.ao_integral_source = resolved_ao_integral_source;

  result.orbital_preparation_input.n_basis_functions = static_topology.n_basis_functions;
  result.orbital_preparation_input.n_orbitals = resolved_n_orbitals;
  result.orbital_preparation_input.n_active_orbitals = resolved_n_active_orbitals;
  result.orbital_preparation_input.n_total_electrons = resolved_n_total_electrons;
  result.orbital_preparation_input.n_active_electrons = resolved_n_active_electrons;
  result.orbital_preparation_input.spin_multiplicity = resolved_spin_multiplicity;
  result.orbital_preparation_input.orbital_type = input_deck_metadata.orbital_type;
  result.orbital_preparation_input.orbital_value_table.assign(
      static_topology.n_basis_functions *
          resolved_n_orbitals,
      0.0);
  if (!options.skip_orbital_guess &&
      options.orbital_guess_source != OrbitalGuessSource::Cpp) {
    throw std::runtime_error(
        "only the pure C++ orbital guess path is supported");
  }
  InputDeckOrbitalSupportBuildInput support_build_input;
  support_build_input.n_atoms = resolved_n_atoms;
  support_build_input.n_basis_functions = static_topology.n_basis_functions;
  support_build_input.n_orbitals = resolved_n_orbitals;
  support_build_input.n_active_orbitals = resolved_n_active_orbitals;
  support_build_input.orbital_type = input_deck_metadata.orbital_type;
  support_build_input.fragment_type = input_deck_metadata.fragment_type;
  support_build_input.ao_to_atom = static_topology.ao_to_atom;
  support_build_input.ao_cartesian_exponents =
      static_topology.ao_cartesian_exponents;
  if (!can_build_input_deck_orbital_support_chart(input_deck, support_build_input)) {
    throw std::runtime_error(
        "pure C++ standalone input loader could not rebuild the orbital support chart");
  }
  const InputDeckOrbitalSupportChart support_chart =
      build_input_deck_orbital_support_chart(
          input_deck,
          support_build_input);
  result.orbital_preparation_input.orbital_basis_index_table =
      support_chart.orbital_basis_index_table;
  result.orbital_preparation_input.orbital_basis_counts =
      support_chart.orbital_basis_counts;
  result.orbital_preparation_input.original_orbital_basis_counts =
      support_chart.original_orbital_basis_counts;
  result.orbital_preparation_input.ao_overlap_matrix =
      active_ao_overlap_matrix;
  result.orbital_preparation_input.ao_normalization =
      build_ao_normalization(result.orbital_preparation_input);
  populate_detected_orbital_blocks(&result.orbital_preparation_input);
  // Keep the orbital support chart in XMVB's historical deck semantics. In
  // particular, `orbtyp=oeo` remains a full-AO chart for every orbital; the
  // C++ loader must not reinterpret `$ORB/$ACTORB` into a different sparse
  // active-space manifold.
  enforce_strict_sparse_orbital_support(&result.orbital_preparation_input);
  const OrbitalPreparationInput original_orbital_preparation_input =
      result.orbital_preparation_input;
  if (input_deck_metadata.guess_type == kGuessTypeMo) {
    result.orbital_preparation_input.mo_gauge_reference_orbital_basis_counts =
        original_orbital_preparation_input.orbital_basis_counts;
    result.orbital_preparation_input.mo_gauge_reference_orbital_basis_index_table =
        original_orbital_preparation_input.orbital_basis_index_table;
  }
  result.libcint_input = primary_libcint_input;
  if (use_standard_ri_two_electron_mode) {
    LibcintAuxiliaryBasisBuilder auxiliary_basis_builder;
    result.auxiliary_libcint_input =
        auxiliary_basis_builder.build(primary_libcint_input);
  }

  const auto ao_integral_provider_start_time = std::chrono::steady_clock::now();
  MaterializedAoIntegralBuffers ao_integral_buffers;
  if (resolved_ao_integral_source == AoIntegralSource::LibcintMaterializedCpp) {
    LibcintMaterializedIntegralProvider provider;
    ao_integral_buffers = provider.build(result.libcint_input);
  } else if (resolved_ao_integral_source != AoIntegralSource::RuntimeCoreHamiltonianOnly) {
    throw std::invalid_argument("invalid AO integral source");
  }
  load_result.ao_integral_provider_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - ao_integral_provider_start_time)
          .count();
  log_load_stage("ao_integral_provider", load_result.ao_integral_provider_seconds);
  const auto ao_integral_input_build_start_time = std::chrono::steady_clock::now();
  if (resolved_ao_integral_source == AoIntegralSource::RuntimeCoreHamiltonianOnly) {
    const Eigen::MatrixXd core_hamiltonian_matrix =
        build_full_ao_core_hamiltonian_matrix(result.libcint_input);
    result.ao_integral_input = build_core_hamiltonian_only_ao_integral_input(
        static_topology.n_basis_functions,
        core_hamiltonian_matrix);
  } else {
    MaterializedAoIntegralInputBuildOptions ao_input_build_options;
    ao_input_build_options.build_pair_graph =
        active_pair_count(result.orbital_preparation_input.n_active_orbitals) <= 64 &&
        configured_openmp_thread_count() > 1;
    ao_input_build_options.build_pair_indices =
        !ao_input_build_options.build_pair_graph;
    ao_input_build_options.build_ao_effective_one_electron_graph =
        options.build_ao_effective_one_electron_graph &&
        ao_effective_one_electron_graph_enabled();
    result.ao_integral_input = build_materialized_ao_integral_input(
        std::move(ao_integral_buffers),
        ao_input_build_options);
  }
  load_result.ao_integral_input_build_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - ao_integral_input_build_start_time)
          .count();
  log_load_stage("ao_integral_input_build", load_result.ao_integral_input_build_seconds);

  if (!options.skip_orbital_guess &&
      options.orbital_guess_source == OrbitalGuessSource::Cpp) {
    const auto orbital_guess_start_time = std::chrono::steady_clock::now();
    build_cpp_initial_guess(
        input_file_path,
        input_deck_metadata.guess_type,
        result.libcint_input,
        result.ao_integral_input,
        input_deck.guess_block.present ? &input_deck.guess_block.raw_lines : nullptr,
        &result.orbital_preparation_input);
    if (input_deck_metadata.guess_type == kGuessTypeMo) {
      // Keep `GUESS=MO` on the original sparse support chart from the deck.
      // Expanding inactive supports here changes the variational manifold and
      // shifts the converged energy away from the historical `.xmo`
      // reference. The support-aware gauge fix below is therefore only
      // allowed to act when some other upstream path has already changed that
      // recorded chart.
      apply_support_aware_inactive_mo_gauge_fix(
          &result.orbital_preparation_input);
    }
    load_result.orbital_guess_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - orbital_guess_start_time)
            .count();
    log_load_stage("orbital_guess", load_result.orbital_guess_seconds);
  }

  const auto raw_structure_selection_start_time = std::chrono::steady_clock::now();
  const int source_raw_structure_count = raw_structure_data.n_structures;
  const auto selected_raw_structure_indices =
      select_raw_structure_indices(raw_structure_data, options.raw_structure_selection);
  RawStructureData selected_raw_structure_data;
  if (static_cast<int>(selected_raw_structure_indices.size()) ==
      raw_structure_data.n_structures) {
    selected_raw_structure_data = std::move(raw_structure_data);
  } else {
    selected_raw_structure_data = build_raw_structure_subset(
        raw_structure_data,
        selected_raw_structure_indices);
  }
  load_result.raw_structure_selection_seconds =
      std::chrono::duration<double>(
          std::chrono::steady_clock::now() - raw_structure_selection_start_time)
          .count();
  log_load_stage("raw_structure_selection", load_result.raw_structure_selection_seconds);

  CppVbStaticMoleculeMetadata static_molecule_metadata;
  static_molecule_metadata.n_atoms = resolved_n_atoms;
  static_molecule_metadata.n_shells = static_topology.n_shells;
  static_molecule_metadata.atomic_numbers = atomic_numbers;
  static_molecule_metadata.atomic_coordinates = atomic_coordinates_bohr;
  static_molecule_metadata.shell_to_atom = static_topology.shell_to_atom;
  static_molecule_metadata.shell_angular_momenta =
      static_topology.shell_angular_momenta;
  static_molecule_metadata.shell_n_primitives =
      static_topology.shell_n_primitives;
  static_molecule_metadata.shell_ao_starts = static_topology.shell_ao_starts;
  static_molecule_metadata.shell_ao_counts = static_topology.shell_ao_counts;
  static_molecule_metadata.ao_to_atom = static_topology.ao_to_atom;
  static_molecule_metadata.ao_to_shell = static_topology.ao_to_shell;
  static_molecule_metadata.ao_angular_momenta =
      static_topology.ao_angular_momenta;
  static_molecule_metadata.ao_shell_local_indices =
      static_topology.ao_shell_local_indices;
  static_molecule_metadata.ao_cartesian_exponents =
      static_topology.ao_cartesian_exponents;

  if (options.expand_selected_raw_structures) {
    const auto structure_expansion_start_time = std::chrono::steady_clock::now();
    FullDeterminantStructureExpander expander;
    result.structure_data = expander.expand(selected_raw_structure_data);
    load_result.structure_expansion_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - structure_expansion_start_time)
            .count();
    log_load_stage("structure_expansion", load_result.structure_expansion_seconds);
  }
  load_result.input = std::move(result);
  load_result.raw_structure_data = std::move(selected_raw_structure_data);
  load_result.static_molecule_metadata = std::move(static_molecule_metadata);
  load_result.runtime_timings = runtime_timings;
  load_result.basis_name = primary_basis_build_result.basis_file_path.string();
  load_result.orbital_guess_source = options.orbital_guess_source;
  load_result.raw_structure_selection = options.raw_structure_selection;
  load_result.source_raw_structure_count = source_raw_structure_count;
  load_result.nuclear_repulsion_energy = nuclear_repulsion_energy;
  load_result.total_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start_time).count();
  log_load_stage("load_total", load_result.total_seconds);
  return load_result;
}

CppVbInput load_cpp_vb_input(
    const std::string& input_file_path,
    const CppVbInputLoadOptions& options) {
  return load_cpp_vb_input_with_timings(input_file_path, options).input;
}

}  // namespace xmvb::vb
