#include "vb/model/deepvbh_jax_inference_runner.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#ifdef XMVB_CPP_ENABLE_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include "vb/matrices/two_electron_indexer.hpp"

namespace xmvb::vb {

namespace {

namespace fs = std::filesystem;

std::string shell_quote(const std::string& value) {
  std::string quoted = "'";
  for (const char character : value) {
    if (character == '\'') {
      quoted += "'\"'\"'";
    } else {
      quoted.push_back(character);
    }
  }
  quoted.push_back('\'');
  return quoted;
}

template <typename T>
void write_binary_buffer(
    const fs::path& path,
    const T* data,
    std::size_t count) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to open binary output file: " + path.string());
  }
  if (count > 0) {
    output.write(
        reinterpret_cast<const char*>(data),
        static_cast<std::streamsize>(sizeof(T) * count));
  }
  if (!output) {
    throw std::runtime_error("failed to write binary output file: " + path.string());
  }
}

template <typename Container>
void write_binary_container(
    const fs::path& path,
    const Container& values) {
  using ValueType = typename Container::value_type;
  write_binary_buffer<ValueType>(path, values.data(), values.size());
}

void write_text_file(
    const fs::path& path,
    const std::string& contents) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("failed to open text output file: " + path.string());
  }
  output << contents;
  if (!output) {
    throw std::runtime_error("failed to write text output file: " + path.string());
  }
}

template <typename T>
std::vector<T> read_binary_vector(
    const fs::path& path,
    std::size_t count) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open binary input file: " + path.string());
  }
  std::vector<T> values(count);
  if (count > 0) {
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(sizeof(T) * count));
  }
  if (!input) {
    throw std::runtime_error("failed to read binary input file: " + path.string());
  }
  return values;
}

double read_binary_scalar(const fs::path& path) {
  const auto values = read_binary_vector<double>(path, 1);
  return values.front();
}

template <typename T>
std::optional<std::vector<T>> read_binary_vector_if_exists(
    const fs::path& path,
    std::size_t count) {
  if (!fs::exists(path)) {
    return std::nullopt;
  }
  return read_binary_vector<T>(path, count);
}

std::optional<double> read_binary_scalar_if_exists(const fs::path& path) {
  if (!fs::exists(path)) {
    return std::nullopt;
  }
  return read_binary_scalar(path);
}

int get_sparse_coefficient_count(
    const OrbitalPreparationInput& orbital_preparation_input,
    int orbital_index) {
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  const int explicit_count =
      orbital_preparation_input.orbital_basis_counts[static_cast<std::size_t>(orbital_index)];
  if (explicit_count > 1) {
    return explicit_count;
  }

  int coefficient_count = 0;
  while (coefficient_count < n_basis_functions) {
    const int basis_function_index =
        orbital_preparation_input.orbital_basis_index_table
            [static_cast<std::size_t>(orbital_index) * n_basis_functions + coefficient_count];
    if (basis_function_index == 0) {
      break;
    }
    ++coefficient_count;
  }
  return coefficient_count;
}

std::vector<int> collect_differentiable_parameter_indices(
    const OrbitalPreparationInput& orbital_preparation_input) {
  std::vector<int> differentiable_parameter_indices;
  for (int orbital_index = 0; orbital_index < orbital_preparation_input.n_orbitals; ++orbital_index) {
    const int coefficient_count =
        get_sparse_coefficient_count(orbital_preparation_input, orbital_index);
    for (int coefficient_index = 0; coefficient_index < coefficient_count; ++coefficient_index) {
      differentiable_parameter_indices.push_back(
          orbital_index * orbital_preparation_input.n_basis_functions + coefficient_index);
    }
  }
  return differentiable_parameter_indices;
}

std::vector<double> sparse_to_dense_orbital_coefficients(
    const OrbitalPreparationInput& orbital_preparation_input,
    const std::vector<double>& sparse_coefficients) {
  const int n_orbitals = orbital_preparation_input.n_orbitals;
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  std::vector<double> dense_coefficients(
      static_cast<std::size_t>(n_orbitals) * static_cast<std::size_t>(n_basis_functions),
      0.0);
  for (int orbital_index = 0; orbital_index < n_orbitals; ++orbital_index) {
    const int coefficient_count =
        get_sparse_coefficient_count(orbital_preparation_input, orbital_index);
    for (int coefficient_index = 0; coefficient_index < coefficient_count; ++coefficient_index) {
      const int basis_function_index =
          orbital_preparation_input.orbital_basis_index_table
              [static_cast<std::size_t>(orbital_index) * n_basis_functions + coefficient_index] -
          1;
      if (basis_function_index < 0) {
        continue;
      }
      dense_coefficients[static_cast<std::size_t>(orbital_index) * n_basis_functions +
                         basis_function_index] =
          sparse_coefficients[static_cast<std::size_t>(orbital_index) * n_basis_functions +
                              coefficient_index];
    }
  }
  return dense_coefficients;
}

std::vector<double> dense_to_sparse_orbital_coefficients(
    const OrbitalPreparationInput& orbital_preparation_input,
    const std::vector<double>& dense_coefficients) {
  const int n_orbitals = orbital_preparation_input.n_orbitals;
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  std::vector<double> sparse_coefficients(
      static_cast<std::size_t>(n_orbitals) * static_cast<std::size_t>(n_basis_functions),
      0.0);
  for (int orbital_index = 0; orbital_index < n_orbitals; ++orbital_index) {
    const int coefficient_count =
        get_sparse_coefficient_count(orbital_preparation_input, orbital_index);
    for (int coefficient_index = 0; coefficient_index < coefficient_count; ++coefficient_index) {
      const int basis_function_index =
          orbital_preparation_input.orbital_basis_index_table
              [static_cast<std::size_t>(orbital_index) * n_basis_functions + coefficient_index] -
          1;
      if (basis_function_index < 0) {
        continue;
      }
      sparse_coefficients[static_cast<std::size_t>(orbital_index) * n_basis_functions +
                          coefficient_index] =
          dense_coefficients[static_cast<std::size_t>(orbital_index) * n_basis_functions +
                             basis_function_index];
    }
  }
  return sparse_coefficients;
}

double compute_one_electron_reference_energy(
    const std::vector<double>& inactive_density_matrix,
    const std::vector<double>& ao_effective_h1e,
    const std::vector<double>& ao_core_hamiltonian_matrix,
    int n_basis_functions) {
  double one_electron_reference_energy = 0.0;
  for (int column = 0; column < n_basis_functions; ++column) {
    for (int row = 0; row < n_basis_functions; ++row) {
      const std::size_t index =
          static_cast<std::size_t>(column) * n_basis_functions + row;
      one_electron_reference_energy +=
          inactive_density_matrix[index] *
          (ao_effective_h1e[index] + ao_core_hamiltonian_matrix[index]);
    }
  }
  return one_electron_reference_energy;
}

std::size_t packed_active_two_electron_size(int n_active_orbitals) {
  if (n_active_orbitals <= 0) {
    throw std::invalid_argument("n_active_orbitals must be positive");
  }
  return static_cast<std::size_t>(
             TwoElectronIndexer::two_electron_storage_index(
                 n_active_orbitals - 1,
                 n_active_orbitals - 1,
                 n_active_orbitals - 1,
                 n_active_orbitals - 1)) +
      1;
}

fs::path make_unique_work_directory(const fs::path& parent_directory) {
  fs::create_directories(parent_directory);
  const auto timestamp =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path candidate =
      parent_directory / ("deepvbh_cpp_inference_" + std::to_string(timestamp));
  fs::create_directories(candidate);
  return candidate;
}

fs::path resolve_relative_to(
    const fs::path& base_directory,
    const fs::path& path) {
  if (path.is_absolute()) {
    return path;
  }
  return fs::absolute(base_directory / path);
}

class TemporaryDirectoryGuard {
public:
  TemporaryDirectoryGuard(fs::path path, bool remove_on_destruction)
      : path_(std::move(path)),
        remove_on_destruction_(remove_on_destruction) {}

  ~TemporaryDirectoryGuard() {
    if (!remove_on_destruction_ || path_.empty()) {
      return;
    }
    std::error_code error;
    fs::remove_all(path_, error);
  }

  void keep() {
    remove_on_destruction_ = false;
  }

private:
  fs::path path_;
  bool remove_on_destruction_ = false;
};

struct PreparedInferenceInput {
  StructureAccumulationResult structure_matrices;
  double one_electron_reference_energy = 0.0;
};

struct StructurePairTopology {
  int n_active_beta_electrons = 0;
  int n_open_shell_electrons = 0;
  std::vector<int> structure_pair_orbital_indices;
  std::vector<std::uint8_t> structure_pair_mask;
  std::vector<int> structure_open_shell_orbitals;
  std::vector<std::uint8_t> structure_open_shell_mask;
};

struct PreparedOnnxFeatures {
  std::vector<double> structure_occupancy;
  std::vector<std::uint8_t> orbital_basis_mask;
  std::vector<std::uint8_t> orbital_shell_dense_mask;
  std::vector<double> dense_orbital_coefficients;
  StructurePairTopology pair_topology;
};

struct OnnxRuntimeInferenceOutput {
  std::vector<double> two_electron_hamiltonian;
  std::vector<double> dense_orbital_residual;
  double reference_energy_residual = 0.0;
  bool has_total_energy = false;
  double total_energy = 0.0;
};

void write_column_major_matrix(
    const fs::path& path,
    const std::vector<double>& row_major_values,
    int rows,
    int columns) {
  if (rows < 0 || columns < 0) {
    throw std::invalid_argument("matrix shape must be non-negative");
  }
  if (row_major_values.size() !=
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns)) {
    throw std::invalid_argument(
        "matrix size mismatch when writing column-major output: " + path.string());
  }
  std::vector<double> column_major_values(
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns),
      0.0);
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      column_major_values[static_cast<std::size_t>(column) * rows + row] =
          row_major_values[static_cast<std::size_t>(row) * columns + column];
    }
  }
  write_binary_container(path, column_major_values);
}

std::vector<double> build_structure_occupancy(
    const RawStructureData& raw_structure_data,
    int n_orbitals) {
  if (n_orbitals <= 0) {
    throw std::invalid_argument("n_orbitals must be positive");
  }
  std::vector<double> occupancy(
      static_cast<std::size_t>(raw_structure_data.n_structures) *
          static_cast<std::size_t>(n_orbitals),
      0.0);
  for (int structure_index = 0;
       structure_index < raw_structure_data.n_structures;
       ++structure_index) {
    const int* structure_orbitals =
        raw_structure_data.structure_orbitals_data(structure_index);
    for (int electron_index = 0;
         electron_index < raw_structure_data.n_total_electrons;
         ++electron_index) {
      const int orbital_index = structure_orbitals[electron_index] - 1;
      if (orbital_index < 0 || orbital_index >= n_orbitals) {
        throw std::runtime_error(
            "raw structure orbital index is out of range for structure occupancy");
      }
      occupancy[static_cast<std::size_t>(structure_index) * n_orbitals +
                static_cast<std::size_t>(orbital_index)] += 1.0;
    }
  }
  return occupancy;
}

std::vector<std::uint8_t> build_orbital_basis_mask(
    const OrbitalPreparationInput& orbital_preparation_input) {
  const int n_orbitals = orbital_preparation_input.n_orbitals;
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  std::vector<std::uint8_t> mask(
      static_cast<std::size_t>(n_orbitals) * static_cast<std::size_t>(n_basis_functions),
      0);
  for (int orbital_index = 0; orbital_index < n_orbitals; ++orbital_index) {
    const int coefficient_count =
        orbital_preparation_input.orbital_basis_counts[static_cast<std::size_t>(orbital_index)];
    if (coefficient_count < 0 || coefficient_count > n_basis_functions) {
      throw std::runtime_error("orbital_basis_counts contains an invalid coefficient count");
    }
    for (int coefficient_index = 0; coefficient_index < coefficient_count; ++coefficient_index) {
      mask[static_cast<std::size_t>(orbital_index) * n_basis_functions + coefficient_index] = 1;
    }
  }
  return mask;
}

std::vector<std::uint8_t> build_orbital_shell_dense_mask(
    const OrbitalPreparationInput& orbital_preparation_input,
    const CppVbStaticMoleculeMetadata& static_molecule_metadata,
    const std::vector<std::uint8_t>& orbital_basis_mask) {
  const int n_orbitals = orbital_preparation_input.n_orbitals;
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  if (orbital_basis_mask.size() !=
      static_cast<std::size_t>(n_orbitals) * static_cast<std::size_t>(n_basis_functions)) {
    throw std::invalid_argument("orbital_basis_mask size mismatch");
  }

  std::vector<std::uint8_t> dense_mask(
      static_cast<std::size_t>(n_orbitals) * static_cast<std::size_t>(n_basis_functions),
      0);
  for (int orbital_index = 0; orbital_index < n_orbitals; ++orbital_index) {
    std::vector<std::uint8_t> touched_shells(
        static_cast<std::size_t>(static_molecule_metadata.n_shells),
        0);
    for (int coefficient_index = 0; coefficient_index < n_basis_functions; ++coefficient_index) {
      const std::size_t sparse_offset =
          static_cast<std::size_t>(orbital_index) * n_basis_functions + coefficient_index;
      if (orbital_basis_mask[sparse_offset] == 0) {
        continue;
      }
      const int basis_function_index =
          orbital_preparation_input.orbital_basis_index_table[sparse_offset] - 1;
      if (basis_function_index < 0 ||
          basis_function_index >= orbital_preparation_input.n_basis_functions) {
        throw std::runtime_error("orbital_basis_index_table contains an invalid AO index");
      }
      const int shell_index =
          static_molecule_metadata.ao_to_shell[static_cast<std::size_t>(basis_function_index)];
      if (shell_index < 0 || shell_index >= static_molecule_metadata.n_shells) {
        throw std::runtime_error("ao_to_shell contains an invalid shell index");
      }
      touched_shells[static_cast<std::size_t>(shell_index)] = 1;
    }

    for (int shell_index = 0; shell_index < static_molecule_metadata.n_shells; ++shell_index) {
      if (touched_shells[static_cast<std::size_t>(shell_index)] == 0) {
        continue;
      }
      const int shell_start =
          static_molecule_metadata.shell_ao_starts[static_cast<std::size_t>(shell_index)];
      const int shell_count =
          static_molecule_metadata.shell_ao_counts[static_cast<std::size_t>(shell_index)];
      if (shell_start < 0 || shell_count < 0 ||
          shell_start + shell_count > n_basis_functions) {
        throw std::runtime_error("shell AO layout is out of range when building dense mask");
      }
      for (int local_ao_index = 0; local_ao_index < shell_count; ++local_ao_index) {
        dense_mask[static_cast<std::size_t>(orbital_index) * n_basis_functions +
                   static_cast<std::size_t>(shell_start + local_ao_index)] = 1;
      }
    }
  }
  return dense_mask;
}

std::vector<double> build_dense_orbital_coefficients(
    const OrbitalPreparationInput& orbital_preparation_input,
    const std::vector<std::uint8_t>& orbital_basis_mask,
    const std::vector<std::uint8_t>& orbital_shell_dense_mask) {
  const int n_orbitals = orbital_preparation_input.n_orbitals;
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  const std::size_t matrix_size =
      static_cast<std::size_t>(n_orbitals) * static_cast<std::size_t>(n_basis_functions);
  if (orbital_basis_mask.size() != matrix_size ||
      orbital_shell_dense_mask.size() != matrix_size) {
    throw std::invalid_argument("orbital mask size mismatch when building dense coefficients");
  }

  std::vector<double> dense_coefficients(matrix_size, 0.0);
  for (int orbital_index = 0; orbital_index < n_orbitals; ++orbital_index) {
    for (int coefficient_index = 0; coefficient_index < n_basis_functions; ++coefficient_index) {
      const std::size_t sparse_offset =
          static_cast<std::size_t>(orbital_index) * n_basis_functions + coefficient_index;
      if (orbital_basis_mask[sparse_offset] == 0) {
        continue;
      }
      const int basis_function_index =
          orbital_preparation_input.orbital_basis_index_table[sparse_offset] - 1;
      if (basis_function_index < 0 || basis_function_index >= n_basis_functions) {
        throw std::runtime_error("orbital_basis_index_table contains an invalid AO index");
      }
      dense_coefficients[static_cast<std::size_t>(orbital_index) * n_basis_functions +
                         static_cast<std::size_t>(basis_function_index)] +=
          orbital_preparation_input.orbital_value_table[sparse_offset];
    }
  }
  for (std::size_t index = 0; index < dense_coefficients.size(); ++index) {
    if (orbital_shell_dense_mask[index] == 0) {
      dense_coefficients[index] = 0.0;
    }
  }
  return dense_coefficients;
}

StructurePairTopology build_structure_pair_topology(
    const RawStructureData& raw_structure_data) {
  StructurePairTopology topology;
  const int n_inactive_doubly_occupied_orbitals =
      (raw_structure_data.n_total_electrons - raw_structure_data.n_active_electrons) / 2;
  topology.n_open_shell_electrons = raw_structure_data.spin_multiplicity - 1;
  topology.n_active_beta_electrons =
      (raw_structure_data.n_active_electrons - topology.n_open_shell_electrons) / 2;

  const int active_start = 2 * n_inactive_doubly_occupied_orbitals;
  const int active_stop = active_start + raw_structure_data.n_active_electrons;
  if (active_start < 0 || active_stop > raw_structure_data.n_total_electrons) {
    throw std::runtime_error("active-electron window is out of range for raw structures");
  }

  if (topology.n_active_beta_electrons > 0) {
    topology.structure_pair_orbital_indices.resize(
        static_cast<std::size_t>(raw_structure_data.n_structures) *
            static_cast<std::size_t>(topology.n_active_beta_electrons) * 2,
        0);
    topology.structure_pair_mask.resize(
        static_cast<std::size_t>(raw_structure_data.n_structures) *
            static_cast<std::size_t>(topology.n_active_beta_electrons),
        1);
  }
  if (topology.n_open_shell_electrons > 0) {
    topology.structure_open_shell_orbitals.resize(
        static_cast<std::size_t>(raw_structure_data.n_structures) *
            static_cast<std::size_t>(topology.n_open_shell_electrons),
        0);
    topology.structure_open_shell_mask.resize(
        static_cast<std::size_t>(raw_structure_data.n_structures) *
            static_cast<std::size_t>(topology.n_open_shell_electrons),
        1);
  }

  for (int structure_index = 0;
       structure_index < raw_structure_data.n_structures;
       ++structure_index) {
    const int* structure_orbitals =
        raw_structure_data.structure_orbitals_data(structure_index);
    for (int pair_index = 0; pair_index < topology.n_active_beta_electrons; ++pair_index) {
      const int left_orbital =
          structure_orbitals[active_start + 2 * pair_index] - 1;
      const int right_orbital =
          structure_orbitals[active_start + 2 * pair_index + 1] - 1;
      const std::size_t pair_offset =
          (static_cast<std::size_t>(structure_index) * topology.n_active_beta_electrons +
           static_cast<std::size_t>(pair_index)) *
          2;
      topology.structure_pair_orbital_indices[pair_offset] = left_orbital;
      topology.structure_pair_orbital_indices[pair_offset + 1] = right_orbital;
    }
    for (int open_shell_index = 0;
         open_shell_index < topology.n_open_shell_electrons;
         ++open_shell_index) {
      const int orbital_index =
          structure_orbitals[active_start + 2 * topology.n_active_beta_electrons +
                             open_shell_index] -
          1;
      topology.structure_open_shell_orbitals[
          static_cast<std::size_t>(structure_index) * topology.n_open_shell_electrons +
          static_cast<std::size_t>(open_shell_index)] = orbital_index;
    }
  }
  return topology;
}

PreparedOnnxFeatures prepare_onnx_features(
    const CppVbInput& input,
    const RawStructureData& raw_structure_data,
    const CppVbStaticMoleculeMetadata& static_molecule_metadata) {
  PreparedOnnxFeatures prepared;
  prepared.structure_occupancy = build_structure_occupancy(
      raw_structure_data,
      input.orbital_preparation_input.n_orbitals);
  prepared.orbital_basis_mask =
      build_orbital_basis_mask(input.orbital_preparation_input);
  prepared.orbital_shell_dense_mask = build_orbital_shell_dense_mask(
      input.orbital_preparation_input,
      static_molecule_metadata,
      prepared.orbital_basis_mask);
  prepared.dense_orbital_coefficients = build_dense_orbital_coefficients(
      input.orbital_preparation_input,
      prepared.orbital_basis_mask,
      prepared.orbital_shell_dense_mask);
  prepared.pair_topology = build_structure_pair_topology(raw_structure_data);
  return prepared;
}

PreparedInferenceInput prepare_model_input(
    const CppVbInput& input,
    const ActiveSpaceOrbitalPreparer& orbital_preparer,
    const AoEffectiveOneElectronBuilder& ao_effective_one_electron_builder,
    const ActiveSpaceOneElectronBuilder& active_space_one_electron_builder,
    const FullDeterminantStructureHamiltonianOverlapBuilder& structure_builder) {
  const int n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) /
      2;
  const auto orbital_result =
      orbital_preparer.prepare(input.orbital_preparation_input);
  const auto ao_effective_one_electron_result =
      ao_effective_one_electron_builder.build(
          orbital_result.inactive_density_matrix,
          input.ao_integral_input.ao_core_hamiltonian_matrix,
          input.ao_integral_input.ao_two_electron_integral_values,
          input.ao_integral_input.ao_two_electron_integral_indices,
          input.ao_integral_input.n_basis_functions);
  const auto active_space_one_electron_result =
      active_space_one_electron_builder.build(
          ao_effective_one_electron_result.ao_effective_h1e,
          orbital_result.auxiliary_orbital_matrix,
          input.ao_integral_input.n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          input.orbital_preparation_input.n_active_orbitals);

  const std::vector<double> zero_ERI(
      packed_active_two_electron_size(input.orbital_preparation_input.n_active_orbitals),
      0.0);
  PreparedInferenceInput prepared;
  prepared.structure_matrices = structure_builder.build(
      input.structure_data.alpha_det,
      input.structure_data.beta_det,
      input.structure_data.determinant_to_structure_terms,
      orbital_result.active_orbital_overlap_matrix,
      active_space_one_electron_result.h1e_act,
      input.orbital_preparation_input.n_active_orbitals,
      zero_ERI,
      input.structure_data.n_structures);
  prepared.one_electron_reference_energy =
      compute_one_electron_reference_energy(
          orbital_result.inactive_density_matrix,
          ao_effective_one_electron_result.ao_effective_h1e,
          input.ao_integral_input.ao_core_hamiltonian_matrix,
          input.ao_integral_input.n_basis_functions);
  return prepared;
}

fs::path write_single_step_inference_sample(
    const fs::path& work_directory,
    const CppVbInput& input,
    const RawStructureData& raw_structure_data,
    const CppVbStaticMoleculeMetadata& static_molecule_metadata,
    const StructureAccumulationResult& structure_matrices,
    double one_electron_reference_energy,
    double nuclear_repulsion_energy) {
  const fs::path sample_dir = work_directory / "inference_sample";
  const fs::path static_dir = sample_dir / "static";
  const fs::path step_dir = sample_dir / "steps" / "step_000000";
  fs::create_directories(static_dir);
  fs::create_directories(step_dir);

  const auto differentiable_parameter_indices =
      collect_differentiable_parameter_indices(input.orbital_preparation_input);
  const std::vector<double> zero_gradient(
      input.orbital_preparation_input.orbital_value_table.size(),
      0.0);

  write_binary_container(
      static_dir / "raw_structure_orbitals_i32.bin",
      raw_structure_data.raw_structure_orbitals);
  write_binary_container(
      static_dir / "atomic_numbers_i32.bin",
      static_molecule_metadata.atomic_numbers);
  write_binary_container(
      static_dir / "atomic_coordinates_f64.bin",
      static_molecule_metadata.atomic_coordinates);
  write_binary_container(
      static_dir / "shell_to_atom_i32.bin",
      static_molecule_metadata.shell_to_atom);
  write_binary_container(
      static_dir / "shell_angular_momenta_i32.bin",
      static_molecule_metadata.shell_angular_momenta);
  write_binary_container(
      static_dir / "shell_n_primitives_i32.bin",
      static_molecule_metadata.shell_n_primitives);
  write_binary_container(
      static_dir / "shell_ao_starts_i32.bin",
      static_molecule_metadata.shell_ao_starts);
  write_binary_container(
      static_dir / "shell_ao_counts_i32.bin",
      static_molecule_metadata.shell_ao_counts);
  write_binary_container(
      static_dir / "ao_to_atom_i32.bin",
      static_molecule_metadata.ao_to_atom);
  write_binary_container(
      static_dir / "ao_to_shell_i32.bin",
      static_molecule_metadata.ao_to_shell);
  write_binary_container(
      static_dir / "ao_angular_momenta_i32.bin",
      static_molecule_metadata.ao_angular_momenta);
  write_binary_container(
      static_dir / "ao_shell_local_indices_i32.bin",
      static_molecule_metadata.ao_shell_local_indices);
  write_binary_container(
      static_dir / "ao_cartesian_exponents_i32.bin",
      static_molecule_metadata.ao_cartesian_exponents);
  write_binary_container(
      static_dir / "orbital_basis_index_table_i32.bin",
      input.orbital_preparation_input.orbital_basis_index_table);
  write_binary_container(
      static_dir / "orbital_basis_counts_i32.bin",
      input.orbital_preparation_input.orbital_basis_counts);
  write_binary_container(
      static_dir / "original_orbital_basis_counts_i32.bin",
      input.orbital_preparation_input.original_orbital_basis_counts);
  write_binary_container(
      static_dir / "differentiable_parameter_indices_i32.bin",
      differentiable_parameter_indices);

  std::ostringstream static_metadata_stream;
  static_metadata_stream << "{\n"
                         << "  \"index_base\": 0,\n"
                         << "  \"n_atoms\": " << static_molecule_metadata.n_atoms << ",\n"
                         << "  \"n_shells\": " << static_molecule_metadata.n_shells << ",\n"
                         << "  \"n_basis_functions\": "
                         << input.orbital_preparation_input.n_basis_functions << ",\n"
                         << "  \"n_orbitals\": "
                         << input.orbital_preparation_input.n_orbitals << ",\n"
                         << "  \"n_differentiable_parameters\": "
                         << differentiable_parameter_indices.size() << ",\n"
                         << "  \"atomic_coordinate_unit\": \"bohr\",\n"
                         << "  \"atomic_coordinates_layout\": ["
                         << static_molecule_metadata.n_atoms << ", 3],\n"
                         << "  \"ao_cartesian_exponents_layout\": ["
                         << input.orbital_preparation_input.n_basis_functions << ", 3],\n"
                         << "  \"shell_ordering\": \"basis_file_shell_order\",\n"
                         << "  \"ao_ordering\": \"cartesian_shell_local_order\",\n"
                         << "  \"ao_cartesian_exponents_columns\": [\"lx\", \"ly\", \"lz\"]\n"
                         << "}\n";
  write_text_file(static_dir / "metadata.json", static_metadata_stream.str());

  write_binary_container(
      step_dir / "orbital_value_table_f64.bin",
      input.orbital_preparation_input.orbital_value_table);
  write_binary_container(
      step_dir / "overlap_matrix_f64.bin",
      structure_matrices.overlap_matrix);
  write_binary_container(
      step_dir / "hamiltonian_matrix_f64.bin",
      structure_matrices.one_electron_hamiltonian_matrix);
  write_binary_container(
      step_dir / "one_electron_hamiltonian_matrix_f64.bin",
      structure_matrices.one_electron_hamiltonian_matrix);
  write_binary_container(
      step_dir / "sparse_orbital_energy_gradient_f64.bin",
      zero_gradient);
  write_binary_container(
      step_dir / "sparse_orbital_reference_energy_gradient_f64.bin",
      zero_gradient);

  std::ostringstream step_metadata_stream;
  step_metadata_stream << "{\n"
                       << "  \"accepted_iteration_index\": 0,\n"
                       << "  \"total_energy\": 0.0,\n"
                       << "  \"one_electron_reference_energy\": "
                       << std::setprecision(17) << one_electron_reference_energy << ",\n"
                       << "  \"n_orbitals\": "
                       << input.orbital_preparation_input.n_orbitals << ",\n"
                       << "  \"n_basis_functions\": "
                       << input.orbital_preparation_input.n_basis_functions << ",\n"
                       << "  \"n_structures\": " << input.structure_data.n_structures << ",\n"
                       << "  \"n_differentiable_parameters\": "
                       << differentiable_parameter_indices.size() << "\n"
                       << "}\n";
  write_text_file(step_dir / "metadata.json", step_metadata_stream.str());

  std::ostringstream sample_metadata_stream;
  sample_metadata_stream << "{\n"
                         << "  \"format_version\": 5,\n"
                         << "  \"status\": \"completed\",\n"
                         << "  \"sample_name\": \"inference_sample\",\n"
                         << "  \"source_input_path\": \"\",\n"
                         << "  \"optimizer_backend\": \"inference\",\n"
                         << "  \"algorithm\": \"original\",\n"
                         << "  \"n_structures\": " << raw_structure_data.n_structures << ",\n"
                         << "  \"n_total_electrons\": " << raw_structure_data.n_total_electrons << ",\n"
                         << "  \"n_active_electrons\": " << raw_structure_data.n_active_electrons << ",\n"
                         << "  \"spin_multiplicity\": " << raw_structure_data.spin_multiplicity << ",\n"
                         << "  \"n_atoms\": " << static_molecule_metadata.n_atoms << ",\n"
                         << "  \"n_shells\": " << static_molecule_metadata.n_shells << ",\n"
                         << "  \"n_basis_functions\": "
                         << input.orbital_preparation_input.n_basis_functions << ",\n"
                         << "  \"n_orbitals\": "
                         << input.orbital_preparation_input.n_orbitals << ",\n"
                         << "  \"n_active_orbitals\": "
                         << input.orbital_preparation_input.n_active_orbitals << ",\n"
                         << "  \"n_differentiable_parameters\": "
                         << differentiable_parameter_indices.size() << ",\n"
                         << "  \"nuclear_repulsion_energy\": " << std::setprecision(17)
                         << nuclear_repulsion_energy << ",\n"
                         << "  \"accepted_iteration_count\": 1,\n"
                         << "  \"converged\": false,\n"
                         << "  \"termination_reason\": \"single_step_inference\"\n"
                         << "}\n";
  write_text_file(sample_dir / "metadata.json", sample_metadata_stream.str());

  return sample_dir;
}

void run_python_inference(
    const DeepVBHJaxInferenceOptions& options,
    const fs::path& work_directory,
    const fs::path& prediction_directory) {
  const fs::path repo_root = fs::absolute(options.repo_root);
  const fs::path python_executable =
      resolve_relative_to(repo_root, options.python_executable);
  const fs::path checkpoint_path =
      resolve_relative_to(repo_root, options.checkpoint_path);
  const fs::path stdout_path = work_directory / "python_stdout.log";
  const fs::path stderr_path = work_directory / "python_stderr.log";
  const std::string environment_prefix =
      (options.device == "cpu"
           ? "DEEPVBH_SANITIZE_CUDA_RUNTIME=1 LD_LIBRARY_PATH= JAX_PLATFORMS=cpu "
           : "DEEPVBH_SANITIZE_CUDA_RUNTIME=1 LD_LIBRARY_PATH= ");
  const std::string inner_command =
      "cd " + shell_quote(repo_root.string()) +
      " && " + environment_prefix + shell_quote(python_executable.string()) +
      " -m deepVBH.infer_jax"
      " --data-root " + shell_quote(fs::absolute(work_directory).string()) +
      " --sample-name inference_sample"
      " --accepted-iteration-index 0"
      " --checkpoint " + shell_quote(checkpoint_path.string()) +
      " --output-dir " + shell_quote(fs::absolute(prediction_directory).string()) +
      " --device " + shell_quote(options.device) +
      " --dtype " + shell_quote(options.dtype) +
      " > " + shell_quote(stdout_path.string()) +
      " 2> " + shell_quote(stderr_path.string());
  const std::string command =
      "/usr/bin/bash -lc " + shell_quote(inner_command);
  const int status = std::system(command.c_str());
  if (status != 0) {
    throw std::runtime_error(
        "DeepVBH JAX inference command failed with status " + std::to_string(status) +
        ". stdout=" + stdout_path.string() +
        " stderr=" + stderr_path.string() +
        " work_directory=" + work_directory.string());
  }
}

#ifdef XMVB_CPP_ENABLE_ONNX_RUNTIME

struct OnnxTensorBuffer {
  std::vector<int64_t> shape;
  ONNXTensorElementDataType element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  std::vector<std::uint8_t> bytes;
};

std::string onnx_type_name(ONNXTensorElementDataType element_type) {
  switch (element_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return "bool";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return "float64";
    default:
      return "unsupported";
  }
}

std::size_t element_count(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return 1;
  }
  std::size_t count = 1;
  for (const int64_t dimension : shape) {
    if (dimension < 0) {
      throw std::invalid_argument("runtime tensor shape must not contain negative dimensions");
    }
    count *= static_cast<std::size_t>(dimension);
  }
  return count;
}

std::vector<int64_t> resolve_runtime_shape(
    const std::vector<int64_t>& expected_shape,
    const std::vector<int64_t>& actual_shape,
    const std::string& tensor_name) {
  if (expected_shape.empty()) {
    return actual_shape;
  }
  if (expected_shape == actual_shape) {
    return actual_shape;
  }

  bool contains_dynamic_dimension = false;
  if (expected_shape.size() == actual_shape.size()) {
    std::vector<int64_t> resolved_shape = expected_shape;
    for (std::size_t index = 0; index < expected_shape.size(); ++index) {
      if (expected_shape[index] < 0) {
        resolved_shape[index] = actual_shape[index];
        contains_dynamic_dimension = true;
        continue;
      }
      if (expected_shape[index] != actual_shape[index]) {
        resolved_shape.clear();
        break;
      }
    }
    if (!resolved_shape.empty()) {
      return contains_dynamic_dimension ? resolved_shape : actual_shape;
    }
  }

  const std::size_t actual_count = element_count(actual_shape);
  if (actual_count == 1 && element_count(expected_shape) == 1) {
    return expected_shape;
  }

  std::ostringstream message;
  message << "tensor shape mismatch for input `" << tensor_name << "`: expected [";
  for (std::size_t index = 0; index < expected_shape.size(); ++index) {
    if (index != 0) {
      message << ", ";
    }
    message << expected_shape[index];
  }
  message << "], got [";
  for (std::size_t index = 0; index < actual_shape.size(); ++index) {
    if (index != 0) {
      message << ", ";
    }
    message << actual_shape[index];
  }
  message << "]";
  throw std::runtime_error(message.str());
}

template <typename DestinationType, typename SourceType>
std::vector<std::uint8_t> cast_vector_to_bytes(
    const std::vector<SourceType>& values) {
  std::vector<DestinationType> cast_values(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    cast_values[index] = static_cast<DestinationType>(values[index]);
  }
  std::vector<std::uint8_t> bytes(sizeof(DestinationType) * cast_values.size(), 0);
  if (!cast_values.empty()) {
    std::memcpy(bytes.data(), cast_values.data(), bytes.size());
  }
  return bytes;
}

OnnxTensorBuffer make_double_input_tensor(
    const std::vector<double>& values,
    const std::vector<int64_t>& actual_shape,
    const std::vector<int64_t>& expected_shape,
    ONNXTensorElementDataType expected_type,
    const std::string& tensor_name) {
  OnnxTensorBuffer tensor;
  tensor.shape = resolve_runtime_shape(expected_shape, actual_shape, tensor_name);
  tensor.element_type = expected_type;
  if (element_count(tensor.shape) != values.size()) {
    throw std::runtime_error("double tensor size mismatch for `" + tensor_name + "`");
  }
  switch (expected_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      tensor.bytes = cast_vector_to_bytes<double>(values);
      return tensor;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      tensor.bytes = cast_vector_to_bytes<float>(values);
      return tensor;
    default:
      throw std::runtime_error(
          "unsupported floating-point input type for `" + tensor_name + "`: " +
          onnx_type_name(expected_type));
  }
}

OnnxTensorBuffer make_int_input_tensor(
    const std::vector<int>& values,
    const std::vector<int64_t>& actual_shape,
    const std::vector<int64_t>& expected_shape,
    ONNXTensorElementDataType expected_type,
    const std::string& tensor_name) {
  OnnxTensorBuffer tensor;
  tensor.shape = resolve_runtime_shape(expected_shape, actual_shape, tensor_name);
  tensor.element_type = expected_type;
  if (element_count(tensor.shape) != values.size()) {
    throw std::runtime_error("integer tensor size mismatch for `" + tensor_name + "`");
  }
  switch (expected_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      tensor.bytes = cast_vector_to_bytes<int64_t>(values);
      return tensor;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      tensor.bytes = cast_vector_to_bytes<int32_t>(values);
      return tensor;
    default:
      throw std::runtime_error(
          "unsupported integer input type for `" + tensor_name + "`: " +
          onnx_type_name(expected_type));
  }
}

OnnxTensorBuffer make_bool_input_tensor(
    const std::vector<std::uint8_t>& values,
    const std::vector<int64_t>& actual_shape,
    const std::vector<int64_t>& expected_shape,
    ONNXTensorElementDataType expected_type,
    const std::string& tensor_name) {
  if (expected_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
    throw std::runtime_error(
        "unsupported boolean input type for `" + tensor_name + "`: " +
        onnx_type_name(expected_type));
  }
  OnnxTensorBuffer tensor;
  tensor.shape = resolve_runtime_shape(expected_shape, actual_shape, tensor_name);
  tensor.element_type = expected_type;
  if (element_count(tensor.shape) != values.size()) {
    throw std::runtime_error("boolean tensor size mismatch for `" + tensor_name + "`");
  }
  tensor.bytes = values;
  return tensor;
}

Ort::Value make_ort_value(
    const Ort::MemoryInfo& memory_info,
    OnnxTensorBuffer& tensor) {
  static std::uint8_t zero_sized_tensor_storage = 0;
  void* data = tensor.bytes.empty()
                   ? static_cast<void*>(&zero_sized_tensor_storage)
                   : static_cast<void*>(tensor.bytes.data());
  return Ort::Value::CreateTensor(
      memory_info,
      data,
      tensor.bytes.size(),
      tensor.shape.data(),
      tensor.shape.size(),
      tensor.element_type);
}

std::vector<double> copy_output_tensor_as_double(
    const Ort::Value& tensor_value,
    const std::string& tensor_name) {
  if (!tensor_value.IsTensor()) {
    throw std::runtime_error("ONNX output `" + tensor_name + "` is not a tensor");
  }
  const auto tensor_info = tensor_value.GetTensorTypeAndShapeInfo();
  const auto element_type = tensor_info.GetElementType();
  const std::size_t value_count = tensor_info.GetElementCount();
  std::vector<double> values(value_count, 0.0);
  switch (element_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
      const double* data = tensor_value.GetTensorData<double>();
      std::copy(data, data + value_count, values.begin());
      return values;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
      const float* data = tensor_value.GetTensorData<float>();
      for (std::size_t index = 0; index < value_count; ++index) {
        values[index] = static_cast<double>(data[index]);
      }
      return values;
    }
    default:
      throw std::runtime_error(
          "unsupported floating-point output type for `" + tensor_name + "`: " +
          onnx_type_name(element_type));
  }
}

void write_required_onnx_outputs(
    const fs::path& prediction_directory,
    const OnnxRuntimeInferenceOutput& output,
    int n_structures) {
  write_column_major_matrix(
      prediction_directory / "predicted_two_electron_hamiltonian_f64.bin",
      output.two_electron_hamiltonian,
      n_structures,
      n_structures);
  write_binary_container(
      prediction_directory / "predicted_dense_orbital_residual_f64.bin",
      output.dense_orbital_residual);
  write_binary_container(
      prediction_directory / "predicted_reference_energy_residual_f64.bin",
      std::vector<double>{output.reference_energy_residual});
  if (output.has_total_energy) {
    write_binary_container(
        prediction_directory / "predicted_total_energy_f64.bin",
        std::vector<double>{output.total_energy});
  }
}

OnnxRuntimeInferenceOutput collect_required_onnx_outputs(
    const std::unordered_map<std::string, Ort::Value>& outputs,
    int n_structures,
    int n_orbitals,
    int n_basis_functions) {
  const auto two_electron_iterator = outputs.find("two_electron_hamiltonian");
  if (two_electron_iterator == outputs.end()) {
    throw std::runtime_error(
        "ONNX model did not produce required output `two_electron_hamiltonian`");
  }
  const auto dense_residual_iterator = outputs.find("dense_orbital_residual");
  if (dense_residual_iterator == outputs.end()) {
    throw std::runtime_error(
        "ONNX model did not produce required output `dense_orbital_residual`");
  }
  const auto reference_residual_iterator = outputs.find("reference_energy_residual");
  if (reference_residual_iterator == outputs.end()) {
    throw std::runtime_error(
        "ONNX model did not produce required output `reference_energy_residual`");
  }

  OnnxRuntimeInferenceOutput output;
  output.two_electron_hamiltonian =
      copy_output_tensor_as_double(
          two_electron_iterator->second,
          two_electron_iterator->first);
  if (output.two_electron_hamiltonian.size() !=
      static_cast<std::size_t>(n_structures) * static_cast<std::size_t>(n_structures)) {
    throw std::runtime_error("unexpected size for ONNX output `two_electron_hamiltonian`");
  }

  output.dense_orbital_residual =
      copy_output_tensor_as_double(
          dense_residual_iterator->second,
          dense_residual_iterator->first);
  if (output.dense_orbital_residual.size() !=
      static_cast<std::size_t>(n_orbitals) * static_cast<std::size_t>(n_basis_functions)) {
    throw std::runtime_error("unexpected size for ONNX output `dense_orbital_residual`");
  }

  const std::vector<double> reference_energy_residual =
      copy_output_tensor_as_double(
          reference_residual_iterator->second,
          reference_residual_iterator->first);
  if (reference_energy_residual.size() != 1) {
    throw std::runtime_error("unexpected size for ONNX output `reference_energy_residual`");
  }
  output.reference_energy_residual = reference_energy_residual.front();

  const auto total_energy_iterator = outputs.find("total_energy");
  if (total_energy_iterator != outputs.end()) {
    const std::vector<double> total_energy =
        copy_output_tensor_as_double(
            total_energy_iterator->second,
            total_energy_iterator->first);
    if (total_energy.size() != 1) {
      throw std::runtime_error("unexpected size for ONNX output `total_energy`");
    }
    output.has_total_energy = true;
    output.total_energy = total_energy.front();
  }
  return output;
}

struct CachedOnnxSession {
  explicit CachedOnnxSession(
      const fs::path& model_path)
      : environment(ORT_LOGGING_LEVEL_WARNING, "deepvbh_onnx_runtime"),
        session_options(),
        session(nullptr),
        allocator() {
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session = Ort::Session(
        environment,
        model_path.string().c_str(),
        session_options);
    const std::size_t input_count = session.GetInputCount();
    input_name_storage.reserve(input_count);
    input_names.reserve(input_count);
    for (std::size_t input_index = 0; input_index < input_count; ++input_index) {
      auto input_name = session.GetInputNameAllocated(input_index, allocator);
      input_name_storage.emplace_back(input_name.get());
      input_names.push_back(input_name_storage.back().c_str());
    }
    const std::size_t output_count = session.GetOutputCount();
    output_name_storage.reserve(output_count);
    output_names.reserve(output_count);
    for (std::size_t output_index = 0; output_index < output_count; ++output_index) {
      auto output_name = session.GetOutputNameAllocated(output_index, allocator);
      output_name_storage.emplace_back(output_name.get());
      output_names.push_back(output_name_storage.back().c_str());
    }
  }

  Ort::Env environment;
  Ort::SessionOptions session_options;
  Ort::Session session;
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> input_name_storage;
  std::vector<const char*> input_names;
  std::vector<std::string> output_name_storage;
  std::vector<const char*> output_names;
};

std::shared_ptr<CachedOnnxSession> get_cached_onnx_session(
    const fs::path& model_path) {
  static std::mutex cache_mutex;
  static std::map<std::string, std::weak_ptr<CachedOnnxSession>> cache;
  const std::string cache_key = fs::absolute(model_path).string();
  std::lock_guard<std::mutex> lock(cache_mutex);
  const auto iterator = cache.find(cache_key);
  if (iterator != cache.end()) {
    if (auto existing = iterator->second.lock()) {
      return existing;
    }
  }
  auto created = std::make_shared<CachedOnnxSession>(fs::absolute(model_path));
  cache[cache_key] = created;
  return created;
}

OnnxRuntimeInferenceOutput run_onnx_runtime_inference(
    const DeepVBHJaxInferenceOptions& options,
    const CppVbInput& input,
    const RawStructureData& raw_structure_data,
    const CppVbStaticMoleculeMetadata& static_molecule_metadata,
    const PreparedInferenceInput& prepared_input,
    double nuclear_repulsion_energy,
    const fs::path& prediction_directory) {
  if (options.onnx_model_path.empty()) {
    throw std::invalid_argument(
        "DeepVBH ONNX Runtime backend requires onnx_model_path");
  }

  const OrbitalPreparationInput& orbital_input = input.orbital_preparation_input;
  const PreparedOnnxFeatures prepared_features = prepare_onnx_features(
      input,
      raw_structure_data,
      static_molecule_metadata);

  const fs::path repo_root = fs::absolute(options.repo_root);
  const fs::path onnx_model_path =
      resolve_relative_to(repo_root, options.onnx_model_path);
  std::string phase = "initializing ONNX Runtime";
  try {
    phase = "acquiring cached ONNX Runtime session";
    const auto cached_session = get_cached_onnx_session(onnx_model_path);
    static const Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator,
        OrtMemTypeDefault);
    const ONNXTensorElementDataType floating_type =
        options.dtype == "float64"
            ? ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
            : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    const std::size_t input_count = cached_session->input_name_storage.size();
    std::vector<OnnxTensorBuffer> input_buffers;
    input_buffers.reserve(input_count);
    std::vector<Ort::Value> input_values;
    input_values.reserve(input_count);

    for (std::size_t input_index = 0; input_index < input_count; ++input_index) {
      const std::string& input_name = cached_session->input_name_storage[input_index];
      phase = "preparing input `" + input_name + "`";

      if (input_name == "atomic_numbers") {
        input_buffers.push_back(make_int_input_tensor(
            static_molecule_metadata.atomic_numbers,
            {static_molecule_metadata.n_atoms},
            {static_molecule_metadata.n_atoms},
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            input_name));
      } else if (input_name == "atomic_coordinates") {
        input_buffers.push_back(make_double_input_tensor(
            static_molecule_metadata.atomic_coordinates,
            {static_molecule_metadata.n_atoms, 3},
            {static_molecule_metadata.n_atoms, 3},
            floating_type,
            input_name));
      } else if (input_name == "ao_to_atom") {
        input_buffers.push_back(make_int_input_tensor(
            static_molecule_metadata.ao_to_atom,
            {orbital_input.n_basis_functions},
            {orbital_input.n_basis_functions},
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            input_name));
      } else if (input_name == "ao_angular_momenta") {
        input_buffers.push_back(make_int_input_tensor(
            static_molecule_metadata.ao_angular_momenta,
            {orbital_input.n_basis_functions},
            {orbital_input.n_basis_functions},
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            input_name));
      } else if (input_name == "ao_shell_local_indices") {
        input_buffers.push_back(make_int_input_tensor(
            static_molecule_metadata.ao_shell_local_indices,
            {orbital_input.n_basis_functions},
            {orbital_input.n_basis_functions},
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            input_name));
      } else if (input_name == "ao_cartesian_exponents") {
        input_buffers.push_back(make_int_input_tensor(
            static_molecule_metadata.ao_cartesian_exponents,
            {orbital_input.n_basis_functions, 3},
            {orbital_input.n_basis_functions, 3},
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            input_name));
      } else if (input_name == "structure_occupancy") {
        input_buffers.push_back(make_double_input_tensor(
            prepared_features.structure_occupancy,
            {raw_structure_data.n_structures, orbital_input.n_orbitals},
            {raw_structure_data.n_structures, orbital_input.n_orbitals},
            floating_type,
            input_name));
      } else if (input_name == "orbital_basis_counts") {
        input_buffers.push_back(make_int_input_tensor(
            orbital_input.orbital_basis_counts.vector(),
            {orbital_input.n_orbitals},
            {orbital_input.n_orbitals},
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            input_name));
      } else if (input_name == "structure_pair_orbital_indices") {
        input_buffers.push_back(make_int_input_tensor(
            prepared_features.pair_topology.structure_pair_orbital_indices,
            {raw_structure_data.n_structures,
             prepared_features.pair_topology.n_active_beta_electrons,
             2},
            {raw_structure_data.n_structures,
             prepared_features.pair_topology.n_active_beta_electrons,
             2},
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            input_name));
      } else if (input_name == "structure_pair_mask") {
        input_buffers.push_back(make_bool_input_tensor(
            prepared_features.pair_topology.structure_pair_mask,
            {raw_structure_data.n_structures,
             prepared_features.pair_topology.n_active_beta_electrons},
            {raw_structure_data.n_structures,
             prepared_features.pair_topology.n_active_beta_electrons},
            ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
            input_name));
      } else if (input_name == "structure_open_shell_orbitals") {
        input_buffers.push_back(make_int_input_tensor(
            prepared_features.pair_topology.structure_open_shell_orbitals,
            {raw_structure_data.n_structures,
             prepared_features.pair_topology.n_open_shell_electrons},
            {raw_structure_data.n_structures,
             prepared_features.pair_topology.n_open_shell_electrons},
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            input_name));
      } else if (input_name == "structure_open_shell_mask") {
        input_buffers.push_back(make_bool_input_tensor(
            prepared_features.pair_topology.structure_open_shell_mask,
            {raw_structure_data.n_structures,
             prepared_features.pair_topology.n_open_shell_electrons},
            {raw_structure_data.n_structures,
             prepared_features.pair_topology.n_open_shell_electrons},
            ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
            input_name));
      } else if (input_name == "overlap_matrix") {
        input_buffers.push_back(make_double_input_tensor(
            prepared_input.structure_matrices.overlap_matrix,
            {raw_structure_data.n_structures, raw_structure_data.n_structures},
            {raw_structure_data.n_structures, raw_structure_data.n_structures},
            floating_type,
            input_name));
      } else if (input_name == "one_electron_reference_energy") {
        input_buffers.push_back(make_double_input_tensor(
            {prepared_input.one_electron_reference_energy},
            {},
            {},
            floating_type,
            input_name));
      } else if (input_name == "one_electron_hamiltonian_matrix") {
        input_buffers.push_back(make_double_input_tensor(
            prepared_input.structure_matrices.one_electron_hamiltonian_matrix,
            {raw_structure_data.n_structures, raw_structure_data.n_structures},
            {raw_structure_data.n_structures, raw_structure_data.n_structures},
            floating_type,
            input_name));
      } else if (input_name == "nuclear_repulsion_energy") {
        input_buffers.push_back(make_double_input_tensor(
            {nuclear_repulsion_energy},
            {},
            {},
            floating_type,
            input_name));
      } else if (input_name == "orbital_value_table") {
        input_buffers.push_back(make_double_input_tensor(
            orbital_input.orbital_value_table.vector(),
            {orbital_input.n_orbitals, orbital_input.n_basis_functions},
            {orbital_input.n_orbitals, orbital_input.n_basis_functions},
            floating_type,
            input_name));
      } else if (input_name == "dense_orbital_coefficients") {
        input_buffers.push_back(make_double_input_tensor(
            prepared_features.dense_orbital_coefficients,
            {orbital_input.n_orbitals, orbital_input.n_basis_functions},
            {orbital_input.n_orbitals, orbital_input.n_basis_functions},
            floating_type,
            input_name));
      } else if (input_name == "orbital_shell_dense_mask") {
        input_buffers.push_back(make_bool_input_tensor(
            prepared_features.orbital_shell_dense_mask,
            {orbital_input.n_orbitals, orbital_input.n_basis_functions},
            {orbital_input.n_orbitals, orbital_input.n_basis_functions},
            ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
            input_name));
      } else if (input_name == "orbital_basis_mask") {
        input_buffers.push_back(make_bool_input_tensor(
            prepared_features.orbital_basis_mask,
            {orbital_input.n_orbitals, orbital_input.n_basis_functions},
            {orbital_input.n_orbitals, orbital_input.n_basis_functions},
            ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
            input_name));
      } else if (input_name == "orbital_basis_index_table") {
        input_buffers.push_back(make_int_input_tensor(
            orbital_input.orbital_basis_index_table.vector(),
            {orbital_input.n_orbitals, orbital_input.n_basis_functions},
            {orbital_input.n_orbitals, orbital_input.n_basis_functions},
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            input_name));
      } else {
        throw std::runtime_error(
            "unsupported ONNX input name in current DeepVBH C++ runtime: `" + input_name + "`");
      }

      phase = "binding input `" + input_name + "`";
      input_values.push_back(make_ort_value(memory_info, input_buffers.back()));
    }

    phase = "executing ONNX Runtime session";
    auto output_values = cached_session->session.Run(
        Ort::RunOptions{nullptr},
        cached_session->input_names.data(),
        input_values.data(),
        input_values.size(),
        cached_session->output_names.data(),
        cached_session->output_names.size());
    if (output_values.size() != cached_session->output_names.size()) {
      throw std::runtime_error("ONNX Runtime returned an unexpected number of outputs");
    }

    phase = "collecting ONNX outputs";
    std::unordered_map<std::string, Ort::Value> outputs;
    outputs.reserve(output_values.size());
    for (std::size_t output_index = 0; output_index < output_values.size(); ++output_index) {
      outputs.emplace(
          cached_session->output_name_storage[output_index],
          std::move(output_values[output_index]));
    }

    phase = "collecting ONNX prediction outputs";
    OnnxRuntimeInferenceOutput output = collect_required_onnx_outputs(
        outputs,
        raw_structure_data.n_structures,
        orbital_input.n_orbitals,
        orbital_input.n_basis_functions);
    if (!prediction_directory.empty()) {
      phase = "writing ONNX prediction outputs";
      write_required_onnx_outputs(
          prediction_directory,
          output,
          raw_structure_data.n_structures);
    }
    return output;
  } catch (const std::exception& error) {
    throw std::runtime_error(
        "DeepVBH ONNX Runtime inference failed while " + phase + ": " + error.what());
  }
}

#else

OnnxRuntimeInferenceOutput run_onnx_runtime_inference(
    const DeepVBHJaxInferenceOptions& options,
    const CppVbInput& input,
    const RawStructureData& raw_structure_data,
    const CppVbStaticMoleculeMetadata& static_molecule_metadata,
    const PreparedInferenceInput& prepared_input,
    double nuclear_repulsion_energy,
    const fs::path& prediction_directory) {
  (void)input;
  (void)raw_structure_data;
  (void)static_molecule_metadata;
  (void)prepared_input;
  (void)nuclear_repulsion_energy;
  (void)prediction_directory;
  if (options.onnx_model_path.empty()) {
    throw std::invalid_argument(
        "DeepVBH ONNX Runtime backend requires onnx_model_path");
  }
  throw std::runtime_error(
      "DeepVBH ONNX Runtime backend requested, but this build does not enable ONNX Runtime. "
      "Reconfigure with -DXMVB_CPP_ENABLE_ONNX_RUNTIME=ON and set ONNXRUNTIME_ROOT_DIR.");
}

#endif

}  // namespace

DeepVBHJaxInferenceRunner::DeepVBHJaxInferenceRunner(
    DeepVBHJaxInferenceOptions options)
    : orbital_preparer_(),
      ao_effective_one_electron_builder_(),
      active_space_one_electron_builder_(),
      structure_builder_(options.algorithm),
      options_(std::move(options)) {}

DeepVBHJaxPrediction DeepVBHJaxInferenceRunner::predict(
    const CppVbInput& input,
    const RawStructureData& raw_structure_data,
    const CppVbStaticMoleculeMetadata& static_molecule_metadata,
    double nuclear_repulsion_energy) const {
  if (options_.backend == "python_jax" && options_.checkpoint_path.empty()) {
    throw std::invalid_argument("DeepVBH checkpoint_path must not be empty");
  }
  if (options_.backend == "onnx_runtime" && options_.onnx_model_path.empty()) {
    throw std::invalid_argument("DeepVBH onnx_model_path must not be empty");
  }

  const auto total_start_time = std::chrono::steady_clock::now();
  const auto preparation_start_time = std::chrono::steady_clock::now();
  const PreparedInferenceInput prepared = prepare_model_input(
      input,
      orbital_preparer_,
      ao_effective_one_electron_builder_,
      active_space_one_electron_builder_,
      structure_builder_);
  DeepVBHJaxPrediction prediction;
  prediction.backend = options_.backend;
  prediction.one_electron_reference_energy = prepared.one_electron_reference_energy;
  prediction.preparation_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - preparation_start_time).count();

  fs::path work_directory;
  fs::path sample_directory;
  fs::path prediction_directory;
  std::unique_ptr<TemporaryDirectoryGuard> work_directory_guard;
  if (options_.backend == "python_jax" || options_.keep_work_directory) {
    work_directory = make_unique_work_directory(fs::absolute(options_.work_directory));
    work_directory_guard = std::make_unique<TemporaryDirectoryGuard>(
        work_directory,
        !options_.keep_work_directory);
    sample_directory = write_single_step_inference_sample(
        work_directory,
        input,
        raw_structure_data,
        static_molecule_metadata,
        prepared.structure_matrices,
        prepared.one_electron_reference_energy,
        nuclear_repulsion_energy);
    prediction_directory = work_directory / "prediction";
    fs::create_directories(prediction_directory);
  }

  const auto backend_start_time = std::chrono::steady_clock::now();
  std::optional<OnnxRuntimeInferenceOutput> onnx_output;
  try {
    if (options_.backend == "python_jax") {
      run_python_inference(options_, work_directory, prediction_directory);
    } else if (options_.backend == "onnx_runtime") {
      onnx_output = run_onnx_runtime_inference(
          options_,
          input,
          raw_structure_data,
          static_molecule_metadata,
          prepared,
          nuclear_repulsion_energy,
          prediction_directory);
    } else {
      throw std::invalid_argument("unsupported DeepVBH inference backend: " + options_.backend);
    }
  } catch (...) {
    if (work_directory_guard != nullptr) {
      work_directory_guard->keep();
    }
    throw;
  }
  prediction.backend_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - backend_start_time).count();
  prediction.python_wall_time_seconds =
      options_.backend == "python_jax" ? prediction.backend_wall_time_seconds : 0.0;

  try {
    const std::size_t orbital_matrix_size =
        static_cast<std::size_t>(input.orbital_preparation_input.n_orbitals) *
        static_cast<std::size_t>(input.orbital_preparation_input.n_basis_functions);
    if (options_.backend == "onnx_runtime") {
      if (!onnx_output.has_value()) {
        throw std::runtime_error("missing ONNX Runtime prediction outputs");
      }
      prediction.predicted_two_electron_hamiltonian_matrix =
          std::move(onnx_output->two_electron_hamiltonian);
      prediction.predicted_dense_orbital_residual =
          std::move(onnx_output->dense_orbital_residual);
      prediction.predicted_sparse_orbital_residual = dense_to_sparse_orbital_coefficients(
          input.orbital_preparation_input,
          prediction.predicted_dense_orbital_residual);
      prediction.predicted_updated_dense_orbital_coefficients =
          sparse_to_dense_orbital_coefficients(
              input.orbital_preparation_input,
              input.orbital_preparation_input.orbital_value_table);
      for (std::size_t index = 0;
           index < prediction.predicted_updated_dense_orbital_coefficients.size();
           ++index) {
        prediction.predicted_updated_dense_orbital_coefficients[index] +=
            prediction.predicted_dense_orbital_residual[index];
      }
      prediction.predicted_updated_orbital_value_table =
          input.orbital_preparation_input.orbital_value_table;
      for (std::size_t index = 0;
           index < prediction.predicted_updated_orbital_value_table.size();
           ++index) {
        prediction.predicted_updated_orbital_value_table[index] +=
            prediction.predicted_sparse_orbital_residual[index];
      }
      prediction.reference_energy_residual = onnx_output->reference_energy_residual;
      if (onnx_output->has_total_energy) {
        prediction.has_predicted_total_energy = true;
        prediction.predicted_total_energy = onnx_output->total_energy;
      }
    } else {
      prediction.predicted_two_electron_hamiltonian_matrix = read_binary_vector<double>(
          prediction_directory / "predicted_two_electron_hamiltonian_f64.bin",
          prepared.structure_matrices.n_structures * prepared.structure_matrices.n_structures);
      prediction.predicted_dense_orbital_residual = read_binary_vector<double>(
          prediction_directory / "predicted_dense_orbital_residual_f64.bin",
          orbital_matrix_size);
      const auto sparse_residual = read_binary_vector_if_exists<double>(
          prediction_directory / "predicted_orbital_residual_f64.bin",
          orbital_matrix_size);
      if (sparse_residual.has_value()) {
        prediction.predicted_sparse_orbital_residual = *sparse_residual;
      } else {
        prediction.predicted_sparse_orbital_residual = dense_to_sparse_orbital_coefficients(
            input.orbital_preparation_input,
            prediction.predicted_dense_orbital_residual);
      }
      const auto updated_dense = read_binary_vector_if_exists<double>(
          prediction_directory / "predicted_updated_dense_orbital_coefficients_f64.bin",
          orbital_matrix_size);
      if (updated_dense.has_value()) {
        prediction.predicted_updated_dense_orbital_coefficients = *updated_dense;
      } else {
        prediction.predicted_updated_dense_orbital_coefficients =
            sparse_to_dense_orbital_coefficients(
                input.orbital_preparation_input,
                input.orbital_preparation_input.orbital_value_table);
        for (std::size_t index = 0;
             index < prediction.predicted_updated_dense_orbital_coefficients.size();
             ++index) {
          prediction.predicted_updated_dense_orbital_coefficients[index] +=
              prediction.predicted_dense_orbital_residual[index];
        }
      }
      const auto updated_sparse = read_binary_vector_if_exists<double>(
          prediction_directory / "predicted_updated_orbital_value_table_f64.bin",
          orbital_matrix_size);
      if (updated_sparse.has_value()) {
        prediction.predicted_updated_orbital_value_table = *updated_sparse;
      } else {
        prediction.predicted_updated_orbital_value_table =
            input.orbital_preparation_input.orbital_value_table;
        for (std::size_t index = 0;
             index < prediction.predicted_updated_orbital_value_table.size();
             ++index) {
          prediction.predicted_updated_orbital_value_table[index] +=
              prediction.predicted_sparse_orbital_residual[index];
        }
      }
      prediction.reference_energy_residual =
          read_binary_scalar(prediction_directory / "predicted_reference_energy_residual_f64.bin");
      const auto predicted_total_energy = read_binary_scalar_if_exists(
          prediction_directory / "predicted_total_energy_f64.bin");
      if (predicted_total_energy.has_value()) {
        prediction.has_predicted_total_energy = true;
        prediction.predicted_total_energy = *predicted_total_energy;
      }
    }
  } catch (...) {
    if (work_directory_guard != nullptr) {
      work_directory_guard->keep();
    }
    throw;
  }

  prediction.structure_matrices = prepared.structure_matrices;
  if (prediction.structure_matrices.hamiltonian_matrix.size() !=
      prediction.predicted_two_electron_hamiltonian_matrix.size()) {
    if (work_directory_guard != nullptr) {
      work_directory_guard->keep();
    }
    throw std::runtime_error("predicted two-electron Hamiltonian size mismatch");
  }
  for (std::size_t index = 0;
       index < prediction.structure_matrices.hamiltonian_matrix.size();
       ++index) {
    prediction.structure_matrices.hamiltonian_matrix[index] =
        prediction.structure_matrices.one_electron_hamiltonian_matrix[index] +
        prediction.predicted_two_electron_hamiltonian_matrix[index];
  }
  prediction.work_directory = work_directory;
  prediction.sample_directory = sample_directory;
  prediction.total_wall_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start_time).count();
  return prediction;
}

}  // namespace xmvb::vb
