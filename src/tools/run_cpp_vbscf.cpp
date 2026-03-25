#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/scf/deepvbh_onnx_direct_final_optimizer.hpp"
#include "vb/scf/deepvbh_onnx_hybrid_optimizer.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/scf/cpp_vb_scf_optimizer.hpp"

namespace {

namespace fs = std::filesystem;

std::string escape_json_string(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char character : value) {
    switch (character) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += character;
        break;
    }
  }
  return escaped;
}

std::string sanitize_path_component(const std::string& input) {
  std::string sanitized;
  sanitized.reserve(input.size());
  bool previous_was_separator = false;
  for (const unsigned char character : input) {
    if (std::isalnum(character) != 0) {
      sanitized.push_back(static_cast<char>(character));
      previous_was_separator = false;
      continue;
    }
    if (!previous_was_separator) {
      sanitized.push_back('_');
      previous_was_separator = true;
    }
  }
  while (!sanitized.empty() && sanitized.front() == '_') {
    sanitized.erase(sanitized.begin());
  }
  while (!sanitized.empty() && sanitized.back() == '_') {
    sanitized.pop_back();
  }
  if (sanitized.empty()) {
    return "sample";
  }
  return sanitized;
}

std::string format_index_name(
    const std::string& prefix,
    int index) {
  std::ostringstream stream;
  stream << prefix << '_' << std::setw(6) << std::setfill('0') << index;
  return stream.str();
}

int get_sparse_coefficient_count(
    const xmvb::vb::OrbitalPreparationInput& orbital_preparation_input,
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
    const xmvb::vb::OrbitalPreparationInput& orbital_preparation_input) {
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

std::size_t packed_active_two_electron_size(int n_active_orbitals) {
  if (n_active_orbitals <= 0) {
    throw std::invalid_argument("n_active_orbitals must be positive");
  }
  return static_cast<std::size_t>(
             xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
                 n_active_orbitals - 1,
                 n_active_orbitals - 1,
                 n_active_orbitals - 1,
                 n_active_orbitals - 1)) +
      1;
}

std::vector<double> build_coulomb_diagonal_matrix(
    const std::vector<double>& packed_active_two_electron_integrals,
    int n_active_orbitals) {
  const std::size_t expected_size =
      packed_active_two_electron_size(n_active_orbitals);
  if (packed_active_two_electron_integrals.size() != expected_size) {
    throw std::runtime_error("packed_active_two_electron_integrals size mismatch");
  }
  std::vector<double> matrix(
      static_cast<std::size_t>(n_active_orbitals) *
          static_cast<std::size_t>(n_active_orbitals),
      0.0);
  for (int column = 0; column < n_active_orbitals; ++column) {
    for (int row = 0; row < n_active_orbitals; ++row) {
      const int packed_index =
          xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
              row,
              row,
              column,
              column);
      matrix[static_cast<std::size_t>(column) * n_active_orbitals + row] =
          packed_active_two_electron_integrals[static_cast<std::size_t>(packed_index)];
    }
  }
  return matrix;
}

std::vector<double> build_exchange_diagonal_matrix(
    const std::vector<double>& packed_active_two_electron_integrals,
    int n_active_orbitals) {
  const std::size_t expected_size =
      packed_active_two_electron_size(n_active_orbitals);
  if (packed_active_two_electron_integrals.size() != expected_size) {
    throw std::runtime_error("packed_active_two_electron_integrals size mismatch");
  }
  std::vector<double> matrix(
      static_cast<std::size_t>(n_active_orbitals) *
          static_cast<std::size_t>(n_active_orbitals),
      0.0);
  for (int column = 0; column < n_active_orbitals; ++column) {
    for (int row = 0; row < n_active_orbitals; ++row) {
      const int packed_index =
          xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
              row,
              column,
              column,
              row);
      matrix[static_cast<std::size_t>(column) * n_active_orbitals + row] =
          packed_active_two_electron_integrals[static_cast<std::size_t>(packed_index)];
    }
  }
  return matrix;
}

struct StructurePairTopology {
  int n_active_beta_electrons = 0;
  int n_open_shell_electrons = 0;
  std::vector<int> structure_pair_orbital_indices;
  std::vector<std::uint8_t> structure_pair_mask;
  std::vector<int> structure_open_shell_orbitals;
  std::vector<std::uint8_t> structure_open_shell_mask;
};

std::vector<int> build_local_active_structure_pair_indices(
    const StructurePairTopology& topology,
    int active_start,
    int n_active_orbitals) {
  std::vector<int> local_indices = topology.structure_pair_orbital_indices;
  for (int& orbital_index : local_indices) {
    const int local_index = orbital_index - active_start;
    if (local_index < 0 || local_index >= n_active_orbitals) {
      throw std::runtime_error(
          "structure pair orbital index is outside the active orbital window");
    }
    orbital_index = local_index;
  }
  return local_indices;
}

std::vector<int> build_local_active_open_shell_orbitals(
    const StructurePairTopology& topology,
    int active_start,
    int n_active_orbitals) {
  std::vector<int> local_orbitals = topology.structure_open_shell_orbitals;
  for (int& orbital_index : local_orbitals) {
    const int local_index = orbital_index - active_start;
    if (local_index < 0 || local_index >= n_active_orbitals) {
      throw std::runtime_error(
          "open-shell orbital index is outside the active orbital window");
    }
    orbital_index = local_index;
  }
  return local_orbitals;
}

std::vector<double> build_structure_occupancy(
    const xmvb::vb::RawStructureData& raw_structure_data,
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

StructurePairTopology build_structure_pair_topology(
    const xmvb::vb::RawStructureData& raw_structure_data) {
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

fs::path reserve_sample_directory(
    const fs::path& dataset_root,
    const fs::path& input_file_path) {
  fs::create_directories(dataset_root);
  const std::string base_name =
      sanitize_path_component(input_file_path.stem().string());
  fs::path candidate = dataset_root / base_name;
  if (!fs::exists(candidate)) {
    return candidate;
  }
  for (int suffix = 1; suffix < 1000000; ++suffix) {
    std::ostringstream stream;
    stream << base_name << '_' << std::setw(3) << std::setfill('0') << suffix;
    candidate = dataset_root / stream.str();
    if (!fs::exists(candidate)) {
      return candidate;
    }
  }
  throw std::runtime_error("failed to allocate a unique sample directory");
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

class AcceptedIterationTraceDatasetWriter {
public:
  AcceptedIterationTraceDatasetWriter(
      const fs::path& dataset_root,
      const std::string& input_file_path,
      const xmvb::vb::CppVbInputLoadResult& load_result,
      const xmvb::vb::CppVbScfOptimizerOptions& optimizer_options)
      : dataset_root_(fs::absolute(dataset_root)),
        sample_dir_(reserve_sample_directory(dataset_root_, fs::path(input_file_path))),
        static_dir_(sample_dir_ / "static"),
        steps_dir_(sample_dir_ / "steps"),
        source_input_path_(fs::absolute(fs::path(input_file_path)).string()),
        algorithm_name_(xmvb::vb::vb_scf_algorithm_name(optimizer_options.algorithm)),
        optimizer_backend_name_(
            xmvb::vb::cpp_vb_scf_optimizer_backend_name(optimizer_options.backend)),
        n_structures_(load_result.raw_structure_data.n_structures),
        n_total_electrons_(load_result.raw_structure_data.n_total_electrons),
        n_active_electrons_(load_result.raw_structure_data.n_active_electrons),
        spin_multiplicity_(load_result.raw_structure_data.spin_multiplicity),
        differentiable_parameter_count_(static_cast<int>(
            collect_differentiable_parameter_indices(
                load_result.input.orbital_preparation_input).size())),
        n_atoms_(load_result.static_molecule_metadata.n_atoms),
        n_shells_(load_result.static_molecule_metadata.n_shells),
        n_basis_functions_(load_result.input.orbital_preparation_input.n_basis_functions),
        n_orbitals_(load_result.input.orbital_preparation_input.n_orbitals),
        n_active_orbitals_(load_result.input.orbital_preparation_input.n_active_orbitals),
        nuclear_repulsion_energy_(load_result.nuclear_repulsion_energy) {
    fs::create_directories(static_dir_);
    fs::create_directories(steps_dir_);
    write_static_files(load_result);
    write_metadata_file(nullptr);
  }

  void write_accepted_iteration(
      const xmvb::vb::CppVbScfAcceptedIterationSnapshot& snapshot) {
    const std::string step_name =
        format_index_name("step", snapshot.accepted_iteration_index);
    const fs::path step_dir = steps_dir_ / step_name;
    fs::create_directories(step_dir);

    const std::vector<double> coulomb_diagonal_matrix =
        build_coulomb_diagonal_matrix(
            snapshot.packed_active_two_electron_integrals,
            n_active_orbitals_);
    const std::vector<double> exchange_diagonal_matrix =
        build_exchange_diagonal_matrix(
            snapshot.packed_active_two_electron_integrals,
            n_active_orbitals_);

    write_binary_container(
        step_dir / "orbital_value_table_f64.bin",
        snapshot.orbital_value_table);
    write_binary_container(
        step_dir / "active_orbital_overlap_matrix_f64.bin",
        snapshot.active_orbital_overlap_matrix);
    write_binary_container(
        step_dir / "active_one_electron_integrals_f64.bin",
        snapshot.active_one_electron_integrals);
    write_binary_container(
        step_dir / "packed_active_two_electron_integrals_f64.bin",
        snapshot.packed_active_two_electron_integrals);
    write_binary_container(
        step_dir / "coulomb_diagonal_matrix_f64.bin",
        coulomb_diagonal_matrix);
    write_binary_container(
        step_dir / "exchange_diagonal_matrix_f64.bin",
        exchange_diagonal_matrix);
    write_binary_container(
        step_dir / "overlap_matrix_f64.bin",
        snapshot.structure_matrices.overlap_matrix);
    write_binary_container(
        step_dir / "hamiltonian_matrix_f64.bin",
        snapshot.structure_matrices.hamiltonian_matrix);
    write_binary_container(
        step_dir / "one_electron_hamiltonian_matrix_f64.bin",
        snapshot.structure_matrices.one_electron_hamiltonian_matrix);
    write_binary_container(
        step_dir / "sparse_orbital_energy_gradient_f64.bin",
        snapshot.sparse_orbital_energy_gradient);
    write_binary_container(
        step_dir / "sparse_orbital_reference_energy_gradient_f64.bin",
        snapshot.sparse_orbital_reference_energy_gradient);
    write_binary_container(
        step_dir / "average_structure_overlap_f64.bin",
        std::vector<double>{snapshot.average_structure_overlap});

    std::ostringstream metadata_stream;
    metadata_stream << "{\n"
                    << "  \"accepted_iteration_index\": "
                    << snapshot.accepted_iteration_index << ",\n"
                    << "  \"total_energy\": " << std::setprecision(17)
                    << snapshot.total_energy << ",\n"
                    << "  \"one_electron_reference_energy\": " << std::setprecision(17)
                    << snapshot.one_electron_reference_energy << ",\n"
                    << "  \"average_structure_overlap\": " << std::setprecision(17)
                    << snapshot.average_structure_overlap << ",\n"
                    << "  \"n_orbitals\": " << n_orbitals_ << ",\n"
                    << "  \"n_active_orbitals\": " << n_active_orbitals_ << ",\n"
                    << "  \"n_basis_functions\": " << n_basis_functions_ << ",\n"
                    << "  \"n_structures\": " << n_structures_ << ",\n"
                    << "  \"active_feature_matrix_layout\": ["
                    << n_active_orbitals_ << ", " << n_active_orbitals_ << "],\n"
                    << "  \"packed_active_two_electron_integral_count\": "
                    << snapshot.packed_active_two_electron_integrals.size() << ",\n"
                    << "  \"n_differentiable_parameters\": "
                    << differentiable_parameter_count_ << "\n"
                    << "}\n";
    write_text_file(step_dir / "metadata.json", metadata_stream.str());
    ++accepted_iteration_count_;
  }

  void finalize(
      const xmvb::vb::CppVbScfOptimizerResult& result) {
    write_metadata_file(&result);
  }

  const fs::path& sample_directory() const {
    return sample_dir_;
  }

private:
  void write_static_files(
      const xmvb::vb::CppVbInputLoadResult& load_result) const {
    const auto& static_molecule_metadata = load_result.static_molecule_metadata;
    const auto structure_occupancy = build_structure_occupancy(
        load_result.raw_structure_data,
        n_orbitals_);
    const auto structure_pair_topology =
        build_structure_pair_topology(load_result.raw_structure_data);
    const int active_start =
        (n_total_electrons_ - n_active_electrons_) / 2;
    const auto local_structure_pair_indices =
        build_local_active_structure_pair_indices(
            structure_pair_topology,
            active_start,
            n_active_orbitals_);
    const auto local_structure_open_shell_orbitals =
        build_local_active_open_shell_orbitals(
            structure_pair_topology,
            active_start,
            n_active_orbitals_);
    const auto differentiable_parameter_indices =
        collect_differentiable_parameter_indices(load_result.input.orbital_preparation_input);
    write_binary_container(
        static_dir_ / "raw_structure_orbitals_i32.bin",
        load_result.raw_structure_data.raw_structure_orbitals);
    write_binary_container(
        static_dir_ / "structure_occupancy_f64.bin",
        structure_occupancy);
    write_binary_container(
        static_dir_ / "structure_pair_orbital_indices_i32.bin",
        structure_pair_topology.structure_pair_orbital_indices);
    write_binary_container(
        static_dir_ / "structure_pair_active_orbital_indices_i32.bin",
        local_structure_pair_indices);
    write_binary_container(
        static_dir_ / "structure_pair_mask_u8.bin",
        structure_pair_topology.structure_pair_mask);
    write_binary_container(
        static_dir_ / "structure_open_shell_orbitals_i32.bin",
        structure_pair_topology.structure_open_shell_orbitals);
    write_binary_container(
        static_dir_ / "structure_open_shell_active_orbitals_i32.bin",
        local_structure_open_shell_orbitals);
    write_binary_container(
        static_dir_ / "structure_open_shell_mask_u8.bin",
        structure_pair_topology.structure_open_shell_mask);
    write_binary_container(
        static_dir_ / "atomic_numbers_i32.bin",
        static_molecule_metadata.atomic_numbers);
    write_binary_container(
        static_dir_ / "atomic_coordinates_f64.bin",
        static_molecule_metadata.atomic_coordinates);
    write_binary_container(
        static_dir_ / "shell_to_atom_i32.bin",
        static_molecule_metadata.shell_to_atom);
    write_binary_container(
        static_dir_ / "shell_angular_momenta_i32.bin",
        static_molecule_metadata.shell_angular_momenta);
    write_binary_container(
        static_dir_ / "shell_n_primitives_i32.bin",
        static_molecule_metadata.shell_n_primitives);
    write_binary_container(
        static_dir_ / "shell_ao_starts_i32.bin",
        static_molecule_metadata.shell_ao_starts);
    write_binary_container(
        static_dir_ / "shell_ao_counts_i32.bin",
        static_molecule_metadata.shell_ao_counts);
    write_binary_container(
        static_dir_ / "ao_to_atom_i32.bin",
        static_molecule_metadata.ao_to_atom);
    write_binary_container(
        static_dir_ / "ao_to_shell_i32.bin",
        static_molecule_metadata.ao_to_shell);
    write_binary_container(
        static_dir_ / "ao_angular_momenta_i32.bin",
        static_molecule_metadata.ao_angular_momenta);
    write_binary_container(
        static_dir_ / "ao_shell_local_indices_i32.bin",
        static_molecule_metadata.ao_shell_local_indices);
    write_binary_container(
        static_dir_ / "ao_cartesian_exponents_i32.bin",
        static_molecule_metadata.ao_cartesian_exponents);
    write_binary_container(
        static_dir_ / "orbital_basis_index_table_i32.bin",
        load_result.input.orbital_preparation_input.orbital_basis_index_table);
    write_binary_container(
        static_dir_ / "orbital_basis_counts_i32.bin",
        load_result.input.orbital_preparation_input.orbital_basis_counts);
    write_binary_container(
        static_dir_ / "original_orbital_basis_counts_i32.bin",
        load_result.input.orbital_preparation_input.original_orbital_basis_counts);
    write_binary_container(
        static_dir_ / "differentiable_parameter_indices_i32.bin",
        differentiable_parameter_indices);

    std::ostringstream metadata_stream;
    metadata_stream << "{\n"
                    << "  \"index_base\": 0,\n"
                    << "  \"n_atoms\": " << n_atoms_ << ",\n"
                    << "  \"n_shells\": " << n_shells_ << ",\n"
                    << "  \"n_basis_functions\": " << n_basis_functions_ << ",\n"
                    << "  \"n_orbitals\": " << n_orbitals_ << ",\n"
                    << "  \"n_active_orbitals\": " << n_active_orbitals_ << ",\n"
                    << "  \"active_orbital_start_index\": " << active_start << ",\n"
                    << "  \"n_differentiable_parameters\": "
                    << differentiable_parameter_count_ << ",\n"
                    << "  \"raw_structure_orbitals_layout\": ["
                    << n_structures_ << ", " << n_total_electrons_ << "],\n"
                    << "  \"structure_occupancy_layout\": ["
                    << n_structures_ << ", " << n_orbitals_ << "],\n"
                    << "  \"structure_pair_orbital_indices_layout\": ["
                    << n_structures_ << ", "
                    << structure_pair_topology.n_active_beta_electrons << ", 2],\n"
                    << "  \"structure_pair_active_orbital_indices_layout\": ["
                    << n_structures_ << ", "
                    << structure_pair_topology.n_active_beta_electrons << ", 2],\n"
                    << "  \"structure_pair_mask_layout\": ["
                    << n_structures_ << ", "
                    << structure_pair_topology.n_active_beta_electrons << "],\n"
                    << "  \"structure_open_shell_orbitals_layout\": ["
                    << n_structures_ << ", "
                    << structure_pair_topology.n_open_shell_electrons << "],\n"
                    << "  \"structure_open_shell_active_orbitals_layout\": ["
                    << n_structures_ << ", "
                    << structure_pair_topology.n_open_shell_electrons << "],\n"
                    << "  \"structure_open_shell_mask_layout\": ["
                    << n_structures_ << ", "
                    << structure_pair_topology.n_open_shell_electrons << "],\n"
                    << "  \"atomic_coordinate_unit\": \"bohr\",\n"
                    << "  \"atomic_coordinates_layout\": ["
                    << n_atoms_ << ", 3],\n"
                    << "  \"ao_cartesian_exponents_layout\": ["
                    << n_basis_functions_ << ", 3],\n"
                    << "  \"shell_ordering\": \"basis_file_shell_order\",\n"
                    << "  \"ao_ordering\": \"cartesian_shell_local_order\",\n"
                    << "  \"ao_cartesian_exponents_columns\": [\"lx\", \"ly\", \"lz\"]\n"
                    << "}\n";
    write_text_file(static_dir_ / "metadata.json", metadata_stream.str());
  }

  void write_metadata_file(
      const xmvb::vb::CppVbScfOptimizerResult* result) const {
    std::ostringstream metadata_stream;
    metadata_stream << "{\n"
                    << "  \"format_version\": 5,\n"
                    << "  \"status\": \"" << (result == nullptr ? "running" : "completed")
                    << "\",\n"
                    << "  \"sample_name\": \"" << escape_json_string(sample_dir_.filename().string())
                    << "\",\n"
                    << "  \"source_input_path\": \"" << escape_json_string(source_input_path_)
                    << "\",\n"
                    << "  \"optimizer_backend\": \"" << optimizer_backend_name_ << "\",\n"
                    << "  \"algorithm\": \"" << algorithm_name_ << "\",\n"
                    << "  \"n_structures\": " << n_structures_ << ",\n"
                    << "  \"n_total_electrons\": " << n_total_electrons_ << ",\n"
                    << "  \"n_active_electrons\": " << n_active_electrons_ << ",\n"
                    << "  \"spin_multiplicity\": " << spin_multiplicity_ << ",\n"
                    << "  \"n_atoms\": " << n_atoms_ << ",\n"
                    << "  \"n_shells\": " << n_shells_ << ",\n"
                    << "  \"n_basis_functions\": " << n_basis_functions_ << ",\n"
                    << "  \"n_orbitals\": " << n_orbitals_ << ",\n"
                    << "  \"n_active_orbitals\": " << n_active_orbitals_ << ",\n"
                    << "  \"n_differentiable_parameters\": "
                    << differentiable_parameter_count_ << ",\n"
                    << "  \"nuclear_repulsion_energy\": " << std::setprecision(17)
                    << nuclear_repulsion_energy_ << ",\n"
                    << "  \"accepted_iteration_count\": " << accepted_iteration_count_;
    if (result != nullptr) {
      metadata_stream << ",\n"
                      << "  \"converged\": " << (result->converged ? "true" : "false") << ",\n"
                      << "  \"termination_reason\": \""
                      << escape_json_string(result->termination_reason) << "\",\n"
                      << "  \"n_iterations\": " << result->n_iterations << ",\n"
                      << "  \"initial_total_energy\": " << std::setprecision(17)
                      << result->initial_total_energy << ",\n"
                      << "  \"final_total_energy\": " << std::setprecision(17)
                      << result->final_total_energy << ",\n"
                      << "  \"final_gradient_inf_norm\": " << std::setprecision(17)
                      << result->final_gradient_inf_norm << ",\n"
                      << "  \"total_wall_time_seconds\": " << std::setprecision(17)
                      << result->total_wall_time_seconds;
    }
    metadata_stream << "\n}\n";
    write_text_file(sample_dir_ / "metadata.json", metadata_stream.str());
  }

  fs::path dataset_root_;
  fs::path sample_dir_;
  fs::path static_dir_;
  fs::path steps_dir_;
  std::string source_input_path_;
  std::string algorithm_name_;
  std::string optimizer_backend_name_;
  int n_structures_ = 0;
  int n_total_electrons_ = 0;
  int n_active_electrons_ = 0;
  int spin_multiplicity_ = 1;
  int differentiable_parameter_count_ = 0;
  int n_atoms_ = 0;
  int n_shells_ = 0;
  int n_basis_functions_ = 0;
  int n_orbitals_ = 0;
  int n_active_orbitals_ = 0;
  double nuclear_repulsion_energy_ = 0.0;
  int accepted_iteration_count_ = 0;
};

void apply_optimizer_backend_argument(
    const std::string& backend_name,
    xmvb::vb::CppVbScfOptimizerOptions* options) {
  if (backend_name == "legacy_fortran") {
    if (!xmvb::vb::cpp_vb_scf_optimizer_backend_supported(
            xmvb::vb::CppVbScfOptimizerBackend::LegacyFortran)) {
      throw std::invalid_argument(
          "legacy_fortran backend is not enabled in this build");
    }
    options->backend = xmvb::vb::CppVbScfOptimizerBackend::LegacyFortran;
    return;
  }
  if (backend_name == "lbfgspp") {
    options->backend = xmvb::vb::CppVbScfOptimizerBackend::Lbfgspp;
    return;
  }
  if (backend_name == "deepvbh_onnx") {
    if (!xmvb::vb::cpp_vb_scf_optimizer_backend_supported(
            xmvb::vb::CppVbScfOptimizerBackend::DeepVBHOnnx)) {
      throw std::invalid_argument(
          "deepvbh_onnx backend is not enabled in this build");
    }
    options->backend = xmvb::vb::CppVbScfOptimizerBackend::DeepVBHOnnx;
    return;
  }
  if (backend_name == "deepvbh_onnx_direct_final") {
    if (!xmvb::vb::cpp_vb_scf_optimizer_backend_supported(
            xmvb::vb::CppVbScfOptimizerBackend::DeepVBHOnnxDirectFinal)) {
      throw std::invalid_argument(
          "deepvbh_onnx_direct_final backend is not enabled in this build");
    }
    options->backend = xmvb::vb::CppVbScfOptimizerBackend::DeepVBHOnnxDirectFinal;
    return;
  }
  throw std::invalid_argument("invalid optimizer backend: " + backend_name);
}

bool parse_bool_argument(const std::string& value) {
  if (value == "true" || value == "1" || value == "yes") {
    return true;
  }
  if (value == "false" || value == "0" || value == "no") {
    return false;
  }
  throw std::invalid_argument("invalid boolean value: " + value);
}

void apply_algorithm_argument(
    const std::string& algorithm_name,
    xmvb::vb::CppVbScfOptimizerOptions* options) {
  if (algorithm_name == "original") {
    options->algorithm = xmvb::vb::VBSCFAlgorithm::Original;
    return;
  }
  throw std::invalid_argument("invalid algorithm: " + algorithm_name);
}

void print_usage() {
  std::cerr << "usage: run_cpp_vbscf <input.xmi> "
               "[--optimizer-backend lbfgspp";
  if (xmvb::vb::cpp_vb_scf_optimizer_backend_supported(
          xmvb::vb::CppVbScfOptimizerBackend::LegacyFortran)) {
    std::cerr << "|legacy_fortran";
  }
  if (xmvb::vb::cpp_vb_scf_optimizer_backend_supported(
          xmvb::vb::CppVbScfOptimizerBackend::DeepVBHOnnx)) {
    std::cerr << "|deepvbh_onnx";
  }
  if (xmvb::vb::cpp_vb_scf_optimizer_backend_supported(
          xmvb::vb::CppVbScfOptimizerBackend::DeepVBHOnnxDirectFinal)) {
    std::cerr << "|deepvbh_onnx_direct_final";
  }
  std::cerr << "] [--algorithm original]"
               " [--dump-trace-dir <dataset_root>]"
               " [--onnx-model <path>]"
               " [--ml-initial-step-scale <value>]"
               " [--ml-minimum-step-scale <value>]"
               " [--ml-step-shrink-factor <value>]"
               " [--ml-max-backtracks <count>]"
               " [--ml-fallback-max-iterations <count>]"
               " [--ml-keep-work-dir true|false]"
               " [--ml-work-dir <path>]\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || ((argc - 2) % 2 != 0)) {
    print_usage();
    return 1;
  }

  const std::string input_path = argv[1];
  const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(input_path);
  const auto& input = load_result.input;

  xmvb::vb::CppVbScfOptimizerOptions options;
  options.max_iterations = 256;
  options.gradient_tolerance = 2.0e-3;
  options.initial_step_size = 1.0e20;
  options.minimum_step_size = 1.0e-20;
  options.history_size = 100;
  xmvb::vb::DeepVBHOnnxHybridOptimizerOptions deepvbh_options;
  xmvb::vb::DeepVBHOnnxDirectFinalOptimizerOptions deepvbh_direct_options;
  deepvbh_options.optimizer_options = options;
  deepvbh_options.inference_options.repo_root = fs::current_path();
  deepvbh_options.inference_options.backend = "onnx_runtime";
  deepvbh_direct_options.optimizer_options = options;
  deepvbh_direct_options.inference_options.repo_root = fs::current_path();
  deepvbh_direct_options.inference_options.backend = "onnx_runtime";

  std::string dump_trace_dir;
  for (int argument_index = 2; argument_index < argc; argument_index += 2) {
    const std::string argument_name = argv[argument_index];
    const std::string argument_value = argv[argument_index + 1];
    try {
      if (argument_name == "--optimizer-backend") {
        apply_optimizer_backend_argument(argument_value, &options);
      } else if (argument_name == "--algorithm") {
        apply_algorithm_argument(argument_value, &options);
      } else if (argument_name == "--dump-trace-dir") {
        dump_trace_dir = argument_value;
      } else if (argument_name == "--onnx-model") {
        deepvbh_options.inference_options.onnx_model_path = argument_value;
        deepvbh_direct_options.inference_options.onnx_model_path = argument_value;
      } else if (argument_name == "--ml-initial-step-scale") {
        deepvbh_options.initial_step_scale = std::stod(argument_value);
        deepvbh_direct_options.initial_step_scale = std::stod(argument_value);
      } else if (argument_name == "--ml-minimum-step-scale") {
        deepvbh_options.minimum_step_scale = std::stod(argument_value);
        deepvbh_direct_options.minimum_step_scale = std::stod(argument_value);
      } else if (argument_name == "--ml-step-shrink-factor") {
        deepvbh_options.step_shrink_factor = std::stod(argument_value);
        deepvbh_direct_options.step_shrink_factor = std::stod(argument_value);
      } else if (argument_name == "--ml-max-backtracks") {
        deepvbh_options.max_backtracks = std::stoi(argument_value);
        deepvbh_direct_options.max_backtracks = std::stoi(argument_value);
      } else if (argument_name == "--ml-fallback-max-iterations") {
        deepvbh_options.exact_fallback_max_iterations = std::stoi(argument_value);
        deepvbh_direct_options.exact_fallback_max_iterations = std::stoi(argument_value);
      } else if (argument_name == "--ml-keep-work-dir") {
        deepvbh_options.inference_options.keep_work_directory =
            parse_bool_argument(argument_value);
        deepvbh_direct_options.inference_options.keep_work_directory =
            parse_bool_argument(argument_value);
      } else if (argument_name == "--ml-work-dir") {
        deepvbh_options.inference_options.work_directory = argument_value;
        deepvbh_direct_options.inference_options.work_directory = argument_value;
      } else {
        std::cerr << "unknown argument: " << argument_name << '\n';
        print_usage();
        return 1;
      }
    } catch (const std::exception& error) {
      std::cerr << error.what() << '\n';
      return 1;
    }
  }

  std::shared_ptr<AcceptedIterationTraceDatasetWriter> trace_writer;
  if (!dump_trace_dir.empty()) {
    if (options.algorithm != xmvb::vb::VBSCFAlgorithm::Original) {
      throw std::invalid_argument(
          "--dump-trace-dir currently supports only --algorithm original");
    }
    trace_writer = std::make_shared<AcceptedIterationTraceDatasetWriter>(
        dump_trace_dir,
        input_path,
        load_result,
        options);
    options.retain_accepted_iteration_trace = false;
    options.accepted_iteration_callback =
        [trace_writer](const xmvb::vb::CppVbScfAcceptedIterationSnapshot& snapshot) {
          trace_writer->write_accepted_iteration(snapshot);
        };
  }

  deepvbh_options.optimizer_options = options;
  deepvbh_options.inference_options.algorithm = options.algorithm;
  deepvbh_direct_options.optimizer_options = options;
  deepvbh_direct_options.inference_options.algorithm = options.algorithm;
  xmvb::vb::CppVbScfOptimizerResult result;
  if (options.backend == xmvb::vb::CppVbScfOptimizerBackend::DeepVBHOnnx) {
    if (deepvbh_options.inference_options.onnx_model_path.empty()) {
      throw std::invalid_argument(
          "--onnx-model is required for --optimizer-backend deepvbh_onnx");
    }
    xmvb::vb::DeepVBHOnnxHybridOptimizer optimizer(deepvbh_options);
    result = optimizer.optimize(
        input,
        load_result.raw_structure_data,
        load_result.static_molecule_metadata,
        load_result.nuclear_repulsion_energy);
  } else if (
      options.backend == xmvb::vb::CppVbScfOptimizerBackend::DeepVBHOnnxDirectFinal) {
    if (deepvbh_direct_options.inference_options.onnx_model_path.empty()) {
      throw std::invalid_argument(
          "--onnx-model is required for --optimizer-backend deepvbh_onnx_direct_final");
    }
    xmvb::vb::DeepVBHOnnxDirectFinalOptimizer optimizer(deepvbh_direct_options);
    result = optimizer.optimize(
        input,
        load_result.raw_structure_data,
        load_result.static_molecule_metadata,
        load_result.nuclear_repulsion_energy);
  } else {
    xmvb::vb::CppVbScfOptimizer optimizer(options);
    result = optimizer.optimize(input, load_result.nuclear_repulsion_energy);
  }
  if (trace_writer != nullptr) {
    trace_writer->finalize(result);
  }

  const double initial_electronic_energy =
      result.initial_total_energy - load_result.nuclear_repulsion_energy;
  const double final_electronic_energy =
      result.final_total_energy - load_result.nuclear_repulsion_energy;

  std::cout << "optimizer_backend = "
            << xmvb::vb::cpp_vb_scf_optimizer_backend_name(options.backend) << '\n';
  if (options.backend == xmvb::vb::CppVbScfOptimizerBackend::DeepVBHOnnx) {
    std::cout << "onnx_model = "
              << fs::absolute(deepvbh_options.inference_options.onnx_model_path).string()
              << '\n';
  } else if (
      options.backend == xmvb::vb::CppVbScfOptimizerBackend::DeepVBHOnnxDirectFinal) {
    std::cout << "onnx_model = "
              << fs::absolute(deepvbh_direct_options.inference_options.onnx_model_path).string()
              << '\n';
  }
  std::cout << "algorithm = "
            << xmvb::vb::vb_scf_algorithm_name(options.algorithm) << '\n';
  std::cout << "converged = " << (result.converged ? "true" : "false") << '\n';
  std::cout << "termination_reason = " << result.termination_reason << '\n';
  std::cout << "iterations = " << result.n_iterations << '\n';
  if (trace_writer != nullptr) {
    std::cout << "trace_sample_dir = " << trace_writer->sample_directory().string() << '\n';
  }
  std::cout << "nuclear_repulsion_energy = " << std::setprecision(12)
            << load_result.nuclear_repulsion_energy << '\n';
  std::cout << "initial_total_energy = " << std::setprecision(12)
            << result.initial_total_energy << '\n';
  std::cout << "initial_electronic_energy = " << std::setprecision(12)
            << initial_electronic_energy << '\n';
  std::cout << "final_total_energy = " << std::setprecision(12)
            << result.final_total_energy << '\n';
  std::cout << "final_electronic_energy = " << std::setprecision(12)
            << final_electronic_energy << '\n';
  std::cout << "final_gradient_inf_norm = " << std::setprecision(12)
            << result.final_gradient_inf_norm << '\n';
  std::cout << "input_total_wall_time_seconds = " << std::setprecision(12)
            << load_result.total_seconds << '\n';
  std::cout << "runtime_total_wall_time_seconds = " << std::setprecision(12)
            << load_result.runtime_timings.total_seconds << '\n';
  std::cout << "runtime_read_input_seconds = " << std::setprecision(12)
            << load_result.runtime_timings.read_input_seconds << '\n';
  std::cout << "runtime_vb_input_seconds = " << std::setprecision(12)
            << load_result.runtime_timings.vb_input_seconds << '\n';
  std::cout << "runtime_libcint_buffer_setup_seconds = " << std::setprecision(12)
            << load_result.runtime_timings.libcint_buffer_setup_seconds << '\n';
  std::cout << "runtime_hf_setup_seconds = " << std::setprecision(12)
            << load_result.runtime_timings.hf_setup_seconds << '\n';
  std::cout << "runtime_vbprep_seconds = " << std::setprecision(12)
            << load_result.runtime_timings.vbprep_seconds << '\n';
  std::cout << "runtime_vbguess_seconds = " << std::setprecision(12)
            << load_result.runtime_timings.vbguess_seconds << '\n';
  std::cout << "runtime_one_electron_integrals_seconds = " << std::setprecision(12)
            << load_result.runtime_timings.one_electron_integrals_seconds << '\n';
  std::cout << "runtime_two_electron_integrals_seconds = " << std::setprecision(12)
            << load_result.runtime_timings.two_electron_integrals_seconds << '\n';
  std::cout << "runtime_output_copy_seconds = " << std::setprecision(12)
            << load_result.runtime_timings.output_copy_seconds << '\n';
  std::cout << "structure_expansion_seconds = " << std::setprecision(12)
            << load_result.structure_expansion_seconds << '\n';
  std::cout << "total_wall_time_seconds = " << std::setprecision(12)
            << result.total_wall_time_seconds << '\n';

  return result.converged ? 0 : 2;
}
