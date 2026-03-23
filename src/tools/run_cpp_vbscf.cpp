#include <cctype>
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

    write_binary_container(
        step_dir / "orbital_value_table_f64.bin",
        snapshot.orbital_value_table);
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

    std::ostringstream metadata_stream;
    metadata_stream << "{\n"
                    << "  \"accepted_iteration_index\": "
                    << snapshot.accepted_iteration_index << ",\n"
                    << "  \"total_energy\": " << std::setprecision(17)
                    << snapshot.total_energy << ",\n"
                    << "  \"one_electron_reference_energy\": " << std::setprecision(17)
                    << snapshot.one_electron_reference_energy << ",\n"
                    << "  \"n_orbitals\": " << n_orbitals_ << ",\n"
                    << "  \"n_basis_functions\": " << n_basis_functions_ << ",\n"
                    << "  \"n_structures\": " << n_structures_ << ",\n"
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
    const auto differentiable_parameter_indices =
        collect_differentiable_parameter_indices(load_result.input.orbital_preparation_input);
    write_binary_container(
        static_dir_ / "raw_structure_orbitals_i32.bin",
        load_result.raw_structure_data.raw_structure_orbitals);
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
                    << "  \"n_differentiable_parameters\": "
                    << differentiable_parameter_count_ << ",\n"
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
    options->algorithm = xmvb::vb::VbScfAlgorithm::Original;
    return;
  }
  if (algorithm_name == "biorthogonal") {
    options->algorithm = xmvb::vb::VbScfAlgorithm::Biorthogonal;
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
  std::cerr << "] [--algorithm original|biorthogonal]"
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
    if (options.algorithm != xmvb::vb::VbScfAlgorithm::Original) {
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
