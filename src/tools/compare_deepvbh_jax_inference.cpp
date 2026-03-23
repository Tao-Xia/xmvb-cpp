#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/model/deepvbh_jax_inference_runner.hpp"
#include "vb/scf/cpp_vb_scf_evaluator.hpp"

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

double max_abs_difference(
    const std::vector<double>& left,
    const std::vector<double>& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("vector size mismatch when computing max_abs_difference");
  }
  double maximum = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    maximum = std::max(maximum, std::abs(left[index] - right[index]));
  }
  return maximum;
}

double l2_norm(const std::vector<double>& values) {
  double squared_norm = 0.0;
  for (const double value : values) {
    squared_norm += value * value;
  }
  return std::sqrt(squared_norm);
}

double lower_triangle_rmse(
    const std::vector<double>& left,
    const std::vector<double>& right,
    int n) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("vector size mismatch when computing RMSE");
  }
  double squared_error_sum = 0.0;
  std::size_t count = 0;
  for (int column = 0; column < n; ++column) {
    for (int row = column; row < n; ++row) {
      const std::size_t index =
          static_cast<std::size_t>(column) * n + row;
      const double error = left[index] - right[index];
      squared_error_sum += error * error;
      ++count;
    }
  }
  return count == 0 ? 0.0 : std::sqrt(squared_error_sum / static_cast<double>(count));
}

std::vector<double> subtract_vectors(
    const std::vector<double>& left,
    const std::vector<double>& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("vector size mismatch when subtracting");
  }
  std::vector<double> result(left.size(), 0.0);
  for (std::size_t index = 0; index < left.size(); ++index) {
    result[index] = left[index] - right[index];
  }
  return result;
}

void print_usage() {
  std::cerr
      << "usage: compare_deepvbh_jax_inference <input.xmi>"
      << " --checkpoint <path>"
      << " [--repo-root <path>]"
      << " [--python-exe <path>]"
      << " [--backend python_jax|onnx_runtime]"
      << " [--onnx-model <path>]"
      << " [--work-dir <path>]"
      << " [--device gpu|cpu]"
      << " [--dtype float32|float64]"
      << " [--keep-work-dir true|false]\n";
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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4 || ((argc - 2) % 2 != 0)) {
    print_usage();
    return 1;
  }

  const std::string input_path = argv[1];
  xmvb::vb::DeepVBHJaxInferenceOptions ml_options;

  for (int argument_index = 2; argument_index < argc; argument_index += 2) {
    const std::string argument_name = argv[argument_index];
    const std::string argument_value = argv[argument_index + 1];
    if (argument_name == "--checkpoint") {
      ml_options.checkpoint_path = argument_value;
    } else if (argument_name == "--repo-root") {
      ml_options.repo_root = argument_value;
    } else if (argument_name == "--python-exe") {
      ml_options.python_executable = argument_value;
    } else if (argument_name == "--backend") {
      ml_options.backend = argument_value;
    } else if (argument_name == "--onnx-model") {
      ml_options.onnx_model_path = argument_value;
    } else if (argument_name == "--work-dir") {
      ml_options.work_directory = argument_value;
    } else if (argument_name == "--device") {
      ml_options.device = argument_value;
    } else if (argument_name == "--dtype") {
      ml_options.dtype = argument_value;
    } else if (argument_name == "--keep-work-dir") {
      ml_options.keep_work_directory = parse_bool_argument(argument_value);
    } else {
      std::cerr << "unknown argument: " << argument_name << '\n';
      print_usage();
      return 1;
    }
  }

  if (ml_options.backend == "python_jax" && ml_options.checkpoint_path.empty()) {
    std::cerr << "--checkpoint is required for --backend python_jax\n";
    print_usage();
    return 1;
  }
  if (ml_options.backend == "onnx_runtime" && ml_options.onnx_model_path.empty()) {
    std::cerr << "--onnx-model is required for --backend onnx_runtime\n";
    print_usage();
    return 1;
  }

  try {
    const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(input_path);
    xmvb::vb::CppVbScfEvaluator exact_evaluator(xmvb::vb::VbScfAlgorithm::Original);
    const auto exact_result = exact_evaluator.evaluate(
        load_result.input,
        load_result.nuclear_repulsion_energy);

    xmvb::vb::DeepVBHJaxInferenceRunner runner(ml_options);
    const auto prediction = runner.predict(
        load_result.input,
        load_result.raw_structure_data,
        load_result.static_molecule_metadata,
        load_result.nuclear_repulsion_energy);

    xmvb::core::GeneralizedEigensolver generalized_eigensolver;
    const auto ml_eigen_result = generalized_eigensolver.solve(
        prediction.structure_matrices.hamiltonian_matrix,
        prediction.structure_matrices.overlap_matrix,
        prediction.structure_matrices.n_structures);
    const double ml_cpp_total_energy =
        prediction.one_electron_reference_energy +
        prediction.reference_energy_residual +
        ml_eigen_result.eigenvalues.front() +
        load_result.nuclear_repulsion_energy;
    const double ml_python_cpp_total_energy_abs_diff =
        prediction.has_predicted_total_energy
            ? std::abs(prediction.predicted_total_energy - ml_cpp_total_energy)
            : 0.0;

    const auto exact_two_electron_hamiltonian = subtract_vectors(
        exact_result.structure_matrices.hamiltonian_matrix,
        exact_result.structure_matrices.one_electron_hamiltonian_matrix);

    std::ostringstream output;
    output << "{\n"
           << "  \"input_path\": \"" << escape_json_string(fs::absolute(input_path).string())
           << "\",\n"
           << "  \"checkpoint\": \"" << escape_json_string(
                  ml_options.checkpoint_path.empty()
                      ? std::string()
                      : fs::absolute(ml_options.checkpoint_path).string())
           << "\",\n"
           << "  \"repo_root\": \"" << escape_json_string(fs::absolute(ml_options.repo_root).string())
           << "\",\n"
           << "  \"backend\": \"" << escape_json_string(prediction.backend) << "\",\n"
           << "  \"python_executable\": \""
           << escape_json_string(fs::absolute(ml_options.python_executable).string()) << "\",\n"
           << "  \"onnx_model\": \"" << escape_json_string(ml_options.onnx_model_path.string())
           << "\",\n"
           << "  \"device\": \"" << escape_json_string(ml_options.device) << "\",\n"
           << "  \"dtype\": \"" << escape_json_string(ml_options.dtype) << "\",\n"
           << "  \"n_structures\": " << prediction.structure_matrices.n_structures << ",\n"
           << "  \"n_active_orbitals\": " << load_result.input.orbital_preparation_input.n_active_orbitals
           << ",\n"
           << "  \"exact_total_energy\": " << std::setprecision(17)
           << exact_result.total_energy << ",\n"
           << "  \"ml_python_total_energy\": ";
    if (prediction.has_predicted_total_energy) {
      output << std::setprecision(17) << prediction.predicted_total_energy;
    } else {
      output << "null";
    }
    output << ",\n"
           << "  \"ml_cpp_total_energy\": " << std::setprecision(17)
           << ml_cpp_total_energy << ",\n"
           << "  \"ml_reference_energy_residual\": " << std::setprecision(17)
           << prediction.reference_energy_residual << ",\n"
           << "  \"predicted_sparse_orbital_residual_norm\": " << std::setprecision(17)
           << l2_norm(prediction.predicted_sparse_orbital_residual) << ",\n"
           << "  \"predicted_dense_orbital_residual_norm\": " << std::setprecision(17)
           << l2_norm(prediction.predicted_dense_orbital_residual) << ",\n"
           << "  \"ml_python_cpp_total_energy_abs_diff\": ";
    if (prediction.has_predicted_total_energy) {
      output << std::setprecision(17) << ml_python_cpp_total_energy_abs_diff;
    } else {
      output << "null";
    }
    output << ",\n"
           << "  \"ml_exact_total_energy_abs_diff\": " << std::setprecision(17)
           << std::abs(ml_cpp_total_energy - exact_result.total_energy) << ",\n"
           << "  \"overlap_input_max_abs_diff\": " << std::setprecision(17)
           << max_abs_difference(
                  prediction.structure_matrices.overlap_matrix,
                  exact_result.structure_matrices.overlap_matrix)
           << ",\n"
           << "  \"one_electron_structure_max_abs_diff\": " << std::setprecision(17)
           << max_abs_difference(
                  prediction.structure_matrices.one_electron_hamiltonian_matrix,
                  exact_result.structure_matrices.one_electron_hamiltonian_matrix)
           << ",\n"
           << "  \"two_electron_structure_max_abs_diff\": " << std::setprecision(17)
           << max_abs_difference(
                  prediction.predicted_two_electron_hamiltonian_matrix,
                  exact_two_electron_hamiltonian)
           << ",\n"
           << "  \"two_electron_structure_lower_triangle_rmse\": " << std::setprecision(17)
           << lower_triangle_rmse(
                  prediction.predicted_two_electron_hamiltonian_matrix,
                  exact_two_electron_hamiltonian,
                  prediction.structure_matrices.n_structures)
           << ",\n"
           << "  \"preparation_wall_time_seconds\": " << std::setprecision(17)
           << prediction.preparation_wall_time_seconds << ",\n"
           << "  \"backend_wall_time_seconds\": " << std::setprecision(17)
           << prediction.backend_wall_time_seconds << ",\n"
           << "  \"python_wall_time_seconds\": " << std::setprecision(17)
           << prediction.python_wall_time_seconds << ",\n"
           << "  \"total_inference_wall_time_seconds\": " << std::setprecision(17)
           << prediction.total_wall_time_seconds;
    if (ml_options.keep_work_directory) {
      output << ",\n"
             << "  \"work_directory\": \""
             << escape_json_string(prediction.work_directory.string()) << "\",\n"
             << "  \"sample_directory\": \""
             << escape_json_string(prediction.sample_directory.string()) << "\"\n";
    } else {
      output << "\n";
    }
    output << "}\n";

    std::cout << output.str();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
