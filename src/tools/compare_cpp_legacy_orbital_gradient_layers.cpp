#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>

#include "runtime_c/local_runtime_api_internal.h"
#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/eigen_matrix_storage_utils.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/orbital/active_space_matrix_backpropagator.hpp"
#include "vb/orbital/active_space_two_electron_backpropagator.hpp"
#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"
#include "vb/scf/cpp_orbital_gradient_evaluator.hpp"
#include "vb/vbscf_algorithm.hpp"

extern "C" {
int xmvb_cpp_vbprep(
    hf_info hf,
    inp_info inp_str,
    vb_info vb_str,
    para_info par_str,
    int print_level);
void legacy_prepare_gradient_modules_(vb_info vb_str);
void legacy_sync_structure_results_(vb_info vb_str);
void orbprep_(vb_info vb_str);
void cal_g11_(vb_info vb_str);
void cal_f11_(vb_info vb_str);
void cal_e11_(double* e11, vb_info vb_str);
void xtra_pt0_(vb_info vb_str);
void grad_rdm5_(
    double* col,
    double* g22,
    double* q22,
    double* grda,
    double* grdv,
    double* e22,
    vb_info vb_str);
void grdori_(
    double* g22,
    double* q22,
    double* grda,
    double* grdv,
    double* grdbas,
    double* grad,
    double* weight,
    vb_info vb_str);
void legacy_build_grdaux_(
    double* grda,
    double* grdv,
    double* grdaux,
    vb_info vb_str);
void hamhd_(vb_info vb_str);
void hamov_(vb_info vb_str);
}

extern "C" int eigencalc_c_symbol(
    double* energy,
    double* col,
    double* hvb,
    double* svb,
    double enuc,
    int n,
    vb_info vb_str) asm("eigencalc");

namespace fs = std::filesystem;

namespace {

struct Options {
  std::string input_path;
  std::string executable_path;
  std::string orbital_value_table_bin_path;
  int top_count = 12;
};

struct LegacyGradientLayers {
  double total_energy = 0.0;
  double e11 = 0.0;
  double e22 = 0.0;
  std::vector<double> packed_gradient;
  Eigen::MatrixXd grdaux_matrix;
  Eigen::MatrixXd slot_gradient_matrix;
  Eigen::MatrixXd g22_matrix;
  Eigen::MatrixXd q22_matrix;
  Eigen::MatrixXd grda_matrix;
  Eigen::MatrixXd grdv_matrix;
};

struct DifferenceSummary {
  double max_abs = 0.0;
  double rms = 0.0;
  int count = 0;
};

struct SlotDifference {
  double abs_diff = 0.0;
  double current_value = 0.0;
  double legacy_value = 0.0;
  int orbital_index = -1;
  int slot_index = -1;
  int basis_function_index = -1;
};

struct AuxiliaryDifference {
  double abs_diff = 0.0;
  double current_value = 0.0;
  double legacy_value = 0.0;
  int active_orbital_index = -1;
  int basis_function_index = -1;
};

void print_usage() {
  std::cerr
      << "usage: compare_cpp_legacy_orbital_gradient_layers <input.xmi> "
      << "[--orbital-value-table-bin path] [--top N]\n";
}

Options parse_arguments(int argc, char** argv) {
  if (argc < 2 || ((argc - 2) % 2 != 0)) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }

  Options options;
  options.input_path = argv[1];
  options.executable_path = argv[0];
  for (int argument_index = 2; argument_index < argc; argument_index += 2) {
    const std::string argument_name = argv[argument_index];
    const std::string argument_value = argv[argument_index + 1];
    if (argument_name == "--orbital-value-table-bin") {
      options.orbital_value_table_bin_path = argument_value;
      continue;
    }
    if (argument_name == "--top") {
      options.top_count = std::stoi(argument_value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument_name);
  }

  if (options.top_count <= 0) {
    throw std::invalid_argument("--top must be positive");
  }
  return options;
}

std::vector<double> read_binary_double_vector(
    const std::string& file_path,
    std::size_t expected_size) {
  std::ifstream input_stream(file_path, std::ios::binary);
  if (!input_stream) {
    throw std::runtime_error("failed to open orbital-value-table binary: " + file_path);
  }

  input_stream.seekg(0, std::ios::end);
  const std::streamoff file_size = input_stream.tellg();
  input_stream.seekg(0, std::ios::beg);
  if (file_size < 0) {
    throw std::runtime_error("failed to stat orbital-value-table binary: " + file_path);
  }
  if (file_size !=
      static_cast<std::streamoff>(expected_size * sizeof(double))) {
    throw std::runtime_error(
        "orbital-value-table binary size mismatch for " + file_path);
  }

  std::vector<double> values(expected_size, 0.0);
  input_stream.read(
      reinterpret_cast<char*>(values.data()),
      static_cast<std::streamsize>(expected_size * sizeof(double)));
  if (!input_stream) {
    throw std::runtime_error("failed to read orbital-value-table binary: " + file_path);
  }
  return values;
}

Eigen::MatrixXd build_active_pair_gradient_matrix(
    const std::vector<double>& packed_active_two_electron_gradient,
    int n_active_orbitals) {
  const std::size_t n_active_pairs =
      xmvb::to_size(n_active_orbitals) * (n_active_orbitals + 1) / 2;
  Eigen::MatrixXd active_pair_gradient_matrix =
      Eigen::MatrixXd::Zero(
          static_cast<Eigen::Index>(n_active_pairs),
          static_cast<Eigen::Index>(n_active_pairs));

  for (int row_first = 0; row_first < n_active_orbitals; ++row_first) {
    for (int row_second = 0; row_second <= row_first; ++row_second) {
      const std::size_t row_pair_index =
          xmvb::to_size(row_first) * (row_first + 1) / 2 + row_second;
      for (int column_first = 0; column_first < n_active_orbitals; ++column_first) {
        for (int column_second = 0; column_second <= column_first; ++column_second) {
          const std::size_t column_pair_index =
              xmvb::to_size(column_first) * (column_first + 1) / 2 + column_second;
          const int packed_index =
              (row_pair_index >= column_pair_index)
                  ? xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
                        row_first,
                        row_second,
                        column_first,
                        column_second)
                  : xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
                        column_first,
                        column_second,
                        row_first,
                        row_second);
          double value =
              packed_active_two_electron_gradient[xmvb::to_size(packed_index)];
          if (row_pair_index == column_pair_index) {
            value *= 2.0;
          }
          active_pair_gradient_matrix(
              static_cast<Eigen::Index>(row_pair_index),
              static_cast<Eigen::Index>(column_pair_index)) = value;
        }
      }
    }
  }

  return active_pair_gradient_matrix;
}

std::vector<double> build_ri_active_pair_factor_gradient(
    const std::vector<double>& packed_active_two_electron_gradient,
    const xmvb::vb::ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    int n_active_orbitals) {
  if (active_space_two_electron_result.representation !=
      xmvb::vb::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity) {
    throw std::invalid_argument("RI active-pair-factor gradient requires an RI forward result");
  }
  const std::size_t n_active_pairs =
      xmvb::to_size(n_active_orbitals) * (n_active_orbitals + 1) / 2;
  const std::size_t expected_factor_size =
      xmvb::to_size(active_space_two_electron_result.n_auxiliary_functions) *
      n_active_pairs;
  if (xmvb::to_size(active_space_two_electron_result.ri_active_pair_factors.size()) !=
          expected_factor_size ||
      active_space_two_electron_result.ri_active_pair_factors.rows() !=
          active_space_two_electron_result.n_auxiliary_functions ||
      active_space_two_electron_result.ri_active_pair_factors.cols() !=
          static_cast<Eigen::Index>(n_active_pairs)) {
    throw std::invalid_argument("RI active-pair-factor buffer size mismatch");
  }

  const Eigen::MatrixXd active_pair_gradient_matrix =
      build_active_pair_gradient_matrix(
          packed_active_two_electron_gradient,
          n_active_orbitals);
  const Eigen::MatrixXd ri_active_pair_factor_gradient =
      active_space_two_electron_result.ri_active_pair_factors *
      active_pair_gradient_matrix;
  return xmvb::vb::flatten_matrix_column_major(ri_active_pair_factor_gradient);
}

Eigen::MatrixXd build_current_total_active_auxiliary_gradient(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::CppActiveSpaceGradientResult& active_space_gradient_result) {
  const int n_basis_functions = input.orbital_preparation_input.n_basis_functions;
  const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
  const int n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) / 2;

  xmvb::vb::ActiveSpaceMatrixBackpropagator active_space_matrix_backpropagator;
  xmvb::vb::ActiveSpaceTwoElectronBackpropagator active_space_two_electron_backpropagator;
  const auto active_space_matrix_backpropagation_result =
      active_space_matrix_backpropagator.backpropagate(
          active_space_gradient_result.active_orbital_overlap_gradient,
          active_space_gradient_result.active_one_electron_gradient,
          input.orbital_preparation_input.active_orbital_overlap_matrix,
          active_space_gradient_result.ao_effective_one_electron_result.ao_effective_h1e,
          active_space_gradient_result.orbital_preparation_result.auxiliary_orbital_matrix,
          n_basis_functions,
          n_inactive_doubly_occupied_orbitals,
          n_active_orbitals);
  const auto active_space_two_electron_backpropagation_result =
      active_space_gradient_result.active_space_two_electron_result.representation ==
              xmvb::vb::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity
          ? active_space_two_electron_backpropagator.backpropagate(
                build_ri_active_pair_factor_gradient(
                    active_space_gradient_result.packed_active_two_electron_gradient,
                    active_space_gradient_result.active_space_two_electron_result,
                    n_active_orbitals),
                input,
                active_space_gradient_result.orbital_preparation_result,
                active_space_gradient_result.active_space_two_electron_result,
                n_basis_functions,
                n_inactive_doubly_occupied_orbitals,
                n_active_orbitals)
          : active_space_two_electron_backpropagator.backpropagate(
                active_space_gradient_result.packed_active_two_electron_gradient,
                input.ao_integral_input.ao_two_electron_integral_values,
                input.ao_integral_input.ao_two_electron_integral_indices,
                active_space_gradient_result.orbital_preparation_result,
                n_basis_functions,
                n_inactive_doubly_occupied_orbitals,
                n_active_orbitals);

  if (active_space_matrix_backpropagation_result.active_auxiliary_orbital_gradient.rows() !=
          n_basis_functions ||
      active_space_matrix_backpropagation_result.active_auxiliary_orbital_gradient.cols() !=
          n_active_orbitals ||
      active_space_two_electron_backpropagation_result.active_auxiliary_orbital_gradient.rows() !=
          n_basis_functions ||
      active_space_two_electron_backpropagation_result.active_auxiliary_orbital_gradient.cols() !=
          n_active_orbitals) {
    throw std::runtime_error("current active auxiliary gradient size mismatch");
  }

  // This is the exact object identified in the derivation note: the AO-by-active
  // coefficient gradient of the projected active auxiliary block `T_a`.
  Eigen::MatrixXd total_active_auxiliary_gradient =
      active_space_matrix_backpropagation_result.active_auxiliary_orbital_gradient;
  total_active_auxiliary_gradient.noalias() +=
      active_space_two_electron_backpropagation_result.active_auxiliary_orbital_gradient;
  return total_active_auxiliary_gradient;
}

int get_sparse_coefficient_count(
    const xmvb::vb::OrbitalPreparationInput& orbital_preparation_input,
    int orbital_index) {
  const bool have_original_counts =
      orbital_preparation_input.original_orbital_basis_counts.size() ==
      static_cast<std::size_t>(orbital_preparation_input.n_orbitals);
  const int explicit_count = have_original_counts
      ? orbital_preparation_input.original_orbital_basis_counts[static_cast<std::size_t>(
            orbital_index)]
      : orbital_preparation_input.orbital_basis_counts[static_cast<std::size_t>(orbital_index)];
  if (explicit_count > 1) {
    return explicit_count;
  }
  if (explicit_count == 1) {
    return 1;
  }

  int coefficient_count = 0;
  while (coefficient_count < orbital_preparation_input.n_basis_functions) {
    const int basis_function_index =
        orbital_preparation_input.orbital_basis_index_table[static_cast<std::size_t>(
            orbital_index * orbital_preparation_input.n_basis_functions +
            coefficient_count)];
    if (basis_function_index == 0) {
      break;
    }
    ++coefficient_count;
  }
  return coefficient_count;
}

std::vector<double> pack_gradient_in_legacy_parameter_order(
    const std::vector<double>& sparse_gradient,
    const xmvb::vb::OrbitalPreparationInput& orbital_preparation_input) {
  if (sparse_gradient.size() != orbital_preparation_input.orbital_value_table.size()) {
    throw std::invalid_argument("sparse gradient size does not match orbital value table");
  }

  const int n_inactive_doubly_occupied_orbitals =
      (orbital_preparation_input.n_total_electrons -
       orbital_preparation_input.n_active_electrons) / 2;
  const int n_occupied_orbitals =
      n_inactive_doubly_occupied_orbitals +
      orbital_preparation_input.n_active_orbitals;

  std::vector<double> packed_gradient;
  for (int orbital_index = 0; orbital_index < n_occupied_orbitals; ++orbital_index) {
    const int coefficient_count =
        get_sparse_coefficient_count(orbital_preparation_input, orbital_index);
    if (coefficient_count <= 1) {
      continue;
    }
    for (int coefficient_slot = 0; coefficient_slot < coefficient_count; ++coefficient_slot) {
      const std::size_t flat_index = static_cast<std::size_t>(
          orbital_index * orbital_preparation_input.n_basis_functions +
          coefficient_slot);
      packed_gradient.push_back(sparse_gradient[flat_index]);
    }
  }
  return packed_gradient;
}

Eigen::MatrixXd build_slot_gradient_matrix(
    const std::vector<double>& sparse_gradient,
    const xmvb::vb::OrbitalPreparationInput& orbital_preparation_input) {
  if (sparse_gradient.size() != orbital_preparation_input.orbital_value_table.size()) {
    throw std::invalid_argument("sparse gradient size does not match orbital value table");
  }

  Eigen::MatrixXd slot_gradient_matrix =
      Eigen::MatrixXd::Zero(
          orbital_preparation_input.n_basis_functions,
          orbital_preparation_input.n_orbitals);
  for (int orbital_index = 0; orbital_index < orbital_preparation_input.n_orbitals; ++orbital_index) {
    const int coefficient_count =
        get_sparse_coefficient_count(orbital_preparation_input, orbital_index);
    if (coefficient_count <= 1) {
      continue;
    }
    for (int coefficient_slot = 0; coefficient_slot < coefficient_count; ++coefficient_slot) {
      const std::size_t flat_index = static_cast<std::size_t>(
          orbital_index * orbital_preparation_input.n_basis_functions +
          coefficient_slot);
      slot_gradient_matrix(coefficient_slot, orbital_index) = sparse_gradient[flat_index];
    }
  }
  return slot_gradient_matrix;
}

void overwrite_orbital_value_table(
    const std::vector<double>& orbital_values,
    xmvb::vb::CppVbInput* input) {
  if (input == nullptr) {
    throw std::invalid_argument("input must not be null");
  }
  if (orbital_values.size() != input->orbital_preparation_input.orbital_value_table.size()) {
    throw std::invalid_argument("orbital override size mismatch for C++ input");
  }
  input->orbital_preparation_input.orbital_value_table = orbital_values;
}

void overwrite_runtime_orbital_value_table(
    const std::vector<double>& orbital_values,
    XmvbCppRuntimeHandle* runtime_handle) {
  if (runtime_handle == nullptr || runtime_handle->vb_wavefunction == nullptr ||
      runtime_handle->vb_wavefunction->dv == nullptr) {
    throw std::invalid_argument("runtime handle is incomplete for orbital override");
  }
  const std::size_t expected_size = static_cast<std::size_t>(
      runtime_handle->vb_wavefunction->nb) * runtime_handle->vb_wavefunction->nor;
  if (orbital_values.size() != expected_size) {
    throw std::invalid_argument("orbital override size mismatch for runtime handle");
  }
  std::copy(
      orbital_values.begin(),
      orbital_values.end(),
      runtime_handle->vb_wavefunction->dv);
}

void ensure_legacy_structure_workspace_allocated(vb_info vb_wavefunction) {
  if (vb_wavefunction == nullptr) {
    throw std::invalid_argument("legacy vb_wavefunction must not be null");
  }

  const std::size_t structure_matrix_size =
      xmvb::to_size(vb_wavefunction->nstr) * vb_wavefunction->nstr;
  const std::size_t eigenvector_size =
      xmvb::to_size(vb_wavefunction->nstr) * vb_wavefunction->nsav;

  // The legacy orbital-gradient driver assumes these structure-level buffers
  // are already materialized before `orbprep`/`hamhd`/`hamov` are entered.
  if (vb_wavefunction->hvb == nullptr) {
    vb_wavefunction->hvb = static_cast<double*>(
        std::malloc(structure_matrix_size * sizeof(double)));
  }
  if (vb_wavefunction->svb == nullptr) {
    vb_wavefunction->svb = static_cast<double*>(
        std::malloc(structure_matrix_size * sizeof(double)));
  }
  if (vb_wavefunction->hvb_1e == nullptr) {
    vb_wavefunction->hvb_1e = static_cast<double*>(
        std::malloc(structure_matrix_size * sizeof(double)));
  }
  if (vb_wavefunction->col == nullptr) {
    vb_wavefunction->col = static_cast<double*>(
        std::malloc(eigenvector_size * sizeof(double)));
  }
  if (vb_wavefunction->hvb == nullptr ||
      vb_wavefunction->svb == nullptr ||
      vb_wavefunction->hvb_1e == nullptr ||
      vb_wavefunction->col == nullptr) {
    throw std::runtime_error("failed to allocate legacy structure workspace");
  }
}

void expect_runtime_ok(
    int status,
    const char* stage,
    const char* error_message) {
  if (status == 0) {
    return;
  }
  throw std::runtime_error(
      std::string(stage) + " failed: " +
      (error_message != nullptr && error_message[0] != '\0'
           ? std::string(error_message)
           : std::string("unknown runtime error")));
}

bool legacy_compare_debug_logging_enabled() {
  const char* value = std::getenv("XMVB_CPP_COMPARE_DEBUG");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void log_legacy_compare_stage(const char* stage) {
  if (!legacy_compare_debug_logging_enabled()) {
    return;
  }
  std::cerr << "compare_cpp_legacy_orbital_gradient_layers stage=" << stage << '\n';
  std::cerr.flush();
}

DifferenceSummary summarize_matrix_difference(
    const Eigen::Ref<const Eigen::MatrixXd>& current_matrix,
    const Eigen::Ref<const Eigen::MatrixXd>& legacy_matrix) {
  if (current_matrix.rows() != legacy_matrix.rows() ||
      current_matrix.cols() != legacy_matrix.cols()) {
    throw std::invalid_argument("matrix dimensions differ");
  }

  DifferenceSummary summary;
  summary.count = static_cast<int>(current_matrix.size());
  double sum_squares = 0.0;
  for (int column = 0; column < current_matrix.cols(); ++column) {
    for (int row = 0; row < current_matrix.rows(); ++row) {
      const double difference = current_matrix(row, column) - legacy_matrix(row, column);
      summary.max_abs = std::max(summary.max_abs, std::abs(difference));
      sum_squares += difference * difference;
    }
  }
  if (summary.count > 0) {
    summary.rms = std::sqrt(sum_squares / static_cast<double>(summary.count));
  }
  return summary;
}

std::vector<AuxiliaryDifference> collect_top_auxiliary_differences(
    const Eigen::Ref<const Eigen::MatrixXd>& current_auxiliary_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& legacy_auxiliary_gradient,
    int top_count) {
  if (current_auxiliary_gradient.rows() != legacy_auxiliary_gradient.rows() ||
      current_auxiliary_gradient.cols() != legacy_auxiliary_gradient.cols()) {
    throw std::invalid_argument("auxiliary gradient dimensions differ");
  }

  std::vector<AuxiliaryDifference> differences;
  differences.reserve(
      xmvb::to_size(current_auxiliary_gradient.rows()) * current_auxiliary_gradient.cols());
  for (int active_orbital_index = 0;
       active_orbital_index < current_auxiliary_gradient.cols();
       ++active_orbital_index) {
    for (int basis_function_index = 0;
         basis_function_index < current_auxiliary_gradient.rows();
         ++basis_function_index) {
      AuxiliaryDifference difference;
      difference.current_value =
          current_auxiliary_gradient(basis_function_index, active_orbital_index);
      difference.legacy_value =
          legacy_auxiliary_gradient(basis_function_index, active_orbital_index);
      difference.abs_diff =
          std::abs(difference.current_value - difference.legacy_value);
      difference.active_orbital_index = active_orbital_index;
      difference.basis_function_index = basis_function_index;
      differences.push_back(difference);
    }
  }

  std::sort(
      differences.begin(),
      differences.end(),
      [](const AuxiliaryDifference& left, const AuxiliaryDifference& right) {
        if (left.abs_diff != right.abs_diff) {
          return left.abs_diff > right.abs_diff;
        }
        if (left.active_orbital_index != right.active_orbital_index) {
          return left.active_orbital_index < right.active_orbital_index;
        }
        return left.basis_function_index < right.basis_function_index;
      });
  if (static_cast<int>(differences.size()) > top_count) {
    differences.resize(static_cast<std::size_t>(top_count));
  }
  return differences;
}

LegacyGradientLayers evaluate_legacy_gradient_layers(
    const Options& options,
    const std::vector<double>& orbital_override) {
  XmvbCppRuntimeHandle* runtime_handle = nullptr;
  char error_message[1024] = {0};
  log_legacy_compare_stage("runtime_create");
  expect_runtime_ok(
      xmvb_cpp_runtime_create(
          &runtime_handle,
          error_message,
          sizeof(error_message)),
      "xmvb_cpp_runtime_create",
      error_message);

  try {
    const fs::path runtime_output_stem =
        fs::temp_directory_path() /
        (fs::path(options.input_path).stem().string() + "_legacy_grad_layers");
    log_legacy_compare_stage("load_input");
    expect_runtime_ok(
        xmvb_cpp_runtime_load_input(
            runtime_handle,
            const_cast<char*>(options.input_path.c_str()),
            options.executable_path.c_str(),
            error_message,
            sizeof(error_message)),
        "xmvb_cpp_runtime_load_input",
        error_message);
    log_legacy_compare_stage("initialize_wavefunction");
    expect_runtime_ok(
        xmvb_cpp_runtime_initialize_wavefunction(
            runtime_handle,
            runtime_output_stem.c_str(),
            error_message,
            sizeof(error_message)),
        "xmvb_cpp_runtime_initialize_wavefunction",
        error_message);
    log_legacy_compare_stage("prepare_libcint_buffers");
    expect_runtime_ok(
        xmvb_cpp_runtime_prepare_libcint_buffers(
            runtime_handle,
            error_message,
            sizeof(error_message)),
        "xmvb_cpp_runtime_prepare_libcint_buffers",
        error_message);
    log_legacy_compare_stage("setup_hf");
    expect_runtime_ok(
        xmvb_cpp_runtime_setup_hf(
            runtime_handle,
            error_message,
            sizeof(error_message)),
        "xmvb_cpp_runtime_setup_hf",
        error_message);
    log_legacy_compare_stage("run_vbprep");
    if (xmvb_cpp_vbprep(
            runtime_handle->hf_wavefunction,
            runtime_handle->input_info,
            runtime_handle->vb_wavefunction,
            runtime_handle->parallel_info,
            runtime_handle->input_info->print_level) != 0) {
      throw std::runtime_error("vbprep failed");
    }
    log_legacy_compare_stage("run_one_electron_integrals");
    expect_runtime_ok(
        xmvb_cpp_runtime_run_one_electron_integrals(
            runtime_handle,
            error_message,
            sizeof(error_message)),
        "xmvb_cpp_runtime_run_one_electron_integrals",
        error_message);
    log_legacy_compare_stage("run_two_electron_integrals");
    expect_runtime_ok(
        xmvb_cpp_runtime_run_two_electron_integrals(
            runtime_handle,
            error_message,
            sizeof(error_message)),
        "xmvb_cpp_runtime_run_two_electron_integrals",
        error_message);
    log_legacy_compare_stage("finalize_integral_storage");
    expect_runtime_ok(
        xmvb_cpp_runtime_finalize_integral_storage(
            runtime_handle,
            error_message,
            sizeof(error_message)),
        "xmvb_cpp_runtime_finalize_integral_storage",
        error_message);

    if (!orbital_override.empty()) {
      log_legacy_compare_stage("overwrite_orbital_value_table");
      overwrite_runtime_orbital_value_table(orbital_override, runtime_handle);
    }

    vb_info vb_wavefunction = runtime_handle->vb_wavefunction;
    if (vb_wavefunction == nullptr) {
      throw std::runtime_error("legacy runtime returned null vb_info");
    }

    log_legacy_compare_stage("allocate_structure_workspace");
    ensure_legacy_structure_workspace_allocated(vb_wavefunction);

    const int n_basis_functions = vb_wavefunction->nb;
    const int n_active_orbitals = vb_wavefunction->nao;
    const std::size_t ao_matrix_size =
        static_cast<std::size_t>(n_basis_functions) * n_basis_functions;
    const std::size_t active_matrix_size =
        static_cast<std::size_t>(n_active_orbitals) * n_active_orbitals;

    LegacyGradientLayers result;
    result.packed_gradient.assign(static_cast<std::size_t>(vb_wavefunction->nvar), 0.0);
    result.grdaux_matrix = Eigen::MatrixXd::Zero(n_basis_functions, n_active_orbitals);
    result.slot_gradient_matrix =
        Eigen::MatrixXd::Zero(n_basis_functions, n_basis_functions);
    result.g22_matrix = Eigen::MatrixXd::Zero(n_basis_functions, n_basis_functions);
    result.q22_matrix = Eigen::MatrixXd::Zero(n_basis_functions, n_basis_functions);
    result.grda_matrix = Eigen::MatrixXd::Zero(n_active_orbitals, n_active_orbitals);
    result.grdv_matrix = Eigen::MatrixXd::Zero(n_basis_functions, n_basis_functions);

    // This mirrors the legacy `gradient_rdm` driver exactly enough to expose
    // the same `G22/Q22/Grda/Grdv/Grdbas/Grad` layers at the current sparse
    // orbital point. The goal is not to reimplement legacy optimization, only
    // to measure where the current C++ pullback first diverges.
    log_legacy_compare_stage("prepare_gradient_modules");
    legacy_prepare_gradient_modules_(vb_wavefunction);
    log_legacy_compare_stage("orbprep");
    orbprep_(vb_wavefunction);
    log_legacy_compare_stage("cal_g11");
    cal_g11_(vb_wavefunction);
    log_legacy_compare_stage("cal_f11");
    cal_f11_(vb_wavefunction);
    log_legacy_compare_stage("cal_e11");
    cal_e11_(&result.e11, vb_wavefunction);
    log_legacy_compare_stage("xtra_pt0");
    xtra_pt0_(vb_wavefunction);
    log_legacy_compare_stage("hamhd");
    hamhd_(vb_wavefunction);
    log_legacy_compare_stage("hamov");
    hamov_(vb_wavefunction);
    log_legacy_compare_stage("sync_structure_results");
    legacy_sync_structure_results_(vb_wavefunction);
    result.e22 = 0.0;
    log_legacy_compare_stage("eigencalc");
    eigencalc_c_symbol(
        &result.e22,
        vb_wavefunction->col,
        vb_wavefunction->hvb,
        vb_wavefunction->svb,
        vb_wavefunction->enuc,
        vb_wavefunction->nstr,
        vb_wavefunction);
    result.total_energy = result.e11 + result.e22;

    std::vector<double> grda(active_matrix_size, 0.0);
    std::vector<double> grdv(ao_matrix_size, 0.0);
    std::vector<double> g22(ao_matrix_size, 0.0);
    std::vector<double> q22(ao_matrix_size, 0.0);
    std::vector<double> grdbas(ao_matrix_size, 0.0);

    for (int state_index = 0; state_index < vb_wavefunction->nsav; ++state_index) {
      log_legacy_compare_stage("grad_rdm5");
      std::fill(g22.begin(), g22.end(), 0.0);
      std::fill(q22.begin(), q22.end(), 0.0);
      std::fill(grdbas.begin(), grdbas.end(), 0.0);

      double state_electronic_energy =
          vb_wavefunction->state_energy[state_index] - vb_wavefunction->enuc;
      double state_weight = vb_wavefunction->wstate[state_index];
      grad_rdm5_(
          vb_wavefunction->col + static_cast<std::size_t>(state_index) * vb_wavefunction->nstr,
          g22.data(),
          q22.data(),
          grda.data(),
          grdv.data(),
          &state_electronic_energy,
          vb_wavefunction);
      std::vector<double> grdaux(static_cast<std::size_t>(n_basis_functions) * n_active_orbitals, 0.0);
      log_legacy_compare_stage("legacy_build_grdaux");
      legacy_build_grdaux_(
          grda.data(),
          grdv.data(),
          grdaux.data(),
          vb_wavefunction);
      const Eigen::Map<const Eigen::MatrixXd> grdaux_matrix(
          grdaux.data(),
          n_basis_functions,
          n_active_orbitals);
      result.grdaux_matrix.noalias() += state_weight * grdaux_matrix;
      log_legacy_compare_stage("grdori");
      grdori_(
          g22.data(),
          q22.data(),
          grda.data(),
          grdv.data(),
          grdbas.data(),
          result.packed_gradient.data(),
          &state_weight,
          vb_wavefunction);
    }

    const Eigen::Map<const Eigen::MatrixXd> g22_matrix(
        g22.data(),
        n_basis_functions,
        n_basis_functions);
    const Eigen::Map<const Eigen::MatrixXd> q22_matrix(
        q22.data(),
        n_basis_functions,
        n_basis_functions);
    const Eigen::Map<const Eigen::MatrixXd> grda_matrix(
        grda.data(),
        n_active_orbitals,
        n_active_orbitals);
    const Eigen::Map<const Eigen::MatrixXd> grdv_matrix(
        grdv.data(),
        n_basis_functions,
        n_basis_functions);
    const Eigen::Map<const Eigen::MatrixXd> grdbas_matrix(
        grdbas.data(),
        n_basis_functions,
        n_basis_functions);

    result.g22_matrix = g22_matrix;
    result.q22_matrix = q22_matrix;
    result.grda_matrix = grda_matrix;
    result.grdv_matrix = grdv_matrix;
    result.slot_gradient_matrix = grdbas_matrix;
    log_legacy_compare_stage("destroy_runtime");
    xmvb_cpp_runtime_destroy(runtime_handle);
    return result;
  } catch (...) {
    xmvb_cpp_runtime_destroy(runtime_handle);
    throw;
  }
}

DifferenceSummary summarize_packed_difference(
    const std::vector<double>& current_gradient,
    const std::vector<double>& legacy_gradient) {
  if (current_gradient.size() != legacy_gradient.size()) {
    throw std::invalid_argument("packed gradient sizes differ");
  }

  DifferenceSummary summary;
  summary.count = static_cast<int>(current_gradient.size());
  double sum_squares = 0.0;
  for (std::size_t index = 0; index < current_gradient.size(); ++index) {
    const double difference = current_gradient[index] - legacy_gradient[index];
    summary.max_abs = std::max(summary.max_abs, std::abs(difference));
    sum_squares += difference * difference;
  }
  if (summary.count > 0) {
    summary.rms = std::sqrt(sum_squares / static_cast<double>(summary.count));
  }
  return summary;
}

DifferenceSummary summarize_slot_difference_for_orbital_range(
    const Eigen::Ref<const Eigen::MatrixXd>& current_slot_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& legacy_slot_gradient,
    int orbital_begin,
    int orbital_end,
    const xmvb::vb::OrbitalPreparationInput& orbital_preparation_input) {
  if (current_slot_gradient.rows() != legacy_slot_gradient.rows() ||
      current_slot_gradient.cols() != legacy_slot_gradient.cols()) {
    throw std::invalid_argument("slot gradient dimensions differ");
  }

  DifferenceSummary summary;
  double sum_squares = 0.0;
  for (int orbital_index = orbital_begin; orbital_index < orbital_end; ++orbital_index) {
    const int coefficient_count =
        get_sparse_coefficient_count(orbital_preparation_input, orbital_index);
    if (coefficient_count <= 1) {
      continue;
    }
    for (int coefficient_slot = 0; coefficient_slot < coefficient_count; ++coefficient_slot) {
      const double difference =
          current_slot_gradient(coefficient_slot, orbital_index) -
          legacy_slot_gradient(coefficient_slot, orbital_index);
      summary.max_abs = std::max(summary.max_abs, std::abs(difference));
      sum_squares += difference * difference;
      ++summary.count;
    }
  }
  if (summary.count > 0) {
    summary.rms = std::sqrt(sum_squares / static_cast<double>(summary.count));
  }
  return summary;
}

std::vector<SlotDifference> collect_top_slot_differences(
    const Eigen::Ref<const Eigen::MatrixXd>& current_slot_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& legacy_slot_gradient,
    const xmvb::vb::OrbitalPreparationInput& orbital_preparation_input,
    int top_count) {
  std::vector<SlotDifference> differences;
  for (int orbital_index = 0; orbital_index < orbital_preparation_input.n_orbitals; ++orbital_index) {
    const int coefficient_count =
        get_sparse_coefficient_count(orbital_preparation_input, orbital_index);
    if (coefficient_count <= 1) {
      continue;
    }
    for (int coefficient_slot = 0; coefficient_slot < coefficient_count; ++coefficient_slot) {
      const double current_value = current_slot_gradient(coefficient_slot, orbital_index);
      const double legacy_value = legacy_slot_gradient(coefficient_slot, orbital_index);
      SlotDifference difference;
      difference.abs_diff = std::abs(current_value - legacy_value);
      difference.current_value = current_value;
      difference.legacy_value = legacy_value;
      difference.orbital_index = orbital_index;
      difference.slot_index = coefficient_slot;
      difference.basis_function_index =
          orbital_preparation_input.orbital_basis_index_table[static_cast<std::size_t>(
              orbital_index * orbital_preparation_input.n_basis_functions +
              coefficient_slot)];
      differences.push_back(difference);
    }
  }

  std::sort(
      differences.begin(),
      differences.end(),
      [](const SlotDifference& left, const SlotDifference& right) {
        if (left.abs_diff != right.abs_diff) {
          return left.abs_diff > right.abs_diff;
        }
        if (left.orbital_index != right.orbital_index) {
          return left.orbital_index < right.orbital_index;
        }
        return left.slot_index < right.slot_index;
      });
  if (static_cast<int>(differences.size()) > top_count) {
    differences.resize(static_cast<std::size_t>(top_count));
  }
  return differences;
}

const char* orbital_block_name(
    int orbital_index,
    int n_inactive_doubly_occupied_orbitals,
    int n_active_orbitals) {
  if (orbital_index < n_inactive_doubly_occupied_orbitals) {
    return "inactive";
  }
  if (orbital_index < n_inactive_doubly_occupied_orbitals + n_active_orbitals) {
    return "active";
  }
  return "virtual";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    log_legacy_compare_stage("main_parse_arguments");
    const Options options = parse_arguments(argc, argv);

    log_legacy_compare_stage("main_load_cpp_input");
    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.ao_integral_source = xmvb::vb::AoIntegralSource::LegacyRuntime;
    load_options.orbital_guess_source = xmvb::vb::OrbitalGuessSource::LegacyRuntime;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(options.input_path, load_options);

    std::vector<double> orbital_override;
    if (!options.orbital_value_table_bin_path.empty()) {
      log_legacy_compare_stage("main_read_orbital_override");
      orbital_override = read_binary_double_vector(
          options.orbital_value_table_bin_path,
          load_result.input.orbital_preparation_input.orbital_value_table.size());
    } else {
      // Compare both pipelines at the same current C++ orbital point rather
      // than at a legacy/runtime-specific guess.
      orbital_override = load_result.input.orbital_preparation_input.orbital_value_table;
    }

    xmvb::vb::CppVbInput input = load_result.input;
    if (!orbital_override.empty()) {
      log_legacy_compare_stage("main_overwrite_cpp_orbitals");
      overwrite_orbital_value_table(orbital_override, &input);
    }

    log_legacy_compare_stage("main_current_active_gradient");
    xmvb::vb::CppActiveSpaceGradientEvaluator active_space_gradient_evaluator(
        xmvb::vb::VBSCFAlgorithm::Original);
    const auto active_space_gradient_result =
        active_space_gradient_evaluator.evaluate(
            input,
            load_result.nuclear_repulsion_energy);
    log_legacy_compare_stage("main_current_total_active_auxiliary_gradient");
    const Eigen::MatrixXd current_total_active_auxiliary_gradient =
        build_current_total_active_auxiliary_gradient(
            input,
            active_space_gradient_result);
    log_legacy_compare_stage("main_current_orbital_gradient");
    xmvb::vb::CppOrbitalGradientEvaluator orbital_gradient_evaluator(
        xmvb::vb::VBSCFAlgorithm::Original);
    const auto orbital_gradient_result =
        orbital_gradient_evaluator.evaluate_without_reference_energy_gradient(
            input,
            active_space_gradient_result);

    log_legacy_compare_stage("main_pack_current_gradient");
    const auto current_packed_gradient =
        pack_gradient_in_legacy_parameter_order(
            orbital_gradient_result.sparse_orbital_energy_gradient,
            input.orbital_preparation_input);
    const Eigen::MatrixXd current_slot_gradient =
        build_slot_gradient_matrix(
            orbital_gradient_result.sparse_orbital_energy_gradient,
            input.orbital_preparation_input);

    log_legacy_compare_stage("main_legacy_gradient_layers");
    const auto legacy_gradient_layers =
        evaluate_legacy_gradient_layers(options, orbital_override);

    const int n_inactive_doubly_occupied_orbitals =
        (input.orbital_preparation_input.n_total_electrons -
         input.orbital_preparation_input.n_active_electrons) / 2;
    const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;

    const DifferenceSummary packed_difference =
        summarize_packed_difference(
            current_packed_gradient,
            legacy_gradient_layers.packed_gradient);
    const DifferenceSummary auxiliary_difference =
        summarize_matrix_difference(
            current_total_active_auxiliary_gradient,
            legacy_gradient_layers.grdaux_matrix);
    const DifferenceSummary inactive_slot_difference =
        summarize_slot_difference_for_orbital_range(
            current_slot_gradient,
            legacy_gradient_layers.slot_gradient_matrix,
            0,
            n_inactive_doubly_occupied_orbitals,
            input.orbital_preparation_input);
    const DifferenceSummary active_slot_difference =
        summarize_slot_difference_for_orbital_range(
            current_slot_gradient,
            legacy_gradient_layers.slot_gradient_matrix,
            n_inactive_doubly_occupied_orbitals,
            n_inactive_doubly_occupied_orbitals + n_active_orbitals,
            input.orbital_preparation_input);
    const std::vector<SlotDifference> top_slot_differences =
        collect_top_slot_differences(
            current_slot_gradient,
            legacy_gradient_layers.slot_gradient_matrix,
            input.orbital_preparation_input,
            options.top_count);
    const std::vector<AuxiliaryDifference> top_auxiliary_differences =
        collect_top_auxiliary_differences(
            current_total_active_auxiliary_gradient,
            legacy_gradient_layers.grdaux_matrix,
            options.top_count);

    log_legacy_compare_stage("main_print_results");
    std::cout << std::setprecision(12);
    std::cout << "input = " << options.input_path << '\n';
    std::cout << "orbital_override = "
              << (options.orbital_value_table_bin_path.empty()
                      ? "none"
                      : options.orbital_value_table_bin_path)
              << '\n';
    std::cout << "n_basis_functions = "
              << input.orbital_preparation_input.n_basis_functions << '\n';
    std::cout << "n_orbitals = "
              << input.orbital_preparation_input.n_orbitals << '\n';
    std::cout << "n_inactive_doubly_occupied_orbitals = "
              << n_inactive_doubly_occupied_orbitals << '\n';
    std::cout << "n_active_orbitals = " << n_active_orbitals << '\n';
    std::cout << "current_total_energy = "
              << orbital_gradient_result.scf_result.total_energy << '\n';
    std::cout << "legacy_total_energy = "
              << legacy_gradient_layers.total_energy << '\n';
    std::cout << "total_energy_diff = "
              << (orbital_gradient_result.scf_result.total_energy -
                  legacy_gradient_layers.total_energy)
              << '\n';
    std::cout << "legacy_e11 = " << legacy_gradient_layers.e11 << '\n';
    std::cout << "legacy_e22 = " << legacy_gradient_layers.e22 << '\n';
    std::cout << "current_packed_gradient_size = "
              << current_packed_gradient.size() << '\n';
    std::cout << "legacy_packed_gradient_size = "
              << legacy_gradient_layers.packed_gradient.size() << '\n';
    std::cout << "auxiliary_gradient_rows = "
              << current_total_active_auxiliary_gradient.rows() << '\n';
    std::cout << "auxiliary_gradient_cols = "
              << current_total_active_auxiliary_gradient.cols() << '\n';
    std::cout << "auxiliary_gradient_max_abs_diff = "
              << auxiliary_difference.max_abs << '\n';
    std::cout << "auxiliary_gradient_rms_diff = "
              << auxiliary_difference.rms << '\n';
    std::cout << "packed_gradient_max_abs_diff = "
              << packed_difference.max_abs << '\n';
    std::cout << "packed_gradient_rms_diff = "
              << packed_difference.rms << '\n';
    std::cout << "inactive_slot_max_abs_diff = "
              << inactive_slot_difference.max_abs << '\n';
    std::cout << "inactive_slot_rms_diff = "
              << inactive_slot_difference.rms << '\n';
    std::cout << "active_slot_max_abs_diff = "
              << active_slot_difference.max_abs << '\n';
    std::cout << "active_slot_rms_diff = "
              << active_slot_difference.rms << '\n';

    for (std::size_t index = 0; index < top_slot_differences.size(); ++index) {
      const SlotDifference& difference = top_slot_differences[index];
      std::cout << "top_slot_diff[" << index << "]"
                << " block="
                << orbital_block_name(
                       difference.orbital_index,
                       n_inactive_doubly_occupied_orbitals,
                       n_active_orbitals)
                << " orbital=" << (difference.orbital_index + 1)
                << " slot=" << (difference.slot_index + 1)
                << " basis=" << difference.basis_function_index
                << " current=" << difference.current_value
                << " legacy=" << difference.legacy_value
                << " abs_diff=" << difference.abs_diff
                << '\n';
    }
    for (std::size_t index = 0; index < top_auxiliary_differences.size(); ++index) {
      const AuxiliaryDifference& difference = top_auxiliary_differences[index];
      std::cout << "top_aux_diff[" << index << "]"
                << " active_orbital=" << (difference.active_orbital_index + 1)
                << " basis=" << (difference.basis_function_index + 1)
                << " current=" << difference.current_value
                << " legacy=" << difference.legacy_value
                << " abs_diff=" << difference.abs_diff
                << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "compare_cpp_legacy_orbital_gradient_layers: "
              << error.what() << '\n';
    return 1;
  }
}
