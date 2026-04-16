#include <algorithm>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/molden_file_writer.hpp"
#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/scf/deepvbh_onnx_direct_final_optimizer.hpp"
#include "vb/scf/deepvbh_onnx_hybrid_optimizer.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/scf/adaptive_structure_space_optimizer.hpp"
#include "vb/scf/cpp_vb_scf_optimizer.hpp"

namespace {

namespace fs = std::filesystem;

enum class StructureSpaceMode {
  Standard,
  AdaptiveMvp,
};

const char* structure_space_mode_name(StructureSpaceMode mode) {
  switch (mode) {
    case StructureSpaceMode::Standard:
      return "standard";
    case StructureSpaceMode::AdaptiveMvp:
      return "adaptive_mvp";
  }
  return "unknown";
}

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

const char* gradient_tolerance_metric_name(
    xmvb::vb::CppVbScfOptimizerBackend backend) {
  switch (backend) {
    case xmvb::vb::CppVbScfOptimizerBackend::LegacyFortran:
    case xmvb::vb::CppVbScfOptimizerBackend::Lbfgspp:
      return "full_gradient_l2_norm";
    case xmvb::vb::CppVbScfOptimizerBackend::NonredundantProjectedGradient:
    case xmvb::vb::CppVbScfOptimizerBackend::NonredundantLbfgspp:
    case xmvb::vb::CppVbScfOptimizerBackend::NonredundantTruncatedNewton:
      return "projected_gradient_inf_norm";
    case xmvb::vb::CppVbScfOptimizerBackend::DeepVBHOnnx:
    case xmvb::vb::CppVbScfOptimizerBackend::DeepVBHOnnxDirectFinal:
      return "full_gradient_inf_norm";
  }
  return "unknown";
}

const char* bool_name(bool value) {
  return value ? "true" : "false";
}

std::optional<bool> parse_env_optional_flag(
    const char* variable_name) {
  const char* value = std::getenv(variable_name);
  if (value == nullptr || value[0] == '\0') {
    return std::nullopt;
  }
  return std::strcmp(value, "0") != 0 &&
      std::strcmp(value, "false") != 0 &&
      std::strcmp(value, "FALSE") != 0;
}

bool parse_env_flag_with_default(
    const char* variable_name,
    bool default_value) {
  const auto value = parse_env_optional_flag(variable_name);
  return value.has_value() ? *value : default_value;
}

int parse_env_int_with_default(
    const char* variable_name,
    int default_value) {
  const char* value = std::getenv(variable_name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(value, &end, 10);
  if (errno != 0 || end == value || (end != nullptr && end[0] != '\0') ||
      parsed < static_cast<long>(std::numeric_limits<int>::min()) ||
      parsed > static_cast<long>(std::numeric_limits<int>::max())) {
    return default_value;
  }
  return static_cast<int>(parsed);
}

double parse_env_double_with_default(
    const char* variable_name,
    double default_value) {
  const char* value = std::getenv(variable_name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  char* end = nullptr;
  errno = 0;
  const double parsed = std::strtod(value, &end);
  if (errno != 0 || end == value || (end != nullptr && end[0] != '\0') ||
      !std::isfinite(parsed)) {
    return default_value;
  }
  return parsed;
}

const char* env_value_or_unset(const char* variable_name) {
  const char* value = std::getenv(variable_name);
  return (value == nullptr || value[0] == '\0') ? "<unset>" : value;
}

bool optimizer_chart_uses_sparse_orbital_support(
    const xmvb::vb::OrbitalPreparationInput& orbital_preparation_input) {
  constexpr int kLegacyOrbitalTypeHao = 1;
  constexpr int kLegacyOrbitalTypeBdo = 2;
  constexpr int kLegacyOrbitalTypeOeo = 3;
  // Mirror the optimizer's chart classification so the CLI summary can state
  // whether exact_ctx starts from the full-AO OEO manifold or from the sparser
  // HAO/BDO support charts that use different TN/HVP heuristics.
  if (orbital_preparation_input.orbital_type == kLegacyOrbitalTypeOeo) {
    return false;
  }
  if (orbital_preparation_input.orbital_type == kLegacyOrbitalTypeHao ||
      orbital_preparation_input.orbital_type == kLegacyOrbitalTypeBdo) {
    return true;
  }
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  if (n_basis_functions <= 0) {
    return false;
  }
  for (const int basis_count : orbital_preparation_input.orbital_basis_counts) {
    if (basis_count > 0 && basis_count < n_basis_functions) {
      return true;
    }
  }
  return false;
}

std::string exact_ctx_initial_outer_response_policy_name(
    const xmvb::vb::OrbitalPreparationInput& orbital_preparation_input) {
  const auto override =
      parse_env_optional_flag("XMVB_CPP_EXACT_CTX_INNER_SOLVE_USE_OUTER_RESPONSE");
  if (override.has_value()) {
    return *override ? "forced_full_outer_response" : "forced_core_only";
  }

  const int n_active_orbitals = orbital_preparation_input.n_active_orbitals;
  if (n_active_orbitals >
      parse_env_int_with_default(
          "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_ENABLE_MAX_ACTIVE_ORBITALS",
          6)) {
    return "cheap_core_only";
  }

  const int full_inner_solve_begin =
      std::max(
          0,
          parse_env_int_with_default(
              "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_BEGIN",
              0));
  int full_inner_solve_count =
      std::max(
          0,
          parse_env_int_with_default(
              "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_COUNT",
              2));
  int max_extra_count =
      std::max(
          0,
          parse_env_int_with_default(
              "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_MAX_EXTRA_COUNT",
              0));
  if (n_active_orbitals >
      std::max(
          0,
          parse_env_int_with_default(
              "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_MULTI_STEP_MAX_ACTIVE_ORBITALS",
              8))) {
    full_inner_solve_count = std::min(full_inner_solve_count, 1);
    max_extra_count = 0;
  }

  if (0 < full_inner_solve_begin) {
    return "cheap_core_only";
  }
  if (full_inner_solve_count > 0) {
    return "startup_full_outer_response";
  }
  if (max_extra_count > 0) {
    return "gradient_tail_conditional";
  }
  return "cheap_core_only";
}

void print_exact_ctx_policy_summary(
    const xmvb::vb::CppVbScfOptimizerOptions& options,
    const xmvb::vb::CppVbInput& input) {
  if (options.backend !=
          xmvb::vb::CppVbScfOptimizerBackend::NonredundantTruncatedNewton ||
      options.nonredundant_truncated_newton_hvp_mode !=
          xmvb::vb::NonredundantTruncatedNewtonHvpMode::ExactContextDirectAction) {
    return;
  }

  const auto& orbital_preparation_input = input.orbital_preparation_input;
  const bool sparse_orbital_chart =
      optimizer_chart_uses_sparse_orbital_support(orbital_preparation_input);
  const auto outer_response_override =
      parse_env_optional_flag("XMVB_CPP_EXACT_CTX_INNER_SOLVE_USE_OUTER_RESPONSE");
  std::cout << "exact_ctx_initial_inner_solve_policy = "
            << exact_ctx_initial_outer_response_policy_name(orbital_preparation_input)
            << '\n';
  std::cout << "exact_ctx_inner_solve_use_outer_response_override = ";
  if (!outer_response_override.has_value()) {
    std::cout << "auto\n";
  } else {
    std::cout << bool_name(*outer_response_override) << '\n';
  }
  std::cout << "exact_ctx_outer_response_runtime_enabled = "
            << bool_name(
                   !parse_env_flag_with_default(
                       "XMVB_CPP_DISABLE_EXACT_CTX_OUTER_RESPONSE",
                       false))
            << '\n';
  std::cout << "exact_ctx_initial_chart = "
            << (sparse_orbital_chart ? "sparse_support" : "full_ao")
            << '\n';
  std::cout << "exact_ctx_n_active_orbitals = "
            << orbital_preparation_input.n_active_orbitals << '\n';
  std::cout << "exact_ctx_startup_full_inner_solve_begin = "
            << parse_env_int_with_default(
                   "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_BEGIN",
                   0)
            << '\n';
  std::cout << "exact_ctx_startup_full_inner_solve_count = "
            << parse_env_int_with_default(
                   "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_COUNT",
                   2)
            << '\n';
  std::cout << "exact_ctx_startup_full_inner_solve_max_extra_count = "
            << parse_env_int_with_default(
                   "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_MAX_EXTRA_COUNT",
                   0)
            << '\n';
  std::cout << "exact_ctx_startup_full_inner_solve_grad_ratio_threshold = "
            << std::setprecision(12)
            << parse_env_double_with_default(
                   "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_GRAD_RATIO_THRESHOLD",
                   0.2)
            << '\n';
  std::cout << "exact_ctx_startup_full_inner_solve_base_max_cg_iterations = "
            << parse_env_int_with_default(
                   "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_BASE_MAX_CG_ITERATIONS",
                   6)
            << '\n';
  std::cout << "exact_ctx_startup_full_inner_solve_tail_max_cg_iterations = "
            << parse_env_int_with_default(
                   "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_TAIL_MAX_CG_ITERATIONS",
                   8)
            << '\n';
  std::cout << "exact_ctx_startup_full_inner_solve_multi_step_max_active_orbitals = "
            << parse_env_int_with_default(
                   "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_MULTI_STEP_MAX_ACTIVE_ORBITALS",
                   8)
            << '\n';
  std::cout << "exact_ctx_startup_full_inner_solve_enable_max_active_orbitals = "
            << parse_env_int_with_default(
                   "XMVB_CPP_EXACT_CTX_STARTUP_FULL_INNER_SOLVE_ENABLE_MAX_ACTIVE_ORBITALS",
                   6)
            << '\n';
  std::cout << "exact_ctx_full_model_correction_enable_max_active_orbitals = "
            << parse_env_int_with_default(
                   "XMVB_CPP_EXACT_CTX_FULL_MODEL_CORRECTION_ENABLE_MAX_ACTIVE_ORBITALS",
                   6)
            << '\n';
  std::cout << "exact_ctx_hybrid_followup_full_enable_max_active_orbitals = "
            << parse_env_int_with_default(
                   "XMVB_CPP_EXACT_CTX_HYBRID_FOLLOWUP_FULL_ENABLE_MAX_ACTIVE_ORBITALS",
                   6)
            << '\n';
  std::cout << "exact_ctx_stall_full_model_correction_enable_max_active_orbitals = "
            << parse_env_int_with_default(
                   "XMVB_CPP_EXACT_CTX_STALL_FULL_MODEL_CORRECTION_ENABLE_MAX_ACTIVE_ORBITALS",
                   6)
            << '\n';
  std::cout << "exact_ctx_retry_rejected_with_full_operator = "
            << bool_name(
                   parse_env_flag_with_default(
                       "XMVB_CPP_EXACT_CTX_RETRY_REJECTED_WITH_FULL_OPERATOR",
                       true))
            << '\n';
  std::cout << "exact_ctx_hybrid_refine_max_cg_iterations = "
            << parse_env_int_with_default(
                   "XMVB_CPP_EXACT_CTX_HYBRID_REFINE_MAX_CG_ITERATIONS",
                   4)
            << '\n';
  std::cout << "exact_ctx_sparse_followup_probe_cooldown = "
            << parse_env_int_with_default(
                   "XMVB_CPP_EXACT_CTX_SPARSE_FOLLOWUP_PROBE_COOLDOWN",
                   2)
            << '\n';
  std::cout << "env_OMP_NUM_THREADS = "
            << env_value_or_unset("OMP_NUM_THREADS") << '\n';
  std::cout << "env_OPENBLAS_NUM_THREADS = "
            << env_value_or_unset("OPENBLAS_NUM_THREADS") << '\n';
  std::cout << "env_MKL_NUM_THREADS = "
            << env_value_or_unset("MKL_NUM_THREADS") << '\n';
}

std::function<void(const xmvb::vb::CppVbScfAcceptedIterationSnapshot&)>
compose_accepted_iteration_callbacks(
    std::function<void(const xmvb::vb::CppVbScfAcceptedIterationSnapshot&)> first,
    std::function<void(const xmvb::vb::CppVbScfAcceptedIterationSnapshot&)> second) {
  if (!first) {
    return second;
  }
  if (!second) {
    return first;
  }
  return [first = std::move(first), second = std::move(second)](
             const xmvb::vb::CppVbScfAcceptedIterationSnapshot& snapshot) {
    first(snapshot);
    second(snapshot);
  };
}

std::function<void(const xmvb::vb::CppVbScfAcceptedIterationSnapshot&)>
build_terminal_iteration_logger() {
  // Mirror the legacy CLI monitor: keep the initial accepted snapshot only as
  // the energy reference for `DE`, then print one fixed-width row per accepted
  // optimization step.
  struct LoggerState {
    bool has_reference_energy = false;
    double previous_total_energy = 0.0;
    std::chrono::steady_clock::time_point previous_iteration_time =
        std::chrono::steady_clock::now();
  };

  auto state = std::make_shared<LoggerState>();
  std::cout << "cpp_vbscf_iterations\n"
            << "ITER           ENERGY               DE          GRAD_INF"
               "             G_L2         DT_S\n";
  std::cout.flush();

  return [state](const xmvb::vb::CppVbScfAcceptedIterationSnapshot& snapshot) {
    const auto iteration_time = std::chrono::steady_clock::now();
    const double elapsed_seconds =
        std::chrono::duration<double>(iteration_time - state->previous_iteration_time)
            .count();
    state->previous_iteration_time = iteration_time;

    if (snapshot.accepted_iteration_index == 0) {
      state->previous_total_energy = snapshot.total_energy;
      state->has_reference_energy = true;
      return;
    }

    if (!state->has_reference_energy) {
      state->previous_total_energy = snapshot.total_energy;
      state->has_reference_energy = true;
    }
    const double delta_energy = snapshot.total_energy - state->previous_total_energy;
    std::ostringstream stream;
    stream << std::setw(4) << snapshot.accepted_iteration_index
           << std::setw(23) << std::fixed << std::setprecision(12)
           << snapshot.total_energy
           << std::setw(18) << std::scientific << std::setprecision(8)
           << delta_energy
           << std::setw(18) << std::scientific << std::setprecision(8)
           << snapshot.sparse_orbital_energy_gradient_inf_norm
           << std::setw(18) << std::scientific << std::setprecision(8)
           << snapshot.sparse_orbital_energy_gradient_l2_norm
           << std::setw(14) << std::fixed << std::setprecision(6)
           << elapsed_seconds
           << '\n';
    std::cout << stream.str();
    std::cout.flush();
    state->previous_total_energy = snapshot.total_energy;
    state->has_reference_energy = true;
  };
}

int get_sparse_coefficient_count(
    const xmvb::vb::OrbitalPreparationInput& orbital_preparation_input,
    int orbital_index) {
  const int n_basis_functions = orbital_preparation_input.n_basis_functions;
  const int explicit_count =
      orbital_preparation_input.orbital_basis_counts[xmvb::to_size(orbital_index)];
  if (explicit_count > 1) {
    return explicit_count;
  }

  int coefficient_count = 0;
  while (coefficient_count < n_basis_functions) {
    const int basis_function_index =
        orbital_preparation_input.orbital_basis_index_table
            [xmvb::to_size(orbital_index) * n_basis_functions + coefficient_count];
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
  return xmvb::to_size(
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
      xmvb::to_size(n_active_orbitals) *
          xmvb::to_size(n_active_orbitals),
      0.0);
  for (int column = 0; column < n_active_orbitals; ++column) {
    for (int row = 0; row < n_active_orbitals; ++row) {
      const int packed_index =
          xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
              row,
              row,
              column,
              column);
      matrix[xmvb::to_size(column) * n_active_orbitals + row] =
          packed_active_two_electron_integrals[xmvb::to_size(packed_index)];
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
      xmvb::to_size(n_active_orbitals) *
          xmvb::to_size(n_active_orbitals),
      0.0);
  for (int column = 0; column < n_active_orbitals; ++column) {
    for (int row = 0; row < n_active_orbitals; ++row) {
      const int packed_index =
          xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
              row,
              column,
              column,
              row);
      matrix[xmvb::to_size(column) * n_active_orbitals + row] =
          packed_active_two_electron_integrals[xmvb::to_size(packed_index)];
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
      xmvb::to_size(raw_structure_data.n_structures) *
          xmvb::to_size(n_orbitals),
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
      occupancy[xmvb::to_size(structure_index) * n_orbitals +
                xmvb::to_size(orbital_index)] += 1.0;
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
        xmvb::to_size(raw_structure_data.n_structures) *
            xmvb::to_size(topology.n_active_beta_electrons) * 2,
        0);
    topology.structure_pair_mask.resize(
        xmvb::to_size(raw_structure_data.n_structures) *
            xmvb::to_size(topology.n_active_beta_electrons),
        1);
  }
  if (topology.n_open_shell_electrons > 0) {
    topology.structure_open_shell_orbitals.resize(
        xmvb::to_size(raw_structure_data.n_structures) *
            xmvb::to_size(topology.n_open_shell_electrons),
        0);
    topology.structure_open_shell_mask.resize(
        xmvb::to_size(raw_structure_data.n_structures) *
            xmvb::to_size(topology.n_open_shell_electrons),
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
          (xmvb::to_size(structure_index) * topology.n_active_beta_electrons +
           xmvb::to_size(pair_index)) *
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
          xmvb::to_size(structure_index) * topology.n_open_shell_electrons +
          xmvb::to_size(open_shell_index)] = orbital_index;
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
        ao_integral_source_name_(
            xmvb::vb::ao_integral_source_name(load_result.ao_integral_source)),
        orbital_guess_source_name_(
            xmvb::vb::orbital_guess_source_name(load_result.orbital_guess_source)),
        raw_structure_selection_name_(
            xmvb::vb::raw_structure_selection_mode_name(load_result.raw_structure_selection)),
        source_raw_structure_count_(load_result.source_raw_structure_count),
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
                    << "  \"ao_integral_source\": \""
                    << escape_json_string(ao_integral_source_name_) << "\",\n"
                    << "  \"orbital_guess_source\": \""
                    << escape_json_string(orbital_guess_source_name_) << "\",\n"
                    << "  \"raw_structure_selection\": \""
                    << escape_json_string(raw_structure_selection_name_) << "\",\n"
                    << "  \"source_raw_structure_count\": "
                    << source_raw_structure_count_ << ",\n"
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
  std::string ao_integral_source_name_;
  std::string orbital_guess_source_name_;
  std::string raw_structure_selection_name_;
  int source_raw_structure_count_ = 0;
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
  if (backend_name == "nonredundant_projected_gradient") {
    options->backend =
        xmvb::vb::CppVbScfOptimizerBackend::NonredundantProjectedGradient;
    return;
  }
  if (backend_name == "nonredundant_lbfgspp") {
    options->backend =
        xmvb::vb::CppVbScfOptimizerBackend::NonredundantLbfgspp;
    return;
  }
  if (backend_name == "nonredundant_truncated_newton") {
    options->backend =
        xmvb::vb::CppVbScfOptimizerBackend::NonredundantTruncatedNewton;
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

void apply_nonredundant_truncated_newton_hvp_mode_argument(
    const std::string& mode_name,
    xmvb::vb::CppVbScfOptimizerOptions* options) {
  if (mode_name == "full_fd") {
    options->nonredundant_truncated_newton_hvp_mode =
        xmvb::vb::NonredundantTruncatedNewtonHvpMode::FullFiniteDifference;
    return;
  }
  if (mode_name == "exact_ctx" || mode_name == "exact_context") {
    options->nonredundant_truncated_newton_hvp_mode =
        xmvb::vb::NonredundantTruncatedNewtonHvpMode::ExactContextDirectAction;
    return;
  }
  throw std::invalid_argument(
      "invalid nonredundant truncated-Newton HVP mode: " + mode_name);
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

void apply_structure_space_mode_argument(
    const std::string& mode_name,
    StructureSpaceMode* mode) {
  if (mode == nullptr) {
    throw std::invalid_argument("mode must not be null");
  }
  if (mode_name == "standard") {
    *mode = StructureSpaceMode::Standard;
    return;
  }
  if (mode_name == "adaptive_mvp") {
    *mode = StructureSpaceMode::AdaptiveMvp;
    return;
  }
  throw std::invalid_argument("invalid structure space mode: " + mode_name);
}

void apply_raw_structure_selection_argument(
    const std::string& selection_name,
    xmvb::vb::CppVbInputLoadOptions* options) {
  if (selection_name == "full") {
    options->raw_structure_selection = xmvb::vb::RawStructureSelectionMode::Full;
    return;
  }
  if (selection_name == "covalent") {
    options->raw_structure_selection = xmvb::vb::RawStructureSelectionMode::Covalent;
    return;
  }
  throw std::invalid_argument("invalid raw structure selection: " + selection_name);
}

void apply_ao_integral_source_argument(
    const std::string& source_name,
    xmvb::vb::CppVbInputLoadOptions* options) {
  if (options == nullptr) {
    throw std::invalid_argument("load options must not be null");
  }
  if (source_name == "auto") {
    options->ao_integral_source = xmvb::vb::AoIntegralSource::Auto;
    return;
  }
  if (source_name == "legacy") {
    options->ao_integral_source = xmvb::vb::AoIntegralSource::LegacyRuntime;
    return;
  }
  if (source_name == "libcint_cpp") {
    options->ao_integral_source = xmvb::vb::AoIntegralSource::LibcintMaterializedCpp;
    return;
  }
  if (source_name == "runtime_hcore") {
    options->ao_integral_source = xmvb::vb::AoIntegralSource::RuntimeCoreHamiltonianOnly;
    return;
  }
  throw std::invalid_argument("invalid AO integral source: " + source_name);
}

void apply_orbital_guess_source_argument(
    const std::string& source_name,
    xmvb::vb::CppVbInputLoadOptions* options) {
  if (options == nullptr) {
    throw std::invalid_argument("load options must not be null");
  }
  if (source_name == "legacy") {
    options->orbital_guess_source = xmvb::vb::OrbitalGuessSource::LegacyRuntime;
    return;
  }
  if (source_name == "cpp") {
    options->orbital_guess_source = xmvb::vb::OrbitalGuessSource::Cpp;
    return;
  }
  throw std::invalid_argument("invalid orbital guess source: " + source_name);
}

void apply_standard_two_electron_mode_argument(
    const std::string& mode_name,
    xmvb::vb::CppVbInputLoadOptions* options) {
  if (options == nullptr) {
    throw std::invalid_argument("load options must not be null");
  }
  if (mode_name == "auto") {
    options->standard_two_electron_mode = xmvb::vb::StandardTwoElectronMode::Auto;
    return;
  }
  if (mode_name == "exact") {
    options->standard_two_electron_mode = xmvb::vb::StandardTwoElectronMode::Exact;
    return;
  }
  if (mode_name == "ri") {
    options->standard_two_electron_mode =
        xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
    return;
  }
  throw std::invalid_argument("invalid standard two-electron mode: " + mode_name);
}

void apply_adaptive_seed_selection_argument(
    const std::string& selection_name,
    xmvb::vb::AdaptiveStructureSpaceOptimizerOptions* options) {
  if (options == nullptr) {
    throw std::invalid_argument("adaptive options must not be null");
  }
  if (selection_name == "full") {
    options->seed_selection = xmvb::vb::RawStructureSelectionMode::Full;
    return;
  }
  if (selection_name == "covalent") {
    options->seed_selection = xmvb::vb::RawStructureSelectionMode::Covalent;
    return;
  }
  throw std::invalid_argument("invalid adaptive seed selection: " + selection_name);
}

void apply_adaptive_determinant_score_mode_argument(
    const std::string& mode_name,
    xmvb::vb::AdaptiveStructureSpaceOptimizerOptions* options) {
  if (options == nullptr) {
    throw std::invalid_argument("adaptive options must not be null");
  }
  if (mode_name == "proposal_all") {
    options->determinant_score_mode =
        xmvb::vb::AdaptiveDeterminantScoreMode::ProposalAll;
    return;
  }
  if (mode_name == "outside_only") {
    options->determinant_score_mode =
        xmvb::vb::AdaptiveDeterminantScoreMode::OutsideOnly;
    return;
  }
  throw std::invalid_argument("invalid adaptive determinant score mode: " + mode_name);
}

void print_usage() {
  std::cerr << "usage: run_cpp_vbscf <input.xmi> "
               "[--optimizer-backend lbfgspp|nonredundant_projected_gradient|nonredundant_lbfgspp|nonredundant_truncated_newton";
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
  std::cerr << "] [--structure-space-mode standard|adaptive_mvp]"
               " [--max-iterations <count>]"
               " [--verbose true|false]"
               " [--gradient-tolerance <value>]"
               " [--energy-tolerance <value>]"
               " [--nonredundant-polish-max-iterations <count>]"
               " [--nonredundant-polish-gradient-scale <value>]"
               " [--nonredundant-truncated-newton-max-cg-iterations <count|0=auto>]"
               " [--nonredundant-truncated-newton-hvp-mode full_fd|exact_ctx]"
               " [--nonredundant-truncated-newton-hvp-step-size <value>]"
               " [--nonredundant-truncated-newton-transport-history-size <count>]"
               " [--standard-two-electron-mode auto|exact|ri]"
               " [--ao-integral-source auto|legacy|libcint_cpp|runtime_hcore]"
               " [--skip-orbital-guess true|false]"
               " [--orbital-guess-source legacy|cpp]"
               " [--raw-structure-selection full|covalent]"
               " [--adaptive-seed-selection full|covalent]"
               " [--adaptive-determinant-score-mode proposal_all|outside_only]"
               " [--adaptive-max-outer-iterations <count>]"
               " [--adaptive-max-topology-distance <count>]"
               " [--adaptive-max-neighbors-per-structure <count>]"
               " [--adaptive-max-candidate-pool-size <count>]"
               " [--adaptive-batch-size <count>]"
               " [--adaptive-max-total-structures <count>]"
               " [--adaptive-minimum-candidate-score <value>]"
               " [--adaptive-verbose true|false]"
               " [--dump-trace-dir <dataset_root>]"
               " [--onnx-model <path>]"
               " [--ml-initial-step-scale <value>]"
               " [--ml-minimum-step-scale <value>]"
               " [--ml-step-shrink-factor <value>]"
               " [--ml-max-backtracks <count>]"
               " [--ml-fallback-max-iterations <count>]"
               " [--ml-keep-work-dir true|false]"
               " [--ml-work-dir <path>]\n"
               "default optimizer backend: nonredundant_lbfgspp\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || ((argc - 2) % 2 != 0)) {
    print_usage();
    return 1;
  }

  const std::string input_path = argv[1];
  StructureSpaceMode structure_space_mode = StructureSpaceMode::Standard;
  xmvb::vb::CppVbInputLoadOptions load_options;
  load_options.ao_integral_source = xmvb::vb::AoIntegralSource::Auto;
  load_options.orbital_guess_source = xmvb::vb::OrbitalGuessSource::Cpp;
  load_options.standard_two_electron_mode = xmvb::vb::StandardTwoElectronMode::Auto;
  xmvb::vb::CppVbScfOptimizerOptions options;
  xmvb::vb::AdaptiveStructureSpaceOptimizerOptions adaptive_options;
  // The CLI path only needs accepted-iterate snapshots when trace dumping is
  // explicitly requested. Keep the default optimizer API behavior unchanged,
  // but disable trace retention here so routine runs avoid the extra
  // reference-gradient work performed per accepted iterate.
  options.retain_accepted_iteration_trace = false;
  options.accepted_iteration_callback_requires_full_snapshot = false;
  options.max_iterations = 2000;
  options.gradient_tolerance = 1.0e-3;
  options.energy_tolerance = 1.0e-5;
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
  bool user_specified_ao_integral_source = false;
  bool user_specified_raw_structure_selection = false;
  bool user_specified_max_iterations = false;
  for (int argument_index = 2; argument_index < argc; argument_index += 2) {
    const std::string argument_name = argv[argument_index];
    const std::string argument_value = argv[argument_index + 1];
    try {
      if (argument_name == "--optimizer-backend") {
        apply_optimizer_backend_argument(argument_value, &options);
      } else if (argument_name == "--structure-space-mode") {
        apply_structure_space_mode_argument(argument_value, &structure_space_mode);
      } else if (argument_name == "--algorithm") {
        throw std::invalid_argument(
            "--algorithm is no longer supported; set the algorithm via the input .xmi "
            "keywords such as tbvbscf");
      } else if (argument_name == "--max-iterations") {
        user_specified_max_iterations = true;
        options.max_iterations = std::stoi(argument_value);
      } else if (argument_name == "--verbose") {
        options.verbose = parse_bool_argument(argument_value);
      } else if (argument_name == "--gradient-tolerance") {
        options.gradient_tolerance = std::stod(argument_value);
      } else if (argument_name == "--energy-tolerance") {
        options.energy_tolerance = std::stod(argument_value);
      } else if (argument_name == "--nonredundant-polish-max-iterations") {
        options.nonredundant_polish_max_iterations = std::stoi(argument_value);
      } else if (argument_name == "--nonredundant-polish-gradient-scale") {
        options.nonredundant_polish_gradient_scale = std::stod(argument_value);
      } else if (
          argument_name ==
          "--nonredundant-truncated-newton-max-cg-iterations") {
        options.nonredundant_truncated_newton_max_cg_iterations =
            std::stoi(argument_value);
      } else if (
          argument_name ==
          "--nonredundant-truncated-newton-hvp-step-size") {
        options.nonredundant_truncated_newton_hvp_step_size =
            std::stod(argument_value);
      } else if (
          argument_name ==
          "--nonredundant-truncated-newton-hvp-mode") {
        apply_nonredundant_truncated_newton_hvp_mode_argument(
            argument_value,
            &options);
      } else if (
          argument_name ==
          "--nonredundant-truncated-newton-transport-history-size") {
        options.nonredundant_truncated_newton_transport_history_size =
            std::stoi(argument_value);
      } else if (argument_name == "--ao-integral-source") {
        user_specified_ao_integral_source = true;
        apply_ao_integral_source_argument(argument_value, &load_options);
      } else if (argument_name == "--standard-two-electron-mode") {
        apply_standard_two_electron_mode_argument(argument_value, &load_options);
      } else if (argument_name == "--skip-orbital-guess") {
        load_options.skip_orbital_guess = parse_bool_argument(argument_value);
      } else if (argument_name == "--orbital-guess-source") {
        apply_orbital_guess_source_argument(argument_value, &load_options);
      } else if (argument_name == "--raw-structure-selection") {
        user_specified_raw_structure_selection = true;
        apply_raw_structure_selection_argument(argument_value, &load_options);
      } else if (argument_name == "--adaptive-seed-selection") {
        apply_adaptive_seed_selection_argument(argument_value, &adaptive_options);
      } else if (argument_name == "--adaptive-determinant-score-mode") {
        apply_adaptive_determinant_score_mode_argument(argument_value, &adaptive_options);
      } else if (argument_name == "--adaptive-max-outer-iterations") {
        adaptive_options.max_outer_iterations = std::stoi(argument_value);
      } else if (argument_name == "--adaptive-max-topology-distance") {
        adaptive_options.max_topology_distance = std::stoi(argument_value);
      } else if (argument_name == "--adaptive-max-neighbors-per-structure") {
        adaptive_options.max_neighbors_per_structure = std::stoi(argument_value);
      } else if (argument_name == "--adaptive-max-candidate-pool-size") {
        adaptive_options.max_candidate_pool_size = std::stoi(argument_value);
      } else if (argument_name == "--adaptive-batch-size") {
        adaptive_options.batch_size = std::stoi(argument_value);
      } else if (argument_name == "--adaptive-max-total-structures") {
        adaptive_options.max_total_structures = std::stoi(argument_value);
      } else if (argument_name == "--adaptive-minimum-candidate-score") {
        adaptive_options.minimum_candidate_score = std::stod(argument_value);
      } else if (argument_name == "--adaptive-verbose") {
        adaptive_options.verbose = parse_bool_argument(argument_value);
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
  if (!user_specified_ao_integral_source &&
      (options.backend == xmvb::vb::CppVbScfOptimizerBackend::DeepVBHOnnx ||
       options.backend == xmvb::vb::CppVbScfOptimizerBackend::DeepVBHOnnxDirectFinal)) {
    load_options.ao_integral_source =
        xmvb::vb::AoIntegralSource::LibcintMaterializedCpp;
  }

  if (structure_space_mode == StructureSpaceMode::AdaptiveMvp) {
    if (options.backend == xmvb::vb::CppVbScfOptimizerBackend::DeepVBHOnnx ||
        options.backend == xmvb::vb::CppVbScfOptimizerBackend::DeepVBHOnnxDirectFinal) {
      throw std::invalid_argument(
          "adaptive_mvp currently supports only standard C++ VBSCF optimizer backends");
    }
    if (!dump_trace_dir.empty()) {
      throw std::invalid_argument(
          "--dump-trace-dir is not yet supported with --structure-space-mode adaptive_mvp");
    }
    if (user_specified_raw_structure_selection &&
        load_options.raw_structure_selection != xmvb::vb::RawStructureSelectionMode::Full) {
      throw std::invalid_argument(
          "--raw-structure-selection is only for standard mode; "
          "use --adaptive-seed-selection in adaptive_mvp mode");
    }
    load_options.raw_structure_selection = xmvb::vb::RawStructureSelectionMode::Full;
    load_options.expand_selected_raw_structures = false;
  }

  const auto load_result = xmvb::vb::load_cpp_vb_input_with_timings(input_path, load_options);
  const auto& input = load_result.input;
  options.algorithm = load_result.requested_algorithm;
  if (!user_specified_max_iterations) {
    // Keep the standalone SCF loop aligned with the legacy deck semantics:
    // `.xmi` `itmax` controls the maximum iteration count, and omitted `itmax`
    // falls back to the project default of 2000.
    options.max_iterations = load_result.requested_scf_max_iterations;
  }

  if (options.algorithm == xmvb::vb::VBSCFAlgorithm::BiorthogonalExactSelected &&
      structure_space_mode == StructureSpaceMode::AdaptiveMvp) {
    throw std::invalid_argument(
        "tbvbscf is not yet wired into adaptive_mvp structure-space expansion");
  }

  std::shared_ptr<AcceptedIterationTraceDatasetWriter> trace_writer;
  if (options.verbose) {
    options.accepted_iteration_callback = compose_accepted_iteration_callbacks(
        std::move(options.accepted_iteration_callback),
        build_terminal_iteration_logger());
  }
  if (!dump_trace_dir.empty()) {
    if (options.algorithm != xmvb::vb::VBSCFAlgorithm::Original) {
      throw std::invalid_argument(
          "--dump-trace-dir currently supports only input files that request "
          "the original VBSCF algorithm");
    }
    trace_writer = std::make_shared<AcceptedIterationTraceDatasetWriter>(
        dump_trace_dir,
        input_path,
        load_result,
        options);
    options.retain_accepted_iteration_trace = false;
    options.accepted_iteration_callback_requires_reference_gradient = true;
    options.accepted_iteration_callback_requires_full_snapshot = true;
    options.accepted_iteration_callback = compose_accepted_iteration_callbacks(
        std::move(options.accepted_iteration_callback),
        [trace_writer](const xmvb::vb::CppVbScfAcceptedIterationSnapshot& snapshot) {
          trace_writer->write_accepted_iteration(snapshot);
        });
  }

  deepvbh_options.optimizer_options = options;
  deepvbh_options.inference_options.algorithm = options.algorithm;
  deepvbh_direct_options.optimizer_options = options;
  deepvbh_direct_options.inference_options.algorithm = options.algorithm;
  xmvb::vb::CppVbScfOptimizerResult result;
  std::optional<xmvb::vb::AdaptiveStructureSpaceOptimizerResult> adaptive_result;
  if (structure_space_mode == StructureSpaceMode::AdaptiveMvp) {
    xmvb::vb::AdaptiveStructureSpaceOptimizer optimizer(options, adaptive_options);
    adaptive_result = optimizer.optimize(load_result);
    result = adaptive_result->inner_result;
  } else if (options.backend == xmvb::vb::CppVbScfOptimizerBackend::DeepVBHOnnx) {
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
  std::optional<fs::path> molden_output_path;
  if (load_result.request_molden_output) {
    molden_output_path =
        xmvb::vb::write_molden_file(fs::path(input_path), result.optimized_input);
  }

  const double initial_electronic_energy =
      result.initial_total_energy - load_result.nuclear_repulsion_energy;
  const double final_electronic_energy =
      result.final_total_energy - load_result.nuclear_repulsion_energy;
  const double initial_valence_structure_eigenvalue =
      result.initial_total_energy -
      load_result.nuclear_repulsion_energy -
      result.initial_one_electron_reference_energy;
  const double final_valence_structure_eigenvalue =
      result.final_total_energy -
      load_result.nuclear_repulsion_energy -
      result.final_one_electron_reference_energy;

  std::cout << "optimizer_backend = "
            << xmvb::vb::cpp_vb_scf_optimizer_backend_name(options.backend) << '\n';
  if (structure_space_mode == StructureSpaceMode::AdaptiveMvp) {
    std::cout << "structure_space_mode = "
              << structure_space_mode_name(structure_space_mode) << '\n';
    std::cout << "adaptive_seed_selection = "
              << xmvb::vb::adaptive_structure_space_seed_selection_name(
                     adaptive_options.seed_selection)
              << '\n';
    std::cout << "adaptive_determinant_score_mode = "
              << xmvb::vb::adaptive_determinant_score_mode_name(
                     adaptive_options.determinant_score_mode)
              << '\n';
  }
  std::cout << "ao_integral_source = "
            << xmvb::vb::ao_integral_source_name(load_result.ao_integral_source)
            << '\n';
  std::cout << "standard_two_electron_mode = "
            << xmvb::vb::standard_two_electron_mode_name(
                   load_result.standard_two_electron_mode)
            << '\n';
  std::cout << "gradient_tolerance = "
            << std::setprecision(12) << options.gradient_tolerance << '\n';
  std::cout << "energy_tolerance = "
            << std::setprecision(12) << options.energy_tolerance << '\n';
  std::cout << "max_iterations = " << options.max_iterations << '\n';
  std::cout << "gradient_tolerance_metric = "
            << gradient_tolerance_metric_name(options.backend)
            << '\n';
  if (options.backend ==
      xmvb::vb::CppVbScfOptimizerBackend::NonredundantTruncatedNewton) {
    std::cout << "nonredundant_truncated_newton_hvp_mode = "
              << xmvb::vb::nonredundant_truncated_newton_hvp_mode_name(
                     options.nonredundant_truncated_newton_hvp_mode)
              << '\n';
    if (options.nonredundant_truncated_newton_max_cg_iterations > 0) {
      std::cout << "nonredundant_truncated_newton_max_cg_iterations = "
                << options.nonredundant_truncated_newton_max_cg_iterations << '\n';
    } else {
      std::cout << "nonredundant_truncated_newton_max_cg_iterations = auto\n";
    }
  }
  print_exact_ctx_policy_summary(options, input);
  std::cout << "orbital_guess_source = "
            << xmvb::vb::orbital_guess_source_name(load_result.orbital_guess_source)
            << '\n';
  std::cout << "skip_orbital_guess = "
            << (load_options.skip_orbital_guess ? "true" : "false")
            << '\n';
  std::cout << "raw_structure_selection = "
            << xmvb::vb::raw_structure_selection_mode_name(load_result.raw_structure_selection)
            << '\n';
  std::cout << "source_raw_structure_count = "
            << load_result.source_raw_structure_count << '\n';
  if (adaptive_result.has_value()) {
    double adaptive_total_scoring_wall_time_seconds = 0.0;
    std::uint64_t adaptive_total_boundary_pair_evaluation_count = 0;
    for (const auto& iteration_summary : adaptive_result->iteration_summaries) {
      adaptive_total_scoring_wall_time_seconds +=
          iteration_summary.scoring_wall_time_seconds;
      adaptive_total_boundary_pair_evaluation_count +=
          iteration_summary.boundary_pair_evaluation_count;
    }
    const auto& last_iteration_summary = adaptive_result->iteration_summaries.back();
    std::cout << "selected_raw_structure_count = "
              << adaptive_result->selected_raw_structure_indices.size() << '\n';
    std::cout << "expanded_determinant_count = "
              << result.optimized_input.structure_data.alpha_det.size() << '\n';
    std::cout << "adaptive_outer_iterations = "
              << adaptive_result->iteration_summaries.size() << '\n';
    std::cout << "adaptive_last_unique_proposal_determinant_count = "
              << last_iteration_summary.unique_proposal_determinant_count << '\n';
    std::cout << "adaptive_last_unique_outside_determinant_count = "
              << last_iteration_summary.unique_outside_determinant_count << '\n';
    std::cout << "adaptive_last_unique_shared_determinant_count = "
              << last_iteration_summary.unique_shared_determinant_count << '\n';
    std::cout << "adaptive_last_boundary_pair_evaluation_count = "
              << last_iteration_summary.boundary_pair_evaluation_count << '\n';
    std::cout << "adaptive_last_scoring_wall_time_seconds = " << std::setprecision(12)
              << last_iteration_summary.scoring_wall_time_seconds << '\n';
    std::cout << "adaptive_total_boundary_pair_evaluation_count = "
              << adaptive_total_boundary_pair_evaluation_count << '\n';
    std::cout << "adaptive_total_scoring_wall_time_seconds = " << std::setprecision(12)
              << adaptive_total_scoring_wall_time_seconds << '\n';
    std::cout << "adaptive_total_wall_time_seconds = " << std::setprecision(12)
              << adaptive_result->total_wall_time_seconds << '\n';
  } else {
    std::cout << "selected_raw_structure_count = "
              << load_result.raw_structure_data.n_structures << '\n';
    std::cout << "expanded_determinant_count = "
              << input.structure_data.alpha_det.size() << '\n';
  }
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
  std::cout << "input_requested_algorithm = "
            << xmvb::vb::vb_scf_algorithm_name(load_result.requested_algorithm) << '\n';
  std::cout << "converged = "
            << ((adaptive_result.has_value() ? adaptive_result->converged : result.converged)
                    ? "true"
                    : "false")
            << '\n';
  std::cout << "termination_reason = "
            << (adaptive_result.has_value() ? adaptive_result->termination_reason
                                            : result.termination_reason)
            << '\n';
  std::cout << "iterations = " << result.n_iterations << '\n';
  if (trace_writer != nullptr) {
    std::cout << "trace_sample_dir = " << trace_writer->sample_directory().string() << '\n';
  }
  if (molden_output_path.has_value()) {
    std::cout << "molden_output = " << molden_output_path->string() << '\n';
  }
  std::cout << "nuclear_repulsion_energy = " << std::setprecision(12)
            << load_result.nuclear_repulsion_energy << '\n';
  std::cout << "initial_total_energy = " << std::setprecision(12)
            << result.initial_total_energy << '\n';
  std::cout << "initial_electronic_energy = " << std::setprecision(12)
            << initial_electronic_energy << '\n';
  std::cout << "initial_one_electron_reference_energy = "
            << std::setprecision(12)
            << result.initial_one_electron_reference_energy << '\n';
  std::cout << "initial_valence_structure_eigenvalue = "
            << std::setprecision(12)
            << initial_valence_structure_eigenvalue << '\n';
  std::cout << "final_total_energy = " << std::setprecision(12)
            << result.final_total_energy << '\n';
  std::cout << "final_electronic_energy = " << std::setprecision(12)
            << final_electronic_energy << '\n';
  std::cout << "final_one_electron_reference_energy = "
            << std::setprecision(12)
            << result.final_one_electron_reference_energy << '\n';
  std::cout << "final_valence_structure_eigenvalue = "
            << std::setprecision(12)
            << final_valence_structure_eigenvalue << '\n';
  std::cout << "final_gradient_inf_norm = " << std::setprecision(12)
            << result.final_gradient_inf_norm << '\n';
  std::cout << "final_gradient_l2_norm = " << std::setprecision(12)
            << result.final_gradient_l2_norm << '\n';
  if (options.backend ==
          xmvb::vb::CppVbScfOptimizerBackend::NonredundantProjectedGradient ||
      options.backend ==
          xmvb::vb::CppVbScfOptimizerBackend::NonredundantLbfgspp ||
      options.backend ==
          xmvb::vb::CppVbScfOptimizerBackend::NonredundantTruncatedNewton) {
    std::cout << "final_projected_gradient_inf_norm = "
              << std::setprecision(12)
              << result.final_projected_gradient_inf_norm << '\n';
    std::cout << "final_projected_gradient_l2_norm = "
              << std::setprecision(12)
              << result.final_projected_gradient_l2_norm << '\n';
  }
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

  const bool command_converged =
      adaptive_result.has_value() ? adaptive_result->converged : result.converged;
  return command_converged ? 0 : 2;
}
