#pragma once

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "vb/scf/cpp_active_space_second_order_context.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xmvb::vb {

/**
 * @brief Reads a positive integer runtime override from the environment.
 *
 * The exact_ctx implementation uses a small set of environment-tunable thread
 * caps and cache gates. Centralizing the parsing here keeps all translation
 * units aligned on the same runtime semantics and avoids duplicating the same
 * getenv / validation boilerplate in multiple files.
 */
inline int exact_ctx_positive_env_override(
    const char* env_name,
    int default_value) {
  if (env_name == nullptr || env_name[0] == '\0') {
    throw std::invalid_argument("env_name must not be empty");
  }
  if (default_value <= 0) {
    throw std::invalid_argument("default_value must be positive");
  }
  const char* env_value = std::getenv(env_name);
  if (env_value == nullptr || env_value[0] == '\0') {
    return default_value;
  }
  const int parsed_value = std::stoi(env_value);
  if (parsed_value <= 0) {
    throw std::invalid_argument(
        std::string(env_name) + " must be positive");
  }
  return parsed_value;
}

inline int exact_ctx_openmp_thread_cap() {
  return exact_ctx_positive_env_override(
      "XMVB_CPP_EXACT_CTX_MAX_OMP_THREADS",
      8);
}

inline int exact_ctx_effective_openmp_thread_limit(
    int workload_limited_threads) {
  int omp_max_threads = 1;
#ifdef _OPENMP
  omp_max_threads = omp_get_max_threads();
#endif
  return std::max(
      1,
      std::min(
          std::min(omp_max_threads, exact_ctx_openmp_thread_cap()),
          workload_limited_threads));
}

inline bool exact_ctx_internal_inactive_chart_runtime_enabled() {
  const char* disable_flag =
      std::getenv("XMVB_CPP_DISABLE_EXACT_CTX_INTERNAL_INACTIVE_CHART");
  if (disable_flag != nullptr &&
      disable_flag[0] != '\0' &&
      std::strcmp(disable_flag, "0") != 0 &&
      std::strcmp(disable_flag, "false") != 0 &&
      std::strcmp(disable_flag, "FALSE") != 0) {
    return false;
  }

  const char* enable_flag =
      std::getenv("XMVB_CPP_ENABLE_EXACT_CTX_INTERNAL_INACTIVE_CHART");
  if (enable_flag == nullptr || enable_flag[0] == '\0') {
    return true;
  }
  return std::strcmp(enable_flag, "0") != 0 &&
      std::strcmp(enable_flag, "false") != 0 &&
      std::strcmp(enable_flag, "FALSE") != 0;
}

inline bool has_active_matrix_gradient(
    const CppActiveSpaceSecondOrderContext& context,
    int n_active_orbitals) {
  if (n_active_orbitals < 0) {
    throw std::invalid_argument("n_active_orbitals must not be negative");
  }
  const std::size_t active_orbital_count =
      static_cast<std::size_t>(n_active_orbitals);
  const std::size_t active_matrix_size =
      active_orbital_count * active_orbital_count;
  return context.active_orbital_overlap_gradient.size() == active_matrix_size &&
      context.active_one_electron_gradient.size() == active_matrix_size &&
      context.packed_active_two_electron_gradient.size() ==
      packed_active_two_electron_integral_count(n_active_orbitals);
}

inline bool exact_ctx_stage1_analytic_core_enabled() {
  return true;
}

inline bool exact_ctx_outer_response_enabled() {
  const char* flag = std::getenv("XMVB_CPP_DISABLE_EXACT_CTX_OUTER_RESPONSE");
  if (flag == nullptr || flag[0] == '\0') {
    return true;
  }
  return std::strcmp(flag, "0") == 0 ||
      std::strcmp(flag, "false") == 0 ||
      std::strcmp(flag, "FALSE") == 0;
}

inline bool exact_ctx_orbital_preparation_cache_enabled() {
  const char* flag =
      std::getenv("XMVB_CPP_DISABLE_EXACT_CTX_ORBITAL_PREP_CACHE");
  if (flag == nullptr || flag[0] == '\0') {
    return true;
  }
  return std::strcmp(flag, "0") == 0 ||
      std::strcmp(flag, "false") == 0 ||
      std::strcmp(flag, "FALSE") == 0;
}

}  // namespace xmvb::vb
