#pragma once

#include <vector>

#include "vb/orbital/libcint_input.hpp"

namespace xmvb::vb {

struct LibcintShellBlock {
  int left_shell = 0;
  int right_shell = 0;
  int left_ao_offset = 0;
  int right_ao_offset = 0;
  int left_ao_count = 0;
  int right_ao_count = 0;
  std::vector<double> values;
};

struct LibcintShellQuartet {
  int shell_i = 0;
  int shell_j = 0;
  int shell_k = 0;
  int shell_l = 0;
  int ao_offset_i = 0;
  int ao_offset_j = 0;
  int ao_offset_k = 0;
  int ao_offset_l = 0;
  int ao_count_i = 0;
  int ao_count_j = 0;
  int ao_count_k = 0;
  int ao_count_l = 0;
  std::vector<double> values;
};

struct LibcintThreeCenterShellBlock {
  int primary_left_shell = 0;
  int primary_right_shell = 0;
  int auxiliary_shell = 0;
  int primary_left_ao_offset = 0;
  int primary_right_ao_offset = 0;
  int auxiliary_ao_offset = 0;
  int primary_left_ao_count = 0;
  int primary_right_ao_count = 0;
  int auxiliary_ao_count = 0;

  /**
   * @brief Raw libcint Cartesian block in `[left][right][auxiliary]` order.
   *
   * The flattened index is
   * `left + right * left_count + auxiliary * left_count * right_count`.
   */
  std::vector<double> values;
};

class LibcintDirectShellEvaluator {
public:
  explicit LibcintDirectShellEvaluator(const LibcintInput& input);
  LibcintDirectShellEvaluator(
      const LibcintInput& primary_input,
      const LibcintInput& auxiliary_input);
  ~LibcintDirectShellEvaluator();

  LibcintDirectShellEvaluator(const LibcintDirectShellEvaluator&) = delete;
  LibcintDirectShellEvaluator& operator=(const LibcintDirectShellEvaluator&) = delete;

  int n_basis_functions() const noexcept { return n_basis_functions_; }

  int n_auxiliary_basis_functions() const noexcept {
    return n_auxiliary_basis_functions_;
  }

  const std::vector<double>& ao_normalization() const noexcept {
    return ao_normalization_;
  }

  LibcintShellBlock evaluate_overlap_shell_pair(
      int left_shell,
      int right_shell) const;

  LibcintShellBlock evaluate_core_hamiltonian_shell_pair(
      int left_shell,
      int right_shell) const;

  LibcintShellQuartet evaluate_two_electron_shell_quartet(
      int shell_i,
      int shell_j,
      int shell_k,
      int shell_l) const;

  LibcintShellBlock evaluate_auxiliary_metric_shell_pair(
      int left_auxiliary_shell,
      int right_auxiliary_shell) const;

  LibcintThreeCenterShellBlock evaluate_three_center_shell_block(
      int primary_left_shell,
      int primary_right_shell,
      int auxiliary_shell) const;

  double evaluate_max_abs_raw_two_electron_shell_pair(
      int shell_i,
      int shell_j) const;

private:
  void validate_shell_index(int shell_index) const;
  void validate_auxiliary_shell_index(int shell_index) const;

  LibcintShellBlock evaluate_one_electron_shell_pair(
      int left_shell,
      int right_shell,
      int integral_kind) const;

  const LibcintInput& input_;
  const LibcintInput* auxiliary_input_ = nullptr;
  int n_basis_functions_ = 0;
  int n_auxiliary_basis_functions_ = 0;
  std::vector<double> ao_normalization_;
  std::vector<double> auxiliary_ao_normalization_;
  std::vector<int> combined_atm_;
  std::vector<int> combined_bas_;
  std::vector<double> combined_env_;
  void* two_electron_optimizer_ = nullptr;
  void* auxiliary_metric_optimizer_ = nullptr;
  void* three_center_optimizer_ = nullptr;
};

}  // namespace xmvb::vb
