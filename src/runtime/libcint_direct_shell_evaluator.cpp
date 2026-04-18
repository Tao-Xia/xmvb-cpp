#include "runtime/libcint_direct_shell_evaluator.hpp"

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <vector>

extern "C" {
#include "cint.h"

FINT cint1e_ovlp_cart(
    double* buf,
    FINT* shls,
    FINT* atm,
    FINT natm,
    FINT* bas,
    FINT nbas,
    double* env);
FINT cint1e_kin_cart(
    double* buf,
    FINT* shls,
    FINT* atm,
    FINT natm,
    FINT* bas,
    FINT nbas,
    double* env);
FINT cint1e_nuc_cart(
    double* buf,
    FINT* shls,
    FINT* atm,
    FINT natm,
    FINT* bas,
    FINT nbas,
    double* env);
FINT cint2c2e_cart(
    double* buf,
    FINT* shls,
    FINT* atm,
    FINT natm,
    FINT* bas,
    FINT nbas,
    double* env,
    CINTOpt* opt);
void cint2c2e_cart_optimizer(
    CINTOpt** opt,
    FINT* atm,
    FINT natm,
    FINT* bas,
    FINT nbas,
    double* env);
FINT cint3c2e_cart(
    double* buf,
    FINT* shls,
    FINT* atm,
    FINT natm,
    FINT* bas,
    FINT nbas,
    double* env,
    CINTOpt* opt);
void cint3c2e_cart_optimizer(
    CINTOpt** opt,
    FINT* atm,
    FINT natm,
    FINT* bas,
    FINT nbas,
    double* env);
}

namespace xmvb::vb {

namespace {

enum class OneElectronIntegralKind {
  Overlap,
  CoreHamiltonian,
};

void validate_libcint_input_shape(const LibcintInput& input) {
  if (input.n_atoms < 0) {
    throw std::invalid_argument("LibcintInput n_atoms must be non-negative");
  }
  if (input.n_shells <= 0) {
    throw std::invalid_argument("LibcintInput must contain at least one shell");
  }
  if (input.atm.size() != xmvb::to_size(input.n_atoms) * ATM_SLOTS) {
    throw std::invalid_argument("LibcintInput atm size mismatch");
  }
  if (input.bas.size() != xmvb::to_size(input.n_shells) * BAS_SLOTS) {
    throw std::invalid_argument("LibcintInput bas size mismatch");
  }
  if (input.basidx.size() != xmvb::to_size(input.n_shells) * 2) {
    throw std::invalid_argument("LibcintInput basidx size mismatch");
  }
}

int shell_ao_offset(const LibcintInput& input, int shell_index) {
  return input.basidx[xmvb::to_size(shell_index) * 2];
}

int shell_ao_count(const LibcintInput& input, int shell_index) {
  return input.basidx[xmvb::to_size(shell_index) * 2 + 1];
}

int infer_n_basis_functions(const LibcintInput& input) {
  validate_libcint_input_shape(input);
  const int last_shell_index = input.n_shells - 1;
  return shell_ao_offset(input, last_shell_index) +
         shell_ao_count(input, last_shell_index);
}

std::vector<double> build_ao_normalization(const LibcintInput& input) {
  const int n_basis_functions = infer_n_basis_functions(input);
  std::vector<double> ao_normalization(
      xmvb::to_size(n_basis_functions),
      0.0);
  for (int shell_index = 0; shell_index < input.n_shells; ++shell_index) {
    const int ao_offset = shell_ao_offset(input, shell_index);
    const int ao_count = shell_ao_count(input, shell_index);
    std::vector<double> overlap_buffer(
        xmvb::to_size(ao_count) * ao_count,
        0.0);
    FINT shell_pair[2] = {shell_index, shell_index};
    const FINT status = cint1e_ovlp_cart(
        overlap_buffer.data(),
        shell_pair,
        const_cast<int*>(input.atm.data()),
        input.n_atoms,
        const_cast<int*>(input.bas.data()),
        input.n_shells,
        const_cast<double*>(input.env.data()));
    if (status == 0) {
      throw std::runtime_error("failed to evaluate AO overlap diagonal");
    }
    for (int local_ao = 0; local_ao < ao_count; ++local_ao) {
      const double diagonal_overlap =
          overlap_buffer[xmvb::to_size(local_ao) * (ao_count + 1)];
      if (!(diagonal_overlap > 0.0)) {
        throw std::runtime_error("non-positive AO overlap diagonal encountered");
      }
      ao_normalization[xmvb::to_size(ao_offset + local_ao)] =
          std::sqrt(1.0 / diagonal_overlap);
    }
  }
  return ao_normalization;
}

void validate_shared_atom_tables(
    const LibcintInput& primary_input,
    const LibcintInput& auxiliary_input) {
  if (auxiliary_input.n_atoms != primary_input.n_atoms) {
    throw std::invalid_argument("primary and auxiliary LibcintInput atom counts differ");
  }
  if (auxiliary_input.atm != primary_input.atm) {
    throw std::invalid_argument("primary and auxiliary LibcintInput atm tables differ");
  }
  if (auxiliary_input.env.size() < primary_input.env.size()) {
    throw std::invalid_argument("auxiliary env does not contain the primary env prefix");
  }
  if (!std::equal(
          primary_input.env.begin(),
          primary_input.env.end(),
          auxiliary_input.env.begin())) {
    throw std::invalid_argument("auxiliary env prefix differs from the primary env");
  }
}

std::vector<double> build_combined_env(
    const LibcintInput& primary_input,
    const LibcintInput& auxiliary_input) {
  std::vector<double> combined_env;
  combined_env.reserve(
      primary_input.env.size() +
      (auxiliary_input.env.size() - primary_input.env.size()));
  combined_env.insert(
      combined_env.end(),
      primary_input.env.begin(),
      primary_input.env.end());
  combined_env.insert(
      combined_env.end(),
      auxiliary_input.env.begin() + static_cast<std::ptrdiff_t>(primary_input.env.size()),
      auxiliary_input.env.end());
  return combined_env;
}

std::vector<int> build_combined_basis(
    const LibcintInput& primary_input,
    const LibcintInput& auxiliary_input) {
  std::vector<int> combined_bas;
  combined_bas.reserve(primary_input.bas.size() + auxiliary_input.bas.size());
  combined_bas.insert(combined_bas.end(), primary_input.bas.begin(), primary_input.bas.end());
  combined_bas.insert(combined_bas.end(), auxiliary_input.bas.begin(), auxiliary_input.bas.end());
  return combined_bas;
}

}  // namespace

LibcintDirectShellEvaluator::LibcintDirectShellEvaluator(const LibcintInput& input)
    : input_(input),
      n_basis_functions_(infer_n_basis_functions(input)) {
  validate_libcint_input_shape(input_);
  ao_normalization_ = build_ao_normalization(input_);

  cint2e_cart_optimizer(
      reinterpret_cast<CINTOpt**>(&two_electron_optimizer_),
      const_cast<int*>(input_.atm.data()),
      input_.n_atoms,
      const_cast<int*>(input_.bas.data()),
      input_.n_shells,
      const_cast<double*>(input_.env.data()));
}

LibcintDirectShellEvaluator::LibcintDirectShellEvaluator(
    const LibcintInput& primary_input,
    const LibcintInput& auxiliary_input)
    : input_(primary_input),
      auxiliary_input_(&auxiliary_input),
      n_basis_functions_(infer_n_basis_functions(primary_input)),
      n_auxiliary_basis_functions_(infer_n_basis_functions(auxiliary_input)) {
  validate_libcint_input_shape(input_);
  validate_libcint_input_shape(*auxiliary_input_);
  validate_shared_atom_tables(input_, *auxiliary_input_);

  ao_normalization_ = build_ao_normalization(input_);
  auxiliary_ao_normalization_ = build_ao_normalization(*auxiliary_input_);
  combined_atm_.assign(input_.atm.begin(), input_.atm.end());
  combined_bas_ = build_combined_basis(input_, *auxiliary_input_);
  combined_env_ = build_combined_env(input_, *auxiliary_input_);

  cint2e_cart_optimizer(
      reinterpret_cast<CINTOpt**>(&two_electron_optimizer_),
      const_cast<int*>(input_.atm.data()),
      input_.n_atoms,
      const_cast<int*>(input_.bas.data()),
      input_.n_shells,
      const_cast<double*>(input_.env.data()));
  cint2c2e_cart_optimizer(
      reinterpret_cast<CINTOpt**>(&auxiliary_metric_optimizer_),
      const_cast<int*>(auxiliary_input_->atm.data()),
      auxiliary_input_->n_atoms,
      const_cast<int*>(auxiliary_input_->bas.data()),
      auxiliary_input_->n_shells,
      const_cast<double*>(auxiliary_input_->env.data()));
  cint3c2e_cart_optimizer(
      reinterpret_cast<CINTOpt**>(&three_center_optimizer_),
      combined_atm_.data(),
      input_.n_atoms,
      combined_bas_.data(),
      input_.n_shells + auxiliary_input_->n_shells,
      combined_env_.data());
}

LibcintDirectShellEvaluator::~LibcintDirectShellEvaluator() {
  CINTdel_optimizer(reinterpret_cast<CINTOpt**>(&two_electron_optimizer_));
  CINTdel_optimizer(reinterpret_cast<CINTOpt**>(&auxiliary_metric_optimizer_));
  CINTdel_optimizer(reinterpret_cast<CINTOpt**>(&three_center_optimizer_));
}

void LibcintDirectShellEvaluator::validate_shell_index(int shell_index) const {
  if (shell_index < 0 || shell_index >= input_.n_shells) {
    throw std::out_of_range("shell index out of range");
  }
}

void LibcintDirectShellEvaluator::validate_auxiliary_shell_index(int shell_index) const {
  if (auxiliary_input_ == nullptr) {
    throw std::logic_error("auxiliary shell access requires an auxiliary LibcintInput");
  }
  if (shell_index < 0 || shell_index >= auxiliary_input_->n_shells) {
    throw std::out_of_range("auxiliary shell index out of range");
  }
}

LibcintShellBlock LibcintDirectShellEvaluator::evaluate_one_electron_shell_pair(
    int left_shell,
    int right_shell,
    int integral_kind) const {
  validate_shell_index(left_shell);
  validate_shell_index(right_shell);

  const int left_ao_offset = shell_ao_offset(input_, left_shell);
  const int right_ao_offset = shell_ao_offset(input_, right_shell);
  const int left_ao_count = shell_ao_count(input_, left_shell);
  const int right_ao_count = shell_ao_count(input_, right_shell);
  std::vector<double> buffer(
      xmvb::to_size(left_ao_count) * right_ao_count,
      0.0);
  FINT shell_pair[2] = {left_shell, right_shell};

  FINT status = 0;
  switch (static_cast<OneElectronIntegralKind>(integral_kind)) {
    case OneElectronIntegralKind::Overlap:
      status = cint1e_ovlp_cart(
          buffer.data(),
          shell_pair,
          const_cast<int*>(input_.atm.data()),
          input_.n_atoms,
          const_cast<int*>(input_.bas.data()),
          input_.n_shells,
          const_cast<double*>(input_.env.data()));
      break;
    case OneElectronIntegralKind::CoreHamiltonian: {
      std::vector<double> kinetic_buffer(buffer.size(), 0.0);
      std::vector<double> nuclear_buffer(buffer.size(), 0.0);
      const FINT kinetic_status = cint1e_kin_cart(
          kinetic_buffer.data(),
          shell_pair,
          const_cast<int*>(input_.atm.data()),
          input_.n_atoms,
          const_cast<int*>(input_.bas.data()),
          input_.n_shells,
          const_cast<double*>(input_.env.data()));
      const FINT nuclear_status = cint1e_nuc_cart(
          nuclear_buffer.data(),
          shell_pair,
          const_cast<int*>(input_.atm.data()),
          input_.n_atoms,
          const_cast<int*>(input_.bas.data()),
          input_.n_shells,
          const_cast<double*>(input_.env.data()));
      status = kinetic_status || nuclear_status;
      if (status != 0) {
        for (std::size_t index = 0; index < buffer.size(); ++index) {
          const int row = static_cast<int>(index % xmvb::to_size(left_ao_count));
          const int column = static_cast<int>(index / xmvb::to_size(left_ao_count));
          const double normalization =
              ao_normalization_[xmvb::to_size(left_ao_offset + row)] *
              ao_normalization_[xmvb::to_size(right_ao_offset + column)];
          buffer[index] = (kinetic_buffer[index] + nuclear_buffer[index]) * normalization;
        }
      }
      break;
    }
  }

  if (status == 0) {
    std::fill(buffer.begin(), buffer.end(), 0.0);
  }

  LibcintShellBlock result;
  result.left_shell = left_shell;
  result.right_shell = right_shell;
  result.left_ao_offset = left_ao_offset;
  result.right_ao_offset = right_ao_offset;
  result.left_ao_count = left_ao_count;
  result.right_ao_count = right_ao_count;
  result.values = std::move(buffer);
  return result;
}

LibcintShellBlock LibcintDirectShellEvaluator::evaluate_overlap_shell_pair(
    int left_shell,
    int right_shell) const {
  return evaluate_one_electron_shell_pair(
      left_shell,
      right_shell,
      static_cast<int>(OneElectronIntegralKind::Overlap));
}

LibcintShellBlock LibcintDirectShellEvaluator::evaluate_core_hamiltonian_shell_pair(
    int left_shell,
    int right_shell) const {
  return evaluate_one_electron_shell_pair(
      left_shell,
      right_shell,
      static_cast<int>(OneElectronIntegralKind::CoreHamiltonian));
}

LibcintShellQuartet LibcintDirectShellEvaluator::evaluate_two_electron_shell_quartet(
    int shell_i,
    int shell_j,
    int shell_k,
    int shell_l) const {
  validate_shell_index(shell_i);
  validate_shell_index(shell_j);
  validate_shell_index(shell_k);
  validate_shell_index(shell_l);

  const int ao_offset_i = shell_ao_offset(input_, shell_i);
  const int ao_offset_j = shell_ao_offset(input_, shell_j);
  const int ao_offset_k = shell_ao_offset(input_, shell_k);
  const int ao_offset_l = shell_ao_offset(input_, shell_l);
  const int ao_count_i = shell_ao_count(input_, shell_i);
  const int ao_count_j = shell_ao_count(input_, shell_j);
  const int ao_count_k = shell_ao_count(input_, shell_k);
  const int ao_count_l = shell_ao_count(input_, shell_l);

  std::vector<double> buffer(
      xmvb::to_size(ao_count_i) *
          ao_count_j * ao_count_k * ao_count_l,
      0.0);
  FINT shell_quartet[4] = {shell_i, shell_j, shell_k, shell_l};
  const FINT status = cint2e_cart(
      buffer.data(),
      shell_quartet,
      const_cast<int*>(input_.atm.data()),
      input_.n_atoms,
      const_cast<int*>(input_.bas.data()),
      input_.n_shells,
      const_cast<double*>(input_.env.data()),
      reinterpret_cast<CINTOpt*>(two_electron_optimizer_));

  if (status != 0) {
    for (int s = 0; s < ao_count_l; ++s) {
      const double normalization_l =
          ao_normalization_[xmvb::to_size(ao_offset_l + s)];
      for (int r = 0; r < ao_count_k; ++r) {
        const double normalization_kl =
            normalization_l *
            ao_normalization_[xmvb::to_size(ao_offset_k + r)];
        for (int q = 0; q < ao_count_j; ++q) {
          const double normalization_jkl =
              normalization_kl *
              ao_normalization_[xmvb::to_size(ao_offset_j + q)];
          for (int p = 0; p < ao_count_i; ++p) {
            const std::size_t index =
                xmvb::to_size(p) +
                xmvb::to_size(q) * ao_count_i +
                xmvb::to_size(r) * ao_count_i * ao_count_j +
                xmvb::to_size(s) * ao_count_i * ao_count_j * ao_count_k;
            buffer[index] *=
                normalization_jkl *
                ao_normalization_[xmvb::to_size(ao_offset_i + p)];
          }
        }
      }
    }
  } else {
    std::fill(buffer.begin(), buffer.end(), 0.0);
  }

  LibcintShellQuartet result;
  result.shell_i = shell_i;
  result.shell_j = shell_j;
  result.shell_k = shell_k;
  result.shell_l = shell_l;
  result.ao_offset_i = ao_offset_i;
  result.ao_offset_j = ao_offset_j;
  result.ao_offset_k = ao_offset_k;
  result.ao_offset_l = ao_offset_l;
  result.ao_count_i = ao_count_i;
  result.ao_count_j = ao_count_j;
  result.ao_count_k = ao_count_k;
  result.ao_count_l = ao_count_l;
  result.values = std::move(buffer);
  return result;
}

LibcintShellBlock LibcintDirectShellEvaluator::evaluate_auxiliary_metric_shell_pair(
    int left_auxiliary_shell,
    int right_auxiliary_shell) const {
  validate_auxiliary_shell_index(left_auxiliary_shell);
  validate_auxiliary_shell_index(right_auxiliary_shell);

  const int left_ao_offset = shell_ao_offset(*auxiliary_input_, left_auxiliary_shell);
  const int right_ao_offset = shell_ao_offset(*auxiliary_input_, right_auxiliary_shell);
  const int left_ao_count = shell_ao_count(*auxiliary_input_, left_auxiliary_shell);
  const int right_ao_count = shell_ao_count(*auxiliary_input_, right_auxiliary_shell);

  std::vector<double> buffer(
      xmvb::to_size(left_ao_count) * right_ao_count,
      0.0);
  FINT shell_pair[2] = {left_auxiliary_shell, right_auxiliary_shell};
  const FINT status = cint2c2e_cart(
      buffer.data(),
      shell_pair,
      const_cast<int*>(auxiliary_input_->atm.data()),
      auxiliary_input_->n_atoms,
      const_cast<int*>(auxiliary_input_->bas.data()),
      auxiliary_input_->n_shells,
      const_cast<double*>(auxiliary_input_->env.data()),
      reinterpret_cast<CINTOpt*>(auxiliary_metric_optimizer_));

  if (status != 0) {
    for (int column = 0; column < right_ao_count; ++column) {
      const double right_normalization =
          auxiliary_ao_normalization_[xmvb::to_size(right_ao_offset + column)];
      for (int row = 0; row < left_ao_count; ++row) {
        const std::size_t index =
            xmvb::to_size(row) +
            xmvb::to_size(column) * left_ao_count;
        buffer[index] *=
            auxiliary_ao_normalization_[xmvb::to_size(left_ao_offset + row)] *
            right_normalization;
      }
    }
  } else {
    std::fill(buffer.begin(), buffer.end(), 0.0);
  }

  LibcintShellBlock result;
  result.left_shell = left_auxiliary_shell;
  result.right_shell = right_auxiliary_shell;
  result.left_ao_offset = left_ao_offset;
  result.right_ao_offset = right_ao_offset;
  result.left_ao_count = left_ao_count;
  result.right_ao_count = right_ao_count;
  result.values = std::move(buffer);
  return result;
}

LibcintThreeCenterShellBlock LibcintDirectShellEvaluator::evaluate_three_center_shell_block(
    int primary_left_shell,
    int primary_right_shell,
    int auxiliary_shell) const {
  validate_shell_index(primary_left_shell);
  validate_shell_index(primary_right_shell);
  validate_auxiliary_shell_index(auxiliary_shell);

  const int combined_auxiliary_shell = input_.n_shells + auxiliary_shell;
  const int primary_left_ao_offset = shell_ao_offset(input_, primary_left_shell);
  const int primary_right_ao_offset = shell_ao_offset(input_, primary_right_shell);
  const int auxiliary_ao_offset = shell_ao_offset(*auxiliary_input_, auxiliary_shell);
  const int primary_left_ao_count = shell_ao_count(input_, primary_left_shell);
  const int primary_right_ao_count = shell_ao_count(input_, primary_right_shell);
  const int auxiliary_ao_count = shell_ao_count(*auxiliary_input_, auxiliary_shell);

  std::vector<double> buffer(
      xmvb::to_size(primary_left_ao_count) *
          primary_right_ao_count * auxiliary_ao_count,
      0.0);
  FINT shell_triple[3] = {
      primary_left_shell,
      primary_right_shell,
      combined_auxiliary_shell,
  };
  const FINT status = cint3c2e_cart(
      buffer.data(),
      shell_triple,
      const_cast<int*>(combined_atm_.data()),
      input_.n_atoms,
      const_cast<int*>(combined_bas_.data()),
      input_.n_shells + auxiliary_input_->n_shells,
      const_cast<double*>(combined_env_.data()),
      reinterpret_cast<CINTOpt*>(three_center_optimizer_));

  if (status != 0) {
    for (int auxiliary_local = 0; auxiliary_local < auxiliary_ao_count; ++auxiliary_local) {
      const double auxiliary_normalization =
          auxiliary_ao_normalization_[xmvb::to_size(
              auxiliary_ao_offset + auxiliary_local)];
      for (int right_local = 0; right_local < primary_right_ao_count; ++right_local) {
        const double right_normalization =
            ao_normalization_[xmvb::to_size(primary_right_ao_offset + right_local)];
        for (int left_local = 0; left_local < primary_left_ao_count; ++left_local) {
          const std::size_t index =
              xmvb::to_size(left_local) +
              xmvb::to_size(right_local) * primary_left_ao_count +
              xmvb::to_size(auxiliary_local) * primary_left_ao_count *
                  primary_right_ao_count;
          buffer[index] *=
              ao_normalization_[xmvb::to_size(primary_left_ao_offset + left_local)] *
              right_normalization * auxiliary_normalization;
        }
      }
    }
  } else {
    std::fill(buffer.begin(), buffer.end(), 0.0);
  }

  LibcintThreeCenterShellBlock result;
  result.primary_left_shell = primary_left_shell;
  result.primary_right_shell = primary_right_shell;
  result.auxiliary_shell = auxiliary_shell;
  result.primary_left_ao_offset = primary_left_ao_offset;
  result.primary_right_ao_offset = primary_right_ao_offset;
  result.auxiliary_ao_offset = auxiliary_ao_offset;
  result.primary_left_ao_count = primary_left_ao_count;
  result.primary_right_ao_count = primary_right_ao_count;
  result.auxiliary_ao_count = auxiliary_ao_count;
  result.values = std::move(buffer);
  return result;
}

double LibcintDirectShellEvaluator::evaluate_max_abs_raw_two_electron_shell_pair(
    int shell_i,
    int shell_j) const {
  validate_shell_index(shell_i);
  validate_shell_index(shell_j);

  const int ao_count_i = shell_ao_count(input_, shell_i);
  const int ao_count_j = shell_ao_count(input_, shell_j);
  std::vector<double> buffer(
      xmvb::to_size(ao_count_i) *
          ao_count_j * ao_count_i * ao_count_j,
      0.0);
  FINT shell_quartet[4] = {shell_i, shell_j, shell_i, shell_j};
  const FINT status = cint2e_cart(
      buffer.data(),
      shell_quartet,
      const_cast<int*>(input_.atm.data()),
      input_.n_atoms,
      const_cast<int*>(input_.bas.data()),
      input_.n_shells,
      const_cast<double*>(input_.env.data()),
      reinterpret_cast<CINTOpt*>(two_electron_optimizer_));
  if (status == 0) {
    return 0.0;
  }

  double max_abs_value = 0.0;
  for (double value : buffer) {
    const double abs_value = std::abs(value);
    if (abs_value > max_abs_value) {
      max_abs_value = abs_value;
    }
  }
  return max_abs_value;
}

}  // namespace xmvb::vb
