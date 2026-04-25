#include "vb/orbital/libcint_input_utils.hpp"

#include <cmath>
#include <stdexcept>
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
}

namespace xmvb::vb {

void validate_libcint_input_shape(const LibcintInput& input) {
  if (input.n_atoms <= 0) {
    throw std::invalid_argument("LibcintInput must contain at least one atom");
  }
  if (input.n_shells <= 0) {
    throw std::invalid_argument("LibcintInput must contain at least one shell");
  }
  if (input.atm.size() != input.n_atoms * ATM_SLOTS) {
    throw std::invalid_argument("LibcintInput atm size mismatch");
  }
  if (input.bas.size() != input.n_shells * BAS_SLOTS) {
    throw std::invalid_argument("LibcintInput bas size mismatch");
  }
  if (input.basidx.size() != input.n_shells * 2) {
    throw std::invalid_argument("LibcintInput basidx size mismatch");
  }
  if (input.env.empty()) {
    throw std::invalid_argument("LibcintInput env must not be empty");
  }
}

int shell_ao_offset(const LibcintInput& input, int shell_index) {
  validate_libcint_input_shape(input);
  if (shell_index < 0 || shell_index >= input.n_shells) {
    throw std::out_of_range("shell index out of range");
  }
  return input.basidx[shell_index * 2];
}

int shell_ao_count(const LibcintInput& input, int shell_index) {
  validate_libcint_input_shape(input);
  if (shell_index < 0 || shell_index >= input.n_shells) {
    throw std::out_of_range("shell index out of range");
  }
  return input.basidx[shell_index * 2 + 1];
}

int infer_n_basis_functions(const LibcintInput& input) {
  validate_libcint_input_shape(input);
  const int last_shell_index = input.n_shells - 1;
  const int n_basis_functions =
      shell_ao_offset(input, last_shell_index) +
      shell_ao_count(input, last_shell_index);
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("LibcintInput basis-function count must be positive");
  }
  return n_basis_functions;
}

std::vector<double> build_cartesian_ao_normalization(const LibcintInput& input) {
  const int n_basis_functions = infer_n_basis_functions(input);
  std::vector<double> ao_normalization(
      n_basis_functions,
      0.0);

  // The raw libcint Cartesian shells are not unit normalized shell-by-shell in
  // the standalone runtime snapshot.  Recomputing the overlap diagonal here
  // gives the exact per-AO scale factor needed to match the normalized AO
  // overlap matrix used by the VB orbital linear algebra.
  for (int shell_index = 0; shell_index < input.n_shells; ++shell_index) {
    const int ao_offset = shell_ao_offset(input, shell_index);
    const int ao_count = shell_ao_count(input, shell_index);
    std::vector<double> overlap_buffer(
        ao_count * ao_count,
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
          overlap_buffer[local_ao * (ao_count + 1)];
      if (!(diagonal_overlap > 0.0)) {
        throw std::runtime_error("non-positive AO overlap diagonal encountered");
      }
      ao_normalization[ao_offset + local_ao] =
          std::sqrt(1.0 / diagonal_overlap);
    }
  }

  return ao_normalization;
}

}  // namespace xmvb::vb
