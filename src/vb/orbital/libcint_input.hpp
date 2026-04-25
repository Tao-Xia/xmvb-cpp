#pragma once

#include <vector>

namespace xmvb::vb {

/**
 * @brief Raw libcint arrays required to regenerate AO integrals directly in C++.
 *
 * These arrays mirror the molecule-static `atm`, `bas`, `basidx`, and `env`
 * buffers currently prepared inside the legacy runtime. Keeping them inside
 * `CppVbInput` allows the C++ side to gradually take ownership of AO integral
 * generation without depending on legacy integral materialization forever.
 */
struct LibcintInput {
  /**
   * @brief Number of atoms represented by `atm`.
   */
  int n_atoms = 0;

  /**
   * @brief Number of shells represented by `bas`.
   */
  int n_shells = 0;

  /**
   * @brief Number of Gaussian primitives used to size `env`.
   */
  int n_gaussian_primitives = 0;

  /**
   * @brief Raw libcint atom table.
   */
  std::vector<int> atm;

  /**
   * @brief Raw libcint basis table.
   */
  std::vector<int> bas;

  /**
   * @brief AO start/count table per shell used by the legacy runtime.
   */
  std::vector<int> basidx;

  /**
   * @brief Raw libcint environment array storing coordinates, exponents, and coefficients.
   */
  std::vector<double> env;
};

}  // namespace xmvb::vb
