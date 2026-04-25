#pragma once

#include <Eigen/Core>

#include <cstdint>
#include <vector>

namespace xmvb::vb {

/**
 * @brief Raw AO integral data required by the C++ VBSCF matrix builders.
 */
struct AoIntegralInput {
  /**
   * @brief Number of AO basis functions.
   */
  int n_basis_functions = 0;

  /**
   * @brief Column-major AO core Hamiltonian matrix `HHF`.
   */
  Eigen::MatrixXd ao_core_hamiltonian_matrix;

  /**
   * @brief Sparse AO two-electron integral values `ggf`.
   */
  std::vector<double> ao_two_electron_integral_values;

  /**
   * @brief Flattened AO two-electron index table `g2eidx` with 4 entries per integral.
   */
  std::vector<int> ao_two_electron_integral_indices;

  /**
   * @brief Per-integral symmetry shift count for AO effective-one-electron kernels.
   *
   * Each entry stores how many factors of `0.5` must be applied to the raw AO
   * integral value before using it in the `ao_h1e` / `G11` contraction:
   *
   * - `+1` if `i == j`
   * - `+1` if `k == l`
   * - `+1` if `(i, j) == (k, l)`
   *
   * So the effective multiplier is `0.5 ^ shift`, with shifts in `{0, 1, 2, 3}`.
   * This cache is molecule-static and avoids repeating these branchy symmetry
   * checks inside every SCF iteration.
   */
  std::vector<std::uint8_t> ao_two_electron_integral_symmetry_shifts;

  /**
   * @brief Precomputed packed AO pair indices `(ij_pair, kl_pair)` with 2 entries per integral.
   *
   * This cache is molecule-static and can be reused across all SCF iterations
   * to avoid repeatedly converting AO four-index tuples into packed pair
   * indices inside hot ERI contraction loops.
   */
  std::vector<int> ao_two_electron_pair_indices;

  /**
   * @brief Row offsets for a molecule-static symmetric AO-pair interaction graph.
   *
   * Each row corresponds to one packed AO pair. The graph expands every AO
   * two-electron integral into one or two directed row entries so dense
   * active-space contractions can run as a row-wise sparse-matrix multiply
   * without thread-local full-size accumulation buffers.
   */
  std::vector<int> ao_two_electron_pair_graph_row_offsets;

  /**
   * @brief Column packed-AO-pair indices for the symmetric AO-pair graph.
   */
  std::vector<int> ao_two_electron_pair_graph_column_indices;

  /**
   * @brief Source AO-integral indices for each symmetric AO-pair graph entry.
   *
   * Values remain stored in `ao_two_electron_integral_values`; this array
   * points each graph edge back to its source integral so the static graph
   * cache does not duplicate the AO integral value buffer.
   */
  std::vector<int> ao_two_electron_pair_graph_integral_indices;

  /**
   * @brief Optional 10-entry-per-integral cache of AO matrix linear indices for `G11` kernels.
   *
   * Each integral stores the following column-major AO matrix indices in order:
   * `(ij, kl, ik, jl, il, jk, lj, ki, kj, li)`.
   *
   * These indices are reused by AO effective one-electron forward/backward
   * kernels to avoid recomputing repeated column-offset arithmetic inside hot
   * ERI contraction loops.
   */
  std::vector<int> ao_effective_one_electron_linear_indices;

  /**
   * @brief Row offsets for a molecule-static AO-H1E sparse operator graph.
   *
   * The exact AO effective-one-electron contraction is a linear map
   * `vec(G11) = K * vec(P11)` on the full AO matrix storage. This CSR row
   * graph stores that accepted-point-independent operator once per molecule so
   * hot exact_ctx HVP paths can run row-wise sparse contractions instead of
   * rescattering every ERI into six AO matrix entries on every matvec.
   */
  std::vector<int> ao_effective_one_electron_graph_row_offsets;

  /**
   * @brief Source AO-matrix linear indices for the AO-H1E sparse operator graph.
   *
   * These are full column-major AO matrix indices into `vec(P11)` or the
   * transpose pullback buffer, matching the row graph above.
   */
  std::vector<int> ao_effective_one_electron_graph_source_indices;

  /**
   * @brief Signed AO-H1E sparse operator weights for each row-graph edge.
   *
   * Each entry already includes the AO-integral symmetry multiplier and the
   * Coulomb/exchange prefactor (`+4` or `-1`), so applying the graph reduces
   * to a pure sparse matrix-vector multiply.
   */
  std::vector<double> ao_effective_one_electron_graph_signed_weights;

  /**
   * @brief Source-owned offsets for the transpose AO-H1E sparse operator graph.
   *
   * This CSC-style companion graph stores the same linear map as the CSR row
   * graph above, but grouped by source AO matrix entry. Exact-context HVP
   * kernels use it to apply `K^T * lambda` with thread-owned source blocks and
   * therefore avoid one full AO-matrix-sized transpose buffer per OpenMP
   * worker.
   */
  std::vector<int> ao_effective_one_electron_graph_transpose_source_offsets;

  /**
   * @brief Destination row indices for the transpose AO-H1E sparse graph.
   */
  std::vector<int> ao_effective_one_electron_graph_transpose_row_indices;

  /**
   * @brief Signed edge weights for the source-owned transpose AO-H1E graph.
   */
  std::vector<double> ao_effective_one_electron_graph_transpose_signed_weights;
};

}  // namespace xmvb::vb
