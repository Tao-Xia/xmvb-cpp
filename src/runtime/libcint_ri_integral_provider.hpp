#pragma once

#include <Eigen/Core>

#include <vector>

#include "runtime/libcint_auxiliary_basis_builder.hpp"
#include "vbscf/integrals/ao/libcint_input.hpp"

namespace xmvb::vb {

struct LibcintRiIntegralProviderOptions {
  LibcintAuxiliaryBasisBuilderOptions auxiliary_basis_options;
  double metric_eigenvalue_cutoff = 1.0e-10;
};

struct LibcintRiIntegralProviderResult {
  LibcintInput auxiliary_input;
  int n_basis_functions = 0;
  int n_auxiliary_functions = 0;
  int n_packed_ao_pairs = 0;

  /**
   * @brief Dense auxiliary Coulomb metric in the project-standard column-major storage.
   */
  Eigen::MatrixXd auxiliary_metric_matrix;

  /**
   * @brief Metric-whitened AO-pair RI factors `L_{A,P}`.
   *
   * Rows are auxiliary functions and columns are packed AO-pair indices. The
   * matrix uses the repository-wide column-major Eigen storage, so callers
   * must not assume that each logical row is contiguous in memory.
   */
  Eigen::MatrixXd metric_whitened_ao_pair_factors;

  /**
   * @brief Optional packed AO pair metric `g_{PQ} = sum_A L_{A,P} L_{A,Q}`.
   *
   * The optimized standard-RI path now consumes `metric_whitened_ao_pair_factors`
   * directly and may leave this compatibility buffer empty to avoid building
   * the much larger AO-pair Gram matrix.
   */
  std::vector<double> packed_ao_pair_metric;
};

/**
 * @brief Builds molecule-static AO-side RI factors on the clean C++/libcint path.
 *
 * The provider generates a Coulomb-fitting auxiliary basis, materializes the
 * auxiliary metric and three-center tensors directly through libcint, and
 * returns metric-whitened AO-pair factors suitable for active-space RI
 * contraction.
 */
class LibcintRiIntegralProvider {
public:
  LibcintRiIntegralProviderResult build(
      const LibcintInput& primary_input,
      const LibcintRiIntegralProviderOptions& options = {}) const;

  LibcintRiIntegralProviderResult build(
      const LibcintInput& primary_input,
      const LibcintInput& auxiliary_input,
      const LibcintRiIntegralProviderOptions& options) const;
};

}  // namespace xmvb::vb
