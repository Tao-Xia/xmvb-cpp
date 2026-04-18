#pragma once

#include <memory>

#include <Eigen/Core>

#include "vb/pdft/ao_grid_values.hpp"
#include "vb/orbital/libcint_input.hpp"

namespace xmvb::vb::pdft {

/**
 * @brief Evaluates AO basis functions on grid points using libcint.
 *
 * This evaluator computes AO basis function values chi_mu(r) and
 * optionally their gradients at arbitrary grid points. The implementation
 * uses libcint for efficient evaluation of Gaussian basis functions.
 *
 * The evaluator is designed for batch processing of many grid points
 * to amortize setup costs and enable vectorization.
 */
class LibcintAoGridEvaluator {
public:
  struct LegacyAoBasisAdapter;

  /**
   * @brief Constructs evaluator from libcint input.
   *
   * @param libcint_input Libcint molecular input containing basis set
   *                      and atomic geometry information.
  */
  explicit LibcintAoGridEvaluator(const LibcintInput& libcint_input);

  ~LibcintAoGridEvaluator();

  LibcintAoGridEvaluator(const LibcintAoGridEvaluator&) = delete;
  LibcintAoGridEvaluator& operator=(const LibcintAoGridEvaluator&) = delete;
  LibcintAoGridEvaluator(LibcintAoGridEvaluator&&) noexcept;
  LibcintAoGridEvaluator& operator=(LibcintAoGridEvaluator&&) noexcept;

  /**
   * @brief Evaluates AO values at grid points.
   *
   * Computes chi_mu(r_g) for all basis functions mu and grid points g.
   *
   * @param grid_points Grid point coordinates (n_points x 3, column-major).
   * @return AO values at each grid point.
   */
  AoGridValues evaluate_values(const Eigen::MatrixXd& grid_points) const;

  /**
   * @brief Evaluates AO values and gradients at grid points.
   *
   * Computes both chi_mu(r_g) and d chi_mu / d x_d at all grid points.
   * This is needed for GGA functionals.
   *
   * @param grid_points Grid point coordinates (n_points x 3, column-major).
   * @return AO values and gradients at each grid point.
   */
  AoGridValues evaluate_values_and_gradients(
      const Eigen::MatrixXd& grid_points) const;

  /**
   * @brief Returns the number of AO basis functions.
   */
  int n_basis_functions() const { return n_basis_functions_; }

private:
  LibcintInput libcint_input_;
  std::unique_ptr<LegacyAoBasisAdapter> legacy_basis_adapter_;
  int n_basis_functions_;
};

}  // namespace xmvb::vb::pdft
