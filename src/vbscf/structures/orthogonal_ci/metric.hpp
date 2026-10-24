#pragma once

#include "mcscf/ci/string_table.hpp"

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "vbscf/structures/overlap/spin_metric.hpp"

namespace xmvb::vb {

class StructureAction;
class ExteriorOrbitalTransform;

/** @brief Exact complete-space overlap inverse with occupation-block factors.
 * Borrows an immutable action, which must outlive this object. Completeness of
 * the selected spin sector is required, not just completeness of its carrier.
 * No global dense structure overlap or expansion matrix is stored.
 */
class StructureMetricInverse {
public:
  StructureMetricInverse(
      const StructureAction& action,
      const xmvb::mcscf::StringTable& alpha_strings,
      const xmvb::mcscf::StringTable& beta_strings,
      bool complete_space,
      std::size_t factor_byte_limit,
      std::size_t workspace_byte_limit);
  StructureMetricInverse(const StructureMetricInverse&) = delete;
  StructureMetricInverse& operator=(const StructureMetricInverse&) = delete;
  StructureMetricInverse(StructureMetricInverse&&) noexcept = default;
  StructureMetricInverse& operator=(StructureMetricInverse&&) noexcept = default;

  /** @brief Applies the overlap inverse, chunking RHS by the workspace budget. */
  Eigen::MatrixXd apply(const Eigen::Ref<const Eigen::MatrixXd>& rhs) const;

  /** @brief Complete-sector M^-1 F^T D^-1 F M^-1 with cancelling maps removed.
   * With F=T W and G=W^T W this is G^-1 W^T T^-1 D^-1 T^-T W G^-1.
   * Uses the existing RHS workspace bound; arbitrary nonclosed selections
   * must not use this complete-sector factorization.
   */
  Eigen::MatrixXd apply_carrier_inverse(
      const Eigen::Ref<const Eigen::MatrixXd>& rhs,
      const Eigen::Ref<const Eigen::VectorXd>& energies,double eigenvalue) const;

  /** @brief FULL-only isometric CI coordinates x=U^T T W C, U=W L^-T. */
  Eigen::MatrixXd to_ci_coordinates(const Eigen::Ref<const Eigen::MatrixXd>& coefficients) const;
  /** @brief Return structure coefficients J^-1 x without changing the physical state. */
  Eigen::MatrixXd from_ci_coordinates(const Eigen::Ref<const Eigen::MatrixXd>& coordinates) const;
  /** @brief Pull a CI covector back by J^T; also restores retained H/M Ritz images. */
  Eigen::MatrixXd ci_covector_to_structure(const Eigen::Ref<const Eigen::MatrixXd>& covector) const;
  /** @brief U^T H_orth U; no orbital exterior maps or overlap action. */
  Eigen::MatrixXd apply_ci_hamiltonian(const Eigen::Ref<const Eigen::MatrixXd>& coordinates) const;
  /** @brief U^T D^-1 U, congruent to the original complete-sector inverse. */
  Eigen::MatrixXd apply_ci_preconditioner(
      const Eigen::Ref<const Eigen::MatrixXd>& covector,
      const Eigen::Ref<const Eigen::VectorXd>& energies, double eigenvalue) const;

  /** @brief Trial metric inverse sharing immutable selected-sector factors. */
  Eigen::MatrixXd apply_at_transform(
      const Eigen::Ref<const Eigen::MatrixXd>& rhs,
      const ExteriorOrbitalTransform& alpha,
      const ExteriorOrbitalTransform& beta) const;

  /** @brief Reuse the factors after their same-topology action has been moved.
   * No factor or integral data are copied. The new action must outlive this
   * inverse and retain the original spin-string/structure indexing.
   */
  void rebind(const StructureAction& action);
  std::size_t factor_bytes() const noexcept { return spin_.factor_bytes(); }
  std::size_t block_count() const noexcept { return spin_.block_count(); }
  int max_block_size() const noexcept { return spin_.max_block_size(); }

  /** @brief Plans factor/workspace admission without allocating dense blocks. */
  static bool fits(
      const StructureAction& action,
      const xmvb::mcscf::StringTable& alpha_strings,
      const xmvb::mcscf::StringTable& beta_strings,
      std::size_t factor_byte_limit,
      std::size_t workspace_byte_limit);

private:
  const StructureAction* action_;
  SpinMetric spin_;
  int chunk_width_ = 0;
};

}  // namespace xmvb::vb
