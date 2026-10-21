#pragma once

#include <memory>
#include <Eigen/Core>

#include "cuda/memory/context.hpp"
#include "linalg/views.hpp"
#include "vbscf/structures/assembly/action.hpp"

namespace xmvb::cuda {

class DirectCiSigma;

/** Exact GPU H/M action in the original selected VB structure coordinates.
 * The sparse incidence and elementary orbital transformations are borrowed
 * from the CPU mathematical representation and uploaded once. Numerical
 * action work uses one RHS at a time and bounded channel slabs; no structure
 * square or dense compound orbital transformation is formed.
 */
class StructureAction {
 public:
  StructureAction(Context& context, const vb::StructureAction& source);
  ~StructureAction();
  StructureAction(const StructureAction&) = delete;
  StructureAction& operator=(const StructureAction&) = delete;

  void apply(linalg::ConstMatrixView vectors,
             linalg::MatrixView hamiltonian, linalg::MatrixView overlap) const;
  /** @brief Overlap alone; never evaluate sigma to prepare a metric. */
  void apply_overlap(linalg::ConstMatrixView vectors, linalg::MatrixView overlap) const;
  vb::StructureActionResult apply(
      const Eigen::Ref<const Eigen::MatrixXd>& vectors) const;
  int size() const noexcept;
  double graph_build_seconds() const noexcept;
  std::size_t graph_nodes() const noexcept;
  int n_structures() const noexcept { return size(); }
  const vb::StructureDiagonal& preconditioner_diagonal() const noexcept;
  /** @brief Replace only current numerical integrals/frame; topology is shared. */
  void update_integrals(linalg::ConstMatrixView one_electron,
      linalg::ConstMatrixView pair_kernel,linalg::ConstMatrixView upper);
  const DirectCiSigma& carrier_sigma() const noexcept;

  int n_alpha_strings() const noexcept;
  int n_beta_strings() const noexcept;
  /** @brief Exact sparse structure/product maps, without orbital transforms. */
  void expand_spin(linalg::ConstMatrixView structures, linalg::MatrixView carrier) const;
  void contract_spin(linalg::ConstMatrixView carrier, linalg::MatrixView structures) const;
  /** @brief In-place orbital maps on packed alpha-by-(beta*RHS) panels. */
  void apply_orbital_transform(linalg::MatrixView carrier) const;
  void apply_inverse_orbital_transform(linalg::MatrixView carrier) const;
  void apply_adjoint_orbital_transform(linalg::MatrixView carrier) const;
  void apply_carrier_hamiltonian(linalg::ConstMatrixView carrier,
                                linalg::MatrixView image) const;
  /** @brief Production FULL diagonal inverse; energies is a 1-by-RHS device view. */
  void divide_carrier_diagonal(linalg::MatrixView carrier,
      linalg::ConstMatrixView energies, double eigenvalue) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace xmvb::cuda
