#pragma once

#include <memory>
#include <cstdint>
#include <array>
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
  /** Frozen source-point CPU diagnostic only; dynamic device solvers use
   * diagonal_model() and must never use this after a frame/integral update.
   */
  const vb::StructureDiagonal& preconditioner_diagonal() const noexcept;
  /** Dynamic, positive-overlap carrier Jacobi model (not exact H/M diagonals).
   * W squared weights the current orthogonal carrier diagonal and the current
   * exterior determinant scales. Outputs are device column vectors.
   */
  void diagonal_model(linalg::MatrixView hamiltonian,
                      linalg::MatrixView overlap) const;
  /** Exact current-frame selected overlap W^T (G_alpha tensor G_beta) W.
   * Builds bounded single-spin Gram and touched-product tiles on the device;
   * no full product carrier, structure unit-action loop, or determinant-square
   * matrix is formed. The caller owns the overwritten column-major output.
   */
  void build_selected_metric(linalg::MatrixView overlap) const;
  std::uint64_t version() const noexcept;
  /** Change the orbital frame without modifying the current sigma kernel. */
  void update_frame(linalg::ConstMatrixView upper);
  /** @brief Replace only current numerical integrals/frame; topology is shared. */
  void update_integrals(linalg::ConstMatrixView one_electron,
      linalg::ConstMatrixView pair_kernel,linalg::ConstMatrixView upper);
  const DirectCiSigma& carrier_sigma() const noexcept;
  /** Borrow the two H/S carriers between complete structure actions.
   * Device consumers are serialized on Context's stream. Views must not
   * remain live across another H/S action; captured addresses are unchanged.
   */
  std::array<linalg::MatrixView,2> carrier_workspace() const noexcept;

  int n_alpha_strings() const noexcept;
  int n_beta_strings() const noexcept;
  /** @brief Exact sparse structure/product maps, without orbital transforms. */
  void expand_spin(linalg::ConstMatrixView structures, linalg::MatrixView carrier) const;
  void contract_spin(linalg::ConstMatrixView carrier, linalg::MatrixView structures) const;
  /** @brief In-place orbital maps on packed alpha-by-(beta*RHS) panels. */
  void apply_orbital_transform(linalg::MatrixView carrier) const;
  void apply_inverse_orbital_transform(linalg::MatrixView carrier) const;
  void apply_inverse_adjoint_orbital_transform(linalg::MatrixView carrier) const;
  void apply_adjoint_orbital_transform(linalg::MatrixView carrier) const;
  void apply_carrier_hamiltonian(linalg::ConstMatrixView carrier,
                                linalg::MatrixView image) const;
  /** Maximum orthogonal-carrier amplitude of a structure response.
   * Matches direct-CI CIAH clipping; reuses the action's carrier workspace.
   * Caller must select the intended orbital frame first.
   */
  double maximum_orthogonal_component(linalg::ConstMatrixView structure) const;
  /** @brief Production FULL diagonal inverse; energies is a 1-by-RHS device view. */
  void divide_carrier_diagonal(linalg::MatrixView carrier,
      linalg::ConstMatrixView energies, double eigenvalue) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace xmvb::cuda
