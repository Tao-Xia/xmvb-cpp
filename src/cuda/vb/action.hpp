#pragma once

#include <cstddef>
#include <memory>
#include <cstdint>
#include <Eigen/Core>

#include "cuda/memory/context.hpp"
#include "cuda/vb/frame_identity.hpp"
#include "linalg/views.hpp"
#include "vbscf/structures/assembly/action.hpp"

namespace xmvb::cuda {

class DirectCiSigma;
class ExteriorTransform;
class ProductContraction;

/** Exact W^T op W maps, evaluated with bounded spin-factor panels. */
enum class ProductMap {
    Orbital,
    InverseOrbital,
    AdjointOrbital,
    InverseAdjointOrbital,
    IdentityHamiltonian,
    IdentityDiagonalInverse,
    InverseMetric,
    InverseOrbitalDiagonalInverseAdjointOrbital
};

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
               linalg::MatrixView hamiltonian,
               linalg::MatrixView overlap) const;
    /** H alone when the caller already owns the exact selected overlap. */
    void apply_hamiltonian(linalg::ConstMatrixView vectors, linalg::MatrixView hamiltonian) const;
    /** @brief Overlap alone; never evaluate sigma to prepare a metric. */
    void apply_overlap(linalg::ConstMatrixView vectors, linalg::MatrixView overlap) const;
    vb::StructureActionResult apply(const Eigen::Ref<const Eigen::MatrixXd>& vectors) const;
    int size() const noexcept;
    int n_structures() const noexcept {
        return size();
    }
    /** Frozen source-point CPU diagnostic only; dynamic device solvers use
   * diagonal_model() and must never use this after a frame/integral update.
   */
    const vb::StructureDiagonal& preconditioner_diagonal() const noexcept;
    /** CPU-equivalent Jacobi model from nonorthogonal spin-product self H/M.
   * Squared combined sparse expansion coefficients contract the self values.
   * This is not the coherent selected-structure diagonal. Outputs stay on GPU.
   */
    void diagonal_model(linalg::MatrixView hamiltonian, linalg::MatrixView overlap) const;
    /** Exact current-frame selected overlap W^T (G_alpha tensor G_beta) W.
   * Builds bounded single-spin Gram and touched-product tiles on the device;
   * no full product carrier, structure unit-action loop, or determinant-square
   * matrix is formed. The caller owns the overwritten column-major output.
   */
    void build_selected_metric(linalg::MatrixView overlap) const;
    std::uint64_t version() const noexcept;
    /** Return idle action workspaces to the stream-ordered pool for bounded
   * temporary panels. Immutable topology and point-owned images
   * remain resident.
   */
    void reserve_workspace(std::size_t bytes) const;
    /** Change the orbital frame without modifying the current sigma kernel.
   * A nonempty identity certifies immutable frame contents and permits reuse.
   * Without one, every call updates the frame conservatively.
   */
    void update_frame(linalg::ConstMatrixView upper, OrbitalFrameIdentity identity = {});
    /** @brief Replace only current numerical integrals/frame; topology is shared. */
    void update_integrals(linalg::ConstMatrixView one_electron,
                          linalg::ConstMatrixView pair_kernel,
                          linalg::ConstMatrixView upper,
                          OrbitalFrameIdentity identity = {});
    const DirectCiSigma& carrier_sigma() const noexcept;
    const ExteriorTransform& alpha_transform() const noexcept;
    const ExteriorTransform& beta_transform() const noexcept;
    const vb::StructureAction::OrthogonalView& orthogonal_view() const noexcept;
    /** Independent numerical executor reusing this action's uploaded exact support. */
    std::unique_ptr<ProductContraction> make_product_contraction() const;
    const std::int64_t* product_indices_data() const noexcept;
    const std::int64_t* term_offsets_data() const noexcept;
    const std::int64_t* term_structures_data() const noexcept;
    const double* term_weights_data() const noexcept;
    std::int64_t n_product_groups() const noexcept;
    int n_alpha_strings() const noexcept;
    int n_beta_strings() const noexcept;
    /** Maximum orthogonal-carrier amplitude of a structure response.
   * Matches direct-CI CIAH clipping using bounded source-factor panels.
   * Caller must select the intended orbital frame first.
   */
    double maximum_orthogonal_component(linalg::ConstMatrixView structure) const;
    /** FULL-coordinate boundary and orthogonal CI maps without full carriers. */
    void apply_full_structure_map(ProductMap map,
                                  linalg::ConstMatrixView source,
                                  linalg::MatrixView image,
                                  linalg::ConstMatrixView energies = {},
                                  double eigenvalue = 0) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace xmvb::cuda
