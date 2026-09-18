#pragma once

#include <Eigen/Core>

#include "vbscf/optimization/coupled/operator.hpp"
#include "vbscf/optimization/krylov/minres.hpp"

namespace xmvb::vb {

/** @brief Accepted-point diagonals needed by the structure-response model. */
struct StructureResponsePreconditionerData {
  Eigen::VectorXd hamiltonian_diagonal;
  Eigen::VectorXd overlap_diagonal;
  Eigen::VectorXd selected_energies;
  /** @brief Accepted @f$SC@f$ columns in selected-state order. */
  Eigen::MatrixXd overlap_selected;
};

/**
 * @brief Builds the SPD inverse action for selected-subspace response space.
 *
 * The returned action accepts and returns exactly
 * `response_layout.response_size()` coordinates.  It is therefore directly
 * usable by response-only Krylov solves and `CoupledSubspaceSolver::expand`;
 * no artificial orbital block is required.
 */
SymmetricOperatorAction make_structure_response_inverse_preconditioner(
    const SelectedSubspaceResponseLayout& response_layout,
    const StructureResponsePreconditionerData& structure_data);

/**
 * @brief Builds an SPD inverse preconditioner for the coupled Newton system.
 *
 * The orbital block is supplied by the caller and must be an SPD inverse
 * action for the current trust-region shift.  For selected state @f$i@f$ in
 * an equal-weight cluster, the response block uses
 *
 * @f[
 * D_i=\operatorname{diag}\left(
 * \max\{|H_{aa}-E_iS_{aa}|,\epsilon_i\}\right),\qquad
 * R_i=N^T D_i^{-1}N,
 * @f]
 *
 * where @f$N=(SC)_{\rm cluster}@f$ and @f$\epsilon_i@f$ is only a
 * floating-point rank floor.  Replacing the negative Schur block in the
 * bordered @f$LDL^T@f$ factorization by @f$R_i@f$ gives the SPD response
 * model
 *
 * @f[
 * P_i=L_i\operatorname{diag}(D_i,R_i)L_i^T,
 * \qquad
 * L_i=\begin{pmatrix}I&0\\N^TD_i^{-1}&I\end{pmatrix}.
 * @f]
 *
 * Thus for every nonzero residual @f$r@f$,
 * @f$r^TP_i^{-1}r=(L_i^{-1}r)^T
 * \operatorname{diag}(D_i^{-1},R_i^{-1})(L_i^{-1}r)>0@f$.
 * The small Schur eigensolve regularizes only eigenvalues below its
 * roundoff-scale numerical rank threshold; there is no physical or
 * molecule-specific parameter.
 *
 * The coefficient and full multiplier columns are read directly from the
 * layout's packed coordinates.  Both parts of a cluster carry the same
 * @f$\sqrt{2w}@f$ scale, so every linear triangular solve commutes with that
 * scale and the returned action is exactly in the symmetric packed basis.
 */
SymmetricOperatorAction make_coupled_block_inverse_preconditioner(
    int n_orbital_coordinates,
    const SelectedSubspaceResponseLayout& response_layout,
    const StructureResponsePreconditionerData& structure_data,
    SymmetricOperatorAction apply_orbital_inverse);

}  // namespace xmvb::vb
