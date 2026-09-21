#pragma once

#include <vector>

#include <Eigen/Core>

namespace xmvb::vb {

struct AcceptedPointContext;
struct VbScfInput;

struct AnalyticOrbitalDiagonal {
  Eigen::MatrixXd mo_coefficients;
  Eigen::MatrixXd rotation_diagonal;
  Eigen::MatrixXd inactive_inverse_transform;
  Eigen::MatrixXd active_inverse_transform;
};

/**
 * @brief MO-basis intermediates for the analytic fixed-structure rotation diagonal.
 *
 * The four-index arrays use row-major logical indexing. `ppaa(p,q,a,b)` is
 * `(pq|ab)` and `papa(p,a,q,b)` is `(pa|qb)`. Only the partial transforms
 * required by the diagonal are stored; a full MO ERI tensor is never formed.
 */
struct AnalyticDiagonalIntermediates {
  int n_core = 0;
  int n_active = 0;
  Eigen::MatrixXd h_core;
  Eigen::MatrixXd v_core;
  Eigen::MatrixXd j_pc;
  Eigen::MatrixXd k_pc;
  std::vector<double> ppaa;
  std::vector<double> papa;
};

/**
 * @brief Reconstructs the fully ERI-symmetric spin-free active 2-RDM.
 *
 * The input is the derivative with respect to the project's eight-fold packed
 * active ERIs. The returned tensor is indexed `(p,q,r,s)` in row-major order.
 * This symmetrized representative is sufficient for every orbital derivative
 * because AO/MO electron-repulsion integrals possess the same eight-fold
 * symmetry.
 */
std::vector<double> unpack_symmetric_active_two_rdm(
    const std::vector<double>& packed_eri_gradient,
    int n_active);

/**
 * @brief Builds the fixed-structure orthogonal-frame orbital Hessian diagonal.
 *
 * The result is an `n_mo x n_mo` matrix. Entry `(p,q)` is the diagonal
 * curvature associated with the antisymmetric orbital rotation between
 * orbitals `p` and `q`, before selecting the nonredundant rotation mask.
 */
Eigen::MatrixXd build_analytic_rotation_hessian_diagonal(
    const AnalyticDiagonalIntermediates& integrals,
    const Eigen::Ref<const Eigen::MatrixXd>& active_one_rdm,
    const std::vector<double>& packed_active_eri_gradient);

/**
 * @brief Builds the analytic fixed-structure rotation diagonal at a VBSCF point.
 *
 * HAO and OEO use the same accepted AO-metric orthogonal frame. Sparse
 * supports enter only when this model is pulled into the orbital chart.
 */
AnalyticOrbitalDiagonal build_analytic_orbital_diagonal(
    const VbScfInput& input,
    const AcceptedPointContext& accepted);

}  // namespace xmvb::vb
