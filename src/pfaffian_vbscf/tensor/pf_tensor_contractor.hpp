#pragma once

#include <cstddef>

#include "pfaffian_vbscf/types/eigen_types.hpp"

namespace xmvb::pfaffian_vbscf {

/**
 * @brief Performs packed two-electron tensor contractions for Pfaffian-VBSCF.
 *
 * The `ggo` buffer uses the same packed convention as the legacy active-space
 * tensor: first pack an unordered orbital pair, then pack an unordered
 * pair-of-pairs. All contraction routines assume that storage layout.
 */
class PfTensorContractor {
public:
  /**
   * @brief Returns the number of packed unordered orbital pairs for `m` orbitals.
   *
   * @param n_active_orbitals Number of active spatial orbitals.
   * @return std::size_t Packed pair count `m (m + 1) / 2`.
   */
  static std::size_t packed_pair_count(int n_active_orbitals);

  /**
   * @brief Returns the packed storage size of the two-electron tensor.
   *
   * @param n_active_orbitals Number of active spatial orbitals.
   * @return std::size_t Packed tensor size for the legacy `ggo` layout.
   */
  static std::size_t packed_two_electron_count(int n_active_orbitals);

  /**
   * @brief Evaluates the exchange-like contraction
   * `sum_{q,p,s,r} g_{qp,sr} A_{pr} B_{sq}`.
   *
   * @param ggo Packed active-space two-electron integrals.
   * @param n_active_orbitals Number of active spatial orbitals.
   * @param left Dense matrix `A`.
   * @param right Dense matrix `B`.
   * @return double Contraction value.
   */
  static double contract_exchange(
      const ScalarBuffer& ggo,
      int n_active_orbitals,
      const ConstMatrixRef& left,
      const ConstMatrixRef& right);

  /**
   * @brief Evaluates the direct contraction
   * `sum_{q,p,s,r} g_{qp,sr} A_{qp} B_{sr}`.
   *
   * @param ggo Packed active-space two-electron integrals.
   * @param n_active_orbitals Number of active spatial orbitals.
   * @param left Dense matrix `A`.
   * @param right Dense matrix `B`.
   * @return double Contraction value.
   */
  static double contract_direct(
      const ScalarBuffer& ggo,
      int n_active_orbitals,
      const ConstMatrixRef& left,
      const ConstMatrixRef& right);

  /**
   * @brief Evaluates the Coulomb-like contraction
   * `sum_{q,p,s,r} g_{qp,sr} A_{rp} B_{qs}`.
   *
   * @param ggo Packed active-space two-electron integrals.
   * @param n_active_orbitals Number of active spatial orbitals.
   * @param left Dense matrix `A`.
   * @param right Dense matrix `B`.
   * @return double Contraction value.
   */
  static double contract_coulomb(
      const ScalarBuffer& ggo,
      int n_active_orbitals,
      const ConstMatrixRef& left,
      const ConstMatrixRef& right);

  /**
   * @brief Evaluates the same-spin antisymmetrized exchange contraction
   * `sum_{q,p,s,r} g_{qp,sr} (A_{pr} - A_{rp}) B_{sq}`.
   *
   * @param ggo Packed active-space two-electron integrals.
   * @param n_active_orbitals Number of active spatial orbitals.
   * @param left Dense matrix `A`.
   * @param right Dense matrix `B`.
   * @return double Contraction value.
   */
  static double contract_same_spin_exchange(
      const ScalarBuffer& ggo,
      int n_active_orbitals,
      const ConstMatrixRef& left,
      const ConstMatrixRef& right);

  /**
   * @brief Evaluates the same-spin antisymmetrized Coulomb contraction
   * `sum_{q,p,s,r} g_{qp,sr} (A_{rp} - A_{pr}) B_{qs}`.
   *
   * @param ggo Packed active-space two-electron integrals.
   * @param n_active_orbitals Number of active spatial orbitals.
   * @param left Dense matrix `A`.
   * @param right Dense matrix `B`.
   * @return double Contraction value.
   */
  static double contract_same_spin_coulomb(
      const ScalarBuffer& ggo,
      int n_active_orbitals,
      const ConstMatrixRef& left,
      const ConstMatrixRef& right);

  /**
   * @brief Evaluates the exact same-spin separable contraction
   * `sum_{p<r,q<s} (g_{qp,sr} - g_{qr,sp}) A_{qp} B_{sr}`.
   *
   * This matches the determinant/generic same-spin 2-RDM assembly for products
   * of first-order spatial derivatives.
   */
  static double contract_same_spin_separable(
      const ScalarBuffer& ggo,
      int n_active_orbitals,
      const ConstMatrixRef& left,
      const ConstMatrixRef& right);

  /**
   * @brief Evaluates the exact same-spin bridge contraction
   * `sum_{p<r,q<s} (g_{qp,sr} - g_{qr,sp}) A_{qr} B_{sp}`.
   */
  static double contract_same_spin_bridge(
      const ScalarBuffer& ggo,
      int n_active_orbitals,
      const ConstMatrixRef& left,
      const ConstMatrixRef& right);

  /**
   * @brief Evaluates the fused closed-shell unrestricted quartet contribution
   * `EC(B, M) - 1/2 EC(D_r, A) - 1/2 EC(D, V) + Dir(X, C) + Dir(C, U)`.
   *
   * This helper is specialized to the current exact closed-shell spatial-kernel
   * forward path and accumulates all unrestricted `g_{qp,sr}` contributions in
   * a single packed-tensor traversal.
   */
  static double contract_closed_shell_full_linear_combo(
      const ScalarBuffer& ggo,
      int n_active_orbitals,
      const ConstMatrixRef& b,
      const ConstMatrixRef& m_pair,
      const ConstMatrixRef& d_r,
      const ConstMatrixRef& a,
      const ConstMatrixRef& d,
      const ConstMatrixRef& v,
      const ConstMatrixRef& x,
      const ConstMatrixRef& c,
      const ConstMatrixRef& u);

  /**
   * @brief Evaluates the fused closed-shell same-spin antisymmetrized
   * contribution `2 [Sep(X, C) - Br(X, C)] + 2 [Sep(C, U) - Br(C, U)]`.
   *
   * This helper is specialized to the current exact closed-shell spatial-kernel
   * forward path and accumulates the restricted
   * `(g_{qp,sr} - g_{qr,sp})` contributions in a single traversal.
   */
  static double contract_closed_shell_same_spin_asym_linear_combo(
      const ScalarBuffer& ggo,
      int n_active_orbitals,
      const ConstMatrixRef& x,
      const ConstMatrixRef& c,
      const ConstMatrixRef& u);

  /**
   * @brief Accumulates operand adjoints for the exchange contraction
   * `weight * sum g_{qp,sr} A_{pr} B_{sq}`.
   *
   * @param weight Scalar prefactor of the contraction.
   * @param left Dense matrix `A`.
   * @param right Dense matrix `B`.
   * @param ggo Packed active-space two-electron integrals.
   * @param left_adjoint Output adjoint of `A`, pre-sized to `m x m`.
   * @param right_adjoint Output adjoint of `B`, pre-sized to `m x m`.
   */
  static void compute_exchange_operand_adjoints(
      double weight,
      const ConstMatrixRef& left,
      const ConstMatrixRef& right,
      const ScalarBuffer& ggo,
      MatrixRef left_adjoint,
      MatrixRef right_adjoint);

  /**
   * @brief Accumulates operand adjoints for the Coulomb contraction
   * `weight * sum g_{qp,sr} A_{rp} B_{qs}`.
   *
   * @param weight Scalar prefactor of the contraction.
   * @param left Dense matrix `A`.
   * @param right Dense matrix `B`.
   * @param ggo Packed active-space two-electron integrals.
   * @param left_adjoint Output adjoint of `A`, pre-sized to `m x m`.
   * @param right_adjoint Output adjoint of `B`, pre-sized to `m x m`.
   */
  static void compute_coulomb_operand_adjoints(
      double weight,
      const ConstMatrixRef& left,
      const ConstMatrixRef& right,
      const ScalarBuffer& ggo,
      MatrixRef left_adjoint,
      MatrixRef right_adjoint);

  /**
   * @brief Accumulates operand adjoints for the direct contraction
   * `weight * sum g_{qp,sr} A_{qp} B_{sr}`.
   *
   * @param weight Scalar prefactor of the contraction.
   * @param left Dense matrix `A`.
   * @param right Dense matrix `B`.
   * @param ggo Packed active-space two-electron integrals.
   * @param left_adjoint Output adjoint of `A`, pre-sized to `m x m`.
   * @param right_adjoint Output adjoint of `B`, pre-sized to `m x m`.
   */
  static void compute_direct_operand_adjoints(
      double weight,
      const ConstMatrixRef& left,
      const ConstMatrixRef& right,
      const ScalarBuffer& ggo,
      MatrixRef left_adjoint,
      MatrixRef right_adjoint);

  /**
   * @brief Accumulates operand adjoints for the same-spin exchange contraction
   * `weight * sum g_{qp,sr} (A_{pr} - A_{rp}) B_{sq}`.
   *
   * @param weight Scalar prefactor of the contraction.
   * @param left Dense matrix `A`.
   * @param right Dense matrix `B`.
   * @param ggo Packed active-space two-electron integrals.
   * @param left_adjoint Output adjoint of `A`, pre-sized to `m x m`.
   * @param right_adjoint Output adjoint of `B`, pre-sized to `m x m`.
   */
  static void compute_same_spin_exchange_operand_adjoints(
      double weight,
      const ConstMatrixRef& left,
      const ConstMatrixRef& right,
      const ScalarBuffer& ggo,
      MatrixRef left_adjoint,
      MatrixRef right_adjoint);

  /**
   * @brief Accumulates operand adjoints for the same-spin Coulomb contraction
   * `weight * sum g_{qp,sr} (A_{rp} - A_{pr}) B_{qs}`.
   *
   * @param weight Scalar prefactor of the contraction.
   * @param left Dense matrix `A`.
   * @param right Dense matrix `B`.
   * @param ggo Packed active-space two-electron integrals.
   * @param left_adjoint Output adjoint of `A`, pre-sized to `m x m`.
   * @param right_adjoint Output adjoint of `B`, pre-sized to `m x m`.
   */
  static void compute_same_spin_coulomb_operand_adjoints(
      double weight,
      const ConstMatrixRef& left,
      const ConstMatrixRef& right,
      const ScalarBuffer& ggo,
      MatrixRef left_adjoint,
      MatrixRef right_adjoint);

  /**
   * @brief Accumulates operand adjoints for the exact same-spin separable
   * contraction `weight * sum_{p<r,q<s} (g_{qp,sr} - g_{qr,sp}) A_{qp} B_{sr}`.
   */
  static void compute_same_spin_separable_operand_adjoints(
      double weight,
      const ConstMatrixRef& left,
      const ConstMatrixRef& right,
      const ScalarBuffer& ggo,
      MatrixRef left_adjoint,
      MatrixRef right_adjoint);

  /**
   * @brief Accumulates operand adjoints for the exact same-spin bridge
   * contraction `weight * sum_{p<r,q<s} (g_{qp,sr} - g_{qr,sp}) A_{qr} B_{sp}`.
   */
  static void compute_same_spin_bridge_operand_adjoints(
      double weight,
      const ConstMatrixRef& left,
      const ConstMatrixRef& right,
      const ScalarBuffer& ggo,
      MatrixRef left_adjoint,
      MatrixRef right_adjoint);

  /**
   * @brief Adds the direct exchange outer-product coefficient tensor
   * `A_{pr} B_{sq}` into `ggo_grad`.
   *
   * This is the mix-spin / non-antisymmetrized update path.
   *
   * @param weight Scalar prefactor applied to every outer-product coefficient.
   * @param left Dense matrix `A`.
   * @param right Dense matrix `B`.
   * @param ggo_grad Output packed gradient buffer. If empty, it is initialized
   *     to the correct size and zero-filled before accumulation.
   */
  static void add_exchange_outer_product(
      double weight,
      const ConstMatrixRef& left,
      const ConstMatrixRef& right,
      ScalarBuffer* ggo_grad);

  /**
   * @brief Adds the direct Coulomb outer-product coefficient tensor
   * `A_{rp} B_{qs}` into `ggo_grad`.
   *
   * This is the mix-spin / non-antisymmetrized update path.
   *
   * @param weight Scalar prefactor applied to every outer-product coefficient.
   * @param left Dense matrix `A`.
   * @param right Dense matrix `B`.
   * @param ggo_grad Output packed gradient buffer. If empty, it is initialized
   *     to the correct size and zero-filled before accumulation.
   */
  static void add_coulomb_outer_product(
      double weight,
      const ConstMatrixRef& left,
      const ConstMatrixRef& right,
      ScalarBuffer* ggo_grad);

  /**
   * @brief Adds the direct outer-product coefficient tensor
   * `A_{qp} B_{sr}` into `ggo_grad`.
   *
   * @param weight Scalar prefactor applied to every outer-product coefficient.
   * @param left Dense matrix `A`.
   * @param right Dense matrix `B`.
   * @param ggo_grad Output packed gradient buffer. If empty, it is initialized
   *     to the correct size and zero-filled before accumulation.
   */
  static void add_direct_outer_product(
      double weight,
      const ConstMatrixRef& left,
      const ConstMatrixRef& right,
      ScalarBuffer* ggo_grad);

  /**
   * @brief Adds the exact same-spin separable outer-product coefficient tensor
   * into `ggo_grad`.
   */
  static void add_same_spin_separable_outer_product(
      double weight,
      const ConstMatrixRef& left,
      const ConstMatrixRef& right,
      ScalarBuffer* ggo_grad);

  /**
   * @brief Adds the exact same-spin bridge outer-product coefficient tensor
   * into `ggo_grad`.
   */
  static void add_same_spin_bridge_outer_product(
      double weight,
      const ConstMatrixRef& left,
      const ConstMatrixRef& right,
      ScalarBuffer* ggo_grad);

  /**
   * @brief Adds the same-spin exchange outer-product update with the
   * antisymmetrized chain rule.
   *
   * For each ordered `(q, p, s, r)` contribution this performs
   * `+ A_{pr} B_{sq}` at `g_{qp,sr}` and `- A_{pr} B_{sq}` at `g_{qr,sp}`.
   *
   * @param weight Scalar prefactor applied to every outer-product coefficient.
   * @param left Dense matrix `A`.
   * @param right Dense matrix `B`.
   * @param ggo_grad Output packed gradient buffer. If empty, it is initialized
   *     to the correct size and zero-filled before accumulation.
   */
  static void add_same_spin_exchange_outer_product(
      double weight,
      const ConstMatrixRef& left,
      const ConstMatrixRef& right,
      ScalarBuffer* ggo_grad);

  /**
   * @brief Adds the same-spin Coulomb outer-product update with the
   * antisymmetrized chain rule.
   *
   * For each ordered `(q, p, s, r)` contribution this performs
   * `+ A_{rp} B_{qs}` at `g_{qp,sr}` and `- A_{rp} B_{qs}` at `g_{qr,sp}`.
   *
   * @param weight Scalar prefactor applied to every outer-product coefficient.
   * @param left Dense matrix `A`.
   * @param right Dense matrix `B`.
   * @param ggo_grad Output packed gradient buffer. If empty, it is initialized
   *     to the correct size and zero-filled before accumulation.
   */
  static void add_same_spin_coulomb_outer_product(
      double weight,
      const ConstMatrixRef& left,
      const ConstMatrixRef& right,
      ScalarBuffer* ggo_grad);
};

}  // namespace xmvb::pfaffian_vbscf
