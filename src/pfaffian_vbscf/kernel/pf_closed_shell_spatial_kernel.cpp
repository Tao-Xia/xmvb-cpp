#include "pfaffian_vbscf/kernel/pf_closed_shell_spatial_kernel.hpp"
#include "pfaffian_vbscf/tensor/pf_tensor_contractor.hpp"

namespace xmvb::pfaffian_vbscf {

/**
 * @brief Evaluates the closed-shell exact spatial two-electron matrix element.
 *
 * Every cache matrix used here is an `M x M` spatial object:
 * - `closed_shell_left_ba_block`, `closed_shell_right_ab_block`: Pfaffian
 *   `BA` / `AB` transition blocks
 * - `closed_shell_pair_core`: collapsed pair-core matrix `C_code = L S R`
 * - `closed_shell_d`: bridge matrix reused by the opposite-spin terms
 *
 * The forward value is assembled from the prepacked linear-combination kernels
 * built in `pf_forward_kernel.cpp`, so this function only performs the final
 * packed-integral contractions.
 */
double evaluate_closed_shell_two_electron_spatial_exact(
    const PfKernelCache& cache,
    const ScalarBuffer& ggo) {
  if (cache.trace_order <= 0) {
    return 0.0;
  }

  const int n_active_orbitals = cache.n_active_orbitals;
  const Matrix& left_ba_block = cache.closed_shell_left_ba_block;
  const Matrix& right_ab_block = cache.closed_shell_right_ab_block;
  const Matrix& pair_core = cache.closed_shell_pair_core;
  const Matrix& bridge_matrix = cache.closed_shell_d;

  if (cache.closed_shell_coeff_n2.empty()) {
    return
        PfTensorContractor::contract_exchange(
            ggo,
            n_active_orbitals,
            right_ab_block,
            cache.closed_shell_pair_term_matrix) +
        PfTensorContractor::contract_coulomb(
            ggo,
            n_active_orbitals,
            right_ab_block,
            cache.closed_shell_pair_term_matrix);
  }

  double value =
      PfTensorContractor::contract_closed_shell_full_linear_combo(
          ggo,
          n_active_orbitals,
          right_ab_block,
          cache.closed_shell_pair_term_matrix,
          cache.closed_shell_d_poly_h_n2,
          left_ba_block,
          bridge_matrix,
          cache.closed_shell_opposite_bridge_h_split_matrix,
          cache.closed_shell_c_poly_h_n2,
          pair_core,
          cache.closed_shell_frechet_h_c_h_n2);

  value += PfTensorContractor::contract_closed_shell_same_spin_asym_linear_combo(
      ggo,
      n_active_orbitals,
      cache.closed_shell_c_poly_h_n2,
      pair_core,
      cache.closed_shell_frechet_h_c_h_n2);

  return value;
}

}  // namespace xmvb::pfaffian_vbscf
