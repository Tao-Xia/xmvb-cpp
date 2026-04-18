#include "pfaffian_vbscf/matrices/pf_pair_kernels.hpp"

#include <stdexcept>

#include "pfaffian_vbscf/kernel/pf_forward_kernel.hpp"

namespace xmvb::pfaffian_vbscf {

PfPairKernelResult evaluate_pf_pair_kernel(
    const ConstMatrixRef& left_pairing_matrix,
    const ConstMatrixRef& right_pairing_matrix,
    const ConstMatrixRef& spatial_overlap_matrix,
    const ConstMatrixRef& one_electron_matrix,
    const ScalarBuffer& packed_active_two_electron_integrals,
    int n_alpha_electrons,
    int n_beta_electrons,
    int n_pairs) {
  if (n_alpha_electrons < 0 || n_beta_electrons < 0) {
    throw std::invalid_argument("spin electron counts must be non-negative");
  }
  if (n_pairs < 0) {
    throw std::invalid_argument("n_pairs must be non-negative");
  }
  if (n_alpha_electrons != n_beta_electrons || n_beta_electrons != n_pairs) {
    throw std::invalid_argument(
        "the current exact closed-shell Pf pair kernel requires "
        "n_alpha == n_beta == n_pairs");
  }
  if (left_pairing_matrix.rows() != left_pairing_matrix.cols() ||
      right_pairing_matrix.rows() != right_pairing_matrix.cols()) {
    throw std::invalid_argument("pairing matrices must be square");
  }
  if (left_pairing_matrix.rows() != right_pairing_matrix.rows()) {
    throw std::invalid_argument("left/right pairing matrix dimensions must match");
  }
  if (spatial_overlap_matrix.rows() != spatial_overlap_matrix.cols() ||
      one_electron_matrix.rows() != one_electron_matrix.cols()) {
    throw std::invalid_argument("spatial matrices must be square");
  }
  if (spatial_overlap_matrix.rows() != one_electron_matrix.rows()) {
    throw std::invalid_argument("spatial overlap and one-electron dimensions must match");
  }
  if (left_pairing_matrix.rows() != 2 * spatial_overlap_matrix.rows()) {
    throw std::invalid_argument(
        "pairing matrix dimension does not match the active spin space");
  }

  const Eigen::Index n_active_orbitals = spatial_overlap_matrix.rows();

  const Matrix left_ba =
      left_pairing_matrix.topRightCorner(n_active_orbitals, n_active_orbitals)
          .transpose()
          .eval();
  
  const Matrix right_ab =
      right_pairing_matrix.topRightCorner(n_active_orbitals, n_active_orbitals)
          .eval();

  const PfKernelCache cache =
      PfForwardKernel::build_closed_shell_exact_spatial_cache(
          left_ba,
          spatial_overlap_matrix,
          right_ab,
          n_pairs);

  PfPairKernelResult result;

  result.overlap = std::move(cache.overlap_value);
  result.one_electron_hamiltonian = std::move(
        cache.one_rdm.cwiseProduct(one_electron_matrix).sum()
  );

  result.total_hamiltonian = std::move(
      result.one_electron_hamiltonian +
      PfForwardKernel::evaluate_closed_shell_two_electron(
          cache,
          packed_active_two_electron_integrals)
  );
  
  return result;
}

}  // namespace xmvb::pfaffian_vbscf
