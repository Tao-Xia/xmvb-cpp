#include "vbscf/derivatives/hessian/responses/active_space/ri_factor_adjoint.hpp"

#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/opposite_spin/backward.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"

namespace xmvb::vb {

Eigen::MatrixXd apply_regular_ri_pair_space_adjoint(
    const SameSpinPairCacheContext& cache,
    const SelectedStateDeterminantMatrices& states,
    const std::vector<double>& energies,
    int n_active,
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e,
    const Eigen::Ref<const Eigen::MatrixXd>& factors) {
  const int n_pairs = packed_active_pair_count(n_active);
  if (!cache.has_pair_providers() ||
      states.states.size() != energies.size() ||
      factors.rows() <= 0 || factors.cols() != n_pairs) {
    throw std::invalid_argument("RI factor adjoint dimensions are inconsistent");
  }

  ActiveSpaceTwoElectronResult two_electron;
  two_electron.representation =
      ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity;
  two_electron.n_auxiliary_functions = static_cast<int>(factors.rows());
  two_electron.ri_active_pair_factors = factors;

  const SameSpinMatrixBackwardContribution same_spin =
      build_same_spin_matrix_backward_contribution(
          cache,
          states,
          energies,
          n_active,
          active_overlap,
          h1e,
          two_electron);
  const OppositeSpinBackwardContribution opposite_spin =
      build_opposite_spin_backward_contribution(
          cache,
          states,
          n_active,
          active_overlap,
          h1e,
          two_electron);
  if (same_spin.packed_active_two_electron_gradient.size() !=
          opposite_spin.packed_active_two_electron_gradient.size()) {
    throw std::logic_error(
        "same- and opposite-spin RI adjoints have inconsistent dimensions");
  }

  Eigen::MatrixXd pair_adjoint(n_pairs, n_pairs);
  for (int row = 0; row < n_pairs; ++row) {
    for (int column = 0; column < n_pairs; ++column) {
      const int packed = TwoElectronIndexer::packed_pair_of_pairs_index(
          row, column);
      const double value =
          same_spin.packed_active_two_electron_gradient[packed] +
          opposite_spin.packed_active_two_electron_gradient[packed];
      pair_adjoint(row, column) = row == column ? 2.0 * value : value;
    }
  }
  return factors * pair_adjoint;
}

}  // namespace xmvb::vb
