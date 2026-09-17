#include "vbscf/structures/orthogonal_ci/integrals.hpp"

#include <stdexcept>

#include <Eigen/Cholesky>

#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"

namespace xmvb::vb {

OrthogonalActiveIntegrals orthogonalize_active_integrals(
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& one_electron,
    const ActiveSpaceTwoElectronResult& two_electron,
    int n_active_orbitals) {
  if (n_active_orbitals <= 0 ||
      active_overlap.size() !=
          static_cast<std::size_t>(n_active_orbitals) * n_active_orbitals ||
      one_electron.rows() != n_active_orbitals ||
      one_electron.cols() != n_active_orbitals ||
      !one_electron.allFinite()) {
    throw std::invalid_argument("invalid active-space orthogonalization input");
  }
  const Eigen::Map<const Eigen::MatrixXd> overlap(
      active_overlap.data(),
      n_active_orbitals,
      n_active_orbitals);
  Eigen::LLT<Eigen::MatrixXd> cholesky(overlap);
  if (cholesky.info() != Eigen::Success) {
    throw std::runtime_error(
        "active orbital overlap is not positive definite");
  }

  OrthogonalActiveIntegrals result;
  result.orbital_transform = cholesky.matrixU();
  const Eigen::MatrixXd inverse_transform =
      result.orbital_transform
          .template triangularView<Eigen::Upper>()
          .solve(Eigen::MatrixXd::Identity(
              n_active_orbitals,
              n_active_orbitals));
  result.one_electron =
      inverse_transform.transpose() * one_electron * inverse_transform;

  const int n_pairs = packed_active_pair_count(n_active_orbitals);
  Eigen::MatrixXd pair_transform = Eigen::MatrixXd::Zero(n_pairs, n_pairs);
  for (int new_first = 0; new_first < n_active_orbitals; ++new_first) {
    for (int new_second = 0; new_second <= new_first; ++new_second) {
      const int new_pair = TwoElectronIndexer::packed_pair_index(
          new_first, new_second);
      for (int old_first = 0; old_first < n_active_orbitals; ++old_first) {
        for (int old_second = 0; old_second <= old_first; ++old_second) {
          const int old_pair = TwoElectronIndexer::packed_pair_index(
              old_first, old_second);
          double coefficient =
              inverse_transform(old_first, new_first) *
              inverse_transform(old_second, new_second);
          if (old_first != old_second) {
            coefficient +=
                inverse_transform(old_second, new_first) *
                inverse_transform(old_first, new_second);
          }
          pair_transform(old_pair, new_pair) = coefficient;
        }
      }
    }
  }

  Eigen::MatrixXd original_kernel(n_pairs, n_pairs);
  const ActiveSpaceTwoElectronView view =
      make_active_space_two_electron_view(two_electron);
  for (int column = 0; column < n_pairs; ++column) {
    for (int row = 0; row < n_pairs; ++row) {
      original_kernel(row, column) =
          lookup_active_space_two_electron_kernel_value(
              view, row, column, n_active_orbitals);
    }
  }
  result.pair_kernel =
      pair_transform.transpose() * original_kernel * pair_transform;

  result.two_electron.representation =
      ActiveSpaceTwoElectronRepresentation::PackedExact;
  result.two_electron.packed_active_two_electron_integrals.resize(
      packed_active_two_electron_integral_count(n_active_orbitals));
  for (int column = 0; column < n_pairs; ++column) {
    for (int row = 0; row <= column; ++row) {
      result.two_electron.packed_active_two_electron_integrals[
          TwoElectronIndexer::packed_pair_of_pairs_index(row, column)] =
          result.pair_kernel(row, column);
    }
  }
  return result;
}

}  // namespace xmvb::vb
