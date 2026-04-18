#include "vb/matrices/determinant_hamiltonian_resolver.hpp"

#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"

namespace xmvb::vb {

namespace {

double product_of_leading_singular_values(
    const DeterminantOverlapResult& det_ovlp_result,
    int count) {
  double product = 1.0;

  for (int singular_index = 0; singular_index < count; ++singular_index) {
    product *= det_ovlp_result.singular_values(singular_index);
  }

  return product;
}

void require_svd_overlap_result(
    const DeterminantOverlapResult& det_ovlp_result) {
  if (det_ovlp_result.singular_values.size() != det_ovlp_result.n_electrons ||
      det_ovlp_result.matrix_U.rows() != det_ovlp_result.n_electrons ||
      det_ovlp_result.matrix_U.cols() != det_ovlp_result.n_electrons ||
      det_ovlp_result.matrix_V.rows() != det_ovlp_result.n_electrons ||
      det_ovlp_result.matrix_V.cols() != det_ovlp_result.n_electrons) {
    throw std::invalid_argument("det_ovlp_result SVD dimensions are inconsistent");
  }
}

double calc_second_order_rank_deficient_cofactor(
    const DeterminantOverlapResult& det_ovlp_result,
    int right_first,
    int right_second,
    int left_first,
    int left_second) {
  require_svd_overlap_result(det_ovlp_result);
  const int n_electrons = det_ovlp_result.n_electrons;
  if (det_ovlp_result.nullity == 1) {
    // For rank-(n-1) overlaps, every degree-2 deleted minor is an exact sum of
    // wedge products between the single null mode and each non-null singular
    // direction. This keeps the singular path exact while avoiding an LU on a
    // freshly materialized (n-2) minor for every `(r1, r2, c1, c2)` tuple.
    const int null_index = n_electrons - 1;
    const double prefactor =
        det_ovlp_result.parity *
        product_of_leading_singular_values(det_ovlp_result, n_electrons - 1);

    const double right_null_first = det_ovlp_result.matrix_U(right_first, null_index);
    const double right_null_second = det_ovlp_result.matrix_U(right_second, null_index);
    const double left_null_first = det_ovlp_result.matrix_V(left_first, null_index);
    const double left_null_second = det_ovlp_result.matrix_V(left_second, null_index);

    double second_cofactor = 0.0;
    for (int singular_index = 0; singular_index < n_electrons - 1; ++singular_index) {
      const double right_wedge =
          det_ovlp_result.matrix_U(right_first, singular_index) * right_null_second -
          det_ovlp_result.matrix_U(right_second, singular_index) * right_null_first;
      const double left_wedge =
          det_ovlp_result.matrix_V(left_first, singular_index) * left_null_second -
          det_ovlp_result.matrix_V(left_second, singular_index) * left_null_first;
      second_cofactor +=
          (right_wedge * left_wedge) /
          det_ovlp_result.singular_values(singular_index);
    }
    return prefactor * second_cofactor;
  }

  if (det_ovlp_result.nullity == 2) {
    // For rank-(n-2) overlaps the degree-2 deleted-minor family collapses to a
    // rank-1 outer product between the two-dimensional left/right null wedges.
    const int first_null_index = n_electrons - 2;
    const int second_null_index = n_electrons - 1;
    const double prefactor =
        det_ovlp_result.parity *
        product_of_leading_singular_values(det_ovlp_result, n_electrons - 2);

    const double right_wedge =
        det_ovlp_result.matrix_U(right_first, first_null_index) *
            det_ovlp_result.matrix_U(right_second, second_null_index) -
        det_ovlp_result.matrix_U(right_second, first_null_index) *
            det_ovlp_result.matrix_U(right_first, second_null_index);

    const double left_wedge =
        det_ovlp_result.matrix_V(left_first, first_null_index) *
            det_ovlp_result.matrix_V(left_second, second_null_index) -
        det_ovlp_result.matrix_V(left_second, first_null_index) *
            det_ovlp_result.matrix_V(left_first, second_null_index);
    return prefactor * right_wedge * left_wedge;
  }

  return 0.0;
}

}  // namespace

template <typename PairKernelLookup>
DeterminantHamiltonianResult calc_same_spin_hamiltonian_impl(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const DeterminantOverlapResult& det_ovlp_result,
    const std::vector<double>& h1e_act,
    int n_orb_act,
    PairKernelLookup&& lookup_pair_kernel) {
  // n_elec is not the number total electrons
  // it is the number of alpha/beta active electrons
  const std::size_t n_elec = occ_L.size();

  DeterminantHamiltonianResult res;
  res.overlap_determinant = det_ovlp_result.overlap_determinant;
  res.nullity = det_ovlp_result.nullity;
  res.total_hamiltonian = 0.0;

  if (res.nullity >= 3) return res;

  // active orbital one-electron integral matrix
  const Eigen::Map<const Eigen::MatrixXd> h1e_act_mat(h1e_act.data(), n_orb_act, n_orb_act);
  Eigen::MatrixXd cofactor_1st;
  if (res.nullity <= 1) {
    // Nullity-1 pairs still carry exact one-electron and opposite-spin
    // contributions through the deleted-minor matrix returned by
    // `calc_cofactor_1st(...)`. Nullity-2 pairs have no degree-1 minors, so we
    // skip constructing an all-zero cofactor matrix in that case.
    cofactor_1st = calc_cofactor_1st(det_ovlp_result);
  }

  // 1-electron 
  if (res.nullity <= 1) {
    for (std::size_t cL = 0; cL < n_elec; ++cL) {
      const int orb_L = occ_L[cL];
      for (std::size_t rR = 0; rR < n_elec; ++rR) {
        const int orb_R = occ_R[rR];
        res.total_hamiltonian += h1e_act_mat(orb_R, orb_L) * cofactor_1st(rR, cL);
      }
    }
  }

  res.one_electron_hamiltonian = res.total_hamiltonian;

  if (n_elec < 2) return res;

  const double inv_S_det = (res.nullity == 0) ? (1.0 / det_ovlp_result.overlap_determinant) : 0.0;

  // 2-electron
  if (res.nullity == 0) {
    // fast path 
    for (std::size_t L1 = 0; L1 < n_elec - 1; ++L1) {
      const int oL1 = occ_L[L1];
      for (std::size_t R1 = 0; R1 < n_elec - 1; ++R1) {
        const int oR1 = occ_R[R1];
        const double c11 = cofactor_1st(R1, L1);
        const int pR1_L1 = TwoElectronIndexer::packed_pair_index(oR1, oL1);

        for (std::size_t L2 = L1 + 1; L2 < n_elec; ++L2) {
          const int oL2 = occ_L[L2];
          const double c12 = cofactor_1st(R1, L2);
          const int pR1_L2 = TwoElectronIndexer::packed_pair_index(oR1, oL2);

          for (std::size_t R2 = R1 + 1; R2 < n_elec; ++R2) {
            const int oR2 = occ_R[R2];
            const double c22 = cofactor_1st(R2, L2);
            const double c21 = cofactor_1st(R2, L1);

            const double cofactor_2nd = (c11 * c22 - c12 * c21) * inv_S_det;

            const int pR2_L2 = TwoElectronIndexer::packed_pair_index(oR2, oL2);
            const int pR2_L1 = TwoElectronIndexer::packed_pair_index(oR2, oL1);
            res.total_hamiltonian +=
                (lookup_pair_kernel(pR1_L1, pR2_L2) -
                 lookup_pair_kernel(pR1_L2, pR2_L1)) *
                cofactor_2nd;
          }
        }
      }
    }
  } else {
    // For singular determinant pairs the exact deleted-minor algebra can still
    // be evaluated from one SVD:
    // - nullity 1: sum over wedge products between the null mode and each
    //   regular singular direction
    // - nullity 2: one outer product between the left/right null wedges
    //
    // This replaces the previous O(n_e^4) stream of fresh `(n_e-2)`-minor LU
    // factorizations with O(n_e^5)` work for nullity 1 and O(n_e^4)` work for
    // nullity 2 while preserving exact matrix elements.
    for (std::size_t L1 = 0; L1 < n_elec - 1; ++L1) {
      const int oL1 = occ_L[L1];
      for (std::size_t R1 = 0; R1 < n_elec - 1; ++R1) {
        const int oR1 = occ_R[R1];
        const int pR1_L1 = TwoElectronIndexer::packed_pair_index(oR1, oL1);

        for (std::size_t L2 = L1 + 1; L2 < n_elec; ++L2) {
          const int oL2 = occ_L[L2];
          const int pR1_L2 = TwoElectronIndexer::packed_pair_index(oR1, oL2);

          for (std::size_t R2 = R1 + 1; R2 < n_elec; ++R2) {
            const int oR2 = occ_R[R2];
            const double cofactor_2nd =
                calc_second_order_rank_deficient_cofactor(
                    det_ovlp_result,
                    static_cast<int>(R1),
                    static_cast<int>(R2),
                    static_cast<int>(L1),
                    static_cast<int>(L2));

            const int pR2_L2 = TwoElectronIndexer::packed_pair_index(oR2, oL2);
            const int pR2_L1 = TwoElectronIndexer::packed_pair_index(oR2, oL1);
            res.total_hamiltonian +=
                (lookup_pair_kernel(pR1_L1, pR2_L2) -
                 lookup_pair_kernel(pR1_L2, pR2_L1)) *
                cofactor_2nd;
          }
        }
      }
    }
  }

  return res;
}

DeterminantHamiltonianResult calc_same_spin_hamiltonian(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const DeterminantOverlapResult& det_ovlp_result,
    const std::vector<double>& h1e_act,
    int n_orb_act,
    const std::vector<double>& eri_act) {
  return calc_same_spin_hamiltonian_impl(
      occ_L,
      occ_R,
      det_ovlp_result,
      h1e_act,
      n_orb_act,
      [&eri_act](int row_packed_pair_index, int column_packed_pair_index) {
        const int packed_pair_of_pairs_index =
            TwoElectronIndexer::packed_pair_of_pairs_index(
                row_packed_pair_index,
                column_packed_pair_index);
        return eri_act[xmvb::to_size(packed_pair_of_pairs_index)];
      });
}

DeterminantHamiltonianResult calc_same_spin_hamiltonian(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const DeterminantOverlapResult& det_ovlp_result,
    const std::vector<double>& h1e_act,
    int n_orb_act,
    const ActiveSpaceTwoElectronView& two_electron_view) {
  return calc_same_spin_hamiltonian_impl(
      occ_L,
      occ_R,
      det_ovlp_result,
      h1e_act,
      n_orb_act,
      [&two_electron_view, n_orb_act](
          int row_packed_pair_index,
          int column_packed_pair_index) {
        return lookup_active_space_two_electron_kernel_value(
            two_electron_view,
            row_packed_pair_index,
            column_packed_pair_index,
            n_orb_act);
      });
}


DeterminantHamiltonianResolver::DeterminantHamiltonianResolver(
    VBSCFAlgorithm algorithm)
    : overlap_resolver_(),
      algorithm_(algorithm) {}

DeterminantHamiltonianResolver::DeterminantHamiltonianResolver(
    DeterminantOverlapResolver overlap_resolver,
    VBSCFAlgorithm algorithm)
    : overlap_resolver_(std::move(overlap_resolver)),
      algorithm_(algorithm) {}

DeterminantHamiltonianResult DeterminantHamiltonianResolver::resolve(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& det_ovlp_mat,
    const std::vector<double>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act) const {
  if (occ_L.size() != occ_R.size()) {
    throw std::invalid_argument("left and right determinants must have the same electron count");
  }
  if (occ_L.empty()) {
    throw std::invalid_argument("determinants must not be empty");
  }

  const auto det_ovlp_result = overlap_resolver_.resolve(
      det_ovlp_mat,
      static_cast<int>(occ_L.size()));
  return resolve(
      occ_L,
      occ_R,
      det_ovlp_mat,
      det_ovlp_result,
      h1e_act,
      n_orbitals,
      eri_act);
}

DeterminantHamiltonianResult DeterminantHamiltonianResolver::resolve(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& det_ovlp_mat,
    const std::vector<double>& h1e_act,
    int n_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result) const {
  if (occ_L.size() != occ_R.size()) {
    throw std::invalid_argument("left and right determinants must have the same electron count");
  }
  if (occ_L.empty()) {
    throw std::invalid_argument("determinants must not be empty");
  }

  const auto det_ovlp_result = overlap_resolver_.resolve(
      det_ovlp_mat,
      static_cast<int>(occ_L.size()));
  return resolve(
      occ_L,
      occ_R,
      det_ovlp_mat,
      det_ovlp_result,
      h1e_act,
      n_orbitals,
      active_space_two_electron_result);
}

DeterminantHamiltonianResult DeterminantHamiltonianResolver::resolve(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& det_ovlp_mat,
    const DeterminantOverlapResult& det_ovlp_result,
    const std::vector<double>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act) const {

  const int n_electrons = static_cast<int>(occ_L.size());

  return xmvb::vb::calc_same_spin_hamiltonian(
      occ_L,
      occ_R,
      det_ovlp_result,
      h1e_act,
      n_orbitals,
      eri_act);
}

DeterminantHamiltonianResult DeterminantHamiltonianResolver::resolve(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& det_ovlp_mat,
    const DeterminantOverlapResult& det_ovlp_result,
    const std::vector<double>& h1e_act,
    int n_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result) const {
  return xmvb::vb::calc_same_spin_hamiltonian(
      occ_L,
      occ_R,
      det_ovlp_result,
      h1e_act,
      n_orbitals,
      make_active_space_two_electron_view(active_space_two_electron_result));
}

}  // namespace xmvb::vb
