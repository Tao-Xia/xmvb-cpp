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

inline int packed_pair_index_fast(int orbital_index_a, int orbital_index_b) {
  const int high_index =
      (orbital_index_a > orbital_index_b) ? orbital_index_a : orbital_index_b;
  const int low_index =
      (orbital_index_a > orbital_index_b) ? orbital_index_b : orbital_index_a;
  return high_index * (high_index + 1) / 2 + low_index;
}

inline int packed_pair_of_pairs_index_fast(
    int packed_pair_index_a,
    int packed_pair_index_b) {
  const int high_index =
      (packed_pair_index_a > packed_pair_index_b)
          ? packed_pair_index_a
          : packed_pair_index_b;
  const int low_index =
      (packed_pair_index_a > packed_pair_index_b)
          ? packed_pair_index_b
          : packed_pair_index_a;
  return high_index * (high_index + 1) / 2 + low_index;
}

}  // namespace

template <typename PairKernelLookup>
DeterminantHamiltonianResult calc_same_spin_hamiltonian_impl(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const DeterminantOverlapResult& det_ovlp_result,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orb_act,
    const double* packed_kernel,
    PairKernelLookup&& lookup_pair_kernel) {
  // n_elec is not the number total electrons
  // it is the number of alpha/beta active electrons
  const int n_elec = static_cast<int>(occ_L.size());
  (void)n_orb_act;

  DeterminantHamiltonianResult res;
  res.overlap_determinant = det_ovlp_result.overlap_determinant;
  res.nullity = det_ovlp_result.nullity;
  res.total_hamiltonian = 0.0;

  if (res.nullity >= 3) return res;

  // active orbital one-electron integral matrix
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
    for (int left_column = 0; left_column < n_elec; ++left_column) {
      const int orb_L = occ_L[left_column];
      const double* cofactor_column = cofactor_1st.col(left_column).data();
      for (int right_row = 0; right_row < n_elec; ++right_row) {
        res.total_hamiltonian +=
            h1e_act(occ_R[right_row], orb_L) * cofactor_column[right_row];
      }
    }
  }

  res.one_electron_hamiltonian = res.total_hamiltonian;

  if (n_elec < 2) return res;

  // 2-electron
  if (packed_kernel != nullptr) {
    if (res.nullity == 0) {
      const double inv_S_det = 1.0 / det_ovlp_result.overlap_determinant;
      double two_electron_hamiltonian = 0.0;

      for (int left_first = 0; left_first < n_elec - 1; ++left_first) {
        const double* cofactor_left_first =
            cofactor_1st.col(left_first).data();
        const int orbital_left_first = occ_L[left_first];
        for (int right_first = 0; right_first < n_elec - 1; ++right_first) {
          const double c11 = cofactor_left_first[right_first];
          const int orbital_right_first = occ_R[right_first];
          const int pR1_L1 =
              packed_pair_index_fast(orbital_right_first, orbital_left_first);

          for (int left_second = left_first + 1;
               left_second < n_elec;
               ++left_second) {
            const double* cofactor_left_second =
                cofactor_1st.col(left_second).data();
            const int orbital_left_second = occ_L[left_second];
            const double c12 = cofactor_left_second[right_first];
            const int pR1_L2 =
                packed_pair_index_fast(orbital_right_first, orbital_left_second);

            double pair_sum = 0.0;
            for (int right_second = right_first + 1;
                 right_second < n_elec;
                 ++right_second) {
              const int orbital_right_second = occ_R[right_second];
              const double minor_2nd =
                  c11 * cofactor_left_second[right_second] -
                  c12 * cofactor_left_first[right_second];
              const int pR2_L2 =
                  packed_pair_index_fast(orbital_right_second, orbital_left_second);
              const int pR2_L1 =
                  packed_pair_index_fast(orbital_right_second, orbital_left_first);
              pair_sum +=
                  (packed_kernel[packed_pair_of_pairs_index_fast(pR1_L1, pR2_L2)] -
                   packed_kernel[packed_pair_of_pairs_index_fast(pR1_L2, pR2_L1)]) *
                  minor_2nd;
            }
            two_electron_hamiltonian += pair_sum;
          }
        }
      }
      res.total_hamiltonian += two_electron_hamiltonian * inv_S_det;
    } else {
      // Singular pairs still need the exact second-order cofactor formulas, but
      // the packed-GGO path can still avoid the generic kernel callback and use
      // unchecked packed-index arithmetic plus direct table loads.
      for (int left_first = 0; left_first < n_elec - 1; ++left_first) {
        const int orbital_left_first = occ_L[left_first];
        for (int right_first = 0; right_first < n_elec - 1; ++right_first) {
          const int orbital_right_first = occ_R[right_first];
          const int pR1_L1 =
              packed_pair_index_fast(orbital_right_first, orbital_left_first);

          for (int left_second = left_first + 1;
               left_second < n_elec;
               ++left_second) {
            const int orbital_left_second = occ_L[left_second];
            const int pR1_L2 =
                packed_pair_index_fast(orbital_right_first, orbital_left_second);

            for (int right_second = right_first + 1;
                 right_second < n_elec;
                 ++right_second) {
              const int orbital_right_second = occ_R[right_second];
              const double cofactor_2nd =
                  calc_second_order_cofactor(
                      det_ovlp_result,
                      right_first,
                      right_second,
                      left_first,
                      left_second);
              const int pR2_L2 =
                  packed_pair_index_fast(orbital_right_second, orbital_left_second);
              const int pR2_L1 =
                  packed_pair_index_fast(orbital_right_second, orbital_left_first);
              res.total_hamiltonian +=
                  (packed_kernel[packed_pair_of_pairs_index_fast(pR1_L1, pR2_L2)] -
                   packed_kernel[packed_pair_of_pairs_index_fast(pR1_L2, pR2_L1)]) *
                  cofactor_2nd;
            }
          }
        }
      }
    }
    return res;
  }

  if (res.nullity == 0) {
    const double inv_S_det = 1.0 / det_ovlp_result.overlap_determinant;
    double two_electron_hamiltonian = 0.0;

    for (int left_first = 0; left_first < n_elec - 1; ++left_first) {
      const double* cofactor_left_first =
          cofactor_1st.col(left_first).data();
      const int orbital_left_first = occ_L[left_first];
      for (int right_first = 0; right_first < n_elec - 1; ++right_first) {
        const double c11 = cofactor_left_first[right_first];
        const int orbital_right_first = occ_R[right_first];
        const int pR1_L1 =
            packed_pair_index_fast(orbital_right_first, orbital_left_first);

        for (int left_second = left_first + 1;
             left_second < n_elec;
             ++left_second) {
          const double* cofactor_left_second =
              cofactor_1st.col(left_second).data();
          const int orbital_left_second = occ_L[left_second];
          const double c12 = cofactor_left_second[right_first];
          const int pR1_L2 =
              packed_pair_index_fast(orbital_right_first, orbital_left_second);

          double pair_sum = 0.0;
          for (int right_second = right_first + 1;
               right_second < n_elec;
               ++right_second) {
            const int orbital_right_second = occ_R[right_second];
            const double minor_2nd =
                c11 * cofactor_left_second[right_second] -
                c12 * cofactor_left_first[right_second];
            const int pR2_L2 =
                packed_pair_index_fast(orbital_right_second, orbital_left_second);
            const int pR2_L1 =
                packed_pair_index_fast(orbital_right_second, orbital_left_first);
            pair_sum +=
                (lookup_pair_kernel(pR1_L1, pR2_L2) -
                 lookup_pair_kernel(pR1_L2, pR2_L1)) *
                minor_2nd;
          }
          two_electron_hamiltonian += pair_sum;
        }
      }
    }
    res.total_hamiltonian += two_electron_hamiltonian * inv_S_det;
  } else {
    for (int left_first = 0; left_first < n_elec - 1; ++left_first) {
      const int orbital_left_first = occ_L[left_first];
      for (int right_first = 0; right_first < n_elec - 1; ++right_first) {
        const int orbital_right_first = occ_R[right_first];
        const int pR1_L1 =
            packed_pair_index_fast(orbital_right_first, orbital_left_first);

        for (int left_second = left_first + 1;
             left_second < n_elec;
             ++left_second) {
          const int orbital_left_second = occ_L[left_second];
          const int pR1_L2 =
              packed_pair_index_fast(orbital_right_first, orbital_left_second);

          for (int right_second = right_first + 1;
               right_second < n_elec;
               ++right_second) {
            const int orbital_right_second = occ_R[right_second];
            const double cofactor_2nd =
                calc_second_order_cofactor(
                    det_ovlp_result,
                    right_first,
                    right_second,
                    left_first,
                    left_second);
            const int pR2_L2 =
                packed_pair_index_fast(orbital_right_second, orbital_left_second);
            const int pR2_L1 =
                packed_pair_index_fast(orbital_right_second, orbital_left_first);
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
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orb_act,
    const std::vector<double>& eri_act) {
  return calc_same_spin_hamiltonian_impl(
      occ_L,
      occ_R,
      det_ovlp_result,
      h1e_act,
      n_orb_act,
      eri_act.data(),
      [&eri_act](int row_packed_pair_index, int column_packed_pair_index) {
        const int packed_pair_of_pairs_index =
            TwoElectronIndexer::packed_pair_of_pairs_index(
                row_packed_pair_index,
                column_packed_pair_index);
        return eri_act[packed_pair_of_pairs_index];
      });
}

DeterminantHamiltonianResult calc_same_spin_hamiltonian(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const DeterminantOverlapResult& det_ovlp_result,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orb_act,
    const ActiveSpaceTwoElectronView& two_electron_view) {
  return calc_same_spin_hamiltonian_impl(
      occ_L,
      occ_R,
      det_ovlp_result,
      h1e_act,
      n_orb_act,
      two_electron_view.packed_active_two_electron_integrals != nullptr &&
              !two_electron_view.packed_active_two_electron_integrals->empty()
          ? two_electron_view.packed_active_two_electron_integrals->data()
          : nullptr,
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
    : overlap_resolver_() {
  (void)algorithm;
}

DeterminantHamiltonianResolver::DeterminantHamiltonianResolver(
    DeterminantOverlapResolver overlap_resolver,
    VBSCFAlgorithm algorithm)
    : overlap_resolver_(std::move(overlap_resolver)) {
  (void)algorithm;
}

DeterminantHamiltonianResult DeterminantHamiltonianResolver::resolve(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::Ref<const Eigen::MatrixXd>& det_ovlp_mat,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act) const {
  if (occ_L.size() != occ_R.size()) {
    throw std::invalid_argument("left and right determinants must have the same electron count");
  }
  if (occ_L.empty()) {
    throw std::invalid_argument("determinants must not be empty");
  }

  const auto det_ovlp_result = overlap_resolver_.resolve_matrix(det_ovlp_mat);
  return resolve(
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
    const Eigen::Ref<const Eigen::MatrixXd>& det_ovlp_mat,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result) const {
  if (occ_L.size() != occ_R.size()) {
    throw std::invalid_argument("left and right determinants must have the same electron count");
  }
  if (occ_L.empty()) {
    throw std::invalid_argument("determinants must not be empty");
  }

  const auto det_ovlp_result = overlap_resolver_.resolve_matrix(det_ovlp_mat);
  return resolve(
      occ_L,
      occ_R,
      det_ovlp_result,
      h1e_act,
      n_orbitals,
      active_space_two_electron_result);
}

DeterminantHamiltonianResult DeterminantHamiltonianResolver::resolve(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const DeterminantOverlapResult& det_ovlp_result,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act) const {
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
    const DeterminantOverlapResult& det_ovlp_result,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
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
