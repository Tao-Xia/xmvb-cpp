#include "vb/matrices/determinant_hamiltonian_resolver.hpp"

#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "vb/matrices/spin_pair_utils.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace xmvb::vb {

void build_minor_matrix(
    const Matrix& mat,
    int r1, int r2, int c1, int c2,
    Eigen::Ref<Matrix> minor_wksp) 
{
  const int dim = static_cast<int>(mat.rows());
  
  std::vector<int> row_map, col_map;
  row_map.reserve(dim - 2);
  col_map.reserve(dim - 2);

  for (int i = 0; i < dim; ++i) {
    if (i != r1 && i != r2) row_map.push_back(i);
    if (i != c1 && i != c2) col_map.push_back(i);
  }

  for (int c = 0; c < dim - 2; ++c) {
    for (int r = 0; r < dim - 2; ++r) {
      minor_wksp(r, c) = mat(row_map[r], col_map[c]);
    }
  }
}

double calc_minor_cofactor(
    const Matrix& det_ovlp_mat,
    int r1, int r2, int c1, int c2,
    Eigen::Ref<Matrix> minor_wksp) 
{
  build_minor_matrix(det_ovlp_mat, r1, r2, c1, c2, minor_wksp);

  const double sign = ((r1 + r2 + c1 + c2) & 1) ? -1.0 : 1.0;
  const Eigen::FullPivLU<Eigen::Ref<Matrix>> lu(minor_wksp);
  
  return sign * lu.determinant();
}

DeterminantHamiltonianResult calc_same_spin_hamiltonian(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const std::vector<double>& det_ovlp,
    const DeterminantOverlapResult& det_ovlp_result,
    const std::vector<double>& h1e_act,
    int n_orb_act,
    const std::vector<double>& eri_act) 
{
  // n_elec is not the number total electrons
  // it is the number of alpha/beta active electrons
  const std::size_t n_elec = occ_L.size();

  // overlap matrix between two determinant
  const ConstMatrixMap det_ovlp_mat(det_ovlp.data(),n_elec, n_elec);
  // active orbital one-electron integral matrix
  const ConstMatrixMap h1e_act_mat(h1e_act.data(), n_orb_act, n_orb_act);
  // 1st-cofactor between two determinant
  const Matrix cofactor_1st = calc_cofactor_1st(det_ovlp_result);
  
  DeterminantHamiltonianResult res;
  res.overlap_determinant = det_ovlp_result.overlap_determinant;
  res.nullity = det_ovlp_result.nullity;
  res.total_hamiltonian = 0.0;

  if (res.nullity >= 3) return res;

  // 1-electron 
  for (std::size_t cL = 0; cL < n_elec; ++cL) {
    const int orb_L = occ_L[cL];
    for (std::size_t rR = 0; rR < n_elec; ++rR) {
      const int orb_R = occ_R[rR];
      res.total_hamiltonian += h1e_act_mat(orb_R, orb_L) * cofactor_1st(rR, cL);
    }
  }

  res.one_electron_hamiltonian = res.total_hamiltonian;

  if (n_elec < 2) return res;

  Matrix minor_wksp(n_elec - 2, n_elec - 2);
  const double inv_S_det = (res.nullity == 0) ? (1.0 / det_ovlp_result.overlap_determinant) : 0.0;

  // 2-electron 贡献
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
            const int J_idx = TwoElectronIndexer::packed_pair_of_pairs_index(pR1_L1, pR2_L2);
            const int K_idx = TwoElectronIndexer::packed_pair_of_pairs_index(pR1_L2, pR2_L1);

            res.total_hamiltonian += (eri_act[J_idx] - eri_act[K_idx]) * cofactor_2nd;
          }
        }
      }
    }
  } else {
    // nullity > 0
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
            
            const double cofactor_2nd = calc_minor_cofactor(det_ovlp_mat, R1, R2, L1, L2, minor_wksp);

            const int pR2_L2 = TwoElectronIndexer::packed_pair_index(oR2, oL2);
            const int pR2_L1 = TwoElectronIndexer::packed_pair_index(oR2, oL1);
            const int J_idx = TwoElectronIndexer::packed_pair_of_pairs_index(pR1_L1, pR2_L2);
            const int K_idx = TwoElectronIndexer::packed_pair_of_pairs_index(pR1_L2, pR2_L1);

            res.total_hamiltonian += (eri_act[J_idx] - eri_act[K_idx]) * cofactor_2nd;
          }
        }
      }
    }
  }

  return res;
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
    const DeterminantOverlapResult& det_ovlp_result,
    const std::vector<double>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act) const {

  const int n_electrons = static_cast<int>(occ_L.size());

  return xmvb::vb::calc_same_spin_hamiltonian(
      occ_L,
      occ_R,
      det_ovlp_mat,
      det_ovlp_result,
      h1e_act,
      n_orbitals,
      eri_act);
}

}  // namespace xmvb::vb
