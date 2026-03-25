#pragma once

#include <vector>

#include "vb/matrices/determinant_hamiltonian_resolver.hpp"
#include "vb/matrices/determinant_overlap_resolver.hpp"
#include "vb/matrices/determinant_types.hpp"

namespace xmvb::vb {

struct SpinDeterminantPairEvaluation {
  DeterminantOverlapResult overlap_result;
  double one_electron_hamiltonian = 0.0;
  double total_hamiltonian = 0.0;
};

struct FullDeterminantPairEvaluation {
  SpinDeterminantPairEvaluation alpha;
  SpinDeterminantPairEvaluation beta;
  double overlap_determinant = 0.0;
  double one_electron_hamiltonian = 0.0;
  double total_hamiltonian = 0.0;
};

class FullDeterminantPairEvaluator {
public:
  FullDeterminantPairEvaluator();

  FullDeterminantPairEvaluator(
      DeterminantOverlapResolver determinant_overlap_resolver,
      DeterminantHamiltonianResolver determinant_hamiltonian_resolver);

  FullDeterminantPairEvaluation evaluate(
      const std::vector<int>& alpha_occ_L,
      const std::vector<int>& alpha_occ_R,
      const std::vector<int>& beta_occ_L,
      const std::vector<int>& beta_occ_R,
      const std::vector<double>& ovlp_act,
      const std::vector<double>& h1e_act,
      int n_orbitals,
      const std::vector<double>& eri_act) const;

private:
  DeterminantOverlapResolver determinant_overlap_resolver_;
  DeterminantHamiltonianResolver determinant_hamiltonian_resolver_;
};

}  // namespace xmvb::vb
