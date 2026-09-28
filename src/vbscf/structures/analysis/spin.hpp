#pragma once

#include "vbscf/structures/expansion/types.hpp"

namespace xmvb::vb {

/**
 * @brief Exact highest-weight spin audit of a VB structure space.
 *
 * Every determinant in the expansion has the same spin projection
 * @f$M_S=(N_\alpha-N_\beta)/2@f$.  Applying @f$\hat S_+@f$ to every
 * structure tests whether it is a highest-weight eigenfunction with
 * @f$S=M_S@f$.  If all residuals vanish, every linear combination and hence
 * every structure-space eigenstate has the reported @f$S(S+1)@f$ exactly.
 */
struct StructureSpinAnalysis {
  double spin = 0.0;
  double spin_squared = 0.0;
  double maximum_raising_residual = 0.0;
  int worst_structure_index = -1;

  bool spin_adapted(double tolerance = 1.0e-12) const noexcept {
    return maximum_raising_residual <= tolerance;
  }
};

StructureSpinAnalysis analyze_structure_spin(
    const FullDeterminantStructureData& structures,
    int spin_multiplicity);

}  // namespace xmvb::vb
