#pragma once

#include <vector>

#include "vb/approx/approx_vbscf_cluster_quotient.hpp"

namespace xmvb::vb {

struct ApproxVbScfResonancePairPairTerm {
  int first_pair_index = 0;
  int second_pair_index = 0;
  double connected_energy = 0.0;
};

struct ApproxVbScfResonanceFunctionalModel {
  std::vector<double> one_pair_energy;
  std::vector<ApproxVbScfResonancePairPairTerm> pair_pair_terms;
  int one_pair_term_count = 0;
  int dense_pair_pair_term_count = 0;
  int compatible_pair_pair_term_count = 0;
};

struct ApproxVbScfResonanceFunctionalInput {
  int n_active_orbitals = 0;
  const ApproxVbScfClusterQuotientModel* cluster_quotient_model = nullptr;
};

/**
 * @brief Builds the compact pair-space resonance functional for aVBSCF.
 *
 * The one-pair values and connected pair-pair cumulants come from local
 * quotient clusters.  This object has no structure-space dependency: it is the
 * production counterpart to the exact off-diagonal resonance decomposition
 * used only for small-system labels.
 */
ApproxVbScfResonanceFunctionalModel
build_approx_vbscf_resonance_functional_model(
    const ApproxVbScfResonanceFunctionalInput& input);

}  // namespace xmvb::vb
