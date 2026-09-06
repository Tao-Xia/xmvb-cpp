#include "vb/approx/approx_vbscf_resonance_functional.hpp"

#include <utility>
#include <vector>

#include "vb/approx/approx_vbscf_pair_cluster.hpp"

namespace xmvb::vb {

ApproxVbScfResonanceFunctionalModel
build_approx_vbscf_resonance_functional_model(
    const ApproxVbScfResonanceFunctionalInput& input) {
  const int n_active_pairs =
      approx_vbscf_active_pair_count(input.n_active_orbitals);

  ApproxVbScfResonanceFunctionalModel model;
  model.one_pair_energy =
      input.cluster_quotient_model->one_pair_energy;
  model.one_pair_term_count = n_active_pairs;
  model.dense_pair_pair_term_count =
      n_active_pairs * (n_active_pairs - 1) / 2;
  model.pair_pair_terms.reserve(model.dense_pair_pair_term_count);

  for (int second_pair_index = 0;
       second_pair_index < n_active_pairs;
       ++second_pair_index) {
    const auto second_pair =
        unpack_approx_vbscf_active_pair(second_pair_index);
    for (int first_pair_index = 0;
         first_pair_index < second_pair_index;
         ++first_pair_index) {
      const auto first_pair =
          unpack_approx_vbscf_active_pair(first_pair_index);
      const std::vector<std::pair<int, int>> cluster = {
          first_pair,
          second_pair};
      if (!approx_vbscf_pair_cluster_is_compatible(cluster)) {
        continue;
      }

      ApproxVbScfResonancePairPairTerm term;
      term.first_pair_index = first_pair_index;
      term.second_pair_index = second_pair_index;
      term.connected_energy =
          input.cluster_quotient_model->pair_pair_energy(
              first_pair_index,
              second_pair_index);
      model.pair_pair_terms.push_back(term);
    }
  }

  model.compatible_pair_pair_term_count =
      static_cast<int>(model.pair_pair_terms.size());
  return model;
}

}  // namespace xmvb::vb
