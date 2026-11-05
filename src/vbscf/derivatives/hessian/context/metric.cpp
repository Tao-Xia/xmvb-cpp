#include "vbscf/derivatives/hessian/context/accepted_point.hpp"

#include "vbscf/structures/orthogonal_ci/metric.hpp"

namespace xmvb::vb {

const StructureMetricInverse*
AcceptedPointContext::metric_inverse(const StructureAction& action) const {
    if (!complete_active_space || !action.storage().orthogonal_direct_ci)
        return nullptr;
    if (!structure_metric_planned) {
        const auto& alpha = same_spin_pair_cache.alpha_reuse_table.unique_determinants;
        const auto& beta = same_spin_pair_cache.beta_reuse_table.unique_determinants;
        structure_metric_inverse = StructureMetricInverse::try_build(action, alpha, beta);
        structure_metric_planned = true;
    }
    return structure_metric_inverse.get();
}

} // namespace xmvb::vb
