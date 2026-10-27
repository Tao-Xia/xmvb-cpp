#pragma once

#include "vbscf/structures/assembly/operator_first.hpp"
#include "mcscf/ci/raw_action.hpp"
#include "mcscf/ci/sigma.hpp"

namespace xmvb::vb {
struct OperatorFirstAction::Data {
    const std::shared_ptr<const int> point = std::make_shared<const int>(0);
    StructureAction::OrthogonalView view;
    Eigen::Index n_structures;
    mcscf::SimilarityHamiltonian hamiltonian;
    mcscf::RawCiPlan source;
    Data(StructureAction::OrthogonalView v,
         Eigen::Index count,
         const mcscf::ActiveIntegrals& integrals,
         const Eigen::Ref<const Eigen::MatrixXd>& orbital_map)
        : view(v), n_structures(count), hamiltonian(integrals, orbital_map),
          source(v.sigma.alpha_connections(), v.sigma.beta_connections_view(), v.products) {}
};

} // namespace xmvb::vb
