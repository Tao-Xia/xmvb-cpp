#include "vbscf/optimization/coupled/affine.hpp"

#include <Eigen/Cholesky>
#include <cmath>
#include <chrono>
#include <utility>

#include "core/timing.hpp"
#include "mcscf/optimization/trace.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"

namespace xmvb::vb {

AffineCoupling::AffineCoupling(const AcceptedPointContext& context,
                               const ExactHvpOperator& hessian,
                               const StructureTangentOperator& structure)
    : hessian_(hessian), structure_(structure) {
    // Reuse the accepted images shared with StructureTangentOperator.
    const auto& images = context.selected_state_structure_images.value();
    const auto& c = context.selected_state_eigenvectors;
    residual_ = images.hamiltonian;
    scales_.resize(c.cols());
    for (Eigen::Index i = 0; i < c.cols(); ++i) {
        residual_.col(i) -= context.selected_state_energies[i] * images.overlap.col(i);
        scales_[i] = std::sqrt(2.0 * context.normalized_state_weights[i]);
    }
    const Eigen::MatrixXd gram = c.transpose() * images.overlap;
    metric_inverse_ = gram.llt().solve(Eigen::MatrixXd::Identity(c.cols(), c.cols()));
}

AffineCouplingAction AffineCoupling::apply(const Eigen::VectorXd& p,
                                           const Eigen::VectorXd& q) const {
    const bool direct_ci = hessian_.structure_action().supports_integral_direction();
    const Eigen::MatrixXd raw = structure_.coefficient_response_coordinates(q);
    const Eigen::MatrixXd correction = -metric_inverse_ * residual_.transpose() * raw;
    StructureResponseRequest request{raw};
    AffineCouplingAction result;
    result.orbital = hessian_.apply_orbital_coupling(p, direct_ci ? &request : nullptr);
    profiling::ScopedPhase phase(profiling::Phase::CiahStructure, true);
    const auto start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point coupling_end;
    if (direct_ci) {
        StructureActiveIntegralAdjoint adjoint;
        result.ci = structure_.finish_coupling_coordinates(
            raw, std::move(request.result), &adjoint, &correction);
        coupling_end = std::chrono::steady_clock::now();
        result.adjoint = hessian_.apply_structure_coupling_adjoint(adjoint);
    } else {
        result.ci = structure_.apply_coupling_coordinates(q);
        result.ci.adjoint_multipliers += correction;
        coupling_end = std::chrono::steady_clock::now();
        result.adjoint = hessian_.apply_structure_coupling_adjoint(result.ci.coefficient_response,
                                                                   result.ci.adjoint_multipliers);
    }
    const auto adjoint_end = std::chrono::steady_clock::now();
    Eigen::MatrixXd forcing = residual_ * result.orbital.gauge_frame;
    for (Eigen::Index i = 0; i < forcing.cols(); ++i)
        forcing.col(i) *= scales_[i];
    forcing += result.orbital.scaled_structure_forcing;
    result.forcing = structure_.project_coordinates(forcing);
    mcscf::diagnostics::ciah_trace(
        "ciah_structure_cost",
        "coupling=%.17g,orbital_adjoint=%.17g,projection=%.17g",
        std::chrono::duration<double>(coupling_end - start).count(),
        std::chrono::duration<double>(adjoint_end - coupling_end).count(),
        std::chrono::duration<double>(std::chrono::steady_clock::now() - adjoint_end).count());
    return result;
}

} // namespace xmvb::vb
