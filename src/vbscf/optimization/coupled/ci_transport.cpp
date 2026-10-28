#include "vbscf/optimization/coupled/ci_transport.hpp"

#include <cmath>
#include <chrono>
#include "linalg/metric_solve.hpp"
#include "core/timing.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/derivatives/hessian/coupled/structure.hpp"
#include "mcscf/optimization/trace.hpp"
#include "vbscf/optimization/objective/function.hpp"
#include "vbscf/optimization/trust_region/retraction.hpp"
#include "vbscf/orbitals/preparation/cholesky.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"
#include "vbscf/structures/orthogonal_ci/metric.hpp"
#include "vbscf/structures/overlap/metric.hpp"
#include "vbscf/structures/overlap/spin_metric.hpp"

namespace xmvb::vb {

CiTransport::CiTransport(const OrbitalPreparationInput& input,
                         const SparseParameterLayout& layout,
                         const OrbitalChart& chart,
                         const AcceptedPointContext& origin,
                         const StructureAction& action,
                         const StructureTangentOperator& structure)
    : input_(input), layout_(layout), chart_(chart), origin_(origin), action_(action),
      structure_(structure),
      overlap_(action,
               origin.same_spin_pair_cache.alpha_reuse_table.unique_determinants,
               origin.same_spin_pair_cache.beta_reuse_table.unique_determinants,
               input.n_active_orbitals),
      origin_pullback_(input, origin.prepared_active_space.orbital_result, layout),
      upper_(origin.prepared_active_space.active_space_two_electron_result.orbital_transform),
      zero_(Eigen::MatrixXd::Zero(input.n_active_orbitals, input.n_active_orbitals)),
      c0_(origin.selected_state_eigenvectors),
      coordinate_scale_(std::sqrt(2.0 * origin.normalized_state_weights.at(0))),
      diagonal_(action.preconditioner_diagonal().overlap), inverse_(origin.metric_inverse(action)),
      metric_upper_(upper_), selected_metric_(origin.selected_metric) {
    const auto& images = origin.selected_state_structure_images.value();
    const Eigen::MatrixXd state_h = c0_.transpose() * images.hamiltonian;
    gc0_ =
        (coordinate_scale_ * coordinate_scale_) * (images.hamiltonian - images.overlap * state_h);
}

Eigen::VectorXd CiTransport::raw_direction(const Eigen::VectorXd& reduced) const {
    return gather_nonredundant_retract_tangent(input_, chart_, layout_, reduced);
}

Eigen::MatrixXd CiTransport::solve(const Eigen::MatrixXd& upper,
                                   const Eigen::MatrixXd& rhs,
                                   const Eigen::MatrixXd* initial) const {
    // Opt-in diagnosis only: separate the selected metric solve from HVP work.
    const bool timed = profiling::phase_totals().enabled;
    using Clock = std::chrono::steady_clock;
    const auto start = timed ? Clock::now() : Clock::time_point{};
    double action_seconds = 0.0;
    const auto report =
        [&](const char* mode, int actions, int iterations, int certifications, double residual) {
            if (timed)
                std::fprintf(
                    stderr,
                    "XMVB_TRANSPORT_SOLVE,mode=%s,rows=%ld,rhs=%ld,actions=%d,iterations=%d,"
                    "certifications=%d,relative_residual=%.9g,wall=%.9f,action_wall=%.9f\n",
                    mode,
                    static_cast<long>(rhs.rows()),
                    static_cast<long>(rhs.cols()),
                    actions,
                    iterations,
                    certifications,
                    residual,
                    std::chrono::duration<double>(Clock::now() - start).count(),
                    action_seconds);
        };
    if (inverse_) {
        const ExteriorOrbitalTransform alpha(
            origin_.same_spin_pair_cache.alpha_reuse_table.unique_determinants, upper);
        const ExteriorOrbitalTransform beta(
            origin_.same_spin_pair_cache.beta_reuse_table.unique_determinants, upper);
        Eigen::MatrixXd result = inverse_->apply_at_transform(rhs, alpha, beta);
        report("factorized", 0, 0, 0, 0.0);
        return result;
    }
    // Actual selected M, not the identity on a presumed closed carrier.
    if ((!prepared_metric_ && !selected_metric_) || metric_upper_.rows() != upper.rows() ||
        !(metric_upper_.array() == upper.array()).all()) {
        if (origin_.selected_metric && (upper_.array() == upper.array()).all()) {
            selected_metric_ = origin_.selected_metric;
            prepared_metric_.reset();
        } else {
            prepared_metric_.emplace(overlap_.prepare_metric(upper));
            selected_metric_ = prepared_metric_->selected_metric();
        }
        metric_upper_ = upper;
    }
    // Reuse the exact bounded Gram factor, including at its own trial chart.
    // Streamed selections retain Jacobi; neither path completes the VB space.
    // Both are certified against the same selected metric at unchanged accuracy.
    Eigen::MatrixXd result(rhs.rows(), rhs.cols());
    int actions = 0, iterations = 0, certifications = 0;
    double residual = 0.0;
    for (Eigen::Index s = 0; s < rhs.cols(); ++s) {
        linalg::MetricSolveStats stats;
        Eigen::VectorXd seed;
        if (initial)
            seed = initial->col(s);
        try {
            result.col(s) = linalg::solve_metric(
                rhs.col(s),
                [&](const Eigen::VectorXd& x) {
                    const auto action_start = timed ? Clock::now() : Clock::time_point{};
                    Eigen::VectorXd image = selected_metric_
                                                ? Eigen::VectorXd(selected_metric_->apply(x))
                                                : Eigen::VectorXd((*prepared_metric_)(x).col(0));
                    if (timed)
                        action_seconds +=
                            std::chrono::duration<double>(Clock::now() - action_start).count();
                    return image;
                },
                [&](const Eigen::VectorXd& x) {
                    if (selected_metric_)
                        return Eigen::VectorXd(selected_metric_->solve(x));
                    if (origin_.spin_metric)
                        return origin_.spin_metric->precondition(x, diagonal_);
                    return Eigen::VectorXd(x.cwiseQuotient(diagonal_));
                },
                &stats,
                initial ? &seed : nullptr);
        } catch (const std::runtime_error& error) {
            report("selected_cg_failed",
                   stats.actions,
                   stats.iterations,
                   stats.certifications,
                   stats.relative_residual);
            throw std::runtime_error(std::string("CI transport selected metric: ") + error.what());
        }
        actions += stats.actions;
        iterations += stats.iterations;
        certifications += stats.certifications;
        residual = std::max(residual, stats.relative_residual);
    }
    report("selected_cg", actions, iterations, certifications, residual);
    return result;
}

CiTransportPoint CiTransport::retract(const Eigen::VectorXd& orbital_step,
                                      const Eigen::VectorXd& structure_step) const {
    profiling::ScopedPhase phase(profiling::Phase::CiTransportRetract);
    CiTransportPoint point;
    point.input = chart_.retract_step(input_, orbital_step);
    point.prepared = ActiveSpaceOrbitalPreparer{}.prepare(point.input);
    const auto active = point.prepared.auxiliary_orbital_matrix.middleCols(
        (input_.n_total_electrons - input_.n_active_electrons) / 2, input_.n_active_orbitals);
    point.upper = active_frame(active,
                               point.input.ao_overlap_matrix,
                               Eigen::Map<const Eigen::MatrixXd>(
                                   point.prepared.active_orbital_overlap_matrix.data(),
                                   input_.n_active_orbitals,
                                   input_.n_active_orbitals))
                      .upper;
    point.origin_direction = overlap_upper_direction(
        upper_, origin_pullback_.overlap_direction(raw_direction(orbital_step)));
    point.base_coefficients = c0_ + structure_.coefficient_response_coordinates(structure_step);
    const Eigen::MatrixXd rhs =
        overlap_.apply(point.upper, upper_, point.base_coefficients) +
        overlap_.apply_direction(point.upper, upper_, zero_, point.origin_direction, c0_);
    // The target already agrees with F' * base_coefficients to first order
    // in the orbital step (at zero independent CI step). Correct the defect
    // of this known representative, with the original RHS-relative tolerance.
    point.coefficients = solve(point.upper, rhs, &point.base_coefficients);
    // solve() has prepared this chart's Gram. Retain its immutable ownership,
    // since a later retraction may replace the transport workspace's trial point.
    if (!inverse_)
        point.selected_metric = selected_metric_;
    return point;
}

CiTransportModel CiTransport::evaluate_model(const CiTransportPoint& point,
                                             const VbScfCoupledKeyframeEvaluation& fixed) const {
    profiling::ScopedPhase phase(profiling::Phase::CiTransportGradient);
    // E_hat = E - <gc0, C_transport-C_affine>. Its first jet is physical,
    // while the residual-weighted transport second jet cancels algebraically.
    // Retain the actual finite gc0: no stationarity cutoff or zero replacement.
    const Eigen::MatrixXd lambda = solve(point.upper, fixed.coefficient_gradient - gc0_);
    Eigen::MatrixXd current_upper = zero_, origin_upper = zero_;
    Eigen::MatrixXd structure_gradient(c0_.rows(), c0_.cols());
    for (Eigen::Index s = 0; s < c0_.cols(); ++s) {
        const auto pullback = overlap_.transport_gradient(point.upper,
                                                          upper_,
                                                          point.origin_direction,
                                                          lambda.col(s),
                                                          point.base_coefficients.col(s),
                                                          c0_.col(s),
                                                          point.coefficients.col(s));
        current_upper += pullback.charts.left;
        origin_upper += pullback.charts.right;
        structure_gradient.col(s) = (pullback.image + gc0_.col(s)) / coordinate_scale_;
    }
    const OrbitalOverlapPullback current_pullback(point.input, point.prepared, layout_);
    Eigen::VectorXd orbital =
        fixed.orbital_gradient +
        current_pullback.gradient(overlap_upper_gradient(point.upper, current_upper)) +
        origin_pullback_.gradient(overlap_upper_gradient(upper_, origin_upper));
    CiTransportModel result;
    result.energy = fixed.energy -
                    (gc0_.array() * (point.coefficients - point.base_coefficients).array()).sum();
    result.gradient.resize(chart_.reduced_size() + structure_.tangent_size());
    result.gradient.head(chart_.reduced_size()) = chart_.project_reduced_gradient(orbital);
    result.gradient.tail(structure_.tangent_size()) =
        structure_.project_coordinates(structure_gradient);
    return result;
}

Eigen::VectorXd CiTransport::origin_structure_gradient() const {
    return structure_.project_coordinates(gc0_ / coordinate_scale_);
}

} // namespace xmvb::vb
