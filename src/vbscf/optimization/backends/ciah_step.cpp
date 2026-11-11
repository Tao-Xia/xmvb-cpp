#include <Eigen/SVD>
#include "vbscf/optimization/backends/ciah_step.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <limits>
#include <optional>
#include <string_view>
#include <stdexcept>
#include <utility>

#include "linalg/metric_solve.hpp"

#include "vbscf/derivatives/hessian/coupled/structure.hpp"
#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "mcscf/optimization/trace.hpp"
#include "vbscf/diagnostics/orbitals/ciah_keyframe.hpp"
#include "vbscf/diagnostics/orbitals/step_geometry.hpp"
#include "core/timing.hpp"
#include "core/workspace.hpp"
#include "core/index.hpp"
#include "vbscf/optimization/driver/options.hpp"
#include "vbscf/optimization/driver/result.hpp"
#include "vbscf/optimization/driver/session.hpp"
#include "vbscf/optimization/globalization/line_search.hpp"
#include "mcscf/optimization/ciah/driver.hpp"
#include "mcscf/optimization/ciah/driver_core.hpp"
#include "mcscf/optimization/ciah/macro.hpp"
#include "vbscf/optimization/coupled/ci_transport.hpp"
#include "vbscf/optimization/coupled/affine.hpp"
#include "vbscf/optimization/coupled/response_space.hpp"
#include "mcscf/optimization/preconditioners/coarse.hpp"
#include "vbscf/optimization/trust_region/retraction.hpp"

namespace xmvb::vb::optimizer_detail {

bool CiahMacroStep::check_convergence(const SparseParameterLayout& layout,
                                      const VbScfOptimizerOptions& options,
                                      bool complete_active_space,
                                      double source_energy) {
    converged_geometry.reset();
    if (!(std::abs(trial.energy - source_energy) < options.energy_tolerance))
        return false;
    if (!trial.gradient_ready)
        throw std::logic_error("CIAH terminal candidate requires its true gradient");
    auto geometry = build_orbital_geometry(trial.orbital_preparation_input,
                                           trial.gradient_result.orbital_preparation_result,
                                           layout,
                                           complete_active_space,
                                           OrbitalChart::Coordinates::Unwhitened);
    const auto projected = geometry.project_gradient(trial.gradient);
    if (!orbital_gradient_converged(trial.orbital_preparation_input,
                                    layout,
                                    geometry,
                                    projected.reduced_gradient,
                                    options.gradient_tolerance))
        return false;
    converged_geometry.emplace(std::move(geometry));
    return true;
}

void CiahMacroStep::align_structure_frame(const Eigen::MatrixXd& reference,
                                          const Eigen::MatrixXd& overlap_roots) {
    if (reference.rows() != overlap_roots.rows() || reference.cols() != overlap_roots.cols() ||
        next_structure_guess.rows() != reference.rows() ||
        next_structure_guess.cols() != reference.cols())
        throw std::invalid_argument("invalid CIAH restart frame operands");
    if (reference.cols() == 1) {
        if (reference.col(0).dot(overlap_roots.col(0)) < 0.0)
            next_structure_guess *= -1.0;
        return;
    }
    const Eigen::MatrixXd cross = reference.transpose() * overlap_roots;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(cross, Eigen::ComputeFullU | Eigen::ComputeFullV);
    next_structure_guess =
        (next_structure_guess * (svd.matrixU() * svd.matrixV().transpose())).eval();
}

bool build_ciah_macro_step(VbScfObjective* objective,
                           const SparseParameterLayout& layout,
                           const VbScfOptimizerOptions& options,
                           const OrbitalChart& chart,
                           const NonredundantRetractionMetric& metric,
                           ExactHvpOperator& hessian,
                           const StructureTangentOperator& structure,
                           const Eigen::VectorXd& gradient,
                           double gradient_norm,
                           const Eigen::VectorXd& orbital_guess_packed,
                           const Eigen::MatrixXd& structure_guess,
                           bool use_structure_response,
                           double energy,
                           double* trust_radius,
                           CiahIterationRecord* record,
                           VbScfOptimizerResult* result,
                           CiahMacroStep* accepted) {
    xmvb::profiling::ScopedPhase macro_phase(xmvb::profiling::Phase::CiahMacro, true);
    xmvb::profiling::ScopedPhase setup_phase(xmvb::profiling::Phase::CiahSetup);
    const auto context = objective->second_order_context();
    const auto orbitals = objective->input().orbital_preparation_input;
    const Eigen::Index no = chart.reduced_size();
    const Eigen::Index full_ns = use_structure_response ? structure.tangent_size() : 0;
    mcscf::CiahOptions ciah_options;
    // Active balancing removes inactive additions, and the horizontal lift
    // annihilates the first derivative of the active overlap in both slots.
    // In the orbital-only model the remaining CI cotangents are fixed.
    const bool fixed_active_overlap =
        context->selected_state_indices.size() == 1 &&
        objective->input().standard_two_electron_mode == StandardTwoElectronMode::Exact &&
        objective->active_inactive_orthogonal() && chart.has_full_horizontal_active_lift();
    const bool fixed_overlap_coupling =
        full_ns != 0 && fixed_active_overlap &&
        hessian.structure_action().supports_fixed_overlap_coupling();
    const char* small_ci_flag = std::getenv("XMVB_CIAH_SMALL_CI");
    const bool small_ci = fixed_overlap_coupling && small_ci_flag &&
                          std::string_view(small_ci_flag) == "1";
    const Eigen::Index ns = small_ci ? std::min<Eigen::Index>(full_ns, ciah_options.max_cycle)
                                     : full_ns;
    const char* split_budget = std::getenv("XMVB_CIAH_SPLIT_BUDGET");
    if (small_ci && split_budget && std::string_view(split_budget) == "1")
        ciah_options.complement_capacity = static_cast<int>(ns);
    const Eigen::Index dimension = no + ns;
    const bool fixed_density_orbital_hvp = (ns == 0 || small_ci) && fixed_active_overlap;
    mcscf::diagnostics::ciah_trace(
        "ciah_orbital_hvp_model",
        "fixed_density=%d,balanced=%d,full_horizontal=%d,coupled=%d,fixed_overlap_coupling=%d",
        fixed_density_orbital_hvp,
        objective->active_inactive_orthogonal(),
        chart.has_full_horizontal_active_lift(),
        ns != 0,
        fixed_overlap_coupling);
    const bool step_audit = diagnostics::orbital_step_audit_enabled();
    if (step_audit)
        diagnostics::audit_ciah_state(
            orbitals, context->selected_state_eigenvectors, "origin", energy);
    std::unique_ptr<CiTransport> transport;
    if (ns)
        transport = std::make_unique<CiTransport>(
            orbitals, layout, chart, *context, hessian.structure_action(), structure,
            fixed_active_overlap);
    std::optional<AffineCoupling> affine;
    if (ns)
        affine.emplace(*context, hessian, structure, fixed_overlap_coupling);
    std::optional<CiahCiResponseSpace> response;
    if (small_ci)
        response.emplace(structure, *affine, no, ns);
    const auto has_ci = [&] { return response ? response->rank() != 0 : ns != 0; };
    double energy_scale = 0.0;
    for (const double state_energy : context->selected_state_energies)
        energy_scale = std::max(energy_scale, std::abs(state_energy));
    ciah_options.operator_relative_accuracy =
        context->structure_solve_accuracy.response_backward_error_tolerance(energy_scale);
    mcscf::CoarsePreconditioner coarse(dimension,
                                       ns && !small_ci ? 2 * ciah_options.max_cycle : 0,
                                       ciah_options.lindep,
                                       ciah_options.operator_relative_accuracy);
    Eigen::VectorXd orbital_image(ns && !small_ci ? dimension : 0);
    Eigen::VectorXd structure_image(orbital_image.size());
    record->model_dimension = static_cast<int>(dimension);
    record->source_gradient_norm = chart.coordinate_dual_norm(gradient);
    record->source_physical_gradient_norm = gradient_norm;

    // Internal CIAH control is in the full coordinate vector, not a projected
    // estimate. Only the cheap local AO inverse is used for coordinate units;
    // there is no complete physical-metric solve for progress/keyframes.
    const auto coordinate_norm = [&](const Eigen::VectorXd& g) {
        return std::hypot(chart.coordinate_dual_norm(g.head(no)), ns ? g.tail(ns).norm() : 0.0);
    };
    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> startup_images;
    bool sample_startup = false;
    const auto action = [&](const Eigen::VectorXd& x, Eigen::VectorXd* metric_image) {
        Eigen::VectorXd image(dimension);
        const bool has_orbital = !small_ci || !x.head(no).isZero(0.0);
        const bool share_lift = has_orbital && fixed_density_orbital_hvp && !sample_startup;
        const Eigen::VectorXd packed = share_lift ? chart.expand_step(x.head(no)) : Eigen::VectorXd{};
        if (metric_image) {
            metric_image->resize(dimension);
            if (!has_orbital)
                metric_image->head(no).setZero();
            else if (share_lift)
                metric_image->head(no) = metric.apply_packed(packed);
            else
                metric_image->head(no) = metric.apply(x.head(no));
        }
        if (ns == 0 || small_ci) {
            image.head(no).setZero();
            if (sample_startup && has_orbital) {
                auto [ap, bp] = response->orbital_images(x.head(no));
                image.head(no) = ap;
                startup_images.emplace_back(x.head(no), std::move(bp));
                ++record->coupled_block_actions;
            } else if (has_orbital) {
                image.head(no) = fixed_density_orbital_hvp
                        ? hessian.apply_packed(packed,
                                                {.direct_core_response = true,
                                                 .fixed_upstream_pullback = true,
                                                 .local_active_response = false,
                                                 .structure_response = false})
                        : hessian.apply_orbital(x.head(no));
            }
            if (response) {
                response->add_hessian_image(x.head(no), x.tail(ns), image.head(no), image.tail(ns));
                if (metric_image)
                    metric_image->tail(ns) = response->metric(x.tail(ns));
                if (!has_orbital)
                    return image;
            }
        } else {
            const auto blocks = affine->apply(x.head(no), x.tail(ns));
            const auto& orbital = blocks.orbital;
            const auto& ci = blocks.ci;
            const auto& adjoint = blocks.adjoint;
            if (metric_image)
                metric_image->tail(ns) = ci.metric_coordinates;
            const auto& forcing = blocks.forcing;
            if (orbital.connection.has_value()) {
                xmvb::profiling::ScopedPhase connection_phase(
                    xmvb::profiling::Phase::CiahConnection, true);
                const auto& connection = *orbital.connection;
                Eigen::VectorXd representation =
                    structure.project_coordinates(connection.representation_forcing);
                representation +=
                    forcing - structure.project_coordinates(orbital.scaled_structure_forcing);
                const Eigen::VectorXd integral =
                    structure.project_coordinates(connection.integral_forcing);
                const Eigen::VectorXd rhs = structure.project_coordinates(connection.overlap);
                xmvb::linalg::MetricSolveStats connection_stats;
                const Eigen::VectorXd gamma = xmvb::linalg::solve_metric(
                    rhs,
                    [&](const Eigen::VectorXd& v) { return structure.apply_metric_coordinates(v); },
                    [&](const Eigen::VectorXd& v) {
                        return structure.apply_inverse_metric_model(v);
                    },
                    &connection_stats);
                const Eigen::VectorXd kinematic = structure.apply_hessian_coordinates(gamma);
                const double representation_norm = structure.dual_norm(representation);
                const double integral_norm = structure.dual_norm(integral);
                const double forcing_norm = structure.dual_norm(forcing);
                mcscf::diagnostics::ciah_trace(
                    "ciah_connection",
                    "representation=%.17g,integral=%.17g,total=%.17g,cross=%.17g,"
                    "reconstruction=%.17g,closure_squared=%.17g,carrier_squared=%.17g,"
                    "kinematic_error=%.17g,metric_actions=%d,metric_residual=%.17g",
                    representation_norm,
                    integral_norm,
                    forcing_norm,
                    0.5 * (forcing_norm * forcing_norm - representation_norm * representation_norm -
                           integral_norm * integral_norm),
                    (forcing - representation - integral).norm(),
                    connection.squared_norm.sum() - rhs.dot(gamma),
                    connection.squared_norm.sum(),
                    structure.dual_norm(representation - kinematic),
                    connection_stats.actions,
                    connection_stats.relative_residual);
            }
            image.head(no) = orbital.orbital_hessian + adjoint;
            image.tail(ns) = forcing + ci.hessian_coordinates;
            orbital_image.head(no) = orbital.orbital_hessian;
            orbital_image.tail(ns) = forcing;
            structure_image.head(no) = adjoint;
            structure_image.tail(ns) = ci.hessian_coordinates;
            // The residual-balanced finite model has the physical affine Hessian
            // at this origin, even for a finite CI residual. Its coordinate second
            // jet is cancelled in the keyframe energy AND gradient, not discarded.
            if (mcscf::diagnostics::ciah_event_trace_enabled()) {
                const double orbital_energy = x.head(no).dot(orbital.orbital_hessian);
                const double structure_energy = x.tail(ns).dot(ci.hessian_coordinates);
                const double backward_coupling = x.head(no).dot(adjoint);
                const double forward_coupling = x.tail(ns).dot(forcing);
                mcscf::diagnostics::ciah_trace(
                    "ciah_block_curvature",
                    "orbital=%.17g,structure=%.17g,coupling_forward=%.17g,coupling_backward=%.17g",
                    orbital_energy,
                    structure_energy,
                    forward_coupling,
                    backward_coupling);
            }
            ++record->structure_response_actions;
        }
        ++record->orbital_hvp_actions;
        if (!small_ci)
            ++record->coupled_block_actions;
        return image;
    };
    const auto metric_action = [&](const Eigen::VectorXd& x) {
        Eigen::VectorXd image(dimension);
        image.head(no) = metric.apply(x.head(no));
        if (ns)
            image.tail(ns) = response ? response->metric(x.tail(ns))
                                      : structure.apply_metric_coordinates(x.tail(ns));
        return image;
    };
    const auto diagonal_precondition = [&](const Eigen::VectorXd& g, double eigenvalue) {
        Eigen::VectorXd direction(dimension);
        if (options.orbital_preconditioner == OrbitalPreconditioner::Identity) {
            direction.head(no) = chart.inverse_coordinate_metric(g.head(no));
        } else if (options.orbital_preconditioner == OrbitalPreconditioner::OneElectron) {
            // The one-electron model is a positive block preconditioner, not a
            // signed approximation to the complete AH operator. Restore its
            // unshifted inverse; the complete physical metric stays in metric_action.
            direction.head(no) = chart.apply_inverse_reduced_block_preconditioner(g.head(no));
        } else {
            direction.head(no) =
                chart.apply_inverse_augmented_hessian_diagonal(g.head(no), eigenvalue - 1.0e-8);
        }
        if (response)
            direction.tail(ns).setZero(); // All admitted CI coordinates are already in Davidson.
        else if (ns)
            direction.tail(ns) =
                structure.apply_inverse_augmented_hessian_diagonal(g.tail(ns), eigenvalue - 1.0e-8);
        return direction;
    };
    const auto precondition = [&](const Eigen::VectorXd& g, double eigenvalue) {
        return coarse.apply(g, eigenvalue, [&](const Eigen::VectorXd& v) {
            return diagonal_precondition(v, eigenvalue);
        });
    };
    if (ns && !small_ci)
        ciah_options.observe_images = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& gx) {
            xmvb::profiling::ScopedPhase observe_phase(xmvb::profiling::Phase::CiahObserve);
            Eigen::VectorXd v = Eigen::VectorXd::Zero(dimension), g = v;
            v.head(no) = x.head(no);
            g.head(no) = gx.head(no);
            coarse.observe(v, orbital_image, g);
            v.head(no).setZero();
            g.head(no).setZero();
            v.tail(ns) = x.tail(ns);
            g.tail(ns) = gx.tail(ns);
            coarse.observe(v, structure_image, g);
        };
    Eigen::VectorXd initial_gradient = Eigen::VectorXd::Zero(dimension);
    initial_gradient.head(no) = gradient;
    Eigen::VectorXd ci_gradient;
    if (ns) {
        ci_gradient = transport->origin_structure_gradient();
        if (!response)
            initial_gradient.tail(ns) = ci_gradient;
    }
    // QuadraticProblem also serves trust-region solvers that require a physical dual
    // norm. CIAH deliberately does not call this callback.
    mcscf::QuadraticProblem problem(
        initial_gradient,
        [&](const Eigen::VectorXd& x) { return action(x, nullptr); },
        metric_action,
        [&](const Eigen::VectorXd& g) {
            return std::hypot(metric.dual_norm(g.head(no)),
                              response ? g.tail(ns).norm()
                                       : ns ? structure.dual_norm(g.tail(ns)) : 0.0);
        });
    ciah_options.apply_images = [&](const Eigen::VectorXd& x) {
        Eigen::VectorXd metric_image;
        Eigen::VectorXd hessian_image = action(x, &metric_image);
        return std::make_pair(std::move(hessian_image), std::move(metric_image));
    };
    ciah_options.residual_preconditioner = precondition;
    ciah_options.coordinate_norm = coordinate_norm;
    ciah_options.residual_direction = [&](const Eigen::VectorXd& g) {
        Eigen::VectorXd v = g;
        v.head(no) = chart.inverse_coordinate_metric(g.head(no));
        return v;
    };
    Eigen::VectorXd seed = initial_gradient;
    seed.head(no) = chart.inverse_coordinate_metric(initial_gradient.head(no));
    if (orbital_guess_packed.size() == static_cast<Eigen::Index>(layout.size())) {
        seed.head(no) = chart.project_vector(orbital_guess_packed).reduced_gradient;
    }
    if (ns && !response && structure_guess.rows() == context->selected_state_eigenvectors.rows() &&
        structure_guess.cols() == context->selected_state_eigenvectors.cols()) {
        seed.tail(ns) = structure.project_coordinates(
            std::sqrt(2.0 * context->normalized_state_weights[0]) * structure_guess);
    }
    mcscf::CiahWorkspace workspace(problem, seed, ciah_options);

    const Eigen::MatrixXd& c0 = context->selected_state_eigenvectors;
    std::optional<CiTransportPoint> transported;
    Eigen::VectorXd transported_step;
    const auto transport_at = [&](const Eigen::VectorXd& displacement) -> const CiTransportPoint& {
        // One immutable macro model: equal coordinates request the same map.
        // Keyframe and macro-trial consumers share it without another Gram solve.
        if (!transported || transported_step.size() != displacement.size() ||
            !(transported_step.array() == displacement.array()).all()) {
            if (response)
                transported = transport->retract(displacement.head(no),
                                                 response->lift(displacement.tail(ns)));
            else
                transported = transport->retract(displacement.head(no), displacement.tail(ns));
            transported_step = displacement;
        }
        return *transported;
    };
    const auto evaluate_keyframe = [&](const Eigen::VectorXd& displacement,
                                       Eigen::VectorXd* full_ci_gradient = nullptr) {
        xmvb::profiling::ScopedPhase keyframe_phase(xmvb::profiling::Phase::CiahKeyframe, true);
        mcscf::CiahKeyframe keyframe;
        if (has_ci()) {
            const auto& point = transport_at(displacement);
            const auto exact = objective->evaluate_coupled_keyframe(point);
            if (step_audit)
                diagnostics::audit_ciah_state(
                    point.input, exact.normalized_structure_coefficients, "keyframe", exact.energy);
            auto model = transport->evaluate_model(point, exact);
            if (response) {
                keyframe.gradient.resize(dimension);
                keyframe.gradient.head(no) = model.gradient.head(no);
                keyframe.gradient.tail(ns) = response->project(model.gradient.tail(full_ns));
                if (full_ci_gradient)
                    *full_ci_gradient = model.gradient.tail(full_ns);
            } else {
                keyframe.gradient = std::move(model.gradient);
            }
            keyframe.energy = model.energy;
            keyframe.physical_energy = exact.energy;
        } else {
            const auto parameters = build_nonredundant_lifted_trial_parameters(
                orbitals, chart, layout, displacement.head(no));
            const auto exact = objective->evaluate_coupled_keyframe(parameters, c0);
            keyframe.gradient = Eigen::VectorXd::Zero(dimension);
            keyframe.gradient.head(no) = chart.project_reduced_gradient(exact.orbital_gradient);
            keyframe.energy = exact.energy;
            keyframe.physical_energy = exact.energy;
        }
        ++record->keyframes;
        return keyframe;
    };

    mcscf::CiahMicroCallbacks callbacks;
    callbacks.coordinate_norm = coordinate_norm;
    callbacks.metric_action = metric_action;
    int ci_step_limit_calls = 0, ci_step_limit_skipped = 0;
    double ci_step_limit_seconds = 0.0;
    const bool trace_step_limit = mcscf::diagnostics::ciah_event_trace_enabled();
    const double ci_scale = std::sqrt(2.0 * context->normalized_state_weights.front());
    callbacks.maximum_component = [&](const Eigen::VectorXd& direction) {
        const double largest = chart.maximum_rotation_component(direction.head(no));
        if (!has_ci())
            return largest;
        ++ci_step_limit_calls;
        const auto y = direction.tail(ns);
        const bool bounded = response
            ? response->bounded_by(y, ci_scale * largest, ciah_options.operator_relative_accuracy)
            : y.isZero(0.0);
        if (bounded && !mcscf::diagnostics::ciah_trace_enabled()) {
            ++ci_step_limit_skipped;
            return largest;
        }
        const auto start = trace_step_limit ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point{};
        const double ci_largest = response
            ? structure.maximum_coefficient_component_coordinates(response->lift(y))
            : structure.maximum_coefficient_component_coordinates(y);
        if (trace_step_limit)
            ci_step_limit_seconds +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (bounded && ci_largest > largest)
            throw std::logic_error("CIAH CI metric bound lost the limiting component");
        return std::max(largest, ci_largest);
    };
    xmvb::workspace::Reservation rotation_limit_memory;
    std::vector<Eigen::VectorXd> rotation_images;
    Eigen::VectorXd rotation_trial;
    int rotation_limit_samples = 0, rotation_limit_screened = 0;
    const Eigen::Index rotation_size = chart.rotation_component_count();
    if (rotation_size && (response || ns == 0)) {
        workspace.set_projected_step_limit(
            [&](const std::vector<Eigen::VectorXd>& basis,
                const Eigen::VectorXd& weights) -> std::optional<double> {
                ++rotation_limit_samples;
                if (rotation_images.size() != basis.size()) {
                    const auto bytes = core::checked_multiply(
                        sizeof(double) * (basis.size() + 3), static_cast<std::size_t>(rotation_size));
                    if (!rotation_limit_memory.try_resize(bytes))
                        return std::nullopt;
                    while (rotation_images.size() < basis.size())
                        rotation_images.push_back(chart.frame_rotation_components(
                            basis[rotation_images.size()].head(no)));
                }
                rotation_trial.setZero(rotation_size);
                double absolute_bound = 0.0;
                for (Eigen::Index j = 0; j < weights.size(); ++j) {
                    rotation_trial.noalias() += weights[j] * rotation_images[j];
                    absolute_bound += std::abs(weights[j]) * rotation_images[j].lpNorm<Eigen::Infinity>();
                }
                const double orbital = rotation_trial.lpNorm<Eigen::Infinity>();
                const double cap = mcscf::detail::ciah_step_max_component;
                // Only screen away from the roundoff-scale decision boundary.
                const double guard = std::sqrt(std::numeric_limits<double>::epsilon()) *
                                     std::max(cap, absolute_bound);
                if (!std::isfinite(orbital) || std::abs(orbital - cap) <= guard)
                    return std::nullopt;
                if (response && orbital < cap) {
                    Eigen::VectorXd y = Eigen::VectorXd::Zero(ns);
                    for (Eigen::Index j = 0; j < weights.size(); ++j)
                        y.noalias() += weights[j] * basis[j].tail(ns);
                    if (!response->bounded_by(y, ci_scale * cap, ciah_options.operator_relative_accuracy))
                        return std::nullopt;
                }
                if (mcscf::diagnostics::ciah_trace_enabled()) {
                    Eigen::VectorXd direction = Eigen::VectorXd::Zero(dimension);
                    for (Eigen::Index j = 0; j < weights.size(); ++j)
                        direction.noalias() += weights[j] * basis[j];
                    if ((callbacks.maximum_component(direction) <= cap) != (orbital <= cap))
                        throw std::logic_error("CIAH projected step bound changed cap classification");
                }
                ++rotation_limit_screened;
                return orbital;
            });
    }
    Eigen::VectorXd keyframe_ci_gradient;
    if (response)
        keyframe_ci_gradient = ci_gradient;
    callbacks.evaluate_keyframe = [&](const Eigen::VectorXd& displacement) {
        if (!response)
            return evaluate_keyframe(displacement);
        keyframe_ci_gradient.resize(0);
        return evaluate_keyframe(displacement, &keyframe_ci_gradient);
    };
    callbacks.orbital_size = no;
    callbacks.orbital_coordinate_norm = [&](const Eigen::VectorXd& g) {
        return chart.coordinate_dual_norm(g);
    };
    if (step_audit)
        callbacks.audit_step = [&](const char* label,
                                   const Eigen::VectorXd& prior,
                                   const Eigen::VectorXd& step,
                                   int index) {
            if (std::string_view(label) == "micro_increment")
                diagnostics::audit_orbital_step(
                    orbitals, chart, step.head(no), label, index, prior.head(no));
            else
                diagnostics::audit_orbital_step(orbitals, chart, step.head(no), label, index);
        };
    if (mcscf::diagnostics::ciah_trace_enabled() &&
        mcscf::diagnostics::ciah_candidate_audit_enabled()) {
        callbacks.audit_origin = [&](const Eigen::VectorXd&,
                                     const Eigen::VectorXd& step,
                                     const Eigen::VectorXd& image,
                                     const Eigen::VectorXd&) {
            const double length = std::sqrt(std::max(0.0, step.dot(metric_action(step))));
            if (length > 0.0) {
                const Eigen::VectorXd direction = step / length;
                const Eigen::VectorXd exact_image = image / length;
                const double exact_norm = coordinate_norm(exact_image);
                const int previous_keyframes = record->keyframes;
                for (const double delta : {1.0e-3, 3.0e-4}) {
                    const auto plus = evaluate_keyframe(delta * direction);
                    const auto minus = evaluate_keyframe(-delta * direction);
                    const Eigen::VectorXd finite = (plus.gradient - minus.gradient) / (2.0 * delta);
                    const Eigen::VectorXd error = finite - exact_image;
                    const double derivative = initial_gradient.dot(direction);
                    mcscf::diagnostics::ciah_trace(
                        "ciah_origin_hvp_audit",
                        "delta=%.17g,image_norm=%.17g,finite_norm=%.17g,error_norm=%.17g,orbital_"
                        "error=%.17g,structure_error=%.17g,derivative=%.17g,finite_derivative=%."
                        "17g,orbital_direction_norm=%.17g,structure_direction_norm=%.17g",
                        delta,
                        exact_norm,
                        coordinate_norm(finite),
                        coordinate_norm(error),
                        error.head(no).norm(),
                        error.tail(ns).norm(),
                        derivative,
                        (plus.energy - minus.energy) / (2.0 * delta),
                        step.head(no).norm() / length,
                        ns ? step.tail(ns).norm() / length : 0.0);
                }
                record->keyframes = previous_keyframes;
            }
        };
    }
    if (mcscf::diagnostics::ciah_event_trace_enabled() &&
        mcscf::diagnostics::ciah_candidate_audit_enabled()) {
        callbacks.audit_resumed = [&](const Eigen::VectorXd& displacement,
                                      const Eigen::VectorXd& step,
                                      const Eigen::VectorXd& image,
                                      const Eigen::VectorXd& predicted_gradient,
                                      int basis_size) {
            const double length = std::sqrt(std::max(0.0, step.dot(metric_action(step))));
            if (length > 0.0) {
                const Eigen::VectorXd direction = step / length;
                const Eigen::VectorXd origin_image = image / length;
                const double derivative = predicted_gradient.dot(direction);
                for (const double delta : {1.0e-3, 3.0e-4}) {
                    const auto plus = evaluate_keyframe(displacement + delta * direction);
                    const auto minus = evaluate_keyframe(displacement - delta * direction);
                    const double finite = (plus.energy - minus.energy) / (2.0 * delta);
                    const Eigen::VectorXd local_image =
                        (plus.gradient - minus.gradient) / (2.0 * delta);
                    mcscf::diagnostics::ciah_trace(
                        "ciah_keyframe_direction_audit",
                        "basis=%d,delta=%.17g,analytic=%.17g,finite=%.17g,derivative_error=%.17g,"
                        "local_curvature=%.17g,origin_curvature=%.17g,curvature_change=%.17g,"
                        "orbital_derivative=%.17g,structure_derivative=%.17g",
                        basis_size,
                        delta,
                        derivative,
                        finite,
                        std::abs(derivative - finite),
                        coordinate_norm(local_image),
                        coordinate_norm(origin_image),
                        coordinate_norm(local_image - origin_image),
                        predicted_gradient.head(no).dot(direction.head(no)),
                        ns ? predicted_gradient.tail(ns).dot(direction.tail(ns)) : 0.0);
                }
            }
        };
    }
    if (ns && diagnostics::ciah_keyframe_audit_enabled()) {
        callbacks.audit_keyframe = [&](const Eigen::VectorXd& displacement,
                                       const Eigen::VectorXd& hessian_displacement,
                                       const mcscf::CiahKeyframe& exact) {
            const int saved_keyframes = record->keyframes;
            diagnostics::audit_ciah_keyframe(displacement,
                                             hessian_displacement,
                                             exact.gradient,
                                             static_cast<int>(no),
                                             [&](const Eigen::VectorXd& point) {
                                                 const auto value = evaluate_keyframe(point);
                                                 return diagnostics::CiahKeyframeProbe{
                                                     value.gradient, value.energy};
                                             });
            record->keyframes = saved_keyframes;
        };
    }
    setup_phase.stop();
    int forward_actions = 0;
    const auto probe = [&](const Eigen::VectorXd& direction) {
        if (direction.isZero(0.0))
            return Eigen::VectorXd::Zero(full_ns).eval();
        ++forward_actions;
        ++record->coupled_block_actions;
        return response->forcing(direction);
    };
    const auto has_capacity = [&] {
        return response->rank() < ns &&
               (ciah_options.complement_capacity || workspace.basis_size() < ciah_options.max_cycle);
    };
    const auto install_column = [&] {
        const Eigen::Index j = no + response->rank() - 1;
        const Eigen::VectorXd unit = Eigen::VectorXd::Unit(dimension, j);
        Eigen::VectorXd m;
        Eigen::VectorXd h = action(unit, &m);
        workspace.append_complement_coordinate(j, std::move(h), std::move(m));
    };
    const auto try_column = [&](const Eigen::VectorXd& residual, double omega) {
        if (!has_capacity() || !response->add(residual, omega, ciah_options.lindep))
            return false;
        ++record->coupled_block_actions;
        return true;
    };
    const auto add = [&](const Eigen::VectorXd& residual, double omega) {
        if (!try_column(residual, omega))
            return false;
        install_column();
        return true;
    };
    if (response) {
        const auto start = std::chrono::steady_clock::now();
        workspace.set_step_limit(callbacks.maximum_component, mcscf::detail::ciah_step_max_component);
        const double target = std::hypot(chart.coordinate_dual_norm(gradient), ci_gradient.norm()) /
                              mcscf::detail::ciah_gradient_trust;
        xmvb::workspace::Reservation startup_reservation;
        const auto startup_bytes = core::checked_multiply(
            sizeof(double) * mcscf::detail::ciah_ah_start_cycle,
            static_cast<std::size_t>(no + full_ns));
        sample_startup = startup_reservation.try_resize(startup_bytes);
        if (sample_startup)
            startup_images.reserve(mcscf::detail::ciah_ah_start_cycle);
        mcscf::CiahResult proposal;
        do {
            proposal = workspace.next(initial_gradient);
            ++record->micro_iterations;
        } while (!proposal.negative_curvature && !proposal.exhaustion && !proposal.converged &&
                 proposal.basis_size < mcscf::detail::ciah_ah_start_cycle);
        sample_startup = false;
        int reused_forcing = 0;
        const auto startup_forcing = [&](const Eigen::VectorXd& p) {
            if (startup_images.empty() || p.isZero(0.0))
                return probe(p);
            Eigen::VectorXd remainder = p;
            Eigen::VectorXd result = Eigen::VectorXd::Zero(full_ns);
            for (const auto& [basis, bp] : startup_images) {
                const double weight = basis.dot(p);
                remainder.noalias() -= weight * basis;
                result.noalias() += weight * bp;
            }
            // Ritz directions lie in the sampled orthonormal startup basis.
            // Preserve the exact forward path for any direction outside it.
            if (remainder.stableNorm() > ciah_options.lindep * p.stableNorm())
                return probe(p);
            ++reused_forcing;
            return result;
        };
        const Eigen::VectorXd p = proposal.step.head(no);
        Eigen::VectorXd soft = workspace.softest_ritz_direction().head(no);
        const double soft_norm = soft.stableNorm();
        if (!p.isZero(0.0)) {
            const Eigen::VectorXd unit = p / p.stableNorm();
            soft.noalias() -= unit.dot(soft) * unit;
        }
        const Eigen::VectorXd rhs = ci_gradient + startup_forcing(p);
        const double omega = proposal.eigenvalue;
        const bool soft_independent = soft.stableNorm() > ciah_options.lindep * soft_norm;
        // A soft mode can carry a coupled curvature direction even when the
        // gradient-driven step has little CI forcing.
        if (soft_independent)
            add(startup_forcing(soft), 0.0);
        mcscf::diagnostics::ciah_trace("ciah_small_ci_seed_images",
                                      "sampled=%zu,reused=%d,standalone=%d",
                                      startup_images.size(),
                                      reused_forcing,
                                      forward_actions);
        startup_images.clear();
        startup_reservation.try_resize(0);
        Eigen::VectorXd omitted;
        double omitted_norm = rhs.norm();
        double joint_norm = std::numeric_limits<double>::infinity();
        if (proposal.negative_curvature) {
            add(rhs - ci_gradient, 0.0);
        } else {
            while (true) {
                Eigen::VectorXd orbital_residual =
                    gradient + proposal.hessian_step.head(no) - omega * proposal.metric_step.head(no);
                if (!response->residual(rhs, omega, &omitted, &orbital_residual))
                    break;
                omitted_norm = omitted.norm();
                const double orbital_norm = chart.coordinate_dual_norm(orbital_residual);
                joint_norm = std::hypot(orbital_norm, omitted_norm);
                // This is a fixed-orbital seed, not a converged joint step.
                // Once the orbital block dominates, solve the joint AH before
                // spending more CI actions on the preliminary orbital vector.
                const double seed_target =
                    ciah_options.complement_capacity ? std::max(target, orbital_norm) : target;
                if (omitted_norm <= seed_target || !add(omitted, omega))
                    break;
            }
        }
        initial_gradient.tail(ns) = response->project(ci_gradient);
        workspace.reset_rhs();
        record->model_dimension = static_cast<int>(no + response->rank());
        mcscf::diagnostics::ciah_trace(
            "ciah_small_ci",
            "columns=%ld,forward_actions=%d,ci_residual=%.17g,joint_residual=%.17g,"
            "target=%.17g,seconds=%.17g",
            static_cast<long>(response->rank()),
            forward_actions,
            omitted_norm,
            joint_norm,
            target,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    }
    auto micro = mcscf::run_ciah_microsteps(
        workspace, initial_gradient, energy, options.gradient_tolerance, callbacks);
    if (mcscf::diagnostics::ciah_trace_enabled() && !micro.candidate.negative_curvature) {
        const auto& candidate = micro.candidate;
        // A diagnostic of the actual coupled trial space, not just its A block.
        // Neither inverse callback evaluates an HVP or changes the retained space.
        const auto base =
            workspace.inverse_spectrum(candidate.eigenvalue, [&](const Eigen::VectorXd& v) {
                return diagonal_precondition(v, candidate.eigenvalue);
            });
        const auto balanced =
            workspace.inverse_spectrum(candidate.eigenvalue, [&](const Eigen::VectorXd& v) {
                return precondition(v, candidate.eigenvalue);
            });
        mcscf::diagnostics::ciah_trace(
            "ciah_inverse_spectrum",
            "dimension=%d,positive_curvature=%d,curvature_min=%.17g,curvature_max=%.17g,base_min=%."
            "17g,base_max=%.17g,balanced_min=%.17g,balanced_max=%.17g,base_skew=%.17g,balanced_"
            "skew=%.17g",
            base.dimension,
            static_cast<int>(base.positive_curvature),
            base.curvature_min,
            base.curvature_max,
            base.inverse_min,
            base.inverse_max,
            balanced.inverse_min,
            balanced.inverse_max,
            base.inverse_skew,
            balanced.inverse_skew);
    }
    if (ns && !response && mcscf::diagnostics::ciah_event_trace_enabled()) {
        const auto schur = coarse.schur_spectrum(static_cast<int>(no));
        mcscf::diagnostics::ciah_trace(
            "ciah_schur_spectrum",
            "orbital_dimension=%d,structure_dimension=%d,structure_positive=%d,orbital_positive=%d,"
            "orbital_min=%.17g,orbital_max=%.17g,structure_min=%.17g,structure_max=%.17g,"
            "relaxed_min=%.17g,relaxed_max=%.17g,maximum_fraction_removed=%.17g",
            schur.orbital_dimension,
            schur.structure_dimension,
            static_cast<int>(schur.structure_positive),
            static_cast<int>(schur.orbital_positive),
            schur.orbital_min,
            schur.orbital_max,
            schur.structure_min,
            schur.structure_max,
            schur.relaxed_min,
            schur.relaxed_max,
            schur.maximum_fraction_removed);
    }
    if (ns && !response && mcscf::diagnostics::ciah_candidate_audit_enabled()) {
        const auto spectrum =
            coarse.orbital_spectrum(static_cast<int>(no), [&](const Eigen::VectorXd& v) {
                return diagonal_precondition(v, 0.0);
            });
        mcscf::diagnostics::ciah_trace(
            "ciah_orbital_preconditioner_spectrum",
            "dimension=%d,positive_curvature=%d,curvature_min=%.17g,curvature_max=%.17g,"
            "preconditioned_min=%.17g,preconditioned_max=%.17g",
            spectrum.dimension,
            static_cast<int>(spectrum.positive_curvature),
            spectrum.curvature_min,
            spectrum.curvature_max,
            spectrum.preconditioned_min,
            spectrum.preconditioned_max);
    }

    using MacroOps = mcscf::detail::HostCiahVectorOps;
    using Trial = VbScfObjectiveTrialEvaluation;
    mcscf::CiahMacroCallbacks<MacroOps, Trial> macro_callbacks;
    macro_callbacks.maximum_component = callbacks.maximum_component;
    macro_callbacks.step_norm = [&](const Eigen::VectorXd& step) {
        return metric.norm(step.head(no));
    };
    const Eigen::VectorXd source_parameters = layout.pack(orbitals);
    macro_callbacks.changes_parameters = [&](const Eigen::VectorXd& step) {
        const auto parameters =
            build_nonredundant_lifted_trial_parameters(orbitals, chart, layout, step.head(no));
        return (parameters - source_parameters).norm() >
               std::numeric_limits<double>::epsilon() * source_parameters.norm();
    };
    const double ci_target = context->structure_solve_accuracy.stationarity_tolerance();
    const bool adaptive_ci = context->structure_metric_inverse &&
                             context->selected_state_indices == std::vector<int>{0};
    std::optional<double> strict_source_energy;
    macro_callbacks.evaluate = [&](const Eigen::VectorXd& step, double scale, double prediction) {
        xmvb::profiling::ScopedPhase trial_phase(xmvb::profiling::Phase::CiahTrial, true);
        if (step_audit)
            diagnostics::audit_orbital_step(
                orbitals, chart, scale * step.head(no), "macro_trial", micro.micro_steps);
        const CiTransportPoint* point = has_ci() ? &transport_at(scale * step) : nullptr;
        const Eigen::VectorXd parameters = point
            ? layout.pack(point->input)
            : build_nonredundant_lifted_trial_parameters(
                  orbitals, chart, layout, scale * step.head(no));
        const Eigen::MatrixXd& seed = point ? point->coefficients : c0;
        accepted->strict_source_energy.reset();
        // Forcing budget for intermediate trials. Both scales vanish toward
        // stationarity. The old Ritz energy and relative residual tests still
        // apply, and prospective terminal trials use the original CI target.
        const double scale_budget = std::min(prediction, gradient_norm);
        const double exploratory =
            adaptive_ci && gradient_norm > options.gradient_tolerance &&
                    scale_budget > options.energy_tolerance
                ? std::sqrt(ci_target) * std::sqrt(scale_budget)
                : 0.0;
        mcscf::diagnostics::ciah_trace(
            "ciah_ci_forcing", "predicted=%.17g,orbital_norm=%.17g,stationarity_target=%.17g",
            prediction, gradient_norm, std::max(ci_target, exploratory));
        auto trial = objective->evaluate_trial_energy_with_structure_guess(
            parameters, seed, exploratory);
        if (exploratory > ci_target && energy - trial.energy <= options.energy_tolerance) {
            mcscf::diagnostics::ciah_trace("ciah_ci_polish", "point=trial");
            objective->refine_trial_ci(trial);
        }
        if (context->structure_solve_accuracy.exploratory_stationarity > ci_target &&
            std::abs(energy - trial.energy) < options.energy_tolerance) {
            if (!strict_source_energy) {
                mcscf::diagnostics::ciah_trace("ciah_ci_polish", "point=source");
                strict_source_energy = objective->strict_accepted_energy();
            }
            accepted->strict_source_energy = strict_source_energy;
        }
        if (step_audit)
            diagnostics::audit_ciah_state(trial.orbital_preparation_input,
                                          seed,
                                          "seed",
                                          std::numeric_limits<double>::quiet_NaN());
        return trial;
    };
    macro_callbacks.energy = [](const Trial& trial) { return trial.energy; };
    macro_callbacks.terminal = [&](Trial& trial) {
        objective->complete_trial(&trial);
        accepted->trial = std::move(trial);
        const bool converged = accepted->check_convergence(
            layout, options, objective->input().complete_active_space,
            accepted->strict_source_energy.value_or(energy));
        trial = std::move(accepted->trial);
        return converged;
    };
    macro_callbacks.observe = [&](const mcscf::CiahMacroObservation& point) {
        const double ratio = point.predicted > 0.0 ? point.actual / point.predicted
                                                   : -std::numeric_limits<double>::infinity();
        mcscf::diagnostics::ciah_trace(
            "ciah_macro_trial",
            "scale=%.17g,linear=%.17g,quadratic=%.17g,current_energy=%.17g,trial_energy=%.17g,"
            "predicted=%.17g,actual=%.17g,rho=%.17g,accepted=%d",
            point.scale,
            point.linear,
            point.quadratic,
            energy,
            point.trial_energy,
            point.predicted,
            point.actual,
            ratio,
            static_cast<int>(point.accepted));
    };
    const auto run_macro = [&](mcscf::CiahMicroResult& path) {
        auto result = mcscf::run_ciah_macro(workspace,
                                            path,
                                            initial_gradient,
                                            energy,
                                            options.energy_tolerance,
                                            options.minimum_step_size,
                                            MacroOps{},
                                            macro_callbacks);
        record->rejected_trial_count += result.rejected_trials;
        return result;
    };
    auto macro = run_macro(micro);
    if (!macro.trial) {
        result->termination_reason = macro.reason;
        return false;
    }
    // Only a gradient stop can be overturned by a change in the model gradient.
    std::optional<mcscf::CiahMicroResult> refinement;
    const bool refine_macro = response && std::string_view(micro.stop_reason) == "micro_gradient" &&
                              macro.scale == 1.0 && !macro.keyframe_backtracked &&
                              !macro.origin_restarted && has_capacity();
    while (refine_macro && has_capacity()) {
        const auto& path = refinement ? *refinement : micro;
        if (std::string_view(path.stop_reason) != "micro_gradient" ||
            path.displacement.head(no).isZero(0.0))
            break;
        const double source_energy = accepted->strict_source_energy.value_or(energy);
        if (!refinement &&
            (accepted->converged_geometry ||
             (std::abs(macro.trial->energy - source_energy) < options.energy_tolerance &&
              macro_callbacks.terminal(*macro.trial)))) {
            mcscf::diagnostics::ciah_trace("ciah_small_ci_tail_terminal", "skipped=1");
            break;
        }

        const auto start = std::chrono::steady_clock::now();
        const Eigen::Index j = response->rank();
        const double target = mcscf::detail::ciah_micro_gradient_stop(options.gradient_tolerance);
        // Use the accumulated joint displacement, with no Ritz shift. The CI
        // image is already available in the retained response columns.
        const int original_probes = forward_actions;
        const Eigen::VectorXd omitted = response->omitted_residual(
            ci_gradient + probe(path.displacement.head(no)), path.displacement.tail(ns), 0.0);
        const double missing = omitted.norm();
        mcscf::diagnostics::ciah_trace("ciah_small_ci_tail", "forward_actions=%d,omitted=%.17g",
                                      forward_actions - original_probes, missing);
        const double before = coordinate_norm(path.predicted_gradient);
        if (missing == 0.0 || !try_column(omitted, 0.0)) {
            mcscf::diagnostics::ciah_trace(
                "ciah_small_ci_feedback", "before=%.17g,target=%.17g,omitted=%.17g,tried=0",
                before, target, missing);
            break;
        }

        // The correction relaxes the old CI coordinates along with the new
        // one. Its orbital image is assembled from the retained B^T Z.
        Eigen::VectorXd feedback = Eigen::VectorXd::Zero(no);
        const bool positive = response->residual(omitted, 0.0, nullptr, &feedback);
        const double after = std::hypot(
            chart.coordinate_dual_norm(path.predicted_gradient.head(no) + feedback),
            path.predicted_gradient.tail(ns).norm());
        if (positive && after < target) {
            response->discard_last_column();
            mcscf::diagnostics::ciah_trace(
                "ciah_small_ci_feedback",
                "before=%.17g,after=%.17g,target=%.17g,omitted=%.17g,tried=1,continued=0",
                before, after, target, missing);
            break;
        }

        install_column();
        initial_gradient[no + j] = response->project(ci_gradient)[j];
        record->model_dimension = static_cast<int>(no + response->rank());
        if (!refinement)
            refinement.emplace(micro);
        auto& refined = *refinement;
        const auto extend_image = [&](const Eigen::VectorXd& step, Eigen::VectorXd& image) {
            image[no + j] = response->hessian_coordinate(j, step.head(no), step.tail(ns));
        };
        extend_image(refined.displacement, refined.hessian_displacement);
        refined.keyframes.extend_image(extend_image);
        const bool reused_keyframe = keyframe_ci_gradient.size() == full_ns;
        if (reused_keyframe) {
            const double gradient_j = response->project(keyframe_ci_gradient)[j];
            refined.energy_model.extend_coordinates(
                [&](const Eigen::VectorXd& step, Eigen::VectorXd& gradient, Eigen::VectorXd& image) {
                    extend_image(step, image);
                    gradient[no + j] = gradient_j;
                    refined.predicted_gradient[no + j] = gradient[no + j] + image[no + j];
                });
        } else {
            auto anchor = callbacks.evaluate_keyframe(refined.displacement);
            refined.keyframes.save(anchor.energy, refined.displacement, refined.hessian_displacement,
                                   refined.last_step, refined.candidate);
            refined.predicted_gradient = std::move(anchor.gradient);
            refined.energy_model.refresh(anchor.energy, refined.predicted_gradient);
        }

        refined = mcscf::continue_ciah_microsteps(workspace,
                                                 initial_gradient,
                                                 energy,
                                                 options.gradient_tolerance,
                                                 callbacks,
                                                 std::move(refined));
        mcscf::diagnostics::ciah_trace(
            "ciah_small_ci_feedback",
            "before=%.17g,after=%.17g,target=%.17g,omitted=%.17g,tried=1,continued=1,"
            "reused_keyframe=%d,positive_ci=%d,columns=%ld,seconds=%.17g",
            before, after, target, missing, reused_keyframe, positive,
            static_cast<long>(response->rank()),
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    }
    if (refinement) {
        // All intermediate candidates live in the frozen AH model. Globalize
        // only the final one, retaining the original physical trial.
        auto& refined = *refinement;
        const auto original_source = accepted->strict_source_energy;
        auto original_convergence = std::move(accepted->converged_geometry);
        accepted->converged_geometry.reset();
        auto alternative = run_macro(refined);
        const bool replaced = alternative.trial &&
                              (accepted->converged_geometry ||
                               alternative.trial->energy < macro.trial->energy);
        mcscf::diagnostics::ciah_trace("ciah_small_ci_macro", "replaced=%d", replaced);
        if (replaced) {
            micro = std::move(refined);
            macro = std::move(alternative);
        } else {
            record->micro_iterations += refined.micro_iterations - micro.micro_iterations;
            record->rejected_trial_count += refined.rejected_trial_count - micro.rejected_trial_count;
            accepted->strict_source_energy = original_source;
            accepted->converged_geometry = std::move(original_convergence);
        }
    }
    mcscf::diagnostics::ciah_trace("ciah_ci_step_limit",
                                  "calls=%d,skipped=%d,carrier_seconds=%.17g",
                                  ci_step_limit_calls, ci_step_limit_skipped, ci_step_limit_seconds);
    mcscf::diagnostics::ciah_trace("ciah_projected_step_limit",
                                  "calls=%d,screened=%d,images=%zu,bytes=%zu",
                                  rotation_limit_samples, rotation_limit_screened,
                                  rotation_images.size(), rotation_limit_memory.bytes());
    record->micro_iterations += micro.micro_iterations;
    record->rejected_trial_count += micro.rejected_trial_count;
    const auto& displacement = micro.displacement;
    const auto& hessian_displacement = micro.hessian_displacement;
    const auto& last_step = micro.last_step;
    const int micro_steps = micro.micro_steps;
    const bool energy_model_stopped = micro.energy_model_stopped;
    accepted->trial = std::move(*macro.trial);
    const double scale = macro.scale, accepted_prediction = macro.predicted_reduction;
    const double last_ritz_norm = macro.residual_norm, last_eigenvalue = macro.eigenvalue;
    const bool last_converged = macro.ritz_converged;
    const bool keyframe_backtracked = macro.keyframe_backtracked,
               origin_restarted = macro.origin_restarted;
    // Only the accepted point needs gradient backpropagation and its context.
    // Complete the retained forward evaluation, rather than rebuilding it.
    xmvb::profiling::ScopedPhase completion_phase(xmvb::profiling::Phase::CiahCompletion, true);
    if (!accepted->trial.gradient_ready)
        objective->complete_trial(&accepted->trial);
    completion_phase.stop();
    if (step_audit) {
        diagnostics::audit_orbital_step(
            orbitals, chart, displacement.head(no), "macro_accepted", micro_steps);
        diagnostics::audit_ciah_state(
            accepted->trial.orbital_preparation_input,
            accepted->trial.gradient_result.second_order_context->selected_state_eigenvectors,
            "relaxed",
            accepted->trial.energy);
    }
    if (mcscf::diagnostics::ciah_event_trace_enabled()) {
        // r = g0 + H0*p is the unshifted frozen Newton residual of the actual
        // accepted displacement. It reuses accumulated images, adding no HVP.
        // It is NOT the last Ritz residual or a nonlinear stationarity certificate.
        const Eigen::VectorXd frozen_gradient = initial_gradient + hessian_displacement;
        mcscf::diagnostics::ciah_trace(
            "ciah_macro_model",
            "source_coordinate_norm=%.17g,frozen_model_coordinate_norm=%.17g,orbital=%.17g,"
            "structure_coordinate_norm=%.17g,last_ritz=%.17g,scale=%.17g",
            coordinate_norm(initial_gradient),
            coordinate_norm(frozen_gradient),
            frozen_gradient.head(no).norm(),
            frozen_gradient.tail(ns).norm(),
            last_ritz_norm,
            scale);
        if (mcscf::diagnostics::ciah_candidate_audit_enabled()) {
            // One additional fixed-C keyframe, only in explicit nonlinear audit mode.
            // The accepted relaxed gradient is the normal production completion,
            // brought forward and cached so the caller does not repeat its backprop.
            const mcscf::CiahKeyframe exact = evaluate_keyframe(displacement);
            if (!accepted->trial.gradient_ready)
                objective->complete_trial(&accepted->trial);
            const Eigen::VectorXd relaxed =
                chart.project_reduced_gradient(accepted->trial.gradient);
            mcscf::diagnostics::ciah_trace(
                "ciah_macro_gradient",
                "source=%.17g,fixed=%.17g,fixed_orbital=%.17g,fixed_structure_coordinate_norm=%."
                "17g,linearization_error=%.17g,relaxed_orbital=%.17g,relaxation_change=%.17g,fixed_"
                "energy=%.17g,relaxed_energy=%.17g",
                coordinate_norm(initial_gradient),
                coordinate_norm(exact.gradient),
                exact.gradient.head(no).norm(),
                exact.gradient.tail(ns).norm(),
                coordinate_norm(exact.gradient - frozen_gradient),
                relaxed.norm(),
                (relaxed - exact.gradient.head(no)).norm(),
                exact.energy,
                accepted->trial.energy);
        }
    }
    accepted->next_orbital_guess_packed = chart.expand_step(last_step.head(no));
    if (response && has_ci())
        accepted->next_structure_guess =
            structure.coefficient_response_coordinates(response->lift(last_step.tail(ns)));
    else if (!response && ns)
        accepted->next_structure_guess =
            structure.coefficient_response_coordinates(last_step.tail(ns));
    if (has_ci()) {
        const Eigen::MatrixXd& reference = transport_at(displacement).coefficients;
        const auto& next = *accepted->trial.gradient_result.second_order_context;
        accepted->align_structure_frame(reference,
                                        next.selected_state_structure_images.value().overlap);
    }
    record->gradient_dot_step = initial_gradient.dot(displacement);
    record->step_dot_hessian_step = displacement.dot(hessian_displacement);
    record->predicted_reduction = accepted_prediction;
    record->actual_reduction = energy - accepted->trial.energy;
    record->orbital_step_norm = metric.norm(displacement.head(no));
    record->accepted_trial_radius = record->orbital_step_norm;
    record->next_trust_radius = *trust_radius;
    record->kkt_residual_norm = last_ritz_norm;
    record->kkt_residual_target = ciah_options.tolerance;
    record->shift = -last_eigenvalue;
    // This is the last microstep Ritz residual, or the unscaled origin-root
    // residual after restart. Neither certifies the accepted nonlinear step.
    record->approximate_step = !last_converged || scale < 1.0 || keyframe_backtracked ||
                               origin_restarted || energy_model_stopped;
    return true;
}

} // namespace xmvb::vb::optimizer_detail
