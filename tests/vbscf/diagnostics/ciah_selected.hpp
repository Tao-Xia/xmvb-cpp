#pragma once

#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <iostream>
#include <string>
#include <vector>

#include "ciah_internal.hpp"
#include "ciah_restricted.hpp"
#include "ciah_state_metric.hpp"
#include "vbscf/optimization/coupled/ci_transport.hpp"

namespace xmvb::test {

/** @brief Resolve selected-space residual curvature and CI compensation.
 * Only the tiny, fixed-active-span reference in audit_joint_geometry calls this.
 * Compound determinants provide independent finite wavefunctions; the molecular
 * Hessian is compared against a normalized Rayleigh derivative, not another AO
 * HVP. No optimizer, rank cutoff or convergence condition is changed.
 */
template <class Gradient>
void audit_selected_curvature(const vb::OrbitalPreparationInput& input,
                              const vb::OrbitalPreparationResult& prepared,
                              const vb::AcceptedPointContext& context,
                              const vb::SparseParameterLayout& layout,
                              const vb::OrbitalChart& chart,
                              const vb::StructureTangentOperator& structure,
                              const vb::StructureAction& action,
                              const Eigen::MatrixXd& coefficients,
                              const Eigen::MatrixXd& state,
                              const Eigen::MatrixXd& basis,
                              const Eigen::MatrixXd& jacobian,
                              const Eigen::MatrixXd& mass,
                              const Eigen::MatrixXd& curvature,
                              const InternalStateCurvature& reference,
                              Gradient&& keyframe_gradient) {
    const auto view = action.orthogonal_view();
    const Eigen::Index no = chart.reduced_size(), ns = structure.tangent_size();
    const Eigen::Index internal = basis.cols() - ns, na = input.n_active_orbitals;
    const Eigen::Index nc = (input.n_total_electrons - input.n_active_electrons) / 2;
    const Eigen::MatrixXd sigma = view.sigma.apply(state);
    const double energy = (state.array() * sigma.array()).sum();
    const Eigen::MatrixXd residual = sigma - energy * state;
    const Eigen::Map<const Eigen::VectorXd> flat_residual(residual.data(), residual.size());
    Eigen::MatrixXd selected(state.size(), coefficients.rows());
    for (Eigen::Index j = 0; j < selected.cols(); ++j) {
        const Eigen::MatrixXd column = action.orthogonalize_structure_block(
            Eigen::MatrixXd::Identity(coefficients.rows(), coefficients.rows()).col(j));
        selected.col(j) = Eigen::Map<const Eigen::VectorXd>(column.data(), column.size());
    }
    const Eigen::HouseholderQR<Eigen::MatrixXd> qr(selected);
    const Eigen::MatrixXd orthogonal =
        qr.householderQ() * Eigen::MatrixXd::Identity(selected.rows(), selected.cols());
    const Eigen::VectorXd inside = orthogonal.transpose() * flat_residual;
    std::cout << "SELECTED_RESIDUAL,total=" << residual.norm() << ",inside=" << inside.norm()
              << ",outside=" << (flat_residual - orthogonal * inside).norm()
              << ",hvp_error=" << (curvature - reference.hessian).norm()
              << ",hvp_without_residual_error=" << (curvature - reference.tangent_curvature).norm()
              << ",tangent_curvature=" << reference.tangent_curvature.norm()
              << ",residual_curvature=" << reference.residual_curvature.norm() << std::endl;

    const Eigen::MatrixXd jo = jacobian.leftCols(internal), jc = jacobian.rightCols(ns);
    const Eigen::HouseholderQR<Eigen::MatrixXd> ci_qr(jc);
    // Solve for the exact least-squares CI compensation. Do not form Jc^T Jc.
    const Eigen::MatrixXd compensation = -ci_qr.solve(jo);
    const Eigen::MatrixXd horizontal = jo + jc * compensation;
    const Eigen::MatrixXd physical = 2 * jacobian.transpose() * jacobian;
    const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> physical_scale(physical, mass);
    const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> low(curvature, mass);
    const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> compensate(
        (jo - horizontal).transpose() * (jo - horizontal), jo.transpose() * jo);
    if (physical_scale.info() != Eigen::Success || low.info() != Eigen::Success ||
        compensate.info() != Eigen::Success)
        throw std::runtime_error("selected curvature reference spectrum failed");
    std::cout << "SELECTED_GEOMETRY,physical_to_model_min=" << physical_scale.eigenvalues()[0]
              << ",physical_to_model_max=" << physical_scale.eigenvalues().tail(1)[0]
              << ",largest_compensated_fraction=" << compensate.eigenvalues().tail(1)[0]
              << ",horizontal_ci_overlap=" << (jc.transpose() * horizontal).norm() << std::endl;

    std::vector<std::pair<std::string, Eigen::VectorXd>> directions;
    directions.emplace_back("negative", low.eigenvectors().col(0));
    directions.emplace_back("soft", physical_scale.eigenvectors().col(0));
    Eigen::VectorXd cancelled = Eigen::VectorXd::Zero(basis.cols());
    cancelled.head(internal) = compensate.eigenvectors().rightCols(1);
    directions.emplace_back("orbital", cancelled);
    cancelled.tail(ns) = compensation * cancelled.head(internal);
    directions.emplace_back("compensated", cancelled);
    Eigen::VectorXd mixed(basis.cols());
    for (Eigen::Index i = 0; i < mixed.size(); ++i)
        mixed[i] = std::sin(i + 1.0);
    directions.emplace_back("mixed", mixed);

    const Eigen::MatrixXd r = view.alpha.orbital_transform();
    const Eigen::MatrixXd inverse =
        r.triangularView<Eigen::Upper>().solve(Eigen::MatrixXd::Identity(na, na));
    const Eigen::MatrixXd active = prepared.auxiliary_orbital_matrix.middleCols(nc, na);
    const Eigen::MatrixXd cross = (active * inverse).transpose() * input.ao_overlap_matrix;
    const vb::CiTransport transport(input, layout, chart, context, action, structure);
    const auto value = [&](const Eigen::VectorXd& full, double step, bool transported) {
        const auto point = transport.retract(step * full.head(no), step * full.tail(ns));
        const Eigen::MatrixXd transform =
            Eigen::MatrixXd::Identity(na, na) +
            cross * (point.prepared.auxiliary_orbital_matrix.middleCols(nc, na) - active) * inverse;
        const Eigen::MatrixXd c = transported ? point.coefficients : point.base_coefficients;
        const Eigen::MatrixXd origin = action.orthogonalize_structure_block(c);
        Eigen::MatrixXd y = reference_compound(view.alpha.strings(), transform) * origin *
                            reference_compound(view.beta.strings(), transform).transpose();
        y /= y.norm();
        if ((y.array() * state.array()).sum() < 0)
            y = -y;
        return y;
    };
    const auto dot = [](const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
        return (a.array() * b.array()).sum();
    };
    const auto energy_change = [&](const Eigen::MatrixXd& y) {
        const Eigen::MatrixXd difference = y - state;
        return 2 * dot(difference, residual) +
               dot(difference, view.sigma.apply(difference) - energy * difference);
    };
    for (auto& [label, v] : directions) {
        v /= std::sqrt(v.dot(mass * v));
        const Eigen::VectorXd full = basis * v;
        const Eigen::VectorXd dy = jacobian * v;
        const Eigen::VectorXd orbital = jo * v.head(internal), ci = jc * v.tail(ns);
        const double maximum =
            std::max({1.0,
                      chart.maximum_rotation_component(full.head(no)),
                      structure.maximum_coefficient_component_coordinates(full.tail(ns))});
        std::cout << "SELECTED_DIRECTION,name=" << label << ",orbital=" << orbital.norm()
                  << ",ci=" << ci.norm() << ",total=" << dy.norm()
                  << ",cross=" << 2 * orbital.dot(ci) << ",curvature=" << v.dot(curvature * v)
                  << ",tangent_curvature=" << v.dot(reference.tangent_curvature * v)
                  << ",residual_curvature=" << v.dot(reference.residual_curvature * v)
                  << ",gradient=" << reference.gradient.dot(v) << std::endl;
        for (double mesh : {3e-3, 1e-3, 3e-4}) {
            const double delta = mesh / maximum;
            const Eigen::MatrixXd plus = value(full, delta, true),
                                  minus = value(full, -delta, true);
            const Eigen::MatrixXd affine_plus = value(full, delta, false),
                                  affine_minus = value(full, -delta, false);
            const Eigen::MatrixXd tangent = (plus - minus) / (2 * delta);
            const Eigen::MatrixXd acceleration = (plus - 2 * state + minus) / (delta * delta);
            const Eigen::MatrixXd affine_acceleration =
                (affine_plus - 2 * state + affine_minus) / (delta * delta);
            const Eigen::MatrixXd difference = acceleration - affine_acceleration;
            const Eigen::Map<const Eigen::VectorXd> flat_difference(difference.data(),
                                                                    difference.size());
            const Eigen::VectorXd projection = orthogonal.transpose() * flat_difference;
            const Eigen::VectorXd finite =
                basis.transpose() *
                (keyframe_gradient(delta * full) - keyframe_gradient(-delta * full)) / (2 * delta);
            const double tangent_term =
                2 * dot(tangent, view.sigma.apply(tangent) - energy * tangent);
            const double residual_term = 2 * dot(residual, acceleration);
            std::cout
                << "SELECTED_FD,name=" << label << ",mesh=" << mesh << ",tangent_error="
                << (Eigen::Map<const Eigen::VectorXd>(tangent.data(), tangent.size()) - dy).norm()
                << ",tangent_curvature=" << tangent_term << ",residual_curvature=" << residual_term
                << ",total_curvature_error=" << tangent_term + residual_term - v.dot(curvature * v)
                << ",hvp_error=" << (finite - curvature * v).norm()
                << ",transport_acceleration=" << difference.norm()
                << ",transport_acceleration_outside="
                << (flat_difference - orthogonal * projection).norm()
                << ",transport_residual_curvature=" << 2 * dot(residual, difference) << std::endl;
        }
        // The same finite direction and component scale, with and without
        // transport. This does not change the molecular optimizer's policy.
        const double delta = .03 / maximum;
        const Eigen::MatrixXd transported = value(full, delta, true);
        const Eigen::MatrixXd affine = value(full, delta, false);
        std::cout << "SELECTED_FINITE,name=" << label << ",prediction="
                  << delta * reference.gradient.dot(v) + .5 * delta * delta * v.dot(curvature * v)
                  << ",transported_energy=" << energy_change(transported)
                  << ",affine_energy=" << energy_change(affine)
                  << ",state_change=" << (transported - state).norm()
                  << ",tangent_prediction=" << delta * dy.norm()
                  << ",transport_change=" << (transported - affine).norm() << std::endl;
    }
    const auto maximum = [&](const Eigen::VectorXd& v) {
        const Eigen::VectorXd full = basis * v;
        return std::max(chart.maximum_rotation_component(full.head(no)),
                        structure.maximum_coefficient_component_coordinates(full.tail(ns)));
    };
    // Identical local quadratic, sampled directions and component cap. Vary
    // only the metric to measure whether it explains the local step's quality.
    // A better oracle step alone does not qualify a molecular algorithm change.
    for (const auto& [label, trial_metric] :
         {std::pair{"ray", mass}, std::pair{"wavefunction", physical}}) {
        const Eigen::VectorXd v = restricted_quadratic_reference(
            curvature, trial_metric, reference.gradient, maximum, .03);
        const Eigen::VectorXd full = basis * v;
        const Eigen::MatrixXd transported = value(full, 1, true);
        std::cout << "SELECTED_MODEL,metric=" << label
                  << ",prediction=" << reference.gradient.dot(v) + .5 * v.dot(curvature * v)
                  << ",energy_change=" << energy_change(transported) << ",maximum=" << maximum(v)
                  << ",state_change=" << (transported - state).norm()
                  << ",tangent_prediction=" << (jacobian * v).norm()
                  << ",model_length=" << std::sqrt(v.dot(mass * v))
                  << ",physical_length=" << std::sqrt(v.dot(physical * v)) << std::endl;
        // Probe the finite energy curve on this SAME ray, without accepting
        // steps or changing any solver bound. This distinguishes geometric
        // nonlinearity from an independently binding component limit.
        for (double scale : {2., 4., 8.}) {
            const Eigen::MatrixXd y = value(full, scale, true);
            std::cout << "SELECTED_RAY,metric=" << label << ",scale=" << scale << ",prediction="
                      << scale * reference.gradient.dot(v) +
                             .5 * scale * scale * v.dot(curvature * v)
                      << ",energy_change=" << energy_change(y) << ",maximum=" << scale * maximum(v)
                      << ",state_change=" << (y - state).norm()
                      << ",tangent_prediction=" << scale * (jacobian * v).norm() << std::endl;
        }
    }
}
} // namespace xmvb::test
