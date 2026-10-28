#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <stdexcept>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/LU>
#include <Eigen/SVD>

#include "ciah_internal.hpp"
#include "ciah_state_metric.hpp"
#include "ciah_precision.hpp"
#include "ciah_restricted.hpp"
#include "mcscf/ci/sigma.hpp"
#include "vbscf/derivatives/hessian/coupled/structure.hpp"
#include "vbscf/derivatives/hessian/responses/orbital/preparation.hpp"
#include "vbscf/optimization/trust_region/retraction.hpp"
#include "vbscf/orbitals/preparation/preparer.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"

namespace xmvb::test {

/** @brief Compare orbital-ray and normalized wavefunction metrics in an internal+CI span.
 * All states are represented in one immutable orthogonal active reference.
 * Finite states use general compound matrices; the analytic tangent instead
 * uses the direct-CI one-body generator. No selected-space closure is assumed.
 */
template <class Action, class Gradient, class Energy>
void audit_joint_geometry(const vb::OrbitalPreparationInput& input,
                          const vb::OrbitalPreparationResult& prepared,
                          const vb::AcceptedPointContext& context,
                          const vb::SparseParameterLayout& layout,
                          const vb::OrbitalChart& chart,
                          const vb::StructureTangentOperator& structure,
                          const vb::StructureAction& action,
                          const Eigen::MatrixXd& coefficients,
                          const vb::NonredundantRetractionMetric& metric,
                          const Eigen::VectorXd& gradient,
                          Action&& hessian,
                          Gradient&& keyframe_gradient,
                          Energy&& keyframe_energy) {
    const auto view = action.orthogonal_view();
    const Eigen::Index na = input.n_active_orbitals;
    const Eigen::Index nc = (input.n_total_electrons - input.n_active_electrons) / 2;
    const Eigen::Index no = chart.reduced_size(), ns = structure.tangent_size();
    const Eigen::Index internal = na * (na - 1), columns = internal + ns;
    if (na > 8 || coefficients.rows() > 8 || coefficients.cols() != 1 ||
        view.alpha.strings().size() > 64 || view.beta.strings().size() > 64)
        throw std::length_error("joint geometry reference is restricted to tiny selected spaces");
    const Eigen::MatrixXd r = view.alpha.orbital_transform();
    const Eigen::MatrixXd inverse =
        r.triangularView<Eigen::Upper>().solve(Eigen::MatrixXd::Identity(na, na));
    const Eigen::MatrixXd active = prepared.auxiliary_orbital_matrix.middleCols(nc, na);
    const Eigen::MatrixXd chi = active * inverse;
    const Eigen::MatrixXd cross = chi.transpose() * input.ao_overlap_matrix;
    Eigen::MatrixXd state = action.orthogonalize_structure_block(coefficients);
    const double state_norm = state.stableNorm();
    if (!(state_norm > 0))
        throw std::runtime_error("joint geometry reference has a zero state");
    state /= state_norm;
    Eigen::MatrixXd sources(no, internal);
    Eigen::Index column = 0;
    const auto& physical = prepared.physical_orbital_frame.normalized_orbital_matrix;
    for (Eigen::Index i = 0; i < na; ++i)
        for (Eigen::Index j = 0; j < na; ++j) {
            if (i == j)
                continue;
            std::vector<double> full(input.orbital_value_table.size(), 0.0);
            for (int k = 0; k < stored_sparse_orbital_coefficient_count(input, nc + i); ++k) {
                const auto slot = (nc + i) * input.n_basis_functions + k;
                full[slot] = physical(input.orbital_basis_index_table[slot] - 1, nc + j);
            }
            sources.col(column++) =
                chart.project_vector(layout.gather_from_full(full)).reduced_gradient;
        }
    const Eigen::HouseholderQR<Eigen::MatrixXd> qr(sources);
    Eigen::MatrixXd basis = Eigen::MatrixXd::Zero(no + ns, columns);
    basis.topLeftCorner(no, internal) = qr.householderQ() * Eigen::MatrixXd::Identity(no, internal);
    basis.bottomRightCorner(ns, ns).setIdentity();
    const auto geometry =
        vb::build_accepted_orbital_preparation_cache(input, Eigen::MatrixXd{}, {});
    Eigen::MatrixXd jacobian(state.size(), columns), images(no + ns, columns),
        mass_images(no + ns, columns);
    const auto finite_state = [&](const Eigen::VectorXd& p, const Eigen::MatrixXd& ci, double t) {
        const auto point = chart.retract_step(input, t * p);
        const auto prep = vb::ActiveSpaceOrbitalPreparer{}.prepare(point);
        // I + delta Phi R^-1 avoids an ill-conditioned zero-step roundtrip.
        const Eigen::MatrixXd transform =
            Eigen::MatrixXd::Identity(na, na) +
            cross * (prep.auxiliary_orbital_matrix.middleCols(nc, na) - active) * inverse;
        const Eigen::MatrixXd left = reference_compound(view.alpha.strings(), transform);
        const Eigen::MatrixXd right = reference_compound(view.beta.strings(), transform);
        Eigen::MatrixXd value = left * (state + t * ci) * right.transpose();
        value /= value.stableNorm();
        if ((state.array() * value.array()).sum() < 0)
            value = -value;
        return value;
    };
    std::vector<InternalStateDirection> internal_jets;
    double fd_maximum = 0;
    for (Eigen::Index j = 0; j < columns; ++j) {
        const Eigen::VectorXd p = basis.col(j).head(no), q = basis.col(j).tail(ns);
        const auto raw = vb::gather_nonredundant_retract_tangent(input, chart, layout, p);
        const auto tangent = vb::build_dense_orbital_tangent_context(input, layout, raw, geometry);
        const auto direction = vb::build_orbital_preparation_directional_result(
            input, tangent, nc, na, geometry, false);
        const Eigen::MatrixXd k = cross * direction.delta_active_auxiliary_orbitals * inverse;
        const Eigen::MatrixXd ci =
            action.orthogonalize_structure_block(structure.coefficient_response_coordinates(q)) /
            state_norm;
        const Eigen::MatrixXd orbital = view.sigma.apply_one_body_generator(state, k);
        internal_jets.push_back({k,
                                 orbital,
                                 ci,
                                 tangent.delta_normalized_orbitals.middleCols(nc, na),
                                 Eigen::Map<const Eigen::VectorXd>(
                                     tangent.normalization_direction_projections.data() + nc, na)});
        Eigen::MatrixXd dy = orbital + ci;
        dy -= state * (state.array() * dy.array()).sum();
        jacobian.col(j) = Eigen::Map<const Eigen::VectorXd>(dy.data(), dy.size());
        images.col(j) = hessian(basis.col(j));
        mass_images.col(j).head(no) = metric.apply(p);
        mass_images.col(j).tail(ns) = structure.apply_metric_coordinates(q);
        const double maximum = std::max({1.0,
                                         chart.maximum_rotation_component(p),
                                         structure.maximum_coefficient_component_coordinates(q)});
        // Mesh parameters belong to a finite-difference experiment, not the solver.
        for (double mesh : {1e-3, 3e-4}) {
            const double delta = mesh / maximum;
            const Eigen::MatrixXd finite =
                (finite_state(p, ci, delta) - finite_state(p, ci, -delta)) / (2 * delta);
            const double error = (finite - dy).norm() / std::max(1.0, dy.norm());
            fd_maximum = std::max(fd_maximum, error);
            std::cout << "JOINT_JET,column=" << j << ",mesh=" << mesh << ",tangent=" << dy.norm()
                      << ",relative_error=" << error << std::endl;
        }
    }
    const Eigen::MatrixXd mass =
        .5 * (basis.transpose() * mass_images + mass_images.transpose() * basis);
    const Eigen::MatrixXd curvature =
        .5 * (basis.transpose() * images + images.transpose() * basis);
    const Eigen::MatrixXd quantum = 2 * jacobian.transpose() * jacobian;
    Eigen::MatrixXd inverse_model(internal, internal);
    for (Eigen::Index j = 0; j < internal; ++j)
        inverse_model.col(j) =
            basis.topLeftCorner(no, internal).transpose() *
            chart.apply_inverse_augmented_hessian_diagonal(basis.col(j).head(no), 0);
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> inverse_spectrum(
        (.5 * (inverse_model + inverse_model.transpose())).eval());
    if (inverse_spectrum.info() != Eigen::Success)
        throw std::runtime_error("internal inverse model witness spectrum failed");
    Eigen::Index witness = -1;
    for (Eigen::Index j = 0; j < internal; ++j)
        if (inverse_spectrum.eigenvalues()[j] < 0)
            witness = j;
    std::cout << "PRECONDITIONER_WITNESS,negative=" << (witness >= 0)
              << ",inverse_min=" << inverse_spectrum.eigenvalues()[0]
              << ",inverse_max=" << inverse_spectrum.eigenvalues().tail(1)[0];
    if (witness >= 0) {
        Eigen::VectorXd v = Eigen::VectorXd::Zero(columns);
        v.head(internal) = inverse_spectrum.eigenvectors().col(witness);
        std::cout << ",predicted_curvature=" << 1.0 / inverse_spectrum.eigenvalues()[witness]
                  << ",physical_curvature=" << v.dot(curvature * v) / v.dot(mass * v)
                  << ",gradient=" << (basis.transpose() * gradient).dot(v);
    }
    std::cout << std::endl;

    const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> relative(quantum, mass);
    if (relative.info() != Eigen::Success)
        throw std::runtime_error("joint versus ray metric eigensystem failed");
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian, Eigen::ComputeThinV);
    const auto& singular = svd.singularValues();
    const double floor = std::numeric_limits<double>::epsilon() *
                         std::max(jacobian.rows(), jacobian.cols()) * singular[0];
    const Eigen::Index rank = (singular.array() > floor).count();
    std::cout << "JOINT_METRIC,columns=" << columns << ",rank=" << rank
              << ",singular_min=" << singular.tail(1)[0] << ",singular_max=" << singular[0]
              << ",relative_min=" << relative.eigenvalues()[0]
              << ",relative_max=" << relative.eigenvalues().tail(1)[0]
              << ",mixed_norm=" << quantum.topRightCorner(internal, ns).norm()
              << ",orbital_norm=" << quantum.topLeftCorner(internal, internal).norm()
              << ",ci_norm=" << quantum.bottomRightCorner(ns, ns).norm()
              << ",maximum_fd_error=" << fd_maximum << std::endl;
    const Eigen::MatrixXd ci_error =
        quantum.bottomRightCorner(ns, ns) - mass.bottomRightCorner(ns, ns);
    std::cout << "JOINT_CI_METRIC,relative_error="
              << ci_error.norm() / std::max(1.0, mass.bottomRightCorner(ns, ns).norm())
              << std::endl;
    const Eigen::MatrixXd sigma = view.sigma.apply(state, view.exchange_parity);
    const Eigen::VectorXd projected_gradient = basis.transpose() * gradient;
    audit_internal_precision(input,
                             prepared,
                             context,
                             action,
                             structure,
                             coefficients,
                             state,
                             basis,
                             internal_jets,
                             projected_gradient);
    const auto active_curvature = internal_state_curvature(
        view.sigma, state, sigma, r, inverse, input.ao_overlap_matrix, internal_jets);
    std::cout << "JOINT_ACTIVE_CURVATURE,error=" << (active_curvature.hessian - curvature).norm()
              << ",reference_norm=" << curvature.norm()
              << ",gradient_error=" << (active_curvature.gradient - projected_gradient).norm()
              << ",gradient_norm=" << projected_gradient.norm() << std::endl;

    for (Eigen::Index mode = columns - 1; mode >= std::max<Eigen::Index>(0, columns - 3); --mode) {
        const Eigen::VectorXd v = svd.matrixV().col(mode);
        const Eigen::VectorXd full = basis * v;
        const Eigen::VectorXd dy = jacobian * v;
        const Eigen::VectorXd p = full.head(no), q = full.tail(ns);
        const Eigen::MatrixXd ci =
            action.orthogonalize_structure_block(structure.coefficient_response_coordinates(q)) /
            state_norm;
        const double maximum = std::max({1.0,
                                         chart.maximum_rotation_component(p),
                                         structure.maximum_coefficient_component_coordinates(q)});
        const Eigen::VectorXd direct_image = hessian(full);
        const Eigen::VectorXd cached_image = images * v;
        std::cout << "JOINT_SOFT_LINEARITY,index=" << mode << ",direct_norm=" << direct_image.norm()
                  << ",cached_norm=" << cached_image.norm()
                  << ",difference=" << (direct_image - cached_image).norm() << std::endl;
        // Soft coordinate vectors need a larger FD mesh to resolve the gradient
        // change above subtraction noise. The actual frame displacement is
        // still measured by the unchanged chart, not guessed from raw norm.
        for (double mesh : {.3, .1, .03, .01}) {
            const double delta = mesh / maximum;
            const Eigen::VectorXd finite =
                (keyframe_gradient(delta * full) - keyframe_gradient(-delta * full)) / (2 * delta);
            std::cout << "JOINT_SOFT_HVP_FD,index=" << mode << ",mesh=" << mesh
                      << ",finite_norm=" << finite.norm()
                      << ",direct_error=" << (finite - direct_image).norm()
                      << ",cached_error=" << (finite - cached_image).norm()
                      << ",curvature=" << full.dot(finite) << std::endl;
        }
        const double state_gradient =
            2 * Eigen::Map<const Eigen::VectorXd>(sigma.data(), sigma.size()).dot(dy);
        std::cout << "JOINT_SOFT_MODE,index=" << mode << ",singular=" << singular[mode]
                  << ",gradient=" << projected_gradient.dot(v)
                  << ",state_gradient=" << state_gradient << ",curvature=" << v.dot(curvature * v)
                  << ",ray_norm=" << std::sqrt(v.dot(mass * v))
                  << ",hessian_image=" << (images * v).norm() << std::endl;
        for (double mesh : {1e-3, 3e-4}) {
            const double delta = mesh / maximum;
            const Eigen::MatrixXd finite =
                (finite_state(p, ci, delta) - finite_state(p, ci, -delta)) / (2 * delta);
            const Eigen::Map<const Eigen::VectorXd> flat(finite.data(), finite.size());
            std::cout << "JOINT_SOFT_FD,index=" << mode << ",mesh=" << mesh
                      << ",finite_norm=" << finite.norm() << ",error=" << (flat - dy).norm()
                      << ",state_gradient=" << 2 * (sigma.array() * finite.array()).sum()
                      << std::endl;
        }
    }
    const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ray_hessian(curvature, mass);
    if (ray_hessian.info() != Eigen::Success)
        throw std::runtime_error("ray-metric curvature spectrum failed");
    for (Eigen::Index mode = 0; mode < std::min<Eigen::Index>(3, columns); ++mode) {
        const auto v = ray_hessian.eigenvectors().col(mode);
        std::cout << "JOINT_LOW_MODE,index=" << mode
                  << ",curvature=" << ray_hessian.eigenvalues()[mode]
                  << ",gradient=" << projected_gradient.dot(v)
                  << ",orbital_fraction=" << v.head(internal).norm() / v.norm()
                  << ",wavefunction_norm=" << (jacobian * v).norm() << std::endl;
    }
    std::cout << "JOINT_CURVATURE,metric=ray,minimum=" << ray_hessian.eigenvalues()[0]
              << ",maximum=" << ray_hessian.eigenvalues().tail(1)[0] << std::endl;
    if (rank == columns) {
        const Eigen::MatrixXd whitening =
            svd.matrixV() * (singular.array().inverse() / std::sqrt(2.0)).matrix().asDiagonal();
        const Eigen::MatrixXd h = whitening.transpose() * curvature * whitening;
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(.5 * (h + h.transpose()));
        if (spectrum.info() != Eigen::Success)
            throw std::runtime_error("wavefunction-metric curvature spectrum failed");
        std::cout << "JOINT_CURVATURE,metric=wavefunction,minimum=" << spectrum.eigenvalues()[0]
                  << ",maximum=" << spectrum.eigenvalues().tail(1)[0] << std::endl;
    }
    // Include the actual external gradient direction, without pretending its
    // wavefunction tangent belongs to the internal-only Jacobian above.
    Eigen::VectorXd extra = gradient;
    extra.head(no) = chart.inverse_coordinate_metric(gradient.head(no));
    extra -= basis * (basis.transpose() * extra);
    const double external_norm = extra.norm();
    if (external_norm > 0) {
        extra /= external_norm;
        Eigen::MatrixXd trial_basis(no + ns, columns + 1);
        trial_basis.leftCols(columns) = basis;
        trial_basis.col(columns) = extra;
        Eigen::MatrixXd hi(no + ns, columns + 1), mi(no + ns, columns + 1);
        hi.leftCols(columns) = images;
        hi.col(columns) = hessian(extra);
        mi.leftCols(columns) = mass_images;
        mi.col(columns).head(no) = metric.apply(extra.head(no));
        mi.col(columns).tail(ns) = structure.apply_metric_coordinates(extra.tail(ns));
        const Eigen::MatrixXd th =
            .5 * (trial_basis.transpose() * hi + hi.transpose() * trial_basis);
        const Eigen::MatrixXd tm =
            .5 * (trial_basis.transpose() * mi + mi.transpose() * trial_basis);
        const Eigen::VectorXd tg = trial_basis.transpose() * gradient;
        const auto maximum = [&](const Eigen::VectorXd& p) {
            const Eigen::VectorXd full = trial_basis * p;
            return std::max(chart.maximum_rotation_component(full.head(no)),
                            structure.maximum_coefficient_component_coordinates(full.tail(ns)));
        };
        // Exact wavefunction metric on precisely the same sampled tangent span.
        StateMetricReference joint(
            input, prepared, layout, chart, structure, action, state, state_norm);
        std::vector<StateMetricReference::Feature> features;
        for (Eigen::Index j = 0; j < trial_basis.cols(); ++j)
            features.push_back(joint.feature(trial_basis.col(j)));
        Eigen::MatrixXd qm(columns + 1, columns + 1);
        for (Eigen::Index j = 0; j < qm.cols(); ++j)
            for (Eigen::Index i = 0; i < qm.rows(); ++i)
                qm(i, j) = joint.inner(features[i], features[j]);
        std::cout << "JOINT_EXTERNAL_METRIC,internal_error="
                  << (qm.topLeftCorner(columns, columns) - quantum).norm()
                  << ",external_norm=" << qm(columns, columns)
                  << ",mixed_norm=" << qm.topRightCorner(columns, 1).norm() << std::endl;
        for (Eigen::Index sample = 0; sample < 4; ++sample) {
            Eigen::VectorXd x(columns + 1);
            for (Eigen::Index j = 0; j < x.size(); ++j)
                x[j] = std::sin((sample + 1.) * (j + 1.));
            x /= maximum(x);
            const Eigen::VectorXd direction = trial_basis * x;
            const double analytic = .5 * x.dot(qm * x);
            for (double mesh : {.003, .001}) {
                const auto plus = joint.point(direction, mesh),
                           minus = joint.point(direction, -mesh);
                const double finite = (1 - joint.overlap(minus, plus)) / (2 * mesh * mesh);
                std::cout << "JOINT_EXTERNAL_FD,sample=" << sample << ",mesh=" << mesh
                          << ",analytic=" << analytic << ",finite=" << finite << ",relative_error="
                          << std::abs(finite - analytic) / std::max(1., std::abs(analytic))
                          << std::endl;
            }
        }
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> state_metric(qm);
        const double roundoff = std::numeric_limits<double>::epsilon() * (columns + 1) *
                                state_metric.eigenvalues().tail(1)[0];
        std::cout << "JOINT_EXTERNAL_RANK,minimum=" << state_metric.eigenvalues()[0]
                  << ",maximum=" << state_metric.eigenvalues().tail(1)[0]
                  << ",roundoff=" << roundoff << std::endl;
        if (state_metric.info() == Eigen::Success && state_metric.eigenvalues()[0] > roundoff) {
            const double e0 = keyframe_energy(Eigen::VectorXd::Zero(no + ns));
            const Eigen::VectorXd p = restricted_quadratic_reference(th, qm, tg, maximum, .03);
            std::cout << "RESTRICTED_TRIAL,type=wavefunction,sampled_dimension=" << columns + 1
                      << ",max_component=" << maximum(p)
                      << ",prediction=" << tg.dot(p) + .5 * p.dot(th * p)
                      << ",energy_change=" << keyframe_energy(trial_basis * p) - e0
                      << ",physical_length=" << std::sqrt(p.dot(qm * p)) << std::endl;
        }
        const Eigen::VectorXd clipped = clipped_ah_reference(th, tm, tg, maximum, .03);
        const Eigen::VectorXd restricted = restricted_quadratic_reference(th, tm, tg, maximum, .03);
        const double origin_energy = keyframe_energy(Eigen::VectorXd::Zero(no + ns));
        for (const auto& [label, p] :
             {std::pair{"clipped", clipped}, std::pair{"restricted", restricted}}) {
            const Eigen::VectorXd full = trial_basis * p;
            std::cout << "RESTRICTED_TRIAL,type=" << label << ",sampled_dimension=" << columns + 1
                      << ",max_component=" << maximum(p)
                      << ",prediction=" << tg.dot(p) + .5 * p.dot(th * p)
                      << ",energy_change=" << keyframe_energy(full) - origin_energy
                      << ",physical_length=" << std::sqrt(p.dot(tm * p)) << std::endl;
        }
    }
}
} // namespace xmvb::test
