#include "ciah_path.hpp"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "ciah_response.hpp"
#include "ciah_geometry.hpp"
#include "core/memory.hpp"
#include "linalg/eigensolver.hpp"
#include "mcscf/optimization/preconditioners/coarse.hpp"
#include "vbscf/optimization/preconditioners/hessian_diagonal.hpp"
#include "vbscf/diagnostics/orbitals/step_geometry.hpp"
#include "input/loading/loader.hpp"
#include "vbscf/derivatives/gradient/orbital/evaluator.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/derivatives/hessian/coupled/structure.hpp"
#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "vbscf/optimization/coupled/affine.hpp"
#include "vbscf/optimization/coupled/ci_transport.hpp"
#include "vbscf/optimization/objective/function.hpp"
#include "vbscf/optimization/trust_region/retraction.hpp"
#include "vbscf/orbitals/gauge/support_preserving.hpp"

namespace {
using namespace xmvb::vb;
struct ModelValue {
    CiTransportModel model;
    double physical_energy, ci_residual;
    double frame_change = 0, span_change = 0;
};
struct Direction {
    std::string name;
    Eigen::VectorXd value;
};
} // namespace

/** @brief Fixed-reference production derivative checks; never runs an optimizer.
 * Finite-difference meshes are diagnostic displacements, not solver tolerances.
 * Every perturbed gradient is evaluated at its prescribed CI coefficients,
 * then pulled back to the original chart through the production CI transport.
 */
int main(int argc, char** argv) {
    try {
        if (argc != 2 &&
            !(argc == 3 &&
              (std::string_view(argv[2]).starts_with("--path=") ||
               std::string(argv[2]) == "--refine" || std::string(argv[2]) == "--identity" ||
               std::string(argv[2]) == "--repeat" || std::string(argv[2]) == "--soft-mesh" ||
               std::string(argv[2]) == "--dense-ah" || std::string(argv[2]) == "--joint-geometry" ||
               std::string(argv[2]) == "--selected-curvature")))
            throw std::runtime_error("usage: check_ciah_derivatives INPUT.xmi "
                                     "[--refine|--identity|--repeat|--soft-mesh|--dense-ah|--joint-"
                                     "geometry|--selected-curvature|--path=DIR]");
        const bool soft_mesh = argc == 3 && std::string(argv[2]) == "--soft-mesh";
        const bool refine = argc == 3 && (std::string(argv[2]) == "--refine" || soft_mesh);
        const bool identity_audit = argc == 3 && std::string(argv[2]) == "--identity";
        xmvb::memory::configure({});
        auto loaded = load_vbscf_input_with_timings(argv[1]);
        auto& input = loaded.input;
        auto& o = input.orbital_preparation_input;
        if (!o.maintain_inactive_gauge)
            apply_support_preserving_inactive_gauge(&o);
        balance_active_gauge(&o);
        const OrbitalGradientEvaluator evaluator;
        const auto accepted =
            evaluator.evaluate_without_reference_energy_gradient(input,
                                                                 {0},
                                                                 {1.0},
                                                                 loaded.nuclear_repulsion_energy,
                                                                 StructureEigensolver::Davidson,
                                                                 StructureSolveAccuracy{},
                                                                 Eigen::MatrixXd{});
        const auto& context = *accepted.second_order_context;
        const auto& prep = accepted.orbital_preparation_result;
        const SparseParameterLayout layout(o);
        OrbitalChart chart(o,
                           layout,
                           prep.auxiliary_orbital_matrix.leftCols(o.n_orbitals),
                           prep.physical_orbital_frame.normalized_orbital_matrix,
                           nullptr,
                           false,
                           input.complete_active_space,
                           nullptr,
                           OrbitalChart::Coordinates::Unwhitened);
        const ExactHvpOperator hvp(accepted.second_order_context, &input, layout, &chart);
        if (argc == 3 && std::string_view(argv[2]).starts_with("--path=")) {
            xmvb::test::export_ciah_path(o, hvp.structure_action(), argv[2] + 7);
            return 0;
        }
        const StructureTangentOperator structure(accepted.second_order_context,
                                                 hvp.structure_action());
        const AffineCoupling affine(context, hvp, structure);
        const CiTransport transport(o, layout, chart, context, hvp.structure_action(), structure);
        const NonredundantRetractionMetric metric(chart, layout, o);
        const Eigen::Index no = chart.reduced_size(), ns = structure.tangent_size(),
                           dimension = no + ns;
        const Eigen::Index nc = (o.n_total_electrons - o.n_active_electrons) / 2,
                           na = o.n_active_orbitals;
        if (input.complete_active_space || o.orbital_type != kOrbitalTypeOeo || nc < 1 || na < 2 ||
            nc + na >= o.n_basis_functions || ns == 0)
            throw std::invalid_argument("audit requires selected OEO, inactive and virtual "
                                        "orbitals, and a nonzero CI tangent");
        const Eigen::MatrixXd phi = prep.physical_orbital_frame.normalized_orbital_matrix;
        Eigen::VectorXd origin = Eigen::VectorXd::Zero(dimension);
        origin.head(no) = chart.project_reduced_gradient(
            layout.gather_from_full(accepted.sparse_orbital_energy_gradient));
        origin.tail(ns) = transport.origin_structure_gradient();
        const auto norm = [&](const Eigen::VectorXd& x) {
            return std::hypot(chart.coordinate_dual_norm(x.head(no)), x.tail(ns).norm());
        };
        Eigen::MatrixXd reference_active;
        const auto model = [&](const Eigen::VectorXd& x, bool exact_identity = false) {
            auto point = transport.retract(x.head(no), x.tail(ns));
            if (exact_identity) {
                if (!x.head(no).isZero())
                    throw std::runtime_error("identity audit requires no orbital displacement");
                point.coefficients = point.base_coefficients;
            }
            auto trial = input;
            trial.orbital_preparation_input = point.input;
            const auto fixed = evaluator.evaluate_fixed_structure(trial,
                                                                  {0},
                                                                  {1.0},
                                                                  loaded.nuclear_repulsion_energy,
                                                                  point.coefficients,
                                                                  point.selected_metric.get());
            VbScfCoupledKeyframeEvaluation keyframe;
            keyframe.energy = fixed.gradient.scf_result.total_energy;
            keyframe.orbital_gradient =
                layout.gather_from_full(fixed.gradient.sparse_orbital_energy_gradient);
            keyframe.normalized_structure_coefficients = fixed.normalized_coefficients;
            keyframe.structure_residuals = fixed.structure_residuals;
            keyframe.coefficient_gradient = fixed.coefficient_gradient;
            double frame_change = 0, span_change = 0;
            if (reference_active.size()) {
                const Eigen::MatrixXd inverse = point.upper.triangularView<Eigen::Upper>().solve(
                    Eigen::MatrixXd::Identity(na, na));
                const Eigen::MatrixXd chi =
                    point.prepared.auxiliary_orbital_matrix.middleCols(nc, na) * inverse;
                const Eigen::MatrixXd cross =
                    reference_active.transpose() * o.ao_overlap_matrix * chi;
                frame_change = (cross - Eigen::MatrixXd::Identity(na, na)).norm();
                const Eigen::MatrixXd outside = chi - reference_active * cross;
                span_change = std::sqrt(
                    std::max(0.0, (outside.transpose() * o.ao_overlap_matrix * outside).trace()));
            }
            return ModelValue{transport.evaluate_model(point, keyframe),
                              keyframe.energy,
                              fixed.structure_residuals.norm(),
                              frame_change,
                              span_change};
        };
        const auto action = [&](const Eigen::VectorXd& x) {
            const auto image = affine.apply(x.head(no), x.tail(ns));
            Eigen::VectorXd result(dimension);
            result << image.orbital.orbital_hessian + image.adjoint,
                image.forcing + image.ci.hessian_coordinates;
            return result;
        };
        std::cout << std::setprecision(17);
        const Eigen::Map<const Eigen::MatrixXd> overlap(
            prep.active_orbital_overlap_matrix.data(), na, na);
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(overlap);
        const auto zero = model(Eigen::VectorXd::Zero(dimension));
        std::cout << "POINT,n_bf=" << o.n_basis_functions << ",orbital=" << no << ",ci=" << ns
                  << ",complete=" << input.complete_active_space
                  << ",active_gauge=" << chart.uses_active_subspace_gauge()
                  << ",energy=" << accepted.scf_result.total_energy
                  << ",ci_residual=" << zero.ci_residual
                  << ",origin_gradient_error=" << norm(zero.model.gradient - origin)
                  << ",origin_energy_error=" << zero.model.energy - accepted.scf_result.total_energy
                  << ",active_gram_min=" << spectrum.eigenvalues()[0]
                  << ",active_gram_max=" << spectrum.eigenvalues().tail(1)[0] << std::endl;
        if (argc == 3 && (std::string(argv[2]) == "--joint-geometry" ||
                          std::string(argv[2]) == "--selected-curvature")) {
            const auto diagonal = build_analytic_orbital_diagonal(input, context);
            chart.set_curvature(
                o, nullptr, &diagonal, OrbitalChart::CurvatureUse::AugmentedHessian);
            xmvb::test::audit_joint_geometry(
                o,
                prep,
                context,
                layout,
                chart,
                structure,
                hvp.structure_action(),
                context.selected_state_eigenvectors,
                metric,
                origin,
                action,
                [&](const Eigen::VectorXd& x) { return model(x).model.gradient; },
                [&](const Eigen::VectorXd& x) { return model(x).model.energy; },
                std::string(argv[2]) == "--selected-curvature");
            return 0;
        }
        if (argc == 3 && std::string(argv[2]) == "--dense-ah") {
            // Strictly bounded independent diagnostic. Production remains matrix-free.
            if (dimension > 2048 || context.n_structures > 8)
                throw std::length_error("dense AH oracle is restricted to small selected cases");
            const auto diagonal = build_analytic_orbital_diagonal(input, context);
            chart.set_curvature(
                o, nullptr, &diagonal, OrbitalChart::CurvatureUse::AugmentedHessian);
            reference_active = diagonal.mo_coefficients.middleCols(nc, na);
            Eigen::MatrixXd h(dimension, dimension), mass(dimension, dimension);
            for (Eigen::Index j = 0; j < dimension; ++j) {
                const Eigen::VectorXd v = Eigen::VectorXd::Unit(dimension, j);
                h.col(j) = action(v);
                mass.col(j).head(no) = metric.apply(v.head(no));
                mass.col(j).tail(ns) = structure.apply_metric_coordinates(v.tail(ns));
                if ((j + 1) % 128 == 0)
                    std::cout << "DENSE_PROGRESS,column=" << j + 1 << std::endl;
            }
            const Eigen::VectorXd scale = mass.diagonal().array().sqrt().inverse();
            const Eigen::MatrixXd hs = scale.asDiagonal() * h * scale.asDiagonal();
            const Eigen::MatrixXd ms = scale.asDiagonal() * mass * scale.asDiagonal();
            std::cout << "DENSE_MATRIX,h_skew=" << (h - h.transpose()).norm()
                      << ",h_norm=" << h.norm() << ",scaled_h_skew=" << (hs - hs.transpose()).norm()
                      << ",scaled_h_norm=" << hs.norm()
                      << ",metric_skew=" << (ms - ms.transpose()).norm()
                      << ",metric_norm=" << ms.norm() << std::endl;
            const Eigen::MatrixXd orbital_h =
                .5 * (hs.topLeftCorner(no, no) + hs.topLeftCorner(no, no).transpose());
            const Eigen::MatrixXd orbital_m =
                .5 * (ms.topLeftCorner(no, no) + ms.topLeftCorner(no, no).transpose());
            const auto orbital_eigenvalues =
                xmvb::linalg::GeneralizedEigensolver{}.solve_dense_eigenvalues(
                    {orbital_h.data(), static_cast<std::size_t>(orbital_h.size())},
                    {orbital_m.data(), static_cast<std::size_t>(orbital_m.size())},
                    no);
            const Eigen::MatrixXd ci_h =
                .5 * (hs.bottomRightCorner(ns, ns) + hs.bottomRightCorner(ns, ns).transpose());
            const Eigen::MatrixXd ci_m =
                .5 * (ms.bottomRightCorner(ns, ns) + ms.bottomRightCorner(ns, ns).transpose());
            const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ci_eigenvalues(
                ci_h, ci_m, Eigen::EigenvaluesOnly);
            if (ci_eigenvalues.info() != Eigen::Success)
                throw std::runtime_error("diagnostic CI block spectrum failed");
            std::cout << "BLOCK_SPECTRUM,orbital_min=" << orbital_eigenvalues.front()
                      << ",orbital_max=" << orbital_eigenvalues.back()
                      << ",ci_min=" << ci_eigenvalues.eigenvalues()[0]
                      << ",ci_max=" << ci_eigenvalues.eigenvalues().tail(1)[0] << std::endl;
            if (orbital_eigenvalues.front() > 0) {
                const Eigen::MatrixXd b =
                    .5 * (hs.bottomLeftCorner(ns, no) + hs.topRightCorner(no, ns).transpose());
                const Eigen::MatrixXd inverse_b = orbital_h.ldlt().solve(b.transpose());
                const Eigen::MatrixXd schur = ci_h - b * inverse_b;
                const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> schur_spectrum(
                    (.5 * (schur + schur.transpose())).eval(), ci_m, Eigen::EigenvaluesOnly);
                if (schur_spectrum.info() != Eigen::Success)
                    throw std::runtime_error("diagnostic exact Schur spectrum failed");
                std::cout << "EXACT_SCHUR,minimum=" << schur_spectrum.eigenvalues()[0]
                          << ",maximum=" << schur_spectrum.eigenvalues().tail(1)[0]
                          << ",inverse_residual="
                          << (orbital_h * inverse_b - b.transpose()).norm() /
                                 (orbital_h.norm() * inverse_b.norm() + b.norm())
                          << std::endl;
            }
            if (orbital_eigenvalues.front() > 0) {
                const Eigen::MatrixXd raw_a =
                    .5 * (h.topLeftCorner(no, no) + h.topLeftCorner(no, no).transpose());
                const Eigen::MatrixXd raw_b =
                    .5 * (h.bottomLeftCorner(ns, no) + h.topRightCorner(no, ns).transpose());
                const Eigen::MatrixXd raw_c =
                    .5 * (h.bottomRightCorner(ns, ns) + h.bottomRightCorner(ns, ns).transpose());
                const Eigen::MatrixXd raw_m = .5 * (mass.bottomRightCorner(ns, ns) +
                                                    mass.bottomRightCorner(ns, ns).transpose());
                std::cout << "RESPONSE_MODEL,frame=diagonal" << std::endl;
                xmvb::test::audit_coupling_response(
                    raw_a, raw_b, raw_c, raw_m, no, [&](const Eigen::VectorXd& rhs) {
                        return chart.apply_inverse_augmented_hessian_diagonal(rhs, 0.0);
                    });
            }
            if (orbital_eigenvalues.front() > 0) {
                const Eigen::Index count = na * (na - 1);
                Eigen::MatrixXd sources(no, count);
                const auto& physical = prep.physical_orbital_frame.normalized_orbital_matrix;
                Eigen::Index column = 0;
                for (Eigen::Index i = 0; i < na; ++i)
                    for (Eigen::Index j = 0; j < na; ++j) {
                        if (i == j)
                            continue;
                        std::vector<double> full(o.orbital_value_table.size(), 0.0);
                        for (int k = 0; k < stored_sparse_orbital_coefficient_count(o, nc + i);
                             ++k) {
                            const auto slot = (nc + i) * o.n_basis_functions + k;
                            full[slot] = physical(o.orbital_basis_index_table[slot] - 1, nc + j);
                        }
                        sources.col(column++) =
                            chart.project_vector(layout.gather_from_full(full)).reduced_gradient;
                    }
                const Eigen::HouseholderQR<Eigen::MatrixXd> qr(sources);
                const Eigen::MatrixXd basis =
                    qr.householderQ() * Eigen::MatrixXd::Identity(no, count);
                const Eigen::MatrixXd raw_a =
                    .5 * (h.topLeftCorner(no, no) + h.topLeftCorner(no, no).transpose());
                const Eigen::MatrixXd raw_b =
                    .5 * (h.bottomLeftCorner(ns, no) + h.topRightCorner(no, ns).transpose());
                const Eigen::MatrixXd raw_c =
                    .5 * (h.bottomRightCorner(ns, ns) + h.bottomRightCorner(ns, ns).transpose());
                const Eigen::MatrixXd raw_m = .5 * (mass.bottomRightCorner(ns, ns) +
                                                    mass.bottomRightCorner(ns, ns).transpose());
                xmvb::mcscf::CoarsePreconditioner internal(no, static_cast<int>(count), 1e-14);
                for (Eigen::Index j = 0; j < count; ++j)
                    internal.observe(basis.col(j),
                                     raw_a * basis.col(j),
                                     mass.topLeftCorner(no, no) * basis.col(j));
                std::cout << "RESPONSE_MODEL,frame=exact_internal,setup_orbital_actions=" << count
                          << ",retained=" << internal.size() << std::endl;
                xmvb::test::audit_coupling_response(
                    raw_a, raw_b, raw_c, raw_m, no, [&](const Eigen::VectorXd& rhs) {
                        return internal.apply(rhs, 0.0, [&](const Eigen::VectorXd& v) {
                            return chart.apply_inverse_augmented_hessian_diagonal(v, 0.0);
                        });
                    });
            }
            Eigen::MatrixXd ah = Eigen::MatrixXd::Zero(dimension + 1, dimension + 1);
            Eigen::MatrixXd am = Eigen::MatrixXd::Identity(dimension + 1, dimension + 1);
            ah.block(1, 0, dimension, 1) = scale.cwiseProduct(origin);
            ah.block(0, 1, 1, dimension) = ah.block(1, 0, dimension, 1).transpose();
            ah.bottomRightCorner(dimension, dimension) = .5 * (hs + hs.transpose());
            am.bottomRightCorner(dimension, dimension) = .5 * (ms + ms.transpose());
            const auto exact = xmvb::linalg::GeneralizedEigensolver{}.solve_dense(
                {ah.data(), static_cast<std::size_t>(ah.size())},
                {am.data(), static_cast<std::size_t>(am.size())},
                dimension + 1);
            const Eigen::Map<const Eigen::MatrixXd> vectors(
                exact.eigenvector_matrix.data(), dimension + 1, dimension + 1);
            // Does the inexpensive orbital image of the CI-coupling range
            // contain the missing curvature? Uses the already assembled oracle
            // images, not additional physical actions or an optimizer policy.
            Eigen::MatrixXd coupling_range = Eigen::MatrixXd::Zero(dimension, ns + 1);
            coupling_range.col(0).head(no) =
                chart.apply_inverse_augmented_hessian_diagonal(origin.head(no), 0.0);
            for (Eigen::Index j = 0; j < ns; ++j) {
                coupling_range.col(j + 1).head(no) =
                    -chart.apply_inverse_augmented_hessian_diagonal(h.col(no + j).head(no), 0.0);
                coupling_range(no + j, j + 1) = 1.0;
            }
            Eigen::MatrixXd range_mass = coupling_range.transpose() * mass * coupling_range;
            const Eigen::VectorXd range_scale = range_mass.diagonal().array().sqrt().inverse();
            coupling_range = (coupling_range * range_scale.asDiagonal()).eval();
            range_mass = coupling_range.transpose() * mass * coupling_range;
            const Eigen::MatrixXd range_h = coupling_range.transpose() * h * coupling_range;
            const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> range_spectrum(
                (.5 * (range_h + range_h.transpose())).eval(),
                (.5 * (range_mass + range_mass.transpose())).eval(),
                Eigen::EigenvaluesOnly);
            if (range_spectrum.info() != Eigen::Success)
                throw std::runtime_error("coupling-range diagnostic spectrum failed");
            std::cout << "COUPLING_RANGE,dimension=" << ns + 1
                      << ",minimum=" << range_spectrum.eigenvalues()[0]
                      << ",maximum=" << range_spectrum.eigenvalues().tail(1)[0] << std::endl;
            Eigen::Index selected = 0;
            while (selected < dimension + 1 && std::abs(vectors(0, selected)) <= .1)
                ++selected;
            if (selected == dimension + 1)
                throw std::runtime_error("dense AH has no reference-connected root");
            // Compare both the lowest ray and the production reference-connected root.
            for (Eigen::Index root : {Eigen::Index(0), selected}) {
                const double alpha = vectors(0, root);
                Eigen::VectorXd p = scale.cwiseProduct(vectors.col(root).tail(dimension));
                double rhs_weight = alpha;
                if (std::abs(alpha) > .1) {
                    p /= alpha;
                    rhs_weight = 1.0;
                } else if (origin.dot(p) > 0) {
                    p = -p;
                    rhs_weight = -alpha;
                }
                const Eigen::VectorXd hp = h * p, mp = mass * p;
                const Eigen::VectorXd range_fit =
                    range_mass.ldlt().solve(coupling_range.transpose() * mp);
                const Eigen::VectorXd range_error = p - coupling_range * range_fit;
                const double range_loss =
                    std::sqrt(std::max(0.0, range_error.dot(mass * range_error) / p.dot(mp)));
                const double orbital_frame = chart.maximum_rotation_component(p.head(no));
                const double ci_component =
                    structure.maximum_coefficient_component_coordinates(p.tail(ns));
                const double largest = std::max(orbital_frame, ci_component);
                const Eigen::VectorXd raw = chart.expand_retract_input_tangent(o, p.head(no));
                double angle_max = 0, radial_max = 0;
                Eigen::VectorXd b(o.n_basis_functions), delta(o.n_basis_functions);
                for (Eigen::Index i = 0; i < nc + na; ++i) {
                    b.setZero();
                    delta.setZero();
                    for (int j = 0; j < stored_sparse_orbital_coefficient_count(o, i); ++j) {
                        const auto slot = i * o.n_basis_functions + j;
                        const auto mu = o.orbital_basis_index_table[slot] - 1;
                        b[mu] = o.orbital_value_table[slot];
                        delta[mu] = raw[slot];
                    }
                    const auto geometry =
                        diagnostics::measure_orbital_step(b, delta, o.ao_overlap_matrix);
                    angle_max = std::max(angle_max, geometry.theta);
                    radial_max = std::max(radial_max, std::abs(geometry.alpha));
                }
                const double clip = largest > .03 ? .03 / largest : 1.0;
                const double angular_max = std::max(angle_max, ci_component);
                const double angular_clip = angular_max > .03 ? .03 / angular_max : 1.0;
                std::cout << "DENSE_ROOT,index=" << root
                          << ",eigenvalue=" << exact.eigenvalues[root] << ",reference=" << alpha
                          << ",linear=" << origin.dot(p) << ",curvature=" << p.dot(hp)
                          << ",step_norm=" << p.norm()
                          << ",physical_length=" << std::sqrt(p.dot(mp))
                          << ",max_component=" << largest << ",clip=" << clip
                          << ",model_gradient=" << norm(origin + hp) << ",shifted_residual="
                          << norm(rhs_weight * origin + hp - exact.eigenvalues[root] * mp)
                          << ",orbital_frame_max=" << orbital_frame << ",ci_max=" << ci_component
                          << ",theta_max=" << angle_max << ",radial_max=" << radial_max
                          << ",angular_clip=" << angular_clip
                          << ",coupling_range_loss=" << range_loss << std::endl;
                for (double factor : {clip, angular_clip, angular_clip / 2}) {
                    const Eigen::VectorXd step = factor * p;
                    const auto trial = model(step);
                    const auto point = transport.retract(step.head(no), step.tail(ns));
                    auto relaxed_input = input;
                    relaxed_input.orbital_preparation_input = point.input;
                    const auto relaxed = evaluator.evaluate_without_reference_energy_gradient(
                        relaxed_input,
                        {0},
                        {1.0},
                        loaded.nuclear_repulsion_energy,
                        StructureEigensolver::Davidson,
                        StructureSolveAccuracy{},
                        point.coefficients);
                    const Eigen::VectorXd relaxed_gradient = chart.project_reduced_gradient(
                        layout.gather_from_full(relaxed.sparse_orbital_energy_gradient));
                    const double predicted = origin.dot(step) + .5 * factor * factor * p.dot(hp);
                    std::cout << "DENSE_TRIAL,root=" << root << ",factor=" << factor
                              << ",predicted_change=" << predicted
                              << ",model_change=" << trial.model.energy - zero.model.energy
                              << ",physical_change=" << trial.physical_energy - zero.physical_energy
                              << ",gradient=" << norm(trial.model.gradient)
                              << ",carrier_frame_change=" << trial.frame_change
                              << ",active_span_change=" << trial.span_change
                              << ",relaxed_energy_change="
                              << relaxed.scf_result.total_energy - accepted.scf_result.total_energy
                              << ",relaxed_orbital_gradient="
                              << chart.coordinate_dual_norm(relaxed_gradient) << std::endl;
                }
                for (double factor : {clip, 2 * clip}) {
                    const auto plus = model(factor * p), minus = model(-factor * p);
                    const double finite =
                        (plus.model.energy + minus.model.energy - 2 * zero.model.energy) /
                        (factor * factor);
                    std::cout << "DENSE_CURVATURE_FD,root=" << root << ",step=" << factor
                              << ",analytic=" << p.dot(hp) << ",finite=" << finite
                              << ",error=" << finite - p.dot(hp) << ",gradient_difference="
                              << p.dot(plus.model.gradient - minus.model.gradient) / (2 * factor)
                              << std::endl;
                }
                if (root == selected)
                    break;
            }
            return 0;
        }
        auto make_orbital = [&](const Eigen::MatrixXd& tangent) {
            std::vector<double> full(o.orbital_value_table.size(), 0.0);
            for (Eigen::Index i = 0; i < o.n_orbitals; ++i)
                for (int j = 0; j < stored_sparse_orbital_coefficient_count(o, i); ++j) {
                    const auto slot = i * o.n_basis_functions + j;
                    full[slot] = tangent(o.orbital_basis_index_table[slot] - 1, i);
                }
            Eigen::VectorXd p =
                chart.project_vector(layout.gather_from_full(full)).reduced_gradient;
            const double length = metric.norm(p);
            if (!(length > 0))
                throw std::runtime_error("diagnostic direction was a removed gauge");
            p /= length;
            Eigen::VectorXd result = Eigen::VectorXd::Zero(dimension);
            result.head(no) = p;
            return result;
        };
        std::vector<Direction> directions;
        Eigen::MatrixXd tangent = Eigen::MatrixXd::Zero(o.n_basis_functions, o.n_orbitals);
        tangent.col(nc) = phi.col(nc + 1);
        directions.push_back({"active_local", make_orbital(tangent)});
        tangent.setZero();
        for (Eigen::Index i = 0; i < na; ++i)
            tangent.col(nc + i) = phi.col(nc + (i + 1) % na) + 0.37 * phi.col(nc + (i + 3) % na);
        directions.push_back({"active_collective", make_orbital(tangent)});
        tangent.setZero();
        tangent.col(nc) = prep.auxiliary_orbital_matrix.col(nc + na);
        directions.push_back({"active_virtual", make_orbital(tangent)});
        tangent.setZero();
        tangent.col(nc - 1) = phi.col(nc);
        directions.push_back({"inactive_active", make_orbital(tangent)});
        Eigen::VectorXd ci = Eigen::VectorXd::LinSpaced(ns, 0.63, -0.29);
        ci /= std::sqrt(structure.metric_inner_product(ci, ci));
        Eigen::VectorXd ci_direction = Eigen::VectorXd::Zero(dimension);
        ci_direction.tail(ns) = ci;
        directions.push_back({"ci", ci_direction});
        if (identity_audit) {
            const auto direct_zero = model(Eigen::VectorXd::Zero(dimension), true);
            const auto zero_point =
                transport.retract(Eigen::VectorXd::Zero(no), Eigen::VectorXd::Zero(ns));
            std::cout << "IDENTITY,coefficient_defect="
                      << (zero_point.coefficients - zero_point.base_coefficients).norm()
                      << ",production_gradient_defect=" << norm(zero.model.gradient - origin)
                      << ",exact_gradient_defect=" << norm(direct_zero.model.gradient - origin)
                      << std::endl;
            const Eigen::VectorXd h = action(ci_direction);
            for (double step : {1.e-3, 3.e-4, 1.e-4, 3.e-5, 1.e-5})
                for (bool direct : {false, true}) {
                    const auto plus = model(step * ci_direction, direct),
                               minus = model(-step * ci_direction, direct);
                    const Eigen::VectorXd fd =
                        (plus.model.gradient - minus.model.gradient) / (2 * step);
                    std::cout << "IDENTITY_FD,step=" << step << ",exact=" << direct
                              << ",error=" << norm(fd - h) << ",relative=" << norm(fd - h) / norm(h)
                              << std::endl;
                }
            return 0;
        }

        Eigen::VectorXd mixed = directions[1].value;
        mixed.tail(ns) = 0.7 * ci;
        directions.push_back({"mixed", mixed});
        if (argc == 3 && std::string(argv[2]) == "--repeat") {
            const Eigen::VectorXd zero_x = Eigen::VectorXd::Zero(dimension);
            const auto first = model(zero_x);
            const Eigen::VectorXd reference_h = action(mixed);
            for (int i = 0; i < 6; ++i) {
                const auto value = model(zero_x);
                const Eigen::VectorXd h = action(mixed);
                std::cout << "REPEAT,index=" << i
                          << ",energy_delta=" << value.model.energy - first.model.energy
                          << ",gradient_delta=" << norm(value.model.gradient - first.model.gradient)
                          << ",hvp_delta=" << norm(h - reference_h)
                          << ",origin_gradient_defect=" << norm(value.model.gradient - origin)
                          << std::endl;
            }
            return 0;
        }

        Eigen::MatrixXd ci_hessian(ns, ns);
        for (Eigen::Index j = 0; j < ns; ++j)
            ci_hessian.col(j) = structure.apply_hessian_coordinates(Eigen::VectorXd::Unit(ns, j));
        const Eigen::FullPivLU<Eigen::MatrixXd> ci_factor(ci_hessian);
        if (!ci_factor.isInvertible())
            throw std::runtime_error("diagnostic CI response is singular");
        Eigen::VectorXd relaxed = directions[1].value;
        relaxed.tail(ns) = ci_factor.solve(-action(relaxed).tail(ns));
        directions.push_back({"relaxed_internal", relaxed});
        if (refine) {
            // Only the active-internal + CI span: no full orbital Hessian or
            // repeated minimum-curvature search in the production optimizer.
            const Eigen::Index count = na * (na - 1) + ns;
            Eigen::MatrixXd sources(dimension, count);
            Eigen::Index column = 0;
            for (Eigen::Index i = 0; i < na; ++i)
                for (Eigen::Index j = 0; j < na; ++j)
                    if (i != j) {
                        tangent.setZero();
                        tangent.col(nc + i) = phi.col(nc + j);
                        sources.col(column++) = make_orbital(tangent);
                    }
            for (Eigen::Index j = 0; j < ns; ++j) {
                sources.col(column).setZero();
                sources(no + j, column++) = 1;
            }
            const Eigen::HouseholderQR<Eigen::MatrixXd> qr(sources);
            const Eigen::MatrixXd basis =
                qr.householderQ() * Eigen::MatrixXd::Identity(dimension, count);
            Eigen::MatrixXd images(dimension, count);
            for (Eigen::Index j = 0; j < count; ++j)
                images.col(j) = action(basis.col(j));
            const Eigen::MatrixXd projected = basis.transpose() * images;
            std::cout << "SOFT_SPACE,dimension=" << count
                      << ",skew=" << (projected - projected.transpose()).norm() << std::endl;
            const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(
                (0.5 * (projected + projected.transpose())).eval());
            for (Eigen::Index j : {Eigen::Index(0), Eigen::Index(3), Eigen::Index(5)}) {
                Eigen::VectorXd v = basis * eig.eigenvectors().col(j);
                const double scale =
                    std::sqrt(std::pow(metric.norm(v.head(no)), 2) +
                              structure.metric_inner_product(v.tail(ns), v.tail(ns)));
                v /= scale;
                std::cout << "SOFT_MODE,index=" << j << ",eigenvalue=" << eig.eigenvalues()[j]
                          << ",gdot=" << origin.dot(v) << ",scale=" << scale << std::endl;
                directions.push_back({"soft_" + std::to_string(j), v});
            }
        }
        Eigen::VectorXd other = directions[0].value + 0.43 * ci_direction;
        const Eigen::VectorXd h_other = action(other);
        for (const auto& direction : directions) {
            if (soft_mesh && !direction.name.starts_with("soft_"))
                continue;
            if (refine && direction.name != "active_collective" && direction.name != "mixed" &&
                direction.name != "relaxed_internal" && !direction.name.starts_with("soft_"))
                continue;
            const auto& v = direction.value;
            const Eigen::VectorXd h = action(v);
            std::cout << "DIRECTION," << direction.name << ",coordinate_length=" << v.norm()
                      << ",orbital_length=" << metric.norm(v.head(no)) << ",ci_length="
                      << std::sqrt(
                             std::max(0.0, structure.metric_inner_product(v.tail(ns), v.tail(ns))))
                      << ",gdot=" << origin.dot(v) << ",curvature=" << v.dot(h)
                      << ",hvp_norm=" << norm(h) << ",reciprocity=" << other.dot(h) - v.dot(h_other)
                      << std::endl;
            for (double step :
                 (refine && !soft_mesh ? std::vector<double>{3.e-5, 1.e-5, 3.e-6, 1.e-6}
                                       : std::vector<double>{1.e-2, 3.e-3, 1.e-3, 3.e-4, 1.e-4})) {
                const auto plus = model(step * v), minus = model(-step * v);
                const Eigen::VectorXd fd =
                    (plus.model.gradient - minus.model.gradient) / (2 * step);
                const Eigen::VectorXd error = fd - h;
                const double efd = (plus.model.energy - minus.model.energy) / (2 * step);
                std::cout << "ORIGIN," << direction.name << ",step=" << step << ",energy_fd=" << efd
                          << ",energy_error=" << efd - origin.dot(v) << ",physical_energy_fd="
                          << (plus.physical_energy - minus.physical_energy) / (2 * step)
                          << ",curvature=" << v.dot(h) << ",curvature_fd=" << v.dot(fd)
                          << ",curvature_error=" << v.dot(error) << ",hvp_error=" << norm(error)
                          << ",hvp_relative=" << norm(error) / std::max(norm(h), norm(fd))
                          << ",orbital_error=" << chart.coordinate_dual_norm(error.head(no))
                          << ",ci_error=" << error.tail(ns).norm() << std::endl;
            }
        }
        if (soft_mesh)
            return 0;
        const Eigen::VectorXd base = directions[5].value;
        for (double radius : {0.03, 0.1}) {
            const Eigen::VectorXd z = radius * base;
            const auto fixed = model(z);
            const Eigen::VectorXd image = action(z), predicted = origin + image;
            const double predicted_energy = zero.model.energy + origin.dot(z) + 0.5 * z.dot(image);
            std::cout << "FINITE_MODEL,radius=" << radius << ",actual_energy=" << fixed.model.energy
                      << ",quadratic_energy=" << predicted_energy
                      << ",energy_defect=" << fixed.model.energy - predicted_energy
                      << ",gradient_defect=" << norm(fixed.model.gradient - predicted)
                      << ",gradient_true=" << norm(fixed.model.gradient)
                      << ",gradient_origin=" << norm(origin) << std::endl;

            for (Eigen::Index i : {Eigen::Index(0), Eigen::Index(4), Eigen::Index(5)}) {
                const auto& v = directions[i].value;
                for (double step : (refine ? std::vector<double>{1.e-4, 3.e-5, 1.e-5}
                                           : std::vector<double>{1.e-3, 3.e-4, 1.e-4})) {
                    const auto plus = model(z + step * v), minus = model(z - step * v);
                    const double fd = (plus.model.energy - minus.model.energy) / (2 * step);
                    std::cout << "KEYFRAME," << directions[i].name << ",radius=" << radius
                              << ",step=" << step << ",ci_residual=" << fixed.ci_residual
                              << ",energy_fd=" << fd
                              << ",gradient_pairing=" << fixed.model.gradient.dot(v)
                              << ",error=" << fd - fixed.model.gradient.dot(v) << std::endl;
                }
            }
        }
        std::cout << "DONE" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
