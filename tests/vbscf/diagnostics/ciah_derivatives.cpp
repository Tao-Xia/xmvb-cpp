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

#include "core/memory.hpp"
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
              (std::string(argv[2]) == "--refine" || std::string(argv[2]) == "--identity" ||
               std::string(argv[2]) == "--repeat" || std::string(argv[2]) == "--soft-mesh")))
            throw std::runtime_error("usage: check_ciah_derivatives INPUT.xmi "
                                     "[--refine|--identity|--repeat|--soft-mesh]");
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
        const OrbitalChart chart(o,
                                 layout,
                                 prep.auxiliary_orbital_matrix.leftCols(o.n_orbitals),
                                 prep.physical_orbital_frame.normalized_orbital_matrix,
                                 nullptr,
                                 false,
                                 input.complete_active_space,
                                 nullptr,
                                 OrbitalChart::Coordinates::Unwhitened);
        const ExactHvpOperator hvp(accepted.second_order_context, &input, layout, &chart);
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
            return ModelValue{transport.evaluate_model(point, keyframe),
                              keyframe.energy,
                              fixed.structure_residuals.norm()};
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
