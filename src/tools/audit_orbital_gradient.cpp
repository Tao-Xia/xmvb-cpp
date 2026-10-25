#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "input/loading/loader.hpp"
#include "vbscf/derivatives/gradient/orbital/evaluator.hpp"
#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/derivatives/hessian/coupled/structure.hpp"
#include "vbscf/derivatives/hessian/coupled/operator.hpp"
#include "vbscf/optimization/driver/session.hpp"
#include "vbscf/optimization/objective/function.hpp"
#include "vbscf/optimization/trust_region/retraction.hpp"
#include "vbscf/optimization/neo/curvature.hpp"
#include "vbscf/orbitals/charts/layout.hpp"
#include "vbscf/orbitals/gauge/support_preserving.hpp"

namespace {

using namespace xmvb::vb;

double positive_tolerance(const char* text, const char* label) {
    const double value = std::stod(text);
    if (!(value > 0.0) || !std::isfinite(value)) {
        throw std::invalid_argument(std::string(label) + " must be positive and finite");
    }
    return value;
}

Eigen::VectorXd read_packed_direction(const std::string& path, Eigen::Index expected_size) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open packed direction: " + path);
    }
    const auto expected_bytes = static_cast<std::streamoff>(expected_size * sizeof(double));
    if (stream.tellg() != expected_bytes) {
        throw std::runtime_error("packed direction has the wrong byte count: " + path);
    }
    stream.seekg(0);
    Eigen::VectorXd direction(expected_size);
    stream.read(reinterpret_cast<char*>(direction.data()), expected_bytes);
    if (!stream || !direction.allFinite()) {
        throw std::runtime_error("packed direction is incomplete or non-finite: " + path);
    }
    return direction;
}

void write_doubles(const std::string& path, const double* values, std::size_t count) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot create diagnostic file: " + path);
    }
    stream.write(reinterpret_cast<const char*>(values),
                 static_cast<std::streamsize>(count * sizeof(double)));
    if (!stream) {
        throw std::runtime_error("cannot complete diagnostic file: " + path);
    }
}

double infinity_norm(const Eigen::VectorXd& values) {
    return values.size() == 0 ? 0.0 : values.cwiseAbs().maxCoeff();
}

/** @brief Dense diagnostic of the small coupled structure block, not a solver path. */
void audit_coupled_elimination(const ExactHvpOperator& operation,
                               const VbScfObjective& objective,
                               const Eigen::VectorXd& direction,
                               const Eigen::VectorXd& relaxed_image) {
    const StructureTangentOperator structure(objective.second_order_context(),
                                             operation.structure_action());
    const Eigen::Index n = structure.tangent_size();
    constexpr Eigen::Index maximum_reference_size = 1024;
    if (n == 0 || n > maximum_reference_size) {
        std::cout << "coupled_reference_available = 0\n";
        return;
    }
    Eigen::MatrixXd c(n, n);
    for (Eigen::Index j = 0; j < n; ++j) {
        const Eigen::VectorXd axis = Eigen::VectorXd::Unit(n, j);
        c.col(j) = structure.apply_coupling_coordinates(axis).hessian_coordinates;
    }
    const double symmetry_error = (c - c.transpose()).norm() / std::max(1.0e-12, c.norm());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum((0.5 * (c + c.transpose())).eval());
    if (spectrum.info() != Eigen::Success) {
        throw std::runtime_error("coupled reference eigensystem failed");
    }
    const double cutoff = 256.0 * std::numeric_limits<double>::epsilon() * static_cast<double>(n) *
                          std::max(1.0, spectrum.eigenvalues().cwiseAbs().maxCoeff());
    Eigen::VectorXd inverse = spectrum.eigenvalues();
    for (Eigen::Index j = 0; j < n; ++j) {
        inverse[j] = std::abs(inverse[j]) > cutoff ? 1.0 / inverse[j] : 0.0;
    }
    const auto orbital = operation.apply_orbital_coupling(direction);
    const Eigen::VectorXd b =
        structure.coordinates(structure.project(orbital.scaled_structure_forcing));
    const Eigen::VectorXd q =
        -spectrum.eigenvectors() * inverse.asDiagonal() * (spectrum.eigenvectors().transpose() * b);
    const auto correction = structure.apply_coupling_coordinates(q);
    const Eigen::VectorXd bt = operation.apply_structure_coupling_adjoint(
        correction.coefficient_response, correction.adjoint_multipliers);
    const Eigen::VectorXd reference = orbital.orbital_hessian + bt;
    std::cout << "coupled_reference_available = 1\n"
              << "coupled_reference_structure_dimension = " << n << '\n'
              << "coupled_reference_c_symmetry_error = " << symmetry_error << '\n'
              << "coupled_reference_c_minimum = " << spectrum.eigenvalues()[0] << '\n'
              << "coupled_reference_structure_residual = " << (b + c * q).norm() << '\n'
              << "coupled_reference_hvp_relative_error = "
              << (reference - relaxed_image).norm() /
                     std::max({1.0e-12, reference.norm(), relaxed_image.norm()})
              << '\n';
}

void run_audit(int argc, char** argv) {
    if (argc < 2 || argc > 8) {
        throw std::invalid_argument(
            "usage: audit_orbital_gradient INPUT [dense|davidson] "
            "[energy-tol] [gradient-tol] [packed-direction-f64.bin|gradient|random] "
            "[full-hvp|probe-curvature|vector-hvp|balance-gauge|coupled-keyframe] "
            "[orbital-table-f64.bin]");
    }
    const bool balance_gauge = argc >= 7 && std::string(argv[6]) == "balance-gauge";
    const bool keyframe_audit = argc >= 7 && std::string(argv[6]) == "coupled-keyframe";
    const bool audit_hvp = argc >= 7 && !balance_gauge && !keyframe_audit;
    const bool vector_hvp = audit_hvp && std::string(argv[6]) == "vector-hvp";
    const bool probe_curvature = audit_hvp && std::string(argv[6]) == "probe-curvature";
    if (audit_hvp && !probe_curvature && !vector_hvp && std::string(argv[6]) != "full-hvp") {
        throw std::invalid_argument("unknown HVP audit mode");
    }

    auto loaded = load_vbscf_input_with_timings(argv[1]);
    if (argc == 8) {
        auto& table = loaded.input.orbital_preparation_input.orbital_value_table;
        const Eigen::VectorXd values = read_packed_direction(argv[7], table.size());
        std::copy(values.data(), values.data() + values.size(), table.begin());
    }
    if (balance_gauge)
        balance_active_gauge(&loaded.input.orbital_preparation_input);
    StructureEigensolver solver = loaded.structure_eigensolver;
    if (argc >= 3) {
        const std::string name = argv[2];
        if (name == "dense") {
            solver = StructureEigensolver::Dense;
        } else if (name == "davidson") {
            solver = StructureEigensolver::Davidson;
        } else {
            throw std::invalid_argument("eigensolver must be dense or davidson");
        }
    }
    StructureSolveAccuracy accuracy;
    if (argc >= 4) {
        accuracy.energy_tolerance = positive_tolerance(argv[3], "energy tolerance");
    }
    if (argc >= 5) {
        accuracy.gradient_tolerance = positive_tolerance(argv[4], "gradient tolerance");
    }

    if (loaded.state_average_count <= 0) {
        throw std::runtime_error("loaded state average has no selected states");
    }
    std::vector<int> selected_states(loaded.state_average_count);
    std::iota(selected_states.begin(), selected_states.end(), 0);
    const std::vector<double> state_weights(selected_states.size(), 1.0);

    const SparseParameterLayout layout(loaded.input.orbital_preparation_input);
    OrbitalGradientEvaluator evaluator;
    VbScfObjective objective(std::move(loaded.input),
                             layout,
                             selected_states,
                             state_weights,
                             loaded.nuclear_repulsion_energy,
                             solver,
                             accuracy,
                             &evaluator);
    const Eigen::VectorXd accepted_parameters =
        layout.pack(objective.input().orbital_preparation_input);
    Eigen::VectorXd packed_gradient;
    const double accepted_energy = objective(accepted_parameters, packed_gradient);
    const OrbitalChart chart = optimizer_detail::build_orbital_chart(
        objective,
        layout,
        probe_curvature ? OrbitalPreconditioner::HessianDiagonal : OrbitalPreconditioner::Identity);
    const Eigen::VectorXd reduced_gradient = chart.project_reduced_gradient(packed_gradient);
    const NonredundantRetractionMetric metric(
        chart, layout, objective.input().orbital_preparation_input);

    Eigen::VectorXd reduced_direction = -reduced_gradient;
    if (argc >= 6 && std::string(argv[5]) == "random") {
        Eigen::VectorXd packed_direction(layout.size());
        for (Eigen::Index i = 0; i < packed_direction.size(); ++i) {
            packed_direction[i] = std::sin(0.73 * static_cast<double>(i + 1)) +
                                  std::cos(1.19 * static_cast<double>(i + 1));
        }
        reduced_direction = chart.project_vector(packed_direction).reduced_gradient;
    } else if (argc >= 6 && std::string(argv[5]) != "gradient") {
        const Eigen::VectorXd packed_direction = read_packed_direction(argv[5], layout.size());
        reduced_direction = chart.project_vector(packed_direction).reduced_gradient;
    }
    const double initial_physical_norm = metric.norm(reduced_direction);
    if (!(initial_physical_norm > 0.0) || !std::isfinite(initial_physical_norm)) {
        throw std::runtime_error("probe direction has zero or invalid physical norm");
    }
    reduced_direction /= initial_physical_norm;
    const double analytic_derivative = reduced_gradient.dot(reduced_direction);
    const double raw_derivative = packed_gradient.dot(chart.expand_step(reduced_direction));

    const auto& orbital_input = objective.input().orbital_preparation_input;
    const std::string prefix = "audit_orbital_gradient_";
    write_doubles(prefix + "accepted_raw_f64.bin",
                  orbital_input.orbital_value_table.data(),
                  orbital_input.orbital_value_table.size());
    write_doubles(prefix + "ao_overlap_f64.bin",
                  orbital_input.ao_overlap_matrix.data(),
                  orbital_input.ao_overlap_matrix.size());
    write_doubles(
        prefix + "packed_gradient_f64.bin", packed_gradient.data(), packed_gradient.size());
    const Eigen::VectorXd packed_tangent = chart.expand_step(reduced_direction);
    write_doubles(prefix + "packed_tangent_f64.bin", packed_tangent.data(), packed_tangent.size());
    // Export the existing AO adjoints, not a second gradient evaluation. These
    // separate an orbital pullback discrepancy from an integral/operator one.
    const auto& gradient_data = objective.gradient_result();
    const auto& hcore = objective.input().ao_integral_input.ao_core_hamiltonian_matrix;
    const auto& core_fock = gradient_data.ao_effective_one_electron_result.ao_effective_h1e;
    const auto& density = gradient_data.orbital_preparation_result.inactive_density_matrix;
    const auto& accepted = *objective.second_order_context();
    write_doubles(prefix + "hcore_f64.bin", hcore.data(), hcore.size());
    write_doubles(prefix + "core_fock_f64.bin", core_fock.data(), core_fock.size());
    write_doubles(prefix + "inactive_density_f64.bin", density.data(), density.size());
    write_doubles(prefix + "inactive_adjoint_f64.bin",
                  accepted.total_inactive_density_gradient.data(),
                  accepted.total_inactive_density_gradient.size());
    write_doubles(prefix + "active_adjoint_f64.bin",
                  accepted.total_active_auxiliary_gradient.data(),
                  accepted.total_active_auxiliary_gradient.size());

    std::cout << std::setprecision(17) << "input = " << argv[1] << '\n'
              << "eigensolver = " << structure_eigensolver_name(solver) << '\n'
              << "two_electron_mode = "
              << standard_two_electron_mode_name(loaded.standard_two_electron_mode) << '\n'
              << "energy_tolerance = " << accuracy.energy_tolerance << '\n'
              << "gradient_tolerance = " << accuracy.gradient_tolerance << '\n'
              << "n_basis_functions = " << orbital_input.n_basis_functions << '\n'
              << "n_orbitals = " << orbital_input.n_orbitals << '\n'
              << "n_active_orbitals = " << orbital_input.n_active_orbitals << '\n'
              << "orbital_type = " << orbital_input.orbital_type << '\n'
              << "complete_active_space = " << objective.input().complete_active_space << '\n'
              << "selected_state_count = " << selected_states.size() << '\n'
              << "accepted_energy = " << accepted_energy << '\n'
              << "packed_parameter_count = " << layout.size() << '\n'
              << "reduced_parameter_count = " << chart.reduced_size() << '\n'
              << "packed_gradient_inf = " << infinity_norm(packed_gradient) << '\n'
              << "reduced_gradient_inf = " << infinity_norm(reduced_gradient) << '\n'
              << "physical_gradient_dual_norm = " << metric.dual_norm(reduced_gradient) << '\n'
              << "probe_source = " << (argc >= 6 ? argv[5] : "negative_reduced_gradient") << '\n'
              << "probe_physical_norm = " << metric.norm(reduced_direction) << '\n'
              << "analytic_directional_derivative = " << analytic_derivative << '\n'
              << "raw_directional_derivative = " << raw_derivative << '\n'
              << "chart_adjoint_error = " << std::abs(raw_derivative - analytic_derivative) << '\n'
              << "binary_layout = float64, Eigen column-major, native byte order\n"
              << "accepted_raw_count = " << orbital_input.orbital_value_table.size() << '\n'
              << "ao_overlap_shape = " << orbital_input.ao_overlap_matrix.rows() << " x "
              << orbital_input.ao_overlap_matrix.cols() << '\n'
              << "packed_gradient_count = " << packed_gradient.size() << '\n'
              << "binary_prefix = " << prefix << '\n';
    std::cout.flush();

    if (keyframe_audit) {
        ExactHvpOperator operation(
            objective.second_order_context(), &objective.input(), layout, &chart);
        const auto& context = *objective.second_order_context();
        const StructureAction& action = operation.structure_action();
        const Eigen::MatrixXd carrier =
            action.orthogonalize_structure_block(context.selected_state_eigenvectors);
        write_doubles(prefix + "carrier_f64.bin", carrier.data(), carrier.size());
        const auto write_strings = [&](const std::string& side,
                                       const SpinDeterminantReuseTable& table) {
            std::ofstream stream(prefix + side + "_strings.txt");
            for (const auto& occupied : table.unique_determinants) {
                for (const int orbital : occupied)
                    stream << orbital << ' ';
                stream << '\n';
            }
        };
        write_strings("alpha", context.same_spin_pair_cache.alpha_reuse_table);
        write_strings("beta", context.same_spin_pair_cache.beta_reuse_table);
        std::cout << "carrier_rows = " << carrier.rows() << '\n'
                  << "carrier_cols = " << carrier.cols() << '\n'
                  << "carrier_norm = " << carrier.norm() << '\n';
        StructureTangentOperator structure(objective.second_order_context(), action);
        Eigen::VectorXd q(structure.tangent_size());
        for (Eigen::Index i = 0; i < q.size(); ++i)
            q[i] = std::sin(0.73 * (i + 1));
        q /= std::sqrt(structure.squared_norm(structure.expand(q)));
        CoupledHessianOperator coupled(operation, structure);
        const auto image = coupled.apply_coordinates(reduced_direction, q);
        const Eigen::MatrixXd dc = structure.coefficient_response_coordinates(q);
        const double coefficient_scale = std::sqrt(2.0 * context.normalized_state_weights[0]);
        const auto gradient_at = [&](double h) {
            const auto orbitals = chart.retract_step(
                objective.input().orbital_preparation_input, reduced_direction, h);
            const Eigen::MatrixXd c = context.selected_state_eigenvectors + h * dc;
            const auto keyframe = objective.evaluate_coupled_keyframe(layout.pack(orbitals), c);
            const auto normalized = keyframe.normalized_structure_coefficients.col(0);
            const double norm = c.col(0).dot(normalized) / normalized.squaredNorm();
            Eigen::VectorXd go = chart.project_reduced_gradient(keyframe.orbital_gradient);
            Eigen::VectorXd gs = structure.project_coordinates((coefficient_scale / norm) *
                                                               keyframe.structure_residuals);
            return std::pair{std::move(go), std::move(gs)};
        };
        for (const double h : {1.0e-3, 3.0e-4}) {
            const auto plus = gradient_at(h);
            const auto minus = gradient_at(-h);
            const Eigen::VectorXd fo = (plus.first - minus.first) / (2.0 * h);
            const Eigen::VectorXd fs = (plus.second - minus.second) / (2.0 * h);
            std::cout << "coupled_h = " << h << '\n'
                      << "coupled_orbital_image_norm = " << image.orbital.norm() << '\n'
                      << "coupled_orbital_fd_norm = " << fo.norm() << '\n'
                      << "coupled_orbital_error_norm = " << (image.orbital - fo).norm() << '\n'
                      << "coupled_structure_image_norm = " << image.structure.norm() << '\n'
                      << "coupled_structure_fd_norm = " << fs.norm() << '\n'
                      << "coupled_structure_error_norm = " << (image.structure - fs).norm() << '\n';
            std::cout.flush();
        }
        return;
    }

    double analytic_curvature = 0.0;
    Eigen::VectorXd full_image, fixed_image;
    if (audit_hvp) {
        ExactHvpOperator operation(
            objective.second_order_context(), &objective.input(), layout, &chart);
        const Eigen::VectorXd image = operation.apply_reduced(reduced_direction);
        full_image = image;
        analytic_curvature = reduced_direction.dot(image);
        HvpComponents fixed_ci;
        fixed_ci.structure_response = false;
        fixed_image = operation.apply_reduced(reduced_direction, fixed_ci);
        if (vector_hvp) {
            audit_coupled_elimination(operation, objective, reduced_direction, image);
        }
        const auto diagnostics = operation.diagnostics();
        std::cout << "full_hvp_curvature = " << analytic_curvature << '\n'
                  << "fixed_ci_hvp_curvature = " << reduced_direction.dot(fixed_image) << '\n'
                  << "structure_response_curvature = " << reduced_direction.dot(image - fixed_image)
                  << '\n'
                  << "hvp_response_enabled = " << diagnostics.outer_response_enabled << '\n'
                  << "hvp_response_relative_residual = "
                  << diagnostics.max_structure_response_relative_residual << '\n';
        write_doubles(prefix + "reduced_hvp_f64.bin", image.data(), image.size());
        if (probe_curvature) {
            const StructureTangentOperator structure(objective.second_order_context(),
                                                     operation.structure_action());
            const auto probe = probe_neo_curvature(operation, structure, chart);
            std::cout << "independent_probe_found_negative = "
                      << (probe.stop_reason == xmvb::mcscf::NeoStopReason::NegativeCurvature)
                      << '\n'
                      << "independent_probe_actions = " << probe.hessian_actions << '\n'
                      << "independent_probe_rayleigh = " << probe.minimum_curvature_value << '\n'
                      << "independent_probe_ritz_residual = " << probe.curvature_residual_norm
                      << '\n'
                      << "independent_probe_global_certificate = "
                      << probe.global_curvature_certified << '\n';
            if (probe.stop_reason == xmvb::mcscf::NeoStopReason::NegativeCurvature) {
                Eigen::VectorXd packed_mode =
                    chart.expand_step(probe.minimum_curvature_direction.head(chart.reduced_size()));
                const double norm =
                    metric.norm(probe.minimum_curvature_direction.head(chart.reduced_size()));
                if (!(norm > 0.0))
                    throw std::runtime_error("structure-only negative mode");
                packed_mode /= norm;
                write_doubles(
                    prefix + "independent_mode_f64.bin", packed_mode.data(), packed_mode.size());
                // A sign witness is not a usable nonlinear step certificate. Test
                // both signs at the existing default physical trust radius, without
                // committing either trial or changing the optimization algorithm.
                const Eigen::VectorXd orbital_mode =
                    probe.minimum_curvature_direction.head(chart.reduced_size()) / norm;
                const double radius = xmvb::mcscf::NeoOptions{}.trust_radius;
                double best_energy = accepted_energy;
                Eigen::VectorXd best_orbitals;
                for (double signed_radius : {-radius, radius}) {
                    const auto trial_orbitals =
                        chart.retract_step(orbital_input, orbital_mode, signed_radius);
                    const auto trial =
                        objective.evaluate_trial_energy(layout.pack(trial_orbitals), false);
                    std::cout << "independent_trial_step = " << signed_radius << '\n'
                              << "independent_trial_energy = " << trial.energy << '\n'
                              << "independent_trial_reduction = " << accepted_energy - trial.energy
                              << '\n';
                    if (trial.energy < best_energy) {
                        best_energy = trial.energy;
                        best_orbitals = Eigen::Map<const Eigen::VectorXd>(
                            trial_orbitals.orbital_value_table.data(),
                            trial_orbitals.orbital_value_table.size());
                    }
                }
                std::cout << "independent_escape_found = " << (best_orbitals.size() > 0) << '\n';
                if (best_orbitals.size() > 0)
                    write_doubles(prefix + "escape_orbitals_f64.bin",
                                  best_orbitals.data(),
                                  best_orbitals.size());
            }
        }
        std::cout.flush();
    }
    const std::vector<double> steps =
        vector_hvp  ? std::vector<double>{1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5}
        : audit_hvp ? std::vector<double>{1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4}
                    : std::vector<double>{1.0e-3, 3.0e-4};
    for (const double h : steps) {
        if (vector_hvp) {
            // Keep the accepted affine chart fixed. Its transpose is the Jacobian
            // pullback at both displaced points; rebuilding either chart would
            // compare covectors in different coordinates rather than a Hessian.
            const auto evaluate = [&](double displacement) {
                const auto orbitals =
                    chart.retract_step(orbital_input, reduced_direction, displacement);
                const Eigen::VectorXd parameters = layout.pack(orbitals);
                return objective.evaluate_trial(parameters, false);
            };
            const auto plus = evaluate(h);
            const auto minus = evaluate(-h);
            const Eigen::VectorXd fd =
                chart.project_reduced_gradient((plus.gradient - minus.gradient) / (2.0 * h));
            const auto relative_error = [](const Eigen::VectorXd& actual,
                                           const Eigen::VectorXd& reference) {
                return (actual - reference).norm() /
                       std::max({1.0e-12, actual.norm(), reference.norm()});
            };
            const double fd_gradient = (plus.energy - minus.energy) / (2.0 * h);
            std::cout << "vector_probe_h = " << h << '\n'
                      << "vector_fd_derivative = " << fd_gradient << '\n'
                      << "vector_gradient_absolute_error = "
                      << std::abs(fd_gradient - analytic_derivative) << '\n'
                      << "vector_gradient_relative_error = "
                      << std::abs(fd_gradient - analytic_derivative) /
                             std::max(
                                 {1.0e-12, std::abs(fd_gradient), std::abs(analytic_derivative)})
                      << '\n'
                      << "vector_hvp_absolute_error = " << (full_image - fd).norm() << '\n'
                      << "vector_hvp_relative_error = " << relative_error(full_image, fd) << '\n'
                      << "vector_hvp_norm = " << full_image.norm() << '\n'
                      << "vector_fd_norm = " << fd.norm() << '\n';
            std::cout.flush();
            continue;
        }
        const auto energy_at = [&](double signed_step) {
            const OrbitalPreparationInput trial_orbitals =
                chart.retract_step(orbital_input, reduced_direction, signed_step);
            const Eigen::VectorXd trial_parameters = layout.pack(trial_orbitals);
            return objective.evaluate_trial_energy(trial_parameters, false).energy;
        };
        const double plus_energy = energy_at(h);
        const double minus_energy = energy_at(-h);
        const double finite_difference = (plus_energy - minus_energy) / (2.0 * h);
        const double error = finite_difference - analytic_derivative;
        std::cout << "probe_h = " << h << '\n'
                  << "probe_plus_energy = " << plus_energy << '\n'
                  << "probe_minus_energy = " << minus_energy << '\n'
                  << "probe_fd_derivative = " << finite_difference << '\n'
                  << "probe_fd_minus_analytic = " << error << '\n'
                  << "probe_fd_relative_error = "
                  << std::abs(error) /
                         std::max(
                             {1.0e-12, std::abs(finite_difference), std::abs(analytic_derivative)})
                  << '\n';
        if (audit_hvp) {
            const double curvature =
                ((plus_energy - accepted_energy) + (minus_energy - accepted_energy)) / (h * h);
            std::cout << "probe_fd_curvature = " << curvature << '\n'
                      << "probe_curvature_minus_hvp = " << curvature - analytic_curvature << '\n';
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        run_audit(argc, argv);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "audit_orbital_gradient: " << error.what() << '\n';
        return 1;
    }
}
