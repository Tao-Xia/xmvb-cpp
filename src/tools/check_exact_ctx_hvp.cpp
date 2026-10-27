#include <cstdint>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include "input/loading/loader.hpp"
#include "tools/response_audit.hpp"
#include "vbscf/derivatives/gradient/orbital/evaluator.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/derivatives/hessian/coupled/operator.hpp"
#include "vbscf/derivatives/hessian/coupled/structure.hpp"
#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/layout.hpp"
#include "vbscf/orbitals/charts/physical_metric.hpp"
#include "vbscf/optimization/preconditioners/hessian_diagonal.hpp"
#include "vbscf/optimization/trust_region/retraction.hpp"

namespace {

struct Options {
    std::string input_path;
    std::string orbital_value_table_bin_path;
    xmvb::vb::StructureEigensolver eigensolver = xmvb::vb::StructureEigensolver::Dense;
    xmvb::vb::StandardTwoElectronMode two_electron_mode = xmvb::vb::StandardTwoElectronMode::Exact;
    bool preconditioned_direction = false;
    std::string orbital_sector;
    std::string packed_direction_path;
    std::string projected_model_prefix;
    bool audit_oeo_chart = false;
    double step = 1.0e-4;
    double max_relative_error = std::numeric_limits<double>::infinity();
    double response_tolerance = 1.0e-3;
    bool stream_pair_products = false;
    int diagonal_samples = 0;
    bool coupled_finite_difference = false;
    int curvature_steps = 0;
    int response_spectrum_limit = 0;
    int compare_exchange_mib = 0;
    int operator_timing_repeats = 0;
};

void print_usage() {
    std::cerr << "usage: check_exact_ctx_hvp <input.xmi>"
              << " [--orbital-value-table-bin path]"
              << " [--step h]"
              << " [--max-rel-error tolerance]"
              << " [--response-tolerance tolerance]"
              << " [--eigensolver dense|davidson]"
              << " [--standard-two-electron-mode exact|ri]"
              << " [--direction gradient|preconditioned_gradient]"
              << " [--orbital-sector inactive-active|inactive-external|active-external]"
              << " [--audit-oeo-chart 0|1]"
              << " [--packed-direction-bin path]"
              << " [--projected-model-prefix path]"
              << " [--stream-pair-products 0|1]"
              << " [--diagonal-samples count]"
              << " [--curvature-steps count]"
              << " [--compare-exchange-mib budget]"
              << " [--operator-timing-repeats count]"
              << " [--response-spectrum-limit count]\n";
}

Options parse_arguments(int argc, char** argv) {
    if (argc < 2 || (argc - 2) % 2 != 0) {
        print_usage();
        throw std::invalid_argument("invalid argument count");
    }
    Options options;
    options.input_path = argv[1];
    for (int argument = 2; argument < argc; argument += 2) {
        const std::string name = argv[argument];
        const std::string value = argv[argument + 1];
        if (name == "--orbital-value-table-bin") {
            options.orbital_value_table_bin_path = value;
        } else if (name == "--packed-direction-bin") {
            options.packed_direction_path = value;
        } else if (name == "--projected-model-prefix") {
            options.projected_model_prefix = value;
        } else if (name == "--orbital-sector") {
            if (value != "inactive-active" && value != "inactive-external" &&
                value != "active-external")
                throw std::invalid_argument("invalid orbital sector");
            options.orbital_sector = value;
        } else if (name == "--audit-oeo-chart") {
            if (value != "0" && value != "1")
                throw std::invalid_argument("--audit-oeo-chart must be 0 or 1");
            options.audit_oeo_chart = value == "1";
        } else if (name == "--operator-timing-repeats") {
            options.operator_timing_repeats = std::stoi(value);
        } else if (name == "--compare-exchange-mib") {
            options.compare_exchange_mib = std::stoi(value);
        } else if (name == "--step") {
            options.step = std::stod(value);
        } else if (name == "--max-rel-error") {
            options.max_relative_error = std::stod(value);
        } else if (name == "--response-tolerance") {
            options.response_tolerance = std::stod(value);
        } else if (name == "--eigensolver") {
            if (value == "dense") {
                options.eigensolver = xmvb::vb::StructureEigensolver::Dense;
            } else if (value == "davidson") {
                options.eigensolver = xmvb::vb::StructureEigensolver::Davidson;
            } else {
                throw std::invalid_argument("--eigensolver must be dense or davidson");
            }
        } else if (name == "--standard-two-electron-mode") {
            if (value == "exact") {
                options.two_electron_mode = xmvb::vb::StandardTwoElectronMode::Exact;
            } else if (value == "ri") {
                options.two_electron_mode = xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
            } else {
                throw std::invalid_argument("--standard-two-electron-mode must be exact or ri");
            }
        } else if (name == "--direction") {
            if (value == "gradient") {
                options.preconditioned_direction = false;
            } else if (value == "preconditioned_gradient") {
                options.preconditioned_direction = true;
            } else {
                throw std::invalid_argument(
                    "--direction must be gradient or preconditioned_gradient");
            }
        } else if (name == "--stream-pair-products") {
            if (value != "0" && value != "1") {
                throw std::invalid_argument("--stream-pair-products must be 0 or 1");
            }
            options.stream_pair_products = value == "1";
        } else if (name == "--diagonal-samples") {
            options.diagonal_samples = std::stoi(value);
        } else if (name == "--curvature-steps") {
            options.curvature_steps = std::stoi(value);
        } else if (name == "--response-spectrum-limit") {
            options.response_spectrum_limit = std::stoi(value);
        } else if (name == "--coupled-finite-difference") {
            if (value != "0" && value != "1") {
                throw std::invalid_argument("--coupled-finite-difference must be 0 or 1");
            }
            options.coupled_finite_difference = value == "1";
        } else if (name == "--probe") {
            if (value != "full") {
                throw std::invalid_argument("only the complete exact HVP is supported");
            }
        } else {
            throw std::invalid_argument("unknown argument: " + name);
        }
    }
    if (!(options.step > 0.0) || !(options.max_relative_error >= 0.0) ||
        options.diagonal_samples < 0 || options.curvature_steps < 0 ||
        options.response_spectrum_limit < 0 || options.compare_exchange_mib < 0 ||
        options.operator_timing_repeats < 0 || options.operator_timing_repeats > 20 ||
        !(options.response_tolerance > 0.0)) {
        throw std::invalid_argument("finite-difference tolerances must be non-negative");
    }
    return options;
}

double infinity_norm(const Eigen::Ref<const Eigen::VectorXd>& values) {
    return values.size() == 0 ? 0.0 : values.cwiseAbs().maxCoeff();
}

double relative_matrix_error(const Eigen::Ref<const Eigen::MatrixXd>& actual,
                             const Eigen::Ref<const Eigen::MatrixXd>& reference) {
    if (actual.rows() != reference.rows() || actual.cols() != reference.cols()) {
        return std::numeric_limits<double>::infinity();
    }
    const double error = actual.size() == 0 ? 0.0 : (actual - reference).cwiseAbs().maxCoeff();
    const double scale =
        reference.size() == 0 ? 1.0 : std::max(1.0, reference.cwiseAbs().maxCoeff());
    return error / scale;
}

struct CoupledReferenceAudit {
    double adjoint_relative_error = 0.0;
    double relaxed_relative_error = 0.0;
    int structure_coordinate_dimension = 0;
    int structure_response_rank = 0;
};

Eigen::VectorXd flatten_structure_tangent(const xmvb::vb::StructureTangent& tangent) {
    return Eigen::Map<const Eigen::VectorXd>(tangent.scaled_coefficients.data(),
                                             tangent.scaled_coefficients.size());
}

xmvb::vb::StructureTangent
unflatten_structure_tangent(const Eigen::Ref<const Eigen::VectorXd>& coefficients,
                            std::int64_t n_structures,
                            int n_states) {
    if (coefficients.size() != n_structures * n_states) {
        throw std::invalid_argument("coupled reference tangent has inconsistent dimensions");
    }
    return xmvb::vb::StructureTangent{
        Eigen::Map<const Eigen::MatrixXd>(coefficients.data(), n_structures, n_states)};
}

/** @brief Independently eliminates the small structure block of coupled NEO. */
CoupledReferenceAudit audit_coupled_structure_elimination(
    const xmvb::vb::ExactHvpOperator& exact_hvp,
    const std::shared_ptr<const xmvb::vb::AcceptedPointContext>& accepted,
    const Eigen::Ref<const Eigen::VectorXd>& orbital_direction,
    const Eigen::Ref<const Eigen::VectorXd>& relaxed_hvp,
    bool enforce) {
    const xmvb::vb::StructureTangentOperator structure(accepted, exact_hvp.structure_action());
    const std::int64_t n_structures = structure.n_structures();
    const int n_states = structure.n_states();
    const int dimension = n_structures * n_states;
    constexpr int kMaximumDenseReferenceDimension = 4096;
    if (dimension <= 0 || dimension > kMaximumDenseReferenceDimension) {
        throw std::runtime_error(
            "coupled dense reference dimension is outside the diagnostic limit");
    }
    const xmvb::vb::OrbitalCouplingAction orbital =
        exact_hvp.apply_orbital_coupling(orbital_direction);
    const xmvb::vb::StructureTangent forcing = structure.project(orbital.scaled_structure_forcing);
    const Eigen::VectorXd flat_forcing = flatten_structure_tangent(forcing);

    Eigen::MatrixXd structure_hessian(dimension, dimension);
    for (int coordinate = 0; coordinate < dimension; ++coordinate) {
        Eigen::VectorXd canonical = Eigen::VectorXd::Zero(dimension);
        canonical[coordinate] = 1.0;
        const auto tangent = structure.project(
            unflatten_structure_tangent(canonical, n_structures, n_states).scaled_coefficients);
        structure_hessian.col(coordinate) =
            flatten_structure_tangent(structure.apply_hessian(tangent));
    }
    const double structure_scale = std::max(1.0, structure_hessian.norm());
    const double symmetry_error =
        (structure_hessian - structure_hessian.transpose()).norm() / structure_scale;
    if (symmetry_error > 2.0e-10) {
        throw std::runtime_error("coupled reference structure Hessian is not symmetric");
    }
    structure_hessian = 0.5 * (structure_hessian + structure_hessian.transpose()).eval();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(structure_hessian);
    if (spectrum.info() != Eigen::Success) {
        throw std::runtime_error("coupled reference structure Hessian eigensolve failed");
    }
    const double eigenvalue_scale = std::max(1.0, spectrum.eigenvalues().cwiseAbs().maxCoeff());
    const double rank_tolerance = 1.0e3 * std::numeric_limits<double>::epsilon() *
                                  static_cast<double>(std::max<std::int64_t>(1, dimension)) *
                                  eigenvalue_scale;
    Eigen::VectorXd inverse_eigenvalues = Eigen::VectorXd::Zero(dimension);
    int response_rank = 0;
    for (int mode = 0; mode < dimension; ++mode) {
        const double eigenvalue = spectrum.eigenvalues()[mode];
        if (std::abs(eigenvalue) <= rank_tolerance)
            continue;
        inverse_eigenvalues[mode] = 1.0 / eigenvalue;
        ++response_rank;
    }
    const Eigen::VectorXd flat_response =
        -spectrum.eigenvectors() *
        (inverse_eigenvalues.asDiagonal() * (spectrum.eigenvectors().transpose() * flat_forcing));
    const xmvb::vb::StructureTangent response = structure.project(
        unflatten_structure_tangent(flat_response, n_structures, n_states).scaled_coefficients);
    const xmvb::vb::StructureCouplingAction response_action = structure.apply_coupling(response);
    const Eigen::VectorXd response_adjoint = exact_hvp.apply_structure_coupling_adjoint(
        response_action.coefficient_response, response_action.adjoint_multipliers);
    const Eigen::VectorXd eliminated_hvp = orbital.orbital_hessian + response_adjoint;

    Eigen::VectorXd probe_values(dimension);
    for (int coordinate = 0; coordinate < dimension; ++coordinate) {
        probe_values[coordinate] = std::sin(0.37 * static_cast<double>(coordinate + 1));
    }
    const xmvb::vb::StructureTangent probe = structure.project(
        unflatten_structure_tangent(probe_values, n_structures, n_states).scaled_coefficients);
    const xmvb::vb::StructureCouplingAction probe_action = structure.apply_coupling(probe);
    const Eigen::VectorXd probe_adjoint = exact_hvp.apply_structure_coupling_adjoint(
        probe_action.coefficient_response, probe_action.adjoint_multipliers);
    const double forward_bilinear = flatten_structure_tangent(probe).dot(flat_forcing);
    const double adjoint_bilinear = orbital_direction.dot(probe_adjoint);

    CoupledReferenceAudit audit;
    audit.adjoint_relative_error =
        std::abs(forward_bilinear - adjoint_bilinear) /
        std::max({1.0, std::abs(forward_bilinear), std::abs(adjoint_bilinear)});
    audit.relaxed_relative_error =
        infinity_norm(eliminated_hvp - relaxed_hvp) / std::max(1.0, infinity_norm(relaxed_hvp));
    audit.structure_coordinate_dimension = dimension;
    audit.structure_response_rank = response_rank;
    if (enforce && audit.adjoint_relative_error > 2.0e-9) {
        throw std::runtime_error(
            "coupled orbital-structure actions violate the bilinear adjoint identity");
    }
    if (enforce && audit.relaxed_relative_error > 2.0e-8) {
        throw std::runtime_error("coupled Schur elimination does not reproduce the relaxed HVP");
    }
    return audit;
}

/** @brief Loads an exact accepted-point orbital table for derivative checks. */
void overwrite_orbital_values_from_binary(const std::string& path,
                                          xmvb::vb::OrbitalPreparationInput* orbitals) {
    if (path.empty())
        return;
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const auto expected_bytes =
        static_cast<std::streamoff>(orbitals->orbital_value_table.size() * sizeof(double));
    if (!input || input.tellg() != expected_bytes) {
        throw std::runtime_error("orbital-value-table file size does not match input chart");
    }
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(orbitals->orbital_value_table.data()), expected_bytes);
    if (!input || !Eigen::Map<const Eigen::VectorXd>(orbitals->orbital_value_table.data(),
                                                     orbitals->orbital_value_table.size())
                       .allFinite()) {
        throw std::runtime_error("invalid orbital-value-table file");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_arguments(argc, argv);
        xmvb::vb::VbScfInputLoadOptions load_options;
        load_options.standard_two_electron_mode = options.two_electron_mode;
        auto loaded = xmvb::vb::load_vbscf_input_with_timings(options.input_path, load_options);
        auto& input = loaded.input;
        if (options.two_electron_mode == xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity) {
            // RI derivative checks must remain factor-native.  Removing any exact
            // graph supplied by a future loader change makes an accidental exact
            // fallback fail immediately instead of silently passing this test.
            input.ao_integral_input.pair_graph = {};
        }
        overwrite_orbital_values_from_binary(options.orbital_value_table_bin_path,
                                             &input.orbital_preparation_input);

        xmvb::vb::OrbitalGradientEvaluator evaluator;
        const Eigen::MatrixXd no_initial_eigenvectors;
        std::vector<int> selected_states(loaded.state_average_count);
        std::iota(selected_states.begin(), selected_states.end(), 0);
        const std::vector<double> equal_weights(selected_states.size(), 1.0);
        auto accepted = evaluator.evaluate_without_reference_energy_gradient(
            input,
            selected_states,
            equal_weights,
            loaded.nuclear_repulsion_energy,
            options.eigensolver,
            xmvb::vb::StructureSolveAccuracy{1.0e-7, options.response_tolerance},
            no_initial_eigenvectors);
        if (accepted.second_order_context == nullptr) {
            throw std::runtime_error("accepted-point second-order context is unavailable");
        }
        std::cout << std::setprecision(16)
                  << "accepted_total_energy = " << accepted.scf_result.total_energy
                  << std::setprecision(12) << std::endl;

        xmvb::vb::SparseParameterLayout layout(input.orbital_preparation_input);
        const int n_inactive = (input.orbital_preparation_input.n_total_electrons -
                                input.orbital_preparation_input.n_active_electrons) /
                               2;
        const int n_occupied = n_inactive + input.orbital_preparation_input.n_active_orbitals;
        const Eigen::MatrixXd& normalized_orbitals =
            accepted.orbital_preparation_result.physical_orbital_frame.normalized_orbital_matrix;
        if (normalized_orbitals.size() == 0) {
            throw std::runtime_error("accepted physical orbital frame is unavailable");
        }
        std::optional<xmvb::vb::AnalyticOrbitalDiagonal> analytic_diagonal;
        if (options.diagonal_samples > 0) {
            analytic_diagonal =
                xmvb::vb::build_analytic_orbital_diagonal(input, *accepted.second_order_context);
        }
        xmvb::vb::OrbitalChart chart(
            input.orbital_preparation_input,
            layout,
            accepted.orbital_preparation_result.auxiliary_orbital_matrix.leftCols(n_occupied),
            normalized_orbitals,
            analytic_diagonal ? nullptr
                              : &accepted.ao_effective_one_electron_result.ao_effective_h1e,
            false,
            input.complete_active_space,
            analytic_diagonal ? &*analytic_diagonal : nullptr,
            xmvb::vb::OrbitalChart::Coordinates::Unwhitened);
        const Eigen::VectorXd packed_gradient =
            layout.gather_from_full(accepted.sparse_orbital_energy_gradient);
        const Eigen::VectorXd accepted_reduced_gradient =
            chart.project_reduced_gradient(packed_gradient);
        // Diagnostic only: certify completeness modulo the FULL/OEO gauge, not
        // merely the total dimension. No global packed-by-reduced matrix is stored.
        if (options.audit_oeo_chart || !options.orbital_sector.empty()) {
            if (!chart.uses_active_subspace_gauge() ||
                layout.size() != input.orbital_preparation_input.n_basis_functions * n_occupied)
                throw std::invalid_argument("sector audit requires unrestricted FULL OEO");
            const int n_bf = input.orbital_preparation_input.n_basis_functions;
            const Eigen::VectorXd raw = layout.pack(input.orbital_preparation_input);
            Eigen::MatrixXd occupied(n_bf, n_occupied);
            for (int orbital = 0; orbital < n_occupied; ++orbital)
                for (int row = 0; row < n_bf; ++row)
                    occupied(row, orbital) = raw[layout.packed_index(orbital, row)];
            if (options.audit_oeo_chart) {
                std::vector<Eigen::MatrixXd> tangent(n_occupied, Eigen::MatrixXd::Zero(n_bf, n_bf));
                std::vector<int> columns(n_occupied, 0);
                for (int column = 0; column < chart.reduced_size(); ++column) {
                    Eigen::VectorXd unit = Eigen::VectorXd::Zero(chart.reduced_size());
                    unit[column] = 1.0;
                    const Eigen::VectorXd expanded = chart.expand_step(unit);
                    for (int orbital = 0; orbital < n_occupied; ++orbital) {
                        Eigen::VectorXd local(n_bf);
                        for (int row = 0; row < n_bf; ++row)
                            local[row] = expanded[layout.packed_index(orbital, row)];
                        if (!local.isZero(0.0))
                            tangent[orbital].col(columns[orbital]++) = local;
                    }
                }
                double gauge_derivative = 0.0;
                for (int orbital = 0; orbital < n_occupied; ++orbital) {
                    const int gauge_size = orbital < n_inactive ? n_inactive : n_occupied;
                    Eigen::MatrixXd basis(n_bf, gauge_size + columns[orbital]);
                    basis.leftCols(gauge_size) = occupied.leftCols(gauge_size);
                    basis.rightCols(columns[orbital]) = tangent[orbital].leftCols(columns[orbital]);
                    const int rank = basis.colPivHouseholderQr().rank();
                    Eigen::VectorXd local_gradient(n_bf);
                    for (int row = 0; row < n_bf; ++row)
                        local_gradient[row] = packed_gradient[layout.packed_index(orbital, row)];
                    for (int source = 0; source < gauge_size; ++source)
                        gauge_derivative = std::max<std::int64_t>(
                            gauge_derivative,
                            std::abs(local_gradient.dot(occupied.col(source))) /
                                occupied.col(source).norm());
                    std::cout << "chart_orbital=" << orbital << ",tangent=" << columns[orbital]
                              << ",gauge=" << gauge_size << ",combined_rank=" << rank << '\n';
                    if (rank != n_bf || columns[orbital] + gauge_size != n_bf)
                        throw std::runtime_error("FULL OEO chart loses a nongauge direction");
                }
                std::cout << "gauge_directional_derivative_max=" << gauge_derivative << '\n';
            }
        }
        Eigen::VectorXd direction = accepted_reduced_gradient;
        if (!options.orbital_sector.empty()) {
            const int n_bf = input.orbital_preparation_input.n_basis_functions;
            const auto& auxiliary = accepted.orbital_preparation_result.auxiliary_orbital_matrix;
            const bool active_target = options.orbital_sector == "active-external";
            const int target_begin = active_target ? n_inactive : 0;
            const int target_end = active_target ? n_occupied : n_inactive;
            const int source_begin =
                options.orbital_sector == "inactive-active" ? n_inactive : n_occupied;
            Eigen::VectorXd packed = Eigen::VectorXd::Zero(layout.size());
            for (int orbital = target_begin; orbital < target_end; ++orbital) {
                Eigen::VectorXd local = Eigen::VectorXd::Zero(n_bf);
                const int source_end =
                    options.orbital_sector == "inactive-active" ? n_occupied : auxiliary.cols();
                for (int source = source_begin; source < source_end; ++source)
                    local += std::sin(0.37 * (orbital + 1) * (source + 1)) * auxiliary.col(source);
                for (int row = 0; row < n_bf; ++row)
                    packed[layout.packed_index(orbital, row)] = local[row];
            }
            direction = chart.project_vector(packed).reduced_gradient;
            std::cout << "orbital_sector=" << options.orbital_sector
                      << ",retained_norm=" << direction.norm() << '\n';
        }
        if (options.preconditioned_direction) {
            direction = chart.apply_inverse_reduced_block_preconditioner(direction);
        }
        if (!options.packed_direction_path.empty()) {
            std::ifstream stream(options.packed_direction_path, std::ios::binary | std::ios::ate);
            const auto bytes = static_cast<std::streamoff>(layout.size() * sizeof(double));
            if (!stream || stream.tellg() != bytes)
                throw std::invalid_argument("packed direction file has wrong size");
            stream.seekg(0);
            Eigen::VectorXd packed(layout.size());
            stream.read(reinterpret_cast<char*>(packed.data()), bytes);
            if (!stream || !packed.allFinite())
                throw std::invalid_argument("invalid packed direction file");
            direction = chart.project_vector(packed).reduced_gradient;
        }
        if (direction.size() == 0) {
            throw std::runtime_error("nonredundant orbital space is empty");
        }
        const bool gradient_direction = !options.preconditioned_direction &&
                                        options.orbital_sector.empty() &&
                                        options.packed_direction_path.empty();
        // A gradient is a covector. Use its AO-tangent Riesz vector and norm so
        // coordinate rescaling does not change the finite-difference experiment.
        const double direction_norm = gradient_direction
                                          ? chart.coordinate_dual_norm(accepted_reduced_gradient)
                                          : direction.norm();
        if (gradient_direction)
            direction = chart.inverse_coordinate_metric(accepted_reduced_gradient);
        if (direction_norm <= std::sqrt(std::numeric_limits<double>::epsilon())) {
            direction.setZero();
            direction[0] = 1.0;
        } else {
            direction /= direction_norm;
        }

        if (options.stream_pair_products) {
            if (options.two_electron_mode != xmvb::vb::StandardTwoElectronMode::Exact) {
                throw std::invalid_argument(
                    "--stream-pair-products is only defined for exact AO integrals");
            }
            accepted.second_order_context->prepared_active_space.active_space_two_electron_result
                .dense_ao_pair_products.resize(0, 0);
        }
        if (options.compare_exchange_mib > 0) {
            if (options.two_electron_mode != xmvb::vb::StandardTwoElectronMode::Exact)
                throw std::invalid_argument("exchange comparison requires exact integrals");
            // Both operators borrow ONE integral graph, chart and accepted context.
            // Only the second operator constructs the additional exchange image.
            input.ao_integral_input.exact_exchange_memory_budget_bytes = 0;
            xmvb::vb::ExactHvpOperator reference(
                accepted.second_order_context, &input, layout, &chart);
            input.ao_integral_input.exact_exchange_memory_budget_bytes =
                static_cast<std::size_t>(options.compare_exchange_mib) * 1024 * 1024;
            xmvb::vb::ExactHvpOperator transformed(
                accepted.second_order_context, &input, layout, &chart);
            std::cout << "exchange_comparison_extra_resident_bytes = "
                      << (transformed.diagnostics().resident_exact_pair_elements -
                          reference.diagnostics().resident_exact_pair_elements) *
                             sizeof(double)
                      << std::endl;
            xmvb::vb::StructureTangentOperator structure(accepted.second_order_context,
                                                         reference.structure_action());
            Eigen::VectorXd z(structure.tangent_size());
            for (Eigen::Index i = 0; i < z.size(); ++i)
                z[i] = std::sin(0.37 * (i + 1));
            const auto tangent = structure.expand(z);
            const auto structure_image = structure.apply_coupling(tangent);
            const Eigen::MatrixXd& q = structure_image.coefficient_response;
            const Eigen::MatrixXd& multipliers = structure_image.adjoint_multipliers;
            const Eigen::VectorXd adjoint_ref =
                reference.apply_structure_coupling_adjoint(q, multipliers);
            const Eigen::VectorXd adjoint_new =
                transformed.apply_structure_coupling_adjoint(q, multipliers);
            const double adjoint_error = relative_matrix_error(adjoint_new, adjoint_ref);
            std::cout << "exchange_comparison_adjoint_relative_error = " << adjoint_error
                      << std::endl;
            double worst = adjoint_error;
            Eigen::VectorXd probe =
                chart.apply_inverse_reduced_block_preconditioner(accepted_reduced_gradient);
            for (int sample = 0; sample < 6; ++sample) {
                if (sample >= 1 && sample <= 3)
                    for (Eigen::Index i = 0; i < probe.size(); ++i)
                        probe[i] = std::sin((sample + 1) * 0.17 * (i + 1));
                probe.normalize();
                const auto ref = reference.apply_orbital_coupling(probe);
                const auto value = transformed.apply_orbital_coupling(probe);
                const double orbital_error =
                    relative_matrix_error(value.orbital_hessian, ref.orbital_hessian);
                const double forcing_error = relative_matrix_error(value.scaled_structure_forcing,
                                                                   ref.scaled_structure_forcing);
                const Eigen::VectorXd ref_preconditioned =
                    chart.apply_inverse_reduced_block_preconditioner(ref.orbital_hessian);
                const Eigen::VectorXd new_preconditioned =
                    chart.apply_inverse_reduced_block_preconditioner(value.orbital_hessian);
                const double preconditioned_error =
                    relative_matrix_error(new_preconditioned, ref_preconditioned);
                std::cout << std::setprecision(17) << "EXCHANGE_COMPARE,sample=" << sample
                          << ",orbital_inf="
                          << infinity_norm(value.orbital_hessian - ref.orbital_hessian)
                          << ",orbital_relative=" << orbital_error
                          << ",forcing_relative=" << forcing_error
                          << ",preconditioned_relative=" << preconditioned_error << std::endl;
                worst = std::max<std::int64_t>({worst, orbital_error, forcing_error});
                probe = ref_preconditioned;
            }
            if (!std::isfinite(worst) || worst > 1.0e-10)
                throw std::runtime_error("accepted-point exchange operator comparison failed");
            if (options.operator_timing_repeats > 0) {
                // Interleave both equivalent current-code operators on identical
                // directions. Warm both workspaces before measuring; never time setup.
                Eigen::VectorXd timed_direction =
                    chart.apply_inverse_reduced_block_preconditioner(-accepted_reduced_gradient);
                timed_direction.normalize();
                reference.apply_orbital_coupling(timed_direction);
                transformed.apply_orbital_coupling(timed_direction);
                const auto ref_before = reference.diagnostics();
                const auto new_before = transformed.diagnostics();
                double ref_seconds = 0.0, new_seconds = 0.0, checksum = 0.0;
                const auto time_call = [&](const auto& op, double& seconds) {
                    const auto start = std::chrono::steady_clock::now();
                    const auto image = op.apply_orbital_coupling(timed_direction);
                    seconds +=
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                            .count();
                    checksum += image.orbital_hessian.squaredNorm();
                };
                for (int repeat = 0; repeat < options.operator_timing_repeats; ++repeat) {
                    if (repeat % 2 == 0) {
                        time_call(reference, ref_seconds);
                        time_call(transformed, new_seconds);
                    } else {
                        time_call(transformed, new_seconds);
                        time_call(reference, ref_seconds);
                    }
                }
                const auto print_timing =
                    [&](const char* label, double seconds, const auto& before, const auto& after) {
                        const double calls = options.operator_timing_repeats;
                        std::cout << "OPERATOR_TIMING,path=" << label << ",calls=" << calls
                                  << ",wall_per_call=" << seconds / calls << ",ao_per_call="
                                  << (after.ao_effective_one_electron_fused_wall_time_seconds -
                                      before.ao_effective_one_electron_fused_wall_time_seconds) /
                                         calls
                                  << ",active_two_electron_per_call="
                                  << (after.active_two_electron_wall_time_seconds -
                                      before.active_two_electron_wall_time_seconds) /
                                         calls
                                  << ",setup_per_call="
                                  << (after.core_setup_wall_time_seconds -
                                      before.core_setup_wall_time_seconds) /
                                         calls
                                  << ",checksum=" << checksum << std::endl;
                    };
                print_timing("graph", ref_seconds, ref_before, reference.diagnostics());
                print_timing("exchange", new_seconds, new_before, transformed.diagnostics());
            }
            // The recorded first six microsteps have zero structure dimension and
            // zero shift. Reproduce that orbital Galerkin prefix on one shared chart
            // to measure error amplification, without taking an outer orbital step.
            xmvb::vb::NonredundantRetractionMetric metric(
                chart, layout, input.orbital_preparation_input);
            const auto prefix = [&](const xmvb::vb::ExactHvpOperator& op,
                                    const char* label,
                                    const Eigen::VectorXd& gradient) {
                Eigen::MatrixXd basis(direction.size(), 6), images(direction.size(), 6);
                Eigen::MatrixXd steps(direction.size(), 6);
                Eigen::VectorXd next = chart.apply_inverse_reduced_block_preconditioner(-gradient);
                for (int column = 0; column < 6; ++column) {
                    const double raw_direction_norm = next.stableNorm();
                    for (int pass = 0; pass < 2; ++pass) {
                        const Eigen::VectorXd weights = basis.leftCols(column).transpose() * next;
                        next.noalias() -= basis.leftCols(column) * weights;
                    }
                    const double orthogonal_norm = next.stableNorm();
                    next /= orthogonal_norm;
                    basis.col(column) = next;
                    images.col(column) = op.apply_orbital_coupling(next).orbital_hessian;
                    Eigen::MatrixXd projected =
                        basis.leftCols(column + 1).transpose() * images.leftCols(column + 1);
                    projected = (0.5 * (projected + projected.transpose())).eval();
                    const Eigen::VectorXd rhs = -basis.leftCols(column + 1).transpose() * gradient;
                    const Eigen::VectorXd coefficients = projected.ldlt().solve(rhs);
                    steps.col(column).noalias() = basis.leftCols(column + 1) * coefficients;
                    const Eigen::VectorXd residual =
                        gradient + images.leftCols(column + 1) * coefficients;
                    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(projected);
                    const double min_abs = spectrum.eigenvalues().cwiseAbs().minCoeff();
                    const double max_abs = spectrum.eigenvalues().cwiseAbs().maxCoeff();
                    std::cout << "EXCHANGE_PREFIX,path=" << label << ",micro=" << column + 1
                              << ",ro=" << metric.dual_norm(residual)
                              << ",step=" << metric.norm(steps.col(column))
                              << ",raw_direction_norm=" << raw_direction_norm
                              << ",retained=" << orthogonal_norm / raw_direction_norm
                              << ",residual_norm=" << residual.stableNorm()
                              << ",image_norm=" << images.col(column).stableNorm()
                              << ",projected_min_abs=" << min_abs
                              << ",projected_condition=" << max_abs / min_abs << std::endl;
                    next = chart.apply_inverse_reduced_block_preconditioner(-residual);
                }
                return steps;
            };
            const Eigen::MatrixXd prefix_ref =
                prefix(reference, "graph", accepted_reduced_gradient);
            const Eigen::MatrixXd prefix_new =
                prefix(transformed, "exchange", accepted_reduced_gradient);
            for (int column = 0; column < 6; ++column)
                std::cout << "EXCHANGE_PREFIX_DIFFERENCE,micro=" << column + 1 << ",step_norm="
                          << (prefix_new.col(column) - prefix_ref.col(column)).stableNorm()
                          << ",physical_step_norm="
                          << metric.norm(prefix_new.col(column) - prefix_ref.col(column))
                          << std::endl;
            // Controlled perturbation, not a reconstruction of an unrecorded old
            // gradient: isolate input sensitivity from the exchange representation.
            Eigen::VectorXd perturbation(direction.size());
            for (Eigen::Index i = 0; i < perturbation.size(); ++i)
                perturbation[i] = std::sin(0.31 * (i + 1));
            perturbation *= 1.0e-13 / perturbation.stableNorm();
            std::cout << "exchange_gradient_perturbation_norm = " << perturbation.stableNorm()
                      << std::endl;
            const Eigen::MatrixXd perturbed =
                prefix(transformed, "perturbed_gradient", accepted_reduced_gradient + perturbation);
            for (int column = 0; column < 6; ++column)
                std::cout << "EXCHANGE_INPUT_SENSITIVITY,micro=" << column + 1
                          << ",physical_step_difference="
                          << metric.norm(perturbed.col(column) - prefix_new.col(column))
                          << std::endl;
            return 0;
        }
        xmvb::vb::ExactHvpOperator exact_hvp(accepted.second_order_context, &input, layout, &chart);
        if (!options.projected_model_prefix.empty()) {
            const auto read_matrix = [](const std::string& path) {
                std::ifstream stream(path, std::ios::binary);
                std::int64_t shape[2] = {};
                stream.read(reinterpret_cast<char*>(shape), sizeof(shape));
                if (!stream || shape[0] < 1 || shape[0] > 100000 || shape[1] < 1 || shape[1] > 512)
                    throw std::runtime_error("invalid diagnostic basis dimensions");
                Eigen::MatrixXd matrix(shape[0], shape[1]);
                stream.read(reinterpret_cast<char*>(matrix.data()), matrix.size() * sizeof(double));
                if (!stream || !matrix.allFinite())
                    throw std::runtime_error("invalid diagnostic basis data");
                return matrix;
            };
            const Eigen::MatrixXd basis =
                read_matrix(options.projected_model_prefix + "-basis.bin");
            const Eigen::MatrixXd model =
                read_matrix(options.projected_model_prefix + "-model.bin");
            if (basis.rows() != chart.reduced_size() || model.rows() != basis.cols() ||
                model.cols() != basis.cols())
                throw std::runtime_error("diagnostic model dimensions do not match chart");
            const xmvb::vb::NonredundantRetractionMetric metric(
                chart, layout, input.orbital_preparation_input);
            Eigen::MatrixXd metric_images(basis.rows(), basis.cols());
            Eigen::MatrixXd images(basis.rows(), basis.cols());
            for (Eigen::Index column = 0; column < basis.cols(); ++column) {
                metric_images.col(column) = metric.apply(basis.col(column));
                images.col(column) = exact_hvp.apply_reduced(basis.col(column));
            }
            const Eigen::MatrixXd gram = basis.transpose() * metric_images;
            const Eigen::MatrixXd projected = basis.transpose() * images;
            const Eigen::MatrixXd symmetric = 0.5 * (projected + projected.transpose());
            Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> complete(symmetric, gram);
            Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> original(model, gram);
            if (complete.info() != Eigen::Success || original.info() != Eigen::Success)
                throw std::runtime_error("projected Schur audit eigensolver failed");
            const Eigen::VectorXd projected_mode =
                basis * gram.ldlt().solve(basis.transpose() * metric.apply(direction));
            std::cout << std::setprecision(14) << "audited_orbital_basis_size = " << basis.cols()
                      << '\n'
                      << "original_projected_min_curvature = " << original.eigenvalues()[0] << '\n'
                      << "complete_schur_projected_min_curvature = " << complete.eigenvalues()[0]
                      << '\n'
                      << "complete_schur_model_difference = " << (symmetric - model).norm() << '\n'
                      << "known_mode_metric_coverage = "
                      << metric.norm(projected_mode) / metric.norm(direction) << '\n'
                      << "projected_hessian_asymmetry = "
                      << (projected - projected.transpose()).norm() << '\n';
            return 0;
        }
        if (options.response_spectrum_limit > 0) {
            Eigen::MatrixXd forcing;
            if (input.complete_active_space && selected_states.size() == 1 &&
                accepted.second_order_context->n_structures >= 2 &&
                accepted.second_order_context->n_structures <= options.response_spectrum_limit) {
                forcing = -exact_hvp.apply_orbital_coupling(direction).scaled_structure_forcing;
            }
            xmvb::tools::audit_structure_response(exact_hvp.structure_action(),
                                                  *accepted.second_order_context,
                                                  input.complete_active_space,
                                                  options.response_spectrum_limit,
                                                  forcing);
            return 0;
        }
        if (options.curvature_steps > 0) {
            // Diagnostic only: seek a negative Rayleigh witness without forming H.
            // A nonnegative Ritz value in this incomplete space proves no stability.
            const int width = std::min<int>(options.curvature_steps, direction.size());
            Eigen::MatrixXd basis(direction.size(), width);
            Eigen::MatrixXd images(direction.size(), width);
            Eigen::MatrixXd block_images(direction.size(), width);
            std::mt19937_64 generator(42);
            std::normal_distribution<double> gaussian;
            Eigen::VectorXd next(direction.size());
            for (Eigen::Index row = 0; row < next.size(); ++row)
                next[row] = gaussian(generator);
            if (options.preconditioned_direction)
                next = chart.apply_inverse_reduced_block_preconditioner(next);
            for (int column = 0; column < width; ++column) {
                for (int pass = 0; pass < 2; ++pass) {
                    const Eigen::VectorXd weights = basis.leftCols(column).transpose() * next;
                    next.noalias() -= basis.leftCols(column) * weights;
                }
                const double norm = next.stableNorm();
                if (!(norm > 0.0) || !std::isfinite(norm))
                    break;
                basis.col(column) = next / norm;
                images.col(column) = exact_hvp.apply_reduced(basis.col(column));
                block_images.col(column) = options.preconditioned_direction
                                               ? chart.apply_reduced_curvature(basis.col(column))
                                               : Eigen::VectorXd(basis.col(column));
                Eigen::MatrixXd projected =
                    basis.leftCols(column + 1).transpose() * images.leftCols(column + 1);
                const double defect = (projected - projected.transpose()).stableNorm();
                projected = (0.5 * (projected + projected.transpose())).eval();
                Eigen::MatrixXd metric =
                    basis.leftCols(column + 1).transpose() * block_images.leftCols(column + 1);
                metric = (0.5 * (metric + metric.transpose())).eval();
                Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(projected,
                                                                                   metric);
                if (spectrum.info() != Eigen::Success)
                    throw std::runtime_error("curvature probe eigensolve failed");
                direction = basis.leftCols(column + 1) * spectrum.eigenvectors().col(0);
                const Eigen::VectorXd image =
                    images.leftCols(column + 1) * spectrum.eigenvectors().col(0);
                const Eigen::VectorXd block_image =
                    block_images.leftCols(column + 1) * spectrum.eigenvectors().col(0);
                next = image - spectrum.eigenvalues()[0] * block_image;
                std::cout << std::setprecision(16) << "CURVATURE_PROBE,actions=" << column + 1
                          << ",rayleigh=" << direction.dot(image)
                          << ",residual=" << next.stableNorm() << ",symmetry_defect=" << defect
                          << '\n';
                // Sign detection is a witness, not extremal-eigenvector convergence.
                const double scale = std::max(1.0, spectrum.eigenvalues().cwiseAbs().maxCoeff());
                if (direction.dot(image) <
                    -std::sqrt(std::numeric_limits<double>::epsilon()) * scale)
                    break;
                if (options.preconditioned_direction)
                    next = chart.apply_inverse_reduced_block_preconditioner(next);
            }
            // Reevaluate its full HVP and existing gradient/energy finite differences.
            direction.normalize();
        }
        const Eigen::VectorXd analytic = exact_hvp.apply_reduced(direction);
        const xmvb::vb::OrbitalPhysicalMetric physical_metric(input.orbital_preparation_input,
                                                              chart.uses_active_subspace_gauge());
        const double physical_direction_squared_norm =
            physical_metric.squared_norm(layout, chart.expand_step(direction));
        if (!(physical_direction_squared_norm > 0.0) ||
            !std::isfinite(physical_direction_squared_norm))
            throw std::runtime_error("HVP diagnostic direction has invalid physical norm");

        double coupled_orbital_fd_relative_error = 0.0;
        double coupled_structure_fd_relative_error = 0.0;
        double coupled_directional_curvature_relative_error = 0.0;
        if (options.coupled_finite_difference) {
            xmvb::vb::StructureTangentOperator structure(accepted.second_order_context,
                                                         exact_hvp.structure_action());
            Eigen::VectorXd structure_coordinates = Eigen::VectorXd::LinSpaced(
                structure.tangent_size(), 1.0, static_cast<double>(structure.tangent_size()));
            structure_coordinates = structure_coordinates.array().sin().matrix();
            xmvb::vb::StructureTangent structure_direction =
                structure.expand(structure_coordinates);
            const double structure_norm = std::sqrt(structure.squared_norm(structure_direction));
            if (!(structure_norm > 0.0) || !std::isfinite(structure_norm)) {
                throw std::runtime_error(
                    "coupled finite-difference structure direction is singular");
            }
            structure_coordinates /= structure_norm;
            structure_direction = structure.expand(structure_coordinates);

            xmvb::vb::CoupledHessianOperator coupled_hessian(exact_hvp, structure);
            const xmvb::vb::CoupledDirection joint_direction{direction, structure_direction};
            const xmvb::vb::CoupledDirection joint_analytic =
                coupled_hessian.apply(joint_direction);
            const Eigen::VectorXd analytic_structure =
                structure.coordinates(joint_analytic.structure);
            const Eigen::MatrixXd coefficient_direction =
                structure.coefficient_response(structure_direction);
            const Eigen::MatrixXd accepted_coefficients =
                accepted.second_order_context->selected_state_eigenvectors;

            const auto fixed_joint_gradient = [&](double step_scale) {
                xmvb::vb::VbScfInput displaced_input = input;
                displaced_input.orbital_preparation_input =
                    chart.retract_step(input.orbital_preparation_input, direction, step_scale);
                const Eigen::MatrixXd coefficients =
                    accepted_coefficients + step_scale * coefficient_direction;
                const auto fixed =
                    evaluator.evaluate_fixed_structure(displaced_input,
                                                       selected_states,
                                                       equal_weights,
                                                       loaded.nuclear_repulsion_energy,
                                                       coefficients);
                Eigen::VectorXd orbital = chart.project_reduced_gradient(
                    layout.gather_from_full(fixed.gradient.sparse_orbital_energy_gradient));
                Eigen::MatrixXd scaled_residuals = fixed.coefficient_gradient;
                for (Eigen::Index state = 0; state < scaled_residuals.cols(); ++state) {
                    scaled_residuals.col(state) /= std::sqrt(
                        2.0 * accepted.second_order_context
                                  ->normalized_state_weights[static_cast<std::size_t>(state)]);
                }
                return std::pair{std::move(orbital),
                                 structure.project_coordinates(scaled_residuals)};
            };
            const auto plus_joint = fixed_joint_gradient(options.step);
            const auto minus_joint = fixed_joint_gradient(-options.step);
            const Eigen::VectorXd fd_orbital =
                (plus_joint.first - minus_joint.first) / (2.0 * options.step);
            const Eigen::VectorXd fd_structure =
                (plus_joint.second - minus_joint.second) / (2.0 * options.step);
            coupled_orbital_fd_relative_error = infinity_norm(joint_analytic.orbital - fd_orbital) /
                                                std::max(1.0, infinity_norm(fd_orbital));
            coupled_structure_fd_relative_error = infinity_norm(analytic_structure - fd_structure) /
                                                  std::max(1.0, infinity_norm(fd_structure));
            const double analytic_curvature = direction.dot(joint_analytic.orbital) +
                                              structure_coordinates.dot(analytic_structure);
            const double finite_difference_curvature =
                direction.dot(fd_orbital) + structure_coordinates.dot(fd_structure);
            coupled_directional_curvature_relative_error =
                std::abs(analytic_curvature - finite_difference_curvature) /
                std::max(1.0, std::abs(finite_difference_curvature));
        }

        double diagonal_sample_max_relative_error = 0.0;
        Eigen::Index diagonal_sample_worst_coordinate = -1;
        double diagonal_sample_worst_analytic = 0.0;
        double diagonal_sample_worst_exact = 0.0;
        if (options.diagonal_samples > 0) {
            const Eigen::VectorXd diagonal = chart.augmented_hessian_diagonal();
            const int sample_count =
                std::min<int>(options.diagonal_samples, static_cast<int>(direction.size()));
            for (int sample = 0; sample < sample_count; ++sample) {
                const Eigen::Index coordinate =
                    sample_count == 1 ? 0
                                      : static_cast<Eigen::Index>(sample * (direction.size() - 1) /
                                                                  (sample_count - 1));
                Eigen::VectorXd basis = Eigen::VectorXd::Zero(direction.size());
                basis[coordinate] = 1.0;
                const double exact_diagonal =
                    exact_hvp.apply_orbital_coupling(basis).orbital_hessian[coordinate];
                const double scale = std::max(1.0, std::abs(exact_diagonal));
                const double error = std::abs(diagonal[coordinate] - exact_diagonal) / scale;
                if (error > diagonal_sample_max_relative_error) {
                    diagonal_sample_max_relative_error = error;
                    diagonal_sample_worst_coordinate = coordinate;
                    diagonal_sample_worst_analytic = diagonal[coordinate];
                    diagonal_sample_worst_exact = exact_diagonal;
                }
            }
        }

        Eigen::MatrixXd coupling_directions(direction.size(), 2);
        coupling_directions.col(0) = direction;
        Eigen::VectorXd second_direction = Eigen::VectorXd::LinSpaced(
            direction.size(), 1.0, static_cast<double>(direction.size()));
        second_direction.noalias() -= direction.dot(second_direction) * direction;
        if (second_direction.norm() <= std::sqrt(std::numeric_limits<double>::epsilon())) {
            second_direction = -0.375 * direction;
        } else {
            second_direction.normalize();
        }
        coupling_directions.col(1) = second_direction;
        const xmvb::vb::OrbitalCouplingBlockAction block_coupling =
            exact_hvp.apply_orbital_coupling_batch(coupling_directions);
        Eigen::MatrixXd scalar_orbital_coupling(direction.size(), 2);
        Eigen::MatrixXd orbital_only_action(direction.size(), 2);
        std::vector<Eigen::MatrixXd> scalar_structure_forcing(2);
        for (Eigen::Index column = 0; column < 2; ++column) {
            xmvb::vb::OrbitalCouplingAction scalar_coupling =
                exact_hvp.apply_orbital_coupling(coupling_directions.col(column));
            scalar_orbital_coupling.col(column) = scalar_coupling.orbital_hessian;
            orbital_only_action.col(column) =
                exact_hvp.apply_orbital(coupling_directions.col(column));
            scalar_structure_forcing[static_cast<std::size_t>(column)] =
                std::move(scalar_coupling.scaled_structure_forcing);
        }
        const double coupling_batch_orbital_error =
            relative_matrix_error(block_coupling.orbital_hessian, scalar_orbital_coupling);
        const double orbital_only_error =
            relative_matrix_error(orbital_only_action, scalar_orbital_coupling);
        if (orbital_only_error > 1.0e-10) {
            throw std::runtime_error(
                "orbital-only gauge disagrees with the coupled orbital action: " +
                std::to_string(orbital_only_error));
        }
        double coupling_batch_structure_error = 0.0;
        for (Eigen::Index column = 0; column < 2; ++column) {
            coupling_batch_structure_error = std::max<std::int64_t>(
                coupling_batch_structure_error,
                relative_matrix_error(
                    block_coupling.scaled_structure_forcing[static_cast<std::size_t>(column)],
                    scalar_structure_forcing[static_cast<std::size_t>(column)]));
        }
        if (coupling_batch_orbital_error > 1.0e-10 || coupling_batch_structure_error > 1.0e-10) {
            throw std::runtime_error("orbital coupling block disagrees with scalar actions");
        }
        Eigen::MatrixXd relaxed_images(direction.size(), 2);
        relaxed_images.col(0) = analytic;
        relaxed_images.col(1) = exact_hvp.apply_reduced(second_direction);
        const Eigen::Matrix2d projected_orbital =
            coupling_directions.transpose() * scalar_orbital_coupling;
        const Eigen::Matrix2d projected_relaxed = coupling_directions.transpose() * relaxed_images;
        const auto relative_symmetry_error = [](const Eigen::Matrix2d& matrix) {
            return (matrix - matrix.transpose()).norm() / std::max(1.0, matrix.norm());
        };
        const double orbital_symmetry_error = relative_symmetry_error(projected_orbital);
        const double relaxed_symmetry_error = relative_symmetry_error(projected_relaxed);
        const CoupledReferenceAudit coupled_audit =
            audit_coupled_structure_elimination(exact_hvp,
                                                accepted.second_order_context,
                                                direction,
                                                analytic,
                                                !options.coupled_finite_difference);
        const Eigen::VectorXd analytic_core =
            exact_hvp.apply_reduced(direction,
                                    {.direct_core_response = true,
                                     .fixed_upstream_pullback = true,
                                     .local_active_response = false,
                                     .structure_response = false});
        const Eigen::VectorXd analytic_local_active =
            exact_hvp.apply_reduced(direction,
                                    {.direct_core_response = false,
                                     .fixed_upstream_pullback = false,
                                     .local_active_response = true,
                                     .structure_response = false});
        const Eigen::VectorXd analytic_structure_response =
            exact_hvp.apply_reduced(direction,
                                    {.direct_core_response = false,
                                     .fixed_upstream_pullback = false,
                                     .local_active_response = false,
                                     .structure_response = true});
        const double component_decomposition_error =
            infinity_norm(analytic - analytic_core - analytic_local_active -
                          analytic_structure_response) /
            std::max(1.0, infinity_norm(analytic));
        if (!options.coupled_finite_difference && component_decomposition_error > 1.0e-10) {
            std::cerr << "component_decomposition_error = " << component_decomposition_error
                      << '\n';
            throw std::runtime_error("exact HVP component decomposition is inconsistent");
        }
        const auto hvp_diagnostics = exact_hvp.diagnostics();
        if (hvp_diagnostics.orbital_coupling_batch_count != 1 ||
            hvp_diagnostics.orbital_coupling_batch_chunk_count < 1 ||
            hvp_diagnostics.max_orbital_coupling_batch_width < 1 ||
            hvp_diagnostics.max_orbital_coupling_batch_width > 2) {
            throw std::runtime_error("orbital coupling block diagnostics are inconsistent");
        }

        // A local-only block has no cached orbital/structure forcing direction.
        // Its reverse consumer must still receive the exact integral tangent.
        const xmvb::vb::HvpComponents local_only{.direct_core_response = false,
                                                 .fixed_upstream_pullback = false,
                                                 .local_active_response = true,
                                                 .structure_response = false};
        const Eigen::MatrixXd local_block =
            exact_hvp.apply_reduced_batch(coupling_directions, local_only);
        Eigen::MatrixXd local_scalar(direction.size(), 2);
        local_scalar.col(0) = analytic_local_active;
        local_scalar.col(1) = exact_hvp.apply_reduced(second_direction, local_only);
        const double local_batch_error = relative_matrix_error(local_block, local_scalar);
        if (local_batch_error > 1e-10)
            throw std::runtime_error("local-only HVP block differs from scalar action");

        double state_energy_average_error = 0.0;
        double state_gradient_average_error = 0.0;
        double state_hvp_average_error = 0.0;
        if (selected_states.size() > 1) {
            double state_energy_average = 0.0;
            Eigen::VectorXd state_gradient_average =
                Eigen::VectorXd::Zero(accepted_reduced_gradient.size());
            Eigen::VectorXd state_hvp_average = Eigen::VectorXd::Zero(analytic.size());
            for (std::size_t offset = 0; offset < selected_states.size(); ++offset) {
                const auto state = evaluator.evaluate_without_reference_energy_gradient(
                    input,
                    {selected_states[offset]},
                    {1.0},
                    loaded.nuclear_repulsion_energy,
                    options.eigensolver,
                    xmvb::vb::StructureSolveAccuracy{1.0e-7, options.response_tolerance},
                    no_initial_eigenvectors);
                const double weight =
                    accepted.second_order_context->normalized_state_weights[offset];
                state_energy_average += weight * state.scf_result.total_energy;
                state_gradient_average.noalias() +=
                    weight * chart.project_reduced_gradient(
                                 layout.gather_from_full(state.sparse_orbital_energy_gradient));
                xmvb::vb::ExactHvpOperator state_hvp(
                    state.second_order_context, &input, layout, &chart);
                state_hvp_average.noalias() += weight * state_hvp.apply_reduced(direction);
            }
            state_energy_average_error =
                std::abs(accepted.scf_result.total_energy - state_energy_average);
            state_gradient_average_error =
                infinity_norm(accepted_reduced_gradient - state_gradient_average);
            state_hvp_average_error = infinity_norm(analytic - state_hvp_average);
            if (state_energy_average_error > 1.0e-10 || state_gradient_average_error > 1.0e-9 ||
                state_hvp_average_error > 1.0e-8) {
                std::cerr << "state_energy_average_error = " << state_energy_average_error
                          << "\nstate_gradient_average_error = " << state_gradient_average_error
                          << "\nstate_hvp_average_error = " << state_hvp_average_error << '\n';
                throw std::runtime_error("equal-weight state-average linearity check failed");
            }
        }

        xmvb::vb::VbScfInput displaced = input;
        displaced.orbital_preparation_input =
            chart.retract_step(input.orbital_preparation_input, direction, options.step);
        const auto evaluate_reduced_gradient = [&](const xmvb::vb::VbScfInput& point) {
            const auto point_gradient = evaluator.evaluate_without_reference_energy_gradient(
                point,
                accepted.second_order_context->selected_state_indices,
                accepted.second_order_context->normalized_state_weights,
                loaded.nuclear_repulsion_energy);
            return std::pair{chart.project_reduced_gradient(layout.gather_from_full(
                                 point_gradient.sparse_orbital_energy_gradient)),
                             point_gradient.scf_result.total_energy};
        };
        const auto [plus_reduced, plus_energy] = evaluate_reduced_gradient(displaced);
        const Eigen::VectorXd plus_core_reduced =
            chart.project_reduced_gradient(layout.gather_from_full(
                evaluator.evaluate_sparse_orbital_gradient_with_fixed_active_space_adjoint(
                    displaced, *accepted.second_order_context, loaded.nuclear_repulsion_energy)));

        displaced.orbital_preparation_input =
            chart.retract_step(input.orbital_preparation_input, direction, -options.step);
        const auto [minus_reduced, minus_energy] = evaluate_reduced_gradient(displaced);
        const Eigen::VectorXd minus_core_reduced =
            chart.project_reduced_gradient(layout.gather_from_full(
                evaluator.evaluate_sparse_orbital_gradient_with_fixed_active_space_adjoint(
                    displaced, *accepted.second_order_context, loaded.nuclear_repulsion_energy)));
        const Eigen::VectorXd finite_difference =
            (plus_reduced - minus_reduced) / (2.0 * options.step);
        const Eigen::VectorXd dense_midpoint_reduced_gradient =
            0.5 * (plus_reduced + minus_reduced);
        const double accepted_gradient_vs_dense_midpoint_inf =
            infinity_norm(accepted_reduced_gradient - dense_midpoint_reduced_gradient);
        const Eigen::VectorXd finite_difference_core =
            (plus_core_reduced - minus_core_reduced) / (2.0 * options.step);
        const Eigen::VectorXd analytic_outer = analytic - analytic_core;
        const Eigen::VectorXd finite_difference_outer = finite_difference - finite_difference_core;
        const double analytic_directional_curvature = direction.dot(analytic);
        const double gradient_fd_directional_curvature = direction.dot(finite_difference);
        const double energy_directional_curvature =
            (plus_energy - 2.0 * accepted.scf_result.total_energy + minus_energy) /
            (options.step * options.step);
        const double energy_curvature_relative_error =
            std::abs(analytic_directional_curvature - energy_directional_curvature) /
            std::max(1.0, std::abs(energy_directional_curvature));

        const double max_abs_difference = infinity_norm(analytic - finite_difference);
        const Eigen::VectorXd packed_tangent = chart.expand_step(direction);
        const double raw_derivative = packed_gradient.dot(packed_tangent);
        const double reduced_derivative = accepted_reduced_gradient.dot(direction);
        const auto plus_parameters = layout.pack(
            chart.retract_step(input.orbital_preparation_input, direction, options.step));
        const auto minus_parameters = layout.pack(
            chart.retract_step(input.orbital_preparation_input, direction, -options.step));
        std::cout << "chart_derivative_adjoint_error = "
                  << std::abs(raw_derivative - reduced_derivative) << '\n'
                  << "retraction_tangent_relative_error = "
                  << ((plus_parameters - minus_parameters) / (2.0 * options.step) - packed_tangent)
                             .norm() /
                         std::max(1.0, packed_tangent.norm())
                  << '\n'
                  << "gradient_energy_fd_absolute_error = "
                  << std::abs((plus_energy - minus_energy) / (2.0 * options.step) -
                              reduced_derivative)
                  << '\n';
        const double relative_error =
            max_abs_difference / std::max(1.0, infinity_norm(finite_difference));
        const double core_relative_error = infinity_norm(analytic_core - finite_difference_core) /
                                           std::max(1.0, infinity_norm(finite_difference_core));
        const double outer_relative_error =
            infinity_norm(analytic_outer - finite_difference_outer) /
            std::max(1.0, infinity_norm(finite_difference_outer));
        std::cout
            << std::setprecision(12) << "input = " << options.input_path << '\n'
            << "orbital_value_table_override = "
            << (options.orbital_value_table_bin_path.empty() ? "none"
                                                             : options.orbital_value_table_bin_path)
            << '\n'
            << "stream_pair_products = " << (options.stream_pair_products ? "true" : "false")
            << '\n'
            << "eigensolver = " << xmvb::vb::structure_eigensolver_name(options.eigensolver) << '\n'
            << "direction = "
            << (options.curvature_steps > 0        ? "curvature_probe"
                : options.preconditioned_direction ? "preconditioned_gradient"
                                                   : "gradient")
            << '\n'
            << "reduced_dimension = " << direction.size() << '\n'
            << "finite_difference_step = " << options.step << '\n'
            << "coupled_finite_difference = "
            << (options.coupled_finite_difference ? "true" : "false") << '\n'
            << "coupled_orbital_fd_relative_error = " << coupled_orbital_fd_relative_error << '\n'
            << "coupled_structure_fd_relative_error = " << coupled_structure_fd_relative_error
            << '\n'
            << "coupled_directional_curvature_relative_error = "
            << coupled_directional_curvature_relative_error << '\n'
            << "response_tolerance = " << options.response_tolerance << '\n'
            << "diagonal_sample_max_relative_error = " << diagonal_sample_max_relative_error << '\n'
            << "diagonal_sample_worst_coordinate = " << diagonal_sample_worst_coordinate << '\n'
            << "diagonal_sample_worst_analytic = " << diagonal_sample_worst_analytic << '\n'
            << "diagonal_sample_worst_exact = " << diagonal_sample_worst_exact << '\n'
            << "state_average_count = " << selected_states.size() << '\n'
            << "state_energy_average_error = " << state_energy_average_error << '\n'
            << "state_gradient_average_inf_error = " << state_gradient_average_error << '\n'
            << "state_hvp_average_inf_error = " << state_hvp_average_error << '\n'
            << "component_decomposition_error = " << component_decomposition_error << '\n'
            << "orbital_only_error = " << orbital_only_error << '\n'
            << "coupling_batch_orbital_error = " << coupling_batch_orbital_error << '\n'
            << "coupling_batch_structure_error = " << coupling_batch_structure_error << '\n'
            << "orbital_coupling_symmetry_error = " << orbital_symmetry_error << '\n'
            << "relaxed_hvp_symmetry_error = " << relaxed_symmetry_error << '\n'
            << "coupling_batch_chunks = " << hvp_diagnostics.orbital_coupling_batch_chunk_count
            << '\n'
            << "coupling_batch_max_width = " << hvp_diagnostics.max_orbital_coupling_batch_width
            << '\n'
            << "coupled_structure_coordinate_dimension = "
            << coupled_audit.structure_coordinate_dimension << '\n'
            << "coupled_structure_response_rank = " << coupled_audit.structure_response_rank << '\n'
            << "coupled_adjoint_relative_error = " << coupled_audit.adjoint_relative_error << '\n'
            << "coupled_relaxed_hvp_relative_error = " << coupled_audit.relaxed_relative_error
            << '\n'
            << "accepted_gradient_vs_dense_midpoint_inf = "
            << accepted_gradient_vs_dense_midpoint_inf << '\n'
            << "structure_response_iterations = "
            << hvp_diagnostics.max_structure_response_iterations << '\n'
            << "structure_response_relative_residual = "
            << hvp_diagnostics.max_structure_response_relative_residual << '\n'
            << "analytic_inf_norm = " << infinity_norm(analytic) << '\n'
            << "fd_inf_norm = " << infinity_norm(finite_difference) << '\n'
            << "max_abs_diff = " << max_abs_difference << '\n'
            << "max_rel_diff = " << relative_error << '\n'
            << "analytic_directional_curvature = " << analytic_directional_curvature << '\n'
            << "physical_direction_squared_norm = " << physical_direction_squared_norm << '\n'
            << "physical_rayleigh_curvature = "
            << analytic_directional_curvature / physical_direction_squared_norm << '\n'
            << "accepted_coordinate_gradient_norm = " << accepted_reduced_gradient.norm() << '\n'
            << "accepted_gradient_dot_direction = " << accepted_reduced_gradient.dot(direction)
            << '\n'
            << "gradient_fd_directional_curvature = " << gradient_fd_directional_curvature << '\n'
            << "energy_fd_directional_curvature = " << energy_directional_curvature << '\n'
            << "energy_curvature_max_rel_diff = " << energy_curvature_relative_error << '\n'
            << "core_analytic_inf_norm = " << infinity_norm(analytic_core) << '\n'
            << "core_fd_inf_norm = " << infinity_norm(finite_difference_core) << '\n'
            << "core_max_rel_diff = " << core_relative_error << '\n'
            << "core_analytic_directional_curvature = " << direction.dot(analytic_core) << '\n'
            << "core_gradient_fd_directional_curvature = " << direction.dot(finite_difference_core)
            << '\n'
            << "outer_analytic_inf_norm = " << infinity_norm(analytic_outer) << '\n'
            << "outer_fd_inf_norm = " << infinity_norm(finite_difference_outer) << '\n'
            << "outer_max_rel_diff = " << outer_relative_error << '\n'
            << "outer_analytic_directional_curvature = " << direction.dot(analytic_outer) << '\n'
            << "outer_gradient_fd_directional_curvature = "
            << direction.dot(finite_difference_outer) << '\n';
        if (relative_error > options.max_relative_error) {
            throw std::runtime_error("exact HVP finite-difference tolerance exceeded");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
