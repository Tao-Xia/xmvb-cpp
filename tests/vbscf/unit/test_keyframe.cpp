#include <algorithm>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/resource.h>

#include "vbscf/derivatives/gradient/orbital/evaluator.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "integrals/ao/graph.hpp"
#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/layout.hpp"
#include "vbscf/structures/expansion/expander.hpp"
#include "vbscf/optimization/backends/ciah_step.hpp"
#include "vbscf/optimization/driver/options.hpp"
#include "vbscf/orbitals/preparation/preparer.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

/** @brief A terminal trial must meet both unchanged criteria at its own point. */
void check_terminal_trial() {
    using namespace xmvb::vb;
    OrbitalPreparationInput orbitals;
    orbitals.n_basis_functions = 3;
    orbitals.n_orbitals = orbitals.n_active_orbitals = 2;
    orbitals.n_total_electrons = orbitals.n_active_electrons = 2;
    orbitals.orbital_value_table = {1.0, 0.13, -0.17, 0.21, 0.94, 0.12};
    orbitals.orbital_basis_index_table = {1, 2, 3, 1, 2, 3};
    orbitals.orbital_basis_counts = {3, 3};
    orbitals.ao_overlap_matrix = Eigen::Matrix3d::Identity();
    const SparseParameterLayout layout(orbitals);
    const VbScfOptimizerOptions options;
    for (bool complete : {false, true}) {
        optimizer_detail::CiahMacroStep point;
        point.trial.orbital_preparation_input = orbitals;
        point.trial.gradient_result.orbital_preparation_result =
            ActiveSpaceOrbitalPreparer{}.prepare(orbitals);
        point.trial.gradient_ready = true;
        point.trial.gradient = Eigen::VectorXd::Zero(layout.pack(orbitals).size());
        const double origin = -399.397232509251;
        for (double energy : {origin,
                              std::nextafter(origin, 0.0),
                              std::nextafter(origin, -std::numeric_limits<double>::infinity())}) {
            point.trial.energy = energy;
            require(point.check_convergence(layout, options, complete, origin) &&
                        point.converged_geometry.has_value(),
                    "terminal trial depends on the sign of a sub-threshold energy change");
        }
        point.trial.energy = origin + 2.0 * options.energy_tolerance;
        require(!point.check_convergence(layout, options, complete, origin) &&
                    !point.converged_geometry.has_value(),
                "small gradient accepted an excessive energy change");
        point.trial.energy = options.energy_tolerance;
        require(!point.check_convergence(layout, options, complete, 0.0),
                "CIAH changed the strict energy convergence threshold");
        point.trial.energy = origin;
        point.trial.gradient.setOnes();
        require(!point.check_convergence(layout, options, complete, origin) &&
                    !point.converged_geometry.has_value(),
                "equal energy substituted for a small true trial gradient");
        point.trial.gradient.setZero();
        point.trial.gradient_ready = false;
        bool rejected = false;
        try {
            point.check_convergence(layout, options, complete, origin);
        } catch (const std::logic_error&) {
            rejected = true;
        }
        require(rejected, "energy-only candidate was certified without its true gradient");
    }
}

/** @brief Recycled CI directions transform with a relaxed root's arbitrary phase. */
void check_restart_phase() {
    const auto require_close = [](const Eigen::VectorXd& actual,
                                  const Eigen::VectorXd& expected,
                                  double tolerance,
                                  const char* message) {
        require((actual - expected).stableNorm() <=
                    tolerance * std::max(1.0, expected.stableNorm()),
                message);
    };
    using xmvb::vb::optimizer_detail::CiahMacroStep;
    const Eigen::Vector3d reference(0.8, 0.4, 0.1);
    Eigen::Matrix3d metric;
    metric << 1.0, 0.2, 0.1, 0.2, 2.0, 0.3, 0.1, 0.3, 3.0;
    const Eigen::Vector3d root(0.7, 0.5, 0.2);
    const Eigen::Vector3d response(0.1, -0.2, 0.05);
    CiahMacroStep positive, negative;
    positive.next_structure_guess = response;
    negative.next_structure_guess = response;
    positive.next_orbital_guess_packed = reference;
    negative.next_orbital_guess_packed = reference;
    positive.align_structure_frame(reference, metric * root);
    negative.align_structure_frame(reference, -metric * root);
    require_close(positive.next_structure_guess,
                  response,
                  1.0e-14,
                  "CIAH changed a phase-consistent structure seed");
    require_close(negative.next_structure_guess,
                  -response,
                  1.0e-14,
                  "CIAH failed to reverse the seed with the relaxed root");
    require_close(positive.next_orbital_guess_packed,
                  negative.next_orbital_guess_packed,
                  1.0e-14,
                  "CIAH phase alignment changed the orbital direction");
    // B changes sign with the CI root; the coupled orbital image is invariant.
    Eigen::Matrix3d coupling;
    coupling << 2.0, 0.3, 0.1, 0.2, 1.0, 0.4, 0.3, -0.1, 0.7;
    require_close(coupling * positive.next_structure_guess,
                  -coupling * negative.next_structure_guess,
                  1.0e-14,
                  "CIAH restart violates coupled-action phase covariance");
    const Eigen::MatrixXd frame = Eigen::Matrix3d::Identity().leftCols(2);
    Eigen::Matrix2d rotation;
    rotation << std::cos(0.7), -std::sin(0.7), std::sin(0.7), std::cos(0.7);
    CiahMacroStep block;
    block.next_structure_guess = frame;
    block.align_structure_frame(frame, frame * rotation);
    require((block.next_structure_guess - frame * rotation).norm() < 1e-14,
            "CIAH state-frame alignment fails an equal-weight root rotation");
}

/** @brief Complete accepted-gradient producer, not only a StructureAction consumer. */
void benchmark_accepted_state16(bool check_hvp = false) {
    constexpr int n = 16;
    xmvb::vb::VbScfInput input;
    auto& orbitals = input.orbital_preparation_input;
    orbitals.n_basis_functions = orbitals.n_orbitals = orbitals.n_active_orbitals = n;
    orbitals.n_total_electrons = orbitals.n_active_electrons = n;
    orbitals.orbital_basis_counts.assign(n, n);
    orbitals.orbital_value_table.assign(n * n, 0.0);
    orbitals.orbital_basis_index_table.resize(n * n);
    orbitals.ao_overlap_matrix = Eigen::MatrixXd::Identity(n, n);
    orbitals.ao_overlap_matrix.array() += 0.01;
    for (int i = 0; i < n; ++i) {
        orbitals.orbital_value_table[i * n + i] = 1.0;
        for (int j = 0; j < n; ++j)
            orbitals.orbital_basis_index_table[i * n + j] = j + 1;
    }
    auto& ao = input.ao_integral_input;
    ao.n_basis_functions = n;
    ao.ao_core_hamiltonian_matrix = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i)
        ao.ao_core_hamiltonian_matrix(i, i) = -0.9 + 0.03 * i;
    std::vector<int> left, right;
    std::vector<double> values;
    for (int p = 0; p < n * (n + 1) / 2; ++p)
        for (int q = 0; q <= p; ++q) {
            left.push_back(p);
            right.push_back(q);
            values.push_back(0.02 * std::cos(0.1 * p) * std::cos(0.1 * q) +
                             0.01 * std::sin(0.07 * p) * std::sin(0.07 * q));
        }
    ao.pair_graph = xmvb::integrals::build_ao_pair_graph(left, right, values, n);
    xmvb::vb::RawStructureData raw;
    raw.n_structures = 1;
    raw.n_total_electrons = raw.n_active_electrons = n;
    for (int i = 1; i <= n; ++i)
        raw.raw_structure_orbitals.push_back(i);
    input.structure_data = xmvb::vb::StructureExpander().expand(raw);
    const auto start = std::chrono::steady_clock::now();
    const auto gradient =
        xmvb::vb::OrbitalGradientEvaluator().evaluate_without_reference_energy_gradient(
            input,
            {0},
            {1.0},
            0.0,
            xmvb::vb::StructureEigensolver::Davidson,
            {1e-7, 1e-3},
            Eigen::MatrixXd{});
    const double gradient_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const auto& accepted = *gradient.second_order_context;
    const auto& state = *accepted.structure_adjoint_state;
    require(accepted.selected_state_matrices.states.empty() && state.source_factors &&
                !state.product_images,
            "accepted-gradient producer retained a full product state");
    std::size_t raw_doubles = 0;
    for (const auto& s : accepted.selected_state_matrices.states)
        raw_doubles += s.coefficient_matrix.size();
    double hvp_seconds = 0, hvp_fd_error = 0, hvp_norm = 0;
    if (check_hvp) {
        const xmvb::vb::SparseParameterLayout layout(orbitals);
        const auto& prepared = gradient.orbital_preparation_result;
        const xmvb::vb::OrbitalChart chart(
            orbitals,
            layout,
            prepared.auxiliary_orbital_matrix.leftCols(n),
            prepared.physical_orbital_frame.normalized_orbital_matrix,
            nullptr,
            false,
            false,
            nullptr,
            xmvb::vb::OrbitalChart::Coordinates::Unwhitened);
        Eigen::VectorXd p(chart.reduced_size());
        for (int i = 0; i < p.size(); ++i)
            p[i] = std::sin(0.17 * (i + 1));
        p.normalize();
        const xmvb::vb::ExactHvpOperator hessian(
            gradient.second_order_context, &input, layout, &chart);
        const auto hvp_start = std::chrono::steady_clock::now();
        const auto image = hessian.apply_orbital_coupling(p);
        hvp_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - hvp_start).count();
        hvp_norm = image.orbital_hessian.norm();
        const auto stages = hessian.diagnostics();
        std::cout << std::setprecision(17) << "hvp_integral_direction_seconds="
                  << stages.outer_response_active_space_integrals_wall_time_seconds
                  << "\nhvp_selected_direction_seconds="
                  << stages.outer_response_structure_matrices_wall_time_seconds
                  << "\nhvp_eigen_response_seconds="
                  << stages.outer_response_eigensystem_wall_time_seconds
                  << "\nhvp_active_backward_seconds="
                  << stages.outer_response_active_gradient_wall_time_seconds
                  << "\nhvp_response_pullback_seconds="
                  << stages.outer_response_orbital_pullback_wall_time_seconds
                  << "\nhvp_raw_response_rebuild_seconds="
                  << stages.outer_response_selected_state_rebuild_wall_time_seconds
                  << "\nnormalization_adjoint_reuses=" << stages.normalization_adjoint_reuses
                  << '\n'
                  << std::flush;
        require(stages.normalization_adjoint_reuses == 1,
                "production HVP did not reuse the known normalization response");
        require(stages.outer_response_selected_state_rebuild_wall_time_seconds == 0,
                "production direct-CI HVP rebuilt a raw product response bundle");
        require(image.orbital_hessian.allFinite() && image.scaled_structure_forcing.allFinite(),
                "production source HVP has a nonfinite orbital or structure image");
        // A single structure has no horizontal CI tangent; normalization/gauge
        // response is still required. Differentiate in the same linear raw lift.
        constexpr double step = 1e-4;
        const Eigen::VectorXd origin = layout.pack(orbitals), tangent = chart.expand_step(p);
        const auto shifted_gradient = [&](double scale) -> Eigen::VectorXd {
            auto moved = input;
            layout.unpack(origin + scale * tangent, &moved.orbital_preparation_input);
            const auto value =
                xmvb::vb::OrbitalGradientEvaluator().evaluate_without_reference_energy_gradient(
                    moved,
                    {0},
                    {1.0},
                    0.0,
                    xmvb::vb::StructureEigensolver::Davidson,
                    {1e-7, 1e-3},
                    Eigen::MatrixXd{});
            return chart.project_reduced_gradient(
                layout.gather_from_full(value.sparse_orbital_energy_gradient));
        };
        const Eigen::VectorXd plus = shifted_gradient(step), minus = shifted_gradient(-step);
        hvp_fd_error = ((plus - minus) / (2 * step) - image.orbital_hessian).norm();
        require(hvp_fd_error < 2e-7 * std::max(1.0, hvp_norm),
                "production source HVP differs from the relaxed-gradient derivative");
        require(accepted.selected_state_matrices.states.empty(),
                "production HVP repopulated the accepted product cache");
    }
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    const auto& cache = accepted.same_spin_pair_cache;
    const std::size_t removed_bytes =
        static_cast<std::size_t>(cache.n_unique_alpha) * cache.n_unique_beta * sizeof(double);
    require(std::isfinite(gradient.scf_result.total_energy) &&
                static_cast<std::size_t>(usage.ru_maxrss) * 1024 < removed_bytes,
            "accepted-gradient peak RSS exceeds the removed raw-product allocation");
    std::cout << std::setprecision(17) << "orbitals=" << n << "\nstructures=1\n"
              << "alpha_strings=" << cache.n_unique_alpha
              << "\nbeta_strings=" << cache.n_unique_beta
              << "\naccepted_raw_product_doubles=" << raw_doubles
              << "\naccepted_orthogonal_product_doubles=0\naccepted_residual_product_doubles=0\n"
              << "removed_raw_product_bytes=" << removed_bytes
              << "\nenergy_Eh=" << gradient.scf_result.total_energy
              << "\ngradient_seconds=" << gradient_seconds << "\nhvp_seconds=" << hvp_seconds
              << "\nhvp_norm=" << hvp_norm << "\nhvp_fd_error=" << hvp_fd_error
              << "\ntotal_seconds="
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
              << "\nrss_mib=" << usage.ru_maxrss / 1024.0 << '\n';
}

/** @brief Scalar/batched coupled orbital actions reuse a known state frame. */
void check_frame_hvp(const xmvb::vb::VbScfInput& input, int n_states) {
    const xmvb::vb::OrbitalGradientEvaluator evaluator;
    std::vector<int> states{0};
    if (n_states == 2)
        states.push_back(input.structure_data.n_structures - 1);
    const std::vector<double> weights(n_states, 1.0 / n_states);
    const auto accepted = evaluator.evaluate(input, states, weights, 0.4);
    const auto& orbitals = input.orbital_preparation_input;
    const xmvb::vb::SparseParameterLayout layout(orbitals);
    const auto& prepared = accepted.orbital_preparation_result;
    const xmvb::vb::OrbitalChart chart(
        orbitals,
        layout,
        prepared.auxiliary_orbital_matrix.leftCols(orbitals.n_orbitals),
        prepared.physical_orbital_frame.normalized_orbital_matrix,
        nullptr,
        false,
        false,
        nullptr,
        xmvb::vb::OrbitalChart::Coordinates::Unwhitened);
    Eigen::MatrixXd directions(chart.reduced_size(), 2);
    for (int i = 0; i < directions.rows(); ++i) {
        directions(i, 0) = std::sin(0.3 * (i + 1));
        directions(i, 1) = std::cos(0.7 * (i + 1));
    }
    directions.colwise().normalize();
    const xmvb::vb::ExactHvpOperator hessian(accepted.second_order_context, &input, layout, &chart);
    const auto block = hessian.apply_orbital_coupling_batch(directions);
    const auto scalar = hessian.apply_orbital_coupling(directions.col(0));
    const auto orbital = hessian.apply_orbital(directions.col(0));
    require((block.orbital_hessian.col(0) - scalar.orbital_hessian).norm() < 3e-11 &&
                (orbital - scalar.orbital_hessian).norm() < 3e-11 &&
                (block.scaled_structure_forcing[0] - scalar.scaled_structure_forcing).norm() <
                    3e-11,
            "normalization reuse differs between scalar and batched HVPs");
    const auto reuses = hessian.diagnostics().normalization_adjoint_reuses;
    require(n_states == 1 ? reuses >= 4 : reuses == 0,
            "scalar/batched state-frame reuse used the wrong root multiplicity");
    require(std::abs(directions.col(0).dot(block.orbital_hessian.col(1)) -
                     directions.col(1).dot(block.orbital_hessian.col(0))) < 3e-10,
            "normalization reuse changed orbital Hessian symmetry");
    const auto& coefficients = accepted.second_order_context->selected_state_eigenvectors;
    const Eigen::VectorXd origin = layout.pack(orbitals),
                          tangent = chart.expand_step(directions.col(0));
    const auto shifted = [&](double scale) -> Eigen::VectorXd {
        auto moved = input;
        layout.unpack(origin + scale * tangent, &moved.orbital_preparation_input);
        const auto value =
            evaluator.evaluate_fixed_structure(moved, states, weights, 0.4, coefficients);
        return chart.project_reduced_gradient(
            layout.gather_from_full(value.gradient.sparse_orbital_energy_gradient));
    };
    constexpr double step = 1e-5;
    const Eigen::VectorXd plus = shifted(step), minus = shifted(-step);
    require(((plus - minus) / (2 * step) - scalar.orbital_hessian).norm() < 2e-7,
            "normalization-reused orbital block differs from the fixed-state gradient derivative");
}

/** @brief Nonstationary fixed-state derivatives without any CI eigensolve. */
void check_keyframe(bool complete_carrier) {
    xmvb::vb::VbScfInput input;
    input.standard_two_electron_mode = xmvb::vb::StandardTwoElectronMode::Exact;
    auto& orbitals = input.orbital_preparation_input;
    orbitals.n_basis_functions = 3;
    orbitals.n_orbitals = orbitals.n_active_orbitals = 2;
    orbitals.n_total_electrons = orbitals.n_active_electrons = 2;
    orbitals.orbital_value_table = {1.0, 0.13, -0.17, 0.21, 0.94, 0.12};
    orbitals.orbital_basis_index_table = {1, 2, 3, 1, 2, 3};
    orbitals.orbital_basis_counts = {3, 3};
    orbitals.ao_overlap_matrix = Eigen::Matrix3d::Identity();
    orbitals.ao_overlap_matrix(0, 1) = orbitals.ao_overlap_matrix(1, 0) = 0.1;
    auto& ao = input.ao_integral_input;
    ao.n_basis_functions = 3;
    ao.ao_core_hamiltonian_matrix.resize(3, 3);
    ao.ao_core_hamiltonian_matrix << -1.1, 0.13, 0.04, 0.13, -0.6, -0.08, 0.04, -0.08, 0.2;
    std::vector<int> left, right;
    std::vector<double> values;
    for (int p = 0; p < 6; ++p) {
        for (int q = 0; q <= p; ++q) {
            left.push_back(p);
            right.push_back(q);
            values.push_back(0.07 * std::cos(0.2 * (p - q)));
        }
    }
    ao.pair_graph = xmvb::integrals::build_ao_pair_graph(left, right, values, 3);
    auto& data = input.structure_data;
    data.alpha_strings = complete_carrier ? std::vector<std::vector<int>>{{0}, {1}}
                                          : std::vector<std::vector<int>>{{0}};
    data.beta_strings = {{0}, {1}};
    data.offsets = {0};
    for (int a = 0; a < (complete_carrier ? 2 : 1); ++a) {
        for (int b = 0; b < 2; ++b) {
            data.pair_alpha.push_back(a);
            data.pair_beta.push_back(b);
            data.terms.push_back({data.n_structures++, 1.0});
            data.offsets.push_back(data.terms.size());
        }
    }
    Eigen::MatrixXd c(data.n_structures, 2);
    for (int i = 0; i < c.rows(); ++i) {
        c(i, 0) = 0.37 + 0.19 * i;
        c(i, 1) = std::cos(0.73 * (i + 1));
    }
    const std::vector<int> states{0, data.n_structures - 1};
    const std::vector<double> weights{0.5, 0.5};
    const xmvb::vb::OrbitalGradientEvaluator evaluator;
    const auto relaxed = evaluator.evaluate(input, states, weights, 0.4);
    const auto& accepted = *relaxed.second_order_context;
    require(accepted.structure_action && accepted.structure_action->supports_integral_direction() &&
                accepted.selected_state_matrices.states.empty(),
            "direct-CI accepted point retained unused raw product states");
    check_frame_hvp(input, 1);
    check_frame_hvp(input, 2);
    auto with_inactive = input;
    auto& occupied = with_inactive.orbital_preparation_input;
    occupied.n_orbitals = 3;
    occupied.n_total_electrons = 4;
    occupied.orbital_value_table.insert(occupied.orbital_value_table.begin(), {0.0, 0.0, 1.0});
    occupied.orbital_basis_index_table.insert(occupied.orbital_basis_index_table.begin(),
                                              {1, 2, 3});
    occupied.orbital_basis_counts.insert(occupied.orbital_basis_counts.begin(), 3);
    check_frame_hvp(with_inactive, 1);
    check_frame_hvp(with_inactive, 2);
    const auto evaluate = [&](const Eigen::MatrixXd& coefficients) {
        return evaluator.evaluate_fixed_structure(input, states, weights, 0.4, coefficients);
    };
    const auto fixed = evaluate(c);
    require(fixed.gradient.scf_result.eigenvector_matrix.empty() &&
                fixed.gradient.scf_result.structure_overlap_diagonal.empty(),
            "fixed-state gradient constructed eigensolver output");
    const auto scaled = evaluate(2.7 * c);
    require(std::abs(scaled.gradient.scf_result.total_energy -
                     fixed.gradient.scf_result.total_energy) < 1e-11 &&
                (2.7 * scaled.coefficient_gradient - fixed.coefficient_gradient).norm() < 1e-11,
            "fixed-state normalization changed energy or residual");
    Eigen::Matrix2d change;
    change << 1.2, -0.3, 0.2, 0.8;
    const auto mixed = evaluate(c * change);
    require(
        std::abs(mixed.gradient.scf_result.total_energy - fixed.gradient.scf_result.total_energy) <
                1e-11 &&
            (mixed.coefficient_gradient * change.transpose() - fixed.coefficient_gradient).norm() <
                1e-10,
        "equal-weight keyframe depends on the state frame");
    constexpr double delta = 1e-5;
    for (int i = 0; i < 6; ++i) {
        const double value = orbitals.orbital_value_table[i];
        orbitals.orbital_value_table[i] = value + delta;
        const double plus = evaluate(c).gradient.scf_result.total_energy;
        orbitals.orbital_value_table[i] = value - delta;
        const double minus = evaluate(c).gradient.scf_result.total_energy;
        orbitals.orbital_value_table[i] = value;
        require(std::abs((plus - minus) / (2 * delta) -
                         fixed.gradient.sparse_orbital_energy_gradient[i]) < 2e-7,
                "fixed-state orbital gradient differs from finite differences");
    }
    for (int state = 0; state < 2; ++state) {
        for (int i = 0; i < c.rows(); ++i) {
            const double value = c(i, state);
            c(i, state) = value + delta;
            const double plus = evaluate(c).gradient.scf_result.total_energy;
            c(i, state) = value - delta;
            const double minus = evaluate(c).gradient.scf_result.total_energy;
            c(i, state) = value;
            require(std::abs((plus - minus) / (2 * delta) - fixed.coefficient_gradient(i, state)) <
                        2e-7,
                    "fixed-state structure residual differs from finite differences");
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        check_terminal_trial();
        check_restart_phase();
        if (argc == 2 && (std::string(argv[1]) == "--accepted-state16" ||
                          std::string(argv[1]) == "--accepted-hvp16")) {
            benchmark_accepted_state16(std::string(argv[1]) == "--accepted-hvp16");
            return 0;
        }
        if (argc != 1)
            throw std::invalid_argument(
                "usage: test_keyframe [--accepted-state16|--accepted-hvp16]");
        check_keyframe(true);
        check_keyframe(false);
        std::cout << "fixed-state keyframe actions and derivatives: passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
