#include "vbscf/derivatives/gradient/active_space/evaluator.hpp"

#include "mcscf/states/weights.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include "vbscf/integrals/active/space.hpp"
#include "integrals/active/raw.hpp"
#include "vbscf/orbitals/preparation/cholesky.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "core/pair_index.hpp"
#include "core/memory.hpp"
#include "core/timing.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/derivatives/gradient/active_space/helpers.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin/backward.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"
#include "vbscf/structures/assembly/selected_coefficients.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/assembly/states.hpp"
#include "vbscf/structures/orthogonal_ci/metric.hpp"
#include "vbscf/structures/overlap/representation.hpp"
#include "vbscf/structures/overlap/metric.hpp"
#include "vbscf/structures/overlap/spin_metric.hpp"
#include "mcscf/optimization/trace.hpp"

namespace xmvb::vb {

namespace {

struct ActiveSpaceGradientForwardContext {
    TimedPreparedActiveSpaceContext timed_active_space_context;
    SameSpinPairCacheContext same_spin_pair_cache;
    // Stable address across forward-context moves: the metric inverse borrows it.
    std::unique_ptr<StructureAction> structure_action;
    std::unique_ptr<StructureMetricInverse> structure_metric_inverse;
    std::shared_ptr<const SelectedMetric> selected_metric;
    std::shared_ptr<const SpinMetric> spin_metric;
    std::optional<StructureAdjointState> structure_adjoint_state;
    StructureAccumulationResult structure_matrices;
    xmvb::linalg::GeneralizedEigenResult eigen_result;
    Eigen::VectorXd structure_overlap_diagonal;
    bool structure_overlap_diagonal_exact = false;
    Eigen::MatrixXd overlap_eigenvectors;
    Eigen::MatrixXd hamiltonian_eigenvectors;
    std::optional<DavidsonDiagnostics> davidson_diagnostics;
    Eigen::MatrixXd selected_state_eigenvectors;
    std::vector<double> selected_state_energies;
    double average_structure_overlap = 0.0;
    double structure_matrix_wall_time_seconds = 0.0;
    double eigensolver_wall_time_seconds = 0.0;
};

void validate_state_selection(const std::vector<int>& selected_state_indices,
                              const std::vector<double>& state_average_weights,
                              Eigen::Index n_structures) {
    if (selected_state_indices.empty()) {
        throw std::invalid_argument("selected_state_indices must not be empty");
    }
    if (selected_state_indices.size() != state_average_weights.size()) {
        throw std::invalid_argument(
            "selected_state_indices and state_average_weights must have the same length");
    }
    for (const int state_index : selected_state_indices) {
        if (state_index < 0 || state_index >= n_structures) {
            throw std::out_of_range("selected state index is out of range");
        }
    }
}

double compute_average_structure_overlap(const std::vector<double>& overlap_matrix,
                                         Eigen::Index n_structures) {
    double diagonal_sum = 0.0;
    for (Eigen::Index structure_index = 0; structure_index < n_structures; ++structure_index) {
        diagonal_sum += overlap_matrix[structure_index * n_structures + structure_index];
    }
    return diagonal_sum / static_cast<double>(n_structures);
}

double selected_state_average_energy(const std::vector<double>& eigenvalues,
                                     const std::vector<int>& selected_state_indices,
                                     const std::vector<double>& state_average_weights) {
    double energy = 0.0;
    for (std::size_t selected_state_offset = 0;
         selected_state_offset < selected_state_indices.size();
         ++selected_state_offset) {
        energy += state_average_weights[selected_state_offset] *
                  eigenvalues[selected_state_indices[selected_state_offset]];
    }
    return energy;
}

int required_root_count(const std::vector<int>& selected_state_indices) {
    return *std::max_element(selected_state_indices.begin(), selected_state_indices.end()) + 1;
}

void retain_structure_action(const VbScfInput& input, ActiveSpaceGradientForwardContext* context) {
    if (context->structure_action) {
        return;
    }
    const auto& prepared = context->timed_active_space_context.prepared_active_space;
    std::optional<StructureDiagonal> precomputed_diagonal;
    const std::size_t expected_matrix_size =
        static_cast<std::size_t>(input.structure_data.n_structures) *
        input.structure_data.n_structures;
    if (context->structure_matrices.hamiltonian_matrix.size() == expected_matrix_size &&
        context->structure_matrices.overlap_matrix.size() == expected_matrix_size) {
        precomputed_diagonal.emplace();
        precomputed_diagonal->hamiltonian.resize(input.structure_data.n_structures);
        precomputed_diagonal->overlap.resize(input.structure_data.n_structures);
        for (Eigen::Index structure = 0; structure < input.structure_data.n_structures;
             ++structure) {
            const std::size_t diagonal_index =
                static_cast<std::size_t>(structure) * input.structure_data.n_structures + structure;
            precomputed_diagonal->hamiltonian[structure] =
                context->structure_matrices.hamiltonian_matrix[diagonal_index];
            precomputed_diagonal->overlap[structure] =
                context->structure_matrices.overlap_matrix[diagonal_index];
        }
    }
    context->structure_action =
        std::make_unique<StructureAction>(input.structure_data.incidence(),
                                          input.structure_data.n_structures,
                                          context->same_spin_pair_cache,
                                          prepared.orbital_result.active_orbital_overlap_matrix,
                                          prepared.active_space_one_electron_result.h1e_act,
                                          prepared.active_space_two_electron_result,
                                          input.orbital_preparation_input.n_active_orbitals,
                                          precomputed_diagonal ? &*precomputed_diagonal : nullptr);
}

/** @brief Plan solver storage before admitting reusable numerical factors. */
linalg::DavidsonOptions structure_solve_options(Eigen::Index n_structures,
                                                int n_roots,
                                                StructureSolveAccuracy structure_solve_accuracy) {
    structure_solve_accuracy.validate();
    xmvb::linalg::DavidsonOptions options =
        xmvb::linalg::make_davidson_options(n_structures,
                                            n_roots,
                                            structure_solve_accuracy.energy_tolerance,
                                            structure_solve_accuracy.gradient_tolerance,
                                            structure_solve_accuracy.stationarity_tolerance());
    // The existing Davidson policy fixes the maximum basis dimension. Leave
    // room for its V/HV/MV arrays, restart/root/correction images and projected
    // eigensolver work before allowing the CI tile cache to consume headroom.
    // This accounts for storage only; it does not change the Davidson space.
    const long double n = n_structures, k = options.max_subspace_dimension, r = n_roots;
    const long double solver_bytes = sizeof(double) * (3 * n * (2 * k + 6 * r + 4) + 12 * k * k);
    if (solver_bytes > std::numeric_limits<std::size_t>::max())
        throw std::length_error("Davidson memory estimate overflow");
    memory::prepare_ci(static_cast<std::size_t>(solver_bytes));
    return options;
}

xmvb::linalg::DavidsonResult solve_selected_structure_roots(
    const xmvb::linalg::GeneralizedEigenAction& action,
    const Eigen::Ref<const Eigen::VectorXd>& hamiltonian_diagonal,
    const Eigen::Ref<const Eigen::VectorXd>& overlap_diagonal,
    Eigen::Index n_structures,
    int n_roots,
    StructureSolveAccuracy structure_solve_accuracy,
    const Eigen::Ref<const Eigen::MatrixXd>& initial_eigenvectors,
    const xmvb::linalg::GeneralizedEigensolver& generalized_eigensolver,
    std::function<double(const Eigen::VectorXd&, double)> physical_residual_norm = {},
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, double)> residual_preconditioner = {},
    bool identity_overlap = false) {
    auto options = structure_solve_options(n_structures, n_roots, structure_solve_accuracy);
    options.physical_residual_norm = std::move(physical_residual_norm);
    options.residual_preconditioner = std::move(residual_preconditioner);
    options.identity_overlap = identity_overlap;
    if (mcscf::diagnostics::ciah_event_trace_enabled()) {
        options.observe_iteration = [start =
                                         std::chrono::steady_clock::now()](int iteration,
                                                                           int dimension,
                                                                           double energy,
                                                                           double residual,
                                                                           double stationarity) {
            // Existing quantities only; no additional action or certification.
            mcscf::diagnostics::ciah_trace(
                "ciah_ci_iteration",
                "iteration=%d,basis=%d,energy=%.17g,relative_residual=%.17g,stationarity=%.17g,"
                "seconds=%.17g",
                iteration,
                dimension,
                energy,
                residual,
                stationarity,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        };
    }
    const bool can_recycle = initial_eigenvectors.rows() == n_structures &&
                             initial_eigenvectors.cols() >= n_roots &&
                             initial_eigenvectors.allFinite();
    return can_recycle ? generalized_eigensolver.solve_davidson(
                             action,
                             hamiltonian_diagonal,
                             overlap_diagonal,
                             // The FULL adapter also supplies the transformed default block;
                             // retain its random second half, not only the requested roots.
                             initial_eigenvectors.leftCols(
                                 identity_overlap ? initial_eigenvectors.cols() : n_roots),
                             options)
                       : generalized_eigensolver.solve_davidson(
                             action, hamiltonian_diagonal, overlap_diagonal, options);
}

void retain_davidson_result(xmvb::linalg::DavidsonResult davidson,
                            ActiveSpaceGradientForwardContext* context) {
    context->davidson_diagnostics =
        DavidsonDiagnostics{davidson.iterations,
                            davidson.block_actions,
                            davidson.peak_subspace_dimension,
                            *std::max_element(davidson.relative_residual_norms.begin(),
                                              davidson.relative_residual_norms.end()),
                            *std::max_element(davidson.stationarity_residual_norms.begin(),
                                              davidson.stationarity_residual_norms.end())};
    context->eigen_result = std::move(davidson.eigenpairs);
    context->overlap_eigenvectors = std::move(davidson.overlap_eigenvectors);
    context->hamiltonian_eigenvectors = std::move(davidson.hamiltonian_eigenvectors);
}

/** @brief Same-dimensional FULL solve; orbital maps occur only at its boundary. */
linalg::DavidsonResult
solve_complete_structure_roots(const StructureAction& action,
                               const StructureMetricInverse& coordinates,
                               int n_roots,
                               StructureSolveAccuracy accuracy,
                               const Eigen::Ref<const Eigen::MatrixXd>& initial,
                               const linalg::GeneralizedEigensolver& solver) {
    const auto& diagonal = action.preconditioner_diagonal();
    const Eigen::Index n = action.n_structures();
    const bool recycled = initial.rows() == n && initial.cols() >= n_roots && initial.allFinite();
    const auto options = linalg::make_davidson_options(n,
                                                       n_roots,
                                                       accuracy.energy_tolerance,
                                                       accuracy.gradient_tolerance,
                                                       accuracy.stationarity_tolerance());
    const Eigen::MatrixXd seeds = recycled
                                      ? Eigen::MatrixXd(initial.leftCols(n_roots))
                                      : linalg::make_davidson_initial_vectors(
                                            diagonal.hamiltonian,
                                            diagonal.overlap,
                                            std::min(options.max_subspace_dimension, 2 * n_roots));
    const Eigen::MatrixXd guess = coordinates.to_ci_coordinates(seeds);
    auto result = solve_selected_structure_roots(
        [&coordinates](const Eigen::Ref<const Eigen::MatrixXd>& x) {
            return linalg::GeneralizedEigenActionResult{coordinates.apply_ci_hamiltonian(x), {}};
        },
        diagonal.hamiltonian,
        diagonal.overlap,
        n,
        n_roots,
        accuracy,
        guess,
        solver,
        {},
        [&coordinates](const Eigen::VectorXd& r, double energy) {
            return Eigen::VectorXd(
                coordinates.apply_ci_preconditioner(r, Eigen::VectorXd::Constant(1, energy), 0.0)
                    .col(0));
        },
        true);
    const Eigen::Map<const Eigen::MatrixXd> x(
        result.eigenpairs.eigenvector_matrix.data(), n, n_roots);
    const Eigen::MatrixXd coefficients = coordinates.from_ci_coordinates(x);
    result.hamiltonian_eigenvectors =
        coordinates.ci_covector_to_structure(result.hamiltonian_eigenvectors);
    result.overlap_eigenvectors = coordinates.ci_covector_to_structure(x);
    result.eigenpairs.eigenvector_matrix.assign(coefficients.data(),
                                                coefficients.data() + coefficients.size());
    return result;
}

/** @brief Riesz map for a bounded selected space, without a carrier projection.
 * The residual r is a covector; M^-1 r is its physical tangent representative.
 * This metric preconditioner is positive and invariant to basis congruences.
 */
std::shared_ptr<const SelectedMetric> selected_ci_metric(const StructureAction& action,
                                                         const SameSpinPairCacheContext& topology,
                                                         const std::vector<double>& overlap,
                                                         int n) {
    if (!RepresentationOverlap::PreparedMetric::fits_selected(action.n_structures()))
        return {};
    const auto& alpha = topology.alpha_reuse_table.unique_determinants;
    const auto& beta = topology.beta_reuse_table.unique_determinants;
    const auto upper = overlap_upper(Eigen::Map<const Eigen::MatrixXd>(overlap.data(), n, n));
    const auto metric = RepresentationOverlap(action, alpha, beta, n).prepare_metric(upper);
    if (!metric.selected_matrix().size())
        return {};
    if (mcscf::diagnostics::ciah_event_trace_enabled()) {
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(metric.selected_matrix(),
                                                                      Eigen::EigenvaluesOnly);
        const Eigen::VectorXd scale = action.preconditioner_diagonal().overlap.array().rsqrt();
        const Eigen::MatrixXd scaled =
            scale.asDiagonal() * metric.selected_matrix() * scale.asDiagonal();
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> normalized(scaled,
                                                                        Eigen::EigenvaluesOnly);
        mcscf::diagnostics::ciah_trace(
            "ciah_ci_selected_metric",
            "dimension=%d,condition=%.17g,diagonal_scaled_condition=%.17g,factor_bytes=%zu",
            action.n_structures(),
            spectrum.eigenvalues().tail(1)[0] / spectrum.eigenvalues()[0],
            normalized.eigenvalues().tail(1)[0] / normalized.eigenvalues()[0],
            sizeof(double) * static_cast<std::size_t>(action.n_structures()) *
                action.n_structures());
    }
    return metric.selected_metric();
}

void select_structure_states(Eigen::Index n_structures,
                             const std::vector<int>& selected_state_indices,
                             ActiveSpaceGradientForwardContext* context) {
    const int n_roots = static_cast<int>(context->eigen_result.eigenvalues.size());
    if (n_roots < required_root_count(selected_state_indices) ||
        context->eigen_result.eigenvector_matrix.size() !=
            static_cast<std::size_t>(n_structures) * n_roots) {
        throw std::runtime_error(
            "structure eigensolver returned inconsistent selected-root dimensions");
    }
    const Eigen::Map<const Eigen::MatrixXd> roots(
        context->eigen_result.eigenvector_matrix.data(), n_structures, n_roots);
    context->selected_state_eigenvectors.resize(n_structures,
                                                static_cast<int>(selected_state_indices.size()));
    context->selected_state_energies.resize(selected_state_indices.size());
    for (std::size_t state = 0; state < selected_state_indices.size(); ++state) {
        const int root = selected_state_indices[state];
        context->selected_state_eigenvectors.col(static_cast<int>(state)) = roots.col(root);
        context->selected_state_energies[state] =
            context->eigen_result.eigenvalues[static_cast<std::size_t>(root)];
    }
}

void solve_structure_problem(
    const VbScfInput& input,
    const std::vector<int>& selected_state_indices,
    StructureEigensolver structure_eigensolver,
    StructureSolveAccuracy structure_solve_accuracy,
    const Eigen::Ref<const Eigen::MatrixXd>& initial_eigenvectors,
    const FullDeterminantStructureHamiltonianOverlapBuilder& structure_builder,
    const xmvb::linalg::GeneralizedEigensolver& generalized_eigensolver,
    ActiveSpaceGradientForwardContext* context) {
    const auto& prepared = context->timed_active_space_context.prepared_active_space;
    const Eigen::Index n_structures = input.structure_data.n_structures;
    const int n_roots = required_root_count(selected_state_indices);
    auto stage_start_time = std::chrono::steady_clock::now();
    const bool use_materialized_operator = structure_eigensolver == StructureEigensolver::Dense;

    if (use_materialized_operator) {
        context->structure_matrices =
            structure_builder.build(input.structure_data.alpha(),
                                    input.structure_data.beta(),
                                    input.structure_data.incidence(),
                                    prepared.orbital_result.active_orbital_overlap_matrix,
                                    prepared.active_space_one_electron_result.h1e_act,
                                    input.orbital_preparation_input.n_active_orbitals,
                                    prepared.active_space_two_electron_result,
                                    n_structures,
                                    context->same_spin_pair_cache);
        context->structure_matrix_wall_time_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time)
                .count();
        context->average_structure_overlap = compute_average_structure_overlap(
            context->structure_matrices.overlap_matrix, n_structures);
        context->structure_overlap_diagonal.resize(n_structures);
        for (Eigen::Index structure = 0; structure < n_structures; ++structure) {
            context->structure_overlap_diagonal[structure] =
                context->structure_matrices
                    .overlap_matrix[static_cast<std::size_t>(structure) * n_structures + structure];
        }
        context->structure_overlap_diagonal_exact = true;
        stage_start_time = std::chrono::steady_clock::now();
        if (structure_eigensolver == StructureEigensolver::Dense) {
            context->eigen_result =
                generalized_eigensolver.solve_dense(context->structure_matrices.hamiltonian_matrix,
                                                    context->structure_matrices.overlap_matrix,
                                                    n_structures);
        } else {
            const Eigen::Map<const Eigen::MatrixXd> hamiltonian(
                context->structure_matrices.hamiltonian_matrix.data(), n_structures, n_structures);
            const Eigen::Map<const Eigen::MatrixXd> overlap(
                context->structure_matrices.overlap_matrix.data(), n_structures, n_structures);
            const xmvb::linalg::GeneralizedEigenAction matrix_action =
                [&hamiltonian, &overlap](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
                    return xmvb::linalg::GeneralizedEigenActionResult{hamiltonian * vectors,
                                                                      overlap * vectors};
                };
            retain_davidson_result(solve_selected_structure_roots(matrix_action,
                                                                  hamiltonian.diagonal(),
                                                                  overlap.diagonal(),
                                                                  n_structures,
                                                                  n_roots,
                                                                  structure_solve_accuracy,
                                                                  initial_eigenvectors,
                                                                  generalized_eigensolver),
                                   context);
        }
        if (structure_eigensolver == StructureEigensolver::Davidson ||
            !context->same_spin_pair_cache.enabled()) {
            retain_structure_action(input, context);
        }
    } else {
        profiling::ScopedPhase setup_phase(profiling::Phase::CiOperatorSetup, true);
        std::optional<StructureDiagonal> jacobi_diagonal;
        if (!context->same_spin_pair_cache.enabled()) {
            jacobi_diagonal = structure_builder.build_davidson_jacobi_preconditioner(
                input.structure_data.alpha(),
                input.structure_data.beta(),
                input.structure_data.incidence(),
                prepared.orbital_result.active_orbital_overlap_matrix,
                prepared.active_space_one_electron_result.h1e_act,
                input.orbital_preparation_input.n_active_orbitals,
                prepared.active_space_two_electron_result,
                n_structures,
                context->same_spin_pair_cache);
        }
        context->structure_action =
            std::make_unique<StructureAction>(input.structure_data.incidence(),
                                              n_structures,
                                              context->same_spin_pair_cache,
                                              prepared.orbital_result.active_orbital_overlap_matrix,
                                              prepared.active_space_one_electron_result.h1e_act,
                                              prepared.active_space_two_electron_result,
                                              input.orbital_preparation_input.n_active_orbitals,
                                              jacobi_diagonal ? &*jacobi_diagonal : nullptr);
        context->structure_matrix_wall_time_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time)
                .count();
        const StructureAction& structure_action = *context->structure_action;
        setup_phase.stop();
        structure_solve_options(n_structures, n_roots, structure_solve_accuracy);
        if (context->structure_overlap_diagonal.size() == 0) {
            context->structure_overlap_diagonal =
                structure_action.preconditioner_diagonal().overlap;
            context->structure_overlap_diagonal_exact = context->same_spin_pair_cache.enabled();
        }
        context->average_structure_overlap = context->structure_overlap_diagonal.mean();
        const xmvb::linalg::GeneralizedEigenAction action =
            [&structure_action](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
                profiling::ScopedPhase phase(profiling::Phase::CiSigma, true);
                StructureActionResult images = structure_action.apply(vectors);
                return xmvb::linalg::GeneralizedEigenActionResult{std::move(images.hamiltonian),
                                                                  std::move(images.overlap)};
            };
        stage_start_time = std::chrono::steady_clock::now();
        profiling::ScopedPhase solve_phase(profiling::Phase::StructureSolve, true);
        std::function<double(const Eigen::VectorXd&, double)> physical_residual;
        if (input.complete_active_space && structure_action.storage().orthogonal_direct_ci) {
            physical_residual = [&structure_action](const Eigen::VectorXd& coefficients,
                                                    double energy) {
                const auto start = std::chrono::steady_clock::now();
                const double norm =
                    structure_action.complete_space_residual_norm(coefficients, energy);
                mcscf::diagnostics::ciah_trace(
                    "ciah_ci_physical_residual",
                    "seconds=%.17g",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                        .count());
                return norm;
            };
            constexpr std::size_t budget = 64ULL * 1024ULL * 1024ULL;
            const auto& alpha = context->same_spin_pair_cache.alpha_reuse_table.unique_determinants;
            const auto& beta = context->same_spin_pair_cache.beta_reuse_table.unique_determinants;
            if (structure_action.supports_carrier_diagonal() &&
                StructureMetricInverse::fits(structure_action, alpha, beta, budget, budget)) {
                const auto inverse_start = std::chrono::steady_clock::now();
                context->structure_metric_inverse = std::make_unique<StructureMetricInverse>(
                    structure_action, alpha, beta, true, budget, budget);
                mcscf::diagnostics::ciah_trace(
                    "ciah_ci_inverse_setup",
                    "seconds=%.17g",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - inverse_start)
                        .count());
                const auto* inverse = context->structure_metric_inverse.get();
                mcscf::diagnostics::ciah_trace("ciah_ci_preconditioner",
                                               "carrier=1,factor_bytes=%zu,blocks=%zu,max_block=%d",
                                               inverse->factor_bytes(),
                                               inverse->block_count(),
                                               inverse->max_block_size());
            }
        }
        if (context->structure_metric_inverse) {
            retain_davidson_result(
                solve_complete_structure_roots(structure_action,
                                               *context->structure_metric_inverse,
                                               n_roots,
                                               structure_solve_accuracy,
                                               initial_eigenvectors,
                                               generalized_eigensolver),
                context);
        } else {
            profiling::ScopedPhase preconditioner_phase(profiling::Phase::CiPreconditioner, true);
            auto metric =
                structure_action.storage().orthogonal_direct_ci
                    ? selected_ci_metric(structure_action,
                                         context->same_spin_pair_cache,
                                         prepared.orbital_result.active_orbital_overlap_matrix,
                                         input.orbital_preparation_input.n_active_orbitals)
                    : std::shared_ptr<const SelectedMetric>{};
            context->selected_metric = metric;
            auto selected_action = action;
            std::function<Eigen::VectorXd(const Eigen::VectorXd&, double)> preconditioner;
            if (metric) {
                selected_action = [&structure_action,
                                   metric](const Eigen::Ref<const Eigen::MatrixXd>& v) {
                    profiling::ScopedPhase phase(profiling::Phase::CiSigma, true);
                    return linalg::GeneralizedEigenActionResult{
                        structure_action.apply_hamiltonian(v), metric->overlap * v};
                };
                preconditioner = [metric](const Eigen::VectorXd& r, double) {
                    return Eigen::VectorXd(metric->factor.solve(r));
                };
            } else if (structure_action.storage().orthogonal_direct_ci) {
                // W^T W is occupation-block diagonal for any VB selection, not only
                // FULL. Remove spin recoupling from the preconditioner even when the
                // physical nonorthogonal Gram must remain an exact streamed action.
                const auto budget = workspace::available();
                const auto& alpha =
                    context->same_spin_pair_cache.alpha_reuse_table.unique_determinants;
                const auto& beta =
                    context->same_spin_pair_cache.beta_reuse_table.unique_determinants;
                if (SpinMetric::fits(structure_action, alpha, beta, budget)) {
                    auto spin = std::make_shared<SpinMetric>(structure_action, alpha, beta, budget);
                    context->spin_metric = spin;
                    preconditioner = [spin, &structure_action](const Eigen::VectorXd& r,
                                                               double energy) {
                        const auto& diagonal = structure_action.preconditioner_diagonal();
                        // Match every H-E M diagonal while retaining the selected spin
                        // geometry. Positivity concerns this inverse model, not H-E M.
                        const Eigen::VectorXd floor =
                            std::sqrt(std::numeric_limits<double>::epsilon()) *
                            diagonal.hamiltonian.cwiseAbs()
                                .cwiseMax((energy * diagonal.overlap).cwiseAbs())
                                .cwiseMax(1.0);
                        const Eigen::VectorXd positive =
                            (diagonal.hamiltonian - energy * diagonal.overlap)
                                .cwiseAbs()
                                .cwiseMax(floor);
                        return Eigen::VectorXd(spin->solve_diagonal_model(r, positive).col(0));
                    };
                    mcscf::diagnostics::ciah_trace(
                        "ciah_ci_preconditioner",
                        "spin=1,energy_model=1,factor_bytes=%zu,blocks=%zu,max_block=%d",
                        spin->factor_bytes(),
                        spin->block_count(),
                        spin->max_block_size());
                }
            }
            preconditioner_phase.stop();
            retain_davidson_result(solve_selected_structure_roots(
                                       selected_action,
                                       structure_action.preconditioner_diagonal().hamiltonian,
                                       structure_action.preconditioner_diagonal().overlap,
                                       n_structures,
                                       n_roots,
                                       structure_solve_accuracy,
                                       initial_eigenvectors,
                                       generalized_eigensolver,
                                       std::move(physical_residual),
                                       std::move(preconditioner)),
                                   context);
        }
    }
    context->eigensolver_wall_time_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();
    select_structure_states(n_structures, selected_state_indices, context);
    mcscf::diagnostics::ciah_trace(
        "ciah_ci_solved",
        "seconds=%.17g,energy=%.17g,iterations=%d,actions=%d",
        context->eigensolver_wall_time_seconds,
        context->selected_state_energies[0],
        context->davidson_diagnostics ? context->davidson_diagnostics->iterations : 0,
        context->davidson_diagnostics ? context->davidson_diagnostics->block_actions : 0);
}

ActiveSpaceGradientForwardContext build_active_space_gradient_forward_context(
    const VbScfInput& input,
    const std::vector<int>& selected_state_indices,
    StructureEigensolver structure_eigensolver,
    StructureSolveAccuracy structure_solve_accuracy,
    const Eigen::Ref<const Eigen::MatrixXd>& initial_eigenvectors,
    const FullDeterminantStructureHamiltonianOverlapBuilder& structure_builder,
    const xmvb::linalg::GeneralizedEigensolver& generalized_eigensolver) {
    ActiveSpaceGradientForwardContext context;
    profiling::ScopedPhase preparation_phase(profiling::Phase::ActivePreparation, true);
    context.timed_active_space_context = prepare_timed_active_space_context(input);
    preparation_phase.stop();

    profiling::ScopedPhase topology_phase(profiling::Phase::SpinTopology, true);
    context.same_spin_pair_cache =
        build_orthogonal_spin_topology(input.structure_data.alpha(),
                                       input.structure_data.beta(),
                                       input.orbital_preparation_input.n_active_orbitals);
    topology_phase.stop();
    solve_structure_problem(input,
                            selected_state_indices,
                            structure_eigensolver,
                            structure_solve_accuracy,
                            initial_eigenvectors,
                            structure_builder,
                            generalized_eigensolver,
                            &context);
    return context;
}

ActiveSpaceGradientForwardContext build_active_space_gradient_forward_context(
    const VbScfInput& input,
    TimedPreparedActiveSpaceContext timed_active_space_context,
    const std::vector<int>& selected_state_indices,
    StructureEigensolver structure_eigensolver,
    StructureSolveAccuracy structure_solve_accuracy,
    const Eigen::Ref<const Eigen::MatrixXd>& initial_eigenvectors,
    const FullDeterminantStructureHamiltonianOverlapBuilder& structure_builder,
    const xmvb::linalg::GeneralizedEigensolver& generalized_eigensolver) {
    ActiveSpaceGradientForwardContext context;
    context.timed_active_space_context = std::move(timed_active_space_context);

    context.same_spin_pair_cache =
        build_orthogonal_spin_topology(input.structure_data.alpha(),
                                       input.structure_data.beta(),
                                       input.orbital_preparation_input.n_active_orbitals);

    solve_structure_problem(input,
                            selected_state_indices,
                            structure_eigensolver,
                            structure_solve_accuracy,
                            initial_eigenvectors,
                            structure_builder,
                            generalized_eigensolver,
                            &context);
    return context;
}

std::shared_ptr<AcceptedPointContext>
finalize_active_space_second_order_context(const VbScfInput& input,
                                           const std::vector<int>& selected_state_indices,
                                           const std::vector<double>& normalized_weights,
                                           StructureSolveAccuracy structure_solve_accuracy,
                                           const ActiveSpaceGradientResult& gradient_result,
                                           ActiveSpaceGradientForwardContext* forward_context) {
    if (forward_context == nullptr) {
        throw std::invalid_argument("forward_context must not be null");
    }

    auto context = std::make_shared<AcceptedPointContext>();
    context->complete_active_space = input.complete_active_space;
    context->structure_solve_accuracy = structure_solve_accuracy;
    // Move the heavy accepted-point payload out of the transient forward context
    // once the relaxed gradient has finished using it. This keeps the future
    // second-order cache alive without duplicating the large unique-spin tables.
    context->prepared_active_space =
        std::move(forward_context->timed_active_space_context.prepared_active_space);
    context->same_spin_pair_cache = std::move(forward_context->same_spin_pair_cache);
    context->selected_metric = std::move(forward_context->selected_metric);
    context->spin_metric = std::move(forward_context->spin_metric);
    if (forward_context->structure_action) {
        context->structure_action.emplace(std::move(*forward_context->structure_action));
        if (forward_context->structure_metric_inverse) {
            forward_context->structure_metric_inverse->rebind(*context->structure_action);
            context->structure_metric_inverse =
                std::move(forward_context->structure_metric_inverse);
            context->structure_metric_planned = true;
        }
    }
    if (forward_context->structure_adjoint_state.has_value()) {
        context->structure_adjoint_state = std::move(forward_context->structure_adjoint_state);
    }
    context->active_orbital_overlap_gradient = gradient_result.active_orbital_overlap_gradient;
    context->active_one_electron_gradient = gradient_result.active_one_electron_gradient;
    context->packed_active_two_electron_gradient =
        gradient_result.packed_active_two_electron_gradient;
    context->selected_state_indices = selected_state_indices;
    context->normalized_state_weights = normalized_weights;
    context->n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
    context->n_structures = input.structure_data.n_structures;
    context->use_full_matrix_form_adjoint = context->same_spin_pair_cache.has_pair_providers();
    context->use_pair_graph_opposite_spin_adjoint =
        context->same_spin_pair_cache.has_pair_providers();
    context->selected_state_energies = std::move(forward_context->selected_state_energies);
    context->selected_state_eigenvectors = std::move(forward_context->selected_state_eigenvectors);
    // Davidson already formed both Ritz images. Retain selected columns rather
    // than repeating a spin-product action when the coupled model is built.
    if (forward_context->hamiltonian_eigenvectors.size() != 0) {
        auto& images = context->selected_state_structure_images.emplace();
        const auto n_states = static_cast<Eigen::Index>(selected_state_indices.size());
        images.hamiltonian.resize(context->n_structures, n_states);
        images.overlap.resize(context->n_structures, n_states);
        for (Eigen::Index state = 0; state < n_states; ++state) {
            const int root = selected_state_indices[state];
            images.hamiltonian.col(state) = forward_context->hamiltonian_eigenvectors.col(root);
            images.overlap.col(state) = forward_context->overlap_eigenvectors.col(root);
        }
    } else if (!forward_context->structure_matrices.hamiltonian_matrix.empty()) {
        // Dense CI has no Davidson Ritz images. Supply the same accepted-state
        // contract from its resident matrices, only for the selected roots.
        const Eigen::Index n = context->n_structures;
        const Eigen::Map<const Eigen::MatrixXd> hamiltonian(
            forward_context->structure_matrices.hamiltonian_matrix.data(), n, n);
        const Eigen::Map<const Eigen::MatrixXd> overlap(
            forward_context->structure_matrices.overlap_matrix.data(), n, n);
        auto& images = context->selected_state_structure_images.emplace();
        images.hamiltonian.noalias() = hamiltonian * context->selected_state_eigenvectors;
        images.overlap.noalias() = overlap * context->selected_state_eigenvectors;
    }
    const int n_roots = static_cast<int>(forward_context->eigen_result.eigenvalues.size());
    const Eigen::Map<const Eigen::MatrixXd> accepted_roots(
        forward_context->eigen_result.eigenvector_matrix.data(), context->n_structures, n_roots);
    context->root_eigenvectors = accepted_roots;
    if (n_roots == context->n_structures) {
        context->full_structure_eigenvalues = std::move(forward_context->eigen_result.eigenvalues);
    }
    const bool direct_ci =
        context->structure_action && context->structure_action->supports_integral_direction();
    if (context->use_pair_graph_opposite_spin_adjoint && !direct_ci) {
        context->selected_state_matrices =
            build_selected_state_determinant_matrices_from_selected_columns(
                input.structure_data,
                context->selected_state_eigenvectors,
                selected_state_indices,
                normalized_weights,
                context->same_spin_pair_cache);
    }
    return context;
}

void populate_scf_result(const VbScfInput& input,
                         const std::vector<int>& selected_state_indices,
                         const std::vector<double>& normalized_weights,
                         double nuclear_repulsion_energy,
                         const ActiveSpaceGradientForwardContext& forward_context,
                         VbScfResult* scf_result) {
    const auto& prepared_active_space =
        forward_context.timed_active_space_context.prepared_active_space;
    const auto& structure_matrices = forward_context.structure_matrices;
    const auto& eigen_result = forward_context.eigen_result;

    scf_result->n_structures = input.structure_data.n_structures;
    scf_result->nuclear_repulsion_energy = nuclear_repulsion_energy;
    scf_result->selected_state_indices = selected_state_indices;
    scf_result->state_average_weights = normalized_weights;
    scf_result->structure_matrices = structure_matrices;
    scf_result->average_structure_overlap = forward_context.average_structure_overlap;
    scf_result->davidson_diagnostics = forward_context.davidson_diagnostics;
    scf_result->electronic_state_energies = eigen_result.eigenvalues;
    scf_result->eigenvector_matrix = eigen_result.eigenvector_matrix;
    scf_result->structure_overlap_diagonal.assign(
        forward_context.structure_overlap_diagonal.data(),
        forward_context.structure_overlap_diagonal.data() +
            forward_context.structure_overlap_diagonal.size());
    scf_result->structure_overlap_diagonal_exact = forward_context.structure_overlap_diagonal_exact;
    scf_result->overlap_eigenvector_matrix.clear();
    if (forward_context.overlap_eigenvectors.size() != 0) {
        scf_result->overlap_eigenvector_matrix.assign(
            forward_context.overlap_eigenvectors.data(),
            forward_context.overlap_eigenvectors.data() +
                forward_context.overlap_eigenvectors.size());
    }
    scf_result->one_electron_reference_energy = prepared_active_space.one_electron_reference_energy;
    scf_result->electronic_energy = selected_state_average_energy(
        eigen_result.eigenvalues, selected_state_indices, normalized_weights);
    scf_result->total_energy = scf_result->one_electron_reference_energy +
                               scf_result->electronic_energy + nuclear_repulsion_energy;
    scf_result->selected_state_total_energies.resize(selected_state_indices.size(), 0.0);
    for (std::size_t selected_state_offset = 0;
         selected_state_offset < selected_state_indices.size();
         ++selected_state_offset) {
        scf_result->selected_state_total_energies[selected_state_offset] =
            scf_result->one_electron_reference_energy +
            eigen_result.eigenvalues[selected_state_indices[selected_state_offset]] +
            nuclear_repulsion_energy;
    }
}

/** @brief Prepare derivative storage without constructing eigensolver results. */
void initialize_active_space_derivatives(const VbScfInput& input,
                                         const ActiveSpaceGradientForwardContext& forward_context,
                                         ActiveSpaceGradientResult* result) {
    const auto& timings = forward_context.timed_active_space_context.timings;
    const auto& prepared_active_space =
        forward_context.timed_active_space_context.prepared_active_space;

    result->orbital_preparation_wall_time_seconds = timings.orbital_preparation_wall_time_seconds;
    result->ao_effective_one_electron_wall_time_seconds =
        timings.ao_effective_one_electron_wall_time_seconds;
    result->active_one_electron_wall_time_seconds = timings.active_one_electron_wall_time_seconds;
    result->active_two_electron_wall_time_seconds = timings.active_two_electron_wall_time_seconds;
    result->structure_matrix_wall_time_seconds = forward_context.structure_matrix_wall_time_seconds;
    result->eigensolver_wall_time_seconds = forward_context.eigensolver_wall_time_seconds;
    mcscf::diagnostics::ciah_trace(
        "ciah_forward_cost",
        "orbital=%.17g,ao_effective=%.17g,active_one=%.17g,active_two=%.17g,"
        "structure_setup=%.17g,ci_solve=%.17g",
        result->orbital_preparation_wall_time_seconds,
        result->ao_effective_one_electron_wall_time_seconds,
        result->active_one_electron_wall_time_seconds,
        result->active_two_electron_wall_time_seconds,
        result->structure_matrix_wall_time_seconds,
        result->eigensolver_wall_time_seconds);

    result->orbital_preparation_result = prepared_active_space.orbital_result;
    result->ao_effective_one_electron_result =
        prepared_active_space.ao_effective_one_electron_result;
    result->active_orbital_overlap_matrix =
        prepared_active_space.orbital_result.active_orbital_overlap_matrix;
    result->active_space_one_electron_result =
        prepared_active_space.active_space_one_electron_result;
    result->active_space_two_electron_result =
        prepared_active_space.active_space_two_electron_result;
    result->active_orbital_overlap_gradient.assign(
        prepared_active_space.orbital_result.active_orbital_overlap_matrix.size(), 0.0);
    result->active_one_electron_gradient.assign(
        prepared_active_space.active_space_one_electron_result.h1e_act.size(), 0.0);
    result->packed_active_two_electron_gradient.assign(
        xmvb::integrals::packed_active_two_electron_integral_count(
            input.orbital_preparation_input.n_active_orbitals),
        0.0);
}

void initialize_active_space_gradient_result(
    const VbScfInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& normalized_weights,
    double nuclear_repulsion_energy,
    const ActiveSpaceGradientForwardContext& forward_context,
    ActiveSpaceGradientResult* result) {
    initialize_active_space_derivatives(input, forward_context, result);
    populate_scf_result(input,
                        selected_state_indices,
                        normalized_weights,
                        nuclear_repulsion_energy,
                        forward_context,
                        &result->scf_result);
}

void accumulate_additive_vector(const std::vector<double>& partial, std::vector<double>* total) {
    for (std::size_t index = 0; index < total->size(); ++index) {
        (*total)[index] += partial[index];
    }
}

void write_direct_ci_active_gradient(const StructureActiveIntegralAdjoint& adjoint,
                                     int n_active_orbitals,
                                     ActiveSpaceGradientResult* result) {
    const int n_pairs = xmvb::integrals::packed_active_pair_count(n_active_orbitals);
    if (adjoint.overlap.rows() != n_active_orbitals ||
        adjoint.overlap.cols() != n_active_orbitals ||
        adjoint.one_electron.rows() != n_active_orbitals ||
        adjoint.one_electron.cols() != n_active_orbitals || adjoint.pair_kernel.rows() != n_pairs ||
        adjoint.pair_kernel.cols() != n_pairs) {
        throw std::runtime_error("direct-CI active-integral adjoint has inconsistent dimensions");
    }
    result->active_orbital_overlap_gradient.assign(adjoint.overlap.data(),
                                                   adjoint.overlap.data() + adjoint.overlap.size());
    result->active_one_electron_gradient.assign(
        adjoint.one_electron.data(), adjoint.one_electron.data() + adjoint.one_electron.size());
    for (int column = 0; column < n_pairs; ++column) {
        for (int row = 0; row <= column; ++row) {
            result->packed_active_two_electron_gradient
                [xmvb::core::PairIndex::packed_pair_of_pairs_index(row, column)] =
                row == column ? adjoint.pair_kernel(row, column)
                              : adjoint.pair_kernel(row, column) + adjoint.pair_kernel(column, row);
        }
    }
}

void accumulate_active_space_gradient(const VbScfInput& input,
                                      const std::vector<int>& selected_state_indices,
                                      const std::vector<double>& normalized_weights,
                                      ActiveSpaceGradientForwardContext& forward_context,
                                      ActiveSpaceGradientResult* result) {
    const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
    const auto stage_start_time = std::chrono::steady_clock::now();
    const auto& same_spin_pair_cache = forward_context.same_spin_pair_cache;

    const std::vector<double>& selected_state_energies = forward_context.selected_state_energies;
    if (forward_context.structure_action &&
        forward_context.structure_action->supports_integral_direction()) {
        forward_context.structure_adjoint_state =
            forward_context.structure_action->prepare_active_adjoint(
                forward_context.selected_state_eigenvectors,
                normalized_weights,
                selected_state_energies);
        write_direct_ci_active_gradient(forward_context.structure_action->active_integral_adjoint(
                                            *forward_context.structure_adjoint_state),
                                        n_active_orbitals,
                                        result);
        result->adjoint_wall_time_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time)
                .count();
        return;
    }
    const SelectedStateDeterminantMatrices selected_state_matrices =
        build_selected_state_determinant_matrices_from_selected_columns(
            input.structure_data,
            forward_context.selected_state_eigenvectors,
            selected_state_indices,
            normalized_weights,
            same_spin_pair_cache);
    if (!same_spin_pair_cache.has_pair_providers()) {
        throw std::runtime_error(
            "active-space gradient requires pair providers or direct-CI action");
    }
    const auto& prepared_active_space =
        forward_context.timed_active_space_context.prepared_active_space;
    const auto& framed_two_electron = prepared_active_space.active_space_two_electron_result;
    std::optional<xmvb::integrals::ActiveSpaceTwoElectronResult> raw_two_electron_storage;
    const xmvb::integrals::ActiveSpaceTwoElectronResult* raw_two_electron = &framed_two_electron;
    if (framed_two_electron.orbital_transform.size() != 0) {
        raw_two_electron_storage.emplace(
            xmvb::integrals::materialize_raw_active_integrals(framed_two_electron));
        raw_two_electron = &*raw_two_electron_storage;
    }
    const Eigen::MatrixXd raw_h1e = xmvb::integrals::materialize_raw_one_electron(
        prepared_active_space.active_space_one_electron_result.h1e_act,
        framed_two_electron.orbital_transform);
    const SameSpinMatrixBackwardContribution same_spin_contribution =
        build_same_spin_matrix_backward_contribution(
            same_spin_pair_cache,
            selected_state_matrices,
            selected_state_energies,
            n_active_orbitals,
            forward_context.timed_active_space_context.prepared_active_space.orbital_result
                .active_orbital_overlap_matrix,
            raw_h1e,
            *raw_two_electron);
    const OppositeSpinBackwardContribution opposite_spin_contribution =
        build_opposite_spin_backward_contribution(
            same_spin_pair_cache,
            selected_state_matrices,
            n_active_orbitals,
            forward_context.timed_active_space_context.prepared_active_space.orbital_result
                .active_orbital_overlap_matrix,
            raw_h1e,
            *raw_two_electron);
    accumulate_additive_vector(same_spin_contribution.active_orbital_overlap_gradient,
                               &result->active_orbital_overlap_gradient);
    accumulate_additive_vector(same_spin_contribution.active_one_electron_gradient,
                               &result->active_one_electron_gradient);
    accumulate_additive_vector(opposite_spin_contribution.active_orbital_overlap_gradient,
                               &result->active_orbital_overlap_gradient);
    std::vector<double> raw_pair_gradient =
        same_spin_contribution.packed_active_two_electron_gradient;
    accumulate_additive_vector(opposite_spin_contribution.packed_active_two_electron_gradient,
                               &raw_pair_gradient);
    const std::vector<double> framed_pair_gradient =
        xmvb::integrals::materialize_framed_pair_kernel_adjoint(
            raw_pair_gradient, framed_two_electron.orbital_transform);
    accumulate_additive_vector(framed_pair_gradient, &result->packed_active_two_electron_gradient);

    result->adjoint_wall_time_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();
}

} // namespace

struct ActiveSpaceForwardEvaluation::State {
    ActiveSpaceGradientForwardContext forward_context;
    std::vector<int> selected_state_indices;
    std::vector<double> normalized_weights;
    double nuclear_repulsion_energy = 0.0;
    double total_energy = 0.0;
    double wall_time_seconds = 0.0;
    StructureSolveAccuracy structure_solve_accuracy;
};

ActiveSpaceForwardEvaluation::ActiveSpaceForwardEvaluation() = default;
ActiveSpaceForwardEvaluation::~ActiveSpaceForwardEvaluation() = default;
ActiveSpaceForwardEvaluation::ActiveSpaceForwardEvaluation(
    ActiveSpaceForwardEvaluation&&) noexcept = default;
ActiveSpaceForwardEvaluation&
ActiveSpaceForwardEvaluation::operator=(ActiveSpaceForwardEvaluation&&) noexcept = default;

double ActiveSpaceForwardEvaluation::total_energy() const noexcept {
    return state_ == nullptr ? 0.0 : state_->total_energy;
}

double ActiveSpaceForwardEvaluation::wall_time_seconds() const noexcept {
    return state_ == nullptr ? 0.0 : state_->wall_time_seconds;
}

bool ActiveSpaceForwardEvaluation::valid() const noexcept {
    return state_ != nullptr;
}

ActiveSpaceGradientResult
ActiveSpaceGradientEvaluator::evaluate(const VbScfInput& input,
                                       double nuclear_repulsion_energy) const {
    return evaluate(input, {0}, {1.0}, nuclear_repulsion_energy);
}

ActiveSpaceGradientResult
ActiveSpaceGradientEvaluator::evaluate(const VbScfInput& input,
                                       const std::vector<int>& selected_state_indices,
                                       const std::vector<double>& state_average_weights,
                                       double nuclear_repulsion_energy) const {
    const Eigen::MatrixXd no_initial_eigenvectors;
    return evaluate(input,
                    selected_state_indices,
                    state_average_weights,
                    nuclear_repulsion_energy,
                    StructureEigensolver::Dense,
                    StructureSolveAccuracy{},
                    no_initial_eigenvectors);
}

ActiveSpaceGradientResult ActiveSpaceGradientEvaluator::evaluate(
    const VbScfInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy,
    StructureEigensolver structure_eigensolver,
    StructureSolveAccuracy structure_solve_accuracy,
    const Eigen::Ref<const Eigen::MatrixXd>& initial_eigenvectors) const {
    return complete_gradient(input,
                             evaluate_forward(input,
                                              selected_state_indices,
                                              state_average_weights,
                                              nuclear_repulsion_energy,
                                              structure_eigensolver,
                                              structure_solve_accuracy,
                                              initial_eigenvectors));
}

ActiveSpaceForwardEvaluation ActiveSpaceGradientEvaluator::evaluate_forward(
    const VbScfInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy,
    StructureEigensolver structure_eigensolver,
    StructureSolveAccuracy structure_solve_accuracy,
    const Eigen::Ref<const Eigen::MatrixXd>& initial_eigenvectors) const {
    const auto total_start_time = std::chrono::steady_clock::now();
    if (input.structure_data.n_structures <= 0) {
        throw std::invalid_argument("input.structure_data.n_structures must be positive");
    }
    validate_state_selection(
        selected_state_indices, state_average_weights, input.structure_data.n_structures);
    ActiveSpaceForwardEvaluation evaluation;
    evaluation.state_ = std::make_unique<ActiveSpaceForwardEvaluation::State>();
    auto& state = *evaluation.state_;
    state.selected_state_indices = selected_state_indices;
    state.normalized_weights = mcscf::normalize_weights(state_average_weights);
    state.nuclear_repulsion_energy = nuclear_repulsion_energy;
    state.structure_solve_accuracy = structure_solve_accuracy;
    state.forward_context = build_active_space_gradient_forward_context(input,
                                                                        selected_state_indices,
                                                                        structure_eigensolver,
                                                                        structure_solve_accuracy,
                                                                        initial_eigenvectors,
                                                                        structure_builder_,
                                                                        generalized_eigensolver_);
    const auto& prepared = state.forward_context.timed_active_space_context.prepared_active_space;
    state.total_energy =
        prepared.one_electron_reference_energy +
        selected_state_average_energy(state.forward_context.eigen_result.eigenvalues,
                                      state.selected_state_indices,
                                      state.normalized_weights) +
        nuclear_repulsion_energy;
    state.wall_time_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start_time).count();
    return evaluation;
}

ActiveSpaceGradientResult ActiveSpaceGradientEvaluator::complete_gradient(
    const VbScfInput& input, ActiveSpaceForwardEvaluation forward_evaluation) const {
    if (!forward_evaluation.valid()) {
        throw std::invalid_argument("cannot complete an invalid active-space forward evaluation");
    }
    const auto completion_start_time = std::chrono::steady_clock::now();
    auto state = std::move(forward_evaluation.state_);
    retain_structure_action(input, &state->forward_context);
    ActiveSpaceGradientResult result;
    initialize_active_space_gradient_result(input,
                                            state->selected_state_indices,
                                            state->normalized_weights,
                                            state->nuclear_repulsion_energy,
                                            state->forward_context,
                                            &result);
    accumulate_active_space_gradient(input,
                                     state->selected_state_indices,
                                     state->normalized_weights,
                                     state->forward_context,
                                     &result);
    result.second_order_context =
        finalize_active_space_second_order_context(input,
                                                   state->selected_state_indices,
                                                   state->normalized_weights,
                                                   state->structure_solve_accuracy,
                                                   result,
                                                   &state->forward_context);
    result.total_wall_time_seconds =
        state->wall_time_seconds +
        std::chrono::duration<double>(std::chrono::steady_clock::now() - completion_start_time)
            .count();

    return result;
}

ActiveSpaceFixedStructureGradient ActiveSpaceGradientEvaluator::evaluate_fixed_structure(
    const VbScfInput& input,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy,
    const Eigen::Ref<const Eigen::MatrixXd>& structure_coefficients,
    const SelectedMetric* metric,
    bool retain_context) const {
    const auto total_start_time = std::chrono::steady_clock::now();
    validate_state_selection(
        selected_state_indices, state_average_weights, input.structure_data.n_structures);
    const int n_states = static_cast<int>(selected_state_indices.size());
    if (structure_coefficients.rows() != input.structure_data.n_structures ||
        structure_coefficients.cols() != n_states || !structure_coefficients.allFinite()) {
        throw std::invalid_argument(
            "fixed structure coefficients have inconsistent dimensions or values");
    }
    const std::vector<double> normalized_weights = mcscf::normalize_weights(state_average_weights);

    ActiveSpaceGradientForwardContext context;
    context.timed_active_space_context = prepare_timed_active_space_context(input);
    const auto& prepared = context.timed_active_space_context.prepared_active_space;
    context.same_spin_pair_cache =
        build_orthogonal_spin_topology(input.structure_data.alpha(),
                                       input.structure_data.beta(),
                                       input.orbital_preparation_input.n_active_orbitals);

    const auto structure_start_time = std::chrono::steady_clock::now();
    // Prescribed coefficients need H/S actions and adjoints, not a Davidson
    // diagonal, root solve, or a synthesized eigensystem for result packaging.
    std::optional<StructureDiagonal> diagonal;
    if (retain_context)
        diagonal = structure_builder_.build_davidson_jacobi_preconditioner(
            input.structure_data.alpha(),
            input.structure_data.beta(),
            input.structure_data.incidence(),
            prepared.orbital_result.active_orbital_overlap_matrix,
            prepared.active_space_one_electron_result.h1e_act,
            input.orbital_preparation_input.n_active_orbitals,
            prepared.active_space_two_electron_result,
            input.structure_data.n_structures,
            context.same_spin_pair_cache);
    context.structure_action = std::make_unique<StructureAction>(
        input.structure_data.incidence(),
        input.structure_data.n_structures,
        context.same_spin_pair_cache,
        prepared.orbital_result.active_orbital_overlap_matrix,
        prepared.active_space_one_electron_result.h1e_act,
        prepared.active_space_two_electron_result,
        input.orbital_preparation_input.n_active_orbitals,
        diagonal ? &*diagonal : nullptr,
        retain_context ? StructureActionMode::Davidson : StructureActionMode::ActionOnly);
    context.structure_matrix_wall_time_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - structure_start_time)
            .count();

    const auto action_start_time = std::chrono::steady_clock::now();
    auto states = evaluate_fixed_states(
        *context.structure_action, structure_coefficients, normalized_weights, metric);
    context.selected_state_eigenvectors = std::move(states.coefficients);
    context.selected_state_energies = std::move(states.energies);
    double electronic_energy = 0.0;
    for (int state = 0; state < n_states; ++state)
        electronic_energy += normalized_weights[state] * context.selected_state_energies[state];
    const double action_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - action_start_time).count();

    ActiveSpaceFixedStructureGradient fixed;
    initialize_active_space_derivatives(input, context, &fixed.gradient);
    fixed.gradient.scf_result.total_energy =
        electronic_energy + prepared.one_electron_reference_energy + nuclear_repulsion_energy;
    if (states.integral_adjoint)
        write_direct_ci_active_gradient(*states.integral_adjoint,
                                        input.orbital_preparation_input.n_active_orbitals,
                                        &fixed.gradient);
    else
        accumulate_active_space_gradient(
            input, selected_state_indices, normalized_weights, context, &fixed.gradient);
    fixed.gradient.total_wall_time_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start_time).count();
    if (mcscf::diagnostics::ciah_event_trace_enabled())
        mcscf::diagnostics::ciah_trace(
            "ciah_keyframe_active_cost",
            "seconds=%.17g,setup=%.17g,action=%.17g,adjoint=%.17g,actions=1,diagonal_bytes=%zu",
            fixed.gradient.total_wall_time_seconds,
            context.structure_matrix_wall_time_seconds,
            action_seconds,
            fixed.gradient.adjoint_wall_time_seconds,
            context.structure_action->storage().diagonal_bytes);
    fixed.normalized_coefficients = context.selected_state_eigenvectors;
    fixed.structure_residuals = std::move(states.residuals);
    fixed.coefficient_gradient = std::move(states.coefficient_gradient);
    if (retain_context) {
        fixed.gradient.second_order_context =
            finalize_active_space_second_order_context(input,
                                                       selected_state_indices,
                                                       normalized_weights,
                                                       StructureSolveAccuracy{},
                                                       fixed.gradient,
                                                       &context);
        auto& accepted = *fixed.gradient.second_order_context;
        accepted.selected_state_structure_images =
            accepted.structure_action->apply(accepted.selected_state_eigenvectors);
    }
    return fixed;
}

ActiveSpaceGradientResult ActiveSpaceGradientEvaluator::evaluate(
    const VbScfInput& input,
    TimedPreparedActiveSpaceContext timed_prepared_active_space_context,
    const std::vector<int>& selected_state_indices,
    const std::vector<double>& state_average_weights,
    double nuclear_repulsion_energy) const {
    const auto total_start_time = std::chrono::steady_clock::now();
    if (input.structure_data.n_structures <= 0) {
        throw std::invalid_argument("input.structure_data.n_structures must be positive");
    }
    validate_state_selection(
        selected_state_indices, state_average_weights, input.structure_data.n_structures);
    const std::vector<double> normalized_weights = mcscf::normalize_weights(state_average_weights);

    const Eigen::MatrixXd no_initial_eigenvectors;
    auto forward_context =
        build_active_space_gradient_forward_context(input,
                                                    std::move(timed_prepared_active_space_context),
                                                    selected_state_indices,
                                                    StructureEigensolver::Dense,
                                                    StructureSolveAccuracy{},
                                                    no_initial_eigenvectors,
                                                    structure_builder_,
                                                    generalized_eigensolver_);
    ActiveSpaceGradientResult result;
    initialize_active_space_gradient_result(input,
                                            selected_state_indices,
                                            normalized_weights,
                                            nuclear_repulsion_energy,
                                            forward_context,
                                            &result);
    accumulate_active_space_gradient(
        input, selected_state_indices, normalized_weights, forward_context, &result);
    result.second_order_context =
        finalize_active_space_second_order_context(input,
                                                   selected_state_indices,
                                                   normalized_weights,
                                                   StructureSolveAccuracy{},
                                                   result,
                                                   &forward_context);
    result.total_wall_time_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start_time).count();

    return result;
}

ActiveSpaceGradientResult ActiveSpaceGradientEvaluator::evaluate_with_fixed_active_space_adjoint(
    const VbScfInput& input,
    const AcceptedPointContext& accepted_point_context,
    double nuclear_repulsion_energy) const {
    const auto total_start_time = std::chrono::steady_clock::now();
    if (input.structure_data.n_structures <= 0) {
        throw std::invalid_argument("input.structure_data.n_structures must be positive");
    }

    auto timed_active_space_context = prepare_timed_active_space_context(
        input,
        &accepted_point_context.prepared_active_space.active_space_two_electron_result
             .orbital_transform);

    ActiveSpaceGradientResult result;
    initialize_active_space_gradient_probe_result(
        input, std::move(timed_active_space_context), &result);
    if (result.active_orbital_overlap_gradient.size() !=
            accepted_point_context.active_orbital_overlap_gradient.size() ||
        result.active_one_electron_gradient.size() !=
            accepted_point_context.active_one_electron_gradient.size() ||
        result.packed_active_two_electron_gradient.size() !=
            accepted_point_context.packed_active_two_electron_gradient.size()) {
        throw std::invalid_argument(
            "accepted-point active-space adjoint dimensions do not match the trial orbital space");
    }

    // The accepted-point active-space adjoint is held fixed while the lower
    // orbital-preparation and integral layers are rebuilt at the probe point.
    result.active_orbital_overlap_gradient = accepted_point_context.active_orbital_overlap_gradient;
    result.active_one_electron_gradient = accepted_point_context.active_one_electron_gradient;
    result.packed_active_two_electron_gradient =
        accepted_point_context.packed_active_two_electron_gradient;
    result.total_wall_time_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start_time).count();
    (void)nuclear_repulsion_energy;
    return result;
}

} // namespace xmvb::vb
