#include "scf/driver/solver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

#include "linalg/eigensolver.hpp"
#include "scf/mixing/diis.hpp"
#include "scf/optimization/ciah.hpp"

namespace xmvb::scf {
namespace {

Eigen::MatrixXd symmetrize(const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
    return 0.5 * (matrix + matrix.transpose()).eval();
}

void diagonalize(const linalg::GeneralizedEigensolver& eigensolver,
                 const Eigen::Ref<const Eigen::MatrixXd>& fock,
                 const Eigen::Ref<const Eigen::MatrixXd>& overlap,
                 Eigen::MatrixXd* orbitals,
                 Eigen::VectorXd* orbital_energies) {
    const int n_basis = static_cast<int>(fock.rows());
    auto eigenpairs =
        eigensolver.solve_dense(std::span<const double>(fock.data(), fock.size()),
                                std::span<const double>(overlap.data(), overlap.size()),
                                n_basis);
    *orbital_energies = Eigen::Map<const Eigen::VectorXd>(eigenpairs.eigenvalues.data(), n_basis);
    *orbitals =
        Eigen::Map<const Eigen::MatrixXd>(eigenpairs.eigenvector_matrix.data(), n_basis, n_basis);
}

Eigen::MatrixXd occupied_density(const Eigen::Ref<const Eigen::MatrixXd>& orbitals,
                                 int n_occupied,
                                 double occupation) {
    return occupation *
           (orbitals.leftCols(n_occupied) * orbitals.leftCols(n_occupied).transpose()).eval();
}

double max_abs_difference(const Eigen::Ref<const Eigen::MatrixXd>& left,
                          const Eigen::Ref<const Eigen::MatrixXd>& right) {
    return (left - right).cwiseAbs().maxCoeff();
}

void validate_matrices(const Eigen::Ref<const Eigen::MatrixXd>& overlap,
                       const Eigen::Ref<const Eigen::MatrixXd>& core) {
    const int n_basis = static_cast<int>(core.rows());
    if (n_basis <= 0 || core.cols() != n_basis || overlap.rows() != n_basis ||
        overlap.cols() != n_basis) {
        throw std::invalid_argument("SCF AO matrix dimensions do not match");
    }
}

void validate_options(const Options& options, const char* message) {
    if (options.max_iterations <= 0 || !(options.density_tolerance > 0.0) ||
        options.old_density_weight < 0.0 || options.old_density_weight >= 1.0 ||
        options.diis_history < 0 || options.diis_start_iteration < 0) {
        throw std::invalid_argument(message);
    }
}

void validate_initial_density(const Eigen::MatrixXd& density, int n_basis) {
    if (density.size() != 0 && (density.rows() != n_basis || density.cols() != n_basis)) {
        throw std::invalid_argument("initial SCF density dimensions do not match");
    }
}

// Restricted mode uses one channel with occupation two; unrestricted mode
// uses two channels with occupation one. Both follow this same iteration loop.
UnrestrictedResult solve_channels(int n_channels,
                                  const std::array<int, 2>& occupied,
                                  double occupation,
                                  const Eigen::Ref<const Eigen::MatrixXd>& overlap,
                                  const Eigen::Ref<const Eigen::MatrixXd>& core,
                                  const EvaluateUnrestricted& evaluate,
                                  const Options& options,
                                  const SpinMatrices& initial_density,
                                  const ApplyUnrestrictedResponse& response) {
    const int n_basis = static_cast<int>(core.rows());
    const Eigen::MatrixXd symmetric_overlap = symmetrize(overlap);
    const Eigen::MatrixXd symmetric_core = symmetrize(core);
    linalg::GeneralizedEigensolver eigensolver;
    UnrestrictedResult result;
    SpinMatrices density;
    diagonalize(eigensolver,
                symmetric_core,
                symmetric_overlap,
                &result.orbitals[0],
                &result.orbital_energies[0]);
    for (int spin = 0; spin < n_channels; ++spin) {
        if (spin != 0) {
            result.orbitals[spin] = result.orbitals[0];
            result.orbital_energies[spin] = result.orbital_energies[0];
        }
        density[spin] = initial_density[spin].size() != 0
                            ? symmetrize(initial_density[spin]).eval()
                            : occupied_density(result.orbitals[spin], occupied[spin], occupation);
    }
    Diis diis(options.diis_history);
    std::optional<UnrestrictedEvaluation> queued;
    std::optional<double> previous_error, previous_prediction, previous_energy;
    SpinMatrices density_change;

    const auto checked_evaluation = [&](const SpinMatrices& at_density) {
        UnrestrictedEvaluation physical = evaluate(at_density);
        for (int spin = 0; spin < n_channels; ++spin) {
            if (physical.fock[spin].rows() != n_basis || physical.fock[spin].cols() != n_basis) {
                throw std::invalid_argument("SCF evaluation returned an invalid Fock shape");
            }
            if (!physical.fock[spin].allFinite()) {
                throw std::runtime_error("SCF evaluation returned a non-finite result");
            }
        }
        if (!std::isfinite(physical.electronic_energy)) {
            throw std::runtime_error("SCF evaluation returned a non-finite result");
        }
        return physical;
    };

    for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
        UnrestrictedEvaluation physical = queued ? std::move(*queued) : checked_evaluation(density);
        queued.reset();
        Eigen::MatrixXd joined_fock(n_basis, n_channels * n_basis);
        Eigen::MatrixXd joined_error(n_basis, n_channels * n_basis);
        double residual = 0.0;
        for (int spin = 0; spin < n_channels; ++spin) {
            joined_fock.middleCols(spin * n_basis, n_basis) = physical.fock[spin];
            const Eigen::MatrixXd error =
                commutator_error(physical.fock[spin], density[spin], symmetric_overlap);
            joined_error.middleCols(spin * n_basis, n_basis) = error;
            residual = std::max(residual, error.cwiseAbs().maxCoeff());
        }
        diis.push(joined_fock, joined_error);
        const double error_norm = joined_error.stableNorm();
        // Compare model prediction with the already available physical observation.
        // Escalate only when neither energy nor residual makes progress despite
        // DIIS predicting a residual decrease. Ordinary energy descent stays cheap.
        const bool model_failed = response && previous_error && previous_prediction &&
                                  previous_energy && *previous_prediction < *previous_error &&
                                  error_norm >= *previous_error &&
                                  physical.electronic_energy >= *previous_energy;

        Eigen::MatrixXd orbital_fock;
        double prediction = error_norm;
        const bool use_diis = iteration + 1 >= options.diis_start_iteration &&
                              diis.extrapolate(&orbital_fock, &prediction);
        if (!use_diis) {
            orbital_fock = std::move(joined_fock);
        }
        SpinMatrices candidate;
        bool density_close = true;
        for (int spin = 0; spin < n_channels; ++spin) {
            diagonalize(eigensolver,
                        orbital_fock.middleCols(spin * n_basis, n_basis),
                        symmetric_overlap,
                        &result.orbitals[spin],
                        &result.orbital_energies[spin]);
            candidate[spin] = occupied_density(result.orbitals[spin], occupied[spin], occupation);
            density_close = density_close && max_abs_difference(candidate[spin], density[spin]) <
                                                 options.density_tolerance;
        }
        result.iterations = iteration + 1;

        if (model_failed && !(density_close && residual < options.density_tolerance)) {
            UnrestrictedEvaluation origin = checked_evaluation(candidate);
            const OrbitalModel model(result.orbitals,
                                     occupied,
                                     n_channels,
                                     occupation,
                                     candidate,
                                     origin.fock,
                                     response);
            SpinMatrices change;
            for (int spin = 0; spin < n_channels; ++spin)
                change[spin] = density_change[spin].size() != 0
                                   ? density_change[spin]
                                   : (candidate[spin] - density[spin]).eval();
            auto trial = ciah_step(model,
                                   checked_evaluation,
                                   candidate,
                                   origin,
                                   symmetric_overlap,
                                   model.displacement(change, symmetric_overlap),
                                   &result.response_actions);
            if (trial) {
                ++result.orbital_steps;
                result.orbitals = std::move(trial->orbitals);
                for (int spin = 0; spin < n_channels; ++spin)
                    density_change[spin] = trial->density[spin] - density[spin];
                density = std::move(trial->density);
                queued = std::move(trial->physical);
                diis.clear();
                previous_error.reset();
                previous_prediction.reset();
                previous_energy.reset();
                if (iteration + 1 < options.max_iterations)
                    continue;
                result.density = std::move(density);
                result.fock = std::move(queued->fock);
                result.electronic_energy = queued->electronic_energy;
                for (int spin = 0; spin < n_channels; ++spin) {
                    result.orbital_energies[spin] = (result.orbitals[spin].transpose() *
                                                     result.fock[spin] * result.orbitals[spin])
                                                        .diagonal();
                    result.residual = std::max(
                        result.residual,
                        commutator_error(result.fock[spin], result.density[spin], symmetric_overlap)
                            .cwiseAbs()
                            .maxCoeff());
                }
                return result;
            }
        }
        previous_error = error_norm;
        previous_energy = physical.electronic_energy;
        previous_prediction = use_diis ? std::optional<double>(prediction) : std::nullopt;

        const bool residual_close = residual < options.density_tolerance;
        const bool last_iteration = iteration + 1 == options.max_iterations;
        if ((density_close && residual_close) || last_iteration) {
            // DIIS changes the diagonalized Fock, never the reported physical one.
            UnrestrictedEvaluation candidate_physical = checked_evaluation(candidate);
            double candidate_residual = 0.0;
            for (int spin = 0; spin < n_channels; ++spin) {
                candidate_residual =
                    std::max(candidate_residual,
                             commutator_error(
                                 candidate_physical.fock[spin], candidate[spin], symmetric_overlap)
                                 .cwiseAbs()
                                 .maxCoeff());
            }
            result.converged =
                density_close && residual_close && candidate_residual < options.density_tolerance;
            if (result.converged || last_iteration) {
                result.density = std::move(candidate);
                result.fock = std::move(candidate_physical.fock);
                result.electronic_energy = candidate_physical.electronic_energy;
                result.residual = candidate_residual;
                return result;
            }
        }

        for (int spin = 0; spin < n_channels; ++spin) {
            density_change[spin] =
                (1.0 - options.old_density_weight) * (candidate[spin] - density[spin]);
            density[spin] = options.old_density_weight * density[spin] +
                            (1.0 - options.old_density_weight) * candidate[spin];
        }
    }
    throw std::logic_error("SCF iteration loop exited unexpectedly");
}

} // namespace

Result solve_restricted(int n_electrons,
                        const Eigen::Ref<const Eigen::MatrixXd>& overlap,
                        const Eigen::Ref<const Eigen::MatrixXd>& core,
                        const Evaluate& evaluate,
                        const Options& options,
                        const Eigen::MatrixXd& initial_density,
                        const ApplyResponse& response) {
    validate_matrices(overlap, core);
    const int n_basis = static_cast<int>(core.rows());
    if (n_electrons < 0 || n_electrons % 2 != 0 || n_electrons > 2 * n_basis) {
        throw std::invalid_argument(
            "restricted SCF requires an even electron count within basis capacity");
    }
    if (!evaluate) {
        throw std::invalid_argument("SCF evaluation callback must be set");
    }
    validate_options(options, "invalid restricted SCF options");
    validate_initial_density(initial_density, n_basis);
    SpinMatrices initial;
    initial[0] = initial_density;
    const EvaluateUnrestricted adapted = [&evaluate](const SpinMatrices& density) {
        Evaluation physical = evaluate(density[0]);
        UnrestrictedEvaluation adapted_physical;
        adapted_physical.fock[0] = std::move(physical.fock);
        adapted_physical.electronic_energy = physical.electronic_energy;
        return adapted_physical;
    };
    ApplyUnrestrictedResponse adapted_response;
    if (response)
        adapted_response = [&response](const SpinMatrices& reference,
                                       const SpinMatrices& direction) {
            SpinMatrices result;
            result[0] = response(reference[0], direction[0]);
            return result;
        };
    UnrestrictedResult shared = solve_channels(
        1, {n_electrons / 2, 0}, 2.0, overlap, core, adapted, options, initial, adapted_response);
    Result result;
    result.converged = shared.converged;
    result.iterations = shared.iterations;
    result.orbital_steps = shared.orbital_steps;
    result.response_actions = shared.response_actions;
    result.residual = shared.residual;
    result.electronic_energy = shared.electronic_energy;
    result.orbitals = std::move(shared.orbitals[0]);
    result.orbital_energies = std::move(shared.orbital_energies[0]);
    result.density = std::move(shared.density[0]);
    result.fock = std::move(shared.fock[0]);
    return result;
}

UnrestrictedResult solve_unrestricted(int n_alpha,
                                      int n_beta,
                                      const Eigen::Ref<const Eigen::MatrixXd>& overlap,
                                      const Eigen::Ref<const Eigen::MatrixXd>& core,
                                      const EvaluateUnrestricted& evaluate,
                                      const Options& options,
                                      const SpinMatrices& initial_density,
                                      const ApplyUnrestrictedResponse& response) {
    validate_matrices(overlap, core);
    const int n_basis = static_cast<int>(core.rows());
    if (n_alpha < 0 || n_beta < 0 || n_alpha > n_basis || n_beta > n_basis) {
        throw std::invalid_argument("unrestricted SCF spin occupations exceed basis capacity");
    }
    if (!evaluate) {
        throw std::invalid_argument("SCF evaluation callback must be set");
    }
    validate_options(options, "invalid unrestricted SCF options");
    for (const auto& density : initial_density) {
        validate_initial_density(density, n_basis);
    }
    return solve_channels(
        2, {n_alpha, n_beta}, 1.0, overlap, core, evaluate, options, initial_density, response);
}

} // namespace xmvb::scf
