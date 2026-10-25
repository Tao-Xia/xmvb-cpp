#include "vbscf/integrals/active/space.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "vbscf/integrals/ao/ri/cache.hpp"
#include "integrals/active/exact.hpp"
#include "integrals/active/ri.hpp"
#include "integrals/response/adjoint.hpp"
#include "integrals/response/ri.hpp"
#include "integrals/active/one_electron.hpp"
#include "vbscf/orbitals/preparation/preparer.hpp"
#include "vbscf/integrals/ao/one_electron/builder.hpp"

namespace xmvb::vb {

double compute_one_electron_reference_energy(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_density_matrix,
    const Eigen::Ref<const Eigen::MatrixXd>& ao_effective_h1e,
    const Eigen::Ref<const Eigen::MatrixXd>& ao_core_hamiltonian_matrix,
    int n_basis_functions) {
    if (n_basis_functions <= 0) {
        throw std::invalid_argument("n_basis_functions must be positive");
    }

    if (inactive_density_matrix.rows() != n_basis_functions ||
        inactive_density_matrix.cols() != n_basis_functions ||
        ao_effective_h1e.rows() != n_basis_functions ||
        ao_effective_h1e.cols() != n_basis_functions ||
        ao_core_hamiltonian_matrix.rows() != n_basis_functions ||
        ao_core_hamiltonian_matrix.cols() != n_basis_functions) {
        throw std::invalid_argument("one-electron reference energy input size mismatch");
    }
    return (inactive_density_matrix.array() *
            (ao_effective_h1e.array() + ao_core_hamiltonian_matrix.array()))
        .sum();
}

TimedPreparedActiveSpaceContext
prepare_timed_active_space_context(const VbScfInput& input,
                                   const Eigen::MatrixXd* fixed_integral_frame) {
    TimedPreparedActiveSpaceContext timed_context;
    auto& context = timed_context.prepared_active_space;
    const bool use_ri_active_space =
        input.standard_two_electron_mode == StandardTwoElectronMode::ResolutionOfIdentity;
    const xmvb::integrals::RiAoFactorization* ao_ri_result = nullptr;
    if (use_ri_active_space) {
        ao_ri_result = &ensure_vbscf_input_ri_cache(input);
    }

    auto stage_start_time = std::chrono::steady_clock::now();
    context.n_inactive_doubly_occupied_orbitals =
        (input.orbital_preparation_input.n_total_electrons -
         input.orbital_preparation_input.n_active_electrons) /
        2;
    context.orbital_result = ActiveSpaceOrbitalPreparer{}.prepare(input.orbital_preparation_input);
    timed_context.timings.orbital_preparation_wall_time_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();

    stage_start_time = std::chrono::steady_clock::now();
    if (use_ri_active_space) {
        context.ao_effective_one_electron_result = AoEffectiveOneElectronBuilder{}.build(
            context.orbital_result,
            input.ao_integral_input.ao_core_hamiltonian_matrix,
            *ao_ri_result,
            input.ao_integral_input.n_basis_functions,
            context.n_inactive_doubly_occupied_orbitals);
    } else {
        if (input.ao_integral_input.pair_graph.empty()) {
            throw std::invalid_argument(
                "standard exact AO h1e path requires materialized AO two-electron integrals");
        }
        context.ao_effective_one_electron_result = AoEffectiveOneElectronBuilder{}.build(
            context.orbital_result.inactive_density_matrix, input.ao_integral_input);
    }
    timed_context.timings.ao_effective_one_electron_wall_time_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();

    stage_start_time = std::chrono::steady_clock::now();
    const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
    const int n_basis_functions = input.orbital_preparation_input.n_basis_functions;
    Eigen::MatrixXd integral_frame;
    if (fixed_integral_frame != nullptr) {
        integral_frame = *fixed_integral_frame;
    } else {
        const auto& overlap_storage = context.orbital_result.active_orbital_overlap_matrix;
        if (n_active_orbitals <= 0 ||
            overlap_storage.size() !=
                static_cast<std::size_t>(n_active_orbitals) * n_active_orbitals) {
            throw std::invalid_argument("active integral frame overlap size mismatch");
        }
        const Eigen::Map<const Eigen::MatrixXd> active_overlap(
            overlap_storage.data(), n_active_orbitals, n_active_orbitals);
        Eigen::LLT<Eigen::MatrixXd> cholesky(active_overlap);
        if (cholesky.info() != Eigen::Success) {
            throw std::runtime_error("active integral frame overlap is not positive definite");
        }
        integral_frame = cholesky.matrixU();
    }

    const auto raw_active_coefficients = context.orbital_result.auxiliary_orbital_matrix.middleCols(
        context.n_inactive_doubly_occupied_orbitals, n_active_orbitals);
    Eigen::MatrixXd framed_active_coefficients;
    if (integral_frame.size() != 0) {
        if (integral_frame.rows() != n_active_orbitals ||
            integral_frame.cols() != n_active_orbitals || !integral_frame.allFinite()) {
            throw std::invalid_argument("fixed active integral frame is invalid");
        }
        const Eigen::MatrixXd inverse_frame = integral_frame.triangularView<Eigen::Upper>().solve(
            Eigen::MatrixXd::Identity(n_active_orbitals, n_active_orbitals));
        framed_active_coefficients = raw_active_coefficients * inverse_frame;
    }
    // Keep the shared frame preparation in the existing 2e timing bucket.
    const double frame_wall_time =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();
    const auto one_electron_start = std::chrono::steady_clock::now();
    context.active_space_one_electron_result = integrals::ActiveSpaceOneElectronBuilder{}.build(
        context.ao_effective_one_electron_result.ao_effective_h1e,
        integral_frame.size() ? framed_active_coefficients
                              : Eigen::MatrixXd(raw_active_coefficients));
    timed_context.timings.active_one_electron_wall_time_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - one_electron_start)
            .count();
    stage_start_time = std::chrono::steady_clock::now();
    if (use_ri_active_space) {
        context.active_space_two_electron_result =
            integral_frame.size() != 0
                ? integrals::transform_ri_active_integrals(*ao_ri_result,
                                                           framed_active_coefficients)
                : integrals::transform_ri_active_integrals(*ao_ri_result, raw_active_coefficients);
    } else {
        if (integral_frame.size() != 0) {
            std::vector<int> row_offsets(n_basis_functions + 1, 0);
            std::vector<int> orbital_indices;
            std::vector<double> values;
            for (int row = 0; row < n_basis_functions; ++row) {
                for (int orbital = 0; orbital < n_active_orbitals; ++orbital) {
                    const double value = framed_active_coefficients(row, orbital);
                    if (value != 0.0) {
                        orbital_indices.push_back(orbital);
                        values.push_back(value);
                    }
                }
                row_offsets[row + 1] = static_cast<int>(values.size());
            }
            context.active_space_two_electron_result = integrals::transform_exact_active_integrals(
                input.ao_integral_input, row_offsets, orbital_indices, values, n_active_orbitals);
        } else {
            context.active_space_two_electron_result = integrals::transform_exact_active_integrals(
                input.ao_integral_input,
                context.orbital_result.active_sparse_row_offsets,
                context.orbital_result.active_sparse_orbital_indices,
                context.orbital_result.active_sparse_values,
                n_active_orbitals);
        }
    }
    context.active_space_two_electron_result.orbital_transform = std::move(integral_frame);
    timed_context.timings.active_two_electron_wall_time_seconds =
        frame_wall_time +
        std::chrono::duration<double>(std::chrono::steady_clock::now() - stage_start_time).count();
    context.one_electron_reference_energy = compute_one_electron_reference_energy(
        context.orbital_result.inactive_density_matrix,
        context.ao_effective_one_electron_result.ao_effective_h1e,
        input.ao_integral_input.ao_core_hamiltonian_matrix,
        input.ao_integral_input.n_basis_functions);
    return timed_context;
}

Eigen::MatrixXd pullback_active_integrals(const VbScfInput& input,
                                          const integrals::ActiveSpaceTwoElectronResult& accepted,
                                          const std::vector<double>& packed_gradient) {
    Eigen::MatrixXd framed_gradient;
    if (accepted.representation == integrals::ActiveSpaceTwoElectronRepresentation::PackedExact) {
        framed_gradient = integrals::backpropagate_exact_packed_active_two_electron_gradient(
            packed_gradient, input.ao_integral_input, accepted);
    } else {
        const auto cache = integrals::build_ri_active_two_electron_response_cache(
            ensure_vbscf_input_ri_cache(input),
            accepted,
            input.orbital_preparation_input.n_active_orbitals);
        framed_gradient =
            integrals::backpropagate_ri_packed_active_two_electron_gradient(packed_gradient, cache);
    }
    if (accepted.orbital_transform.size() == 0)
        return framed_gradient;
    const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
    if (accepted.orbital_transform.rows() != n_active_orbitals ||
        accepted.orbital_transform.cols() != n_active_orbitals) {
        throw std::invalid_argument("active two-electron pullback frame size mismatch");
    }
    const Eigen::MatrixXd inverse_frame =
        accepted.orbital_transform.triangularView<Eigen::Upper>().solve(
            Eigen::MatrixXd::Identity(n_active_orbitals, n_active_orbitals));
    return framed_gradient * inverse_frame.transpose();
}

} // namespace xmvb::vb
