#include "vbscf/optimization/preconditioners/hessian_diagonal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <chrono>
#include <stdexcept>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include "parallel/openmp.hpp"
#include "vbscf/optimization/preconditioners/density.hpp"
#include "vbscf/core/contracts/input.hpp"
#include "mcscf/optimization/trace.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "core/pair_index.hpp"
#include "integrals/response/types.hpp"
#include "integrals/response/ao_pair_operator.hpp"
#include "integrals/active/pair_map.hpp"
#include "integrals/active/ri_transform.hpp"
#include "integrals/ao/ri.hpp"

namespace xmvb::vb {
using mcscf::AnalyticDiagonalIntermediates;
using mcscf::assemble_rotation_diagonal;
using mcscf::diagonal_gradient;
using mcscf::DiagonalContractions;
using mcscf::unpack_symmetric_active_two_rdm;
namespace {

Eigen::MatrixXd build_pair_transform(const Eigen::Ref<const Eigen::MatrixXd>& orbital_transform) {
    const int n = static_cast<int>(orbital_transform.rows());
    if (n <= 0 || orbital_transform.cols() != n) {
        throw std::invalid_argument("orbital pair transform must be square");
    }
    const int n_pairs = n * (n + 1) / 2;
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n_pairs, n_pairs);
    for (int new_p = 0; new_p < n; ++new_p) {
        for (int new_q = 0; new_q <= new_p; ++new_q) {
            const int new_pair = xmvb::core::PairIndex::packed_pair_index(new_p, new_q);
            for (int old_p = 0; old_p < n; ++old_p) {
                for (int old_q = 0; old_q <= old_p; ++old_q) {
                    const int old_pair = xmvb::core::PairIndex::packed_pair_index(old_p, old_q);
                    double value =
                        orbital_transform(old_p, new_p) * orbital_transform(old_q, new_q);
                    if (old_p != old_q) {
                        value += orbital_transform(old_q, new_p) * orbital_transform(old_p, new_q);
                    }
                    result(old_pair, new_pair) = value;
                }
            }
        }
    }
    return result;
}

xmvb::integrals::ExactCtxPairMatrix
apply_ao_pair_operator(const VbScfInput& input,
                       const Eigen::Ref<const xmvb::integrals::ExactCtxPairMatrix>& coefficients) {
    xmvb::integrals::ExactCtxPairMatrix result;
    const int n_bf = input.orbital_preparation_input.n_basis_functions;
    if (input.standard_two_electron_mode == StandardTwoElectronMode::ResolutionOfIdentity) {
        if (!input.ri_factorization) {
            throw std::invalid_argument("RI diagonal requires accepted AO factors");
        }
        const auto& factors = input.ri_factorization->metric_whitened_ao_pair_factors;
        result.noalias() = factors.transpose() * (factors * coefficients);
    } else {
        xmvb::integrals::detail::apply_exact_ao_pair_kernel(
            input.ao_integral_input,
            coefficients,
            n_bf,
            static_cast<std::size_t>(coefficients.cols()),
            &result);
    }
    return result;
}

std::vector<double> transform_dm2(const std::vector<double>& source,
                                  const Eigen::Ref<const Eigen::MatrixXd>& transform) {
    const int n = static_cast<int>(transform.rows());
    std::vector<double> current = source;
    std::vector<double> next(current.size());
    for (int axis = 0; axis < 4; ++axis) {
        std::fill(next.begin(), next.end(), 0.0);
        for (int p = 0; p < n; ++p)
            for (int q = 0; q < n; ++q)
                for (int r = 0; r < n; ++r)
                    for (int s = 0; s < n; ++s) {
                        const int output[4] = {p, q, r, s};
                        double value = 0.0;
                        for (int old = 0; old < n; ++old) {
                            int input_index[4] = {p, q, r, s};
                            input_index[axis] = old;
                            value += transform(output[axis], old) *
                                     current[mcscf::orbital_diagonal_tensor4_index(input_index[0],
                                                                                   input_index[1],
                                                                                   input_index[2],
                                                                                   input_index[3],
                                                                                   n)];
                        }
                        next[mcscf::orbital_diagonal_tensor4_index(p, q, r, s, n)] = value;
                    }
        current.swap(next);
    }
    return current;
}

std::vector<double> pack_symmetric_dm2(const std::vector<double>& dm2, int n) {
    std::vector<double> packed(mcscf::orbital_diagonal_packed_eri_count(n), 0.0);
    for (int p = 0; p < n; ++p) {
        for (int q = 0; q <= p; ++q) {
            const int pq = xmvb::core::PairIndex::packed_pair_index(p, q);
            for (int r = 0; r < n; ++r) {
                for (int s = 0; s <= r; ++s) {
                    const int rs = xmvb::core::PairIndex::packed_pair_index(r, s);
                    if (rs > pq)
                        continue;
                    const int multiplicity =
                        (p == q ? 1 : 2) * (r == s ? 1 : 2) * (pq == rs ? 1 : 2);
                    packed[xmvb::core::PairIndex::packed_pair_of_pairs_index(pq, rs)] =
                        0.5 * multiplicity *
                        dm2[mcscf::orbital_diagonal_tensor4_index(p, q, r, s, n)];
                }
            }
        }
    }
    return packed;
}

} // namespace

AnalyticOrbitalDiagonal build_analytic_orbital_diagonal(const VbScfInput& input,
                                                        const AcceptedPointContext& accepted) {
    const bool trace = mcscf::diagnostics::ciah_event_trace_enabled();
    const auto start =
        trace ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto& orbital_input = input.orbital_preparation_input;
    const auto& prepared = accepted.prepared_active_space;
    const auto& orbitals = prepared.orbital_result;
    const int n_bf = orbital_input.n_basis_functions;
    const int n_core = prepared.n_inactive_doubly_occupied_orbitals;
    const int n_active = accepted.n_active_orbitals;
    const int n_occ = n_core + n_active;
    if (n_bf < n_occ || orbitals.auxiliary_orbital_matrix.rows() != n_bf ||
        orbitals.auxiliary_orbital_matrix.cols() != n_bf ||
        accepted.active_one_electron_gradient.size() !=
            static_cast<std::size_t>(n_active) * n_active) {
        throw std::invalid_argument(
            "analytic orbital diagonal requires a complete accepted AO frame");
    }
    const Eigen::Map<const Eigen::MatrixXd> overlap(
        orbital_input.ao_overlap_matrix.data(), n_bf, n_bf);
    const Eigen::Map<const Eigen::MatrixXd> active_overlap(
        orbitals.active_orbital_overlap_matrix.data(), n_active, n_active);
    const auto& active_two_electron = prepared.active_space_two_electron_result;
    const bool two_electron_pretransformed = active_two_electron.orbital_transform.size() != 0;
    Eigen::MatrixXd active_transform;
    if (two_electron_pretransformed) {
        active_transform = active_two_electron.orbital_transform;
    } else {
        Eigen::LLT<Eigen::MatrixXd> active_llt(active_overlap);
        if (active_llt.info() != Eigen::Success) {
            throw std::runtime_error("active overlap factorization failed");
        }
        active_transform = active_llt.matrixU();
    }
    const Eigen::MatrixXd inverse_active_transform =
        active_transform.template triangularView<Eigen::Upper>().solve(
            Eigen::MatrixXd::Identity(n_active, n_active));

    Eigen::MatrixXd mo = orbitals.auxiliary_orbital_matrix;
    mo.leftCols(n_core) = orbitals.physical_orbital_frame.inactive_orthonormal_orbital_matrix;
    if (two_electron_pretransformed)
        mo.middleCols(n_core, n_active) = active_two_electron.dense_active_coefficients;
    else
        mo.middleCols(n_core, n_active) =
            orbitals.auxiliary_orbital_matrix.middleCols(n_core, n_active) *
            inverse_active_transform;
    if ((mo.transpose() * overlap * mo - Eigen::MatrixXd::Identity(n_bf, n_bf))
            .cwiseAbs()
            .maxCoeff() > 2.0e-8) {
        throw std::runtime_error("analytic orbital frame is not AO-metric orthonormal");
    }

    const Eigen::Map<const Eigen::MatrixXd> original_dm1(
        accepted.active_one_electron_gradient.data(), n_active, n_active);
    const Eigen::MatrixXd dm1 = active_transform * original_dm1 * active_transform.transpose();
    std::vector<double> owned_packed_dm2;
    const std::vector<double>* packed_dm2 = &accepted.packed_active_two_electron_gradient;
    if (!two_electron_pretransformed) {
        owned_packed_dm2 = pack_symmetric_dm2(
            transform_dm2(unpack_symmetric_active_two_rdm(
                              accepted.packed_active_two_electron_gradient, n_active),
                          active_transform),
            n_active);
        packed_dm2 = &owned_packed_dm2;
    }

    AnalyticDiagonalIntermediates x;
    x.n_core = n_core;
    x.n_active = n_active;
    x.h_core = mo.transpose() * input.ao_integral_input.ao_core_hamiltonian_matrix * mo;
    x.v_core =
        mo.transpose() * prepared.ao_effective_one_electron_result.ao_coulomb_exchange_matrix * mo;

    xmvb::integrals::ExactCtxPairMatrix owned_active_pair_products;
    const xmvb::integrals::ExactCtxPairMatrix* active_pair_products = &owned_active_pair_products;
    if (active_two_electron.representation ==
        xmvb::integrals::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity) {
        if (active_two_electron.ri_active_pair_factors.size() == 0 || !input.ri_factorization) {
            throw std::invalid_argument(
                "RI analytic Hessian diagonal requires accepted active-pair factors");
        }
        Eigen::MatrixXd transformed_active_factors;
        const Eigen::MatrixXd* orthogonal_active_factors =
            &active_two_electron.ri_active_pair_factors;
        if (!two_electron_pretransformed) {
            transformed_active_factors = xmvb::integrals::transform_ri_pair_factors(
                active_two_electron.ri_active_pair_factors, inverse_active_transform);
            orthogonal_active_factors = &transformed_active_factors;
        }
        owned_active_pair_products.noalias() =
            input.ri_factorization->metric_whitened_ao_pair_factors.transpose() *
            *orthogonal_active_factors;
    } else if (active_two_electron.dense_ao_pair_products.size() != 0) {
        if (two_electron_pretransformed) {
            active_pair_products = &active_two_electron.dense_ao_pair_products;
        } else {
            owned_active_pair_products.noalias() = active_two_electron.dense_ao_pair_products *
                                                   build_pair_transform(inverse_active_transform);
        }
    } else {
        // The memory-bounded exact forward path deliberately does not retain
        // G*Q_active. Rebuild only that absent accepted tensor from the AO graph.
        xmvb::integrals::ExactCtxPairMatrix active_pair_map;
        xmvb::integrals::build_packed_orbital_pair_map(mo.middleCols(n_core, n_active),
                                                       &active_pair_map);
        owned_active_pair_products = apply_ao_pair_operator(input, active_pair_map);
    }
    const int n_active_pairs = n_active * (n_active + 1) / 2;
    x.j_active.resize(static_cast<std::size_t>(n_bf) * n_active * n_active);
    // Only the diagonal MO pair (pp|ab) enters the analytic diagonal.
    // Accumulate directly: each worker owns O(n_active^2), not an AO-pair map.
#pragma omp parallel for schedule(static)
    for (int p = 0; p < n_bf; ++p) {
        Eigen::RowVectorXd contraction = Eigen::RowVectorXd::Zero(n_active_pairs);
        Eigen::Index pair = 0;
        for (int mu = 0; mu < n_bf; ++mu)
            for (int nu = 0; nu <= mu; ++nu, ++pair) {
                const double coefficient = mo(mu, p) * mo(nu, p) * (mu == nu ? 1.0 : 2.0);
                contraction.noalias() += coefficient * active_pair_products->row(pair);
            }
        for (int a = 0; a < n_active; ++a)
            for (int b = 0; b < n_active; ++b)
                x.j_active[mcscf::orbital_diagonal_slice_index(p, a, b, n_active)] =
                    contraction[xmvb::core::PairIndex::packed_pair_index(a, b)];
    }
    x.active_eri.resize(static_cast<std::size_t>(n_active) * n_active * n_active * n_active);
    const auto& packed_active_eri = active_two_electron.packed_active_two_electron_integrals;
    if (two_electron_pretransformed && !packed_active_eri.empty()) {
        const std::size_t expected_size =
            static_cast<std::size_t>(n_active_pairs) * (n_active_pairs + 1) / 2;
        if (packed_active_eri.size() != expected_size) {
            throw std::invalid_argument("framed active two-electron tensor size mismatch");
        }
        for (int a = 0; a < n_active; ++a)
            for (int b = 0; b < n_active; ++b)
                for (int c = 0; c < n_active; ++c)
                    for (int d = 0; d < n_active; ++d)
                        x.active_eri[mcscf::orbital_diagonal_tensor4_index(a, b, c, d, n_active)] =
                            packed_active_eri[xmvb::core::PairIndex::packed_pair_of_pairs_index(
                                xmvb::core::PairIndex::packed_pair_index(a, b),
                                xmvb::core::PairIndex::packed_pair_index(c, d))];
    } else {
        xmvb::integrals::ExactCtxPairMatrix active_pair_map;
        xmvb::integrals::build_packed_orbital_pair_map(mo.middleCols(n_core, n_active),
                                                       &active_pair_map);
        const Eigen::MatrixXd active_pairs = active_pair_map.transpose() * *active_pair_products;
        for (int a = 0; a < n_active; ++a)
            for (int b = 0; b < n_active; ++b)
                for (int c = 0; c < n_active; ++c)
                    for (int d = 0; d < n_active; ++d)
                        x.active_eri[mcscf::orbital_diagonal_tensor4_index(a, b, c, d, n_active)] =
                            active_pairs(xmvb::core::PairIndex::packed_pair_index(a, b),
                                         xmvb::core::PairIndex::packed_pair_index(c, d));
    }
    const auto dm2 = unpack_symmetric_active_two_rdm(*packed_dm2, n_active);
    const auto active_end =
        trace ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto densities = contract_diagonal_densities(input, mo, n_core, dm1, dm2, x.j_active);
    const auto& density_images = densities.active;
    const auto density_end =
        trace ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    DiagonalContractions contractions;
    contractions.potential = x.v_core.diagonal() + 0.5 * density_images.col(0);
    contractions.gradient = diagonal_gradient(x, dm1, dm2, contractions.potential);
    contractions.one_rdm.resize(n_occ, n_active);
    contractions.two_rdm.resize(n_bf, n_active);
    for (int p = 0; p < n_bf; ++p)
        for (int a = 0; a < n_active; ++a) {
            double coulomb = 0.0;
            for (int w = 0; w < n_active; ++w)
                for (int z = 0; z < n_active; ++z) {
                    const double weight =
                        dm2[mcscf::orbital_diagonal_tensor4_index(w, z, a, a, n_active)] +
                        2.0 * (dm2[mcscf::orbital_diagonal_tensor4_index(a, w, z, a, n_active)] +
                               dm2[mcscf::orbital_diagonal_tensor4_index(w, a, z, a, n_active)]);
                    coulomb +=
                        x.j_active[mcscf::orbital_diagonal_slice_index(p, w, z, n_active)] * weight;
                }
            contractions.two_rdm(p, a) = coulomb - density_images(p, 1 + n_active + a);
            if (p < n_occ) {
                double one_coulomb = 0.0;
                for (int v = 0; v < n_active; ++v)
                    one_coulomb +=
                        x.j_active[mcscf::orbital_diagonal_slice_index(p, a, v, n_active)] *
                        dm1(a, v);
                contractions.one_rdm(p, a) = 10.0 * one_coulomb - 6.0 * density_images(p, 1 + a);
            }
        }
    x.j_pc = Eigen::MatrixXd::Zero(n_bf, n_core);
    x.k_pc = Eigen::MatrixXd::Zero(n_bf, n_core);
    const auto core_start =
        trace ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const Eigen::Index n_ao_pairs = static_cast<Eigen::Index>(n_bf) * (n_bf + 1) / 2;
    const std::size_t pair_bytes = sizeof(double) * static_cast<std::size_t>(n_ao_pairs);
    int workers = 0;
    auto pair_end = core_start;
    // The pair map is dead after the AO action. Do not retain it during the
    // projection/exchange sweeps alongside its equally large integral image.
    if (densities.core_coulomb.rows() == n_bf) {
        x.j_pc = densities.core_coulomb;
    } else {
        const xmvb::integrals::ExactCtxPairMatrix core_diagonal_products = [&] {
            xmvb::integrals::ExactCtxPairMatrix map(n_ao_pairs, n_core);
            Eigen::Index pair = 0;
            for (int mu = 0; mu < n_bf; ++mu)
                for (int nu = 0; nu <= mu; ++nu, ++pair)
                    for (int c = 0; c < n_core; ++c)
                        map(pair, c) = mo(mu, c) * mo(nu, c) * (mu == nu ? 1.0 : 2.0);
            return apply_ao_pair_operator(input, map);
        }();
        pair_end =
            trace ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        // Independent output rows share the pair products. Workers own only one
        // packed AO-pair vector; their aggregate is bounded by the existing 64 MiB
        // diagonal worker budget, rather than copying the AO-by-core matrix.
        constexpr std::size_t worker_budget = 64 * 1024 * 1024;
        workers = std::min(effective_openmp_thread_count(),
                           static_cast<int>(std::max<std::size_t>(1, worker_budget / pair_bytes)));
#pragma omp parallel for schedule(static) num_threads(workers)
        for (int p = 0; p < (n_core ? n_bf : 0); ++p) {
            Eigen::VectorXd diagonal_pair(n_ao_pairs);
            Eigen::Index ao_pair = 0;
            for (int mu = 0; mu < n_bf; ++mu)
                for (int nu = 0; nu <= mu; ++nu, ++ao_pair)
                    diagonal_pair[ao_pair] = mo(mu, p) * mo(nu, p) * (mu == nu ? 1.0 : 2.0);
            x.j_pc.row(p) = diagonal_pair.transpose() * core_diagonal_products;
        }
    }
    const auto projection_end =
        trace ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    for (int c = 0; c < n_core; ++c)
        x.k_pc.col(c) = 2.0 * x.j_pc.col(c) - densities.core.col(c);
    const auto exchange_end =
        trace ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    AnalyticOrbitalDiagonal result;
    result.mo_coefficients = std::move(mo);
    result.rotation_diagonal = assemble_rotation_diagonal(x, dm1, contractions);
    result.inactive_inverse_transform =
        orbitals.physical_orbital_frame.localized_representative_selector
            .inactive_inverse_transpose_right_transform.transpose();
    result.active_inverse_transform = inverse_active_transform;
    if (trace)
        mcscf::diagnostics::ciah_trace(
            "ciah_diagonal_cost",
            "seconds=%.17g",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    if (trace)
        mcscf::diagnostics::ciah_trace(
            "ciah_diagonal_sections",
            "frame_active_seconds=%.17g,active_density_seconds=%.17g,density_contraction_seconds=%."
            "17g,core_pair_seconds=%.17g,core_projection_seconds=%.17g,core_exchange_seconds=%.17g,"
            "projection_workers=%d,projection_worker_bytes=%zu",
            std::chrono::duration<double>(active_end - start).count(),
            std::chrono::duration<double>(density_end - active_end).count(),
            std::chrono::duration<double>(core_start - density_end).count(),
            std::chrono::duration<double>(pair_end - core_start).count(),
            std::chrono::duration<double>(projection_end - pair_end).count(),
            std::chrono::duration<double>(exchange_end - projection_end).count(),
            workers,
            pair_bytes * workers);
    return result;
}

} // namespace xmvb::vb
