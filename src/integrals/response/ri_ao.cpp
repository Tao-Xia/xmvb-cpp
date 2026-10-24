#include "integrals/response/ri_ao.hpp"
#include "integrals/response/ri_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>
#include <cblas.h>
#include "parallel/openmp.hpp"
#include "core/timing.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xmvb::integrals {
namespace {

/** @brief Equal-work rectangular GEMM grid with minimum tile perimeter. */
std::pair<int, int> projection_grid(int rows, int cols, int workers) {
    std::pair<int, int> result{workers, 1};
    double perimeter = static_cast<double>(rows) / workers + cols;
    for (int columns = 2; columns <= std::min(cols, workers); ++columns) {
        if (workers % columns != 0)
            continue;
        const int row_blocks = workers / columns;
        if (row_blocks > rows)
            continue;
        const double candidate =
            static_cast<double>(rows) / row_blocks + static_cast<double>(cols) / columns;
        if (candidate < perimeter) {
            perimeter = candidate;
            result = {row_blocks, columns};
        }
    }
    return result;
}

/** @brief Add scale*(U^T V+V^T U), with one writer per lower AO rectangle. */
void add_symmetric_product(const Eigen::MatrixXd& u,
                           const Eigen::MatrixXd& v,
                           double scale,
                           Eigen::MatrixXd& output) {
    profiling::ScopedPhase phase(profiling::Phase::RiAoExchange);
    const int n = output.rows(), leading = u.rows();
    const int workers = std::min(effective_openmp_thread_count(), n);
    const int panels =
        std::min(n, static_cast<int>(std::ceil((std::sqrt(1.0 + 8.0 * workers) - 1) / 2)));
    std::vector<std::pair<int, int>> blocks;
    for (int row = 0; row < panels; ++row)
        for (int col = 0; col <= row; ++col)
            blocks.emplace_back(row, col);
#pragma omp parallel for schedule(dynamic, 1) num_threads(workers)
    for (std::size_t block = 0; block < blocks.size(); ++block) {
        const auto [row, col] = blocks[block];
        const int rb = row * n / panels, re = (row + 1) * n / panels;
        const int cb = col * n / panels, ce = (col + 1) * n / panels;
        double* target = output.data() + static_cast<std::size_t>(cb) * n + rb;
        const double* ur = u.data() + static_cast<std::size_t>(rb) * leading;
        const double* vr = v.data() + static_cast<std::size_t>(rb) * leading;
        const double* uc = u.data() + static_cast<std::size_t>(cb) * leading;
        const double* vc = v.data() + static_cast<std::size_t>(cb) * leading;
        if (row == col) {
            cblas_dsyr2k(CblasColMajor,
                         CblasLower,
                         CblasTrans,
                         re - rb,
                         leading,
                         scale,
                         ur,
                         leading,
                         vr,
                         leading,
                         1,
                         target,
                         n);
        } else {
            cblas_dgemm(CblasColMajor,
                        CblasTrans,
                        CblasNoTrans,
                        re - rb,
                        ce - cb,
                        leading,
                        scale,
                        ur,
                        leading,
                        vc,
                        leading,
                        1,
                        target,
                        n);
            cblas_dgemm(CblasColMajor,
                        CblasTrans,
                        CblasNoTrans,
                        re - rb,
                        ce - cb,
                        leading,
                        scale,
                        vr,
                        leading,
                        uc,
                        leading,
                        1,
                        target,
                        n);
        }
    }
}

/** @brief Add 2 L^T rho and mirror the completed lower triangle. */
void add_coulomb(const Eigen::MatrixXd& factors,
                 const Eigen::VectorXd& charge,
                 Eigen::MatrixXd& output) {
    profiling::ScopedPhase phase(profiling::Phase::RiAoCoulomb);
    const int n = output.rows(), pairs = factors.cols();
    const int workers = std::min(effective_openmp_thread_count(), pairs);
    Eigen::VectorXd packed(pairs);
#pragma omp parallel num_threads(workers)
    {
        int worker = 0;
        int team_size = 1;
#ifdef _OPENMP
        worker = omp_get_thread_num();
        team_size = omp_get_num_threads();
#endif
        const int begin = worker * pairs / team_size;
        const int end = (worker + 1) * pairs / team_size;
        packed.segment(begin, end - begin).noalias() =
            2 * factors.middleCols(begin, end - begin).transpose() * charge;
#pragma omp barrier
#pragma omp for schedule(static)
        for (int mu = 0; mu < n; ++mu)
            for (int nu = 0; nu <= mu; ++nu) {
                output(mu, nu) += packed[static_cast<std::size_t>(mu) * (mu + 1) / 2 + nu];
                output(nu, mu) = output(mu, nu);
            }
    }
}

} // namespace

Eigen::MatrixXd apply_ri_active_density_response(const RiActiveTwoElectronResponseCache& active,
                                                 const Eigen::Ref<const Eigen::MatrixXd>& weights) {
    const int n = active.n_basis_functions, na = active.n_active_orbitals;
    const int n_aux = active.n_auxiliary_functions;
    if (weights.rows() != na || weights.cols() != na || !active.accepted_active_pair_factors ||
        !active.metric_whitened_ao_pair_factors)
        throw std::invalid_argument("invalid RI active-density response dimensions");
    Eigen::VectorXd packed_weights(active.active_pair_first_indices.size());
    for (Eigen::Index p = 0; p < packed_weights.size(); ++p) {
        const int a = active.active_pair_first_indices[p], b = active.active_pair_second_indices[p];
        packed_weights[p] = (a == b ? 1.0 : 2.0) * weights(a, b);
    }
    const Eigen::VectorXd charge = *active.accepted_active_pair_factors * packed_weights;
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n, n);
    const int workers = std::min(effective_openmp_thread_count(), n);
    for (int first = 0; first < n_aux; first += active.auxiliary_tile_size) {
        const int count = std::min(active.auxiliary_tile_size, n_aux - first);
        Eigen::MatrixXd owned;
        const Eigen::MatrixXd* factors = nullptr;
        if (active.accepted_one_index_factors && count == n_aux) {
            factors = active.accepted_one_index_factors.get();
        } else if (active.accepted_one_index_factors) {
            owned.resize(count * na, n);
            for (int a = 0; a < na; ++a)
                owned.middleRows(a * count, count) =
                    active.accepted_one_index_factors->middleRows(a * n_aux + first, count);
            factors = &owned;
        } else {
            owned.resize(static_cast<Eigen::Index>(count) * na, n);
            transform_ri_one_index(*active.metric_whitened_ao_pair_factors,
                                   *active.accepted_active_coefficients,
                                   first,
                                   count,
                                   owned);
            factors = &owned;
        }
        Eigen::MatrixXd weighted(count * na, n);
#pragma omp parallel for schedule(static) num_threads(workers)
        for (int mu = 0; mu < n; ++mu) {
            const Eigen::Map<const Eigen::MatrixXd> source(factors->col(mu).data(), count, na);
            Eigen::Map<Eigen::MatrixXd> target(weighted.col(mu).data(), count, na);
            target.noalias() = source * weights;
        }
        add_symmetric_product(weighted, *factors, -0.5, result);
    }
    add_coulomb(*active.metric_whitened_ao_pair_factors, charge, result);
    return result;
}

std::vector<RiAoProjectedResponse>
apply_ri_projected_ao_response_batch(const RiActiveTwoElectronResponseCache& active,
                                     const Eigen::Ref<const Eigen::MatrixXd>& moving_inactive_duals,
                                     std::span<const RiAoResponseDirection> directions,
                                     const Eigen::Ref<const Eigen::MatrixXd>& full_inactive_duals,
                                     RiAoResponseWorkspace& workspace,
                                     std::size_t workspace_bytes,
                                     RiInactiveFactorView inactive,
                                     const RiActiveHvpInput* fixed_hvp) {
    const int n = active.n_basis_functions;
    const int na = active.n_active_orbitals;
    const int nm = moving_inactive_duals.cols();
    const int nf = full_inactive_duals.cols();
    const int n_aux = active.n_auxiliary_functions;
    const std::size_t n_directions = directions.size();
    const int rank = nm + na;
    if (n <= 0 || na <= 0 || n_aux <= 0 || nm > nf || moving_inactive_duals.rows() != n ||
        full_inactive_duals.rows() != n || active.accepted_active_coefficients == nullptr ||
        active.accepted_active_coefficients->rows() != n ||
        active.accepted_active_coefficients->cols() != na ||
        active.metric_whitened_ao_pair_factors == nullptr ||
        active.metric_whitened_ao_pair_factors->rows() != n_aux ||
        active.metric_whitened_ao_pair_factors->cols() !=
            static_cast<Eigen::Index>(n) * (n + 1) / 2 ||
        (active.accepted_one_index_factors &&
         (active.accepted_one_index_factors->rows() != static_cast<Eigen::Index>(n_aux) * na ||
          active.accepted_one_index_factors->cols() != n))) {
        throw std::invalid_argument("invalid projected RI AO response dimensions");
    }
    for (const auto& direction : directions) {
        if (direction.horizontal_direction.rows() != n ||
            direction.horizontal_direction.cols() != nm || direction.weighted_active.rows() != n ||
            direction.weighted_active.cols() != na ||
            (fixed_hvp != nullptr &&
             (direction.active_direction == nullptr || direction.active_direction->rows() != n ||
              direction.active_direction->cols() != na)))
            throw std::invalid_argument("invalid projected RI AO block direction");
    }
    if (inactive.factors != nullptr &&
        (inactive.factors->rows() != static_cast<Eigen::Index>(n_aux) * nf ||
         inactive.factors->cols() != n)) {
        throw std::invalid_argument("invalid full inactive RI one-index factors");
    }
    if ((!inactive.orbitals.empty() && inactive.orbitals.size() != static_cast<std::size_t>(nm)) ||
        (inactive.factors != nullptr && inactive.orbitals.size() != static_cast<std::size_t>(nm))) {
        throw std::invalid_argument("invalid moving inactive RI factor mapping");
    }
    for (int orbital : inactive.orbitals) {
        if (orbital < 0 || orbital >= nf) {
            throw std::invalid_argument("moving inactive RI factor index is out of range");
        }
    }

    const int active_pairs = na * (na + 1) / 2;
    if (fixed_hvp != nullptr &&
        (fixed_hvp->weight.rows() != na || fixed_hvp->weight.cols() != na ||
         active.accepted_active_pair_factors == nullptr ||
         active.accepted_active_pair_factors->rows() != n_aux ||
         active.accepted_active_pair_factors->cols() != active_pairs ||
         active.active_pair_first_indices.size() != static_cast<std::size_t>(active_pairs) ||
         active.active_pair_second_indices.size() != static_cast<std::size_t>(active_pairs)))
        throw std::invalid_argument("invalid fused RI fixed-HVP dimensions");

    if (n_directions == 0)
        return {};
    // D=0 and no inactive cotangent exists in this case.
    if (nf == 0) {
        std::vector<RiAoProjectedResponse> empty;
        empty.reserve(n_directions);
        for (std::size_t i = 0; i < n_directions; ++i)
            empty.push_back({Eigen::MatrixXd(n, 0), Eigen::MatrixXd::Zero(n, na)});
        return empty;
    }
    const int outputs = nf + na;
    const int workers = std::min(effective_openmp_thread_count(), n);
    const Eigen::Index pairs = static_cast<Eigen::Index>(n) * (n + 1) / 2;
    const auto [inactive_rows, inactive_columns] = projection_grid(n, nf, workers);
    const auto [active_rows, active_columns] = projection_grid(n, na, workers);
    // Existing reusable slabs belong to this request, not to a second kernel.
    // Credit them before replanning; all other accepted data stay charged.
    const std::size_t allowance =
        std::min(workspace_bytes, workspace::available() + workspace.residency.bytes());
    const std::size_t limit = allowance / sizeof(double);
    const std::size_t fixed =
        static_cast<std::size_t>(n) * (rank + workers) +
        n_directions * static_cast<std::size_t>(n) * outputs + pairs + 2ULL * n_aux * n_directions +
        (fixed_hvp != nullptr ? n_directions * (static_cast<std::size_t>(n_aux) * active_pairs +
                                                static_cast<std::size_t>(n) * na) +
                                    static_cast<std::size_t>(active_pairs) * active_pairs
                              : 0);
    // The previous directional tile is dead before the next transform. Keep
    // only accepted + new directional slabs live; a missing accepted image
    // temporarily needs at most one of the inactive/active transforms.
    const int missing_columns =
        std::max(inactive.factors == nullptr ? nm : 0, active.accepted_one_index_factors ? 0 : na);
    const std::size_t per_aux =
        static_cast<std::size_t>(n) * (2ULL * rank + workers + missing_columns) +
        2ULL * rank * outputs +
        (fixed_hvp != nullptr
             ? 2ULL * active_pairs + 2ULL * na * na + static_cast<std::size_t>(workers) * na
             : 0);
    if (fixed + per_aux > limit)
        throw std::length_error("minimum projected RI AO tile exceeds workspace");
    const int tile = static_cast<int>(std::min<std::size_t>(n_aux, (limit - fixed) / per_aux));
    const auto release_oversized = [](Eigen::MatrixXd& slab, Eigen::Index elements) {
        if (slab.size() > elements)
            slab.resize(0, 0);
    };
    release_oversized(workspace.direction, static_cast<Eigen::Index>(n) * rank * tile);
    release_oversized(workspace.accepted, static_cast<Eigen::Index>(n) * rank * tile);
    release_oversized(workspace.direction_projection,
                      static_cast<Eigen::Index>(rank) * tile * outputs);
    release_oversized(workspace.accepted_projection,
                      static_cast<Eigen::Index>(rank) * tile * outputs);
    release_oversized(workspace.charges, 2 * static_cast<Eigen::Index>(n_aux) * n_directions);
    if (inactive.factors != nullptr || workspace.coulomb.size() > pairs)
        workspace.coulomb.resize(0);
    workspace.charges.setZero(n_aux, 2 * static_cast<Eigen::Index>(n_directions));
    if (inactive.factors == nullptr)
        workspace.coulomb.resize(pairs);
    // Shrink obsolete slabs before allocating this request's fixed outputs.
    // Otherwise a smaller request could transiently retain the larger old tile.
    std::vector<RiAoProjectedResponse> results;
    results.reserve(n_directions);
    for (std::size_t i = 0; i < n_directions; ++i) {
        results.push_back({Eigen::MatrixXd::Zero(n, nf), Eigen::MatrixXd::Zero(n, na)});
        if (fixed_hvp != nullptr) {
            results.back().factor_direction.resize(n_aux, active_pairs);
            results.back().fixed_two_electron.setZero(n, na);
        }
    }
    Eigen::MatrixXd pair_adjoint;
    if (fixed_hvp != nullptr) {
        pair_adjoint = detail::build_ri_active_pair_adjoint(fixed_hvp->adjoint, active);
    }
    Eigen::MatrixXd u(n, rank);
    const auto& c = *active.accepted_active_coefficients;
    for (int first = 0; first < n_aux; first += tile) {
        const int count = std::min(tile, n_aux - first);
        const int leading = count * rank, moving_rows = count * nm;
        {
            profiling::ScopedPhase phase(profiling::Phase::RiAoAccepted);
            workspace.accepted.resize(leading, n);
            if (nm != 0) {
                if (inactive.factors != nullptr) {
                    for (int moving = 0; moving < nm; ++moving)
                        workspace.accepted.middleRows(moving * count, count) =
                            inactive.factors->middleRows(inactive.orbitals[moving] * n_aux + first,
                                                         count);
                } else {
                    transform_ri_one_index(*active.metric_whitened_ao_pair_factors,
                                           moving_inactive_duals,
                                           first,
                                           count,
                                           workspace.accepted.topRows(moving_rows));
                }
            }
            if (active.accepted_one_index_factors) {
                for (int orbital = 0; orbital < na; ++orbital)
                    workspace.accepted.middleRows((nm + orbital) * count, count) =
                        active.accepted_one_index_factors->middleRows(orbital * n_aux + first,
                                                                      count);
            } else {
                transform_ri_one_index(*active.metric_whitened_ao_pair_factors,
                                       c,
                                       first,
                                       count,
                                       workspace.accepted.bottomRows(count * na));
            }
        }
        Eigen::MatrixXd accepted_factor_adjoint;
        if (fixed_hvp != nullptr)
            accepted_factor_adjoint =
                active.accepted_active_pair_factors->middleRows(first, count) * pair_adjoint;
        workspace.accepted_projection.resize(leading, outputs);
        // A direction keeps only one transformed AO slab and its projections live.
        for (std::size_t direction_index = 0; direction_index < n_directions; ++direction_index) {
            const auto& request = directions[direction_index];
            auto& result = results[direction_index];
            u.leftCols(nm) = request.horizontal_direction;
            if (fixed_hvp != nullptr)
                u.rightCols(na) = *request.active_direction;
            else
                u.rightCols(na) = request.weighted_active;
            {
                profiling::ScopedPhase phase(profiling::Phase::RiAoDirection);
                workspace.direction.resize(leading, n);
                transform_ri_one_index(
                    *active.metric_whitened_ao_pair_factors, u, first, count, workspace.direction);
            }
            if (fixed_hvp != nullptr) {
                const Eigen::Ref<const Eigen::MatrixXd> active_factors(
                    workspace.accepted.bottomRows(count * na));
                detail::fill_ri_factor_direction_tile(active,
                                                      active_factors,
                                                      *request.active_direction,
                                                      first,
                                                      count,
                                                      result.factor_direction);
                const Eigen::MatrixXd directional_factor_adjoint =
                    result.factor_direction.middleRows(first, count) * pair_adjoint;
                detail::accumulate_ri_fixed_factor_tile(
                    active,
                    active_factors,
                    count,
                    0,
                    workspace.direction.bottomRows(count * na),
                    count,
                    0,
                    accepted_factor_adjoint,
                    directional_factor_adjoint,
                    0,
                    count,
                    sizeof(double) * static_cast<std::size_t>(na) * na * count,
                    result.fixed_two_electron);
                // Each AO column stores a [auxiliary,active] slab. The fixed jet has
                // consumed L D, so this in-place small-matrix action now supplies L Wbar.
#pragma omp parallel for schedule(static) num_threads(workers)
                for (int mu = 0; mu < n; ++mu) {
                    Eigen::Map<Eigen::MatrixXd> direction_image(
                        workspace.direction.col(mu).data() + moving_rows, count, na);
                    const Eigen::MatrixXd weighted_image = direction_image * fixed_hvp->weight;
                    direction_image = weighted_image;
                }
            }
            profiling::ScopedPhase exchange_phase(profiling::Phase::RiAoExchange);
            workspace.direction_projection.resize(leading, outputs);
            // Form (L A, L C)^T A0 and (L X, L W)^T A0 in tall GEMMs.
            // Only moving rows need projection onto C; the unused active rows are
            // neither generated nor consumed. There is no auxiliary-by-auxiliary GEMM.
#pragma omp parallel num_threads(workers)
            {
                int worker = 0;
                int team_size = 1;
#ifdef _OPENMP
                worker = omp_get_thread_num();
                team_size = omp_get_num_threads();
#endif
                const int begin = worker * leading / team_size;
                const int end = (worker + 1) * leading / team_size;
                const int size = end - begin;
                if (size) {
                    if (direction_index == 0) {
                        // The first direction fills the accepted projection; later directions
                        // reuse it without launching another worker team for width one.
                        workspace.accepted_projection.block(begin, 0, size, nf).noalias() =
                            workspace.accepted.middleRows(begin, size) * full_inactive_duals;
                        const int moving_end = std::min(end, moving_rows);
                        if (moving_end > begin && inactive.orbitals.empty())
                            workspace.accepted_projection.block(begin, nf, moving_end - begin, na)
                                .noalias() =
                                -workspace.accepted.middleRows(begin, moving_end - begin) * c;
                        if (moving_end > begin)
                            workspace.accepted_projection.block(begin, 0, moving_end - begin, nf) *=
                                -4;
                        const int active_begin = std::max(begin, moving_rows);
                        if (end > active_begin)
                            workspace.accepted_projection.block(
                                active_begin, 0, end - active_begin, nf) *= -1;
                    }
                    workspace.direction_projection.block(begin, 0, size, nf).noalias() =
                        workspace.direction.middleRows(begin, size) * full_inactive_duals;
                    const int moving_end = std::min(end, moving_rows);
                    if (moving_end > begin) {
                        const int moving_size = moving_end - begin;
                        workspace.direction_projection.block(begin, nf, moving_size, na).noalias() =
                            -workspace.direction.middleRows(begin, moving_size) * c;
                    }
                    const int inactive_end = std::min(end, moving_rows);
                    if (inactive_end > begin)
                        workspace.direction_projection.block(begin, 0, inactive_end - begin, nf) *=
                            -4;
                    const int active_begin = std::max(begin, moving_rows);
                    if (end > active_begin)
                        workspace.direction_projection.block(
                            active_begin, 0, end - active_begin, nf) *= -1;
                }
#pragma omp barrier
                if (direction_index == 0 && !inactive.orbitals.empty()) {
                    // Symmetry moves the accepted active projection into each moving row.
#pragma omp for schedule(static)
                    for (int moving = 0; moving < nm; ++moving)
                        for (int auxiliary = 0; auxiliary < count; ++auxiliary)
                            for (int orbital = 0; orbital < na; ++orbital)
                                workspace.accepted_projection(moving * count + auxiliary,
                                                              nf + orbital) =
                                    workspace.accepted_projection((nm + orbital) * count +
                                                                      auxiliary,
                                                                  inactive.orbitals[moving]);
                }
                const int aux_begin = worker * count / team_size;
                const int aux_end = (worker + 1) * count / team_size;
                const int aux_size = aux_end - aux_begin;
                if (aux_size) {
                    auto rho_d = workspace.charges.col(2 * direction_index)
                                     .segment(first + aux_begin, aux_size);
                    auto rho_r = workspace.charges.col(2 * direction_index + 1)
                                     .segment(first + aux_begin, aux_size);
                    if (!inactive.orbitals.empty()) {
                        // The moving diagonal of -4 (L X)^T A0 is already present.
                        for (int orbital = 0; orbital < nm; ++orbital)
                            for (int auxiliary = aux_begin; auxiliary < aux_end; ++auxiliary)
                                rho_d[auxiliary - aux_begin] -=
                                    0.5 *
                                    workspace.direction_projection(orbital * count + auxiliary,
                                                                   inactive.orbitals[orbital]);
                    }
                    for (int mu = 0; mu < n; ++mu) {
                        if (inactive.orbitals.empty())
                            for (int orbital = 0; orbital < nm; ++orbital)
                                if (request.horizontal_direction(mu, orbital) != 0)
                                    rho_d.noalias() += 2 *
                                                       request.horizontal_direction(mu, orbital) *
                                                       workspace.accepted.col(mu).segment(
                                                           orbital * count + aux_begin, aux_size);
                        for (int orbital = 0; orbital < na; ++orbital)
                            if (request.weighted_active(mu, orbital) != 0)
                                rho_r.noalias() +=
                                    request.weighted_active(mu, orbital) *
                                    workspace.accepted.col(mu).segment(
                                        (nm + orbital) * count + aux_begin, aux_size);
                    }
                    rho_r += 2 * rho_d;
                }
#pragma omp barrier
                // A 2-D output grid keeps GEMMs rectangular instead of reducing their
                // AO dimension to only a few rows when many workers are requested.
#pragma omp for schedule(static)
                for (int block = 0; block < workers; ++block) {
                    const int r = block / inactive_columns, c = block % inactive_columns;
                    const int row = r * n / inactive_rows, rows = (r + 1) * n / inactive_rows - row;
                    const int col = c * nf / inactive_columns,
                              cols = (c + 1) * nf / inactive_columns - col;
                    result.inactive.block(row, col, rows, cols).noalias() +=
                        workspace.direction.middleCols(row, rows).transpose() *
                        workspace.accepted_projection.middleCols(col, cols);
                    result.inactive.block(row, col, rows, cols).noalias() +=
                        workspace.accepted.middleCols(row, rows).transpose() *
                        workspace.direction_projection.middleCols(col, cols);
                }
                if (nm != 0) {
#pragma omp for schedule(static)
                    for (int block = 0; block < workers; ++block) {
                        const int r = block / active_columns, c = block % active_columns;
                        const int row = r * n / active_rows, rows = (r + 1) * n / active_rows - row;
                        const int col = c * na / active_columns,
                                  cols = (c + 1) * na / active_columns - col;
                        result.active.block(row, col, rows, cols).noalias() +=
                            workspace.direction.topRows(moving_rows)
                                .middleCols(row, rows)
                                .transpose() *
                            workspace.accepted_projection.block(0, nf + col, moving_rows, cols);
                        result.active.block(row, col, rows, cols).noalias() +=
                            workspace.accepted.topRows(moving_rows)
                                .middleCols(row, rows)
                                .transpose() *
                            workspace.direction_projection.block(0, nf + col, moving_rows, cols);
                    }
                }
#pragma omp for schedule(static)
                for (int mu = 0; mu < n; ++mu) {
                    if (nm != 0) {
                        for (int orbital = 0; orbital < na; ++orbital)
                            result.active(mu, orbital) +=
                                2 * workspace.accepted.col(mu)
                                        .segment((nm + orbital) * count, count)
                                        .dot(workspace.charges.col(2 * direction_index)
                                                 .segment(first, count));
                    }
                }
            }
        }
    }
    // Coulomb is separable. Resident L_A A0 factors allow direct auxiliary
    // contraction into each output column; otherwise use a packed AO response.
    for (std::size_t direction_index = 0; direction_index < n_directions; ++direction_index) {
        auto& result = results[direction_index];
        profiling::ScopedPhase phase(profiling::Phase::RiAoCoulomb);
        if (inactive.factors != nullptr) {
#pragma omp parallel for schedule(static) num_threads(std::min(workers, nf))
            for (int orbital = 0; orbital < nf; ++orbital) {
                result.inactive.col(orbital).noalias() +=
                    4.0 * inactive.factors->middleRows(orbital * n_aux, n_aux).transpose() *
                    workspace.charges.col(2 * direction_index + 1);
            }
        } else {
#pragma omp parallel num_threads(workers)
            {
                int worker = 0;
                int team_size = 1;
#ifdef _OPENMP
                worker = omp_get_thread_num();
                team_size = omp_get_num_threads();
#endif
                const Eigen::Index begin = worker * pairs / team_size;
                const Eigen::Index end = (worker + 1) * pairs / team_size;
                workspace.coulomb.segment(begin, end - begin).noalias() =
                    4 *
                    active.metric_whitened_ao_pair_factors->middleCols(begin, end - begin)
                        .transpose() *
                    workspace.charges.col(2 * direction_index + 1);
#pragma omp barrier
                Eigen::VectorXd row(n);
#pragma omp for schedule(static)
                for (int mu = 0; mu < n; ++mu) {
                    for (int nu = 0; nu < n; ++nu) {
                        const int hi = std::max(mu, nu), lo = std::min(mu, nu);
                        row[nu] =
                            workspace.coulomb[static_cast<Eigen::Index>(hi) * (hi + 1) / 2 + lo];
                    }
                    result.inactive.row(mu).noalias() += row.transpose() * full_inactive_duals;
                }
            }
        }
    }
    const auto retained =
        sizeof(double) *
        static_cast<std::size_t>(workspace.direction.size() + workspace.accepted.size() +
                                 workspace.direction_projection.size() +
                                 workspace.accepted_projection.size() + workspace.charges.size() +
                                 workspace.coulomb.size());
    if (!workspace.residency.try_resize(retained))
        throw std::length_error("retained RI AO workspace exceeds shared allowance");
    return results;
}

} // namespace xmvb::integrals
