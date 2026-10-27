#include "core/index.hpp"
#include "cuda/vb/chart.hpp"
#include "cuda/vb/orbitals.hpp"
#include "cuda/linalg/blas.hpp"
#include "cuda/linalg/block_map.hpp"
#include "cuda/linalg/svd.hpp"
#include "vbscf/orbitals/charts/partition.hpp"
#include "vbscf/core/contracts/orbital_type.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <cfloat>
#include <cstdint>
#include <memory>
#include <map>
#include <optional>
#include <tuple>
#include <numeric>
#include <utility>
#include <cusolverDn.h>
#include <nvtx3/nvtx3.hpp>

namespace xmvb::cuda {
namespace {

constexpr int kThreads = 128;
int grid(int count) {
    return (count + kThreads - 1) / kThreads;
}

int checked_basis(int basis, int orbitals, int reduced, std::size_t stored_count) {
    if (basis <= 0 || orbitals <= 0 || reduced < 0 ||
        stored_count != static_cast<std::size_t>(orbitals) ||
        static_cast<long long>(basis) * orbitals > std::numeric_limits<int>::max())
        throw std::invalid_argument("CUDA orbital chart dimensions are invalid");
    return basis;
}

void check_view(linalg::ConstMatrixView view, int rows, int cols, const char* label) {
    if (rows == 0 && cols == 1 && view.rows == 0 && view.cols == 1)
        return;
    if (!view.data || view.rows != rows || view.cols != cols || view.ld < rows)
        throw std::invalid_argument(std::string(label) + " matrix shape mismatch");
}

__global__ void expand_rows(const int* slots,
                            const int* row_block,
                            const int* row_local,
                            const int* first_basis,
                            const int* row_count,
                            const int* reduced_offset,
                            const int* reduced_size,
                            const double* basis_values,
                            const double* reduced,
                            double* raw,
                            int mapped_rows,
                            int raw_basis,
                            int raw_ld) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= mapped_rows)
        return;
    const int block = row_block[row];
    const int local = row_local[row];
    const std::int64_t count = row_count[block];
    const int dimension = reduced_size[block];
    const std::int64_t offset = first_basis[block];
    const int coordinate = reduced_offset[block];
    double value = 0.0;
    for (int j = 0; j < dimension; ++j)
        value += basis_values[offset + j * count + local] * reduced[coordinate + j];
    const int slot = slots[row];
    raw[(slot / raw_basis) * raw_ld + slot % raw_basis] = value;
}

__global__ void project_coordinates(const int* slots,
                                    const int* first_row,
                                    const int* row_count,
                                    const int* first_basis,
                                    const int* coordinate_block,
                                    const int* coordinate_local,
                                    const double* basis_values,
                                    const double* raw,
                                    double* reduced,
                                    int coordinates,
                                    int raw_basis,
                                    int raw_ld) {
    const std::int64_t coordinate =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (coordinate >= coordinates)
        return;
    const int block = coordinate_block[coordinate];
    const int local = coordinate_local[coordinate];
    const int rows = row_count[block];
    const int start = first_row[block];
    const int basis = first_basis[block] + local * rows;
    double value = 0.0;
    for (int i = 0; i < rows; ++i)
        value += basis_values[basis + i] *
                 raw[(slots[start + i] / raw_basis) * raw_ld + slots[start + i] % raw_basis];
    reduced[coordinate] = value;
}

__global__ void add_step(linalg::ConstMatrixView accepted,
                         const double* direction,
                         double scale,
                         linalg::MatrixView trial,
                         const int* stored_counts,
                         int* info,
                         int basis,
                         int orbitals) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= basis * orbitals)
        return;
    const std::int64_t row = index % basis, col = index / basis;
    const double value = row >= stored_counts[col]
                             ? 0.0
                             : accepted.data[col * accepted.ld + row] + scale * direction[index];
    trial.data[col * trial.ld + row] = value;
    if (!isfinite(value))
        atomicCAS(info, 0, index + 1);
}

} // namespace

namespace {
using Const = linalg::ConstMatrixView;
using Mut = linalg::MatrixView;
struct ChartSvd {
    Matrix<double> right;
    int rank;
};
void chart_zero(Context&, Mut);
void chart_copy(Context&, Mut, Const);
Matrix<double>
chart_product(Context&, Const, Const, Operation = Operation::none, Operation = Operation::none);
Matrix<double> chart_eye(Context&, int);
ChartSvd chart_svd(Context&, Const);
Buffer<int> chart_indices(Context&, const std::vector<int>&);
void chart_check_view(Const, int, int, const char*);
__global__ void chart_scatter_inactive(Const, const int*, const int*, double*, int, int);
__global__ void chart_gather_rows(Const, const int*, double*, int, int);
__global__ void chart_gather_overlap(Const, const int*, double*, int);
__global__ void chart_gather_coeff(Const, int, double*, int);
__global__ void chart_compose(double*, Const, Const, int, int, int);
__global__ void chart_sign_columns(double*, int, int);
__global__ void chart_ray_reflector(const double*, double*, int, double);
__global__ void chart_stored_quotient(Const, double*, int, int, int);
__global__ void chart_scale_vector(double*, int, double);
__global__ void chart_normalize_columns(Mut, Const);
__global__ void chart_make_basis(Const, Const, Const, double*, int, int, double, bool);
__global__ void chart_rank_one(double*, const double*, int);
__global__ void chart_subtract(double*, const double*, int);
__global__ void chart_sphere_flag(Const, int, int, int, int*);
__global__ void chart_add_retraction(Const, Const, Mut, const int*, int*, int, int, double);
__global__ void chart_scatter_basis(Const, const int*, double*, int, int, int);
__global__ void chart_take_row(Const, int, double*, int);
__global__ void
chart_scatter_raw_dense(Const, const int*, const int*, double*, double*, int, int, int);
__global__ void chart_gather_dense_raw(Const, Const, const int*, const int*, Mut, int, int, int);
__global__ void chart_ray_project(Const, Const, Const, const double*, Mut, int, int);
__global__ void chart_diagonal(Const, double*, int);
__global__ void chart_rotation_weights(Const, Const, double*, int, int, int, int);
__global__ void chart_rotation_curvature(const double*, const double*, double*, int, int);
__global__ void chart_rotation_diagonal(Const, const double*, double*, int, int);
__global__ void chart_ah_divide(double*, Const, int, int, int, double);
Matrix<double> chart_projector_apply(Context&, Const, Const, Const, Const);
} // namespace

struct AcceptedOrbitalChart::Impl {
    struct Meta {
        int orbital = 0, stored = 0, differentiable = 0, block = 0, block_orbital = 0;
        int quotient_group = 0;
        bool local_ray = false;
        std::vector<int> local_ao, stored_ao, forbidden_ao;
        Buffer<int> local_device, stored_device, forbidden_device;
    };
    struct Local {
        int orbital = 0, stored = 0, differentiable = 0, offset = 0, size = 0;
        double delta = 1.0;
        double raw_norm_squared = 1.0;
        bool sphere = false;
        Matrix<double> basis, metric, inverse, raw_inverse_gram, reference_root, projection;
        Matrix<double> rotation_components, source_weights;
        double source_weight_max = 0.0;
        Matrix<double> ah_signed_vectors, ah_signed_values, ah_frame_inverse;
    };

    Impl(Context& ctx,
         const vb::OrbitalPreparationInput& input,
         bool complete,
         const AoMetric& metric)
        : context(ctx), ao_metric(metric), basis(static_cast<int>(input.n_basis_functions)),
          orbitals(static_cast<int>(input.n_orbitals)),
          inactive(static_cast<int>((input.n_total_electrons - input.n_active_electrons) / 2)),
          active(static_cast<int>(input.n_active_orbitals)),
          active_subspace(complete && input.orbital_type == vb::kOrbitalTypeOeo),
          solver(ctx.solver()), full_support(ctx, static_cast<std::size_t>(basis) * orbitals),
          stored_counts(ctx, orbitals), bad_info(ctx, 1), tangent(ctx, basis, orbitals),
          ao_overlap(metric.overlap()) {
        if (basis <= 0 || orbitals <= 0 || inactive < 0 || inactive + active > orbitals ||
            input.n_total_electrons < input.n_active_electrons ||
            input.ao_overlap_matrix.rows() != basis || input.ao_overlap_matrix.cols() != basis ||
            input.orbital_basis_index_table.size() != static_cast<std::size_t>(basis) * orbitals)
            throw std::invalid_argument("accepted CUDA chart orbital dimensions");
        if (&metric.context() != &ctx || ao_overlap.rows != basis || ao_overlap.cols != basis)
            throw std::invalid_argument("accepted CUDA chart AO metric ownership/dimensions");
        std::vector<int> support(static_cast<std::size_t>(basis) * orbitals, -1);
        std::vector<int> counts(orbitals, 0);
        for (int orbital = 0; orbital < orbitals; ++orbital) {
            counts[orbital] = vb::stored_sparse_orbital_coefficient_count(input, orbital);
            if (counts[orbital] <= 0 || counts[orbital] > basis)
                throw std::invalid_argument("accepted CUDA chart stored support");
            for (int j = 0; j < counts[orbital]; ++j)
                support[orbital * basis + j] =
                    input.orbital_basis_index_table[orbital * basis + j] - 1;
        }
        context.copy_to_device(full_support.data(), support.data(), support.size() * sizeof(int));
        context.copy_to_device(stored_counts.data(), counts.data(), counts.size() * sizeof(int));
        const auto blocks = vb::detect_orbital_blocks(input);
        int occupied_block = 0;
        for (std::int64_t block = 0; block < core::checked_index<std::int64_t>(blocks.size());
             ++block) {
            int block_occupied = 0;
            for (const int orbital : blocks[block]) {
                if (orbital >= inactive + active)
                    continue;
                Meta m;
                m.orbital = orbital;
                m.stored = counts[orbital];
                m.differentiable =
                    vb::differentiable_sparse_orbital_parameter_count(input, orbital);
                m.block = occupied_block;
                m.block_orbital = block_occupied++;
                if (m.differentiable < 0 || m.differentiable > m.stored)
                    throw std::invalid_argument("accepted CUDA chart differentiable support");
                std::vector<unsigned char> seen(basis, 0);
                for (int j = 0; j < m.stored; ++j) {
                    const int ao = support[orbital * basis + j];
                    if (ao < 0 || ao >= basis || seen[ao]++)
                        throw std::invalid_argument("accepted CUDA chart sparse AO support");
                    m.stored_ao.push_back(ao);
                    if (j < m.differentiable)
                        m.local_ao.push_back(ao);
                }
                for (int ao = 0; ao < basis; ++ao)
                    if (std::find(m.local_ao.begin(), m.local_ao.end(), ao) == m.local_ao.end())
                        m.forbidden_ao.push_back(ao);
                m.local_device = chart_indices(context, m.local_ao);
                m.stored_device = chart_indices(context, m.stored_ao);
                m.forbidden_device = chart_indices(context, m.forbidden_ao);
                metas.push_back(std::move(m));
            }
            if (block_occupied)
                ++occupied_block;
        }
        // A quotient depends on the ordered allowed rows, forbidden rows and
        // gauge source, not on the orbital's norm/radial representative. Selected
        // active expansions use distinct rays, never an assumed CAS closure.
        std::map<std::tuple<int, std::vector<int>, std::vector<int>>, int> groups;
        for (auto& meta : metas) {
            meta.local_ray =
                meta.orbital >= inactive && !active_subspace && meta.differentiable == meta.stored;
            const std::int64_t source = meta.orbital < inactive || meta.local_ray ? -1
                                        : active_subspace                         ? -2
                                                                                  : meta.orbital;
            const auto [where, inserted] = groups.emplace(
                std::make_tuple(source, meta.local_ao, meta.forbidden_ao), quotient_groups);
            meta.quotient_group = where->second;
            if (inserted)
                ++quotient_groups;
        }
    }

    Context& context;
    const AoMetric& ao_metric;
    int basis, orbitals, inactive, active;
    bool active_subspace;
    cusolverDnHandle_t solver = nullptr;
    Buffer<int> full_support, stored_counts, bad_info;
    Matrix<double> tangent;
    Const ao_overlap;
    Matrix<double> frame_mo, frame_external, inactive_inverse_transform;
    Matrix<double> ah_rotation_diagonal, ah_diagonal, ah_core_source_inverse,
        ah_active_source_inverse;
    bool has_analytic_diagonal = false;
    Matrix<double> metric_core, metric_active, metric_core_inverse, metric_core_root,
        metric_active_inverse, metric_active_root, metric_projected_active, metric_K, metric_CGi,
        metric_SA, metric_ST, metric_ray_norms;
    bool full_rotation_frame = false;
    void build_rotation_frame(const OrbitalGeometry&);
    void build_physical_metric_factors(const OrbitalGeometry&);
    Matrix<double> build_quotient(const OrbitalGeometry&, const Meta&, Const);
    Matrix<double> restrict_ray(const OrbitalGeometry&, const Meta&, Const);
    std::vector<Meta> metas;
    std::vector<Local> locals;
    BlockMap basis_map, metric_map, inverse_map, signed_map;
    Matrix<double> signed_spectrum, signed_scratch;
    int reduced = 0;
    int quotient_groups = 0;
    std::uint64_t signature = 1469598103934665603ull;
};

AcceptedOrbitalChart::AcceptedOrbitalChart(Context& context,
                                           const vb::OrbitalPreparationInput& input,
                                           bool complete_active_space,
                                           const AoMetric& metric)
    : impl_(std::make_unique<Impl>(context, input, complete_active_space, metric)) {}
AcceptedOrbitalChart::~AcceptedOrbitalChart() = default;
int AcceptedOrbitalChart::reduced_size() const noexcept {
    return impl_->reduced;
}
int AcceptedOrbitalChart::raw_size() const noexcept {
    return impl_->basis * impl_->orbitals;
}
std::uint64_t AcceptedOrbitalChart::rank_signature() const noexcept {
    return impl_->signature;
}
linalg::ConstMatrixView AcceptedOrbitalChart::accepted_mo_frame() const noexcept {
    return impl_->frame_mo.view();
}

void AcceptedOrbitalChart::rebuild(const OrbitalGeometry& geometry) {
    const nvtx3::scoped_range range{"chart.prepare"};
    auto& p = *impl_;
    Context& ctx = p.context;
    if (geometry.basis_size() != p.basis || geometry.orbital_count() != p.orbitals ||
        geometry.inactive_count() != p.inactive || geometry.active_count() != p.active)
        throw std::invalid_argument("accepted CUDA chart geometry partition mismatch");
    const Const raw = geometry.raw(), overlap = geometry.ao_overlap();
    p.locals.clear();
    p.reduced = 0;
    p.signature = 1469598103934665603ull;
    p.has_analytic_diagonal = false;
    p.signed_map = BlockMap{};
    p.signed_spectrum = Matrix<double>{};
    p.signed_scratch = Matrix<double>{};

    Matrix<double> inactive_span(ctx, p.basis, p.inactive);
    if (p.inactive) {
        chart_zero(ctx, inactive_span.view());
        chart_scatter_inactive<<<grid(p.basis * p.inactive), kThreads, 0, ctx.stream()>>>(
            raw,
            p.full_support.data(),
            p.stored_counts.data(),
            inactive_span.data(),
            p.basis,
            p.inactive);
        check_cuda(cudaGetLastError(), "CUDA chart inactive scatter");
        auto metric_image = chart_product(ctx, overlap, inactive_span.view());
        auto metric =
            chart_product(ctx, inactive_span.view(), metric_image.view(), Operation::transpose);
        auto eig = symmetric_spectrum(ctx, p.solver, metric.view());
        const double scale = eig.host_values.back();
        if (!(scale > 0.0) || !(eig.host_values.front() > DBL_EPSILON * p.inactive * scale))
            throw std::runtime_error("inactive span is numerically rank deficient in AO metric");
        auto root = spectral_power(ctx, eig, -0.5);
        inactive_span = chart_product(ctx, inactive_span.view(), root.view());
    }

    std::vector<std::optional<Matrix<double>>> quotients(p.quotient_groups);
    std::vector<int> remaining(p.quotient_groups, 0);
    for (const auto& meta : p.metas)
        ++remaining[meta.quotient_group];
    for (const auto& meta : p.metas) {
        Impl::Local local;
        local.orbital = meta.orbital;
        local.stored = meta.stored;
        local.differentiable = meta.differentiable;
        local.offset = p.reduced;
        const int m = meta.differentiable;
        auto& shared = quotients[meta.quotient_group];
        if (!shared)
            shared = p.build_quotient(geometry, meta, inactive_span.view());
        Matrix<double> ray_quotient;
        if (meta.local_ray)
            ray_quotient = p.restrict_ray(geometry, meta, shared->view());
        const auto& quotient = meta.local_ray ? ray_quotient : *shared;
        local.size = core::checked_index<std::int64_t>(quotient.cols());
        for (int value : {meta.block + 1, meta.block_orbital + 1, local.size + 1}) {
            p.signature ^= static_cast<std::uint64_t>(value);
            p.signature *= 1099511628211ull;
        }
        if (!local.size) {
            p.locals.push_back(std::move(local));
            if (!--remaining[meta.quotient_group])
                shared.reset();
            continue;
        }

        const int stored = meta.stored, q = local.size;
        Matrix<double> stored_overlap(ctx, stored, stored), coeff(ctx, stored, 1);
        chart_gather_overlap<<<grid(stored * stored), kThreads, 0, ctx.stream()>>>(
            overlap, meta.stored_device.data(), stored_overlap.data(), stored);
        chart_gather_coeff<<<grid(stored), kThreads, 0, ctx.stream()>>>(
            raw, meta.orbital, coeff.data(), stored);
        check_cuda(cudaGetLastError(), "CUDA chart stored orbital gather");
        auto covariant = chart_product(ctx, stored_overlap.view(), coeff.view());
        double squared_norm = 0.0;
        check_cublas(
            cublasDdot_64(ctx.blas(), stored, coeff.data(), 1, covariant.data(), 1, &squared_norm),
            "cublasDdot chart orbital norm");
        if (!(squared_norm > 0.0) || !std::isfinite(squared_norm))
            throw std::runtime_error("raw quotient has nonpositive orbital norm");
        const double norm = std::sqrt(squared_norm);
        local.raw_norm_squared = squared_norm;
        Matrix<double> stored_u(ctx, stored, q);
        chart_stored_quotient<<<grid(stored * q), kThreads, 0, ctx.stream()>>>(
            quotient.view(), stored_u.data(), stored, m, q);
        check_cuda(cudaGetLastError(), "CUDA chart padded quotient");
        auto projection =
            chart_product(ctx, stored_u.view(), covariant.view(), Operation::transpose);
        chart_scale_vector<<<grid(q), kThreads, 0, ctx.stream()>>>(
            projection.data(), q, 1.0 / norm);
        auto su = chart_product(ctx, stored_overlap.view(), stored_u.view());
        auto base_metric = chart_product(ctx, stored_u.view(), su.view(), Operation::transpose);
        auto base_eig = symmetric_spectrum(ctx, p.solver, base_metric.view());
        if (!(base_eig.host_values.front() > 0.0))
            throw std::runtime_error("raw quotient AO metric is not positive definite");
        auto inverse_root = spectral_power(ctx, base_eig, -0.5);
        auto inverse = chart_product(ctx, inverse_root.view(), inverse_root.view());
        auto image = chart_product(ctx, inverse.view(), projection.view());
        double squared = 0.0;
        check_cublas(cublasDdot_64(ctx.blas(), q, projection.data(), 1, image.data(), 1, &squared),
                     "cublasDdot chart radial projection");
        local.delta = 1.0 - squared;
        if (!(local.delta > 0.0) || !std::isfinite(local.delta))
            throw std::runtime_error("raw normalized-orbital metric is rank deficient");
        local.metric = std::move(base_metric);
        chart_rank_one<<<grid(q * q), kThreads, 0, ctx.stream()>>>(
            local.metric.data(), projection.data(), q);
        check_cuda(cudaGetLastError(), "CUDA chart rank-one metric");
        const double rank_factor = 64.0 * DBL_EPSILON * q;
        // M=A-r*r^T and delta=1-r^T A^-1 r. Congruence gives
        // lambda_min(M)>=delta*lambda_min(A), lambda_max(M)<=lambda_max(A).
        // This certifies the unchanged spectral rank test without another
        // eigensystem in ordinary cases, as on the CPU quotient path.
        if (local.delta * base_eig.host_values.front() >
            rank_factor * base_eig.host_values.back()) {
            local.inverse = Matrix<double>(ctx, q, q);
            chart_copy(ctx, local.inverse.view(), inverse.view());
            const double alpha = 1.0 / local.delta;
            check_cublas(cublasDger_64(ctx.blas(),
                                       q,
                                       q,
                                       &alpha,
                                       image.data(),
                                       1,
                                       image.data(),
                                       1,
                                       local.inverse.data(),
                                       q),
                         "CUDA chart rank-one metric inverse");
        } else {
            // The bound may be inconclusive near the rank boundary. Preserve the
            // original exact spectral decision and its stable cuSOLVER inverse.
            const auto metric_eig = symmetric_spectrum(ctx, p.solver, local.metric.view());
            if (!(metric_eig.host_values.front() > rank_factor * metric_eig.host_values.back()))
                throw std::runtime_error("raw normalized-orbital metric is rank deficient");
            local.inverse = spectral_power(ctx, metric_eig, -1.0);
        }
        local.reference_root = std::move(inverse_root);
        local.projection = std::move(projection);
        Buffer<int> sphere_flag(ctx, 1);
        if (!p.active_subspace) {
            chart_sphere_flag<<<1, 1, 0, ctx.stream()>>>(
                raw, meta.orbital, m, stored, sphere_flag.data());
            check_cuda(cudaGetLastError(), "CUDA chart fixed-tail sphere flag");
            int host_flag = 0;
            ctx.copy_to_host(&host_flag, sphere_flag.data(), sizeof(int));
            local.sphere = host_flag != 0;
        }
        local.basis = Matrix<double>(ctx, m, q);
        chart_make_basis<<<grid(m * q), kThreads, 0, ctx.stream()>>>(quotient.view(),
                                                                     coeff.view(),
                                                                     local.projection.view(),
                                                                     local.basis.data(),
                                                                     m,
                                                                     q,
                                                                     norm,
                                                                     local.sphere);
        check_cuda(cudaGetLastError(), "CUDA chart accepted-point tangent basis");
        local.raw_inverse_gram = Matrix<double>(ctx, q, q);
        chart_copy(ctx,
                   local.raw_inverse_gram.view(),
                   local.sphere ? local.inverse.view() : inverse.view());
        chart_scale_vector<<<grid(q * q), kThreads, 0, ctx.stream()>>>(
            local.raw_inverse_gram.data(), q * q, 1.0 / squared_norm);
        check_cuda(cudaGetLastError(), "CUDA chart raw tangent inverse Gram");
        p.reduced += q;
        p.locals.push_back(std::move(local));
        if (!--remaining[meta.quotient_group])
            shared.reset();
    }
    p.build_rotation_frame(geometry);
    p.build_physical_metric_factors(geometry);
    std::vector<BlockMapEntry> basis_entries, metric_entries, inverse_entries;
    for (const auto& local : p.locals) {
        if (!local.size)
            continue;
        basis_entries.push_back({local.basis.view(), local.offset, local.orbital});
        metric_entries.push_back({local.metric.view(), local.offset, local.offset});
        inverse_entries.push_back({local.inverse.view(), local.offset, local.offset});
    }
    p.basis_map = BlockMap(ctx, basis_entries);
    p.metric_map = BlockMap(ctx, metric_entries);
    p.inverse_map = BlockMap(ctx, inverse_entries);
}

Matrix<double> AcceptedOrbitalChart::Impl::build_quotient(const OrbitalGeometry& geometry,
                                                          const Meta& meta,
                                                          Const inactive_span) {
    const nvtx3::scoped_range range{"chart.quotient"};
    auto& ctx = context;
    const auto overlap = geometry.ao_overlap();
    const int m = meta.differentiable;
    const int nsource =
        inactive +
        (meta.orbital >= inactive && !meta.local_ray ? (active_subspace ? active : 1) : 0);
    Matrix<double> source(ctx, basis, nsource);
    if (nsource) {
        if (meta.orbital >= inactive && !meta.local_ray) {
            const std::int64_t na = nsource - inactive;
            const std::int64_t first = active_subspace ? inactive : meta.orbital;
            Matrix<double> residual(ctx, basis, na);
            chart_copy(ctx, residual.view(), geometry.normalized().columns(first, na));
            if (inactive) {
                auto image = chart_product(ctx, overlap, residual.view());
                auto projection =
                    chart_product(ctx, inactive_span, image.view(), Operation::transpose);
                auto removed = chart_product(ctx, inactive_span, projection.view());
                chart_subtract<<<grid(basis * na), kThreads, 0, ctx.stream()>>>(
                    residual.data(), removed.data(), basis * na);
                check_cuda(cudaGetLastError(), "CUDA chart active residual");
            }
            auto image = chart_product(ctx, overlap, residual.view());
            auto metric = chart_product(ctx, residual.view(), image.view(), Operation::transpose);
            auto eig = symmetric_spectrum(ctx, solver, metric.view());
            const double scale = eig.host_values.back();
            if (!(scale > 0.0) ||
                !(eig.host_values.front() > DBL_EPSILON * std::max<std::int64_t>(1, na) * scale))
                throw std::runtime_error(
                    "active orbital subspace is rank deficient outside inactive span");
            auto root = spectral_power(ctx, eig, -0.5);
            auto active_source = chart_product(ctx, residual.view(), root.view());
            chart_compose<<<grid(basis * nsource), kThreads, 0, ctx.stream()>>>(
                source.data(), inactive_span, active_source.view(), inactive, basis, nsource);
            check_cuda(cudaGetLastError(), "CUDA chart gauge sources");
        } else {
            chart_copy(ctx, source.view(), inactive_span);
        }
    }
    if (!m)
        return Matrix<double>(ctx, m, 0);
    if (!nsource)
        return chart_eye(ctx, m);
    const std::int64_t forbidden = core::checked_index<std::int64_t>(meta.forbidden_ao.size());
    Matrix<double> admissible(ctx, nsource, 0);
    if (!forbidden) {
        admissible = chart_eye(ctx, nsource);
    } else {
        Matrix<double> constrained(ctx, forbidden, nsource);
        chart_gather_rows<<<grid(forbidden * nsource), kThreads, 0, ctx.stream()>>>(
            source.view(), meta.forbidden_device.data(), constrained.data(), forbidden, nsource);
        check_cuda(cudaGetLastError(), "CUDA chart forbidden-source gather");
        auto svd = chart_svd(ctx, constrained.view());
        const int nullity = nsource - svd.rank;
        admissible = Matrix<double>(ctx, nsource, nullity);
        if (nullity)
            chart_copy(ctx, admissible.view(), svd.right.view().columns(svd.rank, nullity));
    }
    if (!admissible.cols())
        return chart_eye(ctx, m);
    Matrix<double> gathered(ctx, m, nsource);
    chart_gather_rows<<<grid(m * nsource), kThreads, 0, ctx.stream()>>>(
        source.view(), meta.local_device.data(), gathered.data(), m, nsource);
    check_cuda(cudaGetLastError(), "CUDA chart local-source gather");
    auto generator = chart_product(ctx, gathered.view(), admissible.view());
    Matrix<double> local_overlap(ctx, m, m);
    chart_gather_overlap<<<grid(m * m), kThreads, 0, ctx.stream()>>>(
        overlap, meta.local_device.data(), local_overlap.data(), m);
    check_cuda(cudaGetLastError(), "CUDA chart local overlap gather");
    auto horizontal =
        chart_product(ctx, generator.view(), local_overlap.view(), Operation::transpose);
    auto svd = chart_svd(ctx, horizontal.view());
    const int q = m - svd.rank;
    Matrix<double> quotient(ctx, m, q);
    if (q) {
        chart_copy(ctx, quotient.view(), svd.right.view().columns(svd.rank, q));
        chart_sign_columns<<<grid(q), kThreads, 0, ctx.stream()>>>(quotient.data(), m, q);
        check_cuda(cudaGetLastError(), "CUDA chart quotient signs");
    }
    return quotient;
}

Matrix<double> AcceptedOrbitalChart::Impl::restrict_ray(const OrbitalGeometry& geometry,
                                                        const Meta& meta,
                                                        Const base) {
    const nvtx3::scoped_range range{"chart.local_ray"};
    auto& ctx = context;
    const int m = meta.differentiable, q0 = static_cast<int>(base.cols);
    if (!q0)
        return Matrix<double>(ctx, m, 0);
    // For a ray a entirely in allowed support L,
    // (I + span(a)) intersect L = (I intersect L) + span(a).
    // Therefore restrict the common inactive quotient by the single row a^T S Q.
    // Fixed nonzero tails are NOT in L and never enter this factorization.
    Matrix<double> orbital(ctx, m, 1), local_overlap(ctx, m, m);
    chart_gather_rows<<<grid(m), kThreads, 0, ctx.stream()>>>(
        geometry.normalized().columns(meta.orbital, 1),
        meta.local_device.data(),
        orbital.data(),
        m,
        1);
    chart_gather_overlap<<<grid(m * m), kThreads, 0, ctx.stream()>>>(
        geometry.ao_overlap(), meta.local_device.data(), local_overlap.data(), m);
    check_cuda(cudaGetLastError(), "CUDA chart local ray gather");
    auto image = chart_product(ctx, local_overlap.view(), orbital.view());
    auto row = chart_product(ctx, image.view(), base, Operation::transpose);
    double norm = 0.0;
    check_cublas(cublasDnrm2_64(ctx.blas(), q0, row.data(), 1, &norm), "CUDA chart ray norm");
    if (!std::isfinite(norm))
        throw std::runtime_error("nonfinite CUDA chart ray norm");
    const int rank = norm > DBL_EPSILON * q0 * norm ? 1 : 0; // Same single-row SVD rank test.
    const int q = q0 - rank;
    Matrix<double> quotient(ctx, m, q);
    if (!rank) {
        chart_copy(ctx, quotient.view(), base);
        return quotient;
    }
    if (!q)
        return quotient;
    double first = 0.0;
    ctx.copy_to_host(&first, row.data(), sizeof(double));
    const double denominator = first + std::copysign(norm, first);
    const double tau = 1.0 + std::abs(first) / norm;
    Matrix<double> v(ctx, q0, 1);
    chart_ray_reflector<<<grid(q0), kThreads, 0, ctx.stream()>>>(
        row.data(), v.data(), q0, denominator);
    check_cuda(cudaGetLastError(), "CUDA chart ray reflector");
    auto reflector_image = chart_product(ctx, base, v.view());
    chart_copy(ctx, quotient.view(), base.columns(1, q));
    const double alpha = -tau;
    check_cublas(cublasDger_64(ctx.blas(),
                               m,
                               q,
                               &alpha,
                               reflector_image.data(),
                               1,
                               v.data() + 1,
                               1,
                               quotient.data(),
                               m),
                 "CUDA chart rank-one ray restriction");
    chart_sign_columns<<<grid(q), kThreads, 0, ctx.stream()>>>(quotient.data(), m, q);
    check_cuda(cudaGetLastError(), "CUDA chart local ray quotient signs");
    return quotient;
}

void AcceptedOrbitalChart::Impl::build_rotation_frame(const OrbitalGeometry& geometry) {
    Context& ctx = context;
    const Const overlap = ao_overlap;
    const int occupied = inactive + active;
    Matrix<double> core(ctx, basis, inactive);
    if (inactive) {
        const Const normalized = geometry.normalized().columns(0, inactive);
        auto image = chart_product(ctx, overlap, normalized);
        auto gram = chart_product(ctx, normalized, image.view(), Operation::transpose);
        auto eig = symmetric_spectrum(ctx, solver, gram.view());
        if (!(eig.host_values.front() > 0.0))
            throw std::runtime_error("CUDA chart inactive rotation frame rank");
        inactive_inverse_transform = spectral_power(ctx, eig, -0.5);
        core = chart_product(ctx, normalized, inactive_inverse_transform.view());
    }
    Matrix<double> occupied_frame(ctx, basis, occupied);
    if (occupied) {
        chart_compose<<<grid(basis * occupied), kThreads, 0, ctx.stream()>>>(
            occupied_frame.data(),
            core.view(),
            geometry.orthogonal_active(),
            inactive,
            basis,
            occupied);
        check_cuda(cudaGetLastError(), "CUDA chart occupied rotation frame");
    }
    const int virtual_count = basis - occupied;
    Matrix<double> virtual_frame(ctx, basis, virtual_count);
    if (virtual_count) {
        // Match the CPU virtual gauge, not just its span. The analytic rotation
        // diagonal and component clipping are not invariant to arbitrary virtual
        // rotations returned by a null-space SVD.
        auto complement = chart_eye(ctx, basis);
        if (occupied) {
            auto image = chart_product(ctx, overlap, occupied_frame.view());
            gemm(ctx,
                 complement.view(),
                 occupied_frame.view(),
                 image.view(),
                 -1.0,
                 1.0,
                 Operation::none,
                 Operation::transpose);
        }
        auto image = chart_product(ctx, overlap, complement.view());
        auto projected = chart_product(ctx, complement.view(), image.view(), Operation::transpose);
        auto eig = symmetric_spectrum(ctx, solver, projected.view());
        // This is the existing CPU virtual-space rank criterion.
        if (!(eig.host_values[occupied] > 1.0e-10))
            throw std::runtime_error("CUDA virtual auxiliary metric rank deficient");
        virtual_frame = chart_product(
            ctx, complement.view(), eig.vectors.view().columns(occupied, virtual_count));
        auto s_virtual = chart_product(ctx, overlap, virtual_frame.view());
        auto norms =
            chart_product(ctx, virtual_frame.view(), s_virtual.view(), Operation::transpose);
        chart_normalize_columns<<<grid(basis * virtual_count), kThreads, 0, ctx.stream()>>>(
            virtual_frame.view(), norms.view());
        check_cuda(cudaGetLastError(), "CUDA virtual auxiliary normalization");
    }
    const int external_count = basis - inactive;
    frame_external = Matrix<double>(ctx, basis, external_count);
    if (external_count) {
        chart_compose<<<grid(basis * external_count), kThreads, 0, ctx.stream()>>>(
            frame_external.data(),
            geometry.orthogonal_active(),
            virtual_frame.view(),
            active,
            basis,
            external_count);
        check_cuda(cudaGetLastError(), "CUDA chart external rotation frame");
    }
    frame_mo = Matrix<double>(ctx, basis, basis);
    chart_compose<<<grid(basis * basis), kThreads, 0, ctx.stream()>>>(
        frame_mo.data(), core.view(), frame_external.view(), inactive, basis, basis);
    check_cuda(cudaGetLastError(), "CUDA chart complete MO frame");
    full_rotation_frame = active_subspace;
    for (const Meta& meta : metas) {
        if (meta.differentiable != basis)
            full_rotation_frame = false;
        for (int i = 0; i < meta.differentiable; ++i)
            if (meta.local_ao[i] != i)
                full_rotation_frame = false;
    }

    for (std::size_t index = 0; index < locals.size(); ++index) {
        Local& block = locals[index];
        if (!block.size)
            continue;
        const Meta& meta = metas[index];
        const int q = block.size;
        Matrix<double> dense(ctx, basis, q);
        chart_zero(ctx, dense.view());
        chart_scatter_basis<<<grid(meta.differentiable * q), kThreads, 0, ctx.stream()>>>(
            block.basis.view(),
            meta.local_device.data(),
            dense.data(),
            basis,
            meta.differentiable,
            q);
        check_cuda(cudaGetLastError(), "CUDA chart AO tangent scatter");
        Matrix<double> x(ctx, meta.stored, 1), s(ctx, meta.stored, meta.stored);
        chart_gather_coeff<<<grid(meta.stored), kThreads, 0, ctx.stream()>>>(
            geometry.raw(), meta.orbital, x.data(), meta.stored);
        chart_gather_overlap<<<grid(meta.stored * meta.stored), kThreads, 0, ctx.stream()>>>(
            overlap, meta.stored_device.data(), s.data(), meta.stored);
        check_cuda(cudaGetLastError(), "CUDA chart rotation orbital norm gather");
        auto sx = chart_product(ctx, s.view(), x.view());
        double norm2 = 0.0;
        check_cublas(cublasDdot_64(ctx.blas(), meta.stored, x.data(), 1, sx.data(), 1, &norm2),
                     "cublasDdot chart rotation orbital norm");
        if (!(norm2 > 0.0))
            throw std::runtime_error("CUDA chart nonpositive rotation orbital norm");
        const Const normalized = geometry.normalized().column(meta.orbital);
        auto sdense = chart_product(ctx, overlap, dense.view());
        auto radial = chart_product(ctx, normalized, sdense.view(), Operation::transpose);
        auto removed = chart_product(ctx, normalized, radial.view());
        chart_subtract<<<grid(basis * q), kThreads, 0, ctx.stream()>>>(
            dense.data(), removed.data(), basis * q);
        chart_scale_vector<<<grid(basis * q), kThreads, 0, ctx.stream()>>>(
            dense.data(), basis * q, 1.0 / std::sqrt(norm2));
        check_cuda(cudaGetLastError(), "CUDA chart normalized rotation tangent");
        if (meta.orbital >= inactive && inactive) {
            auto image = chart_product(ctx, overlap, dense.view());
            auto projection = chart_product(ctx, core.view(), image.view(), Operation::transpose);
            auto inactive_part = chart_product(ctx, core.view(), projection.view());
            chart_subtract<<<grid(basis * q), kThreads, 0, ctx.stream()>>>(
                dense.data(), inactive_part.data(), basis * q);
            check_cuda(cudaGetLastError(), "CUDA chart active rotation tangent");
        }
        auto image = chart_product(ctx, overlap, dense.view());
        block.rotation_components =
            chart_product(ctx, frame_external.view(), image.view(), Operation::transpose);
        const bool is_core = meta.orbital < inactive;
        const Const inverse =
            is_core ? inactive_inverse_transform.view() : geometry.frame_inverse_transform();
        const int source_count = is_core ? inactive : active;
        const std::int64_t source_row = is_core ? meta.orbital : meta.orbital - inactive;
        block.source_weights = Matrix<double>(ctx, source_count, 1);
        chart_take_row<<<grid(source_count), kThreads, 0, ctx.stream()>>>(
            inverse, source_row, block.source_weights.data(), source_count);
        check_cuda(cudaGetLastError(), "CUDA chart rotation source row");
        int pivot = 0;
        check_cublas(cublasIdamax(ctx.blas(), source_count, block.source_weights.data(), 1, &pivot),
                     "cublasIdamax rotation source row");
        double source_value = 0.0;
        ctx.copy_to_host(&source_value, block.source_weights.data() + pivot - 1, sizeof(double));
        block.source_weight_max = std::abs(source_value);
    }
}

void AcceptedOrbitalChart::Impl::build_physical_metric_factors(const OrbitalGeometry& geometry) {
    Context& ctx = context;
    const Const overlap = ao_overlap;
    metric_core = Matrix<double>(ctx, basis, inactive);
    metric_active = Matrix<double>(ctx, basis, active);
    chart_zero(ctx, metric_core.view());
    chart_zero(ctx, metric_active.view());
    if (inactive + active) {
        chart_scatter_raw_dense<<<grid(basis * (inactive + active)), kThreads, 0, ctx.stream()>>>(
            geometry.raw(),
            full_support.data(),
            stored_counts.data(),
            metric_core.data(),
            metric_active.data(),
            basis,
            inactive,
            active);
        check_cuda(cudaGetLastError(), "CUDA chart physical metric accepted raw scatter");
    }
    if (inactive) {
        auto sc = chart_product(ctx, overlap, metric_core.view());
        auto gram = chart_product(ctx, metric_core.view(), sc.view(), Operation::transpose);
        auto eig = symmetric_spectrum(ctx, solver, gram.view());
        if (!(eig.host_values.front() > 0.0))
            throw std::runtime_error("singular inactive span in CUDA physical metric");
        metric_core_inverse = spectral_power(ctx, eig, -1.0);
        metric_core_root = spectral_power(ctx, eig, -0.5);
        metric_CGi = chart_product(ctx, metric_core.view(), metric_core_inverse.view());
    }
    if (active) {
        metric_SA = chart_product(ctx, overlap, metric_active.view());
        metric_projected_active = chart_projector_apply(
            ctx, overlap, metric_core.view(), metric_core_inverse.view(), metric_active.view());
        metric_ST = chart_product(ctx, overlap, metric_projected_active.view());
        auto gram = chart_product(
            ctx, metric_projected_active.view(), metric_ST.view(), Operation::transpose);
        metric_ray_norms = Matrix<double>(ctx, active, 1);
        chart_diagonal<<<grid(active), kThreads, 0, ctx.stream()>>>(
            gram.view(), metric_ray_norms.data(), active);
        check_cuda(cudaGetLastError(), "CUDA chart active ray norms");
        std::vector<double> host_norms(active);
        ctx.copy_to_host(host_norms.data(), metric_ray_norms.data(), active * sizeof(double));
        for (const double norm : host_norms)
            if (!(norm > 0.0) || !std::isfinite(norm))
                throw std::runtime_error("degenerate projected active ray in CUDA physical metric");
        if (active_subspace) {
            auto eig = symmetric_spectrum(ctx, solver, gram.view());
            if (!(eig.host_values.front() > 0.0))
                throw std::runtime_error("singular active subspace in CUDA physical metric");
            metric_active_inverse = spectral_power(ctx, eig, -1.0);
            metric_active_root = spectral_power(ctx, eig, -0.5);
        }
        if (inactive) {
            auto inner =
                chart_product(ctx, metric_core.view(), metric_SA.view(), Operation::transpose);
            metric_K = chart_product(ctx, metric_core_inverse.view(), inner.view());
        }
    }
}

OrbitalChartMap::OrbitalChartMap(Context& context,
                                 int basis,
                                 int orbitals,
                                 int reduced_size,
                                 const std::vector<int>& stored_counts,
                                 const std::vector<OrbitalTangentBlock>& blocks)
    : context_(&context),
      basis_(checked_basis(basis, orbitals, reduced_size, stored_counts.size())),
      orbitals_(orbitals), reduced_size_(reduced_size), mapped_rows_(0), raw_slots_(context, 0),
      row_block_(context, 0), row_local_(context, 0), block_first_row_(context, blocks.size()),
      block_row_count_(context, blocks.size()), block_first_basis_(context, blocks.size()),
      block_reduced_offset_(context, blocks.size()), block_reduced_size_(context, blocks.size()),
      coordinate_block_(context, reduced_size_), coordinate_local_(context, reduced_size_),
      stored_counts_(context, stored_counts.size()), retraction_info_(context, 1),
      basis_values_(context, 0), scratch_(context, basis_, orbitals_) {
    for (const int count : stored_counts)
        if (count < 0 || count > basis)
            throw std::invalid_argument("CUDA orbital chart stored support is invalid");
    std::vector<int> slots, row_block, row_local, first_row, row_count, first_basis;
    std::vector<int> coordinate_block(reduced_size, -1), coordinate_local(reduced_size, -1);
    std::vector<int> offsets, sizes;
    std::vector<double> basis_data;
    std::vector<unsigned char> seen(static_cast<std::size_t>(basis) * orbitals, 0);
    for (std::size_t b = 0; b < blocks.size(); ++b) {
        const auto& block = blocks[b];
        const std::int64_t rows = core::checked_index<std::int64_t>(block.raw_slots.size());
        if (rows <= 0 || block.reduced_size <= 0 || block.reduced_offset < 0 ||
            block.reduced_offset + block.reduced_size > reduced_size ||
            block.basis.size() != static_cast<std::size_t>(rows) * block.reduced_size)
            throw std::invalid_argument("CUDA orbital tangent block dimensions are invalid");
        first_row.push_back(core::checked_index<std::int64_t>(slots.size()));
        row_count.push_back(rows);
        first_basis.push_back(core::checked_index<std::int64_t>(basis_data.size()));
        offsets.push_back(block.reduced_offset);
        sizes.push_back(block.reduced_size);
        const int orbital = block.raw_slots.front() / basis;
        for (int i = 0; i < rows; ++i) {
            const int slot = block.raw_slots[i];
            if (slot < 0 || slot >= basis * orbitals || slot / basis != orbital ||
                slot % basis >= stored_counts[orbital] || seen[slot]++)
                throw std::invalid_argument("CUDA orbital tangent slots are invalid or overlap");
            slots.push_back(slot);
            row_block.push_back(static_cast<int>(b));
            row_local.push_back(i);
        }
        for (int j = 0; j < block.reduced_size; ++j) {
            const int coordinate = block.reduced_offset + j;
            if (coordinate_block[coordinate] >= 0)
                throw std::invalid_argument("CUDA orbital tangent coordinates overlap");
            coordinate_block[coordinate] = static_cast<int>(b);
            coordinate_local[coordinate] = j;
        }
        for (const double value : block.basis)
            if (!std::isfinite(value))
                throw std::invalid_argument("CUDA orbital tangent basis is not finite");
        basis_data.insert(basis_data.end(), block.basis.begin(), block.basis.end());
    }
    if (std::find(coordinate_block.begin(), coordinate_block.end(), -1) != coordinate_block.end())
        throw std::invalid_argument("CUDA orbital tangent coordinates are incomplete");
    mapped_rows_ = core::checked_index<std::int64_t>(slots.size());
    raw_slots_ = Buffer<int>(context, slots.size());
    row_block_ = Buffer<int>(context, row_block.size());
    row_local_ = Buffer<int>(context, row_local.size());
    basis_values_ = Buffer<double>(context, basis_data.size());
    context.copy_to_device(raw_slots_.data(), slots.data(), slots.size() * sizeof(int));
    context.copy_to_device(row_block_.data(), row_block.data(), row_block.size() * sizeof(int));
    context.copy_to_device(row_local_.data(), row_local.data(), row_local.size() * sizeof(int));
    context.copy_to_device(
        block_first_row_.data(), first_row.data(), first_row.size() * sizeof(int));
    context.copy_to_device(
        block_row_count_.data(), row_count.data(), row_count.size() * sizeof(int));
    context.copy_to_device(
        block_first_basis_.data(), first_basis.data(), first_basis.size() * sizeof(int));
    context.copy_to_device(
        block_reduced_offset_.data(), offsets.data(), offsets.size() * sizeof(int));
    context.copy_to_device(block_reduced_size_.data(), sizes.data(), sizes.size() * sizeof(int));
    context.copy_to_device(
        coordinate_block_.data(), coordinate_block.data(), coordinate_block.size() * sizeof(int));
    context.copy_to_device(
        coordinate_local_.data(), coordinate_local.data(), coordinate_local.size() * sizeof(int));
    context.copy_to_device(
        stored_counts_.data(), stored_counts.data(), stored_counts.size() * sizeof(int));
    context.copy_to_device(
        basis_values_.data(), basis_data.data(), basis_data.size() * sizeof(double));
}

void OrbitalChartMap::expand(linalg::ConstMatrixView reduced,
                             linalg::MatrixView raw_direction) const {
    check_view(reduced, reduced_size_, 1, "CUDA orbital reduced direction");
    check_view(raw_direction, basis_, orbitals_, "CUDA orbital raw direction");
    for (int j = 0; j < orbitals_; ++j)
        check_cuda(cudaMemsetAsync(raw_direction.column(j).data,
                                   0,
                                   static_cast<std::size_t>(basis_) * sizeof(double),
                                   context_->stream()),
                   "CUDA orbital chart direction zero");
    if (mapped_rows_) {
        expand_rows<<<grid(mapped_rows_), kThreads, 0, context_->stream()>>>(
            raw_slots_.data(),
            row_block_.data(),
            row_local_.data(),
            block_first_basis_.data(),
            block_row_count_.data(),
            block_reduced_offset_.data(),
            block_reduced_size_.data(),
            basis_values_.data(),
            reduced.data,
            raw_direction.data,
            mapped_rows_,
            basis_,
            static_cast<int>(raw_direction.ld));
        check_cuda(cudaGetLastError(), "CUDA orbital chart tangent expansion");
    }
}

void OrbitalChartMap::project(linalg::ConstMatrixView raw_covector,
                              linalg::MatrixView reduced_covector) const {
    check_view(raw_covector, basis_, orbitals_, "CUDA orbital raw covector");
    check_view(reduced_covector, reduced_size_, 1, "CUDA orbital reduced covector");
    if (reduced_size_) {
        project_coordinates<<<grid(reduced_size_), kThreads, 0, context_->stream()>>>(
            raw_slots_.data(),
            block_first_row_.data(),
            block_row_count_.data(),
            block_first_basis_.data(),
            coordinate_block_.data(),
            coordinate_local_.data(),
            basis_values_.data(),
            raw_covector.data,
            reduced_covector.data,
            reduced_size_,
            basis_,
            static_cast<int>(raw_covector.ld));
        check_cuda(cudaGetLastError(), "CUDA orbital chart covector projection");
    }
}

void OrbitalChartMap::retract(linalg::ConstMatrixView accepted,
                              linalg::ConstMatrixView reduced_step,
                              double scale,
                              linalg::MatrixView trial) const {
    check_view(accepted, basis_, orbitals_, "CUDA accepted raw orbitals");
    check_view(reduced_step, reduced_size_, 1, "CUDA orbital reduced step");
    check_view(trial, basis_, orbitals_, "CUDA trial raw orbitals");
    if (!std::isfinite(scale))
        throw std::invalid_argument("CUDA orbital chart step scale is not finite");
    expand(reduced_step, scratch_.view());
    check_cuda(cudaMemsetAsync(retraction_info_.data(), 0, sizeof(int), context_->stream()),
               "CUDA orbital retraction info reset");
    add_step<<<grid(basis_ * orbitals_), kThreads, 0, context_->stream()>>>(accepted,
                                                                            scratch_.data(),
                                                                            scale,
                                                                            trial,
                                                                            stored_counts_.data(),
                                                                            retraction_info_.data(),
                                                                            basis_,
                                                                            orbitals_);
    check_cuda(cudaGetLastError(), "CUDA orbital chart retraction");
    int bad = 0;
    context_->copy_to_host(&bad, retraction_info_.data(), sizeof(int));
    if (bad)
        throw std::runtime_error("CUDA orbital retraction produced nonfinite slot " +
                                 std::to_string(bad - 1));
}

} // namespace xmvb::cuda
namespace xmvb::cuda {
namespace {

void chart_zero(Context& ctx, Mut output) {
    if (!output.rows || !output.cols)
        return;
    for (int j = 0; j < output.cols; ++j)
        check_cuda(cudaMemsetAsync(output.column(j).data,
                                   0,
                                   static_cast<std::size_t>(output.rows) * sizeof(double),
                                   ctx.stream()),
                   "CUDA accepted chart zero");
}

void chart_copy(Context& ctx, Mut target, Const source) {
    if (target.rows != source.rows || target.cols != source.cols)
        throw std::invalid_argument("accepted chart copy shape mismatch");
    if (!source.rows || !source.cols)
        return;
    if (target.data == source.data && target.ld == source.ld)
        return;
    if (target.ld == target.rows && source.ld == source.rows) {
        check_cuda(
            cudaMemcpyAsync(target.data,
                            source.data,
                            static_cast<std::size_t>(source.rows) * source.cols * sizeof(double),
                            cudaMemcpyDeviceToDevice,
                            ctx.stream()),
            "CUDA accepted chart contiguous copy");
    } else {
        check_cuda(cudaMemcpy2DAsync(target.data,
                                     target.ld * sizeof(double),
                                     source.data,
                                     source.ld * sizeof(double),
                                     source.rows * sizeof(double),
                                     source.cols,
                                     cudaMemcpyDeviceToDevice,
                                     ctx.stream()),
                   "CUDA accepted chart pitched copy");
    }
}

Matrix<double> chart_product(Context& ctx, Const a, Const b, Operation opa, Operation opb) {
    const int rows = static_cast<int>(opa == Operation::none ? a.rows : a.cols);
    const int cols = static_cast<int>(opb == Operation::none ? b.cols : b.rows);
    Matrix<double> result(ctx, rows, cols);
    if (rows && cols)
        gemm(ctx, result.view(), a, b, 1.0, 0.0, opa, opb);
    return result;
}

__global__ void chart_identity(double* x, int n) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < n * n)
        x[k] = (k % n == k / n) ? 1.0 : 0.0;
}

Matrix<double> chart_eye(Context& ctx, int n) {
    Matrix<double> result(ctx, n, n);
    if (n) {
        chart_identity<<<grid(n * n), kThreads, 0, ctx.stream()>>>(result.data(), n);
        check_cuda(cudaGetLastError(), "CUDA accepted chart identity");
    }
    return result;
}

__global__ void chart_scatter_inactive(
    Const raw, const int* support, const int* counts, double* output, int basis, int inactive) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= basis * inactive)
        return;
    const int orbital = k / basis, slot = k % basis;
    if (slot < counts[orbital])
        output[orbital * basis + support[k]] = raw.data[orbital * raw.ld + slot];
}

__global__ void
chart_gather_rows(Const source, const int* rows, double* output, int count, int columns) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < count * columns)
        output[k] = source.data[(k / count) * source.ld + rows[k % count]];
}

__global__ void chart_gather_overlap(Const ao, const int* rows, double* output, int count) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < count * count) {
        const int i = k % count, j = k / count;
        output[k] = ao.data[rows[j] * ao.ld + rows[i]];
    }
}

__global__ void chart_gather_coeff(Const raw, int orbital, double* output, int count) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < count)
        output[k] = raw.data[orbital * raw.ld + k];
}

__global__ void chart_subtract(double* y, const double* x, int n) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < n)
        y[k] -= x[k];
}

__global__ void
chart_compose(double* target, Const left, Const right, int first_right, int rows, int cols) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < rows * cols) {
        const int i = k % rows, j = k / rows;
        target[k] = j < first_right ? left.data[j * left.ld + i]
                                    : right.data[(j - first_right) * right.ld + i];
    }
}

__global__ void chart_ray_reflector(const double* row, double* v, int n, double denominator) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        v[i] = i == 0 ? 1.0 : row[i] / denominator;
}

__global__ void chart_sign_columns(double* x, int rows, int cols) {
    const std::int64_t j = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= cols)
        return;
    int pivot = 0;
    double maximum = 0.0;
    for (int i = 0; i < rows; ++i) {
        const double value = fabs(x[j * rows + i]);
        if (value > maximum) {
            maximum = value;
            pivot = i;
        }
    }
    if (x[j * rows + pivot] < 0.0)
        for (int i = 0; i < rows; ++i)
            x[j * rows + i] = -x[j * rows + i];
}

__global__ void chart_stored_quotient(Const u, double* output, int stored, int local, int columns) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < stored * columns) {
        const int i = k % stored, j = k / stored;
        output[k] = i < local ? u.data[j * u.ld + i] : 0.0;
    }
}

__global__ void chart_scale_vector(double* x, int n, double scale) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < n)
        x[k] *= scale;
}

__global__ void chart_normalize_columns(Mut values, Const gram) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < values.rows * values.cols) {
        const std::int64_t row = k % values.rows, col = k / values.rows;
        values.data[row + col * values.ld] /= sqrt(gram.data[col + col * gram.ld]);
    }
}

__global__ void chart_make_basis(
    Const u, Const x, Const r, double* out, int local, int q, double norm, bool sphere) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < local * q) {
        const int i = k % local, j = k / local;
        out[k] = norm * u.data[j * u.ld + i] - (sphere ? x.data[i] * r.data[j] : 0.0);
    }
}

__global__ void chart_rank_one(double* metric, const double* r, int q) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < q * q)
        metric[k] -= r[k % q] * r[k / q];
}

__global__ void chart_add_retraction(Const accepted,
                                     Const tangent,
                                     Mut trial,
                                     const int* stored,
                                     int* info,
                                     int basis,
                                     int orbitals,
                                     double scale) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= basis * orbitals)
        return;
    const int i = k % basis, j = k / basis;
    const double value = i < stored[j] ? accepted.data[j * accepted.ld + i] +
                                             scale * tangent.data[j * tangent.ld + i]
                                       : 0.0;
    trial.data[j * trial.ld + i] = value;
    if (!isfinite(value))
        atomicCAS(info, 0, k + 1);
}

__global__ void chart_sphere_flag(Const raw, int orbital, int first, int stored, int* flag) {
    if (blockIdx.x || threadIdx.x)
        return;
    int result = 1;
    for (int i = first; i < stored; ++i)
        if (raw.data[orbital * raw.ld + i] != 0.0)
            result = 0;
    *flag = result;
}

__global__ void
chart_scatter_basis(Const local, const int* ao, double* dense, int basis, int rows, int columns) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < rows * columns) {
        const int i = k % rows, j = k / rows;
        dense[j * basis + ao[i]] = local.data[j * local.ld + i];
    }
}

__global__ void chart_take_row(Const matrix, int row, double* output, int columns) {
    const std::int64_t j = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j < columns)
        output[j] = matrix.data[j * matrix.ld + row];
}

__global__ void chart_scatter_raw_dense(Const raw,
                                        const int* support,
                                        const int* counts,
                                        double* core,
                                        double* active,
                                        int basis,
                                        int ncore,
                                        int nactive) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= basis * (ncore + nactive))
        return;
    const int orbital = k / basis, slot = k % basis;
    if (slot >= counts[orbital])
        return;
    const int ao = support[k];
    if (orbital < ncore)
        core[orbital * basis + ao] = raw.data[orbital * raw.ld + slot];
    else
        active[(orbital - ncore) * basis + ao] = raw.data[orbital * raw.ld + slot];
}

__global__ void chart_gather_dense_raw(Const core,
                                       Const active,
                                       const int* support,
                                       const int* counts,
                                       Mut raw,
                                       int basis,
                                       int ncore,
                                       int nactive) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= basis * (ncore + nactive))
        return;
    const int orbital = k / basis, slot = k % basis;
    if (slot >= counts[orbital])
        return;
    const int ao = support[k];
    raw.data[orbital * raw.ld + slot] = orbital < ncore
                                            ? core.data[orbital * core.ld + ao]
                                            : active.data[(orbital - ncore) * active.ld + ao];
}

__global__ void chart_ray_project(Const direction,
                                  Const ray,
                                  Const sray,
                                  const double* norms,
                                  Mut output,
                                  int basis,
                                  int active) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= basis * active)
        return;
    const int ao = k % basis, j = k / basis;
    double component = 0.0;
    for (int i = 0; i < basis; ++i)
        component += sray.data[j * sray.ld + i] * direction.data[j * direction.ld + i];
    output.data[j * output.ld + ao] =
        (direction.data[j * direction.ld + ao] - ray.data[j * ray.ld + ao] * component / norms[j]) /
        sqrt(norms[j]);
}

__global__ void chart_diagonal(Const matrix, double* output, int n) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        output[i] = matrix.data[i * matrix.ld + i];
}

__global__ void chart_rotation_weights(Const diagonal,
                                       Const source,
                                       double* output,
                                       int external,
                                       int source_count,
                                       int first_row,
                                       int first_source) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= external)
        return;
    double value = 0.0;
    for (int j = 0; j < source_count; ++j) {
        const double weight = source.data[j];
        value +=
            diagonal.data[(first_source + j) * diagonal.ld + first_row + row] * weight * weight;
    }
    output[row] = value;
}

__global__ void chart_rotation_curvature(
    const double* components, const double* weights, double* weighted, int external, int q) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < external * q)
        weighted[k] = components[k] * weights[k % external];
}

__global__ void chart_rotation_diagonal(
    Const components, const double* weights, double* output, int external, int q) {
    const std::int64_t j = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= q)
        return;
    double value = 0.0;
    for (int i = 0; i < external; ++i) {
        const double component = components.data[j * components.ld + i];
        value += weights[i] * component * component;
    }
    output[j] = value;
}

__global__ void chart_ah_divide(
    double* frame, Const diagonal, int rows, int cols, int first_row, double eigenvalue) {
    const std::int64_t k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= rows * cols)
        return;
    const int i = k % rows, j = k / rows;
    double denominator = diagonal.data[j * diagonal.ld + first_row + i] - eigenvalue;
    if (fabs(denominator) < 1.0e-8)
        denominator = 1.0e-8;
    frame[k] /= denominator;
}

__global__ void chart_subtract_identity(double* matrix, int n) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        matrix[i * n + i] -= 1.0;
}

Matrix<double>
chart_projector_apply(Context& ctx, Const overlap, Const carrier, Const inverse, Const x) {
    Matrix<double> result(ctx, x.rows, x.cols);
    chart_copy(ctx, result.view(), x);
    if (!carrier.cols || !x.cols)
        return result;
    auto sx = chart_product(ctx, overlap, x);
    auto inner = chart_product(ctx, carrier, sx.view(), Operation::transpose);
    auto weights = chart_product(ctx, inverse, inner.view());
    auto removed = chart_product(ctx, carrier, weights.view());
    const double minus_one = -1.0;
    check_cublas(cublasDaxpy_64(ctx.blas(),
                                static_cast<int>(x.rows * x.cols),
                                &minus_one,
                                removed.data(),
                                1,
                                result.data(),
                                1),
                 "cublasDaxpy chart physical projector");
    return result;
}

ChartSvd chart_svd(Context& ctx, Const input) {
    const int m = static_cast<int>(input.rows), n = static_cast<int>(input.cols);
    if (!n)
        return {Matrix<double>(ctx, 0, 0), 0};
    if (!m)
        return {chart_eye(ctx, n), 0};
    auto svd = right_svd(ctx, input);
    const double maximum = *std::max_element(svd.values.begin(), svd.values.end());
    const double tolerance = DBL_EPSILON * std::max(m, n) * maximum;
    int rank = 0;
    for (double sigma : svd.values) {
        if (!std::isfinite(sigma))
            throw std::runtime_error("nonfinite CUDA chart singular value");
        if (sigma > tolerance)
            ++rank;
    }
    return {std::move(svd.right), rank};
}

Buffer<int> chart_indices(Context& ctx, const std::vector<int>& values) {
    Buffer<int> result(ctx, values.size());
    ctx.copy_to_device(result.data(), values.data(), values.size() * sizeof(int));
    return result;
}

void chart_check_view(Const view, int rows, int cols, const char* label) {
    if (rows == 0 && view.rows == 0 && view.cols == 1)
        return;
    if (!view.data || view.rows != rows || view.cols != cols || view.ld < rows)
        throw std::invalid_argument(std::string(label) + " shape mismatch");
}

} // namespace
void AcceptedOrbitalChart::expand(Const reduced, Mut raw_direction) const {
    auto& p = *impl_;
    chart_check_view(reduced, p.reduced, 1, "CUDA accepted chart direction");
    chart_check_view(raw_direction, p.basis, p.orbitals, "CUDA accepted chart raw direction");
    chart_zero(p.context, raw_direction);
    p.basis_map.apply(reduced.data, raw_direction.data, false, 1, raw_direction.ld);
}

void AcceptedOrbitalChart::project(Const raw_covector, Mut reduced_covector) const {
    auto& p = *impl_;
    chart_check_view(raw_covector, p.basis, p.orbitals, "CUDA accepted chart raw covector");
    chart_check_view(reduced_covector, p.reduced, 1, "CUDA accepted chart covector");
    p.basis_map.apply(raw_covector.data, reduced_covector.data, true, raw_covector.ld, 1);
}

void AcceptedOrbitalChart::project_vector(Const raw_direction, Mut reduced_vector) const {
    auto& p = *impl_;
    Context& ctx = p.context;
    chart_check_view(raw_direction, p.basis, p.orbitals, "CUDA accepted chart raw vector");
    chart_check_view(reduced_vector, p.reduced, 1, "CUDA accepted chart reduced vector");
    for (std::size_t index = 0; index < p.locals.size(); ++index) {
        const auto& block = p.locals[index];
        if (!block.size)
            continue;
        const auto& meta = p.metas[index];
        const int m = block.differentiable;
        Matrix<double> overlap(ctx, m, m);
        const Const local{raw_direction.data + block.orbital * raw_direction.ld, m, 1, m};
        chart_gather_overlap<<<grid(m * m), kThreads, 0, ctx.stream()>>>(
            p.ao_overlap, meta.local_device.data(), overlap.data(), m);
        check_cuda(cudaGetLastError(), "CUDA chart vector AO metric gather");
        auto covariant = chart_product(ctx, overlap.view(), local);
        auto pulled =
            chart_product(ctx, block.basis.view(), covariant.view(), Operation::transpose);
        const double alpha = 1.0, beta = 0.0;
        check_cublas(cublasDgemv_64(ctx.blas(),
                                    CUBLAS_OP_N,
                                    block.size,
                                    block.size,
                                    &alpha,
                                    block.raw_inverse_gram.data(),
                                    block.size,
                                    pulled.data(),
                                    1,
                                    &beta,
                                    reduced_vector.data + block.offset,
                                    1),
                     "cublasDgemv chart raw-vector projection");
    }
}

void AcceptedOrbitalChart::retract(Const accepted,
                                   Const reduced_step,
                                   double scale,
                                   Mut trial) const {
    auto& p = *impl_;
    chart_check_view(accepted, p.basis, p.orbitals, "CUDA accepted chart raw point");
    chart_check_view(reduced_step, p.reduced, 1, "CUDA accepted chart step");
    chart_check_view(trial, p.basis, p.orbitals, "CUDA accepted chart trial");
    if (!std::isfinite(scale))
        throw std::invalid_argument("CUDA chart nonfinite step scale");
    expand(reduced_step, p.tangent.view());
    check_cuda(cudaMemsetAsync(p.bad_info.data(), 0, sizeof(int), p.context.stream()),
               "CUDA accepted chart retraction status reset");
    chart_add_retraction<<<grid(p.basis * p.orbitals), kThreads, 0, p.context.stream()>>>(
        accepted,
        p.tangent.view(),
        trial,
        p.stored_counts.data(),
        p.bad_info.data(),
        p.basis,
        p.orbitals,
        scale);
    check_cuda(cudaGetLastError(), "CUDA accepted chart retraction");
    int bad = 0;
    p.context.copy_to_host(&bad, p.bad_info.data(), sizeof(int));
    if (bad)
        throw std::runtime_error("CUDA orbital retraction produced nonfinite slot " +
                                 std::to_string(bad - 1));
}

void AcceptedOrbitalChart::metric(Const vector, Mut covector) const {
    auto& p = *impl_;
    chart_check_view(vector, p.reduced, 1, "CUDA chart metric vector");
    chart_check_view(covector, p.reduced, 1, "CUDA chart metric covector");
    p.metric_map.apply(vector.data, covector.data);
}

void AcceptedOrbitalChart::physical_metric(Const vector, Mut covector) const {
    const nvtx3::scoped_range range{"chart.physical_metric"};
    auto& p = *impl_;
    Context& ctx = p.context;
    chart_check_view(vector, p.reduced, 1, "CUDA chart physical metric vector");
    chart_check_view(covector, p.reduced, 1, "CUDA chart physical metric covector");
    if (!p.reduced)
        return;
    const Const overlap = p.ao_overlap;
    Matrix<double> raw_direction(ctx, p.basis, p.orbitals);
    expand(vector, raw_direction.view());
    Matrix<double> di(ctx, p.basis, p.inactive), da(ctx, p.basis, p.active);
    chart_zero(ctx, di.view());
    chart_zero(ctx, da.view());
    chart_scatter_raw_dense<<<grid(p.basis * (p.inactive + p.active)), kThreads, 0, ctx.stream()>>>(
        raw_direction.view(),
        p.full_support.data(),
        p.stored_counts.data(),
        di.data(),
        da.data(),
        p.basis,
        p.inactive,
        p.active);
    check_cuda(cudaGetLastError(), "CUDA chart physical tangent AO scatter");
    Matrix<double> hi(ctx, p.basis, p.inactive);
    if (p.inactive)
        hi = chart_projector_apply(
            ctx, overlap, p.metric_core.view(), p.metric_core_inverse.view(), di.view());
    Matrix<double> fi(ctx, p.basis, p.inactive);
    if (p.inactive)
        fi = chart_product(ctx, hi.view(), p.metric_core_root.view());
    Matrix<double> delta_t(ctx, p.basis, p.active);
    if (p.active) {
        delta_t = chart_projector_apply(
            ctx, overlap, p.metric_core.view(), p.metric_core_inverse.view(), da.view());
        if (p.inactive) {
            auto direct = chart_product(ctx, hi.view(), p.metric_K.view());
            auto overlap_active =
                chart_product(ctx, hi.view(), p.metric_SA.view(), Operation::transpose);
            auto feedback = chart_product(ctx, p.metric_CGi.view(), overlap_active.view());
            const double minus_one = -1.0;
            check_cublas(cublasDaxpy_64(ctx.blas(),
                                        p.basis * p.active,
                                        &minus_one,
                                        direct.data(),
                                        1,
                                        delta_t.data(),
                                        1),
                         "cublasDaxpy chart physical inactive-active direct");
            check_cublas(cublasDaxpy_64(ctx.blas(),
                                        p.basis * p.active,
                                        &minus_one,
                                        feedback.data(),
                                        1,
                                        delta_t.data(),
                                        1),
                         "cublasDaxpy chart physical inactive-active feedback");
        }
    }
    Matrix<double> fa(ctx, p.basis, p.active);
    if (p.active) {
        if (p.active_subspace) {
            auto horizontal = chart_projector_apply(ctx,
                                                    overlap,
                                                    p.metric_projected_active.view(),
                                                    p.metric_active_inverse.view(),
                                                    delta_t.view());
            fa = chart_product(ctx, horizontal.view(), p.metric_active_root.view());
        } else {
            chart_ray_project<<<grid(p.basis * p.active), kThreads, 0, ctx.stream()>>>(
                delta_t.view(),
                p.metric_projected_active.view(),
                p.metric_ST.view(),
                p.metric_ray_norms.data(),
                fa.view(),
                p.basis,
                p.active);
            check_cuda(cudaGetLastError(), "CUDA chart physical selected-ray feature");
        }
    }

    Matrix<double> yi(ctx, p.basis, p.inactive), ya(ctx, p.basis, p.active);
    if (p.inactive)
        yi = chart_product(ctx, fi.view(), p.metric_core_root.view());
    if (p.active) {
        if (p.active_subspace) {
            auto mixed = chart_product(ctx, fa.view(), p.metric_active_root.view());
            ya = chart_projector_apply(ctx,
                                       overlap,
                                       p.metric_projected_active.view(),
                                       p.metric_active_inverse.view(),
                                       mixed.view());
        } else {
            chart_ray_project<<<grid(p.basis * p.active), kThreads, 0, ctx.stream()>>>(
                fa.view(),
                p.metric_projected_active.view(),
                p.metric_ST.view(),
                p.metric_ray_norms.data(),
                ya.view(),
                p.basis,
                p.active);
            check_cuda(cudaGetLastError(), "CUDA chart physical selected-ray adjoint");
        }
        if (p.inactive) {
            auto direct = chart_product(
                ctx, ya.view(), p.metric_K.view(), Operation::none, Operation::transpose);
            auto scgi = chart_product(ctx, overlap, p.metric_CGi.view());
            auto inner = chart_product(ctx, ya.view(), scgi.view(), Operation::transpose);
            auto feedback = chart_product(ctx, p.metric_active.view(), inner.view());
            const double minus_one = -1.0;
            check_cublas(
                cublasDaxpy_64(
                    ctx.blas(), p.basis * p.inactive, &minus_one, direct.data(), 1, yi.data(), 1),
                "cublasDaxpy chart physical core adjoint direct");
            check_cublas(
                cublasDaxpy_64(
                    ctx.blas(), p.basis * p.inactive, &minus_one, feedback.data(), 1, yi.data(), 1),
                "cublasDaxpy chart physical core adjoint feedback");
        }
    }
    if (p.inactive)
        yi = chart_projector_apply(
            ctx, overlap, p.metric_core.view(), p.metric_core_inverse.view(), yi.view());
    if (p.active)
        ya = chart_projector_apply(
            ctx, overlap, p.metric_core.view(), p.metric_core_inverse.view(), ya.view());
    Matrix<double> gi(ctx, p.basis, p.inactive), ga(ctx, p.basis, p.active);
    if (p.inactive)
        gi = chart_product(ctx, overlap, yi.view());
    if (p.active)
        ga = chart_product(ctx, overlap, ya.view());
    Matrix<double> raw_covector(ctx, p.basis, p.orbitals);
    chart_zero(ctx, raw_covector.view());
    chart_gather_dense_raw<<<grid(p.basis * (p.inactive + p.active)), kThreads, 0, ctx.stream()>>>(
        gi.view(),
        ga.view(),
        p.full_support.data(),
        p.stored_counts.data(),
        raw_covector.view(),
        p.basis,
        p.inactive,
        p.active);
    check_cuda(cudaGetLastError(), "CUDA chart physical cotangent sparse gather");
    project(raw_covector.view(), covector);
}

void AcceptedOrbitalChart::inverse_metric(Const covector, Mut vector) const {
    auto& p = *impl_;
    chart_check_view(covector, p.reduced, 1, "CUDA chart inverse metric covector");
    chart_check_view(vector, p.reduced, 1, "CUDA chart inverse metric vector");
    p.inverse_map.apply(covector.data, vector.data);
}

double AcceptedOrbitalChart::coordinate_dual_norm(Const covector) const {
    const nvtx3::scoped_range range{"chart.coordinate_norm"};
    auto& p = *impl_;
    chart_check_view(covector, p.reduced, 1, "CUDA chart dual norm covector");
    if (!p.reduced)
        return 0.0;
    Matrix<double> image(p.context, p.reduced, 1);
    inverse_metric(covector, image.view());
    double squared = 0.0;
    check_cublas(
        cublasDdot_64(p.context.blas(), p.reduced, covector.data, 1, image.data(), 1, &squared),
        "cublasDdot chart dual norm");
    if (!(squared >= 0.0) || !std::isfinite(squared))
        throw std::runtime_error("invalid CUDA chart coordinate dual norm");
    return std::sqrt(squared);
}

double AcceptedOrbitalChart::reference_gradient_inf_norm(Const covector) const {
    const nvtx3::scoped_range range{"chart.reference_norm"};
    auto& p = *impl_;
    chart_check_view(covector, p.reduced, 1, "CUDA chart reference covector");
    double maximum = 0.0;
    for (const auto& block : p.locals) {
        if (!block.size)
            continue;
        Matrix<double> u(p.context, block.size, 1), reference(p.context, block.size, 1);
        const double alpha = 1.0, beta = 0.0;
        check_cublas(cublasDgemv_64(p.context.blas(),
                                    CUBLAS_OP_N,
                                    block.size,
                                    block.size,
                                    &alpha,
                                    block.reference_root.data(),
                                    block.size,
                                    block.projection.data(),
                                    1,
                                    &beta,
                                    u.data(),
                                    1),
                     "cublasDgemv chart reference projection");
        check_cublas(cublasDgemv_64(p.context.blas(),
                                    CUBLAS_OP_N,
                                    block.size,
                                    block.size,
                                    &alpha,
                                    block.reference_root.data(),
                                    block.size,
                                    covector.data + block.offset,
                                    1,
                                    &beta,
                                    reference.data(),
                                    1),
                     "cublasDgemv chart reference covector");
        double dot = 0.0;
        check_cublas(
            cublasDdot_64(p.context.blas(), block.size, u.data(), 1, reference.data(), 1, &dot),
            "cublasDdot chart reference radial term");
        const double root = std::sqrt(block.delta);
        const double correction = dot / (root * (1.0 + root));
        check_cublas(
            cublasDaxpy_64(
                p.context.blas(), block.size, &correction, u.data(), 1, reference.data(), 1),
            "cublasDaxpy chart reference covector");
        int pivot = 0;
        check_cublas(cublasIdamax(p.context.blas(), block.size, reference.data(), 1, &pivot),
                     "cublasIdamax chart reference covector");
        double value = 0.0;
        check_cuda(cudaMemcpyAsync(&value,
                                   reference.data() + pivot - 1,
                                   sizeof(double),
                                   cudaMemcpyDeviceToHost,
                                   p.context.stream()),
                   "CUDA chart reference maximum");
        p.context.synchronize();
        maximum = std::max(maximum, std::abs(value));
    }
    return maximum;
}

double AcceptedOrbitalChart::max_component(Const vector) const {
    const nvtx3::scoped_range range{"chart.maximum_step"};
    auto& p = *impl_;
    chart_check_view(vector, p.reduced, 1, "CUDA chart maximum component vector");
    if (!p.reduced)
        return 0.0;
    const int external_count = p.basis - p.inactive;
    const int virtual_count = p.basis - p.inactive - p.active;
    auto maximum_absolute = [&](const double* device, int count) {
        if (!count)
            return 0.0;
        int pivot = 0;
        check_cublas(cublasIdamax(p.context.blas(), count, device, 1, &pivot),
                     "cublasIdamax chart frame component");
        double value = 0.0;
        p.context.copy_to_host(&value, device + pivot - 1, sizeof(double));
        return std::abs(value);
    };
    Matrix<double> core_frame(p.context, external_count, p.inactive);
    Matrix<double> active_frame(p.context, virtual_count, p.active);
    if (p.full_rotation_frame) {
        chart_zero(p.context, core_frame.view());
        chart_zero(p.context, active_frame.view());
    }
    double maximum = 0.0;
    for (const auto& block : p.locals) {
        if (!block.size)
            continue;
        const bool is_core = block.orbital < p.inactive;
        const std::int64_t first = p.full_rotation_frame && !is_core ? p.active : 0;
        const std::int64_t count = external_count - first;
        if (!count)
            continue;
        Matrix<double> external(p.context, count, 1);
        const double alpha = 1.0, beta = 0.0;
        check_cublas(cublasDgemv_64(p.context.blas(),
                                    CUBLAS_OP_N,
                                    count,
                                    block.size,
                                    &alpha,
                                    block.rotation_components.data() + first,
                                    external_count,
                                    vector.data + block.offset,
                                    1,
                                    &beta,
                                    external.data(),
                                    1),
                     "cublasDgemv chart rotation components");
        if (!p.full_rotation_frame) {
            maximum = std::max(maximum,
                               maximum_absolute(external.data(), count) * block.source_weight_max);
        } else {
            Mut frame = is_core ? core_frame.view() : active_frame.view();
            check_cublas(
                cublasDger_64(p.context.blas(),
                              count,
                              core::checked_index<std::int64_t>(block.source_weights.rows()),
                              &alpha,
                              external.data(),
                              1,
                              block.source_weights.data(),
                              1,
                              frame.data,
                              static_cast<int>(frame.ld)),
                "cublasDger chart complete rotation frame");
        }
    }
    if (p.full_rotation_frame) {
        maximum =
            std::max(maximum, maximum_absolute(core_frame.data(), external_count * p.inactive));
        maximum =
            std::max(maximum, maximum_absolute(active_frame.data(), virtual_count * p.active));
    }
    return maximum;
}

void AcceptedOrbitalChart::set_hessian_diagonal(const OrbitalGeometry& geometry,
                                                Const rotation_diagonal) {
    const nvtx3::scoped_range range{"chart.prepare_diagonal"};
    auto& p = *impl_;
    Context& ctx = p.context;
    chart_check_view(rotation_diagonal, p.basis, p.basis, "CUDA chart analytic rotation diagonal");
    if (geometry.basis_size() != p.basis || geometry.orbital_count() != p.orbitals ||
        p.locals.size() != p.metas.size() || p.frame_mo.rows() != p.basis)
        throw std::logic_error("CUDA chart must be rebuilt before analytic Hessian diagonal");
    p.has_analytic_diagonal = false;
    p.ah_rotation_diagonal = Matrix<double>(ctx, p.basis, p.basis);
    chart_copy(ctx, p.ah_rotation_diagonal.view(), rotation_diagonal);
    p.ah_diagonal = Matrix<double>(ctx, p.reduced, 1);
    const int external_count = p.basis - p.inactive;
    const int virtual_count = p.basis - p.inactive - p.active;
    for (auto& block : p.locals) {
        if (!block.size)
            continue;
        const bool core = block.orbital < p.inactive;
        const int sources = core ? p.inactive : p.active;
        const int first_source = core ? 0 : p.inactive;
        Matrix<double> weights(ctx, external_count, 1);
        chart_rotation_weights<<<grid(external_count), kThreads, 0, ctx.stream()>>>(
            p.ah_rotation_diagonal.view(),
            block.source_weights.view(),
            weights.data(),
            external_count,
            sources,
            p.inactive,
            first_source);
        check_cuda(cudaGetLastError(), "CUDA chart analytic rotation weights");
        chart_rotation_diagonal<<<grid(block.size), kThreads, 0, ctx.stream()>>>(
            block.rotation_components.view(),
            weights.data(),
            p.ah_diagonal.data() + block.offset,
            external_count,
            block.size);
        check_cuda(cudaGetLastError(), "CUDA chart analytic local diagonal");
        if (p.full_rotation_frame) {
            const std::int64_t first = core ? 0 : p.active;
            const int rows = core ? external_count : virtual_count;
            if (block.size != rows)
                throw std::runtime_error("full-AO rotation map dimension mismatch");
            const Const forward{
                block.rotation_components.data() + first, rows, block.size, external_count};
            block.ah_frame_inverse = chart_product(
                ctx, block.raw_inverse_gram.view(), forward, Operation::none, Operation::transpose);
            chart_scale_vector<<<grid(rows * rows), kThreads, 0, ctx.stream()>>>(
                block.ah_frame_inverse.data(), rows * rows, block.raw_norm_squared);
            check_cuda(cudaGetLastError(), "CUDA chart rotation inverse raw norm");
            auto certificate = chart_product(ctx, forward, block.ah_frame_inverse.view());
            chart_subtract_identity<<<grid(rows), kThreads, 0, ctx.stream()>>>(certificate.data(),
                                                                               rows);
            check_cuda(cudaGetLastError(), "CUDA chart rotation inverse certificate");
            double residual = 0.0, forward_norm = 0.0, inverse_norm = 0.0;
            check_cublas(cublasDnrm2_64(ctx.blas(), rows * rows, certificate.data(), 1, &residual),
                         "cublasDnrm2 chart rotation inverse residual");
            double column_norm = 0.0;
            for (std::int64_t column = 0; column < rows; ++column) {
                check_cublas(
                    cublasDnrm2_64(
                        ctx.blas(), rows, forward.data + column * forward.ld, 1, &column_norm),
                    "cublasDnrm2 chart rotation forward column");
                forward_norm += column_norm * column_norm;
            }
            forward_norm = std::sqrt(forward_norm);
            check_cublas(
                cublasDnrm2_64(
                    ctx.blas(), rows * rows, block.ah_frame_inverse.data(), 1, &inverse_norm),
                "cublasDnrm2 chart rotation inverse");
            const double bound = 512.0 * DBL_EPSILON * std::max(1, p.basis) *
                                 std::max(1.0, forward_norm * inverse_norm);
            if (!std::isfinite(residual) || residual > bound)
                throw std::runtime_error(
                    "analytic rotation inverse failed its residual certificate");
            continue;
        }
        Matrix<double> weighted(ctx, external_count, block.size);
        chart_rotation_curvature<<<grid(external_count * block.size), kThreads, 0, ctx.stream()>>>(
            block.rotation_components.data(),
            weights.data(),
            weighted.data(),
            external_count,
            block.size);
        check_cuda(cudaGetLastError(), "CUDA chart analytic weighted rotations");
        auto curvature = chart_product(
            ctx, block.rotation_components.view(), weighted.view(), Operation::transpose);
        auto metric_eig = symmetric_spectrum(ctx, p.solver, block.metric.view());
        auto inverse_root = spectral_power(ctx, metric_eig, -0.5);
        auto work = chart_product(ctx, inverse_root.view(), curvature.view());
        auto standard = chart_product(ctx, work.view(), inverse_root.view());
        auto eig = symmetric_spectrum(ctx, p.solver, standard.view());
        block.ah_signed_values = std::move(eig.values);
        block.ah_signed_vectors = chart_product(ctx, inverse_root.view(), eig.vectors.view());
    }
    if (p.full_rotation_frame) {
        if (p.inactive) {
            auto eig = symmetric_spectrum(ctx, p.solver, p.inactive_inverse_transform.view());
            if (!(eig.host_values.front() > 0.0))
                throw std::runtime_error("singular analytic diagonal inactive source transform");
            p.ah_core_source_inverse = spectral_power(ctx, eig, -1.0);
        }
        if (p.active) {
            p.ah_active_source_inverse = Matrix<double>(ctx, p.active, p.active);
            chart_copy(ctx, p.ah_active_source_inverse.view(), geometry.frame_transform());
        }
    } else {
        std::vector<BlockMapEntry> entries;
        p.signed_spectrum = Matrix<double>(ctx, p.reduced, 1);
        p.signed_scratch = Matrix<double>(ctx, p.reduced, 1);
        for (auto& block : p.locals) {
            if (!block.size)
                continue;
            entries.push_back({block.ah_signed_vectors.view(), block.offset, block.offset});
            chart_copy(ctx,
                       {p.signed_spectrum.data() + block.offset, block.size, 1, block.size},
                       block.ah_signed_values.view());
            block.ah_signed_values = Matrix<double>{};
        }
        p.signed_map = BlockMap(ctx, entries);
    }
    p.has_analytic_diagonal = true;
}

linalg::ConstMatrixView AcceptedOrbitalChart::hessian_diagonal() const noexcept {
    return impl_->ah_diagonal.view();
}

void AcceptedOrbitalChart::apply_inverse_hessian_diagonal(Const covector,
                                                          double eigenvalue,
                                                          Mut direction) const {
    const nvtx3::scoped_range range{"chart.precondition"};
    auto& p = *impl_;
    chart_check_view(covector, p.reduced, 1, "CUDA chart analytic Hessian covector");
    chart_check_view(direction, p.reduced, 1, "CUDA chart analytic Hessian direction");
    if (!p.has_analytic_diagonal)
        throw std::logic_error("CUDA chart analytic Hessian diagonal is unavailable");
    if (!std::isfinite(eigenvalue))
        throw std::invalid_argument("CUDA chart analytic Hessian eigenvalue");
    Context& ctx = p.context;
    if (!p.full_rotation_frame) {
        if (!p.reduced)
            return;
        p.signed_map.apply(covector.data, p.signed_scratch.data(), true);
        chart_ah_divide<<<grid(p.reduced), kThreads, 0, ctx.stream()>>>(
            p.signed_scratch.data(), p.signed_spectrum.view(), p.reduced, 1, 0, eigenvalue);
        check_cuda(cudaGetLastError(), "CUDA chart analytic signed denominator");
        p.signed_map.apply(p.signed_scratch.data(), direction.data);
        return;
    }
    const int external_count = p.basis - p.inactive;
    const int virtual_count = p.basis - p.inactive - p.active;
    Matrix<double> core_sources(ctx, external_count, p.inactive);
    Matrix<double> active_sources(ctx, virtual_count, p.active);
    chart_zero(ctx, core_sources.view());
    chart_zero(ctx, active_sources.view());
    const double alpha = 1.0, beta = 0.0;
    for (const auto& block : p.locals) {
        if (!block.size)
            continue;
        const bool core = block.orbital < p.inactive;
        Mut source = core ? core_sources.view().column(block.orbital)
                          : active_sources.view().column(block.orbital - p.inactive);
        check_cublas(cublasDgemv_64(ctx.blas(),
                                    CUBLAS_OP_T,
                                    block.size,
                                    block.size,
                                    &alpha,
                                    block.ah_frame_inverse.data(),
                                    block.size,
                                    covector.data + block.offset,
                                    1,
                                    &beta,
                                    source.data,
                                    1),
                     "cublasDgemv chart full-frame source projection");
    }
    auto solve_group = [&](Const source, Const inverse, Const diagonal, int first_row) {
        Matrix<double> solved(ctx, source.rows, source.cols);
        if (!source.rows || !source.cols)
            return solved;
        auto frame = chart_product(ctx, source, inverse, Operation::none, Operation::transpose);
        chart_ah_divide<<<grid(core::checked_index<std::int64_t>(frame.rows() * frame.cols())),
                          kThreads,
                          0,
                          ctx.stream()>>>(frame.data(),
                                          diagonal,
                                          core::checked_index<std::int64_t>(frame.rows()),
                                          core::checked_index<std::int64_t>(frame.cols()),
                                          first_row,
                                          eigenvalue);
        check_cuda(cudaGetLastError(), "CUDA chart full-frame analytic denominator");
        solved = chart_product(ctx, frame.view(), inverse);
        return solved;
    };
    const Const core_diagonal{p.ah_rotation_diagonal.data(), p.basis, p.basis, p.basis};
    Matrix<double> solved_core = solve_group(
        core_sources.view(), p.ah_core_source_inverse.view(), core_diagonal, p.inactive);
    const Const active_diagonal{
        p.ah_rotation_diagonal.data() + p.inactive * p.basis, p.basis, p.basis, p.basis};
    Matrix<double> solved_active = solve_group(active_sources.view(),
                                               p.ah_active_source_inverse.view(),
                                               active_diagonal,
                                               p.inactive + p.active);
    for (const auto& block : p.locals) {
        if (!block.size)
            continue;
        const bool core = block.orbital < p.inactive;
        const Const source = core ? solved_core.view().column(block.orbital)
                                  : solved_active.view().column(block.orbital - p.inactive);
        check_cublas(cublasDgemv_64(ctx.blas(),
                                    CUBLAS_OP_N,
                                    block.size,
                                    block.size,
                                    &alpha,
                                    block.ah_frame_inverse.data(),
                                    block.size,
                                    source.data,
                                    1,
                                    &beta,
                                    direction.data + block.offset,
                                    1),
                     "cublasDgemv chart full-frame analytic inverse");
    }
}

} // namespace xmvb::cuda
