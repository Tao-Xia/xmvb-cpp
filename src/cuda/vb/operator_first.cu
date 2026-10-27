#include <cstdint>
#include "cuda/vb/operator_first.hpp"
#include "cuda/vb/action.hpp"
#include "cuda/vb/panel.hpp"
#include "cuda/ci/similarity.hpp"
#include "cuda/linalg/blas.hpp"
#include "cuda/ci/raw.hpp"
#include "mcscf/ci/raw_action.hpp"
#include "mcscf/ci/sigma.hpp"
#include <algorithm>
#include <array>
#include <numeric>
#include <optional>
#include <cusparse.h>
#include <nvtx3/nvtx3.hpp>
namespace xmvb::cuda {
namespace {
using View = linalg::ConstMatrixView;
using Out = linalg::MatrixView;
using Mask = std::uint64_t;
unsigned grid(std::int64_t n) {
    return static_cast<unsigned>(std::min<std::int64_t>(65535, (n + 255) / 256));
}
void zero(Context& c, Out out) {
    if (out.data)
        check_cuda(cudaMemset2DAsync(out.data,
                                     out.ld * sizeof(double),
                                     0,
                                     out.rows * sizeof(double),
                                     out.cols,
                                     c.stream()),
                   "clear source panel");
}
void copy(Context& c, Out out, View in) {
    check_cuda(cudaMemcpy2DAsync(out.data,
                                 out.ld * sizeof(double),
                                 in.data,
                                 in.ld * sizeof(double),
                                 out.rows * sizeof(double),
                                 out.cols,
                                 cudaMemcpyDeviceToDevice,
                                 c.stream()),
               "copy source panel");
}
void add(Context& c, Out out, View in, double scale = 1) {
    const double one = 1;
    check_cublas(cublasDgeam_64(c.blas(),
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                out.rows,
                                out.cols,
                                &one,
                                out.data,
                                out.ld,
                                &scale,
                                in.data,
                                in.ld,
                                out.data,
                                out.ld),
                 "add source panel");
}
void scale(Context& c, Out out, double scale) {
    check_cublas(cublasDscal_64(c.blas(), out.rows * out.cols, &scale, out.data, 1),
                 "scale source result");
}
Matrix<double> product(
    Context& c, View a, View b, Operation oa = Operation::none, Operation ob = Operation::none) {
    Matrix<double> out(
        c, oa == Operation::none ? a.rows : a.cols, ob == Operation::none ? b.cols : b.rows);
    gemm(c, out.view(), a, b, 1, 0, oa, ob);
    return out;
}
__global__ void upper_only(Out x) {
    const std::int64_t r = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r < x.rows * x.cols && r % x.rows > r / x.rows)
        x.data[r % x.rows + x.ld * (r / x.rows)] = 0;
}
__global__ void identity_columns(const std::int64_t* ids, int count, Out out) {
    for (auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < out.rows * count;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x)
        out.data[i % out.rows + out.ld * (i / out.rows)] =
            i % out.rows == ids[i / out.rows] ? 1 : 0;
}
__global__ void source_columns(const std::int64_t* alpha,
                               const std::int64_t* beta,
                               std::int64_t groups,
                               View x,
                               const std::int64_t* ids,
                               int count,
                               Out out) {
    for (auto p = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; p < groups;
         p += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        int low = 0, high = count;
        while (low < high) {
            const int m = (low + high) / 2;
            if (ids[m] < beta[p])
                low = m + 1;
            else
                high = m;
        }
        if (low < count && ids[low] == beta[p])
            for (int v = 0; v < x.cols; ++v)
                out.data[alpha[p] + out.ld * (low * x.cols + v)] = x.data[p + x.ld * v];
    }
}
__global__ void gather(const std::int64_t* alpha,
                       const std::int64_t* beta,
                       std::int64_t groups,
                       View z,
                       View source,
                       View right,
                       View dz,
                       View ds,
                       View dr,
                       Out h,
                       Out m,
                       bool tangent,
                       double energy) {
    for (auto p = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; p < groups;
         p += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t a = alpha[p], b = beta[p];
        double hv = 0, mv = 0;
        for (int j = 0; j < right.cols; ++j) {
            if (tangent) {
                mv += ds.data[a + ds.ld * j] * right.data[b + right.ld * j];
                hv += dz.data[a + dz.ld * j] * right.data[b + right.ld * j];
                if (dr.data) {
                    mv += source.data[a + source.ld * j] * dr.data[b + dr.ld * j];
                    hv += z.data[a + z.ld * j] * dr.data[b + dr.ld * j];
                }
            } else {
                mv += source.data[a + source.ld * j] * right.data[b + right.ld * j];
                if (h.data)
                    hv += z.data[a + z.ld * j] * right.data[b + right.ld * j];
            }
        }
        if (h.data)
            h.data[p] += hv + energy * mv;
        if (m.data)
            m.data[p] += mv;
    }
}
void sparse_check(cusparseStatus_t s) {
    if (s != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error(cusparseGetErrorString(s));
}
/** Thin cuSPARSE owner for the sparse original source, never transformed products. */
/** Project every RHS while sharing the same beta metric panel. */
__global__ void gather_block(const std::int64_t* alpha,
                             const std::int64_t* beta,
                             std::int64_t groups,
                             int rhs,
                             View h,
                             View m,
                             View right,
                             const std::int64_t* m_ids,
                             const std::int64_t* target_positions,
                             int first,
                             Out h_products,
                             Out m_products) {
    for (auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < groups * rhs;
         index += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const auto p = index % groups;
        const int v = index / groups, a = alpha[p], b = beta[p];
        double hv = 0, mv = 0;
        if (h_products.data)
            for (int j = 0; j < right.cols; ++j)
                hv += h.data[a + h.ld * (j * rhs + v)] * right.data[b + right.ld * j];
        if (m_products.data)
            for (int j = 0; j < m.cols / rhs; ++j) {
                const std::int64_t position =
                    target_positions ? target_positions[m_ids[j]] - first : j;
                mv += m.data[a + m.ld * (j * rhs + v)] * right.data[b + right.ld * position];
            }
        if (h_products.data)
            h_products.data[p + h_products.ld * v] += hv;
        if (m_products.data)
            m_products.data[p + m_products.ld * v] += mv;
    }
}
struct SourceMatrix {
    Context& c;
    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t matrix = nullptr;
    Matrix<double> row;
    SourceMatrix(Context& context,
                 int na,
                 int nb,
                 std::int64_t nnz,
                 const std::int64_t* offsets,
                 const std::int64_t* columns,
                 const double* values)
        : c(context) {
        sparse_check(cusparseCreate(&handle));
        sparse_check(cusparseSetStream(handle, c.stream()));
        // A one-dimensional alpha sector reduces X to one beta-spin vector.
        // Scatter its support once; both orientations are then ordinary GEMM.
        if (na == 1) {
            row = Matrix<double>(c, nb, 1);
            zero(c, row.view());
            cusparseSpVecDescr_t source;
            cusparseDnVecDescr_t target;
            sparse_check(cusparseCreateSpVec(&source,
                                             nb,
                                             nnz,
                                             const_cast<std::int64_t*>(columns),
                                             const_cast<double*>(values),
                                             CUSPARSE_INDEX_64I,
                                             CUSPARSE_INDEX_BASE_ZERO,
                                             CUDA_R_64F));
            sparse_check(cusparseCreateDnVec(&target, nb, row.data(), CUDA_R_64F));
            sparse_check(cusparseScatter(handle, source, target));
            cusparseDestroySpVec(source);
            cusparseDestroyDnVec(target);
            return;
        }
        sparse_check(cusparseCreateCsr(&matrix,
                                       na,
                                       nb,
                                       nnz,
                                       const_cast<std::int64_t*>(offsets),
                                       const_cast<std::int64_t*>(columns),
                                       const_cast<double*>(values),
                                       CUSPARSE_INDEX_64I,
                                       CUSPARSE_INDEX_64I,
                                       CUSPARSE_INDEX_BASE_ZERO,
                                       CUDA_R_64F));
    }
    ~SourceMatrix() {
        if (matrix)
            cusparseDestroySpMat(matrix);
        cusparseDestroy(handle);
    }
    void multiply(View x, Out y, bool transpose = false, double scale = 1) const {
        if (row.data()) {
            gemm(c, y, row.view(), x, scale, 0, transpose ? Operation::none : Operation::transpose);
            return;
        }
        cusparseDnMatDescr_t a, b;
        sparse_check(cusparseCreateDnMat(
            &a, x.rows, x.cols, x.ld, const_cast<double*>(x.data), CUDA_R_64F, CUSPARSE_ORDER_COL));
        sparse_check(
            cusparseCreateDnMat(&b, y.rows, y.cols, y.ld, y.data, CUDA_R_64F, CUSPARSE_ORDER_COL));
        std::size_t size = 0;
        const double zero = 0;
        const auto op = transpose ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
        sparse_check(cusparseSpMM_bufferSize(handle,
                                             op,
                                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             &scale,
                                             matrix,
                                             a,
                                             &zero,
                                             b,
                                             CUDA_R_64F,
                                             CUSPARSE_SPMM_ALG_DEFAULT,
                                             &size));
        Buffer<std::byte> scratch(c, size);
        sparse_check(cusparseSpMM(handle,
                                  op,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &scale,
                                  matrix,
                                  a,
                                  &zero,
                                  b,
                                  CUDA_R_64F,
                                  CUSPARSE_SPMM_ALG_DEFAULT,
                                  scratch.data()));
        cusparseDestroyDnMat(a);
        cusparseDestroyDnMat(b);
    }
};
struct Panel {
    Matrix<double> value;
    std::vector<Matrix<double>> directions;
    Panel(Context& c, int rows, int cols, int lanes) : value(c, rows, cols) {
        zero(c, value.view());
        for (int l = 0; l < lanes; ++l) {
            directions.emplace_back(c, rows, cols);
            zero(c, directions.back().view());
        }
    }
};
void gram(Panel& p,
          const ExteriorTransform& t,
          const ExteriorDirection* d,
          bool frame_only = false) {
    if (d) {
        t.apply_directional_left(*d, p.value.view(), p.directions[0].view());
        t.apply_directional_left(*d, p.value.view(), p.directions[0].view(), true);
    } else {
        t.apply_left(p.value.view());
        t.apply_left(p.value.view(), true);
    }
    if (!frame_only)
        for (std::size_t l = d ? 1 : 0; l < p.directions.size(); ++l) {
            t.apply_left(p.directions[l].view());
            t.apply_left(p.directions[l].view(), true);
        }
}
} // namespace
struct OperatorFirstAction::Impl {
    struct Topology {
        int n, alpha, beta, multiprocessors;
        bool covalent = false;
        std::int64_t groups;
        std::vector<std::int64_t> targets, source_targets;
        Buffer<std::int64_t> alpha_ids, columns, source_columns, target_positions, sorted_ids;
        Buffer<std::int64_t> beta_ids;
        Buffer<Mask> alpha_masks, beta_masks, sorted_masks;
        Buffer<std::int64_t> offsets;
        Topology(Context& c, const StructureAction& action)
            : n(action.alpha_transform().n_orbitals()), alpha(action.n_alpha_strings()),
              beta(action.n_beta_strings()), groups(action.n_product_groups()),
              alpha_ids(c, groups), beta_ids(c, groups), sorted_ids(c, alpha),
              alpha_masks(c, alpha), beta_masks(c, beta), sorted_masks(c, alpha),
              offsets(c, alpha + 1) {
            int device;
            check_cuda(cudaGetDevice(&device), "source topology CUDA device");
            check_cuda(
                cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device),
                "source topology multiprocessor count");
            const auto& v = action.orthogonal_view();
            if (!std::is_sorted(v.products.begin(), v.products.end()) ||
                !v.alpha.strings().complete() || !v.beta.strings().complete())
                throw std::logic_error(
                    "operator-first source requires ordered complete single-spin indexing");
            mcscf::RawCiPlan plan(
                v.sigma.alpha_connections(), v.sigma.beta_connections_view(), v.products);
            covalent = plan.complete_covalent_support();
            targets.assign(plan.beta_columns().begin(), plan.beta_columns().end());
            source_targets.assign(plan.source_columns().begin(), plan.source_columns().end());
            source_columns = Buffer<std::int64_t>(c, source_targets.size());
            target_positions = Buffer<std::int64_t>(c, beta);
            std::vector<std::int64_t> positions(beta, -1);
            for (std::size_t j = 0; j < targets.size(); ++j)
                positions[targets[j]] = j;
            columns = Buffer<std::int64_t>(c, targets.size());
            std::vector<std::int64_t> as(groups), ids(alpha);
            std::vector<std::int64_t> bs(groups);
            std::vector<std::int64_t> os(alpha + 1);
            for (std::size_t p = 0; p < v.products.size(); ++p) {
                as[p] = v.products[p] / beta;
                bs[p] = v.products[p] % beta;
                ++os[as[p] + 1];
            }
            std::partial_sum(os.begin(), os.end(), os.begin());
            std::iota(ids.begin(), ids.end(), 0);
            const auto& masks = v.alpha.occupations();
            std::sort(ids.begin(), ids.end(), [&](int a, int b) { return masks[a] < masks[b]; });
            std::vector<Mask> sorted(alpha);
            for (std::int64_t a = 0; a < alpha; ++a)
                sorted[a] = masks[ids[a]];
            const auto upload = [&](auto& dst, const auto& src) {
                c.copy_to_device(dst.data(), src.data(), src.size() * sizeof(src[0]));
            };
            upload(alpha_ids, as);
            upload(beta_ids, bs);
            upload(offsets, os);
            upload(sorted_ids, ids);
            upload(sorted_masks, sorted);
            upload(alpha_masks, v.alpha.occupations());
            upload(beta_masks, v.beta.occupations());
            upload(columns, targets);
            upload(source_columns, source_targets);
            upload(target_positions, positions);
        }
    };
    Context& c;
    const StructureAction& action;
    std::shared_ptr<const Topology> topology;
    mutable SimilarityHamiltonian h;
    Matrix<double> physical_one, physical_pair, upper;
    mutable bool dirty = true;
    void ensure() const {
        if (dirty) {
            h.update(physical_one.view(), physical_pair.view(), upper.view());
            dirty = false;
        }
    }
    Matrix<double> base_one, base_two, base_moment;
    double energy = 0;
    bool accepted = false;
    Impl(Context& ctx, const StructureAction& a, std::shared_ptr<const Topology> t = {})
        : c(ctx), action(a), topology(t ? std::move(t) : std::make_shared<Topology>(ctx, a)),
          h(ctx, topology->n), physical_one(ctx, topology->n, topology->n),
          physical_pair(
              ctx, topology->n * (topology->n + 1) / 2, topology->n * (topology->n + 1) / 2),
          upper(ctx, topology->n, topology->n), base_one(ctx, topology->n, topology->n),
          base_two(ctx, topology->n * topology->n, topology->n * topology->n),
          base_moment(ctx, topology->n, topology->n) {}
    void raw(View x,
             const std::int64_t* columns,
             int count,
             Out z,
             View seed = {},
             Out ah = {},
             Out ag = {},
             RawCiBatch batch = {}) const {
        const auto& t = *topology;
        const auto& v = action.orthogonal_view();
        const RawCiTopology view{t.alpha_ids.data(),
                                 t.beta_ids.data(),
                                 t.groups,
                                 t.alpha_masks.data(),
                                 t.beta_masks.data(),
                                 t.sorted_masks.data(),
                                 t.sorted_ids.data(),
                                 t.alpha,
                                 t.n,
                                 v.sigma.alpha_connections().n_occupied(),
                                 v.sigma.beta_connections_view().n_occupied(),
                                 t.multiprocessors,
                                 t.covalent};
        apply_raw_ci(c, view, x, columns, count, h.one(), h.two(), z, seed, ah, ag, batch);
    }
    int width(std::size_t budget, int lanes) const {
        const auto& t = *topology;
        const std::size_t intrinsic = sizeof(double) * t.groups * (6 + 3 * lanes);
        // Integral partials plus their reduction vector fit within one extra
        // image panel. This is part of the same caller-owned workspace budget.
        const std::size_t per =
            sizeof(double) * ((t.alpha + t.beta) * (8 + 5 * lanes) + t.alpha * (1 + lanes));
        if (budget <= intrinsic || budget - intrinsic < per)
            throw std::length_error("operator-first GPU budget cannot hold one panel");
        return static_cast<int>(std::min<std::size_t>(
            t.targets.size(),
            std::min<std::size_t>(product_panel_width(), (budget - intrinsic) / per)));
    }
    Matrix<double> expand(View coefficients) const {
        Matrix<double> x(c, topology->groups, coefficients.cols);
        action.source_map(false, coefficients, x.view());
        return x;
    }
};
OperatorFirstAction::OperatorFirstAction(Context& c, const StructureAction& action)
    : impl_(std::make_unique<Impl>(c, action)) {}
OperatorFirstAction::OperatorFirstAction(std::unique_ptr<Impl> p) : impl_(std::move(p)) {}
OperatorFirstAction::~OperatorFirstAction() = default;
std::unique_ptr<OperatorFirstAction> OperatorFirstAction::clone() const {
    return std::unique_ptr<OperatorFirstAction>(
        new OperatorFirstAction(std::make_unique<Impl>(impl_->c, impl_->action, impl_->topology)));
}
void OperatorFirstAction::update(View one, View pair, View upper) {
    copy(impl_->c, impl_->physical_one.view(), one);
    copy(impl_->c, impl_->physical_pair.view(), pair);
    copy(impl_->c, impl_->upper.view(), upper);
    impl_->dirty = true;
    impl_->accepted = false;
    impl_->energy = 0;
}
void OperatorFirstAction::update_frame(View upper) {
    copy(impl_->c, impl_->upper.view(), upper);
    impl_->dirty = true;
    impl_->accepted = false;
}
void OperatorFirstAction::accept(double energy) {
    if (!impl_->accepted)
        throw std::logic_error("operator-first point has no accepted cotangents");
    impl_->energy = energy;
}
void OperatorFirstAction::apply(View coefficients, Out hout, Out mout, std::size_t budget) const {
    const nvtx3::scoped_range range{"vb.operator_first.action"};
    auto& p = *impl_;
    auto& c = p.c;
    const auto& t = *p.topology;
    const int rhs = coefficients.cols, families = bool(hout.data) + bool(mout.data);
    if (!families)
        return;
    const auto beta_bytes = sizeof(double) * t.beta;
    const auto rhs_bytes = sizeof(double) * (t.groups * (families + 1) + t.alpha * families);
    const auto batch = budget > beta_bytes ? (budget - beta_bytes) / rhs_bytes : 0;
    if (!batch)
        throw std::length_error("operator-first block budget cannot hold one RHS");
    if (batch < static_cast<std::size_t>(rhs)) {
        for (std::int64_t first = 0; first < rhs; first += batch) {
            const std::int64_t count = std::min<std::size_t>(batch, rhs - first);
            apply(coefficients.columns(first, count),
                  hout.data ? hout.columns(first, count) : Out{},
                  mout.data ? mout.columns(first, count) : Out{},
                  budget);
        }
        return;
    }
    const auto& targets = hout.data ? t.targets : t.source_targets;
    const auto device_targets = hout.data ? t.columns.data() : t.source_columns.data();
    const auto intrinsic = sizeof(double) * t.groups * rhs * (families + 1);
    const auto per_column = sizeof(double) * (t.alpha * rhs * families + t.beta);
    const int width = std::min<std::size_t>(
        targets.size(),
        std::min<std::size_t>(
            std::max<std::int64_t>(1, t.beta - 1),
            std::min<std::size_t>(std::max<std::int64_t>(1, product_panel_width() / rhs),
                                  (budget - intrinsic) / per_column)));
    if (!width)
        throw std::length_error("operator-first block budget cannot hold one panel");
    if (hout.data)
        p.ensure();
    auto x = p.expand(coefficients);
    Matrix<double> hp(c, t.groups, hout.data ? rhs : 0), mp(c, t.groups, mout.data ? rhs : 0);
    zero(c, hp.view());
    zero(c, mp.view());
    // H/M and every RHS share a single alpha traversal. Beta is generated once.
    Matrix<double> left(c, t.alpha, width * rhs * families), right(c, t.beta, width);
    for (std::size_t first = 0; first < targets.size(); first += width) {
        const std::int64_t count = std::min<std::size_t>(width, targets.size() - first);
        const auto ids = device_targets + first;
        auto m_begin = t.source_targets.end(), m_end = m_begin;
        if (mout.data) {
            m_begin =
                std::lower_bound(t.source_targets.begin(), t.source_targets.end(), targets[first]);
            m_end = std::upper_bound(m_begin, t.source_targets.end(), targets[first + count - 1]);
        }
        const int m_count = m_end - m_begin, h_count = hout.data ? count : 0;
        const auto m_ids = t.source_columns.data() + (m_begin - t.source_targets.begin());
        auto active = left.view().columns(0, rhs * (h_count + m_count));
        const auto h = hout.data ? active.columns(0, h_count * rhs) : Out{};
        const auto m = mout.data ? active.columns(h_count * rhs, m_count * rhs) : Out{};
        if (hout.data)
            p.raw(x.view(), ids, count, h);
        if (m_count) {
            zero(c, m);
            source_columns<<<grid(t.groups), 256, 0, c.stream()>>>(
                t.alpha_ids.data(), t.beta_ids.data(), t.groups, x.view(), m_ids, m_count, m);
        }
        auto r = right.view().columns(0, count);
        identity_columns<<<grid(static_cast<std::int64_t>(t.beta) * count), 256, 0, c.stream()>>>(
            ids, count, r);
        p.action.alpha_transform().apply_left(active);
        p.action.alpha_transform().apply_left(active, true);
        p.action.beta_transform().apply_left(r);
        p.action.beta_transform().apply_left(r, true);
        gather_block<<<grid(t.groups * rhs), 256, 0, c.stream()>>>(
            t.alpha_ids.data(),
            t.beta_ids.data(),
            t.groups,
            rhs,
            h,
            m,
            r,
            m_ids,
            hout.data ? t.target_positions.data() : nullptr,
            first,
            hp.view(),
            mp.view());
        check_cuda(cudaGetLastError(), "operator-first block projection");
    }
    if (hout.data)
        p.action.source_map(true, hp.view(), hout);
    if (mout.data)
        p.action.source_map(true, mp.view(), mout);
}

ProductJet OperatorFirstAction::jet(const DirectCiSigma& sigma,
                                    View coefficients,
                                    View response,
                                    View dr,
                                    View dh,
                                    View dg,
                                    std::size_t budget,
                                    ProductJet* ci,
                                    View overlap_image) {
    const nvtx3::scoped_range range{"vb.operator_first.jet"};
    auto& p = *impl_;
    auto& c = p.c;
    const auto& t = *p.topology;
    p.ensure();
    const bool tangent = response.data || dr.data || dh.data;
    const int lanes = tangent ? (ci ? 2 : 1) : 0;
    if (tangent && !p.accepted)
        throw std::logic_error("operator-first tangent requires accepted cotangents");
    if (ci && !response.data)
        throw std::invalid_argument("independent CI lane requires its coefficients");
    const int width = p.width(budget, lanes), n = t.n, np = n * (n + 1) / 2, ns = p.action.size();
    auto make = [&] {
        ProductJet j{Matrix<double>(c, ns, 1),
                     Matrix<double>(c, ns, 1),
                     Matrix<double>(c, n, n),
                     Matrix<double>(c, np, np),
                     Matrix<double>(c, n, n)};
        zero(c, j.generator.view());
        return j;
    };
    ProductJet result = make();
    std::array<ProductJet, 2> outputs;
    if (tangent)
        for (int l = 0; l < lanes; ++l)
            outputs[l] = make();
    auto x = p.expand(coefficients);
    Matrix<double> dx;
    if (response.data)
        dx = p.expand(response);
    SourceMatrix xs(c, t.alpha, t.beta, t.groups, t.offsets.data(), t.beta_ids.data(), x.data());
    std::unique_ptr<SourceMatrix> ds;
    if (response.data)
        ds = std::make_unique<SourceMatrix>(
            c, t.alpha, t.beta, t.groups, t.offsets.data(), t.beta_ids.data(), dx.data());
    std::optional<SimilarityHamiltonian::Direction> direction;
    if (dr.data)
        direction = p.h.direction(dh, dg, dr);
    std::optional<ExteriorDirection> da, db;
    if (dr.data) {
        da = p.action.alpha_transform().direction(p.h.upper(), dr);
        db = p.action.beta_transform().direction(p.h.upper(), dr);
    }
    std::vector<Matrix<double>> one, two, moment, hprod, mprod;
    for (int l = 0; l < (tangent ? lanes : 1); ++l) {
        one.emplace_back(c, n, n);
        two.emplace_back(c, n * n, n * n);
        moment.emplace_back(c, n, n);
        hprod.emplace_back(c, t.groups, 1);
        mprod.emplace_back(c, t.groups, 1);
        zero(c, one.back().view());
        zero(c, two.back().view());
        zero(c, moment.back().view());
        zero(c, hprod.back().view());
        zero(c, mprod.back().view());
    }
    const double energy = tangent ? p.energy : 0;
    for (std::size_t first = 0; first < t.targets.size(); first += width) {
        const std::int64_t count = std::min<std::size_t>(width, t.targets.size() - first);
        const auto ids = t.columns.data() + first;
        Panel z(c, t.alpha, count, lanes), source(c, t.alpha, count, lanes),
            right(c, t.beta, count, lanes), left(c, t.alpha, count, lanes),
            beta(c, t.beta, count, lanes);
        source_columns<<<grid(t.groups), 256, 0, c.stream()>>>(t.alpha_ids.data(),
                                                               t.beta_ids.data(),
                                                               t.groups,
                                                               x.view(),
                                                               ids,
                                                               count,
                                                               source.value.view());
        if (response.data)
            for (int l = 0; l < lanes; ++l)
                source_columns<<<grid(t.groups), 256, 0, c.stream()>>>(t.alpha_ids.data(),
                                                                       t.beta_ids.data(),
                                                                       t.groups,
                                                                       dx.view(),
                                                                       ids,
                                                                       count,
                                                                       source.directions[l].view());
        identity_columns<<<grid(static_cast<std::int64_t>(t.beta) * count), 256, 0, c.stream()>>>(
            ids, count, right.value.view());
        gram(right, p.action.beta_transform(), db ? &*db : nullptr, true);
        xs.multiply(right.value.view(), left.value.view());
        for (int l = 0; l < lanes; ++l) {
            if (ds)
                ds->multiply(right.value.view(), left.directions[l].view());
            if (l == 0 && dr.data) {
                Matrix<double> temp(c, t.alpha, count);
                xs.multiply(right.directions[0].view(), temp.view());
                add(c, left.directions[0].view(), temp.view());
            }
        }
        {
            Panel seed(c, t.alpha, count, lanes);
            copy(c, seed.value.view(), left.value.view());
            for (int l = 0; l < lanes; ++l)
                copy(c, seed.directions[l].view(), left.directions[l].view());
            gram(seed, p.action.alpha_transform(), da ? &*da : nullptr);
            RawCiBatch batch{};
            batch.count = lanes;
            for (int l = 0; l < lanes; ++l)
                batch.lanes[l] = {l == 0 && direction ? direction->one.view() : View{},
                                  l == 0 && direction ? direction->two.view() : View{},
                                  dx.view(),
                                  seed.directions[l].view(),
                                  z.directions[l].view(),
                                  one[l].view(),
                                  two[l].view()};
            p.raw(x.view(),
                  ids,
                  count,
                  z.value.view(),
                  seed.value.view(),
                  tangent ? Out{} : one[0].view(),
                  tangent ? Out{} : two[0].view(),
                  batch);
        }
        add(c, z.value.view(), source.value.view(), -energy);
        for (int l = 0; l < lanes; ++l)
            add(c, z.directions[l].view(), source.directions[l].view(), -energy);
        gram(z, p.action.alpha_transform(), da ? &*da : nullptr);
        if (tangent || !overlap_image.data)
            gram(source, p.action.alpha_transform(), da ? &*da : nullptr);
        xs.multiply(z.value.view(), beta.value.view(), true);
        for (int l = 0; l < lanes; ++l) {
            if (ds)
                ds->multiply(z.value.view(), beta.directions[l].view(), true);
            Matrix<double> temp(c, t.beta, count);
            xs.multiply(z.directions[l].view(), temp.view(), true);
            add(c, beta.directions[l].view(), temp.view());
        }
        std::vector<CiSpinGeneratorTerm> at, bt;
        std::vector<Out> moments;
        for (auto& m : moment)
            moments.push_back(m.view());
        if (!tangent) {
            gather<<<grid(t.groups), 256, 0, c.stream()>>>(t.alpha_ids.data(),
                                                           t.beta_ids.data(),
                                                           t.groups,
                                                           z.value.view(),
                                                           source.value.view(),
                                                           right.value.view(),
                                                           {},
                                                           {},
                                                           {},
                                                           hprod[0].view(),
                                                           overlap_image.data ? Out{}
                                                                              : mprod[0].view(),
                                                           false,
                                                           0);
            at.push_back({z.value.view(), left.value.view(), 0});
            bt.push_back({right.value.view(), beta.value.view(), 0});
        }
        for (int l = 0; l < lanes; ++l) {
            gather<<<grid(t.groups), 256, 0, c.stream()>>>(
                t.alpha_ids.data(),
                t.beta_ids.data(),
                t.groups,
                z.value.view(),
                source.value.view(),
                right.value.view(),
                z.directions[l].view(),
                source.directions[l].view(),
                l == 0 && dr.data ? right.directions[l].view() : View{},
                hprod[l].view(),
                mprod[l].view(),
                true,
                energy);
            at.push_back({z.directions[l].view(), left.value.view(), static_cast<std::size_t>(l)});
            at.push_back({z.value.view(), left.directions[l].view(), static_cast<std::size_t>(l)});
            bt.push_back(
                {right.value.view(), beta.directions[l].view(), static_cast<std::size_t>(l)});
            if (l == 0 && dr.data)
                bt.push_back(
                    {right.directions[0].view(), beta.value.view(), static_cast<std::size_t>(l)});
        }
        sigma.spin_generator_adjoints(false, at, moments, true);
        sigma.spin_generator_adjoints(true, bt, moments, true);
    }
    if (!tangent) {
        copy(c, p.base_one.view(), one[0].view());
        copy(c, p.base_two.view(), two[0].view());
        copy(c, p.base_moment.view(), moment[0].view());
        p.accepted = true;
        p.energy = 0;
    }
    Matrix<double> residual;
    if (tangent) {
        residual = Matrix<double>(c, n, n);
        copy(c, residual.view(), p.base_moment.view());
        add(c, residual.view(), p.base_one.view(), -energy);
    }
    for (int l = 0; l < (tangent ? lanes : 1); ++l) {
        auto& out = tangent ? outputs[l] : result;
        p.action.source_map(true, hprod[l].view(), out.hamiltonian.view());
        if (!tangent && overlap_image.data)
            copy(c, out.overlap.view(), overlap_image);
        else
            p.action.source_map(true, mprod[l].view(), out.overlap.view());
        if (l == 0 && direction)
            p.h.pullback(p.base_one.view(),
                         p.base_two.view(),
                         out.one.view(),
                         out.pair.view(),
                         &*direction,
                         dr,
                         one[l].view(),
                         two[l].view());
        else
            p.h.pullback(one[l].view(), two[l].view(), out.one.view(), out.pair.view());
        if (tangent) {
            scale(c, out.one.view(), 0.5);
            scale(c, out.pair.view(), 0.5);
        }
        auto first = product(c, p.h.inverse(), moment[l].view(), Operation::transpose);
        gemm(c,
             out.generator.view(),
             first.view(),
             p.h.upper(),
             2,
             0,
             Operation::none,
             Operation::transpose);
        if (l == 0 && direction) {
            first = product(c, direction->inverse.view(), residual.view(), Operation::transpose);
            gemm(c,
                 out.generator.view(),
                 first.view(),
                 p.h.upper(),
                 2,
                 1,
                 Operation::none,
                 Operation::transpose);
            first = product(c, p.h.inverse(), residual.view(), Operation::transpose);
            gemm(c,
                 out.generator.view(),
                 first.view(),
                 dr,
                 2,
                 1,
                 Operation::none,
                 Operation::transpose);
        }
        upper_only<<<grid(n * n), 256, 0, c.stream()>>>(out.generator.view());
    }
    if (tangent)
        result = std::move(outputs[0]);
    if (ci)
        *ci = std::move(outputs[1]);
    check_cuda(cudaGetLastError(), "operator-first jet final projection");
    return result;
}
} // namespace xmvb::cuda
