#include <cstdint>
#include "cuda/ci/similarity.hpp"
#include "cuda/linalg/blas.hpp"
namespace xmvb::cuda {
namespace {
using View = linalg::ConstMatrixView;
using Out = linalg::MatrixView;
unsigned grid(std::int64_t n) {
    return static_cast<unsigned>((n + 255) / 256);
}
__device__ int pair(int a, int b) {
    if (a < b) {
        int t = a;
        a = b;
        b = t;
    }
    return a * (a + 1) / 2 + b;
}
__global__ void identity(Out a) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < a.rows * a.cols)
        a.data[i % a.rows + a.ld * (i / a.rows)] = i % a.rows == i / a.rows ? 1 : 0;
}
__global__ void unpack(View p, Out full, int n) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= full.rows * full.cols)
        return;
    const std::int64_t a = index % full.rows, b = index / full.rows;
    full.data[a + full.ld * b] = p.data[pair(a / n, a % n) + p.ld * pair(b / n, b % n)];
}
__global__ void changes(View r, View inv, View dr, View di, Out c, int n) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= c.rows * c.cols)
        return;
    const std::int64_t a = index % c.rows, b = index / c.rows, i = a / n, j = a % n, p = b / n,
                       q = b % n;
    c.data[a + c.ld * b] = dr.data ? di.data[i + di.ld * p] * r.data[q + r.ld * j] +
                                         inv.data[i + inv.ld * p] * dr.data[q + dr.ld * j]
                                   : inv.data[i + inv.ld * p] * r.data[q + r.ld * j];
}
__global__ void symmetrize(View full, Out out, bool packed, int n) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= full.rows * full.cols)
        return;
    const std::int64_t a = index % full.rows, b = index / full.rows;
    if (packed) {
        const double v = 0.5 * (full.data[a + full.ld * b] + full.data[b + full.ld * a]);
        atomicAdd(out.data + pair(a / n, a % n) + out.ld * pair(b / n, b % n), v);
    } else
        out.data[a + out.ld * b] = 0.5 * (full.data[a + full.ld * b] + full.data[b + full.ld * a]);
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
               "canonical copy");
}
Matrix<double> product(
    Context& c, View a, View b, Operation oa = Operation::none, Operation ob = Operation::none) {
    Matrix<double> out(
        c, oa == Operation::none ? a.rows : a.cols, ob == Operation::none ? b.cols : b.rows);
    gemm(c, out.view(), a, b, 1, 0, oa, ob);
    return out;
}
void add_product(Context& c,
                 Out out,
                 View a,
                 View b,
                 Operation oa = Operation::none,
                 Operation ob = Operation::none) {
    gemm(c, out, a, b, 1, 1, oa, ob);
}
void physical(Context& c, View h, View g, Out one, Out pair, int n) {
    symmetrize<<<grid(n * n), 256, 0, c.stream()>>>(h, one, false, n);
    check_cuda(cudaMemsetAsync(pair.data, 0, pair.rows * pair.cols * sizeof(double), c.stream()),
               "clear canonical pair pullback");
    symmetrize<<<grid(g.rows * g.cols), 256, 0, c.stream()>>>(g, pair, true, n);
    check_cuda(cudaGetLastError(), "canonical physical pullback");
}
} // namespace
SimilarityHamiltonian::SimilarityHamiltonian(Context& c, int n)
    : context_(c), n_(n), one_(c, n, n), two_(c, n * n, n * n), upper_(c, n, n), inverse_(c, n, n),
      physical_one_(c, n, n), physical_two_(c, n * n, n * n), change_(c, n * n, n * n) {}
void SimilarityHamiltonian::update(View h, View g, View r) {
    auto& c = context_;
    copy(c, upper_.view(), r);
    copy(c, physical_one_.view(), h);
    identity<<<grid(n_ * n_), 256, 0, c.stream()>>>(inverse_.view());
    const double one = 1;
    check_cublas(cublasDtrsm_64(c.blas(),
                                CUBLAS_SIDE_LEFT,
                                CUBLAS_FILL_MODE_UPPER,
                                CUBLAS_OP_N,
                                CUBLAS_DIAG_NON_UNIT,
                                n_,
                                n_,
                                &one,
                                r.data,
                                r.ld,
                                inverse_.data(),
                                n_),
                 "canonical upper inverse");
    unpack<<<grid(two_.rows() * two_.cols()), 256, 0, c.stream()>>>(g, physical_two_.view(), n_);
    changes<<<grid(two_.rows() * two_.cols()), 256, 0, c.stream()>>>(
        r, inverse_.view(), {}, {}, change_.view(), n_);
    auto first = product(c, inverse_.view(), h);
    gemm(c, one_.view(), first.view(), r);
    auto second = product(c, change_.view(), physical_two_.view());
    gemm(
        c, two_.view(), second.view(), change_.view(), 1, 0, Operation::none, Operation::transpose);
    check_cuda(cudaGetLastError(), "canonical integral similarity");
}
SimilarityHamiltonian::Direction SimilarityHamiltonian::direction(View dh, View dg, View dr) const {
    auto& c = context_;
    Direction d{Matrix<double>(c, n_, n_),
                Matrix<double>(c, n_ * n_, n_ * n_),
                Matrix<double>(c, n_, n_),
                Matrix<double>(c, n_ * n_, n_ * n_)};
    auto first = product(c, inverse_.view(), dr);
    gemm(c, d.inverse.view(), first.view(), inverse_.view(), -1);
    changes<<<grid(two_.rows() * two_.cols()), 256, 0, c.stream()>>>(
        upper_.view(), inverse_.view(), dr, d.inverse.view(), d.change.view(), n_);
    first = product(c, d.inverse.view(), physical_one_.view());
    gemm(c, d.one.view(), first.view(), upper_.view());
    first = product(c, inverse_.view(), dh);
    add_product(c, d.one.view(), first.view(), upper_.view());
    first = product(c, inverse_.view(), physical_one_.view());
    add_product(c, d.one.view(), first.view(), dr);
    Matrix<double> delta(c, n_ * n_, n_ * n_);
    unpack<<<grid(delta.rows() * delta.cols()), 256, 0, c.stream()>>>(dg, delta.view(), n_);
    first = product(c, d.change.view(), physical_two_.view());
    gemm(
        c, d.two.view(), first.view(), change_.view(), 1, 0, Operation::none, Operation::transpose);
    first = product(c, change_.view(), delta.view());
    add_product(
        c, d.two.view(), first.view(), change_.view(), Operation::none, Operation::transpose);
    first = product(c, change_.view(), physical_two_.view());
    add_product(
        c, d.two.view(), first.view(), d.change.view(), Operation::none, Operation::transpose);
    return d;
}
void SimilarityHamiltonian::pullback(
    View bh, View bg, Out ph, Out pg, const Direction* d, View dr, View dbh, View dbg) const {
    auto& c = context_;
    Matrix<double> h(c, n_, n_), g(c, n_ * n_, n_ * n_);
    if (!d) {
        auto first = product(c, inverse_.view(), bh, Operation::transpose);
        gemm(c, h.view(), first.view(), upper_.view(), 1, 0, Operation::none, Operation::transpose);
        first = product(c, change_.view(), bg, Operation::transpose);
        gemm(c, g.view(), first.view(), change_.view());
    } else {
        auto first = product(c, d->inverse.view(), bh, Operation::transpose);
        gemm(c, h.view(), first.view(), upper_.view(), 1, 0, Operation::none, Operation::transpose);
        first = product(c, inverse_.view(), dbh, Operation::transpose);
        add_product(
            c, h.view(), first.view(), upper_.view(), Operation::none, Operation::transpose);
        first = product(c, inverse_.view(), bh, Operation::transpose);
        add_product(c, h.view(), first.view(), dr, Operation::none, Operation::transpose);
        first = product(c, d->change.view(), bg, Operation::transpose);
        gemm(c, g.view(), first.view(), change_.view());
        first = product(c, change_.view(), dbg, Operation::transpose);
        add_product(c, g.view(), first.view(), change_.view());
        first = product(c, change_.view(), bg, Operation::transpose);
        add_product(c, g.view(), first.view(), d->change.view());
    }
    physical(c, h.view(), g.view(), ph, pg, n_);
}
} // namespace xmvb::cuda
