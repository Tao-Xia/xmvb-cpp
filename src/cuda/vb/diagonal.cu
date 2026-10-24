#include "cuda/vb/diagonal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include "cuda/integrals/exact.hpp"
#include "cuda/linalg/blas.hpp"

namespace xmvb::cuda {
namespace {

unsigned int grid(std::int64_t count) {
    constexpr std::int64_t threads = 256;
    const auto blocks = (count + threads - 1) / threads;
    if (blocks <= 0 || blocks > std::numeric_limits<unsigned int>::max())
        throw std::overflow_error("CUDA Hessian diagonal grid overflow");
    return static_cast<unsigned int>(blocks);
}

__device__ int pair(int a, int b) {
    const int high = max(a, b), low = min(a, b);
    return high * (high + 1) / 2 + low;
}

__device__ void decode_pair(std::int64_t index, int& a, int& b) {
    std::int64_t high =
        static_cast<std::int64_t>((sqrt(8.0 * static_cast<double>(index) + 1.0) - 1.0) * 0.5);
    while (high * (high + 1) / 2 > index)
        --high;
    while ((high + 1) * (high + 2) / 2 <= index)
        ++high;
    a = static_cast<int>(high);
    b = static_cast<int>(index - high * (high + 1) / 2);
}

__device__ double dm2(linalg::ConstMatrixView pair_adjoint, int a, int b, int c, int d) {
    const int p = pair(a, b), q = pair(c, d);
    const int multiplicity = (a == b ? 1 : 2) * (c == d ? 1 : 2);
    return (pair_adjoint.data[p + pair_adjoint.ld * q] +
            pair_adjoint.data[q + pair_adjoint.ld * p]) /
           multiplicity;
}

__global__ void difference(linalg::ConstMatrixView effective,
                           linalg::ConstMatrixView core,
                           linalg::MatrixView output,
                           std::int64_t elements) {
    const auto k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= elements)
        return;
    const auto row = k % output.rows, col = k / output.rows;
    output.data[row + output.ld * col] =
        effective.data[row + effective.ld * col] - core.data[row + core.ld * col];
}

__global__ void core_density(linalg::ConstMatrixView mo,
                             linalg::MatrixView density,
                             int core,
                             std::int64_t elements) {
    const auto k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= elements)
        return;
    const int n = static_cast<int>(density.rows);
    const int mu = static_cast<int>(k % n), nu = static_cast<int>(k / n);
    density.data[k] = mo.data[mu + mo.ld * core] * mo.data[nu + mo.ld * core];
}

__global__ void active_weight(linalg::ConstMatrixView dm1,
                              linalg::ConstMatrixView pair_adjoint,
                              linalg::MatrixView weight,
                              int active,
                              int channel,
                              bool two_rdm) {
    const auto k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= static_cast<std::int64_t>(active) * active)
        return;
    const int w = static_cast<int>(k % active), z = static_cast<int>(k / active);
    if (!two_rdm) {
        weight.data[k] = 0.5 * ((w == channel ? dm1.data[channel + dm1.ld * z] : 0.0) +
                                (z == channel ? dm1.data[channel + dm1.ld * w] : 0.0));
    } else {
        weight.data[k] =
            0.5 *
            (dm2(pair_adjoint, channel, w, z, channel) + dm2(pair_adjoint, w, channel, z, channel) +
             dm2(pair_adjoint, channel, z, w, channel) + dm2(pair_adjoint, z, channel, w, channel));
    }
}

__global__ void projected_mo_diagonal(linalg::ConstMatrixView mo,
                                      linalg::ConstMatrixView transformed,
                                      linalg::MatrixView output,
                                      int column,
                                      int nbf) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nbf)
        return;
    double value = 0.0;
    for (int mu = 0; mu < nbf; ++mu)
        value += mo.data[mu + mo.ld * p] * transformed.data[mu + transformed.ld * p];
    output.data[p + output.ld * column] = value;
}

__global__ void add_active_coulomb(linalg::ConstMatrixView j_active,
                                   linalg::ConstMatrixView weight,
                                   linalg::MatrixView response,
                                   int column,
                                   int nbf,
                                   int active) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nbf)
        return;
    double coulomb = 0.0;
    for (int a = 0; a < active; ++a)
        for (int b = 0; b < active; ++b)
            coulomb += j_active.data[p + j_active.ld * pair(a, b)] * weight.data[a + weight.ld * b];
    response.data[p + response.ld * column] += 2.0 * coulomb;
}

__global__ void potential_and_gradient(linalg::ConstMatrixView hmo,
                                       linalg::ConstMatrixView vmo,
                                       linalg::ConstMatrixView j_active,
                                       linalg::ConstMatrixView dm1,
                                       linalg::ConstMatrixView pair_adjoint,
                                       linalg::ConstMatrixView active_pair,
                                       linalg::ConstMatrixView density_images,
                                       double* potential,
                                       double* gradient,
                                       int nbf,
                                       int ncore,
                                       int active) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nbf)
        return;
    double pot = vmo.data[p + vmo.ld * p];
    for (int a = 0; a < active; ++a)
        pot += 0.5 * density_images.data[p + density_images.ld * a];
    potential[p] = pot;
    if (p < ncore) {
        gradient[p] = 2.0 * (hmo.data[p + hmo.ld * p] + pot);
        return;
    }
    double value = 0.0;
    const int a = p - ncore;
    if (a < active)
        for (int u = 0; u < active; ++u) {
            value += (hmo.data[p + hmo.ld * (ncore + u)] + vmo.data[p + vmo.ld * (ncore + u)]) *
                     dm1.data[u + dm1.ld * a];
            for (int w = 0; w < active; ++w)
                for (int z = 0; z < active; ++z)
                    value += active_pair.data[pair(a, u) + active_pair.ld * pair(w, z)] *
                             dm2(pair_adjoint, w, z, u, a);
        }
    gradient[p] = value;
}

__global__ void rdm_contractions(linalg::ConstMatrixView j_active,
                                 linalg::ConstMatrixView dm1,
                                 linalg::ConstMatrixView pair_adjoint,
                                 linalg::ConstMatrixView density_images,
                                 linalg::MatrixView one,
                                 linalg::MatrixView two,
                                 int nbf,
                                 int ncore,
                                 int active) {
    const auto k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= static_cast<std::int64_t>(nbf) * active)
        return;
    const int p = static_cast<int>(k % nbf), a = static_cast<int>(k / nbf);
    double coulomb = 0.0;
    for (int w = 0; w < active; ++w)
        for (int z = 0; z < active; ++z) {
            const double weight =
                dm2(pair_adjoint, w, z, a, a) +
                2.0 * (dm2(pair_adjoint, a, w, z, a) + dm2(pair_adjoint, w, a, z, a));
            coulomb += j_active.data[p + j_active.ld * pair(w, z)] * weight;
        }
    two.data[p + two.ld * a] = coulomb - density_images.data[p + density_images.ld * (active + a)];
    if (p < ncore + active) {
        double one_coulomb = 0.0;
        for (int v = 0; v < active; ++v)
            one_coulomb += j_active.data[p + j_active.ld * pair(a, v)] * dm1.data[a + dm1.ld * v];
        one.data[p + one.ld * a] =
            10.0 * one_coulomb - 6.0 * density_images.data[p + density_images.ld * a];
    }
}

__device__ double occupation(linalg::ConstMatrixView dm1, int ncore, int active, int p, int q) {
    if (p < ncore || q < ncore)
        return p == q && p < ncore ? 2.0 : 0.0;
    const int a = p - ncore, b = q - ncore;
    return a < active && b < active ? dm1.data[a + dm1.ld * b] : 0.0;
}

__device__ double raw_diagonal(int p,
                               int q,
                               linalg::ConstMatrixView hmo,
                               linalg::ConstMatrixView vmo,
                               linalg::ConstMatrixView dm1,
                               linalg::ConstMatrixView jpc,
                               linalg::ConstMatrixView core_response,
                               linalg::ConstMatrixView one,
                               linalg::ConstMatrixView two,
                               const double* potential,
                               const double* gradient,
                               int ncore,
                               int active) {
    const int nocc = ncore + active;
    double value = hmo.data[p + hmo.ld * p] * occupation(dm1, ncore, active, q, q) -
                   hmo.data[p + hmo.ld * q] * occupation(dm1, ncore, active, p, q) +
                   hmo.data[q + hmo.ld * q] * occupation(dm1, ncore, active, p, p) -
                   hmo.data[q + hmo.ld * p] * occupation(dm1, ncore, active, q, p) - gradient[p] -
                   gradient[q];
    if (p == q)
        value += 2.0 * gradient[p];
    if (q < ncore)
        value += 2.0 * potential[p];
    if (p < ncore)
        value += 2.0 * potential[q];
    if (p == q && p < ncore)
        value -= 4.0 * potential[p];
    if (q >= ncore && q < nocc)
        value += vmo.data[p + vmo.ld * p] * dm1.data[(q - ncore) * (dm1.ld + 1)];
    if (p >= ncore && p < nocc)
        value += vmo.data[q + vmo.ld * q] * dm1.data[(p - ncore) * (dm1.ld + 1)];
    if (p >= ncore && p < nocc && q >= ncore && q < nocc) {
        const int a = p - ncore, b = q - ncore;
        value -= vmo.data[p + vmo.ld * q] * dm1.data[a + dm1.ld * b];
        value -= vmo.data[q + vmo.ld * p] * dm1.data[b + dm1.ld * a];
    }
    if (p >= ncore && q < ncore)
        value +=
            10.0 * jpc.data[p + jpc.ld * q] - 6.0 * core_response.data[p + core_response.ld * q];
    if (q >= ncore && p < ncore)
        value +=
            10.0 * jpc.data[q + jpc.ld * p] - 6.0 * core_response.data[q + core_response.ld * p];
    if (p < nocc && q >= ncore && q < nocc)
        value -= one.data[p + one.ld * (q - ncore)];
    if (q < nocc && p >= ncore && p < nocc)
        value -= one.data[q + one.ld * (p - ncore)];
    if (p >= ncore && p < nocc)
        value += two.data[q + two.ld * (p - ncore)];
    if (q >= ncore && q < nocc)
        value += two.data[p + two.ld * (q - ncore)];
    return value;
}

__global__ void assemble(linalg::ConstMatrixView hmo,
                         linalg::ConstMatrixView vmo,
                         linalg::ConstMatrixView dm1,
                         linalg::ConstMatrixView jpc,
                         linalg::ConstMatrixView core_response,
                         linalg::ConstMatrixView one,
                         linalg::ConstMatrixView two,
                         const double* potential,
                         const double* gradient,
                         linalg::MatrixView output,
                         int nbf,
                         int ncore,
                         int active) {
    const auto k = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= static_cast<std::int64_t>(nbf) * nbf)
        return;
    const int p = static_cast<int>(k % nbf), q = static_cast<int>(k / nbf);
    output.data[k] =
        raw_diagonal(
            p, q, hmo, vmo, dm1, jpc, core_response, one, two, potential, gradient, ncore, active) +
        raw_diagonal(
            q, p, hmo, vmo, dm1, jpc, core_response, one, two, potential, gradient, ncore, active);
}

std::int64_t pair_tile_rows(Context& context, std::int64_t pair_rows, int channels) {
    const std::size_t available = context.available_device_bytes();
    const std::size_t row_bytes = static_cast<std::size_t>(channels) * sizeof(double);
    if (row_bytes == 0 || available / 2 < row_bytes)
        throw std::runtime_error("CUDA Hessian diagonal cannot fit one AO-pair row");
    return std::min<std::int64_t>(pair_rows, static_cast<std::int64_t>(available / 2 / row_bytes));
}

void validate(linalg::ConstMatrixView mo,
              linalg::ConstMatrixView core,
              linalg::ConstMatrixView effective,
              linalg::ConstMatrixView dm1,
              linalg::ConstMatrixView pair_adjoint,
              linalg::ConstMatrixView pair_kernel,
              int ncore,
              const ExactAoPairOperator& integrals) {
    const auto n = mo.rows, active = dm1.rows;
    const auto pairs = active * (active + 1) / 2;
    if (!mo.data || n <= 0 || mo.cols != n || mo.ld < n || integrals.basis_functions() != n ||
        ncore < 0 || active <= 0 || ncore + active > n || !core.data || core.rows != n ||
        core.cols != n || core.ld < n || !effective.data || effective.rows != n ||
        effective.cols != n || effective.ld < n || !dm1.data || dm1.cols != active ||
        dm1.ld < active || !pair_adjoint.data || pair_adjoint.rows != pairs ||
        pair_adjoint.cols != pairs || pair_adjoint.ld < pairs || !pair_kernel.data ||
        pair_kernel.rows != pairs || pair_kernel.cols != pairs || pair_kernel.ld < pairs)
        throw std::invalid_argument("invalid CUDA analytic Hessian diagonal input");
}

} // namespace

Matrix<double> build_analytic_rotation_diagonal(Context& context,
                                                const ExactAoPairOperator& integrals,
                                                linalg::ConstMatrixView mo,
                                                linalg::ConstMatrixView core,
                                                linalg::ConstMatrixView effective,
                                                linalg::ConstMatrixView dm1,
                                                linalg::ConstMatrixView pair_adjoint,
                                                linalg::ConstMatrixView active_pair,
                                                int ncore,
                                                ConstPairPanel accepted_products,
                                                ConstPairPanel exchange_products,
                                                Buffer<double>* density_projection,
                                                CoreDiagonalView prepared_core) {
    validate(mo, core, effective, dm1, pair_adjoint, active_pair, ncore, integrals);
    const int nbf = static_cast<int>(mo.rows), active = static_cast<int>(dm1.rows);
    const int pairs = active * (active + 1) / 2, nocc = ncore + active;
    if (accepted_products.data && (accepted_products.rows != integrals.pair_count() ||
                                   accepted_products.cols != pairs || accepted_products.ld < pairs))
        throw std::invalid_argument("CUDA diagonal accepted product dimensions");
    Matrix<double> hmo(context, nbf, nbf), vmo(context, nbf, nbf);
    Matrix<double> scratch(context, nbf, nbf);
    gemm(context, scratch.view(), core, mo);
    gemm(context, hmo.view(), mo, scratch.view(), 1.0, 0.0, Operation::transpose);
    difference<<<grid(static_cast<std::int64_t>(nbf) * nbf), 256, 0, context.stream()>>>(
        effective, core, scratch.view(), static_cast<std::int64_t>(nbf) * nbf);
    check_cuda(cudaGetLastError(), "CUDA Hessian core potential difference");
    Matrix<double> projected(context, nbf, nbf);
    gemm(context, projected.view(), scratch.view(), mo);
    gemm(context, vmo.view(), mo, projected.view(), 1.0, 0.0, Operation::transpose);

    Matrix<double> jactive(context, nbf, pairs);
    Matrix<double> active_images(context, nbf, 2 * active);
    // Every weight is symmetric, including when the supplied RDM adjoints are
    // not. Hence its exchange image is symmetric and the packed AO-pair
    // projection is exact; no full AO response or full MO transform is needed.
    Matrix<double> active_weights(context, active, 2 * active * active);
    for (int channel = 0; channel < 2 * active; ++channel) {
        active_weight<<<grid(static_cast<std::int64_t>(active) * active),
                        256,
                        0,
                        context.stream()>>>(dm1,
                                            pair_adjoint,
                                            active_weights.view().columns(channel * active, active),
                                            active,
                                            channel % active,
                                            channel >= active);
        check_cuda(cudaGetLastError(), "CUDA Hessian active density weight");
    }
    {
        const nvtx3::scoped_range range{"diagonal.pair_projection"};
        const auto rows = pair_tile_rows(context,
                                         integrals.pair_count(),
                                         nbf + (exchange_products.data ? 2 * active : 0) +
                                             (!accepted_products.data ? pairs : 0));
        Matrix<double> features(context, nbf, rows), exchange_image;
        if (exchange_products.data)
            exchange_image = Matrix<double>(context, rows, 2 * active);
        Buffer<double> products;
        if (!accepted_products.data)
            products = Buffer<double>(context, static_cast<std::size_t>(rows) * pairs);
        for (std::int64_t begin = 0; begin < integrals.pair_count(); begin += rows) {
            const auto count = std::min(rows, integrals.pair_count() - begin);
            ConstPairPanel tile{accepted_products.data
                                    ? accepted_products.data + begin * accepted_products.ld
                                    : products.data(),
                                count,
                                pairs,
                                accepted_products.data ? accepted_products.ld : pairs};
            if (!accepted_products.data)
                integrals.generated_rows(
                    mo.columns(ncore, active), {}, begin, {products.data(), count, pairs, pairs});
            const auto b = features.view().columns(0, count);
            integrals.density_features(mo, begin, {b.data, count, nbf, b.ld});
            gemm(context,
                 jactive.view(),
                 b,
                 {tile.data, pairs, count, tile.ld},
                 1.0,
                 begin == 0 ? 0.0 : 1.0,
                 Operation::none,
                 Operation::transpose);
            if (exchange_products.data) {
                const auto n2 = static_cast<std::int64_t>(active) * active;
                const linalg::MatrixView image{exchange_image.data(), count, 2 * active, rows};
                // E is row-major in AO pairs. The active pair index is transposed
                // relative to column-major W, which is immaterial for symmetric W.
                gemm(context,
                     image,
                     {exchange_products.data + begin * exchange_products.ld,
                      n2,
                      count,
                      exchange_products.ld},
                     {active_weights.data(), n2, 2 * active, n2},
                     -1.0,
                     0.0,
                     Operation::transpose);
                gemm(context, active_images.view(), b, image, 1.0, begin == 0 ? 0.0 : 1.0);
            }
        }
    }

    Matrix<double> density, response;
    Matrix<double> active_left;
    if (!exchange_products.data)
        active_left = Matrix<double>(context, nbf, active);
    Matrix<double> core_response, jpc;
    auto core_j = prepared_core.coulomb, core_r = prepared_core.response;
    if (core_j.data &&
        (core_j.rows != nbf || core_j.cols != ncore || core_j.ld < nbf || !core_r.data ||
         core_r.rows != nbf || core_r.cols != ncore || core_r.ld < nbf))
        throw std::invalid_argument("CUDA prepared core diagonal dimensions");
    const auto active_mo = mo.columns(ncore, active);
    bool core_projected = core_j.data;
    if (!core_projected) {
        core_response = Matrix<double>(context, nbf, ncore);
        jpc = Matrix<double>(context, nbf, ncore);
        core_j = jpc.view();
        core_r = core_response.view();
        core_projected = integrals.core_diagonal(
            mo, ncore, jpc.view(), core_response.view(), density_projection);
    }
    if (!core_projected && ncore) {
        density = Matrix<double>(context, nbf, nbf);
        response = Matrix<double>(context, nbf, nbf);
    }
    for (int c = 0; !core_projected && c < ncore; ++c) {
        core_density<<<grid(static_cast<std::int64_t>(nbf) * nbf), 256, 0, context.stream()>>>(
            mo, density.view(), c, static_cast<std::int64_t>(nbf) * nbf);
        check_cuda(cudaGetLastError(), "CUDA Hessian core density");
        integrals.contract_jk(density.view(), response.view(), 2.0, -1.0, scratch.view());
        gemm(context, projected.view(), response.view(), mo);
        projected_mo_diagonal<<<grid(nbf), 256, 0, context.stream()>>>(
            mo, projected.view(), core_response.view(), c, nbf);
        check_cuda(cudaGetLastError(), "CUDA Hessian core response diagonal");
        gemm(context, projected.view(), scratch.view(), mo);
        projected_mo_diagonal<<<grid(nbf), 256, 0, context.stream()>>>(
            mo, projected.view(), jpc.view(), c, nbf);
        check_cuda(cudaGetLastError(), "CUDA Hessian core Coulomb diagonal");
    }
    int width = 2 * active;
    Matrix<double> densities, responses;
    if (!exchange_products.data) {
        const auto available = context.available_device_bytes();
        // Only the streamed representation needs full density slabs. The
        // retained-image route above has already completed every K projection.
        const auto channel_bytes = 4 * static_cast<std::size_t>(nbf) * nbf * sizeof(double);
        width = static_cast<int>(std::min<std::size_t>(2 * active, available / channel_bytes));
        if (width < 1)
            throw std::runtime_error("CUDA diagonal budget cannot hold one density batch");
        densities = Matrix<double>(context, nbf, nbf * width);
        responses = Matrix<double>(context, nbf, nbf * width);
    }
    for (int begin = 0; begin < 2 * active; begin += width) {
        const int count = std::min(width, 2 * active - begin);
        for (int local = 0; local < count; ++local) {
            if (!exchange_products.data) {
                const int channel = begin + local;
                const auto source = active_weights.view().columns(channel * active, active);
                gemm(context, active_left.view(), active_mo, source);
                gemm(context,
                     densities.view().columns(local * nbf, nbf),
                     active_left.view(),
                     active_mo,
                     1.0,
                     0.0,
                     Operation::none,
                     Operation::transpose);
            }
        }
        if (!exchange_products.data)
            integrals.contract_jk_batch(densities.view().columns(0, nbf * count),
                                        responses.view().columns(0, nbf * count),
                                        0.0,
                                        -1.0);
        for (int local = 0; local < count; ++local) {
            const int channel = begin + local;
            if (!exchange_products.data) {
                gemm(context, projected.view(), responses.view().columns(local * nbf, nbf), mo);
                projected_mo_diagonal<<<grid(nbf), 256, 0, context.stream()>>>(
                    mo, projected.view(), active_images.view(), channel, nbf);
                check_cuda(cudaGetLastError(), "CUDA Hessian active exchange diagonal");
            }
            add_active_coulomb<<<grid(nbf), 256, 0, context.stream()>>>(
                jactive.view(),
                active_weights.view().columns(channel * active, active),
                active_images.view(),
                channel,
                nbf,
                active);
            check_cuda(cudaGetLastError(), "CUDA Hessian active Coulomb diagonal");
        }
    }
    Buffer<double> potential(context, nbf), gradient(context, nbf);
    Matrix<double> one(context, nocc, active), two(context, nbf, active);
    potential_and_gradient<<<grid(nbf), 256, 0, context.stream()>>>(hmo.view(),
                                                                    vmo.view(),
                                                                    jactive.view(),
                                                                    dm1,
                                                                    pair_adjoint,
                                                                    active_pair,
                                                                    active_images.view(),
                                                                    potential.data(),
                                                                    gradient.data(),
                                                                    nbf,
                                                                    ncore,
                                                                    active);
    check_cuda(cudaGetLastError(), "CUDA Hessian potential and gradient");
    rdm_contractions<<<grid(static_cast<std::int64_t>(nbf) * active), 256, 0, context.stream()>>>(
        jactive.view(),
        dm1,
        pair_adjoint,
        active_images.view(),
        one.view(),
        two.view(),
        nbf,
        ncore,
        active);
    check_cuda(cudaGetLastError(), "CUDA Hessian RDM contractions");
    Matrix<double> output(context, nbf, nbf);
    assemble<<<grid(static_cast<std::int64_t>(nbf) * nbf), 256, 0, context.stream()>>>(
        hmo.view(),
        vmo.view(),
        dm1,
        core_j,
        core_r,
        one.view(),
        two.view(),
        potential.data(),
        gradient.data(),
        output.view(),
        nbf,
        ncore,
        active);
    check_cuda(cudaGetLastError(), "CUDA analytic rotation Hessian diagonal");
    return output;
}

} // namespace xmvb::cuda
