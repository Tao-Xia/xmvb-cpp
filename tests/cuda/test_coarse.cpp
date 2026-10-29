#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <Eigen/Core>

#include "cuda/linalg/blas.hpp"
#include "cuda/optimization/coarse.hpp"
#include "mcscf/optimization/preconditioners/coarse.hpp"

namespace {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using DeviceMatrix = xmvb::cuda::Matrix<double>;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

DeviceMatrix upload(xmvb::cuda::Context& context, const Matrix& host) {
    DeviceMatrix result(context, host.rows(), host.cols());
    xmvb::cuda::copy_to_device(context, result.view(), host);
    return result;
}

DeviceMatrix upload(xmvb::cuda::Context& context, const Vector& host) {
    DeviceMatrix result(context, host.size(), 1);
    context.copy_to_device(result.data(), host.data(), host.size() * sizeof(double));
    return result;
}

Vector download(xmvb::cuda::Context& context, xmvb::linalg::ConstMatrixView device) {
    Matrix host(device.rows, device.cols);
    xmvb::cuda::copy_to_host(context, host, device);
    return host.col(0);
}

void observe(xmvb::cuda::Context& context,
             xmvb::cuda::DeviceCoarsePreconditioner& device,
             xmvb::mcscf::CoarsePreconditioner& host,
             const Vector& v,
             const Vector& hv,
             const Vector& mv) {
    auto dv = upload(context, v), dh = upload(context, hv), dm = upload(context, mv);
    device.observe(dv.view(), dh.view(), dm.view());
    host.observe(v, hv, mv);
    require(device.size() == host.size(), "CUDA coarse admission differs from CPU");
}

void check_balancing() {
    constexpr int n = 5;
    Matrix h = Matrix::Zero(n, n), m = Matrix::Identity(n, n), base = Matrix::Zero(n, n);
    h.diagonal() << -2.0, 3.0, 5.0, 4.0, 6.0;
    h(0, 3) = h(3, 0) = 0.7;
    h(1, 4) = h(4, 1) = 0.5;
    h(2, 3) = h(3, 2) = 0.2;
    h(2, 0) = h(0, 2) = 0.1;
    m.diagonal() << 1.0, 1.25, 1.5, 1.75, 2.0;
    m(0, 1) = m(1, 0) = 0.1;
    base.diagonal() << 0.2, 0.35, 0.5, 0.65, 0.8;
    Matrix sampled = Matrix::Zero(n, 4);
    sampled.col(0) = Vector::Unit(n, 0);
    sampled.col(1) = Vector::Unit(n, 1);
    sampled.col(2) = Vector::Unit(n, 3);
    sampled.col(3) = Vector::Unit(n, 4);

    xmvb::cuda::Context context(32ULL * 1024 * 1024);
    xmvb::cuda::DeviceCoarsePreconditioner gpu(context, n, 4, 1e-14);
    xmvb::mcscf::CoarsePreconditioner cpu(n, 4, 1e-14);
    auto d_base = upload(context, base);
    observe(context, gpu, cpu, sampled.col(0), h * sampled.col(0), m * sampled.col(0));
    observe(context, gpu, cpu, sampled.col(0), h * sampled.col(0), m * sampled.col(0));
    require(gpu.size() == 1, "dependent coarse sample was admitted");
    const Vector mixed = sampled.col(0) + sampled.col(1);
    observe(context, gpu, cpu, mixed, h * mixed, m * mixed);
    for (int i = 2; i < 4; ++i)
        observe(context, gpu, cpu, sampled.col(i), h * sampled.col(i), m * sampled.col(i));
    require(gpu.size() == 4, "independent coarse samples were rejected");

    for (double eigenvalue : {0.0, -0.2}) {
        Matrix actual(n, n);
        int calls = 0;
        for (int i = 0; i < n; ++i) {
            const Vector rhs = Vector::Unit(n, i);
            auto d_rhs = upload(context, rhs);
            DeviceMatrix d_out(context, n, 1);
            gpu.apply(
                d_rhs.view(),
                eigenvalue,
                [&](xmvb::linalg::ConstMatrixView input, xmvb::linalg::MatrixView output) {
                    ++calls;
                    xmvb::cuda::gemm(context, output, d_base.view(), input);
                },
                d_out.view());
            actual.col(i) = download(context, d_out.view());
            const Vector expected = cpu.apply(
                rhs, eigenvalue, [&](const Vector& residual) -> Vector { return base * residual; });
            require((actual.col(i) - expected).norm() < 3e-11,
                    "CUDA coarse balancing differs from CPU formula");
        }
        require(calls == n, "CUDA coarse inverse did not call the base inverse once");
        require((actual - actual.transpose()).norm() < 3e-11,
                "CUDA coarse balancing inverse lost symmetry");
    }
}

/** @brief CPU/GPU retain finite physical curvatures despite different coordinate units. */
void check_physical_scaling() {
    constexpr int n = 3;
    const Eigen::Vector3d units(1.0, 1e-10, 1e3);
    const Matrix t = units.asDiagonal(), inv = units.cwiseInverse().asDiagonal();
    const Matrix h = t * Eigen::Vector3d(2.0, -3.0, 7.0).asDiagonal() * t;
    const Matrix m = t * t, base = .4 * inv * inv;
    xmvb::cuda::Context context(32ULL * 1024 * 1024);
    xmvb::cuda::DeviceCoarsePreconditioner gpu(context, n, 2, 1e-14);
    xmvb::mcscf::CoarsePreconditioner cpu(n, 2, 1e-14);
    for (int i = 0; i < 2; ++i) {
        const Vector v = Vector::Unit(n, i);
        observe(context, gpu, cpu, v, h * v, m * v);
    }
    auto d_base = upload(context, base);
    for (double shift : {0.0, -.2}) {
        Matrix actual(n, n);
        for (int i = 0; i < n; ++i) {
            const Vector rhs = Vector::Unit(n, i);
            auto d_rhs = upload(context, rhs);
            DeviceMatrix out(context, n, 1);
            gpu.apply(
                d_rhs.view(),
                shift,
                [&](xmvb::linalg::ConstMatrixView r, xmvb::linalg::MatrixView z) {
                    xmvb::cuda::gemm(context, z, d_base.view(), r);
                },
                out.view());
            actual.col(i) = download(context, out.view());
        }
        const Matrix expected =
            Eigen::Vector3d(1.0 / (2.0 - shift), 1.0 / (-3.0 - shift), .4).asDiagonal();
        require((t * actual * t - expected).norm() < 3e-11,
                "CUDA coarse numerical rank depends on orbital/CI coordinate units");
    }
}

void check_shift_pole() {
    constexpr int n = 3;
    const Matrix h = Eigen::Vector3d(-2.0, 3.0, 7.0).asDiagonal();
    const Matrix base = Eigen::Vector3d(.7, .4, .2).asDiagonal();
    xmvb::cuda::Context context(32ULL * 1024 * 1024);
    auto d_base = upload(context, base);
    for (double angle : {0.0, .37}) {
        xmvb::cuda::DeviceCoarsePreconditioner gpu(context, n, 2, 1e-14);
        xmvb::mcscf::CoarsePreconditioner cpu(n, 2, 1e-14);
        Vector first(n), second(n);
        first << std::cos(angle), std::sin(angle), 0;
        second << -std::sin(angle), std::cos(angle), 0;
        observe(context, gpu, cpu, first, h * first, first);
        observe(context, gpu, cpu, second, h * second, second);
        for (int i = 0; i < n; ++i) {
            const Vector rhs = Vector::Unit(n, i);
            auto d_rhs = upload(context, rhs);
            DeviceMatrix out(context, n, 1);
            int calls = 0;
            gpu.apply(
                d_rhs.view(),
                -2.0,
                [&](xmvb::linalg::ConstMatrixView r, xmvb::linalg::MatrixView z) {
                    ++calls;
                    xmvb::cuda::gemm(context, z, d_base.view(), r);
                },
                out.view());
            const Vector expected =
                cpu.apply(rhs, -2.0, [&](const Vector& r) -> Vector { return base * r; });
            require(calls == 1 && (download(context, out.view()) - expected).norm() < 3e-11,
                    "CUDA coarse shift pole changed the range or lost the base action");
        }
    }
}

void check_certification() {
    constexpr int n = 4096;
    Vector first(n), second = Vector::Constant(n, 1.0 / 64.0);
    for (int i = 0; i < n; ++i)
        first[i] = (i % 2 ? -1.0 : 1.0) / 64.0;
    for (double skew : {1e-12, 1e-5}) {
        xmvb::cuda::Context context(16ULL * 1024 * 1024);
        xmvb::cuda::DeviceCoarsePreconditioner gpu(context, n, 2, 1e-14);
        xmvb::mcscf::CoarsePreconditioner cpu(n, 2, 1e-14);
        observe(context, gpu, cpu, first, 2 * first, first);
        observe(context, gpu, cpu, second, 3 * second + skew * first, second);
        require(gpu.size() == (skew > 1e-6 ? 1 : 2),
                "CUDA coarse image symmetry certification changed");
        if (skew > 1e-6) {
            observe(context, gpu, cpu, second, 3 * second, second);
            require(gpu.size() == 2, "CUDA coarse rejected valid recovery sample");
        }
        auto d_rhs = upload(context, second);
        DeviceMatrix d_out(context, n, 1);
        gpu.apply(
            d_rhs.view(),
            0.0,
            [&](xmvb::linalg::ConstMatrixView input, xmvb::linalg::MatrixView output) {
                xmvb::cuda::check_cuda(cudaMemcpyAsync(output.data,
                                                       input.data,
                                                       n * sizeof(double),
                                                       cudaMemcpyDeviceToDevice,
                                                       context.stream()),
                                       "CUDA coarse identity inverse copy");
            },
            d_out.view());
        const Vector expected =
            cpu.apply(second, 0.0, [](const Vector& residual) -> Vector { return residual; });
        require((download(context, d_out.view()) - expected).norm() < 3e-10,
                "CUDA coarse rejection damaged retained inverse");
    }
}

void check_scalar_batches() {
    constexpr int n = 513;
    xmvb::cuda::Context context(16ULL * 1024 * 1024);
    xmvb::cuda::DeviceCoarsePreconditioner gpu(context, n, 3, 1e-14, 1e-12);
    xmvb::mcscf::CoarsePreconditioner cpu(n, 3, 1e-14, 1e-12);
    const Vector zero = Vector::Zero(n);
    observe(context, gpu, cpu, zero, zero, zero);
    require(gpu.size() == 0, "zero sample was admitted by scalar batching");
    Vector first(n), second(n);
    for (int i = 0; i < n; ++i) {
        first[i] = (i % 2 ? -1.0 : 1.0) * (1.0 + i * 0.001);
        second[i] = std::sin(i * 0.31);
    }
    observe(context, gpu, cpu, first, 2 * first, first);
    // Both passes must consume the already updated residual, not a batched CGS
    // projection of the original input. The norm pair must retain BLAS scaling.
    const Vector mixed = 1e3 * first + second;
    observe(context, gpu, cpu, mixed, 2 * mixed, mixed);
    require(gpu.size() == 2, "independent mixed sample was lost");
    observe(context, gpu, cpu, 1e150 * first, 2e150 * first, 1e150 * first);
    require(gpu.size() == 2, "large dependent sample was admitted");
    for (int slot = 0; slot < 3; ++slot) {
        Vector v = second, h = 2 * second, m = second;
        (slot == 0   ? v
         : slot == 1 ? h
                     : m)[17] = slot == 1 ? std::numeric_limits<double>::quiet_NaN()
                                          : std::numeric_limits<double>::infinity();
        auto dv = upload(context, v), dh = upload(context, h), dm = upload(context, m);
        bool rejected = false;
        try {
            gpu.observe(dv.view(), dh.view(), dm.view());
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "nonfinite sample escaped batched reductions");
        require(gpu.size() == 2, "nonfinite sample damaged retained rank");
    }
    cublasPointerMode_t mode;
    xmvb::cuda::check_cublas(cublasGetPointerMode(context.blas(), &mode),
                             "check coarse scalar mode");
    require(mode == CUBLAS_POINTER_MODE_HOST, "coarse leaked device scalar mode");
}

void check_split_support() {
    constexpr int no = 3, n = 7, capacity = 5;
    Matrix h = Matrix::Zero(n, n), m = Matrix::Zero(n, n), base = Matrix::Zero(n, n);
    h.diagonal() << 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0;
    h(0, 3) = h(3, 0) = 0.37;
    h(1, 4) = h(4, 1) = -0.23;
    h(2, 6) = h(6, 2) = 0.19;
    h(0, 1) = h(1, 0) = 0.11;
    h(3, 4) = h(4, 3) = -0.13;
    m.diagonal() << 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6;
    m(0, 1) = m(1, 0) = 0.04;
    m(3, 4) = m(4, 3) = 0.05;
    base.diagonal() << 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5;
    xmvb::cuda::Context context(16ULL * 1024 * 1024);
    xmvb::cuda::DeviceCoarsePreconditioner gpu(
        context, n, capacity, 1e-14, 1e-12, {{0, no, 3}, {no, n - no, 2}});
    xmvb::mcscf::CoarsePreconditioner cpu(n, capacity, 1e-14, 1e-12);
    auto d_base = upload(context, base);
    const auto observe_split = [&](const Vector& v,
                                   const Vector& hv,
                                   const Vector& mv,
                                   int begin,
                                   int length,
                                   bool admitted,
                                   int allocated_columns) {
        const Vector compact_v = v.segment(begin, length);
        const Vector compact_m = mv.segment(begin, length);
        auto dv = upload(context, compact_v);
        auto dh = upload(context, hv);
        auto dm = upload(context, compact_m);
        const auto bytes = context.counters().current_device_bytes;
        const int old_size = gpu.size();
        gpu.observe_range(dv.view(), dh.view(), dm.view(), begin);
        cpu.observe(v, hv, mv);
        require(gpu.size() == cpu.size(), "split CUDA coarse admission differs from CPU");
        require(gpu.size() == old_size + static_cast<int>(admitted),
                "split CUDA coarse sample admission unexpected");
        require(context.counters().current_device_bytes ==
                    bytes + 2 * length * allocated_columns * sizeof(double),
                "split CUDA coarse basis and metric allocation unexpected");
    };
    const Vector o0 = Vector::Unit(n, 0), o1 = Vector::Unit(n, 1);
    const Vector s0 = Vector::Unit(n, 3), s1 = Vector::Unit(n, 4);
    observe_split(o0, h * o0, m * o0, 0, no, true, 3);
    observe_split(s0, h * s0, m * s0, no, n - no, true, 2);
    observe_split(o0, h * o0, m * o0, 0, no, false, 0);
    Vector bad = h * o1;
    bad[0] += 0.01;
    observe_split(o1, bad, m * o1, 0, no, false, 0);
    observe_split(o1, h * o1, m * o1, 0, no, true, 0);
    observe_split(s1, h * s1, m * s1, no, n - no, true, 0);
    require(gpu.size() == 4, "split CUDA coarse did not recover after rejection");

    for (double eigenvalue : {0.0, -0.25, 0.4}) {
        int calls = 0;
        for (int i = 0; i < n + 1; ++i) {
            Vector rhs(n);
            if (i < n)
                rhs = Vector::Unit(n, i);
            else
                rhs << 0.2, -0.1, 0.3, 0.4, -0.2, 0.1, -0.3;
            auto d_rhs = upload(context, rhs);
            DeviceMatrix d_out(context, n, 1);
            gpu.apply(
                d_rhs.view(),
                eigenvalue,
                [&](xmvb::linalg::ConstMatrixView input, xmvb::linalg::MatrixView output) {
                    ++calls;
                    xmvb::cuda::gemm(context, output, d_base.view(), input);
                },
                d_out.view());
            const Vector expected = cpu.apply(
                rhs, eigenvalue, [&](const Vector& residual) -> Vector { return base * residual; });
            require((download(context, d_out.view()) - expected).norm() < 3e-11,
                    "split CUDA coarse inverse differs from dense CPU formula");
        }
        require(calls == n + 1, "split CUDA coarse inverse called base inverse more than once");
    }
    const Vector partial_host = Vector::Ones(2), image_host = Vector::Ones(n);
    auto partial = upload(context, partial_host);
    auto image = upload(context, image_host);
    bool rejected = false;
    try {
        gpu.observe_range(partial.view(), image.view(), partial.view(), 1);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected && gpu.size() == 4, "split CUDA coarse accepted overlapping support");
}

void check_split_panels() {
    constexpr int no = 11, n = 23, capacity = 20;
    Matrix h = Matrix::Zero(n, n), m = Matrix::Zero(n, n), base = Matrix::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        h(i, i) = 2.0 + 0.1 * i;
        m(i, i) = 1.0 + 0.02 * i;
        base(i, i) = 0.25 + 0.01 * i;
    }
    for (int i = 0; i < no; ++i) {
        h(i, no + i) = h(no + i, i) = 0.02 * (i + 1);
        if (i + 1 < no) {
            h(i, i + 1) = h(i + 1, i) = 0.03;
            m(i, i + 1) = m(i + 1, i) = 0.01;
        }
    }
    for (int i = no; i + 1 < n; ++i) {
        h(i, i + 1) = h(i + 1, i) = -0.025;
        m(i, i + 1) = m(i + 1, i) = 0.015;
    }
    xmvb::cuda::Context context(32ULL * 1024 * 1024);
    xmvb::cuda::DeviceCoarsePreconditioner gpu(
        context, n, capacity, 1e-14, 1e-12, {{0, no, 10}, {no, n - no, 10}});
    xmvb::mcscf::CoarsePreconditioner cpu(n, capacity, 1e-14, 1e-12);
    auto d_base = upload(context, base);
    const auto admit = [&](int coordinate, bool corrupt, int reserved_columns) {
        const int begin = coordinate < no ? 0 : no;
        const int length = coordinate < no ? no : n - no;
        const Vector v = Vector::Unit(n, coordinate);
        Vector hv = h * v;
        if (corrupt)
            hv[0] += 0.01;
        const Vector mv = m * v;
        const Vector compact_v = v.segment(begin, length);
        const Vector compact_m = mv.segment(begin, length);
        auto dv = upload(context, compact_v), dh = upload(context, hv),
             dm = upload(context, compact_m);
        const auto bytes = context.counters().current_device_bytes;
        const int prior = gpu.size();
        gpu.observe_range(dv.view(), dh.view(), dm.view(), begin);
        cpu.observe(v, hv, mv);
        require(gpu.size() == cpu.size(), "paneled CUDA coarse admission differs from CPU");
        require(gpu.size() == prior + static_cast<int>(!corrupt),
                "paneled CUDA coarse rejected or admitted the wrong sample");
        require(context.counters().current_device_bytes ==
                    bytes + 2 * length * reserved_columns * sizeof(double),
                "paneled CUDA coarse persistent allocation differs from support contract");
    };
    for (int i = 0; i < 8; ++i) {
        admit(i, false, i == 0 ? 10 : 0);
        if (i == 0)
            admit(no, true, 0);
        admit(no + i, false, i == 0 ? 10 : 0);
    }
    require(gpu.size() == 16, "paneled CUDA coarse lost alternating samples");
    // Rejection within an existing panel also preserves rank and bytes.
    admit(8, true, 0);
    require(gpu.size() == 16, "paneled CUDA coarse failed sample rollback");
    admit(8, false, 0);
    admit(no + 8, false, 0);
    admit(9, false, 0);
    admit(no + 9, false, 0);
    require(gpu.size() == capacity, "paneled CUDA coarse did not fill both supports");

    for (double eigenvalue : {0.0, -0.3, 0.45}) {
        int calls = 0;
        for (int k = 0; k < 4; ++k) {
            Vector rhs(n);
            for (int i = 0; i < n; ++i)
                rhs[i] = std::sin((i + 1) * (k + 2) * 0.17);
            auto d_rhs = upload(context, rhs);
            DeviceMatrix d_out(context, n, 1);
            gpu.apply(
                d_rhs.view(),
                eigenvalue,
                [&](xmvb::linalg::ConstMatrixView input, xmvb::linalg::MatrixView output) {
                    ++calls;
                    xmvb::cuda::gemm(context, output, d_base.view(), input);
                },
                d_out.view());
            const Vector expected = cpu.apply(
                rhs, eigenvalue, [&](const Vector& residual) -> Vector { return base * residual; });
            require((download(context, d_out.view()) - expected).norm() < 3e-11,
                    "paneled CUDA coarse shifted inverse differs from dense CPU formula");
        }
        require(calls == 4, "paneled CUDA coarse called base inverse more than once");
    }
}

} // namespace

int main() {
    try {
        check_physical_scaling();
        check_shift_pole();
        check_balancing();
        check_certification();
        check_scalar_batches();
        check_split_support();
        check_split_panels();
        std::cout << "CUDA coupled coarse correction: PASS\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
