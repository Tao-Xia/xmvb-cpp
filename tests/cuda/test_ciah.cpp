#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <Eigen/Core>

#include "cuda/linalg/blas.hpp"
#include "cuda/optimization/ciah.hpp"
#include "mcscf/optimization/ciah/driver.hpp"
#include "mcscf/optimization/ciah/solver.hpp"

namespace {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

Matrix download(xmvb::cuda::Context& context, const xmvb::cuda::Matrix<double>& x) {
    Matrix result(x.rows(), x.cols());
    xmvb::cuda::copy_to_host(context, result, x.view());
    return result;
}

void near(const Matrix& x, const Matrix& y, const char* message) {
    require(x.rows() == y.rows() && x.cols() == y.cols() && x.allFinite() &&
                (x - y).norm() < 2e-10 * std::max(1.0, y.norm()),
            message);
}

void check(bool indefinite, int n = 5) {
    Matrix h = Matrix::Zero(n, n), m = Matrix::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        h(i, i) = indefinite && i == 0 ? -0.7 : 0.4 + 0.7 * i;
        m(i, i) = 0.7 + 0.4 * i;
        if (i)
            h(i, i - 1) = h(i - 1, i) = 0.12;
    }
    Vector gradient(n);
    gradient.head(5) << 0.4, -0.2, 0.1, 0.25, -0.07;
    for (int i = 5; i < n; ++i)
        gradient[i] = 0.1 * std::cos(0.37 * i);
    const xmvb::mcscf::QuadraticProblem problem(
        gradient,
        [&h](const Vector& x) { return Vector(h * x); },
        [&m](const Vector& x) { return Vector(m * x); },
        [](const Vector& x) { return x.norm(); });
    xmvb::mcscf::CiahOptions host_options;
    host_options.max_cycle = n;
    xmvb::mcscf::CiahWorkspace host(problem, host_options);
    xmvb::cuda::Context context(64ULL * 1024 * 1024);
    xmvb::cuda::Matrix<double> dh(context, n, n), dm(context, n, n), dg(context, n, 1);
    xmvb::cuda::copy_to_device(context, dh.view(), h);
    xmvb::cuda::copy_to_device(context, dm.view(), m);
    xmvb::cuda::copy_to_device(context, dg.view(), gradient);
    xmvb::cuda::DeviceCiahOptions options;
    options.max_cycle = n;
    int actions = 0;
    options.apply_images = [&](xmvb::linalg::ConstMatrixView x,
                               xmvb::linalg::MatrixView hx,
                               xmvb::linalg::MatrixView mx) {
        ++actions;
        xmvb::cuda::gemm(context, hx, dh.view(), x);
        xmvb::cuda::gemm(context, mx, dm.view(), x);
    };
    xmvb::cuda::DeviceCiahWorkspace device(context, n, options);
    for (int iteration = 0; iteration < 8; ++iteration) {
        const auto cpu = host.next(gradient);
        const auto gpu = device.next(dg.view());
        require(cpu.basis_size == gpu.basis_size && cpu.new_hvp_actions == gpu.new_hvp_actions,
                "CUDA CIAH changed subspace expansion policy");
        require(std::abs(cpu.eigenvalue - gpu.eigenvalue) < 2e-10,
                "CUDA CIAH Ritz root differs from CPU");
        near(download(context, gpu.step), cpu.step, "CUDA CIAH step differs from CPU");
        near(download(context, gpu.hessian_step),
             cpu.hessian_step,
             "CUDA CIAH Hessian image differs from CPU");
        near(download(context, gpu.metric_step),
             cpu.metric_step,
             "CUDA CIAH metric image differs from CPU");
        near(download(context, gpu.residual),
             cpu.residual,
             "CUDA CIAH full residual differs from CPU");
        if (cpu.converged || cpu.exhaustion)
            break;
    }
    // A keyframe RHS refresh must retain the subspace and make no new HVP.
    gradient *= 0.73;
    host.reset_rhs();
    device.reset_rhs();
    xmvb::cuda::copy_to_device(context, dg.view(), gradient);
    const int before = actions;
    const auto cpu = host.next(gradient);
    const auto gpu = device.next(dg.view());
    require(actions == before && gpu.new_hvp_actions == 0,
            "CUDA CIAH recomputed curvature after an RHS refresh");
    near(download(context, gpu.step), cpu.step, "CUDA CIAH keyframe step differs from CPU");

    // The same frozen quadratic model also exercises clipping, replay, and
    // exact keyframe refresh through the shared microstep driver.
    int host_actions = 0;
    const xmvb::mcscf::QuadraticProblem micro_problem(
        gradient,
        [&](const Vector& x) {
            ++host_actions;
            return Vector(h * x);
        },
        [&m](const Vector& x) { return Vector(m * x); },
        [](const Vector& x) { return x.norm(); });
    xmvb::mcscf::CiahWorkspace micro_host(micro_problem, host_options);
    xmvb::cuda::DeviceCiahWorkspace micro_device(context, n, options);
    xmvb::mcscf::CiahMicroCallbacks host_callbacks;
    host_callbacks.coordinate_norm = [](const Vector& x) { return x.norm(); };
    host_callbacks.metric_action = [&m](const Vector& x) { return Vector(m * x); };
    host_callbacks.maximum_component = [](const Vector& x) { return x.cwiseAbs().maxCoeff(); };
    host_callbacks.orbital_size = n;
    host_callbacks.evaluate_keyframe = [&](const Vector& displacement) {
        const Vector image = h * displacement;
        return xmvb::mcscf::CiahKeyframe{Vector(gradient + image),
                                         gradient.dot(displacement) + 0.5 * displacement.dot(image),
                                         gradient.dot(displacement) +
                                             0.5 * displacement.dot(image)};
    };
    xmvb::cuda::DeviceCiahMicroCallbacks device_callbacks;
    device_callbacks.coordinate_norm = [&](xmvb::linalg::ConstMatrixView x) {
        double value = 0.0;
        xmvb::cuda::check_cublas(cublasDnrm2(context.blas(), n, x.data, 1, &value),
                                 "CIAH test device norm");
        return value;
    };
    device_callbacks.metric_action = [&](xmvb::linalg::ConstMatrixView x,
                                         xmvb::linalg::MatrixView out) {
        xmvb::cuda::gemm(context, out, dm.view(), x);
    };
    device_callbacks.maximum_component = [&](xmvb::linalg::ConstMatrixView x) {
        int one_based = 0;
        xmvb::cuda::check_cublas(cublasIdamax(context.blas(), n, x.data, 1, &one_based),
                                 "CIAH test device maximum");
        double value = 0.0;
        context.copy_to_host(&value, x.data + one_based - 1, sizeof(double));
        return std::abs(value);
    };
    device_callbacks.orbital_size = n;
    device_callbacks.evaluate_keyframe = [&](xmvb::linalg::ConstMatrixView x) {
        xmvb::cuda::Matrix<double> image(context, n, 1);
        xmvb::cuda::Matrix<double> exact_gradient(context, n, 1);
        xmvb::cuda::gemm(context, image.view(), dh.view(), x);
        xmvb::cuda::check_cuda(cudaMemcpyAsync(exact_gradient.data(),
                                               dg.data(),
                                               n * sizeof(double),
                                               cudaMemcpyDeviceToDevice,
                                               context.stream()),
                               "CIAH test device keyframe gradient copy");
        constexpr double one = 1.0;
        xmvb::cuda::check_cublas(
            cublasDaxpy(context.blas(), n, &one, image.data(), 1, exact_gradient.data(), 1),
            "CIAH test device keyframe gradient");
        double linear = 0.0, quadratic = 0.0;
        xmvb::cuda::check_cublas(cublasDdot(context.blas(), n, dg.data(), 1, x.data, 1, &linear),
                                 "CIAH test device linear energy");
        xmvb::cuda::check_cublas(
            cublasDdot(context.blas(), n, x.data, 1, image.data(), 1, &quadratic),
            "CIAH test device quadratic energy");
        const double energy = linear + 0.5 * quadratic;
        return xmvb::cuda::DeviceCiahKeyframe{std::move(exact_gradient), energy, energy};
    };
    const int device_actions_before = actions;
    const auto host_micro =
        xmvb::mcscf::run_ciah_microsteps(micro_host, gradient, 0.0, 1.0e-3, host_callbacks);
    const auto device_micro = xmvb::cuda::run_device_ciah_microsteps(
        micro_device, dg.view(), 0.0, 1.0e-3, device_callbacks);
    require(host_micro.micro_steps == device_micro.micro_steps &&
                host_micro.micro_iterations == device_micro.micro_iterations &&
                host_micro.keyframe_evaluations == device_micro.keyframe_evaluations &&
                host_micro.rejected_trial_count == device_micro.rejected_trial_count &&
                host_micro.energy_model_stopped == device_micro.energy_model_stopped &&
                std::string_view(host_micro.stop_reason) ==
                    std::string_view(device_micro.stop_reason),
            "CUDA CIAH microstep decisions differ from CPU");
    require(host_actions == actions - device_actions_before &&
                micro_host.basis_size() == micro_device.basis_size(),
            "CUDA CIAH microstep Hessian actions differ from CPU");
    near(download(context, device_micro.displacement),
         host_micro.displacement,
         "CUDA CIAH microstep displacement differs from CPU");
    near(download(context, device_micro.hessian_displacement),
         host_micro.hessian_displacement,
         "CUDA CIAH microstep Hessian displacement differs from CPU");
    near(download(context, device_micro.predicted_gradient),
         host_micro.predicted_gradient,
         "CUDA CIAH microstep gradient differs from CPU");
    near(download(context, device_micro.last_step),
         host_micro.last_step,
         "CUDA CIAH microstep last step differs from CPU");
}

/** @brief Scaled weak-reference AH roots retain host/device residual and image parity. */
void check_scaled_hard_case() {
    constexpr int n = 3;
    const Matrix h = Vector((Vector(3) << -2, 1, 7).finished()).asDiagonal();
    const Matrix mass = Vector((Vector(3) << 1, 2, .5).finished()).asDiagonal();
    const Vector g = (Vector(3) << 0, .4, .2).finished();
    const Vector seed = Vector::Ones(n);
    int host_actions = 0, device_actions = 0;
    const xmvb::mcscf::QuadraticProblem problem(
        g,
        [&](const Vector& x) {
            ++host_actions;
            return Vector(h * x);
        },
        [&](const Vector& x) { return Vector(mass * x); },
        [](const Vector& x) { return x.norm(); });
    xmvb::mcscf::CiahWorkspace host(problem, seed);
    host.set_step_limit([](const Vector& x) { return x.cwiseAbs().maxCoeff(); }, .03);
    xmvb::cuda::Context context(64ULL * 1024 * 1024);
    xmvb::cuda::Matrix<double> dh(context, n, n), dm(context, n, n), dg(context, n, 1),
        ds(context, n, 1);
    xmvb::cuda::copy_to_device(context, dh.view(), h);
    xmvb::cuda::copy_to_device(context, dm.view(), mass);
    xmvb::cuda::copy_to_device(context, dg.view(), g);
    xmvb::cuda::copy_to_device(context, ds.view(), seed);
    xmvb::cuda::DeviceCiahOptions options;
    options.max_cycle = n;
    options.apply_images = [&](xmvb::linalg::ConstMatrixView x,
                               xmvb::linalg::MatrixView hx,
                               xmvb::linalg::MatrixView mx) {
        ++device_actions;
        xmvb::cuda::gemm(context, hx, dh.view(), x);
        xmvb::cuda::gemm(context, mx, dm.view(), x);
    };
    xmvb::cuda::DeviceCiahWorkspace device(context, n, options, ds.view());
    device.set_step_limit(
        [&](xmvb::linalg::ConstMatrixView x) {
            int index = 0;
            xmvb::cuda::check_cublas(cublasIdamax(context.blas(), n, x.data, 1, &index),
                                     "scaled AH maximum");
            double value = 0;
            context.copy_to_host(&value, x.data + index - 1, sizeof(value));
            return std::abs(value);
        },
        .03);
    bool scaled = false;
    for (int j = 0; j < 4; ++j) {
        const auto cpu = host.next(g);
        const auto gpu = device.next(dg.view());
        scaled = scaled || cpu.reference_scale > 1;
        require(cpu.basis_size == gpu.basis_size && cpu.new_hvp_actions == gpu.new_hvp_actions,
                "scaled CUDA AH changed its physical image schedule");
        require(std::abs(cpu.reference_scale - gpu.reference_scale) <
                    2e-9 * std::max(1.0, cpu.reference_scale),
                "scaled CUDA AH lost its reference scale");
        near(download(context, gpu.step), cpu.step, "scaled CUDA AH step differs");
        near(download(context, gpu.residual),
             cpu.residual,
             "scaled CUDA AH complete residual differs");
    }
    require(scaled && host_actions == device_actions && host_actions == host.basis_size(),
            "scaled CUDA AH was not exercised or repeated an image");
}

} // namespace

int main() {
    try {
        check_scaled_hard_case();
        check(false);
        check(true);
        check(false, 513);
        check(true, 513);
        std::cout << "CUDA shared CIAH quadratic/keyframe/microstep parity: PASS\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
