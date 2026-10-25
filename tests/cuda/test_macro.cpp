#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "cuda/linalg/blas.hpp"
#include "cuda/optimization/ciah.hpp"
#include "mcscf/optimization/ciah/driver.hpp"
#include "mcscf/optimization/ciah/macro.hpp"

namespace {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using xmvb::mcscf::CiahMacroObservation;

enum class Case { accept, backtrack, restart };
struct Trial {
    double energy = 0.0;
};

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void near(double lhs, double rhs, const char* message) {
    require(std::isfinite(lhs) && std::isfinite(rhs) &&
                std::abs(lhs - rhs) < 2e-9 * std::max({1.0, std::abs(lhs), std::abs(rhs)}),
            message);
}

void near(const Vector& lhs, const Vector& rhs, const char* message) {
    require(lhs.size() == rhs.size() && lhs.allFinite() && rhs.allFinite() &&
                (lhs - rhs).norm() < 2e-9 * std::max(1.0, rhs.norm()),
            message);
}

Vector download(xmvb::cuda::Context& context, const xmvb::cuda::Matrix<double>& source) {
    Matrix result(source.rows(), source.cols());
    xmvb::cuda::copy_to_host(context, result, source.view());
    Vector values(result.size());
    values = Eigen::Map<const Vector>(result.data(), result.size());
    return values;
}

double device_dot(xmvb::cuda::Context& context,
                  xmvb::linalg::ConstMatrixView lhs,
                  xmvb::linalg::ConstMatrixView rhs) {
    double value = 0.0;
    xmvb::cuda::check_cublas(cublasDdot(context.blas(), lhs.rows, lhs.data, 1, rhs.data, 1, &value),
                             "macro test dot");
    return value;
}

double device_norm(xmvb::cuda::Context& context, xmvb::linalg::ConstMatrixView source) {
    double value = 0.0;
    xmvb::cuda::check_cublas(cublasDnrm2(context.blas(), source.rows, source.data, 1, &value),
                             "macro test norm");
    return value;
}

double device_maximum(xmvb::cuda::Context& context, xmvb::linalg::ConstMatrixView source) {
    int index = 0;
    xmvb::cuda::check_cublas(cublasIdamax(context.blas(), source.rows, source.data, 1, &index),
                             "macro test maximum");
    double value = 0.0;
    context.copy_to_host(&value, source.data + index - 1, sizeof(double));
    return std::abs(value);
}

void compare_observations(const std::vector<CiahMacroObservation>& host,
                          const std::vector<CiahMacroObservation>& device) {
    require(host.size() == device.size(), "macro trial count differs");
    for (std::size_t i = 0; i < host.size(); ++i) {
        near(host[i].scale, device[i].scale, "macro trial scale differs");
        near(host[i].linear, device[i].linear, "macro trial linear model differs");
        near(host[i].quadratic, device[i].quadratic, "macro trial quadratic model differs");
        near(host[i].predicted, device[i].predicted, "macro trial prediction differs");
        near(host[i].actual, device[i].actual, "macro trial energy differs");
        require(host[i].accepted == device[i].accepted && host[i].terminal == device[i].terminal,
                "macro physical acceptance policy differs");
    }
}

void check(Case scenario) {
    constexpr int n = 3;
    Matrix h = Matrix::Zero(n, n), metric = Matrix::Identity(n, n);
    h.diagonal() << 0.7, 1.2, 1.8;
    Vector gradient(n);
    gradient << 0.4, -0.2, 0.1;
    const xmvb::mcscf::QuadraticProblem problem(
        gradient,
        [&h](const Vector& direction) { return Vector(h * direction); },
        [](const Vector& direction) { return Vector(direction); },
        [](const Vector& direction) { return direction.norm(); });
    xmvb::mcscf::CiahOptions host_options;
    host_options.max_cycle = n;
    xmvb::mcscf::CiahWorkspace host_workspace(problem, host_options);

    xmvb::cuda::Context context(64ULL * 1024 * 1024);
    xmvb::cuda::Matrix<double> device_h(context, n, n), device_m(context, n, n),
        device_gradient(context, n, 1);
    xmvb::cuda::copy_to_device(context, device_h.view(), h);
    xmvb::cuda::copy_to_device(context, device_m.view(), metric);
    xmvb::cuda::copy_to_device(context, device_gradient.view(), gradient);
    xmvb::cuda::DeviceCiahOptions device_options;
    device_options.max_cycle = n;
    device_options.apply_images = [&](xmvb::linalg::ConstMatrixView direction,
                                      xmvb::linalg::MatrixView image,
                                      xmvb::linalg::MatrixView metric_image) {
        xmvb::cuda::gemm(context, image, device_h.view(), direction);
        xmvb::cuda::gemm(context, metric_image, device_m.view(), direction);
    };
    xmvb::cuda::DeviceCiahWorkspace device_workspace(context, n, device_options);

    xmvb::mcscf::CiahMicroCallbacks host_micro_callbacks;
    host_micro_callbacks.coordinate_norm = [](const Vector& x) { return x.norm(); };
    host_micro_callbacks.metric_action = [](const Vector& x) { return Vector(x); };
    host_micro_callbacks.maximum_component = [](const Vector& x) {
        return x.cwiseAbs().maxCoeff();
    };
    host_micro_callbacks.orbital_size = n;
    host_micro_callbacks.evaluate_keyframe = [&](const Vector& displacement) {
        const Vector image = h * displacement;
        const double energy = gradient.dot(displacement) + 0.5 * displacement.dot(image);
        return xmvb::mcscf::CiahKeyframe{Vector(gradient + image), energy, energy};
    };
    xmvb::cuda::DeviceCiahMicroCallbacks device_micro_callbacks;
    device_micro_callbacks.coordinate_norm = [&](xmvb::linalg::ConstMatrixView x) {
        return device_norm(context, x);
    };
    device_micro_callbacks.metric_action = [&](xmvb::linalg::ConstMatrixView x,
                                               xmvb::linalg::MatrixView out) {
        xmvb::cuda::gemm(context, out, device_m.view(), x);
    };
    device_micro_callbacks.maximum_component = [&](xmvb::linalg::ConstMatrixView x) {
        return device_maximum(context, x);
    };
    device_micro_callbacks.orbital_size = n;
    device_micro_callbacks.evaluate_keyframe = [&](xmvb::linalg::ConstMatrixView x) {
        xmvb::cuda::Matrix<double> image(context, n, 1), exact_gradient(context, n, 1);
        xmvb::cuda::gemm(context, image.view(), device_h.view(), x);
        xmvb::cuda::check_cuda(cudaMemcpyAsync(exact_gradient.data(),
                                               device_gradient.data(),
                                               n * sizeof(double),
                                               cudaMemcpyDeviceToDevice,
                                               context.stream()),
                               "macro test keyframe gradient copy");
        constexpr double one = 1.0;
        xmvb::cuda::check_cublas(
            cublasDaxpy(context.blas(), n, &one, image.data(), 1, exact_gradient.data(), 1),
            "macro test keyframe gradient");
        const double energy = device_dot(context, device_gradient.view(), x) +
                              0.5 * device_dot(context, x, image.view());
        return xmvb::cuda::DeviceCiahKeyframe{std::move(exact_gradient), energy, energy};
    };

    auto host_micro = xmvb::mcscf::run_ciah_microsteps(
        host_workspace, gradient, 0.0, 1.0e-3, host_micro_callbacks);
    auto device_micro = xmvb::cuda::run_device_ciah_microsteps(
        device_workspace, device_gradient.view(), 0.0, 1.0e-3, device_micro_callbacks);
    near(host_micro.displacement,
         download(context, device_micro.displacement),
         "macro source microsteps differ");
    require(host_workspace.origin_ray(gradient).has_value() ==
                device_workspace.origin_ray(device_gradient).has_value(),
            "macro origin ray availability differs");

    xmvb::cuda::DeviceCiahVectorOps device_ops{&context};
    if (scenario == Case::backtrack) {
        require(host_micro.displacement.norm() > 0.0, "macro backtrack lacks a microstep");
        const Vector checkpoint = 0.5 * host_micro.displacement;
        const double checkpoint_energy =
            gradient.dot(checkpoint) + 0.5 * checkpoint.dot(h * checkpoint);
        require(checkpoint_energy < 0.0, "macro checkpoint is not descending");
        host_micro.keyframes = xmvb::mcscf::CiahKeyframes(0.0);
        host_micro.keyframes.save(checkpoint_energy,
                                  checkpoint,
                                  Vector(0.5 * host_micro.hessian_displacement),
                                  checkpoint,
                                  host_micro.candidate);
        auto device_checkpoint = device_ops.clone(device_micro.displacement);
        auto device_image = device_ops.clone(device_micro.hessian_displacement);
        device_ops.scal(device_checkpoint, 0.5);
        device_ops.scal(device_image, 0.5);
        const double device_checkpoint_energy =
            device_dot(context, device_gradient.view(), device_checkpoint.view()) +
            0.5 * device_dot(context, device_checkpoint.view(), device_image.view());
        near(checkpoint_energy,
             device_checkpoint_energy,
             "macro checkpoint quadratic energy differs");
        device_micro.keyframes = decltype(device_micro.keyframes)(0.0, device_ops);
        device_micro.keyframes.save(device_checkpoint_energy,
                                    device_checkpoint,
                                    device_image,
                                    device_checkpoint,
                                    device_micro.candidate);
    } else if (scenario == Case::restart) {
        require(host_workspace.origin_ray(gradient).has_value(),
                "macro restart lacks a retained origin ray");
        if (host_micro.keyframes.available())
            host_micro.keyframes.take();
        if (device_micro.keyframes.available())
            device_micro.keyframes.take();
        host_micro.displacement = gradient;
        host_micro.hessian_displacement = h * gradient;
        host_micro.last_step = gradient;
        device_micro.displacement = device_ops.clone(device_gradient);
        device_micro.hessian_displacement = xmvb::cuda::Matrix<double>(context, n, 1);
        xmvb::cuda::gemm(context,
                         device_micro.hessian_displacement.view(),
                         device_h.view(),
                         device_gradient.view());
        device_micro.last_step = device_ops.clone(device_gradient);
        host_micro.energy_model_stopped = true;
        device_micro.energy_model_stopped = true;
    }

    std::vector<CiahMacroObservation> host_observations, device_observations;
    int host_evaluations = 0, device_evaluations = 0;
    xmvb::mcscf::CiahMacroCallbacks<xmvb::mcscf::detail::HostCiahVectorOps, Trial> host_callbacks;
    host_callbacks.evaluate = [&](const Vector& displacement, double scale) {
        ++host_evaluations;
        const Vector step = scale * displacement;
        double energy = gradient.dot(step) + 0.5 * step.dot(h * step);
        if (scenario == Case::backtrack && host_evaluations == 1)
            energy += 1.0;
        return Trial{energy};
    };
    host_callbacks.energy = [](const Trial& trial) { return trial.energy; };
    host_callbacks.terminal = [](Trial&) { return false; };
    host_callbacks.step_norm = [](const Vector& x) { return x.norm(); };
    host_callbacks.maximum_component = [](const Vector& x) { return x.cwiseAbs().maxCoeff(); };
    host_callbacks.changes_parameters = [](const Vector& x) { return x.norm() > 1.0e-15; };
    host_callbacks.observe = [&](const CiahMacroObservation& x) { host_observations.push_back(x); };

    xmvb::mcscf::CiahMacroCallbacks<xmvb::cuda::DeviceCiahVectorOps, Trial> device_callbacks;
    device_callbacks.evaluate = [&](const xmvb::cuda::Matrix<double>& displacement, double scale) {
        ++device_evaluations;
        auto step = device_ops.clone(displacement);
        device_ops.scal(step, scale);
        xmvb::cuda::Matrix<double> image(context, n, 1);
        xmvb::cuda::gemm(context, image.view(), device_h.view(), step.view());
        double energy = device_dot(context, device_gradient.view(), step.view()) +
                        0.5 * device_dot(context, step.view(), image.view());
        if (scenario == Case::backtrack && device_evaluations == 1)
            energy += 1.0;
        return Trial{energy};
    };
    device_callbacks.energy = [](const Trial& trial) { return trial.energy; };
    device_callbacks.terminal = [](Trial&) { return false; };
    device_callbacks.step_norm = [&](const xmvb::cuda::Matrix<double>& x) {
        return device_ops.norm(x);
    };
    device_callbacks.maximum_component = [&](const xmvb::cuda::Matrix<double>& x) {
        return device_maximum(context, x.view());
    };
    device_callbacks.changes_parameters = [&](const xmvb::cuda::Matrix<double>& x) {
        return device_ops.norm(x) > 1.0e-15;
    };
    device_callbacks.observe = [&](const CiahMacroObservation& x) {
        device_observations.push_back(x);
    };

    const auto host = xmvb::mcscf::run_ciah_macro(host_workspace,
                                                  host_micro,
                                                  gradient,
                                                  0.0,
                                                  1.0e-12,
                                                  1.0e-12,
                                                  xmvb::mcscf::detail::HostCiahVectorOps{},
                                                  host_callbacks);
    const auto device = xmvb::mcscf::run_ciah_macro(device_workspace,
                                                    device_micro,
                                                    device_gradient,
                                                    0.0,
                                                    1.0e-12,
                                                    1.0e-12,
                                                    device_ops,
                                                    device_callbacks);
    require(host.trial.has_value() && device.trial.has_value(),
            "macro quadratic trial was not accepted");
    require(std::string_view(host.reason) == std::string_view(device.reason) &&
                host.rejected_trials == device.rejected_trials &&
                host.keyframe_backtracked == device.keyframe_backtracked &&
                host.origin_restarted == device.origin_restarted &&
                host.ritz_converged == device.ritz_converged,
            "macro branch decision differs");
    require(host_evaluations == device_evaluations, "macro physical evaluation count differs");
    compare_observations(host_observations, device_observations);
    near(host.scale, device.scale, "macro final scale differs");
    near(host.predicted_reduction, device.predicted_reduction, "macro predicted reduction differs");
    near(host.actual_reduction, device.actual_reduction, "macro actual reduction differs");
    near(host.trial->energy, device.trial->energy, "macro accepted energy differs");
    near(host_micro.displacement,
         download(context, device_micro.displacement),
         "macro accepted displacement differs");
    near(host_micro.hessian_displacement,
         download(context, device_micro.hessian_displacement),
         "macro accepted Hessian image differs");
    near(host_micro.last_step,
         download(context, device_micro.last_step),
         "macro accepted final microstep differs");
    if (scenario == Case::accept)
        require(host.rejected_trials == 0 && !host.keyframe_backtracked && !host.origin_restarted,
                "macro direct acceptance branch not exercised");
    if (scenario == Case::backtrack)
        require(host.rejected_trials == 1 && host.keyframe_backtracked && !host.origin_restarted,
                "macro keyframe backtrack not exercised");
    if (scenario == Case::restart)
        require(host.rejected_trials == 1 && !host.keyframe_backtracked && host.origin_restarted,
                "macro origin restart not exercised");
}

} // namespace

int main() {
    try {
        check(Case::accept);
        check(Case::backtrack);
        check(Case::restart);
        std::cout << "CUDA shared CIAH macro policy parity: PASS\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
