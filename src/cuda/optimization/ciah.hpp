#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

#include "cuda/memory/buffer.hpp"
#include "mcscf/optimization/ciah/core.hpp"
#include "mcscf/optimization/ciah/solver.hpp"

namespace xmvb::cuda {

/** All callbacks operate on Context::stream() and keep large vectors on device. */
struct DeviceCiahOptions {
    std::function<void(linalg::ConstMatrixView, linalg::MatrixView, linalg::MatrixView)>
        apply_images;
    /** Used by the optional inverse audit trace; otherwise no separate G action. */
    std::function<void(linalg::ConstMatrixView, linalg::MatrixView)> apply_metric;
    std::function<void(linalg::ConstMatrixView, double, linalg::MatrixView)>
        residual_preconditioner;
    std::function<void(linalg::ConstMatrixView, linalg::MatrixView)> residual_direction;
    std::function<double(linalg::ConstMatrixView)> coordinate_norm;
    std::function<void(linalg::ConstMatrixView, linalg::ConstMatrixView)> observe_images;
    double tolerance = 1.0e-12;
    double lindep = 1.0e-14;
    int max_cycle = 30;
    double operator_relative_accuracy = 0.0;
};

using DeviceCiahResult = mcscf::detail::CiahCoreResult<Matrix<double>>;
using DeviceCiahOriginRay = mcscf::detail::CiahCoreOriginRay<Matrix<double>>;

/** A frozen quadratic operator and retained device-resident CIAH span. */
class DeviceCiahWorkspace {
  public:
    DeviceCiahWorkspace(Context& context,
                        std::int64_t dimension,
                        DeviceCiahOptions options,
                        std::optional<linalg::ConstMatrixView> initial = std::nullopt);
    ~DeviceCiahWorkspace();
    DeviceCiahWorkspace(const DeviceCiahWorkspace&) = delete;
    DeviceCiahWorkspace& operator=(const DeviceCiahWorkspace&) = delete;

    void set_step_limit(std::function<double(linalg::ConstMatrixView)> maximum, double cap);
    void reset_rhs() noexcept;
    int basis_size() const noexcept;
    Context& context() const noexcept;
    std::int64_t dimension() const noexcept;
    DeviceCiahResult next(linalg::ConstMatrixView current_gradient);
    DeviceCiahResult next(const Matrix<double>& current_gradient);
    double projected_dual_norm(linalg::ConstMatrixView covector) const;
    double projected_dual_norm(const Matrix<double>& covector) const;
    std::optional<DeviceCiahOriginRay> origin_ray(linalg::ConstMatrixView origin_gradient) const;
    std::optional<DeviceCiahOriginRay> origin_ray(const Matrix<double>& origin_gradient) const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/** Device vector primitives shared by the CIAH path, model, and checkpoints. */
struct DeviceCiahVectorOps {
    using Vector = Matrix<double>;
    Context* context = nullptr;
    Vector clone(const Vector& v) const;
    Vector zeros_like(const Vector& v) const;
    void axpy(Vector& y, double factor, const Vector& x) const;
    void scal(Vector& x, double factor) const;
    double dot(const Vector& x, const Vector& y) const;
    double norm(const Vector& x) const;
    double segment_norm(const Vector& x, std::int64_t begin, std::int64_t count) const;
    bool is_zero(const Vector& x) const;
};

struct DeviceCiahKeyframe {
    Matrix<double> gradient;
    double energy = 0.0;
    double physical_energy = 0.0;
};

struct DeviceCiahMicroCallbacks {
    std::function<double(linalg::ConstMatrixView)> coordinate_norm;
    std::function<void(linalg::ConstMatrixView, linalg::MatrixView)> metric_action;
    std::function<double(linalg::ConstMatrixView)> maximum_component;
    std::function<DeviceCiahKeyframe(linalg::ConstMatrixView)> evaluate_keyframe;
    std::int64_t orbital_size = 0;
    std::function<double(linalg::ConstMatrixView)> orbital_coordinate_norm;
    std::function<void(const char*, linalg::ConstMatrixView, linalg::ConstMatrixView, int)>
        audit_step;
    std::function<void(linalg::ConstMatrixView,
                       linalg::ConstMatrixView,
                       linalg::ConstMatrixView,
                       linalg::ConstMatrixView)>
        audit_origin;
    std::function<void(linalg::ConstMatrixView,
                       linalg::ConstMatrixView,
                       linalg::ConstMatrixView,
                       linalg::ConstMatrixView,
                       int)>
        audit_resumed;
    std::function<void(linalg::ConstMatrixView, linalg::ConstMatrixView, const DeviceCiahKeyframe&)>
        audit_keyframe;
};

struct DeviceCiahMicroResult {
    DeviceCiahMicroResult(Context& context, double energy, const Matrix<double>& initial_gradient);
    Matrix<double> displacement, hessian_displacement, predicted_gradient, last_step;
    DeviceCiahResult candidate;
    mcscf::detail::CiahEnergyModelT<DeviceCiahVectorOps> energy_model;
    mcscf::detail::CiahKeyframesT<DeviceCiahResult, DeviceCiahVectorOps> keyframes;
    int micro_steps = 0;
    int micro_iterations = 0;
    int keyframe_evaluations = 0;
    int rejected_trial_count = 0;
    bool energy_model_stopped = false;
    const char* stop_reason = "unset";
};

DeviceCiahMicroResult run_device_ciah_microsteps(DeviceCiahWorkspace& workspace,
                                                 linalg::ConstMatrixView initial_gradient,
                                                 double energy,
                                                 double outer_gradient_tolerance,
                                                 const DeviceCiahMicroCallbacks& callbacks);

} // namespace xmvb::cuda
