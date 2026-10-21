#include "mcscf/optimization/ciah/driver.hpp"

#include <stdexcept>

#include "mcscf/optimization/ciah/driver_core.hpp"

namespace xmvb::mcscf {
namespace {

struct HostDriverBackend : detail::HostCiahVectorOps {
  const CiahMicroCallbacks& callbacks;
  const Eigen::Index orbital_size;

  CiahMicroResult make_result(double energy, const Vector& gradient) const {
    return CiahMicroResult(energy, gradient);
  }
  CiahResult next(CiahWorkspace& workspace, const Vector& gradient) const {
    return workspace.next(gradient);
  }
  double projected_dual_norm(CiahWorkspace& workspace,
                             const Vector& covector) const {
    return workspace.projected_dual_norm(covector);
  }
  double coordinate_norm(const Vector& v) const {
    return callbacks.coordinate_norm(v);
  }
  Vector metric_action(const Vector& v) const {
    return callbacks.metric_action(v);
  }
  double maximum_component(const Vector& v) const {
    return callbacks.maximum_component(v);
  }
  CiahKeyframe evaluate_keyframe(const Vector& displacement) const {
    return callbacks.evaluate_keyframe(displacement);
  }
  bool is_zero(const Vector& v) const { return v.isZero(); }
  double orbital_residual_norm(const Vector& v) const {
    return v.head(orbital_size).norm();
  }
  double structure_residual_norm(const Vector& v) const {
    return v.tail(v.size() - orbital_size).norm();
  }
  double orbital_norm(const Vector& v) const {
    return callbacks.orbital_coordinate_norm
        ? callbacks.orbital_coordinate_norm(v.head(orbital_size))
        : v.head(orbital_size).norm();
  }
  void audit_step(const char* label, const Vector& displacement,
                  const Vector& step, int index) const {
    if (callbacks.audit_step)
      callbacks.audit_step(label, displacement, step, index);
  }
  void audit_origin(const Vector& displacement, const Vector& step,
                    const Vector& image, const Vector& gradient) const {
    if (callbacks.audit_origin)
      callbacks.audit_origin(displacement, step, image, gradient);
  }
  void audit_resumed(const Vector& displacement, const Vector& step,
                     const Vector& image, const Vector& gradient,
                     int basis_size) const {
    if (callbacks.audit_resumed)
      callbacks.audit_resumed(displacement, step, image, gradient, basis_size);
  }
  void audit_keyframe(const Vector& displacement, const Vector& image,
                      const CiahKeyframe& keyframe) const {
    if (callbacks.audit_keyframe)
      callbacks.audit_keyframe(displacement, image, keyframe);
  }
  bool has_audit_resumed() const noexcept {
    return static_cast<bool>(callbacks.audit_resumed);
  }
};

}  // namespace

CiahMicroResult run_ciah_microsteps(CiahWorkspace& workspace,
    const Eigen::VectorXd& initial_gradient, double energy,
    double outer_gradient_tolerance, const CiahMicroCallbacks& callbacks) {
  if (!callbacks.coordinate_norm || !callbacks.metric_action ||
      !callbacks.maximum_component || !callbacks.evaluate_keyframe ||
      callbacks.orbital_size < 0 ||
      callbacks.orbital_size > initial_gradient.size())
    throw std::invalid_argument("CIAH microstep callbacks are incomplete");
  return detail::run_ciah_microsteps_core(
      HostDriverBackend{{}, callbacks, callbacks.orbital_size}, workspace,
      initial_gradient, energy, outer_gradient_tolerance);
}

}  // namespace xmvb::mcscf
