#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include "core/timing.hpp"
#include "mcscf/optimization/trace.hpp"

namespace xmvb::mcscf::detail {

struct CiahNumerics {
  double tolerance = 1.0e-12;
  double lindep = 1.0e-14;
  int max_cycle = 30;
  double operator_relative_accuracy = 0.0;
};

/** Signed inner product and absolute-product sum for roundoff certification. */
struct CiahProduct {
  double value=0.0;
  double magnitude=0.0;
};

template <class Vector>
struct CiahCoreResult {
  double eigenvalue = 0.0;
  double reference_component = 0.0;
  Vector step, hessian_step, metric_step, residual;
  double residual_norm = 0.0, model_gradient_norm = 0.0;
  bool converged = false, exhaustion = false;
  int basis_size = 0, new_hvp_actions = 0;
};

template <class Vector>
struct CiahCoreOriginRay {
  Vector step, hessian_step;
  double reference_component = 0.0, eigenvalue = 0.0, residual_norm = 0.0;
};

inline void check_ciah_projected_symmetry(
    Eigen::Index dimension, double basis_norm, double images_norm,
    const Eigen::MatrixXd& magnitude, const Eigen::MatrixXd& action,
    const char* name, double relative_action_error = 0.0) {
  const Eigen::MatrixXd skew = action - action.transpose();
  const double u = 0.5 * std::numeric_limits<double>::epsilon();
  const double nu = static_cast<double>(dimension) * u;
  const double gamma = nu / (1.0 - nu);
  const Eigen::MatrixXd error =
      (gamma / (1.0 - gamma)) * (magnitude + magnitude.transpose()) +
      u * (action.cwiseAbs() + action.transpose().cwiseAbs());
  const double roundoff = error.stableNorm();
  const double tolerance = 256.0 * std::numeric_limits<double>::epsilon() *
      static_cast<double>(action.rows()) * std::max(1.0, action.norm()) +
      roundoff + relative_action_error * basis_norm * images_norm;
  if (!std::isfinite(skew.norm()) || skew.norm() > tolerance) {
    std::ostringstream message;
    message.precision(17);
    message << name << ": skew=" << skew.norm()
            << ", tolerance=" << tolerance << ", dot_roundoff=" << roundoff
            << ", action_norm=" << action.norm()
            << ", basis_norm=" << basis_norm << ", images_norm=" << images_norm
            << ", dimension=" << dimension << ", subspace=" << action.rows();
    throw std::runtime_error(message.str());
  }
}

/** The sole CIAH Davidson iteration. Backend vectors may live on host or device. */
template <class Backend>
class CiahCore {
public:
  using Vector = typename Backend::Vector;
  using Result = CiahCoreResult<Vector>;
  using OriginRay = CiahCoreOriginRay<Vector>;

  CiahCore(Backend backend, std::optional<Vector> initial, CiahNumerics numerics)
      : backend_(std::move(backend)), numerics_(numerics),
        initial_direction_(std::move(initial)) {
    if (!(numerics_.tolerance > 0.0) || !std::isfinite(numerics_.tolerance) ||
        !(numerics_.lindep > 0.0) || !std::isfinite(numerics_.lindep) ||
        numerics_.lindep >= 1.0 || numerics_.max_cycle < 1 ||
        numerics_.operator_relative_accuracy < 0.0 ||
        !std::isfinite(numerics_.operator_relative_accuracy))
      throw std::invalid_argument("invalid CIAH Davidson options");
    if (initial_direction_ && !valid(*initial_direction_))
      throw std::invalid_argument("invalid CIAH initial direction");
    const Eigen::Index capacity =
        std::min<Eigen::Index>(numerics_.max_cycle, backend_.size());
    hessian_projection_ = Eigen::MatrixXd::Zero(capacity, capacity);
    metric_projection_ = hessian_projection_;
    hessian_magnitude_ = hessian_projection_;
    metric_magnitude_ = hessian_projection_;
  }

  int basis_size() const noexcept { return static_cast<int>(basis_.size()); }
  double lindep() const noexcept { return numerics_.lindep; }
  const Eigen::MatrixXd& hessian_projection() const noexcept { return hessian_projection_; }
  const Eigen::MatrixXd& metric_projection() const noexcept { return metric_projection_; }
  const std::vector<Vector>& basis() const noexcept { return basis_; }
  const std::vector<Vector>& hessian_images() const noexcept { return hessian_images_; }
  const std::vector<Vector>& metric_images() const noexcept { return metric_images_; }

  void reset_rhs() noexcept {
    pending_residual_.reset();
    pending_eigenvalue_ = 0.0;
    pending_residual_norm_ = 0.0;
    previous_eigenvalue_.reset();
    exhausted_ = false;
  }

  double projected_dual_norm(const Vector& covector) const {
    if (!valid(covector))
      throw std::invalid_argument("invalid CIAH diagnostic covector");
    if (basis_.empty()) return 0.0;
    if (projected_metric_.rows() != basis_size())
      throw std::logic_error("CIAH diagnostic requires a completed Ritz solve");
    Eigen::VectorXd rhs(basis_size());
    for (int i = 0; i < basis_size(); ++i)
      rhs[i] = backend_.dot(basis_[i], covector);
    const Eigen::LLT<Eigen::MatrixXd> factor(projected_metric_);
    if (factor.info() != Eigen::Success)
      throw std::runtime_error("CIAH diagnostic metric factorization failed");
    const Eigen::VectorXd solution = factor.solve(rhs);
    const double squared = rhs.dot(solution);
    if (!(squared >= 0.0) || !std::isfinite(squared))
      throw std::runtime_error("CIAH diagnostic projection norm is invalid");
    return std::sqrt(squared);
  }

  std::optional<OriginRay> origin_ray(const Vector& origin_gradient) const {
    if (!valid(origin_gradient))
      throw std::invalid_argument("invalid CIAH origin gradient");
    const Eigen::Index m = basis_size();
    if (!m) return std::nullopt;
    if (projected_metric_.rows() != m || projected_metric_.cols() != m)
      throw std::logic_error("CIAH origin ray requires a completed Ritz solve");
    const Eigen::MatrixXd raw_hessian = hessian_projection_.topLeftCorner(m, m);
    const Eigen::MatrixXd projected_hessian =
        0.5 * (raw_hessian + raw_hessian.transpose());
    Eigen::VectorXd projected_gradient(m);
    for (Eigen::Index i = 0; i < m; ++i)
      projected_gradient[i] = backend_.dot(basis_[i], origin_gradient);
    Eigen::MatrixXd augmented_hessian = Eigen::MatrixXd::Zero(m + 1, m + 1);
    augmented_hessian.block(0, 1, 1, m) = projected_gradient.transpose();
    augmented_hessian.block(1, 0, m, 1) = projected_gradient;
    augmented_hessian.bottomRightCorner(m, m) = projected_hessian;
    Eigen::MatrixXd augmented_metric = Eigen::MatrixXd::Identity(m + 1, m + 1);
    augmented_metric.bottomRightCorner(m, m) = projected_metric_;
    const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(
        augmented_hessian, augmented_metric);
    if (spectrum.info() != Eigen::Success ||
        !spectrum.eigenvalues().allFinite() ||
        !spectrum.eigenvectors().allFinite())
      throw std::runtime_error("CIAH origin augmented eigensolver failed");
    const Eigen::VectorXd coefficients = spectrum.eigenvectors().col(0).tail(m);
    OriginRay ray;
    ray.eigenvalue = spectrum.eigenvalues()[0];
    ray.reference_component = std::abs(spectrum.eigenvectors()(0, 0));
    ray.step = combine(basis_, coefficients);
    ray.hessian_step = combine(hessian_images_, coefficients);
    Vector metric_step = combine(metric_images_, coefficients);
    Vector residual = backend_.clone(origin_gradient);
    backend_.scal(residual, spectrum.eigenvectors()(0, 0));
    backend_.axpy(residual, 1.0, ray.hessian_step);
    backend_.axpy(residual, -ray.eigenvalue, metric_step);
    if (!valid(ray.step) || !valid(ray.hessian_step) || !valid(residual))
      throw std::runtime_error("CIAH origin ray is invalid");
    ray.residual_norm = backend_.coordinate_norm(residual, true);
    if (!(ray.residual_norm >= 0.0) || !std::isfinite(ray.residual_norm))
      throw std::runtime_error("CIAH origin ray is invalid");
    if (!(backend_.stable_norm(ray.step) > 0.0)) return std::nullopt;
    if (backend_.dot(origin_gradient, ray.step) > 0.0) {
      backend_.scal(ray.step, -1.0);
      backend_.scal(ray.hessian_step, -1.0);
    }
    if (!(backend_.dot(origin_gradient, ray.step) < 0.0 ||
          backend_.dot(ray.step, ray.hessian_step) < 0.0)) return std::nullopt;
    return std::optional<OriginRay>(std::move(ray));
  }

  Result next(const Vector& current_gradient) {
    xmvb::profiling::ScopedPhase next_phase(xmvb::profiling::Phase::CiahNext);
    if (!valid(current_gradient))
      throw std::invalid_argument("invalid CIAH current gradient");
    Result result;
    if (basis_.empty()) {
      Vector seed = initial_direction_
          ? std::move(*initial_direction_)
          : precondition(negated(current_gradient), 0.0);
      if (!append(std::move(seed))) {
        Vector probe = backend_.ones();
        if (!append(std::move(probe)))
          throw std::runtime_error("CIAH could not construct an initial basis");
      }
      result.new_hvp_actions = 1;
      initial_direction_.reset();
    } else if (pending_residual_ && !exhausted_) {
      if (basis_size() >= std::min<int>(numerics_.max_cycle, backend_.size())) {
        exhausted_ = true;
      } else {
        Vector direction = precondition(*pending_residual_, pending_eigenvalue_);
        if (!append(std::move(direction), &*pending_residual_,
                    pending_residual_norm_, pending_eigenvalue_) &&
            !append(backend_.residual_direction(*pending_residual_),
                    &*pending_residual_, pending_residual_norm_,
                    pending_eigenvalue_))
          exhausted_ = true;
        else
          result.new_hvp_actions = 1;
      }
    }
    pending_residual_.reset();

    const bool trace = diagnostics::ciah_event_trace_enabled();
    const auto projection_start = trace ? std::chrono::steady_clock::now() :
        std::chrono::steady_clock::time_point{};
    const Eigen::Index m = basis_size();
    xmvb::profiling::ScopedPhase projection_phase(xmvb::profiling::Phase::NeoProjectedSolve);
    const Eigen::MatrixXd projected_hessian_action =
        hessian_projection_.topLeftCorner(m, m);
    const Eigen::MatrixXd projected_metric_action =
        metric_projection_.topLeftCorner(m, m);
    check_ciah_projected_symmetry(
        backend_.size(), std::sqrt(basis_squared_norm_),
        std::sqrt(hessian_squared_norm_), hessian_magnitude_.topLeftCorner(m, m),
        projected_hessian_action, "CIAH Hessian action is not symmetric",
        numerics_.operator_relative_accuracy);
    check_ciah_projected_symmetry(
        backend_.size(), std::sqrt(basis_squared_norm_),
        std::sqrt(metric_squared_norm_), metric_magnitude_.topLeftCorner(m, m),
        projected_metric_action, "CIAH metric action is not symmetric");
    const Eigen::MatrixXd projected_hessian = 0.5 *
        (projected_hessian_action + projected_hessian_action.transpose());
    const Eigen::MatrixXd projected_metric = 0.5 *
        (projected_metric_action + projected_metric_action.transpose());
    projected_metric_ = projected_metric;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> metric_spectrum(
        projected_metric, Eigen::EigenvaluesOnly);
    if (metric_spectrum.info() != Eigen::Success ||
        !(metric_spectrum.eigenvalues()[0] > numerics_.lindep *
          metric_spectrum.eigenvalues().tail(1)[0]))
      throw std::runtime_error(
          "CIAH projected metric is not positive definite or is rank deficient");
    Eigen::MatrixXd augmented_hessian = Eigen::MatrixXd::Zero(m + 1, m + 1);
    Eigen::VectorXd projected_gradient(m);
    for (Eigen::Index i = 0; i < m; ++i)
      projected_gradient[i] = backend_.dot(basis_[i], current_gradient);
    augmented_hessian.block(0, 1, 1, m) = projected_gradient.transpose();
    augmented_hessian.block(1, 0, m, 1) = projected_gradient;
    augmented_hessian.bottomRightCorner(m, m) = projected_hessian;
    Eigen::MatrixXd augmented_metric = Eigen::MatrixXd::Identity(m + 1, m + 1);
    augmented_metric.bottomRightCorner(m, m) = projected_metric;
    const auto spectrum_start = trace ? std::chrono::steady_clock::now() :
        std::chrono::steady_clock::time_point{};
    projection_phase.stop();
    xmvb::profiling::ScopedPhase spectrum_phase(xmvb::profiling::Phase::NeoSpectrum);
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(
        augmented_hessian, augmented_metric);
    if (spectrum.info() != Eigen::Success)
      throw std::runtime_error("CIAH augmented eigensolver failed");
    Eigen::Index selected = -1;
    for (Eigen::Index i = 0; i < spectrum.eigenvalues().size(); ++i)
      if (std::abs(spectrum.eigenvectors()(0, i)) > 0.1) {
        selected = i;
        break;
      }
    if (selected < 0)
      throw std::runtime_error(
          "CIAH found no augmented root with sufficient reference weight");
    const double alpha = spectrum.eigenvectors()(0, selected);
    const Eigen::VectorXd coefficients =
        spectrum.eigenvectors().col(selected).tail(m) / alpha;
    result.eigenvalue = spectrum.eigenvalues()[selected];
    result.reference_component = alpha;
    result.step = combine(basis_, coefficients);
    result.hessian_step = combine(hessian_images_, coefficients);
    result.metric_step = combine(metric_images_, coefficients);
    Vector model_gradient = backend_.clone(current_gradient);
    backend_.axpy(model_gradient, 1.0, result.hessian_step);
    result.residual = backend_.clone(model_gradient);
    backend_.axpy(result.residual, -result.eigenvalue, result.metric_step);
    backend_.scal(result.residual, alpha);
    spectrum_phase.stop();
    xmvb::profiling::ScopedPhase residual_phase(
        xmvb::profiling::Phase::NeoProjectionResidual);
    const auto norm_start = trace ? std::chrono::steady_clock::now() :
        std::chrono::steady_clock::time_point{};
    result.residual_norm = backend_.coordinate_norm(result.residual, false);
    result.model_gradient_norm = backend_.coordinate_norm(model_gradient, false);
    const bool eigenvalue_settled = previous_eigenvalue_ &&
        std::abs(result.eigenvalue - *previous_eigenvalue_) < numerics_.tolerance;
    if (trace) {
      Eigen::VectorXd projected_residual(m);
      for (Eigen::Index i = 0; i < m; ++i)
        projected_residual[i] = backend_.dot(basis_[i], result.residual);
      diagnostics::ciah_trace("ciah_ritz_cost",
          "basis=%d,projection_seconds=%.17g,spectrum_image_seconds=%.17g,norm_seconds=%.17g,gradient_euclidean=%.17g,residual_euclidean=%.17g,galerkin_residual=%.17g",
          basis_size(),
          std::chrono::duration<double>(spectrum_start - projection_start).count(),
          std::chrono::duration<double>(norm_start - spectrum_start).count(),
          std::chrono::duration<double>(std::chrono::steady_clock::now() - norm_start).count(),
          backend_.norm(current_gradient), backend_.norm(result.residual),
          projected_residual.norm());
    }
    result.converged = eigenvalue_settled &&
        result.residual_norm < std::sqrt(numerics_.tolerance);
    previous_eigenvalue_ = result.eigenvalue;
    result.basis_size = basis_size();
    result.exhaustion = exhausted_ ||
        basis_size() >= std::min<int>(numerics_.max_cycle, backend_.size());
    const double residual_scale = backend_.stable_norm(current_gradient) +
        backend_.stable_norm(result.hessian_step) +
        std::abs(result.eigenvalue) * backend_.stable_norm(result.metric_step);
    const bool residual_nonzero =
        result.residual_norm > numerics_.lindep * residual_scale;
    if (!residual_nonzero && !result.converged) result.exhaustion = true;
    if (!result.converged && !result.exhaustion && residual_nonzero) {
      pending_residual_ = backend_.clone(result.residual);
      pending_eigenvalue_ = result.eigenvalue;
      pending_residual_norm_ = result.residual_norm;
    }
    return result;
  }

private:
  bool valid(const Vector& vector) const {
    return backend_.dimension(vector) == backend_.size() &&
           backend_.all_finite(vector);
  }
  Vector negated(const Vector& vector) const {
    Vector result = backend_.clone(vector);
    backend_.scal(result, -1.0);
    return result;
  }
  Vector combine(const std::vector<Vector>& images,
                 const Eigen::VectorXd& coefficients) const {
    Vector result = backend_.zeros();
    for (Eigen::Index i = 0; i < coefficients.size(); ++i)
      backend_.axpy(result, coefficients[i], images[i]);
    return result;
  }
  Vector precondition(const Vector& residual, double eigenvalue) const {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::NeoPreconditioner);
    const bool trace = diagnostics::ciah_event_trace_enabled();
    const auto start = trace ? std::chrono::steady_clock::now() :
        std::chrono::steady_clock::time_point{};
    Vector direction = backend_.precondition(residual, eigenvalue);
    if (!valid(direction))
      throw std::runtime_error("CIAH preconditioner returned an invalid vector");
    if (trace) diagnostics::ciah_trace("ciah_precondition_cost",
        "basis=%d,seconds=%.17g", basis_size(),
        std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
    return direction;
  }

  bool append(Vector direction, const Vector* residual = nullptr,
              double residual_norm = 0.0, double shift = 0.0) {
    xmvb::profiling::ScopedPhase basis_phase(xmvb::profiling::Phase::NeoBasis);
    if (!valid(direction))
      throw std::invalid_argument("invalid CIAH Davidson direction");
    const double initial_norm = backend_.stable_norm(direction);
    if (!(initial_norm > 0.0)) return false;
    const bool trace = diagnostics::ciah_event_trace_enabled();
    const auto start = trace ? std::chrono::steady_clock::now() :
        std::chrono::steady_clock::time_point{};
    for (int pass = 0; pass < 2; ++pass)
      for (const Vector& vector : basis_)
        backend_.axpy(direction, -backend_.dot(vector, direction), vector);
    const double norm = backend_.stable_norm(direction);
    if (!(norm > numerics_.lindep * initial_norm) ||
        !(norm > 64.0 * std::numeric_limits<double>::epsilon() * initial_norm))
      return false;
    backend_.scal(direction, 1.0 / norm);
    basis_phase.stop();
    xmvb::profiling::ScopedPhase images_phase(
        xmvb::profiling::Phase::CiahImages, true);
    const auto metric_start = trace ? std::chrono::steady_clock::now() :
        std::chrono::steady_clock::time_point{};
    Vector metric_image;
    if (!backend_.has_joint_images())
      metric_image = backend_.apply_metric(direction);
    const auto hessian_start = trace ? std::chrono::steady_clock::now() :
        std::chrono::steady_clock::time_point{};
    Vector hessian_image;
    if (backend_.has_joint_images()) {
      auto images = backend_.apply_images(direction);
      hessian_image = std::move(images.first);
      metric_image = std::move(images.second);
      if (!valid(hessian_image) || !valid(metric_image))
        throw std::runtime_error("invalid joint CIAH operator images");
    } else {
      hessian_image = backend_.apply_hessian(direction);
    }
    images_phase.stop();
    if (diagnostics::ciah_trace_enabled() && residual) {
      const auto audit_start = std::chrono::steady_clock::now();
      Vector shifted_image = backend_.clone(hessian_image);
      backend_.axpy(shifted_image, -shift, metric_image);
      Vector error = precondition(shifted_image, shift);
      backend_.axpy(error, -1.0, direction);
      Vector complement = backend_.clone(error);
      for (int pass = 0; pass < 2; ++pass)
        for (const Vector& vector : basis_)
          backend_.axpy(complement,
                        -backend_.dot(vector, complement), vector);
      Vector error_metric = backend_.apply_metric(error);
      const double squared = backend_.dot(error, error_metric);
      const double q_squared = backend_.dot(direction, metric_image);
      diagnostics::ciah_trace("ciah_unlearned_inverse",
          "basis=%d,shift=%.17g,relative_physical_error=%.17g,error_euclidean=%.17g,complement_euclidean=%.17g,curvature=%.17g,seconds=%.17g",
          basis_size()+1, shift, std::sqrt(std::max(0.0,squared)/q_squared),
          backend_.norm(error), backend_.norm(complement),
          backend_.dot(direction, shifted_image),
          std::chrono::duration<double>(std::chrono::steady_clock::now()-audit_start).count());
    }
    backend_.observe_images(direction, metric_image);
    if (trace && residual && residual_norm > 0.0) {
      const double metric_squared = backend_.dot(direction, metric_image);
      const double work = backend_.dot(*residual, direction);
      const double curvature = backend_.dot(direction, hessian_image) -
                               shift * metric_squared;
      diagnostics::ciah_trace("ciah_expansion_geometry",
          "basis=%d,residual_coordinate_norm=%.17g,euclidean_cosine=%.17g,metric_squared=%.17g,shifted_curvature=%.17g,physical_rayleigh=%.17g,remaining_fraction_estimate=%.17g",
          basis_size()+1, residual_norm,
          work/(backend_.norm(*residual)*backend_.norm(direction)),
          metric_squared, curvature, curvature/metric_squared, norm/initial_norm);
    }
    if (trace) diagnostics::ciah_trace("ciah_expansion_cost",
        "basis=%d,remaining_fraction=%.17g,orthogonalize_seconds=%.17g,metric_seconds=%.17g,hvp_seconds=%.17g",
        basis_size()+1, norm/initial_norm,
        std::chrono::duration<double>(metric_start-start).count(),
        std::chrono::duration<double>(hessian_start-metric_start).count(),
        std::chrono::duration<double>(std::chrono::steady_clock::now()-hessian_start).count());
    basis_.push_back(std::move(direction));
    metric_images_.push_back(std::move(metric_image));
    hessian_images_.push_back(std::move(hessian_image));
    const auto projection_start = trace ? std::chrono::steady_clock::now() :
        std::chrono::steady_clock::time_point{};
    const Eigen::Index j = basis_size() - 1;
    xmvb::profiling::ScopedPhase projection_phase(
        xmvb::profiling::Phase::NeoProjectionFactor);
    std::vector<std::pair<const Vector*,const Vector*>> requests;
    requests.reserve(4*(j+1)+3);
    for (Eigen::Index i=0;i<=j;++i) {
      requests.emplace_back(&basis_[i],&hessian_images_[j]);
      requests.emplace_back(&basis_[j],&hessian_images_[i]);
      requests.emplace_back(&basis_[i],&metric_images_[j]);
      requests.emplace_back(&basis_[j],&metric_images_[i]);
    }
    requests.emplace_back(&basis_[j],&basis_[j]);
    requests.emplace_back(&hessian_images_[j],&hessian_images_[j]);
    requests.emplace_back(&metric_images_[j],&metric_images_[j]);
    const auto products=[&] {
      if constexpr (requires { backend_.products(requests); }) {
        return backend_.products(requests);
      } else {
        std::vector<CiahProduct> result;
        result.reserve(requests.size());
        for (const auto& [a,b]:requests)
          result.push_back({backend_.dot(*a,*b),backend_.abs_dot(*a,*b)});
        return result;
      }
    }();
    for (Eigen::Index i=0;i<=j;++i) {
      hessian_projection_(i,j)=products[4*i].value;
      hessian_projection_(j,i)=products[4*i+1].value;
      metric_projection_(i,j)=products[4*i+2].value;
      metric_projection_(j,i)=products[4*i+3].value;
      hessian_magnitude_(i,j)=products[4*i].magnitude;
      hessian_magnitude_(j,i)=products[4*i+1].magnitude;
      metric_magnitude_(i,j)=products[4*i+2].magnitude;
      metric_magnitude_(j,i)=products[4*i+3].magnitude;
    }
    basis_squared_norm_+=products[4*(j+1)].value;
    hessian_squared_norm_+=products[4*(j+1)+1].value;
    metric_squared_norm_+=products[4*(j+1)+2].value;
    if (trace) diagnostics::ciah_trace("ciah_projection_update_cost",
        "basis=%d,seconds=%.17g", basis_size(),
        std::chrono::duration<double>(std::chrono::steady_clock::now()-projection_start).count());
    return true;
  }

  Backend backend_;
  CiahNumerics numerics_;
  std::optional<Vector> initial_direction_, pending_residual_;
  std::optional<double> previous_eigenvalue_;
  double pending_eigenvalue_ = 0.0, pending_residual_norm_ = 0.0;
  std::vector<Vector> basis_, hessian_images_, metric_images_;
  Eigen::MatrixXd hessian_projection_, metric_projection_;
  Eigen::MatrixXd hessian_magnitude_, metric_magnitude_, projected_metric_;
  double basis_squared_norm_ = 0.0;
  double hessian_squared_norm_ = 0.0, metric_squared_norm_ = 0.0;
  bool exhausted_ = false;
};

}  // namespace xmvb::mcscf::detail
