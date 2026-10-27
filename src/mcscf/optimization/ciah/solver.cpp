#include "mcscf/optimization/ciah/solver.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

#include "mcscf/optimization/ciah/core.hpp"

namespace xmvb::mcscf {
namespace {

struct HostBackend {
    using Vector = Eigen::VectorXd;
    const QuadraticProblem& problem;
    CiahOptions options;

    Eigen::Index size() const noexcept {
        return problem.size();
    }
    Eigen::Index dimension(const Vector& v) const noexcept {
        return v.size();
    }
    bool all_finite(const Vector& v) const {
        return v.allFinite();
    }
    Vector zeros() const {
        return Vector::Zero(size());
    }
    Vector ones() const {
        return Vector::Ones(size());
    }
    Vector clone(const Vector& v) const {
        return v;
    }
    void axpy(Vector& y, double a, const Vector& x) const {
        y.noalias() += a * x;
    }
    void scal(Vector& x, double a) const {
        x *= a;
    }
    double dot(const Vector& a, const Vector& b) const {
        return a.dot(b);
    }
    double abs_dot(const Vector& a, const Vector& b) const {
        return a.cwiseAbs().dot(b.cwiseAbs());
    }
    double norm(const Vector& v) const {
        return v.norm();
    }
    double stable_norm(const Vector& v) const {
        return v.stableNorm();
    }
    bool has_joint_images() const noexcept {
        return static_cast<bool>(options.apply_images);
    }
    std::pair<Vector, Vector> apply_images(const Vector& v) const {
        return options.apply_images(v);
    }
    Vector apply_metric(const Vector& v) const {
        return problem.apply_metric(v);
    }
    Vector apply_hessian(const Vector& v) const {
        return problem.apply_hessian(v);
    }
    Vector precondition(const Vector& r, double lambda) const {
        return options.residual_preconditioner ? options.residual_preconditioner(r, lambda)
               : problem.has_preconditioner()  ? problem.apply_preconditioner(r)
                                               : r;
    }
    Vector residual_direction(const Vector& r) const {
        return options.residual_direction ? options.residual_direction(r) : r;
    }
    double coordinate_norm(const Vector& v, bool stable_default) const {
        return options.coordinate_norm ? options.coordinate_norm(v)
               : stable_default        ? v.stableNorm()
                                       : v.norm();
    }
    void observe_images(const Vector& v, const Vector& image) const {
        if (options.observe_images)
            options.observe_images(v, image);
    }
};

detail::CiahNumerics numerics(const CiahOptions& options) {
    return {
        options.tolerance, options.lindep, options.max_cycle, options.operator_relative_accuracy};
}

} // namespace

class CiahWorkspace::Impl {
  public:
    explicit Impl(const QuadraticProblem& problem,
                  std::optional<Eigen::VectorXd> initial,
                  CiahOptions options)
        : core(HostBackend{problem, options}, std::move(initial), numerics(options)) {}

    detail::CiahCore<HostBackend> core;
};

bool CiahResult::resolved(double rhs_norm, double relative_accuracy) const noexcept {
    return detail::ciah_result_resolved(*this, rhs_norm, relative_accuracy);
}

CiahWorkspace::CiahWorkspace(const QuadraticProblem& problem,
                             std::optional<Eigen::VectorXd> initial_direction,
                             CiahOptions options)
    : impl_(std::make_unique<Impl>(problem, std::move(initial_direction), std::move(options))) {}

CiahWorkspace::~CiahWorkspace() = default;
CiahWorkspace::CiahWorkspace(const CiahWorkspace& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}
CiahWorkspace::CiahWorkspace(CiahWorkspace&&) noexcept = default;

int CiahWorkspace::basis_size() const noexcept {
    return impl_->core.basis_size();
}

void CiahWorkspace::reset_rhs() noexcept {
    impl_->core.reset_rhs();
}

CiahResult CiahWorkspace::next(const Eigen::VectorXd& current_gradient) {
    auto value = impl_->core.next(current_gradient);
    CiahResult result;
    result.eigenvalue = value.eigenvalue;
    result.reference_component = value.reference_component;
    result.step = std::move(value.step);
    result.hessian_step = std::move(value.hessian_step);
    result.metric_step = std::move(value.metric_step);
    result.residual = std::move(value.residual);
    result.residual_norm = value.residual_norm;
    result.model_gradient_norm = value.model_gradient_norm;
    result.converged = value.converged;
    result.negative_curvature = value.negative_curvature;
    result.basis_size = value.basis_size;
    result.new_hvp_actions = value.new_hvp_actions;
    result.exhaustion = value.exhaustion;
    return result;
}

double CiahWorkspace::projected_dual_norm(const Eigen::VectorXd& covector) const {
    return impl_->core.projected_dual_norm(covector);
}

std::optional<CiahOriginRay>
CiahWorkspace::origin_ray(const Eigen::VectorXd& origin_gradient) const {
    auto value = impl_->core.origin_ray(origin_gradient);
    if (!value)
        return std::nullopt;
    CiahOriginRay ray;
    ray.step = std::move(value->step);
    ray.hessian_step = std::move(value->hessian_step);
    ray.reference_component = value->reference_component;
    ray.eigenvalue = value->eigenvalue;
    ray.residual_norm = value->residual_norm;
    return ray;
}

std::optional<CiahOriginRay>
CiahWorkspace::find_negative_curvature(const Eigen::VectorXd& gradient) {
    // Reproducible dense probe: independent of the gradient and orbital symmetry.
    std::mt19937_64 engine(0);
    std::normal_distribution<double> normal;
    Eigen::VectorXd probe(gradient.size());
    for (auto& entry : probe)
        entry = normal(engine);
    auto value = impl_->core.find_negative_curvature(std::move(probe), gradient);
    if (!value)
        return std::nullopt;
    CiahOriginRay ray;
    ray.step = std::move(value->step);
    ray.hessian_step = std::move(value->hessian_step);
    ray.reference_component = 0.0;
    ray.eigenvalue = value->eigenvalue;
    ray.residual_norm = value->residual_norm;
    return ray;
}

CiahInverseSpectrum CiahWorkspace::inverse_spectrum(
    double shift, const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& inverse) const {
    if (!std::isfinite(shift) || !inverse)
        throw std::invalid_argument("invalid CIAH inverse spectrum input");
    CiahInverseSpectrum result;
    const auto& core = impl_->core;
    const int m = core.basis_size();
    result.dimension = m;
    if (!m)
        return result;
    const Eigen::MatrixXd raw = core.hessian_projection().topLeftCorner(m, m) -
                                shift * core.metric_projection().topLeftCorner(m, m);
    const Eigen::MatrixXd e = 0.5 * (raw + raw.transpose());
    const Eigen::MatrixXd g = 0.5 * (core.metric_projection().topLeftCorner(m, m) +
                                     core.metric_projection().topLeftCorner(m, m).transpose());
    const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> curvature(
        e, g, Eigen::EigenvaluesOnly);
    if (curvature.info() != Eigen::Success)
        throw std::runtime_error("CIAH trial curvature spectrum failed");
    result.curvature_min = curvature.eigenvalues()[0];
    result.curvature_max = curvature.eigenvalues().tail(1)[0];
    result.positive_curvature = result.curvature_min > core.lindep() * result.curvature_max;
    if (!result.positive_curvature)
        return result;
    Eigen::MatrixXd t(m, m);
    for (int j = 0; j < m; ++j) {
        const Eigen::VectorXd rhs = core.hessian_images()[j] - shift * core.metric_images()[j];
        const Eigen::VectorXd image = inverse(rhs);
        if (image.size() != core.hessian_images()[j].size() || !image.allFinite())
            throw std::runtime_error("invalid CIAH diagnostic inverse image");
        for (int i = 0; i < m; ++i)
            t(i, j) =
                core.hessian_images()[i].dot(image) - shift * core.metric_images()[i].dot(image);
    }
    result.inverse_skew = (t - t.transpose()).norm() / std::max(1.0, t.norm());
    t = (0.5 * (t + t.transpose())).eval();
    const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(
        t, e, Eigen::EigenvaluesOnly);
    if (spectrum.info() != Eigen::Success)
        throw std::runtime_error("CIAH trial inverse spectrum failed");
    result.inverse_min = spectrum.eigenvalues()[0];
    result.inverse_max = spectrum.eigenvalues().tail(1)[0];
    return result;
}

} // namespace xmvb::mcscf
