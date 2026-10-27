#pragma once

#include "core/index.hpp"

#include <functional>
#include <cmath>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "mcscf/optimization/problem.hpp"

namespace xmvb::mcscf {

/** @brief Frozen curvature with an energy/gradient anchor refreshed at keyframes.
 *
 * Only displacement since the last keyframe belongs in its quadratic model.
 * The Hessian images remain reusable, as in PySCF's CIAH iteration.
 */
namespace detail {

inline constexpr double ciah_step_max_component = 0.03;

struct HostCiahVectorOps {
    using Vector = Eigen::VectorXd;
    Vector clone(const Vector& v) const {
        return v;
    }
    Vector zeros_like(const Vector& v) const {
        return Vector::Zero(v.size());
    }
    void axpy(Vector& y, double a, const Vector& x) const {
        y.noalias() += a * x;
    }
    void scal(Vector& x, double a) const {
        x *= a;
    }
    double dot(const Vector& x, const Vector& y) const {
        return x.dot(y);
    }
};

template <class Result>
bool ciah_result_resolved(const Result& result,
                          double rhs_norm,
                          double relative_accuracy) noexcept {
    const double alpha = std::abs(result.reference_component);
    return !result.negative_curvature && std::isfinite(result.residual_norm) &&
           result.residual_norm >= 0.0 && std::isfinite(alpha) && alpha > 0.0 &&
           std::isfinite(rhs_norm) && rhs_norm >= 0.0 && std::isfinite(relative_accuracy) &&
           relative_accuracy >= 0.0 && result.residual_norm <= alpha * relative_accuracy * rhs_norm;
}

template <class Ops> class CiahEnergyModelT {
  public:
    using Vector = typename Ops::Vector;
    CiahEnergyModelT(double energy, const Vector& gradient, Ops ops = {})
        : ops_(std::move(ops)), energy_(energy), gradient_(ops_.clone(gradient)),
          step_(ops_.zeros_like(gradient)), image_(ops_.zeros_like(gradient)) {}

    void add(const Vector& step, const Vector& image) {
        ops_.axpy(step_, 1.0, step);
        ops_.axpy(image_, 1.0, image);
    }

    void remove(const Vector& step, const Vector& image) {
        ops_.axpy(step_, -1.0, step);
        ops_.axpy(image_, -1.0, image);
    }

    void refresh(double energy, const Vector& gradient) {
        energy_ = energy;
        gradient_ = ops_.clone(gradient);
        step_ = ops_.zeros_like(gradient);
        image_ = ops_.zeros_like(gradient);
    }

    double energy() const {
        return energy_ + ops_.dot(gradient_, step_) + 0.5 * ops_.dot(step_, image_);
    }

    /** @brief An exact keyframe contradicts the segment's predicted descent.
   * Compare both energies to the last true anchor before refreshing it.
   * This stops extrapolation of an invalid model, not outer convergence.
   */
    bool descent_mismatch(double exact_energy) const {
        return energy() < energy_ && exact_energy >= energy_;
    }

    /** @brief Last true keyframe energy, without subsequent model increments. */
    double anchor_energy() const noexcept {
        return energy_;
    }

    /** @brief Incremental displacement since the last accepted keyframe. */
    const Vector& displacement() const noexcept {
        return step_;
    }

  private:
    Ops ops_;
    double energy_;
    Vector gradient_, step_, image_;
};

} // namespace detail

using CiahEnergyModel = detail::CiahEnergyModelT<detail::HostCiahVectorOps>;

/** Controls one bounded, matrix-free CIAH Davidson workspace. */
struct CiahOptions {
    /** @brief Joint H/G action when both images share physical intermediates. */
    std::function<std::pair<Eigen::VectorXd, Eigen::VectorXd>(const Eigen::VectorXd&)> apply_images;
    /** @brief Local coordinate dual norm; does not solve the physical metric. */
    std::function<double(const Eigen::VectorXd&)> coordinate_norm;
    /** @brief Convert a residual covector to a trial vector if no enrichment remains. */
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> residual_direction;
    /** Eigenvalue tolerance; the complete coordinate residual uses its square root. */
    double tolerance = 1.0e-12;
    /** Relative threshold for dependent directions and projected metric rank. */
    double lindep = 1.0e-14;
    /** Maximum number of admitted directions (and Hessian actions). */
    int max_cycle = 30;
    /** Declared relative error of the Hessian action for symmetry checks. */
    double operator_relative_accuracy = 0.0;
    /** @brief Observe a direction after its HVP, reusing the computed metric image. */
    std::function<void(const Eigen::VectorXd&, const Eigen::VectorXd&)> observe_images;
    /** Optional eigenvalue-shifted residual preconditioner. */
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, double)> residual_preconditioner;
};

/** A Ritz candidate, including its exact matrix-free augmented residual. */
struct CiahResult {
    double eigenvalue = 0.0;
    /** Signed reference component of the generalized-normalized Ritz vector. */
    double reference_component = 0.0;
    Eigen::VectorXd step;
    Eigen::VectorXd hessian_step;
    Eigen::VectorXd metric_step;
    /** AH eigenpair residual. For Newton steps: alpha * (g + H p - eigenvalue * G p).
     * For curvature rays: alpha * g + H p - eigenvalue * G p; p is the undivided tail.
     */
    Eigen::VectorXd residual;
    /** @brief Coordinate dual norm of the full residual, including outside the trial space. */
    double residual_norm = 0.0;
    /** @brief Coordinate dual norm of gradient + H * step, before clipping. */
    double model_gradient_norm = 0.0;
    bool converged = false;
    /** Undivided AH tail with negative curvature; not a resolved Newton step. */
    bool negative_curvature = false;
    int basis_size = 0;
    /** Hessian actions made by this call; zero or one. */
    int new_hvp_actions = 0;
    /** No further subspace expansion is available. */
    bool exhaustion = false;

    /** @brief Inexact solve of the current shifted AH equation.
   * Tests the full residual after removing its reference-component scale.
   * This does not certify a Newton macrostep or decrease of its gradient norm.
   */
    bool resolved(double rhs_norm, double relative_accuracy) const noexcept;
};

/** @brief Follow one clipped AH proposal without reevaluating its operator.
 * A true keyframe invalidates the remaining path. This is step advancement,
 * not a Ritz solve for the gradient at an intermediate point.
 * The borrowed proposal must outlive the path and retain its operator images.
 */
namespace detail {

template <class Result, class Ops> class CiahStepPathT {
  public:
    using Vector = typename Ops::Vector;
    explicit CiahStepPathT(Ops ops = {}) : ops_(std::move(ops)) {}
    void set(const Result& proposal) {
        proposal_ = &proposal;
        remaining_ = 1.0;
    }
    void clear() noexcept {
        remaining_ = 0.0;
    }
    bool pending() const noexcept {
        return remaining_ > 0.0;
    }
    double remaining_fraction() const noexcept {
        return remaining_;
    }
    const Vector& direction() const noexcept {
        return proposal_->step;
    }

    /** @brief Current AH residual of the unused step, using cached H/G images.
   * With s=remaining*p, choose omega=g^T*s to satisfy the reference row.
   * The remaining row is g+H*s-omega*G*s. Its source Ritz residual is not
   * a certificate after the gradient changes, even for a quadratic model.
   */
    Vector residual_at(const Vector& gradient) const {
        const double omega = remaining_ * ops_.dot(gradient, proposal_->step);
        Vector residual = ops_.clone(gradient);
        ops_.axpy(residual, remaining_, proposal_->hessian_step);
        ops_.axpy(residual, -omega * remaining_, proposal_->metric_step);
        return residual;
    }

    /** @brief Take a positive fraction, at most one, of the remaining path. */
    std::pair<Vector, Vector> take(double fraction) {
        const double part = fraction * remaining_;
        remaining_ = fraction < 1.0 ? remaining_ - part : 0.0;
        Vector step = ops_.clone(proposal_->step);
        Vector image = ops_.clone(proposal_->hessian_step);
        ops_.scal(step, part);
        ops_.scal(image, part);
        return {std::move(step), std::move(image)};
    }

  private:
    Ops ops_;
    const Result* proposal_ = nullptr;
    double remaining_ = 0.0;
};

} // namespace detail

using CiahStepPath = detail::CiahStepPathT<CiahResult, detail::HostCiahVectorOps>;

/** @brief Direction from the lowest origin-gradient AH root in the retained span.
 *
 * This is an unscaled direction, not a certified Ritz Newton step. The tail
 * is not divided by its reference component, which may be zero for a pure
 * negative-curvature direction. No convergence claim accompanies this ray.
 */
struct CiahOriginRay {
    Eigen::VectorXd step, hessian_step;
    /** Absolute reference weight of the generalized-normalized eigenvector. */
    double reference_component = 0.0;
    double eigenvalue = 0.0;
    /** Coordinate norm of the full augmented residual before clipping. */
    double residual_norm = 0.0;
};

/** @brief Trial-space inverse spectrum, diagnostic only, not a stopping rule. */
struct CiahInverseSpectrum {
    Eigen::Index dimension = 0;
    bool positive_curvature = false;
    double curvature_min = 0.0, curvature_max = 0.0;
    double inverse_min = 0.0, inverse_max = 0.0;
    double inverse_skew = 0.0;
};

/** @brief A true-energy keyframe and its frozen origin-chart images. */
template <class Vector> struct CiahCheckpointT {
    double energy = 0.0;
    Vector displacement, image, last_step;
    double residual_norm = 0.0, eigenvalue = 0.0;
    bool converged = false;
};

using CiahCheckpoint = CiahCheckpointT<Eigen::VectorXd>;

/** @brief Retain the lowest evaluated descending keyframe, without new actions. */
namespace detail {

template <class Result, class Ops> class CiahKeyframesT {
  public:
    using Vector = typename Ops::Vector;
    using Checkpoint = CiahCheckpointT<Vector>;
    explicit CiahKeyframesT(double source_energy, Ops ops = {})
        : ops_(std::move(ops)), lowest_energy_(source_energy) {}
    void save(double energy,
              const Vector& displacement,
              const Vector& image,
              const Vector& last_step,
              const Result& ritz) {
        if (!(energy < lowest_energy_))
            return;
        lowest_energy_ = energy;
        checkpoint_ = Checkpoint{energy,
                                 ops_.clone(displacement),
                                 ops_.clone(image),
                                 ops_.clone(last_step),
                                 ritz.residual_norm,
                                 ritz.eigenvalue,
                                 ritz.converged};
    }
    bool available() const noexcept {
        return checkpoint_.has_value();
    }
    /** @brief Consume a checkpoint so backtracking cannot retry it indefinitely. */
    Checkpoint take() {
        Checkpoint point = std::move(checkpoint_.value());
        checkpoint_.reset();
        return point;
    }

  private:
    Ops ops_;
    double lowest_energy_;
    std::optional<Checkpoint> checkpoint_;
};

} // namespace detail

using CiahKeyframes = detail::CiahKeyframesT<CiahResult, detail::HostCiahVectorOps>;

/**
 * Davidson iteration for the generalized augmented Hessian of one fixed model.
 *
 * The QuadraticProblem must outlive this workspace. Call `reset_rhs` before supplying
 * a changed gradient to `next`. All admitted directions and their Hessian/metric
 * images remain attached to the same immutable problem.
 */
class CiahWorkspace {
  public:
    ~CiahWorkspace();
    CiahWorkspace(const CiahWorkspace&);
    CiahWorkspace(CiahWorkspace&&) noexcept;
    CiahWorkspace& operator=(const CiahWorkspace&) = delete;
    /** @brief Dual norm captured by the current AH trial space.
   * Diagnostic projection using cached metric images; no operator evaluation.
   */
    double projected_dual_norm(const Eigen::VectorXd& covector) const;
    /** @brief Measure an inverse on cached shifted Hessian images.
   * Computes eigenvalues of (LQ)^T P (LQ) relative to Q^T LQ when the
   * latter is positive definite. Only small matrices and one inverse image
   * are allocated; no Hessian or metric actions are evaluated.
   */
    CiahInverseSpectrum
    inverse_spectrum(double shift,
                     const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& inverse) const;
    explicit CiahWorkspace(const QuadraticProblem& problem,
                           std::optional<Eigen::VectorXd> initial_direction = std::nullopt,
                           CiahOptions options = {});

    explicit CiahWorkspace(const QuadraticProblem& problem, CiahOptions options)
        : CiahWorkspace(problem, std::nullopt, std::move(options)) {}

    /** @brief Mark a changed RHS for a cached-space Ritz solve before expansion.
   * Retains every direction and its H/G images, but discards the old equation's
   * residual and eigenvalue-settling history. With a nonempty retained space,
   * the next call makes no new HVP.
   */
    void reset_rhs() noexcept;

    /** @brief Return the next AH root, enriching on the current RHS residual.
   * Call `reset_rhs` when the supplied gradient changes; that first call only
   * reprojects cached images. Subsequent calls may admit one HVP direction.
   */
    CiahResult next(const Eigen::VectorXd& current_gradient);

    /** @brief Probe outside a gradient-only span, stopping at the first negative witness.
     * No negative direction found does not certify global positive curvature.
     * Retains all generated images for subsequent macro globalization.
     */
    std::optional<CiahOriginRay> find_negative_curvature(const Eigen::VectorXd& gradient);

    /** @brief Re-solve the lowest AH root at an origin gradient using cached data.
   *
   * Requires a completed Ritz solve in the retained span. This does not
   * expand the basis, apply H or G, precondition, or evaluate a physical norm.
   * Returns no ray if that span contains no nonzero descent or negative-
   * curvature direction. Intended for one macro rejection recovery attempt,
   * not as an additional microiteration or convergence criterion.
   */
    std::optional<CiahOriginRay> origin_ray(const Eigen::VectorXd& origin_gradient) const;

    int basis_size() const noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace xmvb::mcscf
