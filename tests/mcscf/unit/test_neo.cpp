/** @file Common-solver regression; links without the VBSCF library. */
#include <algorithm>
#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include "mcscf/optimization/problem.hpp"
#include "mcscf/optimization/neo/solver.hpp"
#include "linalg/symmetry.hpp"

namespace {

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

void require_close(double actual, double reference, double tolerance, const std::string& message) {
    const double scale = std::max({1.0, std::abs(actual), std::abs(reference)});
    require(std::abs(actual - reference) <= tolerance * scale, message);
}

void require_close(const Eigen::VectorXd& actual,
                   const Eigen::VectorXd& reference,
                   double tolerance,
                   const std::string& message) {
    const double scale = std::max(1.0, reference.stableNorm());
    require((actual - reference).stableNorm() <= tolerance * scale, message);
}

double physical_dual_norm(const Eigen::MatrixXd& metric, const Eigen::VectorXd& covector) {
    return std::sqrt(covector.dot(metric.ldlt().solve(covector)));
}

xmvb::mcscf::QuadraticProblem
dense_problem(const Eigen::MatrixXd& hessian,
              const Eigen::MatrixXd& metric,
              const Eigen::VectorXd& gradient,
              std::optional<double> hessian_lower_bound = std::nullopt) {
    const Eigen::MatrixXd metric_inverse =
        metric.ldlt().solve(Eigen::MatrixXd::Identity(metric.rows(), metric.cols()));
    return xmvb::mcscf::QuadraticProblem(
        gradient,
        [hessian](const Eigen::VectorXd& vector) { return hessian * vector; },
        [metric](const Eigen::VectorXd& vector) { return metric * vector; },
        [metric_inverse](const Eigen::VectorXd& covector) {
            return std::sqrt(covector.dot(metric_inverse * covector));
        },
        {},
        hessian_lower_bound);
}

void verify_kkt(const std::string& name,
                const Eigen::MatrixXd& hessian,
                const Eigen::MatrixXd& metric,
                const Eigen::VectorXd& gradient,
                double radius,
                const xmvb::mcscf::NeoResult& result) {
    require(result.converged(), name + ": solver did not converge");
    require(result.step.allFinite(), name + ": non-finite step");
    require(result.shift >= 0.0 && std::isfinite(result.shift),
            name + ": invalid trust-region shift");

    const Eigen::VectorXd hessian_step = hessian * result.step;
    const Eigen::VectorXd metric_step = metric * result.step;
    const Eigen::VectorXd residual = gradient + hessian_step + result.shift * metric_step;
    const double step_norm = std::sqrt(result.step.dot(metric_step));
    const double residual_norm = physical_dual_norm(metric, residual);
    const double residual_scale =
        std::max({1.0,
                  physical_dual_norm(metric, gradient),
                  physical_dual_norm(metric, hessian_step),
                  result.shift * physical_dual_norm(metric, metric_step)});

    require(step_norm <= radius * (1.0 + 2.0e-10),
            name + ": step violates the physical trust region");
    require(residual_norm <= 2.0e-9 * residual_scale, name + ": KKT stationarity is not certified");
    require(result.curvature_residual_norm <= 2.0 * result.curvature_residual_target,
            name + ": lowest-root Ritz residual is not converged");
    require_close(result.hessian_step,
                  hessian_step,
                  2.0e-10,
                  name + ": returned Hessian image is inconsistent");
    require_close(
        result.metric_step, metric_step, 2.0e-10, name + ": returned metric image is inconsistent");
    require_close(
        result.kkt_residual, residual, 2.0e-10, name + ": returned KKT residual is inconsistent");
    require_close(
        result.step_norm, step_norm, 2.0e-10, name + ": returned physical norm is inconsistent");
    require_close(result.residual_norm,
                  residual_norm,
                  2.0e-9,
                  name + ": returned residual norm is inconsistent");

    const double predicted = -gradient.dot(result.step) - 0.5 * result.step.dot(hessian_step);
    require_close(result.predicted_reduction,
                  predicted,
                  2.0e-10,
                  name + ": predicted reduction is inconsistent");
    require(predicted >= -2.0e-12 * residual_scale,
            name + ": NEO step increases its quadratic model");

    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> shifted_spectrum(
        hessian + result.shift * metric, metric);
    require(shifted_spectrum.info() == Eigen::Success,
            name + ": shifted generalized eigensystem failed");
    const double spectral_scale =
        std::max(1.0, shifted_spectrum.eigenvalues().cwiseAbs().maxCoeff());
    require(shifted_spectrum.eigenvalues().minCoeff() >= -2.0e-10 * spectral_scale,
            name + ": shifted Hessian is not positive semidefinite");

    if (result.shift > 2.0e-10 * spectral_scale) {
        require(std::abs(step_norm - radius) <= 2.0e-9 * radius,
                name + ": complementarity fails on a shifted step");
        require(result.boundary, name + ": boundary step was not classified");
    }
}

void check_positive_definite_newton_limit() {
    Eigen::Matrix2d hessian;
    hessian << 2.0, 0.5, 0.5, 4.0;
    const Eigen::Matrix2d metric = Eigen::Matrix2d::Identity();
    const Eigen::Vector2d gradient(1.0, -2.0);
    constexpr double radius = 2.0;

    xmvb::mcscf::NeoOptions options;
    options.trust_radius = radius;
    options.relative_residual_tolerance = 1.0e-12;
    const xmvb::mcscf::QuadraticProblem problem = dense_problem(hessian, metric, gradient);
    require(!problem.has_preconditioner(),
            "dense reference unexpectedly installed a preconditioner");
    const xmvb::mcscf::NeoResult result = xmvb::mcscf::solve_neo(problem, options);
    verify_kkt("positive-definite Newton limit", hessian, metric, gradient, radius, result);

    const Eigen::Vector2d newton = -hessian.ldlt().solve(gradient);
    require_close(result.step, newton, 2.0e-10, "interior NEO step differs from the Newton step");
    require(result.shift <= 2.0e-10, "interior positive-definite problem has a nonzero shift");
    require(!result.boundary, "interior positive-definite step was classified as a boundary step");
    require(!result.augmented_certificate_valid,
            "interior Newton limit was reported as a finite-alpha certificate");
}

void check_generalized_metric_boundary() {
    Eigen::Matrix2d hessian;
    hessian << 3.5, 0.7, 0.7, 2.0;
    Eigen::Matrix2d metric;
    metric << 2.0, 0.35, 0.35, 0.8;
    const Eigen::Vector2d gradient(1.2, -0.9);
    constexpr double radius = 0.18;

    xmvb::mcscf::NeoOptions options;
    options.trust_radius = radius;
    options.relative_residual_tolerance = 1.0e-12;
    const xmvb::mcscf::QuadraticProblem problem = dense_problem(hessian, metric, gradient);
    require(!problem.has_preconditioner(),
            "generalized-metric reference unexpectedly requires a preconditioner");
    const xmvb::mcscf::NeoResult result = xmvb::mcscf::solve_neo(problem, options);
    verify_kkt("generalized-metric boundary", hessian, metric, gradient, radius, result);
    require(result.shift > 0.0, "small generalized trust region did not produce a shift");
    require(result.augmented_certificate_valid,
            "regular boundary step lacks its augmented certificate");

    // At the returned gradient scale, the NEO step must come from the lowest
    // root of the explicit augmented generalized eigenproblem.
    require(result.gradient_scale > 0.0, "boundary NEO result has no positive gradient scale");
    Eigen::Matrix3d augmented = Eigen::Matrix3d::Zero();
    augmented.block<1, 2>(0, 1) = result.gradient_scale * gradient.transpose();
    augmented.block<2, 1>(1, 0) = result.gradient_scale * gradient;
    augmented.bottomRightCorner<2, 2>() = hessian;
    Eigen::Matrix3d augmented_metric = Eigen::Matrix3d::Zero();
    augmented_metric(0, 0) = 1.0;
    augmented_metric.bottomRightCorner<2, 2>() = metric;
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(augmented,
                                                                          augmented_metric);
    require(eigensolver.info() == Eigen::Success, "explicit augmented eigensystem failed");
    require_close(result.augmented_eigenvalue,
                  eigensolver.eigenvalues().minCoeff(),
                  2.0e-9,
                  "NEO did not select the lowest augmented root");
    require_close(result.augmented_eigenvalue,
                  -result.shift,
                  2.0e-9,
                  "augmented root and KKT shift have inconsistent signs");

    Eigen::Vector3d reconstructed;
    reconstructed[0] = 1.0;
    reconstructed.tail<2>() = result.gradient_scale * result.step;
    const Eigen::Vector3d augmented_residual =
        augmented * reconstructed - result.augmented_eigenvalue * augmented_metric * reconstructed;
    require(augmented_residual.stableNorm() <= 2.0e-9 * std::max(1.0, augmented.norm()),
            "returned step does not satisfy the augmented eigenproblem");
}

void check_indefinite_regular_boundary() {
    Eigen::Matrix2d hessian;
    hessian << -1.4, 0.25, 0.25, 2.2;
    const Eigen::Matrix2d metric = Eigen::Matrix2d::Identity();
    const Eigen::Vector2d gradient(0.8, -0.45);
    constexpr double radius = 0.6;

    xmvb::mcscf::NeoOptions options;
    options.trust_radius = radius;
    options.relative_residual_tolerance = 1.0e-12;
    const xmvb::mcscf::NeoResult result =
        xmvb::mcscf::solve_neo(dense_problem(hessian, metric, gradient), options);
    verify_kkt("indefinite regular boundary", hessian, metric, gradient, radius, result);
    require(result.shift > 0.0, "indefinite problem did not stabilize its Hessian");
    require(!result.hard_case, "regular indefinite problem was classified as a hard case");
    require(result.augmented_certificate_valid,
            "regular indefinite boundary lacks its augmented certificate");
}

void check_hard_case() {
    Eigen::Matrix2d hessian;
    hessian << -2.0, 0.0, 0.0, 3.0;
    const Eigen::Matrix2d metric = Eigen::Matrix2d::Identity();
    const Eigen::Vector2d gradient(0.0, 1.0);
    constexpr double radius = 1.0;

    xmvb::mcscf::NeoOptions options;
    options.trust_radius = radius;
    options.relative_residual_tolerance = 1.0e-12;
    const xmvb::mcscf::NeoResult result =
        xmvb::mcscf::solve_neo(dense_problem(hessian, metric, gradient), options);
    verify_kkt("hard case", hessian, metric, gradient, radius, result);
    require(result.hard_case, "hard case was not detected");
    require(!result.augmented_certificate_valid,
            "hard-case curvature vector was reported as a complete NEO step");
    require_close(result.shift, 2.0, 2.0e-10, "hard case has the wrong spectral shift");
    require_close(result.step[1], -0.2, 2.0e-10, "hard-case pseudoinverse component is wrong");
    require_close(std::abs(result.step[0]),
                  std::sqrt(0.96),
                  2.0e-10,
                  "hard-case minimum-mode component is wrong");
}

Eigen::Matrix3d hidden_curvature_hessian() {
    const Eigen::Vector3d visible(0.0, -4.0 / 3.0, 5.0 / 3.0);
    const Eigen::Vector3d hidden(0.0, 5.0 / 3.0, 4.0 / 3.0);
    Eigen::Vector3d gradient_mode = Eigen::Vector3d::Zero();
    gradient_mode[0] = 1.0;
    return 2.0 * gradient_mode * gradient_mode.transpose() +
           3.0 * visible * visible.transpose() / visible.squaredNorm() -
           100.0 * hidden * hidden.transpose() / hidden.squaredNorm();
}

void check_hidden_negative_curvature() {
    const Eigen::Matrix3d hessian = hidden_curvature_hessian();
    const Eigen::Matrix3d metric = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d gradient(1.0, 0.0, 0.0);

    xmvb::mcscf::NeoOptions budgeted_options;
    budgeted_options.trust_radius = 1.0;
    budgeted_options.relative_residual_tolerance = 1.0e-12;
    budgeted_options.maximum_subspace_dimension = 2;
    const xmvb::mcscf::NeoResult budgeted =
        xmvb::mcscf::solve_neo(dense_problem(hessian, metric, gradient), budgeted_options);
    require(budgeted.stop_reason == xmvb::mcscf::NeoStopReason::SubspaceLimit,
            "an incomplete subspace falsely certified hidden curvature");
    require(!budgeted.converged(), "a work-limited hidden-curvature solve reported convergence");
    require(budgeted.hessian_actions == 2,
            "the explicit subspace budget used the wrong number of actions");

    xmvb::mcscf::NeoOptions complete_options = budgeted_options;
    complete_options.maximum_subspace_dimension = 0;
    const xmvb::mcscf::NeoResult complete =
        xmvb::mcscf::solve_neo(dense_problem(hessian, metric, gradient), complete_options);
    verify_kkt("hidden negative curvature", hessian, metric, gradient, 1.0, complete);
    require(complete.hessian_actions == 3,
            "the full solve did not expose the third curvature mode");
    require_close(complete.shift, 100.0, 2.0e-10, "hidden negative curvature has the wrong shift");
    require(complete.hard_case, "gradient-orthogonal hidden curvature was not a hard case");
    require(complete.global_curvature_certified,
            "a complete basis lacks its global curvature certificate");
}

void check_zero_gradient_curvature_probe() {
    const Eigen::Matrix3d hessian = hidden_curvature_hessian();
    Eigen::Matrix3d metric = Eigen::Matrix3d::Zero();
    metric.diagonal() << 2.0, 3.0, 4.0;
    const Eigen::Vector3d first_order_gradient(1.0, 0.0, 0.0);
    const Eigen::Vector3d hidden_mode(0.0, 5.0 / 3.0, 4.0 / 3.0);
    require_close(first_order_gradient.dot(hidden_mode),
                  0.0,
                  0.0,
                  "hidden curvature is not orthogonal to the first-order gradient");

    xmvb::mcscf::NeoOptions options;
    options.trust_radius = 1.0;
    options.relative_residual_tolerance = 1.0e-12;
    options.maximum_subspace_dimension = 3;
    options.stop_on_negative_curvature = true;
    const Eigen::Vector3d zero_gradient = Eigen::Vector3d::Zero();
    const xmvb::mcscf::NeoResult result =
        xmvb::mcscf::solve_neo(dense_problem(hessian, metric, zero_gradient), options);
    require(result.stop_reason == xmvb::mcscf::NeoStopReason::NegativeCurvature,
            "zero-gradient probe missed hidden negative curvature");
    require(!result.converged(), "curvature witness was mislabeled as KKT convergence");
    require(result.hessian_actions <= 3, "curvature probe exceeded its action budget");
    const Eigen::VectorXd& direction = result.minimum_curvature_direction;
    require_close(direction.dot(metric * direction),
                  1.0,
                  2.0e-10,
                  "curvature witness is not metric-normalized");
    require_close(result.minimum_curvature_hessian_image,
                  hessian * direction,
                  2.0e-10,
                  "curvature witness has the wrong Hessian image");
    require_close(result.minimum_curvature_metric_image,
                  metric * direction,
                  2.0e-10,
                  "curvature witness has the wrong metric image");
    require_close(result.minimum_curvature_value,
                  direction.dot(hessian * direction) / direction.dot(metric * direction),
                  2.0e-10,
                  "curvature witness has the wrong Rayleigh quotient");
    require(result.minimum_curvature_value < -1.0, "curvature witness is not decisively negative");
    require_close(result.hessian_step,
                  hessian * result.step,
                  2.0e-10,
                  "curvature probe lost the spectral step Hessian image");
    require_close(result.metric_step,
                  metric * result.step,
                  2.0e-10,
                  "curvature probe lost the spectral step metric image");

    bool rejected = false;
    try {
        (void)xmvb::mcscf::solve_neo(dense_problem(hessian, metric, first_order_gradient), options);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "curvature-only stopping accepted a nonzero Newton gradient");
}

void check_positive_zero_gradient_curvature_probe() {
    Eigen::Matrix2d hessian;
    hessian << 2.0, 0.5, 0.5, 4.0;
    Eigen::Matrix2d metric;
    metric << 2.0, 0.25, 0.25, 3.0;
    const Eigen::Vector2d gradient = Eigen::Vector2d::Zero();
    xmvb::mcscf::NeoOptions options;
    options.trust_radius = 1.0;
    options.relative_residual_tolerance = 1.0e-12;
    options.stop_on_negative_curvature = true;
    const xmvb::mcscf::NeoResult result =
        xmvb::mcscf::solve_neo(dense_problem(hessian, metric, gradient), options);
    verify_kkt("positive zero-gradient curvature probe",
               hessian,
               metric,
               gradient,
               options.trust_radius,
               result);
    require(result.stop_reason == xmvb::mcscf::NeoStopReason::Converged,
            "positive curvature triggered negative-curvature stopping");
    require(result.minimum_curvature_value > 0.0,
            "positive probe returned a nonpositive Ritz value");
    require_close(result.minimum_curvature_value,
                  result.minimum_curvature_direction.dot(result.minimum_curvature_hessian_image),
                  2.0e-10,
                  "positive probe returned an inconsistent Ritz value");
}

/** @brief A negative quadratic form does not require an eigenvector residual. */
void check_negative_witness_without_spectral_refinement() {
    Eigen::Matrix3d hessian = Eigen::Matrix3d::Zero();
    hessian.diagonal() << -30.0, 100.0, 200.0;
    const Eigen::Matrix3d metric = Eigen::Matrix3d::Identity();
    xmvb::mcscf::NeoOptions options;
    options.relative_residual_tolerance = 1.0e-12;
    options.stop_on_negative_curvature = true;
    const xmvb::mcscf::NeoResult result =
        xmvb::mcscf::solve_neo(dense_problem(hessian, metric, Eigen::Vector3d::Zero()), options);
    require(result.stop_reason == xmvb::mcscf::NeoStopReason::NegativeCurvature,
            "negative quadratic form was not recognized");
    require(result.hessian_actions == 1,
            "negative witness unnecessarily refined an extremal eigenvector");
    require(result.curvature_residual_norm > std::abs(result.minimum_curvature_value),
            "fixture no longer distinguishes a witness from an eigenvector");
    require(!result.global_curvature_certified && !result.converged(),
            "partial negative witness claimed a complete Newton or spectral certificate");
    const auto& v = result.minimum_curvature_direction;
    require(v.dot(hessian * v) < 0.0,
            "reported witness does not have independently negative curvature");
    require_close(result.minimum_curvature_hessian_image,
                  hessian * v,
                  1.0e-12,
                  "negative witness has an inconsistent Hessian image");
}

void check_certified_lower_bounds() {
    const Eigen::Matrix3d hessian = hidden_curvature_hessian();
    const Eigen::Matrix3d metric = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d gradient(1.0, 0.0, 0.0);

    xmvb::mcscf::NeoOptions options;
    options.trust_radius = 1.0 / 102.0;
    options.relative_residual_tolerance = 1.0e-12;
    options.maximum_subspace_dimension = 1;

    const xmvb::mcscf::NeoResult tight =
        xmvb::mcscf::solve_neo(dense_problem(hessian, metric, gradient, -100.0), options);
    verify_kkt(
        "tight certified lower bound", hessian, metric, gradient, options.trust_radius, tight);
    require(tight.hessian_actions == 1,
            "a tight certified bound did not avoid the hidden-mode action");
    require_close(tight.shift, 100.0, 2.0e-10, "tight lower-bound solve has the wrong shift");

    const xmvb::mcscf::NeoResult conservative =
        xmvb::mcscf::solve_neo(dense_problem(hessian, metric, gradient, -101.0), options);
    require(conservative.stop_reason == xmvb::mcscf::NeoStopReason::SubspaceLimit,
            "an insufficient lower bound falsely certified global curvature");
    require(!conservative.converged(), "an insufficient lower bound reported convergence");
}

void check_scalable_lowest_root_convergence() {
    constexpr int dimension = 40;
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Identity(dimension, dimension);
    hessian.diagonal().head(20).array() = 2.0;
    hessian.diagonal().tail(19).array() = 4.0;
    hessian(dimension - 1, dimension - 1) = -3.0;
    const Eigen::MatrixXd metric = Eigen::MatrixXd::Identity(dimension, dimension);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(dimension);
    gradient.head(20).setOnes();

    xmvb::mcscf::NeoOptions options;
    options.trust_radius = 0.25;
    options.relative_residual_tolerance = 1.0e-11;
    options.maximum_subspace_dimension = 4;
    const xmvb::mcscf::NeoResult result =
        xmvb::mcscf::solve_neo(dense_problem(hessian, metric, gradient), options);
    verify_kkt("scalable lowest-root convergence",
               hessian,
               metric,
               gradient,
               options.trust_radius,
               result);
    require(result.hessian_actions < dimension,
            "lowest-root convergence unnecessarily built the full space");
    require(!result.global_curvature_certified,
            "an incomplete Davidson space claimed a rigorous global certificate");
}

void check_optional_curvature_certificate() {
    Eigen::Matrix3d hessian = Eigen::Matrix3d::Zero();
    hessian.diagonal() << 2.0, 3.0, 4.0;
    const Eigen::Matrix3d metric = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d gradient(1.0, 0.0, 0.0);

    xmvb::mcscf::NeoOptions kkt_options;
    kkt_options.trust_radius = 1.0;
    kkt_options.relative_residual_tolerance = 1.0e-12;
    kkt_options.require_curvature_certificate = false;
    const xmvb::mcscf::NeoResult kkt =
        xmvb::mcscf::solve_neo(dense_problem(hessian, metric, gradient), kkt_options);
    require(kkt.converged(), "KKT-only interior solve did not converge");
    require(kkt.hessian_actions == 1,
            "KKT-only interior solve evaluated unrelated curvature modes");
    require_close(kkt.step,
                  Eigen::Vector3d(-0.5, 0.0, 0.0),
                  2.0e-10,
                  "KKT-only interior solve returned the wrong Newton step");

    xmvb::mcscf::NeoOptions certified_options = kkt_options;
    certified_options.require_curvature_certificate = true;
    const xmvb::mcscf::NeoResult certified =
        xmvb::mcscf::solve_neo(dense_problem(hessian, metric, gradient), certified_options);
    verify_kkt("explicit curvature certificate",
               hessian,
               metric,
               gradient,
               certified_options.trust_radius,
               certified);
    require(certified.hessian_actions > kkt.hessian_actions,
            "explicit curvature certification did not inspect extra modes");
}

void check_anisotropic_dual_residual() {
    Eigen::Matrix2d hessian = Eigen::Matrix2d::Zero();
    hessian.diagonal() << 1.0, 1.0e4;
    Eigen::Matrix2d metric = Eigen::Matrix2d::Zero();
    metric.diagonal() << 1.0e-4, 1.0;
    const Eigen::Vector2d gradient(1.0e-4, 1.0e-4);
    const Eigen::Vector2d newton(-1.0e-4, -1.0e-8);

    xmvb::mcscf::NeoOptions options;
    options.trust_radius = 1.0;
    options.absolute_residual_tolerance = 1.0e-3;
    options.relative_residual_tolerance = 1.0e-8;
    options.require_curvature_certificate = false;
    const xmvb::mcscf::NeoResult result =
        xmvb::mcscf::solve_neo(dense_problem(hessian, metric, gradient), options);

    require(result.converged(), "anisotropic NEO solve did not converge");
    require(result.hessian_actions >= 2, "anisotropic NEO accepted its incomplete first direction");
    require_close(result.step, newton, 1.0e-10, "anisotropic NEO missed the full Newton step");
    const Eigen::Vector2d residual = gradient + hessian * result.step;
    require(physical_dual_norm(metric, residual) <= result.residual_target,
            "anisotropic NEO physical residual exceeds its target");
    require_close(result.residual_norm,
                  physical_dual_norm(metric, residual),
                  1.0e-12,
                  "anisotropic NEO reported a coordinate residual norm");
}

void check_coordinate_rescaling_invariance() {
    Eigen::Matrix2d hessian = Eigen::Matrix2d::Zero();
    hessian.diagonal() << 1.0, 1.0e4;
    Eigen::Matrix2d metric = Eigen::Matrix2d::Zero();
    metric.diagonal() << 1.0e-4, 1.0;
    const Eigen::Vector2d gradient(1.0e-4, 1.0e-4);
    Eigen::Matrix2d coordinates = Eigen::Matrix2d::Zero();
    coordinates.diagonal() << 0.1, 10.0;

    xmvb::mcscf::NeoOptions options;
    options.trust_radius = 1.0;
    options.absolute_residual_tolerance = 1.0e-3;
    options.relative_residual_tolerance = 1.0e-8;
    options.require_curvature_certificate = false;
    const xmvb::mcscf::NeoResult original =
        xmvb::mcscf::solve_neo(dense_problem(hessian, metric, gradient), options);
    const xmvb::mcscf::NeoResult rescaled =
        xmvb::mcscf::solve_neo(dense_problem(coordinates.transpose() * hessian * coordinates,
                                             coordinates.transpose() * metric * coordinates,
                                             coordinates.transpose() * gradient),
                               options);

    require(original.converged() && rescaled.converged(),
            "coordinate-rescaled NEO solve did not converge");
    require(rescaled.hessian_actions >= 2,
            "coordinate-rescaled NEO accepted its incomplete first direction");
    const Eigen::Vector2d physical_step = coordinates * rescaled.step;
    require_close(physical_step,
                  original.step,
                  1.0e-10,
                  "coordinate rescaling changed the physical NEO step");
    require_close(physical_step,
                  Eigen::Vector2d(-1.0e-4, -1.0e-8),
                  1.0e-10,
                  "coordinate-rescaled NEO missed the full Newton step");
    const Eigen::Vector2d physical_residual = gradient + hessian * physical_step;
    require(physical_dual_norm(metric, physical_residual) <= rescaled.residual_target,
            "coordinate-rescaled NEO residual fails in physical units");
}

void check_boundary_requires_curvature_certificate() {
    Eigen::Matrix2d hessian;
    hessian << -1.0, 0.5, 0.5, 2.0;
    const Eigen::Matrix2d metric = Eigen::Matrix2d::Identity();
    const Eigen::Vector2d gradient(1.0, 0.0);

    xmvb::mcscf::NeoOptions options;
    options.trust_radius = 0.1;
    options.relative_residual_tolerance = 1.0e-12;
    options.require_curvature_certificate = false;
    const xmvb::mcscf::NeoResult result =
        xmvb::mcscf::solve_neo(dense_problem(hessian, metric, gradient), options);
    verify_kkt("automatic boundary curvature certificate",
               hessian,
               metric,
               gradient,
               options.trust_radius,
               result);
    require(result.boundary, "negative-curvature test did not produce a boundary step");
    require(result.hessian_actions == 2, "boundary solve did not expand the curvature residual");
}

void check_projection_roundoff() {
    constexpr Eigen::Index n = 37879;
    Eigen::MatrixXd q(n, 2);
    q.col(0).setConstant(1.0 / std::sqrt(static_cast<double>(n)));
    q.col(1).setZero();
    for (Eigen::Index i = 0; i < n - 1; ++i) {
        q(i, 1) = (i % 2 == 0 ? 1.0 : -1.0) / std::sqrt(static_cast<double>(n - 1));
    }
    Eigen::MatrixXd images = 5.0 * q;
    images.col(1) += 1.14e-12 * q.col(0);
    const Eigen::MatrixXd projected = q.transpose() * images;
    const double bound = xmvb::linalg::projection_skew_roundoff(q, images, projected);
    require((projected - projected.transpose()).norm() < bound,
            "large-dimensional projection rejects a roundoff-sized defect");
    const xmvb::mcscf::QuadraticProblem problem(
        (q.col(0) + q.col(1)).eval(),
        [&q](const Eigen::VectorXd& v) -> Eigen::VectorXd {
            return 5.0 * v + 4.0 * q.col(1).dot(v) * q.col(1) +
                   1.14e-12 * q.col(1).dot(v) * q.col(0);
        },
        [](const Eigen::VectorXd& v) -> Eigen::VectorXd { return v; },
        [](const Eigen::VectorXd& v) { return v.norm(); });
    xmvb::mcscf::NeoOptions options;
    options.require_curvature_certificate = false;
    options.relative_residual_tolerance = 1e-10;
    const xmvb::mcscf::NeoResult result = xmvb::mcscf::solve_neo(problem, options);
    require(result.converged(), "large-dimensional roundoff regression did not converge");
    images.col(1) += 1e-4 * q.col(0);
    const Eigen::MatrixXd invalid = q.transpose() * images;
    require((invalid - invalid.transpose()).norm() >
                xmvb::linalg::projection_skew_roundoff(q, images, invalid),
            "roundoff bound hides a materially nonsymmetric action");
}

void check_core_early_trial_preserves_certificate() {
    const Eigen::MatrixXd hessian = Eigen::Vector4d(1, 2, 4, 8).asDiagonal();
    const Eigen::MatrixXd metric = 1e-4 * Eigen::MatrixXd::Identity(4, 4);
    const Eigen::VectorXd gradient = Eigen::VectorXd::Ones(4);
    const auto problem = dense_problem(hessian, metric, gradient);
    xmvb::mcscf::NeoOptions options;
    options.trust_radius = 10;
    options.relative_residual_tolerance = 1e-12;
    options.require_curvature_certificate = false;
    const auto early = xmvb::mcscf::solve_neo(problem, options, true);
    require(early.stop_reason == xmvb::mcscf::NeoStopReason::TrialReady && !early.converged(),
            "core candidate falsely claims physical convergence");
    require(early.residual_norm > early.residual_target,
            "core early-trial fixture already satisfies the physical certificate");
    const auto final = xmvb::mcscf::solve_neo(problem, options);
    require(final.converged() && final.residual_norm <= final.residual_target,
            "disabling early core trials lost physical KKT certification");
    require(early.hessian_actions < final.hessian_actions,
            "core candidate did not avoid strict-solve HVPs");
    require_close(final.step,
                  -hessian.ldlt().solve(gradient),
                  1e-10,
                  "core candidate policy changed the certified Newton solution");
    options.require_curvature_certificate = true;
    require(xmvb::mcscf::solve_neo(problem, options, true).converged(),
            "core trial bypassed an explicitly requested curvature certificate");
}

void check_core_early_trial_rejects_absolute_floor() {
    const Eigen::MatrixXd hessian = Eigen::Vector4d(1, 2, 4, 8).asDiagonal();
    const Eigen::MatrixXd metric = 1e-4 * Eigen::MatrixXd::Identity(4, 4);
    const Eigen::VectorXd gradient = 2e-4 * Eigen::VectorXd::Ones(4);
    xmvb::mcscf::NeoOptions options;
    options.trust_radius = 10.0;
    options.relative_residual_tolerance = 0.1;
    options.absolute_residual_tolerance = 1e-3;
    options.require_curvature_certificate = false;
    const auto step =
        xmvb::mcscf::solve_neo(dense_problem(hessian, metric, gradient), options, true);
    require(step.converged() || step.stop_reason == xmvb::mcscf::NeoStopReason::TrialReady,
            "compressed-gradient candidate failed");
    require(step.hessian_actions > 1,
            "absolute coordinate floor admitted an unresolved first direction");
    require(step.residual_norm < physical_dual_norm(metric, gradient),
            "candidate did not reduce its physical residual");
}

void check_nonsymmetric_hessian_rejected() {
    Eigen::Matrix2d hessian;
    hessian << 2.0, 1.0, 0.0, 3.0;
    const Eigen::Matrix2d metric = Eigen::Matrix2d::Identity();
    const Eigen::Vector2d gradient(1.0, -0.5);
    xmvb::mcscf::NeoOptions options;
    options.relative_residual_tolerance = 1.0e-12;

    bool rejected = false;
    try {
        (void)xmvb::mcscf::solve_neo(dense_problem(hessian, metric, gradient), options);
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        require(message.find("skew_norm=") != std::string::npos &&
                    message.find("threshold=") != std::string::npos &&
                    message.find("repeated_bilinear_skew=") != std::string::npos,
                "nonsymmetric Hessian failure lacks reproducibility diagnostics");
        rejected = true;
    }
    require(rejected, "materially nonsymmetric Hessian action was accepted");
}

} // namespace

int main() {
    try {
        check_positive_definite_newton_limit();
        check_generalized_metric_boundary();
        check_indefinite_regular_boundary();
        check_hard_case();
        check_hidden_negative_curvature();
        check_zero_gradient_curvature_probe();
        check_negative_witness_without_spectral_refinement();
        check_positive_zero_gradient_curvature_probe();
        check_certified_lower_bounds();
        check_scalable_lowest_root_convergence();
        check_optional_curvature_certificate();
        check_anisotropic_dual_residual();
        check_coordinate_rescaling_invariance();
        check_boundary_requires_curvature_certificate();
        check_projection_roundoff();
        check_nonsymmetric_hessian_rejected();
        check_core_early_trial_preserves_certificate();
        check_core_early_trial_rejects_absolute_floor();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
