#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/LU>

#include "vbscf/optimization/coupled/defect.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void require_close(
    const Eigen::Ref<const Eigen::VectorXd>& actual,
    const Eigen::Ref<const Eigen::VectorXd>& expected,
    double tolerance,
    const char* message) {
  if (actual.size() != expected.size() ||
      (actual - expected).stableNorm() >
          tolerance * std::max(1.0, expected.stableNorm())) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main() {
  try {
    using xmvb::vb::CoupledNewtonActions;
    using xmvb::vb::CoupledNewtonOperator;
    using xmvb::vb::MinresOptions;
    using xmvb::vb::SelectedStateCluster;
    using xmvb::vb::SelectedSubspaceResponseLayout;

    constexpr int n_orbitals = 2;
    const SelectedSubspaceResponseLayout layout(
        4, {SelectedStateCluster{1, 1.0}});
    const int n_response = layout.response_size();
    require(n_response == 5, "unexpected synthetic response dimension");

    Eigen::Matrix2d orbital_hessian;
    orbital_hessian << 2.0, 0.1,
                       0.1, 1.5;
    Eigen::Matrix<double, 5, 2> coupling;
    coupling << 0.4, -0.2,
               -0.3,  0.5,
                0.7,  0.1,
               -0.1,  0.6,
                0.2, -0.4;
    Eigen::MatrixXd response_hessian(5, 5);
    response_hessian << 4.0, 1.0, 0.0, 0.0, 0.2,
                        1.0, -3.0, 0.5, 0.0, 0.0,
                        0.0, 0.5, 2.0, 0.3, 0.0,
                        0.0, 0.0, 0.3, -1.5, 0.4,
                        0.2, 0.0, 0.0, 0.4, 1.0;
    require(response_hessian.fullPivLu().isInvertible(),
            "synthetic response Hessian is singular");

    int orbital_hessian_actions = 0;
    int forward_coupling_actions = 0;
    int adjoint_coupling_actions = 0;
    int response_hessian_actions = 0;
    CoupledNewtonOperator coupled_operator(
        n_orbitals,
        layout,
        CoupledNewtonActions{
            [&](const auto& directions) {
              ++orbital_hessian_actions;
              return (orbital_hessian * directions).eval();
            },
            [&](const auto& directions) {
              ++forward_coupling_actions;
              return (coupling * directions).eval();
            },
            [&](const auto& directions) {
              ++adjoint_coupling_actions;
              return (coupling.transpose() * directions).eval();
            },
            [&](const auto& directions) {
              ++response_hessian_actions;
              return (response_hessian * directions).eval();
            },
            {}});
    const Eigen::Vector2d orbital_gradient(0.8, -0.6);
    const Eigen::VectorXd structure_residual =
        (Eigen::VectorXd(5) << -1.0, 2.0, -0.5, -3.0, 1.0).finished();
    MinresOptions options;
    options.relative_residual_tolerance = 2.0e-12;
    require(options.maximum_iterations == 0,
            "test must exercise the algebraic-dimension default");

    const auto correction = xmvb::vb::correct_structure_defect(
        coupled_operator,
        orbital_gradient,
        structure_residual,
        options);
    const Eigen::VectorXd dense_response =
        -response_hessian.fullPivLu().solve(structure_residual);
    const Eigen::Vector2d dense_gradient =
        orbital_gradient + coupling.transpose() * dense_response;
    require(correction.converged(),
            "matrix-free structure-defect solve did not converge");
    require_close(
        correction.response_correction,
        dense_response,
        2.0e-11,
        "structure correction disagrees with dense reference");
    require_close(
        correction.corrected_orbital_gradient,
        dense_gradient,
        2.0e-11,
        "defect-corrected orbital gradient disagrees with dense reference");
    const Eigen::VectorXd explicit_residual =
        -structure_residual -
        response_hessian * correction.response_correction;
    require_close(
        correction.linear_result.residual,
        explicit_residual,
        2.0e-13,
        "reported structure residual is not the explicit residual");
    require_close(
        correction.remaining_structure_residual,
        -explicit_residual,
        2.0e-13,
        "remaining structure defect has the wrong sign or value");
    require(correction.linear_result.residual_norm <=
                correction.linear_result.residual_target,
            "structure correction lacks an explicit residual certificate");
    const double dense_model_correction =
        0.5 * structure_residual.dot(dense_response);
    require(std::abs(correction.stationary_limit_model_change -
                     dense_model_correction) <= 2.0e-12,
            "stationary-limit model change has the wrong value or sign");
    require(std::abs(correction.exact_model_change -
                     dense_model_correction) <= 2.0e-12,
            "exact converged model change has the wrong value or sign");
    require(correction.linear_result.iterations <= n_response,
            "default solve exceeded the response algebraic dimension");
    require(response_hessian_actions > 0 &&
                adjoint_coupling_actions == 1 &&
                orbital_hessian_actions == 0 &&
                forward_coupling_actions == 0,
            "structure correction evaluated unrelated coupled blocks");

    MinresOptions limited_options = options;
    limited_options.maximum_iterations = 1;
    const int adjoint_actions_before_limited = adjoint_coupling_actions;
    const auto limited_correction = xmvb::vb::correct_structure_defect(
        coupled_operator,
        orbital_gradient,
        structure_residual,
        limited_options);
    require(!limited_correction.converged(),
            "work-limited structure solve unexpectedly converged");
    require(limited_correction.corrected_orbital_gradient.size() == 0 &&
                adjoint_coupling_actions == adjoint_actions_before_limited,
            "uncertified structure solve exposed a corrected gradient");
    const Eigen::VectorXd limited_defect = structure_residual +
        response_hessian * limited_correction.response_correction;
    require(limited_defect.stableNorm() >
                limited_correction.linear_result.residual_target,
            "work-limited structure solve has no remaining defect");
    require_close(
        limited_correction.remaining_structure_residual,
        limited_defect,
        2.0e-13,
        "work-limited remaining defect is not explicit");
    const double limited_exact_model_change =
        structure_residual.dot(limited_correction.response_correction) +
        0.5 * limited_correction.response_correction.dot(
            response_hessian * limited_correction.response_correction);
    require(std::abs(limited_correction.exact_model_change -
                     limited_exact_model_change) <= 2.0e-12,
            "work-limited exact model change is inaccurate");
    const double residual_form_model_change =
        limited_correction.stationary_limit_model_change +
        0.5 * limited_correction.response_correction.dot(limited_defect);
    require(std::abs(limited_correction.exact_model_change -
                     residual_form_model_change) <= 2.0e-12,
            "finite-solve model-change identities disagree");

    const int adjoint_actions_before_zero = adjoint_coupling_actions;
    const int response_actions_before_zero = response_hessian_actions;
    const auto zero_correction = xmvb::vb::correct_structure_defect(
        coupled_operator,
        orbital_gradient,
        Eigen::VectorXd::Zero(n_response),
        options);
    require(zero_correction.converged(),
            "zero structure defect was not accepted exactly");
    require(zero_correction.response_correction.isZero(0.0) &&
                zero_correction.remaining_structure_residual.isZero(0.0) &&
                zero_correction.linear_result.residual.isZero(0.0) &&
                zero_correction.exact_model_change == 0.0 &&
                zero_correction.stationary_limit_model_change == 0.0,
            "zero structure defect produced a nonzero response");
    require_close(
        zero_correction.corrected_orbital_gradient,
        orbital_gradient,
        0.0,
        "zero structure defect changed the orbital gradient");
    require(zero_correction.linear_result.operator_actions == 0 &&
                zero_correction.linear_result.preconditioner_actions == 0 &&
                zero_correction.linear_result.residual_checks == 0 &&
                response_hessian_actions == response_actions_before_zero &&
                adjoint_coupling_actions == adjoint_actions_before_zero,
            "zero structure defect evaluated a matrix-free action");

    std::cout << "coupled structure-defect correction: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
