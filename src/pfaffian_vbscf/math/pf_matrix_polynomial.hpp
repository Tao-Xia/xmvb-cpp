#pragma once

#include <vector>

#include "pfaffian_vbscf/types/eigen_types.hpp"

namespace xmvb::pfaffian_vbscf {

struct MatrixPolynomialAdjointResult {
  Matrix kernel_adjoint;
  Matrix source_adjoint;
  ScalarBuffer coefficient_adjoints;
};

/**
 * @brief Builds the left-applied matrix polynomial `sum_p coeffs[p] G^p X`.
 */
Matrix apply_left_matrix_polynomial(
    const ConstMatrixRef& kernel,
    const ScalarBuffer& coefficients,
    const ConstMatrixRef& source);

/**
 * @brief Builds the right-applied matrix polynomial `sum_p coeffs[p] X G^p`.
 */
Matrix apply_right_matrix_polynomial(
    const ConstMatrixRef& kernel,
    const ScalarBuffer& coefficients,
    const ConstMatrixRef& source);

/**
 * @brief Builds the Fréchet derivative
 * `sum_{p>=1} coeffs[p] sum_{a=0}^{p-1} G^a X G^{p-1-a}`.
 */
Matrix apply_left_matrix_polynomial_frechet(
    const ConstMatrixRef& kernel,
    const ScalarBuffer& coefficients,
    const ConstMatrixRef& source);

/**
 * @brief Backpropagates an adjoint through the left-applied matrix polynomial.
 */
MatrixPolynomialAdjointResult backpropagate_left_matrix_polynomial(
    const ConstMatrixRef& kernel,
    const ScalarBuffer& coefficients,
    const ConstMatrixRef& source,
    const ConstMatrixRef& output_adjoint);

/**
 * @brief Backpropagates an adjoint through the right-applied matrix polynomial.
 */
MatrixPolynomialAdjointResult backpropagate_right_matrix_polynomial(
    const ConstMatrixRef& kernel,
    const ScalarBuffer& coefficients,
    const ConstMatrixRef& source,
    const ConstMatrixRef& output_adjoint);

/**
 * @brief Backpropagates an adjoint through the Fréchet matrix polynomial.
 */
MatrixPolynomialAdjointResult backpropagate_left_matrix_polynomial_frechet(
    const ConstMatrixRef& kernel,
    const ScalarBuffer& coefficients,
    const ConstMatrixRef& source,
    const ConstMatrixRef& output_adjoint);

/**
 * @brief Builds the double-sided matrix polynomial
 * `sum_m coeffs[m] sum_{a=0}^m G^a X G^(m-a)`.
 *
 * This is the natural projected pair-density kernel that appears when the
 * total power is distributed across two resolvent legs.
 */
Matrix apply_bilateral_matrix_polynomial(
    const ConstMatrixRef& kernel,
    const ScalarBuffer& coefficients,
    const ConstMatrixRef& source);

}  // namespace xmvb::pfaffian_vbscf
