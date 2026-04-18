#include "pfaffian_vbscf/math/pf_matrix_polynomial.hpp"

#include <stdexcept>

namespace xmvb::pfaffian_vbscf {

namespace {

void validate_apply_inputs(
    const ConstMatrixRef& kernel,
    const ScalarBuffer& coefficients,
    const ConstMatrixRef& source) {
  if (kernel.rows() != kernel.cols()) {
    throw std::invalid_argument("kernel must be square");
  }
  if (source.rows() != source.cols()) {
    throw std::invalid_argument("source must be square");
  }
  if (kernel.rows() != source.rows()) {
    throw std::invalid_argument("kernel/source dimensions do not match");
  }
  if (coefficients.empty()) {
    throw std::invalid_argument("coefficients must not be empty");
  }
}

void validate_output_adjoint_inputs(
    const ConstMatrixRef& kernel,
    const ScalarBuffer& coefficients,
    const ConstMatrixRef& source,
    const ConstMatrixRef& output_adjoint) {
  validate_apply_inputs(kernel, coefficients, source);
  if (output_adjoint.rows() != source.rows() ||
      output_adjoint.cols() != source.cols()) {
    throw std::invalid_argument("output_adjoint dimensions do not match the output");
  }
}

double frobenius_inner_product(
    const ConstMatrixRef& left,
    const ConstMatrixRef& right) {
  if (left.rows() != right.rows() || left.cols() != right.cols()) {
    throw std::invalid_argument("Frobenius inner product dimension mismatch");
  }
  return left.cwiseProduct(right).sum();
}

}  // namespace

Matrix apply_left_matrix_polynomial(
    const ConstMatrixRef& kernel,
    const ScalarBuffer& coefficients,
    const ConstMatrixRef& source) {
  validate_apply_inputs(kernel, coefficients, source);

  const std::size_t degree = coefficients.size();
  Matrix value = coefficients.front() * source;
  
  if (degree == 1) {
    return value;
  }
  if (degree == 2) {
    const double scale = coefficients[1];
    if (scale != 0.0) {
      value.noalias() += scale * (kernel * source);
    }
    return value;
  }
  if (degree == 3) {
    const Matrix kx = kernel * source;
    if (coefficients[1] != 0.0) {
      value.noalias() += coefficients[1] * kx;
    }
    if (coefficients[2] != 0.0) {
      value.noalias() += coefficients[2] * (kernel * kx);
    }
    return value;
  }

  Matrix current = source;
  for (std::size_t index = 1; index < degree; ++index) {
    current = kernel * current;
    const double scale = coefficients[index];
    if (scale == 0.0) {
      continue;
    }
    value.noalias() += scale * current;
  }
  return value;
}

Matrix apply_right_matrix_polynomial(
    const ConstMatrixRef& kernel,
    const ScalarBuffer& coefficients,
    const ConstMatrixRef& source) {
  validate_apply_inputs(kernel, coefficients, source);

  const std::size_t degree = coefficients.size();
  Matrix value = coefficients.front() * source;
  if (degree == 1) {
    return value;
  }
  if (degree == 2) {
    const double scale = coefficients[1];
    if (scale != 0.0) {
      value.noalias() += scale * (source * kernel);
    }
    return value;
  }
  if (degree == 3) {
    const Matrix xk = source * kernel;
    if (coefficients[1] != 0.0) {
      value.noalias() += coefficients[1] * xk;
    }
    if (coefficients[2] != 0.0) {
      value.noalias() += coefficients[2] * (xk * kernel);
    }
    return value;
  }

  Matrix current = source;
  for (std::size_t index = 1; index < degree; ++index) {
    current = current * kernel;
    const double scale = coefficients[index];
    if (scale == 0.0) {
      continue;
    }
    value.noalias() += scale * current;
  }
  return value;
}

Matrix apply_left_matrix_polynomial_frechet(
    const ConstMatrixRef& kernel,
    const ScalarBuffer& coefficients,
    const ConstMatrixRef& source) {
  validate_apply_inputs(kernel, coefficients, source);

  const int degree = static_cast<int>(coefficients.size());
  if (degree <= 1) {
    return Matrix::Zero(kernel.rows(), kernel.cols());
  }
  if (degree == 2) {
    return coefficients[1] * source;
  }
  if (degree == 3) {
    Matrix value = Matrix::Zero(kernel.rows(), kernel.cols());
    if (coefficients[1] != 0.0) {
      value.noalias() += coefficients[1] * source;
    }
    if (coefficients[2] != 0.0) {
      value.noalias() += coefficients[2] * (kernel * source + source * kernel);
    }
    return value;
  }

  Matrix value = Matrix::Zero(kernel.rows(), kernel.cols());
  Matrix frechet_term = source;
  Matrix right_power = source;
  if (coefficients[1] != 0.0) {
    value.noalias() += coefficients[1] * frechet_term;
  }
  for (int power = 2; power < degree; ++power) {
    right_power = right_power * kernel;
    frechet_term = kernel * frechet_term + right_power;
    const double scale = coefficients[xmvb::to_size(power)];
    if (scale == 0.0) {
      continue;
    }
    value.noalias() += scale * frechet_term;
  }
  return value;
}

Matrix apply_bilateral_matrix_polynomial(
    const ConstMatrixRef& kernel,
    const ScalarBuffer& coefficients,
    const ConstMatrixRef& source) {
  validate_apply_inputs(kernel, coefficients, source);

  Matrix value = coefficients.front() * source;
  if (coefficients.size() == 1) {
    return value;
  }

  Matrix bilateral_term = source;
  Matrix right_power = source;
  for (std::size_t total_power = 1; total_power < coefficients.size(); ++total_power) {
    right_power = right_power * kernel;
    bilateral_term = kernel * bilateral_term + right_power;
    const double scale = coefficients[total_power];
    if (scale == 0.0) {
      continue;
    }
    value.noalias() += scale * bilateral_term;
  }
  return value;
}

MatrixPolynomialAdjointResult backpropagate_left_matrix_polynomial(
    const ConstMatrixRef& kernel,
    const ScalarBuffer& coefficients,
    const ConstMatrixRef& source,
    const ConstMatrixRef& output_adjoint) {
  validate_output_adjoint_inputs(kernel, coefficients, source, output_adjoint);

  MatrixPolynomialAdjointResult result;
  result.kernel_adjoint = Matrix::Zero(kernel.rows(), kernel.cols());
  result.source_adjoint = Matrix::Zero(source.rows(), source.cols());
  result.coefficient_adjoints.assign(coefficients.size(), 0.0);

  if (coefficients.size() == 1) {
    result.coefficient_adjoints[0] =
        frobenius_inner_product(output_adjoint, source);
    result.source_adjoint.noalias() += coefficients[0] * output_adjoint;
    return result;
  }
  if (coefficients.size() == 2) {
    const Matrix kx = kernel * source;
    result.coefficient_adjoints[0] =
        frobenius_inner_product(output_adjoint, source);
    result.coefficient_adjoints[1] =
        frobenius_inner_product(output_adjoint, kx);
    result.kernel_adjoint.noalias() +=
        coefficients[1] * output_adjoint * source.transpose();
    result.source_adjoint.noalias() += coefficients[0] * output_adjoint;
    result.source_adjoint.noalias() +=
        coefficients[1] * kernel.transpose() * output_adjoint;
    return result;
  }
  if (coefficients.size() == 3) {
    const Matrix kx = kernel * source;
    const Matrix k2x = kernel * kx;
    const Matrix kt_o = kernel.transpose() * output_adjoint;
    result.coefficient_adjoints[0] =
        frobenius_inner_product(output_adjoint, source);
    result.coefficient_adjoints[1] =
        frobenius_inner_product(output_adjoint, kx);
    result.coefficient_adjoints[2] =
        frobenius_inner_product(output_adjoint, k2x);
    result.kernel_adjoint.noalias() +=
        coefficients[1] * output_adjoint * source.transpose();
    result.kernel_adjoint.noalias() +=
        coefficients[2] * output_adjoint * kx.transpose();
    result.kernel_adjoint.noalias() +=
        coefficients[2] * kt_o * source.transpose();
    result.source_adjoint.noalias() += coefficients[0] * output_adjoint;
    result.source_adjoint.noalias() += coefficients[1] * kt_o;
    result.source_adjoint.noalias() +=
        coefficients[2] * kernel.transpose() * kt_o;
    return result;
  }

  std::vector<Matrix> states(
      coefficients.size(),
      Matrix::Zero(source.rows(), source.cols()));
  states.front() = source;
  for (std::size_t index = 1; index < states.size(); ++index) {
    states[index].noalias() = kernel * states[index - 1];
  }

  std::vector<Matrix> state_adjoints(
      coefficients.size(),
      Matrix::Zero(source.rows(), source.cols()));
  for (int index = static_cast<int>(coefficients.size()) - 1; index >= 0; --index) {
    const std::size_t idx = xmvb::to_size(index);
    result.coefficient_adjoints[idx] +=
        frobenius_inner_product(output_adjoint, states[idx]);
    state_adjoints[idx].noalias() += coefficients[idx] * output_adjoint;
    if (index == 0) {
      result.source_adjoint.noalias() += state_adjoints[idx];
      continue;
    }
    result.kernel_adjoint.noalias() +=
        state_adjoints[idx] * states[xmvb::to_size(index - 1)].transpose();
    state_adjoints[xmvb::to_size(index - 1)].noalias() +=
        kernel.transpose() * state_adjoints[idx];
  }

  return result;
}

MatrixPolynomialAdjointResult backpropagate_right_matrix_polynomial(
    const ConstMatrixRef& kernel,
    const ScalarBuffer& coefficients,
    const ConstMatrixRef& source,
    const ConstMatrixRef& output_adjoint) {
  validate_output_adjoint_inputs(kernel, coefficients, source, output_adjoint);

  MatrixPolynomialAdjointResult result;
  result.kernel_adjoint = Matrix::Zero(kernel.rows(), kernel.cols());
  result.source_adjoint = Matrix::Zero(source.rows(), source.cols());
  result.coefficient_adjoints.assign(coefficients.size(), 0.0);

  if (coefficients.size() == 1) {
    result.coefficient_adjoints[0] =
        frobenius_inner_product(output_adjoint, source);
    result.source_adjoint.noalias() += coefficients[0] * output_adjoint;
    return result;
  }
  if (coefficients.size() == 2) {
    const Matrix xk = source * kernel;
    result.coefficient_adjoints[0] =
        frobenius_inner_product(output_adjoint, source);
    result.coefficient_adjoints[1] =
        frobenius_inner_product(output_adjoint, xk);
    result.kernel_adjoint.noalias() +=
        coefficients[1] * source.transpose() * output_adjoint;
    result.source_adjoint.noalias() += coefficients[0] * output_adjoint;
    result.source_adjoint.noalias() +=
        coefficients[1] * output_adjoint * kernel.transpose();
    return result;
  }
  if (coefficients.size() == 3) {
    const Matrix xk = source * kernel;
    const Matrix xk2 = xk * kernel;
    const Matrix o_kt = output_adjoint * kernel.transpose();
    result.coefficient_adjoints[0] =
        frobenius_inner_product(output_adjoint, source);
    result.coefficient_adjoints[1] =
        frobenius_inner_product(output_adjoint, xk);
    result.coefficient_adjoints[2] =
        frobenius_inner_product(output_adjoint, xk2);
    result.kernel_adjoint.noalias() +=
        coefficients[1] * source.transpose() * output_adjoint;
    result.kernel_adjoint.noalias() +=
        coefficients[2] * xk.transpose() * output_adjoint;
    result.kernel_adjoint.noalias() +=
        coefficients[2] * source.transpose() * o_kt;
    result.source_adjoint.noalias() += coefficients[0] * output_adjoint;
    result.source_adjoint.noalias() += coefficients[1] * o_kt;
    result.source_adjoint.noalias() +=
        coefficients[2] * o_kt * kernel.transpose();
    return result;
  }

  std::vector<Matrix> states(
      coefficients.size(),
      Matrix::Zero(source.rows(), source.cols()));
  states.front() = source;
  for (std::size_t index = 1; index < states.size(); ++index) {
    states[index].noalias() = states[index - 1] * kernel;
  }

  std::vector<Matrix> state_adjoints(
      coefficients.size(),
      Matrix::Zero(source.rows(), source.cols()));
  for (int index = static_cast<int>(coefficients.size()) - 1; index >= 0; --index) {
    const std::size_t idx = xmvb::to_size(index);
    result.coefficient_adjoints[idx] +=
        frobenius_inner_product(output_adjoint, states[idx]);
    state_adjoints[idx].noalias() += coefficients[idx] * output_adjoint;
    if (index == 0) {
      result.source_adjoint.noalias() += state_adjoints[idx];
      continue;
    }
    result.kernel_adjoint.noalias() +=
        states[xmvb::to_size(index - 1)].transpose() * state_adjoints[idx];
    state_adjoints[xmvb::to_size(index - 1)].noalias() +=
        state_adjoints[idx] * kernel.transpose();
  }

  return result;
}

MatrixPolynomialAdjointResult backpropagate_left_matrix_polynomial_frechet(
    const ConstMatrixRef& kernel,
    const ScalarBuffer& coefficients,
    const ConstMatrixRef& source,
    const ConstMatrixRef& output_adjoint) {
  validate_output_adjoint_inputs(kernel, coefficients, source, output_adjoint);

  MatrixPolynomialAdjointResult result;
  result.kernel_adjoint = Matrix::Zero(kernel.rows(), kernel.cols());
  result.source_adjoint = Matrix::Zero(source.rows(), source.cols());
  result.coefficient_adjoints.assign(coefficients.size(), 0.0);

  const int degree = static_cast<int>(coefficients.size());
  if (degree <= 1) {
    return result;
  }
  if (degree == 2) {
    result.coefficient_adjoints[1] =
        frobenius_inner_product(output_adjoint, source);
    result.source_adjoint.noalias() += coefficients[1] * output_adjoint;
    return result;
  }
  if (degree == 3) {
    const Matrix kx_plus_xk = kernel * source + source * kernel;
    result.coefficient_adjoints[1] =
        frobenius_inner_product(output_adjoint, source);
    result.coefficient_adjoints[2] =
        frobenius_inner_product(output_adjoint, kx_plus_xk);
    result.kernel_adjoint.noalias() +=
        coefficients[2] * output_adjoint * source.transpose();
    result.kernel_adjoint.noalias() +=
        coefficients[2] * source.transpose() * output_adjoint;
    result.source_adjoint.noalias() += coefficients[1] * output_adjoint;
    result.source_adjoint.noalias() +=
        coefficients[2] * kernel.transpose() * output_adjoint;
    result.source_adjoint.noalias() +=
        coefficients[2] * output_adjoint * kernel.transpose();
    return result;
  }

  std::vector<Matrix> right_terms(
      coefficients.size(),
      Matrix::Zero(source.rows(), source.cols()));
  std::vector<Matrix> frechet_terms(
      coefficients.size(),
      Matrix::Zero(source.rows(), source.cols()));
  right_terms[1] = source;
  frechet_terms[1] = source;
  for (int power = 2; power < degree; ++power) {
    right_terms[xmvb::to_size(power)].noalias() =
        right_terms[xmvb::to_size(power - 1)] * kernel;
    frechet_terms[xmvb::to_size(power)].noalias() =
        kernel * frechet_terms[xmvb::to_size(power - 1)] +
        right_terms[xmvb::to_size(power)];
  }

  std::vector<Matrix> right_adjoints(
      coefficients.size(),
      Matrix::Zero(source.rows(), source.cols()));
  std::vector<Matrix> frechet_adjoints(
      coefficients.size(),
      Matrix::Zero(source.rows(), source.cols()));
  for (int power = degree - 1; power >= 1; --power) {
    const std::size_t idx = xmvb::to_size(power);
    result.coefficient_adjoints[idx] +=
        frobenius_inner_product(output_adjoint, frechet_terms[idx]);
    frechet_adjoints[idx].noalias() += coefficients[idx] * output_adjoint;
  }

  for (int power = degree - 1; power >= 2; --power) {
    const std::size_t idx = xmvb::to_size(power);
    result.kernel_adjoint.noalias() +=
        frechet_adjoints[idx] *
        frechet_terms[xmvb::to_size(power - 1)].transpose();
    frechet_adjoints[xmvb::to_size(power - 1)].noalias() +=
        kernel.transpose() * frechet_adjoints[idx];
    right_adjoints[idx].noalias() += frechet_adjoints[idx];

    result.kernel_adjoint.noalias() +=
        right_terms[xmvb::to_size(power - 1)].transpose() *
        right_adjoints[idx];
    right_adjoints[xmvb::to_size(power - 1)].noalias() +=
        right_adjoints[idx] * kernel.transpose();
  }

  result.source_adjoint.noalias() += frechet_adjoints[1];
  result.source_adjoint.noalias() += right_adjoints[1];
  return result;
}

}  // namespace xmvb::pfaffian_vbscf
