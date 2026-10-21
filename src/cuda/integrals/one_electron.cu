#include "cuda/integrals/one_electron.hpp"

#include <stdexcept>

#include "cuda/linalg/blas.hpp"
#include "cuda/memory/buffer.hpp"

namespace xmvb::cuda {
namespace {

void validate(linalg::ConstMatrixView core,
              linalg::ConstMatrixView coefficients,
              linalg::MatrixView output) {
  if (!core.data || core.rows <= 0 || core.cols != core.rows ||
      core.ld < core.rows || !coefficients.data ||
      coefficients.rows != core.rows || coefficients.cols <= 0 ||
      coefficients.ld < coefficients.rows || !output.data ||
      output.cols != coefficients.cols || output.ld < output.rows)
    throw std::invalid_argument("invalid CUDA one-electron dimensions");
}

void validate_active(linalg::ConstMatrixView matrix, std::int64_t n) {
  if (!matrix.data || matrix.rows != n || matrix.cols != n || matrix.ld < n)
    throw std::invalid_argument("invalid CUDA one-electron active adjoint");
}

void validate_direction(linalg::ConstMatrixView direction,
                        linalg::ConstMatrixView coefficients) {
  if (!direction.data || direction.rows != coefficients.rows ||
      direction.cols != coefficients.cols || direction.ld < direction.rows)
    throw std::invalid_argument("invalid CUDA one-electron direction");
}

void add_adjoint(Context& context, linalg::MatrixView output,
                 linalg::ConstMatrixView hc,
                 linalg::ConstMatrixView g, double beta) {
  gemm(context, output, hc, g, 1.0, beta);
  gemm(context, output, hc, g, 1.0, 1.0,
       Operation::none, Operation::transpose);
}

}  // namespace

void transform_one_electron(Context& context, linalg::ConstMatrixView core,
                            linalg::ConstMatrixView coefficients,
                            linalg::MatrixView active) {
  validate(core, coefficients, active);
  if (active.rows != coefficients.cols)
    throw std::invalid_argument("invalid CUDA active one-electron output");
  Matrix<double> hc(context, core.rows, coefficients.cols);
  gemm(context, hc.view(), core, coefficients);
  gemm(context, active, coefficients, hc.view(), 1.0, 0.0,
       Operation::transpose);
}

void transform_one_electron_directional(Context& context,
    linalg::ConstMatrixView core, linalg::ConstMatrixView coefficients,
    linalg::ConstMatrixView direction, linalg::MatrixView active_direction) {
  validate(core, coefficients, active_direction);
  validate_direction(direction, coefficients);
  if (active_direction.rows != coefficients.cols)
    throw std::invalid_argument("invalid CUDA directional one-electron output");
  Matrix<double> hc(context, core.rows, coefficients.cols);
  Matrix<double> hd(context, core.rows, coefficients.cols);
  gemm(context, hc.view(), core, coefficients);
  gemm(context, hd.view(), core, direction);
  gemm(context, active_direction, direction, hc.view(), 1.0, 0.0,
       Operation::transpose);
  gemm(context, active_direction, coefficients, hd.view(), 1.0, 1.0,
       Operation::transpose);
}

void transform_one_electron_adjoint(Context& context,
    linalg::ConstMatrixView core, linalg::ConstMatrixView coefficients,
    linalg::ConstMatrixView active_adjoint, linalg::MatrixView gradient) {
  validate(core, coefficients, gradient);
  if (gradient.rows != core.rows)
    throw std::invalid_argument("invalid CUDA one-electron gradient");
  validate_active(active_adjoint, coefficients.cols);
  Matrix<double> hc(context, core.rows, coefficients.cols);
  gemm(context, hc.view(), core, coefficients);
  add_adjoint(context, gradient, hc.view(), active_adjoint, 0.0);
}

void transform_one_electron_directional_adjoint(Context& context,
    linalg::ConstMatrixView core, linalg::ConstMatrixView coefficients,
    linalg::ConstMatrixView direction,
    linalg::ConstMatrixView active_adjoint,
    linalg::ConstMatrixView active_adjoint_direction,
    linalg::MatrixView gradient_direction) {
  validate(core, coefficients, gradient_direction);
  if (gradient_direction.rows != core.rows)
    throw std::invalid_argument("invalid CUDA directional one-electron gradient");
  validate_direction(direction, coefficients);
  validate_active(active_adjoint, coefficients.cols);
  validate_active(active_adjoint_direction, coefficients.cols);
  Matrix<double> hc(context, core.rows, coefficients.cols);
  Matrix<double> hd(context, core.rows, coefficients.cols);
  gemm(context, hc.view(), core, coefficients);
  gemm(context, hd.view(), core, direction);
  add_adjoint(context, gradient_direction, hd.view(), active_adjoint, 0.0);
  add_adjoint(context, gradient_direction, hc.view(),
              active_adjoint_direction, 1.0);
}

}  // namespace xmvb::cuda
