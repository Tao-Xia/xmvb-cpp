#include "cuda/integrals/one_electron.hpp"

#include <stdexcept>

#include "cuda/linalg/blas.hpp"
#include "cuda/memory/buffer.hpp"

namespace xmvb::cuda {
namespace {

void validate(linalg::ConstMatrixView core,
              linalg::ConstMatrixView coefficients,
              linalg::MatrixView output) {
    if (!core.data || core.rows <= 0 || core.cols != core.rows || core.ld < core.rows ||
        !coefficients.data || coefficients.rows != core.rows || coefficients.cols <= 0 ||
        coefficients.ld < coefficients.rows || !output.data || output.cols != coefficients.cols ||
        output.ld < output.rows)
        throw std::invalid_argument("invalid CUDA one-electron dimensions");
}

void validate_active(linalg::ConstMatrixView matrix, std::int64_t n) {
    if (!matrix.data || matrix.rows != n || matrix.cols != n || matrix.ld < n)
        throw std::invalid_argument("invalid CUDA one-electron active adjoint");
}

void validate_direction(linalg::ConstMatrixView direction, linalg::ConstMatrixView coefficients) {
    if (!direction.data || direction.rows != coefficients.rows ||
        direction.cols != coefficients.cols || direction.ld < direction.rows)
        throw std::invalid_argument("invalid CUDA one-electron direction");
}

void validate_image(linalg::ConstMatrixView image) {
    if (!image.data || image.rows <= 0 || image.cols <= 0 || image.ld < image.rows)
        throw std::invalid_argument("invalid CUDA one-electron AO image");
}

void validate_output(linalg::MatrixView output, std::int64_t rows, std::int64_t cols) {
    if (!output.data || output.rows != rows || output.cols != cols || output.ld < output.rows)
        throw std::invalid_argument("invalid CUDA one-electron output");
}

void add_adjoint(Context& context,
                 linalg::MatrixView output,
                 linalg::ConstMatrixView hc,
                 linalg::ConstMatrixView g,
                 double beta) {
    gemm(context, output, hc, g, 1.0, beta);
    gemm(context, output, hc, g, 1.0, 1.0, Operation::none, Operation::transpose);
}

} // namespace

void transform_one_electron(Context& context,
                            linalg::ConstMatrixView core,
                            linalg::ConstMatrixView coefficients,
                            linalg::MatrixView active,
                            linalg::MatrixView ao_image) {
    validate(core, coefficients, active);
    if (active.rows != coefficients.cols)
        throw std::invalid_argument("invalid CUDA active one-electron output");
    validate_output(ao_image, core.rows, coefficients.cols);
    gemm(context, ao_image, core, coefficients);
    gemm(context, active, coefficients, ao_image, 1.0, 0.0, Operation::transpose);
}

void transform_one_electron_directional_from_image(Context& context,
                                                   linalg::ConstMatrixView ao_image,
                                                   linalg::ConstMatrixView direction,
                                                   linalg::MatrixView active_direction) {
    validate_image(ao_image);
    validate_direction(direction, ao_image);
    validate_output(active_direction, ao_image.cols, ao_image.cols);
    gemm(context, active_direction, direction, ao_image, 1.0, 0.0, Operation::transpose);
    gemm(context, active_direction, ao_image, direction, 1.0, 1.0, Operation::transpose);
}

void transform_one_electron_adjoint_from_image(Context& context,
                                               linalg::ConstMatrixView ao_image,
                                               linalg::ConstMatrixView active_adjoint,
                                               linalg::MatrixView gradient) {
    validate_image(ao_image);
    validate_active(active_adjoint, ao_image.cols);
    validate_output(gradient, ao_image.rows, ao_image.cols);
    add_adjoint(context, gradient, ao_image, active_adjoint, 0.0);
}

void transform_one_electron_directional_adjoint_from_image(
    Context& context,
    linalg::ConstMatrixView core,
    linalg::ConstMatrixView ao_image,
    linalg::ConstMatrixView direction,
    linalg::ConstMatrixView active_adjoint,
    linalg::ConstMatrixView active_adjoint_direction,
    linalg::MatrixView gradient_direction,
    linalg::ConstMatrixView component,
    linalg::MatrixView component_gradient) {
    validate(core, ao_image, gradient_direction);
    if (gradient_direction.rows != core.rows)
        throw std::invalid_argument("invalid CUDA directional one-electron gradient");
    validate_direction(direction, ao_image);
    validate_active(active_adjoint, ao_image.cols);
    validate_active(active_adjoint_direction, ao_image.cols);
    Matrix<double> hd(context, core.rows, ao_image.cols);
    gemm(context, hd.view(), core, direction);
    add_adjoint(context, gradient_direction, hd.view(), active_adjoint, 0.0);
    add_adjoint(context, gradient_direction, ao_image, active_adjoint_direction, 1.0);
    if (component.data) {
        validate_active(component, ao_image.cols);
        validate(core, ao_image, component_gradient);
        if (component_gradient.rows != core.rows)
            throw std::invalid_argument("invalid CUDA component one-electron gradient");
        add_adjoint(context, component_gradient, ao_image, component, 0.0);
    }
}

} // namespace xmvb::cuda
