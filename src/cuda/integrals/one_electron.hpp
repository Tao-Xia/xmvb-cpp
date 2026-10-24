#pragma once

#include "cuda/memory/context.hpp"
#include "linalg/views.hpp"

namespace xmvb::cuda {

/** h(C) = C^T H C; retain H C for derivatives at the accepted point.
 * H must be symmetric and column-major.
 */
void transform_one_electron(Context& context,
                            linalg::ConstMatrixView core,
                            linalg::ConstMatrixView coefficients,
                            linalg::MatrixView active,
                            linalg::MatrixView ao_image);

/** Fixed-H direction dC^T (H C) + (H C)^T dC. */
void transform_one_electron_directional_from_image(Context& context,
                                                   linalg::ConstMatrixView ao_image,
                                                   linalg::ConstMatrixView direction,
                                                   linalg::MatrixView active_direction);

/** Frobenius pullback: dL/dC = (H C) (G + G^T). */
void transform_one_electron_adjoint_from_image(Context& context,
                                               linalg::ConstMatrixView ao_image,
                                               linalg::ConstMatrixView active_adjoint,
                                               linalg::MatrixView gradient);

/** Fixed-H directional pullback; H dC is formed once, H C is shared. */
void transform_one_electron_directional_adjoint_from_image(
    Context& context,
    linalg::ConstMatrixView core,
    linalg::ConstMatrixView ao_image,
    linalg::ConstMatrixView direction,
    linalg::ConstMatrixView active_adjoint,
    linalg::ConstMatrixView active_adjoint_direction,
    linalg::MatrixView gradient_direction,
    linalg::ConstMatrixView component_adjoint = {},
    linalg::MatrixView component_gradient = {});

} // namespace xmvb::cuda
