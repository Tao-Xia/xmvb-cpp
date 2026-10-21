#pragma once

#include "cuda/memory/context.hpp"
#include "linalg/views.hpp"

namespace xmvb::cuda {

/** h(C) = C^T H C for a symmetric column-major AO core Hamiltonian H. */
void transform_one_electron(Context& context, linalg::ConstMatrixView core,
                            linalg::ConstMatrixView coefficients,
                            linalg::MatrixView active);

/** dh(C;dC) = dC^T H C + C^T H dC. */
void transform_one_electron_directional(Context& context,
    linalg::ConstMatrixView core, linalg::ConstMatrixView coefficients,
    linalg::ConstMatrixView direction, linalg::MatrixView active_direction);

/** Frobenius pullback: dL/dC = H C (G + G^T). */
void transform_one_electron_adjoint(Context& context,
    linalg::ConstMatrixView core, linalg::ConstMatrixView coefficients,
    linalg::ConstMatrixView active_adjoint, linalg::MatrixView gradient);

/** Directional pullback including a changing active-kernel adjoint. */
void transform_one_electron_directional_adjoint(Context& context,
    linalg::ConstMatrixView core, linalg::ConstMatrixView coefficients,
    linalg::ConstMatrixView direction,
    linalg::ConstMatrixView active_adjoint,
    linalg::ConstMatrixView active_adjoint_direction,
    linalg::MatrixView gradient_direction);

}  // namespace xmvb::cuda
