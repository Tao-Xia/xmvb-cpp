#pragma once

#include "cuda/integrals/exact.hpp"
#include "linalg/views.hpp"

namespace xmvb::cuda {

class ExactAoPairOperator;

/** Exact fixed-CI VB orbital-rotation Hessian diagonal at an accepted point.
 * All inputs and the returned nbf-by-nbf signed diagonal remain on device.
 * The MO frame is AO-metric orthonormal and ordered core, active, virtual.
 * Active density/cotangent inputs use the orthogonal active frame; pair
 * cotangent is the full symmetric Frobenius adjoint, not packed-upper data.
 * The AO graph is traversed in bounded pair-row and density channels; neither
 * a full AO four-index tensor nor a unit-rotation HVP loop is formed.
 */
Matrix<double> build_analytic_rotation_diagonal(
    Context& context,const ExactAoPairOperator& integrals,
    linalg::ConstMatrixView accepted_mo,linalg::ConstMatrixView ao_core,
    linalg::ConstMatrixView ao_effective,
    linalg::ConstMatrixView active_one_adjoint,
    linalg::ConstMatrixView full_pair_adjoint,
    linalg::ConstMatrixView active_pair_kernel,int n_core,
    ConstPairPanel accepted_products = {},ConstPairPanel exchange_products = {});

}  // namespace xmvb::cuda
