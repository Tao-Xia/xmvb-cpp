#pragma once

#include "pfaffian_vbscf/matrices/pf_matrix_builder.hpp"

namespace xmvb::pfaffian_vbscf {

PfMatrixBuildResult project_spin_adapted_matrices(
    const PfMatrixBuildResult& primitive_mats,
    const ConstMatrixRef& primitive_to_adapted_coefficients);

Matrix lift_spin_adapted_matrix_gradient(
    const ConstMatrixRef& spin_adapted_gradient,
    const ConstMatrixRef& primitive_to_adapted_coefficients);

}  // namespace xmvb::pfaffian_vbscf
