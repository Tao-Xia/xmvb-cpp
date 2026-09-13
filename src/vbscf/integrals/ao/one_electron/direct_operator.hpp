#pragma once

#include <vector>

#include <Eigen/Core>

#include "vbscf/integrals/ao/contracts/input.hpp"

namespace xmvb::vb {

/** @brief Applies the exact AO-H1E operator directly from unique AO ERIs. */
std::vector<double> apply_ao_h1e(
    const double* source,
    const AoIntegralInput& ao,
    int n_threads);

/** @brief Applies the transpose exact AO-H1E operator directly from AO ERIs. */
std::vector<double> apply_ao_h1e_transpose(
    const double* adjoint,
    const AoIntegralInput& ao,
    int n_threads);

/**
 * @brief Applies the exact AO-H1E operator and its transpose.
 *
 * The unique ERI stream is read once. Caller-owned buffers allow repeated
 * Hessian-vector products to reuse their AO-sized storage.
 */
void apply_ao_h1e_fused(
    const double* source,
    const double* adjoint,
    const AoIntegralInput& ao,
    int n_threads,
    std::vector<double>* forward,
    std::vector<double>* transpose);

/** @brief Applies the fused operator to several independent directions. */
void apply_ao_h1e_fused_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& sources,
    const Eigen::Ref<const Eigen::MatrixXd>& adjoints,
    const AoIntegralInput& ao,
    int n_threads,
    Eigen::MatrixXd* forward,
    Eigen::MatrixXd* transpose);

}  // namespace xmvb::vb
