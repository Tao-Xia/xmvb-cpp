#include "pfaffian_vbscf/kernel/pf_forward_kernel.hpp"

#include <stdexcept>
#include <utility>

#include "pfaffian_vbscf/kernel/pf_closed_shell_ri_kernel.hpp"
#include "pfaffian_vbscf/kernel/pf_closed_shell_spatial_kernel.hpp"
#include "pfaffian_vbscf/kernel/pf_operand_views.hpp"
#include "pfaffian_vbscf/math/dense_utils.hpp"
#include "pfaffian_vbscf/math/pf_matrix_polynomial.hpp"
#include "pfaffian_vbscf/math/trace_projector.hpp"
#include "pfaffian_vbscf/tensor/pf_tensor_contractor.hpp"

namespace xmvb::pfaffian_vbscf {

namespace {

/**
 * @brief Builds the alternating-sign projected coefficient family used by the
 * closed-shell matrix-polynomial formulas.
 *
 * `cache.projected_overlap_coefficients` stores the projected overlap
 * coefficients `c_0, ..., c_n`. This helper converts that family into the
 * coefficient vector needed by the projected matrix polynomials of order
 * `order`.
 */
ScalarBuffer build_closed_shell_projected_coefficients(
    const PfKernelCache& cache,
    int order) {
  if (order < 0) {
    return {};
  }
  if (static_cast<int>(cache.projected_overlap_coefficients.size()) !=
      cache.trace_order + 1) {
    throw std::invalid_argument(
        "cache.projected_overlap_coefficients size does not match "
        "cache.trace_order + 1");
  }

  ScalarBuffer coefficients(order + 1, 0.0);
  for (int power = 0; power <= order; ++power) {
    const double sign = ((power % 2) == 0) ? 1.0 : -1.0;
    coefficients[power] =
        sign *
        cache.projected_overlap_coefficients[
            order - power];
  }
  return coefficients;
}

/**
 * @brief Evaluates one generic tensor term against the packed active-space
 * two-electron tensor.
 *
 * `left_block` and `right_block` are both `M x M` spatial operands
 * materialized from the cache according to the tensor-term descriptor.
 */
double evaluate_one_term(
    const PfKernelCache& cache,
    const ScalarBuffer& ggo,
    const PfTensorTerm& term) {
  const detail::MatrixBlock left_block =
      detail::materialize_operand_block(cache, term.left_operand);
  const detail::MatrixBlock right_block =
      detail::materialize_operand_block(cache, term.right_operand);

  double value = 0.0;
  switch (term.contraction) {
    case PfTensorContraction::Direct:
      value = PfTensorContractor::contract_direct(
          ggo,
          cache.n_active_orbitals,
          left_block,
          right_block);
      break;
    case PfTensorContraction::Exchange:
      value = term.same_spin
          ? PfTensorContractor::contract_same_spin_exchange(
                ggo,
                cache.n_active_orbitals,
                left_block,
                right_block)
          : PfTensorContractor::contract_exchange(
                ggo,
                cache.n_active_orbitals,
                left_block,
                right_block);
      break;
    case PfTensorContraction::Coulomb:
      value = term.same_spin
          ? PfTensorContractor::contract_same_spin_coulomb(
                ggo,
                cache.n_active_orbitals,
                left_block,
                right_block)
          : PfTensorContractor::contract_coulomb(
                ggo,
                cache.n_active_orbitals,
                left_block,
                right_block);
      break;
    case PfTensorContraction::SameSpinSeparable:
      value = PfTensorContractor::contract_same_spin_separable(
          ggo,
          cache.n_active_orbitals,
          left_block,
          right_block);
      break;
    case PfTensorContraction::SameSpinBridge:
      value = PfTensorContractor::contract_same_spin_bridge(
          ggo,
          cache.n_active_orbitals,
          left_block,
          right_block);
      break;
  }

  // If a tensor term is weighted by one projected overlap coefficient, the
  // caller must fold that physical prefactor into term.scale explicitly.
  return term.scale * value;
}


/**
 * @brief Builds the generic helper `left * sigma * right`.
 *
 * All matrices in this helper are `2M x 2M`.
 */
void build_common_kernel_auxiliaries(PfKernelCache* cache) {
  if (cache == nullptr) {
    throw std::invalid_argument("cache must not be null");
  }

  cache->left_sigma_right.noalias() =
      cache->left * cache->sigma * cache->right;
}

/**
 * @brief Builds the generic auxiliary matrix families used by the older
 * tensor-term framework.
 *
 * Every matrix built here has shape `2M x 2M`.
 */
void build_full_kernel_auxiliaries(PfKernelCache* cache) {
  if (cache == nullptr) {
    throw std::invalid_argument("cache must not be null");
  }

  build_common_kernel_auxiliaries(cache);
  cache->right_sigma.noalias() = cache->right * cache->sigma;
  cache->kernel_power_times_left.resize(cache->kernel_powers.size());
  cache->kernel_power_times_right.resize(cache->kernel_powers.size());
  cache->kernel_power_times_left_sigma_right.resize(cache->kernel_powers.size());
  cache->right_sigma_times_kernel_power_times_left_sigma_right.resize(
      cache->kernel_powers.size());

  for (std::size_t power = 0; power < cache->kernel_powers.size(); ++power) {
    cache->kernel_power_times_left[power] =
        cache->kernel_powers[power] * cache->left;
    cache->kernel_power_times_right[power] =
        cache->kernel_powers[power] * cache->right;
    cache->kernel_power_times_left_sigma_right[power] =
        cache->kernel_powers[power] * cache->left_sigma_right;
    cache->right_sigma_times_kernel_power_times_left_sigma_right[power] =
        cache->right_sigma * cache->kernel_power_times_left_sigma_right[power];
  }
}

/**
 * @brief Builds one generic trace-derived spin-orbital density helper.
 *
 * The returned matrix has shape `2M x 2M`.
 */
Matrix build_trace_rdm(
    const PfKernelCache& cache,
    int trace_index);

/**
 * @brief Builds the full family of generic trace-derived spin-orbital density
 * helpers used by the older tensor-term path.
 */
void build_trace_rdms(PfKernelCache* cache) {
  if (cache == nullptr) {
    throw std::invalid_argument("cache must not be null");
  }

  cache->trace_rdms.clear();
  cache->trace_rdms.reserve(cache->trace_order);
  for (int trace_index = 0; trace_index < cache->trace_order; ++trace_index) {
    cache->trace_rdms.push_back(build_trace_rdm(*cache, trace_index));
  }
}

Matrix build_trace_rdm(
    const PfKernelCache& cache,
    int trace_index) {
  if (trace_index < 0 || trace_index >= cache.trace_order) {
    throw std::invalid_argument("trace_index is out of range");
  }

  const int n = cache.n_active_orbitals;
  const Matrix gc =
      cache.kernel_powers[trace_index]
          .topLeftCorner(n, n) *
      cache.left_sigma_right.topLeftCorner(n, n);
  const double scale = 2.0 * static_cast<double>(trace_index + 1);

  Matrix trace_rdm = Matrix::Zero(cache.n_spin_orbitals, cache.n_spin_orbitals);
  trace_rdm.topLeftCorner(n, n).noalias() = scale * gc;
  trace_rdm.bottomRightCorner(n, n).noalias() = scale * gc;
  return trace_rdm;
}

/**
 * @brief Builds the physical closed-shell spatial 1-RDM.
 *
 * `closed_shell_pair_core` and `closed_shell_spatial_kernel` are the `M x M`
 * spatial objects used by the determinant-free closed-shell fast path.
 */
Matrix build_closed_shell_one_rdm(
    const PfKernelCache& cache) {
  if (cache.trace_order <= 0) {
    return Matrix::Zero(cache.n_active_orbitals, cache.n_active_orbitals);
  }

  const ScalarBuffer density_coefficients =
      build_closed_shell_projected_coefficients(cache, cache.trace_order - 1);

  return 2.0 *
      apply_left_matrix_polynomial(
          cache.closed_shell_spatial_kernel,
          density_coefficients,
          cache.closed_shell_pair_core);
}

/**
 * @brief Builds the closed-shell exact spatial kernel from spatial `M x M`
 * inputs or directly from `M x M` block views cut out of a full `2M x 2M`
 * spin-orbital cache build.
 *
 * The current implementation uses the right-multiplied spatial kernel
 * convention
 *
 * `C_code = left_ba^T * S * right_ab^T`
 * `H_code = C_code * S`
 *
 * where `C_code` is stored in `closed_shell_pair_core` and `H_code` is stored
 * in `closed_shell_spatial_kernel`.
 */
template <typename LeftBADerived, typename SpatialOverlapDerived, typename RightABDerived>
void build_closed_shell_spatial_core(
    PfKernelCache* cache,
    const Eigen::MatrixBase<LeftBADerived>& left_ba,
    const Eigen::MatrixBase<SpatialOverlapDerived>& spatial_overlap,
    const Eigen::MatrixBase<RightABDerived>& right_ab) 
{
  if (cache == nullptr) {
    throw std::invalid_argument("cache must not be null");
  }

  cache->n_active_orbitals = static_cast<int>(left_ba.rows());
  cache->n_spin_orbitals = 2 * cache->n_active_orbitals;
  cache->closed_shell_spatial_overlap = spatial_overlap;
  cache->closed_shell_left_ba_block = left_ba;
  cache->closed_shell_right_ab_block = right_ab;

  cache->closed_shell_pair_core.resize(
      cache->n_active_orbitals,
      cache->n_active_orbitals);

  cache->closed_shell_spatial_kernel.resize(
      cache->n_active_orbitals,
      cache->n_active_orbitals);

  cache->closed_shell_trace_powers.clear();
  cache->closed_shell_traces.clear();

  // The remaining matrices depend on the projected overlap coefficients. They
  // are cleared here so the cache shape matches the actual build stage.
  cache->closed_shell_sa.resize(0, 0);
  cache->closed_shell_h.resize(0, 0);
  cache->closed_shell_d.resize(0, 0);
  cache->closed_shell_pair_poly.resize(0, 0);
  cache->closed_shell_pair_term_matrix.resize(0, 0);
  cache->closed_shell_c_poly_h_n2.resize(0, 0);
  cache->closed_shell_frechet_h_c_n2.resize(0, 0);
  cache->closed_shell_frechet_h_c_h_n2.resize(0, 0);
  cache->closed_shell_d_poly_h_n2.resize(0, 0);
  cache->closed_shell_opposite_bridge_h_split_matrix.resize(0, 0);
  cache->closed_shell_pair_coefficients.clear();
  cache->closed_shell_pair_tail_coefficients.clear();
  cache->closed_shell_coeff_n2.clear();

  if (cache->n_active_orbitals <= 0) {
    return;
  }

  // `closed_shell_left_ba_block` and `closed_shell_right_ab_block` are raw
  // `BA` / `AB` slices cut out of the full spin-orbital Pfaffian matrices.
  // The closed-shell spatial formulas instead use one common pair-orbital
  // ordering, so the block objects must first be reoriented into the formula
  // operands `L` and `R`.
  const Matrix left_pair =
      cache->closed_shell_left_ba_block.transpose().eval();
  const Matrix left_pair_times_overlap =
      left_pair * cache->closed_shell_spatial_overlap;
  const Matrix right_pair_transpose =
      cache->closed_shell_right_ab_block.transpose().eval();

  cache->closed_shell_pair_core.noalias() =
      left_pair_times_overlap * right_pair_transpose;
  cache->closed_shell_spatial_kernel.noalias() =
      cache->closed_shell_pair_core *
      cache->closed_shell_spatial_overlap;

  // compute the power of kernal matrix and its trace
  // kernal power: G^0, G^1, G^2 .... G^{n-1}
  // traces: Tr(G^0), Tr(G^1), Tr(G^2) ... Tr(G^{n-1})
  KernelTraceData trace_data =
      compute_kernel_trace_data(
          cache->closed_shell_spatial_kernel,
          cache->trace_order
      );

  cache->closed_shell_trace_powers = std::move(trace_data.kernel_powers);
  cache->closed_shell_traces = std::move(trace_data.traces);

  // The closed-shell spatial reduction collapses two identical spin-diagonal
  // sectors, so the stored trace series must be doubled before projection.
  for (double& trace_value : cache->closed_shell_traces) {
    trace_value *= 2.0;
  }
}

/**
 * @brief Builds the `(n-1)` and `(n-2)` closed-shell spatial helpers after the
 * projected overlap coefficients have already been prepared.
 *
 * This function assumes that:
 * - `projected_overlap_coefficients` already match the active closed-shell
 *   overlap projection convention
 * - `closed_shell_pair_core`, `closed_shell_spatial_kernel`, and
 *   `closed_shell_spatial_overlap` have already been constructed
 */
void populate_closed_shell_auxiliaries_from_projection(
    PfKernelCache* cache) {
  if (cache == nullptr) {
    throw std::invalid_argument("cache must not be null");
  }

  if (cache->trace_order <= 0) {
    return;
  }

  const int n = cache->n_active_orbitals;

  // All matrices below are `M x M` spatial objects used by the determinant-free
  // closed-shell exact forward/adjoint path.
  cache->closed_shell_sa =
      cache->closed_shell_spatial_overlap * cache->closed_shell_left_ba_block;
  cache->closed_shell_h =
      cache->closed_shell_spatial_overlap * cache->closed_shell_pair_core;
  cache->closed_shell_d =
      cache->closed_shell_right_ab_block * cache->closed_shell_h;

  cache->closed_shell_pair_coefficients =
      build_closed_shell_projected_coefficients(
          *cache,
          cache->trace_order - 1);
  
  for (double& coefficient : cache->closed_shell_pair_coefficients) {
    coefficient *= 0.5;
  }
  if (cache->closed_shell_pair_coefficients.size() > 1) {
    cache->closed_shell_pair_tail_coefficients.assign(
        cache->closed_shell_pair_coefficients.begin() + 1,
        cache->closed_shell_pair_coefficients.end());
  }

  cache->closed_shell_pair_term_matrix = Matrix::Zero(n, n);
  cache->closed_shell_pair_poly.resize(0, 0);
  cache->closed_shell_c_poly_h_n2.resize(0, 0);
  cache->closed_shell_frechet_h_c_n2.resize(0, 0);
  cache->closed_shell_frechet_h_c_h_n2.resize(0, 0);
  cache->closed_shell_d_poly_h_n2.resize(0, 0);
  cache->closed_shell_opposite_bridge_h_split_matrix.resize(0, 0);

  if (!cache->closed_shell_pair_coefficients.empty()) {
    cache->closed_shell_pair_term_matrix =
        cache->closed_shell_pair_coefficients.front() *
        cache->closed_shell_left_ba_block;
  }
  if (!cache->closed_shell_pair_tail_coefficients.empty()) {
    cache->closed_shell_pair_poly =
        apply_left_matrix_polynomial(
            cache->closed_shell_h,
            cache->closed_shell_pair_tail_coefficients,
            cache->closed_shell_sa);
    cache->closed_shell_pair_term_matrix.noalias() +=
        cache->closed_shell_pair_core * cache->closed_shell_pair_poly;
  }

  cache->closed_shell_coeff_n2 =
      build_closed_shell_projected_coefficients(
          *cache,
          cache->trace_order - 2);
  if (cache->closed_shell_coeff_n2.empty()) {
    return;
  }

  cache->closed_shell_c_poly_h_n2 =
      apply_right_matrix_polynomial(
          cache->closed_shell_h,
          cache->closed_shell_coeff_n2,
          cache->closed_shell_pair_core);
  cache->closed_shell_frechet_h_c_n2 =
      apply_left_matrix_polynomial_frechet(
          cache->closed_shell_h,
          cache->closed_shell_coeff_n2,
          cache->closed_shell_pair_core);
  cache->closed_shell_frechet_h_c_h_n2.noalias() =
      cache->closed_shell_frechet_h_c_n2 * cache->closed_shell_h;
  cache->closed_shell_d_poly_h_n2 =
      apply_right_matrix_polynomial(
          cache->closed_shell_h,
          cache->closed_shell_coeff_n2,
          cache->closed_shell_d);
  cache->closed_shell_opposite_bridge_h_split_matrix.noalias() =
      cache->closed_shell_frechet_h_c_n2 * cache->closed_shell_sa;
}

/**
 * @brief Builds the fully generic spin-orbital cache.
 *
 * This path still supports the older tensor-term framework, so it constructs
 * the generic `2M x 2M` kernel family and, in the same pass, reuses the
 * extracted `BA`, `S`, and `AB` block views to populate the closed-shell
 * production fast path.
 */
PfKernelCache build_cache_impl(
    const ConstMatrixRef& left,
    const ConstMatrixRef& sigma,
    const ConstMatrixRef& right,
    int trace_order,
    bool build_full_auxiliaries,
    bool include_trace_rdms)
{
  PfKernelCache cache;
  cache.n_spin_orbitals = static_cast<int>(left.rows());
  cache.n_active_orbitals = cache.n_spin_orbitals / 2;
  cache.trace_order = trace_order;
  cache.left = left;
  cache.sigma = sigma;
  cache.right = right;
  cache.kernel = build_kernel(left, sigma, right);

  const KernelTraceData trace_data =
      compute_kernel_trace_data(cache.kernel, trace_order);
  cache.kernel_powers = trace_data.kernel_powers;
  cache.traces = trace_data.traces;

  if (build_full_auxiliaries) {
    build_full_kernel_auxiliaries(&cache);
  } else {
    build_common_kernel_auxiliaries(&cache);
  }

  const TraceProjectionResult projection =
      TraceProjector::project(cache.traces, trace_order);
  cache.trace_weights = projection.trace_weights;
  cache.projected_overlap_coefficients = projection.overlap_coefficients;
  cache.overlap_value = projection.overlap_value;

  // Reuse the same cache build for the production closed-shell fast path, but
  // feed it directly from the `BA`, `S`, and `AB` block views instead of
  // materializing an intermediate transport struct.
  build_closed_shell_spatial_core(
      &cache,
      left.bottomLeftCorner(cache.n_active_orbitals, cache.n_active_orbitals),
      sigma.topLeftCorner(cache.n_active_orbitals, cache.n_active_orbitals),
      right.topRightCorner(cache.n_active_orbitals, cache.n_active_orbitals)
  );

  populate_closed_shell_auxiliaries_from_projection(&cache);

  if (include_trace_rdms) {
    build_trace_rdms(&cache);
  }
  cache.one_rdm = build_closed_shell_one_rdm(cache);
  return cache;
}

}  // namespace

PfKernelCache PfForwardKernel::build_cache(
    const ConstMatrixRef& left,
    const ConstMatrixRef& sigma,
    const ConstMatrixRef& right,
    int trace_order) {
  // return a cache
  return build_cache_impl(
      left,
      sigma,
      right,
      trace_order,
      true,
      true);
}

PfKernelCache PfForwardKernel::build_closed_shell_exact_cache(
    const ConstMatrixRef& left,
    const ConstMatrixRef& sigma,
    const ConstMatrixRef& right,
    int trace_order) 
{
  PfKernelCache cache;
  cache.trace_order = trace_order;

  build_closed_shell_spatial_core(
      &cache,
      left.bottomLeftCorner(left.rows() / 2, left.rows() / 2),
      sigma.topLeftCorner(sigma.rows() / 2, sigma.rows() / 2),
      right.topRightCorner(right.rows() / 2, right.rows() / 2));

  const TraceProjectionResult projection =
      TraceProjector::project(cache.closed_shell_traces, trace_order);

  cache.trace_weights = projection.trace_weights;
  cache.projected_overlap_coefficients = projection.overlap_coefficients;
  cache.overlap_value = projection.overlap_value;
  cache.traces = cache.closed_shell_traces;
  populate_closed_shell_auxiliaries_from_projection(&cache);
  cache.one_rdm = build_closed_shell_one_rdm(cache);

  // Preserve the full `2M x 2M` generic inputs for callers that still need the
  // original spin-orbital objects after the closed-shell fast path is built.
  cache.left = left;
  cache.sigma = sigma;
  cache.right = right;
  return cache;
}

PfKernelCache PfForwardKernel::build_closed_shell_exact_spatial_cache(
    const ConstMatrixRef& left_ba,
    const ConstMatrixRef& spatial_overlap,
    const ConstMatrixRef& right_ab,
    int trace_order)
{
  PfKernelCache cache;
  cache.trace_order = trace_order;

  // 1. compute the kernal
  // 2. compute the kernal power
  // 3. compute the trace of kernal power
  build_closed_shell_spatial_core(
      &cache,
      left_ba,
      spatial_overlap,
      right_ab);
  
  // compute the overlap of two VB structures through Newton recursion
  const TraceProjectionResult projection = TraceProjector::project(
        cache.closed_shell_traces, 
        trace_order
  );
  
  cache.trace_weights = projection.trace_weights;
  cache.projected_overlap_coefficients = projection.overlap_coefficients;
  cache.overlap_value = projection.overlap_value;
  cache.traces = cache.closed_shell_traces;

  populate_closed_shell_auxiliaries_from_projection(&cache);
  cache.one_rdm = build_closed_shell_one_rdm(cache);

  return cache;
}

double PfForwardKernel::evaluate_tensor_terms(
    const PfKernelCache& cache,
    const ScalarBuffer& ggo,
    const std::vector<PfTensorTerm>& terms) {
  double value = 0.0;
  for (const PfTensorTerm& term : terms) {
    value += evaluate_one_term(cache, ggo, term);
  }
  return value;
}

double PfForwardKernel::evaluate_closed_shell_two_electron(
    const PfKernelCache& cache,
    const ScalarBuffer& ggo) {
  return evaluate_closed_shell_two_electron_spatial_exact(cache, ggo);
}

double PfForwardKernel::evaluate_closed_shell_two_electron_ri(
    const PfKernelCache& cache,
    int n_auxiliary_functions,
    const ScalarBuffer& ri_active_pair_factors) {
  return evaluate_closed_shell_two_electron_spatial_ri(
      cache,
      n_auxiliary_functions,
      ri_active_pair_factors);
}

}  // namespace xmvb::pfaffian_vbscf
