#include "vbscf/integrals/active/active_space_two_electron_adjoint.hpp"

#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/integrals/active/active_space_two_electron_response_internal.hpp"

namespace xmvb::vb {
namespace {

using detail::accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache;
using detail::apply_exact_ao_pair_kernel;
using detail::build_active_pair_gradient_matrix;
using detail::build_active_pair_list;
using detail::build_ao_pair_component_tables;
using detail::build_ao_pair_to_active_pair_coefficients;
using detail::build_mixed_ao_pair_to_active_pair_coefficients_from_cache;
using detail::multiply_pair_coefficients_by_gradient_matrix;

}  // namespace

Eigen::MatrixXd backpropagate_exact_packed_active_two_electron_gradient(
    const std::vector<double>& packed_active_two_electron_gradient,
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache) {
  const int n_bf = accepted_cache.n_basis_functions;
  const int n_ao = accepted_cache.n_active_orbitals;
  const std::size_t n_active_pairs =
      accepted_cache.active_pair_first_indices.size();
  const std::size_t expected_packed_size =
      n_active_pairs * (n_active_pairs + 1) / 2;
  const std::size_t n_bf_pairs =
      static_cast<std::size_t>(n_bf) *
      (n_bf + 1) / 2;
  if (n_bf <= 0 || n_ao <= 0 ||
      accepted_cache.active_pair_second_indices.size() != n_active_pairs ||
      packed_active_two_electron_gradient.size() != expected_packed_size ||
      accepted_cache.accepted_base_pair_products.rows() !=
          static_cast<Eigen::Index>(n_bf_pairs) ||
      accepted_cache.accepted_base_pair_products.cols() !=
          static_cast<Eigen::Index>(n_active_pairs) ||
      accepted_cache.accepted_dense_active_coefficients.rows() !=
          n_bf ||
      accepted_cache.accepted_dense_active_coefficients.cols() !=
          n_ao) {
    throw std::invalid_argument(
        "cached exact active-2e adjoint pullback dimensions are inconsistent");
  }

  const auto active_pairs = build_active_pair_list(n_ao);
  const ExactCtxPairMatrix active_pair_gradient_matrix =
      build_active_pair_gradient_matrix(
          packed_active_two_electron_gradient,
          active_pairs);
  ExactCtxPairMatrix pair_gradients(n_bf_pairs, n_active_pairs);
  pair_gradients.noalias() =
      accepted_cache.accepted_base_pair_products *
      active_pair_gradient_matrix;

  Eigen::MatrixXd dense_active_gradient = Eigen::MatrixXd::Zero(
      n_bf,
      n_ao);
  accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache(
      pair_gradients,
      accepted_cache.accepted_dense_active_coefficients,
      accepted_cache,
      &dense_active_gradient);
  return dense_active_gradient;
}

ExactPackedActiveTwoElectronAdjointCache
build_exact_packed_active_two_electron_adjoint_cache(
    const std::vector<double>& packed_active_two_electron_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& accepted_dense_active_coefficients,
    const AoIntegralInput& ao_integral_input,
    int n_ao,
    const ActiveSpaceTwoElectronResult* accepted_active_space_two_electron_result) {
  const int n_bf = ao_integral_input.n_basis_functions;
  if (n_bf <= 0 || n_ao <= 0) {
    throw std::invalid_argument("exact 2e HVP cache dimensions must be positive");
  }
  if (ao_integral_input.ao_two_electron_integral_values.empty()) {
    throw std::invalid_argument("exact 2e HVP cache requires materialized AO integrals");
  }

  if (accepted_dense_active_coefficients.rows() != n_bf ||
      accepted_dense_active_coefficients.cols() != n_ao) {
    throw std::invalid_argument("accepted dense active coefficient shape mismatch");
  }

  const auto active_pairs = build_active_pair_list(n_ao);
  const std::size_t n_active_pairs = active_pairs.size();
  const std::size_t n_bf_pairs =
      n_bf * (n_bf + 1) / 2;
  const std::size_t expected_packed_gradient_size =
      n_active_pairs * (n_active_pairs + 1) / 2;
  if (packed_active_two_electron_gradient.size() != expected_packed_gradient_size) {
    throw std::invalid_argument("packed active two-electron gradient size mismatch");
  }

  ExactPackedActiveTwoElectronAdjointCache cache;
  cache.n_basis_functions = n_bf;
  cache.n_active_orbitals = n_ao;
  cache.accepted_dense_active_coefficients = accepted_dense_active_coefficients;
  build_ao_pair_component_tables(
      n_bf,
      &cache.ao_pair_first_indices,
      &cache.ao_pair_second_indices);
  cache.active_pair_first_indices.reserve(n_active_pairs);
  cache.active_pair_second_indices.reserve(n_active_pairs);
  for (const auto& active_pair : active_pairs) {
    cache.active_pair_first_indices.push_back(active_pair.first);
    cache.active_pair_second_indices.push_back(active_pair.second);
  }
  build_ao_pair_to_active_pair_coefficients(
      cache.accepted_dense_active_coefficients,
      n_bf,
      n_ao,
      active_pairs,
      &cache.accepted_pair_coefficients);
  cache.active_pair_gradient_matrix =
      build_active_pair_gradient_matrix(
          packed_active_two_electron_gradient,
          active_pairs);
  if (accepted_active_space_two_electron_result != nullptr &&
      accepted_active_space_two_electron_result->dense_ao_pair_products.size() != 0) {
    if (accepted_active_space_two_electron_result->dense_ao_pair_products.size() !=
        n_bf_pairs * n_active_pairs) {
      throw std::invalid_argument("accepted dense AO pair product cache size mismatch");
    }
    cache.accepted_base_pair_products =
        accepted_active_space_two_electron_result->dense_ao_pair_products;
  } else {
    apply_exact_ao_pair_kernel(
        ao_integral_input,
        cache.accepted_pair_coefficients,
        n_bf,
        n_active_pairs,
        &cache.accepted_base_pair_products);
  }

  cache.accepted_base_pair_gradients =
      multiply_pair_coefficients_by_gradient_matrix(
          cache.accepted_base_pair_products,
          cache.active_pair_gradient_matrix);
  return cache;
}

Eigen::MatrixXd apply_exact_packed_active_two_electron_adjoint_hessian_vector(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input) {
  ExactPackedActiveTwoElectronApplyWorkspace workspace;
  Eigen::MatrixXd dense_active_gradient_direction;
  apply_exact_packed_active_two_electron_adjoint_hessian_vector(
      accepted_cache,
      dense_active_direction,
      ao_integral_input,
      &workspace,
      &dense_active_gradient_direction);
  return dense_active_gradient_direction;
}

void apply_exact_packed_active_two_electron_adjoint_hessian_vector(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input,
    ExactPackedActiveTwoElectronApplyWorkspace* workspace,
    Eigen::MatrixXd* dense_active_gradient_direction) {
  if (workspace == nullptr) {
    throw std::invalid_argument("exact 2e workspace must not be null");
  }
  const int n_bf = ao_integral_input.n_basis_functions;
  const int n_ao = accepted_cache.n_active_orbitals;
  if (n_bf <= 0 || n_ao <= 0) {
    throw std::invalid_argument("exact two-electron HVP dimensions must be positive");
  }
  if (accepted_cache.n_basis_functions != n_bf) {
    throw std::invalid_argument("exact 2e cache basis dimension mismatch");
  }
  if (ao_integral_input.ao_two_electron_integral_values.empty()) {
    throw std::invalid_argument("exact two-electron HVP requires materialized AO integrals");
  }

  if (dense_active_direction.rows() != n_bf ||
      dense_active_direction.cols() != n_ao) {
    throw std::invalid_argument("dense active coefficient shape mismatch in exact two-electron HVP");
  }
  const std::size_t n_active_pairs =
      accepted_cache.active_pair_first_indices.size();
  if (accepted_cache.active_pair_second_indices.size() != n_active_pairs) {
    throw std::invalid_argument("exact 2e cache active-pair index size mismatch");
  }
  const std::size_t n_bf_pairs =
      n_bf * (n_bf + 1) / 2;
  workspace->dense_active_direction = dense_active_direction;

  if (accepted_cache.active_pair_gradient_matrix.rows() !=
          static_cast<Eigen::Index>(n_active_pairs) ||
      accepted_cache.active_pair_gradient_matrix.cols() !=
          static_cast<Eigen::Index>(n_active_pairs)) {
    throw std::invalid_argument("exact 2e cache pair-gradient size mismatch");
  }
  if (accepted_cache.accepted_base_pair_gradients.rows() !=
          static_cast<Eigen::Index>(n_bf_pairs) ||
      accepted_cache.accepted_base_pair_gradients.cols() !=
          static_cast<Eigen::Index>(n_active_pairs)) {
    throw std::invalid_argument("exact 2e cache base-pair-gradient size mismatch");
  }

  workspace->dense_active_gradient_direction.resize(
      n_bf,
      n_ao);
  workspace->dense_active_gradient_direction.setZero();
  // Keep the fixed-backprop term on the generic packed-pair contraction until
  // the cached full-matrix shortcut is validated against finite differences on
  // the sparse mixed-chart exact_ctx path. The direct-core 241 diagnostic
  // currently shows the exact-2e mismatch lives in this stage rather than in
  // the AO-H1E or orbital-pullback chains.
  accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache(
      accepted_cache.accepted_base_pair_gradients,
      workspace->dense_active_direction,
      accepted_cache,
      &workspace->dense_active_gradient_direction);
  build_mixed_ao_pair_to_active_pair_coefficients_from_cache(
      accepted_cache.accepted_dense_active_coefficients,
      workspace->dense_active_direction,
      accepted_cache,
      &workspace->mixed_pair_coefficients);

  multiply_pair_coefficients_by_gradient_matrix(
      workspace->mixed_pair_coefficients,
      accepted_cache.active_pair_gradient_matrix,
      &workspace->transformed_pair_coefficients);
  apply_exact_ao_pair_kernel(
      ao_integral_input,
      workspace->transformed_pair_coefficients,
      n_bf,
      n_active_pairs,
      &workspace->pair_gradients);
  accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache(
      workspace->pair_gradients,
      accepted_cache.accepted_dense_active_coefficients,
      accepted_cache,
      &workspace->dense_active_gradient_direction);

  if (dense_active_gradient_direction != nullptr) {
    *dense_active_gradient_direction =
        workspace->dense_active_gradient_direction;
  }
}

void apply_exact_packed_active_two_electron_adjoint_hessian_vector_fused(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input,
    const ExactCtxPairMatrix& directional_pair_products,
    ExactPackedActiveTwoElectronApplyWorkspace* workspace,
    Eigen::MatrixXd* dense_active_gradient_direction) {
  if (workspace == nullptr) {
    throw std::invalid_argument("exact 2e fused workspace must not be null");
  }
  const int n_bf = ao_integral_input.n_basis_functions;
  const int n_ao = accepted_cache.n_active_orbitals;
  const std::size_t n_active_pairs =
      accepted_cache.active_pair_first_indices.size();
  const std::size_t n_bf_pairs =
      n_bf * (n_bf + 1) / 2;

  workspace->dense_active_direction = dense_active_direction;
  workspace->dense_active_gradient_direction.resize(
      n_bf, n_ao);
  workspace->dense_active_gradient_direction.setZero();

  // Fixed backprop term (same as non-fused path).
  accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache(
      accepted_cache.accepted_base_pair_gradients,
      workspace->dense_active_direction,
      accepted_cache,
      &workspace->dense_active_gradient_direction);

  // Reuse forward K*mixed: pair_gradients = (K * mixed) * gradient_matrix.
  // Saves one full apply_exact_ao_pair_kernel call (~500M FLOPs) per HVP.
  workspace->pair_gradients.resize(n_bf_pairs, n_active_pairs);
  workspace->pair_gradients.noalias() =
      directional_pair_products * accepted_cache.active_pair_gradient_matrix;

  // Final backprop (same as non-fused path).
  accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache(
      workspace->pair_gradients,
      accepted_cache.accepted_dense_active_coefficients,
      accepted_cache,
      &workspace->dense_active_gradient_direction);

  if (dense_active_gradient_direction != nullptr) {
    *dense_active_gradient_direction =
        workspace->dense_active_gradient_direction;
  }
}

Eigen::MatrixXd apply_exact_packed_active_two_electron_adjoint_hessian_vector(
    const std::vector<double>& packed_active_two_electron_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input,
    int n_ao,
    const ActiveSpaceTwoElectronResult* accepted_active_space_two_electron_result) {
  const auto accepted_cache =
      build_exact_packed_active_two_electron_adjoint_cache(
          packed_active_two_electron_gradient,
          dense_active_coefficients,
          ao_integral_input,
          n_ao,
          accepted_active_space_two_electron_result);
  return apply_exact_packed_active_two_electron_adjoint_hessian_vector(
      accepted_cache,
      dense_active_direction,
      ao_integral_input);
}

}  // namespace xmvb::vb
