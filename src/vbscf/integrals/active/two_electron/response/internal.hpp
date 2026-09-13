#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "vbscf/integrals/active/two_electron/response/types.hpp"

namespace xmvb::vb::detail {

struct ActivePair {
  int first = 0;
  int second = 0;
};

inline std::size_t ao_pair_index(int first, int second) {
  if (first >= second) {
    const std::size_t first_index = first;
    return first_index * (first_index + 1) / 2 + second;
  }
  const std::size_t second_index = second;
  return second_index * (second_index + 1) / 2 + first;
}

void resize_for_overwrite(
    std::vector<double>* values,
    std::size_t size);

void build_ao_pair_component_tables(
    int n_bf,
    std::vector<int>* first_indices,
    std::vector<int>* second_indices);

std::vector<ActivePair> build_active_pair_list(int n_ao);

ExactCtxPairMatrix build_active_pair_gradient_matrix(
    const std::vector<double>& packed_active_two_electron_gradient,
    const std::vector<ActivePair>& active_pairs);

void build_ao_pair_to_active_pair_coefficients(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    int n_bf,
    int n_ao,
    const std::vector<ActivePair>& active_pairs,
    ExactCtxPairMatrix* ao_pair_to_active_pair_coefficients);

void build_mixed_ao_pair_to_active_pair_coefficients(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    int n_bf,
    int n_ao,
    const std::vector<ActivePair>& active_pairs,
    ExactCtxPairMatrix* mixed_ao_pair_to_active_pair_coefficients);

void build_mixed_ao_pair_to_active_pair_coefficients_from_cache(
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    ExactCtxPairMatrix* mixed_ao_pair_to_active_pair_coefficients);

void multiply_pair_coefficients_by_gradient_matrix(
    const ExactCtxPairMatrix& ao_pair_to_active_pair_coefficients,
    const ExactCtxPairMatrix& active_pair_gradient_matrix,
    ExactCtxPairMatrix* transformed_pair_coefficients);

ExactCtxPairMatrix multiply_pair_coefficients_by_gradient_matrix(
    const ExactCtxPairMatrix& ao_pair_to_active_pair_coefficients,
    const ExactCtxPairMatrix& active_pair_gradient_matrix);

void accumulate_backpropagated_pair_coefficients_to_dense_active_coefficients_from_cache(
    const ExactCtxPairMatrix& pair_gradients,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_coefficients,
    const ExactPackedActiveTwoElectronAdjointCache& cache,
    Eigen::MatrixXd* dense_active_gradients);

}  // namespace xmvb::vb::detail
