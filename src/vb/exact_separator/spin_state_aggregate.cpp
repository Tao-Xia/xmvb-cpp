#include "vb/exact_separator/spin_state_aggregate.hpp"

#include "vb/matrices/two_electron_indexer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace xmvb::vb::exact_separator {

namespace {

struct SpinStateBlockFactor {
  const SpinPairStateKey* state = nullptr;
  Eigen::MatrixXd overlap_block;
  bool is_square = false;
  DeterminantOverlapResult overlap_result;
  Eigen::MatrixXd inverse_overlap_block;
};

void hash_combine(std::size_t* seed, std::size_t value) {
  if (seed == nullptr) {
    throw std::invalid_argument("hash seed must not be null");
  }
  *seed ^= value + 0x9e3779b97f4a7c15ULL + (*seed << 6) + (*seed >> 2);
}

std::size_t hash_occ_list(const std::vector<int>& occ) {
  std::size_t seed = occ.size();
  for (const int orbital : occ) {
    hash_combine(&seed, xmvb::to_size(orbital + 0x10000));
  }
  return seed;
}

struct DirectSpinAggregateCacheKey {
  std::vector<int> left_occ;
  std::vector<int> right_occ;
};

bool operator==(
    const DirectSpinAggregateCacheKey& left,
    const DirectSpinAggregateCacheKey& right) {
  return left.left_occ == right.left_occ &&
      left.right_occ == right.right_occ;
}

struct DirectSpinAggregateCacheKeyHasher {
  std::size_t operator()(const DirectSpinAggregateCacheKey& key) const {
    std::size_t seed = 0U;
    hash_combine(&seed, hash_occ_list(key.left_occ));
    hash_combine(&seed, hash_occ_list(key.right_occ));
    return seed;
  }
};

struct DirectSpinAggregateReuseCache {
  std::unordered_map<DirectSpinAggregateCacheKey, int, DirectSpinAggregateCacheKeyHasher>
      aggregate_index_by_key;
  std::vector<DirectSpinStateAggregate> aggregates;
};

int pair_count(int count) {
  if (count < 0) {
    throw std::invalid_argument("pair_count input must be non-negative");
  }
  return (count * (count - 1)) / 2;
}

void assign_concatenated_occ(
    const std::vector<int>& first,
    const std::vector<int>& second,
    std::vector<int>* result) {
  if (result == nullptr) {
    throw std::invalid_argument("result must not be null");
  }
  result->clear();
  result->reserve(first.size() + second.size());
  result->insert(result->end(), first.begin(), first.end());
  result->insert(result->end(), second.begin(), second.end());
}

Eigen::MatrixXd build_support_submatrix_matrix(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& support_storage,
    int support_size) {
  Eigen::MatrixXd block(
      static_cast<int>(right_occ.size()),
      static_cast<int>(left_occ.size()));
  for (int left_column = 0; left_column < static_cast<int>(left_occ.size()); ++left_column) {
    const int left_orbital = left_occ[xmvb::to_size(left_column)];
    for (int right_row = 0; right_row < static_cast<int>(right_occ.size()); ++right_row) {
      const int right_orbital = right_occ[xmvb::to_size(right_row)];
      block(right_row, left_column) =
          support_storage[xmvb::to_size(left_orbital) *
                              xmvb::to_size(support_size) +
                          xmvb::to_size(right_orbital)];
    }
  }
  return block;
}

void scatter_first_cofactor_to_support_coordinates(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const Eigen::MatrixXd& local_first_cofactor,
    Eigen::MatrixXd* support_first_cofactor) {
  if (support_first_cofactor == nullptr) {
    throw std::invalid_argument("support_first_cofactor must not be null");
  }
  if (local_first_cofactor.rows() != static_cast<int>(right_occ.size()) ||
      local_first_cofactor.cols() != static_cast<int>(left_occ.size())) {
    throw std::invalid_argument("local first-cofactor dimensions are inconsistent");
  }

  for (int left_column = 0; left_column < static_cast<int>(left_occ.size()); ++left_column) {
    const int left_orbital = left_occ[xmvb::to_size(left_column)];
    for (int right_row = 0; right_row < static_cast<int>(right_occ.size()); ++right_row) {
      const int right_orbital = right_occ[xmvb::to_size(right_row)];
      (*support_first_cofactor)(right_orbital, left_orbital) =
          local_first_cofactor(right_row, left_column);
    }
  }
}

double contract_local_first_cofactor_with_support_storage(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const Eigen::MatrixXd& local_first_cofactor,
    const std::vector<double>& support_one_electron_storage,
    int support_size) {
  if (local_first_cofactor.rows() != static_cast<int>(right_occ.size()) ||
      local_first_cofactor.cols() != static_cast<int>(left_occ.size())) {
    throw std::invalid_argument("local first-cofactor dimensions are inconsistent");
  }

  double total = 0.0;
  for (int left_column = 0; left_column < static_cast<int>(left_occ.size()); ++left_column) {
    const int left_orbital = left_occ[xmvb::to_size(left_column)];
    for (int right_row = 0; right_row < static_cast<int>(right_occ.size()); ++right_row) {
      const int right_orbital = right_occ[xmvb::to_size(right_row)];
      total +=
          support_one_electron_storage[xmvb::to_size(left_orbital) *
                                           xmvb::to_size(support_size) +
                                       xmvb::to_size(right_orbital)] *
          local_first_cofactor(right_row, left_column);
    }
  }
  return total;
}

double compute_same_spin_two_electron_from_first_cofactor(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const Eigen::MatrixXd& local_first_cofactor,
    double overlap,
    const std::vector<double>& packed_active_two_electron_integrals) {
  if (local_first_cofactor.rows() != static_cast<int>(right_occ.size()) ||
      local_first_cofactor.cols() != static_cast<int>(left_occ.size())) {
    throw std::invalid_argument("local first-cofactor dimensions are inconsistent");
  }
  if (std::abs(overlap) <= 1.0e-15 || left_occ.size() < 2U) {
    return 0.0;
  }

  double total = 0.0;
  const double inverse_overlap = 1.0 / overlap;
  for (int left_first = 0; left_first + 1 < static_cast<int>(left_occ.size()); ++left_first) {
    const int left_first_orbital = left_occ[xmvb::to_size(left_first)];
    for (int right_first = 0; right_first + 1 < static_cast<int>(right_occ.size()); ++right_first) {
      const int right_first_orbital = right_occ[xmvb::to_size(right_first)];
      const double cofactor_11 = local_first_cofactor(right_first, left_first);
      for (int left_second = left_first + 1;
           left_second < static_cast<int>(left_occ.size());
           ++left_second) {
        const int left_second_orbital = left_occ[xmvb::to_size(left_second)];
        const double cofactor_12 = local_first_cofactor(right_first, left_second);
        for (int right_second = right_first + 1;
             right_second < static_cast<int>(right_occ.size());
             ++right_second) {
          const int right_second_orbital =
              right_occ[xmvb::to_size(right_second)];
          const double cofactor_22 = local_first_cofactor(right_second, left_second);
          const double cofactor_21 = local_first_cofactor(right_second, left_first);
          const double second_cofactor =
              (cofactor_11 * cofactor_22 - cofactor_12 * cofactor_21) *
              inverse_overlap;
          const int direct_index =
              xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
                  right_first_orbital,
                  left_first_orbital,
                  right_second_orbital,
                  left_second_orbital);
          const int exchange_index =
              xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
                  right_first_orbital,
                  left_second_orbital,
                  right_second_orbital,
                  left_first_orbital);
          total +=
              (packed_active_two_electron_integrals[xmvb::to_size(direct_index)] -
               packed_active_two_electron_integrals[xmvb::to_size(exchange_index)]) *
              second_cofactor;
        }
      }
    }
  }
  return total;
}

void populate_direct_spin_state_aggregate(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const Eigen::MatrixXd& local_first_cofactor,
    double overlap,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    bool compute_same_spin_two_electron,
    bool materialize_support_first_cofactor,
    DirectSpinStateAggregate* aggregate) {
  if (aggregate == nullptr) {
    throw std::invalid_argument("aggregate must not be null");
  }

  aggregate->overlap = overlap;
  aggregate->left_occ = left_occ;
  aggregate->right_occ = right_occ;
  aggregate->local_first_cofactor = local_first_cofactor;
  if (materialize_support_first_cofactor) {
    aggregate->first_cofactor = Eigen::MatrixXd::Zero(support_size, support_size);
    scatter_first_cofactor_to_support_coordinates(
        left_occ,
        right_occ,
        local_first_cofactor,
        &aggregate->first_cofactor);
  } else {
    aggregate->first_cofactor.resize(0, 0);
  }
  aggregate->one_electron =
      contract_local_first_cofactor_with_support_storage(
          left_occ,
          right_occ,
          local_first_cofactor,
          support_one_electron_storage,
          support_size);
  if (compute_same_spin_two_electron) {
    aggregate->same_spin_two_electron =
        compute_same_spin_two_electron_from_first_cofactor(
            left_occ,
            right_occ,
            local_first_cofactor,
            overlap,
            packed_active_two_electron_integrals);
  }
}

double contract_local_opposite_spin_channel(
    const DirectSpinStateAggregate& alpha_aggregate,
    const DirectSpinStateAggregate& beta_aggregate,
    const std::vector<double>& packed_active_two_electron_integrals) {
  if (alpha_aggregate.local_first_cofactor.rows() !=
          static_cast<int>(alpha_aggregate.right_occ.size()) ||
      alpha_aggregate.local_first_cofactor.cols() !=
          static_cast<int>(alpha_aggregate.left_occ.size()) ||
      beta_aggregate.local_first_cofactor.rows() !=
          static_cast<int>(beta_aggregate.right_occ.size()) ||
      beta_aggregate.local_first_cofactor.cols() !=
          static_cast<int>(beta_aggregate.left_occ.size())) {
    throw std::invalid_argument(
        "local opposite-spin contraction received inconsistent aggregate dimensions");
  }

  double total = 0.0;
  for (int beta_col = 0; beta_col < static_cast<int>(beta_aggregate.left_occ.size()); ++beta_col) {
    const int beta_left_orbital =
        beta_aggregate.left_occ[xmvb::to_size(beta_col)];
    for (int beta_row = 0; beta_row < static_cast<int>(beta_aggregate.right_occ.size()); ++beta_row) {
      const double beta_value =
          beta_aggregate.local_first_cofactor(beta_row, beta_col);
      if (std::abs(beta_value) <= 1.0e-15) {
        continue;
      }
      const int beta_right_orbital =
          beta_aggregate.right_occ[xmvb::to_size(beta_row)];
      for (int alpha_col = 0;
           alpha_col < static_cast<int>(alpha_aggregate.left_occ.size());
           ++alpha_col) {
        const int alpha_left_orbital =
            alpha_aggregate.left_occ[xmvb::to_size(alpha_col)];
        for (int alpha_row = 0;
             alpha_row < static_cast<int>(alpha_aggregate.right_occ.size());
             ++alpha_row) {
          const double alpha_value =
              alpha_aggregate.local_first_cofactor(alpha_row, alpha_col);
          if (std::abs(alpha_value) <= 1.0e-15) {
            continue;
          }
          const int alpha_right_orbital =
              alpha_aggregate.right_occ[xmvb::to_size(alpha_row)];
          const int eri_index =
              xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
                  beta_right_orbital,
                  beta_left_orbital,
                  alpha_right_orbital,
                  alpha_left_orbital);
          total +=
              packed_active_two_electron_integrals[xmvb::to_size(eri_index)] *
              beta_value *
              alpha_value;
        }
      }
    }
  }
  return total;
}

std::vector<SpinStateBlockFactor> build_spin_state_block_factors(
    const std::vector<SpinPairStateKey>& states,
    const std::vector<double>& support_overlap_storage,
    int support_size,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }

  std::vector<SpinStateBlockFactor> factors;
  factors.reserve(states.size());
  for (const auto& state : states) {
    SpinStateBlockFactor factor;
    factor.state = &state;
    factor.overlap_block = build_support_submatrix_matrix(
        state.left_occ,
        state.right_occ,
        support_overlap_storage,
        support_size);
    factor.is_square = state.left_occ.size() == state.right_occ.size();
    if (factor.is_square) {
      ++(*subdeterminant_evaluations);
      factor.overlap_result = overlap_resolver.resolve_matrix(factor.overlap_block);
      if (factor.overlap_result.nullity == 0) {
        factor.inverse_overlap_block =
            xmvb::vb::build_inverse_overlap_submatrix_from_result(
                factor.overlap_result);
      }
    }
    factors.push_back(std::move(factor));
  }
  return factors;
}

bool all_spin_pair_states_square(const std::vector<SpinPairStateKey>& states) {
  for (const auto& state : states) {
    if (state.left_occ.size() != state.right_occ.size()) {
      return false;
    }
  }
  return true;
}

int max_spin_pair_state_size(const std::vector<SpinPairStateKey>& states) {
  int max_size = 0;
  for (const auto& state : states) {
    max_size = std::max(max_size, static_cast<int>(state.left_occ.size()));
  }
  return max_size;
}

bool try_build_regular_direct_spin_state_aggregate_from_root_pivot(
    const SpinStateBlockFactor& root_factor,
    const SpinStateBlockFactor& leaf_factor,
    const Eigen::MatrixXd& top_right_block,
    const Eigen::MatrixXd& bottom_left_block,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    bool compute_same_spin_two_electron,
    bool materialize_support_first_cofactor,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    DirectSpinStateAggregate* aggregate) {
  if (aggregate == nullptr || subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("root-pivot aggregate outputs must not be null");
  }
  if (!root_factor.is_square || !leaf_factor.is_square ||
      root_factor.overlap_result.nullity != 0) {
    return false;
  }

  const Eigen::MatrixXd& inverse_root = root_factor.inverse_overlap_block;
  const Eigen::MatrixXd root_inverse_times_top_right = inverse_root * top_right_block;
  const Eigen::MatrixXd bottom_left_times_root_inverse = bottom_left_block * inverse_root;
  Eigen::MatrixXd schur_complement = leaf_factor.overlap_block;
  schur_complement.noalias() -= bottom_left_times_root_inverse * top_right_block;

  ++(*subdeterminant_evaluations);
  const DeterminantOverlapResult schur_result =
      overlap_resolver.resolve_matrix(schur_complement);
  if (schur_result.nullity != 0) {
    return false;
  }

  const Eigen::MatrixXd inverse_leaf_schur =
      xmvb::vb::build_inverse_overlap_submatrix_from_result(schur_result);
  Eigen::MatrixXd inverse_root_root = inverse_root;
  inverse_root_root.noalias() +=
      root_inverse_times_top_right *
      inverse_leaf_schur *
      bottom_left_times_root_inverse;
  const Eigen::MatrixXd inverse_root_leaf =
      -(root_inverse_times_top_right * inverse_leaf_schur);
  const Eigen::MatrixXd inverse_leaf_root =
      -(inverse_leaf_schur * bottom_left_times_root_inverse);

  const double overlap =
      root_factor.overlap_result.overlap_determinant *
      schur_result.overlap_determinant;
  const int root_size = static_cast<int>(root_factor.state->left_occ.size());
  const int leaf_size = static_cast<int>(leaf_factor.state->left_occ.size());
  Eigen::MatrixXd local_first_cofactor = Eigen::MatrixXd::Zero(root_size + leaf_size, root_size + leaf_size);
  local_first_cofactor.topLeftCorner(root_size, root_size) =
      overlap * inverse_root_root.transpose();
  local_first_cofactor.topRightCorner(root_size, leaf_size) =
      overlap * inverse_leaf_root.transpose();
  local_first_cofactor.bottomLeftCorner(leaf_size, root_size) =
      overlap * inverse_root_leaf.transpose();
  local_first_cofactor.bottomRightCorner(leaf_size, leaf_size) =
      overlap * inverse_leaf_schur.transpose();

  std::vector<int> full_left_occ;
  std::vector<int> full_right_occ;
  assign_concatenated_occ(
      root_factor.state->left_occ,
      leaf_factor.state->left_occ,
      &full_left_occ);
  assign_concatenated_occ(
      root_factor.state->right_occ,
      leaf_factor.state->right_occ,
      &full_right_occ);
  populate_direct_spin_state_aggregate(
      full_left_occ,
      full_right_occ,
      local_first_cofactor,
      overlap,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      support_size,
      compute_same_spin_two_electron,
      materialize_support_first_cofactor,
      aggregate);
  return true;
}

bool try_build_regular_direct_spin_state_aggregate_from_leaf_pivot(
    const SpinStateBlockFactor& root_factor,
    const SpinStateBlockFactor& leaf_factor,
    const Eigen::MatrixXd& top_right_block,
    const Eigen::MatrixXd& bottom_left_block,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    bool compute_same_spin_two_electron,
    bool materialize_support_first_cofactor,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations,
    DirectSpinStateAggregate* aggregate) {
  if (aggregate == nullptr || subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("leaf-pivot aggregate outputs must not be null");
  }
  if (!root_factor.is_square || !leaf_factor.is_square ||
      leaf_factor.overlap_result.nullity != 0) {
    return false;
  }

  const Eigen::MatrixXd& inverse_leaf = leaf_factor.inverse_overlap_block;
  const Eigen::MatrixXd top_right_times_leaf_inverse = top_right_block * inverse_leaf;
  const Eigen::MatrixXd leaf_inverse_times_bottom_left = inverse_leaf * bottom_left_block;
  Eigen::MatrixXd schur_complement = root_factor.overlap_block;
  schur_complement.noalias() -= top_right_times_leaf_inverse * bottom_left_block;

  ++(*subdeterminant_evaluations);
  const DeterminantOverlapResult schur_result =
      overlap_resolver.resolve_matrix(schur_complement);
  if (schur_result.nullity != 0) {
    return false;
  }

  const Eigen::MatrixXd inverse_root_schur =
      xmvb::vb::build_inverse_overlap_submatrix_from_result(schur_result);
  const Eigen::MatrixXd inverse_root_leaf =
      -(inverse_root_schur * top_right_times_leaf_inverse);
  const Eigen::MatrixXd inverse_leaf_root =
      -(leaf_inverse_times_bottom_left * inverse_root_schur);
  Eigen::MatrixXd inverse_leaf_leaf = inverse_leaf;
  inverse_leaf_leaf.noalias() +=
      leaf_inverse_times_bottom_left *
      inverse_root_schur *
      top_right_times_leaf_inverse;

  const double overlap =
      leaf_factor.overlap_result.overlap_determinant *
      schur_result.overlap_determinant;
  const int root_size = static_cast<int>(root_factor.state->left_occ.size());
  const int leaf_size = static_cast<int>(leaf_factor.state->left_occ.size());
  Eigen::MatrixXd local_first_cofactor = Eigen::MatrixXd::Zero(root_size + leaf_size, root_size + leaf_size);
  local_first_cofactor.topLeftCorner(root_size, root_size) =
      overlap * inverse_root_schur.transpose();
  local_first_cofactor.topRightCorner(root_size, leaf_size) =
      overlap * inverse_leaf_root.transpose();
  local_first_cofactor.bottomLeftCorner(leaf_size, root_size) =
      overlap * inverse_root_leaf.transpose();
  local_first_cofactor.bottomRightCorner(leaf_size, leaf_size) =
      overlap * inverse_leaf_leaf.transpose();

  std::vector<int> full_left_occ;
  std::vector<int> full_right_occ;
  assign_concatenated_occ(
      root_factor.state->left_occ,
      leaf_factor.state->left_occ,
      &full_left_occ);
  assign_concatenated_occ(
      root_factor.state->right_occ,
      leaf_factor.state->right_occ,
      &full_right_occ);
  populate_direct_spin_state_aggregate(
      full_left_occ,
      full_right_occ,
      local_first_cofactor,
      overlap,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      support_size,
      compute_same_spin_two_electron,
      materialize_support_first_cofactor,
      aggregate);
  return true;
}

}  // namespace

int IndexedOneLeafDirectSpinAggregateTable::index(
    int root_state_index,
    int leaf_state_index) const {
  if (root_state_index < 0 || root_state_index >= root_state_count ||
      leaf_state_index < 0 || leaf_state_index >= leaf_state_count) {
    throw std::invalid_argument("one-leaf direct aggregate index is out of range");
  }
  return root_state_index * leaf_state_count + leaf_state_index;
}

Eigen::MatrixXd apply_opposite_spin_kernel_to_first_cofactor(
    const Eigen::MatrixXd& beta_first_cofactor,
    int support_size,
    const std::vector<double>& packed_active_two_electron_integrals) {
  if (beta_first_cofactor.rows() != support_size ||
      beta_first_cofactor.cols() != support_size) {
    throw std::invalid_argument(
        "beta first cofactor has inconsistent support dimensions");
  }

  Eigen::MatrixXd potential = Eigen::MatrixXd::Zero(support_size, support_size);
  for (int beta_col = 0; beta_col < support_size; ++beta_col) {
    for (int beta_row = 0; beta_row < support_size; ++beta_row) {
      const double beta_value = beta_first_cofactor(beta_row, beta_col);
      if (std::abs(beta_value) <= 1.0e-15) {
        continue;
      }
      for (int alpha_col = 0; alpha_col < support_size; ++alpha_col) {
        for (int alpha_row = 0; alpha_row < support_size; ++alpha_row) {
          const int eri_index =
              xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
                  beta_row,
                  beta_col,
                  alpha_row,
                  alpha_col);
          potential(alpha_row, alpha_col) +=
              packed_active_two_electron_integrals[xmvb::to_size(eri_index)] *
              beta_value;
        }
      }
    }
  }
  return potential;
}

double contract_direct_opposite_spin_channel(
    const Eigen::MatrixXd& alpha_first_cofactor,
    const Eigen::MatrixXd& beta_potential_first_cofactor) {
  if (alpha_first_cofactor.rows() != beta_potential_first_cofactor.rows() ||
      alpha_first_cofactor.cols() != beta_potential_first_cofactor.cols()) {
    throw std::invalid_argument(
        "opposite-spin first-cofactor contraction has inconsistent dimensions");
  }
  return alpha_first_cofactor.cwiseProduct(beta_potential_first_cofactor).sum();
}

DirectSpinStateAggregate build_direct_spin_state_aggregate_impl(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    bool compute_same_spin_two_electron,
    bool materialize_support_first_cofactor,
    const DeterminantOverlapResolver& overlap_resolver,
    const DeterminantHamiltonianResolver& hamiltonian_resolver,
    std::uint64_t* subdeterminant_evaluations);

DirectSpinStateAggregate build_direct_spin_state_aggregate(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    bool compute_same_spin_two_electron,
    const DeterminantOverlapResolver& overlap_resolver,
    const DeterminantHamiltonianResolver& hamiltonian_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  return build_direct_spin_state_aggregate_impl(
      left_occ,
      right_occ,
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      support_size,
      compute_same_spin_two_electron,
      true,
      overlap_resolver,
      hamiltonian_resolver,
      subdeterminant_evaluations);
}

DirectSpinStateAggregate build_direct_spin_state_aggregate_impl(
    const std::vector<int>& left_occ,
    const std::vector<int>& right_occ,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    bool compute_same_spin_two_electron,
    bool materialize_support_first_cofactor,
    const DeterminantOverlapResolver& overlap_resolver,
    const DeterminantHamiltonianResolver& hamiltonian_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }
  if (support_size <= 0) {
    throw std::invalid_argument("support_size must be positive");
  }
  if (left_occ.size() != right_occ.size()) {
    throw std::invalid_argument("direct spin aggregate requires equal left/right electron counts");
  }
  if (support_overlap_storage.size() !=
          xmvb::to_size(support_size) * xmvb::to_size(support_size) ||
      support_one_electron_storage.size() !=
          xmvb::to_size(support_size) * xmvb::to_size(support_size)) {
    throw std::invalid_argument("support matrices have inconsistent dimensions");
  }
  if (compute_same_spin_two_electron &&
      left_occ.size() >= 2U &&
      packed_active_two_electron_integrals.empty()) {
    throw std::invalid_argument(
        "same-spin direct aggregate requires packed two-electron integrals");
  }

  DirectSpinStateAggregate aggregate;
  aggregate.left_occ = left_occ;
  aggregate.right_occ = right_occ;
  if (left_occ.empty()) {
    aggregate.overlap = 1.0;
    if (materialize_support_first_cofactor) {
      aggregate.first_cofactor = Eigen::MatrixXd::Zero(support_size, support_size);
    }
    return aggregate;
  }

  const std::vector<double> overlap_submatrix_storage =
      xmvb::vb::build_overlap_submatrix(
          left_occ,
          right_occ,
          support_overlap_storage,
          support_size);
  ++(*subdeterminant_evaluations);
  const DeterminantOverlapResult overlap_result =
      overlap_resolver.resolve(
          overlap_submatrix_storage,
          static_cast<int>(left_occ.size()));
  aggregate.overlap = overlap_result.overlap_determinant;

  const Eigen::MatrixXd local_first_cofactor =
      xmvb::vb::calc_cofactor_1st(overlap_result);
  aggregate.local_first_cofactor = local_first_cofactor;
  if (materialize_support_first_cofactor) {
    aggregate.first_cofactor = Eigen::MatrixXd::Zero(support_size, support_size);
    scatter_first_cofactor_to_support_coordinates(
        left_occ,
        right_occ,
        local_first_cofactor,
        &aggregate.first_cofactor);
  }
  aggregate.one_electron = contract_local_first_cofactor_with_support_storage(
      left_occ,
      right_occ,
      local_first_cofactor,
      support_one_electron_storage,
      support_size);

  if (!compute_same_spin_two_electron ||
      left_occ.size() < 2U ||
      overlap_result.nullity >= 3) {
    return aggregate;
  }

  if (overlap_result.nullity == 0) {
    aggregate.same_spin_two_electron =
        compute_same_spin_two_electron_from_first_cofactor(
            left_occ,
            right_occ,
            local_first_cofactor,
            overlap_result.overlap_determinant,
            packed_active_two_electron_integrals);
    return aggregate;
  }

  const DeterminantHamiltonianResult hamiltonian_result =
      hamiltonian_resolver.resolve(
          left_occ,
          right_occ,
          overlap_submatrix_storage,
          overlap_result,
          std::vector<double>(
              xmvb::to_size(support_size) *
                  xmvb::to_size(support_size),
              0.0),
          support_size,
          packed_active_two_electron_integrals);
  aggregate.same_spin_two_electron = hamiltonian_result.total_hamiltonian;
  *subdeterminant_evaluations += static_cast<std::uint64_t>(
      pair_count(static_cast<int>(left_occ.size()))) *
      static_cast<std::uint64_t>(
          pair_count(static_cast<int>(right_occ.size())));
  return aggregate;
}

IndexedOneLeafDirectSpinAggregateTable
build_indexed_one_leaf_direct_spin_aggregate_table_impl(
    const std::vector<SpinPairStateKey>& root_states,
    const std::vector<SpinPairStateKey>& leaf_states,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    bool compute_same_spin_two_electron,
    bool enable_reused_block_family,
    bool materialize_support_first_cofactor,
    const DeterminantOverlapResolver& overlap_resolver,
    DirectSpinAggregateReuseCache* aggregate_cache,
    std::uint64_t* subdeterminant_evaluations) {
  if (subdeterminant_evaluations == nullptr) {
    throw std::invalid_argument("subdeterminant_evaluations must not be null");
  }
  if (support_size <= 0) {
    throw std::invalid_argument("support_size must be positive");
  }
  if (support_overlap_storage.size() !=
          xmvb::to_size(support_size) * xmvb::to_size(support_size) ||
      support_one_electron_storage.size() !=
          xmvb::to_size(support_size) * xmvb::to_size(support_size)) {
    throw std::invalid_argument("support matrices have inconsistent dimensions");
  }

  IndexedOneLeafDirectSpinAggregateTable table;
  table.root_state_count = static_cast<int>(root_states.size());
  table.leaf_state_count = static_cast<int>(leaf_states.size());
  table.aggregates.reserve(
      xmvb::to_size(table.root_state_count) *
      xmvb::to_size(table.leaf_state_count));
  const DeterminantHamiltonianResolver hamiltonian_resolver(overlap_resolver);

  const bool all_root_states_square = all_spin_pair_states_square(root_states);
  const bool all_leaf_states_square = all_spin_pair_states_square(leaf_states);
  const int root_max_size = max_spin_pair_state_size(root_states);
  const int leaf_max_size = max_spin_pair_state_size(leaf_states);
  const int smaller_family_size = std::min(root_max_size, leaf_max_size);
  const int larger_family_size = std::max(root_max_size, leaf_max_size);
  const bool is_leaf_small_regime =
      smaller_family_size > 0 &&
      2 * smaller_family_size <= larger_family_size;

  const bool use_reused_block_family =
      enable_reused_block_family &&
      all_root_states_square &&
      all_leaf_states_square &&
      is_leaf_small_regime;
  std::vector<int> full_left_occ;
  std::vector<int> full_right_occ;

  std::vector<SpinStateBlockFactor> root_factors;
  std::vector<SpinStateBlockFactor> leaf_factors;
  if (use_reused_block_family) {
    root_factors = build_spin_state_block_factors(
        root_states,
        support_overlap_storage,
        support_size,
        overlap_resolver,
        subdeterminant_evaluations);
    leaf_factors = build_spin_state_block_factors(
        leaf_states,
        support_overlap_storage,
        support_size,
        overlap_resolver,
        subdeterminant_evaluations);
  }

  // Reuse is intentionally scoped to one paired alpha/beta table build. The
  // aggregate only depends on the final full occupied-orbital pair together
  // with the fixed support operators of the current call, so identical alpha
  // and beta full states can safely share the already materialized exact
  // determinant/cofactor payload.
  for (int root_state_index = 0; root_state_index < table.root_state_count; ++root_state_index) {
    const SpinPairStateKey& root_state =
        root_states[xmvb::to_size(root_state_index)];
    for (int leaf_state_index = 0; leaf_state_index < table.leaf_state_count; ++leaf_state_index) {
      const SpinPairStateKey& leaf_state =
          leaf_states[xmvb::to_size(leaf_state_index)];
      assign_concatenated_occ(root_state.left_occ, leaf_state.left_occ, &full_left_occ);
      assign_concatenated_occ(root_state.right_occ, leaf_state.right_occ, &full_right_occ);

      if (aggregate_cache != nullptr) {
        const auto cached_iterator = aggregate_cache->aggregate_index_by_key.find(
            DirectSpinAggregateCacheKey{
            full_left_occ,
            full_right_occ,
        });
        if (cached_iterator != aggregate_cache->aggregate_index_by_key.end()) {
          table.aggregates.push_back(
              aggregate_cache->aggregates[xmvb::to_size(cached_iterator->second)]);
          ++table.total_state_count;
          continue;
        }
      }

      DirectSpinStateAggregate aggregate;
      if (!use_reused_block_family) {
        aggregate = build_direct_spin_state_aggregate_impl(
            full_left_occ,
            full_right_occ,
            support_overlap_storage,
            support_one_electron_storage,
            packed_active_two_electron_integrals,
            support_size,
            compute_same_spin_two_electron,
            materialize_support_first_cofactor,
            overlap_resolver,
            hamiltonian_resolver,
            subdeterminant_evaluations);
      } else {
        const SpinStateBlockFactor& root_factor =
            root_factors[xmvb::to_size(root_state_index)];
        const SpinStateBlockFactor& leaf_factor =
            leaf_factors[xmvb::to_size(leaf_state_index)];
        const Eigen::MatrixXd top_right_block = build_support_submatrix_matrix(
            leaf_factor.state->left_occ,
            root_factor.state->right_occ,
            support_overlap_storage,
            support_size);
        const Eigen::MatrixXd bottom_left_block = build_support_submatrix_matrix(
            root_factor.state->left_occ,
            leaf_factor.state->right_occ,
            support_overlap_storage,
            support_size);

        const bool prefer_root_pivot =
            leaf_factor.state->left_occ.size() <= root_factor.state->left_occ.size();
        bool built = false;
        if (prefer_root_pivot) {
          built = try_build_regular_direct_spin_state_aggregate_from_root_pivot(
              root_factor,
              leaf_factor,
              top_right_block,
              bottom_left_block,
              support_one_electron_storage,
              packed_active_two_electron_integrals,
              support_size,
              compute_same_spin_two_electron,
              materialize_support_first_cofactor,
              overlap_resolver,
              subdeterminant_evaluations,
              &aggregate);
          if (!built) {
            built = try_build_regular_direct_spin_state_aggregate_from_leaf_pivot(
                root_factor,
                leaf_factor,
                top_right_block,
                bottom_left_block,
                support_one_electron_storage,
                packed_active_two_electron_integrals,
                support_size,
                compute_same_spin_two_electron,
                materialize_support_first_cofactor,
                overlap_resolver,
                subdeterminant_evaluations,
                &aggregate);
          }
        } else {
          built = try_build_regular_direct_spin_state_aggregate_from_leaf_pivot(
              root_factor,
              leaf_factor,
              top_right_block,
              bottom_left_block,
              support_one_electron_storage,
              packed_active_two_electron_integrals,
              support_size,
              compute_same_spin_two_electron,
              materialize_support_first_cofactor,
              overlap_resolver,
              subdeterminant_evaluations,
              &aggregate);
          if (!built) {
            built = try_build_regular_direct_spin_state_aggregate_from_root_pivot(
                root_factor,
                leaf_factor,
                top_right_block,
                bottom_left_block,
                support_one_electron_storage,
                packed_active_two_electron_integrals,
                support_size,
                compute_same_spin_two_electron,
                materialize_support_first_cofactor,
                overlap_resolver,
                subdeterminant_evaluations,
                &aggregate);
          }
        }

        if (!built) {
          aggregate = build_direct_spin_state_aggregate_impl(
              full_left_occ,
              full_right_occ,
              support_overlap_storage,
              support_one_electron_storage,
              packed_active_two_electron_integrals,
              support_size,
              compute_same_spin_two_electron,
              materialize_support_first_cofactor,
              overlap_resolver,
              hamiltonian_resolver,
              subdeterminant_evaluations);
        }
      }

      if (aggregate_cache != nullptr) {
        const int aggregate_index =
            static_cast<int>(aggregate_cache->aggregates.size());
        aggregate_cache->aggregate_index_by_key.emplace(
            DirectSpinAggregateCacheKey{
                aggregate.left_occ,
                aggregate.right_occ,
            },
            aggregate_index);
        aggregate_cache->aggregates.push_back(aggregate);
      }
      table.aggregates.push_back(std::move(aggregate));
      ++table.total_state_count;
    }
  }

  return table;
}

IndexedOneLeafDirectSpinAggregateTable
build_indexed_one_leaf_direct_spin_aggregate_table(
    const std::vector<SpinPairStateKey>& root_states,
    const std::vector<SpinPairStateKey>& leaf_states,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    bool compute_same_spin_two_electron,
    bool enable_reused_block_family,
    bool materialize_support_first_cofactor,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  return build_indexed_one_leaf_direct_spin_aggregate_table_impl(
      root_states,
      leaf_states,
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      support_size,
      compute_same_spin_two_electron,
      enable_reused_block_family,
      materialize_support_first_cofactor,
      overlap_resolver,
      nullptr,
      subdeterminant_evaluations);
}

PairedOneLeafDirectSpinAggregateTables
build_paired_one_leaf_direct_spin_aggregate_tables(
    const std::vector<SpinPairStateKey>& root_alpha_states,
    const std::vector<SpinPairStateKey>& leaf_alpha_states,
    const std::vector<SpinPairStateKey>& root_beta_states,
    const std::vector<SpinPairStateKey>& leaf_beta_states,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& support_one_electron_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    bool compute_same_spin_two_electron,
    bool enable_reused_block_family,
    bool materialize_support_first_cofactor,
    const DeterminantOverlapResolver& overlap_resolver,
    std::uint64_t* subdeterminant_evaluations) {
  PairedOneLeafDirectSpinAggregateTables result;
  result.alpha_table = build_indexed_one_leaf_direct_spin_aggregate_table(
      root_alpha_states,
      leaf_alpha_states,
      support_overlap_storage,
      support_one_electron_storage,
      packed_active_two_electron_integrals,
      support_size,
      compute_same_spin_two_electron,
      enable_reused_block_family,
      materialize_support_first_cofactor,
      overlap_resolver,
      subdeterminant_evaluations);
  if (root_alpha_states == root_beta_states &&
      leaf_alpha_states == leaf_beta_states) {
    result.beta_table = result.alpha_table;
    return result;
  }
  if (enable_reused_block_family) {
    result.beta_table = build_indexed_one_leaf_direct_spin_aggregate_table(
        root_beta_states,
        leaf_beta_states,
        support_overlap_storage,
        support_one_electron_storage,
        packed_active_two_electron_integrals,
        support_size,
        compute_same_spin_two_electron,
        enable_reused_block_family,
        materialize_support_first_cofactor,
        overlap_resolver,
        subdeterminant_evaluations);
    return result;
  }

  std::unordered_map<DirectSpinAggregateCacheKey, int, DirectSpinAggregateCacheKeyHasher>
      alpha_aggregate_index_by_full_state;
  alpha_aggregate_index_by_full_state.reserve(result.alpha_table.aggregates.size());
  std::vector<int> full_left_occ;
  std::vector<int> full_right_occ;
  int alpha_flat_index = 0;
  for (const auto& root_state : root_alpha_states) {
    for (const auto& leaf_state : leaf_alpha_states) {
      assign_concatenated_occ(root_state.left_occ, leaf_state.left_occ, &full_left_occ);
      assign_concatenated_occ(root_state.right_occ, leaf_state.right_occ, &full_right_occ);
      alpha_aggregate_index_by_full_state.emplace(
          DirectSpinAggregateCacheKey{full_left_occ, full_right_occ},
          alpha_flat_index);
      ++alpha_flat_index;
    }
  }

  result.beta_table.root_state_count = static_cast<int>(root_beta_states.size());
  result.beta_table.leaf_state_count = static_cast<int>(leaf_beta_states.size());
  result.beta_table.aggregates.reserve(
      xmvb::to_size(result.beta_table.root_state_count) *
      xmvb::to_size(result.beta_table.leaf_state_count));
  const DeterminantHamiltonianResolver hamiltonian_resolver(overlap_resolver);
  for (const auto& root_state : root_beta_states) {
    for (const auto& leaf_state : leaf_beta_states) {
      assign_concatenated_occ(root_state.left_occ, leaf_state.left_occ, &full_left_occ);
      assign_concatenated_occ(root_state.right_occ, leaf_state.right_occ, &full_right_occ);
      const auto alpha_match = alpha_aggregate_index_by_full_state.find(
          DirectSpinAggregateCacheKey{full_left_occ, full_right_occ});
      if (alpha_match != alpha_aggregate_index_by_full_state.end()) {
        result.beta_table.aggregates.push_back(
            result.alpha_table.aggregates[xmvb::to_size(alpha_match->second)]);
      } else {
        result.beta_table.aggregates.push_back(
            build_direct_spin_state_aggregate_impl(
                full_left_occ,
                full_right_occ,
                support_overlap_storage,
                support_one_electron_storage,
                packed_active_two_electron_integrals,
                support_size,
                compute_same_spin_two_electron,
                materialize_support_first_cofactor,
                overlap_resolver,
                hamiltonian_resolver,
                subdeterminant_evaluations));
      }
      ++result.beta_table.total_state_count;
    }
  }
  return result;
}

OneLeafDirectAggregateContractionResult contract_one_leaf_direct_aggregate_channels(
    const ComponentSpinCoefficientOperator& root_operator,
    const ComponentSpinCoefficientOperator& leaf_operator,
    const IndexedOneLeafDirectSpinAggregateTable& alpha_table,
    const IndexedOneLeafDirectSpinAggregateTable& beta_table,
    const std::vector<double>* packed_active_two_electron_integrals) {
  if (alpha_table.root_state_count != static_cast<int>(root_operator.alpha_states.size()) ||
      alpha_table.leaf_state_count != static_cast<int>(leaf_operator.alpha_states.size()) ||
      beta_table.root_state_count != static_cast<int>(root_operator.beta_states.size()) ||
      beta_table.leaf_state_count != static_cast<int>(leaf_operator.beta_states.size())) {
    throw std::invalid_argument(
        "direct aggregate fused contraction received inconsistent dimensions");
  }

  OneLeafDirectAggregateContractionResult result;
  for (const auto& root_entry : root_operator.entries) {
    for (const auto& leaf_entry : leaf_operator.entries) {
      const double coefficient =
          root_entry.coefficient * leaf_entry.coefficient;
      const DirectSpinStateAggregate& alpha_aggregate =
          alpha_table.aggregates[xmvb::to_size(
              alpha_table.index(
                  root_entry.alpha_state_index,
                  leaf_entry.alpha_state_index))];
      const DirectSpinStateAggregate& beta_aggregate =
          beta_table.aggregates[xmvb::to_size(
              beta_table.index(
                  root_entry.beta_state_index,
                  leaf_entry.beta_state_index))];

      const double overlap_term =
          coefficient * alpha_aggregate.overlap * beta_aggregate.overlap;
      result.overlap += overlap_term;
      result.one_electron +=
          coefficient *
          (alpha_aggregate.one_electron * beta_aggregate.overlap +
           alpha_aggregate.overlap * beta_aggregate.one_electron);
      result.same_spin_alpha_two_electron +=
          coefficient *
          alpha_aggregate.same_spin_two_electron *
          beta_aggregate.overlap;
      result.same_spin_beta_two_electron +=
          coefficient *
          alpha_aggregate.overlap *
          beta_aggregate.same_spin_two_electron;

      if (packed_active_two_electron_integrals != nullptr) {
        result.opposite_spin_two_electron +=
            coefficient *
            contract_local_opposite_spin_channel(
                alpha_aggregate,
                beta_aggregate,
                *packed_active_two_electron_integrals);
      }
    }
  }
  return result;
}

}  // namespace xmvb::vb::exact_separator
