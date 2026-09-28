#include "vbscf/structures/orthogonal_ci/sigma.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>

#include "core/openmp.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/structures/orthogonal_ci/planner.hpp"

namespace xmvb::vb {
namespace {

std::uint64_t determinant_mask(
    const std::vector<int>& occupied,
    int n_orbitals) {
  std::uint64_t mask = 0;
  int previous = -1;
  for (const int orbital : occupied) {
    if (orbital <= previous || orbital < 0 || orbital >= n_orbitals) {
      throw std::invalid_argument("invalid direct-CI determinant occupation");
    }
    mask |= std::uint64_t{1} << orbital;
    previous = orbital;
  }
  return mask;
}

int only_set_bit(std::uint64_t mask) {
  if (mask == 0 || (mask & (mask - 1)) != 0) {
    throw std::logic_error("direct-CI single excitation is inconsistent");
  }
  return __builtin_ctzll(mask);
}

double replacement_sign(std::uint64_t source, int removed, int inserted) {
  const int low = std::min(removed, inserted);
  const int high = std::max(removed, inserted);
  std::uint64_t between = 0;
  if (high - low > 1) {
    between = source &
        (((std::uint64_t{1} << high) - 1) ^
         ((std::uint64_t{1} << (low + 1)) - 1));
  }
  return __builtin_popcountll(between) % 2 == 0 ? 1.0 : -1.0;
}

void add_symmetric_entry(
    Eigen::MatrixXd* matrix,
    int row,
    int column,
    double value) {
  if (row == column) {
    (*matrix)(row, column) += value;
    return;
  }
  (*matrix)(row, column) += 0.5 * value;
  (*matrix)(column, row) += 0.5 * value;
}

}  // namespace

void DirectCiSigmaAction::DensityConnections::reserve(std::size_t capacity) {
  sources.reserve(capacity);
  pairs.reserve(capacity);
  signs.reserve(capacity);
  created_orbitals.reserve(capacity);
  annihilated_orbitals.reserve(capacity);
}

void DirectCiSigmaAction::DensityConnections::append(
    int source,
    int pair,
    double sign,
    int created_orbital,
    int annihilated_orbital) {
  sources.push_back(source);
  pairs.push_back(pair);
  signs.push_back(sign);
  created_orbitals.push_back(created_orbital);
  annihilated_orbitals.push_back(annihilated_orbital);
}

std::size_t
DirectCiSigmaAction::DensityConnections::dynamic_bytes() const noexcept {
  return sources.capacity() * sizeof(int) +
      pairs.capacity() * sizeof(int) +
      signs.capacity() * sizeof(double) +
      created_orbitals.capacity() * sizeof(int) +
      annihilated_orbitals.capacity() * sizeof(int);
}

DirectCiSigmaAction::DirectCiSigmaAction(
    const std::vector<std::vector<int>>& alpha_determinants,
    const std::vector<std::vector<int>>& beta_determinants,
    const OrthogonalActiveIntegrals& integrals)
    : n_orbitals_(static_cast<int>(integrals.one_electron.rows())),
      n_alpha_(static_cast<int>(alpha_determinants.size())),
      n_beta_(static_cast<int>(beta_determinants.size())),
      pair_kernel_(integrals.pair_kernel) {
  if (n_orbitals_ <= 0 || n_orbitals_ > 63 ||
      integrals.one_electron.cols() != n_orbitals_ ||
      n_alpha_ <= 0 || n_beta_ <= 0) {
    throw std::invalid_argument("invalid orthogonal direct-CI sigma input");
  }
  if (!plan_orthogonal_direct_ci_action(
           alpha_determinants,
           beta_determinants,
           n_orbitals_,
           1).complete()) {
    throw std::invalid_argument(
        "orthogonal direct-CI sigma requires complete fixed-spin spaces");
  }
  const int n_electrons =
      static_cast<int>(alpha_determinants.front().size() +
                       beta_determinants.front().size());
  hamiltonian_pair_kernel_ =
      build_hamiltonian_pair_kernel(integrals, n_electrons);
  alpha_ = build_spin_connections(alpha_determinants);
  if (beta_determinants != alpha_determinants) {
    distinct_beta_.emplace(build_spin_connections(beta_determinants));
  }
}

DirectCiSigmaAction::SpinConnections
DirectCiSigmaAction::build_spin_connections(
    const std::vector<std::vector<int>>& determinants) const {
  const int dimension = static_cast<int>(determinants.size());
  std::vector<std::uint64_t> masks(dimension);
  std::unordered_map<std::uint64_t, int> index_by_mask;
  index_by_mask.reserve(determinants.size());
  for (int determinant = 0; determinant < dimension; ++determinant) {
    masks[determinant] = determinant_mask(
        determinants[determinant], n_orbitals_);
    if (!index_by_mask.emplace(masks[determinant], determinant).second) {
      throw std::invalid_argument(
          "direct-CI determinant space contains a duplicate");
    }
  }
  SpinConnections result;
  result.singles.resize(dimension);
  result.occupied = determinants;
  const int n_threads = std::max(
      1,
      std::min(effective_openmp_thread_count(), dimension));

#pragma omp parallel for schedule(dynamic, 1) if(n_threads > 1) num_threads(n_threads)
  for (int target = 0; target < dimension; ++target) {
    auto& singles = result.singles[target];
    const auto& target_occupied = determinants[target];
    std::vector<int> target_virtual;
    target_virtual.reserve(n_orbitals_ - target_occupied.size());
    for (int orbital = 0; orbital < n_orbitals_; ++orbital) {
      if ((masks[target] & (std::uint64_t{1} << orbital)) == 0) {
        target_virtual.push_back(orbital);
      }
    }
    const std::size_t n_singles =
        target_occupied.size() * target_virtual.size();
    singles.reserve(n_singles);

    const auto add_source = [&](std::uint64_t source_mask) {
      const auto source_entry = index_by_mask.find(source_mask);
      if (source_entry == index_by_mask.end()) {
        throw std::logic_error(
            "direct-CI excitation left the fixed-spin determinant space");
      }
      const int source = source_entry->second;
      const int removed = only_set_bit(source_mask & ~masks[target]);
      const int inserted = only_set_bit(masks[target] & ~source_mask);
      const double sign = replacement_sign(source_mask, removed, inserted);
      singles.append(
          source,
          TwoElectronIndexer::packed_pair_index(removed, inserted),
          sign,
          inserted,
          removed);
    };

    for (const int removed_from_target : target_occupied) {
      for (const int inserted_into_source : target_virtual) {
        const std::uint64_t source_mask =
            (masks[target] ^
             (std::uint64_t{1} << removed_from_target)) |
            (std::uint64_t{1} << inserted_into_source);
        add_source(source_mask);
      }
    }
  }
  return result;
}

Eigen::MatrixXd DirectCiSigmaAction::build_hamiltonian_pair_kernel(
    const OrthogonalActiveIntegrals& integrals,
    int n_electrons) const {
  const int n_pairs = static_cast<int>(pair_kernel_.rows());
  if (n_electrons <= 0 || pair_kernel_.cols() != n_pairs ||
      integrals.one_electron.rows() != n_orbitals_ ||
      integrals.one_electron.cols() != n_orbitals_ ||
      !pair_kernel_.allFinite()) {
    throw std::invalid_argument(
        "invalid direct-CI packed Hamiltonian kernel input");
  }

  // In an orthonormal active basis,
  //   H = sum_pq h_pq E_pq
  //     + 1/2 sum_pqrs (pq|rs) (E_pq E_rs - delta_qr E_ps).
  // Absorb the one-body and contraction terms into the packed pair kernel so
  // the complete Hamiltonian is one scatter--GEMM--gather operation.
  Eigen::MatrixXd effective_one_electron = integrals.one_electron;
  for (int row = 0; row < n_orbitals_; ++row) {
    for (int column = 0; column < n_orbitals_; ++column) {
      double contraction = 0.0;
      for (int orbital = 0; orbital < n_orbitals_; ++orbital) {
        contraction += pair_kernel_(
            TwoElectronIndexer::packed_pair_index(row, orbital),
            TwoElectronIndexer::packed_pair_index(orbital, column));
      }
      effective_one_electron(row, column) -= 0.5 * contraction;
    }
  }
  effective_one_electron /= static_cast<double>(n_electrons);

  Eigen::MatrixXd result = 0.5 * pair_kernel_;
  for (int orbital = 0; orbital < n_orbitals_; ++orbital) {
    const int diagonal_pair =
        TwoElectronIndexer::packed_pair_index(orbital, orbital);
    for (int row = 0; row < n_orbitals_; ++row) {
      for (int column = 0; column <= row; ++column) {
        const int pair =
            TwoElectronIndexer::packed_pair_index(row, column);
        const double value = 0.5 * effective_one_electron(row, column);
        result(diagonal_pair, pair) += value;
        result(pair, diagonal_pair) += value;
      }
    }
  }
  return 0.5 * (result + result.transpose());
}

const DirectCiSigmaAction::SpinConnections&
DirectCiSigmaAction::beta_connections() const noexcept {
  return distinct_beta_.has_value() ? *distinct_beta_ : alpha_;
}

void DirectCiSigmaAction::scatter_one_body_intermediate(
    const Eigen::Ref<const Eigen::MatrixXd>& coefficients,
    int alpha_begin,
    int alpha_count,
    int beta_target,
    Eigen::Ref<Eigen::MatrixXd> intermediate) const {
  const int block_width = static_cast<int>(coefficients.cols()) / n_beta_;
  const int active_columns = alpha_count * block_width;
  const SpinConnections& beta_graph = beta_connections();
  if (alpha_begin < 0 || alpha_count <= 0 ||
      alpha_begin + alpha_count > n_alpha_ ||
      beta_target < 0 || beta_target >= n_beta_ ||
      intermediate.rows() != hamiltonian_pair_kernel_.rows() ||
      intermediate.cols() != active_columns) {
    throw std::invalid_argument(
        "direct-CI scatter intermediate has incompatible dimensions");
  }
  intermediate.setZero();

  const auto add_beta_link = [&] (
      int beta_source, int pair, double sign) {
    for (int block = 0; block < block_width; ++block) {
      intermediate.row(pair)
          .segment(block * alpha_count, alpha_count)
          .noalias() += sign * coefficients.block(
              alpha_begin,
              block * n_beta_ + beta_source,
              alpha_count,
              1).transpose();
    }
  };
  for (const int occupied : beta_graph.occupied[beta_target]) {
    add_beta_link(
        beta_target,
        TwoElectronIndexer::packed_pair_index(occupied, occupied),
        1.0);
  }
  const DensityConnections& beta_singles = beta_graph.singles[beta_target];
  for (std::size_t link = 0; link < beta_singles.size(); ++link) {
    add_beta_link(
        beta_singles.sources[link],
        beta_singles.pairs[link],
        beta_singles.signs[link]);
  }

  for (int local_alpha = 0; local_alpha < alpha_count; ++local_alpha) {
    const int alpha_target = alpha_begin + local_alpha;
    const auto add_alpha_link = [&] (
        int alpha_source, int pair, double sign) {
      for (int block = 0; block < block_width; ++block) {
        intermediate(pair, block * alpha_count + local_alpha) +=
            sign * coefficients(
                alpha_source, block * n_beta_ + beta_target);
      }
    };
    for (const int occupied : alpha_.occupied[alpha_target]) {
      add_alpha_link(
          alpha_target,
          TwoElectronIndexer::packed_pair_index(occupied, occupied),
          1.0);
    }
    const DensityConnections& alpha_singles = alpha_.singles[alpha_target];
    for (std::size_t link = 0; link < alpha_singles.size(); ++link) {
      add_alpha_link(
          alpha_singles.sources[link],
          alpha_singles.pairs[link],
          alpha_singles.signs[link]);
    }
  }
}

Eigen::MatrixXd DirectCiSigmaAction::apply(
    const Eigen::Ref<const Eigen::MatrixXd>& coefficients) const {
  if (coefficients.rows() != n_alpha_ || coefficients.cols() <= 0 ||
      coefficients.cols() % n_beta_ != 0) {
    throw std::invalid_argument(
        "direct-CI coefficient block has incompatible dimensions");
  }
  const int block_width = static_cast<int>(coefficients.cols()) / n_beta_;
  const SpinConnections& beta_graph = beta_connections();
  const int n_pairs = static_cast<int>(hamiltonian_pair_kernel_.rows());
  Eigen::MatrixXd sigma = Eigen::MatrixXd::Zero(
      n_alpha_, coefficients.cols());
  const int n_threads = std::max(
      1, std::min(effective_openmp_thread_count(), n_beta_));

  // Bound all thread-local scatter, transformed-scatter, and reduction
  // buffers by one explicit workspace budget.  A wider alpha tile increases
  // GEMM efficiency without changing the mathematical action.
  constexpr std::size_t kWorkspaceBytes = 64ULL * 1024ULL * 1024ULL;
  constexpr int kMaximumAlphaTile = 160;
  const std::size_t bytes_per_alpha =
      static_cast<std::size_t>(n_threads) * block_width *
      (static_cast<std::size_t>(n_beta_) + 2ULL * n_pairs) * sizeof(double);
  const int alpha_tile = std::min(
      n_alpha_,
      std::max(
          1,
          std::min(
              kMaximumAlphaTile,
              static_cast<int>(kWorkspaceBytes /
                  std::max<std::size_t>(bytes_per_alpha, 1)))));
  const int maximum_columns = alpha_tile * block_width;
  std::vector<Eigen::MatrixXd> partial_sigma(
      n_threads,
      Eigen::MatrixXd::Zero(maximum_columns, n_beta_));

#pragma omp parallel if(n_threads > 1) num_threads(n_threads)
  {
    int thread = 0;
#ifdef _OPENMP
    thread = omp_get_thread_num();
#endif
    Eigen::MatrixXd intermediate(n_pairs, maximum_columns);
    Eigen::MatrixXd transformed(n_pairs, maximum_columns);
    Eigen::MatrixXd& beta_reduction = partial_sigma[thread];

    for (int alpha_begin = 0;
         alpha_begin < n_alpha_;
         alpha_begin += alpha_tile) {
      const int alpha_count = std::min(alpha_tile, n_alpha_ - alpha_begin);
      const int active_columns = alpha_count * block_width;
      beta_reduction.topRows(active_columns).setZero();

#pragma omp for schedule(static)
      for (int beta_target = 0; beta_target < n_beta_; ++beta_target) {
        auto active_intermediate = intermediate.leftCols(active_columns);
        scatter_one_body_intermediate(
            coefficients,
            alpha_begin,
            alpha_count,
            beta_target,
            active_intermediate);

        auto active_transformed = transformed.leftCols(active_columns);
        active_transformed.noalias() =
            hamiltonian_pair_kernel_ * active_intermediate;

        // Alpha links write one beta column owned by this loop iteration, so
        // no synchronization is required.
        for (int local_alpha = 0;
             local_alpha < alpha_count;
             ++local_alpha) {
          const int alpha_target = alpha_begin + local_alpha;
          const auto gather_alpha_link = [&] (
              int alpha_source, int pair, double sign) {
            for (int block = 0; block < block_width; ++block) {
              sigma(alpha_source, block * n_beta_ + beta_target) +=
                  sign * active_transformed(
                      pair, block * alpha_count + local_alpha);
            }
          };
          for (const int occupied : alpha_.occupied[alpha_target]) {
            gather_alpha_link(
                alpha_target,
                TwoElectronIndexer::packed_pair_index(occupied, occupied),
                1.0);
          }
        const DensityConnections& alpha_singles =
              alpha_.singles[alpha_target];
          for (std::size_t link = 0; link < alpha_singles.size(); ++link) {
            gather_alpha_link(
                alpha_singles.sources[link],
                alpha_singles.pairs[link],
                alpha_singles.signs[link]);
          }
        }

        // Contributions scattered through beta links can collide between
        // beta targets.  Accumulate them in one bounded buffer per thread.
        const auto gather_beta_link = [&] (
            int beta_source, int pair, double sign) {
          for (int column = 0; column < active_columns; ++column) {
            beta_reduction(column, beta_source) +=
                sign * active_transformed(pair, column);
          }
        };
        for (const int occupied : beta_graph.occupied[beta_target]) {
          gather_beta_link(
              beta_target,
              TwoElectronIndexer::packed_pair_index(occupied, occupied),
              1.0);
        }
        const DensityConnections& beta_singles =
            beta_graph.singles[beta_target];
        for (std::size_t link = 0; link < beta_singles.size(); ++link) {
          gather_beta_link(
              beta_singles.sources[link],
              beta_singles.pairs[link],
              beta_singles.signs[link]);
        }
      }

#pragma omp for schedule(static)
      for (int beta = 0; beta < n_beta_; ++beta) {
        for (int block = 0; block < block_width; ++block) {
          for (int local_alpha = 0;
               local_alpha < alpha_count;
               ++local_alpha) {
            const int row = block * alpha_count + local_alpha;
            double value = 0.0;
            for (int source_thread = 0;
                 source_thread < n_threads;
                 ++source_thread) {
              value += partial_sigma[source_thread](row, beta);
            }
            sigma(
                alpha_begin + local_alpha,
                block * n_beta_ + beta) += value;
          }
        }
      }
    }
  }
  return sigma;
}

Eigen::MatrixXd DirectCiSigmaAction::apply_one_body_generator(
    const Eigen::Ref<const Eigen::MatrixXd>& coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& generator) const {
  if (coefficients.rows() != n_alpha_ || coefficients.cols() <= 0 ||
      coefficients.cols() % n_beta_ != 0 ||
      generator.rows() != n_orbitals_ ||
      generator.cols() != n_orbitals_ ||
      !coefficients.allFinite() || !generator.allFinite()) {
    throw std::invalid_argument(
        "direct-CI one-body generator has incompatible dimensions");
  }
  const int block_width = static_cast<int>(coefficients.cols()) / n_beta_;
  const SpinConnections& beta_graph = beta_connections();
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(
      n_alpha_, coefficients.cols());
  const int work_items = block_width * n_alpha_ * n_beta_;
  const int n_threads = std::max(
      1,
      std::min(effective_openmp_thread_count(), work_items));

#pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
  for (int work = 0; work < work_items; ++work) {
    const int alpha = work % n_alpha_;
    const int packed_column = work / n_alpha_;
    const int beta = packed_column % n_beta_;
    const int block = packed_column / n_beta_;
    const int column = packed_column;
    double value = 0.0;
    for (const int orbital : alpha_.occupied[alpha]) {
      value += generator(orbital, orbital) * coefficients(alpha, column);
    }
    for (const int orbital : beta_graph.occupied[beta]) {
      value += generator(orbital, orbital) * coefficients(alpha, column);
    }
    const DensityConnections& alpha_singles = alpha_.singles[alpha];
    for (std::size_t single = 0; single < alpha_singles.size(); ++single) {
      value += alpha_singles.signs[single] *
          generator(
              alpha_singles.created_orbitals[single],
              alpha_singles.annihilated_orbitals[single]) *
          coefficients(alpha_singles.sources[single], column);
    }
    const DensityConnections& beta_singles = beta_graph.singles[beta];
    for (std::size_t single = 0; single < beta_singles.size(); ++single) {
      value += beta_singles.signs[single] *
          generator(
              beta_singles.created_orbitals[single],
              beta_singles.annihilated_orbitals[single]) *
          coefficients(
              alpha,
              block * n_beta_ + beta_singles.sources[single]);
    }
    result(alpha, column) = value;
  }
  return result;
}

DirectCiIntegralAdjoint DirectCiSigmaAction::integral_adjoint(
    const Eigen::Ref<const Eigen::MatrixXd>& left,
    const Eigen::Ref<const Eigen::MatrixXd>& right) const {
  if (left.rows() != n_alpha_ || right.rows() != n_alpha_ ||
      left.cols() <= 0 || left.cols() != right.cols() ||
      left.cols() % n_beta_ != 0 || !left.allFinite() ||
      !right.allFinite()) {
    throw std::invalid_argument(
        "direct-CI adjoint coefficient blocks have incompatible dimensions");
  }

  const int n_pairs = static_cast<int>(pair_kernel_.rows());
  const int block_width = static_cast<int>(left.cols()) / n_beta_;
  constexpr int kAlphaTile = 160;
  const int n_alpha_tiles = (n_alpha_ + kAlphaTile - 1) / kAlphaTile;
  const int work_items = n_beta_ * n_alpha_tiles;
  const int n_threads = std::max(
      1,
      std::min(effective_openmp_thread_count(), work_items));
  std::vector<Eigen::MatrixXd> partial_effective_kernel(
      n_threads,
      Eigen::MatrixXd::Zero(n_pairs, n_pairs));

#pragma omp parallel if(n_threads > 1) num_threads(n_threads)
  {
    int thread = 0;
#ifdef _OPENMP
    thread = omp_get_thread_num();
#endif
    Eigen::MatrixXd left_intermediate(
        n_pairs, kAlphaTile * block_width);
    Eigen::MatrixXd right_intermediate(
        n_pairs, kAlphaTile * block_width);
    Eigen::MatrixXd& effective_kernel = partial_effective_kernel[thread];

#pragma omp for schedule(static)
    for (int work = 0; work < work_items; ++work) {
      const int beta = work / n_alpha_tiles;
      const int alpha_begin = (work % n_alpha_tiles) * kAlphaTile;
      const int alpha_count = std::min(kAlphaTile, n_alpha_ - alpha_begin);
      const int active_columns = alpha_count * block_width;
      auto active_left = left_intermediate.leftCols(active_columns);
      auto active_right = right_intermediate.leftCols(active_columns);
      scatter_one_body_intermediate(
          left, alpha_begin, alpha_count, beta, active_left);
      scatter_one_body_intermediate(
          right, alpha_begin, alpha_count, beta, active_right);
      effective_kernel.noalias() += active_left * active_right.transpose();
    }
  }

  Eigen::MatrixXd effective_kernel_adjoint = Eigen::MatrixXd::Zero(
      n_pairs, n_pairs);
  for (const Eigen::MatrixXd& partial : partial_effective_kernel) {
    effective_kernel_adjoint += partial;
  }

  Eigen::VectorXd absorbed_one_adjoint = Eigen::VectorXd::Zero(n_pairs);
  for (int orbital = 0; orbital < n_orbitals_; ++orbital) {
    const int diagonal_pair =
        TwoElectronIndexer::packed_pair_index(orbital, orbital);
    absorbed_one_adjoint.noalias() +=
        0.5 * (effective_kernel_adjoint.row(diagonal_pair).transpose() +
               effective_kernel_adjoint.col(diagonal_pair));
  }

  DirectCiIntegralAdjoint result;
  result.one_electron = Eigen::MatrixXd::Zero(
      n_orbitals_, n_orbitals_);
  result.pair_kernel =
      0.25 * (effective_kernel_adjoint +
              effective_kernel_adjoint.transpose());
  const int n_electrons =
      static_cast<int>(alpha_.occupied.front().size() +
                       beta_connections().occupied.front().size());
  for (int row = 0; row < n_orbitals_; ++row) {
    for (int column = 0; column <= row; ++column) {
      const int pair =
          TwoElectronIndexer::packed_pair_index(row, column);
      const double one_weight =
          absorbed_one_adjoint[pair] / static_cast<double>(n_electrons);
      add_symmetric_entry(
          &result.one_electron, row, column, one_weight);
      for (int orbital = 0; orbital < n_orbitals_; ++orbital) {
        add_symmetric_entry(
            &result.pair_kernel,
            TwoElectronIndexer::packed_pair_index(row, orbital),
            TwoElectronIndexer::packed_pair_index(orbital, column),
            -0.5 * one_weight);
      }
    }
  }
  return result;
}

Eigen::MatrixXd DirectCiSigmaAction::one_body_generator_adjoint(
    const Eigen::Ref<const Eigen::MatrixXd>& left,
    const Eigen::Ref<const Eigen::MatrixXd>& right) const {
  if (left.rows() != n_alpha_ || right.rows() != n_alpha_ ||
      left.cols() <= 0 || left.cols() != right.cols() ||
      left.cols() % n_beta_ != 0 || !left.allFinite() ||
      !right.allFinite()) {
    throw std::invalid_argument(
        "direct-CI generator-adjoint blocks have incompatible dimensions");
  }

  const int block_width = static_cast<int>(left.cols()) / n_beta_;
  const SpinConnections& beta_graph = beta_connections();
  const int work_items = block_width * n_alpha_ * n_beta_;
  const int n_threads = std::max(
      1,
      std::min(effective_openmp_thread_count(), work_items));
  std::vector<Eigen::MatrixXd> partials(
      n_threads,
      Eigen::MatrixXd::Zero(n_orbitals_, n_orbitals_));

#pragma omp parallel if(n_threads > 1) num_threads(n_threads)
  {
    int thread = 0;
#ifdef _OPENMP
    thread = omp_get_thread_num();
#endif
    Eigen::MatrixXd& density = partials[thread];

#pragma omp for schedule(static)
    for (int work = 0; work < work_items; ++work) {
      const int beta = work % n_beta_;
      const int alpha = (work / n_beta_) % n_alpha_;
      const int block = work / (n_alpha_ * n_beta_);
      const int column = block * n_beta_ + beta;
      const double left_value = left(alpha, column);
      if (left_value == 0.0) {
        continue;
      }
      const double diagonal_weight =
          left_value * right(alpha, column);
      for (const int orbital : alpha_.occupied[alpha]) {
        density(orbital, orbital) += diagonal_weight;
      }
      for (const int orbital : beta_graph.occupied[beta]) {
        density(orbital, orbital) += diagonal_weight;
      }
      const DensityConnections& alpha_singles = alpha_.singles[alpha];
      for (std::size_t link = 0; link < alpha_singles.size(); ++link) {
        density(
            alpha_singles.annihilated_orbitals[link],
            alpha_singles.created_orbitals[link]) +=
            left_value * right(alpha_singles.sources[link], column) *
            alpha_singles.signs[link];
      }
      const DensityConnections& beta_singles = beta_graph.singles[beta];
      for (std::size_t link = 0; link < beta_singles.size(); ++link) {
        density(
            beta_singles.annihilated_orbitals[link],
            beta_singles.created_orbitals[link]) +=
            left_value * right(
                alpha,
                block * n_beta_ + beta_singles.sources[link]) *
            beta_singles.signs[link];
      }
    }
  }

  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(
      n_orbitals_, n_orbitals_);
  for (const Eigen::MatrixXd& partial : partials) {
    result += partial;
  }
  // The determinant Hamiltonian stores a one-body element for the
  // source-orbital/target-orbital pair. An orbital generator instead carries
  // the target/source convention in `E(kappa)`, hence the transpose.
  return result.transpose();
}

std::size_t DirectCiSigmaAction::dynamic_bytes() const noexcept {
  const auto spin_bytes = [](const SpinConnections& spin) {
    std::size_t bytes =
        spin.singles.capacity() * sizeof(DensityConnections) +
        spin.occupied.capacity() * sizeof(std::vector<int>);
    for (const DensityConnections& connections : spin.singles) {
      bytes += connections.dynamic_bytes();
    }
    for (const auto& occupied : spin.occupied) {
      bytes += occupied.capacity() * sizeof(int);
    }
    return bytes;
  };
  std::size_t bytes = static_cast<std::size_t>(
      pair_kernel_.size() + hamiltonian_pair_kernel_.size()) * sizeof(double) +
      spin_bytes(alpha_);
  if (distinct_beta_.has_value()) {
    bytes += spin_bytes(*distinct_beta_);
  }
  return bytes;
}

}  // namespace xmvb::vb
