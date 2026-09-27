#include "vbscf/structures/orthogonal_ci/sigma.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>

#include "core/openmp.hpp"
#include "vbscf/determinants/pairs/evaluator.hpp"
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

std::vector<int> orbitals_not_in_mask(
    const std::vector<int>& first,
    std::uint64_t second_mask) {
  std::vector<int> result;
  for (const int orbital : first) {
    if ((second_mask & (std::uint64_t{1} << orbital)) == 0) {
      result.push_back(orbital);
    }
  }
  return result;
}

double deleted_minor_sign(
    const std::vector<int>& left,
    const std::vector<int>& right,
    const std::vector<int>& inserted,
    const std::vector<int>& removed) {
  int parity = 0;
  for (const int orbital : inserted) {
    parity += static_cast<int>(
        std::lower_bound(left.begin(), left.end(), orbital) - left.begin());
  }
  for (const int orbital : removed) {
    parity += static_cast<int>(
        std::lower_bound(right.begin(), right.end(), orbital) - right.begin());
  }
  return parity % 2 == 0 ? 1.0 : -1.0;
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

void add_symmetric_entry_atomic(
    Eigen::MatrixXd* matrix,
    int row,
    int column,
    double value) {
  if (row == column) {
#pragma omp atomic update
    (*matrix)(row, column) += value;
    return;
  }
#pragma omp atomic update
  (*matrix)(row, column) += 0.5 * value;
#pragma omp atomic update
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
  alpha_ = build_spin_connections(alpha_determinants, integrals);
  alpha_coulomb_diagonal_ = build_coulomb_diagonal(alpha_determinants);
  if (beta_determinants != alpha_determinants) {
    distinct_beta_.emplace(
        build_spin_connections(beta_determinants, integrals));
    distinct_beta_coulomb_diagonal_.emplace(
        build_coulomb_diagonal(beta_determinants));
  }
}

DirectCiSigmaAction::SpinConnections
DirectCiSigmaAction::build_spin_connections(
    const std::vector<std::vector<int>>& determinants,
    const OrthogonalActiveIntegrals& integrals) const {
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
  std::vector<double> identity_overlap(
      static_cast<std::size_t>(n_orbitals_) * n_orbitals_, 0.0);
  for (int orbital = 0; orbital < n_orbitals_; ++orbital) {
    identity_overlap[static_cast<std::size_t>(orbital) * n_orbitals_ + orbital] =
        1.0;
  }

  SpinConnections result;
  result.diagonal.resize(dimension);
  result.off_diagonal.resize(dimension);
  result.singles.resize(dimension);
  result.occupied = determinants;
  const DeterminantPairEvaluator evaluator;
  const int n_threads = std::max(
      1,
      std::min(effective_openmp_thread_count(), dimension));

#pragma omp parallel for schedule(dynamic, 1) if(n_threads > 1) num_threads(n_threads)
  for (int target = 0; target < dimension; ++target) {
    auto& connections = result.off_diagonal[target];
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
    const std::size_t n_doubles =
        target_occupied.size() * (target_occupied.size() - 1) / 2 *
        target_virtual.size() * (target_virtual.size() - 1) / 2;
    connections.reserve(n_singles + n_doubles);
    singles.reserve(n_singles);

    const auto add_source = [&](std::uint64_t source_mask) {
      const auto source_entry = index_by_mask.find(source_mask);
      if (source_entry == index_by_mask.end()) {
        throw std::logic_error(
            "direct-CI excitation left the fixed-spin determinant space");
      }
      const int source = source_entry->second;
      const auto pair = evaluator.evaluate_same_spin_pair(
          determinants[target],
          determinants[source],
          identity_overlap,
          integrals.one_electron,
          n_orbitals_,
          integrals.two_electron,
          false);
      HamiltonianConnection connection;
      connection.source = source;
      connection.value = pair.total_hamiltonian;
      const int changed_orbitals =
          __builtin_popcountll(masks[target] ^ source_mask);
      const std::vector<int> inserted = orbitals_not_in_mask(
          target_occupied, source_mask);
      const std::vector<int> removed = orbitals_not_in_mask(
          determinants[source], masks[target]);
      const double cofactor_sign = deleted_minor_sign(
          target_occupied,
          determinants[source],
          inserted,
          removed);
      if (changed_orbitals == 2) {
        const int removed = only_set_bit(source_mask & ~masks[target]);
        const int inserted = only_set_bit(masks[target] & ~source_mask);
        connection.density_pair = TwoElectronIndexer::packed_pair_index(
            removed, inserted);
        connection.density_sign = replacement_sign(
            source_mask, removed, inserted);
        if (connection.density_sign != cofactor_sign) {
          throw std::logic_error(
              "direct-CI single-excitation signs are inconsistent");
        }
        connection.one_electron_row = removed;
        connection.one_electron_column = inserted;
        connection.one_electron_sign = cofactor_sign;
        singles.append(
            source,
            connection.density_pair,
            connection.density_sign,
            inserted,
            removed);
      } else if (changed_orbitals == 4) {
        if (inserted.size() != 2 || removed.size() != 2) {
          throw std::logic_error(
              "direct-CI double-excitation topology is inconsistent");
        }
        connection.pair_terms[0] = PairKernelTerm{
            TwoElectronIndexer::packed_pair_index(removed[0], inserted[0]),
            TwoElectronIndexer::packed_pair_index(removed[1], inserted[1]),
            cofactor_sign};
        connection.pair_terms[1] = PairKernelTerm{
            TwoElectronIndexer::packed_pair_index(removed[0], inserted[1]),
            TwoElectronIndexer::packed_pair_index(removed[1], inserted[0]),
            -cofactor_sign};
        connection.n_pair_terms = 2;
      }
      connections.push_back(connection);
    };

    result.diagonal[target] = evaluator.evaluate_same_spin_pair(
        target_occupied,
        target_occupied,
        identity_overlap,
        integrals.one_electron,
        n_orbitals_,
        integrals.two_electron,
        false).total_hamiltonian;

    for (const int removed_from_target : target_occupied) {
      for (const int inserted_into_source : target_virtual) {
        const std::uint64_t source_mask =
            (masks[target] ^
             (std::uint64_t{1} << removed_from_target)) |
            (std::uint64_t{1} << inserted_into_source);
        add_source(source_mask);
      }
    }
    for (std::size_t first_occupied = 0;
         first_occupied < target_occupied.size();
         ++first_occupied) {
      for (std::size_t second_occupied = first_occupied + 1;
           second_occupied < target_occupied.size();
           ++second_occupied) {
        for (std::size_t first_virtual = 0;
             first_virtual < target_virtual.size();
             ++first_virtual) {
          for (std::size_t second_virtual = first_virtual + 1;
               second_virtual < target_virtual.size();
               ++second_virtual) {
            std::uint64_t source_mask = masks[target];
            source_mask ^= std::uint64_t{1}
                << target_occupied[first_occupied];
            source_mask ^= std::uint64_t{1}
                << target_occupied[second_occupied];
            source_mask |= std::uint64_t{1}
                << target_virtual[first_virtual];
            source_mask |= std::uint64_t{1}
                << target_virtual[second_virtual];
            add_source(source_mask);
          }
        }
      }
    }
  }
  return result;
}

Eigen::MatrixXd DirectCiSigmaAction::build_coulomb_diagonal(
    const std::vector<std::vector<int>>& determinants) const {
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(
      static_cast<int>(determinants.size()),
      pair_kernel_.rows());
  for (int determinant = 0;
       determinant < static_cast<int>(determinants.size());
       ++determinant) {
    for (const int occupied : determinants[determinant]) {
      const int diagonal_pair =
          TwoElectronIndexer::packed_pair_index(occupied, occupied);
      result.row(determinant) += pair_kernel_.col(diagonal_pair).transpose();
    }
  }
  return result;
}

const DirectCiSigmaAction::SpinConnections&
DirectCiSigmaAction::beta_connections() const noexcept {
  return distinct_beta_.has_value() ? *distinct_beta_ : alpha_;
}

const Eigen::MatrixXd&
DirectCiSigmaAction::beta_coulomb_diagonal() const noexcept {
  return distinct_beta_coulomb_diagonal_.has_value()
      ? *distinct_beta_coulomb_diagonal_
      : alpha_coulomb_diagonal_;
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
  const Eigen::MatrixXd& beta_coulomb = beta_coulomb_diagonal();
  Eigen::MatrixXd sigma = Eigen::MatrixXd::Zero(
      n_alpha_, coefficients.cols());
  const int work_items = block_width * n_alpha_ * n_beta_;
  const int n_threads = std::max(
      1,
      std::min(effective_openmp_thread_count(), work_items));

#pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
  for (int work = 0; work < work_items; ++work) {
    // Eigen stores the alpha index contiguously. Mapping the flattened work
    // index to that storage order gives every OpenMP chunk contiguous output
    // writes and keeps all fixed-beta coefficient reads in one column.
    const int alpha = work % n_alpha_;
    const int packed_column = work / n_alpha_;
    const int beta = packed_column % n_beta_;
    const int block = packed_column / n_beta_;
    const int column = packed_column;
    double value =
        (alpha_.diagonal[alpha] + beta_graph.diagonal[beta]) *
        coefficients(alpha, column);
    for (const int occupied : alpha_.occupied[alpha]) {
      const int pair = TwoElectronIndexer::packed_pair_index(
          occupied, occupied);
      value += beta_coulomb(beta, pair) *
          coefficients(alpha, column);
    }

    for (const HamiltonianConnection& connection :
         alpha_.off_diagonal[alpha]) {
      double matrix_element = connection.value;
      if (connection.density_pair >= 0) {
        matrix_element += connection.density_sign *
            beta_coulomb(beta, connection.density_pair);
      }
      value += matrix_element * coefficients(connection.source, column);
    }
    for (const HamiltonianConnection& connection :
         beta_graph.off_diagonal[beta]) {
      double matrix_element = connection.value;
      if (connection.density_pair >= 0) {
        matrix_element += connection.density_sign *
            alpha_coulomb_diagonal_(alpha, connection.density_pair);
      }
      value += matrix_element * coefficients(
          alpha,
          block * n_beta_ + connection.source);
    }
    const DensityConnections& alpha_singles = alpha_.singles[alpha];
    const DensityConnections& beta_singles = beta_graph.singles[beta];
    for (std::size_t alpha_single = 0;
         alpha_single < alpha_singles.size();
         ++alpha_single) {
      const int alpha_source = alpha_singles.sources[alpha_single];
      const int alpha_pair = alpha_singles.pairs[alpha_single];
      const double alpha_sign = alpha_singles.signs[alpha_single];
      for (std::size_t beta_single = 0;
           beta_single < beta_singles.size();
           ++beta_single) {
        value +=
            alpha_sign * beta_singles.signs[beta_single] *
            pair_kernel_(alpha_pair, beta_singles.pairs[beta_single]) *
            coefficients(
                alpha_source,
                block * n_beta_ + beta_singles.sources[beta_single]);
      }
    }
    sigma(alpha, column) = value;
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
  const SpinConnections& beta_graph = beta_connections();
  const int work_items = block_width * n_alpha_ * n_beta_;
  const int n_threads = std::max(
      1,
      std::min(effective_openmp_thread_count(), work_items));
  std::vector<Eigen::MatrixXd> partial_one_electron(
      n_threads,
      Eigen::MatrixXd::Zero(n_orbitals_, n_orbitals_));
  DirectCiIntegralAdjoint result;
  result.one_electron = Eigen::MatrixXd::Zero(
      n_orbitals_, n_orbitals_);
  result.pair_kernel = Eigen::MatrixXd::Zero(n_pairs, n_pairs);

#pragma omp parallel if(n_threads > 1) num_threads(n_threads)
  {
    int thread = 0;
#ifdef _OPENMP
    thread = omp_get_thread_num();
#endif
    auto& one = partial_one_electron[thread];
    auto& pair = result.pair_kernel;

    const auto add_spin_diagonal = [&](
        const std::vector<int>& occupied,
        double weight) {
      for (const int orbital : occupied) {
        one(orbital, orbital) += weight;
      }
      for (int first = 0;
           first < static_cast<int>(occupied.size());
           ++first) {
        const int first_orbital = occupied[first];
        for (int second = first + 1;
             second < static_cast<int>(occupied.size());
             ++second) {
          const int second_orbital = occupied[second];
          add_symmetric_entry_atomic(
              &pair,
              TwoElectronIndexer::packed_pair_index(
                  first_orbital, first_orbital),
              TwoElectronIndexer::packed_pair_index(
                  second_orbital, second_orbital),
              weight);
          const int exchange_pair =
              TwoElectronIndexer::packed_pair_index(
                  first_orbital, second_orbital);
#pragma omp atomic update
          pair(exchange_pair, exchange_pair) -= weight;
        }
      }
    };

    const auto add_spin_connection = [&](
        const SpinConnections& graph,
        const HamiltonianConnection& connection,
        double weight) {
      if (connection.one_electron_row >= 0) {
        const double signed_weight =
            weight * connection.one_electron_sign;
        add_symmetric_entry(
            &one,
            connection.one_electron_row,
            connection.one_electron_column,
            signed_weight);
        const int inserted = connection.one_electron_column;
        const int removed = connection.one_electron_row;
        for (const int common : graph.occupied[connection.source]) {
          if (common == removed) {
            continue;
          }
          add_symmetric_entry_atomic(
              &pair,
              TwoElectronIndexer::packed_pair_index(removed, inserted),
              TwoElectronIndexer::packed_pair_index(common, common),
              signed_weight);
          add_symmetric_entry_atomic(
              &pair,
              TwoElectronIndexer::packed_pair_index(removed, common),
              TwoElectronIndexer::packed_pair_index(inserted, common),
              -signed_weight);
        }
      }
      for (int term = 0; term < connection.n_pair_terms; ++term) {
        const PairKernelTerm& entry = connection.pair_terms[term];
        add_symmetric_entry_atomic(
            &pair,
            entry.first_pair,
            entry.second_pair,
            weight * entry.coefficient);
      }
    };

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
      add_spin_diagonal(alpha_.occupied[alpha], diagonal_weight);
      add_spin_diagonal(beta_graph.occupied[beta], diagonal_weight);
      for (const int alpha_orbital : alpha_.occupied[alpha]) {
        const int alpha_pair = TwoElectronIndexer::packed_pair_index(
            alpha_orbital, alpha_orbital);
        for (const int beta_orbital : beta_graph.occupied[beta]) {
          add_symmetric_entry_atomic(
              &pair,
              alpha_pair,
              TwoElectronIndexer::packed_pair_index(
                  beta_orbital, beta_orbital),
              diagonal_weight);
        }
      }

      for (const HamiltonianConnection& connection :
           alpha_.off_diagonal[alpha]) {
        const double weight =
            left_value * right(connection.source, column);
        add_spin_connection(alpha_, connection, weight);
        if (connection.density_pair >= 0) {
          for (const int beta_orbital : beta_graph.occupied[beta]) {
            add_symmetric_entry_atomic(
                &pair,
                connection.density_pair,
                TwoElectronIndexer::packed_pair_index(
                    beta_orbital, beta_orbital),
                weight * connection.density_sign);
          }
        }
      }
      for (const HamiltonianConnection& connection :
           beta_graph.off_diagonal[beta]) {
        const double weight = left_value * right(
            alpha,
            block * n_beta_ + connection.source);
        add_spin_connection(beta_graph, connection, weight);
        if (connection.density_pair >= 0) {
          for (const int alpha_orbital : alpha_.occupied[alpha]) {
            add_symmetric_entry_atomic(
                &pair,
                connection.density_pair,
                TwoElectronIndexer::packed_pair_index(
                    alpha_orbital, alpha_orbital),
                weight * connection.density_sign);
          }
        }
      }
      const DensityConnections& alpha_singles = alpha_.singles[alpha];
      const DensityConnections& beta_singles = beta_graph.singles[beta];
      for (std::size_t alpha_single = 0;
           alpha_single < alpha_singles.size();
           ++alpha_single) {
        for (std::size_t beta_single = 0;
             beta_single < beta_singles.size();
             ++beta_single) {
          const double weight = left_value * right(
              alpha_singles.sources[alpha_single],
              block * n_beta_ + beta_singles.sources[beta_single]);
          add_symmetric_entry_atomic(
              &pair,
              alpha_singles.pairs[alpha_single],
              beta_singles.pairs[beta_single],
              weight * alpha_singles.signs[alpha_single] *
                  beta_singles.signs[beta_single]);
        }
      }
    }
  }

  for (const auto& partial : partial_one_electron) {
    result.one_electron += partial;
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
      for (const HamiltonianConnection& connection :
           alpha_.off_diagonal[alpha]) {
        if (connection.one_electron_row >= 0) {
          density(
              connection.one_electron_row,
              connection.one_electron_column) +=
              left_value * right(connection.source, column) *
              connection.one_electron_sign;
        }
      }
      for (const HamiltonianConnection& connection :
           beta_graph.off_diagonal[beta]) {
        if (connection.one_electron_row >= 0) {
          density(
              connection.one_electron_row,
              connection.one_electron_column) +=
              left_value *
              right(alpha, block * n_beta_ + connection.source) *
              connection.one_electron_sign;
        }
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
        spin.diagonal.capacity() * sizeof(double) +
        spin.off_diagonal.capacity() *
            sizeof(std::vector<HamiltonianConnection>) +
        spin.singles.capacity() * sizeof(DensityConnections) +
        spin.occupied.capacity() * sizeof(std::vector<int>);
    for (const auto& connections : spin.off_diagonal) {
      bytes += connections.capacity() * sizeof(HamiltonianConnection);
    }
    for (const DensityConnections& connections : spin.singles) {
      bytes += connections.dynamic_bytes();
    }
    for (const auto& occupied : spin.occupied) {
      bytes += occupied.capacity() * sizeof(int);
    }
    return bytes;
  };
  std::size_t bytes = static_cast<std::size_t>(
      pair_kernel_.size() + alpha_coulomb_diagonal_.size()) * sizeof(double) +
      spin_bytes(alpha_);
  if (distinct_beta_.has_value()) {
    bytes += spin_bytes(*distinct_beta_) +
        static_cast<std::size_t>(distinct_beta_coulomb_diagonal_->size()) *
            sizeof(double);
  }
  return bytes;
}

}  // namespace xmvb::vb
