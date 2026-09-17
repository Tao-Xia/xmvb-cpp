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

}  // namespace

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
      if (changed_orbitals == 2) {
        const int removed = only_set_bit(source_mask & ~masks[target]);
        const int inserted = only_set_bit(masks[target] & ~source_mask);
        connection.density_pair = TwoElectronIndexer::packed_pair_index(
            removed, inserted);
        connection.density_sign = replacement_sign(
            source_mask, removed, inserted);
        singles.push_back(DensityConnection{
            source,
            connection.density_pair,
            connection.density_sign});
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
    const int beta = work % n_beta_;
    const int alpha = (work / n_beta_) % n_alpha_;
    const int block = work / (n_alpha_ * n_beta_);
    const int column = block * n_beta_ + beta;
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
    for (const DensityConnection& alpha_connection :
         alpha_.singles[alpha]) {
      for (const DensityConnection& beta_connection :
           beta_graph.singles[beta]) {
        value +=
            alpha_connection.sign * beta_connection.sign *
            pair_kernel_(alpha_connection.pair, beta_connection.pair) *
            coefficients(
                alpha_connection.source,
                block * n_beta_ + beta_connection.source);
      }
    }
    sigma(alpha, column) = value;
  }
  return sigma;
}

std::size_t DirectCiSigmaAction::dynamic_bytes() const noexcept {
  const auto spin_bytes = [](const SpinConnections& spin) {
    std::size_t bytes =
        spin.diagonal.capacity() * sizeof(double) +
        spin.off_diagonal.capacity() *
            sizeof(std::vector<HamiltonianConnection>) +
        spin.singles.capacity() * sizeof(std::vector<DensityConnection>) +
        spin.occupied.capacity() * sizeof(std::vector<int>);
    for (const auto& connections : spin.off_diagonal) {
      bytes += connections.capacity() * sizeof(HamiltonianConnection);
    }
    for (const auto& connections : spin.singles) {
      bytes += connections.capacity() * sizeof(DensityConnection);
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
