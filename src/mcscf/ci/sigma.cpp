#include "mcscf/ci/sigma.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include "parallel/openmp.hpp"
#include "core/timing.hpp"
#include "core/pair_index.hpp"
#include "mcscf/ci/planner.hpp"

namespace xmvb::mcscf {
namespace {

// Use the same contiguous alpha extent for standalone and fused sigma work.
constexpr int kSigmaAlphaTile = 160;

/** @brief Packed rows of a rectangular tile intersected with an exchange triangle.
 * Offsets are exact arithmetic prefix sums; no product-sized index array.
 */
struct AdjointRows {
  int alpha_begin, alpha_count, beta_begin, block_count, parity;

  int count(int beta) const {
    return parity == 0 ? alpha_count :
        std::min(alpha_count, beta_begin + beta - alpha_begin + (parity > 0));
  }

  int offset(int beta) const {
    if (parity == 0) return beta * alpha_count * block_count;
    const int first = beta_begin - alpha_begin + (parity > 0);
    const int ramp = std::clamp(alpha_count - first, 0, beta);
    return block_count * (ramp * first + ramp * (ramp - 1) / 2 +
                          (beta - ramp) * alpha_count);
  }
};

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

DirectCiSigmaAction::DirectCiSigmaAction(
    const std::vector<std::vector<int>>& alpha_determinants,
    const std::vector<std::vector<int>>& beta_determinants,
    const ActiveIntegrals& integrals)
    : n_orbitals_(static_cast<int>(integrals.one_electron.rows())),
      n_alpha_(static_cast<int>(alpha_determinants.size())),
      n_beta_(static_cast<int>(beta_determinants.size())),
      pair_kernel_(integrals.pair_kernel) {
  if (n_orbitals_ <= 0 || n_orbitals_ > 63 ||
      integrals.one_electron.cols() != n_orbitals_ ||
      !integrals.one_electron.allFinite() ||
      integrals.pair_kernel.rows() != n_orbitals_ * (n_orbitals_ + 1) / 2 ||
      integrals.pair_kernel.cols() != integrals.pair_kernel.rows() ||
      n_alpha_ <= 0 || n_beta_ <= 0) {
    throw std::invalid_argument("invalid orthogonal direct-CI sigma input");
  }
  if (!plan_orthogonal_direct_ci_action(
           alpha_determinants,
           beta_determinants,
           n_orbitals_,
           1).carrier_complete()) {
    throw std::invalid_argument(
        "orthogonal direct-CI sigma requires complete fixed-spin spaces");
  }
  const int n_electrons =
      static_cast<int>(alpha_determinants.front().size() +
                       beta_determinants.front().size());
  hamiltonian_pair_kernel_ =
      build_hamiltonian_pair_kernel(integrals, n_electrons);
  alpha_ = std::make_shared<const SpinConnections>(
      build_spin_connections(alpha_determinants));
  if (beta_determinants != alpha_determinants) {
    distinct_beta_ = std::make_shared<const SpinConnections>(
        build_spin_connections(beta_determinants));
  }
}

DirectCiSigmaAction::DirectCiSigmaAction(
    const DirectCiSigmaAction& reference,
    const ActiveIntegrals& integrals)
    : n_orbitals_(reference.n_orbitals_),
      n_alpha_(reference.n_alpha_),
      n_beta_(reference.n_beta_),
      pair_kernel_(integrals.pair_kernel),
      alpha_(reference.alpha_),
      distinct_beta_(reference.distinct_beta_) {
  if (integrals.one_electron.rows() != n_orbitals_ ||
      integrals.pair_kernel.rows() != reference.pair_kernel_.rows()) {
    throw std::invalid_argument("direct-CI shared links have incompatible orbitals");
  }
  const int n_electrons = static_cast<int>(
      alpha_->n_occupied() + beta_connections().n_occupied());
  hamiltonian_pair_kernel_ = build_hamiltonian_pair_kernel(integrals, n_electrons);
}

std::size_t DirectCiSigmaAction::spin_link_count() const noexcept {
  const std::size_t a = alpha_->n_occupied(), b = beta_connections().n_occupied();
  return static_cast<std::size_t>(n_alpha_) * a * (n_orbitals_-a+1) +
      static_cast<std::size_t>(n_beta_) * b * (n_orbitals_-b+1);
}

long double DirectCiSigmaAction::channel_search_work() const noexcept {
  const auto comparisons = [](std::size_t count) {
    int levels = 0;
    for (; count; count >>= 1) ++levels;
    return 2*levels;
  };
  const auto count = [&](std::size_t rows, std::size_t occupied) {
    return static_cast<long double>(rows) *
        (comparisons(occupied)+comparisons(occupied*(n_orbitals_-occupied)));
  };
  return count(n_alpha_,alpha_->n_occupied()) +
      count(n_beta_,beta_connections().n_occupied());
}

Eigen::MatrixXd DirectCiSigmaAction::diagonal(
    const Eigen::Ref<const Eigen::MatrixXd>& one_electron) const {
  if (one_electron.rows()!=n_orbitals_ || one_electron.cols()!=n_orbitals_ ||
      !one_electron.allFinite())
    throw std::invalid_argument("invalid direct-CI diagonal one-electron matrix");
  Eigen::MatrixXd coulomb(n_orbitals_,n_orbitals_);
  Eigen::MatrixXd exchange(n_orbitals_,n_orbitals_);
  for (int p=0;p<n_orbitals_;++p) for (int q=0;q<n_orbitals_;++q) {
    const int pp=xmvb::core::PairIndex::packed_pair_index(p,p);
    const int qq=xmvb::core::PairIndex::packed_pair_index(q,q);
    const int pq=xmvb::core::PairIndex::packed_pair_index(p,q);
    coulomb(p,q)=pair_kernel_(pp,qq);
    exchange(p,q)=pair_kernel_(pq,pq);
  }
  const auto spin_diagonal=[&](const SpinConnections& links,
      Eigen::MatrixXd* occupations) {
    const Eigen::Index count=links.n_strings();
    occupations->setZero(count,n_orbitals_);
    Eigen::VectorXd result=Eigen::VectorXd::Zero(count);
    for (Eigen::Index string=0;string<count;++string) {
      for (const int p:links.occupied(string)) {
        (*occupations)(string,p)=1.0;
        result[string]+=one_electron(p,p);
        for (const int q:links.occupied(string))
          result[string]+=0.5*(coulomb(p,q)-exchange(p,q));
      }
    }
    return result;
  };
  Eigen::MatrixXd alpha_occupations,beta_occupations;
  const Eigen::VectorXd alpha_energy=spin_diagonal(*alpha_,&alpha_occupations);
  const Eigen::VectorXd beta_energy=spin_diagonal(beta_connections(),&beta_occupations);
  Eigen::MatrixXd result=alpha_occupations*coulomb*beta_occupations.transpose();
  result.colwise()+=alpha_energy;
  result.rowwise()+=beta_energy.transpose();
  return result;
}

SpinConnections
DirectCiSigmaAction::build_spin_connections(
    const std::vector<std::vector<int>>& determinants) const {
  xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::CiConnections);
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
  const int electrons=determinants.front().size();
  const int n_singles=electrons*(n_orbitals_-electrons);
  result.strings=dimension;
  result.degree=n_singles;
  result.orbitals=n_orbitals_;
  result.orbital_offsets.resize(static_cast<std::size_t>(dimension)*(n_orbitals_+1));
  result.links.resize(static_cast<std::size_t>(n_singles)*dimension);
  result.occupations.resize(electrons,dimension);
  const int n_threads = coarse_openmp_thread_count(
      effective_openmp_thread_count(), static_cast<std::size_t>(dimension));

#pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
  for (int target = 0; target < dimension; ++target) {
    const auto& target_occupied = determinants[target];
    for (int i=0;i<electrons;++i) result.occupations(i,target)=target_occupied[i];
    std::array<int,63> target_virtual;
    int n_virtual=0,link=0;
    for (int orbital = 0; orbital < n_orbitals_; ++orbital) {
      if ((masks[target] & (std::uint64_t{1} << orbital)) == 0) {
        target_virtual[n_virtual++]=orbital;
      }
    }
    const auto add_source = [&](int removed,int inserted) {
      const auto source_mask=masks[target] ^ (std::uint64_t{1} << removed) ^
          (std::uint64_t{1} << inserted);
      const auto source_entry = index_by_mask.find(source_mask);
      if (source_entry == index_by_mask.end()) {
        throw std::logic_error(
            "direct-CI excitation left the fixed-spin determinant space");
      }
      const int source = source_entry->second;
      auto& entry=result.links[static_cast<std::size_t>(target)*n_singles+link];
      entry.source=source;
      entry.pair=core::PairIndex::packed_pair_index(removed,inserted);
      entry.sign=replacement_sign(source_mask,removed,inserted);
      entry.created=inserted;
      entry.annihilated=removed;
      ++link;
    };

    // Increasing maximum/minimum orbital indices are packed-channel order.
    // Opposite occupations generate every single replacement exactly once;
    // no sorting copy or channel-by-string offset table is needed.
    auto* offsets=result.orbital_offsets.data()+static_cast<std::size_t>(target)*(n_orbitals_+1);
    for (int high = 0; high < n_orbitals_; ++high) {
      offsets[high]=link;
      const bool occupied=masks[target] & (std::uint64_t{1} << high);
      const int* opposite=occupied ? target_virtual.data():target_occupied.data();
      const int count=occupied ? n_virtual:electrons;
      for (int i=0;i<count;++i) {
        const int low=opposite[i];
        if (low >= high) break;
        // These labels are already known: do not decode the same two bits again.
        add_source(occupied ? low:high,occupied ? high:low);
      }
    }
    offsets[n_orbitals_]=link;
  }
  result.masks=std::move(masks);
  if (profiling::phase_totals().enabled)
    std::fprintf(stderr,"XMVB_LINKS,strings=%d,occupied=%d,singles=%d,retained_bytes=%zu\n",
        dimension,electrons,n_singles,result.dynamic_bytes());
  return result;
}

Eigen::MatrixXd DirectCiSigmaAction::build_hamiltonian_pair_kernel(
    const ActiveIntegrals& integrals,
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
            xmvb::core::PairIndex::packed_pair_index(row, orbital),
            xmvb::core::PairIndex::packed_pair_index(orbital, column));
      }
      effective_one_electron(row, column) -= 0.5 * contraction;
    }
  }
  effective_one_electron /= static_cast<double>(n_electrons);

  Eigen::MatrixXd result = 0.5 * pair_kernel_;
  for (int orbital = 0; orbital < n_orbitals_; ++orbital) {
    const int diagonal_pair =
        xmvb::core::PairIndex::packed_pair_index(orbital, orbital);
    for (int row = 0; row < n_orbitals_; ++row) {
      for (int column = 0; column <= row; ++column) {
        const int pair =
            xmvb::core::PairIndex::packed_pair_index(row, column);
        const double value = 0.5 * effective_one_electron(row, column);
        result(diagonal_pair, pair) += value;
        result(pair, diagonal_pair) += value;
      }
    }
  }
  return 0.5 * (result + result.transpose());
}

const SpinConnections&
DirectCiSigmaAction::beta_connections() const noexcept {
  return distinct_beta_ ? *distinct_beta_ : *alpha_;
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
      intermediate.rows() != active_columns ||
      intermediate.cols() != hamiltonian_pair_kernel_.rows()) {
    throw std::invalid_argument(
        "direct-CI scatter intermediate has incompatible dimensions");
  }
  intermediate.setZero();

  const auto add_beta_link = [&] (
      int beta_source, int pair, double sign) {
    for (int block = 0; block < block_width; ++block) {
      intermediate.col(pair)
          .segment(block * alpha_count, alpha_count)
          .noalias() += sign * coefficients.block(
              alpha_begin,
              block * n_beta_ + beta_source,
              alpha_count,
              1);
    }
  };
  for (const int occupied : beta_graph.occupied(beta_target)) {
    add_beta_link(
        beta_target,
        xmvb::core::PairIndex::packed_pair_index(occupied, occupied),
        1.0);
  }
  const DensityConnections& beta_singles = beta_graph.singles(beta_target);
  for (std::size_t link = 0; link < beta_singles.size(); ++link) {
    add_beta_link(
        beta_singles[link].source,
        beta_singles[link].pair,
        beta_singles[link].sign);
  }

  for (int local_alpha = 0; local_alpha < alpha_count; ++local_alpha) {
    const int alpha_target = alpha_begin + local_alpha;
    const auto add_alpha_link = [&] (
        int alpha_source, int pair, double sign) {
      for (int block = 0; block < block_width; ++block) {
        intermediate(block * alpha_count + local_alpha, pair) +=
            sign * coefficients(
                alpha_source, block * n_beta_ + beta_target);
      }
    };
    for (const int occupied : alpha_->occupied(alpha_target)) {
      add_alpha_link(
          alpha_target,
          xmvb::core::PairIndex::packed_pair_index(occupied, occupied),
          1.0);
    }
    const DensityConnections& alpha_singles = alpha_->singles(alpha_target);
    for (std::size_t link = 0; link < alpha_singles.size(); ++link) {
      add_alpha_link(
          alpha_singles[link].source,
          alpha_singles[link].pair,
          alpha_singles[link].sign);
    }
  }
}

Eigen::MatrixXd DirectCiSigmaAction::apply(
    const Eigen::Ref<const Eigen::MatrixXd>& coefficients, int exchange_parity) const {
  profiling::ScopedPhase phase(profiling::Phase::CiSigma);
  Eigen::MatrixXd result;
  contract_channels(coefficients, coefficients, &result, exchange_parity, false);
  return result;
}

Eigen::MatrixXd DirectCiSigmaAction::apply_one_body_generator(
    const Eigen::Ref<const Eigen::MatrixXd>& coefficients,
    const Eigen::Ref<const Eigen::MatrixXd>& generator) const {
  Eigen::MatrixXd result;
  apply_one_body_generators({{coefficients, generator, &result}});
  return result;
}

void DirectCiSigmaAction::apply_one_body_generators(
    std::initializer_list<DirectCiGeneratorRequest> requests) const {
  xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::CiGenerator);
  if (requests.size() == 0) return;
  const Eigen::Index columns = requests.begin()->coefficients.cols();
  for (const auto& request : requests) {
    if (request.result == nullptr || request.coefficients.rows() != n_alpha_ ||
        columns <= 0 || columns % n_beta_ != 0 ||
        request.coefficients.cols() != columns ||
        request.generator.rows() != n_orbitals_ ||
        request.generator.cols() != n_orbitals_ ||
        !request.coefficients.allFinite() || !request.generator.allFinite()) {
      throw std::invalid_argument(
          "direct-CI generator requests have incompatible dimensions");
    }
    request.result->resize(n_alpha_, columns);
  }
  const SpinConnections& beta_graph = beta_connections();
  const Eigen::Index work_items = columns * static_cast<Eigen::Index>(n_alpha_);
  const int n_threads = coarse_openmp_thread_count(
      effective_openmp_thread_count(), static_cast<std::size_t>(work_items));

#pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
  for (Eigen::Index work = 0; work < work_items; ++work) {
    const int alpha = work % n_alpha_;
    const Eigen::Index packed_column = work / n_alpha_;
    const int beta = packed_column % n_beta_;
    const Eigen::Index block = packed_column / n_beta_;
    const Eigen::Index column = packed_column;
    for (const auto& request : requests) {
      const auto& coefficients = request.coefficients;
      const auto& generator = request.generator;
      double value = 0.0;
      for (const int orbital : alpha_->occupied(alpha)) {
        value += generator(orbital, orbital) * coefficients(alpha, column);
      }
      for (const int orbital : beta_graph.occupied(beta)) {
        value += generator(orbital, orbital) * coefficients(alpha, column);
      }
      const DensityConnections& alpha_singles = alpha_->singles(alpha);
      for (std::size_t single = 0; single < alpha_singles.size(); ++single) {
        value += alpha_singles[single].sign *
            generator(
                alpha_singles[single].created,
                alpha_singles[single].annihilated) *
            coefficients(alpha_singles[single].source, column);
      }
      const DensityConnections& beta_singles = beta_graph.singles(beta);
      for (std::size_t single = 0; single < beta_singles.size(); ++single) {
        value += beta_singles[single].sign *
            generator(
                beta_singles[single].created,
                beta_singles[single].annihilated) *
            coefficients(
                alpha,
                block * n_beta_ + beta_singles[single].source);
      }
      (*request.result)(alpha, column) = value;
    }
  }
}

DirectCiIntegralAdjoint DirectCiSigmaAction::integral_adjoint(
    const Eigen::Ref<const Eigen::MatrixXd>& left,
    const Eigen::Ref<const Eigen::MatrixXd>& right, int exchange_parity) const {
  xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::CiIntegralAdjoint);
  return contract_channels(left, right, nullptr, exchange_parity, true);
}

DirectCiResponse DirectCiSigmaAction::apply_with_adjoint(
    const Eigen::Ref<const Eigen::MatrixXd>& response,
    const Eigen::Ref<const Eigen::MatrixXd>& state, int exchange_parity) const {
  xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::CiResponse);
  DirectCiResponse result;
  result.integrals = contract_channels(response, state, &result.sigma, exchange_parity, true);
  return result;
}

void DirectCiSigmaAction::apply_sigma_tile(
    int alpha_begin, int alpha_count, int block_begin, int block_count,
    int beta_begin, int beta_count, int exchange_parity,
    const Eigen::Ref<const Eigen::MatrixXd>& scattered,
    Eigen::Ref<Eigen::MatrixXd> workspace,
    Eigen::Ref<Eigen::MatrixXd> sigma) const {
  const AdjointRows layout{alpha_begin, alpha_count, beta_begin, block_count, exchange_parity};
  const int n_pairs = static_cast<int>(hamiltonian_pair_kernel_.rows());
  // Alpha links only change the row: each beta target owns its output columns.
#pragma omp for schedule(static)
  for (int beta = 0; beta < beta_count; ++beta) {
    const int count = layout.count(beta);
    const int rows = count * block_count;
    // The previous adjoint no longer needs this storage. Change its layout to
    // contiguous per-beta matrices and consume each GEMM while it is local.
    Eigen::Map<Eigen::MatrixXd> transformed(
        workspace.data() + static_cast<Eigen::Index>(layout.offset(beta)) * n_pairs,
        rows, n_pairs);
    transformed.noalias() = scattered.middleRows(layout.offset(beta), rows) *
        hamiltonian_pair_kernel_.transpose();
    for (int alpha = 0; alpha < count; ++alpha) {
      const int target = alpha_begin + alpha;
      const auto add = [&](int source, int pair, double sign) {
        for (int block = 0; block < block_count; ++block) {
          sigma(source, (block_begin + block) * n_beta_ + beta_begin + beta) +=
              sign * transformed(block * count + alpha, pair);
        }
      };
      for (int occupied : alpha_->occupied(target)) {
        add(target, xmvb::core::PairIndex::packed_pair_index(occupied, occupied), 1.0);
      }
      const auto& links = alpha_->singles(target);
      for (std::size_t link = 0; link < links.size(); ++link) {
        add(links[link].source, links[link].pair, links[link].sign);
      }
    }
  }

  // Packed one-body generators are Hermitian: reverse links have the same
  // pair and sign. Pull contributions into the owned beta column rather than
  // scattering into thread-private copies of the output carrier.
  const SpinConnections& beta_graph = beta_connections();
#pragma omp for schedule(static)
  for (int beta = 0; beta < n_beta_; ++beta) {
    const auto add = [&](int source_beta, int pair, double sign) {
      if (source_beta < beta_begin || source_beta >= beta_begin + beta_count) return;
      const int local_beta = source_beta - beta_begin;
      const int count = layout.count(local_beta);
      const int rows = count * block_count;
      for (int block = 0; block < block_count; ++block) {
        const double* source = workspace.data() +
            static_cast<Eigen::Index>(layout.offset(local_beta)) * n_pairs + pair * rows +
            block * count;
        double* target = sigma.col((block_begin + block) * n_beta_ + beta).data() +
            alpha_begin;
#pragma omp simd
        for (int alpha = 0; alpha < count; ++alpha) {
          target[alpha] += sign * source[alpha];
        }
      }
    };
    for (int occupied : beta_graph.occupied(beta)) {
      add(beta, xmvb::core::PairIndex::packed_pair_index(occupied, occupied), 1.0);
    }
    const auto& links = beta_graph.singles(beta);
    for (std::size_t link = 0; link < links.size(); ++link) {
      add(links[link].source, links[link].pair, links[link].sign);
    }
  }
}

DirectCiIntegralAdjoint DirectCiSigmaAction::contract_channels(
    const Eigen::Ref<const Eigen::MatrixXd>& left,
    const Eigen::Ref<const Eigen::MatrixXd>& right,
    Eigen::MatrixXd* sigma, int exchange_parity, bool collect_integrals) const {
  if (exchange_parity < -1 || exchange_parity > 1 ||
      (exchange_parity != 0 && distinct_beta_)) {
    throw std::invalid_argument("direct-CI adjoint exchange sector requires identical spin lists");
  }
  if (left.rows() != n_alpha_ || right.rows() != n_alpha_ ||
      left.cols() <= 0 || left.cols() != right.cols() ||
      left.cols() % n_beta_ != 0 ||
      (collect_integrals && (!left.allFinite() || !right.allFinite()))) {
    throw std::invalid_argument(
        "direct-CI adjoint coefficient blocks have incompatible dimensions");
  }

  const int n_pairs = static_cast<int>(pair_kernel_.rows());
  const int block_width = static_cast<int>(left.cols()) / n_beta_;
  const bool same_input = left.data() == right.data() &&
      left.outerStride() == right.outerStride() && left.innerStride() == right.innerStride();
  // Flatten the contracted spin-string rows into two shared, bounded tiles.
  // Each output block has one owner: no thread-private pair-square gradient
  // and no serial reduction of those gradients are needed.
  constexpr std::size_t workspace_bytes = 64ULL * 1024ULL * 1024ULL;
  const std::size_t bytes_per_row = 2ULL * n_pairs * sizeof(double);
  if (bytes_per_row > workspace_bytes) {
    throw std::runtime_error("one direct-CI adjoint row exceeds workspace budget");
  }
  const int block_tile = static_cast<int>(std::min<std::size_t>(
      block_width, workspace_bytes / bytes_per_row));
  const std::size_t bytes_per_alpha =
      bytes_per_row * block_tile;
  // A fused tile also feeds sigma's link gather. Retain its established alpha
  // extent instead of using the entire alpha space as an adjoint-only tile.
  // The remaining budget becomes beta rows, not extra per-worker storage.
  const int alpha_limit = sigma == nullptr && exchange_parity == 0
      ? n_alpha_ : std::min(n_alpha_, kSigmaAlphaTile);
  const int alpha_tile = static_cast<int>(std::min<std::size_t>(
      alpha_limit, workspace_bytes / bytes_per_alpha));
  const int rows_per_beta = alpha_tile * block_tile;
  const int beta_tile = static_cast<int>(std::min<std::size_t>(
      n_beta_, workspace_bytes / (bytes_per_alpha * alpha_tile)));
  const int maximum_rows = rows_per_beta * beta_tile;
  const int n_threads = std::max(
      1,
      std::min(effective_openmp_thread_count(), std::max(beta_tile, n_pairs)));
  // Among exact factorizations of the worker count, the most nearly square
  // output grid minimizes repeated reads of these equally sized operands.
  int row_parts = static_cast<int>(std::sqrt(n_threads));
  while (n_threads % row_parts != 0) --row_parts;
  const int column_parts = n_threads / row_parts;
  Eigen::MatrixXd left_intermediate(maximum_rows, n_pairs);
  Eigen::MatrixXd right_intermediate(maximum_rows, n_pairs);
  Eigen::MatrixXd effective_kernel_adjoint = Eigen::MatrixXd::Zero(
      collect_integrals ? n_pairs : 0, collect_integrals ? n_pairs : 0);
  if (sigma != nullptr) sigma->setZero(n_alpha_, left.cols());

#pragma omp parallel if(n_threads > 1) num_threads(n_threads)
  {
    for (int block_begin = 0; block_begin < block_width; block_begin += block_tile) {
      const int block_count = std::min(block_tile, block_width - block_begin);
      const auto left_block = left.middleCols(block_begin * n_beta_, block_count * n_beta_);
      const auto right_block = right.middleCols(block_begin * n_beta_, block_count * n_beta_);
      for (int alpha_begin = 0; alpha_begin < n_alpha_; alpha_begin += alpha_tile) {
        const int alpha_count = std::min(alpha_tile, n_alpha_ - alpha_begin);
        const int beta_first = exchange_parity == 0 ? 0 : alpha_begin + (exchange_parity < 0);
        for (int beta_begin = beta_first; beta_begin < n_beta_; beta_begin += beta_tile) {
          const int beta_count = std::min(beta_tile, n_beta_ - beta_begin);
          const AdjointRows layout{alpha_begin, alpha_count, beta_begin, block_count, exchange_parity};
          const int used_rows = layout.offset(beta_count);
#pragma omp for schedule(static)
          for (int beta = 0; beta < beta_count; ++beta) {
            const int count = layout.count(beta);
            const int rows = count * block_count;
            auto active_left = left_intermediate.middleRows(layout.offset(beta), rows);
            auto active_right = right_intermediate.middleRows(layout.offset(beta), rows);
            scatter_one_body_intermediate(
                left_block, alpha_begin, count, beta_begin + beta, active_left);
            if (collect_integrals) {
              if (same_input) active_right = active_left;
              else scatter_one_body_intermediate(
                  right_block, alpha_begin, count, beta_begin + beta, active_right);
            }
            if (exchange_parity != 0) {
              // Q^T D Q weights one operand only; omitted rows are never scattered.
              active_left *= 2.0;
              const int diagonal = beta_begin + beta - alpha_begin;
              if (exchange_parity > 0 && diagonal < alpha_count) {
                for (int block = 0; block < block_count; ++block) {
                  active_left.row(block * count + diagonal) *= 0.5;
                }
              }
            }
          }
          // The scatter barrier publishes every shared row before GEMM; the
          // following barrier prevents overwriting the tile during contraction.
          if (collect_integrals) {
#pragma omp for schedule(static)
            for (int partition = 0; partition < n_threads; ++partition) {
              const int row_partition = partition / column_parts;
              const int column_partition = partition % column_parts;
              const int row_first = static_cast<int>(
                  static_cast<std::int64_t>(n_pairs) * row_partition / row_parts);
              const int row_end = static_cast<int>(
                  static_cast<std::int64_t>(n_pairs) * (row_partition + 1) / row_parts);
              const int first = static_cast<int>(
                  static_cast<std::int64_t>(n_pairs) * column_partition / column_parts);
              const int end = static_cast<int>(
                  static_cast<std::int64_t>(n_pairs) * (column_partition + 1) / column_parts);
              if (first == end || row_first == row_end) continue;
              effective_kernel_adjoint.block(
                  row_first, first, row_end - row_first, end - first).noalias() +=
                  left_intermediate.topRows(used_rows)
                      .middleCols(row_first, row_end - row_first).transpose() *
                  right_intermediate.topRows(used_rows).middleCols(first, end - first);
            }
          }
          if (sigma != nullptr) {
            // The adjoint has consumed the right tile. Reuse its storage for
            // K times the response scatter, without regenerating that scatter.
            apply_sigma_tile(alpha_begin, alpha_count, block_begin, block_count,
                beta_begin, beta_count, exchange_parity, left_intermediate, right_intermediate, *sigma);
          }
        }
      }
    }
  }

  if (sigma != nullptr && exchange_parity != 0) {
    // H commutes with exchange. P_eta E^T Q^T D K Q E X = H X.
    // Each unordered pair has one owner; no transpose-sized temporary.
#pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
    for (int column = 0; column < block_width * n_beta_; ++column) {
      const int beta = column % n_beta_;
      const int offset = column - beta;
      for (int alpha = 0; alpha < beta; ++alpha) {
        const double value = 0.5 * ((*sigma)(alpha, column) +
            exchange_parity * (*sigma)(beta, offset + alpha));
        (*sigma)(alpha, column) = value;
        (*sigma)(beta, offset + alpha) = exchange_parity * value;
      }
      if (exchange_parity < 0) (*sigma)(beta, column) = 0.0;
    }
  }

  if (!collect_integrals) return {};
  return pullback_channel_kernel(effective_kernel_adjoint);
}

DirectCiIntegralAdjoint DirectCiSigmaAction::pullback_channel_kernel(
    const Eigen::Ref<const Eigen::MatrixXd>& effective_kernel_adjoint) const {
  const int n_pairs = pair_kernel_.rows();
  if (effective_kernel_adjoint.rows() != n_pairs || effective_kernel_adjoint.cols() != n_pairs)
    throw std::invalid_argument("invalid Hamiltonian channel adjoint dimensions");
  Eigen::VectorXd absorbed_one_adjoint = Eigen::VectorXd::Zero(n_pairs);
  for (int orbital = 0; orbital < n_orbitals_; ++orbital) {
    const int diagonal_pair =
        xmvb::core::PairIndex::packed_pair_index(orbital, orbital);
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
      static_cast<int>(alpha_->n_occupied() +
                       beta_connections().n_occupied());
  for (int row = 0; row < n_orbitals_; ++row) {
    for (int column = 0; column <= row; ++column) {
      const int pair =
          xmvb::core::PairIndex::packed_pair_index(row, column);
      const double one_weight =
          absorbed_one_adjoint[pair] / static_cast<double>(n_electrons);
      add_symmetric_entry(
          &result.one_electron, row, column, one_weight);
      for (int orbital = 0; orbital < n_orbitals_; ++orbital) {
        add_symmetric_entry(
            &result.pair_kernel,
            xmvb::core::PairIndex::packed_pair_index(row, orbital),
            xmvb::core::PairIndex::packed_pair_index(orbital, column),
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
  return generator_adjoint_columns(0,n_beta_,left,{{right,1.0,false}});
}

Eigen::MatrixXd DirectCiSigmaAction::generator_adjoint_columns(
    int beta_begin,int beta_count,const Eigen::Ref<const Eigen::MatrixXd>& left,
    std::initializer_list<DirectCiGeneratorAdjointRequest> requests,bool upper_only) const {
  xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::CiGeneratorAdjoint);
  if (beta_begin<0 || beta_count<=0 || beta_begin+beta_count>n_beta_ ||
      left.rows()!=n_alpha_ || left.cols()<=0 || left.cols()%beta_count || requests.size()==0)
    throw std::invalid_argument("invalid generator-adjoint column panel");
  const int block_width=left.cols()/beta_count;
  for (const auto& request:requests)
    if (request.right.rows()!=n_alpha_ || request.right.cols()!=block_width*n_beta_)
      throw std::invalid_argument("invalid generator-adjoint right carrier");
  const SpinConnections& beta_graph = beta_connections();
  // Partition beta link events, not only beta columns: a narrow panel must
  // still distribute its long contiguous spectator dots across the team.
  std::vector<std::size_t> beta_offsets(beta_count+1,0);
  for (int b=0;b<beta_count;++b)
    beta_offsets[b+1]=beta_offsets[b]+1+beta_graph.singles(beta_begin+b).size();
  const std::size_t work_items=block_width*(n_alpha_+beta_offsets.back());
  bool in_team=false;
  int team=1;
#ifdef _OPENMP
  in_team=omp_in_parallel();
  if (in_team) team=omp_get_num_threads();
#endif
  const int n_threads = coarse_openmp_thread_count(
      in_team ? team : effective_openmp_thread_count(), static_cast<std::size_t>(work_items));
  std::vector<Eigen::MatrixXd> partials(
      n_threads,
      Eigen::MatrixXd::Zero(n_orbitals_, n_orbitals_));

  const auto accumulate=[&](int part) {
    Eigen::MatrixXd& density=partials[part];
    const auto consume=[&](bool beta,const SpinConnections& graph,int rows) {
      const std::size_t units=beta ? beta_offsets.back():static_cast<std::size_t>(rows);
      const std::size_t base=units/n_threads,extra=units%n_threads;
      const std::size_t first_unit=part*base+std::min<std::size_t>(part,extra);
      const std::size_t last_unit=first_unit+base+(static_cast<std::size_t>(part)<extra);
      if (first_unit==last_unit) return;
      const int first=beta ? std::upper_bound(beta_offsets.begin(),beta_offsets.end(),first_unit)-
          beta_offsets.begin()-1:static_cast<int>(first_unit);
      const int last=beta ? std::lower_bound(beta_offsets.begin(),beta_offsets.end(),last_unit)-
          beta_offsets.begin():static_cast<int>(last_unit);
      for (int block=0;block<block_width;++block)
        for (int local=first;local<last;++local) {
          const int target=local+(beta ? beta_begin:0);
          const auto& links=graph.singles(target);
          const std::size_t begin=beta ? std::max(first_unit,beta_offsets[local])-beta_offsets[local]:0;
          const std::size_t end=beta ? std::min(last_unit,beta_offsets[local+1])-beta_offsets[local]:
              links.size()+1;
          if (beta ? (left.col(block*beta_count+local).array()==0.0).all() :
              (left.row(target).segment(block*beta_count,beta_count).array()==0.0).all()) continue;
          const auto moment=[&](int source,const DirectCiGeneratorAdjointRequest& request) {
            // Contract the spectator spin before revisiting any excitation
            // metadata. Beta moments are contiguous column dots; alpha moments
            // borrow strided row views, without transposed carrier copies.
            if (beta) return left.col(block*beta_count+local).dot(
                request.right.col(block*n_beta_+source));
            return left.row(target).segment(block*beta_count,beta_count).dot(
                request.right.row(source).segment(block*n_beta_+beta_begin,beta_count));
          };
          if (begin==0 && graph.occupied(target).size()!=0) {
            double diagonal=0;
            for (const auto& request:requests) diagonal+=request.scale*moment(target,request);
            for (const int orbital:graph.occupied(target)) density(orbital,orbital)+=diagonal;
          }
          for (std::size_t link=begin>0 ? begin-1:0;link+1<end;++link)
            for (const auto& request:requests) {
              const int p=links[link].created,q=links[link].annihilated;
              const int row=request.transpose ? q:p,column=request.transpose ? p:q;
              if (upper_only && row>column) continue;
              density(row,column)+=request.scale*links[link].sign*moment(links[link].source,request);
            }
        }
    };
    consume(false,*alpha_,n_alpha_);
    consume(true,beta_graph,beta_count);
  };
  // Consume evicted output tiles in the existing sigma team. Stable logical
  // partitions own only n_orbital-square reductions, independent of scheduling.
  if (in_team) {
#pragma omp taskloop num_tasks(n_threads) shared(accumulate)
    for (int part=0;part<n_threads;++part) accumulate(part);
  } else {
#pragma omp parallel for schedule(static) if(n_threads>1) num_threads(n_threads)
    for (int part=0;part<n_threads;++part) accumulate(part);
  }

  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(
      n_orbitals_, n_orbitals_);
  for (const Eigen::MatrixXd& partial : partials) {
    result += partial;
  }
  return result;
}

Eigen::MatrixXd DirectCiSigmaAction::generator_adjoint_streamed(int first,
    const Eigen::Ref<const Eigen::MatrixXd>& left,
    std::initializer_list<DirectCiStreamedGeneratorRequest> requests,
    bool symmetric_sum, bool spin_exchange_reduced) const {
  const Eigen::Ref<const Eigen::MatrixXd> lefts[]{left};
  auto results=generator_adjoint_streamed(first,lefts,requests,symmetric_sum,spin_exchange_reduced);
  return std::move(results.front());
}

std::vector<Eigen::MatrixXd> DirectCiSigmaAction::generator_adjoint_streamed(int first,
    std::span<const Eigen::Ref<const Eigen::MatrixXd>> lefts,
    std::initializer_list<DirectCiStreamedGeneratorRequest> requests,
    bool symmetric_sum, bool spin_exchange_reduced) const {
  profiling::ScopedPhase phase(profiling::Phase::CiGeneratorAdjoint);
  if (lefts.empty()) throw std::invalid_argument("empty streamed generator panels");
  const Eigen::Index cols=lefts.front().cols();
  if (first<0 || cols<=0 || cols>n_beta_ || first>n_beta_-cols ||
      requests.size()==0 ||
      (spin_exchange_reduced && distinct_beta_))
    throw std::invalid_argument("invalid streamed generator panel");
  for (const auto& left:lefts)
    if (left.rows()!=n_alpha_ || left.cols()!=cols)
      throw std::invalid_argument("incompatible streamed generator panels");
  for (const auto& request:requests)
    if (!request.read || !request.column || !std::isfinite(request.scale))
      throw std::invalid_argument("invalid streamed generator request");

  // The caller already supplies a bounded panel. Reuse its width for both
  // generated rows and spectator batches instead of fragmenting it again.
  // All requests share the team; no worker owns a numerical panel copy.
  const int tile_width=static_cast<int>(cols);
  const int n_requests=static_cast<int>(requests.size());
  const bool retained=std::all_of(requests.begin(),requests.end(),
      [](const DirectCiStreamedGeneratorRequest& request) { return request.retained_columns; });
  const int spectator_width=spin_exchange_reduced ? 0:(retained ? n_beta_:tile_width);
  std::vector<int> rows(n_alpha_);
  std::iota(rows.begin(),rows.end(),0);
  std::size_t beta_work=0;
  const auto& graph=beta_connections();
  if (!spin_exchange_reduced)
    for (int b=0;b<cols;++b) beta_work+=graph.singles(first+b).size();
  const int workers=coarse_openmp_thread_count(
      cooperative_openmp_thread_count(),
      static_cast<std::size_t>(n_alpha_)*cols+beta_work);
  const int n_lefts=static_cast<int>(lefts.size());
  std::vector<Eigen::MatrixXd> partials(workers*n_lefts,
      Eigen::MatrixXd::Zero(n_orbitals_,n_orbitals_));
  std::vector<Eigen::MatrixXd> values(n_requests);
  struct Event { int source,local,p,q,slot; double sign; };
  std::vector<Event> events;
  struct OwnedColumn {
    std::vector<double> values;
    std::vector<int> rows;
    DirectCiColumnView view{nullptr,nullptr,0};
  };
  std::vector<std::vector<OwnedColumn>> gathered(n_requests,
      std::vector<OwnedColumn>(spectator_width));
  std::vector<int> sources(spectator_width);

  for (int offset=0;offset<cols;offset+=tile_width) {
    const int width=std::min<int>(tile_width,cols-offset);
    std::vector<int> betas(width);
    std::iota(betas.begin(),betas.end(),first+offset);
    for (int r=0;r<n_requests;++r) {
      values[r].resize(n_alpha_,width);
      requests.begin()[r].read(betas,rows,values[r]);
    }
    // One metadata traversal serves every generated right argument. Tasks
    // only read the completed tiles and own their small orbital reductions.
    const auto alpha_contract=[&](std::ptrdiff_t part) {
      for (int a=n_alpha_*part/workers;a<n_alpha_*(part+1)/workers;++a) {
        if (std::all_of(lefts.begin(),lefts.end(),[&](const auto& left) {
              return (left.row(a).segment(offset,width).array()==0.0).all();
            })) continue;
        for (int l=0;l<n_lefts;++l) {
          auto& density=partials[l*workers+part];
          double diagonal=0;
          for (int r=0;r<n_requests;++r)
            diagonal+=requests.begin()[r].scale*
                lefts[l].row(a).segment(offset,width).dot(values[r].row(a));
          for (const int p:alpha_->occupied(a)) density(p,p)+=diagonal;
        }
        for (const auto& link:alpha_->singles(a)) {
          for (int r=0;r<n_requests;++r) {
            const auto& request=requests.begin()[r];
            const int p=request.transpose ? link.annihilated:link.created;
            const int q=request.transpose ? link.created:link.annihilated;
            if (!symmetric_sum && p>q) continue;
            for (int l=0;l<n_lefts;++l)
              partials[l*workers+part](p,q)+=request.scale*link.sign*
                  lefts[l].row(a).segment(offset,width).dot(values[r].row(link.source));
          }
        }
      }
    };

    if (spin_exchange_reduced) {
      // For full equal-parity carriers P L=s L, P R=s R and
      // P E_alpha P=E_beta. Sum the alpha panels once, with multiplicity two.
      // Do not apply this identity to an isolated panel as a standalone action.
      cooperative_openmp_for(workers,workers,alpha_contract);
      continue;
    }
    events.clear();
    for (int local=0;local<width;++local) {
      const int b=first+offset+local;
      if (std::all_of(lefts.begin(),lefts.end(),[&](const auto& left) {
            return (left.col(offset+local).array()==0.0).all();
          })) continue;
      for (int l=0;l<n_lefts;++l) {
        double diagonal=0;
        for (int r=0;r<n_requests;++r)
          diagonal+=requests.begin()[r].scale*
              lefts[l].col(offset+local).dot(values[r].col(local));
        for (const int p:graph.occupied(b)) partials[l*workers](p,p)+=diagonal;
      }
      for (const auto& link:graph.singles(b)) {
        const bool used=symmetric_sum || std::any_of(requests.begin(),requests.end(),
            [&](const DirectCiStreamedGeneratorRequest& request) {
              return request.transpose ? link.annihilated<=link.created:
                  link.created<=link.annihilated;
            });
        if (used) events.push_back({link.source,offset+local,link.created,
                                   link.annihilated,0,link.sign});
      }
    }
    std::sort(events.begin(),events.end(),[](const Event& a,const Event& b) {
      return a.source<b.source;
    });
    if (events.empty()) cooperative_openmp_for(workers,workers,alpha_contract);

    // Accepted immutable panels need only borrowed view metadata. Transient
    // LRU views are copied before the next request can evict their owner.
    // Only the latter need numerical slabs, bounded by the left panel width.
    for (std::size_t begin=0;begin<events.size();) {
      std::size_t end=begin;
      int slots=0;
      while (end<events.size()) {
        if (end==begin || events[end].source!=events[end-1].source) {
          if (slots==spectator_width) break;
          sources[slots++]=events[end].source;
        }
        events[end].slot=slots-1;
        ++end;
      }
      for (int r=0;r<n_requests;++r)
        for (int slot=0;slot<slots;++slot) {
          const auto view=requests.begin()[r].column(sources[slot]);
          if (view.count<0 || view.count>n_alpha_ ||
              (view.count>0 && (!view.values || (!view.rows && view.count!=n_alpha_))))
            throw std::invalid_argument("invalid streamed generator spectator");
          auto& copy=gathered[r][slot];
          if (requests.begin()[r].retained_columns) { copy.view=view; continue; }
          copy.values.clear();
          copy.rows.clear();
          if (view.count>0) {
            copy.values.assign(view.values,view.values+view.count);
            if (view.rows) copy.rows.assign(view.rows,view.rows+view.count);
          }
          copy.view={copy.values.data(),view.rows ? copy.rows.data():nullptr,view.count};
        }
      cooperative_openmp_for(workers,workers,[&](std::ptrdiff_t part) {
        if (begin==0) alpha_contract(part);
        const std::size_t count=end-begin;
        const std::size_t base=count/workers,extra=count%workers;
        const std::size_t lo=begin+part*base+std::min<std::size_t>(part,extra);
        const std::size_t hi=lo+base+(static_cast<std::size_t>(part)<extra);
        for (std::size_t i=lo;i<hi;++i) {
          const auto& e=events[i];
          for (int r=0;r<n_requests;++r) {
            const auto& request=requests.begin()[r];
            const int p=request.transpose ? e.q:e.p;
            const int q=request.transpose ? e.p:e.q;
            if (!symmetric_sum && p>q) continue;
            const auto& right=gathered[r][e.slot].view;
            for (int l=0;l<n_lefts;++l) {
              const double* target=lefts[l].col(e.local).data();
              double moment=0;
              if (right.rows) {
#pragma omp simd reduction(+:moment)
                for (int a=0;a<right.count;++a)
                  moment+=target[right.rows[a]]*right.values[a];
              } else {
#pragma omp simd reduction(+:moment)
                for (int a=0;a<right.count;++a)
                  moment+=target[a]*right.values[a];
              }
              partials[l*workers+part](p,q)+=request.scale*e.sign*moment;
            }
          }
        }
      });
      begin=end;
    }
  }
  std::vector<Eigen::MatrixXd> results(n_lefts,Eigen::MatrixXd::Zero(n_orbitals_,n_orbitals_));
  for (int l=0;l<n_lefts;++l) {
    auto& result=results[l];
    for (int part=0;part<workers;++part) result+=partials[l*workers+part];
    if (spin_exchange_reduced) result*=2.0;
    if (symmetric_sum) {
      result=(result+result.transpose()).eval();
      result.triangularView<Eigen::StrictlyLower>().setZero();
    }
  }
  return results;
}

std::size_t DirectCiSigmaAction::dynamic_bytes() const noexcept {
  std::size_t bytes = static_cast<std::size_t>(
      pair_kernel_.size() + hamiltonian_pair_kernel_.size()) * sizeof(double) +
      alpha_->dynamic_bytes();
  if (distinct_beta_) {
    bytes += distinct_beta_->dynamic_bytes();
  }
  return bytes;
}

}  // namespace xmvb::mcscf
