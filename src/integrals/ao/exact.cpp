#include "integrals/ao/exact.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "parallel/openmp.hpp"
#include "core/timing.hpp"
#include "parallel/workers.hpp"

namespace xmvb::integrals {
namespace {

std::size_t matrix_size(const AoIntegralInput& ao) {
  const int n_bf = ao.n_basis_functions;
  if (n_bf <= 0) {
    throw std::invalid_argument("invalid AO-H1E integral input");
  }
  const std::size_t n_pairs =
      static_cast<std::size_t>(n_bf) * (n_bf + 1) / 2;
  const AoPairGraph& graph = ao.pair_graph;
  const bool valid_row_graph =
      graph.has_row_graph() &&
      graph.row_offsets.size() == n_pairs + 1 &&
      graph.columns.size() == graph.values.size() &&
      graph.integral_rows.size() == graph.integral_edges.size();
  const bool valid_compact_stream =
      graph.has_compact_integral_stream() &&
      graph.valid_unique_storage();
  if ((!valid_row_graph && !valid_compact_stream) ||
      graph.pair_first.size() != n_pairs ||
      graph.pair_second.size() != n_pairs) {
    throw std::invalid_argument("invalid AO-H1E integral input");
  }
  const std::size_t n = n_bf;
  return n * n;
}

/**
 * @brief Contracts contiguous unique-integral groups with a common left pair.
 *
 * CSR references are storage-ordered, so each group reuses its left AO indices
 * and Coulomb source. Compact streams need not be ordered: singleton groups
 * obey the same contraction. Only two scalar Coulomb sums are retained; ERIs
 * and density matrices stay shared, with no decoded-stencil cache.
 */
template <bool Forward, bool Transpose>
void contract_integrals(
    const AoIntegralInput& ao, std::size_t begin, std::size_t end,
    const double* source, const double* adjoint,
    double* forward, double* transpose) {
  const AoPairGraph& graph = ao.pair_graph;
  const int n_bf = ao.n_basis_functions;
  while (begin < end) {
    const int left = graph.integral(begin).left_pair;
    const int i = graph.pair_first[left];
    const int j = graph.pair_second[left];
    const int i_column = i * n_bf;
    const int j_column = j * n_bf;
    const int ij = j_column + i;
    const double left_scale = i == j ? 0.5 : 1.0;
    double source_ij = 0.0, adjoint_ij = 0.0;
    if constexpr (Forward) source_ij = source[ij];
    if constexpr (Transpose) adjoint_ij = adjoint[ij];
    double forward_coulomb = 0.0, transpose_coulomb = 0.0;
    do {
      const auto eri=graph.integral(begin);
      const int right=eri.right_pair;
      const int k = graph.pair_first[right];
      const int l = graph.pair_second[right];
      double value=eri.value*left_scale;
      if (k == l) value *= 0.5;
      if (left == right) value *= 0.5;
      const int k_column = k * n_bf;
      const int l_column = l * n_bf;
      const int kl = l_column + k;
      const int ik = k_column + i, jl = l_column + j;
      const int il = l_column + i, jk = k_column + j;
      const int lj = j_column + l, ki = i_column + k;
      const int kj = j_column + k, li = i_column + l;
      const double coulomb = 4.0 * value;
      if constexpr (Forward) {
        forward_coulomb += coulomb * source[kl];
        forward[kl] += coulomb * source_ij;
        forward[ik] -= value * source[lj];
        forward[jl] -= value * source[ki];
        forward[il] -= value * source[kj];
        forward[jk] -= value * source[li];
      }
      if constexpr (Transpose) {
        transpose_coulomb += coulomb * adjoint[kl];
        transpose[kl] += coulomb * adjoint_ij;
        transpose[lj] -= value * adjoint[ik];
        transpose[ki] -= value * adjoint[jl];
        transpose[kj] -= value * adjoint[il];
        transpose[li] -= value * adjoint[jk];
      }
      ++begin;
    } while (begin < end && graph.integral(begin).left_pair == left);
    // Exchange terms can also touch ij. Keeping the Coulomb sum separate is
    // algebraically exact even for coincident indices; only sum order changes.
    if constexpr (Forward) forward[ij] += forward_coulomb;
    if constexpr (Transpose) transpose[ij] += transpose_coulomb;
  }
}

/** @brief Thread-ordered reduction; output elements are owned by distinct workers. */
double sum_element(
    const std::vector<std::vector<double>>& partial,
    std::size_t index,
    int team_size) {
  double sum = 0.0;
  for (int thread = 0; thread < team_size; ++thread)
    sum += partial[thread][index];
  return sum;
}

/** @brief Interleaved symmetric channels share all integral and AO decoding. */
void contract_symmetric_integrals(
    const AoIntegralInput& ao, std::size_t begin, std::size_t end,
    const double* source, double* target) {
  using ConstChannels = Eigen::Map<const Eigen::Array2d>;
  using Channels = Eigen::Map<Eigen::Array2d>;
  const auto& graph = ao.pair_graph;
  const int n_bf = ao.n_basis_functions;
  while (begin < end) {
    const int left = graph.integral(begin).left_pair;
    const int i = graph.pair_first[left], j = graph.pair_second[left];
    const int ij = j * n_bf + i;
    const double scale = i == j ? 0.5 : 1.0;
    const Eigen::Array2d source_ij = ConstChannels(source + 2 * ij);
    Eigen::Array2d coulomb = Eigen::Array2d::Zero();
    do {
      const auto eri=graph.integral(begin);
      const int right=eri.right_pair;
      const int k = graph.pair_first[right], l = graph.pair_second[right];
      double value=eri.value*scale;
      if (k == l) value *= 0.5;
      if (left == right) value *= 0.5;
      const int kl = l * n_bf + k;
      const int ik = k * n_bf + i, jl = l * n_bf + j;
      const int il = l * n_bf + i, jk = k * n_bf + j;
      const int lj = j * n_bf + l, ki = i * n_bf + k;
      const int kj = j * n_bf + k, li = i * n_bf + l;
      coulomb += (4.0 * value) * ConstChannels(source + 2 * kl);
      Channels(target + 2 * kl) += (4.0 * value) * source_ij;
      Channels(target + 2 * ik) -= value * ConstChannels(source + 2 * lj);
      Channels(target + 2 * jl) -= value * ConstChannels(source + 2 * ki);
      Channels(target + 2 * il) -= value * ConstChannels(source + 2 * kj);
      Channels(target + 2 * jk) -= value * ConstChannels(source + 2 * li);
      ++begin;
    } while (begin < end && graph.integral(begin).left_pair == left);
    Channels(target + 2 * ij) += coulomb;
  }
}

/** @brief Common-left-pair stencil, vectorized over interleaved batch channels. */
template<bool Packed>
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
__attribute__((target_clones("avx512f", "avx2", "default")))
#endif
void contract_symmetric_block(
    const AoIntegralInput& ao, std::size_t begin, std::size_t end,
    const double* source, double* target, Eigen::Index width, double* coulomb) {
  const auto& graph = ao.pair_graph;
  const int n_bf = ao.n_basis_functions;
  const auto output = [&](int row,int column) -> Eigen::Index {
    if constexpr (Packed) {
      const Eigen::Index hi=std::max(row,column),lo=std::min(row,column);
      return (hi*(hi+1)/2+lo)*width;
    } else return (column*n_bf+row)*width;
  };
  while (begin < end) {
    const int left = graph.integral(begin).left_pair;
    const int i = graph.pair_first[left], j = graph.pair_second[left];
    const Eigen::Index ij = (j * n_bf + i) * width;
    const double scale = i == j ? 0.5 : 1.0;
    std::fill_n(coulomb, width, 0.0);
    do {
      const auto eri=graph.integral(begin);
      const int right=eri.right_pair;
      const int k = graph.pair_first[right], l = graph.pair_second[right];
      double value=eri.value*scale;
      if (k == l) value *= 0.5;
      if (left == right) value *= 0.5;
      const Eigen::Index kl = (l * n_bf + k) * width;
      const Eigen::Index lj = (j * n_bf + l) * width, ki = (i * n_bf + k) * width;
      const Eigen::Index kj = (j * n_bf + k) * width, li = (i * n_bf + l) * width;
      const Eigen::Index tkl=output(k,l),tik=output(i,k),tjl=output(j,l);
      const Eigen::Index til=output(i,l),tjk=output(j,k);
#pragma omp simd
      for (Eigen::Index channel = 0; channel < width; ++channel) {
        coulomb[channel] += (4.0 * value) * source[kl + channel];
        target[tkl + channel] += (4.0 * value) * source[ij + channel];
        target[tik + channel] -= value * source[lj + channel];
        target[tjl + channel] -= value * source[ki + channel];
        target[til + channel] -= value * source[kj + channel];
        target[tjk + channel] -= value * source[li + channel];
      }
      ++begin;
    } while (begin < end && graph.integral(begin).left_pair == left);
#pragma omp simd
    for (Eigen::Index channel = 0; channel < width; ++channel)
      target[output(i,j) + channel] += coulomb[channel];
  }
}

}  // namespace

void apply_symmetric_ao_h1e_fused(
    const double* source, const double* adjoint, const AoIntegralInput& ao,
    int n_threads, AoH1eSymmetricWorkspace* workspace,
    std::vector<double>* forward, std::vector<double>* transpose) {
  xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::AoH1eFused);
  if (source == nullptr || adjoint == nullptr || workspace == nullptr ||
      forward == nullptr || transpose == nullptr)
    throw std::invalid_argument("symmetric AO-H1E buffers must not be null");
  const std::size_t size = matrix_size(ao);
  const int n_bf = ao.n_basis_functions;
  const std::size_t n_integrals = ao.pair_graph.integral_count();
  n_threads = xmvb::bounded_openmp_thread_count(n_threads, n_integrals);
  const auto chunk = xmvb::openmp_chunk_size(n_integrals, n_threads);
  workspace->sources.resize(size, 2);
  workspace->partial.resize(n_threads);
  forward->resize(size);
  transpose->resize(size);
  // Packing is only AO-sized work. Complete it before launching the team:
  // avoid an extra work-sharing barrier for this small preparation loop.
  for (Eigen::Index index = 0; index < static_cast<Eigen::Index>(size); ++index) {
    workspace->sources(index, 0) = source[index];
    workspace->sources(index, 1) = adjoint[index];
  }
  ::xmvb::parallel::WorkerLoopTimes work(n_threads);
#pragma omp parallel num_threads(n_threads)
  {
    int thread = 0, team_size = 1;
#ifdef _OPENMP
    thread = omp_get_thread_num();
    team_size = omp_get_num_threads();
#endif
    auto& local = workspace->partial[thread];
    local.setZero(size, 2);
    work.start(thread);
#pragma omp for schedule(dynamic, 1) nowait
    for (std::size_t begin = 0; begin < n_integrals; begin += chunk)
      contract_symmetric_integrals(ao, begin, std::min(begin + chunk, n_integrals),
          workspace->sources.data(), local.data());
    work.stop(thread);
#pragma omp barrier
#pragma omp for schedule(static) nowait
    for (Eigen::Index pair = 0; pair < static_cast<Eigen::Index>(ao.pair_graph.pair_first.size()); ++pair) {
      const Eigen::Index i = ao.pair_graph.pair_first[pair];
      const Eigen::Index j = ao.pair_graph.pair_second[pair];
      const Eigen::Index index = j * n_bf + i, partner = i * n_bf + j;
      Eigen::Array2d sum = Eigen::Array2d::Zero();
      for (int worker = 0; worker < team_size; ++worker) {
        const auto& values = workspace->partial[worker];
        sum += values.row(index).array().transpose();
        sum += values.row(partner).array().transpose();
      }
      (*forward)[index] = sum[0];
      (*forward)[partner] = sum[0];
      (*transpose)[index] = 0.5 * sum[1];
      (*transpose)[partner] = 0.5 * sum[1];
    }
  }
  work.finish(xmvb::profiling::Phase::AoH1eFused);
}

std::vector<double> apply_ao_h1e(
    const double* source,
    const AoIntegralInput& ao,
    int n_threads) {
  xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::AoH1eForward);
  if (source == nullptr) {
    throw std::invalid_argument("AO-H1E source must not be null");
  }
  const std::size_t size = matrix_size(ao);
  const std::size_t n_integrals = ao.pair_graph.integral_count();
  n_threads = xmvb::bounded_openmp_thread_count(n_threads, n_integrals);
  const auto chunk = xmvb::openmp_chunk_size(n_integrals, n_threads);
  std::vector<std::vector<double>> partial(n_threads);
  std::vector<double> result(size);
  ::xmvb::parallel::WorkerLoopTimes work(n_threads);
#pragma omp parallel num_threads(n_threads)
  {
    int thread = 0;
    int team_size = 1;
#ifdef _OPENMP
    thread = omp_get_thread_num();
    team_size = omp_get_num_threads();
#endif
    auto& local = partial[thread];
    local.assign(size, 0.0);
    work.start(thread);
#pragma omp for schedule(dynamic, 1) nowait
    for (std::size_t begin = 0; begin < n_integrals; begin += chunk) {
      contract_integrals<true, false>(
          ao, begin, std::min(begin + chunk, n_integrals),
          source, nullptr, local.data(), nullptr);
    }
    work.stop(thread);
#pragma omp barrier
#pragma omp for schedule(static) nowait
    for (std::size_t index = 0; index < size; ++index)
      result[index] = sum_element(partial, index, team_size);
  }
  work.finish(xmvb::profiling::Phase::AoH1eForward);
  return result;
}

std::vector<double> apply_ao_h1e_transpose(
    const double* adjoint,
    const AoIntegralInput& ao,
    int n_threads) {
  xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::AoH1eTranspose);
  if (adjoint == nullptr) {
    throw std::invalid_argument("AO-H1E adjoint must not be null");
  }
  const std::size_t size = matrix_size(ao);
  const std::size_t n_integrals = ao.pair_graph.integral_count();
  n_threads = xmvb::bounded_openmp_thread_count(n_threads, n_integrals);
  const auto chunk = xmvb::openmp_chunk_size(n_integrals, n_threads);
  std::vector<std::vector<double>> partial(n_threads);
  std::vector<double> result(size);
  ::xmvb::parallel::WorkerLoopTimes work(n_threads);
#pragma omp parallel num_threads(n_threads)
  {
    int thread = 0;
    int team_size = 1;
#ifdef _OPENMP
    thread = omp_get_thread_num();
    team_size = omp_get_num_threads();
#endif
    auto& local = partial[thread];
    local.assign(size, 0.0);
    work.start(thread);
#pragma omp for schedule(dynamic, 1) nowait
    for (std::size_t begin = 0; begin < n_integrals; begin += chunk) {
      contract_integrals<false, true>(
          ao, begin, std::min(begin + chunk, n_integrals),
          nullptr, adjoint, nullptr, local.data());
    }
    work.stop(thread);
#pragma omp barrier
#pragma omp for schedule(static) nowait
    for (std::size_t index = 0; index < size; ++index)
      result[index] = sum_element(partial, index, team_size);
  }
  work.finish(xmvb::profiling::Phase::AoH1eTranspose);
  return result;
}

/** @brief One contraction/reduction implementation for distinct storage roles. */
template<bool Packed>
void apply_symmetric_channels(const AoIntegralInput& ao,int n_threads,
    AoH1eBatchWorkspace* workspace,Eigen::MatrixXd* response) {
  xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::AoH1eFused);
  const std::size_t size=matrix_size(ao);
  if (!workspace || !response || workspace->sources.rows()!=static_cast<Eigen::Index>(size))
    throw std::invalid_argument("invalid interleaved AO response channels");
  const Eigen::Index width = workspace->sources.cols();
  response->resize(size, width);
  if (width == 0) return;
  const std::size_t n_integrals = ao.pair_graph.integral_count();
  n_threads = bounded_openmp_thread_count(n_threads, n_integrals);
  const auto chunk = openmp_chunk_size(n_integrals, n_threads);
  workspace->partial.resize(n_threads);
  workspace->coulomb.resize(n_threads);
  const int n_bf = ao.n_basis_functions;
  const Eigen::Index n_pairs=static_cast<Eigen::Index>(n_bf)*(n_bf+1)/2;
  ::xmvb::parallel::WorkerLoopTimes work(n_threads);
#pragma omp parallel num_threads(n_threads)
  {
    int thread = 0, team_size = 1;
#ifdef _OPENMP
    thread = omp_get_thread_num();
    team_size = omp_get_num_threads();
#endif
    auto& local = workspace->partial[thread];
    local.setZero(Packed ? n_pairs : static_cast<Eigen::Index>(size), width);
    auto& coulomb = workspace->coulomb[thread];
    coulomb.resize(width);
    work.start(thread);
#pragma omp for schedule(dynamic, 1) nowait
    for (std::size_t begin = 0; begin < n_integrals; begin += chunk) {
      if constexpr (Packed) {
        contract_symmetric_block<true>(ao, begin, std::min(begin + chunk, n_integrals),
            workspace->sources.data(), local.data(), width, coulomb.data());
      } else {
        if (width==1)
          contract_integrals<true,false>(ao,begin,std::min(begin+chunk,n_integrals),
              workspace->sources.data(),nullptr,local.data(),nullptr);
        else
          contract_symmetric_block<false>(ao,begin,std::min(begin+chunk,n_integrals),
              workspace->sources.data(),local.data(),width,coulomb.data());
      }
    }
    work.stop(thread);
#pragma omp barrier
#pragma omp for schedule(static) nowait
    for (int column = 0; column < n_bf; ++column) {
      for (int row = column; row < n_bf; ++row) {
        const Eigen::Index index = column * n_bf + row;
        const Eigen::Index partner = row * n_bf + column;
        const Eigen::Index pair=static_cast<Eigen::Index>(row)*(row+1)/2+column;
        for (Eigen::Index channel = 0; channel < width; ++channel) {
          double sum = 0.0;
          for (int worker = 0; worker < team_size; ++worker) {
            if constexpr (Packed) sum+=workspace->partial[worker](pair,channel);
            else sum+=workspace->partial[worker](index,channel)+workspace->partial[worker](partner,channel);
          }
          if constexpr (Packed) if (row==column) sum*=2.0;
          (*response)(index, channel) = (*response)(partner, channel) = sum;
        }
      }
    }
  }
  work.finish(xmvb::profiling::Phase::AoH1eFused);
}

void apply_symmetric_ao_h1e_batch(const Eigen::Ref<const Eigen::MatrixXd>& sources,
    const AoIntegralInput& ao,int n_threads,AoH1eBatchWorkspace* workspace,
    Eigen::MatrixXd* response) {
  const std::size_t size=matrix_size(ao);
  if (!workspace || !response || sources.rows()!=static_cast<Eigen::Index>(size))
    throw std::invalid_argument("invalid symmetric AO response batch");
  workspace->sources=sources;
  apply_symmetric_channels<false>(ao,n_threads,workspace,response);
}

void apply_symmetric_ao_h1e_channels(const AoIntegralInput& ao,int n_threads,
    AoH1eBatchWorkspace* workspace,Eigen::MatrixXd* response) {
  apply_symmetric_channels<true>(ao,n_threads,workspace,response);
}

void apply_symmetric_ao_h1e_fused_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& sources,
    const Eigen::Ref<const Eigen::MatrixXd>& adjoints,
    const AoIntegralInput& ao,
    int n_threads,
    AoH1eBatchWorkspace* workspace,
    Eigen::MatrixXd* forward,
    Eigen::MatrixXd* transpose) {
  xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::AoH1eFused);
  if (workspace == nullptr || forward == nullptr || transpose == nullptr) {
    throw std::invalid_argument("AO-H1E batch outputs must not be null");
  }
  const std::size_t size = matrix_size(ao);
  if (sources.rows() != static_cast<Eigen::Index>(size) ||
      adjoints.rows() != static_cast<Eigen::Index>(size) ||
      sources.cols() != adjoints.cols()) {
    throw std::invalid_argument("AO-H1E batch input shape mismatch");
  }
  const std::size_t n_integrals = ao.pair_graph.integral_count();
  n_threads = xmvb::bounded_openmp_thread_count(n_threads, n_integrals);
  const auto chunk = xmvb::openmp_chunk_size(n_integrals, n_threads);
  const Eigen::Index n_directions = sources.cols();
  forward->resize(size, n_directions);
  transpose->resize(size, n_directions);
  if (n_directions == 0) return;
  const Eigen::Index width = 2 * n_directions;
  workspace->sources.resize(size, width);
  workspace->partial.resize(n_threads);
  workspace->coulomb.resize(n_threads);
  for (Eigen::Index index = 0; index < static_cast<Eigen::Index>(size); ++index) {
    for (Eigen::Index direction = 0; direction < n_directions; ++direction) {
      workspace->sources(index, 2 * direction) = sources(index, direction);
      workspace->sources(index, 2 * direction + 1) = adjoints(index, direction);
    }
  }
  const int n_bf = ao.n_basis_functions;
  ::xmvb::parallel::WorkerLoopTimes work(n_threads);
#pragma omp parallel num_threads(n_threads)
  {
    int thread = 0;
    int team_size = 1;
#ifdef _OPENMP
    thread = omp_get_thread_num();
    team_size = omp_get_num_threads();
#endif
    auto& local = workspace->partial[thread];
    local.setZero(size, width);
    auto& coulomb = workspace->coulomb[thread];
    coulomb.resize(width);
    work.start(thread);
#pragma omp for schedule(dynamic, 1) nowait
    for (std::size_t begin = 0; begin < n_integrals; begin += chunk)
      contract_symmetric_block<false>(ao, begin, std::min(begin + chunk, n_integrals),
          workspace->sources.data(), local.data(), width, coulomb.data());
    work.stop(thread);
#pragma omp barrier
#pragma omp for schedule(static) nowait
    for (int column = 0; column < n_bf; ++column) {
      for (int row = column; row < n_bf; ++row) {
        const Eigen::Index index = column * n_bf + row;
        const Eigen::Index partner = row * n_bf + column;
        for (Eigen::Index direction = 0; direction < n_directions; ++direction) {
          double sum_forward = 0.0, sum_transpose = 0.0;
          for (int worker = 0; worker < team_size; ++worker) {
            const auto& partial = workspace->partial[worker];
            sum_forward += partial(index, 2 * direction) + partial(partner, 2 * direction);
            sum_transpose += partial(index, 2 * direction + 1) + partial(partner, 2 * direction + 1);
          }
          (*forward)(index, direction) = (*forward)(partner, direction) = sum_forward;
          (*transpose)(index, direction) = (*transpose)(partner, direction) = 0.5 * sum_transpose;
        }
      }
    }
  }
  work.finish(xmvb::profiling::Phase::AoH1eFused);
}

}  // namespace xmvb::integrals
