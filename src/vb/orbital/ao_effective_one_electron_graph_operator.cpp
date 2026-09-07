#include "vb/orbital/ao_effective_one_electron_graph_operator.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <mutex>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xmvb::vb {

namespace {

constexpr double kIntegralSymmetryMultipliers[] = {
    1.0,
    0.5,
    0.25,
    0.125,
};

constexpr std::size_t kAoH1eTransposeStripeSize = 8192;
constexpr int kAoH1eTransposeStripeCacheSlots = 4;

std::size_t ao_matrix_size(int n_basis_functions) {
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("n_basis_functions must be positive");
  }
  return n_basis_functions * n_basis_functions;
}

void validate_graph_storage_shapes(
    const AoIntegralInput& ao_integral_input,
    std::size_t matrix_size) {
  const auto& row_offsets =
      ao_integral_input.ao_effective_one_electron_graph_row_offsets;
  const auto& source_indices =
      ao_integral_input.ao_effective_one_electron_graph_source_indices;
  const auto& signed_weights =
      ao_integral_input.ao_effective_one_electron_graph_signed_weights;
  if (row_offsets.size() != matrix_size + 1) {
    throw std::invalid_argument(
        "AO-H1E graph row-offset size does not match AO matrix size");
  }
  if (row_offsets.empty()) {
    throw std::invalid_argument("AO-H1E graph row offsets must not be empty");
  }
  if (row_offsets.front() != 0) {
    throw std::invalid_argument("AO-H1E graph row offsets must start at zero");
  }
  if (row_offsets.back() < 0) {
    throw std::invalid_argument("AO-H1E graph edge count must be nonnegative");
  }
  const std::size_t edge_count = row_offsets.back();
  if (source_indices.size() != edge_count ||
      signed_weights.size() != edge_count) {
    throw std::invalid_argument(
        "AO-H1E graph edge arrays do not match the stored edge count");
  }
}

void validate_transpose_graph_storage_shapes(
    const AoIntegralInput& ao_integral_input,
    std::size_t matrix_size) {
  const auto& source_offsets =
      ao_integral_input.ao_effective_one_electron_graph_transpose_source_offsets;
  const auto& row_indices =
      ao_integral_input.ao_effective_one_electron_graph_transpose_row_indices;
  const auto& signed_weights =
      ao_integral_input.ao_effective_one_electron_graph_transpose_signed_weights;
  if (source_offsets.empty() && row_indices.empty() && signed_weights.empty()) {
    return;
  }
  if (source_offsets.size() != matrix_size + 1) {
    throw std::invalid_argument(
        "AO-H1E transpose graph source-offset size does not match AO matrix size");
  }
  if (source_offsets.front() != 0) {
    throw std::invalid_argument(
        "AO-H1E transpose graph source offsets must start at zero");
  }
  if (source_offsets.back() < 0) {
    throw std::invalid_argument(
        "AO-H1E transpose graph edge count must be nonnegative");
  }
  const std::size_t edge_count = source_offsets.back();
  if (row_indices.size() != edge_count ||
      signed_weights.size() != edge_count) {
    throw std::invalid_argument(
        "AO-H1E transpose graph edge arrays do not match the stored edge count");
  }
}

bool transpose_graph_available(const AoIntegralInput& ao_integral_input) noexcept {
  return !ao_integral_input
              .ao_effective_one_electron_graph_transpose_source_offsets.empty();
}

int sanitize_graph_thread_count(int n_threads) {
  if (n_threads <= 0) {
    throw std::invalid_argument("AO-H1E graph thread count must be positive");
  }
  return n_threads;
}

/**
 * @brief Bounded-memory striped reducer for AO-H1E transpose accumulations.
 *
 * The multithreaded exact_ctx fused graph apply partitions destination rows
 * across threads, so `K * x` stays row-local. The transpose pullback
 * `K^T * lambda` still collides on source AO entries, however. This reducer
 * keeps only a few dense source stripes per worker and flushes them under
 * stripe-local locks, which avoids one full AO matrix per thread while
 * preserving the single-row-sweep algorithm.
 */
class AoH1eStripedTransposeReducer {
public:
  class ThreadLocalAccumulator {
  public:
    explicit ThreadLocalAccumulator(AoH1eStripedTransposeReducer* reducer)
        : reducer_(reducer),
          stripe_indices_(
              reducer->cache_slots_,
              -1),
          stripe_buffers_(
              reducer->cache_slots_,
              std::vector<double>(reducer->stripe_size_, 0.0)),
          touched_offsets_(reducer->cache_slots_) {
      if (reducer_ == nullptr) {
        throw std::invalid_argument("AO-H1E striped reducer must not be null");
      }
    }

    ~ThreadLocalAccumulator() {
      flush_all();
    }

    void add(std::size_t source_index, double value) {
      if (value == 0.0) {
        return;
      }
      if (source_index >= reducer_->target_->size()) {
        throw std::out_of_range("AO-H1E transpose source index is out of range");
      }
      const std::ptrdiff_t stripe_index = static_cast<std::ptrdiff_t>(
          source_index / reducer_->stripe_size_);
      const std::size_t stripe_offset = source_index % reducer_->stripe_size_;
      const int slot = acquire_slot(stripe_index);
      auto& stripe_buffer = stripe_buffers_[slot];
      double& buffered_value = stripe_buffer[stripe_offset];
      if (buffered_value == 0.0) {
        touched_offsets_[slot].push_back(stripe_offset);
      }
      buffered_value += value;
    }

    void flush_all() noexcept {
      for (int slot = 0; slot < reducer_->cache_slots_; ++slot) {
        flush_slot(slot);
      }
    }

  private:
    int acquire_slot(std::ptrdiff_t stripe_index) {
      for (int slot = 0; slot < reducer_->cache_slots_; ++slot) {
        if (stripe_indices_[slot] == stripe_index) {
          return slot;
        }
      }
      for (int slot = 0; slot < reducer_->cache_slots_; ++slot) {
        if (stripe_indices_[slot] < 0) {
          stripe_indices_[slot] = stripe_index;
          return slot;
        }
      }
      const int victim_slot = next_victim_slot_;
      next_victim_slot_ = (next_victim_slot_ + 1) % reducer_->cache_slots_;
      flush_slot(victim_slot);
      stripe_indices_[victim_slot] = stripe_index;
      return victim_slot;
    }

    void flush_slot(int slot) noexcept {
      const std::ptrdiff_t stripe_index = stripe_indices_[slot];
      auto& touched_offsets = touched_offsets_[slot];
      if (stripe_index < 0 || touched_offsets.empty()) {
        stripe_indices_[slot] = -1;
        touched_offsets.clear();
        return;
      }

      const std::size_t stripe_begin =
          static_cast<std::size_t>(stripe_index) * reducer_->stripe_size_;
      const std::size_t stripe_length = std::min(
          reducer_->stripe_size_,
          reducer_->target_->size() - stripe_begin);
      std::lock_guard<std::mutex> guard(
          reducer_->stripe_mutexes_[stripe_index]);
      auto& stripe_buffer = stripe_buffers_[slot];
      for (const std::size_t stripe_offset : touched_offsets) {
        if (stripe_offset >= stripe_length) {
          continue;
        }
        const std::size_t source_index = stripe_begin + stripe_offset;
        (*reducer_->target_)[source_index] += stripe_buffer[stripe_offset];
        stripe_buffer[stripe_offset] = 0.0;
      }
      touched_offsets.clear();
      stripe_indices_[slot] = -1;
    }

    AoH1eStripedTransposeReducer* reducer_ = nullptr;
    std::vector<std::ptrdiff_t> stripe_indices_;
    std::vector<std::vector<double>> stripe_buffers_;
    std::vector<std::vector<std::size_t>> touched_offsets_;
    int next_victim_slot_ = 0;
  };

  explicit AoH1eStripedTransposeReducer(std::vector<double>* target)
      : target_(target),
        stripe_size_(kAoH1eTransposeStripeSize),
        cache_slots_(kAoH1eTransposeStripeCacheSlots),
        stripe_mutexes_(compute_stripe_count(target)) {
    if (target_ == nullptr) {
      throw std::invalid_argument("AO-H1E striped reducer target must not be null");
    }
  }

private:
  static std::size_t compute_stripe_count(const std::vector<double>* target) {
    if (target == nullptr) {
      throw std::invalid_argument("AO-H1E striped reducer target must not be null");
    }
    return std::max<std::size_t>(
        1,
        (target->size() + kAoH1eTransposeStripeSize - 1) /
            kAoH1eTransposeStripeSize);
  }

  std::vector<double>* target_ = nullptr;
  std::size_t stripe_size_ = 0;
  int cache_slots_ = 0;
  std::vector<std::mutex> stripe_mutexes_;
};

}  // namespace

AoEffectiveOneElectronGraphBuffers build_ao_effective_one_electron_graph(
    const std::vector<int>& ao_effective_one_electron_linear_indices,
    const std::vector<double>& ao_two_electron_integral_values,
    const std::vector<std::uint8_t>& ao_two_electron_integral_symmetry_shifts,
    int n_basis_functions) {
  const std::size_t matrix_size = ao_matrix_size(n_basis_functions);
  if (ao_effective_one_electron_linear_indices.size() % 10 != 0) {
    throw std::invalid_argument(
        "AO-H1E graph requires 10 linear indices per AO integral");
  }
  const std::size_t n_integrals =
      ao_effective_one_electron_linear_indices.size() / 10;
  if (ao_two_electron_integral_values.size() != n_integrals ||
      ao_two_electron_integral_symmetry_shifts.size() != n_integrals) {
    throw std::invalid_argument(
        "AO-H1E graph integral buffers have inconsistent sizes");
  }

  constexpr std::size_t kEdgesPerIntegral = 6;
  if (n_integrals >
      std::numeric_limits<int>::max() / kEdgesPerIntegral) {
    throw std::overflow_error("AO-H1E graph edge count exceeds 32-bit storage");
  }
  const std::size_t n_edges = n_integrals * kEdgesPerIntegral;

  std::vector<int> row_counts(matrix_size, 0);
  for (std::size_t integral_index = 0; integral_index < n_integrals; ++integral_index) {
    const int* linear_indices =
        ao_effective_one_electron_linear_indices.data() + integral_index * 10;
    for (int entry_index = 0; entry_index < 6; ++entry_index) {
      const int row_index = linear_indices[entry_index];
      if (row_index < 0 || row_index >= matrix_size) {
        throw std::invalid_argument("AO-H1E graph destination row is out of range");
      }
      ++row_counts[row_index];
    }
    const int source_entries[] = {
        linear_indices[1],
        linear_indices[0],
        linear_indices[6],
        linear_indices[7],
        linear_indices[8],
        linear_indices[9],
    };
    for (int source_entry : source_entries) {
      if (source_entry < 0 || source_entry >= matrix_size) {
        throw std::invalid_argument("AO-H1E graph source row is out of range");
      }
    }
  }

  AoEffectiveOneElectronGraphBuffers graph;
  graph.row_offsets.resize(matrix_size + 1, 0);
  for (std::size_t row_index = 0; row_index < matrix_size; ++row_index) {
    graph.row_offsets[row_index + 1] =
        graph.row_offsets[row_index] + row_counts[row_index];
  }
  graph.source_indices.resize(n_edges, 0);
  graph.signed_weights.resize(n_edges, 0.0);

  std::vector<int> next_offsets = graph.row_offsets;
  for (std::size_t integral_index = 0; integral_index < n_integrals; ++integral_index) {
    const int* linear_indices =
        ao_effective_one_electron_linear_indices.data() + integral_index * 10;
    const std::uint8_t symmetry_shift =
        ao_two_electron_integral_symmetry_shifts[integral_index];
    if (symmetry_shift >= 4) {
      throw std::invalid_argument("AO-H1E graph symmetry shift is out of range");
    }
    const double scaled_value =
        ao_two_electron_integral_values[integral_index] *
        kIntegralSymmetryMultipliers[symmetry_shift];

    const int destination_entries[] = {
        linear_indices[0],
        linear_indices[1],
        linear_indices[2],
        linear_indices[3],
        linear_indices[4],
        linear_indices[5],
    };
    const int source_entries[] = {
        linear_indices[1],
        linear_indices[0],
        linear_indices[6],
        linear_indices[7],
        linear_indices[8],
        linear_indices[9],
    };
    const double signed_edge_weights[] = {
        4.0 * scaled_value,
        4.0 * scaled_value,
        -scaled_value,
        -scaled_value,
        -scaled_value,
        -scaled_value,
    };
    for (int entry_index = 0; entry_index < 6; ++entry_index) {
      const int row_index = destination_entries[entry_index];
      const int destination_offset = next_offsets[row_index]++;
      graph.source_indices[destination_offset] =
          source_entries[entry_index];
      graph.signed_weights[destination_offset] =
          signed_edge_weights[entry_index];
    }
  }

  std::vector<int> source_counts(matrix_size, 0);
  for (std::size_t edge_index = 0; edge_index < n_edges; ++edge_index) {
    ++source_counts[graph.source_indices[edge_index]];
  }
  graph.transpose_source_offsets.resize(matrix_size + 1, 0);
  for (std::size_t source_index = 0; source_index < matrix_size; ++source_index) {
    graph.transpose_source_offsets[source_index + 1] =
        graph.transpose_source_offsets[source_index] + source_counts[source_index];
  }
  graph.transpose_row_indices.resize(n_edges, 0);
  graph.transpose_signed_weights.resize(n_edges, 0.0);
  std::vector<int> next_source_offsets = graph.transpose_source_offsets;
  for (std::size_t row_index = 0; row_index < matrix_size; ++row_index) {
    for (int edge_offset = graph.row_offsets[row_index];
         edge_offset < graph.row_offsets[row_index + 1];
         ++edge_offset) {
      const std::size_t edge_index = edge_offset;
      const int source_index = graph.source_indices[edge_index];
      const int destination_offset =
          next_source_offsets[source_index]++;
      graph.transpose_row_indices[destination_offset] =
          static_cast<int>(row_index);
      graph.transpose_signed_weights[destination_offset] =
          graph.signed_weights[edge_index];
    }
  }

  return graph;
}
bool ao_effective_one_electron_graph_available(
    const AoIntegralInput& ao_integral_input) noexcept {
  return !ao_integral_input.ao_effective_one_electron_graph_row_offsets.empty();
}

std::vector<double> apply_ao_effective_one_electron_graph_forward(
    const double* source_matrix_storage,
    const AoIntegralInput& ao_integral_input) {
  return apply_ao_effective_one_electron_graph_forward(
      source_matrix_storage,
      ao_integral_input,
      1);
}

std::vector<double> apply_ao_effective_one_electron_graph_forward(
    const double* source_matrix_storage,
    const AoIntegralInput& ao_integral_input,
    int n_threads) {
  if (source_matrix_storage == nullptr) {
    throw std::invalid_argument("AO-H1E graph forward source must not be null");
  }
  const std::size_t matrix_size =
      ao_matrix_size(ao_integral_input.n_basis_functions);
  validate_graph_storage_shapes(ao_integral_input, matrix_size);
  n_threads = sanitize_graph_thread_count(n_threads);

  std::vector<double> result(matrix_size, 0.0);
  const auto& row_offsets =
      ao_integral_input.ao_effective_one_electron_graph_row_offsets;
  const auto& source_indices =
      ao_integral_input.ao_effective_one_electron_graph_source_indices;
  const auto& signed_weights =
      ao_integral_input.ao_effective_one_electron_graph_signed_weights;

  if (n_threads <= 1) {
    for (std::size_t row_index = 0; row_index < matrix_size; ++row_index) {
      double row_value = 0.0;
      for (int edge_offset = row_offsets[row_index];
           edge_offset < row_offsets[row_index + 1];
           ++edge_offset) {
        row_value +=
            signed_weights[edge_offset] *
            source_matrix_storage[
                source_indices[edge_offset]];
      }
      result[row_index] = row_value;
    }
    return result;
  }

#pragma omp parallel for schedule(static) num_threads(n_threads)
  for (std::ptrdiff_t row_offset = 0;
       row_offset < static_cast<std::ptrdiff_t>(matrix_size);
       ++row_offset) {
    const std::size_t row_index = row_offset;
    double row_value = 0.0;
    for (int edge_offset = row_offsets[row_index];
         edge_offset < row_offsets[row_index + 1];
         ++edge_offset) {
      row_value +=
          signed_weights[edge_offset] *
          source_matrix_storage[
              source_indices[edge_offset]];
    }
    result[row_index] = row_value;
  }
  return result;
}

std::vector<double> apply_ao_effective_one_electron_graph_transpose(
    const double* row_adjoint_storage,
    const AoIntegralInput& ao_integral_input) {
  return apply_ao_effective_one_electron_graph_transpose(
      row_adjoint_storage,
      ao_integral_input,
      1);
}

std::vector<double> apply_ao_effective_one_electron_graph_transpose(
    const double* row_adjoint_storage,
    const AoIntegralInput& ao_integral_input,
    int n_threads) {
  if (row_adjoint_storage == nullptr) {
    throw std::invalid_argument("AO-H1E graph transpose source must not be null");
  }
  const std::size_t matrix_size =
      ao_matrix_size(ao_integral_input.n_basis_functions);
  validate_graph_storage_shapes(ao_integral_input, matrix_size);
  validate_transpose_graph_storage_shapes(ao_integral_input, matrix_size);
  n_threads = sanitize_graph_thread_count(n_threads);

  std::vector<double> result(matrix_size, 0.0);

  if (transpose_graph_available(ao_integral_input)) {
    const auto& source_offsets =
        ao_integral_input
            .ao_effective_one_electron_graph_transpose_source_offsets;
    const auto& row_indices =
        ao_integral_input.ao_effective_one_electron_graph_transpose_row_indices;
    const auto& signed_weights =
        ao_integral_input
            .ao_effective_one_electron_graph_transpose_signed_weights;
    if (n_threads <= 1) {
      for (std::size_t source_index = 0; source_index < matrix_size; ++source_index) {
        double source_value = 0.0;
        for (int edge_offset = source_offsets[source_index];
             edge_offset < source_offsets[source_index + 1];
             ++edge_offset) {
          const std::size_t edge_index = edge_offset;
          source_value +=
              signed_weights[edge_index] *
              row_adjoint_storage[row_indices[edge_index]];
        }
        result[source_index] = source_value;
      }
      return result;
    }
#pragma omp parallel for schedule(static) num_threads(n_threads)
    for (std::ptrdiff_t source_offset = 0;
         source_offset < static_cast<std::ptrdiff_t>(matrix_size);
         ++source_offset) {
      const std::size_t source_index = source_offset;
      double source_value = 0.0;
      for (int edge_offset = source_offsets[source_index];
           edge_offset < source_offsets[source_index + 1];
           ++edge_offset) {
        const std::size_t edge_index = edge_offset;
        source_value +=
            signed_weights[edge_index] *
            row_adjoint_storage[row_indices[edge_index]];
      }
      result[source_index] = source_value;
    }
    return result;
  }

  const auto& row_offsets =
      ao_integral_input.ao_effective_one_electron_graph_row_offsets;
  const auto& source_indices =
      ao_integral_input.ao_effective_one_electron_graph_source_indices;
  const auto& signed_weights =
      ao_integral_input.ao_effective_one_electron_graph_signed_weights;
  if (n_threads <= 1) {
    for (std::size_t row_index = 0; row_index < matrix_size; ++row_index) {
      const double row_adjoint = row_adjoint_storage[row_index];
      if (row_adjoint == 0.0) {
        continue;
      }
      for (int edge_offset = row_offsets[row_index];
           edge_offset < row_offsets[row_index + 1];
           ++edge_offset) {
        result[source_indices[edge_offset]] +=
            signed_weights[edge_offset] * row_adjoint;
      }
    }
    return result;
  }

  // If the input deck did not materialize the transpose companion graph we
  // still keep the multithreaded graph path on bounded memory by reducing the
  // source updates through a striped cache.
  AoH1eStripedTransposeReducer transpose_reducer(&result);
#pragma omp parallel num_threads(n_threads)
  {
    AoH1eStripedTransposeReducer::ThreadLocalAccumulator
        local_transpose_accumulator(&transpose_reducer);

#pragma omp for schedule(static)
    for (std::ptrdiff_t row_offset = 0;
         row_offset < static_cast<std::ptrdiff_t>(matrix_size);
         ++row_offset) {
      const std::size_t row_index = row_offset;
      const double row_adjoint = row_adjoint_storage[row_index];
      if (row_adjoint == 0.0) {
        continue;
      }
      for (int edge_offset = row_offsets[row_index];
           edge_offset < row_offsets[row_index + 1];
           ++edge_offset) {
        const std::size_t edge_index = edge_offset;
        local_transpose_accumulator.add(
            source_indices[edge_index],
            signed_weights[edge_index] * row_adjoint);
      }
    }
  }
  return result;
}

AoEffectiveOneElectronGraphFusedResult
apply_fused_ao_effective_one_electron_graph(
    const double* source_matrix_storage,
    const double* row_adjoint_storage,
    const AoIntegralInput& ao_integral_input) {
  return apply_fused_ao_effective_one_electron_graph(
      source_matrix_storage,
      row_adjoint_storage,
      ao_integral_input,
      1);
}

AoEffectiveOneElectronGraphFusedResult
apply_fused_ao_effective_one_electron_graph(
    const double* source_matrix_storage,
    const double* row_adjoint_storage,
    const AoIntegralInput& ao_integral_input,
    int n_threads) {
  AoEffectiveOneElectronGraphFusedResult result;
  apply_fused_ao_effective_one_electron_graph(
      source_matrix_storage,
      row_adjoint_storage,
      ao_integral_input,
      n_threads,
      &result.forward_output,
      &result.transpose_output);
  return result;
}

void apply_fused_ao_effective_one_electron_graph(
    const double* source_matrix_storage,
    const double* row_adjoint_storage,
    const AoIntegralInput& ao_integral_input,
    std::vector<double>* forward_output,
    std::vector<double>* transpose_output) {
  apply_fused_ao_effective_one_electron_graph(
      source_matrix_storage,
      row_adjoint_storage,
      ao_integral_input,
      1,
      forward_output,
      transpose_output);
}

void apply_fused_ao_effective_one_electron_graph(
    const double* source_matrix_storage,
    const double* row_adjoint_storage,
    const AoIntegralInput& ao_integral_input,
    int n_threads,
    std::vector<double>* forward_output,
    std::vector<double>* transpose_output) {
  if (source_matrix_storage == nullptr || row_adjoint_storage == nullptr) {
    throw std::invalid_argument("AO-H1E graph fused inputs must not be null");
  }
  if (forward_output == nullptr || transpose_output == nullptr) {
    throw std::invalid_argument("AO-H1E graph fused outputs must not be null");
  }
  const std::size_t matrix_size =
      ao_matrix_size(ao_integral_input.n_basis_functions);
  validate_graph_storage_shapes(ao_integral_input, matrix_size);
  n_threads = sanitize_graph_thread_count(n_threads);

  forward_output->assign(matrix_size, 0.0);
  transpose_output->assign(matrix_size, 0.0);
  const auto& row_offsets =
      ao_integral_input.ao_effective_one_electron_graph_row_offsets;
  const auto& source_indices =
      ao_integral_input.ao_effective_one_electron_graph_source_indices;
  const auto& signed_weights =
      ao_integral_input.ao_effective_one_electron_graph_signed_weights;

  if (n_threads <= 1) {
    // On one thread the row-owned graph is still the cheapest route because it
    // computes the forward image and transpose pullback together without
    // revisiting the sparse edge list.
    for (std::size_t row_index = 0; row_index < matrix_size; ++row_index) {
      double row_forward_value = 0.0;
      const double row_adjoint = row_adjoint_storage[row_index];
      for (int edge_offset = row_offsets[row_index];
           edge_offset < row_offsets[row_index + 1];
           ++edge_offset) {
        const std::size_t edge_index = edge_offset;
        const std::size_t source_index =
            source_indices[edge_index];
        const double signed_weight = signed_weights[edge_index];
        row_forward_value += signed_weight * source_matrix_storage[source_index];
        (*transpose_output)[source_index] += signed_weight * row_adjoint;
      }
      (*forward_output)[row_index] = row_forward_value;
    }
    return;
  }

  if (transpose_graph_available(ao_integral_input)) {
    validate_transpose_graph_storage_shapes(ao_integral_input, matrix_size);
    const auto& transpose_source_offsets =
        ao_integral_input
            .ao_effective_one_electron_graph_transpose_source_offsets;
    const auto& transpose_row_indices =
        ao_integral_input.ao_effective_one_electron_graph_transpose_row_indices;
    const auto& transpose_signed_weights =
        ao_integral_input
            .ao_effective_one_electron_graph_transpose_signed_weights;

    // The molecule-static companion graph gives the transpose apply a
    // source-owned partition, so the multithreaded exact_ctx path can stay
    // lock-free and avoid the cache-thrashing flush traffic of the striped
    // fallback.
#pragma omp parallel num_threads(n_threads)
    {
#pragma omp for schedule(static)
      for (std::ptrdiff_t row_offset = 0;
           row_offset < static_cast<std::ptrdiff_t>(matrix_size);
           ++row_offset) {
        const std::size_t row_index = row_offset;
        double row_forward_value = 0.0;
        for (int edge_offset = row_offsets[row_index];
             edge_offset < row_offsets[row_index + 1];
             ++edge_offset) {
          const std::size_t edge_index = edge_offset;
          row_forward_value +=
              signed_weights[edge_index] *
              source_matrix_storage[source_indices[edge_index]];
        }
        (*forward_output)[row_index] = row_forward_value;
      }

#pragma omp for schedule(static)
      for (std::ptrdiff_t source_offset = 0;
           source_offset < static_cast<std::ptrdiff_t>(matrix_size);
           ++source_offset) {
        const std::size_t source_index = source_offset;
        double source_value = 0.0;
        for (int edge_offset = transpose_source_offsets[source_index];
             edge_offset < transpose_source_offsets[source_index + 1];
             ++edge_offset) {
          const std::size_t edge_index = edge_offset;
          source_value +=
              transpose_signed_weights[edge_index] *
              row_adjoint_storage[transpose_row_indices[edge_index]];
        }
        (*transpose_output)[source_index] = source_value;
      }
    }
    return;
  }

  // If the transpose companion graph is unavailable, stay on bounded memory by
  // reducing the row-owned transpose contributions through a small striped
  // cache instead of falling back to one full AO matrix per worker.
  AoH1eStripedTransposeReducer transpose_reducer(transpose_output);
#pragma omp parallel num_threads(n_threads)
  {
    AoH1eStripedTransposeReducer::ThreadLocalAccumulator
        local_transpose_accumulator(&transpose_reducer);

#pragma omp for schedule(static)
    for (std::ptrdiff_t row_offset = 0;
         row_offset < static_cast<std::ptrdiff_t>(matrix_size);
         ++row_offset) {
      const std::size_t row_index = row_offset;
      double row_forward_value = 0.0;
      const double row_adjoint = row_adjoint_storage[row_index];
      for (int edge_offset = row_offsets[row_index];
           edge_offset < row_offsets[row_index + 1];
           ++edge_offset) {
        const std::size_t edge_index = edge_offset;
        const std::size_t source_index =
            source_indices[edge_index];
        const double signed_weight = signed_weights[edge_index];
        row_forward_value += signed_weight * source_matrix_storage[source_index];
        local_transpose_accumulator.add(
            source_index,
            signed_weight * row_adjoint);
      }
      (*forward_output)[row_index] = row_forward_value;
    }
  }
}

void apply_fused_ao_effective_one_electron_graph_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& source_matrix_columns,
    const Eigen::Ref<const Eigen::MatrixXd>& row_adjoint_columns,
    const AoIntegralInput& ao_integral_input,
    int n_threads,
    Eigen::MatrixXd* forward_output_columns,
    Eigen::MatrixXd* transpose_output_columns) {
  if (forward_output_columns == nullptr || transpose_output_columns == nullptr) {
    throw std::invalid_argument("AO-H1E graph batch outputs must not be null");
  }
  const std::size_t matrix_size =
      ao_matrix_size(ao_integral_input.n_basis_functions);
  validate_graph_storage_shapes(ao_integral_input, matrix_size);
  if (source_matrix_columns.rows() !=
          static_cast<Eigen::Index>(matrix_size) ||
      row_adjoint_columns.rows() !=
          static_cast<Eigen::Index>(matrix_size) ||
      source_matrix_columns.cols() != row_adjoint_columns.cols()) {
    throw std::invalid_argument("AO-H1E graph batch input shape mismatch");
  }
  const Eigen::Index n_directions = source_matrix_columns.cols();
  using DirectionMajorMatrix =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const DirectionMajorMatrix source_rows = source_matrix_columns;
  const DirectionMajorMatrix adjoint_rows = row_adjoint_columns;
  DirectionMajorMatrix forward_rows =
      DirectionMajorMatrix::Zero(matrix_size, n_directions);
  DirectionMajorMatrix transpose_rows =
      DirectionMajorMatrix::Zero(matrix_size, n_directions);
  forward_output_columns->resize(matrix_size, n_directions);
  transpose_output_columns->resize(matrix_size, n_directions);
  if (n_directions == 0) {
    return;
  }
  n_threads = sanitize_graph_thread_count(n_threads);
  const auto& row_offsets =
      ao_integral_input.ao_effective_one_electron_graph_row_offsets;
  const auto& source_indices =
      ao_integral_input.ao_effective_one_electron_graph_source_indices;
  const auto& signed_weights =
      ao_integral_input.ao_effective_one_electron_graph_signed_weights;

#pragma omp parallel for schedule(static) num_threads(n_threads) if(n_threads > 1)
  for (std::ptrdiff_t row_offset = 0;
       row_offset < static_cast<std::ptrdiff_t>(matrix_size);
       ++row_offset) {
    const Eigen::Index row = row_offset;
    for (int edge_offset = row_offsets[row];
         edge_offset < row_offsets[row + 1];
         ++edge_offset) {
      const Eigen::Index source = source_indices[edge_offset];
      const double weight = signed_weights[edge_offset];
      for (Eigen::Index direction = 0;
           direction < n_directions;
           ++direction) {
        forward_rows(row, direction) +=
            weight * source_rows(source, direction);
      }
    }
  }

  if (!transpose_graph_available(ao_integral_input)) {
    // The source-owned companion is required for a lock-free fused transpose.
    // Preserve exact semantics on older inputs without allocating a dense
    // per-thread block accumulator.
    for (Eigen::Index direction = 0;
         direction < n_directions;
         ++direction) {
      std::vector<double> forward_column;
      std::vector<double> transpose_column;
      apply_fused_ao_effective_one_electron_graph(
          source_matrix_columns.col(direction).data(),
          row_adjoint_columns.col(direction).data(),
          ao_integral_input,
          n_threads,
          &forward_column,
          &transpose_column);
      forward_output_columns->col(direction) = Eigen::Map<Eigen::VectorXd>(
          forward_column.data(), forward_column.size());
      transpose_output_columns->col(direction) = Eigen::Map<Eigen::VectorXd>(
          transpose_column.data(), transpose_column.size());
    }
    return;
  }

  validate_transpose_graph_storage_shapes(ao_integral_input, matrix_size);
  const auto& transpose_source_offsets =
      ao_integral_input
          .ao_effective_one_electron_graph_transpose_source_offsets;
  const auto& transpose_row_indices =
      ao_integral_input.ao_effective_one_electron_graph_transpose_row_indices;
  const auto& transpose_signed_weights =
      ao_integral_input
          .ao_effective_one_electron_graph_transpose_signed_weights;
#pragma omp parallel for schedule(static) num_threads(n_threads) if(n_threads > 1)
  for (std::ptrdiff_t source_offset = 0;
       source_offset < static_cast<std::ptrdiff_t>(matrix_size);
       ++source_offset) {
    const Eigen::Index source = source_offset;
    for (int edge_offset = transpose_source_offsets[source];
         edge_offset < transpose_source_offsets[source + 1];
         ++edge_offset) {
      const Eigen::Index row = transpose_row_indices[edge_offset];
      const double weight = transpose_signed_weights[edge_offset];
      for (Eigen::Index direction = 0;
           direction < n_directions;
           ++direction) {
        transpose_rows(source, direction) +=
            weight * adjoint_rows(row, direction);
      }
    }
  }
  *forward_output_columns = forward_rows;
  *transpose_output_columns = transpose_rows;
}

}  // namespace xmvb::vb
