#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace xmvb::vb {

struct UnorderedDeterminantPairIndices {
    Eigen::Index left = 0;
    Eigen::Index right = 0;
};

inline std::size_t unordered_determinant_pair_count(Eigen::Index n_determinants) {
    if (n_determinants < 0)
        throw std::invalid_argument("negative spin-pair dimension");
    const auto n = static_cast<std::size_t>(n_determinants);
    const auto a = n % 2 ? n : n / 2, b = n % 2 ? (n + 1) / 2 : n + 1;
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("spin-pair count exceeds size_t range");
    return a * b;
}

/**
 * @brief Chooses a deterministic chunk size for striped determinant-pair work.
 *
 * The unordered pair storage is triangular: contiguous ranges can align with
 * determinant rows that have different skip rates or cache behavior.  Splitting
 * the linear storage into a few chunks per worker and assigning those chunks in
 * thread-index order gives better load balance while preserving a stable
 * mapping from pair index to thread for a fixed thread count.
 */
inline std::size_t unordered_determinant_pair_parallel_chunk_size(std::size_t n_determinant_pairs,
                                                                  int n_threads) {
    if (n_determinant_pairs == 0) {
        return 1;
    }
    const std::size_t thread_count = n_threads > 0 ? static_cast<std::size_t>(n_threads) : 1;
    const std::size_t target_chunks = std::max<std::size_t>(1, thread_count * 8);
    const std::size_t chunk_size =
        n_determinant_pairs / target_chunks + (n_determinant_pairs % target_chunks != 0);
    return std::max<std::size_t>(1, std::min<std::size_t>(chunk_size, 4096));
}

inline std::size_t canonical_determinant_pair_storage_index(Eigen::Index determinant_index_left,
                                                            Eigen::Index determinant_index_right) {
    if (determinant_index_left < 0 || determinant_index_right < 0)
        throw std::invalid_argument("negative spin-pair index");
    if (determinant_index_left < determinant_index_right) {
        std::swap(determinant_index_left, determinant_index_right);
    }
    const auto first = unordered_determinant_pair_count(determinant_index_left);
    const auto right = static_cast<std::size_t>(determinant_index_right);
    if (right > std::numeric_limits<std::size_t>::max() - first)
        throw std::overflow_error("spin-pair address exceeds size_t range");
    return first + right;
}

inline UnorderedDeterminantPairIndices
determinant_pair_from_storage_index(std::size_t storage_index) {
    const long double storage_index_ld = static_cast<long double>(storage_index);
    Eigen::Index determinant_index_left =
        static_cast<Eigen::Index>((std::sqrt(8.0L * storage_index_ld + 1.0L) - 1.0L) * 0.5L);
    const auto fits = [&](Eigen::Index row) {
        const auto n = static_cast<std::size_t>(row);
        const auto a = n % 2 ? n : n / 2, b = n % 2 ? (n + 1) / 2 : n + 1;
        return !b || a <= storage_index / b;
    };
    // Correct the floating-point estimate with exact comparisons. Even the
    // next row may exceed size_t, so never form its unchecked product.
    while (!fits(determinant_index_left)) {
        --determinant_index_left;
    }
    std::size_t row_start = unordered_determinant_pair_count(determinant_index_left);
    while (storage_index - row_start >= static_cast<std::size_t>(determinant_index_left) + 1) {
        ++determinant_index_left;
        row_start += determinant_index_left;
    }
    return {determinant_index_left, static_cast<Eigen::Index>(storage_index - row_start)};
}

inline void advance_unordered_determinant_pair(UnorderedDeterminantPairIndices* determinant_pair) {
    if (determinant_pair == nullptr) {
        return;
    }
    if (determinant_pair->right < determinant_pair->left) {
        ++determinant_pair->right;
    } else {
        ++determinant_pair->left;
        determinant_pair->right = 0;
    }
}

} // namespace xmvb::vb
