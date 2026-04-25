#pragma once

#include <cstddef>
#include <deque>
#include <stdexcept>
#include <vector>

namespace xmvb::pfaffian_vbscf {

struct PfPackedIndexCache {
  int n_active_orbitals = 0;
  int n_packed_pairs = 0;
  std::vector<std::size_t> pair_of_pairs_indices;
  std::vector<std::size_t> quartet_indices;
};

inline int packed_pair_index(int index_a, int index_b) {
  if (index_a >= index_b) {
    return index_a * (index_a + 1) / 2 + index_b;
  }
  return index_b * (index_b + 1) / 2 + index_a;
}

inline std::size_t packed_pair_of_pairs_index(
    int packed_pair_index_a,
    int packed_pair_index_b) {
  return 
      packed_pair_index(packed_pair_index_a, packed_pair_index_b);
}

inline int packed_pair_index_slow(int index_a, int index_b) {
  return packed_pair_index(index_a, index_b);
}

inline PfPackedIndexCache build_packed_index_cache(int n_active_orbitals) {
  if (n_active_orbitals < 0) {
    throw std::invalid_argument("n_active_orbitals must be non-negative");
  }

  PfPackedIndexCache cache;
  cache.n_active_orbitals = n_active_orbitals;
  cache.n_packed_pairs = n_active_orbitals * (n_active_orbitals + 1) / 2;
  cache.pair_of_pairs_indices.resize(
      cache.n_packed_pairs * cache.n_packed_pairs,
      0);
  for (int packed_pair_a = 0; packed_pair_a < cache.n_packed_pairs; ++packed_pair_a) {
    for (int packed_pair_b = 0; packed_pair_b < cache.n_packed_pairs; ++packed_pair_b) {
      cache.pair_of_pairs_indices[
          packed_pair_a * cache.n_packed_pairs +
          packed_pair_b] =
          packed_pair_of_pairs_index(packed_pair_a, packed_pair_b);
    }
  }
  cache.quartet_indices.resize(
      n_active_orbitals *
      n_active_orbitals *
      n_active_orbitals *
      n_active_orbitals,
      0);

  for (int q = 0; q < n_active_orbitals; ++q) {
    for (int p = 0; p < n_active_orbitals; ++p) {
      const int idx_qp = packed_pair_index(q, p);
      for (int s = 0; s < n_active_orbitals; ++s) {
        for (int r = 0; r < n_active_orbitals; ++r) {
          const int idx_sr = packed_pair_index(s, r);
          cache.quartet_indices[
              
                  (((q * n_active_orbitals) + p) * n_active_orbitals + s) *
                  n_active_orbitals + r] =
              cache.pair_of_pairs_indices[
                  idx_qp * cache.n_packed_pairs +
                  idx_sr];
        }
      }
    }
  }
  return cache;
}

inline const PfPackedIndexCache& get_packed_index_cache(int n_active_orbitals) {
  if (n_active_orbitals < 0) {
    throw std::invalid_argument("n_active_orbitals must be non-negative");
  }

  struct CacheStore {
    std::deque<PfPackedIndexCache> caches;
  };

  static thread_local CacheStore store;
  for (const PfPackedIndexCache& cache : store.caches) {
    if (cache.n_active_orbitals == n_active_orbitals) {
      return cache;
    }
  }
  store.caches.push_back(build_packed_index_cache(n_active_orbitals));
  return store.caches.back();
}

inline std::size_t quartet_index(
    const PfPackedIndexCache& cache,
    int q,
    int p,
    int s,
    int r) {
  const int n = cache.n_active_orbitals;
  return cache.quartet_indices[
      
          (((q * n) + p) * n + s) * n + r];
}

inline std::size_t pair_of_pairs_index(
    const PfPackedIndexCache& cache,
    int packed_pair_index_a,
    int packed_pair_index_b) {
  return cache.pair_of_pairs_indices[
      packed_pair_index_a * cache.n_packed_pairs +
      packed_pair_index_b];
}

}  // namespace xmvb::pfaffian_vbscf
