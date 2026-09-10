#include "vbscf/determinants/same_spin_pair_cache.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "core/openmp_utils.hpp"
#include "vbscf/determinants/pair_storage.hpp"
#include "vbscf/determinants/spin_pair_contractions.hpp"
#include "vbscf/determinants/cofactor_differential.hpp"
#include "vbscf/integrals/active/two_electron_indexer.hpp"
#include "vbscf/integrals/active/active_space_two_electron_kernel.hpp"

namespace xmvb::vb {

namespace {

constexpr double kContributionTolerance = 1.0e-15;

int same_spin_pair_cache_thread_count(int n_unique_determinants) {
  // Same-spin cache construction is row-parallel over unique determinants.
  // Keep tiny determinant spaces serial and cap the team at the number of rows
  // so dynamic scheduling does not launch workers with no cache rows to own.
  if (n_unique_determinants <= 10) {
    return 1;
  }
  return std::max(
      1,
      std::min(
          xmvb::effective_openmp_thread_count(),
          n_unique_determinants));
}

bool can_reuse_close_shell_same_spin_pair_cache(
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det) {
  return alpha_det == beta_det;
}

struct SpinDeterminantHasher {
  std::size_t operator()(const std::vector<int>& occupied_orbitals) const {
    std::size_t hash_value = occupied_orbitals.size();
    for (const int orbital_index : occupied_orbitals) {
      hash_value =
          hash_value * 1315423911u + orbital_index + 257;
    }
    return hash_value;
  }
};

bool try_build_reuse_table_in_reference_basis(
    const std::vector<std::vector<int>>& spin_determinants,
    const SpinDeterminantReuseTable& reference_reuse_table,
    SpinDeterminantReuseTable* mapped_reuse_table) {
  if (mapped_reuse_table == nullptr) {
    throw std::invalid_argument("mapped_reuse_table must not be null");
  }

  std::unordered_map<std::vector<int>, int, SpinDeterminantHasher>
      reference_unique_id_by_determinant;
  reference_unique_id_by_determinant.reserve(
      reference_reuse_table.unique_determinants.size());
  for (int unique_index = 0;
       unique_index < static_cast<int>(
           reference_reuse_table.unique_determinants.size());
       ++unique_index) {
    reference_unique_id_by_determinant.emplace(
        reference_reuse_table.unique_determinants[unique_index],
        unique_index);
  }

  mapped_reuse_table->unique_determinants =
      reference_reuse_table.unique_determinants;
  mapped_reuse_table->determinant_to_unique_id.assign(
      spin_determinants.size(),
      -1);
  std::vector<unsigned char> touched_unique_ids(
      reference_reuse_table.unique_determinants.size(),
      0u);

  for (std::size_t determinant_index = 0;
       determinant_index < spin_determinants.size();
       ++determinant_index) {
    const auto iterator =
        reference_unique_id_by_determinant.find(
            spin_determinants[determinant_index]);
    if (iterator == reference_unique_id_by_determinant.end()) {
      mapped_reuse_table->unique_determinants.clear();
      mapped_reuse_table->determinant_to_unique_id.clear();
      return false;
    }
    mapped_reuse_table->determinant_to_unique_id[determinant_index] =
        iterator->second;
    touched_unique_ids[iterator->second] = 1u;
  }

  for (const unsigned char touched : touched_unique_ids) {
    if (touched == 0u) {
      mapped_reuse_table->unique_determinants.clear();
      mapped_reuse_table->determinant_to_unique_id.clear();
      return false;
    }
  }
  return true;
}

std::size_t expected_opposite_spin_scalar_cache_size(
    const SameSpinPairCacheContext& cache_context) {
  return cache_context.n_unique_alpha *
      cache_context.n_unique_alpha *
      cache_context.n_unique_beta *
      cache_context.n_unique_beta;
}

bool has_materialized_opposite_spin_scalar_cache(
    const SameSpinPairCacheContext& cache_context) {
  const std::size_t expected_size =
      expected_opposite_spin_scalar_cache_size(cache_context);
  if (expected_size == 0) {
    return false;
  }
  if (cache_context.opposite_spin_cache.empty() &&
      cache_context.opposite_spin_cache_computed.empty()) {
    return false;
  }
  if (cache_context.opposite_spin_cache.size() != expected_size ||
      cache_context.opposite_spin_cache_computed.size() != expected_size) {
    throw std::invalid_argument(
        "opposite-spin scalar cache storage does not match unique alpha/beta dimensions");
  }
  return true;
}

std::vector<double> build_dense_active_pair_kernel(
    const std::vector<double>& eri_act,
    int n_orbitals) {
  const int n_packed_active_pairs = packed_active_pair_count(n_orbitals);
  std::vector<double> dense_pair_kernel(
      n_packed_active_pairs *
          n_packed_active_pairs,
      0.0);

  for (int column_pair = 0; column_pair < n_packed_active_pairs; ++column_pair) {
    for (int row_pair = 0; row_pair < n_packed_active_pairs; ++row_pair) {
      const int packed_pair_of_pairs_index =
          TwoElectronIndexer::packed_pair_of_pairs_index(row_pair, column_pair);
      dense_pair_kernel[column_pair *
                            n_packed_active_pairs +
                        row_pair] =
          eri_act[packed_pair_of_pairs_index];
    }
  }

  return dense_pair_kernel;
}

OppositeSpinPackedPairProjection build_sparse_packed_pair_projection(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& coefficient_matrix,
    bool coefficient_matrix_is_right_by_left,
    int n_orbitals,
    const ActiveSpaceTwoElectronView& two_electron_view,
    const std::vector<double>* dense_pair_kernel,
    bool materialize_projected_pair_values) {
  const int n_packed_active_pairs = packed_active_pair_count(n_orbitals);
  OppositeSpinPackedPairProjection projection;
  if (occ_L.empty()) {
    return projection;
  }

  std::vector<double> dense_pair_values(
      n_packed_active_pairs,
      0.0);
  std::vector<unsigned char> touched_mask(
      n_packed_active_pairs,
      0);
  std::vector<int> touched_indices;
  touched_indices.reserve(occ_L.size() * occ_R.size());

  for (int left_column = 0; left_column < static_cast<int>(occ_L.size()); ++left_column) {
    const int orbital_index_left = occ_L[left_column];
    for (int right_row = 0; right_row < static_cast<int>(occ_R.size()); ++right_row) {
      const int orbital_index_right = occ_R[right_row];
      const int packed_pair_index = TwoElectronIndexer::packed_pair_index(
          orbital_index_right,
          orbital_index_left);
      if (touched_mask[packed_pair_index] == 0) {
        touched_mask[packed_pair_index] = 1;
        touched_indices.push_back(packed_pair_index);
      }
      const double coefficient =
          coefficient_matrix_is_right_by_left
              ? coefficient_matrix(right_row, left_column)
              : coefficient_matrix(left_column, right_row);
      dense_pair_values[packed_pair_index] += coefficient;
    }
  }

  projection.packed_pair_indices.reserve(touched_indices.size());
  projection.packed_pair_values.reserve(touched_indices.size());
  for (const int packed_pair_index : touched_indices) {
    const double packed_pair_value =
        dense_pair_values[packed_pair_index];
    if (packed_pair_value == 0.0) {
      continue;
    }
    projection.packed_pair_indices.push_back(packed_pair_index);
    projection.packed_pair_values.push_back(packed_pair_value);
  }

  if (projection.packed_pair_indices.empty()) {
    projection.projected_pair_values.clear();
    return projection;
  }

  if (!materialize_projected_pair_values) {
    projection.projected_pair_values.clear();
    return projection;
  }

  if (dense_pair_kernel != nullptr) {
    projection.projected_pair_values.assign(
        n_packed_active_pairs,
        0.0);
    for (std::size_t entry_index = 0;
         entry_index < projection.packed_pair_indices.size();
         ++entry_index) {
      const int packed_pair_index = projection.packed_pair_indices[entry_index];
      const double packed_pair_value = projection.packed_pair_values[entry_index];
      const std::size_t kernel_column_offset =
          packed_pair_index * n_packed_active_pairs;
      for (int row_pair = 0; row_pair < n_packed_active_pairs; ++row_pair) {
        projection.projected_pair_values[row_pair] +=
            (*dense_pair_kernel)[kernel_column_offset + row_pair] *
            packed_pair_value;
      }
    }
  } else {
    projection.projected_pair_values =
        apply_active_space_two_electron_kernel_to_sparse_projection(
            two_electron_view,
            n_orbitals,
            projection.packed_pair_indices,
            projection.packed_pair_values);
  }
  return projection;
}

void attach_opposite_spin_pair_cache(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    int n_orbitals,
    const ActiveSpaceTwoElectronView& two_electron_view,
    const std::vector<double>* dense_pair_kernel,
    bool materialize_projected_pair_values,
    SpinDeterminantPairEvaluation* pair_evaluation) {
  if (pair_evaluation == nullptr) {
    throw std::invalid_argument("pair_evaluation must not be null");
  }

  pair_evaluation->opposite_spin_pair_cache.n_packed_active_pairs =
      packed_active_pair_count(n_orbitals);

  const auto& overlap_result = pair_evaluation->overlap_result;
  if (occ_L.empty() || overlap_result.nullity >= 2) {
    return;
  }
  pair_evaluation->opposite_spin_pair_cache.first_order_cofactor_projection =
      build_sparse_packed_pair_projection(
          occ_L,
          occ_R,
          calc_cofactor_1st(overlap_result),
          true,
          n_orbitals,
          two_electron_view,
          dense_pair_kernel,
          materialize_projected_pair_values);

  if (overlap_result.nullity == 0) {
    pair_evaluation->opposite_spin_pair_cache.inverse_overlap_projection =
        build_sparse_packed_pair_projection(
            occ_L,
            occ_R,
            build_inverse_overlap_submatrix_from_result(overlap_result),
            false,
            n_orbitals,
            two_electron_view,
            dense_pair_kernel,
            materialize_projected_pair_values);
  }
}

Eigen::MatrixXd build_spin_one_electron_block_matrix(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act) {
  const int n_electrons = static_cast<int>(occ_L.size());
  Eigen::MatrixXd one_electron_block(n_electrons, n_electrons);
  for (int left_column = 0; left_column < n_electrons; ++left_column) {
    const int orbital_index_left = occ_L[left_column];
    for (int right_row = 0; right_row < n_electrons; ++right_row) {
      const int orbital_index_right = occ_R[right_row];
      one_electron_block(right_row, left_column) =
          h1e_act(orbital_index_right, orbital_index_left);
    }
  }
  return one_electron_block;
}

Eigen::MatrixXd build_polynomial_same_spin_hamiltonian_overlap_gradient(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_active_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    SpinDeterminantPairEvaluation* pair_evaluation) {
  if (pair_evaluation == nullptr)
    throw std::invalid_argument("same-spin polynomial cache output must not be null");
  const CofactorDifferential& cofactor = cached_cofactor_differential(*pair_evaluation);
  pair_evaluation->same_spin_one_electron_block =
      build_spin_one_electron_block_matrix(occ_L, occ_R, h1e_act);
  pair_evaluation->same_spin_antisymmetrized_interaction =
      build_spin_antisymmetrized_interaction_matrix(
      occ_L, occ_R, n_active_orbitals,
      make_active_space_two_electron_view(active_space_two_electron_result));
  return cofactor.first(pair_evaluation->same_spin_one_electron_block) +
      cofactor.second_contraction_gradient(
          pair_evaluation->same_spin_antisymmetrized_interaction);
}

std::vector<SpinDeterminantPairEvaluation> build_same_spin_pair_cache(
    const std::vector<std::vector<int>>& unique_spin_determinants,
    const DeterminantPairEvaluator& pair_evaluator,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act,
    const std::vector<double>* dense_pair_kernel,
    bool materialize_projected_pair_values) {
  const int n_unique_determinants = static_cast<int>(unique_spin_determinants.size());
  std::vector<SpinDeterminantPairEvaluation> pair_cache(
      n_unique_determinants *
      n_unique_determinants);

  // Each cached entry is one exact same-spin determinant kernel for a specific
  // left/right orientation. The scalar same-spin matrix elements are symmetric,
  // but the overlap/cofactor payload is directional because rows belong to the
  // right determinant and columns to the left determinant.
  const ActiveSpaceTwoElectronView two_electron_view =
      make_active_space_two_electron_view(eri_act);
  const int n_threads =
      same_spin_pair_cache_thread_count(n_unique_determinants);

#pragma omp parallel for schedule(dynamic, 1) if(n_threads > 1) num_threads(n_threads)
  for (int left_index = 0; left_index < n_unique_determinants; ++left_index) {
    for (int right_index = 0; right_index < n_unique_determinants; ++right_index) {
      pair_cache[ordered_spin_pair_storage_index(
          left_index,
          right_index,
          n_unique_determinants)] =
          pair_evaluator.evaluate_same_spin_pair(
              unique_spin_determinants[left_index],
              unique_spin_determinants[right_index],
              ovlp_act,
              h1e_act,
              n_orbitals,
              eri_act);
      attach_opposite_spin_pair_cache(
          unique_spin_determinants[left_index],
          unique_spin_determinants[right_index],
          n_orbitals,
          two_electron_view,
          dense_pair_kernel,
          materialize_projected_pair_values,
          &pair_cache[ordered_spin_pair_storage_index(
              left_index,
              right_index,
              n_unique_determinants)]);
    }
  }

  return pair_cache;
}

std::vector<SpinDeterminantPairEvaluation> build_same_spin_pair_cache(
    const std::vector<std::vector<int>>& unique_spin_determinants,
    const DeterminantPairEvaluator& pair_evaluator,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const std::vector<double>* dense_pair_kernel,
    bool materialize_projected_pair_values) {
  const int n_unique_determinants = static_cast<int>(unique_spin_determinants.size());
  std::vector<SpinDeterminantPairEvaluation> pair_cache(
      n_unique_determinants *
      n_unique_determinants);
  const ActiveSpaceTwoElectronView two_electron_view =
      make_active_space_two_electron_view(active_space_two_electron_result);
  const int n_threads =
      same_spin_pair_cache_thread_count(n_unique_determinants);

#pragma omp parallel for schedule(dynamic, 1) if(n_threads > 1) num_threads(n_threads)
  for (int left_index = 0; left_index < n_unique_determinants; ++left_index) {
    for (int right_index = 0; right_index < n_unique_determinants; ++right_index) {
      pair_cache[ordered_spin_pair_storage_index(
          left_index,
          right_index,
          n_unique_determinants)] =
          pair_evaluator.evaluate_same_spin_pair(
              unique_spin_determinants[left_index],
              unique_spin_determinants[right_index],
              ovlp_act,
              h1e_act,
              n_orbitals,
              active_space_two_electron_result);
      attach_opposite_spin_pair_cache(
          unique_spin_determinants[left_index],
          unique_spin_determinants[right_index],
          n_orbitals,
          two_electron_view,
          dense_pair_kernel,
          materialize_projected_pair_values,
          &pair_cache[ordered_spin_pair_storage_index(
              left_index,
              right_index,
              n_unique_determinants)]);
    }
  }

  return pair_cache;
}

void populate_same_spin_phi_cache_entries(
    const std::vector<std::vector<int>>& unique_spin_determinants,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    std::vector<SpinDeterminantPairEvaluation>* pair_cache) {
  if (pair_cache == nullptr) {
    throw std::invalid_argument("pair_cache must not be null");
  }

  const int n_unique_determinants =
      static_cast<int>(unique_spin_determinants.size());
  if (pair_cache->size() !=
      n_unique_determinants *
          n_unique_determinants) {
    throw std::invalid_argument(
        "same-spin phi cache population size does not match unique-spin dimensions");
  }
  const int n_threads =
      same_spin_pair_cache_thread_count(n_unique_determinants);

#pragma omp parallel for schedule(dynamic, 1) if(n_threads > 1) num_threads(n_threads)
  for (int left_index = 0; left_index < n_unique_determinants; ++left_index) {
    for (int right_index = 0; right_index < n_unique_determinants; ++right_index) {
      auto& pair_evaluation = (*pair_cache)[ordered_spin_pair_storage_index(
          left_index,
          right_index,
          n_unique_determinants)];
      pair_evaluation.has_same_spin_phi_cache = false;
      pair_evaluation.same_spin_one_electron_phi = 0.0;
      pair_evaluation.same_spin_total_phi = 0.0;
      pair_evaluation.same_spin_inverse_overlap_gradient.resize(0, 0);
      pair_evaluation.same_spin_overlap_hamiltonian_gradient.resize(0, 0);

      if (pair_evaluation.overlap_result.nullity == 0 &&
          pair_evaluation.overlap_result.overlap_determinant != 0.0) {
        Eigen::MatrixXd inverse_overlap_gradient;
        const SameSpinPhiResult phi_result = compute_same_spin_original_phi(
            unique_spin_determinants[left_index],
            unique_spin_determinants[right_index],
            h1e_act,
            n_orbitals,
            active_space_two_electron_result,
            pair_evaluation.overlap_result,
            &inverse_overlap_gradient);
        pair_evaluation.has_same_spin_phi_cache = true;
        pair_evaluation.same_spin_one_electron_phi = phi_result.one_electron_phi;
        pair_evaluation.same_spin_total_phi = phi_result.total_phi;
        pair_evaluation.same_spin_inverse_overlap_gradient =
            std::move(inverse_overlap_gradient);
      }

      pair_evaluation.same_spin_overlap_hamiltonian_gradient =
          build_polynomial_same_spin_hamiltonian_overlap_gradient(
              unique_spin_determinants[left_index],
              unique_spin_determinants[right_index],
              h1e_act,
              n_orbitals,
              active_space_two_electron_result,
              &pair_evaluation);
    }
  }
}

}  // namespace

SpinDeterminantReuseTable build_spin_determinant_reuse_table(
    const std::vector<std::vector<int>>& spin_determinants) {
  SpinDeterminantReuseTable reuse_table;
  reuse_table.determinant_to_unique_id.resize(spin_determinants.size(), -1);
  std::unordered_map<std::vector<int>, int, SpinDeterminantHasher>
      determinant_to_unique_id;

  // the number of unique-determinants
  determinant_to_unique_id.reserve(spin_determinants.size());

  for (std::size_t determinant_index = 0;
       determinant_index < spin_determinants.size();
       ++determinant_index)
  {
    const auto [iterator, inserted] = determinant_to_unique_id.emplace(
        spin_determinants[determinant_index],
        static_cast<int>(reuse_table.unique_determinants.size())
    );
    
    if (inserted) {
      reuse_table.unique_determinants.push_back(spin_determinants[determinant_index]);
    }

    reuse_table.determinant_to_unique_id[determinant_index] = iterator->second;
  }

  return reuse_table;
}

std::size_t ordered_spin_pair_storage_index(
    int left_index,
    int right_index,
    int n_unique_determinants) {
  return left_index *
             n_unique_determinants +
      right_index;
}

std::size_t estimate_same_spin_pair_cache_bytes(
    const std::vector<std::vector<int>>& unique_spin_determinants,
    int n_orbitals) {
  int max_electron_count = 0;
  for (const auto& determinant : unique_spin_determinants) {
    max_electron_count =
        std::max(max_electron_count, static_cast<int>(determinant.size()));
  }
  const std::size_t n_packed_active_pairs =
      packed_active_pair_count(n_orbitals);
  const std::size_t max_occupied_pair_count =
      static_cast<std::size_t>(max_electron_count) *
      static_cast<std::size_t>(std::max(0, max_electron_count - 1)) / 2;

  // A cached same-spin pair keeps one `SpinDeterminantPairEvaluation`, plus
  // the dominant dynamic overlap payload:
  // - regular pairs: inverse overlap and first deleted-minor matrices
  // - singular pairs: singular values, full U/V factors, and first cofactors
  // - opposite-spin projected caches: sparse packed-pair regroupings plus the
  //   dense `G u` / `G x` images in active-pair space
  // - gradient reuse: cached same-spin `\partial \phi / \partial X`
  // The bound intentionally overestimates so the cache heuristic prefers the
  // direct path rather than risking a large allocation.
  const std::size_t pair_count =
      unique_spin_determinants.size() * unique_spin_determinants.size();
  const std::size_t per_pair_double_count =
      4ull * max_electron_count *
          max_electron_count +
      max_electron_count +
      2ull * n_packed_active_pairs +
      max_electron_count *
          max_electron_count +
      4ull * max_electron_count *
          max_electron_count +
      4ull * max_occupied_pair_count * max_occupied_pair_count;
  const std::size_t per_pair_int_count =
      4ull * max_electron_count *
      max_electron_count;
  return pair_count *
      (sizeof(SpinDeterminantPairEvaluation) +
       per_pair_double_count * sizeof(double) +
       per_pair_int_count * sizeof(int));
}

SameSpinPairCacheContext build_same_spin_pair_cache_context(
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    const DeterminantPairEvaluator& pair_evaluator,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act) {
  return build_same_spin_pair_cache_context(
      alpha_det,
      beta_det,
      pair_evaluator,
      ovlp_act,
      h1e_act,
      n_orbitals,
      eri_act,
      SameSpinPairCacheBuildOptions{});
}

SameSpinPairCacheContext build_same_spin_pair_cache_context(
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    const DeterminantPairEvaluator& pair_evaluator,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act,
    SameSpinPairCacheBuildOptions build_options) {
  if (alpha_det.size() != beta_det.size()) {
    throw std::invalid_argument("alpha_det and beta_det must have the same size");
  }

  SameSpinPairCacheContext cache_context;
  cache_context.alpha_reuse_table = build_spin_determinant_reuse_table(alpha_det);
  cache_context.close_shell_diagonal_reuses_same_spin_pair_cache =
      can_reuse_close_shell_same_spin_pair_cache(alpha_det, beta_det);
  cache_context.beta_reuses_alpha_pair_cache =
      cache_context.close_shell_diagonal_reuses_same_spin_pair_cache ||
      try_build_reuse_table_in_reference_basis(
          beta_det,
          cache_context.alpha_reuse_table,
          &cache_context.beta_reuse_table);
  // In determinant-wise close-shell cases the shared unique basis is diagonal.
  // More generally, singlet expansions may still span the same unique alpha
  // and beta occupied strings with a different determinant pairing. Those
  // cases can share the ordered same-spin cache even though later structure
  // coefficient blocks remain non-diagonal.
  if (cache_context.close_shell_diagonal_reuses_same_spin_pair_cache) {
    cache_context.beta_reuse_table =
        cache_context.alpha_reuse_table;
  } else if (!cache_context.beta_reuses_alpha_pair_cache) {
    cache_context.beta_reuse_table =
        build_spin_determinant_reuse_table(beta_det);
  }

  const int n_determinants = static_cast<int>(alpha_det.size());

  cache_context.cached_same_spin_evaluation_count =
      cache_context.alpha_reuse_table.unique_determinants.size() *
          cache_context.alpha_reuse_table.unique_determinants.size();
  if (!cache_context.beta_reuses_alpha_pair_cache) {
    cache_context.cached_same_spin_evaluation_count +=
        cache_context.beta_reuse_table.unique_determinants.size() *
        cache_context.beta_reuse_table.unique_determinants.size();
  }

  cache_context.estimated_cache_bytes =
      estimate_same_spin_pair_cache_bytes(
          cache_context.alpha_reuse_table.unique_determinants,
          n_orbitals);
  if (!cache_context.beta_reuses_alpha_pair_cache) {
    cache_context.estimated_cache_bytes +=
        estimate_same_spin_pair_cache_bytes(
            cache_context.beta_reuse_table.unique_determinants,
            n_orbitals);
  }
  // Production policy: always materialize the ordered same-spin determinant
  // kernels. Callers may still opt out of the dense opposite-spin projected
  // images when a streamed matrix-form path can rebuild only the touched
  // packed-pair rows on demand.
  cache_context.use_same_spin_pair_cache = (n_determinants > 0);

  std::vector<double> dense_pair_kernel;
  const std::vector<double>* dense_pair_kernel_ptr = nullptr;
  if (build_options.materialize_projected_pair_values) {
    dense_pair_kernel = build_dense_active_pair_kernel(eri_act, n_orbitals);
    dense_pair_kernel_ptr = &dense_pair_kernel;
  }
  cache_context.alpha_pair_cache = build_same_spin_pair_cache(
      cache_context.alpha_reuse_table.unique_determinants,
      pair_evaluator,
      ovlp_act,
      h1e_act,
      n_orbitals,
      eri_act,
      dense_pair_kernel_ptr,
      build_options.materialize_projected_pair_values);
  if (!cache_context.beta_reuses_alpha_pair_cache) {
    cache_context.beta_pair_cache = build_same_spin_pair_cache(
        cache_context.beta_reuse_table.unique_determinants,
        pair_evaluator,
        ovlp_act,
        h1e_act,
        n_orbitals,
        eri_act,
        dense_pair_kernel_ptr,
        build_options.materialize_projected_pair_values);
  } else {
    cache_context.beta_pair_cache.clear();
  }

  cache_context.n_unique_alpha = static_cast<int>(cache_context.alpha_reuse_table.unique_determinants.size());
  cache_context.n_unique_beta = static_cast<int>(cache_context.beta_reuse_table.unique_determinants.size());

  return cache_context;
}

SameSpinPairCacheContext build_same_spin_pair_cache_context(
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    const DeterminantPairEvaluator& pair_evaluator,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result) {
  return build_same_spin_pair_cache_context(
      alpha_det,
      beta_det,
      pair_evaluator,
      ovlp_act,
      h1e_act,
      n_orbitals,
      active_space_two_electron_result,
      SameSpinPairCacheBuildOptions{});
}

SameSpinPairCacheContext build_same_spin_pair_cache_context(
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    const DeterminantPairEvaluator& pair_evaluator,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    SameSpinPairCacheBuildOptions build_options) {
  if (alpha_det.size() != beta_det.size()) {
    throw std::invalid_argument("alpha_det and beta_det must have the same size");
  }

  SameSpinPairCacheContext cache_context;
  cache_context.alpha_reuse_table = build_spin_determinant_reuse_table(alpha_det);
  cache_context.close_shell_diagonal_reuses_same_spin_pair_cache =
      can_reuse_close_shell_same_spin_pair_cache(alpha_det, beta_det);
  cache_context.beta_reuses_alpha_pair_cache =
      cache_context.close_shell_diagonal_reuses_same_spin_pair_cache ||
      try_build_reuse_table_in_reference_basis(
          beta_det,
          cache_context.alpha_reuse_table,
          &cache_context.beta_reuse_table);
  // Reuse the alpha ordered same-spin table whenever the beta occupied
  // strings span the same unique determinant space. Determinant-wise diagonal
  // close-shell is just the strongest special case of this reuse.
  if (cache_context.close_shell_diagonal_reuses_same_spin_pair_cache) {
    cache_context.beta_reuse_table =
        cache_context.alpha_reuse_table;
  } else if (!cache_context.beta_reuses_alpha_pair_cache) {
    cache_context.beta_reuse_table =
        build_spin_determinant_reuse_table(beta_det);
  }

  const int n_determinants = static_cast<int>(alpha_det.size());
  cache_context.cached_same_spin_evaluation_count =
      cache_context.alpha_reuse_table.unique_determinants.size() *
          cache_context.alpha_reuse_table.unique_determinants.size();
  if (!cache_context.beta_reuses_alpha_pair_cache) {
    cache_context.cached_same_spin_evaluation_count +=
        cache_context.beta_reuse_table.unique_determinants.size() *
        cache_context.beta_reuse_table.unique_determinants.size();
  }
  cache_context.estimated_cache_bytes =
      estimate_same_spin_pair_cache_bytes(
          cache_context.alpha_reuse_table.unique_determinants,
          n_orbitals);
  if (!cache_context.beta_reuses_alpha_pair_cache) {
    cache_context.estimated_cache_bytes +=
        estimate_same_spin_pair_cache_bytes(
            cache_context.beta_reuse_table.unique_determinants,
            n_orbitals);
  }
  // Production policy: always materialize the ordered same-spin determinant
  // kernels. Callers may still opt out of the dense opposite-spin projected
  // images when a streamed matrix-form path can rebuild only the touched
  // packed-pair rows on demand.
  cache_context.use_same_spin_pair_cache = (n_determinants > 0);

  const std::size_t expected_packed_size =
      packed_active_two_electron_integral_count(n_orbitals);

  if (!active_space_two_electron_result.packed_active_two_electron_integrals.empty() &&
      active_space_two_electron_result.packed_active_two_electron_integrals.size() ==
          expected_packed_size) {
    // Materialized packed integrals available — use the packed-exact cache
    // builder which constructs a cache-friendly dense_pair_kernel for the
    // opposite-spin projection and avoids view-dispatch indirection.
    const auto& eri_act =
        active_space_two_electron_result.packed_active_two_electron_integrals;
    std::vector<double> dense_pair_kernel;
    const std::vector<double>* dense_pair_kernel_ptr = nullptr;
    if (build_options.materialize_projected_pair_values) {
      dense_pair_kernel = build_dense_active_pair_kernel(eri_act, n_orbitals);
      dense_pair_kernel_ptr = &dense_pair_kernel;
    }

    cache_context.alpha_pair_cache = build_same_spin_pair_cache(
        cache_context.alpha_reuse_table.unique_determinants,
        pair_evaluator,
        ovlp_act,
        h1e_act,
        n_orbitals,
        eri_act,
        dense_pair_kernel_ptr,
        build_options.materialize_projected_pair_values);

    if (!cache_context.beta_reuses_alpha_pair_cache) {
      cache_context.beta_pair_cache = build_same_spin_pair_cache(
          cache_context.beta_reuse_table.unique_determinants,
          pair_evaluator,
          ovlp_act,
          h1e_act,
          n_orbitals,
          eri_act,
          dense_pair_kernel_ptr,
          build_options.materialize_projected_pair_values);
    } else {
      cache_context.beta_pair_cache.clear();
    }
  } else {
    // Direct RI representation — no materialized packed integrals.
    // Streamed callers can skip the dense `G u` / `G x` images entirely and
    // keep only the sparse packed-pair coefficients in the cache.
    std::vector<double> reconstructed_eri;
    std::vector<double> dense_pair_kernel;
    const std::vector<double>* dense_pair_kernel_ptr = nullptr;
    if (build_options.materialize_projected_pair_values) {
      reconstructed_eri =
          reconstruct_packed_active_two_electron_integrals(
              make_active_space_two_electron_view(active_space_two_electron_result),
              n_orbitals);
      dense_pair_kernel = build_dense_active_pair_kernel(reconstructed_eri, n_orbitals);
      dense_pair_kernel_ptr = &dense_pair_kernel;
    }

    cache_context.alpha_pair_cache = build_same_spin_pair_cache(
        cache_context.alpha_reuse_table.unique_determinants,
        pair_evaluator,
        ovlp_act,
        h1e_act,
        n_orbitals,
        active_space_two_electron_result,
        dense_pair_kernel_ptr,
        build_options.materialize_projected_pair_values);

    if (!cache_context.beta_reuses_alpha_pair_cache) {
      cache_context.beta_pair_cache = build_same_spin_pair_cache(
          cache_context.beta_reuse_table.unique_determinants,
          pair_evaluator,
          ovlp_act,
          h1e_act,
          n_orbitals,
          active_space_two_electron_result,
          dense_pair_kernel_ptr,
          build_options.materialize_projected_pair_values);
    } else {
      cache_context.beta_pair_cache.clear();
    }
  }

  cache_context.n_unique_alpha = static_cast<int>(cache_context.alpha_reuse_table.unique_determinants.size());
  cache_context.n_unique_beta = static_cast<int>(cache_context.beta_reuse_table.unique_determinants.size());

  return cache_context;
}

void populate_same_spin_phi_cache(
    SameSpinPairCacheContext* same_spin_pair_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result) {
  if (same_spin_pair_cache == nullptr) {
    throw std::invalid_argument("same_spin_pair_cache must not be null");
  }
  if (!same_spin_pair_cache->enabled()) {
    return;
  }

  // The matrix-form same-spin backward reuses the accepted-point same-spin
  // pair kernels many times. Regular pairs cache `phi` and
  // `\partial \phi / \partial X^{-1}`, while singular pairs cache the exact
  // occupied-block overlap gradient of the same-spin Hamiltonian directly.
  // Both cases keep the later matrix-form adjoint sweep on the ordered
  // unique-spin grid and avoid repeated determinant-level integral lookup.
  populate_same_spin_phi_cache_entries(
      same_spin_pair_cache->alpha_reuse_table.unique_determinants,
      h1e_act,
      n_orbitals,
      active_space_two_electron_result,
      same_spin_pair_cache->mutable_alpha_pair_cache_ref());
  if (!same_spin_pair_cache->shares_same_spin_pair_cache_between_spins()) {
    populate_same_spin_phi_cache_entries(
        same_spin_pair_cache->beta_reuse_table.unique_determinants,
        h1e_act,
        n_orbitals,
        active_space_two_electron_result,
        same_spin_pair_cache->mutable_beta_pair_cache_ref());
  }
}

DeterminantPairEvaluation evaluate_full_determinant_pair_with_optional_same_spin_cache(
    const SameSpinPairCacheContext* same_spin_pair_cache,
    const DeterminantPairEvaluator& pair_evaluator,
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    int determinant_index_left,
    int determinant_index_right,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act,
    bool retain_spin_pair_evaluations) {

  if (same_spin_pair_cache != nullptr && same_spin_pair_cache->enabled()) {
    const auto& alpha_occ_L =
        alpha_det[determinant_index_left];
    const auto& alpha_occ_R =
        alpha_det[determinant_index_right];
    const auto& beta_occ_L =
        beta_det[determinant_index_left];
    const auto& beta_occ_R =
        beta_det[determinant_index_right];
    const int alpha_left_id =
        same_spin_pair_cache->alpha_reuse_table.determinant_to_unique_id[
            determinant_index_left];
    const int alpha_right_id =
        same_spin_pair_cache->alpha_reuse_table.determinant_to_unique_id[
            determinant_index_right];
    const int beta_left_id =
        same_spin_pair_cache->beta_reuse_table.determinant_to_unique_id[
            determinant_index_left];
    const int beta_right_id =
        same_spin_pair_cache->beta_reuse_table.determinant_to_unique_id[
            determinant_index_right];
    const auto& alpha_pair_result =
        same_spin_pair_cache->alpha_pair_cache[ordered_spin_pair_storage_index(
            alpha_left_id,
            alpha_right_id,
            static_cast<int>(same_spin_pair_cache->alpha_reuse_table.unique_determinants.size()))];
    const auto& beta_pair_result =
        same_spin_pair_cache->beta_pair_cache_ref()[ordered_spin_pair_storage_index(
            beta_left_id,
            beta_right_id,
            static_cast<int>(same_spin_pair_cache->beta_reuse_table.unique_determinants.size()))];

    double opposite_spin_coupling = 0.0;
    if (has_materialized_opposite_spin_scalar_cache(*same_spin_pair_cache)) {
      const std::size_t opp_cache_idx = same_spin_pair_cache->opposite_spin_cache_index(
          alpha_left_id, alpha_right_id, beta_left_id, beta_right_id);
      if (!same_spin_pair_cache->opposite_spin_cache_computed[opp_cache_idx]) {
        const double opp_coupling = evaluate_opposite_spin_coulomb_coupling(
            alpha_occ_L, alpha_occ_R, alpha_pair_result,
            beta_occ_L, beta_occ_R, beta_pair_result, eri_act);
        // Note: This is not thread-safe, but redundant writes are acceptable.
        const_cast<SameSpinPairCacheContext*>(same_spin_pair_cache)->opposite_spin_cache[opp_cache_idx] =
            opp_coupling;
        const_cast<SameSpinPairCacheContext*>(same_spin_pair_cache)
            ->opposite_spin_cache_computed[opp_cache_idx] = true;
      }
      opposite_spin_coupling =
          same_spin_pair_cache->opposite_spin_cache[opp_cache_idx];
    } else {
      opposite_spin_coupling = evaluate_opposite_spin_coulomb_coupling(
          alpha_occ_L, alpha_occ_R, alpha_pair_result,
          beta_occ_L, beta_occ_R, beta_pair_result, eri_act);
    }

    // Combine results using cached opposite-spin coupling
    DeterminantPairEvaluation result;
    if (retain_spin_pair_evaluations) {
      result.alpha = alpha_pair_result;
      result.beta = beta_pair_result;
    }
    result.overlap_determinant =
        alpha_pair_result.overlap_result.overlap_determinant *
        beta_pair_result.overlap_result.overlap_determinant;
    result.one_electron_hamiltonian =
        alpha_pair_result.one_electron_hamiltonian *
            beta_pair_result.overlap_result.overlap_determinant +
        beta_pair_result.one_electron_hamiltonian *
            alpha_pair_result.overlap_result.overlap_determinant;
    result.total_hamiltonian =
        alpha_pair_result.total_hamiltonian *
            beta_pair_result.overlap_result.overlap_determinant +
        beta_pair_result.total_hamiltonian *
            alpha_pair_result.overlap_result.overlap_determinant +
        opposite_spin_coupling;

    return result;
  }

  return pair_evaluator.evaluate(
      alpha_det[determinant_index_left],
      alpha_det[determinant_index_right],
      beta_det[determinant_index_left],
      beta_det[determinant_index_right],
      ovlp_act,
      h1e_act,
      n_orbitals,
      eri_act,
      retain_spin_pair_evaluations);
}

DeterminantPairEvaluation evaluate_full_determinant_pair_with_optional_same_spin_cache(
    const SameSpinPairCacheContext* same_spin_pair_cache,
    const DeterminantPairEvaluator& pair_evaluator,
    const std::vector<std::vector<int>>& alpha_det,
    const std::vector<std::vector<int>>& beta_det,
    int determinant_index_left,
    int determinant_index_right,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    bool retain_spin_pair_evaluations) {
  if (same_spin_pair_cache != nullptr && same_spin_pair_cache->enabled()) {
    const auto& alpha_occ_L =
        alpha_det[determinant_index_left];
    const auto& alpha_occ_R =
        alpha_det[determinant_index_right];
    const auto& beta_occ_L =
        beta_det[determinant_index_left];
    const auto& beta_occ_R =
        beta_det[determinant_index_right];
    const int alpha_left_id =
        same_spin_pair_cache->alpha_reuse_table.determinant_to_unique_id[
            determinant_index_left];
    const int alpha_right_id =
        same_spin_pair_cache->alpha_reuse_table.determinant_to_unique_id[
            determinant_index_right];
    const int beta_left_id =
        same_spin_pair_cache->beta_reuse_table.determinant_to_unique_id[
            determinant_index_left];
    const int beta_right_id =
        same_spin_pair_cache->beta_reuse_table.determinant_to_unique_id[
            determinant_index_right];
    const auto& alpha_pair_result =
        same_spin_pair_cache->alpha_pair_cache[ordered_spin_pair_storage_index(
            alpha_left_id,
            alpha_right_id,
            static_cast<int>(same_spin_pair_cache->alpha_reuse_table.unique_determinants.size()))];
    const auto& beta_pair_result =
        same_spin_pair_cache->beta_pair_cache_ref()[ordered_spin_pair_storage_index(
            beta_left_id,
            beta_right_id,
            static_cast<int>(same_spin_pair_cache->beta_reuse_table.unique_determinants.size()))];

    double opposite_spin_coupling = 0.0;
    if (has_materialized_opposite_spin_scalar_cache(*same_spin_pair_cache)) {
      const std::size_t opp_cache_idx = same_spin_pair_cache->opposite_spin_cache_index(
          alpha_left_id, alpha_right_id, beta_left_id, beta_right_id);
      if (!same_spin_pair_cache->opposite_spin_cache_computed[opp_cache_idx]) {
        const double opp_coupling =
            !active_space_two_electron_result.packed_active_two_electron_integrals.empty()
            ? evaluate_opposite_spin_coulomb_coupling(
                  alpha_occ_L, alpha_occ_R, alpha_pair_result,
                  beta_occ_L, beta_occ_R, beta_pair_result,
                  active_space_two_electron_result.packed_active_two_electron_integrals)
            : evaluate_opposite_spin_coulomb_coupling(
                  alpha_occ_L, alpha_occ_R, alpha_pair_result,
                  beta_occ_L, beta_occ_R, beta_pair_result,
                  reconstruct_packed_active_two_electron_integrals(
                      make_active_space_two_electron_view(active_space_two_electron_result),
                      n_orbitals));
        const_cast<SameSpinPairCacheContext*>(same_spin_pair_cache)->opposite_spin_cache[opp_cache_idx] =
            opp_coupling;
        const_cast<SameSpinPairCacheContext*>(same_spin_pair_cache)
            ->opposite_spin_cache_computed[opp_cache_idx] = true;
      }
      opposite_spin_coupling =
          same_spin_pair_cache->opposite_spin_cache[opp_cache_idx];
    } else {
      opposite_spin_coupling =
          !active_space_two_electron_result.packed_active_two_electron_integrals.empty()
          ? evaluate_opposite_spin_coulomb_coupling(
                alpha_occ_L, alpha_occ_R, alpha_pair_result,
                beta_occ_L, beta_occ_R, beta_pair_result,
                active_space_two_electron_result.packed_active_two_electron_integrals)
          : evaluate_opposite_spin_coulomb_coupling(
                alpha_occ_L, alpha_occ_R, alpha_pair_result,
                beta_occ_L, beta_occ_R, beta_pair_result,
                reconstruct_packed_active_two_electron_integrals(
                    make_active_space_two_electron_view(active_space_two_electron_result),
                    n_orbitals));
    }

    // Combine results using cached opposite-spin coupling
    DeterminantPairEvaluation result;
    if (retain_spin_pair_evaluations) {
      result.alpha = alpha_pair_result;
      result.beta = beta_pair_result;
    }
    result.overlap_determinant =
        alpha_pair_result.overlap_result.overlap_determinant *
        beta_pair_result.overlap_result.overlap_determinant;
    result.one_electron_hamiltonian =
        alpha_pair_result.one_electron_hamiltonian *
            beta_pair_result.overlap_result.overlap_determinant +
        beta_pair_result.one_electron_hamiltonian *
            alpha_pair_result.overlap_result.overlap_determinant;
    result.total_hamiltonian =
        alpha_pair_result.total_hamiltonian *
            beta_pair_result.overlap_result.overlap_determinant +
        beta_pair_result.total_hamiltonian *
            alpha_pair_result.overlap_result.overlap_determinant +
        opposite_spin_coupling;

    return result;
  }

  return pair_evaluator.evaluate(
      alpha_det[determinant_index_left],
      alpha_det[determinant_index_right],
      beta_det[determinant_index_left],
      beta_det[determinant_index_right],
      ovlp_act,
      h1e_act,
      n_orbitals,
      active_space_two_electron_result,
      retain_spin_pair_evaluations);
}

}  // namespace xmvb::vb
