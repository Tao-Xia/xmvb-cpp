#include "vbscf/structures/assembly/action.hpp"

#include <algorithm>
#include <stdexcept>

#include "core/openmp.hpp"
#include "vbscf/determinants/pairs/storage.hpp"

namespace xmvb::vb {
namespace {

double contract_opposite_spin(
    const OppositeSpinPackedPairProjection& alpha,
    const OppositeSpinPackedPairProjection& beta,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals,
    int n_packed_pairs) {
  if (alpha.packed_pair_indices.empty() ||
      beta.packed_pair_indices.empty()) {
    return 0.0;
  }
  const bool alpha_projected =
      static_cast<int>(alpha.projected_pair_values.size()) == n_packed_pairs;
  const bool beta_projected =
      static_cast<int>(beta.projected_pair_values.size()) == n_packed_pairs;
  double value = 0.0;
  if (beta_projected &&
      (!alpha_projected ||
       alpha.packed_pair_indices.size() <= beta.packed_pair_indices.size())) {
    for (std::size_t entry = 0;
         entry < alpha.packed_pair_indices.size();
         ++entry) {
      value += alpha.packed_pair_values[entry] *
          beta.projected_pair_values[alpha.packed_pair_indices[entry]];
    }
  } else if (alpha_projected) {
    for (std::size_t entry = 0;
         entry < beta.packed_pair_indices.size();
         ++entry) {
      value += beta.packed_pair_values[entry] *
          alpha.projected_pair_values[beta.packed_pair_indices[entry]];
    }
  } else {
    for (std::size_t alpha_entry = 0;
         alpha_entry < alpha.packed_pair_indices.size();
         ++alpha_entry) {
      for (std::size_t beta_entry = 0;
           beta_entry < beta.packed_pair_indices.size();
           ++beta_entry) {
        value +=
            alpha.packed_pair_values[alpha_entry] *
            lookup_active_space_two_electron_kernel_value(
                two_electron_view,
                alpha.packed_pair_indices[alpha_entry],
                beta.packed_pair_indices[beta_entry],
                n_active_orbitals) *
            beta.packed_pair_values[beta_entry];
      }
    }
  }
  return value;
}

struct DeterminantPairScalars {
  double hamiltonian = 0.0;
  double overlap = 0.0;
};

DeterminantPairScalars evaluate_pair(
    const SameSpinPairCacheContext& cache,
    int left_determinant,
    int right_determinant,
    const ActiveSpaceTwoElectronView& two_electron_view,
    int n_active_orbitals,
    int n_packed_pairs) {
  const int alpha_left =
      cache.alpha_reuse_table.determinant_to_unique_id[left_determinant];
  const int alpha_right =
      cache.alpha_reuse_table.determinant_to_unique_id[right_determinant];
  const int beta_left =
      cache.beta_reuse_table.determinant_to_unique_id[left_determinant];
  const int beta_right =
      cache.beta_reuse_table.determinant_to_unique_id[right_determinant];

  const auto& alpha = cache.alpha_pair_cache_ref()[
      ordered_spin_pair_storage_index(
          alpha_left,
          alpha_right,
          static_cast<int>(
              cache.alpha_reuse_table.unique_determinants.size()))];
  const auto& beta = cache.beta_pair_cache_ref()[
      ordered_spin_pair_storage_index(
          beta_left,
          beta_right,
          static_cast<int>(
              cache.beta_reuse_table.unique_determinants.size()))];

  const double alpha_overlap = alpha.overlap_result.overlap_determinant;
  const double beta_overlap = beta.overlap_result.overlap_determinant;
  DeterminantPairScalars result;
  result.overlap = alpha_overlap * beta_overlap;
  result.hamiltonian =
      alpha.total_hamiltonian * beta_overlap +
      alpha_overlap * beta.total_hamiltonian +
      contract_opposite_spin(
          alpha.opposite_spin_pair_cache.first_order_cofactor_projection,
          beta.opposite_spin_pair_cache.first_order_cofactor_projection,
          two_electron_view,
          n_active_orbitals,
          n_packed_pairs);
  return result;
}

}  // namespace

StructureAction::StructureAction(
    const std::vector<std::vector<StructureExpansionTerm>>&
        determinant_to_structure_terms,
    int n_structures,
    const SameSpinPairCacheContext& same_spin_pair_cache,
    const ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    int n_active_orbitals)
    : same_spin_pair_cache_(&same_spin_pair_cache),
      two_electron_view_(
          make_active_space_two_electron_view(
              active_space_two_electron_result)),
      determinant_to_structure_terms_(&determinant_to_structure_terms),
      n_determinants_(
          static_cast<int>(determinant_to_structure_terms.size())),
      n_structures_(n_structures),
      n_packed_pairs_(packed_active_pair_count(n_active_orbitals)),
      n_active_orbitals_(n_active_orbitals) {
  if (n_determinants_ <= 0 || n_structures_ <= 0) {
    throw std::invalid_argument(
        "matrix-free structure action requires positive dimensions");
  }
  if (!same_spin_pair_cache.enabled()) {
    throw std::invalid_argument(
        "matrix-free structure action requires a same-spin pair cache");
  }
  if (static_cast<int>(
          same_spin_pair_cache.alpha_reuse_table
              .determinant_to_unique_id.size()) != n_determinants_ ||
      static_cast<int>(
          same_spin_pair_cache.beta_reuse_table
              .determinant_to_unique_id.size()) != n_determinants_) {
    throw std::invalid_argument(
        "same-spin pair cache does not match determinant expansion");
  }

  const auto validate_pair_cache = [this](
      const std::vector<SpinDeterminantPairEvaluation>& pair_cache,
      int n_unique) {
    if (pair_cache.size() !=
        static_cast<std::size_t>(n_unique) * n_unique) {
      throw std::invalid_argument(
          "ordered same-spin pair cache has incompatible dimensions");
    }
    for (const auto& pair : pair_cache) {
      const auto& opposite_spin = pair.opposite_spin_pair_cache;
      if (opposite_spin.n_packed_active_pairs != n_packed_pairs_) {
        throw std::invalid_argument(
            "same-spin pair caches use inconsistent packed-pair dimensions");
      }
      const auto& projection =
          opposite_spin.first_order_cofactor_projection;
      if (projection.packed_pair_indices.size() !=
          projection.packed_pair_values.size()) {
        throw std::invalid_argument(
            "packed-pair projection indices and values differ in size");
      }
      if (!projection.projected_pair_values.empty() &&
          static_cast<int>(projection.projected_pair_values.size()) !=
              n_packed_pairs_) {
        throw std::invalid_argument(
            "projected packed-pair vector has incompatible dimensions");
      }
      for (const int packed_pair : projection.packed_pair_indices) {
        if (packed_pair < 0 || packed_pair >= n_packed_pairs_) {
          throw std::out_of_range(
              "packed-pair projection index is out of range");
        }
      }
    }
  };
  validate_pair_cache(
      same_spin_pair_cache.alpha_pair_cache_ref(),
      static_cast<int>(
          same_spin_pair_cache.alpha_reuse_table.unique_determinants.size()));
  if (!same_spin_pair_cache.shares_same_spin_pair_cache_between_spins()) {
    validate_pair_cache(
        same_spin_pair_cache.beta_pair_cache_ref(),
        static_cast<int>(
            same_spin_pair_cache.beta_reuse_table.unique_determinants.size()));
  }

  structure_to_determinant_terms_.resize(n_structures_);
  for (int determinant = 0;
       determinant < n_determinants_;
       ++determinant) {
    for (const auto& term : determinant_to_structure_terms[determinant]) {
      if (term.structure_index < 0 ||
          term.structure_index >= n_structures_) {
        throw std::out_of_range(
            "determinant expansion structure index is out of range");
      }
      structure_to_determinant_terms_[term.structure_index].push_back(
          DeterminantTerm{determinant, term.coefficient});
    }
  }
}

StructureActionResult StructureAction::apply(
    const Eigen::Ref<const Eigen::MatrixXd>& vectors) const {
  if (vectors.rows() != n_structures_ || vectors.cols() <= 0) {
    throw std::invalid_argument(
        "structure vector block has incompatible dimensions");
  }

  const int block_width = static_cast<int>(vectors.cols());
  Eigen::MatrixXd determinant_vectors =
      Eigen::MatrixXd::Zero(n_determinants_, block_width);
  for (int determinant = 0;
       determinant < n_determinants_;
       ++determinant) {
    for (const auto& term :
         (*determinant_to_structure_terms_)[determinant]) {
      determinant_vectors.row(determinant).noalias() +=
          term.coefficient * vectors.row(term.structure_index);
    }
  }

  Eigen::MatrixXd determinant_hamiltonian =
      Eigen::MatrixXd::Zero(n_determinants_, block_width);
  Eigen::MatrixXd determinant_overlap =
      Eigen::MatrixXd::Zero(n_determinants_, block_width);
  const int n_threads = std::max(
      1,
      std::min(
          xmvb::effective_openmp_thread_count(),
          n_determinants_));

  StructureActionResult result;
  result.hamiltonian =
      Eigen::MatrixXd::Zero(n_structures_, block_width);
  result.overlap =
      Eigen::MatrixXd::Zero(n_structures_, block_width);

#pragma omp parallel if(n_threads > 1) num_threads(n_threads)
  {
#pragma omp for schedule(static)
    for (int left_determinant = 0;
         left_determinant < n_determinants_;
         ++left_determinant) {
      for (int right_determinant = 0;
           right_determinant < n_determinants_;
           ++right_determinant) {
        const DeterminantPairScalars pair = evaluate_pair(
            *same_spin_pair_cache_,
            left_determinant,
            right_determinant,
            two_electron_view_,
            n_active_orbitals_,
            n_packed_pairs_);
        for (int vector = 0; vector < block_width; ++vector) {
          const double coefficient =
              determinant_vectors(right_determinant, vector);
          determinant_hamiltonian(left_determinant, vector) +=
              pair.hamiltonian * coefficient;
          determinant_overlap(left_determinant, vector) +=
              pair.overlap * coefficient;
        }
      }
    }

#pragma omp for schedule(static)
    for (int structure = 0; structure < n_structures_; ++structure) {
      for (const auto& term :
           structure_to_determinant_terms_[structure]) {
        result.hamiltonian.row(structure).noalias() +=
            term.coefficient *
            determinant_hamiltonian.row(term.determinant);
        result.overlap.row(structure).noalias() +=
            term.coefficient *
            determinant_overlap.row(term.determinant);
      }
    }
  }
  return result;
}

StructureDiagonal StructureAction::diagonal() const {
  StructureDiagonal result;
  result.hamiltonian = Eigen::VectorXd::Zero(n_structures_);
  result.overlap = Eigen::VectorXd::Zero(n_structures_);
  const int n_threads = std::max(
      1,
      std::min(
          xmvb::effective_openmp_thread_count(),
          n_structures_));

#pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
  for (int structure = 0; structure < n_structures_; ++structure) {
    const auto& terms = structure_to_determinant_terms_[structure];
    double hamiltonian = 0.0;
    double overlap = 0.0;
    for (const auto& left : terms) {
      for (const auto& right : terms) {
        const DeterminantPairScalars pair = evaluate_pair(
            *same_spin_pair_cache_,
            left.determinant,
            right.determinant,
            two_electron_view_,
            n_active_orbitals_,
            n_packed_pairs_);
        const double expansion_coefficient =
            left.coefficient * right.coefficient;
        hamiltonian += expansion_coefficient * pair.hamiltonian;
        overlap += expansion_coefficient * pair.overlap;
      }
    }
    result.hamiltonian[structure] = hamiltonian;
    result.overlap[structure] = overlap;
  }
  return result;
}

int StructureAction::n_determinants() const noexcept {
  return n_determinants_;
}

int StructureAction::n_structures() const noexcept {
  return n_structures_;
}

}  // namespace xmvb::vb
