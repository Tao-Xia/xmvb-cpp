#include "vb/biorthogonal_vbscf/biorthogonal_exact_selected_structure.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_determinant_hamiltonian.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_prepared_input.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_structure_local_utils.hpp"
#include "vb/biorthogonal_vbscf/biorthogonal_spin_pair_tiles.hpp"
#include "vb/matrices/structure_block_kernels.hpp"
#include "vb/matrices/structure_coefficient_blocks.hpp"
#include "vb/matrices/two_electron_indexer.hpp"
#include "vb/orbital/active_space_two_electron_utils.hpp"

namespace xmvb::vb::biorthogonal_vbscf {

namespace {

void validate_shape(
    const Eigen::MatrixXd& matrix,
    int expected_rows,
    int expected_columns,
    const char* label) {
  if (matrix.rows() != expected_rows || matrix.cols() != expected_columns) {
    throw std::invalid_argument(std::string(label) + " dimensions are inconsistent");
  }
}

std::vector<double> dense_matrix_to_vector(
    const Eigen::MatrixXd& matrix) {
  return std::vector<double>(
      matrix.data(),
      matrix.data() + matrix.size());
}

double compute_max_abs_symmetry_residual(
    const Eigen::MatrixXd& matrix) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("matrix must be square");
  }
  if (matrix.size() == 0) {
    return 0.0;
  }
  return (matrix - matrix.transpose()).cwiseAbs().maxCoeff();
}

void validate_selected_structure_indices(
    int n_structures,
    const std::vector<int>& selected_structure_indices) {
  if (n_structures <= 0) {
    throw std::invalid_argument("n_structures must be positive");
  }
  if (selected_structure_indices.empty()) {
    throw std::invalid_argument("selected_structure_indices must not be empty");
  }

  std::vector<bool> seen(xmvb::to_size(n_structures), false);
  for (const int structure_index : selected_structure_indices) {
    if (structure_index < 0 || structure_index >= n_structures) {
      throw std::out_of_range("selected structure index is out of range");
    }
    if (seen[xmvb::to_size(structure_index)]) {
      throw std::invalid_argument("selected_structure_indices must be unique");
    }
    seen[xmvb::to_size(structure_index)] = true;
  }
}

struct SelectedBiorthogonalStructureExpansion {
  std::vector<BiorthogonalDeterminant> determinants;
  Eigen::MatrixXd selected_structure_to_determinant;
  std::vector<int> determinant_to_unique_alpha_id;
  std::vector<int> determinant_to_unique_beta_id;
  std::vector<StructureCoefficientBlock> selected_structure_blocks;
  SpinDeterminantReuseTable alpha_reuse_table;
  SpinDeterminantReuseTable beta_reuse_table;
};

struct DeterminantActionTileEntry {
  int determinant_index = 0;
  int alpha_local = 0;
  int beta_local = 0;
};

struct SelectedStructureContractionLookup {
  int n_unique_alpha = 0;
  int n_unique_beta = 0;
  std::vector<int> determinant_index_by_unique_pair;
};

std::vector<int> build_selected_structure_lookup(
    int n_structures,
    const std::vector<int>& selected_structure_indices) {
  std::vector<int> structure_to_selected_column(
      xmvb::to_size(n_structures),
      -1);
  for (int selected_column = 0;
       selected_column < static_cast<int>(selected_structure_indices.size());
       ++selected_column) {
    structure_to_selected_column[xmvb::to_size(
        selected_structure_indices[xmvb::to_size(selected_column)])] =
        selected_column;
  }
  return structure_to_selected_column;
}

std::vector<int> build_contiguous_index_range(
    int begin,
    int end) {
  if (begin < 0 || end < begin) {
    throw std::invalid_argument("index range must satisfy 0 <= begin <= end");
  }
  std::vector<int> indices(xmvb::to_size(end - begin));
  for (int index = begin; index < end; ++index) {
    indices[xmvb::to_size(index - begin)] = index;
  }
  return indices;
}

std::size_t determinant_action_tile_linear_index(
    int alpha_tile,
    int beta_tile,
    int n_beta_tiles) {
  if (alpha_tile < 0 || beta_tile < 0 || n_beta_tiles <= 0) {
    throw std::invalid_argument("determinant action tile index is invalid");
  }
  return xmvb::to_size(alpha_tile) * xmvb::to_size(n_beta_tiles) +
      xmvb::to_size(beta_tile);
}

void scatter_action_block_to_determinant_vector(
    const Eigen::MatrixXd& action_block,
    const std::vector<DeterminantActionTileEntry>& tile_entries,
    Eigen::VectorXd* action_vector) {
  if (action_vector == nullptr) {
    throw std::invalid_argument("action_vector must not be null");
  }
  for (const auto& tile_entry : tile_entries) {
    (*action_vector)(tile_entry.determinant_index) =
        action_block(tile_entry.alpha_local, tile_entry.beta_local);
  }
}

SelectedBiorthogonalStructureExpansion build_selected_structure_expansion(
    const xmvb::vb::FullDeterminantStructureData& structure_data,
    const std::vector<int>& selected_structure_indices,
    bool materialize_dense_selected_structure_to_determinant) {
  if (structure_data.n_structures <= 0) {
    throw std::invalid_argument("structure_data.n_structures must be positive");
  }
  if (structure_data.n_active_orbitals <= 0) {
    throw std::invalid_argument("structure_data.n_active_orbitals must be positive");
  }
  if (structure_data.alpha_det.size() != structure_data.beta_det.size() ||
      structure_data.alpha_det.size() !=
          structure_data.determinant_to_structure_terms.size()) {
    throw std::invalid_argument("full determinant structure arrays are inconsistent");
  }
  if (structure_data.alpha_det.empty()) {
    throw std::invalid_argument("at least one determinant is required");
  }

  const int n_determinants = static_cast<int>(structure_data.alpha_det.size());
  const int n_selected_structures =
      static_cast<int>(selected_structure_indices.size());
  const std::vector<int> structure_to_selected_column =
      build_selected_structure_lookup(
          structure_data.n_structures,
          selected_structure_indices);

  SelectedBiorthogonalStructureExpansion expansion;
  expansion.alpha_reuse_table =
      build_spin_determinant_reuse_table(structure_data.alpha_det);
  expansion.beta_reuse_table =
      build_spin_determinant_reuse_table(structure_data.beta_det);
  expansion.determinants.reserve(xmvb::to_size(n_determinants));
  if (materialize_dense_selected_structure_to_determinant) {
    expansion.selected_structure_to_determinant =
        Eigen::MatrixXd::Zero(n_determinants, n_selected_structures);
  }
  expansion.determinant_to_unique_alpha_id =
      expansion.alpha_reuse_table.determinant_to_unique_id;
  expansion.determinant_to_unique_beta_id =
      expansion.beta_reuse_table.determinant_to_unique_id;

  for (int determinant_index = 0;
       determinant_index < n_determinants;
       ++determinant_index) {
    BiorthogonalDeterminant determinant{
        structure_data.alpha_det[xmvb::to_size(determinant_index)],
        structure_data.beta_det[xmvb::to_size(determinant_index)],
    };
    validate_biorthogonal_determinant(
        determinant,
        structure_data.n_active_orbitals);
    expansion.determinants.push_back(std::move(determinant));
    const int unique_alpha_id =
        expansion.determinant_to_unique_alpha_id[xmvb::to_size(determinant_index)];
    const int unique_beta_id =
        expansion.determinant_to_unique_beta_id[xmvb::to_size(determinant_index)];

    // This is the exact selected-column extraction of `T_sel`: each full
    // determinant row accumulates only the requested structure amplitudes,
    // while repeated `(determinant, structure)` terms are summed algebraically.
    for (const auto& term :
         structure_data.determinant_to_structure_terms[xmvb::to_size(
             determinant_index)]) {
      if (term.structure_index < 0 ||
          term.structure_index >= structure_data.n_structures) {
        throw std::out_of_range("structure expansion term index is out of range");
      }
      if (!std::isfinite(term.coefficient)) {
        throw std::runtime_error("structure expansion coefficient is not finite");
      }
      const int selected_column =
          structure_to_selected_column[xmvb::to_size(term.structure_index)];
      if (selected_column < 0) {
        continue;
      }
      if (materialize_dense_selected_structure_to_determinant) {
        expansion.selected_structure_to_determinant(
            determinant_index,
            selected_column) += term.coefficient;
      }
    }
  }
  const auto coefficient_blocks =
      build_structure_coefficient_blocks(
          structure_data.determinant_to_structure_terms,
          structure_to_selected_column,
          n_selected_structures,
          expansion.alpha_reuse_table,
          expansion.beta_reuse_table,
          true);
  expansion.selected_structure_blocks = coefficient_blocks;

  if (materialize_dense_selected_structure_to_determinant) {
    throw_if_nonfinite(
        expansion.selected_structure_to_determinant,
        "selected_structure_to_determinant");
  }
  for (const auto& structure_block : expansion.selected_structure_blocks) {
    throw_if_nonfinite(
        structure_block.local_coefficients,
        "selected_structure_block.local_coefficients");
  }
  return expansion;
}

int selected_structure_count(
    const SelectedBiorthogonalStructureExpansion& expansion) {
  return static_cast<int>(expansion.selected_structure_blocks.size());
}

double contract_selected_structure_with_action_block(
    const StructureCoefficientBlock& left_block,
    const Eigen::MatrixXd& action_block) {
  if (left_block.local_coefficients.size() == 0) {
    return 0.0;
  }
  return xmvb::vb::dense_frobenius_inner_product(
      left_block.local_coefficients,
      action_block);
}

SelectedStructureContractionLookup build_selected_structure_contraction_lookup(
    const SelectedBiorthogonalStructureExpansion& selected_structure_expansion) {
  SelectedStructureContractionLookup lookup;
  lookup.n_unique_alpha = static_cast<int>(
      selected_structure_expansion.alpha_reuse_table.unique_determinants.size());
  lookup.n_unique_beta = static_cast<int>(
      selected_structure_expansion.beta_reuse_table.unique_determinants.size());
  lookup.determinant_index_by_unique_pair.assign(
      xmvb::product_size(lookup.n_unique_alpha, lookup.n_unique_beta),
      -1);

  for (int determinant_index = 0;
       determinant_index < static_cast<int>(
           selected_structure_expansion.determinant_to_unique_alpha_id.size());
       ++determinant_index) {
    const int unique_alpha_id =
        selected_structure_expansion.determinant_to_unique_alpha_id[xmvb::to_size(
            determinant_index)];
    const int unique_beta_id =
        selected_structure_expansion.determinant_to_unique_beta_id[xmvb::to_size(
            determinant_index)];
    if (unique_alpha_id < 0 || unique_alpha_id >= lookup.n_unique_alpha ||
        unique_beta_id < 0 || unique_beta_id >= lookup.n_unique_beta) {
      throw std::out_of_range("determinant-to-unique spin id is out of range");
    }
    const std::size_t linear_index =
        xmvb::col_major_index(
            unique_alpha_id,
            unique_beta_id,
            lookup.n_unique_alpha);
    int& stored_determinant_index =
        xmvb::index_at(lookup.determinant_index_by_unique_pair, linear_index);
    if (stored_determinant_index >= 0) {
      throw std::logic_error(
          "full determinant topology contains duplicate unique spin-string pairs");
    }
    stored_determinant_index = determinant_index;
  }
  return lookup;
}

double contract_selected_structure_with_action_column(
    const StructureCoefficientBlock& left_block,
    const SelectedStructureContractionLookup& contraction_lookup,
    const Eigen::Ref<const Eigen::VectorXd>& action_column) {
  if (left_block.local_coefficients.size() == 0) {
    return 0.0;
  }
  double contraction = 0.0;
  for (int beta_local = 0;
       beta_local < static_cast<int>(left_block.beta_support.size());
       ++beta_local) {
    const int beta_index =
        left_block.beta_support[xmvb::to_size(beta_local)];
    if (beta_index < 0 || beta_index >= contraction_lookup.n_unique_beta) {
      throw std::out_of_range("left structure beta support index is out of range");
    }
    for (int alpha_local = 0;
         alpha_local < static_cast<int>(left_block.alpha_support.size());
         ++alpha_local) {
      const int alpha_index =
          left_block.alpha_support[xmvb::to_size(alpha_local)];
      if (alpha_index < 0 || alpha_index >= contraction_lookup.n_unique_alpha) {
        throw std::out_of_range("left structure alpha support index is out of range");
      }
      const std::size_t determinant_lookup_index =
          xmvb::col_major_index(
              alpha_index,
              beta_index,
              contraction_lookup.n_unique_alpha);
      const int determinant_index =
          xmvb::index_at(
              contraction_lookup.determinant_index_by_unique_pair,
              determinant_lookup_index);
      if (determinant_index < 0) {
        continue;
      }
      contraction +=
          left_block.local_coefficients(alpha_local, beta_local) *
          action_column(determinant_index);
    }
  }
  return contraction;
}

void contract_selected_structure_action_column(
    const SelectedBiorthogonalStructureExpansion& selected_structure_expansion,
    const SelectedStructureContractionLookup& contraction_lookup,
    const Eigen::Ref<const Eigen::VectorXd>& action_column,
    Eigen::VectorXd* physical_structure_column) {
  if (physical_structure_column == nullptr) {
    throw std::invalid_argument("physical_structure_column must not be null");
  }
  const int n_selected_structures =
      selected_structure_count(selected_structure_expansion);
  if (action_column.size() != static_cast<int>(
                                selected_structure_expansion.determinants.size())) {
    throw std::invalid_argument(
        "action_column size does not match the full determinant topology");
  }
  *physical_structure_column = Eigen::VectorXd::Zero(n_selected_structures);
  for (int left_structure = 0;
       left_structure < n_selected_structures;
       ++left_structure) {
    const StructureCoefficientBlock& left_block =
        selected_structure_expansion.selected_structure_blocks[xmvb::to_size(
            left_structure)];
    (*physical_structure_column)(left_structure) =
        contract_selected_structure_with_action_column(
            left_block,
            contraction_lookup,
            action_column);
  }
}

struct SelectedStructureActionContext {
  int n_unique_alpha = 0;
  int n_unique_beta = 0;
  int action_tile_size = 0;
  int n_alpha_tiles = 0;
  int n_beta_tiles = 0;
  std::vector<std::vector<int>> alpha_row_indices_by_tile;
  std::vector<std::vector<int>> beta_row_indices_by_tile;
  std::vector<std::vector<DeterminantActionTileEntry>>
      determinant_entries_by_tile;
};

struct SelectedStructureActionThreadContext {
  NonorthogonalOverlapSpinPairTileProvider alpha_nonorthogonal_overlap_provider;
  NonorthogonalOverlapSpinPairTileProvider beta_nonorthogonal_overlap_provider;
  BiorthogonalForwardSpinPairTileProvider alpha_biorthogonal_provider;
  BiorthogonalForwardSpinPairTileProvider beta_biorthogonal_provider;
};

SelectedStructureActionContext build_selected_structure_action_context(
    const SelectedBiorthogonalStructureExpansion& selected_structure_expansion) {
  SelectedStructureActionContext action_context;
  action_context.n_unique_alpha = static_cast<int>(
      selected_structure_expansion.alpha_reuse_table.unique_determinants.size());
  action_context.n_unique_beta = static_cast<int>(
      selected_structure_expansion.beta_reuse_table.unique_determinants.size());
  action_context.action_tile_size = structure_matrix_tile_size();
  action_context.n_alpha_tiles =
      (action_context.n_unique_alpha + action_context.action_tile_size - 1) /
      action_context.action_tile_size;
  action_context.n_beta_tiles =
      (action_context.n_unique_beta + action_context.action_tile_size - 1) /
      action_context.action_tile_size;
  action_context.alpha_row_indices_by_tile.resize(
      xmvb::to_size(action_context.n_alpha_tiles));
  for (int alpha_tile = 0;
       alpha_tile < action_context.n_alpha_tiles;
       ++alpha_tile) {
    const int alpha_begin = alpha_tile * action_context.action_tile_size;
    const int alpha_end = std::min(
        action_context.n_unique_alpha,
        alpha_begin + action_context.action_tile_size);
    xmvb::index_at(
        action_context.alpha_row_indices_by_tile,
        alpha_tile) = build_contiguous_index_range(alpha_begin, alpha_end);
  }
  action_context.beta_row_indices_by_tile.resize(
      xmvb::to_size(action_context.n_beta_tiles));
  for (int beta_tile = 0;
       beta_tile < action_context.n_beta_tiles;
       ++beta_tile) {
    const int beta_begin = beta_tile * action_context.action_tile_size;
    const int beta_end = std::min(
        action_context.n_unique_beta,
        beta_begin + action_context.action_tile_size);
    xmvb::index_at(
        action_context.beta_row_indices_by_tile,
        beta_tile) = build_contiguous_index_range(beta_begin, beta_end);
  }
  action_context.determinant_entries_by_tile.resize(
      xmvb::to_size(action_context.n_alpha_tiles) *
      xmvb::to_size(action_context.n_beta_tiles));
  for (int determinant_index = 0;
       determinant_index < static_cast<int>(
           selected_structure_expansion.determinant_to_unique_alpha_id.size());
       ++determinant_index) {
    const int unique_alpha_id =
        selected_structure_expansion.determinant_to_unique_alpha_id[xmvb::to_size(
            determinant_index)];
    const int unique_beta_id =
        selected_structure_expansion.determinant_to_unique_beta_id[xmvb::to_size(
            determinant_index)];
    if (unique_alpha_id < 0 || unique_alpha_id >= action_context.n_unique_alpha ||
        unique_beta_id < 0 || unique_beta_id >= action_context.n_unique_beta) {
      throw std::out_of_range("determinant-to-unique spin id is out of range");
    }
    const int alpha_tile = unique_alpha_id / action_context.action_tile_size;
    const int beta_tile = unique_beta_id / action_context.action_tile_size;
    xmvb::index_at(
        action_context.determinant_entries_by_tile,
        determinant_action_tile_linear_index(
            alpha_tile,
            beta_tile,
            action_context.n_beta_tiles)).push_back(
                DeterminantActionTileEntry{
                    determinant_index,
                    unique_alpha_id - alpha_tile * action_context.action_tile_size,
                    unique_beta_id - beta_tile * action_context.action_tile_size});
  }
  return action_context;
}

int parallel_task_thread_count(
    int n_tasks) {
  if (n_tasks <= 1) {
    return 1;
  }

  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  if (n_threads > n_tasks) {
    n_threads = n_tasks;
  }
  if (n_threads < 1) {
    n_threads = 1;
  }
  return n_threads;
}

void capture_parallel_exception(
    std::atomic<bool>* failed,
    std::exception_ptr* first_exception) {
  if (failed == nullptr || first_exception == nullptr) {
    throw std::invalid_argument("parallel failure outputs must not be null");
  }
#pragma omp critical
  {
    if (!failed->load(std::memory_order_relaxed)) {
      *first_exception = std::current_exception();
      failed->store(true, std::memory_order_relaxed);
    }
  }
}

SelectedStructureActionThreadContext build_selected_structure_action_thread_context(
    const SelectedBiorthogonalStructureExpansion& selected_structure_expansion,
    const std::vector<double>& ovlp_act,
    int n_active_orbitals,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view) {
  // Each thread needs its own nonorthogonal/biorthogonal tile providers
  // because the cached unique-spin tiles are mutable LRU state keyed by the
  // current block walk.
  return SelectedStructureActionThreadContext{
      NonorthogonalOverlapSpinPairTileProvider(
          selected_structure_expansion.alpha_reuse_table.unique_determinants,
          ovlp_act,
          n_active_orbitals),
      NonorthogonalOverlapSpinPairTileProvider(
          selected_structure_expansion.beta_reuse_table.unique_determinants,
          ovlp_act,
          n_active_orbitals),
      BiorthogonalForwardSpinPairTileProvider(
          selected_structure_expansion.alpha_reuse_table.unique_determinants,
          true,
          orbital_integrals,
          right_right_two_electron_view),
      BiorthogonalForwardSpinPairTileProvider(
          selected_structure_expansion.beta_reuse_table.unique_determinants,
          false,
          orbital_integrals,
          right_right_two_electron_view)};
}

void build_selected_structure_action_block(
    const xmvb::vb::FullDeterminantStructureData& full_structure_data,
    const SelectedBiorthogonalStructureExpansion& selected_structure_expansion,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view,
    const SelectedStructureActionContext& action_context,
    const NonorthogonalOverlapSpinPairTileProvider& alpha_nonorthogonal_overlap_provider,
    const NonorthogonalOverlapSpinPairTileProvider& beta_nonorthogonal_overlap_provider,
    const BiorthogonalForwardSpinPairTileProvider& alpha_biorthogonal_provider,
    const BiorthogonalForwardSpinPairTileProvider& beta_biorthogonal_provider,
    const std::vector<int>& left_alpha_indices,
    const std::vector<int>& left_beta_indices,
    int selected_structure,
    Eigen::MatrixXd* overlap_block,
    Eigen::MatrixXd* hamiltonian_block) {
  if (overlap_block == nullptr && hamiltonian_block == nullptr) {
    throw std::invalid_argument(
        "at least one selected-structure action block output is required");
  }

  const StructureCoefficientBlock& right_block =
      selected_structure_expansion.selected_structure_blocks[xmvb::to_size(
          selected_structure)];
  const Eigen::MatrixXd& right_coefficients = right_block.local_coefficients;
  if (right_coefficients.size() == 0) {
    if (overlap_block != nullptr) {
      *overlap_block = Eigen::MatrixXd::Zero(
          static_cast<int>(left_alpha_indices.size()),
          static_cast<int>(left_beta_indices.size()));
    }
    if (hamiltonian_block != nullptr) {
      *hamiltonian_block = Eigen::MatrixXd::Zero(
          static_cast<int>(left_alpha_indices.size()),
          static_cast<int>(left_beta_indices.size()));
    }
    return;
  }

  Eigen::MatrixXd beta_push;
  Eigen::MatrixXd scratch_image;
  Eigen::MatrixXd beta_projected_channel_block;
  if (overlap_block != nullptr) {
    Eigen::MatrixXd alpha_overlap_block;
    Eigen::MatrixXd beta_overlap_block;
    gather_nonorthogonal_same_spin_overlap_block(
        alpha_nonorthogonal_overlap_provider,
        left_alpha_indices,
        right_block.alpha_support,
        &alpha_overlap_block);
    gather_nonorthogonal_same_spin_overlap_block(
        beta_nonorthogonal_overlap_provider,
        left_beta_indices,
        right_block.beta_support,
        &beta_overlap_block);
    build_structure_action_image(
        right_coefficients,
        alpha_overlap_block,
        beta_overlap_block,
        &beta_push,
        overlap_block);
  }

  if (hamiltonian_block != nullptr) {
    Eigen::MatrixXd alpha_biorthogonal_overlap_block;
    Eigen::MatrixXd alpha_biorthogonal_total_block;
    Eigen::MatrixXd beta_biorthogonal_overlap_block;
    Eigen::MatrixXd beta_biorthogonal_total_block;
    LocalSpinProjectionBlock alpha_projection_block;
    LocalSpinProjectionBlock beta_projection_block;
    gather_biorthogonal_forward_spin_block(
        alpha_biorthogonal_provider,
        left_alpha_indices,
        right_block.alpha_support,
        &alpha_biorthogonal_overlap_block,
        nullptr,
        &alpha_biorthogonal_total_block,
        &alpha_projection_block);
    gather_biorthogonal_forward_spin_block(
        beta_biorthogonal_provider,
        left_beta_indices,
        right_block.beta_support,
        &beta_biorthogonal_overlap_block,
        nullptr,
        &beta_biorthogonal_total_block,
        &beta_projection_block);
    build_structure_action_image(
        right_coefficients,
        alpha_biorthogonal_total_block,
        beta_biorthogonal_overlap_block,
        &beta_push,
        hamiltonian_block);
    build_structure_action_image(
        right_coefficients,
        alpha_biorthogonal_overlap_block,
        beta_biorthogonal_total_block,
        &beta_push,
        &scratch_image);
    *hamiltonian_block += scratch_image;
    add_local_opposite_spin_action_image(
        right_coefficients,
        alpha_projection_block,
        beta_projection_block,
        right_right_two_electron_view,
        full_structure_data.n_active_orbitals,
        &beta_projected_channel_block,
        &beta_push,
        &scratch_image,
        hamiltonian_block);
  }
}

void accumulate_selected_structure_action_column(
    const xmvb::vb::FullDeterminantStructureData& full_structure_data,
    const SelectedBiorthogonalStructureExpansion& selected_structure_expansion,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view,
    const SelectedStructureActionContext& action_context,
    const NonorthogonalOverlapSpinPairTileProvider& alpha_nonorthogonal_overlap_provider,
    const NonorthogonalOverlapSpinPairTileProvider& beta_nonorthogonal_overlap_provider,
    const BiorthogonalForwardSpinPairTileProvider& alpha_biorthogonal_provider,
    const BiorthogonalForwardSpinPairTileProvider& beta_biorthogonal_provider,
    int selected_structure,
    Eigen::VectorXd* overlap_action_column,
    Eigen::VectorXd* hamiltonian_action_column) {
  if (overlap_action_column == nullptr && hamiltonian_action_column == nullptr) {
    throw std::invalid_argument(
        "at least one selected-structure action column output is required");
  }

  const int n_determinants =
      static_cast<int>(selected_structure_expansion.determinants.size());
  if (overlap_action_column != nullptr) {
    *overlap_action_column = Eigen::VectorXd::Zero(n_determinants);
  }
  if (hamiltonian_action_column != nullptr) {
    *hamiltonian_action_column = Eigen::VectorXd::Zero(n_determinants);
  }

  const StructureCoefficientBlock& right_block =
      selected_structure_expansion.selected_structure_blocks[xmvb::to_size(
          selected_structure)];
  if (right_block.local_coefficients.size() == 0) {
    return;
  }

  // Stream the determinant action one left unique-pair tile at a time. Each
  // local tile has the same algebra as the full rectangular image
  // `K_alpha(left,right) * C(right) * K_beta(left,right)^T`, but we only build
  // the tiles that actually contain determinant rows.
  Eigen::MatrixXd overlap_action_tile;
  Eigen::MatrixXd hamiltonian_action_tile;
  for (int alpha_tile = 0;
       alpha_tile < action_context.n_alpha_tiles;
       ++alpha_tile) {
    const auto& left_alpha_indices = xmvb::index_at(
        action_context.alpha_row_indices_by_tile,
        alpha_tile);
    for (int beta_tile = 0;
         beta_tile < action_context.n_beta_tiles;
         ++beta_tile) {
      const auto& tile_entries = xmvb::index_at(
          action_context.determinant_entries_by_tile,
          determinant_action_tile_linear_index(
              alpha_tile,
              beta_tile,
              action_context.n_beta_tiles));
      if (tile_entries.empty()) {
        continue;
      }

      const auto& left_beta_indices = xmvb::index_at(
          action_context.beta_row_indices_by_tile,
          beta_tile);
      build_selected_structure_action_block(
          full_structure_data,
          selected_structure_expansion,
          right_right_two_electron_view,
          action_context,
          alpha_nonorthogonal_overlap_provider,
          beta_nonorthogonal_overlap_provider,
          alpha_biorthogonal_provider,
          beta_biorthogonal_provider,
          left_alpha_indices,
          left_beta_indices,
          selected_structure,
          overlap_action_column != nullptr ? &overlap_action_tile : nullptr,
          hamiltonian_action_column != nullptr ? &hamiltonian_action_tile : nullptr);
      if (overlap_action_column != nullptr) {
        scatter_action_block_to_determinant_vector(
            overlap_action_tile,
            tile_entries,
            overlap_action_column);
      }
      if (hamiltonian_action_column != nullptr) {
        scatter_action_block_to_determinant_vector(
            hamiltonian_action_tile,
            tile_entries,
            hamiltonian_action_column);
      }
    }
  }
}

void accumulate_selected_structure_actions(
    const xmvb::vb::FullDeterminantStructureData& full_structure_data,
    const SelectedBiorthogonalStructureExpansion& selected_structure_expansion,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view,
    Eigen::MatrixXd* overlap_action_on_selected_columns,
    Eigen::MatrixXd* biorthogonal_hamiltonian_action_on_selected_columns) {
  if (overlap_action_on_selected_columns == nullptr ||
      biorthogonal_hamiltonian_action_on_selected_columns == nullptr) {
    throw std::invalid_argument("action output matrices must not be null");
  }
  if (selected_structure_expansion.determinants.empty()) {
    throw std::invalid_argument("determinants must not be empty");
  }
  validate_shape(
      selected_structure_expansion.selected_structure_to_determinant,
      static_cast<int>(selected_structure_expansion.determinants.size()),
      selected_structure_expansion.selected_structure_to_determinant.cols(),
      "selected_structure_to_determinant");

  const int n_determinants =
      static_cast<int>(selected_structure_expansion.determinants.size());
  const int n_selected_structures =
      selected_structure_count(selected_structure_expansion);
  *overlap_action_on_selected_columns =
      Eigen::MatrixXd::Zero(n_determinants, n_selected_structures);
  *biorthogonal_hamiltonian_action_on_selected_columns =
      Eigen::MatrixXd::Zero(n_determinants, n_selected_structures);

  const SelectedStructureActionContext action_context =
      build_selected_structure_action_context(
          selected_structure_expansion);
  const int n_threads = parallel_task_thread_count(n_selected_structures);
  std::atomic<bool> failed(false);
  std::exception_ptr first_exception;

  // The selected columns are independent exact actions. Parallelizing over
  // columns keeps the full-determinant topology intact while letting each
  // thread reuse its own biorthogonal unique-spin tile cache.
#pragma omp parallel num_threads(n_threads) if(n_selected_structures > 1)
  {
    const SelectedStructureActionThreadContext thread_context =
        build_selected_structure_action_thread_context(
            selected_structure_expansion,
            full_structure_data.ovlp_act,
            full_structure_data.n_active_orbitals,
            orbital_integrals,
            right_right_two_electron_view);
    Eigen::VectorXd overlap_action_column;
    Eigen::VectorXd hamiltonian_action_column;

#pragma omp for schedule(dynamic)
    for (int selected_structure = 0;
         selected_structure < n_selected_structures;
         ++selected_structure) {
      if (failed.load(std::memory_order_relaxed)) {
        continue;
      }
      try {
        accumulate_selected_structure_action_column(
            full_structure_data,
            selected_structure_expansion,
            right_right_two_electron_view,
            action_context,
            thread_context.alpha_nonorthogonal_overlap_provider,
            thread_context.beta_nonorthogonal_overlap_provider,
            thread_context.alpha_biorthogonal_provider,
            thread_context.beta_biorthogonal_provider,
            selected_structure,
            &overlap_action_column,
            &hamiltonian_action_column);
        overlap_action_on_selected_columns->col(selected_structure) =
            overlap_action_column;
        biorthogonal_hamiltonian_action_on_selected_columns->col(selected_structure) =
            hamiltonian_action_column;
      } catch (...) {
        capture_parallel_exception(&failed, &first_exception);
      }
    }
  }
  if (first_exception) {
    std::rethrow_exception(first_exception);
  }

  throw_if_nonfinite(
      *overlap_action_on_selected_columns,
      "overlap_action_on_selected_columns");
  throw_if_nonfinite(
      *biorthogonal_hamiltonian_action_on_selected_columns,
      "biorthogonal_hamiltonian_action_on_selected_columns");
}

}  // namespace

BiorthogonalExactSelectedStructureMatrixBuildResult
build_biorthogonal_exact_selected_structure_matrices(
    const xmvb::vb::FullDeterminantStructureData& full_structure_data,
    const std::vector<int>& selected_structure_indices,
    double symmetry_tolerance) {
  if (symmetry_tolerance < 0.0) {
    throw std::invalid_argument("symmetry_tolerance must be non-negative");
  }
  validate_selected_structure_indices(
      full_structure_data.n_structures,
      selected_structure_indices);

  BiorthogonalExactSelectedStructureMatrixBuildResult result;
  result.n_active_orbitals = full_structure_data.n_active_orbitals;
  result.n_full_structures = full_structure_data.n_structures;
  result.n_selected_structures =
      static_cast<int>(selected_structure_indices.size());
  result.selected_structure_indices = selected_structure_indices;

  const SelectedBiorthogonalStructureExpansion selected_structure_expansion =
      build_selected_structure_expansion(
          full_structure_data,
          selected_structure_indices,
          false);
  result.n_determinants =
      static_cast<int>(selected_structure_expansion.determinants.size());

  const auto orbital_integrals = build_biorthogonal_orbital_integrals(
      full_structure_data.n_active_orbitals,
      full_structure_data.ovlp_act,
      full_structure_data.h1e_act);
  const ActiveSpaceTwoElectronView right_right_two_electron_view =
      make_active_space_two_electron_view(full_structure_data.eri_act);
  const SelectedStructureActionContext action_context =
      build_selected_structure_action_context(
          selected_structure_expansion);
  const SelectedStructureContractionLookup contraction_lookup =
      build_selected_structure_contraction_lookup(
          selected_structure_expansion);
  const int n_selected_structures =
      selected_structure_count(selected_structure_expansion);
  const int n_threads = parallel_task_thread_count(n_selected_structures);
  std::atomic<bool> failed(false);
  std::exception_ptr first_exception;
  Eigen::MatrixXd overlap_action_on_selected_columns =
      Eigen::MatrixXd::Zero(result.n_determinants, n_selected_structures);
  result.physical_structure_overlap =
      Eigen::MatrixXd::Zero(n_selected_structures, n_selected_structures);
  result.physical_structure_hamiltonian =
      Eigen::MatrixXd::Zero(n_selected_structures, n_selected_structures);

  // The exact selected-space matrix build has two column-wise passes. Reusing
  // one thread-local provider per worker keeps the tile caches warm across the
  // overlap and Hamiltonian passes without sharing mutable cache state.
#pragma omp parallel num_threads(n_threads) if(n_selected_structures > 1)
  {
    const SelectedStructureActionThreadContext thread_context =
        build_selected_structure_action_thread_context(
            selected_structure_expansion,
            full_structure_data.ovlp_act,
            full_structure_data.n_active_orbitals,
            orbital_integrals,
            right_right_two_electron_view);
    Eigen::VectorXd thread_overlap_action_column;
    Eigen::VectorXd thread_hamiltonian_action_column;
    Eigen::VectorXd thread_physical_column;

    // Pass 1 keeps only `U_sel = S_det T_sel` in determinant-indexed form.
    // That is enough to build the physical overlap immediately and to reuse the
    // same stored left action later when contracting `H_str = U_sel^T Y_sel`.
#pragma omp for schedule(dynamic)
    for (int selected_structure = 0;
         selected_structure < n_selected_structures;
         ++selected_structure) {
      if (failed.load(std::memory_order_relaxed)) {
        continue;
      }
      try {
        accumulate_selected_structure_action_column(
            full_structure_data,
            selected_structure_expansion,
            right_right_two_electron_view,
            action_context,
            thread_context.alpha_nonorthogonal_overlap_provider,
            thread_context.beta_nonorthogonal_overlap_provider,
            thread_context.alpha_biorthogonal_provider,
            thread_context.beta_biorthogonal_provider,
            selected_structure,
            &thread_overlap_action_column,
            nullptr);
        overlap_action_on_selected_columns.col(selected_structure) =
            thread_overlap_action_column;
        contract_selected_structure_action_column(
            selected_structure_expansion,
            contraction_lookup,
            thread_overlap_action_column,
            &thread_physical_column);
        result.physical_structure_overlap.col(selected_structure) =
            thread_physical_column;
      } catch (...) {
        capture_parallel_exception(&failed, &first_exception);
      }
    }

#pragma omp for schedule(dynamic)
    for (int selected_structure = 0;
         selected_structure < n_selected_structures;
         ++selected_structure) {
      if (failed.load(std::memory_order_relaxed)) {
        continue;
      }
      try {
        // Pass 2 streams one Hamiltonian action column `Y_j` at a time and
        // contracts it against the retained `U_sel`, which avoids storing the
        // full dense `Y_sel` matrix for matrix-only callers.
        accumulate_selected_structure_action_column(
            full_structure_data,
            selected_structure_expansion,
            right_right_two_electron_view,
            action_context,
            thread_context.alpha_nonorthogonal_overlap_provider,
            thread_context.beta_nonorthogonal_overlap_provider,
            thread_context.alpha_biorthogonal_provider,
            thread_context.beta_biorthogonal_provider,
            selected_structure,
            nullptr,
            &thread_hamiltonian_action_column);
        result.physical_structure_hamiltonian.col(selected_structure).noalias() =
            overlap_action_on_selected_columns.transpose() *
            thread_hamiltonian_action_column;
      } catch (...) {
        capture_parallel_exception(&failed, &first_exception);
      }
    }
  }
  if (first_exception) {
    std::rethrow_exception(first_exception);
  }

  throw_if_nonfinite(
      overlap_action_on_selected_columns,
      "overlap_action_on_selected_columns");
  throw_if_nonfinite(
      result.physical_structure_overlap,
      "physical_structure_overlap");

  throw_if_nonfinite(
      result.physical_structure_hamiltonian,
      "physical_structure_hamiltonian");

  result.structure_overlap_symmetry_residual_max_abs =
      compute_max_abs_symmetry_residual(result.physical_structure_overlap);
  result.structure_hamiltonian_symmetry_residual_max_abs =
      compute_max_abs_symmetry_residual(result.physical_structure_hamiltonian);
  result.physical_structure_overlap =
      0.5 * (result.physical_structure_overlap +
             result.physical_structure_overlap.transpose());
  result.physical_structure_hamiltonian =
      0.5 * (result.physical_structure_hamiltonian +
             result.physical_structure_hamiltonian.transpose());

  validate_biorthogonal_exact_selected_structure_matrix_build_result(
      result,
      symmetry_tolerance);
  return result;
}

BiorthogonalExactSelectedStructureMatrixBuildResult
build_biorthogonal_exact_selected_structure_matrices(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::PreparedActiveSpaceContext& prepared_active_space,
    const std::vector<int>& selected_structure_indices,
    double symmetry_tolerance) {
  return build_biorthogonal_exact_selected_structure_matrices(
      build_biorthogonal_full_structure_data(input, prepared_active_space),
      selected_structure_indices,
      symmetry_tolerance);
}

BiorthogonalExactSelectedStructureMatrixBuildResult
build_biorthogonal_exact_selected_structure_matrices(
    const xmvb::vb::CppVbInput& input,
    const std::vector<int>& selected_structure_indices,
    double symmetry_tolerance) {
  const PreparedBiorthogonalInput prepared_input =
      prepare_biorthogonal_input(input);
  return build_biorthogonal_exact_selected_structure_matrices(
      prepared_input.structure_data,
      selected_structure_indices,
      symmetry_tolerance);
}

BiorthogonalExactSelectedStructureEvaluationResult
evaluate_biorthogonal_exact_selected_structure_subspace(
    const xmvb::vb::FullDeterminantStructureData& full_structure_data,
    const std::vector<int>& selected_structure_indices,
    double symmetry_tolerance) {
  if (symmetry_tolerance < 0.0) {
    throw std::invalid_argument("symmetry_tolerance must be non-negative");
  }
  validate_selected_structure_indices(
      full_structure_data.n_structures,
      selected_structure_indices);

  BiorthogonalExactSelectedStructureEvaluationResult result;
  result.n_active_orbitals = full_structure_data.n_active_orbitals;
  result.n_full_structures = full_structure_data.n_structures;
  result.n_selected_structures =
      static_cast<int>(selected_structure_indices.size());
  result.selected_structure_indices = selected_structure_indices;

  const SelectedBiorthogonalStructureExpansion selected_structure_expansion =
      build_selected_structure_expansion(
          full_structure_data,
          selected_structure_indices,
          true);
  const SelectedStructureContractionLookup contraction_lookup =
      build_selected_structure_contraction_lookup(
          selected_structure_expansion);
  result.n_determinants =
      static_cast<int>(selected_structure_expansion.determinants.size());

  // The exact selected-space formulation keeps the full determinant basis and
  // only restricts the right structure columns. `T_sel` therefore has one row
  // for every full determinant and one column for every selected structure.
  result.selected_structure_to_determinant =
      selected_structure_expansion.selected_structure_to_determinant;

  const auto orbital_integrals = build_biorthogonal_orbital_integrals(
      full_structure_data.n_active_orbitals,
      full_structure_data.ovlp_act,
      full_structure_data.h1e_act);
  const ActiveSpaceTwoElectronView right_right_two_electron_view =
      make_active_space_two_electron_view(full_structure_data.eri_act);

  // The exact projection only needs the two full-space actions on `T_sel`.
  // Applying them directly keeps the algebra identical to `S_det T_sel` and
  // `h_det^(bi) T_sel` while avoiding dense `N_det x N_det` materialization.
  accumulate_selected_structure_actions(
      full_structure_data,
      selected_structure_expansion,
      orbital_integrals,
      right_right_two_electron_view,
      &result.overlap_action_on_selected_columns,
      &result.biorthogonal_hamiltonian_action_on_selected_columns);

  Eigen::MatrixXd raw_physical_structure_overlap =
      Eigen::MatrixXd::Zero(
          result.n_selected_structures,
          result.n_selected_structures);
  const int n_threads = parallel_task_thread_count(result.n_selected_structures);
  std::atomic<bool> failed(false);
  std::exception_ptr first_exception;
#pragma omp parallel num_threads(n_threads) if(result.n_selected_structures > 1)
  {
    Eigen::VectorXd thread_physical_column;
#pragma omp for schedule(dynamic)
    for (int selected_structure = 0;
         selected_structure < result.n_selected_structures;
         ++selected_structure) {
      if (failed.load(std::memory_order_relaxed)) {
        continue;
      }
      try {
        contract_selected_structure_action_column(
            selected_structure_expansion,
            contraction_lookup,
            result.overlap_action_on_selected_columns.col(selected_structure),
            &thread_physical_column);
        raw_physical_structure_overlap.col(selected_structure) =
            thread_physical_column;
      } catch (...) {
        capture_parallel_exception(&failed, &first_exception);
      }
    }
  }
  if (first_exception) {
    std::rethrow_exception(first_exception);
  }
  const Eigen::MatrixXd raw_physical_structure_hamiltonian =
      result.overlap_action_on_selected_columns.transpose() *
      result.biorthogonal_hamiltonian_action_on_selected_columns;

  result.structure_overlap_symmetry_residual_max_abs =
      compute_max_abs_symmetry_residual(raw_physical_structure_overlap);
  result.structure_hamiltonian_symmetry_residual_max_abs =
      compute_max_abs_symmetry_residual(raw_physical_structure_hamiltonian);

  // The exact projected matrices are symmetric in exact arithmetic. We
  // symmetrize the final small selected-space matrices before calling the
  // Hermitian-definite generalized eigensolver so the physical pencil is
  // preserved while roundoff asymmetry is removed.
  result.physical_structure_overlap =
      0.5 * (raw_physical_structure_overlap +
             raw_physical_structure_overlap.transpose());
  result.physical_structure_hamiltonian =
      0.5 * (raw_physical_structure_hamiltonian +
             raw_physical_structure_hamiltonian.transpose());

  const xmvb::core::GeneralizedEigensolver generalized_eigensolver;
  const xmvb::core::GeneralizedEigenResult eigen_result =
      generalized_eigensolver.solve(
          dense_matrix_to_vector(result.physical_structure_hamiltonian),
          dense_matrix_to_vector(result.physical_structure_overlap),
          result.n_selected_structures);
  result.eigenvalues = eigen_result.eigenvalues;
  const Eigen::Map<const Eigen::MatrixXd> mapped_structure_coefficients(
      eigen_result.eigenvector_matrix.data(),
      result.n_selected_structures,
      result.n_selected_structures);
  result.structure_coefficient_matrix = mapped_structure_coefficients;
  result.right_determinant_coefficient_matrix.resize(0, 0);
  result.left_determinant_coefficient_matrix.resize(0, 0);

  validate_biorthogonal_exact_selected_structure_evaluation_result(
      result,
      symmetry_tolerance);
  return result;
}

BiorthogonalExactSelectedStructureEvaluationResult
evaluate_biorthogonal_exact_selected_structure_subspace(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::PreparedActiveSpaceContext& prepared_active_space,
    const std::vector<int>& selected_structure_indices,
    double symmetry_tolerance) {
  return evaluate_biorthogonal_exact_selected_structure_subspace(
      build_biorthogonal_full_structure_data(input, prepared_active_space),
      selected_structure_indices,
      symmetry_tolerance);
}

BiorthogonalExactSelectedStructureEvaluationResult
evaluate_biorthogonal_exact_selected_structure_subspace(
    const xmvb::vb::CppVbInput& input,
    const std::vector<int>& selected_structure_indices,
    double symmetry_tolerance) {
  const PreparedBiorthogonalInput prepared_input =
      prepare_biorthogonal_input(input);
  return evaluate_biorthogonal_exact_selected_structure_subspace(
      prepared_input.structure_data,
      selected_structure_indices,
      symmetry_tolerance);
}

void validate_biorthogonal_exact_selected_structure_matrix_build_result(
    const BiorthogonalExactSelectedStructureMatrixBuildResult& matrix_result,
    double symmetry_tolerance) {
  if (symmetry_tolerance < 0.0) {
    throw std::invalid_argument("symmetry_tolerance must be non-negative");
  }
  if (matrix_result.n_active_orbitals <= 0) {
    throw std::invalid_argument("n_active_orbitals must be positive");
  }
  if (matrix_result.n_full_structures <= 0) {
    throw std::invalid_argument("n_full_structures must be positive");
  }
  if (matrix_result.n_selected_structures <= 0) {
    throw std::invalid_argument("n_selected_structures must be positive");
  }
  if (matrix_result.n_determinants <= 0) {
    throw std::invalid_argument("n_determinants must be positive");
  }
  if (static_cast<int>(matrix_result.selected_structure_indices.size()) !=
      matrix_result.n_selected_structures) {
    throw std::invalid_argument("selected_structure_indices size is inconsistent");
  }
  validate_shape(
      matrix_result.physical_structure_overlap,
      matrix_result.n_selected_structures,
      matrix_result.n_selected_structures,
      "physical_structure_overlap");
  validate_shape(
      matrix_result.physical_structure_hamiltonian,
      matrix_result.n_selected_structures,
      matrix_result.n_selected_structures,
      "physical_structure_hamiltonian");
  throw_if_nonfinite(
      matrix_result.physical_structure_overlap,
      "physical_structure_overlap");
  throw_if_nonfinite(
      matrix_result.physical_structure_hamiltonian,
      "physical_structure_hamiltonian");
  if (!std::isfinite(
          matrix_result.structure_overlap_symmetry_residual_max_abs) ||
      !std::isfinite(
          matrix_result.structure_hamiltonian_symmetry_residual_max_abs)) {
    throw std::invalid_argument("symmetry residuals must be finite");
  }
  if (matrix_result.structure_overlap_symmetry_residual_max_abs >
          symmetry_tolerance ||
      matrix_result.structure_hamiltonian_symmetry_residual_max_abs >
          symmetry_tolerance) {
    throw std::runtime_error(
        "exact selected-space matrix build lost symmetry beyond tolerance");
  }
}

void validate_biorthogonal_exact_selected_structure_evaluation_result(
    const BiorthogonalExactSelectedStructureEvaluationResult& evaluation_result,
    double symmetry_tolerance) {
  if (symmetry_tolerance < 0.0) {
    throw std::invalid_argument("symmetry_tolerance must be non-negative");
  }
  if (evaluation_result.n_active_orbitals <= 0) {
    throw std::invalid_argument("n_active_orbitals must be positive");
  }
  if (evaluation_result.n_full_structures <= 0) {
    throw std::invalid_argument("n_full_structures must be positive");
  }
  if (evaluation_result.n_selected_structures <= 0) {
    throw std::invalid_argument("n_selected_structures must be positive");
  }
  if (evaluation_result.n_determinants <= 0) {
    throw std::invalid_argument("n_determinants must be positive");
  }
  if (static_cast<int>(evaluation_result.selected_structure_indices.size()) !=
      evaluation_result.n_selected_structures) {
    throw std::invalid_argument("selected_structure_indices size is inconsistent");
  }
  validate_shape(
      evaluation_result.selected_structure_to_determinant,
      evaluation_result.n_determinants,
      evaluation_result.n_selected_structures,
      "selected_structure_to_determinant");
  validate_shape(
      evaluation_result.overlap_action_on_selected_columns,
      evaluation_result.n_determinants,
      evaluation_result.n_selected_structures,
      "overlap_action_on_selected_columns");
  validate_shape(
      evaluation_result.biorthogonal_hamiltonian_action_on_selected_columns,
      evaluation_result.n_determinants,
      evaluation_result.n_selected_structures,
      "biorthogonal_hamiltonian_action_on_selected_columns");
  validate_shape(
      evaluation_result.physical_structure_overlap,
      evaluation_result.n_selected_structures,
      evaluation_result.n_selected_structures,
      "physical_structure_overlap");
  validate_shape(
      evaluation_result.physical_structure_hamiltonian,
      evaluation_result.n_selected_structures,
      evaluation_result.n_selected_structures,
      "physical_structure_hamiltonian");
  if (static_cast<int>(evaluation_result.eigenvalues.size()) !=
      evaluation_result.n_selected_structures) {
    throw std::invalid_argument("eigenvalue count is inconsistent");
  }
  validate_shape(
      evaluation_result.structure_coefficient_matrix,
      evaluation_result.n_selected_structures,
      evaluation_result.n_selected_structures,
      "structure_coefficient_matrix");
  if (evaluation_result.right_determinant_coefficient_matrix.size() != 0) {
    validate_shape(
        evaluation_result.right_determinant_coefficient_matrix,
        evaluation_result.n_determinants,
        evaluation_result.n_selected_structures,
        "right_determinant_coefficient_matrix");
  }
  if (evaluation_result.left_determinant_coefficient_matrix.size() != 0) {
    validate_shape(
        evaluation_result.left_determinant_coefficient_matrix,
        evaluation_result.n_determinants,
        evaluation_result.n_selected_structures,
        "left_determinant_coefficient_matrix");
  }
  throw_if_nonfinite(
      evaluation_result.selected_structure_to_determinant,
      "selected_structure_to_determinant");
  throw_if_nonfinite(
      evaluation_result.overlap_action_on_selected_columns,
      "overlap_action_on_selected_columns");
  throw_if_nonfinite(
      evaluation_result.biorthogonal_hamiltonian_action_on_selected_columns,
      "biorthogonal_hamiltonian_action_on_selected_columns");
  throw_if_nonfinite(
      evaluation_result.physical_structure_overlap,
      "physical_structure_overlap");
  throw_if_nonfinite(
      evaluation_result.physical_structure_hamiltonian,
      "physical_structure_hamiltonian");
  throw_if_nonfinite(
      evaluation_result.structure_coefficient_matrix,
      "structure_coefficient_matrix");
  if (evaluation_result.right_determinant_coefficient_matrix.size() != 0) {
    throw_if_nonfinite(
        evaluation_result.right_determinant_coefficient_matrix,
        "right_determinant_coefficient_matrix");
  }
  if (evaluation_result.left_determinant_coefficient_matrix.size() != 0) {
    throw_if_nonfinite(
        evaluation_result.left_determinant_coefficient_matrix,
        "left_determinant_coefficient_matrix");
  }
  if (!std::isfinite(
          evaluation_result.structure_overlap_symmetry_residual_max_abs) ||
      !std::isfinite(
          evaluation_result.structure_hamiltonian_symmetry_residual_max_abs)) {
    throw std::invalid_argument("symmetry residuals must be finite");
  }
  if (evaluation_result.structure_overlap_symmetry_residual_max_abs >
          symmetry_tolerance ||
      evaluation_result.structure_hamiltonian_symmetry_residual_max_abs >
          symmetry_tolerance) {
    throw std::runtime_error(
        "exact selected-space projection lost symmetry beyond tolerance");
  }
}

}  // namespace xmvb::vb::biorthogonal_vbscf
