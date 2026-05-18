#include "vb/orbital/nonredundant_orbital_space.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "runtime/cpp_block_guess_builder.hpp"

#include "vb/orbital/sparse_orbital_parameter_view.hpp"

namespace xmvb::vb {

namespace {

double parse_env_double_with_default(const char* name, double default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return default_value;
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value || (end != nullptr && end[0] != '\0') || !std::isfinite(parsed))
    return default_value;
  return parsed;
}

double nonredundant_preconditioner_min_curvature() {
  return std::max(1.0e-12,
                  parse_env_double_with_default(
                      "XMVB_CPP_NONREDUNDANT_PRECONDITIONER_MIN_CURVATURE", 1.0e-3));
}

double nonredundant_preconditioner_max_curvature() {
  return std::max(nonredundant_preconditioner_min_curvature(),
                  parse_env_double_with_default(
                      "XMVB_CPP_NONREDUNDANT_PRECONDITIONER_MAX_CURVATURE", 1.0e2));
}

// Extract the local AO overlap seen by one sparse orbital on its own support.
Eigen::MatrixXd build_local_sparse_overlap_metric(
    const Eigen::MatrixXd& block_overlap,
    const std::vector<int>& block_rows) {
  const Eigen::Index n = static_cast<Eigen::Index>(block_rows.size());
  Eigen::MatrixXd local(n, n);
  for (Eigen::Index i = 0; i < n; ++i)
    for (Eigen::Index j = 0; j < n; ++j)
      local(i, j) = block_overlap(block_rows[i], block_rows[j]);
  return local;
}

// Gather selected rows and a column range from a block matrix.
Eigen::MatrixXd gather_block_rows(
    const Eigen::MatrixXd& src,
    const std::vector<int>& rows,
    int col_begin, int col_count) {
  const Eigen::Index n = static_cast<Eigen::Index>(rows.size());
  Eigen::MatrixXd out(n, col_count);
  for (Eigen::Index i = 0; i < n; ++i)
    for (int j = 0; j < col_count; ++j)
      out(i, j) = src(rows[i], col_begin + j);
  return out;
}

// Gather selected rows and selected columns from a block matrix.
Eigen::MatrixXd gather_block_rows_cols(
    const Eigen::MatrixXd& src,
    const std::vector<int>& rows,
    const std::vector<int>& cols) {
  const Eigen::Index nr = static_cast<Eigen::Index>(rows.size());
  const Eigen::Index nc = static_cast<Eigen::Index>(cols.size());
  Eigen::MatrixXd out(nr, nc);
  for (Eigen::Index i = 0; i < nr; ++i)
    for (Eigen::Index j = 0; j < nc; ++j)
      out(i, j) = src(rows[i], cols[j]);
  return out;
}

void require_finite_matrix(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    const char* label) {
  if (!matrix.allFinite()) {
    throw std::runtime_error(
        std::string(label) + " contains non-finite values");
  }
}

void require_finite_vector(
    const Eigen::VectorXd& vector,
    const char* label) {
  if (!vector.allFinite()) {
    throw std::runtime_error(
        std::string(label) + " contains non-finite values");
  }
}

void require_finite_vector_size(
    const Eigen::VectorXd& vector,
    Eigen::Index expected_size,
    const char* label) {
  if (vector.size() != expected_size) {
    throw std::invalid_argument(
        std::string(label) + " dimension mismatch");
  }
  require_finite_vector(vector, label);
}

std::vector<int> build_block_basis_function_indices(
    const OrbitalPreparationInput& input,
    const std::vector<int>& block_orbitals) {
  if (block_orbitals.empty()) return {};
  std::vector<int> indices;
  indices.reserve(input.n_basis_functions);
  std::vector<char> seen(input.n_basis_functions, 0);
  for (const int orb : block_orbitals) {
    const int ncoeff = stored_sparse_orbital_coefficient_count(input, orb);
    for (int c = 0; c < ncoeff; ++c) {
      const int bf = input.orbital_basis_index_table[
          orb * input.n_basis_functions + c] - 1;
      if (bf < 0 || bf >= input.n_basis_functions)
        throw std::runtime_error("invalid block basis-function index");
      if (!seen[bf]) { seen[bf] = 1; indices.push_back(bf); }
    }
  }
  return indices;
}

struct BlockOrbitalMapping {
  int orbital_index = 0;
  int coefficient_count = 0;
  std::vector<int> block_row_for_slot;
};

BlockOrbitalMapping build_block_orbital_mapping(
    const OrbitalPreparationInput& input,
    int orbital_index,
    const std::unordered_map<int, int>& basis_to_block_row) {
  BlockOrbitalMapping m;
  m.orbital_index = orbital_index;
  m.coefficient_count = stored_sparse_orbital_coefficient_count(input, orbital_index);
  m.block_row_for_slot.resize(m.coefficient_count, -1);
  for (int c = 0; c < m.coefficient_count; ++c) {
    const int bf = input.orbital_basis_index_table[
        orbital_index * input.n_basis_functions + c] - 1;
    m.block_row_for_slot[c] = basis_to_block_row.at(bf);
  }
  return m;
}

Eigen::VectorXd gather_block_orbital_from_dense_matrix(
    const Eigen::Ref<const Eigen::MatrixXd>& mat,
    int orbital_index,
    const std::vector<int>& bf_indices) {
  Eigen::VectorXd block_col =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(bf_indices.size()));
  for (Eigen::Index i = 0; i < block_col.size(); ++i)
    block_col[i] = mat(bf_indices[i], orbital_index);
  return block_col;
}

Eigen::MatrixXd build_block_virtual_orbitals(
    const Eigen::MatrixXd& block_occupied,
    const Eigen::MatrixXd& block_overlap) {
  const int nbasis = static_cast<int>(block_overlap.rows());
  const int nocc = static_cast<int>(block_occupied.cols());
  const int nvirt = nbasis - nocc;
  if (nvirt <= 0) return Eigen::MatrixXd::Zero(nbasis, 0);

  // Occupied projector → metric eigensystem on complement space.
  const Eigen::MatrixXd occ_overlap =
      block_occupied.transpose() * block_overlap * block_occupied;
  Eigen::LDLT<Eigen::MatrixXd> occ_ldlt(occ_overlap);
  if (occ_ldlt.info() != Eigen::Success)
    throw std::runtime_error("failed to factorize occupied block overlap");
  const Eigen::MatrixXd occ_metric_action =
      occ_ldlt.solve(block_occupied.transpose() * block_overlap);
  const Eigen::MatrixXd P_comp =
      Eigen::MatrixXd::Identity(nbasis, nbasis) -
      block_occupied * occ_metric_action;
  const Eigen::MatrixXd virt_overlap =
      P_comp.transpose() * block_overlap * P_comp;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(virt_overlap);
  if (solver.info() != Eigen::Success)
    throw std::runtime_error("failed to diagonalize virtual block overlap");

  constexpr double kTol = 1.0e-10;
  Eigen::MatrixXd virtuals(nbasis, nvirt);
  int col = 0;
  // Eigenvalues are in increasing order. Iterate from the largest.
  for (int e = static_cast<int>(solver.eigenvalues().size()) - 1;
       e >= 0 && col < nvirt; --e) {
    if (!(solver.eigenvalues()[e] > kTol)) continue;
    Eigen::VectorXd v = P_comp * solver.eigenvectors().col(e);
    const double norm = v.dot(block_overlap * v);
    if (!(norm > std::numeric_limits<double>::epsilon()) || !std::isfinite(norm))
      continue;
    virtuals.col(col++) = v / std::sqrt(norm);
  }
  if (col != nvirt)
    throw std::runtime_error("failed to build the full block-local virtual space");
  return virtuals;
}

}  // namespace

// --- Non-member helpers used by the constructor ---

Eigen::MatrixXd build_block_effective_one_electron_matrix(
    const Eigen::Ref<const Eigen::MatrixXd>& ao_h1e,
    int n_basis,
    const std::vector<int>& bf_indices) {
  const int n = static_cast<int>(bf_indices.size());
  Eigen::MatrixXd block(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      block(i, j) = ao_h1e(bf_indices[i], bf_indices[j]);
  return block;
}

Eigen::MatrixXd build_block_effective_one_electron_matrix_on_rows(
    const Eigen::MatrixXd& block_h1e,
    const std::vector<int>& rows) {
  const Eigen::Index n = static_cast<Eigen::Index>(rows.size());
  Eigen::MatrixXd local(n, n);
  for (Eigen::Index i = 0; i < n; ++i)
    for (Eigen::Index j = 0; j < n; ++j)
      local(i, j) = block_h1e(rows[i], rows[j]);
  return local;
}

Eigen::VectorXd normalize_curvature_diagonal(const Eigen::VectorXd& diag) {
  if (diag.size() == 0) return Eigen::VectorXd::Zero(0);
  Eigen::VectorXd out = diag;
  const double lo = nonredundant_preconditioner_min_curvature();
  const double hi = nonredundant_preconditioner_max_curvature();
  for (Eigen::Index i = 0; i < out.size(); ++i) {
    double v = out[i];
    if (!std::isfinite(v) || v <= 0.0) v = 1.0;
    out[i] = std::clamp(v, lo, hi);
  }
  return out;
}

Eigen::MatrixXd build_positive_curvature_block(
    const Eigen::MatrixXd& curvature_block) {
  if (curvature_block.rows() == 0) {
    return Eigen::MatrixXd::Zero(0, 0);
  }
  if (curvature_block.rows() != curvature_block.cols() ||
      !curvature_block.allFinite()) {
    return Eigen::MatrixXd::Identity(
        curvature_block.rows(),
        curvature_block.rows());
  }

  const Eigen::MatrixXd sym_block =
      0.5 * (curvature_block + curvature_block.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(sym_block);
  if (solver.info() != Eigen::Success ||
      !solver.eigenvalues().allFinite() ||
      !solver.eigenvectors().allFinite()) {
    return Eigen::MatrixXd::Identity(
        curvature_block.rows(),
        curvature_block.rows());
  }

  const double lo = nonredundant_preconditioner_min_curvature();
  const double hi = nonredundant_preconditioner_max_curvature();
  Eigen::VectorXd safe_eigenvalues = solver.eigenvalues();
  for (Eigen::Index i = 0; i < safe_eigenvalues.size(); ++i) {
    double value = std::abs(safe_eigenvalues[i]);
    if (!std::isfinite(value) || value <= 0.0) value = 1.0;
    safe_eigenvalues[i] = std::clamp(value, lo, hi);
  }
  return solver.eigenvectors() *
      safe_eigenvalues.asDiagonal() *
      solver.eigenvectors().transpose();
}

Eigen::MatrixXd build_deterministic_orthonormal_tangent_basis(
    const Eigen::MatrixXd& tangent_generators) {
  if (tangent_generators.rows() == 0 ||
      tangent_generators.cols() == 0 ||
      !tangent_generators.allFinite()) {
    return Eigen::MatrixXd::Zero(tangent_generators.rows(), 0);
  }

  const Eigen::Index rows = tangent_generators.rows();
  const Eigen::Index max_cols = tangent_generators.cols();
  Eigen::MatrixXd basis(rows, max_cols);
  Eigen::Index admitted = 0;

  const double generator_norm = tangent_generators.norm();
  const double absolute_floor =
      std::max(1.0e-12, 1.0e-10 * generator_norm);
  for (Eigen::Index column = 0; column < max_cols; ++column) {
    Eigen::VectorXd vector = tangent_generators.col(column);
    const double original_norm = vector.norm();
    if (!(original_norm > absolute_floor) ||
        !std::isfinite(original_norm)) {
      continue;
    }

    // Modified Gram-Schmidt in the deterministic raw-generator order.  A
    // second pass removes residual components left by nearly dependent sparse
    // directions without invoking a degenerate eigensystem whose basis can
    // rotate between runs.
    for (int pass = 0; pass < 2; ++pass) {
      for (Eigen::Index existing = 0; existing < admitted; ++existing) {
        const double coefficient = basis.col(existing).dot(vector);
        if (!std::isfinite(coefficient)) {
          return Eigen::MatrixXd::Zero(rows, 0);
        }
        vector.noalias() -= coefficient * basis.col(existing);
      }
    }

    const double orthogonal_norm = vector.norm();
    if (!(orthogonal_norm >
          std::max(absolute_floor, 1.0e-8 * original_norm)) ||
        !std::isfinite(orthogonal_norm)) {
      continue;
    }
    vector /= orthogonal_norm;

    Eigen::Index pivot = 0;
    vector.cwiseAbs().maxCoeff(&pivot);
    if (vector[pivot] < 0.0) {
      vector *= -1.0;
    }

    basis.col(admitted) = std::move(vector);
    ++admitted;
  }
  if (admitted == 0 || !basis.leftCols(admitted).allFinite()) {
    return Eigen::MatrixXd::Zero(rows, 0);
  }
  return basis.leftCols(admitted);
}

std::uint64_t mix_rank_signature(
    std::uint64_t signature,
    int block_index,
    int orbital_index,
    int local_reduced_size) noexcept {
  auto mix_one = [&signature](std::uint64_t value) {
    signature ^= value;
    signature *= 1099511628211ull;
  };
  mix_one(static_cast<std::uint64_t>(block_index + 1));
  mix_one(static_cast<std::uint64_t>(orbital_index + 1));
  mix_one(static_cast<std::uint64_t>(local_reduced_size + 1));
  return signature;
}

// ===========================================================================
// Constructor
// ===========================================================================

NonredundantOrbitalSpace::NonredundantOrbitalSpace(
    const OrbitalPreparationInput& input,
    const SparseOrbitalParameterView& parameter_view,
    const Eigen::Ref<const Eigen::MatrixXd>& occ_basis_matrix,
    const Eigen::Ref<const Eigen::MatrixXd>& phys_orbital_matrix,
    const Eigen::MatrixXd* ao_effective_h1e)
    : packed_parameter_size_(parameter_view.size()) {
  if (ao_effective_h1e != nullptr &&
      (ao_effective_h1e->rows() != input.n_basis_functions ||
       ao_effective_h1e->cols() != input.n_basis_functions)) {
    throw std::invalid_argument("AO effective one-electron matrix shape mismatch");
  }

  const int n_inactive =
      (input.n_total_electrons - input.n_active_electrons) / 2;
  const int n_occupied = n_inactive + input.n_active_orbitals;
  if (n_inactive < 0 || n_occupied > input.n_orbitals)
    throw std::invalid_argument("invalid occupied-space partition");
  if (occ_basis_matrix.rows() != input.n_basis_functions ||
      occ_basis_matrix.cols() != n_occupied)
    throw std::invalid_argument("occupied orbital basis matrix shape mismatch");
  if (phys_orbital_matrix.rows() != input.n_basis_functions ||
      phys_orbital_matrix.cols() != input.n_orbitals)
    throw std::invalid_argument("physical orbital matrix shape mismatch");

  // The dense local curvature block is retained as an opt-in diagnostic
  // preconditioner.  Its clipped absolute spectrum is useful on some large
  // sparse charts, but it can over-steer tiny nearly degenerate tangent spaces;
  // the production default therefore stays with the safer diagonal scaling
  // unless the caller enables the block path explicitly.
  use_block_preconditioner_by_default_ = has_reduced_curvature_diagonal_;

  const Eigen::Map<const Eigen::MatrixXd> S(
      input.active_orbital_overlap_matrix.data(),
      input.n_basis_functions, input.n_basis_functions);
  require_finite_matrix(S, "NROS AO overlap matrix");
  const auto blocks = detect_orbital_blocks(input);

  int block_index = 0;
  for (const auto& block : blocks) {
    // Collect occupied orbitals in this block.
    std::vector<int> occ_block;
    for (const int orb : block)
      if (orb < n_occupied) occ_block.push_back(orb);
    if (occ_block.empty()) continue;

    // Block-local AO basis support.
    const std::vector<int> bf_indices =
        build_block_basis_function_indices(input, block);
    const int nbasis = static_cast<int>(bf_indices.size());
    if (nbasis < static_cast<int>(occ_block.size()))
      throw std::runtime_error("block basis count smaller than occupied count");

    std::unordered_map<int, int> bf_to_row;
    bf_to_row.reserve(bf_indices.size());
    for (int r = 0; r < nbasis; ++r)
      bf_to_row.emplace(bf_indices[r], r);

    // Build block-local occupied matrices.
    // block_occ_raw: from orbital_value_table (raw sparse coefficients).
    // block_occ_basis: from occ_basis_matrix (orthogonal AO occupied orbitals).
    const int nocc = static_cast<int>(occ_block.size());
    int block_n_inactive = 0;
    Eigen::MatrixXd block_occ_raw = Eigen::MatrixXd::Zero(nbasis, nocc);
    Eigen::MatrixXd block_occ_basis = Eigen::MatrixXd::Zero(nbasis, nocc);
    std::vector<BlockOrbitalMapping> mappings;
    mappings.reserve(nocc);
    for (int k = 0; k < nocc; ++k) {
      const int orb = occ_block[k];
      const int ncoeff = stored_sparse_orbital_coefficient_count(input, orb);
      for (int c = 0; c < ncoeff; ++c) {
        const int bf = input.orbital_basis_index_table[
            orb * input.n_basis_functions + c] - 1;
        block_occ_raw(bf_to_row.at(bf), k) =
            input.orbital_value_table[orb * input.n_basis_functions + c];
      }
      block_occ_basis.col(k) =
          gather_block_orbital_from_dense_matrix(occ_basis_matrix, orb, bf_indices);
      mappings.push_back(build_block_orbital_mapping(input, orb, bf_to_row));
      if (orb < n_inactive) ++block_n_inactive;
    }
    const int n_active = nocc - block_n_inactive;

    // Block-local overlap.
    Eigen::MatrixXd block_S(nbasis, nbasis);
    for (int i = 0; i < nbasis; ++i)
      for (int j = 0; j < nbasis; ++j)
        block_S(i, j) = S(bf_indices[i], bf_indices[j]);

    // Block-local virtual complement (from raw sparse coefficients).
    const Eigen::MatrixXd block_virt =
        build_block_virtual_orbitals(block_occ_raw, block_S);
    require_finite_matrix(block_virt, "NROS block virtual orbital basis");
    const int nvirt = static_cast<int>(block_virt.cols());
    if (n_active + nvirt == 0 && block_n_inactive == 0) continue;

    // Block-local Fock submatrix (for curvature).
    Eigen::MatrixXd block_h1e;
    if (ao_effective_h1e != nullptr)
      block_h1e = build_block_effective_one_electron_matrix(
          *ao_effective_h1e, input.n_basis_functions, bf_indices);

    BlockBasis bb;
    bb.n_inactive = block_n_inactive;
    bb.n_occupied = nocc;
    bb.n_virtual = nvirt;
    bb.basis_function_indices = bf_indices;
    bb.block_overlap_matrix = block_S;
    bb.orbitals.reserve(nocc);

    // Build per-orbital projectors.
    for (int k = 0; k < nocc; ++k) {
      const auto& m = mappings[k];
      OrbitalProjector proj;
      proj.flat_indices.reserve(m.coefficient_count);
      proj.packed_indices.reserve(m.coefficient_count);
      proj.block_rows.reserve(m.coefficient_count);
      for (int c = 0; c < m.coefficient_count; ++c) {
        const int flat = m.orbital_index * input.n_basis_functions + c;
        const int packed = parameter_view.packed_index(m.orbital_index, c);
        if (packed < 0)
          throw std::runtime_error("failed to locate packed sparse-orbital parameter");
        proj.flat_indices.push_back(flat);
        proj.packed_indices.push_back(packed);
        proj.block_rows.push_back(m.block_row_for_slot[c]);
      }

      // --- Build B_p directly from block matrices ---
      const Eigen::Index local_size =
          static_cast<Eigen::Index>(proj.block_rows.size());

      // Gather local coefficients.
      Eigen::VectorXd x_p = Eigen::VectorXd::Zero(local_size);
      for (Eigen::Index i = 0; i < local_size; ++i)
        x_p[i] = input.orbital_value_table[proj.flat_indices[i]];
      require_finite_vector(x_p, "NROS local sparse orbital coefficients");

      const Eigen::MatrixXd local_S =
          build_local_sparse_overlap_metric(block_S, proj.block_rows);
      require_finite_matrix(local_S, "NROS local sparse overlap metric");
      const Eigen::VectorXd Sx = local_S * x_p;
      const double rho2 = x_p.dot(Sx);
      if (!Sx.allFinite() || !std::isfinite(rho2)) {
        throw std::runtime_error(
            "NROS sparse orbital metric norm contains non-finite values");
      }

      // Determine raw basis dimension.
      int raw_dim = 0;
      if (k < block_n_inactive) {
        raw_dim = n_active + nvirt;
      } else {
        raw_dim = block_n_inactive + (n_active > 0 ? n_active - 1 : 0) + nvirt;
      }
      if (raw_dim == 0) {
        proj.tangent_basis = Eigen::MatrixXd::Zero(local_size, 0);
        proj.local_reduced_offset = reduced_size_;
        proj.local_reduced_size = 0;
        rank_signature_ = mix_rank_signature(
            rank_signature_,
            block_index,
            k,
            proj.local_reduced_size);
        bb.orbitals.push_back(std::move(proj));
        continue;
      }

      // Fill B_p by sampling block_occ_raw and block_virt on block_rows.
      // Raw sparse coefficients match the retraction coordinate system and
      // are always finite, unlike the auxiliary orbital matrix which may
      // contain infinity for HAO systems.
      Eigen::MatrixXd B_p(local_size, raw_dim);
      int col = 0;
      if (k < block_n_inactive) {
        // inactive: [active cols of occ_raw, block_virt]
        if (n_active > 0) {
          B_p.middleCols(col, n_active) =
              gather_block_rows(block_occ_raw, proj.block_rows,
                                block_n_inactive, n_active);
          col += n_active;
        }
      } else {
        const int active_idx = k - block_n_inactive;
        // active: [inactive cols, active cols excl self, virtual]
        if (block_n_inactive > 0) {
          B_p.middleCols(col, block_n_inactive) =
              gather_block_rows(block_occ_raw, proj.block_rows, 0,
                                block_n_inactive);
          col += block_n_inactive;
        }
        if (n_active > 0) {
          std::vector<int> active_cols;
          active_cols.reserve(n_active - 1);
          for (int j = 0; j < n_active; ++j)
            if (j != active_idx) active_cols.push_back(block_n_inactive + j);
          B_p.middleCols(col, static_cast<int>(active_cols.size())) =
              gather_block_rows_cols(block_occ_raw, proj.block_rows,
                                     active_cols);
          col += static_cast<int>(active_cols.size());
        }
      }
      if (nvirt > 0) {
        B_p.middleCols(col, nvirt) =
            gather_block_rows(block_virt, proj.block_rows, 0, nvirt);
      }
      require_finite_matrix(B_p, "NROS raw tangent generator");

      // Drop near-zero columns of B_p before projection to avoid ill-conditioned
      // Gram matrix.  Raw sparse coefficients for HAO orbitals with disjoint
      // support produce degenerate direction columns.
      {
        const double col_norm_threshold = 1.0e-12 * x_p.norm();
        Eigen::Index kept = 0;
        for (Eigen::Index c = 0; c < B_p.cols(); ++c) {
          if (B_p.col(c).norm() > col_norm_threshold) {
            if (kept != c) B_p.col(kept) = B_p.col(c);
            ++kept;
          }
        }
        B_p = B_p.leftCols(kept);
      }
      if (B_p.cols() == 0) {
        proj.tangent_basis = Eigen::MatrixXd::Zero(local_size, 0);
        proj.local_reduced_offset = reduced_size_;
        proj.local_reduced_size = 0;
        rank_signature_ = mix_rank_signature(
            rank_signature_,
            block_index,
            k,
            proj.local_reduced_size);
        bb.orbitals.push_back(std::move(proj));
        continue;
      }

      // For non-orthogonal VB orbitals there is no sphere constraint, so the
      // tangent space is B_p itself (the complement of other occupied orbitals).
      Eigen::MatrixXd U_p =
          build_deterministic_orthonormal_tangent_basis(B_p);
      if (U_p.cols() == 0 || !U_p.allFinite()) {
        proj.tangent_basis = Eigen::MatrixXd::Zero(local_size, 0);
        proj.local_reduced_offset = reduced_size_;
        proj.local_reduced_size = 0;
        rank_signature_ = mix_rank_signature(
            rank_signature_,
            block_index,
            k,
            proj.local_reduced_size);
        bb.orbitals.push_back(std::move(proj));
        continue;
      }

      proj.tangent_basis = std::move(U_p);
      proj.local_reduced_offset = reduced_size_;
      proj.local_reduced_size = static_cast<int>(proj.tangent_basis.cols());
      reduced_size_ += proj.local_reduced_size;
      rank_signature_ = mix_rank_signature(
          rank_signature_,
          block_index,
          k,
          proj.local_reduced_size);

      // Curvature diagonal: diag(U_p^T (F - eps S) U_p)
      if (ao_effective_h1e != nullptr && proj.local_reduced_size > 0) {
        const Eigen::MatrixXd local_F =
            build_block_effective_one_electron_matrix_on_rows(
                block_h1e, proj.block_rows);
        Eigen::VectorXd curv =
            Eigen::VectorXd::Ones(proj.local_reduced_size);
        Eigen::MatrixXd block_curv =
            Eigen::MatrixXd::Identity(
                proj.local_reduced_size,
                proj.local_reduced_size);
        if (local_F.allFinite() &&
            local_S.allFinite() &&
            proj.tangent_basis.allFinite() &&
            rho2 > std::numeric_limits<double>::epsilon()) {
          const Eigen::VectorXd F_x = local_F * x_p;
          const double eps_p = x_p.dot(F_x) / rho2;
          if (F_x.allFinite() && std::isfinite(eps_p)) {
            const Eigen::MatrixXd F_U = local_F * proj.tangent_basis;
            const Eigen::MatrixXd S_U = local_S * proj.tangent_basis;
            if (F_U.allFinite() && S_U.allFinite()) {
              block_curv =
                  proj.tangent_basis.transpose() *
                  (F_U - eps_p * S_U);
              block_curv =
                  0.5 * (block_curv + block_curv.transpose());
              for (int d = 0; d < proj.local_reduced_size; ++d) {
                curv[d] = block_curv(d, d);
              }
            }
          }
        }
        // The diagonal is a preconditioner only.  If the local effective
        // Fock block is not numerically usable, fall back to the identity
        // instead of letting NaN curvature poison a reduced Newton solve.
        proj.curvature_diagonal = normalize_curvature_diagonal(curv);
        proj.curvature_block = build_positive_curvature_block(block_curv);
        has_reduced_curvature_diagonal_ = true;
      }

      bb.orbitals.push_back(std::move(proj));
    }

    block_bases_.push_back(std::move(bb));
    ++block_index;
  }
}

// ===========================================================================
// Runtime functions
// ===========================================================================

NonredundantOrbitalSpace::ProjectionResult
NonredundantOrbitalSpace::project_impl(
    const Eigen::VectorXd& packed_vector,
    bool recover_tangent_coordinates,
    bool build_packed_projection) const {
  (void)recover_tangent_coordinates;
  require_finite_vector_size(
      packed_vector,
      static_cast<Eigen::Index>(packed_parameter_size_),
      "NROS packed projection input");
  ProjectionResult result;
  result.reduced_gradient =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(reduced_size_));
  result.packed_projected_gradient =
      build_packed_projection
          ? Eigen::VectorXd::Zero(static_cast<Eigen::Index>(packed_parameter_size_))
          : Eigen::VectorXd();

  for (const auto& block_basis : block_bases_) {
    for (const auto& projector : block_basis.orbitals) {
      if (projector.local_reduced_size <= 0) continue;
      const Eigen::Index local_size =
          static_cast<Eigen::Index>(projector.packed_indices.size());
      if (projector.tangent_basis.rows() != local_size ||
          projector.tangent_basis.cols() != projector.local_reduced_size ||
          !projector.tangent_basis.allFinite()) {
        throw std::runtime_error("NROS projector tangent basis is invalid");
      }
      Eigen::VectorXd g_p = Eigen::VectorXd::Zero(local_size);
      for (Eigen::Index i = 0; i < local_size; ++i) {
        if (projector.packed_indices[i] < 0 ||
            projector.packed_indices[i] >= packed_parameter_size_) {
          throw std::runtime_error("NROS projector packed index is out of range");
        }
        g_p[i] = packed_vector[projector.packed_indices[i]];
      }
      const Eigen::VectorXd z_p = projector.tangent_basis.transpose() * g_p;
      require_finite_vector(z_p, "NROS reduced projected block");
      result.reduced_gradient.segment(
          projector.local_reduced_offset,
          projector.local_reduced_size) = z_p;
      if (build_packed_projection) {
        const Eigen::VectorXd dx_p = projector.tangent_basis * z_p;
        require_finite_vector(dx_p, "NROS packed projected block");
        for (Eigen::Index i = 0; i < dx_p.size(); ++i)
          result.packed_projected_gradient[projector.packed_indices[i]] += dx_p[i];
      }
    }
  }
  require_finite_vector(result.reduced_gradient, "NROS reduced projection");
  if (build_packed_projection) {
    require_finite_vector(
        result.packed_projected_gradient,
        "NROS packed projection");
  }
  return result;
}

Eigen::VectorXd NonredundantOrbitalSpace::project_reduced_gradient(
    const Eigen::VectorXd& packed_gradient) const {
  return project_impl(packed_gradient, false, false).reduced_gradient;
}

NonredundantOrbitalSpace::ProjectionResult
NonredundantOrbitalSpace::project_gradient(
    const Eigen::VectorXd& packed_gradient) const {
  return project_impl(packed_gradient, false, true);
}

NonredundantOrbitalSpace::ProjectionResult
NonredundantOrbitalSpace::project_vector(
    const Eigen::VectorXd& packed_vector) const {
  return project_impl(packed_vector, true, true);
}

Eigen::VectorXd NonredundantOrbitalSpace::apply_inverse_reduced_curvature(
    const Eigen::VectorXd& reduced_vector) const {
  require_finite_vector_size(
      reduced_vector,
      static_cast<Eigen::Index>(reduced_size_),
      "NROS inverse-curvature input");
  if (!has_reduced_curvature_diagonal_) return reduced_vector;
  Eigen::VectorXd out = reduced_vector;
  constexpr double kMin = 1.0e-12;
  for (const auto& bb : block_bases_) {
    for (const auto& p : bb.orbitals) {
      if (p.curvature_diagonal.size() == 0) continue;
      if (p.curvature_diagonal.size() != p.local_reduced_size ||
          !p.curvature_diagonal.allFinite()) {
        continue;
      }
      out.segment(p.local_reduced_offset, p.local_reduced_size).array() /=
          p.curvature_diagonal.array().max(kMin);
    }
  }
  return out.allFinite() ? out : reduced_vector;
}

Eigen::VectorXd
NonredundantOrbitalSpace::apply_inverse_reduced_block_preconditioner(
    const Eigen::VectorXd& reduced_vector) const {
  require_finite_vector_size(
      reduced_vector,
      static_cast<Eigen::Index>(reduced_size_),
      "NROS block inverse-curvature input");
  if (!has_reduced_curvature_diagonal_) return reduced_vector;

  Eigen::VectorXd out = reduced_vector;
  for (const auto& bb : block_bases_) {
    for (const auto& p : bb.orbitals) {
      if (p.local_reduced_size <= 0) continue;
      if (p.curvature_block.rows() != p.local_reduced_size ||
          p.curvature_block.cols() != p.local_reduced_size ||
          !p.curvature_block.allFinite()) {
        if (p.curvature_diagonal.size() == p.local_reduced_size &&
            p.curvature_diagonal.allFinite()) {
          out.segment(p.local_reduced_offset, p.local_reduced_size).array() /=
              p.curvature_diagonal.array().max(1.0e-12);
        }
        continue;
      }
      Eigen::LDLT<Eigen::MatrixXd> ldlt(p.curvature_block);
      if (ldlt.info() != Eigen::Success) {
        return apply_inverse_reduced_curvature(reduced_vector);
      }
      const Eigen::VectorXd solved =
          ldlt.solve(
              reduced_vector.segment(
                  p.local_reduced_offset,
                  p.local_reduced_size));
      if (ldlt.info() != Eigen::Success || !solved.allFinite()) {
        return apply_inverse_reduced_curvature(reduced_vector);
      }
      out.segment(p.local_reduced_offset, p.local_reduced_size) = solved;
    }
  }
  return out.allFinite() ? out : apply_inverse_reduced_curvature(reduced_vector);
}

Eigen::VectorXd NonredundantOrbitalSpace::apply_reduced_curvature(
    const Eigen::VectorXd& reduced_vector) const {
  require_finite_vector_size(
      reduced_vector,
      static_cast<Eigen::Index>(reduced_size_),
      "NROS curvature input");
  if (!has_reduced_curvature_diagonal_) return reduced_vector;
  Eigen::VectorXd out = reduced_vector;
  for (const auto& bb : block_bases_) {
    for (const auto& p : bb.orbitals) {
      if (p.curvature_block.rows() == p.local_reduced_size &&
          p.curvature_block.cols() == p.local_reduced_size &&
          p.curvature_block.allFinite()) {
        out.segment(p.local_reduced_offset, p.local_reduced_size).noalias() =
            p.curvature_block *
            reduced_vector.segment(
                p.local_reduced_offset,
                p.local_reduced_size);
        continue;
      }
      if (p.curvature_diagonal.size() == 0) continue;
      if (p.curvature_diagonal.size() != p.local_reduced_size ||
          !p.curvature_diagonal.allFinite()) {
        continue;
      }
      out.segment(p.local_reduced_offset, p.local_reduced_size).array() *=
          p.curvature_diagonal.array();
    }
  }
  return out.allFinite() ? out : reduced_vector;
}

Eigen::VectorXd NonredundantOrbitalSpace::expand_step(
    const Eigen::VectorXd& reduced_step) const {
  require_finite_vector_size(
      reduced_step,
      static_cast<Eigen::Index>(reduced_size_),
      "NROS reduced expansion input");
  Eigen::VectorXd packed =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(packed_parameter_size_));
  for (const auto& bb : block_bases_) {
    for (const auto& p : bb.orbitals) {
      if (p.local_reduced_size <= 0) continue;
      if (p.tangent_basis.rows() !=
              static_cast<Eigen::Index>(p.packed_indices.size()) ||
          p.tangent_basis.cols() != p.local_reduced_size ||
          !p.tangent_basis.allFinite()) {
        throw std::runtime_error("NROS expansion tangent basis is invalid");
      }
      const Eigen::VectorXd dx = p.tangent_basis *
          reduced_step.segment(p.local_reduced_offset, p.local_reduced_size);
      require_finite_vector(dx, "NROS packed expansion block");
      for (Eigen::Index i = 0; i < dx.size(); ++i) {
        if (p.packed_indices[i] < 0 ||
            p.packed_indices[i] >= packed_parameter_size_) {
          throw std::runtime_error("NROS expansion packed index is out of range");
        }
        packed[p.packed_indices[i]] += dx[i];
      }
    }
  }
  require_finite_vector(packed, "NROS packed expansion");
  return packed;
}

Eigen::VectorXd NonredundantOrbitalSpace::expand_retract_input_tangent(
    const OrbitalPreparationInput& orbital_preparation_input,
    const Eigen::VectorXd& reduced_step) const {
  require_finite_vector_size(
      reduced_step,
      static_cast<Eigen::Index>(reduced_size_),
      "NROS retraction tangent input");
  Eigen::VectorXd tangent = Eigen::VectorXd::Zero(
      static_cast<Eigen::Index>(
          orbital_preparation_input.orbital_value_table.size()));
  for (const auto& bb : block_bases_) {
    for (const auto& p : bb.orbitals) {
      if (p.local_reduced_size <= 0) continue;
      if (p.tangent_basis.rows() !=
              static_cast<Eigen::Index>(p.flat_indices.size()) ||
          p.tangent_basis.cols() != p.local_reduced_size ||
          !p.tangent_basis.allFinite()) {
        throw std::runtime_error("NROS retraction tangent basis is invalid");
      }
      const Eigen::VectorXd dx = p.tangent_basis *
          reduced_step.segment(p.local_reduced_offset, p.local_reduced_size);
      require_finite_vector(dx, "NROS retraction tangent block");
      for (Eigen::Index i = 0; i < dx.size(); ++i) {
        if (p.flat_indices[i] < 0 ||
            p.flat_indices[i] >= tangent.size()) {
          throw std::runtime_error("NROS retraction flat index is out of range");
        }
        tangent[p.flat_indices[i]] += dx[i];
      }
    }
  }
  require_finite_vector(tangent, "NROS retraction tangent");
  return tangent;
}

OrbitalPreparationInput NonredundantOrbitalSpace::retract_step(
    const OrbitalPreparationInput& orbital_preparation_input,
    const Eigen::VectorXd& reduced_step,
    double step_scale) const {
  if (!std::isfinite(step_scale)) {
    throw std::invalid_argument("NROS retraction step scale is non-finite");
  }
  OrbitalPreparationInput trial = orbital_preparation_input;
  std::vector<double> updated = orbital_preparation_input.orbital_value_table;
  const Eigen::VectorXd tangent =
      expand_retract_input_tangent(
          orbital_preparation_input,
          reduced_step);
  if (tangent.size() != static_cast<Eigen::Index>(updated.size())) {
    throw std::runtime_error("NROS retraction tangent size mismatch");
  }
  for (Eigen::Index i = 0; i < tangent.size(); ++i) {
    updated[static_cast<std::size_t>(i)] += step_scale * tangent[i];
    if (!std::isfinite(updated[static_cast<std::size_t>(i)])) {
      throw std::runtime_error(
          "NROS retraction produced non-finite sparse coefficients");
    }
  }
  trial.orbital_value_table = std::move(updated);
  enforce_strict_sparse_orbital_support(&trial);
  return trial;
}

}  // namespace xmvb::vb
