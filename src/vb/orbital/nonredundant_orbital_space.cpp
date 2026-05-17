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

#include "runtime/cpp_block_guess_builder.hpp"
#include "vb/orbital/legacy_jacobi_diagonalizer.hpp"
#include "vb/orbital/sparse_orbital_parameter_view.hpp"

namespace xmvb::vb {

namespace {

constexpr int kLegacyOrbitalTypeOeo = 3;

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

  // Match legacy genVirtualOrb: occupied projector → metric eigensystem.
  const Eigen::MatrixXd occ_overlap =
      block_occupied.transpose() * block_overlap * block_occupied;
  Eigen::LDLT<Eigen::MatrixXd> occ_ldlt(occ_overlap);
  if (occ_ldlt.info() != Eigen::Success)
    throw std::runtime_error("failed to factorize occupied block overlap");
  const Eigen::MatrixXd occ_metric_action =
      occ_ldlt.solve(block_occupied.transpose() * block_overlap);
  Eigen::MatrixXd P_comp =
      Eigen::MatrixXd::Identity(nbasis, nbasis) -
      block_occupied * occ_metric_action;
  const Eigen::MatrixXd virt_overlap =
      P_comp.transpose() * block_overlap * P_comp;

  const LegacyJacobiDiagonalizationResult eigenpairs =
      diagonalize_self_adjoint_legacy_jacobi(virt_overlap);

  constexpr double kTol = 1.0e-10;
  Eigen::MatrixXd virtuals(nbasis, nvirt);
  int col = 0;
  for (int e = 0; e < eigenpairs.eigenvalues.size() && col < nvirt; ++e) {
    if (!(eigenpairs.eigenvalues[e] > kTol)) continue;
    Eigen::VectorXd v = P_comp * eigenpairs.eigenvectors.col(e);
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

  use_block_preconditioner_by_default_ =
      !(input.spin_multiplicity > 1 &&
        input.orbital_type == kLegacyOrbitalTypeOeo &&
        input.n_active_orbitals > 0 &&
        input.n_active_orbitals <= 7);

  const Eigen::Map<const Eigen::MatrixXd> S(
      input.active_orbital_overlap_matrix.data(),
      input.n_basis_functions, input.n_basis_functions);
  const auto blocks = detect_orbital_blocks(input);

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

      const Eigen::MatrixXd local_S =
          build_local_sparse_overlap_metric(block_S, proj.block_rows);
      const Eigen::VectorXd Sx = local_S * x_p;
      const double rho2 = x_p.dot(Sx);

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
        bb.orbitals.push_back(std::move(proj));
        continue;
      }

      // Fill B_p by sampling block_occ_basis and block_virt on block_rows.
      Eigen::MatrixXd B_p = Eigen::MatrixXd::Zero(local_size, raw_dim);
      int col = 0;
      if (k < block_n_inactive) {
        // inactive: [active cols of occ_basis, block_virt]
        if (n_active > 0) {
          for (Eigen::Index i = 0; i < local_size; ++i)
            for (int a = 0; a < n_active; ++a)
              B_p(i, col + a) = block_occ_basis(proj.block_rows[i],
                                                 block_n_inactive + a);
          col += n_active;
        }
      } else {
        const int active_idx = k - block_n_inactive;
        // active: [inactive cols, active cols excl self, virtual]
        if (block_n_inactive > 0) {
          for (Eigen::Index i = 0; i < local_size; ++i)
            for (int j = 0; j < block_n_inactive; ++j)
              B_p(i, col + j) = block_occ_basis(proj.block_rows[i], j);
          col += block_n_inactive;
        }
        if (n_active > 0) {
          int d = 0;
          for (int j = 0; j < n_active; ++j) {
            if (j == active_idx) continue;
            for (Eigen::Index i = 0; i < local_size; ++i)
              B_p(i, col + d) = block_occ_basis(proj.block_rows[i],
                                                 block_n_inactive + j);
            ++d;
          }
          col += n_active - 1;
        }
      }
      if (nvirt > 0) {
        for (Eigen::Index i = 0; i < local_size; ++i)
          for (int v = 0; v < nvirt; ++v)
            B_p(i, col + v) = block_virt(proj.block_rows[i], v);
      }

      // Tangent projection: T_p = (I - x_p x_p^T S / rho^2) B_p
      Eigen::MatrixXd T_p = B_p;
      if (rho2 > std::numeric_limits<double>::epsilon()) {
        const Eigen::VectorXd nSx = Sx / rho2;
        T_p.noalias() -= x_p * (nSx.transpose() * B_p).transpose();
      }

      // Whiten: G_p = T_p^T T_p, Cholesky, U_p = T_p L^{-T}
      Eigen::MatrixXd G_p = T_p.transpose() * T_p;
      Eigen::LLT<Eigen::MatrixXd> llt(G_p);
      if (llt.info() != Eigen::Success) {
        proj.tangent_basis = Eigen::MatrixXd::Zero(local_size, 0);
        proj.local_reduced_offset = reduced_size_;
        proj.local_reduced_size = 0;
        bb.orbitals.push_back(std::move(proj));
        continue;
      }
      Eigen::MatrixXd U_p = llt.matrixU().solve(T_p.transpose()).transpose();

      proj.tangent_basis = std::move(U_p);
      proj.local_reduced_offset = reduced_size_;
      proj.local_reduced_size = static_cast<int>(proj.tangent_basis.cols());
      reduced_size_ += proj.local_reduced_size;

      // Curvature diagonal: diag(U_p^T (F - eps S) U_p)
      if (ao_effective_h1e != nullptr && proj.local_reduced_size > 0) {
        const Eigen::MatrixXd local_F =
            build_block_effective_one_electron_matrix_on_rows(
                block_h1e, proj.block_rows);
        const double eps_p =
            rho2 > std::numeric_limits<double>::epsilon()
                ? x_p.dot(local_F * x_p) / rho2 : 0.0;
        const Eigen::MatrixXd F_U = local_F * proj.tangent_basis;
        const Eigen::MatrixXd S_U = local_S * proj.tangent_basis;
        Eigen::VectorXd curv(proj.local_reduced_size);
        for (int d = 0; d < proj.local_reduced_size; ++d)
          curv[d] = proj.tangent_basis.col(d).dot(F_U.col(d)) -
                    eps_p * proj.tangent_basis.col(d).dot(S_U.col(d));
        proj.curvature_diagonal = normalize_curvature_diagonal(curv);
        has_reduced_curvature_diagonal_ = true;
      }

      bb.orbitals.push_back(std::move(proj));
    }

    block_bases_.push_back(std::move(bb));
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
      Eigen::VectorXd g_p = Eigen::VectorXd::Zero(local_size);
      for (Eigen::Index i = 0; i < local_size; ++i)
        g_p[i] = packed_vector[projector.packed_indices[i]];
      const Eigen::VectorXd z_p = projector.tangent_basis.transpose() * g_p;
      result.reduced_gradient.segment(
          projector.local_reduced_offset,
          projector.local_reduced_size) = z_p;
      if (build_packed_projection) {
        const Eigen::VectorXd dx_p = projector.tangent_basis * z_p;
        for (Eigen::Index i = 0; i < dx_p.size(); ++i)
          result.packed_projected_gradient[projector.packed_indices[i]] += dx_p[i];
      }
    }
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
  if (!has_reduced_curvature_diagonal_) return reduced_vector;
  Eigen::VectorXd out = reduced_vector;
  constexpr double kMin = 1.0e-12;
  for (const auto& bb : block_bases_) {
    for (const auto& p : bb.orbitals) {
      if (p.curvature_diagonal.size() == 0) continue;
      out.segment(p.local_reduced_offset, p.local_reduced_size).array() /=
          p.curvature_diagonal.array().max(kMin);
    }
  }
  return out;
}

Eigen::VectorXd
NonredundantOrbitalSpace::apply_inverse_reduced_block_preconditioner(
    const Eigen::VectorXd& reduced_vector) const {
  return apply_inverse_reduced_curvature(reduced_vector);
}

Eigen::VectorXd NonredundantOrbitalSpace::apply_reduced_curvature(
    const Eigen::VectorXd& reduced_vector) const {
  if (!has_reduced_curvature_diagonal_) return reduced_vector;
  Eigen::VectorXd out = reduced_vector;
  for (const auto& bb : block_bases_) {
    for (const auto& p : bb.orbitals) {
      if (p.curvature_diagonal.size() == 0) continue;
      out.segment(p.local_reduced_offset, p.local_reduced_size).array() *=
          p.curvature_diagonal.array();
    }
  }
  return out;
}

Eigen::VectorXd NonredundantOrbitalSpace::expand_step(
    const Eigen::VectorXd& reduced_step) const {
  Eigen::VectorXd packed =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(packed_parameter_size_));
  for (const auto& bb : block_bases_) {
    for (const auto& p : bb.orbitals) {
      if (p.local_reduced_size <= 0) continue;
      const Eigen::VectorXd dx = p.tangent_basis *
          reduced_step.segment(p.local_reduced_offset, p.local_reduced_size);
      for (Eigen::Index i = 0; i < dx.size(); ++i)
        packed[p.packed_indices[i]] += dx[i];
    }
  }
  return packed;
}

Eigen::VectorXd NonredundantOrbitalSpace::expand_retract_input_tangent(
    const OrbitalPreparationInput& orbital_preparation_input,
    const Eigen::VectorXd& reduced_step) const {
  Eigen::VectorXd tangent = Eigen::VectorXd::Zero(
      static_cast<Eigen::Index>(
          orbital_preparation_input.orbital_value_table.size()));
  for (const auto& bb : block_bases_) {
    for (const auto& p : bb.orbitals) {
      if (p.local_reduced_size <= 0) continue;
      const Eigen::VectorXd dx = p.tangent_basis *
          reduced_step.segment(p.local_reduced_offset, p.local_reduced_size);
      for (Eigen::Index i = 0; i < dx.size(); ++i)
        tangent[p.flat_indices[i]] += dx[i];
    }
  }
  return tangent;
}

OrbitalPreparationInput NonredundantOrbitalSpace::retract_step(
    const OrbitalPreparationInput& orbital_preparation_input,
    const Eigen::VectorXd& reduced_step,
    double step_scale) const {
  OrbitalPreparationInput trial = orbital_preparation_input;
  std::vector<double> updated = orbital_preparation_input.orbital_value_table;
  for (const auto& bb : block_bases_) {
    for (const auto& p : bb.orbitals) {
      if (p.local_reduced_size <= 0) continue;
      const Eigen::VectorXd dx = p.tangent_basis *
          reduced_step.segment(p.local_reduced_offset, p.local_reduced_size);
      for (Eigen::Index i = 0; i < dx.size(); ++i)
        updated[p.flat_indices[i]] += step_scale * dx[i];
    }
  }
  trial.orbital_value_table = std::move(updated);
  enforce_strict_sparse_orbital_support(&trial);
  return trial;
}

}  // namespace xmvb::vb
