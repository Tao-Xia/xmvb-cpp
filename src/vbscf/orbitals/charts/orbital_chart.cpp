#include "vbscf/orbitals/charts/orbital_chart.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include "vbscf/orbitals/charts/orbital_block_partition.hpp"
#include "vbscf/orbitals/charts/sparse_parameter_layout.hpp"
#include "vbscf/optimization/preconditioners/normalized_orbital_curvature.hpp"
#include "vbscf/optimization/preconditioners/projected_orbital_surrogate.hpp"

namespace xmvb::vb {

namespace {

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

// --- Non-member helpers used by the constructor ---

Eigen::VectorXd normalize_curvature_diagonal(const Eigen::VectorXd& diag) {
  if (diag.size() == 0) return Eigen::VectorXd::Zero(0);
  Eigen::VectorXd out = diag.cwiseAbs();
  double spectral_scale = 0.0;
  for (Eigen::Index i = 0; i < out.size(); ++i) {
    if (std::isfinite(out[i])) spectral_scale = std::max(spectral_scale, out[i]);
  }
  if (!(spectral_scale > 0.0)) return Eigen::VectorXd::Ones(diag.size());
  const double spectral_floor =
      std::sqrt(std::numeric_limits<double>::epsilon()) * spectral_scale;
  for (Eigen::Index i = 0; i < out.size(); ++i) {
    if (!std::isfinite(out[i])) out[i] = spectral_scale;
    out[i] = std::max(out[i], spectral_floor);
  }
  return out;
}

struct PositiveCurvatureBlock {
  Eigen::MatrixXd matrix;
  Eigen::MatrixXd inverse;
};

PositiveCurvatureBlock build_positive_curvature_block(
    const Eigen::MatrixXd& curvature_block) {
  if (curvature_block.rows() == 0) {
    return {Eigen::MatrixXd::Zero(0, 0), Eigen::MatrixXd::Zero(0, 0)};
  }
  if (curvature_block.rows() != curvature_block.cols() ||
      !curvature_block.allFinite()) {
    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(
        curvature_block.rows(), curvature_block.rows());
    return {identity, identity};
  }

  const Eigen::MatrixXd sym_block =
      0.5 * (curvature_block + curvature_block.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(sym_block);
  if (solver.info() != Eigen::Success ||
      !solver.eigenvalues().allFinite() ||
      !solver.eigenvectors().allFinite()) {
    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(
        curvature_block.rows(), curvature_block.rows());
    return {identity, identity};
  }

  Eigen::VectorXd safe_eigenvalues = solver.eigenvalues().cwiseAbs();
  const double spectral_scale = safe_eigenvalues.maxCoeff();
  if (!(spectral_scale > 0.0) || !std::isfinite(spectral_scale)) {
    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(
        curvature_block.rows(), curvature_block.rows());
    return {identity, identity};
  }
  const double spectral_floor =
      std::sqrt(std::numeric_limits<double>::epsilon()) * spectral_scale;
  for (Eigen::Index i = 0; i < safe_eigenvalues.size(); ++i) {
    safe_eigenvalues[i] = std::max(safe_eigenvalues[i], spectral_floor);
  }
  PositiveCurvatureBlock result;
  result.matrix = solver.eigenvectors() *
      safe_eigenvalues.asDiagonal() * solver.eigenvectors().transpose();
  result.inverse = solver.eigenvectors() *
      safe_eigenvalues.cwiseInverse().asDiagonal() *
      solver.eigenvectors().transpose();
  return result;
}

Eigen::MatrixXd build_exact_local_sparse_quotient_basis(
    const OrbitalPreparationInput& input,
    int orbital_index,
    int n_inactive,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_orbitals,
    const std::vector<int>& local_basis_indices) {
  const int local_size = static_cast<int>(local_basis_indices.size());
  const bool is_active = orbital_index >= n_inactive;
  const int source_count = n_inactive + (is_active ? 1 : 0);
  if (local_size == 0) return Eigen::MatrixXd::Zero(0, 0);
  if (source_count == 0) {
    return Eigen::MatrixXd::Identity(local_size, local_size);
  }

  Eigen::MatrixXd gauge_sources = Eigen::MatrixXd::Zero(
      input.n_basis_functions, source_count);
  if (n_inactive > 0) {
    gauge_sources.leftCols(n_inactive) = inactive_orbitals;
  }
  if (is_active) {
    const int count =
        stored_sparse_orbital_coefficient_count(input, orbital_index);
    for (int coefficient = 0; coefficient < count; ++coefficient) {
      const int basis = input.orbital_basis_index_table[
          orbital_index * input.n_basis_functions + coefficient] - 1;
      gauge_sources(basis, n_inactive) = input.orbital_value_table[
          orbital_index * input.n_basis_functions + coefficient];
    }
  }
  require_finite_matrix(gauge_sources, "orbital-chart global gauge sources");

  std::vector<char> is_allowed(input.n_basis_functions, 0);
  for (const int basis : local_basis_indices) {
    if (basis < 0 || basis >= input.n_basis_functions) {
      throw std::runtime_error("invalid local AO index in exact quotient");
    }
    is_allowed[basis] = 1;
  }
  int forbidden_count = 0;
  for (const char allowed : is_allowed) {
    if (!allowed) ++forbidden_count;
  }
  Eigen::MatrixXd forbidden_sources(forbidden_count, source_count);
  int forbidden_row = 0;
  for (int basis = 0; basis < input.n_basis_functions; ++basis) {
    if (!is_allowed[basis]) {
      forbidden_sources.row(forbidden_row++) = gauge_sources.row(basis);
    }
  }

  Eigen::MatrixXd admissible_source_parameters;
  if (forbidden_count == 0) {
    admissible_source_parameters =
        Eigen::MatrixXd::Identity(source_count, source_count);
  } else {
    Eigen::JacobiSVD<Eigen::MatrixXd> constraint_svd(
        forbidden_sources, Eigen::ComputeFullV);
    if (constraint_svd.info() != Eigen::Success) {
      throw std::runtime_error(
          "failed to factor exact sparse gauge constraints");
    }
    const double sigma_max = constraint_svd.singularValues().size() > 0
        ? constraint_svd.singularValues()[0]
        : 0.0;
    const double tolerance =
        std::numeric_limits<double>::epsilon() *
        static_cast<double>(
            std::max(forbidden_sources.rows(), forbidden_sources.cols())) *
        sigma_max;
    int constraint_rank = 0;
    for (Eigen::Index index = 0;
         index < constraint_svd.singularValues().size();
         ++index) {
      if (constraint_svd.singularValues()[index] > tolerance) {
        ++constraint_rank;
      }
    }
    admissible_source_parameters =
        constraint_svd.matrixV().rightCols(source_count - constraint_rank);
  }

  Eigen::MatrixXd local_gauge_generators(
      local_size, admissible_source_parameters.cols());
  for (int row = 0; row < local_size; ++row) {
    local_gauge_generators.row(row) =
        gauge_sources.row(local_basis_indices[row]) *
        admissible_source_parameters;
  }
  if (local_gauge_generators.cols() == 0) {
    return Eigen::MatrixXd::Identity(local_size, local_size);
  }

  // LAPACKE requires the actual column-major stride, not a transpose view.
  const Eigen::MatrixXd local_gauge_transpose =
      local_gauge_generators.transpose();
  Eigen::JacobiSVD<Eigen::MatrixXd> gauge_svd(
      local_gauge_transpose, Eigen::ComputeFullV);
  if (gauge_svd.info() != Eigen::Success) {
    throw std::runtime_error("failed to factor exact local sparse gauge");
  }
  const double sigma_max = gauge_svd.singularValues().size() > 0
      ? gauge_svd.singularValues()[0]
      : 0.0;
  const double tolerance =
      std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max(
          local_gauge_generators.rows(),
          local_gauge_generators.cols())) *
      sigma_max;
  int gauge_rank = 0;
  for (Eigen::Index index = 0;
       index < gauge_svd.singularValues().size();
       ++index) {
    if (gauge_svd.singularValues()[index] > tolerance) ++gauge_rank;
  }
  Eigen::MatrixXd quotient_basis =
      gauge_svd.matrixV().rightCols(local_size - gauge_rank);
  for (Eigen::Index column = 0; column < quotient_basis.cols(); ++column) {
    Eigen::Index pivot = 0;
    quotient_basis.col(column).cwiseAbs().maxCoeff(&pivot);
    if (quotient_basis(pivot, column) < 0.0) {
      quotient_basis.col(column) *= -1.0;
    }
  }
  return quotient_basis;
}

// FNV-1a-style hash mixing block/orbital/reduced-size into a running signature.
// Used to detect tangent-space rank changes across SCF iterations so the
// optimizer can invalidate cached Hessians or restart CG when the reduced
// space dimension changes between accepted points.
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

}  // namespace

// ===========================================================================
// Constructor
// ===========================================================================

OrbitalChart::OrbitalChart(
    const OrbitalPreparationInput& input,
    const SparseParameterLayout& parameter_view,
    const Eigen::Ref<const Eigen::MatrixXd>& occ_basis_matrix,
    const Eigen::Ref<const Eigen::MatrixXd>& phys_orbital_matrix,
    const Eigen::MatrixXd* ao_effective_h1e,
    bool collect_structural_diagnostics)
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

  const Eigen::Map<const Eigen::MatrixXd> ao_overlap(
      input.ao_overlap_matrix.data(),
      input.n_basis_functions, input.n_basis_functions);
  require_finite_matrix(ao_overlap, "orbital-chart AO overlap matrix");
  Eigen::MatrixXd global_inactive_orbitals = Eigen::MatrixXd::Zero(
      input.n_basis_functions, n_inactive);
  for (int orbital = 0; orbital < n_inactive; ++orbital) {
    const int count = stored_sparse_orbital_coefficient_count(input, orbital);
    for (int coefficient = 0; coefficient < count; ++coefficient) {
      const int basis = input.orbital_basis_index_table[
          orbital * input.n_basis_functions + coefficient] - 1;
      global_inactive_orbitals(basis, orbital) = input.orbital_value_table[
          orbital * input.n_basis_functions + coefficient];
    }
  }
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

    std::unordered_map<int, int> bf_to_row;
    bf_to_row.reserve(bf_indices.size());
    for (int r = 0; r < nbasis; ++r)
      bf_to_row.emplace(bf_indices[r], r);

    // Blocks group storage only; every gauge uses the global inactive span.
    const int nocc = static_cast<int>(occ_block.size());
    int block_n_inactive = 0;
    std::vector<BlockOrbitalMapping> mappings;
    mappings.reserve(nocc);
    for (int k = 0; k < nocc; ++k) {
      const int orb = occ_block[k];
      mappings.push_back(build_block_orbital_mapping(input, orb, bf_to_row));
      mappings.back().coefficient_count =
          parameter_view.orbital_coefficient_count(orb);
      if (orb < n_inactive) ++block_n_inactive;
    }

    // Block-local overlap.
    Eigen::MatrixXd block_S(nbasis, nbasis);
    for (int i = 0; i < nbasis; ++i)
      for (int j = 0; j < nbasis; ++j)
        block_S(i, j) = ao_overlap(bf_indices[i], bf_indices[j]);

    BlockBasis bb;
    bb.n_inactive = block_n_inactive;
    bb.n_occupied = nocc;
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

      // Construct the quotient directly on the differentiable sparse slots.
      const Eigen::Index local_size =
          static_cast<Eigen::Index>(proj.block_rows.size());
      proj.local_parameter_size = static_cast<int>(local_size);

      // Gather local coefficients.
      Eigen::VectorXd x_p = Eigen::VectorXd::Zero(local_size);
      for (Eigen::Index i = 0; i < local_size; ++i)
        x_p[i] = input.orbital_value_table[proj.flat_indices[i]];
      require_finite_vector(x_p, "orbital-chart local sparse orbital coefficients");

      std::vector<int> local_basis_indices(local_size);
      for (Eigen::Index row = 0; row < local_size; ++row) {
        local_basis_indices[row] = bf_indices[proj.block_rows[row]];
      }
      Eigen::MatrixXd U_p = build_exact_local_sparse_quotient_basis(
          input, m.orbital_index, n_inactive,
          global_inactive_orbitals, local_basis_indices);
      proj.local_gauge_rank = static_cast<int>(local_size - U_p.cols());
      proj.local_combined_rank = static_cast<int>(local_size);
      proj.expected_quotient_dimension = static_cast<int>(U_p.cols());
      if (!U_p.allFinite()) {
        throw std::runtime_error("non-finite strict-sparse quotient basis");
      }
      if (U_p.cols() == 0) {
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
      if (collect_structural_diagnostics) {
        proj.gauge_intersection_dimension =
            proj.local_gauge_rank +
            static_cast<int>(proj.tangent_basis.cols()) -
            proj.local_combined_rank;
        const double local_coefficient_norm = x_p.norm();
        if (local_coefficient_norm > std::numeric_limits<double>::epsilon()) {
          const Eigen::VectorXd scaling_residual =
              x_p - proj.tangent_basis *
                        (proj.tangent_basis.transpose() * x_p);
          proj.relative_scaling_residual =
              scaling_residual.norm() / local_coefficient_norm;
        }
      }
      proj.local_reduced_offset = reduced_size_;
      proj.local_reduced_size = static_cast<int>(proj.tangent_basis.cols());
      reduced_size_ += proj.local_reduced_size;
      rank_signature_ = mix_rank_signature(
          rank_signature_,
          block_index,
          k,
          proj.local_reduced_size);

      // Pull back the normalized, inactive-projected one-electron surrogate
      // into the same additive sparse chart as the exact HVP. Other inactive
      // orbitals remain fixed for an inactive target; all are removed for an
      // active target. Frozen coefficients contribute but have zero tangent.
      if (ao_effective_h1e != nullptr && proj.local_reduced_size > 0) {
        const auto& stored_rows = m.block_row_for_slot;
        const bool inactive_target = m.orbital_index < n_inactive;
        Eigen::MatrixXd fixed_inactive(input.n_basis_functions,
            n_inactive - (inactive_target ? 1 : 0));
        int fixed_column = 0;
        for (int j = 0; j < n_inactive; ++j) {
          if (j != m.orbital_index)
            fixed_inactive.col(fixed_column++) = global_inactive_orbitals.col(j);
        }
        std::vector<int> stored_support(stored_rows.size());
        for (std::size_t j = 0; j < stored_rows.size(); ++j)
          stored_support[j] = bf_indices[stored_rows[j]];
        const auto surrogate = projected_orbital_surrogate(
            *ao_effective_h1e, ao_overlap, fixed_inactive, stored_support);
        Eigen::VectorXd stored_x(stored_rows.size());
        for (Eigen::Index j = 0; j < stored_x.size(); ++j)
          stored_x[j] = input.orbital_value_table[
              m.orbital_index * input.n_basis_functions + j];
        Eigen::MatrixXd stored_U = Eigen::MatrixXd::Zero(
            stored_x.size(), proj.local_reduced_size);
        stored_U.topRows(local_size) = proj.tangent_basis;
        Eigen::MatrixXd block_curv = normalized_orbital_curvature(
            surrogate.one_electron, surrogate.overlap, stored_x, stored_U);
        // The inactive density is an unweighted occupied-space projector;
        // its frozen mean-field energy differential is 2 * tr(F11 dP).
        if (inactive_target) block_curv *= 2.0;
        const Eigen::VectorXd curv = block_curv.diagonal();
        // Positive spectral regularization is used only for preconditioning;
        // negative curvature in the exact HVP model is left untouched.
        proj.curvature_diagonal = normalize_curvature_diagonal(curv);
        PositiveCurvatureBlock positive_block =
            build_positive_curvature_block(block_curv);
        proj.curvature_block = std::move(positive_block.matrix);
        proj.inverse_curvature_block = std::move(positive_block.inverse);
        has_reduced_curvature_diagonal_ = true;
      }

      bb.orbitals.push_back(std::move(proj));
    }

    block_bases_.push_back(std::move(bb));
    ++block_index;
  }
  use_block_preconditioner_by_default_ = has_reduced_curvature_diagonal_;
}

OrbitalChart::StructuralDiagnostics
OrbitalChart::structural_diagnostics() const noexcept {
  StructuralDiagnostics diagnostics;
  diagnostics.packed_parameter_size = packed_parameter_size_;
  diagnostics.reduced_size = reduced_size_;
  for (const auto& block_basis : block_bases_) {
    for (const auto& projector : block_basis.orbitals) {
      ++diagnostics.orbital_count;
      if (projector.local_reduced_size >= projector.local_parameter_size) {
        ++diagnostics.full_local_rank_orbital_count;
      }
      if (projector.local_reduced_size ==
          std::max(0, projector.local_parameter_size - 1)) {
        ++diagnostics.codimension_one_orbital_count;
      }
      if (projector.local_combined_rank < projector.local_parameter_size) {
        ++diagnostics.incomplete_local_span_orbital_count;
      }
      if (projector.gauge_intersection_dimension != 0) {
        ++diagnostics.gauge_intersection_orbital_count;
      }
      if (projector.local_reduced_size !=
          projector.expected_quotient_dimension) {
        ++diagnostics.quotient_dimension_mismatch_orbital_count;
      }
      diagnostics.total_gauge_rank += projector.local_gauge_rank;
      diagnostics.total_expected_quotient_dimension +=
          projector.expected_quotient_dimension;
      diagnostics.minimum_relative_scaling_residual =
          std::min(
              diagnostics.minimum_relative_scaling_residual,
              projector.relative_scaling_residual);
      diagnostics.maximum_relative_scaling_residual =
          std::max(
              diagnostics.maximum_relative_scaling_residual,
              projector.relative_scaling_residual);
    }
  }
  if (diagnostics.orbital_count == 0) {
    diagnostics.minimum_relative_scaling_residual = 0.0;
  }
  return diagnostics;
}

OrbitalChart::ProjectionResult
OrbitalChart::project_impl(
    const Eigen::VectorXd& packed_vector,
    bool recover_tangent_coordinates,
    bool build_packed_projection) const {
  (void)recover_tangent_coordinates;
  require_finite_vector_size(
      packed_vector,
      static_cast<Eigen::Index>(packed_parameter_size_),
      "orbital-chart packed projection input");
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
      assert(projector.tangent_basis.rows() == local_size);
      assert(projector.tangent_basis.cols() == projector.local_reduced_size);
      assert(projector.tangent_basis.allFinite());
      Eigen::VectorXd g_p = Eigen::VectorXd::Zero(local_size);
      for (Eigen::Index i = 0; i < local_size; ++i) {
        assert(projector.packed_indices[i] >= 0 &&
               projector.packed_indices[i] < packed_parameter_size_);
        g_p[i] = packed_vector[projector.packed_indices[i]];
      }
      const Eigen::VectorXd z_p = projector.tangent_basis.transpose() * g_p;
      assert(z_p.allFinite());
      result.reduced_gradient.segment(
          projector.local_reduced_offset,
          projector.local_reduced_size) = z_p;
      if (build_packed_projection) {
        const Eigen::VectorXd dx_p = projector.tangent_basis * z_p;
        assert(dx_p.allFinite());
        for (Eigen::Index i = 0; i < dx_p.size(); ++i)
          result.packed_projected_gradient[projector.packed_indices[i]] += dx_p[i];
      }
    }
  }
  require_finite_vector(result.reduced_gradient, "orbital-chart reduced projection");
  if (build_packed_projection) {
    require_finite_vector(
        result.packed_projected_gradient,
        "orbital-chart packed projection");
  }
  return result;
}

Eigen::VectorXd OrbitalChart::project_reduced_gradient(
    const Eigen::VectorXd& packed_gradient) const {
  return project_impl(packed_gradient, false, false).reduced_gradient;
}

OrbitalChart::ProjectionResult
OrbitalChart::project_gradient(
    const Eigen::VectorXd& packed_gradient) const {
  return project_impl(packed_gradient, false, true);
}

OrbitalChart::ProjectionResult
OrbitalChart::project_vector(
    const Eigen::VectorXd& packed_vector) const {
  return project_impl(packed_vector, true, true);
}

Eigen::VectorXd OrbitalChart::apply_inverse_reduced_curvature(
    const Eigen::VectorXd& reduced_vector) const {
  require_finite_vector_size(
      reduced_vector,
      static_cast<Eigen::Index>(reduced_size_),
      "orbital-chart inverse-curvature input");
  if (!has_reduced_curvature_diagonal_) return reduced_vector;
  Eigen::VectorXd out = reduced_vector;
  constexpr double kMin = 1.0e-12;
  for (const auto& bb : block_bases_) {
    for (const auto& p : bb.orbitals) {
      if (p.curvature_diagonal.size() == 0) continue;
      assert(p.curvature_diagonal.size() == p.local_reduced_size);
      assert(p.curvature_diagonal.allFinite());
      out.segment(p.local_reduced_offset, p.local_reduced_size).array() /=
          p.curvature_diagonal.array().max(kMin);
    }
  }
  return out.allFinite() ? out : reduced_vector;
}

Eigen::VectorXd
OrbitalChart::apply_inverse_reduced_block_preconditioner(
    const Eigen::VectorXd& reduced_vector) const {
  require_finite_vector_size(
      reduced_vector,
      static_cast<Eigen::Index>(reduced_size_),
      "orbital-chart block inverse-curvature input");
  if (!has_reduced_curvature_diagonal_) return reduced_vector;

  Eigen::VectorXd out = reduced_vector;
  for (const auto& bb : block_bases_) {
    for (const auto& p : bb.orbitals) {
      if (p.local_reduced_size <= 0) continue;
      if (p.inverse_curvature_block.rows() != p.local_reduced_size ||
          p.inverse_curvature_block.cols() != p.local_reduced_size ||
          !p.inverse_curvature_block.allFinite()) {
        if (p.curvature_diagonal.size() == p.local_reduced_size &&
            p.curvature_diagonal.allFinite()) {
          out.segment(p.local_reduced_offset, p.local_reduced_size).array() /=
              p.curvature_diagonal.array().max(1.0e-12);
        }
        continue;
      }
      const Eigen::VectorXd solved =
          p.inverse_curvature_block *
          reduced_vector.segment(
              p.local_reduced_offset,
              p.local_reduced_size);
      if (!solved.allFinite()) {
        return apply_inverse_reduced_curvature(reduced_vector);
      }
      out.segment(p.local_reduced_offset, p.local_reduced_size) = solved;
    }
  }
  return out.allFinite() ? out : apply_inverse_reduced_curvature(reduced_vector);
}

Eigen::VectorXd OrbitalChart::apply_reduced_curvature(
    const Eigen::VectorXd& reduced_vector) const {
  require_finite_vector_size(
      reduced_vector,
      static_cast<Eigen::Index>(reduced_size_),
      "orbital-chart curvature input");
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
      assert(p.curvature_diagonal.size() == p.local_reduced_size);
      assert(p.curvature_diagonal.allFinite());
      out.segment(p.local_reduced_offset, p.local_reduced_size).array() *=
          p.curvature_diagonal.array();
    }
  }
  return out.allFinite() ? out : reduced_vector;
}

Eigen::VectorXd OrbitalChart::expand_step(
    const Eigen::VectorXd& reduced_step) const {
  require_finite_vector_size(
      reduced_step,
      static_cast<Eigen::Index>(reduced_size_),
      "orbital-chart reduced expansion input");
  Eigen::VectorXd packed =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(packed_parameter_size_));
  for (const auto& bb : block_bases_) {
    for (const auto& p : bb.orbitals) {
      if (p.local_reduced_size <= 0) continue;
      assert(p.tangent_basis.rows() ==
                 static_cast<Eigen::Index>(p.packed_indices.size()));
      assert(p.tangent_basis.cols() == p.local_reduced_size);
      assert(p.tangent_basis.allFinite());
      const Eigen::VectorXd dx = p.tangent_basis *
          reduced_step.segment(p.local_reduced_offset, p.local_reduced_size);
      assert(dx.allFinite());
      for (Eigen::Index i = 0; i < dx.size(); ++i) {
        assert(p.packed_indices[i] >= 0 &&
               p.packed_indices[i] < packed_parameter_size_);
        packed[p.packed_indices[i]] += dx[i];
      }
    }
  }
  require_finite_vector(packed, "orbital-chart packed expansion");
  return packed;
}

Eigen::VectorXd OrbitalChart::expand_retract_input_tangent(
    const OrbitalPreparationInput& orbital_preparation_input,
    const Eigen::VectorXd& reduced_step) const {
  require_finite_vector_size(
      reduced_step,
      static_cast<Eigen::Index>(reduced_size_),
      "orbital-chart retraction tangent input");
  Eigen::VectorXd tangent = Eigen::VectorXd::Zero(
      static_cast<Eigen::Index>(
          orbital_preparation_input.orbital_value_table.size()));
  for (const auto& bb : block_bases_) {
    for (const auto& p : bb.orbitals) {
      if (p.local_reduced_size <= 0) continue;
      assert(p.tangent_basis.rows() ==
                 static_cast<Eigen::Index>(p.flat_indices.size()));
      assert(p.tangent_basis.cols() == p.local_reduced_size);
      assert(p.tangent_basis.allFinite());
      const Eigen::VectorXd dx = p.tangent_basis *
          reduced_step.segment(p.local_reduced_offset, p.local_reduced_size);
      assert(dx.allFinite());
      for (Eigen::Index i = 0; i < dx.size(); ++i) {
        assert(p.flat_indices[i] >= 0 &&
               p.flat_indices[i] < tangent.size());
        tangent[p.flat_indices[i]] += dx[i];
      }
    }
  }
  require_finite_vector(tangent, "orbital-chart retraction tangent");
  return tangent;
}

OrbitalPreparationInput OrbitalChart::retract_step(
    const OrbitalPreparationInput& orbital_preparation_input,
    const Eigen::VectorXd& reduced_step,
    double step_scale) const {
  if (!std::isfinite(step_scale)) {
    throw std::invalid_argument("orbital-chart retraction step scale is non-finite");
  }

  OrbitalPreparationInput trial = orbital_preparation_input;
  std::vector<double> updated = orbital_preparation_input.orbital_value_table;
  const Eigen::VectorXd tangent =
      expand_retract_input_tangent(
          orbital_preparation_input,
          reduced_step);
  if (tangent.size() != static_cast<Eigen::Index>(updated.size())) {
    throw std::runtime_error("orbital-chart retraction tangent size mismatch");
  }
  for (Eigen::Index i = 0; i < tangent.size(); ++i) {
    updated[static_cast<std::size_t>(i)] += step_scale * tangent[i];
    if (!std::isfinite(updated[static_cast<std::size_t>(i)])) {
      throw std::runtime_error(
          "orbital-chart retraction produced non-finite sparse coefficients");
    }
  }
  trial.orbital_value_table = std::move(updated);
  enforce_strict_sparse_orbital_support(&trial);
  return trial;
}

}  // namespace xmvb::vb
