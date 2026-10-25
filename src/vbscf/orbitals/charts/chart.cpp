#include "vbscf/orbitals/charts/chart.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <lapacke.h>
#include "vbscf/orbitals/charts/partition.hpp"
#include "vbscf/orbitals/charts/physical_metric.hpp"
#include "vbscf/orbitals/charts/layout.hpp"
#include "vbscf/orbitals/charts/whitening.hpp"
#include "vbscf/orbitals/charts/raw.hpp"
#include "vbscf/optimization/preconditioners/one_electron.hpp"
#include "vbscf/optimization/preconditioners/hessian_diagonal.hpp"
#include "vbscf/core/contracts/orbital_type.hpp"
#include "core/timing.hpp"

namespace xmvb::vb {

namespace {

void require_finite_matrix(const Eigen::Ref<const Eigen::MatrixXd>& matrix, const char* label) {
    if (!matrix.allFinite()) {
        throw std::runtime_error(std::string(label) + " contains non-finite values");
    }
}

void require_finite_vector(const Eigen::VectorXd& vector, const char* label) {
    if (!vector.allFinite()) {
        throw std::runtime_error(std::string(label) + " contains non-finite values");
    }
}

void require_finite_vector_size(const Eigen::VectorXd& vector,
                                Eigen::Index expected_size,
                                const char* label) {
    if (vector.size() != expected_size) {
        throw std::invalid_argument(std::string(label) + " dimension mismatch");
    }
    require_finite_vector(vector, label);
}

std::vector<int> build_block_basis_function_indices(const OrbitalPreparationInput& input,
                                                    const std::vector<int>& block_orbitals) {
    if (block_orbitals.empty())
        return {};
    std::vector<int> indices;
    indices.reserve(input.n_basis_functions);
    std::vector<char> seen(input.n_basis_functions, 0);
    for (const int orb : block_orbitals) {
        const int ncoeff = stored_sparse_orbital_coefficient_count(input, orb);
        for (int c = 0; c < ncoeff; ++c) {
            const int bf = input.orbital_basis_index_table[orb * input.n_basis_functions + c] - 1;
            if (bf < 0 || bf >= input.n_basis_functions)
                throw std::runtime_error("invalid block basis-function index");
            if (!seen[bf]) {
                seen[bf] = 1;
                indices.push_back(bf);
            }
        }
    }
    return indices;
}

struct BlockOrbitalMapping {
    int orbital_index = 0;
    int coefficient_count = 0;
    std::vector<int> block_row_for_slot;
};

BlockOrbitalMapping
build_block_orbital_mapping(const OrbitalPreparationInput& input,
                            int orbital_index,
                            const std::unordered_map<int, int>& basis_to_block_row) {
    BlockOrbitalMapping m;
    m.orbital_index = orbital_index;
    m.coefficient_count = stored_sparse_orbital_coefficient_count(input, orbital_index);
    m.block_row_for_slot.resize(m.coefficient_count, -1);
    for (int c = 0; c < m.coefficient_count; ++c) {
        const int bf =
            input.orbital_basis_index_table[orbital_index * input.n_basis_functions + c] - 1;
        m.block_row_for_slot[c] = basis_to_block_row.at(bf);
    }
    return m;
}

// --- Non-member helpers used by the constructor ---

Eigen::VectorXd normalize_curvature_diagonal(const Eigen::VectorXd& diag) {
    if (diag.size() == 0)
        return Eigen::VectorXd::Zero(0);
    require_finite_vector(diag, "orbital-chart curvature diagonal");
    Eigen::VectorXd out = diag.cwiseAbs();
    const double spectral_scale = out.maxCoeff();
    if (!(spectral_scale > 0.0)) {
        throw std::runtime_error("orbital-chart curvature diagonal has zero spectral scale");
    }
    const double spectral_floor =
        std::sqrt(std::numeric_limits<double>::epsilon()) * spectral_scale;
    for (Eigen::Index i = 0; i < out.size(); ++i) {
        out[i] = std::max(out[i], spectral_floor);
    }
    return out;
}

struct PositiveCurvatureBlock {
    Eigen::MatrixXd matrix;
    Eigen::MatrixXd inverse;
    Eigen::VectorXd signed_eigenvalues;
    Eigen::MatrixXd eigenvectors;
};

PositiveCurvatureBlock
build_positive_curvature_block(const Eigen::MatrixXd& curvature_block,
                               const Eigen::MatrixXd* coordinate_metric = nullptr) {
    xmvb::profiling::ScopedPhase block_phase(xmvb::profiling::Phase::BlockFactor);
    if (curvature_block.rows() == 0) {
        return {Eigen::MatrixXd::Zero(0, 0), Eigen::MatrixXd::Zero(0, 0)};
    }
    if (curvature_block.rows() != curvature_block.cols() || !curvature_block.allFinite()) {
        throw std::invalid_argument("orbital-chart curvature block must be finite and square");
    }

    Eigen::MatrixXd eigenvectors = 0.5 * (curvature_block + curvature_block.transpose());
    Eigen::VectorXd eigenvalues(curvature_block.rows());
    xmvb::profiling::ScopedPhase eigen_phase(xmvb::profiling::Phase::BlockEigen);
    const lapack_int dimension = static_cast<lapack_int>(eigenvalues.size());
    Eigen::MatrixXd metric;
    if (coordinate_metric)
        metric = *coordinate_metric;
    const lapack_int info = coordinate_metric ? LAPACKE_dsygvd(LAPACK_COL_MAJOR,
                                                               1,
                                                               'V',
                                                               'L',
                                                               dimension,
                                                               eigenvectors.data(),
                                                               dimension,
                                                               metric.data(),
                                                               dimension,
                                                               eigenvalues.data())
                                              : LAPACKE_dsyevd(LAPACK_COL_MAJOR,
                                                               'V',
                                                               'L',
                                                               dimension,
                                                               eigenvectors.data(),
                                                               dimension,
                                                               eigenvalues.data());
    eigen_phase.stop();
    if (info != 0 || !eigenvalues.allFinite() || !eigenvectors.allFinite()) {
        throw std::runtime_error("failed to diagonalize orbital-chart curvature block");
    }

    Eigen::VectorXd safe_eigenvalues = eigenvalues.cwiseAbs();
    const double spectral_scale = safe_eigenvalues.maxCoeff();
    if (!(spectral_scale > 0.0) || !std::isfinite(spectral_scale)) {
        throw std::runtime_error("orbital-chart curvature block has zero spectral scale");
    }
    const double spectral_floor =
        std::sqrt(std::numeric_limits<double>::epsilon()) * spectral_scale;
    for (Eigen::Index i = 0; i < safe_eigenvalues.size(); ++i) {
        safe_eigenvalues[i] = std::max(safe_eigenvalues[i], spectral_floor);
    }
    PositiveCurvatureBlock result;
    const Eigen::MatrixXd covectors =
        coordinate_metric ? (*coordinate_metric * eigenvectors).eval() : eigenvectors;
    result.matrix = covectors * safe_eigenvalues.asDiagonal() * covectors.transpose();
    result.inverse =
        eigenvectors * safe_eigenvalues.cwiseInverse().asDiagonal() * eigenvectors.transpose();
    result.signed_eigenvalues = std::move(eigenvalues);
    result.eigenvectors = std::move(eigenvectors);
    return result;
}

Eigen::MatrixXd build_exact_local_sparse_quotient_basis(
    const OrbitalPreparationInput& input,
    int orbital_index,
    int n_inactive,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_span_basis,
    const Eigen::Ref<const Eigen::MatrixXd>& physical_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& ao_overlap,
    const std::vector<int>& local_basis_indices,
    bool use_active_subspace_gauge) {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::ChartQuotient);
    const int local_size = static_cast<int>(local_basis_indices.size());
    const bool is_active = orbital_index >= n_inactive;
    const int n_active = static_cast<int>(input.n_active_orbitals);
    const int active_gauge_count = is_active ? (use_active_subspace_gauge ? n_active : 1) : 0;
    const int source_count = n_inactive + active_gauge_count;
    if (local_size == 0)
        return Eigen::MatrixXd::Zero(0, 0);
    if (source_count == 0) {
        return Eigen::MatrixXd::Identity(local_size, local_size);
    }

    Eigen::MatrixXd gauge_sources = Eigen::MatrixXd::Zero(input.n_basis_functions, source_count);
    if (n_inactive > 0) {
        gauge_sources.leftCols(n_inactive) = inactive_span_basis;
    }
    if (is_active) {
        const int first_active = n_inactive;
        Eigen::MatrixXd active_residuals =
            use_active_subspace_gauge ? physical_orbitals.middleCols(first_active, n_active).eval()
                                      : physical_orbitals.middleCols(orbital_index, 1).eval();
        if (n_inactive > 0) {
            active_residuals.noalias() -= inactive_span_basis * (inactive_span_basis.transpose() *
                                                                 ao_overlap * active_residuals);
        }
        Eigen::MatrixXd active_metric =
            active_residuals.transpose() * ao_overlap * active_residuals;
        active_metric = (0.5 * (active_metric + active_metric.transpose())).eval();
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> active_solver(active_metric);
        const double active_scale =
            active_solver.info() == Eigen::Success ? active_solver.eigenvalues().maxCoeff() : 0.0;
        const double active_floor = std::numeric_limits<double>::epsilon() *
                                    static_cast<double>(std::max(1, active_gauge_count)) *
                                    active_scale;
        if (active_solver.info() != Eigen::Success || !(active_scale > 0.0) ||
            active_solver.eigenvalues().minCoeff() <= active_floor) {
            throw std::runtime_error(
                "active orbital subspace is rank deficient outside the inactive span");
        }
        const Eigen::MatrixXd active_inverse_square_root =
            active_solver.eigenvectors() *
            active_solver.eigenvalues().cwiseSqrt().cwiseInverse().asDiagonal() *
            active_solver.eigenvectors().transpose();
        gauge_sources.rightCols(active_gauge_count) = active_residuals * active_inverse_square_root;
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
        if (!allowed)
            ++forbidden_count;
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
        admissible_source_parameters = Eigen::MatrixXd::Identity(source_count, source_count);
    } else {
        xmvb::profiling::record_matrix_shape(xmvb::profiling::Phase::ChartConstraintSvd,
                                             forbidden_sources.rows(),
                                             forbidden_sources.cols());
        Eigen::JacobiSVD<Eigen::MatrixXd> constraint_svd(forbidden_sources, Eigen::ComputeFullV);
        if (constraint_svd.info() != Eigen::Success) {
            throw std::runtime_error("failed to factor exact sparse gauge constraints");
        }
        const double sigma_max =
            constraint_svd.singularValues().size() > 0 ? constraint_svd.singularValues()[0] : 0.0;
        const double tolerance =
            std::numeric_limits<double>::epsilon() *
            static_cast<double>(std::max(forbidden_sources.rows(), forbidden_sources.cols())) *
            sigma_max;
        int constraint_rank = 0;
        for (Eigen::Index index = 0; index < constraint_svd.singularValues().size(); ++index) {
            if (constraint_svd.singularValues()[index] > tolerance) {
                ++constraint_rank;
            }
        }
        admissible_source_parameters =
            constraint_svd.matrixV().rightCols(source_count - constraint_rank);
    }

    Eigen::MatrixXd local_gauge_generators(local_size, admissible_source_parameters.cols());
    for (int row = 0; row < local_size; ++row) {
        local_gauge_generators.row(row) =
            gauge_sources.row(local_basis_indices[row]) * admissible_source_parameters;
    }
    if (local_gauge_generators.cols() == 0) {
        return Eigen::MatrixXd::Identity(local_size, local_size);
    }

    // The admissible gauge is unchanged. Choose its S-horizontal complement:
    // G^T S_local U=0, not G^T U=0. Otherwise a small physical tangent can have
    // large raw radial/active-parallel parts and fold its finite additive path.
    Eigen::MatrixXd horizontal_constraints(local_gauge_generators.cols(), local_size);
    if (local_size == input.n_basis_functions) {
        // Borrow the full AO overlap; only gauge-width matrices are temporary.
        // Scatter also supports a full-AO table whose entries are permuted.
        Eigen::MatrixXd gauge =
            Eigen::MatrixXd::Zero(input.n_basis_functions, local_gauge_generators.cols());
        for (int row = 0; row < local_size; ++row)
            gauge.row(local_basis_indices[row]) = local_gauge_generators.row(row);
        const Eigen::MatrixXd covariant_gauge = ao_overlap * gauge;
        for (int row = 0; row < local_size; ++row)
            horizontal_constraints.col(row) =
                covariant_gauge.row(local_basis_indices[row]).transpose();
    } else {
        Eigen::MatrixXd local_overlap(local_size, local_size);
        for (int row = 0; row < local_size; ++row)
            for (int column = 0; column < local_size; ++column)
                local_overlap(row, column) =
                    ao_overlap(local_basis_indices[row], local_basis_indices[column]);
        horizontal_constraints.noalias() = local_gauge_generators.transpose() * local_overlap;
    }
    xmvb::profiling::record_matrix_shape(xmvb::profiling::Phase::ChartGaugeSvd,
                                         horizontal_constraints.rows(),
                                         horizontal_constraints.cols());
    Eigen::JacobiSVD<Eigen::MatrixXd> gauge_svd(horizontal_constraints, Eigen::ComputeFullV);
    if (gauge_svd.info() != Eigen::Success) {
        throw std::runtime_error("failed to factor exact local sparse gauge");
    }
    const double sigma_max =
        gauge_svd.singularValues().size() > 0 ? gauge_svd.singularValues()[0] : 0.0;
    const double tolerance = std::numeric_limits<double>::epsilon() *
                             static_cast<double>(std::max(local_gauge_generators.rows(),
                                                          local_gauge_generators.cols())) *
                             sigma_max;
    int gauge_rank = 0;
    for (Eigen::Index index = 0; index < gauge_svd.singularValues().size(); ++index) {
        if (gauge_svd.singularValues()[index] > tolerance)
            ++gauge_rank;
    }
    Eigen::MatrixXd quotient_basis = gauge_svd.matrixV().rightCols(local_size - gauge_rank);
    for (Eigen::Index column = 0; column < quotient_basis.cols(); ++column) {
        Eigen::Index pivot = 0;
        quotient_basis.col(column).cwiseAbs().maxCoeff(&pivot);
        if (quotient_basis(pivot, column) < 0.0) {
            quotient_basis.col(column) *= -1.0;
        }
    }
    return quotient_basis;
}

Eigen::MatrixXd
build_stable_inactive_span_basis(const Eigen::Ref<const Eigen::MatrixXd>& inactive_orbitals,
                                 const Eigen::Ref<const Eigen::MatrixXd>& ao_overlap) {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::ChartSpan);
    if (inactive_orbitals.cols() == 0)
        return inactive_orbitals;
    const Eigen::MatrixXd metric = inactive_orbitals.transpose() * ao_overlap * inactive_orbitals;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(metric);
    if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite() ||
        !solver.eigenvectors().allFinite()) {
        throw std::runtime_error(
            "failed to factor the inactive span for sparse quotient construction");
    }
    const double spectral_scale = solver.eigenvalues().maxCoeff();
    const double positivity_floor = std::numeric_limits<double>::epsilon() *
                                    static_cast<double>(std::max(metric.rows(), metric.cols())) *
                                    spectral_scale;
    if (!(spectral_scale > 0.0) || solver.eigenvalues().minCoeff() <= positivity_floor) {
        throw std::runtime_error("inactive span is numerically rank deficient in the AO metric");
    }
    const Eigen::MatrixXd inverse_square_root =
        solver.eigenvectors() * solver.eigenvalues().cwiseSqrt().cwiseInverse().asDiagonal() *
        solver.eigenvectors().transpose();
    Eigen::MatrixXd basis = inactive_orbitals * inverse_square_root;
    require_finite_matrix(basis, "stable inactive gauge basis");
    return basis;
}

// FNV-1a-style hash mixing block/orbital/reduced-size into a running signature.
// Used to detect tangent-space rank changes across SCF iterations so the
// optimizer can invalidate cached Hessians or restart CG when the reduced
// space dimension changes between accepted points.
std::uint64_t mix_rank_signature(std::uint64_t signature,
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

} // namespace

// ===========================================================================
// Constructor
// ===========================================================================

OrbitalChart::OrbitalChart(const OrbitalPreparationInput& input,
                           const SparseParameterLayout& parameter_view,
                           const Eigen::Ref<const Eigen::MatrixXd>& occ_basis_matrix,
                           const Eigen::Ref<const Eigen::MatrixXd>& phys_orbital_matrix,
                           const Eigen::MatrixXd* ao_effective_h1e,
                           bool collect_structural_diagnostics,
                           bool complete_active_space,
                           const AnalyticOrbitalDiagonal* hessian_diagonal,
                           Coordinates coordinates)
    : whiten_tangent_(coordinates == Coordinates::Whitened),
      packed_parameter_size_(parameter_view.size()) {
    xmvb::profiling::ScopedPhase chart_phase(xmvb::profiling::Phase::Chart);
    const int n_inactive = (input.n_total_electrons - input.n_active_electrons) / 2;
    const int n_occupied = n_inactive + input.n_active_orbitals;
    if (n_inactive < 0 || n_occupied > input.n_orbitals)
        throw std::invalid_argument("invalid occupied-space partition");
    if (occ_basis_matrix.rows() != input.n_basis_functions || occ_basis_matrix.cols() != n_occupied)
        throw std::invalid_argument("occupied orbital basis matrix shape mismatch");
    if (phys_orbital_matrix.rows() != input.n_basis_functions ||
        phys_orbital_matrix.cols() != input.n_orbitals)
        throw std::invalid_argument("physical orbital matrix shape mismatch");

    const Eigen::Map<const Eigen::MatrixXd> ao_overlap(
        input.ao_overlap_matrix.data(), input.n_basis_functions, input.n_basis_functions);
    require_finite_matrix(ao_overlap, "orbital-chart AO overlap matrix");
    Eigen::MatrixXd global_inactive_orbitals =
        Eigen::MatrixXd::Zero(input.n_basis_functions, n_inactive);
    for (int orbital = 0; orbital < n_inactive; ++orbital) {
        const int count = stored_sparse_orbital_coefficient_count(input, orbital);
        for (int coefficient = 0; coefficient < count; ++coefficient) {
            const int basis =
                input.orbital_basis_index_table[orbital * input.n_basis_functions + coefficient] -
                1;
            global_inactive_orbitals(basis, orbital) =
                input.orbital_value_table[orbital * input.n_basis_functions + coefficient];
        }
    }
    const Eigen::MatrixXd stable_inactive_span_basis =
        build_stable_inactive_span_basis(global_inactive_orbitals, ao_overlap);
    const bool use_active_subspace_gauge =
        complete_active_space && input.orbital_type == kOrbitalTypeOeo;
    active_subspace_gauge_ = use_active_subspace_gauge;
    const auto blocks = detect_orbital_blocks(input);
    // Full-AO inactive orbitals share one gauge. Complete-space OEO active
    // orbitals share a second gauge. Reuse their exact complements only within
    // this accepted point; restricted supports and individual active rays do
    // not share these complements.
    std::optional<Eigen::MatrixXd> full_inactive_quotient;
    std::optional<Eigen::MatrixXd> full_active_quotient;
    std::optional<TangentWhitening> full_inactive_whitening;
    std::optional<TangentWhitening> full_active_whitening;
    std::shared_ptr<const RawTangentMetric> full_inactive_metric, full_active_metric;

    int block_index = 0;
    for (const auto& block : blocks) {
        // Collect occupied orbitals in this block.
        std::vector<int> occ_block;
        for (const int orb : block)
            if (orb < n_occupied)
                occ_block.push_back(orb);
        if (occ_block.empty())
            continue;

        // Block-local AO basis support.
        const std::vector<int> bf_indices = build_block_basis_function_indices(input, block);
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
            mappings.back().coefficient_count = parameter_view.orbital_coefficient_count(orb);
            if (orb < n_inactive)
                ++block_n_inactive;
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
            proj.orbital_index = m.orbital_index;
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
            const Eigen::Index local_size = static_cast<Eigen::Index>(proj.block_rows.size());
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
            bool full_support = local_size == input.n_basis_functions;
            for (Eigen::Index row = 0; row < local_size && full_support; ++row)
                full_support = local_basis_indices[row] == row;
            const bool shared_gauge =
                full_support && (m.orbital_index < n_inactive || use_active_subspace_gauge);
            Eigen::MatrixXd local_quotient;
            const Eigen::MatrixXd* quotient;
            const TangentWhitening* whitening = nullptr;
            std::shared_ptr<const RawTangentMetric> raw_metric;
            if (shared_gauge) {
                auto& shared =
                    m.orbital_index < n_inactive ? full_inactive_quotient : full_active_quotient;
                auto& shared_whitening =
                    m.orbital_index < n_inactive ? full_inactive_whitening : full_active_whitening;
                if (!shared) {
                    shared.emplace(
                        build_exact_local_sparse_quotient_basis(input,
                                                                m.orbital_index,
                                                                n_inactive,
                                                                stable_inactive_span_basis,
                                                                phys_orbital_matrix,
                                                                ao_overlap,
                                                                local_basis_indices,
                                                                use_active_subspace_gauge));
                    if (whiten_tangent_)
                        shared_whitening.emplace(*shared, ao_overlap);
                    else {
                        auto& shared_metric = m.orbital_index < n_inactive ? full_inactive_metric
                                                                           : full_active_metric;
                        shared_metric = std::make_shared<RawTangentMetric>(*shared, ao_overlap);
                    }
                }
                quotient = &*shared;
                if (whiten_tangent_)
                    whitening = &*shared_whitening;
                else
                    raw_metric =
                        (m.orbital_index < n_inactive ? full_inactive_metric : full_active_metric);
            } else {
                local_quotient = build_exact_local_sparse_quotient_basis(input,
                                                                         m.orbital_index,
                                                                         n_inactive,
                                                                         stable_inactive_span_basis,
                                                                         phys_orbital_matrix,
                                                                         ao_overlap,
                                                                         local_basis_indices,
                                                                         use_active_subspace_gauge);
                quotient = &local_quotient;
            }
            const Eigen::MatrixXd& U_p = *quotient;
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
                rank_signature_ =
                    mix_rank_signature(rank_signature_, block_index, k, proj.local_reduced_size);
                bb.orbitals.push_back(std::move(proj));
                continue;
            }

            const int stored_count =
                stored_sparse_orbital_coefficient_count(input, m.orbital_index);
            Eigen::VectorXd stored_coefficients(stored_count);
            Eigen::MatrixXd stored_overlap(stored_count, stored_count);
            Eigen::MatrixXd stored_quotient_basis = Eigen::MatrixXd::Zero(stored_count, U_p.cols());
            stored_quotient_basis.topRows(local_size) = U_p;
            for (int row = 0; row < stored_count; ++row) {
                stored_coefficients[row] =
                    input.orbital_value_table[m.orbital_index * input.n_basis_functions + row];
                const int ao_row =
                    input.orbital_basis_index_table[m.orbital_index * input.n_basis_functions +
                                                    row] -
                    1;
                for (int column = 0; column < stored_count; ++column) {
                    const int ao_column =
                        input.orbital_basis_index_table[m.orbital_index * input.n_basis_functions +
                                                        column] -
                        1;
                    stored_overlap(row, column) = ao_overlap(ao_row, ao_column);
                }
            }
            const bool scaling_gauge_is_admissible =
                stored_count == local_size ||
                stored_coefficients.tail(stored_count - local_size).isZero(0.0);
            const bool use_product_sphere_representative =
                !use_active_subspace_gauge && scaling_gauge_is_admissible;
            if (whiten_tangent_) {
                std::optional<TangentWhitening> local_whitening;
                if (!whitening) {
                    local_whitening.emplace(stored_quotient_basis, stored_overlap);
                    whitening = &*local_whitening;
                }
                WhitenedTangent tangent = whitening->apply(
                    stored_coefficients, stored_overlap, use_product_sphere_representative);
                proj.tangent_basis = tangent.basis.topRows(local_size);
                proj.inverse_raw_tangent_gram = std::move(tangent.inverse_gram);
            } else {
                if (!raw_metric) {
                    raw_metric =
                        std::make_shared<RawTangentMetric>(stored_quotient_basis, stored_overlap);
                }
                RawTangent tangent = raw_metric->apply(stored_quotient_basis,
                                                       stored_coefficients,
                                                       stored_overlap,
                                                       use_product_sphere_representative);
                proj.tangent_basis = tangent.basis.topRows(local_size);
                proj.inverse_raw_tangent_gram = std::move(tangent.inverse_gram);
                proj.coordinate_base = std::move(raw_metric);
                proj.coordinate_projection = std::move(tangent.projection);
                proj.inverse_coordinate_projection = std::move(tangent.inverse_projection);
                proj.coordinate_radial_gap = tangent.radial_gap;
            }
            proj.uses_product_sphere_representative = use_product_sphere_representative;
            if (proj.uses_product_sphere_representative) {
                const Eigen::VectorXd overlap_times_coefficients =
                    stored_overlap * stored_coefficients;
                const Eigen::MatrixXd local_overlap =
                    stored_overlap.topLeftCorner(local_size, local_size);
                const double orbital_metric_norm =
                    std::sqrt(stored_coefficients.dot(overlap_times_coefficients));
                for (Eigen::Index column = 0; column < proj.tangent_basis.cols(); ++column) {
                    const Eigen::VectorXd direction = proj.tangent_basis.col(column);
                    const double direction_metric_norm =
                        std::sqrt(direction.dot(local_overlap * direction));
                    if (orbital_metric_norm > 0.0 && direction_metric_norm > 0.0) {
                        proj.sphere_tangency_residual = std::max(
                            proj.sphere_tangency_residual,
                            std::abs(overlap_times_coefficients.head(local_size).dot(direction)) /
                                (orbital_metric_norm * direction_metric_norm));
                    }
                }
            }
            if (collect_structural_diagnostics) {
                proj.gauge_intersection_dimension = proj.local_gauge_rank +
                                                    static_cast<int>(proj.tangent_basis.cols()) -
                                                    proj.local_combined_rank;
                const double local_coefficient_norm = x_p.norm();
                if (local_coefficient_norm > std::numeric_limits<double>::epsilon()) {
                    const Eigen::VectorXd scaling_residual =
                        x_p - proj.tangent_basis * (proj.inverse_raw_tangent_gram *
                                                    (proj.tangent_basis.transpose() *
                                                     local_metric_image(bb, proj, x_p)));
                    proj.relative_scaling_residual =
                        scaling_residual.norm() / local_coefficient_norm;
                }
            }
            proj.local_reduced_offset = reduced_size_;
            proj.local_reduced_size = static_cast<int>(proj.tangent_basis.cols());
            reduced_size_ += proj.local_reduced_size;
            rank_signature_ =
                mix_rank_signature(rank_signature_, block_index, k, proj.local_reduced_size);

            bb.orbitals.push_back(std::move(proj));
        }

        block_bases_.push_back(std::move(bb));
        ++block_index;
    }
    if (ao_effective_h1e || hessian_diagonal) {
        xmvb::profiling::ScopedPhase curvature_phase(xmvb::profiling::Phase::ChartCurvature);
        set_curvature(input, ao_effective_h1e, hessian_diagonal);
    }
}

void OrbitalChart::set_curvature(const OrbitalPreparationInput& input,
                                 const Eigen::MatrixXd* ao_effective_h1e,
                                 const AnalyticOrbitalDiagonal* hessian_diagonal,
                                 CurvatureUse use) {
    if (ao_effective_h1e != nullptr && (ao_effective_h1e->rows() != input.n_basis_functions ||
                                        ao_effective_h1e->cols() != input.n_basis_functions)) {
        throw std::invalid_argument("AO effective one-electron matrix shape mismatch");
    }
    if (hessian_diagonal != nullptr &&
        (hessian_diagonal->mo_coefficients.rows() != input.n_basis_functions ||
         hessian_diagonal->mo_coefficients.cols() != input.n_basis_functions ||
         hessian_diagonal->rotation_diagonal.rows() != input.n_basis_functions ||
         hessian_diagonal->rotation_diagonal.cols() != input.n_basis_functions)) {
        throw std::invalid_argument("analytic Hessian diagonal shape mismatch");
    }

    const bool rotation_frame =
        hessian_diagonal && supports_rotation_frame(input.n_basis_functions);
    // Full-OEO AH uses the analytic frame inverse, not a local spectral inverse.
    // Positive blocks remain necessary for TN/L-BFGS, state averaging, and the
    // explicitly selected one-electron CIAH preconditioner.
    const bool frame_only =
        rotation_frame && ao_effective_h1e == nullptr && use == CurvatureUse::AugmentedHessian;
    const int n_inactive = (input.n_total_electrons - input.n_active_electrons) / 2;
    const Eigen::Map<const Eigen::MatrixXd> ao_overlap(
        input.ao_overlap_matrix.data(), input.n_basis_functions, input.n_basis_functions);
    Eigen::MatrixXd global_inactive_orbitals;
    std::optional<InactiveOrbitalProjector> inactive_projector;
    if (ao_effective_h1e) {
        global_inactive_orbitals = Eigen::MatrixXd::Zero(input.n_basis_functions, n_inactive);
        for (int orbital = 0; orbital < n_inactive; ++orbital) {
            const int count = stored_sparse_orbital_coefficient_count(input, orbital);
            for (int coefficient = 0; coefficient < count; ++coefficient) {
                const int basis =
                    input.orbital_basis_index_table[orbital * input.n_basis_functions +
                                                    coefficient] -
                    1;
                global_inactive_orbitals(basis, orbital) =
                    input.orbital_value_table[orbital * input.n_basis_functions + coefficient];
            }
        }
        inactive_projector.emplace(*ao_effective_h1e, ao_overlap, global_inactive_orbitals);
    }
    for (auto& block : block_bases_) {
        const auto& bf_indices = block.basis_function_indices;
        for (auto& proj : block.orbitals) {
            if (proj.local_reduced_size == 0)
                continue;
            const Eigen::Index local_size = proj.tangent_basis.rows();
            const int stored_count =
                stored_sparse_orbital_coefficient_count(input, proj.orbital_index);
            // Only materialize the local metric while factoring the preconditioner;
            // full-AO orbitals retain one shared base and two radial vectors each.
            const Eigen::MatrixXd coordinate_metric =
                whiten_tangent_ || frame_only
                    ? Eigen::MatrixXd()
                    : (proj.coordinate_base->metric() -
                       proj.coordinate_projection * proj.coordinate_projection.transpose())
                          .eval();
            // Pull back the normalized, inactive-projected one-electron surrogate
            // into the same additive sparse chart as the exact HVP. Other inactive
            // orbitals remain fixed for an inactive target; all are removed for an
            // active target. Frozen coefficients contribute but have zero tangent.
            if (hessian_diagonal != nullptr) {
                const int target = proj.orbital_index;
                const int local_active = target - n_inactive;
                Eigen::MatrixXd raw_direction =
                    Eigen::MatrixXd::Zero(input.n_basis_functions, proj.local_reduced_size);
                for (Eigen::Index row = 0; row < local_size; ++row) {
                    const int ao = bf_indices[proj.block_rows[row]];
                    raw_direction.row(ao) = proj.tangent_basis.row(row);
                }
                Eigen::VectorXd raw_orbital = Eigen::VectorXd::Zero(input.n_basis_functions);
                for (int row = 0; row < stored_count; ++row) {
                    const int ao =
                        input.orbital_basis_index_table[target * input.n_basis_functions + row] - 1;
                    raw_orbital[ao] =
                        input.orbital_value_table[target * input.n_basis_functions + row];
                }
                const double norm = std::sqrt(raw_orbital.dot(ao_overlap * raw_orbital));
                if (!(norm > 0.0)) {
                    throw std::runtime_error("nonpositive orbital norm");
                }
                const Eigen::VectorXd normalized = raw_orbital / norm;
                const Eigen::RowVectorXd norm_direction =
                    normalized.transpose() * ao_overlap * raw_direction / norm;
                Eigen::MatrixXd physical_direction = raw_direction / norm;
                physical_direction.noalias() -= normalized * norm_direction;

                const auto& mo = hessian_diagonal->mo_coefficients;
                const auto& diagonal = hessian_diagonal->rotation_diagonal;
                const int n_external = input.n_basis_functions - n_inactive;
                Eigen::VectorXd rotation_weights;
                if (target < n_inactive) {
                    const auto& inverse = hessian_diagonal->inactive_inverse_transform;
                    if (inverse.rows() != n_inactive || inverse.cols() != n_inactive) {
                        throw std::invalid_argument(
                            "inactive analytic Hessian transform shape mismatch");
                    }
                    proj.ah_source_weights = inverse.row(target).transpose();
                    rotation_weights = diagonal.block(n_inactive, 0, n_external, n_inactive) *
                                       proj.ah_source_weights.array().square().matrix();
                } else {
                    const auto& inverse = hessian_diagonal->active_inverse_transform;
                    if (local_active < 0 || local_active >= input.n_active_orbitals ||
                        inverse.rows() != input.n_active_orbitals ||
                        inverse.cols() != input.n_active_orbitals) {
                        throw std::invalid_argument(
                            "active analytic Hessian transform shape mismatch");
                    }
                    if (n_inactive > 0) {
                        const auto core = mo.leftCols(n_inactive);
                        physical_direction.noalias() -=
                            core * (core.transpose() * ao_overlap * physical_direction);
                    }
                    proj.ah_source_weights = inverse.row(local_active).transpose();
                    // Complete OEO removes active--active rotations from U_p, whereas
                    // strict-sparse HAO retains the support-admissible ones.  Starting
                    // at the active boundary therefore gives both orbital types the
                    // same analytic model without silently assigning zero curvature to
                    // physical HAO internal-active directions.
                    rotation_weights =
                        diagonal.block(
                            n_inactive, n_inactive, n_external, input.n_active_orbitals) *
                        proj.ah_source_weights.array().square().matrix();
                }
                // Sum source weights before constructing each tangent outer product.
                // All external components share one AO-metric projection and GEMM.
                proj.ah_rotation_components.noalias() =
                    mo.rightCols(n_external).transpose() * ao_overlap * physical_direction;
                const auto& components = proj.ah_rotation_components;
                if (frame_only) {
                    // diag(E^T D E) is a weighted column-square sum; no q-by-q
                    // curvature, eigensystem, positive reconstruction or inverse.
                    proj.hessian_diagonal =
                        (components.array().square().colwise() * rotation_weights.array())
                            .colwise()
                            .sum()
                            .transpose()
                            .matrix();
                    proj.curvature_diagonal = normalize_curvature_diagonal(proj.hessian_diagonal);
                    proj.curvature_block.resize(0, 0);
                    proj.inverse_curvature_block.resize(0, 0);
                    has_reduced_curvature_diagonal_ = true;
                    continue;
                }
                const Eigen::MatrixXd block_curvature =
                    components.transpose() * rotation_weights.asDiagonal() * components;
                const Eigen::VectorXd curvature = block_curvature.diagonal();
                proj.hessian_diagonal = curvature;
                proj.curvature_diagonal = normalize_curvature_diagonal(curvature);
                PositiveCurvatureBlock positive_block = build_positive_curvature_block(
                    block_curvature, whiten_tangent_ ? nullptr : &coordinate_metric);
                proj.ah_block_eigenvalues = std::move(positive_block.signed_eigenvalues);
                proj.ah_block_eigenvectors = std::move(positive_block.eigenvectors);
                proj.curvature_block = std::move(positive_block.matrix);
                proj.inverse_curvature_block = std::move(positive_block.inverse);
                has_reduced_curvature_diagonal_ = true;
            }
            if (ao_effective_h1e != nullptr) {
                const bool inactive_target = proj.orbital_index < n_inactive;
                std::vector<int> stored_support(stored_count);
                for (std::size_t j = 0; j < stored_support.size(); ++j)
                    stored_support[j] =
                        input.orbital_basis_index_table[proj.orbital_index *
                                                            input.n_basis_functions +
                                                        j] -
                        1;
                const auto surrogate = inactive_projector->model(
                    stored_support, inactive_target ? proj.orbital_index : -1);
                Eigen::VectorXd stored_x(stored_count);
                for (Eigen::Index j = 0; j < stored_x.size(); ++j)
                    stored_x[j] =
                        input.orbital_value_table[proj.orbital_index * input.n_basis_functions + j];
                Eigen::MatrixXd stored_U =
                    Eigen::MatrixXd::Zero(stored_x.size(), proj.local_reduced_size);
                stored_U.topRows(local_size) = proj.tangent_basis;
                Eigen::MatrixXd block_curv = normalized_orbital_curvature(
                    surrogate.one_electron, surrogate.overlap, stored_x, stored_U);
                // The inactive density is an unweighted occupied-space projector;
                // its frozen mean-field energy differential is 2 * tr(F11 dP).
                if (inactive_target)
                    block_curv *= 2.0;
                const Eigen::VectorXd curv = block_curv.diagonal();
                // Positive spectral regularization is used only for preconditioning;
                // negative curvature in the exact HVP model is left untouched.
                proj.curvature_diagonal = normalize_curvature_diagonal(curv);
                PositiveCurvatureBlock positive_block = build_positive_curvature_block(
                    block_curv, whiten_tangent_ ? nullptr : &coordinate_metric);
                if (hessian_diagonal != nullptr) {
                    // Keep the physical step map, but honor the explicitly selected
                    // one-electron model for signed AH residual preconditioning.
                    proj.hessian_diagonal = curv;
                    proj.ah_block_eigenvalues = positive_block.signed_eigenvalues;
                    proj.ah_block_eigenvectors = positive_block.eigenvectors;
                }
                proj.curvature_block = std::move(positive_block.matrix);
                proj.inverse_curvature_block = std::move(positive_block.inverse);
                has_reduced_curvature_diagonal_ = true;
            }
        }
    }
    if (rotation_frame)
        build_rotation_inverse(input, *hessian_diagonal);
}

OrbitalChart::StructuralDiagnostics OrbitalChart::structural_diagnostics() const noexcept {
    StructuralDiagnostics diagnostics;
    diagnostics.packed_parameter_size = packed_parameter_size_;
    diagnostics.reduced_size = reduced_size_;
    for (const auto& block_basis : block_bases_) {
        for (const auto& projector : block_basis.orbitals) {
            ++diagnostics.orbital_count;
            if (projector.local_reduced_size >= projector.local_parameter_size) {
                ++diagnostics.full_local_rank_orbital_count;
            }
            if (projector.local_reduced_size == std::max(0, projector.local_parameter_size - 1)) {
                ++diagnostics.codimension_one_orbital_count;
            }
            if (projector.local_combined_rank < projector.local_parameter_size) {
                ++diagnostics.incomplete_local_span_orbital_count;
            }
            if (projector.gauge_intersection_dimension != 0) {
                ++diagnostics.gauge_intersection_orbital_count;
            }
            if (projector.local_reduced_size != projector.expected_quotient_dimension) {
                ++diagnostics.quotient_dimension_mismatch_orbital_count;
            }
            diagnostics.total_gauge_rank += projector.local_gauge_rank;
            diagnostics.total_expected_quotient_dimension += projector.expected_quotient_dimension;
            if (projector.uses_product_sphere_representative) {
                ++diagnostics.product_sphere_orbital_count;
            }
            diagnostics.minimum_relative_scaling_residual = std::min(
                diagnostics.minimum_relative_scaling_residual, projector.relative_scaling_residual);
            diagnostics.maximum_relative_scaling_residual = std::max(
                diagnostics.maximum_relative_scaling_residual, projector.relative_scaling_residual);
            diagnostics.maximum_sphere_tangency_residual = std::max(
                diagnostics.maximum_sphere_tangency_residual, projector.sphere_tangency_residual);
        }
    }
    if (diagnostics.orbital_count == 0) {
        diagnostics.minimum_relative_scaling_residual = 0.0;
    }
    return diagnostics;
}

Eigen::VectorXd OrbitalChart::local_metric_image(const BlockBasis& block,
                                                 const OrbitalProjector& orbital,
                                                 const Eigen::VectorXd& local) const {
    Eigen::VectorXd embedded = Eigen::VectorXd::Zero(block.basis_function_indices.size());
    for (Eigen::Index row = 0; row < local.size(); ++row)
        embedded[orbital.block_rows[row]] = local[row];
    const Eigen::VectorXd image = block.block_overlap_matrix * embedded;
    Eigen::VectorXd result(local.size());
    for (Eigen::Index row = 0; row < local.size(); ++row)
        result[row] = image[orbital.block_rows[row]];
    return result;
}

OrbitalChart::ProjectionResult OrbitalChart::project_impl(const Eigen::VectorXd& packed_vector,
                                                          bool recover_tangent_coordinates,
                                                          bool build_packed_projection) const {
    require_finite_vector_size(packed_vector,
                               static_cast<Eigen::Index>(packed_parameter_size_),
                               "orbital-chart packed projection input");
    ProjectionResult result;
    result.reduced_gradient = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(reduced_size_));
    result.packed_projected_gradient =
        build_packed_projection
            ? Eigen::VectorXd::Zero(static_cast<Eigen::Index>(packed_parameter_size_))
            : Eigen::VectorXd();

    for (const auto& block_basis : block_bases_) {
        for (const auto& projector : block_basis.orbitals) {
            if (projector.local_reduced_size <= 0)
                continue;
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
            if (recover_tangent_coordinates)
                g_p = local_metric_image(block_basis, projector, g_p);
            const Eigen::VectorXd pulled_back = projector.tangent_basis.transpose() * g_p;
            const Eigen::VectorXd z_p = recover_tangent_coordinates
                                            ? projector.inverse_raw_tangent_gram * pulled_back
                                            : pulled_back;
            assert(z_p.allFinite());
            result.reduced_gradient.segment(projector.local_reduced_offset,
                                            projector.local_reduced_size) = z_p;
            if (build_packed_projection) {
                const Eigen::VectorXd projection_coordinates =
                    recover_tangent_coordinates ? z_p : projector.inverse_raw_tangent_gram * z_p;
                Eigen::VectorXd dx_p = projector.tangent_basis * projection_coordinates;
                if (!recover_tangent_coordinates)
                    dx_p = local_metric_image(block_basis, projector, dx_p);
                assert(dx_p.allFinite());
                for (Eigen::Index i = 0; i < dx_p.size(); ++i)
                    result.packed_projected_gradient[projector.packed_indices[i]] += dx_p[i];
            }
        }
    }
    require_finite_vector(result.reduced_gradient, "orbital-chart reduced projection");
    if (build_packed_projection) {
        require_finite_vector(result.packed_projected_gradient, "orbital-chart packed projection");
    }
    return result;
}

Eigen::VectorXd
OrbitalChart::project_reduced_gradient(const Eigen::VectorXd& packed_gradient) const {
    return project_impl(packed_gradient, false, false).reduced_gradient;
}

OrbitalChart::ProjectionResult
OrbitalChart::project_gradient(const Eigen::VectorXd& packed_gradient) const {
    return project_impl(packed_gradient, false, true);
}

OrbitalChart::ProjectionResult
OrbitalChart::project_vector(const Eigen::VectorXd& packed_vector) const {
    return project_impl(packed_vector, true, true);
}

Eigen::VectorXd OrbitalChart::orbital_pairings(const Eigen::VectorXd& vector,
                                               const Eigen::VectorXd& covector) const {
    require_finite_vector_size(vector, reduced_size_, "orbital pairing vector");
    require_finite_vector_size(covector, reduced_size_, "orbital pairing covector");
    int count = 0;
    for (const auto& block : block_bases_)
        for (const auto& orbital : block.orbitals)
            count = std::max(count, orbital.orbital_index + 1);
    Eigen::VectorXd result = Eigen::VectorXd::Zero(count);
    for (const auto& block : block_bases_) {
        for (const auto& orbital : block.orbitals) {
            result[orbital.orbital_index] +=
                vector.segment(orbital.local_reduced_offset, orbital.local_reduced_size)
                    .dot(
                        covector.segment(orbital.local_reduced_offset, orbital.local_reduced_size));
        }
    }
    return result;
}

Eigen::VectorXd
OrbitalChart::apply_inverse_reduced_curvature(const Eigen::VectorXd& reduced_vector) const {
    require_finite_vector_size(reduced_vector,
                               static_cast<Eigen::Index>(reduced_size_),
                               "orbital-chart inverse-curvature input");
    if (!has_reduced_curvature_diagonal_)
        return reduced_vector;
    Eigen::VectorXd out = reduced_vector;
    constexpr double kMin = 1.0e-12;
    for (const auto& bb : block_bases_) {
        for (const auto& p : bb.orbitals) {
            if (p.curvature_diagonal.size() == 0)
                continue;
            assert(p.curvature_diagonal.size() == p.local_reduced_size);
            assert(p.curvature_diagonal.allFinite());
            out.segment(p.local_reduced_offset, p.local_reduced_size).array() /=
                p.curvature_diagonal.array().max(kMin);
        }
    }
    require_finite_vector(out, "orbital-chart inverse-curvature result");
    return out;
}

Eigen::VectorXd OrbitalChart::apply_inverse_reduced_block_preconditioner(
    const Eigen::VectorXd& reduced_vector) const {
    return apply_inverse_reduced_shifted_block_preconditioner(reduced_vector, 0.0);
}

Eigen::VectorXd OrbitalChart::apply_inverse_reduced_shifted_block_preconditioner(
    const Eigen::VectorXd& reduced_vector, double shift) const {
    require_finite_vector_size(reduced_vector,
                               static_cast<Eigen::Index>(reduced_size_),
                               "orbital-chart block inverse-curvature input");
    if (!(shift >= 0.0) || !std::isfinite(shift)) {
        throw std::invalid_argument(
            "orbital-chart preconditioner shift must be finite and nonnegative");
    }
    if (!has_reduced_curvature_diagonal_)
        return reduced_vector;

    if (shift != 0.0) {
        return shifted_block_inverse(shift)(reduced_vector);
    }

    Eigen::VectorXd out = reduced_vector;
    for (const auto& bb : block_bases_) {
        for (const auto& p : bb.orbitals) {
            if (p.local_reduced_size <= 0)
                continue;
            if (p.inverse_curvature_block.rows() != p.local_reduced_size ||
                p.inverse_curvature_block.cols() != p.local_reduced_size ||
                p.curvature_block.rows() != p.local_reduced_size ||
                p.curvature_block.cols() != p.local_reduced_size ||
                !p.curvature_block.allFinite() || !p.inverse_curvature_block.allFinite()) {
                throw std::runtime_error("orbital-chart inverse-curvature block is inconsistent");
            }
            const auto local_vector =
                reduced_vector.segment(p.local_reduced_offset, p.local_reduced_size);
            const Eigen::VectorXd solved = p.inverse_curvature_block * local_vector;
            if (!solved.allFinite()) {
                throw std::runtime_error(
                    "orbital-chart block preconditioner returned non-finite values");
            }
            out.segment(p.local_reduced_offset, p.local_reduced_size) = solved;
        }
    }
    require_finite_vector(out, "orbital-chart block preconditioner result");
    return out;
}

std::function<Eigen::VectorXd(const Eigen::VectorXd&)>
OrbitalChart::shifted_block_inverse(double shift) const {
    if (!(shift > 0.0) || !std::isfinite(shift)) {
        throw std::invalid_argument("shifted block inverse requires a positive shift");
    }

    struct Block {
        Eigen::Index offset;
        Eigen::LDLT<Eigen::MatrixXd> factor;
    };
    std::vector<Block> blocks;
    if (has_reduced_curvature_diagonal_) {
        for (const auto& basis : block_bases_) {
            for (const auto& orbital : basis.orbitals) {
                if (orbital.local_reduced_size <= 0)
                    continue;
                Eigen::MatrixXd matrix = orbital.curvature_block;
                if (matrix.rows() != orbital.local_reduced_size ||
                    matrix.cols() != orbital.local_reduced_size || !matrix.allFinite()) {
                    throw std::runtime_error("invalid shifted orbital curvature block");
                }
                if (whiten_tangent_)
                    matrix.diagonal().array() += shift;
                else
                    matrix.noalias() += shift * (orbital.coordinate_base->metric() -
                                                 orbital.coordinate_projection *
                                                     orbital.coordinate_projection.transpose());
                Eigen::LDLT<Eigen::MatrixXd> factor(matrix);
                if (factor.info() != Eigen::Success || !factor.isPositive()) {
                    throw std::runtime_error("failed to factor shifted orbital curvature block");
                }
                blocks.push_back({orbital.local_reduced_offset, std::move(factor)});
            }
        }
    }
    const Eigen::Index size = reduced_size_;
    return [size, blocks = std::move(blocks)](const Eigen::VectorXd& reduced_vector) {
        require_finite_vector_size(reduced_vector, size, "orbital-chart block inverse input");
        Eigen::VectorXd out = reduced_vector;
        for (const Block& block : blocks) {
            const Eigen::Index length = block.factor.rows();
            out.segment(block.offset, length) =
                block.factor.solve(reduced_vector.segment(block.offset, length));
        }
        require_finite_vector(out, "orbital-chart block inverse result");
        return out;
    };
}

Eigen::VectorXd OrbitalChart::apply_reduced_curvature(const Eigen::VectorXd& reduced_vector) const {
    require_finite_vector_size(
        reduced_vector, static_cast<Eigen::Index>(reduced_size_), "orbital-chart curvature input");
    if (!has_reduced_curvature_diagonal_)
        return reduced_vector;
    Eigen::VectorXd out = reduced_vector;
    for (const auto& bb : block_bases_) {
        for (const auto& p : bb.orbitals) {
            if (p.curvature_block.rows() == p.local_reduced_size &&
                p.curvature_block.cols() == p.local_reduced_size && p.curvature_block.allFinite()) {
                out.segment(p.local_reduced_offset, p.local_reduced_size).noalias() =
                    p.curvature_block *
                    reduced_vector.segment(p.local_reduced_offset, p.local_reduced_size);
                continue;
            }
            if (p.local_reduced_size > 0)
                throw std::logic_error("positive curvature action requires positive blocks");
        }
    }
    return out.allFinite() ? out : reduced_vector;
}

Eigen::VectorXd
OrbitalChart::apply_inverse_augmented_hessian_diagonal(const Eigen::VectorXd& covector,
                                                       double eigenvalue) const {
    require_finite_vector_size(covector,
                               static_cast<Eigen::Index>(reduced_size_),
                               "orbital-chart augmented-Hessian preconditioner input");
    if (!std::isfinite(eigenvalue)) {
        throw std::invalid_argument("augmented-Hessian eigenvalue must be finite");
    }
    if (!has_reduced_curvature_diagonal_) {
        throw std::logic_error("augmented-Hessian preconditioning requires an analytic diagonal");
    }

    if (has_rotation_inverse_)
        return apply_inverse_rotation_diagonal(covector, eigenvalue);

    Eigen::VectorXd out = covector;
    constexpr double kDenominatorFloor = 1.0e-8;
    for (const auto& block : block_bases_) {
        for (const auto& orbital : block.orbitals) {
            if (orbital.local_reduced_size <= 0)
                continue;
            if (orbital.hessian_diagonal.size() != orbital.local_reduced_size ||
                !orbital.hessian_diagonal.allFinite() ||
                orbital.ah_block_eigenvalues.size() != orbital.local_reduced_size ||
                orbital.ah_block_eigenvectors.rows() != orbital.local_reduced_size ||
                orbital.ah_block_eigenvectors.cols() != orbital.local_reduced_size ||
                !orbital.ah_block_eigenvalues.allFinite() ||
                !orbital.ah_block_eigenvectors.allFinite()) {
                throw std::runtime_error("orbital-chart analytic Hessian diagonal is inconsistent");
            }
            const Eigen::VectorXd local =
                covector.segment(orbital.local_reduced_offset, orbital.local_reduced_size);
            Eigen::VectorXd denominator = orbital.ah_block_eigenvalues.array() - eigenvalue;
            for (Eigen::Index i = 0; i < denominator.size(); ++i) {
                if (std::abs(denominator[i]) < kDenominatorFloor) {
                    denominator[i] = kDenominatorFloor;
                }
            }
            out.segment(orbital.local_reduced_offset, orbital.local_reduced_size).noalias() =
                orbital.ah_block_eigenvectors *
                (orbital.ah_block_eigenvectors.transpose() * local).cwiseQuotient(denominator);
        }
    }
    require_finite_vector(out, "orbital-chart augmented-Hessian preconditioner result");
    return out;
}

double OrbitalChart::maximum_rotation_component(const Eigen::VectorXd& reduced_step) const {
    require_finite_vector_size(
        reduced_step, static_cast<Eigen::Index>(reduced_size_), "orbital-chart AH step");
    if (has_rotation_inverse_)
        return maximum_frame_component(reduced_step);
    double maximum = 0.0;
    for (const auto& block : block_bases_) {
        for (const auto& orbital : block.orbitals) {
            if (orbital.local_reduced_size <= 0)
                continue;
            if (orbital.ah_rotation_components.cols() != orbital.local_reduced_size ||
                orbital.ah_source_weights.size() == 0 ||
                !orbital.ah_rotation_components.allFinite() ||
                !orbital.ah_source_weights.allFinite()) {
                throw std::runtime_error("orbital-chart rotation map is unavailable");
            }
            const Eigen::VectorXd external_components =
                orbital.ah_rotation_components *
                reduced_step.segment(orbital.local_reduced_offset, orbital.local_reduced_size);
            maximum = std::max(maximum,
                               external_components.cwiseAbs().maxCoeff() *
                                   orbital.ah_source_weights.cwiseAbs().maxCoeff());
        }
    }
    return maximum;
}

Eigen::VectorXd OrbitalChart::augmented_hessian_diagonal() const {
    if (!has_reduced_curvature_diagonal_) {
        throw std::logic_error("analytic augmented-Hessian diagonal is unavailable");
    }
    Eigen::VectorXd diagonal(reduced_size_);
    for (const auto& block : block_bases_) {
        for (const auto& orbital : block.orbitals) {
            if (orbital.local_reduced_size <= 0)
                continue;
            if (orbital.hessian_diagonal.size() != orbital.local_reduced_size ||
                !orbital.hessian_diagonal.allFinite()) {
                throw std::runtime_error("orbital-chart analytic Hessian diagonal is inconsistent");
            }
            diagonal.segment(orbital.local_reduced_offset, orbital.local_reduced_size) =
                orbital.hessian_diagonal;
        }
    }
    return diagonal;
}

std::function<Eigen::VectorXd(const Eigen::VectorXd&)>
OrbitalChart::metric_block_inverse(const OrbitalPhysicalMetric& metric) const {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::MetricBlock);
    struct Block {
        int offset;
        Eigen::MatrixXd inverse;
    };
    std::vector<Block> blocks;
    for (const auto& group : block_bases_) {
        for (const auto& orbital : group.orbitals) {
            if (orbital.local_reduced_size == 0)
                continue;
            std::vector<int> support(orbital.block_rows.size());
            for (std::size_t row = 0; row < support.size(); ++row)
                support[row] = group.basis_function_indices[orbital.block_rows[row]];
            const Eigen::MatrixXd gram =
                metric.orbital_block(orbital.orbital_index, support, orbital.tangent_basis);
            Eigen::LLT<Eigen::MatrixXd> factor(gram);
            if (factor.info() != Eigen::Success)
                throw std::runtime_error("physical metric orbital block is not positive definite");
            blocks.push_back({orbital.local_reduced_offset,
                              factor.solve(Eigen::MatrixXd::Identity(gram.rows(), gram.cols()))});
            if (!blocks.back().inverse.allFinite())
                throw std::runtime_error("non-finite physical metric block inverse");
        }
    }
    return [blocks = std::move(blocks), size = reduced_size_](const Eigen::VectorXd& rhs) {
        require_finite_vector_size(rhs, size, "physical metric block inverse input");
        Eigen::VectorXd result(size);
        for (const auto& block : blocks)
            result.segment(block.offset, block.inverse.rows()).noalias() =
                block.inverse * rhs.segment(block.offset, block.inverse.rows());
        return result;
    };
}

Eigen::VectorXd OrbitalChart::expand_step(const Eigen::VectorXd& reduced_step) const {
    require_finite_vector_size(reduced_step,
                               static_cast<Eigen::Index>(reduced_size_),
                               "orbital-chart reduced expansion input");
    Eigen::VectorXd packed =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(packed_parameter_size_));
    for (const auto& bb : block_bases_) {
        for (const auto& p : bb.orbitals) {
            if (p.local_reduced_size <= 0)
                continue;
            assert(p.tangent_basis.rows() == static_cast<Eigen::Index>(p.packed_indices.size()));
            assert(p.tangent_basis.cols() == p.local_reduced_size);
            assert(p.tangent_basis.allFinite());
            const Eigen::VectorXd dx =
                p.tangent_basis *
                reduced_step.segment(p.local_reduced_offset, p.local_reduced_size);
            assert(dx.allFinite());
            for (Eigen::Index i = 0; i < dx.size(); ++i) {
                assert(p.packed_indices[i] >= 0 && p.packed_indices[i] < packed_parameter_size_);
                packed[p.packed_indices[i]] += dx[i];
            }
        }
    }
    require_finite_vector(packed, "orbital-chart packed expansion");
    return packed;
}

Eigen::VectorXd OrbitalChart::inverse_coordinate_metric(const Eigen::VectorXd& covector) const {
    require_finite_vector_size(covector, reduced_size_, "coordinate metric covector");
    if (whiten_tangent_)
        return covector;
    Eigen::VectorXd result(covector.size());
    for (const auto& block : block_bases_)
        for (const auto& orbital : block.orbitals) {
            const int start = orbital.local_reduced_offset, size = orbital.local_reduced_size;
            if (size) {
                const auto local = covector.segment(start, size);
                result.segment(start, size).noalias() = orbital.coordinate_base->inverse() * local;
                result.segment(start, size).noalias() +=
                    orbital.inverse_coordinate_projection *
                    (orbital.inverse_coordinate_projection.dot(local) /
                     orbital.coordinate_radial_gap);
            }
        }
    return result;
}

double OrbitalChart::coordinate_dual_norm(const Eigen::VectorXd& covector) const {
    if (whiten_tangent_)
        return covector.norm();
    const double squared = covector.dot(inverse_coordinate_metric(covector));
    if (!(squared >= 0.0) || !std::isfinite(squared))
        throw std::runtime_error("invalid coordinate dual norm");
    return std::sqrt(squared);
}

double OrbitalChart::reference_gradient_inf_norm(const Eigen::VectorXd& covector) const {
    require_finite_vector_size(covector, reduced_size_, "reference gradient covector");
    if (!covector.size())
        return 0.0;
    if (whiten_tangent_)
        return covector.cwiseAbs().maxCoeff();
    double maximum = 0.0;
    for (const auto& block : block_bases_)
        for (const auto& orbital : block.orbitals) {
            if (!orbital.local_reduced_size)
                continue;
            const Eigen::VectorXd reference = orbital.coordinate_base->reference_covector(
                covector.segment(orbital.local_reduced_offset, orbital.local_reduced_size),
                orbital.coordinate_projection);
            maximum = std::max(maximum, reference.cwiseAbs().maxCoeff());
        }
    return maximum;
}

Eigen::VectorXd OrbitalChart::expand_gradient(const Eigen::VectorXd& reduced_gradient) const {
    require_finite_vector_size(reduced_gradient,
                               static_cast<Eigen::Index>(reduced_size_),
                               "orbital-chart reduced-gradient expansion input");
    Eigen::VectorXd packed =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(packed_parameter_size_));
    for (const auto& block_basis : block_bases_) {
        for (const auto& projector : block_basis.orbitals) {
            if (projector.local_reduced_size <= 0)
                continue;
            const Eigen::VectorXd representative =
                projector.tangent_basis * (projector.inverse_raw_tangent_gram *
                                           reduced_gradient.segment(projector.local_reduced_offset,
                                                                    projector.local_reduced_size));
            const Eigen::VectorXd local =
                local_metric_image(block_basis, projector, representative);
            for (Eigen::Index row = 0; row < local.size(); ++row) {
                packed[projector.packed_indices[row]] += local[row];
            }
        }
    }
    require_finite_vector(packed, "orbital-chart packed-gradient expansion");
    return packed;
}

Eigen::VectorXd
OrbitalChart::expand_retract_input_tangent(const OrbitalPreparationInput& orbital_preparation_input,
                                           const Eigen::VectorXd& reduced_step) const {
    require_finite_vector_size(reduced_step,
                               static_cast<Eigen::Index>(reduced_size_),
                               "orbital-chart retraction tangent input");
    Eigen::VectorXd tangent = Eigen::VectorXd::Zero(
        static_cast<Eigen::Index>(orbital_preparation_input.orbital_value_table.size()));
    for (const auto& bb : block_bases_) {
        for (const auto& p : bb.orbitals) {
            if (p.local_reduced_size <= 0)
                continue;
            assert(p.tangent_basis.rows() == static_cast<Eigen::Index>(p.flat_indices.size()));
            assert(p.tangent_basis.cols() == p.local_reduced_size);
            assert(p.tangent_basis.allFinite());
            const Eigen::VectorXd dx =
                p.tangent_basis *
                reduced_step.segment(p.local_reduced_offset, p.local_reduced_size);
            assert(dx.allFinite());
            for (Eigen::Index i = 0; i < dx.size(); ++i) {
                assert(p.flat_indices[i] >= 0 && p.flat_indices[i] < tangent.size());
                tangent[p.flat_indices[i]] += dx[i];
            }
        }
    }
    require_finite_vector(tangent, "orbital-chart retraction tangent");
    return tangent;
}

OrbitalPreparationInput
OrbitalChart::retract_step(const OrbitalPreparationInput& orbital_preparation_input,
                           const Eigen::VectorXd& reduced_step,
                           double step_scale) const {
    if (!std::isfinite(step_scale)) {
        throw std::invalid_argument("orbital-chart retraction step scale is non-finite");
    }

    OrbitalPreparationInput trial = orbital_preparation_input;
    std::vector<double> updated = orbital_preparation_input.orbital_value_table;
    const Eigen::VectorXd tangent =
        expand_retract_input_tangent(orbital_preparation_input, reduced_step);
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

} // namespace xmvb::vb
