#pragma once

#include <cstdint>
#include <array>
#include <functional>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include "vbscf/orbitals/preparation/input.hpp"
#include "vbscf/orbitals/charts/layout.hpp"

namespace xmvb::vb {

struct AnalyticOrbitalDiagonal;
class OrbitalPhysicalMetric;
class RawTangentMetric;

/**
 * @brief Accepted-point quotient chart for VBSCF orbital optimization.
 *
 * The chart removes support-admissible occupied-orbital gauge directions from
 * the packed raw coefficients. Strict-sparse HAO orbitals and full-AO OEO
 * orbitals use the same construction; their different supports determine the
 * local tangent blocks rather than selecting separate optimizer algorithms.
 * The chart owns only accepted-point linear maps and curvature surrogates. It
 * does not evaluate the energy or form the exact Hessian.
 */
class OrbitalChart {
  public:
    /** @brief CIAH uses Unwhitened; existing TN/L-BFGS charts use Whitened. */
    enum class Coordinates { Whitened, Unwhitened };
    /** @brief Positive blocks for TN/L-BFGS, or signed AH inverse only. */
    enum class CurvatureUse { PositiveBlock, AugmentedHessian };
    struct StructuralDiagnostics {
        int packed_parameter_size = 0;
        int reduced_size = 0;
        int orbital_count = 0;
        int full_local_rank_orbital_count = 0;
        int codimension_one_orbital_count = 0;
        int incomplete_local_span_orbital_count = 0;
        int gauge_intersection_orbital_count = 0;
        int quotient_dimension_mismatch_orbital_count = 0;
        int total_gauge_rank = 0;
        int total_expected_quotient_dimension = 0;
        int product_sphere_orbital_count = 0;
        double minimum_relative_scaling_residual = 1.0;
        double maximum_relative_scaling_residual = 0.0;
        double maximum_sphere_tangency_residual = 0.0;
    };

    struct ProjectionResult {
        Eigen::VectorXd reduced_gradient;
        Eigen::VectorXd packed_projected_gradient;
    };

    OrbitalChart(const OrbitalPreparationInput& orbital_preparation_input,
                 const SparseParameterLayout& parameter_view,
                 const Eigen::Ref<const Eigen::MatrixXd>& occupied_orbital_basis_matrix,
                 const Eigen::Ref<const Eigen::MatrixXd>& physical_orbital_matrix,
                 const Eigen::MatrixXd* ao_effective_h1e = nullptr,
                 bool collect_structural_diagnostics = false,
                 bool complete_active_space = false,
                 const AnalyticOrbitalDiagonal* hessian_diagonal = nullptr,
                 Coordinates coordinates = Coordinates::Whitened);

    bool uses_explicit_whitening() const noexcept {
        return whiten_tangent_;
    }

    /** @brief Cheap dual norm in the normalized-orbital AO tangent metric.
   * This is the old Euclidean progress norm in different coordinates, not
   * an inverse of the coupled wavefunction/structure metric.
   */
    double coordinate_dual_norm(const Eigen::VectorXd& covector) const;
    Eigen::VectorXd inverse_coordinate_metric(const Eigen::VectorXd& covector) const;
    /** @brief Established gradient infinity norm without building a whitened chart.
   * Converts covectors only; used by outer stopping and terminal reporting.
   */
    double reference_gradient_inf_norm(const Eigen::VectorXd& covector) const;

    /** @brief Attach curvature to a geometry-only chart at the same orbital point.
   * Does not rebuild tangent bases, gauge complements, or metric factors.
   * Call once before solving for a step; convergence checks need only geometry.
   */
    void set_curvature(const OrbitalPreparationInput& input,
                       const Eigen::MatrixXd* ao_effective_h1e,
                       const AnalyticOrbitalDiagonal* hessian_diagonal,
                       CurvatureUse use = CurvatureUse::PositiveBlock);

    int reduced_size() const noexcept {
        return reduced_size_;
    }

    int n_blocks() const noexcept {
        return static_cast<int>(block_bases_.size());
    }

    /** @brief Whether this chart removes the complete active-space GL gauge. */
    bool uses_active_subspace_gauge() const noexcept {
        return active_subspace_gauge_;
    }

    StructuralDiagnostics structural_diagnostics() const noexcept;

    Eigen::VectorXd project_reduced_gradient(const Eigen::VectorXd& packed_gradient) const;

    ProjectionResult project_gradient(const Eigen::VectorXd& packed_gradient) const;

    /**
   * @brief Projects an ambient vector into this accepted-point chart.
   *
   * This is not a differentiated retraction or a cross-point vector
   * transport. It may initialize a recycled subspace, but exact transported
   * secant identities require a dedicated manifold transport.
   */
    ProjectionResult project_vector(const Eigen::VectorXd& packed_vector) const;

    bool has_reduced_curvature_diagonal() const noexcept {
        return has_reduced_curvature_diagonal_;
    }

    std::uint64_t rank_signature() const noexcept {
        return rank_signature_;
    }

    // Reduced-space algebra expects vectors in the accepted-point orbital-chart chart:
    // packed vectors have `packed_parameter_size_` entries and reduced vectors
    // have `reduced_size_` entries. These routines validate dimensions and
    // finite values before applying the local tangent projectors.
    Eigen::VectorXd apply_inverse_reduced_curvature(const Eigen::VectorXd& reduced_vector) const;

    Eigen::VectorXd apply_reduced_curvature(const Eigen::VectorXd& reduced_vector) const;

    /**
   * @brief Partitions a reduced vector-covector pairing by orbital index.
   *
   * Entries sum to the full pairing. This is a coordinate-block diagnostic,
   * not a decomposition of the coupled physical dual norm.
   */
    Eigen::VectorXd orbital_pairings(const Eigen::VectorXd& vector,
                                     const Eigen::VectorXd& covector) const;

    Eigen::VectorXd
    apply_inverse_reduced_block_preconditioner(const Eigen::VectorXd& reduced_vector) const;

    /** @brief Applies the local block inverse of @f$P+\lambda G@f$ (G=I if whitened). */
    Eigen::VectorXd
    apply_inverse_reduced_shifted_block_preconditioner(const Eigen::VectorXd& reduced_vector,
                                                       double shift) const;

    /** @brief Factors @f$P+\lambda G@f$ once for repeated Krylov actions. */
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)>
    shifted_block_inverse(double shift) const;

    /** @brief Factors exact physical-metric orbital blocks for Riesz PCG. */
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)>
    metric_block_inverse(const OrbitalPhysicalMetric& metric) const;

    /**
   * @brief Applies the signed analytic AH preconditioner in quotient coordinates.
   *
   * Full-AO complete spaces use @f$F^{-1}(D-\omega I)^{-1}F^{-T}@f$,
   * retaining source-orbital mixing. Constrained charts use their signed
   * local pullbacks. Unlike the positive block
   * preconditioner used by MINRES and L-BFGS, this action deliberately keeps
   * the sign of the analytic rotation diagonal before its exact pullback by
   * the quotient-to-rotation Jacobian @f$J@f$.
   */
    Eigen::VectorXd apply_inverse_augmented_hessian_diagonal(const Eigen::VectorXd& covector,
                                                             double eigenvalue) const;

    /** @brief Maximum absolute orthogonal-frame rotation represented by a step. */
    double maximum_rotation_component(const Eigen::VectorXd& reduced_step) const;

    /** @brief Returns the signed analytic Hessian diagonal in reduced order. */
    Eigen::VectorXd augmented_hessian_diagonal() const;

    Eigen::VectorXd expand_step(const Eigen::VectorXd& reduced_step) const;

    /** @brief Lifts a reduced gradient covector into packed coordinates. */
    Eigen::VectorXd expand_gradient(const Eigen::VectorXd& reduced_gradient) const;

    // `expand_retract_input_tangent` returns a full sparse-orbital tangent table,
    // while `retract_step` adds that tangent on the immutable sparse support.
    // Normalization belongs to the downstream orbital preparation map.
    Eigen::VectorXd
    expand_retract_input_tangent(const OrbitalPreparationInput& orbital_preparation_input,
                                 const Eigen::VectorXd& reduced_step) const;

    OrbitalPreparationInput retract_step(const OrbitalPreparationInput& orbital_preparation_input,
                                         const Eigen::VectorXd& reduced_step,
                                         double step_scale = 1.0) const;

  private:
    struct OrbitalProjector {
        int orbital_index = 0;
        std::vector<int> flat_indices;
        std::vector<int> packed_indices;
        std::vector<int> block_rows;
        // S-horizontal representative of the support-admissible quotient. CIAH
        // leaves its AO coordinate metric explicit; TN/L-BFGS normalize it.
        Eigen::MatrixXd tangent_basis;
        // Inverse of L^T S_local L. Vectors use this with L^T S_local;
        // covectors pull back with L^T and lift with the transpose dual map.
        Eigen::MatrixXd inverse_raw_tangent_gram;
        // Only unwhitened charts store these local AO tangent metrics.
        std::shared_ptr<const RawTangentMetric> coordinate_base;
        Eigen::VectorXd coordinate_projection, inverse_coordinate_projection;
        double coordinate_radial_gap = 1.0;
        // Positive diagonal/block approximations of the complete normalized,
        // inactive-projected one-electron curvature in the additive quotient chart.
        Eigen::VectorXd curvature_diagonal;
        // Signed analytic Hessian diagonal for CIAH Davidson.  This must not be
        // replaced by the positive regularization used by SPD preconditioners.
        Eigen::VectorXd hessian_diagonal;
        // Generalized spectral factorization of the signed pullback (J^T D J,G).
        // Shift P-omega*G through its eigenvalues, without refactoring each RHS.
        Eigen::VectorXd ah_block_eigenvalues;
        Eigen::MatrixXd ah_block_eigenvectors;
        Eigen::MatrixXd ah_rotation_components;
        Eigen::VectorXd ah_source_weights;
        /** @brief Inverse external-frame map for a full-AO quotient orbital. */
        Eigen::MatrixXd ah_frame_inverse;
        // Positive spectral regularization of that local surrogate curvature;
        // this is not the exact relaxed VBSCF Hessian block.
        Eigen::MatrixXd curvature_block;
        // Cached inverse action of curvature_block. The accepted-point block is
        // immutable, so factoring it again in every Krylov iteration is wasted
        // cubic work.
        Eigen::MatrixXd inverse_curvature_block;
        int local_parameter_size = 0;
        int local_gauge_rank = 0;
        int local_combined_rank = 0;
        int expected_quotient_dimension = 0;
        int gauge_intersection_dimension = 0;
        bool uses_product_sphere_representative = false;
        double relative_scaling_residual = 0.0;
        double sphere_tangency_residual = 0.0;
        int local_reduced_offset = 0;
        int local_reduced_size = 0;
    };

    struct BlockBasis {
        int n_inactive = 0;
        int n_occupied = 0;
        std::vector<int> basis_function_indices;
        Eigen::MatrixXd block_overlap_matrix;
        std::vector<OrbitalProjector> orbitals;
    };

    /** @brief Apply a borrowed block AO metric on one orbital's allowed slots. */
    Eigen::VectorXd local_metric_image(const BlockBasis& block,
                                       const OrbitalProjector& orbital,
                                       const Eigen::VectorXd& local) const;

    ProjectionResult project_impl(const Eigen::VectorXd& packed_vector,
                                  bool recover_tangent_coordinates,
                                  bool build_packed_projection) const;

    bool supports_rotation_frame(int n_bf) const;
    void build_rotation_inverse(const OrbitalPreparationInput& input,
                                const AnalyticOrbitalDiagonal& model);
    Eigen::VectorXd apply_inverse_rotation_diagonal(const Eigen::VectorXd& covector,
                                                    double eigenvalue) const;
    double maximum_frame_component(const Eigen::VectorXd& step) const;

    std::vector<BlockBasis> block_bases_;
    bool active_subspace_gauge_ = false;
    bool whiten_tangent_ = true;
    int packed_parameter_size_ = 0;
    int reduced_size_ = 0;
    std::uint64_t rank_signature_ = 1469598103934665603ull;
    bool has_reduced_curvature_diagonal_ = false;
    /** @brief Full-AO diagonal model with inter-orbital basis mixing retained. */
    bool has_rotation_inverse_ = false;
    int rotation_core_count_ = 0;
    std::vector<Eigen::MatrixXd> active_internal_bases_;
    Eigen::VectorXd active_internal_eigenvalues_;
    Eigen::MatrixXd active_internal_eigenvectors_;
    std::array<Eigen::MatrixXd, 2> rotation_diagonals_;
    std::array<Eigen::MatrixXd, 2> rotation_source_inverses_;
};

} // namespace xmvb::vb
