#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace xmvb::vb {

class ExteriorRowPlan;

/** @brief First derivative of one retained exterior-power transform. */
struct ExteriorTransformDirection {
  Eigen::VectorXd determinant_scales;
  std::vector<double> shear_coefficients;
};

/** @brief Causal shear edges needed for selected rows of one exterior adjoint.
 * The plan belongs to the transform that constructed it; it stores no matrix.
 */
struct ExteriorProjection {
  std::vector<std::vector<int>> shear_pairs;
};

/** @brief Borrowed bilinear generator moment and optional directional panels. */
struct ExteriorGeneratorTerm {
  const Eigen::MatrixXd& seed;
  const Eigen::MatrixXd& image;
  double scale=1.0;
  const Eigen::MatrixXd* delta_seed=nullptr;
  const Eigen::MatrixXd* delta_image=nullptr;
};

/**
 * @brief Exterior-power action of an upper-triangular orbital transform.
 *
 * If nonorthogonal orbitals satisfy `Phi = Chi R`, this object applies
 * `wedge^n R` to fixed-spin determinant coefficients without forming its
 * dense compound matrix. Completeness of the determinant space is mandatory.
 */
class ExteriorOrbitalTransform {
public:
  struct ShearEdge {
    int source = 0;
    int target = 0;
    double sign = 1.0;
  };
  ExteriorOrbitalTransform(
      const std::vector<std::vector<int>>& determinants,
      const Eigen::Ref<const Eigen::MatrixXd>& upper_orbital_transform);

  /** @brief Applies the exterior transform to matrix rows in place. */
  void apply_left(Eigen::MatrixXd* coefficients) const;
  /** @brief Left transform of a full-row view into a shared workspace. */
  void apply_left_block(Eigen::Ref<Eigen::MatrixXd> coefficients) const;
  /** @brief Applies the exterior transform to matrix columns in place. */
  void apply_right(Eigen::MatrixXd* coefficients) const;
  /** @brief Applies the exterior transform to a mutable matrix block in place. */
  void apply_right_block(Eigen::Ref<Eigen::MatrixXd> coefficients) const;
  /** @brief Applies the transpose exterior transform to matrix rows. */
  void apply_adjoint_left(Eigen::MatrixXd* coefficients) const;
  /** @brief Backward dependency closure of requested output rows. */
  ExteriorProjection projection(const std::vector<int>& rows) const;
  /** @brief Exact requested adjoint rows; other output rows are unspecified. */
  void apply_adjoint_left(Eigen::MatrixXd* coefficients, const ExteriorProjection& projection) const;
  /** @brief Selected adjoint rows in a full-row shared workspace view. */
  void apply_adjoint_left_block(Eigen::Ref<Eigen::MatrixXd> coefficients,
      const ExteriorProjection& projection) const;
  /** @brief Left transform within exact selected support, with ordered beta columns.
   * For the adjoint this discards only paths that cannot return to a selected
   * input under the remaining upper-triangular beta adjoint. The CSR rows are
   * the exact depth-zero support of each beta column; consecutive RHS blocks
   * repeat the same beta-column order.
   */
  void apply_supported_left(Eigen::MatrixXd* coefficients,
      std::span<const int> offsets, std::span<const int> rows,
      bool adjoint = false) const;
  /** @brief Applies the transpose exterior transform to matrix columns. */
  void apply_adjoint_right(Eigen::MatrixXd* coefficients) const;
  /** @brief Applies the transpose transform to a mutable matrix block in place. */
  void apply_adjoint_right_block(
      Eigen::Ref<Eigen::MatrixXd> coefficients) const;

  /** @brief Retained rows of the exterior matrix for a contiguous row range.
   * The immutable cache holds at most 64 MiB; borrowed rows remain valid
   * after eviction through the returned shared ownership.
   */
  std::shared_ptr<const Eigen::MatrixXd> row_block(int begin, int count) const;

  /** @brief Requested rows in exact CSC format, sharing the bounded factor LRU.
   * Only exact zeros are omitted. No unrequested beta columns are generated.
   */
  Eigen::SparseMatrix<double> generate_rows(std::span<const int> rows) const;

  /** @brief Applies the inverse exterior transform to rows in place. */
  void apply_inverse_left(Eigen::MatrixXd* coefficients) const;
  /** @brief Applies the inverse transpose to rows in place. */
  void apply_inverse_adjoint_left(Eigen::MatrixXd* coefficients) const;
  /** @brief Applies the inverse exterior transform to a column block in place. */
  void apply_inverse_right_block(Eigen::Ref<Eigen::MatrixXd> coefficients) const;
  /** @brief Applies the inverse transpose to a column block in place. */
  void apply_inverse_adjoint_right_block(Eigen::Ref<Eigen::MatrixXd> coefficients) const;

  /** @brief Linearizes the exterior transform for an upper-triangular `delta R`. */
  ExteriorTransformDirection direction(
      const Eigen::Ref<const Eigen::MatrixXd>& delta_orbital_transform) const;

  /** @brief Applies the transform and its tangent to matrix rows in place. */
  void apply_directional_left(
      const ExteriorTransformDirection& direction,
      Eigen::MatrixXd* coefficients,
      Eigen::MatrixXd* directional_coefficients) const;
  /** @brief Applies the transform and its tangent to matrix columns in place. */
  void apply_directional_right(
      const ExteriorTransformDirection& direction,
      Eigen::MatrixXd* coefficients,
      Eigen::MatrixXd* directional_coefficients) const;
  /** @brief Applies the adjoint transform and its tangent to matrix rows. */
  void apply_directional_adjoint_left(
      const ExteriorTransformDirection& direction,
      Eigen::MatrixXd* coefficients,
      Eigen::MatrixXd* directional_coefficients) const;
  /** @brief Applies the adjoint transform and its tangent to matrix columns. */
  void apply_directional_adjoint_right(
      const ExteriorTransformDirection& direction,
      Eigen::MatrixXd* coefficients,
      Eigen::MatrixXd* directional_coefficients) const;

  int dimension() const noexcept { return static_cast<int>(masks_.size()); }
  const ExteriorRowPlan& row_plan() const noexcept { return *row_plan_; }
  /** Borrowed factors in the exact order used by the CPU exterior action. */
  const Eigen::VectorXd& determinant_scales() const noexcept { return determinant_scales_; }
  std::size_t shear_count() const noexcept { return shears_.size(); }
  double shear_coefficient(std::size_t shear) const { return shears_.at(shear).coefficient; }
  std::span<const ShearEdge> shear_edges(std::size_t shear) const {
    return shears_.at(shear).pairs;
  }
  int n_orbitals() const noexcept { return n_orbitals_; }
  /** @brief Upper-chart pullback of a one-body generator contraction.
   * delta Y = E(delta R R^-1) Y. Inputs are bounded single-spin factor panels.
   * With all three tangent arguments, returns the derivative of this pullback.
   */
  Eigen::MatrixXd generator_pullback(const Eigen::MatrixXd& seed,
      const Eigen::MatrixXd& image,
      const Eigen::MatrixXd* delta_seed = nullptr,
      const Eigen::MatrixXd* delta_image = nullptr,
      const Eigen::MatrixXd* delta_upper = nullptr) const;
  /** @brief Sum bilinear moments before a single upper-chart pullback.
   * Terms share occupations and shear links, not carrier storage. Independent
   * upper entries have exclusive owners; only diagonal orbital vectors are
   * reduced between workers. Both orders of a metric derivative are retained.
   */
  Eigen::MatrixXd generator_pullback(std::span<const ExteriorGeneratorTerm> terms,
      const Eigen::MatrixXd* delta_upper=nullptr) const;
  /** @brief Upper-chart pullback of an already contracted one-body moment. */
  Eigen::MatrixXd moment_pullback(
      const Eigen::Ref<const Eigen::MatrixXd>& moment,
      const Eigen::MatrixXd* delta_moment = nullptr,
      const Eigen::MatrixXd* delta_upper = nullptr) const;
  const Eigen::MatrixXd& orbital_transform() const noexcept { return orbital_transform_; }
  int n_electrons() const noexcept { return n_electrons_; }
  /** @brief Spin-string occupations in the transform's immutable ordering. */
  const std::vector<std::uint64_t>& occupations() const noexcept { return masks_; }
  std::size_t shear_pair_count() const noexcept;
  std::size_t dynamic_bytes() const noexcept;

private:
  struct RowCache;
  struct Shear {
    double coefficient = 0.0;
    std::vector<ShearEdge> pairs;
    /** Source-bit rank maps a support intersection back to the shared edges. */
    std::vector<std::uint64_t> source_words;
    std::vector<int> word_offsets;
  };

  void validate_left(const Eigen::MatrixXd& coefficients) const;
  void validate_right(const Eigen::MatrixXd& coefficients) const;
  void validate_inverse() const;
  /** @brief Ordered shears on disjoint columns; no per-worker carrier copy. */
  void transform_left(Eigen::Ref<Eigen::MatrixXd> coefficients, bool adjoint, bool inverse) const;
  /** @brief Primal and tangent ordered products on independent columns. */
  void transform_directional_left(const ExteriorTransformDirection& direction,
      Eigen::MatrixXd* coefficients, Eigen::MatrixXd* tangent, bool adjoint) const;
  /** @brief Ordered shears on disjoint row ranges of a column-major block. */
  void transform_right(Eigen::Ref<Eigen::MatrixXd> coefficients,
                       bool adjoint, bool inverse) const;

  int n_orbitals_ = 0;
  int n_electrons_ = 0;
  Eigen::MatrixXd orbital_transform_;
  std::vector<std::uint64_t> masks_;
  Eigen::VectorXd determinant_scales_;
  std::vector<Shear> shears_;
  std::shared_ptr<RowCache> row_cache_;
  std::shared_ptr<const ExteriorRowPlan> row_plan_;
};

}  // namespace xmvb::vb
