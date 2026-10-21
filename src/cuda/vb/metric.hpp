#pragma once

#include <cstddef>
#include <vector>

#include "cuda/memory/buffer.hpp"
#include "linalg/views.hpp"
#include "vbscf/structures/overlap/spin_metric.hpp"

namespace xmvb::vb { class StructureAction; }

namespace xmvb::cuda {

class StructureAction;

/** Device occupation-block Cholesky factors of W^T W.
 * Construction assembles and factors the Grams on the device.
 * Blocks of equal size share a batched triangular solve. The scratch space
 * is one selected-space vector, independent of the number of right-hand sides.
 */
class SpinMetric {
 public:
  using Part = vb::SpinMetric::Part;

  /** Construct occupation Gram factors numerically on the device from sparse W.
   * Host work is limited to occupation grouping and CSC topology packing.
   */
  SpinMetric(Context& context, const vb::StructureAction& source,
             const std::vector<std::vector<int>>& alpha,
             const std::vector<std::vector<int>>& beta,
             std::size_t factor_byte_limit);
  SpinMetric(const SpinMetric&) = delete;
  SpinMetric& operator=(const SpinMetric&) = delete;

  /** Solve G, L, or L^T, where G=L L^T. Input and output may alias. */
  void solve(linalg::ConstMatrixView rhs, linalg::MatrixView output,
             Part part = Part::Gram) const;

  /** Solve (K G K) z=rhs with K_ii=sqrt(model_i/G_ii).
   * model is a device column vector of positive, finite entries, validated by
   * its owner when prepared. Input and output may alias.
   */
  void solve_diagonal_model(linalg::ConstMatrixView rhs,
                            linalg::ConstMatrixView model,
                            linalg::MatrixView output) const;

  int dimension() const noexcept { return dimension_; }
  std::size_t resident_bytes() const noexcept;

 private:
  struct Group {
    int dimension = 0;
    int count = 0;
    std::size_t first_pointer = 0;
    std::size_t first_index = 0;
    std::size_t first_factor = 0;
  };

  void initialize_layout(
      Context& context, const std::vector<std::vector<int>>& blocks,
      std::size_t factor_byte_limit);

  Context* context_;
  int dimension_;
  std::vector<Group> groups_;
  Buffer<int> indices_;
  Buffer<double> factors_;
  Buffer<double> gram_diagonal_;
  mutable Matrix<double> scratch_;
  Buffer<const double*> factor_pointers_;
  Buffer<double*> scratch_pointers_;
};

/** Same-dimensional FULL coordinates, x=U^T T W C with U=W L^-T.
 * The inner H and preconditioner actions use only W, L, and the orthogonal CI
 * sigma/diagonal; orbital maps occur only at the structure-coordinate boundary.
 */
class FullMetric {
 public:
  FullMetric(Context& context, const SpinMetric& spin,
             const StructureAction& action);
  FullMetric(const FullMetric&) = delete;
  FullMetric& operator=(const FullMetric&) = delete;

  void to_ci_coordinates(linalg::ConstMatrixView structure,
                         linalg::MatrixView coordinates) const;
  void from_ci_coordinates(linalg::ConstMatrixView coordinates,
                           linalg::MatrixView structure) const;
  void ci_covector_to_structure(linalg::ConstMatrixView covector,
                                linalg::MatrixView structure) const;
  void apply_ci_hamiltonian(linalg::ConstMatrixView coordinates,
                            linalg::MatrixView image) const;
  /** energies is a device row vector with one value per RHS. */
  void apply_ci_preconditioner(linalg::ConstMatrixView covector,
                               linalg::ConstMatrixView energies,
                               double eigenvalue,
                               linalg::MatrixView result) const;

 private:
  void validate(linalg::ConstMatrixView input,
                linalg::MatrixView output) const;

  const SpinMetric* spin_;
  const StructureAction* action_;
  mutable Matrix<double> structure_;
  mutable Matrix<double> carrier_;
  mutable Matrix<double> image_;
};

}  // namespace xmvb::cuda
