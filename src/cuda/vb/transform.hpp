#pragma once

#include <cstddef>
#include <vector>

#include "cuda/memory/buffer.hpp"
#include "linalg/views.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"

namespace xmvb::cuda {

/** Device replay of the CPU transform's retained diagonal and ordered shears. */
class ExteriorTransform {
 public:
  /** Fixed device schedule; coefficients are updated separately in place. */
  struct Shear {
    std::size_t offset;
    int count;
  };
  ExteriorTransform(Context& context, const vb::ExteriorOrbitalTransform& source);
  ExteriorTransform(const ExteriorTransform&) = delete;
  ExteriorTransform& operator=(const ExteriorTransform&) = delete;

  /** @brief Update a positive upper orbital frame without rebuilding topology.
   * Numerical scales/shears remain device-resident and at fixed addresses so
   * captured actions observe subsequent accepted/trial orbital frames.
   */
  void update_upper(linalg::ConstMatrixView upper);

  /** Apply the exterior transform to spin rows, in place. */
  void apply_left(linalg::MatrixView coefficients, bool adjoint = false,
                  bool inverse = false) const;
  /** Same exact invariant column support as CPU apply_supported_left(). */
  void apply_supported_left(linalg::MatrixView coefficients,std::span<const int> offsets,
      std::span<const int> rows,bool adjoint=false) const;
  /** Apply it to spin columns in each consecutive RHS block, in place. */
  void apply_right(linalg::MatrixView coefficients, bool adjoint = false,
                   bool inverse = false) const;

  /** CPU ExteriorRowPlan recursion, shared verbatim with the host executor.
   * Output column j contains T(rows[j],:)^T. Only the requested single-spin
   * rows and two bounded recursion slabs exist; no product carrier is used.
   */
  void generate_rows(linalg::ConstMatrixView upper,std::span<const int> rows,
                     linalg::MatrixView output) const;

  int dimension() const noexcept { return dimension_; }
  /** Current determinant diagonal scales, resident at a stable device address. */
  const double* scale_data() const noexcept { return scales_.data(); }
  std::size_t resident_bytes() const noexcept {
    return (scales_.size()+inverse_scales_.size()+shear_coefficients_.size()+
            unit_upper_.rows()*unit_upper_.cols())*sizeof(double) +
           (occupations_.size()+binomial_.size())*sizeof(std::uint64_t) +
           column_ranks_.size()*sizeof(int) +
           schedule_.size()*sizeof(Shear) +
           edges_.size()*sizeof(vb::ExteriorOrbitalTransform::ShearEdge);
  }

 private:
  Context* context_;
  int dimension_;
  int n_orbitals_;
  int electrons_;
  int shared_values_=0;
  Buffer<double> scales_;
  Buffer<double> inverse_scales_;
  Buffer<double> shear_coefficients_;
  Buffer<std::uint64_t> occupations_;
  Buffer<std::uint64_t> binomial_;
  Buffer<int> column_ranks_;
  Matrix<double> unit_upper_;
  Buffer<vb::ExteriorOrbitalTransform::ShearEdge> edges_;
  std::vector<Shear> shears_;
  Buffer<Shear> schedule_;
};

}  // namespace xmvb::cuda
