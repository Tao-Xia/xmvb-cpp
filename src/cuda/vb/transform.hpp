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
  /** Apply it to spin columns in each consecutive RHS block, in place. */
  void apply_right(linalg::MatrixView coefficients, bool adjoint = false,
                   bool inverse = false) const;

  int dimension() const noexcept { return dimension_; }
  std::size_t resident_bytes() const noexcept {
    return (scales_.size()+inverse_scales_.size()+shear_coefficients_.size()+
            unit_upper_.rows()*unit_upper_.cols())*sizeof(double) +
           occupations_.size()*sizeof(std::uint64_t) +
           edges_.size()*sizeof(vb::ExteriorOrbitalTransform::ShearEdge);
  }

 private:
  struct Shear {
    std::size_t offset;
    int count;
  };
  Context* context_;
  int dimension_;
  int n_orbitals_;
  Buffer<double> scales_;
  Buffer<double> inverse_scales_;
  Buffer<double> shear_coefficients_;
  Buffer<std::uint64_t> occupations_;
  Matrix<double> unit_upper_;
  Buffer<vb::ExteriorOrbitalTransform::ShearEdge> edges_;
  std::vector<Shear> shears_;
};

}  // namespace xmvb::cuda
