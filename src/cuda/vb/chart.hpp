#pragma once

#include <vector>

#include "cuda/memory/buffer.hpp"
#include "linalg/views.hpp"

namespace xmvb::cuda {

/** One exact accepted-point OrbitalChart tangent block.
 * basis is column-major [raw_slots.size() x reduced_size], using the
 * unwhitened tangent_basis already selected by the CPU chart. raw_slots are
 * zero-based addresses in the fixed-width nbf x norb raw coefficient table.
 * This is a chart export, not a request to reconstruct a quotient on GPU.
 */
struct OrbitalTangentBlock {
  std::vector<int> raw_slots;
  int reduced_offset = 0;
  int reduced_size = 0;
  std::vector<double> basis;
};

/** GPU application of a frozen accepted-point raw orbital chart.
 * B maps reduced CIAH vectors to raw sparse slots. The exact adjoint B^T
 * projects raw coefficient covectors. Retraction is x+scale*B step, with no
 * normalization or gauge repair: OrbitalGeometry handles normalization in
 * its physical forward map. Padded slots beyond stored_counts are cleared,
 * matching the CPU chart retraction. Rebuild this map whenever the accepted chart
 * changes. All numerical map applications remain on the device.
 */
class OrbitalChartMap {
 public:
  OrbitalChartMap(Context& context,int basis,int orbitals,int reduced_size,
                  const std::vector<int>& stored_counts,
                  const std::vector<OrbitalTangentBlock>& blocks);
  OrbitalChartMap(const OrbitalChartMap&)=delete;
  OrbitalChartMap& operator=(const OrbitalChartMap&)=delete;

  int reduced_size() const noexcept { return reduced_size_; }
  int raw_size() const noexcept { return basis_*orbitals_; }

  /** raw_direction receives B*reduced (nbf x norb); fixed slots are zero. */
  void expand(linalg::ConstMatrixView reduced,
              linalg::MatrixView raw_direction) const;
  /** reduced_covector receives B^T*raw_covector (nreduced x 1). */
  void project(linalg::ConstMatrixView raw_covector,
               linalg::MatrixView reduced_covector) const;
  /** trial receives accepted+scale*B*step, with original fixed slots retained. */
  void retract(linalg::ConstMatrixView accepted,
               linalg::ConstMatrixView reduced_step,double scale,
               linalg::MatrixView trial) const;

 private:
  Context* context_;
  int basis_,orbitals_,reduced_size_,mapped_rows_;
  Buffer<int> raw_slots_,row_block_,row_local_;
  Buffer<int> block_first_row_,block_row_count_,block_first_basis_;
  Buffer<int> block_reduced_offset_,block_reduced_size_;
  Buffer<int> coordinate_block_,coordinate_local_;
  Buffer<int> stored_counts_;
  mutable Buffer<int> retraction_info_;
  Buffer<double> basis_values_;
  mutable Matrix<double> scratch_;
};

}  // namespace xmvb::cuda
