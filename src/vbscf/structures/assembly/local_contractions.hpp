#pragma once

#include <stdexcept>
#include <vector>

#include <Eigen/SparseCore>

#include "vbscf/determinants/pairs/tiles.hpp"

namespace xmvb::vb {

using SparseLocalCoefficientMatrix =
    Eigen::SparseMatrix<double, Eigen::RowMajor, int>;

/**
 * @brief Visits one output tile of a sparse coefficient sandwich.
 *
 * For every nonzero product contributing to `L K R^T`, `consumer` receives
 * the output-tile coordinates, the two partner-kernel coordinates, and the
 * coefficient product. The partner kernel is not stored here, allowing one
 * sparse traversal to accumulate several physical channels in bounded memory.
 */
template <typename Consumer>
inline void for_each_sparse_coefficient_pair_in_tile(
    const SparseLocalCoefficientMatrix& left_coefficients,
    const std::vector<int>& left_primary_support,
    const std::vector<int>& left_partner_support,
    const SparseLocalCoefficientMatrix& right_coefficients,
    const std::vector<int>& right_primary_support,
    const std::vector<int>& right_partner_support,
    int primary_left_begin,
    int primary_left_end,
    int primary_right_begin,
    int primary_right_end,
    Consumer&& consumer) {
  if (left_coefficients.rows() !=
          static_cast<int>(left_primary_support.size()) ||
      left_coefficients.cols() !=
          static_cast<int>(left_partner_support.size()) ||
      right_coefficients.rows() !=
          static_cast<int>(right_primary_support.size()) ||
      right_coefficients.cols() !=
          static_cast<int>(right_partner_support.size())) {
    throw std::invalid_argument(
        "sparse coefficient shapes do not match their supports");
  }

  const SupportWindow left_window = find_support_window(
      left_primary_support,
      primary_left_begin,
      primary_left_end);
  const SupportWindow right_window = find_support_window(
      right_primary_support,
      primary_right_begin,
      primary_right_end);
  for (int right_local = right_window.begin;
       right_local < right_window.end;
       ++right_local) {
    const int tile_column =
        right_primary_support[right_local] - primary_right_begin;
    for (int left_local = left_window.begin;
         left_local < left_window.end;
         ++left_local) {
      const int tile_row =
          left_primary_support[left_local] - primary_left_begin;
      for (SparseLocalCoefficientMatrix::InnerIterator left_entry(
               left_coefficients,
               left_local);
           left_entry;
           ++left_entry) {
        const int partner_left = left_partner_support[left_entry.col()];
        for (SparseLocalCoefficientMatrix::InnerIterator right_entry(
                 right_coefficients,
                 right_local);
             right_entry;
             ++right_entry) {
          consumer(
              tile_row,
              tile_column,
              partner_left,
              right_partner_support[right_entry.col()],
              left_entry.value() * right_entry.value());
        }
      }
    }
  }
}

}  // namespace xmvb::vb
