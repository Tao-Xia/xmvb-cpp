#pragma once

#include <cstddef>
#include <cstdint>

#include <cusparse.h>

#include "cuda/memory/buffer.hpp"
#include "integrals/ao/input.hpp"
#include "linalg/views.hpp"

namespace xmvb::cuda {

/** A borrowed row-major AO-pair panel. The leading dimension is in doubles. */
template <class T>
struct BasicPairPanel {
  T* data = nullptr;
  std::int64_t rows = 0;
  std::int64_t cols = 0;
  std::int64_t ld = 0;
};
using PairPanel = BasicPairPanel<double>;
using ConstPairPanel = BasicPairPanel<const double>;

/** Device-resident exact AO-pair ERI graph and its matrix-free orbital response.
 * All matrix arguments and packed outputs refer to device memory. The graph is
 * uploaded once; CPU work in the constructor only validates and packs metadata.
 * No complete AO four-index tensor or AO-pair by active-pair coefficient image
 * is retained by this class. Forward/response work uses bounded row tiles.
 */
class ExactAoPairOperator {
 public:
  ExactAoPairOperator(Context& context, const integrals::AoPairGraph& graph,
                      int n_basis_functions);
  ExactAoPairOperator(const ExactAoPairOperator&) = delete;
  ExactAoPairOperator& operator=(const ExactAoPairOperator&) = delete;
  ~ExactAoPairOperator();

  int basis_functions() const noexcept { return n_bf_; }
  std::int64_t pair_count() const noexcept { return n_pairs_; }

  /** target = K source. Panels are [AO pair, channel], row-major. */
  void apply(ConstPairPanel source, PairPanel target) const;

  /** B(C) and optionally dB(C;dC) on a contiguous AO-pair row tile. */
  void pair_features(linalg::ConstMatrixView coefficients,
                     linalg::ConstMatrixView direction,
                     std::int64_t row_begin, PairPanel features,
                     PairPanel directional_features = {}) const;

  /** Generate only target rows of K B(C), optionally K dB(C;dC). */
  void generated_rows(linalg::ConstMatrixView coefficients,
                      linalg::ConstMatrixView direction,
                      std::int64_t row_begin, PairPanel products,
                      PairPanel directional_products = {}) const;

  /** Full active pair kernel B(C)^T K B(C), column-major. */
  void forward(linalg::ConstMatrixView coefficients,
               linalg::MatrixView kernel) const;

  /** Full directional pair kernel for dC at the accepted C. */
  void directional(linalg::ConstMatrixView coefficients,
                   linalg::ConstMatrixView direction,
                   linalg::MatrixView kernel_direction) const;

  /** Pullback of a full Frobenius pair-kernel adjoint into AO-by-active C. */
  void adjoint(linalg::ConstMatrixView coefficients,
               linalg::ConstMatrixView kernel_adjoint,
               linalg::MatrixView gradient) const;

  /** Directional derivative of the pullback, including changing adjoint. */
  void directional_adjoint(linalg::ConstMatrixView coefficients,
                           linalg::ConstMatrixView direction,
                           linalg::ConstMatrixView kernel_adjoint,
                           linalg::ConstMatrixView kernel_adjoint_direction,
                           linalg::MatrixView gradient_direction) const;


  /** Symmetric AO-density contraction coulomb*J(P)+exchange*K(P).
   * The exact VBSCF density response G(P)=2J(P)-K(P) uses (2,-1).
   * Both input and output are column-major full AO matrices on device.
   */
  void contract_jk(linalg::ConstMatrixView density,
                   linalg::MatrixView output,
                   double coulomb = 2.0, double exchange = -1.0) const;

  /** heff=H+G(P), E=Tr[P(H+heff)]; E is one device double. */
  void core_energy(linalg::ConstMatrixView density,
                   linalg::ConstMatrixView core,
                   linalg::MatrixView effective_core,
                   double* energy) const;

  /** dheff=G(dP), dE=2 Tr(dP heff) at a symmetric accepted density. */
  void core_directional(linalg::ConstMatrixView density_direction,
                        linalg::ConstMatrixView effective_core,
                        linalg::MatrixView effective_core_direction,
                        double* energy_direction) const;

  /** Density pullback of C^T G(P) C for a full active Frobenius adjoint A. */
  void active_core_density_adjoint(linalg::ConstMatrixView coefficients,
                                   linalg::ConstMatrixView active_adjoint,
                                   linalg::MatrixView density_adjoint) const;

  /** Direction of the density pullback under dC and dA. */
  void active_core_density_directional_adjoint(
      linalg::ConstMatrixView coefficients,
      linalg::ConstMatrixView direction,
      linalg::ConstMatrixView active_adjoint,
      linalg::ConstMatrixView active_adjoint_direction,
      linalg::MatrixView density_adjoint_direction) const;

 private:
  void validate_coefficients(linalg::ConstMatrixView coefficients) const;
  void validate_panel(ConstPairPanel panel) const;
  void validate_panel(PairPanel panel) const;
  void generated_rows_impl(linalg::ConstMatrixView coefficients,
                           linalg::ConstMatrixView direction,
                           std::int64_t row_begin, PairPanel products,
                           PairPanel directional_products) const;

  Context* context_;
  int n_bf_;
  std::int64_t n_pairs_;
  std::int64_t n_integrals_ = 0;
  std::int64_t n_unique_ = 0;
  bool compact_ = false;
  Buffer<int> pair_first_;
  Buffer<int> pair_second_;
  Buffer<int> row_offsets_;
  Buffer<int> columns_;
  Buffer<double> values_;
  Buffer<int> integral_rows_;
  Buffer<int> integral_columns_;
  Buffer<int> integral_edges_;
  Buffer<double> integral_values_;
  cusparseHandle_t sparse_ = nullptr;
  cusparseSpMatDescr_t csr_ = nullptr;
};

}  // namespace xmvb::cuda
