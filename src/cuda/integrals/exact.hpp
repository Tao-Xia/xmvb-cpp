#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include <cusparse.h>

#include "cuda/memory/buffer.hpp"
#include "cuda/integrals/panel.hpp"
#include "integrals/ao/input.hpp"
#include "linalg/views.hpp"

namespace xmvb::cuda {

class PackedAoPairMatrix;

/** Prepared core MO-diagonal contractions; empty views mean not prepared. */
struct CoreDiagonalView {
    linalg::ConstMatrixView coulomb, response;
};

/** One accepted point's shared exchange/density-response preparation. */
struct ExactAoResponses {
    Buffer<double> exchange, inactive;
    Matrix<double> active, core_coulomb, core_response;
    bool core_ready = false;
};

/** Device-resident exact AO-pair ERI graph and its matrix-free orbital response.
 * All matrix arguments and packed outputs refer to device memory. The graph is
 * uploaded once; CPU work in the constructor only validates and packs metadata.
 * Only one symmetry-aware pair representation is retained, never an expanded
 * AO four-index tensor or a duplicate layout. Point-owned half transforms
 * may be borrowed by consumers; otherwise work uses bounded device row tiles.
 */
class ExactAoPairOperator {
  public:
    ExactAoPairOperator(Context& context,
                        const integrals::AoPairGraph& graph,
                        int n_basis_functions);
    ExactAoPairOperator(const ExactAoPairOperator&) = delete;
    ExactAoPairOperator& operator=(const ExactAoPairOperator&) = delete;
    ~ExactAoPairOperator();

    int basis_functions() const noexcept {
        return n_bf_;
    }
    std::int64_t pair_count() const noexcept {
        return n_pairs_;
    }

    /** target = K source. Panels are [AO pair, channel], row-major. */
    void apply(ConstPairPanel source, PairPanel target) const;

    /** B(C) and optionally dB(C;dC) on a contiguous AO-pair row tile. */
    void pair_features(linalg::ConstMatrixView coefficients,
                       linalg::ConstMatrixView direction,
                       std::int64_t row_begin,
                       PairPanel features,
                       PairPanel directional_features = {}) const;

    /** AO-pair features of rank-one densities C_i C_i^T, one column per
   * orbital. Row-major panels include the off-diagonal AO multiplicity.
   */
    void density_features(linalg::ConstMatrixView coefficients,
                          std::int64_t row_begin,
                          PairPanel features) const;

    /** Generate only target rows of K B(C), optionally K dB(C;dC). */
    void generated_rows(linalg::ConstMatrixView coefficients,
                        linalg::ConstMatrixView direction,
                        std::int64_t row_begin,
                        PairPanel products,
                        PairPanel directional_products = {}) const;

    /** Full active pair kernel B(C)^T K B(C), column-major. */
    void forward(linalg::ConstMatrixView coefficients,
                 linalg::MatrixView kernel,
                 PairPanel products = {}) const;

    /** Reserve a point-owned KB(C) image only when the live budget also admits
   * another point and response workspaces. Empty means device-streamed tiles.
   * forward() fills the image, avoiding a separate preparation contraction.
   */
    Buffer<double> reserve_products(int n_active) const;

    /** Point-owned E(mu,nu;a,b)=(mu a|nu b), triangular in the AO indices.
   * A one-active-index graph sweep and GEMM build this shared exchange image.
   * Empty means the exact device-streamed representation remains in use.
   */
    Buffer<double> prepare_exchange(linalg::ConstMatrixView coefficients,
                                    Matrix<double>* density_projection = nullptr,
                                    linalg::ConstMatrixView targets = {}) const;

    /** Share AO slices between active exchange and core response consumers.
   * Orbitals in mo are ordered core, active, virtual. No integral copy is kept.
   */
    ExactAoResponses
    prepare_responses(linalg::ConstMatrixView active, linalg::ConstMatrixView mo, int n_core) const;

    /** Accepted map vec(V) -> vec(G(V C^T+C V^T) T), column-major.
   * Only the consumed target projections are stored, never an AO^3 image.
   * Empty means exact device-streamed contractions remain necessary.
   */
    Matrix<double> prepare_density_projection(linalg::ConstMatrixView coefficients,
                                              linalg::ConstMatrixView targets) const;

    /** Contract K(C weights C^T) from a prepared exchange image, without ERIs. */
    void exchange_density(ConstPairPanel exchange,
                          linalg::ConstMatrixView weights,
                          linalg::MatrixView output,
                          double scale = -1.0) const;

    /** Add J_B(C)^T K dB (G+G^T), using only the exchange image and GEMM. */
    void exchange_pair_direction(ConstPairPanel exchange,
                                 linalg::ConstMatrixView direction,
                                 linalg::ConstMatrixView kernel_adjoint,
                                 linalg::MatrixView gradient) const;

    /** Core-only J(p,p;c,c) and 2J(p,p;c,c)-K(p,c;p,c) projections.
   * Unique CSR batches rank-one source factors directly;
   * that representation does not retain a full density-response map.
   * Packed pair storage uses one shared half transform; unique CSR applies
   * Coulomb and exchange to the same rank-one factor batch. Both are tiled
   * within the live memory budget and projected directly to MO diagonals.
   * Optional density_projection is the upper-column-packed symmetric map
   * vec(V) -> vec(G(V C^T+C V^T) C), C=mo[:,0:n_core], from the same sweep.
   */
    bool core_diagonal(linalg::ConstMatrixView mo,
                       int n_core,
                       linalg::MatrixView coulomb,
                       linalg::MatrixView response,
                       Buffer<double>* density_projection = nullptr) const;

    /** Full directional pair kernel for dC at the accepted C. */
    void directional(linalg::ConstMatrixView coefficients,
                     linalg::ConstMatrixView direction,
                     linalg::MatrixView kernel_direction,
                     ConstPairPanel products = {}) const;

    /** Pullback of a full Frobenius pair-kernel adjoint into AO-by-active C. */
    void adjoint(linalg::ConstMatrixView coefficients,
                 linalg::ConstMatrixView kernel_adjoint,
                 linalg::MatrixView gradient,
                 ConstPairPanel products = {}) const;

    /** Directional derivative of the pullback, including changing adjoint.
   * Optional component_adjoint shares the generated pair tiles and receives
   * its ordinary pullback; it never starts another integral traversal.
   */
    void directional_adjoint(linalg::ConstMatrixView coefficients,
                             linalg::ConstMatrixView direction,
                             linalg::ConstMatrixView kernel_adjoint,
                             linalg::ConstMatrixView kernel_adjoint_direction,
                             linalg::MatrixView gradient_direction,
                             ConstPairPanel products = {},
                             ConstPairPanel exchange = {},
                             linalg::ConstMatrixView component_adjoint = {},
                             linalg::MatrixView component_gradient = {}) const;

    /** Symmetric AO-density contraction coulomb*J(P)+exchange*K(P).
   * The exact VBSCF density response G(P)=2J(P)-K(P) uses (2,-1).
   * Both input and output are column-major full AO matrices on device.
   */
    void contract_jk(linalg::ConstMatrixView density,
                     linalg::MatrixView output,
                     double coulomb = 2.0,
                     double exchange = -1.0,
                     linalg::MatrixView coulomb_output = {}) const;

    /** Independent symmetric densities, packed as nbf-by-(nbf*channels).
   * One shared graph traversal and SpMM serves the entire device batch.
   */
    void contract_jk_batch(linalg::ConstMatrixView densities,
                           linalg::MatrixView outputs,
                           double coulomb = 2.0,
                           double exchange = -1.0) const;

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

    /** Density pullback of C^T G(P) C for a full active Frobenius adjoint A.
   * Borrow point-owned J/K images when available; never regenerate them here.
   */
    void active_core_density_adjoint(linalg::ConstMatrixView coefficients,
                                     linalg::ConstMatrixView active_adjoint,
                                     linalg::MatrixView density_adjoint,
                                     ConstPairPanel products = {},
                                     ConstPairPanel exchange = {}) const;

    /** Direction of the density pullback under dC and dA. */
    void active_core_density_directional_adjoint(linalg::ConstMatrixView coefficients,
                                                 linalg::ConstMatrixView direction,
                                                 linalg::ConstMatrixView active_adjoint,
                                                 linalg::ConstMatrixView active_adjoint_direction,
                                                 linalg::MatrixView density_adjoint_direction,
                                                 linalg::ConstMatrixView component_adjoint = {},
                                                 linalg::MatrixView component_gradient = {}) const;

  private:
    std::size_t response_capacity() const;
    std::size_t apply_workspace(ConstPairPanel, PairPanel) const;
    void upload_unique(const integrals::AoPairGraph&);
    void unique_exchange(linalg::ConstMatrixView, linalg::MatrixView, double, int) const;
    /** Symmetric rank-one exchange, stored as channels-by-packed-AO-pair. */
    void rank_one_exchange(linalg::ConstMatrixView, linalg::MatrixView) const;
    void unique_half(linalg::ConstMatrixView, int, int, double*) const;
    bool unique_core_diagonal(linalg::ConstMatrixView,
                              int,
                              linalg::MatrixView,
                              linalg::MatrixView) const;
    struct ExchangeRequest {
        linalg::ConstMatrixView coefficients, targets;
        PairPanel image, coulomb;
        linalg::MatrixView response;
        bool diagonal = false, packed = false;
        int first_column = 0;
    };
    std::int64_t exchange_workspace(const ExchangeRequest&) const;
    void project_exchange_half(const ExchangeRequest&, int, const double*, double*) const;
    void build_exchange_images(std::span<const ExchangeRequest>) const;
    Buffer<double> reserve_exchange(linalg::ConstMatrixView) const;
    bool prepare_core(linalg::ConstMatrixView,
                      int,
                      linalg::MatrixView,
                      linalg::MatrixView,
                      Buffer<double>*,
                      const ExchangeRequest* = nullptr) const;
    Matrix<double> reserve_density_projection(linalg::ConstMatrixView coefficients,
                                              linalg::ConstMatrixView targets) const;
    void packed_exchange(linalg::ConstMatrixView densities,
                         linalg::MatrixView outputs,
                         double weight,
                         int channels) const;
    void validate_coefficients(linalg::ConstMatrixView coefficients) const;
    void validate_panel(ConstPairPanel panel) const;
    void validate_panel(PairPanel panel) const;
    void generated_rows_impl(linalg::ConstMatrixView coefficients,
                             linalg::ConstMatrixView direction,
                             std::int64_t row_begin,
                             PairPanel products,
                             PairPanel directional_products,
                             bool direction_only = false) const;

    Context* context_;
    int n_bf_;
    std::int64_t n_pairs_;
    std::int64_t n_integrals_ = 0;
    int exchange_shared_values_ = 0;
    bool compact_ = false;
    bool wide_ = false;
    std::unique_ptr<PackedAoPairMatrix> packed_;
    Buffer<int> pair_first_;
    Buffer<int> pair_second_;
    Buffer<std::int64_t> row_offsets_;
    Buffer<std::int64_t> columns_;
    Buffer<double> values_;
    Buffer<std::int64_t> wide_offsets_, wide_columns_;
    Buffer<double> diagonal_;
    cusparseHandle_t sparse_ = nullptr;
    cusparseSpMatDescr_t csr_ = nullptr;
};

} // namespace xmvb::cuda
