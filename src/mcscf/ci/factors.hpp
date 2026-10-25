#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include <Eigen/Core>

#include "mcscf/ci/sigma.hpp"
#include "core/workspace.hpp"

namespace xmvb::mcscf {

inline constexpr std::size_t kFactorWorkspaceBytes = 64ULL * 1024ULL * 1024ULL;

/** @brief Exact orthogonal spin factors supplied a bounded source panel at a time.
 * load(first,count,A,B) fills the borrowed full-row views so that the represented
 * amplitude is sum_j A_j B_j^T. No complete product or all-source array is needed.
 */
struct SpinFactorSource {
    int n_alpha = 0, n_beta = 0, count = 0;
    std::function<void(int, int, Eigen::Ref<Eigen::MatrixXd>, Eigen::Ref<Eigen::MatrixXd>)> load;
    /** Temporary doubles per source column during load; excludes output views. */
    std::size_t workspace_rows = 0;
    /** Nominal scalar operations/element touches per column; scheduling only.
   * Zero declares no work beyond filling the output rows. Never a tolerance.
   */
    long double load_work = 0;
    /** @brief Add scale * Y V (transpose=false) or scale * Y^T V to output.
   * An independent exact contraction order, not a rank approximation. Inputs
   * and output must not alias. Scratch is bounded per input column below.
   */
    std::function<void(
        bool, const Eigen::Ref<const Eigen::MatrixXd>&, Eigen::Ref<Eigen::MatrixXd>, double)>
        contract;
    std::size_t contract_rows = 0;
    /** Work for the pair Y V_beta and Y^T V_alpha, per common input column. */
    long double contract_work = 0;
};

/** @brief Bounded amplitude multiplication through factors or an exact support map.
 * Chooses a contraction order by arithmetic work. No product carrier, source
 * Gram cache, or per-thread panel is allocated. All calls share these slabs.
 */
class SpinAmplitudeWorkspace {
  public:
    SpinAmplitudeWorkspace(const SpinFactorSource& source, int source_tile, int columns);
    void add(const Eigen::Ref<const Eigen::MatrixXd>& alpha,
             const Eigen::Ref<const Eigen::MatrixXd>& beta,
             Eigen::Ref<Eigen::MatrixXd> result_alpha,
             Eigen::Ref<Eigen::MatrixXd> result_beta,
             double scale = 1.0);
    std::size_t loads() const noexcept {
        return loads_;
    }
    std::size_t columns() const noexcept {
        return columns_;
    }

  private:
    const SpinFactorSource& source_;
    int width_;
    bool mapped_;
    workspace::Temporary temporary_;
    Eigen::MatrixXd a_, b_, moment_;
    std::size_t loads_ = 0, columns_ = 0;
};

/** @brief Exact concatenation of factor families; no rank truncation or carrier. */
SpinFactorSource sum_factors(std::vector<SpinFactorSource> sources);

/** @brief Factor representation of E(kappa)Y = (E_alpha L)F^T + L(E_beta F)^T.
 * The sigma, generator and the original source's borrowed inputs must outlive
 * this provider. Its one-factor loading scratch is included in workspace_rows.
 */
SpinFactorSource generator_factors(const DirectCiSigmaAction& sigma,
                                   const SpinFactorSource& source,
                                   const Eigen::Ref<const Eigen::MatrixXd>& generator);

/** @brief Cross-spin Hamiltonian factors after their same-spin use is complete.
 * alpha_channels/beta_channels contain E_(spin,P) factors; mixed_alpha/beta
 * contain their K-channel mixtures. Same-spin gather has already consumed them.
 * Column P*width+j belongs to one channel/source. The consumer may overwrite
 * the views, but must finish before returning and may not retain them.
 */
struct FactorCrossBlock {
    int source, width, channel, channels;
    Eigen::Ref<Eigen::MatrixXd> alpha_channels, beta_channels, mixed_alpha, mixed_beta;
};

/** @brief Base and same-spin images; all channel consumers have finished.
 * same_alpha/beta contain D_alpha A and D_beta B, or zero columns for an
 * overlap-only workspace. These borrowed views may be consumed in place.
 */
struct FactorImageBlock {
    int source, width;
    Eigen::Ref<Eigen::MatrixXd> alpha, beta, same_alpha, same_beta;
};

/** @brief Work and explicit numerical-slab bound, not total process memory. */
struct FactorActionWork {
    int source_tile = 0, channel_tile = 0;
    std::size_t panels = 0, sources = 0, channel_blocks = 0, workspace_bound = 0;
    double scatter_seconds = 0, mix_seconds = 0, gather_seconds = 0;
};

/** @brief Shared Slater--Condon source/channel sweep, reusable across RHS.
 * Owns one set of numerical slabs, shared by the OpenMP team. The workspace
 * budget controls only residency, never a physical-space approximation.
 */
class FactorActionWorkspace {
  public:
    FactorActionWorkspace(const SpinFactorSource& source,
                          int channels,
                          std::size_t workspace_bytes,
                          bool overlap_only = false,
                          int reserved_channel_panels = 0);

    /** @brief Shared consumer slab dimensions; reservations count inside the budget. */
    int source_tile() const noexcept {
        return source_tile_;
    }
    int channel_tile() const noexcept {
        return channel_tile_;
    }

    FactorActionWork apply(const DirectCiSigmaAction& sigma,
                           const SpinFactorSource& source,
                           const std::function<void(FactorCrossBlock)>& cross,
                           const std::function<void(FactorImageBlock)>& finish);

  private:
    int n_alpha_, n_beta_, channels_, source_tile_, channel_tile_;
    std::size_t load_rows_;
    bool overlap_only_;
    int reserved_channel_panels_;
    workspace::Temporary temporary_;
    Eigen::MatrixXd alpha_, beta_, ha_, hb_;
    Eigen::MatrixXd ap_, bp_, va_, vb_, aq_, bq_;
};

/** @brief Exact derivative of <left, H right> with respect to active integrals.
 * Sources describe the amplitudes of this bilinear form. Any parity quotient
 * must be justified by the caller's left projection; it is never inferred here.
 * Rank and channel panels are streamed; no alpha-by-beta array is allocated.
 * The budget covers numerical workspace, excluding immutable input metadata
 * and the required returned orbital-pair gradient. Left/right may differ.
 */
DirectCiIntegralAdjoint
factor_integral_adjoint(const DirectCiSigmaAction& sigma,
                        const SpinFactorSource& left,
                        const SpinFactorSource& right,
                        std::size_t workspace_bytes = kFactorWorkspaceBytes);

/** @brief Oriented one-body adjoint <left, E(kappa) right> from spin factors.
 * No symmetry of the returned orbital matrix or closure of a selected space
 * is assumed. Shared slabs and provider scratch obey the numerical budget;
 * immutable metadata and the required returned orbital matrix are separate.
 */
Eigen::MatrixXd factor_generator_adjoint(const DirectCiSigmaAction& sigma,
                                         const SpinFactorSource& left,
                                         const SpinFactorSource& right,
                                         std::size_t workspace_bytes = kFactorWorkspaceBytes);

/** @brief Oriented adjoint <(H-energy) left, E(kappa) right>, without a residual carrier.
 * H acts on the LEFT source. Exchanging sources is generally invalid even if
 * H is self-adjoint. Reuses the common factor/channel sweep, not a new sigma.
 */
Eigen::MatrixXd
factor_residual_generator_adjoint(const DirectCiSigmaAction& sigma,
                                  const SpinFactorSource& left,
                                  const SpinFactorSource& right,
                                  double energy,
                                  std::size_t workspace_bytes = kFactorWorkspaceBytes);

} // namespace xmvb::mcscf
