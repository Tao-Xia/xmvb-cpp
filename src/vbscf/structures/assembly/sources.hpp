#pragma once

#include <cstddef>
#include <vector>

#include "mcscf/ci/factors.hpp"
#include "mcscf/ci/product_index.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"

namespace xmvb::vb {

/** @brief One exact source factor anchored on a row or column of raw support. */
struct SpinSource {
    Eigen::Index string = 0;
    bool beta = true;
};

/** @brief Minimum row/column cover of the original touched spin products.
 * Each input product belongs to exactly one source. The cover minimizes the
 * number of coordinate-anchored outer products, without inspecting amplitudes.
 * Original output support remains separate from an optional parity quotient.
 */
struct SpinSources {
    std::vector<Eigen::Index> columns, alpha_rows;
    std::vector<std::size_t> alpha_offsets;
    std::vector<std::vector<std::size_t>> beta_groups;
    std::vector<SpinSource> nodes;
    std::vector<std::size_t> offsets, groups;
    ExteriorProjection alpha_projection, beta_projection;

    SpinSources(const std::vector<mcscf::ProductIndex>& products,
                Eigen::Index na,
                Eigen::Index nb,
                int parity = 0);

    /** @brief Load raw source factors from original-support coefficients.
   * Both output views are overwritten. Parity weights belong to this plan;
   * callers apply their orbital transforms after loading the raw factors.
   */
    void load(const std::vector<mcscf::ProductIndex>& products,
              const Eigen::Ref<const Eigen::VectorXd>& coefficients,
              Eigen::Index first,
              Eigen::Index width,
              Eigen::Ref<Eigen::MatrixXd> alpha,
              Eigen::Ref<Eigen::MatrixXd> beta) const;

    /** @brief Borrow a transformed factor provider over original support values.
   * physical=true reconstructs the full amplitude from a certified common
   * exchange-parity representative. Metadata, values and transforms must
   * outlive the provider; only its requested source panel is materialized.
   */
    mcscf::SpinFactorSource orthogonal_source(const std::vector<mcscf::ProductIndex>& products,
                                              const Eigen::Ref<const Eigen::VectorXd>& coefficients,
                                              const ExteriorOrbitalTransform& alpha,
                                              const ExteriorOrbitalTransform& beta,
                                              bool physical = false) const;

    /** @brief Exact orbital/coefficient tangent with at most two factor families.
   * The coefficient tangent changes only the non-anchor factor and is merged
   * into its orbital derivative. A coefficient-only direction uses one family.
   * All borrowed inputs must outlive the provider. Loading needs only one
   * single-spin scratch panel, never a product array or all-source factors.
   */
    mcscf::SpinFactorSource tangent_source(const std::vector<mcscf::ProductIndex>& products,
                                           const Eigen::Ref<const Eigen::VectorXd>& coefficients,
                                           const Eigen::VectorXd* coefficient_direction,
                                           const ExteriorOrbitalTransform& alpha,
                                           const ExteriorOrbitalTransform& beta,
                                           const mcscf::DirectCiSigmaAction& sigma,
                                           const Eigen::MatrixXd* generator,
                                           bool physical = false) const;

    /** @brief Exact integral tangent with work-planned exchange-projector placement.
   * The two physical sources are the certified projections of their respective
   * representatives. This identity is not valid for an oriented residual VJP.
   */
    mcscf::DirectCiIntegralAdjoint
    integral_tangent(const mcscf::DirectCiSigmaAction& sigma,
                     const mcscf::SpinFactorSource& state,
                     const mcscf::SpinFactorSource& physical_state,
                     const mcscf::SpinFactorSource& direction,
                     const mcscf::SpinFactorSource& physical_direction) const;

    std::size_t dynamic_bytes() const noexcept {
        std::size_t bytes = (columns.capacity() + alpha_rows.capacity()) * sizeof(Eigen::Index) +
                            (alpha_offsets.capacity() + offsets.capacity() + groups.capacity()) *
                                sizeof(std::size_t) +
                            beta_groups.capacity() * sizeof(std::vector<std::size_t>) +
                            nodes.capacity() * sizeof(SpinSource);
        for (const auto& groups : beta_groups)
            bytes += groups.capacity() * sizeof(std::size_t);
        for (const auto* plan : {&alpha_projection, &beta_projection}) {
            bytes += plan->shear_pairs.capacity() * sizeof(std::vector<Eigen::Index>);
            for (const auto& pairs : plan->shear_pairs)
                bytes += pairs.capacity() * sizeof(Eigen::Index);
        }
        return bytes;
    }

  private:
    int parity_ = 0;
};

} // namespace xmvb::vb
