#pragma once

#include <memory>

#include "mcscf/ci/integrals.hpp"
#include "vbscf/structures/assembly/action.hpp"

namespace xmvb::vb {

/** @brief Immutable accepted-point contractions; no spin-product images. */
struct OperatorFirstState {
    /** Shared identity token prevents reuse after the physical point is replaced. */
    std::shared_ptr<const void> point;
    /** Canonical one-body and ordered-pair integral cotangents. */
    Eigen::MatrixXd one, two;
    /** Raw residual-generator moment, including the fixed Rayleigh multiplier. */
    Eigen::MatrixXd moment;
};

struct OperatorFirstWork {
    std::size_t element_evaluations = 0, source_products = 0, beta_columns = 0;
    std::size_t plan_bytes = 0, panel_bytes = 0;
    double raw_seconds = 0, metric_seconds = 0;
    bool covalent_reduction = false;
};

struct OperatorFirstImages {
    StructureActionResult images;
    OperatorFirstWork work;
};

/** @brief Exact selected-space action with the operator before orbital transforms.
 * Owns only point-local canonical integrals and immutable source indexing.
 * Orbital maps, incidence and single-spin links are borrowed from the view.
 * The view owner must outlive this executor. FULL solver coordinates and CIAH
 * policy remain outside this numerical adapter.
 */
class OperatorFirstAction {
  public:
    OperatorFirstAction(StructureAction::OrthogonalView view,
                        Eigen::Index n_structures,
                        const mcscf::ActiveIntegrals& integrals,
                        const Eigen::Ref<const Eigen::MatrixXd>& orbital_map);
    ~OperatorFirstAction();
    OperatorFirstAction(const OperatorFirstAction&) = delete;
    OperatorFirstAction& operator=(const OperatorFirstAction&) = delete;

    OperatorFirstImages apply(const Eigen::Ref<const Eigen::MatrixXd>& vectors,
                              std::size_t workspace_bytes,
                              bool collect_overlap = true,
                              bool overlap_only = false) const;

    /** @brief Fused integral and relative-orbital VJP, or its analytic tangent.
     * Coefficient directions move both bra and ket. Energies are the fixed
     * Rayleigh multipliers; their independent variations remain with the caller.
     * A joint CI request shares the primal sweep but retains independent images
     * and cotangents, so orbital and CI curvature cannot be counted twice.
     */
    StructureResponseSweep jet(const StructureAdjointState& state,
                               const mcscf::ActiveIntegrals* integral_direction,
                               const Eigen::MatrixXd* orbital_direction,
                               const Eigen::MatrixXd* coefficient_direction,
                               bool collect_images,
                               std::size_t workspace_bytes,
                               std::vector<Eigen::MatrixXd>* densities = nullptr,
                               OperatorFirstState* accepted = nullptr,
                               StructureResponseRequest* ci_response = nullptr) const;

  private:
    struct Data;
    std::unique_ptr<Data> data_;
};
} // namespace xmvb::vb
