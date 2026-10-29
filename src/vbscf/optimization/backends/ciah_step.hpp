#pragma once

#include <optional>

#include "vbscf/optimization/objective/function.hpp"
#include "vbscf/orbitals/charts/chart.hpp"

namespace xmvb::vb {
struct VbScfOptimizerOptions;
struct VbScfOptimizerResult;
struct CiahIterationRecord;
class ExactHvpOperator;
class StructureTangentOperator;
class NonredundantRetractionMetric;
namespace optimizer_detail {

/** @brief One CIAH macro step with a fully relaxed final structure solve. */
struct CiahMacroStep {
    VbScfObjective::TrialEvaluation trial;
    Eigen::VectorXd parameters;
    Eigen::VectorXd next_orbital_guess_packed;
    Eigen::MatrixXd next_structure_guess;
    /** @brief Retained only after both existing terminal conditions pass. */
    std::optional<OrbitalChart> converged_geometry;
    /** @brief Check this trial's energy change and true gradient in its own chart. */
    bool check_convergence(const SparseParameterLayout& layout,
                           const VbScfOptimizerOptions& options,
                           bool complete_active_space,
                           double source_energy);
    /** @brief Align the recycled CI block with the relaxed equal-weight frame. */
    void align_structure_frame(const Eigen::MatrixXd& reference,
                               const Eigen::MatrixXd& overlap_roots);
};

/** @brief Equal-weight coupled CIAH in an immutable nonorthogonal chart. */
bool build_ciah_macro_step(VbScfObjective* objective,
                           const SparseParameterLayout& layout,
                           const VbScfOptimizerOptions& options,
                           const OrbitalChart& chart,
                           const NonredundantRetractionMetric& metric,
                           ExactHvpOperator& hessian,
                           const StructureTangentOperator& structure,
                           const Eigen::VectorXd& gradient,
                           double gradient_norm,
                           const Eigen::VectorXd& orbital_guess_packed,
                           const Eigen::MatrixXd& structure_guess,
                           bool use_structure_response,
                           double energy,
                           double* trust_radius,
                           CiahIterationRecord* record,
                           VbScfOptimizerResult* result,
                           CiahMacroStep* accepted,
                           bool seek_curvature);

} // namespace optimizer_detail
} // namespace xmvb::vb
