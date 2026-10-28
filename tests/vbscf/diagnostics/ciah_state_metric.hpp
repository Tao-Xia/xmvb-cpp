#pragma once

#include <cmath>
#include <stdexcept>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

#include "vbscf/derivatives/hessian/responses/orbital/preparation.hpp"
#include "vbscf/derivatives/hessian/coupled/structure.hpp"
#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/preparation/preparer.hpp"
#include "vbscf/optimization/trust_region/retraction.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"

namespace xmvb::test {

/** @brief Independent tiny compound matrix for general, not just triangular, frames.
 * This bounded reference never enters a production action or allocates an
 * ambient carrier for a molecular optimization.
 */
inline Eigen::MatrixXd reference_compound(const mcscf::SpinStrings& strings,
                                          const Eigen::MatrixXd& transform) {
    const auto occupations = strings.decode();
    Eigen::MatrixXd result(strings.size(), strings.size());
    Eigen::MatrixXd minor(strings.electrons(), strings.electrons());
    for (Eigen::Index i = 0; i < result.rows(); ++i)
        for (Eigen::Index j = 0; j < result.cols(); ++j) {
            for (Eigen::Index a = 0; a < minor.rows(); ++a)
                for (Eigen::Index b = 0; b < minor.cols(); ++b)
                    minor(a, b) = transform(occupations[i][a], occupations[j][b]);
            result(i, j) = minor.determinant();
        }
    return result;
}

/** @brief Tiny independent projective-state metric, including external motion.
 * Active amplitudes are bounded by the caller's diagnostic guard. Production
 * must use its factor/tile contractions, never this reference's dense carrier.
 * Distinct inactive-hole/external-particle sectors contract through the 1-RDM.
 */
class StateMetricReference {
  public:
    struct Feature {
        Eigen::MatrixXd active, active_external, core_active, core_external;
    };
    struct Point {
        Eigen::MatrixXd core, active, state;
    };

    StateMetricReference(const vb::OrbitalPreparationInput& input,
                         const vb::OrbitalPreparationResult& prepared,
                         const vb::SparseParameterLayout& layout,
                         const vb::OrbitalChart& chart,
                         const vb::StructureTangentOperator& structure,
                         const vb::StructureAction& action,
                         const Eigen::MatrixXd& state,
                         double state_norm)
        : input_(input), layout_(layout), chart_(chart), structure_(structure), action_(action),
          view_(action.orthogonal_view()), state_(state), na_(input.n_active_orbitals),
          nc_((input.n_total_electrons - input.n_active_electrons) / 2),
          preparation_(vb::build_accepted_orbital_preparation_cache(input, Eigen::MatrixXd{}, {})),
          state_norm_(state_norm) {
        const auto& r = view_.alpha.orbital_transform();
        inverse_ = r.triangularView<Eigen::Upper>().solve(Eigen::MatrixXd::Identity(na_, na_));
        active_ = prepared.auxiliary_orbital_matrix.middleCols(nc_, na_) * inverse_;
        const Eigen::MatrixXd raw_core = preparation_.normalized_orbitals.leftCols(nc_);
        if (nc_) {
            const Eigen::LLT<Eigen::MatrixXd> core_factor(raw_core.transpose() *
                                                          input.ao_overlap_matrix * raw_core);
            if (core_factor.info() != Eigen::Success)
                throw std::runtime_error("state metric reference core factor failed");
            core_inverse_ = core_factor.matrixU().solve(Eigen::MatrixXd::Identity(nc_, nc_));
        } else
            core_inverse_.resize(0, 0);
        core_ = raw_core * core_inverse_;
        density_ = view_.sigma.one_body_generator_adjoint(state_, state_);
        holes_ = 2 * Eigen::MatrixXd::Identity(na_, na_) - density_;
    }

    Feature feature(const Eigen::VectorXd& direction) const {
        const auto no = chart_.reduced_size();
        const auto raw =
            vb::gather_nonredundant_retract_tangent(input_, chart_, layout_, direction.head(no));
        const auto tangent =
            vb::build_dense_orbital_tangent_context(input_, layout_, raw, preparation_);
        const auto derivative = vb::build_orbital_preparation_directional_result(
            input_, tangent, nc_, na_, preparation_, false);
        const Eigen::MatrixXd da = derivative.delta_active_auxiliary_orbitals * inverse_;
        const Eigen::MatrixXd dc = tangent.delta_normalized_orbitals.leftCols(nc_) * core_inverse_;
        const auto& overlap = input_.ao_overlap_matrix;
        const Eigen::MatrixXd k = active_.transpose() * overlap * da;
        Feature result;
        result.active = view_.sigma.apply_one_body_generator(state_, k);
        result.active +=
            action_.orthogonalize_structure_block(structure_.coefficient_response_coordinates(
                direction.tail(structure_.tangent_size()))) /
            state_norm_;
        result.active -= state_ * (state_.array() * result.active.array()).sum();
        result.core_active = active_.transpose() * overlap * dc;
        result.core_external =
            dc - core_ * (core_.transpose() * overlap * dc) - active_ * result.core_active;
        result.active_external = da - core_ * (core_.transpose() * overlap * da) - active_ * k;
        return result;
    }

    double inner(const Feature& left, const Feature& right) const {
        const auto& overlap = input_.ao_overlap_matrix;
        return 2 *
               ((left.active.array() * right.active.array()).sum() +
                (density_.array() *
                 (left.active_external.transpose() * overlap * right.active_external).array())
                    .sum() +
                (left.core_active.array() * (holes_ * right.core_active).array()).sum() +
                2 * (left.core_external.array() * (overlap * right.core_external).array()).sum());
    }

    /** General cross determinant factors include inactive and active span motion. */
    double overlap(const Point& left, const Point& right) const {
        const auto& s = input_.ao_overlap_matrix;
        Eigen::MatrixXd cross = left.active.transpose() * s * right.active;
        double determinant = 1;
        if (nc_) {
            const Eigen::MatrixXd core_cross = left.core.transpose() * s * right.core;
            const Eigen::PartialPivLU<Eigen::MatrixXd> lu(core_cross);
            determinant = lu.determinant();
            cross -= left.active.transpose() * s * right.core *
                     lu.solve((left.core.transpose() * s * right.active).eval());
        }
        const Eigen::MatrixXd alpha = reference_compound(view_.alpha.strings(), cross);
        const Eigen::MatrixXd beta = reference_compound(view_.beta.strings(), cross);
        return determinant * determinant *
               (left.state.array() * (alpha * right.state * beta.transpose()).array()).sum();
    }

    Point point(const Eigen::VectorXd& direction, double fraction) const {
        const auto no = chart_.reduced_size();
        const auto trial = chart_.retract_step(input_, fraction * direction.head(no));
        const auto prepared = vb::ActiveSpaceOrbitalPreparer{}.prepare(trial);
        const Eigen::MatrixXd raw_core =
            prepared.physical_orbital_frame.normalized_orbital_matrix.leftCols(nc_);
        Point result;
        if (nc_) {
            const Eigen::LLT<Eigen::MatrixXd> core_factor(raw_core.transpose() *
                                                          input_.ao_overlap_matrix * raw_core);
            result.core =
                raw_core * core_factor.matrixU().solve(Eigen::MatrixXd::Identity(nc_, nc_));
        } else
            result.core.resize(input_.n_basis_functions, 0);
        const Eigen::Map<const Eigen::MatrixXd> gram(
            prepared.active_orbital_overlap_matrix.data(), na_, na_);
        const Eigen::LLT<Eigen::MatrixXd> factor(gram);
        const Eigen::MatrixXd r = factor.matrixU();
        result.active = prepared.auxiliary_orbital_matrix.middleCols(nc_, na_) *
                        factor.matrixU().solve(Eigen::MatrixXd::Identity(na_, na_));
        const Eigen::MatrixXd ci =
            action_.orthogonalize_structure_block(structure_.coefficient_response_coordinates(
                direction.tail(structure_.tangent_size()))) /
            state_norm_;
        const Eigen::MatrixXd a = reference_compound(view_.alpha.strings(), r * inverse_);
        const Eigen::MatrixXd b = reference_compound(view_.beta.strings(), r * inverse_);
        result.state = a * (state_ + fraction * ci) * b.transpose();
        result.state /= result.state.stableNorm();
        return result;
    }

  private:
    const vb::OrbitalPreparationInput& input_;
    const vb::SparseParameterLayout& layout_;
    const vb::OrbitalChart& chart_;
    const vb::StructureTangentOperator& structure_;
    const vb::StructureAction& action_;
    vb::StructureAction::OrthogonalView view_;
    Eigen::MatrixXd state_, inverse_, core_inverse_, active_, core_, density_, holes_;
    Eigen::Index na_, nc_;
    vb::AcceptedOrbitalPreparationCache preparation_;
    double state_norm_ = 1;
};

} // namespace xmvb::test
