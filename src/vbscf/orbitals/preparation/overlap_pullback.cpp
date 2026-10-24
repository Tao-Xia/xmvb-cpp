#include "vbscf/orbitals/preparation/overlap_pullback.hpp"

#include <stdexcept>
#include <vector>

namespace xmvb::vb {
namespace {

AcceptedOrbitalPreparationCache build_overlap_geometry(const OrbitalPreparationInput& input,
                                                       const OrbitalPreparationResult& prepared) {
    if (input.n_total_electrons < input.n_active_electrons ||
        (input.n_total_electrons - input.n_active_electrons) % 2 != 0 ||
        input.n_active_orbitals == 0 || input.ao_overlap_matrix.rows() != input.n_basis_functions ||
        input.ao_overlap_matrix.cols() != input.n_basis_functions) {
        throw std::invalid_argument("invalid geometry for active overlap pullback");
    }
    const int n_inactive_orbitals = (input.n_total_electrons - input.n_active_electrons) / 2;
    if (prepared.auxiliary_orbital_matrix.rows() != input.n_basis_functions ||
        prepared.auxiliary_orbital_matrix.cols() < n_inactive_orbitals + input.n_active_orbitals) {
        throw std::invalid_argument("prepared active overlap dimensions are inconsistent");
    }
    return build_accepted_orbital_preparation_cache(input, Eigen::MatrixXd{}, {});
}

} // namespace

OrbitalOverlapPullback::OrbitalOverlapPullback(const OrbitalPreparationInput& input,
                                               const OrbitalPreparationResult& prepared,
                                               const SparseParameterLayout& layout)
    : input_(input), prepared_(prepared), layout_(layout),
      geometry_(build_overlap_geometry(input, prepared)),
      zero_inactive_density_gradient_(
          Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_basis_functions)) {}

void OrbitalOverlapPullback::validate_overlap_adjoint(
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_adjoint) const {
    if (overlap_adjoint.rows() != input_.n_active_orbitals ||
        overlap_adjoint.cols() != input_.n_active_orbitals) {
        throw std::invalid_argument("active overlap adjoint dimensions are inconsistent");
    }
}

void OrbitalOverlapPullback::validate_direction(const Eigen::VectorXd& packed_raw_direction) const {
    if (packed_raw_direction.size() != layout_.size()) {
        throw std::invalid_argument("packed orbital direction size is inconsistent");
    }
}

Eigen::MatrixXd OrbitalOverlapPullback::active_auxiliary_gradient(
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_adjoint) const {
    validate_overlap_adjoint(overlap_adjoint);
    const int n_inactive_orbitals = (input_.n_total_electrons - input_.n_active_electrons) / 2;
    const auto active_auxiliary = prepared_.auxiliary_orbital_matrix.middleCols(
        n_inactive_orbitals, input_.n_active_orbitals);
    return input_.ao_overlap_matrix * active_auxiliary *
           (overlap_adjoint + overlap_adjoint.transpose());
}

Eigen::MatrixXd
OrbitalOverlapPullback::overlap_direction(const Eigen::VectorXd& packed_raw_direction) const {
    validate_direction(packed_raw_direction);
    const auto tangent =
        build_dense_orbital_tangent_context(input_, layout_, packed_raw_direction, geometry_);
    const int n_inactive_orbitals = (input_.n_total_electrons - input_.n_active_electrons) / 2;
    const auto directional = build_orbital_preparation_directional_result(
        input_, tangent, n_inactive_orbitals, input_.n_active_orbitals, geometry_, false);
    const auto active_auxiliary = prepared_.auxiliary_orbital_matrix.middleCols(
        n_inactive_orbitals, input_.n_active_orbitals);
    const Eigen::MatrixXd left = active_auxiliary.transpose() * input_.ao_overlap_matrix *
                                 directional.delta_active_auxiliary_orbitals;
    return left + left.transpose();
}

Eigen::VectorXd
OrbitalOverlapPullback::gradient(const Eigen::Ref<const Eigen::MatrixXd>& overlap_adjoint) const {
    const Eigen::MatrixXd active_gradient = active_auxiliary_gradient(overlap_adjoint);
    const auto full_gradient = backpropagate_active_space_orbital_gradient(
        active_gradient, zero_inactive_density_gradient_, input_, prepared_, geometry_);
    return layout_.gather_from_full(full_gradient);
}

Eigen::VectorXd OrbitalOverlapPullback::gradient_direction(
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_adjoint,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_adjoint_direction,
    const Eigen::VectorXd& packed_raw_direction) const {
    validate_overlap_adjoint(overlap_adjoint);
    validate_overlap_adjoint(overlap_adjoint_direction);
    validate_direction(packed_raw_direction);

    const auto tangent =
        build_dense_orbital_tangent_context(input_, layout_, packed_raw_direction, geometry_);
    const int n_inactive_orbitals = (input_.n_total_electrons - input_.n_active_electrons) / 2;
    const auto directional = build_orbital_preparation_directional_result(
        input_, tangent, n_inactive_orbitals, input_.n_active_orbitals, geometry_, false);
    const Eigen::MatrixXd active_gradient = active_auxiliary_gradient(overlap_adjoint);
    const Eigen::MatrixXd directional_active_gradient =
        input_.ao_overlap_matrix * directional.delta_active_auxiliary_orbitals *
            (overlap_adjoint + overlap_adjoint.transpose()) +
        active_auxiliary_gradient(overlap_adjoint_direction);

    const auto pullback = build_orbital_preparation_pullback(input_, active_gradient, geometry_);
    Eigen::VectorXd full_raw_direction = Eigen::VectorXd::Zero(input_.orbital_value_table.size());
    const auto& parameter_indices = layout_.differentiable_parameter_indices();
    for (int packed_index = 0; packed_index < layout_.size(); ++packed_index) {
        full_raw_direction[parameter_indices[packed_index]] = packed_raw_direction[packed_index];
    }
    const auto fixed_upstream_direction = apply_fixed_upstream_orbital_pullback_direction(
        input_,
        tangent,
        active_gradient,
        directional.basis_overlap_times_delta_active_orbitals,
        zero_inactive_density_gradient_,
        full_raw_direction,
        geometry_,
        pullback);
    const auto varying_upstream_direction = backpropagate_active_space_orbital_gradient(
        directional_active_gradient, zero_inactive_density_gradient_, input_, prepared_, geometry_);
    return layout_.gather_from_full(fixed_upstream_direction) +
           layout_.gather_from_full(varying_upstream_direction);
}

} // namespace xmvb::vb
