#include "vbscf/derivatives/hessian/responses/active_space/integral_direction.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/core/vbscf_input.hpp"
#include "vbscf/integrals/active/two_electron/response/directional.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"

namespace xmvb::vb {
namespace {

void resize_for_overwrite(
    std::vector<double>* values,
    std::size_t size) {
  if (values->size() != size) {
    values->resize(size);
  }
}

}  // namespace

ActiveSpaceIntegralDirectionView build_active_space_integral_direction(
    const ActiveSpaceIntegralDirectionContext& context,
    const ActiveSpaceIntegralTangent& tangent,
    ActiveSpaceIntegralDirectionWorkspace* workspace) {
  const int n_basis_functions =
      context.input.orbital_preparation_input.n_basis_functions;
  const int n_active_orbitals =
      context.input.orbital_preparation_input.n_active_orbitals;
  const std::size_t active_matrix_size =
      n_active_orbitals * n_active_orbitals;
  if (n_basis_functions <= 0 || n_active_orbitals <= 0) {
    throw std::invalid_argument(
        "active-space directional integrals require positive dimensions");
  }
  resize_for_overwrite(&workspace->overlap, active_matrix_size);
  resize_for_overwrite(&workspace->one_electron, active_matrix_size);
  Eigen::Map<Eigen::MatrixXd> delta_overlap(
      workspace->overlap.data(),
      n_active_orbitals,
      n_active_orbitals);
  Eigen::Map<Eigen::MatrixXd> delta_one_electron(
      workspace->one_electron.data(),
      n_active_orbitals,
      n_active_orbitals);

  const Eigen::MatrixXd delta_overlap_left =
      tangent.active_auxiliary_orbitals.transpose() *
      context.overlap_times_active_auxiliary_orbitals;
  delta_overlap.noalias() = delta_overlap_left;
  delta_overlap.noalias() += delta_overlap_left.transpose();

  const Eigen::MatrixXd delta_one_electron_left =
      tangent.active_auxiliary_orbitals.transpose() *
      context.effective_h1e_times_active_auxiliary_orbitals;
  delta_one_electron.noalias() = delta_one_electron_left;
  delta_one_electron.noalias() +=
      context.active_auxiliary_orbitals.transpose() *
      tangent.delta_effective_h1e_times_active_auxiliary_orbitals;
  delta_one_electron.noalias() +=
      context.effective_h1e_transpose_times_active_auxiliary_orbitals
          .transpose() *
      tangent.active_auxiliary_orbitals;

  if (context.dense_active_coefficients.rows() != n_basis_functions ||
      context.dense_active_coefficients.cols() != n_active_orbitals ||
      tangent.dense_active_coefficients.rows() != n_basis_functions ||
      tangent.dense_active_coefficients.cols() != n_active_orbitals) {
    throw std::invalid_argument(
        "dense active coefficient shapes are inconsistent in active-space "
        "directional integrals");
  }

  if (tangent.precomputed_packed_two_electron != nullptr) {
    const std::size_t packed_size =
        packed_active_two_electron_integral_count(n_active_orbitals);
    if (tangent.precomputed_packed_two_electron->size() !=
        static_cast<Eigen::Index>(packed_size)) {
      throw std::invalid_argument(
          "precomputed block delta GGO has inconsistent dimensions");
    }
    workspace->packed_two_electron.assign(
        tangent.precomputed_packed_two_electron->data(),
        tangent.precomputed_packed_two_electron->data() + packed_size);
  } else {
    compute_exact_packed_active_two_electron_integral_directional_derivative(
        context.dense_active_coefficients,
        tangent.dense_active_coefficients,
        context.input.ao_integral_input,
        n_active_orbitals,
        &workspace->two_electron,
        &workspace->packed_two_electron,
        &context.accepted_two_electron_integrals);
  }

  return {
      workspace->overlap,
      workspace->one_electron,
      workspace->packed_two_electron};
}

}  // namespace xmvb::vb
