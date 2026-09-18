#include "vbscf/optimization/coupled/preconditioner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Eigenvalues>

namespace xmvb::vb {
namespace {

struct StateResponseInverse {
  int coefficient_offset = 0;
  int multiplier_offset = 0;
  Eigen::VectorXd inverse_diagonal;
  Eigen::MatrixXd overlap_selected;
  Eigen::MatrixXd inverse_schur;
};

double roundoff_floor(double scale, int dimension) {
  const double epsilon = std::numeric_limits<double>::epsilon();
  const double floor = epsilon * std::max(1, dimension) * scale;
  return std::max(floor, std::numeric_limits<double>::min());
}

Eigen::MatrixXd regularized_spd_inverse(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(matrix);
  if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) {
    throw std::runtime_error(
        "structure-response Schur eigensolve failed");
  }
  const double scale = solver.eigenvalues().cwiseAbs().maxCoeff();
  if (!(scale > 0.0) || !std::isfinite(scale)) {
    throw std::invalid_argument(
        "selected overlap columns have zero numerical rank");
  }
  const double floor = roundoff_floor(scale, matrix.rows());
  if (solver.eigenvalues().minCoeff() < -floor) {
    throw std::runtime_error(
        "structure-response Schur matrix is not positive semidefinite");
  }
  const Eigen::VectorXd inverse_eigenvalues =
      solver.eigenvalues().cwiseMax(floor).cwiseInverse();
  Eigen::MatrixXd inverse =
      solver.eigenvectors() * inverse_eigenvalues.asDiagonal() *
      solver.eigenvectors().transpose();
  inverse = 0.5 * (inverse + inverse.transpose());
  if (!inverse.allFinite()) {
    throw std::runtime_error(
        "structure-response Schur inverse is non-finite");
  }
  return inverse;
}

void validate_response_inputs(
    const SelectedSubspaceResponseLayout& layout,
    const StructureResponsePreconditionerData& data) {
  int n_selected = 0;
  for (int cluster = 0; cluster < layout.n_clusters(); ++cluster) {
    n_selected += layout.cluster(cluster).n_states;
  }
  if (data.hamiltonian_diagonal.size() != layout.n_structures() ||
      data.overlap_diagonal.size() != layout.n_structures() ||
      data.selected_energies.size() != n_selected ||
      data.overlap_selected.rows() != layout.n_structures() ||
      data.overlap_selected.cols() != n_selected ||
      !data.hamiltonian_diagonal.allFinite() ||
      !data.overlap_diagonal.allFinite() ||
      !data.selected_energies.allFinite() ||
      !data.overlap_selected.allFinite()) {
    throw std::invalid_argument(
        "coupled block preconditioner inputs are inconsistent");
  }
}

Eigen::VectorXd apply_response_inverses(
    const std::vector<StateResponseInverse>& response_inverses,
    int response_size,
    const Eigen::VectorXd& residual) {
  if (residual.size() != response_size || !residual.allFinite()) {
    throw std::invalid_argument(
        "structure-response inverse received an invalid residual");
  }
  Eigen::VectorXd image(response_size);
  for (const StateResponseInverse& inverse : response_inverses) {
    const int n_structures = inverse.inverse_diagonal.size();
    const int width = inverse.inverse_schur.rows();
    const auto coefficient_residual = residual.segment(
        inverse.coefficient_offset,
        n_structures);
    const auto multiplier_residual = residual.segment(
        inverse.multiplier_offset,
        width);

    const Eigen::VectorXd diagonal_solution =
        inverse.inverse_diagonal.array() *
        coefficient_residual.array();
    const Eigen::VectorXd multiplier_solution =
        inverse.inverse_schur *
        (multiplier_residual -
         inverse.overlap_selected.transpose() * diagonal_solution);
    image.segment(inverse.coefficient_offset, n_structures) =
        diagonal_solution -
        inverse.inverse_diagonal.asDiagonal() *
            inverse.overlap_selected * multiplier_solution;
    image.segment(inverse.multiplier_offset, width) = multiplier_solution;
  }
  if (!image.allFinite()) {
    throw std::runtime_error(
        "structure-response inverse produced a non-finite vector");
  }
  return image;
}

std::vector<StateResponseInverse> build_response_inverses(
    const SelectedSubspaceResponseLayout& layout,
    const StructureResponsePreconditionerData& data) {
  std::vector<StateResponseInverse> inverses;
  inverses.reserve(static_cast<std::size_t>(
      data.selected_energies.size()));
  int first_state = 0;
  for (int cluster = 0; cluster < layout.n_clusters(); ++cluster) {
    const int width = layout.cluster(cluster).n_states;
    const Eigen::MatrixXd cluster_overlap =
        data.overlap_selected.middleCols(first_state, width);
    for (int state = 0; state < width; ++state) {
      const double energy = data.selected_energies[first_state + state];
      const Eigen::VectorXd gap =
          data.hamiltonian_diagonal - energy * data.overlap_diagonal;
      const double input_scale = std::max(
          data.hamiltonian_diagonal.cwiseAbs().maxCoeff(),
          std::abs(energy) * data.overlap_diagonal.cwiseAbs().maxCoeff());
      const double floor = roundoff_floor(
          input_scale, layout.n_structures());
      const Eigen::VectorXd inverse_diagonal =
          gap.cwiseAbs().cwiseMax(floor).cwiseInverse();
      const Eigen::MatrixXd schur =
          cluster_overlap.transpose() *
          inverse_diagonal.asDiagonal() * cluster_overlap;
      inverses.push_back(StateResponseInverse{
          layout.coefficient_offset(cluster) +
              state * layout.n_structures(),
          layout.multiplier_offset(cluster) + state * width,
          inverse_diagonal,
          cluster_overlap,
          regularized_spd_inverse(schur)});
    }
    first_state += width;
  }
  return inverses;
}

}  // namespace

SymmetricOperatorAction make_structure_response_inverse_preconditioner(
    const SelectedSubspaceResponseLayout& response_layout,
    const StructureResponsePreconditionerData& structure_data) {
  validate_response_inputs(response_layout, structure_data);
  const int response_size = response_layout.response_size();
  std::vector<StateResponseInverse> response_inverses =
      build_response_inverses(response_layout, structure_data);
  return [response_size,
          response_inverses = std::move(response_inverses)](
             const Eigen::VectorXd& residual) {
    return apply_response_inverses(
        response_inverses,
        response_size,
        residual);
  };
}

SymmetricOperatorAction make_coupled_block_inverse_preconditioner(
    int n_orbital_coordinates,
    const SelectedSubspaceResponseLayout& response_layout,
    const StructureResponsePreconditionerData& structure_data,
    SymmetricOperatorAction apply_orbital_inverse) {
  if (n_orbital_coordinates <= 0 || !apply_orbital_inverse) {
    throw std::invalid_argument(
        "coupled block preconditioner requires an orbital inverse action");
  }
  const int n_response = response_layout.response_size();
  const int size = n_orbital_coordinates + n_response;
  SymmetricOperatorAction response_inverse =
      make_structure_response_inverse_preconditioner(
          response_layout,
          structure_data);

  return [n_orbital_coordinates,
          n_response,
          size,
          orbital_inverse = std::move(apply_orbital_inverse),
          response_inverse = std::move(response_inverse)](
             const Eigen::VectorXd& residual) {
    if (residual.size() != size || !residual.allFinite()) {
      throw std::invalid_argument(
          "coupled block inverse received an invalid residual");
    }
    Eigen::VectorXd image(size);
    const Eigen::VectorXd orbital_image =
        orbital_inverse(residual.head(n_orbital_coordinates));
    if (orbital_image.size() != n_orbital_coordinates ||
        !orbital_image.allFinite()) {
      throw std::runtime_error(
          "orbital inverse preconditioner returned an invalid vector");
    }
    image.head(n_orbital_coordinates) = orbital_image;
    image.tail(n_response) = response_inverse(residual.tail(n_response));
    if (!image.allFinite()) {
      throw std::runtime_error(
          "coupled block inverse produced a non-finite vector");
    }
    return image;
  };
}

}  // namespace xmvb::vb
