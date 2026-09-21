#include "vbscf/derivatives/hessian/exact/operator.hpp"

#include "vbscf/derivatives/hessian/exact/state_internal.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/LU>

namespace xmvb::vb {

Eigen::MatrixXd ResponseLowRankModel::apply(
    const Eigen::Ref<const Eigen::MatrixXd>& orbital_directions) const {
  if (!orbital_directions.allFinite() ||
      orbital_couplings.rows() != orbital_directions.rows() ||
      projected_inverse.rows() != orbital_couplings.cols() ||
      projected_inverse.cols() != orbital_couplings.cols() ||
      !orbital_couplings.allFinite() || !projected_inverse.allFinite()) {
    throw std::invalid_argument(
        "response low-rank model dimensions or values are invalid");
  }
  if (orbital_couplings.cols() == 0) {
    return Eigen::MatrixXd::Zero(
        orbital_directions.rows(), orbital_directions.cols());
  }
  // Physical structure coefficients are packed with sqrt(2 w_I). The
  // coupling columns above use that metric, so the ordinary Schur form has no
  // additional state-weight factor here.
  return -orbital_couplings *
      (projected_inverse *
       (orbital_couplings.transpose() * orbital_directions));
}

ResponseLowRankModel ExactHvpOperator::State::response_low_rank_model() const {
  const auto& response =
      outer_response_context().selected_state_eigen_response_operator;
  const int n_states = static_cast<int>(response.selected_eigenvalues.size());
  const int n_structures = accepted_point_context_->n_structures;
  const int n_orbital = nonredundant_space_->reduced_size();
  if (n_states <= 0 ||
      response.response_recycle_spaces.size() !=
          static_cast<std::size_t>(n_states) ||
      response.selected_eigenvectors.rows() != n_structures ||
      response.selected_eigenvectors.cols() != n_states ||
      response.overlap_selected.rows() != n_structures ||
      response.overlap_selected.cols() != n_states ||
      response.structure_action == nullptr) {
    throw std::logic_error(
        "accepted response model dimensions are inconsistent");
  }

  if (response_orbital_couplings_by_state_.size() !=
      static_cast<std::size_t>(n_states)) {
    response_orbital_couplings_by_state_.assign(
        static_cast<std::size_t>(n_states), Eigen::MatrixXd(n_orbital, 0));
  }
  bool reset = false;
  int new_columns = 0;
  for (int state = 0; state < n_states; ++state) {
    const int basis_size = response.response_recycle_spaces[state].size();
    const Eigen::MatrixXd& cached =
        response_orbital_couplings_by_state_[state];
    if (cached.rows() != n_orbital || cached.cols() > basis_size) {
      reset = true;
      break;
    }
    new_columns += basis_size - static_cast<int>(cached.cols());
  }
  if (reset) {
    response_orbital_couplings_by_state_.assign(
        static_cast<std::size_t>(n_states), Eigen::MatrixXd(n_orbital, 0));
    new_columns = 0;
    for (const auto& space : response.response_recycle_spaces) {
      new_columns += space.size();
    }
  }

  struct NewColumn {
    int state;
    int basis_column;
  };
  std::vector<NewColumn> additions;
  additions.reserve(static_cast<std::size_t>(new_columns));
  Eigen::MatrixXd response_vectors(n_structures, new_columns);
  int added = 0;
  for (int state = 0; state < n_states; ++state) {
    const auto& space = response.response_recycle_spaces[state];
    const int first_new = static_cast<int>(
        response_orbital_couplings_by_state_[state].cols());
    for (int column = first_new; column < space.size(); ++column) {
      response_vectors.col(added) = space.basis().col(column);
      additions.push_back({state, column});
      ++added;
    }
  }

  if (new_columns > 0) {
    const StructureActionResult action =
        response.structure_action->apply(response_vectors);
    if (action.hamiltonian.rows() != n_structures ||
        action.hamiltonian.cols() != new_columns ||
        action.overlap.rows() != n_structures ||
        action.overlap.cols() != new_columns ||
        !action.hamiltonian.allFinite() || !action.overlap.allFinite()) {
      throw std::runtime_error(
          "response low-rank basis action returned invalid images");
    }
    const Eigen::MatrixXd metric =
        response.selected_eigenvectors.transpose() * response.overlap_selected;
    Eigen::FullPivLU<Eigen::MatrixXd> metric_solver(metric);
    if (!metric_solver.isInvertible()) {
      throw std::runtime_error(
          "selected-state response metric is singular");
    }

    for (int column = 0; column < new_columns; ++column) {
      const int state = additions[static_cast<std::size_t>(column)].state;
      Eigen::MatrixXd coefficients = Eigen::MatrixXd::Zero(
          n_structures, n_states);
      coefficients.col(state) = response_vectors.col(column);
      const Eigen::VectorXd shifted = action.hamiltonian.col(column) -
          response.selected_eigenvalues[state] * action.overlap.col(column);
      Eigen::MatrixXd multipliers = Eigen::MatrixXd::Zero(
          n_states, n_states);
      if (response.use_equal_weight_subspace_response) {
        multipliers.col(state) = -metric_solver.solve(
            response.selected_eigenvectors.transpose() * shifted);
      } else {
        multipliers(state, state) =
            -response.selected_eigenvectors.col(state).dot(shifted) /
            metric(state, state);
      }
      const Eigen::VectorXd coupling =
          apply_structure_response_adjoint(coefficients, multipliers);
      const double coordinate_scale = std::sqrt(
          2.0 * accepted_point_context_->normalized_state_weights[state]);
      if (!(coordinate_scale > 0.0) || !std::isfinite(coordinate_scale)) {
        throw std::logic_error(
            "response low-rank coordinate scale is invalid");
      }
      Eigen::MatrixXd& cached =
          response_orbital_couplings_by_state_[state];
      const Eigen::Index old_size = cached.cols();
      cached.conservativeResize(Eigen::NoChange, old_size + 1);
      cached.col(old_size) = coupling / coordinate_scale;
    }
  }

  int rank = 0;
  for (const auto& space : response.response_recycle_spaces) {
    rank += space.size();
  }
  ResponseLowRankModel model;
  model.revision = response.revision();
  model.orbital_couplings.resize(n_orbital, rank);
  model.projected_inverse = Eigen::MatrixXd::Zero(rank, rank);
  int offset = 0;
  for (int state = 0; state < n_states; ++state) {
    const auto& space = response.response_recycle_spaces[state];
    const int width = space.size();
    if (response_orbital_couplings_by_state_[state].cols() != width) {
      throw std::logic_error(
          "response low-rank coupling cache is incomplete");
    }
    if (width == 0) continue;
    model.orbital_couplings.middleCols(offset, width) =
        response_orbital_couplings_by_state_[state];
    model.projected_inverse.block(offset, offset, width, width) =
        space.projected_inverse();
    offset += width;
  }
  return model;
}

}  // namespace xmvb::vb
