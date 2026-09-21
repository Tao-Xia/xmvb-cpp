#include "vbscf/derivatives/hessian/exact/operator.hpp"

#include "vbscf/derivatives/hessian/exact/state_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/LU>

#include "core/openmp.hpp"

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
  const auto build_start = std::chrono::steady_clock::now();
  ++apply_timing_totals_.response_low_rank_build_count;
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
    apply_timing_totals_.response_low_rank_new_columns +=
        static_cast<std::size_t>(new_columns);
    const auto action_start = std::chrono::steady_clock::now();
    const StructureActionResult action =
        response.structure_action->apply(response_vectors);
    apply_timing_totals_.response_low_rank_structure_action_wall_time_seconds +=
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - action_start).count();
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
    if (response.structure_action->supports_integral_direction() &&
        !accepted_point_context_->structure_adjoint_state.has_value()) {
      accepted_point_context_->structure_adjoint_state =
          response.structure_action->prepare_active_adjoint(
              accepted_point_context_->selected_state_matrices,
              accepted_point_context_->selected_state_energies);
    }

    std::vector<Eigen::MatrixXd> multipliers_by_column;
    multipliers_by_column.reserve(static_cast<std::size_t>(new_columns));
    for (int column = 0; column < new_columns; ++column) {
      const int state = additions[static_cast<std::size_t>(column)].state;
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
      multipliers_by_column.push_back(std::move(multipliers));
    }

    // Every column is an independent application of the same accepted-point
    // adjoint B^T. Parallelizing here avoids serial AO pullbacks and also
    // prevents each small column from opening a separate OpenMP team. Nested
    // kernels detect the active region and use one worker, so temporary
    // storage grows with the physical worker count rather than its square.
    Eigen::MatrixXd new_couplings(n_orbital, new_columns);
    std::exception_ptr worker_error;
    const int n_workers = std::max(
        1,
        std::min(effective_openmp_thread_count(), new_columns));
    const auto adjoint_start = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static) if(n_workers > 1) num_threads(n_workers)
    for (int column = 0; column < new_columns; ++column) {
      try {
        const int state = additions[static_cast<std::size_t>(column)].state;
        Eigen::MatrixXd coefficients = Eigen::MatrixXd::Zero(
            n_structures, n_states);
        coefficients.col(state) = response_vectors.col(column);
        new_couplings.col(column) = apply_structure_response_adjoint(
            coefficients,
            multipliers_by_column[static_cast<std::size_t>(column)]);
      } catch (...) {
#pragma omp critical(xmvb_response_low_rank_error)
        {
          if (worker_error == nullptr) worker_error = std::current_exception();
        }
      }
    }
    if (worker_error != nullptr) std::rethrow_exception(worker_error);
    apply_timing_totals_.response_low_rank_adjoint_wall_time_seconds +=
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - adjoint_start).count();

    for (int state = 0; state < n_states; ++state) {
      response_orbital_couplings_by_state_[state].conservativeResize(
          Eigen::NoChange,
          response.response_recycle_spaces[state].size());
    }
    for (int column = 0; column < new_columns; ++column) {
      const int state = additions[static_cast<std::size_t>(column)].state;
      const double coordinate_scale = std::sqrt(
          2.0 * accepted_point_context_->normalized_state_weights[state]);
      if (!(coordinate_scale > 0.0) || !std::isfinite(coordinate_scale)) {
        throw std::logic_error(
            "response low-rank coordinate scale is invalid");
      }
      Eigen::MatrixXd& cached =
          response_orbital_couplings_by_state_[state];
      cached.col(additions[static_cast<std::size_t>(column)].basis_column) =
          new_couplings.col(column) / coordinate_scale;
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
  const double build_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - build_start).count();
  apply_timing_totals_.response_low_rank_wall_time_seconds += build_seconds;
  // Low-rank Schur construction is part of the exact response operator even
  // though it is requested between matrix-vector products. Include it in the
  // accepted-point HVP and response totals so production traces account for
  // the complete second-order model cost.
  apply_timing_totals_.outer_response_wall_time_seconds += build_seconds;
  apply_timing_totals_.total_apply_wall_time_seconds += build_seconds;
  return model;
}

}  // namespace xmvb::vb
