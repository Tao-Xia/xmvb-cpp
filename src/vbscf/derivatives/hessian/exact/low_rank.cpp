#include "vbscf/derivatives/hessian/exact/operator.hpp"

#include "vbscf/derivatives/hessian/exact/state_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>

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

ResponseSpectrumSummary summarize_response_spectrum(
    const ResponseLowRankModel& model) {
  ResponseSpectrumSummary summary;
  const Eigen::Index rank = model.orbital_couplings.cols();
  summary.model_rank = static_cast<int>(rank);
  if (rank == 0) return summary;
  if (model.projected_inverse.rows() != rank ||
      model.projected_inverse.cols() != rank) {
    throw std::logic_error("response Schur diagnostic dimensions differ");
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> inverse_eigensolver(
      0.5 * (model.projected_inverse + model.projected_inverse.transpose()),
      Eigen::ComputeEigenvectors);
  if (inverse_eigensolver.info() != Eigen::Success) {
    throw std::runtime_error(
        "response Schur projected inverse eigensolve failed");
  }
  const Eigen::ArrayXd inverse_eigenvalues =
      inverse_eigensolver.eigenvalues().array();
  const Eigen::MatrixXd schur_factors = model.orbital_couplings *
      inverse_eigensolver.eigenvectors() *
      inverse_eigenvalues.abs().sqrt().matrix().asDiagonal();
  const Eigen::VectorXd signs = inverse_eigenvalues.sign().matrix();
  Eigen::MatrixXd compact_schur;
  if (schur_factors.rows() >= rank) {
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(schur_factors);
    const Eigen::MatrixXd triangular =
        qr.matrixQR().topRows(rank).template triangularView<Eigen::Upper>();
    compact_schur = triangular * signs.asDiagonal() * triangular.transpose();
  } else {
    compact_schur =
        schur_factors * signs.asDiagonal() * schur_factors.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> schur_eigensolver(
      0.5 * (compact_schur + compact_schur.transpose()),
      Eigen::EigenvaluesOnly);
  if (schur_eigensolver.info() != Eigen::Success) {
    throw std::runtime_error("response Schur eigensolve failed");
  }
  std::vector<double> weights(
      static_cast<std::size_t>(schur_eigensolver.eigenvalues().size()), 0.0);
  for (Eigen::Index mode = 0;
       mode < schur_eigensolver.eigenvalues().size(); ++mode) {
    weights[static_cast<std::size_t>(mode)] =
        std::abs(schur_eigensolver.eigenvalues()[mode]);
  }
  std::sort(weights.begin(), weights.end(), std::greater<double>());
  const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
  double square_sum = 0.0;
  for (const double weight : weights) square_sum += weight * weight;
  if (!(total > 0.0) || !std::isfinite(total) || !(square_sum > 0.0) ||
      !std::isfinite(square_sum)) {
    return summary;
  }

  summary.effective_rank = total * total / square_sum;
  summary.top_mode_fraction = weights.front() / total;
  double cumulative = 0.0;
  const int spectrum_size = static_cast<int>(weights.size());
  for (int mode = 0; mode < spectrum_size; ++mode) {
    cumulative += weights[static_cast<std::size_t>(mode)];
    if (mode == std::min(4, spectrum_size - 1)) {
      summary.top_5_fraction = cumulative / total;
    }
    if (mode == std::min(9, spectrum_size - 1)) {
      summary.top_10_fraction = cumulative / total;
    }
    if (summary.rank_90 == 0 && cumulative >= 0.9 * total) {
      summary.rank_90 = mode + 1;
    }
    if (summary.rank_99 == 0 && cumulative >= 0.99 * total) {
      summary.rank_99 = mode + 1;
    }
  }
  return summary;
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

    const auto adjoint_start = std::chrono::steady_clock::now();
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
    apply_timing_totals_.response_low_rank_adjoint_wall_time_seconds +=
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - adjoint_start).count();
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
