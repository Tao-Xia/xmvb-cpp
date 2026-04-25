#include "pfaffian_vbscf/kernel/pf_fixed_ms_open_shell_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/LU>
#include <Eigen/QR>

#include "pfaffian_vbscf/math/antisymm_codec.hpp"
#include "pfaffian_vbscf/tensor/pf_tensor_contractor.hpp"

namespace xmvb::pfaffian_vbscf {

namespace {

constexpr double kInterpolationNodeScale = 0.008;

void validate_square_matrix(
    const ConstMatrixRef& matrix,
    const char* label) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument(std::string(label) + " must be square");
  }
}

std::vector<int> build_pair_orbitals(
    int n_active_orbitals,
    const std::vector<int>& blocked_alpha_orbitals,
    const std::vector<int>& blocked_beta_orbitals,
    const char* label_prefix) {
  std::vector<bool> blocked_mask(
      n_active_orbitals,
      false);
  for (const int orbital : blocked_alpha_orbitals) {
    if (orbital < 0 || orbital >= n_active_orbitals) {
      throw std::invalid_argument(
          std::string(label_prefix) +
          "_blocked_alpha_orbitals index is outside the active-space range");
    }
    if (blocked_mask[orbital]) {
      throw std::invalid_argument(
          std::string(label_prefix) +
          " blocked alpha orbitals contain a duplicated active orbital");
    }
    blocked_mask[orbital] = true;
  }
  for (const int orbital : blocked_beta_orbitals) {
    if (orbital < 0 || orbital >= n_active_orbitals) {
      throw std::invalid_argument(
          std::string(label_prefix) +
          "_blocked_beta_orbitals index is outside the active-space range");
    }
    if (blocked_mask[orbital]) {
      throw std::invalid_argument(
          std::string(label_prefix) +
          " blocked alpha/beta orbitals reuse an active orbital");
    }
    blocked_mask[orbital] = true;
  }

  std::vector<int> pair_orbitals;
  pair_orbitals.reserve(
      std::max(
          0,
          n_active_orbitals -
              static_cast<int>(blocked_alpha_orbitals.size()) -
              static_cast<int>(blocked_beta_orbitals.size())));
  for (int orbital = 0; orbital < n_active_orbitals; ++orbital) {
    if (!blocked_mask[orbital]) {
      pair_orbitals.push_back(orbital);
    }
  }
  return pair_orbitals;
}

Matrix extract_submatrix(
    const ConstMatrixRef& matrix,
    const std::vector<int>& row_indices,
    const std::vector<int>& col_indices) {
  Matrix submatrix = Matrix::Zero(
      static_cast<int>(row_indices.size()),
      static_cast<int>(col_indices.size()));
  for (int col = 0; col < static_cast<int>(col_indices.size()); ++col) {
    for (int row = 0; row < static_cast<int>(row_indices.size()); ++row) {
      submatrix(row, col) =
          matrix(
              row_indices[row],
              col_indices[col]);
    }
  }
  return submatrix;
}

void accumulate_submatrix(
    Matrix* target,
    const std::vector<int>& row_indices,
    const std::vector<int>& col_indices,
    const ConstMatrixRef& source) {
  if (target == nullptr) {
    throw std::invalid_argument("target must not be null");
  }
  if (source.rows() != static_cast<int>(row_indices.size()) ||
      source.cols() != static_cast<int>(col_indices.size())) {
    throw std::invalid_argument("source dimensions do not match the index sets");
  }
  for (int col = 0; col < source.cols(); ++col) {
    for (int row = 0; row < source.rows(); ++row) {
      (*target)(
          row_indices[row],
          col_indices[col]) += source(row, col);
    }
  }
}

Matrix scatter_submatrix(
    int dimension,
    const std::vector<int>& row_indices,
    const std::vector<int>& col_indices,
    const ConstMatrixRef& submatrix) {
  if (submatrix.rows() != static_cast<int>(row_indices.size()) ||
      submatrix.cols() != static_cast<int>(col_indices.size())) {
    throw std::invalid_argument("scatter_submatrix dimensions do not match the index sets");
  }
  Matrix matrix = Matrix::Zero(dimension, dimension);
  for (int col = 0; col < static_cast<int>(col_indices.size()); ++col) {
    for (int row = 0; row < static_cast<int>(row_indices.size()); ++row) {
      matrix(
          row_indices[row],
          col_indices[col]) = submatrix(row, col);
    }
  }
  return matrix;
}

struct InterpolationCache {
  std::vector<double> nodes;
  Eigen::MatrixXd inverse_vandermonde;
};

Eigen::MatrixXd build_vandermonde_inverse(int degree) {
  if (degree < 0) {
    throw std::invalid_argument("degree must be non-negative");
  }

  Eigen::MatrixXd vandermonde =
      Eigen::MatrixXd::Zero(degree + 1, degree + 1);
  for (int node_index = 0; node_index <= degree; ++node_index) {
    const double node_value =
        kInterpolationNodeScale * static_cast<double>(node_index);
    double power_value = 1.0;
    for (int power = 0; power <= degree; ++power) {
      vandermonde(node_index, power) = power_value;
      power_value *= node_value;
    }
  }

  return Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>(vandermonde)
      .solve(Eigen::MatrixXd::Identity(degree + 1, degree + 1));
}

const InterpolationCache& get_interpolation_cache(int degree) {
  static std::mutex cache_mutex;
  static std::map<int, InterpolationCache> cache;

  std::lock_guard<std::mutex> lock(cache_mutex);
  auto [iterator, inserted] = cache.try_emplace(degree);
  if (inserted) {
    InterpolationCache& interpolation_cache = iterator->second;
    interpolation_cache.nodes.resize(
        degree + 1,
        0.0);
    for (int node = 0; node <= degree; ++node) {
      interpolation_cache.nodes[node] =
          kInterpolationNodeScale * static_cast<double>(node);
    }
    interpolation_cache.inverse_vandermonde =
        build_vandermonde_inverse(degree);
  }
  return iterator->second;
}

struct CandidateContext {
  int n_active_orbitals = 0;
  int n_singlet_pairs = 0;
  int interpolation_degree = 0;
  std::vector<int> left_blocked_alpha_orbitals;
  std::vector<int> left_blocked_beta_orbitals;
  std::vector<int> right_blocked_alpha_orbitals;
  std::vector<int> right_blocked_beta_orbitals;
  std::vector<int> left_pair_orbitals;
  std::vector<int> right_pair_orbitals;
  std::vector<int> left_alpha_ordered_orbitals;
  std::vector<int> right_alpha_ordered_orbitals;
  std::vector<int> left_beta_ordered_orbitals;
  std::vector<int> right_beta_ordered_orbitals;
  Matrix left_pair_ba_reduced;
  Matrix right_pair_ab_reduced;
  Matrix blocked_overlap_alpha;
  Matrix blocked_overlap_beta;
  Matrix left_pair_to_block_alpha;
  Matrix left_pair_to_block_beta;
  Matrix blocked_alpha_to_right_pair;
  Matrix blocked_beta_to_right_pair;
  Matrix pair_overlap;
  Matrix alpha_upper_right;
  Matrix alpha_lower_left;
  Matrix pair_kernel_seed;
  Matrix beta_top_right_seed;
  Matrix beta_bottom_left_seed;
  Matrix bottom_left_alpha_seed;
  Matrix bottom_right_seed;
  Eigen::MatrixXd inverse_vandermonde;
  std::vector<double> interpolation_nodes;
};

struct SampleEvaluation {
  double determinant = 0.0;
  Matrix sample_matrix_inverse;
  Matrix spatial_gradient;
};

struct FastSampleOperands {
  double determinant = 0.0;
  Matrix u_alpha;
  Matrix v_alpha;
  Matrix u_beta;
  Matrix v_beta;
  Matrix inverse_times_u_alpha;
  Matrix inverse_times_u_beta;
  Matrix alpha_trace_ordered;
  Matrix beta_trace_ordered;
  Matrix opposite_exchange_left_ordered;
  Matrix opposite_exchange_difference_right_ordered;
  Matrix alpha_trace_matrix;
  Matrix beta_trace_matrix;
  Matrix opposite_exchange_left_matrix;
  Matrix opposite_exchange_difference_right_matrix;
  Matrix opposite_mixed_right_matrix;
};

CandidateContext build_candidate_context(
    const ConstMatrixRef& left_pair_ba,
    const ConstMatrixRef& right_pair_ab,
    const std::vector<int>& left_blocked_alpha_orbitals,
    const std::vector<int>& left_blocked_beta_orbitals,
    const std::vector<int>& right_blocked_alpha_orbitals,
    const std::vector<int>& right_blocked_beta_orbitals,
    const ConstMatrixRef& spatial_overlap,
    int n_singlet_pairs) {
  const int n_active_orbitals = static_cast<int>(spatial_overlap.rows());
  CandidateContext context;
  context.n_active_orbitals = n_active_orbitals;
  context.n_singlet_pairs = n_singlet_pairs;
  context.left_blocked_alpha_orbitals = left_blocked_alpha_orbitals;
  context.left_blocked_beta_orbitals = left_blocked_beta_orbitals;
  context.right_blocked_alpha_orbitals = right_blocked_alpha_orbitals;
  context.right_blocked_beta_orbitals = right_blocked_beta_orbitals;
  context.left_pair_orbitals =
      build_pair_orbitals(
          n_active_orbitals,
          left_blocked_alpha_orbitals,
          left_blocked_beta_orbitals,
          "left");
  context.right_pair_orbitals =
      build_pair_orbitals(
          n_active_orbitals,
          right_blocked_alpha_orbitals,
          right_blocked_beta_orbitals,
          "right");
  context.left_alpha_ordered_orbitals = context.left_blocked_alpha_orbitals;
  context.left_alpha_ordered_orbitals.insert(
      context.left_alpha_ordered_orbitals.end(),
      context.left_pair_orbitals.begin(),
      context.left_pair_orbitals.end());
  context.right_alpha_ordered_orbitals = context.right_blocked_alpha_orbitals;
  context.right_alpha_ordered_orbitals.insert(
      context.right_alpha_ordered_orbitals.end(),
      context.right_pair_orbitals.begin(),
      context.right_pair_orbitals.end());
  context.left_beta_ordered_orbitals = context.left_blocked_beta_orbitals;
  context.left_beta_ordered_orbitals.insert(
      context.left_beta_ordered_orbitals.end(),
      context.left_pair_orbitals.begin(),
      context.left_pair_orbitals.end());
  context.right_beta_ordered_orbitals = context.right_blocked_beta_orbitals;
  context.right_beta_ordered_orbitals.insert(
      context.right_beta_ordered_orbitals.end(),
      context.right_pair_orbitals.begin(),
      context.right_pair_orbitals.end());
  if (context.left_pair_orbitals.size() != context.right_pair_orbitals.size()) {
    throw std::invalid_argument(
        "left/right fixed-M_s open-shell states do not leave the same number of pair orbitals");
  }

  context.interpolation_degree =
      static_cast<int>(context.left_pair_orbitals.size());
  if (n_singlet_pairs > context.interpolation_degree) {
    throw std::invalid_argument("n_singlet_pairs exceeds the interpolation degree");
  }
  if (static_cast<int>(context.left_pair_orbitals.size()) < n_singlet_pairs ||
      static_cast<int>(context.right_pair_orbitals.size()) < n_singlet_pairs) {
    throw std::invalid_argument(
        "blocked open-shell orbitals leave too few pair orbitals for n_singlet_pairs");
  }

  context.left_pair_ba_reduced =
      extract_submatrix(
          left_pair_ba,
          context.left_pair_orbitals,
          context.left_pair_orbitals);
  context.right_pair_ab_reduced =
      extract_submatrix(
          right_pair_ab,
          context.right_pair_orbitals,
          context.right_pair_orbitals);
  context.blocked_overlap_alpha =
      extract_submatrix(
          spatial_overlap,
          context.left_blocked_alpha_orbitals,
          context.right_blocked_alpha_orbitals);
  context.blocked_overlap_beta =
      extract_submatrix(
          spatial_overlap,
          context.left_blocked_beta_orbitals,
          context.right_blocked_beta_orbitals);
  context.left_pair_to_block_alpha =
      extract_submatrix(
          spatial_overlap,
          context.left_pair_orbitals,
          context.right_blocked_alpha_orbitals);
  context.left_pair_to_block_beta =
      extract_submatrix(
          spatial_overlap,
          context.left_pair_orbitals,
          context.right_blocked_beta_orbitals);
  context.blocked_alpha_to_right_pair =
      extract_submatrix(
          spatial_overlap,
          context.left_blocked_alpha_orbitals,
          context.right_pair_orbitals);
  context.blocked_beta_to_right_pair =
      extract_submatrix(
          spatial_overlap,
          context.left_blocked_beta_orbitals,
          context.right_pair_orbitals);
  context.pair_overlap =
      extract_submatrix(
          spatial_overlap,
          context.left_pair_orbitals,
          context.right_pair_orbitals);

  context.alpha_upper_right =
      context.blocked_alpha_to_right_pair *
      context.right_pair_ab_reduced;
  context.alpha_lower_left =
      context.left_pair_ba_reduced *
      context.left_pair_to_block_alpha;
  context.pair_kernel_seed =
      context.left_pair_ba_reduced *
      context.pair_overlap *
      context.right_pair_ab_reduced;
  context.beta_top_right_seed =
      context.left_pair_to_block_beta.transpose();
  context.beta_bottom_left_seed =
      context.blocked_beta_to_right_pair.transpose();
  context.bottom_left_alpha_seed =
      context.pair_overlap.transpose() *
      context.alpha_lower_left;
  context.bottom_right_seed =
      context.pair_overlap.transpose() *
      context.pair_kernel_seed;

  const InterpolationCache& interpolation_cache =
      get_interpolation_cache(context.interpolation_degree);
  context.inverse_vandermonde = interpolation_cache.inverse_vandermonde;
  context.interpolation_nodes = interpolation_cache.nodes;
  return context;
}

SampleEvaluation evaluate_sample(
    const CandidateContext& context,
    double t_value) {
  const int n_active_orbitals = context.n_active_orbitals;
  const int n_blocked_alpha =
      static_cast<int>(context.left_blocked_alpha_orbitals.size());
  const int n_blocked_beta =
      static_cast<int>(context.left_blocked_beta_orbitals.size());
  const int n_pair_orbitals =
      static_cast<int>(context.left_pair_orbitals.size());

  const Matrix beta_alpha_coupling =
      context.beta_top_right_seed *
      context.alpha_lower_left;
  const Matrix beta_pair_top_right =
      context.beta_top_right_seed *
      context.pair_kernel_seed;

  Matrix sample_matrix = Matrix::Zero(
      n_blocked_beta + n_blocked_alpha + n_pair_orbitals,
      n_blocked_beta + n_blocked_alpha + n_pair_orbitals);
  if (n_blocked_beta > 0) {
    sample_matrix.topLeftCorner(
        n_blocked_beta,
        n_blocked_beta) =
        context.blocked_overlap_beta.transpose();
    if (n_blocked_alpha > 0) {
      sample_matrix.block(
          0,
          n_blocked_beta,
          n_blocked_beta,
          n_blocked_alpha) = beta_alpha_coupling;
    }
    if (n_pair_orbitals > 0) {
      sample_matrix.topRightCorner(
          n_blocked_beta,
          n_pair_orbitals) =
          t_value * beta_pair_top_right;
    }
  }
  if (n_blocked_alpha > 0) {
    sample_matrix.block(
        n_blocked_beta,
        n_blocked_beta,
        n_blocked_alpha,
        n_blocked_alpha) =
        context.blocked_overlap_alpha;
    if (n_pair_orbitals > 0) {
      sample_matrix.block(
          n_blocked_beta,
          n_blocked_beta + n_blocked_alpha,
          n_blocked_alpha,
          n_pair_orbitals) =
          t_value * context.alpha_upper_right;
    }
  }
  if (n_pair_orbitals > 0) {
    if (n_blocked_beta > 0) {
      sample_matrix.bottomLeftCorner(
          n_pair_orbitals,
          n_blocked_beta) =
          context.beta_bottom_left_seed;
    }
    if (n_blocked_alpha > 0) {
      sample_matrix.block(
          n_blocked_beta + n_blocked_alpha,
          n_blocked_beta,
          n_pair_orbitals,
          n_blocked_alpha) =
          context.bottom_left_alpha_seed;
    }
    sample_matrix.bottomRightCorner(
        n_pair_orbitals,
        n_pair_orbitals) =
        Matrix::Identity(n_pair_orbitals, n_pair_orbitals) +
        t_value * context.bottom_right_seed;
  }

  const Eigen::FullPivLU<Matrix> lu(sample_matrix);
  SampleEvaluation sample;
  sample.determinant = lu.determinant();
  if (!std::isfinite(sample.determinant)) {
    throw std::runtime_error("sample determinant is not finite");
  }
  sample.sample_matrix_inverse = lu.inverse();
  const Matrix sample_matrix_inverse_transpose =
      sample.sample_matrix_inverse.transpose();
  const Matrix sample_matrix_adjoint =
      sample.determinant *
      sample_matrix_inverse_transpose;

  Matrix blocked_overlap_alpha_adjoint =
      Matrix::Zero(n_blocked_alpha, n_blocked_alpha);
  Matrix blocked_overlap_beta_adjoint =
      Matrix::Zero(n_blocked_beta, n_blocked_beta);
  Matrix left_pair_to_block_alpha_adjoint =
      Matrix::Zero(n_pair_orbitals, n_blocked_alpha);
  Matrix left_pair_to_block_beta_adjoint =
      Matrix::Zero(n_pair_orbitals, n_blocked_beta);
  Matrix blocked_alpha_to_right_pair_adjoint =
      Matrix::Zero(n_blocked_alpha, n_pair_orbitals);
  Matrix blocked_beta_to_right_pair_adjoint =
      Matrix::Zero(n_blocked_beta, n_pair_orbitals);
  Matrix pair_overlap_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix left_pair_ba_reduced_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix right_pair_ab_reduced_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);

  Matrix alpha_upper_right_adjoint =
      Matrix::Zero(n_blocked_alpha, n_pair_orbitals);
  Matrix alpha_lower_left_adjoint =
      Matrix::Zero(n_pair_orbitals, n_blocked_alpha);
  Matrix pair_kernel_seed_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix beta_top_right_seed_adjoint =
      Matrix::Zero(n_blocked_beta, n_pair_orbitals);
  Matrix beta_bottom_left_seed_adjoint =
      Matrix::Zero(n_pair_orbitals, n_blocked_beta);
  Matrix bottom_left_alpha_seed_adjoint =
      Matrix::Zero(n_pair_orbitals, n_blocked_alpha);
  Matrix bottom_right_seed_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);

  Matrix beta_alpha_coupling_adjoint =
      Matrix::Zero(n_blocked_beta, n_blocked_alpha);
  if (n_blocked_beta > 0 && n_blocked_alpha > 0) {
    beta_alpha_coupling_adjoint =
        sample_matrix_adjoint.block(
            0,
            n_blocked_beta,
            n_blocked_beta,
            n_blocked_alpha);
  }
  Matrix beta_pair_top_right_adjoint =
      Matrix::Zero(n_blocked_beta, n_pair_orbitals);
  if (n_blocked_beta > 0 && n_pair_orbitals > 0) {
    beta_pair_top_right_adjoint =
        t_value *
        sample_matrix_adjoint.topRightCorner(
            n_blocked_beta,
            n_pair_orbitals);
  }

  if (n_blocked_beta > 0) {
    blocked_overlap_beta_adjoint.noalias() +=
        sample_matrix_adjoint.topLeftCorner(
            n_blocked_beta,
            n_blocked_beta).transpose();
  }
  if (n_blocked_alpha > 0) {
    blocked_overlap_alpha_adjoint.noalias() +=
        sample_matrix_adjoint.block(
            n_blocked_beta,
            n_blocked_beta,
            n_blocked_alpha,
            n_blocked_alpha);
  }
  if (n_blocked_alpha > 0 && n_pair_orbitals > 0) {
    alpha_upper_right_adjoint.noalias() +=
        t_value *
        sample_matrix_adjoint.block(
            n_blocked_beta,
            n_blocked_beta + n_blocked_alpha,
            n_blocked_alpha,
            n_pair_orbitals);
    bottom_left_alpha_seed_adjoint.noalias() +=
        sample_matrix_adjoint.block(
            n_blocked_beta + n_blocked_alpha,
            n_blocked_beta,
            n_pair_orbitals,
            n_blocked_alpha);
  }
  if (n_blocked_beta > 0 && n_pair_orbitals > 0) {
    beta_bottom_left_seed_adjoint.noalias() +=
        sample_matrix_adjoint.bottomLeftCorner(
            n_pair_orbitals,
            n_blocked_beta);
  }
  if (n_pair_orbitals > 0) {
    bottom_right_seed_adjoint.noalias() +=
        t_value *
        sample_matrix_adjoint.bottomRightCorner(
            n_pair_orbitals,
            n_pair_orbitals);
  }

  if (n_blocked_beta > 0 && n_blocked_alpha > 0) {
    beta_top_right_seed_adjoint.noalias() +=
        beta_alpha_coupling_adjoint *
        context.alpha_lower_left.transpose();
    alpha_lower_left_adjoint.noalias() +=
        context.beta_top_right_seed.transpose() *
        beta_alpha_coupling_adjoint;
  }
  if (n_blocked_beta > 0 && n_pair_orbitals > 0) {
    beta_top_right_seed_adjoint.noalias() +=
        beta_pair_top_right_adjoint *
        context.pair_kernel_seed.transpose();
    pair_kernel_seed_adjoint.noalias() +=
        context.beta_top_right_seed.transpose() *
        beta_pair_top_right_adjoint;
  }

  if (n_pair_orbitals > 0 && n_blocked_alpha > 0) {
    pair_overlap_adjoint.noalias() +=
        context.alpha_lower_left *
        bottom_left_alpha_seed_adjoint.transpose();
    alpha_lower_left_adjoint.noalias() +=
        context.pair_overlap *
        bottom_left_alpha_seed_adjoint;
  }
  if (n_pair_orbitals > 0) {
    pair_overlap_adjoint.noalias() +=
        context.pair_kernel_seed *
        bottom_right_seed_adjoint.transpose();
    pair_kernel_seed_adjoint.noalias() +=
        context.pair_overlap *
        bottom_right_seed_adjoint;
  }

  if (n_pair_orbitals > 0 && n_blocked_beta > 0) {
    left_pair_to_block_beta_adjoint.noalias() +=
        beta_top_right_seed_adjoint.transpose();
    blocked_beta_to_right_pair_adjoint.noalias() +=
        beta_bottom_left_seed_adjoint.transpose();
  }

  if (n_pair_orbitals > 0 && n_blocked_alpha > 0) {
    blocked_alpha_to_right_pair_adjoint.noalias() +=
        alpha_upper_right_adjoint *
        context.right_pair_ab_reduced.transpose();
    right_pair_ab_reduced_adjoint.noalias() +=
        context.blocked_alpha_to_right_pair.transpose() *
        alpha_upper_right_adjoint;
    left_pair_ba_reduced_adjoint.noalias() +=
        alpha_lower_left_adjoint *
        context.left_pair_to_block_alpha.transpose();
    left_pair_to_block_alpha_adjoint.noalias() +=
        context.left_pair_ba_reduced.transpose() *
        alpha_lower_left_adjoint;
  }

  if (n_pair_orbitals > 0) {
    const Matrix pair_times_right =
        context.pair_overlap * context.right_pair_ab_reduced;
    left_pair_ba_reduced_adjoint.noalias() +=
        pair_kernel_seed_adjoint *
        pair_times_right.transpose();
    pair_overlap_adjoint.noalias() +=
        context.left_pair_ba_reduced.transpose() *
        pair_kernel_seed_adjoint *
        context.right_pair_ab_reduced.transpose();
    right_pair_ab_reduced_adjoint.noalias() +=
        (context.left_pair_ba_reduced * context.pair_overlap).transpose() *
        pair_kernel_seed_adjoint;
  }

  sample.spatial_gradient =
      Matrix::Zero(n_active_orbitals, n_active_orbitals);
  accumulate_submatrix(
      &sample.spatial_gradient,
      context.left_blocked_alpha_orbitals,
      context.right_blocked_alpha_orbitals,
      blocked_overlap_alpha_adjoint);
  accumulate_submatrix(
      &sample.spatial_gradient,
      context.left_blocked_beta_orbitals,
      context.right_blocked_beta_orbitals,
      blocked_overlap_beta_adjoint);
  accumulate_submatrix(
      &sample.spatial_gradient,
      context.left_pair_orbitals,
      context.right_blocked_alpha_orbitals,
      left_pair_to_block_alpha_adjoint);
  accumulate_submatrix(
      &sample.spatial_gradient,
      context.left_pair_orbitals,
      context.right_blocked_beta_orbitals,
      left_pair_to_block_beta_adjoint);
  accumulate_submatrix(
      &sample.spatial_gradient,
      context.left_blocked_alpha_orbitals,
      context.right_pair_orbitals,
      blocked_alpha_to_right_pair_adjoint);
  accumulate_submatrix(
      &sample.spatial_gradient,
      context.left_blocked_beta_orbitals,
      context.right_pair_orbitals,
      blocked_beta_to_right_pair_adjoint);
  accumulate_submatrix(
      &sample.spatial_gradient,
      context.left_pair_orbitals,
      context.right_pair_orbitals,
      pair_overlap_adjoint);
  return sample;
}

FastSampleOperands build_fast_sample_operands(
    const CandidateContext& context,
    const SampleEvaluation& sample,
    double t_value) {
  const int n_active_orbitals = context.n_active_orbitals;
  const int n_blocked_alpha =
      static_cast<int>(context.left_blocked_alpha_orbitals.size());
  const int n_blocked_beta =
      static_cast<int>(context.left_blocked_beta_orbitals.size());
  const int n_pair_orbitals =
      static_cast<int>(context.left_pair_orbitals.size());
  const int sample_dimension =
      n_blocked_beta + n_blocked_alpha + n_pair_orbitals;
  const Matrix pair_left_factor =
      context.pair_overlap.transpose() *
      context.left_pair_ba_reduced;

  Matrix u_alpha =
      Matrix::Zero(
          sample_dimension,
          n_blocked_alpha + n_pair_orbitals);
  Matrix v_alpha =
      Matrix::Zero(
          sample_dimension,
          n_blocked_alpha + n_pair_orbitals);
  if (n_blocked_alpha > 0) {
    u_alpha.block(
        n_blocked_beta,
        0,
        n_blocked_alpha,
        n_blocked_alpha) =
        Matrix::Identity(n_blocked_alpha, n_blocked_alpha);
    v_alpha.block(
        n_blocked_beta,
        0,
        n_blocked_alpha,
        n_blocked_alpha) =
        Matrix::Identity(n_blocked_alpha, n_blocked_alpha);
  }
  if (n_pair_orbitals > 0) {
    if (n_blocked_beta > 0) {
      u_alpha.block(
          0,
          n_blocked_alpha,
          n_blocked_beta,
          n_pair_orbitals) =
          context.beta_top_right_seed *
          context.left_pair_ba_reduced;
    }
    u_alpha.block(
        n_blocked_beta + n_blocked_alpha,
        n_blocked_alpha,
        n_pair_orbitals,
        n_pair_orbitals) =
        pair_left_factor;
    v_alpha.block(
        n_blocked_beta + n_blocked_alpha,
        n_blocked_alpha,
        n_pair_orbitals,
        n_pair_orbitals) =
        t_value *
        context.right_pair_ab_reduced.transpose();
  }

  Matrix u_beta =
      Matrix::Zero(
          sample_dimension,
          n_blocked_beta + n_pair_orbitals);
  Matrix v_beta =
      Matrix::Zero(
          sample_dimension,
          n_blocked_beta + n_pair_orbitals);
  if (n_blocked_beta > 0) {
    u_beta.topLeftCorner(
        n_blocked_beta,
        n_blocked_beta) =
        Matrix::Identity(n_blocked_beta, n_blocked_beta);
    v_beta.topLeftCorner(
        n_blocked_beta,
        n_blocked_beta) =
        Matrix::Identity(n_blocked_beta, n_blocked_beta);
  }
  if (n_pair_orbitals > 0) {
    u_beta.block(
        n_blocked_beta + n_blocked_alpha,
        n_blocked_beta,
        n_pair_orbitals,
        n_pair_orbitals) =
        Matrix::Identity(n_pair_orbitals, n_pair_orbitals);
    if (n_blocked_alpha > 0) {
      v_beta.block(
          n_blocked_beta,
          n_blocked_beta,
          n_blocked_alpha,
          n_pair_orbitals) =
          context.alpha_lower_left.transpose();
    }
    v_beta.block(
        n_blocked_beta + n_blocked_alpha,
        n_blocked_beta,
        n_pair_orbitals,
        n_pair_orbitals) =
        t_value *
        context.pair_kernel_seed.transpose();
  }

  const Matrix inverse_times_u_alpha =
      sample.sample_matrix_inverse * u_alpha;
  const Matrix inverse_times_u_beta =
      sample.sample_matrix_inverse * u_beta;

  const Matrix alpha_trace_ordered =
      inverse_times_u_alpha.transpose() * v_alpha;
  const Matrix beta_trace_ordered =
      v_beta.transpose() * inverse_times_u_beta;
  const Matrix opposite_exchange_left_ordered =
      v_alpha.transpose() * inverse_times_u_beta;
  const Matrix opposite_exchange_difference_right_ordered =
      v_beta.transpose() * inverse_times_u_alpha;
  Matrix opposite_mixed_right_ordered =
      Matrix::Zero(
          n_blocked_beta + n_pair_orbitals,
          n_blocked_alpha + n_pair_orbitals);
  if (n_pair_orbitals > 0) {
    opposite_mixed_right_ordered.bottomRightCorner(
        n_pair_orbitals,
        n_pair_orbitals) =
        context.left_pair_ba_reduced;
  }

  FastSampleOperands operands;
  operands.determinant = sample.determinant;
  operands.u_alpha = std::move(u_alpha);
  operands.v_alpha = std::move(v_alpha);
  operands.u_beta = std::move(u_beta);
  operands.v_beta = std::move(v_beta);
  operands.inverse_times_u_alpha = inverse_times_u_alpha;
  operands.inverse_times_u_beta = inverse_times_u_beta;
  operands.alpha_trace_ordered = alpha_trace_ordered;
  operands.beta_trace_ordered = beta_trace_ordered;
  operands.opposite_exchange_left_ordered =
      opposite_exchange_left_ordered;
  operands.opposite_exchange_difference_right_ordered =
      opposite_exchange_difference_right_ordered;
  operands.alpha_trace_matrix =
      scatter_submatrix(
          n_active_orbitals,
          context.left_alpha_ordered_orbitals,
          context.right_alpha_ordered_orbitals,
          alpha_trace_ordered);
  operands.beta_trace_matrix =
      scatter_submatrix(
          n_active_orbitals,
          context.left_beta_ordered_orbitals,
          context.right_beta_ordered_orbitals,
          beta_trace_ordered);
  operands.opposite_exchange_left_matrix =
      scatter_submatrix(
          n_active_orbitals,
          context.right_alpha_ordered_orbitals,
          context.right_beta_ordered_orbitals,
          opposite_exchange_left_ordered);
  operands.opposite_exchange_difference_right_matrix =
      scatter_submatrix(
          n_active_orbitals,
          context.left_beta_ordered_orbitals,
          context.left_alpha_ordered_orbitals,
          opposite_exchange_difference_right_ordered);
  operands.opposite_mixed_right_matrix =
      scatter_submatrix(
          n_active_orbitals,
          context.left_beta_ordered_orbitals,
          context.left_alpha_ordered_orbitals,
          opposite_mixed_right_ordered);
  return operands;
}

void accumulate_same_operand_adjoint(
    double weight,
    const ScalarBuffer& packed_two_electron_integrals,
    const ConstMatrixRef& operand,
    void (*adjoint_function)(
        double,
        const ConstMatrixRef&,
        const ConstMatrixRef&,
        const ScalarBuffer&,
        MatrixRef,
        MatrixRef),
    Matrix* operand_adjoint) {
  if (operand_adjoint == nullptr) {
    throw std::invalid_argument("operand_adjoint must not be null");
  }
  Matrix left_adjoint =
      Matrix::Zero(operand.rows(), operand.cols());
  Matrix right_adjoint =
      Matrix::Zero(operand.rows(), operand.cols());
  adjoint_function(
      weight,
      operand,
      operand,
      packed_two_electron_integrals,
      left_adjoint,
      right_adjoint);
  operand_adjoint->noalias() += left_adjoint + right_adjoint;
}

Matrix scatter_base_block_adjoints_to_spatial_gradient(
    const CandidateContext& context,
    const ConstMatrixRef& blocked_overlap_alpha_adjoint,
    const ConstMatrixRef& blocked_overlap_beta_adjoint,
    const ConstMatrixRef& left_pair_to_block_alpha_adjoint,
    const ConstMatrixRef& left_pair_to_block_beta_adjoint,
    const ConstMatrixRef& blocked_alpha_to_right_pair_adjoint,
    const ConstMatrixRef& blocked_beta_to_right_pair_adjoint,
    const ConstMatrixRef& pair_overlap_adjoint) {
  Matrix spatial_gradient =
      Matrix::Zero(context.n_active_orbitals, context.n_active_orbitals);
  accumulate_submatrix(
      &spatial_gradient,
      context.left_blocked_alpha_orbitals,
      context.right_blocked_alpha_orbitals,
      blocked_overlap_alpha_adjoint);
  accumulate_submatrix(
      &spatial_gradient,
      context.left_blocked_beta_orbitals,
      context.right_blocked_beta_orbitals,
      blocked_overlap_beta_adjoint);
  accumulate_submatrix(
      &spatial_gradient,
      context.left_pair_orbitals,
      context.right_blocked_alpha_orbitals,
      left_pair_to_block_alpha_adjoint);
  accumulate_submatrix(
      &spatial_gradient,
      context.left_pair_orbitals,
      context.right_blocked_beta_orbitals,
      left_pair_to_block_beta_adjoint);
  accumulate_submatrix(
      &spatial_gradient,
      context.left_blocked_alpha_orbitals,
      context.right_pair_orbitals,
      blocked_alpha_to_right_pair_adjoint);
  accumulate_submatrix(
      &spatial_gradient,
      context.left_blocked_beta_orbitals,
      context.right_pair_orbitals,
      blocked_beta_to_right_pair_adjoint);
  accumulate_submatrix(
      &spatial_gradient,
      context.left_pair_orbitals,
      context.right_pair_orbitals,
      pair_overlap_adjoint);
  return spatial_gradient;
}

Matrix backpropagate_sample_pullback_only(
    const CandidateContext& context,
    const SampleEvaluation& sample,
    double t_value,
    double determinant_adjoint,
    const ConstMatrixRef& sample_inverse_adjoint) {
  if (sample_inverse_adjoint.rows() != sample.sample_matrix_inverse.rows() ||
      sample_inverse_adjoint.cols() != sample.sample_matrix_inverse.cols()) {
    throw std::invalid_argument("sample_inverse_adjoint dimensions do not match the sample inverse");
  }

  const int n_blocked_alpha =
      static_cast<int>(context.left_blocked_alpha_orbitals.size());
  const int n_blocked_beta =
      static_cast<int>(context.left_blocked_beta_orbitals.size());
  const int n_pair_orbitals =
      static_cast<int>(context.left_pair_orbitals.size());

  const Matrix sample_inverse_transpose =
      sample.sample_matrix_inverse.transpose();
  Matrix sample_matrix_adjoint =
      determinant_adjoint *
      sample.determinant *
      sample_inverse_transpose;
  sample_matrix_adjoint.noalias() -=
      sample_inverse_transpose *
      sample_inverse_adjoint *
      sample_inverse_transpose;

  Matrix blocked_overlap_alpha_adjoint =
      Matrix::Zero(n_blocked_alpha, n_blocked_alpha);
  Matrix blocked_overlap_beta_adjoint =
      Matrix::Zero(n_blocked_beta, n_blocked_beta);
  Matrix left_pair_to_block_alpha_adjoint =
      Matrix::Zero(n_pair_orbitals, n_blocked_alpha);
  Matrix left_pair_to_block_beta_adjoint =
      Matrix::Zero(n_pair_orbitals, n_blocked_beta);
  Matrix blocked_alpha_to_right_pair_adjoint =
      Matrix::Zero(n_blocked_alpha, n_pair_orbitals);
  Matrix blocked_beta_to_right_pair_adjoint =
      Matrix::Zero(n_blocked_beta, n_pair_orbitals);
  Matrix pair_overlap_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix alpha_upper_right_adjoint =
      Matrix::Zero(n_blocked_alpha, n_pair_orbitals);
  Matrix alpha_lower_left_adjoint =
      Matrix::Zero(n_pair_orbitals, n_blocked_alpha);
  Matrix pair_kernel_seed_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix beta_top_right_seed_adjoint =
      Matrix::Zero(n_blocked_beta, n_pair_orbitals);
  Matrix beta_bottom_left_seed_adjoint =
      Matrix::Zero(n_pair_orbitals, n_blocked_beta);
  Matrix bottom_left_alpha_seed_adjoint =
      Matrix::Zero(n_pair_orbitals, n_blocked_alpha);
  Matrix bottom_right_seed_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);

  Matrix beta_alpha_coupling_adjoint =
      Matrix::Zero(n_blocked_beta, n_blocked_alpha);
  if (n_blocked_beta > 0 && n_blocked_alpha > 0) {
    beta_alpha_coupling_adjoint =
        sample_matrix_adjoint.block(
            0,
            n_blocked_beta,
            n_blocked_beta,
            n_blocked_alpha);
  }
  Matrix beta_pair_top_right_adjoint =
      Matrix::Zero(n_blocked_beta, n_pair_orbitals);
  if (n_blocked_beta > 0 && n_pair_orbitals > 0) {
    beta_pair_top_right_adjoint =
        t_value *
        sample_matrix_adjoint.topRightCorner(
            n_blocked_beta,
            n_pair_orbitals);
  }

  if (n_blocked_beta > 0) {
    blocked_overlap_beta_adjoint.noalias() +=
        sample_matrix_adjoint.topLeftCorner(
            n_blocked_beta,
            n_blocked_beta).transpose();
  }
  if (n_blocked_alpha > 0) {
    blocked_overlap_alpha_adjoint.noalias() +=
        sample_matrix_adjoint.block(
            n_blocked_beta,
            n_blocked_beta,
            n_blocked_alpha,
            n_blocked_alpha);
  }
  if (n_blocked_alpha > 0 && n_pair_orbitals > 0) {
    alpha_upper_right_adjoint.noalias() +=
        t_value *
        sample_matrix_adjoint.block(
            n_blocked_beta,
            n_blocked_beta + n_blocked_alpha,
            n_blocked_alpha,
            n_pair_orbitals);
    bottom_left_alpha_seed_adjoint.noalias() +=
        sample_matrix_adjoint.block(
            n_blocked_beta + n_blocked_alpha,
            n_blocked_beta,
            n_pair_orbitals,
            n_blocked_alpha);
  }
  if (n_blocked_beta > 0 && n_pair_orbitals > 0) {
    beta_bottom_left_seed_adjoint.noalias() +=
        sample_matrix_adjoint.bottomLeftCorner(
            n_pair_orbitals,
            n_blocked_beta);
  }
  if (n_pair_orbitals > 0) {
    bottom_right_seed_adjoint.noalias() +=
        t_value *
        sample_matrix_adjoint.bottomRightCorner(
            n_pair_orbitals,
            n_pair_orbitals);
  }

  if (n_blocked_beta > 0 && n_blocked_alpha > 0) {
    beta_top_right_seed_adjoint.noalias() +=
        beta_alpha_coupling_adjoint *
        context.alpha_lower_left.transpose();
    alpha_lower_left_adjoint.noalias() +=
        context.beta_top_right_seed.transpose() *
        beta_alpha_coupling_adjoint;
  }
  if (n_blocked_beta > 0 && n_pair_orbitals > 0) {
    beta_top_right_seed_adjoint.noalias() +=
        beta_pair_top_right_adjoint *
        context.pair_kernel_seed.transpose();
    pair_kernel_seed_adjoint.noalias() +=
        context.beta_top_right_seed.transpose() *
        beta_pair_top_right_adjoint;
  }

  if (n_pair_orbitals > 0 && n_blocked_alpha > 0) {
    pair_overlap_adjoint.noalias() +=
        context.alpha_lower_left *
        bottom_left_alpha_seed_adjoint.transpose();
    alpha_lower_left_adjoint.noalias() +=
        context.pair_overlap *
        bottom_left_alpha_seed_adjoint;
  }
  if (n_pair_orbitals > 0) {
    pair_overlap_adjoint.noalias() +=
        context.pair_kernel_seed *
        bottom_right_seed_adjoint.transpose();
    pair_kernel_seed_adjoint.noalias() +=
        context.pair_overlap *
        bottom_right_seed_adjoint;
  }

  if (n_pair_orbitals > 0 && n_blocked_beta > 0) {
    left_pair_to_block_beta_adjoint.noalias() +=
        beta_top_right_seed_adjoint.transpose();
    blocked_beta_to_right_pair_adjoint.noalias() +=
        beta_bottom_left_seed_adjoint.transpose();
  }

  if (n_pair_orbitals > 0 && n_blocked_alpha > 0) {
    blocked_alpha_to_right_pair_adjoint.noalias() +=
        alpha_upper_right_adjoint *
        context.right_pair_ab_reduced.transpose();
    left_pair_to_block_alpha_adjoint.noalias() +=
        context.left_pair_ba_reduced.transpose() *
        alpha_lower_left_adjoint;
  }

  if (n_pair_orbitals > 0) {
    pair_overlap_adjoint.noalias() +=
        context.left_pair_ba_reduced.transpose() *
        pair_kernel_seed_adjoint *
        context.right_pair_ab_reduced.transpose();
  }

  return scatter_base_block_adjoints_to_spatial_gradient(
      context,
      blocked_overlap_alpha_adjoint,
      blocked_overlap_beta_adjoint,
      left_pair_to_block_alpha_adjoint,
      left_pair_to_block_beta_adjoint,
      blocked_alpha_to_right_pair_adjoint,
      blocked_beta_to_right_pair_adjoint,
      pair_overlap_adjoint);
}

Matrix backpropagate_fast_sample_operands_pullback_only(
    const CandidateContext& context,
    const SampleEvaluation& sample,
    const FastSampleOperands& operands,
    double t_value,
    const ConstMatrixRef& alpha_trace_adjoint,
    const ConstMatrixRef& beta_trace_adjoint,
    const ConstMatrixRef& opposite_exchange_left_adjoint,
    const ConstMatrixRef& opposite_exchange_difference_right_adjoint) {
  const int n_blocked_alpha =
      static_cast<int>(context.left_blocked_alpha_orbitals.size());
  const int n_blocked_beta =
      static_cast<int>(context.left_blocked_beta_orbitals.size());
  const int n_pair_orbitals =
      static_cast<int>(context.left_pair_orbitals.size());

  const Matrix alpha_trace_ordered_adjoint =
      extract_submatrix(
          alpha_trace_adjoint,
          context.left_alpha_ordered_orbitals,
          context.right_alpha_ordered_orbitals);
  const Matrix beta_trace_ordered_adjoint =
      extract_submatrix(
          beta_trace_adjoint,
          context.left_beta_ordered_orbitals,
          context.right_beta_ordered_orbitals);
  const Matrix opposite_exchange_left_ordered_adjoint =
      extract_submatrix(
          opposite_exchange_left_adjoint,
          context.right_alpha_ordered_orbitals,
          context.right_beta_ordered_orbitals);
  const Matrix opposite_exchange_difference_right_ordered_adjoint =
      extract_submatrix(
          opposite_exchange_difference_right_adjoint,
          context.left_beta_ordered_orbitals,
          context.left_alpha_ordered_orbitals);

  Matrix inverse_times_u_alpha_adjoint =
      operands.v_alpha *
      alpha_trace_ordered_adjoint.transpose();
  inverse_times_u_alpha_adjoint.noalias() +=
      operands.v_beta *
      opposite_exchange_difference_right_ordered_adjoint;

  Matrix inverse_times_u_beta_adjoint =
      operands.v_beta *
      beta_trace_ordered_adjoint;
  inverse_times_u_beta_adjoint.noalias() +=
      operands.v_alpha *
      opposite_exchange_left_ordered_adjoint;

  Matrix v_alpha_adjoint =
      operands.inverse_times_u_alpha *
      alpha_trace_ordered_adjoint;
  v_alpha_adjoint.noalias() +=
      operands.inverse_times_u_beta *
      opposite_exchange_left_ordered_adjoint.transpose();

  Matrix v_beta_adjoint =
      operands.inverse_times_u_beta *
      beta_trace_ordered_adjoint.transpose();
  v_beta_adjoint.noalias() +=
      operands.inverse_times_u_alpha *
      opposite_exchange_difference_right_ordered_adjoint.transpose();

  Matrix sample_inverse_adjoint =
      inverse_times_u_alpha_adjoint *
      operands.u_alpha.transpose();
  sample_inverse_adjoint.noalias() +=
      inverse_times_u_beta_adjoint *
      operands.u_beta.transpose();

  Matrix u_alpha_adjoint =
      sample.sample_matrix_inverse.transpose() *
      inverse_times_u_alpha_adjoint;
  Matrix u_beta_adjoint =
      sample.sample_matrix_inverse.transpose() *
      inverse_times_u_beta_adjoint;

  Matrix left_pair_to_block_alpha_adjoint =
      Matrix::Zero(n_pair_orbitals, n_blocked_alpha);
  Matrix left_pair_to_block_beta_adjoint =
      Matrix::Zero(n_pair_orbitals, n_blocked_beta);
  Matrix pair_overlap_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix blocked_overlap_alpha_adjoint =
      Matrix::Zero(n_blocked_alpha, n_blocked_alpha);
  Matrix blocked_overlap_beta_adjoint =
      Matrix::Zero(n_blocked_beta, n_blocked_beta);
  Matrix blocked_alpha_to_right_pair_adjoint =
      Matrix::Zero(n_blocked_alpha, n_pair_orbitals);
  Matrix blocked_beta_to_right_pair_adjoint =
      Matrix::Zero(n_blocked_beta, n_pair_orbitals);

  Matrix beta_top_right_seed_adjoint =
      Matrix::Zero(n_blocked_beta, n_pair_orbitals);
  Matrix pair_left_factor_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  if (n_pair_orbitals > 0 && n_blocked_beta > 0) {
    beta_top_right_seed_adjoint.noalias() +=
        u_alpha_adjoint.block(
            0,
            n_blocked_alpha,
            n_blocked_beta,
            n_pair_orbitals) *
        context.left_pair_ba_reduced.transpose();
  }
  if (n_pair_orbitals > 0) {
    pair_left_factor_adjoint.noalias() +=
        u_alpha_adjoint.block(
            n_blocked_beta + n_blocked_alpha,
            n_blocked_alpha,
            n_pair_orbitals,
            n_pair_orbitals);
  }

  Matrix alpha_lower_left_adjoint =
      Matrix::Zero(n_pair_orbitals, n_blocked_alpha);
  Matrix pair_kernel_seed_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  if (n_pair_orbitals > 0 && n_blocked_alpha > 0) {
    alpha_lower_left_adjoint.noalias() +=
        v_beta_adjoint.block(
            n_blocked_beta,
            n_blocked_beta,
            n_blocked_alpha,
            n_pair_orbitals).transpose();
  }
  if (n_pair_orbitals > 0) {
    pair_kernel_seed_adjoint.noalias() +=
        t_value *
        v_beta_adjoint.block(
            n_blocked_beta + n_blocked_alpha,
            n_blocked_beta,
            n_pair_orbitals,
            n_pair_orbitals).transpose();
  }

  if (n_pair_orbitals > 0 && n_blocked_beta > 0) {
    left_pair_to_block_beta_adjoint.noalias() +=
        beta_top_right_seed_adjoint.transpose();
  }
  if (n_pair_orbitals > 0 && n_blocked_alpha > 0) {
    left_pair_to_block_alpha_adjoint.noalias() +=
        context.left_pair_ba_reduced.transpose() *
        alpha_lower_left_adjoint;
  }
  if (n_pair_orbitals > 0) {
    pair_overlap_adjoint.noalias() +=
        context.left_pair_ba_reduced *
        pair_left_factor_adjoint.transpose();
    pair_overlap_adjoint.noalias() +=
        context.left_pair_ba_reduced.transpose() *
        pair_kernel_seed_adjoint *
        context.right_pair_ab_reduced.transpose();
  }

  Matrix spatial_gradient =
      backpropagate_sample_pullback_only(
          context,
          sample,
          t_value,
          0.0,
          sample_inverse_adjoint);
  spatial_gradient.noalias() +=
      scatter_base_block_adjoints_to_spatial_gradient(
          context,
          blocked_overlap_alpha_adjoint,
          blocked_overlap_beta_adjoint,
          left_pair_to_block_alpha_adjoint,
          left_pair_to_block_beta_adjoint,
          blocked_alpha_to_right_pair_adjoint,
          blocked_beta_to_right_pair_adjoint,
          pair_overlap_adjoint);
  return spatial_gradient;
}

void backpropagate_sample_total(
    const CandidateContext& context,
    const SampleEvaluation& sample,
    const FastSampleOperands& operands,
    double t_value,
    const ConstMatrixRef& one_electron_matrix,
    const ScalarBuffer& packed_two_electron_integrals,
    double sample_weight,
    Matrix* total_spatial_gradient,
    Matrix* one_electron_matrix_gradient,
    ScalarBuffer* packed_two_electron_gradient) {
  if (total_spatial_gradient == nullptr ||
      one_electron_matrix_gradient == nullptr ||
      packed_two_electron_gradient == nullptr) {
    throw std::invalid_argument("sample adjoint outputs must not be null");
  }

  const int n_active_orbitals = context.n_active_orbitals;
  const Matrix total_trace_matrix =
      operands.alpha_trace_matrix +
      operands.beta_trace_matrix;
  const double one_electron_factor =
      total_trace_matrix.cwiseProduct(one_electron_matrix).sum();
  const double alpha_same_separable =
      PfTensorContractor::contract_same_spin_separable(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.alpha_trace_matrix,
          operands.alpha_trace_matrix);
  const double alpha_same_bridge =
      PfTensorContractor::contract_same_spin_bridge(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.alpha_trace_matrix,
          operands.alpha_trace_matrix);
  const double beta_same_separable =
      PfTensorContractor::contract_same_spin_separable(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.beta_trace_matrix,
          operands.beta_trace_matrix);
  const double beta_same_bridge =
      PfTensorContractor::contract_same_spin_bridge(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.beta_trace_matrix,
          operands.beta_trace_matrix);
  const double opposite_direct =
      PfTensorContractor::contract_direct(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.alpha_trace_matrix,
          operands.beta_trace_matrix);
  const double opposite_exchange =
      PfTensorContractor::contract_exchange(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.opposite_exchange_left_matrix,
          operands.opposite_exchange_difference_right_matrix);
  const double opposite_mixed_exchange =
      PfTensorContractor::contract_exchange(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.opposite_exchange_left_matrix,
          operands.opposite_mixed_right_matrix);
  const double two_electron_factor =
      alpha_same_separable - alpha_same_bridge +
      beta_same_separable - beta_same_bridge +
      opposite_direct - opposite_exchange + opposite_mixed_exchange;
  const double total_factor =
      one_electron_factor + two_electron_factor;
  const double scaled_determinant =
      sample_weight * operands.determinant;

  one_electron_matrix_gradient->noalias() +=
      scaled_determinant * total_trace_matrix;

  PfTensorContractor::add_same_spin_separable_outer_product(
      scaled_determinant,
      operands.alpha_trace_matrix,
      operands.alpha_trace_matrix,
      packed_two_electron_gradient);
  PfTensorContractor::add_same_spin_bridge_outer_product(
      -scaled_determinant,
      operands.alpha_trace_matrix,
      operands.alpha_trace_matrix,
      packed_two_electron_gradient);
  PfTensorContractor::add_same_spin_separable_outer_product(
      scaled_determinant,
      operands.beta_trace_matrix,
      operands.beta_trace_matrix,
      packed_two_electron_gradient);
  PfTensorContractor::add_same_spin_bridge_outer_product(
      -scaled_determinant,
      operands.beta_trace_matrix,
      operands.beta_trace_matrix,
      packed_two_electron_gradient);
  PfTensorContractor::add_direct_outer_product(
      scaled_determinant,
      operands.alpha_trace_matrix,
      operands.beta_trace_matrix,
      packed_two_electron_gradient);
  PfTensorContractor::add_exchange_outer_product(
      -scaled_determinant,
      operands.opposite_exchange_left_matrix,
      operands.opposite_exchange_difference_right_matrix,
      packed_two_electron_gradient);
  PfTensorContractor::add_exchange_outer_product(
      scaled_determinant,
      operands.opposite_exchange_left_matrix,
      operands.opposite_mixed_right_matrix,
      packed_two_electron_gradient);

  const double determinant_adjoint =
      sample_weight * total_factor;
  Matrix alpha_trace_matrix_adjoint =
      scaled_determinant * one_electron_matrix;
  Matrix beta_trace_matrix_adjoint =
      scaled_determinant * one_electron_matrix;
  Matrix opposite_exchange_left_matrix_adjoint =
      Matrix::Zero(n_active_orbitals, n_active_orbitals);
  Matrix opposite_exchange_difference_right_matrix_adjoint =
      Matrix::Zero(n_active_orbitals, n_active_orbitals);
  Matrix opposite_mixed_right_matrix_adjoint =
      Matrix::Zero(n_active_orbitals, n_active_orbitals);

  accumulate_same_operand_adjoint(
      scaled_determinant,
      packed_two_electron_integrals,
      operands.alpha_trace_matrix,
      &PfTensorContractor::compute_same_spin_separable_operand_adjoints,
      &alpha_trace_matrix_adjoint);
  accumulate_same_operand_adjoint(
      -scaled_determinant,
      packed_two_electron_integrals,
      operands.alpha_trace_matrix,
      &PfTensorContractor::compute_same_spin_bridge_operand_adjoints,
      &alpha_trace_matrix_adjoint);
  accumulate_same_operand_adjoint(
      scaled_determinant,
      packed_two_electron_integrals,
      operands.beta_trace_matrix,
      &PfTensorContractor::compute_same_spin_separable_operand_adjoints,
      &beta_trace_matrix_adjoint);
  accumulate_same_operand_adjoint(
      -scaled_determinant,
      packed_two_electron_integrals,
      operands.beta_trace_matrix,
      &PfTensorContractor::compute_same_spin_bridge_operand_adjoints,
      &beta_trace_matrix_adjoint);
  PfTensorContractor::compute_direct_operand_adjoints(
      scaled_determinant,
      operands.alpha_trace_matrix,
      operands.beta_trace_matrix,
      packed_two_electron_integrals,
      alpha_trace_matrix_adjoint,
      beta_trace_matrix_adjoint);
  PfTensorContractor::compute_exchange_operand_adjoints(
      -scaled_determinant,
      operands.opposite_exchange_left_matrix,
      operands.opposite_exchange_difference_right_matrix,
      packed_two_electron_integrals,
      opposite_exchange_left_matrix_adjoint,
      opposite_exchange_difference_right_matrix_adjoint);
  PfTensorContractor::compute_exchange_operand_adjoints(
      scaled_determinant,
      operands.opposite_exchange_left_matrix,
      operands.opposite_mixed_right_matrix,
      packed_two_electron_integrals,
      opposite_exchange_left_matrix_adjoint,
      opposite_mixed_right_matrix_adjoint);

  total_spatial_gradient->noalias() +=
      determinant_adjoint * sample.spatial_gradient;
  total_spatial_gradient->noalias() +=
      backpropagate_fast_sample_operands_pullback_only(
          context,
          sample,
          operands,
          t_value,
          alpha_trace_matrix_adjoint,
          beta_trace_matrix_adjoint,
          opposite_exchange_left_matrix_adjoint,
          opposite_exchange_difference_right_matrix_adjoint);
}

double evaluate_sample_two_electron(
    const FastSampleOperands& operands,
    const ScalarBuffer& packed_two_electron_integrals,
    int n_active_orbitals) {
  double value = 0.0;
  value +=
      operands.determinant *
      PfTensorContractor::contract_same_spin_separable(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.alpha_trace_matrix,
          operands.alpha_trace_matrix);
  value -=
      operands.determinant *
      PfTensorContractor::contract_same_spin_bridge(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.alpha_trace_matrix,
          operands.alpha_trace_matrix);
  value +=
      operands.determinant *
      PfTensorContractor::contract_same_spin_separable(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.beta_trace_matrix,
          operands.beta_trace_matrix);
  value -=
      operands.determinant *
      PfTensorContractor::contract_same_spin_bridge(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.beta_trace_matrix,
          operands.beta_trace_matrix);
  value +=
      operands.determinant *
      PfTensorContractor::contract_direct(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.alpha_trace_matrix,
          operands.beta_trace_matrix);
  value -=
      operands.determinant *
      PfTensorContractor::contract_exchange(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.opposite_exchange_left_matrix,
          operands.opposite_exchange_difference_right_matrix);
  value +=
      operands.determinant *
      PfTensorContractor::contract_exchange(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.opposite_exchange_left_matrix,
          operands.opposite_mixed_right_matrix);
  return value;
}

}  // namespace

PfFixedMsOpenShellResult evaluate_fixed_ms_open_shell_overlap_and_one_electron(
    const ConstMatrixRef& left_pair_ba,
    const ConstMatrixRef& right_pair_ab,
    const std::vector<int>& left_blocked_alpha_orbitals,
    const std::vector<int>& left_blocked_beta_orbitals,
    const std::vector<int>& right_blocked_alpha_orbitals,
    const std::vector<int>& right_blocked_beta_orbitals,
    const ConstMatrixRef& spatial_overlap_matrix,
    const ConstMatrixRef& one_electron_matrix,
    int n_singlet_pairs) {
  validate_square_matrix(left_pair_ba, "left_pair_ba");
  validate_square_matrix(right_pair_ab, "right_pair_ab");
  validate_square_matrix(spatial_overlap_matrix, "spatial_overlap_matrix");
  validate_square_matrix(one_electron_matrix, "one_electron_matrix");
  if (n_singlet_pairs < 0) {
    throw std::invalid_argument("n_singlet_pairs must be non-negative");
  }
  if (left_pair_ba.rows() != right_pair_ab.rows() ||
      left_pair_ba.rows() != spatial_overlap_matrix.rows() ||
      spatial_overlap_matrix.rows() != one_electron_matrix.rows()) {
    throw std::invalid_argument(
        "pair blocks, spatial overlap, and one-electron matrix dimensions must match");
  }
  if (left_blocked_alpha_orbitals.size() != right_blocked_alpha_orbitals.size()) {
    throw std::invalid_argument("left/right blocked alpha counts must match");
  }
  if (left_blocked_beta_orbitals.size() != right_blocked_beta_orbitals.size()) {
    throw std::invalid_argument("left/right blocked beta counts must match");
  }

  const CandidateContext context =
      build_candidate_context(
          left_pair_ba,
          right_pair_ab,
          left_blocked_alpha_orbitals,
          left_blocked_beta_orbitals,
          right_blocked_alpha_orbitals,
          right_blocked_beta_orbitals,
          spatial_overlap_matrix,
          n_singlet_pairs);
  const Eigen::RowVectorXd weights =
      context.inverse_vandermonde.row(n_singlet_pairs);

  PfFixedMsOpenShellResult result;
  result.interpolation_degree = context.interpolation_degree;
  result.spatial_overlap_gradient =
      Matrix::Zero(context.n_active_orbitals, context.n_active_orbitals);
  for (int node = 0; node <= context.interpolation_degree; ++node) {
    const double weight = weights(node);
    const SampleEvaluation sample =
        evaluate_sample(
            context,
            context.interpolation_nodes[node]);
    result.overlap += weight * sample.determinant;
    result.spatial_overlap_gradient.noalias() +=
        weight * sample.spatial_gradient;
  }
  result.one_electron_hamiltonian =
      result.spatial_overlap_gradient.cwiseProduct(one_electron_matrix).sum();
  return result;
}

PfFixedMsOpenShellResult evaluate_fixed_ms_open_shell_pf_state_pair(
    const PfState& left_state,
    const PfState& right_state,
    const ConstMatrixRef& spatial_overlap_matrix,
    const ConstMatrixRef& one_electron_matrix) {
  if (left_state.n_active_orbitals != right_state.n_active_orbitals ||
      left_state.n_spin_orbitals != right_state.n_spin_orbitals) {
    throw std::invalid_argument("left/right PfState dimensions must match");
  }
  if (left_state.n_singlet_pairs != right_state.n_singlet_pairs) {
    throw std::invalid_argument("left/right PfState singlet-pair counts must match");
  }

  const Matrix left_pairing_matrix = decode_pf_state(left_state);
  const Matrix right_pairing_matrix = decode_pf_state(right_state);
  const int n_active_orbitals = left_state.n_active_orbitals;
  const Matrix left_pair_ba =
      left_pairing_matrix.transpose().bottomLeftCorner(
          n_active_orbitals,
          n_active_orbitals);
  const Matrix right_pair_ab =
      right_pairing_matrix.topRightCorner(
          n_active_orbitals,
          n_active_orbitals);

  return evaluate_fixed_ms_open_shell_overlap_and_one_electron(
      left_pair_ba,
      right_pair_ab,
      left_state.blocked_alpha_orbitals,
      left_state.blocked_beta_orbitals,
      right_state.blocked_alpha_orbitals,
      right_state.blocked_beta_orbitals,
      spatial_overlap_matrix,
      one_electron_matrix,
      left_state.n_singlet_pairs);
}

PfFixedMsOpenShellHamiltonianResult evaluate_fixed_ms_open_shell_pair_hamiltonian(
    const ConstMatrixRef& left_pair_ba,
    const ConstMatrixRef& right_pair_ab,
    const std::vector<int>& left_blocked_alpha_orbitals,
    const std::vector<int>& left_blocked_beta_orbitals,
    const std::vector<int>& right_blocked_alpha_orbitals,
    const std::vector<int>& right_blocked_beta_orbitals,
    const ConstMatrixRef& spatial_overlap_matrix,
    const ConstMatrixRef& one_electron_matrix,
    const ScalarBuffer& packed_two_electron_integrals,
    int n_singlet_pairs,
    bool include_gradients) {
  validate_square_matrix(left_pair_ba, "left_pair_ba");
  validate_square_matrix(right_pair_ab, "right_pair_ab");
  validate_square_matrix(spatial_overlap_matrix, "spatial_overlap_matrix");
  validate_square_matrix(one_electron_matrix, "one_electron_matrix");
  if (n_singlet_pairs < 0) {
    throw std::invalid_argument("n_singlet_pairs must be non-negative");
  }
  if (left_pair_ba.rows() != right_pair_ab.rows() ||
      left_pair_ba.rows() != spatial_overlap_matrix.rows() ||
      spatial_overlap_matrix.rows() != one_electron_matrix.rows()) {
    throw std::invalid_argument(
        "pair blocks, spatial overlap, and one-electron matrix dimensions must match");
  }
  if (left_blocked_alpha_orbitals.size() != right_blocked_alpha_orbitals.size()) {
    throw std::invalid_argument("left/right blocked alpha counts must match");
  }
  if (left_blocked_beta_orbitals.size() != right_blocked_beta_orbitals.size()) {
    throw std::invalid_argument("left/right blocked beta counts must match");
  }

  const CandidateContext context =
      build_candidate_context(
          left_pair_ba,
          right_pair_ab,
          left_blocked_alpha_orbitals,
          left_blocked_beta_orbitals,
          right_blocked_alpha_orbitals,
          right_blocked_beta_orbitals,
          spatial_overlap_matrix,
          n_singlet_pairs);
  const Eigen::RowVectorXd weights =
      context.inverse_vandermonde.row(n_singlet_pairs);

  PfFixedMsOpenShellHamiltonianResult result;
  result.interpolation_degree = context.interpolation_degree;
  if (include_gradients) {
    result.overlap_spatial_gradient =
        Matrix::Zero(context.n_active_orbitals, context.n_active_orbitals);
    result.total_hamiltonian_spatial_gradient =
        Matrix::Zero(context.n_active_orbitals, context.n_active_orbitals);
    result.one_electron_matrix_gradient =
        Matrix::Zero(context.n_active_orbitals, context.n_active_orbitals);
    result.packed_two_electron_gradient.assign(
        PfTensorContractor::packed_two_electron_count(context.n_active_orbitals),
        0.0);
    result.gradients_computed = true;
  }
  for (int node = 0; node <= context.interpolation_degree; ++node) {
    const double weight = weights(node);
    const double t_value =
        context.interpolation_nodes[node];
    const SampleEvaluation sample =
        evaluate_sample(
            context,
            t_value);
    const FastSampleOperands operands =
        build_fast_sample_operands(
            context,
            sample,
            t_value);

    const Matrix total_trace_matrix =
        operands.alpha_trace_matrix +
        operands.beta_trace_matrix;
    const double sample_one_electron =
        operands.determinant *
        total_trace_matrix.cwiseProduct(one_electron_matrix).sum();
    const double sample_two_electron =
        evaluate_sample_two_electron(
            operands,
            packed_two_electron_integrals,
            context.n_active_orbitals);

    result.overlap += weight * operands.determinant;
    result.one_electron_hamiltonian += weight * sample_one_electron;
    result.two_electron_hamiltonian += weight * sample_two_electron;
    if (include_gradients) {
      result.overlap_spatial_gradient.noalias() +=
          weight * sample.spatial_gradient;
      backpropagate_sample_total(
          context,
          sample,
          operands,
          t_value,
          one_electron_matrix,
          packed_two_electron_integrals,
          weight,
          &result.total_hamiltonian_spatial_gradient,
          &result.one_electron_matrix_gradient,
          &result.packed_two_electron_gradient);
    }
  }
  result.total_hamiltonian =
      result.one_electron_hamiltonian +
      result.two_electron_hamiltonian;
  return result;
}

PfFixedMsOpenShellHamiltonianResult evaluate_fixed_ms_open_shell_pf_state_pair_hamiltonian(
    const PfState& left_state,
    const PfState& right_state,
    const ConstMatrixRef& spatial_overlap_matrix,
    const ConstMatrixRef& one_electron_matrix,
    const ScalarBuffer& packed_two_electron_integrals,
    bool include_gradients) {
  if (left_state.n_active_orbitals != right_state.n_active_orbitals ||
      left_state.n_spin_orbitals != right_state.n_spin_orbitals) {
    throw std::invalid_argument("left/right PfState dimensions must match");
  }
  if (left_state.n_singlet_pairs != right_state.n_singlet_pairs) {
    throw std::invalid_argument("left/right PfState singlet-pair counts must match");
  }

  const Matrix left_pairing_matrix = decode_pf_state(left_state);
  const Matrix right_pairing_matrix = decode_pf_state(right_state);
  const int n_active_orbitals = left_state.n_active_orbitals;
  const Matrix left_pair_ba =
      left_pairing_matrix.transpose().bottomLeftCorner(
          n_active_orbitals,
          n_active_orbitals);
  const Matrix right_pair_ab =
      right_pairing_matrix.topRightCorner(
          n_active_orbitals,
          n_active_orbitals);

  return evaluate_fixed_ms_open_shell_pair_hamiltonian(
      left_pair_ba,
      right_pair_ab,
      left_state.blocked_alpha_orbitals,
      left_state.blocked_beta_orbitals,
      right_state.blocked_alpha_orbitals,
      right_state.blocked_beta_orbitals,
      spatial_overlap_matrix,
      one_electron_matrix,
      packed_two_electron_integrals,
      left_state.n_singlet_pairs,
      include_gradients);
}

}  // namespace xmvb::pfaffian_vbscf
