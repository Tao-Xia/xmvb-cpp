#include "pfaffian_vbscf/kernel/pf_high_spin_open_shell_kernel.hpp"

#include <algorithm>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <Eigen/LU>
#include <Eigen/QR>

#include "pfaffian_vbscf/math/dense_utils.hpp"

namespace xmvb::pfaffian_vbscf {

namespace {

constexpr double kInterpolationNodeScale = 0.008;

std::vector<int> build_complement_indices(
    int n_active_orbitals,
    const std::vector<int>& blocked_orbitals,
    const char* label) {
  std::vector<bool> blocked_mask(
      n_active_orbitals,
      false);
  for (const int orbital : blocked_orbitals) {
    if (orbital < 0 || orbital >= n_active_orbitals) {
      throw std::invalid_argument(
          std::string(label) + " index is outside the active-space range");
    }
    if (blocked_mask[orbital]) {
      throw std::invalid_argument(
          std::string(label) + " contains a duplicated blocked orbital");
    }
    blocked_mask[orbital] = true;
  }

  std::vector<int> complement_indices;
  complement_indices.reserve(
      
          std::max(0, n_active_orbitals - static_cast<int>(blocked_orbitals.size())));
  for (int orbital = 0; orbital < n_active_orbitals; ++orbital) {
    if (!blocked_mask[orbital]) {
      complement_indices.push_back(orbital);
    }
  }
  return complement_indices;
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

struct OpenShellSampleBlocks {
  Matrix blocked_overlap;
  Matrix blocked_to_pair;
  Matrix sample_bottom_left;
  Matrix sample_bottom_right;
  Matrix pair_to_block;
  Matrix pair_kernel_seed;
  Matrix left_pair_overlap_factor;
  Matrix right_pair_transpose;
};

struct SampleEvaluation {
  double value = 0.0;
  Matrix spatial_gradient;
};

SampleEvaluation evaluate_sample(
    double t_value,
    const OpenShellSampleBlocks& blocks,
    int n_active_orbitals,
    const std::vector<int>& left_pair_orbitals,
    const std::vector<int>& right_pair_orbitals,
    const std::vector<int>& left_blocked_alpha_orbitals,
    const std::vector<int>& right_blocked_alpha_orbitals) {
  const int n_pair_orbitals = static_cast<int>(left_pair_orbitals.size());
  const int n_blocked_alpha = static_cast<int>(left_blocked_alpha_orbitals.size());
  if (blocks.blocked_overlap.rows() != n_blocked_alpha ||
      blocks.blocked_overlap.cols() != n_blocked_alpha ||
      blocks.blocked_to_pair.rows() != n_blocked_alpha ||
      blocks.blocked_to_pair.cols() != n_pair_orbitals ||
      blocks.sample_bottom_left.rows() != n_pair_orbitals ||
      blocks.sample_bottom_left.cols() != n_blocked_alpha ||
      blocks.sample_bottom_right.rows() != n_pair_orbitals ||
      blocks.sample_bottom_right.cols() != n_pair_orbitals ||
      blocks.pair_to_block.rows() != n_pair_orbitals ||
      blocks.pair_to_block.cols() != n_blocked_alpha ||
      blocks.pair_kernel_seed.rows() != n_pair_orbitals ||
      blocks.pair_kernel_seed.cols() != n_pair_orbitals ||
      blocks.left_pair_overlap_factor.rows() != n_pair_orbitals ||
      blocks.left_pair_overlap_factor.cols() != n_pair_orbitals ||
      blocks.right_pair_transpose.rows() != n_pair_orbitals ||
      blocks.right_pair_transpose.cols() != n_pair_orbitals) {
    throw std::invalid_argument("reduced open-shell sample blocks are inconsistent");
  }

  Matrix sample_matrix = Matrix::Zero(
      n_blocked_alpha + n_pair_orbitals,
      n_blocked_alpha + n_pair_orbitals);
  if (n_blocked_alpha > 0) {
    sample_matrix.topLeftCorner(
        n_blocked_alpha,
        n_blocked_alpha) = blocks.blocked_overlap;
    sample_matrix.topRightCorner(
        n_blocked_alpha,
        n_pair_orbitals) = t_value * blocks.blocked_to_pair;
    sample_matrix.bottomLeftCorner(
        n_pair_orbitals,
        n_blocked_alpha) = blocks.sample_bottom_left;
  }
  sample_matrix.bottomRightCorner(
      n_pair_orbitals,
      n_pair_orbitals) =
      Matrix::Identity(n_pair_orbitals, n_pair_orbitals) +
      t_value * blocks.sample_bottom_right;

  Eigen::FullPivLU<Matrix> lu(sample_matrix);
  const double determinant_value = lu.determinant();
  if (!std::isfinite(determinant_value)) {
    throw std::runtime_error("sample determinant is not finite");
  }

  Matrix sample_matrix_inverse_transpose =
      lu.inverse().transpose();
  Matrix sample_matrix_gradient =
      determinant_value * sample_matrix_inverse_transpose;

  SampleEvaluation result;
  result.value = determinant_value;
  result.spatial_gradient = Matrix::Zero(n_active_orbitals, n_active_orbitals);

  Matrix pair_overlap_gradient =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix left_pair_to_block_gradient =
      Matrix::Zero(n_pair_orbitals, n_blocked_alpha);
  Matrix blocked_to_right_pair_gradient =
      Matrix::Zero(n_blocked_alpha, n_pair_orbitals);

  if (n_blocked_alpha > 0) {
    const Matrix gradient_oo =
        sample_matrix_gradient.topLeftCorner(
            n_blocked_alpha,
            n_blocked_alpha);
    const Matrix gradient_x =
        sample_matrix_gradient.topRightCorner(
            n_blocked_alpha,
            n_pair_orbitals);
    const Matrix gradient_y =
        sample_matrix_gradient.bottomLeftCorner(
            n_pair_orbitals,
            n_blocked_alpha);

    blocked_to_right_pair_gradient.noalias() +=
        t_value *
        gradient_x *
        blocks.right_pair_transpose;
    pair_overlap_gradient.noalias() +=
        blocks.pair_to_block *
        gradient_y.transpose();
    left_pair_to_block_gradient.noalias() +=
        blocks.left_pair_overlap_factor *
        gradient_y;

    accumulate_submatrix(
        &result.spatial_gradient,
        left_blocked_alpha_orbitals,
        right_blocked_alpha_orbitals,
        gradient_oo);
  }

  const Matrix gradient_g =
      sample_matrix_gradient.bottomRightCorner(
          n_pair_orbitals,
          n_pair_orbitals);
  pair_overlap_gradient.noalias() +=
      t_value *
      blocks.pair_kernel_seed *
      gradient_g.transpose();
  pair_overlap_gradient.noalias() +=
      t_value *
      blocks.left_pair_overlap_factor *
      gradient_g *
      blocks.right_pair_transpose;

  accumulate_submatrix(
      &result.spatial_gradient,
      left_blocked_alpha_orbitals,
      right_pair_orbitals,
      blocked_to_right_pair_gradient);
  accumulate_submatrix(
      &result.spatial_gradient,
      left_pair_orbitals,
      right_blocked_alpha_orbitals,
      left_pair_to_block_gradient);
  accumulate_submatrix(
      &result.spatial_gradient,
      left_pair_orbitals,
      right_pair_orbitals,
      pair_overlap_gradient);

  return result;
}

void validate_square_matrix(
    const ConstMatrixRef& matrix,
    const char* label) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument(std::string(label) + " must be square");
  }
}

}  // namespace

PfHighSpinOpenShellResult evaluate_high_spin_open_shell_overlap_and_one_electron(
    const ConstMatrixRef& left_pair_ba,
    const ConstMatrixRef& right_pair_ab,
    const std::vector<int>& left_blocked_alpha_orbitals,
    const std::vector<int>& right_blocked_alpha_orbitals,
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

  const int n_active_orbitals = static_cast<int>(spatial_overlap_matrix.rows());
  const std::vector<int> left_pair_orbitals =
      build_complement_indices(
          n_active_orbitals,
          left_blocked_alpha_orbitals,
          "left_blocked_alpha_orbitals");
  const std::vector<int> right_pair_orbitals =
      build_complement_indices(
          n_active_orbitals,
          right_blocked_alpha_orbitals,
          "right_blocked_alpha_orbitals");
  const int interpolation_degree =
      static_cast<int>(left_pair_orbitals.size());
  if (n_singlet_pairs > interpolation_degree) {
    throw std::invalid_argument("n_singlet_pairs exceeds the interpolation degree");
  }
  if (static_cast<int>(left_pair_orbitals.size()) < n_singlet_pairs ||
      static_cast<int>(right_pair_orbitals.size()) < n_singlet_pairs) {
    throw std::invalid_argument(
        "blocked open-shell orbitals leave too few pair orbitals for n_singlet_pairs");
  }

  const Matrix left_pair_ba_reduced =
      extract_submatrix(
          left_pair_ba,
          left_pair_orbitals,
          left_pair_orbitals);
  const Matrix right_pair_ab_reduced =
      extract_submatrix(
          right_pair_ab,
          right_pair_orbitals,
          right_pair_orbitals);
  const Matrix pair_overlap =
      extract_submatrix(
          spatial_overlap_matrix,
          left_pair_orbitals,
          right_pair_orbitals);
  const Matrix left_pair_to_block_overlap =
      extract_submatrix(
          spatial_overlap_matrix,
          left_pair_orbitals,
          right_blocked_alpha_orbitals);
  const Matrix blocked_to_right_pair_overlap =
      extract_submatrix(
          spatial_overlap_matrix,
          left_blocked_alpha_orbitals,
          right_pair_orbitals);
  const Matrix blocked_overlap =
      extract_submatrix(
          spatial_overlap_matrix,
          left_blocked_alpha_orbitals,
          right_blocked_alpha_orbitals);
  OpenShellSampleBlocks blocks;
  blocks.blocked_overlap = blocked_overlap;
  blocks.blocked_to_pair =
      blocked_to_right_pair_overlap *
      right_pair_ab_reduced;
  blocks.pair_to_block =
      left_pair_ba_reduced *
      left_pair_to_block_overlap;
  blocks.pair_kernel_seed =
      left_pair_ba_reduced *
      pair_overlap *
      right_pair_ab_reduced;
  blocks.sample_bottom_left =
      pair_overlap.transpose() *
      blocks.pair_to_block;
  blocks.sample_bottom_right =
      pair_overlap.transpose() *
      blocks.pair_kernel_seed;
  blocks.left_pair_overlap_factor =
      left_pair_ba_reduced.transpose() *
      pair_overlap;
  blocks.right_pair_transpose =
      right_pair_ab_reduced.transpose();

  const InterpolationCache& interpolation_cache =
      get_interpolation_cache(interpolation_degree);
  const Eigen::RowVectorXd weights =
      interpolation_cache.inverse_vandermonde.row(n_singlet_pairs);

  PfHighSpinOpenShellResult result;
  result.interpolation_degree = interpolation_degree;
  result.spatial_overlap_gradient =
      Matrix::Zero(n_active_orbitals, n_active_orbitals);
  for (int node = 0; node <= interpolation_degree; ++node) {
    const double weight = weights(node);
    const SampleEvaluation sample =
        evaluate_sample(
            interpolation_cache.nodes[node],
            blocks,
            n_active_orbitals,
            left_pair_orbitals,
            right_pair_orbitals,
            left_blocked_alpha_orbitals,
            right_blocked_alpha_orbitals);
    result.overlap += weight * sample.value;
    result.spatial_overlap_gradient.noalias() +=
        weight * sample.spatial_gradient;
  }
  result.one_electron_hamiltonian =
      result.spatial_overlap_gradient.cwiseProduct(one_electron_matrix).sum();
  return result;
}

PfHighSpinOpenShellResult evaluate_high_spin_open_shell_pf_state_pair(
    const PfState& left_state,
    const PfState& right_state,
    const ConstMatrixRef& spatial_overlap_matrix,
    const ConstMatrixRef& one_electron_matrix) {
  if (left_state.n_active_orbitals != right_state.n_active_orbitals ||
      left_state.n_spin_orbitals != right_state.n_spin_orbitals) {
    throw std::invalid_argument("left/right PfState dimensions must match");
  }
  if (!right_state.blocked_beta_orbitals.empty() ||
      !left_state.blocked_beta_orbitals.empty()) {
    throw std::invalid_argument(
        "high-spin open-shell evaluator currently supports blocked alpha orbitals only");
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

  return evaluate_high_spin_open_shell_overlap_and_one_electron(
      left_pair_ba,
      right_pair_ab,
      left_state.blocked_alpha_orbitals,
      right_state.blocked_alpha_orbitals,
      spatial_overlap_matrix,
      one_electron_matrix,
      left_state.n_singlet_pairs);
}

}  // namespace xmvb::pfaffian_vbscf
