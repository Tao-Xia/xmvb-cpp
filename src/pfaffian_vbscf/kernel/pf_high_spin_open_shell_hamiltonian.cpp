#include "pfaffian_vbscf/kernel/pf_high_spin_open_shell_hamiltonian.hpp"

#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <Eigen/LU>
#include <Eigen/QR>

#include "pfaffian_vbscf/math/dense_utils.hpp"
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

std::vector<int> build_complement_indices(
    int n_active_orbitals,
    const std::vector<int>& blocked_orbitals,
    const char* label) {
  std::vector<bool> blocked_mask(
      xmvb::to_size(n_active_orbitals),
      false);
  for (const int orbital : blocked_orbitals) {
    if (orbital < 0 || orbital >= n_active_orbitals) {
      throw std::invalid_argument(
          std::string(label) + " index is outside the active-space range");
    }
    if (blocked_mask[xmvb::to_size(orbital)]) {
      throw std::invalid_argument(
          std::string(label) + " contains a duplicated blocked orbital");
    }
    blocked_mask[xmvb::to_size(orbital)] = true;
  }

  std::vector<int> complement_indices;
  complement_indices.reserve(
      xmvb::to_size(
          std::max(0, n_active_orbitals - static_cast<int>(blocked_orbitals.size()))));
  for (int orbital = 0; orbital < n_active_orbitals; ++orbital) {
    if (!blocked_mask[xmvb::to_size(orbital)]) {
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
              row_indices[xmvb::to_size(row)],
              col_indices[xmvb::to_size(col)]);
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
          row_indices[xmvb::to_size(row)],
          col_indices[xmvb::to_size(col)]) += source(row, col);
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
          row_indices[xmvb::to_size(row)],
          col_indices[xmvb::to_size(col)]) = submatrix(row, col);
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
  for (int node = 0; node <= degree; ++node) {
    const double node_value =
        kInterpolationNodeScale * static_cast<double>(node);
    double power_value = 1.0;
    for (int power = 0; power <= degree; ++power) {
      vandermonde(node, power) = power_value;
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
        xmvb::to_size(degree + 1),
        0.0);
    for (int node = 0; node <= degree; ++node) {
      interpolation_cache.nodes[xmvb::to_size(node)] =
          kInterpolationNodeScale * static_cast<double>(node);
    }
    interpolation_cache.inverse_vandermonde =
        build_vandermonde_inverse(degree);
  }
  return iterator->second;
}

struct SampleContext {
  double determinant = 0.0;
  Matrix sample_matrix_inverse;
};

struct CandidateContext {
  int n_active_orbitals = 0;
  int n_singlet_pairs = 0;
  int interpolation_degree = 0;
  std::vector<int> left_blocked_alpha_orbitals;
  std::vector<int> right_blocked_alpha_orbitals;
  std::vector<int> left_pair_orbitals;
  std::vector<int> right_pair_orbitals;
  std::vector<int> left_ordered_orbitals;
  std::vector<int> right_ordered_orbitals;
  Matrix left_pair_ba_reduced;
  Matrix right_pair_ab_reduced;
  Matrix blocked_overlap_alpha;
  Matrix blocked_to_right_pair_overlap_alpha;
  Matrix left_pair_to_block_overlap_alpha;
  Matrix pair_overlap_alpha;
  Matrix pair_overlap_beta;
  Matrix blocked_to_pair_matrix;
  Matrix beta_left_factor;
  Matrix alpha_to_block_factor;
  Matrix pair_kernel_seed;
  Matrix sample_bottom_left;
  Matrix sample_bottom_right;
  std::vector<double> interpolation_nodes;
  Eigen::MatrixXd inverse_vandermonde;
};

struct FastSampleOperands {
  double determinant = 0.0;
  Matrix sample_inverse;
  Matrix beta_trace_reduced;
  Matrix alpha_trace_matrix;
  Matrix beta_trace_matrix;
  Matrix opposite_exchange_left_matrix;
  Matrix opposite_exchange_right_matrix;
  Matrix opposite_mixed_right_matrix;
};

CandidateContext build_candidate_context(
    const ConstMatrixRef& left_pair_ba,
    const ConstMatrixRef& right_pair_ab,
    const std::vector<int>& left_blocked_alpha_orbitals,
    const std::vector<int>& right_blocked_alpha_orbitals,
    const ConstMatrixRef& spatial_overlap,
    int n_singlet_pairs) {
  const int n_active_orbitals = static_cast<int>(spatial_overlap.rows());
  CandidateContext context;
  context.n_active_orbitals = n_active_orbitals;
  context.n_singlet_pairs = n_singlet_pairs;
  context.left_blocked_alpha_orbitals = left_blocked_alpha_orbitals;
  context.right_blocked_alpha_orbitals = right_blocked_alpha_orbitals;
  context.left_pair_orbitals =
      build_complement_indices(
          n_active_orbitals,
          left_blocked_alpha_orbitals,
          "left_blocked_alpha_orbitals");
  context.right_pair_orbitals =
      build_complement_indices(
          n_active_orbitals,
          right_blocked_alpha_orbitals,
          "right_blocked_alpha_orbitals");
  context.left_ordered_orbitals = context.left_blocked_alpha_orbitals;
  context.left_ordered_orbitals.insert(
      context.left_ordered_orbitals.end(),
      context.left_pair_orbitals.begin(),
      context.left_pair_orbitals.end());
  context.right_ordered_orbitals = context.right_blocked_alpha_orbitals;
  context.right_ordered_orbitals.insert(
      context.right_ordered_orbitals.end(),
      context.right_pair_orbitals.begin(),
      context.right_pair_orbitals.end());

  context.interpolation_degree =
      static_cast<int>(context.left_pair_orbitals.size());
  if (n_singlet_pairs > context.interpolation_degree ||
      static_cast<int>(context.right_pair_orbitals.size()) < n_singlet_pairs) {
    throw std::invalid_argument("invalid singlet-pair count for the blocked open-shell layout");
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
  context.blocked_to_right_pair_overlap_alpha =
      extract_submatrix(
          spatial_overlap,
          context.left_blocked_alpha_orbitals,
          context.right_pair_orbitals);
  context.left_pair_to_block_overlap_alpha =
      extract_submatrix(
          spatial_overlap,
          context.left_pair_orbitals,
          context.right_blocked_alpha_orbitals);
  context.pair_overlap_alpha =
      extract_submatrix(
          spatial_overlap,
          context.left_pair_orbitals,
          context.right_pair_orbitals);
  context.pair_overlap_beta = context.pair_overlap_alpha;

  context.blocked_to_pair_matrix =
      context.blocked_to_right_pair_overlap_alpha *
      context.right_pair_ab_reduced;
  context.beta_left_factor =
      context.pair_overlap_beta.transpose() *
      context.left_pair_ba_reduced;
  context.alpha_to_block_factor =
      context.left_pair_ba_reduced *
      context.left_pair_to_block_overlap_alpha;
  context.pair_kernel_seed =
      context.left_pair_ba_reduced *
      context.pair_overlap_alpha *
      context.right_pair_ab_reduced;
  context.sample_bottom_left =
      context.beta_left_factor *
      context.left_pair_to_block_overlap_alpha;
  context.sample_bottom_right =
      context.beta_left_factor *
      context.pair_overlap_alpha *
      context.right_pair_ab_reduced;

  const InterpolationCache& interpolation_cache =
      get_interpolation_cache(context.interpolation_degree);
  context.interpolation_nodes = interpolation_cache.nodes;
  context.inverse_vandermonde = interpolation_cache.inverse_vandermonde;
  return context;
}

SampleContext evaluate_sample(
    const CandidateContext& context,
    double t_value) {
  const int n_blocked_alpha =
      static_cast<int>(context.left_blocked_alpha_orbitals.size());
  const int n_pair_orbitals =
      static_cast<int>(context.left_pair_orbitals.size());

  Matrix sample_matrix = Matrix::Zero(
      n_blocked_alpha + n_pair_orbitals,
      n_blocked_alpha + n_pair_orbitals);
  if (n_blocked_alpha > 0) {
    sample_matrix.topLeftCorner(
        n_blocked_alpha,
        n_blocked_alpha) = context.blocked_overlap_alpha;
    sample_matrix.topRightCorner(
        n_blocked_alpha,
        n_pair_orbitals) = t_value * context.blocked_to_pair_matrix;
    sample_matrix.bottomLeftCorner(
        n_pair_orbitals,
        n_blocked_alpha) = context.sample_bottom_left;
  }
  sample_matrix.bottomRightCorner(
      n_pair_orbitals,
      n_pair_orbitals) =
      Matrix::Identity(n_pair_orbitals, n_pair_orbitals) +
      t_value * context.sample_bottom_right;

  const Eigen::FullPivLU<Matrix> lu(sample_matrix);
  SampleContext sample;
  sample.determinant = lu.determinant();
  if (!std::isfinite(sample.determinant)) {
    throw std::runtime_error("sample determinant is not finite");
  }
  sample.sample_matrix_inverse = lu.inverse();
  return sample;
}

FastSampleOperands build_fast_sample_operands(
    const CandidateContext& context,
    double t_value) {
  const SampleContext sample =
      evaluate_sample(context, t_value);
  const int n_active_orbitals = context.n_active_orbitals;
  const int n_blocked_alpha =
      static_cast<int>(context.left_blocked_alpha_orbitals.size());
  const int n_pair_orbitals =
      static_cast<int>(context.left_pair_orbitals.size());
  const Matrix& sample_inverse = sample.sample_matrix_inverse;
  const Matrix sample_inverse_oo =
      sample_inverse.topLeftCorner(
          n_blocked_alpha,
          n_blocked_alpha);
  const Matrix sample_inverse_op =
      sample_inverse.topRightCorner(
          n_blocked_alpha,
          n_pair_orbitals);
  const Matrix sample_inverse_po =
      sample_inverse.bottomLeftCorner(
          n_pair_orbitals,
          n_blocked_alpha);
  const Matrix sample_inverse_pp =
      sample_inverse.bottomRightCorner(
          n_pair_orbitals,
          n_pair_orbitals);

  Matrix alpha_trace_ordered =
      Matrix::Zero(
          n_blocked_alpha + n_pair_orbitals,
          n_blocked_alpha + n_pair_orbitals);
  if (n_blocked_alpha > 0) {
    alpha_trace_ordered.topLeftCorner(
        n_blocked_alpha,
        n_blocked_alpha) =
        sample_inverse_oo.transpose();
    alpha_trace_ordered.topRightCorner(
        n_blocked_alpha,
        n_pair_orbitals) =
        t_value *
        sample_inverse_po.transpose() *
        context.right_pair_ab_reduced.transpose();
    alpha_trace_ordered.bottomLeftCorner(
        n_pair_orbitals,
        n_blocked_alpha) =
        context.beta_left_factor.transpose() *
        sample_inverse_op.transpose();
  }
  alpha_trace_ordered.bottomRightCorner(
      n_pair_orbitals,
      n_pair_orbitals) =
      t_value *
      context.beta_left_factor.transpose() *
      sample_inverse_pp.transpose() *
      context.right_pair_ab_reduced.transpose();

  Matrix beta_trace_reduced =
      context.alpha_to_block_factor * sample_inverse_op;
  beta_trace_reduced.noalias() +=
      t_value *
      context.pair_kernel_seed *
      sample_inverse_pp;

  Matrix opposite_exchange_left_ordered =
      Matrix::Zero(
          n_blocked_alpha + n_pair_orbitals,
          n_pair_orbitals);
  if (n_blocked_alpha > 0) {
    opposite_exchange_left_ordered.topRows(n_blocked_alpha) =
        sample_inverse_op;
  }
  opposite_exchange_left_ordered.bottomRows(n_pair_orbitals) =
      t_value *
      context.right_pair_ab_reduced *
      sample_inverse_pp;

  Matrix opposite_exchange_right_ordered =
      Matrix::Zero(
          n_pair_orbitals,
          n_blocked_alpha + n_pair_orbitals);
  if (n_blocked_alpha > 0) {
    opposite_exchange_right_ordered.leftCols(n_blocked_alpha).noalias() =
        context.alpha_to_block_factor * sample_inverse_oo +
        t_value * context.pair_kernel_seed * sample_inverse_po;
  }
  opposite_exchange_right_ordered.rightCols(n_pair_orbitals).noalias() =
      beta_trace_reduced * context.beta_left_factor;

  Matrix opposite_mixed_right_ordered =
      Matrix::Zero(
          n_pair_orbitals,
          n_blocked_alpha + n_pair_orbitals);
  opposite_mixed_right_ordered.rightCols(n_pair_orbitals) =
      context.left_pair_ba_reduced;

  FastSampleOperands operands;
  operands.determinant = sample.determinant;
  operands.sample_inverse = sample.sample_matrix_inverse;
  operands.beta_trace_reduced = beta_trace_reduced;
  operands.alpha_trace_matrix =
      scatter_submatrix(
          n_active_orbitals,
          context.left_ordered_orbitals,
          context.right_ordered_orbitals,
          alpha_trace_ordered);
  operands.beta_trace_matrix =
      scatter_submatrix(
          n_active_orbitals,
          context.left_pair_orbitals,
          context.right_pair_orbitals,
          beta_trace_reduced);
  operands.opposite_exchange_left_matrix =
      scatter_submatrix(
          n_active_orbitals,
          context.right_ordered_orbitals,
          context.right_pair_orbitals,
          opposite_exchange_left_ordered);
  operands.opposite_exchange_right_matrix =
      scatter_submatrix(
          n_active_orbitals,
          context.left_pair_orbitals,
          context.left_ordered_orbitals,
          opposite_exchange_right_ordered);
  operands.opposite_mixed_right_matrix =
      scatter_submatrix(
          n_active_orbitals,
          context.left_pair_orbitals,
          context.left_ordered_orbitals,
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

Matrix backpropagate_determinant_pullback_only(
    const CandidateContext& context,
    const FastSampleOperands& operands,
    double t_value,
    double determinant_adjoint) {
  const int n_active_orbitals = context.n_active_orbitals;
  const int n_blocked_alpha =
      static_cast<int>(context.left_blocked_alpha_orbitals.size());
  const int n_pair_orbitals =
      static_cast<int>(context.left_pair_orbitals.size());

  Matrix blocked_overlap_alpha_adjoint =
      Matrix::Zero(n_blocked_alpha, n_blocked_alpha);
  Matrix blocked_to_right_pair_overlap_alpha_adjoint =
      Matrix::Zero(n_blocked_alpha, n_pair_orbitals);
  Matrix left_pair_to_block_overlap_alpha_adjoint =
      Matrix::Zero(n_pair_orbitals, n_blocked_alpha);
  Matrix pair_overlap_alpha_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix pair_overlap_beta_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix left_pair_ba_reduced_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix right_pair_ab_reduced_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix blocked_to_pair_matrix_adjoint =
      Matrix::Zero(n_blocked_alpha, n_pair_orbitals);
  Matrix beta_left_factor_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix sample_bottom_left_adjoint =
      Matrix::Zero(n_pair_orbitals, n_blocked_alpha);
  Matrix sample_bottom_right_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);

  Matrix sample_matrix_adjoint =
      determinant_adjoint *
      operands.determinant *
      operands.sample_inverse.transpose();

  if (n_blocked_alpha > 0) {
    blocked_overlap_alpha_adjoint.noalias() +=
        sample_matrix_adjoint.topLeftCorner(
            n_blocked_alpha,
            n_blocked_alpha);
    blocked_to_pair_matrix_adjoint.noalias() +=
        t_value *
        sample_matrix_adjoint.topRightCorner(
            n_blocked_alpha,
            n_pair_orbitals);
    sample_bottom_left_adjoint.noalias() +=
        sample_matrix_adjoint.bottomLeftCorner(
            n_pair_orbitals,
            n_blocked_alpha);
  }
  sample_bottom_right_adjoint.noalias() +=
      t_value *
      sample_matrix_adjoint.bottomRightCorner(
          n_pair_orbitals,
          n_pair_orbitals);

  beta_left_factor_adjoint.noalias() +=
      sample_bottom_left_adjoint *
      context.left_pair_to_block_overlap_alpha.transpose();
  left_pair_to_block_overlap_alpha_adjoint.noalias() +=
      context.beta_left_factor.transpose() *
      sample_bottom_left_adjoint;

  beta_left_factor_adjoint.noalias() +=
      sample_bottom_right_adjoint *
      (context.pair_overlap_alpha * context.right_pair_ab_reduced).transpose();
  pair_overlap_alpha_adjoint.noalias() +=
      context.beta_left_factor.transpose() *
      sample_bottom_right_adjoint *
      context.right_pair_ab_reduced.transpose();
  right_pair_ab_reduced_adjoint.noalias() +=
      (context.beta_left_factor * context.pair_overlap_alpha).transpose() *
      sample_bottom_right_adjoint;

  blocked_to_right_pair_overlap_alpha_adjoint.noalias() +=
      blocked_to_pair_matrix_adjoint *
      context.right_pair_ab_reduced.transpose();
  right_pair_ab_reduced_adjoint.noalias() +=
      context.blocked_to_right_pair_overlap_alpha.transpose() *
      blocked_to_pair_matrix_adjoint;

  pair_overlap_beta_adjoint.noalias() +=
      context.left_pair_ba_reduced *
      beta_left_factor_adjoint.transpose();
  left_pair_ba_reduced_adjoint.noalias() +=
      context.pair_overlap_beta *
      beta_left_factor_adjoint;

  Matrix total_spatial_gradient =
      Matrix::Zero(n_active_orbitals, n_active_orbitals);
  accumulate_submatrix(
      &total_spatial_gradient,
      context.left_blocked_alpha_orbitals,
      context.right_blocked_alpha_orbitals,
      blocked_overlap_alpha_adjoint);
  accumulate_submatrix(
      &total_spatial_gradient,
      context.left_blocked_alpha_orbitals,
      context.right_pair_orbitals,
      blocked_to_right_pair_overlap_alpha_adjoint);
  accumulate_submatrix(
      &total_spatial_gradient,
      context.left_pair_orbitals,
      context.right_blocked_alpha_orbitals,
      left_pair_to_block_overlap_alpha_adjoint);
  accumulate_submatrix(
      &total_spatial_gradient,
      context.left_pair_orbitals,
      context.right_pair_orbitals,
      pair_overlap_alpha_adjoint + pair_overlap_beta_adjoint);
  return total_spatial_gradient;
}

Matrix backpropagate_trace_pullback_only(
    const CandidateContext& context,
    const FastSampleOperands& operands,
    double t_value,
    const ConstMatrixRef& alpha_trace_adjoint,
    const ConstMatrixRef& beta_trace_adjoint) {
  const int n_active_orbitals = context.n_active_orbitals;
  const int n_blocked_alpha =
      static_cast<int>(context.left_blocked_alpha_orbitals.size());
  const int n_pair_orbitals =
      static_cast<int>(context.left_pair_orbitals.size());

  Matrix alpha_trace_ordered_adjoint =
      extract_submatrix(
          alpha_trace_adjoint,
          context.left_ordered_orbitals,
          context.right_ordered_orbitals);
  Matrix beta_trace_reduced_adjoint =
      extract_submatrix(
          beta_trace_adjoint,
          context.left_pair_orbitals,
          context.right_pair_orbitals);

  Matrix blocked_overlap_alpha_adjoint =
      Matrix::Zero(n_blocked_alpha, n_blocked_alpha);
  Matrix blocked_to_right_pair_overlap_alpha_adjoint =
      Matrix::Zero(n_blocked_alpha, n_pair_orbitals);
  Matrix left_pair_to_block_overlap_alpha_adjoint =
      Matrix::Zero(n_pair_orbitals, n_blocked_alpha);
  Matrix pair_overlap_alpha_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix pair_overlap_beta_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix left_pair_ba_reduced_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix right_pair_ab_reduced_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix blocked_to_pair_matrix_adjoint =
      Matrix::Zero(n_blocked_alpha, n_pair_orbitals);
  Matrix beta_left_factor_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix alpha_to_block_factor_adjoint =
      Matrix::Zero(n_pair_orbitals, n_blocked_alpha);
  Matrix pair_kernel_seed_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix sample_bottom_left_adjoint =
      Matrix::Zero(n_pair_orbitals, n_blocked_alpha);
  Matrix sample_bottom_right_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);

  Matrix sample_inverse_adjoint =
      Matrix::Zero(
          n_blocked_alpha + n_pair_orbitals,
          n_blocked_alpha + n_pair_orbitals);
  const Matrix sample_inverse = operands.sample_inverse;
  const Matrix sample_inverse_oo =
      sample_inverse.topLeftCorner(
          n_blocked_alpha,
          n_blocked_alpha);
  const Matrix sample_inverse_op =
      sample_inverse.topRightCorner(
          n_blocked_alpha,
          n_pair_orbitals);
  const Matrix sample_inverse_po =
      sample_inverse.bottomLeftCorner(
          n_pair_orbitals,
          n_blocked_alpha);
  const Matrix sample_inverse_pp =
      sample_inverse.bottomRightCorner(
          n_pair_orbitals,
          n_pair_orbitals);
  const Matrix alpha_oo_adjoint =
      alpha_trace_ordered_adjoint.topLeftCorner(
          n_blocked_alpha,
          n_blocked_alpha);
  const Matrix alpha_op_adjoint =
      alpha_trace_ordered_adjoint.topRightCorner(
          n_blocked_alpha,
          n_pair_orbitals);
  const Matrix alpha_po_adjoint =
      alpha_trace_ordered_adjoint.bottomLeftCorner(
          n_pair_orbitals,
          n_blocked_alpha);
  const Matrix alpha_pp_adjoint =
      alpha_trace_ordered_adjoint.bottomRightCorner(
          n_pair_orbitals,
          n_pair_orbitals);
  Matrix sample_inverse_oo_adjoint =
      alpha_oo_adjoint.transpose();
  Matrix sample_inverse_op_adjoint =
      Matrix::Zero(n_blocked_alpha, n_pair_orbitals);
  Matrix sample_inverse_po_adjoint =
      Matrix::Zero(n_pair_orbitals, n_blocked_alpha);
  Matrix sample_inverse_pp_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);

  const Matrix alpha_op_intermediate =
      sample_inverse_po.transpose();
  sample_inverse_po_adjoint.noalias() +=
      (t_value *
       alpha_op_adjoint *
       context.right_pair_ab_reduced)
          .transpose();
  right_pair_ab_reduced_adjoint.noalias() +=
      t_value *
      alpha_op_adjoint.transpose() *
      alpha_op_intermediate;

  const Matrix alpha_po_intermediate =
      sample_inverse_op.transpose();
  beta_left_factor_adjoint.noalias() +=
      alpha_po_intermediate *
      alpha_po_adjoint.transpose();
  sample_inverse_op_adjoint.noalias() +=
      (context.beta_left_factor *
       alpha_po_adjoint)
          .transpose();

  const Matrix alpha_pp_intermediate_1 =
      sample_inverse_pp.transpose();
  const Matrix alpha_pp_intermediate_2 =
      context.beta_left_factor.transpose() *
      alpha_pp_intermediate_1;
  Matrix alpha_pp_intermediate_2_adjoint =
      t_value *
      alpha_pp_adjoint *
      context.right_pair_ab_reduced;
  right_pair_ab_reduced_adjoint.noalias() +=
      t_value *
      alpha_pp_adjoint.transpose() *
      alpha_pp_intermediate_2;
  beta_left_factor_adjoint.noalias() +=
      alpha_pp_intermediate_1 *
      alpha_pp_intermediate_2_adjoint.transpose();
  sample_inverse_pp_adjoint.noalias() +=
      (context.beta_left_factor *
       alpha_pp_intermediate_2_adjoint)
          .transpose();

  alpha_to_block_factor_adjoint.noalias() +=
      beta_trace_reduced_adjoint *
      sample_inverse_op.transpose();
  sample_inverse_op_adjoint.noalias() +=
      context.alpha_to_block_factor.transpose() *
      beta_trace_reduced_adjoint;
  pair_kernel_seed_adjoint.noalias() +=
      t_value *
      beta_trace_reduced_adjoint *
      sample_inverse_pp.transpose();
  sample_inverse_pp_adjoint.noalias() +=
      t_value *
      context.pair_kernel_seed.transpose() *
      beta_trace_reduced_adjoint;

  sample_inverse_adjoint.topLeftCorner(
      n_blocked_alpha,
      n_blocked_alpha).noalias() +=
      sample_inverse_oo_adjoint;
  sample_inverse_adjoint.topRightCorner(
      n_blocked_alpha,
      n_pair_orbitals).noalias() +=
      sample_inverse_op_adjoint;
  sample_inverse_adjoint.bottomLeftCorner(
      n_pair_orbitals,
      n_blocked_alpha).noalias() +=
      sample_inverse_po_adjoint;
  sample_inverse_adjoint.bottomRightCorner(
      n_pair_orbitals,
      n_pair_orbitals).noalias() +=
      sample_inverse_pp_adjoint;

  Matrix sample_matrix_adjoint =
      -sample_inverse.transpose() *
      sample_inverse_adjoint *
      sample_inverse.transpose();

  if (n_blocked_alpha > 0) {
    blocked_overlap_alpha_adjoint.noalias() +=
        sample_matrix_adjoint.topLeftCorner(
            n_blocked_alpha,
            n_blocked_alpha);
    blocked_to_pair_matrix_adjoint.noalias() +=
        t_value *
        sample_matrix_adjoint.topRightCorner(
            n_blocked_alpha,
            n_pair_orbitals);
    sample_bottom_left_adjoint.noalias() +=
        sample_matrix_adjoint.bottomLeftCorner(
            n_pair_orbitals,
            n_blocked_alpha);
  }
  sample_bottom_right_adjoint.noalias() +=
      t_value *
      sample_matrix_adjoint.bottomRightCorner(
          n_pair_orbitals,
          n_pair_orbitals);

  beta_left_factor_adjoint.noalias() +=
      sample_bottom_left_adjoint *
      context.left_pair_to_block_overlap_alpha.transpose();
  left_pair_to_block_overlap_alpha_adjoint.noalias() +=
      context.beta_left_factor.transpose() *
      sample_bottom_left_adjoint;

  beta_left_factor_adjoint.noalias() +=
      sample_bottom_right_adjoint *
      (context.pair_overlap_alpha * context.right_pair_ab_reduced).transpose();
  pair_overlap_alpha_adjoint.noalias() +=
      context.beta_left_factor.transpose() *
      sample_bottom_right_adjoint *
      context.right_pair_ab_reduced.transpose();
  right_pair_ab_reduced_adjoint.noalias() +=
      (context.beta_left_factor * context.pair_overlap_alpha).transpose() *
      sample_bottom_right_adjoint;

  blocked_to_right_pair_overlap_alpha_adjoint.noalias() +=
      blocked_to_pair_matrix_adjoint *
      context.right_pair_ab_reduced.transpose();
  right_pair_ab_reduced_adjoint.noalias() +=
      context.blocked_to_right_pair_overlap_alpha.transpose() *
      blocked_to_pair_matrix_adjoint;

  pair_overlap_beta_adjoint.noalias() +=
      context.left_pair_ba_reduced *
      beta_left_factor_adjoint.transpose();
  left_pair_ba_reduced_adjoint.noalias() +=
      context.pair_overlap_beta *
      beta_left_factor_adjoint;

  left_pair_ba_reduced_adjoint.noalias() +=
      alpha_to_block_factor_adjoint *
      context.left_pair_to_block_overlap_alpha.transpose();
  left_pair_to_block_overlap_alpha_adjoint.noalias() +=
      context.left_pair_ba_reduced.transpose() *
      alpha_to_block_factor_adjoint;

  left_pair_ba_reduced_adjoint.noalias() +=
      pair_kernel_seed_adjoint *
      (context.pair_overlap_alpha * context.right_pair_ab_reduced).transpose();
  pair_overlap_alpha_adjoint.noalias() +=
      context.left_pair_ba_reduced.transpose() *
      pair_kernel_seed_adjoint *
      context.right_pair_ab_reduced.transpose();
  right_pair_ab_reduced_adjoint.noalias() +=
      (context.left_pair_ba_reduced * context.pair_overlap_alpha).transpose() *
      pair_kernel_seed_adjoint;

  Matrix total_spatial_gradient =
      Matrix::Zero(n_active_orbitals, n_active_orbitals);
  accumulate_submatrix(
      &total_spatial_gradient,
      context.left_blocked_alpha_orbitals,
      context.right_blocked_alpha_orbitals,
      blocked_overlap_alpha_adjoint);
  accumulate_submatrix(
      &total_spatial_gradient,
      context.left_blocked_alpha_orbitals,
      context.right_pair_orbitals,
      blocked_to_right_pair_overlap_alpha_adjoint);
  accumulate_submatrix(
      &total_spatial_gradient,
      context.left_pair_orbitals,
      context.right_blocked_alpha_orbitals,
      left_pair_to_block_overlap_alpha_adjoint);
  accumulate_submatrix(
      &total_spatial_gradient,
      context.left_pair_orbitals,
      context.right_pair_orbitals,
      pair_overlap_alpha_adjoint + pair_overlap_beta_adjoint);
  return total_spatial_gradient;
}

Matrix backpropagate_exchange_pullback_only(
    const CandidateContext& context,
    const FastSampleOperands& operands,
    double t_value,
    const ConstMatrixRef& opposite_exchange_left_adjoint,
    const ConstMatrixRef& opposite_exchange_right_adjoint,
    const ConstMatrixRef& opposite_mixed_right_adjoint) {
  const int n_active_orbitals = context.n_active_orbitals;
  const int n_blocked_alpha =
      static_cast<int>(context.left_blocked_alpha_orbitals.size());
  const int n_pair_orbitals =
      static_cast<int>(context.left_pair_orbitals.size());

  Matrix opposite_exchange_left_ordered_adjoint =
      extract_submatrix(
          opposite_exchange_left_adjoint,
          context.right_ordered_orbitals,
          context.right_pair_orbitals);
  Matrix opposite_exchange_right_ordered_adjoint =
      extract_submatrix(
          opposite_exchange_right_adjoint,
          context.left_pair_orbitals,
          context.left_ordered_orbitals);
  Matrix opposite_mixed_right_ordered_adjoint =
      extract_submatrix(
          opposite_mixed_right_adjoint,
          context.left_pair_orbitals,
          context.left_ordered_orbitals);

  Matrix blocked_overlap_alpha_adjoint =
      Matrix::Zero(n_blocked_alpha, n_blocked_alpha);
  Matrix blocked_to_right_pair_overlap_alpha_adjoint =
      Matrix::Zero(n_blocked_alpha, n_pair_orbitals);
  Matrix left_pair_to_block_overlap_alpha_adjoint =
      Matrix::Zero(n_pair_orbitals, n_blocked_alpha);
  Matrix pair_overlap_alpha_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix pair_overlap_beta_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix left_pair_ba_reduced_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix right_pair_ab_reduced_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix blocked_to_pair_matrix_adjoint =
      Matrix::Zero(n_blocked_alpha, n_pair_orbitals);
  Matrix beta_trace_reduced_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix beta_left_factor_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix alpha_to_block_factor_adjoint =
      Matrix::Zero(n_pair_orbitals, n_blocked_alpha);
  Matrix pair_kernel_seed_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);
  Matrix sample_bottom_left_adjoint =
      Matrix::Zero(n_pair_orbitals, n_blocked_alpha);
  Matrix sample_bottom_right_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);

  Matrix sample_inverse_adjoint =
      Matrix::Zero(
          n_blocked_alpha + n_pair_orbitals,
          n_blocked_alpha + n_pair_orbitals);
  const Matrix sample_inverse = operands.sample_inverse;
  const Matrix sample_inverse_oo =
      sample_inverse.topLeftCorner(
          n_blocked_alpha,
          n_blocked_alpha);
  const Matrix sample_inverse_op =
      sample_inverse.topRightCorner(
          n_blocked_alpha,
          n_pair_orbitals);
  const Matrix sample_inverse_po =
      sample_inverse.bottomLeftCorner(
          n_pair_orbitals,
          n_blocked_alpha);
  const Matrix sample_inverse_pp =
      sample_inverse.bottomRightCorner(
          n_pair_orbitals,
          n_pair_orbitals);
  Matrix sample_inverse_oo_adjoint =
      Matrix::Zero(n_blocked_alpha, n_blocked_alpha);
  Matrix sample_inverse_op_adjoint =
      Matrix::Zero(n_blocked_alpha, n_pair_orbitals);
  Matrix sample_inverse_po_adjoint =
      Matrix::Zero(n_pair_orbitals, n_blocked_alpha);
  Matrix sample_inverse_pp_adjoint =
      Matrix::Zero(n_pair_orbitals, n_pair_orbitals);

  const Matrix opposite_left_top_adjoint =
      opposite_exchange_left_ordered_adjoint.topRows(n_blocked_alpha);
  const Matrix opposite_left_bottom_adjoint =
      opposite_exchange_left_ordered_adjoint.bottomRows(n_pair_orbitals);
  sample_inverse_op_adjoint.noalias() +=
      opposite_left_top_adjoint;
  right_pair_ab_reduced_adjoint.noalias() +=
      t_value *
      opposite_left_bottom_adjoint *
      sample_inverse_pp.transpose();
  sample_inverse_pp_adjoint.noalias() +=
      t_value *
      context.right_pair_ab_reduced.transpose() *
      opposite_left_bottom_adjoint;

  const Matrix opposite_right_left_adjoint =
      opposite_exchange_right_ordered_adjoint.leftCols(n_blocked_alpha);
  const Matrix opposite_right_right_adjoint =
      opposite_exchange_right_ordered_adjoint.rightCols(n_pair_orbitals);
  alpha_to_block_factor_adjoint.noalias() +=
      opposite_right_left_adjoint *
      sample_inverse_oo.transpose();
  sample_inverse_oo_adjoint.noalias() +=
      context.alpha_to_block_factor.transpose() *
      opposite_right_left_adjoint;
  pair_kernel_seed_adjoint.noalias() +=
      t_value *
      opposite_right_left_adjoint *
      sample_inverse_po.transpose();
  sample_inverse_po_adjoint.noalias() +=
      t_value *
      context.pair_kernel_seed.transpose() *
      opposite_right_left_adjoint;
  beta_trace_reduced_adjoint.noalias() +=
      opposite_right_right_adjoint *
      context.beta_left_factor.transpose();
  beta_left_factor_adjoint.noalias() +=
      operands.beta_trace_reduced.transpose() *
      opposite_right_right_adjoint;
  left_pair_ba_reduced_adjoint.noalias() +=
      opposite_mixed_right_ordered_adjoint.rightCols(n_pair_orbitals);

  alpha_to_block_factor_adjoint.noalias() +=
      beta_trace_reduced_adjoint *
      sample_inverse_op.transpose();
  sample_inverse_op_adjoint.noalias() +=
      context.alpha_to_block_factor.transpose() *
      beta_trace_reduced_adjoint;
  pair_kernel_seed_adjoint.noalias() +=
      t_value *
      beta_trace_reduced_adjoint *
      sample_inverse_pp.transpose();
  sample_inverse_pp_adjoint.noalias() +=
      t_value *
      context.pair_kernel_seed.transpose() *
      beta_trace_reduced_adjoint;

  sample_inverse_adjoint.topLeftCorner(
      n_blocked_alpha,
      n_blocked_alpha).noalias() +=
      sample_inverse_oo_adjoint;
  sample_inverse_adjoint.topRightCorner(
      n_blocked_alpha,
      n_pair_orbitals).noalias() +=
      sample_inverse_op_adjoint;
  sample_inverse_adjoint.bottomLeftCorner(
      n_pair_orbitals,
      n_blocked_alpha).noalias() +=
      sample_inverse_po_adjoint;
  sample_inverse_adjoint.bottomRightCorner(
      n_pair_orbitals,
      n_pair_orbitals).noalias() +=
      sample_inverse_pp_adjoint;

  Matrix sample_matrix_adjoint =
      -sample_inverse.transpose() *
      sample_inverse_adjoint *
      sample_inverse.transpose();

  if (n_blocked_alpha > 0) {
    blocked_overlap_alpha_adjoint.noalias() +=
        sample_matrix_adjoint.topLeftCorner(
            n_blocked_alpha,
            n_blocked_alpha);
    blocked_to_pair_matrix_adjoint.noalias() +=
        t_value *
        sample_matrix_adjoint.topRightCorner(
            n_blocked_alpha,
            n_pair_orbitals);
    sample_bottom_left_adjoint.noalias() +=
        sample_matrix_adjoint.bottomLeftCorner(
            n_pair_orbitals,
            n_blocked_alpha);
  }
  sample_bottom_right_adjoint.noalias() +=
      t_value *
      sample_matrix_adjoint.bottomRightCorner(
          n_pair_orbitals,
          n_pair_orbitals);

  beta_left_factor_adjoint.noalias() +=
      sample_bottom_left_adjoint *
      context.left_pair_to_block_overlap_alpha.transpose();
  left_pair_to_block_overlap_alpha_adjoint.noalias() +=
      context.beta_left_factor.transpose() *
      sample_bottom_left_adjoint;

  beta_left_factor_adjoint.noalias() +=
      sample_bottom_right_adjoint *
      (context.pair_overlap_alpha * context.right_pair_ab_reduced).transpose();
  pair_overlap_alpha_adjoint.noalias() +=
      context.beta_left_factor.transpose() *
      sample_bottom_right_adjoint *
      context.right_pair_ab_reduced.transpose();
  right_pair_ab_reduced_adjoint.noalias() +=
      (context.beta_left_factor * context.pair_overlap_alpha).transpose() *
      sample_bottom_right_adjoint;

  blocked_to_right_pair_overlap_alpha_adjoint.noalias() +=
      blocked_to_pair_matrix_adjoint *
      context.right_pair_ab_reduced.transpose();
  right_pair_ab_reduced_adjoint.noalias() +=
      context.blocked_to_right_pair_overlap_alpha.transpose() *
      blocked_to_pair_matrix_adjoint;

  pair_overlap_beta_adjoint.noalias() +=
      context.left_pair_ba_reduced *
      beta_left_factor_adjoint.transpose();
  left_pair_ba_reduced_adjoint.noalias() +=
      context.pair_overlap_beta *
      beta_left_factor_adjoint;

  left_pair_ba_reduced_adjoint.noalias() +=
      alpha_to_block_factor_adjoint *
      context.left_pair_to_block_overlap_alpha.transpose();
  left_pair_to_block_overlap_alpha_adjoint.noalias() +=
      context.left_pair_ba_reduced.transpose() *
      alpha_to_block_factor_adjoint;

  left_pair_ba_reduced_adjoint.noalias() +=
      pair_kernel_seed_adjoint *
      (context.pair_overlap_alpha * context.right_pair_ab_reduced).transpose();
  pair_overlap_alpha_adjoint.noalias() +=
      context.left_pair_ba_reduced.transpose() *
      pair_kernel_seed_adjoint *
      context.right_pair_ab_reduced.transpose();
  right_pair_ab_reduced_adjoint.noalias() +=
      (context.left_pair_ba_reduced * context.pair_overlap_alpha).transpose() *
      pair_kernel_seed_adjoint;

  Matrix total_spatial_gradient =
      Matrix::Zero(n_active_orbitals, n_active_orbitals);
  accumulate_submatrix(
      &total_spatial_gradient,
      context.left_blocked_alpha_orbitals,
      context.right_blocked_alpha_orbitals,
      blocked_overlap_alpha_adjoint);
  accumulate_submatrix(
      &total_spatial_gradient,
      context.left_blocked_alpha_orbitals,
      context.right_pair_orbitals,
      blocked_to_right_pair_overlap_alpha_adjoint);
  accumulate_submatrix(
      &total_spatial_gradient,
      context.left_pair_orbitals,
      context.right_blocked_alpha_orbitals,
      left_pair_to_block_overlap_alpha_adjoint);
  accumulate_submatrix(
      &total_spatial_gradient,
      context.left_pair_orbitals,
      context.right_pair_orbitals,
      pair_overlap_alpha_adjoint + pair_overlap_beta_adjoint);
  return total_spatial_gradient;
}

void backpropagate_sample_total(
    const CandidateContext& context,
    const FastSampleOperands& operands,
    double t_value,
    const ConstMatrixRef& one_electron,
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
      total_trace_matrix.cwiseProduct(one_electron).sum();
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
          operands.opposite_exchange_right_matrix);
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
      operands.opposite_exchange_right_matrix,
      packed_two_electron_gradient);
  PfTensorContractor::add_exchange_outer_product(
      scaled_determinant,
      operands.opposite_exchange_left_matrix,
      operands.opposite_mixed_right_matrix,
      packed_two_electron_gradient);

  const double determinant_adjoint =
      sample_weight * total_factor;
  Matrix alpha_trace_adjoint =
      scaled_determinant * one_electron;
  Matrix beta_trace_adjoint =
      scaled_determinant * one_electron;
  Matrix opposite_exchange_left_adjoint =
      Matrix::Zero(n_active_orbitals, n_active_orbitals);
  Matrix opposite_exchange_right_adjoint =
      Matrix::Zero(n_active_orbitals, n_active_orbitals);
  Matrix opposite_mixed_right_adjoint =
      Matrix::Zero(n_active_orbitals, n_active_orbitals);

  accumulate_same_operand_adjoint(
      scaled_determinant,
      packed_two_electron_integrals,
      operands.alpha_trace_matrix,
      &PfTensorContractor::compute_same_spin_separable_operand_adjoints,
      &alpha_trace_adjoint);
  accumulate_same_operand_adjoint(
      -scaled_determinant,
      packed_two_electron_integrals,
      operands.alpha_trace_matrix,
      &PfTensorContractor::compute_same_spin_bridge_operand_adjoints,
      &alpha_trace_adjoint);
  accumulate_same_operand_adjoint(
      scaled_determinant,
      packed_two_electron_integrals,
      operands.beta_trace_matrix,
      &PfTensorContractor::compute_same_spin_separable_operand_adjoints,
      &beta_trace_adjoint);
  accumulate_same_operand_adjoint(
      -scaled_determinant,
      packed_two_electron_integrals,
      operands.beta_trace_matrix,
      &PfTensorContractor::compute_same_spin_bridge_operand_adjoints,
      &beta_trace_adjoint);
  PfTensorContractor::compute_direct_operand_adjoints(
      scaled_determinant,
      operands.alpha_trace_matrix,
      operands.beta_trace_matrix,
      packed_two_electron_integrals,
      alpha_trace_adjoint,
      beta_trace_adjoint);
  PfTensorContractor::compute_exchange_operand_adjoints(
      -scaled_determinant,
      operands.opposite_exchange_left_matrix,
      operands.opposite_exchange_right_matrix,
      packed_two_electron_integrals,
      opposite_exchange_left_adjoint,
      opposite_exchange_right_adjoint);
  PfTensorContractor::compute_exchange_operand_adjoints(
      scaled_determinant,
      operands.opposite_exchange_left_matrix,
      operands.opposite_mixed_right_matrix,
      packed_two_electron_integrals,
      opposite_exchange_left_adjoint,
      opposite_mixed_right_adjoint);

  total_spatial_gradient->noalias() +=
      backpropagate_determinant_pullback_only(
          context,
          operands,
          t_value,
          determinant_adjoint);
  total_spatial_gradient->noalias() +=
      backpropagate_trace_pullback_only(
          context,
          operands,
          t_value,
          alpha_trace_adjoint,
          beta_trace_adjoint);
  total_spatial_gradient->noalias() +=
      backpropagate_exchange_pullback_only(
          context,
          operands,
          t_value,
          opposite_exchange_left_adjoint,
          opposite_exchange_right_adjoint,
          opposite_mixed_right_adjoint);
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
          operands.opposite_exchange_right_matrix);
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

PfHighSpinOpenShellHamiltonianResult evaluate_high_spin_open_shell_pair_hamiltonian(
    const ConstMatrixRef& left_pair_ba,
    const ConstMatrixRef& right_pair_ab,
    const std::vector<int>& left_blocked_alpha_orbitals,
    const std::vector<int>& right_blocked_alpha_orbitals,
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

  const CandidateContext context =
      build_candidate_context(
          left_pair_ba,
          right_pair_ab,
          left_blocked_alpha_orbitals,
          right_blocked_alpha_orbitals,
          spatial_overlap_matrix,
          n_singlet_pairs);
  const int n_active_orbitals = context.n_active_orbitals;
  const Eigen::RowVectorXd weights =
      context.inverse_vandermonde.row(n_singlet_pairs);

  PfHighSpinOpenShellHamiltonianResult result;
  result.interpolation_degree = context.interpolation_degree;
  if (include_gradients) {
    result.overlap_spatial_gradient =
        Matrix::Zero(n_active_orbitals, n_active_orbitals);
    result.total_hamiltonian_spatial_gradient =
        Matrix::Zero(n_active_orbitals, n_active_orbitals);
    result.one_electron_matrix_gradient =
        Matrix::Zero(n_active_orbitals, n_active_orbitals);
    result.packed_two_electron_gradient.assign(
        PfTensorContractor::packed_two_electron_count(n_active_orbitals),
        0.0);
    result.gradients_computed = true;
  }

  for (int node = 0; node <= context.interpolation_degree; ++node) {
    const double weight = weights(node);
    const double t_value =
        context.interpolation_nodes[xmvb::to_size(node)];
    const FastSampleOperands operands =
        build_fast_sample_operands(
            context,
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
            n_active_orbitals);

    result.overlap += weight * operands.determinant;
    result.one_electron_hamiltonian += weight * sample_one_electron;
    result.total_hamiltonian +=
        weight * (sample_one_electron + sample_two_electron);

    if (include_gradients) {
      result.overlap_spatial_gradient.noalias() +=
          backpropagate_determinant_pullback_only(
              context,
              operands,
              t_value,
              weight);
      backpropagate_sample_total(
          context,
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

  return result;
}

PfHighSpinOpenShellHamiltonianResult evaluate_high_spin_open_shell_pf_state_pair_hamiltonian(
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
  if (!left_state.blocked_beta_orbitals.empty() ||
      !right_state.blocked_beta_orbitals.empty()) {
    throw std::invalid_argument(
        "high-spin open-shell Hamiltonian evaluator currently supports blocked alpha orbitals only");
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

  return evaluate_high_spin_open_shell_pair_hamiltonian(
      left_pair_ba,
      right_pair_ab,
      left_state.blocked_alpha_orbitals,
      right_state.blocked_alpha_orbitals,
      spatial_overlap_matrix,
      one_electron_matrix,
      packed_two_electron_integrals,
      left_state.n_singlet_pairs,
      include_gradients);
}

}  // namespace xmvb::pfaffian_vbscf
