#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/LU>
#include <Eigen/QR>

#include "pfaffian_vbscf/kernel/pf_high_spin_open_shell_hamiltonian.hpp"
#include "pfaffian_vbscf/tensor/pf_tensor_contractor.hpp"
#include "vb/matrices/full_determinant_pair_evaluator.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace {

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using ScalarBuffer = xmvb::pfaffian_vbscf::ScalarBuffer;
using ConstMatrixRef = xmvb::pfaffian_vbscf::ConstMatrixRef;
using MatrixRef = xmvb::pfaffian_vbscf::MatrixRef;

constexpr double kInterpolationNodeScale = 0.008;

struct DeterminantTerm {
  std::vector<int> alpha_occ;
  std::vector<int> beta_occ;
  double coefficient = 0.0;
};

struct ExactStatePairResult {
  double overlap = 0.0;
  double one_electron_hamiltonian = 0.0;
  double total_hamiltonian = 0.0;
};

struct CandidateResult {
  double overlap = 0.0;
  double one_electron_hamiltonian = 0.0;
  double total_hamiltonian = 0.0;
  int interpolation_degree = 0;
  Matrix total_hamiltonian_spatial_gradient;
  Matrix one_electron_matrix_gradient;
  ScalarBuffer packed_two_electron_gradient;
};

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

struct FastSampleOperands;

CandidateResult evaluate_candidate(
    const Matrix& left_pair_ab,
    const Matrix& right_pair_ab,
    const std::vector<int>& left_blocked_alpha_orbitals,
    const std::vector<int>& right_blocked_alpha_orbitals,
    const Matrix& spatial_overlap,
    const Matrix& one_electron,
    const std::vector<double>& packed_two_electron_integrals,
    int n_singlet_pairs);

double matrix_dot(
    const Matrix& left,
    const Matrix& right);

Matrix backpropagate_exchange_pullback_only(
    const CandidateContext& context,
    const FastSampleOperands& operands,
    double t_value,
    const Matrix& opposite_exchange_left_adjoint,
    const Matrix& opposite_exchange_right_adjoint,
    const Matrix& opposite_mixed_right_adjoint);

Matrix backpropagate_trace_pullback_only(
    const CandidateContext& context,
    const FastSampleOperands& operands,
    double t_value,
    const Matrix& alpha_trace_adjoint,
    const Matrix& beta_trace_adjoint);

Matrix backpropagate_determinant_pullback_only(
    const CandidateContext& context,
    const FastSampleOperands& operands,
    double t_value,
    double determinant_adjoint);

int canonicalize_spin_string(std::vector<int>* occupied_orbitals) {
  if (occupied_orbitals == nullptr) {
    throw std::invalid_argument("occupied_orbitals must not be null");
  }
  int permutation_sign = 1;
  for (std::size_t left_index = 0; left_index + 1 < occupied_orbitals->size(); ++left_index) {
    for (std::size_t right_index = left_index + 1;
         right_index < occupied_orbitals->size();
         ++right_index) {
      if ((*occupied_orbitals)[left_index] > (*occupied_orbitals)[right_index]) {
        std::swap((*occupied_orbitals)[left_index], (*occupied_orbitals)[right_index]);
        permutation_sign = -permutation_sign;
      }
    }
  }
  return permutation_sign;
}

void enumerate_combinations_recursive(
    int start,
    int remaining,
    int dimension,
    std::vector<int>* current,
    std::vector<std::vector<int>>* combinations) {
  if (current == nullptr || combinations == nullptr) {
    throw std::invalid_argument("enumeration buffers must not be null");
  }
  if (remaining == 0) {
    combinations->push_back(*current);
    return;
  }
  for (int value = start; value <= dimension - remaining; ++value) {
    current->push_back(value);
    enumerate_combinations_recursive(
        value + 1,
        remaining - 1,
        dimension,
        current,
        combinations);
    current->pop_back();
  }
}

std::vector<std::vector<int>> enumerate_combinations(
    int dimension,
    int choose) {
  if (choose < 0 || choose > dimension) {
    throw std::invalid_argument("invalid combination size");
  }
  std::vector<std::vector<int>> combinations;
  std::vector<int> current;
  enumerate_combinations_recursive(
      0,
      choose,
      dimension,
      &current,
      &combinations);
  return combinations;
}

Matrix extract_submatrix(
    const Matrix& matrix,
    const std::vector<int>& rows,
    const std::vector<int>& cols) {
  Matrix submatrix =
      Matrix::Zero(
          static_cast<int>(rows.size()),
          static_cast<int>(cols.size()));
  for (int col = 0; col < static_cast<int>(cols.size()); ++col) {
    for (int row = 0; row < static_cast<int>(rows.size()); ++row) {
      submatrix(row, col) =
          matrix(rows[xmvb::to_size(row)],
                 cols[xmvb::to_size(col)]);
    }
  }
  return submatrix;
}

void accumulate_submatrix(
    Matrix* target,
    const std::vector<int>& rows,
    const std::vector<int>& cols,
    const Matrix& source) {
  if (target == nullptr) {
    throw std::invalid_argument("target must not be null");
  }
  if (source.rows() != static_cast<int>(rows.size()) ||
      source.cols() != static_cast<int>(cols.size())) {
    throw std::invalid_argument("source dimensions do not match the index sets");
  }
  for (int col = 0; col < source.cols(); ++col) {
    for (int row = 0; row < source.rows(); ++row) {
      (*target)(
          rows[xmvb::to_size(row)],
          cols[xmvb::to_size(col)]) += source(row, col);
    }
  }
}

Matrix scatter_submatrix(
    int dimension,
    const std::vector<int>& rows,
    const std::vector<int>& cols,
    const Matrix& submatrix) {
  if (submatrix.rows() != static_cast<int>(rows.size()) ||
      submatrix.cols() != static_cast<int>(cols.size())) {
    throw std::invalid_argument("scatter_submatrix dimensions do not match the index sets");
  }
  Matrix matrix = Matrix::Zero(dimension, dimension);
  for (int col = 0; col < static_cast<int>(cols.size()); ++col) {
    for (int row = 0; row < static_cast<int>(rows.size()); ++row) {
      matrix(
          rows[xmvb::to_size(row)],
          cols[xmvb::to_size(col)]) = submatrix(row, col);
    }
  }
  return matrix;
}

std::vector<double> flatten_matrix(const Matrix& matrix) {
  std::vector<double> data(
      xmvb::to_size(matrix.rows() * matrix.cols()),
      0.0);
  for (int col = 0; col < matrix.cols(); ++col) {
    for (int row = 0; row < matrix.rows(); ++row) {
      data[xmvb::to_size(col) * matrix.rows() + row] = matrix(row, col);
    }
  }
  return data;
}

std::vector<DeterminantTerm> enumerate_state_terms(
    const Matrix& pair_matrix_ab,
    const std::vector<int>& blocked_alpha_orbitals,
    int n_singlet_pairs) {
  const int n_active_orbitals = static_cast<int>(pair_matrix_ab.rows());
  if (pair_matrix_ab.rows() != pair_matrix_ab.cols()) {
    throw std::invalid_argument("pair_matrix_ab must be square");
  }

  std::vector<DeterminantTerm> terms;
  const auto alpha_combinations =
      enumerate_combinations(n_active_orbitals, n_singlet_pairs);
  const auto beta_combinations =
      enumerate_combinations(n_active_orbitals, n_singlet_pairs);

  for (const auto& alpha_subset : alpha_combinations) {
    for (const auto& beta_subset : beta_combinations) {
      const Matrix pair_submatrix =
          extract_submatrix(pair_matrix_ab, alpha_subset, beta_subset);
      const double pair_coefficient =
          Eigen::FullPivLU<Matrix>(pair_submatrix).determinant();
      if (std::abs(pair_coefficient) <= 1.0e-12) {
        continue;
      }

      DeterminantTerm term;
      term.alpha_occ = blocked_alpha_orbitals;
      term.alpha_occ.insert(
          term.alpha_occ.end(),
          alpha_subset.begin(),
          alpha_subset.end());
      term.beta_occ = beta_subset;
      term.coefficient = pair_coefficient;
      const int alpha_sign = canonicalize_spin_string(&term.alpha_occ);
      const int beta_sign = canonicalize_spin_string(&term.beta_occ);
      term.coefficient *= static_cast<double>(alpha_sign * beta_sign);
      terms.push_back(std::move(term));
    }
  }

  return terms;
}

ExactStatePairResult exact_state_pair_evaluation(
    const std::vector<DeterminantTerm>& left_terms,
    const std::vector<DeterminantTerm>& right_terms,
    const Matrix& spatial_overlap,
    const Matrix& one_electron,
    const std::vector<double>& packed_two_electron_integrals) {
  const int n_active_orbitals = static_cast<int>(spatial_overlap.rows());
  if (spatial_overlap.rows() != spatial_overlap.cols() ||
      one_electron.rows() != one_electron.cols() ||
      spatial_overlap.rows() != one_electron.rows()) {
    throw std::invalid_argument("spatial matrices must be square and match");
  }

  const auto flat_overlap = flatten_matrix(spatial_overlap);
  const auto flat_one_electron = flatten_matrix(one_electron);
  const xmvb::vb::FullDeterminantPairEvaluator evaluator;

  ExactStatePairResult result;
  for (const auto& left_term : left_terms) {
    for (const auto& right_term : right_terms) {
      const auto determinant_pair =
          evaluator.evaluate(
              left_term.alpha_occ,
              right_term.alpha_occ,
              left_term.beta_occ,
              right_term.beta_occ,
              flat_overlap,
              flat_one_electron,
              n_active_orbitals,
              packed_two_electron_integrals);
      const double prefactor =
          left_term.coefficient * right_term.coefficient;
      result.overlap +=
          prefactor * determinant_pair.overlap_determinant;
      result.one_electron_hamiltonian +=
          prefactor * determinant_pair.one_electron_hamiltonian;
      result.total_hamiltonian +=
          prefactor * determinant_pair.total_hamiltonian;
    }
  }
  return result;
}

Matrix random_spd_matrix(
    int dimension,
    std::mt19937* generator,
    std::normal_distribution<double>* distribution) {
  Matrix random_matrix = Matrix::Zero(dimension, dimension);
  for (int col = 0; col < dimension; ++col) {
    for (int row = 0; row < dimension; ++row) {
      random_matrix(row, col) = (*distribution)(*generator);
    }
  }
  Matrix spd =
      random_matrix.transpose() * random_matrix +
      0.5 * Matrix::Identity(dimension, dimension);
  return 0.5 * (spd + spd.transpose());
}

Matrix random_pair_matrix(
    int dimension,
    const std::vector<int>& blocked_alpha_orbitals,
    std::mt19937* generator,
    std::normal_distribution<double>* distribution) {
  Matrix pair_matrix = Matrix::Zero(dimension, dimension);
  std::vector<bool> blocked_mask(xmvb::to_size(dimension), false);
  for (const int orbital : blocked_alpha_orbitals) {
    blocked_mask[xmvb::to_size(orbital)] = true;
  }
  for (int col = 0; col < dimension; ++col) {
    for (int row = 0; row < dimension; ++row) {
      if (blocked_mask[xmvb::to_size(row)] ||
          blocked_mask[xmvb::to_size(col)]) {
        continue;
      }
      pair_matrix(row, col) = (*distribution)(*generator);
    }
  }
  return pair_matrix;
}

std::vector<double> random_packed_two_electron_integrals(
    int n_active_orbitals,
    std::mt19937* generator,
    std::normal_distribution<double>* distribution) {
  const int size =
      xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
          n_active_orbitals - 1,
          n_active_orbitals - 1,
          n_active_orbitals - 1,
          n_active_orbitals - 1) +
      1;
  std::vector<double> packed(size, 0.0);
  for (double& value : packed) {
    value = (*distribution)(*generator);
  }
  return packed;
}

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

CandidateContext build_candidate_context(
    const Matrix& left_pair_ba,
    const Matrix& right_pair_ab,
    const std::vector<int>& left_blocked_alpha_orbitals,
    const std::vector<int>& right_blocked_alpha_orbitals,
    const Matrix& spatial_overlap,
    int n_singlet_pairs) {
  const int n_active_orbitals = static_cast<int>(spatial_overlap.rows());
  CandidateContext context;
  context.n_active_orbitals = n_active_orbitals;
  context.n_singlet_pairs = n_singlet_pairs;
  context.left_blocked_alpha_orbitals = left_blocked_alpha_orbitals;
  context.right_blocked_alpha_orbitals = right_blocked_alpha_orbitals;

  std::vector<bool> left_blocked_mask(
      xmvb::to_size(n_active_orbitals),
      false);
  for (int index = 0; index < static_cast<int>(left_blocked_alpha_orbitals.size()); ++index) {
    const int orbital = left_blocked_alpha_orbitals[xmvb::to_size(index)];
    if (orbital < 0 || orbital >= n_active_orbitals) {
      throw std::invalid_argument("left blocked alpha orbital is out of range");
    }
    if (left_blocked_mask[xmvb::to_size(orbital)]) {
      throw std::invalid_argument("left blocked alpha orbital is duplicated");
    }
    left_blocked_mask[xmvb::to_size(orbital)] = true;
  }

  std::vector<bool> right_blocked_mask(
      xmvb::to_size(n_active_orbitals),
      false);
  for (int index = 0; index < static_cast<int>(right_blocked_alpha_orbitals.size()); ++index) {
    const int orbital = right_blocked_alpha_orbitals[xmvb::to_size(index)];
    if (orbital < 0 || orbital >= n_active_orbitals) {
      throw std::invalid_argument("right blocked alpha orbital is out of range");
    }
    if (right_blocked_mask[xmvb::to_size(orbital)]) {
      throw std::invalid_argument("right blocked alpha orbital is duplicated");
    }
    right_blocked_mask[xmvb::to_size(orbital)] = true;
  }

  for (int orbital = 0; orbital < n_active_orbitals; ++orbital) {
    if (!left_blocked_mask[xmvb::to_size(orbital)]) {
      context.left_pair_orbitals.push_back(orbital);
    }
    if (!right_blocked_mask[xmvb::to_size(orbital)]) {
      context.right_pair_orbitals.push_back(orbital);
    }
  }
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

  context.interpolation_nodes.resize(
      xmvb::to_size(context.interpolation_degree + 1),
      0.0);
  for (int node = 0; node <= context.interpolation_degree; ++node) {
    context.interpolation_nodes[xmvb::to_size(node)] =
        kInterpolationNodeScale * static_cast<double>(node);
  }
  context.inverse_vandermonde =
      build_vandermonde_inverse(context.interpolation_degree);
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
    const Matrix& operand,
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

double contraction_directional_from_adjoints(
    const ScalarBuffer& packed_two_electron_integrals,
    const Matrix& left,
    const Matrix& right,
    const Matrix& left_direction,
    const Matrix& right_direction,
    void (*adjoint_function)(
        double,
        const ConstMatrixRef&,
        const ConstMatrixRef&,
        const ScalarBuffer&,
        MatrixRef,
        MatrixRef)) {
  Matrix left_adjoint =
      Matrix::Zero(left.rows(), left.cols());
  Matrix right_adjoint =
      Matrix::Zero(right.rows(), right.cols());
  adjoint_function(
      1.0,
      left,
      right,
      packed_two_electron_integrals,
      left_adjoint,
      right_adjoint);
  return
      matrix_dot(left_adjoint, left_direction) +
      matrix_dot(right_adjoint, right_direction);
}

double same_operand_contraction_directional_from_adjoints(
    const ScalarBuffer& packed_two_electron_integrals,
    const Matrix& operand,
    const Matrix& operand_direction,
    void (*adjoint_function)(
        double,
        const ConstMatrixRef&,
        const ConstMatrixRef&,
        const ScalarBuffer&,
        MatrixRef,
        MatrixRef)) {
  Matrix left_adjoint =
      Matrix::Zero(operand.rows(), operand.cols());
  Matrix right_adjoint =
      Matrix::Zero(operand.rows(), operand.cols());
  adjoint_function(
      1.0,
      operand,
      operand,
      packed_two_electron_integrals,
      left_adjoint,
      right_adjoint);
  return
      matrix_dot(
          left_adjoint + right_adjoint,
          operand_direction);
}

double sample_total_directional_derivative_forward_mode(
    const CandidateContext& context,
    const FastSampleOperands& operands,
    double t_value,
    const Matrix& spatial_direction,
    const Matrix& one_electron,
    const ScalarBuffer& packed_two_electron_integrals,
    double sample_weight) {
  const int n_active_orbitals = context.n_active_orbitals;
  const int n_blocked_alpha =
      static_cast<int>(context.left_blocked_alpha_orbitals.size());
  const int n_pair_orbitals =
      static_cast<int>(context.left_pair_orbitals.size());

  const Matrix blocked_overlap_alpha_direction =
      extract_submatrix(
          spatial_direction,
          context.left_blocked_alpha_orbitals,
          context.right_blocked_alpha_orbitals);
  const Matrix blocked_to_right_pair_overlap_alpha_direction =
      extract_submatrix(
          spatial_direction,
          context.left_blocked_alpha_orbitals,
          context.right_pair_orbitals);
  const Matrix left_pair_to_block_overlap_alpha_direction =
      extract_submatrix(
          spatial_direction,
          context.left_pair_orbitals,
          context.right_blocked_alpha_orbitals);
  const Matrix pair_overlap_alpha_direction =
      extract_submatrix(
          spatial_direction,
          context.left_pair_orbitals,
          context.right_pair_orbitals);
  const Matrix pair_overlap_beta_direction =
      pair_overlap_alpha_direction;

  const Matrix blocked_to_pair_matrix_direction =
      blocked_to_right_pair_overlap_alpha_direction *
      context.right_pair_ab_reduced;
  const Matrix beta_left_factor_direction =
      pair_overlap_beta_direction.transpose() *
      context.left_pair_ba_reduced;
  const Matrix alpha_to_block_factor_direction =
      context.left_pair_ba_reduced *
      left_pair_to_block_overlap_alpha_direction;
  const Matrix pair_kernel_seed_direction =
      context.left_pair_ba_reduced *
      pair_overlap_alpha_direction *
      context.right_pair_ab_reduced;
  const Matrix sample_bottom_left_direction =
      beta_left_factor_direction *
      context.left_pair_to_block_overlap_alpha +
      context.beta_left_factor *
      left_pair_to_block_overlap_alpha_direction;
  const Matrix sample_bottom_right_direction =
      beta_left_factor_direction *
      context.pair_overlap_alpha *
      context.right_pair_ab_reduced +
      context.beta_left_factor *
      pair_overlap_alpha_direction *
      context.right_pair_ab_reduced;

  Matrix sample_matrix_direction =
      Matrix::Zero(
          n_blocked_alpha + n_pair_orbitals,
          n_blocked_alpha + n_pair_orbitals);
  if (n_blocked_alpha > 0) {
    sample_matrix_direction.topLeftCorner(
        n_blocked_alpha,
        n_blocked_alpha) =
        blocked_overlap_alpha_direction;
    sample_matrix_direction.topRightCorner(
        n_blocked_alpha,
        n_pair_orbitals) =
        t_value * blocked_to_pair_matrix_direction;
    sample_matrix_direction.bottomLeftCorner(
        n_pair_orbitals,
        n_blocked_alpha) =
        sample_bottom_left_direction;
  }
  sample_matrix_direction.bottomRightCorner(
      n_pair_orbitals,
      n_pair_orbitals) =
      t_value * sample_bottom_right_direction;

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
  const Matrix sample_inverse_direction =
      -sample_inverse *
      sample_matrix_direction *
      sample_inverse;
  const double determinant_direction =
      operands.determinant *
      (sample_inverse * sample_matrix_direction).trace();

  const Matrix sample_inverse_oo_direction =
      sample_inverse_direction.topLeftCorner(
          n_blocked_alpha,
          n_blocked_alpha);
  const Matrix sample_inverse_op_direction =
      sample_inverse_direction.topRightCorner(
          n_blocked_alpha,
          n_pair_orbitals);
  const Matrix sample_inverse_po_direction =
      sample_inverse_direction.bottomLeftCorner(
          n_pair_orbitals,
          n_blocked_alpha);
  const Matrix sample_inverse_pp_direction =
      sample_inverse_direction.bottomRightCorner(
          n_pair_orbitals,
          n_pair_orbitals);

  Matrix alpha_trace_ordered_direction =
      Matrix::Zero(
          n_blocked_alpha + n_pair_orbitals,
          n_blocked_alpha + n_pair_orbitals);
  if (n_blocked_alpha > 0) {
    alpha_trace_ordered_direction.topLeftCorner(
        n_blocked_alpha,
        n_blocked_alpha) =
        sample_inverse_oo_direction.transpose();
    alpha_trace_ordered_direction.topRightCorner(
        n_blocked_alpha,
        n_pair_orbitals) =
        t_value *
        sample_inverse_po_direction.transpose() *
        context.right_pair_ab_reduced.transpose();
    alpha_trace_ordered_direction.bottomLeftCorner(
        n_pair_orbitals,
        n_blocked_alpha) =
        beta_left_factor_direction.transpose() *
        sample_inverse_op.transpose() +
        context.beta_left_factor.transpose() *
        sample_inverse_op_direction.transpose();
  }
  alpha_trace_ordered_direction.bottomRightCorner(
      n_pair_orbitals,
      n_pair_orbitals) =
      t_value *
      (beta_left_factor_direction.transpose() *
           sample_inverse_pp.transpose() +
       context.beta_left_factor.transpose() *
           sample_inverse_pp_direction.transpose()) *
      context.right_pair_ab_reduced.transpose();

  Matrix beta_trace_reduced_direction =
      alpha_to_block_factor_direction * sample_inverse_op +
      context.alpha_to_block_factor * sample_inverse_op_direction;
  beta_trace_reduced_direction.noalias() +=
      t_value *
      (pair_kernel_seed_direction * sample_inverse_pp +
       context.pair_kernel_seed * sample_inverse_pp_direction);

  Matrix opposite_exchange_left_ordered_direction =
      Matrix::Zero(
          n_blocked_alpha + n_pair_orbitals,
          n_pair_orbitals);
  if (n_blocked_alpha > 0) {
    opposite_exchange_left_ordered_direction.topRows(n_blocked_alpha) =
        sample_inverse_op_direction;
  }
  opposite_exchange_left_ordered_direction.bottomRows(n_pair_orbitals) =
      t_value *
      context.right_pair_ab_reduced *
      sample_inverse_pp_direction;

  Matrix opposite_exchange_right_ordered_direction =
      Matrix::Zero(
          n_pair_orbitals,
          n_blocked_alpha + n_pair_orbitals);
  if (n_blocked_alpha > 0) {
    opposite_exchange_right_ordered_direction.leftCols(n_blocked_alpha).noalias() =
        alpha_to_block_factor_direction * sample_inverse.topLeftCorner(
            n_blocked_alpha,
            n_blocked_alpha) +
        context.alpha_to_block_factor * sample_inverse_oo_direction +
        t_value * pair_kernel_seed_direction * sample_inverse.bottomLeftCorner(
            n_pair_orbitals,
            n_blocked_alpha) +
        t_value * context.pair_kernel_seed * sample_inverse_po_direction;
  }
  opposite_exchange_right_ordered_direction.rightCols(n_pair_orbitals).noalias() =
      beta_trace_reduced_direction * context.beta_left_factor +
      operands.beta_trace_reduced * beta_left_factor_direction;

  Matrix alpha_trace_direction =
      scatter_submatrix(
          n_active_orbitals,
          context.left_ordered_orbitals,
          context.right_ordered_orbitals,
          alpha_trace_ordered_direction);
  Matrix beta_trace_direction =
      scatter_submatrix(
          n_active_orbitals,
          context.left_pair_orbitals,
          context.right_pair_orbitals,
          beta_trace_reduced_direction);
  Matrix opposite_exchange_left_direction =
      scatter_submatrix(
          n_active_orbitals,
          context.right_ordered_orbitals,
          context.right_pair_orbitals,
          opposite_exchange_left_ordered_direction);
  Matrix opposite_exchange_right_direction =
      scatter_submatrix(
          n_active_orbitals,
          context.left_pair_orbitals,
          context.left_ordered_orbitals,
          opposite_exchange_right_ordered_direction);
  Matrix opposite_mixed_right_direction =
      Matrix::Zero(n_active_orbitals, n_active_orbitals);

  const Matrix total_trace_matrix =
      operands.alpha_trace_matrix +
      operands.beta_trace_matrix;
  const Matrix total_trace_direction =
      alpha_trace_direction +
      beta_trace_direction;
  const double one_electron_factor =
      total_trace_matrix.cwiseProduct(one_electron).sum();
  const double one_electron_factor_direction =
      total_trace_direction.cwiseProduct(one_electron).sum();

  const double alpha_same_separable =
      xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_separable(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.alpha_trace_matrix,
          operands.alpha_trace_matrix);
  const double alpha_same_bridge =
      xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_bridge(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.alpha_trace_matrix,
          operands.alpha_trace_matrix);
  const double beta_same_separable =
      xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_separable(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.beta_trace_matrix,
          operands.beta_trace_matrix);
  const double beta_same_bridge =
      xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_bridge(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.beta_trace_matrix,
          operands.beta_trace_matrix);
  const double opposite_direct =
      xmvb::pfaffian_vbscf::PfTensorContractor::contract_direct(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.alpha_trace_matrix,
          operands.beta_trace_matrix);
  const double opposite_exchange =
      xmvb::pfaffian_vbscf::PfTensorContractor::contract_exchange(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.opposite_exchange_left_matrix,
          operands.opposite_exchange_right_matrix);
  const double opposite_mixed_exchange =
      xmvb::pfaffian_vbscf::PfTensorContractor::contract_exchange(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.opposite_exchange_left_matrix,
          operands.opposite_mixed_right_matrix);

  const double alpha_same_separable_direction =
      same_operand_contraction_directional_from_adjoints(
          packed_two_electron_integrals,
          operands.alpha_trace_matrix,
          alpha_trace_direction,
          &xmvb::pfaffian_vbscf::PfTensorContractor::compute_same_spin_separable_operand_adjoints);
  const double alpha_same_bridge_direction =
      same_operand_contraction_directional_from_adjoints(
          packed_two_electron_integrals,
          operands.alpha_trace_matrix,
          alpha_trace_direction,
          &xmvb::pfaffian_vbscf::PfTensorContractor::compute_same_spin_bridge_operand_adjoints);
  const double beta_same_separable_direction =
      same_operand_contraction_directional_from_adjoints(
          packed_two_electron_integrals,
          operands.beta_trace_matrix,
          beta_trace_direction,
          &xmvb::pfaffian_vbscf::PfTensorContractor::compute_same_spin_separable_operand_adjoints);
  const double beta_same_bridge_direction =
      same_operand_contraction_directional_from_adjoints(
          packed_two_electron_integrals,
          operands.beta_trace_matrix,
          beta_trace_direction,
          &xmvb::pfaffian_vbscf::PfTensorContractor::compute_same_spin_bridge_operand_adjoints);
  const double opposite_direct_direction =
      contraction_directional_from_adjoints(
          packed_two_electron_integrals,
          operands.alpha_trace_matrix,
          operands.beta_trace_matrix,
          alpha_trace_direction,
          beta_trace_direction,
          &xmvb::pfaffian_vbscf::PfTensorContractor::compute_direct_operand_adjoints);
  const double opposite_exchange_direction =
      contraction_directional_from_adjoints(
          packed_two_electron_integrals,
          operands.opposite_exchange_left_matrix,
          operands.opposite_exchange_right_matrix,
          opposite_exchange_left_direction,
          opposite_exchange_right_direction,
          &xmvb::pfaffian_vbscf::PfTensorContractor::compute_exchange_operand_adjoints);
  const double opposite_mixed_exchange_direction =
      contraction_directional_from_adjoints(
          packed_two_electron_integrals,
          operands.opposite_exchange_left_matrix,
          operands.opposite_mixed_right_matrix,
          opposite_exchange_left_direction,
          opposite_mixed_right_direction,
          &xmvb::pfaffian_vbscf::PfTensorContractor::compute_exchange_operand_adjoints);

  const double two_electron_factor =
      alpha_same_separable - alpha_same_bridge +
      beta_same_separable - beta_same_bridge +
      opposite_direct - opposite_exchange + opposite_mixed_exchange;
  const double two_electron_factor_direction =
      alpha_same_separable_direction - alpha_same_bridge_direction +
      beta_same_separable_direction - beta_same_bridge_direction +
      opposite_direct_direction - opposite_exchange_direction +
      opposite_mixed_exchange_direction;
  const double total_factor =
      one_electron_factor + two_electron_factor;
  const double total_factor_direction =
      one_electron_factor_direction + two_electron_factor_direction;
  return
      sample_weight *
      (determinant_direction * total_factor +
       operands.determinant * total_factor_direction);
}

double candidate_total_hamiltonian_spatial_directional_forward_mode(
    const Matrix& left_pair,
    const Matrix& right_pair,
    const std::vector<int>& left_blocked_alpha_orbitals,
    const std::vector<int>& right_blocked_alpha_orbitals,
    const Matrix& spatial_overlap,
    const Matrix& spatial_direction,
    const Matrix& one_electron,
    const ScalarBuffer& packed_two_electron_integrals,
    int n_singlet_pairs) {
  const CandidateContext context =
      build_candidate_context(
          left_pair.transpose(),
          right_pair,
          left_blocked_alpha_orbitals,
          right_blocked_alpha_orbitals,
          spatial_overlap,
          n_singlet_pairs);
  const Eigen::RowVectorXd weights =
      context.inverse_vandermonde.row(n_singlet_pairs);
  double value = 0.0;
  for (int node = 0; node <= context.interpolation_degree; ++node) {
    const double weight = weights(node);
    const double t_value =
        context.interpolation_nodes[xmvb::to_size(node)];
    const FastSampleOperands operands =
        build_fast_sample_operands(
            context,
            t_value);
    value +=
        sample_total_directional_derivative_forward_mode(
            context,
            operands,
            t_value,
            spatial_direction,
            one_electron,
            packed_two_electron_integrals,
            weight);
  }
  return value;
}

void backpropagate_sample_total(
    const CandidateContext& context,
    const FastSampleOperands& operands,
    double t_value,
    const Matrix& one_electron,
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
  const int n_blocked_alpha =
      static_cast<int>(context.left_blocked_alpha_orbitals.size());
  const int n_pair_orbitals =
      static_cast<int>(context.left_pair_orbitals.size());
  const Matrix total_trace_matrix =
      operands.alpha_trace_matrix +
      operands.beta_trace_matrix;

  const double one_electron_factor =
      total_trace_matrix.cwiseProduct(one_electron).sum();
  const double alpha_same_separable =
      xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_separable(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.alpha_trace_matrix,
          operands.alpha_trace_matrix);
  const double alpha_same_bridge =
      xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_bridge(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.alpha_trace_matrix,
          operands.alpha_trace_matrix);
  const double beta_same_separable =
      xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_separable(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.beta_trace_matrix,
          operands.beta_trace_matrix);
  const double beta_same_bridge =
      xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_bridge(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.beta_trace_matrix,
          operands.beta_trace_matrix);
  const double opposite_direct =
      xmvb::pfaffian_vbscf::PfTensorContractor::contract_direct(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.alpha_trace_matrix,
          operands.beta_trace_matrix);
  const double opposite_exchange =
      xmvb::pfaffian_vbscf::PfTensorContractor::contract_exchange(
          packed_two_electron_integrals,
          n_active_orbitals,
          operands.opposite_exchange_left_matrix,
          operands.opposite_exchange_right_matrix);
  const double opposite_mixed_exchange =
      xmvb::pfaffian_vbscf::PfTensorContractor::contract_exchange(
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

  xmvb::pfaffian_vbscf::PfTensorContractor::add_same_spin_separable_outer_product(
      scaled_determinant,
      operands.alpha_trace_matrix,
      operands.alpha_trace_matrix,
      packed_two_electron_gradient);
  xmvb::pfaffian_vbscf::PfTensorContractor::add_same_spin_bridge_outer_product(
      -scaled_determinant,
      operands.alpha_trace_matrix,
      operands.alpha_trace_matrix,
      packed_two_electron_gradient);
  xmvb::pfaffian_vbscf::PfTensorContractor::add_same_spin_separable_outer_product(
      scaled_determinant,
      operands.beta_trace_matrix,
      operands.beta_trace_matrix,
      packed_two_electron_gradient);
  xmvb::pfaffian_vbscf::PfTensorContractor::add_same_spin_bridge_outer_product(
      -scaled_determinant,
      operands.beta_trace_matrix,
      operands.beta_trace_matrix,
      packed_two_electron_gradient);
  xmvb::pfaffian_vbscf::PfTensorContractor::add_direct_outer_product(
      scaled_determinant,
      operands.alpha_trace_matrix,
      operands.beta_trace_matrix,
      packed_two_electron_gradient);
  xmvb::pfaffian_vbscf::PfTensorContractor::add_exchange_outer_product(
      -scaled_determinant,
      operands.opposite_exchange_left_matrix,
      operands.opposite_exchange_right_matrix,
      packed_two_electron_gradient);
  xmvb::pfaffian_vbscf::PfTensorContractor::add_exchange_outer_product(
      scaled_determinant,
      operands.opposite_exchange_left_matrix,
      operands.opposite_mixed_right_matrix,
      packed_two_electron_gradient);

  double determinant_adjoint =
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
      &xmvb::pfaffian_vbscf::PfTensorContractor::compute_same_spin_separable_operand_adjoints,
      &alpha_trace_adjoint);
  accumulate_same_operand_adjoint(
      -scaled_determinant,
      packed_two_electron_integrals,
      operands.alpha_trace_matrix,
      &xmvb::pfaffian_vbscf::PfTensorContractor::compute_same_spin_bridge_operand_adjoints,
      &alpha_trace_adjoint);
  accumulate_same_operand_adjoint(
      scaled_determinant,
      packed_two_electron_integrals,
      operands.beta_trace_matrix,
      &xmvb::pfaffian_vbscf::PfTensorContractor::compute_same_spin_separable_operand_adjoints,
      &beta_trace_adjoint);
  accumulate_same_operand_adjoint(
      -scaled_determinant,
      packed_two_electron_integrals,
      operands.beta_trace_matrix,
      &xmvb::pfaffian_vbscf::PfTensorContractor::compute_same_spin_bridge_operand_adjoints,
      &beta_trace_adjoint);
  xmvb::pfaffian_vbscf::PfTensorContractor::compute_direct_operand_adjoints(
      scaled_determinant,
      operands.alpha_trace_matrix,
      operands.beta_trace_matrix,
      packed_two_electron_integrals,
      alpha_trace_adjoint,
      beta_trace_adjoint);
  constexpr bool kDebugDisableOppositeExchangeAdjoints = false;
  if (!kDebugDisableOppositeExchangeAdjoints) {
    xmvb::pfaffian_vbscf::PfTensorContractor::compute_exchange_operand_adjoints(
        -scaled_determinant,
        operands.opposite_exchange_left_matrix,
        operands.opposite_exchange_right_matrix,
        packed_two_electron_integrals,
        opposite_exchange_left_adjoint,
        opposite_exchange_right_adjoint);
    xmvb::pfaffian_vbscf::PfTensorContractor::compute_exchange_operand_adjoints(
        scaled_determinant,
        operands.opposite_exchange_left_matrix,
        operands.opposite_mixed_right_matrix,
        packed_two_electron_integrals,
        opposite_exchange_left_adjoint,
        opposite_mixed_right_adjoint);
  }

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
  return;

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
      determinant_adjoint *
      operands.determinant *
      sample_inverse.transpose();
  sample_matrix_adjoint.noalias() -=
      sample_inverse.transpose() *
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

  accumulate_submatrix(
      total_spatial_gradient,
      context.left_blocked_alpha_orbitals,
      context.right_blocked_alpha_orbitals,
      blocked_overlap_alpha_adjoint);
  accumulate_submatrix(
      total_spatial_gradient,
      context.left_blocked_alpha_orbitals,
      context.right_pair_orbitals,
      blocked_to_right_pair_overlap_alpha_adjoint);
  accumulate_submatrix(
      total_spatial_gradient,
      context.left_pair_orbitals,
      context.right_blocked_alpha_orbitals,
      left_pair_to_block_overlap_alpha_adjoint);
  accumulate_submatrix(
      total_spatial_gradient,
      context.left_pair_orbitals,
      context.right_pair_orbitals,
      pair_overlap_alpha_adjoint + pair_overlap_beta_adjoint);
}

Matrix backpropagate_exchange_pullback_only(
    const CandidateContext& context,
    const FastSampleOperands& operands,
    double t_value,
    const Matrix& opposite_exchange_left_adjoint,
    const Matrix& opposite_exchange_right_adjoint,
    const Matrix& opposite_mixed_right_adjoint) {
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

Matrix backpropagate_trace_pullback_only(
    const CandidateContext& context,
    const FastSampleOperands& operands,
    double t_value,
    const Matrix& alpha_trace_adjoint,
    const Matrix& beta_trace_adjoint) {
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

CandidateResult evaluate_candidate(
    const Matrix& left_pair_ab,
    const Matrix& right_pair_ab,
    const std::vector<int>& left_blocked_alpha_orbitals,
    const std::vector<int>& right_blocked_alpha_orbitals,
    const Matrix& spatial_overlap,
    const Matrix& one_electron,
    const std::vector<double>& packed_two_electron_integrals,
    int n_singlet_pairs) {
  const CandidateContext context =
      build_candidate_context(
          left_pair_ab.transpose(),
          right_pair_ab,
          left_blocked_alpha_orbitals,
          right_blocked_alpha_orbitals,
          spatial_overlap,
          n_singlet_pairs);
  const int n_active_orbitals = context.n_active_orbitals;
  const Eigen::RowVectorXd weights =
      context.inverse_vandermonde.row(n_singlet_pairs);

  CandidateResult result;
  result.interpolation_degree = context.interpolation_degree;
  result.total_hamiltonian_spatial_gradient =
      Matrix::Zero(n_active_orbitals, n_active_orbitals);
  result.one_electron_matrix_gradient =
      Matrix::Zero(n_active_orbitals, n_active_orbitals);
  result.packed_two_electron_gradient.assign(
      xmvb::pfaffian_vbscf::PfTensorContractor::packed_two_electron_count(
          n_active_orbitals),
      0.0);

  for (int node = 0; node <= context.interpolation_degree; ++node) {
    const double weight = weights(node);
    const double t_value =
        context.interpolation_nodes[xmvb::to_size(node)];
    const FastSampleOperands operands =
        build_fast_sample_operands(context, t_value);

    const Matrix total_trace_matrix =
        operands.alpha_trace_matrix +
        operands.beta_trace_matrix;
    const double sample_one_electron =
        operands.determinant *
        total_trace_matrix.cwiseProduct(one_electron).sum();

    double sample_two_electron = 0.0;
    sample_two_electron +=
        operands.determinant *
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_separable(
            packed_two_electron_integrals,
            n_active_orbitals,
            operands.alpha_trace_matrix,
            operands.alpha_trace_matrix);
    sample_two_electron -=
        operands.determinant *
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_bridge(
            packed_two_electron_integrals,
            n_active_orbitals,
            operands.alpha_trace_matrix,
            operands.alpha_trace_matrix);
    sample_two_electron +=
        operands.determinant *
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_separable(
            packed_two_electron_integrals,
            n_active_orbitals,
            operands.beta_trace_matrix,
            operands.beta_trace_matrix);
    sample_two_electron -=
        operands.determinant *
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_bridge(
            packed_two_electron_integrals,
            n_active_orbitals,
            operands.beta_trace_matrix,
            operands.beta_trace_matrix);
    sample_two_electron +=
        operands.determinant *
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_direct(
            packed_two_electron_integrals,
            n_active_orbitals,
            operands.alpha_trace_matrix,
            operands.beta_trace_matrix);
    sample_two_electron -=
        operands.determinant *
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_exchange(
            packed_two_electron_integrals,
            n_active_orbitals,
            operands.opposite_exchange_left_matrix,
            operands.opposite_exchange_right_matrix);
    sample_two_electron +=
        operands.determinant *
        xmvb::pfaffian_vbscf::PfTensorContractor::contract_exchange(
            packed_two_electron_integrals,
            n_active_orbitals,
            operands.opposite_exchange_left_matrix,
            operands.opposite_mixed_right_matrix);

    result.overlap += weight * operands.determinant;
    result.one_electron_hamiltonian += weight * sample_one_electron;
    result.total_hamiltonian +=
        weight * (sample_one_electron + sample_two_electron);
    backpropagate_sample_total(
        context,
        operands,
        t_value,
        one_electron,
        packed_two_electron_integrals,
        weight,
        &result.total_hamiltonian_spatial_gradient,
        &result.one_electron_matrix_gradient,
        &result.packed_two_electron_gradient);
  }

  return result;
}

double max_abs(double value, double candidate) {
  return std::max(value, std::abs(candidate));
}

double max_abs_diff(
    const Matrix& left,
    const Matrix& right) {
  if (left.rows() != right.rows() || left.cols() != right.cols()) {
    throw std::invalid_argument("matrix dimensions must match for max_abs_diff");
  }
  double max_error = 0.0;
  for (int col = 0; col < left.cols(); ++col) {
    for (int row = 0; row < left.rows(); ++row) {
      max_error =
          std::max(
              max_error,
              std::abs(left(row, col) - right(row, col)));
    }
  }
  return max_error;
}

double max_abs_diff(
    const ScalarBuffer& left,
    const ScalarBuffer& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("buffer sizes must match for max_abs_diff");
  }
  double max_error = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    max_error =
        std::max(
            max_error,
            std::abs(left[index] - right[index]));
  }
  return max_error;
}

double matrix_dot(
    const Matrix& left,
    const Matrix& right) {
  if (left.rows() != right.rows() || left.cols() != right.cols()) {
    throw std::invalid_argument("matrix dimensions must match for matrix_dot");
  }
  return left.cwiseProduct(right).sum();
}

double buffer_dot(
    const ScalarBuffer& left,
    const ScalarBuffer& right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("buffer sizes must match for buffer_dot");
  }
  double value = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    value += left[index] * right[index];
  }
  return value;
}

double contraction_directional_fd(
    double (*contract_function)(
        const ScalarBuffer&,
        int,
        const ConstMatrixRef&,
        const ConstMatrixRef&),
    const ScalarBuffer& ggo,
    const Matrix& left,
    const Matrix& right,
    const Matrix& left_direction,
    const Matrix& right_direction,
    double step) {
  return
      (contract_function(
           ggo,
           static_cast<int>(left.rows()),
           left + step * left_direction,
           right + step * right_direction) -
       contract_function(
           ggo,
           static_cast<int>(left.rows()),
           left - step * left_direction,
           right - step * right_direction)) /
      (2.0 * step);
}

void run_tensor_operand_adjoint_diagnostics() {
  std::mt19937 generator(20260329);
  std::normal_distribution<double> distribution(0.0, 1.0);
  const int n = 5;
  const ScalarBuffer ggo =
      random_packed_two_electron_integrals(
          n,
          &generator,
          &distribution);
  auto random_matrix = [&]() {
    Matrix matrix = Matrix::Zero(n, n);
    for (int col = 0; col < n; ++col) {
      for (int row = 0; row < n; ++row) {
        matrix(row, col) = distribution(generator);
      }
    }
    return matrix;
  };
  auto normalized = [](Matrix matrix) {
    const double norm = matrix.norm();
    if (norm > 0.0) {
      matrix /= norm;
    }
    return matrix;
  };

  const Matrix left = random_matrix();
  const Matrix right = random_matrix();
  const Matrix left_direction = normalized(random_matrix());
  const Matrix right_direction = normalized(random_matrix());
  const double step = 1.0e-6;

  auto print_check =
      [&](const char* label,
          double (*contract_function)(
              const ScalarBuffer&,
              int,
              const ConstMatrixRef&,
              const ConstMatrixRef&),
          void (*adjoint_function)(
              double,
              const ConstMatrixRef&,
              const ConstMatrixRef&,
              const ScalarBuffer&,
              MatrixRef,
              MatrixRef)) {
        Matrix left_bar = Matrix::Zero(n, n);
        Matrix right_bar = Matrix::Zero(n, n);
        adjoint_function(
            1.0,
            left,
            right,
            ggo,
            left_bar,
            right_bar);
        const double analytic =
            matrix_dot(left_bar, left_direction) +
            matrix_dot(right_bar, right_direction);
        const double finite_difference =
            contraction_directional_fd(
                contract_function,
                ggo,
                left,
                right,
                left_direction,
                right_direction,
                step);
        std::cout
            << "tensor_debug " << label
            << " analytic=" << analytic
            << " fd=" << finite_difference
            << " err=" << std::abs(analytic - finite_difference)
            << '\n';
      };

  print_check(
      "direct",
      &xmvb::pfaffian_vbscf::PfTensorContractor::contract_direct,
      &xmvb::pfaffian_vbscf::PfTensorContractor::compute_direct_operand_adjoints);
  print_check(
      "exchange",
      &xmvb::pfaffian_vbscf::PfTensorContractor::contract_exchange,
      &xmvb::pfaffian_vbscf::PfTensorContractor::compute_exchange_operand_adjoints);
  print_check(
      "same_sep",
      &xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_separable,
      &xmvb::pfaffian_vbscf::PfTensorContractor::compute_same_spin_separable_operand_adjoints);
  print_check(
      "same_bridge",
      &xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_bridge,
      &xmvb::pfaffian_vbscf::PfTensorContractor::compute_same_spin_bridge_operand_adjoints);

  auto print_same_operand_check =
      [&](const char* label,
          double (*contract_function)(
              const ScalarBuffer&,
              int,
              const ConstMatrixRef&,
              const ConstMatrixRef&),
          void (*adjoint_function)(
              double,
              const ConstMatrixRef&,
              const ConstMatrixRef&,
              const ScalarBuffer&,
              MatrixRef,
              MatrixRef)) {
        Matrix left_bar = Matrix::Zero(n, n);
        Matrix right_bar = Matrix::Zero(n, n);
        adjoint_function(
            1.0,
            left,
            left,
            ggo,
            left_bar,
            right_bar);
        const double analytic =
            matrix_dot(left_bar + right_bar, left_direction);
        const double finite_difference =
            (contract_function(
                 ggo,
                 n,
                 left + step * left_direction,
                 left + step * left_direction) -
             contract_function(
                 ggo,
                 n,
                 left - step * left_direction,
                 left - step * left_direction)) /
            (2.0 * step);
        std::cout
            << "tensor_same_debug " << label
            << " analytic=" << analytic
            << " fd=" << finite_difference
            << " err=" << std::abs(analytic - finite_difference)
            << '\n';
      };
  print_same_operand_check(
      "same_sep",
      &xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_separable,
      &xmvb::pfaffian_vbscf::PfTensorContractor::compute_same_spin_separable_operand_adjoints);
  print_same_operand_check(
      "same_bridge",
      &xmvb::pfaffian_vbscf::PfTensorContractor::contract_same_spin_bridge,
      &xmvb::pfaffian_vbscf::PfTensorContractor::compute_same_spin_bridge_operand_adjoints);
}

void run_exchange_pullback_diagnostic() {
  std::mt19937 generator(20260330);
  std::normal_distribution<double> distribution(0.0, 1.0);
  const int n_active_orbitals = 5;
  const int n_singlet_pairs = 2;
  const std::vector<int> left_blocked_alpha_orbitals;
  const std::vector<int> right_blocked_alpha_orbitals;
  const Matrix spatial_overlap =
      random_spd_matrix(
          n_active_orbitals,
          &generator,
          &distribution);
  const Matrix left_pair =
      random_pair_matrix(
          n_active_orbitals,
          left_blocked_alpha_orbitals,
          &generator,
          &distribution);
  const Matrix right_pair =
      random_pair_matrix(
          n_active_orbitals,
          right_blocked_alpha_orbitals,
          &generator,
          &distribution);
  const CandidateContext context =
      build_candidate_context(
          left_pair.transpose(),
          right_pair,
          left_blocked_alpha_orbitals,
          right_blocked_alpha_orbitals,
          spatial_overlap,
          n_singlet_pairs);
  const double t_value =
      context.interpolation_nodes[1];
  const FastSampleOperands operands =
      build_fast_sample_operands(
          context,
          t_value);

  auto random_matrix = [&]() {
    Matrix matrix = Matrix::Zero(n_active_orbitals, n_active_orbitals);
    for (int col = 0; col < n_active_orbitals; ++col) {
      for (int row = 0; row < n_active_orbitals; ++row) {
        matrix(row, col) = distribution(generator);
      }
    }
    const double norm = matrix.norm();
    if (norm > 0.0) {
      matrix /= norm;
    }
    return matrix;
  };
  const Matrix left_adjoint = random_matrix();
  const Matrix right_adjoint = random_matrix();
  const Matrix mixed_adjoint = random_matrix();
  const Matrix spatial_direction = random_matrix();

  const Matrix analytic_gradient =
      backpropagate_exchange_pullback_only(
          context,
          operands,
          t_value,
          left_adjoint,
          right_adjoint,
          mixed_adjoint);
  const double analytic_directional =
      matrix_dot(
          analytic_gradient,
          spatial_direction);

  auto pullback_value = [&](const Matrix& current_spatial_overlap) {
    const CandidateContext current_context =
        build_candidate_context(
            left_pair.transpose(),
            right_pair,
            left_blocked_alpha_orbitals,
            right_blocked_alpha_orbitals,
            current_spatial_overlap,
            n_singlet_pairs);
    const FastSampleOperands current_operands =
        build_fast_sample_operands(
            current_context,
            t_value);
    return
        matrix_dot(
            current_operands.opposite_exchange_left_matrix,
            left_adjoint) +
        matrix_dot(
            current_operands.opposite_exchange_right_matrix,
            right_adjoint) +
        matrix_dot(
            current_operands.opposite_mixed_right_matrix,
            mixed_adjoint);
  };

  const double step = 1.0e-6;
  const double finite_difference =
      (pullback_value(spatial_overlap + step * spatial_direction) -
       pullback_value(spatial_overlap - step * spatial_direction)) /
      (2.0 * step);
  std::cout
      << "exchange_pullback_debug analytic=" << analytic_directional
      << " fd=" << finite_difference
      << " err=" << std::abs(analytic_directional - finite_difference)
      << '\n';
}

void run_trace_pullback_diagnostic() {
  std::mt19937 generator(20260331);
  std::normal_distribution<double> distribution(0.0, 1.0);
  const int n_active_orbitals = 5;
  const int n_singlet_pairs = 2;
  const std::vector<int> left_blocked_alpha_orbitals;
  const std::vector<int> right_blocked_alpha_orbitals;
  const Matrix spatial_overlap =
      random_spd_matrix(
          n_active_orbitals,
          &generator,
          &distribution);
  const Matrix left_pair =
      random_pair_matrix(
          n_active_orbitals,
          left_blocked_alpha_orbitals,
          &generator,
          &distribution);
  const Matrix right_pair =
      random_pair_matrix(
          n_active_orbitals,
          right_blocked_alpha_orbitals,
          &generator,
          &distribution);
  const CandidateContext context =
      build_candidate_context(
          left_pair.transpose(),
          right_pair,
          left_blocked_alpha_orbitals,
          right_blocked_alpha_orbitals,
          spatial_overlap,
          n_singlet_pairs);
  const double t_value =
      context.interpolation_nodes[1];
  const FastSampleOperands operands =
      build_fast_sample_operands(
          context,
          t_value);

  auto random_matrix = [&]() {
    Matrix matrix = Matrix::Zero(n_active_orbitals, n_active_orbitals);
    for (int col = 0; col < n_active_orbitals; ++col) {
      for (int row = 0; row < n_active_orbitals; ++row) {
        matrix(row, col) = distribution(generator);
      }
    }
    const double norm = matrix.norm();
    if (norm > 0.0) {
      matrix /= norm;
    }
    return matrix;
  };
  const Matrix alpha_adjoint = random_matrix();
  const Matrix beta_adjoint = random_matrix();
  const Matrix spatial_direction = random_matrix();

  const Matrix analytic_gradient =
      backpropagate_trace_pullback_only(
          context,
          operands,
          t_value,
          alpha_adjoint,
          beta_adjoint);
  const double analytic_directional =
      matrix_dot(
          analytic_gradient,
          spatial_direction);

  auto pullback_value = [&](const Matrix& current_spatial_overlap) {
    const CandidateContext current_context =
        build_candidate_context(
            left_pair.transpose(),
            right_pair,
            left_blocked_alpha_orbitals,
            right_blocked_alpha_orbitals,
            current_spatial_overlap,
            n_singlet_pairs);
    const FastSampleOperands current_operands =
        build_fast_sample_operands(
            current_context,
            t_value);
    return
        matrix_dot(
            current_operands.alpha_trace_matrix,
            alpha_adjoint) +
        matrix_dot(
            current_operands.beta_trace_matrix,
            beta_adjoint);
  };

  const double step = 1.0e-6;
  const double finite_difference =
      (pullback_value(spatial_overlap + step * spatial_direction) -
       pullback_value(spatial_overlap - step * spatial_direction)) /
      (2.0 * step);
  std::cout
      << "trace_pullback_debug analytic=" << analytic_directional
      << " fd=" << finite_difference
      << " err=" << std::abs(analytic_directional - finite_difference)
      << '\n';
}

double exact_total_hamiltonian(
    const std::vector<DeterminantTerm>& left_terms,
    const std::vector<DeterminantTerm>& right_terms,
    const Matrix& spatial_overlap,
    const Matrix& one_electron,
    const ScalarBuffer& packed_two_electron_integrals) {
  return
      exact_state_pair_evaluation(
          left_terms,
          right_terms,
          spatial_overlap,
          one_electron,
          packed_two_electron_integrals)
          .total_hamiltonian;
}

Matrix exact_total_hamiltonian_spatial_gradient_fd(
    const std::vector<DeterminantTerm>& left_terms,
    const std::vector<DeterminantTerm>& right_terms,
    const Matrix& spatial_overlap,
    const Matrix& one_electron,
    const ScalarBuffer& packed_two_electron_integrals,
    double step) {
  Matrix gradient =
      Matrix::Zero(
          spatial_overlap.rows(),
          spatial_overlap.cols());
  for (int col = 0; col < spatial_overlap.cols(); ++col) {
    for (int row = 0; row < spatial_overlap.rows(); ++row) {
      Matrix plus = spatial_overlap;
      Matrix minus = spatial_overlap;
      plus(row, col) += step;
      minus(row, col) -= step;
      const double plus_value =
          exact_total_hamiltonian(
              left_terms,
              right_terms,
              plus,
              one_electron,
              packed_two_electron_integrals);
      const double minus_value =
          exact_total_hamiltonian(
              left_terms,
              right_terms,
              minus,
              one_electron,
              packed_two_electron_integrals);
      gradient(row, col) =
          (plus_value - minus_value) / (2.0 * step);
    }
  }
  return gradient;
}

ScalarBuffer exact_total_hamiltonian_ggo_gradient_fd(
    const std::vector<DeterminantTerm>& left_terms,
    const std::vector<DeterminantTerm>& right_terms,
    const Matrix& spatial_overlap,
    const Matrix& one_electron,
    const ScalarBuffer& packed_two_electron_integrals,
    double step) {
  ScalarBuffer gradient(
      packed_two_electron_integrals.size(),
      0.0);
  for (std::size_t index = 0; index < packed_two_electron_integrals.size(); ++index) {
    ScalarBuffer plus = packed_two_electron_integrals;
    ScalarBuffer minus = packed_two_electron_integrals;
    plus[index] += step;
    minus[index] -= step;
    const double plus_value =
        exact_total_hamiltonian(
            left_terms,
            right_terms,
            spatial_overlap,
            one_electron,
            plus);
    const double minus_value =
        exact_total_hamiltonian(
            left_terms,
            right_terms,
            spatial_overlap,
            one_electron,
            minus);
    gradient[index] =
        (plus_value - minus_value) / (2.0 * step);
  }
  return gradient;
}

double candidate_total_hamiltonian(
    const Matrix& left_pair,
    const Matrix& right_pair,
    const std::vector<int>& left_blocked_alpha_orbitals,
    const std::vector<int>& right_blocked_alpha_orbitals,
    const Matrix& spatial_overlap,
    const Matrix& one_electron,
    const ScalarBuffer& packed_two_electron_integrals,
    int n_singlet_pairs) {
  return
      evaluate_candidate(
          left_pair,
          right_pair,
          left_blocked_alpha_orbitals,
          right_blocked_alpha_orbitals,
          spatial_overlap,
          one_electron,
          packed_two_electron_integrals,
          n_singlet_pairs)
          .total_hamiltonian;
}

Matrix candidate_total_hamiltonian_spatial_gradient_fd(
    const Matrix& left_pair,
    const Matrix& right_pair,
    const std::vector<int>& left_blocked_alpha_orbitals,
    const std::vector<int>& right_blocked_alpha_orbitals,
    const Matrix& spatial_overlap,
    const Matrix& one_electron,
    const ScalarBuffer& packed_two_electron_integrals,
    int n_singlet_pairs,
    double step) {
  Matrix gradient =
      Matrix::Zero(
          spatial_overlap.rows(),
          spatial_overlap.cols());
  for (int col = 0; col < spatial_overlap.cols(); ++col) {
    for (int row = 0; row < spatial_overlap.rows(); ++row) {
      Matrix plus = spatial_overlap;
      Matrix minus = spatial_overlap;
      plus(row, col) += step;
      minus(row, col) -= step;
      const double plus_value =
          candidate_total_hamiltonian(
              left_pair,
              right_pair,
              left_blocked_alpha_orbitals,
              right_blocked_alpha_orbitals,
              plus,
              one_electron,
              packed_two_electron_integrals,
              n_singlet_pairs);
      const double minus_value =
          candidate_total_hamiltonian(
              left_pair,
              right_pair,
              left_blocked_alpha_orbitals,
              right_blocked_alpha_orbitals,
              minus,
              one_electron,
              packed_two_electron_integrals,
              n_singlet_pairs);
      gradient(row, col) =
          (plus_value - minus_value) / (2.0 * step);
    }
  }
  return gradient;
}

ScalarBuffer candidate_total_hamiltonian_ggo_gradient_fd(
    const Matrix& left_pair,
    const Matrix& right_pair,
    const std::vector<int>& left_blocked_alpha_orbitals,
    const std::vector<int>& right_blocked_alpha_orbitals,
    const Matrix& spatial_overlap,
    const Matrix& one_electron,
    const ScalarBuffer& packed_two_electron_integrals,
    int n_singlet_pairs,
    double step) {
  ScalarBuffer gradient(
      packed_two_electron_integrals.size(),
      0.0);
  for (std::size_t index = 0; index < packed_two_electron_integrals.size(); ++index) {
    ScalarBuffer plus = packed_two_electron_integrals;
    ScalarBuffer minus = packed_two_electron_integrals;
    plus[index] += step;
    minus[index] -= step;
    const double plus_value =
        candidate_total_hamiltonian(
            left_pair,
            right_pair,
            left_blocked_alpha_orbitals,
            right_blocked_alpha_orbitals,
            spatial_overlap,
            one_electron,
            plus,
            n_singlet_pairs);
    const double minus_value =
        candidate_total_hamiltonian(
            left_pair,
            right_pair,
            left_blocked_alpha_orbitals,
            right_blocked_alpha_orbitals,
            spatial_overlap,
            one_electron,
            minus,
            n_singlet_pairs);
    gradient[index] =
        (plus_value - minus_value) / (2.0 * step);
  }
  return gradient;
}

void validate_production_pair_helper(
    const Matrix& left_pair,
    const Matrix& right_pair,
    const std::vector<int>& left_blocked_alpha_orbitals,
    const std::vector<int>& right_blocked_alpha_orbitals,
    const Matrix& spatial_overlap,
    const Matrix& one_electron,
    const ScalarBuffer& packed_two_electron_integrals,
    int n_singlet_pairs,
    const CandidateResult& candidate) {
  const auto production_no_grad =
      xmvb::pfaffian_vbscf::evaluate_high_spin_open_shell_pair_hamiltonian(
          left_pair.transpose(),
          right_pair,
          left_blocked_alpha_orbitals,
          right_blocked_alpha_orbitals,
          spatial_overlap,
          one_electron,
          packed_two_electron_integrals,
          n_singlet_pairs,
          false);
  const auto production =
      xmvb::pfaffian_vbscf::evaluate_high_spin_open_shell_pair_hamiltonian(
          left_pair.transpose(),
          right_pair,
          left_blocked_alpha_orbitals,
          right_blocked_alpha_orbitals,
          spatial_overlap,
          one_electron,
          packed_two_electron_integrals,
          n_singlet_pairs,
          true);
  constexpr double kProductionTolerance = 1.0e-6;
  if (!production.gradients_computed) {
    throw std::runtime_error(
        "production high-spin open-shell pair helper did not return gradients");
  }
  if (std::abs(production_no_grad.overlap - candidate.overlap) > kProductionTolerance ||
      std::abs(
          production_no_grad.one_electron_hamiltonian -
          candidate.one_electron_hamiltonian) > kProductionTolerance ||
      std::abs(
          production_no_grad.total_hamiltonian - candidate.total_hamiltonian) >
          kProductionTolerance) {
    std::ostringstream stream;
    stream << std::setprecision(16)
           << "production high-spin open-shell pair helper forward-only path deviates: "
           << "overlap_no_grad=" << production_no_grad.overlap
           << ", one_electron_no_grad=" << production_no_grad.one_electron_hamiltonian
           << ", total_no_grad=" << production_no_grad.total_hamiltonian
           << ", candidate_overlap=" << candidate.overlap
           << ", candidate_one_electron=" << candidate.one_electron_hamiltonian
           << ", candidate_total=" << candidate.total_hamiltonian;
    throw std::runtime_error(stream.str());
  }
  if (std::abs(production.overlap - candidate.overlap) > kProductionTolerance ||
      std::abs(
          production.one_electron_hamiltonian - candidate.one_electron_hamiltonian) >
          kProductionTolerance ||
      std::abs(production.total_hamiltonian - candidate.total_hamiltonian) >
          kProductionTolerance) {
    throw std::runtime_error(
        "production high-spin open-shell pair helper forward values deviate from the validated prototype");
  }
  if (max_abs_diff(
          production.total_hamiltonian_spatial_gradient,
          candidate.total_hamiltonian_spatial_gradient) > kProductionTolerance ||
      max_abs_diff(
          production.one_electron_matrix_gradient,
          candidate.one_electron_matrix_gradient) > kProductionTolerance ||
      max_abs_diff(
          production.packed_two_electron_gradient,
          candidate.packed_two_electron_gradient) > kProductionTolerance) {
    std::ostringstream stream;
    stream << std::setprecision(16)
           << "production high-spin open-shell pair helper deviates from the validated prototype: "
           << "d_total_spatial="
           << max_abs_diff(
                  production.total_hamiltonian_spatial_gradient,
                  candidate.total_hamiltonian_spatial_gradient)
           << ", d_h1="
           << max_abs_diff(
                  production.one_electron_matrix_gradient,
                  candidate.one_electron_matrix_gradient)
           << ", d_ggo="
           << max_abs_diff(
                  production.packed_two_electron_gradient,
                  candidate.packed_two_electron_gradient);
    throw std::runtime_error(stream.str());
  }
}

void run_case(
    int n_active_orbitals,
    int n_singlet_pairs,
    const std::vector<int>& left_blocked_alpha_orbitals,
    const std::vector<int>& right_blocked_alpha_orbitals,
    std::mt19937* generator,
    std::normal_distribution<double>* distribution,
    double* max_overlap_error,
    double* max_one_electron_error,
    double* max_total_error,
    double* max_two_electron_error,
    double* max_spatial_gradient_error,
    double* max_ggo_gradient_error,
    bool check_gradients) {
  if (generator == nullptr || distribution == nullptr ||
      max_overlap_error == nullptr || max_one_electron_error == nullptr ||
      max_total_error == nullptr || max_two_electron_error == nullptr) {
    throw std::invalid_argument("run_case buffers must not be null");
  }

  const Matrix spatial_overlap =
      random_spd_matrix(n_active_orbitals, generator, distribution);
  const Matrix one_electron =
      random_spd_matrix(n_active_orbitals, generator, distribution);
  const std::vector<double> packed_two_electron_integrals =
      random_packed_two_electron_integrals(
          n_active_orbitals,
          generator,
          distribution);
  const Matrix left_pair =
      random_pair_matrix(
          n_active_orbitals,
          left_blocked_alpha_orbitals,
          generator,
          distribution);
  const Matrix right_pair =
      random_pair_matrix(
          n_active_orbitals,
          right_blocked_alpha_orbitals,
          generator,
          distribution);

  const auto left_terms =
      enumerate_state_terms(
          left_pair,
          left_blocked_alpha_orbitals,
          n_singlet_pairs);
  const auto right_terms =
      enumerate_state_terms(
          right_pair,
          right_blocked_alpha_orbitals,
          n_singlet_pairs);
  const ExactStatePairResult exact =
      exact_state_pair_evaluation(
          left_terms,
          right_terms,
          spatial_overlap,
          one_electron,
          packed_two_electron_integrals);
  const CandidateResult candidate =
      evaluate_candidate(
          left_pair,
          right_pair,
          left_blocked_alpha_orbitals,
          right_blocked_alpha_orbitals,
          spatial_overlap,
          one_electron,
          packed_two_electron_integrals,
          n_singlet_pairs);
  validate_production_pair_helper(
      left_pair,
      right_pair,
      left_blocked_alpha_orbitals,
      right_blocked_alpha_orbitals,
      spatial_overlap,
      one_electron,
      packed_two_electron_integrals,
      n_singlet_pairs,
      candidate);

  *max_overlap_error =
      max_abs(*max_overlap_error, candidate.overlap - exact.overlap);
  *max_one_electron_error =
      max_abs(
          *max_one_electron_error,
          candidate.one_electron_hamiltonian - exact.one_electron_hamiltonian);
  *max_total_error =
      max_abs(
          *max_total_error,
          candidate.total_hamiltonian - exact.total_hamiltonian);
  *max_two_electron_error =
      max_abs(
          *max_two_electron_error,
          (candidate.total_hamiltonian - candidate.one_electron_hamiltonian) -
              (exact.total_hamiltonian - exact.one_electron_hamiltonian));

  if (check_gradients) {
    if (max_spatial_gradient_error == nullptr ||
        max_ggo_gradient_error == nullptr) {
      throw std::invalid_argument("gradient error buffers must not be null");
    }
    for (int trial = 0; trial < 3; ++trial) {
      Matrix spatial_direction =
          Matrix::Zero(n_active_orbitals, n_active_orbitals);
      for (int col = 0; col < n_active_orbitals; ++col) {
        for (int row = 0; row < n_active_orbitals; ++row) {
          spatial_direction(row, col) = (*distribution)(*generator);
        }
      }
      const double spatial_norm = spatial_direction.norm();
      if (spatial_norm > 0.0) {
        spatial_direction /= spatial_norm;
      }
      const double exact_spatial_directional =
          candidate_total_hamiltonian_spatial_directional_forward_mode(
              left_pair,
              right_pair,
              left_blocked_alpha_orbitals,
              right_blocked_alpha_orbitals,
              spatial_overlap,
              spatial_direction,
              one_electron,
              packed_two_electron_integrals,
              n_singlet_pairs);
      const double candidate_spatial_directional =
          matrix_dot(
              candidate.total_hamiltonian_spatial_gradient,
              spatial_direction);
      *max_spatial_gradient_error =
          std::max(
              *max_spatial_gradient_error,
              std::abs(
                  candidate_spatial_directional -
                  exact_spatial_directional));

      ScalarBuffer ggo_direction(
          packed_two_electron_integrals.size(),
          0.0);
      double ggo_norm_sq = 0.0;
      for (double& value : ggo_direction) {
        value = (*distribution)(*generator);
        ggo_norm_sq += value * value;
      }
      const double ggo_norm = std::sqrt(ggo_norm_sq);
      if (ggo_norm > 0.0) {
        for (double& value : ggo_direction) {
          value /= ggo_norm;
        }
      }
      const ScalarBuffer zero_ggo(
          packed_two_electron_integrals.size(),
          0.0);
      const double exact_ggo_directional =
          candidate_total_hamiltonian(
              left_pair,
              right_pair,
              left_blocked_alpha_orbitals,
              right_blocked_alpha_orbitals,
              spatial_overlap,
              one_electron,
              ggo_direction,
              n_singlet_pairs) -
          candidate_total_hamiltonian(
              left_pair,
              right_pair,
              left_blocked_alpha_orbitals,
              right_blocked_alpha_orbitals,
              spatial_overlap,
              one_electron,
              zero_ggo,
              n_singlet_pairs);
      const double candidate_ggo_directional =
          buffer_dot(
              candidate.packed_two_electron_gradient,
              ggo_direction);
      *max_ggo_gradient_error =
          std::max(
              *max_ggo_gradient_error,
              std::abs(
                  candidate_ggo_directional -
                  exact_ggo_directional));
    }
  }
}

}  // namespace

int main() {
  try {
    std::mt19937 generator(20260329);
    std::normal_distribution<double> distribution(0.0, 1.0);

    double max_overlap_error = 0.0;
    double max_one_electron_error = 0.0;
    double max_total_error = 0.0;
    double max_two_electron_error = 0.0;
    double max_spatial_gradient_error = 0.0;
    double max_ggo_gradient_error = 0.0;

    for (int repeat = 0; repeat < 4; ++repeat) {
      run_case(
          4,
          1,
          {},
          {},
          &generator,
          &distribution,
          &max_overlap_error,
          &max_one_electron_error,
          &max_total_error,
          &max_two_electron_error,
          &max_spatial_gradient_error,
          &max_ggo_gradient_error,
          repeat == 0);
      run_case(
          4,
          1,
          {0},
          {1},
          &generator,
          &distribution,
          &max_overlap_error,
          &max_one_electron_error,
          &max_total_error,
          &max_two_electron_error,
          &max_spatial_gradient_error,
          &max_ggo_gradient_error,
          repeat == 0);
      run_case(
          5,
          2,
          {},
          {},
          &generator,
          &distribution,
          &max_overlap_error,
          &max_one_electron_error,
          &max_total_error,
          &max_two_electron_error,
          &max_spatial_gradient_error,
          &max_ggo_gradient_error,
          repeat == 0);
      run_case(
          5,
          1,
          {0},
          {2},
          &generator,
          &distribution,
          &max_overlap_error,
          &max_one_electron_error,
          &max_total_error,
          &max_two_electron_error,
          &max_spatial_gradient_error,
          &max_ggo_gradient_error,
          false);
      run_case(
          6,
          2,
          {0, 3},
          {1, 4},
          &generator,
          &distribution,
          &max_overlap_error,
          &max_one_electron_error,
          &max_total_error,
          &max_two_electron_error,
          &max_spatial_gradient_error,
          &max_ggo_gradient_error,
          false);
      run_case(
          6,
          1,
          {0, 2, 4},
          {1, 3, 5},
          &generator,
          &distribution,
          &max_overlap_error,
          &max_one_electron_error,
          &max_total_error,
          &max_two_electron_error,
          &max_spatial_gradient_error,
          &max_ggo_gradient_error,
          false);
    }

    std::cout << std::setprecision(16);
    std::cout << "max_overlap_error = " << max_overlap_error << '\n';
    std::cout << "max_one_electron_error = " << max_one_electron_error << '\n';
    std::cout << "max_two_electron_error = " << max_two_electron_error << '\n';
    std::cout << "max_total_error = " << max_total_error << '\n';
    std::cout << "max_spatial_gradient_error = " << max_spatial_gradient_error << '\n';
    std::cout << "max_ggo_gradient_error = " << max_ggo_gradient_error << '\n';

    const double tolerance = 1.0e-6;
    if (max_overlap_error > tolerance ||
        max_one_electron_error > tolerance ||
        max_two_electron_error > tolerance ||
        max_total_error > tolerance ||
        max_spatial_gradient_error > tolerance ||
        max_ggo_gradient_error > tolerance) {
      std::cerr << "high-spin open-shell two-electron validation failed\n";
      return 1;
    }
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "error: " << exception.what() << '\n';
    return 1;
  }
}
