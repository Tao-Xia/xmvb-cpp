#include "pfaffian_vbscf/kernel/pf_adjoint_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "pfaffian_vbscf/kernel/pf_operand_views.hpp"
#include "pfaffian_vbscf/math/dense_utils.hpp"
#include "pfaffian_vbscf/math/pf_matrix_polynomial.hpp"
#include "pfaffian_vbscf/math/trace_projector.hpp"
#include "pfaffian_vbscf/tensor/pf_packed_index_cache.hpp"
#include "pfaffian_vbscf/tensor/pf_tensor_contractor.hpp"

namespace xmvb::pfaffian_vbscf {

namespace {

Matrix build_trace_path_kernel_adjoint(
    const PfKernelCache& cache,
    const ScalarBuffer& trace_weights) {
  if (trace_weights.empty()) {
    return Matrix::Zero(cache.n_spin_orbitals, cache.n_spin_orbitals);
  }
  return build_kernel_trace_adjoint(cache.kernel_powers, trace_weights);
}

Matrix build_trace_path_kernel_adjoint(const PfKernelCache& cache) {
  return build_trace_path_kernel_adjoint(cache, cache.trace_weights);
}

const ScalarBuffer& closed_shell_trace_series(const PfKernelCache& cache) {
  if (!cache.closed_shell_traces.empty()) {
    return cache.closed_shell_traces;
  }
  return cache.traces;
}

Matrix build_closed_shell_trace_path_kernel_block_adjoint(
    const PfKernelCache& cache,
    const ScalarBuffer& trace_weights) {
  if (trace_weights.empty()) {
    return Matrix::Zero(cache.n_active_orbitals, cache.n_active_orbitals);
  }
  if (cache.closed_shell_trace_powers.size() < trace_weights.size()) {
    throw std::invalid_argument(
        "closed-shell trace powers do not cover the requested trace weights");
  }
  return 2.0 *
      build_kernel_trace_adjoint(cache.closed_shell_trace_powers, trace_weights);
}

double frobenius_inner_product(
    const ConstMatrixRef& left,
    const ConstMatrixRef& right) {
  if (left.rows() != right.rows() || left.cols() != right.cols()) {
    throw std::invalid_argument("Frobenius inner product dimension mismatch");
  }
  return left.cwiseProduct(right).sum();
}

void resize_and_zero_matrix(
    int rows,
    int cols,
    Matrix* matrix) {
  if (matrix == nullptr) {
    throw std::invalid_argument("workspace matrix must not be null");
  }
  matrix->resize(rows, cols);
  matrix->setZero();
}

void resize_and_zero_buffer(
    std::size_t size,
    ScalarBuffer* buffer) {
  if (buffer == nullptr) {
    throw std::invalid_argument("workspace buffer must not be null");
  }
  if (buffer->size() != size) {
    buffer->assign(size, 0.0);
  } else {
    std::fill(buffer->begin(), buffer->end(), 0.0);
  }
}

void prepare_closed_shell_two_electron_workspace(
    const PfKernelCache& cache,
    PfClosedShellTwoElectronAdjointWorkspace* workspace) {
  if (workspace == nullptr) {
    throw std::invalid_argument("closed-shell adjoint workspace must not be null");
  }

  const int n = cache.n_active_orbitals;
  resize_and_zero_matrix(n, n, &workspace->pair_term_matrix_adjoint);
  resize_and_zero_matrix(n, n, &workspace->c_poly_h_n2_adjoint);
  resize_and_zero_matrix(n, n, &workspace->c_adjoint);
  resize_and_zero_matrix(n, n, &workspace->frechet_h_c_h_n2_adjoint);
  resize_and_zero_matrix(n, n, &workspace->d_poly_h_n2_adjoint);
  resize_and_zero_matrix(n, n, &workspace->d_adjoint);
  resize_and_zero_matrix(n, n, &workspace->opposite_bridge_h_split_matrix_adjoint);
  resize_and_zero_matrix(n, n, &workspace->pair_poly_adjoint);
  resize_and_zero_matrix(n, n, &workspace->frechet_h_c_n2_adjoint);
  resize_and_zero_matrix(n, n, &workspace->h_adjoint);
  resize_and_zero_matrix(n, n, &workspace->sa_adjoint);
  resize_and_zero_buffer(
      cache.closed_shell_pair_coefficients.size(),
      &workspace->pair_coefficient_adjoints);
  resize_and_zero_buffer(
      cache.closed_shell_coeff_n2.size(),
      &workspace->coeff_n2_adjoints);
  resize_and_zero_buffer(
      xmvb::to_size(cache.trace_order + 1),
      &workspace->overlap_coefficient_adjoints);
}

std::size_t closed_shell_packed_pair_count(int n_active_orbitals) {
  const std::size_t n = xmvb::to_size(n_active_orbitals);
  return n * (n + 1) / 2;
}

Matrix build_spin_diagonal_one_rdm_adjoint(
    const PfKernelCache& cache,
    const ConstMatrixRef& spatial_adjoint) {
  if (spatial_adjoint.rows() != cache.n_active_orbitals ||
      spatial_adjoint.cols() != cache.n_active_orbitals) {
    throw std::invalid_argument(
        "spatial_adjoint does not match the active-space dimension");
  }

  Matrix full_adjoint =
      Matrix::Zero(cache.n_spin_orbitals, cache.n_spin_orbitals);
  full_adjoint.topLeftCorner(cache.n_active_orbitals, cache.n_active_orbitals) =
      spatial_adjoint;
  full_adjoint.bottomRightCorner(
      cache.n_active_orbitals,
      cache.n_active_orbitals) = spatial_adjoint;
  return full_adjoint;
}

void accumulate_one_term_gradient(
    const PfKernelCache& cache,
    const PfTensorTerm& term,
    ScalarBuffer* ggo_grad) {
  const detail::MatrixBlock left_block =
      detail::materialize_operand_block(cache, term.left_operand);
  const detail::MatrixBlock right_block =
      detail::materialize_operand_block(cache, term.right_operand);

  switch (term.contraction) {
    case PfTensorContraction::Direct:
      PfTensorContractor::add_direct_outer_product(
          term.scale,
          left_block,
          right_block,
          ggo_grad);
      break;
    case PfTensorContraction::Exchange:
      if (term.same_spin) {
        PfTensorContractor::add_same_spin_exchange_outer_product(
            term.scale,
            left_block,
            right_block,
            ggo_grad);
      } else {
        PfTensorContractor::add_exchange_outer_product(
            term.scale,
            left_block,
            right_block,
            ggo_grad);
      }
      break;
    case PfTensorContraction::Coulomb:
      if (term.same_spin) {
        PfTensorContractor::add_same_spin_coulomb_outer_product(
            term.scale,
            left_block,
            right_block,
            ggo_grad);
      } else {
        PfTensorContractor::add_coulomb_outer_product(
            term.scale,
            left_block,
            right_block,
            ggo_grad);
      }
      break;
    case PfTensorContraction::SameSpinSeparable:
      PfTensorContractor::add_same_spin_separable_outer_product(
          term.scale,
          left_block,
          right_block,
          ggo_grad);
      break;
    case PfTensorContraction::SameSpinBridge:
      PfTensorContractor::add_same_spin_bridge_outer_product(
          term.scale,
          left_block,
          right_block,
          ggo_grad);
      break;
  }
}

void write_spin_block(
    PfSpinBlock block,
    const ConstMatrixRef& block_adjoint,
    MatrixRef full_adjoint) {
  const Eigen::Index n = block_adjoint.rows();
  switch (block) {
    case PfSpinBlock::AlphaAlpha:
      full_adjoint.block(0, 0, n, n) = block_adjoint;
      return;
    case PfSpinBlock::AlphaBeta:
      full_adjoint.block(0, n, n, n) = block_adjoint;
      return;
    case PfSpinBlock::BetaAlpha:
      full_adjoint.block(n, 0, n, n) = block_adjoint;
      return;
    case PfSpinBlock::BetaBeta:
      full_adjoint.block(n, n, n, n) = block_adjoint;
      return;
  }
}

void accumulate_kernel_power_adjoint(
    const PfKernelCache& cache,
    int power,
    const ConstMatrixRef& power_adjoint,
    MatrixRef kernel_adjoint) {
  if (power <= 0) {
    return;
  }

  for (int left_power = 0; left_power < power; ++left_power) {
    const Matrix& left_factor = detail::kernel_power_matrix(cache, left_power);
    const Matrix& right_factor = detail::kernel_power_matrix(cache, power - 1 - left_power);
    kernel_adjoint.noalias() +=
        left_factor.transpose() * power_adjoint * right_factor.transpose();
  }
}

void accumulate_trace_weighted_source_adjoint(
    const PfKernelCache& cache,
    const ConstMatrixRef& full_adjoint,
    const ScalarBuffer& source_trace_weights,
    bool backpropagate_trace_weight_adjoints,
    MatrixRef source_workspace,
    MatrixRef kernel_adjoint,
    MatrixRef sigma_adjoint) {
  const Matrix trace_path_kernel_adjoint =
      build_trace_path_kernel_adjoint(cache, source_trace_weights);

  sigma_adjoint.noalias() +=
      trace_path_kernel_adjoint.transpose() *
      cache.left *
      full_adjoint *
      cache.right;
  sigma_adjoint.noalias() +=
      cache.left.transpose() *
      trace_path_kernel_adjoint *
      full_adjoint *
      cache.right.transpose();

  Matrix trace_kernel_adjoint =
      cache.left * full_adjoint * cache.right * cache.sigma.transpose();
  trace_kernel_adjoint.noalias() +=
      full_adjoint *
      cache.right.transpose() *
      cache.sigma.transpose() *
      cache.left.transpose();

  ScalarBuffer trace_weight_adjoints(
      xmvb::to_size(cache.trace_order),
      0.0);
  for (int power = 0; power < cache.trace_order; ++power) {
    const Matrix& kernel_power = detail::kernel_power_matrix(cache, power);
    const double scaled_power = static_cast<double>(power + 1);
    const double trace_weight =
        source_trace_weights[xmvb::to_size(power)];

    if (backpropagate_trace_weight_adjoints) {
      trace_weight_adjoints[xmvb::to_size(power)] +=
          scaled_power *
          frobenius_inner_product(
              trace_kernel_adjoint,
              kernel_power.transpose());
    }

    if (trace_weight == 0.0 || power == 0) {
      continue;
    }

    source_workspace.noalias() =
        (trace_weight * scaled_power) *
        trace_kernel_adjoint.transpose();
    accumulate_kernel_power_adjoint(
        cache,
        power,
        source_workspace,
        kernel_adjoint);
  }

  if (backpropagate_trace_weight_adjoints) {
    const ScalarBuffer trace_adjoints =
        TraceProjector::backpropagate_trace_weight_adjoints(
            cache.traces,
            cache.trace_order,
            trace_weight_adjoints);
    if (!trace_adjoints.empty()) {
      kernel_adjoint.noalias() +=
          build_kernel_trace_adjoint(cache.kernel_powers, trace_adjoints);
    }
  }
}

void accumulate_one_rdm_source_adjoint(
    const PfKernelCache& cache,
    const ConstMatrixRef& spatial_adjoint,
    const ScalarBuffer& source_trace_weights,
    bool backpropagate_trace_weight_adjoints,
    MatrixRef source_workspace,
    MatrixRef kernel_adjoint,
    MatrixRef sigma_adjoint) {
  accumulate_trace_weighted_source_adjoint(
      cache,
      build_spin_diagonal_one_rdm_adjoint(cache, spatial_adjoint),
      source_trace_weights,
      backpropagate_trace_weight_adjoints,
      source_workspace,
      kernel_adjoint,
      sigma_adjoint);
}

void accumulate_trace_rdm_source_adjoint(
    const PfKernelCache& cache,
    int trace_index,
    PfSpinBlock block,
    const ConstMatrixRef& block_adjoint,
    MatrixRef full_workspace,
    MatrixRef source_workspace,
    MatrixRef kernel_adjoint,
    MatrixRef sigma_adjoint) {
  if (trace_index < 0 || trace_index >= cache.trace_order) {
    throw std::invalid_argument("trace RDM index is out of range");
  }

  ScalarBuffer source_trace_weights(
      xmvb::to_size(cache.trace_order),
      0.0);
  source_trace_weights[xmvb::to_size(trace_index)] = 1.0;
  full_workspace.setZero();
  write_spin_block(block, block_adjoint, full_workspace);
  accumulate_trace_weighted_source_adjoint(
      cache,
      full_workspace,
      source_trace_weights,
      false,
      source_workspace,
      kernel_adjoint,
      sigma_adjoint);
}

void accumulate_kernel_power_times_left_sigma_right_source_adjoint(
    const PfKernelCache& cache,
    int power,
    const ConstMatrixRef& full_adjoint,
    MatrixRef source_workspace,
    MatrixRef kernel_adjoint,
    MatrixRef sigma_adjoint) {
  if (power < 0 ||
      power >= static_cast<int>(cache.kernel_power_times_left_sigma_right.size())) {
    throw std::invalid_argument("kernel power-times-left-sigma-right is out of range");
  }

  if (power > 0) {
    source_workspace.noalias() =
        full_adjoint * cache.left_sigma_right.transpose();
    accumulate_kernel_power_adjoint(
        cache,
        power,
        source_workspace,
        kernel_adjoint);
  }

  source_workspace.noalias() =
      detail::kernel_power_matrix(cache, power).transpose() * full_adjoint;
  sigma_adjoint.noalias() +=
      cache.left.transpose() *
      source_workspace *
      cache.right.transpose();
}

void accumulate_source_adjoint(
    const PfKernelCache& cache,
    const PfTensorOperand& operand,
    const ConstMatrixRef& block_adjoint,
    MatrixRef full_workspace,
    MatrixRef source_workspace,
    MatrixRef kernel_adjoint,
    MatrixRef sigma_adjoint) 
{
  full_workspace.setZero();

  switch (operand.source) {
    case PfMatrixSource::Identity:
    case PfMatrixSource::Left:
    case PfMatrixSource::Right:
      return;
    case PfMatrixSource::OneRDM:
      accumulate_one_rdm_source_adjoint(
          cache,
          block_adjoint,
          cache.trace_weights,
          true,
          source_workspace,
          kernel_adjoint,
          sigma_adjoint);
      return;
    case PfMatrixSource::TraceRDM:
      accumulate_trace_rdm_source_adjoint(
          cache,
          operand.power,
          operand.block,
          block_adjoint,
          full_workspace,
          source_workspace,
          kernel_adjoint,
          sigma_adjoint);
      return;
  }

  write_spin_block(operand.block, block_adjoint, full_workspace);

  switch (operand.source) {
    case PfMatrixSource::Sigma:
      sigma_adjoint += full_workspace;
      return;
    case PfMatrixSource::KernelPower:
      accumulate_kernel_power_adjoint(cache, operand.power, full_workspace, kernel_adjoint);
      return;
    case PfMatrixSource::KernelPowerTimesLeft:
      if (operand.power <= 0) {
        return;
      }
      source_workspace.noalias() = full_workspace * cache.left.transpose();
      accumulate_kernel_power_adjoint(cache, operand.power, source_workspace, kernel_adjoint);
      return;
    case PfMatrixSource::KernelPowerTimesRight:
      if (operand.power <= 0) {
        return;
      }
      source_workspace.noalias() = full_workspace * cache.right.transpose();
      accumulate_kernel_power_adjoint(cache, operand.power, source_workspace, kernel_adjoint);
      return;
    case PfMatrixSource::KernelPowerTimesLeftSigmaRight:
      accumulate_kernel_power_times_left_sigma_right_source_adjoint(
          cache,
          operand.power,
          full_workspace,
          source_workspace,
          kernel_adjoint,
          sigma_adjoint);
      return;
    case PfMatrixSource::RightSigmaTimesKernelPowerTimesLeftSigmaRight:
      if (operand.power < 0 ||
          operand.power >=
              static_cast<int>(
                  cache.right_sigma_times_kernel_power_times_left_sigma_right.size())) {
        throw std::invalid_argument(
            "right-sigma-times-kernel-power-times-left-sigma-right is out of range");
      }
      source_workspace = full_workspace;
      full_workspace.noalias() =
          source_workspace *
          cache.kernel_power_times_left_sigma_right[
              xmvb::to_size(operand.power)]
              .transpose();
      sigma_adjoint.noalias() +=
          cache.right.transpose() * full_workspace;
      full_workspace.noalias() =
          cache.right_sigma.transpose() * source_workspace;
      accumulate_kernel_power_times_left_sigma_right_source_adjoint(
          cache,
          operand.power,
          full_workspace,
          source_workspace,
          kernel_adjoint,
          sigma_adjoint);
      return;
    case PfMatrixSource::Identity:
    case PfMatrixSource::Left:
    case PfMatrixSource::Right:
    case PfMatrixSource::OneRDM:
    case PfMatrixSource::TraceRDM:
      return;
  }
}

void accumulate_tensor_path_adjoints(
    const PfKernelCache& cache,
    const ScalarBuffer& ggo,
    const std::vector<PfTensorTerm>& terms,
    MatrixRef kernel_adjoint,
    MatrixRef sigma_adjoint) {
  const int n = cache.n_active_orbitals;
  const int n_spin = cache.n_spin_orbitals;
  Matrix left_operand_adjoint = Matrix::Zero(n, n);
  Matrix right_operand_adjoint = Matrix::Zero(n, n);
  Matrix full_workspace = Matrix::Zero(n_spin, n_spin);
  Matrix source_workspace = Matrix::Zero(n_spin, n_spin);

  for (const PfTensorTerm& term : terms) {
    left_operand_adjoint.setZero();
    right_operand_adjoint.setZero();
    const detail::MatrixBlock left_block =
        detail::materialize_operand_block(cache, term.left_operand);
    const detail::MatrixBlock right_block =
        detail::materialize_operand_block(cache, term.right_operand);

    switch (term.contraction) {
      case PfTensorContraction::Direct:
        PfTensorContractor::compute_direct_operand_adjoints(
            term.scale,
            left_block,
            right_block,
            ggo,
            left_operand_adjoint,
            right_operand_adjoint);
        break;
      case PfTensorContraction::Exchange:
        if (term.same_spin) {
          PfTensorContractor::compute_same_spin_exchange_operand_adjoints(
              term.scale,
              left_block,
              right_block,
              ggo,
              left_operand_adjoint,
              right_operand_adjoint);
        } else {
          PfTensorContractor::compute_exchange_operand_adjoints(
              term.scale,
              left_block,
              right_block,
              ggo,
              left_operand_adjoint,
              right_operand_adjoint);
        }
        break;
      case PfTensorContraction::Coulomb:
        if (term.same_spin) {
          PfTensorContractor::compute_same_spin_coulomb_operand_adjoints(
              term.scale,
              left_block,
              right_block,
              ggo,
              left_operand_adjoint,
              right_operand_adjoint);
        } else {
          PfTensorContractor::compute_coulomb_operand_adjoints(
              term.scale,
              left_block,
              right_block,
              ggo,
              left_operand_adjoint,
              right_operand_adjoint);
        }
        break;
      case PfTensorContraction::SameSpinSeparable:
        PfTensorContractor::compute_same_spin_separable_operand_adjoints(
            term.scale,
            left_block,
            right_block,
            ggo,
            left_operand_adjoint,
            right_operand_adjoint);
        break;
      case PfTensorContraction::SameSpinBridge:
        PfTensorContractor::compute_same_spin_bridge_operand_adjoints(
            term.scale,
            left_block,
            right_block,
            ggo,
            left_operand_adjoint,
            right_operand_adjoint);
        break;
    }

    // Route dE/dA and dE/dB through the operand source definitions.
    accumulate_source_adjoint(
        cache,
        term.left_operand,
        left_operand_adjoint,
        full_workspace,
        source_workspace,
        kernel_adjoint,
        sigma_adjoint);
    accumulate_source_adjoint(
        cache,
        term.right_operand,
        right_operand_adjoint,
        full_workspace,
        source_workspace,
        kernel_adjoint,
      sigma_adjoint);
  }
}

PfTensorOperand operand(
    PfMatrixSource source,
    PfSpinBlock block,
    int power = 0) {
  PfTensorOperand term_operand;
  term_operand.source = source;
  term_operand.power = power;
  term_operand.block = block;
  return term_operand;
}

PfTensorTerm term(
    PfTensorContraction contraction,
    const PfTensorOperand& left_operand,
    const PfTensorOperand& right_operand,
    double scale,
    bool same_spin = false) {
  PfTensorTerm tensor_term;
  tensor_term.contraction = contraction;
  tensor_term.left_operand = left_operand;
  tensor_term.right_operand = right_operand;
  tensor_term.scale = scale;
  tensor_term.same_spin = same_spin;
  return tensor_term;
}

Matrix copy_block_matrix(
    const PfKernelCache& cache,
    const ConstMatrixRef& source,
    PfSpinBlock block) {
  return detail::copy_spin_block(cache, source, block).eval();
}

std::vector<Matrix> build_block_sequence(
    const PfKernelCache& cache,
    const std::vector<Matrix>& source_family,
    PfSpinBlock block,
    int count) {
  if (count < 0 || count > static_cast<int>(source_family.size())) {
    throw std::invalid_argument("requested block sequence size is out of range");
  }

  std::vector<Matrix> sequence;
  sequence.reserve(xmvb::to_size(count));
  for (int index = 0; index < count; ++index) {
    sequence.push_back(
        detail::copy_spin_block(
            cache,
            source_family[xmvb::to_size(index)],
            block)
            .eval());
  }
  return sequence;
}

std::vector<Matrix> build_trace_block_sequence(
    const PfKernelCache& cache,
    PfSpinBlock block) {
  std::vector<Matrix> sequence;
  sequence.reserve(xmvb::to_size(cache.trace_order));
  for (int index = 0; index < cache.trace_order; ++index) {
    sequence.push_back(
        detail::copy_spin_block(
            cache,
            cache.trace_rdms[xmvb::to_size(index)],
            block)
            .eval());
  }
  return sequence;
}

std::vector<Matrix> build_scaled_sequence(
    const std::vector<Matrix>& sequence,
    const ScalarBuffer& scales) {
  if (sequence.size() != scales.size()) {
    throw std::invalid_argument("sequence/scales size mismatch");
  }

  std::vector<Matrix> scaled_sequence;
  scaled_sequence.reserve(sequence.size());
  for (std::size_t index = 0; index < sequence.size(); ++index) {
    scaled_sequence.push_back(scales[index] * sequence[index]);
  }
  return scaled_sequence;
}

std::vector<Matrix> build_zero_sequence(
    int count,
    int n_active_orbitals) {
  return std::vector<Matrix>(
      xmvb::to_size(count),
      Matrix::Zero(n_active_orbitals, n_active_orbitals));
}

std::vector<Matrix> build_hankel_aggregates(
    const std::vector<Matrix>& sequence,
    const ScalarBuffer& sum_weights) {
  if (sequence.empty()) {
    return {};
  }
  if (sum_weights.size() != sequence.size()) {
    throw std::invalid_argument("sum_weights size does not match the sequence length");
  }

  const Eigen::Index n = sequence.front().rows();
  std::vector<Matrix> aggregates(
      sequence.size(),
      Matrix::Zero(n, n));
  for (std::size_t right_index = 0; right_index < sequence.size(); ++right_index) {
    for (std::size_t left_index = 0;
         left_index + right_index < sequence.size();
         ++left_index) {
      const double weight = sum_weights[left_index + right_index];
      if (weight == 0.0) {
        continue;
      }
      aggregates[right_index].noalias() +=
          weight * sequence[left_index];
    }
  }
  return aggregates;
}

void scatter_hankel_adjoints(
    const ScalarBuffer& sum_weights,
    const std::vector<Matrix>& aggregate_adjoints,
    std::vector<Matrix>* sequence_adjoints) {
  if (sequence_adjoints == nullptr) {
    throw std::invalid_argument("sequence_adjoints must not be null");
  }
  if (sum_weights.size() != aggregate_adjoints.size() ||
      sum_weights.size() != sequence_adjoints->size()) {
    throw std::invalid_argument("Hankel adjoint dimensions do not match");
  }

  for (std::size_t right_index = 0;
       right_index < aggregate_adjoints.size();
       ++right_index) {
    for (std::size_t left_index = 0;
         left_index + right_index < aggregate_adjoints.size();
         ++left_index) {
      const double weight = sum_weights[left_index + right_index];
      if (weight == 0.0) {
        continue;
      }
      (*sequence_adjoints)[left_index].noalias() +=
          weight * aggregate_adjoints[right_index];
    }
  }
}

void accumulate_sequence_source_adjoints(
    const PfKernelCache& cache,
    PfMatrixSource source,
    PfSpinBlock block,
    const std::vector<Matrix>& sequence_adjoints,
    MatrixRef full_workspace,
    MatrixRef source_workspace,
    MatrixRef kernel_adjoint,
    MatrixRef sigma_adjoint) {
  for (std::size_t index = 0; index < sequence_adjoints.size(); ++index) {
    if (sequence_adjoints[index].rows() == 0) {
      continue;
    }
    accumulate_source_adjoint(
        cache,
        operand(
            source,
            block,
            static_cast<int>(index)),
        sequence_adjoints[index],
        full_workspace,
        source_workspace,
        kernel_adjoint,
        sigma_adjoint);
  }
}

ScalarBuffer build_closed_shell_projected_coefficients(
    const PfKernelCache& cache,
    int order,
    double scale = 1.0) {
  if (order < 0) {
    return {};
  }
  if (static_cast<int>(cache.projected_overlap_coefficients.size()) !=
      cache.trace_order + 1) {
    throw std::invalid_argument(
        "cache.projected_overlap_coefficients size does not match "
        "cache.trace_order + 1");
  }

  ScalarBuffer coefficients(xmvb::to_size(order + 1), 0.0);
  for (int power = 0; power <= order; ++power) {
    const double sign = ((power % 2) == 0) ? 1.0 : -1.0;
    coefficients[xmvb::to_size(power)] =
        scale *
        sign *
        cache.projected_overlap_coefficients[xmvb::to_size(
            order - power)];
  }
  return coefficients;
}

void accumulate_projected_coefficient_family_adjoints(
    int order,
    double scale,
    const ScalarBuffer& family_adjoints,
    ScalarBuffer* overlap_coefficient_adjoints) {
  if (overlap_coefficient_adjoints == nullptr) {
    throw std::invalid_argument("overlap_coefficient_adjoints must not be null");
  }
  if (order < 0) {
    return;
  }
  if (static_cast<int>(family_adjoints.size()) != order + 1) {
    throw std::invalid_argument("family_adjoints size does not match order + 1");
  }
  if (static_cast<int>(overlap_coefficient_adjoints->size()) < order + 1) {
    throw std::invalid_argument(
        "overlap_coefficient_adjoints size is smaller than order + 1");
  }

  for (int power = 0; power <= order; ++power) {
    const double sign = ((power % 2) == 0) ? 1.0 : -1.0;
    (*overlap_coefficient_adjoints)[xmvb::to_size(order - power)] +=
        scale * sign * family_adjoints[xmvb::to_size(power)];
  }
}

void accumulate_closed_shell_spatial_sigma_adjoint(
    const ConstMatrixRef& spatial_adjoint,
    MatrixRef sigma_adjoint) {
  if (spatial_adjoint.rows() != spatial_adjoint.cols()) {
    throw std::invalid_argument("spatial_adjoint must be square");
  }
  if (sigma_adjoint.rows() != sigma_adjoint.cols()) {
    throw std::invalid_argument("sigma_adjoint must be square");
  }

  const Eigen::Index n = spatial_adjoint.rows();
  if (sigma_adjoint.rows() != 2 * n) {
    throw std::invalid_argument(
        "sigma_adjoint dimension does not match the closed-shell spatial block");
  }

  sigma_adjoint.topLeftCorner(n, n).noalias() += 0.5 * spatial_adjoint;
  sigma_adjoint.bottomRightCorner(n, n).noalias() += 0.5 * spatial_adjoint;
}

void accumulate_closed_shell_kernel_block_spatial_adjoint(
    const PfKernelCache& cache,
    const ConstMatrixRef& kernel_block_adjoint,
    MatrixRef spatial_adjoint) {
  if (kernel_block_adjoint.rows() != cache.n_active_orbitals ||
      kernel_block_adjoint.cols() != cache.n_active_orbitals) {
    throw std::invalid_argument(
        "closed-shell kernel-block adjoint does not match the active-space dimension");
  }
  if (spatial_adjoint.rows() != cache.n_active_orbitals ||
      spatial_adjoint.cols() != cache.n_active_orbitals) {
    throw std::invalid_argument(
        "closed-shell spatial adjoint does not match the active-space dimension");
  }

  Matrix c_adjoint =
      kernel_block_adjoint * cache.closed_shell_spatial_overlap;
  spatial_adjoint.noalias() +=
      cache.closed_shell_pair_core.transpose() * kernel_block_adjoint;
  // `closed_shell_pair_core = left_ba_block^T * S * right_ab_block^T`.
  // For a matrix product `C = L * S * R`, the overlap adjoint is
  // `S_bar += L^T * C_bar * R^T`. Here `L = left_ba_block^T` and
  // `R = right_ab_block^T`, so the physical spatial-overlap contribution is
  // `left_ba_block * C_bar * right_ab_block`.
  spatial_adjoint.noalias() +=
      cache.closed_shell_left_ba_block *
      c_adjoint *
      cache.closed_shell_right_ab_block;
}

void accumulate_closed_shell_kernel_block_sigma_adjoint(
    const PfKernelCache& cache,
    const ConstMatrixRef& kernel_block_adjoint,
    MatrixRef sigma_adjoint) {
  Matrix spatial_adjoint =
      Matrix::Zero(cache.n_active_orbitals, cache.n_active_orbitals);
  accumulate_closed_shell_kernel_block_spatial_adjoint(
      cache,
      kernel_block_adjoint,
      spatial_adjoint);
  accumulate_closed_shell_spatial_sigma_adjoint(spatial_adjoint, sigma_adjoint);
}

void accumulate_closed_shell_trace_path_spatial_adjoint(
    const PfKernelCache& cache,
    const ScalarBuffer& trace_weights,
    MatrixRef spatial_adjoint) {
  if (trace_weights.empty()) {
    return;
  }
  accumulate_closed_shell_kernel_block_spatial_adjoint(
      cache,
      build_closed_shell_trace_path_kernel_block_adjoint(cache, trace_weights),
      spatial_adjoint);
}

void accumulate_closed_shell_trace_path_sigma_adjoint(
    const PfKernelCache& cache,
    const ScalarBuffer& trace_weights,
    MatrixRef sigma_adjoint) {
  Matrix spatial_adjoint =
      Matrix::Zero(cache.n_active_orbitals, cache.n_active_orbitals);
  accumulate_closed_shell_trace_path_spatial_adjoint(
      cache,
      trace_weights,
      spatial_adjoint);
  accumulate_closed_shell_spatial_sigma_adjoint(spatial_adjoint, sigma_adjoint);
}

void accumulate_closed_shell_one_rdm_exact_terms(
    const PfKernelCache& cache,
    const ConstMatrixRef& one_rdm_adjoint,
    MatrixRef h_adjoint,
    MatrixRef c_adjoint,
    ScalarBuffer* overlap_coefficient_adjoints) {
  if (overlap_coefficient_adjoints == nullptr) {
    throw std::invalid_argument("overlap_coefficient_adjoints must not be null");
  }
  if (one_rdm_adjoint.rows() != cache.n_active_orbitals ||
      one_rdm_adjoint.cols() != cache.n_active_orbitals) {
    throw std::invalid_argument(
        "one_rdm_adjoint does not match the active-space dimension");
  }
  if (h_adjoint.rows() != cache.n_active_orbitals ||
      h_adjoint.cols() != cache.n_active_orbitals ||
      c_adjoint.rows() != cache.n_active_orbitals ||
      c_adjoint.cols() != cache.n_active_orbitals) {
    throw std::invalid_argument(
        "closed-shell exact adjoint workspaces do not match the active-space dimension");
  }
  if (static_cast<int>(overlap_coefficient_adjoints->size()) !=
      cache.trace_order + 1) {
    throw std::invalid_argument(
        "overlap_coefficient_adjoints size does not match cache.trace_order + 1");
  }

  if (cache.trace_order <= 0) {
    return;
  }

  const ScalarBuffer density_coefficients =
      build_closed_shell_projected_coefficients(
          cache,
          cache.trace_order - 1,
          2.0);
  const MatrixPolynomialAdjointResult one_rdm_result =
      backpropagate_left_matrix_polynomial(
          cache.closed_shell_spatial_kernel,
          density_coefficients,
          cache.closed_shell_pair_core,
          one_rdm_adjoint);
  h_adjoint.noalias() += one_rdm_result.kernel_adjoint;
  c_adjoint.noalias() += one_rdm_result.source_adjoint;
  accumulate_projected_coefficient_family_adjoints(
      cache.trace_order - 1,
      2.0,
      one_rdm_result.coefficient_adjoints,
      overlap_coefficient_adjoints);
}

void finalize_closed_shell_exact_spatial_adjoint(
    const PfKernelCache& cache,
    MatrixRef h_adjoint,
    MatrixRef c_adjoint,
    const ConstMatrixRef& sa_adjoint,
    const ScalarBuffer& overlap_coefficient_adjoints,
    MatrixRef spatial_adjoint) {
  if (h_adjoint.rows() != cache.n_active_orbitals ||
      h_adjoint.cols() != cache.n_active_orbitals ||
      c_adjoint.rows() != cache.n_active_orbitals ||
      c_adjoint.cols() != cache.n_active_orbitals ||
      sa_adjoint.rows() != cache.n_active_orbitals ||
      sa_adjoint.cols() != cache.n_active_orbitals ||
      spatial_adjoint.rows() != cache.n_active_orbitals ||
      spatial_adjoint.cols() != cache.n_active_orbitals) {
    throw std::invalid_argument(
        "closed-shell exact spatial adjoint dimensions do not match the active-space dimension");
  }
  if (static_cast<int>(overlap_coefficient_adjoints.size()) !=
      cache.trace_order + 1) {
    throw std::invalid_argument(
        "overlap_coefficient_adjoints size does not match cache.trace_order + 1");
  }

  const Matrix& spatial_overlap = cache.closed_shell_spatial_overlap;
  const Matrix& a = cache.closed_shell_left_ba_block;
  const Matrix& b = cache.closed_shell_right_ab_block;
  const Matrix& c = cache.closed_shell_pair_core;

  // `h = c * S`, so the matrix-product reverse sweep is
  // `S_bar += c^T * h_bar` and `c_bar += h_bar * S^T`.
  spatial_adjoint.noalias() += c.transpose() * h_adjoint;
  c_adjoint.noalias() += h_adjoint * spatial_overlap.transpose();
  spatial_adjoint.noalias() += sa_adjoint * a.transpose();
  // `c = a^T * S * b^T` with raw cache blocks `a = left_ba_block` and
  // `b = right_ab_block`, so the spatial-overlap adjoint is `a * c_bar * b`.
  spatial_adjoint.noalias() +=
      a * c_adjoint * b;

  if (cache.trace_order <= 0) {
    return;
  }

  const ScalarBuffer trace_adjoints =
      TraceProjector::backpropagate_overlap_coefficient_adjoints(
          closed_shell_trace_series(cache),
      cache.trace_order,
      overlap_coefficient_adjoints);
  if (!trace_adjoints.empty()) {
    accumulate_closed_shell_trace_path_spatial_adjoint(
        cache,
        trace_adjoints,
        spatial_adjoint);
  }
}

void finalize_closed_shell_left_aux_spatial_adjoint(
    const PfKernelCache& cache,
    MatrixRef h_adjoint,
    MatrixRef c_adjoint,
    const ConstMatrixRef& sa_adjoint,
    MatrixRef spatial_adjoint) {
  if (h_adjoint.rows() != cache.n_active_orbitals ||
      h_adjoint.cols() != cache.n_active_orbitals ||
      c_adjoint.rows() != cache.n_active_orbitals ||
      c_adjoint.cols() != cache.n_active_orbitals ||
      sa_adjoint.rows() != cache.n_active_orbitals ||
      sa_adjoint.cols() != cache.n_active_orbitals ||
      spatial_adjoint.rows() != cache.n_active_orbitals ||
      spatial_adjoint.cols() != cache.n_active_orbitals) {
    throw std::invalid_argument(
        "closed-shell left-aux spatial adjoint dimensions do not match the active-space dimension");
  }

  const Matrix& spatial_overlap = cache.closed_shell_spatial_overlap;
  const Matrix& a = cache.closed_shell_left_ba_block;
  const Matrix& b = cache.closed_shell_right_ab_block;
  const Matrix& c = cache.closed_shell_pair_core;

  // The two-electron auxiliary `closed_shell_h` is defined as `h = S * c`.
  // Its reverse sweep is therefore `S_bar += h_bar * c^T` and
  // `c_bar += S^T * h_bar`.
  spatial_adjoint.noalias() += h_adjoint * c.transpose();
  c_adjoint.noalias() += spatial_overlap.transpose() * h_adjoint;
  spatial_adjoint.noalias() += sa_adjoint * a.transpose();
  spatial_adjoint.noalias() += a * c_adjoint * b;
}

void accumulate_closed_shell_overlap_coefficient_spatial_adjoint(
    const PfKernelCache& cache,
    const ScalarBuffer& overlap_coefficient_adjoints,
    MatrixRef spatial_adjoint) {
  if (static_cast<int>(overlap_coefficient_adjoints.size()) !=
      cache.trace_order + 1) {
    throw std::invalid_argument(
        "overlap_coefficient_adjoints size does not match cache.trace_order + 1");
  }
  if (cache.trace_order <= 0) {
    return;
  }

  const ScalarBuffer trace_adjoints =
      TraceProjector::backpropagate_overlap_coefficient_adjoints(
          closed_shell_trace_series(cache),
          cache.trace_order,
          overlap_coefficient_adjoints);
  if (!trace_adjoints.empty()) {
    accumulate_closed_shell_trace_path_spatial_adjoint(
        cache,
        trace_adjoints,
        spatial_adjoint);
  }
}

bool is_effectively_zero(double value) {
  return std::abs(value) < 1.0e-15;
}

double* prepare_closed_shell_ggo_gradient_buffer(
    int n_active_orbitals,
    ScalarBuffer* ggo_grad) {
  if (ggo_grad == nullptr) {
    throw std::invalid_argument("ggo_grad must not be null");
  }

  const std::size_t expected_size =
      PfTensorContractor::packed_two_electron_count(n_active_orbitals);
  if (ggo_grad->empty()) {
    ggo_grad->assign(expected_size, 0.0);
  } else if (ggo_grad->size() != expected_size) {
    throw std::invalid_argument(
        "ggo_grad size does not match packed active-space tensor size");
  }
  return ggo_grad->data();
}

void validate_closed_shell_contraction_inputs(
    const ScalarBuffer& ggo,
    int n_active_orbitals,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right,
    const ConstMatrixRef& left_adjoint,
    const ConstMatrixRef& right_adjoint) {
  if (left.rows() != left.cols() ||
      right.rows() != right.cols() ||
      left_adjoint.rows() != left_adjoint.cols() ||
      right_adjoint.rows() != right_adjoint.cols()) {
    throw std::invalid_argument("closed-shell contraction matrices must be square");
  }
  if (left.rows() != n_active_orbitals ||
      right.rows() != n_active_orbitals ||
      left_adjoint.rows() != n_active_orbitals ||
      right_adjoint.rows() != n_active_orbitals) {
    throw std::invalid_argument(
        "closed-shell contraction dimensions do not match n_active_orbitals");
  }
  if (ggo.size() != PfTensorContractor::packed_two_electron_count(n_active_orbitals)) {
    throw std::invalid_argument(
        "ggo size does not match packed active-space tensor size");
  }
}

void accumulate_exchange_coulomb_adjoints(
    double scale,
    const ScalarBuffer& ggo,
    int n_active_orbitals,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right,
    MatrixRef left_adjoint,
    MatrixRef right_adjoint,
    ScalarBuffer* ggo_grad) {
  if (scale == 0.0) {
    return;
  }
  validate_closed_shell_contraction_inputs(
      ggo,
      n_active_orbitals,
      left,
      right,
      left_adjoint,
      right_adjoint);

  const double* ggo_data = ggo.data();
  double* ggo_grad_data =
      prepare_closed_shell_ggo_gradient_buffer(n_active_orbitals, ggo_grad);
  const PfPackedIndexCache& index_cache =
      get_packed_index_cache(n_active_orbitals);
  const double* left_data = left.data();
  const double* right_data = right.data();
  double* left_bar_data = left_adjoint.data();
  double* right_bar_data = right_adjoint.data();
  const Eigen::Index left_stride = left.outerStride();
  const Eigen::Index right_stride = right.outerStride();
  const Eigen::Index left_bar_stride = left_adjoint.outerStride();
  const Eigen::Index right_bar_stride = right_adjoint.outerStride();

  for (int q = 0; q < n_active_orbitals; ++q) {
    const double* right_col_q =
        right_data + static_cast<Eigen::Index>(q) * right_stride;
    double* right_bar_col_q =
        right_bar_data + static_cast<Eigen::Index>(q) * right_bar_stride;
    for (int r = 0; r < n_active_orbitals; ++r) {
      const double* left_col_r =
          left_data + static_cast<Eigen::Index>(r) * left_stride;
      double* left_bar_col_r =
          left_bar_data + static_cast<Eigen::Index>(r) * left_bar_stride;
      for (int s = 0; s < n_active_orbitals; ++s) {
        const double right_sq = right_col_q[s];
        const bool update_left = !is_effectively_zero(right_sq);
        const double w_right_sq = scale * right_sq;
        double right_bar_sq = 0.0;
        for (int p = 0; p < n_active_orbitals; ++p) {
          const std::size_t packed_index =
              quartet_index(index_cache, q, p, s, r);
          const double g = ggo_data[packed_index];
          if (is_effectively_zero(g)) {
            continue;
          }
          const double left_pr = left_col_r[p];
          if (update_left) {
            left_bar_col_r[p] += g * w_right_sq;
            ggo_grad_data[packed_index] += w_right_sq * left_pr;
          }
          if (!is_effectively_zero(left_pr)) {
            right_bar_sq += scale * g * left_pr;
          }
        }
        right_bar_col_q[s] += right_bar_sq;
      }
    }
  }

  for (int q = 0; q < n_active_orbitals; ++q) {
    for (int p = 0; p < n_active_orbitals; ++p) {
      const double* left_col_p =
          left_data + static_cast<Eigen::Index>(p) * left_stride;
      double* left_bar_col_p =
          left_bar_data + static_cast<Eigen::Index>(p) * left_bar_stride;
      for (int s = 0; s < n_active_orbitals; ++s) {
        const double right_qs =
            right_data[q + static_cast<Eigen::Index>(s) * right_stride];
        const bool update_left = !is_effectively_zero(right_qs);
        const double w_right_qs = scale * right_qs;
        double* right_bar_col_s =
            right_bar_data + static_cast<Eigen::Index>(s) * right_bar_stride;
        double right_bar_qs = 0.0;
        for (int r = 0; r < n_active_orbitals; ++r) {
          const std::size_t packed_index =
              quartet_index(index_cache, q, p, s, r);
          const double g = ggo_data[packed_index];
          if (is_effectively_zero(g)) {
            continue;
          }
          const double left_rp = left_col_p[r];
          if (update_left) {
            left_bar_col_p[r] += g * w_right_qs;
            ggo_grad_data[packed_index] += w_right_qs * left_rp;
          }
          if (!is_effectively_zero(left_rp)) {
            right_bar_qs += scale * g * left_rp;
          }
        }
        right_bar_col_s[q] += right_bar_qs;
      }
    }
  }
}

void accumulate_closed_shell_pair_term_adjoints(
    const ScalarBuffer& ggo,
    int n_active_orbitals,
    const ConstMatrixRef& b,
    const ConstMatrixRef& pair_term_matrix,
    MatrixRef pair_term_matrix_adjoint,
    ScalarBuffer* ggo_grad) {
  validate_closed_shell_contraction_inputs(
      ggo,
      n_active_orbitals,
      b,
      pair_term_matrix,
      pair_term_matrix_adjoint,
      pair_term_matrix_adjoint);

  const double* ggo_data = ggo.data();
  double* ggo_grad_data =
      prepare_closed_shell_ggo_gradient_buffer(n_active_orbitals, ggo_grad);
  const PfPackedIndexCache& index_cache =
      get_packed_index_cache(n_active_orbitals);
  const double* b_data = b.data();
  const double* m_data = pair_term_matrix.data();
  double* m_bar_data = pair_term_matrix_adjoint.data();
  const Eigen::Index b_stride = b.outerStride();
  const Eigen::Index m_stride = pair_term_matrix.outerStride();
  const Eigen::Index m_bar_stride = pair_term_matrix_adjoint.outerStride();

  for (int q = 0; q < n_active_orbitals; ++q) {
    const double* m_col_q = m_data + static_cast<Eigen::Index>(q) * m_stride;
    for (int s = 0; s < n_active_orbitals; ++s) {
      const double m_sq = m_col_q[s];
      const double m_qs = m_data[q + static_cast<Eigen::Index>(s) * m_stride];
      double* m_bar_col_q = m_bar_data + static_cast<Eigen::Index>(q) * m_bar_stride;
      for (int p = 0; p < n_active_orbitals; ++p) {
        const double* b_col_p = b_data + static_cast<Eigen::Index>(p) * b_stride;
        for (int r = 0; r < n_active_orbitals; ++r) {
          const std::size_t packed_index =
              quartet_index(index_cache, q, p, s, r);
          const double b_pr =
              b_data[p + static_cast<Eigen::Index>(r) * b_stride];
          const double b_rp = b_col_p[r];
          const double coefficient = b_pr * m_sq + b_rp * m_qs;
          if (!is_effectively_zero(coefficient)) {
            ggo_grad_data[packed_index] += coefficient;
          }
          const double g = ggo_data[packed_index];
          if (is_effectively_zero(g)) {
            continue;
          }
          m_bar_col_q[s] += g * b_pr;
          m_bar_data[q + static_cast<Eigen::Index>(s) * m_bar_stride] += g * b_rp;
        }
      }
    }
  }
}

void accumulate_closed_shell_full_linear_combo_adjoints(
    const ScalarBuffer& ggo,
    int n_active_orbitals,
    const ConstMatrixRef& b,
    const ConstMatrixRef& pair_term_matrix,
    const ConstMatrixRef& d_poly_h_n2,
    const ConstMatrixRef& a,
    const ConstMatrixRef& d,
    const ConstMatrixRef& opposite_bridge_h_split_matrix,
    const ConstMatrixRef& c_poly_h_n2,
    const ConstMatrixRef& c,
    const ConstMatrixRef& frechet_h_c_h_n2,
    MatrixRef pair_term_matrix_adjoint,
    MatrixRef d_poly_h_n2_adjoint,
    MatrixRef d_adjoint,
    MatrixRef opposite_bridge_h_split_matrix_adjoint,
    MatrixRef c_poly_h_n2_adjoint,
    MatrixRef c_adjoint,
    MatrixRef frechet_h_c_h_n2_adjoint,
    ScalarBuffer* ggo_grad) {
  validate_closed_shell_contraction_inputs(
      ggo,
      n_active_orbitals,
      b,
      pair_term_matrix,
      pair_term_matrix_adjoint,
      pair_term_matrix_adjoint);
  validate_closed_shell_contraction_inputs(
      ggo,
      n_active_orbitals,
      d_poly_h_n2,
      a,
      d_poly_h_n2_adjoint,
      d_poly_h_n2_adjoint);
  validate_closed_shell_contraction_inputs(
      ggo,
      n_active_orbitals,
      d,
      opposite_bridge_h_split_matrix,
      d_adjoint,
      opposite_bridge_h_split_matrix_adjoint);
  validate_closed_shell_contraction_inputs(
      ggo,
      n_active_orbitals,
      c_poly_h_n2,
      c,
      c_poly_h_n2_adjoint,
      c_adjoint);
  validate_closed_shell_contraction_inputs(
      ggo,
      n_active_orbitals,
      c,
      frechet_h_c_h_n2,
      c_adjoint,
      frechet_h_c_h_n2_adjoint);

  const double* ggo_data = ggo.data();
  double* ggo_grad_data =
      prepare_closed_shell_ggo_gradient_buffer(n_active_orbitals, ggo_grad);
  const PfPackedIndexCache& index_cache =
      get_packed_index_cache(n_active_orbitals);
  const double* b_data = b.data();
  const double* m_data = pair_term_matrix.data();
  const double* dr_data = d_poly_h_n2.data();
  const double* a_data = a.data();
  const double* d_data = d.data();
  const double* v_data = opposite_bridge_h_split_matrix.data();
  const double* x_data = c_poly_h_n2.data();
  const double* c_data = c.data();
  const double* u_data = frechet_h_c_h_n2.data();
  double* m_bar_data = pair_term_matrix_adjoint.data();
  double* dr_bar_data = d_poly_h_n2_adjoint.data();
  double* d_bar_data = d_adjoint.data();
  double* v_bar_data = opposite_bridge_h_split_matrix_adjoint.data();
  double* x_bar_data = c_poly_h_n2_adjoint.data();
  double* c_bar_data = c_adjoint.data();
  double* u_bar_data = frechet_h_c_h_n2_adjoint.data();
  const Eigen::Index b_stride = b.outerStride();
  const Eigen::Index m_stride = pair_term_matrix.outerStride();
  const Eigen::Index dr_stride = d_poly_h_n2.outerStride();
  const Eigen::Index a_stride = a.outerStride();
  const Eigen::Index d_stride = d.outerStride();
  const Eigen::Index v_stride = opposite_bridge_h_split_matrix.outerStride();
  const Eigen::Index x_stride = c_poly_h_n2.outerStride();
  const Eigen::Index c_stride = c.outerStride();
  const Eigen::Index u_stride = frechet_h_c_h_n2.outerStride();
  const Eigen::Index m_bar_stride = pair_term_matrix_adjoint.outerStride();
  const Eigen::Index dr_bar_stride = d_poly_h_n2_adjoint.outerStride();
  const Eigen::Index d_bar_stride = d_adjoint.outerStride();
  const Eigen::Index v_bar_stride = opposite_bridge_h_split_matrix_adjoint.outerStride();
  const Eigen::Index x_bar_stride = c_poly_h_n2_adjoint.outerStride();
  const Eigen::Index c_bar_stride = c_adjoint.outerStride();
  const Eigen::Index u_bar_stride = frechet_h_c_h_n2_adjoint.outerStride();

  for (int q = 0; q < n_active_orbitals; ++q) {
    const double* m_col_q = m_data + static_cast<Eigen::Index>(q) * m_stride;
    const double* a_col_q = a_data + static_cast<Eigen::Index>(q) * a_stride;
    const double* v_col_q = v_data + static_cast<Eigen::Index>(q) * v_stride;
    double* m_bar_col_q = m_bar_data + static_cast<Eigen::Index>(q) * m_bar_stride;
    for (int s = 0; s < n_active_orbitals; ++s) {
      const double m_sq = m_col_q[s];
      const double m_qs = m_data[q + static_cast<Eigen::Index>(s) * m_stride];
      const double a_sq = a_col_q[s];
      const double a_qs = a_data[q + static_cast<Eigen::Index>(s) * a_stride];
      const double v_sq = v_col_q[s];
      const double v_qs = v_data[q + static_cast<Eigen::Index>(s) * v_stride];
      for (int p = 0; p < n_active_orbitals; ++p) {
        const double x_qp =
            x_data[q + static_cast<Eigen::Index>(p) * x_stride];
        const double c_qp =
            c_data[q + static_cast<Eigen::Index>(p) * c_stride];
        const double* b_col_p = b_data + static_cast<Eigen::Index>(p) * b_stride;
        const double* dr_col_p = dr_data + static_cast<Eigen::Index>(p) * dr_stride;
        const double* d_col_p = d_data + static_cast<Eigen::Index>(p) * d_stride;
        double* dr_bar_col_p =
            dr_bar_data + static_cast<Eigen::Index>(p) * dr_bar_stride;
        double* d_bar_col_p =
            d_bar_data + static_cast<Eigen::Index>(p) * d_bar_stride;
        for (int r = 0; r < n_active_orbitals; ++r) {
          const std::size_t packed_index =
              quartet_index(index_cache, q, p, s, r);
          const double b_pr =
              b_data[p + static_cast<Eigen::Index>(r) * b_stride];
          const double b_rp = b_col_p[r];
          const double dr_pr =
              dr_data[p + static_cast<Eigen::Index>(r) * dr_stride];
          const double dr_rp = dr_col_p[r];
          const double d_pr =
              d_data[p + static_cast<Eigen::Index>(r) * d_stride];
          const double d_rp = d_col_p[r];
          const double c_sr =
              c_data[s + static_cast<Eigen::Index>(r) * c_stride];
          const double u_sr =
              u_data[s + static_cast<Eigen::Index>(r) * u_stride];
          const double coefficient =
              b_pr * m_sq + b_rp * m_qs -
              0.5 * (dr_pr * a_sq + dr_rp * a_qs + d_pr * v_sq + d_rp * v_qs) +
              x_qp * c_sr + c_qp * u_sr;
          if (!is_effectively_zero(coefficient)) {
            ggo_grad_data[packed_index] += coefficient;
          }

          const double g = ggo_data[packed_index];
          if (is_effectively_zero(g)) {
            continue;
          }

          m_bar_col_q[s] += g * b_pr;
          m_bar_data[q + static_cast<Eigen::Index>(s) * m_bar_stride] += g * b_rp;

          dr_bar_data[p + static_cast<Eigen::Index>(r) * dr_bar_stride] +=
              -0.5 * g * a_sq;
          dr_bar_col_p[r] += -0.5 * g * a_qs;

          d_bar_data[p + static_cast<Eigen::Index>(r) * d_bar_stride] +=
              -0.5 * g * v_sq;
          d_bar_col_p[r] += -0.5 * g * v_qs;

          v_bar_data[s + static_cast<Eigen::Index>(q) * v_bar_stride] +=
              -0.5 * g * d_pr;
          v_bar_data[q + static_cast<Eigen::Index>(s) * v_bar_stride] +=
              -0.5 * g * d_rp;

          x_bar_data[q + static_cast<Eigen::Index>(p) * x_bar_stride] += g * c_sr;
          c_bar_data[s + static_cast<Eigen::Index>(r) * c_bar_stride] += g * x_qp;

          c_bar_data[q + static_cast<Eigen::Index>(p) * c_bar_stride] += g * u_sr;
          u_bar_data[s + static_cast<Eigen::Index>(r) * u_bar_stride] += g * c_qp;
        }
      }
    }
  }
}

void accumulate_closed_shell_same_spin_asym_linear_combo_adjoints(
    const ScalarBuffer& ggo,
    int n_active_orbitals,
    const ConstMatrixRef& c_poly_h_n2,
    const ConstMatrixRef& c,
    const ConstMatrixRef& frechet_h_c_h_n2,
    MatrixRef c_poly_h_n2_adjoint,
    MatrixRef c_adjoint,
    MatrixRef frechet_h_c_h_n2_adjoint,
    ScalarBuffer* ggo_grad) {
  validate_closed_shell_contraction_inputs(
      ggo,
      n_active_orbitals,
      c_poly_h_n2,
      c,
      c_poly_h_n2_adjoint,
      c_adjoint);
  validate_closed_shell_contraction_inputs(
      ggo,
      n_active_orbitals,
      c,
      frechet_h_c_h_n2,
      c_adjoint,
      frechet_h_c_h_n2_adjoint);

  const double* ggo_data = ggo.data();
  double* ggo_grad_data =
      prepare_closed_shell_ggo_gradient_buffer(n_active_orbitals, ggo_grad);
  const double* x_data = c_poly_h_n2.data();
  const double* c_data = c.data();
  const double* u_data = frechet_h_c_h_n2.data();
  double* x_bar_data = c_poly_h_n2_adjoint.data();
  double* c_bar_data = c_adjoint.data();
  double* u_bar_data = frechet_h_c_h_n2_adjoint.data();
  const Eigen::Index x_stride = c_poly_h_n2.outerStride();
  const Eigen::Index c_stride = c.outerStride();
  const Eigen::Index u_stride = frechet_h_c_h_n2.outerStride();
  const Eigen::Index x_bar_stride = c_poly_h_n2_adjoint.outerStride();
  const Eigen::Index c_bar_stride = c_adjoint.outerStride();
  const Eigen::Index u_bar_stride = frechet_h_c_h_n2_adjoint.outerStride();

  for (int p = 0; p < n_active_orbitals - 1; ++p) {
    const double* c_col_p = c_data + static_cast<Eigen::Index>(p) * c_stride;
    const double* u_col_p = u_data + static_cast<Eigen::Index>(p) * u_stride;
    for (int q = 0; q < n_active_orbitals - 1; ++q) {
      const double x_qp =
          x_data[q + static_cast<Eigen::Index>(p) * x_stride];
      const double c_qp =
          c_data[q + static_cast<Eigen::Index>(p) * c_stride];
      const int direct_qp = packed_pair_index(q, p);
      for (int r = p + 1; r < n_active_orbitals; ++r) {
        const double x_qr =
            x_data[q + static_cast<Eigen::Index>(r) * x_stride];
        const double c_qr =
            c_data[q + static_cast<Eigen::Index>(r) * c_stride];
        const int exchange_qr = packed_pair_index(q, r);
        for (int s = q + 1; s < n_active_orbitals; ++s) {
          const int direct_sr = packed_pair_index(s, r);
          const int exchange_sp = packed_pair_index(s, p);
          const std::size_t direct_index =
              packed_pair_of_pairs_index(direct_qp, direct_sr);
          const std::size_t exchange_index =
              packed_pair_of_pairs_index(exchange_qr, exchange_sp);
          const double c_sr =
              c_data[s + static_cast<Eigen::Index>(r) * c_stride];
          const double c_sp = c_col_p[s];
          const double u_sr =
              u_data[s + static_cast<Eigen::Index>(r) * u_stride];
          const double u_sp = u_col_p[s];
          const double coefficient =
              2.0 * (x_qp * c_sr - x_qr * c_sp + c_qp * u_sr - c_qr * u_sp);
          if (!is_effectively_zero(coefficient)) {
            ggo_grad_data[direct_index] += coefficient;
            ggo_grad_data[exchange_index] -= coefficient;
          }

          const double interaction =
              ggo_data[direct_index] - ggo_data[exchange_index];
          if (is_effectively_zero(interaction)) {
            continue;
          }
          const double weighted_interaction = 2.0 * interaction;

          x_bar_data[q + static_cast<Eigen::Index>(p) * x_bar_stride] +=
              weighted_interaction * c_sr;
          x_bar_data[q + static_cast<Eigen::Index>(r) * x_bar_stride] +=
              -weighted_interaction * c_sp;

          c_bar_data[s + static_cast<Eigen::Index>(r) * c_bar_stride] +=
              weighted_interaction * x_qp;
          c_bar_data[s + static_cast<Eigen::Index>(p) * c_bar_stride] +=
              -weighted_interaction * x_qr;

          c_bar_data[q + static_cast<Eigen::Index>(p) * c_bar_stride] +=
              weighted_interaction * u_sr;
          c_bar_data[q + static_cast<Eigen::Index>(r) * c_bar_stride] +=
              -weighted_interaction * u_sp;

          u_bar_data[s + static_cast<Eigen::Index>(r) * u_bar_stride] +=
              weighted_interaction * c_qp;
          u_bar_data[s + static_cast<Eigen::Index>(p) * u_bar_stride] +=
              -weighted_interaction * c_qr;
        }
      }
    }
  }
}

void accumulate_cross_adjoints(
    const ScalarBuffer& ggo,
    int n_active_orbitals,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right,
    MatrixRef left_adjoint,
    MatrixRef right_adjoint,
    ScalarBuffer* ggo_grad) {
  validate_closed_shell_contraction_inputs(
      ggo,
      n_active_orbitals,
      left,
      right,
      left_adjoint,
      right_adjoint);

  const double* ggo_data = ggo.data();
  double* ggo_grad_data =
      prepare_closed_shell_ggo_gradient_buffer(n_active_orbitals, ggo_grad);
  const PfPackedIndexCache& index_cache =
      get_packed_index_cache(n_active_orbitals);
  const double* left_data = left.data();
  const double* right_data = right.data();
  double* left_bar_data = left_adjoint.data();
  double* right_bar_data = right_adjoint.data();
  const Eigen::Index left_stride = left.outerStride();
  const Eigen::Index right_stride = right.outerStride();
  const Eigen::Index left_bar_stride = left_adjoint.outerStride();
  const Eigen::Index right_bar_stride = right_adjoint.outerStride();

  for (int left_first = 0; left_first < n_active_orbitals - 1; ++left_first) {
    for (int right_first = 0; right_first < n_active_orbitals - 1; ++right_first) {
      const double left_entry =
          left_data[right_first + static_cast<Eigen::Index>(left_first) * left_stride];
      for (int left_second = left_first + 1;
           left_second < n_active_orbitals;
           ++left_second) {
        for (int right_second = right_first + 1;
             right_second < n_active_orbitals;
             ++right_second) {
          const double right_entry =
              right_data[right_second + static_cast<Eigen::Index>(left_second) *
                                             right_stride];
          const std::size_t direct_index =
              quartet_index(
                  index_cache,
                  right_first,
                  left_first,
                  right_second,
                  left_second);
          const std::size_t exchange_index =
              quartet_index(
                  index_cache,
                  right_first,
                  left_second,
                  right_second,
                  left_first);
          const double interaction =
              ggo_data[direct_index] - ggo_data[exchange_index];
          if (is_effectively_zero(interaction)) {
            continue;
          }
          if (!is_effectively_zero(right_entry)) {
            left_bar_data[right_first +
                          static_cast<Eigen::Index>(left_first) * left_bar_stride] +=
                2.0 * interaction * right_entry;
          }
          if (!is_effectively_zero(left_entry)) {
            right_bar_data[right_second +
                           static_cast<Eigen::Index>(left_second) *
                               right_bar_stride] +=
                2.0 * interaction * left_entry;
          }
          if (!is_effectively_zero(left_entry) &&
              !is_effectively_zero(right_entry)) {
            const double coeff = 2.0 * left_entry * right_entry;
            ggo_grad_data[direct_index] += coeff;
            ggo_grad_data[exchange_index] -= coeff;
          }
        }
      }
    }
  }

  for (int q = 0; q < n_active_orbitals; ++q) {
    for (int p = 0; p < n_active_orbitals; ++p) {
      const double left_qp =
          left_data[q + static_cast<Eigen::Index>(p) * left_stride];
      double left_bar_qp = 0.0;
      const double weighted_left = left_qp;
      for (int s = 0; s < n_active_orbitals; ++s) {
        for (int r = 0; r < n_active_orbitals; ++r) {
          const std::size_t packed_index =
              quartet_index(index_cache, q, p, s, r);
          const double g = ggo_data[packed_index];
          if (is_effectively_zero(g)) {
            continue;
          }
          const double right_sr =
              right_data[s + static_cast<Eigen::Index>(r) * right_stride];
          if (!is_effectively_zero(right_sr)) {
            left_bar_qp += g * right_sr;
          }
          if (!is_effectively_zero(left_qp)) {
            right_bar_data[s + static_cast<Eigen::Index>(r) * right_bar_stride] +=
                g * left_qp;
            ggo_grad_data[packed_index] += weighted_left * right_sr;
          }
        }
      }
      left_bar_data[q + static_cast<Eigen::Index>(p) * left_bar_stride] +=
          left_bar_qp;
    }
  }
}

void accumulate_same_spin_bridge_adjoints(
    double scale,
    const ScalarBuffer& ggo,
    int n_active_orbitals,
    const ConstMatrixRef& left,
    const ConstMatrixRef& right,
    MatrixRef left_adjoint,
    MatrixRef right_adjoint,
    ScalarBuffer* ggo_grad) {
  if (scale == 0.0) {
    return;
  }
  validate_closed_shell_contraction_inputs(
      ggo,
      n_active_orbitals,
      left,
      right,
      left_adjoint,
      right_adjoint);

  const double* ggo_data = ggo.data();
  double* ggo_grad_data =
      prepare_closed_shell_ggo_gradient_buffer(n_active_orbitals, ggo_grad);
  const PfPackedIndexCache& index_cache =
      get_packed_index_cache(n_active_orbitals);
  const double* left_data = left.data();
  const double* right_data = right.data();
  double* left_bar_data = left_adjoint.data();
  double* right_bar_data = right_adjoint.data();
  const Eigen::Index left_stride = left.outerStride();
  const Eigen::Index right_stride = right.outerStride();
  const Eigen::Index left_bar_stride = left_adjoint.outerStride();
  const Eigen::Index right_bar_stride = right_adjoint.outerStride();

  for (int left_first = 0; left_first < n_active_orbitals - 1; ++left_first) {
    for (int right_first = 0; right_first < n_active_orbitals - 1; ++right_first) {
      for (int left_second = left_first + 1;
           left_second < n_active_orbitals;
           ++left_second) {
        const double left_entry =
            left_data[right_first + static_cast<Eigen::Index>(left_second) *
                                         left_stride];
        for (int right_second = right_first + 1;
             right_second < n_active_orbitals;
             ++right_second) {
          const double right_entry =
              right_data[right_second + static_cast<Eigen::Index>(left_first) *
                                             right_stride];
          const std::size_t direct_index =
              quartet_index(
                  index_cache,
                  right_first,
                  left_first,
                  right_second,
                  left_second);
          const std::size_t exchange_index =
              quartet_index(
                  index_cache,
                  right_first,
                  left_second,
                  right_second,
                  left_first);
          const double interaction =
              ggo_data[direct_index] - ggo_data[exchange_index];
          if (is_effectively_zero(interaction)) {
            continue;
          }
          if (!is_effectively_zero(right_entry)) {
            left_bar_data[right_first +
                          static_cast<Eigen::Index>(left_second) * left_bar_stride] +=
                scale * interaction * right_entry;
          }
          if (!is_effectively_zero(left_entry)) {
            right_bar_data[right_second +
                           static_cast<Eigen::Index>(left_first) *
                               right_bar_stride] +=
                scale * interaction * left_entry;
          }
          if (!is_effectively_zero(left_entry) &&
              !is_effectively_zero(right_entry)) {
            const double coeff = scale * left_entry * right_entry;
            ggo_grad_data[direct_index] += coeff;
            ggo_grad_data[exchange_index] -= coeff;
          }
        }
      }
    }
  }
}

void accumulate_closed_shell_term(
    const PfKernelCache& cache,
    const ScalarBuffer& ggo,
    const PfTensorTerm& current_term,
    MatrixRef full_workspace,
    MatrixRef source_workspace,
    MatrixRef kernel_adjoint,
    MatrixRef sigma_adjoint,
    ScalarBuffer* ggo_grad) {
  const int n = cache.n_active_orbitals;
  Matrix left_operand_adjoint = Matrix::Zero(n, n);
  Matrix right_operand_adjoint = Matrix::Zero(n, n);
  const detail::MatrixBlock left_block =
      detail::materialize_operand_block(cache, current_term.left_operand);
  const detail::MatrixBlock right_block =
      detail::materialize_operand_block(cache, current_term.right_operand);

  switch (current_term.contraction) {
    case PfTensorContraction::Direct:
      PfTensorContractor::compute_direct_operand_adjoints(
          current_term.scale,
          left_block,
          right_block,
          ggo,
          left_operand_adjoint,
          right_operand_adjoint);
      break;
    case PfTensorContraction::Exchange:
      if (current_term.same_spin) {
        PfTensorContractor::compute_same_spin_exchange_operand_adjoints(
            current_term.scale,
            left_block,
            right_block,
            ggo,
            left_operand_adjoint,
            right_operand_adjoint);
      } else {
        PfTensorContractor::compute_exchange_operand_adjoints(
            current_term.scale,
            left_block,
            right_block,
            ggo,
            left_operand_adjoint,
            right_operand_adjoint);
      }
      break;
    case PfTensorContraction::Coulomb:
      if (current_term.same_spin) {
        PfTensorContractor::compute_same_spin_coulomb_operand_adjoints(
            current_term.scale,
            left_block,
            right_block,
            ggo,
            left_operand_adjoint,
            right_operand_adjoint);
      } else {
        PfTensorContractor::compute_coulomb_operand_adjoints(
            current_term.scale,
            left_block,
            right_block,
            ggo,
            left_operand_adjoint,
            right_operand_adjoint);
      }
      break;
    case PfTensorContraction::SameSpinSeparable:
      PfTensorContractor::compute_same_spin_separable_operand_adjoints(
          current_term.scale,
          left_block,
          right_block,
          ggo,
          left_operand_adjoint,
          right_operand_adjoint);
      break;
    case PfTensorContraction::SameSpinBridge:
      PfTensorContractor::compute_same_spin_bridge_operand_adjoints(
          current_term.scale,
          left_block,
          right_block,
          ggo,
          left_operand_adjoint,
          right_operand_adjoint);
      break;
  }

  accumulate_source_adjoint(
      cache,
      current_term.left_operand,
      left_operand_adjoint,
      full_workspace,
      source_workspace,
      kernel_adjoint,
      sigma_adjoint);
  accumulate_source_adjoint(
      cache,
      current_term.right_operand,
      right_operand_adjoint,
      full_workspace,
      source_workspace,
      kernel_adjoint,
      sigma_adjoint);
  accumulate_one_term_gradient(cache, current_term, ggo_grad);
}

void accumulate_closed_shell_two_electron_exact_terms(
    const PfKernelCache& cache,
    const ScalarBuffer& ggo,
    ScalarBuffer* ggo_grad,
    PfClosedShellTwoElectronAdjointWorkspace& workspace) {
  if (static_cast<int>(cache.projected_overlap_coefficients.size()) !=
      cache.trace_order + 1) {
    throw std::invalid_argument(
        "cache.projected_overlap_coefficients size does not match cache.trace_order + 1");
  }

  const int n = cache.n_active_orbitals;
  const int trace_order = cache.trace_order;
  if (trace_order <= 0) {
    return;
  }

  const Matrix& a = cache.closed_shell_left_ba_block;
  const Matrix& b = cache.closed_shell_right_ab_block;
  const Matrix& c = cache.closed_shell_pair_core;
  const Matrix& sa = cache.closed_shell_sa;
  const Matrix& h = cache.closed_shell_h;
  const Matrix& d = cache.closed_shell_d;
  const ScalarBuffer& pair_coefficients = cache.closed_shell_pair_coefficients;
  const ScalarBuffer& pair_tail_coefficients =
      cache.closed_shell_pair_tail_coefficients;
  const ScalarBuffer& coeff_n2 = cache.closed_shell_coeff_n2;
  const Matrix& pair_poly = cache.closed_shell_pair_poly;
  const Matrix& pair_term_matrix = cache.closed_shell_pair_term_matrix;
  const Matrix& c_poly_h_n2 = cache.closed_shell_c_poly_h_n2;
  const Matrix& frechet_h_c_n2 = cache.closed_shell_frechet_h_c_n2;
  const Matrix& frechet_h_c_h_n2 = cache.closed_shell_frechet_h_c_h_n2;
  const Matrix& d_poly_h_n2 = cache.closed_shell_d_poly_h_n2;
  const Matrix& opposite_bridge_h_split_matrix =
      cache.closed_shell_opposite_bridge_h_split_matrix;

  Matrix& pair_term_matrix_adjoint = workspace.pair_term_matrix_adjoint;
  Matrix& c_poly_h_n2_adjoint = workspace.c_poly_h_n2_adjoint;
  Matrix& c_adjoint = workspace.c_adjoint;
  Matrix& frechet_h_c_h_n2_adjoint = workspace.frechet_h_c_h_n2_adjoint;
  Matrix& d_poly_h_n2_adjoint = workspace.d_poly_h_n2_adjoint;
  Matrix& d_adjoint = workspace.d_adjoint;
  Matrix& opposite_bridge_h_split_matrix_adjoint =
      workspace.opposite_bridge_h_split_matrix_adjoint;

  if (coeff_n2.empty()) {
    accumulate_closed_shell_pair_term_adjoints(
        ggo,
        n,
        b,
        pair_term_matrix,
        pair_term_matrix_adjoint,
        ggo_grad);
  } else {
    accumulate_closed_shell_full_linear_combo_adjoints(
        ggo,
        n,
        b,
        pair_term_matrix,
        d_poly_h_n2,
        a,
        d,
        opposite_bridge_h_split_matrix,
        c_poly_h_n2,
        c,
        frechet_h_c_h_n2,
        pair_term_matrix_adjoint,
        d_poly_h_n2_adjoint,
        d_adjoint,
        opposite_bridge_h_split_matrix_adjoint,
        c_poly_h_n2_adjoint,
        c_adjoint,
        frechet_h_c_h_n2_adjoint,
        ggo_grad);
    accumulate_closed_shell_same_spin_asym_linear_combo_adjoints(
        ggo,
        n,
        c_poly_h_n2,
        c,
        frechet_h_c_h_n2,
        c_poly_h_n2_adjoint,
        c_adjoint,
        frechet_h_c_h_n2_adjoint,
        ggo_grad);
  }

  ScalarBuffer& pair_coefficient_adjoints = workspace.pair_coefficient_adjoints;
  Matrix& pair_poly_adjoint = workspace.pair_poly_adjoint;
  if (!pair_coefficients.empty()) {
    pair_coefficient_adjoints.front() +=
        frobenius_inner_product(pair_term_matrix_adjoint, a);
  }
  if (!pair_tail_coefficients.empty()) {
    c_adjoint.noalias() += pair_term_matrix_adjoint * pair_poly.transpose();
    pair_poly_adjoint.noalias() += c.transpose() * pair_term_matrix_adjoint;
  }

  ScalarBuffer& coeff_n2_adjoints = workspace.coeff_n2_adjoints;
  Matrix& frechet_h_c_n2_adjoint = workspace.frechet_h_c_n2_adjoint;
  Matrix& h_adjoint = workspace.h_adjoint;
  Matrix& sa_adjoint = workspace.sa_adjoint;

  if (!coeff_n2.empty()) {
    frechet_h_c_n2_adjoint.noalias() +=
        frechet_h_c_h_n2_adjoint * h.transpose();
    h_adjoint.noalias() +=
        frechet_h_c_n2.transpose() * frechet_h_c_h_n2_adjoint;

    frechet_h_c_n2_adjoint.noalias() +=
        opposite_bridge_h_split_matrix_adjoint * sa.transpose();
    sa_adjoint.noalias() +=
        frechet_h_c_n2.transpose() * opposite_bridge_h_split_matrix_adjoint;

    const MatrixPolynomialAdjointResult d_poly_result =
        backpropagate_right_matrix_polynomial(
            h,
            coeff_n2,
            d,
            d_poly_h_n2_adjoint);
    h_adjoint.noalias() += d_poly_result.kernel_adjoint;
    d_adjoint.noalias() += d_poly_result.source_adjoint;
    h_adjoint.noalias() += b.transpose() * d_adjoint;
    for (std::size_t index = 0; index < coeff_n2_adjoints.size(); ++index) {
      coeff_n2_adjoints[index] += d_poly_result.coefficient_adjoints[index];
    }

    const MatrixPolynomialAdjointResult c_poly_result =
        backpropagate_right_matrix_polynomial(
            h,
            coeff_n2,
            c,
            c_poly_h_n2_adjoint);
    h_adjoint.noalias() += c_poly_result.kernel_adjoint;
    c_adjoint.noalias() += c_poly_result.source_adjoint;
    for (std::size_t index = 0; index < coeff_n2_adjoints.size(); ++index) {
      coeff_n2_adjoints[index] += c_poly_result.coefficient_adjoints[index];
    }

    const MatrixPolynomialAdjointResult frechet_result =
        backpropagate_left_matrix_polynomial_frechet(
            h,
            coeff_n2,
            c,
            frechet_h_c_n2_adjoint);
    h_adjoint.noalias() += frechet_result.kernel_adjoint;
    c_adjoint.noalias() += frechet_result.source_adjoint;
    for (std::size_t index = 0; index < coeff_n2_adjoints.size(); ++index) {
      coeff_n2_adjoints[index] += frechet_result.coefficient_adjoints[index];
    }
  }

  if (!pair_tail_coefficients.empty()) {
    const MatrixPolynomialAdjointResult pair_poly_result =
        backpropagate_left_matrix_polynomial(
            h,
            pair_tail_coefficients,
            sa,
            pair_poly_adjoint);
    h_adjoint.noalias() += pair_poly_result.kernel_adjoint;
    sa_adjoint.noalias() += pair_poly_result.source_adjoint;
    for (std::size_t index = 0; index < pair_poly_result.coefficient_adjoints.size(); ++index) {
      pair_coefficient_adjoints[index + 1] +=
          pair_poly_result.coefficient_adjoints[index];
    }
  }

  ScalarBuffer& overlap_coefficient_adjoints =
      workspace.overlap_coefficient_adjoints;
  if (!pair_coefficient_adjoints.empty()) {
    accumulate_projected_coefficient_family_adjoints(
        trace_order - 1,
        0.5,
        pair_coefficient_adjoints,
        &overlap_coefficient_adjoints);
  }
  if (!coeff_n2_adjoints.empty()) {
    accumulate_projected_coefficient_family_adjoints(
        trace_order - 2,
        1.0,
        coeff_n2_adjoints,
        &overlap_coefficient_adjoints);
  }
}

}  // namespace

Matrix PfAdjointKernel::build_kernel_adjoint(
    const PfKernelCache& cache,
    const ScalarBuffer& ggo,
    const std::vector<PfTensorTerm>& terms,
    bool include_trace_path) {
  Matrix kernel_adjoint =
      include_trace_path
          ? build_trace_path_kernel_adjoint(cache)
          : Matrix::Zero(cache.n_spin_orbitals, cache.n_spin_orbitals);
  Matrix sigma_dummy = Matrix::Zero(cache.n_spin_orbitals, cache.n_spin_orbitals);
  accumulate_tensor_path_adjoints(
      cache,
      ggo,
      terms,
      kernel_adjoint,
      sigma_dummy);
  return kernel_adjoint;
}

Matrix PfAdjointKernel::build_sigma_adjoint(
    const PfKernelCache& cache,
    const ScalarBuffer& ggo,
    const std::vector<PfTensorTerm>& terms,
    bool include_trace_path) {
  Matrix kernel_adjoint =
      include_trace_path
          ? build_trace_path_kernel_adjoint(cache)
          : Matrix::Zero(cache.n_spin_orbitals, cache.n_spin_orbitals);
  Matrix sigma_adjoint = Matrix::Zero(cache.n_spin_orbitals, cache.n_spin_orbitals);
  accumulate_tensor_path_adjoints(
      cache,
      ggo,
      terms,
      kernel_adjoint,
      sigma_adjoint);
  sigma_adjoint.noalias() +=
      backpropagate_sigma(cache.left, cache.sigma, cache.right, kernel_adjoint);
  return sigma_adjoint;
}

Matrix PfAdjointKernel::build_spatial_density(
    const PfKernelCache& cache,
    const ScalarBuffer& ggo,
    const std::vector<PfTensorTerm>& terms,
    bool include_trace_path) {
  return collapse_spin_diagonal_blocks(
      build_sigma_adjoint(cache, ggo, terms, include_trace_path));
}

Matrix PfAdjointKernel::build_one_rdm_source_sigma_adjoint(
    const PfKernelCache& cache,
    const ConstMatrixRef& one_rdm_adjoint) {
  if (one_rdm_adjoint.rows() != cache.n_active_orbitals ||
      one_rdm_adjoint.cols() != cache.n_active_orbitals) {
    throw std::invalid_argument(
        "one_rdm_adjoint does not match the active-space dimension");
  }

  const int n = cache.n_active_orbitals;
  const int trace_order = cache.trace_order;
  Matrix sigma_adjoint =
      Matrix::Zero(cache.n_spin_orbitals, cache.n_spin_orbitals);

  if (trace_order <= 0) {
    return sigma_adjoint;
  }

  if (!cache.closed_shell_trace_powers.empty() &&
      cache.closed_shell_spatial_overlap.rows() == n &&
      cache.closed_shell_spatial_overlap.cols() == n &&
      cache.closed_shell_left_ba_block.rows() == n &&
      cache.closed_shell_left_ba_block.cols() == n &&
      cache.closed_shell_right_ab_block.rows() == n &&
      cache.closed_shell_right_ab_block.cols() == n &&
      cache.closed_shell_spatial_kernel.rows() == n &&
      cache.closed_shell_spatial_kernel.cols() == n &&
      cache.closed_shell_pair_core.rows() == n &&
      cache.closed_shell_pair_core.cols() == n) {
    Matrix h_adjoint = Matrix::Zero(n, n);
    Matrix c_adjoint = Matrix::Zero(n, n);
    Matrix sa_adjoint = Matrix::Zero(n, n);
    Matrix spatial_adjoint = Matrix::Zero(n, n);
    ScalarBuffer overlap_coefficient_adjoints(
        xmvb::to_size(trace_order + 1),
        0.0);

    accumulate_closed_shell_one_rdm_exact_terms(
        cache,
        one_rdm_adjoint,
        h_adjoint,
        c_adjoint,
        &overlap_coefficient_adjoints);
    finalize_closed_shell_exact_spatial_adjoint(
        cache,
        h_adjoint,
        c_adjoint,
        sa_adjoint,
        overlap_coefficient_adjoints,
        spatial_adjoint);
    accumulate_closed_shell_spatial_sigma_adjoint(
        spatial_adjoint,
        sigma_adjoint);
    return sigma_adjoint;
  }

  Matrix kernel_adjoint =
      Matrix::Zero(cache.n_spin_orbitals, cache.n_spin_orbitals);
  const ScalarBuffer density_coefficients =
      build_closed_shell_projected_coefficients(
          cache,
          trace_order - 1,
          2.0);
  const Matrix kernel_block =
      copy_block_matrix(cache, cache.kernel, PfSpinBlock::AlphaAlpha);
  const Matrix source_block =
      copy_block_matrix(cache, cache.left_sigma_right, PfSpinBlock::AlphaAlpha);
  const MatrixPolynomialAdjointResult one_rdm_result =
      backpropagate_left_matrix_polynomial(
          kernel_block,
          density_coefficients,
          source_block,
          one_rdm_adjoint);

  kernel_adjoint.topLeftCorner(n, n).noalias() +=
      one_rdm_result.kernel_adjoint;

  Matrix source_adjoint =
      Matrix::Zero(cache.n_spin_orbitals, cache.n_spin_orbitals);
  write_spin_block(
      PfSpinBlock::AlphaAlpha,
      one_rdm_result.source_adjoint,
      source_adjoint);
  sigma_adjoint.noalias() +=
      cache.left.transpose() *
      source_adjoint *
      cache.right.transpose();

  ScalarBuffer overlap_coefficient_adjoints(
      xmvb::to_size(trace_order + 1),
      0.0);
  accumulate_projected_coefficient_family_adjoints(
      trace_order - 1,
      2.0,
      one_rdm_result.coefficient_adjoints,
      &overlap_coefficient_adjoints);
  const ScalarBuffer trace_adjoints =
      TraceProjector::backpropagate_overlap_coefficient_adjoints(
          cache.traces,
          trace_order,
          overlap_coefficient_adjoints);
  if (!trace_adjoints.empty()) {
    kernel_adjoint.noalias() +=
        build_kernel_trace_adjoint(cache.kernel_powers, trace_adjoints);
  }

  sigma_adjoint.noalias() +=
      backpropagate_sigma(cache.left, cache.sigma, cache.right, kernel_adjoint);
  return sigma_adjoint;
}

ScalarBuffer PfAdjointKernel::build_tensor_gradient(
    const PfKernelCache& cache,
    const std::vector<PfTensorTerm>& terms) {
  ScalarBuffer ggo_grad;
  for (const PfTensorTerm& term : terms) {
    accumulate_one_term_gradient(cache, term, &ggo_grad);
  }
  return ggo_grad;
}

PfAdjointResult PfAdjointKernel::evaluate(
    const PfKernelCache& cache,
    const ScalarBuffer& ggo,
    const std::vector<PfTensorTerm>& terms,
    bool include_trace_path) {
  PfAdjointResult result;
  result.kernel_adjoint =
      include_trace_path
          ? build_trace_path_kernel_adjoint(cache)
          : Matrix::Zero(cache.n_spin_orbitals, cache.n_spin_orbitals);
  result.sigma_adjoint = Matrix::Zero(cache.n_spin_orbitals, cache.n_spin_orbitals);
  accumulate_tensor_path_adjoints(
      cache,
      ggo,
      terms,
      result.kernel_adjoint,
      result.sigma_adjoint);
  result.sigma_adjoint.noalias() +=
      backpropagate_sigma(cache.left, cache.sigma, cache.right, result.kernel_adjoint);
  result.spatial_density = collapse_spin_diagonal_blocks(result.sigma_adjoint);
  result.ggo_grad = build_tensor_gradient(cache, terms);
  return result;
}

PfAdjointResult PfAdjointKernel::evaluate_closed_shell_two_electron(
    const PfKernelCache& cache,
    const ScalarBuffer& ggo) {
  PfAdjointResult result;
  Matrix full_workspace;
  Matrix source_workspace;
  evaluate_closed_shell_two_electron_inplace(
      cache,
      ggo,
      &full_workspace,
      &source_workspace,
      &result.kernel_adjoint,
      &result.sigma_adjoint,
      &result.spatial_density,
      &result.ggo_grad);
  return result;
}

void PfAdjointKernel::evaluate_closed_shell_two_electron_inplace(
    const PfKernelCache& cache,
    const ScalarBuffer& ggo,
    Matrix* full_workspace,
    Matrix* source_workspace,
    Matrix* kernel_adjoint,
    Matrix* sigma_adjoint,
    Matrix* spatial_density,
    ScalarBuffer* ggo_grad,
    PfClosedShellTwoElectronAdjointWorkspace* workspace) {
  if (full_workspace == nullptr ||
      source_workspace == nullptr ||
      kernel_adjoint == nullptr ||
      sigma_adjoint == nullptr ||
      spatial_density == nullptr ||
      ggo_grad == nullptr) {
    throw std::invalid_argument(
        "closed-shell in-place adjoint outputs must not be null");
  }

  full_workspace->resize(cache.n_spin_orbitals, cache.n_spin_orbitals);
  full_workspace->setZero();
  source_workspace->resize(cache.n_spin_orbitals, cache.n_spin_orbitals);
  source_workspace->setZero();
  kernel_adjoint->resize(cache.n_spin_orbitals, cache.n_spin_orbitals);
  kernel_adjoint->setZero();
  sigma_adjoint->resize(cache.n_spin_orbitals, cache.n_spin_orbitals);
  sigma_adjoint->setZero();
  spatial_density->resize(cache.n_active_orbitals, cache.n_active_orbitals);
  spatial_density->setZero();
  const std::size_t ggo_size =
      PfTensorContractor::packed_two_electron_count(cache.n_active_orbitals);
  if (ggo_grad->size() != ggo_size) {
    ggo_grad->assign(ggo_size, 0.0);
  } else {
    std::fill(ggo_grad->begin(), ggo_grad->end(), 0.0);
  }

  PfClosedShellTwoElectronAdjointWorkspace local_workspace;
  PfClosedShellTwoElectronAdjointWorkspace* active_workspace =
      workspace != nullptr ? workspace : &local_workspace;
  prepare_closed_shell_two_electron_workspace(cache, active_workspace);

  accumulate_closed_shell_two_electron_exact_terms(
      cache,
      ggo,
      ggo_grad,
      *active_workspace);
  finalize_closed_shell_left_aux_spatial_adjoint(
      cache,
      active_workspace->h_adjoint,
      active_workspace->c_adjoint,
      active_workspace->sa_adjoint,
      *spatial_density);
  accumulate_closed_shell_overlap_coefficient_spatial_adjoint(
      cache,
      active_workspace->overlap_coefficient_adjoints,
      *spatial_density);
  accumulate_closed_shell_spatial_sigma_adjoint(*spatial_density, *sigma_adjoint);
}

void PfAdjointKernel::evaluate_closed_shell_hamiltonian_inplace(
    const PfKernelCache& cache,
    const ConstMatrixRef& one_rdm_adjoint,
    const ScalarBuffer& ggo,
    Matrix* spatial_density,
    ScalarBuffer* ggo_grad,
    PfClosedShellTwoElectronAdjointWorkspace* workspace) {
  if (spatial_density == nullptr || ggo_grad == nullptr) {
    throw std::invalid_argument(
        "closed-shell Hamiltonian adjoint outputs must not be null");
  }

  if (one_rdm_adjoint.rows() != cache.n_active_orbitals ||
      one_rdm_adjoint.cols() != cache.n_active_orbitals) {
    throw std::invalid_argument(
        "one_rdm_adjoint does not match the active-space dimension");
  }

  spatial_density->resize(cache.n_active_orbitals, cache.n_active_orbitals);
  spatial_density->setZero();
  const std::size_t ggo_size =
      PfTensorContractor::packed_two_electron_count(cache.n_active_orbitals);
  if (ggo_grad->size() != ggo_size) {
    ggo_grad->assign(ggo_size, 0.0);
  } else {
    std::fill(ggo_grad->begin(), ggo_grad->end(), 0.0);
  }

  // The closed-shell one-electron and two-electron exact paths propagate
  // through different helper kernels:
  // - one_rdm uses the right-multiplied spatial kernel `H = C * S`
  // - the two-electron fast path uses the left-multiplied auxiliary `h = S * C`
  // Reusing one shared `(h_adjoint, c_adjoint)` workspace mixes those two
  // semantics. Assemble the total response from the already validated split
  // adjoints instead.
  const Matrix one_electron_sigma_adjoint =
      build_one_rdm_source_sigma_adjoint(cache, one_rdm_adjoint);
  const PfAdjointResult two_electron_adjoint =
      evaluate_closed_shell_two_electron(cache, ggo);
  *spatial_density = collapse_spin_diagonal_blocks(one_electron_sigma_adjoint);
  spatial_density->noalias() += two_electron_adjoint.spatial_density;
  *ggo_grad = two_electron_adjoint.ggo_grad;
}

}  // namespace xmvb::pfaffian_vbscf
