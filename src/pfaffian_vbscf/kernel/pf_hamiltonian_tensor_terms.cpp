#include "pfaffian_vbscf/kernel/pf_hamiltonian_tensor_terms.hpp"

#include <stdexcept>

namespace xmvb::pfaffian_vbscf {

namespace {

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

}  // namespace

std::vector<PfTensorTerm> build_two_electron_hamiltonian_tensor_terms(
    const PfKernelCache& cache) {
  if (static_cast<int>(cache.trace_weights.size()) != cache.trace_order) {
    throw std::invalid_argument("cache.trace_weights size does not match cache.trace_order");
  }
  if (cache.one_rdm.rows() != cache.n_active_orbitals ||
      cache.one_rdm.cols() != cache.n_active_orbitals) {
    throw std::invalid_argument("cache.one_rdm does not match the active-space dimension");
  }
  if (static_cast<int>(cache.trace_rdms.size()) != cache.trace_order) {
    throw std::invalid_argument("cache.trace_rdms size does not match cache.trace_order");
  }
  for (const Matrix& trace_rdm : cache.trace_rdms) {
    if (trace_rdm.rows() != cache.n_spin_orbitals ||
        trace_rdm.cols() != cache.n_spin_orbitals) {
      throw std::invalid_argument(
          "cache.trace_rdms entry does not match the spin-orbital dimension");
    }
  }

  std::vector<PfTensorTerm> terms;
  terms.reserve(
      3 * cache.trace_order * cache.trace_order +
      6 * cache.trace_order * cache.trace_order);

  for (int p_idx = 0; p_idx < cache.trace_order; ++p_idx) {
    for (int q_idx = 0; q_idx < cache.trace_order; ++q_idx) {
      const int p = p_idx + 1;
      const int q = q_idx + 1;
      const int combined_order = p + q;
      if (combined_order > cache.trace_order) {
        continue;
      }

      const double w_combined =
          cache.trace_weights[combined_order - 1];
      if (w_combined == 0.0) {
        continue;
      }

      const double cross_weight =
          -static_cast<double>(combined_order) /
          (2.0 * static_cast<double>(p) * static_cast<double>(q)) *
          w_combined;
      terms.push_back(
          term(
              PfTensorContraction::SameSpinSeparable,
              operand(PfMatrixSource::TraceRDM, PfSpinBlock::AlphaAlpha, p_idx),
              operand(PfMatrixSource::TraceRDM, PfSpinBlock::AlphaAlpha, q_idx),
              cross_weight));
      terms.push_back(
          term(
              PfTensorContraction::SameSpinSeparable,
              operand(PfMatrixSource::TraceRDM, PfSpinBlock::BetaBeta, p_idx),
              operand(PfMatrixSource::TraceRDM, PfSpinBlock::BetaBeta, q_idx),
              cross_weight));
      terms.push_back(
          term(
              PfTensorContraction::Direct,
              operand(PfMatrixSource::TraceRDM, PfSpinBlock::AlphaAlpha, p_idx),
              operand(PfMatrixSource::TraceRDM, PfSpinBlock::BetaBeta, q_idx),
              cross_weight));
    }
  }

  for (int power = 0; power < cache.trace_order; ++power) {
    const double scale =
        cache.trace_weights[power] *
        static_cast<double>(power + 1);
    if (scale == 0.0) {
      continue;
    }

    terms.push_back(
        term(
            PfTensorContraction::Exchange,
            operand(PfMatrixSource::Right, PfSpinBlock::AlphaBeta),
            operand(PfMatrixSource::KernelPowerTimesLeft, PfSpinBlock::BetaAlpha, power),
            scale));
    terms.push_back(
        term(
            PfTensorContraction::Coulomb,
            operand(PfMatrixSource::Right, PfSpinBlock::BetaAlpha),
            operand(PfMatrixSource::KernelPowerTimesLeft, PfSpinBlock::AlphaBeta, power),
            scale));

    if (power == 0) {
      continue;
    }

    for (int split_power = 0; split_power <= power - 1; ++split_power) {
      const int tail_power = power - 1 - split_power;
      terms.push_back(
          term(
              PfTensorContraction::SameSpinBridge,
              operand(
                  PfMatrixSource::KernelPowerTimesLeftSigmaRight,
                  PfSpinBlock::AlphaAlpha,
                  split_power),
              operand(
                  PfMatrixSource::KernelPowerTimesLeftSigmaRight,
                  PfSpinBlock::AlphaAlpha,
                  tail_power),
              2.0 * scale));
      terms.push_back(
          term(
              PfTensorContraction::SameSpinBridge,
              operand(
                  PfMatrixSource::KernelPowerTimesLeftSigmaRight,
                  PfSpinBlock::BetaBeta,
                  split_power),
              operand(
                  PfMatrixSource::KernelPowerTimesLeftSigmaRight,
                  PfSpinBlock::BetaBeta,
                  tail_power),
              2.0 * scale));
      terms.push_back(
          term(
              PfTensorContraction::Exchange,
              operand(
                  PfMatrixSource::RightSigmaTimesKernelPowerTimesLeftSigmaRight,
                  PfSpinBlock::AlphaBeta,
                  split_power),
              operand(
                  PfMatrixSource::KernelPowerTimesLeft,
                  PfSpinBlock::BetaAlpha,
                  tail_power),
              scale));
      terms.push_back(
          term(
              PfTensorContraction::Coulomb,
              operand(
                  PfMatrixSource::RightSigmaTimesKernelPowerTimesLeftSigmaRight,
                  PfSpinBlock::BetaAlpha,
                  tail_power),
              operand(
                  PfMatrixSource::KernelPowerTimesLeft,
                  PfSpinBlock::AlphaBeta,
                  split_power),
              scale));
    }
  }

  return terms;
}

}  // namespace xmvb::pfaffian_vbscf
