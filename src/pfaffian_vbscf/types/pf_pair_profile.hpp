#pragma once

namespace xmvb::pfaffian_vbscf {

/**
 * @brief Lightweight timing counters for the current Pfaffian molecule tests.
 *
 * The existing diagnostic tools print these fields, so the struct preserves the
 * expected surface area even though the current implementation only populates a
 * subset of the counters.
 */
struct PfPairProfile {
  int forward_pair_calls = 0;
  int forward_sample_calls = 0;
  double forward_pair_total_dt = 0.0;
  double forward_source_dt = 0.0;
  double forward_same_spin_dt = 0.0;
  double forward_mix_dt = 0.0;

  int grad_pair_calls = 0;
  int grad_sample_calls = 0;
  double grad_pair_total_dt = 0.0;
  double grad_source_dt = 0.0;
  double grad_same_spin_dt = 0.0;
  double grad_mix_dt = 0.0;
  double grad_sso_dt = 0.0;
};

}  // namespace xmvb::pfaffian_vbscf
