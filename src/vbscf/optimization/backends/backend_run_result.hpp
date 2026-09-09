#pragma once

namespace xmvb::vb::optimizer_detail {

struct BackendRunResult {
  int n_iterations = 0;
  double final_gradient_l2_norm = 0.0;
  bool final_projected_gradient_ready = false;
};

}  // namespace xmvb::vb::optimizer_detail
