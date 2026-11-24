#pragma once

namespace xmvb::vb {

/** Fixed-bond model configuration, independent of any tensor-library headers. */
struct TnVbConfiguration {
  bool enabled = false;
  int structure_bond = 0;
  int total_bond = 0;
};

}  // namespace xmvb::vb
