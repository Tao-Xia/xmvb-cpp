#pragma once

namespace xmvb::vb {

enum class VbScfAlgorithm {
  Original,
};

// Temporary source-compatibility alias for the excluded DeepVBH integration.
using VBSCFAlgorithm = VbScfAlgorithm;

inline const char* vb_scf_algorithm_name(VbScfAlgorithm algorithm) {
  switch (algorithm) {
    case VbScfAlgorithm::Original:
      return "original";
  }
  return "unknown";
}

}  // namespace xmvb::vb
