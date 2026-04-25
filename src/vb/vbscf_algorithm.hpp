#pragma once

namespace xmvb::vb {

enum class VBSCFAlgorithm {
  Original,
};

inline const char* vb_scf_algorithm_name(VBSCFAlgorithm algorithm) {
  switch (algorithm) {
    case VBSCFAlgorithm::Original:
      return "original";
  }
  return "unknown";
}

}  // namespace xmvb::vb
