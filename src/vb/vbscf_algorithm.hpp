#pragma once

namespace xmvb::vb {

enum class VBSCFAlgorithm {
  Original,
  BiorthogonalExactSelected,
};

inline const char* vb_scf_algorithm_name(VBSCFAlgorithm algorithm) {
  switch (algorithm) {
    case VBSCFAlgorithm::Original:
      return "original";
    case VBSCFAlgorithm::BiorthogonalExactSelected:
      return "tbvbscf";
  }
  return "unknown";
}

}  // namespace xmvb::vb
