#pragma once

namespace xmvb::vb {

enum class VBSCFAlgorithm {
  Original,
  Biorthogonal,
};

inline const char* vb_scf_algorithm_name(VBSCFAlgorithm algorithm) {
  switch (algorithm) {
    case VBSCFAlgorithm::Original:
      return "original";
    case VBSCFAlgorithm::Biorthogonal:
      return "biorthogonal";
  }
  return "unknown";
}

}  // namespace xmvb::vb
