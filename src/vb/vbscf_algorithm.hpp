#pragma once

namespace xmvb::vb {

enum class VbScfAlgorithm {
  Original,
  Biorthogonal,
};

inline const char* vb_scf_algorithm_name(VbScfAlgorithm algorithm) {
  switch (algorithm) {
    case VbScfAlgorithm::Original:
      return "original";
    case VbScfAlgorithm::Biorthogonal:
      return "biorthogonal";
  }
  return "unknown";
}

}  // namespace xmvb::vb
