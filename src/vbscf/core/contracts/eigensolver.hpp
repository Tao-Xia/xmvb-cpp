#pragma once

namespace xmvb::vb {

/** @brief Structure-space generalized eigensolver used during optimization. */
enum class StructureEigensolver {
  Davidson,
  Dense,
};

/** @brief Returns the stable input/output name of a structure eigensolver. */
inline const char* structure_eigensolver_name(StructureEigensolver solver) {
  switch (solver) {
    case StructureEigensolver::Davidson:
      return "davidson";
    case StructureEigensolver::Dense:
      return "dense";
  }
  return "unknown";
}

}  // namespace xmvb::vb
