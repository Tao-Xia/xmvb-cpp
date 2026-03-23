#pragma once

namespace xmvb::vb {

enum class WavefunctionType {
  Structure = 0,
  Determinant = 1,
};

enum class VbFunctionType {
  Determinant = 0,
  PerfectPairing = 1,
};

inline constexpr int wavefunction_type_code(
    WavefunctionType wavefunction_type) {
  return static_cast<int>(wavefunction_type);
}

inline constexpr int vb_function_type_code(
    VbFunctionType vb_function_type) {
  return static_cast<int>(vb_function_type);
}

}  // namespace xmvb::vb
