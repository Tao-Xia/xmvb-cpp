#pragma once

namespace xmvb::vb {

inline constexpr int kOrbitalTypeGen = 0;
inline constexpr int kOrbitalTypeHao = 1;
inline constexpr int kOrbitalTypeBdo = 2;
inline constexpr int kOrbitalTypeOeo = 3;

inline constexpr int kFragmentTypeAtom = 0;
inline constexpr int kFragmentTypeSao = 1;

inline constexpr int kWavefunctionTypeStructure = 0;
inline constexpr int kWavefunctionTypeDeterminant = 1;

inline constexpr int kVbFunctionTypeDeterminant = 0;
inline constexpr int kVbFunctionTypePerfectPairingDeterminant = 1;

inline constexpr int kGuessTypeAuto = 0;
inline constexpr int kGuessTypeUnit = 1;
inline constexpr int kGuessTypeRead = 2;
inline constexpr int kGuessTypeRdci = 3;
inline constexpr int kGuessTypeMo = 4;
inline constexpr int kGuessTypeNbo = 5;

}  // namespace xmvb::vb
