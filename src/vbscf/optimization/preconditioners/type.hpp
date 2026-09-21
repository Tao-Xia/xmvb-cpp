#pragma once

namespace xmvb::vb {

/** @brief Orbital model used only to precondition a matrix-free solve. */
enum class OrbitalPreconditioner {
  Automatic,
  Identity,
  OneElectron,
  CasscfDiagonal,
};

}  // namespace xmvb::vb
