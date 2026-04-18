#pragma once

namespace xmvb::vb::pdft {

/**
 * @brief Returns the legacy Bragg-Slater atomic radius in bohr.
 *
 * The PDFT grid code should stay numerically aligned with the existing
 * `mol/xgrids.c` implementation while the new C++ path is being validated.
 * This helper therefore exposes the same radius table that the legacy grid
 * builder uses for atom-dependent radial scaling and Becke heteronuclear
 * size correction.
 */
double bragg_slater_radius_bohr(int atomic_number);

}  // namespace xmvb::vb::pdft
