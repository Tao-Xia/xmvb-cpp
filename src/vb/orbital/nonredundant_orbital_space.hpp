#pragma once

// Compatibility header. New code should include
// "vbscf/orbitals/charts/orbital_chart.hpp".
#include "vb/orbital/sparse_orbital_parameter_view.hpp"
#include "vbscf/orbitals/charts/orbital_chart.hpp"

namespace xmvb::vb {

using NonredundantOrbitalSpace = OrbitalChart;

}  // namespace xmvb::vb
