#include "vb/pdft/bragg_slater_radii.hpp"

#include <stdexcept>

namespace xmvb::vb::pdft {

namespace {

// Bragg-Slater radii in bohr copied from the established legacy grid path in
// `src/mol/xgrids.c`.  Reusing the same table keeps the new PDFT quadrature
// scale consistent with the historical runtime instead of introducing a second
// incompatible atomic-radius convention.
constexpr double kBraggSlaterRadiiBohr[] = {
    1.000,  // dummy for index 0
    0.661, 0.661,
    2.740, 1.984, 1.606, 1.323, 1.228, 1.134, 0.945, 0.900,
    3.402, 2.835, 2.362, 2.079, 1.890, 1.890, 1.890, 1.890,
    4.157, 3.402, 3.024, 2.656, 2.551, 2.656, 2.656, 2.656, 2.551, 2.551,
    2.551, 2.551, 2.457, 2.362, 2.173, 2.173, 2.173, 2.173,
    4.441, 3.780,
    3.402, 3.024, 2.929, 2.740, 2.656, 2.551, 2.551, 2.551, 2.551, 2.551,
    2.646, 2.646, 2.551, 2.457, 2.457, 2.362, 2.173, 2.173, 2.173, 2.173,
    4.441, 3.780,
    3.402, 2.929, 2.740, 2.740, 2.551, 2.457, 2.551, 2.646, 3.024, 2.929,
    2.929, 2.740, 2.740, 2.646, 2.646, 2.646,
    4.063, 4.063,
    3.685, 3.496, 3.496, 3.496, 3.496, 3.496, 3.496, 3.402, 3.307, 3.307,
    3.307, 3.307, 3.307, 3.307, 3.307,
    2.929, 2.740, 2.551, 2.551, 2.457, 2.551, 2.551, 2.551, 2.835, 3.591,
    3.024, 3.024, 3.591, 3.591, 3.591,
    4.063, 4.063,
    3.685, 3.401, 3.401, 3.307, 3.307, 3.307, 3.307, 3.307, 3.307, 3.307,
    3.307, 3.307, 3.307, 3.307, 3.307,
};

constexpr int kMaxAtomicNumber =
    static_cast<int>(sizeof(kBraggSlaterRadiiBohr) /
                         sizeof(kBraggSlaterRadiiBohr[0])) -
    1;

}  // namespace

double bragg_slater_radius_bohr(int atomic_number) {
  if (atomic_number < 1 || atomic_number > kMaxAtomicNumber) {
    throw std::invalid_argument("Atomic number out of supported Bragg-Slater range");
  }
  return kBraggSlaterRadiiBohr[atomic_number];
}

}  // namespace xmvb::vb::pdft
