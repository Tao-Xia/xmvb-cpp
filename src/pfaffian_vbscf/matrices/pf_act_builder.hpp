#pragma once

#include "pfaffian_vbscf/types/pf_active_space_data.hpp"
#include "vb/matrices/cpp_vb_input.hpp"

namespace xmvb::pfaffian_vbscf {

struct PfPreparedActiveSpaceData {
  PfActiveSpaceData active_space;
  double reference_energy = 0.0;
};

/**
 * @brief Extracts the active-space overlap and Hamiltonian payload from one
 * `CppVbInput`.
 */
class PfActBuilder {
public:
  /**
   * @brief Builds the Pfaffian active-space payload together with `e_ref`.
   *
   * This reuses the shared C++ active-space preparation path so that the Pf
   * forward evaluation consumes the same exact `sso / hho / ggo` and
   * one-electron reference energy as the optimized VB active-space builder.
   */
  PfPreparedActiveSpaceData prepare(const xmvb::vb::CppVbInput& input) const;

  /**
   * @brief Builds the active-space payload used by the Pfaffian matrix builder.
   *
   * @param input Molecule-level C++ VB input bundle.
   * @return PfActiveSpaceData Active overlap, one-electron, and two-electron data.
   */
  PfActiveSpaceData build(const xmvb::vb::CppVbInput& input) const;
};

}  // namespace xmvb::pfaffian_vbscf
