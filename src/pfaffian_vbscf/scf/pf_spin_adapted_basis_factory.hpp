#pragma once

#include <limits>

#include "pfaffian_vbscf/types/pf_spin_adapted_basis_data.hpp"
#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/matrices/structure_types.hpp"

namespace xmvb::pfaffian_vbscf {

struct PfSpinAdaptedBasisFactoryOptions {
  int n_structures = 0;
  int seed = 20260328;
  double pairing_noise = 0.0;
  int target_spin_multiplicity = 0;
  int ms_twice = std::numeric_limits<int>::min();
};

PfSpinAdaptedBasisData build_structure_pf_spin_adapted_basis(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::RawStructureData& raw_structure_data,
    const PfSpinAdaptedBasisFactoryOptions& options = {});

}  // namespace xmvb::pfaffian_vbscf
