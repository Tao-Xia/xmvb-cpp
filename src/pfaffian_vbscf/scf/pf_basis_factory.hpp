#pragma once

#include <limits>

#include "pfaffian_vbscf/types/pf_basis_data.hpp"
#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/matrices/structure_types.hpp"

namespace xmvb::pfaffian_vbscf {

struct PfBasisFactoryOptions {
  int n_states = 0;
  int seed = 20260328;
  double pairing_noise = 0.0;
};

struct PfFixedMsBasisFactoryOptions {
  int n_structures = 0;
  int seed = 20260328;
  double pairing_noise = 0.0;
  int ms_twice = std::numeric_limits<int>::max();
};

PfBasisData build_fixed_ms_pf_basis(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::RawStructureData& raw_structure_data,
    const PfFixedMsBasisFactoryOptions& options = {});

PfBasisData build_structure_pf_basis(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::RawStructureData& raw_structure_data,
    const PfBasisFactoryOptions& options = {});

}  // namespace xmvb::pfaffian_vbscf
