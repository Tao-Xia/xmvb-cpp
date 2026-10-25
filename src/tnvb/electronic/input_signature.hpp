#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include "vbscf/core/contracts/input.hpp"

namespace xmvb::tnvb {

/** Non-security cache fingerprint of all scientific input, without copying AO ERIs.
 * Lazy RI caches and execution-only budgets are excluded; the immutable RI
 * provider identity and both primary/auxiliary basis definitions are included.
 */
inline std::array<std::uint64_t,2> input_signature(const vb::VbScfInput& input) {
  struct Hash {
    std::array<std::uint64_t,2> value{0x243f6a8885a308d3ULL,0x13198a2e03707344ULL};
    void word(std::uint64_t x) {
      value[0]=(value[0]^x)*0x9e3779b185ebca87ULL;
      value[1]=std::rotl(value[1]+x*0xc2b2ae3d27d4eb4fULL,31)*0x165667b19e3779f9ULL;
    }
    void bytes(const void* source,std::size_t length) {
      word(length);
      const auto* data=static_cast<const unsigned char*>(source);
      while (length>=8) { std::uint64_t x;std::memcpy(&x,data,8);word(x);data+=8;length-=8; }
      if (length) { std::uint64_t x=0;std::memcpy(&x,data,length);word(x); }
    }
  } hash;
  const auto scalar=[&](const auto& x) { hash.bytes(&x,sizeof(x)); };
  const auto vector=[&](const auto& x) { hash.bytes(x.data(),x.size()*sizeof(*x.data())); };
  const auto matrix=[&](const auto& x) { scalar(x.rows());scalar(x.cols());vector(x); };
  const auto basis=[&](const auto& x) {
    scalar(x.n_atoms);scalar(x.n_shells);scalar(x.n_gaussian_primitives);
    vector(x.atm);vector(x.bas);vector(x.basidx);vector(x.env);
  };
  scalar(input.tnvb.enabled);scalar(input.tnvb.structure_bond);scalar(input.tnvb.total_bond);
  scalar(input.complete_active_space);scalar(input.standard_two_electron_mode);
  const auto& raw=input.raw_structure_data;
  scalar(raw.n_structures);scalar(raw.n_total_electrons);scalar(raw.n_active_electrons);
  scalar(raw.spin_multiplicity);scalar(raw.wavefunction_type);scalar(raw.vb_function_type);
  vector(raw.raw_structure_orbitals);
  const auto& o=input.orbital_preparation_input;
  scalar(o.n_basis_functions);scalar(o.n_orbitals);scalar(o.n_active_orbitals);
  scalar(o.n_total_electrons);scalar(o.n_active_electrons);scalar(o.spin_multiplicity);
  scalar(o.orbital_type);scalar(o.maintain_inactive_gauge);
  vector(o.orbital_value_table);vector(o.orbital_basis_index_table);vector(o.orbital_basis_counts);
  vector(o.mo_gauge_reference_orbital_basis_counts);vector(o.mo_gauge_reference_orbital_basis_index_table);
  vector(o.original_orbital_basis_counts);matrix(o.ao_overlap_matrix);matrix(o.hf_overlap_matrix);
  scalar(o.n_blocks);scalar(o.block_storage_dimension);scalar(o.block_partial_overlap);
  vector(o.block_members);vector(o.block_orbital_counts);vector(o.block_basis_counts);vector(o.ao_normalization);
  const auto& ao=input.ao_integral_input;
  scalar(ao.n_basis_functions);matrix(ao.ao_core_hamiltonian_matrix);
  const auto& g=ao.pair_graph;
  vector(g.row_offsets);vector(g.columns);vector(g.values);vector(g.integral_rows);
  vector(g.integral_edges);vector(g.integral_columns);vector(g.integral_values);
  // Upstream may retain the exact AO payload as producer-owned chunks.
  // Hash that representation without materializing a second integral array.
  scalar(g.chunks.size());vector(g.chunk_offsets);
  for (const auto& chunk:g.chunks) {
    vector(chunk.rows);vector(chunk.columns);vector(chunk.values);
  }
  vector(g.pair_first);vector(g.pair_second);
  basis(input.libcint_input);basis(input.auxiliary_libcint_input);
  scalar(reinterpret_cast<std::uintptr_t>(input.ri_factorization_provider.get()));
  return hash.value;
}

}  // namespace xmvb::tnvb
