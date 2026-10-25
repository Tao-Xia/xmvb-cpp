/** Standalone interop fixture writer; no determinant expansion or RDM. */
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include "input/loading/loader.hpp"
#include "vbscf/integrals/active/space.hpp"

int main(int argc,char** argv) {
  try {
    if (argc!=4) throw std::invalid_argument("usage: export_tnvb_request INPUT REQUEST.bin exact|ri");
    xmvb::vb::VbScfInputLoadOptions options;
    if (std::string(argv[3])=="exact") options.standard_two_electron_mode=xmvb::vb::StandardTwoElectronMode::Exact;
    else if (std::string(argv[3])=="ri") options.standard_two_electron_mode=xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
    else throw std::invalid_argument("invalid integral model");
    const auto loaded=xmvb::vb::load_vbscf_input_with_timings(argv[1],options);
    const auto& input=loaded.input;
    if (!input.tnvb.enabled || loaded.structure_expansion_seconds!=0 || !input.structure_data.pair_alpha.empty()) {
      throw std::invalid_argument("requires unexpanded TNVB input");
    }
    const auto timed=xmvb::vb::prepare_timed_active_space_context(input);
    const auto& prepared=timed.prepared_active_space;
    const auto& raw=input.raw_structure_data;
    const int k=input.orbital_preparation_input.n_active_orbitals;
    const int np=k*(k+1)/2;
    auto eri=prepared.active_space_two_electron_result.packed_active_two_electron_integrals;
    if (eri.empty()) {
      const auto& factors=prepared.active_space_two_electron_result.ri_active_pair_factors;
      if (factors.cols()!=np) throw std::invalid_argument("invalid RI pair dimension");
      eri.resize(static_cast<std::size_t>(np)*(np+1)/2);
      for (int j=0;j<np;++j) for (int i=0;i<=j;++i) eri[j*(j+1)/2+i]=factors.col(i).dot(factors.col(j));
    }
    std::ofstream out(argv[2],std::ios::binary);
    const auto write=[&](const auto* values,std::size_t count) {
      out.write(reinterpret_cast<const char*>(values),count*sizeof(*values));
      if (!out) throw std::runtime_error("interop request write failed");
    };
    out.write("CPTNREQ3",8);
    const std::array<std::int32_t,10> header{3,raw.n_structures,k,raw.n_active_electrons,
        raw.n_total_electrons,input.tnvb.structure_bond,input.tnvb.total_bond,1,
        loaded.structure_eigensolver==xmvb::vb::StructureEigensolver::Davidson ? 1 : 0,0};
    write(header.data(),header.size());
    const std::array<double,2> accuracy{1e-7,1e-3};
    write(accuracy.data(),accuracy.size());
    for (const auto index:raw.raw_structure_orbitals) {
      const std::int32_t value=index; write(&value,1);
    }
    const auto& s=prepared.orbital_result.active_orbital_overlap_matrix;
    const auto& h=prepared.active_space_one_electron_result.h1e_act;
    const auto& frame=prepared.active_space_two_electron_result.orbital_transform;
    write(s.data(),s.size()); write(frame.data(),frame.size());
    write(h.data(),h.size()); write(eri.data(),eri.size());
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
