#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

#include "input/deck/model.hpp"
#include "input/loading/loader.hpp"

namespace {
void require(bool value,const char* message) {
  if (!value) throw std::runtime_error(message);
}
struct TempDirectory {
  std::filesystem::path path;
  TempDirectory() {
    std::string pattern=(std::filesystem::temp_directory_path()/"cpp-tnvb-input-XXXXXX").string();
    if (!mkdtemp(pattern.data())) throw std::runtime_error("cannot create isolated test directory");
    path=pattern;
  }
  ~TempDirectory() { std::error_code error; std::filesystem::remove_all(path,error); }
};
void replace(std::string* text,const std::string& before,const std::string& after) {
  const auto position=text->find(before);
  if (position==std::string::npos) throw std::runtime_error("missing fixture token");
  text->replace(position,before.size(),after);
}
}  // namespace

int main(int argc,char** argv) {
  try {
    if (argc!=2) throw std::invalid_argument("expected benzene fixture");
    std::ifstream stream(argv[1]);
    const std::string original{std::istreambuf_iterator<char>(stream),{}};
    TempDirectory scratch;
    const auto path=scratch.path/"input.xmi";
    const auto write=[&](const std::string& configuration,const std::string& itmax="itmax=0") {
      auto text=original;
      replace(&text,"tnvb\ntnbond=128\ntnsumbond=256\n",configuration);
      replace(&text,"itmax=0",itmax);
      std::ofstream output(path);
      output << text;
      if (!output) throw std::runtime_error("test fixture write failed");
    };
    write("TNVB TNBOND=128\n");
    auto loaded=xmvb::vb::load_vbscf_input_with_timings(path.string());
    require(loaded.input.tnvb.enabled && loaded.input.tnvb.structure_bond==128,"missing minimum configuration");
    require(loaded.input.tnvb.total_bond==0,"ENERGY must not invent TNSUMBOND");
    require(loaded.requested_scf_max_iterations==0,"TNVB ITMAX=0 was overridden");
    require(loaded.structure_eigensolver==xmvb::vb::StructureEigensolver::Dense,"TNVB did not default to dense solving");
    require(loaded.structure_expansion_seconds==0 && loaded.input.structure_data.pair_alpha.empty() &&
        loaded.input.structure_data.pair_beta.empty() && loaded.input.structure_data.terms.empty() &&
        loaded.input.structure_data.alpha_strings.empty() && loaded.input.structure_data.beta_strings.empty() &&
        loaded.input.structure_data.offsets.empty(),"TNVB expanded spin-string topology");
    require(!loaded.input.complete_active_space && loaded.input.raw_structure_data.n_structures==5,"raw selected structures lost or mislabeled CAS");
    write("TNVB TNBOND=128 TNSUMBOND=256\n","itmax=200");
    loaded=xmvb::vb::load_vbscf_input_with_timings(path.string());
    require(loaded.requested_scf_max_iterations==200 && loaded.input.tnvb.total_bond==256,"optimization settings lost");
    write("TNVB TNBOND=128 EIGENSOLVER=DAVIDSON\n");
    loaded=xmvb::vb::load_vbscf_input_with_timings(path.string());
    require(loaded.structure_eigensolver==xmvb::vb::StructureEigensolver::Davidson,"explicit algorithm request silently changed");
    write("");
    loaded=xmvb::vb::load_vbscf_input_with_timings(path.string());
    require(!loaded.input.tnvb.enabled && loaded.requested_scf_max_iterations==2000 &&
        loaded.structure_eigensolver==xmvb::vb::StructureEigensolver::Davidson &&
        !loaded.input.structure_data.pair_alpha.empty(),"ordinary defaults or determinant loading changed");
    for (const auto* invalid : {"TNVB TNBOND=0\n","TNVB TNBOND=-1\n","TNVB TNBOND=1.5\n",
         "TNVB TNBOND=12foo\n","TNVB TNBOND=2147483648\n","TNVB TNBOND=128 TNSUMBOND=0\n","TNVB\n"}) {
      write(invalid);
      bool rejected=false;
      try { (void)xmvb::vb::load_vbscf_input_with_timings(path.string()); }
      catch (const std::exception&) { rejected=true; }
      require(rejected,"invalid or absent bond accepted");
    }
    std::cout << "TNVB input invariants passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
