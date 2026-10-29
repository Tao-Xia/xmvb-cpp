#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "input/loading/loader.hpp"
#include "vbscf/orbitals/preparation/preparer.hpp"

/** Export the actual HAO physical guess, including support zeros, into OEO GUS.
 * Historical autoVB guesses can reuse one full-AO column for different local
 * active orbitals. Merely changing ORBTYP admits these duplicate columns;
 * projecting the HAO supports first retains the original physical orbitals.
 */
int main(int argc,char** argv) {
  try {
    if (argc!=3) throw std::invalid_argument("usage: prepare_tnvb_oeo HAO.xmi OUTPUT.xmi");
    const auto input=xmvb::vb::load_vbscf_input(argv[1]);
    const auto prepared=xmvb::vb::ActiveSpaceOrbitalPreparer{}.prepare(input.orbital_preparation_input);
    const auto& matrix=prepared.physical_orbital_frame.normalized_orbital_matrix;
    const int n_orb=input.orbital_preparation_input.n_orbitals;
    std::ifstream source(argv[1]);
    std::ofstream target(argv[2]);
    if (!source || !target) throw std::runtime_error("cannot open guess-conversion files");
    std::string line;
    bool found=false;
    while (std::getline(source,line)) {
      std::string lower=line;
      std::transform(lower.begin(),lower.end(),lower.begin(),[](unsigned char c) { return std::tolower(c); });
      if (lower.find("$gus")!=std::string::npos) { found=true; break; }
      const auto position=lower.find("orbtyp=hao");
      if (position!=std::string::npos) line.replace(position,10,"ORBTYP=OEO");
      target << line << '\n';
    }
    if (!found) throw std::invalid_argument("guess converter requires a $GUS block");
    target << "$GUS\n";
    for (int orbital=0;orbital<n_orb;++orbital) target << matrix.rows() << ' ';
    target << '\n' << std::scientific << std::setprecision(17);
    for (int orbital=0;orbital<n_orb;++orbital) {
      target << "# physical HAO guess orbital " << orbital+1 << '\n';
      for (int ao=0;ao<matrix.rows();++ao) {
        target << matrix(ao,orbital) << ' ' << ao+1 << ((ao+1)%4==0 ? '\n' : ' ');
      }
      if (matrix.rows()%4) target << '\n';
    }
    target << "$END\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
