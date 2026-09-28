#include "output/text/structure.hpp"

#include <sstream>
#include <stdexcept>
#include <string>

namespace {

xmvb::vb::RawStructureData paired_structures() {
  xmvb::vb::RawStructureData structures;
  structures.n_structures = 2;
  structures.n_total_electrons = 42;
  structures.n_active_electrons = 6;
  structures.spin_multiplicity = 1;
  for (int structure = 0; structure < structures.n_structures; ++structure) {
    for (int orbital = 1; orbital <= 18; ++orbital) {
      structures.raw_structure_orbitals.push_back(orbital);
      structures.raw_structure_orbitals.push_back(orbital);
    }
    if (structure == 0) {
      structures.raw_structure_orbitals.insert(
          structures.raw_structure_orbitals.end(),
          {19, 24, 20, 23, 21, 22});
    } else {
      structures.raw_structure_orbitals.insert(
          structures.raw_structure_orbitals.end(),
          {19, 23, 20, 24, 21, 22});
    }
  }
  return structures;
}

std::string format(
    const xmvb::vb::RawStructureData& structures,
    int index,
    int prefix_width = 34) {
  std::ostringstream output;
  xmvb::output::print_structure(output, structures, index, prefix_width);
  return output.str();
}

void require_equal(
    const std::string& actual,
    const std::string& expected,
    const char* message) {
  if (actual != expected) {
    throw std::runtime_error(
        std::string(message) + "\nexpected: [" + expected +
        "]\nactual:   [" + actual + "]");
  }
}

}  // namespace

int main() {
  const auto structures = paired_structures();
  require_equal(
      format(structures, 0),
      "  1:18   19-24    20-23    21-22   ",
      "covalent pair topology was not preserved");
  require_equal(
      format(structures, 1),
      "  1:18   19-23    20-24    21-22   ",
      "distinct structures were formatted identically");

  xmvb::vb::RawStructureData open_shell;
  open_shell.n_structures = 1;
  open_shell.n_total_electrons = 5;
  open_shell.n_active_electrons = 3;
  open_shell.spin_multiplicity = 2;
  open_shell.raw_structure_orbitals = {1, 1, 2, 2, 3};
  require_equal(
      format(open_shell, 0),
      "   1:1    2 2      3  ",
      "ionic pair or open-shell formatting is incorrect");

  xmvb::vb::RawStructureData long_structure;
  long_structure.n_structures = 1;
  long_structure.n_total_electrons = 22;
  long_structure.n_active_electrons = 22;
  long_structure.spin_multiplicity = 1;
  for (int orbital = 1; orbital <= 22; orbital += 2) {
    long_structure.raw_structure_orbitals.push_back(orbital);
    long_structure.raw_structure_orbitals.push_back(orbital + 1);
  }
  const std::string wrapped = format(long_structure, 0, 18);
  if (wrapped.find("\n" + std::string(18, ' ') + " 21-22") ==
      std::string::npos) {
    throw std::runtime_error("long structure continuation is misaligned");
  }
}
