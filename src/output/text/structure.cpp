#include "output/text/structure.hpp"

#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>

namespace xmvb::output {

void print_structure(
    std::ostream& output,
    const vb::RawStructureData& structures,
    int structure_index,
    int prefix_width) {
  if (structure_index < 0 || structure_index >= structures.n_structures) {
    throw std::out_of_range("structure index is out of range");
  }
  if (structures.n_total_electrons < 0 ||
      structures.n_active_electrons < 0 ||
      structures.n_active_electrons > structures.n_total_electrons ||
      structures.spin_multiplicity < 1 || prefix_width < 0) {
    throw std::invalid_argument("invalid raw structure dimensions");
  }
  const int beta_numerator =
      structures.n_total_electrons - structures.spin_multiplicity + 1;
  const int inactive_numerator =
      structures.n_total_electrons - structures.n_active_electrons;
  if (beta_numerator < 0 || beta_numerator % 2 != 0 ||
      inactive_numerator % 2 != 0) {
    throw std::invalid_argument("raw structure spin parity is inconsistent");
  }

  const int n_beta = beta_numerator / 2;
  const int n_inactive = inactive_numerator / 2;
  if (n_inactive > n_beta ||
      structures.raw_structure_orbitals.size() <
          structures.flat_orbital_count()) {
    throw std::invalid_argument("raw structure storage is inconsistent");
  }

  const int* orbitals = structures.structure_orbitals_data(structure_index);
  int continuation_width = prefix_width;
  if (n_inactive > 0) {
    output << std::setw(6) << ("1:" + std::to_string(n_inactive)) << "  ";
    continuation_width += 8;
  }

  for (int beta = n_inactive; beta < n_beta; ++beta) {
    const int left_orbital = orbitals[2 * beta];
    const int right_orbital = orbitals[2 * beta + 1];
    output << std::setw(3) << left_orbital;
    if (left_orbital != right_orbital) {
      output << '-';
    } else {
      output << ' ';
    }
    output << std::left << std::setw(3) << right_orbital << std::right << "  ";

    if ((beta - n_inactive + 1) % 10 == 0) {
      output << '\n' << std::string(continuation_width, ' ');
    }
  }

  for (int electron = 2 * n_beta;
       electron < structures.n_total_electrons;
       ++electron) {
    output << std::setw(3) << orbitals[electron] << "  ";
  }
}

}  // namespace xmvb::output
