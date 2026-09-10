#include "output/text/sections.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <unistd.h>

#include "libcint/c_api.hpp"
#include "libcint/direct_shell.hpp"

namespace xmvb::output {
namespace {

namespace fs = std::filesystem;

constexpr std::array<const char*, 87> kElementSymbols = {{
    "X",
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
}};

const char* element_symbol(int atomic_number) {
  if (atomic_number <= 0 ||
      atomic_number >= static_cast<int>(kElementSymbols.size())) {
    return "X";
  }
  return kElementSymbols[atomic_number];
}

std::string format_timestamp(
    const std::chrono::system_clock::time_point& time_point) {
  const std::time_t raw_time = std::chrono::system_clock::to_time_t(time_point);
  std::tm local_time{};
  localtime_r(&raw_time, &local_time);
  std::array<char, 32> buffer{};
  if (std::strftime(
          buffer.data(), buffer.size(), "%Y-%m-%d %H:%M:%S", &local_time) == 0) {
    return "<unavailable>";
  }
  return buffer.data();
}

std::string read_input_file(const std::string& input_path) {
  std::ifstream stream(input_path);
  if (!stream) {
    throw std::runtime_error("failed to reopen input deck for report: " + input_path);
  }
  std::ostringstream contents;
  contents << stream.rdbuf();
  return contents.str();
}

std::string orbital_range(const std::vector<int>& orbitals) {
  if (orbitals.empty()) {
    return {};
  }
  std::ostringstream text;
  std::size_t begin = 0;
  while (begin < orbitals.size()) {
    std::size_t end = begin;
    while (end + 1 < orbitals.size() && orbitals[end + 1] == orbitals[end] + 1) {
      ++end;
    }
    if (begin != 0) {
      text << ' ';
    }
    if (end > begin) {
      text << orbitals[begin] << '-' << orbitals[end];
    } else {
      text << orbitals[begin];
    }
    begin = end + 1;
  }
  return text.str();
}

std::string structure_description(
    const vb::RawStructureData& structures,
    int structure_index) {
  const int n_inactive =
      (structures.n_total_electrons - structures.n_active_electrons) / 2;
  std::map<int, int> occupations;
  const int* orbitals = structures.structure_orbitals_data(structure_index);
  for (int electron = 0; electron < structures.n_total_electrons; ++electron) {
    ++occupations[orbitals[electron]];
  }

  std::ostringstream text;
  if (n_inactive > 0) {
    text << "1:" << n_inactive;
  }
  std::vector<int> singly_occupied;
  for (const auto& [orbital, occupation] : occupations) {
    if (orbital <= n_inactive) {
      continue;
    }
    if (occupation == 1) {
      singly_occupied.push_back(orbital);
    } else {
      for (int copy = 0; copy < occupation; ++copy) {
        text << ' ' << orbital;
      }
    }
  }
  if (!singly_occupied.empty()) {
    text << ' ' << orbital_range(singly_occupied);
  }
  return text.str();
}

int structure_ionic_order(
    const vb::RawStructureData& structures,
    int structure_index) {
  const int n_inactive =
      (structures.n_total_electrons - structures.n_active_electrons) / 2;
  const int* orbitals = structures.structure_orbitals_data(structure_index);
  std::map<int, int> active_occupations;
  for (int electron = 2 * n_inactive;
       electron < structures.n_total_electrons;
       ++electron) {
    ++active_occupations[orbitals[electron]];
  }
  int ionic_order = 0;
  for (const auto& entry : active_occupations) {
    ionic_order += entry.second / 2;
  }
  return ionic_order;
}

void print_ionic_class_summary(
    std::ostream& output,
    const vb::RawStructureData& structures,
    const vb::FullDeterminantStructureData& determinants) {
  struct IonicClass {
    int n_structures = 0;
    int n_determinants = 0;
    int first_structure = 0;
    int last_structure = 0;
  };
  std::map<int, IonicClass> classes;
  std::vector<int> ionic_order_by_structure(structures.n_structures, 0);
  for (int structure = 0; structure < structures.n_structures; ++structure) {
    const int ionic_order = structure_ionic_order(structures, structure);
    ionic_order_by_structure[structure] = ionic_order;
    auto& ionic_class = classes[ionic_order];
    if (ionic_class.n_structures == 0) {
      ionic_class.first_structure = structure + 1;
    }
    ionic_class.last_structure = structure + 1;
    ++ionic_class.n_structures;
  }
  for (const auto& terms : determinants.determinant_to_structure_terms) {
    for (const auto& term : terms) {
      if (term.structure_index < 0 ||
          term.structure_index >= structures.n_structures) {
        throw std::out_of_range(
            "determinant expansion refers to an invalid structure index");
      }
      ++classes[ionic_order_by_structure[term.structure_index]].n_determinants;
    }
  }

  int first_determinant = 1;
  for (const auto& [ionic_order, ionic_class] : classes) {
    const int last_determinant =
        first_determinant + ionic_class.n_determinants - 1;
    output << "Number of " << std::setw(3) << ionic_order
           << "th ion structures   is : " << std::setw(10)
           << ionic_class.n_structures << "  from " << std::setw(10)
           << ionic_class.first_structure << " to " << std::setw(10)
           << ionic_class.last_structure << '\n'
           << "Number of " << std::setw(3) << ionic_order
           << "th ion determinants is : " << std::setw(10)
           << ionic_class.n_determinants << "  from " << std::setw(10)
           << first_determinant << " to " << std::setw(10)
           << last_determinant << '\n';
    first_determinant = last_determinant + 1;
  }
  output << '\n';
}

std::string ao_label(
    const vb::VbScfStaticMoleculeMetadata& metadata,
    int basis_index) {
  const int angular_momentum = metadata.ao_angular_momenta[basis_index];
  if (angular_momentum == 0) {
    return "S";
  }
  if (angular_momentum < 0 || angular_momentum >= 6) {
    return "?";
  }
  const int exponent_offset = 3 * basis_index;
  if (exponent_offset + 2 >=
      static_cast<int>(metadata.ao_cartesian_exponents.size())) {
    return "?";
  }
  std::string label;
  if (angular_momentum > 0) {
    label.push_back("SPDFGH"[angular_momentum]);
  }
  label.append(metadata.ao_cartesian_exponents[exponent_offset], 'X');
  label.append(metadata.ao_cartesian_exponents[exponent_offset + 1], 'Y');
  label.append(metadata.ao_cartesian_exponents[exponent_offset + 2], 'Z');
  return label;
}

Eigen::VectorXd selected_structure_coefficients(const vb::VbScfResult& result) {
  const int n_str = result.n_structures;
  if (n_str <= 0 || result.eigenvector_matrix.size() != n_str * n_str) {
    return {};
  }
  const int state = result.selected_state_indices.empty()
      ? 0
      : result.selected_state_indices.front();
  if (state < 0 || state >= n_str) {
    return {};
  }
  Eigen::VectorXd coefficients = Eigen::Map<const Eigen::VectorXd>(
      result.eigenvector_matrix.data() + state * n_str, n_str);
  for (int index = 0; index < n_str; ++index) {
    if (std::abs(coefficients[index]) > 1.0e-14) {
      if (coefficients[index] > 0.0) {
        coefficients = -coefficients;
      }
      break;
    }
  }
  return coefficients;
}

Eigen::MatrixXd structure_matrix(
    const std::vector<double>& storage,
    int n_str) {
  if (n_str <= 0 || storage.size() != n_str * n_str) {
    return {};
  }
  return Eigen::Map<const Eigen::MatrixXd>(storage.data(), n_str, n_str);
}

struct NormalizedStructureState {
  Eigen::MatrixXd overlap;
  Eigen::MatrixXd electronic_hamiltonian;
  Eigen::VectorXd coefficients;
};

NormalizedStructureState normalize_structure_state(
    const vb::VbScfResult& result) {
  const int n_str = result.n_structures;
  const Eigen::MatrixXd raw_overlap =
      structure_matrix(result.structure_matrices.overlap_matrix, n_str);
  const Eigen::MatrixXd raw_hamiltonian =
      structure_matrix(result.structure_matrices.hamiltonian_matrix, n_str);
  const Eigen::VectorXd raw_coefficients =
      selected_structure_coefficients(result);
  if (raw_overlap.rows() != n_str || raw_hamiltonian.rows() != n_str ||
      raw_coefficients.size() != n_str) {
    throw std::invalid_argument(
        "final structure state has inconsistent matrix or coefficient dimensions");
  }

  Eigen::VectorXd structure_norms(n_str);
  for (int structure = 0; structure < n_str; ++structure) {
    const double squared_norm = raw_overlap(structure, structure);
    if (!std::isfinite(squared_norm) || squared_norm <= 0.0) {
      throw std::domain_error(
          "a final VB structure has a non-positive or non-finite norm");
    }
    structure_norms[structure] = std::sqrt(squared_norm);
  }

  NormalizedStructureState normalized;
  normalized.overlap.resize(n_str, n_str);
  normalized.electronic_hamiltonian.resize(n_str, n_str);
  for (int column = 0; column < n_str; ++column) {
    for (int row = 0; row < n_str; ++row) {
      const double norm_product =
          structure_norms[row] * structure_norms[column];
      normalized.overlap(row, column) = raw_overlap(row, column) / norm_product;
      normalized.electronic_hamiltonian(row, column) =
          raw_hamiltonian(row, column) / norm_product +
          result.one_electron_reference_energy *
              normalized.overlap(row, column);
    }
  }

  // For |Phi_i'> = |Phi_i>/sqrt(S_ii), preserving
  // |Psi> = sum_i c_i |Phi_i> requires c_i' = sqrt(S_ii) c_i.
  normalized.coefficients =
      structure_norms.array() * raw_coefficients.array();
  return normalized;
}

std::string determinant_spin_string(
    const std::vector<int>& active_orbitals,
    int n_inactive) {
  std::ostringstream text;
  if (n_inactive == 1) {
    text << '1';
  } else if (n_inactive > 1) {
    text << "1-" << n_inactive;
  }
  std::vector<int> absolute_active_orbitals;
  absolute_active_orbitals.reserve(active_orbitals.size());
  for (const int orbital : active_orbitals) {
    absolute_active_orbitals.push_back(n_inactive + orbital + 1);
  }
  if (!absolute_active_orbitals.empty()) {
    if (n_inactive > 0) {
      text << ' ';
    }
    text << orbital_range(absolute_active_orbitals);
  }
  return text.str();
}

void print_determinant_coefficients(
    std::ostream& output,
    const Eigen::VectorXd& structure_coefficients,
    const vb::FullDeterminantStructureData& determinants,
    int n_inactive) {
  const int n_determinants = static_cast<int>(determinants.alpha_det.size());
  if (determinants.beta_det.size() != determinants.alpha_det.size() ||
      determinants.determinant_to_structure_terms.size() !=
          determinants.alpha_det.size()) {
    throw std::invalid_argument(
        "final determinant topology has inconsistent dimensions");
  }

  output << "\n\n          ******  COEFFICIENTS OF DETERMINANTS WITHOUT NORMALIZED ******\n\n"
         << "                                     INA   A\n"
         << "                                     INA   B\n";
  std::vector<int> display_order(n_determinants);
  std::iota(display_order.begin(), display_order.end(), 0);
  std::stable_sort(
      display_order.begin(),
      display_order.end(),
      [&determinants](int left, int right) {
        const auto& left_terms =
            determinants.determinant_to_structure_terms[left];
        const auto& right_terms =
            determinants.determinant_to_structure_terms[right];
        const int left_structure = left_terms.empty()
            ? std::numeric_limits<int>::max()
            : left_terms.front().structure_index;
        const int right_structure = right_terms.empty()
            ? std::numeric_limits<int>::max()
            : right_terms.front().structure_index;
        if (left_structure != right_structure) {
          return left_structure < right_structure;
        }
        return determinants.alpha_det[left] > determinants.alpha_det[right];
      });
  for (int display_index = 0; display_index < n_determinants; ++display_index) {
    const int determinant = display_order[display_index];
    double coefficient = 0.0;
    for (const auto& term :
         determinants.determinant_to_structure_terms[determinant]) {
      if (term.structure_index < 0 ||
          term.structure_index >= structure_coefficients.size()) {
        throw std::out_of_range(
            "determinant expansion refers to an invalid structure index");
      }
      coefficient +=
          term.coefficient * structure_coefficients[term.structure_index];
    }
    output << std::setw(8) << display_index + 1
           << std::fixed << std::setprecision(8) << std::setw(17)
           << coefficient << "  ******    "
           << determinant_spin_string(determinants.alpha_det[determinant], n_inactive)
           << '\n'
           << std::string(37, ' ')
           << determinant_spin_string(determinants.beta_det[determinant], n_inactive)
           << '\n';
  }
}

void print_structure_matrix(
    std::ostream& output,
    const char* title,
    const Eigen::MatrixXd& matrix) {
  output << "\n\n              ******  " << title << " OF VB STRUCTURES  ******\n";
  constexpr int kColumnsPerBlock = 6;
  for (int first_column = 0; first_column < matrix.cols();
       first_column += kColumnsPerBlock) {
    const int last_column = std::min(
        first_column + kColumnsPerBlock, static_cast<int>(matrix.cols()));
    output << "\n       ";
    for (int column = first_column; column < last_column; ++column) {
      output << std::setw(13) << column + 1;
    }
    output << '\n';
    for (int row = 0; row < matrix.rows(); ++row) {
      output << std::setw(4) << row + 1 << "   ";
      for (int column = first_column; column < last_column; ++column) {
        output << std::fixed << std::setprecision(6) << std::setw(13)
               << matrix(row, column);
      }
      output << '\n';
    }
  }
}

void print_weight_table(
    std::ostream& output,
    const char* title,
    const Eigen::VectorXd& weights,
    const vb::RawStructureData& structures) {
  output << "\n         " << title << "\n\n";
  for (int index = 0; index < weights.size(); ++index) {
    output << std::setw(8) << index + 1
           << std::fixed << std::setprecision(8) << std::setw(17)
           << weights[index] << "  ******    "
           << structure_description(structures, index) << '\n';
  }
}

void print_structure_weights(
    std::ostream& output,
    const Eigen::VectorXd& coefficients,
    const Eigen::MatrixXd& overlap,
    const vb::RawStructureData& structures) {
  if (coefficients.size() == 0 || overlap.rows() != coefficients.size()) {
    throw std::invalid_argument(
        "structure-weight dimensions are inconsistent");
  }
  const double norm = coefficients.dot(overlap * coefficients);
  if (!(norm > 0.0)) {
    throw std::domain_error("structure coefficients have a non-positive norm");
  }
  const Eigen::VectorXd cc =
      coefficients.array() * (overlap * coefficients).array() / norm;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(overlap);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("structure-overlap diagonalization failed");
  }
  const double spectral_scale = solver.eigenvalues().cwiseAbs().maxCoeff();
  const double rank_tolerance = std::numeric_limits<double>::epsilon() *
      static_cast<double>(overlap.rows()) * spectral_scale;
  if (solver.eigenvalues().minCoeff() < -rank_tolerance) {
    throw std::domain_error("structure-overlap matrix is not positive semidefinite");
  }
  const Eigen::VectorXd safe_eigenvalues =
      solver.eigenvalues().cwiseMax(0.0);
  const Eigen::MatrixXd square_root =
      solver.eigenvectors() * safe_eigenvalues.cwiseSqrt().asDiagonal() *
      solver.eigenvectors().transpose();
  Eigen::VectorXd lowdin = (square_root * coefficients).array().square();
  lowdin /= lowdin.sum();

  Eigen::VectorXd inverse_eigenvalues = Eigen::VectorXd::Zero(overlap.rows());
  for (int index = 0; index < inverse_eigenvalues.size(); ++index) {
    if (safe_eigenvalues[index] > rank_tolerance) {
      inverse_eigenvalues[index] = 1.0 / safe_eigenvalues[index];
    }
  }
  const Eigen::MatrixXd inverse_overlap =
      solver.eigenvectors() * inverse_eigenvalues.asDiagonal() *
      solver.eigenvectors().transpose();
  if ((inverse_overlap.diagonal().array() <= 0.0).any()) {
    throw std::domain_error(
        "inverse structure weights require positive inverse-overlap diagonals");
  }
  Eigen::VectorXd inverse = coefficients.array().square() /
      inverse_overlap.diagonal().array();
  inverse /= inverse.sum();

  Eigen::VectorXd renormalized = coefficients.array().square();
  renormalized /= renormalized.sum();

  output << "\n\n             ******  WEIGHTS OF STRUCTURES ******\n";
  print_weight_table(output, "Coulson-Chirgwin Weights", cc, structures);
  print_weight_table(output, "Lowdin Weights", lowdin, structures);
  print_weight_table(output, "Inverse Weights", inverse, structures);
  print_weight_table(output, "Renormalized Weights", renormalized, structures);
}

void print_orbital_table(
    std::ostream& output,
    const char* title,
    const vb::VbScfStaticMoleculeMetadata& metadata,
    const Eigen::MatrixXd& coefficients,
    const Eigen::VectorXd* occupations = nullptr) {
  const int n_bf = static_cast<int>(coefficients.rows());
  const int n_orb = static_cast<int>(coefficients.cols());
  if (n_bf <= 0 || n_orb <= 0) {
    return;
  }
  if (metadata.ao_to_atom.size() != static_cast<std::size_t>(n_bf) ||
      (occupations != nullptr && occupations->size() != n_orb)) {
    throw std::invalid_argument("orbital report dimensions are inconsistent");
  }

  output << "\n\n      ******  " << title << "  ******\n";
  constexpr int kColumnsPerBlock = 5;
  for (int first = 0; first < n_orb; first += kColumnsPerBlock) {
    const int last = std::min(first + kColumnsPerBlock, n_orb);
    output << "\n\n                         ";
    for (int orbital = first; orbital < last; ++orbital) {
      output << std::setw(11) << orbital + 1;
    }
    output << '\n';
    if (occupations != nullptr) {
      output << "                         ";
      for (int orbital = first; orbital < last; ++orbital) {
        output << std::fixed << std::setprecision(6) << std::setw(11)
               << (*occupations)[orbital];
      }
      output << '\n';
    }
    for (int basis = 0; basis < n_bf; ++basis) {
      const int atom = metadata.ao_to_atom[basis];
      const int atomic_number = metadata.atomic_numbers[atom];
      output << std::setw(5) << basis + 1 << "  "
             << std::setw(2) << element_symbol(atomic_number) << "  "
             << std::setw(3) << atom + 1 << "  "
             << std::left << std::setw(5) << ao_label(metadata, basis)
             << std::right;
      for (int orbital = first; orbital < last; ++orbital) {
        output << std::fixed << std::setprecision(6) << std::setw(11)
               << coefficients(basis, orbital);
      }
      output << '\n';
    }
  }
}

Eigen::MatrixXd sparse_orbital_matrix(
    const vb::OrbitalPreparationInput& orbitals) {
  const int n_bf = static_cast<int>(orbitals.n_basis_functions);
  const int n_orb = static_cast<int>(orbitals.n_orbitals);
  Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(n_bf, n_orb);
  for (int orbital = 0; orbital < n_orb; ++orbital) {
    const int count = vb::stored_sparse_orbital_coefficient_count(
        orbitals, orbital);
    for (int coefficient = 0; coefficient < count; ++coefficient) {
      const int offset = orbital * n_bf + coefficient;
      const int basis = orbitals.orbital_basis_index_table[offset] - 1;
      if (basis < 0 || basis >= n_bf) {
        throw std::out_of_range("orbital report contains an invalid AO index");
      }
      dense(basis, orbital) = orbitals.orbital_value_table[offset];
    }
  }
  return dense;
}

struct NaturalOrbitalAnalysis {
  Eigen::MatrixXd coefficients;
  Eigen::VectorXd occupations;
  Eigen::MatrixXd overlap_square_root;
};

NaturalOrbitalAnalysis compute_natural_orbitals(
    const Eigen::MatrixXd& density,
    const Eigen::MatrixXd& overlap,
    int n_orbitals,
    int n_electrons) {
  if (density.rows() != overlap.rows() || density.cols() != overlap.cols() ||
      density.rows() != density.cols() || n_orbitals <= 0 ||
      n_orbitals > density.rows()) {
    throw std::invalid_argument(
        "natural-orbital analysis dimensions are inconsistent");
  }
  const double electron_count = (density * overlap).trace();
  const double electron_count_tolerance =
      std::sqrt(std::numeric_limits<double>::epsilon()) *
      std::max(1.0, static_cast<double>(n_electrons));
  if (!std::isfinite(electron_count) ||
      std::abs(electron_count - static_cast<double>(n_electrons)) >
          electron_count_tolerance) {
    throw std::domain_error(
        "AO one-particle density does not integrate to the electron count");
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> overlap_solver(overlap);
  if (overlap_solver.info() != Eigen::Success ||
      overlap_solver.eigenvalues().minCoeff() <= 0.0) {
    throw std::domain_error(
        "AO overlap matrix is not positive definite in natural-orbital analysis");
  }
  const Eigen::MatrixXd overlap_square_root =
      overlap_solver.eigenvectors() *
      overlap_solver.eigenvalues().cwiseSqrt().asDiagonal() *
      overlap_solver.eigenvectors().transpose();
  const Eigen::MatrixXd overlap_inverse_square_root =
      overlap_solver.eigenvectors() *
      overlap_solver.eigenvalues().cwiseInverse().cwiseSqrt().asDiagonal() *
      overlap_solver.eigenvectors().transpose();
  const Eigen::MatrixXd orthogonal_density =
      overlap_square_root * density * overlap_square_root;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> density_solver(
      0.5 * (orthogonal_density + orthogonal_density.transpose()));
  if (density_solver.info() != Eigen::Success) {
    throw std::runtime_error("natural-orbital diagonalization failed");
  }

  NaturalOrbitalAnalysis analysis;
  analysis.coefficients.resize(density.rows(), n_orbitals);
  analysis.occupations.resize(n_orbitals);
  analysis.overlap_square_root = overlap_square_root;
  for (int orbital = 0; orbital < n_orbitals; ++orbital) {
    const int source = density.rows() - orbital - 1;
    analysis.occupations[orbital] = density_solver.eigenvalues()[source];
    analysis.coefficients.col(orbital) =
        overlap_inverse_square_root * density_solver.eigenvectors().col(source);
    Eigen::Index largest_index = 0;
    analysis.coefficients.col(orbital).cwiseAbs().maxCoeff(&largest_index);
    if (analysis.coefficients(largest_index, orbital) < 0.0) {
      analysis.coefficients.col(orbital) *= -1.0;
    }
  }
  return analysis;
}

void print_population_analysis(
    std::ostream& output,
    const vb::VbScfStaticMoleculeMetadata& metadata,
    const Eigen::MatrixXd& density,
    const Eigen::MatrixXd& overlap,
    const Eigen::MatrixXd& overlap_square_root,
    int spin_multiplicity) {
  const Eigen::MatrixXd mulliken_density = density * overlap;
  const Eigen::MatrixXd lowdin_density =
      overlap_square_root * density * overlap_square_root;
  std::vector<double> mulliken_population(metadata.n_atoms, 0.0);
  std::vector<double> lowdin_population(metadata.n_atoms, 0.0);
  for (int basis = 0; basis < density.rows(); ++basis) {
    const int atom = metadata.ao_to_atom[basis];
    mulliken_population[atom] += mulliken_density(basis, basis);
    lowdin_population[atom] += lowdin_density(basis, basis);
  }

  output << "\n\n                 ===============================================\n"
         << "                         XMVB ATOMIC POPULATION ANALYSIS\n"
         << "                 ===============================================\n\n"
         << "                     ******  POPULATION AND CHARGE  ******\n\n"
         << "       ATOM         MULL.POP.    CHARGE          LOW.POP.     CHARGE\n";
  for (int atom = 0; atom < metadata.n_atoms; ++atom) {
    const int atomic_number = metadata.atomic_numbers[atom];
    output << std::setw(5) << atom + 1 << ' '
           << std::left << std::setw(10) << element_symbol(atomic_number)
           << std::right << std::fixed << std::setprecision(6)
           << std::setw(12) << mulliken_population[atom]
           << std::setw(12) << atomic_number - mulliken_population[atom]
           << std::setw(17) << lowdin_population[atom]
           << std::setw(12) << atomic_number - lowdin_population[atom] << '\n';
  }

  if (spin_multiplicity == 1) {
    output << "\n              ******  ATOMIC SPIN POLARIZATION POPULATION ******\n\n"
           << "        ATOM          MULL.POP.                    LOW.POP.\n\n";
    for (int atom = 0; atom < metadata.n_atoms; ++atom) {
      output << std::setw(7) << atom + 1 << "   "
             << std::setw(2) << element_symbol(metadata.atomic_numbers[atom])
             << std::fixed << std::setprecision(6)
             << std::setw(19) << 0.0 << std::setw(29) << 0.0 << '\n';
    }
  }

  output << "\n                            ******  BOND ORDER  ******\n\n"
         << "                  ATOM 1        ATOM 2           DIST     BOND ORDER\n\n";
  constexpr double kBohrPerAngstrom = 1.8897261246257702;
  Eigen::MatrixXd bond_orders =
      Eigen::MatrixXd::Zero(metadata.n_atoms, metadata.n_atoms);
  for (int atom_a = 0; atom_a < metadata.n_atoms; ++atom_a) {
    for (int atom_b = atom_a + 1; atom_b < metadata.n_atoms; ++atom_b) {
      double bond_order = 0.0;
      for (int basis_a = 0; basis_a < density.rows(); ++basis_a) {
        if (metadata.ao_to_atom[basis_a] != atom_a) {
          continue;
        }
        for (int basis_b = 0; basis_b < density.rows(); ++basis_b) {
          if (metadata.ao_to_atom[basis_b] == atom_b) {
            bond_order += mulliken_density(basis_a, basis_b) *
                mulliken_density(basis_b, basis_a);
          }
        }
      }
      bond_orders(atom_a, atom_b) = bond_order;
      bond_orders(atom_b, atom_a) = bond_order;
      double squared_distance = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        const double displacement =
            metadata.atomic_coordinates[3 * atom_a + axis] -
            metadata.atomic_coordinates[3 * atom_b + axis];
        squared_distance += displacement * displacement;
      }
      output << std::setw(20) << atom_a + 1 << ' '
             << std::setw(2) << element_symbol(metadata.atomic_numbers[atom_a])
             << std::setw(11) << atom_b + 1 << ' '
             << std::setw(2) << element_symbol(metadata.atomic_numbers[atom_b])
             << std::fixed << std::setprecision(3) << std::setw(17)
             << std::sqrt(squared_distance) / kBohrPerAngstrom
             << std::setw(13) << bond_order << '\n';
    }
  }

  output << "\n                     ******    VALENCE ANALYSIS    ******\n\n"
         << "                                TOTAL       BONDED        FREE\n"
         << "               ATOM            VALENCE     VALENCE     VALENCE\n";
  for (int atom = 0; atom < metadata.n_atoms; ++atom) {
    double squared_bond_density = 0.0;
    for (int basis = 0; basis < density.rows(); ++basis) {
      if (metadata.ao_to_atom[basis] != atom) {
        continue;
      }
      for (int other_basis = 0; other_basis < density.cols(); ++other_basis) {
        if (metadata.ao_to_atom[other_basis] != atom) {
          continue;
        }
        squared_bond_density += mulliken_density(basis, other_basis) *
            mulliken_density(other_basis, basis);
      }
    }
    const double total_valence =
        2.0 * mulliken_population[atom] - squared_bond_density;
    const double bonded_valence = bond_orders.row(atom).sum();
    output << std::setw(14) << atom + 1 << ' '
           << std::left << std::setw(10)
           << element_symbol(metadata.atomic_numbers[atom]) << std::right
           << std::fixed << std::setprecision(3)
           << std::setw(13) << total_valence
           << std::setw(12) << bonded_valence
           << std::setw(12) << total_valence - bonded_valence << '\n';
  }
}

Eigen::MatrixXd build_kinetic_matrix(const vb::LibcintInput& input) {
  vb::LibcintDirectShellEvaluator evaluator(input);
  const int n_bf = evaluator.n_basis_functions();
  Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(n_bf, n_bf);
  for (int left_shell = 0; left_shell < input.n_shells; ++left_shell) {
    for (int right_shell = 0; right_shell <= left_shell; ++right_shell) {
      const auto block =
          evaluator.evaluate_kinetic_shell_pair(left_shell, right_shell);
      for (int column = 0; column < block.right_ao_count; ++column) {
        for (int row = 0; row < block.left_ao_count; ++row) {
          const double value = block.values[column * block.left_ao_count + row];
          const int global_row = block.left_ao_offset + row;
          const int global_column = block.right_ao_offset + column;
          matrix(global_row, global_column) = value;
          matrix(global_column, global_row) = value;
        }
      }
    }
  }
  return matrix;
}

std::array<Eigen::MatrixXd, 3> build_position_matrices(
    const vb::LibcintInput& input) {
  vb::LibcintDirectShellEvaluator evaluator(input);
  const int n_bf = evaluator.n_basis_functions();
  std::array<Eigen::MatrixXd, 3> matrices = {
      Eigen::MatrixXd::Zero(n_bf, n_bf),
      Eigen::MatrixXd::Zero(n_bf, n_bf),
      Eigen::MatrixXd::Zero(n_bf, n_bf)};
  for (int left_shell = 0; left_shell < input.n_shells; ++left_shell) {
    for (int right_shell = 0; right_shell <= left_shell; ++right_shell) {
      const auto block =
          evaluator.evaluate_position_shell_pair(left_shell, right_shell);
      const std::size_t component_size =
          block.left_ao_count * block.right_ao_count;
      for (int component = 0; component < 3; ++component) {
        for (int column = 0; column < block.right_ao_count; ++column) {
          for (int row = 0; row < block.left_ao_count; ++row) {
            const double value = block.values[
                component * component_size +
                column * block.left_ao_count + row];
            const int global_row = block.left_ao_offset + row;
            const int global_column = block.right_ao_offset + column;
            matrices[component](global_row, global_column) = value;
            matrices[component](global_column, global_row) = value;
          }
        }
      }
    }
  }
  return matrices;
}

void print_dipole_and_virial_analysis(
    std::ostream& output,
    const vb::VbScfInputLoadResult& load_result,
    const vb::VbScfOptimizerResult& result) {
  const Eigen::MatrixXd kinetic =
      build_kinetic_matrix(result.optimized_input.libcint_input);
  const auto positions =
      build_position_matrices(result.optimized_input.libcint_input);
  const Eigen::MatrixXd& density = result.one_particle_density_matrix;

  Eigen::Vector3d dipole = Eigen::Vector3d::Zero();
  for (int atom = 0;
       atom < load_result.static_molecule_metadata.n_atoms;
       ++atom) {
    for (int component = 0; component < 3; ++component) {
      dipole[component] +=
          load_result.static_molecule_metadata.atomic_numbers[atom] *
          load_result.static_molecule_metadata
              .atomic_coordinates[3 * atom + component];
    }
  }
  for (int component = 0; component < 3; ++component) {
    dipole[component] -=
        (density.array() * positions[component].array()).sum();
  }
  constexpr double kDebyePerAtomicUnit = 2.541746473;
  dipole *= kDebyePerAtomicUnit;
  output << "\n                     ****** DIPOLE MOMENT ANALYSIS ******\n\n"
         << "                DX         DY         DZ        TOTAL\n\n"
         << std::fixed << std::setprecision(6)
         << std::setw(20) << dipole.x()
         << std::setw(12) << dipole.y()
         << std::setw(12) << dipole.z()
         << std::setw(12) << dipole.norm() << '\n';

  const Eigen::MatrixXd& core_hamiltonian =
      result.optimized_input.ao_integral_input.ao_core_hamiltonian_matrix;
  const double kinetic_energy = (density.array() * kinetic.array()).sum();
  const double one_electron_energy =
      (density.array() * core_hamiltonian.array()).sum();
  const double nuclear_electron_potential =
      one_electron_energy - kinetic_energy;
  const double electronic_energy =
      result.final_total_energy - load_result.nuclear_repulsion_energy;
  const double two_electron_energy = electronic_energy - one_electron_energy;
  const double potential_energy = nuclear_electron_potential +
      two_electron_energy + load_result.nuclear_repulsion_energy;
  const double virial_ratio = -potential_energy / kinetic_energy;

  output << "\n\n                 ******    VIRIAL THEOREM ANALYSIS    ******\n\n"
         << std::fixed << std::setprecision(12)
         << "                      TOTAL ENERGY : " << std::setw(21)
         << result.final_total_energy << '\n'
         << "               NUCLEAR REP. ENERGY : " << std::setw(21)
         << load_result.nuclear_repulsion_energy << '\n'
         << "                 ELECTRONIC ENERGY : " << std::setw(21)
         << electronic_energy << '\n'
         << "               ONE-ELECTRON ENERGY : " << std::setw(21)
         << one_electron_energy << '\n'
         << "               TWO-ELECTRON ENERGY : " << std::setw(21)
         << two_electron_energy << '\n'
         << "                    KINETIC ENERGY : " << std::setw(21)
         << kinetic_energy << '\n'
         << "               NUC-ELE POT. ENERGY : " << std::setw(21)
         << nuclear_electron_potential << '\n'
         << "                  POTENTIAL ENERGY : " << std::setw(21)
         << potential_energy << '\n'
         << "              VIRIAL THEOREM VALUE : " << std::setw(21)
         << virial_ratio << '\n';
}

void print_orbitals(
    std::ostream& output,
    const vb::VbScfStaticMoleculeMetadata& metadata,
    const vb::OrbitalPreparationInput& orbitals) {
  print_orbital_table(
      output,
      "ORBITALS IN PRIMITIVE BASIS FUNCTIONS",
      metadata,
      sparse_orbital_matrix(orbitals));
}

}  // namespace

void print_program_preamble(
    std::ostream& output,
    const std::string& input_path,
    const std::chrono::system_clock::time_point& start_time,
    int n_threads) {
  output << "\n\n"
         << "    *************************************************************\n"
         << "                                                               \n"
         << "           M   M         MM MM         M   M         MMMM       \n"
         << "            M M          M M M         M   M         M   M      \n"
         << "             M           M M M          M M          MMMM       \n"
         << "            M M          M   M          M M          M   M      \n"
         << "           M   M         M   M           M           MMMM       \n"
         << "                                                               \n"
         << "    *************************************************************\n"
         << "                                                               \n"
         << "                    XMVB-CPP                                   \n"
         << "                    Version: development                       \n"
         << "                                                               \n"
         << "    Developed by: Tao Xia                                     \n"
         << "    Software development assistance: OpenAI Codex             \n\n\n"
         << " Job started at " << format_timestamp(start_time)
         << "  with " << std::setw(5) << n_threads << " processors.\n\n"
         << "Running Command: xmvb-cpp.exe " << input_path << "\n\n"
         << "Work Directory at " << fs::current_path().string()
         << " PID = " << getpid() << "\n\n";
}

void print_input_sections(
    std::ostream& output,
    const std::string& input_path,
    const vb::VbScfInputLoadResult& load_result,
    const char* optimizer_name,
    int max_iterations) {
  const auto& input = load_result.input;
  const auto& orbitals = input.orbital_preparation_input;
  const auto& molecule = load_result.static_molecule_metadata;
  const auto& libcint = input.libcint_input;

  const std::string deck = read_input_file(input_path);
  output << "---------------Input File---------------\n"
         << deck;
  if (deck.empty() || deck.back() != '\n') {
    output << '\n';
  }
  output << "---------------End of Input--------------\n\n";

  output << " ATOM      ATOMIC                      COORDINATES (BOHR)\n"
         << "           CHARGE         X                   Y                   Z\n";
  for (int atom = 0; atom < molecule.n_atoms; ++atom) {
    output << std::setw(3) << element_symbol(molecule.atomic_numbers[atom])
           << std::fixed << std::setprecision(1) << std::setw(13)
           << static_cast<double>(molecule.atomic_numbers[atom]);
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      output << std::fixed << std::setprecision(8) << std::setw(20)
             << molecule.atomic_coordinates[3 * atom + coordinate];
    }
    output << '\n';
  }

  output << "\n     ATOMIC BASIS SET\n"
         << "     ----------------\n"
         << " THE CONTRACTED PRIMITIVE FUNCTIONS HAVE BEEN UNNORMALIZED\n"
         << " THE CONTRACTED BASIS FUNCTIONS ARE NOW NORMALIZED TO UNITY\n\n"
         << "  SHELL TYPE  PRIMITIVE        EXPONENT          CONTRACTION COEFFICIENT(S)\n";
  int primitive = 0;
  constexpr std::array<char, 6> kShellType = {'S', 'P', 'D', 'F', 'G', 'H'};
  for (int atom = 0; atom < libcint.n_atoms; ++atom) {
    output << "\n " << element_symbol(molecule.atomic_numbers[atom]) << " \n\n";
    for (int shell = 0; shell < libcint.n_shells; ++shell) {
      const int shell_offset = shell * BAS_SLOTS;
      if (libcint.bas[shell_offset + ATOM_OF] != atom) {
        continue;
      }
      const int angular_momentum = libcint.bas[shell_offset + ANG_OF];
      const int n_primitives = libcint.bas[shell_offset + NPRIM_OF];
      const int exponent_offset = libcint.bas[shell_offset + PTR_EXP];
      const int coefficient_offset = libcint.bas[shell_offset + PTR_COEFF];
      for (int local = 0; local < n_primitives; ++local) {
        ++primitive;
        output << std::setw(7) << shell + 1 << "   "
               << (angular_momentum >= 0 && angular_momentum < 6
                       ? kShellType[angular_momentum]
                       : '?')
               << std::setw(8) << primitive
               << std::fixed << std::setprecision(7) << std::setw(22)
               << libcint.env[exponent_offset + local]
               << std::fixed << std::setprecision(12) << std::setw(18)
               << libcint.env[coefficient_offset + local] << '\n';
      }
      output << '\n';
    }
  }

  const int molecular_charge =
      std::accumulate(
          molecule.atomic_numbers.begin(), molecule.atomic_numbers.end(), 0) -
      static_cast<int>(orbitals.n_total_electrons);
  output << " TOTAL NUMBER OF BASIS SET SHELLS             = "
         << std::setw(4) << molecule.n_shells << '\n'
         << " NUMBER OF CARTESIAN GAUSSIAN BASIS FUNCTIONS = "
         << std::setw(4) << orbitals.n_basis_functions << '\n'
         << " NUMBER OF ELECTRONS                          = "
         << std::setw(4) << orbitals.n_total_electrons << '\n'
         << " CHARGE OF MOLECULE                           = "
         << std::setw(4) << molecular_charge << '\n'
         << " SPIN MULTIPLICITY                            = "
         << std::setw(4) << orbitals.spin_multiplicity << '\n'
         << " TOTAL NUMBER OF ATOMS                        = "
         << std::setw(4) << molecule.n_atoms << "\n\n";
  print_ionic_class_summary(
      output, load_result.raw_structure_data, input.structure_data);
  output << " Number of structures: " << std::setw(5)
         << load_result.raw_structure_data.n_structures << "\n\n"
         << " The following structures are used in calculation"
            " (First 10 structures if more than 10):\n\n";
  const int n_printed_structures =
      std::min(10, load_result.raw_structure_data.n_structures);
  for (int structure = 0; structure < n_printed_structures; ++structure) {
    output << std::setw(8) << structure + 1 << "   ******    "
           << structure_description(load_result.raw_structure_data, structure)
           << '\n';
  }

  int n_variables = 0;
  for (std::size_t orbital = 0; orbital < orbitals.n_orbitals; ++orbital) {
    n_variables += vb::differentiable_sparse_orbital_parameter_count(
        orbitals, orbital);
  }
  output << "\n\n Number of variables for VBSCF/BOVB : " << n_variables
         << "\n\n VBSCF algorithm: " << optimizer_name
         << ".\n\n Maximum number of Iterations: " << max_iterations << "\n\n"
         << " Integral evaluation: "
         << (load_result.standard_two_electron_mode == vb::StandardTwoElectronMode::Exact
                 ? "precise integrals by Libcint."
                 : "resolution-of-identity integrals by Libcint.")
         << '\n'
         << " 2-e integral strategy: "
         << (load_result.standard_two_electron_mode == vb::StandardTwoElectronMode::Exact
                 ? "Continuous storage."
                 : "Factorized storage.")
         << '\n';
  if (load_result.standard_two_electron_mode == vb::StandardTwoElectronMode::Exact) {
    output << " Non-zero 2-e integrals: "
           << input.ao_integral_input.ao_two_electron_integral_values.size()
           << '\n';
  }

  output << "\n---------------Initial Guess---------------\n";
  for (std::size_t orbital = 0; orbital < orbitals.n_orbitals; ++orbital) {
    output << std::setw(6)
           << vb::stored_sparse_orbital_coefficient_count(orbitals, orbital);
    if ((orbital + 1) % 10 == 0 || orbital + 1 == orbitals.n_orbitals) {
      output << '\n';
    }
  }
  for (std::size_t orbital = 0; orbital < orbitals.n_orbitals; ++orbital) {
    const int count = vb::stored_sparse_orbital_coefficient_count(orbitals, orbital);
    for (int coefficient = 0; coefficient < count; ++coefficient) {
      const std::size_t offset = orbital * orbitals.n_basis_functions + coefficient;
      output << std::fixed << std::setprecision(10) << std::setw(14)
             << orbitals.orbital_value_table[offset]
             << std::setw(6) << orbitals.orbital_basis_index_table[offset] << "  ";
      if ((coefficient + 1) % 4 == 0 || coefficient + 1 == count) {
        output << '\n';
      }
    }
  }
  output << "---------------End of Guess--------------\n";
}

void print_final_state_sections(
    std::ostream& output,
    const vb::VbScfInputLoadResult& load_result,
    const vb::VbScfOptimizerResult& result) {
  const auto& scf = result.scf_result;
  const NormalizedStructureState normalized = normalize_structure_state(scf);

  print_structure_matrix(output, "OVERLAP", normalized.overlap);
  print_structure_matrix(
      output, "HAMILTONIAN", normalized.electronic_hamiltonian);
  if (normalized.coefficients.size() != 0) {
    output << "\n\n              ******  COEFFICIENTS OF STRUCTURES ******\n\n";
    for (int structure = 0; structure < normalized.coefficients.size(); ++structure) {
      output << std::setw(8) << structure + 1
             << std::fixed << std::setprecision(8) << std::setw(17)
             << normalized.coefficients[structure] << "  ******    "
             << structure_description(load_result.raw_structure_data, structure)
             << '\n';
    }
    const auto& orbitals = result.optimized_input.orbital_preparation_input;
    const int n_inactive =
        (orbitals.n_total_electrons - orbitals.n_active_electrons) / 2;
    print_determinant_coefficients(
        output,
        normalized.coefficients,
        result.optimized_input.structure_data,
        n_inactive);
    print_structure_weights(
        output,
        normalized.coefficients,
        normalized.overlap,
        load_result.raw_structure_data);
  }
  print_orbitals(
      output,
      load_result.static_molecule_metadata,
      result.optimized_input.orbital_preparation_input);
  const auto& orbitals = result.optimized_input.orbital_preparation_input;
  const NaturalOrbitalAnalysis natural_orbitals = compute_natural_orbitals(
      result.one_particle_density_matrix,
      orbitals.ao_overlap_matrix,
      static_cast<int>(orbitals.n_orbitals),
      static_cast<int>(orbitals.n_total_electrons));
  print_orbital_table(
      output,
      "COMPUTED NATURAL ORBITALS",
      load_result.static_molecule_metadata,
      natural_orbitals.coefficients,
      &natural_orbitals.occupations);
  print_population_analysis(
      output,
      load_result.static_molecule_metadata,
      result.one_particle_density_matrix,
      orbitals.ao_overlap_matrix,
      natural_orbitals.overlap_square_root,
      orbitals.spin_multiplicity);
  print_dipole_and_virial_analysis(output, load_result, result);
}

}  // namespace xmvb::output
