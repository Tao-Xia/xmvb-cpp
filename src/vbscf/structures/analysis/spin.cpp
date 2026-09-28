#include "vbscf/structures/analysis/spin.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xmvb::vb {
namespace {

struct RaisedDeterminant {
  std::vector<int> alpha;
  std::vector<int> beta;

  bool operator==(const RaisedDeterminant& other) const noexcept {
    return alpha == other.alpha && beta == other.beta;
  }
};

struct RaisedDeterminantHash {
  std::size_t operator()(const RaisedDeterminant& determinant) const noexcept {
    std::size_t value = 0;
    for (const int orbital : determinant.alpha) {
      value = value * 1315423911u + static_cast<std::size_t>(orbital + 257);
    }
    value = value * 2654435761u + 17u;
    for (const int orbital : determinant.beta) {
      value = value * 1315423911u + static_cast<std::size_t>(orbital + 257);
    }
    return value;
  }
};

struct DeterminantTerm {
  int determinant = 0;
  double coefficient = 0.0;
};

void validate_spin_string(const std::vector<int>& occupied) {
  if (!std::is_sorted(occupied.begin(), occupied.end()) ||
      std::adjacent_find(occupied.begin(), occupied.end()) != occupied.end() ||
      (!occupied.empty() && occupied.front() < 0)) {
    throw std::invalid_argument(
        "spin analysis requires sorted unique non-negative occupations");
  }
}

}  // namespace

StructureSpinAnalysis analyze_structure_spin(
    const FullDeterminantStructureData& structures,
    int spin_multiplicity) {
  if (spin_multiplicity <= 0 || structures.n_structures <= 0 ||
      structures.alpha_det.empty() ||
      structures.alpha_det.size() != structures.beta_det.size() ||
      structures.alpha_det.size() !=
          structures.determinant_to_structure_terms.size()) {
    throw std::invalid_argument("structure spin dimensions are inconsistent");
  }

  const int n_alpha = static_cast<int>(structures.alpha_det.front().size());
  const int n_beta = static_cast<int>(structures.beta_det.front().size());
  if (n_alpha - n_beta != spin_multiplicity - 1) {
    throw std::invalid_argument(
        "determinant spin projection differs from the requested multiplicity");
  }

  std::vector<std::vector<DeterminantTerm>> structure_terms(
      structures.n_structures);
  for (int determinant = 0;
       determinant < static_cast<int>(structures.alpha_det.size());
       ++determinant) {
    const auto& alpha = structures.alpha_det[determinant];
    const auto& beta = structures.beta_det[determinant];
    validate_spin_string(alpha);
    validate_spin_string(beta);
    if (static_cast<int>(alpha.size()) != n_alpha ||
        static_cast<int>(beta.size()) != n_beta) {
      throw std::invalid_argument(
          "structure expansion mixes electron-number sectors");
    }
    for (const StructureExpansionTerm& term :
         structures.determinant_to_structure_terms[determinant]) {
      if (term.structure_index < 0 ||
          term.structure_index >= structures.n_structures ||
          !std::isfinite(term.coefficient)) {
        throw std::invalid_argument(
            "structure spin expansion term is invalid");
      }
      structure_terms[term.structure_index].push_back(
          DeterminantTerm{determinant, term.coefficient});
    }
  }

  StructureSpinAnalysis result;
  result.spin = 0.5 * static_cast<double>(spin_multiplicity - 1);
  result.spin_squared = result.spin * (result.spin + 1.0);

  for (int structure = 0; structure < structures.n_structures; ++structure) {
    if (structure_terms[structure].empty()) {
      throw std::invalid_argument("structure has no determinant expansion");
    }
    std::unordered_map<RaisedDeterminant, double, RaisedDeterminantHash>
        raised_coefficients;
    for (const DeterminantTerm& term : structure_terms[structure]) {
      const auto& alpha = structures.alpha_det[term.determinant];
      const auto& beta = structures.beta_det[term.determinant];
      for (int beta_position = 0;
           beta_position < static_cast<int>(beta.size());
           ++beta_position) {
        const int orbital = beta[beta_position];
        const auto alpha_position =
            std::lower_bound(alpha.begin(), alpha.end(), orbital);
        if (alpha_position != alpha.end() && *alpha_position == orbital) {
          continue;
        }
        const int insertion_position =
            static_cast<int>(alpha_position - alpha.begin());
        const int fermion_parity =
            n_alpha + beta_position + insertion_position;
        const double sign = fermion_parity % 2 == 0 ? 1.0 : -1.0;

        RaisedDeterminant raised{alpha, beta};
        raised.alpha.insert(
            raised.alpha.begin() + insertion_position,
            orbital);
        raised.beta.erase(raised.beta.begin() + beta_position);
        raised_coefficients[std::move(raised)] += sign * term.coefficient;
      }
    }

    double structure_residual = 0.0;
    for (const auto& [determinant, coefficient] : raised_coefficients) {
      static_cast<void>(determinant);
      structure_residual = std::max(structure_residual, std::abs(coefficient));
    }
    if (structure_residual > result.maximum_raising_residual) {
      result.maximum_raising_residual = structure_residual;
      result.worst_structure_index = structure;
    }
  }
  return result;
}

}  // namespace xmvb::vb
