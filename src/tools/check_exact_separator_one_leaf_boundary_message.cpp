#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vb/exact_separator/leaf_boundary_message.hpp"
#include "vb/exact_separator/spin_state_aggregate.hpp"
#include "vb/exact_separator/two_electron.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace {

using xmvb::vb::DeterminantOverlapResolver;
using xmvb::vb::Matrix;
using xmvb::vb::TwoElectronIndexer;
using xmvb::vb::exact_separator::CollapsedTwoElectronOneLeafStarPairStats;
using xmvb::vb::exact_separator::ComponentData;
using xmvb::vb::exact_separator::DirectSpinStateAggregate;
using xmvb::vb::exact_separator::OneLeafBoundarySpinAggregate;
using xmvb::vb::exact_separator::OneLeafBoundarySpinMessage;
using xmvb::vb::exact_separator::OrientationTerm;

struct TestCase {
  std::string name;
  std::vector<ComponentData> ordered_components;
  std::vector<double> support_overlap_storage;
  std::vector<double> packed_active_two_electron_integrals;
  int support_size = 0;
};

std::vector<double> build_support_overlap_storage(int support_size) {
  std::vector<double> storage(
      xmvb::to_size(support_size * support_size),
      0.0);
  for (int col = 0; col < support_size; ++col) {
    for (int row = 0; row < support_size; ++row) {
      const double value =
          (row == col)
              ? (1.0 + 0.04 * static_cast<double>(row + 1))
              : (0.03 * static_cast<double>(row + 1) +
                 0.02 * static_cast<double>(col + 1));
      storage[xmvb::to_size(col * support_size + row)] = value;
    }
  }
  return storage;
}

std::vector<double> build_packed_eri_storage(int support_size) {
  const int packed_size =
      TwoElectronIndexer::two_electron_storage_index(
          support_size - 1,
          support_size - 1,
          support_size - 1,
          support_size - 1) +
      1;
  std::vector<double> packed(
      xmvb::to_size(packed_size),
      0.0);
  for (int p = 0; p < support_size; ++p) {
    for (int q = 0; q < support_size; ++q) {
      for (int r = 0; r < support_size; ++r) {
        for (int s = 0; s < support_size; ++s) {
          packed[xmvb::to_size(
              TwoElectronIndexer::two_electron_storage_index(p, q, r, s))] =
              0.20 +
              0.11 * static_cast<double>(p + 1) +
              0.07 * static_cast<double>(q + 1) +
              0.05 * static_cast<double>(r + 1) +
              0.03 * static_cast<double>(s + 1);
        }
      }
    }
  }
  return packed;
}

TestCase make_balanced_case() {
  ComponentData root_component;
  root_component.graph_node = 0;
  root_component.left_orientation_terms.push_back(OrientationTerm{{2}, {2}, 1.0});
  root_component.right_orientation_terms.push_back(OrientationTerm{{0}, {0}, 1.0});

  ComponentData leaf_component;
  leaf_component.graph_node = 1;
  leaf_component.left_orientation_terms.push_back(OrientationTerm{{0}, {0}, 1.0});
  leaf_component.right_orientation_terms.push_back(OrientationTerm{{2}, {2}, 1.0});

  TestCase test_case;
  test_case.name = "balanced";
  test_case.ordered_components = {root_component, leaf_component};
  test_case.support_size = 3;
  test_case.support_overlap_storage = build_support_overlap_storage(3);
  test_case.packed_active_two_electron_integrals = build_packed_eri_storage(3);
  return test_case;
}

TestCase make_row_excess_two_case() {
  ComponentData root_component;
  root_component.graph_node = 0;
  root_component.left_orientation_terms.push_back(OrientationTerm{{2, 3}, {}, 1.0});
  root_component.right_orientation_terms.push_back(OrientationTerm{{}, {}, 1.0});

  ComponentData leaf_component;
  leaf_component.graph_node = 1;
  leaf_component.left_orientation_terms.push_back(OrientationTerm{{}, {}, 1.0});
  leaf_component.right_orientation_terms.push_back(OrientationTerm{{0, 1}, {}, 1.0});

  TestCase test_case;
  test_case.name = "row_excess_two";
  test_case.ordered_components = {root_component, leaf_component};
  test_case.support_size = 4;
  test_case.support_overlap_storage = build_support_overlap_storage(4);
  test_case.packed_active_two_electron_integrals = build_packed_eri_storage(4);
  return test_case;
}

TestCase make_col_excess_two_case() {
  ComponentData root_component;
  root_component.graph_node = 0;
  root_component.left_orientation_terms.push_back(OrientationTerm{{}, {}, 1.0});
  root_component.right_orientation_terms.push_back(OrientationTerm{{2, 3}, {}, 1.0});

  ComponentData leaf_component;
  leaf_component.graph_node = 1;
  leaf_component.left_orientation_terms.push_back(OrientationTerm{{0, 1}, {}, 1.0});
  leaf_component.right_orientation_terms.push_back(OrientationTerm{{}, {}, 1.0});

  TestCase test_case;
  test_case.name = "col_excess_two";
  test_case.ordered_components = {root_component, leaf_component};
  test_case.support_size = 4;
  test_case.support_overlap_storage = build_support_overlap_storage(4);
  test_case.packed_active_two_electron_integrals = build_packed_eri_storage(4);
  return test_case;
}

TestCase make_row_open_case() {
  ComponentData root_component;
  root_component.graph_node = 0;
  root_component.left_orientation_terms.push_back(OrientationTerm{{2, 3}, {}, 1.0});
  root_component.right_orientation_terms.push_back(OrientationTerm{{2}, {}, 1.0});

  ComponentData leaf_component;
  leaf_component.graph_node = 1;
  leaf_component.left_orientation_terms.push_back(OrientationTerm{{0}, {}, 1.0});
  leaf_component.right_orientation_terms.push_back(OrientationTerm{{0, 1}, {}, 1.0});

  TestCase test_case;
  test_case.name = "row_open";
  test_case.ordered_components = {root_component, leaf_component};
  test_case.support_size = 4;
  test_case.support_overlap_storage = build_support_overlap_storage(4);
  test_case.packed_active_two_electron_integrals = build_packed_eri_storage(4);
  return test_case;
}

std::vector<int> concatenate_occ(
    const std::vector<int>& first,
    const std::vector<int>& second) {
  std::vector<int> result;
  result.reserve(first.size() + second.size());
  result.insert(result.end(), first.begin(), first.end());
  result.insert(result.end(), second.begin(), second.end());
  return result;
}

double max_abs_matrix_difference(const Matrix& left, const Matrix& right) {
  if (left.rows() != right.rows() || left.cols() != right.cols()) {
    throw std::invalid_argument("matrix dimension mismatch");
  }
  double max_abs_error = 0.0;
  for (int col = 0; col < left.cols(); ++col) {
    for (int row = 0; row < left.rows(); ++row) {
      max_abs_error = std::max(
          max_abs_error,
          std::abs(left(row, col) - right(row, col)));
    }
  }
  return max_abs_error;
}

std::size_t total_sector_count(const OneLeafBoundarySpinMessage& message) {
  std::size_t total = 0U;
  for (const auto& family : message.families) {
    total += family.sectors.size();
  }
  return total;
}

double contract_opposite_spin_from_first_cofactors(
    const Matrix& alpha_first_cofactor,
    const Matrix& beta_first_cofactor,
    int support_size,
    const std::vector<double>& packed_active_two_electron_integrals) {
  double total = 0.0;
  for (int beta_col = 0; beta_col < support_size; ++beta_col) {
    for (int beta_row = 0; beta_row < support_size; ++beta_row) {
      const double beta_value = beta_first_cofactor(beta_row, beta_col);
      if (std::abs(beta_value) <= 1.0e-15) {
        continue;
      }
      for (int alpha_col = 0; alpha_col < support_size; ++alpha_col) {
        for (int alpha_row = 0; alpha_row < support_size; ++alpha_row) {
          const double alpha_value = alpha_first_cofactor(alpha_row, alpha_col);
          if (std::abs(alpha_value) <= 1.0e-15) {
            continue;
          }
          const int eri_index =
              TwoElectronIndexer::two_electron_storage_index(
                  beta_row,
                  beta_col,
                  alpha_row,
                  alpha_col);
          total +=
              packed_active_two_electron_integrals[xmvb::to_size(eri_index)] *
              alpha_value *
              beta_value;
        }
      }
    }
  }
  return total;
}

DirectSpinStateAggregate build_direct_spin_reference(
    const std::vector<int>& left_root_occ,
    const std::vector<int>& left_leaf_occ,
    const std::vector<int>& right_root_occ,
    const std::vector<int>& right_leaf_occ,
    const std::vector<double>& support_overlap_storage,
    const std::vector<double>& packed_active_two_electron_integrals,
    int support_size,
    std::uint64_t* subdeterminant_evaluations) {
  const std::vector<int> full_left_occ =
      concatenate_occ(left_root_occ, left_leaf_occ);
  const std::vector<int> full_right_occ =
      concatenate_occ(right_root_occ, right_leaf_occ);
  const std::vector<double> zero_one_electron_storage(
      xmvb::to_size(support_size) * xmvb::to_size(support_size),
      0.0);
  return xmvb::vb::exact_separator::build_direct_spin_state_aggregate(
      full_left_occ,
      full_right_occ,
      support_overlap_storage,
      zero_one_electron_storage,
      packed_active_two_electron_integrals,
      support_size,
      true,
      DeterminantOverlapResolver(),
      xmvb::vb::DeterminantHamiltonianResolver(DeterminantOverlapResolver()),
      subdeterminant_evaluations);
}

bool run_case(const TestCase& test_case) {
  const ComponentData& root_component = test_case.ordered_components.front();
  const ComponentData& leaf_component = test_case.ordered_components.back();
  if (root_component.left_orientation_terms.size() != 1U ||
      root_component.right_orientation_terms.size() != 1U ||
      leaf_component.left_orientation_terms.size() != 1U ||
      leaf_component.right_orientation_terms.size() != 1U) {
    throw std::invalid_argument("boundary-message checker expects one orientation term per side");
  }

  const OrientationTerm& left_root_term = root_component.left_orientation_terms.front();
  const OrientationTerm& right_root_term = root_component.right_orientation_terms.front();
  const OrientationTerm& left_leaf_term = leaf_component.left_orientation_terms.front();
  const OrientationTerm& right_leaf_term = leaf_component.right_orientation_terms.front();

  std::uint64_t alpha_boundary_subdeterminants = 0;
  const OneLeafBoundarySpinMessage alpha_message =
      xmvb::vb::exact_separator::build_one_leaf_spin_boundary_message(
          left_root_term.alpha_occ,
          left_leaf_term.alpha_occ,
          right_root_term.alpha_occ,
          right_leaf_term.alpha_occ,
          test_case.support_overlap_storage,
          test_case.support_size,
          DeterminantOverlapResolver(),
          &alpha_boundary_subdeterminants);
  const OneLeafBoundarySpinAggregate alpha_aggregate =
      xmvb::vb::exact_separator::contract_one_leaf_spin_boundary_message(
          alpha_message,
          test_case.packed_active_two_electron_integrals);

  std::uint64_t beta_boundary_subdeterminants = 0;
  const OneLeafBoundarySpinMessage beta_message =
      xmvb::vb::exact_separator::build_one_leaf_spin_boundary_message(
          left_root_term.beta_occ,
          left_leaf_term.beta_occ,
          right_root_term.beta_occ,
          right_leaf_term.beta_occ,
          test_case.support_overlap_storage,
          test_case.support_size,
          DeterminantOverlapResolver(),
          &beta_boundary_subdeterminants);
  const OneLeafBoundarySpinAggregate beta_aggregate =
      xmvb::vb::exact_separator::contract_one_leaf_spin_boundary_message(
          beta_message,
          test_case.packed_active_two_electron_integrals);

  std::uint64_t alpha_direct_subdeterminants = 0;
  const DirectSpinStateAggregate alpha_direct =
      build_direct_spin_reference(
          left_root_term.alpha_occ,
          left_leaf_term.alpha_occ,
          right_root_term.alpha_occ,
          right_leaf_term.alpha_occ,
          test_case.support_overlap_storage,
          test_case.packed_active_two_electron_integrals,
          test_case.support_size,
          &alpha_direct_subdeterminants);
  std::uint64_t beta_direct_subdeterminants = 0;
  const DirectSpinStateAggregate beta_direct =
      build_direct_spin_reference(
          left_root_term.beta_occ,
          left_leaf_term.beta_occ,
          right_root_term.beta_occ,
          right_leaf_term.beta_occ,
          test_case.support_overlap_storage,
          test_case.packed_active_two_electron_integrals,
          test_case.support_size,
          &beta_direct_subdeterminants);

  const double alpha_first_cofactor_error =
      max_abs_matrix_difference(alpha_aggregate.first_cofactor, alpha_direct.first_cofactor);
  const double beta_first_cofactor_error =
      max_abs_matrix_difference(beta_aggregate.first_cofactor, beta_direct.first_cofactor);

  const CollapsedTwoElectronOneLeafStarPairStats full_stats =
      xmvb::vb::exact_separator::
          evaluate_component_ordered_open_state_star_pair_two_electron_one_leaf(
              test_case.support_overlap_storage,
              test_case.packed_active_two_electron_integrals,
              test_case.support_size,
              test_case.ordered_components,
              DeterminantOverlapResolver());

  const double reconstructed_overlap =
      alpha_aggregate.overlap * beta_aggregate.overlap;
  const double reconstructed_same_spin_alpha =
      alpha_aggregate.same_spin_two_electron * beta_aggregate.overlap;
  const double reconstructed_same_spin_beta =
      alpha_aggregate.overlap * beta_aggregate.same_spin_two_electron;
  const double reconstructed_opposite_spin =
      contract_opposite_spin_from_first_cofactors(
          alpha_aggregate.first_cofactor,
          beta_aggregate.first_cofactor,
          test_case.support_size,
          test_case.packed_active_two_electron_integrals);
  const double reconstructed_total =
      reconstructed_same_spin_alpha +
      reconstructed_same_spin_beta +
      reconstructed_opposite_spin;

  const double alpha_overlap_error =
      std::abs(alpha_aggregate.overlap - alpha_direct.overlap);
  const double beta_overlap_error =
      std::abs(beta_aggregate.overlap - beta_direct.overlap);
  const double alpha_same_spin_error =
      std::abs(alpha_aggregate.same_spin_two_electron - alpha_direct.same_spin_two_electron);
  const double beta_same_spin_error =
      std::abs(beta_aggregate.same_spin_two_electron - beta_direct.same_spin_two_electron);
  const double overlap_error =
      std::abs(reconstructed_overlap - full_stats.exact_overlap);
  const double same_spin_alpha_error =
      std::abs(
          reconstructed_same_spin_alpha -
          full_stats.exact_same_spin_alpha_two_electron);
  const double same_spin_beta_error =
      std::abs(
          reconstructed_same_spin_beta -
          full_stats.exact_same_spin_beta_two_electron);
  const double opposite_spin_error =
      std::abs(
          reconstructed_opposite_spin -
          full_stats.exact_opposite_spin_two_electron);
  const double total_error =
      std::abs(reconstructed_total - full_stats.exact_two_electron);

  std::cout << "case = " << test_case.name << '\n';
  std::cout << "  alpha_family_count = "
            << alpha_message.families.size() << '\n';
  std::cout << "  alpha_sector_count = "
            << total_sector_count(alpha_message) << '\n';
  for (const auto& family : alpha_message.families) {
    std::cout << "    alpha_family_delta = "
              << family.family_indexer.family_delta
              << ", sectors = "
              << family.sectors.size() << '\n';
  }
  std::cout << "  beta_family_count = "
            << beta_message.families.size() << '\n';
  std::cout << "  beta_sector_count = "
            << total_sector_count(beta_message) << '\n';
  for (const auto& family : beta_message.families) {
    std::cout << "    beta_family_delta = "
              << family.family_indexer.family_delta
              << ", sectors = "
              << family.sectors.size() << '\n';
  }
  std::cout << "  alpha_boundary_subdeterminants = "
            << alpha_boundary_subdeterminants << '\n';
  std::cout << "  beta_boundary_subdeterminants = "
            << beta_boundary_subdeterminants << '\n';
  std::cout << "  alpha_direct_subdeterminants = "
            << alpha_direct_subdeterminants << '\n';
  std::cout << "  beta_direct_subdeterminants = "
            << beta_direct_subdeterminants << '\n';
  std::cout << "  alpha_overlap_error = " << alpha_overlap_error << '\n';
  std::cout << "  beta_overlap_error = " << beta_overlap_error << '\n';
  std::cout << "  alpha_first_cofactor_error = "
            << alpha_first_cofactor_error << '\n';
  std::cout << "  beta_first_cofactor_error = "
            << beta_first_cofactor_error << '\n';
  std::cout << "  alpha_same_spin_error = "
            << alpha_same_spin_error << '\n';
  std::cout << "  beta_same_spin_error = "
            << beta_same_spin_error << '\n';
  std::cout << "  overlap_error = " << overlap_error << '\n';
  std::cout << "  same_spin_alpha_error = "
            << same_spin_alpha_error << '\n';
  std::cout << "  same_spin_beta_error = "
            << same_spin_beta_error << '\n';
  std::cout << "  opposite_spin_error = "
            << opposite_spin_error << '\n';
  std::cout << "  total_error = " << total_error << '\n';

  const bool ok =
      alpha_overlap_error <= 1.0e-12 &&
      beta_overlap_error <= 1.0e-12 &&
      alpha_first_cofactor_error <= 1.0e-12 &&
      beta_first_cofactor_error <= 1.0e-12 &&
      alpha_same_spin_error <= 1.0e-12 &&
      beta_same_spin_error <= 1.0e-12 &&
      overlap_error <= 1.0e-12 &&
      same_spin_alpha_error <= 1.0e-12 &&
      same_spin_beta_error <= 1.0e-12 &&
      opposite_spin_error <= 1.0e-12 &&
      total_error <= 1.0e-12;
  if (!ok) {
    std::cerr << "one-leaf boundary-message exact check failed for case "
              << test_case.name << '\n';
  }
  return ok;
}

int main_impl() {
  std::cout << std::setprecision(15);
  const std::vector<TestCase> test_cases{
      make_balanced_case(),
      make_row_open_case(),
      make_row_excess_two_case(),
      make_col_excess_two_case(),
  };

  bool all_ok = true;
  for (const auto& test_case : test_cases) {
    all_ok = run_case(test_case) && all_ok;
  }
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace

int main() {
  try {
    return main_impl();
  } catch (const std::exception& exception) {
    std::cerr << "error: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
}
