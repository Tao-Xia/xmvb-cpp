#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vb/exact_separator/two_electron.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace {

using xmvb::vb::DeterminantOverlapResolver;
using xmvb::vb::TwoElectronIndexer;
using xmvb::vb::exact_separator::CollapsedTwoElectronOneLeafStarPairStats;
using xmvb::vb::exact_separator::ComponentData;
using xmvb::vb::exact_separator::OrientationTerm;

struct TestCase {
  std::string name;
  std::vector<ComponentData> ordered_components;
  std::vector<double> support_overlap_storage;
  std::vector<double> packed_active_two_electron_integrals;
  int support_size = 0;
  bool expect_alpha_nonzero = false;
  bool expect_beta_nonzero = false;
  bool expect_opposite_nonzero = false;
};

std::vector<double> build_support_overlap_storage(int support_size) {
  // Builds a mildly nonorthogonal dense overlap matrix in the repository-wide
  // column-major storage convention `storage[col * support_size + row]`.
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
  // Fills the packed ERI tensor with deterministic non-symmetric values so the
  // direct-minus-exchange same-spin contractions are guaranteed to be
  // nontrivial in the validation cases below.
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
  // Balanced one-root/one-leaf case with all three channels nonzero. This is
  // the original `d = 0` style regression case.
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
  test_case.expect_alpha_nonzero = true;
  test_case.expect_beta_nonzero = true;
  test_case.expect_opposite_nonzero = true;
  return test_case;
}

TestCase make_row_excess_two_case() {
  // Alpha-only case whose one-leaf masks include frontier sectors with
  // `d = +2`. This specifically exercises the `U_F wedge V_R` branch of the
  // new same-spin merge.
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
  test_case.expect_alpha_nonzero = true;
  return test_case;
}

TestCase make_col_excess_two_case() {
  // Alpha-only case whose one-leaf masks include frontier sectors with
  // `d = -2`. This specifically exercises the `V_F wedge U_R` branch.
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
  test_case.expect_alpha_nonzero = true;
  return test_case;
}

TestCase make_row_open_case() {
  // Alpha-only case whose valid masks include frontier sectors with
  // `d = +1` and `d = -1`. This exercises the mixed `P_F / Q_F` with
  // `v_R / u_R` branches of the same-spin merge.
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
  test_case.expect_alpha_nonzero = true;
  return test_case;
}

bool run_case(const TestCase& test_case) {
  const CollapsedTwoElectronOneLeafStarPairStats stats =
      xmvb::vb::exact_separator::
          evaluate_component_ordered_open_state_star_pair_two_electron_one_leaf(
              test_case.support_overlap_storage,
              test_case.packed_active_two_electron_integrals,
              test_case.support_size,
              test_case.ordered_components,
              DeterminantOverlapResolver());

  std::cout << "case = " << test_case.name << '\n';
  std::cout << "  exact_overlap = " << stats.exact_overlap << '\n';
  std::cout << "  collapsed_overlap = " << stats.collapsed_overlap << '\n';
  std::cout << "  exact_same_spin_alpha = "
            << stats.exact_same_spin_alpha_two_electron << '\n';
  std::cout << "  collapsed_same_spin_alpha = "
            << stats.collapsed_same_spin_alpha_two_electron << '\n';
  std::cout << "  exact_same_spin_beta = "
            << stats.exact_same_spin_beta_two_electron << '\n';
  std::cout << "  collapsed_same_spin_beta = "
            << stats.collapsed_same_spin_beta_two_electron << '\n';
  std::cout << "  exact_opposite_spin = "
            << stats.exact_opposite_spin_two_electron << '\n';
  std::cout << "  collapsed_opposite_spin = "
            << stats.collapsed_opposite_spin_two_electron << '\n';
  std::cout << "  exact_total = " << stats.exact_two_electron << '\n';
  std::cout << "  collapsed_total = " << stats.collapsed_two_electron << '\n';
  std::cout << "  overlap_absolute_error = " << stats.overlap_absolute_error << '\n';
  std::cout << "  same_spin_alpha_absolute_error = "
            << stats.same_spin_alpha_absolute_error << '\n';
  std::cout << "  same_spin_beta_absolute_error = "
            << stats.same_spin_beta_absolute_error << '\n';
  std::cout << "  opposite_spin_absolute_error = "
            << stats.opposite_spin_absolute_error << '\n';
  std::cout << "  total_two_electron_absolute_error = "
            << stats.total_two_electron_absolute_error << '\n';
  std::cout << "  processed_term_quadruple_count = "
            << stats.processed_term_quadruple_count << '\n';
  std::cout << "  alpha_mask_state_count = " << stats.alpha_mask_state_count << '\n';
  std::cout << "  beta_mask_state_count = " << stats.beta_mask_state_count << '\n';
  std::cout << "  subdeterminant_evaluations = "
            << stats.subdeterminant_evaluations << '\n';

  const bool overlap_ok = stats.overlap_absolute_error <= 1.0e-12;
  const bool alpha_ok = stats.same_spin_alpha_absolute_error <= 1.0e-12;
  const bool beta_ok = stats.same_spin_beta_absolute_error <= 1.0e-12;
  const bool opposite_ok = stats.opposite_spin_absolute_error <= 1.0e-12;
  const bool total_ok = stats.total_two_electron_absolute_error <= 1.0e-12;
  const bool alpha_nonzero =
      !test_case.expect_alpha_nonzero ||
      std::abs(stats.exact_same_spin_alpha_two_electron) > 1.0e-10;
  const bool beta_nonzero =
      !test_case.expect_beta_nonzero ||
      std::abs(stats.exact_same_spin_beta_two_electron) > 1.0e-10;
  const bool opposite_nonzero =
      !test_case.expect_opposite_nonzero ||
      std::abs(stats.exact_opposite_spin_two_electron) > 1.0e-10;
  const bool ok =
      overlap_ok && alpha_ok && beta_ok && opposite_ok && total_ok &&
      alpha_nonzero && beta_nonzero && opposite_nonzero;
  if (!ok) {
    std::cerr << "collapsed one-leaf full two-electron separator test failed for case "
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
