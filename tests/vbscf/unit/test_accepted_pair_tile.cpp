#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/determinants/pairs/accepted_action.hpp"
#include "vbscf/determinants/pairs/accepted_tile.hpp"
#include "vbscf/determinants/pairs/contractions.hpp"
#include "vbscf/determinants/pairs/traversal.hpp"
#include "vbscf/determinants/pairs/woodbury_overlap.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

double max_difference(
    const Eigen::Ref<const Eigen::MatrixXd>& left,
    const Eigen::Ref<const Eigen::MatrixXd>& right) {
  if (left.rows() != right.rows() || left.cols() != right.cols()) {
    return std::numeric_limits<double>::infinity();
  }
  return left.size() == 0
      ? 0.0
      : (left - right).cwiseAbs().maxCoeff();
}

double relative_difference(
    const Eigen::Ref<const Eigen::MatrixXd>& left,
    const Eigen::Ref<const Eigen::MatrixXd>& right) {
  const double scale = left.size() == 0
      ? 1.0
      : std::max(1.0, left.cwiseAbs().maxCoeff());
  return max_difference(left, right) / scale;
}

std::vector<std::vector<int>> two_electron_strings(int n_active) {
  std::vector<std::vector<int>> strings;
  for (int first = 0; first < n_active - 1; ++first) {
    for (int second = first + 1; second < n_active; ++second) {
      strings.push_back({first, second});
    }
  }
  return strings;
}

void compare_projection(
    const xmvb::vb::OppositeSpinPackedPairProjection& reference,
    const xmvb::vb::OppositeSpinPackedPairProjection& generated) {
  require(
      reference.packed_pair_indices == generated.packed_pair_indices,
      "accepted tile changed packed-pair support");
  require(
      reference.packed_pair_values.size() == generated.packed_pair_values.size(),
      "accepted tile changed packed-pair payload size");
  for (std::size_t entry = 0;
       entry < reference.packed_pair_values.size();
       ++entry) {
    require(
        std::abs(reference.packed_pair_values[entry] -
                 generated.packed_pair_values[entry]) <= 2.0e-11,
        "accepted tile changed packed-pair values");
  }
}

}  // namespace

int main() {
  constexpr int n_active = 5;
  const auto strings = two_electron_strings(n_active);
  Eigen::MatrixXd overlap(n_active, n_active);
  for (int column = 0; column < n_active; ++column) {
    for (int row = 0; row < n_active; ++row) {
      overlap(row, column) =
          std::exp(-0.31 * std::abs(row - column)) +
          0.017 * (row + 1) * (column + 2);
    }
  }
  overlap = 0.5 * (overlap + overlap.transpose()).eval();
  std::vector<double> overlap_storage(
      overlap.data(), overlap.data() + overlap.size());

  Eigen::MatrixXd h1e(n_active, n_active);
  for (int column = 0; column < n_active; ++column) {
    for (int row = 0; row < n_active; ++row) {
      h1e(row, column) = 0.13 * (row + 1) - 0.09 * (column + 1) +
          (row == column ? -0.7 : 0.0);
    }
  }
  h1e = 0.5 * (h1e + h1e.transpose()).eval();

  const int n_pairs = xmvb::vb::packed_active_pair_count(n_active);
  xmvb::vb::ActiveSpaceTwoElectronResult ri;
  ri.representation =
      xmvb::vb::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity;
  ri.n_auxiliary_functions = 7;
  ri.ri_active_pair_factors.resize(ri.n_auxiliary_functions, n_pairs);
  for (int auxiliary = 0; auxiliary < ri.n_auxiliary_functions; ++auxiliary) {
    for (int pair = 0; pair < n_pairs; ++pair) {
      ri.ri_active_pair_factors(auxiliary, pair) =
          0.021 * (auxiliary + 1) - 0.013 * (pair + 2) +
          0.004 * ((auxiliary + pair) % 3);
    }
  }

  xmvb::vb::DeterminantPairEvaluator evaluator;
  auto reference = xmvb::vb::build_same_spin_pair_cache_context(
      strings,
      strings,
      evaluator,
      overlap_storage,
      h1e,
      n_active,
      ri);
  xmvb::vb::populate_same_spin_phi_cache(
      &reference, h1e, n_active, ri);

  const xmvb::vb::AcceptedPairTileProvider provider(strings, n_active);
  const auto traversal = xmvb::vb::build_pair_update_traversal(
      strings, 4, 0, static_cast<int>(strings.size()));
  require(
      traversal.size() == strings.size() && traversal.front() == 4,
      "pair traversal did not retain the diagonal anchor");
  std::vector<int> traversal_counts(strings.size(), 0);
  for (const int index : traversal) {
    require(
        index >= 0 && index < static_cast<int>(strings.size()),
        "pair traversal returned an invalid string index");
    ++traversal_counts[index];
  }
  require(
      std::all_of(
          traversal_counts.begin(),
          traversal_counts.end(),
          [](int count) { return count == 1; }),
      "pair traversal did not visit every string exactly once");
  xmvb::vb::DeterminantOverlapResolver overlap_resolver;
  const auto update_anchor = overlap_resolver.resolve_matrix(
      xmvb::vb::build_overlap_submatrix(
          strings[0], strings[0], overlap_storage, n_active));
  const auto certified_update =
      xmvb::vb::try_woodbury_right_overlap_update(
          strings[0], strings[0], strings[1], overlap, update_anchor);
  require(
      certified_update.has_value(),
      "well-conditioned rank-one Woodbury update was rejected");
  const auto direct_update = overlap_resolver.resolve_matrix(
      xmvb::vb::build_overlap_submatrix(
          strings[0], strings[1], overlap_storage, n_active));
  require(
      std::abs(certified_update->overlap_determinant -
               direct_update.overlap_determinant) <= 2.0e-12 &&
          relative_difference(
              certified_update->inverse_overlap_submatrix,
              direct_update.inverse_overlap_submatrix) <= 2.0e-12,
      "certified Woodbury overlap update differs from direct factorization");
  const auto tile = provider.build(
      0,
      static_cast<int>(strings.size()),
      0,
      static_cast<int>(strings.size()),
      overlap_storage,
      h1e,
      ri,
      xmvb::vb::AcceptedPairTileBuildOptions{false, true});
  const int n_unique = static_cast<int>(strings.size());
  for (int left = 0; left < n_unique; ++left) {
    for (int right = 0; right < n_unique; ++right) {
      const auto& expected = reference.alpha_pair_cache_ref()[
          xmvb::vb::ordered_spin_pair_storage_index(left, right, n_unique)];
      const auto& actual = tile.pair(left, right);
      require(
          expected.overlap_result.nullity == actual.overlap_result.nullity,
          "accepted tile changed pair nullity");
      require(
          std::abs(expected.overlap_result.overlap_determinant -
                   actual.overlap_result.overlap_determinant) <= 2.0e-11,
          "accepted tile changed overlap determinant");
      require(
          relative_difference(
              expected.overlap_result.inverse_overlap_submatrix,
              actual.overlap_result.inverse_overlap_submatrix) <= 5.0e-12,
          "accepted tile changed inverse overlap");
      require(
          std::abs(expected.total_hamiltonian - actual.total_hamiltonian) <=
              2.0e-10,
          "accepted tile changed same-spin Hamiltonian");
      require(
          std::abs(expected.same_spin_total_phi -
                   actual.same_spin_total_phi) <= 2.0e-10,
          "accepted tile changed same-spin phi");
      require(
          relative_difference(
              expected.same_spin_inverse_overlap_gradient,
              actual.same_spin_inverse_overlap_gradient) <= 2.0e-10,
          "accepted tile changed inverse-overlap gradient");
      require(
          relative_difference(
              expected.same_spin_overlap_hamiltonian_gradient,
              actual.same_spin_overlap_hamiltonian_gradient) <= 2.0e-10,
          "accepted tile changed overlap-Hamiltonian gradient");
      compare_projection(
          expected.opposite_spin_pair_cache.first_order_cofactor_projection,
          actual.opposite_spin_pair_cache.first_order_cofactor_projection);
    }
  }

  Eigen::MatrixXd vectors(n_unique, 3);
  for (int column = 0; column < vectors.cols(); ++column) {
    for (int row = 0; row < vectors.rows(); ++row) {
      vectors(row, column) =
          0.03 * (row + 1) - 0.017 * (column + 2);
    }
  }
  Eigen::MatrixXd reference_overlap(n_unique, n_unique);
  Eigen::MatrixXd reference_hamiltonian(n_unique, n_unique);
  for (int left = 0; left < n_unique; ++left) {
    for (int right = 0; right < n_unique; ++right) {
      const auto& pair = reference.alpha_pair_cache_ref()[
          xmvb::vb::ordered_spin_pair_storage_index(
              left, right, n_unique)];
      reference_overlap(left, right) =
          pair.overlap_result.overlap_determinant;
      reference_hamiltonian(left, right) = pair.total_hamiltonian;
    }
  }
  const auto streamed_action = xmvb::vb::apply_accepted_spin_pair_action(
      provider,
      overlap_storage,
      h1e,
      ri,
      vectors,
      4096);
  require(
      relative_difference(
          reference_overlap * vectors,
          streamed_action.overlap) <= 2.0e-11,
      "streamed accepted overlap action differs from cached action");
  require(
      relative_difference(
          reference_hamiltonian * vectors,
          streamed_action.hamiltonian) <= 2.0e-10,
      "streamed accepted Hamiltonian action differs from cached action");

  const std::vector<std::vector<int>> singular_strings{{0}, {1}, {2}};
  const Eigen::Matrix3d singular_overlap = Eigen::Matrix3d::Identity();
  const std::vector<double> singular_overlap_storage(
      singular_overlap.data(),
      singular_overlap.data() + singular_overlap.size());
  const Eigen::Matrix3d singular_h1e = Eigen::Matrix3d::Identity();
  xmvb::vb::ActiveSpaceTwoElectronResult singular_two_electron;
  singular_two_electron.packed_active_two_electron_integrals.assign(21, 0.0);
  const xmvb::vb::AcceptedPairTileProvider singular_provider(
      singular_strings, 3);
  const auto singular_tile = singular_provider.build(
      0,
      1,
      0,
      3,
      singular_overlap_storage,
      singular_h1e,
      singular_two_electron);
  require(
      singular_tile.pair(0, 0).overlap_result.nullity == 0 &&
      singular_tile.pair(0, 1).overlap_result.nullity == 1 &&
      singular_tile.pair(0, 2).overlap_result.nullity == 1,
      "exact tiled evaluation changed singular-pair classification");

  const std::vector<std::vector<int>> mixed_strings{
      {0, 2, 4}, {1, 2, 3}, {0, 3, 4}};
  const Eigen::MatrixXd mixed_overlap =
      Eigen::MatrixXd::Identity(n_active, n_active);
  const std::vector<double> mixed_overlap_storage(
      mixed_overlap.data(), mixed_overlap.data() + mixed_overlap.size());
  const xmvb::vb::AcceptedPairTileProvider mixed_provider(
      mixed_strings, n_active);
  const auto mixed_tile = mixed_provider.build(
      0,
      static_cast<int>(mixed_strings.size()),
      0,
      static_cast<int>(mixed_strings.size()),
      mixed_overlap_storage,
      h1e,
      ri,
      xmvb::vb::AcceptedPairTileBuildOptions{false, true});
  for (int left = 0; left < static_cast<int>(mixed_strings.size()); ++left) {
    for (int right = 0; right < static_cast<int>(mixed_strings.size()); ++right) {
      const auto singleton = mixed_provider.build(
          left,
          left + 1,
          right,
          right + 1,
          mixed_overlap_storage,
          h1e,
          ri,
          xmvb::vb::AcceptedPairTileBuildOptions{false, true});
      const auto& traversed = mixed_tile.pair(left, right);
      const auto& exact = singleton.pair(0, 0);
      require(
          traversed.overlap_result.nullity == exact.overlap_result.nullity,
          "Woodbury traversal changed mixed-pair nullity");
      require(
          std::abs(traversed.overlap_result.overlap_determinant -
                   exact.overlap_result.overlap_determinant) <= 2.0e-12,
          "Woodbury traversal changed mixed-pair determinant");
      require(
          std::abs(traversed.total_hamiltonian - exact.total_hamiltonian) <=
              2.0e-11,
          "Woodbury traversal changed mixed-pair Hamiltonian");
      require(
          std::abs(traversed.same_spin_total_phi -
                   exact.same_spin_total_phi) <= 2.0e-11,
          "Woodbury traversal changed mixed-pair phi");
      require(
          relative_difference(
              traversed.same_spin_inverse_overlap_gradient,
              exact.same_spin_inverse_overlap_gradient) <= 2.0e-11,
          "Woodbury traversal changed mixed-pair inverse gradient");
    }
  }
  Eigen::MatrixXd mixed_vectors(mixed_strings.size(), 2);
  mixed_vectors << 0.2, -0.1, 0.3, 0.4, -0.5, 0.7;
  const auto mixed_full_action = xmvb::vb::apply_accepted_spin_pair_action(
      mixed_provider,
      mixed_overlap_storage,
      h1e,
      ri,
      mixed_vectors,
      1ull << 20);
  const auto mixed_unit_action = xmvb::vb::apply_accepted_spin_pair_action(
      mixed_provider,
      mixed_overlap_storage,
      h1e,
      ri,
      mixed_vectors,
      1);
  require(
      relative_difference(
          mixed_full_action.overlap, mixed_unit_action.overlap) <= 2.0e-12,
      "Woodbury traversal changed mixed overlap action across tile sizes");
  require(
      relative_difference(
          mixed_full_action.hamiltonian,
          mixed_unit_action.hamiltonian) <= 2.0e-11,
      "Woodbury traversal changed mixed Hamiltonian action across tile sizes");
  return 0;
}
