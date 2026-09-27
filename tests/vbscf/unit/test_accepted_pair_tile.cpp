#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/determinants/pairs/accepted_action.hpp"
#include "vbscf/determinants/pairs/accepted_tile.hpp"
#include "vbscf/determinants/pairs/contractions.hpp"
#include "vbscf/determinants/pairs/ri_update.hpp"
#include "vbscf/determinants/pairs/traversal.hpp"
#include "vbscf/determinants/pairs/woodbury_overlap.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
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
          0.04 * std::sin((auxiliary + 1) * (pair + 2)) +
          0.03 * std::cos((auxiliary + 2) * (pair + 1));
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
  const auto certified_left_update =
      xmvb::vb::try_woodbury_left_overlap_update(
          strings[0], strings[1], strings[0], overlap, update_anchor);
  require(
      certified_left_update.has_value(),
      "well-conditioned left rank-one Woodbury update was rejected");
  const auto direct_left_update = overlap_resolver.resolve_matrix(
      xmvb::vb::build_overlap_submatrix(
          strings[1], strings[0], overlap_storage, n_active));
  require(
      std::abs(certified_left_update->overlap_determinant -
               direct_left_update.overlap_determinant) <= 2.0e-12 &&
          relative_difference(
              certified_left_update->inverse_overlap_submatrix,
              direct_left_update.inverse_overlap_submatrix) <= 2.0e-12,
      "left Woodbury overlap update differs from direct factorization");
  const auto certified_mixed_update =
      xmvb::vb::try_woodbury_right_overlap_update(
          strings[1], strings[0], strings[1], overlap, *certified_left_update);
  require(
      certified_mixed_update.has_value(),
      "mixed left/right Woodbury path was rejected");
  const auto direct_mixed_update = overlap_resolver.resolve_matrix(
      xmvb::vb::build_overlap_submatrix(
          strings[1], strings[1], overlap_storage, n_active));
  require(
      std::abs(certified_mixed_update->overlap_determinant -
               direct_mixed_update.overlap_determinant) <= 2.0e-12 &&
          relative_difference(
              certified_mixed_update->inverse_overlap_submatrix,
              direct_mixed_update.inverse_overlap_submatrix) <= 2.0e-12,
      "mixed Woodbury overlap path differs from direct factorization");
  xmvb::vb::RiPairUpdateState ri_update;
  require(
      ri_update.initialize(
          strings[0],
          strings[0],
          update_anchor,
          ri.ri_active_pair_factors) &&
          ri_update.update_right(
              strings[0],
              strings[0],
              strings[1],
              update_anchor,
              *certified_update,
              ri.ri_active_pair_factors),
      "regular RI channel update was rejected");
  const auto direct_phi = xmvb::vb::compute_same_spin_original_phi(
      strings[0],
      strings[1],
      h1e,
      n_active,
      ri,
      direct_update,
      nullptr);
  double direct_one_electron_phi = 0.0;
  for (int left = 0; left < static_cast<int>(strings[0].size()); ++left) {
    for (int right = 0; right < static_cast<int>(strings[1].size()); ++right) {
      direct_one_electron_phi +=
          h1e(strings[1][right], strings[0][left]) *
          direct_update.inverse_overlap_submatrix(left, right);
    }
  }
  require(
      std::abs(
          ri_update.two_electron_phi() -
          (direct_phi.total_phi - direct_one_electron_phi)) <= 2.0e-12,
      "low-rank RI channel update changed the two-electron contraction");
  xmvb::vb::RiPairUpdateState ri_left_update;
  require(
      ri_left_update.initialize(
          strings[0],
          strings[0],
          update_anchor,
          ri.ri_active_pair_factors) &&
          ri_left_update.update_left(
              strings[0],
              strings[1],
              strings[0],
              update_anchor,
              *certified_left_update,
              ri.ri_active_pair_factors),
      "regular left RI channel update was rejected");
  const auto direct_left_phi = xmvb::vb::compute_same_spin_original_phi(
      strings[1],
      strings[0],
      h1e,
      n_active,
      ri,
      direct_left_update,
      nullptr);
  double direct_left_one_electron_phi = 0.0;
  for (int left = 0; left < static_cast<int>(strings[1].size()); ++left) {
    for (int right = 0; right < static_cast<int>(strings[0].size()); ++right) {
      direct_left_one_electron_phi +=
          h1e(strings[0][right], strings[1][left]) *
          direct_left_update.inverse_overlap_submatrix(left, right);
    }
  }
  require(
      std::abs(
          ri_left_update.two_electron_phi() -
          (direct_left_phi.total_phi -
           direct_left_one_electron_phi)) <= 2.0e-12,
      "left low-rank RI update changed the two-electron contraction");

  Eigen::VectorXd auxiliary_feature(ri.n_auxiliary_functions);
  require(
      ri_update.first_order_cofactor_auxiliary(
          certified_update->overlap_determinant,
          auxiliary_feature),
      "certified RI channel traces did not produce a cofactor feature");
  Eigen::VectorXd packed_cofactor = Eigen::VectorXd::Zero(n_pairs);
  const Eigen::MatrixXd first_cofactor =
      xmvb::vb::calc_cofactor_1st(*certified_update);
  for (int left = 0; left < static_cast<int>(strings[0].size()); ++left) {
    for (int right = 0; right < static_cast<int>(strings[1].size()); ++right) {
      packed_cofactor(
          xmvb::vb::TwoElectronIndexer::packed_pair_index(
              strings[1][right], strings[0][left])) +=
          first_cofactor(right, left);
    }
  }
  require(
      relative_difference(
          auxiliary_feature,
          ri.ri_active_pair_factors * packed_cofactor) <= 2.0e-11,
      "low-rank RI cofactor feature differs from direct projection");

  const auto projected_tile = provider.build(
      0,
      static_cast<int>(strings.size()),
      0,
      static_cast<int>(strings.size()),
      overlap_storage,
      h1e,
      ri,
      xmvb::vb::AcceptedPairTileBuildOptions{
          .materialize_projected_pair_values = true,
          .populate_response_payload = false,
          .populate_opposite_spin_projection = true});
  for (int left = 0; left < static_cast<int>(strings.size()); ++left) {
    for (int right = 0; right < static_cast<int>(strings.size()); ++right) {
      const auto& projection = projected_tile.pair(left, right)
          .opposite_spin_pair_cache.first_order_cofactor_projection;
      const auto reference_projected =
          xmvb::vb::apply_active_space_two_electron_kernel_to_sparse_projection(
              xmvb::vb::make_active_space_two_electron_view(ri),
              n_active,
              projection.packed_pair_indices,
              projection.packed_pair_values);
      require(
          projection.projected_pair_values.size() ==
                  reference_projected.size() &&
              std::equal(
                  projection.projected_pair_values.begin(),
                  projection.projected_pair_values.end(),
                  reference_projected.begin(),
                  [](double generated, double reference_value) {
                    return std::abs(generated - reference_value) <= 2.0e-11;
                  }),
          "low-rank RI projected cofactor image differs from direct kernel");
    }
  }

  xmvb::vb::DeterminantOverlapResult identity_pair;
  identity_pair.n_electrons = 2;
  identity_pair.overlap_submatrix = Eigen::MatrixXd::Identity(2, 2);
  identity_pair.inverse_overlap_submatrix = Eigen::MatrixXd::Identity(2, 2);
  identity_pair.overlap_determinant = 1.0;
  identity_pair.log_abs_determinant = 0.0;
  identity_pair.determinant_sign = 1.0;
  identity_pair.nullity = 0;
  Eigen::MatrixXd cancellation_factors = Eigen::MatrixXd::Zero(1, 6);
  cancellation_factors(
      0, xmvb::vb::TwoElectronIndexer::packed_pair_index(0, 1)) = 1.0e16;
  cancellation_factors(
      0, xmvb::vb::TwoElectronIndexer::packed_pair_index(1, 1)) = 1.0e16;
  cancellation_factors(
      0, xmvb::vb::TwoElectronIndexer::packed_pair_index(0, 2)) = 1.0;
  cancellation_factors(
      0, xmvb::vb::TwoElectronIndexer::packed_pair_index(1, 2)) = 1.0;
  xmvb::vb::RiPairUpdateState cancellation_update;
  require(
      cancellation_update.initialize(
          {0, 1}, {0, 1}, identity_pair, cancellation_factors),
      "RI cancellation test anchor failed");
  require(
      !cancellation_update.update_right(
          {0, 1},
          {0, 1},
          {0, 2},
          identity_pair,
          identity_pair,
          cancellation_factors),
      "uncertified cancelling RI channel update was accepted");
  require(
      cancellation_update.initialize(
          {0, 1}, {0, 2}, identity_pair, cancellation_factors) &&
          std::abs(cancellation_update.two_electron_phi() + 1.0e16) <= 1.0,
      "direct RI reanchor did not recover the cancelling channel");

  const std::vector<int> four_left{0, 1, 2, 3};
  const std::vector<int> four_right_old{0, 1, 2, 3};
  const std::vector<int> four_right_new{0, 1, 2, 4};
  Eigen::MatrixXd chain_overlap =
      Eigen::MatrixXd::Constant(n_active, n_active, 0.15);
  chain_overlap.diagonal().setOnes();
  const std::vector<double> chain_overlap_storage(
      chain_overlap.data(), chain_overlap.data() + chain_overlap.size());
  const auto four_anchor = overlap_resolver.resolve_matrix(
      xmvb::vb::build_overlap_submatrix(
          four_left, four_right_old, chain_overlap_storage, n_active));
  const auto four_update = xmvb::vb::try_woodbury_right_overlap_update(
      four_left,
      four_right_old,
      four_right_new,
      chain_overlap,
      four_anchor);
  require(
      four_update.has_value(),
      "four-electron Woodbury response update was rejected");
  xmvb::vb::RiPairUpdateState response_update;
  require(
      response_update.initialize(
          four_left,
          four_right_old,
          four_anchor,
          ri.ri_active_pair_factors,
          true) &&
          response_update.update_right(
              four_left,
              four_right_old,
              four_right_new,
              four_anchor,
              *four_update,
              ri.ri_active_pair_factors),
      "four-electron low-rank RI response update was rejected");
  Eigen::MatrixXd direct_response_gradient;
  const auto direct_response = xmvb::vb::compute_same_spin_original_phi(
      four_left,
      four_right_new,
      Eigen::MatrixXd::Zero(n_active, n_active),
      n_active,
      ri,
      *four_update,
      &direct_response_gradient);
  require(
      std::abs(
          response_update.two_electron_phi() -
          direct_response.total_phi) <= 2.0e-11 &&
          relative_difference(
              response_update.two_electron_inverse_overlap_gradient(
                  *four_update),
              direct_response_gradient) <= 2.0e-11,
      "low-rank RI response aggregate differs from direct contraction");
  xmvb::vb::RiPairUpdateState left_response_update;
  require(
      left_response_update.initialize(
          four_left,
          four_right_old,
          four_anchor,
          ri.ri_active_pair_factors,
          true),
      "left RI response anchor was rejected");
  const std::vector<int> four_left_new{0, 1, 2, 4};
  const auto four_left_overlap =
      xmvb::vb::try_woodbury_left_overlap_update(
          four_left,
          four_left_new,
          four_right_old,
          chain_overlap,
          four_anchor);
  require(
      four_left_overlap.has_value() &&
          left_response_update.update_left(
              four_left,
              four_left_new,
              four_right_old,
              four_anchor,
              *four_left_overlap,
              ri.ri_active_pair_factors),
      "left RI response update was rejected");
  Eigen::MatrixXd direct_left_response_gradient;
  const auto direct_left_response =
      xmvb::vb::compute_same_spin_original_phi(
          four_left_new,
          four_right_old,
          Eigen::MatrixXd::Zero(n_active, n_active),
          n_active,
          ri,
          *four_left_overlap,
          &direct_left_response_gradient);
  require(
      std::abs(
          left_response_update.two_electron_phi() -
          direct_left_response.total_phi) <= 2.0e-11 &&
          relative_difference(
              left_response_update.two_electron_inverse_overlap_gradient(
                  *four_left_overlap),
              direct_left_response_gradient) <= 2.0e-11,
      "left low-rank RI response differs from direct contraction");
  const std::vector<int> four_right_next{0, 1, 3, 4};
  const auto four_second_update =
      xmvb::vb::try_woodbury_right_overlap_update(
          four_left,
          four_right_new,
          four_right_next,
          chain_overlap,
          *four_update);
  require(
      four_second_update.has_value(),
      "successive four-electron Woodbury overlap update was rejected");
  xmvb::vb::RiPairUpdateState scalar_chain;
  require(
      scalar_chain.initialize(
          four_left,
          four_right_old,
          four_anchor,
          ri.ri_active_pair_factors) &&
          scalar_chain.update_right(
              four_left,
              four_right_old,
              four_right_new,
              four_anchor,
              *four_update,
              ri.ri_active_pair_factors) &&
          scalar_chain.update_right(
              four_left,
              four_right_new,
              four_right_next,
              *four_update,
              *four_second_update,
              ri.ri_active_pair_factors),
      "successive scalar RI channel updates were rejected");
  const bool response_updated = response_update.update_right(
      four_left,
      four_right_new,
      four_right_next,
      *four_update,
      *four_second_update,
      ri.ri_active_pair_factors);
  require(
      response_updated,
      "successive four-electron RI response update was rejected");
  Eigen::MatrixXd direct_second_gradient;
  const auto direct_second_response =
      xmvb::vb::compute_same_spin_original_phi(
          four_left,
          four_right_next,
          Eigen::MatrixXd::Zero(n_active, n_active),
          n_active,
          ri,
          *four_second_update,
          &direct_second_gradient);
  require(
      std::abs(
          response_update.two_electron_phi() -
          direct_second_response.total_phi) <= 2.0e-11 &&
          relative_difference(
              response_update.two_electron_inverse_overlap_gradient(
                  *four_second_update),
              direct_second_gradient) <= 2.0e-11,
      "successive low-rank RI response update drifted from direct contraction");
  const xmvb::vb::AcceptedPairTileProvider four_provider(
      {four_right_old, four_right_new}, n_active);
  const auto four_tile = four_provider.build(
      0,
      2,
      0,
      2,
      overlap_storage,
      h1e,
      ri,
      xmvb::vb::AcceptedPairTileBuildOptions{
          .materialize_projected_pair_values = false,
          .populate_response_payload = true,
          .populate_opposite_spin_projection = false});
  auto four_reference = evaluator.evaluate_same_spin_pair(
      four_left,
      four_right_new,
      overlap_storage,
      h1e,
      n_active,
      ri,
      true);
  xmvb::vb::complete_same_spin_pair_evaluation(
      four_left,
      four_right_new,
      h1e,
      n_active,
      ri,
      false,
      false,
      true,
      &four_reference);
  const auto& four_generated = four_tile.pair(0, 1);
  require(
      four_generated.has_same_spin_phi_cache &&
          std::abs(
              four_generated.same_spin_total_phi -
              four_reference.same_spin_total_phi) <= 2.0e-11 &&
          relative_difference(
              four_generated.same_spin_inverse_overlap_gradient,
              four_reference.same_spin_inverse_overlap_gradient) <= 2.0e-11 &&
          relative_difference(
              four_generated.same_spin_overlap_hamiltonian_gradient,
              four_reference.same_spin_overlap_hamiltonian_gradient) <= 2.0e-11,
      "accepted tile changed the regular RI response payload");
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
