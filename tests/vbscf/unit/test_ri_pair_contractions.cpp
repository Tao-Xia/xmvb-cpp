#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/determinants/algebra/hamiltonian.hpp"
#include "vbscf/determinants/algebra/overlap.hpp"
#include "vbscf/determinants/pairs/contractions.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/derivatives/hessian/responses/active_space/ri_factor_adjoint.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin/backward.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin/pair_response_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/pair_response_internal.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"

namespace {

void require_close(double left, double right, double tolerance, const char* label) {
  const double scale = std::max({1.0, std::abs(left), std::abs(right)});
  if (std::abs(left - right) > tolerance * scale) {
    throw std::runtime_error(std::string(label) + " mismatch");
  }
}

void require_matrix_close(
    const Eigen::MatrixXd& left,
    const Eigen::MatrixXd& right,
    double tolerance,
    const char* label) {
  if (left.rows() != right.rows() || left.cols() != right.cols()) {
    throw std::runtime_error(std::string(label) + " shape mismatch");
  }
  const double scale = std::max({1.0, left.norm(), right.norm()});
  if ((left - right).norm() > tolerance * scale) {
    throw std::runtime_error(std::string(label) + " value mismatch");
  }
}

}  // namespace

int main() {
  try {
    constexpr int n_active = 5;
    constexpr int n_electrons = 3;
    constexpr int n_auxiliary = 7;
    const std::vector<int> occ_left{0, 2, 4};
    const std::vector<int> occ_right{1, 2, 3};

    Eigen::MatrixXd overlap_block(n_electrons, n_electrons);
    overlap_block <<
        1.10, 0.08, -0.03,
        0.04, 0.93, 0.06,
        -0.02, 0.05, 1.07;
    xmvb::vb::DeterminantOverlapResolver overlap_resolver;
    const auto overlap = overlap_resolver.resolve_matrix(overlap_block);

    Eigen::MatrixXd h1e(n_active, n_active);
    for (int column = 0; column < n_active; ++column) {
      for (int row = 0; row < n_active; ++row) {
        h1e(row, column) =
            0.03 * (row + 1) * (column + 2) - 0.01 * (row - column);
      }
    }

    xmvb::vb::ActiveSpaceTwoElectronResult direct_ri;
    direct_ri.representation =
        xmvb::vb::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity;
    direct_ri.n_auxiliary_functions = n_auxiliary;
    direct_ri.ri_active_pair_factors.resize(
        n_auxiliary,
        xmvb::vb::packed_active_pair_count(n_active));
    for (int pair = 0; pair < direct_ri.ri_active_pair_factors.cols(); ++pair) {
      for (int auxiliary = 0; auxiliary < n_auxiliary; ++auxiliary) {
        direct_ri.ri_active_pair_factors(auxiliary, pair) =
            0.017 * (auxiliary + 1) - 0.006 * (pair + 2) +
            0.001 * ((auxiliary + 2) * (pair + 3) % 5);
      }
    }

    xmvb::vb::ActiveSpaceTwoElectronResult packed_ri = direct_ri;
    packed_ri.packed_active_two_electron_integrals =
        xmvb::vb::reconstruct_packed_active_two_electron_integrals(
            xmvb::vb::make_active_space_two_electron_view(direct_ri),
            n_active);

    xmvb::vb::DeterminantHamiltonianResolver hamiltonian_resolver;
    const auto direct_hamiltonian = hamiltonian_resolver.resolve(
        occ_left,
        occ_right,
        overlap,
        h1e,
        n_active,
        direct_ri);
    const auto packed_hamiltonian = hamiltonian_resolver.resolve(
        occ_left,
        occ_right,
        overlap,
        h1e,
        n_active,
        packed_ri);
    require_close(
        direct_hamiltonian.one_electron_hamiltonian,
        packed_hamiltonian.one_electron_hamiltonian,
        2.0e-13,
        "one-electron Hamiltonian");
    require_close(
        direct_hamiltonian.total_hamiltonian,
        packed_hamiltonian.total_hamiltonian,
        2.0e-13,
        "total Hamiltonian");

    Eigen::MatrixXd direct_inverse_gradient;
    Eigen::MatrixXd packed_inverse_gradient;
    const auto direct_phi = xmvb::vb::compute_same_spin_original_phi(
        occ_left,
        occ_right,
        h1e,
        n_active,
        direct_ri,
        overlap,
        &direct_inverse_gradient);
    const auto packed_phi = xmvb::vb::compute_same_spin_original_phi(
        occ_left,
        occ_right,
        h1e,
        n_active,
        packed_ri,
        overlap,
        &packed_inverse_gradient);
    require_close(
        direct_phi.one_electron_phi,
        packed_phi.one_electron_phi,
        2.0e-13,
        "one-electron phi");
    require_close(
        direct_phi.total_phi,
        packed_phi.total_phi,
        2.0e-13,
        "total phi");
    require_matrix_close(
        direct_inverse_gradient,
        packed_inverse_gradient,
        5.0e-13,
        "inverse-overlap gradient");

    Eigen::MatrixXd delta_overlap(n_electrons, n_electrons);
    delta_overlap <<
        0.013, -0.021, 0.008,
        -0.017, 0.006, 0.014,
        0.009, -0.012, -0.004;
    Eigen::MatrixXd delta_h1e(n_active, n_active);
    for (int column = 0; column < n_active; ++column) {
      for (int row = 0; row < n_active; ++row) {
        delta_h1e(row, column) =
            0.004 * (row - column) + 0.001 * (row + 2) * (column + 1);
      }
    }
    Eigen::MatrixXd delta_factors = direct_ri.ri_active_pair_factors;
    for (Eigen::Index column = 0; column < delta_factors.cols(); ++column) {
      for (Eigen::Index row = 0; row < delta_factors.rows(); ++row) {
        delta_factors(row, column) =
            0.002 * (row + 1) - 0.0013 * (column + 1);
      }
    }

    const auto direction =
        xmvb::vb::evaluate_regular_ri_same_spin_direction(
            occ_left,
            occ_right,
            h1e,
            delta_h1e,
            n_active,
            direct_ri.ri_active_pair_factors,
            delta_factors,
            overlap,
            delta_overlap,
            direct_phi.total_phi,
            direct_inverse_gradient);

    constexpr double epsilon = 1.0e-6;
    auto displaced_data = [&](double scale) {
      const auto displaced_overlap = overlap_resolver.resolve_matrix(
          overlap_block + scale * delta_overlap);
      xmvb::vb::ActiveSpaceTwoElectronResult displaced_ri = direct_ri;
      displaced_ri.ri_active_pair_factors =
          direct_ri.ri_active_pair_factors + scale * delta_factors;
      Eigen::MatrixXd inverse_gradient;
      const auto phi = xmvb::vb::compute_same_spin_original_phi(
          occ_left,
          occ_right,
          h1e + scale * delta_h1e,
          n_active,
          displaced_ri,
          displaced_overlap,
          &inverse_gradient);
      return std::make_tuple(
          displaced_overlap.overlap_determinant * phi.total_phi,
          xmvb::vb::calc_cofactor_1st(displaced_overlap),
          xmvb::vb::build_regular_same_spin_overlap_hamiltonian_gradient(
              displaced_overlap, phi.total_phi, inverse_gradient));
    };
    const auto plus = displaced_data(epsilon);
    const auto minus = displaced_data(-epsilon);
    require_close(
        direction.delta_total_hamiltonian,
        (std::get<0>(plus) - std::get<0>(minus)) / (2.0 * epsilon),
        2.0e-8,
        "RI directional Hamiltonian");
    require_matrix_close(
        direction.delta_first_cofactor,
        (std::get<1>(plus) - std::get<1>(minus)) / (2.0 * epsilon),
        2.0e-8,
        "RI directional first cofactor");
    require_matrix_close(
        direction.delta_overlap_hamiltonian_gradient,
        (std::get<2>(plus) - std::get<2>(minus)) / (2.0 * epsilon),
        3.0e-8,
        "RI directional overlap-Hamiltonian gradient");

    const std::vector<Eigen::MatrixXd> delta_h1e_block{
        delta_h1e,
        -0.37 * delta_h1e};
    const std::vector<Eigen::MatrixXd> delta_factor_block{
        delta_factors,
        0.21 * delta_factors};
    const std::vector<Eigen::MatrixXd> delta_overlap_block{
        delta_overlap,
        -0.16 * delta_overlap};
    const auto block_directions =
        xmvb::vb::evaluate_regular_ri_same_spin_direction_batch(
            occ_left,
            occ_right,
            h1e,
            delta_h1e_block,
            n_active,
            direct_ri.ri_active_pair_factors,
            delta_factor_block,
            overlap,
            delta_overlap_block,
            direct_phi.total_phi,
            direct_inverse_gradient);
    for (std::size_t block = 0; block < block_directions.size(); ++block) {
      const auto scalar_direction =
          xmvb::vb::evaluate_regular_ri_same_spin_direction(
              occ_left,
              occ_right,
              h1e,
              delta_h1e_block[block],
              n_active,
              direct_ri.ri_active_pair_factors,
              delta_factor_block[block],
              overlap,
              delta_overlap_block[block],
              direct_phi.total_phi,
              direct_inverse_gradient);
      require_close(
          block_directions[block].delta_total_hamiltonian,
          scalar_direction.delta_total_hamiltonian,
          1.0e-13,
          "block RI directional Hamiltonian");
      require_matrix_close(
          block_directions[block].delta_first_cofactor,
          scalar_direction.delta_first_cofactor,
          1.0e-13,
          "block RI directional first cofactor");
      require_matrix_close(
          block_directions[block].delta_overlap_hamiltonian_gradient,
          scalar_direction.delta_overlap_hamiltonian_gradient,
          1.0e-13,
          "block RI directional overlap-Hamiltonian gradient");
    }

    xmvb::vb::SameSpinPairCacheContext pair_cache;
    pair_cache.alpha_reuse_table.unique_determinants = {occ_left};
    pair_cache.alpha_pair_cache.resize(1);
    pair_cache.alpha_pair_cache.front().overlap_result = overlap;
    pair_cache.alpha_pair_cache.front().has_same_spin_phi_cache = true;
    pair_cache.alpha_pair_cache.front().same_spin_total_phi =
        direct_phi.total_phi;
    pair_cache.alpha_pair_cache.front().same_spin_inverse_overlap_gradient =
        direct_inverse_gradient;
    pair_cache.beta_reuses_alpha_pair_cache = true;
    pair_cache.close_shell_diagonal_reuses_same_spin_pair_cache = true;
    pair_cache.use_same_spin_pair_cache = true;
    Eigen::MatrixXd global_delta_overlap =
        Eigen::MatrixXd::Zero(n_active, n_active);
    for (int left = 0; left < n_electrons; ++left) {
      for (int right = 0; right < n_electrons; ++right) {
        global_delta_overlap(occ_right[right], occ_left[left]) =
            delta_overlap(right, left);
      }
    }
    std::vector<double> overlap_direction(
        global_delta_overlap.data(),
        global_delta_overlap.data() + global_delta_overlap.size());
    std::vector<double> h1e_direction(
        delta_h1e.data(), delta_h1e.data() + delta_h1e.size());
    const std::vector<double> unused_packed_direction;
    xmvb::vb::ActiveSpaceIntegralDirectionView first_view{
        overlap_direction, h1e_direction, unused_packed_direction};
    Eigen::MatrixXd second_global_overlap = -0.16 * global_delta_overlap;
    Eigen::MatrixXd second_h1e = -0.37 * delta_h1e;
    std::vector<double> second_overlap_direction(
        second_global_overlap.data(),
        second_global_overlap.data() + second_global_overlap.size());
    std::vector<double> second_h1e_direction(
        second_h1e.data(), second_h1e.data() + second_h1e.size());
    xmvb::vb::ActiveSpaceIntegralDirectionView second_view{
        second_overlap_direction,
        second_h1e_direction,
        unused_packed_direction};
    const std::vector<xmvb::vb::ActiveSpaceIntegralDirectionView> views{
        first_view, second_view};
    const auto scalar_cache =
        xmvb::vb::build_same_spin_directional_pair_cache(
            pair_cache,
            n_active,
            first_view,
            &h1e,
            &direct_ri.ri_active_pair_factors,
            &delta_factors);
    const auto scalar_tile =
        xmvb::vb::detail::build_directional_pair_tile(
            pair_cache.alpha_reuse_table.unique_determinants,
            pair_cache.alpha_pair_cache_ref(),
            1,
            n_active,
            first_view,
            0,
            1,
            0,
            1,
            &h1e,
            &direct_ri.ri_active_pair_factors,
            &delta_factors);
    require_matrix_close(
        scalar_tile.delta_regular_hamiltonian,
        scalar_cache.alpha.delta_regular_total_hamiltonian_matrix,
        1.0e-13,
        "RI directional pair tile Hamiltonian");
    require_matrix_close(
        scalar_tile.delta_overlap,
        scalar_cache.alpha.delta_overlap_determinant_matrix,
        1.0e-13,
        "RI directional pair tile overlap");
    require_matrix_close(
        scalar_tile.pair(0, 0).delta_cofactor_1st,
        scalar_cache.alpha.ordered_pair_data.front().delta_cofactor_1st,
        1.0e-13,
        "RI directional pair tile cofactor");
    const auto opposite_cache =
        xmvb::vb::detail::build_directional_opposite_spin_pair_data(
            pair_cache.alpha_reuse_table.unique_determinants,
            pair_cache.alpha_pair_cache_ref(),
            1,
            n_active,
            direct_ri,
            first_view,
            scalar_cache.alpha.ordered_pair_data,
            &direct_ri.ri_active_pair_factors,
            &delta_factors);
    const auto opposite_tile =
        xmvb::vb::detail::build_directional_opposite_spin_pair_tile(
            pair_cache.alpha_reuse_table.unique_determinants,
            pair_cache.alpha_pair_cache_ref(),
            1,
            n_active,
            direct_ri,
            first_view,
            scalar_tile,
            &direct_ri.ri_active_pair_factors,
            &delta_factors);
    require_matrix_close(
        opposite_tile.pair(0, 0).delta_overlap_submatrix,
        opposite_cache.front().delta_overlap_submatrix,
        1.0e-13,
        "RI opposite-spin tile overlap");
    const auto& tile_projection = opposite_tile.pair(0, 0)
                                      .delta_first_order_cofactor_projection;
    const auto& cache_projection = opposite_cache.front()
                                       .delta_first_order_cofactor_projection;
    if (tile_projection.packed_pair_indices !=
        cache_projection.packed_pair_indices) {
      throw std::runtime_error(
          "RI opposite-spin tile projection support mismatch");
    }
    for (std::size_t entry = 0;
         entry < tile_projection.packed_pair_values.size();
         ++entry) {
      require_close(
          tile_projection.packed_pair_values[entry],
          cache_projection.packed_pair_values[entry],
          1.0e-13,
          "RI opposite-spin tile projection");
    }
    for (std::size_t entry = 0;
         entry < tile_projection.projected_pair_values.size();
         ++entry) {
      require_close(
          tile_projection.projected_pair_values[entry],
          cache_projection.projected_pair_values[entry],
          1.0e-13,
          "RI opposite-spin tile kernel image");
    }
    const auto cache_block =
        xmvb::vb::build_same_spin_directional_pair_cache_batch(
            pair_cache,
            n_active,
            views,
            &h1e,
            &direct_ri.ri_active_pair_factors,
            &delta_factor_block);
    require_matrix_close(
        cache_block.front().alpha.delta_regular_total_hamiltonian_matrix,
        scalar_cache.alpha.delta_regular_total_hamiltonian_matrix,
        1.0e-13,
        "block RI pair-cache Hamiltonian");
    require_matrix_close(
        cache_block.front().alpha.delta_overlap_determinant_matrix,
        scalar_cache.alpha.delta_overlap_determinant_matrix,
        1.0e-13,
        "block RI pair-cache overlap");

    const std::vector<std::vector<int>> closed_shell_determinants{
        {0, 1, 2, 3}};
    Eigen::MatrixXd active_overlap_matrix =
        Eigen::MatrixXd::Identity(n_active, n_active);
    std::vector<double> active_overlap(
        active_overlap_matrix.data(),
        active_overlap_matrix.data() + active_overlap_matrix.size());
    xmvb::vb::DeterminantPairEvaluator pair_evaluator;
    auto closed_shell_cache =
        xmvb::vb::build_same_spin_pair_cache_context(
            closed_shell_determinants,
            closed_shell_determinants,
            pair_evaluator,
            active_overlap,
            h1e,
            n_active,
            direct_ri,
            xmvb::vb::SameSpinPairCacheBuildOptions{
                xmvb::vb::PairProjectionCache::Both,
                false});
    xmvb::vb::populate_same_spin_phi_cache(
        &closed_shell_cache,
        h1e,
        n_active,
        direct_ri);
    xmvb::vb::SelectedStateDeterminantMatrices selected;
    selected.n_unique_alpha = 1;
    selected.n_unique_beta = 1;
    selected.n_determinants = 1;
    xmvb::vb::SelectedStateDeterminantCoefficients selected_state;
    selected_state.normalized_state_weight = 1.0;
    selected_state.coefficient_matrix = Eigen::MatrixXd::Ones(1, 1);
    selected.states.push_back(std::move(selected_state));
    const std::vector<double> selected_energies{0.0};
    const auto packed_same =
        xmvb::vb::build_same_spin_matrix_backward_contribution(
            closed_shell_cache,
            selected,
            selected_energies,
            n_active);
    const auto packed_opposite =
        xmvb::vb::build_opposite_spin_backward_contribution(
            closed_shell_cache,
            selected,
            n_active,
            direct_ri);
    std::vector<double> packed_total =
        packed_same.packed_active_two_electron_gradient;
    for (std::size_t index = 0; index < packed_total.size(); ++index) {
      packed_total[index] +=
          packed_opposite.packed_active_two_electron_gradient[index];
    }
    const int n_pairs = xmvb::vb::packed_active_pair_count(n_active);
    Eigen::MatrixXd pair_adjoint(n_pairs, n_pairs);
    for (int row = 0; row < n_pairs; ++row) {
      for (int column = 0; column < n_pairs; ++column) {
        const double value = packed_total[
            xmvb::vb::TwoElectronIndexer::packed_pair_of_pairs_index(
                row, column)];
        pair_adjoint(row, column) = row == column ? 2.0 * value : value;
      }
    }
    const Eigen::MatrixXd packed_factor_adjoint =
        direct_ri.ri_active_pair_factors * pair_adjoint;
    const Eigen::MatrixXd native_factor_adjoint =
        xmvb::vb::apply_regular_ri_pair_space_adjoint(
            closed_shell_cache,
            selected,
            selected_energies,
            n_active,
            direct_ri.ri_active_pair_factors);
    require_matrix_close(
        native_factor_adjoint,
        packed_factor_adjoint,
        2.0e-12,
        "RI-native full pair-space adjoint");

    std::cout << "RI pair contractions agree with the packed reference\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
