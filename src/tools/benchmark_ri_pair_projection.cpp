#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/determinants/pairs/accepted_tile.hpp"
#include "vbscf/derivatives/hessian/responses/opposite_spin/pair_response_internal.hpp"
#include "vbscf/derivatives/hessian/responses/same_spin/pair_response_internal.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"

namespace {

void append_combinations(
    int n_orbitals,
    int n_electrons,
    int next,
    std::vector<int>* current,
    std::vector<std::vector<int>>* strings) {
  if (static_cast<int>(current->size()) == n_electrons) {
    strings->push_back(*current);
    return;
  }
  const int remaining = n_electrons - static_cast<int>(current->size());
  for (int orbital = next; orbital <= n_orbitals - remaining; ++orbital) {
    current->push_back(orbital);
    append_combinations(
        n_orbitals, n_electrons, orbital + 1, current, strings);
    current->pop_back();
  }
}

std::vector<std::vector<int>> complete_spin_space(
    int n_orbitals,
    int n_electrons) {
  std::vector<std::vector<int>> strings;
  std::vector<int> current;
  append_combinations(
      n_orbitals, n_electrons, 0, &current, &strings);
  return strings;
}

Eigen::MatrixXd gather_raw_channels(
    const xmvb::vb::AcceptedSpinPairTile& tile,
    int n_pairs) {
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(
      tile.left_size * tile.right_size, n_pairs);
  for (int left = 0; left < tile.left_size; ++left) {
    for (int right = 0; right < tile.right_size; ++right) {
      const int work = left + tile.left_size * right;
      const auto& projection = tile.pair(left, right)
          .opposite_spin_pair_cache.first_order_cofactor_projection;
      for (std::size_t entry = 0;
           entry < projection.packed_pair_indices.size();
           ++entry) {
        result(work, projection.packed_pair_indices[entry]) +=
            projection.packed_pair_values[entry];
      }
    }
  }
  return result;
}

Eigen::MatrixXd gather_projected_channels(
    const xmvb::vb::AcceptedSpinPairTile& tile,
    int n_pairs) {
  Eigen::MatrixXd result(tile.left_size * tile.right_size, n_pairs);
  for (int left = 0; left < tile.left_size; ++left) {
    for (int right = 0; right < tile.right_size; ++right) {
      const int work = left + tile.left_size * right;
      const auto& projected = tile.pair(left, right)
          .opposite_spin_pair_cache.first_order_cofactor_projection
          .projected_pair_values;
      if (static_cast<int>(projected.size()) != n_pairs) {
        throw std::runtime_error("projected RI pair image is missing");
      }
      result.row(work) = Eigen::Map<const Eigen::RowVectorXd>(
          projected.data(), n_pairs);
    }
  }
  return result;
}

template <typename Operation>
double minimum_seconds(int repeats, Operation&& operation) {
  double best = std::numeric_limits<double>::infinity();
  for (int repeat = 0; repeat < repeats; ++repeat) {
    const auto begin = std::chrono::steady_clock::now();
    operation();
    best = std::min(
        best,
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count());
  }
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const int n_active = argc > 1 ? std::atoi(argv[1]) : 12;
    const int n_electrons = argc > 2 ? std::atoi(argv[2]) : 6;
    const int n_auxiliary = argc > 3 ? std::atoi(argv[3]) : 360;
    const int requested_extent = argc > 4 ? std::atoi(argv[4]) : 64;
    const int repeats = argc > 5 ? std::atoi(argv[5]) : 3;
    if (n_active <= 0 || n_electrons <= 0 || n_electrons > n_active ||
        n_auxiliary <= 0 || requested_extent <= 0 || repeats <= 0) {
      throw std::invalid_argument("benchmark dimensions must be positive");
    }

    const auto strings = complete_spin_space(n_active, n_electrons);
    const int extent = std::min(
        requested_extent, static_cast<int>(strings.size()));
    Eigen::MatrixXd overlap(n_active, n_active);
    Eigen::MatrixXd h1e(n_active, n_active);
    for (int column = 0; column < n_active; ++column) {
      for (int row = 0; row < n_active; ++row) {
        overlap(row, column) =
            0.18 * std::exp(-0.37 * std::abs(row - column));
        h1e(row, column) =
            0.03 * std::cos((row + 1) * (column + 2));
      }
    }
    overlap.diagonal().array() += 1.0;
    h1e = 0.5 * (h1e + h1e.transpose()).eval();
    const std::vector<double> overlap_values(
        overlap.data(), overlap.data() + overlap.size());

    const int n_pairs = xmvb::vb::packed_active_pair_count(n_active);
    xmvb::vb::ActiveSpaceTwoElectronResult ri;
    ri.representation =
        xmvb::vb::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity;
    ri.n_auxiliary_functions = n_auxiliary;
    ri.ri_active_pair_factors.resize(n_auxiliary, n_pairs);
    for (int pair = 0; pair < n_pairs; ++pair) {
      for (int auxiliary = 0; auxiliary < n_auxiliary; ++auxiliary) {
        ri.ri_active_pair_factors(auxiliary, pair) =
            0.025 * std::sin(0.013 * (auxiliary + 1) * (pair + 3)) +
            0.017 * std::cos(0.021 * (auxiliary + 2) * (pair + 1));
      }
    }

    const xmvb::vb::AcceptedPairTileProvider provider(strings, n_active);
    Eigen::MatrixXd reference;
    Eigen::MatrixXd projected;
    const auto run_reference = [&] {
      const auto tile = provider.build(
          0,
          extent,
          0,
          extent,
          overlap_values,
          h1e,
          ri,
          xmvb::vb::AcceptedPairTileBuildOptions{
              .materialize_projected_pair_values = false,
              .populate_response_payload = false,
              .populate_opposite_spin_projection = true});
      const Eigen::MatrixXd raw = gather_raw_channels(tile, n_pairs);
      reference =
          xmvb::vb::apply_active_space_two_electron_kernel_block(
              ri, n_active, raw.transpose()).transpose();
    };
    const auto run_projected = [&] {
      const auto tile = provider.build(
          0,
          extent,
          0,
          extent,
          overlap_values,
          h1e,
          ri,
          xmvb::vb::AcceptedPairTileBuildOptions{
              .materialize_projected_pair_values = true,
              .populate_response_payload = false,
              .populate_opposite_spin_projection = true});
      projected = gather_projected_channels(tile, n_pairs);
    };

    run_reference();
    run_projected();
    const double scale = std::max(1.0, reference.cwiseAbs().maxCoeff());
    const double relative_difference =
        (reference - projected).cwiseAbs().maxCoeff() / scale;
    if (relative_difference > 2.0e-10) {
      throw std::runtime_error("projected RI pair benchmark changed values");
    }
    const double reference_seconds = minimum_seconds(repeats, run_reference);
    const double projected_seconds = minimum_seconds(repeats, run_projected);

    Eigen::MatrixXd overlap_direction(n_active, n_active);
    Eigen::MatrixXd one_electron_direction(n_active, n_active);
    for (int column = 0; column < n_active; ++column) {
      for (int row = 0; row < n_active; ++row) {
        overlap_direction(row, column) =
            0.004 * std::sin((row + 2) * (column + 1));
        one_electron_direction(row, column) =
            0.006 * std::cos((row + 1) * (column + 4));
      }
    }
    overlap_direction =
        0.5 * (overlap_direction + overlap_direction.transpose()).eval();
    one_electron_direction = 0.5 *
        (one_electron_direction + one_electron_direction.transpose()).eval();
    std::vector<double> overlap_direction_values(
        overlap_direction.data(),
        overlap_direction.data() + overlap_direction.size());
    std::vector<double> one_electron_direction_values(
        one_electron_direction.data(),
        one_electron_direction.data() + one_electron_direction.size());
    const std::vector<double> unused_packed_direction;
    const xmvb::vb::ActiveSpaceIntegralDirectionView direction{
        overlap_direction_values,
        one_electron_direction_values,
        unused_packed_direction};
    Eigen::MatrixXd factor_direction(n_auxiliary, n_pairs);
    for (int pair = 0; pair < n_pairs; ++pair) {
      for (int auxiliary = 0; auxiliary < n_auxiliary; ++auxiliary) {
        factor_direction(auxiliary, pair) =
            0.002 * std::cos(0.017 * (auxiliary + 3) * (pair + 2));
      }
    }
    const auto accepted = provider.build(
        0,
        extent,
        0,
        extent,
        overlap_values,
        h1e,
        ri,
        xmvb::vb::AcceptedPairTileBuildOptions{
            .materialize_projected_pair_values = false,
            .populate_response_payload = true,
            .populate_opposite_spin_projection = true});
    Eigen::MatrixXd direction_reference;
    Eigen::MatrixXd direction_projected;
    const auto run_direction_reference = [&] {
      const auto same_spin =
          xmvb::vb::detail::build_directional_pair_tile(
              strings,
              accepted,
              nullptr,
              n_active,
              direction,
              &h1e,
              &ri.ri_active_pair_factors,
              &factor_direction,
              false);
      direction_reference =
          xmvb::vb::detail::build_directional_opposite_spin_pair_tile(
              strings,
              accepted,
              n_active,
              ri,
              direction,
              same_spin.view(),
              &ri.ri_active_pair_factors,
              &factor_direction)
              .projected_channel_values;
    };
    const auto run_direction_projected = [&] {
      auto same_spin = xmvb::vb::detail::build_directional_pair_tile(
          strings,
          accepted,
          nullptr,
          n_active,
          direction,
          &h1e,
          &ri.ri_active_pair_factors,
          &factor_direction,
          true);
      direction_projected =
          xmvb::vb::detail::build_directional_opposite_spin_pair_tile(
              strings,
              accepted,
              n_active,
              ri,
              direction,
              same_spin.view(),
              &ri.ri_active_pair_factors,
              &factor_direction,
              &same_spin)
              .projected_channel_values;
    };
    run_direction_reference();
    run_direction_projected();
    const double direction_scale = std::max(
        1.0, direction_reference.cwiseAbs().maxCoeff());
    const double direction_relative_difference =
        (direction_reference - direction_projected)
            .cwiseAbs()
            .maxCoeff() /
        direction_scale;
    if (direction_relative_difference > 2.0e-10) {
      throw std::runtime_error(
          "directional projected RI benchmark changed values");
    }
    const double direction_reference_seconds =
        minimum_seconds(repeats, run_direction_reference);
    const double direction_projected_seconds =
        minimum_seconds(repeats, run_direction_projected);

    Eigen::MatrixXd stream_reference;
    Eigen::MatrixXd stream_reused;
    const xmvb::vb::AcceptedPairRiDirectionView ri_direction{
        overlap_direction_values,
        one_electron_direction_values,
        factor_direction,
        true};
    const auto run_stream_reference = [&] {
      const auto stream_accepted = provider.build(
          0,
          extent,
          0,
          extent,
          overlap_values,
          h1e,
          ri,
          xmvb::vb::AcceptedPairTileBuildOptions{
              .materialize_projected_pair_values = false,
              .populate_response_payload = true,
              .populate_opposite_spin_projection = true});
      auto same_spin = xmvb::vb::detail::build_directional_pair_tile(
          strings,
          stream_accepted,
          nullptr,
          n_active,
          direction,
          &h1e,
          &ri.ri_active_pair_factors,
          &factor_direction,
          true);
      stream_reference =
          xmvb::vb::detail::build_directional_opposite_spin_pair_tile(
              strings,
              stream_accepted,
              n_active,
              ri,
              direction,
              same_spin.view(),
              &ri.ri_active_pair_factors,
              &factor_direction,
              &same_spin)
              .projected_channel_values;
    };
    const auto run_stream_reused = [&] {
      xmvb::vb::AcceptedPairRiDirectionTile sidecar;
      const auto stream_accepted = provider.build(
          0,
          extent,
          0,
          extent,
          overlap_values,
          h1e,
          ri,
          xmvb::vb::AcceptedPairTileBuildOptions{
              .materialize_projected_pair_values = false,
              .populate_response_payload = true,
              .populate_opposite_spin_projection = true},
          &ri_direction,
          &sidecar);
      auto same_spin = xmvb::vb::detail::build_directional_pair_tile(
          strings,
          stream_accepted,
          &sidecar,
          n_active,
          direction,
          &h1e,
          &ri.ri_active_pair_factors,
          &factor_direction,
          true);
      stream_reused =
          xmvb::vb::detail::build_directional_opposite_spin_pair_tile(
              strings,
              stream_accepted,
              n_active,
              ri,
              direction,
              same_spin.view(),
              &ri.ri_active_pair_factors,
              &factor_direction,
              &same_spin)
              .projected_channel_values;
    };
    run_stream_reference();
    run_stream_reused();
    const double stream_scale = std::max(
        1.0, stream_reference.cwiseAbs().maxCoeff());
    const double stream_relative_difference =
        (stream_reference - stream_reused).cwiseAbs().maxCoeff() /
        stream_scale;
    if (stream_relative_difference > 2.0e-10) {
      throw std::runtime_error(
          "reused RI directional stream benchmark changed values");
    }
    const double stream_reference_seconds =
        minimum_seconds(repeats, run_stream_reference);
    const double stream_reused_seconds =
        minimum_seconds(repeats, run_stream_reused);

    std::cout << std::setprecision(12)
              << "n_active=" << n_active << '\n'
              << "n_electrons=" << n_electrons << '\n'
              << "n_auxiliary=" << n_auxiliary << '\n'
              << "n_unique=" << strings.size() << '\n'
              << "tile_extent=" << extent << '\n'
              << "pair_count=" << extent * extent << '\n'
              << "reference_seconds=" << reference_seconds << '\n'
              << "projected_seconds=" << projected_seconds << '\n'
              << "speedup=" << reference_seconds / projected_seconds << '\n'
              << "relative_difference=" << relative_difference << '\n'
              << "direction_reference_seconds="
              << direction_reference_seconds << '\n'
              << "direction_projected_seconds="
              << direction_projected_seconds << '\n'
              << "direction_speedup="
              << direction_reference_seconds / direction_projected_seconds
              << '\n'
              << "direction_relative_difference="
              << direction_relative_difference << '\n'
              << "stream_reference_seconds="
              << stream_reference_seconds << '\n'
              << "stream_reused_seconds=" << stream_reused_seconds << '\n'
              << "stream_speedup="
              << stream_reference_seconds / stream_reused_seconds << '\n'
              << "stream_relative_difference="
              << stream_relative_difference << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "benchmark_ri_pair_projection: " << error.what() << '\n';
    return 1;
  }
}
