#include <algorithm>
#include <array>
#include <barrier>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SparseCore>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"
#include "vbscf/structures/orthogonal_ci/integrals.hpp"
#include "vbscf/structures/orthogonal_ci/metric.hpp"
#include "mcscf/ci/planner.hpp"
#include "mcscf/ci/product_support.hpp"
#include "mcscf/ci/sigma.hpp"
#include "vbscf/determinants/pairs/evaluator.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "core/pair_index.hpp"
#include "integrals/active/two_electron.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/assembly/selected_coefficients.hpp"
#include "vbscf/structures/assembly/hamiltonian_overlap.hpp"

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void check_fused_response(
    const xmvb::mcscf::DirectCiSigmaAction& action,
    const Eigen::Ref<const Eigen::MatrixXd>& response,
    const Eigen::Ref<const Eigen::MatrixXd>& state) {
  const auto fused = action.apply_with_adjoint(response, state);
  const auto sigma = action.apply(response);
  const auto adjoint = action.integral_adjoint(response, state);
  const Eigen::MatrixXd moment=action.one_body_generator_adjoint(response,state);
  const Eigen::MatrixXd density=0.5*(moment+moment.transpose()).eval();
  require((fused.integrals.one_electron-density).norm()<3e-12*std::max(1.0,density.norm()),
          "one-electron integral adjoint differs from the symmetrized generator density");
  require((fused.sigma - sigma).norm() < 3e-12 * std::max(1.0, sigma.norm()),
          "fused response sigma differs from separate action");
  require((fused.integrals.one_electron - adjoint.one_electron).norm() <
              3e-12 * std::max(1.0, adjoint.one_electron.norm()),
          "fused response one-electron adjoint differs from separate contraction");
  require((fused.integrals.pair_kernel - adjoint.pair_kernel).norm() <
              3e-12 * std::max(1.0, adjoint.pair_kernel.norm()),
          "fused response two-electron adjoint differs from separate contraction");
}

/** @brief Joint sweep agrees with independent H/S and differentiated adjoints. */
void check_joint_structure_response(const xmvb::vb::StructureAction& action,
    int n_alpha, int n_beta) {
  const int n = action.n_structures();
  xmvb::vb::SelectedStateDeterminantMatrices states;
  states.n_structures = n;
  states.n_unique_alpha = n_alpha;
  states.n_unique_beta = n_beta;
  states.normalized_state_weights = {0.3, 0.7};
  states.states.resize(2);
  Eigen::MatrixXd raw(n, 2);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < n; ++j) raw(j,i) = std::sin(0.23*(i+1)*(j+2));
    states.states[i].normalized_state_weight = states.normalized_state_weights[i];
    states.states[i].coefficient_matrix = action.expand_structure_block(raw.col(i));
  }
  const auto accepted = action.prepare_active_adjoint(raw, states.normalized_state_weights, {-1.3, -0.9});
  const Eigen::MatrixXd multipliers =
      (Eigen::Matrix2d() << 0.13, -0.04, 0.08, -0.21).finished();
  for (bool zero : {false, true}) {
    if (zero) raw.setZero();
    else raw = raw.array().cos().matrix();
    auto sweep = action.apply_response(accepted, raw);
    const auto images = action.apply(raw);
    require((sweep.images.hamiltonian-images.hamiltonian).norm() <
        3e-11*std::max(1.0,images.hamiltonian.norm()) &&
        (sweep.images.overlap-images.overlap).norm() <
        3e-11*std::max(1.0,images.overlap.norm()), "joint response changed H/S images");
    const auto joint = action.finish_response(accepted, std::move(sweep), multipliers);
    const auto reference = action.active_integral_adjoint_direction(accepted,
        &raw, &multipliers, {}, {}, {}, nullptr, false);
    require((joint.overlap-reference.overlap).norm() <
        3e-11*std::max(1.0,reference.overlap.norm()) &&
        (joint.one_electron-reference.one_electron).norm() <
        3e-11*std::max(1.0,reference.one_electron.norm()) &&
        (joint.pair_kernel-reference.pair_kernel).norm() <
        3e-11*std::max(1.0,reference.pair_kernel.norm()),
        "joint response changed the independent differentiated adjoint");
  }
}

void append_combinations(
    int n_orbitals,
    int n_electrons,
    int first,
    std::vector<int>* occupied,
    std::vector<std::vector<int>>* determinants) {
  if (static_cast<int>(occupied->size()) == n_electrons) {
    determinants->push_back(*occupied);
    return;
  }
  const int remaining = n_electrons - static_cast<int>(occupied->size());
  for (int orbital = first;
       orbital <= n_orbitals - remaining;
       ++orbital) {
    occupied->push_back(orbital);
    append_combinations(
        n_orbitals,
        n_electrons,
        orbital + 1,
        occupied,
        determinants);
    occupied->pop_back();
  }
}

std::vector<std::vector<int>> complete_space(
    int n_orbitals,
    int n_electrons) {
  std::vector<std::vector<int>> determinants;
  std::vector<int> occupied;
  append_combinations(
      n_orbitals,
      n_electrons,
      0,
      &occupied,
      &determinants);
  return determinants;
}

Eigen::MatrixXd explicit_exterior(
    const std::vector<std::vector<int>>& determinants,
    const Eigen::MatrixXd& orbital_transform) {
  const int dimension = static_cast<int>(determinants.size());
  const int n_electrons = static_cast<int>(determinants.front().size());
  Eigen::MatrixXd exterior(dimension, dimension);
  for (int target = 0; target < dimension; ++target) {
    for (int source = 0; source < dimension; ++source) {
      Eigen::MatrixXd minor(n_electrons, n_electrons);
      for (int row = 0; row < n_electrons; ++row) {
        for (int column = 0; column < n_electrons; ++column) {
          minor(row, column) = orbital_transform(
              determinants[target][row],
              determinants[source][column]);
        }
      }
      exterior(target, source) = minor.determinant();
    }
  }
  return exterior;
}

void check_generated_exterior_rows() {
  {
    Eigen::MatrixXd upper=Eigen::MatrixXd::Identity(3,3);
    upper(0,1)=-1e-250;
    upper(0,2)=1e-180;
    const xmvb::vb::ExteriorOrbitalTransform transform(complete_space(3,1),upper);
    const std::vector<int> selected{0,1,2};
    const auto factors=transform.generate_rows(selected);
    require(factors.nonZeros()==5 && factors.coeff(0,1)==-1e-250 && factors.coeff(0,2)==1e-180,
        "exact exterior CSC packing dropped a nonzero small factor");
  }
  for (int n:{3,6,8}) for (int electrons=0;electrons<=n;++electrons) {
    auto strings=complete_space(n,electrons);
    std::reverse(strings.begin(),strings.end());
    Eigen::MatrixXd upper=Eigen::MatrixXd::Zero(n,n);
    for (int i=0;i<n;++i) {
      upper(i,i)=0.7+0.13*(i+1);
      for (int j=i+1;j<n;++j) upper(i,j)=0.2*std::cos(0.31*(i+1)*(j+2));
    }
    const xmvb::vb::ExteriorOrbitalTransform transform(strings,upper);
    const Eigen::MatrixXd reference=explicit_exterior(strings,upper);
    std::vector<int> selected{static_cast<int>(strings.size())-1,0,
        static_cast<int>(strings.size()/2)};
    const Eigen::MatrixXd rows=transform.generate_rows(selected);
    for (int i=0;i<static_cast<int>(selected.size());++i)
      require((rows.row(i)-reference.row(selected[i])).cwiseAbs().maxCoeff()<2e-13,
          "direct exterior rows differ from independent minors/permuted occupations");
    const auto retained=transform.dynamic_bytes();
    const Eigen::MatrixXd repeated=transform.generate_rows(selected);
    const auto single=transform.row_block(selected.front(),1);
    require(transform.dynamic_bytes()==retained && (repeated-rows).norm()==0.0 &&
        (single->row(0)-rows.row(0)).norm()==0.0,
        "input and projection rows do not share the same bounded orbital factors");
    Eigen::MatrixXd team;
#pragma omp parallel num_threads(2)
#pragma omp single
    team=transform.generate_rows(selected);
    require((team-rows).norm()<1e-14,"direct exterior rows changed inside the sigma team");
    const auto cached=transform.row_block(0,strings.size());
    require((*cached-reference).cwiseAbs().maxCoeff()<2e-13,
        "cached exterior row recursion differs from independent minors");
    const auto empty=transform.generate_rows({});
    require(empty.rows()==0 && empty.cols()==reference.cols(),"empty exterior row selection changed shape");
  }
}

void check_exterior_transform(int width) {
  const auto determinants = complete_space(6, 3);
  Eigen::MatrixXd orbital = Eigen::MatrixXd::Zero(6, 6);
  for (int row = 0; row < 6; ++row) {
    orbital(row, row) = 0.8 + 0.07 * static_cast<double>(row + 1);
    for (int column = row + 1; column < 6; ++column) {
      orbital(row, column) =
          0.03 * std::sin(static_cast<double>((row + 2) * (column + 1)));
    }
  }
  const Eigen::MatrixXd exterior = explicit_exterior(determinants, orbital);
  const xmvb::vb::ExteriorOrbitalTransform transform(determinants, orbital);

  Eigen::MatrixXd coefficients(exterior.rows(), width);
  for (int column = 0; column < coefficients.cols(); ++column) {
    for (int row = 0; row < coefficients.rows(); ++row) {
      coefficients(row, column) =
          std::sin(0.17 * static_cast<double>((row + 1) * (column + 2)));
    }
  }
  Eigen::MatrixXd left = coefficients;
  transform.apply_left(&left);
  require(
      (left - exterior * coefficients).cwiseAbs().maxCoeff() < 2.0e-13,
      "left exterior action differs from explicit compound matrix");

  Eigen::MatrixXd right(width - 2, exterior.rows());
  for (int column = 0; column < right.cols(); ++column) {
    for (int row = 0; row < right.rows(); ++row) {
      right(row, column) =
          std::cos(0.11 * static_cast<double>((row + 3) * (column + 1)));
    }
  }
  const Eigen::MatrixXd original_right = right;
  transform.apply_right(&right);
  require(
      (right - original_right * exterior.transpose())
              .cwiseAbs()
              .maxCoeff() < 2.0e-13,
      "right exterior action differs from explicit compound matrix");

  Eigen::MatrixXd block_storage = Eigen::MatrixXd::Zero(
      original_right.rows(), original_right.cols() + 4);
  block_storage.middleCols(2, original_right.cols()) = original_right;
  auto right_block =
      block_storage.middleCols(2, original_right.cols());
  transform.apply_right_block(right_block);
  require(
      (right_block - right).cwiseAbs().maxCoeff() < 2.0e-13 &&
          block_storage.leftCols(2).isZero(0.0) &&
          block_storage.rightCols(2).isZero(0.0),
      "right exterior block action differs from the owning-matrix path");

  transform.apply_adjoint_right(&right);
  require(
      (right - original_right * exterior.transpose() * exterior)
              .cwiseAbs()
              .maxCoeff() < 4.0e-13,
      "right exterior adjoint is inconsistent");
  transform.apply_adjoint_right_block(right_block);
  require(
      (right_block - right).cwiseAbs().maxCoeff() < 4.0e-13 &&
          block_storage.leftCols(2).isZero(0.0) &&
          block_storage.rightCols(2).isZero(0.0),
      "right exterior block adjoint differs from the owning-matrix path");

  Eigen::MatrixXd adjoint = coefficients;
  transform.apply_adjoint_left(&adjoint);
  require(
      (adjoint - exterior.transpose() * coefficients)
              .cwiseAbs()
              .maxCoeff() < 2.0e-13,
      "left exterior adjoint differs from explicit compound matrix");
  require(transform.shear_pair_count() > 0, "test transform has no shears");

  transform.apply_inverse_left(&left);
  require((left - coefficients).cwiseAbs().maxCoeff() < 3.0e-13,
          "inverse exterior left action fails a forward round trip");
  transform.apply_inverse_adjoint_left(&adjoint);
  require((adjoint - coefficients).cwiseAbs().maxCoeff() < 3.0e-13,
          "inverse exterior adjoint fails a forward round trip");
  Eigen::MatrixXd inverse = coefficients;
  transform.apply_inverse_left(&inverse);
  require((exterior * inverse - coefficients).cwiseAbs().maxCoeff() < 3.0e-13,
          "inverse exterior differs from an independent compound-matrix solve");
  Eigen::MatrixXd inverse_adjoint = coefficients;
  transform.apply_inverse_adjoint_left(&inverse_adjoint);
  require((exterior.transpose() * inverse_adjoint - coefficients).cwiseAbs().maxCoeff() < 3.0e-13,
          "inverse exterior transpose differs from an independent compound solve");
  Eigen::MatrixXd inverse_right = original_right;
  transform.apply_right(&inverse_right);
  transform.apply_inverse_right_block(inverse_right);
  require((inverse_right - original_right).cwiseAbs().maxCoeff() < 3.0e-13,
          "inverse exterior right action fails a forward round trip");
  transform.apply_adjoint_right(&inverse_right);
  transform.apply_inverse_adjoint_right_block(inverse_right);
  require((inverse_right - original_right).cwiseAbs().maxCoeff() < 3.0e-13,
          "inverse exterior right adjoint fails a forward round trip");
  block_storage.middleCols(2, original_right.cols()) = original_right;
  transform.apply_inverse_right_block(right_block);
  const Eigen::MatrixXd right_inverse_reference =
      exterior.partialPivLu().solve(original_right.transpose()).transpose();
  require((right_block - right_inverse_reference).cwiseAbs().maxCoeff() < 3.0e-13 &&
              block_storage.leftCols(2).isZero(0.0) &&
              block_storage.rightCols(2).isZero(0.0),
          "inverse exterior block differs from a compound solve or modifies neighbors");
  block_storage.middleCols(2, original_right.cols()) = original_right;
  transform.apply_inverse_adjoint_right_block(right_block);
  const Eigen::MatrixXd right_inverse_adjoint_reference =
      exterior.transpose().partialPivLu().solve(original_right.transpose()).transpose();
  require((right_block - right_inverse_adjoint_reference).cwiseAbs().maxCoeff() < 3.0e-13 &&
              block_storage.leftCols(2).isZero(0.0) &&
              block_storage.rightCols(2).isZero(0.0),
          "inverse exterior block adjoint differs from a compound solve or modifies neighbors");

  // Non-owning row blocks have an outer stride larger than their row count.
  // Check both their result and untouched neighboring storage with parallel
  // widths that do not divide evenly among workers.
  Eigen::MatrixXd strided = Eigen::MatrixXd::Zero(right.rows() + 4, right.cols());
  strided.middleRows(2, right.rows()) = original_right;
  auto rows = strided.middleRows(2, right.rows());
  transform.apply_right_block(rows);
  require((rows - original_right * exterior.transpose()).cwiseAbs().maxCoeff() < 2.0e-13 &&
              strided.topRows(2).isZero(0.0) && strided.bottomRows(2).isZero(0.0),
          "parallel exterior transform modifies neighbors of a strided block");
  transform.apply_inverse_right_block(rows);
  require((rows - original_right).cwiseAbs().maxCoeff() < 3.0e-13,
          "strided exterior inverse does not recover its input");
}

void check_planner() {
  const auto cerras_alpha = complete_space(11, 5);
  const auto cerras_beta = complete_space(11, 6);
  const auto cerras = xmvb::mcscf::plan_orthogonal_direct_ci_action(
      cerras_alpha,
      cerras_beta,
      11,
      1);
  require(
      cerras.carrier_complete(),
      "CERRAS-size fixed-spin carrier was not recognized");
  require(cerras.n_alpha_strings == 462 && cerras.n_beta_strings == 462,
          "CERRAS-size determinant count is wrong");
  require(cerras.alpha_same_spin_connections == 181 &&
              cerras.beta_same_spin_connections == 181 &&
              cerras.opposite_spin_connections == 900,
          "CERRAS-size connection count is wrong");
  require(cerras.favors_direct_ci(), "CERRAS-size FLOP plan rejected direct CI");

  const auto loflea_space = complete_space(12, 6);
  const auto loflea = xmvb::mcscf::plan_orthogonal_direct_ci_action(
      loflea_space,
      loflea_space,
      12,
      1);
  require(
      loflea.carrier_complete(),
      "LOFLEA-size fixed-spin carrier was not recognized");
  require(loflea.n_alpha_strings == 924 && loflea.n_beta_strings == 924,
          "LOFLEA-size determinant count is wrong");
  require(loflea.alpha_same_spin_connections == 262 &&
              loflea.opposite_spin_connections == 1296,
          "LOFLEA-size connection count is wrong");
  require(loflea.favors_direct_ci(), "LOFLEA-size FLOP plan rejected direct CI");

  auto incomplete = cerras_alpha;
  incomplete.pop_back();
  const auto rejected = xmvb::mcscf::plan_orthogonal_direct_ci_action(
      incomplete,
      cerras_beta,
      11,
      1);
  require(!rejected.carrier_complete() && !rejected.favors_direct_ci(),
          "incomplete fixed-spin space was admitted");

  auto duplicate = cerras_alpha;
  duplicate.back() = duplicate.front();
  const auto duplicate_rejected = xmvb::mcscf::plan_orthogonal_direct_ci_action(
      duplicate,
      cerras_beta,
      11,
      1);
  require(
      !duplicate_rejected.carrier_complete(),
      "count-complete carrier with a duplicate string was admitted");

  const auto large_half_filled = complete_space(16, 8);
  const auto large_plan = xmvb::mcscf::plan_orthogonal_direct_ci_action(
      large_half_filled,
      large_half_filled,
      16,
      1);
  require(
      large_plan.carrier_complete(),
      "large fixed-spin carrier was not recognized");
}

void check_sigma_action() {
  constexpr int n_orbitals = 4;
  const auto alpha = complete_space(n_orbitals, 2);
  const auto beta = complete_space(n_orbitals, 2);
  Eigen::MatrixXd orbital = Eigen::MatrixXd::Zero(n_orbitals, n_orbitals);
  orbital <<
      1.10, 0.08, -0.03, 0.05,
      0.00, 0.91,  0.06, 0.02,
      0.00, 0.00,  1.07, 0.04,
      0.00, 0.00,  0.00, 0.96;
  const Eigen::MatrixXd overlap = orbital.transpose() * orbital;
  std::vector<double> packed_overlap(
      overlap.data(), overlap.data() + overlap.size());

  Eigen::MatrixXd one_electron(n_orbitals, n_orbitals);
  for (int column = 0; column < n_orbitals; ++column) {
    for (int row = 0; row <= column; ++row) {
      const double value =
          0.13 * std::cos(static_cast<double>((row + 2) * (column + 1)));
      one_electron(row, column) = value;
      one_electron(column, row) = value;
    }
  }
  const int n_pairs = xmvb::integrals::packed_active_pair_count(n_orbitals);
  Eigen::MatrixXd pair_factor(7, n_pairs);
  for (int column = 0; column < n_pairs; ++column) {
    for (int row = 0; row < pair_factor.rows(); ++row) {
      pair_factor(row, column) =
          0.07 * std::sin(static_cast<double>((row + 1) * (column + 3)));
    }
  }
  const Eigen::MatrixXd pair_kernel = pair_factor.transpose() * pair_factor;
  xmvb::integrals::ActiveSpaceTwoElectronResult two_electron;
  two_electron.packed_active_two_electron_integrals.resize(
      xmvb::integrals::packed_active_two_electron_integral_count(n_orbitals));
  for (int column = 0; column < n_pairs; ++column) {
    for (int row = 0; row <= column; ++row) {
      two_electron.packed_active_two_electron_integrals[
          xmvb::core::PairIndex::packed_pair_of_pairs_index(
              row, column)] = pair_kernel(row, column);
    }
  }

  const auto orthogonal = xmvb::vb::orthogonalize_active_integrals(
      packed_overlap,
      one_electron,
      two_electron,
      n_orbitals);

  Eigen::MatrixXd overlap_direction(n_orbitals, n_orbitals);
  Eigen::MatrixXd one_electron_direction(n_orbitals, n_orbitals);
  for (int column = 0; column < n_orbitals; ++column) {
    for (int row = 0; row <= column; ++row) {
      const double overlap_value =
          0.02 * std::sin(static_cast<double>((row + 1) * (column + 3)));
      const double one_electron_value =
          0.03 * std::cos(static_cast<double>((row + 2) * (column + 4)));
      overlap_direction(row, column) = overlap_value;
      overlap_direction(column, row) = overlap_value;
      one_electron_direction(row, column) = one_electron_value;
      one_electron_direction(column, row) = one_electron_value;
    }
  }
  std::vector<double> packed_overlap_direction(
      overlap_direction.data(),
      overlap_direction.data() + overlap_direction.size());
  std::vector<double> packed_two_electron_direction(
      two_electron.packed_active_two_electron_integrals.size());
  for (std::size_t index = 0;
       index < packed_two_electron_direction.size();
       ++index) {
    packed_two_electron_direction[index] =
        0.01 * std::sin(0.13 * static_cast<double>(index + 1));
  }
  const auto orthogonal_direction =
      xmvb::vb::orthogonalize_active_integral_direction(
          orthogonal,
          packed_overlap_direction,
          one_electron_direction,
          packed_two_electron_direction,
          n_orbitals);
  constexpr double direction_step = 1.0e-6;
  std::vector<double> plus_overlap = packed_overlap;
  std::vector<double> minus_overlap = packed_overlap;
  auto plus_two_electron = two_electron;
  auto minus_two_electron = two_electron;
  for (std::size_t index = 0; index < plus_overlap.size(); ++index) {
    plus_overlap[index] += direction_step * packed_overlap_direction[index];
    minus_overlap[index] -= direction_step * packed_overlap_direction[index];
  }
  for (std::size_t index = 0;
       index < packed_two_electron_direction.size();
       ++index) {
    plus_two_electron.packed_active_two_electron_integrals[index] +=
        direction_step * packed_two_electron_direction[index];
    minus_two_electron.packed_active_two_electron_integrals[index] -=
        direction_step * packed_two_electron_direction[index];
  }
  const auto plus_orthogonal = xmvb::vb::orthogonalize_active_integrals(
      plus_overlap,
      one_electron + direction_step * one_electron_direction,
      plus_two_electron,
      n_orbitals);
  const auto minus_orthogonal = xmvb::vb::orthogonalize_active_integrals(
      minus_overlap,
      one_electron - direction_step * one_electron_direction,
      minus_two_electron,
      n_orbitals);
  require(
      (((plus_orthogonal.orbital_transform -
         minus_orthogonal.orbital_transform) /
        (2.0 * direction_step)) -
       orthogonal_direction.orbital_transform).cwiseAbs().maxCoeff() < 2.0e-9,
      "orthogonal orbital-transform direction fails finite differences");
  require(
      (((plus_orthogonal.one_electron - minus_orthogonal.one_electron) /
        (2.0 * direction_step)) -
       orthogonal_direction.one_electron).cwiseAbs().maxCoeff() < 2.0e-9,
      "orthogonal one-electron direction fails finite differences");
  require(
      (((plus_orthogonal.pair_kernel - minus_orthogonal.pair_kernel) /
        (2.0 * direction_step)) -
       orthogonal_direction.pair_kernel).cwiseAbs().maxCoeff() < 2.0e-9,
      "orthogonal two-electron direction fails finite differences");

  const xmvb::vb::ExteriorOrbitalTransform alpha_transform(
      alpha, orthogonal.orbital_transform);
  const xmvb::vb::ExteriorOrbitalTransform beta_transform(
      beta, orthogonal.orbital_transform);
  const xmvb::mcscf::DirectCiSigmaAction sigma_action(
      alpha, beta, orthogonal);
  const xmvb::mcscf::DirectCiSigmaAction sigma_direction(
      alpha, beta, orthogonal_direction);

  const Eigen::MatrixXd diagonal=sigma_action.diagonal(orthogonal.one_electron);
  for (Eigen::Index a=0;a<diagonal.rows();++a) for (Eigen::Index b=0;b<diagonal.cols();++b) {
    Eigen::MatrixXd probe=Eigen::MatrixXd::Zero(diagonal.rows(),diagonal.cols());
    probe(a,b)=1.0;
    require(std::abs(sigma_action.apply(probe)(a,b)-diagonal(a,b))<1e-12,
        "occupation-only carrier diagonal differs from exact sigma diagonal");
  }

  constexpr int block_width = 2;
  Eigen::MatrixXd coefficients(
      static_cast<int>(alpha.size()),
      block_width * static_cast<int>(beta.size()));
  for (int column = 0; column < coefficients.cols(); ++column) {
    for (int row = 0; row < coefficients.rows(); ++row) {
      coefficients(row, column) =
          std::sin(0.19 * static_cast<double>((row + 1) * (column + 2)));
    }
  }

  const xmvb::mcscf::DirectCiSigmaAction shared_direction(
      sigma_action, orthogonal_direction);
  require((shared_direction.apply(coefficients) - sigma_direction.apply(coefficients))
              .cwiseAbs().maxCoeff() < 2.0e-12,
          "shared spin-string links changed the directional sigma action");
  const auto surviving_direction = [&] {
    const xmvb::mcscf::DirectCiSigmaAction temporary(alpha, beta, orthogonal);
    return xmvb::mcscf::DirectCiSigmaAction(temporary, orthogonal_direction);
  }();
  require((surviving_direction.apply(coefficients) - sigma_direction.apply(coefficients))
              .cwiseAbs().maxCoeff() < 2.0e-12,
          "shared spin-string links did not survive their original action");
  const xmvb::mcscf::DirectCiSigmaAction equal_spin_action(alpha, alpha, orthogonal);
  const xmvb::mcscf::DirectCiSigmaAction equal_spin_direction(
      equal_spin_action, orthogonal_direction);
  const xmvb::mcscf::DirectCiSigmaAction equal_spin_reference(
      alpha, alpha, orthogonal_direction);
  const Eigen::MatrixXd equal_spin_values = Eigen::MatrixXd::Ones(
      alpha.size(), 2 * alpha.size());
  require((equal_spin_direction.apply(equal_spin_values) -
           equal_spin_reference.apply(equal_spin_values)).cwiseAbs().maxCoeff() < 2.0e-12,
          "shared alpha/beta links changed the equal-spin sigma action");

  Eigen::MatrixXd left_coefficients(coefficients.rows(), coefficients.cols());
  for (int column = 0; column < left_coefficients.cols(); ++column) {
    for (int row = 0; row < left_coefficients.rows(); ++row) {
      left_coefficients(row, column) =
          std::cos(0.23 * static_cast<double>((row + 2) * (column + 1)));
    }
  }
  const xmvb::mcscf::DirectCiIntegralAdjoint integral_adjoint =
      sigma_action.integral_adjoint(left_coefficients, coefficients);
  check_fused_response(sigma_action, left_coefficients, coefficients);
  check_fused_response(sigma_action, coefficients, coefficients);
  check_fused_response(sigma_action, Eigen::MatrixXd::Zero(
      coefficients.rows(), coefficients.cols()), coefficients);
  Eigen::MatrixXd padded_left = Eigen::MatrixXd::Zero(
      coefficients.rows() + 3, coefficients.cols());
  Eigen::MatrixXd padded_right = padded_left;
  padded_left.middleRows(1, coefficients.rows()) = left_coefficients;
  padded_right.middleRows(2, coefficients.rows()) = coefficients;
  check_fused_response(sigma_action, padded_left.middleRows(1, coefficients.rows()),
                       padded_right.middleRows(2, coefficients.rows()));
  const xmvb::mcscf::DirectCiIntegralAdjoint reverse_integral_adjoint =
      sigma_action.integral_adjoint(coefficients, left_coefficients);
  require(
      (integral_adjoint.one_electron -
       reverse_integral_adjoint.one_electron).cwiseAbs().maxCoeff() < 2.0e-12,
      "direct-CI one-electron adjoint violates bilinear reciprocity");
  require(
      (integral_adjoint.pair_kernel -
       reverse_integral_adjoint.pair_kernel).cwiseAbs().maxCoeff() < 2.0e-12,
      "direct-CI pair-kernel adjoint violates bilinear reciprocity");
  const double predicted_integral_direction =
      (integral_adjoint.one_electron.cwiseProduct(
           orthogonal_direction.one_electron)).sum() +
      (integral_adjoint.pair_kernel.cwiseProduct(
           orthogonal_direction.pair_kernel)).sum();
  const double applied_integral_direction =
      (left_coefficients.cwiseProduct(
           sigma_direction.apply(coefficients))).sum();
  require(
      std::abs(predicted_integral_direction - applied_integral_direction) <
          3.0e-10 * std::max(1.0, std::abs(applied_integral_direction)),
      "direct-CI integral adjoint fails the bilinear directional identity");
  constexpr double adjoint_step = 1.0e-6;
  auto plus_sigma_integrals = orthogonal;
  auto minus_sigma_integrals = orthogonal;
  plus_sigma_integrals.one_electron +=
      adjoint_step * orthogonal_direction.one_electron;
  minus_sigma_integrals.one_electron -=
      adjoint_step * orthogonal_direction.one_electron;
  plus_sigma_integrals.pair_kernel +=
      adjoint_step * orthogonal_direction.pair_kernel;
  minus_sigma_integrals.pair_kernel -=
      adjoint_step * orthogonal_direction.pair_kernel;
  const double plus_bilinear =
      (left_coefficients.cwiseProduct(
           xmvb::mcscf::DirectCiSigmaAction(
               alpha, beta, plus_sigma_integrals).apply(coefficients))).sum();
  const double minus_bilinear =
      (left_coefficients.cwiseProduct(
           xmvb::mcscf::DirectCiSigmaAction(
               alpha, beta, minus_sigma_integrals).apply(coefficients))).sum();
  const double finite_difference_bilinear =
      (plus_bilinear - minus_bilinear) / (2.0 * adjoint_step);
  require(
      std::abs(predicted_integral_direction - finite_difference_bilinear) <
          2.0e-8 * std::max(1.0, std::abs(finite_difference_bilinear)),
      "direct-CI integral adjoint fails central finite differences");

  const auto alpha_transform_direction = alpha_transform.direction(
      orthogonal_direction.orbital_transform);
  Eigen::MatrixXd transformed_coefficients = coefficients;
  Eigen::MatrixXd transformed_direction = Eigen::MatrixXd::Zero(
      coefficients.rows(), coefficients.cols());
  alpha_transform.apply_directional_left(
      alpha_transform_direction,
      &transformed_coefficients,
      &transformed_direction);
  Eigen::MatrixXd plus_transformed = coefficients;
  Eigen::MatrixXd minus_transformed = coefficients;
  xmvb::vb::ExteriorOrbitalTransform(
      alpha, plus_orthogonal.orbital_transform).apply_left(&plus_transformed);
  xmvb::vb::ExteriorOrbitalTransform(
      alpha, minus_orthogonal.orbital_transform).apply_left(&minus_transformed);
  require(
      (((plus_transformed - minus_transformed) /
        (2.0 * direction_step)) -
       transformed_direction).cwiseAbs().maxCoeff() < 2.0e-9,
      "exterior-transform direction fails finite differences");

  Eigen::MatrixXd right_probe(5, static_cast<int>(alpha.size()));
  for (int column = 0; column < right_probe.cols(); ++column) {
    for (int row = 0; row < right_probe.rows(); ++row) {
      right_probe(row, column) =
          std::cos(0.09 * static_cast<double>((row + 2) * (column + 1)));
    }
  }
  Eigen::MatrixXd directional_right = right_probe;
  Eigen::MatrixXd delta_directional_right = Eigen::MatrixXd::Zero(
      right_probe.rows(), right_probe.cols());
  alpha_transform.apply_directional_right(
      alpha_transform_direction,
      &directional_right,
      &delta_directional_right);
  Eigen::MatrixXd plus_right = right_probe;
  Eigen::MatrixXd minus_right = right_probe;
  xmvb::vb::ExteriorOrbitalTransform(
      alpha, plus_orthogonal.orbital_transform).apply_right(&plus_right);
  xmvb::vb::ExteriorOrbitalTransform(
      alpha, minus_orthogonal.orbital_transform).apply_right(&minus_right);
  require(
      (((plus_right - minus_right) / (2.0 * direction_step)) -
       delta_directional_right).cwiseAbs().maxCoeff() < 2.0e-9,
      "right exterior-transform direction fails finite differences");

  Eigen::MatrixXd directional_adjoint_left = coefficients;
  Eigen::MatrixXd delta_directional_adjoint_left = Eigen::MatrixXd::Zero(
      coefficients.rows(), coefficients.cols());
  alpha_transform.apply_directional_adjoint_left(
      alpha_transform_direction,
      &directional_adjoint_left,
      &delta_directional_adjoint_left);
  Eigen::MatrixXd plus_adjoint_left = coefficients;
  Eigen::MatrixXd minus_adjoint_left = coefficients;
  xmvb::vb::ExteriorOrbitalTransform(
      alpha,
      plus_orthogonal.orbital_transform).apply_adjoint_left(
          &plus_adjoint_left);
  xmvb::vb::ExteriorOrbitalTransform(
      alpha,
      minus_orthogonal.orbital_transform).apply_adjoint_left(
          &minus_adjoint_left);
  require(
      (((plus_adjoint_left - minus_adjoint_left) /
        (2.0 * direction_step)) -
       delta_directional_adjoint_left).cwiseAbs().maxCoeff() < 2.0e-9,
      "adjoint-left exterior direction fails finite differences");

  Eigen::MatrixXd directional_adjoint_right = right_probe;
  Eigen::MatrixXd delta_directional_adjoint_right = Eigen::MatrixXd::Zero(
      right_probe.rows(), right_probe.cols());
  alpha_transform.apply_directional_adjoint_right(
      alpha_transform_direction,
      &directional_adjoint_right,
      &delta_directional_adjoint_right);
  Eigen::MatrixXd plus_adjoint_right = right_probe;
  Eigen::MatrixXd minus_adjoint_right = right_probe;
  xmvb::vb::ExteriorOrbitalTransform(
      alpha,
      plus_orthogonal.orbital_transform).apply_adjoint_right(
          &plus_adjoint_right);
  xmvb::vb::ExteriorOrbitalTransform(
      alpha,
      minus_orthogonal.orbital_transform).apply_adjoint_right(
          &minus_adjoint_right);
  require(
      (((plus_adjoint_right - minus_adjoint_right) /
        (2.0 * direction_step)) -
       delta_directional_adjoint_right).cwiseAbs().maxCoeff() < 2.0e-9,
      "adjoint-right exterior direction fails finite differences");

  Eigen::MatrixXd sigma_probe = coefficients;
  const Eigen::MatrixXd plus_sigma =
      xmvb::mcscf::DirectCiSigmaAction(
          alpha, beta, plus_orthogonal).apply(sigma_probe);
  const Eigen::MatrixXd minus_sigma =
      xmvb::mcscf::DirectCiSigmaAction(
          alpha, beta, minus_orthogonal).apply(sigma_probe);
  require(
      (((plus_sigma - minus_sigma) / (2.0 * direction_step)) -
       sigma_direction.apply(sigma_probe)).cwiseAbs().maxCoeff() < 2.0e-9,
      "direct-CI sigma direction fails finite differences");

  const auto apply_orthogonal_action = [&alpha, &beta, block_width](
      const xmvb::vb::OrthogonalActiveIntegrals& integrals,
      const Eigen::MatrixXd& input) {
    const xmvb::vb::ExteriorOrbitalTransform alpha_action_transform(
        alpha, integrals.orbital_transform);
    const xmvb::vb::ExteriorOrbitalTransform beta_action_transform(
        beta, integrals.orbital_transform);
    const xmvb::mcscf::DirectCiSigmaAction action_sigma(
        alpha, beta, integrals);
    Eigen::MatrixXd values = input;
    alpha_action_transform.apply_left(&values);
    for (int block = 0; block < block_width; ++block) {
      Eigen::MatrixXd value_block = values.middleCols(
          block * static_cast<int>(beta.size()),
          static_cast<int>(beta.size()));
      beta_action_transform.apply_right(&value_block);
      values.middleCols(
          block * static_cast<int>(beta.size()),
          static_cast<int>(beta.size())) = value_block;
    }
    Eigen::MatrixXd hamiltonian = action_sigma.apply(values);
    Eigen::MatrixXd overlap_image = values;
    for (int block = 0; block < block_width; ++block) {
      Eigen::MatrixXd hamiltonian_block = hamiltonian.middleCols(
          block * static_cast<int>(beta.size()),
          static_cast<int>(beta.size()));
      Eigen::MatrixXd overlap_block = overlap_image.middleCols(
          block * static_cast<int>(beta.size()),
          static_cast<int>(beta.size()));
      beta_action_transform.apply_adjoint_right(&hamiltonian_block);
      beta_action_transform.apply_adjoint_right(&overlap_block);
      hamiltonian.middleCols(
          block * static_cast<int>(beta.size()),
          static_cast<int>(beta.size())) = hamiltonian_block;
      overlap_image.middleCols(
          block * static_cast<int>(beta.size()),
          static_cast<int>(beta.size())) = overlap_block;
    }
    alpha_action_transform.apply_adjoint_left(&hamiltonian);
    alpha_action_transform.apply_adjoint_left(&overlap_image);
    return std::make_pair(hamiltonian, overlap_image);
  };

  Eigen::MatrixXd action_values = coefficients;
  Eigen::MatrixXd delta_action_values = Eigen::MatrixXd::Zero(
      coefficients.rows(), coefficients.cols());
  alpha_transform.apply_directional_left(
      alpha_transform_direction,
      &action_values,
      &delta_action_values);
  const auto beta_transform_direction = beta_transform.direction(
      orthogonal_direction.orbital_transform);
  for (int block = 0; block < block_width; ++block) {
    Eigen::MatrixXd value_block = action_values.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size()));
    Eigen::MatrixXd delta_value_block = delta_action_values.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size()));
    beta_transform.apply_directional_right(
        beta_transform_direction,
        &value_block,
        &delta_value_block);
    action_values.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size())) = value_block;
    delta_action_values.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size())) = delta_value_block;
  }
  const Eigen::MatrixXd inverse_orbital_transform =
      orthogonal.orbital_transform
          .template triangularView<Eigen::Upper>()
          .solve(Eigen::MatrixXd::Identity(n_orbitals, n_orbitals));
  const Eigen::MatrixXd relative_orbital_direction =
      orthogonal_direction.orbital_transform * inverse_orbital_transform;
  const Eigen::MatrixXd generator_adjoint =
      sigma_action.one_body_generator_adjoint(
          left_coefficients,
          action_values);
  const double predicted_exterior_direction =
      (generator_adjoint.cwiseProduct(relative_orbital_direction)).sum();
  const double applied_exterior_direction =
      (left_coefficients.cwiseProduct(delta_action_values)).sum();
  require(
      std::abs(predicted_exterior_direction - applied_exterior_direction) <
          3.0e-10 * std::max(1.0, std::abs(applied_exterior_direction)),
      "direct-CI orbital-generator adjoint is inconsistent: predicted=" +
          std::to_string(predicted_exterior_direction) +
          ", applied=" + std::to_string(applied_exterior_direction));
  const Eigen::MatrixXd generator_action =
      sigma_action.apply_one_body_generator(
          action_values,
          relative_orbital_direction);
  require(
      (generator_action - delta_action_values).cwiseAbs().maxCoeff() < 2.0e-11,
      "direct-CI one-body generator action is inconsistent with the "
      "exterior derivative");
  Eigen::MatrixXd batched_forward, batched_transpose;
  const Eigen::MatrixXd generator_transpose = relative_orbital_direction.transpose();
  sigma_action.apply_one_body_generators({
      {action_values, relative_orbital_direction, &batched_forward},
      {left_coefficients, generator_transpose, &batched_transpose}});
  require((batched_forward - generator_action).cwiseAbs().maxCoeff() < 2.0e-12,
          "batched generator changed the exterior derivative");
  const Eigen::MatrixXd expected_transpose =
      sigma_action.apply_one_body_generator(left_coefficients, generator_transpose);
  require((batched_transpose - expected_transpose).cwiseAbs().maxCoeff() < 2.0e-12,
          "batched generator mixed coefficient or generator views");
  Eigen::MatrixXd action_hamiltonian = sigma_action.apply(action_values);
  Eigen::MatrixXd delta_action_hamiltonian =
      sigma_action.apply(delta_action_values) +
      sigma_direction.apply(action_values);
  Eigen::MatrixXd action_overlap = action_values;
  Eigen::MatrixXd delta_action_overlap = delta_action_values;
  for (int block = 0; block < block_width; ++block) {
    Eigen::MatrixXd hamiltonian_block = action_hamiltonian.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size()));
    Eigen::MatrixXd delta_hamiltonian_block =
        delta_action_hamiltonian.middleCols(
            block * static_cast<int>(beta.size()),
            static_cast<int>(beta.size()));
    Eigen::MatrixXd overlap_block = action_overlap.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size()));
    Eigen::MatrixXd delta_overlap_block =
        delta_action_overlap.middleCols(
            block * static_cast<int>(beta.size()),
            static_cast<int>(beta.size()));
    beta_transform.apply_directional_adjoint_right(
        beta_transform_direction,
        &hamiltonian_block,
        &delta_hamiltonian_block);
    beta_transform.apply_directional_adjoint_right(
        beta_transform_direction,
        &overlap_block,
        &delta_overlap_block);
    action_hamiltonian.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size())) = hamiltonian_block;
    delta_action_hamiltonian.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size())) = delta_hamiltonian_block;
    action_overlap.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size())) = overlap_block;
    delta_action_overlap.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size())) = delta_overlap_block;
  }
  alpha_transform.apply_directional_adjoint_left(
      alpha_transform_direction,
      &action_hamiltonian,
      &delta_action_hamiltonian);
  alpha_transform.apply_directional_adjoint_left(
      alpha_transform_direction,
      &action_overlap,
      &delta_action_overlap);
  const auto plus_action = apply_orthogonal_action(
      plus_orthogonal, coefficients);
  const auto minus_action = apply_orthogonal_action(
      minus_orthogonal, coefficients);
  require(
      ((((plus_action.first - minus_action.first) /
         (2.0 * direction_step)) -
        delta_action_hamiltonian).cwiseAbs().maxCoeff()) < 5.0e-9,
      "complete direct-CI Hamiltonian direction fails finite differences");
  require(
      ((((plus_action.second - minus_action.second) /
         (2.0 * direction_step)) -
        delta_action_overlap).cwiseAbs().maxCoeff()) < 5.0e-9,
      "complete direct-CI overlap direction fails finite differences");

  constexpr double test_energy = 0.37;
  const Eigen::MatrixXd state_coefficients = coefficients.leftCols(
      static_cast<int>(beta.size()));
  const Eigen::MatrixXd orthogonal_state = action_values.leftCols(
      static_cast<int>(beta.size()));
  const Eigen::MatrixXd orthogonal_sigma = sigma_action.apply(
      orthogonal_state);
  const Eigen::MatrixXd orthogonal_residual =
      orthogonal_sigma - test_energy * orthogonal_state;
  const xmvb::mcscf::DirectCiIntegralAdjoint state_integral_adjoint =
      sigma_action.integral_adjoint(
          orthogonal_state,
          orthogonal_state);
  const Eigen::MatrixXd state_generator_adjoint =
      2.0 * sigma_action.one_body_generator_adjoint(
          orthogonal_residual,
          orthogonal_state);
  const xmvb::vb::NonorthogonalActiveIntegralAdjoint state_adjoint =
      xmvb::vb::backpropagate_orthogonal_active_integral_adjoint(
          orthogonal,
          state_integral_adjoint.one_electron,
          state_integral_adjoint.pair_kernel,
          state_generator_adjoint);
  Eigen::MatrixXd original_pair_direction(n_pairs, n_pairs);
  for (int column = 0; column < n_pairs; ++column) {
    for (int row = 0; row < n_pairs; ++row) {
      original_pair_direction(row, column) =
          packed_two_electron_direction[
              xmvb::core::PairIndex::packed_pair_of_pairs_index(
                  row, column)];
    }
  }
  const double predicted_state_direction =
      (state_adjoint.overlap.cwiseProduct(overlap_direction)).sum() +
      (state_adjoint.one_electron.cwiseProduct(
           one_electron_direction)).sum() +
      (state_adjoint.pair_kernel.cwiseProduct(
           original_pair_direction)).sum();
  const auto state_lagrangian = [test_energy, &state_coefficients](
      const std::pair<Eigen::MatrixXd, Eigen::MatrixXd>& images) {
    return (state_coefficients.cwiseProduct(
        images.first.leftCols(state_coefficients.cols()) -
        test_energy * images.second.leftCols(
            state_coefficients.cols()))).sum();
  };
  const double finite_difference_state_direction =
      (state_lagrangian(plus_action) - state_lagrangian(minus_action)) /
      (2.0 * direction_step);
  require(
      std::abs(predicted_state_direction -
               finite_difference_state_direction) <
          3.0e-8 *
          std::max(1.0, std::abs(finite_difference_state_direction)),
      "direct-CI active-integral adjoint fails finite differences");

  const Eigen::MatrixXd response_coefficients =
      left_coefficients.leftCols(static_cast<int>(beta.size()));
  Eigen::MatrixXd orthogonal_response = response_coefficients;
  alpha_transform.apply_left(&orthogonal_response);
  beta_transform.apply_right(&orthogonal_response);
  const Eigen::MatrixXd response_residual =
      sigma_action.apply(orthogonal_response) -
      test_energy * orthogonal_response;
  const xmvb::mcscf::DirectCiIntegralAdjoint response_integral_adjoint =
      sigma_action.integral_adjoint(
          orthogonal_response, orthogonal_state);
  const Eigen::MatrixXd response_generator_adjoint = 2.0 *
      (sigma_action.one_body_generator_adjoint(
           response_residual, orthogonal_state) +
       sigma_action.one_body_generator_adjoint(
           orthogonal_residual, orthogonal_response));
  const xmvb::vb::NonorthogonalActiveIntegralAdjoint response_adjoint =
      xmvb::vb::backpropagate_orthogonal_active_integral_adjoint(
          orthogonal,
          2.0 * response_integral_adjoint.one_electron,
          2.0 * response_integral_adjoint.pair_kernel,
          response_generator_adjoint);
  const double predicted_response_direction =
      (response_adjoint.overlap.cwiseProduct(overlap_direction)).sum() +
      (response_adjoint.one_electron.cwiseProduct(
           one_electron_direction)).sum() +
      (response_adjoint.pair_kernel.cwiseProduct(
           original_pair_direction)).sum();
  const double applied_response_direction = 2.0 *
      (response_coefficients.cwiseProduct(
           delta_action_hamiltonian.leftCols(response_coefficients.cols()) -
           test_energy *
               delta_action_overlap.leftCols(response_coefficients.cols())))
          .sum();
  require(
      std::abs(predicted_response_direction - applied_response_direction) <
          3.0e-10 * std::max(1.0, std::abs(applied_response_direction)),
      "direct-CI response adjoint is inconsistent: predicted=" +
          std::to_string(predicted_response_direction) +
          ", applied=" + std::to_string(applied_response_direction));

  const Eigen::MatrixXd directional_orthogonal_state =
      delta_action_values.leftCols(static_cast<int>(beta.size()));
  const Eigen::MatrixXd directional_orthogonal_sigma =
      sigma_action.apply(directional_orthogonal_state) +
      sigma_direction.apply(orthogonal_state);
  const Eigen::MatrixXd directional_orthogonal_residual =
      directional_orthogonal_sigma -
      test_energy * directional_orthogonal_state;
  const xmvb::mcscf::DirectCiIntegralAdjoint left_state_adjoint_direction =
      sigma_action.integral_adjoint(
          directional_orthogonal_state,
          orthogonal_state);
  const xmvb::mcscf::DirectCiIntegralAdjoint right_state_adjoint_direction =
      sigma_action.integral_adjoint(
          orthogonal_state,
          directional_orthogonal_state);
  const Eigen::MatrixXd one_gradient_direction =
      left_state_adjoint_direction.one_electron +
      right_state_adjoint_direction.one_electron;
  const Eigen::MatrixXd pair_gradient_direction =
      left_state_adjoint_direction.pair_kernel +
      right_state_adjoint_direction.pair_kernel;
  const Eigen::MatrixXd generator_gradient_direction = 2.0 *
      (sigma_action.one_body_generator_adjoint(
           directional_orthogonal_residual,
           orthogonal_state) +
       sigma_action.one_body_generator_adjoint(
           orthogonal_residual,
           directional_orthogonal_state));
  const xmvb::vb::NonorthogonalActiveIntegralAdjoint
      analytic_state_adjoint_direction =
          xmvb::vb::backpropagate_orthogonal_active_integral_adjoint_direction(
              orthogonal,
              orthogonal_direction,
              state_integral_adjoint.one_electron,
              state_integral_adjoint.pair_kernel,
              state_generator_adjoint,
              one_gradient_direction,
              pair_gradient_direction,
              generator_gradient_direction);
  const auto build_state_adjoint = [
      &alpha,
      &beta,
      &state_coefficients,
      test_energy](const xmvb::vb::OrthogonalActiveIntegrals& integrals) {
    const xmvb::vb::ExteriorOrbitalTransform alpha_state_transform(
        alpha, integrals.orbital_transform);
    const xmvb::vb::ExteriorOrbitalTransform beta_state_transform(
        beta, integrals.orbital_transform);
    const xmvb::mcscf::DirectCiSigmaAction state_sigma(alpha, beta, integrals);
    Eigen::MatrixXd state = state_coefficients;
    alpha_state_transform.apply_left(&state);
    beta_state_transform.apply_right(&state);
    const Eigen::MatrixXd sigma = state_sigma.apply(state);
    const Eigen::MatrixXd residual = sigma - test_energy * state;
    const xmvb::mcscf::DirectCiIntegralAdjoint integral_gradient =
        state_sigma.integral_adjoint(state, state);
    const Eigen::MatrixXd generator_gradient =
        2.0 * state_sigma.one_body_generator_adjoint(residual, state);
    return xmvb::vb::backpropagate_orthogonal_active_integral_adjoint(
        integrals,
        integral_gradient.one_electron,
        integral_gradient.pair_kernel,
        generator_gradient);
  };
  const xmvb::vb::NonorthogonalActiveIntegralAdjoint plus_state_adjoint =
      build_state_adjoint(plus_orthogonal);
  const xmvb::vb::NonorthogonalActiveIntegralAdjoint minus_state_adjoint =
      build_state_adjoint(minus_orthogonal);
  require(
      ((((plus_state_adjoint.overlap - minus_state_adjoint.overlap) /
         (2.0 * direction_step)) -
        analytic_state_adjoint_direction.overlap).cwiseAbs().maxCoeff()) <
          2.0e-7,
      "direct-CI overlap-adjoint direction fails finite differences");
  require(
      ((((plus_state_adjoint.one_electron -
          minus_state_adjoint.one_electron) /
         (2.0 * direction_step)) -
        analytic_state_adjoint_direction.one_electron)
           .cwiseAbs()
           .maxCoeff()) < 2.0e-7,
      "direct-CI one-electron-adjoint direction fails finite differences");
  require(
      ((((plus_state_adjoint.pair_kernel -
          minus_state_adjoint.pair_kernel) /
         (2.0 * direction_step)) -
        analytic_state_adjoint_direction.pair_kernel)
           .cwiseAbs()
           .maxCoeff()) < 2.0e-7,
      "direct-CI pair-kernel-adjoint direction fails finite differences");

  Eigen::MatrixXd orthogonal_coefficients = coefficients;
  for (int block = 0; block < block_width; ++block) {
    Eigen::MatrixXd block_coefficients =
        orthogonal_coefficients.middleCols(
            block * static_cast<int>(beta.size()),
            static_cast<int>(beta.size()));
    alpha_transform.apply_left(&block_coefficients);
    beta_transform.apply_right(&block_coefficients);
    orthogonal_coefficients.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size())) = block_coefficients;
  }
  Eigen::MatrixXd direct_hamiltonian =
      sigma_action.apply(orthogonal_coefficients);
  Eigen::MatrixXd direct_overlap = orthogonal_coefficients;
  for (int block = 0; block < block_width; ++block) {
    Eigen::MatrixXd hamiltonian_block = direct_hamiltonian.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size()));
    Eigen::MatrixXd overlap_block = direct_overlap.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size()));
    beta_transform.apply_adjoint_right(&hamiltonian_block);
    alpha_transform.apply_adjoint_left(&hamiltonian_block);
    beta_transform.apply_adjoint_right(&overlap_block);
    alpha_transform.apply_adjoint_left(&overlap_block);
    direct_hamiltonian.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size())) = hamiltonian_block;
    direct_overlap.middleCols(
        block * static_cast<int>(beta.size()),
        static_cast<int>(beta.size())) = overlap_block;
  }

  const int product_dimension =
      static_cast<int>(alpha.size() * beta.size());
  Eigen::MatrixXd dense_hamiltonian(product_dimension, product_dimension);
  Eigen::MatrixXd dense_overlap(product_dimension, product_dimension);
  const xmvb::vb::DeterminantPairEvaluator evaluator;
  for (int target_alpha = 0;
       target_alpha < static_cast<int>(alpha.size());
       ++target_alpha) {
    for (int target_beta = 0;
         target_beta < static_cast<int>(beta.size());
         ++target_beta) {
      const int target =
          target_alpha * static_cast<int>(beta.size()) + target_beta;
      for (int source_alpha = 0;
           source_alpha < static_cast<int>(alpha.size());
           ++source_alpha) {
        for (int source_beta = 0;
             source_beta < static_cast<int>(beta.size());
             ++source_beta) {
          const int source =
              source_alpha * static_cast<int>(beta.size()) + source_beta;
          const auto pair = evaluator.evaluate(
              alpha[target_alpha],
              alpha[source_alpha],
              beta[target_beta],
              beta[source_beta],
              packed_overlap,
              one_electron,
              n_orbitals,
              two_electron,
              false);
          dense_hamiltonian(target, source) = pair.total_hamiltonian;
          dense_overlap(target, source) = pair.overlap_determinant;
        }
      }
    }
  }

  Eigen::MatrixXd coefficient_columns(product_dimension, block_width);
  for (int block = 0; block < block_width; ++block) {
    for (int alpha_index = 0;
         alpha_index < static_cast<int>(alpha.size());
         ++alpha_index) {
      for (int beta_index = 0;
           beta_index < static_cast<int>(beta.size());
           ++beta_index) {
        coefficient_columns(
            alpha_index * static_cast<int>(beta.size()) + beta_index,
            block) = coefficients(
                alpha_index,
                block * static_cast<int>(beta.size()) + beta_index);
      }
    }
  }
  const Eigen::MatrixXd reference_hamiltonian =
      dense_hamiltonian * coefficient_columns;
  const Eigen::MatrixXd reference_overlap =
      dense_overlap * coefficient_columns;
  for (int block = 0; block < block_width; ++block) {
    for (int alpha_index = 0;
         alpha_index < static_cast<int>(alpha.size());
         ++alpha_index) {
      for (int beta_index = 0;
           beta_index < static_cast<int>(beta.size());
           ++beta_index) {
        const int product =
            alpha_index * static_cast<int>(beta.size()) + beta_index;
        const int column =
            block * static_cast<int>(beta.size()) + beta_index;
        require(
            std::abs(
                direct_hamiltonian(alpha_index, column) -
                reference_hamiltonian(product, block)) < 2.0e-10,
            "orthogonal direct-CI Hamiltonian action is inconsistent");
        require(
            std::abs(
                direct_overlap(alpha_index, column) -
                reference_overlap(product, block)) < 2.0e-10,
            "orthogonal direct-CI overlap action is inconsistent");
      }
    }
  }

  // Project the exact determinant-product action onto three arbitrary sparse
  // structure vectors. The structure subspace is deliberately tiny and is
  // not invariant under H; exactness therefore cannot rely on a full-CI
  // structure basis.
  constexpr int n_test_structures = 3;
  Eigen::MatrixXd structure_expansion = Eigen::MatrixXd::Zero(
      product_dimension, n_test_structures);
  for (int structure = 0; structure < n_test_structures; ++structure) {
    for (int term = 0; term < 7; ++term) {
      const int product =
          (5 * term + 11 * structure + structure * term) % product_dimension;
      structure_expansion(product, structure) +=
          std::sin(0.31 * static_cast<double>((term + 1) * (structure + 2)));
    }
  }
  Eigen::MatrixXd structure_vectors(n_test_structures, block_width);
  for (int column = 0; column < block_width; ++column) {
    for (int row = 0; row < n_test_structures; ++row) {
      structure_vectors(row, column) =
          std::cos(0.27 * static_cast<double>((row + 1) * (column + 3)));
    }
  }
  const Eigen::MatrixXd product_vectors =
      structure_expansion * structure_vectors;
  Eigen::MatrixXd packed_product_vectors(
      static_cast<int>(alpha.size()),
      block_width * static_cast<int>(beta.size()));
  for (int block = 0; block < block_width; ++block) {
    for (int alpha_index = 0;
         alpha_index < static_cast<int>(alpha.size());
         ++alpha_index) {
      for (int beta_index = 0;
           beta_index < static_cast<int>(beta.size());
           ++beta_index) {
        packed_product_vectors(
            alpha_index,
            block * static_cast<int>(beta.size()) + beta_index) =
            product_vectors(
                alpha_index * static_cast<int>(beta.size()) + beta_index,
                block);
      }
    }
  }
  const auto projected_action = apply_orthogonal_action(
      orthogonal, packed_product_vectors);
  Eigen::MatrixXd projected_hamiltonian_products(
      product_dimension, block_width);
  Eigen::MatrixXd projected_overlap_products(
      product_dimension, block_width);
  for (int block = 0; block < block_width; ++block) {
    for (int alpha_index = 0;
         alpha_index < static_cast<int>(alpha.size());
         ++alpha_index) {
      for (int beta_index = 0;
           beta_index < static_cast<int>(beta.size());
           ++beta_index) {
        const int product =
            alpha_index * static_cast<int>(beta.size()) + beta_index;
        const int packed_column =
            block * static_cast<int>(beta.size()) + beta_index;
        projected_hamiltonian_products(product, block) =
            projected_action.first(alpha_index, packed_column);
        projected_overlap_products(product, block) =
            projected_action.second(alpha_index, packed_column);
      }
    }
  }
  const Eigen::MatrixXd direct_three_structure_hamiltonian =
      structure_expansion.transpose() * projected_hamiltonian_products;
  const Eigen::MatrixXd direct_three_structure_overlap =
      structure_expansion.transpose() * projected_overlap_products;
  const Eigen::MatrixXd reference_three_structure_hamiltonian =
      structure_expansion.transpose() * dense_hamiltonian *
      structure_expansion * structure_vectors;
  const Eigen::MatrixXd reference_three_structure_overlap =
      structure_expansion.transpose() * dense_overlap *
      structure_expansion * structure_vectors;
  require(
      (direct_three_structure_hamiltonian -
       reference_three_structure_hamiltonian).cwiseAbs().maxCoeff() < 2.0e-10,
      "direct CI is inexact on an arbitrary three-structure subspace");
  require(
      (direct_three_structure_overlap -
       reference_three_structure_overlap).cwiseAbs().maxCoeff() < 2.0e-10,
      "direct-CI overlap is inexact on an arbitrary three-structure subspace");

  std::vector<std::vector<int>> product_alpha;
  std::vector<std::vector<int>> product_beta;
  std::vector<std::vector<xmvb::vb::StructureExpansionTerm>>
      determinant_to_structure_terms(product_dimension);
  product_alpha.reserve(product_dimension);
  product_beta.reserve(product_dimension);
  for (int alpha_index = 0;
       alpha_index < static_cast<int>(alpha.size());
       ++alpha_index) {
    for (int beta_index = 0;
         beta_index < static_cast<int>(beta.size());
         ++beta_index) {
      const int product =
          alpha_index * static_cast<int>(beta.size()) + beta_index;
      product_alpha.push_back(alpha[alpha_index]);
      product_beta.push_back(beta[beta_index]);
      for (int structure = 0; structure < n_test_structures; ++structure) {
        const double coefficient = structure_expansion(product, structure);
        if (coefficient != 0.0) {
          determinant_to_structure_terms[product].push_back(
              xmvb::vb::StructureExpansionTerm{structure, coefficient});
        }
      }
    }
  }
  const auto topology = xmvb::vb::build_same_spin_pair_topology(
      product_alpha, product_beta, n_orbitals);
  require(!topology.enabled(), "topology-only context materialized pair data");
  require(topology.alpha_pair_cache.empty() && topology.beta_pair_cache.empty(),
          "topology-only context retained ordered pair arrays");

  xmvb::vb::StructureDiagonal exact_diagonal;
  exact_diagonal.hamiltonian =
      (structure_expansion.transpose() * dense_hamiltonian *
       structure_expansion).diagonal();
  exact_diagonal.overlap =
      (structure_expansion.transpose() * dense_overlap *
       structure_expansion).diagonal();
  const xmvb::vb::FullDeterminantStructureHamiltonianOverlapBuilder builder;
  const xmvb::vb::StructureDiagonal streamed_diagonal =
      builder.build_diagonal(
          product_alpha,
          product_beta,
          determinant_to_structure_terms,
          packed_overlap,
          one_electron,
          n_orbitals,
          two_electron,
          n_test_structures,
          topology);
  require(
      (streamed_diagonal.hamiltonian - exact_diagonal.hamiltonian)
              .cwiseAbs()
              .maxCoeff() < 2.0e-10,
      "streamed structure Hamiltonian diagonal is inexact");
  require(
      (streamed_diagonal.overlap - exact_diagonal.overlap)
              .cwiseAbs()
              .maxCoeff() < 2.0e-10,
      "streamed structure overlap diagonal is inexact");
  const Eigen::VectorXd overlap_only_diagonal =
      builder.build_exact_overlap_diagonal(
          product_alpha,
          product_beta,
          determinant_to_structure_terms,
          packed_overlap,
          n_orbitals,
          n_test_structures,
          topology);
  require(
      (overlap_only_diagonal - exact_diagonal.overlap)
              .cwiseAbs()
              .maxCoeff() < 2.0e-10,
      "overlap-only structure diagonal is inexact");

  xmvb::vb::StructureDiagonal expected_jacobi;
  expected_jacobi.hamiltonian =
      Eigen::VectorXd::Zero(n_test_structures);
  expected_jacobi.overlap = Eigen::VectorXd::Zero(n_test_structures);
  for (int determinant = 0; determinant < product_dimension; ++determinant) {
    for (int structure = 0; structure < n_test_structures; ++structure) {
      const double coefficient =
          structure_expansion(determinant, structure);
      expected_jacobi.hamiltonian[structure] +=
          coefficient * coefficient *
          dense_hamiltonian(determinant, determinant);
      expected_jacobi.overlap[structure] +=
          coefficient * coefficient * dense_overlap(determinant, determinant);
    }
  }
  const xmvb::vb::StructureDiagonal jacobi =
      builder.build_davidson_jacobi_preconditioner(
          product_alpha,
          product_beta,
          determinant_to_structure_terms,
          packed_overlap,
          one_electron,
          n_orbitals,
          two_electron,
          n_test_structures,
          topology);
  require(
      (jacobi.hamiltonian - expected_jacobi.hamiltonian)
              .cwiseAbs()
              .maxCoeff() < 2.0e-10,
      "determinant-diagonal Hamiltonian Jacobi model is inexact");
  require(
      (jacobi.overlap - expected_jacobi.overlap)
              .cwiseAbs()
              .maxCoeff() < 2.0e-10,
      "determinant-diagonal overlap Jacobi model is inexact");
  require(
      (jacobi.hamiltonian - exact_diagonal.hamiltonian)
              .cwiseAbs()
              .maxCoeff() > 1.0e-8,
      "Jacobi model unexpectedly retained determinant cross terms");
  xmvb::vb::StructureAction topology_action(
      determinant_to_structure_terms,
      n_test_structures,
      topology,
      packed_overlap,
      one_electron,
      two_electron,
      n_orbitals,
      &exact_diagonal);
  require(
      topology_action.storage().orthogonal_direct_ci,
      "complete fixed-spin carrier did not select direct CI");
  bool selected_metric_rejected = false;
  try {
    const xmvb::vb::StructureMetricInverse invalid(
        topology_action, topology.alpha_reuse_table.unique_determinants,
        topology.beta_reuse_table.unique_determinants, false, 65536, 65536);
  } catch (const std::invalid_argument&) {
    selected_metric_rejected = true;
  }
  require(selected_metric_rejected,
          "complete carrier incorrectly admitted a selected structure metric inverse");

  // Complete spin-product basis with nonorthogonal recoupling inside each
  // spatial occupation pattern. This tests nontrivial Gram blocks and chunks.
  std::vector<std::vector<xmvb::vb::StructureExpansionTerm>> full_terms(product_dimension);
  Eigen::MatrixXd full_expansion = Eigen::MatrixXd::Identity(product_dimension, product_dimension);
  std::map<std::vector<int>, int> first_product;
  for (int product = 0; product < product_dimension; ++product) {
    full_terms[product].push_back({product, 1.0});
    std::vector<int> occupation(n_orbitals, 0);
    for (int orbital : product_alpha[product]) ++occupation[orbital];
    for (int orbital : product_beta[product]) ++occupation[orbital];
    const auto [entry, inserted] = first_product.emplace(occupation, product);
    if (!inserted) {
      full_terms[entry->second].push_back({product, 0.25});
      full_expansion(entry->second, product) = 0.25;
    }
  }
  const Eigen::MatrixXd full_overlap = full_expansion.transpose() * dense_overlap * full_expansion;
  xmvb::vb::StructureDiagonal full_diagonal;
  full_diagonal.overlap = full_overlap.diagonal();
  full_diagonal.hamiltonian =
      (full_expansion.transpose() * dense_hamiltonian * full_expansion).diagonal();
  const xmvb::vb::StructureAction full_action(
      full_terms, product_dimension, topology, packed_overlap,
      one_electron, two_electron, n_orbitals, &full_diagonal);
  check_joint_structure_response(full_action, alpha.size(), beta.size());
  check_joint_structure_response(topology_action, alpha.size(), beta.size());
  const std::size_t one_column_bytes = 5 * product_dimension * sizeof(double);
  require(xmvb::vb::StructureMetricInverse::fits(
      full_action, topology.alpha_reuse_table.unique_determinants,
      topology.beta_reuse_table.unique_determinants, 65536, one_column_bytes),
      "structure metric planner rejected an affordable block factorization");
  const xmvb::vb::StructureMetricInverse full_metric(
      full_action, topology.alpha_reuse_table.unique_determinants,
      topology.beta_reuse_table.unique_determinants, true, 65536, one_column_bytes);
  Eigen::MatrixXd metric_rhs(product_dimension, 7);
  for (int column = 0; column < metric_rhs.cols(); ++column)
    for (int row = 0; row < metric_rhs.rows(); ++row)
      metric_rhs(row, column) = std::sin(0.19 * (row + 1) * (column + 2));
  const Eigen::MatrixXd metric_solution = full_metric.apply(metric_rhs);
  for (int column = 0; column < metric_rhs.cols(); ++column) {
    const Eigen::VectorXd coefficients = metric_rhs.col(column);
    const double energy = 0.2 * (column + 1);
    const auto images = full_action.apply(coefficients);
    const Eigen::MatrixXd residual = images.hamiltonian - energy * images.overlap;
    const double reference = std::sqrt(
        residual.col(0).dot(full_metric.apply(residual).col(0)) /
        coefficients.dot(images.overlap.col(0)));
    require(std::abs(full_action.complete_space_residual_norm(coefficients, energy) - reference) <
                1e-10 * std::max(1.0, reference),
            "orthogonal carrier residual differs from the physical dual metric");
  }
  const Eigen::MatrixXd reference_metric_solution = full_overlap.partialPivLu().solve(metric_rhs);
  require((metric_solution - reference_metric_solution).cwiseAbs().maxCoeff() < 3.0e-11 &&
              (full_action.apply(metric_solution).overlap - metric_rhs).cwiseAbs().maxCoeff() < 3.0e-11,
          "sparse complete-space metric inverse differs from its dense reference");
  require(full_metric.factor_bytes() < static_cast<std::size_t>(product_dimension) * product_dimension * sizeof(double) &&
              full_metric.max_block_size() == 6 && full_metric.block_count() == 19,
          "spin metric lost its occupation-block representation");
  // Independent explicit pullback of the carrier diagonal, including BOTH
  // Riesz maps. A single F^T D^-1 F is not a structure-coordinate inverse.
  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(product_dimension,product_dimension);
  const Eigen::MatrixXd carriers = full_action.orthogonalize_structure_block(identity);
  Eigen::MatrixXd f(product_dimension,product_dimension);
  for (int column=0;column<product_dimension;++column)
    for (int a=0;a<static_cast<int>(alpha.size());++a)
      for (int b=0;b<static_cast<int>(beta.size());++b)
        f(a*beta.size()+b,column)=carriers(a,column*beta.size()+b);
  const Eigen::MatrixXd inverse_m = full_overlap.partialPivLu().solve(identity);
  const double root_energy = diagonal.minCoeff()-1.0;
  Eigen::VectorXd inverse_d(product_dimension);
  for (int a=0;a<static_cast<int>(alpha.size());++a)
    for (int b=0;b<static_cast<int>(beta.size());++b)
      inverse_d[a*beta.size()+b]=1.0/(diagonal(a,b)-root_energy);
  const Eigen::MatrixXd reference_p =
      inverse_m*f.transpose()*inverse_d.asDiagonal()*f*inverse_m;
  const Eigen::MatrixXd correction = full_metric.apply_carrier_inverse(
      metric_rhs.leftCols(1),Eigen::VectorXd::Constant(1,root_energy),0.0);
  const Eigen::MatrixXd expected_correction = reference_p*metric_rhs.leftCols(1);
  require((correction-expected_correction).norm()<
              1e-10*std::max(1.0,expected_correction.norm()),
          "carrier Davidson inverse differs from the dense metric pullback");
  const Eigen::MatrixXd cancelled = full_metric.apply_carrier_inverse(
      metric_rhs,Eigen::VectorXd::Constant(metric_rhs.cols(),root_energy),0.0);
  require((cancelled-reference_p*metric_rhs).norm()<
              1e-10*std::max(1.0,(reference_p*metric_rhs).norm()),
          "cancelled multi-RHS carrier inverse differs from the independent pullback");
  const Eigen::MatrixXd j=full_metric.to_ci_coordinates(identity);
  const Eigen::MatrixXd ji=full_metric.from_ci_coordinates(identity);
  require((j.transpose()*j-full_overlap).norm()<2e-11 &&
              (j*ji-identity).norm()<2e-11,
          "complete CI boundary maps changed the physical metric or round trip");
  const Eigen::MatrixXd raw_h=full_expansion.transpose()*dense_hamiltonian*full_expansion;
  const Eigen::MatrixXd ci_h=full_metric.apply_ci_hamiltonian(identity);
  require((ci_h-ji.transpose()*raw_h*ji).norm()<2e-10 &&
              (ci_h-ci_h.transpose()).norm()<2e-11,
          "complete CI Hamiltonian is not the independent symmetric congruence");
  require((full_metric.ci_covector_to_structure(metric_rhs)-j.transpose()*metric_rhs).norm()<2e-11,
          "complete CI Ritz covector pullback is inconsistent");
  require((full_metric.apply_ci_preconditioner(metric_rhs,
              Eigen::VectorXd::Constant(metric_rhs.cols(),root_energy),0.0)-
              j*reference_p*j.transpose()*metric_rhs).norm()<2e-10,
          "complete CI preconditioner changed under the coordinate congruence");
  const Eigen::MatrixXd restored=full_metric.from_ci_coordinates(metric_rhs);
  const auto raw_images=full_action.apply(restored);
  require((full_metric.ci_covector_to_structure(
              full_metric.apply_ci_hamiltonian(metric_rhs))-raw_images.hamiltonian).norm()<2e-10 &&
              (full_metric.ci_covector_to_structure(metric_rhs)-raw_images.overlap).norm()<2e-11,
          "complete CI exit failed to restore H C and M C");
  auto original_action = std::make_unique<xmvb::vb::StructureAction>(
      full_terms,product_dimension,topology,packed_overlap,
      one_electron,two_electron,n_orbitals,&full_diagonal);
  xmvb::vb::StructureMetricInverse moved_metric(
      *original_action,topology.alpha_reuse_table.unique_determinants,
      topology.beta_reuse_table.unique_determinants,true,65536,one_column_bytes);
  xmvb::vb::StructureAction moved_action(std::move(*original_action));
  moved_metric.rebind(moved_action);
  original_action.reset();
  require((moved_metric.apply(metric_rhs)-metric_solution).norm()<1e-11,
          "moving a forward action invalidated its reusable metric factors");
  for (bool factor_limit : {true, false}) {
    require(!xmvb::vb::StructureMetricInverse::fits(
        full_action, topology.alpha_reuse_table.unique_determinants,
        topology.beta_reuse_table.unique_determinants,
        factor_limit ? 0 : 65536,
        factor_limit ? one_column_bytes : one_column_bytes - 1),
        "structure metric planner ignored its byte budget");
    bool rejected = false;
    try {
      const xmvb::vb::StructureMetricInverse invalid(
          full_action, topology.alpha_reuse_table.unique_determinants,
          topology.beta_reuse_table.unique_determinants, true,
          factor_limit ? 0 : 65536,
          factor_limit ? one_column_bytes : one_column_bytes - 1);
    } catch (const std::length_error&) {
      rejected = true;
    }
    require(rejected, "structure metric ignored its factor or workspace byte limit");
  }
  const auto topology_images = topology_action.apply(structure_vectors);
  const xmvb::vb::StructureAction action_only(
      determinant_to_structure_terms, n_test_structures, topology,
      packed_overlap, one_electron, two_electron, n_orbitals, nullptr,
      xmvb::vb::StructureActionMode::ActionOnly);
  const auto bare_images = action_only.apply(structure_vectors);
  require(action_only.storage().diagonal_bytes == 0 &&
              action_only.preconditioner_diagonal().overlap.size() == 0 &&
              (bare_images.hamiltonian - topology_images.hamiltonian).norm() < 2e-10 &&
              (bare_images.overlap - topology_images.overlap).norm() < 2e-10,
          "action-only direct CI built a diagonal or changed H/S");
  Eigen::MatrixXd normalized = structure_vectors;
  auto scaled = bare_images;
  for (int col = 0; col < normalized.cols(); ++col) {
    const double norm = std::sqrt(normalized.col(col).dot(scaled.overlap.col(col)));
    normalized.col(col) /= norm;
    scaled.hamiltonian.col(col) /= norm;
    scaled.overlap.col(col) /= norm;
  }
  const auto normalized_images = action_only.apply(normalized);
  require((scaled.hamiltonian - normalized_images.hamiltonian).norm() < 2e-10 &&
              (scaled.overlap - normalized_images.overlap).norm() < 2e-10,
          "column normalization requires a second H/S action");
  require((topology_action.apply_overlap(structure_vectors) -
      topology_images.overlap).cwiseAbs().maxCoeff() < 2.0e-12,
      "overlap-only direct-CI action differs from the combined action");
  require(
      (topology_images.hamiltonian -
       reference_three_structure_hamiltonian).cwiseAbs().maxCoeff() < 2.0e-10,
      "topology-only Hamiltonian action differs from the dense reference");
  require(
      (topology_images.overlap -
       reference_three_structure_overlap).cwiseAbs().maxCoeff() < 2.0e-10,
      "topology-only overlap action differs from the dense reference");
  std::vector<std::vector<int>> incomplete_alpha;
  std::vector<std::vector<int>> incomplete_beta;
  std::vector<std::vector<xmvb::vb::StructureExpansionTerm>>
      incomplete_terms;
  std::vector<int> retained_products;
  for (int product = 0; product < product_dimension; ++product) {
    if (product_alpha[product] == alpha.back()) {
      continue;
    }
    incomplete_alpha.push_back(product_alpha[product]);
    incomplete_beta.push_back(product_beta[product]);
    incomplete_terms.push_back(determinant_to_structure_terms[product]);
    retained_products.push_back(product);
  }
  Eigen::MatrixXd incomplete_expansion(
      static_cast<int>(retained_products.size()), n_test_structures);
  Eigen::MatrixXd incomplete_hamiltonian(
      retained_products.size(), retained_products.size());
  Eigen::MatrixXd incomplete_overlap(
      retained_products.size(), retained_products.size());
  for (int left = 0; left < static_cast<int>(retained_products.size()); ++left) {
    incomplete_expansion.row(left) =
        structure_expansion.row(retained_products[left]);
    for (int right = 0;
         right < static_cast<int>(retained_products.size());
         ++right) {
      incomplete_hamiltonian(left, right) =
          dense_hamiltonian(retained_products[left], retained_products[right]);
      incomplete_overlap(left, right) =
          dense_overlap(retained_products[left], retained_products[right]);
    }
  }
  const auto incomplete_topology = xmvb::vb::build_same_spin_pair_topology(
      incomplete_alpha, incomplete_beta, n_orbitals);
  xmvb::vb::StructureAction streamed_action(
      incomplete_terms,
      n_test_structures,
      incomplete_topology,
      packed_overlap,
      one_electron,
      two_electron,
      n_orbitals,
      &exact_diagonal);
  require(
      !streamed_action.storage().orthogonal_direct_ci,
      "incomplete fixed-spin carrier incorrectly selected direct CI");
  const auto streamed_images =
      streamed_action.apply_streamed(structure_vectors);
  const xmvb::vb::StructureAction pair_action_only(
      incomplete_terms, n_test_structures, incomplete_topology,
      packed_overlap, one_electron, two_electron, n_orbitals, nullptr,
      xmvb::vb::StructureActionMode::ActionOnly);
  const auto bare_pair_images = pair_action_only.apply(structure_vectors);
  require(pair_action_only.storage().diagonal_bytes == 0 &&
              (bare_pair_images.hamiltonian - streamed_images.hamiltonian).norm() < 2e-10 &&
              (bare_pair_images.overlap - streamed_images.overlap).norm() < 2e-10,
          "action-only incomplete carrier built a diagonal or changed H/S");
  require((streamed_action.apply_overlap(structure_vectors) -
      streamed_images.overlap).cwiseAbs().maxCoeff() < 2.0e-12,
      "overlap-only pair action differs from the streamed action");
  const Eigen::MatrixXd incomplete_reference_hamiltonian =
      incomplete_expansion.transpose() * incomplete_hamiltonian *
      incomplete_expansion * structure_vectors;
  const Eigen::MatrixXd incomplete_reference_overlap =
      incomplete_expansion.transpose() * incomplete_overlap *
      incomplete_expansion * structure_vectors;
  require(
      (streamed_images.hamiltonian -
       incomplete_reference_hamiltonian).cwiseAbs().maxCoeff() < 2.0e-10,
      "streamed Hamiltonian action differs from the dense reference");
  require(
      (streamed_images.overlap -
       incomplete_reference_overlap).cwiseAbs().maxCoeff() < 2.0e-10,
      "streamed overlap action differs from the dense reference");
}

/** @brief Exercise partial final tiles and multiple RHS with a known spectrum. */
void check_sigma_tile_boundary() {
  constexpr int n_orbitals = 10;
  constexpr int block_width = 2;
  // 252 strings: partial alpha tile and uneven beta ownership at 32 workers.
  const auto spin = complete_space(n_orbitals, 5);
  xmvb::vb::OrthogonalActiveIntegrals integrals;
  integrals.one_electron = Eigen::MatrixXd::Zero(n_orbitals, n_orbitals);
  for (int i = 0; i < n_orbitals; ++i)
    integrals.one_electron(i, i) = 0.17 * (i + 1);
  const int n_pairs = xmvb::integrals::packed_active_pair_count(n_orbitals);
  integrals.pair_kernel = Eigen::MatrixXd::Zero(n_pairs, n_pairs);
  const xmvb::mcscf::DirectCiSigmaAction action(spin, spin, integrals);
  const int n_strings = static_cast<int>(spin.size());
  Eigen::MatrixXd coefficients(n_strings, block_width * n_strings);
  Eigen::VectorXd energies = Eigen::VectorXd::Zero(n_strings);
  for (int i = 0; i < n_strings; ++i)
    for (int orbital : spin[i]) energies[i] += integrals.one_electron(orbital, orbital);
  Eigen::MatrixXd expected(coefficients.rows(), coefficients.cols());
  for (int j = 0; j < coefficients.cols(); ++j)
    for (int i = 0; i < coefficients.rows(); ++i) {
      coefficients(i, j) = std::sin(0.031 * (i + 1) * (j + 2));
      expected(i, j) = (energies[i] + energies[j % n_strings]) * coefficients(i, j);
    }
  require((action.apply(coefficients) - expected).cwiseAbs().maxCoeff() < 2e-12,
          "direct-CI partial tile or RHS packing differs from known spectrum");
  check_fused_response(action, coefficients, coefficients);
#ifdef _OPENMP
  const int dynamic = omp_get_dynamic();
  omp_set_dynamic(1);
  const Eigen::MatrixXd dynamic_result = action.apply(coefficients);
  check_fused_response(action, coefficients, coefficients);
  omp_set_dynamic(dynamic);
  require((dynamic_result - expected).cwiseAbs().maxCoeff() < 2e-12,
          "direct-CI dynamic team reduction differs from known spectrum");
  Eigen::MatrixXd nested_result;
#pragma omp parallel num_threads(2)
  {
#pragma omp single
    {
      nested_result = action.apply(coefficients);
      check_fused_response(action, coefficients, coefficients);
    }
  }
  require((nested_result - expected).cwiseAbs().maxCoeff() < 2e-12,
          "direct-CI nested team reduction differs from known spectrum");
#endif
  const auto adjoint = action.integral_adjoint(coefficients, coefficients);
  require(std::abs(adjoint.one_electron.cwiseProduct(integrals.one_electron).sum() -
                   coefficients.cwiseProduct(expected).sum()) < 2e-8,
          "tiled direct-CI adjoint differs from action bilinear form");
#ifdef _OPENMP
  omp_set_dynamic(1);
  const auto dynamic_adjoint = action.integral_adjoint(coefficients, coefficients);
  omp_set_dynamic(dynamic);
  require((dynamic_adjoint.pair_kernel - adjoint.pair_kernel).norm() <
              1e-11 * std::max(1.0, adjoint.pair_kernel.norm()),
          "adjoint output ownership depends on the actual team size");
  xmvb::mcscf::DirectCiIntegralAdjoint nested_adjoint;
#pragma omp parallel num_threads(2)
  {
#pragma omp single
    nested_adjoint = action.integral_adjoint(coefficients, coefficients);
  }
  require((nested_adjoint.one_electron - adjoint.one_electron).norm() <
              1e-11 * std::max(1.0, adjoint.one_electron.norm()),
          "nested adjoint differs from the shared-tile contraction");
#endif
}

void check_distinct_spin_sigma_action() {
  constexpr int n_orbitals = 4;
  constexpr int block_width = 2;
  const auto alpha = complete_space(n_orbitals, 1);
  const auto beta = complete_space(n_orbitals, 2);

  Eigen::MatrixXd overlap = Eigen::MatrixXd::Identity(
      n_orbitals, n_orbitals);
  std::vector<double> packed_overlap(
      overlap.data(), overlap.data() + overlap.size());
  Eigen::MatrixXd one_electron(n_orbitals, n_orbitals);
  for (int column = 0; column < n_orbitals; ++column) {
    for (int row = 0; row <= column; ++row) {
      const double value =
          0.11 * std::cos(0.3 * static_cast<double>((row + 1) * (column + 2)));
      one_electron(row, column) = value;
      one_electron(column, row) = value;
    }
  }
  const int n_pairs = xmvb::integrals::packed_active_pair_count(n_orbitals);
  Eigen::MatrixXd factor(6, n_pairs);
  for (int column = 0; column < n_pairs; ++column) {
    for (int row = 0; row < factor.rows(); ++row) {
      factor(row, column) =
          0.09 * std::sin(0.17 * static_cast<double>((row + 2) * (column + 1)));
    }
  }
  const Eigen::MatrixXd pair_kernel = factor.transpose() * factor;
  const auto pack_kernel = [n_orbitals](
      const Eigen::MatrixXd& kernel) {
    xmvb::integrals::ActiveSpaceTwoElectronResult result;
    result.packed_active_two_electron_integrals.resize(
        xmvb::integrals::packed_active_two_electron_integral_count(n_orbitals));
    for (int column = 0; column < kernel.cols(); ++column) {
      for (int row = 0; row <= column; ++row) {
        result.packed_active_two_electron_integrals[
            xmvb::core::PairIndex::packed_pair_of_pairs_index(
                row, column)] = kernel(row, column);
      }
    }
    return result;
  };
  const xmvb::integrals::ActiveSpaceTwoElectronResult two_electron =
      pack_kernel(pair_kernel);
  xmvb::vb::OrthogonalActiveIntegrals integrals;
  integrals.orbital_transform=Eigen::MatrixXd::Identity(n_orbitals,n_orbitals);
  integrals.one_electron=one_electron;
  integrals.pair_kernel=pair_kernel;
  const xmvb::mcscf::DirectCiSigmaAction action(alpha, beta, integrals);

  Eigen::MatrixXd coefficients(
      static_cast<int>(alpha.size()),
      block_width * static_cast<int>(beta.size()));
  for (int column = 0; column < coefficients.cols(); ++column) {
    for (int row = 0; row < coefficients.rows(); ++row) {
      coefficients(row, column) =
          std::sin(0.21 * static_cast<double>((row + 2) * (column + 1)));
    }
  }
  const int n_beta = static_cast<int>(beta.size());
  const int product_dimension = static_cast<int>(alpha.size()) * n_beta;
  std::vector<std::vector<int>> product_alpha, product_beta;
  std::vector<std::vector<xmvb::vb::StructureExpansionTerm>> terms(product_dimension);
  for (const auto& a : alpha)
    for (const auto& b : beta) {
      const int i = product_alpha.size();
      product_alpha.push_back(a);
      product_beta.push_back(b);
      terms[i].push_back({i, 1.0});
    }
  const auto topology = xmvb::vb::build_same_spin_pair_topology(
      product_alpha, product_beta, n_orbitals);
  const xmvb::vb::StructureAction structure_action(terms, product_dimension,
      topology, packed_overlap, one_electron, two_electron, n_orbitals,
      nullptr, xmvb::vb::StructureActionMode::ActionOnly);
  check_joint_structure_response(structure_action, alpha.size(), beta.size());
  Eigen::MatrixXd dense_hamiltonian(product_dimension, product_dimension);
  const xmvb::vb::DeterminantPairEvaluator evaluator;
  for (int target_alpha = 0;
       target_alpha < static_cast<int>(alpha.size());
       ++target_alpha) {
    for (int target_beta = 0; target_beta < n_beta; ++target_beta) {
      const int target = target_alpha * n_beta + target_beta;
      for (int source_alpha = 0;
           source_alpha < static_cast<int>(alpha.size());
           ++source_alpha) {
        for (int source_beta = 0; source_beta < n_beta; ++source_beta) {
          const int source = source_alpha * n_beta + source_beta;
          dense_hamiltonian(target, source) = evaluator.evaluate(
              alpha[target_alpha],
              alpha[source_alpha],
              beta[target_beta],
              beta[source_beta],
              packed_overlap,
              one_electron,
              n_orbitals,
              two_electron,
              false).total_hamiltonian;
        }
      }
    }
  }
  const Eigen::MatrixXd sigma = action.apply(coefficients);
  check_fused_response(action, coefficients, coefficients);
  for (int block = 0; block < block_width; ++block) {
    Eigen::VectorXd packed_coefficients(product_dimension);
    for (int alpha_index = 0;
         alpha_index < static_cast<int>(alpha.size());
         ++alpha_index) {
      for (int beta_index = 0; beta_index < n_beta; ++beta_index) {
        packed_coefficients[alpha_index * n_beta + beta_index] =
            coefficients(alpha_index, block * n_beta + beta_index);
      }
    }
    const Eigen::VectorXd expected = dense_hamiltonian * packed_coefficients;
    for (int alpha_index = 0;
         alpha_index < static_cast<int>(alpha.size());
         ++alpha_index) {
      for (int beta_index = 0; beta_index < n_beta; ++beta_index) {
        require(
            std::abs(
                sigma(alpha_index, block * n_beta + beta_index) -
                expected[alpha_index * n_beta + beta_index]) < 2.0e-12,
            "distinct-spin direct-CI sigma differs from dense assembly");
      }
    }
  }
}

void check_streamed_generator_adjoint_requests() {
  constexpr int n_orbitals = 4;
  const auto alpha = complete_space(n_orbitals, 1);
  const auto beta = complete_space(n_orbitals, 2);
  const int n_alpha = static_cast<int>(alpha.size());
  const int n_beta = static_cast<int>(beta.size());
  constexpr int first = 2;
  constexpr int count = 3;
  xmvb::vb::OrthogonalActiveIntegrals integrals;
  integrals.one_electron = Eigen::MatrixXd::Zero(n_orbitals, n_orbitals);
  const int n_pairs = xmvb::integrals::packed_active_pair_count(n_orbitals);
  integrals.pair_kernel = Eigen::MatrixXd::Zero(n_pairs, n_pairs);
  const xmvb::mcscf::DirectCiSigmaAction action(alpha, beta, integrals);

  Eigen::MatrixXd left(n_alpha, count), right_a(n_alpha, n_beta),
      right_b(n_alpha, n_beta);
  for (int b = 0; b < n_beta; ++b)
    for (int a = 0; a < n_alpha; ++a) {
      right_a(a, b) = (b == n_beta - 1 || (a + b) % 3 == 0)
          ? 0.0 : std::sin(0.31 * (a + 1) * (b + 2));
      right_b(a, b) = std::cos(0.19 * (a + 2) * (b + 1));
    }
  for (int b = 0; b < count; ++b)
    for (int a = 0; a < n_alpha; ++a)
      left(a, b) = (a == 1 && b == 1) ? 0.0 :
          std::sin(0.27 * (a + 2) * (b + 1));

  // Both readers borrow this storage. Every tile or column request overwrites
  // it, including requests belonging to the other right argument.
  std::vector<double> borrowed_values(n_alpha);
  std::vector<int> borrowed_rows;
  const auto request = [&](const Eigen::MatrixXd& right, double scale,
                           bool transpose, bool compact)
      -> xmvb::mcscf::DirectCiStreamedGeneratorRequest {
    const Eigen::MatrixXd* carrier = &right;
    xmvb::mcscf::DirectCiTileReader read =
        [&, carrier](const std::vector<int>& betas,
                     const std::vector<int>& rows,
                     Eigen::Ref<Eigen::MatrixXd> values,xmvb::mcscf::DirectCiRead mode) {
          borrowed_values.assign(n_alpha, -97.0);
          borrowed_rows.clear();
          for (int b = 0; b < static_cast<int>(betas.size()); ++b)
            for (int a = 0; a < static_cast<int>(rows.size()); ++a)
              values(a, b) = (*carrier)(rows[a], betas[b]);
        };
    xmvb::mcscf::DirectCiColumnReader column =
        [&, carrier, compact](int beta_column)
            -> xmvb::mcscf::DirectCiColumnView {
          borrowed_values.assign(n_alpha, -97.0);
          borrowed_rows.clear();
          if (compact) {
            int used = 0;
            for (int a = 0; a < n_alpha; ++a)
              if ((*carrier)(a, beta_column) != 0.0) {
                borrowed_rows.push_back(a);
                borrowed_values[used++] = (*carrier)(a, beta_column);
              }
            return {borrowed_values.data(), borrowed_rows.data(), used};
          }
          for (int a = 0; a < n_alpha; ++a)
            borrowed_values[a] = (*carrier)(a, beta_column);
          return {borrowed_values.data(), nullptr, n_alpha};
        };
    return {read, column, scale, transpose, false};
  };

  constexpr double scale_a = -0.37;
  constexpr double scale_b = 0.61;
  Eigen::MatrixXd padded_left = Eigen::MatrixXd::Zero(n_alpha, n_beta);
  padded_left.middleCols(first, count) = left;
  const Eigen::MatrixXd expected_full =
      scale_a * action.one_body_generator_adjoint(padded_left, right_a) +
      scale_b * action.one_body_generator_adjoint(padded_left, right_b).transpose();
  Eigen::MatrixXd expected_upper = expected_full;
  expected_upper.triangularView<Eigen::StrictlyLower>().setZero();
  const Eigen::MatrixXd columns_upper = action.generator_adjoint_columns(
      first, count, left,
      {{right_a, scale_a, false}, {right_b, scale_b, true}}, true);
  require((columns_upper - expected_upper).norm() < 2e-12,
          "column-panel generator reference differs from the full adjoint");
  const auto streamed = [&](const Eigen::MatrixXd& panel, bool symmetric_sum) {
    return action.generator_adjoint_streamed(first, panel,
        {request(right_a, scale_a, false, true),
         request(right_b, scale_b, true, false)}, symmetric_sum);
  };
  require((streamed(left, false) - expected_upper).norm() < 2e-12,
          "fused streamed generator differs from oriented upper adjoints");
  Eigen::MatrixXd expected_symmetric =
      (expected_full + expected_full.transpose()).eval();
  expected_symmetric.triangularView<Eigen::StrictlyLower>().setZero();
  require((streamed(left, true) - expected_symmetric).norm() < 2e-12,
          "fused streamed generator symmetric sum differs from full adjoints");
  const Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(n_alpha, count);
  require(streamed(zero, false).cwiseAbs().maxCoeff() == 0.0 &&
              streamed(zero, true).cwiseAbs().maxCoeff() == 0.0,
          "zero streamed generator panel produced a nonzero adjoint");

  // Retained columns own fixed storage for the entire contraction. The first
  // right argument uses sparse sorted rows; the second borrows dense Eigen
  // columns directly. Neither reader changes a returned column or row list.
  std::vector<std::vector<double>> retained_values(n_beta);
  std::vector<std::vector<int>> retained_rows(n_beta);
  for (int b = 0; b < n_beta; ++b)
    for (int a = 0; a < n_alpha; ++a)
      if (right_a(a, b) != 0.0) {
        retained_rows[b].push_back(a);
        retained_values[b].push_back(right_a(a, b));
      }
  const auto retained_request = [&](const Eigen::MatrixXd& right, double scale,
                                    bool transpose, bool compact)
      -> xmvb::mcscf::DirectCiStreamedGeneratorRequest {
    const Eigen::MatrixXd* carrier = &right;
    xmvb::mcscf::DirectCiTileReader read =
        [carrier](const std::vector<int>& betas,
                  const std::vector<int>& rows,
                  Eigen::Ref<Eigen::MatrixXd> values,xmvb::mcscf::DirectCiRead mode) {
          for (int b = 0; b < static_cast<int>(betas.size()); ++b)
            for (int a = 0; a < static_cast<int>(rows.size()); ++a)
              values(a, b) = (*carrier)(rows[a], betas[b]);
        };
    xmvb::mcscf::DirectCiColumnReader column =
        [&, carrier, compact](int b) -> xmvb::mcscf::DirectCiColumnView {
          if (compact)
            return {retained_values[b].data(), retained_rows[b].data(),
                    static_cast<int>(retained_values[b].size())};
          return {carrier->col(b).data(), nullptr, n_alpha};
        };
    return {read, column, scale, transpose, true};
  };
  const auto retained = [&](bool symmetric_sum) {
    return action.generator_adjoint_streamed(first, left,
        {retained_request(right_a, scale_a, false, true),
         retained_request(right_b, scale_b, true, false)}, symmetric_sum);
  };
  require((retained(false) - expected_upper).norm() < 2e-12 &&
              (retained(true) - expected_symmetric).norm() < 2e-12,
          "retained streamed columns differ from oriented generator adjoints");
  Eigen::MatrixXd other = 0.37 * left;
  for (int b = 0; b < count; ++b)
    for (int a = 0; a < n_alpha; ++a)
      other(a, b) += 0.21 * std::cos(0.23 * (a + 1) * (b + 3));
  const Eigen::MatrixXd zero_left = Eigen::MatrixXd::Zero(n_alpha, count);
  const std::array<Eigen::Ref<const Eigen::MatrixXd>, 3> lefts{left, other, zero_left};
  const auto check_multiple = [&](bool retained_columns, bool symmetric_sum) {
    const auto images = retained_columns
        ? action.generator_adjoint_streamed(first,
            std::span<const Eigen::Ref<const Eigen::MatrixXd>>(lefts),
            {retained_request(right_a, scale_a, false, true),
             retained_request(right_b, scale_b, true, false)}, symmetric_sum)
        : action.generator_adjoint_streamed(first,
            std::span<const Eigen::Ref<const Eigen::MatrixXd>>(lefts),
            {request(right_a, scale_a, false, true),
             request(right_b, scale_b, true, false)}, symmetric_sum);
    require(images.size() == lefts.size(),
            "multi-left streamed generator returned the wrong number of cotangents");
    for (std::size_t i = 0; i < lefts.size(); ++i) {
      Eigen::MatrixXd padded = Eigen::MatrixXd::Zero(n_alpha, n_beta);
      padded.middleCols(first, count) = lefts[i];
      Eigen::MatrixXd expected =
          scale_a * action.one_body_generator_adjoint(padded, right_a) +
          scale_b * action.one_body_generator_adjoint(padded, right_b).transpose();
      if (symmetric_sum) expected = (expected + expected.transpose()).eval();
      expected.triangularView<Eigen::StrictlyLower>().setZero();
      require((images[i] - expected).norm() < 2e-12,
              "multi-left streamed generator differs from independent dense adjoints");
    }
    require(images.back().isZero(),
            "zero left in a multi-left streamed generator produced a cotangent");
  };
  for (const bool retained_columns : {false, true})
    for (const bool symmetric_sum : {false, true})
      check_multiple(retained_columns, symmetric_sum);
#ifdef _OPENMP
  Eigen::MatrixXd nested, nested_retained, nested_retained_symmetric;
#pragma omp parallel num_threads(2)
  {
#pragma omp single
    {
      nested = streamed(left, true);
      nested_retained = retained(false);
      nested_retained_symmetric = retained(true);
    }
  }
  require((nested - expected_symmetric).norm() < 2e-12,
          "streamed generator differs inside an existing OpenMP team");
  require((nested_retained - expected_upper).norm() < 2e-12 &&
              (nested_retained_symmetric - expected_symmetric).norm() < 2e-12,
          "retained streamed columns differ inside an existing OpenMP team");
#endif
}

void check_generated_generator_panels() {
  constexpr int n=4;
  const auto alpha=complete_space(n,2);
  xmvb::vb::OrthogonalActiveIntegrals integrals;
  integrals.one_electron=Eigen::MatrixXd::Zero(n,n);
  integrals.pair_kernel=Eigen::MatrixXd::Zero(n*(n+1)/2,n*(n+1)/2);
  for (const int n_beta_electrons:{1,2}) {
    const auto beta=complete_space(n,n_beta_electrons);
    const int na=alpha.size(),nb=beta.size();
    const xmvb::mcscf::DirectCiSigmaAction action(alpha,beta,integrals);
    const xmvb::mcscf::DirectCiSigmaAction alpha_action(alpha,complete_space(n,0),integrals);
    Eigen::MatrixXd carrier(na,nb),generator(n,n);
    for (int b=0;b<nb;++b)
      for (int a=0;a<na;++a)
        carrier(a,b)=(a+b)%3==0 ? 0.0:std::sin(0.27*(a+2)*(b+1));
    for (int q=0;q<n;++q)
      for (int p=0;p<n;++p) generator(p,q)=std::cos(0.13*(p+1)*(q+3));
    std::vector<std::vector<int>> rows(nb);
    std::vector<std::vector<double>> values(nb);
    for (int b=0;b<nb;++b)
      for (int a=0;a<na;++a)
        if (carrier(a,b)!=0) { rows[b].push_back(a); values[b].push_back(carrier(a,b)); }
    for (const bool compact:{false,true}) {
      const xmvb::mcscf::DirectCiColumnReader column=[&](int b) {
        return compact ? xmvb::mcscf::DirectCiColumnView{values[b].data(),rows[b].data(),
                static_cast<int>(rows[b].size())}:
            xmvb::mcscf::DirectCiColumnView{carrier.col(b).data(),nullptr,na};
      };
      for (const bool alpha_only:{false,true}) {
        const Eigen::MatrixXd expected=(alpha_only ? alpha_action:action)
            .apply_one_body_generator(carrier,generator);
        Eigen::MatrixXd storage=Eigen::MatrixXd::Constant(na+3,nb,-97.0);
        for (int first=0;first<nb;first+=2)
          action.apply_one_body_generator_columns(first,column,generator,
              storage.topRows(na).middleCols(first,std::min(2,nb-first)),alpha_only);
        require((storage.topRows(na)-expected).norm()<2e-12 &&
                (storage.bottomRows(3).array()==-97.0).all(),
            "generated one-body panel differs from the independent full-carrier action");
      }
    }
  }
}

void check_exchange_reduced_generator() {
  constexpr int n=4;
  const auto spin=complete_space(n,2);
  const int u=spin.size(),pairs=n*(n+1)/2;
  xmvb::vb::OrthogonalActiveIntegrals integrals;
  integrals.one_electron=Eigen::MatrixXd::Zero(n,n);
  integrals.pair_kernel=Eigen::MatrixXd::Zero(pairs,pairs);
  const xmvb::mcscf::DirectCiSigmaAction action(spin,spin,integrals);
  for (const int parity:{-1,1}) {
    Eigen::MatrixXd left(u,u),right_a(u,u),right_b(u,u);
    for (int b=0;b<u;++b)
      for (int a=0;a<=b;++a) {
        left(a,b)=std::sin(0.27*(a+2)*(b+1));
        right_a(a,b)=std::cos(0.39*(a+1)*(b+3));
        right_b(a,b)=(a+b)%3==0 ? 0.0:std::sin(0.13*(a+3)*(b+2));
        left(b,a)=parity*left(a,b);
        right_a(b,a)=parity*right_a(a,b);
        right_b(b,a)=parity*right_b(a,b);
        if (parity<0 && a==b) left(a,b)=right_a(a,b)=right_b(a,b)=0.0;
      }
    const Eigen::MatrixXd other = (0.37 * left + 0.19 * right_a).eval();
    const auto request=[&](const Eigen::MatrixXd& right,double scale,bool transpose) {
      const auto* carrier=&right;
      xmvb::mcscf::DirectCiTileReader read=[carrier](const std::vector<int>& betas,
          const std::vector<int>& rows,Eigen::Ref<Eigen::MatrixXd> values,xmvb::mcscf::DirectCiRead mode) {
        for (int b=0;b<static_cast<int>(betas.size());++b)
          for (int a=0;a<static_cast<int>(rows.size());++a)
            values(a,b)=(*carrier)(rows[a],betas[b]);
      };
      xmvb::mcscf::DirectCiColumnReader column=[](int) -> xmvb::mcscf::DirectCiColumnView {
        throw std::runtime_error("exchange reduction must not gather beta spectators");
      };
      return xmvb::mcscf::DirectCiStreamedGeneratorRequest{read,column,-scale,transpose,false};
    };
    Eigen::MatrixXd reference=-0.37*action.one_body_generator_adjoint(left,right_a)+
        0.61*action.one_body_generator_adjoint(left,right_b).transpose();
    for (const bool symmetric:{false,true}) {
      Eigen::MatrixXd expected=reference;
      if (symmetric) expected=(reference+reference.transpose()).eval();
      expected.triangularView<Eigen::StrictlyLower>().setZero();
      Eigen::MatrixXd result=Eigen::MatrixXd::Zero(n,n);
      std::array<Eigen::MatrixXd,3> multi_result{
          Eigen::MatrixXd::Zero(n,n), Eigen::MatrixXd::Zero(n,n),
          Eigen::MatrixXd::Zero(n,n)};
      // Incomplete panels intentionally have no exchange parity themselves.
      for (int first=0;first<u;first+=2) {
        const Eigen::MatrixXd panel=left.middleCols(first,std::min(2,u-first));
        result+=action.generator_adjoint_streamed(first,panel,
            {request(right_a,0.37,false),request(right_b,-0.61,true)},symmetric,true);
        const Eigen::MatrixXd other_panel=other.middleCols(first,std::min(2,u-first));
        const Eigen::MatrixXd zero_panel=Eigen::MatrixXd::Zero(u,panel.cols());
        const std::array<Eigen::Ref<const Eigen::MatrixXd>,3> lefts{
            panel,other_panel,zero_panel};
        const auto images=action.generator_adjoint_streamed(first,
            std::span<const Eigen::Ref<const Eigen::MatrixXd>>(lefts),
            {request(right_a,0.37,false),request(right_b,-0.61,true)},symmetric,true);
        require(images.size()==lefts.size(),
            "exchange-reduced multi-left generator returned the wrong number of cotangents");
        for (std::size_t i=0;i<images.size();++i) multi_result[i]+=images[i];
      }
      require((result-expected).norm()<3e-12,
          "exchange-reduced panel sum differs from independent two-spin adjoint");
      for (std::size_t i=0;i<multi_result.size();++i) {
        Eigen::MatrixXd full_left=Eigen::MatrixXd::Zero(u,u);
        if (i==0) full_left=left;
        if (i==1) full_left=other;
        Eigen::MatrixXd independent=
            -0.37*action.one_body_generator_adjoint(full_left,right_a)+
            0.61*action.one_body_generator_adjoint(full_left,right_b).transpose();
        if (symmetric) independent=(independent+independent.transpose()).eval();
        independent.triangularView<Eigen::StrictlyLower>().setZero();
        require((multi_result[i]-independent).norm()<3e-12,
            "exchange-reduced multi-left panels differ from independent two-spin adjoints");
      }
      require(multi_result.back().isZero(),
          "zero exchange-reduced multi-left panel produced a cotangent");
    }
  }
}

void check_demand_driven_structure_diagonal() {
  // Put one demanded pair in the first cell of a unique-spin space larger
  // than the forward matrix tile. A tile-backed diagonal lookup would evaluate
  // the entire 256x256 tile even though all other determinants have zero
  // structure coefficient.
  constexpr int n_orbitals = 11;
  constexpr int n_unique_alpha = 257;
  auto alpha = complete_space(n_orbitals, 5);
  alpha.resize(n_unique_alpha);
  std::vector<std::vector<int>> beta(n_unique_alpha, std::vector<int>{0});
  std::vector<std::vector<xmvb::vb::StructureExpansionTerm>> expansion(
      n_unique_alpha);
  expansion.front().push_back({0, 1.0});

  const Eigen::MatrixXd overlap =
      Eigen::MatrixXd::Identity(n_orbitals, n_orbitals);
  const std::vector<double> packed_overlap(
      overlap.data(),
      overlap.data() + overlap.size());
  Eigen::MatrixXd one_electron(n_orbitals, n_orbitals);
  for (int column = 0; column < n_orbitals; ++column) {
    for (int row = 0; row <= column; ++row) {
      const double value =
          0.03 * std::cos(static_cast<double>((row + 1) * (column + 2)));
      one_electron(row, column) = value;
      one_electron(column, row) = value;
    }
  }
  xmvb::integrals::ActiveSpaceTwoElectronResult two_electron;
  two_electron.packed_active_two_electron_integrals.assign(
      xmvb::integrals::packed_active_two_electron_integral_count(n_orbitals),
      0.0);

  const auto large_topology = xmvb::vb::build_same_spin_pair_topology(
      alpha,
      beta,
      n_orbitals);
  require(!large_topology.enabled(),
          "demand-driven diagonal regression unexpectedly built pair caches");
  require(large_topology.alpha_reuse_table.unique_determinants.size() ==
              n_unique_alpha,
          "demand-driven diagonal regression did not span multiple tiles");

  const xmvb::vb::FullDeterminantStructureHamiltonianOverlapBuilder builder;
  const xmvb::vb::StructureDiagonal large_diagonal = builder.build_diagonal(
      alpha,
      beta,
      expansion,
      packed_overlap,
      one_electron,
      n_orbitals,
      two_electron,
      1,
      large_topology);

  const std::vector<std::vector<int>> reference_alpha{alpha.front()};
  const std::vector<std::vector<int>> reference_beta{beta.front()};
  const std::vector<std::vector<xmvb::vb::StructureExpansionTerm>>
      reference_expansion{{{0, 1.0}}};
  const auto reference_topology = xmvb::vb::build_same_spin_pair_topology(
      reference_alpha,
      reference_beta,
      n_orbitals);
  const xmvb::vb::StructureDiagonal reference_diagonal =
      builder.build_diagonal(
          reference_alpha,
          reference_beta,
          reference_expansion,
          packed_overlap,
          one_electron,
          n_orbitals,
          two_electron,
          1,
          reference_topology);

  require(
      std::abs(
          large_diagonal.hamiltonian[0] -
          reference_diagonal.hamiltonian[0]) < 2.0e-12,
      "unused unique determinants changed the demand-driven Hamiltonian diagonal");
  require(
      std::abs(
          large_diagonal.overlap[0] -
          reference_diagonal.overlap[0]) < 2.0e-12,
      "unused unique determinants changed the demand-driven overlap diagonal");

  std::vector<std::vector<xmvb::vb::StructureExpansionTerm>>
      jacobi_expansion(n_unique_alpha);
  double expected_jacobi_hamiltonian = 0.0;
  double expected_jacobi_overlap = 0.0;
  const xmvb::vb::DeterminantPairEvaluator evaluator;
  for (int determinant = 0; determinant < n_unique_alpha; ++determinant) {
    const double coefficient =
        std::sin(0.013 * static_cast<double>(determinant + 1));
    if (determinant == 0) {
      // Duplicate determinant/structure terms must be combined before the
      // coefficient is squared.
      jacobi_expansion[determinant].push_back({0, 0.4 * coefficient});
      jacobi_expansion[determinant].push_back({0, 0.6 * coefficient});
    } else {
      jacobi_expansion[determinant].push_back({0, coefficient});
    }
    const auto self_pair = evaluator.evaluate(
        alpha[determinant],
        alpha[determinant],
        beta[determinant],
        beta[determinant],
        packed_overlap,
        one_electron,
        n_orbitals,
        two_electron,
        false);
    expected_jacobi_hamiltonian +=
        coefficient * coefficient * self_pair.total_hamiltonian;
    expected_jacobi_overlap +=
        coefficient * coefficient * self_pair.overlap_determinant;
  }
  const xmvb::vb::StructureDiagonal jacobi =
      builder.build_davidson_jacobi_preconditioner(
          alpha,
          beta,
          jacobi_expansion,
          packed_overlap,
          one_electron,
          n_orbitals,
          two_electron,
          1,
          large_topology);
  require(
      std::abs(jacobi.hamiltonian[0] - expected_jacobi_hamiltonian) < 2.0e-10,
      "large determinant-diagonal Hamiltonian Jacobi model is inexact");
  require(
      std::abs(jacobi.overlap[0] - expected_jacobi_overlap) < 2.0e-10,
      "large determinant-diagonal overlap Jacobi model is inexact");

  const std::vector<std::vector<xmvb::vb::StructureExpansionTerm>>
      cancelling_expansion{{{0, 1.0}, {0, -1.0}}};
  bool rejected_zero_overlap = false;
  try {
    static_cast<void>(builder.build_davidson_jacobi_preconditioner(
        reference_alpha,
        reference_beta,
        cancelling_expansion,
        packed_overlap,
        one_electron,
        n_orbitals,
        two_electron,
        1,
        reference_topology));
  } catch (const std::domain_error&) {
    rejected_zero_overlap = true;
  }
  require(
      rejected_zero_overlap,
      "zero Davidson Jacobi overlap was not rejected explicitly");
}

/** @brief Distinct spin sectors, including equal sizes but different occupations. */
void check_fused_spin_sectors() {
  for (const auto& sector : {std::array<int, 3>{11, 5, 6}, {10, 4, 5}, {10, 5, 4},
                            {11, 1, 5}, {11, 5, 1}, {10, 5, 0}, {10, 0, 5}}) {
    const int n_orbitals = sector[0];
    const auto alpha = complete_space(n_orbitals, sector[1]);
    const auto beta = complete_space(n_orbitals, sector[2]);
    const int n_pairs = xmvb::integrals::packed_active_pair_count(n_orbitals);
    xmvb::vb::OrthogonalActiveIntegrals integrals;
    integrals.one_electron = Eigen::MatrixXd::Identity(n_orbitals, n_orbitals);
    Eigen::MatrixXd factors(5, n_pairs);
    for (int pair = 0; pair < n_pairs; ++pair)
      for (int row = 0; row < factors.rows(); ++row)
        factors(row, pair) = 0.04 * std::cos(0.13 * (row + 1) * (pair + 2));
    integrals.pair_kernel = factors.transpose() * factors;
    const xmvb::mcscf::DirectCiSigmaAction action(alpha, beta, integrals);
    Eigen::MatrixXd response(alpha.size(), 2 * beta.size());
    Eigen::MatrixXd state(response.rows(), response.cols());
    for (int column = 0; column < response.cols(); ++column)
      for (int row = 0; row < response.rows(); ++row) {
        response(row, column) = std::sin(0.07 * (row + 2) * (column + 3));
        state(row, column) = std::cos(0.03 * (row + 3) * (column + 1));
      }
    check_fused_response(action, response, state);
  }
}

/** @brief Same-binary joint/separate response comparison, not an optimizer replay. */
void benchmark_structure_response() {
  using Clock = std::chrono::steady_clock;
  std::cout << "kernel,n_orbitals,n_alpha,n_beta,calls,wall_seconds,checksum\n"
            << std::setprecision(17);
  for (const auto sector : {std::array<int,3>{6,3,3}, {11,5,5}, {11,5,6}, {12,6,6}}) {
    const int n = sector[0];
    const auto alpha = complete_space(n, sector[1]), beta = complete_space(n, sector[2]);
    const int products = alpha.size()*beta.size();
    std::vector<std::vector<int>> product_alpha, product_beta;
    std::vector<std::vector<xmvb::vb::StructureExpansionTerm>> terms(products);
    product_alpha.reserve(products);
    product_beta.reserve(products);
    for (const auto& a : alpha)
      for (const auto& b : beta) {
        const int k = product_alpha.size();
        product_alpha.push_back(a);
        product_beta.push_back(b);
        for (int j = 0; j < 3; ++j)
          terms[k].push_back({j, std::sin(0.17*(k+1)*(j+2))/std::sqrt(products)});
      }
    const auto topology = xmvb::vb::build_same_spin_pair_topology(product_alpha, product_beta, n);
    Eigen::MatrixXd r = Eigen::MatrixXd::Identity(n,n);
    for (int i = 0; i < n; ++i)
      for (int j = i+1; j < n; ++j) r(i,j) = 0.02*std::sin((i+1)*(j+2));
    const Eigen::MatrixXd overlap = r.transpose()*r;
    const std::vector<double> packed_overlap(overlap.data(), overlap.data()+overlap.size());
    const Eigen::MatrixXd h = -Eigen::MatrixXd::Identity(n,n);
    const int pairs = xmvb::integrals::packed_active_pair_count(n);
    xmvb::integrals::ActiveSpaceTwoElectronResult eri;
    eri.packed_active_two_electron_integrals.resize(
        xmvb::integrals::packed_active_two_electron_integral_count(n));
    for (int j = 0; j < pairs; ++j)
      for (int i = 0; i <= j; ++i)
        eri.packed_active_two_electron_integrals[
            xmvb::core::PairIndex::packed_pair_of_pairs_index(i,j)] =
            0.03*std::cos(0.07*(i+1)*(j+2));
    const xmvb::vb::StructureAction action(terms, 3, topology, packed_overlap,
        h, eri, n, nullptr, xmvb::vb::StructureActionMode::ActionOnly);
    xmvb::vb::SelectedStateDeterminantMatrices states;
    states.n_structures = 3;
    states.n_unique_alpha = alpha.size();
    states.n_unique_beta = beta.size();
    states.normalized_state_weights = {1.0};
    states.states.resize(1);
    states.states[0].normalized_state_weight = 1.0;
    const Eigen::Vector3d c(0.3,-0.2,0.7), q(0.4,0.6,-0.3);
    states.states[0].coefficient_matrix = action.expand_structure_block(c);
    const auto accepted = action.prepare_active_adjoint(c, {1.0}, {-1.3});
    const auto evaluate = [&](bool joint) {
      xmvb::vb::StructureActionResult images;
      xmvb::vb::StructureResponseSweep sweep;
      Eigen::MatrixXd metric;
      if (joint) {
        sweep = action.apply_response(accepted,q);
        images = std::move(sweep.images);
        metric = images.overlap;
      } else {
        metric = action.apply_overlap(q);
        images = action.apply(q);
      }
      Eigen::MatrixXd multiplier(1,1);
      multiplier(0,0) = -c.dot(images.hamiltonian.col(0)+1.3*images.overlap.col(0));
      xmvb::vb::StructureActiveIntegralAdjoint adjoint;
      if (joint) adjoint = action.finish_response(accepted,std::move(sweep),multiplier);
      else {
        adjoint = action.active_integral_response_adjoint(accepted,q,multiplier);
      }
      return metric.squaredNorm()+images.hamiltonian.squaredNorm()+
          adjoint.overlap.squaredNorm()+adjoint.one_electron.squaredNorm()+
          adjoint.pair_kernel.squaredNorm();
    };
    const double separate = evaluate(false), joint = evaluate(true);
    require(std::abs(separate-joint) < 3e-11*std::max(1.0,std::abs(separate)),
        "joint response timing checksum differs");
    for (bool fused : {false,true}) {
      constexpr int repeats = 3;
      double checksum = 0;
      const auto start = Clock::now();
      for (int i = 0; i < repeats; ++i) checksum += evaluate(fused);
      std::cout << (fused ? "joint," : "separate,") << n << ',' << alpha.size()
                << ',' << beta.size() << ',' << repeats << ','
                << std::chrono::duration<double>(Clock::now()-start).count()
                << ',' << checksum << '\n';
    }
  }
}

/** @brief Complete singlet/triplet sectors are closed without spanning all M_S products. */
void check_spin_sector_carrier_inverse() {
  const auto spin=complete_space(3,1);
  Eigen::Matrix3d r=Eigen::Matrix3d::Identity();
  r(0,1)=0.17; r(0,2)=-0.09; r(1,2)=0.12;
  const Eigen::Matrix3d s=r.transpose()*r;
  const std::vector<double> overlap(s.data(),s.data()+s.size());
  const Eigen::Matrix3d h=Eigen::Vector3d(-0.8,-0.3,0.2).asDiagonal();
  xmvb::integrals::ActiveSpaceTwoElectronResult eri;
  eri.packed_active_two_electron_integrals.resize(xmvb::integrals::packed_active_two_electron_integral_count(3),0.0);
  const auto orth=xmvb::vb::orthogonalize_active_integrals(overlap,h,eri,3);
  const xmvb::mcscf::DirectCiSigmaAction sigma(spin,spin,orth);
  const Eigen::MatrixXd diagonal=sigma.diagonal(orth.one_electron);
  std::vector<std::vector<int>> pa,pb;
  for (const auto& a:spin) for (const auto& b:spin) { pa.push_back(a); pb.push_back(b); }
  const auto topology=xmvb::vb::build_same_spin_pair_topology(pa,pb,3);
  for (int parity:{1,-1}) {
    const int n=parity==1 ? 6:3;
    std::vector<std::vector<xmvb::vb::StructureExpansionTerm>> terms(9);
    int column=0;
    for (int a=0;a<3;++a) for (int b=a+(parity<0);b<3;++b) {
      const double scale=a==b ? 1.0:1/std::sqrt(2.0);
      terms[3*a+b].push_back({column,scale});
      if (a!=b) terms[3*b+a].push_back({column,parity*scale});
      ++column;
    }
    const xmvb::vb::StructureAction action(terms,n,topology,overlap,h,eri,3,
        nullptr,xmvb::vb::StructureActionMode::ActionOnly);
    const xmvb::vb::StructureMetricInverse metric(action,spin,spin,true,65536,65536);
    const auto carrier=action.orthogonalize_structure_block(Eigen::MatrixXd::Identity(n,n));
    Eigen::MatrixXd f(9,n);
    for (int j=0;j<n;++j) for (int a=0;a<3;++a) for (int b=0;b<3;++b)
      f(3*a+b,j)=carrier(a,3*j+b);
    const Eigen::MatrixXd inverse=(f.transpose()*f).inverse();
    Eigen::VectorXd dinv(9);
    constexpr double energy=-3.0,shift=0.11;
    for (int a=0;a<3;++a) for (int b=0;b<3;++b)
      dinv[3*a+b]=1/(diagonal(a,b)-energy-shift);
    const Eigen::MatrixXd reference=inverse*f.transpose()*dinv.asDiagonal()*f*inverse;
    const auto actual=metric.apply_carrier_inverse(Eigen::MatrixXd::Identity(n,n),
        Eigen::VectorXd::Constant(n,energy),shift);
    require((actual-reference).norm()<1e-11,
        "complete spin-sector cancellation changed the carrier preconditioner");
    const Eigen::MatrixXd j=metric.to_ci_coordinates(Eigen::MatrixXd::Identity(n,n));
    const Eigen::MatrixXd ji=metric.from_ci_coordinates(Eigen::MatrixXd::Identity(n,n));
    require((j.transpose()*j-f.transpose()*f).norm()<2e-12 &&
                (j*ji-Eigen::MatrixXd::Identity(n,n)).norm()<2e-12,
            "pure-spin CI coordinates are not an isometry");
    const auto image=action.apply(ji);
    const auto ci_h=metric.apply_ci_hamiltonian(Eigen::MatrixXd::Identity(n,n));
    require((ci_h-ji.transpose()*image.hamiltonian).norm()<2e-12,
            "complete pure-spin CI action changed the restricted Hamiltonian");
  }
}

/** @brief The streamed directional image and integral adjoint have separate oracles. */
void check_tiled_integral_direction_case(
    const std::vector<std::vector<int>>& alpha,
    const std::vector<std::vector<int>>& beta, int width, int parity,
    const xmvb::mcscf::ProductSupport* support = nullptr,
    bool zero_direction = false, bool zero_state = false,
    bool compact_output = false) {
  constexpr int n_orbitals = 4;
  const int n_alpha = static_cast<int>(alpha.size());
  const int n_beta = static_cast<int>(beta.size());
  const int n_pairs = xmvb::integrals::packed_active_pair_count(n_orbitals);
  xmvb::mcscf::ActiveIntegrals integrals, direction_integrals;
  integrals.one_electron.resize(n_orbitals, n_orbitals);
  direction_integrals.one_electron.resize(n_orbitals, n_orbitals);
  integrals.pair_kernel.resize(n_pairs, n_pairs);
  direction_integrals.pair_kernel.resize(n_pairs, n_pairs);
  for (int q = 0; q < n_orbitals; ++q)
    for (int p = 0; p <= q; ++p) {
      const double h = 0.16 * std::cos(0.31 * (p + 1) * (q + 2)) +
                       (p == q ? 0.23 : 0.0);
      const double dh = zero_direction ? 0.0 :
          0.07 * std::sin(0.27 * (p + 2) * (q + 1));
      integrals.one_electron(p, q) = integrals.one_electron(q, p) = h;
      direction_integrals.one_electron(p, q) =
          direction_integrals.one_electron(q, p) = dh;
    }
  for (int q = 0; q < n_pairs; ++q)
    for (int p = 0; p <= q; ++p) {
      const double kernel = 0.035 * std::cos(0.17 * (p + 1) * (q + 3));
      const double dkernel = zero_direction ? 0.0 :
          0.021 * std::sin(0.19 * (p + 2) * (q + 1));
      integrals.pair_kernel(p, q) = integrals.pair_kernel(q, p) = kernel;
      direction_integrals.pair_kernel(p, q) =
          direction_integrals.pair_kernel(q, p) = dkernel;
    }
  const xmvb::mcscf::DirectCiSigmaAction action(alpha, beta, integrals);
  const xmvb::mcscf::DirectCiSigmaAction direction(action, direction_integrals);

  Eigen::MatrixXd response(n_alpha, width * n_beta);
  Eigen::MatrixXd state(n_alpha, width * n_beta);
  const auto fill = [&](int vector, Eigen::MatrixXd* values, bool cosine) {
    Eigen::MatrixXd block(n_alpha, n_beta);
    for (int b = 0; b < n_beta; ++b)
      for (int a = 0; a < n_alpha; ++a)
        block(a, b) = cosine
            ? std::cos(0.23 * (a + 2) * (b + 1) + 0.17 * vector)
            : std::sin(0.29 * (a + 1) * (b + 3) + 0.11 * vector);
    if (parity != 0)
      block = (block + parity * block.transpose()).eval();
    values->middleCols(vector * n_beta, n_beta) = block;
  };
  fill(0, &response, false);                         // Y
  if (width == 2) fill(1, &response, true);          // Z
  if (width == 1) fill(0, &state, true);             // Z
  else {
    state.middleCols(0, n_beta).setZero();           // 0
    state.middleCols(n_beta, n_beta) = response.leftCols(n_beta); // Y
  }
  if (zero_state) state.setZero();
  if (support)
    for (int b = 0; b < n_beta; ++b)
      for (int a = 0; a < n_alpha; ++a)
        if (!support->contains(a, b))
          for (int vector = 0; vector < width; ++vector) {
            response(a, vector * n_beta + b) = 0.0;
            state(a, vector * n_beta + b) = 0.0;
          }

  const auto reader = [width, n_beta](const Eigen::MatrixXd& carrier)
      -> xmvb::mcscf::DirectCiTileReader {
    const Eigen::MatrixXd* source = &carrier;
    return [source, width, n_beta](const std::vector<int>& betas,
        const std::vector<int>& rows, Eigen::Ref<Eigen::MatrixXd> tile,xmvb::mcscf::DirectCiRead mode) {
      for (int b = 0; b < static_cast<int>(betas.size()); ++b)
        for (int vector = 0; vector < width; ++vector)
          for (int a = 0; a < static_cast<int>(rows.size()); ++a)
            tile(a, b * width + vector) =
                (*source)(rows[a], vector * n_beta + betas[b]);
    };
  };
  const auto writer = [width, n_beta,compact_output](Eigen::MatrixXd& image)
      -> xmvb::mcscf::DirectCiTileWriter {
    Eigen::MatrixXd* target = &image;
    return [target, width, n_beta,compact_output](const std::vector<int>& betas,
        const std::vector<int>& rows,
        const xmvb::mcscf::DirectCiColumnAccumulator& accumulate) {
      for (int b = 0; b < static_cast<int>(betas.size()); ++b)
        for (int vector = 0; vector < width; ++vector) {
          if (compact_output) {
            // Two disjoint, sorted output subsets with holes. Each receives
            // three halo-range chunks, exercising bounds across sparse rows.
            for (int part=0;part<2;++part) {
              std::vector<int> selected;
              for (int i=part;i<static_cast<int>(rows.size());i+=2) selected.push_back(rows[i]);
              if (selected.empty()) continue;
              Eigen::VectorXd values=Eigen::VectorXd::Zero(selected.size());
              for (int chunk=0;chunk<3;++chunk)
                accumulate(b,vector,rows.size()*chunk/3,rows.size()*(chunk+1)/3,
                    values.data(),selected.data(),selected.size());
              for (int i=0;i<static_cast<int>(selected.size());++i)
                (*target)(selected[i],vector*n_beta+betas[b])+=values[i];
            }
            continue;
          }
          accumulate(b, vector, 0, static_cast<int>(rows.size()),
              target->col(vector * n_beta + betas[b]).data(), nullptr, 0);
        }
    };
  };
  const auto response_reader = reader(response);
  const auto state_reader = reader(state);
  Eigen::MatrixXd combined = Eigen::MatrixXd::Zero(n_alpha, width * n_beta);
  Eigen::MatrixXd forcing = Eigen::MatrixXd::Zero(n_alpha, width * n_beta);
  const auto actual = action.apply_tiled_direction(width, response_reader,
      state_reader, direction, writer(combined), writer(forcing), support, parity);
  const Eigen::MatrixXd expected_forcing = direction.apply(state);
  const Eigen::MatrixXd expected_combined =
      action.apply(response) + expected_forcing;
  const auto expected_adjoint = action.integral_adjoint(response, state, parity);
  const auto project = [width, n_beta, parity](Eigen::MatrixXd image) {
    if (parity != 0)
      for (int vector = 0; vector < width; ++vector) {
        const Eigen::MatrixXd block = image.middleCols(vector * n_beta, n_beta);
        image.middleCols(vector * n_beta, n_beta) =
            0.5 * (block + parity * block.transpose());
      }
    return image;
  };
  const auto close = [](const Eigen::MatrixXd& value,
                        const Eigen::MatrixXd& reference) {
    return (value - reference).norm() <
        3e-11 * std::max(1.0, reference.norm());
  };
  const Eigen::MatrixXd ci_response = 0.37 * response + 0.21 * state;
  const Eigen::MatrixXd adjoint_response = response - 0.29 * ci_response;
  Eigen::MatrixXd ci_image = Eigen::MatrixXd::Zero(n_alpha, width * n_beta);
  Eigen::MatrixXd orbital_image = Eigen::MatrixXd::Zero(n_alpha, width * n_beta);
  Eigen::MatrixXd orbital_forcing = Eigen::MatrixXd::Zero(n_alpha, width * n_beta);
  int state_reads = 0;
  std::array<int, 3> response_reads{};
  const auto counted_reader = [&](const Eigen::MatrixXd& carrier, int& count)
      -> xmvb::mcscf::DirectCiTileReader {
    const auto base = reader(carrier);
    return [base, &count](const std::vector<int>& betas,
                          const std::vector<int>& rows,
                          Eigen::Ref<Eigen::MatrixXd> tile,xmvb::mcscf::DirectCiRead mode) {
      ++count;
      base(betas, rows, tile,mode);
    };
  };
  const std::array<xmvb::mcscf::DirectCiTileRequest, 3> requests{{
      {counted_reader(ci_response, response_reads[0]), writer(ci_image), nullptr, {}},
      {counted_reader(response, response_reads[1]), writer(orbital_image),
       &direction, writer(orbital_forcing)},
      {counted_reader(adjoint_response, response_reads[2]), {}, nullptr, {}}}};
  const auto joint = action.apply_tiled_responses(width, requests,
      counted_reader(state, state_reads), support, parity);
  const auto expected_ci_adjoint = action.integral_adjoint(ci_response, state, parity);
  const auto expected_empty_adjoint =
      action.integral_adjoint(adjoint_response, state, parity);
  require(joint.size() == requests.size(),
          "joint tiled response lost a direction's integral adjoint");
  require(state_reads > 0 && state_reads == response_reads[0] &&
              state_reads == response_reads[1] &&
              state_reads == response_reads[2],
          "joint tiled response repeated or skipped the accepted-state tile read");
  require(close(project(ci_image), action.apply(ci_response)) &&
              close(project(orbital_image), expected_combined) &&
              close(project(orbital_forcing), expected_forcing),
          "joint tiled response changed separate CI, orbital, or forcing images");
  require(close(joint[0].one_electron, expected_ci_adjoint.one_electron) &&
              close(joint[0].pair_kernel, expected_ci_adjoint.pair_kernel) &&
              close(joint[1].one_electron, expected_adjoint.one_electron) &&
              close(joint[1].pair_kernel, expected_adjoint.pair_kernel) &&
              close(joint[2].one_electron, expected_empty_adjoint.one_electron) &&
              close(joint[2].pair_kernel, expected_empty_adjoint.pair_kernel),
          "joint tiled response mixed independent integral adjoints");
  // With the orbital direction last, its forcing may overwrite the accepted
  // slab only after both Grams have consumed it. Exercise that smaller layout.
  ci_image.setZero(); orbital_image.setZero(); orbital_forcing.setZero();
  state_reads=0; response_reads={};
  const auto ordered=action.apply_tiled_responses(width,
      std::span<const xmvb::mcscf::DirectCiTileRequest>(requests).first(2),
      counted_reader(state,state_reads),support,parity);
  require(ordered.size()==2 && state_reads>0 && state_reads==response_reads[0] &&
              state_reads==response_reads[1] &&
              close(project(ci_image),action.apply(ci_response)) &&
              close(project(orbital_image),expected_combined) &&
              close(project(orbital_forcing),expected_forcing) &&
              close(ordered[0].one_electron,expected_ci_adjoint.one_electron) &&
              close(ordered[0].pair_kernel,expected_ci_adjoint.pair_kernel) &&
              close(ordered[1].one_electron,expected_adjoint.one_electron) &&
              close(ordered[1].pair_kernel,expected_adjoint.pair_kernel),
          "ordered temporal reuse changed an independent image or cotangent");
  // The final orbital image consumes both Hamiltonian products, but has no
  // separate forcing consumer. Its scratch may be fused after the Grams.
  const int ordered_state_reads=state_reads;
  ci_image.setZero(); orbital_image.setZero(); orbital_forcing.setConstant(-97.0);
  state_reads=0; response_reads={};
  const std::array<xmvb::mcscf::DirectCiTileRequest,2> no_forcing_requests{{
      {counted_reader(ci_response,response_reads[0]),writer(ci_image),nullptr,{}},
      {counted_reader(response,response_reads[1]),writer(orbital_image),&direction,{}}}};
  const auto no_forcing=action.apply_tiled_responses(width,no_forcing_requests,
      counted_reader(state,state_reads),support,parity);
  require(no_forcing.size()==2 && state_reads==ordered_state_reads &&
              state_reads==response_reads[0] && state_reads==response_reads[1] &&
              close(project(ci_image),action.apply(ci_response)) &&
              close(project(orbital_image),expected_combined) &&
              close(no_forcing[0].one_electron,expected_ci_adjoint.one_electron) &&
              close(no_forcing[0].pair_kernel,expected_ci_adjoint.pair_kernel) &&
              close(no_forcing[1].one_electron,expected_adjoint.one_electron) &&
              close(no_forcing[1].pair_kernel,expected_adjoint.pair_kernel) &&
              (orbital_forcing.array()==-97.0).all(),
          "ordered temporal direction without forcing changed images or adjoints");
  require(close(project(combined), expected_combined),
          "tiled direction changed H*response + deltaH*state");
  require(close(project(forcing), expected_forcing),
          "tiled direction forcing differs from the direction action");
  require(close(actual.one_electron, expected_adjoint.one_electron) &&
              close(actual.pair_kernel, expected_adjoint.pair_kernel),
          "tiled direction changed the unweighted integral adjoint");
  const double derivative =
      actual.one_electron.cwiseProduct(direction_integrals.one_electron).sum() +
      actual.pair_kernel.cwiseProduct(direction_integrals.pair_kernel).sum();
  const double bilinear = response.cwiseProduct(expected_forcing).sum();
  require(std::abs(derivative - bilinear) <
              3e-11 * std::max(1.0, std::abs(bilinear)),
          "tiled direction adjoint has incorrect integral weighting");

  Eigen::MatrixXd combined_only = Eigen::MatrixXd::Zero(n_alpha, width * n_beta);
  const auto without_forcing = action.apply_tiled_direction(width,
      response_reader, state_reader, direction, writer(combined_only), {},
      support, parity);
  require(close(project(combined_only), expected_combined) &&
              close(combined_only, combined) &&
              close(without_forcing.one_electron, expected_adjoint.one_electron) &&
              close(without_forcing.pair_kernel, expected_adjoint.pair_kernel),
          "omitting the forcing writer changed the combined image or adjoint");
  Eigen::MatrixXd forcing_only=Eigen::MatrixXd::Zero(n_alpha,width*n_beta);
  const auto forcing_adjoint=action.apply_tiled_direction(width,response_reader,
      state_reader,direction,{},writer(forcing_only),support,parity);
  const auto adjoint_only=action.apply_tiled_direction(width,response_reader,
      state_reader,direction,{},{},support,parity);
  require(close(project(forcing_only),expected_forcing) &&
              close(forcing_adjoint.one_electron,expected_adjoint.one_electron) &&
              close(forcing_adjoint.pair_kernel,expected_adjoint.pair_kernel) &&
              close(adjoint_only.one_electron,expected_adjoint.one_electron) &&
              close(adjoint_only.pair_kernel,expected_adjoint.pair_kernel),
          "forcing-only or adjoint-only direction changed the physical contraction");
  if (support) {
    double outside = 0.0;
    for (int b = 0; b < n_beta; ++b)
      for (int a = 0; a < n_alpha; ++a)
        if (!support->contains(a, b))
          for (int vector = 0; vector < width; ++vector)
            outside += std::abs(expected_combined(a, vector * n_beta + b));
    require(outside > 1e-8,
            "selected-support direction case did not exercise outside-support images");
  }
}

/** @brief Range restriction must use actual indices, not occupation ordering. */
void check_permuted_tiled_direction() {
  auto alpha=complete_space(4,2),beta=complete_space(4,1);
  std::rotate(alpha.begin(),alpha.begin()+2,alpha.end());
  std::swap(alpha[1],alpha[4]);
  std::reverse(beta.begin(),beta.end());
  const auto masks=[](const auto& strings) {
    std::vector<std::uint64_t> result;
    for (const auto& occupied:strings) {
      std::uint64_t mask=0;
      for (int orbital:occupied) mask|=std::uint64_t{1}<<orbital;
      result.push_back(mask);
    }
    return result;
  };
  const auto a=masks(alpha),b=masks(beta);
  const int u=alpha.size();
  const xmvb::mcscf::ProductSupport equal(4,a,a,{u-1,(u-1)*u,2*u+1,u+2});
  const xmvb::mcscf::ProductSupport unequal(4,a,b,{0,static_cast<int>(beta.size())+2});
  for (int parity:{0,1,-1})
    check_tiled_integral_direction_case(alpha,alpha,2,parity,&equal,false,false,true);
  check_tiled_integral_direction_case(alpha,beta,2,0,&unequal,false,false,true);
}

/** @brief The complete 8-orbital halo holds one full Gram tile per worker. */
void check_full_square_tiled_gram() {
  constexpr int n = 8, width = 2;
  const auto spin = complete_space(n, 3);
  const int u = static_cast<int>(spin.size());
  const int pairs = xmvb::integrals::packed_active_pair_count(n);
  require(u == 56 && static_cast<std::size_t>(u * u * width) >= 4u * pairs * pairs,
          "full-square Gram fixture does not fit four partials in its input halo");
  xmvb::mcscf::ActiveIntegrals integrals;
  integrals.one_electron.resize(n, n);
  integrals.pair_kernel.resize(pairs, pairs);
  for (int q = 0; q < n; ++q)
    for (int p = 0; p <= q; ++p)
      integrals.one_electron(p, q) = integrals.one_electron(q, p) =
          0.13 * std::cos(0.17 * (p + 1) * (q + 2)) + (p == q ? 0.27 : 0.0);
  for (int q = 0; q < pairs; ++q)
    for (int p = 0; p <= q; ++p)
      integrals.pair_kernel(p, q) = integrals.pair_kernel(q, p) =
          0.019 * std::sin(0.11 * (p + 2) * (q + 1));
  const xmvb::mcscf::DirectCiSigmaAction action(spin, spin, integrals);
  const auto reader = [u, width](const Eigen::MatrixXd& carrier)
      -> xmvb::mcscf::DirectCiTileReader {
    const Eigen::MatrixXd* source = &carrier;
    return [source, u, width](const std::vector<int>& betas,
                       const std::vector<int>& rows,
                       Eigen::Ref<Eigen::MatrixXd> tile,xmvb::mcscf::DirectCiRead mode) {
      for (int b = 0; b < static_cast<int>(betas.size()); ++b)
        for (int v = 0; v < width; ++v)
          for (int a = 0; a < static_cast<int>(rows.size()); ++a)
            tile(a, b * width + v) = (*source)(rows[a], v * u + betas[b]);
    };
  };
  const auto writer = [u, width](Eigen::MatrixXd& image)
      -> xmvb::mcscf::DirectCiTileWriter {
    Eigen::MatrixXd* target = &image;
    return [target, u, width](const std::vector<int>& betas,
                       const std::vector<int>& rows,
                       const xmvb::mcscf::DirectCiColumnAccumulator& accumulate) {
      for (int b = 0; b < static_cast<int>(betas.size()); ++b)
        for (int v = 0; v < width; ++v)
          accumulate(b, v, 0, static_cast<int>(rows.size()),
              target->col(v * u + betas[b]).data(), nullptr, 0);
    };
  };
  const auto close = [](const Eigen::MatrixXd& actual,
                        const Eigen::MatrixXd& expected) {
    return (actual - expected).norm() < 3e-11 * std::max(1.0, expected.norm());
  };
#ifdef _OPENMP
  const int saved_threads = omp_get_max_threads();
  const std::vector<int> thread_counts{2, 4};
#else
  const std::vector<int> thread_counts{1};
#endif
  for (const int parity : {0, 1, -1}) {
    Eigen::MatrixXd response(u, width * u), state(u, width * u);
    for (int v = 0; v < width; ++v) {
      Eigen::MatrixXd r(u, u), s(u, u);
      for (int b = 0; b < u; ++b)
        for (int a = 0; a < u; ++a) {
          r(a, b) = std::sin(0.071 * (a + 1) * (b + 2) + 0.23 * v);
          s(a, b) = std::cos(0.053 * (a + 3) * (b + 1) - 0.31 * v);
        }
      if (parity) {
        r = (r + parity * r.transpose()).eval();
        s = (s + parity * s.transpose()).eval();
      }
      response.middleCols(v * u, u) = r;
      state.middleCols(v * u, u) = s;
    }
    require((response - state).norm() > 1.0,
            "full-square Gram response accidentally equals its state");
    const Eigen::MatrixXd second = 0.37 * response + 0.21 * state;
    const std::array<Eigen::MatrixXd, 2> responses{response, second};
    const std::array<Eigen::MatrixXd, 2> expected_images{
        action.apply(response), action.apply(second)};
    const std::array<xmvb::mcscf::DirectCiIntegralAdjoint, 2> expected_adjoints{
        action.integral_adjoint(response, state, parity),
        action.integral_adjoint(second, state, parity)};
    const auto verify=[&](bool recursive=false,std::barrier<>* rendezvous=nullptr) {
      std::array<Eigen::MatrixXd, 2> images{
          Eigen::MatrixXd::Zero(u, width * u),
          Eigen::MatrixXd::Zero(u, width * u)};
      bool entered=false;
      const auto base_reader=reader(responses[0]);
      const xmvb::mcscf::DirectCiTileReader first_reader=[&](const auto& betas,
          const auto& rows,Eigen::Ref<Eigen::MatrixXd> tile,xmvb::mcscf::DirectCiRead mode) {
        if (!entered) {
          entered=true;
          // Both calls already own live scratch when they meet here.
          if (rendezvous) rendezvous->arrive_and_wait();
          if (recursive) {
            Eigen::MatrixXd nested=Eigen::MatrixXd::Zero(u,width*u);
            action.apply_tiled(width,reader(responses[1]),writer(nested));
            require(close(nested,expected_images[1]),"recursive sigma corrupted its scratch");
          }
        }
        base_reader(betas,rows,tile,mode);
      };
      const std::array<xmvb::mcscf::DirectCiTileRequest, 2> requests{{
          {first_reader, writer(images[0]), nullptr, {}},
          {reader(responses[1]), writer(images[1]), nullptr, {}}}};
      const auto actual = action.apply_tiled_responses(
          width, requests, reader(state), nullptr, parity);
      require(actual.size() == requests.size(),
              "full-square Gram lost a temporal integral adjoint");
      for (std::size_t i = 0; i < requests.size(); ++i) {
        if (parity)
          for (int v = 0; v < width; ++v) {
            const Eigen::MatrixXd block = images[i].middleCols(v * u, u);
            images[i].middleCols(v * u, u) =
                0.5 * (block + parity * block.transpose());
          }
        require(close(images[i], expected_images[i]) &&
                close(actual[i].one_electron, expected_adjoints[i].one_electron) &&
                close(actual[i].pair_kernel, expected_adjoints[i].pair_kernel),
                "full-square row-partitioned Gram differs from dense CI oracles");
      }
    };
    for (const int threads : thread_counts) {
#ifdef _OPENMP
      omp_set_num_threads(threads);
#else
      (void)threads;
#endif
      verify();
    }
    verify(true);
    std::barrier rendezvous(2);
    const auto concurrent=[&] {
#ifdef _OPENMP
      omp_set_num_threads(2);
#endif
      verify(false,&rendezvous);
    };
    auto first=std::async(std::launch::async,concurrent);
    auto second_call=std::async(std::launch::async,concurrent);
    first.get();
    second_call.get();
  }
#ifdef _OPENMP
  omp_set_num_threads(saved_threads);
#endif
}

void check_tiled_integral_direction() {
  const auto unequal_alpha = complete_space(4, 1);
  const auto unequal_beta = complete_space(4, 2);
  const auto equal_spin = complete_space(4, 2);
  for (int width : {1, 2}) {
    check_tiled_integral_direction_case(unequal_alpha, unequal_beta, width, 0);
    for (int parity : {0, 1, -1})
      check_tiled_integral_direction_case(equal_spin, equal_spin, width, parity);
  }
  std::vector<std::uint64_t> alpha_masks, beta_masks;
  for (const auto& occupied : unequal_alpha) {
    std::uint64_t mask = 0;
    for (int orbital : occupied) mask |= std::uint64_t{1} << orbital;
    alpha_masks.push_back(mask);
  }
  for (const auto& occupied : unequal_beta) {
    std::uint64_t mask = 0;
    for (int orbital : occupied) mask |= std::uint64_t{1} << orbital;
    beta_masks.push_back(mask);
  }
  const xmvb::mcscf::ProductSupport selected(4, alpha_masks, beta_masks,
      std::vector<xmvb::mcscf::ProductIndex>{0, 8});
  int halo_only = 0;
  for (int b = 0; b < static_cast<int>(unequal_beta.size()); ++b)
    for (int a = 0; a < static_cast<int>(unequal_alpha.size()); ++a)
      halo_only += !selected.contains(a, b) && selected.contains(a, b, 1);
  require(halo_only > 0, "selected support was already generator-closed");
  check_tiled_integral_direction_case(unequal_alpha, unequal_beta, 2, 0, &selected);
  check_tiled_integral_direction_case(unequal_alpha, unequal_beta, 1, 0,
      nullptr, true);
  check_tiled_integral_direction_case(equal_spin, equal_spin, 1, 1,
      nullptr, false, true);
  // Tiny unequal-spin halos also need correct width and worker ownership.
#ifdef _OPENMP
  const int saved_threads=omp_get_max_threads();
  omp_set_num_threads(4);
#endif
  for (int width:{1,2})
    check_tiled_integral_direction_case(complete_space(4,0),complete_space(4,1),width,0);
  // One occupied full-spin carrier leaves fewer rows than four workers and a
  // minimal input halo, which must use direct output ownership without partials.
  for (int width:{1,2})
    check_tiled_integral_direction_case(complete_space(4,4),complete_space(4,0),width,0);
#ifdef _OPENMP
  omp_set_num_threads(saved_threads);
#endif
#ifdef _OPENMP
#pragma omp parallel num_threads(2)
  {
#pragma omp single
    check_tiled_integral_direction_case(unequal_alpha,unequal_beta,2,0);
  }
#endif
}

/** @brief Check the complete response image, not just a projected scalar. */
void check_exchange_adjoint(bool benchmark = false) {
  using Clock = std::chrono::steady_clock;
  if (benchmark)
    std::cout << "n_orbitals,width,parity,kind,wall_seconds,checksum,max_error\n"
              << std::setprecision(17);
  for (int n : (benchmark ? std::vector<int>{6, 10, 12} : std::vector<int>{4, 10})) {
    const auto spin = complete_space(n, n / 2);
    const int count = static_cast<int>(spin.size());
    const int pairs = xmvb::integrals::packed_active_pair_count(n);
    xmvb::vb::OrthogonalActiveIntegrals integrals;
    integrals.one_electron = Eigen::MatrixXd::Identity(n, n);
    integrals.one_electron(0, 1) = integrals.one_electron(1, 0) = 0.13;
    Eigen::MatrixXd factors(7, pairs);
    for (int j = 0; j < pairs; ++j)
      for (int i = 0; i < factors.rows(); ++i)
        factors(i,j) = 0.07 * std::sin(0.17 * (i+1) * (j+2));
    integrals.pair_kernel = factors.transpose() * factors;
    const xmvb::mcscf::DirectCiSigmaAction action(spin, spin, integrals);
    const int width = n == 12 ? 1 : 2;
    // A row-sliced matrix exercises non-owning inputs with an outer stride.
    Eigen::MatrixXd left_storage(count + 3, width * count);
    Eigen::MatrixXd right_storage(count + 3, width * count);
    auto left = left_storage.topRows(count);
    auto right = right_storage.topRows(count);
    for (int parity : {-1, 1}) {
      for (int block = 0; block < width; ++block)
        for (int beta = 0; beta < count; ++beta)
          for (int alpha = 0; alpha <= beta; ++alpha) {
            const double a = (alpha == beta && parity < 0) ? 0.0 :
                std::sin(0.031 * (alpha+1) * (beta+2+block)) / count;
            const double b = (alpha == beta && parity < 0) ? 0.0 :
                std::cos(0.027 * (alpha+3) * (beta+1+block)) / count;
            left(alpha, block*count+beta) = a;
            left(beta, block*count+alpha) = parity * a;
            right(alpha, block*count+beta) = b;
            right(beta, block*count+alpha) = parity * b;
          }
      const auto full = action.apply_with_adjoint(left, right);
      const auto quotient = action.apply_with_adjoint(left, right, parity);
      const auto adjoint = action.integral_adjoint(left, right, parity);
      const double error = std::max({
          (full.sigma - quotient.sigma).cwiseAbs().maxCoeff(),
          (full.integrals.one_electron - quotient.integrals.one_electron).cwiseAbs().maxCoeff(),
          (full.integrals.pair_kernel - quotient.integrals.pair_kernel).cwiseAbs().maxCoeff(),
          (adjoint.one_electron - full.integrals.one_electron).cwiseAbs().maxCoeff(),
          (adjoint.pair_kernel - full.integrals.pair_kernel).cwiseAbs().maxCoeff()});
      require(error < 3e-11, "exchange quotient changed the full sigma or integral adjoint");
      require((quotient.sigma - action.apply(left)).norm() < 3e-10,
          "reconstructed exchange sigma differs from independent full sigma");
      require((quotient.sigma - action.apply(left, parity)).norm() < 3e-10,
          "standalone exchange sigma differs from fused sigma");
      const auto accepted = action.apply_with_adjoint(left, left, parity);
      const Eigen::MatrixXd copy = left;
      const auto independent = action.integral_adjoint(left, copy);
      require((accepted.integrals.pair_kernel - independent.pair_kernel).norm() < 3e-11 &&
              (accepted.integrals.one_electron - independent.one_electron).norm() < 3e-11,
          "shared accepted-point scatter changed integral adjoint");
      if (!benchmark) continue;
      for (int sector : {0, parity}) {
        const auto sigma_start = Clock::now();
        const auto sigma = action.apply(left, sector);
        const double sigma_wall = std::chrono::duration<double>(Clock::now() - sigma_start).count();
        std::cout << n << ',' << width << ',' << parity << ','
                  << (sector == 0 ? "full_sigma" : "quotient_sigma") << ',' << sigma_wall << ','
                  << sigma.squaredNorm() << ',' << error << '\n';
        for (bool same : {false, true}) {
          const auto start = Clock::now();
          const auto result = same ? action.apply_with_adjoint(left, left, sector)
                                   : action.apply_with_adjoint(left, right, sector);
          const double wall = std::chrono::duration<double>(Clock::now() - start).count();
          std::cout << n << ',' << width << ',' << parity << ','
                    << (sector == 0 ? "full_" : "quotient_")
                    << (same ? "accepted" : "response") << ',' << wall << ','
                    << result.sigma.squaredNorm() + result.integrals.pair_kernel.squaredNorm()
                    << ',' << error << '\n';
        }
      }
    }
  }
}

/** @brief Bounded kernel timings; keep separate from convergence benchmarks. */
void benchmark_sigma() {
  using Clock = std::chrono::steady_clock;
  std::cout << "kernel,n_orbitals,n_alpha,n_beta,block_width,calls,wall_seconds,checksum\n"
            << std::setprecision(17);
  for (const auto& sector : {std::array<int, 3>{6, 3, 3}, {11, 5, 5}, {11, 5, 6},
                            {12, 6, 6}, {11, 1, 5}, {11, 5, 1}}) {
    const int n_orbitals = sector[0];
    const auto alpha = complete_space(n_orbitals, sector[1]);
    const auto beta = complete_space(n_orbitals, sector[2]);
    const int n_alpha = static_cast<int>(alpha.size());
    const int n_beta = static_cast<int>(beta.size());
    const int n_pairs = xmvb::integrals::packed_active_pair_count(n_orbitals);
    xmvb::vb::OrthogonalActiveIntegrals integrals;
    integrals.one_electron = Eigen::MatrixXd::Identity(n_orbitals, n_orbitals);
    Eigen::MatrixXd factors(7, n_pairs);
    for (int j = 0; j < n_pairs; ++j)
      for (int i = 0; i < factors.rows(); ++i)
        factors(i, j) = 0.07 * std::sin(0.17 * (i + 1) * (j + 2));
    integrals.pair_kernel = factors.transpose() * factors;
    const xmvb::mcscf::DirectCiSigmaAction action(alpha, beta, integrals);
    for (int width : {1, 4}) {
      Eigen::MatrixXd coefficients(n_alpha, width * n_beta);
      for (int j = 0; j < coefficients.cols(); ++j)
        for (int i = 0; i < coefficients.rows(); ++i)
          coefficients(i, j) = std::sin(0.031 * (i + 1) * (j + 2));
      double checksum = action.apply(coefficients).squaredNorm(); // warmup
      constexpr int repeats = 5;
      auto start = Clock::now();
      for (int repeat = 0; repeat < repeats; ++repeat)
        checksum += action.apply(coefficients).squaredNorm();
      const double wall = std::chrono::duration<double>(Clock::now() - start).count();
      std::cout << "sigma," << n_orbitals << ',' << n_alpha << ',' << n_beta << ',' << width
                << ',' << repeats << ',' << wall << ',' << checksum << '\n';
      start = Clock::now();
      const auto adjoint = action.integral_adjoint(coefficients, coefficients);
      const double adjoint_wall = std::chrono::duration<double>(Clock::now() - start).count();
      std::cout << "adjoint," << n_orbitals << ',' << n_alpha << ',' << n_beta << ',' << width
                << ",1," << adjoint_wall << ','
                << adjoint.one_electron.squaredNorm() + adjoint.pair_kernel.squaredNorm()
                << '\n';
      Eigen::MatrixXd state = coefficients.array().cos().matrix();
      check_fused_response(action, coefficients, state);
      for (bool fused : {false, true}) {
        double total = 0.0;
        start = Clock::now();
        for (int repeat = 0; repeat < repeats; ++repeat) {
          if (fused) {
            const auto result = action.apply_with_adjoint(coefficients, state);
            total += result.sigma.squaredNorm() + result.integrals.one_electron.squaredNorm() +
                result.integrals.pair_kernel.squaredNorm();
          } else {
            const auto value = action.apply(coefficients);
            const auto gradient = action.integral_adjoint(coefficients, state);
            total += value.squaredNorm() + gradient.one_electron.squaredNorm() +
                gradient.pair_kernel.squaredNorm();
          }
        }
        const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
        std::cout << (fused ? "response_fused," : "response_separate,")
                  << n_orbitals << ',' << n_alpha << ',' << n_beta << ',' << width << ',' << repeats
                  << ',' << elapsed << ',' << total << '\n';
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--benchmark-exchange") {
      check_exchange_adjoint(true);
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--benchmark-response") {
      benchmark_structure_response();
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--benchmark") {
      benchmark_sigma();
      return 0;
    }
    check_generated_exterior_rows();
    check_exterior_transform(7);
    check_exterior_transform(259);
    check_planner();
    check_sigma_action();
    check_sigma_tile_boundary();
    check_distinct_spin_sigma_action();
    check_streamed_generator_adjoint_requests();
    check_generated_generator_panels();
    check_exchange_reduced_generator();
    check_fused_spin_sectors();
    check_tiled_integral_direction();
    check_permuted_tiled_direction();
    check_full_square_tiled_gram();
    check_spin_sector_carrier_inverse();
    check_exchange_adjoint();
    check_demand_driven_structure_diagonal();
    std::cout << "orthogonal direct-CI planner and exterior transform: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "orthogonal direct-CI test failed: " << error.what() << '\n';
    return 1;
  }
}
