#include <array>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "cint.h"
#include "runtime/cpp_vb_input_loader.hpp"
#include "runtime_c/local_runtime_api.h"
#include "runtime_c/local_runtime_grid_api.h"
#include "vb/pdft/libcint_ao_grid_evaluator.hpp"
#include "vb/pdft/molecular_grid.hpp"
#include "vb/pdft/selected_state_exact_physical_one_rdm_builder.hpp"
#include "vb/pdft/vb_pdft_energy_evaluator.hpp"
#include "vb/scf/cpp_active_space_gradient_evaluator.hpp"

namespace {

struct Options {
  std::string input_path;
  int state_index = 0;
  int functional_id = 1;
  int radial_points = 50;
  int angular_points = 302;
  double density_threshold = 1.0e-12;
  xmvb::vb::VBSCFAlgorithm algorithm = xmvb::vb::VBSCFAlgorithm::Original;
  xmvb::vb::StandardTwoElectronMode standard_two_electron_mode =
      xmvb::vb::StandardTwoElectronMode::Auto;
};

struct GridDiagnostics {
  double exact_ao_metric_trace = 0.0;
  double hf_ao_metric_trace = 0.0;
  double grid_ao_metric_trace = 0.0;
  double snorm_scaled_grid_ao_metric_trace = 0.0;
  double inverse_snorm_scaled_grid_ao_metric_trace = 0.0;
  double legacy_grid_ao_metric_trace = 0.0;
  double legacy_snorm_scaled_grid_ao_metric_trace = 0.0;
  double legacy_inverse_snorm_scaled_grid_ao_metric_trace = 0.0;
  double hf_ao_overlap_relative_frobenius_error = 0.0;
  double hf_ao_overlap_max_abs_error = 0.0;
  double legacy_ao_overlap_relative_frobenius_error = 0.0;
  double legacy_ao_overlap_max_abs_error = 0.0;
  double ao_overlap_relative_frobenius_error = 0.0;
  double ao_overlap_max_abs_error = 0.0;
  double ao_normalization_min = 0.0;
  double ao_normalization_max = 0.0;
  double min_grid_weight = 0.0;
  double max_grid_weight = 0.0;
  double min_density = 0.0;
  double max_density = 0.0;
};

void print_usage() {
  std::cerr << "usage: check_vb_pdft_energy_evaluator <input.xmi>"
               " [--state-index N]"
               " [--functional-id N]"
               " [--radial-points N]"
               " [--angular-points N]"
               " [--density-threshold X]"
               " [--algorithm original]"
               " [--standard-two-electron-mode auto|exact|ri]\n";
}

Options parse_arguments(int argc, char** argv) {
  if (argc < 2 || ((argc - 2) % 2 != 0)) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }

  Options options;
  options.input_path = argv[1];
  for (int argument_index = 2; argument_index < argc; argument_index += 2) {
    const std::string name = argv[argument_index];
    const std::string value = argv[argument_index + 1];
    if (name == "--state-index") {
      options.state_index = std::stoi(value);
      continue;
    }
    if (name == "--functional-id") {
      options.functional_id = std::stoi(value);
      continue;
    }
    if (name == "--radial-points") {
      options.radial_points = std::stoi(value);
      continue;
    }
    if (name == "--angular-points") {
      options.angular_points = std::stoi(value);
      continue;
    }
    if (name == "--density-threshold") {
      options.density_threshold = std::stod(value);
      continue;
    }
    if (name == "--algorithm") {
      if (value == "original") {
        options.algorithm = xmvb::vb::VBSCFAlgorithm::Original;
      } else {
        throw std::invalid_argument("invalid algorithm: " + value);
      }
      continue;
    }
    if (name == "--standard-two-electron-mode") {
      if (value == "auto") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::Auto;
      } else if (value == "exact") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::Exact;
      } else if (value == "ri") {
        options.standard_two_electron_mode =
            xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
      } else {
        throw std::invalid_argument(
            "invalid standard two-electron mode: " + value);
      }
      continue;
    }
    throw std::invalid_argument("unknown argument: " + name);
  }

  if (options.state_index < 0) {
    throw std::invalid_argument("--state-index must be non-negative");
  }
  if (options.functional_id <= 0) {
    throw std::invalid_argument("--functional-id must be positive");
  }
  if (options.radial_points <= 0) {
    throw std::invalid_argument("--radial-points must be positive");
  }
  if (options.angular_points <= 0) {
    throw std::invalid_argument("--angular-points must be positive");
  }
  if (options.density_threshold <= 0.0) {
    throw std::invalid_argument("--density-threshold must be positive");
  }
  return options;
}

void extract_atomic_geometry_from_libcint_input(
    const xmvb::vb::LibcintInput& libcint_input,
    Eigen::MatrixXd* atomic_coords,
    Eigen::VectorXd* atomic_charges) {
  if (atomic_coords == nullptr || atomic_charges == nullptr) {
    throw std::invalid_argument("atomic geometry output must not be null");
  }
  *atomic_coords = Eigen::MatrixXd::Zero(libcint_input.n_atoms, 3);
  *atomic_charges = Eigen::VectorXd::Zero(libcint_input.n_atoms);
  for (int atom_index = 0; atom_index < libcint_input.n_atoms; ++atom_index) {
    const int atom_offset = xmvb::to_size(atom_index) * ATM_SLOTS;
    const int coordinate_offset = libcint_input.atm[atom_offset + PTR_COORD];
    (*atomic_coords)(atom_index, 0) =
        libcint_input.env[xmvb::to_size(coordinate_offset)];
    (*atomic_coords)(atom_index, 1) =
        libcint_input.env[xmvb::to_size(coordinate_offset + 1)];
    (*atomic_coords)(atom_index, 2) =
        libcint_input.env[xmvb::to_size(coordinate_offset + 2)];
    (*atomic_charges)(atom_index) = libcint_input.atm[atom_offset + CHARGE_OF];
  }
}

GridDiagnostics compute_grid_diagnostics(
    const std::string& input_path,
    const char* executable_path,
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::CppActiveSpaceGradientResult& gradient_result,
    const xmvb::vb::pdft::VbPdftConfig& config,
    const xmvb::vb::CppActiveSpaceGradientEvaluator& gradient_evaluator) {
  xmvb::vb::SelectedStateExactPhysicalOneRdmBuilder one_rdm_builder(
      gradient_evaluator);
  const xmvb::vb::SelectedStateExactPhysicalOneRdmResult one_rdm_result =
      one_rdm_builder.build_from_state_specific_gradient(input, gradient_result);

  Eigen::MatrixXd atomic_coords;
  Eigen::VectorXd atomic_charges;
  extract_atomic_geometry_from_libcint_input(
      input.libcint_input,
      &atomic_coords,
      &atomic_charges);
  xmvb::vb::pdft::MolecularGridBuilder grid_builder(config.grid_config);
  const xmvb::vb::pdft::MolecularGrid grid =
      grid_builder.build(atomic_coords, atomic_charges);

  xmvb::vb::pdft::LibcintAoGridEvaluator ao_grid_evaluator(input.libcint_input);
  const xmvb::vb::pdft::AoGridValues ao_values =
      ao_grid_evaluator.evaluate_values(grid.points);

  const int n_basis_functions =
      input.orbital_preparation_input.n_basis_functions;
  const std::size_t ao_matrix_size =
      xmvb::to_size(n_basis_functions) * n_basis_functions;
  if (input.orbital_preparation_input.active_orbital_overlap_matrix.size() !=
      ao_matrix_size) {
    throw std::invalid_argument("AO overlap matrix size mismatch in diagnostic");
  }
  const Eigen::Map<const Eigen::MatrixXd> exact_ao_overlap(
      input.orbital_preparation_input.active_orbital_overlap_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const Eigen::Map<const Eigen::MatrixXd> hf_ao_overlap(
      input.orbital_preparation_input.hf_overlap_matrix.data(),
      n_basis_functions,
      n_basis_functions);

  Eigen::MatrixXd weighted_ao_values = ao_values.values;
  weighted_ao_values.array().colwise() *= grid.weights.array();
  Eigen::MatrixXd grid_ao_overlap =
      ao_values.values.transpose() * weighted_ao_values;
  Eigen::VectorXd ao_normalization(input.orbital_preparation_input.n_basis_functions);
  for (int basis_function = 0;
       basis_function < input.orbital_preparation_input.n_basis_functions;
       ++basis_function) {
    ao_normalization(basis_function) =
        input.orbital_preparation_input.ao_normalization[
            xmvb::to_size(basis_function)];
  }
  Eigen::MatrixXd snorm_scaled_grid_ao_overlap = grid_ao_overlap;
  snorm_scaled_grid_ao_overlap.array().rowwise() *= ao_normalization.transpose().array();
  snorm_scaled_grid_ao_overlap.array().colwise() *= ao_normalization.array();
  Eigen::VectorXd inverse_ao_normalization = ao_normalization.cwiseInverse();
  Eigen::MatrixXd inverse_snorm_scaled_grid_ao_overlap = grid_ao_overlap;
  inverse_snorm_scaled_grid_ao_overlap.array().rowwise() *=
      inverse_ao_normalization.transpose().array();
  inverse_snorm_scaled_grid_ao_overlap.array().colwise() *=
      inverse_ao_normalization.array();

  std::array<char, 2048> legacy_error_buffer{};
  XmvbCppRuntimeHandle* runtime_handle = nullptr;
  auto destroy_runtime_handle = [&]() {
    if (runtime_handle != nullptr) {
      xmvb_cpp_runtime_destroy(runtime_handle);
      runtime_handle = nullptr;
    }
  };
  auto throw_legacy_runtime_error = [&](const std::string& stage) {
    const std::string detail(legacy_error_buffer.data());
    destroy_runtime_handle();
    throw std::runtime_error(stage + ": " + detail);
  };
  if (xmvb_cpp_runtime_create(
          &runtime_handle,
          legacy_error_buffer.data(),
          legacy_error_buffer.size()) != 0) {
    throw std::runtime_error(
        "legacy runtime create failed: " + std::string(legacy_error_buffer.data()));
  }
  if (xmvb_cpp_runtime_load_input(
          runtime_handle,
          const_cast<char*>(input_path.c_str()),
          executable_path,
          legacy_error_buffer.data(),
          legacy_error_buffer.size()) != 0) {
    throw_legacy_runtime_error("legacy runtime load_input failed");
  }
  char runtime_output_stem[] = "/tmp/check_vb_pdft_energy_evaluator_legacy";
  if (xmvb_cpp_runtime_initialize_wavefunction(
          runtime_handle,
          runtime_output_stem,
          legacy_error_buffer.data(),
          legacy_error_buffer.size()) != 0) {
    throw_legacy_runtime_error("legacy runtime initialize_wavefunction failed");
  }
  Eigen::MatrixXd legacy_grid_ao_overlap(
      input.orbital_preparation_input.n_basis_functions,
      input.orbital_preparation_input.n_basis_functions);
  if (xmvb_cpp_runtime_compute_medium_grid_ao_overlap(
          runtime_handle,
          legacy_grid_ao_overlap.data(),
          xmvb::to_size(legacy_grid_ao_overlap.size()),
          legacy_error_buffer.data(),
          legacy_error_buffer.size()) != 0) {
    throw_legacy_runtime_error(
        "legacy medium-grid AO overlap evaluation failed");
  }
  Eigen::MatrixXd legacy_snorm_scaled_grid_ao_overlap = legacy_grid_ao_overlap;
  legacy_snorm_scaled_grid_ao_overlap.array().rowwise() *= ao_normalization.transpose().array();
  legacy_snorm_scaled_grid_ao_overlap.array().colwise() *= ao_normalization.array();
  Eigen::MatrixXd legacy_inverse_snorm_scaled_grid_ao_overlap = legacy_grid_ao_overlap;
  legacy_inverse_snorm_scaled_grid_ao_overlap.array().rowwise() *=
      inverse_ao_normalization.transpose().array();
  legacy_inverse_snorm_scaled_grid_ao_overlap.array().colwise() *=
      inverse_ao_normalization.array();
  const Eigen::MatrixXd legacy_overlap_error =
      legacy_grid_ao_overlap - exact_ao_overlap;
  // The legacy `xgrids/value_ao` diagnostic path currently corrupts the
  // standalone runtime teardown sequence.  This handle is used only for the
  // one-shot overlap comparison inside this diagnostic executable, so keep it
  // alive until process exit instead of crashing during destroy.
  runtime_handle = nullptr;

  const Eigen::MatrixXd overlap_error = grid_ao_overlap - exact_ao_overlap;
  const Eigen::MatrixXd hf_overlap_error = hf_ao_overlap - exact_ao_overlap;
  const Eigen::MatrixXd chi_gamma =
      ao_values.values * one_rdm_result.physical_ao_total_density_matrix;
  const Eigen::VectorXd rho =
      (chi_gamma.cwiseProduct(ao_values.values)).rowwise().sum();

  GridDiagnostics diagnostics;
  diagnostics.exact_ao_metric_trace = one_rdm_result.ao_total_metric_trace;
  diagnostics.hf_ao_metric_trace =
      one_rdm_result.physical_ao_total_density_matrix.cwiseProduct(
          hf_ao_overlap).sum();
  diagnostics.grid_ao_metric_trace =
      one_rdm_result.physical_ao_total_density_matrix.cwiseProduct(
          grid_ao_overlap).sum();
  diagnostics.snorm_scaled_grid_ao_metric_trace =
      one_rdm_result.physical_ao_total_density_matrix.cwiseProduct(
          snorm_scaled_grid_ao_overlap).sum();
  diagnostics.inverse_snorm_scaled_grid_ao_metric_trace =
      one_rdm_result.physical_ao_total_density_matrix.cwiseProduct(
          inverse_snorm_scaled_grid_ao_overlap).sum();
  diagnostics.legacy_grid_ao_metric_trace =
      one_rdm_result.physical_ao_total_density_matrix.cwiseProduct(
          legacy_grid_ao_overlap).sum();
  diagnostics.legacy_snorm_scaled_grid_ao_metric_trace =
      one_rdm_result.physical_ao_total_density_matrix.cwiseProduct(
          legacy_snorm_scaled_grid_ao_overlap).sum();
  diagnostics.legacy_inverse_snorm_scaled_grid_ao_metric_trace =
      one_rdm_result.physical_ao_total_density_matrix.cwiseProduct(
          legacy_inverse_snorm_scaled_grid_ao_overlap).sum();
  diagnostics.legacy_ao_overlap_relative_frobenius_error =
      legacy_overlap_error.norm() /
      std::max(1.0e-16, exact_ao_overlap.norm());
  diagnostics.legacy_ao_overlap_max_abs_error =
      legacy_overlap_error.cwiseAbs().maxCoeff();
  diagnostics.hf_ao_overlap_relative_frobenius_error =
      hf_overlap_error.norm() / std::max(1.0e-16, exact_ao_overlap.norm());
  diagnostics.hf_ao_overlap_max_abs_error =
      hf_overlap_error.cwiseAbs().maxCoeff();
  diagnostics.ao_overlap_relative_frobenius_error =
      overlap_error.norm() / std::max(1.0e-16, exact_ao_overlap.norm());
  diagnostics.ao_overlap_max_abs_error =
      overlap_error.cwiseAbs().maxCoeff();
  diagnostics.ao_normalization_min = ao_normalization.minCoeff();
  diagnostics.ao_normalization_max = ao_normalization.maxCoeff();
  diagnostics.min_grid_weight = grid.weights.minCoeff();
  diagnostics.max_grid_weight = grid.weights.maxCoeff();
  diagnostics.min_density = rho.minCoeff();
  diagnostics.max_density = rho.maxCoeff();
  return diagnostics;
}

void print_result(
    const xmvb::vb::pdft::VbPdftEnergyResult& result,
    const Options& options,
    const GridDiagnostics& diagnostics) {
  std::cout << std::setprecision(15)
            << "state_index = " << result.state_index << '\n'
            << "functional_id = " << options.functional_id << '\n'
            << "grid_radial_points = " << options.radial_points << '\n'
            << "grid_angular_points = " << options.angular_points << '\n'
            << "n_grid_points = " << result.n_grid_points << '\n'
            << "exact_ao_metric_trace = "
            << diagnostics.exact_ao_metric_trace << '\n'
            << "hf_ao_metric_trace = "
            << diagnostics.hf_ao_metric_trace << '\n'
            << "grid_ao_metric_trace = "
            << diagnostics.grid_ao_metric_trace << '\n'
            << "snorm_scaled_grid_ao_metric_trace = "
            << diagnostics.snorm_scaled_grid_ao_metric_trace << '\n'
            << "inverse_snorm_scaled_grid_ao_metric_trace = "
            << diagnostics.inverse_snorm_scaled_grid_ao_metric_trace << '\n'
            << "legacy_grid_ao_metric_trace = "
            << diagnostics.legacy_grid_ao_metric_trace << '\n'
            << "legacy_snorm_scaled_grid_ao_metric_trace = "
            << diagnostics.legacy_snorm_scaled_grid_ao_metric_trace << '\n'
            << "legacy_inverse_snorm_scaled_grid_ao_metric_trace = "
            << diagnostics.legacy_inverse_snorm_scaled_grid_ao_metric_trace << '\n'
            << "ao_overlap_relative_frobenius_error = "
            << diagnostics.ao_overlap_relative_frobenius_error << '\n'
            << "ao_overlap_max_abs_error = "
            << diagnostics.ao_overlap_max_abs_error << '\n'
            << "hf_ao_overlap_relative_frobenius_error = "
            << diagnostics.hf_ao_overlap_relative_frobenius_error << '\n'
            << "hf_ao_overlap_max_abs_error = "
            << diagnostics.hf_ao_overlap_max_abs_error << '\n'
            << "legacy_ao_overlap_relative_frobenius_error = "
            << diagnostics.legacy_ao_overlap_relative_frobenius_error << '\n'
            << "legacy_ao_overlap_max_abs_error = "
            << diagnostics.legacy_ao_overlap_max_abs_error << '\n'
            << "ao_normalization_min = "
            << diagnostics.ao_normalization_min << '\n'
            << "ao_normalization_max = "
            << diagnostics.ao_normalization_max << '\n'
            << "grid_weight_min = "
            << diagnostics.min_grid_weight << '\n'
            << "grid_weight_max = "
            << diagnostics.max_grid_weight << '\n'
            << "rho_min = "
            << diagnostics.min_density << '\n'
            << "rho_max = "
            << diagnostics.max_density << '\n'
            << "nuclear_repulsion_energy = "
            << result.nuclear_repulsion_energy << '\n'
            << "one_electron_energy = "
            << result.one_electron_energy << '\n'
            << "coulomb_energy = "
            << result.coulomb_energy << '\n'
            << "on_top_energy = "
            << result.on_top_energy << '\n'
            << "integrated_electron_count = "
            << result.integrated_electron_count << '\n'
            << "total_energy = "
            << result.total_energy << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);

    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.standard_two_electron_mode =
        options.standard_two_electron_mode;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(
            options.input_path,
            load_options);

    xmvb::vb::CppActiveSpaceGradientEvaluator gradient_evaluator(
        options.algorithm);

    // Reuse one state-specific active-space result so VB-PDFT does not pay for
    // the determinant/structure/eigensolver path twice inside the diagnostic.
    const auto gradient_result =
        gradient_evaluator.evaluate(
            load_result.input,
            std::vector<int>{options.state_index},
            std::vector<double>{1.0},
            load_result.nuclear_repulsion_energy);

    xmvb::vb::pdft::VbPdftConfig config;
    config.grid_config.radial_points = options.radial_points;
    config.grid_config.angular_points = options.angular_points;
    config.functional_id = options.functional_id;
    config.density_threshold = options.density_threshold;
    config.use_gga = false;

    xmvb::vb::pdft::VbPdftEnergyEvaluator evaluator(
        config,
        gradient_evaluator);
    const GridDiagnostics diagnostics =
        compute_grid_diagnostics(
            options.input_path,
            argv[0],
            load_result.input,
            gradient_result,
            config,
            gradient_evaluator);
    const auto result =
        evaluator.evaluate_from_state_specific_gradient(
            load_result.input,
            gradient_result,
            load_result.nuclear_repulsion_energy);

    print_result(result, options, diagnostics);
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "error: " << exception.what() << '\n';
    return 1;
  }
}
