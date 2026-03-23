#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/cpp_vb_input.hpp"
#include "vb/matrices/full_determinant_structure_hamiltonian_overlap_builder.hpp"
#include "vb/matrices/raw_structure_data.hpp"
#include "vb/matrices/structure_accumulation_result.hpp"
#include "vb/orbital/active_space_one_electron_builder.hpp"
#include "vb/orbital/active_space_orbital_preparer.hpp"
#include "vb/orbital/ao_effective_one_electron_builder.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace xmvb::vb {

struct DeepVBHJaxInferenceOptions {
  std::filesystem::path repo_root = ".";
  std::filesystem::path python_executable = ".venv/bin/python";
  std::filesystem::path checkpoint_path;
  std::filesystem::path onnx_model_path;
  std::filesystem::path work_directory = "/tmp";
  std::string backend = "python_jax";
  std::string device = "gpu";
  std::string dtype = "float32";
  bool keep_work_directory = false;
  VbScfAlgorithm algorithm = VbScfAlgorithm::Original;
};

struct DeepVBHJaxPrediction {
  StructureAccumulationResult structure_matrices;
  std::vector<double> predicted_two_electron_hamiltonian_matrix;
  std::vector<double> predicted_sparse_orbital_residual;
  std::vector<double> predicted_dense_orbital_residual;
  std::vector<double> predicted_updated_orbital_value_table;
  std::vector<double> predicted_updated_dense_orbital_coefficients;
  std::string backend = "python_jax";
  double one_electron_reference_energy = 0.0;
  double reference_energy_residual = 0.0;
  bool has_predicted_total_energy = false;
  double predicted_total_energy = 0.0;
  std::filesystem::path work_directory;
  std::filesystem::path sample_directory;
  double preparation_wall_time_seconds = 0.0;
  double backend_wall_time_seconds = 0.0;
  double python_wall_time_seconds = 0.0;
  double total_wall_time_seconds = 0.0;
};

class DeepVBHJaxInferenceRunner {
public:
  explicit DeepVBHJaxInferenceRunner(
      DeepVBHJaxInferenceOptions options = {});

  DeepVBHJaxPrediction predict(
      const CppVbInput& input,
      const RawStructureData& raw_structure_data,
      const CppVbStaticMoleculeMetadata& static_molecule_metadata,
      double nuclear_repulsion_energy) const;

private:
  ActiveSpaceOrbitalPreparer orbital_preparer_;
  AoEffectiveOneElectronBuilder ao_effective_one_electron_builder_;
  ActiveSpaceOneElectronBuilder active_space_one_electron_builder_;
  FullDeterminantStructureHamiltonianOverlapBuilder structure_builder_;
  DeepVBHJaxInferenceOptions options_;
};

}  // namespace xmvb::vb
