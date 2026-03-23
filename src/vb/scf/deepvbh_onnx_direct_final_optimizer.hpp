#pragma once

#include <vector>

#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/model/deepvbh_jax_inference_runner.hpp"
#include "vb/scf/cpp_orbital_gradient_evaluator.hpp"
#include "vb/scf/cpp_vb_scf_optimizer.hpp"
#include "vb/scf/cpp_vb_scf_optimizer_result.hpp"

namespace xmvb::vb {

struct DeepVBHOnnxDirectFinalOptimizerOptions {
  CppVbScfOptimizerOptions optimizer_options{};
  DeepVBHJaxInferenceOptions inference_options{};
  double initial_step_scale = 1.0;
  double minimum_step_scale = 1.0e-3;
  double step_shrink_factor = 0.5;
  int max_backtracks = 8;
  int exact_fallback_max_iterations = 64;
};

class DeepVBHOnnxDirectFinalOptimizer {
public:
  explicit DeepVBHOnnxDirectFinalOptimizer(
      DeepVBHOnnxDirectFinalOptimizerOptions options = {});

  CppVbScfOptimizerResult optimize(
      const CppVbInput& input,
      const RawStructureData& raw_structure_data,
      const CppVbStaticMoleculeMetadata& static_molecule_metadata,
      double nuclear_repulsion_energy = 0.0) const;

  CppVbScfOptimizerResult optimize(
      const CppVbInput& input,
      const RawStructureData& raw_structure_data,
      const CppVbStaticMoleculeMetadata& static_molecule_metadata,
      const std::vector<int>& selected_state_indices,
      const std::vector<double>& state_average_weights,
      double nuclear_repulsion_energy) const;

private:
  DeepVBHJaxInferenceRunner inference_runner_;
  CppOrbitalGradientEvaluator orbital_gradient_evaluator_;
  CppVbScfEvaluator scf_evaluator_;
  DeepVBHOnnxDirectFinalOptimizerOptions options_;
};

}  // namespace xmvb::vb
