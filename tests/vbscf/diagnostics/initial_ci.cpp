#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "core/memory.hpp"
#include "input/loading/loader.hpp"
#include "vbscf/orbitals/gauge/support_preserving.hpp"
#include "vbscf/optimization/driver/options.hpp"
#include "vbscf/optimization/objective/function.hpp"

/** @brief Time the production initial CI point, stopping before any gradient. */
int main(int argc,char** argv) {
  try {
    if (argc!=2) throw std::invalid_argument("usage: test_initial_ci input.xmi");
    const auto start=std::chrono::steady_clock::now();
    xmvb::memory::configure({});
    auto loaded=xmvb::vb::load_vbscf_input_with_timings(argv[1]);
    auto& input=loaded.input;
    if (!input.orbital_preparation_input.maintain_inactive_gauge)
      xmvb::vb::apply_support_preserving_inactive_gauge(&input.orbital_preparation_input);
    xmvb::vb::balance_active_gauge(&input.orbital_preparation_input);
    const xmvb::vb::SparseParameterLayout layout(input.orbital_preparation_input);
    const Eigen::VectorXd parameters=layout.pack(input.orbital_preparation_input);
    std::vector<int> states(loaded.state_average_count);
    std::iota(states.begin(),states.end(),0);
    const std::vector<double> weights(states.size(),1.0/states.size());
    const xmvb::vb::VbScfOptimizerOptions options;
    const xmvb::vb::OrbitalGradientEvaluator evaluator;
    xmvb::vb::VbScfObjective objective(std::move(input),layout,states,weights,
        loaded.nuclear_repulsion_energy,loaded.structure_eigensolver,
        {options.energy_tolerance,options.gradient_tolerance},&evaluator);
    const auto forward_start=std::chrono::steady_clock::now();
    const auto result=objective.evaluate_trial_energy(parameters);
    const auto end=std::chrono::steady_clock::now();
    std::cout<<std::setprecision(17)<<"initial_energy_Eh="<<result.energy
        <<"\ninitial_forward_seconds="
        <<std::chrono::duration<double>(end-forward_start).count()
        <<"\nend_to_end_seconds="<<std::chrono::duration<double>(end-start).count()
        <<"\nhvp_actions=0\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
  }
}
