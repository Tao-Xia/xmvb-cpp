#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "cuda/vb/scf.hpp"
#include "input/loading/loader.hpp"
#include "output/orbitals/writer.hpp"

int main(int argc,char** argv) {
  try {
    if (argc!=2) throw std::invalid_argument("usage: cuvbscf INPUT.xmi");
    std::cout<<std::unitbuf<<std::setprecision(16);
    const auto start=std::chrono::steady_clock::now();
    const auto loaded=xmvb::vb::load_vbscf_input_with_timings(argv[1]);
    if (loaded.state_average_count!=1)
      throw std::invalid_argument("cuvbscf currently supports single-state exact calculations");
    xmvb::vb::VbScfOptimizerOptions options;
    options.backend=xmvb::vb::VbScfOptimizerBackend::Ciah;
    options.orbital_preconditioner=xmvb::vb::OrbitalPreconditioner::HessianDiagonal;
    options.max_iterations=loaded.requested_scf_max_iterations;
    xmvb::cuda::Context context(4ULL*1024ULL*1024ULL*1024ULL);
    std::cout<<"method=cuvbscf_ciah\npreconditioner=analytic_hessian_diagonal\n"
             <<"structures="<<loaded.input.structure_data.n_structures<<'\n'
             <<"basis_functions="<<loaded.input.orbital_preparation_input.n_basis_functions<<'\n'
             <<"active_orbitals="<<loaded.input.orbital_preparation_input.n_active_orbitals<<'\n'
             <<"input_seconds="<<loaded.total_seconds<<'\n'
             <<"iteration,energy,delta_energy,gradient_inf,hvp,ci_actions,keyframes,seconds\n";
    auto result=xmvb::cuda::solve_scf(context,loaded.input,loaded.nuclear_repulsion_energy,
        options,[](const xmvb::cuda::ScfIteration& row) {
          std::cout<<row.iteration<<','<<row.energy<<','<<row.energy_change<<','
              <<row.gradient_inf<<','<<row.hvp<<','<<row.ci_actions<<','
              <<row.keyframes<<','<<row.seconds<<'\n';
        });
    if (result.converged) {
      auto output_orbitals=loaded.input.orbital_preparation_input;
      Eigen::MatrixXd values(result.orbitals.rows(),result.orbitals.cols());
      xmvb::cuda::copy_to_host(context,values,result.orbitals.view());
      output_orbitals.orbital_value_table.assign(values.data(),values.data()+values.size());
      std::cout<<"orbital_output="<<xmvb::output::write_orbital_file(argv[1],output_orbitals).string()<<'\n';
    }
    const auto& counters=context.counters();
    std::cout<<"converged="<<result.converged<<'\n'
             <<"termination="<<result.reason<<'\n'
             <<"total_energy_hartree="<<result.energy<<'\n'
             <<"scf_iterations="<<result.iterations<<'\n'
             <<"orbital_hvp_actions="<<result.hvp<<'\n'
             <<"ci_block_actions="<<result.ci_actions<<'\n'
             <<"ci_prepared_points="<<result.ci_prepared_points<<'\n'
             <<"selected_metric_builds="<<result.selected_metric_builds<<'\n'
             <<"gradient_inf="<<result.gradient_inf<<'\n'
             <<"scf_seconds="<<result.seconds<<'\n'
             <<"process_seconds="<<std::chrono::duration<double>(
                 std::chrono::steady_clock::now()-start).count()<<'\n'
             <<"gpu_peak_device_bytes="<<counters.peak_device_bytes<<'\n'
             <<"gpu_host_to_device_bytes="<<counters.host_to_device_bytes<<'\n'
             <<"gpu_device_to_host_bytes="<<counters.device_to_host_bytes<<'\n';
    return result.converged?0:2;
  } catch (const std::exception& error) {
    std::cerr<<"cuvbscf: "<<error.what()<<'\n';return 1;
  }
}
