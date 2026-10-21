#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <unistd.h>

#include <Eigen/Core>

#include "c12_fixture.hpp"
#include "cuda/linalg/davidson.hpp"
#include "cuda/memory/context.hpp"
#include "cuda/vb/action.hpp"
#include "linalg/eigensolver.hpp"
#include "action_benchmark.hpp"

namespace {

using xmvb::cuda::test::Clock;
using xmvb::cuda::test::elapsed;

void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

Eigen::MatrixXd probe_vectors(int rows) {
  Eigen::MatrixXd vectors(rows,3);
  for (int j = 0; j < vectors.cols(); ++j)
    for (int i = 0; i < rows; ++i)
      vectors(i,j) = std::sin(0.31*(i+1)*(j+2));
  return vectors;
}

void report_fixture(const xmvb::cuda::test::C12Fixture& fixture) {
  std::cout << "structures=" << fixture.action->n_structures() << '\n'
            << "load_seconds=" << fixture.load_seconds << '\n'
            << "active_space_seconds=" << fixture.active_space_seconds << '\n'
            << "topology_seconds=" << fixture.topology_seconds << '\n'
            << "cpu_action_init_seconds=" << fixture.cpu_action_init_seconds << '\n'
            << "orbital_source=xmi_gus\n"
            << "energy_offset_hartree=" << fixture.energy_offset() << '\n';
}

void report_result(const xmvb::linalg::DavidsonResult& result,
                   double offset, double seconds) {
  require(!result.eigenpairs.eigenvalues.empty(),"Davidson returned no root");
  std::cout << "ci_electronic_energy_hartree=" << result.eigenpairs.eigenvalues[0] << '\n'
            << "total_energy_hartree=" << result.eigenpairs.eigenvalues[0]+offset << '\n'
            << "ci_solve_seconds=" << seconds << '\n'
            << "ci_iterations=" << result.iterations << '\n'
            << "ci_block_actions=" << result.block_actions << '\n'
            << "ci_peak_subspace_dimension=" << result.peak_subspace_dimension << '\n'
            << "ci_relative_residual=" << result.relative_residual_norms.at(0) << '\n'
            << "ci_stationarity_residual_hartree="
            << result.stationarity_residual_norms.at(0) << '\n';
}

void report_device(const xmvb::cuda::Context& context) {
  const auto& counters = context.counters();
  std::cout << "gpu_peak_device_bytes=" << counters.peak_device_bytes << '\n'
            << "gpu_current_device_bytes=" << counters.current_device_bytes << '\n'
            << "gpu_host_to_device_bytes=" << counters.host_to_device_bytes << '\n'
            << "gpu_device_to_host_bytes=" << counters.device_to_host_bytes << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3 || (std::string(argv[1]) != "--cpu" &&
                      std::string(argv[1]) != "--gpu" &&
                      std::string(argv[1]) != "--action" &&
                      std::string(argv[1]) != "--cpu-action" &&
                      std::string(argv[1]) != "--check"))
      throw std::invalid_argument("usage: c12_cuda_benchmark --cpu|--gpu|--check|--action INPUT.xmi");
    const std::string mode = argv[1];
    std::cout << std::unitbuf;
    const auto process_start = Clock::now();
    xmvb::cuda::test::C12Fixture fixture(argv[2]);
    const auto& cpu = *fixture.action;
    const auto& diagonal = cpu.preconditioner_diagonal();
    const int n = cpu.n_structures();
    require(diagonal.hamiltonian.size()==n && diagonal.overlap.size()==n,
            "C12 Davidson diagonal has the wrong size");
    std::cout << std::setprecision(17) << "mode=" << mode << '\n'
              << "process_id=" << getpid() << '\n';
    report_fixture(fixture);

    const auto options = xmvb::linalg::make_davidson_options(
        n,1,1.0e-7,1.0e-3,1.0e-7);
    if (mode=="--cpu-action") {
      xmvb::cuda::test::benchmark_cpu_action(cpu,1);
      xmvb::cuda::test::benchmark_cpu_action(cpu,3);
    } else if (mode=="--cpu") {
      const xmvb::linalg::GeneralizedEigenAction action = [&cpu](
          const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
        auto result = cpu.apply(vectors);
        return xmvb::linalg::GeneralizedEigenActionResult{
            std::move(result.hamiltonian),std::move(result.overlap)};
      };
      const auto start = Clock::now();
      const auto result = xmvb::linalg::GeneralizedEigensolver().solve_davidson(
          action,diagonal.hamiltonian,diagonal.overlap,options);
      report_result(result,fixture.energy_offset(),elapsed(start,Clock::now()));
    } else {
      const auto device_start = Clock::now();
      xmvb::cuda::Context context(64ULL*1024ULL*1024ULL*1024ULL);
      xmvb::cuda::StructureAction gpu(context,cpu);
      context.synchronize();
      std::cout << "gpu_action_init_seconds=" << elapsed(device_start,Clock::now()) << '\n';
      if (mode=="--action") {
        xmvb::cuda::test::benchmark_action(context,gpu,1);
        xmvb::cuda::test::benchmark_action(context,gpu,3);
      } else if (mode=="--check") {
        const auto vectors = probe_vectors(n);
        const auto expected = cpu.apply(vectors);
        const auto actual = gpu.apply(vectors);
        const double h_error=(actual.hamiltonian-expected.hamiltonian).norm()/
            std::max(1.0,expected.hamiltonian.norm());
        const double s_error=(actual.overlap-expected.overlap).norm()/
            std::max(1.0,expected.overlap.norm());
        std::cout << "h_block_relative_error=" << h_error << '\n'
                  << "s_block_relative_error=" << s_error << '\n';
        require(h_error<2e-9 && s_error<2e-9,
                "C12 GPU H/S block differs from CPU");
      } else {
        const xmvb::cuda::DeviceEigenAction action = [&gpu](
            xmvb::linalg::ConstMatrixView v,
            xmvb::linalg::MatrixView h,
            xmvb::linalg::MatrixView s) { gpu.apply(v,h,s); };
        const auto start = Clock::now();
        const auto result = xmvb::cuda::solve_davidson(
            context,action,diagonal.hamiltonian,diagonal.overlap,options);
        context.synchronize();
        report_result(result,fixture.energy_offset(),elapsed(start,Clock::now()));
      }
      report_device(context);
    }
    std::cout << "benchmark_elapsed_seconds=" << elapsed(process_start,Clock::now()) << '\n';
  } catch (const std::exception& error) {
    std::cerr << "c12_cuda_benchmark: " << error.what() << '\n';
    return 1;
  }
}
