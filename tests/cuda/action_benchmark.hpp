#pragma once

#include <cuda_profiler_api.h>
#include <nvtx3/nvtx3.hpp>

#include "core/timing.hpp"
#include "c12_fixture.hpp"
#include "cuda/memory/buffer.hpp"
#include "cuda/vb/product.hpp"

namespace xmvb::cuda::test {

inline Eigen::MatrixXd action_vectors(int rows, int width) {
  Eigen::MatrixXd result(rows,width);
  for (int j=0;j<width;++j)
    for (int i=0;i<rows;++i)
      result(i,j)=std::sin(0.31*(i+1)*(j+2));
  return result;
}

/** Same-point GPU path comparison; no CPU action or optimizer changes. */
inline void compare_action_paths(Context& context, const cuda::StructureAction& gpu) {
  const auto vectors=action_vectors(gpu.n_structures(),1);
  Matrix<double> x(context,vectors.rows(),1),h(context,vectors.rows(),1),m(context,vectors.rows(),1);
  copy_to_device(context,x.view(),vectors);
  Eigen::MatrixXd reference_h(vectors.rows(),1),reference_m(vectors.rows(),1);
  const auto measure=[&](const char* name,const auto& apply) {
    context.synchronize();
    const nvtx3::scoped_range range{name};
    const auto before=context.counters();
    const auto start=Clock::now();
    apply();
    context.synchronize();
    const auto after=context.counters();
    std::cout << name << "_seconds=" << elapsed(start,Clock::now()) << '\n'
              << name << "_resident_bytes=" << after.current_device_bytes << '\n'
              << name << "_h2d_bytes=" << after.host_to_device_bytes-before.host_to_device_bytes << '\n';
  };
  check_cuda(cudaProfilerStart(),"path profiler start");
  measure("resident_cold",[&]{gpu.apply(x.view(),h.view(),m.view());});
  measure("resident_warm",[&]{gpu.apply(x.view(),h.view(),m.view());});
  copy_to_host(context,reference_h,h.view());copy_to_host(context,reference_m,m.view());
  // Release the resident graph and its full carriers before the tiled measurement.
  gpu.reserve_workspace(context.device_byte_budget());
  auto product=gpu.make_product_contraction();
  const ProductIncidence incidence{gpu.term_offsets_data(),gpu.term_structures_data(),
      gpu.term_weights_data(),gpu.n_structures()};
  const auto selected=[&]{product->apply_selected(incidence,x.view(),gpu.carrier_sigma(),
      h.view(),m.view(),context.available_device_bytes());};
  measure("selected_cold",selected);
  measure("selected_warm",selected);
  check_cuda(cudaProfilerStop(),"path profiler stop");
  Eigen::MatrixXd actual_h(vectors.rows(),1),actual_m(vectors.rows(),1);
  copy_to_host(context,actual_h,h.view());copy_to_host(context,actual_m,m.view());
  const double he=(actual_h-reference_h).norm()/std::max(1.0,reference_h.norm());
  const double me=(actual_m-reference_m).norm()/std::max(1.0,reference_m.norm());
  std::cout << "path_h_relative_error=" << he << '\n' << "path_m_relative_error=" << me << '\n';
  if (!actual_h.allFinite() || !actual_m.allFinite() || he>2e-9 || me>2e-9)
    throw std::runtime_error("resident/tiled GPU H/M mismatch");
}

/** One saved CPU baseline; existing phase scopes may be nested, not additive. */
inline void benchmark_cpu_action(const vb::StructureAction& cpu, int width) {
  constexpr int repetitions=3;
  const auto vectors=action_vectors(cpu.n_structures(),width);
  const auto warmup=cpu.apply(vectors);
  auto& phases=profiling::phase_totals();
  const bool previous=phases.enabled;
  std::array<unsigned long long,static_cast<std::size_t>(profiling::Phase::Count)> ns{},calls{};
  for (std::size_t i=0;i<ns.size();++i) {
    ns[i]=phases.nanoseconds[i].load(std::memory_order_relaxed);
    calls[i]=phases.calls[i].load(std::memory_order_relaxed);
  }
  phases.enabled=true;
  const auto start=Clock::now();
  try {
    for (int i=0;i<repetitions;++i) {
      const auto result=cpu.apply(vectors);
      if (!result.hamiltonian.allFinite() || !result.overlap.allFinite())
        throw std::runtime_error("nonfinite CPU H/M action");
    }
  } catch (...) {
    phases.enabled=previous;
    throw;
  }
  const double seconds=elapsed(start,Clock::now())/repetitions;
  phases.enabled=previous;
  const std::string prefix="action_b"+std::to_string(width)+"_cpu_";
  std::cout << prefix << "seconds=" << seconds << '\n'
            << prefix << "phase_semantics=nested_inclusive\n";
  for (std::size_t i=0;i<ns.size();++i) {
    const auto count=phases.calls[i].load(std::memory_order_relaxed)-calls[i];
    if (count==0) continue;
    const auto name=profiling::PhaseTotals::name(static_cast<profiling::Phase>(i));
    const auto duration=phases.nanoseconds[i].load(std::memory_order_relaxed)-ns[i];
    std::cout << prefix << "phase_" << name << "_seconds=" << duration*1e-9/repetitions << '\n'
              << prefix << "phase_" << name << "_calls=" << count << '\n';
  }
}

/** GPU-only resident H/M timing; CPU comparisons reuse saved benchmark data. */
inline void benchmark_action(Context& context, const cuda::StructureAction& gpu, int width) {
  constexpr int repetitions = 5;
  const auto vectors=action_vectors(gpu.n_structures(),width);
  Matrix<double> x(context, vectors.rows(), width),
      h(context, vectors.rows(), width), m(context, vectors.rows(), width);
  context.copy_to_device(x.data(), vectors.data(), vectors.size()*sizeof(double));
  gpu.apply(x.view(), h.view(), m.view());
  context.synchronize();

  const auto before = context.counters();
  cudaEvent_t begin = nullptr, end = nullptr;
  check_cuda(cudaEventCreate(&begin), "action event begin");
  try {
    check_cuda(cudaEventCreate(&end), "action event end");
    check_cuda(cudaProfilerStart(), "action profiler start");
    check_cuda(cudaEventRecord(begin, context.stream()), "action event record begin");
    const auto start = Clock::now();
    for (int i = 0; i < repetitions; ++i)
      gpu.apply(x.view(), h.view(), m.view());
    check_cuda(cudaEventRecord(end, context.stream()), "action event record end");
    context.synchronize();
    const double gpu_seconds = elapsed(start, Clock::now())/repetitions;
    check_cuda(cudaProfilerStop(), "action profiler stop");
    float milliseconds = 0;
    check_cuda(cudaEventElapsedTime(&milliseconds, begin, end), "action event elapsed");
    const auto after = context.counters();
    Eigen::MatrixXd hx(vectors.rows(), width), mx(vectors.rows(), width);
    context.copy_to_host(hx.data(), h.data(), hx.size()*sizeof(double));
    context.copy_to_host(mx.data(), m.data(), mx.size()*sizeof(double));
    if (!hx.allFinite() || !mx.allFinite())
      throw std::runtime_error("nonfinite GPU H/M action");
    const std::string prefix = "action_b"+std::to_string(width)+"_";
    std::cout << prefix << "gpu_wall_seconds=" << gpu_seconds << '\n'
              << prefix << "gpu_event_seconds=" << milliseconds/(1000*repetitions) << '\n'
              << prefix << "h_norm=" << hx.norm() << '\n'
              << prefix << "m_norm=" << mx.norm() << '\n'
              << prefix << "h2d_bytes=" << after.host_to_device_bytes-before.host_to_device_bytes << '\n'
              << prefix << "d2h_bytes=" << after.device_to_host_bytes-before.device_to_host_bytes << '\n';
  } catch (...) {
    if (end) cudaEventDestroy(end);
    cudaEventDestroy(begin);
    throw;
  }
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
}

} // namespace xmvb::cuda::test
