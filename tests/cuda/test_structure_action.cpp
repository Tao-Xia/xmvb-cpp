#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "core/pair_index.hpp"
#include "cuda/ci/sigma.hpp"
#include "cuda/linalg/davidson.hpp"
#include "cuda/memory/context.hpp"
#include "cuda/vb/action.hpp"
#include "linalg/eigensolver.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/assembly/hamiltonian_overlap.hpp"

namespace {

using Strings = std::vector<std::vector<int>>;
using Terms = std::vector<std::vector<xmvb::vb::StructureExpansionTerm>>;

void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

void check_action(int count, bool transformed) {
  // Four orbitals, two alpha electrons and one beta electron: the two spin
  // string spaces have different dimensions and different exterior maps.
  constexpr int n = 4;
  const Strings alpha{{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
  const Strings beta{{0},{1},{2},{3}};
  Strings products_alpha, products_beta;
  Terms terms;
  for (const auto& a : alpha)
    for (const auto& b : beta) {
      products_alpha.push_back(a);
      products_beta.push_back(b);
      terms.emplace_back();
    }
  // The third column shares determinant products with the first two. This
  // exercises a nontrivial selected overlap, including for the 3-column case.
  terms[0].push_back({0,1.0});
  if (count == 3) {
    terms[5].push_back({1,1.0});
    terms[0].push_back({2,0.6});
    terms[5].push_back({2,-0.3});
    terms[23].push_back({2,0.7});
  }
  Eigen::Matrix4d s = Eigen::Matrix4d::Identity();
  if (transformed) {
    s(0,1) = s(1,0) = 0.11;
    s(1,2) = s(2,1) = -0.07;
    s(0,2) = s(2,0) = 0.04;
  }
  const std::vector<double> overlap(s.data(),s.data()+s.size());
  Eigen::Matrix4d h;
  h << -1.0,0.08,-0.03,0.02,
       0.08,-0.7,0.05,-0.01,
       -0.03,0.05,-0.3,0.04,
       0.02,-0.01,0.04,0.2;
  xmvb::integrals::ActiveSpaceTwoElectronResult eri;
  auto& packed = eri.packed_active_two_electron_integrals;
  packed.resize(xmvb::integrals::packed_active_two_electron_integral_count(n));
  for (int p = 0; p < 10; ++p)
    for (int q = 0; q <= p; ++q)
      packed[xmvb::core::PairIndex::packed_pair_of_pairs_index(p,q)] =
          0.015 * std::cos(0.23 * (p+1) * (q+2));
  const auto topology = xmvb::vb::build_orthogonal_spin_topology(
      products_alpha,products_beta,n);
  const auto diagonal_model = xmvb::vb::FullDeterminantStructureHamiltonianOverlapBuilder{}
      .build_davidson_jacobi_preconditioner(products_alpha,products_beta,terms,
          overlap,h,n,eri,count,topology);
  const xmvb::vb::StructureAction cpu(terms,count,topology,overlap,h,eri,n,&diagonal_model);
  xmvb::cuda::Context context(256ULL*1024ULL*1024ULL);
  const xmvb::cuda::StructureAction gpu(context,cpu);
  Eigen::MatrixXd vectors(count,3);
  for (int j = 0; j < vectors.cols(); ++j)
    for (int i = 0; i < count; ++i)
      vectors(i,j) = std::sin(0.37*(i+1)*(j+2));
  const auto expected = cpu.apply(vectors);
  const auto actual = gpu.apply(vectors);
  const double h_error = (actual.hamiltonian-expected.hamiltonian).norm() /
      std::max(1.0,expected.hamiltonian.norm());
  const double s_error = (actual.overlap-expected.overlap).norm() /
      std::max(1.0,expected.overlap.norm());
  require(h_error < 2e-10 && s_error < 2e-10,
          "GPU structure H/S block differs from the CPU action");
  require(gpu.size() == count,"GPU action structure dimension changed");
  require(context.counters().peak_device_bytes != 0,
          "GPU action allocated no device storage");
  {
    xmvb::cuda::DirectCiSigma point_sigma(context,gpu.carrier_sigma());
    const auto initial_generation=gpu.carrier_sigma().workspace_generation();
    point_sigma.prepare_with_adjoint(1);
    require(point_sigma.workspace_generation()!=initial_generation &&
        point_sigma.workspace_generation()==gpu.carrier_sigma().workspace_generation(),
        "CUDA point sigma did not share action scratch generation");
    const auto allocated=context.counters().current_device_bytes;
    point_sigma.prepare_with_adjoint(1);
    require(context.counters().current_device_bytes==allocated,
        "CUDA point sigma duplicated an already prepared workspace");
    const auto after_fused=gpu.apply(vectors);
    require((after_fused.hamiltonian-expected.hamiltonian).norm()<2e-10 &&
        (after_fused.overlap-expected.overlap).norm()<2e-10,
        "CUDA structure graph retained stale fused scratch addresses");
    point_sigma.prepare(3);
    const auto after_resize=gpu.apply(vectors);
    require((after_resize.hamiltonian-expected.hamiltonian).norm()<2e-10 &&
        (after_resize.overlap-expected.overlap).norm()<2e-10,
        "CUDA structure graph retained stale resized scratch addresses");
  }
  if (count == 3) {
    const auto& diagonal = cpu.preconditioner_diagonal();
    const auto options = xmvb::linalg::make_davidson_options(
        count,1,1e-9,1e-8,1e-9);
    const xmvb::linalg::GeneralizedEigenAction cpu_action = [&cpu](
        const Eigen::Ref<const Eigen::MatrixXd>& x) {
      auto images = cpu.apply(x);
      return xmvb::linalg::GeneralizedEigenActionResult{
          std::move(images.hamiltonian),std::move(images.overlap)};
    };
    const xmvb::cuda::DeviceEigenAction gpu_action = [&gpu](
        xmvb::linalg::ConstMatrixView x,xmvb::linalg::MatrixView hx,
        xmvb::linalg::MatrixView sx) { gpu.apply(x,hx,sx); };
    const auto cpu_roots = xmvb::linalg::GeneralizedEigensolver().solve_davidson(
        cpu_action,diagonal.hamiltonian,diagonal.overlap,options);
    const auto gpu_roots = xmvb::cuda::solve_davidson(
        context,gpu_action,diagonal.hamiltonian,diagonal.overlap,options);
    require(std::abs(cpu_roots.eigenpairs.eigenvalues.at(0)-
                     gpu_roots.eigenpairs.eigenvalues.at(0)) < 1e-8,
            "GPU Davidson energy differs from current CPU source");
    require(gpu_roots.stationarity_residual_norms.at(0) < 1.01e-9,
            "GPU Davidson did not satisfy CI stationarity");
  }
}

}  // namespace

int main() {
  try {
    check_action(1,false);
    check_action(1,true);
    check_action(3,false);
    check_action(3,true);
    std::cout << "CUDA structure H/S block parity: PASS\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
