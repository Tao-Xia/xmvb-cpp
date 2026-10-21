#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Cholesky>

#include "core/pair_index.hpp"
#include "cuda/ci/sigma.hpp"
#include "cuda/linalg/cholesky.hpp"
#include "cuda/memory/buffer.hpp"
#include "cuda/memory/context.hpp"
#include "cuda/memory/graph.hpp"
#include "cuda/vb/action.hpp"
#include "cuda/vb/metric.hpp"
#include "cuda/vb/transform.hpp"
#include "integrals/active/two_electron.hpp"
#include "mcscf/ci/integrals.hpp"
#include "mcscf/ci/sigma.hpp"
#include "mcscf/ci/strings.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"
#include "vbscf/structures/orthogonal_ci/metric.hpp"
#include "vbscf/structures/overlap/spin_metric.hpp"

namespace {

using Matrix = Eigen::MatrixXd;
using Strings = std::vector<std::vector<int>>;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void near(const Matrix& actual, const Matrix& expected, double tolerance,
          const char* message) {
  require(actual.rows() == expected.rows() && actual.cols() == expected.cols(),
          "CUDA production CI result has the wrong shape");
  require(actual.allFinite() && expected.allFinite() &&
              (actual - expected).norm() <=
                  tolerance * std::max(1.0, expected.norm()), message);
}

Matrix values(int rows, int cols, double phase) {
  Matrix result(rows, cols);
  for (int j = 0; j < cols; ++j)
    for (int i = 0; i < rows; ++i)
      result(i, j) = std::sin(phase + 0.31 * (i + 1) * (j + 2)) +
                     0.17 * std::cos(0.53 * (i + 2) * (j + 1));
  return result;
}

Matrix symmetric_values(int size, double phase) {
  const Matrix raw = values(size, size, phase);
  return (0.5 * (raw + raw.transpose())).eval();
}

xmvb::mcscf::ActiveIntegrals active_integrals(int n, bool rotated) {
  const int pairs = n * (n + 1) / 2;
  xmvb::mcscf::ActiveIntegrals integrals{
      symmetric_values(n, 0.21), 0.16 * symmetric_values(pairs, 0.63)};
  if (!rotated) return integrals;

  // An explicit orthogonal orbital map exercises a genuinely transformed h
  // and four-index pair kernel, including off-diagonal packed channels.
  Matrix orbital = Matrix::Identity(n, n);
  const double angle = 0.37;
  orbital(0, 0) = orbital(1, 1) = std::cos(angle);
  orbital(0, 1) = std::sin(angle);
  orbital(1, 0) = -std::sin(angle);
  const Matrix old_pair = integrals.pair_kernel;
  integrals.one_electron =
      (orbital.transpose() * integrals.one_electron * orbital).eval();
  integrals.pair_kernel.setZero(pairs, pairs);
  for (int p = 0; p < n; ++p)
    for (int q = 0; q <= p; ++q)
      for (int r = 0; r < n; ++r)
        for (int s = 0; s <= r; ++s) {
          double transformed = 0.0;
          for (int a = 0; a < n; ++a)
            for (int b = 0; b < n; ++b)
              for (int c = 0; c < n; ++c)
                for (int d = 0; d < n; ++d)
                  transformed += orbital(a, p) * orbital(b, q) *
                      old_pair(xmvb::core::PairIndex::packed_pair_index(a, b),
                               xmvb::core::PairIndex::packed_pair_index(c, d)) *
                      orbital(c, r) * orbital(d, s);
          const int pq = xmvb::core::PairIndex::packed_pair_index(p, q);
          const int rs = xmvb::core::PairIndex::packed_pair_index(r, s);
          integrals.pair_kernel(pq, rs) = transformed;
          integrals.pair_kernel(rs, pq) = transformed;
        }
  return integrals;
}

Matrix download(xmvb::cuda::Context& context,
                const xmvb::cuda::Matrix<double>& device) {
  Matrix result(device.rows(), device.cols());
  xmvb::cuda::copy_to_host(context, result, device.view());
  return result;
}

double bilinear(const Matrix& left, const Matrix& image) {
  return (left.array() * image.array()).sum();
}

void check_directional_derivative(
    const Strings& alpha, const Strings& beta,
    const xmvb::mcscf::ActiveIntegrals& base, const Matrix& left,
    const Matrix& right, const Matrix& h_adjoint, const Matrix& g_adjoint,
    bool one_electron) {
  constexpr double step = 2.0e-6;
  auto plus = base, minus = base;
  if (one_electron) {
    const Matrix direction = symmetric_values(base.one_electron.rows(), 1.17);
    plus.one_electron += step * direction;
    minus.one_electron -= step * direction;
    const xmvb::mcscf::DirectCiSigmaAction positive(alpha, beta, plus);
    const xmvb::mcscf::DirectCiSigmaAction negative(alpha, beta, minus);
    const double finite_difference =
        (bilinear(left, positive.apply(right)) -
         bilinear(left, negative.apply(right))) / (2.0 * step);
    const double analytic = bilinear(h_adjoint, direction);
    require(std::abs(finite_difference - analytic) <
                3.0e-7 * std::max(1.0, std::abs(finite_difference)),
            "CUDA one-electron adjoint fails a full-H finite difference");
  } else {
    const Matrix direction = symmetric_values(base.pair_kernel.rows(), 1.49);
    plus.pair_kernel += step * direction;
    minus.pair_kernel -= step * direction;
    const xmvb::mcscf::DirectCiSigmaAction positive(alpha, beta, plus);
    const xmvb::mcscf::DirectCiSigmaAction negative(alpha, beta, minus);
    const double finite_difference =
        (bilinear(left, positive.apply(right)) -
         bilinear(left, negative.apply(right))) / (2.0 * step);
    const double analytic = bilinear(g_adjoint, direction);
    require(std::abs(finite_difference - analytic) <
                3.0e-7 * std::max(1.0, std::abs(finite_difference)),
            "CUDA pair-kernel adjoint fails a full-H finite difference");
  }
}

void check_sigma(int rhs_count,bool rotated,int maximum_tile=3,bool one_product=false) {
  constexpr int n = 4;
  const xmvb::mcscf::SpinStrings alpha_space(n,one_product ? n:2),
      beta_space(n,one_product ? 0:1);
  const Strings alpha = alpha_space.occupations(), beta = beta_space.occupations();
  const auto integrals = active_integrals(n, rotated);
  const xmvb::mcscf::DirectCiSigmaAction cpu(alpha, beta, integrals);
  const Matrix left = values(alpha.size(), beta.size() * rhs_count, 0.26);
  const Matrix right = values(alpha.size(), beta.size() * rhs_count, 0.82);
  xmvb::cuda::Context context(128ULL * 1024ULL * 1024ULL);
  xmvb::cuda::DirectCiSigma gpu(context,cpu,maximum_tile);
  require(one_product || gpu.n_alpha_strings() != gpu.n_beta_strings(),
          "CUDA direct-CI fixture lost unequal spin dimensions");
  xmvb::cuda::Matrix<double> d_left(context, left.rows(), left.cols());
  xmvb::cuda::Matrix<double> d_right(context, right.rows(), right.cols());
  xmvb::cuda::Matrix<double> d_image(context, left.rows(), left.cols());
  xmvb::cuda::Matrix<double> d_h(context, n, n);
  const int pairs = n * (n + 1) / 2;
  xmvb::cuda::Matrix<double> d_g(context, pairs, pairs);
  xmvb::cuda::copy_to_device(context, d_left.view(), left);
  xmvb::cuda::copy_to_device(context, d_right.view(), right);

  gpu.apply(d_left.view(),d_image.view());
  near(download(context,d_image),cpu.apply(left),3.0e-10,
       "CUDA source-row sigma differs from CPU");
  gpu.apply(d_right.view(),d_image.view(),true);
  near(download(context,d_image),cpu.apply(left)+cpu.apply(right),3.0e-10,
       "CUDA source-row sigma add mode differs from CPU");

  const auto cpu_adjoint = cpu.integral_adjoint(left, right);
  gpu.integral_adjoint(d_left.view(), d_right.view(), d_h.view(), d_g.view());
  near(download(context, d_h), cpu_adjoint.one_electron, 3.0e-10,
       "CUDA direct-CI one-electron adjoint differs from CPU");
  near(download(context, d_g), cpu_adjoint.pair_kernel, 3.0e-10,
       "CUDA direct-CI pair-kernel adjoint differs from CPU");
  check_directional_derivative(alpha, beta, integrals, left, right,
                               download(context, d_h), download(context, d_g), true);
  check_directional_derivative(alpha, beta, integrals, left, right,
                               download(context, d_h), download(context, d_g), false);

  gpu.apply_with_adjoint(d_right.view(),d_right.view(),d_image.view(),d_h.view(),d_g.view());
  const auto same_source=cpu.apply_with_adjoint(right,right);
  near(download(context,d_image),same_source.sigma,3.0e-10,
       "CUDA same-source sigma differs from CPU");
  near(download(context,d_h),same_source.integrals.one_electron,3.0e-10,
       "CUDA shared scatter h adjoint differs from CPU");
  near(download(context,d_g),same_source.integrals.pair_kernel,3.0e-10,
       "CUDA shared scatter g adjoint differs from CPU");

  gpu.apply_with_adjoint(d_left.view(), d_right.view(), d_image.view(),
                         d_h.view(), d_g.view());
  const auto fused = cpu.apply_with_adjoint(left, right);
  near(download(context, d_image), fused.sigma, 3.0e-10,
       "CUDA fused direct-CI sigma differs from CPU");
  near(download(context, d_h), fused.integrals.one_electron, 3.0e-10,
       "CUDA fused direct-CI h adjoint differs from CPU");
  near(download(context, d_g), fused.integrals.pair_kernel, 3.0e-10,
       "CUDA fused direct-CI g adjoint differs from CPU");

  // A genuinely independent integral direction and state tangent exercise
  // H*left+dH*right without materializing two carrier images. The adjoint
  // remains that of the fixed Hamiltonian, not the directional Hamiltonian.
  const xmvb::mcscf::ActiveIntegrals delta{
      symmetric_values(n,1.74),0.11*symmetric_values(pairs,2.36)};
  const xmvb::mcscf::DirectCiSigmaAction cpu_direction(alpha,beta,delta);
  xmvb::cuda::DirectCiSigma gpu_direction(context,gpu);
  xmvb::cuda::Matrix<double> d_delta_h(context,n,n),d_delta_g(context,pairs,pairs);
  xmvb::cuda::copy_to_device(context,d_delta_h.view(),delta.one_electron);
  xmvb::cuda::copy_to_device(context,d_delta_g.view(),delta.pair_kernel);
  gpu_direction.update_integrals(d_delta_h.view(),d_delta_g.view());
  const auto direction_bytes=gpu_direction.resident_bytes();
  const Matrix combined_reference=cpu.apply(left)+cpu_direction.apply(right);
  gpu.apply_direction_with_adjoint(d_left.view(),d_right.view(),gpu_direction,
      d_image.view(),d_h.view(),d_g.view());
  near(download(context,d_image),combined_reference,3.0e-10,
       "CUDA combined directional CI image differs from CPU");
  near(download(context,d_h),fused.integrals.one_electron,3.0e-10,
       "CUDA combined directional h adjoint differs from CPU");
  near(download(context,d_g),fused.integrals.pair_kernel,3.0e-10,
       "CUDA combined directional g adjoint differs from CPU");
  const auto accepted=cpu.integral_adjoint(right,right);
  const double explicit_forcing=bilinear(right,cpu_direction.apply(right));
  const double adjoint_forcing=
      bilinear(accepted.one_electron,delta.one_electron)+
      bilinear(accepted.pair_kernel,delta.pair_kernel);
  require(std::abs(explicit_forcing-adjoint_forcing)<
          3.0e-10*std::max(1.0,std::abs(explicit_forcing)),
      "CUDA directional energy elimination violates the integral adjoint identity");
  require(gpu_direction.resident_bytes()==direction_bytes,
       "CUDA combined directional sweep allocated a directional channel panel");
  const xmvb::cuda::Graph directional_graph(context,[&] {
    gpu.apply_direction_with_adjoint(d_left.view(),d_right.view(),gpu_direction,
        d_image.view(),d_h.view(),d_g.view());
  });

  xmvb::cuda::Matrix<double> d_diagonal(context, alpha.size(), beta.size());
  gpu.diagonal(d_diagonal.view());
  near(download(context, d_diagonal), cpu.diagonal(integrals.one_electron),
       3.0e-10, "CUDA direct-CI diagonal differs from CPU");

  // A generic generator is intentionally nonsymmetric: the symmetric packed
  // channel convention must not accidentally symmetrize CI transport.
  const Matrix generator=values(n,n,0.57);
  xmvb::cuda::Matrix<double> d_generator(context,n,n),d_generator_adjoint(context,n,n);
  xmvb::cuda::copy_to_device(context,d_generator.view(),generator);
  gpu.apply_generator(d_right.view(),d_generator.view(),d_image.view());
  near(download(context,d_image),cpu.apply_one_body_generator(right,generator),
       3.0e-10,"CUDA nonsymmetric CI generator differs from CPU");
  const auto fused_bytes=gpu.resident_bytes();
  gpu.one_body_generator_adjoint(d_left.view(),d_right.view(),d_generator_adjoint.view());
  require(gpu.resident_bytes()-fused_bytes <= sizeof(double)*
          (2*static_cast<std::size_t>(pairs)+left.size()),
          "CUDA generator adjoint duplicated a fused channel slab");
  const Matrix generator_adjoint=download(context,d_generator_adjoint);
  near(generator_adjoint,cpu.one_body_generator_adjoint(left,right),3.0e-10,
       "CUDA nonsymmetric generator adjoint differs from CPU");
  require(std::abs(bilinear(generator_adjoint,generator)-
                   bilinear(left,download(context,d_image))) <
              3.0e-10*std::max(1.0,left.norm()*right.norm()*generator.norm()),
          "CUDA generator violates forward/adjoint pairing");
  gpu.apply_with_adjoint(d_left.view(),d_right.view(),d_image.view(),
                         d_h.view(),d_g.view());
  near(download(context,d_image),fused.sigma,3.0e-10,
       "CUDA fused sigma changed after generator workspace reuse");
  near(download(context,d_g),fused.integrals.pair_kernel,3.0e-10,
       "CUDA fused adjoint changed after generator workspace reuse");
  directional_graph.launch();
  near(download(context,d_image),combined_reference,3.0e-10,
       "CUDA captured directional sweep changed after generator workspace reuse");
  near(download(context,d_g),fused.integrals.pair_kernel,3.0e-10,
       "CUDA captured directional adjoint changed after generator workspace reuse");

  // The ordinary action captures scattered_/transformed_ addresses. A
  // generator VJP must borrow them without invalidating the captured graph.
  xmvb::cuda::DirectCiSigma captured_generator(context,cpu);
  captured_generator.prepare(rhs_count);
  const xmvb::cuda::Graph captured_before_generator(context,[&] {
    captured_generator.apply(d_right.view(),d_image.view());
  });
  const auto captured_bytes=captured_generator.resident_bytes();
  captured_generator.one_body_generator_adjoint(
      d_left.view(),d_right.view(),d_generator_adjoint.view());
  require(captured_generator.resident_bytes()-captured_bytes <= sizeof(double)*
          (2*static_cast<std::size_t>(pairs)+left.size()),
          "CUDA generator adjoint duplicated a captured channel slab");
  near(download(context,d_generator_adjoint),generator_adjoint,3.0e-10,
       "CUDA captured sigma generator adjoint differs from CPU");
  captured_before_generator.launch();
  near(download(context,d_image),cpu.apply(right),3.0e-10,
       "CUDA generator workspace reuse invalidated captured sigma");

  auto changed=integrals;
  changed.one_electron+=0.03*symmetric_values(n,1.92);
  changed.pair_kernel+=0.02*symmetric_values(pairs,2.13);
  const xmvb::mcscf::DirectCiSigmaAction updated(alpha,beta,changed);
  xmvb::cuda::Matrix<double> d_new_h(context,n,n),d_new_g(context,pairs,pairs);
  xmvb::cuda::copy_to_device(context,d_new_h.view(),changed.one_electron);
  xmvb::cuda::copy_to_device(context,d_new_g.view(),changed.pair_kernel);
  gpu.prepare(rhs_count);
  const xmvb::cuda::Graph captured(context,[&] {
    gpu.apply(d_right.view(),d_image.view());
  });
  gpu.update_integrals(d_new_h.view(),d_new_g.view());
  captured.launch();
  near(download(context,d_image),updated.apply(right),3.0e-10,
       "GPU integral update differs from a fresh CPU Hamiltonian");
  gpu.diagonal(d_diagonal.view());
  near(download(context,d_diagonal),updated.diagonal(changed.one_electron),3.0e-10,
       "GPU integral update left a stale CI diagonal");
  require(context.counters().peak_device_bytes < context.device_byte_budget(),
          "CUDA production CI test exceeded its device budget");
}

void check_spin_metric(int rhs_count) {
  constexpr int n = 4;
  const Strings alpha = xmvb::mcscf::SpinStrings(n, 2).occupations();
  const Strings beta = xmvb::mcscf::SpinStrings(n, 2).occupations();
  std::vector<std::vector<xmvb::vb::StructureExpansionTerm>> terms(
      alpha.size() * beta.size());
  std::vector<std::vector<int>> product_alpha, product_beta;
  Matrix selected = Matrix::Zero(terms.size(), 4);
  for (std::size_t a = 0; a < alpha.size(); ++a)
    for (std::size_t b = 0; b < beta.size(); ++b) {
      const std::size_t product = a * beta.size() + b;
      product_alpha.push_back(alpha[a]);
      product_beta.push_back(beta[b]);
      if (alpha[a] == Strings::value_type{0, 1} &&
          beta[b] == Strings::value_type{2, 3}) {
        selected(product, 0) = 1.0;
        selected(product, 1) = 0.3;
      } else if (alpha[a] == Strings::value_type{0, 2} &&
                 beta[b] == Strings::value_type{1, 3}) {
        selected(product, 0) = 0.2;
        selected(product, 1) = 1.0;
        selected(product, 2) = -0.2;
      } else if (alpha[a] == Strings::value_type{0, 3} &&
                 beta[b] == Strings::value_type{1, 2}) {
        selected(product, 1) = 0.15;
        selected(product, 2) = 1.0;
      } else if (alpha[a] == Strings::value_type{2, 3} &&
                 beta[b] == Strings::value_type{0, 1}) {
        selected(product, 0) = -0.1;
        selected(product, 2) = 0.35;
      } else if (alpha[a] == Strings::value_type{0, 1} &&
                 beta[b] == Strings::value_type{0, 1}) {
        selected(product, 3) = 1.1;
      }
      for (int column = 0; column < selected.cols(); ++column)
        if (selected(product, column) != 0.0)
          terms[product].push_back({column, selected(product, column)});
    }
  const auto topology = xmvb::vb::build_same_spin_pair_topology(
      product_alpha, product_beta, n);
  const Matrix one_electron = Matrix::Zero(n, n);
  xmvb::integrals::ActiveSpaceTwoElectronResult eri;
  eri.packed_active_two_electron_integrals.assign(
      xmvb::integrals::packed_active_two_electron_integral_count(n), 0.0);
  const Matrix overlap = Matrix::Identity(n, n);
  const std::vector<double> overlap_data(overlap.data(), overlap.data() + overlap.size());
  const xmvb::vb::StructureAction action(
      terms, 4, topology, overlap_data, one_electron, eri, n, nullptr,
      xmvb::vb::StructureActionMode::ActionOnly);
  const xmvb::vb::SpinMetric cpu(action, alpha, beta, 10 * sizeof(double));
  require(cpu.block_count() == 2 && cpu.max_block_size() == 3,
          "spin metric fixture did not form a three-structure occupation block");
  const Matrix gram = selected.transpose() * selected;
  require(std::abs(gram(0, 1)) > 0.01 && std::abs(gram(1, 2)) > 0.01,
          "spin metric fixture lost nontrivial recoupling");
  const Matrix rhs = values(4, rhs_count, 0.43);
  const Eigen::VectorXd positive =
      (cpu.diagonal().array() * Eigen::Array4d(1.2, 0.8, 1.5, 0.9)).matrix();

  xmvb::cuda::Context context(64ULL * 1024ULL * 1024ULL);
  const xmvb::cuda::SpinMetric gpu(context, action, alpha, beta, 10 * sizeof(double));
  xmvb::cuda::Matrix<double> d_rhs(context, rhs.rows(), rhs.cols());
  xmvb::cuda::Matrix<double> d_out(context, rhs.rows(), rhs.cols());
  xmvb::cuda::Matrix<double> d_positive(context, positive.size(), 1);
  xmvb::cuda::copy_to_device(context, d_rhs.view(), rhs);
  xmvb::cuda::copy_to_device(context, d_positive.view(), positive);
  for (const auto part : {xmvb::vb::SpinMetric::Part::Gram,
                          xmvb::vb::SpinMetric::Part::Lower,
                          xmvb::vb::SpinMetric::Part::Upper}) {
    gpu.solve(d_rhs.view(), d_out.view(), part);
    near(download(context, d_out), cpu.solve(rhs, part), 2.0e-11,
         "CUDA spin metric factor solve differs from CPU");
  }
  gpu.solve_diagonal_model(d_rhs.view(), d_positive.view(), d_out.view());
  near(download(context, d_out), cpu.solve_diagonal_model(rhs, positive),
       2.0e-11, "CUDA spin metric diagonal-model solve differs from CPU");
  near(gram * cpu.solve(rhs), rhs, 2.0e-11,
       "spin metric fixture has an inconsistent selected Gram");
}

void check_exterior(int rhs_count,int n=4,int nalpha=2,int nbeta=1) {
  const Strings alpha = xmvb::mcscf::SpinStrings(n, nalpha).occupations();
  const Strings beta = xmvb::mcscf::SpinStrings(n, nbeta).occupations();
  Matrix upper = Matrix::Identity(n, n);
  for (int p = 0; p < n; ++p) {
    upper(p, p) = 0.86 + 0.11 * p;
    for (int q = p + 1; q < n; ++q)
      upper(p, q) = 0.045 * (p + 1) * (q + 2)*16.0/(n*n);
  }
  const xmvb::vb::ExteriorOrbitalTransform cpu_alpha(alpha, upper);
  const xmvb::vb::ExteriorOrbitalTransform cpu_beta(beta, upper);
  xmvb::cuda::Context context(64ULL * 1024ULL * 1024ULL);
  const xmvb::cuda::ExteriorTransform gpu_alpha(context, cpu_alpha);
  const xmvb::cuda::ExteriorTransform gpu_beta(context, cpu_beta);
  const Matrix seed = values(alpha.size(), beta.size() * rhs_count, 0.34);
  xmvb::cuda::Matrix<double> device(context, seed.rows(), seed.cols());
  const auto cpu_right = [&](Matrix* panel, bool adjoint) {
    for (int block = 0; block < rhs_count; ++block) {
      auto rhs = panel->middleCols(block * beta.size(), beta.size());
      if (adjoint) cpu_beta.apply_adjoint_right_block(rhs);
      else cpu_beta.apply_right_block(rhs);
    }
  };

  Matrix forward = seed;
  cpu_alpha.apply_left(&forward);
  cpu_right(&forward, false);
  xmvb::cuda::copy_to_device(context, device.view(), seed);
  gpu_alpha.apply_left(device.view());
  gpu_beta.apply_right(device.view());
  near(download(context, device), forward, 2.0e-11,
       "CUDA exterior forward transform differs from CPU");
  {
    Matrix padded=Matrix::Constant(seed.rows()+2,seed.cols(),0.125);
    padded.topRows(seed.rows())=seed;
    xmvb::cuda::Matrix<double> storage(context,padded.rows(),padded.cols());
    xmvb::cuda::copy_to_device(context,storage.view(),padded);
    const xmvb::linalg::MatrixView view{storage.data(),seed.rows(),seed.cols(),padded.rows()};
    gpu_alpha.apply_left(view);
    gpu_beta.apply_right(view);
    padded.topRows(seed.rows())=forward;
    near(download(context,storage),padded,2.0e-11,
        "CUDA exterior tiles changed padded rows or partial row lanes");
  }
  gpu_beta.apply_right(device.view(), false, true);
  gpu_alpha.apply_left(device.view(), false, true);
  near(download(context, device), seed, 2.0e-11,
       "CUDA exterior inverse does not undo forward transform");

  Matrix adjoint = seed;
  cpu_alpha.apply_adjoint_left(&adjoint);
  cpu_right(&adjoint, true);
  xmvb::cuda::copy_to_device(context, device.view(), seed);
  gpu_alpha.apply_left(device.view(), true);
  gpu_beta.apply_right(device.view(), true);
  const Matrix gpu_adjoint = download(context, device);
  near(gpu_adjoint, adjoint, 2.0e-11,
       "CUDA exterior adjoint transform differs from CPU");
  gpu_beta.apply_right(device.view(), true, true);
  gpu_alpha.apply_left(device.view(), true, true);
  near(download(context, device), seed, 2.0e-11,
       "CUDA exterior inverse adjoint does not undo adjoint transform");

  const Matrix left = values(seed.rows(), seed.cols(), 0.91);
  Matrix cpu_left_adjoint = left;
  cpu_right(&cpu_left_adjoint, true);
  cpu_alpha.apply_adjoint_left(&cpu_left_adjoint);
  xmvb::cuda::copy_to_device(context, device.view(), left);
  gpu_beta.apply_right(device.view(), true);
  gpu_alpha.apply_left(device.view(), true);
  near(download(context, device), cpu_left_adjoint, 2.0e-11,
       "CUDA exterior adjoint ordering differs from CPU");
  require(std::abs(bilinear(left, forward) -
                   bilinear(cpu_left_adjoint, seed)) <
              2.0e-11 * std::max(1.0, forward.norm()),
          "exterior transform violates forward/adjoint pairing");
}

void check_cholesky(int rhs_count) {
  constexpr int n = 4;
  const Matrix raw = values(n, n, 0.74);
  const Matrix spd = (raw.transpose() * raw +
                      0.7 * Matrix::Identity(n, n)).eval();
  const Matrix rhs = values(n, rhs_count, 0.48);
  const Matrix reference = spd.llt().solve(rhs);
  xmvb::cuda::Context context(16ULL * 1024ULL * 1024ULL);
  Matrix padded_spd = Matrix::Zero(n + 2, n);
  padded_spd.topRows(n) = spd;
  xmvb::cuda::Matrix<double> d_spd(context, n + 2, n);
  xmvb::cuda::copy_to_device(context, d_spd.view(), padded_spd);
  const xmvb::linalg::ConstMatrixView metric_view{d_spd.data(), n, n, n + 2};
  const xmvb::cuda::Cholesky factor(context, metric_view);
  require(factor.dimension() == n, "CUDA Cholesky dimension changed");

  Matrix padded_rhs = Matrix::Constant(n + 3, rhs_count, -9.0);
  padded_rhs.topRows(n) = rhs;
  xmvb::cuda::Matrix<double> d_rhs(context, n + 3, rhs_count);
  xmvb::cuda::Matrix<double> d_out(context, n + 4, rhs_count);
  xmvb::cuda::copy_to_device(context, d_rhs.view(), padded_rhs);
  const xmvb::linalg::ConstMatrixView rhs_view{
      d_rhs.data(), n, rhs_count, n + 3};
  const xmvb::linalg::MatrixView out_view{
      d_out.data(), n, rhs_count, n + 4};
  factor.solve(rhs_view, out_view);
  const Matrix answer = download(context, d_out).topRows(n);
  near(answer, reference, 2.0e-11,
       "CUDA Cholesky strided RHS solve differs from CPU");
  near(spd * answer, rhs, 2.0e-11,
       "CUDA Cholesky strided RHS has a nonzero original residual");

  const xmvb::linalg::MatrixView in_place{
      d_rhs.data(), n, rhs_count, n + 3};
  factor.solve(rhs_view, in_place);
  near(download(context, d_rhs).topRows(n), reference, 2.0e-11,
       "CUDA Cholesky in-place solve differs from CPU");
}

void check_full_metric(int rhs_count) {
  constexpr int n = 2;
  const Strings alpha = xmvb::mcscf::SpinStrings(n, 1).occupations();
  const Strings beta = alpha;
  const double singlet_weight = 1.0 / std::sqrt(2.0);
  // Three independent complete singlet-sector columns: ionic 00, covalent
  // (01+10)/sqrt(2), ionic 11. The spatial occupations are definite.
  std::vector<std::vector<xmvb::vb::StructureExpansionTerm>> terms(4);
  terms[0].push_back({0, 1.0});
  terms[1].push_back({1, singlet_weight});
  terms[2].push_back({1, singlet_weight});
  terms[3].push_back({2, 1.0});
  const Strings product_alpha{alpha[0], alpha[0], alpha[1], alpha[1]};
  const Strings product_beta{beta[0], beta[1], beta[0], beta[1]};
  const auto topology = xmvb::vb::build_same_spin_pair_topology(
      product_alpha, product_beta, n);
  Matrix upper = Matrix::Identity(n, n);
  upper(0, 0) = 1.08;
  upper(0, 1) = 0.19;
  upper(1, 1) = 0.91;
  const Matrix overlap = upper.transpose() * upper;
  const std::vector<double> overlap_data(
      overlap.data(), overlap.data() + overlap.size());
  Matrix one_electron(n, n);
  one_electron << -0.84, 0.13, 0.13, -0.27;
  xmvb::integrals::ActiveSpaceTwoElectronResult eri;
  auto& packed = eri.packed_active_two_electron_integrals;
  packed.resize(xmvb::integrals::packed_active_two_electron_integral_count(n));
  for (std::size_t i = 0; i < packed.size(); ++i)
    packed[i] = 0.06 * std::cos(0.41 * (i + 1));
  const xmvb::vb::StructureAction source(
      terms, 3, topology, overlap_data, one_electron, eri, n, nullptr,
      xmvb::vb::StructureActionMode::ActionOnly);
  require(source.storage().orthogonal_direct_ci,
          "FULL fixture lacks its exact direct-CI carrier");
  const xmvb::vb::StructureMetricInverse cpu(
      source, alpha, beta, true, 3 * sizeof(double), 4096);
  const Matrix vectors = values(3, rhs_count, 0.66);
  Eigen::VectorXd energies(rhs_count);
  for (int col = 0; col < rhs_count; ++col)
    energies[col] = -5.0 - 0.2 * col;
  Matrix energy_row(1, rhs_count);
  energy_row.row(0) = energies.transpose();

  xmvb::cuda::Context context(128ULL * 1024ULL * 1024ULL);
  const xmvb::cuda::SpinMetric spin(context, source, alpha, beta, 3 * sizeof(double));
  const xmvb::cuda::StructureAction action(context, source);
  const xmvb::cuda::FullMetric full(context, spin, action);
  xmvb::cuda::Matrix<double> d_vectors(context, 3, rhs_count);
  xmvb::cuda::Matrix<double> d_out(context, 3, rhs_count);
  xmvb::cuda::Matrix<double> d_energies(context, 1, rhs_count);
  xmvb::cuda::copy_to_device(context, d_vectors.view(), vectors);
  xmvb::cuda::copy_to_device(context, d_energies.view(), energy_row);

  full.to_ci_coordinates(d_vectors.view(), d_out.view());
  near(download(context, d_out), cpu.to_ci_coordinates(vectors), 3.0e-10,
       "CUDA FULL entrance coordinates differ from CPU");
  xmvb::cuda::Matrix<double> d_round_trip(context, 3, rhs_count);
  full.from_ci_coordinates(d_out.view(), d_round_trip.view());
  near(download(context, d_round_trip), vectors, 3.0e-10,
       "CUDA FULL coordinate round trip changed structure coefficients");
  full.from_ci_coordinates(d_vectors.view(), d_out.view());
  near(download(context, d_out), cpu.from_ci_coordinates(vectors), 3.0e-10,
       "CUDA FULL exit coordinates differ from CPU");
  full.ci_covector_to_structure(d_vectors.view(), d_out.view());
  near(download(context, d_out), cpu.ci_covector_to_structure(vectors),
       3.0e-10, "CUDA FULL covector pullback differs from CPU");
  full.apply_ci_hamiltonian(d_vectors.view(), d_out.view());
  near(download(context, d_out), cpu.apply_ci_hamiltonian(vectors),
       3.0e-10, "CUDA FULL CI Hamiltonian differs from CPU");
  full.apply_ci_preconditioner(d_vectors.view(), d_energies.view(), 0.0,
                               d_out.view());
  near(download(context, d_out),
       cpu.apply_ci_preconditioner(vectors, energies, 0.0), 3.0e-10,
       "CUDA FULL CI preconditioner differs from CPU");
}

}  // namespace

int main() {
  try {
    for (bool rotated : {false, true})
      for (int rhs_count : {1, 3}) {
        check_sigma(rhs_count,rotated);
        check_sigma(rhs_count,rotated,1);
        check_sigma(rhs_count,rotated,1,true);
      }
    for (int rhs_count : {1, 3}) check_spin_metric(rhs_count);
    for (int rhs_count : {1, 3}) check_exterior(rhs_count);
    check_exterior(1,12,6,5); // Unequal wide spin sectors in shared tiles.
    check_exterior(1,16,8,1); // Left column exceeds block shared capacity.
    for (int rhs_count : {1, 3}) check_cholesky(rhs_count);
    for (int rhs_count : {1, 3}) check_full_metric(rhs_count);
    std::cout << "CUDA production CI sigma/adjoint/metric/exterior/FULL parity: PASS\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
