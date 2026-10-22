#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "core/pair_index.hpp"
#include "cuda/ci/sigma.hpp"
#include "cuda/linalg/cholesky.hpp"
#include "cuda/memory/buffer.hpp"
#include "cuda/memory/context.hpp"
#include "cuda/memory/graph.hpp"
#include "cuda/vb/action.hpp"
#include "cuda/vb/columns.hpp"
#include "cuda/vb/metric.hpp"
#include "cuda/vb/transform.hpp"
#include "integrals/active/two_electron.hpp"
#include "mcscf/ci/integrals.hpp"
#include "mcscf/ci/sigma.hpp"
#include "mcscf/ci/product_support.hpp"
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

void check_compact_endpoints() {
  using namespace xmvb::cuda;
  Context context(8ULL*1024*1024);
  const auto pattern=[&](std::vector<int> rows,bool full=false,int size=0) {
    auto result=std::make_shared<ProductColumnPattern>();
    result->full=full;result->size=full ? size:rows.size();
    result->rows=std::move(rows);
    if (!full) {
      result->device_rows=Buffer<int>(context,result->rows.size());
      context.copy_to_device(result->device_rows.data(),result->rows.data(),
          result->rows.size()*sizeof(int));
    }
    return result;
  };
  const std::vector<std::shared_ptr<const ProductColumnPattern>> patterns{
      pattern({0,3,8}),pattern({}),pattern({1,7})};
  ProductColumns compact(context,patterns,2);
  const Matrix source=values(9,6,0.71);
  xmvb::cuda::Matrix<double> dense(context,9,6),unpacked(context,9,6);
  copy_to_device(context,dense.view(),source);
  compact.pack(context,dense.view(),false);
  check_cuda(cudaMemsetAsync(unpacked.data(),0,9*6*sizeof(double),context.stream()),
      "clear compact endpoint oracle");
  compact.pack(context,unpacked.view(),true);
  Matrix expected=Matrix::Zero(9,6);
  for (int v=0;v<2;++v) for (int b=0;b<3;++b)
    for (int row:patterns[b]->rows) expected(row,3*v+b)=source(row,3*v+b);
  near(download(context,unpacked),expected,0.0,
      "compact endpoint loses exact holes or RHS lanes");
  require(compact.values.rows()==5 && compact.values.cols()==2,
      "compact endpoint allocated the full product rectangle");
  const auto bytes=compact.bytes();
  ProductImages retained(2*bytes);
  retained.begin(2);
  require(retained.retain(0,std::move(compact)),"accepted endpoint rejected fitting tile");
  bool incomplete=false;
  try { retained.at(0); } catch (const std::logic_error&) { incomplete=true; }
  require(incomplete,"partial endpoint was exposed to readers");
  ProductColumns other(context,patterns,2);
  other.pack(context,dense.view(),false);
  require(retained.retain(1,std::move(other)),"accepted endpoint rejected exact budget");
  retained.finish();
  require(retained.complete() && retained.bytes()==2*bytes,
      "accepted endpoint budget or completeness changed");
  const auto& first=retained.at(0);
  require(first.patterns[0]==retained.at(1).patterns[0],
      "accepted lanes do not share immutable row metadata");

  ProductImages bounded(bytes);
  bounded.begin(2);
  require(bounded.retain(0,ProductColumns(context,patterns,2)),
      "bounded endpoint rejected fitting first tile");
  require(!bounded.retain(1,ProductColumns(context,patterns,2)),
      "bounded endpoint exceeded its allowance");
  bounded.finish();
  require(!bounded.complete() && bounded.bytes()==0,
      "over-budget endpoint left partial accepted data resident");

  ProductColumns full(context,{pattern({},true,9)},2);
  xmvb::cuda::Matrix<double> full_input(context,9,2),full_output(context,9,2);
  copy_to_device(context,full_input.view(),source.leftCols(2));
  full.pack(context,full_input.view(),false);
  full.pack(context,full_output.view(),true);
  near(download(context,full_output),source.leftCols(2),0.0,
      "full column endpoint changed its implicit row identity");
  require(full.patterns[0]->device_rows.empty(),
      "full column allocated redundant identity row indices");

  // C16 spin dimension, but only two touched entries: the endpoint contract
  // must not secretly allocate any 12870-by-12870 rectangular carrier.
  const auto before=context.counters().current_device_bytes;
  ProductColumns sparse(context,{pattern({0,12869})},2);
  require(context.counters().current_device_bytes-before<1024,
      "compact endpoint scales with the enclosing spin-product rectangle");
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

/** Exercise the production tile executor against the CPU tile chain, including
 * out-of-support halo targets, exchange weights and two fused response lanes.
 * Full carriers are test oracles only; tile callbacks move device values.
 */
void check_product_tiles(int parity,bool covalent,std::size_t budget) {
  constexpr int n=4,width=2;
  const xmvb::mcscf::SpinStrings as(n,2),bs(n,parity || covalent ? 2:1);
  const auto base=active_integrals(n,false),delta=active_integrals(n,true);
  const xmvb::mcscf::DirectCiSigmaAction cpu(as.occupations(),bs.occupations(),base);
  const xmvb::mcscf::DirectCiSigmaAction change(cpu,delta);
  std::unique_ptr<xmvb::mcscf::ProductSupport> support;
  if (covalent) support=std::make_unique<xmvb::mcscf::ProductSupport>(n,as.masks(),bs.masks());
  else support=std::make_unique<xmvb::mcscf::ProductSupport>(n,as.masks(),bs.masks(),
      std::vector<xmvb::mcscf::ProductIndex>{0,static_cast<xmvb::mcscf::ProductIndex>(bs.size()+1)});
  Matrix left=values(as.size(),bs.size()*width,0.28),right=values(as.size(),bs.size()*width,0.71);
  for (Matrix* values:{&left,&right}) for (int v=0;v<width;++v) {
    if (parity) {
      const Matrix block=values->middleCols(v*bs.size(),bs.size());
      values->middleCols(v*bs.size(),bs.size())=(0.5*(block+parity*block.transpose())).eval();
    }
    for (int b=0;b<bs.size();++b) for (int a=0;a<as.size();++a)
      if (!support->contains(a,b)) (*values)(a,v*bs.size()+b)=0.0;
  }
  const auto host_reader=[&](const Matrix& values) -> xmvb::mcscf::DirectCiTileReader {
    return [&,source=&values](auto betas,auto rows,Eigen::Ref<Matrix> target) {
      for (int b=0;b<betas.size();++b) for (int v=0;v<width;++v) for (int a=0;a<rows.size();++a)
        target(a,b*width+v)=(*source)(rows[a],v*bs.size()+betas[b]);
    };
  };
  Matrix expected=Matrix::Zero(left.rows(),left.cols());
  const xmvb::mcscf::DirectCiTileWriter host_writer=[&](auto betas,auto rows,const auto& accumulate) {
    for (int b=0;b<betas.size();++b) for (int v=0;v<width;++v)
      accumulate(b,v,0,rows.size(),expected.col(v*bs.size()+betas[b]).data(),nullptr,0);
  };
  const auto expected_adjoint=cpu.apply_tiled_direction(width,host_reader(left),host_reader(right),change,
      host_writer,{},support.get(),parity,{},budget);
  xmvb::cuda::Context context(128ULL*1024*1024);
  xmvb::cuda::DirectCiSigma gpu(context,cpu,1,parity),direction(context,gpu);
  xmvb::cuda::Matrix<double> dh(context,n,n),dg(context,n*(n+1)/2,n*(n+1)/2);
  xmvb::cuda::copy_to_device(context,dh.view(),delta.one_electron);
  xmvb::cuda::copy_to_device(context,dg.view(),delta.pair_kernel);
  direction.update_integrals(dh.view(),dg.view());
  xmvb::cuda::Matrix<double> dl(context,left.rows(),left.cols()),dr(context,right.rows(),right.cols()),
      out(context,left.rows(),left.cols()),second(context,left.rows(),left.cols()),h2(context,n,n),g2(context,dg.rows(),dg.cols());
  xmvb::cuda::copy_to_device(context,dl.view(),left);
  xmvb::cuda::copy_to_device(context,dr.view(),right);
  const auto device_reader=[&](xmvb::linalg::ConstMatrixView source) -> xmvb::cuda::CiTileReader {
    return [&,source](auto betas,auto rows,xmvb::linalg::MatrixView target) {
      for (int b=0;b<betas.size();++b) for (int v=0;v<width;++v) for (int a=0;a<rows.size();++a)
        xmvb::cuda::check_cuda(cudaMemcpyAsync(target.data+a+target.ld*(b*width+v),
            source.data+rows[a]+source.ld*(v*bs.size()+betas[b]),sizeof(double),
            cudaMemcpyDeviceToDevice,context.stream()),"test tile reader");
    };
  };
  const auto device_writer=[&](xmvb::linalg::MatrixView destination) -> xmvb::cuda::CiTileWriter {
    xmvb::cuda::check_cuda(cudaMemsetAsync(destination.data,0,destination.rows*destination.cols*sizeof(double),
        context.stream()),"clear tile oracle");
    return [&,destination](auto betas,auto rows,const xmvb::cuda::CiTileGather& gather) {
      xmvb::cuda::Matrix<double> tile(context,rows.size(),betas.size()*width);
      const xmvb::cuda::CiTileColumn target{-1,nullptr,tile.view()};
      gather({&target,1},false);
      const double one=1;
      for (int b=0;b<betas.size();++b) for (int v=0;v<width;++v) for (int a=0;a<rows.size();++a)
        xmvb::cuda::check_cublas(cublasDaxpy(context.blas(),1,&one,tile.data()+a+tile.rows()*(b*width+v),1,
            destination.data+rows[a]+destination.ld*(v*bs.size()+betas[b]),1),"test tile writer");
    };
  };
  const std::array<xmvb::cuda::CiTileRequest,2> requests{{
      {device_reader(dl.view()),device_writer(out.view()),&direction,dh.view(),dg.view()},
      {device_reader(dl.view()),device_writer(second.view()),nullptr,h2.view(),g2.view()}}};
  gpu.apply_tiled_responses(width,requests,device_reader(dr.view()),support.get(),budget);
  near(download(context,out),expected,3e-10,"GPU directional product tiles differ from CPU");
  near(download(context,dh),expected_adjoint.one_electron,3e-10,"GPU tiled h adjoint differs from CPU");
  near(download(context,dg),expected_adjoint.pair_kernel,3e-10,"GPU tiled g adjoint differs from CPU");
  expected.setZero();
  cpu.apply_tiled(width,host_reader(left),host_writer,support.get(),parity,{},budget);
  near(download(context,second),expected,3e-10,"GPU second response lost the shared state scatter");
  near(download(context,h2),expected_adjoint.one_electron,3e-10,"GPU second response adjoint differs");
  near(download(context,g2),expected_adjoint.pair_kernel,3e-10,"GPU second response pair adjoint differs");
  const std::array<xmvb::cuda::CiCarrierRequest,2> carriers{{
      {dl.view(),out.view(),&direction,dh.view(),dg.view()},
      {dl.view(),second.view(),nullptr,h2.view(),g2.view()}}};
  gpu.apply_supported_responses(carriers,dr.view(),*support,budget);
  near(download(context,out),cpu.apply(left)+change.apply(right),3e-10,
      "supported resident HVP endpoint is not the full physical image");
  near(download(context,second),cpu.apply(left),3e-10,
      "supported resident CI endpoint is not the full physical image");
  const auto full_adjoint=cpu.integral_adjoint(left,right);
  near(download(context,dh),full_adjoint.one_electron,3e-10,"supported endpoint h adjoint differs from full sigma");
  near(download(context,dg),full_adjoint.pair_kernel,3e-10,"supported endpoint g adjoint differs from full sigma");
  const xmvb::cuda::CiCarrierRequest same{dr.view(),out.view(),nullptr,dh.view(),dg.view()};
  gpu.apply_supported_responses({&same,1},dr.view(),*support,budget);
  near(download(context,out),cpu.apply(right),3e-10,"accepted same-source supported sigma differs");
  const auto density=cpu.integral_adjoint(right,right);
  near(download(context,dh),density.one_electron,3e-10,"accepted same-source supported h density differs");
  near(download(context,dg),density.pair_kernel,3e-10,"accepted same-source supported g density differs");
  // Repeat the symbolic tile with different amplitudes. Only indices may be
  // reused: neither scattered sources nor the weighted Gram are immutable.
  xmvb::cuda::copy_to_device(context,dr.view(),(-0.7*right).eval());
  gpu.apply_supported_responses({&same,1},dr.view(),*support,budget);
  near(download(context,out),-0.7*cpu.apply(right),3e-10,"reused tile retained old amplitudes");
  near(download(context,dh),0.49*density.one_electron,3e-10,"reused weighted Gram retained old h density");
  near(download(context,dg),0.49*density.pair_kernel,3e-10,"reused weighted Gram retained old g density");
}

void check_sigma(int rhs_count,bool rotated,int maximum_tile=3,bool one_product=false,int n=4) {
  const xmvb::mcscf::SpinStrings alpha_space(n,one_product ? n:2),
      beta_space(n,one_product ? 0:1);
  const Strings alpha = alpha_space.occupations(), beta = beta_space.occupations();
  const auto integrals = active_integrals(n, rotated);
  const xmvb::mcscf::DirectCiSigmaAction cpu(alpha, beta, integrals);
  const Matrix left = values(alpha.size(), beta.size() * rhs_count, 0.26);
  const Matrix right = values(alpha.size(), beta.size() * rhs_count, 0.82);
  xmvb::cuda::Context context(128ULL * 1024ULL * 1024ULL);
  if (n==4 && rhs_count==1 && !rotated && !one_product && maximum_tile==3) {
    bool rejected=false;
    try { xmvb::cuda::DirectCiSigma invalid(context,cpu,maximum_tile,1); }
    catch (const std::invalid_argument&) {rejected=true;}
    require(rejected,"CUDA exchange sector was accepted for unequal spin graphs");
  }
  xmvb::cuda::DirectCiSigma gpu(context,cpu,maximum_tile);
  require(one_product || gpu.n_alpha_strings() != gpu.n_beta_strings(),
          "CUDA direct-CI fixture lost unequal spin dimensions");
  xmvb::cuda::Matrix<double> d_left(context, left.rows(), left.cols());
  xmvb::cuda::Matrix<double> d_right(context, right.rows(), right.cols());
  xmvb::cuda::Matrix<double> d_image(context, left.rows(), left.cols());
  xmvb::cuda::Matrix<double> d_h(context, n, n);
  const int pairs = n * (n + 1) / 2;
  const auto sigma_bytes=gpu.resident_bytes();
  gpu.prepare_with_adjoint(rhs_count);
  const auto panel_bytes=gpu.resident_bytes()-sigma_bytes-pairs*pairs*sizeof(double);
  require(panel_bytes%(3*pairs*sizeof(double))==0,
      "CUDA fused sigma allocated an unused full-carrier panel tail");
  gpu.reset_workspace();
  xmvb::cuda::Matrix<double> d_g(context, pairs, pairs);
  xmvb::cuda::copy_to_device(context, d_left.view(), left);
  xmvb::cuda::copy_to_device(context, d_right.view(), right);

  gpu.apply(d_left.view(),d_image.view());
  near(download(context,d_image),cpu.apply(left),3.0e-10,
       "CUDA source-row sigma differs from CPU");
  gpu.apply(d_right.view(),d_image.view(),true);
  near(download(context,d_image),cpu.apply(left)+cpu.apply(right),3.0e-10,
       "CUDA source-row sigma add mode differs from CPU");

  // Clearing the scattered destination must not touch padded rows. Both
  // overwrite and accumulation modes must honor the caller's leading dimension.
  xmvb::cuda::Matrix<double> padded(context,left.rows()+3,left.cols());
  const Matrix sentinels=Matrix::Constant(padded.rows(),padded.cols(),7.25);
  xmvb::cuda::copy_to_device(context,padded.view(),sentinels);
  const xmvb::linalg::MatrixView padded_image{
      padded.data(),left.rows(),left.cols(),padded.rows()};
  gpu.apply(d_left.view(),padded_image);
  auto padded_result=download(context,padded);
  near(padded_result.topRows(left.rows()),cpu.apply(left),3.0e-10,
       "CUDA source-scattered sigma ignores output stride");
  near(padded_result.bottomRows(3),sentinels.bottomRows(3),0.0,
       "CUDA source-scattered sigma cleared padded rows");
  gpu.apply(d_right.view(),padded_image,true);
  padded_result=download(context,padded);
  near(padded_result.topRows(left.rows()),cpu.apply(left)+cpu.apply(right),3.0e-10,
       "CUDA source-scattered sigma add mode ignores output stride");
  near(padded_result.bottomRows(3),sentinels.bottomRows(3),0.0,
       "CUDA source-scattered sigma add mode changed padded rows");

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
  auto fused_rows=std::max<std::int64_t>(1,
      left.size()*std::min(pairs,maximum_tile)/pairs);
  if (fused_rows>=32) fused_rows=(fused_rows/32)*32;
  require(gpu.resident_bytes()-sigma_bytes==sizeof(double)*
      (static_cast<std::size_t>(pairs)*pairs+2*fused_rows*pairs),
      "CUDA fixed-kernel fused sigma retained an unnecessary third panel");

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
  // Directional HVP needs three shared panels; the fixed-kernel fused action
  // above now uses only two. Prepare this phase before measuring clone reuse.
  gpu.prepare_with_adjoint(rhs_count);
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
  gpu.generator_moments(std::array<xmvb::linalg::ConstMatrixView,1>{d_left.view()},
      d_right.view(),std::array<xmvb::linalg::MatrixView,1>{d_generator_adjoint.view()});
  require(gpu.resident_bytes()-fused_bytes <= sizeof(double)*
          (2*static_cast<std::size_t>(pairs)+left.size()),
          "CUDA generator adjoint duplicated a fused channel slab");
  const Matrix generator_adjoint=download(context,d_generator_adjoint);
  near(generator_adjoint,cpu.one_body_generator_adjoint(left,right),3.0e-10,
       "CUDA nonsymmetric generator adjoint differs from CPU");
  const Matrix mixed_left=0.37*left-0.19*right;
  xmvb::cuda::Matrix<double> d_mixed_left(context,left.rows(),left.cols());
  xmvb::cuda::Matrix<double> d_moments(context,n,3*n);
  xmvb::cuda::copy_to_device(context,d_mixed_left.view(),mixed_left);
  gpu.generator_moments(
      std::array<xmvb::linalg::ConstMatrixView,3>{
          d_left.view(),d_right.view(),d_mixed_left.view()},d_right.view(),
      std::array<xmvb::linalg::MatrixView,3>{d_moments.view().columns(0,n),
          d_moments.view().columns(n,n),d_moments.view().columns(2*n,n)});
  const Matrix moments=download(context,d_moments);
  near(moments.leftCols(n),generator_adjoint,3.0e-10,"CUDA batched left moment");
  near(moments.middleCols(n,n),cpu.one_body_generator_adjoint(right,right),
       3.0e-10,"CUDA batched equal-carrier moment");
  near(moments.rightCols(n),cpu.one_body_generator_adjoint(mixed_left,right),
       3.0e-10,"CUDA batched mixed-carrier moment");
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
  captured_generator.generator_moments(
      std::array<xmvb::linalg::ConstMatrixView,1>{d_left.view()},d_right.view(),
      std::array<xmvb::linalg::MatrixView,1>{d_generator_adjoint.view()});
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

void check_exchange_parity(int n,int rhs,int parity,int capacity) {
  const auto strings=xmvb::mcscf::SpinStrings(n,n/2).occupations();
  const auto base=active_integrals(n,false);
  const xmvb::mcscf::DirectCiSigmaAction cpu(strings,strings,base);
  const int dim=strings.size(),pairs=n*(n+1)/2;
  const auto sector=[&](double phase) {
    Matrix result=values(dim,dim*rhs,phase);
    for (int b=0;b<rhs;++b) {
      auto block=result.middleCols(b*dim,dim);
      block=(0.5*(block+parity*block.transpose())).eval();
    }
    return result;
  };
  const Matrix left=sector(0.26),right=sector(0.82);
  const auto expected=cpu.apply_with_adjoint(left,right);
  near(cpu.apply(left,parity),expected.sigma,3e-10,
      "CPU exchange sector disagrees with the unrestricted oracle");
  xmvb::cuda::Context context(128ULL*1024*1024);
  xmvb::cuda::DirectCiSigma gpu(context,cpu,capacity,parity);
  require(gpu.exchange_parity()==parity,"CUDA sigma lost its certified sector");
  xmvb::cuda::Matrix<double> dl(context,dim+1,dim*rhs),dr(context,dim+2,dim*rhs);
  xmvb::cuda::Matrix<double> image(context,dim+3,dim*rhs);
  xmvb::cuda::Matrix<double> h(context,n,n),g(context,pairs,pairs),moments(context,n,2*n);
  Matrix padded_left=Matrix::Constant(dim+1,dim*rhs,19.0);
  Matrix padded_right=Matrix::Constant(dim+2,dim*rhs,23.0);
  padded_left.topRows(dim)=left;padded_right.topRows(dim)=right;
  xmvb::cuda::copy_to_device(context,dl.view(),padded_left);
  xmvb::cuda::copy_to_device(context,dr.view(),padded_right);
  xmvb::cuda::copy_to_device(context,image.view(),Matrix::Constant(dim+3,dim*rhs,-17.0));
  const xmvb::linalg::ConstMatrixView l{dl.data(),dim,dim*rhs,dim+1};
  const xmvb::linalg::ConstMatrixView r{dr.data(),dim,dim*rhs,dim+2};
  const xmvb::linalg::MatrixView out{image.data(),dim,dim*rhs,dim+3};
  const auto read=[&] {return download(context,image).topRows(dim).eval();};
  const auto resident=gpu.resident_bytes();
  gpu.prepare(rhs);
  auto rows=std::max<std::int64_t>(1,
      static_cast<std::int64_t>(dim)*(dim+1)/2*rhs*std::min(pairs,capacity)/pairs);
  if (rows>=32) rows=(rows/32)*32;
  require(gpu.resident_bytes()-resident==2*rows*pairs*sizeof(double),
      "CUDA certified sigma still allocated full-product source panels");
  gpu.apply(l,out);
  near(read(),expected.sigma,3e-10,"CUDA exchange sigma differs from full CPU action");
  gpu.apply(r,out,true);
  near(read(),expected.sigma+cpu.apply(right),3e-10,"CUDA exchange sigma add mode differs");
  gpu.apply_with_adjoint(l,r,out,h.view(),g.view());
  near(read(),expected.sigma,3e-10,"CUDA exchange fused sigma differs");
  near(download(context,h),expected.integrals.one_electron,3e-10,
      "CUDA exchange one-electron adjoint has wrong row weights");
  near(download(context,g),expected.integrals.pair_kernel,3e-10,
      "CUDA exchange pair adjoint has wrong row weights");
  const auto same=cpu.apply_with_adjoint(right,right);
  gpu.apply_with_adjoint(r,r,out,h.view(),g.view());
  near(read(),same.sigma,3e-10,"CUDA same-source exchange sigma differs");
  near(download(context,g),same.integrals.pair_kernel,3e-10,
      "CUDA same-source exchange adjoint squared the row weights");
  const xmvb::mcscf::ActiveIntegrals delta{
      symmetric_values(n,1.74),0.11*symmetric_values(pairs,2.36)};
  const xmvb::mcscf::DirectCiSigmaAction direction(strings,strings,delta);
  xmvb::cuda::DirectCiSigma dgpu(context,gpu);
  xmvb::cuda::copy_to_device(context,h.view(),delta.one_electron);
  xmvb::cuda::copy_to_device(context,g.view(),delta.pair_kernel);
  dgpu.update_integrals(h.view(),g.view());
  gpu.apply_direction_with_adjoint(l,r,dgpu,out,h.view(),g.view());
  const Matrix combined=expected.sigma+direction.apply(right);
  near(read(),combined,3e-10,"CUDA exchange directional sigma differs");
  near(download(context,h),expected.integrals.one_electron,3e-10,
      "CUDA exchange directional one-electron adjoint differs");
  near(download(context,g),expected.integrals.pair_kernel,3e-10,
      "CUDA exchange directional pair adjoint differs");
  const xmvb::cuda::Graph captured(context,[&] {
    gpu.apply_direction_with_adjoint(l,r,dgpu,out,h.view(),g.view());
  });
  xmvb::cuda::Matrix<double> generator(context,n,n);
  const Matrix rotation=values(n,n,0.57);
  xmvb::cuda::copy_to_device(context,generator.view(),rotation);
  gpu.apply_generator(r,generator.view(),out);
  near(read(),cpu.apply_one_body_generator(right,rotation),3e-10,
      "CUDA exchange nonsymmetric generator differs");
  gpu.generator_moments(std::array{l,r},r,
      std::array{moments.view().columns(0,n),moments.view().columns(n,n)});
  const Matrix moment=download(context,moments);
  near(moment.leftCols(n),cpu.one_body_generator_adjoint(left,right),3e-10,
      "CUDA exchange mixed generator moments differ");
  near(moment.rightCols(n),cpu.one_body_generator_adjoint(right,right),3e-10,
      "CUDA exchange same-source generator moments differ");
  captured.launch();
  near(read(),combined,3e-10,"CUDA exchange graph replay differs");
  require((download(context,image).bottomRows(3).array()==-17.0).all(),
      "CUDA exchange output touched padded rows");
}

void check_distinct_spin_graphs() {
  constexpr int n=5;
  const auto a=xmvb::mcscf::SpinStrings(n,2).occupations();
  const auto b=xmvb::mcscf::SpinStrings(n,3).occupations();
  const xmvb::mcscf::DirectCiSigmaAction cpu(a,b,active_integrals(n,false));
  require(a.size()==b.size() && !cpu.shared_spin_connections(),
      "CUDA unequal-spin fixture lost its equal dimension coincidence");
  xmvb::cuda::Context context(128ULL*1024*1024);
  for (int parity:{-1,1}) {
    bool rejected=false;
    try {xmvb::cuda::DirectCiSigma invalid(context,cpu,16,parity);}
    catch (const std::invalid_argument&) {rejected=true;}
    require(rejected,"CUDA inferred spin-exchange symmetry from equal dimensions");
  }
  xmvb::cuda::DirectCiSigma gpu(context,cpu);
  const Matrix input=values(a.size(),b.size(),0.61);
  xmvb::cuda::Matrix<double> x(context,input.rows(),input.cols()),y(context,input.rows(),input.cols());
  xmvb::cuda::copy_to_device(context,x.view(),input);
  gpu.apply(x.view(),y.view());
  near(download(context,y),cpu.apply(input),3e-10,
      "CUDA equal-size distinct-spin action changed its physical space");
}

void check_row_budget() {
  constexpr int n=8,pairs=n*(n+1)/2,tile_rows=17;
  const auto alpha=xmvb::mcscf::SpinStrings(n,4).occupations();
  const auto beta=xmvb::mcscf::SpinStrings(n,3).occupations();
  const auto base=active_integrals(n,false);
  const xmvb::mcscf::ActiveIntegrals delta{
      symmetric_values(n,1.74),0.11*symmetric_values(pairs,2.36)};
  const xmvb::mcscf::DirectCiSigmaAction cpu(alpha,beta,base),direction(alpha,beta,delta);
  const Matrix left=values(alpha.size(),beta.size(),0.26);
  const Matrix right=values(alpha.size(),beta.size(),0.82);
  const auto expected=cpu.apply_with_adjoint(left,right);
  const Matrix combined=expected.sigma+direction.apply(right);
  xmvb::cuda::Context context(128ULL*1024*1024);
  xmvb::cuda::DirectCiSigma gpu(context,cpu),gpu_direction(context,gpu);
  xmvb::cuda::Matrix<double> dl(context,left.rows(),left.cols());
  xmvb::cuda::Matrix<double> dr(context,right.rows(),right.cols());
  xmvb::cuda::Matrix<double> image(context,left.rows(),left.cols());
  xmvb::cuda::Matrix<double> h(context,n,n),g(context,pairs,pairs);
  xmvb::cuda::copy_to_device(context,dl.view(),left);
  xmvb::cuda::copy_to_device(context,dr.view(),right);
  xmvb::cuda::copy_to_device(context,h.view(),delta.one_electron);
  xmvb::cuda::copy_to_device(context,g.view(),delta.pair_kernel);
  gpu_direction.update_integrals(h.view(),g.view());
  const auto resident=gpu.resident_bytes();
  const auto room=(static_cast<std::size_t>(pairs)*pairs+3*pairs*tile_rows)*sizeof(double);
  require(room<static_cast<std::size_t>(left.size())*sizeof(double),
      "CUDA row-budget fixture accidentally holds a full-carrier channel");
  const auto free=context.device_byte_budget()-context.counters().current_device_bytes;
  require(free>room,"CUDA row-budget fixture lacks its small workspace");
  xmvb::cuda::Buffer<unsigned char> padding(context,free-room);
  gpu.prepare_with_adjoint(1);
  require(gpu.resident_bytes()-resident==room,
      "CUDA fused row planner reserved an unused full-carrier channel");
  gpu.apply_direction_with_adjoint(dl.view(),dr.view(),gpu_direction,
      image.view(),h.view(),g.view());
  near(download(context,image),combined,3e-10,
      "CUDA sub-channel-budget directional sigma differs from CPU");
  near(download(context,h),expected.integrals.one_electron,3e-10,
      "CUDA sub-channel-budget one-electron adjoint differs from CPU");
  near(download(context,g),expected.integrals.pair_kernel,3e-10,
      "CUDA sub-channel-budget pair adjoint differs from CPU");
  const xmvb::cuda::Graph captured(context,[&] {
    gpu.apply_direction_with_adjoint(dl.view(),dr.view(),gpu_direction,
        image.view(),h.view(),g.view());
  });
  captured.launch();
  near(download(context,image),combined,3e-10,
      "CUDA row-budget graph replay differs from CPU");
  require(context.counters().peak_device_bytes<=context.device_byte_budget(),
      "CUDA source-row planning exceeded the shared budget");
}

void check_partitioned_moments() {
  constexpr int n=8,rhs=3;
  const xmvb::mcscf::SpinStrings a(n,4),b(n,3);
  const xmvb::mcscf::DirectCiSigmaAction cpu(a.occupations(),b.occupations(),
      active_integrals(n,false));
  const int rows=a.occupations().size(),cols=b.occupations().size()*rhs;
  const Matrix left=values(rows,cols,0.29),right=values(rows,cols,0.83);
  Matrix padded_left=Matrix::Constant(rows+3,cols,19.0);
  Matrix padded_right=Matrix::Constant(rows+1,cols,23.0);
  padded_left.topRows(rows)=left;padded_right.topRows(rows)=right;
  xmvb::cuda::Context context(128ULL*1024*1024);
  xmvb::cuda::DirectCiSigma gpu(context,cpu);
  xmvb::cuda::Matrix<double> dl(context,rows+3,cols),dr(context,rows+1,cols);
  xmvb::cuda::Matrix<double> output(context,n+2,2*n);
  xmvb::cuda::copy_to_device(context,dl.view(),padded_left);
  xmvb::cuda::copy_to_device(context,dr.view(),padded_right);
  const Matrix padding=Matrix::Constant(n+2,2*n,-17.0);
  xmvb::cuda::copy_to_device(context,output.view(),padding);
  const xmvb::linalg::ConstMatrixView l{dl.data(),rows,cols,rows+3};
  const xmvb::linalg::ConstMatrixView r{dr.data(),rows,cols,rows+1};
  const xmvb::linalg::MatrixView o1{output.data(),n,n,n+2};
  const xmvb::linalg::MatrixView o2{output.data()+output.rows()*n,n,n,n+2};
  const auto bytes=gpu.resident_bytes(),generation=gpu.workspace_generation();
  gpu.generator_moments(std::array{l,r},r,std::array{o1,o2});
  const Matrix result=download(context,output);
  near(result.topRows(n).leftCols(n),cpu.one_body_generator_adjoint(left,right),
      3e-10,"CUDA partitioned unequal-spin moments differ from CPU");
  near(result.topRows(n).rightCols(n),cpu.one_body_generator_adjoint(right,right),
      3e-10,"CUDA partitioned equal-carrier moments differ from CPU");
  require((result.bottomRows(2).array()==-17.0).all(),
      "CUDA generator moment overwrote padded output rows");
  require(gpu.resident_bytes()==bytes && gpu.workspace_generation()==generation,
      "CUDA direct moments allocated resident channel panels or invalidated sigma");
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
  xmvb::cuda::ExteriorTransform gpu_beta(context, cpu_beta);
  if (rhs_count >= 2 && n == 4) {
    const int alpha_dim=static_cast<int>(alpha.size());
    const int columns=static_cast<int>(beta.size());
    const int words=(alpha_dim+63)/64;
    std::vector<int> offsets{0},supported_rows;
    std::vector<std::uint64_t> masks(static_cast<std::size_t>(columns)*words,0);
    for (int b=0;b<columns;++b) {
      for (int a=0;a<alpha_dim;++a) if ((a+b)%3!=0) {
        supported_rows.push_back(a);
        masks[static_cast<std::size_t>(b)*words+a/64]|=
            std::uint64_t{1}<<(a%64);
      }
      offsets.push_back(static_cast<int>(supported_rows.size()));
    }
    xmvb::cuda::Buffer<std::uint64_t> device_masks(context,masks.size());
    context.copy_to_device(device_masks.data(),masks.data(),
        masks.size()*sizeof(std::uint64_t));
    const Matrix initial=values(alpha.size(),columns*rhs_count,0.48);
    xmvb::cuda::Matrix<double> padded(context,initial.rows()+2,initial.cols());
    for (bool adjoint:{false,true}) {
      Matrix expected=initial;
      cpu_alpha.apply_supported_left(&expected,offsets,supported_rows,adjoint);
      Matrix input=Matrix::Constant(padded.rows(),padded.cols(),-7.0);
      input.topRows(initial.rows())=initial;
      xmvb::cuda::copy_to_device(context,padded.view(),input);
      gpu_alpha.apply_supported_left(
          {padded.data(),initial.rows(),initial.cols(),padded.rows()},
          device_masks.data(),columns,adjoint);
      input.topRows(initial.rows())=expected;
      near(download(context,padded),input,2e-11,
          "multi-RHS supported CUDA exterior transform differs from CPU");
    }
  }
  {
    const std::vector<int> requested{static_cast<int>(beta.size())-1,0};
    xmvb::cuda::Matrix<double> d_upper(context,n,n),rows(context,beta.size(),requested.size());
    xmvb::cuda::copy_to_device(context,d_upper.view(),upper);
    gpu_beta.sample_rows(requested,rows.view());
    near(download(context,rows).transpose(),Matrix(cpu_beta.generate_rows(requested)),
        2e-11,"device requested rows differ from the shared CPU row recursion");
    Matrix changed=upper;
    changed(0,n-1)+=0.07;
    const xmvb::vb::ExteriorOrbitalTransform cpu_changed(beta,changed);
    xmvb::cuda::copy_to_device(context,d_upper.view(),changed);
    gpu_beta.update_upper(d_upper.view());
    gpu_beta.sample_rows(requested,rows.view());
    near(download(context,rows).transpose(),Matrix(cpu_changed.generate_rows(requested)),
        2e-11,"device requested rows did not follow the updated orbital frame");
    xmvb::cuda::copy_to_device(context,d_upper.view(),upper);
    gpu_beta.update_upper(d_upper.view());
  }
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
  {
    Matrix delta=Matrix::Zero(n,n);
    for (int j=0;j<n;++j) for (int i=0;i<=j;++i)
      delta(i,j)=0.02*std::sin(0.17*(i+1)*(j+2));
    const Matrix initial=values(alpha.size(),rhs_count,0.29);
    const Matrix tangent=0.07*values(alpha.size(),rhs_count,0.82);
    xmvb::cuda::Matrix<double> u(context,n,n),du(context,n,n),
        primal(context,initial.rows()+2,rhs_count),jet(context,initial.rows()+3,rhs_count);
    xmvb::cuda::copy_to_device(context,u.view(),upper);
    xmvb::cuda::copy_to_device(context,du.view(),delta);
    const auto cpu_direction=cpu_alpha.direction(delta);
    const auto gpu_direction=gpu_alpha.direction(u.view(),du.view());
    for (bool adjoint:{false,true}) {
      Matrix expected=initial,expected_delta=tangent;
      if (adjoint) cpu_alpha.apply_directional_adjoint_left(
          cpu_direction,&expected,&expected_delta);
      else cpu_alpha.apply_directional_left(cpu_direction,&expected,&expected_delta);
      Matrix p=Matrix::Constant(primal.rows(),rhs_count,-17.0);
      Matrix dp=Matrix::Constant(jet.rows(),rhs_count,23.0);
      p.topRows(initial.rows())=initial;dp.topRows(initial.rows())=tangent;
      xmvb::cuda::copy_to_device(context,primal.view(),p);
      xmvb::cuda::copy_to_device(context,jet.view(),dp);
      gpu_alpha.apply_directional_left(gpu_direction,
          {primal.data(),initial.rows(),rhs_count,primal.rows()},
          {jet.data(),initial.rows(),rhs_count,jet.rows()},adjoint);
      p.topRows(initial.rows())=expected;dp.topRows(initial.rows())=expected_delta;
      near(download(context,primal),p,3e-11,
          "directional exterior primal or padding differs from CPU");
      near(download(context,jet),dp,3e-11,
          "directional exterior tangent or padding differs from CPU");
    }
  }
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
    check_compact_endpoints();
    for (int parity:{0,1,-1}) for (std::size_t budget:{std::size_t{4096},std::size_t{1024*1024}})
      check_product_tiles(parity,parity!=0,budget);
    check_product_tiles(0,true,1024*1024);
    for (bool rotated : {false, true})
      for (int rhs_count : {1, 3}) {
        check_sigma(rhs_count,rotated);
        check_sigma(rhs_count,rotated,1);
        check_sigma(rhs_count,rotated,1,true);
      }
    // Wide, unequal spin sectors and one-channel-equivalent workspace force
    // many source tiles. Exercise transpose scatter, add mode and fused HVP.
    check_sigma(2,false,1,false,8);
    for (int parity:{-1,1}) for (int rhs:{1,3}) {
      check_exchange_parity(4,rhs,parity,1);
      check_exchange_parity(6,rhs,parity,3);
    }
    check_distinct_spin_graphs();
    check_row_budget();
    check_partitioned_moments();
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
