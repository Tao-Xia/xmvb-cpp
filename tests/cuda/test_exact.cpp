#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "cuda/integrals/exact.hpp"
#include "cuda/integrals/packed.hpp"
#include "cuda/integrals/one_electron.hpp"
#include "cuda/linalg/blas.hpp"
#include "cuda/memory/buffer.hpp"
#include "integrals/ao/graph.hpp"
#include "integrals/ao/exact.hpp"

namespace {

using Matrix = Eigen::MatrixXd;
using DeviceMatrix = xmvb::cuda::Matrix<double>;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void near(const Matrix& actual, const Matrix& expected, const char* message,
          double tolerance = 2e-10) {
  require(actual.rows() == expected.rows() && actual.cols() == expected.cols(),
          "CUDA exact integral result has wrong shape");
  require(actual.allFinite() && expected.allFinite() &&
              (actual - expected).norm() <=
                  tolerance * std::max(1.0, expected.norm()), message);
}

void near_scalar(double actual, double expected, const char* message,
                 double tolerance = 2e-10) {
  require(std::isfinite(actual) && std::isfinite(expected) &&
              std::abs(actual - expected) <=
                  tolerance * std::max(1.0, std::abs(expected)), message);
}

Matrix values(int rows, int cols, double phase) {
  Matrix result(rows, cols);
  for (int j = 0; j < cols; ++j)
    for (int i = 0; i < rows; ++i)
      result(i, j) = std::sin(phase + 0.31 * (i + 1) * (j + 2)) +
                     0.13 * std::cos(0.41 * (i + 2) * (j + 1));
  return result;
}

Matrix symmetric_values(int n, double phase) {
  const Matrix raw = values(n, n, phase);
  return (0.5 * (raw + raw.transpose())).eval();
}

int pair_index(int i, int j) {
  if (i < j) std::swap(i, j);
  return i * (i + 1) / 2 + j;
}

struct Fixture {
  int n_bf;
  int ao_pairs;
  Matrix pair_kernel;
  std::vector<int> unique_left, unique_right;
  std::vector<double> unique_values;
  std::vector<int> first, second;

  explicit Fixture(bool sparse=false,int n=4)
      : n_bf(n),ao_pairs(n*(n+1)/2),pair_kernel(ao_pairs, ao_pairs) {
    pair_kernel.setZero();
    for (int i = 0; i < n_bf; ++i)
      for (int j = 0; j <= i; ++j) {
        first.push_back(i);
        second.push_back(j);
      }
    for (int p = 0; p < ao_pairs; ++p)
      for (int q = 0; q <= p; ++q) {
        if (sparse && (p+q)%7!=0) continue;
        const double value =
            0.07 * std::cos(0.19 * (p + 1) * (q + 2)) +
            0.011 / static_cast<double>(p + q + 1);
        unique_left.push_back(p);
        unique_right.push_back(q);
        unique_values.push_back(value);
        pair_kernel(p, q) = pair_kernel(q, p) = value;
      }
  }

  xmvb::integrals::AoPairGraph graph(bool compact,bool chunked=false) const {
    if (chunked) {
      std::vector<xmvb::integrals::AoIntegralChunk> chunks(5);
      const std::size_t edges[]={0,2,2,unique_values.size()/2,unique_values.size(),unique_values.size()};
      for (int i=0;i<5;++i) {
        chunks[i].rows.assign(unique_left.begin()+edges[i],unique_left.begin()+edges[i+1]);
        chunks[i].columns.assign(unique_right.begin()+edges[i],unique_right.begin()+edges[i+1]);
        chunks[i].values.assign(unique_values.begin()+edges[i],unique_values.begin()+edges[i+1]);
      }
      std::reverse(chunks.begin(),chunks.end());
      return xmvb::integrals::build_ao_pair_graph(std::move(chunks),n_bf);
    }
    if (compact) {
      auto rows=unique_left,columns=unique_right;
      auto coefficients=unique_values;
      for (int i=0;i<2;++i) {
        rows.push_back(rows[i]);columns.push_back(columns[i]);
        coefficients.push_back(0.25*coefficients[i]);coefficients[i]*=0.75;
      }
      std::reverse(rows.begin(),rows.end());
      std::reverse(columns.begin(),columns.end());
      std::reverse(coefficients.begin(),coefficients.end());
      return xmvb::integrals::build_ao_pair_graph(std::move(rows),std::move(columns),
          std::move(coefficients),n_bf,xmvb::integrals::AoPairLayout::Unique);
    }
    return xmvb::integrals::build_ao_pair_graph(unique_left,unique_right,unique_values,n_bf,
        compact ? xmvb::integrals::AoPairLayout::Unique:xmvb::integrals::AoPairLayout::Automatic);
  }

  double eri(int i, int j, int k, int l) const {
    return pair_kernel(pair_index(i, j), pair_index(k, l));
  }
};

void upload(xmvb::cuda::Context& context, DeviceMatrix& device,
            const Matrix& host) {
  xmvb::cuda::copy_to_device(context, device.view(), host);
}

Matrix download(xmvb::cuda::Context& context, const DeviceMatrix& device) {
  Matrix result(device.rows(), device.cols());
  xmvb::cuda::copy_to_host(context, result, device.view());
  return result;
}

Matrix features(const Fixture& fixture, const Matrix& c) {
  const int active = c.cols(), pairs = active * (active + 1) / 2;
  Matrix result(fixture.ao_pairs, pairs);
  for (int row = 0; row < fixture.ao_pairs; ++row) {
    const int i = fixture.first[row], j = fixture.second[row];
    for (int a = 0; a < active; ++a)
      for (int b = 0; b <= a; ++b) {
        const int p = pair_index(a, b);
        result(row, p) = c(i, a) * c(j, b) +
                         (i == j ? 0.0 : c(j, a) * c(i, b));
      }
  }
  return result;
}

Matrix directional_features(const Fixture& fixture, const Matrix& c,
                            const Matrix& direction) {
  return (features(fixture, c + direction) - features(fixture, c) -
          features(fixture, direction)).eval();
}

Matrix pair_pullback(const Fixture& fixture,const Matrix& c,const Matrix& tile_gradient) {
  Matrix gradient = Matrix::Zero(c.rows(), c.cols());
  for (int row = 0; row < fixture.ao_pairs; ++row) {
    const int i = fixture.first[row], j = fixture.second[row];
    for (int a = 0; a < c.cols(); ++a)
      for (int b = 0; b <= a; ++b) {
        const double value = tile_gradient(row, pair_index(a, b));
        gradient(i, a) += value * c(j, b);
        gradient(j, b) += value * c(i, a);
        if (i != j) {
          gradient(j, a) += value * c(i, b);
          gradient(i, b) += value * c(j, a);
        }
      }
  }
  return gradient;
}

Matrix pair_gradient(const Fixture& fixture, const Matrix& c,
                     const Matrix& adjoint) {
  const Matrix product = fixture.pair_kernel * features(fixture, c);
  return pair_pullback(fixture,c,product*(adjoint+adjoint.transpose()));
}

Matrix reference_jk(const Fixture& fixture, const Matrix& density,
                    double coulomb = 2.0, double exchange = -1.0) {
  Matrix result = Matrix::Zero(fixture.n_bf, fixture.n_bf);
  for (int i = 0; i < fixture.n_bf; ++i)
    for (int j = 0; j < fixture.n_bf; ++j)
      for (int k = 0; k < fixture.n_bf; ++k)
        for (int l = 0; l < fixture.n_bf; ++l)
          result(i, j) +=
              (coulomb * fixture.eri(i, j, k, l) +
               exchange * fixture.eri(i, k, j, l)) * density(k, l);
  return result;
}

void check_packed_exchange_blocks() {
  // 1035 AO pairs cross both the 512-wide packed-tile boundary and multiple
  // 256-thread source iterations, unlike the tiny integral oracle fixtures.
  const Fixture fixture(false,45);
  xmvb::cuda::Context context(128ULL*1024ULL*1024ULL);
  const xmvb::cuda::ExactAoPairOperator gpu(context,fixture.graph(false),fixture.n_bf);
  DeviceMatrix density(context,fixture.n_bf,fixture.n_bf),out(context,fixture.n_bf,fixture.n_bf);
  for (int kind=0;kind<3;++kind) {
    Matrix host=kind==0 ? symmetric_values(fixture.n_bf,0.83):
        values(fixture.n_bf,fixture.n_bf,1.39);
    if (kind==2) {
      host.setZero();
      host(3,41)=0.7;host(37,5)=-0.2;host(19,19)=0.4;
    }
    upload(context,density,host);
    gpu.contract_jk(density.view(),out.view(),0.0,1.0);
    near(download(context,out),reference_jk(fixture,host,0.0,1.0),
        "CUDA packed exchange differs from four-index oracle across block boundaries");
  }
}

void check_chunk_consumers() {
  const Fixture fixture;
  xmvb::integrals::AoIntegralInput flat,chunked;
  flat.n_basis_functions=chunked.n_basis_functions=fixture.n_bf;
  flat.pair_graph=fixture.graph(true);
  chunked.pair_graph=fixture.graph(true,true);
  const Matrix density=symmetric_values(fixture.n_bf,0.91);
  Matrix result;
  xmvb::integrals::apply_weighted_symmetric_ao_jk(density,chunked,2.0,-1.0,2,&result);
  near(result,reference_jk(fixture,density),"CPU chunked weighted JK differs from oracle");
  const Matrix sources=Eigen::Map<const Matrix>(density.data(),density.size(),1);
  Matrix flat_result,chunk_result;
  xmvb::integrals::AoH1eBatchWorkspace a,b;
  xmvb::integrals::apply_symmetric_ao_h1e_batch(sources,flat,2,&a,&flat_result);
  xmvb::integrals::apply_symmetric_ao_h1e_batch(sources,chunked,2,&b,&chunk_result);
  near(chunk_result,flat_result,"CPU chunked H1e batch differs from flat storage");
}

double trace_product(const Matrix& left, const Matrix& right) {
  return (left.array() * right.transpose().array()).sum();
}

void check_graph_and_pair(int active, bool compact,bool sparse=false,bool chunked=false) {
  const Fixture fixture(sparse);
  xmvb::cuda::Context context(128ULL * 1024ULL * 1024ULL);
  const xmvb::cuda::ExactAoPairOperator gpu(context, fixture.graph(compact,chunked),
                                            fixture.n_bf);
  const int pairs = active * (active + 1) / 2;
  const Matrix c = values(fixture.n_bf, active, 0.31);
  const Matrix direction = values(fixture.n_bf, active, 1.17);
  const Matrix upstream = values(pairs, pairs, 0.63);
  const Matrix upstream_direction = values(pairs, pairs, 1.43);
  DeviceMatrix dc(context, fixture.n_bf, active), dd(context, fixture.n_bf, active);
  DeviceMatrix dg(context, pairs, pairs), ddg(context, pairs, pairs);
  DeviceMatrix dkernel(context, pairs, pairs), ddelta(context, pairs, pairs);
  DeviceMatrix dgradient(context, fixture.n_bf, active);
  upload(context, dc, c);
  upload(context, dd, direction);
  upload(context, dg, upstream);
  upload(context, ddg, upstream_direction);

  // The borrowed row-major descriptor is intentionally different from the
  // column-major MatrixView contract of the orbital and active-kernel API.
  constexpr int width = 3;
  std::vector<double> source(fixture.ao_pairs * width);
  for (int row = 0; row < fixture.ao_pairs; ++row)
    for (int channel = 0; channel < width; ++channel)
      source[row * width + channel] =
          std::sin(0.27 * (row + 1) * (channel + 2));
  xmvb::cuda::Buffer<double> dsource(context, source.size());
  xmvb::cuda::Buffer<double> dtarget(context, source.size());
  context.copy_to_device(dsource.data(), source.data(), source.size() * sizeof(double));
  gpu.apply({dsource.data(), fixture.ao_pairs, width, width},
            {dtarget.data(), fixture.ao_pairs, width, width});
  std::vector<double> target(source.size());
  context.copy_to_host(target.data(), dtarget.data(), target.size() * sizeof(double));
  for (int row = 0; row < fixture.ao_pairs; ++row)
    for (int channel = 0; channel < width; ++channel) {
      double expected = 0.0;
      for (int source_row = 0; source_row < fixture.ao_pairs; ++source_row)
        expected += fixture.pair_kernel(row, source_row) *
                    source[source_row * width + channel];
      near_scalar(target[row * width + channel], expected,
                  "CUDA AO-pair graph action differs from dense oracle");
    }

  const Matrix b = features(fixture, c);
  const Matrix db = directional_features(fixture, c, direction);
  const Matrix p = fixture.pair_kernel * b;
  const Matrix dp = fixture.pair_kernel * db;
  constexpr int begin = 2, rows = 3;
  xmvb::cuda::Buffer<double> d_b_tile(context, rows * pairs),
      d_db_tile(context, rows * pairs);
  xmvb::cuda::Buffer<double> d_p_tile(context, rows * pairs),
      d_dp_tile(context, rows * pairs);
  gpu.pair_features(dc.view(), dd.view(), begin,
                    {d_b_tile.data(), rows, pairs, pairs},
                    {d_db_tile.data(), rows, pairs, pairs});
  gpu.generated_rows(dc.view(), dd.view(), begin,
                     {d_p_tile.data(), rows, pairs, pairs},
                     {d_dp_tile.data(), rows, pairs, pairs});
  std::vector<double> btile(rows * pairs), dbtile(rows * pairs);
  std::vector<double> ptile(rows * pairs), dptile(rows * pairs);
  context.copy_to_host(btile.data(), d_b_tile.data(), btile.size() * sizeof(double));
  context.copy_to_host(dbtile.data(), d_db_tile.data(), dbtile.size() * sizeof(double));
  context.copy_to_host(ptile.data(), d_p_tile.data(), ptile.size() * sizeof(double));
  context.copy_to_host(dptile.data(), d_dp_tile.data(), dptile.size() * sizeof(double));
  for (int row = 0; row < rows; ++row)
    for (int col = 0; col < pairs; ++col) {
      near_scalar(btile[row * pairs + col], b(begin + row, col),
                  "CUDA pair feature tile differs from oracle");
      near_scalar(dbtile[row * pairs + col], db(begin + row, col),
                  "CUDA directional pair feature tile differs from oracle");
      near_scalar(ptile[row * pairs + col], p(begin + row, col),
                  "CUDA generated pair image differs from oracle");
      near_scalar(dptile[row * pairs + col], dp(begin + row, col),
                  "CUDA generated directional image differs from oracle");
    }

  // A complete AO-row request takes the CSR feature-then-SpMM route. The
  // direction shares its temporary feature slab but must keep a distinct
  // full-row output, including all off-diagonal active-pair channels.
  const int full_stride=pairs+2;
  xmvb::cuda::Buffer<double> d_full_p(context,fixture.ao_pairs*full_stride);
  xmvb::cuda::Buffer<double> d_full_dp(context,fixture.ao_pairs*full_stride);
  std::vector<double> full_p(fixture.ao_pairs*full_stride,-17.0),full_dp(full_p);
  context.copy_to_device(d_full_p.data(),full_p.data(),full_p.size()*sizeof(double));
  context.copy_to_device(d_full_dp.data(),full_dp.data(),full_dp.size()*sizeof(double));
  gpu.generated_rows(dc.view(),dd.view(),0,
      {d_full_p.data(),fixture.ao_pairs,pairs,full_stride},
      {d_full_dp.data(),fixture.ao_pairs,pairs,full_stride});
  context.copy_to_host(full_p.data(),d_full_p.data(),full_p.size()*sizeof(double));
  context.copy_to_host(full_dp.data(),d_full_dp.data(),full_dp.size()*sizeof(double));
  for (int row=0;row<fixture.ao_pairs;++row)
    for (int col=0;col<pairs;++col) {
      near_scalar(full_p[row*full_stride+col],p(row,col),
                  "CUDA full-row CSR generated image differs from oracle");
      near_scalar(full_dp[row*full_stride+col],dp(row,col),
                  "CUDA full-row CSR directional image differs from oracle");
    }
  for (int row=0;row<fixture.ao_pairs;++row)
    for (int col=pairs;col<full_stride;++col)
      require(full_p[row*full_stride+col]==-17.0 && full_dp[row*full_stride+col]==-17.0,
              "CUDA generated AO-pair clearing overwrote stride padding");

  gpu.forward(dc.view(), dkernel.view());
  near(download(context, dkernel), b.transpose() * p,
       "CUDA full active pair kernel differs from dense oracle");
  gpu.directional(dc.view(), dd.view(), ddelta.view());
  const Matrix expected_delta = db.transpose() * p + b.transpose() * dp;
  near(download(context, ddelta), expected_delta,
       "CUDA active pair direction differs from dense oracle");

  gpu.adjoint(dc.view(), dg.view(), dgradient.view());
  const Matrix gradient = download(context, dgradient);
  near(gradient, pair_gradient(fixture, c, upstream),
       "CUDA full Frobenius pair adjoint differs from oracle");
  near_scalar((gradient.array() * direction.array()).sum(),
              (upstream.array() * expected_delta.array()).sum(),
              "CUDA full nonsymmetric pair adjoint fails dot test");
  gpu.directional_adjoint(dc.view(), dd.view(), dg.view(), ddg.view(),
                          dgradient.view());
  constexpr double step = 2e-6;
  const Matrix expected_hvp =
      (pair_gradient(fixture, c + step * direction,
                     upstream + step * upstream_direction) -
       pair_gradient(fixture, c - step * direction,
                     upstream - step * upstream_direction)) / (2 * step);
  near(download(context, dgradient), expected_hvp,
       "CUDA changing-adjoint pair HVP differs from finite difference", 2e-7);

  // Point-owned half transform is filled by forward, then borrowed by every
  // consumer. A padded row stride catches accidental packed-only addressing.
  xmvb::cuda::Buffer<double> accepted(context,fixture.ao_pairs*(pairs+2));
  const xmvb::cuda::PairPanel accepted_panel{accepted.data(),fixture.ao_pairs,pairs,pairs+2};
  const xmvb::cuda::ConstPairPanel accepted_view{accepted.data(),fixture.ao_pairs,pairs,pairs+2};
  gpu.forward(dc.view(),dkernel.view(),accepted_panel);
  near(download(context,dkernel),b.transpose()*p,"CUDA retained pair forward differs from oracle");
  gpu.directional(dc.view(),dd.view(),ddelta.view(),accepted_view);
  near(download(context,ddelta),expected_delta,"CUDA retained pair direction differs from oracle");
  gpu.adjoint(dc.view(),dg.view(),dgradient.view(),accepted_view);
  near(download(context,dgradient),gradient,"CUDA retained pair gradient differs from oracle");
  gpu.directional_adjoint(dc.view(),dd.view(),dg.view(),ddg.view(),dgradient.view(),accepted_view);
  near(download(context,dgradient),expected_hvp,"CUDA retained pair HVP differs from finite difference",2e-7);
  auto exchange=gpu.prepare_exchange(dc.view());
  require(!exchange.empty(),"CUDA exchange image was not admitted within the test budget");
  if (!exchange.empty()) {
    const xmvb::cuda::ConstPairPanel image{exchange.data(),fixture.ao_pairs,active*active,active*active};
    std::vector<double> host(exchange.size());
    context.copy_to_host(host.data(),exchange.data(),host.size()*sizeof(double));
    for (int mu=0;mu<fixture.n_bf;++mu) for (int nu=0;nu<=mu;++nu)
      for (int a=0;a<active;++a) for (int b=0;b<active;++b) {
        double reference=0.0;
        for (int k=0;k<fixture.n_bf;++k) for (int l=0;l<fixture.n_bf;++l)
          reference+=fixture.eri(mu,k,nu,l)*c(k,a)*c(l,b);
        near_scalar(host[pair_index(mu,nu)*active*active+a*active+b],reference,
            "CUDA exchange half transform differs from four-index oracle");
      }
    gpu.directional_adjoint(dc.view(),dd.view(),dg.view(),ddg.view(),dgradient.view(),accepted_view,image);
    near(download(context,dgradient),expected_hvp,"CUDA exchange projected HVP differs",2e-7);
    gpu.directional_adjoint(dc.view(),dd.view(),dg.view(),ddg.view(),dgradient.view(),{},image);
    near(download(context,dgradient),expected_hvp,"CUDA exchange streamed-product HVP differs",2e-7);
    const Matrix weight=symmetric_values(active,0.82);
    DeviceMatrix dweight(context,active,active),density_image(context,fixture.n_bf,fixture.n_bf);
    upload(context,dweight,weight);
    gpu.exchange_density(image,dweight.view(),density_image.view());
    near(download(context,density_image),reference_jk(fixture,c*weight*c.transpose(),0.0,-1.0),
        "CUDA exchange density projection differs from four-index oracle");
    gpu.directional_adjoint(dc.view(),dd.view(),dg.view(),{},dgradient.view(),accepted_view,image);
    const Matrix fixed_hvp=(pair_gradient(fixture,c+step*direction,upstream)-
        pair_gradient(fixture,c-step*direction,upstream))/(2*step);
    near(download(context,dgradient),fixed_hvp,"CUDA fixed-adjoint exchange HVP differs",2e-7);
    {
      const auto n2=static_cast<std::size_t>(active)*active,n3=n2*active;
      const auto fixed_bytes=(active*n3+fixture.n_bf*n3)*sizeof(double);
      const auto row_bytes=fixture.n_bf*n2*sizeof(double);
      const auto free=context.device_byte_budget()-context.counters().current_device_bytes;
      require(free>fixed_bytes+2*row_bytes,"CUDA exchange tile fixture lacks workspace");
      xmvb::cuda::Buffer<unsigned char> padding(context,free-fixed_bytes-2*row_bytes);
      require(gpu.prepare_exchange(dc.view()).empty(),"CUDA exchange admission ignored the live budget");
      xmvb::cuda::check_cuda(cudaMemsetAsync(dgradient.data(),0,
          fixture.n_bf*active*sizeof(double),context.stream()),"clear tiled exchange oracle");
      gpu.exchange_pair_direction(image,dd.view(),dg.view(),dgradient.view());
      near(download(context,dgradient),pair_pullback(fixture,c,
          fixture.pair_kernel*db*(upstream+upstream.transpose())),
          "CUDA budgeted exchange output-row projection differs from oracle");
      require(context.counters().peak_device_bytes<=context.device_byte_budget(),
          "CUDA exchange projection exceeded its memory budget");
    }
  }
}

void check_core_symmetry_orientations() {
  constexpr int n=4,ncore=3;
  const Matrix mo=values(n,n,0.33);
  const std::vector<std::pair<int,int>> cases{
      {pair_index(3,2),pair_index(1,0)},
      {pair_index(3,1),pair_index(3,0)},
      {pair_index(3,1),pair_index(3,1)},
      {pair_index(2,2),pair_index(2,2)},
      {pair_index(3,3),pair_index(3,0)},
      {pair_index(3,3),pair_index(1,1)}};
  for (const auto [u,v]:cases) {
    Fixture fixture(true,n);
    fixture.pair_kernel.setZero();
    fixture.pair_kernel(u,v)=fixture.pair_kernel(v,u)=0.17;
    fixture.unique_left={u};fixture.unique_right={v};fixture.unique_values={0.17};
    xmvb::cuda::Context context(128ULL*1024*1024);
    const xmvb::cuda::ExactAoPairOperator gpu(context,fixture.graph(true),n);
    DeviceMatrix c(context,n,n),j(context,n,ncore),r(context,n,ncore);
    upload(context,c,mo);
    require(gpu.core_diagonal(c.view(),ncore,j.view(),r.view()),
        "CUDA isolated-ERI core projection was not prepared");
    Matrix expected_j(n,ncore),expected_r(n,ncore);
    for (int core=0;core<ncore;++core) {
      const Matrix density=mo.col(core)*mo.col(core).transpose();
      const Matrix cj=reference_jk(fixture,density,1.0,0.0);
      const Matrix cr=reference_jk(fixture,density);
      expected_j.col(core)=(mo.transpose()*cj*mo).diagonal();
      expected_r.col(core)=(mo.transpose()*cr*mo).diagonal();
    }
    near(download(context,j),expected_j,"CUDA canonical core Coulomb orientation differs");
    near(download(context,r),expected_r,"CUDA canonical exchange symmetry multiplicity differs");
  }
}

void check_unique_half_tiles() {
  constexpr int n=12,active=3;
  const Fixture fixture(true,n);
  xmvb::cuda::Context context(128ULL*1024*1024);
  const xmvb::cuda::ExactAoPairOperator gpu(context,fixture.graph(true),n);
  DeviceMatrix c(context,n,active);
  Matrix factors=values(n,active,0.73);
  for (int j=0;j<active;++j) for (int i=0;i<n;++i)
    if ((i+j)%3) factors(i,j)=0.0;
  upload(context,c,factors);
  auto image=gpu.prepare_exchange(c.view());
  std::vector<double> expected(image.size()),actual(image.size());
  context.copy_to_host(expected.data(),image.data(),expected.size()*sizeof(double));
  image={};
  const auto image_count=static_cast<std::size_t>(fixture.ao_pairs)*active*active;
  const auto scratch=std::max(n*n*active+n*active*active,
      n*active*active*active+active*active*active*active+2*n*active*active);
  const auto room=(2*image_count+scratch)*sizeof(double);
  const auto free=context.device_byte_budget()-context.counters().current_device_bytes;
  require(free>room,"CUDA unique half fixture lacks its minimum workspace");
  xmvb::cuda::Buffer<unsigned char> padding(context,free-room);
  // Admission leaves only a few mu slices. Multiple unique-ERI sweeps must
  // recover the same complete image without a copied adjacency or AO^4 slab.
  image=gpu.prepare_exchange(c.view());
  require(image.size()==expected.size(),"CUDA unique half tile image was not admitted");
  context.copy_to_host(actual.data(),image.data(),actual.size()*sizeof(double));
  for (std::size_t i=0;i<actual.size();++i)
    near_scalar(actual[i],expected[i],"CUDA tiled unique half differs from full-mu scheduling");
  require(context.counters().peak_device_bytes<=context.device_byte_budget(),
      "CUDA unique half exceeded the shared budget");
}

void check_one_electron_and_core(bool compact,bool sparse=false,bool chunked=false,int n_bf=4) {
  const Fixture fixture(sparse,n_bf);
  constexpr int active = 3;
  xmvb::cuda::Context context(128ULL * 1024ULL * 1024ULL);
  const xmvb::cuda::ExactAoPairOperator gpu(context, fixture.graph(compact,chunked),
                                            fixture.n_bf);
  {
    Matrix mo=values(fixture.n_bf,fixture.n_bf,0.33);
    for (int j=0;j<fixture.n_bf;++j) for (int i=0;i<fixture.n_bf;++i)
      if ((i+j)%3) mo(i,j)=0.0;
    mo.col(1).setZero(); // One null source must not contaminate other batch channels.
    DeviceMatrix dmo(context,fixture.n_bf,fixture.n_bf);
    upload(context,dmo,mo);
    const int ncore=std::min(3,fixture.n_bf);
    DeviceMatrix dj(context,fixture.n_bf,ncore),dr(context,fixture.n_bf,ncore);
    require(gpu.core_diagonal(dmo.view(),ncore,dj.view(),dr.view()),
        "CUDA core diagonal was not prepared within the test budget");
    Matrix expected_j(fixture.n_bf,ncore),expected_r(fixture.n_bf,ncore);
    for (int c=0;c<ncore;++c) {
      const Matrix density=mo.col(c)*mo.col(c).transpose();
      const Matrix j=reference_jk(fixture,density,1.0,0.0);
      const Matrix r=reference_jk(fixture,density);
      for (int p=0;p<fixture.n_bf;++p) {
        expected_j(p,c)=mo.col(p).dot(j*mo.col(p));
        expected_r(p,c)=mo.col(p).dot(r*mo.col(p));
      }
    }
    near(download(context,dj),expected_j,"CUDA rank-one core Coulomb differs from oracle");
    near(download(context,dr),expected_r,"CUDA rank-one core response differs from oracle");
    // A short feature panel keeps a padded channel stride, as the final
    // budgeted core batch does. Untouched padding must remain intact.
    const int begin=1,rows=fixture.ao_pairs-2,width=ncore+2;
    std::vector<double> features(rows*width,-17.0);
    xmvb::cuda::Buffer<double> df(context,features.size());
    context.copy_to_device(df.data(),features.data(),features.size()*sizeof(double));
    gpu.density_features(dmo.view().columns(0,ncore),begin,
        {df.data(),rows,ncore,width});
    context.copy_to_host(features.data(),df.data(),features.size()*sizeof(double));
    for (int row=0;row<rows;++row) {
      const int i=fixture.first[begin+row],j=fixture.second[begin+row];
      for (int c=0;c<width;++c)
        near_scalar(features[row*width+c],c<ncore ?
            (i==j ? 1.0:2.0)*mo(i,c)*mo(j,c):-17.0,
            "CUDA rank-one density features changed a coefficient or its padding");
    }
  }
  const Matrix c = values(fixture.n_bf, active, 0.27);
  const Matrix direction = values(fixture.n_bf, active, 0.83);
  const Matrix core = symmetric_values(fixture.n_bf, 0.59);
  const Matrix upstream = values(active, active, 1.19);
  const Matrix upstream_direction = values(active, active, 1.79);
  DeviceMatrix dc(context, fixture.n_bf, active), dd(context, fixture.n_bf, active);
  DeviceMatrix dh(context, fixture.n_bf, fixture.n_bf);
  DeviceMatrix da(context, active, active), dda(context, active, active);
  DeviceMatrix dactive(context, active, active), dgradient(context, fixture.n_bf, active);
  upload(context, dc, c);
  upload(context, dd, direction);
  upload(context, dh, core);
  upload(context, da, upstream);
  upload(context, dda, upstream_direction);
  xmvb::cuda::transform_one_electron(context, dh.view(), dc.view(), dactive.view());
  near(download(context, dactive), c.transpose() * core * c,
       "CUDA one-electron transform differs from oracle");
  xmvb::cuda::transform_one_electron_directional(
      context, dh.view(), dc.view(), dd.view(), dactive.view());
  const Matrix expected_direction =
      direction.transpose() * core * c + c.transpose() * core * direction;
  near(download(context, dactive), expected_direction,
       "CUDA one-electron direction differs from oracle");
  xmvb::cuda::transform_one_electron_adjoint(
      context, dh.view(), dc.view(), da.view(), dgradient.view());
  const Matrix expected_gradient = core * c * (upstream + upstream.transpose());
  near(download(context, dgradient), expected_gradient,
       "CUDA one-electron adjoint differs from oracle");
  near_scalar((expected_gradient.array() * direction.array()).sum(),
              (upstream.array() * expected_direction.array()).sum(),
              "CUDA one-electron adjoint fails dot test");
  xmvb::cuda::transform_one_electron_directional_adjoint(
      context, dh.view(), dc.view(), dd.view(), da.view(), dda.view(),
      dgradient.view());
  const Matrix expected_hvp = core * direction *
          (upstream + upstream.transpose()) +
      core * c * (upstream_direction + upstream_direction.transpose());
  near(download(context, dgradient), expected_hvp,
       "CUDA one-electron changing adjoint differs from oracle");

  const Matrix density = symmetric_values(fixture.n_bf, 1.37);
  const Matrix density_direction = symmetric_values(fixture.n_bf, 1.93);
  DeviceMatrix dp(context, fixture.n_bf, fixture.n_bf);
  DeviceMatrix ddp(context, fixture.n_bf, fixture.n_bf);
  DeviceMatrix dout(context, fixture.n_bf, fixture.n_bf);
  DeviceMatrix dheff(context, fixture.n_bf, fixture.n_bf);
  xmvb::cuda::Buffer<double> denergy(context, 1);
  upload(context, dp, density);
  upload(context, ddp, density_direction);
  gpu.contract_jk(dp.view(), dout.view());
  near(download(context, dout), reference_jk(fixture, density),
       "CUDA exact JK differs from four-index oracle");
  gpu.contract_jk(dp.view(), dout.view(), 1.0, -0.5);
  near(download(context, dout), reference_jk(fixture, density, 1.0, -0.5),
       "CUDA weighted exact JK differs from four-index oracle");
  gpu.contract_jk(dp.view(),dout.view(),2.0,-1.0,dheff.view());
  near(download(context,dout),reference_jk(fixture,density),"CUDA shared JK differs from oracle");
  near(download(context,dheff),reference_jk(fixture,density,1.0,0.0),
       "CUDA reused Coulomb image differs from oracle");
  {
    const Matrix nonsymmetric=values(fixture.n_bf,fixture.n_bf,1.71);
    DeviceMatrix padded(context,fixture.n_bf+2,fixture.n_bf);
    DeviceMatrix image(context,fixture.n_bf+1,fixture.n_bf);
    const xmvb::linalg::MatrixView pview{padded.data(),fixture.n_bf,fixture.n_bf,fixture.n_bf+2};
    const xmvb::linalg::MatrixView oview{image.data(),fixture.n_bf,fixture.n_bf,fixture.n_bf+1};
    xmvb::cuda::copy_to_device(context,pview,nonsymmetric);
    gpu.contract_jk(pview,oview,0.0,1.0);
    Matrix actual(fixture.n_bf,fixture.n_bf);
    xmvb::cuda::copy_to_host(context,actual,oview);
    near(actual,reference_jk(fixture,nonsymmetric,0.0,1.0),
        "CUDA exchange transposed the supplied density row");
    gpu.contract_jk_batch(pview,oview,0.0,1.0);
    xmvb::cuda::copy_to_host(context,actual,oview);
    near(actual,reference_jk(fixture,nonsymmetric,0.0,1.0),
        "CUDA single-channel exchange batch differs from scalar contraction");
  }
  Matrix density_batch(fixture.n_bf,3*fixture.n_bf);
  density_batch << density,density_direction,(density+0.37*density_direction).eval();
  DeviceMatrix dbatch(context,fixture.n_bf,3*fixture.n_bf),obatch(context,fixture.n_bf,3*fixture.n_bf);
  upload(context,dbatch,density_batch);
  for (const auto weights:std::vector<std::pair<double,double>>{{2.0,-1.0},{0.0,-1.0},{1.0,0.0}}) {
    gpu.contract_jk_batch(dbatch.view(),obatch.view(),weights.first,weights.second);
    const Matrix result=download(context,obatch);
    for (int channel=0;channel<3;++channel)
      near(result.middleCols(channel*fixture.n_bf,fixture.n_bf),
          reference_jk(fixture,density_batch.middleCols(channel*fixture.n_bf,fixture.n_bf),
              weights.first,weights.second),"CUDA batched JK differs from oracle");
  }
  {
    Matrix sparse=Matrix::Zero(fixture.n_bf,fixture.n_bf);
    sparse(0,fixture.n_bf-1)=0.37;
    sparse(fixture.n_bf-1,0)=-0.21;
    upload(context,dp,sparse);
    gpu.contract_jk(dp.view(),dout.view(),0.0,1.0);
    near(download(context,dout),reference_jk(fixture,sparse,0.0,1.0),
        "CUDA exact-zero exchange support changed a nonzero contribution");
    sparse.setZero();upload(context,dp,sparse);
    gpu.contract_jk(dp.view(),dout.view(),0.0,1.0);
    near(download(context,dout),sparse,"CUDA null density left a stale exchange image");
    upload(context,dp,density);
  }
  gpu.core_energy(dp.view(), dh.view(), dheff.view(), denergy.data());
  const Matrix expected_heff = core + reference_jk(fixture, density);
  near(download(context, dheff), expected_heff,
       "CUDA exact effective core differs from oracle");
  double energy = 0.0;
  context.copy_to_host(&energy, denergy.data(), sizeof(double));
  near_scalar(energy, trace_product(density, core + expected_heff),
              "CUDA exact core energy differs from oracle");
  gpu.core_directional(ddp.view(), dheff.view(), dout.view(), denergy.data());
  near(download(context, dout), reference_jk(fixture, density_direction),
       "CUDA effective-core direction differs from oracle");
  context.copy_to_host(&energy, denergy.data(), sizeof(double));
  near_scalar(energy, 2.0 * trace_product(density_direction, expected_heff),
              "CUDA exact core energy direction differs from oracle");

  gpu.active_core_density_adjoint(dc.view(), da.view(), dout.view());
  const Matrix sym = 0.5 * (upstream + upstream.transpose());
  const Matrix expected_density_adjoint =
      reference_jk(fixture, c * sym * c.transpose());
  near(download(context, dout), expected_density_adjoint,
       "CUDA active-core density adjoint differs from oracle");
  const Matrix active_density_direction =
      direction * sym * c.transpose() + c * sym * direction.transpose() +
      c * (0.5 * (upstream_direction + upstream_direction.transpose())) *
          c.transpose();
  gpu.active_core_density_directional_adjoint(
      dc.view(), dd.view(), da.view(), dda.view(), dout.view());
  near(download(context, dout), reference_jk(fixture, active_density_direction),
       "CUDA directional active-core density adjoint differs from oracle");
  DeviceMatrix target(context,fixture.n_bf,2),projected(context,fixture.n_bf,2);
  const Matrix target_host=values(fixture.n_bf,2,0.51);
  upload(context,target,target_host);
  DeviceMatrix response;
  auto exchange=gpu.prepare_exchange(dc.view(),&response,target.view());
  const int active_pairs=active*(active+1)/2;
  DeviceMatrix active_kernel(context,active_pairs,active_pairs);
  xmvb::cuda::Buffer<double> accepted_products(context,fixture.ao_pairs*active_pairs);
  const xmvb::cuda::PairPanel accepted{accepted_products.data(),fixture.ao_pairs,
      active_pairs,active_pairs};
  gpu.forward(dc.view(),active_kernel.view(),accepted);
  for (bool use_j:{false,true}) for (bool use_k:{false,true}) {
    if (use_k && exchange.empty()) continue;
    gpu.active_core_density_adjoint(dc.view(),da.view(),dout.view(),
        use_j ? xmvb::cuda::ConstPairPanel{accepted.data,accepted.rows,accepted.cols,accepted.ld}:
            xmvb::cuda::ConstPairPanel{},
        use_k ? xmvb::cuda::ConstPairPanel{exchange.data(),fixture.ao_pairs,active*active,active*active}:
            xmvb::cuda::ConstPairPanel{});
    near(download(context,dout),expected_density_adjoint,
        "CUDA accepted density adjoint differs from four-index oracle");
  }
  require(response.data()!=nullptr,"CUDA projected density image was not admitted within the test budget");
  if (response.data()) {
    auto standalone=gpu.prepare_density_projection(dc.view(),target.view());
    near(download(context,response),download(context,standalone),
        "CUDA fused and standalone density projections differ");
    const auto apply=[&](const Matrix& factor) {
      upload(context,dd,factor);
      xmvb::cuda::gemm(context,{projected.data(),fixture.n_bf*2,1,fixture.n_bf*2},
          response.view(),{dd.data(),fixture.n_bf*active,1,fixture.n_bf*active});
      return download(context,projected);
    };
    near(apply(direction),reference_jk(fixture,
        direction*c.transpose()+c*direction.transpose())*target_host,
        "CUDA two-sided density projection differs from four-index oracle");
    const Matrix factor=direction*sym+
        0.25*c*(upstream_direction+upstream_direction.transpose());
    near(apply(factor),reference_jk(fixture,active_density_direction)*target_host,
        "CUDA projected density derivative differs from oracle");
  }
}

void check_exchange_row_graph() {
  const Fixture fixture;
  auto left=fixture.unique_left,right=fixture.unique_right;
  auto values=fixture.unique_values;
  // Change primary ERI orientation/order; exchange must derive its own
  // pair orientations from CSR, not silently depend on reference metadata.
  for (std::size_t i=0;i<left.size();++i)
    if (i%2) std::swap(left[i],right[i]);
  std::reverse(left.begin(),left.end());
  std::reverse(right.begin(),right.end());
  std::reverse(values.begin(),values.end());
  auto graph=xmvb::integrals::build_ao_pair_graph(left,right,values,fixture.n_bf);
  // No auxiliary unique list is part of the GPU row-graph contract.
  graph.integral_rows.clear();
  graph.integral_edges.clear();
  xmvb::cuda::Context context(128ULL*1024*1024);
  const auto before=context.counters().current_device_bytes;
  const xmvb::cuda::ExactAoPairOperator gpu(context,graph,fixture.n_bf);
  const auto csr=static_cast<std::size_t>(fixture.ao_pairs+1)*sizeof(int)+
      graph.columns.size()*(sizeof(int)+sizeof(double));
  const auto expected=2*fixture.ao_pairs*sizeof(int)+std::min(csr,
      xmvb::cuda::PackedAoPairMatrix::storage_bytes(fixture.ao_pairs));
  require(context.counters().current_device_bytes-before==expected,
      "CUDA AO operator retained duplicate integral representations");
  Matrix densities(fixture.n_bf,2*fixture.n_bf);
  densities.leftCols(fixture.n_bf)=symmetric_values(fixture.n_bf,0.41);
  densities.rightCols(fixture.n_bf)=symmetric_values(fixture.n_bf,1.23);
  DeviceMatrix density(context,fixture.n_bf,2*fixture.n_bf);
  DeviceMatrix response(context,fixture.n_bf,2*fixture.n_bf);
  upload(context,density,densities);
  gpu.contract_jk_batch(density.view(),response.view());
  const auto actual=download(context,response);
  for (int channel=0;channel<2;++channel)
    near(actual.middleCols(channel*fixture.n_bf,fixture.n_bf),
        reference_jk(fixture,densities.middleCols(channel*fixture.n_bf,fixture.n_bf)),
        "CUDA reordered row-graph exchange differs from four-index oracle");
  {
    int device=0,shared_bytes=0;
    xmvb::cuda::check_cuda(cudaGetDevice(&device),"query tiled exchange test device");
    xmvb::cuda::check_cuda(cudaDeviceGetAttribute(&shared_bytes,
        cudaDevAttrMaxSharedMemoryPerBlock,device),"query tiled exchange test capacity");
    const int channels=shared_bytes/(2*fixture.n_bf*sizeof(double))+3;
    Matrix batch(fixture.n_bf,fixture.n_bf*channels);
    for (int c=0;c<channels;++c)
      batch.middleCols(c*fixture.n_bf,fixture.n_bf)=
          symmetric_values(fixture.n_bf,0.11+0.017*c);
    DeviceMatrix dbatch(context,fixture.n_bf,fixture.n_bf*channels);
    DeviceMatrix obatch(context,fixture.n_bf,fixture.n_bf*channels);
    upload(context,dbatch,batch);
    gpu.contract_jk_batch(dbatch.view(),obatch.view());
    const auto image=download(context,obatch);
    for (int c=0;c<channels;++c)
      near(image.middleCols(c*fixture.n_bf,fixture.n_bf),
          reference_jk(fixture,batch.middleCols(c*fixture.n_bf,fixture.n_bf)),
          "CUDA shared-memory channel-tiled exchange differs from oracle");
  }
}

void check_symmetric_tiles() {
  constexpr int nb=46,pairs=nb*(nb+1)/2,channels=3,source_stride=5,target_stride=7;
  Matrix kernel=Matrix::Zero(pairs,pairs),source(pairs,channels);
  std::vector<int> left,right;
  std::vector<double> values;
  for (int p=0;p<pairs;++p) {
    for (int c=0;c<channels;++c) source(p,c)=std::sin(0.003*(p+1)*(c+2));
    for (int q=0;q<=p;++q) {
      if ((p+q)%7==0) continue;
      const double v=0.07*std::cos(0.00017*(p+1)*(q+1));
      kernel(p,q)=kernel(q,p)=v;
      left.push_back(p);right.push_back(q);values.push_back(v);
    }
  }
  auto graph=xmvb::integrals::build_ao_pair_graph(left,right,values,nb);
  xmvb::cuda::Context context(64ULL*1024*1024);
  const xmvb::cuda::PackedAoPairMatrix packed(context,graph,nb);
  require(context.counters().current_device_bytes==
      xmvb::cuda::PackedAoPairMatrix::storage_bytes(pairs),
      "CUDA packed integral matrix retained a duplicate CSR");
  const Matrix expected=kernel*source;
  std::vector<double> input(pairs*source_stride,-71.0);
  for (int p=0;p<pairs;++p) for (int c=0;c<channels;++c)
    input[p*source_stride+c]=source(p,c);
  xmvb::cuda::Buffer<double> device_input(context,input.size());
  context.copy_to_device(device_input.data(),input.data(),input.size()*sizeof(double));
  const std::vector<std::pair<int,int>> intervals{{0,pairs},{1,pairs-2},{511,5},
      {512,512},{1022,59},{1079,2}};
  for (auto [first,count]:intervals) {
    std::vector<double> output(count*target_stride,-83.0);
    xmvb::cuda::Buffer<double> device_output(context,output.size());
    context.copy_to_device(device_output.data(),output.data(),output.size()*sizeof(double));
    packed.apply({device_input.data(),pairs,channels,source_stride},
        {device_output.data(),count,channels,target_stride},first);
    context.copy_to_host(output.data(),device_output.data(),output.size()*sizeof(double));
    Matrix actual(count,channels);
    for (int p=0;p<count;++p) {
      for (int c=0;c<channels;++c) actual(p,c)=output[p*target_stride+c];
      for (int c=channels;c<target_stride;++c)
        require(output[p*target_stride+c]==-83.0,"CUDA packed action overwrote stride padding");
    }
    near(actual,expected.middleRows(first,count),
        "CUDA multi-tile symmetric pair action differs from oracle");
  }
  DeviceMatrix plane(context,nb+2,nb*2);
  const xmvb::linalg::MatrixView view{plane.data(),nb,nb*2,nb+2};
  for (int mu:{0,nb-1}) {
    packed.ao_plane(mu,view,3);
    Matrix actual(nb,nb*2),reference(nb,nb*2);
    xmvb::cuda::copy_to_host(context,actual,view);
    for (int nu=0;nu<2;++nu) for (int k=0;k<nb;++k) for (int l=0;l<nb;++l)
      reference(l,nu*nb+k)=kernel(pair_index(mu,nu+3),pair_index(k,l));
    near(actual,reference,"CUDA sliced packed AO plane differs from oracle");
  }
  auto ul=left,ur=right;
  auto uv=values;
  for (std::size_t k=0;k<ul.size();++k) if (k%2) std::swap(ul[k],ur[k]);
  for (int k=0;k<17;++k) {
    ul.push_back(ul[k]);ur.push_back(ur[k]);uv.push_back(0.25*uv[k]);uv[k]*=0.75;
  }
  const auto unique=xmvb::integrals::build_ao_pair_graph(std::move(ul),std::move(ur),
      std::move(uv),nb,xmvb::integrals::AoPairLayout::Unique);
  require(!unique.has_row_graph() && unique.has_compact_integral_stream(),
      "unique input contract constructed a symmetric row graph");
  const auto available=context.device_byte_budget()-context.counters().current_device_bytes;
  const auto payload=xmvb::cuda::PackedAoPairMatrix::storage_bytes(pairs);
  constexpr std::size_t staging=65536*(2*sizeof(int)+sizeof(double));
  require(available>payload+staging,"unique tile fixture lacks staging budget");
  xmvb::cuda::Buffer<unsigned char> padding(context,available-payload-staging);
  const auto before=context.counters().host_to_device_bytes;
  const xmvb::cuda::PackedAoPairMatrix streamed(context,unique,nb);
  require(context.counters().host_to_device_bytes-before==
      unique.integral_count()*(2*sizeof(int)+sizeof(double)),
      "unique packing uploaded duplicated symmetric ERIs");
  xmvb::cuda::Buffer<double> unique_output(context,pairs*target_stride);
  streamed.apply({device_input.data(),pairs,channels,source_stride},
      {unique_output.data(),pairs,channels,target_stride});
  std::vector<double> output(pairs*target_stride);
  context.copy_to_host(output.data(),unique_output.data(),output.size()*sizeof(double));
  Matrix actual(pairs,channels);
  for (int p=0;p<pairs;++p) for (int c=0;c<channels;++c)
    actual(p,c)=output[p*target_stride+c];
  near(actual,expected,"unique multi-chunk packing differs from full symmetric oracle");
  require(context.counters().peak_device_bytes<=context.device_byte_budget(),
      "unique packing exceeded the shared device budget");
}

void check_empty_unique() {
  const auto graph=xmvb::integrals::build_ao_pair_graph({}, {}, {},4,
      xmvb::integrals::AoPairLayout::Unique);
  require(graph.has_compact_integral_stream() && !graph.has_row_graph() && !graph.integral_count(),
      "empty unique operator lost its layout contract");
  xmvb::cuda::Context context(1024*1024);
  const xmvb::cuda::ExactAoPairOperator zero(context,graph,4);
  DeviceMatrix p(context,4,4),out(context,4,4);
  upload(context,p,values(4,4,0.73));
  zero.contract_jk(p.view(),out.view());
  near(download(context,out),Matrix::Zero(4,4),"empty unique JK operator is not zero");
  bool rejected=false;
  try {
    check_chunk_consumers();
    (void)xmvb::integrals::build_ao_pair_graph({10},{0},{1.0},4,
        xmvb::integrals::AoPairLayout::Unique);
  } catch (const std::invalid_argument&) { rejected=true; }
  require(rejected,"unique builder accepted an out-of-range AO pair");
}

void check_wide_density_projection() {
  const Fixture fixture;
  xmvb::cuda::Context context(128ULL*1024*1024);
  const xmvb::cuda::ExactAoPairOperator gpu(context,fixture.graph(false),fixture.n_bf);
  int device=0,shared_bytes=0;
  xmvb::cuda::check_cuda(cudaGetDevice(&device),"query half projection test device");
  xmvb::cuda::check_cuda(cudaDeviceGetAttribute(&shared_bytes,
      cudaDevAttrMaxSharedMemoryPerBlock,device),"query half projection test capacity");
  // Redundant factor columns are valid in a density map. Crossing the device
  // panel capacity tests channel tails and multiple source-row partitions.
  const int columns=shared_bytes/(fixture.n_bf*sizeof(double))+3;
  const Matrix c=values(fixture.n_bf,columns,0.23);
  const Matrix v=values(fixture.n_bf,columns,0.81);
  const Matrix target=values(fixture.n_bf,2,1.31);
  DeviceMatrix dc(context,fixture.n_bf,columns),dv(context,fixture.n_bf,columns);
  DeviceMatrix dt(context,fixture.n_bf,2),out(context,fixture.n_bf,2);
  upload(context,dc,c);upload(context,dv,v);upload(context,dt,target);
  auto response=gpu.prepare_density_projection(dc.view(),dt.view());
  require(response.data()!=nullptr,"CUDA wide-factor density projection was not admitted");
  xmvb::cuda::gemm(context,{out.data(),fixture.n_bf*2,1,fixture.n_bf*2},
      response.view(),{dv.data(),fixture.n_bf*columns,1,fixture.n_bf*columns});
  near(download(context,out),reference_jk(fixture,v*c.transpose()+c*v.transpose())*target,
      "CUDA partitioned and channel-tiled half projection differs from oracle");
}

void check_budgeted_pair_tiles(bool compact) {
  const Fixture fixture;
  xmvb::cuda::Context context(1024ULL*1024ULL);
  const xmvb::cuda::ExactAoPairOperator gpu(context,fixture.graph(compact),fixture.n_bf);
  constexpr int active=3,pairs=6;
  const Matrix c=values(fixture.n_bf,active,0.31);
  const Matrix d=values(fixture.n_bf,active,1.17);
  DeviceMatrix dc(context,fixture.n_bf,active),dd(context,fixture.n_bf,active);
  DeviceMatrix kernel(context,pairs,pairs);
  upload(context,dc,c);upload(context,dd,d);
  const Matrix b=features(fixture,c),db=directional_features(fixture,c,d);
  const auto remaining=context.device_byte_budget()-context.counters().current_device_bytes;
  const std::size_t matrix_bytes=pairs*pairs*sizeof(double);
  const std::size_t two_row_bytes=2*2*pairs*sizeof(double);
  {
    // Reserve precisely the output scratch allowance and two pair rows.
    xmvb::cuda::Buffer<unsigned char> padding(context,remaining-matrix_bytes-two_row_bytes);
    gpu.forward(dc.view(),kernel.view());
    near(download(context,kernel),b.transpose()*fixture.pair_kernel*b,
         "CUDA budgeted forward tiles differ");
  }
  {
    // Directional also owns one active-pair square before selecting rows.
    xmvb::cuda::Buffer<unsigned char> padding(context,remaining-2*matrix_bytes-two_row_bytes);
    gpu.directional(dc.view(),dd.view(),kernel.view());
    near(download(context,kernel),db.transpose()*fixture.pair_kernel*b+
         b.transpose()*fixture.pair_kernel*db,"CUDA budgeted directional tiles differ");
  }
  if (!compact) {
    // Leave room for at most two AO-row feature channels, much less than the
    // six-channel full feature panel. Whether cuSPARSE's own scratch fits is
    // device-dependent; either a split SpMM or the exact generated CSR GPU
    // path must produce the same full-row answer without exceeding budget.
    xmvb::cuda::Buffer<double> image(context,fixture.ao_pairs*pairs);
    xmvb::cuda::Buffer<double> dimage(context,fixture.ao_pairs*pairs);
    const auto free_bytes=context.device_byte_budget()-context.counters().current_device_bytes;
    const std::size_t channel_bytes=fixture.ao_pairs*sizeof(double);
    require(free_bytes>2*channel_bytes,"CUDA budget fixture has no row-channel allowance");
    xmvb::cuda::Buffer<unsigned char> padding(context,free_bytes-2*channel_bytes);
    gpu.generated_rows(dc.view(),dd.view(),0,
        {image.data(),fixture.ao_pairs,pairs,pairs},
        {dimage.data(),fixture.ao_pairs,pairs,pairs});
    std::vector<double> actual(fixture.ao_pairs*pairs),actual_direction(actual.size());
    context.copy_to_host(actual.data(),image.data(),actual.size()*sizeof(double));
    context.copy_to_host(actual_direction.data(),dimage.data(),actual_direction.size()*sizeof(double));
    const Matrix expected=fixture.pair_kernel*b;
    const Matrix expected_direction=fixture.pair_kernel*db;
    for (int row=0;row<fixture.ao_pairs;++row)
      for (int col=0;col<pairs;++col) {
        near_scalar(actual[row*pairs+col],expected(row,col),
                    "CUDA low-budget full-row generated image differs");
        near_scalar(actual_direction[row*pairs+col],expected_direction(row,col),
                    "CUDA low-budget full-row directional image differs");
      }
  }
  {
    // Retain one accepted image, but force the changing-adjoint consumer to
    // use two-row direction-only CSR/compact tiles, not another full image.
    xmvb::cuda::Buffer<double> products(context,fixture.ao_pairs*pairs);
    gpu.forward(dc.view(),kernel.view(),{products.data(),fixture.ao_pairs,pairs,pairs});
    const Matrix g=values(pairs,pairs,0.78),dg=values(pairs,pairs,1.36);
    DeviceMatrix upstream(context,pairs,pairs),dupstream(context,pairs,pairs);
    DeviceMatrix gradient(context,fixture.n_bf,active);
    upload(context,upstream,g);upload(context,dupstream,dg);
    const auto free_bytes=context.device_byte_budget()-context.counters().current_device_bytes;
    const std::size_t allowance=2*matrix_bytes+fixture.n_bf*active*sizeof(double)+
        2*4*pairs*sizeof(double);
    require(free_bytes>allowance,"CUDA retained tile fixture has insufficient allowance");
    xmvb::cuda::Buffer<unsigned char> padding(context,free_bytes-allowance);
    require(gpu.reserve_products(active).empty(),"CUDA accepted products ignored the live budget");
    gpu.directional_adjoint(dc.view(),dd.view(),upstream.view(),dupstream.view(),gradient.view(),
        {products.data(),fixture.ao_pairs,pairs,pairs});
    constexpr double step=2e-6;
    const Matrix expected=(pair_gradient(fixture,c+step*d,g+step*dg)-
        pair_gradient(fixture,c-step*d,g-step*dg))/(2*step);
    near(download(context,gradient),expected,"CUDA budgeted retained pair HVP differs",2e-7);
    require(context.counters().peak_device_bytes<=context.device_byte_budget(),
        "CUDA retained pair response exceeded its budget");
  }
}

// Certify the library contract needed by the unique-row AO representation:
// Same-width CSR row/column indices, unsorted columns, FP64 row-major
// panels and both N/T operations, including the scalar SpMV dispatch.
template<class Index>
void check_unique_csr_library() {
  xmvb::cuda::Context context(64ULL*1024*1024);
  const std::vector<Index> offsets{0,2,3,5,6};
  const std::vector<Index> columns{2,0,1,3,1,3};
  const std::vector<double> coefficients{0.8,1.2,-0.2,1.4,0.7,0.3};
  xmvb::cuda::Buffer<Index> row(context,offsets.size()),col(context,columns.size());
  xmvb::cuda::Buffer<double> val(context,coefficients.size());
  context.copy_to_device(row.data(),offsets.data(),offsets.size()*sizeof(offsets[0]));
  context.copy_to_device(col.data(),columns.data(),columns.size()*sizeof(columns[0]));
  context.copy_to_device(val.data(),coefficients.data(),coefficients.size()*sizeof(double));
  struct Descriptors {
    cusparseHandle_t handle=nullptr;
    cusparseSpMatDescr_t matrix=nullptr;
    cusparseDnMatDescr_t source=nullptr,target=nullptr;
    ~Descriptors() {
      if (target) cusparseDestroyDnMat(target);
      if (source) cusparseDestroyDnMat(source);
      if (matrix) cusparseDestroySpMat(matrix);
      if (handle) cusparseDestroy(handle);
    }
  } d;
  const auto check=[](cusparseStatus_t status) {
    require(status==CUSPARSE_STATUS_SUCCESS,"unique CSR library operation failed");
  };
  check(cusparseCreate(&d.handle));
  check(cusparseSetStream(d.handle,context.stream()));
  constexpr auto index_type=sizeof(Index)==8 ? CUSPARSE_INDEX_64I:CUSPARSE_INDEX_32I;
  check(cusparseCreateCsr(&d.matrix,4,4,coefficients.size(),row.data(),col.data(),val.data(),
      index_type,index_type,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
  Matrix u=Matrix::Zero(4,4);
  for (int r=0;r<4;++r)
    for (auto e=offsets[r];e<offsets[r+1];++e) u(r,columns[e])+=coefficients[e];
  for (int width:{1,3}) {
    const auto x=values(4,width,0.79);
    const int source_stride=width+2,target_stride=width+3;
    std::vector<double> source(4*source_stride,-71.0),target(4*target_stride,-83.0);
    for (int r=0;r<4;++r) for (int c=0;c<width;++c) source[r*source_stride+c]=x(r,c);
    xmvb::cuda::Buffer<double> sx(context,source.size()),sy(context,target.size());
    context.copy_to_device(sx.data(),source.data(),source.size()*sizeof(double));
    context.copy_to_device(sy.data(),target.data(),target.size()*sizeof(double));
    check(cusparseCreateDnMat(&d.source,4,width,source_stride,sx.data(),CUDA_R_64F,CUSPARSE_ORDER_ROW));
    check(cusparseCreateDnMat(&d.target,4,width,target_stride,sy.data(),CUDA_R_64F,CUSPARSE_ORDER_ROW));
    const double one=1.0,zero=0.0;
    std::size_t scratch_bytes=0;
    for (auto op:{CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_TRANSPOSE}) {
      std::size_t bytes=0;
      check(cusparseSpMM_bufferSize(d.handle,op,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,
          d.matrix,d.source,&zero,d.target,CUDA_R_64F,CUSPARSE_SPMM_ALG_DEFAULT,&bytes));
      scratch_bytes=std::max(scratch_bytes,bytes);
    }
    xmvb::cuda::Buffer<char> scratch(context,scratch_bytes);
    check(cusparseSpMM(d.handle,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one,d.matrix,d.source,&zero,d.target,CUDA_R_64F,CUSPARSE_SPMM_ALG_DEFAULT,scratch.data()));
    check(cusparseSpMM(d.handle,CUSPARSE_OPERATION_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one,d.matrix,d.source,&one,d.target,CUDA_R_64F,CUSPARSE_SPMM_ALG_DEFAULT,scratch.data()));
    context.copy_to_host(target.data(),sy.data(),target.size()*sizeof(double));
    Matrix result(4,width);
    for (int r=0;r<4;++r) {
      for (int c=0;c<width;++c) result(r,c)=target[r*target_stride+c];
      for (int c=width;c<target_stride;++c)
        require(target[r*target_stride+c]==-83.0,"unique CSR SpMM overwrote padding");
    }
    near(result,(u+u.transpose())*x,"unique CSR N/T action differs from dense oracle");
    check(cusparseDestroyDnMat(d.target));d.target=nullptr;
    check(cusparseDestroyDnMat(d.source));d.source=nullptr;
  }
}

}  // namespace

int main() {
  try {
    check_core_symmetry_orientations();
    check_unique_half_tiles();
    check_unique_csr_library<int>();
    check_unique_csr_library<std::int64_t>();
    // Multiple warps/blocks, empty CSR rows and a final partial warp exercise
    // edge-balanced row lookup, scalar/batched exchange and density adjoints.
    check_one_electron_and_core(true,true,false,12);
    for (bool sparse:{false,true}) {
      check_graph_and_pair(3,true,sparse,true);
      check_one_electron_and_core(true,sparse,true);
    }
    check_packed_exchange_blocks();
    check_exchange_row_graph();
    check_empty_unique();
    check_symmetric_tiles();
    check_wide_density_projection();
    check_graph_and_pair(3,false,true);
    check_one_electron_and_core(false,true);
    check_graph_and_pair(3,true,true);
    check_one_electron_and_core(true,true);
    for (bool compact : {false, true}) {
      check_graph_and_pair(2, compact);
      check_graph_and_pair(3, compact);
      check_one_electron_and_core(compact);
      check_budgeted_pair_tiles(compact);
    }
    std::cout << "CUDA exact AO/active integral oracle: PASS\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
