#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "cuda/integrals/exact.hpp"
#include "cuda/integrals/one_electron.hpp"
#include "cuda/memory/buffer.hpp"
#include "integrals/ao/graph.hpp"

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
  int n_bf = 4;
  int ao_pairs = 10;
  Matrix pair_kernel;
  std::vector<int> unique_left, unique_right;
  std::vector<double> unique_values;
  std::vector<int> first, second;

  Fixture() : pair_kernel(ao_pairs, ao_pairs) {
    pair_kernel.setZero();
    for (int i = 0; i < n_bf; ++i)
      for (int j = 0; j <= i; ++j) {
        first.push_back(i);
        second.push_back(j);
      }
    for (int p = 0; p < ao_pairs; ++p)
      for (int q = 0; q <= p; ++q) {
        const double value =
            0.07 * std::cos(0.19 * (p + 1) * (q + 2)) +
            0.011 / static_cast<double>(p + q + 1);
        unique_left.push_back(p);
        unique_right.push_back(q);
        unique_values.push_back(value);
        pair_kernel(p, q) = pair_kernel(q, p) = value;
      }
  }

  xmvb::integrals::AoPairGraph graph(bool compact) const {
    if (!compact)
      return xmvb::integrals::build_ao_pair_graph(
          unique_left, unique_right, unique_values, n_bf);
    xmvb::integrals::AoPairGraph result;
    result.pair_first = first;
    result.pair_second = second;
    result.integral_rows = unique_left;
    result.integral_columns = unique_right;
    result.integral_values = unique_values;
    return result;
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

Matrix pair_gradient(const Fixture& fixture, const Matrix& c,
                     const Matrix& adjoint) {
  const Matrix product = fixture.pair_kernel * features(fixture, c);
  const Matrix tile_gradient = product * (adjoint + adjoint.transpose());
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

double trace_product(const Matrix& left, const Matrix& right) {
  return (left.array() * right.transpose().array()).sum();
}

void check_graph_and_pair(int active, bool compact) {
  const Fixture fixture;
  xmvb::cuda::Context context(128ULL * 1024ULL * 1024ULL);
  const xmvb::cuda::ExactAoPairOperator gpu(context, fixture.graph(compact),
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
  xmvb::cuda::Buffer<double> d_full_p(context,fixture.ao_pairs*pairs);
  xmvb::cuda::Buffer<double> d_full_dp(context,fixture.ao_pairs*pairs);
  gpu.generated_rows(dc.view(),dd.view(),0,
      {d_full_p.data(),fixture.ao_pairs,pairs,pairs},
      {d_full_dp.data(),fixture.ao_pairs,pairs,pairs});
  std::vector<double> full_p(fixture.ao_pairs*pairs),full_dp(full_p.size());
  context.copy_to_host(full_p.data(),d_full_p.data(),full_p.size()*sizeof(double));
  context.copy_to_host(full_dp.data(),d_full_dp.data(),full_dp.size()*sizeof(double));
  for (int row=0;row<fixture.ao_pairs;++row)
    for (int col=0;col<pairs;++col) {
      near_scalar(full_p[row*pairs+col],p(row,col),
                  "CUDA full-row CSR generated image differs from oracle");
      near_scalar(full_dp[row*pairs+col],dp(row,col),
                  "CUDA full-row CSR directional image differs from oracle");
    }

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
}

void check_one_electron_and_core(bool compact) {
  const Fixture fixture;
  constexpr int active = 3;
  xmvb::cuda::Context context(128ULL * 1024ULL * 1024ULL);
  const xmvb::cuda::ExactAoPairOperator gpu(context, fixture.graph(compact),
                                            fixture.n_bf);
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

}  // namespace

int main() {
  try {
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
