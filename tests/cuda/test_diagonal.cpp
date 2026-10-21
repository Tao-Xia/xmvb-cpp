#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "cuda/integrals/exact.hpp"
#include "cuda/memory/buffer.hpp"
#include "cuda/vb/diagonal.hpp"
#include "integrals/ao/graph.hpp"
#include "mcscf/optimization/preconditioners/orbital_diagonal.hpp"

namespace {

using Matrix=Eigen::MatrixXd;
using DeviceMatrix=xmvb::cuda::Matrix<double>;

int pair(int a,int b) {
  if (a<b) std::swap(a,b);
  return a*(a+1)/2+b;
}

struct Fixture {
  static constexpr int nbf=5;
  static constexpr int npair=nbf*(nbf+1)/2;
  Matrix graph=Matrix::Zero(npair,npair);
  std::vector<int> left,right;
  std::vector<double> values;

  Fixture() {
    for (int p=0;p<npair;++p) for (int q=0;q<=p;++q) {
      const double value=0.06*std::cos(0.23*(p+1)*(q+2))+
          0.008*std::sin(0.19*(p+2)*(q+1));
      graph(p,q)=graph(q,p)=value;
      left.push_back(p); right.push_back(q); values.push_back(value);
    }
  }

  xmvb::integrals::AoPairGraph pair_graph(bool compact) const {
    if (!compact)
      return xmvb::integrals::build_ao_pair_graph(left,right,values,nbf);
    xmvb::integrals::AoPairGraph result;
    for (int a=0;a<nbf;++a) for (int b=0;b<=a;++b) {
      result.pair_first.push_back(a);
      result.pair_second.push_back(b);
    }
    result.integral_rows=left;
    result.integral_columns=right;
    result.integral_values=values;
    return result;
  }

  double eri(int a,int b,int c,int d) const {
    return graph(pair(a,b),pair(c,d));
  }

  double mo_eri(const Matrix& mo,int p,int q,int r,int s) const {
    double result=0.0;
    for (int a=0;a<nbf;++a) for (int b=0;b<nbf;++b)
      for (int c=0;c<nbf;++c) for (int d=0;d<nbf;++d)
        result+=mo(a,p)*mo(b,q)*mo(c,r)*mo(d,s)*eri(a,b,c,d);
    return result;
  }
};

void check(bool compact,int ncore) {
  constexpr int na=2;
  const Fixture fixture;
  xmvb::cuda::Context context(128ULL*1024ULL*1024ULL);
  xmvb::cuda::ExactAoPairOperator graph(context,fixture.pair_graph(compact),
      Fixture::nbf);
  Matrix mo=Matrix::Identity(Fixture::nbf,Fixture::nbf);
  const auto rotate=[&](int p,int q,double theta) {
    const Matrix left=mo.col(p),right=mo.col(q);
    mo.col(p)=std::cos(theta)*left+std::sin(theta)*right;
    mo.col(q)=-std::sin(theta)*left+std::cos(theta)*right;
  };
  rotate(0,2,0.23); rotate(1,4,-0.17); rotate(2,3,0.11);
  Matrix core=Matrix::Zero(Fixture::nbf,Fixture::nbf);
  for (int a=0;a<Fixture::nbf;++a) for (int b=0;b<Fixture::nbf;++b)
    core(a,b)=(a==b ? -1.1+0.22*a:0.0)+
        0.045*std::cos(0.27*(a+1)*(b+1));
  Matrix pcore=Matrix::Zero(Fixture::nbf,Fixture::nbf);
  for (int c=0;c<ncore;++c) pcore.noalias()+=mo.col(c)*mo.col(c).transpose();
  Matrix vcore=Matrix::Zero(Fixture::nbf,Fixture::nbf);
  for (int a=0;a<Fixture::nbf;++a) for (int b=0;b<Fixture::nbf;++b)
    for (int c=0;c<Fixture::nbf;++c) for (int d=0;d<Fixture::nbf;++d)
      vcore(a,b)+=(2.0*fixture.eri(a,b,c,d)-fixture.eri(a,c,b,d))*pcore(c,d);
  const Matrix effective=core+vcore;
  Matrix dm1(na,na);
  dm1 << 1.13,0.16,0.16,0.71;
  constexpr int nactivepairs=na*(na+1)/2;
  Matrix pair_adjoint(nactivepairs,nactivepairs);
  for (int p=0;p<nactivepairs;++p) for (int q=0;q<nactivepairs;++q)
    pair_adjoint(p,q)=0.2*std::cos(0.38*(p+1)*(q+1));
  pair_adjoint=0.5*(pair_adjoint+pair_adjoint.transpose()).eval();
  Matrix active_pair(nactivepairs,nactivepairs);
  for (int a=0;a<na;++a) for (int b=0;b<=a;++b)
    for (int c=0;c<na;++c) for (int d=0;d<=c;++d)
      active_pair(pair(a,b),pair(c,d))=fixture.mo_eri(mo,
          ncore+a,ncore+b,ncore+c,ncore+d);
  xmvb::mcscf::AnalyticDiagonalIntermediates x;
  x.n_core=ncore; x.n_active=na;
  x.h_core=mo.transpose()*core*mo;
  x.v_core=mo.transpose()*vcore*mo;
  x.j_pc=Matrix(Fixture::nbf,ncore);
  x.k_pc=Matrix(Fixture::nbf,ncore);
  x.j_active.resize(Fixture::nbf*na*na);
  x.k_active.resize(Fixture::nbf*na*na);
  x.active_eri.resize(na*na*na*na);
  for (int p=0;p<Fixture::nbf;++p) {
    for (int c=0;c<ncore;++c) {
      x.j_pc(p,c)=fixture.mo_eri(mo,p,p,c,c);
      x.k_pc(p,c)=fixture.mo_eri(mo,p,c,p,c);
    }
    for (int a=0;a<na;++a) for (int b=0;b<na;++b) {
      const auto index=xmvb::mcscf::orbital_diagonal_slice_index(p,a,b,na);
      x.j_active[index]=fixture.mo_eri(mo,p,p,ncore+a,ncore+b);
      x.k_active[index]=fixture.mo_eri(mo,p,ncore+a,p,ncore+b);
    }
  }
  for (int a=0;a<na;++a) for (int b=0;b<na;++b)
    for (int c=0;c<na;++c) for (int d=0;d<na;++d)
      x.active_eri[xmvb::mcscf::orbital_diagonal_tensor4_index(a,b,c,d,na)]=
          active_pair(pair(a,b),pair(c,d));
  std::vector<double> packed(xmvb::mcscf::orbital_diagonal_packed_eri_count(na));
  for (int p=0;p<nactivepairs;++p) for (int q=0;q<=p;++q)
    packed[p*(p+1)/2+q]=p==q ? pair_adjoint(p,q):
        pair_adjoint(p,q)+pair_adjoint(q,p);
  const Matrix expected=xmvb::mcscf::build_analytic_rotation_hessian_diagonal(
      x,dm1,packed);

  DeviceMatrix dmo(context,Fixture::nbf,Fixture::nbf);
  DeviceMatrix dcore(context,Fixture::nbf,Fixture::nbf);
  DeviceMatrix deffective(context,Fixture::nbf,Fixture::nbf);
  DeviceMatrix ddm1(context,na,na);
  DeviceMatrix dadjoint(context,nactivepairs,nactivepairs);
  DeviceMatrix dpair(context,nactivepairs,nactivepairs);
  xmvb::cuda::copy_to_device(context,dmo.view(),mo);
  xmvb::cuda::copy_to_device(context,dcore.view(),core);
  xmvb::cuda::copy_to_device(context,deffective.view(),effective);
  xmvb::cuda::copy_to_device(context,ddm1.view(),dm1);
  xmvb::cuda::copy_to_device(context,dadjoint.view(),pair_adjoint);
  xmvb::cuda::copy_to_device(context,dpair.view(),active_pair);
  if (!compact && ncore>0) {
    DeviceMatrix dj(context,Fixture::nbf,ncore),dr(context,Fixture::nbf,ncore);
    Matrix expected_j(Fixture::nbf,ncore),expected_r(Fixture::nbf,ncore);
    for (int p=0;p<Fixture::nbf;++p) for (int c=0;c<ncore;++c) {
      expected_j(p,c)=fixture.mo_eri(mo,p,p,c,c);
      expected_r(p,c)=2.0*expected_j(p,c)-fixture.mo_eri(mo,p,c,p,c);
    }
    const auto check_projected=[&] {
      if (!graph.core_diagonal(dmo.view(),ncore,dj.view(),dr.view()))
        throw std::runtime_error("CUDA row-graph core projection was not admitted");
      Matrix j(Fixture::nbf,ncore),r(Fixture::nbf,ncore);
      xmvb::cuda::copy_to_host(context,j,dj.view());
      xmvb::cuda::copy_to_host(context,r,dr.view());
      if ((j-expected_j).norm()>2e-10 || (r-expected_r).norm()>2e-10)
        throw std::runtime_error("CUDA direct core projection differs from four-index oracle");
    };
    check_projected();
    Matrix expected_density_response;
    {
      xmvb::cuda::Buffer<double> response;
      if (!graph.core_diagonal(dmo.view(),ncore,dj.view(),dr.view(),&response) ||
          !response.data())
        throw std::runtime_error("CUDA fused core density response was not admitted");
      Matrix j(Fixture::nbf,ncore),r(Fixture::nbf,ncore);
      xmvb::cuda::copy_to_host(context,j,dj.view());
      xmvb::cuda::copy_to_host(context,r,dr.view());
      if ((j-expected_j).norm()>2e-10 || (r-expected_r).norm()>2e-10)
        throw std::runtime_error("CUDA shared core image differs from four-index oracle");
      auto standalone=graph.prepare_density_projection(dmo.view().columns(0,ncore),
          dmo.view().columns(0,ncore));
      const int dimension=Fixture::nbf*ncore;
      Matrix fused(dimension,dimension),reference(dimension,dimension);
      std::vector<double> packed_response(response.size());
      context.copy_to_host(packed_response.data(),response.data(),response.size()*sizeof(double));
      for (int i=0;i<dimension;++i) for (int j=0;j<dimension;++j) {
        const int high=std::max(i,j),low=std::min(i,j);
        fused(i,j)=packed_response[high*(high+1)/2+low];
      }
      xmvb::cuda::copy_to_host(context,reference,standalone.view());
      if ((fused-reference).norm()>2e-10)
        throw std::runtime_error("CUDA fused core density response differs from separate transform");
      expected_density_response=std::move(fused);
    }
    const auto available=context.device_byte_budget()-context.counters().current_device_bytes;
    const std::size_t one_channel=2*(2*Fixture::npair+Fixture::nbf*Fixture::nbf+
        Fixture::nbf)*sizeof(double);
    {
      xmvb::cuda::Buffer<unsigned char> padding(context,available-one_channel);
      check_projected(); // Forces one core channel per tile, with identical physics.
      if (context.counters().peak_device_bytes>context.device_byte_budget())
        throw std::runtime_error("CUDA core projection exceeded the live budget");
    }
    if (ncore==2) {
      // Admit the response map, but force its half-transform into proper
      // core-column slices. This exercises nonzero offsets in the shared U view.
      const std::size_t dimension=Fixture::nbf*ncore;
      const std::size_t count=dimension*(dimension+1)/2;
      const std::size_t scratch=Fixture::nbf*Fixture::nbf*ncore+
          2*Fixture::nbf*ncore*ncore+Fixture::nbf*Fixture::nbf;
      xmvb::cuda::Buffer<unsigned char> padding(context,
          available-(2*count+scratch)*sizeof(double));
      xmvb::cuda::Buffer<double> response;
      if (!graph.core_diagonal(dmo.view(),ncore,dj.view(),dr.view(),&response) ||
          response.empty())
        throw std::runtime_error("CUDA tiled core response was not admitted");
      Matrix j(Fixture::nbf,ncore),r(Fixture::nbf,ncore);
      xmvb::cuda::copy_to_host(context,j,dj.view());
      xmvb::cuda::copy_to_host(context,r,dr.view());
      if ((j-expected_j).norm()>2e-10 || (r-expected_r).norm()>2e-10)
        throw std::runtime_error("CUDA tiled shared core image differs from oracle");
      std::vector<double> packed_response(response.size());
      context.copy_to_host(packed_response.data(),response.data(),response.size()*sizeof(double));
      Matrix actual_response(dimension,dimension);
      for (std::size_t i=0;i<dimension;++i) for (std::size_t j=0;j<dimension;++j) {
        const auto high=std::max(i,j),low=std::min(i,j);
        actual_response(i,j)=packed_response[high*(high+1)/2+low];
      }
      if ((actual_response-expected_density_response).norm()>2e-10)
        throw std::runtime_error("CUDA tiled core response differs from complete-column projection");
    }
  }
  const DeviceMatrix actual_device=xmvb::cuda::build_analytic_rotation_diagonal(
      context,graph,dmo.view(),dcore.view(),deffective.view(),ddm1.view(),
      dadjoint.view(),dpair.view(),ncore);
  Matrix actual(Fixture::nbf,Fixture::nbf);
  xmvb::cuda::copy_to_host(context,actual,actual_device.view());
  if (!actual.allFinite() ||
      (actual-expected).norm()>2e-9*std::max(1.0,expected.norm())) {
    std::cerr << "compact=" << compact << " core=" << ncore
              << " error=" << (actual-expected).norm()
              << " expected_norm=" << expected.norm() << '\n';
    throw std::runtime_error("GPU analytic Hessian diagonal differs from CPU oracle");
  }
  auto products=graph.reserve_products(na);
  if (products.empty()) throw std::runtime_error("diagonal fixture lacks retained product memory");
  DeviceMatrix prepared_pair(context,nactivepairs,nactivepairs);
  graph.forward(dmo.view().columns(ncore,na),prepared_pair.view(),
      {products.data(),graph.pair_count(),nactivepairs,nactivepairs});
  auto exchange=graph.prepare_exchange(dmo.view().columns(ncore,na));
  const auto reused=xmvb::cuda::build_analytic_rotation_diagonal(
      context,graph,dmo.view(),dcore.view(),deffective.view(),ddm1.view(),
      dadjoint.view(),dpair.view(),ncore,
      {products.data(),graph.pair_count(),nactivepairs,nactivepairs},
      exchange.empty() ? xmvb::cuda::ConstPairPanel{}:
          xmvb::cuda::ConstPairPanel{exchange.data(),graph.pair_count(),na*na,na*na});
  xmvb::cuda::copy_to_host(context,actual,reused.view());
  if ((actual-expected).norm()>2e-9*std::max(1.0,expected.norm()))
    throw std::runtime_error("GPU retained analytic Hessian diagonal differs from CPU oracle");
}

}  // namespace

int main() {
  try {
    check(false,0);
    check(false,1);
    check(false,2);
    check(false,3);
    check(true,1);
    check(true,2);
    std::cout << "CUDA analytic orbital diagonal passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
