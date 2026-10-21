#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/Core>

#include "cuda/linalg/blas.hpp"
#include "cuda/optimization/coarse.hpp"
#include "mcscf/optimization/preconditioners/coarse.hpp"

namespace {

using Matrix=Eigen::MatrixXd;
using Vector=Eigen::VectorXd;
using DeviceMatrix=xmvb::cuda::Matrix<double>;

void require(bool condition,const char* message) {
  if (!condition) throw std::runtime_error(message);
}

DeviceMatrix upload(xmvb::cuda::Context& context,const Matrix& host) {
  DeviceMatrix result(context,host.rows(),host.cols());
  xmvb::cuda::copy_to_device(context,result.view(),host);
  return result;
}

DeviceMatrix upload(xmvb::cuda::Context& context,const Vector& host) {
  DeviceMatrix result(context,host.size(),1);
  context.copy_to_device(result.data(),host.data(),host.size()*sizeof(double));
  return result;
}

Vector download(xmvb::cuda::Context& context,
                xmvb::linalg::ConstMatrixView device) {
  Matrix host(device.rows,device.cols);
  xmvb::cuda::copy_to_host(context,host,device);
  return host.col(0);
}

void observe(xmvb::cuda::Context& context,
    xmvb::cuda::DeviceCoarsePreconditioner& device,
    xmvb::mcscf::CoarsePreconditioner& host,
    const Vector& v,const Vector& hv,const Vector& mv) {
  auto dv=upload(context,v),dh=upload(context,hv),dm=upload(context,mv);
  device.observe(dv.view(),dh.view(),dm.view());
  host.observe(v,hv,mv);
  require(device.size()==host.size(),"CUDA coarse admission differs from CPU");
}

void check_balancing() {
  constexpr int n=5;
  Matrix h=Matrix::Zero(n,n),m=Matrix::Identity(n,n),base=Matrix::Zero(n,n);
  h.diagonal()<<-2.0,3.0,5.0,4.0,6.0;
  h(0,3)=h(3,0)=0.7;
  h(1,4)=h(4,1)=0.5;
  h(2,3)=h(3,2)=0.2;
  h(2,0)=h(0,2)=0.1;
  m.diagonal()<<1.0,1.25,1.5,1.75,2.0;
  m(0,1)=m(1,0)=0.1;
  base.diagonal()<<0.2,0.35,0.5,0.65,0.8;
  Matrix sampled=Matrix::Zero(n,4);
  sampled.col(0)=Vector::Unit(n,0);
  sampled.col(1)=Vector::Unit(n,1);
  sampled.col(2)=Vector::Unit(n,3);
  sampled.col(3)=Vector::Unit(n,4);

  xmvb::cuda::Context context(32ULL*1024*1024);
  xmvb::cuda::DeviceCoarsePreconditioner gpu(context,n,4,1e-14);
  xmvb::mcscf::CoarsePreconditioner cpu(n,4,1e-14);
  auto d_base=upload(context,base);
  observe(context,gpu,cpu,sampled.col(0),h*sampled.col(0),m*sampled.col(0));
  observe(context,gpu,cpu,sampled.col(0),h*sampled.col(0),m*sampled.col(0));
  require(gpu.size()==1,"dependent coarse sample was admitted");
  const Vector mixed=sampled.col(0)+sampled.col(1);
  observe(context,gpu,cpu,mixed,h*mixed,m*mixed);
  for (int i=2;i<4;++i)
    observe(context,gpu,cpu,sampled.col(i),h*sampled.col(i),m*sampled.col(i));
  require(gpu.size()==4,"independent coarse samples were rejected");

  for (double eigenvalue:{0.0,-0.2}) {
    Matrix actual(n,n);
    int calls=0;
    for (int i=0;i<n;++i) {
      const Vector rhs=Vector::Unit(n,i);
      auto d_rhs=upload(context,rhs);
      DeviceMatrix d_out(context,n,1);
      gpu.apply(d_rhs.view(),eigenvalue,[&](xmvb::linalg::ConstMatrixView input,
          xmvb::linalg::MatrixView output) {
        ++calls;
        xmvb::cuda::gemm(context,output,d_base.view(),input);
      },d_out.view());
      actual.col(i)=download(context,d_out.view());
      const Vector expected=cpu.apply(rhs,eigenvalue,
          [&](const Vector& residual) -> Vector { return base*residual; });
      require((actual.col(i)-expected).norm()<3e-11,
          "CUDA coarse balancing differs from CPU formula");
    }
    require(calls==n,"CUDA coarse inverse did not call the base inverse once");
    require((actual-actual.transpose()).norm()<3e-11,
        "CUDA coarse balancing inverse lost symmetry");
  }
}

void check_certification() {
  constexpr int n=4096;
  Vector first(n),second=Vector::Constant(n,1.0/64.0);
  for (int i=0;i<n;++i) first[i]=(i%2 ? -1.0:1.0)/64.0;
  for (double skew:{1e-12,1e-5}) {
    xmvb::cuda::Context context(16ULL*1024*1024);
    xmvb::cuda::DeviceCoarsePreconditioner gpu(context,n,2,1e-14);
    xmvb::mcscf::CoarsePreconditioner cpu(n,2,1e-14);
    observe(context,gpu,cpu,first,2*first,first);
    observe(context,gpu,cpu,second,3*second+skew*first,second);
    require(gpu.size()==(skew>1e-6 ? 1:2),
        "CUDA coarse image symmetry certification changed");
    if (skew>1e-6) {
      observe(context,gpu,cpu,second,3*second,second);
      require(gpu.size()==2,"CUDA coarse rejected valid recovery sample");
    }
    auto d_rhs=upload(context,second);
    DeviceMatrix d_out(context,n,1);
    gpu.apply(d_rhs.view(),0.0,[&](xmvb::linalg::ConstMatrixView input,
        xmvb::linalg::MatrixView output) {
      xmvb::cuda::check_cuda(cudaMemcpyAsync(output.data,input.data,
          n*sizeof(double),cudaMemcpyDeviceToDevice,context.stream()),
          "CUDA coarse identity inverse copy");
    },d_out.view());
    const Vector expected=cpu.apply(second,0.0,
        [](const Vector& residual) -> Vector { return residual; });
    require((download(context,d_out.view())-expected).norm()<3e-10,
        "CUDA coarse rejection damaged retained inverse");
  }
}

}  // namespace

int main() {
  try {
    check_balancing();
    check_certification();
    std::cout<<"CUDA coupled coarse correction: PASS\n";
  } catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
  }
}
