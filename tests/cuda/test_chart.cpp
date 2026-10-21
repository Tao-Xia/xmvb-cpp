#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "cuda/memory/buffer.hpp"
#include "cuda/vb/chart.hpp"

namespace {

using Matrix=Eigen::MatrixXd;
using DeviceMatrix=xmvb::cuda::Matrix<double>;

void near(const Matrix& actual,const Matrix& expected,const char* label) {
  if (actual.rows()!=expected.rows() || actual.cols()!=expected.cols() ||
      !actual.allFinite() ||
      (actual-expected).norm()>2e-13*std::max(1.0,expected.norm()))
    throw std::runtime_error(label);
}

DeviceMatrix upload(xmvb::cuda::Context& context,const Matrix& host) {
  DeviceMatrix result(context,host.rows(),host.cols());
  xmvb::cuda::copy_to_device(context,result.view(),host);
  return result;
}

Matrix download(xmvb::cuda::Context& context,
                xmvb::linalg::ConstMatrixView source) {
  Matrix result(source.rows,source.cols);
  xmvb::cuda::copy_to_host(context,result,source);
  return result;
}

void check_map() {
  xmvb::cuda::Context context;
  // Exact block values would normally be exported from one accepted
  // OrbitalChart. The GPU map must not infer or alter its quotient.
  const std::vector<xmvb::cuda::OrbitalTangentBlock> blocks{
      {{0,2},0,1,{0.7,-0.4}},
      {{4,5,6},1,2,{0.9,0.2,-0.1,-0.3,0.6,0.5}},
      {{8,9,11},3,1,{0.15,-0.8,0.25}}};
  xmvb::cuda::OrbitalChartMap map(context,4,3,4,{3,3,4},blocks);
  Matrix basis=Matrix::Zero(12,4);
  for (const auto& block:blocks)
    for (int j=0;j<block.reduced_size;++j)
      for (int i=0;i<static_cast<int>(block.raw_slots.size());++i)
        basis(block.raw_slots[i],block.reduced_offset+j)=
            block.basis[j*block.raw_slots.size()+i];
  Matrix accepted(4,3),covector(4,3),step(4,1);
  for (int j=0;j<3;++j)
    for (int i=0;i<4;++i) {
      accepted(i,j)=0.2+0.13*j+0.07*i;
      covector(i,j)=std::sin(0.4+2*i+j);
    }
  step<<0.17,-0.29,0.31,0.14;
  const double scale=-0.43;
  auto daccepted=upload(context,accepted);
  auto dcovector=upload(context,covector);
  auto dstep=upload(context,step);
  DeviceMatrix raw(context,4,3),trial(context,4,3),reduced(context,4,1);
  map.expand(dstep.view(),raw.view());
  const Eigen::VectorXd expanded=basis*step;
  const Matrix expected_raw=Eigen::Map<const Matrix>(expanded.data(),4,3);
  near(download(context,raw.view()),expected_raw,"CUDA chart expansion differs");
  map.project(dcovector.view(),reduced.view());
  const Matrix expected_reduced=basis.transpose()*
      Eigen::Map<const Eigen::VectorXd>(covector.data(),12);
  near(download(context,reduced.view()),expected_reduced,
       "CUDA chart adjoint differs");
  map.retract(daccepted.view(),dstep.view(),scale,trial.view());
  Matrix expected_trial=accepted+scale*expected_raw;
  expected_trial(3,0)=0.0;
  expected_trial(3,1)=0.0;
  near(download(context,trial.view()),expected_trial,
       "CUDA chart retraction differs");
  const double left=(covector.array()*expected_raw.array()).sum();
  const double right=(step.array()*expected_reduced.array()).sum();
  if (std::abs(left-right)>2e-13)
    throw std::runtime_error("CUDA chart gradient/step pairing differs");
  // Fixed HAO support slots are untouched by the additive retraction.
  const Matrix actual=download(context,trial.view());
  if (actual(1,0)!=accepted(1,0) || actual(3,1)!=0.0 ||
      actual(2,2)!=accepted(2,2))
    throw std::runtime_error("CUDA chart changed a fixed sparse slot");
}

}  // namespace

int main() {
  try {
    check_map();
    std::cout<<"CUDA orbital chart map checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
  }
}
