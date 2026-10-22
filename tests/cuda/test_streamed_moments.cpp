#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "cuda/ci/sigma.hpp"
#include "mcscf/ci/integrals.hpp"
#include "mcscf/ci/sigma.hpp"
#include "mcscf/ci/strings.hpp"

namespace {
using Matrix=Eigen::MatrixXd;

void near(const Matrix& actual,const Matrix& expected,const char* what) {
  if (actual.rows()!=expected.rows() || actual.cols()!=expected.cols() ||
      !actual.allFinite() ||
      (actual-expected).norm()>3e-10*std::max(1.0,expected.norm()))
    throw std::runtime_error(what);
}

Matrix values(int rows,int cols,double phase) {
  Matrix result(rows,cols);
  for (int b=0;b<cols;++b)
    for (int a=0;a<rows;++a)
      result(a,b)=std::sin(phase+0.19*(a+1)*(b+2));
  return result;
}

Matrix download(xmvb::cuda::Context& context,const xmvb::cuda::Matrix<double>& device) {
  Matrix result(device.rows(),device.cols());
  xmvb::cuda::copy_to_host(context,result,device.view());
  return result;
}

void check_unequal() {
  const int n=4;
  const auto alpha=xmvb::mcscf::SpinStrings(n,1).occupations();
  const auto beta=xmvb::mcscf::SpinStrings(n,2).occupations();
  const int na=alpha.size(),nb=beta.size(),pairs=n*(n+1)/2;
  const xmvb::mcscf::ActiveIntegrals integrals{
      Matrix::Zero(n,n),Matrix::Zero(pairs,pairs)};
  const xmvb::mcscf::DirectCiSigmaAction cpu(alpha,beta,integrals);
  xmvb::cuda::Context context(32ULL*1024*1024);
  xmvb::cuda::DirectCiSigma gpu(context,cpu,16);
  const Matrix right_a=values(na,nb,0.31),right_b=values(na,nb,0.79);
  const Matrix generator=values(n,n,0.23); // Intentionally nonsymmetric.
  xmvb::cuda::Matrix<double> d_generator(context,n,n),d_action(context,na,2);
  xmvb::cuda::copy_to_device(context,d_generator.view(),generator);
  const auto reader=[&](const Matrix& source) -> xmvb::cuda::CiTileReader {
    return [&,carrier=&source](std::span<const int> betas,std::span<const int> rows,
                                xmvb::linalg::MatrixView output) {
      Matrix host(rows.size(),betas.size());
      for (int j=0;j<host.cols();++j)
        for (int i=0;i<host.rows();++i) host(i,j)=(*carrier)(rows[i],betas[j]);
      xmvb::cuda::copy_to_device(context,output,host);
    };
  };
  gpu.apply_one_body_generator_columns(1,2,reader(right_a),d_generator.view(),
      d_action.view());
  const Matrix full_action=cpu.apply_one_body_generator(right_a,generator);
  near(download(context,d_action),full_action.middleCols(1,2),
      "streamed generator action differs from CPU");
  const xmvb::mcscf::DirectCiColumnReader column=[&](int b) {
    return xmvb::mcscf::DirectCiColumnView{right_a.col(b).data(),nullptr,na};
  };
  Matrix alpha_only(na,2);
  cpu.apply_one_body_generator_columns(1,column,generator,alpha_only,true);
  gpu.apply_one_body_generator_columns(1,2,reader(right_a),d_generator.view(),
      d_action.view(),true);
  near(download(context,d_action),alpha_only,
      "streamed alpha-only generator action differs from CPU");

  const int first=1,count=2;
  Matrix left=values(na,count,0.47),other=values(na,count,1.03);
  xmvb::cuda::Matrix<double> d_left(context,na,count),d_other(context,na,count);
  xmvb::cuda::Matrix<double> d_out(context,n,n),d_out_other(context,n,n);
  xmvb::cuda::copy_to_device(context,d_left.view(),left);
  xmvb::cuda::copy_to_device(context,d_other.view(),other);
  const std::array<xmvb::linalg::ConstMatrixView,2> lefts{d_left.view(),d_other.view()};
  const std::array<xmvb::linalg::MatrixView,2> outputs{d_out.view(),d_out_other.view()};
  const std::array<xmvb::cuda::CiStreamedGeneratorRequest,2> requests{{
      {reader(right_a),-0.37,false},{reader(right_b),0.61,true}}};
  bool rejected=false;
  try { gpu.generator_adjoint_streamed(first,lefts,requests,outputs,false,true); }
  catch (const std::invalid_argument&) { rejected=true; }
  if (!rejected) throw std::runtime_error("uncertified exchange reduction was accepted");
  for (const bool symmetric:{false,true}) {
    gpu.generator_adjoint_streamed(first,lefts,requests,outputs,symmetric,false);
    std::array<Matrix,2> expected;
    for (int k=0;k<2;++k) {
      Matrix padded=Matrix::Zero(na,nb);
      padded.middleCols(first,count)=k ? other:left;
      expected[k]=-0.37*cpu.one_body_generator_adjoint(padded,right_a)+
          0.61*cpu.one_body_generator_adjoint(padded,right_b).transpose();
      if (symmetric) expected[k]=(expected[k]+expected[k].transpose()).eval();
      expected[k].triangularView<Eigen::StrictlyLower>().setZero();
      near(download(context,k ? d_out_other:d_out),expected[k],
          "streamed generator adjoint differs from CPU");
    }
    gpu.generator_adjoint_streamed(first,lefts,requests,outputs,symmetric,false,true);
    for (int k=0;k<2;++k)
      near(download(context,k ? d_out_other:d_out),2*expected[k],
          "streamed generator adjoint add mode differs from CPU");
  }
}

void check_exchange() {
  constexpr int n=4;
  const auto strings=xmvb::mcscf::SpinStrings(n,2).occupations();
  const int u=strings.size(),pairs=n*(n+1)/2;
  const xmvb::mcscf::ActiveIntegrals integrals{
      Matrix::Zero(n,n),Matrix::Zero(pairs,pairs)};
  const xmvb::mcscf::DirectCiSigmaAction cpu(strings,strings,integrals);
  xmvb::cuda::Context context(32ULL*1024*1024);
  xmvb::cuda::DirectCiSigma gpu(context,cpu,16,1);
  const auto parity_carrier=[&](double phase) {
    Matrix raw=values(u,u,phase);
    return Matrix((raw+raw.transpose())*0.5);
  };
  const Matrix left=parity_carrier(0.27),right=parity_carrier(0.63);
  const auto read=[&](std::span<const int> betas,std::span<const int> rows,
                      xmvb::linalg::MatrixView output) {
    Matrix tile(rows.size(),betas.size());
    for (int j=0;j<tile.cols();++j)
      for (int i=0;i<tile.rows();++i) tile(i,j)=right(rows[i],betas[j]);
    xmvb::cuda::copy_to_device(context,output,tile);
  };
  const std::array<xmvb::cuda::CiStreamedGeneratorRequest,1> requests{{
      {read,0.73,true}}};
  xmvb::cuda::Matrix<double> d_left(context,u,2),d_out(context,n,n);
  xmvb::cuda::check_cuda(cudaMemsetAsync(d_out.data(),0,n*n*sizeof(double),
      context.stream()),"clear exchange generator output");
  for (int first=0;first<u;first+=2) {
    const int count=std::min(2,u-first);
    xmvb::cuda::copy_to_device(context,d_left.view().columns(0,count),
        left.middleCols(first,count).eval());
    const std::array<xmvb::linalg::ConstMatrixView,1> lefts{
        d_left.view().columns(0,count)};
    const std::array<xmvb::linalg::MatrixView,1> outputs{d_out.view()};
    gpu.generator_adjoint_streamed(first,lefts,requests,outputs,true,true,true);
  }
  Matrix expected=0.73*cpu.one_body_generator_adjoint(left,right).transpose();
  expected=(expected+expected.transpose()).eval();
  expected.triangularView<Eigen::StrictlyLower>().setZero();
  near(download(context,d_out),expected,
      "exchange-reduced streamed generator adjoint differs from CPU");
}
} // namespace

int main() {
  try {
    check_unequal();check_exchange();
    std::cout<<"streamed CUDA generator moments passed\n";
  }
  catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';return 1;
  }
}
