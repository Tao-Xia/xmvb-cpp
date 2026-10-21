#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Eigen/Core>

#include "cuda/linalg/blas.hpp"
#include "cuda/vb/tangent.hpp"

namespace {

using Matrix=Eigen::MatrixXd;
using Vector=Eigen::VectorXd;

void require(bool condition,const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void near(const Vector& actual,const Vector& expected,const char* message) {
  require(actual.size()==expected.size() && actual.allFinite() &&
      (actual-expected).norm()<2.0e-11*std::max(1.0,expected.norm()),message);
}

Vector download(xmvb::cuda::Context& context,
                const xmvb::cuda::Matrix<double>& values) {
  Matrix host(values.rows(),1);
  xmvb::cuda::copy_to_host(context,host,values.view());
  return host.col(0);
}

void check_single_state_selected_space() {
  constexpr int n=5;
  Matrix h=Matrix::Zero(n,n),s=Matrix::Zero(n,n);
  for (int i=0;i<n;++i) {
    h(i,i)=0.3+0.8*i;
    s(i,i)=1.0+0.25*i;
    if (i) {
      h(i,i-1)=h(i-1,i)=0.11;
      s(i,i-1)=s(i-1,i)=0.04;
    }
  }
  Vector selected(n);
  selected<<0.42,-0.31,0.18,0.51,-0.23;
  selected/=std::sqrt(selected.dot(s*selected));
  const Vector constraint=s*selected;
  const auto horizontal=[&](const Vector& x) {
    return Vector(x-constraint*(constraint.dot(x)/constraint.squaredNorm()));
  };
  constexpr double energy=0.82;
  constexpr double weight=1.0;
  const double scale=std::sqrt(2.0*weight);
  xmvb::cuda::Context context(8ULL*1024*1024);
  xmvb::cuda::Matrix<double> dh(context,n,n),ds(context,n,n),dc(context,n,1);
  xmvb::cuda::copy_to_device(context,dh.view(),h);
  xmvb::cuda::copy_to_device(context,ds.view(),s);
  xmvb::cuda::copy_to_device(context,dc.view(),selected);
  xmvb::cuda::SingleStateCiActions actions;
  actions.hamiltonian_metric=[&](xmvb::linalg::ConstMatrixView x,
      xmvb::linalg::MatrixView hx,xmvb::linalg::MatrixView sx) {
    xmvb::cuda::gemm(context,hx,dh.view(),x);
    xmvb::cuda::gemm(context,sx,ds.view(),x);
  };
  actions.metric=[&](xmvb::linalg::ConstMatrixView x,
                     xmvb::linalg::MatrixView sx) {
    xmvb::cuda::gemm(context,sx,ds.view(),x);
  };
  Matrix inverse_model=Matrix::Identity(n,n);
  for (int i=0;i<n;++i) inverse_model(i,i)=0.7+0.2*i;
  inverse_model(0,1)=inverse_model(1,0)=0.08;
  inverse_model(2,3)=inverse_model(3,2)=-0.06;
  xmvb::cuda::Matrix<double> dinverse(context,n,n);
  xmvb::cuda::copy_to_device(context,dinverse.view(),inverse_model);
  actions.augmented_inverse=[&](xmvb::linalg::ConstMatrixView x,double,
                                 xmvb::linalg::MatrixView result) {
    xmvb::cuda::gemm(context,result,dinverse.view(),x);
  };
  xmvb::cuda::SingleStateCiTangent tangent(
      context,n,dc.view(),energy,weight,actions);
  require(tangent.tangent_size()==n-1,"CI tangent dimension changed");

  Vector arbitrary(n);
  arbitrary<<0.2,-0.7,0.5,0.11,-0.3;
  xmvb::cuda::Matrix<double> dx(context,n,1),dz(context,n-1,1),du(context,n,1);
  xmvb::cuda::copy_to_device(context,dx.view(),arbitrary);
  tangent.project_coordinates(dx.view(),dz.view());
  tangent.expand_coordinates(dz.view(),du.view());
  near(download(context,du),horizontal(arbitrary),
       "CI tangent ambient projection differs");

  Vector coordinates(n-1);
  coordinates<<0.17,-0.23,0.45,0.09;
  xmvb::cuda::copy_to_device(context,dz.view(),coordinates);
  tangent.expand_coordinates(dz.view(),du.view());
  const Vector ambient=download(context,du);
  require(std::abs(constraint.dot(ambient))<2.0e-12,
          "CI tangent direction is not horizontal");
  require(std::abs(ambient.norm()-coordinates.norm())<2.0e-12,
          "CI tangent coordinates changed Euclidean length");
  xmvb::cuda::Matrix<double> round_trip(context,n-1,1);
  tangent.project_coordinates(du.view(),round_trip.view());
  near(download(context,round_trip),coordinates,
       "CI tangent coordinate round trip differs");

  xmvb::cuda::Matrix<double> metric(context,n-1,1),hessian(context,n-1,1);
  xmvb::cuda::Matrix<double> projected_metric(context,n,1);
  xmvb::cuda::Matrix<double> projected_hessian(context,n,1);
  tangent.apply_metric_coordinates(dz.view(),metric.view());
  tangent.expand_coordinates(metric.view(),projected_metric.view());
  near(download(context,projected_metric),horizontal(s*ambient),
       "CI tangent metric is not the projected selected overlap");
  tangent.apply_images_coordinates(dz.view(),hessian.view(),metric.view());
  tangent.expand_coordinates(hessian.view(),projected_hessian.view());
  tangent.expand_coordinates(metric.view(),projected_metric.view());
  near(download(context,projected_hessian),horizontal((h-energy*s)*ambient),
       "CI tangent Hessian is not the projected shifted action");
  near(download(context,projected_metric),horizontal(s*ambient),
       "CI tangent joint metric differs from standalone metric");

  xmvb::cuda::Matrix<double> origin(context,n-1,1),ambient_origin(context,n,1);
  tangent.origin_gradient(origin.view());
  tangent.expand_coordinates(origin.view(),ambient_origin.view());
  near(download(context,ambient_origin),
       horizontal(scale*(h-energy*s)*selected),
       "CI tangent origin gradient differs");
  xmvb::cuda::Matrix<double> response(context,n,1);
  tangent.coefficient_response_coordinates(dz.view(),response.view());
  near(download(context,response),ambient/scale,
       "CI tangent coefficient response differs");
  const double maximum=tangent.maximum_coefficient_component_coordinates(dz.view());
  require(std::abs(maximum-(ambient.cwiseAbs().maxCoeff()/scale))<2.0e-12,
          "CI tangent maximum coefficient differs");

  constexpr double eigenvalue=-0.037;
  xmvb::cuda::Matrix<double> inverse_coordinates(context,n-1,1);
  tangent.apply_inverse_augmented_hessian_diagonal(
      dz.view(),eigenvalue,inverse_coordinates.view());
  xmvb::cuda::Matrix<double> inverse_ambient(context,n,1);
  tangent.expand_coordinates(inverse_coordinates.view(),inverse_ambient.view());
  const Vector right=ambient-constraint*selected.dot(ambient);
  const Vector solved=inverse_model*right;
  const Vector inverse_constraint=inverse_model*constraint;
  const Vector expected=horizontal(solved-inverse_constraint*
      (constraint.dot(solved)/constraint.dot(inverse_constraint)));
  near(download(context,inverse_ambient),expected,
       "CI tangent constrained AH inverse differs from CPU Schur oracle");

  Matrix carrier=Matrix::Identity(n,n);
  carrier.diagonal()<<1.3,0.8,1.7,0.9,1.2;
  xmvb::cuda::Matrix<double> dcarrier(context,n,n);
  xmvb::cuda::copy_to_device(context,dcarrier.view(),carrier);
  actions.maximum_component=[&](xmvb::linalg::ConstMatrixView x) {
    xmvb::cuda::Matrix<double> orthogonal(context,n,1);
    xmvb::cuda::gemm(context,orthogonal.view(),dcarrier.view(),x);
    int index=0;
    xmvb::cuda::check_cublas(cublasIdamax(context.blas(),n,
        orthogonal.data(),1,&index),"CI tangent carrier maximum index");
    double value=0.0;
    context.copy_to_host(&value,orthogonal.data()+index-1,sizeof(double));
    return std::abs(value);
  };
  xmvb::cuda::SingleStateCiTangent clipped(
      context,n,dc.view(),energy,weight,actions);
  const double carrier_maximum=
      clipped.maximum_coefficient_component_coordinates(dz.view());
  require(std::abs(carrier_maximum-
      (carrier*(ambient/scale)).cwiseAbs().maxCoeff())<2.0e-12,
      "CI tangent clipping did not use the orthogonal carrier callback");
}

}  // namespace

int main() {
  try {
    check_single_state_selected_space();
    std::cout<<"CUDA single-state selected CI tangent: PASS\n";
  } catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
  }
}
