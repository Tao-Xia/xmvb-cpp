#include "cuda/optimization/coarse.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/LU>
#include <Eigen/SVD>

#include "cuda/linalg/blas.hpp"

namespace xmvb::cuda {
namespace {

__global__ void finite_vectors(linalg::ConstMatrixView first,
    linalg::ConstMatrixView second,linalg::ConstMatrixView third,
    int* valid,std::int64_t dimension) {
  const auto i=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i>=dimension) return;
  if (!isfinite(first.data[i]) || (second.data && !isfinite(second.data[i])) ||
      (third.data && !isfinite(third.data[i]))) atomicExch(valid,0);
}

__global__ void absolute_product(const double* left,const double* right,
    double* product,std::int64_t dimension) {
  const auto i=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i<dimension) product[i]=fabs(left[i]*right[i]);
}

unsigned int grid(std::int64_t count) {
  constexpr std::int64_t threads=256;
  const auto blocks=(count+threads-1)/threads;
  if (blocks<=0 || blocks>std::numeric_limits<unsigned int>::max())
    throw std::overflow_error("CUDA coarse grid overflow");
  return static_cast<unsigned int>(blocks);
}

}  // namespace

DeviceCoarsePreconditioner::DeviceCoarsePreconditioner(Context& context,
    int dimension,int capacity,double lindep,double action_accuracy)
    : context_(&context),dimension_(dimension),
      capacity_(std::min(dimension,capacity)),lindep_(lindep),
      action_accuracy_(action_accuracy),
      basis_(context,dimension,std::max(0,capacity_)),
      images_(context,dimension,std::max(0,capacity_)),
      metrics_(context,dimension,std::max(0,capacity_)),
      sample_v_(context,dimension,1),sample_h_(context,dimension,1),
      sample_m_(context,dimension,1),finite_(context,1),
      hessian_projection_(Eigen::MatrixXd::Zero(std::max(0,capacity_),
                                                  std::max(0,capacity_))),
      metric_projection_(Eigen::MatrixXd::Zero(std::max(0,capacity_),
                                               std::max(0,capacity_))),
      sample_factor_(Eigen::MatrixXd::Zero(std::max(0,capacity_),
                                           std::max(0,capacity_))) {
  if (dimension<=0 || capacity<0 || !(lindep>0.0) ||
      !std::isfinite(action_accuracy) || action_accuracy<0.0)
    throw std::invalid_argument("invalid CUDA coarse preconditioner dimensions");
  check_cublas(cublasSetPointerMode(context.blas(),CUBLAS_POINTER_MODE_HOST),
      "CUDA coarse host scalar pointer mode");
}

void DeviceCoarsePreconditioner::validate_vector(
    linalg::ConstMatrixView vector) const {
  if (vector.rows!=dimension_ || vector.cols!=1 || vector.ld<vector.rows ||
      !vector.data)
    throw std::invalid_argument("invalid CUDA coarse vector dimensions");
}

void DeviceCoarsePreconditioner::ensure_finite(linalg::ConstMatrixView first,
    linalg::ConstMatrixView second,linalg::ConstMatrixView third) const {
  validate_vector(first);
  if (second.data) validate_vector(second);
  if (third.data) validate_vector(third);
  const int valid=1;
  context_->copy_to_device(finite_.data(),&valid,sizeof(valid));
  finite_vectors<<<grid(dimension_),256,0,context_->stream()>>>(
      first,second,third,finite_.data(),dimension_);
  check_cuda(cudaGetLastError(),"CUDA coarse finite validation");
  int host=0;
  context_->copy_to_host(&host,finite_.data(),sizeof(host));
  if (!host) throw std::invalid_argument("nonfinite CUDA coarse sample");
}

double DeviceCoarsePreconditioner::dot(linalg::ConstMatrixView left,
    linalg::ConstMatrixView right) const {
  double value=0.0;
  check_cublas(cublasDdot(context_->blas(),dimension_,left.data,1,right.data,1,
      &value),"CUDA coarse dot");
  return value;
}

double DeviceCoarsePreconditioner::norm(linalg::ConstMatrixView vector) const {
  double value=0.0;
  check_cublas(cublasDnrm2(context_->blas(),dimension_,vector.data,1,&value),
      "CUDA coarse norm");
  return value;
}

double DeviceCoarsePreconditioner::frobenius(linalg::ConstMatrixView panel,
    int columns) const {
  double value=0.0;
  for (int column=0;column<columns;++column)
    value=std::hypot(value,norm(panel.column(column)));
  return value;
}

void DeviceCoarsePreconditioner::copy(linalg::MatrixView out,
    linalg::ConstMatrixView in) const {
  check_cuda(cudaMemcpyAsync(out.data,in.data,dimension_*sizeof(double),
      cudaMemcpyDeviceToDevice,context_->stream()),"CUDA coarse vector copy");
}

void DeviceCoarsePreconditioner::axpy(linalg::MatrixView out,double coefficient,
    linalg::ConstMatrixView in) const {
  check_cublas(cublasDaxpy(context_->blas(),dimension_,&coefficient,
      in.data,1,out.data,1),"CUDA coarse axpy");
}

void DeviceCoarsePreconditioner::scale(linalg::MatrixView out,
    double coefficient) const {
  check_cublas(cublasDscal(context_->blas(),dimension_,&coefficient,out.data,1),
      "CUDA coarse scale");
}

double DeviceCoarsePreconditioner::projection_roundoff(
    linalg::ConstMatrixView basis,linalg::ConstMatrixView action,
    const Eigen::MatrixXd& projected) const {
  const int rank=static_cast<int>(projected.rows());
  Eigen::MatrixXd magnitude(rank,rank);
  for (int j=0;j<rank;++j) for (int i=0;i<rank;++i) {
    absolute_product<<<grid(dimension_),256,0,context_->stream()>>>(
        basis.column(i).data,action.column(j).data,sample_m_.data(),dimension_);
    check_cuda(cudaGetLastError(),"CUDA coarse projection magnitude");
    double value=0.0;
    check_cublas(cublasDasum(context_->blas(),dimension_,sample_m_.data(),1,
        &value),"CUDA coarse projection absolute sum");
    magnitude(i,j)=value;
  }
  const double u=0.5*std::numeric_limits<double>::epsilon();
  const double nu=static_cast<double>(dimension_)*u;
  const double gamma=nu/(1.0-nu);
  const Eigen::MatrixXd error=
      (gamma/(1.0-gamma))*(magnitude+magnitude.transpose())+
      u*(projected.cwiseAbs()+projected.transpose().cwiseAbs());
  return error.stableNorm();
}

void DeviceCoarsePreconditioner::observe(linalg::ConstMatrixView direction,
    linalg::ConstMatrixView image,linalg::ConstMatrixView metric_image) {
  ensure_finite(direction,image,metric_image);
  if (size_==capacity_) return;
  check_cublas(cublasSetPointerMode(context_->blas(),CUBLAS_POINTER_MODE_HOST),
      "CUDA coarse observe scalar pointer mode");
  copy(sample_v_.view(),direction);
  copy(sample_h_.view(),image);
  copy(sample_m_.view(),metric_image);
  const double initial=norm(sample_v_.view());
  if (!(initial>0.0)) return;
  Eigen::VectorXd coefficients=Eigen::VectorXd::Zero(size_);
  for (int pass=0;pass<2;++pass) for (int i=0;i<size_;++i) {
    const double coefficient=dot(basis_.view().column(i),sample_v_.view());
    coefficients[i]+=coefficient;
    axpy(sample_v_.view(),-coefficient,basis_.view().column(i));
    axpy(sample_h_.view(),-coefficient,images_.view().column(i));
    axpy(sample_m_.view(),-coefficient,metrics_.view().column(i));
  }
  const double remaining=norm(sample_v_.view());
  Eigen::MatrixXd factor=Eigen::MatrixXd::Zero(size_+1,size_+1);
  factor.topLeftCorner(size_,size_)=sample_factor_.topLeftCorner(size_,size_);
  factor.col(size_).head(size_)=coefficients/initial;
  factor(size_,size_)=remaining/initial;
  const Eigen::JacobiSVD<Eigen::MatrixXd> spectrum(factor);
  const double ratio=std::pow(spectrum.singularValues().tail(1)[0]/
      spectrum.singularValues()[0],2);
  if (!(ratio>lindep_)) return;
  const int candidate=size_;
  copy(basis_.view().column(candidate),sample_v_.view());
  copy(images_.view().column(candidate),sample_h_.view());
  copy(metrics_.view().column(candidate),sample_m_.view());
  const double inverse=1.0/remaining;
  scale(basis_.view().column(candidate),inverse);
  scale(images_.view().column(candidate),inverse);
  scale(metrics_.view().column(candidate),inverse);
  for (int i=0;i<candidate;++i) {
    hessian_projection_(i,candidate)=dot(basis_.view().column(i),
        images_.view().column(candidate));
    hessian_projection_(candidate,i)=dot(images_.view().column(i),
        basis_.view().column(candidate));
    metric_projection_(i,candidate)=dot(basis_.view().column(i),
        metrics_.view().column(candidate));
    metric_projection_(candidate,i)=dot(metrics_.view().column(i),
        basis_.view().column(candidate));
  }
  hessian_projection_(candidate,candidate)=dot(
      basis_.view().column(candidate),images_.view().column(candidate));
  metric_projection_(candidate,candidate)=dot(
      basis_.view().column(candidate),metrics_.view().column(candidate));
  const auto certified=[&](const Eigen::MatrixXd& projection,
      linalg::ConstMatrixView images,double action_accuracy) {
    const Eigen::MatrixXd raw=projection.topLeftCorner(candidate+1,candidate+1);
    const double skew=(raw-raw.transpose()).norm();
    double bound=256*std::numeric_limits<double>::epsilon()*(candidate+1)*
        std::max(1.0,raw.norm());
    if (action_accuracy>0.0)
      bound+=action_accuracy*frobenius(basis_.view(),candidate+1)*
          frobenius(images,candidate+1);
    if (skew>bound)
      bound+=projection_roundoff(basis_.view(),images,raw);
    return std::isfinite(skew) && std::isfinite(bound) && skew<=bound;
  };
  if (!certified(hessian_projection_,images_.view(),action_accuracy_) ||
      !certified(metric_projection_,metrics_.view(),0.0)) return;
  for (int i=0;i<candidate;++i) {
    const double hessian=0.5*(hessian_projection_(i,candidate)+
        hessian_projection_(candidate,i));
    const double metric=0.5*(metric_projection_(i,candidate)+
        metric_projection_(candidate,i));
    hessian_projection_(i,candidate)=hessian_projection_(candidate,i)=hessian;
    metric_projection_(i,candidate)=metric_projection_(candidate,i)=metric;
  }
  sample_factor_.topLeftCorner(size_+1,size_+1)=factor;
  ++size_;
}

void DeviceCoarsePreconditioner::apply(linalg::ConstMatrixView residual,
    double eigenvalue,const BaseInverse& inverse,linalg::MatrixView output) const {
  validate_vector(residual);
  validate_vector(output);
  if (!std::isfinite(eigenvalue) || !inverse)
    throw std::invalid_argument("invalid CUDA coarse inverse request");
  ensure_finite(residual);
  check_cublas(cublasSetPointerMode(context_->blas(),CUBLAS_POINTER_MODE_HOST),
      "CUDA coarse apply scalar pointer mode");
  if (!size_) {
    copy(sample_v_.view(),residual);
    inverse(sample_v_.view(),sample_h_.view());
    check_cublas(cublasSetPointerMode(context_->blas(),CUBLAS_POINTER_MODE_HOST),
        "CUDA coarse restore scalar pointer mode");
    ensure_finite(sample_h_.view());
    copy(output,sample_h_.view());
    return;
  }
  const Eigen::MatrixXd e=hessian_projection_.topLeftCorner(size_,size_)-
      eigenvalue*metric_projection_.topLeftCorner(size_,size_);
  Eigen::FullPivLU<Eigen::MatrixXd> factor(e);
  if (!factor.isInvertible())
    throw std::runtime_error("singular exact CUDA coarse shifted operator");
  const auto solve=[&](const Eigen::VectorXd& rhs) {
    const Eigen::VectorXd solution=factor.solve(rhs);
    const double bound=256*std::numeric_limits<double>::epsilon()*size_*
        std::max(1.0,e.norm()*solution.norm()+rhs.norm());
    if (!solution.allFinite() || (e*solution-rhs).norm()>bound)
      throw std::runtime_error("CUDA coarse inverse failed residual certificate");
    return solution;
  };
  Eigen::VectorXd projected(size_);
  for (int i=0;i<size_;++i)
    projected[i]=dot(basis_.view().column(i),residual);
  const Eigen::VectorXd a=solve(projected);
  copy(sample_v_.view(),residual);
  for (int i=0;i<size_;++i) {
    axpy(sample_v_.view(),-a[i],images_.view().column(i));
    axpy(sample_v_.view(),eigenvalue*a[i],metrics_.view().column(i));
  }
  inverse(sample_v_.view(),sample_h_.view());
  check_cublas(cublasSetPointerMode(context_->blas(),CUBLAS_POINTER_MODE_HOST),
      "CUDA coarse restore scalar pointer mode");
  ensure_finite(sample_h_.view());
  for (int i=0;i<size_;++i)
    projected[i]=dot(images_.view().column(i),sample_h_.view())-
        eigenvalue*dot(metrics_.view().column(i),sample_h_.view());
  const Eigen::VectorXd b=solve(projected);
  copy(output,sample_h_.view());
  for (int i=0;i<size_;++i)
    axpy(output,a[i]-b[i],basis_.view().column(i));
}

}  // namespace xmvb::cuda
