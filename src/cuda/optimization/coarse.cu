#include "cuda/optimization/coarse.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <cub/block/block_reduce.cuh>

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

__global__ void projection_magnitudes(linalg::ConstMatrixView basis,
    linalg::ConstMatrixView action,double* result,int rank) {
  using Reduce=cub::BlockReduce<double,256>;
  __shared__ typename Reduce::TempStorage scratch;
  const int row=blockIdx.x%rank,column=blockIdx.x/rank;
  double sum=0.0;
  for (std::int64_t i=threadIdx.x;i<basis.rows;i+=blockDim.x)
    sum+=fabs(basis.data[row*basis.ld+i]*action.data[column*action.ld+i]);
  const double total=Reduce(scratch).Sum(sum);
  if (!threadIdx.x) result[blockIdx.x]=total;
}

// The following AXPYs consume this coefficient on the same stream. MGS is
// still sequential, including its second pass; no host scalar is needed.
__global__ void accumulate_coefficient(double* coefficient,double* total) {
  *total+=*coefficient;
  *coefficient=-*coefficient;
}

class DeviceScalars {
 public:
  explicit DeviceScalars(cublasHandle_t handle):handle_(handle) {
    check_cublas(cublasSetPointerMode(handle_,CUBLAS_POINTER_MODE_DEVICE),
        "CUDA coarse device scalar pointer mode");
  }
  ~DeviceScalars() { cublasSetPointerMode(handle_,CUBLAS_POINTER_MODE_HOST); }
  DeviceScalars(const DeviceScalars&)=delete;
  DeviceScalars& operator=(const DeviceScalars&)=delete;
 private:
  cublasHandle_t handle_;
};

std::size_t reduction_size(int capacity) {
  const auto n=static_cast<std::size_t>(std::max(0,capacity));
  return std::max({n*n,4*n+4,n+3});
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
      sample_m_(context,dimension,1),
      coefficients_(context,std::max(1,capacity_),1),
      reductions_(context,reduction_size(capacity_)),finite_(context,1),
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
  check_cuda(cudaMemsetAsync(finite_.data(),1,sizeof(int),context_->stream()),
      "CUDA coarse finite flag initialization");
  finite_vectors<<<grid(dimension_),256,0,context_->stream()>>>(
      first,second,third,finite_.data(),dimension_);
  check_cuda(cudaGetLastError(),"CUDA coarse finite validation");
  int host=0;
  context_->copy_to_host(&host,finite_.data(),sizeof(host));
  if (!host) throw std::invalid_argument("nonfinite CUDA coarse sample");
}

Eigen::VectorXd DeviceCoarsePreconditioner::project(linalg::ConstMatrixView panel,
    linalg::ConstMatrixView vector,int columns) const {
  Eigen::VectorXd result(columns);
  if (!columns) return result;
  const linalg::MatrixView output{coefficients_.data(),columns,1,columns};
  gemm(*context_,output,panel.columns(0,columns),vector,1.0,0.0,Operation::transpose);
  context_->copy_to_host(result.data(),output.data,columns*sizeof(double));
  return result;
}

void DeviceCoarsePreconditioner::combine(linalg::MatrixView output,
    linalg::ConstMatrixView panel,const Eigen::VectorXd& coefficients,double scale) const {
  const int columns=static_cast<int>(coefficients.size());
  if (!columns) return;
  context_->copy_to_device(coefficients_.data(),coefficients.data(),columns*sizeof(double));
  gemm(*context_,output,panel.columns(0,columns),
      {coefficients_.data(),columns,1,columns},scale,1.0);
}

void DeviceCoarsePreconditioner::copy(linalg::MatrixView out,
    linalg::ConstMatrixView in) const {
  check_cuda(cudaMemcpyAsync(out.data,in.data,dimension_*sizeof(double),
      cudaMemcpyDeviceToDevice,context_->stream()),"CUDA coarse vector copy");
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
  projection_magnitudes<<<rank*rank,256,0,context_->stream()>>>(
      basis,action,reductions_.data(),rank);
  check_cuda(cudaGetLastError(),"CUDA coarse projection magnitudes");
  context_->copy_to_host(magnitude.data(),reductions_.data(),
      magnitude.size()*sizeof(double));
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
  double* const scalars=reductions_.data();
  check_cuda(cudaMemsetAsync(scalars,0,size_*sizeof(double),context_->stream()),
      "CUDA coarse coefficient initialization");
  {
    DeviceScalars mode(context_->blas());
    check_cublas(cublasDnrm2(context_->blas(),dimension_,sample_v_.data(),1,
        scalars+size_),"CUDA coarse initial norm");
    for (int pass=0;pass<2;++pass) for (int i=0;i<size_;++i) {
      double* const coefficient=scalars+size_+2;
      check_cublas(cublasDdot(context_->blas(),dimension_,
          basis_.view().column(i).data,1,sample_v_.data(),1,coefficient),
          "CUDA coarse device dot");
      accumulate_coefficient<<<1,1,0,context_->stream()>>>(coefficient,scalars+i);
      check_cuda(cudaGetLastError(),"CUDA coarse coefficient accumulation");
      for (auto panels:{std::pair{sample_v_.data(),basis_.view().column(i).data},
                        std::pair{sample_h_.data(),images_.view().column(i).data},
                        std::pair{sample_m_.data(),metrics_.view().column(i).data}})
        check_cublas(cublasDaxpy(context_->blas(),dimension_,coefficient,
            panels.second,1,panels.first,1),"CUDA coarse device axpy");
    }
    check_cublas(cublasDnrm2(context_->blas(),dimension_,sample_v_.data(),1,
        scalars+size_+1),"CUDA coarse remaining norm");
  }
  Eigen::VectorXd samples(size_+2);
  context_->copy_to_host(samples.data(),scalars,samples.size()*sizeof(double));
  const double initial=samples[size_],remaining=samples[size_+1];
  if (!(initial>0.0)) return;
  Eigen::MatrixXd factor=Eigen::MatrixXd::Zero(size_+1,size_+1);
  factor.topLeftCorner(size_,size_)=sample_factor_.topLeftCorner(size_,size_);
  factor.col(size_).head(size_)=samples.head(size_)/initial;
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
  int offset=0;
  const auto enqueue=[&](linalg::ConstMatrixView panel,
      linalg::ConstMatrixView vector,int columns) {
    if (columns) gemm(*context_,{scalars+offset,columns,1,columns},
        panel.columns(0,columns),vector,1.0,0.0,Operation::transpose);
    offset+=columns;
  };
  enqueue(basis_.view(),images_.view().column(candidate),candidate+1);
  enqueue(images_.view(),basis_.view().column(candidate),candidate);
  enqueue(basis_.view(),metrics_.view().column(candidate),candidate+1);
  enqueue(metrics_.view(),basis_.view().column(candidate),candidate);
  if (action_accuracy_>0.0) {
    DeviceScalars mode(context_->blas());
    const int count=blas_int(static_cast<std::int64_t>(dimension_)*(candidate+1));
    check_cublas(cublasDnrm2(context_->blas(),count,basis_.data(),1,scalars+offset),
        "CUDA coarse basis norm");
    check_cublas(cublasDnrm2(context_->blas(),count,images_.data(),1,scalars+offset+1),
        "CUDA coarse image norm");
  }
  Eigen::VectorXd projections(offset+(action_accuracy_>0.0 ? 2:0));
  context_->copy_to_host(projections.data(),scalars,projections.size()*sizeof(double));
  hessian_projection_.col(candidate).head(candidate+1)=projections.head(candidate+1);
  hessian_projection_.row(candidate).head(candidate)=
      projections.segment(candidate+1,candidate).transpose();
  metric_projection_.col(candidate).head(candidate+1)=
      projections.segment(2*candidate+1,candidate+1);
  metric_projection_.row(candidate).head(candidate)=
      projections.segment(3*candidate+2,candidate).transpose();
  const auto certified=[&](const Eigen::MatrixXd& projection,
      linalg::ConstMatrixView images,double action_accuracy) {
    const Eigen::MatrixXd raw=projection.topLeftCorner(candidate+1,candidate+1);
    const double skew=(raw-raw.transpose()).norm();
    double bound=256*std::numeric_limits<double>::epsilon()*(candidate+1)*
        std::max(1.0,raw.norm());
    if (action_accuracy>0.0)
      bound+=action_accuracy*projections[offset]*projections[offset+1];
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
  Eigen::VectorXd projected=project(basis_.view(),residual,size_);
  const Eigen::VectorXd a=solve(projected);
  copy(sample_v_.view(),residual);
  combine(sample_v_.view(),images_.view(),a,-1.0);
  combine(sample_v_.view(),metrics_.view(),a,eigenvalue);
  inverse(sample_v_.view(),sample_h_.view());
  check_cublas(cublasSetPointerMode(context_->blas(),CUBLAS_POINTER_MODE_HOST),
      "CUDA coarse restore scalar pointer mode");
  ensure_finite(sample_h_.view());
  const linalg::MatrixView projection{reductions_.data(),size_,1,size_};
  gemm(*context_,projection,images_.view().columns(0,size_),sample_h_.view(),
      1.0,0.0,Operation::transpose);
  gemm(*context_,projection,metrics_.view().columns(0,size_),sample_h_.view(),
      -eigenvalue,1.0,Operation::transpose);
  context_->copy_to_host(projected.data(),projection.data,size_*sizeof(double));
  const Eigen::VectorXd b=solve(projected);
  copy(output,sample_h_.view());
  combine(output,basis_.view(),a-b);
}

}  // namespace xmvb::cuda
