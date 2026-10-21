#include "cuda/vb/orbitals.hpp"

#include <cfloat>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cusolverDn.h>

#include "cuda/linalg/blas.hpp"
#include "cuda/linalg/cholesky.hpp"

namespace xmvb::cuda {
namespace {

constexpr int kThreads = 128;
int grid(int count) { return (count + kThreads - 1) / kThreads; }

void check_solver(cusolverStatus_t status, const char* label) {
  if (status != CUSOLVER_STATUS_SUCCESS)
    throw std::runtime_error(std::string(label) + " failed (cuSOLVER status " +
                             std::to_string(static_cast<int>(status)) + ")");
}

using View = linalg::ConstMatrixView;
using Mutable = linalg::MatrixView;

void check_view(View view, int rows, int cols, const char* label,
                bool optional = false) {
  if (optional && !view.data && view.rows == 0 && view.cols == 0) return;
  if (!view.data || view.rows != rows || view.cols != cols || view.ld < rows)
    throw std::invalid_argument(std::string(label) + " device matrix shape mismatch");
}

bool supplied(View view) { return view.data != nullptr; }

void zero(Context& context, Mutable output) {
  for (int column=0;column<output.cols;++column)
    check_cuda(cudaMemsetAsync(output.column(column).data,0,
        static_cast<std::size_t>(output.rows)*sizeof(double),
        context.stream()),"CUDA orbital zero");
}

void copy(Context& context, Mutable output, View input) {
  if (output.rows != input.rows || output.cols != input.cols)
    throw std::invalid_argument("CUDA orbital copy shape mismatch");
  for (int column = 0; column < input.cols; ++column)
    check_cuda(cudaMemcpyAsync(output.column(column).data,
        input.column(column).data, input.rows * sizeof(double),
        cudaMemcpyDeviceToDevice, context.stream()), "CUDA orbital device copy");
}

__global__ void identity_kernel(double* output, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n*n) output[index] = index/n == index%n ? 1.0 : 0.0;
}

void identity(Context& context, Mutable output) {
  if (output.rows != output.cols || output.ld != output.rows)
    throw std::invalid_argument("CUDA orbital identity shape mismatch");
  const int n = static_cast<int>(output.rows);
  if (n) {
    identity_kernel<<<grid(n*n),kThreads,0,context.stream()>>>(output.data,n);
    check_cuda(cudaGetLastError(), "CUDA orbital identity kernel");
  }
}

void axpy(Context& context, Mutable y, View x, double alpha) {
  if (y.rows != x.rows || y.cols != x.cols ||
      y.ld != y.rows || x.ld != x.rows)
    throw std::invalid_argument("CUDA orbital axpy shape mismatch");
  const int size = static_cast<int>(y.rows*y.cols);
  if (size) check_cublas(cublasDaxpy(context.blas(),size,&alpha,x.data,1,
                                     y.data,1),"cublasDaxpy orbital accumulation");
}

__global__ void sym_kernel(double* target, const double* source,
                           int n, double factor) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n*n) {
    const int row = index%n, col = index/n;
    target[index] += factor*(source[index]+source[col+row*n]);
  }
}

void add_sym(Context& context, Mutable target, View source, double factor = 1.0) {
  const int n = static_cast<int>(target.rows);
  if (n && source.data) {
    sym_kernel<<<grid(n*n),kThreads,0,context.stream()>>>(
        target.data,source.data,n,factor);
    check_cuda(cudaGetLastError(), "CUDA orbital symmetrization kernel");
  }
}

Matrix<double> product(Context& context, View left, View right,
                       Operation op_left = Operation::none,
                       Operation op_right = Operation::none) {
  const int rows = static_cast<int>(op_left == Operation::none ? left.rows : left.cols);
  const int cols = static_cast<int>(op_right == Operation::none ? right.cols : right.rows);
  Matrix<double> result(context,rows,cols);
  if (rows && cols) gemm(context,result.view(),left,right,
                         1.0,0.0,op_left,op_right);
  return result;
}

Matrix<double> duplicate(Context& context, View source) {
  Matrix<double> result(context,source.rows,source.cols);
  copy(context,result.view(),source);
  return result;
}

Matrix<double> vacant(Context& context, int rows, int cols) {
  Matrix<double> result(context,rows,cols);
  zero(context,result.view());
  return result;
}

__global__ void normalize_kernel(const double* raw, const int* support,
    const int* counts, const double* metric, int metric_ld,int basis, int orbitals,
    double* normalized, double* inverse_norms, int* info) {
  const int orbital = blockIdx.x*blockDim.x+threadIdx.x;
  if (orbital >= orbitals) return;
  const int count = counts[orbital];
  double norm2 = 0.0;
  for (int j=0;j<count;++j) {
    const int bj = support[orbital*basis+j];
    const double xj = raw[orbital*basis+j];
    for (int i=0;i<count;++i) {
      const int bi = support[orbital*basis+i];
      norm2 += xj*raw[orbital*basis+i]*metric[bj*metric_ld+bi];
    }
  }
  if (!(norm2 > DBL_EPSILON) || !isfinite(norm2)) {
    atomicCAS(info,0,orbital+1);
    return;
  }
  const double inv = rsqrt(norm2);
  inverse_norms[orbital] = inv;
  for (int b=0;b<basis;++b) normalized[orbital*basis+b] = 0.0;
  for (int j=0;j<count;++j)
    normalized[orbital*basis+support[orbital*basis+j]] = raw[orbital*basis+j]*inv;
}

__global__ void normalize_direction_kernel(const double* direction,
    const int* support, const int* differentiable,
    const double* normalized, const double* s_normalized,
    const double* inverse_norms, int basis, int orbitals,
    double* delta_normalized) {
  const int orbital = blockIdx.x*blockDim.x+threadIdx.x;
  if (orbital >= orbitals) return;
  const double inv = inverse_norms[orbital];
  double projection = 0.0;
  for (int j=0;j<differentiable[orbital];++j) {
    const int b = support[orbital*basis+j];
    projection += direction[orbital*basis+j]*s_normalized[orbital*basis+b];
  }
  projection *= inv;
  for (int b=0;b<basis;++b)
    delta_normalized[orbital*basis+b] = -projection*normalized[orbital*basis+b];
  for (int j=0;j<differentiable[orbital];++j) {
    const int b = support[orbital*basis+j];
    delta_normalized[orbital*basis+b] += inv*direction[orbital*basis+j];
  }
}

__global__ void normalization_pullback_kernel(const int* support,
    const int* differentiable, const double* normalized,
    const double* s_normalized, const double* inverse_norms,
    const double* gradient, int basis, int orbitals, double* raw_gradient) {
  const int orbital = blockIdx.x*blockDim.x+threadIdx.x;
  if (orbital >= orbitals) return;
  double scalar = 0.0;
  for (int b=0;b<basis;++b)
    scalar += gradient[orbital*basis+b]*normalized[orbital*basis+b];
  for (int j=0;j<basis;++j) raw_gradient[orbital*basis+j] = 0.0;
  for (int j=0;j<differentiable[orbital];++j) {
    const int b = support[orbital*basis+j];
    raw_gradient[orbital*basis+j] = inverse_norms[orbital]*
        (gradient[orbital*basis+b]-scalar*s_normalized[orbital*basis+b]);
  }
}

__global__ void normalization_pullback_direction_kernel(const int* support,
    const int* differentiable, const double* normalized,
    const double* delta_normalized, const double* s_normalized,
    const double* s_delta_normalized, const double* inverse_norms,
    const double* raw_direction, const double* gradient,
    const double* delta_gradient, int basis, int orbitals,
    double* raw_gradient_direction) {
  const int orbital = blockIdx.x*blockDim.x+threadIdx.x;
  if (orbital >= orbitals) return;
  const double inv = inverse_norms[orbital];
  double scalar=0.0,delta_scalar=0.0,projection=0.0;
  for (int b=0;b<basis;++b) {
    scalar += gradient[orbital*basis+b]*normalized[orbital*basis+b];
    delta_scalar += delta_gradient[orbital*basis+b]*
                        normalized[orbital*basis+b] +
                    gradient[orbital*basis+b]*
                        delta_normalized[orbital*basis+b];
  }
  for (int j=0;j<differentiable[orbital];++j)
    projection += raw_direction[orbital*basis+j]*
        s_normalized[orbital*basis+support[orbital*basis+j]];
  projection *= inv;
  for (int j=0;j<basis;++j) raw_gradient_direction[orbital*basis+j]=0.0;
  for (int j=0;j<differentiable[orbital];++j) {
    const int b=support[orbital*basis+j], index=orbital*basis+b;
    raw_gradient_direction[orbital*basis+j]=inv*
        (delta_gradient[index]-delta_scalar*s_normalized[index]-
         scalar*s_delta_normalized[index]-
         projection*(gradient[index]-scalar*s_normalized[index]));
  }
}

__global__ void phi_kernel(double* output,const double* input,int n) {
  const int index=blockIdx.x*blockDim.x+threadIdx.x;
  if (index<n*n) {
    const int row=index%n,col=index/n;
    output[index]=row>col?0.0:(row==col?0.5:1.0)*input[index];
  }
}

// Phi maps symmetric Z to the upper triangular Cholesky generator K.
// Its adjoint is symmetric under the full Frobenius inner product.
__global__ void phi_adjoint_kernel(double* output,const double* input,int n) {
  const int index=blockIdx.x*blockDim.x+threadIdx.x;
  if (index<n*n) {
    const int row=index%n,col=index/n;
    output[index]=0.5*input[row<col?row+col*n:col+row*n];
  }
}

__global__ void upper_from_lower_kernel(double* upper,const double* lower,int n) {
  const int index=blockIdx.x*blockDim.x+threadIdx.x;
  if (index<n*n) {
    const int row=index%n,col=index/n;
    upper[index]=row<=col?lower[col+row*n]:0.0;
  }
}

void phi(Context& context,Mutable output,View input,bool adjoint) {
  const int n=static_cast<int>(output.rows);
  if (n) {
    if (adjoint)
      phi_adjoint_kernel<<<grid(n*n),kThreads,0,context.stream()>>>(
          output.data,input.data,n);
    else
      phi_kernel<<<grid(n*n),kThreads,0,context.stream()>>>(
          output.data,input.data,n);
    check_cuda(cudaGetLastError(),"CUDA orbital Cholesky differential kernel");
  }
}

}  // namespace

struct OrbitalGeometry::Impl {
  explicit Impl(Context& context):info(context,1) {}
  Buffer<int> info;
  std::unique_ptr<Cholesky> inactive_cholesky;
};

OrbitalDirection::OrbitalDirection(Context& context,int basis,int orbitals,
                                   int inactive,int active)
    : raw(context,basis,orbitals),normalized(context,basis,orbitals),
      inactive_dual(context,basis,inactive),
      inactive_density(context,basis,basis),
      active_auxiliary(context,basis,active),
      active_overlap(context,active,active),
      frame_transform(context,active,active),
      orthogonal_active(context,basis,active) {}

OrbitalGeometry::OrbitalGeometry(Context& context,
    const vb::OrbitalPreparationInput& input,View overlap)
    : impl_(std::make_unique<Impl>(context)),context_(&context),
      basis_(static_cast<int>(input.n_basis_functions)),
      orbitals_(static_cast<int>(input.n_orbitals)),
      inactive_(static_cast<int>((input.n_total_electrons-input.n_active_electrons)/2)),
      active_count_(static_cast<int>(input.n_active_orbitals)),
      support_(context,input.n_basis_functions*input.n_orbitals),
      stored_counts_(context,input.n_orbitals),
      differentiable_counts_(context,input.n_orbitals),
      overlap_(overlap),raw_(context,basis_,orbitals_),
      normalized_(context,basis_,orbitals_),
      overlap_normalized_(context,basis_,orbitals_),
      inverse_norms_(context,orbitals_,1),
      inverse_inactive_metric_(context,inactive_,inactive_),
      dual_(context,basis_,inactive_),density_(context,basis_,basis_),
      active_(context,basis_,active_count_),
      active_overlap_(context,active_count_,active_count_),
      frame_transform_(context,active_count_,active_count_),
      frame_inverse_(context,active_count_,active_count_),
      frame_(context,basis_,active_count_),
      inactive_active_overlap_(context,inactive_,active_count_) {
  check_view(overlap_,basis_,basis_,"CUDA shared AO overlap");
  if (input.n_total_electrons<input.n_active_electrons ||
      (input.n_total_electrons-input.n_active_electrons)%2 ||
      basis_<=0 || orbitals_<=0 || active_count_<=0 ||
      inactive_<0 || inactive_+active_count_>orbitals_ ||
      input.ao_overlap_matrix.rows()!=basis_ ||
      input.ao_overlap_matrix.cols()!=basis_ ||
      input.orbital_basis_index_table.size()!=
          static_cast<std::size_t>(basis_)*orbitals_ ||
      input.orbital_value_table.size()!=
          static_cast<std::size_t>(basis_)*orbitals_ ||
      input.orbital_basis_counts.size()!=static_cast<std::size_t>(orbitals_))
    throw std::invalid_argument("CUDA orbital preparation input is invalid");
  std::vector<int> support(input.orbital_basis_index_table.size(),-1);
  std::vector<int> stored(orbitals_),differentiable(orbitals_);
  for (int o=0;o<orbitals_;++o) {
    stored[o]=vb::stored_sparse_orbital_coefficient_count(input,o);
    differentiable[o]=vb::differentiable_sparse_orbital_parameter_count(input,o);
    if (stored[o]<=0 || stored[o]>basis_ || differentiable[o]<0 ||
        differentiable[o]>stored[o])
      throw std::invalid_argument("CUDA orbital sparse coefficient count is invalid");
    std::vector<unsigned char> seen(basis_,0);
    for (int j=0;j<stored[o];++j) {
      const int b=input.orbital_basis_index_table[o*basis_+j]-1;
      if (b<0 || b>=basis_ || seen[b]++)
        throw std::invalid_argument("CUDA orbital sparse support is invalid");
      support[o*basis_+j]=b;
    }
  }
  context.copy_to_device(support_.data(),support.data(),support.size()*sizeof(int));
  context.copy_to_device(stored_counts_.data(),stored.data(),stored.size()*sizeof(int));
  context.copy_to_device(differentiable_counts_.data(),differentiable.data(),
                         differentiable.size()*sizeof(int));
  context.copy_to_device(raw_.data(),input.orbital_value_table.data(),
                         static_cast<std::size_t>(basis_)*orbitals_*sizeof(double));
}

OrbitalGeometry::~OrbitalGeometry()=default;

void OrbitalGeometry::set_raw(View raw) {
  check_view(raw,basis_,orbitals_,"CUDA raw orbitals");
  copy(*context_,raw_.view(),raw);
  prepared_=false;
}

void OrbitalGeometry::prepare() {
  auto& context=*context_;
  prepared_=false;
  check_cuda(cudaMemsetAsync(impl_->info.data(),0,sizeof(int),context.stream()),
             "CUDA orbital normalization info reset");
  normalize_kernel<<<grid(orbitals_),kThreads,0,context.stream()>>>(
      raw_.data(),support_.data(),stored_counts_.data(),overlap_.data,
      blas_int(overlap_.ld),basis_,orbitals_,normalized_.data(),inverse_norms_.data(),
      impl_->info.data());
  check_cuda(cudaGetLastError(),"CUDA orbital normalization kernel");
  int status=0;
  context.copy_to_host(&status,impl_->info.data(),sizeof(int));
  if (status)
    throw std::runtime_error("CUDA orbital normalization failed for orbital "+
                             std::to_string(status-1));
  gemm(context,overlap_normalized_.view(),overlap_,normalized_.view());
  zero(context,density_.view());
  if (inactive_) {
    auto metric=product(context,normalized_.view().columns(0,inactive_),
                        overlap_normalized_.view().columns(0,inactive_),
                        Operation::transpose);
    impl_->inactive_cholesky=std::make_unique<Cholesky>(context,metric.view());
    identity(context,inverse_inactive_metric_.view());
    impl_->inactive_cholesky->solve(inverse_inactive_metric_.view(),
                                    inverse_inactive_metric_.view());
    gemm(context,dual_.view(),normalized_.view().columns(0,inactive_),
         inverse_inactive_metric_.view());
    gemm(context,density_.view(),dual_.view(),
         normalized_.view().columns(0,inactive_),1.0,0.0,
         Operation::none,Operation::transpose);
    gemm(context,inactive_active_overlap_.view(),
         normalized_.view().columns(0,inactive_),
         overlap_normalized_.view().columns(inactive_,active_count_),
         1.0,0.0,Operation::transpose);
  }
  copy(context,active_.view(),normalized_.view().columns(inactive_,active_count_));
  if (inactive_)
    gemm(context,active_.view(),dual_.view(),inactive_active_overlap_.view(),
         -1.0,1.0);
  auto s_active=product(context,overlap_,active_.view());
  gemm(context,active_overlap_.view(),active_.view(),s_active.view(),
       1.0,0.0,Operation::transpose);

  // The CPU direct-CI convention is R^T R=A with upper triangular R.
  Matrix<double> factor=duplicate(context,active_overlap_.view());
  cusolverDnHandle_t solver=nullptr;
  check_solver(cusolverDnCreate(&solver),"cusolverDnCreate orbital frame");
  try {
    check_solver(cusolverDnSetStream(solver,context.stream()),
                 "cusolverDnSetStream orbital frame");
    int workspace_size=0;
    check_solver(cusolverDnDpotrf_bufferSize(solver,CUBLAS_FILL_MODE_LOWER,
        active_count_,factor.data(),active_count_,&workspace_size),
        "cusolverDnDpotrf_bufferSize orbital frame");
    if (workspace_size<0) throw std::runtime_error("negative orbital frame workspace");
    Buffer<double> workspace(context,workspace_size);
    check_solver(cusolverDnDpotrf(solver,CUBLAS_FILL_MODE_LOWER,
        active_count_,factor.data(),active_count_,workspace.data(),
        workspace_size,impl_->info.data()),"cusolverDnDpotrf orbital frame");
    context.copy_to_host(&status,impl_->info.data(),sizeof(int));
    if (status) throw std::runtime_error("CUDA active overlap is not positive definite");
    const double one=1.0;
    identity(context,frame_inverse_.view());
    check_cublas(cublasDtrsm(context.blas(),CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,
        active_count_,active_count_,&one,factor.data(),active_count_,
        frame_inverse_.data(),active_count_),
        "cublasDtrsm orbital frame inverse");
    // R is the transpose of the lower Cholesky triangle.
    upper_from_lower_kernel<<<grid(active_count_*active_count_),kThreads,
        0,context.stream()>>>(frame_transform_.data(),factor.data(),active_count_);
    check_cuda(cudaGetLastError(),"CUDA orbital upper factor extraction");
  } catch (...) {
    cusolverDnDestroy(solver);
    throw;
  }
  cusolverDnDestroy(solver);
  gemm(context,frame_.view(),active_.view(),frame_inverse_.view());
  prepared_=true;
}

OrbitalDirection OrbitalGeometry::direction(View raw_direction) const {
  if (!prepared_) throw std::logic_error("CUDA orbital direction requires prepare");
  check_view(raw_direction,basis_,orbitals_,"CUDA raw orbital direction");
  auto& context=*context_;
  OrbitalDirection tangent(context,basis_,orbitals_,inactive_,active_count_);
  copy(context,tangent.raw.view(),raw_direction);
  normalize_direction_kernel<<<grid(orbitals_),kThreads,0,context.stream()>>>(
      tangent.raw.data(),support_.data(),differentiable_counts_.data(),
      normalized_.data(),overlap_normalized_.data(),inverse_norms_.data(),
      basis_,orbitals_,tangent.normalized.data());
  check_cuda(cudaGetLastError(),"CUDA orbital normalization direction kernel");
  auto s_delta=product(context,overlap_,tangent.normalized.view());
  zero(context,tangent.inactive_density.view());
  if (inactive_) {
    auto ci=normalized_.view().columns(0,inactive_);
    auto dci=tangent.normalized.view().columns(0,inactive_);
    auto sci=overlap_normalized_.view().columns(0,inactive_);
    auto sdci=s_delta.view().columns(0,inactive_);
    auto dm=product(context,dci,sci,Operation::transpose);
    gemm(context,dm.view(),ci,sdci,1.0,1.0,Operation::transpose);
    auto db=product(context,inverse_inactive_metric_.view(),dm.view());
    auto db_full=product(context,db.view(),inverse_inactive_metric_.view());
    auto ddual=product(context,dci,inverse_inactive_metric_.view());
    gemm(context,ddual.view(),ci,db_full.view(),-1.0,1.0);
    tangent.inactive_dual=std::move(ddual);
    gemm(context,tangent.inactive_density.view(),
         tangent.inactive_dual.view(),ci,1.0,0.0,
         Operation::none,Operation::transpose);
    gemm(context,tangent.inactive_density.view(),dual_.view(),dci,
         1.0,1.0,Operation::none,Operation::transpose);
  }
  copy(context,tangent.active_auxiliary.view(),
       tangent.normalized.view().columns(inactive_,active_count_));
  if (inactive_) {
    auto ci=normalized_.view().columns(0,inactive_);
    auto dci=tangent.normalized.view().columns(0,inactive_);
    auto sca=overlap_normalized_.view().columns(inactive_,active_count_);
    auto sdca=s_delta.view().columns(inactive_,active_count_);
    auto dk=product(context,dci,sca,Operation::transpose);
    gemm(context,dk.view(),ci,sdca,1.0,1.0,Operation::transpose);
    gemm(context,tangent.active_auxiliary.view(),tangent.inactive_dual.view(),
         inactive_active_overlap_.view(),-1.0,1.0);
    gemm(context,tangent.active_auxiliary.view(),dual_.view(),dk.view(),
         -1.0,1.0);
  }
  auto sa=product(context,overlap_,active_.view());
  auto sda=product(context,overlap_,tangent.active_auxiliary.view());
  gemm(context,tangent.active_overlap.view(),tangent.active_auxiliary.view(),
       sa.view(),1.0,0.0,Operation::transpose);
  gemm(context,tangent.active_overlap.view(),active_.view(),sda.view(),
       1.0,1.0,Operation::transpose);
  auto zleft=product(context,frame_inverse_.view(),
       tangent.active_overlap.view(),Operation::transpose);
  auto z=product(context,zleft.view(),frame_inverse_.view());
  Matrix<double> k(context,active_count_,active_count_);
  phi(context,k.view(),z.view(),false);
  gemm(context,tangent.frame_transform.view(),k.view(),frame_transform_.view());
  gemm(context,tangent.orthogonal_active.view(),
       tangent.active_auxiliary.view(),frame_inverse_.view());
  gemm(context,tangent.orthogonal_active.view(),frame_.view(),k.view(),
       -1.0,1.0);
  return tangent;
}

namespace {

struct FrameAdjoints {
  FrameAdjoints(Context& context,int basis,int active)
      : active_auxiliary(context,basis,active),overlap(context,active,active) {}
  Matrix<double> active_auxiliary,overlap;
};

FrameAdjoints frame_pullback(Context& context,View active,View frame,
    View inverse,View transform,View active_auxiliary_gradient,
    View active_overlap_gradient,View transform_gradient,View frame_gradient) {
  const int basis=static_cast<int>(active.rows);
  const int n=static_cast<int>(active.cols);
  FrameAdjoints result(context,basis,n);
  zero(context,result.active_auxiliary.view());
  zero(context,result.overlap.view());
  if (supplied(active_auxiliary_gradient))
    axpy(context,result.active_auxiliary.view(),active_auxiliary_gradient,1.0);
  if (supplied(active_overlap_gradient))
    axpy(context,result.overlap.view(),active_overlap_gradient,1.0);
  Matrix<double> kbar=vacant(context,n,n);
  if (supplied(transform_gradient)) {
    auto term=product(context,transform_gradient,transform,
                      Operation::none,Operation::transpose);
    axpy(context,kbar.view(),term.view(),1.0);
  }
  if (supplied(frame_gradient)) {
    auto direct=product(context,frame_gradient,inverse,
                        Operation::none,Operation::transpose);
    axpy(context,result.active_auxiliary.view(),direct.view(),1.0);
    auto term=product(context,frame,frame_gradient,Operation::transpose);
    axpy(context,kbar.view(),term.view(),-1.0);
  }
  if (supplied(transform_gradient) || supplied(frame_gradient)) {
    Matrix<double> zbar(context,n,n);
    phi(context,zbar.view(),kbar.view(),true);
    auto left=product(context,inverse,zbar.view());
    auto abar=product(context,left.view(),inverse,
                      Operation::none,Operation::transpose);
    axpy(context,result.overlap.view(),abar.view(),1.0);
  }
  return result;
}

}  // namespace

void OrbitalGeometry::pullback(View gt,View gp,View ga,View gr,View gf,
                               Mutable raw_gradient) const {
  if (!prepared_) throw std::logic_error("CUDA orbital pullback requires prepare");
  check_view(gt,basis_,active_count_,"active auxiliary gradient",true);
  check_view(gp,basis_,basis_,"inactive density gradient",true);
  check_view(ga,active_count_,active_count_,"active overlap gradient",true);
  check_view(gr,active_count_,active_count_,"frame transform gradient",true);
  check_view(gf,basis_,active_count_,"orthogonal active gradient",true);
  check_view(raw_gradient,basis_,orbitals_,"raw orbital gradient");
  auto& context=*context_;
  auto adj=frame_pullback(context,active_.view(),frame_.view(),
      frame_inverse_.view(),frame_transform_.view(),gt,ga,gr,gf);
  auto s_active=product(context,overlap_,active_.view());
  auto t_extra=product(context,s_active.view(),adj.overlap.view());
  gemm(context,adj.active_auxiliary.view(),s_active.view(),adj.overlap.view(),
       1.0,1.0,Operation::none,Operation::transpose);
  axpy(context,adj.active_auxiliary.view(),t_extra.view(),1.0);
  Matrix<double> gc=vacant(context,basis_,orbitals_);
  if (!inactive_) {
    copy(context,gc.view().columns(0,active_count_),
         adj.active_auxiliary.view());
  } else {
    const auto ci=normalized_.view().columns(0,inactive_);
    const auto sca=overlap_normalized_.view().columns(inactive_,active_count_);
    auto& g=adj.active_auxiliary;
    auto q=vacant(context,basis_,basis_);
    if (supplied(gp)) add_sym(context,q.view(),gp);
    auto cross=product(context,g.view(),sca,Operation::none,
                       Operation::transpose);
    add_sym(context,q.view(),cross.view(),-1.0);
    auto qd=product(context,q.view(),dual_.view());
    auto h=product(context,ci,qd.view(),Operation::transpose);
    auto sh=product(context,overlap_,dual_.view());
    gemm(context,qd.view(),sh.view(),h.view(),-1.0,1.0);
    copy(context,gc.view().columns(0,inactive_),qd.view());
    auto ctg=product(context,ci,g.view(),Operation::transpose);
    auto gca=duplicate(context,g.view());
    gemm(context,gca.view(),sh.view(),ctg.view(),-1.0,1.0);
    copy(context,gc.view().columns(inactive_,active_count_),gca.view());
  }
  normalization_pullback_kernel<<<grid(orbitals_),kThreads,0,context.stream()>>>(
      support_.data(),differentiable_counts_.data(),normalized_.data(),
      overlap_normalized_.data(),inverse_norms_.data(),gc.data(),
      basis_,orbitals_,raw_gradient.data);
  check_cuda(cudaGetLastError(),"CUDA orbital normalization pullback kernel");
}

void OrbitalGeometry::pullback_direction(const OrbitalDirection& tangent,
    View gt,View gp,View ga,View gr,View gf,
    View dgt,View dgp,View dga,View dgr,View dgf,
    Mutable raw_gradient_direction,View density_direction_projection) const {
  if (!prepared_) throw std::logic_error("CUDA orbital pullback direction requires prepare");
  check_view(gt,basis_,active_count_,"active auxiliary gradient",true);
  check_view(gp,basis_,basis_,"inactive density gradient",true);
  check_view(ga,active_count_,active_count_,"active overlap gradient",true);
  check_view(gr,active_count_,active_count_,"frame transform gradient",true);
  check_view(gf,basis_,active_count_,"orthogonal active gradient",true);
  check_view(dgt,basis_,active_count_,"active auxiliary gradient direction",true);
  check_view(dgp,basis_,basis_,"inactive density gradient direction",true);
  check_view(density_direction_projection,basis_,inactive_,
      "projected inactive density gradient direction",true);
  if (supplied(dgp) && supplied(density_direction_projection))
    throw std::invalid_argument("duplicate inactive density gradient direction");
  check_view(dga,active_count_,active_count_,"active overlap gradient direction",true);
  check_view(dgr,active_count_,active_count_,"frame transform gradient direction",true);
  check_view(dgf,basis_,active_count_,"orthogonal active gradient direction",true);
  check_view(raw_gradient_direction,basis_,orbitals_,"raw orbital gradient direction");
  auto& context=*context_;
  const int n=active_count_;
  const auto u=frame_inverse_.view();
  const auto r=frame_transform_.view();
  // K=dR R^-1 and dU=-U K for U=R^-1.
  auto k=product(context,tangent.frame_transform.view(),u);
  auto du=product(context,u,k.view());
  auto adj=frame_pullback(context,active_.view(),frame_.view(),u,r,
                          gt,ga,gr,gf);
  FrameAdjoints dadj(context,basis_,n);
  zero(context,dadj.active_auxiliary.view());
  zero(context,dadj.overlap.view());
  if (supplied(dgt)) axpy(context,dadj.active_auxiliary.view(),dgt,1.0);
  if (supplied(dga)) axpy(context,dadj.overlap.view(),dga,1.0);
  Matrix<double> kbar=vacant(context,n,n);
  Matrix<double> dkbar=vacant(context,n,n);
  if (supplied(gr)) {
    auto term=product(context,gr,r,Operation::none,Operation::transpose);
    axpy(context,kbar.view(),term.view(),1.0);
    auto dterm=product(context,gr,tangent.frame_transform.view(),
                       Operation::none,Operation::transpose);
    axpy(context,dkbar.view(),dterm.view(),1.0);
  }
  if (supplied(dgr)) {
    auto dterm=product(context,dgr,r,Operation::none,Operation::transpose);
    axpy(context,dkbar.view(),dterm.view(),1.0);
  }
  if (supplied(gf)) {
    auto term=product(context,frame_.view(),gf,Operation::transpose);
    axpy(context,kbar.view(),term.view(),-1.0);
    // Already present in adj.active_auxiliary, but its directional term is new.
    auto ddirect=product(context,gf,du.view(),
                         Operation::none,Operation::transpose);
    axpy(context,dadj.active_auxiliary.view(),ddirect.view(),-1.0);
    auto dterm=product(context,tangent.orthogonal_active.view(),gf,
                       Operation::transpose);
    axpy(context,dkbar.view(),dterm.view(),-1.0);
  }
  if (supplied(dgf)) {
    auto direct=product(context,dgf,u,Operation::none,Operation::transpose);
    axpy(context,dadj.active_auxiliary.view(),direct.view(),1.0);
    auto dterm=product(context,frame_.view(),dgf,Operation::transpose);
    axpy(context,dkbar.view(),dterm.view(),-1.0);
  }
  if (supplied(gr)||supplied(dgr)||supplied(gf)||supplied(dgf)) {
    Matrix<double> zbar(context,n,n),dzbar(context,n,n);
    phi(context,zbar.view(),kbar.view(),true);
    phi(context,dzbar.view(),dkbar.view(),true);
    auto uz=product(context,u,zbar.view());
    auto udz=product(context,u,dzbar.view());
    auto term=product(context,udz.view(),u,
                      Operation::none,Operation::transpose);
    axpy(context,dadj.overlap.view(),term.view(),1.0);
    auto duz=product(context,du.view(),zbar.view());
    term=product(context,duz.view(),u,
                 Operation::none,Operation::transpose);
    axpy(context,dadj.overlap.view(),term.view(),-1.0);
    term=product(context,uz.view(),du.view(),
                 Operation::none,Operation::transpose);
    axpy(context,dadj.overlap.view(),term.view(),-1.0);
  }
  auto sa=product(context,overlap_,active_.view());
  auto sda=product(context,overlap_,tangent.active_auxiliary.view());
  auto aterm=product(context,sa.view(),adj.overlap.view());
  gemm(context,adj.active_auxiliary.view(),sa.view(),adj.overlap.view(),
       1.0,1.0,Operation::none,Operation::transpose);
  axpy(context,adj.active_auxiliary.view(),aterm.view(),1.0);
  auto daterm=product(context,sda.view(),adj.overlap.view());
  axpy(context,dadj.active_auxiliary.view(),daterm.view(),1.0);
  gemm(context,dadj.active_auxiliary.view(),sda.view(),adj.overlap.view(),
       1.0,1.0,Operation::none,Operation::transpose);
  daterm=product(context,sa.view(),dadj.overlap.view());
  axpy(context,dadj.active_auxiliary.view(),daterm.view(),1.0);
  gemm(context,dadj.active_auxiliary.view(),sa.view(),dadj.overlap.view(),
       1.0,1.0,Operation::none,Operation::transpose);

  Matrix<double> gc=vacant(context,basis_,orbitals_);
  Matrix<double> dgc=vacant(context,basis_,orbitals_);
  if (!inactive_) {
    copy(context,gc.view().columns(0,n),adj.active_auxiliary.view());
    copy(context,dgc.view().columns(0,n),dadj.active_auxiliary.view());
  } else {
    const auto ci=normalized_.view().columns(0,inactive_);
    const auto dci=tangent.normalized.view().columns(0,inactive_);
    const auto sca=overlap_normalized_.view().columns(inactive_,n);
    const auto g=adj.active_auxiliary.view();
    const auto dg=dadj.active_auxiliary.view();
    auto sdca=product(context,overlap_,
                      tangent.normalized.view().columns(inactive_,n));
    auto q=vacant(context,basis_,basis_);
    auto dq=vacant(context,basis_,basis_);
    if (supplied(gp)) add_sym(context,q.view(),gp);
    if (supplied(dgp)) add_sym(context,dq.view(),dgp);
    auto cross=product(context,g,sca,Operation::none,Operation::transpose);
    add_sym(context,q.view(),cross.view(),-1.0);
    auto dcross=product(context,dg,sca,Operation::none,Operation::transpose);
    add_sym(context,dq.view(),dcross.view(),-1.0);
    dcross=product(context,g,sdca.view(),Operation::none,
                   Operation::transpose);
    add_sym(context,dq.view(),dcross.view(),-1.0);
    auto qd=product(context,q.view(),dual_.view());
    auto dqd=product(context,dq.view(),dual_.view());
    if (supplied(density_direction_projection))
      axpy(context,dqd.view(),density_direction_projection,1.0);
    gemm(context,dqd.view(),q.view(),tangent.inactive_dual.view(),
         1.0,1.0);
    auto h=product(context,ci,qd.view(),Operation::transpose);
    auto dh=product(context,dci,qd.view(),Operation::transpose);
    gemm(context,dh.view(),ci,dqd.view(),1.0,1.0,Operation::transpose);
    auto sdual=product(context,overlap_,dual_.view());
    auto dsdual=product(context,overlap_,tangent.inactive_dual.view());
    gemm(context,qd.view(),sdual.view(),h.view(),-1.0,1.0);
    gemm(context,dqd.view(),dsdual.view(),h.view(),-1.0,1.0);
    gemm(context,dqd.view(),sdual.view(),dh.view(),-1.0,1.0);
    copy(context,gc.view().columns(0,inactive_),qd.view());
    copy(context,dgc.view().columns(0,inactive_),dqd.view());
    auto ctg=product(context,ci,g,Operation::transpose);
    auto dctg=product(context,dci,g,Operation::transpose);
    gemm(context,dctg.view(),ci,dg,1.0,1.0,Operation::transpose);
    auto gca=duplicate(context,g);
    auto dgca=duplicate(context,dg);
    gemm(context,gca.view(),sdual.view(),ctg.view(),-1.0,1.0);
    gemm(context,dgca.view(),dsdual.view(),ctg.view(),-1.0,1.0);
    gemm(context,dgca.view(),sdual.view(),dctg.view(),-1.0,1.0);
    copy(context,gc.view().columns(inactive_,n),gca.view());
    copy(context,dgc.view().columns(inactive_,n),dgca.view());
  }
  auto s_delta_c=product(context,overlap_,tangent.normalized.view());
  normalization_pullback_direction_kernel<<<grid(orbitals_),kThreads,
      0,context.stream()>>>(support_.data(),differentiable_counts_.data(),
      normalized_.data(),tangent.normalized.data(),
      overlap_normalized_.data(),s_delta_c.data(),inverse_norms_.data(),
      tangent.raw.data(),gc.data(),dgc.data(),basis_,orbitals_,
      raw_gradient_direction.data);
  check_cuda(cudaGetLastError(),"CUDA orbital normalization pullback direction");
}

}  // namespace xmvb::cuda
