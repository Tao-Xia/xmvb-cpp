#include "cuda/vb/ci.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "cuda/linalg/cholesky.hpp"
#include "cuda/vb/action.hpp"
#include "vbscf/structures/overlap/representation.hpp"

namespace xmvb::cuda {
namespace {

__global__ void identity(linalg::MatrixView x, std::int64_t count) {
  const auto i=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i<count) x.data[i]=(i%x.rows==i/x.rows) ? 1.0:0.0;
}

__global__ void positive_model(const double* h,const double* m,double energy,
    double* model,int n) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n) {
    const double em=energy*m[i];
    const double scale=fmax(1.0,fmax(fabs(h[i]),fabs(em)));
    model[i]=fmax(fabs(h[i]-em),sqrt(2.2204460492503131e-16)*scale);
  }
}

__global__ void extract_diagonal(linalg::ConstMatrixView metric,double* diagonal,int n) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n) diagonal[i]=metric.data[i+metric.ld*i];
}

__global__ void select_jacobi_seed(linalg::MatrixView seed,
    const double* h,const double* m,int* selected,int n,int column) {
  __shared__ double minima[256];
  __shared__ int indices[256];
  const int lane=threadIdx.x;
  double best=INFINITY;
  int chosen=n;
  for (int i=lane;i<n;i+=blockDim.x) {
    bool used=false;
    for (int prior=0;prior<column;++prior) used|=(selected[prior]==i);
    if (used) continue;
    const double quotient=(isfinite(h[i]) && isfinite(m[i]) && m[i]>0.0)
        ? h[i]/m[i] : INFINITY;
    if (quotient<best || (quotient==best && i<chosen)) {
      best=quotient;
      chosen=i;
    }
  }
  minima[lane]=best;
  indices[lane]=chosen;
  __syncthreads();
  for (int stride=blockDim.x/2;stride>0;stride/=2) {
    if (lane<stride && (minima[lane+stride]<minima[lane] ||
        (minima[lane+stride]==minima[lane] && indices[lane+stride]<indices[lane]))) {
      minima[lane]=minima[lane+stride];
      indices[lane]=indices[lane+stride];
    }
    __syncthreads();
  }
  if (lane==0) {
    selected[column]=indices[0];
    seed.data[indices[0]+seed.ld*column]=1.0;
  }
}

__global__ void rayleigh_norm(double* energy,double* norm) {
  if (blockIdx.x || threadIdx.x) return;
  const double squared=*norm;
  *energy/=squared;
  *norm=sqrt(squared);
}

linalg::DavidsonResult download_result(Context& context,
    const DeviceDavidsonResult& device) {
  linalg::DavidsonResult host;
  host.eigenpairs.eigenvalues=device.eigenpairs.eigenvalues;
  const int n=static_cast<int>(device.eigenpairs.eigenvector_matrix.rows());
  const int roots=static_cast<int>(device.eigenpairs.eigenvector_matrix.cols());
  host.eigenpairs.eigenvector_matrix.resize(static_cast<std::size_t>(n)*roots);
  context.copy_to_host(host.eigenpairs.eigenvector_matrix.data(),
      device.eigenpairs.eigenvector_matrix.data(),
      host.eigenpairs.eigenvector_matrix.size()*sizeof(double));
  host.hamiltonian_eigenvectors.resize(n,roots);
  copy_to_host(context,host.hamiltonian_eigenvectors,
      device.hamiltonian_eigenvectors.view());
  host.overlap_eigenvectors.resize(n,roots);
  copy_to_host(context,host.overlap_eigenvectors,device.overlap_view());
  host.relative_residual_norms=device.relative_residual_norms;
  host.stationarity_residual_norms=device.stationarity_residual_norms;
  host.iterations=device.iterations;
  host.block_actions=device.block_actions;
  host.peak_subspace_dimension=device.peak_subspace_dimension;
  return host;
}

}  // namespace

StructureCi::StructureCi(Context& context,const StructureAction& action,
    const SpinMetric& spin,bool complete_space)
    : context_(&context),action_(&action),spin_(&spin),
      diagonal_h_(context,action.size(),1),diagonal_m_(context,action.size(),1),
      positive_(context,action.size(),1),energy_(context,1,1) {
  if (spin.dimension()!=action.size())
    throw std::invalid_argument("CUDA CI spin metric dimension mismatch");
  if (complete_space) {
    full_=std::make_unique<FullMetric>(context,spin,action);
  } else if (vb::RepresentationOverlap::PreparedMetric::fits_selected(action.size())) {
    selected_mode_=true;
  }
  refresh();
}

StructureCi::~StructureCi()=default;

const char* StructureCi::preconditioner_name() const noexcept {
  return full_ ? "full_carrier_inverse" : selected_mode_ ? "selected_metric_inverse" :
      "spin_diagonal_model";
}

void StructureCi::refresh() {
  if (refreshed_version_==action_->version()) return;
  action_->diagonal_model(diagonal_h_.view(),diagonal_m_.view());
  if (selected_mode_) {
    // The exact selected metric changes with the orbital frame. Rebuild the
    // bounded factor, and use its exact diagonal in the Davidson model.
    selected_.reset();
    Matrix<double> unit(*context_,action_->size(),action_->size());
    Matrix<double> metric(*context_,action_->size(),action_->size());
    const std::int64_t count=unit.rows()*unit.cols();
    if ((count+255)/256>std::numeric_limits<unsigned int>::max())
      throw std::overflow_error("CUDA selected metric identity grid overflow");
    identity<<<static_cast<unsigned int>((count+255)/256),256,0,context_->stream()>>>(
        unit.view(),count);
    check_cuda(cudaGetLastError(),"CUDA selected metric identity");
    action_->apply_overlap(unit.view(),metric.view());
    extract_diagonal<<<(action_->size()+255)/256,256,0,context_->stream()>>>(
        metric.view(),diagonal_m_.data(),action_->size());
    check_cuda(cudaGetLastError(),"CUDA selected metric diagonal");
    selected_=std::make_unique<Cholesky>(*context_,metric.view());
  }
  refreshed_version_=action_->version();
}

void StructureCi::inverse_metric(linalg::ConstMatrixView rhs,
    linalg::MatrixView image) {
  if (!has_exact_metric_inverse())
    throw std::logic_error("CUDA CI has no exact selected metric inverse");
  refresh();
  if (full_) full_->inverse_metric(rhs,image);
  else selected_->solve(rhs,image);
}

void StructureCi::precondition(linalg::ConstMatrixView residual,double energy,
    linalg::MatrixView output) const {
  if (full_) {
    context_->copy_to_device(energy_.data(),&energy,sizeof(double));
    full_->apply_ci_preconditioner(residual,energy_.view(),0.0,output);
  } else if (selected_) {
    selected_->solve(residual,output);
  } else {
    const int n=action_->size();
    positive_model<<<(n+255)/256,256,0,context_->stream()>>>(
        diagonal_h_.data(),diagonal_m_.data(),energy,
        positive_.data(),n);
    check_cuda(cudaGetLastError(),"CUDA CI positive inverse model");
    spin_->solve_diagonal_model(residual,positive_.view(),output);
  }
}

linalg::DavidsonResult StructureCi::solve(const linalg::DavidsonOptions& options) {
  return download_result(*context_,solve_device(options));
}

linalg::DavidsonResult StructureCi::solve(
    const Eigen::Ref<const Eigen::MatrixXd>& initial,const linalg::DavidsonOptions& requested) {
  Matrix<double> input(*context_,initial.rows(),initial.cols());
  copy_to_device(*context_,input.view(),initial);
  return download_result(*context_,solve_device(input.view(),requested));
}

DeviceDavidsonResult StructureCi::solve_device(
    linalg::ConstMatrixView initial,const linalg::DavidsonOptions& requested) {
  refresh();
  auto options=requested;
  if (options.physical_residual_norm || options.residual_preconditioner)
    throw std::invalid_argument("CUDA structure CI does not accept CPU numerical callbacks");
  options.identity_overlap=identity_overlap();
  if (initial.rows!=action_->size() || initial.cols<options.n_roots ||
      initial.cols>options.max_subspace_dimension || initial.ld<initial.rows ||
      !initial.data)
    throw std::invalid_argument("CUDA structure CI initial block dimensions");
  Matrix<double> ci_guess;
  linalg::ConstMatrixView guess=initial;
  if (full_) {
    ci_guess=Matrix<double>(*context_,initial.rows,initial.cols);
    full_->to_ci_coordinates(initial,ci_guess.view());
    guess=ci_guess.view();
  }
  const DeviceEigenAction action=[this](linalg::ConstMatrixView x,
      linalg::MatrixView h,linalg::MatrixView m) {
    if (full_) full_->apply_ci_hamiltonian(x,h);
    else action_->apply(x,h,m);
  };
  const DeviceEigenPreconditioner inverse=[this](linalg::ConstMatrixView r,double e,
      linalg::MatrixView p) { precondition(r,e,p); };
  auto result=cuda::solve_device_davidson(*context_,action,diagonal_h_.view(),
      diagonal_m_.view(),guess,options,inverse);
  if (full_) {
    const int n=action_->size(),roots=options.n_roots;
    Matrix<double> m_structure(*context_,n,roots);
    full_->ci_covector_to_structure(result.eigenpairs.eigenvector_matrix.view(),
        m_structure.view());
    full_->ci_covector_to_structure(result.hamiltonian_eigenvectors.view(),
        result.hamiltonian_eigenvectors.view());
    full_->from_ci_coordinates(result.eigenpairs.eigenvector_matrix.view(),
        result.eigenpairs.eigenvector_matrix.view());
    result.overlap_eigenvectors=std::move(m_structure);
    result.identity_overlap=false;
  }
  return result;
}

DeviceDavidsonResult StructureCi::solve_device(const linalg::DavidsonOptions& options) {
  refresh();
  const int count=std::min(options.max_subspace_dimension,2*options.n_roots);
  if (count<=0 || count>action_->size())
    throw std::invalid_argument("CUDA structure CI initial count");
  Matrix<double> seed(*context_,action_->size(),count);
  Buffer<int> selected(*context_,count);
  check_cuda(cudaMemsetAsync(seed.data(),0,static_cast<std::size_t>(seed.rows())*
      seed.cols()*sizeof(double),context_->stream()),"CUDA structure CI clear seed");
  for (int column=0;column<count;++column) {
    select_jacobi_seed<<<1,256,0,context_->stream()>>>(seed.view(),
        diagonal_h_.data(),diagonal_m_.data(),selected.data(),action_->size(),column);
    check_cuda(cudaGetLastError(),"CUDA structure CI select Jacobi seed");
  }
  return solve_device(seed.view(),options);
}

void StructureCi::energy_norm_device(linalg::ConstMatrixView state,
    linalg::MatrixView energy,linalg::MatrixView norm) const {
  if (state.rows!=action_->size() || state.cols!=1 || state.ld<state.rows ||
      !state.data || energy.rows!=1 || energy.cols!=1 || !energy.data ||
      norm.rows!=1 || norm.cols!=1 || !norm.data || energy.data==norm.data)
    throw std::invalid_argument("CUDA CI energy/norm dimensions");
  Matrix<double> h(*context_,state.rows,1),m(*context_,state.rows,1);
  action_->apply(state,h.view(),m.view());
  check_cublas(cublasSetPointerMode(context_->blas(),CUBLAS_POINTER_MODE_DEVICE),
      "CUDA CI device scalar pointer mode");
  try {
    check_cublas(cublasDdot(context_->blas(),static_cast<int>(state.rows),
        state.data,1,h.data(),1,energy.data),"CUDA CI device energy dot");
    check_cublas(cublasDdot(context_->blas(),static_cast<int>(state.rows),
        state.data,1,m.data(),1,norm.data),"CUDA CI device norm dot");
  } catch (...) {
    cublasSetPointerMode(context_->blas(),CUBLAS_POINTER_MODE_HOST);
    throw;
  }
  check_cublas(cublasSetPointerMode(context_->blas(),CUBLAS_POINTER_MODE_HOST),
      "CUDA CI restore scalar pointer mode");
  rayleigh_norm<<<1,1,0,context_->stream()>>>(energy.data,norm.data);
  check_cuda(cudaGetLastError(),"CUDA CI device Rayleigh quotient");
}

}  // namespace xmvb::cuda
