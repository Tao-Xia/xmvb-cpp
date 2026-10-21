#include "cuda/vb/ci.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

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

}  // namespace

StructureCi::StructureCi(Context& context,const StructureAction& action,
    const SpinMetric& spin,bool complete_space)
    : context_(&context),action_(&action),spin_(&spin),
      diagonal_h_(context,action.size(),1),diagonal_m_(context,action.size(),1),
      positive_(context,action.size(),1),energy_(context,1,1) {
  if (spin.dimension()!=action.size())
    throw std::invalid_argument("CUDA CI spin metric dimension mismatch");
  const auto& diagonal=action.preconditioner_diagonal();
  copy_to_device(context,diagonal_h_.view(),diagonal.hamiltonian);
  copy_to_device(context,diagonal_m_.view(),diagonal.overlap);
  if (complete_space) {
    full_=std::make_unique<FullMetric>(context,spin,action);
  } else if (vb::RepresentationOverlap::PreparedMetric::fits_selected(action.size())) {
    // This is the production bounded selected-metric preconditioner, not an
    // admission test for the physical action. Larger selections use spin factors.
    Matrix<double> unit(context,action.size(),action.size());
    Matrix<double> metric(context,action.size(),action.size());
    const std::int64_t count=unit.rows()*unit.cols();
    identity<<<static_cast<unsigned int>((count+255)/256),256,0,context.stream()>>>(
        unit.view(),count);
    check_cuda(cudaGetLastError(),"CUDA selected metric identity");
    action.apply_overlap(unit.view(),metric.view());
    selected_=std::make_unique<Cholesky>(context,metric.view());
  }
}

StructureCi::~StructureCi()=default;

const char* StructureCi::preconditioner_name() const noexcept {
  return full_ ? "full_carrier_inverse" : selected_ ? "selected_metric_inverse" :
      "spin_diagonal_model";
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
  const auto& d=action_->preconditioner_diagonal();
  const auto initial=linalg::make_davidson_initial_vectors(d.hamiltonian,d.overlap,
      std::min(options.max_subspace_dimension,2*options.n_roots));
  return solve(initial,options);
}

linalg::DavidsonResult StructureCi::solve(
    const Eigen::Ref<const Eigen::MatrixXd>& initial,const linalg::DavidsonOptions& requested) {
  auto options=requested;
  if (options.physical_residual_norm || options.residual_preconditioner)
    throw std::invalid_argument("CUDA structure CI does not accept CPU numerical callbacks");
  options.identity_overlap=identity_overlap();
  Eigen::MatrixXd guess=initial;
  if (full_) {
    Matrix<double> input(*context_,initial.rows(),initial.cols());
    Matrix<double> output(*context_,initial.rows(),initial.cols());
    copy_to_device(*context_,input.view(),initial);
    full_->to_ci_coordinates(input.view(),output.view());
    copy_to_host(*context_,guess,output.view());
  }
  const DeviceEigenAction action=[this](linalg::ConstMatrixView x,
      linalg::MatrixView h,linalg::MatrixView m) {
    if (full_) full_->apply_ci_hamiltonian(x,h);
    else action_->apply(x,h,m);
  };
  const DeviceEigenPreconditioner inverse=[this](linalg::ConstMatrixView r,double e,
      linalg::MatrixView p) { precondition(r,e,p); };
  const auto& d=action_->preconditioner_diagonal();
  auto result=cuda::solve_davidson(*context_,action,d.hamiltonian,d.overlap,
      guess,options,inverse);
  if (full_) {
    const int n=action_->size(),roots=options.n_roots;
    const Eigen::Map<const Eigen::MatrixXd> x(result.eigenpairs.eigenvector_matrix.data(),n,roots);
    Matrix<double> input(*context_,n,roots),output(*context_,n,roots);
    Eigen::MatrixXd coefficients(n,roots);
    copy_to_device(*context_,input.view(),x);
    full_->from_ci_coordinates(input.view(),output.view());
    copy_to_host(*context_,coefficients,output.view());
    result.eigenpairs.eigenvector_matrix.assign(coefficients.data(),
        coefficients.data()+coefficients.size());
    full_->ci_covector_to_structure(input.view(),output.view());
    result.overlap_eigenvectors.resize(n,roots);
    copy_to_host(*context_,result.overlap_eigenvectors,output.view());
    copy_to_device(*context_,input.view(),result.hamiltonian_eigenvectors);
    full_->ci_covector_to_structure(input.view(),output.view());
    copy_to_host(*context_,result.hamiltonian_eigenvectors,output.view());
  }
  return result;
}

}  // namespace xmvb::cuda
