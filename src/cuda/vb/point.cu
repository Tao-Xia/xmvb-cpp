#include "cuda/vb/point.hpp"

#include <cmath>
#include <stdexcept>

#include <nvtx3/nvtx3.hpp>

#include "cuda/integrals/one_electron.hpp"
#include "cuda/linalg/blas.hpp"
#include "cuda/vb/diagonal.hpp"

namespace xmvb::cuda {
namespace {
using View=linalg::ConstMatrixView;
using Out=linalg::MatrixView;

template<class Emit>
void device_scalars(Context& c,Emit emit) {
  check_cublas(cublasSetPointerMode(c.blas(),CUBLAS_POINTER_MODE_DEVICE),
      "point device scalar mode");
  try { emit(); }
  catch (...) { cublasSetPointerMode(c.blas(),CUBLAS_POINTER_MODE_HOST);throw; }
  check_cublas(cublasSetPointerMode(c.blas(),CUBLAS_POINTER_MODE_HOST),
      "point restore scalar mode");
}

// First four entries are device dot products. The last four are BLAS
// coefficients, including every finite-CI-residual normalization term.
__global__ void hvp_coefficients(double* values,double norm,double energy) {
  if (blockIdx.x || threadIdx.x) return;
  const double dn=2.0*values[0];
  const double de=2.0*values[1]/norm+values[2]+values[3]-energy*dn/norm;
  values[4]=2.0/norm;values[5]=-dn/norm;
  values[6]=-de;values[7]=-energy;
}

void copy(Context& context,Out out,View in) {
  if (out.rows!=in.rows || out.cols!=in.cols)
    throw std::invalid_argument("CUDA point copy dimensions");
  check_cuda(cudaMemcpy2DAsync(out.data,out.ld*sizeof(double),in.data,in.ld*sizeof(double),
      in.rows*sizeof(double),in.cols,cudaMemcpyDeviceToDevice,context.stream()),"point copy");
}
void scale(Context& context,Out out,double value) {
  if (out.ld==out.rows) {
    check_cublas(cublasDscal(context.blas(),blas_int(out.rows*out.cols),&value,
        out.data,1),"point scale");
    return;
  }
  for (std::int64_t j=0;j<out.cols;++j)
    check_cublas(cublasDscal(context.blas(),blas_int(out.rows),&value,
        out.column(j).data,1),"point scale");
}
void add(Context& context,Out out,View in,double value=1.0) {
  if (out.rows!=in.rows || out.cols!=in.cols)
    throw std::invalid_argument("CUDA point sum dimensions");
  if (out.ld==out.rows && in.ld==in.rows) {
    check_cublas(cublasDaxpy(context.blas(),blas_int(out.rows*out.cols),&value,
        in.data,1,out.data,1),"point sum");
    return;
  }
  for (std::int64_t j=0;j<out.cols;++j)
    check_cublas(cublasDaxpy(context.blas(),blas_int(out.rows),&value,
        in.column(j).data,1,out.column(j).data,1),"point sum");
}
double dot(Context& context,View x,View y) {
  if (x.rows!=y.rows || x.cols!=y.cols)
    throw std::invalid_argument("CUDA point dot dimensions");
  double result=0.0;
  if (x.ld==x.rows && y.ld==y.rows) {
    check_cublas(cublasDdot(context.blas(),blas_int(x.rows*x.cols),x.data,1,
        y.data,1,&result),"point dot");
    return result;
  }
  for (std::int64_t j=0;j<x.cols;++j) {
    double column=0.0;
    check_cublas(cublasDdot(context.blas(),blas_int(x.rows),x.column(j).data,1,
        y.column(j).data,1,&column),"point dot");
    result+=column;
  }
  return result;
}
Matrix<double> product(Context& c,View a,View b,Operation oa=Operation::none,
                       Operation ob=Operation::none) {
  Matrix<double> result(c,oa==Operation::none?a.rows:a.cols,
                          ob==Operation::none?b.cols:b.rows);
  gemm(c,result.view(),a,b,1.0,0.0,oa,ob);
  return result;
}

// vec(G(V C^T+C V^T) T) = R(C,T) vec(V). Owned factors are contiguous.
void density_projection(Context& c,View response,View factor,Out output,
    Operation operation=Operation::none) {
  gemm(c,{output.data,output.rows*output.cols,1,output.rows*output.cols},response,
      {factor.data,factor.rows*factor.cols,1,factor.rows*factor.cols},1.0,0.0,operation);
}

Matrix<double> active_density_factor(Context& c,View f,View df,View a,View da) {
  Matrix<double> v(c,f.rows,f.cols);
  gemm(c,v.view(),df,a,0.5);
  gemm(c,v.view(),df,a,0.5,1.0,Operation::none,Operation::transpose);
  gemm(c,v.view(),f,da,0.25,1.0);
  gemm(c,v.view(),f,da,0.25,1.0,Operation::none,Operation::transpose);
  return v;
}
void check_structure(View c,int n) {
  if (c.rows!=n || c.cols!=1 || c.ld<c.rows || !c.data)
    throw std::invalid_argument("CUDA point requires one selected-structure state");
}
}  // namespace

VbPoint::VbPoint(Context& context,const vb::OrbitalPreparationInput& input,
    StructureAction& action,const ExactAoPairOperator& integrals,
    View core,double nuclear)
    : context_(&context),action_(&action),integrals_(&integrals),core_(core),nuclear_(nuclear),
      geometry_(context,input),sigma_(context,action.carrier_sigma()),
      direction_sigma_(context,sigma_),
      effective_(context,input.n_basis_functions,input.n_basis_functions),
      one_(context,input.n_active_orbitals,input.n_active_orbitals),
      pair_(context,input.n_active_orbitals*(input.n_active_orbitals+1)/2,
                    input.n_active_orbitals*(input.n_active_orbitals+1)/2),
      state_(context,action.n_alpha_strings(),action.n_beta_strings()),
      image_(context,state_.rows(),state_.cols()),
      carrier_gradient_(context,state_.rows(),state_.cols()),
      one_adjoint_(context,one_.rows(),one_.cols()),
      pair_adjoint_(context,pair_.rows(),pair_.cols()),
      frame_gradient_(context,one_.rows(),one_.cols()),
      frame_moment_(context,one_.rows(),one_.cols()),
      active_gradient_(context,input.n_basis_functions,input.n_active_orbitals),
      density_gradient_(context,input.n_basis_functions,input.n_basis_functions),
      carrier_scratch_(context,state_.rows(),state_.cols()) {
  if (core.rows!=input.n_basis_functions || core.cols!=core.rows ||
      integrals.basis_functions()!=core.rows || !std::isfinite(nuclear))
    throw std::invalid_argument("CUDA point AO dimensions");
}

void VbPoint::select_frame() const {
  // An accepted point is immutable between prepare() calls. Only a different
  // point/transport frame invalidates its binding to the shared action.
  if (prepared_ && selected_action_version_==action_->version()) return;
  action_->update_integrals(one_.view(),pair_.view(),geometry_.frame_transform());
  selected_action_version_=action_->version();
}

double VbPoint::evaluate(View raw,View coefficients) {
  prepare(raw);
  return evaluate_coefficients(coefficients);
}

void VbPoint::prepare(View raw) {
  const nvtx3::scoped_range range{"point.prepare"};
  ready_=false;
  prepared_=false;
  exchange_products_={};
  inactive_response_={};active_response_={};overlap_core_={};core_basis_={};core_to_dual_={};
  auto& c=*context_;
  geometry_.set_raw(raw);geometry_.prepare();
  Matrix<double> core_energy(c,1,1);
  integrals_->core_energy(geometry_.inactive_density(),core_,effective_.view(),core_energy.data());
  transform_one_electron(c,effective_.view(),geometry_.orthogonal_active(),one_.view());
  pair_products_={};
  pair_products_=integrals_->reserve_products(one_.rows());
  integrals_->forward(geometry_.orthogonal_active(),pair_.view(),
      pair_products_.empty() ? PairPanel{}:
          PairPanel{pair_products_.data(),integrals_->pair_count(),pair_.rows(),pair_.rows()});
  sigma_.update_integrals(one_.view(),pair_.view());
  select_frame();
  c.copy_to_host(&core_energy_,core_energy.data(),sizeof(double));
  prepared_=true;
}

Matrix<double> VbPoint::hessian_diagonal(View mo) {
  if (!ready_) throw std::logic_error("CUDA exchange preparation requires an evaluated point");
  const auto f=geometry_.orthogonal_active(),dual=geometry_.inactive_dual();
  const auto core=mo.columns(0,dual.cols);
  if (!pair_products_.empty() && exchange_products_.empty())
    exchange_products_=integrals_->prepare_exchange(f,&active_response_,core);
  if (dual.cols) {
    core_basis_=Matrix<double>(*context_,core.rows,core.cols);
    copy(*context_,core_basis_.view(),core);
    overlap_core_=product(*context_,geometry_.ao_overlap(),core);
    core_to_dual_=product(*context_,overlap_core_.view(),dual,Operation::transpose);
  }
  auto diagonal=build_analytic_rotation_diagonal(*context_,*integrals_,mo,core_,
      effective_.view(),one_adjoint_.view(),pair_adjoint_.view(),pair_.view(),
      geometry_.inactive_count(),pair_products(),exchange_products(),
      active_response_.data() ? &inactive_response_:nullptr);
  return diagonal;
}

double VbPoint::evaluate_coefficients(View coefficients) {
  const nvtx3::scoped_range range{"point.energy_gradient"};
  if (!prepared_) throw std::logic_error("CUDA point coefficients require preparation");
  check_structure(coefficients,action_->size());
  ready_=false;
  auto& c=*context_;
  select_frame();
  action_->expand_spin(coefficients,state_.view());
  action_->apply_orbital_transform(state_.view());
  // Cloned sigma instances share stream-local scratch. The CI graph recaptures
  // only when preparing a different scratch layout changes its addresses.
  sigma_.apply_with_adjoint(state_.view(),state_.view(),image_.view(),
      one_adjoint_.view(),pair_adjoint_.view());
  norm_=dot(c,state_.view(),state_.view());
  if (!(norm_>0.0) || !std::isfinite(norm_))
    throw std::runtime_error("CUDA VB state has invalid norm");
  active_energy_=dot(c,state_.view(),image_.view())/norm_;
  total_energy_=core_energy_+active_energy_+nuclear_;
  scale(c,one_adjoint_.view(),1.0/norm_);
  scale(c,pair_adjoint_.view(),1.0/norm_);
  copy(c,carrier_gradient_.view(),image_.view());
  add(c,carrier_gradient_.view(),state_.view(),-active_energy_);
  scale(c,carrier_gradient_.view(),2.0/norm_);
  sigma_.one_body_generator_adjoint(carrier_gradient_.view(),state_.view(),frame_moment_.view());
  gemm(c,frame_gradient_.view(),frame_moment_.view(),geometry_.frame_inverse_transform(),
      1.0,0.0,Operation::none,Operation::transpose);
  integrals_->adjoint(geometry_.orthogonal_active(),pair_adjoint_.view(),
      active_gradient_.view(),pair_products());
  Matrix<double> one_gradient(c,active_gradient_.rows(),active_gradient_.cols());
  transform_one_electron_adjoint(c,effective_.view(),geometry_.orthogonal_active(),
      one_adjoint_.view(),one_gradient.view());
  add(c,active_gradient_.view(),one_gradient.view());
  integrals_->active_core_density_adjoint(geometry_.orthogonal_active(),
      one_adjoint_.view(),density_gradient_.view(),pair_products(),exchange_products());
  add(c,density_gradient_.view(),effective_.view(),2.0);
  ready_=true;
  return total_energy_;
}

void VbPoint::project_carrier(View carrier,Out output) const {
  copy(*context_,carrier_scratch_.view(),carrier);
  action_->apply_adjoint_orbital_transform(carrier_scratch_.view());
  action_->contract_spin(carrier_scratch_.view(),output);
}

void VbPoint::gradient(Out orbital,Out structure) const {
  if (!ready_) throw std::logic_error("CUDA point gradient requires evaluation");
  select_frame();
  geometry_.pullback({},density_gradient_.view(),{},frame_gradient_.view(),
      active_gradient_.view(),orbital);
  project_carrier(carrier_gradient_.view(),structure);
}

void VbPoint::hessian(View orbital,View structure,Out orbital_image,Out structure_image) const {
  const nvtx3::scoped_range range{"point.hvp"};
  if (!ready_) throw std::logic_error("CUDA point HVP requires evaluation");
  check_structure(structure,action_->size());
  auto& c=*context_;
  select_frame();
  const auto d=geometry_.direction(orbital);
  auto generator=product(c,d.frame_transform.view(),geometry_.frame_inverse_transform());
  Matrix<double> dy(c,state_.rows(),state_.cols()),dc(c,state_.rows(),state_.cols());
  action_->expand_spin(structure,dy.view());
  action_->apply_orbital_transform(dy.view());
  sigma_.apply_generator(state_.view(),generator.view(),dc.view());
  add(c,dy.view(),dc.view());
  const int nc=geometry_.inactive_count(),na=geometry_.active_count(),nb=geometry_.basis_size();
  Matrix<double> dheff,dheff_active(c,nb,na),inactive_projection;
  if (inactive_response_.data()) {
    Matrix<double> factor(c,nb,nc);
    // For the occupied projector P=C C^T, dP=dP S P+P S dP;
    // V=dP S C therefore gives dP=V C^T+C V^T without a gauge choice.
    gemm(c,factor.view(),d.inactive_density.view(),overlap_core_.view());
    inactive_projection=Matrix<double>(c,nb,nc);
    const double one=1.0,zero=0.0;
    check_cublas(cublasDspmv(c.blas(),CUBLAS_FILL_MODE_UPPER,blas_int(nb*nc),&one,
        inactive_response_.data(),factor.data(),1,&zero,inactive_projection.data(),1),
        "CUDA symmetric core density projection");
    // Self-adjoint G makes the reverse cross response the transpose of the
    // active-density response. No second cross image or transform exists.
    density_projection(c,active_response_.view(),factor.view(),dheff_active.view(),
        Operation::transpose);
  } else if (nc) {
    dheff=Matrix<double>(c,nb,nb);
    integrals_->contract_jk(d.inactive_density.view(),dheff.view());
    gemm(c,dheff_active.view(),dheff.view(),geometry_.orthogonal_active());
  } else {
    check_cuda(cudaMemsetAsync(dheff_active.data(),0,nb*na*sizeof(double),c.stream()),
        "zero empty core response");
  }
  Matrix<double> dh(c,one_.rows(),one_.cols()),dg(c,pair_.rows(),pair_.cols());
  transform_one_electron_directional(c,effective_.view(),geometry_.orthogonal_active(),
      d.orthogonal_active.view(),dh.view());
  gemm(c,dh.view(),geometry_.orthogonal_active(),dheff_active.view(),
      1.0,1.0,Operation::transpose);
  integrals_->directional(geometry_.orthogonal_active(),d.orthogonal_active.view(),
      dg.view(),pair_products());
  direction_sigma_.update_integrals(dh.view(),dg.view());
  Matrix<double> dhy(c,state_.rows(),state_.cols());
  Matrix<double> coefficients(c,8,1);
  // The accepted normalized integral adjoints already represent Y^T dH Y/N.
  // Only H*dy+dH*Y is consumed below, so no separate directional carrier or
  // second sigma gather is mathematically required.
  device_scalars(c,[&] {
    const auto inner=[&](View a,View b,int index) {
      check_cublas(cublasDdot(c.blas(),blas_int(a.rows*a.cols),a.data,1,b.data,1,
          coefficients.data()+index),"point HVP device dot");
    };
    inner(state_.view(),dy.view(),0);inner(dy.view(),image_.view(),1);
    inner(one_adjoint_.view(),dh.view(),2);inner(pair_adjoint_.view(),dg.view(),3);
  });
  hvp_coefficients<<<1,1,0,c.stream()>>>(coefficients.data(),norm_,active_energy_);
  check_cuda(cudaGetLastError(),"point HVP normalization coefficients");
  Matrix<double> dha(c,one_.rows(),one_.cols()),dga(c,pair_.rows(),pair_.cols());
  sigma_.apply_direction_with_adjoint(dy.view(),state_.view(),direction_sigma_,
      dhy.view(),dha.view(),dga.view());
  device_scalars(c,[&] {
    const auto scaled=[&](Out x,int coefficient) {
      check_cublas(cublasDscal(c.blas(),blas_int(x.rows*x.cols),
          coefficients.data()+coefficient,x.data,1),"point HVP device scale");
    };
    const auto added=[&](Out x,View y,int coefficient) {
      check_cublas(cublasDaxpy(c.blas(),blas_int(x.rows*x.cols),
          coefficients.data()+coefficient,y.data,1,x.data,1),"point HVP device sum");
    };
    scaled(dha.view(),4);added(dha.view(),one_adjoint_.view(),5);
    scaled(dga.view(),4);added(dga.view(),pair_adjoint_.view(),5);
    added(dhy.view(),state_.view(),6);added(dhy.view(),dy.view(),7);
    scaled(dhy.view(),4);added(dhy.view(),carrier_gradient_.view(),5);
  });
  // dhy now denotes the exact directional carrier gradient.
  Matrix<double> moment(c,one_.rows(),one_.cols()),other(c,one_.rows(),one_.cols());
  sigma_.one_body_generator_adjoint(dhy.view(),state_.view(),moment.view());
  sigma_.one_body_generator_adjoint(carrier_gradient_.view(),dy.view(),other.view());
  add(c,moment.view(),other.view());
  auto dinverse=product(c,geometry_.frame_inverse_transform(),generator.view());
  scale(c,dinverse.view(),-1.0);
  auto dframe_gradient=product(c,moment.view(),geometry_.frame_inverse_transform(),
      Operation::none,Operation::transpose);
  auto frame_extra=product(c,frame_moment_.view(),dinverse.view(),
      Operation::none,Operation::transpose);
  add(c,dframe_gradient.view(),frame_extra.view());
  Matrix<double> dfgrad(c,active_gradient_.rows(),active_gradient_.cols());
  Matrix<double> one_extra(c,active_gradient_.rows(),active_gradient_.cols());
  integrals_->directional_adjoint(geometry_.orthogonal_active(),d.orthogonal_active.view(),
      pair_adjoint_.view(),dga.view(),dfgrad.view(),pair_products(),exchange_products());
  transform_one_electron_directional_adjoint(c,effective_.view(),geometry_.orthogonal_active(),
      d.orthogonal_active.view(),one_adjoint_.view(),dha.view(),one_extra.view());
  add(c,dfgrad.view(),one_extra.view());
  gemm(c,one_extra.view(),dheff_active.view(),one_adjoint_.view());
  gemm(c,one_extra.view(),dheff_active.view(),one_adjoint_.view(),1.0,1.0,
      Operation::none,Operation::transpose);
  add(c,dfgrad.view(),one_extra.view());
  Matrix<double> dpgrad,projected_density;
  if (nc) {
    projected_density=Matrix<double>(c,nb,nc);
    if (active_response_.data()) {
      auto factor=active_density_factor(c,geometry_.orthogonal_active(),
          d.orthogonal_active.view(),one_adjoint_.view(),dha.view());
      density_projection(c,active_response_.view(),factor.view(),projected_density.view());
      if (inactive_projection.data())
        add(c,projected_density.view(),inactive_projection.view(),2.0);
      else
        gemm(c,projected_density.view(),dheff.view(),core_basis_.view(),2.0,1.0);
      Matrix<double> dual_projection(c,nb,nc);
      gemm(c,dual_projection.view(),projected_density.view(),core_to_dual_.view());
      projected_density=std::move(dual_projection);
    } else {
      dpgrad=Matrix<double>(c,nb,nb);
      integrals_->active_core_density_directional_adjoint(geometry_.orthogonal_active(),
          d.orthogonal_active.view(),one_adjoint_.view(),dha.view(),dpgrad.view());
      gemm(c,projected_density.view(),dpgrad.view(),geometry_.inactive_dual());
      gemm(c,projected_density.view(),dheff.view(),geometry_.inactive_dual(),2.0,1.0);
    }
    // OrbitalGeometry consumes (dgp+dgp^T)*dual. Both exact G images are symmetric.
    scale(c,projected_density.view(),2.0);
  }
  geometry_.pullback_direction(d,{},density_gradient_.view(),{},frame_gradient_.view(),
      active_gradient_.view(),{},{},{},dframe_gradient.view(),dfgrad.view(),orbital_image,
      projected_density.view());
  // d(T^T gY)=T^T[dgY+E(dR R^-1)^T gY], without a second H/M action.
  Matrix<double> transpose(c,generator.rows(),generator.cols());
  constexpr double one=1.0,zero=0.0;
  check_cublas(cublasDgeam(c.blas(),CUBLAS_OP_T,CUBLAS_OP_T,
      blas_int(generator.cols()),blas_int(generator.rows()),&one,generator.data(),
      blas_int(generator.rows()),&zero,generator.data(),blas_int(generator.rows()),
      transpose.data(),blas_int(transpose.rows())),"point generator transpose");
  sigma_.apply_generator(carrier_gradient_.view(),transpose.view(),dc.view());
  add(c,dhy.view(),dc.view());
  project_carrier(dhy.view(),structure_image);
}

}  // namespace xmvb::cuda
