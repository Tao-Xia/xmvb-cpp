#include "cuda/vb/point.hpp"

#include <cmath>
#include <stdexcept>

#include <nvtx3/nvtx3.hpp>

#include "cuda/integrals/one_electron.hpp"
#include "cuda/linalg/blas.hpp"

namespace xmvb::cuda {
namespace {
using View=linalg::ConstMatrixView;
using Out=linalg::MatrixView;

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

void VbPoint::prepare_exchange() {
  if (!ready_) throw std::logic_error("CUDA exchange preparation requires an evaluated point");
  if (pair_products_.empty() || !exchange_products_.empty()) return;
  exchange_products_=integrals_->prepare_exchange(geometry_.orthogonal_active());
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
  // Prepare the fused mode once. Later ordinary actions on this sigma retain
  // these addresses; the separate CI graph owns an independent workspace.
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
      one_adjoint_.view(),density_gradient_.view());
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
  Matrix<double> dheff(c,effective_.rows(),effective_.cols());
  integrals_->contract_jk(d.inactive_density.view(),dheff.view());
  Matrix<double> dh(c,one_.rows(),one_.cols()),dg(c,pair_.rows(),pair_.cols());
  transform_one_electron_directional(c,effective_.view(),geometry_.orthogonal_active(),
      d.orthogonal_active.view(),dh.view());
  Matrix<double> extra_h(c,one_.rows(),one_.cols());
  transform_one_electron(c,dheff.view(),geometry_.orthogonal_active(),extra_h.view());
  add(c,dh.view(),extra_h.view());
  integrals_->directional(geometry_.orthogonal_active(),d.orthogonal_active.view(),
      dg.view(),pair_products());
  direction_sigma_.update_integrals(dh.view(),dg.view());
  Matrix<double> dhy(c,state_.rows(),state_.cols());
  const double dn=2.0*dot(c,state_.view(),dy.view());
  // The accepted normalized integral adjoints already represent Y^T dH Y/N.
  // Only H*dy+dH*Y is consumed below, so no separate directional carrier or
  // second sigma gather is mathematically required.
  const double de=2.0*dot(c,dy.view(),image_.view())/norm_+
      dot(c,one_adjoint_.view(),dh.view())+dot(c,pair_adjoint_.view(),dg.view())-
      active_energy_*dn/norm_;
  Matrix<double> dha(c,one_.rows(),one_.cols()),dga(c,pair_.rows(),pair_.cols());
  sigma_.apply_direction_with_adjoint(dy.view(),state_.view(),direction_sigma_,
      dhy.view(),dha.view(),dga.view());
  scale(c,dha.view(),2.0/norm_);add(c,dha.view(),one_adjoint_.view(),-dn/norm_);
  scale(c,dga.view(),2.0/norm_);add(c,dga.view(),pair_adjoint_.view(),-dn/norm_);
  add(c,dhy.view(),state_.view(),-de);
  add(c,dhy.view(),dy.view(),-active_energy_);
  scale(c,dhy.view(),2.0/norm_);
  add(c,dhy.view(),carrier_gradient_.view(),-dn/norm_);
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
  transform_one_electron_adjoint(c,dheff.view(),geometry_.orthogonal_active(),
      one_adjoint_.view(),one_extra.view());
  add(c,dfgrad.view(),one_extra.view());
  Matrix<double> dpgrad(c,density_gradient_.rows(),density_gradient_.cols());
  integrals_->active_core_density_directional_adjoint(geometry_.orthogonal_active(),
      d.orthogonal_active.view(),one_adjoint_.view(),dha.view(),dpgrad.view());
  add(c,dpgrad.view(),dheff.view(),2.0);
  geometry_.pullback_direction(d,{},density_gradient_.view(),{},frame_gradient_.view(),
      active_gradient_.view(),{},dpgrad.view(),{},dframe_gradient.view(),dfgrad.view(),orbital_image);
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
