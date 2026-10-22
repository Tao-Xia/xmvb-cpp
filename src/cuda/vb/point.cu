#include "cuda/vb/point.hpp"

#include <array>
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
    View core,View overlap,double nuclear)
    : context_(&context),action_(&action),integrals_(&integrals),core_(core),nuclear_(nuclear),
      geometry_(context,input,overlap),sigma_(context,action.carrier_sigma()),
      direction_sigma_(context,sigma_),
      effective_(context,input.n_basis_functions,input.n_basis_functions),
      one_(context,input.n_active_orbitals,input.n_active_orbitals),
      pair_(context,input.n_active_orbitals*(input.n_active_orbitals+1)/2,
                    input.n_active_orbitals*(input.n_active_orbitals+1)/2),
      endpoint_{Matrix<double>(context,action.size(),1)},
      products_(std::make_unique<ProductContraction>(context,action.orthogonal_view(),
          action.alpha_transform(),action.beta_transform(),action.orthogonal_view().support)),
      selected_gradient_(context,action.size(),1),
      one_adjoint_(context,one_.rows(),one_.cols()),
      pair_adjoint_(context,pair_.rows(),pair_.cols()),
      frame_gradient_(context,one_.rows(),one_.cols()),
      frame_moment_(context,one_.rows(),one_.cols()),
      active_gradient_(context,input.n_basis_functions,input.n_active_orbitals),
      density_gradient_(context,input.n_basis_functions,input.n_basis_functions) {
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
  products_->update_frame();
  endpoint_.images.reset();
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
  ExactAoResponses prepared;
  if (!pair_products_.empty() && exchange_products_.empty()) {
    prepared=integrals_->prepare_responses(f,mo,geometry_.inactive_count());
    exchange_products_=std::move(prepared.exchange);
    active_response_=std::move(prepared.active);
    inactive_response_=std::move(prepared.inactive);
  }
  if (dual.cols) {
    core_basis_=Matrix<double>(*context_,core.rows,core.cols);
    copy(*context_,core_basis_.view(),core);
    overlap_core_=product(*context_,geometry_.ao_overlap(),core);
    core_to_dual_=product(*context_,overlap_core_.view(),dual,Operation::transpose);
  }
  auto diagonal=build_analytic_rotation_diagonal(*context_,*integrals_,mo,core_,
      effective_.view(),one_adjoint_.view(),pair_adjoint_.view(),pair_.view(),
      geometry_.inactive_count(),pair_products(),exchange_products(),
      active_response_.data() && !prepared.core_ready ? &inactive_response_:nullptr,
      prepared.core_ready ? CoreDiagonalView{prepared.core_coulomb.view(),
          prepared.core_response.view()}:CoreDiagonalView{});
  return diagonal;
}

double VbPoint::evaluate_coefficients(View coefficients) {
  const nvtx3::scoped_range range{"point.energy_gradient"};
  if (!prepared_) throw std::logic_error("CUDA point coefficients require preparation");
  check_structure(coefficients,action_->size());
  ready_=false;
  auto& c=*context_;
  endpoint_.images.reset();
  select_frame();
  // The completed CI action may still own large channel slabs. Release them
  // before pricing the accepted tile geometry, not after that geometry has
  // already been frozen for every subsequent HVP at this orbital point.
  sigma_.reset_workspace();
  copy(c,endpoint_.coefficients.view(),coefficients);
  const ProductIncidence incidence{action_->term_offsets_data(),action_->term_structures_data(),
      action_->term_weights_data(),action_->size()};
  auto jet=products_->jet(incidence,sigma_,endpoint_,geometry_.frame_transform(),{},{},nullptr,
      product_workspace_bytes());
  selected_h_=std::move(jet.hamiltonian);selected_m_=std::move(jet.overlap);
  one_adjoint_=std::move(jet.one);pair_adjoint_=std::move(jet.pair);
  frame_moment_=std::move(jet.generator);
  norm_=dot(c,coefficients,selected_m_.view());
  if (!(norm_>0.0) || !std::isfinite(norm_))
    throw std::runtime_error("CUDA VB state has invalid norm");
  active_energy_=dot(c,coefficients,selected_h_.view())/norm_;
  products_->accept(endpoint_,active_energy_);
  total_energy_=core_energy_+active_energy_+nuclear_;
  scale(c,one_adjoint_.view(),1.0/norm_);
  scale(c,pair_adjoint_.view(),1.0/norm_);
  scale(c,frame_moment_.view(),1.0/norm_);
  add_upper(c,frame_moment_.view(),one_adjoint_.view(),-2.0*active_energy_);
  copy(c,selected_gradient_.view(),selected_h_.view());
  add(c,selected_gradient_.view(),selected_m_.view(),-active_energy_);
  scale(c,selected_gradient_.view(),2.0/norm_);
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

std::size_t VbPoint::product_workspace_bytes() const {
  return context_->device_byte_budget()-context_->counters().current_device_bytes;
}

void VbPoint::gradient(Out orbital,Out structure) const {
  if (!ready_) throw std::logic_error("CUDA point gradient requires evaluation");
  if (orbital.data)
    geometry_.pullback({},density_gradient_.view(),{},frame_gradient_.view(),
        active_gradient_.view(),orbital);
  if (structure.data) {
    copy(*context_,structure,selected_gradient_.view());
  }
}

void VbPoint::hessian(View orbital,View structure,Out orbital_image,Out structure_image,
    Out ci_orbital_image,Out ci_structure_image) const {
  const nvtx3::scoped_range range{"point.hvp"};
  if (!ready_) throw std::logic_error("CUDA point HVP requires evaluation");
  if (static_cast<bool>(ci_orbital_image.data)!=static_cast<bool>(ci_structure_image.data))
    throw std::invalid_argument("both CI column images must be supplied together");
  check_structure(structure,action_->size());
  auto& c=*context_;
  select_frame();
  sigma_.reset_workspace();
  const auto d=geometry_.direction(orbital);
  auto generator=product(c,d.frame_transform.view(),geometry_.frame_inverse_transform());
  const bool split=ci_orbital_image.data && ci_structure_image.data;
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
  const ProductIncidence incidence{action_->term_offsets_data(),action_->term_structures_data(),
      action_->term_weights_data(),action_->size()};
  ProductJet ci_jet;
  auto jet=products_->jet(incidence,sigma_,endpoint_,geometry_.frame_transform(),structure,
      d.frame_transform.view(),&direction_sigma_,product_workspace_bytes(),split ? &ci_jet:nullptr,generator.view());
  const auto normalize=[&](ProductJet& value,Out selected) {
    const double dn=dot(c,structure,selected_m_.view())+
        dot(c,endpoint_.coefficients.view(),value.overlap.view());
    const double de=(dot(c,structure,selected_h_.view())+
        dot(c,endpoint_.coefficients.view(),value.hamiltonian.view())-active_energy_*dn)/norm_;
    scale(c,value.one.view(),2.0/norm_);add(c,value.one.view(),one_adjoint_.view(),-dn/norm_);
    scale(c,value.pair.view(),2.0/norm_);add(c,value.pair.view(),pair_adjoint_.view(),-dn/norm_);
    scale(c,value.generator.view(),1.0/norm_);
    add(c,value.generator.view(),frame_moment_.view(),-dn/norm_);
    add_upper(c,value.generator.view(),one_adjoint_.view(),-2.0*de);
    copy(c,selected,value.hamiltonian.view());add(c,selected,value.overlap.view(),-active_energy_);
    add(c,selected,selected_m_.view(),-de);scale(c,selected,2.0/norm_);
    add(c,selected,selected_gradient_.view(),-dn/norm_);
  };
  normalize(jet,structure_image);
  if (split) normalize(ci_jet,ci_structure_image);
  auto dha=std::move(jet.one),dga=std::move(jet.pair),moment=std::move(jet.generator);
  auto ci_dha=std::move(ci_jet.one),ci_dga=std::move(ci_jet.pair);
  Matrix<double> ci_frame;
  if (split) ci_frame=product(c,ci_jet.generator.view(),geometry_.frame_inverse_transform(),
      Operation::none,Operation::transpose);
  auto dinverse=product(c,geometry_.frame_inverse_transform(),generator.view());
  scale(c,dinverse.view(),-1.0);
  auto dframe_gradient=product(c,moment.view(),geometry_.frame_inverse_transform(),
      Operation::none,Operation::transpose);
  auto frame_extra=product(c,frame_moment_.view(),dinverse.view(),
      Operation::none,Operation::transpose);
  add(c,dframe_gradient.view(),frame_extra.view());
  Matrix<double> dfgrad(c,active_gradient_.rows(),active_gradient_.cols());
  Matrix<double> one_extra(c,active_gradient_.rows(),active_gradient_.cols());
  Matrix<double> ci_dfgrad,ci_one;
  if (split) { ci_dfgrad=Matrix<double>(c,nb,na);ci_one=Matrix<double>(c,nb,na); }
  integrals_->directional_adjoint(geometry_.orthogonal_active(),d.orthogonal_active.view(),
      pair_adjoint_.view(),dga.view(),dfgrad.view(),pair_products(),exchange_products(),
      ci_dga.view(),ci_dfgrad.view());
  transform_one_electron_directional_adjoint(c,effective_.view(),geometry_.orthogonal_active(),
      d.orthogonal_active.view(),one_adjoint_.view(),dha.view(),one_extra.view(),
      ci_dha.view(),ci_one.view());
  add(c,dfgrad.view(),one_extra.view());
  if (split) add(c,ci_dfgrad.view(),ci_one.view());
  gemm(c,one_extra.view(),dheff_active.view(),one_adjoint_.view());
  gemm(c,one_extra.view(),dheff_active.view(),one_adjoint_.view(),1.0,1.0,
      Operation::none,Operation::transpose);
  add(c,dfgrad.view(),one_extra.view());
  Matrix<double> dpgrad,projected_density,ci_projected_density;
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
      if (split) {
        Matrix<double> factor_ci(c,nb,na);
        gemm(c,factor_ci.view(),geometry_.orthogonal_active(),ci_dha.view(),0.25);
        gemm(c,factor_ci.view(),geometry_.orthogonal_active(),ci_dha.view(),0.25,1.0,
            Operation::none,Operation::transpose);
        Matrix<double> core_projection(c,nb,nc);
        density_projection(c,active_response_.view(),factor_ci.view(),core_projection.view());
        ci_projected_density=Matrix<double>(c,nb,nc);
        gemm(c,ci_projected_density.view(),core_projection.view(),core_to_dual_.view(),2.0);
      }
    } else {
      dpgrad=Matrix<double>(c,nb,nb);
      Matrix<double> ci_dpgrad;
      if (split) ci_dpgrad=Matrix<double>(c,nb,nb);
      integrals_->active_core_density_directional_adjoint(geometry_.orthogonal_active(),
          d.orthogonal_active.view(),one_adjoint_.view(),dha.view(),dpgrad.view(),
          ci_dha.view(),ci_dpgrad.view());
      gemm(c,projected_density.view(),dpgrad.view(),geometry_.inactive_dual());
      gemm(c,projected_density.view(),dheff.view(),geometry_.inactive_dual(),2.0,1.0);
      if (split) {
        ci_projected_density=Matrix<double>(c,nb,nc);
        gemm(c,ci_projected_density.view(),ci_dpgrad.view(),geometry_.inactive_dual(),2.0);
      }
    }
    // OrbitalGeometry consumes (dgp+dgp^T)*dual. Both exact G images are symmetric.
    scale(c,projected_density.view(),2.0);
  }
  geometry_.pullback_direction(d,{},density_gradient_.view(),{},frame_gradient_.view(),
      active_gradient_.view(),{},{},{},dframe_gradient.view(),dfgrad.view(),orbital_image,
      projected_density.view());
  if (split) {
    geometry_.pullback({},{},{},ci_frame.view(),ci_dfgrad.view(),ci_orbital_image,
        ci_projected_density.view());
  }
}

}  // namespace xmvb::cuda
