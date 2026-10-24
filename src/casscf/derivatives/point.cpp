#include "casscf/derivatives/point.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/pair_index.hpp"
#include "integrals/active/two_electron.hpp"
#include "integrals/ao/exact.hpp"
#include "integrals/response/adjoint.hpp"
#include "integrals/response/directional.hpp"
#include "parallel/openmp.hpp"

namespace xmvb::casscf {
namespace {

int alpha_count(int nactive, int electrons, int spin) {
  if (nactive < 1 || nactive > 63 || electrons < 1 || electrons > 2*nactive ||
      spin < 0 || spin > std::min(electrons, 2*nactive-electrons) ||
      (electrons-spin)%2 != 0)
    throw std::invalid_argument("invalid complete-CAS electron and spin counts");
  return (electrons+spin)/2;
}


Eigen::MatrixXd ao_response(const integrals::AoIntegralInput& ao,
    const Eigen::MatrixXd& density) {
  const int n=ao.n_basis_functions;
  if (density.rows()!=n || density.cols()!=n)
    throw std::invalid_argument("AO density response shape mismatch");
  const Eigen::Map<const Eigen::MatrixXd> source(density.data(),n*n,1);
  integrals::AoH1eBatchWorkspace workspace;
  Eigen::MatrixXd result;
  integrals::apply_symmetric_ao_h1e_batch(
      source,ao,effective_openmp_thread_count(),&workspace,&result);
  return Eigen::Map<const Eigen::MatrixXd>(result.data(),n,n);
}


std::vector<double> pack_pair_adjoint(const Eigen::MatrixXd& adjoint) {
  if (adjoint.rows()!=adjoint.cols())
    throw std::invalid_argument("pair-kernel cotangent must be square");
  const int pairs=static_cast<int>(adjoint.rows());
  std::vector<double> packed(static_cast<std::size_t>(pairs)*(pairs+1)/2);
  for (int p=0; p<pairs; ++p)
    for (int q=0; q<=p; ++q)
      packed[core::PairIndex::packed_pair_index(p,q)]=
          p==q ? adjoint(p,q) : adjoint(p,q)+adjoint(q,p);
  return packed;
}

}  // namespace

FixedCiPoint::FixedCiPoint(const integrals::AoIntegralInput& ao,
    const Eigen::MatrixXd& orbitals, int ncore, int nactive,
    double nuclear, int electrons, int spin, const Eigen::MatrixXd& coefficients)
    : ao_(&ao), orbitals_(orbitals), ncore_(ncore), nactive_(nactive),
      twice_spin_(spin),
      spin_space_(nactive,alpha_count(nactive,electrons,spin),
          electrons-alpha_count(nactive,electrons,spin)),
      prepared_(prepare_active_integrals(ao,orbitals_,ncore,nactive,nuclear)),
      sigma_(spin_space_.alpha(),spin_space_.beta(),
          prepared_.integrals.active) {
  set_coefficients(coefficients);
}

void FixedCiPoint::set_coefficients(const Eigen::MatrixXd& coefficients) {
  if (coefficients.rows()!=spin_space_.alpha().size() ||
      coefficients.cols()!=spin_space_.beta().size() ||
      !coefficients.allFinite() ||
      std::abs(coefficients.squaredNorm()-1.0)>1e-8)
    throw std::invalid_argument("CI state must be a normalized complete-CAS carrier");
  if ((spin_space_.project(coefficients,twice_spin_)-coefficients).norm()>1e-8)
    throw std::invalid_argument("CI state is outside the requested total-spin sector");
  coefficients_=coefficients;
  energy_prepared_=false;
  state_prepared_=false;
  orbital_gradient_prepared_=false;
  sigma_image_.resize(0,0);
  orbital_gradient_.resize(0,0);
  hessian_cache_prepared_=false;
  two_electron_cache_={};
}

void FixedCiPoint::prepare_state() const {
  if (state_prepared_) return;
  if (energy_prepared_) {
    adjoint_=sigma_.integral_adjoint(coefficients_,coefficients_);
  } else {
    const auto image=sigma_.apply_with_adjoint(coefficients_,coefficients_);
    adjoint_=image.integrals;
    sigma_image_=image.sigma;
    active_energy_=(coefficients_.array()*sigma_image_.array()).sum();
    energy_=prepared_.integrals.core_energy+active_energy_;
    energy_prepared_=true;
  }
  const Eigen::MatrixXd active=orbitals_.middleCols(ncore_,nactive_);
  const Eigen::MatrixXd active_density=active*adjoint_.one_electron*active.transpose();
  active_density_response_=apply_ao_response(active_density);
  state_prepared_=true;
}

void FixedCiPoint::prepare_hessian_cache() const {
  prepare_state();
  if (hessian_cache_prepared_) return;
  two_electron_cache_=integrals::build_exact_packed_active_two_electron_adjoint_cache(
      pack_pair_adjoint(adjoint_.pair_kernel),prepared_.two_electron.dense_active_coefficients,
      *ao_,nactive_,prepared_.two_electron);
  hessian_cache_prepared_=true;
}

double FixedCiPoint::energy() const {
  if (!energy_prepared_) {
    sigma_image_=sigma_.apply(coefficients_);
    active_energy_=(coefficients_.array()*sigma_image_.array()).sum();
    energy_=prepared_.integrals.core_energy+active_energy_;
    energy_prepared_=true;
  }
  return energy_;
}

const mcscf::DirectCiIntegralAdjoint& FixedCiPoint::integral_adjoint() const {
  prepare_state();
  return adjoint_;
}

Eigen::MatrixXd FixedCiPoint::apply_ao_response(const Eigen::MatrixXd& density) const {
  return ao_response(*ao_,density);
}

Eigen::MatrixXd FixedCiPoint::project_ci(const Eigen::MatrixXd& direction) const {
  if (direction.rows()!=coefficients_.rows() || direction.cols()!=coefficients_.cols())
    throw std::invalid_argument("CI direction shape mismatch");
  return spin_space_.project(direction,twice_spin_);
}

Eigen::MatrixXd FixedCiPoint::ci_residual() const {
  if (!energy_prepared_) prepare_state();
  return sigma_image_-active_energy_*coefficients_;
}

Eigen::MatrixXd FixedCiPoint::two_electron_pullback(
    const Eigen::MatrixXd& pair_adjoint) const {
  if (hessian_cache_prepared_)
    return integrals::backpropagate_exact_packed_active_two_electron_gradient(
        pack_pair_adjoint(pair_adjoint),two_electron_cache_);
  return integrals::backpropagate_exact_packed_active_two_electron_gradient(
      pack_pair_adjoint(pair_adjoint),*ao_,prepared_.two_electron);
}

const Eigen::MatrixXd& FixedCiPoint::orbital_gradient_raw() const {
  if (orbital_gradient_prepared_) return orbital_gradient_;
  prepare_state();
  const int n=ao_->n_basis_functions;
  orbital_gradient_=Eigen::MatrixXd::Zero(n,n);
  const auto core=orbitals_.leftCols(ncore_);
  const auto active=orbitals_.middleCols(ncore_,nactive_);
  orbital_gradient_.leftCols(ncore_)=
      4*prepared_.fock*core+2*active_density_response_*core;
  orbital_gradient_.middleCols(ncore_,nactive_)=
      2*prepared_.fock*active*adjoint_.one_electron+
      two_electron_pullback(adjoint_.pair_kernel);
  orbital_gradient_prepared_=true;
  return orbital_gradient_;
}

OrbitalResponse FixedCiPoint::direction(const Eigen::MatrixXd& direction) const {
  const int n=ao_->n_basis_functions;
  if (direction.rows()!=n || direction.cols()!=n || !direction.allFinite())
    throw std::invalid_argument("orbital direction shape mismatch");
  prepare_hessian_cache();
  const auto core=orbitals_.leftCols(ncore_);
  const auto active=orbitals_.middleCols(ncore_,nactive_);
  const auto dcore=direction.leftCols(ncore_);
  const auto dactive=direction.middleCols(ncore_,nactive_);
  const Eigen::MatrixXd ddensity=dcore*core.transpose()+core*dcore.transpose();
  const Eigen::MatrixXd dactive_density=
      dactive*adjoint_.one_electron*active.transpose()+
      active*adjoint_.one_electron*dactive.transpose();
  Eigen::MatrixXd sources(n*n,2);
  sources.col(0)=Eigen::Map<const Eigen::VectorXd>(ddensity.data(),n*n);
  sources.col(1)=Eigen::Map<const Eigen::VectorXd>(dactive_density.data(),n*n);
  integrals::AoH1eBatchWorkspace ao_workspace;
  Eigen::MatrixXd response;
  integrals::apply_symmetric_ao_h1e_batch(
      sources,*ao_,effective_openmp_thread_count(),&ao_workspace,&response);
  const Eigen::Map<const Eigen::MatrixXd> dfock(response.col(0).data(),n,n);
  const Eigen::Map<const Eigen::MatrixXd> dweighted(response.col(1).data(),n,n);

  const std::vector<Eigen::MatrixXd> active_directions{dactive.eval()};
  std::vector<integrals::ExactCtxPairMatrix> pair_products;
  std::vector<Eigen::MatrixXd> fixed_adjoint_directions;
  const Eigen::MatrixXd packed_directions=
      integrals::compute_exact_packed_active_two_electron_integral_directional_derivative_batch(
          two_electron_cache_,active_directions,*ao_,
          &pair_products,&fixed_adjoint_directions);
  if (packed_directions.cols()!=1 || fixed_adjoint_directions.size()!=1 ||
      pair_products.size()!=1)
    throw std::logic_error("incomplete shared exact two-electron response");
  Eigen::MatrixXd d2;
  if (fixed_adjoint_directions[0].size()!=0) {
    d2=std::move(fixed_adjoint_directions[0]);
  } else if (pair_products[0].size()!=0) {
    integrals::ExactPackedActiveTwoElectronApplyWorkspace two_workspace;
    integrals::apply_exact_packed_active_two_electron_adjoint_hessian_vector_fused(
        two_electron_cache_,dactive,*ao_,pair_products[0],&two_workspace,&d2);
  } else {
    throw std::logic_error("missing exact two-electron fixed-adjoint response");
  }

  OrbitalResponse result;
  result.integrals.one_electron=dactive.transpose()*prepared_.fock*active+
      active.transpose()*prepared_.fock*dactive+active.transpose()*dfock*active;
  const int pairs=integrals::packed_active_pair_count(nactive_);
  std::vector<double> packed(packed_directions.data(),
      packed_directions.data()+packed_directions.rows());
  const auto view=integrals::make_active_space_two_electron_view(packed);
  result.integrals.pair_kernel.resize(pairs,pairs);
  for (int p=0; p<pairs; ++p)
    for (int q=0; q<pairs; ++q)
      result.integrals.pair_kernel(p,q)=
          integrals::lookup_active_space_two_electron_kernel_value(view,p,q,nactive_);
  result.orbital_hessian=Eigen::MatrixXd::Zero(n,n);
  result.orbital_hessian.leftCols(ncore_)=
      4*dfock*core+4*prepared_.fock*dcore+
      2*dweighted*core+2*active_density_response_*dcore;
  result.orbital_hessian.middleCols(ncore_,nactive_)=
      2*dfock*active*adjoint_.one_electron+
      2*prepared_.fock*dactive*adjoint_.one_electron+d2;
  return result;
}

Eigen::MatrixXd FixedCiPoint::orbital_gradient_from_adjoint(
    const mcscf::DirectCiIntegralAdjoint& cotangent) const {
  const int n=ao_->n_basis_functions;
  if (cotangent.one_electron.rows()!=nactive_ ||
      cotangent.one_electron.cols()!=nactive_ ||
      cotangent.pair_kernel.rows()!=prepared_.integrals.active.pair_kernel.rows() ||
      cotangent.pair_kernel.cols()!=prepared_.integrals.active.pair_kernel.cols())
    throw std::invalid_argument("active integral cotangent shape mismatch");
  const auto core=orbitals_.leftCols(ncore_);
  const auto active=orbitals_.middleCols(ncore_,nactive_);
  const Eigen::MatrixXd gamma=0.5*(cotangent.one_electron+
      cotangent.one_electron.transpose()).eval();
  const Eigen::MatrixXd weighted=apply_ao_response(active*gamma*active.transpose());
  Eigen::MatrixXd result=Eigen::MatrixXd::Zero(n,n);
  result.leftCols(ncore_)=2*weighted*core;
  result.middleCols(ncore_,nactive_)=
      2*prepared_.fock*active*gamma+two_electron_pullback(cotangent.pair_kernel);
  return result;
}

Eigen::MatrixXd FixedCiPoint::transition_orbital_gradient(
    const Eigen::MatrixXd& left, const Eigen::MatrixXd& right) const {
  if (left.rows()!=coefficients_.rows() || left.cols()!=coefficients_.cols() ||
      right.rows()!=coefficients_.rows() || right.cols()!=coefficients_.cols())
    throw std::invalid_argument("transition CI carrier shape mismatch");
  return orbital_gradient_from_adjoint(sigma_.integral_adjoint(left,right));
}

}  // namespace xmvb::casscf
