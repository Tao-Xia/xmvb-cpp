#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "core/pair_index.hpp"
#include "cuda/ci/sigma.hpp"
#include "cuda/ci/spin_self.hpp"
#include "cuda/linalg/davidson.hpp"
#include "cuda/memory/context.hpp"
#include "cuda/vb/action.hpp"
#include "cuda/vb/product.hpp"
#include "linalg/eigensolver.hpp"
#include "mcscf/ci/sigma.hpp"
#include "mcscf/ci/integrals.hpp"
#include "mcscf/ci/product_support.hpp"
#include "mcscf/ci/strings.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/assembly/hamiltonian_overlap.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"

namespace {

using Strings = std::vector<std::vector<int>>;
using Terms = std::vector<std::vector<xmvb::vb::StructureExpansionTerm>>;

void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

/** Force small block lifetimes in the production selected-coordinate action.
 * Unequal spin sectors, COV and broad proper supports use the same test.
 */
void check_packed_blocks(const xmvb::vb::StructureAction& source,const Eigen::MatrixXd& vectors) {
  const auto view=source.orthogonal_view();
  xmvb::cuda::Context context(64ULL*1024*1024);
  const xmvb::cuda::StructureAction action(context,source);
  auto product=action.make_product_contraction();
  const auto uploaded=context.counters().host_to_device_bytes;
  auto sibling=action.make_product_contraction();
  require(context.counters().host_to_device_bytes==uploaded,
      "second product executor reuploaded immutable support");
  const xmvb::cuda::DirectCiSigma sigma(context,view.sigma,1,view.exchange_parity);
  const xmvb::cuda::ProductIncidence incidence{action.term_offsets_data(),
      action.term_structures_data(),action.term_weights_data(),source.n_structures()};
  for (int width:{1,2,3}) {
    xmvb::cuda::Matrix<double> x(context,source.n_structures(),width),
        h(context,source.n_structures(),width),m(context,source.n_structures(),width);
    xmvb::cuda::copy_to_device(context,x.view(),vectors.leftCols(width));
    const auto expected=source.apply(vectors.leftCols(width));
    const auto host=[&](const auto& device) {
      Eigen::MatrixXd result(source.n_structures(),width);
      xmvb::cuda::copy_to_host(context,result,device.view());return result;
    };
    // The row-panel executor retains an input and an output halo instead of
    // overwriting the input after a full-channel scatter.
    const std::size_t small_budget=width==3 ? 12800U:6400U;
    for (std::size_t budget:{small_budget,std::size_t{1024*1024}}) {
      product->apply_selected(incidence,x.view(),sigma,h.view(),m.view(),budget);
      require((host(h)-expected.hamiltonian).norm()<2e-10,
          "packed product H block differs from selected CPU action");
      require((host(m)-expected.overlap).norm()<2e-10,
          "packed product overlap block differs from selected CPU action");
      if (width==1 && budget==small_budget) {
        const auto first_factor_upload=context.counters().host_to_device_bytes;
        sibling->apply_selected(incidence,x.view(),sigma,h.view(),m.view(),budget);
        require(context.counters().host_to_device_bytes>first_factor_upload,
            "product executors incorrectly shared orbital factor cache");
        require((host(h)-expected.hamiltonian).norm()<2e-10 &&
            (host(m)-expected.overlap).norm()<2e-10,
            "shared-support product executor changed selected H/M");
      }
      product->apply_selected(incidence,x.view(),sigma,{},m.view(),budget,true);
      require((host(m)-expected.overlap).norm()<2e-10,
          "packed overlap-only block changed the selected-space projection");
    }
  }
}

/** Deliberately materialize the tiny CPU carrier here: this oracle does not
 * use the product-panel implementation, its accepted cache, or selected-space
 * closure. In particular, each selected bra is transformed before projection.
 */
void check_product_jet(const xmvb::vb::StructureAction& source,
    const xmvb::cuda::StructureAction& action,xmvb::cuda::Context& context,
    const Strings& alpha_strings,const Strings& beta_strings) {
  const auto view=source.orthogonal_view();
  const int count=source.n_structures(),na=view.alpha.dimension(),nb=view.beta.dimension();
  const Eigen::MatrixXd upper=view.alpha.orbital_transform();
  Eigen::MatrixXd du=Eigen::MatrixXd::Zero(upper.rows(),upper.cols());
  du(0,0)=0.017;du(1,1)=-0.013;du(2,2)=0.009;du(0,1)=0.021;
  du(0,2)=-0.014;du(1,3)=0.011;
  Eigen::MatrixXd c(count,1),q(count,1);
  for (int i=0;i<count;++i) {c(i,0)=std::cos(0.37*(i+1));q(i,0)=std::sin(0.53*(i+2));}
  struct Oracle {
    Eigen::MatrixXd h,m,one,pair,generator;
  };
  const int channels=view.sigma.n_orbitals()*(view.sigma.n_orbitals()+1)/2;
  xmvb::mcscf::ActiveIntegrals delta_integrals{
      Eigen::MatrixXd::Zero(upper.rows(),upper.cols()),
      Eigen::MatrixXd::Zero(channels,channels)};
  for (int j=0;j<upper.cols();++j) for (int i=0;i<=j;++i)
    delta_integrals.one_electron(i,j)=delta_integrals.one_electron(j,i)=
        0.007*std::cos(0.37*(i+1)*(j+2));
  for (int j=0;j<channels;++j) for (int i=0;i<=j;++i)
    delta_integrals.pair_kernel(i,j)=delta_integrals.pair_kernel(j,i)=
        0.003*std::sin(0.19*(i+1)*(j+2));
  const xmvb::mcscf::DirectCiSigmaAction delta_cpu(view.sigma,delta_integrals);
  const auto reference=[&](const Eigen::MatrixXd& coefficients,const Eigen::MatrixXd& frame,
      double integral_step=0.0) {
    const xmvb::vb::ExteriorOrbitalTransform ta(alpha_strings,frame),tb(beta_strings,frame);
    const auto expand=[&](const Eigen::MatrixXd& x) {
      Eigen::MatrixXd result=Eigen::MatrixXd::Zero(na,nb);
      for (int g=0;g<view.products.size();++g)
        for (auto t=view.offsets[g];t<view.offsets[g+1];++t)
          result(view.products[g]/nb,view.products[g]%nb)+=
              view.terms[t].coefficient*x(view.terms[t].structure,0);
      ta.apply_left(&result);tb.apply_right(&result);return result;
    };
    const Eigen::MatrixXd y=expand(coefficients);
    Eigen::MatrixXd hy=view.sigma.apply(y,0);
    if (integral_step!=0.0) hy.noalias()+=integral_step*delta_cpu.apply(y,0);
    Oracle result{Eigen::MatrixXd::Zero(count,1),Eigen::MatrixXd::Zero(count,1),
        {},{},2.0*view.sigma.one_body_generator_adjoint(hy,y)};
    const auto density=view.sigma.integral_adjoint(y,y,0);
    result.one=density.one_electron;result.pair=density.pair_kernel;
    for (int s=0;s<count;++s) {
      Eigen::MatrixXd basis=Eigen::MatrixXd::Zero(count,1);basis(s,0)=1.0;
      const Eigen::MatrixXd bra=expand(basis);
      result.h(s,0)=(bra.array()*hy.array()).sum();
      result.m(s,0)=(bra.array()*y.array()).sum();
    }
    return result;
  };
  const Oracle base=reference(c,upper);
  const double energy=(c.transpose()*base.h)(0,0)/(c.transpose()*base.m)(0,0);
  auto product=action.make_product_contraction();
  const xmvb::cuda::ProductIncidence incidence{action.term_offsets_data(),
      action.term_structures_data(),action.term_weights_data(),count};
  xmvb::cuda::Matrix<double> dc(context,count,1),dq(context,count,1),r(context,upper.rows(),upper.cols()),
      dr(context,upper.rows(),upper.cols()),rel(context,upper.rows(),upper.cols());
  xmvb::cuda::copy_to_device(context,dc.view(),c);
  xmvb::cuda::copy_to_device(context,dq.view(),q);
  xmvb::cuda::copy_to_device(context,r.view(),upper);
  xmvb::cuda::copy_to_device(context,dr.view(),du);
  const Eigen::MatrixXd relative=du*upper.inverse();
  xmvb::cuda::copy_to_device(context,rel.view(),relative);
  xmvb::cuda::DirectCiSigma sigma(context,view.sigma,1,view.exchange_parity);
  xmvb::cuda::ProductEndpoint accepted{xmvb::cuda::Matrix<double>(context,count,1)};
  xmvb::cuda::copy_to_device(context,accepted.coefficients.view(),c);
  const std::size_t budget=8ULL*1024*1024;
  const auto primal=product->jet(incidence,sigma,accepted,r.view(),{},{},nullptr,budget);
  const auto host=[&](const auto& device) {
    Eigen::MatrixXd result(device.rows(),device.cols());
    xmvb::cuda::copy_to_host(context,result,device.view());return result;
  };
  require((host(primal.hamiltonian)-base.h).norm()<2e-10,
      "product jet primal H differs from expanded CPU carrier");
  require((host(primal.overlap)-base.m).norm()<2e-10,
      "product jet primal M differs from expanded CPU carrier");
  require((host(primal.one)-base.one).norm()<2e-10 &&
      (host(primal.pair)-base.pair).norm()<2e-10,
      "product jet primal integral adjoint differs from CPU carrier");
  const Eigen::MatrixXd primal_generator=host(primal.generator);
  for (int j=0;j<primal_generator.cols();++j)
    for (int i=0;i<=j;++i)
      require(std::abs(primal_generator(i,j)-base.generator(i,j))<2e-10,
          "product jet primal generator differs from CPU carrier");
  product->accept(accepted,energy);
  if (accepted.images && accepted.images->complete()) {
    const int beta_strings=source.n_beta_strings();
    require(accepted.extent>0 && accepted.extent<=beta_strings &&
        (beta_strings==1 || accepted.extent<beta_strings),
        "accepted endpoint used a complete beta-width panel");
    for (int panel=0;panel<accepted.images->size();++panel)
      require(accepted.images->at(panel).patterns.size()<=
          static_cast<std::size_t>(accepted.extent),
          "accepted endpoint panel exceeded its beta width");
  }
  const double step=2e-6;
  const Oracle plus=reference(c+step*q,upper+step*du);
  const Oracle minus=reference(c-step*q,upper-step*du);
  const Oracle ci_plus=reference(c+step*q,upper);
  const Oracle ci_minus=reference(c-step*q,upper);
  const auto compare=[&](const xmvb::cuda::ProductJet& value,
      const xmvb::cuda::ProductJet& ci,const Oracle& high,const Oracle& low) {
    const auto finite=[&](const Eigen::MatrixXd& a,const Eigen::MatrixXd& b) {
      return Eigen::MatrixXd((a-b)/(2*step));
    };
    require((host(value.hamiltonian)-finite(high.h,low.h)).norm()<2e-8,
        "product jet directional H differs from carrier finite difference");
    require((host(value.overlap)-finite(high.m,low.m)).norm()<2e-8,
        "product jet directional M differs from carrier finite difference");
    require((2.0*host(value.one)-finite(high.one,low.one)).norm()<2e-8 &&
        (2.0*host(value.pair)-finite(high.pair,low.pair)).norm()<2e-8,
        "product jet directional integral adjoint differs from finite difference");
    const Eigen::MatrixXd expected_generator=finite(high.generator,low.generator)-
        2.0*energy*finite(high.one,low.one);
    const Eigen::MatrixXd actual_generator=host(value.generator);
    for (int j=0;j<actual_generator.cols();++j)
      for (int i=0;i<=j;++i)
        require(std::abs(actual_generator(i,j)-expected_generator(i,j))<3e-8,
            "product jet directional generator differs from finite difference");
    require((host(ci.hamiltonian)-finite(ci_plus.h,ci_minus.h)).norm()<2e-8 &&
        (host(ci.overlap)-finite(ci_plus.m,ci_minus.m)).norm()<2e-8,
        "independent CI jet image includes orbital direction");
    require((2.0*host(ci.one)-finite(ci_plus.one,ci_minus.one)).norm()<2e-8 &&
        (2.0*host(ci.pair)-finite(ci_plus.pair,ci_minus.pair)).norm()<2e-8,
        "independent CI jet integral adjoint includes orbital direction");
    const Eigen::MatrixXd expected_ci=finite(ci_plus.generator,ci_minus.generator)-
        2.0*energy*finite(ci_plus.one,ci_minus.one);
    const Eigen::MatrixXd actual_ci=host(ci.generator);
    for (int j=0;j<actual_ci.cols();++j)
      for (int i=0;i<=j;++i)
        require(std::abs(actual_ci(i,j)-expected_ci(i,j))<3e-8,
            "independent CI jet generator includes orbital direction");
  };
  xmvb::cuda::ProductJet cached_ci;
  const auto cached=product->jet(incidence,sigma,accepted,r.view(),dq.view(),dr.view(),nullptr,
      budget,&cached_ci,rel.view());
  compare(cached,cached_ci,plus,minus);
  xmvb::cuda::ProductEndpoint uncached{xmvb::cuda::Matrix<double>(context,count,1)};
  xmvb::cuda::copy_to_device(context,uncached.coefficients.view(),c);
  uncached.energy=energy;
  xmvb::cuda::ProductJet uncached_ci;
  const auto fresh=product->jet(incidence,sigma,uncached,r.view(),dq.view(),dr.view(),nullptr,
      budget,&uncached_ci,rel.view());
  compare(fresh,uncached_ci,plus,minus);
  // A cached orbital-only action contains G Y, not a CI-response lane.
  // Check it separately from the joint q/GY reader and against the same
  // action without retained accepted images.
  const auto orbital_cached=product->jet(incidence,sigma,accepted,r.view(),{},dr.view(),
      nullptr,budget,nullptr,rel.view());
  const auto orbital_fresh=product->jet(incidence,sigma,uncached,r.view(),{},dr.view(),
      nullptr,budget,nullptr,rel.view());
  for (const auto& pair:std::array{
      std::pair{host(orbital_cached.hamiltonian),host(orbital_fresh.hamiltonian)},
      std::pair{host(orbital_cached.overlap),host(orbital_fresh.overlap)},
      std::pair{host(orbital_cached.one),host(orbital_fresh.one)},
      std::pair{host(orbital_cached.pair),host(orbital_fresh.pair)}})
    require((pair.first-pair.second).norm()<3e-8*std::max(1.0,pair.second.norm()),
        "cached orbital-only product reader differs from fresh action");
  const auto cached_generator=host(orbital_cached.generator);
  const auto fresh_generator=host(orbital_fresh.generator);
  for (int j=0;j<cached_generator.cols();++j)
    for (int i=0;i<=j;++i)
      require(std::abs(cached_generator(i,j)-fresh_generator(i,j))<3e-8,
          "cached orbital-only generator differs from fresh action");
  // The former 3200-byte allowance did not price the channel cotangents.
  // Enforce the resource error, then test the same jet with room for its
  // fixed integral-space scratch and retained input/output halos.
  xmvb::cuda::ProductEndpoint streaming{xmvb::cuda::Matrix<double>(context,count,1)};
  xmvb::cuda::copy_to_device(context,streaming.coefficients.view(),c);
  streaming.energy=energy;
  xmvb::cuda::ProductJet streaming_ci;
  bool limited=false;
  try {
    product->jet(incidence,sigma,streaming,r.view(),dq.view(),dr.view(),
        nullptr,3200,&streaming_ci,rel.view());
  } catch (const std::length_error&) { limited=true; }
  require(limited,"undersized joint sigma workspace was not rejected");
  const auto bounded=product->jet(incidence,sigma,streaming,r.view(),dq.view(),dr.view(),
      nullptr,32ULL*1024,&streaming_ci,rel.view());
  compare(bounded,streaming_ci,plus,minus);
  if (view.exchange_parity) {
    xmvb::cuda::Matrix<double> delta_h(context,upper.rows(),upper.cols()),
        delta_g(context,channels,channels);
    xmvb::cuda::copy_to_device(context,delta_h.view(),delta_integrals.one_electron);
    xmvb::cuda::copy_to_device(context,delta_g.view(),delta_integrals.pair_kernel);
    xmvb::cuda::DirectCiSigma delta_gpu(context,sigma);
    delta_gpu.update_integrals(delta_h.view(),delta_g.view());
    const Oracle coupled_plus=reference(c+step*q,upper+step*du,step);
    const Oracle coupled_minus=reference(c-step*q,upper-step*du,-step);
    xmvb::cuda::ProductJet coupled_ci;
    const auto coupled=product->jet(incidence,sigma,accepted,r.view(),dq.view(),dr.view(),
        &delta_gpu,budget,&coupled_ci,rel.view());
    compare(coupled,coupled_ci,coupled_plus,coupled_minus);
    xmvb::cuda::ProductJet coupled_fresh_ci;
    const auto coupled_fresh=product->jet(incidence,sigma,uncached,r.view(),dq.view(),dr.view(),
        &delta_gpu,budget,&coupled_fresh_ci,rel.view());
    compare(coupled_fresh,coupled_fresh_ci,coupled_plus,coupled_minus);
  }
}

void check_selected_below_carrier_budget() {
  // The complete 5/5 one-spin spaces are much larger than the two selected
  // structures. The old graph path required two full 252 x 252 carriers.
  constexpr int n=10,electrons=5,selected=2;
  Strings strings;
  for (unsigned mask=0;mask<(1u<<n);++mask) if (std::popcount(mask)==electrons) {
    strings.emplace_back();
    for (int p=0;p<n;++p) if (mask&(1u<<p)) strings.back().push_back(p);
  }
  require(strings.size()==252,"large selected test has incomplete spin strings");
  Strings products_alpha,products_beta;
  Terms terms(strings.size()*strings.size());
  products_alpha.reserve(terms.size());products_beta.reserve(terms.size());
  for (const auto& a:strings) for (const auto& b:strings) {
    products_alpha.push_back(a);products_beta.push_back(b);
  }
  terms[0].push_back({0,1.0});
  terms[terms.size()/2].push_back({1,-0.6});
  terms.back().push_back({0,0.2});
  terms.back().push_back({1,0.8});
  Eigen::MatrixXd overlap=Eigen::MatrixXd::Identity(n,n);
  for (int p=0;p<n-1;++p) overlap(p,p+1)=overlap(p+1,p)=0.015;
  const std::vector<double> packed_overlap(overlap.data(),overlap.data()+overlap.size());
  Eigen::MatrixXd one=Eigen::MatrixXd::Zero(n,n);
  for (int p=0;p<n;++p) {
    one(p,p)=-0.3+0.1*p;
    if (p+1<n) one(p,p+1)=one(p+1,p)=0.025;
  }
  xmvb::integrals::ActiveSpaceTwoElectronResult eri;
  eri.packed_active_two_electron_integrals.resize(
      xmvb::integrals::packed_active_two_electron_integral_count(n));
  for (std::size_t i=0;i<eri.packed_active_two_electron_integrals.size();++i)
    eri.packed_active_two_electron_integrals[i]=0.002*std::cos(0.13*(i+1));
  const auto topology=xmvb::vb::build_orthogonal_spin_topology(
      products_alpha,products_beta,n);
  const xmvb::vb::StructureAction cpu(terms,selected,topology,packed_overlap,one,eri,n,
      nullptr,xmvb::vb::StructureActionMode::ActionOnly);
  std::size_t baseline=0;
  {
    xmvb::cuda::Context wide(256ULL*1024*1024);
    const xmvb::cuda::StructureAction warm(wide,cpu);
    baseline=wide.counters().current_device_bytes;
  }
  const std::size_t full_carriers=2ULL*strings.size()*strings.size()*sizeof(double);
  const std::size_t budget=baseline+full_carriers-1;
  xmvb::cuda::Context context(budget);
  const xmvb::cuda::StructureAction gpu(context,cpu);
  require(context.counters().current_device_bytes<budget,
      "large selected action constructor used the old full-carrier budget");
  Eigen::MatrixXd v(selected,1);v<<0.73,-0.41;
  const auto expected=cpu.apply(v);
  xmvb::cuda::Matrix<double> x(context,selected,1),h(context,selected,1),m(context,selected,1);
  xmvb::cuda::copy_to_device(context,x.view(),v);
  gpu.apply(x.view(),h.view(),m.view());
  Eigen::MatrixXd actual_h(selected,1),actual_m(selected,1);
  xmvb::cuda::copy_to_host(context,actual_h,h.view());
  xmvb::cuda::copy_to_host(context,actual_m,m.view());
  require((actual_h-expected.hamiltonian).norm()<2e-9 &&
      (actual_m-expected.overlap).norm()<2e-9,
      "small-budget selected H/M differs from CPU after first apply");
  Eigen::MatrixXd carrier=cpu.expand_structure_block(v);
  const auto view=cpu.orthogonal_view();
  view.alpha.apply_left(&carrier);view.beta.apply_right(&carrier);
  require(std::abs(gpu.maximum_orthogonal_component(x.view())-
      carrier.cwiseAbs().maxCoeff())<2e-10,
      "small-budget selected maximum differs from explicit transformed carrier");
  require(context.counters().peak_device_bytes<=budget,
      "small-budget selected action exceeded the device allocation limit");
}

void check_self_tile_budget(const xmvb::vb::StructureAction& source) {
  const auto view=source.orthogonal_view();
  const auto& transform=view.alpha;
  const auto& connections=view.sigma.alpha_connections();
  const auto& kernel=view.sigma.channel_kernel();
  const int n=transform.n_orbitals(),k=connections.n_occupied(),u=transform.dimension();
  const int channels=static_cast<int>(kernel.rows());
  const std::size_t persistent=(1ULL*n*n+1ULL*channels*channels+1ULL*u*channels+2ULL*u)*sizeof(double)+
      (u*k+2*channels)*sizeof(int);
  const std::size_t one_string=(2ULL*n*k+2ULL*k*k+1ULL*n*n)*sizeof(double)+
      2*sizeof(double*)+(k+1)*sizeof(int);
  xmvb::cuda::Context context(persistent+one_string);
  xmvb::cuda::Matrix<double> frame(context,n,n),device_kernel(context,channels,channels);
  xmvb::cuda::copy_to_device(context,frame.view(),transform.orbital_transform());
  xmvb::cuda::copy_to_device(context,device_kernel.view(),kernel);
  xmvb::cuda::SpinSelf model(context,n,transform.occupations());
  model.update(frame.view(),device_kernel.view());

  // Independent small exterior-space oracle; it never uses a Gram inverse.
  Eigen::MatrixXd exterior=Eigen::MatrixXd::Identity(u,u);
  transform.apply_left(&exterior);
  std::vector<Eigen::MatrixXd> generators(channels,Eigen::MatrixXd::Zero(u,u));
  for (int row=0;row<u;++row) {
    for (int orbital:connections.occupied(row))
      generators[orbital*(orbital+1)/2+orbital](row,row)=1.0;
    for (const auto& link:connections.singles(row))
      generators[link.pair](row,link.source)=link.sign;
  }
  Eigen::MatrixXd h=Eigen::MatrixXd::Zero(u,u);
  for (int p=0;p<channels;++p) for (int q=0;q<channels;++q)
    h.noalias()+=kernel(p,q)*generators[p]*generators[q];
  Eigen::MatrixXd energy(u,1),overlap(u,1),means(u,channels);
  xmvb::cuda::copy_to_host(context,energy,model.energy());
  xmvb::cuda::copy_to_host(context,overlap,model.overlap());
  xmvb::cuda::copy_to_host(context,means,model.moments());
  for (int row=0;row<u;++row) {
    const auto column=exterior.col(row);
    const double norm=column.squaredNorm();
    require(std::abs(overlap(row,0)-norm)<2e-11,"tiled spin-self norm differs from exterior oracle");
    require(std::abs(energy(row,0)-column.dot(h*column)/norm)<2e-10,
        "tiled spin-self energy differs from exterior oracle");
    for (int p=0;p<channels;++p)
      require(std::abs(means(row,p)-column.dot(generators[p]*column)/norm)<2e-10,
          "tiled spin-self moment differs from exterior oracle");
  }
  require(context.counters().peak_device_bytes<=persistent+one_string,
      "spin-self tile exceeded its exact one-string budget");
  const std::vector<std::uint64_t> empty_spin{0};
  xmvb::cuda::SpinSelf vacuum(context,n,empty_spin);
  vacuum.update(frame.view(),device_kernel.view());
  Eigen::MatrixXd vacuum_overlap(1,1),vacuum_energy(1,1);
  xmvb::cuda::copy_to_host(context,vacuum_overlap,vacuum.overlap());
  xmvb::cuda::copy_to_host(context,vacuum_energy,vacuum.energy());
  require(vacuum_overlap(0,0)==1.0 && vacuum_energy(0,0)==0.0,
      "empty spin self diagonal is not the vacuum");
}

void check_action(int count, bool transformed,bool covalent=false,bool broad=false,
    bool full=false,bool exchange_covalent=false,bool sparse_three=false) {
  // Four orbitals, two alpha electrons and one beta electron: the two spin
  // string spaces have different dimensions and different exterior maps.
  constexpr int n = 4;
  const Strings alpha{{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
  const Strings beta=covalent ? alpha:Strings{{0},{1},{2},{3}};
  Strings products_alpha, products_beta;
  Terms terms;
  for (const auto& a : alpha)
    for (const auto& b : beta) {
      products_alpha.push_back(a);
      products_beta.push_back(b);
      terms.emplace_back();
    }
  // The third column shares determinant products with the first two. This
  // exercises a nontrivial selected overlap, including for the 3-column case.
  if (full) {
    for (int g=0;g<terms.size();++g) for (int s=0;s<count;++s)
      terms[g].push_back({s,std::sin(0.23*(g+1)*(s+1))});
  } else if (broad) {
    // A proper support broad enough to prefer CPU's ordered-beta contraction.
    // Exclude the unique maximal seed, so it is not the complete product grid.
    for (int g=0;g+1<terms.size();++g) for (int s=0;s<count;++s)
      terms[g].push_back({s,std::sin(0.23*(g+1)*(s+1))});
  } else if (covalent) {
    // The full singly occupied family has strict transformed support and
    // nontrivial frontier channel envelopes. Both selected coefficient
    // columns remain arbitrary; no closure is assumed for their span.
    for (int a=0;a<alpha.size();++a) for (int b=0;b<beta.size();++b) {
      bool disjoint=true;
      for (int p:alpha[a]) for (int q:beta[b]) disjoint=disjoint && p!=q;
      if (!disjoint) continue;
      terms[a*beta.size()+b].push_back({0,1.0});
      if (count==3) {
        const int key=exchange_covalent ? a*b+1:a+1;
        terms[a*beta.size()+b].push_back({1,std::sin(0.27*key)});
        terms[a*beta.size()+b].push_back({2,std::cos(0.41*key)});
      }
    }
  } else if (sparse_three) {
    // Three independent, nonmaximal seeds leave beta columns 2 and 3
    // exactly empty even though the high-budget panel contains all four.
    terms[0].push_back({0,1.0});
    terms[5].push_back({1,1.0});
    terms[9].push_back({2,1.0});
  } else terms[0].push_back({0,1.0});
  if (count == 3 && !covalent && !broad && !sparse_three) {
    terms[5].push_back({1,1.0});
    terms[0].push_back({2,0.6});
    terms[5].push_back({2,-0.3});
    terms[23].push_back({2,0.7});
  }
  Eigen::Matrix4d s = Eigen::Matrix4d::Identity();
  if (transformed) {
    s(0,1) = s(1,0) = 0.11;
    s(1,2) = s(2,1) = -0.07;
    s(0,2) = s(2,0) = 0.04;
  }
  const std::vector<double> overlap(s.data(),s.data()+s.size());
  Eigen::Matrix4d h;
  h << -1.0,0.08,-0.03,0.02,
       0.08,-0.7,0.05,-0.01,
       -0.03,0.05,-0.3,0.04,
       0.02,-0.01,0.04,0.2;
  xmvb::integrals::ActiveSpaceTwoElectronResult eri;
  auto& packed = eri.packed_active_two_electron_integrals;
  packed.resize(xmvb::integrals::packed_active_two_electron_integral_count(n));
  for (int p = 0; p < 10; ++p)
    for (int q = 0; q <= p; ++q)
      packed[xmvb::core::PairIndex::packed_pair_of_pairs_index(p,q)] =
          0.015 * std::cos(0.23 * (p+1) * (q+2));
  const auto topology = xmvb::vb::build_orthogonal_spin_topology(
      products_alpha,products_beta,n);
  const auto diagonal_model = xmvb::vb::FullDeterminantStructureHamiltonianOverlapBuilder{}
      .build_davidson_jacobi_preconditioner(products_alpha,products_beta,terms,
          overlap,h,n,eri,count,topology);
  const xmvb::vb::StructureAction cpu(terms,count,topology,overlap,h,eri,n,&diagonal_model);
  if (sparse_three) {
    require(cpu.orthogonal_view().support && !cpu.orthogonal_view().support->full(),
        "three-seed fixture unexpectedly has complete product support");
    std::vector<int> empty;
    cpu.orthogonal_view().support->append_column_rows(2,empty);
    require(empty.empty(),"three-seed fixture lost its empty beta column");
  }
  if (exchange_covalent)
    require(cpu.orthogonal_view().exchange_parity==1,
        "equal-spin selected COV fixture did not certify positive exchange parity");
  if (count==1 && transformed) check_self_tile_budget(cpu);
  xmvb::cuda::Context context(256ULL*1024ULL*1024ULL);
  const xmvb::cuda::StructureAction gpu(context,cpu);
  require(gpu.carrier_sigma().exchange_parity()==cpu.orthogonal_view().exchange_parity,
      "CUDA structure action lost the common spin-exchange certificate");
  xmvb::cuda::Matrix<double> model_h(context,count,1),model_m(context,count,1);
  gpu.diagonal_model(model_h.view(),model_m.view());
  Eigen::MatrixXd self_h(count,1),self_m(count,1);
  xmvb::cuda::copy_to_host(context,self_h,model_h.view());
  xmvb::cuda::copy_to_host(context,self_m,model_m.view());
  require((self_h.col(0)-diagonal_model.hamiltonian).norm()<
      2e-10*std::max(1.0,diagonal_model.hamiltonian.norm()),
      "GPU nonorthogonal self Hamiltonian model differs from CPU");
  require((self_m.col(0)-diagonal_model.overlap).norm()<
      2e-10*std::max(1.0,diagonal_model.overlap.norm()),
      "GPU nonorthogonal self overlap model differs from CPU");
  Eigen::MatrixXd vectors(count,3);
  for (int j = 0; j < vectors.cols(); ++j)
    for (int i = 0; i < count; ++i)
      vectors(i,j) = std::sin(0.37*(i+1)*(j+2));
  const auto expected = cpu.apply(vectors);
  check_packed_blocks(cpu,vectors);
  const auto actual = gpu.apply(vectors);
  const double h_error = (actual.hamiltonian-expected.hamiltonian).norm() /
      std::max(1.0,expected.hamiltonian.norm());
  const double s_error = (actual.overlap-expected.overlap).norm() /
      std::max(1.0,expected.overlap.norm());
  require(h_error < 2e-10 && s_error < 2e-10,
          "GPU structure H/S block differs from the CPU action");
  if (count==3 && transformed && (!covalent || exchange_covalent))
    check_product_jet(cpu,gpu,context,alpha,beta);
  require(gpu.size() == count,"GPU action structure dimension changed");
  require(context.counters().peak_device_bytes != 0,
          "GPU action allocated no device storage");
  {
    xmvb::cuda::DirectCiSigma point_sigma(context,gpu.carrier_sigma());
    const auto initial_generation=gpu.carrier_sigma().workspace_generation();
    point_sigma.prepare_with_adjoint(1);
    require(point_sigma.workspace_generation()!=initial_generation &&
        point_sigma.workspace_generation()==gpu.carrier_sigma().workspace_generation(),
        "CUDA point sigma did not share action scratch generation");
    const auto allocated=context.counters().current_device_bytes;
    point_sigma.prepare_with_adjoint(1);
    require(context.counters().current_device_bytes==allocated,
        "CUDA point sigma duplicated an already prepared workspace");
    const auto after_fused=gpu.apply(vectors);
    require((after_fused.hamiltonian-expected.hamiltonian).norm()<2e-10 &&
        (after_fused.overlap-expected.overlap).norm()<2e-10,
        "CUDA structure graph retained stale fused scratch addresses");
    point_sigma.prepare(3);
    const auto after_resize=gpu.apply(vectors);
    require((after_resize.hamiltonian-expected.hamiltonian).norm()<2e-10 &&
        (after_resize.overlap-expected.overlap).norm()<2e-10,
        "CUDA structure graph retained stale resized scratch addresses");
    // A product-column action can return the resident executor's idle slab
    // to the shared pool. Prepare the lease whose release is tested here.
    point_sigma.prepare(3);
    const auto resident=context.counters().current_device_bytes;
    const auto generation=point_sigma.workspace_generation();
    point_sigma.reset_workspace();
    require(context.counters().current_device_bytes<resident &&
        point_sigma.workspace_generation()!=generation,
        "CUDA sigma phase did not release its scratch or invalidate the graph");
    const auto after_release=gpu.apply(vectors);
    require((after_release.hamiltonian-expected.hamiltonian).norm()<2e-10 &&
        (after_release.overlap-expected.overlap).norm()<2e-10,
        "CUDA structure graph replayed released scratch addresses");
  }
  if (count == 3) {
    const auto& diagonal = cpu.preconditioner_diagonal();
    const auto options = xmvb::linalg::make_davidson_options(
        count,1,1e-9,1e-8,1e-9);
    const xmvb::linalg::GeneralizedEigenAction cpu_action = [&cpu](
        const Eigen::Ref<const Eigen::MatrixXd>& x) {
      auto images = cpu.apply(x);
      return xmvb::linalg::GeneralizedEigenActionResult{
          std::move(images.hamiltonian),std::move(images.overlap)};
    };
    const xmvb::cuda::DeviceEigenAction gpu_action = [&gpu](
        xmvb::linalg::ConstMatrixView x,xmvb::linalg::MatrixView hx,
        xmvb::linalg::MatrixView sx) { gpu.apply(x,hx,sx); };
    const auto cpu_roots = xmvb::linalg::GeneralizedEigensolver().solve_davidson(
        cpu_action,diagonal.hamiltonian,diagonal.overlap,options);
    const auto gpu_roots = xmvb::cuda::solve_davidson(
        context,gpu_action,diagonal.hamiltonian,diagonal.overlap,options);
    require(std::abs(cpu_roots.eigenpairs.eigenvalues.at(0)-
                     gpu_roots.eigenpairs.eigenvalues.at(0)) < 1e-8,
            "GPU Davidson energy differs from current CPU source");
    require(gpu_roots.stationarity_residual_norms.at(0) < 1.01e-9,
            "GPU Davidson did not satisfy CI stationarity");
  }
}

}  // namespace

int main() {
  try {
    check_action(1,false);
    check_action(1,true);
    check_action(3,false);
    check_action(3,true);
    check_action(1,true,true);
    check_action(3,true,true);
    check_action(3,true,true,false,false,true);
    check_action(3,true,false,true);
    check_action(3,true,false,false,true);
    check_action(3,true,false,false,false,false,true);
    check_selected_below_carrier_budget();
    std::cout << "CUDA structure H/S block parity: PASS\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
