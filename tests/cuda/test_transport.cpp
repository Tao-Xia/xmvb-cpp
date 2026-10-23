#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "core/pair_index.hpp"
#include "cuda/vb/action.hpp"
#include "cuda/vb/metric.hpp"
#include "cuda/vb/transport.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/assembly/hamiltonian_overlap.hpp"
#include "vbscf/structures/overlap/representation.hpp"

namespace {

using Matrix=Eigen::MatrixXd;
using Vector=Eigen::VectorXd;
using Strings=std::vector<std::vector<int>>;
using Terms=std::vector<std::vector<xmvb::vb::StructureExpansionTerm>>;

void require(bool condition,const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void near(const Matrix& actual,const Matrix& expected,double tolerance,
          const char* message) {
  require(actual.rows()==expected.rows() && actual.cols()==expected.cols() &&
          actual.allFinite() &&
          (actual-expected).norm()<tolerance*std::max(1.0,expected.norm()),
          message);
}

Matrix download(xmvb::cuda::Context& context,
                xmvb::linalg::ConstMatrixView source) {
  Matrix host(source.rows,source.cols);
  xmvb::cuda::copy_to_host(context,host,source);
  return host;
}

struct Fixture {
  constexpr static int orbitals=4,structures=3;
  Strings alpha{{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
  Strings beta{{0},{1},{2},{3}};
  Strings products_alpha,products_beta;
  Terms terms;
  Matrix r0=Matrix::Identity(orbitals,orbitals);
  Matrix h=Matrix::Identity(orbitals,orbitals);
  xmvb::integrals::ActiveSpaceTwoElectronResult eri;
  xmvb::vb::SameSpinPairCacheContext topology;
  std::unique_ptr<xmvb::vb::StructureAction> cpu;

  Fixture() {
    r0(0,0)=1.08; r0(1,1)=1.04; r0(2,2)=1.12; r0(3,3)=1.06;
    r0(0,1)=0.06; r0(0,2)=-0.025; r0(1,2)=0.035;
    r0(1,3)=-0.018; r0(2,3)=0.027;
    h(0,0)=-1.0; h(1,1)=-0.72; h(2,2)=-0.37; h(3,3)=0.14;
    h(0,1)=h(1,0)=0.035;
    for (const auto& a:alpha)
      for (const auto& b:beta) {
        products_alpha.push_back(a);
        products_beta.push_back(b);
        terms.emplace_back();
      }
    terms[0].push_back({0,1.0});
    terms[5].push_back({1,1.0});
    terms[0].push_back({2,0.6});
    terms[5].push_back({2,-0.3});
    terms[23].push_back({2,0.7});
    auto& packed=eri.packed_active_two_electron_integrals;
    packed.resize(xmvb::integrals::packed_active_two_electron_integral_count(orbitals));
    for (int p=0;p<10;++p)
      for (int q=0;q<=p;++q)
        packed[xmvb::core::PairIndex::packed_pair_of_pairs_index(p,q)]=
            0.012*std::cos(0.19*(p+1)*(q+2));
    topology=xmvb::vb::build_orthogonal_spin_topology(
        products_alpha,products_beta,orbitals);
    const Matrix overlap=r0.transpose()*r0;
    const std::vector<double> flat(overlap.data(),overlap.data()+overlap.size());
    const auto diagonal=xmvb::vb::FullDeterminantStructureHamiltonianOverlapBuilder{}
        .build_davidson_jacobi_preconditioner(
            products_alpha,products_beta,terms,flat,h,orbitals,eri,structures,
            topology);
    cpu=std::make_unique<xmvb::vb::StructureAction>(
        terms,structures,topology,flat,h,eri,orbitals,&diagonal,
        xmvb::vb::StructureActionMode::Davidson);
  }
};

void check_selected_transport() {
  Fixture f;
  xmvb::vb::RepresentationOverlap reference(*f.cpu,
      f.topology.alpha_reuse_table.unique_determinants,
      f.topology.beta_reuse_table.unique_determinants,Fixture::orbitals);
  xmvb::cuda::Context context(128ULL*1024ULL*1024ULL);
  xmvb::cuda::StructureAction action(context,*f.cpu);
  xmvb::cuda::SelectedCrossOverlap overlap(context,action,Fixture::orbitals);
  Matrix d=Matrix::Zero(4,4),e=Matrix::Zero(4,4);
  d(0,0)=0.12; d(1,1)=-0.09; d(2,2)=0.07; d(3,3)=0.03;
  d(0,1)=0.14; d(0,3)=-0.05; d(1,2)=0.08; d(2,3)=-0.06;
  e(0,0)=-0.04; e(1,1)=0.02; e(2,2)=0.03; e(3,3)=-0.01;
  e(0,2)=0.07; e(1,3)=-0.035;
  const Matrix trial=f.r0+0.08*d;
  Vector x(3),y(3),rhs(3);
  x<<0.37,-0.21,0.49;
  y<<-0.18,0.33,0.27;
  rhs<<0.19,-0.41,0.16;
  xmvb::cuda::Matrix<double> dr0(context,4,4),dt(context,4,4),dd(context,4,4),de(context,4,4);
  xmvb::cuda::Matrix<double> dx(context,3,1),dy(context,3,1),dout(context,3,1);
  xmvb::cuda::Matrix<double> dgleft(context,4,4),dgright(context,4,4);
  xmvb::cuda::copy_to_device(context,dr0.view(),f.r0);
  xmvb::cuda::copy_to_device(context,dt.view(),trial);
  xmvb::cuda::copy_to_device(context,dd.view(),d);
  xmvb::cuda::copy_to_device(context,de.view(),e);
  xmvb::cuda::copy_to_device(context,dx.view(),x);
  xmvb::cuda::copy_to_device(context,dy.view(),y);
  overlap.apply(dt.view(),dr0.view(),dx.view(),dout.view());
  near(download(context,dout.view()),reference.apply(trial,f.r0,x),2e-10,
       "selected cross-frame overlap differs from CPU");
  overlap.apply_direction(dt.view(),dr0.view(),dd.view(),de.view(),
                          dx.view(),dout.view());
  near(download(context,dout.view()),
       reference.apply_direction(trial,f.r0,d,e,x),2e-9,
       "selected cross-frame direction differs from CPU");
  overlap.adjoint(dt.view(),dr0.view(),dy.view(),dx.view(),
                  dgleft.view(),dgright.view());
  const auto expected=reference.adjoint(trial,f.r0,y,x);
  near(download(context,dgleft.view()),expected.left,3e-9,
       "selected cross-frame left adjoint differs from CPU");
  near(download(context,dgright.view()),expected.right,3e-9,
       "selected cross-frame right adjoint differs from CPU");
  {
    Vector r(3),w(3);r<<0.23,-0.07,0.41;w<<-0.12,0.19,0.37;
    xmvb::cuda::Matrix<double> dr(context,3,1),dw(context,3,1);
    xmvb::cuda::copy_to_device(context,dr.view(),r);
    xmvb::cuda::copy_to_device(context,dw.view(),w);
    overlap.transport_image(dt.view(),dr0.view(),de.view(),dx.view(),dr.view(),dout.view());
    const Matrix zero=Matrix::Zero(4,4);
    near(download(context,dout.view()),reference.apply(trial,f.r0,x)+
        reference.apply_direction(trial,f.r0,zero,e,r),3e-9,
        "contracted transport RHS differs from unfused CPU formula");
    overlap.transport_gradient(dt.view(),dr0.view(),de.view(),dy.view(),dx.view(),
        dr.view(),dw.view(),dgleft.view(),dgright.view(),dout.view());
    const auto a=reference.adjoint(trial,f.r0,y,x);
    const auto b=reference.adjoint_direction(trial,f.r0,zero,e,y,r);
    const auto m=reference.adjoint(trial,trial,y,w);
    const auto o=reference.adjoint(trial,f.r0,y,r);
    near(download(context,dgleft.view()),a.left+b.left-m.left-m.right,4e-8,
        "contracted transport current cotangent differs from unfused CPU formula");
    near(download(context,dgright.view()),o.right,4e-8,
        "contracted transport origin cotangent differs from unfused CPU formula");
    near(download(context,dout.view()),reference.apply(f.r0,trial,y),3e-9,
        "shared transport left image differs from independent overlap");
  }

  xmvb::cuda::copy_to_device(context,dx.view(),rhs);
  overlap.solve_metric(dt.view(),dx.view(),dout.view());
  const Matrix identity=Matrix::Identity(3,3);
  const Matrix gram=reference.apply(trial,trial,identity);
  near(download(context,dout.view()),gram.llt().solve(rhs),3e-8,
       "selected metric GPU solve differs from exact selected Gram");
  xmvb::cuda::Matrix<double> dgram(context,3,3);
  xmvb::cuda::copy_to_device(context,dgram.view(),gram);
  auto shared_metric=std::make_shared<xmvb::cuda::SelectedMetric>(context,std::move(dgram));
  const auto before_cached_solve=action.version();
  xmvb::linalg::MetricSolveStats cached_stats;
  overlap.solve_metric(dt.view(),dx.view(),dout.view(),{},&cached_stats,
                       shared_metric.get());
  near(download(context,dout.view()),gram.llt().solve(rhs),3e-8,
       "point-owned selected metric solve differs from exact selected Gram");
  require(action.version()==before_cached_solve &&
          cached_stats.actions==1 && cached_stats.certifications==1 &&
          cached_stats.relative_residual<3e-8,
          "point-owned metric solve changed action frame or skipped certification");
  overlap.solve_metric(dt.view(),dx.view(),dout.view(),{},&cached_stats,
                       shared_metric.get());
  require(action.version()==before_cached_solve,
          "repeated point-owned metric solve changed action frame");

  const auto trial_identity=xmvb::cuda::OrbitalFrameIdentity::fresh();
  action.update_frame(dt.view(),trial_identity);
  const auto bound_version=action.version();
  action.update_frame(dt.view(),trial_identity);
  require(action.version()==bound_version,
          "same orbital-frame identity invalidated the action cache");
  overlap.apply_metric(dt.view(),dx.view(),dout.view(),trial_identity);
  near(download(context,dout.view()),gram*rhs,3e-9,
       "same-identity selected metric action changed numerically");
  require(action.version()==bound_version,
          "same-identity selected metric rebound the orbital frame");
  action.update_frame(dt.view(),xmvb::cuda::OrbitalFrameIdentity::fresh());
  require(action.version()>bound_version,
          "fresh identity did not invalidate reused frame storage");

  Vector c0(3); c0<<0.51,-0.36,0.28;
  const Matrix gram0=reference.apply(f.r0,f.r0,identity);
  c0/=std::sqrt(c0.dot(gram0*c0));
  Matrix c0_block(3,1);
  c0_block.col(0)=c0;
  const auto images=f.cpu->apply(c0_block);
  xmvb::cuda::Matrix<double> dc0(context,3,1),dhc0(context,3,1),dmc0(context,3,1);
  xmvb::cuda::copy_to_device(context,dc0.view(),c0);
  xmvb::cuda::copy_to_device(context,dhc0.view(),images.hamiltonian);
  xmvb::cuda::copy_to_device(context,dmc0.view(),images.overlap);
  xmvb::cuda::SingleStateCiTransport transport(
      context,overlap,dr0.view(),dc0.view(),dhc0.view(),dmc0.view(),1.0);
  const Vector gc0=download(context,transport.origin_coefficient_gradient());
  require(gc0.norm()>1e-3,
          "transport test did not retain a finite CI residual");
  Matrix b(3,3); b<<0.4,0.03,-0.02, 0.03,0.7,0.06, -0.02,0.06,0.5;
  Vector u(3); u<<0.12,-0.08,0.17;
  struct Value { double energy; double orbital; double ci; };
  const auto evaluate=[&](double orbital,double ci,
      std::shared_ptr<const xmvb::cuda::SelectedMetric> selected_metric={}) -> Value {
    const Matrix frame=f.r0+orbital*d;
    const Matrix origin_direction=orbital*d;
    const Vector base=c0+ci*u;
    if (!selected_metric) {
      const Matrix selected_gram=reference.apply(frame,frame,identity);
      xmvb::cuda::Matrix<double> device_gram(context,3,3);
      xmvb::cuda::copy_to_device(context,device_gram.view(),selected_gram);
      selected_metric=std::make_shared<xmvb::cuda::SelectedMetric>(
          context,std::move(device_gram));
    }
    xmvb::cuda::copy_to_device(context,dt.view(),frame);
    xmvb::cuda::copy_to_device(context,dd.view(),origin_direction);
    xmvb::cuda::copy_to_device(context,dx.view(),base);
    const auto point=transport.retract(dt.view(),dd.view(),dx.view(),selected_metric);
    require(point.selected_metric.get()==selected_metric.get(),
            "transport point did not retain its selected metric");
    selected_metric.reset();
    const Vector coeff=download(context,point.coefficients.view());
    const Vector delta=coeff-c0;
    const double physical=gc0.dot(delta)+0.5*delta.dot(b*delta);
    const Vector coefficient_gradient=gc0+b*delta;
    xmvb::cuda::copy_to_device(context,dy.view(),coefficient_gradient);
    const auto before_model=action.version();
    const auto model=transport.evaluate_model(point,physical,dy.view());
    require(action.version()==before_model,
            "transport model invalidated its unchanged trial frame");
    const Matrix current=download(context,model.current_upper.view());
    const Matrix origin=download(context,model.origin_upper.view());
    const Vector ci_covector=download(context,model.structure_covector.view());
    return {model.energy,(current+origin).cwiseProduct(d).sum(),
            std::sqrt(2.0)*ci_covector.dot(u)};
  };
  constexpr double step=2e-5;
  const auto model=evaluate(0.035,-0.023);
  const Matrix repeated_frame=f.r0+0.035*d;
  const Matrix repeated_gram=reference.apply(repeated_frame,repeated_frame,identity);
  xmvb::cuda::Matrix<double> repeated_device_gram(context,3,3);
  xmvb::cuda::copy_to_device(context,repeated_device_gram.view(),repeated_gram);
  auto repeated_metric=std::make_shared<xmvb::cuda::SelectedMetric>(
      context,std::move(repeated_device_gram));
  const auto repeated_a=evaluate(0.035,-0.023,repeated_metric);
  const auto repeated_b=evaluate(0.035,-0.023,repeated_metric);
  require(std::abs(repeated_a.energy-repeated_b.energy)<1e-12 &&
          std::abs(repeated_a.orbital-repeated_b.orbital)<1e-12 &&
          std::abs(repeated_a.ci-repeated_b.ci)<1e-12,
          "same-frame point-owned metric changed repeated transport gradient");
  const double orbital_fd=(evaluate(0.035+step,-0.023).energy-
                           evaluate(0.035-step,-0.023).energy)/(2*step);
  const double ci_fd=(evaluate(0.035,-0.023+step).energy-
                      evaluate(0.035,-0.023-step).energy)/(2*step);
  require(std::abs(model.orbital-orbital_fd)<2e-5,
          "finite-residual transport orbital VJP fails model difference");
  require(std::abs(model.ci-ci_fd)<2e-5,
          "finite-residual transport CI VJP fails model difference");
  const double origin=evaluate(0.0,0.0).energy;
  const double second_orbital=(evaluate(1e-3,0.0).energy+
      evaluate(-1e-3,0.0).energy-2*origin)/1e-6;
  require(std::abs(second_orbital)<3e-4,
          "residual-balanced origin orbital Hessian was not affine");
  const double second_ci=(evaluate(0.0,1e-3).energy+
      evaluate(0.0,-1e-3).energy-2*origin)/1e-6;
  require(std::abs(second_ci-u.dot(b*u))<3e-4,
          "residual-balanced origin CI Hessian differs from affine model");
  const double mixed=(evaluate(1e-3,1e-3).energy-
      evaluate(1e-3,-1e-3).energy-
      evaluate(-1e-3,1e-3).energy+
      evaluate(-1e-3,-1e-3).energy)/(4e-6);
  require(std::abs(mixed)<3e-4,
          "residual-balanced origin mixed Hessian was not affine");
}

void check_wide_transport_sources() {
  // Unequal spin sectors and >128 source columns exercise every register
  // lane, the second RHS batch, sparse incidence and the final short tile.
  constexpr int n=10,structures=7;
  Strings alpha,beta,products_alpha,products_beta;
  for (unsigned mask=0;mask<(1u<<n);++mask) {
    if (std::popcount(mask)!=5 && std::popcount(mask)!=4) continue;
    std::vector<int> occupied;
    for (int p=0;p<n;++p) if (mask&(1u<<p)) occupied.push_back(p);
    (occupied.size()==5 ? alpha:beta).push_back(std::move(occupied));
  }
  Terms terms(alpha.size()*beta.size());
  for (const auto& a:alpha) for (const auto& b:beta) {
    products_alpha.push_back(a);products_beta.push_back(b);
  }
  for (int b=0;b<static_cast<int>(beta.size());++b)
    for (int path=0;path<3;++path) {
      auto& group=terms[((3*b+path)%alpha.size())*beta.size()+b];
      group.push_back({b%structures,1.0});
      group.push_back({b%structures,0.17});
      group.push_back({(b+3)%structures,-0.25});
    }
  Matrix origin=Matrix::Identity(n,n),direction=Matrix::Zero(n,n);
  for (int j=0;j<n;++j) for (int i=0;i<=j;++i) {
    if (i==j) origin(i,j)=1.02+0.002*i;
    else origin(i,j)=0.01*std::sin(0.23*(i+1)*(j+2));
    direction(i,j)=0.015*std::cos(0.31*(i+2)*(j+1));
  }
  const Matrix current=origin+0.03*direction,zero=Matrix::Zero(n,n);
  const Matrix gram=origin.transpose()*origin;
  const std::vector<double> flat(gram.data(),gram.data()+gram.size());
  xmvb::integrals::ActiveSpaceTwoElectronResult eri;
  eri.packed_active_two_electron_integrals.resize(
      xmvb::integrals::packed_active_two_electron_integral_count(n),0.0);
  const auto topology=xmvb::vb::build_orthogonal_spin_topology(products_alpha,products_beta,n);
  const Matrix h=Matrix::Identity(n,n);
  const xmvb::vb::StructureAction cpu(terms,structures,topology,flat,h,eri,n,
      nullptr,xmvb::vb::StructureActionMode::ActionOnly);
  xmvb::vb::RepresentationOverlap reference(cpu,
      topology.alpha_reuse_table.unique_determinants,
      topology.beta_reuse_table.unique_determinants,n);
  xmvb::cuda::Context context(256ULL*1024*1024);
  xmvb::cuda::StructureAction action(context,cpu);
  xmvb::cuda::SelectedCrossOverlap overlap(context,action,n);
  Vector lambda(structures),base(structures),ref(structures),transported(structures);
  for (int i=0;i<structures;++i) {
    lambda[i]=std::sin(0.31*(i+1));base[i]=std::cos(0.27*(i+2));
    ref[i]=std::sin(0.19*(i+3));transported[i]=std::cos(0.13*(i+4));
  }
  const auto upload=[&](const Matrix& x) {
    xmvb::cuda::Matrix<double> result(context,x.rows(),x.cols());
    xmvb::cuda::copy_to_device(context,result.view(),x);return result;
  };
  auto dcurrent=upload(current),dorigin=upload(origin),ddirection=upload(direction);
  auto dlambda=upload(lambda),dbase=upload(base),dref=upload(ref),dt=upload(transported);
  xmvb::cuda::Matrix<double> left(context,n,n),right(context,n,n),image(context,structures,1);
  overlap.transport_gradient(dcurrent.view(),dorigin.view(),ddirection.view(),
      dlambda.view(),dbase.view(),dref.view(),dt.view(),left.view(),right.view(),image.view());
  const auto a=reference.adjoint(current,origin,lambda,base);
  const auto b=reference.adjoint_direction(current,origin,zero,direction,lambda,ref);
  const auto m=reference.adjoint(current,current,lambda,transported);
  const auto o=reference.adjoint(current,origin,lambda,ref);
  near(download(context,left.view()),a.left+b.left-m.left-m.right,4e-8,
      "wide source current cotangent differs from CPU");
  near(download(context,right.view()),o.right,4e-8,
      "wide source origin cotangent differs from CPU");
  near(download(context,image.view()),reference.apply(origin,current,lambda),3e-9,
      "wide source image differs from CPU");
}

}  // namespace

int main() {
  try {
    check_selected_transport();
    check_wide_transport_sources();
    std::cout<<"CUDA selected CI transport and residual-balanced model: PASS\n";
  } catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
  }
}
