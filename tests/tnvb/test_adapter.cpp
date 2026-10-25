#include "tnvb/gradient/adapter.hpp"
#include "tnvb/electronic/input_signature.hpp"
#include <cmath>
#include <stdexcept>
#include <limits>
#include "acceptance.hpp"

int main() {
  // Producer-chunk AO integrals are part of the scientific input identity.
  // A changed value or address must invalidate ENERGY-to-GRADIENT reuse.
  xmvb::vb::VbScfInput chunked;
  auto& graph=chunked.ao_integral_input.pair_graph;
  graph.pair_first={0,0,1}; graph.pair_second={0,1,1};
  graph.chunks={{{0},{0},{0.3}}}; graph.chunk_offsets={0,1};
  const auto original_signature=xmvb::tnvb::input_signature(chunked);
  if (xmvb::tnvb::input_signature(chunked)!=original_signature) {
    throw std::runtime_error("unchanged chunked AO input changed its signature");
  }
  for (int field=0;field<3;++field) {
    auto changed=chunked;
    auto& chunk=changed.ao_integral_input.pair_graph.chunks.front();
    if (field==0) chunk.rows[0]=1;
    if (field==1) chunk.columns[0]=1;
    if (field==2) chunk.values[0]+=1e-4;
    if (xmvb::tnvb::input_signature(changed)==original_signature) {
      throw std::runtime_error("changed chunked AO input retained its signature");
    }
  }
  // Preserve benzene's strict matrix gate while applying the separately
  // confirmed C14 policy; nonfinite diagnostics can never pass either profile.
  const tnvb_test::ComparisonErrors measured{3.23e-11,3.18e-9,1.524e-10,7.05e-10,9.11e-10,1.75e-9};
  if (tnvb_test::acceptance_profile(6).passed(measured) ||
      !tnvb_test::acceptance_profile(14).passed(measured) ||
      tnvb_test::AcceptanceProfile::strict_matrices_passed(measured)) {
    throw std::runtime_error("benzene/C14 acceptance scopes were mixed");
  }
  auto invalid=measured;
  invalid.s=std::numeric_limits<double>::quiet_NaN();
  if (tnvb_test::acceptance_profile(14).passed(invalid)) {
    throw std::runtime_error("nonfinite C14 matrix diagnostic bypassed validation");
  }
  // An orthonormal two-electron closed-shell determinant on orbital zero.
  // E=2h00+g0000. At fixed h/g its norm derivative is -2h00-2g0000.
  xmvb::tnvb::WorkerResponse response;
  response.energies=Eigen::VectorXd::Constant(1,-1.7);
  response.density_one=Eigen::MatrixXd::Zero(2,2);
  response.density_one(0,0)=2.0;
  response.density_two_buffer.assign(16,0.0);
  response.density_two_buffer[0]=2.0;
  response.contracted_three=Eigen::MatrixXd::Zero(2,2);
  xmvb::vb::ActiveSpaceGradientResult result;
  result.active_space_one_electron_result.h1e_act=Eigen::MatrixXd::Zero(2,2);
  result.active_space_one_electron_result.h1e_act(0,0)=-1.0;
  result.active_space_two_electron_result.orbital_transform=Eigen::MatrixXd::Identity(2,2);
  xmvb::tnvb::populate_active_space_adjoints(response,&result);
  if (std::abs(result.active_orbital_overlap_gradient[0]-1.4)>1e-14 ||
      result.active_one_electron_gradient[0]!=2.0 ||
      result.packed_active_two_electron_gradient[0]!=1.0) {
    throw std::runtime_error("two-electron TNVB adjoint convention failed");
  }
  // Independent coordinate check: transform a generic rank-four density to
  // the raw frame, evaluate the original formula, and compare to the new
  // two-index-only pullback. A nondiagonal R detects accidental identity or
  // transpose assumptions and mixed h/G frames.
  Eigen::MatrixXd r(2,2);r<<1.2,0.35,0.0,0.8;
  const Eigen::MatrixXd q=r.triangularView<Eigen::Upper>().solve(Eigen::MatrixXd::Identity(2,2));
  result.active_space_two_electron_result.orbital_transform=r;
  auto& h=result.active_space_one_electron_result.h1e_act;
  h<<-0.9,0.17,0.17,-0.6;
  response.density_one<<1.3,0.12,0.12,0.7;
  response.contracted_three<<0.3,0.04,0.04,0.2;
  for (int i=0;i<16;++i) response.density_two_buffer[i]=std::sin(0.3+0.21*i);
  const auto index=[](int i,int j,int k,int l){return i+2*(j+2*(k+2*l));};
  std::vector<double> raw_pi(16,0.0);
  for (int i=0;i<2;++i) for (int j=0;j<2;++j)
    for (int k=0;k<2;++k) for (int l=0;l<2;++l)
      for (int a=0;a<2;++a) for (int b=0;b<2;++b)
        for (int c=0;c<2;++c) for (int d=0;d<2;++d)
          raw_pi[index(i,j,k,l)]+=q(i,a)*q(j,b)*q(k,c)*q(l,d)*
              response.density_two_buffer[index(a,b,c,d)];
  const Eigen::MatrixXd raw_h=r.transpose()*h*r;
  const Eigen::MatrixXd raw_p=q*response.density_one*q.transpose();
  Eigen::MatrixXd reference=0.5*q*response.contracted_three*q.transpose()-response.energies[0]*raw_p;
  for (int i=0;i<2;++i) for (int j=0;j<2;++j)
    for (int u=0;u<2;++u) for (int v=0;v<2;++v)
      reference(i,j)+=raw_h(u,v)*raw_pi[index(v,i,u,j)];
  reference=(0.5*(reference+reference.transpose())).eval();
  xmvb::tnvb::populate_active_space_adjoints(response,&result);
  const Eigen::Map<const Eigen::MatrixXd> actual_s(result.active_orbital_overlap_gradient.data(),2,2);
  const Eigen::Map<const Eigen::MatrixXd> actual_h(result.active_one_electron_gradient.data(),2,2);
  if ((actual_s-reference).cwiseAbs().maxCoeff()>2e-14 ||
      (actual_h-raw_p).cwiseAbs().maxCoeff()>2e-14 ||
      std::abs(result.packed_active_two_electron_gradient[0]-0.5*response.density_two_buffer[0])>1e-14) {
    throw std::runtime_error("framed TNVB cotangent does not match the raw-coordinate reference");
  }
  return 0;
}
