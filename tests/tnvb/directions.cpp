#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "input/loading/loader.hpp"
#include "tnvb/electronic/evaluator.hpp"
#include "vbscf/derivatives/gradient/orbital/evaluator.hpp"
#include "vbscf/orbitals/charts/layout.hpp"
#include "integrals/active/raw.hpp"
#include "vbscf/structures/orthogonal_ci/integrals.hpp"

namespace {

double dot(const std::vector<double>& a,const Eigen::Ref<const Eigen::VectorXd>& b) {
  if (a.size()!=static_cast<std::size_t>(b.size())) throw std::invalid_argument("direction dimension mismatch");
  return Eigen::Map<const Eigen::VectorXd>(a.data(),a.size()).dot(b);
}

Eigen::VectorXd symmetric_direction(int n,double phase) {
  Eigen::MatrixXd matrix(n,n);
  for (int j=0;j<n;++j) for (int i=0;i<n;++i) matrix(i,j)=std::sin(phase+2.3*i+3.7*j);
  matrix=(0.5*(matrix+matrix.transpose())).eval(); matrix/=matrix.norm();
  return Eigen::Map<const Eigen::VectorXd>(matrix.data(),matrix.size());
}

void materialize_active_eri(xmvb::integrals::ActiveSpaceTwoElectronResult* value,int n) {
  if (!value->packed_active_two_electron_integrals.empty()) return;
  const auto& factors=value->ri_active_pair_factors;
  const int np=n*(n+1)/2;
  if (factors.cols()!=np) throw std::runtime_error("invalid RI factor dimension for integral FD");
  value->packed_active_two_electron_integrals.resize(static_cast<std::size_t>(np)*(np+1)/2);
  for (int j=0;j<np;++j) for (int i=0;i<=j;++i) {
    value->packed_active_two_electron_integrals[j*(j+1)/2+i]=factors.col(i).dot(factors.col(j));
  }
}

void reframe_raw_trial(xmvb::vb::PreparedActiveSpaceContext* trial,int n) {
  auto& two=trial->active_space_two_electron_result;
  const auto framed=xmvb::vb::orthogonalize_active_integrals(
      trial->orbital_result.active_orbital_overlap_matrix,
      trial->active_space_one_electron_result.h1e_act,two,n);
  trial->active_space_one_electron_result.h1e_act=framed.one_electron;
  two.orbital_transform=framed.orbital_transform;
  const int np=n*(n+1)/2;
  two.packed_active_two_electron_integrals.resize(static_cast<std::size_t>(np)*(np+1)/2);
  for (int j=0;j<np;++j) for (int i=0;i<=j;++i)
    two.packed_active_two_electron_integrals[j*(j+1)/2+i]=framed.pair_kernel(i,j);
}

}  // namespace

int main(int argc,char** argv) {
  try {
    if (argc!=4) throw std::invalid_argument("usage: check_tnvb_directions INPUT REPORT.json exact|ri");
    xmvb::vb::VbScfInputLoadOptions options;
    if (std::string(argv[3])=="ri") options.standard_two_electron_mode=xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
    else if (std::string(argv[3])=="exact") options.standard_two_electron_mode=xmvb::vb::StandardTwoElectronMode::Exact;
    else throw std::invalid_argument("invalid integral model");
    const auto loaded=xmvb::vb::load_vbscf_input_with_timings(argv[1],options);
    const auto& input=loaded.input;
    xmvb::vb::ElectronicStateRequest request;
    request.nuclear_repulsion_energy=loaded.nuclear_repulsion_energy;
    xmvb::tnvb::ElectronicEvaluator backend;
    auto active=backend.evaluate_energy_gradient(input,request);
    const int n=input.orbital_preparation_input.n_active_orbitals;
    xmvb::vb::PreparedActiveSpaceContext prepared;
    prepared.orbital_result=active.orbital_preparation_result;
    prepared.ao_effective_one_electron_result=active.ao_effective_one_electron_result;
    prepared.active_space_one_electron_result=active.active_space_one_electron_result;
    prepared.active_space_two_electron_result=active.active_space_two_electron_result;
    prepared.one_electron_reference_energy=active.scf_result.one_electron_reference_energy;
    materialize_active_eri(&prepared.active_space_two_electron_result,n);
    const std::array<std::vector<double>,3> adjoints{
        active.active_orbital_overlap_gradient,active.active_one_electron_gradient,
        active.packed_active_two_electron_gradient};
    auto orbital=xmvb::vb::OrbitalGradientEvaluator{}.evaluate_without_reference_energy_gradient(input,std::move(active));
    const std::array<double,3> steps{1e-4,3e-5,1e-5};
    std::ofstream output(argv[2]);
    if (!output) throw std::runtime_error("cannot create directional FD report");
    output << std::setprecision(17) << "{\n\"integrals\":\"" << argv[3] << "\",\n\"directions\":[\n";
    const std::array<const char*,5> labels{"S","h","packed_g","inactive_orbitals","active_orbitals"};
    xmvb::tnvb::WorkerProcess process;
    bool all_passed=true;
    for (int component=0;component<5;++component) {
      Eigen::VectorXd direction;
      xmvb::vb::SparseParameterLayout layout(input.orbital_preparation_input);
      const Eigen::VectorXd initial=layout.pack(input.orbital_preparation_input);
      if (component<2) direction=symmetric_direction(n,0.4+component);
      else if (component==2) {
        direction.resize(adjoints[2].size());
        for (int i=0;i<direction.size();++i) direction[i]=std::sin(1.1+0.73*i);
        direction.normalize();
      } else {
        direction=Eigen::VectorXd::Zero(layout.size());
        const int ni=(input.orbital_preparation_input.n_total_electrons-input.orbital_preparation_input.n_active_electrons)/2;
        const int nb=input.orbital_preparation_input.n_basis_functions;
        for (int i=0;i<layout.size();++i) {
          const int orbital_index=layout.differentiable_parameter_indices()[i]/nb;
          if ((component==3 && orbital_index<ni) || (component==4 && orbital_index>=ni)) {
            direction[i]=std::sin(0.8+0.27*i);
          }
        }
        if (direction.norm()==0) throw std::runtime_error("missing orbital direction sector");
        direction.normalize();
      }
      const double analytic=component<3 ? dot(adjoints[component],direction)
          : layout.gather_from_full(orbital.sparse_orbital_energy_gradient).dot(direction);
      std::array<double,3> errors{};
      output << (component ? ",\n" : "") << "{\"component\":\"" << labels[component]
             << "\",\"analytic\":" << analytic << ",\"steps\":[";
      for (int istep=0;istep<3;++istep) {
        std::array<double,2> values{};
        for (int isign=0;isign<2;++isign) {
          const double amount=(isign ? 1.0 : -1.0)*steps[istep];
          if (component<3) {
            auto trial=prepared;
            if (component<2) {
              trial.active_space_one_electron_result.h1e_act=
                  xmvb::integrals::materialize_raw_one_electron(
                      prepared.active_space_one_electron_result.h1e_act,
                      prepared.active_space_two_electron_result.orbital_transform);
              trial.active_space_two_electron_result=
                  xmvb::integrals::materialize_raw_active_integrals(
                      prepared.active_space_two_electron_result);
            }
            if (component==0) {
              Eigen::Map<Eigen::VectorXd>(trial.orbital_result.active_orbital_overlap_matrix.data(),n*n)+=amount*direction;
            } else if (component==1) {
              auto& h=trial.active_space_one_electron_result.h1e_act;
              Eigen::Map<Eigen::VectorXd>(h.data(),h.size())+=amount*direction;
            } else {
              auto& g=trial.active_space_two_electron_result.packed_active_two_electron_integrals;
              Eigen::Map<Eigen::VectorXd>(g.data(),g.size())+=amount*direction;
            }
            if (component<2) reframe_raw_trial(&trial,n);
            values[isign]=process.evaluate(input,trial,request,false).energies[0];
          } else {
            auto trial=input;
            layout.unpack(initial+amount*direction,&trial.orbital_preparation_input);
            values[isign]=backend.evaluate_energy_result(trial,request).scf_result.total_energy;
          }
        }
        const double numerical=(values[1]-values[0])/(2*steps[istep]);
        errors[istep]=std::abs(numerical-analytic);
        output << (istep ? "," : "") << "{\"epsilon\":" << steps[istep]
               << ",\"numerical\":" << numerical << ",\"error\":" << errors[istep] << '}';
      }
      // Two adjacent step sizes must pass; a lucky single finite difference
      // cannot establish a stable precision interval.
      const bool stable=(errors[0]<=1e-6 && errors[1]<=1e-6) ||
                        (errors[1]<=1e-6 && errors[2]<=1e-6);
      all_passed=all_passed && stable && std::isfinite(analytic+errors[0]+errors[1]+errors[2]);
      output << "],\"stable\":" << (stable ? "true" : "false") << '}';
      std::cout << labels[component] << " analytic=" << std::setprecision(17) << analytic
                << " errors=" << errors[0] << ',' << errors[1] << ',' << errors[2]
                << " stable=" << stable << '\n';
    }
    output << "\n],\n\"passed\":" << (all_passed ? "true" : "false") << "\n}\n";
    return all_passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "TNVB directional check failed: " << error.what() << '\n'; return 2;
  }
}
