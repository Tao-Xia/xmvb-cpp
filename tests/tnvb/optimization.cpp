#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/resource.h>

#include <Eigen/Eigenvalues>

#include "input/loading/loader.hpp"
#include "tnvb/electronic/evaluator.hpp"
#include "vbscf/derivatives/gradient/active_space/determinant_backend.hpp"
#include "vbscf/derivatives/gradient/orbital/evaluator.hpp"
#include "vbscf/optimization/driver/optimizer.hpp"
#include "vbscf/structures/expansion/expander.hpp"
#include "acceptance.hpp"

namespace {

struct State {
  Eigen::MatrixXd h,s,weights;
  Eigen::VectorXd c;
};

State normalize(const xmvb::vb::VbScfResult& result) {
  const int n=result.n_structures;
  State value;
  const Eigen::Map<const Eigen::MatrixXd> h(result.structure_matrices.hamiltonian_matrix.data(),n,n);
  const Eigen::Map<const Eigen::MatrixXd> s(result.structure_matrices.overlap_matrix.data(),n,n);
  const Eigen::VectorXd norms=s.diagonal().cwiseSqrt();
  value.h=norms.cwiseInverse().asDiagonal()*h*norms.cwiseInverse().asDiagonal();
  value.s=norms.cwiseInverse().asDiagonal()*s*norms.cwiseInverse().asDiagonal();
  value.c=norms.asDiagonal()*Eigen::Map<const Eigen::VectorXd>(result.eigenvector_matrix.data(),n);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(value.s);
  if (eig.info()!=Eigen::Success || eig.eigenvalues().minCoeff()<=0) throw std::runtime_error("invalid final structure metric");
  const Eigen::MatrixXd root=eig.eigenvectors()*eig.eigenvalues().cwiseSqrt().asDiagonal()*eig.eigenvectors().transpose();
  const Eigen::MatrixXd inverse=eig.eigenvectors()*eig.eigenvalues().cwiseInverse().asDiagonal()*eig.eigenvectors().transpose();
  value.weights.resize(n,4);
  value.weights.col(0)=value.c.array()*(value.s*value.c).array()/value.c.dot(value.s*value.c);
  value.weights.col(1)=(root*value.c).array().square();
  value.weights.col(2)=value.c.array().square()/inverse.diagonal().array();
  value.weights.col(3)=value.c.array().square();
  for (int j=1;j<4;++j) value.weights.col(j)/=value.weights.col(j).sum();
  return value;
}

void write_array(std::ostream& out,const double* values,std::size_t count) {
  out << '[';
  for (std::size_t i=0;i<count;++i) out << (i?",":"") << values[i];
  out << ']';
}

void write_result(std::ostream& out,const xmvb::vb::VbScfOptimizerResult& r) {
  const auto state=normalize(r.scf_result);
  out << "{\"converged\":" << (r.converged?"true":"false")
      << ",\"termination\":\"" << r.termination_reason << "\",\"iterations\":" << r.n_iterations
      << ",\"energy\":" << r.final_total_energy << ",\"gradient_l2\":" << r.final_gradient_l2_norm
      << ",\"projected_gradient_inf\":" << r.final_projected_gradient_inf_norm
      << ",\"projected_gradient_l2\":" << r.final_projected_gradient_l2_norm
      << ",\"wall_seconds\":" << r.total_wall_time_seconds << ",\"coefficients\":";
  write_array(out,state.c.data(),state.c.size());
  out << ",\"weights_column_major\":"; write_array(out,state.weights.data(),state.weights.size());
  out << ",\"orbital_value_table\":";
  const auto& orbitals=r.optimized_input.orbital_preparation_input.orbital_value_table;
  write_array(out,orbitals.data(),orbitals.size());
  out << ",\"energy_history\":"; write_array(out,r.total_energy_history.data(),r.total_energy_history.size());
  if (r.scf_result.tnvb_diagnostics) {
    const auto& diagnostic=*r.scf_result.tnvb_diagnostics;
    out << ",\"fit_energy_error\":" << diagnostic.fit_energy_error
        << ",\"rdm_identity_error\":" << diagnostic.rdm_identity_error;
  }
  out << '}';
}

struct CrossCheck {
  double energy=0,h=0,s=0,c=0,weight=0,gradient=0,fit=0;
  double tensor_energy=0,reference_energy=0;
  bool passed=false,matrix_required=true,strict_matrices_passed=false;
};

CrossCheck compare_endpoint(xmvb::vb::VbScfInput input,
    const xmvb::vb::StructureExpansion& expanded,
    const xmvb::vb::TnVbConfiguration& config,
    const xmvb::vb::ElectronicStateRequest& request,
    const xmvb::tnvb::ElectronicEvaluator& backend) {
  input.tnvb=config;
  input.structure_data={};
  auto tensor_active=backend.evaluate_energy_gradient(input,request);
  if (tensor_active.second_order_context) throw std::runtime_error("TNVB endpoint created a second-order determinant context");
  input.tnvb.enabled=false;
  input.structure_data=expanded;
  auto reference_active=xmvb::vb::DeterminantElectronicStateBackend{}.evaluate_energy_gradient(input,request);
  xmvb::vb::OrbitalGradientEvaluator pullback;
  input.tnvb=config;
  input.structure_data={};
  auto tensor=pullback.evaluate_without_reference_energy_gradient(input,std::move(tensor_active));
  input.tnvb.enabled=false;
  input.structure_data=expanded;
  auto reference=pullback.evaluate_without_reference_energy_gradient(input,std::move(reference_active));
  auto t=normalize(tensor.scf_result),d=normalize(reference.scf_result);
  if (t.c.dot(d.c)<0) t.c=-t.c;
  CrossCheck error;
  error.tensor_energy=tensor.scf_result.total_energy;
  error.reference_energy=reference.scf_result.total_energy;
  error.energy=std::abs(tensor.scf_result.total_energy-reference.scf_result.total_energy);
  error.h=(t.h-d.h).cwiseAbs().maxCoeff(); error.s=(t.s-d.s).cwiseAbs().maxCoeff();
  error.c=(t.c-d.c).cwiseAbs().maxCoeff(); error.weight=(t.weights-d.weights).cwiseAbs().maxCoeff();
  const auto& tg=tensor.sparse_orbital_energy_gradient;
  const auto& dg=reference.sparse_orbital_energy_gradient;
  if (tg.size()!=dg.size()) throw std::runtime_error("endpoint gradient size mismatch");
  for (std::size_t i=0;i<tg.size();++i) error.gradient=std::max(error.gradient,std::abs(tg[i]-dg[i]));
  error.fit=tensor.scf_result.tnvb_diagnostics->fit_energy_error;
  const auto profile=tnvb_test::acceptance_profile(input.orbital_preparation_input.n_active_orbitals);
  const tnvb_test::ComparisonErrors errors{error.energy,error.h,error.s,error.c,error.weight,error.gradient};
  error.matrix_required=profile.matrices_required;
  error.strict_matrices_passed=profile.strict_matrices_passed(errors);
  error.passed=std::isfinite(error.fit) && profile.passed(errors);
  return error;
}

void write_cross(std::ostream& out,const CrossCheck& e) {
  out << "{\"passed\":" << (e.passed?"true":"false")
      << ",\"matrix_comparison_required\":" << (e.matrix_required?"true":"false")
      << ",\"strict_matrix_check_passed\":" << (e.strict_matrices_passed?"true":"false")
      << ",\"energy_error\":" << e.energy << ",\"H_error\":" << e.h << ",\"S_error\":" << e.s
      << ",\"coefficient_error\":" << e.c << ",\"weight_error\":" << e.weight
      << ",\"gradient_error\":" << e.gradient << ",\"fit_energy_error\":" << e.fit
      << ",\"tnvb_energy\":" << e.tensor_energy << ",\"reference_energy\":" << e.reference_energy << '}';
}

}  // namespace

int main(int argc,char** argv) {
  try {
    if (argc!=5) throw std::invalid_argument("usage: optimize_tnvb INPUT REPORT.json exact|ri MAX_ITERATIONS (original optimizer criteria only; C14: <=200)");
    xmvb::vb::VbScfInputLoadOptions load_options;
    const std::string mode=argv[3];
    if (mode=="exact") load_options.standard_two_electron_mode=xmvb::vb::StandardTwoElectronMode::Exact;
    else if (mode=="ri") load_options.standard_two_electron_mode=xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
    else throw std::invalid_argument("invalid integral mode");
    std::size_t consumed=0;
    const int iterations=std::stoi(argv[4],&consumed);
    if (consumed!=std::string(argv[4]).size() || iterations<1 || iterations>1000) throw std::invalid_argument("invalid bounded iteration count");
    auto loaded=xmvb::vb::load_vbscf_input_with_timings(argv[1],load_options);
    const auto& input=loaded.input;
    const int k=input.orbital_preparation_input.n_active_orbitals;
    if (k==14 && iterations>200) throw std::invalid_argument("C14 acceptance optimization is limited to 200 steps");
    if (!input.tnvb.enabled || (k!=6 && k!=14) || input.complete_active_space ||
        loaded.structure_expansion_seconds!=0 || !input.structure_data.pair_alpha.empty()) {
      throw std::runtime_error("acceptance optimization requires unexpanded TNVB benzene or C14 input");
    }
    const auto expanded=xmvb::vb::StructureExpander{}.expand(loaded.raw_structure_data);
    auto reference_input=input;
    reference_input.tnvb.enabled=false;
    reference_input.structure_data=expanded;
    xmvb::vb::VbScfOptimizerOptions options;
    if (loaded.scf_optimizer!=xmvb::vb::InputScfOptimizer::Plbfgs) {
      throw std::invalid_argument("TNVB optimization acceptance requires explicit input ISCF=8");
    }
    options.backend=xmvb::vb::VbScfOptimizerBackend::Plbfgs;
    options.structure_eigensolver=xmvb::vb::StructureEigensolver::Dense;
    options.max_iterations=iterations;
    options.verbose=false;
    // Keep the existing options' convergence defaults for both backends.
    // Alternative tolerances are not an acceptance-tool capability.
    const auto callback=[](const char* name) {
      return [name](const xmvb::vb::VbScfAcceptedIterationSnapshot& s) {
        std::cout << std::setprecision(17) << "[OPT-" << name << "] i=" << s.accepted_iteration_index
          << " E=" << s.total_energy << " gradient_l2=" << s.sparse_orbital_energy_gradient_l2_norm;
        if (s.has_projected_gradient) std::cout << " projected_gradient_inf=" << s.projected_gradient_inf_norm;
        std::cout << std::endl;
      };
    };
    options.accepted_iteration_callback=callback("REFERENCE");
    auto reference=xmvb::vb::VbScfOptimizer{options}.optimize(std::move(reference_input),loaded.nuclear_repulsion_energy);
    // Preserve a machine-readable reference even if the tensor worker later fails.
    {
      std::ofstream checkpoint(std::string(argv[2])+".reference.json");
      if (!checkpoint) throw std::runtime_error("cannot write reference checkpoint");
      checkpoint << std::setprecision(17); write_result(checkpoint,reference); checkpoint << '\n';
    }
    options.accepted_iteration_callback=callback("TNVB");
    auto tensor_backend=std::make_shared<xmvb::tnvb::ElectronicEvaluator>();
    auto tensor=xmvb::vb::VbScfOptimizer{options,tensor_backend}.optimize(input,loaded.nuclear_repulsion_energy);
    xmvb::vb::ElectronicStateRequest request;
    request.nuclear_repulsion_energy=loaded.nuclear_repulsion_energy;
    const auto tensor_endpoint=compare_endpoint(tensor.optimized_input,expanded,input.tnvb,request,*tensor_backend);
    auto ref_endpoint_input=reference.optimized_input;
    ref_endpoint_input.raw_structure_data=input.raw_structure_data;
    const auto reference_endpoint=compare_endpoint(std::move(ref_endpoint_input),expanded,input.tnvb,request,*tensor_backend);
    auto t=normalize(tensor.scf_result),d=normalize(reference.scf_result);
    if (t.c.dot(d.c)<0) t.c=-t.c;
    const double de=std::abs(tensor.final_total_energy-reference.final_total_energy);
    const double dc=(t.c-d.c).cwiseAbs().maxCoeff(),dw=(t.weights-d.weights).cwiseAbs().maxCoeff();
    // User-confirmed cross-backend optimized-energy target; this is not
    // the optimizer's energy-change or gradient convergence tolerance.
    const double energy_limit=1e-6;
    const double coefficient_limit=k==6 ? 1e-8 : 1e-4;
    const double tensor_export_error=std::abs(tensor_endpoint.tensor_energy-tensor.final_total_energy);
    const double reference_export_error=std::abs(reference_endpoint.reference_energy-reference.final_total_energy);
    const bool passed=tensor.converged && reference.converged &&
        tensor_endpoint.passed && reference_endpoint.passed && de<=energy_limit &&
        (k==6 || (dc<=coefficient_limit && dw<=coefficient_limit)) && tensor_endpoint.fit<=1e-7 &&
        tensor_export_error<=1e-9 && reference_export_error<=1e-9;
    rusage self{},children{};
    getrusage(RUSAGE_SELF,&self); getrusage(RUSAGE_CHILDREN,&children);
    std::ofstream report(argv[2]);
    if (!report) throw std::runtime_error("cannot write optimization report");
    report << std::setprecision(17) << "{\n\"passed\":" << (passed?"true":"false")
        << ",\n\"integrals\":\"" << mode << "\",\n\"max_iterations\":" << iterations
        << ",\n\"convergence_profile\":\"original\""
        << ",\n\"optimizer\":\"" << xmvb::vb::vbscf_optimizer_backend_name(options.backend) << "\""
        << ",\n\"energy_tolerance\":" << options.energy_tolerance
        << ",\n\"gradient_tolerance\":" << options.gradient_tolerance
        << ",\n\"gradient_metric\":\"projected_inf\"";
    report
        << ",\n\"final_energy_error_limit\":" << energy_limit
        << ",\n\"final_energy_error\":" << de << ",\n\"final_coefficient_error\":" << dc
        << ",\n\"final_weight_error\":" << dw << ",\n\"parent_peak_rss_kib\":" << self.ru_maxrss
        << ",\n\"tnvb_export_energy_error\":" << tensor_export_error
        << ",\n\"reference_export_energy_error\":" << reference_export_error
        << ",\n\"largest_reaped_child_peak_rss_kib\":" << children.ru_maxrss
        << ",\n\"reference\":"; write_result(report,reference);
    report << ",\n\"tnvb\":"; write_result(report,tensor);
    report << ",\n\"tnvb_endpoint_crosscheck\":"; write_cross(report,tensor_endpoint);
    report << ",\n\"reference_endpoint_crosscheck\":"; write_cross(report,reference_endpoint);
    report << "\n}\n";
    std::cout << std::setprecision(17) << "optimization comparison: dE=" << de << " dc=" << dc
        << " dw=" << dw << " converged=" << reference.converged << '/' << tensor.converged
        << " passed=" << passed << std::endl;
    return passed?0:1;
  } catch (const std::exception& error) {
    std::cerr << "optimization comparison failed: " << error.what() << std::endl;
    return 2;
  }
}
