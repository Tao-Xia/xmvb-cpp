#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Eigenvalues>

#include "input/loading/loader.hpp"
#include "tnvb/electronic/evaluator.hpp"
#include "vbscf/derivatives/gradient/active_space/determinant_backend.hpp"
#include "vbscf/derivatives/gradient/orbital/evaluator.hpp"
#include "vbscf/structures/expansion/expander.hpp"
#include "vbscf/workflow/evaluator.hpp"
#include "acceptance.hpp"

namespace {

struct NormalizedState {
  Eigen::MatrixXd h, s;
  Eigen::VectorXd c;
};

NormalizedState normalize(const xmvb::vb::VbScfResult& value) {
  const int n=value.n_structures;
  NormalizedState state;
  const Eigen::Map<const Eigen::MatrixXd> h(value.structure_matrices.hamiltonian_matrix.data(),n,n);
  const Eigen::Map<const Eigen::MatrixXd> s(value.structure_matrices.overlap_matrix.data(),n,n);
  const Eigen::VectorXd norms=s.diagonal().cwiseSqrt();
  state.h=norms.cwiseInverse().asDiagonal()*h*norms.cwiseInverse().asDiagonal();
  state.s=norms.cwiseInverse().asDiagonal()*s*norms.cwiseInverse().asDiagonal();
  state.c=norms.asDiagonal()*Eigen::Map<const Eigen::VectorXd>(value.eigenvector_matrix.data(),n);
  return state;
}

Eigen::MatrixXd weights(const NormalizedState& state) {
  Eigen::MatrixXd result(state.c.size(),4);
  const double norm=state.c.dot(state.s*state.c);
  result.col(0)=state.c.array()*(state.s*state.c).array()/norm;
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(state.s);
  if (solver.info()!=Eigen::Success || solver.eigenvalues().minCoeff()<=0) {
    throw std::runtime_error("invalid normalized structure metric in comparison");
  }
  const Eigen::MatrixXd root=solver.eigenvectors()*solver.eigenvalues().cwiseSqrt().asDiagonal()*solver.eigenvectors().transpose();
  const Eigen::MatrixXd inverse=solver.eigenvectors()*solver.eigenvalues().cwiseInverse().asDiagonal()*solver.eigenvectors().transpose();
  result.col(1)=(root*state.c).array().square(); result.col(1)/=result.col(1).sum();
  result.col(2)=state.c.array().square()/inverse.diagonal().array(); result.col(2)/=result.col(2).sum();
  result.col(3)=state.c.array().square(); result.col(3)/=result.col(3).sum();
  return result;
}

double max_difference(const std::vector<double>& a,const std::vector<double>& b) {
  if (a.size()!=b.size()) throw std::runtime_error("comparison vector dimension mismatch");
  double result=0;
  for (std::size_t i=0;i<a.size();++i) result=std::max(result,std::abs(a[i]-b[i]));
  return result;
}

void write_vector(std::ostream& output, const Eigen::Ref<const Eigen::VectorXd>& v) {
  output << '[';
  for (int i=0;i<v.size();++i) output << (i ? "," : "") << v[i];
  output << ']';
}

}  // namespace

int main(int argc,char** argv) {
  try {
    if (argc!=5 && argc!=6) throw std::invalid_argument("usage: compare_tnvb INPUT REPORT.json exact|ri energy|gradient [ORBITAL_VALUES.txt]");
    const bool gradient=std::string(argv[4])=="gradient";
    if (!gradient && std::string(argv[4])!="energy") throw std::invalid_argument("invalid comparison mode");
    xmvb::vb::VbScfInputLoadOptions options;
    if (std::string(argv[3])=="ri") options.standard_two_electron_mode=xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
    else if (std::string(argv[3])=="exact") options.standard_two_electron_mode=xmvb::vb::StandardTwoElectronMode::Exact;
    else throw std::invalid_argument("invalid integral mode");
    auto loaded=xmvb::vb::load_vbscf_input_with_timings(argv[1],options);
    auto& tensor_input=loaded.input;
    const auto profile=tnvb_test::acceptance_profile(tensor_input.orbital_preparation_input.n_active_orbitals);
    if (argc==6) {
      std::ifstream orbitals(argv[5]);
      auto& table=tensor_input.orbital_preparation_input.orbital_value_table;
      for (double& value:table) if (!(orbitals>>value) || !std::isfinite(value)) {
        throw std::invalid_argument("invalid or truncated endpoint orbital table");
      }
      double extra=0;
      if (orbitals>>extra || !orbitals.eof()) throw std::invalid_argument("trailing or malformed endpoint orbital table");
    }
    if (!tensor_input.tnvb.enabled || !tensor_input.structure_data.pair_alpha.empty() ||
        !tensor_input.structure_data.pair_beta.empty() ||
        !tensor_input.structure_data.terms.empty() ||
        !tensor_input.structure_data.alpha_strings.empty() ||
        !tensor_input.structure_data.beta_strings.empty() ||
        !tensor_input.structure_data.offsets.empty() ||
        loaded.structure_expansion_seconds!=0.0 || tensor_input.complete_active_space) {
      throw std::runtime_error("TNVB load expanded determinants or incorrectly marked a complete CAS");
    }
    // Only the explicit small-system reference branch expands structures.
    xmvb::vb::VbScfInput determinant_input=tensor_input;
    determinant_input.tnvb.enabled=false;
    determinant_input.structure_data=xmvb::vb::StructureExpander{}.expand(loaded.raw_structure_data);
    xmvb::vb::ElectronicStateRequest request;
    request.nuclear_repulsion_energy=loaded.nuclear_repulsion_energy;
    request.eigensolver=loaded.structure_eigensolver;
    auto reference_request=request;
    reference_request.eigensolver=xmvb::vb::StructureEigensolver::Dense;
    xmvb::tnvb::ElectronicEvaluator tensor_backend;
    xmvb::vb::VbScfResult determinant,tensor;
    double gradient_error=0;
    if (gradient) {
      auto tensor_active=tensor_backend.evaluate_energy_gradient(tensor_input,request);
      if (tensor_active.second_order_context) throw std::runtime_error("TNVB created a determinant second-order cache");
      auto determinant_active=xmvb::vb::DeterminantElectronicStateBackend{}.evaluate_energy_gradient(determinant_input,reference_request);
      xmvb::vb::OrbitalGradientEvaluator pullback;
      const auto tensor_gradient=pullback.evaluate_without_reference_energy_gradient(tensor_input,std::move(tensor_active));
      const auto determinant_gradient=pullback.evaluate_without_reference_energy_gradient(determinant_input,std::move(determinant_active));
      gradient_error=max_difference(tensor_gradient.sparse_orbital_energy_gradient,determinant_gradient.sparse_orbital_energy_gradient);
      tensor=tensor_gradient.scf_result; determinant=determinant_gradient.scf_result;
    } else {
      auto value=tensor_backend.evaluate_energy_result(tensor_input,request);
      if (!value.active_one_electron_gradient.empty() ||
          !value.packed_active_two_electron_gradient.empty() || value.second_order_context) {
        throw std::runtime_error("TNVB ENERGY produced derivatives/second-order cache");
      }
      tensor=std::move(value.scf_result);
      determinant=xmvb::vb::VbScfEvaluator{}.evaluate(determinant_input,loaded.nuclear_repulsion_energy);
    }
    auto t=normalize(tensor), d=normalize(determinant);
    if (t.c.dot(d.c)<0) t.c=-t.c;
    const double de=std::abs(tensor.total_energy-determinant.total_energy);
    const double dh=(t.h-d.h).cwiseAbs().maxCoeff();
    const double ds=(t.s-d.s).cwiseAbs().maxCoeff();
    const double dc=(t.c-d.c).cwiseAbs().maxCoeff();
    const double dw=(weights(t)-weights(d)).cwiseAbs().maxCoeff();
    const tnvb_test::ComparisonErrors errors{de,dh,ds,dc,dw,gradient_error};
    const bool passed=profile.passed(errors);
    std::ofstream report(argv[2]);
    if (!report) throw std::runtime_error("cannot create comparison report");
    report << std::setprecision(17) << "{\n\"passed\":" << (passed ? "true" : "false")
      << ",\n\"acceptance_profile\":\"" << profile.name << "\""
      << ",\n\"matrix_comparison_required\":" << (profile.matrices_required?"true":"false")
      << ",\n\"strict_matrix_check_passed\":" << (profile.strict_matrices_passed(errors)?"true":"false")
      << ",\n\"mode\":\"" << argv[4] << "\",\n\"integrals\":\"" << argv[3] << "\""
      << ",\n\"tnvb_energy\":" << tensor.total_energy
      << ",\n\"determinant_energy\":" << determinant.total_energy
      << ",\n\"energy_error\":" << de << ",\n\"H_error\":" << dh
      << ",\n\"S_error\":" << ds << ",\n\"coefficient_error\":" << dc
      << ",\n\"all_weight_error\":" << dw << ",\n\"gradient_error\":";
    if (gradient) report << gradient_error; else report << "null";
    report << ",\n\"tnvb_coefficients\":"; write_vector(report,t.c);
    if (tensor.davidson_diagnostics) {
      const auto& diagnostics=*tensor.davidson_diagnostics;
      report << ",\n\"davidson_iterations\":" << diagnostics.iterations
             << ",\n\"davidson_actions\":" << diagnostics.block_actions
             << ",\n\"davidson_stationarity_residual\":" << diagnostics.maximum_stationarity_residual;
    }
    if (tensor.tnvb_diagnostics) {
      report << ",\n\"fit_energy_error\":" << tensor.tnvb_diagnostics->fit_energy_error
             << ",\n\"rdm_identity_error\":" << tensor.tnvb_diagnostics->rdm_identity_error;
    }
    report << ",\n\"determinant_coefficients\":"; write_vector(report,d.c);
    report << ",\n\"tnvb_determinants\":0,\n\"determinant_reference_count\":"
           << determinant_input.structure_data.pair_alpha.size() << "\n}\n";
    std::cout << std::setprecision(17) << "comparison: dE=" << de << " dH=" << dh
              << " dS=" << ds << " dc=" << dc << " dw=" << dw
              << " dg=" << gradient_error << " passed=" << passed << '\n';
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "comparison failed: " << error.what() << '\n';
    return 2;
  }
}
