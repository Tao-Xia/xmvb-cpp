/** Same-orbital direct FULL versus ENERGY handoff; no determinant reference. */
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <Eigen/Eigenvalues>
#include "input/loading/loader.hpp"
#include "tnvb/electronic/evaluator.hpp"
#include "vbscf/derivatives/gradient/orbital/evaluator.hpp"

namespace {
double difference(const std::vector<double>& a,const std::vector<double>& b) {
  if (a.size()!=b.size()) throw std::runtime_error("handoff dimension mismatch");
  double result=0;
  for (std::size_t i=0;i<a.size();++i) result=std::max(result,std::abs(a[i]-b[i]));
  return result;
}
Eigen::MatrixXd weights(const xmvb::vb::VbScfResult& value) {
  const int n=value.n_structures;
  const Eigen::Map<const Eigen::MatrixXd> s(value.structure_matrices.overlap_matrix.data(),n,n);
  const Eigen::Map<const Eigen::VectorXd> c(value.eigenvector_matrix.data(),n);
  const Eigen::VectorXd norms=s.diagonal().cwiseSqrt();
  const Eigen::MatrixXd metric=norms.cwiseInverse().asDiagonal()*s*norms.cwiseInverse().asDiagonal();
  const Eigen::VectorXd v=norms.asDiagonal()*c;
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(metric);
  Eigen::MatrixXd w(n,4);
  w.col(0)=v.array()*(metric*v).array()/(v.dot(metric*v));
  const Eigen::MatrixXd root=eig.eigenvectors()*eig.eigenvalues().cwiseSqrt().asDiagonal()*eig.eigenvectors().transpose();
  const Eigen::MatrixXd inv=eig.eigenvectors()*eig.eigenvalues().cwiseInverse().asDiagonal()*eig.eigenvectors().transpose();
  w.col(1)=(root*v).array().square();w.col(1)/=w.col(1).sum();
  w.col(2)=v.array().square()/inv.diagonal().array();w.col(2)/=w.col(2).sum();
  w.col(3)=v.array().square();w.col(3)/=w.col(3).sum();
  return w;
}
}

int main(int argc,char** argv) {
  try {
    if (argc!=5 && argc!=6) throw std::invalid_argument("usage: handoff_tnvb INPUT REPORT exact|ri dense|davidson [control]");
    xmvb::vb::VbScfInputLoadOptions options;
    const std::string model=argv[3],solver=argv[4];
    if (model=="exact") options.standard_two_electron_mode=xmvb::vb::StandardTwoElectronMode::Exact;
    else if (model=="ri") options.standard_two_electron_mode=xmvb::vb::StandardTwoElectronMode::ResolutionOfIdentity;
    else throw std::invalid_argument("invalid handoff integral model");
    if (solver!="dense" && solver!="davidson") throw std::invalid_argument("invalid handoff solver");
    auto loaded=xmvb::vb::load_vbscf_input_with_timings(argv[1],options);
    xmvb::vb::ElectronicStateRequest request;
    request.nuclear_repulsion_energy=loaded.nuclear_repulsion_energy;
    request.eigensolver=solver=="dense" ? xmvb::vb::StructureEigensolver::Dense : xmvb::vb::StructureEigensolver::Davidson;
    if (argc==6) {
      if (std::string(argv[5])!="control") throw std::invalid_argument("invalid handoff control argument");
      const auto prepared=xmvb::vb::prepare_timed_active_space_context(loaded.input);
      std::shared_ptr<const xmvb::tnvb::StateDirectory> surviving;
      std::filesystem::path surviving_path,root;
      {
        xmvb::tnvb::WorkerProcess owner,foreign;
        auto retained=owner.evaluate(loaded.input,prepared.prepared_active_space,request,false,true);
        const auto path=retained.retained_state->path();
        if (!std::filesystem::is_regular_file(path/"forward/ready.json")) throw std::runtime_error("missing retained bank");
        bool rejected=false;
        try { (void)foreign.evaluate(loaded.input,prepared.prepared_active_space,request,true,false,retained.retained_state); }
        catch (const std::invalid_argument&) { rejected=true; }
        if (!rejected) throw std::runtime_error("foreign owner accepted retained bank");
        retained.retained_state.reset();
        if (std::filesystem::exists(path)) throw std::runtime_error("discarded trial leaked its bank");
        retained=owner.evaluate(loaded.input,prepared.prepared_active_space,request,false,true);
        surviving=retained.retained_state;
        surviving_path=surviving->path();root=surviving_path.parent_path();
      }
      if (!std::filesystem::exists(surviving_path)) throw std::runtime_error("lease lost its owner directory");
      surviving.reset();
      if (std::filesystem::exists(root)) throw std::runtime_error("last lease leaked calculation scratch");
      std::cout << "[TNVB-HANDOFF-CONTROL] foreign=reject discarded=clean owner_lifetime=pass\n";
    }
    xmvb::tnvb::ElectronicEvaluator backend;
    xmvb::vb::OrbitalGradientEvaluator pullback;
    auto direct=backend.evaluate_energy_gradient(loaded.input,request);
    const auto direct_gradient=pullback.evaluate_without_reference_energy_gradient(loaded.input,std::move(direct));
    auto forward=backend.evaluate_energy(loaded.input,request);
    const double energy=forward->total_energy();
    auto resumed=backend.complete_gradient(loaded.input,std::move(forward));
    const auto resumed_gradient=pullback.evaluate_without_reference_energy_gradient(loaded.input,std::move(resumed));
    const auto& a=direct_gradient.scf_result;const auto& b=resumed_gradient.scf_result;
    const int n=a.n_structures;
    Eigen::VectorXd ac=Eigen::Map<const Eigen::VectorXd>(a.eigenvector_matrix.data(),n);
    Eigen::VectorXd bc=Eigen::Map<const Eigen::VectorXd>(b.eigenvector_matrix.data(),n);
    if (ac.dot(bc)<0)bc=-bc;
    const double de=std::abs(a.total_energy-b.total_energy);
    const double dc=(ac-bc).cwiseAbs().maxCoeff();
    const double dh=difference(a.structure_matrices.hamiltonian_matrix,b.structure_matrices.hamiltonian_matrix);
    const double ds=difference(a.structure_matrices.overlap_matrix,b.structure_matrices.overlap_matrix);
    const double dw=(weights(a)-weights(b)).cwiseAbs().maxCoeff();
    const double dg=difference(direct_gradient.sparse_orbital_energy_gradient,resumed_gradient.sparse_orbital_energy_gradient);
    const double drift=std::abs(energy-b.total_energy);
    const bool passed=std::isfinite(de+dc+dh+ds+dw+dg+drift) &&
        de<=1e-9 && dc<=1e-7 && dh<=1e-8 && ds<=1e-9 && dw<=1e-7 && dg<=1e-7 && drift<=1e-12;
    std::ofstream out(argv[2]);
    out << std::setprecision(17) << "{\n\"passed\":" << (passed?"true":"false")
        << ",\n\"integrals\":\"" << model << "\",\n\"solver\":\"" << solver
        << "\",\n\"energy_error\":" << de << ",\n\"coefficient_error\":" << dc
        << ",\n\"H_error\":" << dh << ",\n\"S_error\":" << ds
        << ",\n\"all_weight_error\":" << dw << ",\n\"gradient_error\":" << dg
        << ",\n\"energy_handoff_drift\":" << drift << "\n}\n";
    std::cout << "handoff dE=" << de << " dc=" << dc << " dw=" << dw << " dg=" << dg << " passed=" << passed << '\n';
    return passed?0:1;
  } catch (const std::exception& error) {
    std::cerr << "handoff comparison failed: " << error.what() << '\n';return 1;
  }
}
