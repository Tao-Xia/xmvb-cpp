#include "tnvb/electronic/evaluator.hpp"
#include "tnvb/gradient/adapter.hpp"
#include "tnvb/electronic/input_signature.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <Eigen/Cholesky>
#include "input/deck/keywords.hpp"

namespace xmvb::tnvb {
namespace {

class Forward final : public vb::ElectronicStateForwardEvaluation {
 public:
  vb::ElectronicStateRequest request;
  vb::ActiveSpaceGradientResult result;
  std::shared_ptr<const StateDirectory> state;
  std::vector<double> orbitals;
  std::array<std::uint64_t,2> input_key;
  double total_energy() const override { return result.scf_result.total_energy; }
};

void validate_response(const WorkerResponse& response) {
  if (!response.hamiltonian.allFinite() || !response.overlap.allFinite() ||
      !response.coefficients.allFinite() || !response.energies.allFinite() ||
      !std::isfinite(response.fit_energy_error) || response.fit_energy_error < 0 ||
      !std::isfinite(response.rdm_identity_error) || response.rdm_identity_error > 1e-7 ||
      response.rdm_identity_error < 0) {
    throw std::runtime_error("TNVB worker returned invalid numerical data");
  }
  Eigen::LLT<Eigen::MatrixXd> metric(response.overlap);
  if (metric.info() != Eigen::Success) throw std::runtime_error("TNVB structure metric is not positive definite");
  const double hscale=std::max(1.0,response.hamiltonian.cwiseAbs().maxCoeff());
  if ((response.hamiltonian-response.hamiltonian.transpose()).cwiseAbs().maxCoeff() > 1e-10*hscale ||
      (response.overlap-response.overlap.transpose()).cwiseAbs().maxCoeff() > 1e-10) {
    throw std::runtime_error("TNVB H/S matrices are not symmetric");
  }
  for (int root=0; root<response.energies.size(); ++root) {
    const auto c=response.coefficients.col(root);
    const Eigen::VectorXd sc=response.overlap*c;
    const Eigen::VectorXd hc=response.hamiltonian*c;
    const double scale=std::max(1.0,hc.norm()+std::abs(response.energies[root])*sc.norm());
    if (std::abs(c.dot(sc)-1.0)>1e-9 ||
        (hc-response.energies[root]*sc).norm()>1e-9*scale) {
      throw std::runtime_error("TNVB eigenpair failed the C++ norm/residual check");
    }
  }
}

}  // namespace

vb::ActiveSpaceGradientResult ElectronicEvaluator::evaluate(
    const vb::VbScfInput& input, const vb::ElectronicStateRequest& request,
    bool gradient, std::shared_ptr<const StateDirectory>* retain) const {
  if (!input.tnvb.enabled || request.selected_states != std::vector<int>{0} ||
      request.state_weights.size()!=1 || !std::isfinite(request.state_weights[0]) ||
      request.state_weights[0]<=0 ||
      input.orbital_preparation_input.spin_multiplicity!=1 ||
      (input.orbital_preparation_input.orbital_type!=vb::kOrbitalTypeHao &&
       input.orbital_preparation_input.orbital_type!=vb::kOrbitalTypeOeo)) {
    throw std::invalid_argument("TNVB supports HAO/OEO and one singlet root");
  }
  if (gradient && input.tnvb.total_bond < input.tnvb.structure_bond) {
    throw std::invalid_argument("TNVB orbital gradients require TNSUMBOND >= TNBOND");
  }
  // Reject invalid trial coordinates before entering parallel preparation,
  // where a thrown exception cannot cross an OpenMP region safely. This guard
  // is confined to the TNVB backend and leaves ordinary numerical kernels intact.
  const auto& orbitals=input.orbital_preparation_input.orbital_value_table;
  if (!std::isfinite(request.nuclear_repulsion_energy) ||
      !std::all_of(orbitals.begin(),orbitals.end(),[](double value) { return std::isfinite(value); })) {
    throw std::invalid_argument("TNVB received nonfinite orbitals or nuclear energy");
  }
  const auto started=std::chrono::steady_clock::now();
  // Shared preparation supplies h/G in C R^-1 and retains R for the
  // nonorthogonal VB pair creators.  The worker consumes these integrals
  // directly; the gradient adapter follows the shared framed pullback.
  auto timed=vb::prepare_timed_active_space_context(input);
  auto& prepared=timed.prepared_active_space;
  WorkerResponse response=process_.evaluate(input,prepared,request,gradient,
                                            retain!=nullptr);
  validate_response(response);
  if (retain) *retain=std::move(response.retained_state);
  vb::ActiveSpaceGradientResult result;
  auto& scf=result.scf_result;
  scf.davidson_diagnostics=response.davidson_diagnostics;
  scf.tnvb_diagnostics=vb::TnVbDiagnostics{
      input.tnvb.structure_bond, input.tnvb.total_bond, gradient,
      response.fit_energy_error, response.rdm_identity_error,
      response.hs_wall_seconds, response.rdm_wall_seconds};
  scf.n_structures=input.raw_structure_data.n_structures;
  scf.nuclear_repulsion_energy=request.nuclear_repulsion_energy;
  scf.one_electron_reference_energy=prepared.one_electron_reference_energy;
  // The existing C++ result convention stores the active eigenvalue here;
  // E11 is separate and enters the total exactly once.
  scf.electronic_energy=response.energies[0];
  scf.total_energy=prepared.one_electron_reference_energy+scf.electronic_energy+
      request.nuclear_repulsion_energy;
  scf.selected_state_indices={0}; scf.state_average_weights={1.0};
  scf.selected_state_total_energies={scf.total_energy};
  scf.electronic_state_energies.assign(response.energies.data(),response.energies.data()+response.energies.size());
  scf.eigenvector_matrix.assign(response.coefficients.data(),response.coefficients.data()+response.coefficients.size());
  const Eigen::VectorXd diagonal=response.overlap.diagonal();
  scf.structure_overlap_diagonal.assign(diagonal.data(),diagonal.data()+diagonal.size());
  scf.structure_overlap_diagonal_exact=true;
  scf.average_structure_overlap=diagonal.mean();
  scf.structure_matrices.n_structures=scf.n_structures;
  scf.structure_matrices.hamiltonian_matrix.assign(response.hamiltonian.data(),response.hamiltonian.data()+response.hamiltonian.size());
  scf.structure_matrices.overlap_matrix.assign(response.overlap.data(),response.overlap.data()+response.overlap.size());
  result.orbital_preparation_wall_time_seconds=timed.timings.orbital_preparation_wall_time_seconds;
  result.ao_effective_one_electron_wall_time_seconds=timed.timings.ao_effective_one_electron_wall_time_seconds;
  result.active_one_electron_wall_time_seconds=timed.timings.active_one_electron_wall_time_seconds;
  result.active_two_electron_wall_time_seconds=timed.timings.active_two_electron_wall_time_seconds;
  result.structure_matrix_wall_time_seconds=response.hs_wall_seconds;
  result.adjoint_wall_time_seconds=response.rdm_wall_seconds;
  result.active_orbital_overlap_matrix=prepared.orbital_result.active_orbital_overlap_matrix;
  result.orbital_preparation_result=std::move(prepared.orbital_result);
  result.ao_effective_one_electron_result=std::move(prepared.ao_effective_one_electron_result);
  result.active_space_one_electron_result=std::move(prepared.active_space_one_electron_result);
  result.active_space_two_electron_result=std::move(prepared.active_space_two_electron_result);
  if (gradient) populate_active_space_adjoints(response,&result);
  result.total_wall_time_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  return result;
}

std::unique_ptr<vb::ElectronicStateForwardEvaluation> ElectronicEvaluator::evaluate_energy(
    const vb::VbScfInput& input, const vb::ElectronicStateRequest& request) const {
  auto result=std::make_unique<Forward>();
  result->request=request;
  result->orbitals=input.orbital_preparation_input.orbital_value_table;
  result->result=evaluate(input,request,false,&result->state);
  result->input_key=input_signature(input);
  return result;
}

vb::ActiveSpaceGradientResult ElectronicEvaluator::complete_gradient(
    const vb::VbScfInput& input,
    std::unique_ptr<vb::ElectronicStateForwardEvaluation> forward) const {
  auto* state=dynamic_cast<Forward*>(forward.get());
  if (state==nullptr) throw std::invalid_argument("TNVB backend received foreign forward state");
  const auto started=std::chrono::steady_clock::now();
  if (!state->state || state->orbitals!=input.orbital_preparation_input.orbital_value_table ||
      state->input_key!=input_signature(input)) {
    throw std::invalid_argument("TNVB handoff requires the same evaluated scientific input");
  }
  // Worker-side request/integral/settings validation precedes native RDM work.
  // Consuming this Forward releases its private bank on both success/failure.
  if (input.tnvb.total_bond<input.tnvb.structure_bond) {
    throw std::invalid_argument("TNVB orbital gradients require TNSUMBOND >= TNBOND");
  }
  const auto response=process_.complete_gradient(input,state->request,state->state);
  validate_response(response);
  auto result=std::move(state->result);
  if (std::abs(response.energies[0]-result.scf_result.electronic_energy)>1e-10) {
    throw std::runtime_error("TNVB handoff changed its selected energy");
  }
  std::copy_n(response.coefficients.data(),response.coefficients.rows(),
              result.scf_result.eigenvector_matrix.begin());
  auto& diagnostics=*result.scf_result.tnvb_diagnostics;
  diagnostics.gradient_evaluated=true;
  diagnostics.fit_energy_error=response.fit_energy_error;
  diagnostics.rdm_identity_error=response.rdm_identity_error;
  diagnostics.rdm_wall_seconds=response.rdm_wall_seconds;
  result.adjoint_wall_time_seconds=response.rdm_wall_seconds;
  populate_active_space_adjoints(response,&result);
  result.total_wall_time_seconds+=std::chrono::duration<double>(
      std::chrono::steady_clock::now()-started).count();
  return result;
}

vb::ActiveSpaceGradientResult ElectronicEvaluator::evaluate_energy_gradient(
    const vb::VbScfInput& input, const vb::ElectronicStateRequest& request) const {
  return evaluate(input,request,true);
}

vb::ActiveSpaceGradientResult ElectronicEvaluator::evaluate_energy_result(
    const vb::VbScfInput& input, const vb::ElectronicStateRequest& request) const {
  return evaluate(input,request,false);
}

}  // namespace xmvb::tnvb
