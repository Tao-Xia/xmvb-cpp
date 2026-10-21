#include "cuda/vb/scf.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>

#include <nvtx3/nvtx3.hpp>

#include "cuda/linalg/blas.hpp"
#include "cuda/optimization/ciah.hpp"
#include "cuda/optimization/coarse.hpp"
#include "cuda/vb/chart.hpp"
#include "cuda/vb/diagonal.hpp"
#include "cuda/vb/ci.hpp"
#include "cuda/vb/point.hpp"
#include "cuda/vb/tangent.hpp"
#include "cuda/vb/transport.hpp"
#include "integrals/active/two_electron.hpp"
#include "mcscf/optimization/ciah/macro.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"

namespace xmvb::cuda {
namespace {
using View=linalg::ConstMatrixView;
using Out=linalg::MatrixView;
using Clock=std::chrono::steady_clock;

View segment(View x,int begin,int count) { return {x.data+begin,count,1,x.ld}; }
Out segment(Out x,int begin,int count) { return {x.data+begin,count,1,x.ld}; }
void copy(Context& c,Out y,View x) {
  if (x.rows!=y.rows || x.cols!=y.cols) throw std::invalid_argument("SCF copy shape");
  for (int j=0;j<x.cols;++j)
    check_cuda(cudaMemcpyAsync(y.column(j).data,x.column(j).data,x.rows*sizeof(double),
        cudaMemcpyDeviceToDevice,c.stream()),"SCF device copy");
}
Matrix<double> clone(Context& c,View x) {
  Matrix<double> result(c,x.rows,x.cols);copy(c,result.view(),x);return result;
}
void add(Context& c,Out y,View x,double factor=1.0) {
  for (int j=0;j<x.cols;++j)
    check_cublas(cublasDaxpy(c.blas(),blas_int(x.rows),&factor,x.column(j).data,1,
        y.column(j).data,1),"SCF vector sum");
}
void scale(Context& c,Out x,double factor) {
  for (int j=0;j<x.cols;++j)
    check_cublas(cublasDscal(c.blas(),blas_int(x.rows),&factor,x.column(j).data,1),
                 "SCF vector scaling");
}
double norm(Context& c,View x) {
  double square=0.0;
  for (int j=0;j<x.cols;++j) {
    double value=0.0;
    check_cublas(cublasDdot(c.blas(),blas_int(x.rows),x.column(j).data,1,
        x.column(j).data,1,&value),"SCF norm");square+=value;
  }
  return std::sqrt(std::max(0.0,square));
}
double dot(Context& c,View x,View y) {
  double result=0.0;
  check_cublas(cublasDdot(c.blas(),blas_int(x.rows),x.data,1,y.data,1,&result),"SCF dot");
  return result;
}

struct Trial {
  std::unique_ptr<VbPoint> point;
  std::shared_ptr<const StructureCiPoint> ci_point;
  Matrix<double> raw,coefficients;
  DeviceDavidsonResult ci;
  double energy=0.0;
};

/** W and spin links only. Zero integral tensors are topology sentinels, never
 * physical CPU evaluations; the first actual orbital/integral point is GPU.
 */
std::unique_ptr<vb::StructureAction> make_topology(const vb::VbScfInput& input) {
  const int na=input.orbital_preparation_input.n_active_orbitals;
  const auto topology=vb::build_orthogonal_spin_topology(input.structure_data.alpha_det,
      input.structure_data.beta_det,na);
  Eigen::MatrixXd identity=Eigen::MatrixXd::Identity(na,na);
  std::vector<double> overlap(identity.data(),identity.data()+identity.size());
  integrals::ActiveSpaceTwoElectronResult two;
  two.packed_active_two_electron_integrals.assign(
      integrals::packed_active_two_electron_integral_count(na),0.0);
  return std::make_unique<vb::StructureAction>(input.structure_data.determinant_to_structure_terms,
      input.structure_data.n_structures,topology,overlap,Eigen::MatrixXd::Zero(na,na),two,na,
      nullptr,vb::StructureActionMode::ActionOnly);
}
}  // namespace

ScfResult solve_scf(Context& context,const vb::VbScfInput& input,double nuclear,
    const vb::VbScfOptimizerOptions& options,const std::function<void(const ScfIteration&)>& observe) {
  if (input.standard_two_electron_mode==vb::StandardTwoElectronMode::ResolutionOfIdentity ||
      input.ao_integral_input.pair_graph.empty())
    throw std::invalid_argument("cuvbscf requires exact AO integrals");
  if (options.orbital_preconditioner!=vb::OrbitalPreconditioner::HessianDiagonal)
    throw std::invalid_argument("GPU CIAH requires analytic Hessian-diagonal preconditioning");
  const auto start=Clock::now();
  const auto& orbital=input.orbital_preparation_input;
  const int nb=orbital.n_basis_functions,norb=orbital.n_orbitals;
  const int na=orbital.n_active_orbitals,nstr=input.structure_data.n_structures;
  auto metadata=make_topology(input);
  StructureAction action(context,*metadata);
  const auto topology=vb::build_orthogonal_spin_topology(input.structure_data.alpha_det,
      input.structure_data.beta_det,na);
  SpinMetric spin(context,*metadata,topology.alpha_reuse_table.unique_determinants,
      topology.beta_reuse_table.unique_determinants,context.device_byte_budget());
  StructureCi ci_solver(context,action,spin,input.complete_active_space);
  ExactAoPairOperator ao(context,input.ao_integral_input.pair_graph,nb);
  Matrix<double> core(context,nb,nb),raw(context,nb,norb);
  copy_to_device(context,core.view(),input.ao_integral_input.ao_core_hamiltonian_matrix);
  context.copy_to_device(raw.data(),orbital.orbital_value_table.data(),nb*norb*sizeof(double));
  const auto new_point=[&] { return std::make_unique<VbPoint>(context,orbital,action,ao,core.view(),nuclear); };
  auto ci_options=linalg::make_davidson_options(nstr,1,1.0e-7,
      options.gradient_tolerance,options.energy_tolerance);
  ScfResult result;
  const auto finish=[&](Trial& trial,View guess) {
    trial.ci_point=ci_solver.point();
    trial.coefficients=clone(context,trial.ci.eigenpairs.eigenvector_matrix.view());
    if (guess.data && dot(context,guess,trial.ci.overlap_view())<0.0) {
      scale(context,trial.coefficients.view(),-1.0);
      scale(context,trial.ci.eigenpairs.eigenvector_matrix.view(),-1.0);
      scale(context,trial.ci.hamiltonian_eigenvectors.view(),-1.0);
      if (!trial.ci.identity_overlap) scale(context,trial.ci.overlap_eigenvectors.view(),-1.0);
    }
    trial.energy=trial.point->evaluate_coefficients(trial.coefficients.view());
  };
  const auto relax=[&](Matrix<double> trial_raw,View guess)->Trial {
    const nvtx3::scoped_range range{"scf.relax_ci"};
    Trial trial;trial.raw=std::move(trial_raw);trial.point=new_point();
    trial.point->prepare(trial.raw.view());
    trial.ci=guess.data ? ci_solver.solve_device(guess,ci_options) : ci_solver.solve_device(ci_options);
    result.ci_actions+=trial.ci.block_actions;
    finish(trial,guess);
    return trial;
  };
  Trial accepted=relax(std::move(raw),{});
  DeviceCiahVectorOps ops{&context};
  Matrix<double> last_raw,last_ci;
  std::unique_ptr<AcceptedOrbitalChart> next_chart;
  static const nvtx3::registered_string macro_name{"gpu.scf.macro"};
  for (int iteration=0;iteration<=options.max_iterations;++iteration) {
    const nvtx3::scoped_range macro_range{macro_name};
    auto& origin=*accepted.point;
    auto chart_owner=std::move(next_chart);
    if (!chart_owner) {
      chart_owner=std::make_unique<AcceptedOrbitalChart>(context,orbital,input.complete_active_space);
      chart_owner->rebuild(origin.geometry());
    }
    auto& chart=*chart_owner;
    const int no=chart.reduced_size();
    Matrix<double> raw_gradient(context,nb,norb),coefficient_gradient(context,nstr,1);
    origin.gradient(raw_gradient.view(),coefficient_gradient.view());
    Matrix<double> reduced(context,no,1);
    chart.project(raw_gradient.view(),reduced.view());
    result.energy=accepted.energy;
    result.gradient_inf=chart.reference_gradient_inf_norm(reduced.view());
    if (iteration==0 && observe)
      observe({0,0,result.ci_actions,0,result.energy,0.0,result.gradient_inf,
          std::chrono::duration<double>(Clock::now()-start).count()});
    if ((iteration==0 || std::abs(result.trace.back().energy_change)<options.energy_tolerance) &&
        result.gradient_inf<options.gradient_tolerance) {
      result.converged=true;result.reason="ciah_projected_gradient_tolerance";break;
    }
    if (iteration==options.max_iterations) { result.reason="ciah_iteration_limit";break; }
    if (no==0) { result.reason="ciah_nonredundant_space_empty";break; }
    context.set_allocation_phase("analytic Hessian diagonal");
    auto diagonal=origin.hessian_diagonal(chart.accepted_mo_frame());
    chart.set_hessian_diagonal(origin.geometry(),diagonal.view());
    context.set_allocation_phase("CI tangent and transport");
    std::unique_ptr<SingleStateCiTangent> tangent;
    if (nstr>1) tangent=std::make_unique<SingleStateCiTangent>(context,nstr,
        accepted.coefficients.view(),accepted.ci.eigenpairs.eigenvalues.front(),1.0,
        SingleStateCiActions{
          [&](View x,Out h,Out m) { origin.activate();action.apply(x,h,m); },
          [&](View x,Out m) { origin.activate();action.apply_overlap(x,m); },
          [&](View x,double eigenvalue,Out y) {
            ci_solver.inverse_augmented_hessian_model(x,
                accepted.ci.eigenpairs.eigenvalues.front(),eigenvalue,y,
                accepted.ci_point.get());
          },
          [&](View x) { origin.activate();return action.maximum_orthogonal_component(x); },
          input.complete_active_space ? CiInverseConstraint::Carrier:
              (ci_solver.uses_selected_metric() ? CiInverseConstraint::Diagonal:
                  CiInverseConstraint::Spin)});
    const int ns=tangent ? tangent->tangent_size():0,dimension=no+ns;
    SelectedCrossOverlap overlap(context,action,na,
        ci_solver.has_exact_metric_inverse() ? SelectedCrossOverlap::ExactMetricInverse(
          [&](View x,Out y) { ci_solver.inverse_metric(x,y); }):SelectedCrossOverlap::ExactMetricInverse{});
    if (!ci_solver.has_exact_metric_inverse())
      overlap.set_preconditioner([&](View x,Out y) {
        spin.solve_diagonal_model(x,overlap.current_metric_diagonal(),y);
      });
    std::unique_ptr<SingleStateCiTransport> transport;
    if (ns) transport=std::make_unique<SingleStateCiTransport>(context,overlap,
        origin.geometry().frame_transform(),accepted.coefficients.view(),
        accepted.ci.hamiltonian_eigenvectors.view(),accepted.ci.overlap_view(),1.0,
        coefficient_gradient.view());
    Matrix<double> gradient(context,dimension,1),seed(context,dimension,1);
    copy(context,segment(gradient.view(),0,no),reduced.view());
    if (ns) tangent->origin_gradient(segment(gradient.view(),no,ns));
    chart.inverse_metric(reduced.view(),segment(seed.view(),0,no));
    if (ns) copy(context,segment(seed.view(),no,ns),segment(gradient.view(),no,ns));
    if (last_raw.data()) chart.project_vector(last_raw.view(),segment(seed.view(),0,no));
    if (ns && last_ci.data()) {
      auto guess=clone(context,last_ci.view());scale(context,guess.view(),std::sqrt(2.0));
      tangent->project_coordinates(guess.view(),segment(seed.view(),no,ns));
    }
    const auto coordinate_norm=[&](View g) {
      return std::hypot(chart.coordinate_dual_norm(segment(g,0,no)),
          ns ? norm(context,segment(g,no,ns)):0.0);
    };
    const auto metric=[&](View x,Out mx) {
      chart.physical_metric(segment(x,0,no),segment(mx,0,no));
      if (ns) tangent->apply_metric_coordinates(segment(x,no,ns),segment(mx,no,ns));
    };
    Matrix<double> last_image(context,dimension,1);
    const auto base_inverse=[&](View r,double eigenvalue,Out p) {
      chart.apply_inverse_hessian_diagonal(segment(r,0,no),eigenvalue-1.0e-8,segment(p,0,no));
      if (ns) {
        origin.activate();
        tangent->apply_inverse_augmented_hessian_diagonal(segment(r,no,ns),
            eigenvalue-1.0e-8,segment(p,no,ns));
      }
    };
    const int hvp_before=result.hvp,ci_before=result.ci_actions;
    DeviceCiahOptions ciah;
    const vb::StructureSolveAccuracy accuracy{options.energy_tolerance,options.gradient_tolerance};
    ciah.operator_relative_accuracy=accuracy.response_backward_error_tolerance(
        std::abs(accepted.ci.eigenpairs.eigenvalues.front()));
    context.set_allocation_phase("CIAH coarse workspace");
    // One full coupled sample is observed per retained AH direction. Unlike
    // the CPU split-block collector, this path never emits two samples.
    DeviceCoarsePreconditioner coarse(context,dimension,ns?ciah.max_cycle:0,
        ciah.lindep,ciah.operator_relative_accuracy);
    ciah.apply_images=[&](View x,Out hx,Out mx) {
      const nvtx3::scoped_range range{"scf.coupled_action"};
      AllocationPhase allocation(context,"coupled orbital-CI HVP");
      Matrix<double> dr(context,nb,norb),dc(context,nstr,1),hr(context,nb,norb),hc(context,nstr,1);
      chart.expand(segment(x,0,no),dr.view());
      if (ns) tangent->coefficient_response_coordinates(segment(x,no,ns),dc.view());
      else check_cuda(cudaMemsetAsync(dc.data(),0,nstr*sizeof(double),context.stream()),"zero CI direction");
      origin.hessian(dr.view(),dc.view(),hr.view(),hc.view());
      chart.project(hr.view(),segment(hx,0,no));
      if (ns) {
        tangent->project_coordinates(hc.view(),segment(hx,no,ns));
        scale(context,segment(hx,no,ns),1.0/std::sqrt(2.0));
      }
      metric(x,mx);copy(context,last_image.view(),hx);++result.hvp;
    };
    ciah.apply_metric=metric;ciah.coordinate_norm=coordinate_norm;
    ciah.residual_preconditioner=[&](View r,double value,Out p) {
      coarse.apply(r,value,[&](View x,Out y) {base_inverse(x,value,y);},p);
    };
    ciah.residual_direction=[&](View r,Out p) {
      chart.inverse_metric(segment(r,0,no),segment(p,0,no));
      if (ns) copy(context,segment(p,no,ns),segment(r,no,ns));
    };
    ciah.observe_images=[&](View x,View mx) { coarse.observe(x,last_image.view(),mx); };
    context.set_allocation_phase("CIAH subspace");
    DeviceCiahWorkspace workspace(context,dimension,ciah,seed.view());
    const auto maximum_component=[&](View x) {
      return std::max(chart.max_component(segment(x,0,no)),ns ?
          tangent->maximum_coefficient_component_coordinates(segment(x,no,ns)):0.0);
    };
    struct Keyframe { Matrix<double> displacement;std::unique_ptr<VbPoint> point;
      std::shared_ptr<const StructureCiPoint> ci_point;
      Matrix<double> raw;std::optional<SingleStateCiTransportPoint> transport; };
    std::optional<Keyframe> cache;
    const auto keyframe=[&](View x)->Keyframe& {
      AllocationPhase allocation(context,"CI transported keyframe");
      if (cache) {
        auto difference=clone(context,x);add(context,difference.view(),cache->displacement.view(),-1.0);
        if (norm(context,difference.view())==0.0) return *cache;
      }
      Keyframe frame;frame.displacement=clone(context,x);frame.raw=Matrix<double>(context,nb,norb);
      chart.retract(accepted.raw.view(),segment(x,0,no),1.0,frame.raw.view());
      frame.point=new_point();frame.point->prepare(frame.raw.view());
      if (ci_solver.uses_selected_metric()) frame.ci_point=ci_solver.point();
      if (ns) {
        Matrix<double> dr(context,nb,norb),dc(context,nstr,1);
        chart.expand(segment(x,0,no),dr.view());
        auto direction=origin.geometry().direction(dr.view());
        tangent->coefficient_response_coordinates(segment(x,no,ns),dc.view());
        add(context,dc.view(),accepted.coefficients.view());
        frame.transport.emplace(transport->retract(frame.point->geometry().frame_transform(),
            direction.frame_transform.view(),dc.view(),
            frame.ci_point ? frame.ci_point->metric:nullptr));
      }
      cache.emplace(std::move(frame));return *cache;
    };
    DeviceCiahMicroCallbacks callbacks;
    callbacks.coordinate_norm=coordinate_norm;callbacks.metric_action=metric;
    callbacks.maximum_component=maximum_component;callbacks.orbital_size=no;
    callbacks.orbital_coordinate_norm=[&](View g){return chart.coordinate_dual_norm(g);};
    callbacks.evaluate_keyframe=[&](View x) {
      auto& frame=keyframe(x);
      const auto coefficients=ns ? frame.transport->coefficients.view():accepted.coefficients.view();
      const double physical=frame.point->evaluate_coefficients(coefficients);
      Matrix<double> gr(context,nb,norb),gc(context,nstr,1),g(context,dimension,1);
      frame.point->gradient(gr.view(),gc.view());
      double energy=physical;
      if (ns) {
        auto model=transport->evaluate_model(*frame.transport,physical,gc.view());
        Matrix<double> extra(context,nb,norb);
        frame.point->geometry().pullback({},{},{},model.current_upper.view(),{},extra.view());
        add(context,gr.view(),extra.view());
        origin.geometry().pullback({},{},{},model.origin_upper.view(),{},extra.view());
        add(context,gr.view(),extra.view());
        tangent->project_coordinates(model.structure_covector.view(),segment(g.view(),no,ns));
        energy=model.energy;
      }
      chart.project(gr.view(),segment(g.view(),0,no));
      return DeviceCiahKeyframe{std::move(g),energy,physical};
    };
    auto micro=run_device_ciah_microsteps(workspace,gradient.view(),accepted.energy,
        options.gradient_tolerance,callbacks);
    mcscf::CiahMacroCallbacks<DeviceCiahVectorOps,Trial> macro;
    macro.maximum_component=[&](const Matrix<double>& x){return maximum_component(x.view());};
    macro.step_norm=[&](const Matrix<double>& x) {
      Matrix<double> mx(context,no,1);
      chart.physical_metric(segment(x.view(),0,no),mx.view());
      return std::sqrt(std::max(0.0,dot(context,segment(x.view(),0,no),mx.view())));
    };
    macro.changes_parameters=[&](const Matrix<double>& x) {
      Matrix<double> dr(context,nb,norb);chart.expand(segment(x.view(),0,no),dr.view());
      return norm(context,dr.view())>std::numeric_limits<double>::epsilon()*norm(context,accepted.raw.view());
    };
    macro.evaluate=[&](const Matrix<double>& displacement,double step_scale) {
      AllocationPhase allocation(context,"macro CI relaxation");
      auto scaled=clone(context,displacement.view());scale(context,scaled.view(),step_scale);
      auto& frame=keyframe(scaled.view());
      const auto guess=ns ? frame.transport->coefficients.view():accepted.coefficients.view();
      // Reuse the already prepared point instead of regenerating AO response.
      Trial trial;trial.raw=clone(context,frame.raw.view());trial.point=std::move(frame.point);
      trial.point->activate();
      if (frame.ci_point) ci_solver.bind_point(frame.ci_point);
      trial.ci=ci_solver.solve_device(guess,ci_options);
      result.ci_actions+=trial.ci.block_actions;
      finish(trial,guess);
      cache.reset();return trial;
    };
    macro.energy=[](const Trial& trial){return trial.energy;};
    macro.terminal=[&](Trial& trial) {
      AcceptedOrbitalChart trial_chart(context,orbital,input.complete_active_space);
      trial_chart.rebuild(trial.point->geometry());
      Matrix<double> gr(context,nb,norb),gc(context,nstr,1),g(context,trial_chart.reduced_size(),1);
      trial.point->gradient(gr.view(),gc.view());trial_chart.project(gr.view(),g.view());
      return trial_chart.reference_gradient_inf_norm(g.view())<options.gradient_tolerance;
    };
    auto chosen=mcscf::run_ciah_macro(workspace,micro,gradient,accepted.energy,
        options.energy_tolerance,options.minimum_step_size,ops,macro);
    if (!chosen.trial) {result.reason=chosen.reason;break;}
    last_raw=Matrix<double>(context,nb,norb);
    chart.expand(segment(micro.last_step.view(),0,no),last_raw.view());
    if (ns) {
      last_ci=Matrix<double>(context,nstr,1);
      tangent->coefficient_response_coordinates(segment(micro.last_step.view(),no,ns),last_ci.view());
    }
    const double change=chosen.trial->energy-accepted.energy;
    accepted=std::move(*chosen.trial);++result.iterations;
    ScfIteration record{result.iterations,result.hvp-hvp_before,result.ci_actions-ci_before,
        micro.keyframe_evaluations+1,accepted.energy,change,0.0,
        std::chrono::duration<double>(Clock::now()-start).count()};
    // The accepted-point gradient is already prepared; the next loop reports
    // its stationarity in its own newly built quotient, not the old model.
    next_chart=std::make_unique<AcceptedOrbitalChart>(context,orbital,input.complete_active_space);
    auto& next=*next_chart;next.rebuild(accepted.point->geometry());
    Matrix<double> gr(context,nb,norb),gc(context,nstr,1),g(context,next.reduced_size(),1);
    accepted.point->gradient(gr.view(),gc.view());next.project(gr.view(),g.view());
    record.gradient_inf=next.reference_gradient_inf_norm(g.view());
    result.trace.push_back(record);if (observe) observe(record);
  }
  result.energy=accepted.energy;result.orbitals=std::move(accepted.raw);
  result.coefficients=std::move(accepted.coefficients);
  result.ci_prepared_points=ci_solver.prepared_points();
  result.selected_metric_builds=ci_solver.metric_builds();
  result.seconds=std::chrono::duration<double>(Clock::now()-start).count();
  return result;
}

}  // namespace xmvb::cuda
