#include "cuda/vb/transport.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

#include "cuda/linalg/blas.hpp"
#include "cuda/vb/action.hpp"
#include "cuda/ci/sigma.hpp"

namespace xmvb::cuda {
namespace {

void copy(Context& context, linalg::MatrixView destination,
          linalg::ConstMatrixView source) {
  if (destination.rows != source.rows || destination.cols != source.cols ||
      destination.ld < destination.rows || source.ld < source.rows)
    throw std::invalid_argument("transport copy dimensions");
  for (std::int64_t col = 0; col < source.cols; ++col)
    check_cuda(cudaMemcpyAsync(destination.column(col).data,
        source.column(col).data, source.rows*sizeof(double),
        cudaMemcpyDeviceToDevice, context.stream()), "transport device copy");
}

void zero(Context& context, linalg::MatrixView destination) {
  for (std::int64_t col = 0; col < destination.cols; ++col)
    check_cuda(cudaMemsetAsync(destination.column(col).data, 0,
        destination.rows*sizeof(double), context.stream()), "transport zero");
}

void axpy(Context& context, linalg::MatrixView destination, double factor,
          linalg::ConstMatrixView source) {
  if (destination.rows != source.rows || destination.cols != source.cols)
    throw std::invalid_argument("transport AXPY dimensions");
  for (std::int64_t col = 0; col < source.cols; ++col)
    check_cublas(cublasDaxpy(context.blas(), blas_int(source.rows), &factor,
        source.column(col).data, 1, destination.column(col).data, 1),
        "transport AXPY");
}

double dot(Context& context, linalg::ConstMatrixView lhs,
           linalg::ConstMatrixView rhs) {
  double value = 0.0;
  check_cublas(cublasDdot(context.blas(), blas_int(lhs.rows), lhs.data, 1,
                         rhs.data, 1, &value), "transport dot");
  return value;
}

double norm(Context& context, linalg::ConstMatrixView vector) {
  double value = 0.0;
  check_cublas(cublasDnrm2(context.blas(), blas_int(vector.rows), vector.data,
                          1, &value), "transport norm");
  return value;
}

void right_upper_solve(Context& context, linalg::ConstMatrixView upper,
                       linalg::MatrixView matrix, bool transpose) {
  constexpr double one = 1.0;
  check_cublas(cublasDtrsm(context.blas(), CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
      transpose ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
      blas_int(matrix.rows), blas_int(matrix.cols), &one, upper.data,
      blas_int(upper.ld), matrix.data, blas_int(matrix.ld)),
      "transport frame triangular solve");
}

__global__ void upper_only_kernel(double* values, int size, int ld) {
  const int index = blockIdx.x*blockDim.x+threadIdx.x;
  if (index >= size*size) return;
  const int row = index%size, col = index/size;
  if (row > col) values[row+col*ld] = 0.0;
}

__global__ void jacobi_kernel(double* image, const double* residual,
                              const double* diagonal, int size) {
  const int index = blockIdx.x*blockDim.x+threadIdx.x;
  if (index < size) image[index] = residual[index]/diagonal[index];
}

void upper_only(Context& context, linalg::MatrixView values) {
  const int size = blas_int(values.rows);
  upper_only_kernel<<<(size*size+255)/256,256,0,context.stream()>>>(
      values.data,size,blas_int(values.ld));
  check_cuda(cudaGetLastError(), "transport upper projection kernel");
}

}  // namespace

SelectedCrossOverlap::SelectedCrossOverlap(Context& context,
                                           StructureAction& action,
                                           int active_orbitals,
                                           ExactMetricInverse exact_inverse,
                                           MetricPreconditioner preconditioner)
    : context_(&context), action_(&action), structures_(action.size()),
      alpha_(action.n_alpha_strings()), beta_(action.n_beta_strings()),
      active_(active_orbitals), diagonal_(context, action.size()),
      diagonal_h_scratch_(context,action.size(),1),
      exact_inverse_(std::move(exact_inverse)),
      preconditioner_(std::move(preconditioner)) {
  if (structures_ <= 0 || alpha_ <= 0 || beta_ <= 0 || active_ <= 0)
    throw std::invalid_argument("invalid selected cross-overlap dimensions");
  check_cublas(cublasSetPointerMode(context.blas(), CUBLAS_POINTER_MODE_HOST),
               "transport cuBLAS pointer mode");
}

void SelectedCrossOverlap::check_frame(linalg::ConstMatrixView frame) const {
  if (frame.rows != active_ || frame.cols != active_ ||
      frame.ld < active_ || frame.data == nullptr)
    throw std::invalid_argument("invalid transport orbital frame");
}

void SelectedCrossOverlap::check_structure(linalg::ConstMatrixView vector) const {
  if (vector.rows != structures_ || vector.cols != 1 ||
      vector.ld < structures_ || vector.data == nullptr)
    throw std::invalid_argument("invalid transport structure vector");
}

void SelectedCrossOverlap::check_structure(linalg::MatrixView vector) const {
  check_structure(static_cast<linalg::ConstMatrixView>(vector));
}

void SelectedCrossOverlap::check_gradient(linalg::MatrixView matrix) const {
  if (matrix.rows != active_ || matrix.cols != active_ ||
      matrix.ld < active_ || matrix.data == nullptr)
    throw std::invalid_argument("invalid transport frame gradient");
}

void SelectedCrossOverlap::frame_image(linalg::ConstMatrixView frame,
    linalg::ConstMatrixView coefficients, linalg::MatrixView carrier) const {
  check_frame(frame);
  check_structure(coefficients);
  if (carrier.rows != alpha_ || carrier.cols != beta_ || carrier.ld != alpha_)
    throw std::invalid_argument("invalid transport carrier");
  action_->expand_spin(coefficients,carrier);
  action_->update_frame(frame);
  action_->apply_orbital_transform(carrier);
}

void SelectedCrossOverlap::frame_direction(linalg::ConstMatrixView frame,
    linalg::ConstMatrixView direction, linalg::ConstMatrixView image,
    linalg::MatrixView directional_image) const {
  check_frame(frame);
  check_frame(direction);
  Matrix<double> generator(*context_,active_,active_);
  copy(*context_,generator.view(),direction);
  right_upper_solve(*context_,frame,generator.view(),false);
  action_->carrier_sigma().apply_generator(image,generator.view(),directional_image);
}

void SelectedCrossOverlap::apply(linalg::ConstMatrixView left_frame,
    linalg::ConstMatrixView right_frame,
    linalg::ConstMatrixView coefficients, linalg::MatrixView image) const {
  check_structure(image);
  Matrix<double> carrier(*context_,alpha_,beta_);
  frame_image(right_frame,coefficients,carrier.view());
  action_->update_frame(left_frame);
  action_->apply_adjoint_orbital_transform(carrier.view());
  action_->contract_spin(carrier.view(),image);
}

void SelectedCrossOverlap::apply_direction(linalg::ConstMatrixView left_frame,
    linalg::ConstMatrixView right_frame,
    linalg::ConstMatrixView left_direction,
    linalg::ConstMatrixView right_direction,
    linalg::ConstMatrixView coefficients, linalg::MatrixView image) const {
  check_structure(image);
  Matrix<double> right(*context_,alpha_,beta_);
  Matrix<double> direction(*context_,alpha_,beta_);
  Matrix<double> left_generator(*context_,active_,active_);
  Matrix<double> left_transpose(*context_,active_,active_);
  frame_image(right_frame,coefficients,right.view());
  frame_direction(right_frame,right_direction,right.view(),direction.view());
  copy(*context_,left_generator.view(),left_direction);
  right_upper_solve(*context_,left_frame,left_generator.view(),false);
  const double one=1.0,zero_value=0.0;
  check_cublas(cublasDgeam(context_->blas(),CUBLAS_OP_T,CUBLAS_OP_N,
      active_,active_,&one,left_generator.data(),active_,&zero_value,
      left_generator.data(),active_,left_transpose.data(),active_),
      "transport generator transpose");
  Matrix<double> left_term(*context_,alpha_,beta_);
  action_->carrier_sigma().apply_generator(right.view(),left_transpose.view(),
                                            left_term.view());
  axpy(*context_,direction.view(),1.0,left_term.view());
  action_->update_frame(left_frame);
  action_->apply_adjoint_orbital_transform(direction.view());
  action_->contract_spin(direction.view(),image);
}

void SelectedCrossOverlap::pullback(linalg::ConstMatrixView frame,
    linalg::ConstMatrixView moment, linalg::MatrixView gradient) const {
  check_frame(frame);
  check_frame(moment);
  check_gradient(gradient);
  copy(*context_,gradient,moment);
  right_upper_solve(*context_,frame,gradient,true);
  upper_only(*context_,gradient);
}

void SelectedCrossOverlap::pullback_direction(linalg::ConstMatrixView frame,
    linalg::ConstMatrixView direction, linalg::ConstMatrixView moment,
    linalg::ConstMatrixView moment_direction,
    linalg::MatrixView gradient_direction) const {
  check_frame(frame);
  check_frame(direction);
  check_frame(moment);
  check_frame(moment_direction);
  check_gradient(gradient_direction);
  Matrix<double> full_gradient(*context_,active_,active_);
  Matrix<double> correction(*context_,active_,active_);
  copy(*context_,full_gradient.view(),moment);
  right_upper_solve(*context_,frame,full_gradient.view(),true);
  copy(*context_,gradient_direction,moment_direction);
  right_upper_solve(*context_,frame,gradient_direction,true);
  gemm(*context_,correction.view(),full_gradient.view(),direction,1.0,0.0,
       Operation::none,Operation::transpose);
  right_upper_solve(*context_,frame,correction.view(),true);
  axpy(*context_,gradient_direction,-1.0,correction.view());
  upper_only(*context_,gradient_direction);
}

void SelectedCrossOverlap::adjoint(linalg::ConstMatrixView left_frame,
    linalg::ConstMatrixView right_frame,
    linalg::ConstMatrixView left_coefficients,
    linalg::ConstMatrixView right_coefficients,
    linalg::MatrixView left_gradient,
    linalg::MatrixView right_gradient) const {
  check_gradient(left_gradient);
  check_gradient(right_gradient);
  Matrix<double> left(*context_,alpha_,beta_), right(*context_,alpha_,beta_);
  Matrix<double> moment(*context_,active_,active_);
  frame_image(left_frame,left_coefficients,left.view());
  frame_image(right_frame,right_coefficients,right.view());
  action_->carrier_sigma().one_body_generator_adjoint(
      right.view(),left.view(),moment.view());
  pullback(left_frame,moment.view(),left_gradient);
  action_->carrier_sigma().one_body_generator_adjoint(
      left.view(),right.view(),moment.view());
  pullback(right_frame,moment.view(),right_gradient);
}

void SelectedCrossOverlap::adjoint_direction(
    linalg::ConstMatrixView left_frame,
    linalg::ConstMatrixView right_frame,
    linalg::ConstMatrixView left_direction,
    linalg::ConstMatrixView right_direction,
    linalg::ConstMatrixView left_coefficients,
    linalg::ConstMatrixView right_coefficients,
    linalg::ConstMatrixView left_coeff_direction,
    linalg::ConstMatrixView right_coeff_direction,
    linalg::MatrixView left_gradient_direction,
    linalg::MatrixView right_gradient_direction) const {
  check_gradient(left_gradient_direction);
  check_gradient(right_gradient_direction);
  Matrix<double> left(*context_,alpha_,beta_), right(*context_,alpha_,beta_);
  Matrix<double> dleft(*context_,alpha_,beta_), dright(*context_,alpha_,beta_);
  Matrix<double> scratch(*context_,alpha_,beta_);
  Matrix<double> moment(*context_,active_,active_);
  Matrix<double> dmoment(*context_,active_,active_);
  frame_image(left_frame,left_coefficients,left.view());
  frame_image(right_frame,right_coefficients,right.view());
  frame_direction(left_frame,left_direction,left.view(),dleft.view());
  frame_direction(right_frame,right_direction,right.view(),dright.view());
  if (!left_coeff_direction.empty()) {
    frame_image(left_frame,left_coeff_direction,scratch.view());
    axpy(*context_,dleft.view(),1.0,scratch.view());
  }
  if (!right_coeff_direction.empty()) {
    frame_image(right_frame,right_coeff_direction,scratch.view());
    axpy(*context_,dright.view(),1.0,scratch.view());
  }
  action_->carrier_sigma().one_body_generator_adjoint(
      right.view(),left.view(),moment.view());
  action_->carrier_sigma().one_body_generator_adjoint(
      dright.view(),left.view(),dmoment.view());
  Matrix<double> extra(*context_,active_,active_);
  action_->carrier_sigma().one_body_generator_adjoint(
      right.view(),dleft.view(),extra.view());
  axpy(*context_,dmoment.view(),1.0,extra.view());
  pullback_direction(left_frame,left_direction,moment.view(),dmoment.view(),
                     left_gradient_direction);
  action_->carrier_sigma().one_body_generator_adjoint(
      left.view(),right.view(),moment.view());
  action_->carrier_sigma().one_body_generator_adjoint(
      dleft.view(),right.view(),dmoment.view());
  action_->carrier_sigma().one_body_generator_adjoint(
      left.view(),dright.view(),extra.view());
  axpy(*context_,dmoment.view(),1.0,extra.view());
  pullback_direction(right_frame,right_direction,moment.view(),dmoment.view(),
                     right_gradient_direction);
}

void SelectedCrossOverlap::apply_metric(linalg::ConstMatrixView frame,
    linalg::ConstMatrixView coefficients, linalg::MatrixView image) const {
  check_frame(frame);
  check_structure(coefficients);
  check_structure(image);
  action_->update_frame(frame);
  action_->apply_overlap(coefficients,image);
}

void SelectedCrossOverlap::solve_metric(linalg::ConstMatrixView frame,
    linalg::ConstMatrixView rhs, linalg::MatrixView solution,
    linalg::ConstMatrixView initial, linalg::MetricSolveStats* stats) const {
  check_frame(frame);
  check_structure(rhs);
  check_structure(solution);
  if (!initial.empty()) check_structure(initial);
  if (rhs.data==solution.data)
    throw std::invalid_argument("transport metric RHS and solution alias");
  linalg::MetricSolveStats local;
  if (!stats) stats=&local;
  *stats={};
  Matrix<double> r(*context_,structures_,1),z(*context_,structures_,1);
  Matrix<double> p(*context_,structures_,1),ap(*context_,structures_,1);
  Matrix<double> image(*context_,structures_,1);
  const double rhs_norm=norm(*context_,rhs);
  if (!std::isfinite(rhs_norm))
    throw std::invalid_argument("non-finite metric RHS");
  action_->update_frame(frame);
  const auto apply=[&](linalg::ConstMatrixView input,linalg::MatrixView output) {
    ++stats->actions;
    action_->apply_overlap(input,output);
  };
  const auto certify=[&] {
    ++stats->certifications;
    apply(solution,image.view());
    copy(*context_,r.view(),rhs);
    axpy(*context_,r.view(),-1.0,image.view());
    const double residual_norm=norm(*context_,r.view());
    stats->relative_residual=residual_norm/rhs_norm;
    return residual_norm;
  };
  const auto precondition=[&] {
    if (preconditioner_) {
      preconditioner_(r.view(),z.view());
    } else {
      jacobi_kernel<<<(structures_+255)/256,256,0,context_->stream()>>>(
          z.data(),r.data(),diagonal_.data(),structures_);
      check_cuda(cudaGetLastError(),"transport Jacobi kernel");
    }
  };
  if (rhs_norm==0.0) {
    zero(*context_,solution);
    return;
  }
  const double target=std::sqrt(std::numeric_limits<double>::epsilon())*rhs_norm;
  if (exact_inverse_) {
    exact_inverse_(rhs,solution);
    const double residual_norm=certify();
    if (!std::isfinite(residual_norm) || residual_norm>target)
      throw std::runtime_error(
          "exact selected metric inverse failed true-residual certification");
    return;
  }
  action_->diagonal_model(diagonal_h_scratch_.view(),
                          {diagonal_.data(),structures_,1,structures_});
  if (initial.empty()) {
    zero(*context_,solution);
    copy(*context_,r.view(),rhs);
  } else {
    copy(*context_,solution,initial);
    if (certify()<=target) return;
  }
  precondition();
  copy(*context_,p.view(),z.view());
  double rz=dot(*context_,r.view(),z.view());
  double cycle_norm=norm(*context_,r.view());
  int cycle=0;
  while (true) {
    if (!(rz>0.0) || !std::isfinite(rz))
      throw std::runtime_error("metric inverse is not positive definite");
    ++stats->iterations;
    apply(p.view(),ap.view());
    double pap=dot(*context_,p.view(),ap.view());
    if (!(pap>0.0) || !std::isfinite(pap)) {
      const double residual_norm=certify();
      if (residual_norm<=target) return;
      precondition();
      copy(*context_,p.view(),z.view());
      apply(p.view(),ap.view());
      pap=dot(*context_,p.view(),ap.view());
      rz=dot(*context_,r.view(),z.view());
      cycle_norm=residual_norm;
      cycle=0;
      if (!(pap>0.0) || !std::isfinite(pap) || !(rz>0.0))
        throw std::runtime_error("metric is singular in tangent space");
    }
    const double alpha=rz/pap;
    if (!std::isfinite(alpha))
      throw std::runtime_error("non-finite metric CG step");
    axpy(*context_,solution,alpha,p.view());
    axpy(*context_,r.view(),-alpha,ap.view());
    const double recurrence_norm=norm(*context_,r.view());
    if (!std::isfinite(recurrence_norm))
      throw std::runtime_error("non-finite metric residual");
    if (recurrence_norm<=target || ++cycle==structures_) {
      const double residual_norm=certify();
      if (!std::isfinite(residual_norm))
        throw std::runtime_error("non-finite true metric residual");
      if (residual_norm<=target) return;
      if (!(cycle_norm-residual_norm>
          std::numeric_limits<double>::epsilon()*
              std::max(cycle_norm,rhs_norm))) {
        std::ostringstream message;
        message.precision(17);
        message<<"metric CG stagnated above certified tolerance"
               <<": dimension="<<structures_
               <<", iterations="<<stats->iterations
               <<", certifications="<<stats->certifications
               <<", rhs_norm="<<rhs_norm
               <<", residual="<<residual_norm
               <<", target="<<target
               <<", cycle_initial_residual="<<cycle_norm;
        throw std::runtime_error(message.str());
      }
      precondition();
      copy(*context_,p.view(),z.view());
      rz=dot(*context_,r.view(),z.view());
      cycle_norm=residual_norm;
      cycle=0;
      continue;
    }
    precondition();
    const double next_rz=dot(*context_,r.view(),z.view());
    const double beta=next_rz/rz;
    if (!(next_rz>0.0) || !std::isfinite(beta))
      throw std::runtime_error("invalid metric CG recurrence");
    const double old_p=beta;
    check_cublas(cublasDscal(context_->blas(),structures_,&old_p,p.data(),1),
                 "transport CG scale");
    axpy(*context_,p.view(),1.0,z.view());
    rz=next_rz;
  }
}

SingleStateCiTransport::SingleStateCiTransport(
    Context& context, SelectedCrossOverlap& overlap,
    linalg::ConstMatrixView origin_frame,
    linalg::ConstMatrixView selected_state,
    linalg::ConstMatrixView hamiltonian_image,
    linalg::ConstMatrixView metric_image,
    double normalized_weight,
    linalg::ConstMatrixView physical_origin_coefficient_gradient)
    : context_(&context), overlap_(&overlap), structures_(overlap.size()),
      active_(overlap.active_orbitals()),
      coordinate_scale_(std::sqrt(2.0*normalized_weight)),
      origin_frame_(context,active_,active_),
      selected_(context,structures_,1),
      origin_gradient_(context,structures_,1), zero_(context,active_,active_) {
  if (!(normalized_weight>0.0) || !std::isfinite(normalized_weight) ||
      !std::isfinite(coordinate_scale_))
    throw std::invalid_argument("invalid CI transport state weight");
  if (origin_frame.rows!=active_ || origin_frame.cols!=active_ ||
      origin_frame.ld<active_ || !origin_frame.data ||
      selected_state.rows!=structures_ || selected_state.cols!=1 ||
      selected_state.ld<structures_ || !selected_state.data ||
      hamiltonian_image.rows!=structures_ || hamiltonian_image.cols!=1 ||
      hamiltonian_image.ld<structures_ || !hamiltonian_image.data ||
      metric_image.rows!=structures_ || metric_image.cols!=1 ||
      metric_image.ld<structures_ || !metric_image.data)
    throw std::invalid_argument("invalid CI transport origin data");
  copy(context,origin_frame_.view(),origin_frame);
  copy(context,selected_.view(),selected_state);
  zero(context,zero_.view());
  const double selected_norm=dot(context,selected_state,metric_image);
  const double tolerance=1.0e3*std::numeric_limits<double>::epsilon()*
      std::max(1,structures_);
  if (!std::isfinite(selected_norm) ||
      std::abs(selected_norm-1.0)>tolerance)
    throw std::invalid_argument("CI transport origin must be metric normalized");
  const double state_h=dot(context,selected_state,hamiltonian_image);
  if (physical_origin_coefficient_gradient.empty()) {
    copy(context,origin_gradient_.view(),hamiltonian_image);
    axpy(context,origin_gradient_.view(),-state_h,metric_image);
    const double scale_squared=coordinate_scale_*coordinate_scale_;
    check_cublas(cublasDscal(context.blas(),structures_,&scale_squared,
        origin_gradient_.data(),1),"CI transport origin gradient scale");
  } else {
    if (physical_origin_coefficient_gradient.rows!=structures_ ||
        physical_origin_coefficient_gradient.cols!=1 ||
        physical_origin_coefficient_gradient.ld<structures_ ||
        !physical_origin_coefficient_gradient.data)
      throw std::invalid_argument("invalid physical CI origin gradient");
    copy(context,origin_gradient_.view(),physical_origin_coefficient_gradient);
  }
}

SingleStateCiTransportPoint SingleStateCiTransport::retract(
    linalg::ConstMatrixView trial_frame,
    linalg::ConstMatrixView origin_frame_direction,
    linalg::ConstMatrixView base_coefficients) const {
  if (trial_frame.rows!=active_ || trial_frame.cols!=active_ ||
      trial_frame.ld<active_ || !trial_frame.data ||
      origin_frame_direction.rows!=active_ ||
      origin_frame_direction.cols!=active_ ||
      origin_frame_direction.ld<active_ || !origin_frame_direction.data ||
      base_coefficients.rows!=structures_ || base_coefficients.cols!=1 ||
      base_coefficients.ld<structures_ || !base_coefficients.data)
    throw std::invalid_argument("invalid CI transport retraction inputs");
  SingleStateCiTransportPoint point{
      Matrix<double>(*context_,active_,active_),
      Matrix<double>(*context_,active_,active_),
      Matrix<double>(*context_,structures_,1),
      Matrix<double>(*context_,structures_,1)};
  copy(*context_,point.frame.view(),trial_frame);
  copy(*context_,point.origin_direction.view(),origin_frame_direction);
  copy(*context_,point.base_coefficients.view(),base_coefficients);
  Matrix<double> rhs(*context_,structures_,1),direction(*context_,structures_,1);
  overlap_->apply(point.frame.view(),origin_frame_.view(),
                  point.base_coefficients.view(),rhs.view());
  overlap_->apply_direction(point.frame.view(),origin_frame_.view(),
      zero_.view(),point.origin_direction.view(),selected_.view(),
      direction.view());
  axpy(*context_,rhs.view(),1.0,direction.view());
  overlap_->solve_metric(point.frame.view(),rhs.view(),
                         point.coefficients.view(),
                         point.base_coefficients.view());
  return point;
}

SingleStateCiTransportModel SingleStateCiTransport::evaluate_model(
    const SingleStateCiTransportPoint& point, double physical_energy,
    linalg::ConstMatrixView physical_coefficient_gradient) const {
  if (!std::isfinite(physical_energy) ||
      physical_coefficient_gradient.rows!=structures_ ||
      physical_coefficient_gradient.cols!=1 ||
      physical_coefficient_gradient.ld<structures_ ||
      !physical_coefficient_gradient.data)
    throw std::invalid_argument("invalid CI transport model inputs");
  SingleStateCiTransportModel model{
      physical_energy,
      Matrix<double>(*context_,active_,active_),
      Matrix<double>(*context_,active_,active_),
      Matrix<double>(*context_,structures_,1)};
  Matrix<double> rhs(*context_,structures_,1),lambda(*context_,structures_,1);
  Matrix<double> left(*context_,active_,active_);
  Matrix<double> right(*context_,active_,active_);
  Matrix<double> dleft(*context_,active_,active_);
  Matrix<double> dright(*context_,active_,active_);
  copy(*context_,rhs.view(),physical_coefficient_gradient);
  axpy(*context_,rhs.view(),-1.0,origin_gradient_.view());
  overlap_->solve_metric(point.frame.view(),rhs.view(),lambda.view());

  overlap_->adjoint(point.frame.view(),origin_frame_.view(),lambda.view(),
                    point.base_coefficients.view(),left.view(),right.view());
  copy(*context_,model.current_upper.view(),left.view());
  overlap_->adjoint_direction(point.frame.view(),origin_frame_.view(),
      zero_.view(),point.origin_direction.view(),lambda.view(),selected_.view(),
      {},{},dleft.view(),dright.view());
  axpy(*context_,model.current_upper.view(),1.0,dleft.view());
  overlap_->adjoint(point.frame.view(),point.frame.view(),lambda.view(),
                    point.coefficients.view(),left.view(),right.view());
  axpy(*context_,model.current_upper.view(),-1.0,left.view());
  axpy(*context_,model.current_upper.view(),-1.0,right.view());
  overlap_->adjoint(point.frame.view(),origin_frame_.view(),lambda.view(),
                    selected_.view(),left.view(),model.origin_upper.view());

  overlap_->apply(origin_frame_.view(),point.frame.view(),lambda.view(),
                  model.structure_covector.view());
  axpy(*context_,model.structure_covector.view(),1.0,origin_gradient_.view());
  const double inverse_scale=1.0/coordinate_scale_;
  check_cublas(cublasDscal(context_->blas(),structures_,&inverse_scale,
      model.structure_covector.data(),1),"CI transport structure covector scale");

  copy(*context_,rhs.view(),point.coefficients.view());
  axpy(*context_,rhs.view(),-1.0,point.base_coefficients.view());
  model.energy-=dot(*context_,origin_gradient_.view(),rhs.view());
  return model;
}

}  // namespace xmvb::cuda
