#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include "cuda/memory/buffer.hpp"
#include "cuda/vb/chart.hpp"
#include "cuda/vb/orbitals.hpp"
#include "vbscf/core/contracts/orbital_type.hpp"
#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/physical_metric.hpp"
#include "vbscf/optimization/preconditioners/hessian_diagonal.hpp"

namespace {

using Matrix=Eigen::MatrixXd;
using DeviceMatrix=xmvb::cuda::Matrix<double>;

void near(const Matrix& actual,const Matrix& expected,const char* label) {
  if (actual.rows()!=expected.rows() || actual.cols()!=expected.cols() ||
      !actual.allFinite() ||
      (actual-expected).norm()>2e-13*std::max(1.0,expected.norm()))
    throw std::runtime_error(label);
}

void near_accepted(const Matrix& actual,const Matrix& expected,const char* label) {
  if (actual.rows()!=expected.rows() || actual.cols()!=expected.cols() ||
      !actual.allFinite() ||
      (actual-expected).norm()>2e-9*std::max(1.0,expected.norm()))
    throw std::runtime_error(label);
}

DeviceMatrix upload(xmvb::cuda::Context& context,const Matrix& host) {
  DeviceMatrix result(context,host.rows(),host.cols());
  xmvb::cuda::copy_to_device(context,result.view(),host);
  return result;
}

Matrix download(xmvb::cuda::Context& context,
                xmvb::linalg::ConstMatrixView source) {
  Matrix result(source.rows,source.cols);
  xmvb::cuda::copy_to_host(context,result,source);
  return result;
}

void check_map() {
  xmvb::cuda::Context context;
  // Exact block values would normally be exported from one accepted
  // OrbitalChart. The GPU map must not infer or alter its quotient.
  const std::vector<xmvb::cuda::OrbitalTangentBlock> blocks{
      {{0,2},0,1,{0.7,-0.4}},
      {{4,5,6},1,2,{0.9,0.2,-0.1,-0.3,0.6,0.5}},
      {{8,9,11},3,1,{0.15,-0.8,0.25}}};
  xmvb::cuda::OrbitalChartMap map(context,4,3,4,{3,3,4},blocks);
  Matrix basis=Matrix::Zero(12,4);
  for (const auto& block:blocks)
    for (int j=0;j<block.reduced_size;++j)
      for (int i=0;i<static_cast<int>(block.raw_slots.size());++i)
        basis(block.raw_slots[i],block.reduced_offset+j)=
            block.basis[j*block.raw_slots.size()+i];
  Matrix accepted(4,3),covector(4,3),step(4,1);
  for (int j=0;j<3;++j)
    for (int i=0;i<4;++i) {
      accepted(i,j)=0.2+0.13*j+0.07*i;
      covector(i,j)=std::sin(0.4+2*i+j);
    }
  step<<0.17,-0.29,0.31,0.14;
  const double scale=-0.43;
  auto daccepted=upload(context,accepted);
  auto dcovector=upload(context,covector);
  auto dstep=upload(context,step);
  DeviceMatrix raw(context,4,3),trial(context,4,3),reduced(context,4,1);
  map.expand(dstep.view(),raw.view());
  const Eigen::VectorXd expanded=basis*step;
  const Matrix expected_raw=Eigen::Map<const Matrix>(expanded.data(),4,3);
  near(download(context,raw.view()),expected_raw,"CUDA chart expansion differs");
  map.project(dcovector.view(),reduced.view());
  const Matrix expected_reduced=basis.transpose()*
      Eigen::Map<const Eigen::VectorXd>(covector.data(),12);
  near(download(context,reduced.view()),expected_reduced,
       "CUDA chart adjoint differs");
  map.retract(daccepted.view(),dstep.view(),scale,trial.view());
  Matrix expected_trial=accepted+scale*expected_raw;
  expected_trial(3,0)=0.0;
  expected_trial(3,1)=0.0;
  near(download(context,trial.view()),expected_trial,
       "CUDA chart retraction differs");
  const double left=(covector.array()*expected_raw.array()).sum();
  const double right=(step.array()*expected_reduced.array()).sum();
  if (std::abs(left-right)>2e-13)
    throw std::runtime_error("CUDA chart gradient/step pairing differs");
  // Fixed HAO support slots are untouched by the additive retraction.
  const Matrix actual=download(context,trial.view());
  if (actual(1,0)!=accepted(1,0) || actual(3,1)!=0.0 ||
      actual(2,2)!=accepted(2,2))
    throw std::runtime_error("CUDA chart changed a fixed sparse slot");
}

xmvb::vb::OrbitalPreparationInput fixture(bool oeo,int inactive) {
  xmvb::vb::OrbitalPreparationInput input;
  input.n_basis_functions=5;
  input.n_orbitals=inactive+2;
  input.n_active_orbitals=2;
  input.n_active_electrons=2;
  input.n_total_electrons=2*inactive+2;
  input.orbital_type=oeo?xmvb::vb::kOrbitalTypeOeo:xmvb::vb::kOrbitalTypeHao;
  input.ao_overlap_matrix=Matrix::Identity(5,5);
  input.ao_overlap_matrix.array()+=0.03;
  input.ao_overlap_matrix(0,3)+=0.01;
  input.ao_overlap_matrix(3,0)+=0.01;
  const int norb=static_cast<int>(input.n_orbitals);
  input.orbital_basis_index_table.assign(5*norb,0);
  input.orbital_value_table.assign(5*norb,0.0);
  input.orbital_basis_counts.resize(norb);
  input.original_orbital_basis_counts.resize(norb);
  const std::vector<std::vector<int>> hao=inactive==2?
      std::vector<std::vector<int>>{{1,2,3},{1,2,3},{2,3,4},{1,4,5}}:inactive?
      std::vector<std::vector<int>>{{1,2,3},{2,3,4},{1,4,5}}:
      std::vector<std::vector<int>>{{1,2,3},{3,4,5}};
  for (int o=0;o<norb;++o) {
    const std::vector<int> support=oeo?std::vector<int>{1,2,3,4,5}:hao[o];
    input.orbital_basis_counts[o]=static_cast<int>(support.size());
    input.original_orbital_basis_counts[o]=static_cast<int>(support.size());
    for (int j=0;j<static_cast<int>(support.size());++j) {
      input.orbital_basis_index_table[o*5+j]=support[j];
      input.orbital_value_table[o*5+j]=j==0?0.87+0.11*o:
          0.07*std::sin(0.3+o+2*j)+0.13;
    }
  }
  if (!oeo) input.original_orbital_basis_counts.back()=2;
  return input;
}

Matrix raw_table(const xmvb::vb::OrbitalPreparationInput& input) {
  return Eigen::Map<const Matrix>(input.orbital_value_table.data(),
      input.n_basis_functions,input.n_orbitals);
}

void check_accepted(bool oeo,int inactive,bool complete=true) {
  xmvb::cuda::Context context;
  auto input=fixture(oeo,inactive);
  const int basis=static_cast<int>(input.n_basis_functions);
  const int orbitals=static_cast<int>(input.n_orbitals);
  xmvb::cuda::OrbitalGeometry geometry(context,input);
  geometry.prepare();
  const Matrix normalized=download(context,geometry.normalized());
  const int occupied=static_cast<int>(input.n_active_orbitals)+inactive;
  const bool full=oeo && complete;
  xmvb::vb::SparseParameterLayout layout(input);
  xmvb::vb::OrbitalChart reference(input,layout,normalized.leftCols(occupied),
      normalized,nullptr,false,full,nullptr,
      xmvb::vb::OrbitalChart::Coordinates::Unwhitened);
  xmvb::cuda::AcceptedOrbitalChart chart(context,input,full);
  chart.rebuild(geometry);
  if (chart.reduced_size()!=reference.reduced_size() ||
      chart.rank_signature()!=reference.rank_signature())
    throw std::runtime_error("CUDA accepted chart rank differs from CPU chart");
  const int q=chart.reduced_size();
  Matrix gpu_basis=Matrix::Zero(basis*orbitals,q);
  Matrix cpu_basis=Matrix::Zero(basis*orbitals,q);
  for (int j=0;j<q;++j) {
    Matrix e=Matrix::Zero(q,1);
    e(j,0)=1.0;
    auto de=upload(context,e);
    DeviceMatrix direction(context,basis,orbitals);
    chart.expand(de.view(),direction.view());
    const Matrix gpu=download(context,direction.view());
    gpu_basis.col(j)=Eigen::Map<const Eigen::VectorXd>(gpu.data(),gpu.size());
    const Eigen::VectorXd packed=reference.expand_step(e);
    const auto& indices=layout.differentiable_parameter_indices();
    for (int i=0;i<packed.size();++i)
      cpu_basis(indices[i],j)=packed[i];
  }
  if (q) {
    const Matrix gpu_projector=gpu_basis*
        (gpu_basis.transpose()*gpu_basis).ldlt().solve(gpu_basis.transpose());
    const Matrix cpu_projector=cpu_basis*
        (cpu_basis.transpose()*cpu_basis).ldlt().solve(cpu_basis.transpose());
    near_accepted(gpu_projector,cpu_projector,"CUDA accepted chart quotient span differs");
  }
  Matrix full_gradient(basis,orbitals);
  for (int j=0;j<orbitals;++j)
    for (int i=0;i<basis;++i)
      full_gradient(i,j)=std::sin(0.3+0.7*i+0.2*j);
  auto dgradient=upload(context,full_gradient);
  DeviceMatrix reduced(context,q,1),inverse(context,q,1),metric_image(context,q,1);
  chart.project(dgradient.view(),reduced.view());
  DeviceMatrix projected_vector(context,q,1);
  chart.project_vector(dgradient.view(),projected_vector.view());
  const Eigen::VectorXd packed_gradient_for_projection=layout.gather_from_full(
      std::vector<double>(full_gradient.data(),full_gradient.data()+full_gradient.size()));
  const Eigen::VectorXd cpu_projected_vector=
      reference.project_vector(packed_gradient_for_projection).reduced_gradient;
  DeviceMatrix expanded_vector(context,basis,orbitals);
  chart.expand(projected_vector.view(),expanded_vector.view());
  Matrix cpu_expanded_vector=Matrix::Zero(basis,orbitals);
  const Eigen::VectorXd cpu_packed_vector=reference.expand_step(cpu_projected_vector);
  const auto& projection_indices=layout.differentiable_parameter_indices();
  for (int i=0;i<cpu_packed_vector.size();++i)
    cpu_expanded_vector.data()[projection_indices[i]]=cpu_packed_vector[i];
  near_accepted(download(context,expanded_vector.view()),cpu_expanded_vector,
       "CUDA accepted chart vector projection differs");
  chart.inverse_metric(reduced.view(),inverse.view());
  chart.metric(inverse.view(),metric_image.view());
  near_accepted(download(context,metric_image.view()),download(context,reduced.view()),
       "CUDA accepted chart metric inverse differs");
  if (q) {
    DeviceMatrix physical_image(context,q,1);
    chart.physical_metric(inverse.view(),physical_image.view());
    const Matrix ambient=download(context,physical_image.view());
    const Matrix direction=download(context,inverse.view());
    const double gpu_squared=(direction.array()*ambient.array()).sum();
    DeviceMatrix raw_physical(context,basis,orbitals);
    chart.expand(inverse.view(),raw_physical.view());
    const Matrix raw_host=download(context,raw_physical.view());
    const Eigen::VectorXd packed=layout.gather_from_full(
        std::vector<double>(raw_host.data(),raw_host.data()+raw_host.size()));
    const xmvb::vb::OrbitalPhysicalMetric cpu_metric(input,full);
    const double cpu_squared=cpu_metric.squared_norm(layout,packed);
    if (std::abs(gpu_squared-cpu_squared)>
        1e-8*std::max(1.0,std::abs(cpu_squared)))
      throw std::runtime_error("CUDA coupled physical metric differs from CPU metric");
    const Eigen::VectorXd packed_action=cpu_metric.apply(layout,packed);
    Eigen::VectorXd full_action=Eigen::VectorXd::Zero(basis*orbitals);
    const auto& raw_indices=layout.differentiable_parameter_indices();
    for (int i=0;i<packed_action.size();++i)
      full_action[raw_indices[i]]=packed_action[i];
    const Matrix expected_action=gpu_basis.transpose()*full_action;
    near_accepted(ambient,expected_action,
        "CUDA coupled physical metric adjoint differs from CPU metric");
    if (!std::isfinite(chart.max_component(inverse.view())))
      throw std::runtime_error("CUDA accepted chart rotation map is not finite");
  }
  Matrix trial=raw_table(input);
  DeviceMatrix dtrial(context,basis,orbitals);
  chart.retract(geometry.raw(),inverse.view(),0.09,dtrial.view());
  DeviceMatrix dstep_raw(context,basis,orbitals);
  chart.expand(inverse.view(),dstep_raw.view());
  trial+=0.09*download(context,dstep_raw.view());
  for (int j=0;j<orbitals;++j)
    for (int i=xmvb::vb::stored_sparse_orbital_coefficient_count(input,j);i<basis;++i)
      trial(i,j)=0.0;
  near_accepted(download(context,dtrial.view()),trial,"CUDA accepted chart additive retraction differs");

  const Eigen::VectorXd packed_gradient=layout.gather_from_full(
      std::vector<double>(full_gradient.data(),full_gradient.data()+full_gradient.size()));
  const auto& indices=layout.differentiable_parameter_indices();

  xmvb::vb::AnalyticOrbitalDiagonal analytic;
  analytic.mo_coefficients=download(context,chart.accepted_mo_frame());
  analytic.rotation_diagonal=Matrix(basis,basis);
  for (int j=0;j<basis;++j)
    for (int i=0;i<basis;++i)
      analytic.rotation_diagonal(i,j)=((i+2*j)%3==0?-1.0:1.0)*
          (0.8+0.17*i+0.11*j+0.09*std::sin(0.31+0.4*i+0.7*j));
  const int ncore=inactive;
  if (ncore) {
    const Matrix raw_core=normalized.leftCols(ncore);
    const Matrix gram=raw_core.transpose()*input.ao_overlap_matrix*raw_core;
    Eigen::SelfAdjointEigenSolver<Matrix> eig(gram);
    analytic.inactive_inverse_transform=eig.operatorInverseSqrt();
  }
  analytic.active_inverse_transform=download(context,
      geometry.frame_inverse_transform());
  xmvb::vb::OrbitalChart reference_analytic(input,layout,
      normalized.leftCols(occupied),normalized,nullptr,false,full,nullptr,
      xmvb::vb::OrbitalChart::Coordinates::Unwhitened);
  reference_analytic.set_curvature(input,nullptr,&analytic,
      xmvb::vb::OrbitalChart::CurvatureUse::AugmentedHessian);
  auto d_rotation=upload(context,analytic.rotation_diagonal);
  chart.set_hessian_diagonal(geometry,d_rotation.view());
  const Eigen::VectorXd cpu_analytic_covector=
      reference_analytic.project_reduced_gradient(packed_gradient);
  const double eigenvalue=0.073;
  const Eigen::VectorXd cpu_analytic_step=
      reference_analytic.apply_inverse_augmented_hessian_diagonal(
          cpu_analytic_covector,eigenvalue);
  const Eigen::VectorXd cpu_analytic_raw=reference_analytic.expand_step(
      cpu_analytic_step);
  DeviceMatrix gpu_analytic_step(context,q,1),gpu_analytic_raw(context,basis,orbitals);
  chart.apply_inverse_hessian_diagonal(reduced.view(),eigenvalue,
      gpu_analytic_step.view());
  chart.expand(gpu_analytic_step.view(),gpu_analytic_raw.view());
  Matrix cpu_analytic_full=Matrix::Zero(basis,orbitals);
  for (int i=0;i<cpu_analytic_raw.size();++i)
    cpu_analytic_full.data()[indices[i]]=cpu_analytic_raw[i];
  near_accepted(download(context,gpu_analytic_raw.view()),cpu_analytic_full,
       "CUDA analytic Hessian-diagonal raw step differs");
  const double cpu_max=reference_analytic.maximum_rotation_component(
      cpu_analytic_step);
  const double gpu_max=chart.max_component(gpu_analytic_step.view());
  if (std::abs(cpu_max-gpu_max)>2e-8*std::max(1.0,cpu_max))
    throw std::runtime_error("CUDA analytic rotation max differs");
}

}  // namespace

int main() {
  try {
    check_map();
    for (bool oeo:{false,true})
      for (bool inactive:{false,true}) check_accepted(oeo,inactive);
    check_accepted(false,2); // Equal inactive support, different fixed active tails.
    check_accepted(true,2);
    check_accepted(true,2,false); // Selected active rays must not be merged.
    std::cout<<"CUDA orbital chart map checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
  }
}
