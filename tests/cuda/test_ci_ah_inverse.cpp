#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "core/pair_index.hpp"
#include "cuda/vb/action.hpp"
#include "cuda/vb/ci.hpp"
#include "cuda/vb/tangent.hpp"
#include "integrals/active/two_electron.hpp"
#include "mcscf/ci/strings.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/assembly/hamiltonian_overlap.hpp"
#include "vbscf/structures/orthogonal_ci/metric.hpp"
#include "vbscf/structures/overlap/spin_metric.hpp"

namespace {

using Matrix=Eigen::MatrixXd;
using Vector=Eigen::VectorXd;
using DeviceMatrix=xmvb::cuda::Matrix<double>;
using Strings=std::vector<std::vector<int>>;

void require(bool condition,const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void near(const Matrix& actual,const Matrix& expected,const char* branch,
          const char* message) {
  if (actual.rows()!=expected.rows() || actual.cols()!=expected.cols()) {
    std::cerr<<branch<<": "<<message<<" shape actual="<<actual.rows()<<'x'
             <<actual.cols()<<" expected="<<expected.rows()<<'x'
             <<expected.cols()<<'\n';
    throw std::runtime_error(message);
  }
  const Matrix error=actual-expected;
  if (!actual.allFinite() || !expected.allFinite() ||
      !(error.norm()<5.0e-10*std::max(1.0,expected.norm()))) {
    std::cerr<<branch<<": "<<message<<" abs_norm="<<error.norm()
             <<" max_abs="<<error.cwiseAbs().maxCoeff()
             <<" expected_norm="<<expected.norm()
             <<" actual_norm="<<actual.norm()
             <<" actual="<<actual.transpose()
             <<" expected="<<expected.transpose()<<'\n';
    throw std::runtime_error(message);
  }
}
Matrix download(xmvb::cuda::Context& context,
                xmvb::linalg::ConstMatrixView vector) {
  Matrix host(vector.rows,vector.cols);
  xmvb::cuda::copy_to_host(context,host,vector);
  return host;
}
DeviceMatrix upload(xmvb::cuda::Context& context,const Matrix& host) {
  DeviceMatrix device(context,host.rows(),host.cols());
  xmvb::cuda::copy_to_device(context,device.view(),host);
  return device;
}
DeviceMatrix upload(xmvb::cuda::Context& context,const Vector& host) {
  DeviceMatrix device(context,host.size(),1);
  context.copy_to_device(device.data(),host.data(),host.size()*sizeof(double));
  return device;
}

struct Fixture {
  Strings strings=xmvb::mcscf::SpinStrings(2,1).occupations();
  std::unique_ptr<xmvb::vb::StructureAction> source;
  xmvb::vb::StructureDiagonal diagonal;
  explicit Fixture(bool complete) {
    constexpr double half_root=0.7071067811865475244;
    std::vector<std::vector<xmvb::vb::StructureExpansionTerm>> terms(4);
    terms[0].push_back({0,1.0});
    terms[1].push_back({1,half_root});
    terms[2].push_back({1,half_root});
    if (complete) terms[3].push_back({2,1.0});
    const Strings alpha{strings[0],strings[0],strings[1],strings[1]};
    const Strings beta{strings[0],strings[1],strings[0],strings[1]};
    const auto topology=xmvb::vb::build_same_spin_pair_topology(alpha,beta,2);
    Matrix upper=Matrix::Identity(2,2);
    upper(0,0)=1.08;upper(0,1)=0.19;upper(1,1)=0.91;
    const Matrix overlap=upper.transpose()*upper;
    const std::vector<double> overlap_data(
        overlap.data(),overlap.data()+overlap.size());
    Matrix one(2,2);
    one<<-0.84,0.13,0.13,-0.27;
    xmvb::integrals::ActiveSpaceTwoElectronResult eri;
    auto& packed=eri.packed_active_two_electron_integrals;
    packed.resize(xmvb::integrals::packed_active_two_electron_integral_count(2));
    for (std::size_t i=0;i<packed.size();++i)
      packed[i]=0.06*std::cos(0.41*(i+1));
    diagonal=xmvb::vb::FullDeterminantStructureHamiltonianOverlapBuilder{}
        .build_davidson_jacobi_preconditioner(alpha,beta,terms,overlap_data,
            one,2,eri,complete?3:2,topology);
    source=std::make_unique<xmvb::vb::StructureAction>(terms,complete?3:2,
        topology,overlap_data,one,eri,2,nullptr,
        xmvb::vb::StructureActionMode::ActionOnly);
  }
};

void check(bool full,bool spin_model=false) {
  const char* branch=full ? "FULL":(spin_model ? "SPIN":"SELECTED");
  Fixture fixture(full);
  const int n=full?3:2;
  xmvb::vb::SpinMetric cpu_spin(*fixture.source,fixture.strings,
                                fixture.strings,4096);
  std::unique_ptr<xmvb::vb::StructureMetricInverse> cpu_full;
  if (full)
    cpu_full=std::make_unique<xmvb::vb::StructureMetricInverse>(
        *fixture.source,fixture.strings,fixture.strings,true,4096,4096);
  xmvb::cuda::Context context(128ULL*1024ULL*1024ULL);
  xmvb::cuda::StructureAction action(context,*fixture.source);
  xmvb::cuda::SpinMetric spin(context,*fixture.source,fixture.strings,
                              fixture.strings,4096);
  xmvb::cuda::StructureCi ci(context,action,spin,full);
  DeviceMatrix dh(context,n,1),dm(context,n,1);
  action.diagonal_model(dh.view(),dm.view());
  const Vector diagonal_h=download(context,dh.view()).col(0);
  Vector diagonal_m=download(context,dm.view()).col(0);
  near(diagonal_h,fixture.diagonal.hamiltonian,branch,"CI self H differs from independent CPU model");
  near(diagonal_m,fixture.diagonal.overlap,branch,"CI self M differs from independent CPU model");
  const auto dense=fixture.source->apply(Matrix::Identity(n,n));
  if (!full) {
    require(std::string(ci.preconditioner_name())=="selected_metric_inverse",
            "expected bounded selected metric inverse");
    // Actual M includes coherent cross terms; the AH Jacobi diagonal does
    // not. Never replace the independently constructed CPU model with M_II.
    require((diagonal_m-dense.overlap.diagonal()).norm()>1e-5,
        "CI fixture does not distinguish Jacobi and exact overlap diagonals");
  }
  Vector selected(n),rhs(n);
  for (int i=0;i<n;++i) {
    selected[i]=0.31+0.13*std::sin(0.7+i);
    rhs[i]=0.21*std::cos(0.4+0.8*i);
  }
  selected/=std::sqrt(selected.dot(dense.overlap*selected));
  const double selected_energy=selected.dot(dense.hamiltonian*selected);
  constexpr double eigenvalue=0.073;
  const auto cpu_inverse=[&](const Vector& x) -> Vector {
    if (full) {
      const Vector energies=Vector::Constant(1,selected_energy);
      const Matrix image=cpu_full->apply_carrier_inverse(x,energies,eigenvalue);
      return image.col(0);
    }
    Vector model=diagonal_h-(selected_energy+eigenvalue)*diagonal_m;
    if (spin_model) {
      model=model.cwiseAbs().cwiseMax(1.0e-8);
      const Matrix image=cpu_spin.solve_diagonal_model(x,model);
      return image.col(0);
    }
    for (int i=0;i<n;++i) if (std::abs(model[i])<1.0e-8) model[i]=1.0e-8;
    return (x.array()/model.array()).matrix();
  };

  auto d_rhs=upload(context,rhs);
  DeviceMatrix d_out(context,n,1);
  if (!spin_model) {
    ci.inverse_augmented_hessian_model(d_rhs.view(),selected_energy,eigenvalue,
                                      d_out.view());
    near(download(context,d_out.view()),cpu_inverse(rhs),branch,
         "CUDA ambient AH inverse differs from CPU model oracle");
  }
  auto spin_diagonal=upload(context,Vector((diagonal_h.array()-
      (selected_energy+eigenvalue)*diagonal_m.array()).abs().max(1.0e-8)));

  auto d_selected=upload(context,selected);
  xmvb::cuda::SingleStateCiActions actions;
  actions.hamiltonian_metric=[&](xmvb::linalg::ConstMatrixView x,
      xmvb::linalg::MatrixView h,xmvb::linalg::MatrixView m) {
    action.apply(x,h,m);
  };
  actions.metric=[&](xmvb::linalg::ConstMatrixView x,
                     xmvb::linalg::MatrixView m) {
    action.apply_overlap(x,m);
  };
  actions.augmented_inverse=[&](xmvb::linalg::ConstMatrixView x,double shift,
                                 xmvb::linalg::MatrixView y) {
    if (spin_model) spin.solve_diagonal_model(x,spin_diagonal.view(),y);
    else ci.inverse_augmented_hessian_model(x,selected_energy,shift,y);
  };
  actions.maximum_component=[&](xmvb::linalg::ConstMatrixView x) {
    return action.maximum_orthogonal_component(x);
  };
  actions.inverse_constraint=full ? xmvb::cuda::CiInverseConstraint::Carrier :
      (spin_model ? xmvb::cuda::CiInverseConstraint::Spin:
          xmvb::cuda::CiInverseConstraint::Diagonal);
  xmvb::cuda::SingleStateCiTangent tangent(context,n,d_selected.view(),
                                            selected_energy,1.0,actions);
  DeviceMatrix d_z(context,n-1,1),d_ambient(context,n,1);
  Vector coordinates(n-1);
  for (int i=0;i<n-1;++i) coordinates[i]=0.14-0.09*i;
  xmvb::cuda::copy_to_device(context,d_z.view(),coordinates);
  tangent.expand_coordinates(d_z.view(),d_ambient.view());
  const Vector ambient=download(context,d_ambient.view()).col(0);
  const Vector constraint=dense.overlap*selected;
  Vector expected;
  if (full) {
    // CPU FULL uses P_L R P_R; R is signed and need not be positive.
    const Vector right=ambient-constraint*selected.dot(ambient);
    const Vector solved=cpu_inverse(right);
    expected=solved-selected*constraint.dot(solved);
  } else if (spin_model) {
    const Vector solved=cpu_inverse(ambient);
    const Vector constraint_inverse=cpu_inverse(constraint);
    expected=solved-constraint_inverse*
        (constraint.dot(solved)/constraint.dot(constraint_inverse));
  } else expected=cpu_inverse(ambient);
  DeviceMatrix d_inverse(context,n-1,1),d_projected(context,n,1);
  tangent.apply_inverse_augmented_hessian_diagonal(d_z.view(),eigenvalue,
                                                    d_inverse.view());
  tangent.expand_coordinates(d_inverse.view(),d_projected.view());
  const Vector horizontal=expected-constraint*
      (constraint.dot(expected)/constraint.squaredNorm());
  near(download(context,d_projected.view()),horizontal,branch,
       "CUDA constrained AH inverse differs from CPU tangent oracle");

  DeviceMatrix d_response(context,n,1);
  tangent.coefficient_response_coordinates(d_z.view(),d_response.view());
  const Vector response=download(context,d_response.view()).col(0);
  const double cpu_max=fixture.source->orthogonalize_structure_block(response)
      .cwiseAbs().maxCoeff();
  const double gpu_max=tangent.maximum_coefficient_component_coordinates(
      d_z.view());
  if (!(std::abs(cpu_max-gpu_max)<5.0e-10*std::max(1.0,cpu_max))) {
    std::cerr<<branch
             <<": CUDA CI clipping differs from CPU orthogonal carrier maximum"
             <<" abs_error="<<std::abs(cpu_max-gpu_max)
             <<" gpu_max="<<gpu_max<<" cpu_max="<<cpu_max<<'\n';
    throw std::runtime_error(
        "CUDA CI clipping differs from CPU orthogonal carrier maximum");
  }
}

}  // namespace

int main() {
  try {
    check(false);
    check(false,true);
    check(true);
    std::cout<<"CUDA CI AH inverse/carrier clipping CPU oracle: PASS\n";
  } catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
  }
}
