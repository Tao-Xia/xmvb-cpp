#pragma once

#include <cmath>
#include <array>
#include <numeric>
#include <iostream>
#include <stdexcept>

#include "c12_fixture.hpp"
#include "cuda/vb/point.hpp"
#include "cuda/linalg/blas.hpp"
#include "cuda/linalg/spectrum.hpp"
#include "vbscf/orbitals/preparation/input.hpp"

namespace xmvb::cuda::test {

/** Physical derivative diagnostic, not an orbital optimization benchmark.
 * Only the test harness downloads derivatives for an independent difference.
 * No CPU energy, gradient, HVP, or baseline timing is evaluated here.
 */
inline void benchmark_point(Context& context,StructureAction& action,
    const C12Fixture& fixture,const linalg::DavidsonResult& ci) {
  using Host=Eigen::MatrixXd;
  const auto& input=fixture.loaded.input;
  const auto& orbitals=input.orbital_preparation_input;
  const int n_bf=orbitals.n_basis_functions,n_orb=orbitals.n_orbitals;
  const int n_str=action.size();
  Host raw=Eigen::Map<const Host>(orbitals.orbital_value_table.data(),n_bf,n_orb);
  Host coefficients=Eigen::Map<const Host>(ci.eigenpairs.eigenvector_matrix.data(),n_str,1);
  Host dr=Host::Zero(n_bf,n_orb),dc(n_str,1);
  for (int o=0;o<n_orb;++o)
    for (int j=0;j<vb::differentiable_sparse_orbital_parameter_count(orbitals,o);++j)
      dr(j,o)=std::sin(0.37*(j+1)*(o+2));
  if (dr.norm()>0.0) dr/=dr.norm();
  for (int i=0;i<n_str;++i) dc(i,0)=std::cos(0.19*(i+1));
  dc*=coefficients.norm()/std::max(1.0,dc.norm());
  // Keep a finite CI residual so the frame and normalization terms are tested.
  coefficients+=0.001*dc;
  std::cout<<"point_ci_coefficient_norm="<<coefficients.norm()<<'\n';
  const auto upload=[&](const Host& x) {
    Matrix<double> result(context,x.rows(),x.cols());
    copy_to_device(context,result.view(),x);
    return result;
  };
  auto draw=upload(raw),dcoeff=upload(coefficients),direction=upload(dr),ci_direction=upload(dc);
  auto core=upload(input.ao_integral_input.ao_core_hamiltonian_matrix);
  ExactAoPairOperator ao(context,input.ao_integral_input.pair_graph,n_bf);
  const AoMetric ao_metric(context,orbitals.ao_overlap_matrix);
  VbPoint point(context,orbitals,action,ao,core.view(),ao_metric.overlap(),fixture.loaded.nuclear_repulsion_energy);
  Matrix<double> gr(context,n_bf,n_orb),gc(context,n_str,1);
  Matrix<double> hr(context,n_bf,n_orb),hc(context,n_str,1);
  auto start=Clock::now();
  const double energy=point.evaluate(draw.view(),dcoeff.view());
  point.gradient(gr.view(),gc.view());
  context.synchronize();
  std::cout<<"point_energy_gradient_seconds="<<elapsed(start,Clock::now())<<'\n'
           <<"point_total_energy_hartree="<<energy<<'\n';
  {
    // Independent small-C12 device oracle, deliberately resident only in this
    // diagnostic harness. Production endpoints remain source-backed.
    Matrix<double> y(context,action.n_alpha_strings(),action.n_beta_strings()),
        hy(context,y.rows(),y.cols()),one(context,input.orbital_preparation_input.n_active_orbitals,
        input.orbital_preparation_input.n_active_orbitals),
        pair(context,point.pair_kernel().rows,point.pair_kernel().cols),
        moment(context,one.rows(),one.cols());
    action.expand_spin(dcoeff.view(),y.view());action.apply_orbital_transform(y.view());
    DirectCiSigma dense(context,action.carrier_sigma());
    dense.update_integrals(point.one_electron(),point.pair_kernel());
    {
      ProductContraction product(context,action.orthogonal_view(),action.alpha_transform(),
          action.beta_transform(),action.orthogonal_view().support);
      ProductEndpoint source{Matrix<double>(context,n_str,1)};
      copy_to_device(context,source.coefficients.view(),coefficients);
      const ProductIncidence incidence{action.term_offsets_data(),action.term_structures_data(),
          action.term_weights_data(),n_str};
      auto jet=product.jet(incidence,dense,source,point.geometry().frame_transform(),{},{},nullptr,
          256ULL*1024*1024);
      if (!source.images || !source.images->complete())
        throw std::runtime_error("point diagnostic could not retain compact primal");
      std::vector<const ProductColumns*> panels(source.images->size());
      for (int k=0;k<source.images->size();++k) panels[k]=&source.images->at(k);
      std::vector<int> betas(y.cols()),rows(y.rows());
      std::iota(betas.begin(),betas.end(),0);std::iota(rows.begin(),rows.end(),0);
      Matrix<double> compact_y(context,y.rows(),y.cols());
      read_product_columns(context,panels,source.extent,0,betas,rows,compact_y.view());
      Host a(y.rows(),y.cols()),b(y.rows(),y.cols());
      copy_to_host(context,a,y.view());copy_to_host(context,b,compact_y.view());
      std::cout<<"point_dense_source_error="<<(a-b).norm()/std::max(1.0,a.norm())<<'\n';
    }
    dense.apply_with_adjoint(y.view(),y.view(),hy.view(),one.view(),pair.view());
    const auto dot=[&](linalg::ConstMatrixView a,linalg::ConstMatrixView b) {
      double value=0.0;check_cublas(cublasDdot(context.blas(),blas_int(a.rows*a.cols),
          a.data,1,b.data,1,&value),"point oracle dot");return value;
    };
    const double norm=dot(y.view(),y.view()),e=dot(y.view(),hy.view())/norm;
    const double inv=1.0/norm,minus=-e,two=2.0/norm;
    check_cublas(cublasDscal(context.blas(),blas_int(one.rows()*one.cols()),&inv,one.data(),1),"oracle one scale");
    check_cublas(cublasDscal(context.blas(),blas_int(pair.rows()*pair.cols()),&inv,pair.data(),1),"oracle pair scale");
    check_cublas(cublasDaxpy(context.blas(),blas_int(y.rows()*y.cols()),&minus,y.data(),1,hy.data(),1),"oracle residual");
    check_cublas(cublasDscal(context.blas(),blas_int(y.rows()*y.cols()),&two,hy.data(),1),"oracle residual scale");
    const std::array<linalg::ConstMatrixView,1> left{hy.view()};
    const std::array<linalg::MatrixView,1> moments{moment.view()};
    dense.generator_moments(left,y.view(),moments);
    const auto error=[&](linalg::ConstMatrixView a,linalg::ConstMatrixView b,bool upper=false) {
      Host x(a.rows,a.cols),z(b.rows,b.cols);copy_to_host(context,x,a);copy_to_host(context,z,b);
      if (upper) {
        x.template triangularView<Eigen::StrictlyLower>().setZero();
        z.template triangularView<Eigen::StrictlyLower>().setZero();
      }
      return (x-z).norm()/std::max(1.0,z.norm());
    };
    std::cout<<"point_dense_norm="<<norm<<'\n'
        <<"point_dense_energy="<<e<<'\n'
        <<"point_dense_one_error="<<error(point.one_electron_adjoint(),one.view())<<'\n'
        <<"point_dense_pair_error="<<error(point.pair_kernel_adjoint(),pair.view())<<'\n'
        <<"point_dense_moment_error="<<error(point.frame_moment(),moment.view(),true)<<'\n';
    Matrix<double> tiled_one(context,one.rows(),one.cols()),tiled_pair(context,pair.rows(),pair.cols());
    const CiCarrierRequest request{y.view(),{},nullptr,tiled_one.view(),tiled_pair.view()};
    dense.apply_supported_responses({&request,1},y.view(),*action.orthogonal_view().support,
        256ULL*1024*1024);
    check_cublas(cublasDscal(context.blas(),blas_int(one.rows()*one.cols()),&inv,tiled_one.data(),1),"oracle tiled one scale");
    check_cublas(cublasDscal(context.blas(),blas_int(pair.rows()*pair.cols()),&inv,tiled_pair.data(),1),"oracle tiled pair scale");
    std::cout<<"point_resident_reader_one_error="<<error(tiled_one.view(),one.view())<<'\n'
        <<"point_resident_reader_pair_error="<<error(tiled_pair.view(),pair.view())<<'\n';
    dense.reset_workspace();
  }
  start=Clock::now();
  const auto before=context.counters();
  point.hessian(direction.view(),ci_direction.view(),hr.view(),hc.view());
  context.synchronize();
  const auto after=context.counters();
  std::cout<<"point_joint_hvp_seconds="<<elapsed(start,Clock::now())<<'\n'
           <<"point_hvp_context_h2d_bytes="<<after.host_to_device_bytes-before.host_to_device_bytes<<'\n'
           <<"point_hvp_context_d2h_bytes="<<after.device_to_host_bytes-before.device_to_host_bytes<<'\n';
  Host hraw(n_bf,n_orb),hci(n_str,1),graw(n_bf,n_orb),gci(n_str,1);
  copy_to_host(context,hraw,hr.view());copy_to_host(context,hci,hc.view());
  copy_to_host(context,graw,gr.view());copy_to_host(context,gci,gc.view());
  double step=1.0e-5;
  Host plus_raw(n_bf,n_orb),plus_ci(n_str,1),minus_raw(n_bf,n_orb),minus_ci(n_str,1);
  const auto at=[&](double sign,Host& orbital_gradient,Host& structure_gradient) {
    const Host r=raw+sign*step*dr,c=coefficients+sign*step*dc;
    copy_to_device(context,draw.view(),r);copy_to_device(context,dcoeff.view(),c);
    const double e=point.evaluate(draw.view(),dcoeff.view());
    point.gradient(gr.view(),gc.view());
    copy_to_host(context,orbital_gradient,gr.view());
    copy_to_host(context,structure_gradient,gc.view());
    return e;
  };
  const double plus=at(1.0,plus_raw,plus_ci),minus=at(-1.0,minus_raw,minus_ci);
  const Host fd_raw=(plus_raw-minus_raw)/(2.0*step),fd_ci=(plus_ci-minus_ci)/(2.0*step);
  const double analytic=(graw.array()*dr.array()).sum()+(gci.array()*dc.array()).sum();
  const double gradient_error=std::abs((plus-minus)/(2.0*step)-analytic)/std::max(1.0,std::abs(analytic));
  const double hvp_error=std::hypot((hraw-fd_raw).norm(),(hci-fd_ci).norm())/
      std::max(1.0,std::hypot(fd_raw.norm(),fd_ci.norm()));
  std::cout<<"point_gradient_difference_error="<<gradient_error<<'\n'
           <<"point_hvp_difference_error="<<hvp_error<<'\n'
           <<"point_scope=physical_derivatives_not_scf\n";
  // Diagnose truncation versus cancellation without changing the original
  // 1e-5 difference or its acceptance tolerance below.
  for (double probe:{1.0e-4,1.0e-3}) {
    step=probe;
    const double ep=at(1.0,plus_raw,plus_ci),em=at(-1.0,minus_raw,minus_ci);
    const Host fd_r=(plus_raw-minus_raw)/(2.0*step),fd_c=(plus_ci-minus_ci)/(2.0*step);
    std::cout<<"point_difference_probe="<<step<<','
        <<std::abs((ep-em)/(2.0*step)-analytic)/std::max(1.0,std::abs(analytic))<<','
        <<std::hypot((hraw-fd_r).norm(),(hci-fd_c).norm())/
            std::max(1.0,std::hypot(fd_r.norm(),fd_c.norm()))<<'\n';
  }
  if (!std::isfinite(gradient_error) || !std::isfinite(hvp_error) ||
      gradient_error>2.0e-6 || hvp_error>2.0e-6)
    throw std::runtime_error("C12 GPU physical derivative difference failed");
}

}  // namespace xmvb::cuda::test
