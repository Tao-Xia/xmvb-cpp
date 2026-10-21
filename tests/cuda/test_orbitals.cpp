#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "cuda/memory/buffer.hpp"
#include "cuda/vb/orbitals.hpp"
#include "vbscf/core/contracts/orbital_type.hpp"
#include "vbscf/orbitals/preparation/preparer.hpp"

namespace {

using Matrix = Eigen::MatrixXd;
using Input = xmvb::vb::OrbitalPreparationInput;
using Geometry = xmvb::cuda::OrbitalGeometry;
using DeviceMatrix = xmvb::cuda::Matrix<double>;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void near(const Matrix& actual, const Matrix& expected, double tolerance,
          const std::string& label) {
  if (actual.rows()!=expected.rows() || actual.cols()!=expected.cols() ||
      !actual.allFinite() || !expected.allFinite() ||
      (actual-expected).norm() > tolerance*std::max(1.0,expected.norm()))
    throw std::runtime_error(label+" mismatch");
}

Matrix values(int rows,int cols,double phase) {
  Matrix result(rows,cols);
  for (int j=0;j<cols;++j)
    for (int i=0;i<rows;++i)
      result(i,j)=std::sin(phase+0.31*(i+1)*(j+2))+
                  0.17*std::cos(phase+0.53*(i+2)*(j+1));
  return result;
}

DeviceMatrix upload(xmvb::cuda::Context& context,const Matrix& host) {
  DeviceMatrix result(context,host.rows(),host.cols());
  context.copy_to_device(result.data(),host.data(),
                         host.size()*sizeof(double));
  return result;
}

Matrix download(xmvb::cuda::Context& context,
                xmvb::linalg::ConstMatrixView source) {
  Matrix result(source.rows,source.cols);
  xmvb::cuda::copy_to_host(context,result,source);
  return result;
}

Input fixture(bool full_oeo,bool inactive) {
  Input input;
  input.n_basis_functions=5;
  input.n_active_orbitals=2;
  input.n_orbitals=inactive?3:2;
  input.n_active_electrons=2;
  input.n_total_electrons=inactive?4:2;
  input.orbital_type=full_oeo?xmvb::vb::kOrbitalTypeOeo:
                              xmvb::vb::kOrbitalTypeHao;
  input.ao_overlap_matrix=Matrix::Identity(5,5);
  input.ao_overlap_matrix.array()+=0.04;
  input.ao_overlap_matrix(0,2)+=0.03;
  input.ao_overlap_matrix(2,0)+=0.03;
  const int basis=5,orbitals=static_cast<int>(input.n_orbitals);
  input.orbital_basis_index_table.assign(basis*orbitals,0);
  input.orbital_value_table.assign(basis*orbitals,0.0);
  input.orbital_basis_counts.resize(orbitals);
  input.original_orbital_basis_counts.resize(orbitals);
  if (full_oeo) {
    for (int o=0;o<orbitals;++o) {
      input.orbital_basis_counts[o]=basis;
      input.original_orbital_basis_counts[o]=basis;
      for (int j=0;j<basis;++j) {
        input.orbital_basis_index_table[o*basis+j]=j+1;
        input.orbital_value_table[o*basis+j]=
            (j==o?0.95:0.12*std::sin(0.4+o+2*j));
      }
    }
  } else {
    const std::vector<std::vector<int>> supports=inactive?
        std::vector<std::vector<int>>{{1,2},{2,3,4},{1,4,5}}:
        std::vector<std::vector<int>>{{1,2,3},{3,4,5}};
    for (int o=0;o<orbitals;++o) {
      const int count=static_cast<int>(supports[o].size());
      input.orbital_basis_counts[o]=count;
      input.original_orbital_basis_counts[o]=count;
      for (int j=0;j<count;++j) {
        input.orbital_basis_index_table[o*basis+j]=supports[o][j];
        input.orbital_value_table[o*basis+j]=
            (j==0?0.86:0.15+0.08*j+0.03*o);
      }
    }
    // The final stored support slot remains physically present but fixed in
    // the HAO variational chart. Its contribution to normalization is tested.
    input.original_orbital_basis_counts.back()=2;
  }
  return input;
}

struct Weights {
  Matrix t,p,a,r,f;
};

Weights weights(int basis,int active,double phase) {
  return {values(basis,active,phase),values(basis,basis,phase+0.3),
          values(active,active,phase+0.5),
          values(active,active,phase+0.7),
          values(basis,active,phase+0.9)};
}

double dot(const Matrix& left,const Matrix& right) {
  return (left.array()*right.array()).sum();
}

double objective(xmvb::cuda::Context& context,const Geometry& geometry,
                 const Weights& upstream) {
  return dot(upstream.t,download(context,geometry.active_auxiliary()))+
         dot(upstream.p,download(context,geometry.inactive_density()))+
         dot(upstream.a,download(context,geometry.active_overlap()))+
         dot(upstream.r,download(context,geometry.frame_transform()))+
         dot(upstream.f,download(context,geometry.orthogonal_active()));
}

Matrix pullback(xmvb::cuda::Context& context,const Geometry& geometry,
                const Weights& upstream) {
  auto gt=upload(context,upstream.t),gp=upload(context,upstream.p);
  auto ga=upload(context,upstream.a),gr=upload(context,upstream.r);
  auto gf=upload(context,upstream.f);
  DeviceMatrix output(context,geometry.basis_size(),geometry.orbital_count());
  geometry.pullback(gt.view(),gp.view(),ga.view(),gr.view(),gf.view(),
                    output.view());
  return download(context,output.view());
}

void verify_forward(xmvb::cuda::Context& context,const Input& input,
                    const Geometry& geometry) {
  const auto cpu=xmvb::vb::ActiveSpaceOrbitalPreparer{}.prepare(input);
  const int basis=static_cast<int>(input.n_basis_functions);
  const int active=static_cast<int>(input.n_active_orbitals);
  const int inactive=static_cast<int>(
      (input.n_total_electrons-input.n_active_electrons)/2);
  const Matrix t=cpu.auxiliary_orbital_matrix.middleCols(inactive,active);
  const Matrix a=Eigen::Map<const Matrix>(
      cpu.active_orbital_overlap_matrix.data(),active,active);
  const Matrix r=Eigen::LLT<Matrix>(a).matrixU();
  const Matrix u=r.triangularView<Eigen::Upper>().solve(
      Matrix::Identity(active,active));
  near(download(context,geometry.normalized()),
       cpu.physical_orbital_frame.normalized_orbital_matrix,2e-12,
       "normalized physical orbitals");
  near(download(context,geometry.inactive_density()),
       cpu.inactive_density_matrix,2e-12,"inactive density");
  near(download(context,geometry.active_auxiliary()),t,2e-12,
       "active auxiliary orbitals");
  near(download(context,geometry.active_overlap()),a,2e-12,
       "active overlap");
  near(download(context,geometry.frame_transform()),r,2e-12,
       "active upper Cholesky frame");
  near(download(context,geometry.orthogonal_active()),t*u,2e-12,
       "orthogonal active frame");
  require(basis==geometry.basis_size(),"basis dimension changed");
}

void run_case(bool full_oeo,bool inactive) {
  Input input=fixture(full_oeo,inactive);
  xmvb::cuda::Context context;
  const int n_bf=static_cast<int>(input.n_basis_functions);
  Matrix padded=Matrix::Constant(n_bf+2,n_bf,0.91);
  padded.topRows(n_bf)=input.ao_overlap_matrix;
  auto overlap=upload(context,padded);
  Geometry geometry(context,input,{overlap.data(),n_bf,n_bf,n_bf+2});
  geometry.prepare();
  verify_forward(context,input,geometry);
  const int basis=static_cast<int>(input.n_basis_functions);
  const int orbitals=static_cast<int>(input.n_orbitals);
  const int active=static_cast<int>(input.n_active_orbitals);
  const Matrix raw=Eigen::Map<const Matrix>(input.orbital_value_table.data(),
                                            basis,orbitals);
  const Weights w=weights(basis,active,0.31);
  const Weights dw=weights(basis,active,1.17);
  const Matrix gradient=pullback(context,geometry,w);
  Matrix direction=Matrix::Zero(basis,orbitals);
  for (int o=0;o<orbitals;++o) {
    const int count=xmvb::vb::differentiable_sparse_orbital_parameter_count(
        input,o);
    for (int j=0;j<count;++j) direction(j,o)=
        0.17*std::sin(0.3+o+2*j);
    for (int j=count;j<basis;++j)
      require(gradient(j,o)==0.0,"fixed support slot has nonzero gradient");
  }
  const double epsilon=1e-5;
  for (int o=0;o<orbitals;++o) {
    const int count=xmvb::vb::differentiable_sparse_orbital_parameter_count(
        input,o);
    for (int j=0;j<count;++j) {
      Matrix plus=raw,minus=raw;
      plus(j,o)+=epsilon;
      minus(j,o)-=epsilon;
      auto dplus=upload(context,plus),dminus=upload(context,minus);
      geometry.set_raw(dplus.view()); geometry.prepare();
      const double eplus=objective(context,geometry,w);
      geometry.set_raw(dminus.view()); geometry.prepare();
      const double eminus=objective(context,geometry,w);
      const double reference=(eplus-eminus)/(2*epsilon);
      require(std::abs(gradient(j,o)-reference)<
          3e-6*std::max(1.0,std::abs(reference)),
          "raw orbital gradient finite difference mismatch");
    }
  }
  auto dbase=upload(context,raw);
  geometry.set_raw(dbase.view()); geometry.prepare();
  auto ddirection=upload(context,direction);
  const auto tangent=geometry.direction(ddirection.view());
  auto gt=upload(context,w.t),gp=upload(context,w.p);
  auto ga=upload(context,w.a),gr=upload(context,w.r),gf=upload(context,w.f);
  auto dgt=upload(context,dw.t),dgp=upload(context,dw.p);
  auto dga=upload(context,dw.a),dgr=upload(context,dw.r),dgf=upload(context,dw.f);
  DeviceMatrix dh_device(context,basis,orbitals);
  geometry.pullback_direction(tangent,gt.view(),gp.view(),ga.view(),
      gr.view(),gf.view(),dgt.view(),dgp.view(),dga.view(),dgr.view(),
      dgf.view(),dh_device.view());
  const Matrix dh=download(context,dh_device.view());
  const Matrix plus=raw+epsilon*direction;
  const Matrix minus=raw-epsilon*direction;
  auto dplus=upload(context,plus),dminus=upload(context,minus);
  geometry.set_raw(dplus.view()); geometry.prepare();
  const Weights wplus={w.t+epsilon*dw.t,w.p+epsilon*dw.p,
                       w.a+epsilon*dw.a,w.r+epsilon*dw.r,w.f+epsilon*dw.f};
  const Matrix gplus=pullback(context,geometry,wplus);
  geometry.set_raw(dminus.view()); geometry.prepare();
  const Weights wminus={w.t-epsilon*dw.t,w.p-epsilon*dw.p,
                        w.a-epsilon*dw.a,w.r-epsilon*dw.r,w.f-epsilon*dw.f};
  const Matrix gminus=pullback(context,geometry,wminus);
  near(dh,(gplus-gminus)/(2*epsilon),2e-5,
       "orbital directional pullback");
}

}  // namespace

int main() {
  try {
    run_case(false,false);
    run_case(false,true);
    run_case(true,false);
    run_case(true,true);
    std::cout<<"CUDA orbital geometry and pullback checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
  }
}
