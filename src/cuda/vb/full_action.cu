#include "cuda/vb/full_action.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <nvtx3/nvtx3.hpp>

#include "cuda/linalg/blas.hpp"
#include "cuda/vb/action.hpp"
#include "cuda/vb/product.hpp"
#include "cuda/vb/transport.hpp"
#include "cuda/vb/transform.hpp"

namespace xmvb::cuda {
namespace {

__global__ void identity_matrix(linalg::MatrixView matrix) {
  const int index=blockIdx.x*blockDim.x+threadIdx.x;
  if (index<matrix.rows*matrix.cols) {
    const int row=index%matrix.rows,column=index/matrix.rows;
    matrix.data[row+matrix.ld*column]=row==column ? 1.0:0.0;
  }
}

__global__ void divide_diagonal_column(linalg::MatrixView source,
    linalg::ConstMatrixView diagonal,linalg::ConstMatrixView energy,double shift) {
  const std::int64_t index=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if (index<source.rows*source.cols) {
    const auto row=index%source.rows,column=index/source.rows;
    double denominator=diagonal.data[row+diagonal.ld*column]-energy.data[0]-shift;
    if (fabs(denominator)<1.0e-8) denominator=1.0e-8;
    source.data[row+source.ld*column]/=denominator;
  }
}

__global__ void beta_identity_columns(linalg::MatrixView values,int first) {
  const int column=blockIdx.x*blockDim.x+threadIdx.x;
  if (column<values.cols) values.data[first+column+values.ld*column]=1.0;
}

void clear(Context& context,linalg::MatrixView image) {
  check_cuda(cudaMemset2DAsync(image.data,image.ld*sizeof(double),0,
      image.rows*sizeof(double),image.cols,context.stream()),
      "clear FULL selected product map");
}

}  // namespace

struct FullAction::Impl {
  Context& context;
  const StructureAction& action;
  StructureSourceFactors sources;
  ExteriorTransform identity_alpha,identity_beta;
  ProductContraction identity_product;

  Impl(Context& ctx,const StructureAction& source)
      :context(ctx),action(source),sources(ctx,source),
       identity_alpha(ctx,source.alpha_transform()),
       identity_beta(ctx,source.beta_transform()),
       identity_product(ctx,source.orthogonal_view(),identity_alpha,identity_beta,
                        source.orthogonal_view().support) {
    const int active=identity_alpha.n_orbitals();
    if (active!=identity_beta.n_orbitals() || sources.source_count()<=0)
      throw std::invalid_argument("CUDA FULL action dimensions disagree");
    Matrix<double> orbital_identity(ctx,active,active);
    identity_matrix<<<(active*active+255)/256,256,0,ctx.stream()>>>(
        orbital_identity.view());
    check_cuda(cudaGetLastError(),"CUDA FULL identity frame");
    identity_alpha.update_upper(orbital_identity.view());
    identity_beta.update_upper(orbital_identity.view());
  }

  void apply_orbital(ProductMap map,linalg::ConstMatrixView source,
                     linalg::MatrixView image) const {
    const int extent=sources.source_extent();
    Matrix<double> alpha(context,action.n_alpha_strings(),extent);
    Matrix<double> beta(context,action.n_beta_strings(),extent);
    const bool adjoint=map==ProductMap::AdjointOrbital ||
        map==ProductMap::InverseAdjointOrbital;
    const bool inverse=map==ProductMap::InverseOrbital ||
        map==ProductMap::InverseAdjointOrbital;
    const auto& alpha_map=action.alpha_transform();
    const auto& beta_map=action.beta_transform();
    for (std::int64_t column=0;column<source.cols;++column) {
      auto target=image.column(column);
      clear(context,target);
      for (int first=0;first<sources.source_count();first+=extent) {
        const int count=std::min(extent,sources.source_count()-first);
        auto a=alpha.view().columns(0,count),b=beta.view().columns(0,count);
        sources.load(first,count,source.column(column),a,b);
        alpha_map.apply_left(a,adjoint,inverse);
        beta_map.apply_left(b,adjoint,inverse);
        sources.contract(a,b,target);
      }
    }
  }

  void apply_diagonal(linalg::ConstMatrixView source,linalg::MatrixView image,
                      linalg::ConstMatrixView energies,double shift) const {
    const int extent=sources.source_extent();
    Matrix<double> alpha(context,action.n_alpha_strings(),extent);
    Matrix<double> beta(context,action.n_beta_strings(),extent);
    Matrix<double> diagonal(context,action.n_alpha_strings(),extent);
    for (std::int64_t column=0;column<source.cols;++column) {
      auto target=image.column(column);
      clear(context,target);
      for (int first=0;first<sources.source_count();) {
        const int first_beta=sources.source_column(first);
        int count=1;
        while (count<extent && first+count<sources.source_count() &&
               sources.source_column(first+count)==first_beta+count)
          ++count;
        auto a=alpha.view().columns(0,count);
        auto b=beta.view().columns(0,count);
        auto values=diagonal.view().columns(0,count);
        sources.load(first,count,source.column(column),a,b);
        action.carrier_sigma().diagonal_columns(first_beta,values);
        const auto elements=static_cast<std::int64_t>(a.rows)*count;
        divide_diagonal_column<<<(elements+255)/256,256,0,context.stream()>>>(
            a,values,energies.column(column),shift);
        check_cuda(cudaGetLastError(),"CUDA FULL tiled diagonal inverse");
        sources.contract(a,b,target);
        first+=count;
      }
    }
  }

  void apply_inverse_metric(linalg::ConstMatrixView source,
                            linalg::MatrixView image) const {
    const int extent=sources.source_extent();
    Matrix<double> alpha(context,action.n_alpha_strings(),extent);
    Matrix<double> beta(context,action.n_beta_strings(),extent);
    for (std::int64_t column=0;column<source.cols;++column) {
      auto target=image.column(column);
      clear(context,target);
      for (int first=0;first<sources.source_count();first+=extent) {
        const int count=std::min(extent,sources.source_count()-first);
        auto a=alpha.view().columns(0,count),b=beta.view().columns(0,count);
        sources.load(first,count,source.column(column),a,b);
        action.alpha_transform().apply_left(a,true,true);
        action.beta_transform().apply_left(b,true,true);
        action.alpha_transform().apply_left(a,false,true);
        action.beta_transform().apply_left(b,false,true);
        sources.contract(a,b,target);
      }
    }
  }

  void apply_compound(linalg::ConstMatrixView source,linalg::MatrixView image,
      linalg::ConstMatrixView energies,double shift,bool diagonal_inverse) const {
    const int extent=sources.source_extent();
    const int alpha_count=action.n_alpha_strings();
    const int beta_count=action.n_beta_strings();
    Matrix<double> source_alpha(context,alpha_count,extent);
    Matrix<double> source_beta(context,beta_count,extent);
    Matrix<double> target_alpha(context,alpha_count,extent);
    Matrix<double> target_beta(context,beta_count,extent);
    Matrix<double> diagonal;
    if (diagonal_inverse) diagonal=Matrix<double>(context,alpha_count,extent);
    for (std::int64_t rhs=0;rhs<source.cols;++rhs) {
      auto target=image.column(rhs);
      clear(context,target);
      for (int first_beta=0;first_beta<beta_count;first_beta+=extent) {
        const int target_count=std::min(extent,beta_count-first_beta);
        auto carrier=target_alpha.view().columns(0,target_count);
        clear(context,carrier);
        for (int first_source=0;first_source<sources.source_count();first_source+=extent) {
          const int source_count=std::min(extent,sources.source_count()-first_source);
          auto a=source_alpha.view().columns(0,source_count);
          auto b=source_beta.view().columns(0,source_count);
          sources.load(first_source,source_count,source.column(rhs),a,b);
          action.alpha_transform().apply_left(a,true,true);
          action.beta_transform().apply_left(b,true,true);
          const linalg::ConstMatrixView beta_rows{
              b.data+first_beta,target_count,source_count,b.ld};
          gemm(context,carrier,a,beta_rows,1.0,1.0,
               Operation::none,Operation::transpose);
        }
        if (diagonal_inverse) {
          auto values=diagonal.view().columns(0,target_count);
          action.carrier_sigma().diagonal_columns(first_beta,values);
          const auto elements=static_cast<std::int64_t>(alpha_count)*target_count;
          divide_diagonal_column<<<(elements+255)/256,256,0,context.stream()>>>(
              carrier,values,energies.column(rhs),shift);
          check_cuda(cudaGetLastError(),"CUDA FULL tiled compound diagonal inverse");
        }
        action.alpha_transform().apply_left(carrier,false,true);
        auto beta_basis=target_beta.view().columns(0,target_count);
        clear(context,beta_basis);
        beta_identity_columns<<<(target_count+255)/256,256,0,context.stream()>>>(
            beta_basis,first_beta);
        check_cuda(cudaGetLastError(),"CUDA FULL beta identity panel");
        action.beta_transform().apply_left(beta_basis,false,true);
        sources.contract(carrier,beta_basis,target);
      }
    }
  }

  void apply_hamiltonian(linalg::ConstMatrixView source,
                         linalg::MatrixView image) const {
    const ProductIncidence incidence{action.term_offsets_data(),
        action.term_structures_data(),action.term_weights_data(),action.size()};
    Matrix<double> overlap(context,action.size(),1);
    for (std::int64_t column=0;column<source.cols;++column)
      identity_product.apply_selected(incidence,source.column(column),
          action.carrier_sigma(),image.column(column),overlap.view(),
          context.available_device_bytes());
  }
};

FullAction::FullAction(Context& context,const StructureAction& action)
    :impl_(std::make_unique<Impl>(context,action)) {}
FullAction::~FullAction()=default;

void FullAction::apply(ProductMap map,linalg::ConstMatrixView source,
    linalg::MatrixView image,linalg::ConstMatrixView energies,
    double eigenvalue) const {
  const auto& p=*impl_;
  if (source.rows!=p.action.size() || source.cols<=0 || source.ld<source.rows ||
      image.rows!=source.rows || image.cols!=source.cols || image.ld<image.rows ||
      !source.data || !image.data || source.data==image.data ||
      !std::isfinite(eigenvalue))
    throw std::invalid_argument("CUDA FULL selected product map dimensions");
  const nvtx3::scoped_range range{"ci.full_product_map"};
  switch (map) {
    case ProductMap::Orbital:
    case ProductMap::InverseOrbital:
    case ProductMap::AdjointOrbital:
    case ProductMap::InverseAdjointOrbital:
      p.apply_orbital(map,source,image);
      return;
    case ProductMap::IdentityHamiltonian:
      p.apply_hamiltonian(source,image);
      return;
    case ProductMap::IdentityDiagonalInverse:
      if (energies.rows!=1 || energies.cols!=source.cols || energies.ld<1 ||
          !energies.data)
        throw std::invalid_argument("CUDA FULL product inverse energies dimensions");
      p.apply_diagonal(source,image,energies,eigenvalue);
      return;
    case ProductMap::InverseMetric:
      p.apply_inverse_metric(source,image);
      return;
    case ProductMap::InverseOrbitalDiagonalInverseAdjointOrbital:
      if (energies.rows!=1 || energies.cols!=source.cols || energies.ld<1 ||
          !energies.data)
        throw std::invalid_argument("CUDA FULL compound inverse energies dimensions");
      p.apply_compound(source,image,energies,eigenvalue,true);
      return;
  }
  throw std::invalid_argument("CUDA FULL product map type");
}

}  // namespace xmvb::cuda
