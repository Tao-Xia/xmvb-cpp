#include "cuda/vb/transport.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <nvtx3/nvtx3.hpp>

#include "cuda/ci/sigma.hpp"
#include "cuda/linalg/blas.hpp"
#include "cuda/vb/action.hpp"
#include "cuda/vb/metric.hpp"
#include "cuda/vb/transform.hpp"
#include "cuda/vb/panel.hpp"

namespace xmvb::cuda {
namespace {

void copy(Context& context, linalg::MatrixView destination,
          linalg::ConstMatrixView source) {
  if (destination.rows != source.rows || destination.cols != source.cols ||
      destination.ld < destination.rows || source.ld < source.rows)
    throw std::invalid_argument("transport copy dimensions");
  if (source.empty()) return;
  check_cuda(cudaMemcpy2DAsync(destination.data,destination.ld*sizeof(double),
      source.data,source.ld*sizeof(double),source.rows*sizeof(double),source.cols,
      cudaMemcpyDeviceToDevice,context.stream()),"transport device copy");
}

void zero(Context& context, linalg::MatrixView destination) {
  if (destination.empty()) return;
  check_cuda(cudaMemset2DAsync(destination.data,destination.ld*sizeof(double),0,
      destination.rows*sizeof(double),destination.cols,context.stream()),"transport zero");
}

void axpy(Context& context, linalg::MatrixView destination, double factor,
          linalg::ConstMatrixView source) {
  if (destination.rows != source.rows || destination.cols != source.cols)
    throw std::invalid_argument("transport AXPY dimensions");
  if (source.cols==1 || (source.ld==source.rows && destination.ld==destination.rows)) {
    check_cublas(cublasDaxpy(context.blas(),blas_int(source.rows*source.cols),&factor,
        source.data,1,destination.data,1),"transport packed AXPY");
    return;
  }
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

unsigned int source_grid(std::int64_t size) {
  return static_cast<unsigned int>(std::min<std::int64_t>(65535,(size+255)/256));
}

constexpr int moment_threads=256;
constexpr int moment_chunk=16*moment_threads;

__global__ void source_alpha(const int* source_groups,
    const int* source_offsets,const int* source_columns,int begin,int count,
    const std::int64_t* products,const std::int64_t* offsets,
    const std::int64_t* structures,const double* terms,
    const double* coefficients,int alpha,linalg::MatrixView output,
    linalg::MatrixView beta) {
  for (int j=blockIdx.x*blockDim.x+threadIdx.x;j<count;j+=gridDim.x*blockDim.x) {
    const int source=source_columns[begin+j];
    beta.data[source+beta.ld*j]=1.0;
    for (int k=source_offsets[source];k<source_offsets[source+1];++k) {
      const int group=source_groups[k];
      double value=0.0;
      for (auto t=offsets[group];t<offsets[group+1];++t)
        value+=terms[t]*coefficients[structures[t]];
      output.data[products[group]%alpha+output.ld*j]=value;
    }
  }
}

__global__ void source_contract(const std::int64_t* products,
    const std::int64_t* offsets,const std::int64_t* structures,const double* terms,
    int groups,int alpha,linalg::ConstMatrixView a,linalg::ConstMatrixView b,
    double* image) {
  for (int group=blockIdx.x*blockDim.x+threadIdx.x;group<groups;
       group+=gridDim.x*blockDim.x) {
    const auto product=products[group];
    double value=0.0;
    for (int j=0;j<a.cols;++j)
      value+=a.data[product%alpha+a.ld*j]*b.data[product/alpha+b.ld*j];
    for (auto t=offsets[group];t<offsets[group+1];++t)
      atomicAdd(image+structures[t],terms[t]*value);
  }
}

__global__ void source_contract_raw(const int* source_columns,
    const int* source_offsets,const int* source_groups,int begin,int count,
    const std::int64_t* products,const std::int64_t* offsets,
    const std::int64_t* structures,const double* terms,int alpha,
    linalg::ConstMatrixView a,double* image) {
  for (int j=blockIdx.y;j<count;j+=gridDim.y) {
    const int beta=source_columns[begin+j];
    for (int k=source_offsets[beta]+blockIdx.x*blockDim.x+threadIdx.x;
         k<source_offsets[beta+1];k+=gridDim.x*blockDim.x) {
      const int group=source_groups[k];
      const double value=a.data[products[group]%alpha+a.ld*j];
      for (auto t=offsets[group];t<offsets[group+1];++t)
        atomicAdd(image+structures[t],terms[t]*value);
    }
  }
}

__global__ void source_seed_rows(const std::int64_t* products,
    const std::int64_t* offsets,const std::int64_t* structures,
    const double* terms,const double* coefficients,const int* alpha_offsets,
    const int* beta_offsets,const int* beta_groups,int alpha,int beta,
    linalg::ConstMatrixView a,
    linalg::ConstMatrixView b,double* za,double* zb) {
  const int column=blockIdx.y*blockDim.x+threadIdx.x;
  for (int owner=blockIdx.x;owner<alpha+beta;owner+=gridDim.x) {
    const bool left=owner<alpha;
    const int row=left ? owner:owner-alpha;
    const int first=left ? alpha_offsets[row]:beta_offsets[row];
    const int last=left ? alpha_offsets[row+1]:beta_offsets[row+1];
    double value=0.0;
    for (int k=first;k<last;++k) {
      const int group=left ? k:beta_groups[k];
      double weight=0.0;
      if (threadIdx.x==0)
        for (auto term=offsets[group];term<offsets[group+1];++term)
          weight+=terms[term]*coefficients[structures[term]];
      weight=__shfl_sync(0xffffffff,weight,0);
      const auto product=products[group];
      if (column<a.cols)
        value+=weight*(left ? b.data[product/alpha+b.ld*column] :
            a.data[product%alpha+a.ld*column]);
    }
    if (column<a.cols) {
      if (left) za[row+static_cast<std::int64_t>(column)*a.rows]=value;
      else zb[row+static_cast<std::int64_t>(column)*b.rows]=value;
    }
  }
}

__global__ void source_seed_rows_three(const std::int64_t* products,
    const std::int64_t* offsets,const std::int64_t* structures,
    const double* terms,const double* coefficients,const int* alpha_offsets,
    const int* beta_offsets,const int* beta_groups,int alpha,int beta,
    linalg::ConstMatrixView a0,linalg::ConstMatrixView b0,
    linalg::ConstMatrixView a1,linalg::ConstMatrixView b1,
    linalg::ConstMatrixView a2,linalg::ConstMatrixView b2,
    linalg::MatrixView za0,linalg::MatrixView zb0,
    linalg::MatrixView za1,linalg::MatrixView zb1,
    linalg::MatrixView za2,linalg::MatrixView zb2) {
  const int column=blockIdx.y*blockDim.x+threadIdx.x;
  for (int owner=blockIdx.x;owner<alpha+beta;owner+=gridDim.x) {
    const bool left=owner<alpha;
    const int row=left ? owner:owner-alpha;
    const int first=left ? alpha_offsets[row]:beta_offsets[row];
    const int last=left ? alpha_offsets[row+1]:beta_offsets[row+1];
    double value0=0.0,value1=0.0,value2=0.0;
    for (int k=first;k<last;++k) {
      const int group=left ? k:beta_groups[k];
      double weight=0.0;
      if (threadIdx.x==0)
        for (auto term=offsets[group];term<offsets[group+1];++term)
          weight+=terms[term]*coefficients[structures[term]];
      weight=__shfl_sync(0xffffffff,weight,0);
      if (column<a0.cols) {
        const auto product=products[group];
        const int ai=product%alpha,bi=product/alpha;
        if (left) {
          value0+=weight*b0.data[bi+b0.ld*column];
          value1+=weight*b1.data[bi+b1.ld*column];
          value2+=weight*b2.data[bi+b2.ld*column];
        } else {
          value0+=weight*a0.data[ai+a0.ld*column];
          value1+=weight*a1.data[ai+a1.ld*column];
          value2+=weight*a2.data[ai+a2.ld*column];
        }
      }
    }
    if (column<a0.cols) {
      if (left) {
        za0.data[row+za0.ld*column]=value0;
        za1.data[row+za1.ld*column]=value1;
        za2.data[row+za2.ld*column]=value2;
      } else {
        zb0.data[row+zb0.ld*column]=value0;
        zb1.data[row+zb1.ld*column]=value1;
        zb2.data[row+zb2.ld*column]=value2;
      }
    }
  }
}

__global__ void moment_diagonal(const std::uint64_t* occupations,int strings,
    int orbitals,linalg::ConstMatrixView seed,linalg::ConstMatrixView image,
    double scale,linalg::MatrixView moment) {
  using Reduce=cub::BlockReduce<double,moment_threads>;
  __shared__ typename Reduce::TempStorage scratch;
  const auto total=static_cast<std::int64_t>(strings)*seed.cols;
  for (int orbital=blockIdx.x;orbital<orbitals;orbital+=gridDim.x) {
    const auto bit=std::uint64_t{1}<<orbital;
    for (auto chunk=blockIdx.y;static_cast<std::int64_t>(chunk)*moment_chunk<total;
         chunk+=gridDim.y) {
      const auto first=static_cast<std::int64_t>(chunk)*moment_chunk;
      const auto end=first+moment_chunk;
      const auto last=end<total ? end:total;
      double value=0.0;
      for (auto index=first+threadIdx.x;index<last;index+=blockDim.x) {
        const auto string=index%strings,column=index/strings;
        if (occupations[string]&bit)
          value+=seed.data[string+seed.ld*column]*
              image.data[string+image.ld*column];
      }
      value=Reduce(scratch).Sum(value);
      if (threadIdx.x==0)
        atomicAdd(moment.data+orbital+moment.ld*orbital,scale*value);
      __syncthreads();
    }
  }
}

__global__ void moment_shears(
    const ExteriorTransform::Shear* schedule,
    const vb::ExteriorOrbitalTransform::ShearEdge* edges,int shears,
    int orbitals,linalg::ConstMatrixView seed,linalg::ConstMatrixView image,
    double scale,linalg::MatrixView moment) {
  using Reduce=cub::BlockReduce<double,moment_threads>;
  __shared__ typename Reduce::TempStorage scratch;
  for (int shear=blockIdx.x;shear<shears;shear+=gridDim.x) {
    int remaining=shear,inserted=0,removed=0;
    for (int r=1;r<orbitals;++r) {
      if (remaining<r) { inserted=r-1-remaining; removed=r; break; }
      remaining-=r;
    }
    const auto part=schedule[shear];
    const auto total=static_cast<std::int64_t>(part.count)*seed.cols;
    for (auto chunk=blockIdx.y;static_cast<std::int64_t>(chunk)*moment_chunk<total;
         chunk+=gridDim.y) {
      const auto first=static_cast<std::int64_t>(chunk)*moment_chunk;
      const auto end=first+moment_chunk;
      const auto last=end<total ? end:total;
      double value=0.0;
      for (auto index=first+threadIdx.x;index<last;index+=blockDim.x) {
        const auto edge=index/seed.cols,column=index%seed.cols;
        const auto pair=edges[part.offset+edge];
        value+=pair.sign*seed.data[pair.target+seed.ld*column]*
            image.data[pair.source+image.ld*column];
      }
      value=Reduce(scratch).Sum(value);
      if (threadIdx.x==0)
        atomicAdd(moment.data+inserted+moment.ld*removed,scale*value);
      __syncthreads();
    }
  }
}

void upper_only(Context& context, linalg::MatrixView values) {
  const int size = blas_int(values.rows);
  upper_only_kernel<<<(size*size+255)/256,256,0,context.stream()>>>(
      values.data,size,blas_int(values.ld));
  check_cuda(cudaGetLastError(), "transport upper projection kernel");
}

struct FramePair {
  ExteriorTransform alpha;
  std::unique_ptr<ExteriorTransform> beta_storage;
  const ExteriorTransform* beta;

  FramePair(Context& context,const StructureAction& action)
      : alpha(context,action.alpha_transform()),
        beta_storage(&action.alpha_transform()==&action.beta_transform() ? nullptr :
            std::make_unique<ExteriorTransform>(context,action.beta_transform())),
        beta(beta_storage ? beta_storage.get() : &alpha) {}

  void update(linalg::ConstMatrixView frame) {
    alpha.update_upper(frame);
    if (beta_storage) beta_storage->update_upper(frame);
  }
};

}  // namespace

StructureSourceFactors::StructureSourceFactors(Context& context,
    const StructureAction& action)
    : context_(&context),action_(&action),structures_(action.size()),
      alpha_(action.n_alpha_strings()),beta_(action.n_beta_strings()),
      groups_(blas_int(action.n_product_groups())),
      source_groups_(context,groups_),source_offsets_device_(context,beta_+1),
      alpha_offsets_device_(context,alpha_+1) {
  if (structures_ <= 0 || alpha_ <= 0 || beta_ <= 0 || groups_ <= 0)
    throw std::invalid_argument("invalid selected source dimensions");
  const auto view=action.orthogonal_view();
  if (view.products.size()!=static_cast<std::size_t>(groups_))
    throw std::invalid_argument("transport source topology mismatch");
  std::vector<std::vector<int>> by_beta(beta_);
  std::vector<int> alpha_offsets(alpha_+1);
  for (int group=0;group<groups_;++group) {
    if (group && view.products[group]<view.products[group-1])
      throw std::invalid_argument("transport source products must be sorted");
    const int alpha=static_cast<int>(view.products[group]/beta_);
    const int beta=static_cast<int>(view.products[group]%beta_);
    ++alpha_offsets[alpha+1];
    by_beta[beta].push_back(group);
  }
  for (int alpha=0;alpha<alpha_;++alpha)
    alpha_offsets[alpha+1]+=alpha_offsets[alpha];
  std::vector<int> source_offsets(beta_+1);
  std::vector<int> groups;
  groups.reserve(groups_);
  for (int beta=0;beta<beta_;++beta) {
    source_offsets[beta]=static_cast<int>(groups.size());
    if (!by_beta[beta].empty()) {
      source_columns_.push_back(beta);
      source_group_counts_.push_back(static_cast<int>(by_beta[beta].size()));
    }
    groups.insert(groups.end(),by_beta[beta].begin(),by_beta[beta].end());
  }
  source_offsets[beta_]=static_cast<int>(groups.size());
  source_columns_device_=Buffer<int>(context,source_columns_.size());
  context.copy_to_device(source_offsets_device_.data(),source_offsets.data(),
      source_offsets.size()*sizeof(int));
  context.copy_to_device(alpha_offsets_device_.data(),alpha_offsets.data(),
      alpha_offsets.size()*sizeof(int));
  context.copy_to_device(source_columns_device_.data(),source_columns_.data(),
      source_columns_.size()*sizeof(int));
  if (!groups.empty())
    context.copy_to_device(source_groups_.data(),groups.data(),groups.size()*sizeof(int));
}

SelectedCrossOverlap::SelectedCrossOverlap(Context& context,
                                           StructureAction& action,
                                           int active_orbitals,
                                           ExactMetricInverse exact_inverse,
                                           MetricPreconditioner preconditioner)
    : context_(&context), action_(&action), structures_(action.size()),
      alpha_(action.n_alpha_strings()), beta_(action.n_beta_strings()),
      active_(active_orbitals), sources_(context,action),
      diagonal_(context, action.size()),
      diagonal_h_scratch_(context,action.size(),1),
      exact_inverse_(std::move(exact_inverse)),
      preconditioner_(std::move(preconditioner)) {
  if (active_<=0)
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

int StructureSourceFactors::source_extent() const {
  const auto bytes=16ULL*sizeof(double)*
      (static_cast<std::size_t>(alpha_)+beta_);
  const auto available=context_->available_device_bytes();
  // Keep every factor panel narrower than the beta dimension when possible,
  // even when the device budget could hold all touched source columns.
  const auto width=std::min<std::size_t>({
      static_cast<std::size_t>(product_panel_width()),
      source_columns_.size(),static_cast<std::size_t>(std::max(1,beta_-1)),
      available/bytes});
  if (!width)
    throw std::length_error("transport cannot hold one factor-seed column");
  return static_cast<int>(width);
}

void StructureSourceFactors::load(int begin,int count,
    linalg::ConstMatrixView coefficients,linalg::MatrixView alpha,
    linalg::MatrixView beta) const {
  if (coefficients.rows!=structures_ || coefficients.cols!=1 ||
      coefficients.ld<structures_ || !coefficients.data)
    throw std::invalid_argument("invalid selected source coefficients");
  if (begin<0 || count<=0 || count>std::max(1,beta_-1) ||
      begin+count>static_cast<int>(source_columns_.size()) ||
      alpha.rows!=alpha_ || alpha.cols!=count ||
      beta.rows!=beta_ || beta.cols!=count ||
      alpha.ld<alpha_ || beta.ld<beta_ || !alpha.data || !beta.data)
    throw std::invalid_argument("invalid transport source panel");
  zero(*context_,alpha);zero(*context_,beta);
  source_alpha<<<source_grid(count),256,0,context_->stream()>>>(
        source_groups_.data(),source_offsets_device_.data(),
        source_columns_device_.data(),begin,count,action_->product_indices_data(),
        action_->term_offsets_data(),action_->term_structures_data(),
        action_->term_weights_data(),coefficients.data,alpha_,alpha,beta);
  check_cuda(cudaGetLastError(),"transport source panel");
}

void StructureSourceFactors::contract(linalg::ConstMatrixView alpha,
    linalg::ConstMatrixView beta,linalg::MatrixView image) const {
  if (alpha.rows!=alpha_ || alpha.cols<=0 ||
      alpha.cols>std::max(1,beta_-1) || beta.rows!=beta_ ||
      beta.cols!=alpha.cols || alpha.ld<alpha_ || beta.ld<beta_ ||
      image.rows!=structures_ || image.cols!=1 || image.ld<structures_ ||
      !alpha.data || !beta.data || !image.data)
    throw std::invalid_argument("invalid transport contraction panels");
  source_contract<<<source_grid(groups_),256,0,context_->stream()>>>(
      action_->product_indices_data(),action_->term_offsets_data(),
      action_->term_structures_data(),action_->term_weights_data(),groups_,alpha_,
      alpha,beta,image.data);
  check_cuda(cudaGetLastError(),"transport source contraction");
}

void StructureSourceFactors::contract_raw_sources(int begin,
    linalg::ConstMatrixView alpha,linalg::MatrixView image) const {
  const int count=static_cast<int>(alpha.cols);
  if (begin<0 || count<=0 || count>std::max(1,beta_-1) ||
      begin+count>source_count() || alpha.rows!=alpha_ || alpha.ld<alpha_ ||
      image.rows!=structures_ || image.cols!=1 || image.ld<structures_ ||
      !alpha.data || !image.data)
    throw std::invalid_argument("invalid raw-beta source contraction panel");
  const int max_groups=*std::max_element(source_group_counts_.begin()+begin,
      source_group_counts_.begin()+begin+count);
  source_contract_raw<<<dim3(source_grid(max_groups),std::min(count,65535)),
      256,0,context_->stream()>>>(source_columns_device_.data(),
      source_offsets_device_.data(),source_groups_.data(),begin,count,
      action_->product_indices_data(),action_->term_offsets_data(),
      action_->term_structures_data(),action_->term_weights_data(),alpha_,
      alpha,image.data);
  check_cuda(cudaGetLastError(),"transport raw-beta source contraction");
}

void SelectedCrossOverlap::seeds(const ExteriorTransform& alpha_map,
    const ExteriorTransform& beta_map,linalg::ConstMatrixView coefficients,
    linalg::ConstMatrixView alpha,linalg::ConstMatrixView beta,
    linalg::MatrixView alpha_seed,linalg::MatrixView beta_seed,
    linalg::MatrixView alpha_work,linalg::MatrixView beta_work) const {
  check_structure(coefficients);
  copy(*context_,alpha_work,alpha);copy(*context_,beta_work,beta);
  alpha_map.apply_left(alpha_work,true);
  beta_map.apply_left(beta_work,true);
  const dim3 grid(std::min(65535,alpha_+beta_),
      static_cast<unsigned int>((alpha.cols+31)/32));
  source_seed_rows<<<grid,32,0,context_->stream()>>>(
      action_->product_indices_data(),action_->term_offsets_data(),
      action_->term_structures_data(),action_->term_weights_data(),
      coefficients.data,sources_.alpha_offsets_device_.data(),
      sources_.source_offsets_device_.data(),sources_.source_groups_.data(),
      alpha_,beta_,
      alpha_work,beta_work,alpha_seed.data,beta_seed.data);
  check_cuda(cudaGetLastError(),"transport factor seeds");
  alpha_map.apply_left(alpha_seed);
  beta_map.apply_left(beta_seed);
}

void SelectedCrossOverlap::seeds_three(const ExteriorTransform& alpha_map,
    const ExteriorTransform& beta_map,linalg::ConstMatrixView coefficients,
    linalg::ConstMatrixView alpha0,linalg::ConstMatrixView beta0,
    linalg::ConstMatrixView alpha1,linalg::ConstMatrixView beta1,
    linalg::ConstMatrixView alpha2,linalg::ConstMatrixView beta2,
    linalg::MatrixView alpha_seed0,linalg::MatrixView beta_seed0,
    linalg::MatrixView alpha_seed1,linalg::MatrixView beta_seed1,
    linalg::MatrixView alpha_seed2,linalg::MatrixView beta_seed2,
    linalg::MatrixView alpha_work0,linalg::MatrixView beta_work0,
    linalg::MatrixView alpha_work1,linalg::MatrixView beta_work1,
    linalg::MatrixView alpha_work2,linalg::MatrixView beta_work2) const {
  check_structure(coefficients);
  const int count=static_cast<int>(alpha0.cols);
  copy(*context_,alpha_work0,alpha0);copy(*context_,alpha_work1,alpha1);
  copy(*context_,alpha_work2,alpha2);
  copy(*context_,beta_work0,beta0);copy(*context_,beta_work1,beta1);
  copy(*context_,beta_work2,beta2);
  alpha_map.apply_left(alpha_work0,true);alpha_map.apply_left(alpha_work1,true);
  alpha_map.apply_left(alpha_work2,true);
  beta_map.apply_left(beta_work0,true);beta_map.apply_left(beta_work1,true);
  beta_map.apply_left(beta_work2,true);
  const dim3 grid(std::min(65535,alpha_+beta_),
      static_cast<unsigned int>((count+31)/32));
  source_seed_rows_three<<<grid,32,0,context_->stream()>>>(
      action_->product_indices_data(),action_->term_offsets_data(),
      action_->term_structures_data(),action_->term_weights_data(),
      coefficients.data,sources_.alpha_offsets_device_.data(),
      sources_.source_offsets_device_.data(),sources_.source_groups_.data(),
      alpha_,beta_,alpha_work0,beta_work0,alpha_work1,beta_work1,
      alpha_work2,beta_work2,
      alpha_seed0,beta_seed0,alpha_seed1,beta_seed1,alpha_seed2,beta_seed2);
  check_cuda(cudaGetLastError(),"transport joint factor seeds");
  alpha_map.apply_left(alpha_seed0);alpha_map.apply_left(alpha_seed1);
  alpha_map.apply_left(alpha_seed2);
  beta_map.apply_left(beta_seed0);beta_map.apply_left(beta_seed1);
  beta_map.apply_left(beta_seed2);
}

void SelectedCrossOverlap::accumulate_moment(const ExteriorTransform& transform,
    linalg::ConstMatrixView seed,linalg::ConstMatrixView image,double scale,
    linalg::MatrixView moment) const {
  const int orbitals=transform.n_orbitals();
  const auto work=seed.rows*seed.cols;
  const auto chunks=static_cast<unsigned int>(std::min<std::int64_t>(
      65535,(work+moment_chunk-1)/moment_chunk));
  const dim3 diagonal_grid(std::min(65535,orbitals),chunks);
  moment_diagonal<<<diagonal_grid,moment_threads,0,context_->stream()>>>(
      transform.occupations_data(),transform.dimension(),orbitals,
      seed,image,scale,moment);
  if (transform.shear_count())
    moment_shears<<<dim3(std::min<std::size_t>(65535,transform.shear_count()),chunks),
        moment_threads,0,context_->stream()>>>(
        transform.shear_schedule_data(),transform.shear_edges_data(),
        static_cast<int>(transform.shear_count()),orbitals,seed,image,scale,moment);
  check_cuda(cudaGetLastError(),"transport generator moments");
}

void SelectedCrossOverlap::apply(linalg::ConstMatrixView left_frame,
    linalg::ConstMatrixView right_frame,
    linalg::ConstMatrixView coefficients, linalg::MatrixView image,
    OrbitalFrameIdentity right_identity) const {
  check_frame(left_frame);check_frame(right_frame);
  check_structure(coefficients);check_structure(image);
  zero(*context_,image);
  action_->update_frame(right_frame,right_identity);
  FramePair left(*context_,*action_);
  left.update(left_frame);
  const int extent=source_extent();
  Matrix<double> a(*context_,alpha_,extent),b(*context_,beta_,extent);
  for (int begin=0;begin<source_count();begin+=extent) {
    const int count=std::min(extent,source_count()-begin);
    auto av=a.view().columns(0,count),bv=b.view().columns(0,count);
    load_sources(begin,count,coefficients,av,bv);
    action_->alpha_transform().apply_left(av);
    action_->beta_transform().apply_left(bv);
    left.alpha.apply_left(av,true);left.beta->apply_left(bv,true);
    contract_source(av,bv,image);
  }
}

void SelectedCrossOverlap::apply_direction(linalg::ConstMatrixView left_frame,
    linalg::ConstMatrixView right_frame,
    linalg::ConstMatrixView left_direction,
    linalg::ConstMatrixView right_direction,
    linalg::ConstMatrixView coefficients, linalg::MatrixView image,
    OrbitalFrameIdentity right_identity) const {
  check_frame(left_frame);check_frame(right_frame);
  check_frame(left_direction);check_frame(right_direction);
  check_structure(coefficients);check_structure(image);
  zero(*context_,image);
  action_->update_frame(right_frame,right_identity);
  FramePair left(*context_,*action_);
  left.update(left_frame);
  auto ra=action_->alpha_transform().direction(right_frame,right_direction);
  auto rb=action_->beta_transform().direction(right_frame,right_direction);
  auto la=left.alpha.direction(left_frame,left_direction);
  auto lb=left.beta->direction(left_frame,left_direction);
  const int extent=source_extent();
  Matrix<double> a(*context_,alpha_,extent),da(*context_,alpha_,extent);
  Matrix<double> b(*context_,beta_,extent),db(*context_,beta_,extent);
  for (int begin=0;begin<source_count();begin+=extent) {
    const int count=std::min(extent,source_count()-begin);
    auto av=a.view().columns(0,count),dav=da.view().columns(0,count);
    auto bv=b.view().columns(0,count),dbv=db.view().columns(0,count);
    load_sources(begin,count,coefficients,av,bv);
    zero(*context_,dav);zero(*context_,dbv);
    action_->alpha_transform().apply_directional_left(ra,av,dav);
    action_->beta_transform().apply_directional_left(rb,bv,dbv);
    left.alpha.apply_directional_left(la,av,dav,true);
    left.beta->apply_directional_left(lb,bv,dbv,true);
    contract_source(dav,bv,image);
    contract_source(av,dbv,image);
  }
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

void SelectedCrossOverlap::adjoint(linalg::ConstMatrixView left_frame,
    linalg::ConstMatrixView right_frame,
    linalg::ConstMatrixView left_coefficients,
    linalg::ConstMatrixView right_coefficients,
    linalg::MatrixView left_gradient,
    linalg::MatrixView right_gradient,
    OrbitalFrameIdentity right_identity) const {
  check_frame(left_frame);check_frame(right_frame);
  check_structure(left_coefficients);check_structure(right_coefficients);
  check_gradient(left_gradient);check_gradient(right_gradient);
  Matrix<double> ml(*context_,active_,active_),mr(*context_,active_,active_);
  zero(*context_,ml.view());zero(*context_,mr.view());
  action_->update_frame(right_frame,right_identity);
  FramePair left(*context_,*action_);
  left.update(left_frame);
  const int extent=source_extent();
  Matrix<double> u(*context_,alpha_,extent),v(*context_,beta_,extent);
  Matrix<double> za(*context_,alpha_,extent),zb(*context_,beta_,extent);
  Matrix<double> wa(*context_,alpha_,extent),wb(*context_,beta_,extent);
  for (int begin=0;begin<source_count();begin+=extent) {
    const int count=std::min(extent,source_count()-begin);
    auto uv=u.view().columns(0,count),vv=v.view().columns(0,count);
    auto zav=za.view().columns(0,count),zbv=zb.view().columns(0,count);
    auto wav=wa.view().columns(0,count),wbv=wb.view().columns(0,count);
    load_sources(begin,count,right_coefficients,uv,vv);
    action_->alpha_transform().apply_left(uv);
    action_->beta_transform().apply_left(vv);
    seeds(left.alpha,*left.beta,left_coefficients,uv,vv,zav,zbv,wav,wbv);
    accumulate_moment(left.alpha,uv,zav,1.0,ml.view());
    accumulate_moment(*left.beta,vv,zbv,1.0,ml.view());
    accumulate_moment(action_->alpha_transform(),zav,uv,1.0,mr.view());
    accumulate_moment(action_->beta_transform(),zbv,vv,1.0,mr.view());
  }
  pullback(left_frame,ml.view(),left_gradient);
  pullback(right_frame,mr.view(),right_gradient);
}

void SelectedCrossOverlap::transport_image(linalg::ConstMatrixView current,
    linalg::ConstMatrixView origin,linalg::ConstMatrixView direction,
    linalg::ConstMatrixView base,linalg::ConstMatrixView reference,
    linalg::MatrixView image,OrbitalFrameIdentity current_identity) const {
  check_frame(current);check_frame(origin);check_frame(direction);
  check_structure(base);check_structure(reference);check_structure(image);
  zero(*context_,image);
  action_->update_frame(current,current_identity);
  FramePair oa(*context_,*action_);
  oa.update(origin);
  auto da=oa.alpha.direction(origin,direction);
  auto db=oa.beta->direction(origin,direction);
  const int extent=source_extent();
  Matrix<double> u(*context_,alpha_,extent),du(*context_,alpha_,extent),
      ub(*context_,alpha_,extent);
  Matrix<double> v(*context_,beta_,extent),dv(*context_,beta_,extent),
      vb(*context_,beta_,extent);
  for (int begin=0;begin<source_count();begin+=extent) {
    const int count=std::min(extent,source_count()-begin);
    auto uv=u.view().columns(0,count),duv=du.view().columns(0,count);
    auto ubv=ub.view().columns(0,count);
    auto vv=v.view().columns(0,count),dvv=dv.view().columns(0,count);
    auto vbv=vb.view().columns(0,count);
    load_sources(begin,count,reference,uv,vv);
    zero(*context_,duv);zero(*context_,dvv);
    oa.alpha.apply_directional_left(da,uv,duv);
    oa.beta->apply_directional_left(db,vv,dvv);
    load_sources(begin,count,base,ubv,vbv);
    oa.alpha.apply_left(ubv);
    axpy(*context_,ubv,1.0,duv);
    action_->alpha_transform().apply_left(ubv,true);
    action_->beta_transform().apply_left(vv,true);
    contract_source(ubv,vv,image);
    action_->alpha_transform().apply_left(uv,true);
    action_->beta_transform().apply_left(dvv,true);
    contract_source(uv,dvv,image);
  }
}

void SelectedCrossOverlap::transport_gradient(linalg::ConstMatrixView current,
    linalg::ConstMatrixView origin,linalg::ConstMatrixView direction,
    linalg::ConstMatrixView lambda,linalg::ConstMatrixView base,
    linalg::ConstMatrixView reference,linalg::ConstMatrixView transported,
    linalg::MatrixView current_gradient,linalg::MatrixView origin_gradient,
    linalg::MatrixView image,OrbitalFrameIdentity current_identity) const {
  const nvtx3::scoped_range range{"transport.contracted_pullback"};
  check_frame(current);check_frame(origin);check_frame(direction);
  check_structure(lambda);check_structure(base);check_structure(reference);
  check_structure(transported);check_structure(image);
  check_gradient(current_gradient);check_gradient(origin_gradient);
  zero(*context_,image);
  Matrix<double> mc(*context_,active_,active_),mo(*context_,active_,active_);
  zero(*context_,mc.view());zero(*context_,mo.view());
  action_->update_frame(current,current_identity);
  FramePair oa(*context_,*action_);
  oa.update(origin);
  auto da=oa.alpha.direction(origin,direction);
  auto db=oa.beta->direction(origin,direction);
  const int extent=source_extent();
  Matrix<double> u(*context_,alpha_,extent),du(*context_,alpha_,extent),
      ub(*context_,alpha_,extent),uc(*context_,alpha_,extent);
  Matrix<double> first_a(*context_,alpha_,extent),second_a(*context_,alpha_,extent),
      metric_a(*context_,alpha_,extent),projected(*context_,alpha_,extent);
  Matrix<double> v(*context_,beta_,extent),dv(*context_,beta_,extent),
      vb(*context_,beta_,extent),vc(*context_,beta_,extent);
  Matrix<double> first_b(*context_,beta_,extent),second_b(*context_,beta_,extent),
      metric_b(*context_,beta_,extent);
  Matrix<double> wa0(*context_,alpha_,extent),wa1(*context_,alpha_,extent),
      wa2(*context_,alpha_,extent);
  Matrix<double> wb0(*context_,beta_,extent),wb1(*context_,beta_,extent),
      wb2(*context_,beta_,extent);
  for (int begin=0;begin<source_count();begin+=extent) {
    const int count=std::min(extent,source_count()-begin);
    auto uv=u.view().columns(0,count),duv=du.view().columns(0,count);
    auto ubv=ub.view().columns(0,count),ucv=uc.view().columns(0,count);
    auto fav=first_a.view().columns(0,count),sav=second_a.view().columns(0,count);
    auto mav=metric_a.view().columns(0,count),pv=projected.view().columns(0,count);
    auto vv=v.view().columns(0,count),dvv=dv.view().columns(0,count);
    auto vbv=vb.view().columns(0,count),vcv=vc.view().columns(0,count);
    auto fbv=first_b.view().columns(0,count),sbv=second_b.view().columns(0,count);
    auto mbv=metric_b.view().columns(0,count);
    auto wa0v=wa0.view().columns(0,count),wa1v=wa1.view().columns(0,count),
         wa2v=wa2.view().columns(0,count);
    auto wb0v=wb0.view().columns(0,count),wb1v=wb1.view().columns(0,count),
         wb2v=wb2.view().columns(0,count);
    load_sources(begin,count,reference,uv,vv);
    zero(*context_,duv);zero(*context_,dvv);
    oa.alpha.apply_directional_left(da,uv,duv);
    oa.beta->apply_directional_left(db,vv,dvv);
    load_sources(begin,count,base,ubv,vbv);
    oa.alpha.apply_left(ubv);
    axpy(*context_,ubv,1.0,duv);
    load_sources(begin,count,transported,ucv,vcv);
    action_->alpha_transform().apply_left(ucv);
    action_->beta_transform().apply_left(vcv);
    const auto& ca=action_->alpha_transform();
    const auto& cb=action_->beta_transform();
    seeds_three(ca,cb,lambda,ubv,vv,uv,dvv,ucv,vcv,
        fav,fbv,sav,sbv,mav,mbv,wa0v,wb0v,wa1v,wb1v,wa2v,wb2v);
    accumulate_moment(ca,ubv,fav,1.0,mc.view());
    accumulate_moment(ca,uv,sav,1.0,mc.view());
    accumulate_moment(ca,ucv,mav,-1.0,mc.view());
    accumulate_moment(ca,mav,ucv,-1.0,mc.view());
    accumulate_moment(cb,vv,fbv,1.0,mc.view());
    accumulate_moment(cb,dvv,sbv,1.0,mc.view());
    accumulate_moment(cb,vcv,mbv,-1.0,mc.view());
    accumulate_moment(cb,mbv,vcv,-1.0,mc.view());
    accumulate_moment(oa.alpha,fav,uv,1.0,mo.view());
    accumulate_moment(*oa.beta,sbv,vv,1.0,mo.view());
    copy(*context_,pv,fav);
    oa.alpha.apply_left(pv,true);
    // The raw beta identity restricts this back-projection to its sources.
    sources_.contract_raw_sources(begin,pv,image);
  }
  pullback(current,mc.view(),current_gradient);
  pullback(origin,mo.view(),origin_gradient);
}

void SelectedCrossOverlap::apply_metric(linalg::ConstMatrixView frame,
    linalg::ConstMatrixView coefficients, linalg::MatrixView image,
    OrbitalFrameIdentity frame_identity) const {
  check_frame(frame);
  check_structure(coefficients);
  check_structure(image);
  action_->update_frame(frame,frame_identity);
  action_->apply_overlap(coefficients,image);
}

void SelectedCrossOverlap::solve_metric(linalg::ConstMatrixView frame,
    linalg::ConstMatrixView rhs, linalg::MatrixView solution,
    linalg::ConstMatrixView initial, linalg::MetricSolveStats* stats,
    const SelectedMetric* metric,OrbitalFrameIdentity frame_identity) const {
  const nvtx3::scoped_range range{"transport.metric_inverse"};
  check_frame(frame);
  check_structure(rhs);
  check_structure(solution);
  if (!initial.empty()) check_structure(initial);
  if (rhs.data==solution.data)
    throw std::invalid_argument("transport metric RHS and solution alias");
  linalg::MetricSolveStats local;
  if (!stats) stats=&local;
  *stats={};
  const double rhs_norm=norm(*context_,rhs);
  if (!std::isfinite(rhs_norm))
    throw std::invalid_argument("non-finite metric RHS");
  if (metric) {
    if (metric->dimension()!=structures_)
      throw std::invalid_argument("point selected metric dimension mismatch");
    if (rhs_norm==0.0) {
      zero(*context_,solution);
      return;
    }
    // The caller owns this immutable metric at exactly `frame`. Neither a
    // shared-action frame update nor a CI-model refresh is needed here.
    metric->solve(rhs,solution);
    Matrix<double> residual(*context_,structures_,1);
    metric->apply(solution,residual.view());
    ++stats->actions;
    ++stats->certifications;
    axpy(*context_,residual.view(),-1.0,rhs);
    const double residual_norm=norm(*context_,residual.view());
    stats->relative_residual=residual_norm/rhs_norm;
    const double target=std::sqrt(std::numeric_limits<double>::epsilon())*rhs_norm;
    if (!std::isfinite(residual_norm) || residual_norm>target)
      throw std::runtime_error(
          "exact selected metric inverse failed true-residual certification");
    return;
  }
  Matrix<double> r(*context_,structures_,1),z(*context_,structures_,1);
  Matrix<double> p(*context_,structures_,1),ap(*context_,structures_,1);
  Matrix<double> image(*context_,structures_,1);
  action_->update_frame(frame,frame_identity);
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
      origin_gradient_(context,structures_,1),
      origin_coefficients_(context,structures_,1) {
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
  copy(context,origin_coefficients_.view(),selected_state);
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
    linalg::ConstMatrixView base_coefficients,
    std::shared_ptr<const SelectedMetric> selected_metric,
    OrbitalFrameIdentity trial_identity) const {
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
      trial_identity ? trial_identity : OrbitalFrameIdentity::fresh(),
      Matrix<double>(*context_,active_,active_),
      Matrix<double>(*context_,structures_,1),
      Matrix<double>(*context_,structures_,1),
      std::move(selected_metric)};
  copy(*context_,point.frame.view(),trial_frame);
  copy(*context_,point.origin_direction.view(),origin_frame_direction);
  copy(*context_,point.base_coefficients.view(),base_coefficients);
  Matrix<double> rhs(*context_,structures_,1);
  overlap_->transport_image(point.frame.view(),origin_frame_.view(),
      point.origin_direction.view(),point.base_coefficients.view(),
      origin_coefficients_.view(),rhs.view(),point.frame_identity);
  overlap_->solve_metric(point.frame.view(),rhs.view(),
                         point.coefficients.view(),
                         point.base_coefficients.view(),nullptr,
                         point.selected_metric.get(),point.frame_identity);
  return point;
}

SingleStateCiTransportModel SingleStateCiTransport::evaluate_model(
    const SingleStateCiTransportPoint& point, double physical_energy,
    linalg::ConstMatrixView physical_coefficient_gradient) const {
  const nvtx3::scoped_range range{"transport.keyframe_model"};
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
  copy(*context_,rhs.view(),physical_coefficient_gradient);
  axpy(*context_,rhs.view(),-1.0,origin_gradient_.view());
  overlap_->solve_metric(point.frame.view(),rhs.view(),lambda.view(),{},nullptr,
                         point.selected_metric.get(),point.frame_identity);

  overlap_->transport_gradient(point.frame.view(),origin_frame_.view(),
      point.origin_direction.view(),lambda.view(),point.base_coefficients.view(),
      origin_coefficients_.view(),point.coefficients.view(),
      model.current_upper.view(),model.origin_upper.view(),
      model.structure_covector.view(),point.frame_identity);
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
