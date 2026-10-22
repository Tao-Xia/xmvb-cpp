#include "cuda/vb/transform.hpp"

#include <cstdint>
#include <algorithm>
#include <limits>
#include <stdexcept>

#include "cuda/linalg/blas.hpp"
#include "mcscf/ci/exterior_row.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_rows.hpp"

namespace xmvb::cuda {
namespace {

using Edge = vb::ExteriorOrbitalTransform::ShearEdge;

__global__ void requested_rows(mcscf::detail::ExteriorRowTopology plan,
    linalg::ConstMatrixView upper,const std::uint64_t* masks,const int* rows,
    int first,int count,double* scratch,linalg::MatrixView output) {
  const int row=blockIdx.x*blockDim.x+threadIdx.x;
  if (row>=count) return;
  int occupied[63],indices[63];
  auto mask=masks[rows[first+row]];
  for (int k=0;mask;++k) {
    occupied[k]=__ffsll(static_cast<long long>(mask))-1;
    mask&=mask-1;
  }
  double* previous=scratch+2LL*plan.dimension*row;
  mcscf::detail::exterior_row(plan,upper.data,upper.ld,occupied,indices,
      previous,previous+plan.dimension,output.data+(first+row)*output.ld);
}

// The ordered elementary-factor coefficients are a small triangular problem,
// not a product-space contraction. Use exactly the CPU elimination ordering.
__global__ void factor_upper(linalg::ConstMatrixView upper,
    linalg::MatrixView unit,double* coefficients) {
  if (blockIdx.x || threadIdx.x) return;
  const int n=upper.rows;
  for (int j=0;j<n;++j) for (int i=0;i<n;++i)
    unit.data[i+unit.ld*j]=upper.data[i+upper.ld*j]/upper.data[j+upper.ld*j];
  int step=0;
  for (int removed=1;removed<n;++removed)
    for (int inserted=removed-1;inserted>=0;--inserted) {
      const double coefficient=unit.data[inserted+unit.ld*removed];
      coefficients[step++]=coefficient;
      for (int i=0;i<n;++i)
        unit.data[i+unit.ld*removed]-=coefficient*unit.data[i+unit.ld*inserted];
    }
}

__global__ void update_scales(linalg::ConstMatrixView upper,
    const std::uint64_t* occupations,double* scales,double* inverse,int dimension) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=dimension) return;
  double value=1.0;
  for (int p=0;p<upper.rows;++p)
    if (occupations[i]&(std::uint64_t{1}<<p)) value*=upper.data[p+upper.ld*p];
  scales[i]=value;
  inverse[i]=1.0/value;
}

__global__ void inverse_scale(const double* scales, double* inverse, int n) {
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) inverse[i] = 1.0/scales[i];
}

unsigned int grid_size(std::int64_t count) {
  constexpr std::int64_t threads = 256;
  const auto blocks = (count + threads - 1) / threads;
  if (blocks > std::numeric_limits<unsigned int>::max())
    throw std::overflow_error("CUDA exterior kernel grid is too large");
  return static_cast<unsigned int>(blocks);
}

// An elementary orbital shear has disjoint determinant source/target pairs.
// Thus each updated element has one owner, while the complete ordered product
// still follows the CPU factorization exactly.
__global__ void shear_left(linalg::MatrixView coefficients,
                           const Edge* edges, int edge_count,
                           const double* coefficient, double orientation, bool adjoint,
                           std::int64_t elements,const std::uint64_t* allowed=nullptr,
                           int support_columns=0,int words=0) {
  const std::int64_t index = blockIdx.x * static_cast<std::int64_t>(blockDim.x)
                             + threadIdx.x;
  if (index >= elements) return;
  const Edge edge = edges[index % edge_count];
  const std::int64_t column = index / edge_count;
  const int source = adjoint ? edge.target : edge.source;
  const int target = adjoint ? edge.source : edge.target;
  if (allowed) {
    const auto* mask=allowed+(column%support_columns)*words;
    if (!(mask[source/64]&(std::uint64_t{1}<<(source%64))) ||
        !(mask[target/64]&(std::uint64_t{1}<<(target%64)))) return;
  }
  coefficients.data[target + coefficients.ld*column] +=
      orientation*(*coefficient)*edge.sign*coefficients.data[source + coefficients.ld*column];
}

__global__ void shear_right(linalg::MatrixView coefficients,
                            const Edge* edges, int edge_count,
                            const double* coefficient, double orientation, bool adjoint,
                            int spin_dimension, std::int64_t elements) {
  const std::int64_t index = blockIdx.x * static_cast<std::int64_t>(blockDim.x)
                             + threadIdx.x;
  if (index >= elements) return;
  // Adjacent lanes update adjacent rows in one column, not strided columns.
  const std::int64_t row = index % coefficients.rows;
  const std::int64_t side = index / coefficients.rows;
  const Edge edge = edges[side % edge_count];
  const std::int64_t rhs = side / edge_count;
  const int source = adjoint ? edge.target : edge.source;
  const int target = adjoint ? edge.source : edge.target;
  const std::int64_t source_column = rhs*spin_dimension + source;
  const std::int64_t target_column = rhs*spin_dimension + target;
  coefficients.data[row + coefficients.ld*target_column] +=
      orientation*(*coefficient)*edge.sign*
      coefficients.data[row + coefficients.ld*source_column];
}

// Each spin column/row tile is independent. Retain it in block-local memory
// for the complete ordered exterior product: sources and targets of one
// elementary shear are disjoint, and the barrier orders consecutive shears.
// This is not a new factorization, truncation, or full product-space cache.
template<bool Right>
__global__ void exterior_tile(linalg::MatrixView panel,const Edge* edges,
    const ExteriorTransform::Shear* schedule,int steps,const double* coefficients,const double* scales,
    int dimension,int lanes,bool adjoint,bool inverse,const std::uint64_t* allowed=nullptr,
    int support_columns=0,int words=0) {
  extern __shared__ double values[];
  const auto row_tiles=(panel.rows+lanes-1)/lanes;
  const auto column=Right ? blockIdx.x/row_tiles*dimension:blockIdx.x;
  const auto row_begin=Right ? blockIdx.x%row_tiles*lanes:0;
  const int work=dimension*lanes;
  for (int k=threadIdx.x;k<work;k+=blockDim.x) {
    const int spin=k/lanes,lane=k%lanes;
    const auto row=Right ? row_begin+lane:spin;
    const auto col=Right ? column+spin:column;
    double v=row<panel.rows ? panel.data[row+panel.ld*col]:0.0;
    if (allowed && !(allowed[(col%support_columns)*words+spin/64]&(std::uint64_t{1}<<(spin%64)))) v=0;
    if (adjoint==inverse) v*=scales[spin];
    values[k]=v;
  }
  __syncthreads();
  for (int step=0;step<steps;++step) {
    const int index=adjoint!=inverse ? steps-1-step:step;
    const auto shear=schedule[index];
    for (int k=threadIdx.x;k<shear.count*lanes;k+=blockDim.x) {
      const auto edge=edges[shear.offset+k/lanes];
      const int lane=k%lanes;
      const int source=adjoint ? edge.target:edge.source;
      const int target=adjoint ? edge.source:edge.target;
      if (allowed) {
        const auto* mask=allowed+(column%support_columns)*words;
        if (!(mask[source/64]&(std::uint64_t{1}<<(source%64))) ||
            !(mask[target/64]&(std::uint64_t{1}<<(target%64)))) continue;
      }
      values[target*lanes+lane]+=(inverse ? -1.0:1.0)*coefficients[index]*edge.sign*
          values[source*lanes+lane];
    }
    __syncthreads();
  }
  for (int k=threadIdx.x;k<work;k+=blockDim.x) {
    const int spin=k/lanes,lane=k%lanes;
    const auto row=Right ? row_begin+lane:spin;
    const auto col=Right ? column+spin:column;
    double v=values[k];
    if (adjoint!=inverse) v*=scales[spin];
    if (row<panel.rows) panel.data[row+panel.ld*col]=v;
  }
}

__global__ void restrict_columns(linalg::MatrixView panel,const std::uint64_t* allowed,
    int columns,int words) {
  const auto i=static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i>=panel.rows*panel.cols) return;
  const auto row=i%panel.rows,column=i/panel.rows;
  if (!(allowed[(column%columns)*words+row/64]&(std::uint64_t{1}<<(row%64))))
    panel.data[row+panel.ld*column]=0.0;
}

}  // namespace

ExteriorTransform::ExteriorTransform(
    Context& context, const vb::ExteriorOrbitalTransform& source)
    : context_(&context), dimension_(source.dimension()),n_orbitals_(source.n_orbitals()),
      electrons_(source.n_electrons()),
      scales_(context, static_cast<std::size_t>(source.dimension())),
      inverse_scales_(context, static_cast<std::size_t>(source.dimension())),
      shear_coefficients_(context,source.shear_count()),
      occupations_(context,source.occupations().size()),
      binomial_(context,source.row_plan().binomial().size()),
      column_ranks_(context,source.row_plan().column_ranks().size()),
      unit_upper_(context,source.n_orbitals(),source.n_orbitals()) {
  if (dimension_ <= 0 || source.determinant_scales().size() != dimension_)
    throw std::invalid_argument("invalid CUDA exterior transform dimensions");
  context.copy_to_device(binomial_.data(),source.row_plan().binomial().data(),
      binomial_.size()*sizeof(std::uint64_t));
  context.copy_to_device(column_ranks_.data(),source.row_plan().column_ranks().data(),
      column_ranks_.size()*sizeof(int));
  int device=0,shared_bytes=0;
  check_cuda(cudaGetDevice(&device),"query exterior device");
  check_cuda(cudaDeviceGetAttribute(&shared_bytes,cudaDevAttrMaxSharedMemoryPerBlock,device),
      "query exterior block capacity");
  shared_values_=shared_bytes/sizeof(double);
  context.copy_to_device(scales_.data(), source.determinant_scales().data(),
                         scales_.size()*sizeof(double));
  inverse_scale<<<grid_size(dimension_),256,0,context.stream()>>>(
      scales_.data(), inverse_scales_.data(), dimension_);
  check_cuda(cudaGetLastError(), "inverse exterior scaling");
  std::vector<Edge> all_edges;
  std::vector<double> coefficients;
  coefficients.reserve(source.shear_count());
  shears_.reserve(source.shear_count());
  for (std::size_t i = 0; i < source.shear_count(); ++i) {
    const auto edges = source.shear_edges(i);
    if (edges.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("CUDA exterior shear is too large");
    shears_.push_back({all_edges.size(), static_cast<int>(edges.size())});
    coefficients.push_back(source.shear_coefficient(i));
    all_edges.insert(all_edges.end(), edges.begin(), edges.end());
  }
  edges_ = Buffer<Edge>(context, all_edges.size());
  schedule_=Buffer<Shear>(context,shears_.size());
  context.copy_to_device(schedule_.data(),shears_.data(),shears_.size()*sizeof(Shear));
  context.copy_to_device(edges_.data(), all_edges.data(),
                         all_edges.size()*sizeof(Edge));
  context.copy_to_device(shear_coefficients_.data(),coefficients.data(),
                         coefficients.size()*sizeof(double));
  context.copy_to_device(occupations_.data(),source.occupations().data(),
                         occupations_.size()*sizeof(std::uint64_t));
}

void ExteriorTransform::update_upper(linalg::ConstMatrixView upper) {
  if (upper.rows!=n_orbitals_ || upper.cols!=n_orbitals_ ||
      upper.ld<upper.rows || !upper.data)
    throw std::invalid_argument("CUDA exterior frame dimensions");
  factor_upper<<<1,1,0,context_->stream()>>>(upper,unit_upper_.view(),
      shear_coefficients_.data());
  update_scales<<<grid_size(dimension_),256,0,context_->stream()>>>(upper,
      occupations_.data(),scales_.data(),inverse_scales_.data(),dimension_);
  check_cuda(cudaGetLastError(),"CUDA exterior frame update");
}

void ExteriorTransform::apply_left(linalg::MatrixView coefficients,
                                    bool adjoint, bool inverse) const {
  if (coefficients.rows != dimension_ || coefficients.cols <= 0 ||
      coefficients.ld < coefficients.rows || coefficients.data == nullptr)
    throw std::invalid_argument("invalid CUDA exterior left panel");
  if (!shears_.empty() && dimension_<=shared_values_) {
    exterior_tile<false><<<grid_size(coefficients.cols*256),256,
        static_cast<std::size_t>(dimension_)*sizeof(double),context_->stream()>>>(
        coefficients,edges_.data(),schedule_.data(),shears_.size(),shear_coefficients_.data(),
        inverse ? inverse_scales_.data():scales_.data(),dimension_,1,adjoint,inverse);
    check_cuda(cudaGetLastError(),"CUDA fused exterior left tile");
    return;
  }
  const auto scale = [&] {
    // cuBLAS permits in-place dgmm when lda == ldc, as here.
    check_cublas(cublasDdgmm(context_->blas(), CUBLAS_SIDE_LEFT,
        blas_int(coefficients.rows), blas_int(coefficients.cols),
        coefficients.data, blas_int(coefficients.ld),
        inverse ? inverse_scales_.data() : scales_.data(), 1,
        coefficients.data, blas_int(coefficients.ld)),
        "cublasDdgmm exterior left scaling");
  };
  if (adjoint == inverse) {
    scale();
  }
  for (std::size_t step = 0; step < shears_.size(); ++step) {
    const auto index=adjoint != inverse ? shears_.size()-1-step : step;
    const Shear& shear = shears_[index];
    if (shear.count == 0) continue;
    const std::int64_t work = static_cast<std::int64_t>(shear.count)*coefficients.cols;
    shear_left<<<grid_size(work),256,0,context_->stream()>>>(
        coefficients, edges_.data()+shear.offset, shear.count,
        shear_coefficients_.data()+index,inverse ? -1.0:1.0,adjoint,work);
    check_cuda(cudaGetLastError(), "CUDA exterior left shear");
  }
  if (adjoint != inverse) {
    scale();
  }
}

void ExteriorTransform::apply_supported_left(linalg::MatrixView panel,
    std::span<const int> offsets,std::span<const int> rows,bool adjoint) const {
  if (offsets.size()<2 || offsets.front()!=0 || offsets.back()!=rows.size() ||
      panel.rows!=dimension_ || panel.ld<panel.rows || panel.cols<=0 ||
      panel.cols%(offsets.size()-1))
    throw std::invalid_argument("invalid CUDA supported exterior pattern");
  const int columns=offsets.size()-1,words=(dimension_+63)/64;
  std::vector<std::uint64_t> bits(static_cast<std::size_t>(columns)*words,0);
  for (int b=0;b<columns;++b) {
    if (offsets[b]<0 || offsets[b]>offsets[b+1] || offsets[b+1]>rows.size())
      throw std::invalid_argument("invalid CUDA supported exterior offsets");
    for (int i=offsets[b];i<offsets[b+1];++i) {
      const int row=rows[i];
      if (row<0 || row>=dimension_) throw std::invalid_argument("supported exterior row is out of range");
      bits[static_cast<std::size_t>(b)*words+row/64]|=std::uint64_t{1}<<(row%64);
    }
  }
  Buffer<std::uint64_t> allowed(*context_,bits.size());
  context_->copy_to_device(allowed.data(),bits.data(),bits.size()*sizeof(std::uint64_t));
  if (dimension_<=shared_values_) {
    exterior_tile<false><<<grid_size(panel.cols*256),256,dimension_*sizeof(double),context_->stream()>>>(
        panel,edges_.data(),schedule_.data(),shears_.size(),shear_coefficients_.data(),scales_.data(),
        dimension_,1,adjoint,false,allowed.data(),columns,words);
  } else {
    restrict_columns<<<grid_size(panel.rows*panel.cols),256,0,context_->stream()>>>(panel,allowed.data(),columns,words);
    const auto scale=[&] {
      check_cublas(cublasDdgmm(context_->blas(),CUBLAS_SIDE_LEFT,blas_int(panel.rows),blas_int(panel.cols),
          panel.data,blas_int(panel.ld),scales_.data(),1,panel.data,blas_int(panel.ld)),"supported exterior scaling");
    };
    if (!adjoint) scale();
    for (std::size_t step=0;step<shears_.size();++step) {
      const auto index=adjoint ? shears_.size()-1-step:step;
      const auto& shear=shears_[index];
      if (shear.count) shear_left<<<grid_size(shear.count*panel.cols),256,0,context_->stream()>>>(panel,
          edges_.data()+shear.offset,shear.count,shear_coefficients_.data()+index,1.0,adjoint,
          shear.count*panel.cols,allowed.data(),columns,words);
    }
    if (adjoint) scale();
  }
  check_cuda(cudaGetLastError(),"supported exterior action");
}

void ExteriorTransform::apply_right(linalg::MatrixView coefficients,
                                     bool adjoint, bool inverse) const {
  if (coefficients.rows <= 0 || coefficients.cols <= 0 ||
      coefficients.cols % dimension_ != 0 ||
      coefficients.ld < coefficients.rows || coefficients.data == nullptr)
    throw std::invalid_argument("invalid CUDA exterior right panel");
  const std::int64_t rhs = coefficients.cols/dimension_;
  if (!shears_.empty() && dimension_<=shared_values_) {
    int lanes=1;
    while (lanes<32 && 2*lanes<=shared_values_/dimension_) lanes*=2;
    const auto tiles=((coefficients.rows+lanes-1)/lanes)*rhs;
    exterior_tile<true><<<grid_size(tiles*256),256,
        static_cast<std::size_t>(dimension_)*lanes*sizeof(double),context_->stream()>>>(
        coefficients,edges_.data(),schedule_.data(),shears_.size(),shear_coefficients_.data(),
        inverse ? inverse_scales_.data():scales_.data(),dimension_,lanes,adjoint,inverse);
    check_cuda(cudaGetLastError(),"CUDA fused exterior right tile");
    return;
  }
  const auto scale = [&] {
    for (std::int64_t block = 0; block < rhs; ++block) {
      const auto panel = coefficients.columns(block*dimension_, dimension_);
      check_cublas(cublasDdgmm(context_->blas(), CUBLAS_SIDE_RIGHT,
          blas_int(panel.rows), blas_int(panel.cols), panel.data,
          blas_int(panel.ld), inverse ? inverse_scales_.data() : scales_.data(), 1, panel.data,
          blas_int(panel.ld)), "cublasDdgmm exterior right scaling");
    }
  };
  if (adjoint == inverse) {
    scale();
  }
  for (std::size_t step = 0; step < shears_.size(); ++step) {
    const auto index=adjoint != inverse ? shears_.size()-1-step : step;
    const Shear& shear = shears_[index];
    if (shear.count == 0) continue;
    const std::int64_t work = static_cast<std::int64_t>(shear.count)*
                              coefficients.rows*rhs;
    shear_right<<<grid_size(work),256,0,context_->stream()>>>(
        coefficients, edges_.data()+shear.offset, shear.count,
        shear_coefficients_.data()+index,inverse ? -1.0:1.0,adjoint,dimension_,work);
    check_cuda(cudaGetLastError(), "CUDA exterior right shear");
  }
  if (adjoint != inverse) {
    scale();
  }
}

void ExteriorTransform::generate_rows(linalg::ConstMatrixView upper,
    std::span<const int> rows,linalg::MatrixView output) const {
  if (upper.rows!=n_orbitals_ || upper.cols!=n_orbitals_ ||
      upper.ld<upper.rows || !upper.data || output.rows!=dimension_ ||
      output.cols!=static_cast<std::int64_t>(rows.size()) || output.ld<output.rows)
    throw std::invalid_argument("requested exterior-row dimensions");
  if (rows.empty()) return;
  if (!output.data || output.data==upper.data ||
      std::any_of(rows.begin(),rows.end(),[&](int row) { return row<0 || row>=dimension_; }))
    throw std::invalid_argument("invalid requested exterior rows");
  auto& c=*context_;
  Buffer<int> selected(c,rows.size());
  c.copy_to_device(selected.data(),rows.data(),rows.size_bytes());
  const auto available=c.device_byte_budget()-c.counters().current_device_bytes;
  const int tile=static_cast<int>(std::min<std::size_t>(rows.size(),
      available/(2ULL*dimension_*sizeof(double))));
  if (!tile) throw std::runtime_error("exterior recursion cannot fit one single-spin row");
  Buffer<double> scratch(c,2ULL*dimension_*tile);
  const mcscf::detail::ExteriorRowTopology plan{
      n_orbitals_,electrons_,dimension_,binomial_.data(),column_ranks_.data()};
  for (int first=0;first<static_cast<int>(rows.size());first+=tile) {
    const int count=std::min(tile,static_cast<int>(rows.size())-first);
    requested_rows<<<grid_size(count),256,0,c.stream()>>>(plan,upper,occupations_.data(),
        selected.data(),first,count,scratch.data(),output);
    check_cuda(cudaGetLastError(),"shared CPU exterior-row recursion");
  }
}

}  // namespace xmvb::cuda
