#include "cuda/vb/chart.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace xmvb::cuda {
namespace {

constexpr int kThreads=128;
int grid(int count) { return (count+kThreads-1)/kThreads; }

int checked_basis(int basis,int orbitals,int reduced,std::size_t stored_count) {
  if (basis<=0 || orbitals<=0 || reduced<0 ||
      stored_count!=static_cast<std::size_t>(orbitals) ||
      static_cast<long long>(basis)*orbitals>std::numeric_limits<int>::max())
    throw std::invalid_argument("CUDA orbital chart dimensions are invalid");
  return basis;
}

void check_view(linalg::ConstMatrixView view,int rows,int cols,
                const char* label) {
  if (rows==0 && cols==1 && view.rows==0 && view.cols==1) return;
  if (!view.data || view.rows!=rows || view.cols!=cols || view.ld<rows)
    throw std::invalid_argument(std::string(label)+" matrix shape mismatch");
}

__global__ void expand_rows(const int* slots,const int* row_block,
    const int* row_local,const int* first_basis,const int* row_count,
    const int* reduced_offset,const int* reduced_size,
    const double* basis_values,const double* reduced,double* raw,
    int mapped_rows,int raw_basis,int raw_ld) {
  const int row=blockIdx.x*blockDim.x+threadIdx.x;
  if (row>=mapped_rows) return;
  const int block=row_block[row];
  const int local=row_local[row];
  const int count=row_count[block];
  const int dimension=reduced_size[block];
  const int offset=first_basis[block];
  const int coordinate=reduced_offset[block];
  double value=0.0;
  for (int j=0;j<dimension;++j)
    value+=basis_values[offset+j*count+local]*reduced[coordinate+j];
  const int slot=slots[row];
  raw[(slot/raw_basis)*raw_ld+slot%raw_basis]=value;
}

__global__ void project_coordinates(const int* slots,const int* first_row,
    const int* row_count,const int* first_basis,const int* coordinate_block,
    const int* coordinate_local,const double* basis_values,
    const double* raw,double* reduced,int coordinates,int raw_basis,int raw_ld) {
  const int coordinate=blockIdx.x*blockDim.x+threadIdx.x;
  if (coordinate>=coordinates) return;
  const int block=coordinate_block[coordinate];
  const int local=coordinate_local[coordinate];
  const int rows=row_count[block];
  const int start=first_row[block];
  const int basis=first_basis[block]+local*rows;
  double value=0.0;
  for (int i=0;i<rows;++i)
    value+=basis_values[basis+i]*
        raw[(slots[start+i]/raw_basis)*raw_ld+slots[start+i]%raw_basis];
  reduced[coordinate]=value;
}

__global__ void add_step(linalg::ConstMatrixView accepted,
    const double* direction,double scale,linalg::MatrixView trial,
    const int* stored_counts,int* info,int basis,int orbitals) {
  const int index=blockIdx.x*blockDim.x+threadIdx.x;
  if (index>=basis*orbitals) return;
  const int row=index%basis,col=index/basis;
  const double value=row>=stored_counts[col]?0.0:
      accepted.data[col*accepted.ld+row]+scale*direction[index];
  trial.data[col*trial.ld+row]=value;
  if (!isfinite(value)) atomicCAS(info,0,index+1);
}

}  // namespace

OrbitalChartMap::OrbitalChartMap(Context& context,int basis,int orbitals,
    int reduced_size,const std::vector<int>& stored_counts,
    const std::vector<OrbitalTangentBlock>& blocks)
    : context_(&context),basis_(checked_basis(basis,orbitals,reduced_size,
                                             stored_counts.size())),orbitals_(orbitals),
      reduced_size_(reduced_size),mapped_rows_(0),
      raw_slots_(context,0),row_block_(context,0),row_local_(context,0),
      block_first_row_(context,blocks.size()),block_row_count_(context,blocks.size()),
      block_first_basis_(context,blocks.size()),
      block_reduced_offset_(context,blocks.size()),
      block_reduced_size_(context,blocks.size()),
      coordinate_block_(context,reduced_size_),coordinate_local_(context,reduced_size_),
      stored_counts_(context,stored_counts.size()),
      retraction_info_(context,1),basis_values_(context,0),
      scratch_(context,basis_,orbitals_) {
  for (const int count:stored_counts)
    if (count<0 || count>basis)
      throw std::invalid_argument("CUDA orbital chart stored support is invalid");
  std::vector<int> slots,row_block,row_local,first_row,row_count,first_basis;
  std::vector<int> coordinate_block(reduced_size,-1),coordinate_local(reduced_size,-1);
  std::vector<int> offsets,sizes;
  std::vector<double> basis_data;
  std::vector<unsigned char> seen(static_cast<std::size_t>(basis)*orbitals,0);
  for (std::size_t b=0;b<blocks.size();++b) {
    const auto& block=blocks[b];
    const int rows=static_cast<int>(block.raw_slots.size());
    if (rows<=0 || block.reduced_size<=0 ||
        block.reduced_offset<0 ||
        block.reduced_offset+block.reduced_size>reduced_size ||
        block.basis.size()!=static_cast<std::size_t>(rows)*block.reduced_size)
      throw std::invalid_argument("CUDA orbital tangent block dimensions are invalid");
    first_row.push_back(static_cast<int>(slots.size()));
    row_count.push_back(rows);
    first_basis.push_back(static_cast<int>(basis_data.size()));
    offsets.push_back(block.reduced_offset);
    sizes.push_back(block.reduced_size);
    const int orbital=block.raw_slots.front()/basis;
    for (int i=0;i<rows;++i) {
      const int slot=block.raw_slots[i];
      if (slot<0 || slot>=basis*orbitals || slot/basis!=orbital ||
          slot%basis>=stored_counts[orbital] || seen[slot]++)
        throw std::invalid_argument("CUDA orbital tangent slots are invalid or overlap");
      slots.push_back(slot);
      row_block.push_back(static_cast<int>(b));
      row_local.push_back(i);
    }
    for (int j=0;j<block.reduced_size;++j) {
      const int coordinate=block.reduced_offset+j;
      if (coordinate_block[coordinate]>=0)
        throw std::invalid_argument("CUDA orbital tangent coordinates overlap");
      coordinate_block[coordinate]=static_cast<int>(b);
      coordinate_local[coordinate]=j;
    }
    for (const double value:block.basis)
      if (!std::isfinite(value))
        throw std::invalid_argument("CUDA orbital tangent basis is not finite");
    basis_data.insert(basis_data.end(),block.basis.begin(),block.basis.end());
  }
  if (std::find(coordinate_block.begin(),coordinate_block.end(),-1)!=
      coordinate_block.end())
    throw std::invalid_argument("CUDA orbital tangent coordinates are incomplete");
  mapped_rows_=static_cast<int>(slots.size());
  raw_slots_=Buffer<int>(context,slots.size());
  row_block_=Buffer<int>(context,row_block.size());
  row_local_=Buffer<int>(context,row_local.size());
  basis_values_=Buffer<double>(context,basis_data.size());
  context.copy_to_device(raw_slots_.data(),slots.data(),slots.size()*sizeof(int));
  context.copy_to_device(row_block_.data(),row_block.data(),row_block.size()*sizeof(int));
  context.copy_to_device(row_local_.data(),row_local.data(),row_local.size()*sizeof(int));
  context.copy_to_device(block_first_row_.data(),first_row.data(),first_row.size()*sizeof(int));
  context.copy_to_device(block_row_count_.data(),row_count.data(),row_count.size()*sizeof(int));
  context.copy_to_device(block_first_basis_.data(),first_basis.data(),first_basis.size()*sizeof(int));
  context.copy_to_device(block_reduced_offset_.data(),offsets.data(),offsets.size()*sizeof(int));
  context.copy_to_device(block_reduced_size_.data(),sizes.data(),sizes.size()*sizeof(int));
  context.copy_to_device(coordinate_block_.data(),coordinate_block.data(),
                         coordinate_block.size()*sizeof(int));
  context.copy_to_device(coordinate_local_.data(),coordinate_local.data(),
                         coordinate_local.size()*sizeof(int));
  context.copy_to_device(stored_counts_.data(),stored_counts.data(),
                         stored_counts.size()*sizeof(int));
  context.copy_to_device(basis_values_.data(),basis_data.data(),
                         basis_data.size()*sizeof(double));
}

void OrbitalChartMap::expand(linalg::ConstMatrixView reduced,
                             linalg::MatrixView raw_direction) const {
  check_view(reduced,reduced_size_,1,"CUDA orbital reduced direction");
  check_view(raw_direction,basis_,orbitals_,"CUDA orbital raw direction");
  for (int j=0;j<orbitals_;++j)
    check_cuda(cudaMemsetAsync(raw_direction.column(j).data,0,
        static_cast<std::size_t>(basis_)*sizeof(double),context_->stream()),
        "CUDA orbital chart direction zero");
  if (mapped_rows_) {
    expand_rows<<<grid(mapped_rows_),kThreads,0,context_->stream()>>>(
        raw_slots_.data(),row_block_.data(),row_local_.data(),
        block_first_basis_.data(),block_row_count_.data(),
        block_reduced_offset_.data(),block_reduced_size_.data(),
        basis_values_.data(),reduced.data,raw_direction.data,mapped_rows_,
        basis_,static_cast<int>(raw_direction.ld));
    check_cuda(cudaGetLastError(),"CUDA orbital chart tangent expansion");
  }
}

void OrbitalChartMap::project(linalg::ConstMatrixView raw_covector,
                              linalg::MatrixView reduced_covector) const {
  check_view(raw_covector,basis_,orbitals_,"CUDA orbital raw covector");
  check_view(reduced_covector,reduced_size_,1,"CUDA orbital reduced covector");
  if (reduced_size_) {
    project_coordinates<<<grid(reduced_size_),kThreads,0,context_->stream()>>>(
        raw_slots_.data(),block_first_row_.data(),block_row_count_.data(),
        block_first_basis_.data(),coordinate_block_.data(),
        coordinate_local_.data(),basis_values_.data(),raw_covector.data,
        reduced_covector.data,reduced_size_,basis_,
        static_cast<int>(raw_covector.ld));
    check_cuda(cudaGetLastError(),"CUDA orbital chart covector projection");
  }
}

void OrbitalChartMap::retract(linalg::ConstMatrixView accepted,
    linalg::ConstMatrixView reduced_step,double scale,
    linalg::MatrixView trial) const {
  check_view(accepted,basis_,orbitals_,"CUDA accepted raw orbitals");
  check_view(reduced_step,reduced_size_,1,"CUDA orbital reduced step");
  check_view(trial,basis_,orbitals_,"CUDA trial raw orbitals");
  if (!std::isfinite(scale))
    throw std::invalid_argument("CUDA orbital chart step scale is not finite");
  expand(reduced_step,scratch_.view());
  check_cuda(cudaMemsetAsync(retraction_info_.data(),0,sizeof(int),
                            context_->stream()),"CUDA orbital retraction info reset");
  add_step<<<grid(basis_*orbitals_),kThreads,0,context_->stream()>>>(
      accepted,scratch_.data(),scale,trial,stored_counts_.data(),
      retraction_info_.data(),
      basis_,orbitals_);
  check_cuda(cudaGetLastError(),"CUDA orbital chart retraction");
  int bad=0;
  context_->copy_to_host(&bad,retraction_info_.data(),sizeof(int));
  if (bad)
    throw std::runtime_error("CUDA orbital retraction produced nonfinite slot "+
                             std::to_string(bad-1));
}

}  // namespace xmvb::cuda
