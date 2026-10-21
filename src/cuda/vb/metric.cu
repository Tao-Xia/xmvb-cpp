#include "cuda/vb/metric.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <cusolverDn.h>

#include "cuda/linalg/blas.hpp"
#include "cuda/vb/action.hpp"
#include "vbscf/structures/assembly/action.hpp"

namespace xmvb::cuda {
namespace {

constexpr int kThreads = 256;

unsigned int grid_size(std::int64_t count) {
  if (count<=0 || (count-1)/kThreads+1>
                      std::numeric_limits<unsigned int>::max())
    throw std::overflow_error("CUDA spin metric kernel grid is too large");
  return static_cast<unsigned int>((count-1)/kThreads+1);
}

void check_solver(cusolverStatus_t status,const char* operation) {
  if (status!=CUSOLVER_STATUS_SUCCESS)
    throw std::runtime_error(std::string(operation)+" failed (cuSOLVER status "+
                             std::to_string(static_cast<int>(status))+")");
}

class SolverHandle {
 public:
  explicit SolverHandle(cudaStream_t stream) {
    check_solver(cusolverDnCreate(&handle_),"cusolverDnCreate");
    try {
      check_solver(cusolverDnSetStream(handle_,stream),"cusolverDnSetStream");
    } catch (...) {
      cusolverDnDestroy(handle_);
      throw;
    }
  }
  ~SolverHandle() { cusolverDnDestroy(handle_); }
  SolverHandle(const SolverHandle&)=delete;
  SolverHandle& operator=(const SolverHandle&)=delete;
  operator cusolverDnHandle_t() const noexcept { return handle_; }
 private:
  cusolverDnHandle_t handle_=nullptr;
};

// Each output cell has one owner and intersects two product-sorted sparse W
// columns. This fixes reduction order without atomics or a global dense Gram.
__global__ void assemble_gram(const int* indices,const std::int64_t* offsets,
    const std::int64_t* products,const double* coefficients,double* factors,
    std::size_t first_index,std::size_t first_factor,int dimension,
    std::int64_t cells) {
  const std::int64_t cell=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
  if (cell>=cells) return;
  const std::int64_t block=cell/(static_cast<std::int64_t>(dimension)*dimension);
  const int row=static_cast<int>(cell%dimension);
  const int column=static_cast<int>((cell/dimension)%dimension);
  if (row<column) return;
  const int left=indices[first_index+block*dimension+row];
  const int right=indices[first_index+block*dimension+column];
  std::int64_t a=offsets[left],b=offsets[right];
  const std::int64_t a_end=offsets[left+1],b_end=offsets[right+1];
  double value=0.0;
  while (a<a_end && b<b_end) {
    const auto ap=products[a],bp=products[b];
    if (ap==bp) {
      value+=coefficients[a]*coefficients[b];
      ++a; ++b;
    } else if (ap<bp) ++a;
    else ++b;
  }
  factors[first_factor+cell]=value;
}

__global__ void extract_gram_diagonal(const int* indices,const double* factors,
    double* diagonal,std::size_t first_index,std::size_t first_factor,
    int dimension,std::int64_t rows) {
  const std::int64_t position=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
  if (position>=rows) return;
  const auto block=position/dimension;
  const int local=static_cast<int>(position%dimension);
  diagonal[indices[first_index+position]]=
      factors[first_factor+block*dimension*dimension+local*(dimension+1)];
}

__global__ void validate_lower_factors(const double* factors,int* info,
    std::size_t first_factor,std::size_t first_pointer,int dimension,
    std::int64_t cells) {
  const std::int64_t cell=blockIdx.x*static_cast<std::int64_t>(blockDim.x)+threadIdx.x;
  if (cell>=cells) return;
  const auto block=cell/(static_cast<std::int64_t>(dimension)*dimension);
  const int row=static_cast<int>(cell%dimension);
  const int column=static_cast<int>((cell/dimension)%dimension);
  if (row<column) return;
  const double value=factors[first_factor+cell];
  if (!isfinite(value) || (row==column && !(value>0.0)))
    atomicCAS(info+first_pointer+block,0,-100);
}

__global__ void gather_blocks(linalg::ConstMatrixView source,
                              const int* indices, double* packed, int count) {
  const int position=blockIdx.x*blockDim.x+threadIdx.x;
  if (position<count) packed[position]=source.data[indices[position]];
}

__global__ void scatter_blocks(const double* packed, const int* indices,
                               linalg::MatrixView destination, int count) {
  const int position=blockIdx.x*blockDim.x+threadIdx.x;
  if (position<count) destination.data[indices[position]]=packed[position];
}

__global__ void scale_diagonal_model(linalg::ConstMatrixView source,
                                     linalg::ConstMatrixView model,
                                     const double* gram_diagonal,
                                     linalg::MatrixView destination, int count) {
  const int position=blockIdx.x*blockDim.x+threadIdx.x;
  if (position<count)
    destination.data[position]=source.data[position]*
        sqrt(gram_diagonal[position]/model.data[position]);
}

void check_panel(linalg::ConstMatrixView input,linalg::MatrixView output,int rows) {
  if (input.rows!=rows || output.rows!=rows || input.cols<=0 ||
      output.cols!=input.cols || input.ld<input.rows || output.ld<output.rows ||
      !input.data || !output.data)
    throw std::invalid_argument("CUDA spin metric panel dimensions");
}

}  // namespace

void SpinMetric::initialize_layout(
    Context& context,const std::vector<std::vector<int>>& blocks,
    std::size_t factor_byte_limit) {
  // Reorder entire occupation blocks, never rows within a block, so equal
  // block dimensions share the same batched TRSM path.
  std::map<int,std::vector<std::size_t>> by_dimension;
  std::vector<unsigned char> seen(dimension_,0);
  std::size_t factor_bytes=0;
  for (std::size_t b=0;b<blocks.size();++b) {
    const auto d=blocks[b].size();
    if (d==0 || d>static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("CUDA spin metric has an invalid factor block");
    if (factor_bytes>factor_byte_limit ||
        d>(factor_byte_limit-factor_bytes)/sizeof(double)/d)
      throw std::length_error("CUDA spin metric factors exceed byte budget");
    factor_bytes+=d*d*sizeof(double);
    for (const int index:blocks[b]) {
      if (index<0 || index>=dimension_ || seen[index]++)
        throw std::invalid_argument("CUDA spin metric block indices overlap");
    }
    by_dimension[static_cast<int>(d)].push_back(b);
  }
  if (std::find(seen.begin(),seen.end(),0)!=seen.end())
    throw std::invalid_argument("CUDA spin metric blocks do not cover the space");

  std::vector<int> host_indices;
  host_indices.reserve(dimension_);
  std::vector<const double*> host_factor_pointers;
  std::vector<double*> host_scratch_pointers;
  std::size_t factor_elements=0;
  for (const auto& [d,batch]:by_dimension) {
    if (batch.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("CUDA spin metric batch count exceeds int range");
    groups_.push_back({d,static_cast<int>(batch.size()),
                       host_factor_pointers.size(),host_indices.size(),factor_elements});
    for (const auto b:batch) {
      host_indices.insert(host_indices.end(),blocks[b].begin(),blocks[b].end());
      factor_elements+=static_cast<std::size_t>(d)*d;
      host_factor_pointers.push_back(nullptr);  // filled after device allocation
      host_scratch_pointers.push_back(nullptr);
    }
  }
  if (host_indices.size()!=static_cast<std::size_t>(dimension_))
    throw std::logic_error("CUDA spin metric packed block length disagrees");

  indices_=Buffer<int>(context,host_indices.size());
  factors_=Buffer<double>(context,factor_elements);
  gram_diagonal_=Buffer<double>(context,dimension_);
  factor_pointers_=Buffer<const double*>(context,host_factor_pointers.size());
  scratch_pointers_=Buffer<double*>(context,host_scratch_pointers.size());
  context.copy_to_device(indices_.data(),host_indices.data(),
                         host_indices.size()*sizeof(int));
  // Device pointer arrays are immutable because both buffers are resident for
  // this metric's lifetime. Compute addresses from offsets without GPU copies.
  std::size_t pointer=0, factor_offset=0, scratch_offset=0;
  for (const auto& group:groups_) for (int b=0;b<group.count;++b) {
    host_factor_pointers[pointer]=factors_.data()+factor_offset;
    host_scratch_pointers[pointer]=scratch_.data()+scratch_offset;
    factor_offset+=static_cast<std::size_t>(group.dimension)*group.dimension;
    scratch_offset+=group.dimension;
    ++pointer;
  }
  context.copy_to_device(factor_pointers_.data(),host_factor_pointers.data(),
                         host_factor_pointers.size()*sizeof(const double*));
  context.copy_to_device(scratch_pointers_.data(),host_scratch_pointers.data(),
                         host_scratch_pointers.size()*sizeof(double*));
}

SpinMetric::SpinMetric(Context& context,const vb::StructureAction& source,
    const std::vector<std::vector<int>>& alpha,
    const std::vector<std::vector<int>>& beta,
    std::size_t factor_byte_limit)
    : context_(&context),dimension_(source.n_structures()),
      scratch_(context,dimension_,1) {
  if (dimension_<=0 || !source.storage().orthogonal_direct_ci)
    throw std::invalid_argument("CUDA spin factors require an orthogonal direct-CI action");
  const auto blocks=vb::SpinMetric::occupation_groups(source,alpha,beta);
  initialize_layout(context,blocks,factor_byte_limit);

  // Transpose only the retained sparse incidence representation. The CPU does
  // not evaluate any overlap product or factor in this construction path.
  const auto view=source.orthogonal_view();
  if (view.offsets.size()!=view.products.size()+1 ||
      view.offsets.back()!=view.terms.size() ||
      view.products.size()>static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      view.terms.size()>static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
    throw std::invalid_argument("CUDA spin incidence topology is invalid");
  std::vector<std::int64_t> host_offsets(static_cast<std::size_t>(dimension_)+1,0);
  for (const auto& term:view.terms) {
    if (term.structure<0 || term.structure>=dimension_ ||
        !std::isfinite(term.coefficient))
      throw std::invalid_argument("CUDA spin incidence term is invalid");
    ++host_offsets[static_cast<std::size_t>(term.structure)+1];
  }
  for (int i=0;i<dimension_;++i)
    host_offsets[i+1]+=host_offsets[i];
  std::vector<std::int64_t> positions=host_offsets;
  std::vector<std::int64_t> host_products(view.terms.size());
  std::vector<double> host_weights(view.terms.size());
  for (std::size_t product=0;product<view.products.size();++product) {
    const auto first=view.offsets[product],last=view.offsets[product+1];
    if (first>last || last>view.terms.size())
      throw std::invalid_argument("CUDA spin incidence row offsets are invalid");
    for (auto term_index=first;term_index<last;++term_index) {
      const auto& term=view.terms[term_index];
      if (term.group!=static_cast<int>(product))
        throw std::invalid_argument("CUDA spin incidence group index is invalid");
      const auto position=positions[term.structure]++;
      host_products[position]=static_cast<std::int64_t>(product);
      host_weights[position]=term.coefficient;
    }
  }
  for (int i=0;i<dimension_;++i)
    if (positions[i]!=host_offsets[i+1])
      throw std::logic_error("CUDA spin incidence CSC packing disagrees");

  Buffer<std::int64_t> column_offsets(context,host_offsets.size());
  Buffer<std::int64_t> product_rows(context,host_products.size());
  Buffer<double> weights(context,host_weights.size());
  context.copy_to_device(column_offsets.data(),host_offsets.data(),
                         host_offsets.size()*sizeof(std::int64_t));
  context.copy_to_device(product_rows.data(),host_products.data(),
                         host_products.size()*sizeof(std::int64_t));
  context.copy_to_device(weights.data(),host_weights.data(),
                         host_weights.size()*sizeof(double));

  for (const auto& group:groups_) {
    const auto cells=static_cast<std::int64_t>(group.count)*
                     group.dimension*group.dimension;
    assemble_gram<<<grid_size(cells),kThreads,0,context.stream()>>>(
        indices_.data(),column_offsets.data(),product_rows.data(),
        weights.data(),factors_.data(),group.first_index,group.first_factor,
        group.dimension,cells);
    check_cuda(cudaGetLastError(),"CUDA sparse spin Gram assembly");
    const auto rows=static_cast<std::int64_t>(group.count)*group.dimension;
    extract_gram_diagonal<<<grid_size(rows),kThreads,0,context.stream()>>>(
        indices_.data(),factors_.data(),gram_diagonal_.data(),
        group.first_index,group.first_factor,group.dimension,rows);
    check_cuda(cudaGetLastError(),"CUDA spin Gram diagonal extraction");
  }

  std::vector<double*> host_mutable_pointers;
  host_mutable_pointers.reserve(factor_pointers_.size());
  for (const auto& group:groups_) for (int block=0;block<group.count;++block)
    host_mutable_pointers.push_back(
        factors_.data()+group.first_factor+
        static_cast<std::size_t>(block)*group.dimension*group.dimension);
  Buffer<double*> mutable_pointers(context,host_mutable_pointers.size());
  Buffer<int> info(context,host_mutable_pointers.size());
  context.copy_to_device(mutable_pointers.data(),host_mutable_pointers.data(),
                         host_mutable_pointers.size()*sizeof(double*));
  SolverHandle solver(context.stream());
  for (const auto& group:groups_) {
    check_solver(cusolverDnDpotrfBatched(solver,CUBLAS_FILL_MODE_LOWER,
        group.dimension,mutable_pointers.data()+group.first_pointer,
        group.dimension,info.data()+group.first_pointer,group.count),
        "cusolverDnDpotrfBatched spin metric");
    const auto cells=static_cast<std::int64_t>(group.count)*
                     group.dimension*group.dimension;
    validate_lower_factors<<<grid_size(cells),kThreads,0,context.stream()>>>(
        factors_.data(),info.data(),group.first_factor,group.first_pointer,
        group.dimension,cells);
    check_cuda(cudaGetLastError(),"CUDA spin Cholesky validation");
  }
  std::vector<int> host_info(host_mutable_pointers.size());
  context.copy_to_host(host_info.data(),info.data(),host_info.size()*sizeof(int));
  for (std::size_t block=0;block<host_info.size();++block)
    if (host_info[block]!=0)
      throw std::runtime_error("CUDA spin Gram block "+std::to_string(block)+
          " is not positive definite (cuSOLVER info "+
          std::to_string(host_info[block])+")");
}

std::size_t SpinMetric::resident_bytes() const noexcept {
  return indices_.size()*sizeof(int)+factors_.size()*sizeof(double)+
         gram_diagonal_.size()*sizeof(double)+
         static_cast<std::size_t>(dimension_)*sizeof(double)+
         factor_pointers_.size()*sizeof(const double*)+
         scratch_pointers_.size()*sizeof(double*);
}

void SpinMetric::solve(linalg::ConstMatrixView rhs,linalg::MatrixView output,
                       Part part) const {
  check_panel(rhs,output,dimension_);
  if (part!=Part::Lower && part!=Part::Upper && part!=Part::Gram)
    throw std::invalid_argument("CUDA spin metric solve part is invalid");
  const double one=1.0;
  for (std::int64_t column=0;column<rhs.cols;++column) {
    gather_blocks<<<grid_size(dimension_),kThreads,0,context_->stream()>>>(
        rhs.column(column),indices_.data(),scratch_.data(),dimension_);
    check_cuda(cudaGetLastError(),"CUDA spin metric gather");
    for (const auto& group:groups_) {
      const auto* factors=factor_pointers_.data()+group.first_pointer;
      auto* values=scratch_pointers_.data()+group.first_pointer;
      const auto triangular=[&](cublasOperation_t operation) {
        check_cublas(cublasDtrsmBatched(context_->blas(),CUBLAS_SIDE_LEFT,
            CUBLAS_FILL_MODE_LOWER,operation,CUBLAS_DIAG_NON_UNIT,
            group.dimension,1,&one,factors,group.dimension,values,
            group.dimension,group.count),"cublasDtrsmBatched spin metric");
      };
      if (part!=Part::Upper) triangular(CUBLAS_OP_N);
      if (part!=Part::Lower) triangular(CUBLAS_OP_T);
    }
    scatter_blocks<<<grid_size(dimension_),kThreads,0,context_->stream()>>>(
        scratch_.data(),indices_.data(),output.column(column),dimension_);
    check_cuda(cudaGetLastError(),"CUDA spin metric scatter");
  }
}

void SpinMetric::solve_diagonal_model(linalg::ConstMatrixView rhs,
    linalg::ConstMatrixView model,linalg::MatrixView output) const {
  check_panel(rhs,output,dimension_);
  if (model.rows!=dimension_ || model.cols!=1 || model.ld<model.rows ||
      !model.data)
    throw std::invalid_argument("CUDA spin model diagonal dimensions");
  for (std::int64_t column=0;column<rhs.cols;++column) {
    scale_diagonal_model<<<grid_size(dimension_),kThreads,0,context_->stream()>>>(
        rhs.column(column),model,gram_diagonal_.data(),output.column(column),
        dimension_);
    check_cuda(cudaGetLastError(),"CUDA spin model input scaling");
  }
  solve(output,output,Part::Gram);
  for (std::int64_t column=0;column<rhs.cols;++column) {
    scale_diagonal_model<<<grid_size(dimension_),kThreads,0,context_->stream()>>>(
        output.column(column),model,gram_diagonal_.data(),output.column(column),
        dimension_);
    check_cuda(cudaGetLastError(),"CUDA spin model output scaling");
  }
}

FullMetric::FullMetric(Context& context,const SpinMetric& spin,
                       const StructureAction& action)
    : spin_(&spin),action_(&action),
      structure_(context,spin.dimension(),1),
      carrier_(context,action.n_alpha_strings(),action.n_beta_strings()),
      image_(context,action.n_alpha_strings(),action.n_beta_strings()) {
  if (spin.dimension()!=action.n_structures() ||
      action.n_alpha_strings()<=0 || action.n_beta_strings()<=0)
    throw std::invalid_argument("CUDA FULL metric dimensions disagree");
}

void FullMetric::validate(linalg::ConstMatrixView input,
                          linalg::MatrixView output) const {
  check_panel(input,output,spin_->dimension());
}

void FullMetric::to_ci_coordinates(linalg::ConstMatrixView structure,
                                   linalg::MatrixView coordinates) const {
  validate(structure,coordinates);
  for (std::int64_t column=0;column<structure.cols;++column) {
    action_->expand_spin(structure.column(column),carrier_.view());
    action_->apply_orbital_transform(carrier_.view());
    action_->contract_spin(carrier_.view(),structure_.view());
    spin_->solve(structure_.view(),coordinates.column(column),SpinMetric::Part::Lower);
  }
}

void FullMetric::from_ci_coordinates(linalg::ConstMatrixView coordinates,
                                     linalg::MatrixView structure) const {
  validate(coordinates,structure);
  for (std::int64_t column=0;column<coordinates.cols;++column) {
    spin_->solve(coordinates.column(column),structure_.view(),SpinMetric::Part::Upper);
    action_->expand_spin(structure_.view(),carrier_.view());
    action_->apply_inverse_orbital_transform(carrier_.view());
    action_->contract_spin(carrier_.view(),structure_.view());
    spin_->solve(structure_.view(),structure.column(column),SpinMetric::Part::Gram);
  }
}

void FullMetric::ci_covector_to_structure(linalg::ConstMatrixView covector,
                                          linalg::MatrixView structure) const {
  validate(covector,structure);
  for (std::int64_t column=0;column<covector.cols;++column) {
    spin_->solve(covector.column(column),structure_.view(),SpinMetric::Part::Upper);
    action_->expand_spin(structure_.view(),carrier_.view());
    action_->apply_adjoint_orbital_transform(carrier_.view());
    action_->contract_spin(carrier_.view(),structure.column(column));
  }
}

void FullMetric::inverse_metric(linalg::ConstMatrixView covector,
                                linalg::MatrixView structure) const {
  validate(covector,structure);
  // Match the factorized CPU FULL inverse: G^-1 W^T T^-1 T^-T W G^-1.
  for (std::int64_t column=0;column<covector.cols;++column) {
    spin_->solve(covector.column(column),structure_.view(),SpinMetric::Part::Gram);
    action_->expand_spin(structure_.view(),carrier_.view());
    action_->apply_inverse_adjoint_orbital_transform(carrier_.view());
    action_->apply_inverse_orbital_transform(carrier_.view());
    action_->contract_spin(carrier_.view(),structure_.view());
    spin_->solve(structure_.view(),structure.column(column),SpinMetric::Part::Gram);
  }
}

void FullMetric::apply_ci_hamiltonian(linalg::ConstMatrixView coordinates,
                                      linalg::MatrixView image) const {
  validate(coordinates,image);
  for (std::int64_t column=0;column<coordinates.cols;++column) {
    spin_->solve(coordinates.column(column),structure_.view(),SpinMetric::Part::Upper);
    action_->expand_spin(structure_.view(),carrier_.view());
    action_->apply_carrier_hamiltonian(carrier_.view(),image_.view());
    action_->contract_spin(image_.view(),structure_.view());
    spin_->solve(structure_.view(),image.column(column),SpinMetric::Part::Lower);
  }
}

void FullMetric::apply_ci_preconditioner(linalg::ConstMatrixView covector,
    linalg::ConstMatrixView energies,double eigenvalue,
    linalg::MatrixView result) const {
  validate(covector,result);
  if (energies.rows!=1 || energies.cols!=covector.cols ||
      energies.ld<energies.rows || !energies.data || !std::isfinite(eigenvalue))
    throw std::invalid_argument("CUDA FULL preconditioner energies dimensions");
  for (std::int64_t column=0;column<covector.cols;++column) {
    spin_->solve(covector.column(column),structure_.view(),SpinMetric::Part::Upper);
    action_->expand_spin(structure_.view(),carrier_.view());
    action_->divide_carrier_diagonal(carrier_.view(),energies.column(column),eigenvalue);
    action_->contract_spin(carrier_.view(),structure_.view());
    spin_->solve(structure_.view(),result.column(column),SpinMetric::Part::Lower);
  }
}

}  // namespace xmvb::cuda
