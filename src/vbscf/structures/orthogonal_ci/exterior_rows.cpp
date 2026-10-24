#include "vbscf/structures/orthogonal_ci/exterior_rows.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <memory>
#include <stdexcept>

#include "core/timing.hpp"
#include "core/workspace.hpp"
#include "parallel/openmp.hpp"

namespace xmvb::vb {

ExteriorRowPlan::ExteriorRowPlan(int n_orbitals, int n_electrons,
    std::span<const std::uint64_t> masks)
    : n_(n_orbitals), electrons_(n_electrons),
      binomial_((n_+1)*(n_+1),0), column_ranks_(masks.size()) {
  for (int n=0;n<=n_;++n) {
    binomial_[n*(n_+1)]=1;
    for (int k=1;k<=n;++k)
      binomial_[n*(n_+1)+k]=choose(n-1,k-1)+choose(n-1,k);
  }
  for (std::size_t column=0;column<masks.size();++column) {
    auto mask=masks[column];
    std::uint64_t rank=0;
    for (int k=1;mask;++k) {
      const int orbital=std::countr_zero(mask);
      rank+=choose(orbital,k);
      mask&=mask-1;
    }
    column_ranks_[column]=static_cast<int>(rank);
  }
}

std::uint64_t ExteriorRowPlan::choose(int n,int k) const noexcept {
  return n<0 || k<0 || k>n ? 0:binomial_[n*(n_+1)+k];
}

ExteriorRowPlan::Rows ExteriorRowPlan::apply(const Eigen::MatrixXd& upper,
    std::span<const std::uint64_t> masks,std::span<const int> rows) const {
  profiling::ScopedPhase phase(profiling::Phase::CiTransformRight);
  const std::size_t dimension=masks.size();
  // A worker owns contiguous rows. Writing a column-major result here would
  // make neighboring workers repeatedly invalidate the same cache lines.
  Rows result(rows.size(),dimension);
  if (rows.empty()) return result;
  // Scratch belongs to the caller's numerical partition, not to remaining
  // cache-admission capacity (which may already be leased). Bound its two
  // vectors per worker by one requested panel, or the irreducible vector pair.
  const auto worker_bytes=2*sizeof(double)*dimension;
  const auto scratch_bytes=std::min(workspace::limit(),
      std::max(worker_bytes,sizeof(double)*dimension*rows.size()));
  if (worker_bytes>scratch_bytes)
    throw std::runtime_error("exterior row scratch exceeds the shared workspace");
  const int workers=bounded_openmp_thread_count(cooperative_openmp_thread_count(),
      std::min(rows.size(),std::max<std::size_t>(1,
          scratch_bytes/worker_bytes)));
  auto scratch=std::make_unique_for_overwrite<double[]>(2*dimension*workers);
  workspace::Temporary scratch_temporary(2*dimension*workers*sizeof(double));
  cooperative_openmp_for(workers,workers,[&](std::ptrdiff_t part) {
    std::array<int,63> occupied{},indices{};
    for (std::size_t row=rows.size()*part/workers;row<rows.size()*(part+1)/workers;++row) {
      auto mask=masks[rows[row]];
      for (int k=0;mask;++k) {
        occupied[k]=std::countr_zero(mask);
        mask&=mask-1;
      }
      double* previous=scratch.get()+2*dimension*part;
      double* current=previous+dimension;
      mcscf::detail::exterior_row(topology(),upper.data(),upper.outerStride(),
          occupied.data(),indices.data(),previous,current,result.row(row).data());
    }
  });
  return result;
}

std::size_t ExteriorRowPlan::dynamic_bytes() const noexcept {
  return binomial_.capacity()*sizeof(std::uint64_t)+column_ranks_.capacity()*sizeof(int);
}

} // namespace xmvb::vb
