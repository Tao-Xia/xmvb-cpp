#include "vbscf/structures/assembly/action.hpp"

#include <algorithm>
#include <array>
#include <numeric>

#include <Eigen/SparseCore>

#include "parallel/openmp.hpp"
#include "core/timing.hpp"
#include "mcscf/ci/product_support.hpp"
#include "mcscf/ci/sigma.hpp"
#include "mcscf/ci/workspace.hpp"
#include "vbscf/structures/assembly/column_tiles.hpp"
#include "vbscf/structures/assembly/input_columns.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"

namespace xmvb::vb {
using detail::ColumnPanels;
using detail::ColumnPattern;
using detail::ColumnTile;

int StructureAction::detect_spin_exchange() const {
  int parity = 0;
  for (std::size_t group = 0; group < spin_products_.size(); ++group) {
    const auto begin = spin_term_offsets_[group], end = spin_term_offsets_[group + 1];
    if (begin == end) continue;
    const auto product = spin_products_[group];
    const auto exchanged = (product % n_unique_beta_) * n_unique_alpha_ +
                          product / n_unique_beta_;
    const auto found = std::lower_bound(spin_products_.begin(), spin_products_.end(), exchanged);
    if (found == spin_products_.end() || *found != exchanged) return 0;
    const auto other = static_cast<std::size_t>(found - spin_products_.begin());
    const auto paired = spin_term_offsets_[other];
    if (end - begin != spin_term_offsets_[other + 1] - paired) return 0;
    for (std::size_t offset = 0; offset < end - begin; ++offset) {
      const auto& a = spin_terms_[begin + offset];
      const auto& b = spin_terms_[paired + offset];
      if (a.structure != b.structure) return 0;
      const int sign = a.coefficient == b.coefficient ? 1 :
                      a.coefficient == -b.coefficient ? -1 : 0;
      if (!sign || (parity && parity != sign)) return 0;
      parity = sign;
    }
  }
  return parity;
}

StructureActionResult StructureAction::apply_orthogonal_tiled(
    const Eigen::Ref<const Eigen::MatrixXd>& vectors,
    const ExteriorOrbitalTransform& alpha_transform,
    const ExteriorOrbitalTransform& beta_transform,
    const mcscf::DirectCiSigmaAction& sigma,
    bool overlap_only,
    std::size_t workspace_bytes) const {
  if (source_contraction_)
    // Source order has no product-column caches or separate sigma slab;
    // its factor workspace receives the unsplit action budget.
    return apply_source_tiles(vectors, alpha_transform, beta_transform, sigma,
        overlap_only,workspace_bytes);
  return contract_product_tiles(vectors,alpha_transform,beta_transform,sigma,
      overlap_only,workspace_bytes).images;
}

StructureResponseSweep StructureAction::contract_product_tiles(
    const Eigen::Ref<const Eigen::MatrixXd>& vectors,
    const ExteriorOrbitalTransform& alpha_transform,const ExteriorOrbitalTransform& beta_transform,
    const mcscf::DirectCiSigmaAction& sigma,bool overlap_only,std::size_t workspace_bytes,
    bool collect_overlap) const {
  const mcscf::CiWorkspace operation(workspace_bytes);
  workspace_bytes=operation.bytes();
  constexpr bool collect_images=true;
  // Opt-in deltas survive an externally bounded diagnostic. Nested phase
  // totals describe call paths; they are not additive stage wall times.
  auto& phases=profiling::phase_totals();
  constexpr std::array measured{profiling::Phase::CiTransformLeft,
      profiling::Phase::CiTransformRight,profiling::Phase::CiProductGenerate,
      profiling::Phase::CiProductProjection,profiling::Phase::CiProductPattern};
  std::array<unsigned long long,measured.size()> phase_begin{};
  std::size_t fused_hs_panels=0;
  if (phases.enabled) for (std::size_t i=0;i<measured.size();++i)
    phase_begin[i]=phases.nanoseconds[static_cast<std::size_t>(measured[i])].load(std::memory_order_relaxed);
  StructureResponseSweep sweep{{Eigen::MatrixXd::Zero(collect_images ? n_structures_:0,vectors.cols()),
      Eigen::MatrixXd::Zero(collect_images && collect_overlap ? n_structures_:0,vectors.cols())},
      Eigen::MatrixXd{},
      Eigen::MatrixXd{},
      Eigen::MatrixXd{}};
  auto& result=sweep.images;
  // The budget controls residency and RHS batching, never admission to CI.
  const std::size_t cache_bytes=std::max<std::size_t>(1,workspace_bytes/4);
  operation.sigma_budget(cache_bytes);
  const std::size_t pairs = static_cast<std::size_t>(alpha_transform.n_orbitals()) *
      (alpha_transform.n_orbitals() + 1) / 2;
  const std::size_t sigma_vector_bytes = sizeof(double) *
      static_cast<std::size_t>(n_unique_alpha_) * (2 * pairs + 4);
  // Keep a small carrier's RHS panels resident across the sigma halo sweeps.
  // A larger carrier evicts columns; RHS batching only needs the sigma
  // workspace, not full-product residency. This never gates representation.
  const std::size_t resident_rhs = cache_bytes / sizeof(double) /
      static_cast<std::size_t>(n_unique_alpha_) / n_unique_beta_;
  const int batch_width = static_cast<int>(std::min<std::size_t>(vectors.cols(),
      std::max<std::size_t>(1,cache_bytes/sigma_vector_bytes)));
  const std::size_t row_bytes = sizeof(double) *
      (static_cast<std::size_t>(n_unique_alpha_) * batch_width + n_unique_beta_);
  // Each worker gets at least a cache-line span in the right-transform
  // panel. A fixed 32-row panel gave one scalar row per worker at 32 cores.
  const std::size_t parallel_extent = static_cast<std::size_t>(
      cooperative_openmp_thread_count()) * (64 / sizeof(double));
  const int extent = static_cast<int>(std::min<std::size_t>(
      n_unique_beta_, std::max<std::size_t>(1,
          std::min(parallel_extent, cache_bytes / std::max<std::size_t>(1, 4 * row_bytes)))));
  const auto* support=product_support_.get();
  std::vector<std::uint8_t> column_support(n_unique_beta_,support ? 0:1);
  std::vector<int> reachable_columns;
  if (support && !support->full()) {
    reachable_columns=support->columns();
    for (const int beta:reachable_columns) column_support[beta]=1;
  } else {
    std::fill(column_support.begin(),column_support.end(),1);
    reachable_columns.resize(n_unique_beta_);
    std::iota(reachable_columns.begin(),reachable_columns.end(),0);
  }
  const bool ordered_beta=detail::prefer_ordered_beta(spin_products_.size(),
      n_unique_alpha_,n_unique_beta_,beta_transform.shear_pair_count(),reachable_columns.size());
  // Weak references share input/output patterns without an unbounded support
  // table. Their strong lifetime and byte charge belong to the bounded LRUs.
  std::vector<std::weak_ptr<const ColumnPattern>> patterns(n_unique_beta_);
  const auto pattern_at=[&](int beta)->std::shared_ptr<const ColumnPattern> {
    if (auto pattern=patterns[beta].lock()) return pattern;
    profiling::ScopedPhase phase(profiling::Phase::CiProductPattern);
    auto pattern=std::make_shared<ColumnPattern>();
    pattern->offsets.resize(2);
    pattern->offsets[0]=0;
    pattern->full=!support || support->full();
    if (pattern->full) pattern->offsets[1]=n_unique_alpha_;
    else {
      std::vector<int> rows;
      support->append_column_rows(beta,rows);
      pattern->offsets[1]=rows.size();
      pattern->rows=Eigen::Map<const Eigen::VectorXi>(rows.data(),rows.size());
      pattern->index_rows(n_unique_alpha_);
    }
    patterns[beta]=pattern;
    return pattern;
  };
  const auto merge_patterns=[](const std::vector<std::shared_ptr<const ColumnPattern>>& columns) {
    ColumnPattern result;
    result.full=columns.front()->full;
    result.offsets.resize(columns.size()+1); result.offsets[0]=0;
    for (std::size_t b=0;b<columns.size();++b)
      result.offsets[b+1]=result.offsets[b]+columns[b]->offsets[1];
    if (!result.full) {
      result.rows.resize(result.offsets[columns.size()]);
      for (std::size_t b=0;b<columns.size();++b)
        result.rows.segment(result.offsets[b],columns[b]->rows.size())=columns[b]->rows;
    }
    return result;
  };
  for (int vector_begin = 0; vector_begin < vectors.cols(); vector_begin += batch_width) {
    ++result.tiles.expansions;
    const int width = std::min<int>(batch_width, vectors.cols() - vector_begin);
    const bool ordered=ordered_beta && resident_rhs>=static_cast<std::size_t>(width);
    // Fusion changes only consumption order. Require enough existing cache
    // capacity for every exact input/output column, including sparse indices.
    // No cache is enlarged and no physical-space admission depends on this.
    const long double column_bound=sizeof(double)*static_cast<long double>(n_unique_alpha_)*width+
        2*sizeof(int)+(support ? sizeof(int)*static_cast<long double>(n_unique_alpha_)+
        (sizeof(std::uint64_t)+sizeof(int))*((static_cast<std::size_t>(n_unique_alpha_)+63)/64):0);
    const bool fuse_overlap=!ordered && !overlap_only && collect_overlap &&
        column_bound*n_unique_beta_<=cache_bytes;
    // Replace the input/output LRU payloads within the same workspace quarters.
    // Zero-scattering a selected raw support does not assume CAS completeness.
    Eigen::MatrixXd input_image,h_image;
    workspace::Temporary input_temporary,h_temporary;
    if (ordered) {
      profiling::ScopedPhase phase(profiling::Phase::CiProductGenerate);
      input_image=expand_structure_block(vectors.middleCols(vector_begin,width));
      input_temporary.resize(sizeof(double)*input_image.size());
      alpha_transform.apply_left(&input_image);
      for (int v=0;v<width;++v)
        beta_transform.apply_right_block(input_image.middleCols(v*n_unique_beta_,n_unique_beta_));
      if (!overlap_only) h_image=Eigen::MatrixXd::Zero(input_image.rows(),input_image.cols());
      h_temporary.resize(sizeof(double)*h_image.size());
    }
    // Expand once per RHS, not once per beta panel. For a covalent space
    // these groups are exactly (A, complement(A)): only O(U) coefficients.
    Eigen::MatrixXd coefficients = Eigen::MatrixXd::Zero(ordered ? 0:spin_products_.size(), width);
    workspace::Temporary coefficients_temporary(sizeof(double)*coefficients.size());
#pragma omp parallel for schedule(static) if(spin_products_.size() > 32)
    for (int group = 0; group < coefficients.rows(); ++group)
      for (int vector = 0; vector < width; ++vector)
        for (std::size_t term = spin_term_offsets_[group];
             term < spin_term_offsets_[group + 1]; ++term)
          coefficients(group, vector) += spin_terms_[term].coefficient *
              vectors(spin_terms_[term].structure, vector_begin + vector);
    Eigen::MatrixXd h_products = Eigen::MatrixXd::Zero(!ordered ? spin_products_.size():0, width);
    Eigen::MatrixXd m_products = Eigen::MatrixXd::Zero(!ordered && collect_overlap ? spin_products_.size():0, width);
    workspace::Temporary h_products_temporary(sizeof(double)*h_products.size());
    workspace::Temporary m_products_temporary(sizeof(double)*m_products.size());
    const auto project_pattern = [&](const ColumnPattern& pattern, const Eigen::SparseMatrix<double>& rows,
                             Eigen::MatrixXd& image, Eigen::MatrixXd& target,
                             Eigen::MatrixXd* second=nullptr) {
      profiling::ScopedPhase phase(profiling::Phase::CiProductProjection);
      // C^T (T_alpha^T image T_beta): only touched original spin products
      // are sampled. This is exact for any selected structure subspace.
      if (pattern.full) alpha_transform.apply_adjoint_left(&image);
      else alpha_transform.apply_supported_left(&image,
          {pattern.offsets.data(),static_cast<std::size_t>(pattern.offsets.size())},
          {pattern.rows.data(),static_cast<std::size_t>(pattern.rows.size())},true);
      const int workers=coarse_openmp_thread_count(cooperative_openmp_thread_count(),spin_products_.size());
      cooperative_openmp_for(spin_products_.size(),workers,[&](std::ptrdiff_t group) {
        const int alpha = spin_products_[group] / n_unique_beta_;
        const int beta = spin_products_[group] % n_unique_beta_;
        for (int vector = 0; vector < width; ++vector) {
          double sum=0.0;
          double other=0.0;
          for (Eigen::SparseMatrix<double>::InnerIterator entry(rows,beta);entry;++entry) {
            sum+=image(alpha,vector*rows.rows()+entry.row())*entry.value();
            if (second) other+=image(alpha,(width+vector)*rows.rows()+entry.row())*entry.value();
          }
          target(group,vector)+=sum;
          if (second) (*second)(group,vector)+=other;
        }
      });
    };
    const auto contract = [&]() {
      if (!collect_images) return;
      ++result.tiles.structure_projections;
      if (ordered) {
        profiling::ScopedPhase phase(profiling::Phase::CiProductProjection);
        // Sigma has finished reading Y. Its storage becomes the overlap
        // pullback; a third full carrier is unnecessary.
        const auto pullback=[&](Eigen::MatrixXd& image) {
          alpha_transform.apply_adjoint_left(&image);
          for (int v=0;v<width;++v)
            beta_transform.apply_adjoint_right_block(image.middleCols(v*n_unique_beta_,n_unique_beta_));
          return contract_spin_product_block(image);
        };
        if (!overlap_only) {
          ++result.tiles.hamiltonian_projections;
          result.hamiltonian.middleCols(vector_begin,width)=pullback(h_image);
        }
        if (collect_overlap) {
          ++result.tiles.overlap_projections;
          result.overlap.middleCols(vector_begin,width)=pullback(input_image);
        }
        return;
      }
      // Apply P^T only after all panels have contributed to each spin pair.
#pragma omp parallel for schedule(static) if(n_structures_ > 32)
      for (int structure = 0; structure < n_structures_; ++structure) {
        for (int vector = 0; vector < width; ++vector) {
          double h = 0.0, m = 0.0;
          for (std::size_t offset = structure_term_offsets_[structure];
               offset < structure_term_offsets_[structure + 1]; ++offset) {
            const auto& term = spin_terms_[structure_term_indices_[offset]];
            h += term.coefficient * h_products(term.group, vector);
            if (collect_overlap) m += term.coefficient * m_products(term.group, vector);
          }
          result.hamiltonian(structure, vector_begin + vector) = h;
          if (collect_overlap) result.overlap(structure, vector_begin + vector) = m;
        }
      }
    };

    std::vector<bool> overlap_consumed(n_unique_beta_, !collect_overlap);
    std::vector<bool> input_retired(n_unique_beta_,false);
    std::size_t factor_entries=0,factor_slots=0;
    const auto sample_factors=[&](std::span<const int> betas) {
      // Exact zero only: no amplitude threshold and no selected-space closure.
      auto factors=beta_transform.generate_rows(betas);
      factor_entries+=factors.nonZeros();
      factor_slots+=static_cast<std::size_t>(factors.rows())*factors.cols();
      return factors;
    };
    detail::InputColumns input(n_unique_alpha_,width,extent,cache_bytes,column_support,
        [&](std::span<const int> betas) -> std::vector<ColumnTile> {
      profiling::ScopedPhase phase(profiling::Phase::CiProductGenerate);
      ++result.tiles.input_panels;
      // One batch contains only requested missing columns, not their entire
      // enclosing beta panels. Support and numerical maps have one lifetime.
      const int cols=betas.size();
      const int workers=cooperative_openmp_thread_count();
      std::vector<std::shared_ptr<const ColumnPattern>> column_patterns(cols);
      cooperative_openmp_for(cols,workers,[&](std::ptrdiff_t b) {
        column_patterns[b]=pattern_at(betas[b]);
      });
      const auto pattern=merge_patterns(column_patterns);
      const auto rows=sample_factors(betas);
      Eigen::MatrixXd values = Eigen::MatrixXd::Zero(n_unique_alpha_, width * rows.rows());
      workspace::Temporary values_temporary(sizeof(double)*values.size());
      // Each alpha owns its row. CSR maps structures straight to strings;
      // no determinant vector or full raw spin-product grid is formed.
      cooperative_openmp_for(n_unique_alpha_,workers,[&](std::ptrdiff_t alpha) {
        const auto first = mcscf::product_index(alpha,0,n_unique_beta_);
        auto group = std::lower_bound(spin_products_.begin(), spin_products_.end(), first);
        for (; group != spin_products_.end() && *group < first + n_unique_beta_; ++group) {
          const auto id = static_cast<std::size_t>(group - spin_products_.begin());
          for (Eigen::SparseMatrix<double>::InnerIterator entry(rows,*group-first);entry;++entry)
            for (int vector=0;vector<width;++vector)
              values(alpha,vector*rows.rows()+entry.row())+=coefficients(id,vector)*entry.value();
        }
      });
      if (pattern.full) alpha_transform.apply_left(&values);
      else alpha_transform.apply_supported_left(&values,
          {pattern.offsets.data(),static_cast<std::size_t>(pattern.offsets.size())},
          {pattern.rows.data(),static_cast<std::size_t>(pattern.rows.size())});
      std::vector<ColumnTile> tiles(cols);
      cooperative_openmp_for(cols,workers,[&](std::ptrdiff_t b) {
        const int size=column_patterns[b]->offsets[1];
        tiles[b]={column_patterns[b],Eigen::MatrixXd(size,width)};
        for (int v=0;v<width;++v)
          for (int a=0;a<size;++a)
            tiles[b].values(a,v)=values(tiles[b].pattern->full ? a:tiles[b].pattern->rows[a],v*cols+b);
      });
      // Packed Y is now owned by the input tiles. Its dense preparation
      // panel can become T_alpha^T Y in place; no overlap clone is needed.
      const bool need_overlap=std::any_of(betas.begin(),betas.end(),
          [&](int beta) { return !overlap_consumed[beta]; });
      if (need_overlap && !fuse_overlap) {
        ++result.tiles.overlap_projections;
        for (int b=0;b<cols;++b)
          if (overlap_consumed[betas[b]])
            for (int v=0;v<width;++v) values.col(v*cols+b).setZero();
        project_pattern(pattern,rows,values,m_products);
        for (const int beta:betas) overlap_consumed[beta]=true;
      }
      return tiles;
    },[&](int beta) {
      const auto pattern=pattern_at(beta);
      return pattern->bytes()+sizeof(double)*static_cast<std::size_t>(pattern->offsets[1])*width;
    });
    if (overlap_only) {
      if (!ordered) input.prepare(reachable_columns);
      result.tiles.input_evictions += input.cache().evictions;
      result.tiles.input_resident_bytes = std::max(result.tiles.input_resident_bytes,
          ordered ? sizeof(double)*input_image.size():input.cache().peak_bytes);
      contract();
      continue;
    }
    // Prepare resident panels while outside the persistent sigma team, so
    // exterior transforms and overlap projection retain their own parallelism.
    // Larger carriers simply generate panels on demand; their space is unchanged.
    if (!ordered && resident_rhs >= static_cast<std::size_t>(width))
      input.prepare(reachable_columns);
    std::vector<int> completed_betas;
    std::vector<ColumnTile> completed_images;
    std::size_t completed_bytes=0;
    const auto flush_output=[&] {
      if (completed_betas.empty()) return;
      ++result.tiles.hamiltonian_projections;
      const int cols=completed_betas.size();
      std::vector<std::shared_ptr<const ColumnPattern>> column_patterns;
      for (const auto& image:completed_images) column_patterns.push_back(image.pattern);
      const auto pattern=merge_patterns(column_patterns);
      std::vector<const ColumnTile*> panels;
      for (const auto& tile:completed_images) panels.push_back(&tile);
      if (fuse_overlap) for (const int beta:completed_betas)
        panels.push_back(&input.column(beta));
      Eigen::MatrixXd image=detail::unpack_columns(n_unique_alpha_,cols,panels);
      workspace::Temporary image_temporary(sizeof(double)*image.size());
      // All compact values have been consumed. Keep only their shared
      // patterns while generating factors and applying the reverse map.
      completed_images.clear();completed_bytes=0;
      const auto rows=sample_factors(completed_betas);
      project_pattern(pattern,rows,image,h_products,fuse_overlap ? &m_products:nullptr);
      if (fuse_overlap) {
        ++fused_hs_panels;
        ++result.tiles.overlap_projections;
        for (const int beta:completed_betas) {
          overlap_consumed[beta]=true;
          if (input_retired[beta]) input.release(beta);
        }
      }
      completed_betas.clear();
    };
    ColumnPanels output(cache_bytes,[&](int beta) -> ColumnTile {
      const auto pattern=pattern_at(beta);
      return {pattern,Eigen::MatrixXd::Zero(pattern->offsets[1],width)};
    },[&](int beta,ColumnTile& tile) {
      const auto bytes=tile.bytes();
      if (!completed_images.empty() &&
          (bytes>cache_bytes || completed_bytes>cache_bytes-bytes)) flush_output();
      completed_betas.push_back(beta);
      completed_bytes+=bytes;
      completed_images.push_back(std::move(tile));
      if (completed_betas.size()==static_cast<std::size_t>(extent)) flush_output();
    },[&](int beta) {
      const auto pattern=pattern_at(beta);
      return pattern->bytes()+sizeof(double)*static_cast<std::size_t>(pattern->offsets[1])*width;
    });
    const mcscf::DirectCiTileReader read =
        [&](const std::vector<int>& betas, const std::vector<int>& alpha_rows,
            Eigen::Ref<Eigen::MatrixXd> columns) {
          int team = 1;
#ifdef _OPENMP
          team = omp_get_num_threads();
#endif
          if (ordered) {
#pragma omp taskloop num_tasks(team) shared(betas, alpha_rows, input_image, columns)
            for (int packed=0;packed<static_cast<int>(betas.size())*width;++packed) {
              const int j=packed/width,v=packed%width;
              for (int row=0;row<static_cast<int>(alpha_rows.size());++row)
                columns(row,packed)=input_image(alpha_rows[row],v*n_unique_beta_+betas[j]);
            }
            return;
          }
          input.read(betas,alpha_rows,columns);
        };
    const mcscf::DirectCiTileWriter add =
        [&](const std::vector<int>& betas, const std::vector<int>& alpha_rows,
            const mcscf::DirectCiColumnAccumulator& accumulate) {
          int team = 1;
#ifdef _OPENMP
          team = omp_get_num_threads();
#endif
          if (ordered) {
            const int chunks=std::min<int>(team,alpha_rows.size());
            const int column_count=betas.size()*width;
#pragma omp taskloop num_tasks(team) shared(betas, alpha_rows, accumulate, h_image)
            for (int packed=0;packed<column_count*chunks;++packed) {
              const int column=packed%column_count,chunk=packed/column_count;
              const int j=column/width,v=column%width;
              accumulate(j,v,alpha_rows.size()*chunk/chunks,
                  alpha_rows.size()*(chunk+1)/chunks,
                  h_image.col(v*n_unique_beta_+betas[j]).data(),nullptr,0);
            }
            return;
          }
          struct BorrowedColumn { ColumnTile* tile; int source; };
          std::vector<BorrowedColumn> borrowed;
          for (int first=0;first<static_cast<int>(betas.size());) {
            borrowed.clear();
            std::size_t bytes=0;
            while (first<static_cast<int>(betas.size())) {
              const int beta=betas[first];
              if (!column_support[beta]) { ++first; continue; }
              const auto pattern=pattern_at(beta);
              const std::size_t payload=pattern->bytes()+
                  sizeof(double)*pattern->offsets[1]*width;
              if (!borrowed.empty() && (payload>cache_bytes || bytes>cache_bytes-payload)) break;
              // Previously borrowed columns are the most recently touched
              // owners. Their sum fits, so this get cannot evict one of them.
              borrowed.push_back({&output.get(beta),first});
              bytes+=payload;
              ++first;
            }
            // Split alpha ranges as well as columns: the alpha-adjoint work
            // is concentrated in target beta columns. Tasks add directly to
            // disjoint destination ranges, without an image slab or atomics.
            const int chunks = std::min<int>(team, alpha_rows.size());
            const int column_count = borrowed.size()*width;
#pragma omp taskloop num_tasks(team) shared(betas, alpha_rows, accumulate, borrowed)
            for (int packed = 0; packed < column_count * chunks; ++packed) {
              // Every task sees all beta columns over its alpha stripe.
              // Column-major task order would concentrate the expensive
              // alpha adjoint (target columns only) on a few workers.
              const int column = packed % column_count;
              const int chunk = packed / column_count;
              const int v=column%width;
              const auto& item=borrowed[column/width];
              const auto& panel=*item.tile;
              const int length=panel.values.rows();
              if (!length) continue;
              double* to=item.tile->values.col(v).data();
              const int begin = alpha_rows.size() * chunk / chunks;
              const int end = alpha_rows.size() * (chunk + 1) / chunks;
              accumulate(item.source,v,begin,end,to,panel.pattern->full ? nullptr:panel.pattern->rows.data(),
                  panel.pattern->full ? 0:length);
            }
          }
        };
    sigma.apply_tiled(width,read,add,support,spin_exchange_parity_,
        [&](std::span<const int> betas) {
      if (ordered) return;
      for (const int beta:betas) {
        input_retired[beta]=true;
        if (!fuse_overlap) input.release(beta);
        output.release(beta);
      }
    },cache_bytes,true);
    output.finish();
    flush_output();
    // Exchange quotienting may never request some input columns. The overlap
    // is still the complete Gram action, independent of the Hamiltonian sweep.
    if (!ordered) {
      std::vector<int> missing;
      for (const int beta:reachable_columns)
        if (!overlap_consumed[beta]) missing.push_back(beta);
      if (!fuse_overlap) input.prepare(missing);
      else for (std::size_t begin=0;begin<missing.size();begin+=extent) {
        const int cols=std::min<std::size_t>(extent,missing.size()-begin);
        std::vector<std::shared_ptr<const ColumnPattern>> patterns;
        std::vector<const ColumnTile*> panels;
        for (int b=0;b<cols;++b) {
          const auto& tile=input.column(missing[begin+b]);
          panels.push_back(&tile);patterns.push_back(tile.pattern);
        }
        const auto pattern=merge_patterns(patterns);
        auto image=detail::unpack_columns(n_unique_alpha_,cols,panels);
        workspace::Temporary temporary(sizeof(double)*image.size());
        const auto rows=sample_factors({missing.data()+begin,static_cast<std::size_t>(cols)});
        project_pattern(pattern,rows,image,m_products);
        ++result.tiles.overlap_projections;
        for (int b=0;b<cols;++b) {
          overlap_consumed[missing[begin+b]]=true;
          input.release(missing[begin+b]);
        }
      }
    }
    if (profiling::phase_totals().enabled)
      std::fprintf(stderr,"XMVB_COLUMN_RELEASE,input=%zu,output=%zu,factor_entries=%zu,factor_slots=%zu\n",
          input.cache().releases,output.releases,factor_entries,factor_slots);
    result.tiles.input_evictions += input.cache().evictions;
    result.tiles.output_evictions += output.evictions;
    result.tiles.input_resident_bytes = std::max(result.tiles.input_resident_bytes,
        ordered ? sizeof(double)*input_image.size():input.cache().peak_bytes);
    result.tiles.output_resident_bytes = std::max(result.tiles.output_resident_bytes,
        ordered ? sizeof(double)*h_image.size():output.peak_bytes);
    contract();
  }
  if (profiling::phase_totals().enabled)
    std::fprintf(stderr,"XMVB_PRODUCT_ACTION,rhs=%ld,overlap=%d,batches=%zu,input=%zu,output=%zu,input_evictions=%zu,output_evictions=%zu,ordered_beta=%d,resident_rhs=%zu,fused_hs_panels=%zu\n",
        static_cast<long>(vectors.cols()),overlap_only,result.tiles.expansions,
        result.tiles.input_panels,result.tiles.hamiltonian_projections,
        result.tiles.input_evictions,result.tiles.output_evictions,ordered_beta,resident_rhs,fused_hs_panels);
  if (phases.enabled) for (std::size_t i=0;i<measured.size();++i)
    std::fprintf(stderr,"XMVB_PRODUCT_PHASE,stage=%s,wall=%.9f\n",
        profiling::PhaseTotals::name(measured[i]),
        (phases.nanoseconds[static_cast<std::size_t>(measured[i])].load(std::memory_order_relaxed)-
          phase_begin[i])*1e-9);
  return sweep;
}

}  // namespace xmvb::vb
