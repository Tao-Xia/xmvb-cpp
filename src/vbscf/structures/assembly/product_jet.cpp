#include "vbscf/structures/assembly/action.hpp"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <numeric>
#include <stdexcept>

#include "parallel/openmp.hpp"
#include "core/timing.hpp"
#include "mcscf/ci/product_support.hpp"
#include "mcscf/ci/sigma.hpp"
#include "vbscf/structures/assembly/column_tiles.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"

namespace xmvb::vb {
namespace {

/** @brief Embed a fully written reader in one interleaved RHS lane.
 * Only unused lanes need clearing; clearing the selected lane would write
 * the same halo twice. The borrowed reader must outlive this adapter.
 */
mcscf::DirectCiTileReader interleaved_reader(
    const mcscf::DirectCiTileReader& reader,int width,int lane) {
  return [&reader,width,lane](const std::vector<int>& betas,
      const std::vector<int>& rows,Eigen::Ref<Eigen::MatrixXd> values) {
    using View=Eigen::Map<Eigen::MatrixXd,0,Eigen::OuterStride<>>;
    const Eigen::OuterStride<> stride(values.outerStride()*width);
    for (int v=0;v<width;++v)
      if (v!=lane)
        View(values.data()+v*values.outerStride(),rows.size(),betas.size(),stride).setZero();
    View selected(values.data()+lane*values.outerStride(),rows.size(),betas.size(),stride);
    reader(betas,rows,selected);
  };
}

} // namespace

StructureResponseSweep StructureAction::product_jet(const StructureAdjointState& state,
    const Eigen::MatrixXd* response,const Eigen::MatrixXd* delta_upper,
    const mcscf::DirectCiSigmaAction* delta_sigma,const ExteriorOrbitalTransform& alpha,
    const ExteriorOrbitalTransform& beta,const mcscf::DirectCiSigmaAction& sigma,
    bool collect_images,StructureIntegralDirection* diagnostic,
    StructureResponseSweep* ci_sweep,const Eigen::MatrixXd* overlap_image) const {
  const bool joint=ci_sweep!=nullptr;
  if (joint && (!response || !delta_upper || !delta_sigma || !collect_images))
    throw std::invalid_argument("joint product jet requires orbital and CI directions");
  const bool tangent=response || delta_upper || delta_sigma;
  const int width=tangent ? 2:1,n=sigma.n_orbitals(),d=sigma.channel_kernel().rows();
  const int input_width=width+(joint ? 1:0);
  const int states=state.structure_coefficients.cols();
  StructureResponseSweep result{
      {Eigen::MatrixXd::Zero(collect_images ? n_structures_:0,states),
       Eigen::MatrixXd::Zero(collect_images ? n_structures_:0,states)},
      Eigen::MatrixXd::Zero(n,n),Eigen::MatrixXd::Zero(d,d),Eigen::MatrixXd::Zero(n,n)};
  if (collect_images && overlap_image) result.images.overlap=*overlap_image;
  if (joint) *ci_sweep={
      {Eigen::MatrixXd::Zero(n_structures_,states),Eigen::MatrixXd::Zero(n_structures_,states)},
      Eigen::MatrixXd::Zero(n,n),Eigen::MatrixXd::Zero(d,d),Eigen::MatrixXd::Zero(n,n)};
  // Reserve accepted panels across all roots, then divide the remaining
  // allowance among input, output, sigma and reverse-transform scratch.
  // Small explicit panel caps are useful for bounded-residency regressions.
  // They never change physical support or the response being differentiated.
  // Credit this context's own retained panels before partitioning the budget;
  // all other accepted CI/RI data remain subtracted from available().
  const auto total_bytes=workspace::available()+
      (state.product_images ? state.product_images->residency.bytes():0);
  const auto retained_bytes=state.product_images ?
      std::min(state.product_images->budget,total_bytes/4):0;
  const auto work_bytes=(total_bytes-retained_bytes)/4;
  const auto cache_bytes=state.product_images ?
      std::min(state.product_images->budget,work_bytes):work_bytes;
  const auto projection_bytes=std::min(cache_bytes,work_bytes/4);
  const auto factor_bytes=projection_bytes;
  const auto temporary_bytes=work_bytes/2;
  const int extent=std::max<std::size_t>(1,std::min<std::size_t>(n_unique_beta_,
      std::min<std::size_t>(8*cooperative_openmp_thread_count(),
          // Three output components plus compact row indices must fit even
          // for highly unequal spin spaces; the bound is shared by accepted
          // and directional calls so diagnostic capture does not change it.
          std::min<std::size_t>(cache_bytes/(sizeof(double)*((joint ? 6:4)*n_unique_alpha_+2*n_unique_beta_)),
              temporary_bytes/(sizeof(double)*(12ULL*n_unique_alpha_+8ULL*n_unique_beta_))))));
  if (!tangent && state.product_images) {
    state.product_images->extent=extent;
    state.product_images->states.resize(states);
    if (states==1) state.product_images->density=Eigen::MatrixXd::Zero(n,n);
  }
  const int panels=(n_unique_beta_+extent-1)/extent;
  const auto* support=product_support_.get();
  std::vector<int> reachable;
  std::vector<bool> reachable_mask(panels,false);
  if (support && !support->full()) {
    for (const int b:support->columns()) reachable_mask[b/extent]=true;
  } else std::fill(reachable_mask.begin(),reachable_mask.end(),true);
  for (int p=0;p<panels;++p) if (reachable_mask[p]) reachable.push_back(p);
  int sampled_beta_columns=0;
  for (const int panel:reachable)
    sampled_beta_columns+=std::min(extent,n_unique_beta_-panel*extent);
  const bool ordered_beta=detail::prefer_ordered_beta(spin_products_.size(),
      n_unique_alpha_,n_unique_beta_,beta.shear_pair_count(),sampled_beta_columns);
  std::vector<std::weak_ptr<const detail::ColumnPattern>> patterns(panels);
  const auto pattern_at=[&](int panel) -> std::shared_ptr<const detail::ColumnPattern> {
    if (auto p=patterns[panel].lock()) return p;
    // The support and panel geometry do not change with a direction or state.
    // Borrow immutable accepted CSR metadata instead of traversing the support
    // generator again. No new numerical cache or product-space array is made.
    if (tangent && state.product_images && state.product_images->extent==extent &&
        !state.product_images->states.empty() &&
        !state.product_images->states.front().empty()) {
      auto p=state.product_images->states.front()[panel].pattern;
      if (p) { patterns[panel]=p; return p; }
    }
    profiling::ScopedPhase phase(profiling::Phase::CiProductPattern);
    auto p=std::make_shared<detail::ColumnPattern>();
    const int count=std::min(extent,n_unique_beta_-panel*extent);
    p->full=!support || support->full(); p->offsets.resize(count+1); p->offsets[0]=0;
    std::vector<int> rows;
    for (int b=0;b<count;++b) {
      if (!p->full) {
        support->append_column_rows(panel*extent+b,rows);
      }
      p->offsets[b+1]=p->full ? (b+1)*n_unique_alpha_:rows.size();
    }
    p->rows=Eigen::Map<const Eigen::VectorXi>(rows.data(),rows.size());
    patterns[panel]=p; return p;
  };
  ExteriorTransformDirection da,db;
  if (delta_upper) { da=alpha.direction(*delta_upper); db=beta.direction(*delta_upper); }
  const Eigen::MatrixXd generator=delta_upper ? Eigen::MatrixXd(
      alpha.orbital_transform().transpose().triangularView<Eigen::Lower>()
          .solve(delta_upper->transpose()).transpose()):Eigen::MatrixXd{};
  // Differentiated beta rows are shared by generation and left projection.
  detail::ColumnPanels beta_directions(factor_bytes,[&](int panel) {
    const int first=panel*extent,count=std::min(extent,n_unique_beta_-first);
    Eigen::MatrixXd rows=Eigen::MatrixXd::Zero(count,n_unique_beta_),dr=rows;
    rows.middleCols(first,count).setIdentity();
    beta.apply_directional_adjoint_right(db,&rows,&dr);
    auto p=std::make_shared<detail::ColumnPattern>();
    p->full=true; p->offsets.resize(count+1);
    for (int b=0;b<=count;++b) p->offsets[b]=b*n_unique_beta_;
    return detail::pack_columns(dr.transpose(),p);
  },{},[&](int panel) {
    const auto count=std::min(extent,n_unique_beta_-panel*extent);
    return sizeof(double)*static_cast<std::size_t>(count)*n_unique_beta_+
        sizeof(int)*(count+1);
  });
  const bool capture=diagnostic && diagnostic->capture_connection;
  if (capture) diagnostic->connection=StructureConnectionDirection{
      Eigen::MatrixXd::Zero(n_structures_,states),Eigen::MatrixXd::Zero(n_structures_,states),
      Eigen::MatrixXd::Zero(n_structures_,states),Eigen::VectorXd::Zero(states)};

  for (int state_id=0;state_id<states;++state_id) {
    const double weight=state.weights[state_id],energy=state.energies[state_id];
    if (weight==0 && !collect_images) continue;
    const bool cached=tangent && state.product_images && state.product_images->extent==extent &&
        !state.product_images->states[state_id].empty();
    const bool anchored=cached && delta_upper;
    const int sigma_width=cached ? 1:width;
    const std::size_t product_count=static_cast<std::size_t>(n_unique_alpha_)*n_unique_beta_;
    const bool resident_input=!anchored && ordered_beta &&
        product_count<=cache_bytes/(input_width*sizeof(double));
    Eigen::MatrixXd primal_image,direction_image,ci_image;
    std::size_t input_bytes=0;
    if (resident_input) {
      // Apply the ordered exterior jet once, instead of multiplying each beta
      // row panel by all original products. No orbital inverse is introduced.
      if (!cached || delta_upper)
        primal_image=expand_structure_block(state.structure_coefficients.col(state_id));
      if (tangent) direction_image=response && !joint ? expand_structure_block(response->col(state_id)):
          Eigen::MatrixXd::Zero(n_unique_alpha_,n_unique_beta_);
      if (joint) {
        ci_image=expand_structure_block(response->col(state_id));
        alpha.apply_left(&ci_image); beta.apply_right(&ci_image);
      }
      if (delta_upper) {
        alpha.apply_directional_left(da,&primal_image,&direction_image);
        beta.apply_directional_right(db,&primal_image,&direction_image);
      } else {
        if (!cached) { alpha.apply_left(&primal_image); beta.apply_right(&primal_image); }
        if (tangent) { alpha.apply_left(&direction_image); beta.apply_right(&direction_image); }
      }
      input_bytes=sizeof(double)*(primal_image.size()+direction_image.size()+ci_image.size());
      if (cached) primal_image.resize(0,0);
    }
    Eigen::VectorXd coefficients;
    if (!resident_input && !anchored && (!cached || delta_upper))
      coefficients=expand_support(state.structure_coefficients.col(state_id));
    Eigen::VectorXd q;
    if (response && !resident_input) q=expand_support(response->col(state_id));
    const auto accepted_column=[&](int b) -> mcscf::DirectCiColumnView {
      if (!reachable_mask[b/extent]) return {nullptr,nullptr,0};
      const auto& tile=state.product_images->states[state_id][b/extent];
      const int first=tile.pattern->offsets[b%extent],last=tile.pattern->offsets[b%extent+1];
      return {tile.values.col(0).data()+first,
          tile.pattern->full ? nullptr:tile.pattern->rows.data()+first,last-first};
    };
    detail::ColumnPanels input(cache_bytes,[&](int panel) {
      profiling::ScopedPhase phase(profiling::Phase::CiProductGenerate);
      ++result.images.tiles.input_panels;
      const auto pattern=pattern_at(panel);
      const int first=panel*extent,count=std::min(extent,n_unique_beta_-first);
      const auto rows=(!anchored || response) ? beta.row_block(first,count):nullptr;
      Eigen::MatrixXd primal=cached ? Eigen::MatrixXd{}:Eigen::MatrixXd::Zero(n_unique_alpha_,count);
      const bool generator_only=anchored && (!response || joint);
      Eigen::MatrixXd dy;
      if (tangent) {
        if (generator_only) dy.resize(n_unique_alpha_,count);
        else dy=Eigen::MatrixXd::Zero(n_unique_alpha_,count);
      }
      Eigen::MatrixXd dq=joint ? Eigen::MatrixXd::Zero(n_unique_alpha_,count):Eigen::MatrixXd{};
      auto& response_image=joint ? dq:dy;
      Eigen::MatrixXd dr;
      if (delta_upper && !anchored) dr=beta_directions.get(panel).unpack(n_unique_beta_).transpose();
      const int workers=coarse_openmp_thread_count(cooperative_openmp_thread_count(),
          static_cast<std::size_t>(n_unique_alpha_)*count);
      if (coefficients.size() || response) cooperative_openmp_for(n_unique_alpha_,workers,[&](std::ptrdiff_t a) {
        const auto first_product=mcscf::product_index(a,0,n_unique_beta_);
        auto group=std::lower_bound(spin_products_.begin(),spin_products_.end(),first_product);
        for (;group!=spin_products_.end() && *group<first_product+n_unique_beta_;++group) {
          const int id=group-spin_products_.begin(),b=*group-first_product;
          if (coefficients.size()) primal.row(a).noalias()+=coefficients[id]*rows->col(b).transpose();
          if (dr.size()) dy.row(a).noalias()+=coefficients[id]*dr.col(b).transpose();
          if (response) response_image.row(a).noalias()+=q[id]*rows->col(b).transpose();
        }
      });
      if (anchored) {
        if (response) {
          if (pattern->full) alpha.apply_left(&response_image);
          else alpha.apply_supported_left(&response_image,
              {pattern->offsets.data(),static_cast<std::size_t>(pattern->offsets.size())},
              {pattern->rows.data(),static_cast<std::size_t>(pattern->rows.size())});
        }
        if (generator_only) {
          sigma.apply_one_body_generator_columns(first,accepted_column,generator,dy,
              spin_exchange_parity_!=0);
          if (spin_exchange_parity_) dy*=2.0;
        } else {
          Eigen::MatrixXd generator_image(n_unique_alpha_,count);
          sigma.apply_one_body_generator_columns(first,accepted_column,generator,generator_image,
              spin_exchange_parity_!=0);
          dy.noalias()+=(spin_exchange_parity_ ? 2.0:1.0)*generator_image;
        }
      }
      else if (delta_upper) {
        alpha.apply_directional_left(da,&primal,&dy);
        if (joint) {
          if (pattern->full) alpha.apply_left(&dq);
          else alpha.apply_supported_left(&dq,
              {pattern->offsets.data(),static_cast<std::size_t>(pattern->offsets.size())},
              {pattern->rows.data(),static_cast<std::size_t>(pattern->rows.size())});
        }
      }
      else {
        if (!cached) {
          if (pattern->full) alpha.apply_left(&primal);
          else alpha.apply_supported_left(&primal,
              {pattern->offsets.data(),static_cast<std::size_t>(pattern->offsets.size())},
              {pattern->rows.data(),static_cast<std::size_t>(pattern->rows.size())});
        }
        if (tangent) {
          if (pattern->full) alpha.apply_left(&dy);
          else alpha.apply_supported_left(&dy,
              {pattern->offsets.data(),static_cast<std::size_t>(pattern->offsets.size())},
              {pattern->rows.data(),static_cast<std::size_t>(pattern->rows.size())});
        }
      }
      if (cached) {
        if (!joint) return detail::pack_columns(std::move(dy),pattern);
        return detail::pack_columns({&dy,&dq},pattern);
      }
      if (!tangent) return detail::pack_columns(std::move(primal),pattern);
      if (joint) return detail::pack_columns({&primal,&dy,&dq},pattern);
      return detail::pack_columns({&primal,&dy},pattern);
    },{},[&](int panel) {
      const auto pattern=pattern_at(panel);
      return pattern->bytes()+sizeof(double)*
          static_cast<std::size_t>(pattern->offsets[pattern->columns()])*(cached ? (joint ? 2:1):input_width);
    });
    if (anchored && spin_exchange_parity_) {
      // One component on the accepted two-component support fits the same
      // budget. Complete it before P_s(2 E_alpha Y + T q), without a new slab.
      std::vector<detail::ColumnTile*> tiles(panels,nullptr);
      for (const int panel:reachable) tiles[panel]=&input.get(panel);
      tiles.resize(reachable.back()+1);
      detail::project_exchange(tiles,extent,spin_exchange_parity_);
    }
    // Independent readers borrow the same bounded primal/tangent panels.
    const auto reader=[&](int family) -> mcscf::DirectCiTileReader {
      return [&,family,position=std::vector<int>(n_unique_alpha_,-1),
          tiles=std::vector<const detail::ColumnTile*>(panels,nullptr)](
          const std::vector<int>& betas,const std::vector<int>& rows,
          Eigen::Ref<Eigen::MatrixXd> values) mutable {
        const bool input_family=family==1 || family==3;
        if (resident_input && (!cached || input_family)) {
          const auto& image=family==3 ? ci_image:(family==1 ? direction_image:primal_image);
          const int workers=coarse_openmp_thread_count(cooperative_openmp_thread_count(),
              rows.size()*betas.size());
          cooperative_openmp_for(betas.size(),workers,[&](std::ptrdiff_t col) {
            for (int a=0;a<static_cast<int>(rows.size());++a)
              values(a,col)=image(rows[a],betas[col]);
          });
          return;
        }
        // Rows are sorted and distinct. A complete alpha range needs no map.
        const bool full_rows=rows.size()==static_cast<std::size_t>(n_unique_alpha_);
        if (!full_rows)
          for (int a=0;a<static_cast<int>(rows.size());++a) position[rows[a]]=a;
        const int component=cached ? ((family==2 || family==3) ? 1:0):(family==3 ? 2:family);
        const auto copy_column=[&](int col,const detail::ColumnTile& tile) {
          const int b=betas[col]%extent;
          const int first=tile.pattern->offsets[b],last=tile.pattern->offsets[b+1];
          if (tile.pattern->full) {
            if (full_rows) values.col(col)=tile.values.col(component).segment(first,n_unique_alpha_);
            else for (int a=0;a<static_cast<int>(rows.size());++a)
                values(a,col)=tile.values(first+rows[a],component);
          } else {
            values.col(col).setZero();
            for (int k=first;k<last;++k) {
              const int a=full_rows ? tile.pattern->rows[k]:position[tile.pattern->rows[k]];
              if (a>=0) values(a,col)=tile.values(k,component);
            }
          }
        };
        if (cached) {
          // A directional component on the accepted two-component support
          // also fits the same budget. Its borrowed owners therefore survive
          // the whole read, just like the immutable accepted Y/residual tiles.
          // Gather only owner pointers, then copy ALL beta columns in one team.
          if (input_family) {
            std::fill(tiles.begin(),tiles.end(),nullptr);
            for (const int b:betas) {
              const int panel=b/extent;
              if (reachable_mask[panel] && !tiles[panel]) tiles[panel]=&input.get(panel);
            }
          }
          const int workers=coarse_openmp_thread_count(cooperative_openmp_thread_count(),
              rows.size()*betas.size());
          cooperative_openmp_for(betas.size(),workers,[&](std::ptrdiff_t col) {
            const int panel=betas[col]/extent;
            if (!reachable_mask[panel]) values.col(col).setZero();
            else copy_column(col,input_family ? *tiles[panel]:
                state.product_images->states[state_id][panel]);
          });
          if (!full_rows) for (const int a:rows) position[a]=-1;
          return;
        }
        for (int j=0;j<static_cast<int>(betas.size());) {
          const int panel=betas[j]/extent;
          int end=j+1;
          while (end<static_cast<int>(betas.size()) && betas[end]/extent==panel) ++end;
          if (!reachable_mask[panel]) { values.middleCols(j,end-j).setZero(); j=end; continue; }
          const auto& tile=input.get(panel);
          const int workers=bounded_openmp_thread_count(cooperative_openmp_thread_count(),end-j);
          cooperative_openmp_for(end-j,workers,[&](std::ptrdiff_t index) {
            copy_column(j+index,tile);
          });
          j=end;
        }
        if (!full_rows) for (const int a:rows) position[a]=-1;
      };
    };
    const auto y=reader(0),dy=reader(tangent ? 1:0);
    const auto column=[&](int family) -> mcscf::DirectCiColumnReader {
      return [&,family](int b) -> mcscf::DirectCiColumnView {
        if (!reachable_mask[b/extent]) return {nullptr,nullptr,0};
        const bool input_family=family==1 || family==3;
        if (resident_input && (!cached || input_family)) {
          const auto& image=family==3 ? ci_image:(family==1 ? direction_image:primal_image);
          return {image.col(b).data(),nullptr,n_unique_alpha_};
        }
        const auto& tile=cached && !input_family ? state.product_images->states[state_id][b/extent]:input.get(b/extent);
        const int offset=tile.pattern->offsets[b%extent],last=tile.pattern->offsets[b%extent+1];
        const int component=cached ? ((family==2 || family==3) ? 1:0):(family==3 ? 2:family);
        return {tile.values.col(component).data()+offset,
            tile.pattern->full ? nullptr:tile.pattern->rows.data()+offset,last-offset};
      };
    };
    const auto y_column=column(0),dy_column=column(tangent ? 1:0);
    const auto residual_reader=reader(2);
    const auto residual_column=column(2);
    const auto q_reader=joint ? reader(3):mcscf::DirectCiTileReader{};
    const auto q_column=joint ? column(3):mcscf::DirectCiColumnReader{};
    const mcscf::DirectCiTileReader read=[&](const std::vector<int>& betas,
        const std::vector<int>& rows,Eigen::Ref<Eigen::MatrixXd> values) {
      if (cached) { dy(betas,rows,values); return; }
      using View=Eigen::Map<Eigen::MatrixXd,0,Eigen::OuterStride<>>;
      View primal(values.data(),rows.size(),betas.size(),Eigen::OuterStride<>(values.outerStride()*width));
      y(betas,rows,primal);
      if (tangent) {
        View direction(values.data()+values.outerStride(),rows.size(),betas.size(),
            Eigen::OuterStride<>(values.outerStride()*width));
        dy(betas,rows,direction);
      }
    };
    const auto right=interleaved_reader(y,sigma_width,tangent && !cached ? 1:0);
    // C^T is linear: first sum emitted panels in bounded source-group tiles,
    // then visit the structure incidence once per completed/evicted tile.
    // Neither the full touched support nor per-thread structure arrays are
    // required. Diagnostic outputs share the same cache and payload budget.
    const int projection_count=2+(joint ? 2:0)+(capture ? 3:0);
    const std::size_t products=static_cast<std::size_t>(n_unique_alpha_)*n_unique_beta_;
    // Associativity removes the per-panel beta-row reconstruction on broad
    // support. Primal and left-direction images share the existing budget.
    const bool resident_projection=(collect_images || capture) &&
        ordered_beta && products<=projection_bytes/
            (sizeof(double)*projection_count*(delta_upper ? 2:1));
    std::vector<Eigen::MatrixXd> projected(projection_count),left_projected(projection_count);
    const auto group_extent=std::max<std::size_t>(1,std::min<std::size_t>(
        static_cast<std::size_t>(n_unique_alpha_)*extent,
        projection_bytes/(sizeof(double)*projection_count)));
    std::vector<double*> projection_targets;
    detail::ColumnPanels projections(projection_bytes,[&](int block) {
      const auto begin=static_cast<std::size_t>(block)*group_extent;
      const auto count=std::min(group_extent,spin_products_.size()-begin);
      auto pattern=std::make_shared<detail::ColumnPattern>();
      pattern->full=true; pattern->offsets.resize(2); pattern->offsets<<0,count;
      return detail::ColumnTile{pattern,Eigen::MatrixXd::Zero(count,projection_count)};
    },[&](int block,detail::ColumnTile& tile) {
      const auto begin=static_cast<std::size_t>(block)*group_extent;
      for (int target=0;target<static_cast<int>(projection_targets.size());++target)
        for (int local=0;local<tile.values.rows();++local) {
          const double value=tile.values(local,target);
          if (value==0.0) continue;
          const auto g=begin+local;
          for (auto t=spin_term_offsets_[g];t<spin_term_offsets_[g+1];++t) {
            const auto& term=spin_terms_[t];
            projection_targets[target][term.structure]+=term.coefficient*value;
          }
        }
    },[&](int block) {
      const auto begin=static_cast<std::size_t>(block)*group_extent;
      return sizeof(double)*std::min(group_extent,spin_products_.size()-begin)*projection_count+
          2*sizeof(int);
    });
    // Consume one or two independent output panels. Concatenated RHS blocks
    // share the beta rows, support traversal and mutable alpha transform.
    const auto project=[&](int panel,Eigen::MatrixXd values,
        std::initializer_list<double*> targets,bool differentiate=false) {
      profiling::ScopedPhase phase(profiling::Phase::CiProductProjection);
      const int first=panel*extent,count=values.cols()/targets.size();
      Eigen::MatrixXd dvalues;
      if (differentiate) {
        dvalues=Eigen::MatrixXd::Zero(values.rows(),values.cols());
        alpha.apply_directional_adjoint_left(da,&values,&dvalues);
      } else {
        const auto pattern=pattern_at(panel);
        if (pattern->full) alpha.apply_adjoint_left(&values);
        else alpha.apply_supported_left(&values,
            {pattern->offsets.data(),static_cast<std::size_t>(pattern->offsets.size())},
            {pattern->rows.data(),static_cast<std::size_t>(pattern->rows.size())},true);
      }
      std::array<int,2> target_ids;
      int lane=0;
      for (double* target:targets) {
        auto found=std::find(projection_targets.begin(),projection_targets.end(),target);
        target_ids[lane++]=found-projection_targets.begin();
        if (found==projection_targets.end()) projection_targets.push_back(target);
      }
      if (resident_projection) {
        for (int v=0;v<lane;++v) {
          const int target_id=target_ids[v];
          if (!projected[target_id].size()) {
            projected[target_id]=Eigen::MatrixXd::Zero(n_unique_alpha_,n_unique_beta_);
          }
          projected[target_id].middleCols(first,count)+=differentiate ?
              dvalues.middleCols(v*count,count):values.middleCols(v*count,count);
          if (differentiate) {
            if (!left_projected[target_id].size())
              left_projected[target_id]=Eigen::MatrixXd::Zero(n_unique_alpha_,n_unique_beta_);
            left_projected[target_id].middleCols(first,count)+=values.middleCols(v*count,count);
          }
        }
        return;
      }
      const auto rows=beta.row_block(first,count);
      Eigen::MatrixXd dr;
      if (differentiate) dr=beta_directions.get(panel).unpack(n_unique_beta_).transpose();
      for (std::size_t begin=0;begin<spin_products_.size();begin+=group_extent) {
        auto& tile=projections.get(begin/group_extent);
        const auto size=std::min(group_extent,spin_products_.size()-begin);
        const int workers=coarse_openmp_thread_count(cooperative_openmp_thread_count(),size*count*lane);
        cooperative_openmp_for(size,workers,[&](std::ptrdiff_t local) {
          const auto g=begin+local;
          const int a=spin_products_[g]/n_unique_beta_,b=spin_products_[g]%n_unique_beta_;
          for (int v=0;v<lane;++v)
            tile.values(local,target_ids[v])+=differentiate ?
                dvalues.row(a).segment(v*count,count).dot(rows->col(b))+
                values.row(a).segment(v*count,count).dot(dr.col(b)):
                values.row(a).segment(v*count,count).dot(rows->col(b));
        });
      }
      result.images.tiles.structure_projections+=lane;
    };
    const auto sweep=[&]() {
      const int w=sigma_width;
      const bool forcing=capture && delta_sigma;
      const int q_slot=w,forcing_slot=w+(joint ? 1:0);
      const int output_width=w+(joint ? 1:0)+(forcing ? 1:0);
      bool retain=false;
      bool exchange_reduced=false;
      detail::ColumnPanels output(cache_bytes,[&](int panel) {
        auto pattern=pattern_at(panel);
        return detail::ColumnTile{pattern,Eigen::MatrixXd::Zero(pattern->offsets[pattern->columns()],output_width)};
      },[&](int panel,detail::ColumnTile& tile) {
        ++result.images.tiles.hamiltonian_projections;
        const int first=panel*extent,count=tile.pattern->columns();
        Eigen::MatrixXd images=tile.unpack(n_unique_alpha_);
        const auto h0=images.leftCols(count);
        if (weight!=0) {
          if (joint) {
            // H*dy and H*q have the same right Y. Their cotangents stay
            // independent while Y reads and signed spin links are shared.
            const Eigen::Ref<const Eigen::MatrixXd> lefts[]{
                images.middleCols((cached ? 0:w-1)*count,count),
                images.middleCols(q_slot*count,count)};
            const auto generators=sigma.generator_adjoint_streamed(first,lefts,
                {{y,y_column,1.0,false,cached}},false,exchange_reduced);
            result.generator.noalias()+=2*weight*generators[0];
            ci_sweep->generator.noalias()+=2*weight*generators[1];
            if (!cached) {
              result.generator.noalias()+=2*weight*sigma.generator_adjoint_streamed(
                  first,h0,{{dy,dy_column}},false,exchange_reduced);
              ci_sweep->generator.noalias()+=2*weight*sigma.generator_adjoint_streamed(
                  first,h0,{{q_reader,q_column}},false,exchange_reduced);
            }
          } else if (!tangent || cached)
            result.generator.noalias()+=2*weight*sigma.generator_adjoint_streamed(first,h0,
                {{y,y_column,1.0,false,cached}},false,exchange_reduced);
          else {
            result.generator.noalias()+=2*weight*sigma.generator_adjoint_streamed(
                first,h0,{{dy,dy_column}},false,exchange_reduced);
            result.generator.noalias()+=2*weight*sigma.generator_adjoint_streamed(
                first,images.middleCols((w-1)*count,count),{{y,y_column}},false,exchange_reduced);
          }
        }
        if (joint) {
          project(panel,cached && !capture ? std::move(images):Eigen::MatrixXd(
              images.middleCols((cached ? 0:w-1)*count,2*count)),
              {result.images.hamiltonian.col(state_id).data(),
               ci_sweep->images.hamiltonian.col(state_id).data()});
          if (delta_upper && !cached)
            project(panel,h0,{result.images.hamiltonian.col(state_id).data()},true);
        } else if (collect_images) {
          if (!tangent || cached) project(panel,capture ? Eigen::MatrixXd(h0):std::move(images),
              {result.images.hamiltonian.col(state_id).data()});
          else {
            project(panel,images.middleCols((w-1)*count,count),{result.images.hamiltonian.col(state_id).data()});
            if (delta_upper) project(panel,h0,{result.images.hamiltonian.col(state_id).data()},true);
          }
        }
        if (capture) {
          if (forcing) project(panel,images.middleCols(forcing_slot*count,count),
              {diagnostic->connection->integral_forcing.col(state_id).data()});
          if (tangent) {
            project(panel,cached ? h0:Eigen::MatrixXd(images.middleCols((w-1)*count,count)),
                {diagnostic->connection->representation_forcing.col(state_id).data()});
            if (delta_upper && !cached)
              project(panel,h0,{diagnostic->connection->representation_forcing.col(state_id).data()},true);
          }
        }
        if (retain) {
          // Store Y and (H-E)Y on the exact support, sharing one pattern.
          // The next coefficient response then needs neither a new Y
          // transform nor a separate -E*D(Y,q) generator contraction.
          detail::ColumnTile accepted{tile.pattern,Eigen::MatrixXd(tile.values.rows(),2)};
          if (resident_input) {
            // Missing raw products may leave holes in the exact output
            // support. A contiguous prefix would shift subsequent columns.
            const auto& pattern=*tile.pattern;
            if (pattern.full)
              accepted.values.col(0)=Eigen::Map<const Eigen::VectorXd>(
                  primal_image.col(first).data(),accepted.values.rows());
            else for (int b=0;b<count;++b)
              for (int i=pattern.offsets[b];i<pattern.offsets[b+1];++i)
                accepted.values(i,0)=primal_image(pattern.rows[i],first+b);
          }
          else accepted.values.col(0)=input.get(panel).values.col(0);
          accepted.values.col(1)=tile.values.col(0)-energy*accepted.values.col(0);
          state.product_images->bytes+=accepted.bytes();
          state.product_images->states[state_id][panel]=std::move(accepted);
        }
      },[&](int panel) {
        const auto pattern=pattern_at(panel);
        return pattern->bytes()+sizeof(double)*
            static_cast<std::size_t>(pattern->offsets[pattern->columns()])*output_width;
      });
      struct OutputColumn { detail::ColumnTile* tile; int source,beta; };
      std::vector<OutputColumn> output_columns;
      const auto writer=[&](int source_begin,int vectors,int destination_begin) -> mcscf::DirectCiTileWriter {
        return [&,source_begin,vectors,destination_begin](const std::vector<int>& betas,
          const std::vector<int>& rows,const mcscf::DirectCiColumnAccumulator& accumulate) {
        const int workers=cooperative_openmp_thread_count(),chunks=std::min<int>(workers,rows.size());
        int j=0;
        while (j<static_cast<int>(betas.size())) {
          output_columns.clear();
          std::size_t bytes=0;
          // Borrow only a budget-fitting group of panels. LRU entries touched
          // here are newer than all unborrowed entries, so no get() in this
          // group can evict an earlier borrowed panel. Tasks finish before
          // the next group mutates the cache; only small pointers are shared.
          while (j<static_cast<int>(betas.size())) {
            const int panel=betas[j]/extent;
            int end=j+1;
            while (end<static_cast<int>(betas.size()) && betas[end]/extent==panel) ++end;
            if (!reachable_mask[panel]) { j=end; continue; }
            const auto pattern=pattern_at(panel);
            const std::size_t payload=pattern->bytes()+sizeof(double)*
                static_cast<std::size_t>(pattern->offsets[pattern->columns()])*output_width;
            if (!output_columns.empty() &&
                (payload>cache_bytes || bytes>cache_bytes-payload)) break;
            auto& tile=output.get(panel);
            for (;j<end;++j) output_columns.push_back({&tile,j,betas[j]%extent});
            bytes+=payload;
          }
          const int columns=output_columns.size()*vectors;
          cooperative_openmp_for(columns*chunks,workers,[&](std::ptrdiff_t index) {
            const int column=index%columns,chunk=index/columns;
            const auto& output_column=output_columns[column/vectors];
            auto& tile=*output_column.tile;
            const int beta=output_column.source,v=column%vectors,b=output_column.beta;
            const int offset=tile.pattern->offsets[b],length=tile.pattern->offsets[b+1]-offset;
            if (length==0) return;
            accumulate(beta,source_begin+v,rows.size()*chunk/chunks,rows.size()*(chunk+1)/chunks,
                tile.values.col(destination_begin+v).data()+offset,
                tile.pattern->full ? nullptr:tile.pattern->rows.data()+offset,tile.pattern->full ? 0:length);
          });
        }
        };
      };
      const auto add=writer(0,w,0);
      // Complete each output panel at its last sigma contributor. Unlike
      // input lifetimes, this must include every temporal response/forcing.
      // Keep a fully resident sweep intact for exchange projection and reuse;
      // once capacity eviction occurs, retire completed panels immediately.
      // The generator reads neighboring INPUT panels, so do not retire those
      // using the shorter sigma-only lifetime.
      std::vector<int> remaining(panels),completed;
      for (int p=0;p<panels;++p) remaining[p]=std::min(extent,n_unique_beta_-p*extent);
      const mcscf::DirectCiColumnRelease release=[&](std::span<const int> columns) {
        for (const int b:columns)
          if (--remaining[b/extent]==0) completed.push_back(b/extent);
        if (output.evictions) {
          for (const int p:completed) output.release(p);
          completed.clear();
        }
      };
      {
        // Accepted bra and ket are identical. Keep the empty-reader identity
        // so the channel kernel reuses its scatter; apply the weight outside.
        mcscf::DirectCiIntegralAdjoint gradient;
        if (joint) {
          const auto ci_read=interleaved_reader(q_reader,w,w-1);
          const mcscf::DirectCiTileRequest requests[]={
            {ci_read,writer(w-1,1,q_slot)},
            {read,add,delta_sigma,forcing ? writer(w-1,1,forcing_slot):mcscf::DirectCiTileWriter{}}
          };
          auto gradients=sigma.apply_tiled_responses(w,requests,right,support,spin_exchange_parity_,release,work_bytes);
          ci_sweep->one_electron.noalias()+=2*weight*gradients[0].one_electron;
          ci_sweep->pair_kernel.noalias()+=2*weight*gradients[0].pair_kernel;
          gradient=std::move(gradients[1]);
        } else gradient=delta_sigma ? sigma.apply_tiled_direction(w,read,right,*delta_sigma,add,
            forcing ? writer(w-1,1,forcing_slot):mcscf::DirectCiTileWriter{},support,spin_exchange_parity_,release,work_bytes):
            sigma.apply_tiled_with_adjoint(w,read,add,
                tangent ? right:mcscf::DirectCiTileReader{},support,spin_exchange_parity_,release,work_bytes);
        // Keep state weights outside the shared scatter and both kernel actions.
        result.one_electron.noalias()+=(tangent ? 2*weight:weight)*gradient.one_electron;
        result.pair_kernel.noalias()+=(tangent ? 2*weight:weight)*gradient.pair_kernel;
        if (!tangent) {
          // d<Y|H|Y>/dh is the same accepted 1-RDM as D(Y,Y).
          // Reuse it for the norm multiplier; a second generator sweep
          // would only reconstruct this already available symmetric moment.
          for (int j=0;j<n;++j)
            for (int i=0;i<=j;++i)
              result.generator(i,j)-=2*weight*energy*gradient.one_electron(i,j);
          if (states==1 && state.product_images)
            state.product_images->density=gradient.one_electron;
        }
        ++result.images.tiles.integral_adjoint_sweeps;
      }
      if (output.evictions==0 && (spin_exchange_parity_ || (!tangent && state.product_images)))
        for (const int panel:reachable) output.get(panel);
      if (spin_exchange_parity_ && output.evictions==0) {
        // Resolve the weighted representative before using alpha=beta in
        // generator contractions or retaining the accepted residual image.
        std::vector<detail::ColumnTile*> tiles(panels,nullptr);
        for (const int panel:reachable) tiles[panel]=&output.get(panel);
        // The last reachable panel fixes the range; absent later columns are zero.
        tiles.resize(reachable.back()+1);
        detail::project_exchange(tiles,extent,spin_exchange_parity_);
        exchange_reduced=true;
      }
      std::size_t accepted_bytes=0;
      if (!tangent && state.product_images && output.evictions==0)
        for (const int panel:reachable) {
          const auto& tile=output.get(panel);
          accepted_bytes+=tile.pattern->bytes()+2*sizeof(double)*tile.values.rows();
        }
      // Count shared row metadata once, not twice. A cached two-lane tangent
      // must fit the same input allowance; otherwise its borrowed views would
      // be invalidated by an LRU insertion during the parallel reader.
      const auto retain_limit=std::min(retained_bytes,cache_bytes);
      if (!tangent && state.product_images && output.evictions==0 &&
          state.product_images->bytes<=retain_limit &&
          accepted_bytes<=retain_limit-state.product_images->bytes &&
          state.product_images->residency.try_resize(
              state.product_images->bytes+accepted_bytes)) {
        retain=true;
        state.product_images->states[state_id].resize(panels);
      }
      output.finish();
      result.images.tiles.output_evictions+=output.evictions;
      result.images.tiles.output_resident_bytes=std::max(result.images.tiles.output_resident_bytes,output.peak_bytes);
    };
    if (cached && collect_images && delta_upper) {
      // delta T = G T implies delta T^T Z = T^T G^T Z. Consume the
      // accepted Y/residual directly, rather than differentiating both exterior
      // transforms again. Only their ordinary selected-space projection remains;
      // no closure of the selected VB space is assumed.
      const Eigen::MatrixXd dual_generator=generator.transpose();
      const auto& images=state.product_images->states[state_id];
      for (const int panel:reachable) {
        if (!images[panel].pattern) continue;
        const int first=panel*extent;
        const int count=images[panel].pattern->columns();
        Eigen::MatrixXd values(n_unique_alpha_,2*count);
        sigma.apply_one_body_generator_columns(first,y_column,dual_generator,
            values.leftCols(count),spin_exchange_parity_!=0);
        sigma.apply_one_body_generator_columns(first,residual_column,dual_generator,
            values.rightCols(count),spin_exchange_parity_!=0);
        // T preserves the certified spin-exchange parity, so its projection
        // of the beta generator equals that of the alpha generator. This
        // identity is used only for a projection, not for a retained carrier.
        if (spin_exchange_parity_) values*=2.0;
        values.rightCols(count).noalias()+=energy*values.leftCols(count);
        if (capture) project(panel,values.rightCols(count),
            {diagnostic->connection->representation_forcing.col(state_id).data()});
        project(panel,std::move(values),{result.images.overlap.col(state_id).data(),
            result.images.hamiltonian.col(state_id).data()});
      }
    }
    sweep();
    double norm=0,parallel=0,square=0;
    std::vector<int> all_rows(n_unique_alpha_); std::iota(all_rows.begin(),all_rows.end(),0);
    // Accepted fixed-state normalization already supplies M*C. The sigma
    // sweep has consumed Y for all densities/generator terms above; only the
    // redundant structure-overlap projection remains, so omit that reread.
    if (tangent || (collect_images && !overlap_image) || capture) for (const int panel:reachable) {
      const int first=panel*extent,count=std::min(extent,n_unique_beta_-first);
      std::vector<int> betas(count); std::iota(betas.begin(),betas.end(),first);
      Eigen::MatrixXd values,directions;
      if (!tangent || (!cached && delta_upper) || capture) {
        values.resize(n_unique_alpha_,count);
        y(betas,all_rows,values);
      }
      if (tangent) directions.resize(n_unique_alpha_,(joint ? 2:1)*count);
      auto dv=directions.leftCols(tangent ? count:0);
      auto dq=directions.rightCols(joint ? count:0);
      if (tangent) dy(betas,all_rows,dv);
      if (joint) q_reader(betas,all_rows,dq);
      if (weight!=0 && tangent) {
          // D(R,Z)=D(Z,R)^T: both accepted cotangents share the generated
          // Z panel and one spin-link traversal, without unpacking R first.
          if (cached && joint) {
            const Eigen::Ref<const Eigen::MatrixXd> lefts[]{dv,dq};
            const auto generators=sigma.generator_adjoint_streamed(first,lefts,
                {{y,y_column,-energy,false,true},{residual_reader,residual_column,1.0,true,true}},
                false,spin_exchange_parity_!=0);
            result.generator.noalias()+=2*weight*generators[0];
            ci_sweep->generator.noalias()+=2*weight*generators[1];
          } else if (cached) result.generator.noalias()+=2*weight*sigma.generator_adjoint_streamed(
              first,dv,{{y,y_column,-energy,false,true},{residual_reader,residual_column,1.0,true,true}},
              false,spin_exchange_parity_!=0);
          else result.generator.noalias()-=2*weight*energy*
              sigma.generator_adjoint_streamed(first,dv,{{y,y_column}},true,spin_exchange_parity_!=0);
      }
      if (joint && weight!=0 && !cached)
        ci_sweep->generator.noalias()-=2*weight*energy*sigma.generator_adjoint_streamed(
              first,dq,{{y,y_column}},true,spin_exchange_parity_!=0);
      // Consume diagnostics before the destructive physical projections too.
      // Capture then needs no second copy of both direction components.
      if (capture) {
        project(panel,dv,{diagnostic->connection->overlap.col(state_id).data()});
        norm+=values.squaredNorm(); parallel+=(values.array()*dv.array()).sum(); square+=dv.squaredNorm();
      }
      // All original-image consumers are finished. Move the shared owner
      // directly into its final selected-space projection.
      if (joint) {
        project(panel,std::move(directions),
            {result.images.overlap.col(state_id).data(),ci_sweep->images.overlap.col(state_id).data()});
      } else if (collect_images) {
        if (!tangent) project(panel,std::move(values),
            {result.images.overlap.col(state_id).data()});
        else project(panel,std::move(directions),
            {result.images.overlap.col(state_id).data()});
      }
      if (collect_images && delta_upper && !cached)
        project(panel,std::move(values),{result.images.overlap.col(state_id).data()},true);
    }
    projections.finish();
    if (resident_projection) {
      profiling::ScopedPhase phase(profiling::Phase::CiProductProjection);
      for (int target=0;target<static_cast<int>(projection_targets.size());++target) {
        // CI-only/forcing targets have no left-direction component. Do not
        // allocate or transform a zero carrier merely because p also exists.
        if (left_projected[target].size())
          beta.apply_directional_adjoint_right(db,&left_projected[target],&projected[target]);
        else beta.apply_adjoint_right(&projected[target]);
        Eigen::Map<Eigen::VectorXd>(projection_targets[target],n_structures_)+=
            contract_spin_product_block(projected[target]).col(0);
        ++result.images.tiles.structure_projections;
      }
    }
    if (capture) {
      if (delta_sigma) diagnostic->connection->representation_forcing.col(state_id)-=
          diagnostic->connection->integral_forcing.col(state_id);
      diagnostic->connection->squared_norm[state_id]=square-(norm>0 ? parallel*parallel/norm:0);
      diagnostic->connection->representation_forcing.col(state_id).noalias()-=energy*result.images.overlap.col(state_id);
    }
    result.images.tiles.input_evictions+=input.evictions;
    result.images.tiles.input_resident_bytes=std::max(result.images.tiles.input_resident_bytes,input.peak_bytes);
    result.images.tiles.input_resident_bytes=std::max(result.images.tiles.input_resident_bytes,input_bytes);
  }
  if (!tangent && states==1 && state.product_images)
    state.product_images->density=state.product_images->density.selfadjointView<Eigen::Upper>();
  if (profiling::phase_totals().enabled)
    std::fprintf(stderr,"XMVB_PRODUCT_JET,tangent=%d,joint=%d,states=%d,extent=%d,"
        "input_panels=%zu,output_consumes=%zu,input_evictions=%zu,output_evictions=%zu,"
        "accepted_bytes=%zu,column_budget=%zu,sigma_budget=%zu\n",
        tangent,joint,states,extent,result.images.tiles.input_panels,
        result.images.tiles.hamiltonian_projections,result.images.tiles.input_evictions,
        result.images.tiles.output_evictions,state.product_images ? state.product_images->bytes:0,
        cache_bytes,work_bytes);
  return result;
}

} // namespace xmvb::vb
