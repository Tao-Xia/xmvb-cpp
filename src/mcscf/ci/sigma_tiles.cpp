#include "mcscf/ci/sigma.hpp"
#include "mcscf/ci/workspace.hpp"
#include "mcscf/ci/scratch.hpp"

#include <algorithm>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include "parallel/openmp.hpp"
#include "core/pair_index.hpp"
#include "core/timing.hpp"
#include "core/workspace.hpp"
#include "mcscf/ci/product_support.hpp"
#include "mcscf/ci/sigma_plan.hpp"
#include "mcscf/ci/channel_product.hpp"
#include "mcscf/ci/sigma_gather.hpp"

namespace xmvb::mcscf {
using detail::ChannelWorkspace;



void DirectCiSigmaAction::apply_tiled(
    int width,
    const DirectCiTileReader& read_columns, const DirectCiTileWriter& add_columns,
    const ProductSupport* support, int exchange_parity,
    const DirectCiColumnRelease& release,std::size_t workspace_bytes,
    bool source_projected_image) const {
  profiling::ScopedPhase phase(profiling::Phase::CiSigma);
  const DirectCiTileRequest request{read_columns,add_columns};
  contract_tiled(width,{&request,1},{},false,support,exchange_parity,release,workspace_bytes,
      source_projected_image);
}

DirectCiIntegralAdjoint DirectCiSigmaAction::apply_tiled_with_adjoint(
    int width, const DirectCiTileReader& response, const DirectCiTileWriter& add_columns,
    const DirectCiTileReader& state, const ProductSupport* support, int exchange_parity,
    const DirectCiColumnRelease& release,std::size_t workspace_bytes) const {
  profiling::ScopedPhase phase(profiling::Phase::CiResponse);
  const DirectCiTileRequest request{response,add_columns};
  auto results=contract_tiled(width,{&request,1},state,true,support,exchange_parity,release,workspace_bytes);
  return std::move(results.front());
}

DirectCiIntegralAdjoint DirectCiSigmaAction::apply_tiled_direction(
    int width, const DirectCiTileReader& response, const DirectCiTileReader& state,
    const DirectCiSigmaAction& direction, const DirectCiTileWriter& add_columns,
    const DirectCiTileWriter& add_forcing, const ProductSupport* support,
    int exchange_parity,const DirectCiColumnRelease& release,std::size_t workspace_bytes) const {
  profiling::ScopedPhase phase(profiling::Phase::CiResponse);
  const DirectCiTileRequest request{response,add_columns,&direction,add_forcing};
  auto results=contract_tiled(width,{&request,1},state,true,support,exchange_parity,release,workspace_bytes);
  return std::move(results.front());
}

std::vector<DirectCiIntegralAdjoint> DirectCiSigmaAction::apply_tiled_responses(
    int width,std::span<const DirectCiTileRequest> requests,const DirectCiTileReader& state,
    const ProductSupport* support,int exchange_parity,
    const DirectCiColumnRelease& release,std::size_t workspace_bytes) const {
  profiling::ScopedPhase phase(profiling::Phase::CiResponse);
  if (!state) throw std::invalid_argument("temporal responses require a common state");
  return contract_tiled(width,requests,state,true,support,exchange_parity,release,workspace_bytes);
}

std::vector<DirectCiIntegralAdjoint> DirectCiSigmaAction::contract_tiled(
    int width,std::span<const DirectCiTileRequest> requests,const DirectCiTileReader& state,
    bool collect_integrals,const ProductSupport* support,int exchange_parity,
    const DirectCiColumnRelease& release,std::size_t workspace_bytes,
    bool source_projected_image) const {
  if (width<=0 || requests.empty() || (requests.size()>1 && !state))
    throw std::invalid_argument("invalid tiled direct-CI width or response sequence");
  if (exchange_parity < -1 || exchange_parity > 1 ||
      (exchange_parity != 0 && distinct_beta_))
    throw std::invalid_argument("exchange projection requires identical spin spaces");
  bool has_direction=false;
  for (const auto& request:requests) {
    const auto* direction=request.direction;
    if (!request.response || (!request.image && !collect_integrals) ||
        (direction && !state) || (request.forcing && !direction))
      throw std::invalid_argument("invalid tiled direct-CI response callbacks");
    if (direction && (n_orbitals_ != direction->n_orbitals_ ||
                    n_alpha_ != direction->n_alpha_ || n_beta_ != direction->n_beta_ ||
                    alpha_ != direction->alpha_ ||
                    distinct_beta_ != direction->distinct_beta_ ||
                    hamiltonian_pair_kernel_.rows() !=
                        direction->hamiltonian_pair_kernel_.rows()))
      throw std::invalid_argument("direction requires shared direct-CI spin links");
    has_direction=has_direction || direction;
  }

  // Direct kernel callers own one allowance too. Structure actions already
  // provide a scope and pass this kernel its sigma partition explicitly.
  std::optional<CiWorkspace> operation;
  if (!workspace::has_scope()) {
    operation.emplace(workspace_bytes);
    workspace_bytes=operation->bytes();
  }

  const auto& beta_graph = beta_connections();
  const int pairs = hamiltonian_pair_kernel_.rows();
  const std::size_t n=n_orbitals_;
  const std::size_t envelopes=n*(n-1)*(n+1)*(n+2)/24;
  const auto directions=std::count_if(requests.begin(),requests.end(),
      [](const auto& request) { return request.direction!=nullptr; });
  std::size_t kernel_entries=pairs*envelopes;
  if (source_projected_image) {
    kernel_entries=0;
    for (int first=0;first<n_orbitals_;++first)
      for (int last=first;last<n_orbitals_-1;++last) {
        const auto k=static_cast<std::size_t>((first+1)*(n_orbitals_-1-last));
        kernel_entries+=k*k;
      }
  }
  const std::size_t kernel_bytes=(sizeof(double)*kernel_entries+
      sizeof(int)*(envelopes+pairs*n*(n-1)/2))*(1+directions);
  // Packing is optional arithmetic scheduling, not a change of CI space.
  // Charge shared immutable kernel panels to this action's existing budget.
  const bool restricted_channels=support && support->has_channel_envelopes() &&
      kernel_bytes<=workspace_bytes/8;
  if (restricted_channels) workspace_bytes-=kernel_bytes;
  ChannelWorkspace::limit_action(workspace_bytes);
  const bool projected_image=source_projected_image && restricted_channels;
  const detail::ChannelProduct channel_product(hamiltonian_pair_kernel_,n_orbitals_,restricted_channels,
      projected_image);
  std::vector<detail::ChannelProduct> directional_products;
  directional_products.reserve(requests.size());
  for (const auto& request:requests)
    directional_products.emplace_back(request.direction ? request.direction->hamiltonian_pair_kernel_:
        hamiltonian_pair_kernel_,n_orbitals_,restricted_channels && request.direction);
  std::vector<Eigen::MatrixXd> kernel_adjoints(requests.size());
  if (collect_integrals) for (auto& gradient:kernel_adjoints)
    gradient=Eigen::MatrixXd::Zero(pairs,pairs);
  workspace::Temporary adjoint_temporary(collect_integrals ?
      sizeof(double)*static_cast<std::size_t>(pairs)*pairs*requests.size():0);
  // Source and state are ragged; images retain all channels. Directional
  // forcing normally adds directly into the image. Only an explicit forcing
  // consumer needs to expand the consumed source slab into a second image.
  const bool separate_output=has_direction || requests.size()>1;
  const bool has_forcing=std::any_of(requests.begin(),requests.end(),
      [](const auto& request) { return static_cast<bool>(request.forcing); });
  const int slab_factor=separate_output ? 3:2;
  const int packed_slabs=restricted_channels ? (separate_output && !has_forcing ? 2:1):0;
  const int row_limit = static_cast<int>(std::max<std::size_t>(1,
      std::min<std::size_t>(std::numeric_limits<int>::max() / width,
          workspace_bytes / (slab_factor * sizeof(double) * width * pairs))));
  const int workers = coarse_openmp_thread_count(effective_openmp_thread_count(),
      static_cast<std::size_t>(row_limit) * width * pairs);

  struct Incoming { int group, pair; double sign; };
  // Match the existing product-column consumer's parallel panel extent.
  // More resource headroom permits tile reuse; it must not inflate one
  // channel tensor over the entire product sector. A beta panel already
  // exposes alpha-row parallelism and the same fused contractions.
  const int panel_columns=static_cast<int>(std::min<std::size_t>(n_beta_,
      static_cast<std::size_t>(cooperative_openmp_thread_count())*(64/sizeof(double))));
  detail::SigmaTilePlan plan(*alpha_,beta_graph,support,exchange_parity,
      {width,pairs,slab_factor,workspace_bytes,packed_slabs,1,panel_columns,projected_image},
      static_cast<bool>(release));
  auto& groups=plan.groups;
  auto& active=plan.active;
  auto& group_index=plan.group_index;
  auto& halo=plan.halo;
  auto& columns=plan.columns;
  auto& halo_index=plan.halo_index;
  auto& column_index=plan.column_index;
  auto& active_support=plan.active_support;
  auto& released_columns=plan.released;
  const auto& diagonal_rows=plan.diagonal_rows;
  std::vector<int> same_group,offsets,positions,image_index,image_groups;
  std::vector<detail::SigmaGatherRow> target_rows;
  const auto& channel_groups=plan.channel_groups;
  detail::ChannelLayout channel_layout;
  std::vector<Eigen::Index> beta_offsets;
  std::vector<Incoming> incoming;
  ChannelWorkspace workspace(workspace_bytes);
  std::size_t panel_size=0,channel_size=0,scratch_size=0,packed_size=0,source_capacity=0;
  int count=0,n_halo=0,n_columns=0;
  bool finished=false;
  std::exception_ptr failure;
  const bool timing = profiling::phase_totals().enabled;
  using Clock = std::chrono::steady_clock;
  const auto tick = [&]() { return timing ? Clock::now() : Clock::time_point{}; };
  const auto seconds = [](Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double>(end - begin).count();
  };
  double prepare_time = 0, read_time = 0, compute_time = 0, write_time = 0, gram_time = 0;
  std::size_t gram_tiles=0;
  std::size_t gram_scratch_bytes=0;
  int gram_partitions=0,gram_extent=0;
  std::size_t batches = 0, source_columns = 0, source_elements = 0, peak_payload = 0;
  std::size_t targets = 0, unsupported_targets = 0, identity_halo_targets=0,beta_map_bytes=0;
  std::size_t source_channels=0;
  long double projected_products=0,full_products=0;
  Clock::time_point compute_start, prepare_start, gram_start;
#pragma omp parallel if(workers > 1) num_threads(workers)
  {
    int thread = 0, team = 1;
#ifdef _OPENMP
    thread = omp_get_thread_num();
    team = omp_get_num_threads();
#endif
    for (;;) {
#pragma omp single
      {
        try {
          prepare_start=tick();
          {
            profiling::ScopedPhase phase(profiling::Phase::CiTilePaths);
            finished=!plan.next();
          }
          if (!failure) {
            unsupported_targets+=plan.unsupported_targets;
            if (!active.empty()) {
              count = active.size();
              {
                profiling::ScopedPhase phase(profiling::Phase::CiTileLayout);
                channel_layout.reset(channel_product,channel_groups,count,width);
              }
              packed_size=channel_layout.entries();
              source_channels+=packed_size/width;
              if (timing) for (int group=0;group<channel_product.groups();++group) {
                const long double entries=static_cast<long double>(channel_layout.size(group))*
                    channel_product.channels(group);
                full_products+=entries*pairs;
                projected_products+=entries*channel_product.kernel(group).rows();
              }
              n_halo = halo.size();
              n_columns = columns.size();
              panel_size = static_cast<std::size_t>(n_halo) * n_columns * width;
              channel_size = static_cast<std::size_t>(count) * width * pairs;
              scratch_size=panel_size;
              source_capacity=has_forcing ? channel_size:packed_size;
              const std::size_t required=scratch_size+source_capacity+
                  (separate_output ? packed_size:0)+channel_size;
              {
                profiling::ScopedPhase phase(profiling::Phase::CiTileStorage);
                workspace.reserve(required,plan.payload_bytes());
              }
              Eigen::Map<Eigen::MatrixXd> input(workspace.get(), n_halo,
                  static_cast<Eigen::Index>(n_columns) * width);
              // Each distinct source column is read once per batch. Callbacks
              // mutate the panel LRU and therefore remain outside worker loops.
              const auto read_start = tick();
              if (timing && batches==0)
                std::fprintf(stderr,"XMVB_TILE_PROGRESS,event=first_read,rows=%d,columns=%d,halo=%d\n",
                    count,n_columns,n_halo);
              requests.front().response(columns, halo, input);
              const double batch_read_time = seconds(read_start, tick());
              read_time += batch_read_time;
              ++batches;
              targets += count;
              if (n_halo==n_alpha_) identity_halo_targets+=count;
              source_columns += n_columns;
              source_elements += panel_size;
              peak_payload=std::max(peak_payload,plan.payload_bytes()+
                  (restricted_channels ? kernel_bytes:0));

              profiling::ScopedPhase indices_phase(profiling::Phase::CiTileIndices);
              same_group.assign(n_columns, -1);
              image_index.assign(static_cast<std::size_t>(n_halo)*groups.size()*width,-1);
              if (projected_image) image_groups.resize(image_index.size());
              target_rows.resize(count);
              offsets.assign(n_columns + 1, 0);
              // Each beta group owns a disjoint target interval and image-map
              // slice. Borrow the existing team; no per-worker arrays or
              // nested parallel region are needed. Tasks finish before sigma.
              const int index_workers=coarse_openmp_thread_count(team,count);
              cooperative_openmp_for(groups.size(),index_workers,[&](std::ptrdiff_t g) {
                same_group[column_index[groups[g].beta]] = g;
                for (int i = groups[g].begin; i < groups[g].end; ++i) {
                  const int image=channel_layout.image(i);
                  const int stride=width>1 ? channel_layout.image(count+i)-image:0;
                  const int channel_group=channel_groups.empty() ? 0:channel_groups[i];
                  target_rows[i]={active[i],image,stride,channel_group};
                  // Compose E_alpha^T's row map before traversing its links.
                  // The same physical image serves every temporal response;
                  // RHS maps are shared, not copied into worker workspaces.
                  const int h=halo_index[active[i]];
                  for (int v=0;v<width;++v) {
                    const auto at=(static_cast<std::size_t>(v)*groups.size()+g)*n_halo+h;
                    image_index[at]=image+v*stride;
                    if (projected_image) image_groups[at]=channel_group;
                  }
                }
                // The column generator already supplies the targets. Compose
                // their image addresses once instead of scanning the whole
                // halo, then decoding them again along every incoming edge.
                const auto begin=target_rows.begin()+groups[g].begin;
                const auto end=target_rows.begin()+groups[g].end;
                const auto before=[](const auto& a,const auto& b) { return a.alpha<b.alpha; };
                if (!std::is_sorted(begin,end,before)) std::sort(begin,end,before);
              });
              indices_phase.stop();
              profiling::ScopedPhase links_phase(profiling::Phase::CiTileLinks);
              for (const auto& group:groups)
                for (const auto& edge:beta_graph.singles(group.beta))
                  ++offsets[column_index[edge.source]+1];
              for (int i = 1; i <= n_columns; ++i) offsets[i] += offsets[i - 1];
              positions = offsets;
              incoming.resize(offsets.back());
              beta_offsets.resize(groups.size()*beta_graph.n_singles());
              beta_map_bytes=std::max(beta_map_bytes,beta_offsets.capacity()*sizeof(Eigen::Index));
              for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
                groups[g].column=column_index[groups[g].beta]*width;
                const auto& links = beta_graph.singles(groups[g].beta);
                for (int link = 0; link < static_cast<int>(links.size()); ++link) {
                  incoming[positions[column_index[links[link].source]]++] =
                      {g, links[link].pair, links[link].sign};
                  // The same beta edge serves every alpha target and RHS.
                  // Offsets follow the shared runtime budget, not a 32-bit
                  // assumption tied to the former 64 MiB halo.
                  beta_offsets[g*beta_graph.n_singles()+link]=static_cast<Eigen::Index>(
                      static_cast<std::size_t>(n_halo)*column_index[links[link].source]*width);
                }
              }
              links_phase.stop();
              profiling::record_matrix_shape(profiling::Phase::CiSigma, count * width, team);
              prepare_time -= batch_read_time;
            }
            compute_start = tick();
            prepare_time += seconds(prepare_start, compute_start);
          }
        } catch (...) { failure = std::current_exception(); }
      }
      if (failure || finished) break;

      Eigen::Map<const Eigen::MatrixXd> input(workspace.get(), n_halo,
          static_cast<Eigen::Index>(n_columns) * width);
      double* scattered=workspace.get()+scratch_size;
      double* state_channels=scattered+source_capacity;
      double* transformed=state_channels+(separate_output ? packed_size:0);
      // Scattered/state channels are ragged row-major until multiplication;
      // full transformed/forcing images use column-major image-by-channel rows.

      const int first = count * width * thread / team;
      const int last = count * width * (thread + 1) / team;
      const bool identity_halo=n_halo==n_alpha_;
      const auto scatter = [&](int row,double* values) {
        const int local = row % count, v = row / count;
        const int channel_group=channel_groups.empty() ? 0:channel_groups[local];
        double* target=values+channel_layout.packed(row,channel_group);
        std::fill_n(target,channel_product.channels(channel_group),0.0);
        const int a=active[local],g=group_index[local],beta=groups[g].beta;
        const int h=identity_halo ? a:halo_index[a];
        const double* column=input.col(groups[g].column+v).data();
        // A depth-one target outside S has no own coefficient, although its
        // one-body neighbors can still contribute to E*c and must be kept.
        if (active_support[local]) {
          const double value = column[h];
          for (const int p : alpha_->occupied(a))
            target[core::PairIndex::packed_pair_index(p, p)] += value;
          for (const int p : beta_graph.occupied(beta))
            target[core::PairIndex::packed_pair_index(p, p)] += value;
        }
        const auto& alpha_links = alpha_->singles(a);
        for (int link = 0; link < static_cast<int>(alpha_links.size()); ++link) {
          const int channel=channel_product.local(channel_group,alpha_links[link].pair);
          if (channel>=0) target[channel] += alpha_links[link].sign *
              column[identity_halo ? alpha_links[link].source:halo_index[alpha_links[link].source]];
        }
        const auto& beta_links = beta_graph.singles(beta);
        const double* source=input.data()+h+static_cast<std::size_t>(n_halo)*v;
        for (int link = 0; link < static_cast<int>(beta_links.size()); ++link) {
          const int channel=channel_product.local(channel_group,beta_links[link].pair);
          if (channel>=0) target[channel] += beta_links[link].sign *
              source[beta_offsets[g*beta_graph.n_singles()+link]];
        }
      };
      for (std::size_t request_id=0;request_id<requests.size();++request_id) {
        const auto& request=requests[request_id];
        const auto& add_columns=request.image;
        const auto& add_forcing=request.forcing;
        const auto* direction=request.direction;
        auto& kernel_adjoint=kernel_adjoints[request_id];
        if (request_id>0) {
#pragma omp single
          {
            try {
              const auto begin=tick();
              Eigen::Map<Eigen::MatrixXd> values(workspace.get(),n_halo,n_columns*width);
              request.response(columns,halo,values);
              read_time+=seconds(begin,tick());
              source_columns+=n_columns;
              source_elements+=panel_size;
              compute_start=tick();
            } catch (...) { failure=std::current_exception(); }
          }
          if (failure) break;
        }
        for (int row = first; row < last; ++row) scatter(row,scattered);
        if (collect_integrals) {
          if (state && request_id==0) {
            // The response scatter is complete. Refill the same input halo for
            // the unweighted state, then scatter it into the second channel slab.
#pragma omp barrier
#pragma omp single
            {
              try {
                const auto begin = tick();
                Eigen::Map<Eigen::MatrixXd> values(workspace.get(),n_halo,n_columns*width);
                state(columns,halo,values);
                const double elapsed=seconds(begin,tick());
                read_time += elapsed;
                compute_time -= elapsed;
                source_columns += n_columns;
                source_elements += panel_size;
              } catch (...) { failure = std::current_exception(); }
            }
            if (failure) break;
            for (int row = first; row < last; ++row) scatter(row,state_channels);
          }
          // An identical bra borrows A directly: no second scatter or copy.
          // A distinct B remains immutable across all temporal responses.
#pragma omp barrier
          const int available=std::min(team,count*width);
          const auto full_area=static_cast<std::size_t>(pairs)*pairs;
          const bool row_partition=full_area<=scratch_size/(2*available);
          const int parts=row_partition ? available:1;
          if (timing) {
#pragma omp single
            {
              gram_start=tick();
              gram_partitions=std::max(gram_partitions,parts);
              gram_extent=std::max(gram_extent,pairs);
              if (row_partition) gram_scratch_bytes=std::max(gram_scratch_bytes,
                  2*parts*full_area*sizeof(double));
            }
          }
          detail::contract_channel_gram(channel_product,channel_layout,scattered,
              state ? state_channels:scattered,kernel_adjoint,diagonal_rows,count,width,
              exchange_parity!=0,workspace.get(),scratch_size,thread,team);
          if (timing) {
#pragma omp single
            { gram_time+=seconds(gram_start,tick()); gram_tiles+=parts*channel_product.groups(); }
          }
        }
        // Rows were scattered in original order; grouped BLAS partitions
        // have different owners. Publish all source rows before multiplication.
#pragma omp barrier
        if (add_columns) detail::multiply_channels(channel_product,channel_layout,
            scattered,transformed,thread,team,false,diagonal_rows,count,width,exchange_parity!=0);
        if (add_forcing) {
          // Expanding packed A into a full forcing image may overlap rows
          // owned by other workers. Every base multiplication must finish.
#pragma omp barrier
          detail::multiply_channels(directional_products[request_id],channel_layout,
              state_channels,scattered,thread,team,false,diagonal_rows,count,width,exchange_parity!=0);
          if (add_columns) {
            // Both images use identical group row ownership.
            const Eigen::OuterStride<> image_stride(count*width);
            for (int group=0;group<channel_product.groups();++group) {
              const int size=channel_layout.size(group);
              const int begin=channel_layout.begin(group)+size*thread/team;
              const int end=channel_layout.begin(group)+size*(thread+1)/team;
              if (begin==end) continue;
              Eigen::Map<Eigen::MatrixXd,0,Eigen::OuterStride<>> output(
                  transformed+begin,end-begin,pairs,image_stride);
              Eigen::Map<const Eigen::MatrixXd,0,Eigen::OuterStride<>> forcing(
                  scattered+begin,end-begin,pairs,image_stride);
              output+=forcing;
            }
          }
        } else if (direction && add_columns) {
          detail::multiply_channels(directional_products[request_id],channel_layout,
              state_channels,transformed,thread,team,true,diagonal_rows,count,width,exchange_parity!=0);
        }
        // Publish grouped images before the single writer schedules tasks.
#pragma omp barrier

#pragma omp single
        {
          try {
            // Evaluate E^T K E directly in the caller's output panel. The callback
            // owns panel lifetimes and schedules disjoint beta/RHS/alpha ranges.
            const auto write_image = [&](const DirectCiTileWriter& writer,
                                         const double* channel_values) {
              const std::size_t image_rows=static_cast<std::size_t>(count)*width;
              const auto value=[&](int row,int channel_group,int pair) {
                const int channel=projected_image ? channel_product.local(channel_group,pair):pair;
                return channel<0 ? 0.0:channel_values[row+static_cast<std::size_t>(channel)*image_rows];
              };
              const DirectCiColumnAccumulator accumulate = [&,channel_values](int source, int v,
                int first,int last,double* destination,const int* rows,int n_rows) {
              if (first==last) return;
              int begin=first,end=last;
              if (rows) {
                begin=std::lower_bound(rows,rows+n_rows,halo[first])-rows;
                end=std::upper_bound(rows+begin,rows+n_rows,halo[last-1])-rows;
                if (begin==end) return;
              }
              const int same = same_group[source];
              if (same >= 0) {
                const int* index=image_index.data()+
                    (static_cast<std::size_t>(v)*groups.size()+same)*n_halo;
                const int* image_group=projected_image ? image_groups.data()+
                    (static_cast<std::size_t>(v)*groups.size()+same)*n_halo:nullptr;
                const auto beta_occupied=beta_graph.occupied(columns[source]);
                for (int i=begin;i<end;++i) {
                  const int a=rows ? rows[i]:halo[i];
                  const int local=rows ? (identity_halo ? a:halo_index[a]):i;
                  if (local<first || local>=last) continue;
                  double sum = 0;
                  if (index[local] >= 0) {
                    const int row=index[local];
                    for (const int p : alpha_->occupied(a))
                      sum += value(row,image_group ? image_group[local]:0,
                          core::PairIndex::packed_pair_index(p,p));
                    for (const int p : beta_occupied)
                      sum += value(row,image_group ? image_group[local]:0,
                          core::PairIndex::packed_pair_index(p,p));
                  }
                  const auto& links = alpha_->singles(a);
                  for (int link = 0; link < static_cast<int>(links.size()); ++link) {
                    const int h = identity_halo ? links[link].source:halo_index[links[link].source];
                    const int target = h < 0 ? -1 : index[h];
                    if (target >= 0)
                      sum += links[link].sign * value(target,image_group ? image_group[h]:0,
                          links[link].pair);
                  }
                  destination[rows ? i:a] += sum;
                }
              }
              // One beta edge fixes the channel, sign and target-row map for
              // the entire alpha range. Do not rediscover them for every pixel.
              for (int edge = offsets[source]; edge < offsets[source + 1]; ++edge) {
                const auto& link = incoming[edge];
                const double* values = projected_image ? channel_values:channel_values +
                    static_cast<std::size_t>(link.pair)*image_rows;
                const auto& group=groups[link.group];
                detail::gather_beta_rows(values,1,v,link.sign,
                    {target_rows.data()+group.begin,static_cast<std::size_t>(group.end-group.begin)},
                    halo[first],halo[last-1]+1,destination,rows,n_rows,
                    projected_image ? &channel_product:nullptr,link.pair,image_rows);
              }
            };
              writer(columns, halo, accumulate);
            };
            const auto write_start = tick();
            compute_time += seconds(compute_start, write_start);
            if (add_columns) write_image(add_columns,transformed);
            if (add_forcing) write_image(add_forcing,scattered);
            // Retire inside the final writer's existing single/barrier.
            // Testing the shared retirement vector outside this region races
            // the next plan.next(): one worker could enter a conditional
            // barrier after another has already advanced to the next tile.
            if (request_id+1==requests.size() && release && !released_columns.empty())
              release(released_columns);
            write_time += seconds(write_start, tick());
            if (timing && ((batches&(batches-1))==0 || batches%256==0))
              std::fprintf(stderr,"XMVB_TILE_PROGRESS,batches=%zu,targets=%zu,prepare=%.9f,read=%.9f,compute=%.9f,write=%.9f,slab_bytes=%zu,temporary_bytes=%zu\n",
                  batches,targets,prepare_time,read_time,compute_time,write_time,
                  workspace.bytes(),workspace::temporary_bytes());
          } catch (...) { failure = std::current_exception(); }
        }
        if (failure) break;
      } // temporal responses share the generated graph and accepted channels
      if (failure) break;
    }
  }
  if (failure) std::rethrow_exception(failure);
  if (timing)
    std::fprintf(stderr,
        "XMVB_TILE,batches=%zu,source_columns=%zu,source_elements=%zu,peak_payload=%zu,"
        "prepare=%.9f,read=%.9f,compute=%.9f,write=%.9f,targets=%zu,exchange=%d,"
        "skipped_diagonal_scatters=%zu,gram=%.9f,gram_tiles=%zu,"
        "identity_halo_targets=%zu,beta_map_bytes=%zu,gram_partitions=%d,"
        "gram_extent=%d,gram_scratch_bytes=%zu,workspace_reused=%d,"
        "workspace_allocations=%zu,workspace_bytes=%zu,"
        "restricted_channels=%d,source_channels=%zu,channel_kernel_bytes=%zu,"
        "projected_image=%d,channel_products=%.0Lf,unrestricted_channel_products=%.0Lf\n",
        batches, source_columns, source_elements, peak_payload,
        prepare_time, read_time, compute_time, write_time, targets, exchange_parity,
        unsupported_targets * static_cast<std::size_t>(width) *
            (requests.size() + static_cast<std::size_t>(collect_integrals && static_cast<bool>(state))),
        gram_time,gram_tiles,identity_halo_targets,beta_map_bytes,
        gram_partitions,gram_extent,gram_scratch_bytes,
        static_cast<int>(workspace.reused()),workspace.allocations(),workspace.bytes(),
        static_cast<int>(restricted_channels),source_channels,restricted_channels ? kernel_bytes:0,
        static_cast<int>(projected_image),projected_products,full_products);
  std::vector<DirectCiIntegralAdjoint> results(requests.size());
  if (collect_integrals) for (std::size_t r=0;r<requests.size();++r)
    results[r]=pullback_channel_kernel(kernel_adjoints[r]);
  return results;
}

} // namespace xmvb::mcscf
