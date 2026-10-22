#include "cuda/vb/columns.hpp"

#include <algorithm>
#include <stdexcept>

namespace xmvb::cuda {
namespace {

struct DeviceProductPanel {
  const ProductColumnMap* maps;
  linalg::MatrixView values;
};

__global__ void project_exchange_columns(const DeviceProductPanel* panels,
    int extent,int n_beta,int parity) {
  const int b=blockIdx.x;
  const auto panel=panels[b/extent];
  if (!panel.maps) return;
  const auto column=panel.maps[b%extent];
  for (int i=blockIdx.y*blockDim.x+threadIdx.x;i<column.size;
       i+=gridDim.y*blockDim.x) {
    const int a=column.rows ? column.rows[i]:i;
    if (a>b || a>=n_beta) continue;
    const auto other=panels[a/extent];
    const auto transposed=other.maps[a%extent];
    int j=b;
    if (transposed.rows) {
      int lo=0,hi=transposed.size;
      while (lo<hi) {
        const int mid=lo+(hi-lo)/2;
        if (transposed.rows[mid]<b) lo=mid+1;else hi=mid;
      }
      j=lo;
    }
    for (int lane=0;lane<panel.values.cols;++lane) {
      double* const x=panel.values.data+column.offset+i+panel.values.ld*lane;
      double* const y=other.values.data+transposed.offset+j+other.values.ld*lane;
      const double value=0.5*(*x+parity*(*y));
      *x=value;*y=parity*value;
    }
  }
}

struct DeviceColumnRead {
  const int* rows;
  const double* values;
  int size;
};

__global__ void gather_product_columns(const DeviceColumnRead* columns,
    const int* requested,int count,linalg::MatrixView output) {
  const int col=blockIdx.x;
  const auto source=columns[col];
  for (int r=blockIdx.y*blockDim.x+threadIdx.x;r<count;
       r+=gridDim.y*blockDim.x) {
    const int a=requested[r];
    double value=0.0;
    if (source.values) {
      if (!source.rows) {
        if (a>=0 && a<source.size) value=source.values[a];
      } else {
        int lo=0,hi=source.size;
        while (lo<hi) {
          const int mid=lo+(hi-lo)/2;
          if (source.rows[mid]<a) lo=mid+1;else hi=mid;
        }
        if (lo<source.size && source.rows[lo]==a) value=source.values[lo];
      }
    }
    output.data[r+output.ld*col]=value;
  }
}

__global__ void pack_columns(const ProductColumnMap* maps,
    linalg::MatrixView full,linalg::MatrixView compact,bool unpack) {
  const int b=blockIdx.x;
  const auto column=maps[b];
  for (auto i=static_cast<std::size_t>(blockIdx.y)*blockDim.x+threadIdx.x;
       i<static_cast<std::size_t>(column.size)*compact.cols;
       i+=static_cast<std::size_t>(gridDim.y)*blockDim.x) {
    const int r=i%column.size,v=i/column.size,a=column.rows ? column.rows[r]:r;
    auto& destination=unpack ? full.data[a+full.ld*(v*gridDim.x+b)]:
        compact.data[column.offset+r+compact.ld*v];
    destination=unpack ? compact.data[column.offset+r+compact.ld*v]:
        full.data[a+full.ld*(v*gridDim.x+b)];
  }
}

} // namespace

void project_product_exchange(Context& context,
    std::span<ProductColumns* const> panels,int extent,int parity) {
  if (panels.empty() || !panels.back() || extent<=0 ||
      panels.back()->patterns.empty() ||
      panels.back()->patterns.size()>static_cast<std::size_t>(extent) ||
      (parity!=1 && parity!=-1))
    throw std::invalid_argument("invalid compact exchange panels");
  const auto n_beta=static_cast<int>((panels.size()-1)*extent+
      panels.back()->patterns.size());
  const int width=static_cast<int>(panels.back()->values.cols());
  if (width<=0) throw std::invalid_argument("empty compact exchange lanes");
  std::vector<DeviceProductPanel> descriptors(panels.size());
  int max_rows=0;
  for (std::size_t p=0;p<panels.size();++p) {
    auto* panel=panels[p];
    if (!panel) continue;
    const int columns=static_cast<int>(panel->patterns.size());
    if (columns<=0 || columns>extent ||
        (p+1<panels.size() && columns!=extent) ||
        panel->values.cols()!=width || panel->offsets.size()!=columns+1 ||
        panel->maps.size()!=static_cast<std::size_t>(columns) ||
        panel->values.rows()!=panel->offsets.back())
      throw std::invalid_argument("incompatible compact exchange panel");
    descriptors[p]={panel->maps.data(),panel->values.view()};
    for (int local=0;local<columns;++local) {
      const auto& pattern=*panel->patterns[local];
      if (pattern.size!=panel->offsets[local+1]-panel->offsets[local] ||
          (pattern.full && pattern.size!=n_beta) ||
          (!pattern.full && (pattern.rows.size()!=static_cast<std::size_t>(pattern.size) ||
              !std::is_sorted(pattern.rows.begin(),pattern.rows.end()) ||
              std::adjacent_find(pattern.rows.begin(),pattern.rows.end())!=pattern.rows.end())))
        throw std::invalid_argument("invalid exact exchange column support");
      max_rows=std::max(max_rows,pattern.size);
      const int b=static_cast<int>(p)*extent+local;
      for (int i=0;i<pattern.size;++i) {
        const int a=pattern.full ? i:pattern.rows[i];
        if (a<0 || a>=n_beta || static_cast<std::size_t>(a/extent)>=panels.size() ||
            !panels[a/extent])
          throw std::invalid_argument("exchange support lacks transposed panel");
        const auto& other=*panels[a/extent];
        if (a%extent>=other.patterns.size())
          throw std::invalid_argument("exchange support lacks transposed column");
        const auto& partner=*other.patterns[a%extent];
        if (!(partner.full ? b<partner.size:
              std::binary_search(partner.rows.begin(),partner.rows.end(),b)))
          throw std::invalid_argument("exchange support is not transpose closed");
      }
    }
  }
  if (!max_rows) return;
  Buffer<DeviceProductPanel> device(context,descriptors.size());
  context.copy_to_device(device.data(),descriptors.data(),
      descriptors.size()*sizeof(DeviceProductPanel));
  const auto blocks=std::min<std::size_t>(65535,
      (static_cast<std::size_t>(max_rows)+255)/256);
  project_exchange_columns<<<dim3(n_beta,blocks),256,0,context.stream()>>>(
      device.data(),extent,n_beta,parity);
  check_cuda(cudaGetLastError(),"project compact exchange columns");
}

void read_product_columns(Context& context,
    std::span<const ProductColumns* const> panels,int extent,int lane,
    std::span<const int> betas,std::span<const int> rows,
    linalg::MatrixView destination) {
  if (panels.empty() || extent<=0 || lane<0 ||
      destination.rows!=rows.size() || destination.cols!=betas.size() ||
      destination.ld<destination.rows ||
      (!destination.data && destination.rows*destination.cols))
    throw std::invalid_argument("invalid compact column read");
  if (betas.empty() || rows.empty()) return;
  std::vector<DeviceColumnRead> sources(betas.size());
  for (std::size_t j=0;j<betas.size();++j) {
    const int b=betas[j];
    if (b<0 || static_cast<std::size_t>(b/extent)>=panels.size())
      throw std::out_of_range("compact beta column read");
    const auto* panel=panels[b/extent];
    if (!panel) continue;
    const int local=b%extent;
    if (local>=panel->patterns.size() || lane>=panel->values.cols())
      throw std::out_of_range("compact product lane or column read");
    const auto& pattern=*panel->patterns[local];
    sources[j]={pattern.full ? nullptr:pattern.device_rows.data(),
        panel->values.data()+panel->offsets[local]+panel->values.rows()*lane,
        pattern.size};
  }
  Buffer<DeviceColumnRead> device_sources(context,sources.size());
  Buffer<int> device_rows(context,rows.size());
  context.copy_to_device(device_sources.data(),sources.data(),
      sources.size()*sizeof(DeviceColumnRead));
  context.copy_to_device(device_rows.data(),rows.data(),rows.size_bytes());
  const auto blocks=std::min<std::size_t>(65535,(rows.size()+255)/256);
  gather_product_columns<<<dim3(betas.size(),blocks),256,0,context.stream()>>>(
      device_sources.data(),device_rows.data(),rows.size(),destination);
  check_cuda(cudaGetLastError(),"read compact product columns");
}

ProductColumns::ProductColumns(Context& context,
    std::vector<std::shared_ptr<const ProductColumnPattern>> columns,int width)
    :patterns(std::move(columns)),offsets{0} {
  if (width<=0 || patterns.empty())
    throw std::invalid_argument("invalid compact product columns");
  std::vector<ProductColumnMap> descriptors;
  descriptors.reserve(patterns.size());
  for (const auto& pattern:patterns) {
    descriptors.push_back({pattern->full ? nullptr:pattern->device_rows.data(),
        pattern->size,offsets.back()});
    offsets.push_back(offsets.back()+pattern->size);
    max_rows=std::max(max_rows,pattern->size);
  }
  maps=Buffer<ProductColumnMap>(context,descriptors.size());
  context.copy_to_device(maps.data(),descriptors.data(),
      descriptors.size()*sizeof(ProductColumnMap));
  values=Matrix<double>(context,offsets.back(),width);
}

std::size_t ProductColumns::bytes() const noexcept {
  // Numerical owners share immutable patterns; charging those rows again in
  // input, output and accepted LRUs can evict all three without freeing any
  // device storage. The pattern cache prices the index owner, and Context
  // accounts its actual lifetime even when only a panel retains it. Offsets
  // are host metadata, not device workspace.
  return values.rows()*values.cols()*sizeof(double)+
      maps.size()*sizeof(ProductColumnMap);
}

linalg::MatrixView ProductColumns::column(int beta) {
  return {values.data() ? values.data()+offsets.at(beta):nullptr,
      patterns.at(beta)->size,values.cols(),values.rows()};
}

void ProductColumns::pack(Context& context,linalg::MatrixView dense,bool unpack) {
  if (dense.cols!=static_cast<std::int64_t>(patterns.size())*values.cols() ||
      dense.ld<dense.rows || dense.rows<max_rows)
    throw std::invalid_argument("invalid compact product transform panel");
  if (!max_rows || !values.cols()) return;
  const auto blocks=std::min<std::size_t>(65535,
      (static_cast<std::size_t>(max_rows)*values.cols()+255)/256);
  pack_columns<<<dim3(patterns.size(),blocks),256,0,context.stream()>>>(
      maps.data(),dense,values.view(),unpack);
  check_cuda(cudaGetLastError(),"pack exact product columns");
}

void ProductImages::begin(int panels) {
  if (panels<=0) throw std::invalid_argument("invalid accepted product panel count");
  panels_.clear();panels_.resize(panels);
  present_.assign(panels,false);
  bytes_=0;retained_=0;complete_=false;enabled_=budget_>0;
}

bool ProductImages::retain(int panel,ProductColumns columns) {
  if (panel<0 || panel>=static_cast<int>(panels_.size()))
    throw std::out_of_range("accepted product panel index");
  if (complete_ || present_[panel])
    throw std::logic_error("accepted product panel was already written");
  if (!enabled_) return false;
  const auto bytes=columns.bytes();
  if (bytes>budget_ || bytes_>budget_-bytes) {
    // Exactly as CPU ProductImages: do not leave a partially valid endpoint
    // that would mix cached and regenerated accepted residuals.
    for (auto& entry:panels_) entry=ProductColumns{};
    std::fill(present_.begin(),present_.end(),false);
    bytes_=0;retained_=0;enabled_=false;
    return false;
  }
  panels_[panel]=std::move(columns);present_[panel]=true;
  bytes_+=bytes;++retained_;
  return true;
}

void ProductImages::finish() {
  complete_=enabled_ && retained_==static_cast<int>(panels_.size());
  if (enabled_ && !complete_)
    throw std::logic_error("accepted product endpoint is incomplete");
}

const ProductColumns& ProductImages::at(int panel) const {
  if (!complete_) throw std::logic_error("accepted product endpoint is not retained");
  return panels_.at(panel);
}

ProductColumns& ProductImages::at(int panel) {
  if (!complete_) throw std::logic_error("accepted product endpoint is not retained");
  return panels_.at(panel);
}

} // namespace xmvb::cuda
