#include "cuda/vb/columns.hpp"

#include <algorithm>
#include <stdexcept>

namespace xmvb::cuda {
namespace {

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
  // Charge shared metadata conservatively, as CPU ColumnTile does. In
  // particular, retained accepted panels must account for their row indices
  // after the temporary pattern cache has released its own byte charge.
  std::size_t bytes=values.rows()*values.cols()*sizeof(double)+
      maps.size()*sizeof(ProductColumnMap)+offsets.size()*sizeof(std::int64_t);
  for (const auto& pattern:patterns)
    bytes+=pattern->device_rows.size()*sizeof(int);
  return bytes;
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

} // namespace xmvb::cuda
