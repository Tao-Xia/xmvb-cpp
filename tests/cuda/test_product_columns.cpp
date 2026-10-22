#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "cuda/vb/columns.hpp"
#include "cuda/memory/upload.hpp"

namespace {
using Matrix=Eigen::MatrixXd;

void near(const Matrix& actual,const Matrix& expected,const char* message) {
  if (actual.rows()!=expected.rows() || actual.cols()!=expected.cols() ||
      !actual.allFinite() || (actual-expected).norm()>2e-12)
    throw std::runtime_error(message);
}

std::shared_ptr<const xmvb::cuda::ProductColumnPattern> pattern(
    xmvb::cuda::Context& context,std::vector<int> rows) {
  auto result=std::make_shared<xmvb::cuda::ProductColumnPattern>();
  result->size=rows.size();result->rows=std::move(rows);
  result->device_rows=xmvb::cuda::Buffer<int>(context,result->size);
  context.copy_to_device(result->device_rows.data(),result->rows.data(),
      result->size*sizeof(int));
  return result;
}

void check(int parity) {
  xmvb::cuda::Context context(16ULL*1024*1024);
  // An exact transpose-closed support with several absent coordinates.
  const std::array<std::vector<int>,4> rows{{
      {0,1,3},{0,1,2},{1,2,3},{0,2,3}}};
  std::vector<std::shared_ptr<const xmvb::cuda::ProductColumnPattern>> patterns;
  for (const auto& selected:rows) patterns.push_back(pattern(context,selected));
  const auto before=context.counters().current_device_bytes;
  std::vector<xmvb::cuda::ProductColumns> blocks;
  blocks.emplace_back(context,
      std::vector{patterns[0],patterns[1]},2);
  blocks.emplace_back(context,
      std::vector{patterns[2],patterns[3]},2);
  if (blocks[0].bytes()+blocks[1].bytes()!=
      context.counters().current_device_bytes-before)
    throw std::runtime_error("product panels double-charge shared index storage");
  std::array<Matrix,2> original{Matrix::Zero(4,4),Matrix::Zero(4,4)};
  for (int p=0;p<2;++p) {
    Matrix packed(blocks[p].values.rows(),2);
    for (int local=0;local<2;++local) {
      const int b=2*p+local;
      for (int i=0;i<rows[b].size();++i) {
        const int a=rows[b][i];
        for (int lane=0;lane<2;++lane) {
          const double value=std::sin(0.17*(a+1)*(b+2)+0.39*lane);
          packed(blocks[p].offsets[local]+i,lane)=value;
          original[lane](a,b)=value;
        }
      }
    }
    xmvb::cuda::copy_to_device(context,blocks[p].values.view(),packed);
  }
  const std::array<xmvb::cuda::ProductColumns*,2> mutable_panels{
      &blocks[0],&blocks[1]};
  xmvb::cuda::project_product_exchange(context,mutable_panels,2,parity);
  const std::array<const xmvb::cuda::ProductColumns*,2> panels{
      &blocks[0],&blocks[1]};
  const std::array<int,4> all{0,1,2,3};
  for (int lane=0;lane<2;++lane) {
    Matrix expected=Matrix::Zero(4,4);
    for (int b=0;b<4;++b) for (int a:rows[b]) if (a<=b) {
      const double value=0.5*(original[lane](a,b)+parity*original[lane](b,a));
      expected(a,b)=value;expected(b,a)=parity*value;
    }
    xmvb::cuda::Matrix<double> dense(context,4,4);
    xmvb::cuda::read_product_columns(context,panels,2,lane,all,all,dense.view());
    Matrix actual(4,4);
    xmvb::cuda::copy_to_host(context,actual,dense.view());
    near(actual,expected,"compact exchange projection differs from CPU rule");

    const std::array<int,3> requested_betas{3,1,0};
    const std::array<int,3> requested_rows{3,0,2};
    xmvb::cuda::Matrix<double> selected(context,3,3);
    xmvb::cuda::read_product_columns(context,panels,2,lane,
        requested_betas,requested_rows,selected.view());
    Matrix selected_expected(3,3);
    for (int j=0;j<3;++j) for (int i=0;i<3;++i)
      selected_expected(i,j)=expected(requested_rows[i],requested_betas[j]);
    Matrix selected_actual(3,3);
    xmvb::cuda::copy_to_host(context,selected_actual,selected.view());
    near(selected_actual,selected_expected,"compact reader missed supported or zero rows");
  }
}

void check_reject_asymmetric() {
  xmvb::cuda::Context context(16ULL*1024*1024);
  const std::array<std::vector<int>,4> rows{{
      {0,1,3},{1,2},{1,2,3},{0,2,3}}};
  std::vector<std::shared_ptr<const xmvb::cuda::ProductColumnPattern>> patterns;
  for (const auto& selected:rows) patterns.push_back(pattern(context,selected));
  xmvb::cuda::ProductColumns first(context,
      std::vector{patterns[0],patterns[1]},1);
  xmvb::cuda::ProductColumns second(context,
      std::vector{patterns[2],patterns[3]},1);
  const std::array<xmvb::cuda::ProductColumns*,2> panels{&first,&second};
  bool rejected=false;
  try { xmvb::cuda::project_product_exchange(context,panels,2,1); }
  catch (const std::invalid_argument&) { rejected=true; }
  if (!rejected) throw std::runtime_error("asymmetric exact support was accepted");
}

void check_read_cache() {
  xmvb::cuda::Context context(16ULL*1024*1024);
  const std::array<std::vector<int>,4> rows{{
      {0,2},{1,3},{0,1},{2,3}}};
  std::vector<std::shared_ptr<const xmvb::cuda::ProductColumnPattern>> patterns;
  for (const auto& selected:rows) patterns.push_back(pattern(context,selected));
  std::vector<xmvb::cuda::ProductColumns> blocks;
  blocks.emplace_back(context,std::vector{patterns[0],patterns[1]},2);
  blocks.emplace_back(context,std::vector{patterns[2],patterns[3]},2);
  std::array<Matrix,2> original{Matrix::Zero(4,4),Matrix::Zero(4,4)};
  for (int panel=0;panel<2;++panel) {
    Matrix packed(blocks[panel].values.rows(),2);
    for (int local=0;local<2;++local) {
      const int b=2*panel+local;
      for (int i=0;i<static_cast<int>(rows[b].size());++i)
        for (int lane=0;lane<2;++lane) {
          const int a=rows[b][i];
          const double value=10.0*lane+b+0.1*a;
          packed(blocks[panel].offsets[local]+i,lane)=value;
          original[lane](a,b)=value;
        }
    }
    xmvb::cuda::copy_to_device(context,blocks[panel].values.view(),packed);
  }
  const std::array<const xmvb::cuda::ProductColumns*,2> panels{&blocks[0],&blocks[1]};
  const std::array<int,3> betas{3,1,0},selected_rows{3,0,2};
  xmvb::cuda::Matrix<double> device_result(context,3,3);
  const auto expected=[&](int lane) {
    Matrix result(3,3);
    for (int j=0;j<3;++j) for (int i=0;i<3;++i)
      result(i,j)=original[lane](selected_rows[i],betas[j]);
    return result;
  };
  const auto verify=[&](const char* message,int lane) {
    Matrix actual(3,3);
    xmvb::cuda::copy_to_host(context,actual,device_result.view());
    near(actual,expected(lane),message);
  };
  xmvb::cuda::ProductColumnReadCache cache(context,panels,2,4096);
  const auto before=context.counters().host_to_device_bytes;
  cache.read(0,betas,selected_rows,device_result.view());
  const auto first=context.counters().host_to_device_bytes;
  if (first<=before || cache.bytes()==0 || cache.peak_bytes()!=cache.bytes())
    throw std::runtime_error("compact read cache did not retain first metadata plan");
  verify("compact cached first lane differs from source",0);
  cache.read(1,betas,selected_rows,device_result.view());
  if (context.counters().host_to_device_bytes!=first)
    throw std::runtime_error("compact read cache uploaded duplicate second-lane metadata");
  verify("compact cached second lane differs from source",1);
  cache.read(0,betas,selected_rows,device_result.view());
  if (context.counters().host_to_device_bytes!=first)
    throw std::runtime_error("compact read cache missed repeated beta/row key");
  verify("compact cached repeated key differs from source",0);

  // Allocate the replacement before releasing the original so the borrowed
  // numerical pointer must change; the cached row plan remains the same.
  xmvb::cuda::Matrix<double> replacement(context,blocks[0].values.rows(),2);
  Matrix packed(replacement.rows(),replacement.cols());
  for (int local=0;local<2;++local) {
    const int b=local;
    for (int i=0;i<static_cast<int>(rows[b].size());++i)
      for (int lane=0;lane<2;++lane) {
        const int a=rows[b][i];
        const double value=100.0+10.0*lane+b+0.1*a;
        packed(blocks[0].offsets[local]+i,lane)=value;
        original[lane](a,b)=value;
      }
  }
  xmvb::cuda::copy_to_device(context,replacement.view(),packed);
  blocks[0].values=std::move(replacement);
  const auto before_rebind=context.counters().host_to_device_bytes;
  cache.read(0,betas,selected_rows,device_result.view());
  if (context.counters().host_to_device_bytes<=before_rebind)
    throw std::runtime_error("compact read cache kept stale replaced-panel descriptor");
  verify("compact cached replacement differs from source",0);
  const auto after_rebind=context.counters().host_to_device_bytes;
  cache.read(0,betas,selected_rows,device_result.view());
  if (context.counters().host_to_device_bytes!=after_rebind)
    throw std::runtime_error("compact read cache repeatedly uploaded unchanged replacement");

  xmvb::cuda::ProductColumnReadCache streamed(context,panels,2,0);
  const auto before_stream=context.counters().host_to_device_bytes;
  streamed.read(1,betas,selected_rows,device_result.view());
  if (streamed.bytes()!=0 || streamed.peak_bytes()!=0 ||
      context.counters().host_to_device_bytes<=before_stream)
    throw std::runtime_error("over-budget compact read retained metadata");
  verify("over-budget compact read differs from streamed source",1);
}

void check_pinned_upload() {
  using xmvb::cuda::Buffer;
  using xmvb::cuda::Context;
  using xmvb::cuda::PinnedUpload;
  Context context(160);
  constexpr std::size_t small=32,large=48;
  Buffer<std::byte> first(context,small),second(context,small),third(context,large);
  const auto resident=context.counters().current_device_bytes;
  const auto copied=context.counters().host_to_device_bytes;
  if (resident!=2*small+large)
    throw std::runtime_error("pinned upload test baseline mismatch");

  std::array<std::byte,small> a{},b{};
  std::array<std::byte,large> c{};
  for (std::size_t i=0;i<small;++i) {
    a[i]=std::byte{static_cast<unsigned char>(i+1)};
    b[i]=std::byte{static_cast<unsigned char>(200-i)};
  }
  for (std::size_t i=0;i<large;++i)
    c[i]=std::byte{static_cast<unsigned char>(i+71)};
  const auto expected_a=a;

  {
    PinnedUpload upload(context);
    upload.assign(std::span<const std::byte>{a});
    if (upload.size()!=small || upload.data()==nullptr ||
        context.counters().host_to_device_bytes!=copied+small)
      throw std::runtime_error("first pinned upload metadata mismatch");
    auto* reused=upload.data();
    xmvb::cuda::check_cuda(cudaMemcpyAsync(first.data(),upload.data(),small,
        cudaMemcpyDeviceToDevice,context.stream()),"first pinned upload consumer");
    a.fill(std::byte{0});  // caller storage may change once assign returns

    upload.assign(small,[&](std::span<std::byte> slot) {
      std::copy(b.begin(),b.end(),slot.begin());
    });
    if (upload.data()!=reused || upload.size()!=small ||
        context.counters().host_to_device_bytes!=copied+2*small)
      throw std::runtime_error("same-size pinned upload did not reuse tile");
    xmvb::cuda::check_cuda(cudaMemcpyAsync(second.data(),upload.data(),small,
        cudaMemcpyDeviceToDevice,context.stream()),"second pinned upload consumer");

    upload.assign(std::span<const std::byte>{c});
    if (upload.size()!=large || upload.data()==nullptr ||
        context.counters().current_device_bytes!=resident+large ||
        context.counters().peak_device_bytes!=resident+large ||
        context.counters().host_to_device_bytes!=copied+2*small+large)
      throw std::runtime_error("resized pinned upload accounting mismatch");
    xmvb::cuda::check_cuda(cudaMemcpyAsync(third.data(),upload.data(),large,
        cudaMemcpyDeviceToDevice,context.stream()),"third pinned upload consumer");

    upload.assign(std::span<const std::byte>{});
    if (upload.size()!=0 || upload.data()!=nullptr ||
        context.counters().current_device_bytes!=resident ||
        context.counters().host_to_device_bytes!=copied+2*small+large)
      throw std::runtime_error("empty pinned upload did not release tile");
  }
  if (context.counters().current_device_bytes!=resident)
    throw std::runtime_error("pinned upload destructor retained device tile");

  // Destruction with an outstanding upload must keep its pinned source alive.
  {
    PinnedUpload upload(context);
    upload.assign(std::span<const std::byte>{c});
    xmvb::cuda::check_cuda(cudaMemcpyAsync(third.data(),upload.data(),large,
        cudaMemcpyDeviceToDevice,context.stream()),"destructor upload consumer");
  }
  if (context.counters().current_device_bytes!=resident ||
      context.counters().host_to_device_bytes!=copied+2*small+2*large ||
      context.counters().peak_device_bytes!=resident+large)
    throw std::runtime_error("pinned upload destructor accounting mismatch");

  std::array<std::byte,small> observed_a{},observed_b{};
  std::array<std::byte,large> observed_c{};
  context.copy_to_host(observed_a.data(),first.data(),small);
  context.copy_to_host(observed_b.data(),second.data(),small);
  context.copy_to_host(observed_c.data(),third.data(),large);
  if (observed_a!=expected_a || observed_b!=b || observed_c!=c)
    throw std::runtime_error("pinned upload overwrote a queued consumer");
}
} // namespace

int main() {
  try {
    check(1);check(-1);check_reject_asymmetric();check_read_cache();
    check_pinned_upload();
    std::cout<<"compact product column tests passed\n";
  } catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';return 1;
  }
}
