#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "cuda/vb/columns.hpp"

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
} // namespace

int main() {
  try {
    check(1);check(-1);check_reject_asymmetric();
    std::cout<<"compact product column tests passed\n";
  } catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';return 1;
  }
}
