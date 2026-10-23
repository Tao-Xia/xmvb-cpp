#include "core/workspace.hpp"

#include <Eigen/Core>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition) {
  if (!condition) throw std::runtime_error("shared workspace accounting failed");
}
}

int main() {
  using namespace xmvb::workspace;
  const auto original=limit();
  set_limit(1024);
  {
    Reservation first;
    require(first.try_resize(256) && available()==768);
    require(!first.try_resize(1025) && retained()==256);
    Reservation second=std::move(first);
    require(first.bytes()==0 && second.bytes()==256);
    require(second.try_resize(128) && available()==896);
    auto matrix=make_matrix<Eigen::MatrixXd>(8,8,128);
    require(matrix && retained()==640 && available()==384);
    auto alias=matrix;
    matrix.reset();
    require(retained()==640);
    require(!make_matrix<Eigen::MatrixXd>(8,8));
    alias.reset();
    require(retained()==128);
    set_limit(64);
    require(available()==0 && !second.try_resize(256));
    require(second.try_resize(0) && available()==64);
  }
  require(retained()==0);
  const auto initial_temporary=temporary_bytes();
  reset_temporary_peak();
  {
    Temporary first(128);
    require(temporary_bytes()==initial_temporary+128);
    require(available()==64); // Observation does not spend a second quota.
    Temporary second=std::move(first);
    require(first.bytes()==0 && second.bytes()==128);
    require(temporary_bytes()==initial_temporary+128);
    Temporary pending(32);
    pending=std::move(second);
    require(second.bytes()==0 && pending.bytes()==128);
    require(temporary_bytes()==initial_temporary+128);
    pending.resize(512);
    require(temporary_bytes()==initial_temporary+512);
    pending.resize(64);
    require(temporary_bytes()==initial_temporary+64);
    require(peak_temporary_bytes()==initial_temporary+512);
    reset_temporary_peak();
    require(peak_temporary_bytes()==temporary_bytes());
  }
  require(temporary_bytes()==initial_temporary);
  set_limit(original);
  std::cout<<"shared_workspace=PASS\n";
}
