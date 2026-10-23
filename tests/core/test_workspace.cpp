#include "core/workspace.hpp"
#include "core/slab.hpp"

#include <Eigen/Core>
#include <iostream>
#include <stdexcept>

namespace {
xmvb::workspace::Reservation* idle=nullptr;
void release_idle() {
  xmvb::workspace::set_reusable(0,nullptr);
  idle->try_resize(0);
}
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
  set_limit(512);
  {
    Reservation reusable,persistent;
    require(reusable.try_resize(128));
    {
      Scope action(512,reusable);
      require(action.bytes()==512 && reusable.bytes()==0);
      require(retained()==512 && available()==512);
      require(persistent.try_resize(64) && available()==448 && retained()==512);
      Reservation unused;
      {
        Scope nested(256,unused);
        require(nested.bytes()==256 && retained()==512);
        require(persistent.try_resize(32) && available()==480);
      }
      require(!persistent.try_resize(513));
      require(reusable.try_resize(128) && available()==352);
    }
    require(persistent.bytes()==32 && reusable.bytes()==128);
    require(retained()==160 && available()==352);
  }
  require(retained()==0 && available()==512);
  {
    Reservation cached,needed;
    require(cached.try_resize(256));
    idle=&cached;
    set_reusable(256,release_idle);
    require(retained()==256 && available()==512);
    require(needed.try_resize(512) && cached.bytes()==0);
    require(retained()==512 && available()==0);
    require(!needed.try_resize(513));
    idle=nullptr;
  }
  require(retained()==0 && available()==512);
  {
    Slab slab;
    require(slab.resize(32*1024,128*1024));
    auto* pointer=slab.get();
    for (int i=0;i<4096;++i) pointer[i]=i+0.5;
    require(!slab.resize(64*1024,128*1024) && slab.get()==pointer);
    require(!slab.resize(16*1024,128*1024));
    require(!slab.resize(32*1024,128*1024));
    for (int i=0;i<2048;++i) require(pointer[i]==i+0.5);
    try {
      slab.resize(std::numeric_limits<std::size_t>::max(),
          std::numeric_limits<std::size_t>::max());
      require(false);
    } catch (const std::bad_alloc&) {}
    require(slab.get()==pointer && pointer[0]==0.5);
    slab.reset();
    require(!slab.resize(64*1024,128*1024) && slab.get()==pointer);
  }
  set_limit(original);
  std::cout<<"shared_workspace=PASS\n";
}
