#include "core/memory.hpp"
#include "core/workspace.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

namespace {
void require(bool condition) {
  if (!condition) throw std::runtime_error("CPU memory resource test failed");
}
void write(const std::filesystem::path& path, const std::string& text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream(path)<<text;
}
struct Fixture {
  std::filesystem::path root=std::filesystem::temp_directory_path()/
      ("xmvb-memory-"+std::to_string(getpid()));
  Fixture() { std::filesystem::remove_all(root); }
  ~Fixture() { std::filesystem::remove_all(root); }
};
}

int main() {
  using namespace xmvb::memory;
  constexpr std::size_t gib=1024ULL*1024*1024;
  Snapshot snapshot{2*gib, {{"host",100*gib,0},{"job",16*gib,3*gib},
      {"parent",20*gib,10*gib}}};
  auto budget=plan(snapshot,2*gib);
  require(budget.source=="parent" && budget.bytes==8*gib);
  require(plan(snapshot,2*gib,4*gib).bytes==4*gib);
  require(plan(snapshot,2*gib,40*gib).bytes==8*gib);
  snapshot.constraints.back().used=21*gib;
  bool exhausted=false;
  try { (void)plan(snapshot); } catch (const std::runtime_error&) { exhausted=true; }
  require(exhausted);
  require(plan({},0,3*gib).bytes==3*gib);
  bool missing=false;
  try { (void)plan({}); } catch (const std::runtime_error&) { missing=true; }
  require(missing);

  Fixture fixture;
  const auto proc=fixture.root/"proc", cg=fixture.root/"cgroup";
  write(proc/"meminfo","MemTotal: 104857600 kB\nMemAvailable: 83886080 kB\n");
  write(proc/"self/status","Name: test\nVmRSS: 2097152 kB\n");
  write(proc/"self/cgroup","0::/jobs/job42/step0\n");
  // The mounted hierarchy starts at /jobs, not the global cgroup root.
  write(proc/"self/mountinfo","1 0 0:1 /jobs "+cg.string()+" rw - cgroup2 cgroup rw\n");
  write(cg/"memory.max","max\n"); write(cg/"memory.current","0\n");
  write(cg/"job42/memory.max",std::to_string(16*gib));
  write(cg/"job42/memory.current",std::to_string(6*gib));
  write(cg/"job42/step0/memory.max","max\n");
  write(cg/"job42/step0/memory.current",std::to_string(3*gib));
  const auto measured=inspect(proc);
  require(measured.rss==2*gib);
  require(measured.constraints.size()==2);
  require(plan(measured,gib).bytes==9*gib);
  // A finite ancestor must still constrain an unlimited leaf and mount root.
  write(cg/"memory.max",std::to_string(32*gib));
  write(cg/"memory.current",std::to_string(30*gib));
  require(plan(inspect(proc)).bytes==2*gib);
  write(cg/"memory.max","0");
  exhausted=false;
  try { (void)plan(inspect(proc)); } catch (const std::runtime_error&) { exhausted=true; }
  require(exhausted);
  // v1 controller lists and unlimited sentinels.
  write(proc/"self/cgroup","4:cpu,memory:/jobs/job42\n");
  write(proc/"self/mountinfo","1 0 0:1 /jobs "+cg.string()+" rw - cgroup cgroup rw,memory\n");
  write(cg/"memory.limit_in_bytes","9223372036854771712");
  write(cg/"memory.usage_in_bytes","0");
  write(cg/"job42/memory.limit_in_bytes",std::to_string(12*gib));
  write(cg/"job42/memory.usage_in_bytes",std::to_string(5*gib));
  require(plan(inspect(proc)).bytes==7*gib);
  // A path outside a namespaced mount must never walk above the mount point.
  write(proc/"self/cgroup","4:memory:/other/job42\n");
  require(inspect(proc).constraints.size()==1);
  std::cout<<"CPU resource discovery and automatic budgeting passed\n";
}
