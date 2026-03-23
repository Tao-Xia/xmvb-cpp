#include <stdio.h>
#include <sys/sysinfo.h>

unsigned long get_total_mem() {
  struct sysinfo sys_info;
  sysinfo(&sys_info);
  unsigned long total_mem=sys_info.totalram*sys_info.mem_unit;
  return total_mem;
}
