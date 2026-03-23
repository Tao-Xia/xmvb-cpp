#ifdef MPI
#include "mpi.h"
#endif

struct ParaInfo {
  int myproc,nprocs,ncores,thread_num;
};

typedef struct ParaInfo* para_info;
