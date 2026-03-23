#include "mol/xint.h"
extern void get_grids();
static int XSCFthread_num = 0;
static int OpenBlsdthread_num = 0;
int XSCF_tm = 1;
void init_xscf_world(int thread_num)
{
    XSCFthread_num = thread_num;
    openblas_set_num_threads(1);
    omp_set_num_threads(thread_num);
    get_grids();
    init_boys();
    XSCF_tm = 1;
    return;
}

void del_xscf_world()
{
    del_boys();
    return;
}

int get_xscf_threadnum()
{
    return XSCFthread_num;
}

void switch_xscf_thread(int blas_xscf)
{
    switch (blas_xscf)
    {
    case 0:
        /* code */
        openblas_set_num_threads(1);
        XSCF_tm = 1;
    default:
        openblas_set_num_threads(get_xscf_threadnum());
        XSCF_tm = 0;
        break;
    }
    return;
}
