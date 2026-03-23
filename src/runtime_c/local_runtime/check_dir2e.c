#include <stdio.h>
#include "vb/vb.h"

void check_dir2e(int *dir2e, long n2e, int nb, int inttyp) {

  if (inttyp==INT_CINT) {
    if (*dir2e==0) {
      unsigned long total_mem=get_total_mem();
      unsigned long nb2=(nb*nb+nb)/2;
      unsigned long nb4=(nb2*nb2+nb2)/2;
      unsigned long mem1d = nb4*8;
//      unsigned long memnb3 = n2e*nb*8;

//      if (mem1d <= total_mem && memnb3 <= total_mem) {
//        if (mem1d <= memnb3)
//          *dir2e=INT2E_1D;
//        else
//          *dir2e=INT2E_NB3;
//      }
//      else if (mem1d <= total_mem) {
      if (mem1d <= total_mem) {
        *dir2e=INT2E_1D;
      }
//      else if (memnb3 <= total_mem) {
//        *dir2e=INT2E_NB3;
//      }
      else {
        *dir2e=INT2E_DIR;
      }
    }
  }

  return;
}
