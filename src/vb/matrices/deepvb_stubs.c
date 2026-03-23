#include <stdio.h>

#include "vb/vb.h"

void deepvbscf_rdm(vb_info vb_str, double *energy, double *sums, double *grad) {
  (void)vb_str;
  (void)energy;
  (void)sums;
  (void)grad;
  fprintf(stderr, "deepvbscf_rdm stub was called unexpectedly.\n");
}

void deepvbscf_bovb(vb_info vb_str, double *energy, double *sums, double *grad) {
  (void)vb_str;
  (void)energy;
  (void)sums;
  (void)grad;
  fprintf(stderr, "deepvbscf_bovb stub was called unexpectedly.\n");
}
