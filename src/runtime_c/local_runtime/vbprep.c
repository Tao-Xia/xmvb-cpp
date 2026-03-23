#include <stdio.h>
#include <stdlib.h>

#include "inpout/input.h"
#include "inpout/output_func.h"
#include "scf/hf.h"
#include "vb/vb.h"

int xmvb_cpp_vbprep(
    hf_info hf,
    inp_info inp_str,
    vb_info vb_str,
    para_info par_str,
    int print_level) {
  (void)par_str;

  if (inp_str->inttyp != INT_READ && hf != NULL) {
    vb_str->enuc = hf->nuc_rep[0];
  }

  if (vb_str->dobovb > 0) {
    init_bovb(vb_str);
  }

  detect_blocks(vb_str);

  if (vb_str->dovbci > 0 && vb_str->block_part_ov > 0) {
    printf("Partly overlap detected. VBCI cannot be proceeded.\n");
    exit(1);
  }

  if (vb_str->dobovb > 0 && vb_str->boysloc > 0) {
    printf("Boys localization cannot be proceeded with BOVB.\n");
    printf("Will be disabled now.\n");
    vb_str->boysloc = 0;
  }

  if ((vb_str->dovbcis + vb_str->dovbcisd + vb_str->dovbcids) > 0 &&
      vb_str->block_part_ov > 0) {
    printf("Error Partly overlaped orbitals found. Cannot proceed VBCI.");
    return 1;
  }

  getvars(vb_str, print_level);
  check_dir2e(&vb_str->dir2e, vb_str->n2e, vb_str->nb, vb_str->inttyp);

  if (print_level > 0) {
    print_vb_comp_info(vb_str);
  }

  return 0;
}
