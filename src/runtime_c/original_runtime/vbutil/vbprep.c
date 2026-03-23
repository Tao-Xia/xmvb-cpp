#include "scf/hf.h"
#include "inpout/input.h"
#include "vb/vb.h"
#include "inpout/output_func.h"

int vbprep(hf_info hf,inp_info inp_str,vb_info vb_str,para_info par_str, int print_level) {


  if (inp_str->inttyp!=INT_READ)
    vb_str->enuc=hf->nuc_rep[0];

//  printf(" enuc for VB = %f\n",vb_str->enuc);
//  #ifdef MPI
//    if (par_str->myproc==0) {
//      detect_blocks(vb_str);
//  
//      print_blk_info(vb_str->blocks,vb_str->nblock,vb_str->nor,NULL);
//  
//      if ((vb_str->dovbcis+vb_str->dovbcisd+vb_str->dovbcids)>0 && vb_str->block_part_ov>0) {
//        printf("Error Partly overlaped orbitals found. Cannot proceed VBCI.");
//        return 1;
//      }
//      getvars(vb_str);
//      print_var_info(vb_str,NULL);
//    }
//  #else
  if (vb_str->dobovb>0)
    init_bovb(vb_str);

  detect_blocks(vb_str);

  if (vb_str->dovbci>0 && vb_str->block_part_ov>0) {
    printf("Partly overlap detected. VBCI cannot be proceeded.\n");
    exit(1);
  }

  if (vb_str->dobovb>0 && vb_str->boysloc>0) {
    printf("Boys localization cannot be proceeded with BOVB.\n");
    printf("Will be disabled now.\n");
    vb_str->boysloc=0;
  }

//  print_blk_info(vb_str->noc_block,vb_str->blocks,vb_str->nblock,vb_str->nb,vb_str->nor,vb_str->dovbci,NULL);

  if ((vb_str->dovbcis+vb_str->dovbcisd+vb_str->dovbcids)>0 && vb_str->block_part_ov>0) {
    printf("Error Partly overlaped orbitals found. Cannot proceed VBCI.");
    return 1;
  }
  getvars(vb_str,print_level);

  check_dir2e(&vb_str->dir2e,vb_str->n2e,vb_str->nb,vb_str->inttyp);

  if (print_level>0)
    print_vb_comp_info(vb_str);
//  print_var_info(vb_str,NULL);
// #endif

//vbguess(hf,inp_str,vb_str,out_name);


  return 0;
}
