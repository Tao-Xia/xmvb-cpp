#include <stdio.h>
#include <stdlib.h>
#include "scf/hf.h"
#include "vb/vb.h"
#include "inpout/input.h"
#include "inpout/output_func.h"

int vbguess(hf_info hf,inp_info inp_str,vb_info vb_str,int print_level) {
  switch (vb_str->iguess) {
    case GUS_AUTO:
      vb_autoguess(hf,vb_str);
      break;
    case GUS_UNIT:
      vb_unitguess(vb_str);
      break;
    case GUS_READ:
    case GUS_RDCI:
      vb_readguess(inp_str,vb_str);
      break;
    case GUS_MO:
    case GUS_NBO:
      vb_moguess(hf,inp_str,vb_str);
//      break;
  }

  if (print_level>0)
    print_init_guess(vb_str);

  return 0;
}
