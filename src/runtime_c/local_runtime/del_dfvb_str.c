#include <stdlib.h>

#include "vb/vb.h"

void del_dfvb_str(dfvb_info dfvb_str) {
  if (dfvb_str == NULL) {
    return;
  }

  free(dfvb_str->dft);
  free_vector(dfvb_str->dft_frac);
  free_list(dfvb_str->dft_id);
  free(dfvb_str);
}
