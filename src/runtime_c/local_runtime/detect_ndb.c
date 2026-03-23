#include <stdio.h>

#include "inpout/output_func.h"
#include "vb/vb.h"

int detect_ndb(int* ntstr, int nstr, int nel, int nmul, int wfntyp) {
  int ndb = 0;
  int nbeta = (nel + 1 - nmul) / 2;
  int ok = 1;
  int iold;
  int icurr;

  if (wfntyp == WFN_STR) {
    for (int i = 0; i < nbeta; i++) {
      if (i > 0) {
        iold = ntstr[2 * i - 2];
        icurr = ntstr[2 * i];
        if (icurr != iold + 1) {
          ok = 0;
          break;
        }
      }
      for (int j = 0; j < nstr; j++) {
        if (*(ntstr + j * nel + 2 * i) != *(ntstr + j * nel + 2 * i + 1)) {
          ok = 0;
          break;
        }
        if (*(ntstr + j * nel + 2 * i) != *(ntstr + 2 * i)) {
          ok = 0;
          break;
        }
      }
      if (ok == 0) {
        break;
      }
      ndb++;
    }
  } else if (wfntyp == WFN_DET) {
    for (int i = 0; i < nbeta; i++) {
      if (i > 0) {
        iold = ntstr[i - 1];
        icurr = ntstr[i];
        if (icurr != iold + 1) {
          ok = 0;
          break;
        }
      }
      for (int j = 0; j < nstr; j++) {
        if (*(ntstr + j * nel + i) != *(ntstr + j * nel + i + nbeta)) {
          ok = 0;
          break;
        }
        if (*(ntstr + j * nel + i) != *(ntstr + i)) {
          ok = 0;
          break;
        }
      }
      if (ok == 0) {
        break;
      }
      ndb++;
    }
  }

  return ndb;
}

int detect_ndb_(int* ntstr, int* nstr, int* nel, int* nmul, int* wfntyp) {
  return detect_ndb(ntstr, *nstr, *nel, *nmul, *wfntyp);
}
