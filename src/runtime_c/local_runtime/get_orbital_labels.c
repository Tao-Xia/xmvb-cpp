#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "mol/mol.h"

static int angular_len[] = {1, 3, 6, 10, 15, 21};
static const char ele_names[] = {
    ' ', ' ', 'H', ' ', 'H', 'e', 'L', 'i', 'B', 'e', 'B', ' ', 'C', ' ', 'N', ' ', 'O', ' ', 'F', ' ', 'N', 'e',
    'N', 'a', 'M', 'g', 'A', 'l', 'S', 'i', 'P', ' ', 'S', ' ', 'C', 'l', 'A', 'r', 'K', ' ', 'C', 'a', 'S', 'c',
    'T', 'i', 'V', ' ', 'C', 'r', 'M', 'n', 'F', 'e', 'C', 'o', 'N', 'i', 'C', 'u', 'Z', 'n', 'G', 'a', 'G', 'e',
    'A', 's', 'S', 'e', 'B', 'r', 'K', 'r'};

int xint_gtolen(int ang) {
  assert(ang <= 5);
  return angular_len[ang];
}

char* charge2ele(int charge) {
  char* ele = (char*)calloc(sizeof(char), 4);
  ele[0] = ele_names[2 * charge];
  ele[1] = ele_names[2 * charge + 1];
  return ele;
}

char* get_ang_tag(int i, int ang) {
  static char result[16];

  if (ang == 0) {
    strcpy(result, "S");
  } else if (ang == 1) {
    if (i == 0) {
      strcpy(result, "PX");
    } else if (i == 1) {
      strcpy(result, "PY");
    } else {
      strcpy(result, "PZ");
    }
  } else if (ang == 2) {
    if (i == 0) {
      strcpy(result, "DXX");
    } else if (i == 1) {
      strcpy(result, "DXY");
    } else if (i == 2) {
      strcpy(result, "DXZ");
    } else if (i == 3) {
      strcpy(result, "DYY");
    } else if (i == 4) {
      strcpy(result, "DYZ");
    } else {
      strcpy(result, "DZZ");
    }
  } else if (ang == 3) {
    if (i == 0) {
      strcpy(result, "FXXX");
    } else if (i == 1) {
      strcpy(result, "FXXY");
    } else if (i == 2) {
      strcpy(result, "FXXZ");
    } else if (i == 3) {
      strcpy(result, "FXYY");
    } else if (i == 4) {
      strcpy(result, "FXYZ");
    } else if (i == 5) {
      strcpy(result, "FXZZ");
    } else if (i == 6) {
      strcpy(result, "FYYY");
    } else if (i == 7) {
      strcpy(result, "FYYZ");
    } else if (i == 8) {
      strcpy(result, "FYZZ");
    } else {
      strcpy(result, "FZZZ");
    }
  } else if (ang == 4) {
    if (i == 0) {
      strcpy(result, "GXXXX");
    } else if (i == 1) {
      strcpy(result, "GXXXY");
    } else if (i == 2) {
      strcpy(result, "GXXXZ");
    } else if (i == 3) {
      strcpy(result, "GXXYY");
    } else if (i == 4) {
      strcpy(result, "GXXYZ");
    } else if (i == 5) {
      strcpy(result, "GXXZZ");
    } else if (i == 6) {
      strcpy(result, "GXYYY");
    } else if (i == 7) {
      strcpy(result, "GXYYZ");
    } else if (i == 8) {
      strcpy(result, "GXYZZ");
    } else if (i == 9) {
      strcpy(result, "GXZZZ");
    } else if (i == 10) {
      strcpy(result, "GYYYY");
    } else if (i == 11) {
      strcpy(result, "GYYYZ");
    } else if (i == 12) {
      strcpy(result, "GYYZZ");
    } else if (i == 13) {
      strcpy(result, "GYZZZ");
    } else {
      strcpy(result, "GZZZZ");
    }
  } else if (ang == 5) {
    if (i == 0) {
      strcpy(result, "HXXXXX");
    } else if (i == 1) {
      strcpy(result, "HXXXXY");
    } else if (i == 2) {
      strcpy(result, "HXXXXZ");
    } else if (i == 3) {
      strcpy(result, "HXXXYY");
    } else if (i == 4) {
      strcpy(result, "HXXXYZ");
    } else if (i == 5) {
      strcpy(result, "HXXXZZ");
    } else if (i == 6) {
      strcpy(result, "HXXYYY");
    } else if (i == 7) {
      strcpy(result, "HXXYYZ");
    } else if (i == 8) {
      strcpy(result, "HXXYZZ");
    } else if (i == 9) {
      strcpy(result, "HXXZZZ");
    } else if (i == 10) {
      strcpy(result, "HXYYYY");
    } else if (i == 11) {
      strcpy(result, "HXYYYZ");
    } else if (i == 12) {
      strcpy(result, "HXYYZZ");
    } else if (i == 13) {
      strcpy(result, "HXYZZZ");
    } else if (i == 14) {
      strcpy(result, "HXZZZZ");
    } else if (i == 15) {
      strcpy(result, "HYYYYY");
    } else if (i == 16) {
      strcpy(result, "HYYYYZ");
    } else if (i == 17) {
      strcpy(result, "HYYYZZ");
    } else if (i == 18) {
      strcpy(result, "HYYZZZ");
    } else if (i == 19) {
      strcpy(result, "HYZZZZ");
    } else {
      strcpy(result, "HZZZZZ");
    }
  }

  return result;
}
