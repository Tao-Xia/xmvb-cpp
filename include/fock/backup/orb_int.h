#ifndef _BUILDER_ORBINT_H_
#define _BUILDER_ORBINT_H_
#include "mol/xint.h"
#include "mol/xgrids.h"
struct OrbIntInfo
{
    bas_info aux;    // used for RI calculation
    grid_info grids; // used for THC calculation
    mol_info mol;
};
typedef struct OrbIntInfo *orbint_info;
#endif
