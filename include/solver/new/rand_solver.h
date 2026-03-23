#ifndef _SOLVER_RAND_SOLVER_H_
#define _SOLVER_RAND_SOLVER_H_
#include "mol/mol.h"
/**
 * @brief get random number following the distribution as
 *        f(x)  = 1/sqrt(pi * zeta) * exp(-zeta * (x - c)^2)
 *         by Box-Muller method
 * @param c the center of Gauss distribution
 * @param zeta var
 * @return double random number
 */
double rand_BoxMuller(double c, double zeta);
#endif
