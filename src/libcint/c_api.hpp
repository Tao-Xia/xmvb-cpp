#pragma once

// libcint's C header defines short macros that collide with common C++
// identifiers. Keep the C boundary isolated and do not export those macros.
#ifdef atm
#undef atm
#endif

#ifdef bas
#undef bas
#endif

extern "C" {
#include "cint.h"
}

#ifdef atm
#undef atm
#endif

#ifdef bas
#undef bas
#endif
