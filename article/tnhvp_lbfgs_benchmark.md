# TNHVP versus nonredundant L-BFGS benchmark

## Benchmark status

The benchmark produced from revision `45e8e48` is withdrawn. Although its
command line selected `nonredundant_lbfgspp`, the implementation applied the
L-BFGS inverse-Hessian history in the full packed sparse-coefficient space.
It then subtracted projected gradients expressed in different accepted-point
quotient charts without transporting either the primal step or the dual
gradient change. Gauge canonicalization therefore corrupted the secant
history, and the resulting iteration counts were not a valid nonredundant
L-BFGS baseline.

The corrected implementation stores each secant as a quotient-space primal
step and dual gradient change lifted into the common packed embedding. After
an accepted orbital update, it:

1. transports a step with the new chart's vector projection;
2. transports a gradient covector with the new chart's covector pullback;
3. checks positive curvature only after both quantities occupy the same new
   quotient chart; and
4. clears history only if the quotient rank changes, not when an equivalent
   orbital gauge representative changes.

A regression calculation for F2 now converges in 8 accepted L-BFGS steps
under the production tolerances, compared with 177 steps from the invalid
implementation. This is a correctness check, not a publication benchmark.
The complete TNHVP/L-BFGS table must be regenerated on the same compute nodes
from the corrected revision before any performance conclusion is reported.
