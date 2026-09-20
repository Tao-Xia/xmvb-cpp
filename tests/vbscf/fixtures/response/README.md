# Fixed-point structure-response regressions

These orbital snapshots hold the accepted point fixed independently of future
changes to the optimizer trajectory. They were captured from Hanhai25 Slurm job
246977 (`consistency-3460c35`) and audited in job 246988. No integral cache or
full structure matrices are required. The corresponding canonical inputs in
`testdata/vbscf/` are byte-identical to the capture inputs.

| Fixture | Canonical input | Accepted step | Basis functions | Orbitals | Audit trust radius |
|---|---|---:|---:|---:|---:|
| `241_orbitals.txt` | `241_VBSCF.xmi` | 3 | 120 | 24 | 0.397717925496941 |
| `7975_orbitals.txt` | `7975_vb.xmi` | 4 | 140 | 28 | 0.00498924840619093 |

## Format and provenance

The first integer is the number of coefficients. Subsequent whitespace-separated
decimal doubles follow the original `orbital_value_table.data()` storage order,
including zeros. The values were transcribed with `od -An -v -t f8` from the
little-endian IEEE-754 snapshots and retain double-precision round trips. The
text fixture reader checks the count, finite coefficients and absence of trailing
data; it does not round or renormalize the input orbitals.

Original binary files are under
`build/diagnostics/consistency-3460c35/capture/<system>/davidson/trace/input/steps/step_<index>/orbital_value_table_f64.bin`.

SHA-256 of original binaries:

```text
241:  0afbe2942a745c54499d15c67e85a6090ce5a013b218c038984f9b21aa063016
7975: 9abd449cc4fde94fbb3e9b8440559bafc32eabe5e370e18e8f4f09ad101ddfcd
```

SHA-256 of canonical inputs at capture:

```text
241:  5dbd9c72c4ab5b8dec9b1bda5ef8c8d5d64741cc181f25cc8604e2464f9f07c8
7975: a97008de42758ef19838b2ac6a62653184341fb842dc3d440b36d7a736321fb7
```

## Assertions and scope

The audit trains a common response model on two directions, enriches it with a
third independent direction and refreshes all earlier images. Within the final
frozen model it asserts symmetry, scalar/block and segmented agreement, reversed
column order, linear combination, repeated action, zero action, and an unchanged
model revision. All defects have explicit scaled floating-point bounds and cause
a nonzero exit on failure. It also checks the optimizer's cached step image
against a fresh frozen action. The underlying optimizer error remains fatal.

Differences between a cold smaller model and an enriched model are legitimate;
historical adaptive replay diagnostics are not mistaken for frozen-model
invariants. A shared approximate model is not necessarily the exact Hessian.
Use the audit's separate `--audit-dense-reference true` and
`--finite-difference-step 1e-4` options to measure derivative accuracy. Those
reference diagnostics are not production fallbacks or exactness claims.

The original 241 audit failed with a raw secant skew of approximately
`1.07e-6` against its `4.73e-7` check. The original 7975 Newton audit produced
only one direction and therefore could not test symmetry. The new deterministic
three-direction probe avoids that vacuous pass.
