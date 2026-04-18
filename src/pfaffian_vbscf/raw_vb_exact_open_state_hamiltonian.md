# Exact Raw-VB Hamiltonian Via Open-State Separator Recurrence

## 1. Purpose

This note replaces the failed idea

```text
reuse the overlap separator keys
+ store only scalar one-electron payloads per key
```

for exact raw-VB Hamiltonian matrix elements.

The new direction is:

- keep the exact raw-VB determinant definition unchanged;
- do not contract the one-electron operator inside a leaf too early;
- redesign the separator recurrence around exact open-state cofactor objects;
- use overlap only as one component of the state algebra, not as the state key.

The immediate target is the exact one-electron Hamiltonian. Two-electron terms
should be treated only after the one-electron open-state recurrence is
validated.

Relevant current prototype:

- `src/tools/analyze_star_separator_one_electron_dataset.cpp`

---

## 2. What Failed In The Scalar-Payload Design

For one spin channel, the exact determinant-pair one-electron contribution is

```math
H^{(1)}_\sigma
=
\sum_{r,c}
h_{rc}\,
C_{rc}(S_\sigma),
```

where:

- `r` indexes rows, hence right-determinant occupied orbitals;
- `c` indexes columns, hence left-determinant occupied orbitals;
- `S_\sigma` is the spin overlap submatrix;
- `C_{rc}(S_\sigma)` is the first cofactor of `S_\sigma`.

The failed prototype stored, for each separator mask, only the already
contracted scalar

```math
\sum_{r,c \in \text{local block}} h_{rc} C_{rc}.
```

This loses the orbital labels `r` and `c`.

That loss is fatal, because exact cross-region terms have the structure

```math
h_{rc}\, R_A(r)\, C_B(c),
```

where:

- region `A` contributes an open-row object carrying the row label `r`;
- region `B` contributes an open-column object carrying the column label `c`.

Once `r` and `c` are summed out locally, these cross terms can no longer be
reconstructed exactly. This is why:

- reusing the overlap keys with `(O, H_alpha, H_beta)` fails;
- adding a simple mixed bridge correction also fails.

The missing information is not a scalar correction. It is an exact open-state
tensor carrying unresolved orbital labels.

---

## 3. Minimal Exact One-Electron State Algebra

Consider one spin channel and one partial region `P` of the structure pair.
The recurrence should carry four exact message families.

### 3.1 Closed state

```math
Z_P(\mu_r,\mu_c)
```

This is the exact closed determinant contribution for boundary row mask
`\mu_r` and boundary column mask `\mu_c`.

This is the overlap-style state already used by the current exact separator
overlap prototype.

### 3.2 Row-open state

```math
R_P(\mu_r,\mu_c; r)
```

This is the exact partial minor with one unresolved deleted row labeled by the
global orbital `r`.

Interpretation:

- the one-electron insertion has already chosen its row orbital inside `P`;
- the matching deleted column is still outside `P`.

### 3.3 Column-open state

```math
C_P(\mu_r,\mu_c; c)
```

This is the exact partial minor with one unresolved deleted column labeled by
the global orbital `c`.

Interpretation:

- the one-electron insertion has already chosen its column orbital inside `P`;
- the matching deleted row is still outside `P`.

### 3.4 Row-column cofactor kernel

```math
Q_P(\mu_r,\mu_c; r,c)
```

This is the exact partial cofactor kernel with both deleted row `r` and
deleted column `c` already resolved inside `P`.

This is the object that should be contracted with the one-electron integral
matrix only after the full recurrence is complete:

```math
H^{(1)}_{\sigma,\mathrm{full}}
=
\sum_{r,c} h_{rc}\, Q_{\mathrm{full}}(r,c).
```

Therefore the separator recurrence should propagate `Q`, not the already
contracted scalar `H^{(1)}`.

---

## 4. Why These Four Objects Are Minimal

Suppose two disjoint partial regions `A` and `B` are merged across a separator.
The exact one-electron cofactor kernel of the union must include four kinds of
terms:

```math
Q_{A \cup B}(r,c)
=
Q_A(r,c)\, Z_B
+ Z_A\, Q_B(r,c)
+ R_A(r)\, C_B(c)
+ R_B(r)\, C_A(c).
```

The first two terms mean:

- both deleted indices lie in `A`, or
- both deleted indices lie in `B`.

The last two terms are the cross-region terms:

- row in `A`, column in `B`;
- row in `B`, column in `A`.

This identity immediately shows:

- `Z` alone is insufficient;
- `Q` alone is insufficient;
- `R` and `C` are also required.

So the minimal exact one-electron separator algebra is:

```text
closed scalar Z
open-row vector R
open-column vector C
open-row/open-column kernel Q
```

---

## 5. Boundary Masks And Balance Conditions

The separator still needs boundary occupancy masks. Those masks remain useful,
but they are no longer the whole state.

For one spin channel, each payload is indexed by a boundary row mask and a
boundary column mask:

```math
(\mu_r,\mu_c).
```

The balance condition depends on the state type:

- closed `Z`: row count equals column count;
- row-open `R`: row count is one smaller than column count after accounting
  for the deleted row;
- column-open `C`: column count is one smaller than row count after accounting
  for the deleted column;
- cofactor kernel `Q`: balanced again after both one row and one column are
  deleted.

In implementation terms:

- `Z` and `Q` live on balanced mask pairs;
- `R` and `C` live on off-by-one mask pairs.

This is the first place where the exact Hamiltonian recurrence genuinely
departs from the overlap recurrence.

---

## 6. Exact Leaf Message Definition

Fix one root determinant-term pair for one spin channel. For one leaf `ell`,
the exact message bundle should be:

```math
\mathcal{M}_\ell
=
\{
Z_\ell(\mu_r,\mu_c),
R_\ell(\mu_r,\mu_c;r),
C_\ell(\mu_r,\mu_c;c),
Q_\ell(\mu_r,\mu_c;r,c)
\}.
```

These are obtained by summing exact local determinant contributions over all
leaf sign/orientation assignments that induce the same boundary masks.

Important:

- `r` and `c` are global orbital labels in the component-ordered support, not
  just local leaf positions;
- the one-electron matrix `h` is **not** used here;
- this keeps the leaf message exact and reusable for any one-electron matrix.

For the current star-separator prototype, the first implementation should build
these objects by direct exact local enumeration, without Schur optimizations.
Only after exactness is verified should we optimize the local builder.

---

## 7. Merge Recurrence For Partial States

Let

```math
\mathcal{F}_k(U_r,U_c)
=
\{Z_k, R_k, C_k, Q_k\}
```

be the exact state after merging the first `k` leaves, where `(U_r,U_c)` are
the accumulated used boundary masks for the current spin channel.

When adding leaf `k+1`, the update is:

```math
Z_{k+1}(U')
=
\sum_{\mu \compat U}
Z_k(U)\, Z_\ell(\mu),
```

```math
R_{k+1}(U';r)
=
\sum_{\mu \compat U}
\Big(
R_k(U;r)\, Z_\ell(\mu)
+ Z_k(U)\, R_\ell(\mu;r)
\Big),
```

```math
C_{k+1}(U';c)
=
\sum_{\mu \compat U}
\Big(
C_k(U;c)\, Z_\ell(\mu)
+ Z_k(U)\, C_\ell(\mu;c)
\Big),
```

```math
Q_{k+1}(U';r,c)
=
\sum_{\mu \compat U}
\Big(
Q_k(U;r,c)\, Z_\ell(\mu)
+ Z_k(U)\, Q_\ell(\mu;r,c)
+ R_k(U;r)\, C_\ell(\mu;c)
+ C_k(U;c)\, R_\ell(\mu;r)
\Big).
```

Here:

- `U'` is the updated used-mask pair after adding the leaf message mask `\mu`;
- `\mu \compat U` means the boundary masks are compatible and do not overlap.

This is the exact open-state analogue of the current overlap frontier
recurrence.

For alpha/beta spin blocks, the same recurrence is run independently, and the
full one-electron structure-pair value is assembled as

```math
H^{(1)}
=
H^{(1)}_\alpha\, O_\beta
+ O_\alpha\, H^{(1)}_\beta.
```

---

## 8. Root Completion

After all leaves are merged, the root remainder provides another exact state

```math
\mathcal{R}(U_r,U_c)
=
\{Z_R, R_R, C_R, Q_R\}
```

on the complementary root masks.

The final full-state cofactor kernel is obtained by the same product algebra:

```math
Q_{\mathrm{full}}(r,c)
=
Q_{\mathrm{frontier}}(r,c)\, Z_R
+ Z_{\mathrm{frontier}}\, Q_R(r,c)
+ R_{\mathrm{frontier}}(r)\, C_R(c)
+ C_{\mathrm{frontier}}(c)\, R_R(r).
```

Then the exact one-electron value is

```math
H^{(1)}_\sigma
=
\sum_{r,c} h_{rc}\, Q_{\mathrm{full}}(r,c).
```

Again, the contraction with `h` happens only here, after the exact global
cofactor kernel has been assembled.

---

## 9. Expected Scaling And The Right Compression Metric

Let:

- `m` be the component-ordered support size for the current structure pair;
- `N_mask` be the number of exact reachable separator mask states.

Then one exact state now carries:

- one scalar `Z`;
- one `O(m)` row-open vector `R`;
- one `O(m)` column-open vector `C`;
- one `O(m^2)` cofactor kernel `Q`.

So the natural one-electron cost model is no longer

```math
O(N_{\mathrm{mask}})
```

but rather

```math
O(m^2\, N_{\mathrm{mask}})
```

up to merge and local-build constants.

This is still acceptable as an exact-compression target, because the right
baseline is not the overlap-state count. The right baseline is the number of
unique exact determinant pairs needed by the determinant-expansion reference.

Therefore the correct question becomes:

> can the number of unique exact open-state work objects
> `(mask, payload index)` fall below the unique determinant-pair count?

That is the metric we should measure after implementation.

---

## 10. Extension Path To Two-Electron

If the one-electron open-state recurrence works, the two-electron exact
recurrence should follow the same philosophy.

The second-cofactor structure implies the need for two deleted rows and two
deleted columns. Therefore the natural exact payloads are higher-rank open
states such as:

- two-row-open;
- two-column-open;
- one-row/one-column-open;
- two-row/two-column cofactor kernel.

In other words, the one-electron algebra

```text
Z, R, C, Q
```

is only the rank-1 version of the general Hamiltonian recurrence. Two-electron
exactness will require the rank-2 analogue.

Because the payload size then becomes `O(m^4)` per mask state, we should not
attempt that until the rank-1 exact recurrence is numerically validated and
its true compression ratio is measured.

---

## 11. Immediate Implementation Plan

The next implementation step should be:

1. keep the current exact determinant-pair reference unchanged;
2. add a one-spin one-leaf exact validator that builds `Z`, `R`, `C`, and `Q`
   by direct enumeration;
3. verify that the open-state merge reproduces the exact one-electron value for
   the previously failing C6H6 structure pairs;
4. only then generalize from one leaf to the full star recurrence;
5. only after exactness is stable, optimize the local leaf builder.

This avoids repeating the earlier mistake of optimizing a state space before
its exactness is proved.

---

## 12. Important Diagnostic Update

The current one-leaf diagnostics already rule out one additional naive
construction.

For the failing `C6H6` structure pair, we grouped the exact cofactor
contraction of each explicit masked full spin block into the three block
classes

```text
[selected-root | leaf | root-remainder].
```

Summing these masked full-block contributions reproduces the previously
observed overcounted quantity, not the exact one-electron structure-pair
matrix element.

Therefore:

- it is **not** sufficient to attach `Q(r,c)` to the naive assembled full
  block for each overlap mask and then sum masks exactly as in the overlap
  recurrence;
- the exact open-state objects must instead be defined at the level of the
  Laplace-expanded `zeroed selected-root block` minors and their complementary
  root completions.

We also tested a stronger variant:

- for each mask, replace the overlap by the block-diagonal Laplace object

```math
\operatorname{diag}(B_{\mathrm{zeroed}}, D_{\mathrm{root}})
```

  in the common occupied-orbital order
  `[selected root | leaf | root remainder]`,
- then evaluate the exact one-electron cofactor contraction on that overlap in
  one shot.

This still reproduces only the old within-block quantity, not the exact
one-electron value. Numerically, for the current failing `C6H6` pair it gives
exactly the same incorrect result as the previous

```text
leaf one-electron * root overlap
+ root one-electron * leaf overlap
```

path.

The reason is structural:

- for a strictly block-diagonal overlap matrix, cross-block first cofactors are
  identically zero;
- therefore the cross-region one-electron terms can never appear from a naive
  cofactor evaluation on `diag(B_{\mathrm{zeroed}}, D_{\mathrm{root}})`.

So the exact recurrence must carry genuine open transfer states connecting
neighboring mask sectors, rather than relying on the closed balanced masks
alone.

The most likely next target is:

- off-by-one separator states that differ from the balanced overlap masks by
  one row or one column at the selected-root interface.

An additional diagnostic refines this picture:

- if we take the already reassembled full masked block and expand each cofactor
  minor by Laplace expansion over the message rows, the resulting transfer
  histogram has support only in the `0` and `1` transfer sectors, while the
  `>=2` sectors remain numerically zero for the current failing `C6H6` pair;
- however, this construction still reproduces the same overcounted
  full-block quantity rather than the exact one-electron matrix element.

So the current evidence is:

- naive full-block transfer expansion is still wrong;
- but the nonzero transfer order appears to be only `off-by-one`, not higher.

This is the first concrete numerical signal that an exact recurrence based on
off-by-one transfer states may be sufficient.

In practical terms, the next validator should build open states from the same
mask-level objects that make the overlap recurrence exact, rather than from the
cofactor matrix of the already reassembled full block.

---

## 13. One-Leaf Exact Update

The one-leaf validator is now numerically exact once the local open-state
objects are defined with the following constraint:

- the zeroed selected-root interface indices are **not** allowed to enter the
  leaf-local `Q`, `R`, or `C` payloads;
- only indices already resolved inside the leaf block may appear in those
  open states;
- interface deletions must remain in the transfer algebra and be completed by
  the complementary root state.

With that restriction, the one-spin one-leaf reconstruction matches the exact
determinant-pair one-electron value on the current `F2` and `C6H6_full`
one-leaf tests to numerical precision.

This rules out the previous overcounted construction where the selected-root
interface was absorbed into the local leaf cofactor kernel.

### 13.1 Directional open-state sign convention

One additional refinement is needed for the true recurrence:

- the row-open and column-open states are not naturally symmetric between the
  frontier side and the complementary side;
- a sign-free

```text
Q <- QZ + ZQ + RC + CR
```

  assembly is obtained only after absorbing the Laplace signs into the open
  states with **direction-aware** conventions.

Empirically, for the current one-leaf validator:

- frontier-side open states require a different sign convention from the
  root/complement-side open states;
- after that redefinition, the one-leaf exact formula no longer needs an
  explicit local-index sign in the `RC` / `CR` contractions.

This is the first concrete signal that the correct multi-leaf Hamiltonian
recurrence may need *directional* open states, for example frontier-left vs.
complement-right variants, rather than a single undirected `R` and `C`.

### 13.2 Compression update

Exactness alone does **not** imply compression at one leaf.

For the current one-leaf validator, counting unique spin-level determinant /
minor work objects gives:

- `F2` one-leaf test:
  reference unique spin determinants = `2`,
  open-state unique work objects = `3`;
- `C6H6_full` pair `(1,0)`:
  reference unique spin determinants = `112`,
  open-state unique work objects = `389`.

So the one-leaf exact open-state construction is currently a **correctness**
result, not yet a compression result.

Therefore the next meaningful target is no longer another one-leaf variant.
The next target is:

- derive the exact multi-leaf merge with the directional open-state
  conventions made explicit;
- then measure whether reuse across many leaves actually reduces the number of
  unique work objects below the reference determinant-pair count.
