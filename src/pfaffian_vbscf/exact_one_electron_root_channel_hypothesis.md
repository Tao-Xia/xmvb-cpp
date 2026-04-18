# Exact One-Electron Root-Channel Hypothesis

## 1. Purpose

This note records the next concrete step after the earlier matrix-level
diagnostics:

- stop searching for more scalar closed corrections;
- inspect the root-pair-resolved target matrices directly;
- extract the smallest exact **root-channel organization** that is consistent
  with the observed width-`1` and width-`2` star-like closed-shell examples.

The goal is still the same:

```math
K_{\rho,\mu}^{(\sigma)}
=
\sum_{t=1}^{m_{\mu,\sigma}}
\phi_{t,\mu}^{(\sigma)}(\rho)\,B_{t,\mu}^{(\sigma)},
```

with `m_{\mu,\sigma}` tied to separator width rather than to the full HLSP
determinant-pair count.

This note does **not** claim that the exact recurrence has already been
completed.  It records the strongest structural pattern currently visible in
the dump data.

---

## 2. Objects Under Inspection

For one spin channel `\sigma`, one mask `\mu`, and one root completion
`\rho`, the diagnostic tool outputs the exact residual leaf-level cofactor
target matrix

```math
K_{\rho,\mu}^{(\sigma)}.
```

The relevant dump files currently inspected are:

- `/tmp/rank2_dump_141780729_70_49.out`
- `/tmp/rank4_dump_144909560_63_4.out`
- `/tmp/rank4_dump_101687105_59_7.out`

These are all one-leaf covered star-like closed-shell examples.

---

## 3. Width-1 Pattern

In the width-`1` example, there is one fixed left-root completion and two
right-root completions.  The dump therefore produces two root-pair target
matrices per spin.

For the alpha channel:

```math
K^{(\alpha)}_{0}, \qquad K^{(\alpha)}_{1}.
```

The beta channel is obtained by the corresponding root-spin swap:

```math
K^{(\beta)}_{0} = K^{(\alpha)}_{1},
\qquad
K^{(\beta)}_{1} = K^{(\alpha)}_{0}.
```

So the width-`1` case is consistent with a **two-channel** organization:

```math
\text{right-root parity} \in \{0,1\}.
```

This matches the observed target span rank `2`.

---

## 4. Width-2 Pattern From The Dump

In both inspected width-`2` examples, the dump produces four root-pair target
matrices per spin.  Label the two left-root completions by `i \in \{0,1\}` and
the two right-root completions by `j \in \{0,1\}`.  Then the alpha-channel
target matrices form a `2 \times 2` table:

```math
K^{(\alpha)}_{ij}.
```

The dump shows a consistent symmetry:

```math
K^{(\alpha)}_{11} = K^{(\beta)}_{00},
\qquad
K^{(\alpha)}_{10} = K^{(\beta)}_{01},
```

and similarly with alpha and beta exchanged.

So the four matrices are not behaving like four unrelated objects.  They are
organized into two pairs under root-spin exchange.

### 4.1 Parity grouping

Define the binary parity

```math
p = i \oplus j.
```

Then the four root pairs split into two classes:

- `p = 0`: aligned left/right root completions, namely `(0,0)` and `(1,1)`
- `p = 1`: crossed left/right root completions, namely `(0,1)` and `(1,0)`

In both rank-`4` dump files, the target matrices within each parity class are
related by alpha/beta exchange under the left-root spin flip:

```math
K^{(\alpha)}_{1j} = K^{(\beta)}_{0,\,j \oplus 1}.
```

Operationally, the width-`2` data look like:

```math
\bigl(
  K^{(\alpha)}_{p,\lambda},
  K^{(\beta)}_{p,\lambda}
\bigr),
\qquad
p \in \{0,1\},
\quad
\lambda \in \{0,1\},
```

where `p` is a parity-like root channel and `\lambda` is the left-root spin
label that determines whether the alpha/beta payloads are swapped.

Equivalently, the dump is consistent with an exact covariance pattern of the
form

```math
K_{ij}^{(\alpha)} = A_{\,i \oplus j,\; i},
\qquad
K_{ij}^{(\beta)} = A_{\,i \oplus j,\; 1-i},
```

for some leaf-pair-dependent matrix payload family `A_{p,\lambda}`.

This is still only a hypothesis about the correct coordinate system, but it is
already much more specific than the earlier mask-only scalar language.

This is a much sharper statement than "the target span rank is 4".

---

## 5. Walsh-Type Decomposition Of The Width-2 Tables

For the width-`2` alpha target table

```math
\{K^{(\alpha)}_{00}, K^{(\alpha)}_{01}, K^{(\alpha)}_{10}, K^{(\alpha)}_{11}\},
```

it is natural to form the four binary-channel combinations

```math
B_{00}
=
\frac{1}{4}
\left(
K_{00}+K_{01}+K_{10}+K_{11}
\right),
```

```math
B_{L}
=
\frac{1}{4}
\left(
K_{00}+K_{01}-K_{10}-K_{11}
\right),
```

```math
B_{R}
=
\frac{1}{4}
\left(
K_{00}-K_{01}+K_{10}-K_{11}
\right),
```

```math
B_{LR}
=
\frac{1}{4}
\left(
K_{00}-K_{01}-K_{10}+K_{11}
\right).
```

For both inspected width-`2` examples, these combinations show the same
qualitative pattern:

- `B_{00}` and `B_{LR}` are dominant;
- `B_{L}` and `B_{R}` are much smaller but not identically zero.

So the data are **close** to a pure parity organization, but not exactly equal
to one.

This explains two facts at once:

- why the width-`2` examples still show strong low-dimensional structure;
- why a naive parity-only exact formula is not yet justified.

---

## 6. Numerical Verification Status

The current hypothesis is no longer supported only by visual inspection.
It is now backed by two concrete diagnostics.

### 6.1 Direct analyzer verification on `C6H6_full`

For the known benzene failure pair

- `test_molecule/C6H6_full.xmi`
- `left_structure = 1`
- `right_structure = 0`

the updated analyzer now reports

```text
explicit_one_leaf_spin_coupled_root_channel_left_state_count
  = 2
explicit_one_leaf_spin_coupled_root_channel_right_state_count
  = 2
explicit_one_leaf_spin_coupled_root_channel_count
  = 4
explicit_one_leaf_spin_coupled_root_channel_max_frobenius_residual
  = 4.562732988774192e-15
explicit_one_leaf_spin_coupled_root_channel_total_frobenius_abs_error
  = 1.072249283855449e-14
explicit_one_leaf_spin_coupled_root_channel_supported
  = true
explicit_one_leaf_root_channel_swap_covariance_max_frobenius_residual
  = 4.562732988774192e-15
explicit_one_leaf_root_channel_swap_covariance_max_relative_residual
  = 5.820470439929941e-16
explicit_one_leaf_root_channel_swap_covariance_pair_count
  = 4
explicit_one_leaf_alpha_width2_walsh_small_component_fraction
  = 0.06026502546807763
explicit_one_leaf_beta_width2_walsh_small_component_fraction
  = 0.06026502546807746
```

So on this width-`2` example:

- the spin-coupled four-channel recurrence itself reconstructs the exact
  root-pair target family to numerical precision;
- the root-swap alpha/beta covariance is exact to numerical precision;
- the parity-dominant Walsh sectors strongly dominate the skew sectors;
- but the skew sectors are still nonzero, so parity-only is not exact.

### 6.2 Direct channel accumulation on the same width-2 example

The analyzer has now been pushed one step further.

Instead of

1. building the full root-pair target matrix table first, and then
2. extracting the width-`1` / width-`2` channel matrices afterward,

the explicit one-leaf loop now accumulates the channel matrices
**directly** during the exact recurrence.

For the same benzene pair, the direct path reports

```text
explicit_one_leaf_direct_spin_coupled_root_channel_left_state_count
  = 2
explicit_one_leaf_direct_spin_coupled_root_channel_right_state_count
  = 2
explicit_one_leaf_direct_spin_coupled_root_channel_count
  = 4
explicit_one_leaf_direct_spin_coupled_root_channel_max_frobenius_residual
  = 4.026376522367723e-15
explicit_one_leaf_direct_spin_coupled_root_channel_total_frobenius_abs_error
  = 2.150039022133084e-14
explicit_one_leaf_direct_spin_coupled_root_channel_max_alpha_beta_basis_disagreement
  = 3.664156496154647e-15
explicit_one_leaf_direct_spin_coupled_root_channel_supported
  = true
```

So for the covered one-leaf width-`2` case, the analyzer is no longer only
showing an exact **post-hoc factorization** of a finished table.  It now
verifies an exact **direct channel-generation step** inside the explicit
recurrence itself.

### 6.3 Offline verification on saved width-1 / width-2 dumps

Applying the same channel diagnostics to the previously saved dump files gives:

```text
/tmp/rank2_dump_141780729_70_49.out
  swap_max_residual = 0
  swap_pair_count = 2

/tmp/rank4_dump_144909560_63_4.out
  swap_max_residual = 0
  swap_pair_count = 4
  alpha_width2_walsh_small_fraction = 0.0020509124973884965
  beta_width2_walsh_small_fraction = 0.0020509124973884965

/tmp/rank4_dump_101687105_59_7.out
  swap_max_residual = 0
  swap_pair_count = 4
  alpha_width2_walsh_small_fraction = 0.045828901112772456
  beta_width2_walsh_small_fraction = 0.04582890111277245
```

This is the strongest current empirical evidence that the observed
root-channel organization is real and not an accidental artifact of one
special pair.

---

## 7. Constructive Channel Formula For The Current Covered Cases

The new diagnostics make it possible to state the current candidate exact
state algebra much more explicitly.

### 7.1 Width-1 spin-coupled two-channel formula

When there is one fixed left-root completion and two right-root completions,
assign a binary sign

```math
s_R(j) \in \{+1,-1\}.
```

Then the alpha-channel table can be written as

```math
K^{(\alpha)}(j) = B_0 + s_R(j)\,B_R,
```

with

```math
B_0 = \frac{1}{2}\left(K^{(\alpha)}_0 + K^{(\alpha)}_1\right),
\qquad
B_R = \frac{1}{2}\left(K^{(\alpha)}_0 - K^{(\alpha)}_1\right).
```

The beta channel is then determined by the root-swap covariance:

```math
K^{(\beta)}(j) = B_0 - s_R(j)\,B_R.
```

So the width-`1` case is governed by **two exact matrix channels across both
spin sectors together**.

### 7.2 Width-2 spin-coupled four-channel formula

When there are two left-root completions and two right-root completions,
assign binary signs

```math
s_L(i) \in \{+1,-1\},
\qquad
s_R(j) \in \{+1,-1\}.
```

Then the alpha-channel table can be written in the Walsh basis as

```math
K^{(\alpha)}_{ij}
=
B_{00}
+ s_L(i)\,B_L
+ s_R(j)\,B_R
+ s_L(i)s_R(j)\,B_{LR},
```

where

```math
B_{00}=\frac{1}{4}(K_{00}+K_{01}+K_{10}+K_{11}),
```

```math
B_{L}=\frac{1}{4}(K_{00}+K_{01}-K_{10}-K_{11}),
```

```math
B_{R}=\frac{1}{4}(K_{00}-K_{01}+K_{10}-K_{11}),
```

```math
B_{LR}=\frac{1}{4}(K_{00}-K_{01}-K_{10}+K_{11}).
```

The beta-channel table is then determined by the exact root-swap covariance:

```math
K^{(\beta)}_{ij}
=
B_{00}
- s_L(i)\,B_L
- s_R(j)\,B_R
+ s_L(i)s_R(j)\,B_{LR}.
```

This means that, for the currently covered width-`2` cases, the combined
alpha/beta target family is controlled by **four exact matrix channels**, not
by eight unrelated spin-labeled root-pair matrices.

### 7.3 What is constructive and what is not

The formulas above are now constructive on the covered one-leaf width-`1/2`
cases:

- they give an explicit basis;
- they give explicit root-context coefficients;
- they reconstruct both alpha and beta sectors from one coupled channel set;
- and the analyzer now accumulates the channel matrices directly inside the
  exact one-leaf loop, rather than extracting them only after the root-pair
  table has already been built.

What is still missing is the next layer:

> replace the current explicit one-leaf determinant enumeration by a genuine
> separator recurrence whose propagated state is exactly this small
> root-channel object.

That is the remaining gap between the present diagnostic success and a finished
exact one-electron algorithm.

---

## 8. Best Current Interpretation

The dump data support the following working interpretation.

### 8.1 What seems true

- The exact local residual family is not chaotic.
- Width-`1` behaves like a two-channel root state.
- Width-`2` behaves like a four-channel root state, but with a strong internal
  organization into two parity sectors plus alpha/beta exchange symmetry.
- The correct exact payload is likely a **spin-coupled root-channel tensor**,
  not an independent scalar correction attached to each mask.

### 8.2 What is still not proved

- We do not yet have a constructive formula for the basis matrices
  `B_{t,\mu}^{(\sigma)}`.
- We do not yet know how to compute the coefficients
  `\phi_{t,\mu}^{(\sigma)}(\rho)` without falling back to determinant-pair
  enumeration.
- We do not yet know whether the same channel algebra remains exact beyond the
  currently covered one-leaf star-like cases.

---

## 9. What This Means For The Exact Route

The exact route is still alive, but the target has become more specific.

The next exact recurrence should **not** be formulated as:

- overlap state plus scalar one-electron payload;
- mask-only closed correction;
- one extra mixed-bridge correction.

It should instead be formulated as:

```math
\text{separator state}
\;\to\;
\text{root-channel-resolved matrix payload}
\;\to\;
\text{final one-electron contraction}.
```

More concretely, the next candidate state algebra should be built around:

- a binary root-channel index for width `1`;
- a `2 \times 2` left/right binary root-channel table for width `2`;
- explicit alpha/beta exchange covariance under root-spin swap.

If this can be promoted from a dump observation to an exact recurrence, then
the relevant exponential factor would no longer be tied directly to the full
HLSP expansion size.  It would instead track the number of exact root channels,
which currently looks much closer to `2^w`.

---

## 10. Immediate Next Step

The next step should be constructive, not statistical.

For the currently covered width-`1` and width-`2` cases:

1. define the exact root-channel basis matrices explicitly;
2. express the dumped target matrices in that basis;
3. identify which coefficients depend only on left-root context, which depend
   only on right-root context, and which depend on the parity coupling;
4. only after that, attempt an exact separator recurrence.

Until that constructive basis is written down, we should regard the current
result as:

> strong evidence that a small exact root-channel algebra exists, but not yet a
> finished exact Hamiltonian algorithm.
