# Exact One-Electron State Space Plan

## 1. Goal

The main goal is no longer "find another formula that looks simpler".
The real goal is:

> rewrite exact HLSP / raw-VB one-electron Hamiltonian evaluation so that the
> exponential part depends on a smaller effective state parameter
>
> ```math
> 2^m, \qquad m \ll n,
> ```
>
> rather than on the full HLSP determinant expansion scale `2^n`.

Here:

- `n` should be understood as the intrinsic combinatorial size of the exact
  HLSP / raw-VB determinant expansion;
- `m` must be a genuinely smaller parameter, not just a renamed form of `n`.

This document defines what we should do next if we want to keep pushing the
**exact** route for the one-electron Hamiltonian.

---

## 2. What Is Already Known

### 2.1 What has been established

- Exact raw-VB overlap admits a useful separator / layered recurrence.
- Exact raw-VB one-electron Hamiltonian does not follow from overlap by adding
  only scalar local payloads.
- The current mask-only closed correction is not exact.
- On the known `C6H6_full` failure case, the residual leaf-level cofactor
  family is not random noise. It has visible low-dimensional structure.

### 2.2 What has been falsified

- "Reuse overlap states and attach one-electron scalars" is false.
- "Add one mixed bridge correction" is false.
- "Use only a 2-state-per-spin mask basis for the residual closed family" is
  false on the known failing pair.

### 2.3 What remains plausible

- A richer exact one-electron state space may still exist.
- That state space likely needs to be more expressive than a pure mask-only
  leaf bundle.
- It may depend on root context or on an exact open-state tensor family.

---

## 3. What We Mean By `m`

At this stage, `m` should not be defined by hand-waving.
We need an operational definition.

The most useful current candidate is:

> `m` = the minimal effective dimension of the exact one-electron separator
> state family required to propagate the Hamiltonian contribution for one
> structure pair under a chosen root / leaf decomposition.

In practice, this can be probed through:

- span rank of the exact residual leaf-level cofactor family;
- number of independent exact open-state message channels;
- number of unique root-aware exact subproblems that must be retained.

This means the working complexity target becomes something like

```math
\mathrm{poly}(M)\,\exp(m)
```

instead of "directly sum over all exact determinant pairs".

The crucial test is:

> is `m` systematically much smaller than the exact determinant-pair scale on
> realistic structure pairs?

---

## 4. Immediate Research Focus

We should narrow the scope aggressively:

- exact only
- one-electron only
- closed-shell only
- star / near-star decompositions first

Do not mix in:

- two-electron terms
- AGP / pairing-state approximations
- thresholded graph truncation
- open-shell generalization

Those may become relevant later, but they will only blur the signal now.

---

## 5. The Right Mathematical Question

The next exact question is not:

> can we fit the final scalar Hamiltonian value?

That question is too weak, because a small number of post-summation equations
can always create misleading exact fits.

The correct question is:

> can the exact leaf-local one-electron object be represented by a small exact
> state algebra **before** full determinant-pair summation?

That means we are looking for an exact state family such as:

- closed state `Z`
- row-open state `R`
- column-open state `C`
- row-column cofactor kernel `Q`

but not necessarily in naive orbital-index form.

The real target is their **minimal exact representation** under the separator
decomposition.

### 5.1 The exact object we should propagate

The next formula should be written at the level of the exact local
one-electron object, not at the level of the final scalar Hamiltonian value.

For one spin channel `\sigma`, one fixed leaf, one fixed boundary mask
`\mu`, and one fixed root completion / root context `\rho`, define

```math
K_{\rho,\mu}^{(\sigma)}
```

to be the exact **leaf-level residual closed cofactor block** that remains
after removing the already captured closed contribution and the explicit
message-to-root cross contribution.

Operationally, this is the leaf-leaf block of the exact residual first
cofactor matrix in the component-ordered basis

```text
[selected root | leaf | root remainder].
```

This matrix object, not the final scalar contraction, is the correct
candidate for the exact separator payload.

### 5.2 Why scalar fitting is too weak

The final one-electron contribution is obtained only after contracting the
leaf-level cofactor object with the leaf-leaf one-electron operator block:

```math
H_{\rho,\mu}^{(1),(\sigma)}
=
\langle h_{\mathrm{leaf,leaf}}^{(\sigma)},
K_{\rho,\mu}^{(\sigma)} \rangle_F,
```

where `\langle A,B \rangle_F = \mathrm{Tr}(A^\mathrm{T} B)` is the Frobenius
inner product.

If we work only with the scalar

```math
H_{\rho,\mu}^{(1),(\sigma)},
```

then many different exact local cofactor structures can collapse onto the same
number. This is exactly why post-summation scalar fits can look better than
they really are.

So the exact state-space question must be asked at the matrix level.

### 5.3 The exact state space we want

For fixed `(\mu,\sigma)`, define the exact local state space

```math
\mathcal{V}_{\mu,\sigma}
=
\mathrm{span}_{\rho}
\left\{
K_{\rho,\mu}^{(\sigma)}
\right\}.
```

The first concrete reduced-state parameter is then

```math
m_{\mu,\sigma}
=
\dim \mathcal{V}_{\mu,\sigma}.
```

This is the strongest current candidate for the `m` in the target complexity

```math
\mathrm{poly}(M)\,\exp(m).
```

### 5.4 The basis-expansion formula we should try to prove

The exact formula worth exploring is:

```math
K_{\rho,\mu}^{(\sigma)}
=
\sum_{t=1}^{m_{\mu,\sigma}}
\phi_{t,\mu}^{(\sigma)}(\rho)\,
B_{t,\mu}^{(\sigma)},
```

where:

- `B_{t,\mu}^{(\sigma)}` are exact basis matrices for the residual local
  cofactor family;
- `\phi_{t,\mu}^{(\sigma)}(\rho)` are exact root-aware coefficients;
- `m_{\mu,\sigma}` is much smaller than the raw exact determinant-pair scale.

If such a formula exists and can be built **before** full determinant-pair
summation, then the leaf output is no longer one scalar per mask. Instead it
becomes a short exact coefficient vector

```math
\mathbf{k}_{\mu}^{(\sigma)}(\rho)
=
\left(
\phi_{1,\mu}^{(\sigma)}(\rho),
\ldots,
\phi_{m_{\mu,\sigma},\mu}^{(\sigma)}(\rho)
\right).
```

This coefficient vector is the right candidate for the exact one-electron
separator state.

### 5.5 Relation to `Z`, `R`, `C`, and `Q`

The earlier open-state language

- closed state `Z`
- row-open state `R`
- column-open state `C`
- cofactor kernel `Q`

should now be treated as a structural guide rather than the final compact
representation.

The basis matrices `B_{t,\mu}^{(\sigma)}` may emerge as:

- compressed combinations of `Q` restricted to the leaf-leaf sector;
- exact mixtures of row-open and column-open channels;
- root-aware bundle states that are richer than pure masks but still much
  smaller than explicit determinant-pair spaces.

So the right next question is not

> can we fit one better scalar correction?

but rather

> can we find the smallest exact basis that spans the family
> `\{K_{\rho,\mu}^{(\sigma)}\}_{\rho}`?

---

## 6. Working Hypothesis

The current best working hypothesis is:

> the exact one-electron state space is not a pure mask-indexed scalar bundle,
> but a richer root-aware open-state algebra whose dimension is still much
> smaller than the raw determinant-pair space.

This is worth pursuing because:

- the known failing example does not look fully chaotic;
- the residual family has small observed span compared with raw exact pair
  enumeration;
- the current failure looks like a wrong coordinate choice, not a proof of
  impossibility.

### 6.1 Current empirical signal from pilot scans

The first non-benzene pilot scans already show a consistent pattern on the
currently covered star-like structure pairs:

- width `0` examples can collapse to target span rank `1`
- width `1` examples can exhibit target span rank `2`
- width `2` examples can exhibit target span rank `4`

This is not yet a theorem, but it is exactly the kind of signal we hoped to
see: the exact one-electron state dimension appears to track separator width
far more closely than the raw determinant-pair count.

In the currently inspected width-`1` / rank-`2` and width-`2` / rank-`4`
examples, the root-pair-resolved dumps show:

- the local residual object is organized by a small number of root completion
  channels;
- each root-pair target matrix is itself low rank;
- the alpha and beta target families are exchanged under the corresponding
  root-spin swap;
- the width-`2` examples produce four root-pair target matrices per spin,
  while width-`1` examples produce two.

That makes the present working conjecture much sharper:

> the exact one-electron separator state may be organized by a small number of
> root-aware channels whose count grows roughly like `2^w` with separator
> width `w`, at least in the currently observed star-like closed-shell cases.

The next formula work should therefore focus on explaining this `1 / 2 / 4`
pattern constructively, rather than continuing to search for scalar
corrections.

A more explicit dump-based interpretation of this pattern has now been written
in:

- [exact_one_electron_root_channel_hypothesis.md](/pool1/home/xiatao/project/xmvb-cpp/src/pfaffian_vbscf/exact_one_electron_root_channel_hypothesis.md)

The current working view is that width-`2` does not look like four unrelated
root-pair payloads.  It looks closer to a `2 \times 2` root-channel table with
strong parity grouping and alpha/beta exchange covariance.  The current note
now also writes down the explicit width-`1` two-channel and width-`2`
spin-coupled Walsh formulas that reconstruct the observed root-pair tables.

---

## 7. What Must Be Measured

From now on, every new exact idea should be judged by the same hard metrics.

### 7.1 State-space metrics

For each tested structure pair:

- exact residual family span rank
- number of exact basis channels actually needed
- dependence on root choice
- dependence on decomposition width

### 7.2 True-work metrics

For each tested structure pair:

- unique baseline determinant-pair count
- unique exact separator subproblem count
- unique root-aware message / bundle count
- ratio of new exact work to baseline exact work

### 7.3 Correctness metrics

For each tested structure pair:

- exact overlap error must be zero within tolerance
- exact one-electron error must be zero within tolerance

If the reformulation is not exact, then it is not solving the present problem.

---

## 8. Benchmark Strategy

Benzene is useful, but it should not be the only benchmark.

Reason:

- `C6H6_full` is highly symmetric;
- symmetry can both help and mislead;
- a method that looks promising on benzene alone may fail to generalize.

So the benchmark pool should be broadened immediately.

### 8.1 Primary benchmark pool

Use the curated `6e6o full` training set:

- repository copy:
  [data/training_xmi/6e6o_full](/pool1/home/xiatao/project/xmvb-cpp/data/training_xmi/6e6o_full)
- original source mentioned by the user:
  `/export/home/xiatao/project/xmvb-cpp/data/training_xmi/6e6o_full`

This dataset already contains many `6e6o`, `str=full` XMVB inputs beyond
benzene.

### 8.2 Benchmark roles

The benchmark set should be split conceptually into three roles.

`Role A: failure diagnostics`

- keep `C6H6_full` because it already exposes a concrete one-electron failure;
- use it to reject weak exact state-space ideas quickly.

`Role B: diversity screening`

- sample many other `6e6o_full` molecules from the training set;
- use them to measure whether the observed exact residual state dimension is
  usually small, moderately small, or large.

`Role C: serious benchmark subset`

- after the screening pass, choose a smaller representative subset:
  - low symmetry
  - different graph widths
  - different overlap-density patterns
  - different raw determinant-pair counts

That representative subset should become the real benchmark suite.

### 8.3 What to avoid

Do not claim success from:

- one benzene structure pair
- one molecule
- one root choice

The method must show compression behavior across a distribution, not just on a
single hand-picked example.

---

## 9. Execution Plan

### Phase 1: map the exact residual state dimension

For many structure pairs across the `6e6o_full` dataset, collect:

- exact one-electron residual family span rank
- root choice
- width upper bound
- unique exact determinant-pair count
- unique exact separator subproblem count

The output of this phase should be a distribution, not anecdotes.

Success signal:

- the residual state dimension stays small on many pairs.

Failure signal:

- the dimension rapidly approaches the raw exact pair scale.

### Phase 2: search for a root-aware exact basis

If Phase 1 is encouraging, try to construct an exact basis that predicts the
residual family before full exact summation.

The candidate basis should be:

- richer than mask-only
- preferably local plus root-aware
- judged by exact reconstruction of the leaf-level object, not only of the
  final scalar Hamiltonian value

Success signal:

- exact reconstruction of the local residual object;
- exact final one-electron value;
- reduced unique exact work.

Failure signal:

- the basis only works after post-summation fitting;
- or it reproduces the object but does not reduce exact work.

### Phase 2.5: test candidate basis formulas directly

Before implementing a full new recurrence, every candidate exact basis formula
should first be tested directly against

```math
K_{\rho,\mu}^{(\sigma)}.
```

For each proposed basis family

```math
\left\{
B_{t,\mu}^{(\sigma)}
\right\}_{t=1}^{m},
```

we should check:

1. whether every sampled `K_{\rho,\mu}^{(\sigma)}` lies exactly in that span;
2. whether the coefficient map
   `\phi_{t,\mu}^{(\sigma)}(\rho)` can be built without reverting to full
   determinant-pair summation;
3. whether the resulting basis size `m` is genuinely smaller than the raw
   exact combinatorial scale.

### Phase 3: convert basis size into `m`

If an exact basis works, define

```math
m = \log_2(\text{effective exact state count})
```

or the analogous parameter controlling the exponential part of the exact
recurrence.

This is the point where we can honestly ask whether we have achieved

```math
2^m, \qquad m \ll n.
```

---

## 10. Success Criterion

We should only call the exact one-electron program successful if all three
conditions hold:

1. exactness:
   zero error in exact one-electron matrix elements within numerical tolerance

2. compression:
   the number of unique exact subproblems is smaller than the baseline exact
   determinant-pair count

3. parameter reduction:
   the exponential part is governed by a smaller effective parameter `m`, and
   `m` is empirically much smaller than the raw HLSP expansion scale on a
   meaningful benchmark set

If any one of these fails, we have not yet solved the real problem.

---

## 11. Failure Criterion

We should be prepared to stop the exact route if the data show:

- residual state dimension grows essentially like the determinant-pair space;
- any exact basis is just a repackaging of the same unique exact work;
- or the required root-aware context becomes so large that the reformulation
  ceases to be useful.

That would not mean the work was wasted.
It would simply mean:

- exact one-electron compression is weaker than hoped, and
- the practical route should switch to controlled approximation or screening.

---

## 12. Recommended Next Concrete Action

The very next task should be:

> build a dataset-level diagnostic pass over the `6e6o_full` pool and collect
> exact one-electron residual-state dimension statistics across many structure
> pairs, not just benzene.

This is the cleanest way to answer the question:

> is the exact one-electron state space usually small enough to justify
> continued effort?

Until that distribution is known, any further exact formula work is too easy
to overfit to one special molecule.
