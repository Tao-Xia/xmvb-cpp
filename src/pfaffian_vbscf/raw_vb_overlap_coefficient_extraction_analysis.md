# Raw-VB Exact Overlap Coefficient Extraction: Complexity And Compression

## 1. Purpose

This note analyzes the computational bottleneck that remains after the exact
closed-shell raw-VB overlap has been rewritten as a bond-labeled generating
function.

The key question is no longer:

- how to write the overlap algebraically.

That part is already settled.

The real question is:

- how hard is the exact bond-labeled coefficient extraction;
- and under what additional structure can it be compressed.

Relevant files:

- `src/pfaffian_vbscf/raw_vb_exact_overlap_generating_function.md`
- `src/pfaffian_vbscf/raw_vb_exact_single_agp_failure.md`
- `src/pfaffian_vbscf/raw_vb_metric_aware_dp_design.md`
- `legacy_pair_generating_function.md`
- `scripts/verify_raw_vb_overlap_generating_function.py`
- `src/tools/check_union_graph_overlap_blocks.cpp`

---

## 2. Notation

Let:

- `n` be the total number of active pairs in each structure;
- `c_X` be the number of covalent pairs in the left structure `X`;
- `c_Y` be the number of covalent pairs in the right structure `Y`;
- `m = c_X + c_Y` be the total number of bond labels that must be projected;
- `S` be the active-space spatial overlap matrix.

Ionic pairs do not introduce orientation variables. Therefore the hidden
combinatorics depends on `m`, not on the total `n`.

The exact overlap formula is

```math
S_{XY}
=
[\mathbf{x}^{0}\mathbf{y}^{0}]\,
\det\!\bigl(U_X(\mathbf{x}) S U_Y(\mathbf{y})^{T}\bigr)\,
\det\!\bigl(V_X(\mathbf{x}) S V_Y(\mathbf{y})^{T}\bigr).
```

One numerical verification of this exact identity is now provided by

```bash
python scripts/verify_raw_vb_overlap_generating_function.py
```

which compares three routes:

1. explicit determinant expansion;
2. exact polynomial constant-term extraction;
3. exact hypercube averaging.

All three agree to machine precision on the included test cases.

---

## 3. Three Different Computational Layers

The most important practical point is that three different "costs" are hiding
inside the exact formula.

### 3.1 One fixed evaluation of the determinant kernel

If the bond labels are assigned concrete scalar values, then the matrices

```math
U_X S U_Y^{T},
\qquad
V_X S V_Y^{T}
```

are both ordinary `n x n` matrices.

Once assembled, one exact evaluation costs:

- `O(n^3)` for the two determinant evaluations;
- plus `O(n^2)` to assemble the small matrices from the selector rows.

So the kernel evaluation itself is polynomial and not the main obstacle.

### 3.2 Exact coefficient extraction

The hard part is not evaluating the determinant kernel once. The hard part is
extracting the exact multivariable coefficient that enforces "use every bond
exactly once".

This is where the exponential dependence on the number of covalent bonds can
reappear.

### 3.3 Verification-only polynomial manipulation

The current Python verification script also contains a direct Laurent-polynomial
determinant implementation. That route is useful for validation, but it is not
an intended algorithmic path.

It is more expensive than the clean exact coefficient-extraction route and
should not be confused with a production method.

---

## 4. Exact Hypercube Extraction

The determinant constant-term formula immediately yields a clean exact
coefficient-extraction algorithm.

For each covalent bond label, the total exponent appearing in the product

```math
\det\!\bigl(U_X(\mathbf{x}) S U_Y(\mathbf{y})^{T}\bigr)\,
\det\!\bigl(V_X(\mathbf{x}) S V_Y(\mathbf{y})^{T}\bigr)
```

is in the set

```math
\{-1,0,+1\}.
```

Therefore the constant term can be extracted exactly by averaging over the
Boolean hypercube:

```math
[\mathbf{x}^{0}\mathbf{y}^{0}]\,F(\mathbf{x},\mathbf{y})
=
\frac{1}{2^{m}}
\sum_{\substack{x_k=\pm 1\\ y_l=\pm 1}}
F(\mathbf{x},\mathbf{y}).
```

In our notation this becomes

```math
S_{XY}
=
\frac{1}{2^{m}}
\sum_{\substack{x_k=\pm 1\\ y_l=\pm 1}}
\det\!\bigl(U_X(\mathbf{x}) S U_Y(\mathbf{y})^{T}\bigr)\,
\det\!\bigl(V_X(\mathbf{x}) S V_Y(\mathbf{y})^{T}\bigr).
```

This is now implemented and verified in
`scripts/verify_raw_vb_overlap_generating_function.py`.

This formula is important because it separates the problem cleanly:

- one determinant-kernel evaluation is polynomial;
- exact coefficient extraction is a finite sum over `2^m` sign assignments.

---

## 5. Baseline Exact Scaling

### 5.1 Determinant-expansion oracle

The explicit determinant oracle enumerates:

- `2^{c_X}` left orientation assignments;
- `2^{c_Y}` right orientation assignments.

For each pair of assignments it evaluates two `n x n` determinants.

So the baseline exact cost is

```math
O(2^{c_X+c_Y} n^3) = O(2^m n^3).
```

If `c_X = c_Y = c`, this is

```math
O(4^c n^3).
```

### 5.2 Hypercube extraction

The exact hypercube formula above has the same asymptotic cost:

```math
O(2^m n^3).
```

So by itself, the generating-function reformulation does **not** remove the
worst-case exponential dependence on the number of covalent bonds.

What it does do is expose the combinatorial layer cleanly, which is the right
starting point for any future compression.

### 5.3 Verification-only polynomial route

The direct polynomial determinant in
`scripts/verify_raw_vb_overlap_generating_function.py` uses Leibniz expansion
of an `n x n` polynomial matrix. It is only a validation tool.

Its cost is much worse than the hypercube route. Roughly:

- `n!` permutation terms per determinant;
- and exponential growth of Laurent-monomial support in the bond labels.

So it should be regarded as:

- exact;
- useful for small-case proof-of-concept validation;
- not algorithmically competitive.

---

## 6. What Exact Compression Would Need To Improve

Because the per-sample determinant kernel is already polynomial, any exact
acceleration must attack one of these two levers:

1. reduce the number of required bond-label configurations below `2^m`;
2. reduce the cost of one fixed-label determinant evaluation below dense
   `O(n^3)`.

The second lever is useful but secondary. The first lever is the real barrier.

In other words:

- `O(n^3)` is not the main problem;
- the `2^m` bond-labeled projection is the main problem.

---

## 7. Exact Factorization Under Strict Block Structure

Suppose the active overlap matrix can be permuted into exact block-diagonal
form,

```math
S = \operatorname{diag}(S^{(1)},\dots,S^{(B)}),
```

and every bond in both structures lies entirely within one of these blocks.

Then:

- the selector matrices split by block;
- both determinant factors split by block;
- the exact constant-term extraction factorizes by block as well.

If block `b` contains `n_b` pairs and `m_b` bond labels, then the total exact
cost becomes

```math
\sum_{b=1}^{B} O(2^{m_b} n_b^3),
```

instead of

```math
O(2^{\sum_b m_b} (\sum_b n_b)^3).
```

This is a genuine exact reduction when the metric really does split.

So in the most favorable exact case, the exponential depends on the largest
independent block, not on the full structure.

The drawback is obvious:

- nonorthogonal localized orbitals often make `S` dense rather than exactly
  block diagonal.

So strict factorization is real but rare.

---

## 8. Low-Rank Off-Block Correction

A more realistic situation is

```math
S = D + E,
```

where:

- `D` is block diagonal after a good orbital permutation;
- `E` contains inter-block couplings;
- `E` may be low rank or numerically compressible.

For one fixed bond-label assignment, the determinant kernel involves matrices of
the form

```math
U_X S U_Y^{T} = U_X D U_Y^{T} + U_X E U_Y^{T}.
```

If `E = AB^{T}` has rank `r`, then one may apply matrix determinant lemma type
rewritings so that one fixed-label evaluation costs

```math
O\!\left(\sum_b n_b^3 + r^3 + r^2 n\right)
```

instead of one dense `O(n^3)` step.

This does **not** remove the `2^m` factor by itself. But it can reduce the
constant significantly if every label assignment reuses the same block
structure and the same low-rank boundary.

So low-rank structure is a plausible exact improvement to the **per-sample**
kernel cost, not yet to the bond-label exponential.

---

## 9. Metric-Aware Graph Dynamic Programming

The more ambitious exact compression route is dynamic programming over a graph
that captures which bond labels are actually coupled through the overlap metric.

The right graph is not just the left-right union of pairings. It must also
encode how the metric `S` couples orbitals belonging to different bonds.

One possible abstraction is:

- vertices: covalent bond labels from the left and right structures;
- edges: two labels are adjacent if their orbitals interact through a
  non-negligible part of the overlap metric, or if they share a small metric
  separator in a chosen block decomposition.

If that graph admits a tree decomposition of width `w`, then an exact dynamic
program is plausible with complexity of the generic form

```math
O(\operatorname{poly}(n)\,\gamma^{\,w}),
```

where `gamma` is a small constant determined by the local separator state.

Two natural state parameterizations exist:

- orientation-state DP: each boundary bond carries a binary orientation state,
  giving a `2^w`-type factor;
- exponent-balance DP: each boundary label carries balance in `{-1,0,+1}`,
  giving a `3^w`-type factor.

The exact base depends on the final derivation. What matters is the structural
message:

- the exponential could move from the total number of covalent bonds `m`
  to the graph separator width `w`.

If `w << m`, that would be a genuine exact acceleration.

At present this is a design direction, not yet a finished algorithm.

---

## 10. Why Union-Graph Alone Is Not Enough

The previously observed obstacle remains important.

The left-right pair-union graph by itself does not guarantee factorization,
because the exact determinant kernel depends on the full orbital overlap metric.

So even if the pairing graph decomposes into multiple alternating cycles or
doubled edges, the overlap still fails to factor exactly whenever the metric
contains inter-component coupling.

Therefore any exact compression criterion must be **metric-aware**.

This is why the code path in
`src/tools/check_union_graph_overlap_blocks.cpp` is still relevant:

- it studies whether support-overlap matrices are nearly block structured after
  organizing orbitals by union-graph components;
- that information is much more useful than pair topology alone.

---

## 11. Can This Reach Generic `O(M^4)`?

At present there is no honest basis for claiming that exact raw-VB overlap can
be evaluated in generic worst-case `O(M^4)` while preserving the current raw
structure basis.

The reason is simple:

- after the single-AGP route failed, the exact problem became a bond-labeled
  projection problem;
- the exact projector is exponential in the number of independent bond labels
  unless some additional structure compresses it.

So the realistic exact-compression targets are parameterized ones such as:

- `O(2^{m_{\max}} \sum_b n_b^3)` for strict block factorization;
- `O(2^m (\sum_b n_b^3 + r^3))` when low-rank off-block corrections help only
  the per-sample cost;
- `O(\operatorname{poly}(n)\,\gamma^w)` if a metric-aware bounded-treewidth
  dynamic program can be derived.

These are meaningful improvements, but they are not the same as a generic
closed-form `O(M^4)` theorem.

---

## 12. Bottom Line

The coefficient-extraction layer is now much clearer.

What is already established:

- exact raw-VB overlap can be written as a bond-labeled determinant
  generating function;
- the exact coefficient can be extracted by hypercube averaging;
- this yields a clean exact algorithm with cost `O(2^m n^3)`;
- the new Python verification script confirms the equality numerically.

What remains open:

- how to compress the `2^m` bond-label projector in exact arithmetic;
- whether metric-aware graph structure can reduce the exponent from `m` to a
  smaller separator width `w`;
- how much low-rank or screened inter-block structure can help in practice.

So the next exact-algorithm problem is now sharply identified:

- not "find another single Pfaffian";
- but "compress the bond-labeled exact projector."
