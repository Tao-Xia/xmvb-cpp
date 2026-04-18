# Closed-Shell Raw-VB Exact AGP Encoding

> Warning
>
> This note should no longer be treated as a valid derivation of
> `raw-VB exact = single AGP`.
>
> The central conjecture in this file was disproved by an explicit coefficient
> comparison. In particular, a closed-shell covalent raw-VB structure such as
> `1-2 3-4` is not reproduced by a single oriented AGP pair matrix `F`.
>
> See:
>
> - `src/pfaffian_vbscf/raw_vb_exact_single_agp_failure.md`
> - `scripts/verify_closed_shell_raw_vb_vs_oriented_agp.py`

## 1. Scope

This note records a concrete route for making the closed-shell
`pfaffian_vbscf` basis exact with respect to the raw-VB structure basis while
keeping the current fast pair-kernel strategy.

The target is:

- exact matrix elements in the closed-shell raw-structure space;
- no determinant expansion inside each structure;
- preservation of the current orbital-scaling philosophy, namely an
  `O(M^4)` active-space two-electron path for one state pair, where:
  - `M` is the number of active spatial orbitals;
  - `n` is the number of closed-shell pairs, with `n <= M`.

This note does **not** claim that the current production code already satisfies
these conditions. In fact, the current closed-shell `PfState` construction is
not raw-VB exact. The purpose of this note is to explain why, and to describe
the exact closed-shell replacement object.

Relevant current files:

- `src/pfaffian_vbscf/scf/pf_basis_factory.cpp`
- `src/pfaffian_vbscf/scf/pf_structure_pattern_utils.cpp`
- `src/pfaffian_vbscf/kernel/closed_shell_forward_formula.md`
- `src/vb/matrices/full_structure_expander.cpp`
- `src/vb/matrices/full_structure_builder.cpp`

---

## 2. Problem Statement

For a closed-shell raw-VB structure, the determinant oracle path
`xmvb-cpp.exe` uses:

1. raw structure definition;
2. signed determinant expansion;
3. determinant-pair matrix elements;
4. accumulation back to the raw-structure overlap and Hamiltonian matrices.

This is the exact raw-VB structure basis.

The current `run_pf_vbscf` path instead maps one raw structure to one
closed-shell `PfState` through:

1. a decoded bonding pattern;
2. a symmetric `alpha_beta_block`;
3. a normalized pairing matrix;
4. a Pfaffian/trace-projector matrix element formula.

The key issue is that step 2 changes the state itself. The resulting object is
not the same raw-VB structure. Therefore:

- equality in some full spaces does **not** prove basis equivalence;
- mismatch in restricted subspaces is expected once the accidental agreement is
  broken;
- exact raw-structure matrix elements require a different structure-to-state
  encoding.

The question is whether the exact replacement object can still be handled by
the same fast AGP/Pfaffian kernel philosophy.

For closed-shell structures, the answer is yes.

---

## 3. Closed-Shell Raw-VB Structure As Perfect-Pairing GVB

Consider one closed-shell raw-VB structure `X` with `n` singlet pairs. Each
pair is either:

- ionic: `(i, i)`;
- covalent: `(i, j)` with `i != j`.

Define the pair-creation operators:

For an ionic pair `(i, i)`:

```math
g_{ii}^{\dagger} = a_{i\alpha}^{\dagger} a_{i\beta}^{\dagger}.
```

For a covalent singlet bond `(i, j)`:

```math
g_{ij}^{\dagger}
=
a_{i\alpha}^{\dagger} a_{j\beta}^{\dagger}
- a_{j\alpha}^{\dagger} a_{i\beta}^{\dagger}.
```

Then the raw-VB structure is exactly:

```math
|\Phi_X\rangle
=
\prod_{\mu=1}^{n} g_{\mu}^{\dagger} |0\rangle.
```

This is the standard closed-shell perfect-pairing GVB form: one singlet
geminal per bond.

For a raw-VB structure, the different pair operators act on disjoint active
orbitals. Therefore:

- `g_{\mu}^{\dagger} g_{\nu}^{\dagger} = g_{\nu}^{\dagger} g_{\mu}^{\dagger}`
  for `\mu != \nu`;
- `(g_{\mu}^{\dagger})^2 = 0`.

Now define the collective pair operator:

```math
\Gamma_X^{\dagger} = \sum_{\mu=1}^{n} g_{\mu}^{\dagger}.
```

Then:

```math
(\Gamma_X^{\dagger})^n
=
n! \prod_{\mu=1}^{n} g_{\mu}^{\dagger},
```

because only the product containing each geminal exactly once survives.
Therefore:

```math
|\Phi_X\rangle
=
\frac{1}{n!} (\Gamma_X^{\dagger})^n |0\rangle.
```

This is an AGP representation.

So, for a closed-shell raw-VB structure:

- the exact physical object is a perfect-pairing GVB state;
- because the geminals have disjoint support, it is also exactly an AGP.

This is the main structural reason the closed-shell exact route is plausible.

---

## 4. Exact Oriented AGP Pair Matrix

### 4.1 Spatial pair matrix

Write the AGP generator as

```math
\Gamma_X^{\dagger}
=
\sum_{p,q=1}^{M} F_X(p,q)\,
a_{p\alpha}^{\dagger} a_{q\beta}^{\dagger}.
```

The exact closed-shell raw-VB structure is obtained if `F_X` is built as
follows.

For each ionic pair `(i, i)`:

```math
F_X(i,i) \mathrel{+}= 1.
```

For each covalent bond `(i, j)`, choose one canonical orientation. For
definiteness, take `i < j`, and set:

```math
F_X(i,j) \mathrel{+}= 1,
\qquad
F_X(j,i) \mathrel{+}= -1.
```

Then:

```math
\Gamma_X^{\dagger}
=
\sum_{(i,i)\in X}
a_{i\alpha}^{\dagger} a_{i\beta}^{\dagger}

+ \sum_{(i,j)\in X,\ i<j}
\left(
a_{i\alpha}^{\dagger} a_{j\beta}^{\dagger}
- a_{j\alpha}^{\dagger} a_{i\beta}^{\dagger}
\right),
```

which is exactly the raw-VB singlet-geminal generator.

### 4.2 Why the current symmetric prototype is wrong

The current closed-shell prototype uses a symmetric spatial block for a
covalent pair:

```math
F_{\mathrm{sym}}(i,j)=F_{\mathrm{sym}}(j,i)=1.
```

That generates

```math
a_{i\alpha}^{\dagger} a_{j\beta}^{\dagger}
+ a_{j\alpha}^{\dagger} a_{i\beta}^{\dagger},
```

whereas the exact singlet requires the minus sign. So the state changes before
any overlap or Hamiltonian formula is applied.

This is the basis-level origin of the currently observed subspace mismatch.

### 4.3 Example

For the raw structure

```text
1-2 3-4
```

the exact oriented pair matrix is

```math
F_X =
\begin{pmatrix}
0 & 1 & 0 & 0 \\
-1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & -1 & 0
\end{pmatrix}.
```

Then

```math
\Gamma_X^{\dagger} = g_{12}^{\dagger} + g_{34}^{\dagger},
```

and

```math
\frac{1}{2!}(\Gamma_X^{\dagger})^2 |0\rangle
=
g_{12}^{\dagger} g_{34}^{\dagger} |0\rangle,
```

which is the exact raw-VB structure.

---

## 5. Exact Closed-Shell Structure Overlap As AGP Overlap

### 5.1 Spin-orbital embedding

Embed one spatial pair matrix `F` into the usual antisymmetric spin-orbital
matrix:

```math
\mathcal{A}(F)
=
\begin{pmatrix}
0 & F \\
-F^T & 0
\end{pmatrix}.
```

Let the active-space spatial overlap matrix be `S`, and define the block spin
metric:

```math
\Sigma = \operatorname{diag}(S, S).
```

For a bra/ket pair `(X, Y)`, define

```math
K_{XY} = \mathcal{A}(F_X)^T \Sigma \mathcal{A}(F_Y) \Sigma^T.
```

Direct block multiplication gives:

```math
K_{XY}
=
\begin{pmatrix}
F_X S F_Y^T S & 0 \\
0 & F_X^T S F_Y S
\end{pmatrix}.
```

Define the spatial AGP kernel

```math
M_{XY} = F_X^T S F_Y S.
```

The two diagonal blocks of `K_{XY}` have the same spectrum, so:

```math
\sqrt{\det(I+tK_{XY})} = \det(I+tM_{XY}).
```

Thus the full spin-orbital Pfaffian generating function collapses exactly to an
ordinary spatial determinant generating function.

### 5.2 Overlap coefficient extraction

Let

```math
\Omega_{XY}(t) = \det(I+tM_{XY}) = \sum_{k=0}^{M} c_k^{XY} t^k.
```

Then the exact overlap between `n`-pair AGP states is

```math
S_{XY} = c_n^{XY}.
```

Equivalently, if `m_p = \operatorname{tr}(M_{XY}^p)`, then the coefficients
obey the usual elementary-symmetric-polynomial recursion:

```math
c_0 = 1,
```

```math
c_k =
\frac{1}{k}
\sum_{p=1}^{k}
(-1)^{p+1} m_p\, c_{k-p}.
```

This is the same trace-projector coefficient-extraction problem already used by
the current fast path, only with the exact oriented `F_X, F_Y` in place of the
current symmetric prototype block.

There is no determinant expansion over the `2^n` internal structure terms at
this stage.

---

## 6. Exact Closed-Shell One-Body Matrix Elements

Let

```math
C_{XY} = F_X^T S F_Y,
\qquad
M_{XY} = C_{XY} S.
```

The AGP transition one-body density is obtained by differentiating the overlap
coefficient with respect to the one-electron source. In the same projected
polynomial language as the current implementation, this gives a matrix
polynomial in `M_{XY}` acting on `C_{XY}`.

Using the notation of
`src/pfaffian_vbscf/kernel/closed_shell_forward_formula.md`, the current
closed-shell one-body derivation carries over after the following replacement:

- current prototype block `A` becomes the exact bra-side oriented matrix;
- current prototype block `B` becomes the exact ket-side oriented matrix;
- current pair core `C` becomes `F_X^T S F_Y`;
- current spatial kernel becomes the corresponding `M_{XY}` built from the
  exact oriented matrices.

In particular, the one-body transition density remains a projected matrix
polynomial of order `n-1`, so the one-electron matrix element still comes from
an `O(M^3)` to `O(M^4)` algebraic path rather than from determinant expansion.

So the exact closed-shell AGP encoding creates no obvious one-body scaling
obstruction.

---

## 7. Two-Electron Fast Path And `O(M^4)` Scaling

### 7.1 What must be true

To preserve the current fast-path philosophy, the exact closed-shell
two-electron matrix element between structures `X` and `Y` must remain
expressible through a finite set of projected pair-density-like matrices built
from:

- `F_X`;
- `F_Y`;
- `S`;
- projected polynomials in `M_{XY}`.

If that is true, then the final contraction with the active-space two-electron
tensor still has the same structure as the current production kernel:

- build a small number of `M x M` intermediates;
- contract them against `g_{pq,rs}`;
- avoid internal `2^n` structure expansion.

This keeps the pairwise closed-shell two-electron cost at `O(M^4)` with the
explicit active-space tensor, exactly as in the current production target.

### 7.2 Why this is plausible

The current production two-electron collapse already works entirely in terms of
projected pair densities and bridge corrections, not in terms of explicit
determinant enumeration. That is a strong sign that the real algebraic object
is the AGP transition density, not the current symmetric prototype itself.

Since the exact closed-shell raw-VB structure is also an AGP, there is no
representation-level reason the same collapse philosophy must fail.

Put differently:

- the current symmetric `alpha_beta_block` is the wrong **state**;
- but the current kernel architecture may still be the right **algorithmic
  form** once the state is replaced by the exact oriented AGP matrix.

### 7.3 What still needs rederivation

This part is the main unresolved theoretical task.

The following statements are already firm:

- exact closed-shell raw-VB structure = perfect-pairing GVB;
- for disjoint pair support, that state = AGP;
- overlap coefficient extraction stays in the same determinant/trace-projector
  family;
- one-body projected-density machinery stays in the same polynomial family.

The part that still must be checked carefully is:

- whether the current collapsed closed-shell total two-electron formula in
  `src/pfaffian_vbscf/kernel/` remains valid for fully general oriented
  opposite-spin pair matrices `F_X, F_Y`, rather than only for the current
  symmetric prototype.

The expected answer is yes, but that step must be rederived explicitly.

---

## 8. Relation To GVB

The phrase "introduce GVB's AGP" should be interpreted carefully.

For the present closed-shell raw-structure problem:

- GVB is the natural physical interpretation of one raw structure;
- AGP is the compact algebraic form that becomes available because the pair
  support is disjoint;
- therefore we do not need a new many-geminal runtime object for one raw
  structure;
- it is enough to encode the structure by its exact oriented AGP pair matrix
  `F`.

So the role of GVB here is conceptual rather than algorithmically expensive:

- it tells us what the correct local bond object is;
- AGP then lets us keep the global fast kernel.

This is precisely why the closed-shell case is promising.

---

## 9. Consequences For The Current Code

For closed-shell raw-structure exactness, the current production issue is not
the pair-kernel complexity. The immediate issue is the structure encoding in:

- `src/pfaffian_vbscf/scf/pf_basis_factory.cpp`
- `src/pfaffian_vbscf/scf/pf_structure_pattern_utils.cpp`

Specifically:

1. covalent bonds must no longer be encoded as symmetric spatial adjacency;
2. the exact oriented singlet-geminal sign structure must be preserved;
3. the resulting AGP pair matrix must be treated as the actual bra/ket basis
   state whose matrix elements are evaluated.

After that replacement, the kernel layer should be rechecked in this order:

1. overlap exactness against determinant-expanded raw-VB structure overlaps;
2. one-electron exactness;
3. total two-electron exactness;
4. adjoint/gradient consistency.

---

## 10. Bottom Line

For closed-shell raw-VB structures:

- exact raw-VB structure basis and AGP/Pfaffian representation are **not**
  contradictory;
- the contradiction only appears if one uses the wrong pair matrix, namely the
  current symmetric prototype;
- with the exact oriented pair matrix `F_X`, one raw closed-shell structure is
  exactly an AGP;
- therefore an exact closed-shell raw-VB Pfaffian-VBSCF path is theoretically
  compatible with the current `O(M^4)` fast-kernel strategy.

The real remaining task is not to invent a new scaling argument, but to
rederive the current closed-shell forward and adjoint formulas for generic
oriented AGP pair matrices and to reconnect them to the production code.
