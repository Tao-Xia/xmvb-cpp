# Blocked Open-Shell Projected-Pfaffian Framework

## 1. Scope of This Draft

This document records the recommended mathematical direction for extending the
current closed-shell `pfaffian_vbscf` kernel to open-shell systems.

It is intentionally written as a design draft rather than a finished proof.
The main purpose is to identify the right organizing variables before code is
written.

The central recommendation is:

- first develop an exact fixed-`M_s` open-shell kernel;
- treat blocked open-shell electrons explicitly;
- keep the existing 1D pair-number projection for the singlet-pair sector;
- postpone spin adaptation to a later outer recoupling layer.

The intended first target is high-spin `M_s = S`, followed by generic fixed
`M_s`, and only then spin-adapted pure-`S` states.

---

## 2. Why the Closed-Shell Formula Cannot Be Extended by Simple Patching

The current production theory relies on the singlet `AB/BA` structure:

- one pair block `A = L_{ba}`;
- one pair block `B = R_{ab}`;
- one spatial overlap `S`;
- one effective kernel block `G = S A S B`.

That is a very special collapse. It should be regarded as a fortunate closed-
shell simplification, not as the expected normal form for arbitrary spin.

For open-shell systems, the correct invariant is not "one matrix again". The
correct invariant is:

- a finite family of `m x m` spin blocks and low-rank blocked-orbital objects;
- `O(n M^3)` polynomial work inside those blocks;
- a fixed number of `O(M^4)` two-electron contractions.

In other words, the right goal is constant-size spin-block algebra, not
single-matrix collapse.

---

## 3. Recommended First Physical Target

The first target should be a fixed-`M_s` open-shell VB structure consisting of:

- `n_p` singlet pairs;
- `n_u^\alpha` blocked alpha electrons;
- `n_u^\beta` blocked beta electrons;
- total active electrons
  `n_act = 2 n_p + n_u^\alpha + n_u^\beta`.

For the first production implementation, take

- `n_u^\beta = 0`,
- `n_u^\alpha = 2S`,
- i.e. high-spin `M_s = S`.

This already covers chemically important radicals and broken-bond high-spin
limits while avoiding the extra complexity of general spin recoupling.

---

## 4. Recommended State Representation

## 4.1 Structural decomposition

Each fixed-`M_s` VB state should be represented by two logically distinct
objects:

- a singlet-pair sector;
- a blocked open-shell sector.

Recommended notation for state `I`:

- `P_I` : pair descriptor for the singlet-pair sector;
- `U_I^\alpha \in R^{M x n_u^\alpha}` : blocked alpha orbital coefficients;
- `U_I^\beta \in R^{M x n_u^\beta}` : blocked beta orbital coefficients.

The key point is that open-shell electrons should not be hidden inside a single
packed antisymmetric state matrix. They should be explicit low-rank objects.

## 4.2 Coding implication

The present
`src/pfaffian_vbscf/data/pf_state.hpp`
is too small for this next phase. A future state object should separate:

- pair-sector data;
- blocked-alpha data;
- blocked-beta data;
- later optional spin-coupling labels for the spin-adapted layer.

This is not only a coding preference. It reflects the actual algebra needed by
the open-shell theory.

---

## 5. Recommended Kernel Ansatz: Low-Rank Blocked Reduction

The most promising route is to treat the blocked open-shell sector as a low-
rank boundary attached to the projected singlet-pair kernel.

## 5.1 Blocked overlap matrix

For one pair of states `I, J`, define the blocked-orbital overlap matrices

```math
O_{IJ}^{\alpha\alpha} = (U_I^\alpha)^T S U_J^\alpha,
\qquad
O_{IJ}^{\beta\beta} = (U_I^\beta)^T S U_J^\beta,
```

and, for the general fixed-`M_s` case, also the mixed blocked overlaps when
they are required by the chosen state ordering.

For the first high-spin implementation with only blocked alpha electrons, the
relevant boundary object is simply

```math
O_{IJ}^{u} = (U_I^\alpha)^T S U_J^\alpha.
```

## 5.2 Effective paired metric by Schur complement

The working hypothesis for the first exact theory is that the blocked sector
can be removed analytically, leaving an effective metric for the singlet-pair
subspace. The generic form should be a low-rank update of `S`,

```math
\widetilde S_{IJ}
=
S - S U_J X_{IJ} U_I^T S,
```

where `U_I, U_J` denote the relevant blocked-orbital matrices and `X_{IJ}` is
the inverse or pseudoinverse of the blocked overlap block.

For the simplest high-spin case one expects

```math
\widetilde S_{IJ}
=
S - S U_J^\alpha (O_{IJ}^{u})^{-1} (U_I^\alpha)^T S.
```

Interpretation:

- the blocked open-shell electrons are integrated out by a Schur complement;
- the singlet-pair sector then propagates in the complement of the blocked
  overlap manifold;
- closed-shell theory is recovered when the blocked sector is empty.

This is the most important structural ansatz in the whole roadmap.

## 5.3 Effective pair kernel

Once the blocked sector is removed, the remaining paired sector should still be
organized by a pair-number generating function. The proposed effective kernel
is

```math
\widetilde K_{IJ}
=
L_I \widetilde\Sigma_{IJ} R_J \widetilde\Sigma_{IJ}^{T},
\qquad
\widetilde\Sigma_{IJ} = \operatorname{diag}(\widetilde S_{IJ}, \widetilde S_{IJ}),
```

or the corresponding spin-block generalization when both blocked alpha and
blocked beta electrons are present.

The overlap should then factor into:

- a blocked-sector determinant or Pfaffian prefactor;
- the same 1D projected pair overlap built from `\widetilde K_{IJ}`.

A candidate high-spin formula is therefore

```math
\mathcal S_{IJ}
=
\det(O_{IJ}^{u})
\,[t^{n_p}]\,\sqrt{\det(I + t \widetilde K_{IJ})}.
```

This formula must be validated against determinant-expanded references before
any production implementation is trusted, but it is the cleanest starting
ansatz currently visible.

---

## 6. Transition 1-RDM: Spectator + Pair-Deformed Decomposition

Once the overlap is written as blocked prefactor times projected pair kernel,
the transition 1-RDM should be decomposed into two contributions.

## 6.1 Blocked spectator term

The blocked electrons contribute a low-rank transition density

```math
\Gamma_{IJ}^{\mathrm{blk}}
=
U_J X_{IJ} U_I^T,
```

with the obvious alpha/beta spin resolution.

For the first high-spin case,

```math
\Gamma_{IJ}^{\mathrm{blk},\alpha\alpha}
=
U_J^\alpha (O_{IJ}^{u})^{-1} (U_I^\alpha)^T,
\qquad
\Gamma_{IJ}^{\mathrm{blk},\beta\beta}=0.
```

## 6.2 Paired contribution in the blocked complement

The paired electrons should then contribute the same projected matrix-
polynomial density as in the closed-shell case, but built from the effective
metric and effective kernel:

```math
\Gamma_{IJ}^{\mathrm{pair}}
=
\text{projected polynomial of }
\widetilde K_{IJ}
\text{ and }
\widetilde C_{IJ}.
```

The exact open-shell block formula remains to be derived, but the expected
structure is:

- same `P_r`, `R_r`, `F_r` polynomial machinery;
- more than one spin block in the generic fixed-`M_s` case;
- closed-shell reduction when `U^\alpha = U^\beta = 0`.

## 6.3 Mixed blocked-pair terms

The main open question is whether a separate blocked-pair mixed 1-RDM term
survives after the Schur complement is applied.

The recommended working plan is:

- derive the expression from the generating overlap before implementation;
- do not assume those mixed terms vanish until verified;
- if they do survive, treat them as low-rank `O(M^2 n_u)` or `O(M^3)` updates,
  not as a new high-scaling component.

---

## 7. One-Electron Matrix Element

Once the transition 1-RDM is known, the one-electron part stays conceptually
simple:

```math
H_{IJ}^{(1)} = \langle \Gamma_{IJ}, h \rangle_F.
```

The main theoretical work is therefore not in the final contraction but in the
correct derivation of `\Gamma_{IJ}` for blocked open-shell states.

From an implementation viewpoint this is good news:

- one-electron open-shell support should be completed at the same time as
  overlap and 1-RDM;
- it should not wait for the full two-electron derivation.

---

## 8. Two-Electron Matrix Element: Recommended Decomposition

The full open-shell two-electron matrix element should be decomposed into three
classes:

1. blocked-blocked contributions;
2. pair-pair contributions;
3. pair-blocked cross contributions.

## 8.1 Blocked-blocked part

This is the determinant-like open-shell spectator sector. For high-spin states
it should look like ordinary Coulomb/exchange built from the blocked transition
density.

Scaling target:

- `O(M^4)` via the same packed integral contraction backend already present in
  `pf_tensor_contractor`.

## 8.2 Pair-pair part

This is the natural deformation of the current closed-shell fast path.

Expected structure:

- projected pair-density objects built from `\widetilde K_{IJ}`;
- same matrix-polynomial hierarchy as the current singlet kernel;
- more spin blocks, but still a finite family.

This should be viewed as the open-shell analogue of the current
`pair + cross + same-spin bridge + opposite-spin bridge` decomposition.

## 8.3 Pair-blocked cross part

This is where `op_spin.md` becomes critical.

The essential point already established there is:

- after exchanging the order of four-index summation and trace algebra,
  the cross terms collapse into a finite set of bilinear `\mathcal W(A,B)`
  contractions over `m x m` blocks;
- therefore the cross sector is compatible with `O(M^4)`.

This means the open-shell problem is not fundamentally blocked by two-electron
scaling. It is blocked mainly by the correct organization of the forward
intermediates.

---

## 9. Adjoint Strategy

The reverse sweep should mirror the forward factorization.

Recommended reverse order:

1. packed tensor-contraction adjoints;
2. block-matrix polynomial adjoints;
3. effective-kernel adjoints;
4. Schur-complement / blocked-overlap adjoints;
5. pair-sector trace-recursion adjoints;
6. final derivatives with respect to `S`, `h`, `g`, and later orbital-rotation
   parameters.

## 9.1 Why the adjoint is still manageable

The proposed forward graph has only three types of nodes:

- low-rank blocked overlaps and their inverses;
- matrix polynomials in a constant-size family of `m x m` blocks;
- packed `O(M^4)` tensor contractions.

Each of these classes already has a standard reverse-mode structure:

- matrix inverse / Schur complement adjoints are textbook;
- polynomial adjoints already exist in the current code;
- packed tensor-contraction adjoints already exist in the current code.

Therefore the open-shell gradient problem is difficult but structurally well
posed.

---

## 10. Proposed Code Architecture Corresponding to This Theory

## 10.1 State objects

Future `PfState`-like objects should carry:

- pair-sector data;
- blocked alpha orbitals;
- blocked beta orbitals;
- later optional spin-coupling metadata.

## 10.2 Kernel cache

Replace the closed-shell-specific cache mindset by a layered cache:

- generic spin-orbital kernel data;
- blocked overlap / inverse blocks;
- effective-metric blocks;
- projected pair-sector polynomial blocks;
- optional specialized closed-shell shortcuts.

## 10.3 Forward kernel classes

The natural class separation is:

- generic fixed-`M_s` cache builder;
- fixed-`M_s` forward evaluator;
- fixed-`M_s` adjoint evaluator;
- later spin-adapted recoupling assembler.

The current closed-shell evaluator can remain as a specialized fast path inside
this broader architecture.

---

## 11. Spin Adaptation Should Be an Outer Layer

This is a key design choice.

The first open-shell kernel should compute exact fixed-`M_s` matrix elements.
Then, for spin-adapted states, introduce an outer linear transformation

```math
|\Phi_A^{S}\rangle
=
\sum_\mu C_{\mu A}^{(S)} |\Phi_\mu^{M_s}\rangle,
```

and assemble

```math
\mathcal H_{AB}^{(S)}
=
\sum_{\mu\nu}
C_{\mu A}^{(S)} C_{\nu B}^{(S)}
\mathcal H_{\mu\nu}^{(M_s)}.
```

Why this is the right order:

- the expensive orbital kernel is reused unchanged;
- spin adaptation remains a small algebraic layer;
- the `O(M^4)` orbital scaling is protected from spin-coupling combinatorics.

---

## 12. Most Important Open Questions

The following points must be resolved explicitly before implementation is
considered final.

### Q1. Exact blocked-overlap factorization

Can the fixed-`M_s` overlap be reduced exactly to:

- a low-rank blocked prefactor;
- a closed-shell-like projected pair kernel in an effective metric?

This is the central theoretical checkpoint.

### Q2. Mixed 1-RDM terms after blocked elimination

Do explicit blocked-pair mixed terms remain after the Schur complement, or are
they fully absorbed into the effective pair kernel plus spectator density?

### Q3. Minimal block set for the open-shell two-electron fast path

What is the smallest closed family of `m x m` blocks needed to express all
pair-pair and pair-blocked open-shell contributions?

### Q4. Best production representation of blocked orbitals

Should blocked open-shell electrons be stored as:

- orbital index lists for structure-built localized states;
- full coefficient matrices;
- or both, with index lists used only for structure construction?

The likely answer is: coefficient matrices in the kernel layer, index lists
only in the structure factory.

---

## 13. Recommended Near-Term Deliverable

The immediate mathematical deliverable should be a fixed-`M_s`, high-spin
proof-of-concept with:

- exact overlap;
- exact transition 1-RDM;
- exact one-electron matrix element;
- exact full two-electron matrix element in `O(M^4)`;
- exact active-space adjoint;
- no spin adaptation yet.

Once that is working, the path to generic fixed-`M_s` and later spin-adapted
Pfaffian-VBSCF will be much clearer.

That is the recommended foundation for the next implementation stage.
