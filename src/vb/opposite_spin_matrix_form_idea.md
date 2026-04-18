# Opposite-Spin Matrix-Form Rewrite Notes

## 1. Goal

This note records an exact reformulation path for the current opposite-spin
determinant-pair bottleneck in VBSCF/NOCI.

The main question is:

$$
\text{Can we remove the explicit } O(N_{\mathrm{det}}^2) \text{ full determinant-pair sweep?}
$$

The short answer is:

1. Within the current pair-driven framework, adding more cache does not remove
   the outer determinant-pair sweep. It only reduces the per-pair kernel cost.
2. An exact matrix-form rewrite does exist for the current selected-state
   objective, because the determinant coefficients can be reorganized into
   an $\alpha \times \beta$ coefficient matrix, and the RI opposite-spin kernel
   already has a separable form.
3. This rewrite does **not** automatically imply
   $O(N_{\alpha}^2 + N_{\beta}^2)$. The realistic exact dense cost is instead
   matrix-multiplication-like, typically
   $O\!\left(N_{\mathrm{aux}} \left(N_{\alpha}^2 N_{\beta} +
   N_{\alpha} N_{\beta}^2 \right)\right)$ per state.
4. Reaching something close to
   $O\!\left(r \left(N_{\alpha}^2 + N_{\beta}^2 \right)\right)$ would require
   an additional low-rank factorization of the state coefficient matrix
   $C^{(n)}$, which is a separate assumption.

## 2. Current Code Picture

The current implementation still performs an explicit full determinant-pair
sweep in the active-space backward path:

- `src/vb/scf/cpp_active_space_gradient_evaluator.cpp`
- `accumulate_active_space_gradient_unordered_pair(...)`
- outer loop over unordered determinant pairs

The current opposite-spin cache path already does the following:

1. For each unique ordered same-spin pair, build one reusable opposite-spin
   projection payload.
2. Reuse those payloads in:
   - forward opposite-spin scalar contraction,
   - backward opposite-spin $\phi$,
   - backward opposite-spin two-electron gradient accumulation.

Therefore, the main unresolved bottleneck is **not** the raw opposite-spin
kernel anymore. The unresolved bottleneck is the fact that the global objective
and adjoint are still accumulated in full determinant-pair space.

## 3. Notation

Let:

- $s \in \{1, \dots, N_{\mathrm{str}}\}$ be a structure index,
- $d \in \{1, \dots, N_{\mathrm{det}}\}$ be a full determinant index,
- $a \in \{1, \dots, N_{\alpha}\}$ be a unique $\alpha$ string index,
- $b \in \{1, \dots, N_{\beta}\}$ be a unique $\beta$ string index,
- $n \in \mathcal{N}$ be one selected state,
- $P \in \{1, \dots, N_{\mathrm{aux}}\}$ be an RI auxiliary index.

Let the determinant expansion coefficients be:

$$
T_{d s},
$$

where $T_{d s}$ is the coefficient of determinant $d$ inside structure $s$.

Let the selected-state structure eigenvector coefficients be:

$$
u_{s}^{(n)}.
$$

Then the determinant coefficient of state $n$ is

$$
c_{d}^{(n)} = \sum_{s=1}^{N_{\mathrm{str}}} T_{d s} \, u_{s}^{(n)}.
$$

Since the full determinant expansion already deduplicates the pair
$(\alpha, \beta)$, every full determinant can be identified with one pair
$(a,b)$. Therefore we can reshape the state coefficient vector into a matrix

$$
C^{(n)} \in \mathbb{R}^{N_{\alpha} \times N_{\beta}},
$$

with entries

$$
C_{a b}^{(n)} =
\begin{cases}
c_{d}^{(n)}, & \text{if determinant } d \leftrightarrow (a,b) \text{ exists}, \\
0, & \text{otherwise}.
\end{cases}
$$

If the determinant space is not the full Cartesian product, then $C^{(n)}$ is a
masked or sparse matrix.

## 4. Exact Kernel Factorization

### 4.1 Overlap and one-electron channels

The determinant-pair overlap factorizes exactly:

$$
S_{(a,b),(a',b')} = S_{a a'}^{\alpha} S_{b b'}^{\beta}.
$$

The one-electron Hamiltonian also factorizes:

$$
H_{(a,b),(a',b')}^{1e}
=
H_{a a'}^{\alpha} S_{b b'}^{\beta}
+
S_{a a'}^{\alpha} H_{b b'}^{\beta}.
$$

Likewise, if the same-spin two-electron kernels are written as
$G_{a a'}^{\alpha}$ and $G_{b b'}^{\beta}$, then

$$
H_{(a,b),(a',b')}^{\mathrm{ss}}
=
G_{a a'}^{\alpha} S_{b b'}^{\beta}
+
S_{a a'}^{\alpha} G_{b b'}^{\beta}.
$$

These channels already explain why same-spin cache works so well: the kernel is
naturally separable in spin space.

### 4.2 RI opposite-spin channel

Under RI, the opposite-spin two-electron contribution can be written as

$$
H_{(a,b),(a',b')}^{\mathrm{os}}
=
\sum_{P=1}^{N_{\mathrm{aux}}}
J_{a a'}^{\alpha}(P) \,
J_{b b'}^{\beta}(P).
$$

Here:

- $J_{a a'}^{\alpha}(P)$ is the RI-projected first-cofactor or inverse-overlap
  payload for the ordered $\alpha$ pair $(a,a')$,
- $J_{b b'}^{\beta}(P)$ is the analogous payload for the ordered $\beta$ pair.

This is exactly the mathematical content behind the current opposite-spin
projection cache: the expensive four-index contraction is first reduced to
single-spin payload construction, and then combined by a much cheaper
cross-spin contraction.

## 5. Exact Forward Reformulation

For one selected state $n$, the opposite-spin contribution to the state
expectation value is

$$
E_{n}^{\mathrm{os}}
=
\sum_{a,a'=1}^{N_{\alpha}}
\sum_{b,b'=1}^{N_{\beta}}
C_{a b}^{(n)}
H_{(a,b),(a',b')}^{\mathrm{os}}
C_{a' b'}^{(n)}.
$$

Substitute the RI form:

$$
E_{n}^{\mathrm{os}}
=
\sum_{P=1}^{N_{\mathrm{aux}}}
\sum_{a,a'=1}^{N_{\alpha}}
\sum_{b,b'=1}^{N_{\beta}}
C_{a b}^{(n)}
J_{a a'}^{\alpha}(P)
J_{b b'}^{\beta}(P)
C_{a' b'}^{(n)}.
$$

Rearrange the sums:

$$
E_{n}^{\mathrm{os}}
=
\sum_{P=1}^{N_{\mathrm{aux}}}
\operatorname{Tr}
\left(
\left[C^{(n)}\right]^{\top}
J^{\alpha}(P)
C^{(n)}
\left[J^{\beta}(P)\right]^{\top}
\right).
$$

This is an exact identity, not an approximation beyond RI itself.

The same state can also be written in matrix form for the overlap and
same-spin channels:

$$
E_{n}^{S}
=
\operatorname{Tr}
\left(
\left[C^{(n)}\right]^{\top}
S^{\alpha}
C^{(n)}
\left[S^{\beta}\right]^{\top}
\right),
$$

$$
E_{n}^{1e}
=
\operatorname{Tr}
\left(
\left[C^{(n)}\right]^{\top}
H^{\alpha}
C^{(n)}
\left[S^{\beta}\right]^{\top}
\right)
+
\operatorname{Tr}
\left(
\left[C^{(n)}\right]^{\top}
S^{\alpha}
C^{(n)}
\left[H^{\beta}\right]^{\top}
\right),
$$

$$
E_{n}^{\mathrm{ss}}
=
\operatorname{Tr}
\left(
\left[C^{(n)}\right]^{\top}
G^{\alpha}
C^{(n)}
\left[S^{\beta}\right]^{\top}
\right)
+
\operatorname{Tr}
\left(
\left[C^{(n)}\right]^{\top}
S^{\alpha}
C^{(n)}
\left[G^{\beta}\right]^{\top}
\right).
$$

For a state-averaged objective with selected-state weights $w_n$,

$$
E_{\mathrm{SA}}^{\mathrm{os}}
=
\sum_{n \in \mathcal{N}} w_n E_{n}^{\mathrm{os}}.
$$

Therefore the current pair-weight tensor is not an arbitrary dense object. For
the selected-state objective, it is generated by a sum over states and can be
reorganized through the coefficient matrices $C^{(n)}$.

## 6. Exact Backward Reformulation

Define

$$
f(C; A, B) = \operatorname{Tr}(C^{\top} A C B^{\top}).
$$

Then the exact matrix derivatives are

$$
\frac{\partial f}{\partial C}
=
A C B^{\top} + A^{\top} C B,
$$

$$
\frac{\partial f}{\partial A}
=
C B^{\top} C^{\top},
$$

$$
\frac{\partial f}{\partial B}
=
C^{\top} A C.
$$

Apply this to the opposite-spin state-average objective:

$$
E_{\mathrm{SA}}^{\mathrm{os}}
=
\sum_{n \in \mathcal{N}} w_n
\sum_{P=1}^{N_{\mathrm{aux}}}
\operatorname{Tr}
\left(
\left[C^{(n)}\right]^{\top}
J^{\alpha}(P)
C^{(n)}
\left[J^{\beta}(P)\right]^{\top}
\right).
$$

Then

$$
\frac{\partial E_{\mathrm{SA}}^{\mathrm{os}}}{\partial C^{(n)}}
=
w_n
\sum_{P=1}^{N_{\mathrm{aux}}}
\left(
J^{\alpha}(P) C^{(n)} \left[J^{\beta}(P)\right]^{\top}
+
\left[J^{\alpha}(P)\right]^{\top} C^{(n)} J^{\beta}(P)
\right),
$$

$$
\frac{\partial E_{\mathrm{SA}}^{\mathrm{os}}}{\partial J^{\alpha}(P)}
=
\sum_{n \in \mathcal{N}} w_n
C^{(n)}
\left[J^{\beta}(P)\right]^{\top}
\left[C^{(n)}\right]^{\top},
$$

$$
\frac{\partial E_{\mathrm{SA}}^{\mathrm{os}}}{\partial J^{\beta}(P)}
=
\sum_{n \in \mathcal{N}} w_n
\left[C^{(n)}\right]^{\top}
J^{\alpha}(P)
C^{(n)}.
$$

These equations show that the backward pass can also be reorganized into matrix
contractions. The remaining task is then to map

$$
\frac{\partial E}{\partial J^{\alpha}(P)}, \qquad
\frac{\partial E}{\partial J^{\beta}(P)}
$$

back to the orbital-space gradients through the same opposite-spin projection
chain that the current implementation already uses.

In other words:

1. the current code already knows how to build the single-spin payloads,
2. the missing step is to accumulate their adjoints in matrix form instead of
   full determinant-pair form.

## 7. Why This Is Better Than More Pair Cache

Suppose we cache the final four-index scalar

$$
\mathcal{H}_{a a' b b'}^{\mathrm{os}}.
$$

This avoids recomputing the final scalar contraction for that specific
$(a,a',b,b')$ tuple, but it does **not** remove the outer determinant-pair
sweep. After full determinant deduplication, a tuple $(a,b)$ is already a full
determinant identity. Therefore a four-index scalar cache is essentially a
renamed full-pair cache.

This means:

1. it can reduce a constant factor,
2. it does not change the global asymptotic structure,
3. it does not address the real bottleneck in the backward sweep.

## 8. Complexity Boundaries

### 8.1 Pair-driven formulation

If the full determinant space is close to the Cartesian product, then

$$
N_{\mathrm{det}} \approx N_{\alpha} N_{\beta},
$$

and the explicit determinant-pair sweep scales like

$$
O(N_{\mathrm{det}}^2)
=
O(N_{\alpha}^2 N_{\beta}^2)
$$

before accounting for the inner kernel cost.

Current same-spin and opposite-spin payload caches reduce the kernel cost per
pair, but do not remove this outer topology.

### 8.2 Dense matrix-form exact formulation

For one state $n$ and one auxiliary index $P$, the dominant operations are
matrix multiplications such as

$$
J^{\alpha}(P) C^{(n)},
\qquad
\left[C^{(n)}\right]^{\top} J^{\alpha}(P) C^{(n)}.
$$

The dense exact cost is therefore on the order of

$$
O\!\left(
N_{\alpha}^2 N_{\beta}
+
N_{\alpha} N_{\beta}^2
\right)
$$

per $(n,P)$, giving

$$
O\!\left(
N_{\mathrm{states}}
N_{\mathrm{aux}}
\left(
N_{\alpha}^2 N_{\beta}
+
N_{\alpha} N_{\beta}^2
\right)
\right).
$$

This is the realistic exact dense target, not

$$
O(N_{\alpha}^2 + N_{\beta}^2).
$$

### 8.3 Sparse or masked coefficient matrices

If the determinant space is not the full Cartesian product, define the fill
factor

$$
\rho =
\frac{N_{\mathrm{det}}}{N_{\alpha} N_{\beta}}.
$$

Then $C^{(n)}$ is sparse or masked, and the actual cost depends strongly on
whether the matrix-form implementation uses:

1. dense GEMM on a mostly empty rectangle,
2. sparse-dense multiplication,
3. masked block contractions.

Therefore the practical payoff depends on the observed fill factor $\rho$.

### 8.4 When can we approach
$O(r (N_{\alpha}^2 + N_{\beta}^2))$?

Only if we also have a low-rank factorization

$$
C^{(n)} \approx \sum_{k=1}^{r} u_{k}^{(n)} \left[v_{k}^{(n)}\right]^{\top}.
$$

Then the trace contractions can be reduced further. But that is an additional
assumption about the state coefficient matrix itself, not a consequence of RI
alone.

## 9. Practical Interpretation for This Code Base

For the current code base, the most important practical conclusions are:

1. The same-spin cache is already doing the right thing.
2. The current opposite-spin payload cache is also doing the right thing at the
   kernel level.
3. The remaining bottleneck is the full determinant-pair backward sweep, not
   the lack of one more scalar cache.
4. The next exact algorithmic step is to form state coefficient matrices
   $C^{(n)}$ after the eigensolve and rewrite the opposite-spin backward path
   as matrix contractions.
5. Forward structure assembly is now much less urgent than the active-space
   adjoint path.

This also suggests a staged rewrite plan:

1. Build $C^{(n)}$ explicitly for the selected states after the generalized
   eigensolve.
2. Measure the fill factor $\rho$ and decide whether dense or sparse matrix
   kernels are appropriate.
3. Prototype the opposite-spin forward contraction in matrix form and compare it
   against the current pair-driven result.
4. Derive and implement the matrix-form backward adjoints
   $\partial E / \partial C^{(n)}$,
   $\partial E / \partial J^{\alpha}(P)$, and
   $\partial E / \partial J^{\beta}(P)$.
5. Reuse the existing projection-payload chain to map the matrix-form adjoints
   back to orbital-space gradients.

## 10. Relation to Existing Repository Experiments

The repository already contains exploratory exact-separator code under

- `src/vb/exact_separator/`

which indicates that matrix-form or separator-form exact contractions are
already of interest in this code base.

This note is narrower in scope:

1. it focuses on the current production RI determinant-pair implementation,
2. it identifies the exact matrix-form reorganization enabled by the existing
   selected-state objective,
3. it explains why another pair-indexed opposite-spin scalar cache is not the
   main path to asymptotic speedup.
