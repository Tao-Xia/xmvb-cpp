# Exact-Integral TNHVP Post-Tile-Contraction Bottleneck Analysis And Optimization Plan

> Status note (2026-04-21):
> this note is the follow-up planning document after the unique-spin
> structure-space tile-contraction rewrite and packed-gradient striped
> reduction.
> The scope here is the current exact-integral path only.
> RI / Cholesky / THC are intentionally out of scope.

Related notes:

- [src/vb/unique_spin_structure_contraction.md](../src/vb/unique_spin_structure_contraction.md)
- [docs/query_driven_unique_spin_contraction.md](./query_driven_unique_spin_contraction.md)
- [docs/outer_response_linear_response_predecomposition.md](./outer_response_linear_response_predecomposition.md)

## 1. Purpose

The tile-contraction rewrite already solved one important problem:

$$
\text{the unique-spin-string } \to \text{ structure-basis contraction is no longer}
\quad
\text{the dominant exact-integral bottleneck.}
$$

That does **not** mean the exact-integral TNHVP path is now cheap.
It means the main optimization target has shifted.

The goal of this note is to write down:

1. what the current `exact_ctx` / TNHVP wall-time is really spent on;
2. why `outer_response` timing can stay nearly flat between two systems with
   the same active space even when AO basis size grows;
3. which next optimizations are still "tile-contraction-class" structural
   rewrites, rather than only constant-factor cleanup;
4. what order those rewrites should follow.

---

## 2. Empirical Facts From The Current Benchmarks

The current comparison is between:

- `test/241_VBSCF.xmi`
- `test/10698_VBSCF.xmi`

Lightweight deck inspection shows:

| system | basis | atoms | declared active electrons | declared active orbitals | orbital support entries |
| --- | --- | ---: | ---: | ---: | ---: |
| `241_VBSCF` | `cc-pvdz` | 12 | 6 | 6 | 24 |
| `10698_VBSCF` | `cc-pvdz` | 22 | 6 | 6 | 40 |

So both systems are on the same declared active space:

$$
\mathrm{CAS}(6,6),
\qquad
n_a = 6,
\qquad
n_{\mathrm{pair}}^{\mathrm{act}} = \frac{n_a(n_a+1)}{2} = 21.
$$

The `benchmark_exact_ctx_hvp` runs gave:

| metric | `241_VBSCF` | `10698_VBSCF` |
| --- | ---: | ---: |
| reduced dimension | 456 | 764 |
| `full_cached_external` | 3.950 s | 12.944 s |
| `h1e_fused` | 1.504 s | 9.949 s |
| `active_2e` | 0.214 s | 0.803 s |
| `outer_response` | 2.227 s | 2.183 s |
| `outer_response_active_space_integrals` | 0.190 s | 0.764 s |
| `outer_response_structure_matrices` | 0.556 s | 0.123 s |
| `outer_response_active_gradient` | 1.335 s | 0.749 s |
| `outer_response_orbital_pullback` | 0.146 s | 0.548 s |

The most important observation is:

$$
T_{\mathrm{outer}}(241) \approx T_{\mathrm{outer}}(10698),
$$

even though

$$
T_{\mathrm{full}}(10698) \approx 3.28 \, T_{\mathrm{full}}(241).
$$

This does **not** mean `outer_response` has become free.
It means that after the tile-contraction rewrite:

1. the structure-side contraction no longer scales primarily with AO basis
   size;
2. the AO-size-dependent parts of `outer_response` are now concentrated in the
   directional active-space integral build and the final orbital pullback;
3. the total `outer_response` time is the sum of several terms with different
   scaling variables, so two systems with the same active space can easily
   have similar totals even when the AO basis size differs.

This matches the physical intuition:

$$
\text{same active space}
\Rightarrow
\text{similar active-pair dimension}
\Rightarrow
\text{similar selected-state / support-side work scale,}
$$

but it does **not** remove the AO-basis dependence of the integral-side stages.

---

## 3. Current Code-Level Decomposition

The current exact-integral HVP apply is timed in
[src/vb/scf/exact_orbital_second_order_operator.cpp](../src/vb/scf/exact_orbital_second_order_operator.cpp)
through the decomposition

$$
T_{\mathrm{HVP}}
=
T_{\mathrm{core}} + T_{\mathrm{outer}}.
$$

More explicitly:

$$
T_{\mathrm{HVP}}
=
T_{\mathrm{h1e\_fused}}
+ T_{\mathrm{act2e}}
+ T_{\mathrm{orbital\_backprop}}
+ T_{\mathrm{fixed\_upstream}}
+ T_{\mathrm{outer}}.
$$

The current `outer_response` part is further split as

$$
T_{\mathrm{outer}}
=
T_{\delta \eta}
+ T_{\mathcal A}
+ T_{\mathcal R}
+ T_{W}
+ T_{T^\ast},
$$

where:

- \(T_{\delta \eta}\): build active-space directional integrals
  `build_active_space_directional_integrals(...)`;
- \(T_{\mathcal A}\): build projected directional structure matrices
  `build_projected_directional_structure_matrices(...)`;
- \(T_{\mathcal R}\): selected-state generalized-eigen response;
- \(T_{W}\): build outer-response active-space gradient direction
  `build_active_space_gradient_direction_from_outer_response(...)`;
- \(T_{T^\ast}\): pull the active-space gradient back to the orbital chart
  `build_orbital_value_gradient_from_active_space_gradient_direction(...)`.

On one accepted point \(x_k\), the current data flow can be written as

$$
p
\xrightarrow{\,T_k\,}
\delta \eta
\xrightarrow{\,\mathcal A_k\,}
\delta \zeta
\xrightarrow{\,\mathcal R_k\,}
\delta y
\xrightarrow{\,W_k\,}
g_{\mathrm{outer}}
\xrightarrow{\,T_k^\ast\,}
B_{\mathrm{outer}} p,
$$

with:

$$
\delta \eta =
\begin{bmatrix}
\operatorname{vec}(\delta S_{\mathrm{act}}) \\
\operatorname{vec}(\delta h_{\mathrm{act}}) \\
\delta g_{\mathrm{act}}^{(2)}
\end{bmatrix}.
$$

This is the right abstraction for the next round of work:
the current bottlenecks are no longer all on the same variable space.

---

## 4. Why `outer_response` Does Not Grow Like The Total Time

The key point is that the current `outer_response` is not one monolithic
kernel. It is a sum of terms with different scaling variables.

Let:

- \(n_b\): AO basis size;
- \(n_a\): active-orbital count;
- \(n_{\mathrm{pair}}^{\mathrm{act}} = n_a(n_a+1)/2\);
- \(n_{\mathrm{str}}\): structure dimension;
- \(n_{\mathrm{sel}}\): number of selected states;
- \(r_I, c_I\): alpha/beta support sizes of one structure-local block.

Then, schematically,

$$
T_{\mathrm{outer}}
\approx
T_{\delta \eta}(n_b, n_a)
+ T_{\mathcal A}(n_{\mathrm{str}}, n_{\mathrm{sel}}, \{r_I,c_I\})
+ T_{\mathcal R}(n_{\mathrm{str}}, n_{\mathrm{sel}})
+ T_{W}(n_a, n_{\mathrm{sel}}, \{r_I,c_I\})
+ T_{T^\ast}(n_b, n_a).
$$

After the tile-contraction rewrite:

1. \(T_{\mathcal A}\) is no longer governed by dense
   \(N_{\alpha}^2\) / \(N_{\beta}^2\) scans;
2. \(T_{W}\) no longer carries the old `O(n_threads * N_{2e})`
   packed-gradient memory blowup;
3. the AO-size dependence now survives mainly in \(T_{\delta \eta}\) and
   \(T_{T^\ast}\).

That is exactly what the benchmark shows:

- `outer_response_active_space_integrals` grows by about \(4.0\times\);
- `outer_response_orbital_pullback` grows by about \(3.7\times\);
- `outer_response_structure_matrices` and
  `outer_response_active_gradient` do **not** grow with the same trend.

Therefore the near equality

$$
T_{\mathrm{outer}}(241) \approx T_{\mathrm{outer}}(10698)
$$

should be interpreted as:

$$
\text{the structure/support-side contraction is no longer the dominant term,}
$$

not as

$$
\text{the whole outer-response path is already optimal.}
$$

---

## 5. Stage-By-Stage Theoretical Bottleneck Model

### 5.1 `h1e_fused`: AO integral stencil apply

The kernel
`accumulate_fused_ao_effective_one_electron_integral(...)`
processes one AO two-electron integral \((ij|kl)\) at a time and performs a
fixed Coulomb/exchange stencil update on:

- the directional AO effective one-electron matrix;
- the inactive-density adjoint.

Schematically, each integral contributes terms of the form

$$
\Delta F_{ij} \mathrel{+}= 4 D_{kl} (ij|kl),
\qquad
\Delta F_{ik} \mathrel{-}= D_{lj} (ij|kl),
\qquad
\Delta F_{il} \mathrel{-}= D_{kj} (ij|kl),
$$

together with the corresponding adjoint updates for the density-side gradient.

The important feature is not the exact coefficients.
The important feature is:

$$
\text{one AO integral}
\Rightarrow
\text{a constant-size scattered update stencil on large AO matrices.}
$$

So the current complexity is roughly

$$
T_{\mathrm{h1e\_fused}}
=
\mathcal O\!\left(N_{2e}^{\mathrm{AO}}\right)
$$

with low arithmetic intensity and poor cache reuse.

This is why `h1e_fused` is now the largest term:
it is fundamentally a scalar integral-streaming kernel, not yet a blocked
contraction kernel.

### 5.2 Exact active-space directional two-electron integrals

Let

$$
C \in \mathbb{R}^{n_b \times n_a},
\qquad
\Delta C \in \mathbb{R}^{n_b \times n_a}
$$

be the accepted active coefficients and one directional perturbation.

Define the active-pair basis size

$$
p = n_{\mathrm{pair}}^{\mathrm{act}} = \frac{n_a(n_a+1)}{2}.
$$

The current exact two-electron directional builder forms AO-pair-to-active-pair
coefficient maps

$$
P(C) \in \mathbb{R}^{N_{\mathrm{pair}}^{\mathrm{AO}} \times p},
\qquad
\Delta P(C,\Delta C) \in
\mathbb{R}^{N_{\mathrm{pair}}^{\mathrm{AO}} \times p},
$$

and then applies the exact AO-pair kernel.

In operator form:

$$
G(C) = P(C)^{\mathsf T} \, V \, P(C),
$$

$$
\delta G(C;\Delta C)
=
\Delta P(C,\Delta C)^{\mathsf T} V P(C)
+
P(C)^{\mathsf T} V \Delta P(C,\Delta C).
$$

The current code evaluates this row by row over AO pairs.
Therefore the directional exact-2e cost is governed by

$$
T_{\delta g_{\mathrm{act}}^{(2)}}
=
\mathcal O\!\left(
N_{\mathrm{pair}}^{\mathrm{AO}} \, p^2
\right),
$$

after the AO-pair kernel applications.

Because both benchmark systems have

$$
n_a = 6,
\qquad
p = 21,
$$

this stage scales mainly with \(N_{\mathrm{pair}}^{\mathrm{AO}}\), not with a
growing active-pair dimension.

### 5.3 Projected directional structure matrices after tile contraction

For support-local coefficient blocks \(L\) and \(R\), and unique-spin kernels
\(K^\alpha\), \(K^\beta\), the same-spin contraction identity is

$$
W^{\alpha}(L,R;K^\beta) = L K^\beta R^{\mathsf T},
\qquad
W^{\beta}(L,R;K^\alpha) = L^{\mathsf T} K^\alpha R,
$$

with

$$
\langle K^\alpha, W^\alpha \rangle_F
=
\langle K^\beta, W^\beta \rangle_F.
$$

The important consequence is:

$$
\text{the contraction can be done on streamed support-local tiles,}
$$

without building global dense weight images.

So the structure stage is no longer governed by

$$
\mathcal O(N_\alpha^2 + N_\beta^2)
$$

memory objects.
It is instead governed by the actually queried support-local tiles:

$$
T_{\mathcal A}
\sim
\sum_{\text{queried tiles}}
\text{cost of local } (L K R^{\mathsf T}) \text{ contractions}.
$$

That is exactly why the structure stage stopped tracking AO basis growth.

### 5.4 Outer-response active-gradient direction

At the accepted point, the outer-response active-space gradient can be written as

$$
g_{\mathrm{outer}}
=
W_{\mathrm{loc}} \, \delta \eta
+
W_{\mathrm{dir}} \, \delta y,
$$

where:

- \(W_{\mathrm{loc}}\) is the local response with accepted selected-state
  weights fixed;
- \(W_{\mathrm{dir}}\) is the directional selected-state response pullback.

Equivalently,

$$
g_{\mathrm{outer}}
=
\left(
W_{\mathrm{loc}} + W_{\mathrm{dir}} \mathcal R_k \mathcal A_k
\right)\delta \eta.
$$

This is the correct high-level formula for the next optimization round:

$$
K_{\mathrm{outer},k}
=
W_{\mathrm{loc}} + W_{\mathrm{dir}} \mathcal R_k \mathcal A_k
$$

is an accepted-point fixed linear operator on active-space directional data.

The key point for the present note is:

$$
T_W
$$

is now mostly a selected-state / support-pattern cost.
It is no longer the dominant AO-basis-dependent term.

### 5.5 Orbital pullback

The current orbital pullback consumes the active-space gradient direction

$$
g_{\mathrm{outer}}
=
\left(
\delta S_{\mathrm{act}}^\ast,
\delta h_{\mathrm{act}}^\ast,
\delta g_{\mathrm{act}}^{(2)\ast}
\right)
$$

and maps it back to the orbital chart:

$$
B_{\mathrm{outer}} p = T_k^\ast g_{\mathrm{outer}}.
$$

In the current implementation this still includes:

1. active matrix symmetrization;
2. matrix-form active-space backprop;
3. exact active-space 2e backprop;
4. AO effective one-electron backprop;
5. final orbital chart pullback and reduced projection.

So this stage remains partly AO-size dependent and still contains repeated
format conversions:

$$
\text{matrix} \leftrightarrow \text{vector} \leftrightarrow \text{legacy row buffer}.
$$

This is why `outer_response_orbital_pullback` is still a real optimization
target even though the structure-side contraction has already been repaired.

---

## 6. What The Current Main Bottlenecks Really Are

### 6.1 First bottleneck: `h1e_fused`

This is now the clearest primary bottleneck.

Reason:

1. it dominates the total `full_cached_external` time on the larger case;
2. it still operates as a scalar AO-integral streaming stencil;
3. unlike the unique-spin contraction, it has not yet been rewritten into a
   streamed block contraction with meaningful data reuse.

In other words:

$$
\text{the next "tile-contraction-class" rewrite should start here.}
$$

### 6.2 Second bottleneck: exact active-space 2e directional build and pullback

This is the next structural bottleneck, because it appears twice:

1. in `outer_response_active_space_integrals`;
2. in `outer_response_orbital_pullback`.

The accepted-point exact pair operator is reusable, but the current path still
reconstructs directional pair data and performs row-buffer conversions around
it.

So this is not a memory emergency.
It is an operator-factorization and dataflow-efficiency issue.

### 6.3 Third bottleneck: accepted-point outer-response operatorization

The current formula already has the right structure:

$$
g_{\mathrm{outer}}
=
K_{\mathrm{outer},k} \, \delta \eta.
$$

But the implementation still applies that operator by rebuilding several
intermediate objects on every direction:

- projected directional structure columns;
- selected-state directional coefficient blocks;
- outer-response active-gradient direction assembly.

The accepted point stays fixed during one inner Krylov solve.
That makes this stage an accepted-point operatorization problem, not just a
micro-kernel problem.

### 6.4 Not the first priority anymore: structure-basis tile contraction

The current benchmark already says this stage is no longer the dominant issue.

That means:

$$
\text{do not spend the next main optimization cycle reworking the same}
\quad
\text{unique-spin contraction machinery again.}
$$

It should be preserved and reused, not reopened as the primary target.

---

## 7. Optimization Directions With Theoretical Support

### 7.1 Direction A: rewrite `h1e_fused` as an AO-pair blocked contraction

#### Core idea

Replace the scalar integral loop by a blocked apply over AO-pair graph rows or
AO-pair tiles, analogous in spirit to the support-local tile contraction used
for unique-spin structure assembly.

The current scalar stencil has the form

$$
(ij|kl)
\Rightarrow
\text{a fixed update stencil on}
\quad
\Delta F \text{ and } \bar D.
$$

This can be reorganized as a blocked bilinear apply between:

- one AO-pair block of the inactive density / unsymmetrized gradient;
- one AO-pair block of the two-electron integral graph;
- one AO-matrix tile accumulator.

#### Why this is theoretically justified

The current cost is memory-bound because each integral contributes only a small
number of flops before scattering into large matrices.
Blocking increases reuse of:

- AO-pair rows;
- density blocks;
- gradient blocks;
- AO output tiles.

So the expected gain is not from reducing asymptotic complexity, but from
moving the kernel from

$$
\text{scalar scatter}
\quad \to \quad
\text{streamed block contraction}.
$$

#### Exit criteria

1. the hot path no longer iterates one stored AO integral at a time in the
   final production kernel;
2. per-thread workspaces are AO-tile bounded rather than full repeated AO
   matrix copies where avoidable;
3. arithmetic intensity improves without reintroducing OOM risk.

### 7.2 Direction B: factor the exact active-space 2e operator around the accepted point

#### Core idea

Keep the accepted-point pair map

$$
P_k = P(C_k),
\qquad
M_k = V P_k
$$

as an accepted-point operator object.

Then one direction only needs

$$
\Delta P_k(\Delta C),
$$

and the directional exact-2e action becomes

$$
\delta G
=
\Delta P_k^{\mathsf T} M_k
+
P_k^{\mathsf T} (V \Delta P_k).
$$

#### Why this is theoretically justified

This is the exact bilinear derivative identity for

$$
G(C) = P(C)^{\mathsf T} V P(C).
$$

So the accepted-point factorization is exact and does not change the physics.
It only removes repeated reconstruction of the accepted side.

#### Immediate implementation goals

1. stop copying accepted dense active coefficients back and forth between
   `Eigen::MatrixXd` and legacy row buffers on every direction;
2. reuse accepted-point AO-pair products in both the directional build and the
   2e backprop;
3. converge the directional-build path and the orbital-pullback path onto the
   same accepted-point exact-pair operator workspace.

### 7.3 Direction C: build the accepted-point outer-response operator explicitly as an operator, not as repeated workflow

#### Core idea

Use the accepted-point factorization

$$
K_{\mathrm{outer},k}
=
W_{\mathrm{loc}} + W_{\mathrm{dir}} \mathcal R_k \mathcal A_k
$$

as the design target.

The objective is **not** to materialize a dense matrix.
The objective is to replace "re-run the entire directional workflow" by
"apply one accepted-point fixed block operator".

#### Why this is theoretically justified

During one Newton / CG inner solve, the accepted point is fixed.
Therefore:

- \(\mathcal A_k\) is fixed;
- \(\mathcal R_k\) is fixed;
- \(W_{\mathrm{loc}}\) and \(W_{\mathrm{dir}}\) are fixed as linear maps.

So every HVP apply is mathematically an operator application at fixed
coefficients, not a new symbolic problem.

#### Immediate implementation goals

1. cache accepted-point projected selected-state structure maps at the right
   abstraction level;
2. avoid rebuilding selected-state directional determinant blocks when a
   cheaper accepted-point block operator is sufficient;
3. keep all storage bounded and block-structured, not globally dense.

### 7.4 Direction D: pull back directly to the packed / reduced orbital chart

#### Core idea

Avoid the current chain

$$
\text{active-space gradient}
\to
\text{full orbital-value gradient}
\to
\text{packed gradient}
\to
\text{reduced gradient}.
$$

Instead, as much of \(T_k^\ast\) as possible should land directly on the packed
or reduced chart.

#### Why this is theoretically justified

The final reduced HVP output only lives in the nonredundant orbital chart.
So any full-chart intermediate that exists only to be gathered and projected
afterward is algorithmically suspicious.

This does not change the derivative.
It only changes where the adjoint is accumulated.

#### Immediate implementation goals

1. remove avoidable full-size `orbital_value_gradient` staging;
2. fuse symmetrization, gather, and projection when the destination chart is
   already known;
3. avoid redundant matrix-to-vector packing on the outer-response path.

---

## 8. Recommended Execution Order

### Stage 1: AO-side blocked rewrite

Goal:
replace the current scalar `h1e_fused` AO integral stencil by an AO-pair
blocked contraction kernel.

Why first:

1. it is now the clearest wall-time dominant term;
2. it is structurally analogous to the earlier tile-contraction rewrite;
3. it does not require changing the high-level accepted-point outer-response
   formulas.

### Stage 2: exact active-space 2e operator factorization

Goal:
turn the accepted-point exact pair map into a reusable operator for both
directional active-space integrals and orbital pullback.

Why second:

1. it attacks both `outer_response_active_space_integrals` and
   `outer_response_orbital_pullback`;
2. it removes known copy / row-buffer churn;
3. it is a lower-risk structural change than immediately reworking the whole
   outer-response linear-response stack.

### Stage 3: accepted-point outer-response operatorization

Goal:
replace repeated directional workflow execution by one accepted-point block
operator apply on \(\delta \eta\).

Why third:

1. it is mathematically the right long-term structure;
2. it should become much easier once Stage 2 has clarified the exact active 2e
   operator boundary;
3. it is more invasive than the first two stages and should be built on top of
   the repaired AO-side kernels.

### Stage 4: reduced-chart direct pullback

Goal:
eliminate avoidable full-chart orbital-gradient staging.

Why fourth:

1. it builds naturally on the cleaner Stage 2 and Stage 3 interfaces;
2. it is important, but it benefits from having the upstream operators already
   stabilized.

---

## 9. Design Constraints For All Later Work

The next rounds should preserve the invariants already established by the
tile-contraction rewrite.

### 9.1 No return to global dense support weights

Do **not** reintroduce:

$$
O(N_\alpha^2), \quad O(N_\beta^2)
$$

persistent weight images on the production path.

### 9.2 No return to `O(n_threads * N_{2e})` packed-gradient temporaries

The striped packed-gradient reduction should remain the memory discipline
baseline for large exact-integral runs.

### 9.3 Avoid repeated matrix copies unless they are algorithmically essential

In practice this means:

1. no repeated `Eigen` to row-buffer round-trips for accepted-point data;
2. no repeated full zero-fill plus full accumulation passes when one moved
   first-contribution path is sufficient;
3. prefer caller-owned workspaces and streamed tile/block accumulation.

---

## 10. Short Conclusion

After the unique-spin tile-contraction rewrite, the main exact-integral TNHVP
bottleneck has moved.

The current priority order is:

$$
\boxed{
\texttt{h1e\_fused}
>
\texttt{exact active 2e directional / pullback}
>
\texttt{accepted-point outer-response operatorization}
>
\texttt{direct reduced-chart pullback}
}
$$

The most important practical conclusion is:

$$
\text{the next major rewrite should be AO-side blocked contraction,}
$$

not another round of reopening the same unique-spin structure contraction.

The unique-spin contraction work was necessary.
It just moved the real bottleneck to the next layer.
