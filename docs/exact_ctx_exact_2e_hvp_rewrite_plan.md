# Exact-Integral Exact-2e HVP Bottleneck Analysis And Rewrite Plan

> Status note (2026-04-21):
> this note is a focused addendum to
> [docs/exact_ctx_post_tile_contraction_optimization_plan.md](./exact_ctx_post_tile_contraction_optimization_plan.md).
> The scope here is narrower:
> the accepted-point exact active-space two-electron Hessian-vector product
> inside the current `exact_ctx` / TNHVP path.
>
> RI / density fitting / Cholesky are intentionally out of scope.

## 1. Why This Note Exists

The earlier post-tile-contraction note was written when the broad picture was:

$$
\texttt{h1e\_fused} + \texttt{outer\_response}
$$

still dominated the wall time.

After restoring the accepted-point TNHVP trajectory to the old fast branch,
the diagnosis changed:

1. the convergence path is now back to the old baseline;
2. the wall time is still much worse than the old baseline;
3. the dominant per-apply cost has moved into the exact active-space
   two-electron HVP itself.

So the current issue is not

$$
\text{``wrong optimization path''}
$$

but rather

$$
\text{``the exact-2e HVP kernel is now the hot path.''}
$$

This note writes down the code-level decomposition, the underlying linear
algebra, why the current kernel is bandwidth-limited, and which rewrite should
come next.

---

## 2. Empirical Facts From The Restored `241_VBSCF` Path

The restored TNHVP run on `test/241_VBSCF.xmi` reproduced the old accepted
energies for the first two iterations:

$$
E_1 = -230.576971441037,
\qquad
E_2 = -230.713194999446.
$$

So the accepted-point model and optimizer trajectory are no longer the main
problem.

However, the first exact_ctx apply-stage RSS / timing log showed:

| stage | elapsed |
| --- | ---: |
| `after_ao_h1e_fused` | `0.197 s` |
| `after_active_two_electron` | `3.789 s` |
| `after_outer_active_space_integrals` | `4.121 s` |
| `before_return` | `4.459 s` |

Therefore the first accepted-point apply is dominated by

$$
T_{\mathrm{act2e}} \gg T_{\mathrm{h1e\_fused}},
$$

at least on the restored `241` path.

The corresponding process RSS was already about

$$
5.9 \text{ GiB}
$$

before the first exact-2e apply returned, while the explicit accepted-point
exact_ctx caches were only a few MiB. So:

1. the current exact-2e cache objects are not themselves the RSS explosion;
2. the current exact-2e HVP is primarily a speed bottleneck;
3. the rewrite should target memory traffic and contraction order first.

An internal exact-2e stage log on the same restored `241` path then showed:

| exact-2e internal stage | elapsed |
| --- | ---: |
| `after_fixed_backprop` | `0.00035 s` |
| `after_mixed_pair_build` | `0.00080 s` |
| `after_cached_row_direct` | `3.66-3.70 s` |

with

$$
n_b = 120,
\qquad
N_{\mathrm{ao}} = 7260,
\qquad
p = 21,
\qquad
N_{2e}^{\mathrm{AO}} = 21{,}156{,}059.
$$

So the current regression is **not** that the code spends most of its time
building the full `mixed_pair` buffer.
The regression is that the current `cached_row_direct` AO-pair graph kernel is
itself too slow.

---

## 3. Current Exact-2e HVP Decomposition In Code

The hot path is
[src/vb/orbital/active_space_two_electron_utils.cpp](../src/vb/orbital/active_space_two_electron_utils.cpp)
inside
`apply_exact_packed_active_two_electron_adjoint_hessian_vector(...)`.

At one accepted point, let

$$
C \in \mathbb{R}^{n_b \times n_a},
\qquad
\Delta C \in \mathbb{R}^{n_b \times n_a},
$$

where:

- \(n_b\) is the AO basis size;
- \(n_a\) is the active-orbital count;
- \(p = n_a(n_a+1)/2\) is the packed active-pair count;
- \(N_{\mathrm{ao}} = n_b(n_b+1)/2\) is the packed AO-pair count.

The current implementation can be written schematically as

$$
\Delta g_C^{(2e)}
=
H_{\mathrm{fixed}} \, \Delta C
+
\mathcal B^\ast_{\mathrm{acc}}
\Bigl(
V_{\mathrm{AO}}
\,
\Delta P(C; \Delta C)
\Bigr),
$$

where:

- \(H_{\mathrm{fixed}} \, \Delta C\) is the accepted-point fixed term
  backpropagated from cached pair-gradient matrices or cached pair-gradients;
- \(\Delta P(C; \Delta C)\) is the mixed AO-pair to active-pair coefficient
  matrix built from the accepted coefficients \(C\) and the direction
  \(\Delta C\);
- \(V_{\mathrm{AO}}\) is the exact AO-pair integral operator;
- \(\mathcal B^\ast_{\mathrm{acc}}\) is the accepted-point pullback from
  active-pair rows to dense active AO rows.

The important code-level point is that the current direct path is not actually
an algebraically tiny operator. It is implemented as:

1. build the full dense buffer
   \(\Delta P \in \mathbb{R}^{N_{\mathrm{ao}} \times p}\);
2. stream the AO-pair graph or sparse integral list;
3. for each graph edge, fetch one source row of \(\Delta P\);
4. combine it with one accepted backprop row block from
   \(\mathcal B^\ast_{\mathrm{acc}}\);
5. accumulate into one dense AO-by-active output row.

So the direct path avoids the old

$$
\Delta P \, G_{\mathrm{pair}}
$$

matrix multiply, but it still relies on an AO-pair graph streaming kernel whose
cost now dominates the entire exact-2e apply.

---

## 4. Why The Current Direct Path Is Slow

### 4.1 The current kernel is bandwidth-dominated

For small and moderate active spaces, \(p\) is not large.
For example, in `CAS(6,6)`:

$$
p = 21.
$$

So each graph edge performs only a short contraction:

$$
\text{one AO-pair edge}
\Rightarrow
\text{read one length-}p\text{ source row}
+\,
\text{read one } p \times n_a \text{ accepted backprop slab}
+\,
\text{update one length-}n_a\text{ target row.}
$$

The arithmetic work per edge is therefore only

$$
\mathcal O(p \, n_a),
$$

but the kernel performs several memory reads with poor temporal locality.

This is the signature of a bandwidth-bound streaming contraction, not a
compute-bound dense kernel.

### 4.2 The current direct kernel, not the `mixed_pair` build, is the measured hot stage

The current code first materializes

$$
\Delta P \in \mathbb{R}^{N_{\mathrm{ao}} \times p}
$$

into `mixed_pair_coefficients_buffer`, and then the graph traversal rereads it
through `column_pair_index` lookups.

On `241`, however, the measured timings show:

$$
T_{\mathrm{mixed\_build}} \approx 8\times 10^{-4}\,\mathrm{s},
\qquad
T_{\mathrm{cached\_row\_direct}} \approx 3.7\,\mathrm{s}.
$$

So the immediate speed problem is not the one-time full-buffer write itself.
The problem is the graph-driven direct contraction that follows.

The full-buffer write still matters algorithmically because it represents extra
traffic:

$$
8 N_{\mathrm{ao}} p \text{ bytes},
$$

followed by a graph-driven reread whose effective traffic is closer to

$$
\mathcal O(E_{\mathrm{graph}} \, p),
$$

where \(E_{\mathrm{graph}}\) is the AO-pair graph edge count.

Even when \(\Delta P\) itself fits in memory, this organization is still bad
for speed because:

1. the source rows are revisited under graph order, not build order;
2. the contraction uses short row fragments, so the kernel cannot hide memory
   latency with high arithmetic intensity;
3. the accepted backprop rows are also streamed repeatedly.

### 4.3 The older materialized path is currently much faster

A direct compute-node comparison with

$$
\texttt{XMVB\_CPP\_ENABLE\_EXACT\_2E\_FUSED\_HVP=0}
$$

showed:

| exact-2e internal stage | elapsed |
| --- | ---: |
| `after_pair_gradient_transform` | `0.0013-0.0016 s` |
| `after_pair_kernel` | `0.320-0.336 s` |
| `after_final_backprop` | `0.321-0.337 s` |

So on the restored `241` path the older materialized exact-2e route is about

$$
\frac{3.66\text{--}3.70}{0.320\text{--}0.336}
\approx 11\times
$$

faster than the current direct cached-row kernel.

This changes the practical priority:

1. the current fused/direct exact-2e path should not remain the default hot
   path;
2. the next structural rewrite should target the direct AO-pair graph kernel
   itself, not just the mixed-row build.

### 4.4 This is exactly the kind of regression that matrix-form code can hide

The original matrix-form implementation was fast because dense BLAS-style
contractions can saturate cache reuse and vector units:

$$
\text{few large dense contractions}
\gg
\text{many short irregular streaming contractions}.
$$

But the old fully materialized path risks large intermediates and can become
unsafe on larger workloads.

So the replacement target is not

$$
\text{``go back to global matrix form everywhere''},
$$

but

$$
\text{``recover dense-kernel locality without restoring the OOM-prone global intermediates.''}
$$

---

## 5. The Right Rewrite Target

### 5.1 Immediate action: keep the current fused/direct path opt-in

Because the existing direct kernel is already measured to be much slower than
the materialized route, the immediate code-level action should be:

$$
\text{materialized exact-2e path by default,}
\qquad
\text{current fused/direct path only by explicit opt-in.}
$$

This does not solve the long-term design problem, but it prevents the known
regression from remaining on the default hot path.

### 5.2 Long-term rewrite: source-tiled / blocked direct contraction

The next structural rewrite should be:

$$
\text{source-tiled / block-fused contraction}
$$

for the exact-2e HVP direct path.

The core idea is:

1. partition AO-pair source rows into tiles
   \(\mathcal J_t \subset \{1, \dots, N_{\mathrm{ao}}\}\);
2. build only the mixed rows
   \(\Delta P_{\mathcal J_t}\) for the current tile;
3. immediately contract that tile against the AO-pair graph contributions;
4. accumulate directly into dense active AO gradients;
5. discard the tile and move on.

In operator form, instead of

$$
\Delta P
\xrightarrow{\text{materialize all}}
V_{\mathrm{AO}}
\xrightarrow{\text{stream}}
\mathcal B^\ast_{\mathrm{acc}},
$$

we want

$$
\Delta P_{\mathcal J_t}
\xrightarrow{\text{build tile}}
V_{\mathrm{AO}, \mathcal J_t}
\xrightarrow{\text{immediate contraction}}
\mathcal B^\ast_{\mathrm{acc}},
$$

with the tile index \(t\) streamed.

### 5.3 Why this avoids OOM

If the tile contains \(T\) AO-pair source rows, the extra transient memory is
only

$$
\mathcal O(T p) + \mathcal O(n_b n_a),
$$

instead of a global source matrix plus larger downstream temporaries.

This is the exact analogue of the earlier unique-spin tile-contraction idea:

$$
\text{keep the accepted-point operator fixed,}
\qquad
\text{stream only the queried directional slab.}
$$

So the source-tiled rewrite keeps the exact integral path stable for large
systems while still recovering locality.

### 5.4 Why this should be faster

The speed argument is not just ``fewer allocations''.
The real gain is that the contraction order changes from

$$
\text{build all rows}
\to
\text{reread them under irregular graph order},
$$

to

$$
\text{build a short row block}
\to
\text{consume it while it is still cache-resident}.
$$

That improves:

1. source-row reuse;
2. prefetch behavior;
3. cache residency of the accepted backprop slabs touched by the same tile;
4. total write-read traffic on `mixed_pair_coefficients_buffer`.

For speed, this is the closest exact-2e analogue to the previous
structure-space tile contraction.

---

## 6. Data-Structure Requirement For The Rewrite

The existing AO-pair graph traversal is target-row oriented.
For source-tiled fusion, we will need an efficient way to enumerate all graph
contributions that consume a tile of source rows.

That can be done in either of two ways:

### Option A: build a source-oriented adjacency once at the accepted point

Construct a source-bucketed view of the AO-pair graph:

$$
\text{source row } j
\mapsto
\{ (i, K_{ij}) \},
$$

so one source tile can be consumed without random lookups into a globally
materialized `mixed_pair` matrix.

This costs only graph-sized integer / index storage and preserves the exact
integral algebra.

### Option B: use symmetric edge tiles from the sparse pair-index list

If the sparse integral list is already the cheaper representation on some
inputs, we can stream the integral list in chunks and compute just the source
rows touched by the current chunk.

This may be easier to prototype, but it generally gives less predictable
source-row reuse than an explicit source-bucketed graph.

For the current exact_ctx hot path, Option A is the cleaner long-term design.

---

## 7. Immediate Implementation Plan

### Stage 1: fine-grained timing inside exact-2e HVP

Completed.
The internal timing already showed:

1. fixed backprop is negligible;
2. mixed-row build is negligible;
3. current `cached_row_direct` dominates the exact-2e apply;
4. the materialized path is already much faster on `241`.

So the next implementation step is no longer just diagnostic.
It is to replace the current direct kernel organization.

This measurement fixes exactly how much of

$$
T_{\mathrm{act2e}}
$$

comes from the graph-stream direct kernel.

### Stage 2: accepted-point source-bucket cache

At accepted-point cache build time, add a source-oriented AO-pair adjacency
view that is compatible with the current exact integral graph.

The cache should still scale as

$$
\mathcal O(E_{\mathrm{graph}})
$$

and must not reintroduce dense AO-pair-by-AO-pair storage.

### Stage 3: tile-local mixed-row workspace

Replace the current global
`mixed_pair_coefficients_buffer` with a tile-local workspace:

$$
\Delta P_{\mathcal J_t}
\in
\mathbb{R}^{T \times p}.
$$

This tile should be built directly from:

- `accepted_dense_active_coefficients_buffer`;
- `dense_active_direction_buffer`.

No extra full-size copy of `\Delta P` should survive outside the tile loop.

### Stage 4: fused tile contraction into dense active gradients

Consume one source tile immediately against the source-bucketed graph and
accumulate directly into
`dense_active_gradient_direction_buffer`.

This must preserve the current exact algebra and should not reintroduce the
old

$$
O(n_{\mathrm{threads}} \, N_{\mathrm{ao}} \, p)
$$

workspace explosion.

---

## 8. Non-Negotiable Constraints

### 8.1 Exact integrals only

This rewrite is for the current exact integral path.
No RI-specific factorization should be mixed into the design.

### 8.2 Speed first, but not by regressing large-case stability

The goal is:

$$
\text{recover old matrix-form speed characteristics}
$$

without returning to globally OOM-prone intermediates.

### 8.3 Avoid frequent matrix copies

The current code already copies the incoming `Eigen::MatrixXd` direction into a
legacy row buffer once per apply.
That copy is small compared with the AO-pair work.

The real copy / traffic problem is the global `mixed_pair` materialization.
That is the buffer to eliminate or localize.

---

## 9. Short Conclusion

The restored `241` path shows that the accepted-point exact_ctx math is back on
the right optimizer trajectory.

The main remaining runtime regression is now:

$$
\texttt{active\_two\_electron}
$$

inside the exact-2e HVP apply.

The current direct path is slow because it still behaves like a bandwidth-bound
streaming contraction:

$$
\text{build full } \Delta P
\to
\text{reread it under irregular AO-pair graph order}.
$$

The next rewrite should therefore be a source-tiled, on-the-fly mixed-row
contraction that:

1. keeps only a small directional tile resident;
2. contracts it immediately against a source-oriented AO-pair graph view;
3. accumulates directly to dense active AO gradients;
4. preserves exact integrals and large-case stability without OOM.
