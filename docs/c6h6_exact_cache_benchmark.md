# C6H6 Exact Same-Spin Cache Benchmark and Interpretation

## 1. Purpose

This note records the exact-integral benchmark results for the current
same-spin cache / unique-spin-string implementation on the C6H6 examples, and
explains why the structure-builder speedup is substantial but still much
smaller than the nominal unique-string reuse factor.

The main question is:

$$
\text{Why is the exact structure-builder speedup about } 7.83\times
\text{ rather than } 20\times ?
$$

The short answer is:

$$
\text{because the } 20\times \text{ factor measures only same-spin string reuse,}
$$

whereas the measured structure-builder wall time still contains:

1. opposite-spin channel contractions,
2. structure-pair level local gather and contraction overhead,
3. other matrix-form assembly work that is not reduced by the same factor.

---

## 2. Benchmark Setup

All measurements below were performed with:

$$
\texttt{OMP\_NUM\_THREADS=1},
\qquad
\texttt{--standard-two-electron-mode exact}.
$$

The cache was toggled through

```bash
XMVB_SAME_SPIN_PAIR_CACHE_MB=4096
```

for cache enabled, and

```bash
XMVB_SAME_SPIN_PAIR_CACHE_MB=0
```

for cache disabled.

Two C6H6 inputs were measured:

1. `src/test_molecule/C6H6.xmi`
2. `src/test_molecule/C6H6.full.xmi`

The first input is a small selected-space case; the second is the full-space
case where unique-spin reuse is substantial.

---

## 3. Reuse Statistics

### 3.1 `C6H6.xmi`

Measured by `inspect_same_spin_reuse`:

$$
N_{\mathrm{det}} = 20,
\qquad
N_\alpha = 20,
\qquad
N_\beta = 20.
$$

Therefore

$$
\rho_\alpha = \frac{N_{\mathrm{det}}}{N_\alpha} = 1,
\qquad
\rho_\beta = \frac{N_{\mathrm{det}}}{N_\beta} = 1.
$$

This case has essentially **no same-spin reuse**.

### 3.2 `C6H6.full.xmi`

Measured by `inspect_same_spin_reuse`:

$$
N_{\mathrm{det}} = 400,
\qquad
N_\alpha = 20,
\qquad
N_\beta = 20.
$$

Hence

$$
\rho_\alpha = \frac{400}{20} = 20,
\qquad
\rho_\beta = \frac{400}{20} = 20.
$$

This is the relevant benchmark for the unique-spin-string algorithm.

---

## 4. Orbital-Evaluation Benchmark

The following data were obtained from

```bash
build/src/benchmark_cpp_orbital_eval
```

with

```bash
--repeat 5 --warmup 1
```

so that the reported numbers are mean per-evaluation times.

### 4.1 `C6H6.xmi`

| metric | cache on | cache off | on/off |
|---|---:|---:|---:|
| `mean_total_dt` | 0.6040752012 s | 0.5585691010 s | 0.924668 |
| `mean_structure_dt` | 0.0009282702 s | 0.0007729384 s | 0.832665 |
| `mean_active_adjoint_dt` | 0.0011249410 s | 0.0003067536 s | 0.272686 |
| `mean_active_2e_backprop_dt` | 0.0011853110 s | 0.0012014428 s | 1.013610 |
| `mean_ao_h1e_backprop_dt` | 0.0982733524 s | 0.0956316940 s | 0.973119 |

Define a backward subtotal

$$
t_{\mathrm{backward}}
=
t_{\mathrm{active\ adjoint}}
+
t_{\mathrm{active\ 2e\ backprop}}
+
t_{\mathrm{ao\ h1e\ backprop}}.
$$

Then

$$
t_{\mathrm{backward,on}} = 0.1005836044\ \mathrm{s},
$$

$$
t_{\mathrm{backward,off}} = 0.0971398904\ \mathrm{s},
$$

so the backward subtotal ratio is

$$
\frac{t_{\mathrm{backward,off}}}{t_{\mathrm{backward,on}}}
=
0.965763.
$$

This is consistent with the absence of same-spin reuse in this input.

### 4.2 `C6H6.full.xmi`

| metric | cache on | cache off | speedup |
|---|---:|---:|---:|
| `mean_total_dt` | 0.6535313136 s | 0.9476052154 s | 1.449977 |
| `mean_structure_dt` | 0.0255691130 s | 0.2838687570 s | 11.102018 |
| `mean_active_adjoint_dt` | 0.0010947532 s | 0.0035470354 s | 3.239957 |
| `mean_active_2e_backprop_dt` | 0.0010972104 s | 0.0010837832 s | 0.987762 |
| `mean_ao_h1e_backprop_dt` | 0.1034893864 s | 0.1288631598 s | 1.245186 |

For the same backward subtotal,

$$
t_{\mathrm{backward,on}} = 0.1056813500\ \mathrm{s},
$$

$$
t_{\mathrm{backward,off}} = 0.1334939784\ \mathrm{s},
$$

so the backward speedup is

$$
\frac{t_{\mathrm{backward,off}}}{t_{\mathrm{backward,on}}}
=
1.263174.
$$

The corresponding reductions are

$$
1 - \frac{0.6535313136}{0.9476052154} = 31.03\%,
$$

$$
1 - \frac{0.0255691130}{0.2838687570} = 90.99\%,
$$

$$
1 - \frac{0.1056813500}{0.1334939784} = 20.83\%.
$$

Thus, in the full-space benzene case:

1. the structure-builder contribution drops very strongly,
2. the total single-step time drops materially,
3. the backward path improves, but much less dramatically than the structure
   builder itself.

---

## 5. Dedicated Structure-Builder Benchmark

The exact structure-builder was also measured directly with

```bash
build/src/check_structure_builder_fast_path
```

on `src/test_molecule/C6H6.full.xmi`.

With cache enabled:

$$
t_{\mathrm{fast,exact}} = 0.023743458\ \mathrm{s}.
$$

With the direct reference builder:

$$
t_{\mathrm{reference}} = 0.185933111\ \mathrm{s}.
$$

Therefore the measured exact structure-builder speedup is

$$
\frac{t_{\mathrm{reference}}}{t_{\mathrm{fast,exact}}}
=
\frac{0.185933111}{0.023743458}
=
7.830920.
$$

The relative reduction is

$$
1 - \frac{0.023743458}{0.185933111} = 87.23\%.
$$

The numerical agreement remained at machine precision:

$$
\max |\Delta S| \sim 10^{-15},
\qquad
\max |\Delta H| \sim 10^{-14}.
$$

---

## 6. Why the Speedup Is Not \( 20\times \)

### 6.1 What the factor \( 20\times \) actually means

For `C6H6.full.xmi`,

$$
N_{\mathrm{det}} = 400,
\qquad
N_\alpha = N_\beta = 20.
$$

Hence the same-spin reuse ratios are

$$
\rho_\alpha = \rho_\beta = 20.
$$

This means:

$$
\text{each unique alpha string is reused on average } 20 \text{ times,}
$$

and similarly for beta.

Equivalently, the same-spin kernel construction cost is reduced from a
determinant-space viewpoint

$$
O(N_{\mathrm{det}}^2)
$$

to a unique-spin viewpoint

$$
O(N_\alpha^2 + N_\beta^2)
$$

for the reusable same-spin ingredients.

However, this factor is **not** the speedup of the entire structure-builder
wall time. It only quantifies the compression of the same-spin determinant
subproblems.

### 6.2 The timed structure builder still does more than same-spin caching

The benchmarked exact structure-builder performs more work than merely looking
up cached same-spin pairs.

For each structure pair \( (I,J) \), the fast exact path still performs:

1. support gathering of local alpha and beta subblocks,
2. overlap contraction,
3. one-electron contraction,
4. same-spin total contraction,
5. opposite-spin channel accumulation.

In formula form, the exact structure-basis Hamiltonian remains

$$
H_{IJ}
=
\left\langle
\mathbf{C}_I^{\mathrm{loc}},\;
\mathbf{T}^\alpha[I,J]\,
\mathbf{C}_J^{\mathrm{loc}}\,
\left( \mathbf{S}^\beta[I,J] \right)^{\mathrm{T}}
+
\mathbf{S}^\alpha[I,J]\,
\mathbf{C}_J^{\mathrm{loc}}\,
\left( \mathbf{T}^\beta[I,J] \right)^{\mathrm{T}}
\right\rangle_{\mathrm{F}}
+
\sum_{P \in \mathcal{P}_{IJ}}
\left\langle
\mathbf{C}_I^{\mathrm{loc}},\;
\mathbf{U}_P^\alpha[I,J]\,
\mathbf{C}_J^{\mathrm{loc}}\,
\left( \widetilde{\mathbf{U}}_P^\beta[I,J] \right)^{\mathrm{T}}
\right\rangle_{\mathrm{F}}.
$$

The first large term is where the same-spin cache helps most strongly. The
second term is the exact opposite-spin channel sum and is **not** eliminated by
the same-spin reuse factor.

### 6.3 Opposite-spin is a major residual cost

In the current exact fast path, the opposite-spin contribution is accumulated as

$$
H_{IJ}^{\alpha\beta}
=
\sum_{P \in \mathcal{P}_{IJ}}
\left\langle
\mathbf{C}_I^{\mathrm{loc}},\;
\mathbf{U}_P^\alpha[I,J]\,
\mathbf{C}_J^{\mathrm{loc}}\,
\left( \widetilde{\mathbf{U}}_P^\beta[I,J] \right)^{\mathrm{T}}
\right\rangle_{\mathrm{F}}.
$$

This means that for each structure pair \( (I,J) \), the code still has to:

1. test channel support,
2. gather local sparse/dense subblocks,
3. perform local matrix contractions.

Therefore the opposite-spin part remains a significant residual cost even after
same-spin determinant reuse has been fully exploited.

### 6.4 Local gather and contraction overhead also remains

Even inside the same-spin-like terms, the fast path is not a free lookup. It
still requires local support handling of the form

$$
\mathbf{M}^\alpha[I,J] = \mathbf{M}^\alpha[A_I, A_J],
\qquad
\mathbf{M}^\beta[I,J] = \mathbf{M}^\beta[B_I, B_J],
$$

followed by local dense contractions with

$$
\mathbf{C}_I^{\mathrm{loc}},
\qquad
\mathbf{C}_J^{\mathrm{loc}}.
$$

Thus the fast path removes the repeated determinant-pair same-spin algebra, but
it does **not** remove:

$$
\text{support gathering}
+
\text{local block assembly}
+
\text{Frobenius contractions}.
$$

These costs are much smaller than the original full determinant-pair route, but
they do not scale away by the raw reuse factor \( 20 \).

### 6.5 Cache build time is not the reason

For `C6H6.full.xmi`, the same-spin cache build itself was only about

$$
0.002\ \mathrm{s},
$$

and in `check_structure_builder_fast_path` this cache construction is performed
before the timed `fast_exact_result_seconds` region.

So the measured

$$
7.83\times
$$

is not being diluted by cache-setup overhead. It reflects the true residual
cost of the structure-builder hot path after same-spin reuse has already been
factored out.

---

## 7. Practical Interpretation

The benchmark implies the following.

First,

$$
7.83\times
$$

for the dedicated exact structure builder is already a strong result. It means
the same-spin determinant-pair bottleneck has been removed to a large extent.

Second, the reason the speedup is not

$$
20\times
$$

is not a contradiction. The quantity

$$
20\times
$$

measures only unique-string reuse, whereas the wall-clock timing includes
remaining opposite-spin work and local contraction overhead.

Third, the benchmark strongly suggests that the next major exact-structure
optimization target is:

$$
\text{opposite-spin channel contraction}.
$$

In other words, for the exact structure builder, same-spin is no longer the
dominant bottleneck on this benzene full-space case; the residual bottleneck is
primarily the opposite-spin part and the associated pair/channel accumulation.

---

## 8. Final Summary

For `C6H6.full.xmi`, the unique-spin same-spin reuse factor is

$$
20\times,
$$

but the measured exact structure-builder speedup is only

$$
7.83\times
$$

because the timed builder still contains non-negligible work that is not
reduced by the same-spin reuse ratio:

$$
\text{fast structure build}
=
\text{same-spin local contractions}
+
\text{opposite-spin channel contractions}
+
\text{support gather / local assembly overhead}.
$$

Therefore the correct interpretation is not

$$
\text{``the implementation failed to realize the } 20\times \text{ reuse''},
$$

but rather

$$
\text{``the same-spin part has been compressed strongly, and the residual
bottleneck is now mainly opposite-spin and local accumulation.''}
$$
