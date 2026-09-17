# Bounded-Memory Unique-Spin to Structure Contraction Design

> Status note (2026-04-21):
> this note is the current design target for replacing the remaining
> query-driven scalar contractions by bounded-memory matrix-form contractions.
> The scope is broader than outer-response alone:
> it covers the general algorithmic pattern for contracting objects defined on
> unique alpha/beta spin strings into structure-space, selected-state, local
> response, and directional response quantities.

## 1. Purpose

The current exact-integral VBSCF implementation already has the right
high-level ingredients:

- unique-spin reuse tables in
  [same_spin_pair_cache.hpp](./matrices/same_spin_pair_cache.hpp);
- structure-local coefficient blocks in
  [structure_coefficient_blocks.hpp](./matrices/structure_coefficient_blocks.hpp);
- trimmed selected-state coefficient blocks in
  [selected_state_determinant_matrices.hpp](./scf/selected_state_determinant_matrices.hpp);
- a streamed matrix-form accepted same-spin backward path in
  [same_spin_matrix_backward.cpp](./scf/same_spin_matrix_backward.cpp).

What is still inconsistent is the contraction algorithm:

1. some paths already use support-local matrix multiplication;
2. some paths still use postings plus scalar accessors;
3. the original fully materialized matrix-form idea was fast but could exceed
   memory limits on large unique-spin spaces;
4. the current scalar fallback avoids the worst memory spikes, but sacrifices
   most of the arithmetic intensity that matrix-form contraction should have.

The goal of this note is to write down one common algorithmic framework that:

1. keeps the matrix-form contraction order;
2. never requires global dense `N_alpha x N_alpha` or `N_beta x N_beta` weight
   matrices as permanent objects;
3. extends to forward structure assembly, selected-state builds, accepted
   active-space gradients, local response, and directional response;
4. is compatible with the current exact integral representation.

This note intentionally assumes the present exact-integral path.
It does **not** assume RI, Cholesky, or THC.

---

## 2. Core Problem Statement

After same-spin reuse compression, every full determinant index `d` maps to one
pair of unique spin-string ids

$$
d \leftrightarrow (a, b),
$$

where

- $a \in \{1, \dots, N_{\alpha}\}$ is a unique alpha string id;
- $b \in \{1, \dots, N_{\beta}\}$ is a unique beta string id.

Therefore the determinant coefficient of any structure or selected state can be
reshaped into a matrix on unique-spin rows and columns.

The general contraction problem is:

$$
\text{Given coefficient blocks on } (a,b),
\text{ contract them with kernels on } (a,a') \text{ and } (b,b').
$$

If done naively, this suggests global dense objects such as

$$
W^{\alpha} \in \mathbb{R}^{N_{\alpha} \times N_{\alpha}},
\qquad
W^{\beta} \in \mathbb{R}^{N_{\beta} \times N_{\beta}},
$$

or even families indexed by packed active-pair labels.

That is exactly what leads to the old memory failure mode:

1. the algorithm becomes fast because it reduces to GEMM-like kernels;
2. but it requires globally materialized pair-weight images or channel
   families;
3. those objects scale like $O(N_{\alpha}^2 + N_{\beta}^2)$ or worse, and can
   dominate memory before the actual active-space orbital gradient is formed.

The correct target is therefore:

$$
\text{matrix-form arithmetic} + \text{streamed memory footprint}.
$$

---

## 3. Data Model Already Present In The Code

### 3.1 Structure-local coefficient blocks

For each structure index $I$, the code already builds one trimmed coefficient
block

$$
T^{(I)} \in \mathbb{R}^{|\mathcal{A}_I| \times |\mathcal{B}_I|},
$$

where:

- $\mathcal{A}_I \subseteq \{1, \dots, N_{\alpha}\}$ is the exact alpha support
  of structure $I$;
- $\mathcal{B}_I \subseteq \{1, \dots, N_{\beta}\}$ is the exact beta support
  of structure $I$.

This is the object stored by `StructureCoefficientBlock` in
[structure_coefficient_blocks.hpp](./matrices/structure_coefficient_blocks.hpp).

The important fact is that the code already trims exact zero rows and columns
after determinant accumulation. So the natural coefficient carrier is not the
global dense `N_alpha x N_beta` matrix, but one support-local block.

### 3.2 Selected-state coefficient blocks

For selected state $n$, the determinant-space coefficients are

$$
c_d^{(n)} = \sum_{I=1}^{N_{\mathrm{str}}} T_{dI} \, u_I^{(n)}.
$$

After unique-spin regrouping, this becomes another trimmed local block

$$
C^{(n)} \in \mathbb{R}^{|\mathcal{A}_n| \times |\mathcal{B}_n|}.
$$

This is stored by `SelectedStateDeterminantCoefficients` in
[selected_state_determinant_matrices.hpp](./scf/selected_state_determinant_matrices.hpp).

### 3.3 Directional selected-state coefficient blocks

For first-order outer-response or any other directional selected-state update,
the directional coefficients can also be represented as a trimmed block

$$
\delta C^{(n)} \in
\mathbb{R}^{|\delta \mathcal{A}_n| \times |\delta \mathcal{B}_n|}.
$$

This is stored by `DirectionalSelectedStateCoefficients`.

### 3.4 Unique-spin kernels

All current exact structure and gradient contractions can be expressed in terms
of kernels on unique-spin pair space:

$$
K^{\alpha}_{a a'},
\qquad
K^{\beta}_{b b'}.
$$

Examples include:

- overlap kernels;
- one-electron plus same-spin Hamiltonian kernels;
- directional overlap and directional Hamiltonian kernels;
- opposite-spin packed-pair projection channel matrices.

The exact data source may differ by path, but the contraction pattern is the
same.

---

## 4. Master Identity

Let

$$
L \in \mathbb{R}^{m_{\alpha} \times m_{\beta}},
\qquad
R \in \mathbb{R}^{n_{\alpha} \times n_{\beta}}
$$

be any two support-local coefficient blocks, aligned with support windows

$$
\mathcal{A}_L, \mathcal{B}_L, \mathcal{A}_R, \mathcal{B}_R.
$$

Let

$$
K^{\alpha} \in \mathbb{R}^{m_{\alpha} \times n_{\alpha}},
\qquad
K^{\beta} \in \mathbb{R}^{m_{\beta} \times n_{\beta}}
$$

be the corresponding unique-spin kernel blocks on those support windows.

Define the alpha-side weight image

$$
W^{\alpha}(L, R; K^{\beta})
=
L \, K^{\beta} \, R^{\top},
$$

and the beta-side weight image

$$
W^{\beta}(L, R; K^{\alpha})
=
L^{\top} \, K^{\alpha} \, R.
$$

Then the same scalar contraction can be written in either spin orientation:

$$
\langle K^{\alpha}, W^{\alpha}(L, R; K^{\beta}) \rangle_F
=
\langle K^{\beta}, W^{\beta}(L, R; K^{\alpha}) \rangle_F,
$$

where

$$
\langle A, B \rangle_F = \operatorname{Tr}(A^{\top} B)
$$

is the Frobenius inner product.

This is the central identity.

It says that for every structure/selected-state contraction:

1. one spin channel can be treated as the "operator" channel;
2. the opposite spin channel can be absorbed into a coefficient-side weight
   image by matrix multiplication;
3. there is no need to ever build that weight image globally;
4. it is enough to build it on the current support-local tile.

---

## 5. General Instances Of The Same Algorithm

## 5.1 Forward structure-matrix assembly

For structures $I$ and $J$, let the support-local coefficient blocks be

$$
L = T^{(I)},
\qquad
R = T^{(J)}.
$$

Then the overlap contribution is

$$
S_{IJ}
=
\langle S^{\alpha}_{IJ}, \,
T^{(I)} S^{\beta}_{IJ} [T^{(J)}]^{\top} \rangle_F.
$$

Likewise, the one-electron plus same-spin Hamiltonian contribution can be
written as

$$
H_{IJ}^{\mathrm{ss+1e}}
=
\langle H^{\alpha}_{IJ}, \,
T^{(I)} S^{\beta}_{IJ} [T^{(J)}]^{\top} \rangle_F
+
\langle S^{\alpha}_{IJ}, \,
T^{(I)} H^{\beta}_{IJ} [T^{(J)}]^{\top} \rangle_F.
$$

This is already the mathematical content of the tiled structure build in
[full_structure_builder.cpp](./matrices/full_structure_builder.cpp).

The bounded-memory rule here is simple:

1. gather only the alpha and beta pair tiles touched by the current structure
   supports;
2. build only the local weight tile
   $T^{(I)} K^{\beta}_{\mathrm{tile}} [T^{(J)}]^{\top}$;
3. contract it immediately with the alpha tile;
4. discard the local tile.

## 5.2 Selected-state build from structure coefficients

The selected-state block itself is another contraction of the same kind.

Because

$$
C^{(n)} = \sum_{I=1}^{N_{\mathrm{str}}} u_I^{(n)} T^{(I)},
$$

the selected-state build is just a linear combination of structure-local
coefficient blocks, followed by exact support trimming.

The key memory rule is again:

1. accumulate only touched unique-spin pairs;
2. trim exact zero rows and columns immediately;
3. do not retain a global dense
   $N_{\alpha} \times N_{\beta}$ matrix per selected state.

## 5.3 Accepted same-spin active-space gradient

For state-averaged accepted-point weights $\omega_n$ and selected-state
energies $E_n$, the accepted alpha-side overlap and Hamiltonian weights are

$$
W_{H,\alpha}
=
\sum_n \omega_n \,
C^{(n)} K_{H,\beta} [C^{(n)}]^{\top},
$$

$$
W_{S,\alpha}
=
-\sum_n \omega_n E_n \,
C^{(n)} K_{S,\beta} [C^{(n)}]^{\top}.
$$

The beta-side expressions are analogous.

The accepted path in
[same_spin_matrix_backward.cpp](./scf/same_spin_matrix_backward.cpp)
already implements the correct bounded-memory strategy:

1. restrict work to active unique-pair tiles;
2. build only support-local weight tiles from the selected-state blocks and the
   partner same-spin cache;
3. contract those tiles directly into active-space gradient buffers.

That is the algorithmic baseline that the other paths should follow.

## 5.4 Local response from directional spin kernels

The local-response case does **not** change the coefficient blocks.
Instead it changes the partner kernels:

$$
\delta K^{\beta}_S,
\qquad
\delta K^{\beta}_H.
$$

Therefore the local alpha-side directional weights are

$$
\delta W_{H,\alpha}^{\mathrm{local}}
=
\sum_n \omega_n \,
C^{(n)} \delta K_{H,\beta} [C^{(n)}]^{\top},
$$

$$
\delta W_{S,\alpha}^{\mathrm{local}}
=
-\sum_n \omega_n E_n \,
C^{(n)} \delta K_{S,\beta} [C^{(n)}]^{\top}.
$$

The beta-side expressions are analogous.

This means the local-response path should also be implemented as a streamed
tile contraction:

1. gather one partner directional tile
   $\delta K_{\beta,\mathrm{tile}}$ at a time;
2. build the small coefficient-side image
   $C^{(n)} \delta K_{\beta,\mathrm{tile}} [C^{(n)}]^{\top}$;
3. accumulate immediately into the same-spin local directional pullback.

The important point is that the correct bounded-memory local-response algorithm
is still matrix-form.
It does **not** need a fallback to scalar pairwise accessors.

## 5.5 Directional selected-state response

Here the partner kernels stay at the accepted point, but the coefficient block
changes from $C^{(n)}$ to $C^{(n)} + \delta C^{(n)}$.

Therefore the first-order alpha-side weight update is

$$
\delta W_{H,\alpha}^{\mathrm{dir}}
=
\sum_n \omega_n
\left(
\delta C^{(n)} K_{H,\beta} [C^{(n)}]^{\top}
+
C^{(n)} K_{H,\beta} [\delta C^{(n)}]^{\top}
\right),
$$

and for the overlap-weight channel,

$$
\delta W_{S,\alpha}^{\mathrm{dir}}
=
-\sum_n \omega_n
\left[
\delta E_n \,
C^{(n)} K_{S,\beta} [C^{(n)}]^{\top}
+
E_n
\left(
\delta C^{(n)} K_{S,\beta} [C^{(n)}]^{\top}
+
C^{(n)} K_{S,\beta} [\delta C^{(n)}]^{\top}
\right)
\right].
$$

Again, beta-side formulas are analogous.

This is the exact directional matrix-form identity behind the current scalar
postings formulas.
The correct implementation is:

1. for one support-local tile, build the left-directional term
   $\delta C \, K_{\beta,\mathrm{tile}} \, C^{\top}$;
2. build the right-directional term
   $C \, K_{\beta,\mathrm{tile}} \, \delta C^{\top}$;
3. add the accepted-energy and directional-energy prefactors;
4. contract the final local tile immediately against the alpha-side operator
   tile.

This yields the same exact result as the postings-driven formulas, but with
matrix multiplication instead of repeated scalar support lookups.

## 5.6 Opposite-spin channel family

The opposite-spin channel is not a single matrix
$K^{\sigma}$ but a family indexed by packed active-pair labels.

For exact integrals, let

$$
J_p^{\alpha},
\qquad
J_q^{\beta}
$$

denote the unique-spin projection matrices associated with packed pair labels
$p$ and $q$.

Then the exact opposite-spin contraction can be written as

$$
H^{\mathrm{os}}(L, R)
=
\sum_{p,q}
g_{pq}
\,
\langle J_p^{\alpha}, \,
L J_q^{\beta} R^{\top} \rangle_F,
$$

where $g_{pq}$ is the packed active-space two-electron kernel in pair-of-pairs
storage.

The directional analogue is

$$
\delta M_q^{\alpha}(L, R)
=
\delta L \, J_q^{\beta} R^{\top}
+
L \, J_q^{\beta} (\delta R)^{\top}
+
L \, \delta J_q^{\beta} R^{\top},
$$

and then

$$
\delta H^{\mathrm{os}}(L, R)
=
\sum_{p,q}
g_{pq}
\,
\langle J_p^{\alpha}, \delta M_q^{\alpha}(L, R) \rangle_F
+
\sum_{p,q}
g_{pq}
\,
\langle \delta J_p^{\alpha}, M_q^{\alpha}(L, R) \rangle_F.
$$

The bounded-memory rule is the same:

1. do not materialize the full family
   $\{ M_q^{\alpha} \}_q$ or $\{ M_p^{\beta} \}_p$;
2. work on one sparse packed-pair block at a time;
3. build only the current support-local mixed pair tile;
4. contract and discard.

---

## 6. Why The Old Fully Materialized Matrix-Form OOMs

The old fast matrix-form idea fails because it materializes the wrong
intermediate objects.

## 6.1 Global same-spin weight images

If one stores

$$
W_{H,\alpha},
W_{S,\alpha}
\in \mathbb{R}^{N_{\alpha} \times N_{\alpha}},
\qquad
W_{H,\beta},
W_{S,\beta}
\in \mathbb{R}^{N_{\beta} \times N_{\beta}},
$$

then the memory cost is already

$$
O(N_{\alpha}^2 + N_{\beta}^2)
$$

per channel family.

In local or directional response, several such families coexist at once.

## 6.2 Directional matrix families

If directional response keeps both accepted and directional weight matrices,
the same quadratic storage is paid again for:

- accepted Hamiltonian weights;
- accepted overlap weights;
- local directional Hamiltonian weights;
- local directional overlap weights;
- directional selected-state Hamiltonian weights;
- directional selected-state overlap weights.

That is exactly the wrong side of the speed-memory tradeoff.

## 6.3 Opposite-spin channel families

For opposite-spin contractions, the dangerous object is not a single
`N_alpha x N_alpha` weight image but a whole family indexed by packed
active-pair labels.

If one materializes

$$
M_q^{\alpha} \in \mathbb{R}^{N_{\alpha} \times N_{\alpha}}
\quad \text{for all } q,
$$

then the storage becomes

$$
O(P \, N_{\alpha}^2),
$$

where

$$
P = \frac{k(k+1)}{2}
$$

is the number of packed active pairs for `k` active orbitals.

That is precisely the kind of object that can dominate memory long before the
final gradient is formed.

## 6.4 Thread-local full packed two-electron gradients

Even with streamed tiles, memory can still explode if every thread owns a full
copy of the packed active-space two-electron gradient.

Let

$$
P = \frac{k(k+1)}{2},
\qquad
N_{2e} = \frac{P(P+1)}{2}.
$$

Then the packed active-space two-electron gradient length is $N_{2e}$, which
scales like

$$
O(k^4).
$$

Replicating that vector across `n_threads` gives

$$
O(n_{\mathrm{threads}} \, k^4)
$$

temporary memory.

Therefore a streamed matrix-form rewrite is only safe if the reduction strategy
is also changed.

---

## 7. Bounded-Memory Rules

The bounded-memory algorithm should enforce the following rules explicitly.

## 7.1 Never materialize global pair-weight matrices

Do **not** store global

$$
W^{\alpha} \in \mathbb{R}^{N_{\alpha} \times N_{\alpha}},
\qquad
W^{\beta} \in \mathbb{R}^{N_{\beta} \times N_{\beta}}
$$

as persistent intermediate objects for accepted, local, or directional
response.

Instead, build only support-local tiles:

$$
W_{\mathrm{tile}}^{\alpha}
=
L_{\mathrm{tile}} K_{\beta,\mathrm{tile}} R_{\mathrm{tile}}^{\top}.
$$

## 7.2 Never materialize all packed-pair channel images at once

For opposite-spin work:

1. process one sparse packed-pair block at a time;
2. build one mixed coefficient-side tile at a time;
3. immediately contract it into the packed two-electron gradient or overlap
   pullback.

## 7.3 Trim supports before any matrix multiplication

The coefficient block must be trimmed before it enters the streamed matrix-form
kernel.

This turns the true complexity into

$$
O(|\mathcal{A}| \, |\mathcal{B}| \, t)
$$

style local work, rather than

$$
O(N_{\alpha} N_{\beta} t).
$$

## 7.4 Restrict to active unique-pair tiles

The streamed driver should operate only on the set of unique pair indices
actually touched by the current structure or selected-state supports.

This avoids scanning the full

$$
N_{\alpha}^2
\quad \text{or} \quad
N_{\beta}^2
$$

grid when the support is sparse.

## 7.5 Fuse tile build and tile consume

The lifespan of the weight tile should be one iteration:

1. gather partner kernel tile;
2. build local coefficient-side image;
3. contract against the operator tile;
4. discard.

No larger staging area should survive beyond that tile.

## 7.6 Use striped reduction for packed two-electron gradients

For large active spaces, do **not** allocate one full
`packed_active_two_electron_gradient` per thread.

Instead use one of:

1. stripe reduction over pair-of-pairs ranges;
2. block reduction aligned with the packed-pair outer loop;
3. a thread team that owns only one packed-gradient block at a time.

This keeps the temporary memory closer to

$$
O(N_{2e} + n_{\mathrm{threads}} N_{\mathrm{stripe}})
$$

instead of

$$
O(n_{\mathrm{threads}} N_{2e}).
$$

---

## 8. Complexity Summary

Let:

- $t$ be the unique-pair tile size;
- $b$ be the packed-pair sparse block size;
- $s_n^{\alpha}$ and $s_n^{\beta}$ be the trimmed supports of state $n$.

Then the intended bounded-memory complexity is:

| Object | Fully materialized matrix-form | Bounded-memory matrix-form |
| --- | --- | --- |
| Same-spin accepted weights | $O(N_{\alpha}^2 + N_{\beta}^2)$ memory | $O(t^2)$ working memory |
| Same-spin local response | $O(N_{\alpha}^2 + N_{\beta}^2)$ memory | $O(t^2)$ working memory |
| Same-spin directional response | $O(N_{\alpha}^2 + N_{\beta}^2)$ memory | $O(t^2)$ working memory |
| Opposite-spin mixed images | $O(P N_{\alpha}^2)$ or $O(P N_{\beta}^2)$ memory | $O(b \cdot t^2)$ style working memory |
| Packed 2e gradient reduction | $O(n_{\mathrm{threads}} N_{2e})$ temp memory | $O(N_{2e} + n_{\mathrm{threads}} N_{\mathrm{stripe}})$ temp memory |

This is the entire point of the redesign:

1. preserve matrix-multiplication arithmetic;
2. remove the quadratic or packed-family global storage.

---

## 9. Recommended Algorithm Skeleton

For any contraction family, the implementation should follow the same staged
pattern.

```text
for each active tile pair (left_tile, right_tile):
    determine the support window on the operator spin
    determine the matching support window on the partner spin

    gather partner kernel tile

    for each coefficient block or state block that intersects this tile:
        gather the relevant local coefficient subblocks
        build one local weight tile by GEMM
        accumulate local prefactors

    contract the final weight tile against the operator tile
    scatter directly into:
        - structure matrix entry,
        - selected-state gradient,
        - local response buffer,
        - directional response buffer,
        - packed 2e stripe buffer

    discard the tile
```

The critical design constraint is:

$$
\text{No step in this loop should create a persistent object larger than the current tile.}
$$

---

## 10. Implications For The Current Code

The current code base already contains the pieces needed to realize this
algorithm consistently.

## 10.1 Reusable pieces

- Structure-local support trimming:
  [structure_coefficient_blocks.hpp](./matrices/structure_coefficient_blocks.hpp)
- Selected-state support trimming:
  [selected_state_determinant_matrices.hpp](./scf/selected_state_determinant_matrices.hpp)
- Streamed accepted same-spin backward:
  [same_spin_matrix_backward.cpp](./scf/same_spin_matrix_backward.cpp)
- Sparse packed-pair opposite-spin block traversal:
  [opposite_spin_matrix_backward.cpp](./scf/opposite_spin_matrix_backward.cpp)
- Tiled forward structure assembly:
  [full_structure_builder.cpp](./matrices/full_structure_builder.cpp)

## 10.2 Forward paths that should be aligned with the common framework

1. forward structure assembly already uses tiled matrix-form contraction in
   [full_structure_builder.cpp](./matrices/full_structure_builder.cpp), but its
   tile gather and tile consume logic should be treated as the forward baseline
   for the rest of the code rather than as an isolated implementation;
2. selected-state build should continue to accumulate only touched unique-spin
   pairs and then trim exact zero support, rather than ever reintroducing a
   global dense `N_alpha x N_beta` state matrix;
3. directional selected-state build from structure columns should be viewed as
   a forward linear map
   $\delta C^{(n)} = \sum_I \delta u_I^{(n)} T^{(I)}$ and optimized with the
   same support-local accumulation logic, not as a separate special-case path;
4. projected directional structure columns in the outer-response cache should
   reuse the same bounded-memory contraction model as forward structure-matrix
   assembly, since they are mathematically the same kind of structure-pair
   contraction with different coefficient-side prefactors.

## 10.3 Backward paths that should be rewritten into the common framework

1. local same-spin response should stop using scalar accessor callbacks and be
   rewritten as streamed local tile contraction;
2. directional same-spin response should stop using postings-driven scalar
   formulas and instead use
   $\delta C K C^{\top} + C K \delta C^{\top}$ on tiles;
3. directional opposite-spin response should stop building pair entries through
   scalar callbacks and instead contract support-local mixed tiles against the
   current sparse packed-pair block;
4. packed two-electron gradient reduction should stop replicating the full
   packed vector per thread on large active spaces.

## 10.4 What should remain small-object based

The following are acceptable as persistent carriers because they are already
trimmed or structurally sparse:

- `StructureCoefficientBlock`
- `SelectedStateDeterminantCoefficients`
- `DirectionalSelectedStateCoefficients`
- postings or active-pair index lists used only as sparse schedulers

What should **not** be promoted back into persistent objects:

- full same-spin pair-weight matrices;
- full directional same-spin pair-weight matrices;
- all opposite-spin mixed images for every packed pair;
- one full packed two-electron gradient per thread.

---

## 11. Forward And Backward Optimization Roadmap

The optimization plan should be described by dataflow stage rather than by
individual files, because the same contraction model crosses forward and
backward boundaries.

## 11.1 Shared infrastructure to build first

Before splitting into forward and backward work, the code should have one
shared abstraction for "bounded-memory contraction over unique-spin supports".

Conceptually, one contraction instance needs:

1. a left coefficient block `L`;
2. a right coefficient block `R`;
3. one operator-spin tile;
4. one partner-spin tile or packed-pair block;
5. one tile-local consumer that immediately accumulates the result.

At the software level, the common helper should expose four operations:

```text
select_active_tiles(...)
gather_partner_tile(...)
build_weight_tile(...)
consume_weight_tile(...)
```

This helper does not need to know whether the caller is:

- assembling one structure matrix element;
- building one selected-state block;
- forming an accepted same-spin gradient tile;
- forming a local directional tile;
- forming a directional selected-state tile;
- accumulating one opposite-spin packed-pair block.

Once this shared helper exists, the forward and backward paths can reuse the
same memory discipline automatically.

## 11.2 Forward optimization stages

### Stage F1: make structure-pair contraction the reference implementation

The existing forward structure builder in
[full_structure_builder.cpp](./matrices/full_structure_builder.cpp)
already performs the mathematically correct bounded-memory contraction:

1. gather alpha/beta tiles only for the supports of the current structure pair;
2. build the local coefficient-side image by dense matrix multiplication;
3. contract and discard.

This stage means:

1. preserve this algorithmic shape;
2. factor its tile gather and local contraction logic into reusable helpers;
3. avoid introducing alternative forward kernels that rebuild global pair
   families.

### Stage F2: treat selected-state build as a forward contraction problem

The selected-state block build

$$
C^{(n)} = \sum_I u_I^{(n)} T^{(I)}
$$

should be optimized with the same bounded-memory philosophy:

1. accumulate only touched `(alpha, beta)` pairs from the structure blocks;
2. trim support immediately;
3. never build the full dense `N_alpha x N_beta` matrix for each state.

For large state batches, this should eventually be implemented as a streamed
column family contraction over `StructureCoefficientBlock`, rather than as a
one-state-at-a-time special path.

### Stage F3: treat directional selected-state build as the same linear map

The directional selected-state build

$$
\delta C^{(n)} = \sum_I \delta u_I^{(n)} T^{(I)}
$$

is the same forward map with different coefficients.

Therefore the implementation target is:

1. reuse the same structure-block accumulation kernel as in Stage F2;
2. keep only trimmed support-local directional blocks;
3. avoid storing both a full directional structure-column matrix and a second
   full dense unique-spin matrix when only the trimmed directional block is
   needed downstream.

### Stage F4: reuse the same framework for projected directional structure columns

In the outer-response cache, the projected directional structure matrices are
still a structure-space contraction driven by `StructureCoefficientBlock`.

The optimization target is:

1. make the projection build reuse the same support-local contraction helper as
   forward structure assembly;
2. accumulate only the selected columns actually required by the generalized
   eigen response;
3. ensure the local projection payload is consumed immediately rather than
   staged as a larger temporary family.

## 11.3 Backward optimization stages

### Stage B1: treat accepted same-spin backward as the baseline

The accepted same-spin backward path in
[same_spin_matrix_backward.cpp](./scf/same_spin_matrix_backward.cpp)
already demonstrates the correct target shape:

1. restrict to active unique-pair tiles;
2. build support-local weight tiles;
3. contract them directly into active-space gradients.

This path should be treated as the baseline backward algorithm that other
response paths converge to.

### Stage B2: rewrite local same-spin response into streamed tile form

The local same-spin response still computes directional partner scalars through
pairwise accessors.

Mathematically it should instead build, on each tile,

$$
\delta W^{\mathrm{local}}
=
\sum_n \omega_n \,
C^{(n)} \delta K_{\mathrm{tile}} [C^{(n)}]^{\top},
$$

and then consume that tile immediately.

So the implementation goal is:

1. gather one directional partner tile
   $(\delta K_{S,\mathrm{tile}}, \delta K_{H,\mathrm{tile}})$;
2. contract it with the accepted selected-state blocks by GEMM;
3. accumulate directly into one-electron, overlap, and packed-2e pullbacks.

### Stage B3: rewrite directional same-spin response into mixed block form

The directional selected-state response should use the exact mixed formula

$$
\delta C K C^{\top} + C K \delta C^{\top}
$$

on tiles, with the accepted and directional energy prefactors folded in before
the tile is consumed.

The implementation goal is:

1. keep accepted and directional coefficient blocks both trimmed;
2. build only one tile-local mixed image at a time;
3. remove repeated postings intersection and scalar support lookup from the
   hot path.

### Stage B4: rewrite directional opposite-spin response around mixed tiles

The opposite-spin directional path should use the same sparse packed-pair block
driver, but the contraction inside each block should become matrix-form:

1. one sparse packed-pair block defines the current family of partner-spin
   projection operators;
2. one support-local mixed coefficient tile is built for that block;
3. that tile is contracted immediately with the operator-spin sparse block;
4. the block is discarded.

The key change is not the sparse block scheduling itself.
The key change is to stop asking for scalar pair entries inside the sparse
contraction loop.

### Stage B5: replace full thread-local packed-gradient copies by striped reduction

This stage is required for the backward rewrite to remain memory-safe at large
active spaces.

The reduction policy should be:

1. keep thread-local `O(k^2)` buffers if needed;
2. partition the packed pair-of-pairs index space into stripes or blocks;
3. let each worker own only one packed-gradient stripe at a time;
4. merge stripes after the block has been consumed.

This stage is orthogonal to the mathematical contraction rewrite, but it is
required for the final bounded-memory guarantee.

## 11.4 Accepted-context implications

The accepted exact-ctx cache should retain only the objects that are useful
across repeated HVP applications and already have trimmed or structured form.

Good accepted-context residents are:

- prepared active-space one-electron and exact-integral data;
- same-spin pair cache payloads;
- structure coefficient blocks;
- trimmed accepted selected-state blocks;
- small scheduler objects such as active unique-pair index lists.

Objects that should **not** become persistent accepted-context residents are:

- global same-spin weight matrices;
- global directional same-spin weight matrices;
- all mixed opposite-spin channel images;
- per-thread full packed two-electron gradient buffers.

This accepted-context rule is the bridge between forward optimization and
backward optimization:

1. forward stages decide what canonical trimmed objects are built once;
2. backward stages consume only those canonical trimmed objects plus tile-local
   workspaces.

---

## 12. Recommended 7-Round Replacement Order

For implementation, the most practical sequence is to execute the rewrite in
seven rounds rather than as one large forward/backward refactor.

The rounds are intentionally organized so that:

1. early rounds change only shared infrastructure and keep formulas fixed;
2. forward-path rewrites happen before the more delicate backward-response
   rewrites;
3. the packed-gradient memory rewrite lands only after the matrix-form
   contraction paths are already stable;
4. large-system validation is done by `sbatch` on compute nodes rather than by
   local runs.

### Round 1: shared tile/support scheduling infrastructure

Goal:
factor the common bounded-memory indexing rules into reusable helpers without
changing any contraction formulas.

Scope:

1. shared `SupportWindow` logic;
2. shared square-tile indexing and task scheduling;
3. forward and backward callers switched to the same tile-window rules.

Exit criteria:

1. no behavioral change;
2. incremental build passes;
3. no new global dense allocations are introduced.

### Round 2: shared support-local contraction skeleton

Goal:
extract one common contraction skeleton for

$$
\text{gather partner tile}
\rightarrow
\text{build local weight image}
\rightarrow
\text{consume operator tile},
$$

while still keeping the old physics formulas unchanged.

Scope:

1. factor the support-local GEMM pattern now duplicated across forward and
   same-spin streamed paths;
2. keep the coefficient carriers unchanged:
   `StructureCoefficientBlock`,
   `SelectedStateDeterminantCoefficients`,
   `DirectionalSelectedStateCoefficients`;
3. ensure the helper enforces tile-local lifetime for every temporary.

Exit criteria:

1. zero mathematical change;
2. forward and same-spin callers can reuse the same tile-local work pattern;
3. the helper interface matches the four operations from Section 11.1:
   `select_active_tiles`, `gather_partner_tile`, `build_weight_tile`,
   `consume_weight_tile`.

### Round 3: forward accepted-path unification

Goal:
make the forward accepted-path builders use one common bounded-memory
contraction model.

Scope:

1. keep structure-pair contraction in
   [full_structure_builder.cpp](./matrices/full_structure_builder.cpp) as the
   reference implementation;
2. move accepted selected-state build onto the same support-local accumulation
   helper used by the forward structure path;
3. preserve immediate support trimming and avoid any dense
   `N_{\alpha} \times N_{\beta}` state matrix.

Exit criteria:

1. forward structure assembly and accepted selected-state build share the same
   contraction skeleton;
2. accepted selected states remain trimmed carriers suitable for reuse by the
   backward passes.

### Round 4: forward directional-path unification

Goal:
move all forward directional objects onto the same trimmed structure-block
accumulation model.

Progress update (2026-04-21):
the full directional structure builder and the projected outer-response column
builder now share one bounded-memory right-column sweep in
[exact_orbital_second_order_operator.cpp](./scf/exact_orbital_second_order_operator.cpp);
the remaining directional selected-state carriers can be moved onto the same
contraction framework incrementally without reintroducing dense unique-spin
staging.

Scope:

1. directional selected-state build
   $\delta C^{(n)} = \sum_I \delta u_I^{(n)} T^{(I)}$;
2. projected directional structure-column / outer-response cache builds;
3. keep only the trimmed directional blocks needed downstream.

Exit criteria:

1. no directional path stages a dense unique-spin matrix as a persistent
   object;
2. accepted and directional forward blocks share the same accumulation helper.

### Round 5: same-spin backward rewrite

Goal:
make same-spin backward contraction fully matrix-form on bounded-memory tiles,
starting from accepted/local response and then absorbing the directional
same-spin formulas.

Progress update (2026-04-21):
the accepted same-spin path, the outer-response directional same-spin path,
and the local-response same-spin path now all stream support-local weight tiles
directly from trimmed selected-state blocks; the hot loops no longer depend on
postings/scalar-accessor queries over unique-spin pairs.

Scope:

1. replace local same-spin scalar accessors by streamed local tile contraction;
2. replace directional same-spin scalar accessors by tile-local
   $\delta C K C^{\top} + C K \delta C^{\top}$ contraction;
3. keep accepted and directional coefficient blocks both trimmed.

Exit criteria:

1. same-spin backward hot paths no longer depend on postings-driven scalar
   pair lookups inside the contraction loop;
2. tile-local matrix multiplication is again the dominant arithmetic kernel.

### Round 6: opposite-spin mixed-tile rewrite

Goal:
rewrite the opposite-spin directional / mixed response around sparse packed
pair blocks plus support-local matrix-form contraction.

Scope:

1. keep the existing sparse packed-pair block scheduler;
2. replace scalar callback assembly inside each sparse block by local mixed
   tile contraction;
3. consume each mixed tile immediately into the response buffers.

Exit criteria:

1. the opposite-spin hot path no longer builds mixed responses through scalar
   pair callbacks;
2. sparse scheduling remains, but arithmetic inside each block is matrix-form.

Implementation status:

1. the opposite-spin packed two-electron accepted, directional, and local
   block contractions now reuse sparse packed-pair blocks together with
   streamed support-local matrix-form images;
2. the block-internal `matrix_entry(row,col)` / `coeff(row,col)` callback path
   has been removed from those packed-gradient contractions;
3. reusable scratch (`partner_tile`, `partner_push`, `image_tile`) is kept per
   streamed contraction to avoid repeated matrix allocation and copy churn.

### Round 7: striped packed-gradient reduction and rollout verification

Goal:
finish the large-active-space memory discipline and validate the full rewrite
under compute-node conditions.

Scope:

1. replace full per-thread packed-gradient copies by striped reduction;
2. expose the new paths behind runtime switches if needed for staged rollout;
3. benchmark and validate on compute nodes with `sbatch`;
4. reserve local execution for very small systems such as `F2`.

Exit criteria:

1. no `O(n_{\mathrm{threads}} N_{2e})` packed-gradient temporary remains on the
   large-system path;
2. the end-to-end exact-integral HVP / response path is stable on large active
   spaces without OOM;
3. performance validation is recorded from `sbatch` jobs, not from local large
   runs.

Implementation status:

1. packed two-electron writes in the streamed same-spin backward path and the
   determinant-pair active-gradient fallback now go through a striped reducer
   instead of one full dense packed vector per thread;
2. serial paths still write directly into the final dense packed gradient, but
   parallel paths keep only small per-thread stripe caches and flush them into
   the shared target by stripe;
3. compute-node validation is now recorded entirely through `sbatch`, not
   local large-system runs:
   `1941005` (`testdata/vbscf/241_VBSCF.xmi`) completed in `00:00:45` with batch-step
   `MaxRSS=2136304K`;
   `1941007` (`testdata/vbscf/241_VBSCF.xmi`) completed in `00:01:18` with batch-step
   `MaxRSS=2200384K`,
   `full_cached_external_avg_wall_time_seconds=3.950472327`,
   `outer_only_cached_external_avg_wall_time_seconds=3.78322635867`;
   `1941006` (`testdata/vbscf/10698_VBSCF.xmi`) completed in `00:04:16` with batch-step
   `MaxRSS=12004112K`,
   `full_cached_external_avg_wall_time_seconds=12.943628066`,
   `outer_only_cached_external_avg_wall_time_seconds=12.2217528767`;
4. the fresh `241` versus `10698` comparison indicates that the
   unique-spin-to-structure contraction is no longer the dominant scaling term
   on this path:
   total `full_cached_external` grows by about `3.28x`, but
   `full_cached_diag_avg_outer_response_wall_time_seconds` stays nearly flat
   (`2.22712674667 -> 2.18340459967`);
   the remaining growth is instead concentrated in
   `full_cached_diag_avg_h1e_fused_wall_time_seconds`
   (`1.504459765 -> 9.94922637`),
   `full_cached_diag_avg_active_2e_wall_time_seconds`
   (`0.213938514333 -> 0.80300345`),
   and
   `full_cached_diag_avg_outer_response_orbital_pullback_wall_time_seconds`
   (`0.146447167667 -> 0.547547795`);
5. an audit of the remaining dense packed-gradient writers found only
   serial or single-output helpers in
   [exact_orbital_second_order_operator.cpp](./scf/exact_orbital_second_order_operator.cpp);
   no major exact-integral parallel path still allocates one full
   `O(n_{\mathrm{threads}} N_{2e})` packed-gradient temporary on the current
   matrix-form large-system route.

This order keeps correctness checks local, keeps the memory-risk changes
separated from the algebraic rewrites, and makes every round produce one clear
replaceable improvement.

---

## 13. Orthogonalized Direct-CI Action for Complete Fixed-Spin Spaces

The factorized nonorthogonal action stores four dense unique-string matrices,

$$
S_{\alpha},\quad H_{\alpha},\quad S_{\beta},\quad H_{\beta},
$$

and applies them to each Davidson block by Kronecker-structured matrix
multiplication.  For block width $b$, the five dense products require

$$
N_{\mathrm{kron}}
=
b\left(
4U_{\alpha}U_{\beta}^{2}
+
6U_{\alpha}^{2}U_{\beta}
\right)
$$

floating-point operations before the opposite-spin channels are included.
This scaling is avoidable when each unique-spin set is the complete
fixed-particle-number Fock space.

At a fixed accepted orbital point, let the active-orbital overlap be

$$
S_{\mathrm{act}}
=
\Phi^{\mathrm T}S_{\mathrm{AO}}\Phi
=
R^{\mathrm T}R,
$$

where $R$ is the upper Cholesky factor.  The orthonormal active orbitals are

$$
\chi=\Phi R^{-1},
$$

and the induced fixed-spin determinant transformation is the exterior power

$$
T_{\sigma}=\bigwedge^{n_{\sigma}}R.
$$

For a coefficient block $X$ on the unique alpha--beta string product, the
orthonormal determinant coefficients are

$$
Y=T_{\alpha}XT_{\beta}^{\mathrm T}.
$$

The exact Hamiltonian and overlap actions in the original nonorthogonal
product basis are therefore

$$
H X
=
T_{\alpha}^{\mathrm T}
\sigma_{\mathrm{orth}}(Y)
T_{\beta},
$$

and

$$
S X
=
T_{\alpha}^{\mathrm T}Y T_{\beta}.
$$

The surrounding sparse structure expansion supplies the maps from structure
coefficients to $X$ and from these product-space images back to the structure
basis.  Thus this replacement changes only the fixed-point $H/S$ action; it
does not alter the VB structure definition.

The compound matrices $T_{\sigma}$ are never assembled.  The triangular
factor $R$ is decomposed into diagonal orbital scalings and elementary orbital
shears.  Applying the corresponding exterior transforms costs

$$
O\!\left(N_{\mathrm{FCI}}n_{\mathrm{act}}^{2}b\right)
$$

and requires only lists of determinant pairs connected by each shear.  In the
orthonormal basis, the Slater--Condon connection counts per determinant are

$$
z_{\sigma}
=
1+n_{\sigma}v_{\sigma}
+
\binom{n_{\sigma}}{2}\binom{v_{\sigma}}{2},
$$

and

$$
z_{\alpha\beta}
=
n_{\alpha}v_{\alpha}n_{\beta}v_{\beta},
$$

where $v_{\sigma}=n_{\mathrm{act}}-n_{\sigma}$.  A target-driven exact sigma
action consequently has the formal cost

$$
O\!\left[
N_{\mathrm{FCI}}
\left(z_{\alpha}+z_{\beta}+z_{\alpha\beta}\right)b
\right]
=
O\!\left(N_{\mathrm{FCI}}n_{\mathrm{act}}^{4}b\right).
$$

The implementation verifies determinant-space completeness from the occupied
orbital bit patterns rather than from the dimension alone.  Incomplete VB
string spaces continue to use the exact nonorthogonal factorized action,
because an exterior transform restricted to an incomplete space is not closed
and would not be mathematically equivalent.  For complete spaces, an exact
FLOP planner selects direct CI only when its predicted fixed-point work is
smaller.  The Slater--Condon graph is generated directly from occupied--virtual
single and double substitutions followed by determinant-bitmask lookup; it
does not contain an $O(U_{\sigma}^{2})$ all-pairs setup scan.  Its construction
therefore has the same
$O(U_{\sigma}n_{\mathrm{act}}^{4})$ connection-count scaling as the retained
same-spin graph.

### 13.1 Exact directional action

The complete-space representation can also be differentiated without returning
to the nonorthogonal cofactor channels.  For an active-overlap direction
$\delta S_{\mathrm{act}}$, define

$$
A
=
R^{-\mathrm T}\delta S_{\mathrm{act}}R^{-1}
=
E^{\mathrm T}+E,
$$

where $E$ is upper triangular, with

$$
E_{ii}=\frac{1}{2}A_{ii},
\qquad
E_{ij}=A_{ij}\quad(i<j).
$$

The Cholesky-factor direction is then

$$
\delta R=ER.
$$

Writing $Q=R^{-1}$ and denoting the accepted orthonormal one-electron matrix by
$\bar h=Q^{\mathrm T}hQ$, its exact direction is

$$
\delta\bar h
=
Q^{\mathrm T}(\delta h)Q
-
E^{\mathrm T}\bar h
-
\bar h E.
$$

Let $P(Q)$ be the induced transformation on packed symmetric orbital pairs,
and let

$$
D
=
\left.
\frac{\mathrm d}{\mathrm d\epsilon}
P(I-\epsilon E)
\right|_{\epsilon=0}.
$$

For the accepted orthonormal pair kernel $\bar G$, the corresponding exact
two-electron direction is

$$
\delta\bar G
=
P(Q)^{\mathrm T}(\delta G)P(Q)
+
D^{\mathrm T}\bar G
+
\bar G D.
$$

The exterior transformations are differentiated through the same sequence of
diagonal scalings and elementary orbital shears used by the primal action.  No
compound matrix or compound-matrix derivative is formed.  For

$$
Y=T_{\alpha}XT_{\beta}^{\mathrm T},
$$

the determinant-coefficient direction is

$$
\delta Y
=
(\delta T_{\alpha})XT_{\beta}^{\mathrm T}
+
T_{\alpha}X(\delta T_{\beta})^{\mathrm T}.
$$

If $Z=\sigma_{\mathrm{orth}}(Y)$, its direction is

$$
\delta Z
=
(\delta\sigma_{\mathrm{orth}})(Y)
+
\sigma_{\mathrm{orth}}(\delta Y).
$$

Therefore the complete Hamiltonian image direction is

$$
\begin{aligned}
\delta(HX)
={}&
(\delta T_{\alpha})^{\mathrm T}ZT_{\beta}
+T_{\alpha}^{\mathrm T}(\delta Z)T_{\beta}\\
&+T_{\alpha}^{\mathrm T}Z(\delta T_{\beta}),
\end{aligned}
$$

and the overlap-image direction follows by replacing $Z$ and $\delta Z$ with
$Y$ and $\delta Y$.  This route removes the directional
$U_{\sigma}\times U_{\sigma}$ cofactor-channel matrices from complete-space
structure response.

The completeness requirement applies to the fixed-spin determinant carriers,
not to the selected VB structure subspace.  Let $B$ expand an arbitrary set of
structure coefficients into the complete alpha--beta determinant product
space.  The projected action is exactly

$$
H_{\mathrm{str}}x=B^{\mathrm T}H_{\mathrm{det}}Bx,
\qquad
S_{\mathrm{str}}x=B^{\mathrm T}S_{\mathrm{det}}Bx.
$$

Consequently, $B$ may contain all structures, three arbitrarily chosen sparse
structures, or any other subspace.  No closure of the structure subspace is
required.  Closure is required only for applying $T_{\sigma}$ within each
fixed-spin string carrier.  A dedicated three-structure test compares this
projection with explicit nonorthogonal determinant-pair $H/S$ assembly and
obtains maximum absolute errors below $2\times10^{-10}$.

### 13.2 Direct active-integral adjoint and exact HVP

The accepted active-space gradient can be obtained without returning to the
nonorthogonal cofactor representation.  For selected state $n$, define

$$
Y_n=T_{\alpha}C_nT_{\beta}^{\mathrm T},
\qquad
Q_n=\sigma_{\mathrm{orth}}(Y_n)-E_nY_n.
$$

Let $\mathcal A_h(L,R)$ and $\mathcal A_G(L,R)$ denote the adjoints of the
orthonormal direct-CI sigma action with respect to the one-electron matrix and
packed-pair kernel, respectively.  Let $\mathcal A_K(L,R)$ denote the adjoint
of an infinitesimal one-body generator acting through the exterior
transformations.  With normalized state weights $w_n$, the orthonormal
integral and generator adjoints are

$$
\bar D_h
=
\sum_n w_n\mathcal A_h(Y_n,Y_n),
$$

$$
\bar D_G
=
\sum_n w_n\mathcal A_G(Y_n,Y_n),
$$

and

$$
\Gamma
=
2\sum_n w_n\mathcal A_K(Q_n,Y_n).
$$

The residual term $Q_n$ is essential.  It accounts for the dependence of the
exterior transformations on the active overlap and therefore supplies the
overlap part of the nonorthogonal orbital gradient.

Write $A=R^{-1}$ and let $P(A)$ be the packed symmetric-pair transform.  The
direct integral pullbacks are

$$
D_h=A\bar D_hA^{\mathrm T},
$$

and

$$
D_G=P(A)\bar D_GP(A)^{\mathrm T}.
$$

Define the relative Cholesky-generator adjoint

$$
\begin{aligned}
B={}&\Gamma
-\bar h\bar D_h^{\mathrm T}
-\bar h^{\mathrm T}\bar D_h\\
&-\mathcal P^{*}
\left(
\bar G\bar D_G^{\mathrm T}
+\bar G^{\mathrm T}\bar D_G
\right),
\end{aligned}
$$

where $\mathcal P^{*}$ is the adjoint of the infinitesimal packed-pair
transform at the identity.  The symmetric normalized-overlap adjoint $C$ is

$$
C_{ii}=\frac{1}{2}B_{ii},
\qquad
C_{ij}=C_{ji}=\frac{1}{2}B_{ij}quad(i<j),
$$

and the original nonorthogonal overlap adjoint is

$$
D_S=ACA^{\mathrm T}.
$$

The exact HVP differentiates this complete reverse sweep.  If $\delta Y_n$
contains both the local integral response and the selected-state response,
then

$$
\delta Q_n
=
\delta\sigma_{\mathrm{orth}}(Y_n)
+\sigma_{\mathrm{orth}}(\delta Y_n)
-E_n\delta Y_n
-\delta E_nY_n.
$$

The directional adjoints are

$$
\delta\bar D_h
=
\sum_n w_n
\left[
\mathcal A_h(\delta Y_n,Y_n)
+\mathcal A_h(Y_n,\delta Y_n)
\right],
$$

$$
\delta\bar D_G
=
\sum_n w_n
\left[
\mathcal A_G(\delta Y_n,Y_n)
+\mathcal A_G(Y_n,\delta Y_n)
\right],
$$

and

$$
\delta\Gamma
=
2\sum_n w_n
\left[
\mathcal A_K(\delta Q_n,Y_n)
+\mathcal A_K(Q_n,\delta Y_n)
\right].
$$

Differentiating the three pullbacks above gives the active-integral component
of the exact matrix-free Hessian--vector product.  The direct adjoint traverses
the same Slater--Condon graph as the sigma action, so its formal cost per
selected state is

$$
O\!\left[
N_{\mathrm{FCI}}
\left(z_{\alpha}+z_{\beta}+z_{\alpha\beta}\right)
\right]
=
O\!\left(N_{\mathrm{FCI}}n_{\mathrm{act}}^4\right),
$$

with working storage independent of $U_{\alpha}^2+U_{\beta}^2$.  Once the
accepted direct-CI action and selected-state coefficient blocks exist, the
ordered same-spin pair evaluations are no longer part of the persistent HVP
state.  Incomplete fixed-spin spaces retain the streamed cofactor adjoint,
because the direct exterior representation is not exact there.

A 32-core Slurm measurement on LOFLEA confirms the memory consequence of this
replacement.  Releasing the accepted ordered-pair payload changed the reported
cofactor storage from $1{,}166{,}697{,}760$ bytes to zero and reduced peak RSS
from $10{,}867{,}100$ KiB to $7{,}130{,}252$ KiB, a $34.4\%$ reduction.  The
single-run wall time for the three-component response benchmark decreased from
$50.28$ s to $46.59$ s.  The local, structure, and combined-outer response
infinity norms remained $0.282004273479$, $0.300863442993$, and
$0.0203509012709$, respectively, and the component-additivity relative error
was $1.07\times10^{-14}$.

A 32-core Slurm comparison on LOFLEA isolates this replacement.  The baseline
already used the orthogonal direct-CI fixed-point action but retained the
cofactor-channel directional action; the candidate differentiates the
orthogonal action directly:

| Component | Cofactor direction / s | Orthogonal direction / s | Speedup |
|---|---:|---:|---:|
| Structure response only | 19.9471 | 11.3440 | 1.76 |
| Complete outer response | 21.6774 | 14.4919 | 1.50 |
| Directional structure-image stage within the outer response | 9.6613 | 2.2684 | 4.26 |

The complete benchmark peak RSS decreased from $13.14$ GiB to $11.52$ GiB
because the accepted dense same-spin factors and channel indices are no longer
retained for a complete-space action.  The structure-only response infinity
norm remained exactly $0.300863442993$ in the printed precision, and the
outer-component additivity error was $1.17\times10^{-14}$.

Independent validation compares the exterior transforms with explicitly
formed compound matrices and compares the full nonorthogonal $H/S$ action with
direct determinant-pair evaluation.  Unit-test maximum absolute errors are of
order $10^{-13}$ for the exterior transform and $10^{-16}$ for the small
end-to-end $H/S$ action.  A 32-core Slurm comparison at the same orbital point
gave:

| System | $U_{\alpha}\times U_{\beta}$ | Factorized action / s | Direct-CI action / s | Speedup | max $|\Delta Hx|$ | max $|\Delta Sx|$ |
|---|---:|---:|---:|---:|---:|---:|
| CERRAS | $462\times462$ | 1.8777 | 0.09491 | 19.79 | $2.57\times10^{-12}$ | $5.46\times10^{-14}$ |
| LOFLEA | $924\times924$ | 26.7548 | 1.3166 | 20.32 | $1.59\times10^{-12}$ | $7.82\times10^{-14}$ |

The measured gain exceeds the five-GEMM-only FLOP ratio because the old count
does not include its opposite-spin channel traversal.  The direct-CI setup is
performed once per accepted orbital point.  Identical alpha and beta string
spaces share both the Slater--Condon graph and exterior-transform schedule.
The final production construction took approximately $0.49$ s for CERRAS and
$1.10$ s for LOFLEA.  After retaining the small orthogonal-integral payload
needed by the exact directional action, the measured persistent representation
sizes were $3.41$ MB and $9.38$ MB, respectively.  The direct active-integral
adjoint described above now also permits the accepted-point pair-evaluation
payload to be released before subsequent HVP applications.

This milestone removes the dense unique-string quadratic action from complete
fixed-spin spaces and removes the cofactor pair-response traversal from their
accepted gradients and exact HVPs.  End-to-end TNHVP speedups still depend on
the admitted outer-response frequency and on the AO-to-active integral and
orbital-pullback costs outside the structure representation.

---

## 14. Short Conclusion

The key conclusion is:

$$
\text{The right fix is not "scalar to avoid OOM" and not "global matrix to go fast".}
$$

For general incomplete unique-string spaces, the bounded-memory contraction is:

$$
\text{trimmed coefficient blocks}
+
\text{active tile scheduling}
+
\text{support-local matrix multiplication}
+
\text{streamed reduction}.
$$

That framework applies to contractions from incomplete unique-spin-string
spaces into structure-space or selected-state-space:

1. forward structure matrices;
2. selected-state builds;
3. accepted same-spin gradients;
4. local response;
5. directional response;
6. opposite-spin packed-pair contractions.

Complete fixed-spin spaces admit the additional exact orthogonal direct-CI
representation derived in Section 13.  The implementation therefore uses two
mathematically defined regimes rather than molecule-specific tuning: streamed
support-local contraction for incomplete spaces, and exterior transformation
plus Slater--Condon sigma action for complete spaces.
