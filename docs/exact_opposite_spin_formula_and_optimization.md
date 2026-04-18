# Exact Opposite-Spin 公式与后续优化

## 1. 范围

本文只讨论当前标准 VB determinant 路径中的 **exact active-space two-electron integrals** 下的 opposite-spin 通道：

- 不是 RI 近似；
- 不是 exact-separator / rooted-component-tree 特化路径；
- 对应当前 production 代码里的几个核心实现：
  - `src/vb/matrices/full_determinant_pair_evaluator.cpp`
  - `src/vb/matrices/spin_pair_utils.cpp`
  - `src/vb/matrices/same_spin_pair_cache.cpp`
  - `src/vb/matrices/full_structure_builder.cpp`
  - `src/vb/scf/opposite_spin_matrix_channels.cpp`
  - `src/vb/scf/opposite_spin_matrix_backward.cpp`

本文目标有两个：

1. 把当前 exact opposite-spin 的数学公式写清楚；
2. 分析当前还剩下哪些瓶颈，以及怎样继续优化。

## 2. 记号

设一个 full determinant 写成

$$
D_L = (\alpha_L,\beta_L), \qquad D_R = (\alpha_R,\beta_R),
$$

其中

$$
\alpha_L = (p_1^L,\dots,p_{n_\alpha}^L), \qquad
\alpha_R = (p_1^R,\dots,p_{n_\alpha}^R),
$$

$$
\beta_L = (q_1^L,\dots,q_{n_\beta}^L), \qquad
\beta_R = (q_1^R,\dots,q_{n_\beta}^R).
$$

对应的同自旋 overlap 子矩阵定义为

$$
S^\alpha_{ba} = \langle p_b^R | p_a^L \rangle,
\qquad
S^\beta_{dc} = \langle q_d^R | q_c^L \rangle.
$$

它们的行列式记为

$$
s_\alpha = \det S^\alpha,
\qquad
s_\beta = \det S^\beta.
$$

### 2.1 一阶余子式矩阵

代码里 `calc_cofactor_1st(...)` 返回的是一阶 deleted-minor 矩阵。本文记为

$$
C^\alpha_{ba}
=
\frac{\partial \det S^\alpha}{\partial S^\alpha_{ba}},
\qquad
C^\beta_{dc}
=
\frac{\partial \det S^\beta}{\partial S^\beta_{dc}}.
$$

对于 regular determinant pair（`nullity = 0`），有标准关系

$$
C^\alpha = s_\alpha \left(S^\alpha\right)^{-T},
\qquad
C^\beta = s_\beta \left(S^\beta\right)^{-T}.
$$

对于 `nullity = 1`，代码仍然通过 SVD 闭式构造 \(C^\sigma\)，所以 forward opposite-spin 不要求 pair 一定 regular；只要某一侧 `nullity >= 2`，opposite-spin 就直接为 0。

## 3. Determinant-Pair 标量公式

### 3.1 full determinant 的 exact opposite-spin Coulomb 项

当前 exact forward 在 `full_determinant_pair_evaluator.cpp` 中计算的 opposite-spin 标量为

$$
V_{LR}^{\mathrm{os}}
=
\sum_{a=1}^{n_\alpha}
\sum_{b=1}^{n_\alpha}
\sum_{c=1}^{n_\beta}
\sum_{d=1}^{n_\beta}
C^\alpha_{ba}
C^\beta_{dc}

\left(
q_d^R q_c^L
\middle|
p_b^R p_a^L
\right).
$$

这里没有 exchange 项，因为 alpha 与 beta 自旋不同，只有 Coulomb mixed term。

这正对应 `src/vb/matrices/full_determinant_pair_evaluator.cpp` 里：

- 先取 `alpha_cofactor_1st` 与 `beta_cofactor_1st`
- 再做四重循环
- 查 exact two-electron integral
- 累加

### 3.2 full determinant 总 Hamiltonian 的拼接

当前 full determinant pair 的总 Hamiltonian 写成

$$
H_{LR}
=
H^\alpha_{LR}\, s_\beta
+
H^\beta_{LR}\, s_\alpha
+
V_{LR}^{\mathrm{os}},
$$

其中

- \(H^\alpha_{LR}\) 是 alpha same-spin 子问题给出的 one-electron + same-spin two-electron 总和；
- \(H^\beta_{LR}\) 是 beta same-spin 子问题给出的 one-electron + same-spin two-electron 总和；
- \(V_{LR}^{\mathrm{os}}\) 是上面的 mixed opposite-spin Coulomb 项。

这正是 `combine_spin_pair_evaluations(...)` 的实现。

## 4. Regular Pair 的原始 \(\phi^{\mathrm{os}}\) 公式

对于 active-space backward，代码在 `spin_pair_utils.cpp` 中还使用了 regular pair 的“原始 phi”形式。设

$$
X^\alpha = \left(S^\alpha\right)^{-1},
\qquad
X^\beta = \left(S^\beta\right)^{-1}.
$$

则 regular pair 下有

$$
V_{LR}^{\mathrm{os}}
=
s_\alpha s_\beta \, \phi_{LR}^{\mathrm{os}},
$$

其中

$$
\phi_{LR}^{\mathrm{os}}
=
\sum_{a=1}^{n_\alpha}
\sum_{b=1}^{n_\alpha}
\sum_{c=1}^{n_\beta}
\sum_{d=1}^{n_\beta}
X^\alpha_{ab}
X^\beta_{cd}
\left(
q_d^R q_c^L
\middle|
p_b^R p_a^L
\right).
$$

这里的下标方向与代码中的内部矩阵约定一致，本质上就是把 forward 的余子式公式除以 \(s_\alpha s_\beta\)。

对 \(\phi^{\mathrm{os}}\) 的导数也很直接：

$$
\frac{\partial \phi^{\mathrm{os}}}{\partial X^\alpha_{ab}}
=
\sum_{c,d}
X^\beta_{cd}
\left(
q_d^R q_c^L
\middle|
p_b^R p_a^L
\right),
$$

$$
\frac{\partial \phi^{\mathrm{os}}}{\partial X^\beta_{cd}}
=
\sum_{a,b}
X^\alpha_{ab}
\left(
q_d^R q_c^L
\middle|
p_b^R p_a^L
\right).
$$

这正对应 `compute_opposite_spin_original_phi(...)` 中同时累加

- `phi`
- `alpha_inverse_overlap_gradient`
- `beta_inverse_overlap_gradient`

的逻辑。

## 5. Packed-Pair / Channel 形式

### 5.1 packed pair 核

定义 packed pair 索引

$$
P \leftrightarrow (r,s), \qquad r \ge s,
$$

packed-pair 空间维数为

$$
N_P = \frac{n_{\mathrm{act}}(n_{\mathrm{act}}+1)}{2}.
$$

exact 两电子核在 packed-pair 空间中记为

$$
G_{QP} = (u v | r s),
\qquad
Q \leftrightarrow (u,v), \quad P \leftrightarrow (r,s).
$$

### 5.2 same-spin pair 对应的 packed-pair 向量

对一个 alpha ordered determinant pair \((\alpha_L,\alpha_R)\)，定义一阶余子式投影向量

$$
u^\alpha(P)
=
\sum_{a,b \,:\, P=(p_b^R,p_a^L)}
C^\alpha_{ba}.
$$

对 regular pair，还可以定义 inverse-overlap 投影向量

$$
x^\alpha(P)
=
\sum_{a,b \,:\, P=(p_b^R,p_a^L)}
X^\alpha_{ab}.
$$

beta 侧同理定义 \(u^\beta(Q)\)、\(x^\beta(Q)\)。

于是 determinant-pair opposite-spin 可以重写为

$$
V_{LR}^{\mathrm{os}}
=
\sum_{Q=1}^{N_P}
\sum_{P=1}^{N_P}
u^\beta(Q)\, G_{QP}\, u^\alpha(P),
$$

而 regular phi 形式可以写为

$$
\phi_{LR}^{\mathrm{os}}
=
\sum_{Q=1}^{N_P}
\sum_{P=1}^{N_P}
x^\beta(Q)\, G_{QP}\, x^\alpha(P).
$$

### 5.3 缓存形式

`same_spin_pair_cache.cpp` 里对每个 ordered same-spin pair 实际缓存的是：

1. 稀疏 packed-pair 系数向量

$$
u^\sigma(P) \quad \text{或} \quad x^\sigma(P),
$$

2. 它们经核作用后的投影

$$
\widetilde{u}^\sigma(Q)
=
\sum_{P} G_{QP} u^\sigma(P),
$$

$$
\widetilde{x}^\sigma(Q)
=
\sum_{P} G_{QP} x^\sigma(P).
$$

于是标量收缩就变成

$$
V_{LR}^{\mathrm{os}}
=
\sum_Q
u^\beta(Q)\, \widetilde{u}^\alpha(Q)
=
\sum_P
u^\alpha(P)\, \widetilde{u}^\beta(P),
$$

$$
\phi_{LR}^{\mathrm{os}}
=
\sum_Q
x^\beta(Q)\, \widetilde{x}^\alpha(Q)
=
\sum_P
x^\alpha(P)\, \widetilde{x}^\beta(P).
$$

这就是当前 `contract_opposite_spin_first_order_projections(...)` 和
`compute_opposite_spin_original_phi(...)` 的 cache 快路径。

## 6. Structure Basis 的 Matrix Form

设 unique alpha / beta determinant 索引分别为 \(i,i'\) 与 \(j,j'\)，结构 \(I\) 的 determinant 展开系数矩阵记为

$$
C_I \in \mathbb{R}^{N_\alpha \times N_\beta}.
$$

对每个 packed pair \(P\)，定义 alpha / beta 通道矩阵

$$
U^\alpha_P(i,i') = u^\alpha_{ii'}(P),
\qquad
U^\beta_P(j,j') = u^\beta_{jj'}(P).
$$

再定义 beta 侧经过 exact packed kernel 投影后的通道矩阵

$$
\widetilde{U}^\beta_P(j,j')
=
\sum_{Q=1}^{N_P}
G_{PQ} U^\beta_Q(j,j').
$$

那么 structure Hamiltonian 的 opposite-spin 项可写成

$$
H_{IJ}^{\mathrm{os}}
=
\sum_{P=1}^{N_P}
\operatorname{Tr}
\left(
C_I^{\top}
U^\alpha_P
C_J
\left[\widetilde{U}^\beta_P\right]^{\top}
\right).
$$

这正是当前 `build_matrix_form_structure_matrices(...)` 里做的事情：

1. 外层遍历 packed pair \(P\)；
2. 先做
   $$
   C_J [\widetilde{U}^\beta_P]^\top
   $$
3. 再左乘
   $$
   U^\alpha_P
   $$
4. 对 \(P\) 求和。

因此 exact forward 已经不再显式遍历 full determinant pairs，而是改成了

$$
\text{packed-pair channel sum}
\quad+\quad
\text{unique-spin matrix contraction}.
$$

## 7. 当前 exact opposite-spin 已经优化到哪一步

截至当前实现，exact opposite-spin 已经完成了三件关键事情：

### 7.1 determinant-pair 级别的 cofactor / inverse payload 只算一次

对每个 ordered unique same-spin pair，first-order cofactor 与 inverse-overlap
投影都已缓存，不再在 full determinant pair 上重复构造。

### 7.2 forward 已经消除了 full determinant-pair 主循环

当前 exact structure matrix build 的主路径已经是

$$
\sum_P
U^\alpha_P C_J [\widetilde{U}^\beta_P]^\top
$$

这种 packed-pair channel 的矩阵形式。

### 7.3 backward 也已经有 packed-channel 形式

exact backward 中，opposite-spin packed 2e adjoint 已经写成

$$
\frac{\partial E_{\mathrm{SA}}^{\mathrm{os}}}{\partial G_{QP}}
=
\sum_n w_n
\operatorname{Tr}
\left(
\left[C^{(n)}\right]^\top
U^\alpha_P
C^{(n)}
\left[U^\beta_Q\right]^\top
\right),
$$

也就是 `opposite_spin_matrix_backward.cpp` 里对 packed-pair-of-pairs 的矩阵收缩。

所以，exact opposite-spin 现在已经不是 “determinant pair 暴力四重循环” 的旧形态了。

## 8. 当前剩余瓶颈到底在哪里

当前 opposite-spin 剩余热点主要不在“同一个 determinant pair 内的四重电子循环”本身，而在下面两件事：

### 8.1 packed-pair channel 求和

即使 determinant-pair 主循环已经消掉，forward / backward 仍然需要显式遍历

$$
P = 1,\dots,N_P,
\qquad
N_P = \frac{n_{\mathrm{act}}(n_{\mathrm{act}}+1)}{2}.
$$

每个 channel 都对应一块

$$
U^\alpha_P,\quad U^\beta_P,\quad \widetilde{U}^\alpha_P,\quad \widetilde{U}^\beta_P.
$$

这就是 current exact opposite-spin 与 same-spin 的最大区别：

- same-spin 只有少数几块全局矩阵 \(S^\alpha,H^\alpha,S^\beta,H^\beta\)；
- opposite-spin 是一整族按 packed pair 编号的 channel 矩阵，然后再对 channel 求和。

### 8.2 structure / selected-state 的矩阵装配

即使 channel 本身都缓存好了，最后仍然要做

$$
U^\alpha_P C_J [\widetilde{U}^\beta_P]^\top
$$

以及

$$
\operatorname{Tr}(C_I^\top \cdots)
$$

这类矩阵装配。对大结构空间来说，这一步本身就会成为显著成本。

## 9. 继续优化的优先路线

下面按“保持 exact 不变”与“可选进一步重构”两层来分析。

### 9.1 路线 A：channel screening

这是最直接、也最稳妥的 exact 优化。

当前 forward 在 opposite-spin 通道上仍然是

$$
\texttt{for } P = 1,\dots,N_P
$$

逐个 packed pair 扫过去。

但很多 channel 其实是空的，或者对当前 structure / selected-state 权重几乎没有贡献。可以做三层筛选：

1. 预先记录
   $$
   \mathcal{P}_\alpha = \{ P : U^\alpha_P \neq 0 \},
   \qquad
   \mathcal{P}_\beta = \{ P : U^\beta_P \neq 0 \}.
   $$
2. forward 时只遍历
   $$
   \mathcal{P}_\alpha \cap \widetilde{\mathcal{P}}_\beta
   $$
   这样的有效 channel。
3. backward 时对 packed-pair-of-pairs 只遍历非零权重块，而不是完整的 \(N_P^2\)。

这条路线完全保持 exact，不改数学，只是更 aggressively 地利用 sparsity。

### 9.2 路线 B：把 channel 装配变成更粗粒度的 BLAS-3 / batched GEMM

当前 opposite-spin forward 的核心操作是对每个 \(P\) 做

$$
C_J [\widetilde{U}^\beta_P]^\top
$$

和

$$
U^\alpha_P \bigl(C_J [\widetilde{U}^\beta_P]^\top\bigr).
$$

这本质上是很多个“小批次”的矩阵乘法。

可以继续优化成：

1. 对多个 channel \(P\) 一起批量做 GEMM；
2. 对多个 structure \(J\) 一起批量做 GEMM；
3. 让数据布局更接近连续 block，减少 cache miss 与 sparse/dense 反复切换的代价。

这一类优化不会改公式，但会显著影响常数。

### 9.3 路线 C：专门压缩 structure accumulation

当前总能量 / 矩阵构建里，真正输出的对象是 structure matrix。即使 determinant-pair 已经消掉，最后仍要做

$$
H_{IJ}^{\mathrm{os}}
=
\sum_P
\operatorname{Tr}
\left(
C_I^\top U^\alpha_P C_J [\widetilde{U}^\beta_P]^\top
\right).
$$

因此还可以继续考虑：

1. 预先把所有 \(C_I\) 堆叠成 block；
2. 一次性生成多个 \(J\) 的 image；
3. 把一批 Frobenius inner product 变成 GEMM / SYRK 形式的 Gram 装配。

如果结构数很大，这条路线的重要性会逐渐上升。

### 9.4 路线 D：exact packed kernel 的显式因子分解

这条路线是可选的，不是当前最优先。

如果把 exact packed kernel 写成

$$
G = L L^\top,
$$

则 opposite-spin 可以改写成

$$
V_{LR}^{\mathrm{os}}
=
\sum_{\mu}
\left(
\sum_P L_{P\mu} u^\alpha(P)
\right)
\left(
\sum_Q L_{Q\mu} u^\beta(Q)
\right).
$$

这会把 “按 packed pair 编号求和” 改写成 “按核因子编号求和”。

优点：

- 如果 exact packed kernel 存在明显低有效秩，可以减少 channel 数；
- 更容易组织成 dense BLAS。

缺点：

- full-rank exact 分解不一定降低渐近复杂度；
- 会把当前稀疏 channel 变得更 dense，常数不一定更好；
- 引入新的预处理与数据布局复杂度。

因此这条路线更像“下一层次的实验方向”，不是当前 first priority。

## 10. 推荐的实际推进顺序

如果目标是继续压 exact opposite-spin 的单步时间，建议按下面顺序做：

1. **先做 channel screening**
   - 这是最 exact、最稳的收益点。
   - 不改数学，只减少空通道与无效通道遍历。

2. **再做 batched / blocked matrix contraction**
   - 把 per-channel / per-structure 的许多小乘法尽量并成大乘法。
   - 这是把当前 matrix form 真正吃满 BLAS 的关键。

3. **最后再评估 kernel factorization**
   - 如果 exact packed kernel 的有效秩确实低，才值得继续推进。
   - 否则可能只是在增加实现复杂度。

## 11. 一句话总结

当前 exact opposite-spin 的核心公式已经不是 full determinant-pair 暴力求和，而是：

$$
\text{same-spin pair} \;\Longrightarrow\; \text{packed-pair coefficient vector}
\;\Longrightarrow\; \text{packed-kernel projection}
\;\Longrightarrow\; \text{channel-wise matrix contraction}.
$$

所以接下来真正该优化的，不是再去“减少重复 determinant pair”，而是：

$$
\boxed{
\text{减少 packed-pair channel 的无效遍历}
\;+\;
\text{把 channel 收缩做成更大颗粒度的 BLAS}
}
$$

这才是 exact opposite-spin 后续还能继续降单步时间的主方向。
