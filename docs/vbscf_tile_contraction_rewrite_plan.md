# VBSCF Tile-Contraction Rewrite Plan

本文档梳理 VBSCF exact matrix-form 的同自旋和反自旋 backward / HVP 链路，并给出统一改成 tile-contraction 的公式和实施方案。

核心目标不是简单 Eigen 化，也不是把 dense 矩阵拆成很多 scalar lookup。目标是：

1. 不再生成全局 `N_unique^2` dense weight matrix。
2. 不再生成全局 `N_det^2` determinant-pair weight table。
3. 不把矩阵元拆成单个 `(left,right)` 现场规约，因为那会丢掉 GEMM 复用。
4. 所有热路径工作区由 tile size、active-pair block size、selected-state local support 决定。
5. 同自旋和反自旋使用同一套 tile scheduler / support slicing / tile workspace 思路。

当前最重要的结论：

```text
旧 matrix-form backward:
  selected-state coefficients -> 全局 unique-spin weight matrices -> pair-cache backward

目标 tile-contraction backward:
  selected-state support tile + partner pair-cache tile -> local weight tile -> immediately consume pair-cache tile
```

也就是说，weight matrix 仍然是数学对象，但不应该作为全局 dense 数据结构存在。

## 1. 记号

令 alpha unique spin strings 为 \(a,a'\in A\)，数量为 \(N_\alpha\)。beta unique spin strings 为 \(b,b'\in B\)，数量为 \(N_\beta\)。

第 \(s\) 个 selected state 的系数矩阵为

\[
C_s(a,b).
\]

实际应使用 support-local 表达：

\[
A_s=\{a:C_s(a,b)\ne 0\},\qquad
B_s=\{b:C_s(a,b)\ne 0\},
\]

\[
C_s^{A_sB_s}=C_s[A_s,B_s].
\]

轨道方向或 outer-response 方向导致的 selected-state 响应记为

\[
\dot C_s(a,b),\qquad \dot E_s.
\]

同自旋 ordered pair cache 中的标量核记为

\[
S^\alpha_{aa'},\quad H^\alpha_{aa'},\quad
\Phi^\alpha_{aa'},\quad X^\alpha_{aa'}.
\]

其中 \(S\) 是同自旋 overlap determinant，\(H\) 是同自旋总 Hamiltonian scalar，\(\Phi\) 是 same-spin phi / partner-transfer 需要的 scalar，\(X\) 泛指一阶余子式、inverse-overlap projection 等 pair-local adjoint payload。

反自旋 packed active pair channel 记为

\[
U^\alpha_P(a,a'),\qquad U^\beta_Q(b,b'),
\]

其中 \(P,Q\) 是 active orbital unordered pair 的 packed index。

active two-electron integral kernel 记为

\[
G_{PQ}=g(P,Q),
\]

并满足 pair-of-pairs 对称 packed 存储。

tile 记号：

\[
I,J\subset A,\qquad K,L\subset B,
\]

其中 \(I,J,K,L\) 是 unique-spin-string tile，不是 orbital tile。

## 2. 当前代码结构和主要问题

### 2.1 Forward 已经有局部 tile 思想

[full_structure_builder.cpp](../src/vb/matrices/full_structure_builder.cpp) 的 `build_tiled_matrix_form_structure_matrices` 已经按 structure-pair support gather 局部 alpha/beta blocks：

```text
alpha_overlap_subblock
alpha_total_subblock
beta_overlap_subblock
beta_total_subblock
alpha_projection_block
beta_projection_block
```

这条 forward 链路的方向是对的：只在当前 structure pair 的 support 上做局部收缩。

同自旋 forward 核心可写为：

\[
S_{LR}
=
\langle C_L,\; S^\alpha C_R S^{\beta T}\rangle,
\]

\[
H^{ss}_{LR}
=
\langle C_L,\; H^\alpha C_R S^{\beta T}\rangle
+
\langle C_L,\; S^\alpha C_R H^{\beta T}\rangle.
\]

反自旋 forward 核心可写为：

\[
H^{ab}_{LR}
=
\sum_{P,Q}G_{PQ}
\langle U^\alpha_P,\; C_L U^\beta_Q C_R^T\rangle.
\]

当前 forward 局部化主要围绕 structure-pair support，后续 backward/HVP 应该沿用这种思想，而不是回到全局 weight matrix。

### 2.2 Same-spin backward 仍生成全局 weight matrices

[same_spin_matrix_backward.cpp](../src/vb/scf/same_spin_matrix_backward.cpp) 当前有这些全局对象：

```cpp
struct SameSpinExactWeightMatrices {
  Eigen::MatrixXd alpha_hamiltonian_weight_matrix;
  Eigen::MatrixXd alpha_overlap_weight_matrix;
  Eigen::MatrixXd alpha_partner_total_transfer_matrix;
  Eigen::MatrixXd alpha_singular_partner_transfer_matrix;
  Eigen::MatrixXd beta_hamiltonian_weight_matrix;
  Eigen::MatrixXd beta_overlap_weight_matrix;
  Eigen::MatrixXd beta_partner_total_transfer_matrix;
  Eigen::MatrixXd beta_singular_partner_transfer_matrix;
};
```

即使 `build_support_sparse_exact_same_spin_weight_matrices` 已经用 selected-state support 做局部计算，它最后仍 scatter 到全局：

\[
W^\alpha\in\mathbb R^{N_\alpha\times N_\alpha},
\qquad
W^\beta\in\mathbb R^{N_\beta\times N_\beta}.
\]

这解决了一部分计算量问题，但没有解决大 active / 大 unique-spin-string 下的内存问题。

同样，`SameSpinPairScalarMatrices` 也会生成全局 pair scalar matrices：

```cpp
struct SameSpinPairScalarMatrices {
  Eigen::MatrixXd overlap_determinant_matrix;
  Eigen::MatrixXd regular_total_hamiltonian_matrix;
  Eigen::MatrixXd singular_total_hamiltonian_matrix;
};
```

这是另一组 \(O(N_\sigma^2)\) dense storage。

### 2.3 Opposite-spin backward 也生成 dense image / dense weight batch

[opposite_spin_matrix_backward.cpp](../src/vb/scf/opposite_spin_matrix_backward.cpp) 当前 packed-2e adjoint 的核心形式是：

\[
g_{QP}
=
\sum_s w_s
\langle U^\alpha_P,\; C_s U^\beta_Q C_s^T\rangle.
\]

代码中先对每个 beta packed pair batch 生成：

\[
W^\alpha_Q
=
\sum_s w_s C_s U^\beta_Q C_s^T,
\]

其中

\[
W^\alpha_Q\in\mathbb R^{N_\alpha\times N_\alpha}.
\]

然后和 alpha sparse channel \(U^\alpha_P\) 收缩：

\[
g_{QP}=\langle U^\alpha_P,W^\alpha_Q\rangle.
\]

这比 determinant-pair 旧路径好，但内存为：

\[
O(B_Q N_\alpha^2)
\]

其中 \(B_Q\) 是 beta packed-pair batch size。

overlap adjoint 里也有类似问题：

```cpp
CachedPackedPairBlockProvider<Eigen::MatrixXd>
```

它缓存的是 `packed_pair -> N_unique x N_unique` dense image。

### 2.4 单矩阵元 streaming 不是正确方案

如果直接按

\[
[W^\alpha_Q]_{aa'}
=
\sum_s w_s
C_s(a,:)U^\beta_QC_s(a',:)^T
\]

每次只算一个 \((a,a')\)，内存可以降下来，但会造成严重重复：

```text
for pair-cache entry (a,a'):
  for Q:
    redo C_s row x U_Q x C_s row contraction
```

这属于寻址规约，不是高性能 tile contraction。正确方案是对一整块 \(I\times J\) 同时算：

\[
W^\alpha_Q[I,J]
=
\sum_s w_s C_s[I,B_s]U^\beta_Q[B_s,B_s]C_s[J,B_s]^T.
\]

然后立刻消费该 tile，不保存全局矩阵。

## 3. Same-Spin Tile-Contraction 公式

### 3.1 Accepted-point backward weights

同自旋 alpha 侧的 Hamiltonian weight、overlap weight、partner transfer 都有统一形式：

\[
W^\alpha_X
=
\sum_s \omega^X_s C_s K^\beta_X C_s^T,
\]

其中 \(X\) 表示不同 partner kernel：

\[
K^\beta_S=S^\beta,\qquad
K^\beta_H=H^\beta,\qquad
K^\beta_{\mathrm{sing}}=H^\beta_{\mathrm{singular}},
\]

\[
\omega^H_s=w_s,\qquad
\omega^S_s=-w_sE_s,\qquad
\omega^T_s=w_s.
\]

beta 侧对应：

\[
W^\beta_X
=
\sum_s \omega^X_s C_s^T K^\alpha_X C_s.
\]

当前代码的问题是直接生成 \(W^\alpha_X\) 和 \(W^\beta_X\) 的全局 dense matrix。

tile 公式应改成：

\[
W^\alpha_X[I,J]
=
\sum_s
\omega^X_s
\sum_{K,L\subset B_s}
C_s[I,K]K^\beta_X[K,L]C_s[J,L]^T.
\]

beta 侧：

\[
W^\beta_X[K,L]
=
\sum_s
\omega^X_s
\sum_{I,J\subset A_s}
C_s[I,K]^T K^\alpha_X[I,J] C_s[J,L].
\]

其中 \(I,J,K,L\) 都是小 tile。

关键点：

```text
W tile is temporary.
It is consumed immediately by pair-cache backward for the same (I,J) tile.
No global W matrix exists.
```

### 3.2 Same-spin backward tile consumption

当前 `accumulate_spin_matrix_backward` 接收全局：

```cpp
hamiltonian_weight_matrix
overlap_weight_matrix
partner_total_transfer_matrix
```

目标改成 tile API：

```cpp
accumulate_spin_matrix_backward_tile(
    unique_determinants,
    ordered_pair_cache,
    tile_left_begin,
    tile_left_end,
    tile_right_begin,
    tile_right_end,
    tile_weights,
    ... gradients ...);
```

对于 tile 内每个 ordered pair \((i,j)\)，使用：

\[
w^H_{ij}=W^H[I,J]_{ij},
\]

\[
w^S_{ij}=W^S[I,J]_{ij},
\]

\[
w^T_{ij}=W^T[I,J]_{ij}.
\]

随后原来的 pair-local backprop 公式不变：

regular pair:

\[
\bar h_{pq} \mathrel{+}=
w^H_{ij}\frac{\partial H_{ij}}{\partial h_{pq}},
\]

\[
\bar g_{pqrs} \mathrel{+}=
w^H_{ij}\frac{\partial H_{ij}}{\partial g_{pqrs}},
\]

\[
\bar S_{pq} \mathrel{+}=
\left(
w^S_{ij}+w^T_{ij}+w^H_{ij}\Phi_{ij}
\right)
\frac{\partial S_{ij}}{\partial S_{pq}}
+w^H_{ij}
\frac{\partial \Phi_{ij}}{\partial S_{pq}}.
\]

singular pair:

\[
\bar S_{pq}
\mathrel{+}=
w^H_{ij}
\frac{\partial H^{\mathrm{sing}}_{ij}}{\partial S_{pq}}
+
(w^S_{ij}+w^T_{ij})
\frac{\partial S_{ij}}{\partial S_{pq}}.
\]

这些 pair-local formulas 已经存在，重构重点是 weight source 从全局 matrix 改成 tile matrix。

### 3.3 Directional selected-state response

`build_directional_same_spin_matrix_backward_contribution` 对应 selected-state coefficient / energy response。

accepted weight：

\[
W^\alpha_X
=
\sum_s \omega_s C_s K^\beta_X C_s^T.
\]

directional weight：

\[
\dot W^\alpha_X
=
\sum_s
\dot\omega_s C_sK^\beta_XC_s^T
+
\omega_s\dot C_sK^\beta_XC_s^T
+
\omega_sC_sK^\beta_X\dot C_s^T.
\]

其中 overlap weight 还含有 \(\dot E_s\)：

\[
\omega^S_s=-w_sE_s,
\qquad
\dot\omega^S_s=-w_s\dot E_s.
\]

tile 版本：

\[
\dot W^\alpha_X[I,J]
=
\sum_s
\sum_{K,L}
\dot\omega_s C_s[I,K]K^\beta_X[K,L]C_s[J,L]^T
\]

\[
\quad+
\sum_s
\sum_{K,L}
\omega_s \dot C_s[I,K]K^\beta_X[K,L]C_s[J,L]^T
\]

\[
\quad+
\sum_s
\sum_{K,L}
\omega_s C_s[I,K]K^\beta_X[K,L]\dot C_s[J,L]^T.
\]

beta 侧同理：

\[
\dot W^\beta_X[K,L]
=
\sum_s
\sum_{I,J}
\dot\omega_s C_s[I,K]^TK^\alpha_X[I,J]C_s[J,L]
\]

\[
\quad+
\sum_s
\sum_{I,J}
\omega_s \dot C_s[I,K]^TK^\alpha_X[I,J]C_s[J,L]
\]

\[
\quad+
\sum_s
\sum_{I,J}
\omega_s C_s[I,K]^TK^\alpha_X[I,J]\dot C_s[J,L].
\]

directional support 的数学作用域是 union：

\[
A_s^\cup=A_s\cup \dot A_s,
\qquad
B_s^\cup=B_s\cup \dot B_s.
\]

实现上不需要显式构造 union dense block。更好的 tile 形式是直接计算两个
mixed-support 项：

\[
\dot C_s[A_s^\dot,B_s^\dot]K[B_s^\dot,B_s]C_s[A_s,B_s]^T,
\]

\[
C_s[A_s,B_s]K[B_s,B_s^\dot]\dot C_s[A_s^\dot,B_s^\dot]^T.
\]

这样既覆盖 union support，又避免把 \(C_s\) 和 \(\dot C_s\) scatter/gather
成新的临时 union matrix。

### 3.4 Local orbital response

`build_local_same_spin_matrix_backward_contribution` 对应 fixed \(C_s\) 下 pair kernels 的方向响应。

若 partner kernel 本身有方向：

\[
\dot K^\beta_X,
\]

则

\[
\dot W^\alpha_X[I,J]
=
\sum_s\omega_s
\sum_{K,L}
C_s[I,K]\dot K^\beta_X[K,L]C_s[J,L]^T.
\]

完整 local HVP 对 same-spin backward tile 的输入应是：

```text
accepted tile weights:
  W_H[I,J], W_S[I,J], W_T[I,J]

local-response tile weights:
  dW_H[I,J], dW_S[I,J], dW_T[I,J]

pair-local directional payload:
  d cofactor, d same-spin phi, d overlap inverse, d same-spin H/S gradients
```

当前代码已经有 pair-local directional payload 构造，问题仍然是权重通过全局 matrix 传递。

## 4. Opposite-Spin Tile-Contraction 公式

### 4.1 Packed 2e adjoint

反自旋 exact energy contribution 可写为：

\[
E^{ab}
=
\sum_{P,Q}G_{PQ}
\sum_s w_s
\langle U^\alpha_P,\; C_s U^\beta_Q C_s^T\rangle.
\]

因此 packed 2e gradient：

\[
\bar G_{PQ}
=
\sum_s w_s
\langle U^\alpha_P,\; C_s U^\beta_Q C_s^T\rangle.
\]

等价形式：

\[
\bar G_{PQ}
=
\sum_s w_s
\langle C_s^T U^\alpha_P C_s,\; U^\beta_Q\rangle.
\]

当前 dense-weight 路径会形成：

\[
W^\alpha_Q=C_sU^\beta_QC_s^T
\in\mathbb R^{N_\alpha\times N_\alpha}.
\]

目标 tile 公式：

\[
\bar G_{PQ}^{I,J,K,L}
=
\sum_s w_s
\left\langle
U^\alpha_P[I,J],
C_s[I,K]U^\beta_Q[K,L]C_s[J,L]^T
\right\rangle.
\]

累加所有 tile：

\[
\bar G_{PQ}
=
\sum_{I,J,K,L}\bar G_{PQ}^{I,J,K,L}.
\]

### 4.2 不能按 scalar entry 规约

错误但省内存的写法是：

\[
\bar G_{PQ}
\mathrel{+}=
U^\alpha_P(a,a')
\sum_s w_s
C_s(a,:)U^\beta_QC_s(a',:)^T.
\]

这会导致每个 \((a,a')\) 都重新做 \(C U C^T\) 的行级收缩。

正确 block 写法是：

1. 取 alpha tile \(I,J\)。
2. 取 beta tile \(K,L\)。
3. 取 packed-pair channel block \(P\in\mathcal P_t\)，\(Q\in\mathcal Q_t\)。
4. 构造或 gather：

\[
A_{\mathcal P}
=
\operatorname{vec}(U^\alpha_P[I,J])
\in\mathbb R^{|I||J|\times |\mathcal P_t|},
\]

\[
B_{\mathcal Q}
=
\operatorname{vec}(U^\beta_Q[K,L])
\in\mathbb R^{|K||L|\times |\mathcal Q_t|}.
\]

对每个 selected state，计算 beta-side projected block：

\[
M_{s,\mathcal Q}[I,J,Q]
=
\sum_{K,L}
C_s[I,K]U^\beta_Q[K,L]C_s[J,L]^T.
\]

把 \(M_{s,\mathcal Q}\) flatten 成：

\[
M_{\mathcal Q}
\in
\mathbb R^{|I||J|\times|\mathcal Q_t|}.
\]

然后一次 block contraction：

\[
\bar G_{\mathcal P,\mathcal Q}
\mathrel{+}=
A_{\mathcal P}^T M_{\mathcal Q}.
\]

这一步是反自旋 packed-gradient 的核心。它既避免 \(N_\alpha^2\) dense image，又避免 scalar entry 重复规约。

### 4.3 Directional and local response

selected-state response：

\[
\delta\bar G_{PQ}^{C}
=
\sum_s w_s
\langle U^\alpha_P,\;
\dot C_s U^\beta_Q C_s^T
+
C_sU^\beta_Q\dot C_s^T
\rangle.
\]

local orbital response：

\[
\delta\bar G_{PQ}^{U}
=
\sum_s w_s
\langle \dot U^\alpha_P,\; C_sU^\beta_QC_s^T\rangle
+
\sum_s w_s
\langle U^\alpha_P,\; C_s\dot U^\beta_QC_s^T\rangle.
\]

active integral direction also enters through projected channel values:

\[
\dot{(G U)}_P
=
\dot G\,U_P+G\,\dot U_P.
\]

tile 形式完全一致，只是 channel block 有 accepted / directional 两套：

\[
\delta\bar G_{\mathcal P,\mathcal Q}
=
\dot A_{\mathcal P}^T M_{\mathcal Q}
+
A_{\mathcal P}^T\dot M_{\mathcal Q}.
\]

其中

\[
\dot M_{\mathcal Q}
=
\operatorname{vec}
\left(
\dot C U^\beta_{\mathcal Q}C^T
+
C\dot U^\beta_{\mathcal Q}C^T
+
CU^\beta_{\mathcal Q}\dot C^T
\right).
\]

### 4.4 Opposite-spin overlap adjoint

overlap adjoint 不是 packed 2e gradient，但也有同样的 dense-image 问题。

对 alpha overlap 的反向传播需要：

\[
\eta^\alpha_{aa'}
=
\sum_P X^\alpha_P(a,a')W^\alpha_P(a,a'),
\]

其中 \(X^\alpha_P\) 是 inverse-overlap projection 或 first-order projection 的 pair-local coefficient，\(W^\alpha_P\) 来自 beta side：

\[
W^\alpha_P
=
\sum_s w_s C_sV^\beta_P C_s^T.
\]

\(V^\beta_P\) 是已经被 active two-electron kernel 投影后的 beta channel：

\[
V^\beta_P(b,b')=(G U^\beta_{bb'})(P).
\]

当前代码会缓存：

\[
W^\alpha_P\in\mathbb R^{N_\alpha\times N_\alpha}.
\]

目标 tile 公式：

\[
W^\alpha_P[I,J]
=
\sum_s w_s
\sum_{K,L}
C_s[I,K]V^\beta_P[K,L]C_s[J,L]^T.
\]

然后对 alpha pair-cache tile 直接消费：

\[
\eta^\alpha[I,J]
=
\sum_{P\in\mathcal P_t}
X^\alpha_P[I,J]\odot W^\alpha_P[I,J].
\]

对 inverse-overlap gradient 所需的小电子矩阵，仍然只需要 occupied orbital pair 对应的 packed \(P\)，可以从当前 tile 的 \(W^\alpha_P[I,J]\) 取值，而不是从全局 dense provider 取值。

## 5. 内存标度

定义：

```text
N_a, N_b      unique alpha/beta spin strings
N_s           selected states
a_s, b_s      selected state s support sizes
T_a, T_b      unique-spin tile sizes
B_P, B_Q      active packed-pair channel block sizes
M             number of active packed pairs = n_act(n_act+1)/2
```

### 5.1 当前 same-spin backward

Same-spin exact accepted path 至少包含：

\[
O(4N_\alpha^2+4N_\beta^2)
\]

的 global weight matrices。

如果构造 scalar partner matrices，还包含：

\[
O(3N_\alpha^2+3N_\beta^2)
\]

的 global scalar matrices。

local/directional path 还会多出 delta weight matrices。

因此 memory leading term 是：

\[
O(N_\alpha^2+N_\beta^2).
\]

### 5.2 当前 opposite-spin backward

packed-2e adjoint beta batch：

\[
O(B_QN_\alpha^2)
\]

overlap dense image provider：

\[
O(C_{\mathrm{cache}}B_PN_\alpha^2)
\quad \text{or}\quad
O(C_{\mathrm{cache}}B_PN_\beta^2).
\]

其中 \(C_{\mathrm{cache}}\) 是 provider cache entry 数。

### 5.3 目标 tile memory

same-spin tile workspace：

\[
O(k_WT_\alpha^2)
+
O(k_KT_\beta^2)
+
O(T_\alpha T_\beta),
\]

其中 \(k_W\) 是同时保留的 weight channels 数，\(k_K\) 是 partner scalar channels 数。

opposite-spin packed-gradient tile workspace：

\[
O(T_\alpha^2B_P)
+
O(T_\beta^2B_Q)
+
O(T_\alpha^2B_Q)
+
O(T_\alpha T_\beta).
\]

关键是没有：

\[
O(N_\alpha^2),\quad O(N_\beta^2),\quad O(N_{\det}^2).
\]

持久数据中仍会有 molecule/input、unique determinant lists、selected-state local support 等必要模型数据。新的 contraction workspace 不应随 \(N_\alpha^2\)、\(N_\beta^2\)、\(N_{\det}^2\) 增长。

## 6. 计算标度和复用

### 6.1 Same-spin

当前 dense weight build：

\[
O\left(
\sum_s N_\alpha N_\beta(N_\alpha+N_\beta)
\right).
\]

当前 support-sparse build：

\[
O\left(
\sum_s a_sb_s(a_s+b_s)
\right)
\]

但仍 scatter 到 global matrix，并且之后 pair-cache backward 以 global weight matrix 为输入。

目标 tile build：

\[
O\left(
\sum_{\mathrm{queried}\ I,J,K,L,s}
|I||K||L|+|I||J||L|
\right),
\]

实际以 GEMM 形式执行：

\[
P=C_s[I,K]K^\beta[K,L],
\]

\[
W[I,J]\mathrel{+}=PC_s[J,L]^T.
\]

这保留 BLAS 复用，同时不生成全局 \(W\)。

### 6.2 Opposite-spin

不推荐 scalar-entry path：

\[
O\left(
\#(a,a',P,Q)\times \text{row contraction cost}
\right).
\]

推荐 block path：

\[
M_{\mathcal Q}[I,J]
=
\sum_{K,L,s}C_s[I,K]U^\beta_{\mathcal Q}[K,L]C_s[J,L]^T,
\]

\[
\bar G_{\mathcal P,\mathcal Q}
\mathrel{+}=
A_{\mathcal P}[I,J]^TM_{\mathcal Q}[I,J].
\]

这里 \(A_{\mathcal P}^TM_{\mathcal Q}\) 是一个小 GEMM / sparse-dense contraction，可复用一个 \(I,J\) tile 上的所有 alpha channel 和 beta channel。

## 7. 数据结构设计

### 7.1 Coefficient storage

当前 `SelectedStateDeterminantCoefficients` 同时有：

```cpp
Eigen::MatrixXd coefficient_matrix;
std::vector<int> alpha_support;
std::vector<int> beta_support;
Eigen::MatrixXd local_coefficient_matrix;
```

最终应改成 local support 为主：

```cpp
struct SelectedStateCoefficientBlock {
  double weight;
  double energy;
  std::vector<int> alpha_support;
  std::vector<int> beta_support;
  Eigen::MatrixXd local_coefficients;
};
```

directional state 使用同样结构。需要 union support 时，在 tile 内临时 gather，而不是保存全局 dense \(\dot C\)。

`coefficient_matrix` 是潜在的大内存项。若保留，也只能作为 dev/reference 路径，不应在 production exact path 中依赖。

### 7.2 Same-spin tile weights

替代 `SameSpinExactWeightMatrices`：

```cpp
struct SameSpinTileWeights {
  int left_begin;
  int right_begin;
  Eigen::MatrixXd hamiltonian;
  Eigen::MatrixXd overlap;
  Eigen::MatrixXd partner_total;
  Eigen::MatrixXd singular_partner;
};
```

local-response 扩展：

```cpp
struct SameSpinLocalTileWeights {
  SameSpinTileWeights accepted;
  Eigen::MatrixXd delta_hamiltonian;
  Eigen::MatrixXd delta_overlap;
  Eigen::MatrixXd delta_partner_total;
};
```

这些对象只在当前 tile 内存在。

### 7.3 Pair scalar provider

替代 `SameSpinPairScalarMatrices`：

```cpp
class SameSpinScalarTileProvider {
public:
  void gather_overlap_tile(indices_left, indices_right, Eigen::MatrixXd* out);
  void gather_regular_total_tile(indices_left, indices_right, Eigen::MatrixXd* out);
  void gather_singular_total_tile(indices_left, indices_right, Eigen::MatrixXd* out);
};
```

provider 从 ordered pair cache 读取 scalar，不生成全局 scalar matrix。

### 7.4 Opposite-spin channel blocks

packed channel block 应以 flatten tile 形式提供：

```cpp
struct PackedChannelTile {
  int packed_begin;
  int packed_end;
  Eigen::SparseMatrix<double, Eigen::ColMajor, int> flattened_sparse;
};
```

其中 rows 是 tile pair entries：

\[
r=(i-i_0)+(j-j_0)|I|.
\]

columns 是 packed channel index block。

对于 beta projected channel，也可以用 dense/sparse block：

\[
B_{\mathcal Q}\in\mathbb R^{|K||L|\times B_Q}.
\]

如果 channel density 很低，用 sparse；如果 tile 小且 channel block 小，转 dense 便于 GEMM。

## 8. 实施顺序

### Step 1: 建立只读 tile provider

新增不改变数学结果的 provider：

```text
SameSpinScalarTileProvider
SameSpinProjectionTileProvider
OppositeSpinChannelTileProvider
SelectedStateTileSlicer
```

要求：

```text
不分配 N_unique^2
不分配 N_det^2
每个 provider 只填 caller-owned tile workspace
```

### Step 2: 改 same-spin accepted backward

替换：

```cpp
build_exact_same_spin_weight_matrices(...)
accumulate_spin_matrix_backward(... global matrices ...)
```

为：

```cpp
for alpha tile I,J:
  build_same_spin_weight_tile(alpha, I, J)
  accumulate_spin_matrix_backward_tile(alpha, I, J, tile_weights)
```

close-shell 情况：

```text
只跑 alpha tile once
最后乘 2
```

不要构造 beta global matrices。

### Step 3: 改 same-spin directional/local HVP

把以下对象替换成 tile-local：

```text
directional_weight_matrices
local_weight_matrices
alpha_directional_scalars
beta_directional_scalars
```

local pair directional payload 仍按 pair tile 构造和消费。

### Step 4: 改 opposite-spin packed 2e gradient

替换 beta batch dense image：

```text
for beta Q batch:
  W_alpha_Q = C U_beta_Q C^T  // delete
```

为：

```text
for alpha tile I,J:
  build A_P[I,J] for P block
  for beta support tile K,L:
    build B_Q[K,L] for Q block
    M_Q[I,J] += C[I,K] B_Q[K,L] C[J,L]^T
  G[P,Q] += A_P[I,J]^T M_Q[I,J]
```

这个阶段最需要 benchmark，因为它决定反自旋 HVP kernel 是否真的变快。

### Step 5: 改 opposite-spin overlap adjoint

替换：

```cpp
CachedPackedPairBlockProvider<Eigen::MatrixXd>
```

为：

```text
for spin pair tile I,J:
  for packed channel block P:
    build W_P[I,J]
    consume inverse/first-order projection payloads in ordered pair cache tile
```

singular nullity path 需要的小电子矩阵从 tile-local \(W_P[I,J]\) 取值。

### Step 6: 删除 production dense coefficient matrix dependency

把 selected-state dense `coefficient_matrix` 从 production exact path 中移除。若需要参考实现，应放在 dev/test target，不作为 runtime fallback。

## 9. 验证策略

禁止用 production fallback 来掩盖问题。可以保留 dev-only reference checker。

最小验证集：

```text
F2: closed-shell small reference
241_VBSCF: closed-shell / current fast case
MnF2: open-shell transition metal sparse-orbital case
FeCl2 or TiCl: open-shell sanity
```

数值验证：

```text
same-spin accepted backward tile vs old reference
same-spin directional backward tile vs finite/reference HVP
same-spin local-response tile vs old reference
opposite-spin packed-gradient tile vs old reference
opposite-spin overlap-adjoint tile vs old reference
full exact_ctx HVP max_abs_diff / max_rel_diff
SCF convergence trajectory sanity
```

性能指标：

```text
peak RSS
HVP avg_apply_s
avg_active_2e_s
avg_h1e_fused_s
same_spin_s / opposite_spin_s from structure-stage logging
CG iterations per accepted step
wall-time to convergence
```

## 10. 关键风险

### 10.1 Tile 太小会退化成 scalar规约

如果 tile size 过小，公式虽然正确，但会变成大量小矩阵乘法和随机寻址。需要保证常见 support 下能形成足够大的 GEMM。

### 10.2 Tile 太大又回到内存问题

tile size 应由内存预算控制：

\[
T_\alpha^2B_P+T_\beta^2B_Q+T_\alpha^2B_Q
\le
M_{\mathrm{workspace}}.
\]

### 10.3 Sparse channel block 的布局决定速度

反自旋 channel block 如果每次从 ordered pair cache 重新扫描，会慢。需要在 tile 内构造 reusable flattened block，并按 packed-pair block 重用。

### 10.4 Directional support union 可能变大

HVP 中 \(\dot C\) support 可能比 \(C\) support 大。tile builder 必须基于 union support，但仍不能回退到 global dense matrix。

### 10.5 当前 same-spin pair cache 本身仍是 \(N_\sigma^2\)

本计划先解决 backward/HVP 的 dense weight / dense image 问题。ordered same-spin pair cache 如果未来也成为内存瓶颈，需要进一步改为 tile cache provider，而不是全量 ordered pair cache。

## 11. 最终形态

最终 exact VBSCF matrix-form backward/HVP 应该满足：

```text
same-spin:
  selected-state support tile
  partner scalar tile
  local weight tile
  immediate pair-cache backprop

opposite-spin:
  selected-state support tile
  alpha/beta packed-channel blocks
  block contraction G[P,Q] += A^T M
  immediate overlap-adjoint tile consumption

persistent memory:
  no N_det^2 dense tables
  no N_unique^2 dense weight matrices
  no packed_pair -> N_unique^2 dense image providers

hot kernels:
  block GEMM / sparse-block contraction
  not scalar-entry on-demand contraction
```

这才是同自旋和反自旋统一 tile-contraction 的正确方向。

## 12. 当前进度

### 2026-04-25: same-spin accepted backward tile path

已完成第一块生产代码落地：

```text
support-sparse same-spin accepted backward:
  selected-state local C block
  partner same-spin scalar subblock from ordered pair cache
  tile-local W_H / W_S / W_partner
  immediate same-spin pair-cache backward consumption
```

具体变化：

```text
build_same_spin_matrix_backward_contribution(...)
  if support-sparse selected-state contraction is selected and tile mode is enabled:
    use tile-contraction path
  else:
    keep old dense/global-weight path
```

这个路径不再为 accepted backward 构造全局
`SameSpinExactWeightMatrices`，也不再先构造全局
`SameSpinPairScalarMatrices`。由于 241 这类中小 unique-spin 空间上该路径会
增加 tile/gather 常数开销，当前默认只在估算的 accepted same-spin dense
weight storage 超过 256 MiB 时自动启用；也可以用
`XMVB_CPP_SAME_SPIN_ACCEPTED_TILE_BACKWARD=on|off` 强制打开或关闭。

当前仍保留 dense path 作为常规运行路径；directional/local HVP 还没有切换到
tile 版本。

验证：

```text
cmake --build build --target run_cpp_vbscf -j8
cmake --build build --target check_exact_ctx_hvp -j8
F2 lbfgspp support_sparse=off/on: converged
F2 check_exact_ctx_hvp support_sparse=off/on:
  matrix_form_sum_sso/hho/ggo max_abs_diff ~= 1e-16
  matrix_vs_pairwise_local_same_spin max_abs_diff ~= 1e-16
241 tnhvp 32 cores with forced tile:
  converged in 8 iterations
  final energy matches previous log to ~1e-12
  wall time is slower on this small unique-spin case, so tile is not default
```

下一步应继续做：

```text
1. same-spin directional/local HVP tile weights
2. opposite-spin packed-gradient tile contraction
3. opposite-spin overlap-adjoint tile consumption
```

### 2026-04-25: opposite-spin packed-gradient tile path

已完成 accepted opposite-spin packed two-electron gradient 的 tile path：

```text
for beta packed-pair batch Q:
  build sparse U_beta(Q)
  for alpha unique tile I,J:
    build tile-local W_alpha_Q[I,J] = sum_s w_s C_s[I,:] U_beta(Q) C_s[J,:]^T
    contract immediately with sparse U_alpha(P)[I,J]
```

这个路径避免为每个 beta packed pair 保留

```text
W_alpha_Q in R^{N_alpha x N_alpha}
```

的 dense image。历史实现中曾保留 dense batch fallback，并用
`XMVB_CPP_OPPOSITE_SPIN_PACKED_GRADIENT_TILE=on|off` 强制控制。2026-04-25
清理后 production 已统一为 tile path；unique tile 大小仍由
`XMVB_CPP_OPPOSITE_SPIN_BACKWARD_UNIQUE_TILE_SIZE` 控制。

验证：

```text
F2 check_exact_ctx_hvp with tile on/off:
  matrix_form_sum_sso/hho/ggo max_abs_diff ~= 1e-16
  local_opposite_spin_ggo_max_abs_diff unchanged
241 tnhvp 32 cores with forced opposite tile:
  converged in 8 iterations
  final energy matches default to ~1e-12
  wall time is similar to default within node/runtime noise
```

结论仍然是：tile path 主要是内存和大体系可扩展性路径，不是 241 这种
small unique-spin case 的确定提速路径。

### 2026-04-25: opposite-spin overlap-adjoint tile path

已完成 accepted opposite-spin overlap adjoint 的 tile consumption：

```text
for alpha unique tile I,J:
  build W_alpha_P[I,J] = sum_s w_s C_s[I,:] V_beta(P) C_s[J,:]^T
  consume alpha inverse/first-order overlap payloads inside tile

for beta unique tile K,L:
  build W_beta_P[K,L] = sum_s w_s C_s[:,K]^T V_alpha(P) C_s[:,L]
  consume beta inverse/first-order overlap payloads inside tile
```

其中 \(V^\sigma(P)\) 是 partner spin side 上已经过 active two-electron
kernel 投影的 first-order cofactor channel。regular pair 的 determinant
overlap weight 必须使用本侧 inverse projection 的未投影 sparse
coefficient：

\[
\eta_{aa'}
=
\sum_P x^\alpha_P(a,a')\,W^\alpha_P(a,a'),
\]

而不是再使用本侧 \(Gx^\alpha\)。这里的 \(G\) 已经在 partner-side
\(W^\alpha_P\) 构造中进入。这个点已经用 F2 local opposite-spin
overlap HVP 诊断覆盖。

历史阶段的控制变量：

```text
XMVB_CPP_OPPOSITE_SPIN_OVERLAP_TILE=on|off
XMVB_CPP_OPPOSITE_SPIN_OVERLAP_TILE_MIN_DENSE_BYTES
XMVB_CPP_OPPOSITE_SPIN_BACKWARD_UNIQUE_TILE_SIZE
```

2026-04-25 清理后前两个开关已退出 production；overlap adjoint 统一走
tile path，只保留 `XMVB_CPP_OPPOSITE_SPIN_BACKWARD_UNIQUE_TILE_SIZE` 作为
tile size 调优参数。

验证：

```text
cmake --build build --target run_cpp_vbscf check_exact_ctx_hvp -j8
F2 check_exact_ctx_hvp with XMVB_CPP_OPPOSITE_SPIN_OVERLAP_TILE=on:
  opposite_spin_fixed_sso_max_abs_diff = 0
  local_opposite_spin_sso_max_abs_diff = 2.8e-17
  matrix_form_sum_sso_max_abs_diff = 3.3e-16
  matrix_form_sum_hho_max_abs_diff = 2.4e-16
  matrix_form_sum_ggo_max_abs_diff = 4.2e-16
241 tnhvp 32 cores forced overlap-tile:
  Slurm job 1943678 converged in 8 iterations
  final energy = -230.720590230795
  SCF iteration wall time = 15.42 s
  slower than dense/default on this small unique-space case
```

当时剩余 tile-contraction 工作收敛到清理项；这些项已在后续
production dense fallback cleanup 中完成：

```text
1. remove production dependence on global selected-state coefficient_matrix
   for support-sparse paths where the local support representation is sufficient
2. keep default threshold strategy conservative; 241-like small unique spaces
   should not be forced onto tile/gather paths
```

### 2026-04-25: opposite-spin local-response packed-gradient tile path

已完成 local orbital response 中 opposite-spin packed two-electron gradient
的 tile path。公式是 accepted packed-gradient 的方向导数：

\[
\delta\bar G_{PQ}
=
\langle \delta U^\alpha_P,\; C U^\beta_Q C^T\rangle
+
\langle U^\alpha_P,\; C \delta U^\beta_Q C^T\rangle.
\]

tile 形式：

\[
\delta\bar G_{PQ}^{I,J}
=
\langle \delta U^\alpha_P[I,J],\; W^\alpha_Q[I,J]\rangle
+
\langle U^\alpha_P[I,J],\; \delta W^\alpha_Q[I,J]\rangle,
\]

其中

\[
W^\alpha_Q[I,J]
=
\sum_s w_s C_s[I,:]U^\beta_Q C_s[J,:]^T,
\]

\[
\delta W^\alpha_Q[I,J]
=
\sum_s w_s C_s[I,:]\delta U^\beta_Q C_s[J,:]^T.
\]

实现上复用 accepted tile helper，只是同时构造 accepted 和 directional 两套
beta channel，再和 accepted / directional alpha sparse channel 做两项收缩。
历史阶段这个路径由 `XMVB_CPP_OPPOSITE_SPIN_PACKED_GRADIENT_TILE=on|off`
控制。2026-04-25 清理后 production 已固定使用 tile path，不再保留
packed-gradient dense fallback。

验证：

```text
F2 check_exact_ctx_hvp with XMVB_CPP_OPPOSITE_SPIN_PACKED_GRADIENT_TILE=on:
  local_opposite_spin_ggo_max_abs_diff = 5.6e-12
  matrix_form_sum_ggo_max_abs_diff = 4.2e-16
F2 check_exact_ctx_hvp with packed-gradient tile + overlap tile both on:
  local_opposite_spin_sso_max_abs_diff = 2.8e-17
  local_opposite_spin_ggo_max_abs_diff = 5.6e-12
  matrix_form_sum_sso/hho/ggo max_abs_diff ~= 1e-16
241 tnhvp 32 cores forced packed-gradient tile:
  Slurm job 1943679 converged in 8 iterations
  final energy = -230.720590230794
  SCF iteration wall time = 15.02 s
  small unique-space case is still slower than default dense path
```

### 2026-04-25: same-spin local-response tile weights

已完成 support-sparse local HVP 中 same-spin local-response weight 的 tile
path。accepted weight 和 local directional weight 同时在 target unique-spin
tile 上构造，并立即送入 pair-local directional backward：

\[
W_H[I,J],\quad W_S[I,J],\quad W_T[I,J],
\]

\[
\delta W_H[I,J],\quad \delta W_S[I,J],\quad \delta W_T[I,J].
\]

其中 accepted overlap/total partner kernels 直接从 same-spin pair cache
按 support tile gather；directional partner kernels 仍使用已验证的
`build_directional_pair_scalar_matrices(...)`，但不再经过全局
`SameSpinLocalResponseWeightMatrices`。directional total 使用

\[
\delta H^{\mathrm{same}} =
\delta H^{\mathrm{regular}} + \delta H^{\mathrm{singular}}
\]

现场逐元素读取，避免额外构造 dense sum matrix。

实现约束：

```text
tile workspaces are reused:
  SameSpinLocalTileWeights tile_weights
  inverse_overlap_gradient
  delta_inverse_overlap_gradient

no local-response global W / dW matrices in tile path:
  alpha_delta_hamiltonian_weight_matrix
  alpha_delta_overlap_weight_matrix
  alpha_delta_partner_total_transfer_matrix
  beta_delta_* equivalents
```

控制变量：

```text
XMVB_CPP_SAME_SPIN_LOCAL_TILE_BACKWARD=on|off
```

默认继承 accepted same-spin tile 的阈值策略，避免 241 这类小 unique-space
体系默认走 tile/gather 常数较高的路径。

验证：

```text
cmake --build build --target run_cpp_vbscf check_exact_ctx_hvp -j8
F2 check_exact_ctx_hvp with XMVB_CPP_SAME_SPIN_LOCAL_TILE_BACKWARD=on:
  matrix_vs_pairwise_local_same_spin_sso_max_abs_diff = 1.7e-16
  matrix_vs_pairwise_local_same_spin_hho_max_abs_diff = 1.4e-17
  matrix_vs_pairwise_local_same_spin_ggo_max_abs_diff = 0
  matrix_form_sum_sso/hho/ggo max_abs_diff ~= 1e-16
241 tnhvp 32 cores forced same-spin local tile:
  Slurm job 1943687 converged in 8 iterations
  final energy = -230.720590230795
  SCF iteration wall time = 14.69 s
```

### 2026-04-25: same-spin directional selected-state tile weights

已完成 support-sparse selected-state directional HVP 的 same-spin tile path。
旧路径先构造全局
`build_support_sparse_directional_exact_same_spin_weight_matrices(...)`，
再把全局 directional weight matrix 送入 pair-cache backward。新路径改为：

```text
for each unique-spin tile:
  build dW_H / dW_S / dW_partner on this tile
  immediately consume the tile in accumulate_spin_matrix_backward_tile(...)
  discard the tile workspace
```

alpha 侧公式：

\[
\dot W^\alpha_H[I,J]
=
\sum_s w_s
\left(
\dot C_s[I,:]S^\beta C_s[J,:]^T
+
C_s[I,:]S^\beta \dot C_s[J,:]^T
\right),
\]

\[
\dot W^\alpha_S[I,J]
=
\sum_s
\left[
-w_s\dot E_s C_s[I,:]S^\beta C_s[J,:]^T
-w_sE_s
\left(
\dot C_s[I,:]S^\beta C_s[J,:]^T
+
C_s[I,:]S^\beta \dot C_s[J,:]^T
\right)
\right],
\]

\[
\dot W^\alpha_T[I,J]
=
\sum_s w_s
\left(
\dot C_s[I,:]H^\beta C_s[J,:]^T
+
C_s[I,:]H^\beta \dot C_s[J,:]^T
\right).
\]

beta 侧完全对应，把 \(C K C^T\) 换成 \(C^T K C\)。闭壳层
same-spin 仍只计算 alpha 侧，然后将最终同自旋梯度乘 2，避免重复计算
alpha-alpha / beta-beta。

实现细节：

```text
control env:
  XMVB_CPP_SAME_SPIN_DIRECTIONAL_TILE_BACKWARD=on|off

new tile builder:
  accumulate_alpha_directional_tile_weights(...)
  accumulate_beta_directional_tile_weights(...)

new direct contribution path:
  build_support_sparse_directional_same_spin_backward_contribution_by_tiles(...)
```

`-w_s \dot E_s C S C^T` 使用单核 support-local tile 累加器，只生成 overlap
base image。两个 \(\dot C\) product-rule 项使用 existing mixed-support helper，
直接处理 \(A_s^\dot/B_s^\dot\) 和 \(A_s/B_s\) 的交叉 support，不再构造
union support matrix。

验证：

```text
cmake --build build --target run_cpp_vbscf check_exact_ctx_hvp -j8
F2 with XMVB_CPP_SAME_SPIN_DIRECTIONAL_TILE_BACKWARD=on:
  matrix_form_sum_sso_max_abs_diff = 3.3e-16
  matrix_form_sum_hho_max_abs_diff = 2.4e-16
  matrix_form_sum_ggo_max_abs_diff = 4.2e-16
F2 with XMVB_CPP_SELECTED_STATE_SUPPORT_SPARSE=on and directional tile on:
  matrix_form_sum_sso_max_abs_diff = 2.8e-16
  matrix_form_sum_hho_max_abs_diff = 2.4e-16
  matrix_form_sum_ggo_max_abs_diff = 4.2e-16
F2 combined with same-spin local tile and opposite-spin tile switches:
  matrix_form_sum_sso/hho/ggo max_abs_diff ~= 1e-16
241 tnhvp 32 cores forced same-spin directional tile:
  Slurm job 1943693 converged in 8 iterations
  final energy = -230.720590230795
  SCF iteration wall time = 14.88 s
  End-to-end wall time = 16.00 s
```

### 2026-04-25: opposite-spin directional selected-state packed-gradient tile path

已完成 selected-state directional outer-response 中 opposite-spin packed
two-electron gradient 的 tile path。旧路径对每个 beta packed pair 先构造全局

\[
\dot W^\alpha_Q\in\mathbb R^{N_\alpha\times N_\alpha},
\]

再与 alpha sparse channel 收缩。新路径改为在 alpha unique tile \(I,J\) 上直接构造：

\[
\dot W^\alpha_Q[I,J]
=
\sum_s w_s
\left(
\dot C_s[I,:]U^\beta_Q C_s[J,:]^T
+
C_s[I,:]U^\beta_Q\dot C_s[J,:]^T
\right).
\]

实现上不再构造 accepted/directional union coefficient matrix，而是在
`accumulate_directional_alpha_pair_matrix_tile(...)` 里对两个 mixed-support
product-rule 项分别 streaming：

```text
dC support x C support
C support x dC support
```

然后立即调用 `contract_sparse_matrix_tile_with_dense_tile_matrix(...)` 和 alpha
sparse packed-pair block 收缩，只保留当前 alpha unique tile 的 weight
workspace。

历史阶段控制变量沿用 accepted/local packed-gradient tile：

```text
XMVB_CPP_OPPOSITE_SPIN_PACKED_GRADIENT_TILE=on|off
```

2026-04-25 清理后该开关已退出 production；directional selected-state
packed-gradient 也统一走 tile path。

验证：

```text
cmake --build build --target run_cpp_vbscf check_exact_ctx_hvp -j8
F2 with XMVB_CPP_OPPOSITE_SPIN_PACKED_GRADIENT_TILE=on:
  analytic_state_opposite_spin_sso_max_abs_diff = 0
  matrix_form_sum_sso_max_abs_diff = 3.3e-16
  matrix_form_sum_hho_max_abs_diff = 2.4e-16
  matrix_form_sum_ggo_max_abs_diff = 4.2e-16
F2 combined with same-spin directional/local tile and opposite-spin tile switches:
  matrix_form_sum_sso/hho/ggo max_abs_diff ~= 1e-16
241 tnhvp 32 cores with all current tile switches forced:
  Slurm job 1943696 converged in 8 iterations
  final energy = -230.720590230794
  SCF iteration wall time = 14.85 s
  End-to-end wall time = 16.06 s
```

### 2026-04-25: opposite-spin directional/local overlap-adjoint tile path

已完成 selected-state directional 和 local-response 两类 opposite-spin
overlap adjoint 的 tile path。selected-state directional 只对结构系数响应求导，
pair cache payload 固定：

\[
\dot W^\alpha_P[I,J]
=
\sum_s w_s
\left(
\dot C_s[I,:]V^\beta_P C_s[J,:]^T
+
C_s[I,:]V^\beta_P\dot C_s[J,:]^T
\right),
\]

\[
\dot\eta_{aa'}
=
\sum_P x^\alpha_P(a,a')\,\dot W^\alpha_P(a,a').
\]

local-response 路径同时包含 partner-side kernel 的方向导数和 target-side
inverse-overlap projection 的方向导数：

\[
\delta\eta_{aa'}
=
\sum_P x^\alpha_P(a,a')\,\delta W^\alpha_P(a,a')
+
\sum_P \delta x^\alpha_P(a,a')\,W^\alpha_P(a,a').
\]

其中

\[
\delta W^\alpha_P[I,J]
=
\sum_s w_s C_s[I,:]\delta V^\beta_P C_s[J,:]^T.
\]

beta 侧完全对应，把 alpha/beta 角色互换即可。

### 2026-04-25: contiguous Eigen tile bundles

opposite-spin packed-gradient tile 和 overlap-adjoint tile 的临时权重布局已经从

```text
std::vector<Eigen::MatrixXd>  // one small matrix per packed active pair
```

改为一个列主序 Eigen 矩阵：

```text
Eigen::MatrixXd tile_matrix(tile_left * tile_right, n_packed_active_pairs)
```

第 \(P\) 列存储 \(W_P(i,j)\)，展平索引为

\[
\mathrm{tile\_index}=i_{\mathrm{local}}+
N_{\mathrm{left}}j_{\mathrm{local}}.
\]

这样仍然只保留当前 unique-pair tile，不引入 \(N_\mathrm{unique}^2\) dense
image cache，但避免了每个 packed active pair 一个小矩阵带来的分配碎片和复制。
packed-gradient 收缩直接从这个 bundle 按展平索引读取；overlap adjoint 构造
occupied-orbital 小矩阵和 inverse-overlap gradient 时也直接读取同一个 bundle。
local-response tile path 中 accepted/directional dense weight tile 是否为零的
判断也已提前到 beta packed-pair batch 构造阶段，避免在 alpha packed-pair block
内重复全 tile 扫描。

验证：

```text
cmake --build build --target run_cpp_vbscf check_exact_ctx_hvp -j8
F2 all tile switches forced:
  matrix_form_sum_sso_max_abs_diff = 3.3e-16
  matrix_form_sum_hho_max_abs_diff = 2.4e-16
  matrix_form_sum_ggo_max_abs_diff = 4.2e-16
  analytic_state_opposite_spin_sso_max_abs_diff = 0
  local_opposite_spin_sso_max_abs_diff = 2.8e-17
  local_opposite_spin_ggo_max_abs_diff = 5.6e-12
241 tnhvp 32 cores with all tile switches forced:
  Slurm job 1943699 on w021 converged in 8 iterations
  final energy = -230.720590230793
  SCF iteration wall time = 14.924927 s
  End-to-end wall time = 16.135858 s
241 tnhvp 32 cores with default threshold strategy:
  Slurm job 1943700 on w021 converged in 8 iterations
  final energy = -230.720590230795
  SCF iteration wall time = 14.801474 s
  End-to-end wall time = 16.034393 s
```

241 上该布局与上一版 all-tile 基本持平；这符合预期，因为 241 的 active/unique
空间较小，主要收益不是减少 FLOP，而是让大体系 tile path 的临时内存更连续、
分配次数更少。

### 2026-04-25: production dense fallback cleanup

`opposite_spin_matrix_backward.cpp` 已删除 opposite-spin production dense fallback。
当前生产路径统一为 sparse packed-pair block + unique-spin tile contraction：

```text
accepted packed-gradient        -> accumulate_opposite_spin_packed_gradient_by_tiles
directional packed-gradient     -> accumulate_directional_opposite_spin_packed_gradient_by_tiles
local-response packed-gradient  -> accumulate_local_opposite_spin_packed_gradient_by_tiles
accepted overlap-adjoint        -> tile overlap weight bundle
directional overlap-adjoint     -> directional tile overlap weight bundle
local-response overlap-adjoint  -> accepted + directional tile overlap weight bundle
```

删除的旧代码包括：

```text
packed_pair -> N_unique^2 dense provider/cache
selected-state dense coefficient_matrix fallback contraction
close-shell diagonal dense fallback helpers
dense fallback threshold/env dispatch
per-packed-pair std::vector<Eigen::MatrixXd> tile bundle
```

旧实验开关
`XMVB_CPP_OPPOSITE_SPIN_PACKED_GRADIENT_TILE`、
`XMVB_CPP_OPPOSITE_SPIN_OVERLAP_TILE` 和
`XMVB_CPP_OPPOSITE_SPIN_OVERLAP_TILE_MIN_DENSE_BYTES` 已不再是 production
控制项；后续只保留 tile/block size 参数用于调优。

验证：

```text
cmake --build build --target run_cpp_vbscf check_exact_ctx_hvp -j8
F2 default check_exact_ctx_hvp after cleanup:
  matrix_form_sum_sso_max_abs_diff = 3.33066907388e-16
  matrix_form_sum_hho_max_abs_diff = 2.42861286637e-16
  matrix_form_sum_ggo_max_abs_diff = 4.16333634234e-16
  local_opposite_spin_sso_max_abs_diff = 2.77555756156e-17
  local_opposite_spin_ggo_max_abs_diff = 5.55111512313e-12
241 tnhvp 32 cores after cleanup:
  Slurm job 1943723 on w021 converged in 8 iterations
  final energy = -230.720590230793
  SCF iteration wall time = 14.878882 s
  End-to-end wall time = 16.109432 s
```
