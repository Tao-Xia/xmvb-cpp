# Unique-Spin 到 Structure Basis 的 Query-Driven 收缩算法

## 1. 目标

本文档讨论当前 VBSCF / `exact_ctx` 路径中一个核心算法问题：

$$
\text{如何把 unique-spin-string 级别的矩阵元收缩到 structure basis，}
$$

同时满足以下三个目标：

1. 不构造全局 \(N_{\alpha} \times N_{\alpha}\) / \(N_{\beta} \times N_{\beta}\) 的大 dense 权重矩阵；
2. 不扫描 \(N_{\mathrm{det}}^2\) 个 determinant pair；
3. 能在大活性空间和大 structure space 上保持可并行、可扩展、且不易 OOM。

本文的结论是：

$$
\text{最优范式不是全局寻址归约，也不是全局 dense 收缩，而是}
\quad
\textbf{query-driven block-sparse contraction}.
$$

更准确地说，应当统一成两层结构：

1. determinant expansion 先压缩成 **support-local coefficient blocks**；
2. 对所有 forward / backward kernel，只对“当前真正被请求的 unique-pair 子集”执行局部收缩。

这意味着：

- forward 结构矩阵构造应采用 **support-local sparse contraction**；
- backward 选态权重构造应采用 **query-driven pair evaluation**；
- 全局 \(N_{\alpha}^2\) / \(N_{\beta}^2\) 权重矩阵应该从生产路径中删除。

---

## 2. 记号

### 2.1 unique-spin 空间

记

$$
N_{\alpha} = \text{number of unique alpha strings}, \qquad
N_{\beta} = \text{number of unique beta strings}.
$$

full determinant \(d\) 经过 unique-spin 压缩后对应

$$
u_{\alpha}(d) \in \{1,\dots,N_{\alpha}\}, \qquad
u_{\beta}(d) \in \{1,\dots,N_{\beta}\}.
$$

这里 \(u_{\alpha}\) 与 \(u_{\beta}\) 对应代码中的

- `alpha_reuse_table.determinant_to_unique_id`
- `beta_reuse_table.determinant_to_unique_id`

见
[same_spin_pair_cache.hpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/matrices/same_spin_pair_cache.hpp).

### 2.2 structure coefficient block

对任意 structure \(I\)，定义其在 unique-spin product basis 上的系数矩阵

$$
\mathbf{C}_I \in \mathbb{R}^{N_{\alpha} \times N_{\beta}},
$$

其中

$$
(\mathbf{C}_I)_{ab}
=
\sum_{d \in I \atop u_{\alpha}(d)=a,\,u_{\beta}(d)=b}
\gamma_{dI}.
$$

实际实现中并不保留整张 \(\mathbf{C}_I\)，而只保留其非零支撑：

$$
A_I = \{a \mid \exists b,\; (\mathbf{C}_I)_{ab} \neq 0\}, \qquad
B_I = \{b \mid \exists a,\; (\mathbf{C}_I)_{ab} \neq 0\}.
$$

令

$$
r_I = |A_I|, \qquad c_I = |B_I|,
$$

则局部块为

$$
\mathbf{C}_I^{\mathrm{loc}}
=
\mathbf{C}_I[A_I, B_I]
\in \mathbb{R}^{r_I \times c_I}.
$$

这对应代码中的
[structure_coefficient_blocks.hpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/matrices/structure_coefficient_blocks.hpp)
里的 `StructureCoefficientBlock`。

### 2.3 selected-state coefficient block

对选定态 \(n\)，定义其 unique-spin 系数矩阵

$$
\mathbf{Z}_n \in \mathbb{R}^{N_{\alpha} \times N_{\beta}},
$$

并同样只保留支撑

$$
A_n, \qquad B_n,
$$

以及局部块

$$
\mathbf{Z}_n^{\mathrm{loc}}
=
\mathbf{Z}_n[A_n, B_n].
$$

对应代码见
[selected_state_determinant_matrices.hpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/selected_state_determinant_matrices.hpp).

### 2.4 unique-spin kernel

记 alpha / beta same-spin 核为

$$
\mathbf{K}^{\alpha} \in \mathbb{R}^{N_{\alpha} \times N_{\alpha}}, \qquad
\mathbf{K}^{\beta} \in \mathbb{R}^{N_{\beta} \times N_{\beta}}.
$$

例如：

- overlap 核 \( \mathbf{S}^{\alpha}, \mathbf{S}^{\beta} \)
- one-electron 核 \( \mathbf{h}^{\alpha}, \mathbf{h}^{\beta} \)
- same-spin total 核 \( \mathbf{T}^{\alpha}, \mathbf{T}^{\beta} \)

另外，对 opposite-spin channel，定义 packed-pair index

$$
P \in \{1,\dots,N_P\}, \qquad
N_P = \frac{n_{\mathrm{act}}(n_{\mathrm{act}}+1)}{2},
$$

对应每个 packed pair 的 sparse same-spin channel

$$
\mathbf{U}^{\alpha}_P \in \mathbb{R}^{N_{\alpha} \times N_{\alpha}}, \qquad
\mathbf{U}^{\beta}_P \in \mathbb{R}^{N_{\beta} \times N_{\beta}}.
$$

---

## 3. 当前几种收缩范式

### 3.1 范式 A：determinant-pair 级寻址归约

最直接的做法是写成

$$
M_{IJ}
=
\sum_{d_L \in I}
\sum_{d_R \in J}
\gamma_{d_L I}\,
\mathcal{K}(d_L,d_R)\,
\gamma_{d_R J},
$$

其中

$$
\mathcal{K}(d_L,d_R)
$$

是 full determinant pair matrix element。

这一算法的核心问题是：

$$
\text{时间复杂度直接依赖 } N_{\mathrm{det}}^2.
$$

即使利用 structure sparsity，也依然要反复处理 determinant pair 粒度对象。对于大活性空间体系，这一层通常完全不可接受。

### 3.2 范式 B：全局 unique-spin dense 矩阵乘法

把 determinant 压缩为 unique-spin 后，可写成

$$
M_{IJ}
=
\left\langle
\mathbf{C}_I,\;
\mathbf{K}^{\alpha} \mathbf{C}_J (\mathbf{K}^{\beta})^{\mathsf T}
\right\rangle_F.
$$

若进一步为 backward 构造全局权重矩阵，例如

$$
\mathbf{W}^{\alpha}
\in
\mathbb{R}^{N_{\alpha} \times N_{\alpha}},
$$

则在数学上很干净，但实现代价是：

$$
\mathcal{O}(N_{\alpha}^2), \qquad
\mathcal{O}(N_{\beta}^2)
$$

级别的常驻 dense 对象，以及大量

$$
\mathcal{O}(N_{\alpha}^2 N_{\beta}), \qquad
\mathcal{O}(N_{\beta}^2 N_{\alpha})
$$

级别的无差别扫描。

这正是当前 OOM 风险的主要来源之一。

### 3.3 范式 C：support-local 收缩

对当前结构对 \((I,J)\)，只 gather

$$
\mathbf{K}^{\alpha}[A_I, A_J], \qquad
\mathbf{K}^{\beta}[B_I, B_J],
$$

然后收缩

$$
M_{IJ}
=
\left\langle
\mathbf{C}_I^{\mathrm{loc}},\;
\mathbf{K}^{\alpha}[A_I, A_J]\,
\mathbf{C}_J^{\mathrm{loc}}\,
\left(\mathbf{K}^{\beta}[B_I, B_J]\right)^{\mathsf T}
\right\rangle_F.
$$

这比全局 dense 做法好很多，因为它的空间规模变成

$$
\mathcal{O}(r_I r_J + c_I c_J + r_I c_I + r_J c_J),
$$

而不是

$$
\mathcal{O}(N_{\alpha}^2 + N_{\beta}^2).
$$

当前 forward 和 exact directional 的主路径已经接近这种思路。

### 3.4 范式 D：query-driven block-sparse contraction

这是本文推荐统一采用的模式。

关键点是：

$$
\text{不提前构造全部 pair 值，只回答当前真正请求的 pair query。}
$$

在 backward 中，这一点尤其重要，因为通常只需要：

- 当前 sparse channel 中出现的 unique-pair；
- 当前 structure pair / selected-state 支撑真正命中的 pair；
- 当前 batch 中被请求的少量 \((a_L,a_R)\) 或 \((b_L,b_R)\)。

因此，最佳算法不是构造

$$
\mathbf{W}^{\alpha},\quad \mathbf{W}^{\beta},
$$

而是定义一个 query evaluator：

$$
(a_L,a_R) \mapsto W^{\alpha}(a_L,a_R),
\qquad
(b_L,b_R) \mapsto W^{\beta}(b_L,b_R),
$$

并且只在实际请求时计算。

---

## 4. forward 结构矩阵的最优公式

### 4.1 overlap / one-electron / same-spin total

对 structure 对 \((I,J)\)，精确公式统一为

$$
M_{IJ}
=
\sum_{a_L \in A_I}
\sum_{a_R \in A_J}
\sum_{b_L \in B_I}
\sum_{b_R \in B_J}
C_I(a_L,b_L)\,
K^{\alpha}(a_L,a_R)\,
K^{\beta}(b_L,b_R)\,
C_J(a_R,b_R).
$$

如果把

$$
\mathbf{r}_{J,a_R}(b_R) = C_J(a_R,b_R)
$$

视为右侧 alpha row 上的一行 beta 向量，则可先做

$$
\mathbf{p}_{J,a_R}(b_L)
=
\sum_{b_R \in B_J}
K^{\beta}(b_L,b_R)\,
\mathbf{r}_{J,a_R}(b_R).
$$

然后再收缩

$$
M_{IJ}
=
\sum_{a_R \in A_J}
\sum_{a_L \in A_I}
K^{\alpha}(a_L,a_R)\,
\sum_{b_L \in B_I}
C_I(a_L,b_L)\,
\mathbf{p}_{J,a_R}(b_L).
$$

这正对应 row-sparse contraction 的实现思路：

1. 固定右侧 alpha row；
2. 先做 beta kernel 投影；
3. 再只遍历左侧真实非零项做 dot。

### 4.2 标度

若记

$$
\mathrm{nnz}(I) = \# \{(a,b) \in A_I \times B_I \mid C_I(a,b) \neq 0\},
$$

则单个 structure pair 的开销可粗略写成

$$
T_{IJ}^{\mathrm{forward}}
\approx
\mathcal{O}\!\left(
\sum_{a_R \in A_J}
\left(
\mathrm{nnz}_{a_R}(J)\, c_I
+
\sum_{a_L \in A_I}
\mathrm{nnz}_{a_L}(I)
\right)
\right),
$$

其中 \(\mathrm{nnz}_{a}(I)\) 是结构 \(I\) 在 alpha row \(a\) 上的非零 beta 数。

若把平均 row 非零数记为

$$
\bar z_I = \frac{\mathrm{nnz}(I)}{r_I}, \qquad
\bar z_J = \frac{\mathrm{nnz}(J)}{r_J},
$$

则可近似写成

$$
T_{IJ}^{\mathrm{forward}}
\approx
\mathcal{O}\!\left(
r_J \bar z_J c_I + r_I r_J \bar z_I
\right).
$$

当 \( \bar z_I, \bar z_J \ll c_I, c_J \) 时，这显著优于局部 dense 三重乘法。

### 4.3 forward 的最优实现结论

forward 最优模式是：

$$
\textbf{support-local sparse contraction}
$$

即：

- 全局不做 determinant-pair 扫描；
- 只 gather 当前 structure pair 需要的 kernel 子块；
- 局部 support 小且稠密时可暂时走 BLAS；
- 局部 support 大且稀疏时走 row-sparse contraction。

---

## 5. backward 的真正瓶颈

设某个 partner kernel 为

$$
\mathbf{B}_n \in \mathbb{R}^{|B_n| \times |B_n|},
$$

则 alpha-side 权重矩阵通常可写成

$$
W^{\alpha}(a_L,a_R)
=
\sum_n
w_n
\sum_{b_L \in B_n}
\sum_{b_R \in B_n}
Z_n(a_L,b_L)\,
B_n(b_L,b_R)\,
Z_n(a_R,b_R).
$$

如果对所有 \((a_L,a_R)\) 都构造完整

$$
\mathbf{W}^{\alpha} \in \mathbb{R}^{N_{\alpha} \times N_{\alpha}},
$$

则：

1. 空间需要
   $$
   \mathcal{O}(N_{\alpha}^2),
   $$
   极易 OOM；
2. 时间需要对大量永远不会被当前 sparse channel 用到的 pair 也做计算；
3. 这一步与实际 structure / selected-state 支撑稀疏性脱节。

因此 backward 中最根本的优化，不是减少线程，而是：

$$
\text{完全删掉全局 } \mathbf{W}^{\alpha}, \mathbf{W}^{\beta} \text{ 的生产式构造。}
$$

---

## 6. query-driven pair evaluation

### 6.1 query set

设当前只需要一组被请求的 alpha pair：

$$
\mathcal{Q}_{\alpha}
\subseteq
\{1,\dots,N_{\alpha}\} \times \{1,\dots,N_{\alpha}\}.
$$

例如：

- 当前 alpha sparse channel 中真实出现的 row/column；
- 当前 overlap pullback 所需要的 unique pair；
- 当前一个 block 内的一小批 query。

目标不再是构造整张 \(\mathbf{W}^{\alpha}\)，而是只计算：

$$
\forall (a_L,a_R) \in \mathcal{Q}_{\alpha},
\qquad
W^{\alpha}(a_L,a_R).
$$

### 6.2 postings index

定义 alpha-side postings：

$$
\mathcal{P}_{\alpha}(a)
=
\{n \mid a \in A_n\}.
$$

同理 beta-side postings：

$$
\mathcal{P}_{\beta}(b)
=
\{n \mid b \in B_n\}.
$$

则对于某个 query pair \((a_L,a_R)\)，只有那些同时满足

$$
n \in \mathcal{P}_{\alpha}(a_L) \cap \mathcal{P}_{\alpha}(a_R)
$$

的 selected state 才可能贡献非零项。

因此：

$$
W^{\alpha}(a_L,a_R)
=
\sum_{n \in \mathcal{P}_{\alpha}(a_L) \cap \mathcal{P}_{\alpha}(a_R)}
w_n\,
\mathbf{z}_{n,a_L}^{\mathsf T}\,
\mathbf{B}_n\,
\mathbf{z}_{n,a_R},
$$

其中

$$
\mathbf{z}_{n,a}
\in
\mathbb{R}^{|B_n|}
$$

表示状态 \(n\) 在 alpha row \(a\) 上的 beta 系数行向量。

### 6.3 单 query 的精确计算

对任一 \((a_L,a_R)\)，可直接按下式求值：

$$
W^{\alpha}(a_L,a_R)
=
\sum_{n \in \mathcal{P}_{\alpha}(a_L) \cap \mathcal{P}_{\alpha}(a_R)}
w_n
\sum_{b_L \in B_n}
\sum_{b_R \in B_n}
Z_n(a_L,b_L)\,
B_n(b_L,b_R)\,
Z_n(a_R,b_R).
$$

这一步完全不要求形成任何全局 \(N_{\alpha} \times N_{\alpha}\) dense 矩阵。

### 6.4 block query

若 query set 是一个小块

$$
\mathcal{Q}_{\alpha}^{\mathrm{blk}}
=
R_L \times R_R,
\qquad
R_L, R_R \subseteq \{1,\dots,N_{\alpha}\},
$$

则更好的方式不是逐 pair 独立做 postings 交，而是按 state 聚合：

1. 对每个 selected state \(n\)，求
   $$
   R_n^L = R_L \cap A_n, \qquad
   R_n^R = R_R \cap A_n;
   $$
2. gather
   $$
   \mathbf{B}_n = \mathbf{K}^{\beta}[B_n, B_n];
   $$
3. 计算
   $$
   \mathbf{P}_n
   =
   \mathbf{Z}_n[R_n^L, B_n]\,
   \mathbf{B}_n;
   $$
4. 输出
   $$
   \mathbf{W}^{\alpha}[R_n^L, R_n^R]
   \mathrel{+}= 
   w_n\,
   \mathbf{P}_n\,
   \mathbf{Z}_n[R_n^R, B_n]^{\mathsf T}.
   $$

于是单个状态的局部工作量为

$$
\mathcal{O}\!\left(
|R_n^L|\, |B_n|^2
+
|R_n^L|\, |R_n^R|\, |B_n|
\right),
$$

而不是

$$
\mathcal{O}(N_{\alpha}^2 |B_n|).
$$

这就是 query-driven block-sparse contraction 的核心公式。

---

## 7. 标度比较

为方便比较，定义平均 selected-state 支撑尺寸：

$$
\bar r = \frac{1}{N_{\mathrm{sel}}}\sum_n |A_n|,
\qquad
\bar c = \frac{1}{N_{\mathrm{sel}}}\sum_n |B_n|.
$$

并记当前 query block 大小为

$$
q_{\alpha} = |\mathcal{Q}_{\alpha}|,
\qquad
q_{\beta} = |\mathcal{Q}_{\beta}|.
$$

### 7.1 determinant-pair 扫描

粗略时间：

$$
T_{\mathrm{det}}
=
\mathcal{O}(N_{\mathrm{det}}^2).
$$

空间：

$$
S_{\mathrm{det}}
=
\mathcal{O}(1)
\quad \text{到} \quad
\mathcal{O}(N_{\mathrm{det}}^2)
$$

取决于是否缓存 determinant pair payload。

### 7.2 全局 dense 权重矩阵

时间：

$$
T_{\mathrm{dense}}
\approx
\mathcal{O}\!\left(
N_{\mathrm{sel}}\, N_{\alpha}^2 \bar c
+
N_{\mathrm{sel}}\, N_{\beta}^2 \bar r
\right).
$$

空间：

$$
S_{\mathrm{dense}}
=
\mathcal{O}(N_{\alpha}^2 + N_{\beta}^2).
$$

若还叠加 packed-pair dense image cache，则会进一步乘上 \(N_P\) 的 batch 因子，最容易 OOM。

### 7.3 support-local forward

对 structure pair \((I,J)\)：

$$
T_{IJ}^{\mathrm{forward}}
\approx
\mathcal{O}(r_I r_J c_{\mathrm{eff}} + c_I c_J r_{\mathrm{eff}})
$$

或更细致地写成 row-sparse 形式：

$$
T_{IJ}^{\mathrm{forward}}
\approx
\mathcal{O}(r_J \bar z_J c_I + r_I r_J \bar z_I).
$$

空间：

$$
S_{IJ}^{\mathrm{forward}}
=
\mathcal{O}(r_I r_J + c_I c_J + \mathrm{nnz}(I) + \mathrm{nnz}(J)).
$$

### 7.4 query-driven block-sparse backward

若把 postings 交后的平均命中状态数记为

$$
\bar m_{\alpha}
=
\frac{1}{q_{\alpha}}
\sum_{(a_L,a_R)\in \mathcal{Q}_{\alpha}}
\left|
\mathcal{P}_{\alpha}(a_L)\cap\mathcal{P}_{\alpha}(a_R)
\right|,
$$

则单 query 方案的时间大致为

$$
T_{\mathrm{query},\alpha}
\approx
\mathcal{O}\!\left(
q_{\alpha}\, \bar m_{\alpha}\, \bar c^2
\right).
$$

若采用按 state 的 block 聚合，则时间更接近

$$
T_{\mathrm{blk},\alpha}
\approx
\sum_n
\mathcal{O}\!\left(
|R_n^L|\, |B_n|^2
+
|R_n^L|\, |R_n^R|\, |B_n|
\right),
$$

空间只需要

$$
S_{\mathrm{blk},\alpha}
=
\mathcal{O}\!\left(
\max_n |R_n^L|\, |B_n|
+
\max_n |R_n^L|\, |R_n^R|
\right),
$$

即单个 block / 单个 worker 的局部 scratch，而不是全局 \(N_{\alpha}^2\) dense matrix。

因此只要

$$
q_{\alpha} \ll N_{\alpha}^2,
\qquad
|A_n| \ll N_{\alpha},
\qquad
|B_n| \ll N_{\beta},
$$

query-driven block-sparse 就会同时在时间和空间上优于全局 dense 权重矩阵。

---

## 8. 并行策略

### 8.1 不应并行复制什么

最危险的并行方式是让每个线程各自持有：

- 一份 \(N_{\alpha} \times N_{\alpha}\) dense weight matrix；
- 一份 \(N_{\beta} \times N_{\beta}\) dense weight matrix；
- 一批 packed-pair dense image block cache；
- 与活性空间大小 \(N_P\) 成正比的大型 thread-local dense buffer。

这会把空间复杂度从

$$
S
$$

放大成

$$
T_{\mathrm{threads}} \times S,
$$

非常容易在 `MnF2` 这类体系上炸掉。

### 8.2 合理的并行粒度

最适合的并行粒度是：

1. structure pair block；
2. selected-state query block；
3. packed-pair block；
4. postings-query batch。

也就是说，线程私有对象应该只包含：

- 当前 block 的局部 support gather；
- 小型 scratch；
- 少量局部投影缓存；

而不能包含按 \(N_{\alpha}^2\)、\(N_{\beta}^2\)、\(N_P\) 成长的大对象。

---

## 9. 推荐统一方案

### 9.1 forward

统一成：

$$
\textbf{support-local sparse contraction}
$$

即：

- determinant 先压成 `StructureCoefficientBlock`；
- 每个 structure pair 只 gather 当前支撑子块；
- 小块 dense、 大块 sparse-row，自适应切换；
- 不回到 determinant-pair 粒度。

### 9.2 backward

统一成：

$$
\textbf{query-driven block-sparse contraction}
$$

即：

- 不构造全局 \(\mathbf{W}^{\alpha}, \mathbf{W}^{\beta}\)；
- 只暴露 pair-query evaluator / block-query evaluator；
- 用 postings 把状态空间裁剪到真正命中的状态；
- 每个状态只在自己的 support-local block 上做收缩；
- 结果直接喂给当前 sparse channel / overlap pullback / local response。

### 9.3 数据结构建议

统一保留以下 accepted-point 常驻对象：

1. determinant-to-unique topology；
2. `StructureCoefficientBlock`；
3. `SelectedStateDeterminantCoefficients`；
4. per-row CSR metadata；
5. per-row / per-column postings index。

应删除或逐步淘汰的常驻对象：

1. 全局 \(N_{\alpha}\times N_{\alpha}\) dense 权重矩阵；
2. 全局 \(N_{\beta}\times N_{\beta}\) dense 权重矩阵；
3. packed-pair dense image block cache；
4. 任何按线程复制上述大对象的策略。

---

## 10. 结论

从算法上看，unique-spin 到 structure basis 的最优收缩范式是：

$$
\boxed{
\text{support-local forward}
\;+\;
\text{query-driven block-sparse backward}
}
$$

它的核心优点是：

1. 避免
   $$
   \mathcal{O}(N_{\mathrm{det}}^2)
   $$
   的 determinant-pair 扫描；
2. 避免
   $$
   \mathcal{O}(N_{\alpha}^2 + N_{\beta}^2)
   $$
   的全局 dense 权重矩阵常驻；
3. 复杂度与真实支撑大小和真实 query set 对齐；
4. 并行时只复制局部 scratch，不复制全局大对象；
5. 可以从根本上消除当前 backward 路径中最危险的 OOM 来源。

因此，若目标是统一整个 VBSCF / `exact_ctx` 路径的 unique-spin 收缩框架，那么后续的实现重构方向应当是：

$$
\textbf{把所有 backward 路径重写为 query-driven evaluator，}
$$

而不是继续围绕全局 dense 权重矩阵做缓存、压缩或线程数调参。
