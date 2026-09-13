# TNHVP 结构性优化分析

## 1. 问题范围

这里讨论的是 `exact_ctx` 路径下，TNHVP 每次 Hessian-vector product (`H v`) 的核心耗时来源。当前代码中最硬的两个热点是：

1. `h1e_fused`
2. `active_2e`

前者对应 AO 有效一电子算子的方向作用与转置回传，后者对应活性二电子项的固定伴随 Hessian-vector product。

本文只讨论“结构性优化”是否存在，即是否能从数学数据流或中间量组织方式上减少整层计算，而不是只调线程数、缓存大小或收敛参数。

## 2. 当前 HVP 的核心数据流

记

\[
n_b = \text{AO 基函数数}, \qquad
n_a = \text{活性轨道数},
\]

\[
n_p = \frac{n_b(n_b+1)}{2}, \qquad
n_q = \frac{n_a(n_a+1)}{2}.
\]

其中：

- `dense_active_direction` 的尺寸为 \(n_b \times n_a\)
- AO-pair 空间尺寸为 \(n_p\)
- active-pair 空间尺寸为 \(n_q\)

### 2.1 `active_2e` 当前最佳路径

当前 accepted-point cache 已经把一部分与方向 \(D\) 无关的量预先存下来了，但每次 HVP 仍然大致执行下面这条链：

\[
D
\;\xrightarrow{\;\mathcal{B}'(C)\;}\;
M
\;\xrightarrow{\;K\;}\;
K M
\;\xrightarrow{\;\text{cached backprop}\;}\;
\nabla_D
\]

其中：

- \(C\) 是 accepted point 的活性轨道系数
- \(D\) 是本次方向
- \(\mathcal{B}(C)\) 表示从活性轨道系数构造 AO-pair 到 active-pair 系数矩阵的映射
- \(M = \mathcal{B}'(C)[D]\) 是方向导数，对应当前代码中的 `mixed_pair_coefficients_buffer`
- \(K\) 是 AO-pair kernel，对应精确二电子积分定义的稀疏线性算子

当前代码的快路径已经做了两件正确的事：

1. 不再每次重建 accepted point 的固定量
2. 在 `apply_exact_ao_pair_kernel_and_backprop_from_cached_rows(...)` 中，把

\[
(K M) G \to \text{backprop}
\]

中的部分中间层融合掉，避免每次都显式构造完整的 `pair_gradients`

但它**还没有**消掉 \(M\) 的显式构造。

### 2.2 `h1e_fused` 当前路径

`h1e_fused` 当前做的是同一份 AO ERI 流上的两个线性算子：

\[
\delta F_{11} = K_{h1e}[\delta P_{11}],
\qquad
\delta P_{11}^{\ast} = K_{h1e}^{T}[\Lambda].
\]

代码已经把两者融合在一次 AO 积分扫过里完成，这一步方向上是对的。问题不在数学上多做了一遍，而在于并行实现上仍然存在：

1. 每线程完整输出缓冲区
2. 末端大规模归并
3. 图路径下转置输出仍然是散写模式

因此 `h1e_fused` 更像“访存组织问题”，而不是公式层面还漏掉了一个大化简。

## 3. 哪些是真正的结构性优化

## 3.1 `active_2e`：存在明确的结构性优化空间

### 3.1.1 当前仍然保留了一层不必要的中间矩阵

对任意 AO-pair \((\mu,\nu)\) 与 active-pair \((a,b)\)，当前构造的混合系数满足

\[
M_{(\mu\nu),(ab)}
=
D_{\mu a} C_{\nu b}
+ C_{\mu a} D_{\nu b}
+ D_{\nu a} C_{\mu b}
+ C_{\nu a} D_{\mu b},
\]

对角项按 packed-pair 约定退化即可。

这个量本质上只是 \(C\) 与 \(D\) 的双线性组合。也就是说，它不是一个需要“先完整生成、再交给后续核函数”的物理对象；它只是后续 AO-pair kernel 在每一行上临时需要的源向量。

因此，当前先构造

\[
M \in \mathbb{R}^{n_p \times n_q}
\]

再把它送进 AO-pair kernel，本质上是在为后续流式算子准备一个完整中间层。这一层在数学上不是必须的。

### 3.1.2 可行的直接方向融合

更直接的路径应当是：

\[
D
\;\xrightarrow{\;\text{on-the-fly } \mathcal{B}'(C)\;}\;
\text{stream into } K
\;\xrightarrow{\;\text{fused cached backprop}\;}\;
\nabla_D.
\]

也就是：

1. 不再显式构造 `mixed_pair_coefficients_buffer`
2. AO-pair kernel 在访问某个源 AO-pair 行 \((\rho,\sigma)\) 时，直接用 accepted \(C\) 与当前方向 \(D\) 的两行现场生成该行对应的 active-pair 源向量
3. 生成后立刻参与 kernel 累积与 cached backprop
4. 不把整个 \(n_p \times n_q\) 矩阵落地

这条改动的意义不是降低理论主标度，而是删掉一整个巨大的中间层及其一次完整预扫。对当前代码而言，这是**最像“本质结构优化”**的一条。

### 3.1.3 复杂度变化

设 AO-pair kernel 图的边数为 \(m_K\)。

当前快路径大致包含：

\[
\mathcal{O}(n_p n_q)
\quad+\quad
\mathcal{O}(m_K \cdot n_q)
\quad+\quad
\mathcal{O}(\text{backprop streaming}).
\]

其中第一项就是显式构造 \(M\) 的成本，同时还带来一份大的写带宽和读带宽。

直接方向融合后，主导项变为：

\[
\mathcal{O}(m_K \cdot n_q)
\quad+\quad
\mathcal{O}(\text{backprop streaming}),
\]

外加每条边或每个源行上对少量 active-pair 分量的现场生成。由于 \(n_q\) 很小，而 \(n_p\) 可能很大，这通常能显著降低 wall-time 中的内存搬运部分。

结论：`active_2e` 的下一步优化，不是再压榨一个更小的 BLAS 常数，而是把 `build_mixed_ao_pair_to_active_pair_coefficients_from_cache(...)` 这一整层从热路径上拿掉。

## 3.2 `h1e_fused`：主要是大常数优化，不是同等级的结构化简

`h1e_fused` 当前已经做对了最重要的一件事：把正向作用和转置回传放在同一遍 AO ERI 扫描中完成。

因此这里很难再像 `active_2e` 那样删掉一整层数学中间量。更现实的方向是：

1. 取消“每线程一个完整转置输出矩阵”的策略
2. 改成 source-tile 或 block-reduction 组织
3. 必要时在多线程下拆成两个更规整的流式 pass，而不是坚持单遍融合

这里的收益主要来自：

- 更少的临时内存
- 更少的归并带宽
- 更好的 cache locality

所以它属于“实现结构优化”，但不属于“公式层面减少一层对象”的那种本质化简。

## 4. 哪些方向不属于当前最值得做的结构优化

### 4.1 继续调 TN 内部 Krylov / warm start

这类工作对收敛迭代数有帮助，但对单次 HVP 的核心结构没有改变。当前热点已经表明，真正慢的是 exact-ctx matvec 本身，而不是 Krylov 外层框架。

### 4.2 单纯扩大 accepted-point cache

如果只是再多缓存几个大矩阵，但每次 HVP 仍然要先构造完整的 `mixed_pair_coefficients_buffer`，那只是把常数换了一个位置，并没有删掉热路径中的主要中间层。

### 4.3 进一步纠缠轨道正交/非正交语义

轨道参数化会影响收敛行为和数值语义，但针对当前已定位的 `active_2e / h1e_fused` 热点，它不是最直接的单步耗时来源。

## 5. 优先级判断

如果只选一条最值得继续落地的“结构性优化”，优先级应当是：

1. `active_2e` 直接方向融合
2. `h1e_fused` 并行归并重构

原因很简单：

- `active_2e` 这边还存在一层可以整体删掉的中间矩阵
- `h1e_fused` 这边主要是在已经正确的单遍融合上继续做并行访存优化

因此，从“本质性”这个标准看，`active_2e` 更强；从“实现风险”看，`h1e_fused` 也许更平滑，但上限通常不如前者明显。

## 6. 结论

有，而且是明确存在的，但主要集中在 `active_2e`。

更准确地说：

1. `active_2e` 还保留着一个并非数学上必须的 \(n_p \times n_q\) 中间层，具备真正的结构性消除空间
2. `h1e_fused` 已经完成了最关键的算子融合，后续更多是并行归并与访存组织优化
3. 因此，下一步若要追求“本质优化效果”，最值得做的是把 `active_2e` 改成“方向直接流入 AO-pair kernel 与 backprop”的单路径实现

这条路线如果落地成功，最可能带来的改进是：

- 更低的单次 HVP wall-time
- 更少的大型临时缓冲区
- 更稳定的多线程表现

但它的收益主要来自**减少中间层与内存搬运**，而不是改变最终数学结果，因此不会改变收敛目标，也不应改变最终轨道或能量语义。
