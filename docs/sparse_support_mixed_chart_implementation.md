# Sparse-Support Mixed Chart 与几何回退实现方案

## 1. 目标

本文档统一整理以下三件事：

1. 为什么当前 sparse-support 轨道更新的加法回退在几何上不理想；
2. 为什么在当前 VBSCF 非正交语义下，只能做**部分旋转**，不能把全部自由度都塞进一个全局 Cayley 旋转；
3. 如何把“内部正交 frame 的部分旋转”与“外部 fixed-support 稀疏存储”结合起来，形成可落地的代码实现方案。

本文档替代以下三份旧文档：

1. `docs/mixed_chart_rotation_parameterization.md`
2. `docs/sparse_support_geometric_retraction.md`
3. `docs/sparse_support_partial_rotation_mixed_chart.md`

---

## 2. 当前代码的真实状态

当前 `xmvb-cpp` 的 sparse VBSCF 主路径有两个关键事实。

### 2.1 外部优化变量是原始 sparse 系数

优化器真正更新的是

\[
\texttt{OrbitalPreparationInput::orbital\_value\_table},
\]

也就是每个轨道在自己显式 AO support 上的 raw sparse coefficient。

它不是已经满足最终几何约束的正交 frame，也不是某种天然的 Lie 群坐标。

### 2.2 当前 finite-step map 已经切到 internal mixed chart

当前 `NonredundantOrbitalSpace::retract_step()` 已经不再直接把 finite step
写成 support 内的原始加法更新。

现在的 finite-step 语义是：

1. 所有 block 先在统一的内部 mixed chart
   \((Q_i,Q_a,Q_v,L_a)\) 上构造 candidate；
2. inactive-active / inactive-virtual / active-virtual 自由度进入
   block 内部 Cayley retraction；
3. active-active 的非正交 shape 仍按 \(L_a + \Delta L_a\) 更新；
4. dense full-support block 直接把生成的 trial occupied columns 写回；
5. sparse block 先把该 internal trial point restriction 到每个轨道自己的
   fixed support，再在对应的 local \(x^{\mathrm T}Sx=1\) 流形上做几何回退。

因此，当前 sparse finite step 的源头已经不是旧的 raw additive local step。

---

## 3. 为什么不能直接做“全局旋转更新”

用户的核心直觉是对的：

1. 当前已经构造了正交化的非活性对象；
2. 如果再对活性辅助轨道作适当拆分，确实会出现可旋转的内部自由度；
3. 所以应该考虑部分使用旋转更新，而不是所有自由度都做加法更新。

但 sparse-support 下不能直接做

\[
Q^{+} = Q \, U
\]

这种全局列旋转，原因是：

1. 不同轨道的显式 support 一般不同；
2. 一旦列之间发生线性混合，某个轨道的分量就会泄露到它原来 support 之外；
3. 这样会直接破坏当前 `orbital_value_table` 的 per-orbital fixed-support 约束。

所以 sparse-support 真正允许的不是

\[
\text{全局 sparse 轨道旋转},
\]

而是

\[
\text{block 内部工作 frame 的部分旋转}
\;+\;
\text{每个轨道自己 support 上的局部几何回退}.
\]

---

## 4. 三类非活性轨道对象

讨论“非活性轨道能不能旋转”之前，必须区分三个对象。

### 4.1 物理非活性轨道

记物理非活性轨道为

\[
C_i \in \mathbb{R}^{N \times n_i}.
\]

一般有

\[
C_i^{\mathrm T} S C_i = M_i,
\qquad
M_i \neq I.
\]

所以 \(C_i\) 不是正交 frame。

### 4.2 双正交辅助非活性轨道

定义

\[
\widetilde C_i = C_i M_i^{-1},
\qquad
M_i = C_i^{\mathrm T} S C_i.
\]

则

\[
\widetilde C_i^{\mathrm T} S C_i = I.
\]

但一般

\[
\widetilde C_i^{\mathrm T} S \widetilde C_i \neq I.
\]

所以 \(\widetilde C_i\) 是双正交对象，不是正交 frame。它适合：

1. 投影；
2. 密度矩阵；
3. pullback；

但不适合直接作为旋转生成元的工作变量。

### 4.3 正交化后的内部非活性 frame

真正能用于旋转更新的对象是

\[
Q_i = C_i M_i^{-1/2}.
\]

于是

\[
Q_i^{\mathrm T} S Q_i = I.
\]

所以如果说“非活性轨道内部已经正交，可以部分旋转”，真正该进入 mixed chart 的对象是 \(Q_i\)，不是 \(\widetilde C_i\)。

---

## 5. 活性辅助轨道与活性非正交 shape

把非活性子空间从活性轨道中投掉。设

\[
P_i = Q_i Q_i^{\mathrm T} S.
\]

对物理活性轨道 \(C_a\)，定义辅助活性轨道

\[
T_a = (I - P_i) C_a.
\]

于是

\[
Q_i^{\mathrm T} S T_a = 0.
\]

但一般

\[
T_a^{\mathrm T} S T_a \neq I.
\]

说明活性内部仍然携带真正的非正交自由度。

令

\[
A_{aa} = T_a^{\mathrm T} S T_a.
\]

定义

\[
Q_a = T_a A_{aa}^{-1/2},
\qquad
L_a = A_{aa}^{1/2}.
\]

于是

\[
T_a = Q_a L_a,
\]

并满足

\[
Q_a^{\mathrm T} S Q_a = I,
\qquad
Q_i^{\mathrm T} S Q_a = 0.
\]

这一步给出一个关键结论：

1. \(Q_a\) 是正交 frame，可以进入旋转部分；
2. \(L_a\) 是 active 内部非正交 shape，不能整体并入旋转部分。

所以 active-active 自由度必须拆成两部分：

1. frame 变化；
2. shape 变化。

---

## 6. block 内部正交 frame 与部分旋转

在一个 block 的 union support 上，再构造一个 \(S_b\)-正交虚轨道补空间

\[
Q_v \in \mathbb{R}^{N_b \times n_v},
\]

满足

\[
Q_v^{\mathrm T} S_b Q_v = I,
\qquad
Q_i^{\mathrm T} S_b Q_v = 0,
\qquad
Q_a^{\mathrm T} S_b Q_v = 0.
\]

则 block 内部得到一个正交 frame

\[
Q = [\,Q_i,\; Q_a,\; Q_v\,],
\qquad
Q^{\mathrm T} S_b Q = I.
\]

现在可以旋转的自由度是：

\[
K_{ia} \in \mathbb{R}^{n_a \times n_i},
\qquad
K_{iv} \in \mathbb{R}^{n_v \times n_i},
\qquad
K_{av} \in \mathbb{R}^{n_v \times n_a}.
\]

把它们组装成反对称生成元

\[
\Omega =
\begin{bmatrix}
0              & -K_{ia}^{\mathrm T} & -K_{iv}^{\mathrm T} \\
K_{ia}         & 0                   & -K_{av}^{\mathrm T} \\
K_{iv}         & K_{av}              & 0
\end{bmatrix},
\qquad
\Omega^{\mathrm T} = -\Omega.
\]

于是 block 内部可以做有限步旋转

\[
Q^{+} = Q \exp(\Omega),
\]

或更适合实现的 Cayley retraction

\[
Q^{+}
=
Q
\left(I - \frac{1}{2}\Omega\right)^{-1}
\left(I + \frac{1}{2}\Omega\right).
\]

并严格保持

\[
{Q^{+}}^{\mathrm T} S_b Q^{+} = I.
\]

因此，以下三类耦合可以部分改为旋转语义：

1. inactive-active；
2. inactive-virtual；
3. active-virtual。

但 active-active 不能整体并入 \(\Omega\)，因为其非正交 shape 仍在 \(L_a\) 中。

---

## 7. sparse-support 的外部流形

对每个轨道 \(p\)，设它的显式 AO support 为

\[
J_p = \{\mu_1,\dots,\mu_{m_p}\}.
\]

设当前存储的 sparse 系数为

\[
x_p \in \mathbb{R}^{m_p}.
\]

令 \(B_p\) 为把 local support 嵌入 full AO 的选择矩阵，则 local overlap 为

\[
S_p = B_p^{\mathrm T} S B_p.
\]

局部归一化后的轨道为

\[
q_p = \frac{x_p}{\rho_p},
\qquad
\rho_p = \sqrt{x_p^{\mathrm T} S_p x_p}.
\]

于是

\[
q_p^{\mathrm T} S_p q_p = 1.
\]

这说明 sparse-support 的外部几何对象是

\[
\mathcal M_{\mathrm{sparse}}
=
\prod_{p=1}^{n_{\mathrm{orb}}}
\left\{
q \in \mathbb{R}^{m_p}
\mid
q^{\mathrm T} S_p q = 1
\right\}.
\]

它只约束每个轨道自己的 local normalization，不约束轨道间正交。

---

## 8. 外部 fixed-support 为什么需要局部几何回退

当前 sparse 有限步是

\[
x_p^{+} = x_p + \alpha d_p.
\]

这保持 support，但不保持

\[
{x_p^{+}}^{\mathrm T} S_p x_p^{+} = 1.
\]

所以正确的 sparse 外部 finite-step map 应该是：

1. 先把 raw step 投到 local normalization manifold 的切空间；
2. 再在同一个 support 上做 retraction。

在 \(q_p\) 处的切空间是

\[
T_{q_p}\mathcal M_p
=
\left\{
\eta_p \mid q_p^{\mathrm T} S_p \eta_p = 0
\right\}.
\]

切空间投影为

\[
\Pi_p = I - q_p q_p^{\mathrm T} S_p.
\]

如果已有一个 local raw step \(d_p\)，则对应的 tangent step 可以写成

\[
\eta_p
=
\frac{1}{\rho_p}
\left(
d_p - q_p(q_p^{\mathrm T} S_p d_p)
\right)
=
\frac{1}{\rho_p}\Pi_p d_p.
\]

然后做 normalized-add retraction：

\[
q_p^{+}
=
\frac{q_p + \alpha \eta_p}
{\sqrt{
\left(q_p + \alpha \eta_p\right)^{\mathrm T}
S_p
\left(q_p + \alpha \eta_p\right)
}}.
\]

这一步：

1. 保持同一组 sparse slots；
2. 保持 exact local normalization；
3. 与当前 step 在一阶上一致。

---

## 9. 统一后的 sparse finite-step 结构

因此 sparse-support 下合理的 finite-step 不是单层结构，而是两层：

\[
(Q, L_a)
\xrightarrow{\text{internal partial rotation + shape update}}
(Q^{+}, L_a^{+})
\xrightarrow{\text{sparse write-back}}
\{x_p^{+}\}.
\]

其中第一层是

\[
Q^{+}
=
Q
\left(I - \frac{1}{2}\Omega\right)^{-1}
\left(I + \frac{1}{2}\Omega\right),
\qquad
L_a^{+} = L_a + \Delta L_a.
\]

第二层不是全局旋转回写，而是逐轨道 fixed-support 回写。

---

## 10. 回写的两种方案

### 10.1 完整方案：受约束局部拟合

设内部更新后第 \(p\) 个轨道在 union support 上的目标向量为

\[
c_p^{\mathrm{int},+}.
\]

外部 sparse 回写应解

\[
\min_{x_p^{+}}
\left\|
B_p x_p^{+} - c_p^{\mathrm{int},+}
\right\|_{S_b}^2
\]

subject to

\[
{x_p^{+}}^{\mathrm T} S_p x_p^{+} = 1.
\]

这个方案最完整，但实现复杂。

### 10.2 第一阶段推荐方案：局部几何回退

先不解完整约束最小二乘，而是把内部更新诱导出的变化转成 local raw step

\[
d_p^{\mathrm{rot}}
=
B_p^{\mathrm T}
\left(
c_p^{\mathrm{int},+} - c_p^{\mathrm{int}}
\right),
\]

再做

\[
\eta_p = \Pi_p d_p^{\mathrm{rot}},
\qquad
q_p^{+} = R_p(q_p,\alpha\eta_p).
\]

这是更适合第一步落地的版本。

---

## 11. 代码落实顺序

下面给出按风险和收益排序的实现方案。

### 11.1 第一阶段：只替换 sparse `retract_step`

目标：先把当前“support 内加法更新”替换成“support 内几何回退”。

这一阶段不改 reduced chart，不改 HVP，不改梯度公式。

建议只改：

1. `src/vb/orbital/nonredundant_orbital_space.hpp`
2. `src/vb/orbital/nonredundant_orbital_space.cpp`

具体做法：

1. 保留当前 sparse branch 的 `local_step` 构造不变；
2. 对每个 `projector.flat_indices` 对应的轨道 support：
   构造 local \(S_p\)；
3. 从 `orbital_value_table` 取出当前 local 系数 \(x_p\)；
4. 计算归一化后的 \(q_p\)；
5. 用上面的切空间投影公式构造 \(\eta_p\)；
6. 用 normalized-add retraction 得到 \(q_p^{+}\)；
7. 把 \(q_p^{+}\) 写回相同 sparse slots。

当前代码已经按这个阶段落地：

1. `OrbitalProjector` 保存 `block_rows`，把每个 sparse slot 映射回 block-local AO 行号；
2. `build_local_sparse_overlap_metric(...)` 按 `block_rows` 从 block overlap 抽取 \(S_p\)；
3. `retract_local_sparse_normalized_step(...)` 在同一 support 上执行
   \(x_p \to q_p \to \eta_p \to q_p^{+}\)；
4. sparse `retract_step()` 不再把旧的 `local_step` 直接写回 `orbital_value_table`，
   而是先拿内部 mixed-chart finite-step 生成的物理 trial occupied 列，再做
   support-local geometric retraction；
5. dense full-support 分支与 sparse 分支现在共用同一 internal mixed-chart
   finite-step 源头。

这一阶段的优点：

1. 改动最小；
2. 不触碰 HVP / gradient 主链；
3. 能直接验证“几何回退”本身对 TN 步长稳定性的影响。

### 11.2 第二阶段：为 sparse blocks 引入内部 mixed chart

目标：不仅 finite step 几何化，而且 block 内真正利用已构造的正交子空间。

这一阶段主要还是改：

1. `src/vb/orbital/nonredundant_orbital_space.hpp`
2. `src/vb/orbital/nonredundant_orbital_space.cpp`

并可能需要少量复用：

3. `src/vb/orbital/active_space_orbital_preparer.cpp`

要做的事情：

1. 对 sparse blocks 也显式构造 block union-support 上的内部 frame
   \((Q_i,Q_a,Q_v,L_a)\)；
2. 把当前的 inactive-active / occupied-virtual 方向重解释为
   \(K_{ia},K_{iv},K_{av}\)；
3. 把 active-active 中真正对应 shape 的部分重解释为 \(\Delta L_a\)；
4. 在 `expand_block_rotation_directions()` 或新的 helper 中显式输出内部 mixed-chart 变量；
5. 在 sparse `retract_step()` 中先做内部更新，再做外部 support-local retraction。

这一阶段会真正进入“部分旋转”。

当前代码已经把 sparse-support 主链切到这一阶段：

1. 所有 block（不只 dense full-support / OEO）都会统一构造
   `Q_i`、`Q_a`、`L_a` 和内部 `Q_v` cache；
2. sparse 的 `project_block_candidate_overlap()` /
   `accumulate_block_candidate_combination()` 已经改为消费这套
   masked mixed-chart 基底；
3. sparse 的 `apply_block_candidate_metric()` /
   `solve_block_candidate_metric()` 现在作用在同一 mixed chart 上，
   `candidate_metric_diagonal` 也切到了对应的 sparse mixed-chart 对角近似；
4. sparse 的 `retract_step()` 现在先通过
   `build_mixed_chart_trial_occupied_orbitals(...)` 在内部 mixed chart 上生成
   block union-support 的 trial occupied block，再 restriction 到每个轨道的
   fixed support 做 per-orbital local geometric retraction；
5. 因此，当前 sparse-support 的 tangent map / metric / finite-step map
   已经对齐到同一 `(Q_i,Q_a,Q_v,L_a)` 语义，只是外部存储仍然保持
   fixed-support sparse coefficient layout。

### 11.3 第三阶段：重写 sparse reduced chart

目标：让 reduced coordinate 本身就更接近 mixed chart，而不是只在 finite step 上做后处理。

更彻底的版本还会继续影响：

1. `project_vector`
2. `project_gradient`
3. `apply_block_candidate_metric`
4. `solve_block_candidate_metric`
5. 以及 secant / transport 语义

这一步风险最大，不建议一开始做。

---

## 12. 当前状态与下一步

目前 finite-step 主链已经完成：

1. `retract_step()` 的 finite-step 源头已经统一为 internal mixed chart；
2. sparse-support 不再走旧的 raw additive finite-step；
3. dense / sparse 两条 finite-step 路径现在只在最后的 write-back 方式上不同。

当前更值得继续做的事情不是再改 finite-step 语义，而是：

1. 继续检查 `expand_retract_input_tangent()` 是否始终保持为新 finite retraction
   的零步导数；
2. 在更多代表性 sparse-support 体系上做数值诊断，而不是只看某一个体系；
3. 后续再决定是否要把 transport / secant / trust-region 的局部策略也一起
   调整到 mixed-chart 语义。

---

## 13. 不建议的做法

有三件事不建议直接做。

### 13.1 不要把全部 sparse 自由度塞进一个全局 Cayley

这会直接破坏 per-orbital fixed support。

### 13.2 不要把 active-active 全部强行改成反对称旋转

这会错误冻结掉 active 内部真正的非正交 shape。

### 13.3 不要一开始就改 HVP / gradient 公式

第一阶段只改 finite-step map 就足够。  
否则很难判断收益到底来自几何回退，还是来自别的公式变化。

---

## 14. 验证标准

第一阶段实现后，至少要检查以下性质。

### 14.1 结构约束

1. 每个轨道的 support 不变；
2. 没有新非零泄露到 support 外；
3. `enforce_strict_sparse_orbital_support` 不再承担主要“修正”职责。

### 14.2 局部几何约束

对每个轨道 \(p\)，应有

\[
{x_p^{+}}^{\mathrm T} S_p x_p^{+} = 1
\]

在数值误差范围内成立。

### 14.3 一阶一致性

当 \(\alpha \to 0\) 时，应满足

\[
x_p^{+} = x_p + \alpha \eta_p + O(\alpha^2).
\]

### 14.4 优化器行为

重点看：

1. TN 拒步是否减少；
2. trust radius 是否更少无谓收缩；
3. 相同步长策略下总迭代数是否下降；
4. 是否不再出现明显“线性更新后再被 normalize 拉回”的震荡。

### 14.5 当前已完成的数值检查

当前代码已经完成两类直接诊断：

1. `F2` 本地 `check_exact_ctx_hvp`：
   `analytic_retract_input_direction_max_abs_diff = 4.6851161839e-11`；
2. `241_VBSCF` 在 `6526Y` 节点的 `check_exact_ctx_hvp`：
   `analytic_retract_input_direction_max_abs_diff = 6.31567124987e-11`；
3. 两个体系的 `plus_support_changed_orbital_count` /
   `minus_support_changed_orbital_count` 都是 `0`。

这说明：

1. 新的 finite retraction 在零步处仍与当前 analytic tangent map 一阶一致；
2. sparse-support 没有发生 support 泄漏；
3. `retract_input_direction_max_abs_diff` 现在显著非零，正是因为 finite-step
   已经不再等于旧的 raw additive packed direction。

---

## 15. 最终统一结论

当前 sparse-support VBSCF 的正确方向不是

\[
\text{全局 sparse orbital rotation},
\]

而是

\[
\text{block 内部 partial rotation}
\;+\;
\text{active nonorthogonal shape update}
\;+\;
\text{per-orbital sparse-support geometric retraction}.
\]

对应到代码上，最合理的落地顺序是：

1. 先把 sparse `retract_step` 从“加法更新”改成“局部几何回退”；
2. 再把 sparse blocks 的内部 chart 逐步升级为
   \((Q_i,Q_a,Q_v,L_a)\) mixed chart；
3. 最后如果确实值得，再去重写 reduced chart、transport 和 secant 语义。

这样做，既不丢掉 VBSCF 的非正交自由度，也能真正把已经构造出来的正交子空间用起来。
