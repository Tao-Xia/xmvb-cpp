# Current-Code Derivation of Linear-Add Orbital Updates and the Mixed Chart

本文只使用两类依据：

1. `docs/VBSCF_Hessian.pdf` 中的 VBSCF Hessian / Newton-Raphson 公式。
2. 当前 C++ 实现中的实际数据流和函数语义。

已有同主题 markdown 文档可能记录了旧实现或中间判断，这里不把它们作为结论来源。

## 1. 代码中的对象和表示

当前优化器提交给目标函数的 stored representative 是 `OrbitalPreparationInput::orbital_value_table` 中的轨道系数。`SparseOrbitalParameterView` 只是在这个表上选择可微分的 packed 系数。目标函数每次评价时通过 `ActiveSpaceOrbitalPreparer::prepare()` 重建辅助轨道表示；后续 Hamiltonian/overlap、active-space integrals、RDM、gradient/HVP 都主要在 auxiliary-orbital 表示中计算。

设 AO overlap 为 $S$，物理占据轨道为

$$
C = [C_i, C_a],
$$

其中 $C_i$ 是 inactive doubly occupied 轨道，$C_a$ 是 active 轨道。当前 preparer 首先对输入轨道按各自支撑归一化，然后形成物理矩阵 `normalized_orbital_matrix`。

inactive overlap 和 projector 为

$$
M_i = C_i^T S C_i,
$$

$$
P_i = C_i M_i^{-1} C_i^T S.
$$

代码中的 `inactive_density_matrix` 是 $C_iM_i^{-1}C_i^T$，`occupied_space_projector` 是 $I-C_iM_i^{-1}C_i^TS$。因此辅助活性轨道为

$$
T_a = (I-P_i) C_a.
$$

这就是当前的三层状态：

1. stored/physical representative $[C_i,C_a]$：`orbital_value_table` 写回和 canonicalization 使用的轨道代表。
2. 辅助活性轨道 $T_a=(I-P_i)C_a$：用于构造 auxiliary occupied block 和 active-space 积分，是能量/梯度/HVP 计算的核心表示。
3. mixed-chart $[Q_i,Q_a,Q_v,L_a]$：当前 nonredundant optimizer 使用的内部局部坐标，也可以理解为高效构造切向量和预条件的计算表示。

这三层不是三个不同的物理波函数；它们是同一个 accepted point 的不同表示。不能把“PDF 使用 linear-add replacement”简单理解成“所有计算都在 stored physical coefficients 上直接完成”。PDF 的程序步骤同样先 prepare auxiliary orbitals，再在 auxiliary-orbital 表示中计算 Hamiltonian/overlap/RDM/Hessian。

当前 NROS block 里还同时缓存了三种不同来源的 occupied block，后续代码阅读必须分开：

1. `block_raw_occupied_orbitals` 来自 `orbital_value_table`，是 stored/raw 写回坐标。
2. `block_reference_occupied_orbitals` 来自 preparer 的 `physical_orbital_frame.normalized_orbital_matrix`，是 normalized physical representative，也是 mixed finite chart 的 zero-step anchor。
3. `block_occupied_orbitals` 来自 `auxiliary_orbital_matrix.leftCols(n_occ)`，active 列已经是 $T_a=(I-P_i)C_a$；virtual complement 当前由这个 auxiliary occupied span 构造。

因此当前 direct-linear family 的 finite write-back 是在 stored/raw coefficient slots 中做 column-local retraction；exact HVP 随后通过 `build_dense_orbital_tangent_context()` 再一次线性化 preparer 的列归一化，得到 $\delta C_i$、$\delta C_a$，再进入 $\delta P_i$ 和 $\delta T_a$。这个设计可以和 auxiliary-aware linear-add 相容，但必须把 raw stored tangent、normalized physical tangent、auxiliary tangent 的缩放和 adjoint 边界写清楚。

## 2. PDF 中的 Hessian 变量是 linear-add

`VBSCF_Hessian.pdf` 对轨道替换矩阵使用

$$
\phi_q^{new} = \sum_p R_{pq}\phi_p^{old}.
$$

在 Newton-Raphson 部分，轨道更新被写成

$$
\phi^{new} = \phi^{old} + x,
$$

对应二阶模型

$$
E(\phi+x) = E(\phi) + \xi^T x + \frac{1}{2}x^T\Omega x.
$$

Newton 方程是

$$
\Omega x + \xi = 0.
$$

这说明 Hessian 文章中的 Newton 变量不是正交旋转参数，也不是 mixed-chart 的 Cayley/SPD finite coordinate，而是非正交 VB orbital replacement / coefficient increment。该 increment 的 Hessian matrix elements 通过 auxiliary orbitals、icSEWF/icDEWF、RDM 和积分来计算。冗余变量和负曲率由 Hessian 对角化、移除小/负特征值处理，而不是通过把轨道更新强制改写成一个非线性正交旋转流形来处理。

因此，如果要复现文章意义上的二次收敛，主 Newton 对象应是 auxiliary-aware linear-add Hessian action，或者是它在一个严格等价线性子空间上的 HVP 投影。当前 C++ 目标是 matrix-free HVP，不是显式构造 PDF 中的完整 Hessian 矩阵。

## 3. HAO 和 OEO 都可以使用 linear-add

HAO/OEO 的差别是 AO 支撑模式，不是 orbital update 的数学类型。

对任意轨道 $p$，令 $E_p$ 是从全 AO 空间选出该轨道显式支撑的选择矩阵。局部系数和局部 overlap 为

$$
c_p = E_p^T C_p,
$$

$$
S_p = E_p^T S E_p.
$$

OEO/full-AO 情况下，$E_p=I$。HAO/sparse-support 情况下，$E_p$ 只选出该轨道的固定 AO 支撑。因此同一套 linear-add 公式同时适用于 HAO 和 OEO：

$$
c_p^+ = c_p + d_p.
$$

若要保持每列轨道的 accepted-point norm，可使用当前代码中的 support-local normalization retraction。定义

$$
\rho_p = (c_p^T S_p c_p)^{1/2},
$$

$$
n_p = c_p / \rho_p.
$$

raw increment $d_p$ 的归一化切向投影是

$$
J_p d_p = d_p - n_p(n_p^T S_p d_p).
$$

对应有限步可写为

$$
r_p(d_p)
=
\rho_p
\frac{c_p + J_p d_p}
{\left((c_p + J_p d_p)^T S_p (c_p + J_p d_p)\right)^{1/2}}.
$$

其 adjoint 是

$$
J_p^T w_p = w_p - (n_p^T w_p) S_p n_p.
$$

这正是当前 `retract_local_sparse_normalized_step()`、`linearize_local_sparse_normalized_step()` 和 `pull_back_local_sparse_normalized_step_adjoint()` 实现的数学对象。注意这里没有任何 HAO 专属假设；full-AO 只是在 $S_p=S$、$E_p=I$ 的特例。

## 4. 当前 direct-linear reduced chart

在一个 block 内，先忽略 stored/raw 与 normalized physical 的缩放差别，设物理占据轨道为

$$
C_{occ} = [C_i,C_a],
$$

并设 $V$ 是当前代码 `build_block_virtual_orbitals()` 构造的 block-local virtual complement。当前 direct-linear family 的物理 chart 可写成

$$
\delta C_i = C_a A + V K_i,
$$

$$
\delta C_a = -C_i A^T + C_a B + V K_a.
$$

这里 $A$ 是 active-inactive coupling，$B$ 是 active-active shape 块，当前代码把 $B$ 作为对称矩阵打包，$K_i$ 和 $K_a$ 是 occupied-virtual 方向。合起来就是

$$
\delta C_{occ}
=
C_{occ}
\begin{bmatrix}
0 & -A^T \\
A & B
\end{bmatrix}
+ V
\begin{bmatrix}
K_i & K_a
\end{bmatrix}.
$$

严格地说，上式右边第一项应理解为 $C_{occ}M+VK$，不是把 $C_{occ}$ 加到 tangent 中；即有限步是

$$
C_{occ}^+ = C_{occ} + \delta C_{occ}.
$$

如果再加 local normalization，有限步就是第 3 节的 $r_p(d_p)$ 逐轨道作用。没有 sparse support 时，仍然可以使用同一个公式，只是每列支撑是全 AO。

这给出了一类与 `VBSCF_Hessian.pdf` 变量一致的 linear-add tangent basis。它不要求后续积分和 HVP 都绕开 auxiliary orbitals；相反，HVP 仍应通过当前 auxiliary preparation 的方向导数来计算。

## 5. 当前 mixed-chart 的构造

当前 mixed-chart 先把 inactive 轨道正交化：

$$
Q_i = C_i M_i^{-1/2},
$$

$$
U_i = M_i^{1/2},
$$

所以

$$
C_i = Q_i U_i.
$$

active 轨道先分解出 inactive-gauge 分量：

$$
G_a = Q_i^T S C_a.
$$

active residual 是

$$
T_a = C_a - Q_iG_a.
$$

active residual overlap 为

$$
A_a = T_a^T S T_a.
$$

然后

$$
Q_a = T_a A_a^{-1/2},
$$

$$
L_a = A_a^{1/2}.
$$

于是

$$
C_a = Q_iG_a + Q_aL_a.
$$

代码中还使用 SPD chart

$$
H_a = \log(L_aL_a^T) = \log(A_a),
$$

并在有限步中用

$$
L_a^+ = \exp((H_a+\Delta H_a)/2).
$$

虚轨道 $Q_v$ 由 $[Q_i,Q_a]$ 的 $S$-orthonormal complement 构造。把 frame 写成

$$
Q = [Q_i,Q_a,Q_v].
$$

mixed-chart 的 frame 部分用 skew matrix

$$
\Omega =
\begin{bmatrix}
0 & -K_{ai}^T & -K_{vi}^T \\
K_{ai} & 0 & -K_{va}^T \\
K_{vi} & K_{va} & 0
\end{bmatrix},
$$

并通过 Cayley transform 更新

$$
Q^+ = Q(I-\Omega/2)^{-1}(I+\Omega/2).
$$

最后写回

$$
C_i^+ = Q_i^+ U_i,
$$

$$
C_a^+ = Q_i^+G_a + Q_a^+L_a^+.
$$

这是当前 `build_mixed_chart_trial_occupied_orbitals()` 的有限步模型。

## 6. mixed-chart 的一阶 tangent

在 accepted point 的一阶，Cayley transform 满足

$$
Q^+ = Q + Q\Omega + O(\|\Omega\|^2).
$$

记

$$
\delta L_a = d\exp_{H_a/2}(\Delta H_a/2),
$$

即代码中的 `build_active_shape_frechet_tangent()`。则当前 mixed-chart 的物理 tangent 是

$$
\delta C_i
=
Q_aK_{ai}U_i
+ Q_vK_{vi}U_i.
$$

active 轨道的 tangent 是

$$
\delta C_a
=
Q_a\delta L_a
+ Q_aK_{ai}G_a
- Q_iK_{ai}^TL_a
+ Q_vK_{va}L_a
+ Q_vK_{vi}G_a.
$$

这正是当前 `build_mixed_chart_physical_occupied_tangent()` 实现的公式。

这个 tangent 本身没有问题：它是 mixed finite map 的一阶导数，也可以作为 auxiliary-aware linear-add HVP 的一组切向量 $B$。问题只在于，如果优化器把同一组坐标解释为完整的 nonlinear mixed finite chart，而 HVP 只使用一阶 $J^TH_yJ$，那么 finite model 和 Hessian action 不再是同一个 Newton 模型。

## 7. 任意内部 chart 下的 Hessian 链式法则

令 stored packed 轨道代表为 $y$，能量为 $E(y)$。这里 $E(y)$ 指当前代码实际评价的目标函数：先由 `SparseOrbitalParameterView::unpack()` 写回 `orbital_value_table`，再经 `ActiveSpaceOrbitalPreparer::prepare()` 做列归一化、inactive projector、active auxiliary rebuild，最后进入 VBSCF energy/gradient/HVP。对任意内部坐标 $\theta$，有限步写成

$$
y(\theta) = R(\theta),
$$

其中

$$
R(0)=y_0,
$$

$$
J = R'(0).
$$

reduced gradient 是

$$
g_\theta = J^T g_y.
$$

若把该内部 chart 自身作为 Newton 变量，reduced Hessian action 对应

$$
H_\theta
=
J^T H_y J
+
\mathcal{G}_R,
$$

其中 retraction geometry 项为

$$
(\mathcal{G}_R)_{\alpha\beta}
=
g_y^T
\frac{\partial^2 R(0)}
{\partial\theta_\alpha \partial\theta_\beta}.
$$

如果 $R$ 是 linear-add tangent chart，

$$
R(\theta)=y_0+B\theta,
$$

则

$$
R''(0)=0,
$$

所以

$$
H_\theta = B^T H_y B.
$$

这就是 Hessian 文章中的 Newton 模型在 HVP 形式下的 reduced action。

如果 $R$ 是 mixed-chart Cayley + SPD exponential + sparse local normalization，则 $R''(0)$ 一般不为零。此时只使用 $J^TH_yJ$ 不等于 mixed finite chart 自身的 Newton-HVP action。接近驻点时 $g_y$ 很小，$\mathcal{G}_R$ 项会变小；但在 MnF2 这类难体系的早期或中期，忽略该项会让方法表现得不像 Newton 法。

## 8. 当前 exact-ctx HVP 的代码含义

从当前调用结构看，`ExactOrbitalSecondOrderOperator::apply_reduced()` 对 reduced direction $p$ 做的是：

1. 用 `expand_retract_input_tangent()` 得到 accepted-point physical tangent $Jp$。
2. 在 physical orbital-preparation / exact-ctx operator 中计算响应。
3. 用 `project_reduced_gradient()` 或 sparse-linear adjoint 把 packed response 拉回 reduced space。

也就是主结构为

$$
p \mapsto J^T H_y Jp.
$$

这对 auxiliary-aware linear-add tangent chart 是正确的 Newton HVP action；对 nonlinear mixed finite chart，则还缺少 $\mathcal{G}_R p$ 这类 finite-retraction 二阶几何 action，除非另有专门补偿。当前代码中 `update_chart_curvature_correction()` 只针对 cheap block curvature diagonal 添加 active-shape chart curvature correction，不等价于 mixed finite chart 自身的二阶 HVP action。

因此当前 TNHVP 不像二次收敛，并不能否定 linear-add，也不能简单证明 mixed 计算框架错。更准确地说，它说明当前默认 mixed finite chart 与 auxiliary-aware linear-add HVP action 没有完全组成同一个 Newton-HVP 模型。

## 9. mixed-chart 设计是否合理

结论要分两层。

作为 orbital preparation / energy evaluation 的三层状态，设计是合理的：

1. $[C_i,C_a]$ 是物理 VB 轨道。
2. $T_a=(I-P_i)C_a$ 消除了 active 对 inactive span 的投影，便于构造 auxiliary occupied block。
3. $T_a=Q_aL_a$ 把 active 子空间取向和 active overlap shape 分开，便于 active-space one-/two-electron integral 的构造、重构和诊断。

作为 Newton optimizer 的 finite chart，mixed-chart 需要更谨慎地区分“一阶切向量基”和“非线性更新流形”：

1. PDF 的 Hessian 是对 nonorthogonal orbital replacement / linear-add 变量构造的，但矩阵元计算使用 auxiliary orbitals。
2. mixed-chart 的 Cayley 和 SPD exponential 引入了非零 $R''(0)$。
3. sparse-support 写回又叠加了 support restriction 和 local normalization retraction。
4. 如果 mixed-chart 只提供 accepted-point tangent $B=J$，则 $J^TH_yJ$ 可以是合法的 projected linear-add HVP。
5. 如果 mixed-chart 同时定义有限步 $R(\theta)$，则只用 $J^TH_yJ$ 不是 mixed finite chart 自身的 $H_\theta p$。

因此 mixed-chart 可以作为 auxiliary preparation 的自然表示、切向量基、预条件、诊断或稳定化层。只有当它被解释成 finite Newton chart 时，才必须同步实现链式法则中的 retraction-geometry HVP action。否则，对于 MnF2 这种 open-shell、near-degenerate、sparse-support 敏感体系，外层行为很容易退化成 trust-region damped first-order 方法。

## 10. 当前实现里的具体不一致点

### 10.1 修复前有两条不同的 linear-add 相关路径

修复前代码里必须区分两件事：

1. `NonredundantOrbitalSpace` 主路径，由 `XMVB_CPP_NONREDUNDANT_SPARSE_LINEAR_RETRACTION` 选择 mixed 或 direct-linear finite chart。它仍调用统一的 `solve_exact_ctx_truncated_newton_step()`、`retract_step()`、`expand_retract_input_tangent()`、`project_reduced_gradient()`、metric/preconditioner 和 accepted-point 后重建逻辑。
2. `XMVB_CPP_SPARSE_LINEAR_HVP=1` 打开的全局 packed coefficient-space 旁路。它绕开 `solve_exact_ctx_truncated_newton_step()`，在 `run_exact_ctx_minimal_truncated_newton()` 内另写了一套 L-BFGS 预条件 CG、trust radius、trial build 和 HVP pullback。

第一条才是应当继续发展的 TNHVP 主路径。它在 override 打开后已经让 OEO/full-support block 不再自动走 dense mixed projector，因此从代码方向上承认了“linear-add 不只适用于 HAO”。第二条只是一个 packed 旁路，不能作为 linear-add 数学正确性的主要证据。本次修复已删除第二条旁路，当前生产 TNHVP 只保留 NROS reduced HVP 主路径。

全局 packed 旁路求解的近似模型更接近

$$
s \mapsto J_{norm}^T H_y J_{norm}s,
$$

其中 $s$ 是 packed differentiable coefficient 方向，而不是 NROS 去冗余后的 block candidate/reduced 方向。这个空间只处理了逐列 local normalization 的径向 gauge，没有等价于 PDF 中通过 Hessian 小/负特征值处理后的 nonredundant Newton 子空间。

### 10.2 packed 旁路接受一步后没有重建 accepted-point chart

`apply_packed_linear_tangent()` 和 `apply_packed_linear_tangent_adjoint()` 使用 `NonredundantOrbitalSpace::block_bases()` 中缓存的 accepted-point 局部轨道、overlap block、support projector 和 local normalization Jacobian。也就是说 $J_{norm}$ 依赖当前 accepted point。

但 `run_exact_ctx_minimal_truncated_newton()` 的 packed sparse-linear 分支在接受 trial 后只更新 `current_parameters`、`current_gradient`、`sparse_reduced_gradient` 和 `energy`。它没有像正常 NROS 分支那样重建

$$
\texttt{current\_space}=\texttt{build\_exact\_ctx\_nonredundant\_space(...)}.
$$

下一轮 HVP operator 使用的是新 `current_input` 加旧 `current_space`。于是方向展开和 adjoint 仍按旧点的 $J_{norm}$，而 orbital-preparation directional derivative 和能量评价已经在新点上。这会直接破坏 Newton 模型：

$$
J(y_k)^T H(y_{k+1}) J(y_k)
\ne
J(y_{k+1})^T H(y_{k+1}) J(y_{k+1}).
$$

这类 accepted-point 状态错位足以解释“完全不像二次收敛”的行为。

### 10.3 packed 旁路的 L-BFGS 预条件没有修正变量空间

修复前 packed 旁路新增了 `prev_params`、`prev_grad`、`lbfgs_s`、`lbfgs_y` 和 two-loop 预条件。这个补丁可能改善某些一阶行为，但没有让模型变成 Newton 模型。

原因是该旁路的 CG 变量是 raw packed 方向 $s$，HVP 使用的是 $J_{norm}s$，gradient 使用的是

$$
g_s = J_{norm}^T g_y.
$$

因此 secant pair 若写在 raw packed 坐标中，位移应与 $s$ 同坐标。该旁路却使用

$$
s_{LBFGS}=y_{k+1}^{packed}-y_k^{packed}=J_{norm}(y_k)s,
$$

而 covector 差仍是旧 `current_space` 下的 $J_{norm}(y_k)^T(g_{y,k+1}-g_{y,k})$。这把 realized tangent displacement 和 raw-coordinate covector 混在同一个 L-BFGS history 中；accepted point 后又没有重建或 transport chart，所以 secant condition 本身不自洽。

同一个问题还出现在 trust radius 和 step diagnostics 中：`step.reduced_step` 被当作 sparse tangent，但 trial 实际使用的是 `apply_packed_linear_tangent(current_space, step.reduced_step)`。也就是说记录的方向范数和实际写入参数的方向不是同一对象。

### 10.4 packed 旁路的 reduced size 和 metadata 混用 NROS projection

packed sparse-linear 分支的有效梯度 size 是 packed differentiable coefficient 数。但同一个循环里仍保留

$$
g_{NROS}=\texttt{current\_space.project\_gradient(current\_gradient).reduced\_gradient},
$$

并在 accepted-step metadata 中使用

$$
-g_{NROS}^T s
$$

记录 model linear decrease。一般情况下 $g_{NROS}$ 的 size 是 NROS reduced size，不等于 packed size。这至少会污染诊断和预测/实际下降统计；在带 Eigen runtime assert 的构建中还可能直接暴露维度错误。即使 release 构建未立刻崩溃，这也说明该旁路没有形成一个自洽的 reduced coordinate。

### 10.5 outer-response 的 pure-Delta-L 快路径可能解释错坐标

`ExactOrbitalSecondOrderOperator::apply_reduced_impl()` 在构造 outer-response 方向积分前会尝试 `try_build_pure_delta_la_directional_integrals(...)`。这个 fast path 通过 `nonredundant_space.expand_block_rotation_directions(reduced_direction)` 判断方向是否为 pure $\Delta L_a$。

在正常 NROS 主路径中，`reduced_direction` 的确是 NROS reduced vector，这个判断有意义。但在 `XMVB_CPP_SPARSE_LINEAR_HVP=1` 旁路中，同一个参数是 packed raw coefficient 方向。此时 fast path 用 NROS 规则解释 packed vector；如果误判成功，就会把 outer-response directional integrals 建在另一个方向上，而 direct-core orbital-preparation tangent 仍来自 $J_{norm}s$。即使大多数方向会 fallback，这个分支也说明 packed 旁路和 operator 内部优化假设不兼容。

### 10.6 full-AO/OEO 的现状：NROS 可走，packed helper 不可靠

`apply_packed_linear_tangent()` 和 adjoint 当前直接跳过 `uses_dense_full_support_projector` block。OEO/full-AO 在默认 `NonredundantSparseLinearRetractionMode::Disabled` 下会被标记为 dense full-support projector。因此，单独打开 `XMVB_CPP_SPARSE_LINEAR_HVP` 并不会自动给 OEO 提供 full-AO linear-add HVP。

另一方面，当前 NROS 构造在 `XMVB_CPP_NONREDUNDANT_SPARSE_LINEAR_RETRACTION` override 非 disabled 时，会让 OEO/full-support block 也进入 physical occupied-column direct-linear chart。这正好支持一个结论：linear-add 原理不应被实现成 sparse-only helper。HAO 和 OEO 的区别应只是每列支撑 $E_p$ 不同；OEO 是 $E_p=I$ 的特例。

### 10.7 dead helper 和注释残留增加了误导性

修复前 `apply_sparse_linear_tangent(const OrbitalPreparationInput&, ...)` 和 `apply_sparse_linear_tangent_adjoint(...)` 仍在 `nonredundant_orbital_space.cpp` 中，但生产调用点搜索不到它们。它们使用 full `orbital_value_table` size，并带有热路径 `stderr` 诊断。实际 packed 旁路使用的是 `apply_packed_linear_tangent()` 和 adjoint。

本次修复已删除这些 dead helper、packed helper 和 `NonredundantOrbitalSpace::block_bases()` 公开 accessor。文档和后续实现不能再把 full-table helper 的历史问题当作当前主路径问题。

### 10.8 mixed cache 的代码注释和实现语义不一致

`NonredundantOrbitalSpace` 构造函数向 `initialize_block_mixed_chart_cache()` 传入 `auxiliary_occupied_orbitals`，附近注释也说 mixed working frame 从 auxiliary occupied block 定义。但当前函数体实际没有使用这个参数，而是为了 `retract_step(0)` identity，从 `reference_occupied_orbitals` 也就是 normalized physical representative 构造 $Q_i$、$Q_a$、$L_a$。

这个选择本身可以是合理的：若 finite chart 要写回 stored physical representative，零步必须严格回到 $C_i,C_a$。但代码命名和注释会误导后续推导，让人以为 mixed state 是直接从 auxiliary active $T_a$ 开始的。应当把注释改成更精确的说法：HVP/energy 的计算对象是 auxiliary preparation；NROS mixed finite chart 的 zero-step representative 目前锚定在 physical $C_i,C_a$。

### 10.9 direct-linear 不应命名成只适用于 sparse

当前 `NonredundantSparseLinearRetractionMode` 名字带 sparse，但 linear-add tangent 原理上适用于 full-AO。代码中 override 被设置后，OEO full-support block 也可以走 occupied-column linear tangent；默认策略只是没有自动启用它。

更准确的命名应是 auxiliary-aware linear-add retraction/tangent，而不是 sparse-linear retraction。

### 10.10 240/241 的 wall-time 不否定 linear-add

240/241 中 direct-linear iteration 数更接近 Newton 行为时，wall-time 长主要来自 exact-ctx inner solve、trust strategy、metric/preconditioner 和 full response 策略仍不够好。它说明实现策略还不成熟，不说明 linear-add 方向错。

### 10.11 MnF2 失败更可能是模型/策略不完整

如果采用文章意义上的 auxiliary-aware linear-add Hessian action，并在 matrix-free HVP 框架中正确处理冗余/负曲率/level shift/trust region，MnF2 理论上不应需要上千个极小步。当前 MnF2 的表现说明现有 TNHVP 路径没有形成那个 Newton-HVP 模型，尤其是 reduced tangent basis、preconditioner、trust radius、accepted-point chart refresh 和 finite retraction geometry 之间仍不一致。

## 11. 推导后的实现方向

从数学上更干净的方向是：

1. 以 auxiliary-aware linear-add tangent 作为 Newton/HVP 主空间。
2. HAO 和 OEO 使用同一个 linear-add 框架；区别只在每列 $E_p$ 支撑选择。
3. local normalization 只作为 column-norm gauge retraction，并显式使用 $J_p$ 和 $J_p^T$。
4. exact HVP 使用 auxiliary preparation 的方向导数计算 linear-add Hessian action $H_y p$，reduced model 使用 $p \mapsto B^T H_y Bp$。
5. 冗余方向通过 metric/Hessian null-space、small eigenvalue removal、level shift 或 trust-region subproblem 处理，而不是依赖 nonlinear mixed finite chart 自动消除。
6. mixed-chart 保留为 preparation、diagnostic、tangent basis 和可选 preconditioner；不要同时把它定义为默认 Newton finite step，除非同步加入 $\mathcal{G}_R p$ 这类 chart-geometry HVP action。

这样得到的算法才与 `VBSCF_Hessian.pdf` 的二次收敛逻辑一致，也不会人为区分 HAO 和 OEO 的 orbital-update 原理。
