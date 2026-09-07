# Auxiliary-Aware Linear-Add TNHVP Implementation Plan

> **历史方案（已被物理 (U_p) 路径取代）**：mixed chart 不再是可选实现，
> 也不能用于解释当前 exact HVP；本文仅保留为推导历史。

本文档只基于当前代码状态、`docs/current_code_linear_add_mixed_chart_derivation.md` 的推导，以及 `docs/VBSCF_Hessian.pdf` 中的 Newton 变量和去冗余处理。目标不是实现 PDF 里的完整 Hessian 矩阵构造，也不是把 auxiliary-orbital 计算退化成纯 stored-coefficient 更新。更准确的目标是：把当前 TNHVP 的 matrix-free Hessian-vector action 放到与 PDF 一致的 auxiliary-aware linear-add 变量上，同时允许 mixed-chart 继续作为高效的计算表示、切空间基和预条件框架。旧文档中的策略结论不作为依据。

## 0. 边界：HVP 不是完整 Hessian 矩阵

`VBSCF_Hessian.pdf` 的实际实现路线是先构造完整 orbital Hessian 矩阵，再通过 Hessian 对角化移除 negative 和 small eigenvalues，最后在保留的子空间里解 Newton-Raphson 方程。它给我们的关键约束有两个：

1. Newton 变量是 orbital replacement / orbital increment，即

$$
\phi^{new}=\phi^{old}+x.
$$

2. 冗余变量不是通过 mixed-chart 的 Cayley/SPD 非线性坐标消失的，而是在线性 orbital-increment Hessian 中通过小特征值、负特征值、level shift 或 trust-region 子问题处理。

当前 C++ 目标是 TNHVP，即不显式构造完整 Hessian 矩阵，而实现同一 auxiliary-aware linear-add 变量上的 Hessian-vector product：

$$
p \mapsto B^T \Omega Bp.
$$

这里 $\Omega$ 表示 PDF 中 orbital replacement / increment 的 Hessian action。该 action 在原文程序里通过 auxiliary orbitals、auxiliary-orbital integrals、RDM 和 Hessian matrix elements 计算；在当前 C++ 中则通过 orbital-preparation directional derivative、active-space directional integrals 和 backpropagation 以 matrix-free 方式计算。$B$ 是当前代码选择的 nonredundant tangent basis，可以来自 mixed-chart 的一阶线性化。原文的 Hessian 对角化去冗余在 HVP 实现中的对应物，不是 mixed finite chart 本身，而应是 $B$ 的构造、accepted-point metric factorization、Krylov/trust-region 中对 near-null 和 negative curvature 的处理。

## 1. 当前代码结论

当前三层轨道状态本身是合理的：

1. stored/physical representative $[C_i,C_a]$ 是 `OrbitalPreparationInput::orbital_value_table` 写回和 canonicalization 使用的轨道代表。
2. 辅助活性轨道 $T_a=(I-P_i)C_a$ 是 `ActiveSpaceOrbitalPreparer` 为 active-space 积分构造的计算对象，也是 HVP 中方向导数的核心表示。
3. mixed-chart $[Q_i,Q_a,Q_v,L_a]$ 是 `NonredundantOrbitalSpace` 当前默认 reduced finite step 的内部坐标。

问题不在这三层状态本身，也不在“使用 auxiliary orbitals 计算”。PDF 原文同样先 prepare auxiliary orbitals，再在 auxiliary-orbital 表示中计算 Hamiltonian/overlap/RDM/Hessian。真正需要分清的是：mixed-chart 是不是只作为一阶 tangent basis 使用，还是被当成 nonlinear finite Newton coordinate。`ExactOrbitalSecondOrderOperator::apply_reduced()` 目前主要实现一个 reduced HVP：

$$
p \mapsto J^T H_y Jp,
$$

其中 $y$ 是 stored orbital representative，`ActiveSpaceOrbitalPreparer` 会从它生成 inactive density 和 active auxiliary orbitals。$J$ 是 `NonredundantOrbitalSpace::expand_retract_input_tangent()` 给出的 accepted-point tangent。如果 reduced coordinate 只表示 linear-add tangent basis $B$，则该式就是 PDF auxiliary-aware linear-add Hessian 在 reduced 方向上的 action。若同一个 reduced coordinate 同时被解释为 mixed Cayley/SPD/local-normalization finite chart，$R''(0)\ne0$，该 finite chart 自身的 reduced Hessian action 应包含

$$
H_\theta = J^T H_y J + g_y^T R''(0).
$$

当前 exact HVP 没有完整加入 $g_y^T R''(0)$ 这类 chart-geometry action，所以它不应被解释为 mixed finite chart 自身的完整 Newton action。更合理的选择有两条：一是把 mixed-chart 只作为 auxiliary-aware linear-add tangent basis，让 HVP action 是 $B^T \Omega Bp$；二是如果坚持把 Cayley/SPD finite chart 作为 Newton coordinate，就必须实现对应的 chart-geometry HVP action。本文档后续实施步骤采用第一条。

## 2. 必须修正的代码事实

`SparseOrbitalParameterView` 的 packed 向量只包含显式可微分系数；full `orbital_value_table` 包含每个轨道长度为 `n_basis_functions` 的 padded slot。任何 TNHVP 主路径都必须保持 packed/full 边界清楚。

修复前代码有两条不同的 linear-add 相关路径，不能混为一个算法：

1. `NonredundantOrbitalSpace` 主路径。它由 `XMVB_CPP_NONREDUNDANT_SPARSE_LINEAR_RETRACTION` 临时选择 mixed 或 direct-linear finite chart，仍复用 `solve_exact_ctx_truncated_newton_step()`、`retract_step()`、`expand_retract_input_tangent()`、`project_reduced_gradient()`、metric/preconditioner 和 accepted-point 后重建逻辑。当前 override 非 disabled 时，OEO/full-support block 也不会自动退回 dense mixed projector，这一点符合“linear-add 不只适用于 HAO”。
2. `XMVB_CPP_SPARSE_LINEAR_HVP` 全局旁路。它在 `exact_ctx_minimal_tn.cpp` 中另写 packed coefficient-space L-BFGS 预条件 CG、trust radius、trial build 和 predicted decrease；在 `ExactOrbitalSecondOrderOperator::apply_reduced_impl()` 中另写 `apply_packed_linear_tangent()` / adjoint HVP pullback。这个旁路不是 NROS 主路径。

本次修复已删除第 2 条全局旁路，生产 TNHVP 只保留 NROS reduced HVP 主路径。该旁路被删除的原因是：

1. `apply_packed_linear_tangent()` / adjoint 使用 `NonredundantOrbitalSpace::block_bases()` 中的 accepted-point local normalization Jacobian，因此必须随 accepted point 重建。
2. sparse-linear 分支接受一步后没有重建 `current_space`，下一轮使用旧点的 $J_{norm}$ 配新点的 energy/HVP chain。
3. 新增的 L-BFGS history 使用 `trial_parameters - prev_params` 作为 $s$，但 CG/HVP 的变量是 raw packed direction，trial 写入的是 $J_{norm}s$；secant pair 的坐标不自洽。
4. outer-response 的 pure-$\Delta L_a$ fast path 仍按 NROS reduced vector 解释 `reduced_direction`，与 packed raw direction 旁路不兼容。
5. full-table `apply_sparse_linear_tangent()` 和 `apply_sparse_linear_tangent_adjoint()` 已经不是生产调用点，但仍作为 dead helper 和 `stderr` 诊断残留在源码里，容易误导分析。

因此不能继续在全局 HVP 旁路上修补算法正确性。主线应进入 `NonredundantOrbitalSpace`，让 `retract_step()`、`expand_retract_input_tangent()`、`project_reduced_gradient()`、metric/preconditioner、trust-region 和 exact HVP pullback 共用同一个 auxiliary-aware linear-add tangent basis。

## 3. 目标行为

目标不是否定 auxiliary-orbital 计算，而是让 Newton/HVP 变量是 linear-add tangent。以 stored representative 写时，一个有限步可以表示成 occupied column 的 additive update：

$$
C_{occ}^+ = C_{occ} + \delta C_{occ}.
$$

HAO 和 OEO/full-AO 的差别只在每个轨道的 AO 支撑选择 $E_p$：

$$
c_p^+ = c_p + d_p.
$$

如果保留当前列归一化 gauge，则每列使用相同的 support-local normalization retraction：

$$
J_p d_p = d_p - n_p(n_p^T S_p d_p),
$$

$$
r_p(d_p)=
\rho_p
\frac{c_p+J_p d_p}
{\left((c_p+J_p d_p)^T S_p(c_p+J_p d_p)\right)^{1/2}}.
$$

full-AO 只是 $E_p=I$、$S_p=S$ 的特例。实现上不能把该路径命名或 gate 成只适用于 sparse/HAO。

对 auxiliary-active 表示，关键的一阶对象是

$$
T_a=(I-P_i)C_a,
$$

以及它的方向导数

$$
\delta T_a=(I-P_i)\delta C_a-\delta P_i C_a.
$$

当前 exact HVP 的核心计算实际消费的是 $\delta P_i$、$\delta T_a$、方向 active-space integrals 和 backpropagated covector。因此 mixed-chart 可以保留，只要它交给 HVP 的是一致的 accepted-point linear-add tangent，而不是把 nonlinear finite chart 的二阶几何项静默丢掉。

## 4. 实施阶段

### 4.1 重命名 chart/tangent 概念

当前入口：

- `NonredundantSparseLinearRetractionMode`
- `choose_nonredundant_sparse_linear_retraction_mode()`
- `nonredundant_sparse_linear_retraction_mode_name()`
- 环境变量 `XMVB_CPP_NONREDUNDANT_SPARSE_LINEAR_RETRACTION`
- 日志字段 `Sparse retraction`

实施步骤：

1. 新增或重命名为 `NonredundantLinearAddTangentMode` 或 `NonredundantAuxiliaryLinearAddMode`，避免暗示只能在 stored physical coefficients 上计算。
2. 保留旧环境变量作为兼容 alias，但新增主环境变量，例如 `XMVB_CPP_NONREDUNDANT_LINEAR_ADD_TANGENT`。
3. 日志字段改为 `Orbital tangent chart` 或 `Linear-add tangent`，取值使用 `mixed_tangent`、`linear_add`、`linear_add_one_sided` 等不会误导的名字。
4. 文档和注释中停止把该算法称为 `sparse-linear`。只有谈 packed sparse storage 或 HAO support 时才使用 sparse。
5. 将 `uses_dense_full_support_projector` 从“chart 类型”语义中拆出来。它只能表示 packed slot 覆盖了 block 所有 AO 行时可用的 gather/scatter 优化，不能决定是否使用 mixed 或 linear-add chart。

验收条件：

- 默认日志能看出当前 HVP tangent basis 是 `mixed_tangent` 还是 `linear_add`。
- 旧 env 仍能临时复现实验，但不会成为新代码的主名称。
- OEO/full-AO 日志不再显示成“非 sparse 所以不能 linear-add”。

### 4.2 把 auxiliary-aware linear-add HVP 作为 NROS 主路径

当前 `NonredundantOrbitalSpace` 已经有 direct-linear family：

- `accumulate_sparse_physical_block_candidate_combination()`
- `project_sparse_physical_block_candidate_overlap()`
- `accumulate_sparse_block_retraction_tangent_combination()`
- `retract_local_sparse_normalized_step()`
- `linearize_local_sparse_normalized_step()`
- `pull_back_local_sparse_normalized_step_adjoint()`

但该路径仍被 `sparse_linear_retraction_mode_` 命名和策略控制，而且默认 auto policy 对 OEO/full-AO 返回 disabled mixed chart。override 非 disabled 时，当前代码已经把 OEO/full-support block 留在 occupied-column direct-linear family；这部分应保留并正规化，而不是继续当作 sparse-only 实验分支。

实施步骤：

1. 在 `BlockBasis` 中加入明确 tangent/finite chart 类型，例如 `BlockTangentBasis::MixedLinearization`、`BlockTangentBasis::LinearAdd`，以及单独的 finite retraction 类型。不要再用 `uses_dense_full_support_projector` 隐式表达 chart。
2. 对 linear-add HVP 模式，HAO 和 OEO 都调用同一个 `accumulate_linear_add_block_candidate_combination()`。
3. 对 full-AO/OEO block，构造每列全 AO 支撑的 projector，使 local normalization helper 可以在 $S_p=S$ 上工作，而不是特殊绕到 dense mixed path。
4. `retract_step()` 的 finite update 必须声明清楚：若使用 linear-add finite step，则使用 $C + D a$ 后逐列 local normalization；若使用 mixed finite step，则 HVP 要么只作为 projected linear-add model 使用，要么补齐 chart-geometry action。
5. `expand_retract_input_tangent()` 在 linear-add HVP 模式下返回 accepted-point full `orbital_value_table` tangent，也就是 HVP 需要的 $Bp$；随后由 orbital preparation 的方向导数生成 $\delta P_i$ 和 $\delta T_a$。
6. `project_reduced_gradient()` 在 linear-add HVP 模式下使用同一个 tangent adjoint，把 packed orbital gradient 拉回 reduced chart。
7. 小/中 block 继续使用 accepted-point metric factorization 去掉 normalization gauge 和近冗余方向；这对应 HVP 版本里的去冗余，不是完整 Hessian 对角化。
8. 明确 raw stored、normalized physical、auxiliary occupied 三个边界：direct-linear finite write-back 可以发生在 stored/raw slots，但 HVP action 的数学 tangent 必须是经过 preparer normalization 后的 $\delta C$，并继续由 orbital-preparation directional result 生成 $\delta P_i$ 和 $\delta T_a$。

验收条件：

- HAO sparse-support block 和 OEO full-support block 走同一个数学路径。
- `retract_step(0)` 严格回到输入 stored coefficients。
- `expand_retract_input_tangent()` 和 `retract_step(eps p)` 的有限差分一致。
- 对 OEO/full-AO，override 打开后不再出现 dense mixed projector 跳过 linear-add helper 的行为。

### 4.3 统一 packed/full tangent API

修复前全局 helper：

- `apply_sparse_linear_tangent(const OrbitalPreparationInput&, const Eigen::VectorXd&)`
- `apply_sparse_linear_tangent_adjoint(const OrbitalPreparationInput&, const Eigen::VectorXd&)`
- `apply_packed_linear_tangent(const NonredundantOrbitalSpace&, const Eigen::VectorXd&)`
- `apply_packed_linear_tangent_adjoint(const NonredundantOrbitalSpace&, const Eigen::VectorXd&)`

前两个 full-table helper 没有生产调用点，使用 full table size，并在热路径中打印 `stderr` 诊断，不适合保留为主实现。后两个 packed helper 被 `XMVB_CPP_SPARSE_LINEAR_HVP` 旁路使用，但它们跳过 dense full-support block，且依赖 `NonredundantOrbitalSpace` 中的 accepted-point cache。本次修复已从源码中删除这些 helper 和相关公开声明。

后续如确实需要重新引入 ambient packed 诊断 helper，必须满足：

1. 不在 optimizer/HVP 主路径调用。
2. 放到 `NonredundantOrbitalSpace` 成员函数中，明确它们只在当前 accepted point 有效，并显式命名为 packed：
   - `apply_linear_add_tangent_packed(const Eigen::VectorXd& packed_direction)`
   - `apply_linear_add_tangent_adjoint_packed(const Eigen::VectorXd& packed_covector)`
3. packed helper 必须覆盖 HAO sparse-support 和 OEO/full-AO block；full-AO 不应被 `uses_dense_full_support_projector` 静默跳过。
4. 对 full table tangent 只在 `expand_retract_input_tangent()` 这一类接口返回，并由 `SparseOrbitalParameterView::gather_from_full()` 显式转换。
5. 所有 helper 开头检查输入 size，错误信息写清楚 expected packed size 或 full table size。
6. 不使用 `std::fprintf` 热路径诊断；需要长期诊断时使用已有日志/diagnostic target 方式。

验收条件：

- 全代码搜索中 `XMVB_CPP_SPARSE_LINEAR_HVP` 不再影响生产 optimizer。
- HVP 和 trial build 不再手写 scatter/gather 循环。
- 任意 `parameter_view.unpack()` 输入都是 packed size。
- 接受一步后所有 accepted-point tangent cache 都来自新 `objective.last_input()` 和新 gradient result。

### 4.4 删除 exact-ctx 全局 sparse-linear HVP 旁路

修复前旁路分布在：

- `src/vb/scf/exact_ctx_minimal_tn.cpp`
- `src/vb/scf/exact_orbital_second_order_operator.cpp`
- `src/vb/orbital/nonredundant_orbital_space.cpp`

本次已完成：

1. 从 `ExactCtxReducedHvpOperator::apply_reduced()` 删除 `exact_ctx_sparse_linear_hvp_enabled()` 分支。
2. 从 `run_exact_ctx_minimal_truncated_newton()` 删除 packed coefficient-space CG/L-BFGS 的专用实现，统一调用 `solve_exact_ctx_truncated_newton_step()`。
3. 删除 sparse-linear 分支专用状态：`sparse_reduced_gradient`、`sparse_reduced_size`、`prev_params`、`prev_grad`、`lbfgs_s`、`lbfgs_y`、专用 trust radius、专用 `step.reduced_step` 求解和 `current_parameters + packed_tangent` trial build。
4. 删除 `exact_ctx_sparse_linear_hvp_enabled()`。
5. 删除该旁路产生的 `[sparse-cg]`、`[spr-tangent]` 等 `stderr` 调试输出。
6. 删除 `apply_packed_linear_tangent()` / adjoint 和 full-table sparse-linear dead helper。

保留要求：

1. `NonredundantOrbitalSpace` 仍负责 finite retraction、tangent expand、adjoint projection、metric/preconditioner 和 accepted-point 后重建。
2. HVP operator 只接受 NROS reduced vector，并通过 `expand_retract_input_tangent()` 生成 orbital-preparation 方向导数。

验收条件：

- TNHVP 只有一条 reduced HVP 主调用链。
- trust-region、preconditioner、step_check、trial build 对 mixed tangent 和 linear-add tangent mode 复用同一套代码。
- `current_space` 和 `current_projection` 在每个 accepted point 后都统一重建，不存在 sparse-linear 特例。

### 4.5 修正 full-AO/OEO 支撑处理

当前 `orbital_input_uses_sparse_orbital_support()` 会把 `orbtyp=oeo` 判为非 sparse，默认 auto policy 选择 disabled mixed chart。但代码已在 override 非 disabled 时让 OEO/full-support block 进入 physical occupied-column chart。linear-add HVP 不应依赖 sparse-only gate，也不应把 OEO 的默认 mixed chart当成数学限制。

实施步骤：

1. 把 chart 选择从“是否 sparse support”改为“用户/策略选择哪个 finite chart”。
2. 对 OEO/full-AO，在 `build_block_orbital_data()` 或等价位置确保每个 occupied orbital 的 differentiable support 是全 AO。
3. linear-add HVP 模式下 `block_basis.uses_dense_full_support_projector` 不再表示 chart，只表示 packed slot 覆盖了 block 所有 AO 行时可用的高效 gather/scatter。
4. local normalization 的 full-AO 版本复用 `retract_local_sparse_normalized_step()`，输入 local size 等于 block/full AO size。
5. `project_sparse_physical_block_candidate_overlap()` 应改名为不带 sparse 的 `project_physical_block_candidate_overlap()`，并允许 dense/full projector。
6. `apply_packed_linear_tangent()` 这类 helper 若继续存在，不能再 `continue` 掉 full-support block；否则会与 NROS 主路径含义冲突。

验收条件：

- FeCl2/OEO 在 linear-add HVP 模式下不会因为 full-AO 被错误 gate 到 sparse-only 路径。
- full-AO path 不触发 sparse-only support 判断。

### 4.6 梯度、HVP 和 metric 一致性

linear-add reduced HVP model 应是

$$
p \mapsto B^T H_y Bp,
$$

若保留列归一化，则 $B$ 应包含逐列 $J_p$，adjoint 应包含 $J_p^T$。关键是 finite step、gradient projection、HVP tangent、HVP pullback、metric/preconditioner 全部使用同一个 $B$。

实施步骤：

1. 将 `project_impl()` 中 `apply_retraction_tangent_adjoint` 的判定改为依赖 block chart，而不是依赖 sparse/mixed helper。
2. `candidate_metric_diagonal`、exact metric factorization、`apply_block_candidate_metric()` 都基于 actual accepted-point tangent $B a$。
3. `apply_inverse_reduced_metric_preconditioner()` 对 exact-factorized block 继续认为 metric 已 whitened；对大 block 使用 actual metric diagonal，不再混用 mixed surrogate。
4. `update_chart_curvature_correction()` 只对 mixed SPD finite chart 使用；linear-add HVP model 下默认跳过 mixed finite chart curvature，因为主 linear-add map 的 $R''(0)=0$。列归一化 gauge 若保留，其二阶影响另列为 gauge correction，不与 active-shape mixed correction 混在一起。
5. HVP diagnostics 比较 `expand_step()` 与 `expand_retract_input_tangent()` 时，应明确一个是 raw candidate packed step，一个是 HVP tangent，不能用该差异判定 mixed 计算框架失败。
6. `build_exact_ctx_nonredundant_space()` 中 `initialize_block_mixed_chart_cache()` 的注释和参数要清理：当前 mixed finite chart 的 zero-step representative 锚定在 normalized physical $C_i,C_a$；auxiliary active $T_a$ 是 energy/HVP preparation 表示，不是该函数当前实际使用的构造输入。
7. `try_build_pure_delta_la_directional_integrals()` 只能消费 NROS reduced direction。删除 packed 旁路后该假设自然成立；若保留任何替代方向入口，必须显式标记坐标类型并禁止误用 fast path。
8. 对每个 chart mode 记录三种 norm：raw packed step norm、normalized-physical tangent norm、auxiliary active tangent norm。trust-region 只能绑定其中一个明确选择的 norm，不能在 step 求解、trial build 和 diagnostics 中混用。

验收条件：

- 对随机 reduced direction $p$，`project_reduced_gradient(Bp)` 与 metric application 一致。
- Adjoint identity 成立：$(Bp)^T w = p^T(B^T w)$。
- exact HVP 在 reduced 空间近似对称：$p^T H q \approx q^T H p$。

### 4.7 Trust-region 和 inner solve 策略

240/241 wall-time 长不能用来否定 linear-add，但必须修策略，否则实现仍不可用。当前风险点包括 raw coefficient norm、block metric/preconditioner、outer-response 选择和 PCG forcing。

实施步骤：

1. trust radius 用 tangent norm 或 metric norm 衡量，不要只用 raw reduced coefficient norm。
2. 对小/中 block 默认启用 exact accepted-point metric factorization，保证 PCG 看到接近 identity 的 chart metric。
3. 对大 block 使用实际 $B^T B$ diagonal 或 block solve，不要退化到纯 orbital-energy-gap diagonal。
4. 保留当前 `choose_truncated_newton_residual_inf_target()` 的 inexact Newton forcing 思路，但记录每步 CG residual、boundary、negative curvature 和实际/预测下降比。
5. 对 open-shell 难体系的 full outer-response 策略单独 gate，不要把 strategy profile 和 chart 正确性耦合。

验收条件：

- 241 不需要大量 rejected tiny steps。
- 240 wall-time 下降来自 inner solve 次数和 HVP 次数减少，而不是改松收敛。
- MnF2 不出现上百/上千步的小 trust-radius first-order 行为。

## 5. 必须新增的诊断和测试

### 5.1 Chart consistency diagnostic

建议新增或扩展现有 diagnostic target，至少检查：

1. `retract_step(0)` identity。
2. finite difference tangent：

$$
\frac{R(\epsilon p)-R(0)}{\epsilon} \approx Bp.
$$

3. adjoint identity：

$$
(Bp)^T w \approx p^T B^T w.
$$

4. packed/full size invariant：所有 packed API 输入输出都是 `SparseOrbitalParameterView::size()`，所有 full API 输入输出都是 `orbital_value_table.size()`。
5. mixed tangent、linear-add tangent、finite retraction type、support size、reduced size 打印到 diagnostic summary。

### 5.2 HVP diagnostic

对 240、241、FeCl2、MnF2 做：

1. reduced HVP symmetry check。
2. analytic HVP vs finite-difference gradient check。
3. `J^THJ` 与 finite objective quadratic model 的 predicted decrease check。
4. trial build 后 `parameter_view.unpack()` 不报 packed size 错。

F2 只能作为 smoke test，不能作为算法普遍性证据。

## 6. 推荐代码修改顺序

1. 先做纯重命名和日志清理，不改变默认行为。
2. 引入 chart enum，把 `uses_dense_full_support_projector` 从 chart 语义中拆出来。
3. 将 linear-add candidate/tangent/project/retract 函数改为 HAO/OEO 共享实现。
4. 删除或停用 `XMVB_CPP_SPARSE_LINEAR_HVP` 全局旁路，保证只有 NROS 主路径。
5. 接入 auxiliary-aware linear-add NROS HVP 模式，但先通过 env opt-in 运行。
6. 加 chart consistency diagnostic，并在 F2/241 上先验证尺寸和 adjoint。
7. 跑 240/241/FeCl2/MnF2 的 sbatch benchmark。
8. 只有大体系通过后，才考虑把该 HVP tangent mode 设为默认或按 system profile 自动选择。

## 7. 需要修改的主要文件

- `src/vb/orbital/nonredundant_orbital_space.hpp`
- `src/vb/orbital/nonredundant_orbital_space.cpp`
- `src/vb/orbital/sparse_orbital_parameter_view.hpp`
- `src/vb/orbital/sparse_orbital_parameter_view.cpp`
- `src/vb/scf/exact_ctx_minimal_tn.cpp`
- `src/vb/scf/exact_orbital_second_order_operator.cpp`
- `src/tools/run_cpp_vbscf.cpp`

如果新增 diagnostic target，还需要修改：

- `src/CMakeLists.txt`
- `src/tests/unit/` 或现有 diagnostic source 所在目录

实现时遵守 `docs/HPC_CPP_STYLE.md`：新增矩阵/向量语义使用 `Eigen::MatrixXd` / `Eigen::VectorXd`，不要引入新的 `std::vector<double>` 假 dense matrix，也不要新增热路径 `std::fprintf` 诊断。

## 8. SLURM 验收方案

构建：

```bash
cmake --build build --target run_cpp_vbscf -j8
```

本地 smoke 只检查可运行性：

```bash
OMP_NUM_THREADS=1 build/src/xmvb-cpp.exe src/test_molecule/F2.xmi --optimizer-backend lbfgspp
```

真正算法验收必须提交到计算节点，使用 240、241、FeCl2、MnF2：

```bash
CPUS_PER_TASK=32 OMP_NUM_THREADS=32 \
  BENCHMARK_DIR=benchmarks/linear_add_hvp_241_$(date +%Y%m%d_%H%M%S) \
  bash scripts/submit_vb_optimizer_benchmark_triplet.sh test/241_VBSCF.xmi

CPUS_PER_TASK=32 OMP_NUM_THREADS=32 \
  BENCHMARK_DIR=benchmarks/linear_add_hvp_240_$(date +%Y%m%d_%H%M%S) \
  bash scripts/submit_vb_optimizer_benchmark_triplet.sh test/240_tnhvp.xmi

CPUS_PER_TASK=32 OMP_NUM_THREADS=32 \
  BENCHMARK_DIR=benchmarks/linear_add_hvp_fecl2_$(date +%Y%m%d_%H%M%S) \
  bash scripts/submit_vb_optimizer_benchmark_triplet.sh test/FeCl2.xmi

CPUS_PER_TASK=32 OMP_NUM_THREADS=32 \
  BENCHMARK_DIR=benchmarks/linear_add_hvp_mnf2_$(date +%Y%m%d_%H%M%S) \
  bash scripts/submit_vb_optimizer_benchmark_triplet.sh test/MnF2.xmi
```

提交后用：

```bash
bash scripts/summarize_vb_optimizer_benchmark.sh <benchmark_dir>
```

每个体系记录：

1. convergence flag。
2. termination reason。
3. iteration count。
4. final energy。
5. projected gradient norm。
6. optimizer wall time。
7. end-to-end wall time。
8. HVP apply count 和平均 HVP wall time。
9. rejected trial count、trust-radius exhaustion、negative curvature 次数。

## 9. 数值验收标准

最低标准：

1. 241 收敛步数维持在个位数或接近个位数。
2. 240 不因 linear-add HVP 改造显著增加迭代数，wall-time 应随策略改善下降。
3. FeCl2/OEO 能走 full-AO linear-add HVP 路径，不出现 step_check 或 packed/full 尺寸错误。
4. MnF2 明显摆脱 tiny-step first-order 行为，目标是 20 步以内；中间版本至少不能出现上百步仍靠极小 trust radius 前进。

强验收：

1. 240、241、FeCl2、MnF2 都没有 `packed parameter size does not match parameter view`。
2. chart consistency diagnostic 在四个体系上通过。
3. reduced HVP symmetry 和 finite-difference HVP 检查在可解释误差范围内。
4. linear-add HVP 模式下不需要 `XMVB_CPP_SPARSE_LINEAR_HVP`。

## 10. 风险和处理

最大风险不是 linear-add 公式，而是冗余/近冗余方向、列归一化 gauge、trust metric 和 open-shell outer response 的组合。处理原则：

1. 先保证一个 chart 内的 finite step、tangent、adjoint、metric、HVP 完全一致。
2. 冗余方向用 linear-add tangent basis、metric/HVP 小特征值过滤、level shift 或 trust-region 子问题处理，不通过切回 mixed finite chart 掩盖。
3. mixed-chart 保留为 preparation、diagnostic、fallback 或 preconditioner，不作为默认 Newton finite chart，除非选择实现 mixed finite chart 自身的完整二阶 Hessian action $g_y^T R''(0)$。
4. 每次策略变化都必须在 240、241、FeCl2、MnF2 上重新提交 sbatch；F2 只用于 smoke。
