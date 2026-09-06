# Nonredundant Physical Tangent Refactor

## 目标

本文给出一个最终态设计: 将当前非正交轨道优化统一到**物理轨道切空间**上, 完全规避 mixed-chart `(Q_i, Q_a, Q_v, L_a)` 的长期依赖.

目标不是继续修补 mixed-chart 上的 cheap curvature, 而是统一下面四件事:

1. 优化变量
2. 有限步更新 `retract_step()`
3. 梯度 pullback / 向量投影
4. TNHVP 的 cheap/full 二阶算子

这四者必须定义在**同一个物理坐标系**里. 只要变量和二阶模型不在同一个 chart 中, open-shell sparse HAO 体系上就会持续出现 MnF2 这类收敛步数多、wall-time 长、cheap curvature 失真的问题.

## 与 `VBSCF_Hessian.pdf` 的关系

[`docs/VBSCF_Hessian.pdf`](/pool1/home/xiatao/project/xmvb-cpp/docs/VBSCF_Hessian.pdf) 是本文档的重要参考. 这篇文献确认了三件核心事实:

1. VBSCF 的轨道变量本质上是**orbital replacement** 变量 `R_p^q`, 而不是正交 MO 旋转变量.
2. 非正交 VB 轨道优化中天然存在冗余变量, 至少包括:
   - occupied -> inactive
   - active -> itself
   - 不同不可约表示之间的不合法耦合
3. 旧 Hessian-based NR 方法的处理方式是: 先构造 Hessian, 再通过对角化识别并移除这些冗余/小特征值方向.

本文档与该文献的关系是:

- **继承**: 我们接受"VB 轨道变量是 orbital replacement"这一物理出发点.
- **继承**: 我们接受 active-self 等方向确实是冗余变量.
- **改进**: 我们不把冗余变量留到 Hessian 对角化阶段再删除, 而是在 reduced variable 的构造阶段就直接排除它们.
- **改进**: 我们不再通过 mixed-chart 的辅助正交框架来间接表示 orbital replacement, 而是直接在物理 occupied orbitals 的切空间中表示它.

因此, 本文档不是否定 `VBSCF_Hessian.pdf`, 而是沿着它的物理变量观点继续前推, 把"冗余变量的数值处理"提前为"变量定义阶段的几何约束".

---

## 一、问题本质

当前数值诊断已经说明:

1. 全梯度是对的.
2. exact-context 的 fixed/full analytic HVP 与有限差分基本一致.
3. 真正不对的是 TNHVP 当前使用的 reduced cheap curvature model.

因此, 主要矛盾不是"梯度/Hessian 全都错了", 而是:

> 现在的 reduced variable 不是适合二阶优化的本征变量.

mixed-chart 的问题不在于它完全错误, 而在于它把物理非正交轨道问题拆成了:

- 正交辅助框架 `(Q_i, Q_a, Q_v)`
- 非正交 shape / gauge 变量 `L_a`, `L_ia`

这在一阶优化里可以工作, 但在二阶优化里, 真正的 Hessian 是定义在**物理轨道流形**上的. 如果 reduced variable 仍然是 mixed-chart 变量, 那么 cheap Hessian、preconditioner、trust model 就天然会带上 chart mismatch.

这也是为什么:

- exact HVP 是对的
- 但 reduced cheap curvature 依然会严重失真

MnF2 只是把这个结构性问题放大了.

---

## 二、统一后的几何对象

### 2.1 物理变量

对每个 occupied orbital `p`, 设其固定 AO support 为 `Omega_p`, 局部系数向量为

`x_p in R^{m_p}`, `m_p = |Omega_p|`.

局部重叠矩阵为

`S_p = P_p^T S P_p`

其中 `P_p` 是从 block union support 选出 `Omega_p` 的选择矩阵.

当前轨道满足局部定范数约束

`x_p^T S_p x_p = rho_p^2`

这里 `rho_p` 取 accepted point 的当前原始范数. 这样做与现有
`retract_local_sparse_normalized_step()` 一致, 可以保证 `retract_step(0)` 回到当前存储轨道, 不发生 chart recentering.

全局流形是

`M = Π_p M_p`

其中

`M_p = { x_p : x_p^T S_p x_p = rho_p^2 }`

### 2.2 切空间

对单个 orbital `p`, 切空间为

`T_{x_p} M_p = { dx_p : x_p^T S_p dx_p = 0 }`

也就是说, 合法的一阶变化必须与当前轨道在 `S_p` 度量下正交.

这条公式同时适用于:

- HAO
- OEO
- 任何 fixed-support 非正交轨道

二者差别只在 support 大小 `m_p`, 不在更新公式本身.

---

## 三、真正统一的 reduced variable

### 3.1 只保留物理轨道上的原生变化

对一个 block `b`, 设:

- `C_b = [c_1, ..., c_nocc]` 为该 block 的物理 occupied columns
- `V_b` 为由 `C_b` 构造的 block-local physical virtual complement
- `S_b` 为 block union support 上的 AO overlap

注意: `V_b` 必须从**物理 occupied orbitals** 构造, 不能从 mixed auxiliary frame 构造.

### 3.2 每个 orbital 的局部原生 basis

对目标轨道 `p`, 先定义一个**原始物理增量 basis** `B_p`.

建议的非冗余选择:

1. 若 `p` 是 inactive orbital:

`B_p = P_p^T [ C_active , V_b ]`

即只允许 inactive <- active 和 inactive <- virtual.

2. 若 `p` 是 active orbital:

`B_p = P_p^T [ C_inactive , C_active(excluding p) , V_b ]`

即允许:

- active <- inactive
- active <- other active
- active <- virtual

但**不包含 `c_p` 自己**. 这样 active self-scaling 这种伪自由度从 reduced space 里直接删除, 而不是再靠零曲率或人工 shift 去补.

### 3.3 局部切空间投影

定义局部切空间投影算子

`Pi_p = I - x_p (x_p^T S_p) / (x_p^T S_p x_p)`

则物理切空间 basis 为

`T_p = Pi_p B_p`

这一步的意义是:

- `B_p` 给出允许的物理变化方向
- `Pi_p` 去掉径向分量
- `T_p` 直接落在真实切空间里

### 3.4 局部白化

定义局部实现度量

`G_p = T_p^T T_p`

这里选 `T_p^T T_p` 而不是 `T_p^T S_p T_p`, 是为了与当前代码里的 packed coefficient gradient / HVP 接口保持一致. 也就是说, reduced chart 的实现度量来自存储系数空间的欧氏内积, 而轨道约束几何仍通过 `S_p` 体现在 `Pi_p` 和 retraction 中.

对 `G_p` 做 Cholesky 或特征分解:

`G_p = L_p L_p^T`

定义白化后的局部 tangent basis

`U_p = T_p L_p^{-T}`

于是

`U_p^T U_p = I`

这给出一个真正适合二阶优化的 reduced variable:

`z = concat(z_1, z_2, ..., z_nocc)`

且局部切向增量为

`dx_p = U_p z_p`

这个定义有几个本质优势:

1. mixed-chart 完全消失.
2. active self-scaling null mode 不再进入 reduced space.
3. 每个 orbital 的局部 metric 都可以精确白化, 不再需要一个巨大的 block CG 去近似解 `J^T J`.
4. HAO 和 OEO 完全统一, 差别只在 `Omega_p` 和 `V_b`.

### 3.5 与文献中冗余变量处理的对应

`VBSCF_Hessian.pdf` 明确指出, 一般 VBSCF 波函数里至少有三类冗余变量:

1. occupied -> inactive
2. active -> itself
3. 不同不可约表示之间的不合法耦合

在新框架里, 这三类变量的处理方式应是:

1. occupied -> inactive:
   不作为 allowed local raw basis 的一部分进入 `B_p`.

2. active -> itself:
   直接从 `B_p` 中删除 `c_p` 本身, 不让它形成 reduced direction.

3. symmetry-forbidden couplings:
   在构造 `B_p` 时按 point-group / support rule 直接过滤, 而不是在 Hessian 特征值上事后清理.

这一步非常关键. 它把文献里的"先建 Hessian 再移除冗余变量"提升为:

> 先定义正确的非冗余物理变量, 再在其上建立 Hessian.

这就是本设计比旧 mixed-chart 或旧 dense Hessian 方案更本质的地方.

---

## 四、统一后的有限步更新

### 4.1 有限步 retraction

对白化后的 reduced step `z_p`, 定义物理有限步

`dx_p = U_p z_p`

`x_p(trial) = rho_p * (x_p + dx_p) / || x_p + dx_p ||_{S_p}`

其中

`||y||_{S_p} = sqrt(y^T S_p y)`

这就是当前 `retract_local_sparse_normalized_step()` 的数学形式, 但现在 `dx_p` 来自**统一的物理 tangent basis**, 不再来自 mixed-chart 和 direct-linear 两套混合逻辑.

### 4.2 一阶线性化

因为 `U_p` 的列已经在切空间中, 所以

`dR_p(0)[z_p] = U_p z_p`

这意味着:

- `expand_step()`
- `expand_retract_input_tangent()`
- `project_gradient()`
- `project_vector()`

都应该围绕同一个 `U_p` 来写.

不再需要:

- mixed-chart 的 `Q/L` 变换
- dense full-support OEO 的特殊 chart
- `Disabled` / `UnifiedDirectLinear` / `PhysicalAllOccupied` 等多模式分支

---

## 五、统一后的梯度与 HVP

### 5.1 梯度 pullback

设 `g_{x_p}` 是局部 packed coefficient gradient 经过局部 normalization tangent adjoint 后的协向量, 则 reduced gradient 为

`g_{z_p} = U_p^T g_{x_p}`

由于 `U_p` 已经是白化后的 tangent basis, reduced metric 就是单位阵.

### 5.2 exact fixed/full HVP

对任意 reduced 方向 `z`, 先展开为 packed physical tangent

`dx = scatter_p(U_p z_p)`

然后现有 exact-context analytic operator 只需要沿着这个 `dx` 做方向导数:

`y_x = H_phys[dx]`

最后投回 reduced space:

`y_z = concat_p(U_p^T y_{x_p})`

这里最关键的是:

> expand、finite-step tangent、gradient pullback、HVP pullback 全部使用同一个 `U_p`.

一旦做到这一点, 当前 mixed-chart 下的 chart mismatch 就消失了.

---

## 六、TNHVP 的本质问题: 不是主 HVP, 而是 preconditioner surrogate

当前代码里 `apply_reduced_curvature()` 使用的是一个 baseline surrogate:

`M = C^{1/2} G C^{1/2}`

其中 `C` 来自 gap-based diagonal curvature.

这个模型的问题不是某个参数调得不够好, 而是:

1. 它建立在旧 reduced variable 上.
2. `C` 只是粗糙的能隙对角近似.
3. 它既不是真实 fixed Hessian, 也不是真实 full Hessian.

这与 `VBSCF_Hessian.pdf` 的思路也形成了明确对照:

- 文献中的 Newton-Raphson 方案构造的是 orbital replacement 变量上的显式 Hessian.
- 当前 TNHVP cheap model 只是一个 gap-based surrogate, 并不是 orbital replacement 变量上的真实二阶模型.

因此, 如果目标是得到与文献中 Hessian-based VBSCF 相一致的二阶收敛行为, 那么必须先把两个概念分开:

1. cheap Krylov matvec
2. cheap preconditioner / local curvature surrogate

它们在当前代码里不是同一个东西.

### 6.1 诊断修正: 主 cheap matvec 已经是 `H_fixed`

MnF2 上的有限差分已经说明:

1. reduced gradient 正确
2. analytic fixed HVP 与 finite difference 一致
3. analytic full HVP 与 finite difference 一致

因此, 当前 unified physical tangent 主链里真正用于 TNHVP Krylov 迭代的
cheap operator 已经应理解为:

`H_fixed = d/dz (g_fixed(z)) |_{z=0}`

也就是 accepted point 上**不含 outer response** 的 analytic fixed HVP.

所以, 现在外层收敛步数多、wall-time 长, 不能再简单归因为:

- linear-add / unified chart 本身错误
- fixed/full analytic HVP 主链错误
- cheap operator 还停留在旧 gap surrogate

这些都不是当前主矛盾.

### 6.2 真正错误的是 preconditioner baseline

虽然 cheap matvec 已经是 `H_fixed`, 但 `apply_reduced_curvature()` /
`apply_inverse_reduced_block_preconditioner()` 仍在使用局部 surrogate.

当前 unified sparse path 尝试使用:

`B_p^{loc} = U_p^T (F_p - epsilon_p S_p) U_p`

再叠加 active shift 与正定化.

MnF2 诊断表明这个模型的量级完全错误:

- `analytic_fixed_directional_curvature ~= 3849.24`
- `reduced_curvature_model_directional_curvature ~= 7.31`

这不是调 shift 能解决的问题, 而是模型物理内容缺项.

### 6.3 fixed Hessian 的正确分解

当前 exact fixed HVP 的解析实现自然分解为:

`H_fixed = H_direct_core + H_fixed_upstream`

其中:

1. `H_direct_core`

   表示 accepted point 上 orbital-preparation -> AO effective H1e /
   active 2e / orbital backprop 的直接方向导数.

2. `H_fixed_upstream`

   表示 accepted adjoints 固定时, orbital pullback 本身对轨道的导数.
   它包含:

   - sparse normalization pullback 的二阶项
   - inactive projector pullback 的方向导数
   - `S * delta C_active` 进入 accepted pullback 的项

MnF2 上这两块都很大:

- `||H_direct_core||_inf ~= 9.90e2`
- `||H_fixed_upstream||_inf ~= 1.69e3`

这说明只保留单轨道 `F_p - epsilon_p S_p` 刚度的 surrogate,
本质上同时漏掉了:

1. nonorthogonal projector / normalization 几何
2. accepted-point pullback 的 fixed-upstream 曲率
3. exact active 2e 与 AO effective H1e 的 fixed-core coupling

因此, 如果目标是得到与文献中 Hessian-based VBSCF 相一致的二阶收敛行为, 那么 preconditioner 的方向就不应是继续修 `C^{1/2} G C^{1/2}` 这类 surrogate, 而应尽量逼近:

`H_fixed = d/dz (g_fixed(z)) |_{z=0}`

其中 `z` 已经是本文档定义的物理 tangent reduced variable.

### 6.4 preconditioner 的角色

这时 gap model 不再扮演 cheap Hessian 本体, 而只扮演 preconditioner seed:

1. 首选:

`B_p^fix = U_p^T H_fixed U_p`

作为 orbital-local block Hessian

2. 若局部维度仍偏大, 再退化为其对角或低秩近似

3. gap-based diagonal 只能作为 `B_p^fix` 的初始化近似, 不应该再承担 trust-region 预测模型本身

如果不显式构造全 Hessian, 那么最合理的低成本局部近似也应来自:

`B_p^fix ≈ B_p^{direct_core} + B_p^{fixed_upstream}`

而不是只保留

`U_p^T (F_p - epsilon_p S_p) U_p`.

这才是二阶优化里变量与模型一致的做法.

---

## 七、代码层面的最终结构

## 7.1 `src/vb/orbital/nonredundant_orbital_space.hpp/.cpp`

这是主重构点.

### 保留的核心几何核

下面三组局部物理核应保留:

- `build_local_sparse_overlap_metric()`
- `retract_local_sparse_normalized_step()`
- `linearize_local_sparse_normalized_step()`
- `pull_back_local_sparse_normalized_step_adjoint()`

这些函数已经表达了正确的非正交列归一化几何, 不需要删除.

### 应删除的 mixed-chart 结构

下面这些函数或成员属于 mixed-chart / 双逻辑共存遗留, 目标态应删除:

- `NonredundantSparseLinearRetractionMode`
- `choose_nonredundant_sparse_linear_retraction_mode()`
- `uses_dense_full_support_projector`
- `initialize_block_mixed_chart_cache()`
- `build_mixed_chart_trial_occupied_orbitals()`
- `build_dense_full_support_block_step()`
- `build_dense_full_support_metric_diagonal()`
- `build_sparse_mixed_chart_metric_diagonal()`
- `apply_dense_full_support_candidate_metric()`
- `solve_dense_full_support_candidate_metric()`
- `initialize_dense_full_support_metric_cache()`

### `BlockBasis` 的新含义

`BlockBasis` 应简化为纯物理量:

- `block_overlap_matrix`
- `reference_occupied_orbitals`
- `virtual_orbitals`
- 每个 orbital 的 support projector / local rows
- 每个 orbital 的 `U_p`
- 可选的局部 `B_p^fix` 或其对角

删除:

- `inactive_working_orbitals`
- `active_working_orbitals`
- `inactive_right_transform`
- `active_shape_matrix`
- `active_inactive_gauge_coefficients`
- `inactive_metric_matrix`
- `active_shape_metric_matrix`
- `inactive_plus_gauge_metric_matrix`

这些都是 mixed-chart 的中间产物, 不应在最终态中继续存在.

### 需要重写的接口

1. `expand_step()`

旧含义: 从 mixed/direct-linear candidate coefficients 展开 raw packed step.

新含义: 直接 scatter `dx_p = U_p z_p`.

2. `expand_retract_input_tangent()`

旧含义: 兼容 mixed-chart finite step 的一阶线性化.

新含义: 直接返回 `scatter_p(U_p z_p)` 到完整 `orbital_value_table`.

3. `retract_step()`

旧含义: mixed-chart 和 sparse linear-add 共存.

新含义: 对每个 orbital 统一做

`retract_local_sparse_normalized_step(S_p, x_p, U_p z_p)`.

4. `project_gradient()` 和 `project_vector()`

旧含义: 先走 candidate overlap, 再按模式求解 block metric.

新含义:

- 局部 adjoint pullback
- 局部乘 `U_p^T`
- reduced metric 直接是单位阵

5. `apply_inverse_reduced_block_preconditioner()`

旧含义: 在大 block 上近似反解 `C^{1/2} G C^{1/2}`.

新含义: 应作用在局部 `B_p^fix` 或其对角上.

6. `apply_reduced_curvature()`

不再作为 mixed surrogate.

两个可接受的终态:

- 方案 A: 删除, 只保留 preconditioner
- 方案 B: 重定义为局部 `B_p^fix` 前向作用

建议采用方案 B, 这样诊断工具仍可直接比较 cheap model 与 exact fixed/full HVP.

### 应新增的帮助函数

建议新增下列纯物理 helper:

- `build_orbital_local_raw_basis(...)`
- `build_orbital_local_tangent_basis(...)`
- `orthonormalize_orbital_local_tangent_basis(...)`
- `scatter_orbital_local_tangent_step(...)`
- `pull_back_orbital_local_tangent_covector(...)`
- `build_orbital_local_fixed_hessian_block(...)`
- `build_orbital_local_direct_core_block(...)`
- `build_orbital_local_fixed_upstream_block(...)`

---

## 7.2 `src/vb/scf/exact_orbital_second_order_operator.cpp`

高层 analytic differentiation 链路可以保留.

它真正依赖的只有三件事:

1. `expand_step()`
2. `expand_retract_input_tangent()`
3. `project_reduced_gradient()`

因此这个文件不需要再理解 mixed-chart.

需要做的是:

- 让它完全依赖新的 physical tangent `NonredundantOrbitalSpace`
- cheap Krylov operator 继续直接对接 fixed analytic HVP
- full operator 继续是 fixed + outer response

也就是说, `ExactOrbitalSecondOrderOperator` 应该成为:

> 在物理 tangent chart 上作用的 exact fixed/full HVP 提供者

而不是 mixed-chart 的修补层.

---

## 7.3 `src/vb/scf/cpp_vb_scf_optimizer.cpp`

### 需要删除或弱化的 mixed-chart 逻辑

1. 与 OEO active representative 相关的 chart canonicalization
2. 与 sparse retraction mode 相关的多模式策略分支
3. 与 mixed-chart chart reset 绑定的 secant/history 清理逻辑

### 应保留的逻辑

只有两类 chart repair 仍可能保留:

1. fixed-support 一致性维护
2. 真正的 inactive gauge fix

注意这两者不是 mixed-chart 本体, 只要它们直接作用在物理 packed orbitals 上即可.

### TNHVP 的最终使用方式

优化器层应当把:

- cheap model
- full model
- secant transport
- trust-radius norm

全部建立在新的 reduced variable `z` 上.

这时:

- `project_vector()` 不再需要巨型 block metric CG
- `SR1` 或 L-BFGS secant transport 也不再受 mixed-chart 影响

---

## 7.4 `src/tools/check_exact_ctx_hvp.cpp`

这个工具应继续保留, 因为它是本重构最重要的闭环验证器.

但在新框架下它会更简单:

- 不再需要区分 mixed vs direct-linear
- reduced direction 只有一种定义
- cheap/full/fixed 三者都在同一个物理 tangent chart 上比较

---

## 八、为什么这是本质方案而不是修补

这个重构不是调参数, 而是替换变量定义.

当前框架的问题是:

1. 变量在 mixed-chart
2. 有限步在物理轨道
3. exact HVP 最终还是物理导数
4. cheap curvature 却在 mixed surrogate 上建模

这四件事不在同一个几何对象上.

新的框架把四者统一为:

- 变量: `z`
- 物理切向增量: `dx_p = U_p z_p`
- retraction: `rho_p (x_p + dx_p) / ||x_p + dx_p||_{S_p}`
- 梯度 pullback: `U_p^T`
- fixed/full HVP pullback: `U_p^T H U_p`

这才是二阶优化真正需要的一致性.

---

## 九、建议的实现顺序

### 第一步: 只改 `NonredundantOrbitalSpace`

先不动 exact HVP 主体, 只完成:

1. 局部物理 tangent basis `U_p`
2. 新的 `expand_step()`
3. 新的 `expand_retract_input_tangent()`
4. 新的 `retract_step()`
5. 新的 `project_gradient()` / `project_vector()`

做到这一步后:

- mixed-chart 在轨道更新层已经被完全规避
- exact HVP 可以直接复用新 chart

### 第二步: 重写 preconditioner, 不是再改 cheap operator

让 TNHVP 继续使用:

- fixed accepted-point analytic HVP

作为 cheap Krylov matvec.

然后单独重写 preconditioner, 使其逼近局部 `B_p^fix`, 至少要显式纳入:

- `B_p^{direct_core}` 的主导 local part
- `B_p^{fixed_upstream}`

而不是继续依赖 `F_p - epsilon_p S_p` 或 gap diagonal.

### 第三步: 删除 mixed-chart 残留

当 F2 / 241 / MnF2 的:

- gradient finite difference
- fixed/full HVP finite difference
- optimizer convergence steps

都通过后, 直接删除 mixed-chart 遗留函数与结构体字段, 不保留 fallback.

---

## 十、最终结论

本质方案不是继续修 `mixed-chart cheap curvature`, 也不是继续调
`F_p - epsilon_p S_p` 这种单轨道 surrogate.

本质方案是:

> 用物理 occupied orbitals 的局部切空间作为唯一优化变量,  
> 用同一个变量定义 finite retraction、gradient pullback、fixed/full HVP 和 TN preconditioner,  
> 并让 preconditioner 逼近 accepted-point `H_fixed` 的局部块分解.

在这个终态里:

- HAO 与 OEO 完全统一
- mixed-chart 被完全规避
- active self-scaling null mode 从 reduced space 直接消失
- giant block `J^T J` 近似求解被局部白化替代
- TNHVP 的 cheap/full 二阶模型终于与变量本身一致
- preconditioner 终于包含 nonorthogonal pullback 的主导固定曲率

这才是 MnF2 这类 open-shell sparse HAO 问题的结构性解法.
