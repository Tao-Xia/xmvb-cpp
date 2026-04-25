# TNHVP 的 Systematic Multi-Fidelity Trust-Region 方案

## 1. 目标

当前 TNHVP 已经具备解析 HVP，因此局部二阶曲率信息并不缺失。真正限制收敛效率和总 wall time 的问题在于：

1. 低成本模型与高保真模型都存在，但高保真模型没有系统地进入 trust-region 子问题；
2. 目前 `full probe` / `full retry` / `hybrid followup` 这类逻辑，本质上都属于经验化补救，而不是一个统一的多保真二阶框架；
3. 若高保真信息只用于事后修正接受判据，而不进入步的构造，则往往只会增加每步代价，而不能显著减少外层迭代数。

因此，更本质的方向不是继续增加启发式分支，而是把当前的 cheap/full 两套 HVP 明确地组织成一个 **multi-fidelity trust-region (MFTR)** 方法。

## 2. 两个保真度的局部模型

在第 \(k\) 个 accepted point 处，记能量和梯度为
\[
f_k = E(x_k), \qquad g_k = \nabla E(x_k).
\]

在非冗余 reduced space 中，当前代码已经自然给出了两种 Hessian-vector product：

1. 低保真模型 \(B_k^{(L)}\)：cheap/core-only HVP；
2. 高保真模型 \(B_k^{(H)}\)：full HVP，即 core + outer-response。

于是可定义两个局部二次模型
\[
m_k^{(L)}(s) = f_k + g_k^\top s + \frac{1}{2} s^\top B_k^{(L)} s,
\]
\[
m_k^{(H)}(s) = f_k + g_k^\top s + \frac{1}{2} s^\top B_k^{(H)} s.
\]

这里

- \(s \in \mathbb{R}^{n_r}\) 为 reduced-space step；
- \(n_r\) 是当前 nonredundant orbital space 的维数；
- \(B_k^{(L)}\) 与 \(B_k^{(H)}\) 都不显式组装，只通过 HVP 给出；
- 两个模型共享同一个 \(f_k\) 和 \(g_k\)，因此它们在一阶上一致，差别仅在二阶项。

这正是适合 MFTR 的情形。

## 3. 为什么“只做方向校正”不够

最直接的想法是：

1. 用低保真模型 \(m_k^{(L)}\) 解出一步 \(s_k\)；
2. 再计算一次高保真方向曲率 \(B_k^{(H)} s_k\)；
3. 用
\[
\operatorname{pred}_k^{(H)}(s_k)
=
- g_k^\top s_k - \frac{1}{2} s_k^\top B_k^{(H)} s_k
\]
替代原先的低保真预测下降，进而计算 trust ratio。

这种做法在理论上是合法的，但它只修正了接受判据，没有修正步本身。结果是：

1. full HVP 的额外代价被真实付出；
2. 但 candidate step 仍然来自低保真子问题；
3. 当低/高保真模型的最优步差别较大时，trust ratio 会变得更严格，导致半径收缩与外层步数增加；
4. 因而总 wall time 往往恶化。

这说明 **高保真信息必须进入子空间步构造，而不应仅进入事后验收**。

## 4. 系统化 MFTR 的核心思想

### 4.1 低保真 Krylov 子空间

先使用低保真模型 \(B_k^{(L)}\) 做 truncated-Newton / PCG，得到一个 Krylov 子空间
\[
\mathcal{K}_m^{(L)} = \mathrm{span}\{v_1, v_2, \dots, v_m\},
\]
其中
\[
V_m = [v_1, v_2, \dots, v_m] \in \mathbb{R}^{n_r \times m},
\qquad
V_m^\top V_m = I.
\]

当前代码中的 `TruncatedNewtonKrylovSubspace` 已经显式保存了这组正交基，因此这一步不需要新增数学对象。

### 4.2 在同一子空间内投影高保真模型

真正系统化的 MFTR 做法不是重新跑一遍 full PCG，而是把高保真模型限制到同一个低保真 Krylov 子空间上：
\[
T_m^{(H)} = V_m^\top B_k^{(H)} V_m \in \mathbb{R}^{m \times m},
\]
\[
g_m = V_m^\top g_k \in \mathbb{R}^{m}.
\]

随后只在这个小子空间内求解高保真 trust-region 子问题：
\[
\min_{y \in \mathbb{R}^m}
g_m^\top y + \frac{1}{2} y^\top T_m^{(H)} y,
\qquad
\|y\| \le \Delta_k.
\]

得到 \(y_k\) 后再回到 reduced space：
\[
s_k = V_m y_k.
\]

这样做有三个关键优点：

1. **步本身被高保真模型修正**，而不是只修正 predicted decrease；
2. 只需要在已有子空间上评估 full HVP，而不需要再做一轮 full PCG；
3. 子问题维数仅为 \(m\)，通常远小于 reduced space 维数 \(n_r\)。

### 4.3 预测下降与接受判据

一旦子空间步 \(s_k = V_m y_k\) 已由高保真投影模型给出，则预测下降直接使用高保真子空间模型：
\[
\operatorname{pred}_k
=
- g_k^\top s_k - \frac{1}{2} s_k^\top B_k^{(H)} s_k
=
- g_m^\top y_k - \frac{1}{2} y_k^\top T_m^{(H)} y_k.
\]

实际下降定义为
\[
\operatorname{ared}_k = f_k - f(x_k + s_k).
\]

于是 trust ratio 为
\[
\rho_k = \frac{\operatorname{ared}_k}{\operatorname{pred}_k}.
\]

这是标准 trust-region 形式，不再依赖“cheap reject 后再 retry”这种额外补救语义。

## 5. 与当前 full retry 的本质区别

当前 same-iteration full retry 的逻辑本质是：

1. 先用 cheap 模型得到一个步；
2. 如果看起来 mismatch 较大，则再用 full HVP 做若干次 warm-started CG；
3. full 模型的作用方式仍然是“重新内层迭代一次”。

而系统化 MFTR 的逻辑是：

1. 低保真模型只负责生成一个有代表性的 Krylov 子空间；
2. 高保真模型只负责在这个子空间内建立更准确的投影 Hessian；
3. 最终 trust-region 子问题只解一次，但解的是高保真投影子问题。

因此，MFTR 的 full 信息使用方式是

\[
\text{cheap solve} \;\to\; \text{build } V_m \;\to\; \text{project full model onto } V_m \;\to\; \text{solve small high-fidelity TR}.
\]

而不是

\[
\text{cheap solve} \;\to\; \text{if bad then full retry}.
\]

前者是统一算法；后者是补丁式流程控制。

## 6. 最小可落地实现

### 6.1 第一版不做任何启发式分支

第一版 MFTR 建议采用最干净的基线：

1. exact-context 模式下，内层始终用 cheap/core-only HVP 生成 Krylov 子空间；
2. 若 full outer-response 可用，则始终构造
\[
T_m^{(H)} = V_m^\top B_k^{(H)} V_m;
\]
3. 在该子空间内求解高保真 trust-region 子问题；
4. 接受/拒绝只依据高保真子空间模型的 \(\rho_k\)；
5. 删除或绕开 `full retry`、`hybrid followup`、`tail full solve` 这类经验分支。

这一版的优点是算法语义最干净，便于直接判断 MFTR 本身是否有效。

### 6.2 代价估计

若 cheap PCG 产生 \(m\) 个 Krylov 基向量，则构造 \(T_m^{(H)}\) 需要对每个基向量做一次 full HVP：
\[
w_i = B_k^{(H)} v_i, \qquad i = 1, \dots, m,
\]
\[
T_m^{(H)} = V_m^\top [w_1, \dots, w_m].
\]

这一步的 full HVP 次数为 \(m\)。它比单次方向校正更贵，但仍明显不同于 full retry：

1. 不需要再做 full PCG 递推；
2. 不会重复搜索方向；
3. 只在一个固定小子空间内完成高保真修正。

若后续发现 \(m\) 过大，再讨论如何做低秩截断或子空间压缩；第一版不引入这些近似。

## 7. 与现有代码的对应关系

### 7.1 已有对象

当前代码已经具备：

1. `solve_nonredundant_truncated_newton_step(...)`
   - 输出 `TruncatedNewtonStepResult`
   - 内含 `TruncatedNewtonKrylovSubspace`

2. `TruncatedNewtonKrylovSubspace`
   - `orthonormal_basis = V_m`
   - `projected_gradient`
   - 以及当前 cheap 模型下的投影 Hessian 信息

3. `ExactContextReducedHvpOperator`
   - cheap/core-only HVP
   - full HVP

因此，MFTR 所需的主要新增逻辑不是轨道数学，而是：

1. 用 full HVP 在 `V_m` 上重建 \(T_m^{(H)}\)；
2. 用这个 \(T_m^{(H)}\) 解一次小规模 trust-region 子问题；
3. 将求得的 \(y_k\) 映射回 reduced step \(s_k = V_m y_k\)。

### 7.2 推荐新增函数

建议新增一个清晰的工具函数，例如

```cpp
TruncatedNewtonStepResult
solve_high_fidelity_projected_trust_region_step(
    const NonredundantOrbitalSpace::ProjectionResult& current_projection,
    double trust_radius,
    const TruncatedNewtonKrylovSubspace& cheap_krylov_subspace,
    ReducedHvpOperator* full_hvp_operator);
```

其职责是：

1. 读取 cheap Krylov basis \(V_m\)；
2. 组装高保真投影 Hessian \(T_m^{(H)}\)；
3. 在该子空间内解 trust-region 子问题；
4. 返回修正后的 reduced step 与 predicted decrease。

## 8. 为什么这条路更有希望带来本质收益

若只看 cheap HVP，二阶信息虽然可用，但模型精度有限，导致：

1. step quality 不足；
2. 需要更多 accepted iterations；
3. 最终 wall time 未必优于一阶方法。

若直接用 full HVP 做整个内层求解，则：

1. 每轮 inner iteration 代价过高；
2. 二阶优势可能被 full HVP 的成本抵消。

MFTR 的目标正是折中这两个极端：

1. cheap HVP 负责快速生成“正确的局部子空间”；
2. full HVP 只在这个小子空间中发挥决定性作用；
3. 从而尽量同时保留
   - 二阶步质量；
   - 较低的总 full-HVP 成本。

## 9. 当前结论

当前最直接的“cheap 步 + full 方向校正 + corrected trust ratio”原型已经验证为不够本质，因为它只修正预测量，不修正步本身。

因此，后续如果继续推进 MFTR，应直接实现 **高保真投影子空间 trust-region**，而不是在现有代码上继续增加 directional probe、full retry、followup 等经验式补丁。
