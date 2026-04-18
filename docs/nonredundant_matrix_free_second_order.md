# Nonredundant Matrix-Free Second-Order Optimization

## 1. 我们真正要做的是什么

目标不是把老版 XMVB 的精确 Hessian 重新抄一遍，而是把它改造成

\[
\text{nonredundant} + \text{matrix-free} + \text{second-order}
\]

的优化器。

更准确地说：

- 老版工作已经完成了 VBSCF 轨道 Hessian 的物理公式、RDM/AFCG 生成、以及 dense Hessian 的求解流程。
- 我们现在真正要做的是：保留这些“精确二阶信息”，但不再显式构造 dense nonredundant Hessian，也不再对它做对角化，而是改成 Krylov / truncated-Newton 所需的算子作用

\[
v \mapsto H_{\mathrm{nr}} v.
\]

这才是新的算法路线。

## 2. 老版 XMVB 已经做了什么

老版 Fortran 的核心流程可以概括为：

1. 计算辅助轨道表示下的梯度和 Hessian。
2. 变换到 original-orbital 表示，得到

\[
g_{\mathrm{oo}}(k,j), \qquad
H_{\mathrm{oo}}(k,j;m,l).
\]

3. 用 nonredundant 映射把它们抽取到 reduced 空间：

\[
g_{\mathrm{nr}} = P^{\mathrm T} g_{\mathrm{oo}}, \qquad
H_{\mathrm{nr}} = P^{\mathrm T} H_{\mathrm{oo}} P.
\]

4. 显式形成 dense \(H_{\mathrm{nr}}\)，再求解

\[
H_{\mathrm{nr}} \Delta x = - g_{\mathrm{nr}},
\]

通常通过对角化 / dense 线性方程求解完成。

其中第 3 步在老版代码里并不神秘，本质上只是 index lifting / projection：

- `putDeltaOrb`：把 reduced 向量放回 original-orbital 旋转矩阵
- `getNonredGrad`：把 \(g_{\mathrm{oo}}\) 投影回 reduced 空间
- `getNonredHess`：显式抽取 \(H_{\mathrm{nr}}\) 的所有矩阵元

所以，从方法学上看：

- “精确 Hessian 公式”不是新的；
- “nonredundant dense Hessian Newton”也不是新的；
- 真正新的，是把它改写成 **nonredundant matrix-free Hessian action**。

## 3. 数学核心：只做 \(H_{\mathrm{nr}} v\)，不做 \(H_{\mathrm{nr}}\)

设 \(x \in \mathbb{R}^{n_{\mathrm{nr}}}\) 是 nonredundant 变量，\(P\) 是把它 lift 到 original-orbital 旋转矩阵的线性映射。定义

\[
\Delta_{\mathrm{oo}} = P v,
\]

其中 \(\Delta_{\mathrm{oo}}(m,l)\) 是 original-orbital 轨道旋转方向。

然后原始 Hessian 作用在这个方向上，得到一个 original-orbital 梯度型对象

\[
Y_{\mathrm{oo}}(k,j)
= \sum_{m l} H_{\mathrm{oo}}(k,j;m,l)\,\Delta_{\mathrm{oo}}(m,l).
\]

最后再投影回 nonredundant 空间：

\[
y = P^{\mathrm T} Y_{\mathrm{oo}}.
\]

于是就有

\[
H_{\mathrm{nr}} v
= P^{\mathrm T} \bigl( H_{\mathrm{oo}} (P v) \bigr).
\]

这就是我们要的 matrix-free reduced Hessian action。

### 3.1 等价的 reduced-space 表述

若把当前 C++ 非冗余空间记成 packed 参数空间中的正交基 \(Q\)，则也可写成

\[
g_{\mathrm{nr}} = Q^{\mathrm T} g_{\mathrm{packed}},
\qquad
H_{\mathrm{nr}} = Q^{\mathrm T} H_{\mathrm{packed}} Q,
\]

以及

\[
H_{\mathrm{nr}} v
= Q^{\mathrm T}\bigl(H_{\mathrm{packed}} (Q v)\bigr).
\]

当前 C++ 中：

- `NonredundantOrbitalSpace::expand_step()` 就是 \(v \mapsto Q v\)
- `NonredundantOrbitalSpace::project_vector()` 就是 \(w \mapsto Q^{\mathrm T} w\)

因此，当前架构其实已经非常接近这个目标了。真正缺的是中间这一步：

\[
w = H_{\mathrm{packed}} (Q v)
\quad\text{或等价地}\quad
Y_{\mathrm{oo}} = H_{\mathrm{oo}} (P v).
\]

## 4. 为什么这不算重复旧工作

因为我们不是重复推一遍 Hessian 公式，而是复用旧版已有的精确二阶物理内容，把求解架构从

\[
\text{assemble } H_{\mathrm{nr}}
\rightarrow
\text{diagonalize / solve dense system}
\]

改成

\[
\text{define operator } v \mapsto H_{\mathrm{nr}} v
\rightarrow
\text{Krylov / truncated Newton / iterative solve}.
\]

两者的差别是本质性的：

- 旧版输出的是矩阵；
- 我们需要的是算子；
- 旧版 bottleneck 包含 reduced Hessian 组装和对角化；
- 我们的新路线要把这两步彻底拿掉。

这正好也对应 Hessian 论文里提到的方向：用 iterative linear equation solver 取代 Hessian diagonalization。

## 5. 对当前 C++ 代码意味着什么

当前 `nonredundant_truncated_newton` 的问题不在外层框架，而在 `H v` 的来源。

现在的实现本质上还是有限差分：

\[
H_{\mathrm{nr}} v
\approx
\frac{g_{\mathrm{nr}}(x + \varepsilon Q v) - g_{\mathrm{nr}}(x)}{\varepsilon}.
\]

这意味着每次 Krylov 里的 `apply(v)`，都要重新做一次完整梯度计算。于是：

- 步数虽然少了；
- 每一步里每个 CG iteration 仍然很贵；
- 所以 total wall time 未必优于 NR-LBFGS。

因此当前真正的瓶颈不是 truncated Newton 这个框架，而是：

\[
\text{finite-difference HVP is too expensive}.
\]

## 6. 直接做最终版还缺什么

这里的“最终版”特指

\[
\text{nonredundant} + \text{exact} + \text{matrix-free} + \text{direct-action}
\]

的二阶优化器，也就是：

- 不显式构造 dense \(H_{\mathrm{nr}}\)；
- 不把显式 \(H_{\mathrm{oo}}\) 当作主工作流；
- 直接对给定方向计算

\[
v_{\mathrm{nr}}
\xrightarrow{P}
\Delta_{\mathrm{oo}}
\xrightarrow{\text{direct action}}
Y_{\mathrm{oo}}
\xrightarrow{P^{\mathrm T}}
y_{\mathrm{nr}}.
\]

### 6.1 当前已经具备的部分

当前 C++ 框架已经具备三块重要基础：

1. nonredundant 坐标空间与 \(Q/Q^{\mathrm T}\) 两端接口；
2. truncated-Newton / trust-region / CG 外层求解骨架；
3. exact 梯度和 same-spin 加速后的主计算路径。

因此问题已经不再是“二阶优化器整体不存在”，而是“二阶物理算子层尚未补上”。

从当前代码结构看，最自然的切口是：

- `src/vb/scf/cpp_active_space_gradient_evaluator.cpp`
  - `build_active_space_gradient_forward_context(...)`
  - `accumulate_active_space_gradient(...)`
- `src/vb/scf/cpp_orbital_gradient_evaluator.cpp`
  - active-space layer backprop 到 AO / orbital 参数层
- `src/vb/scf/cpp_vb_scf_optimizer.cpp`
  - 现有 `MatrixFreeReducedHvpOperator`
- `src/vb/orbital/legacy_style_orbital_gradient_projector.cpp`
  - 当前最接近 old `Grdori` / original-orbital 投影语义的 C++ 实现

也就是说：

- active-space forward/adjoint 层适合插入 exact directional operator；
- optimizer 侧只需要替换 `H v` 来源；
- oo-space / legacy-style 桥接建议独立成新 helper，而不是把逻辑硬塞进 optimizer。

### 6.2 真正缺失的核心模块

#### A. exact directional Hessian action

最核心的缺口，是一个新的二阶算子模块，能够直接计算

\[
Y_{\mathrm{oo}} = H_{\mathrm{oo}} \Delta_{\mathrm{oo}}
\]

而不是先显式 materialize 四指标 Hessian，再去乘方向。

这一步是整个项目的中心。如果这一块没有，当前 TN 就只能继续使用有限差分 HVP。

建议新增模块：

- `src/vb/scf/exact_orbital_second_order_operator.hpp`
- `src/vb/scf/exact_orbital_second_order_operator.cpp`

该模块的职责不是返回显式 Hessian，而是提供：

\[
(\text{state at } x,\, \Delta_{\mathrm{oo}})\;\longmapsto\;Y_{\mathrm{oo}}.
\]

#### B. oo-space lifting / projection helper

虽然当前 packed 参数空间的 `expand_step()` / `project_vector()` 已经存在，但为了复用 old XMVB 的 Hessian 公式与索引结构，仍然需要一个更接近 old `putDeltaOrb` / `getNonredGrad` 语义的辅助层：

\[
v_{\mathrm{nr}} \leftrightarrow \Delta_{\mathrm{oo}}, \qquad
Y_{\mathrm{oo}} \leftrightarrow y_{\mathrm{nr}}.
\]

这层不是最终瓶颈，但它决定了：

- 公式实现是否清晰；
- debug 是否可控；
- 与 old Hessian 代码是否能一一对照。

建议新增模块：

- `src/vb/orbital/original_orbital_direction_map.hpp`
- `src/vb/orbital/original_orbital_direction_map.cpp`

它至少应提供：

- nonredundant / packed direction \(\to \Delta_{\mathrm{oo}}\)
- \(Y_{\mathrm{oo}} \to\) reduced / packed response
- 与 legacy-style projector 相兼容的 original-orbital 索引约定

从当前 C++ 实现看，这里至少还有两个具体缺口：

1. `NonredundantOrbitalSpace` 内部已经持有块内 inactive-active / occupied-virtual 的 candidate 方向系数，但这些 candidate coefficients 目前没有公开接口；
2. 当前 reduced 坐标最终只会被展开回 packed sparse-parameter 向量，而不会被解释成一个显式的 block-local orbital-rotation amplitude 对象。

因此，最终版实现前需要先补一个“方向表示层”，把

\[
v_{\mathrm{nr}}
\longrightarrow
\{\Delta^{(b)}_{\mathrm{cand}}\}_b
\longrightarrow
\Delta_{\mathrm{oo}}
\]

这条链显式化。

#### C. Hessian contributions 的 directional 化

最终版不能停留在

\[
\text{build } H_{\mathrm{oo}} \text{ first}
\]

这一层，而必须把 Hessian 各项贡献改写成直接作用于 \(\Delta_{\mathrm{oo}}\) 的响应项：

\[
H^{(a)}_{\mathrm{oo}}
\;\mapsto\;
Y^{(a)}_{\mathrm{oo}} = H^{(a)}_{\mathrm{oo}} \Delta_{\mathrm{oo}}.
\]

这意味着 old `getOOGrad + hessoo` 路线要被拆开，只保留其物理内容，不保留其“四指标先组装后求解”的数据流。

从当前 C++ 代码组织看，这部分很可能需要把现有一阶 active-space evaluator 的若干逻辑拆成可复用 contribution kernels，例如：

- same-spin / one-electron directional contribution
- opposite-spin directional contribution
- overlap-response directional contribution

这些实现更适合放在 `src/vb/scf/`，并由 `exact_orbital_second_order_operator` 统一调度。

同时也意味着，当前 `CppOrbitalGradientEvaluator` 这一层不能直接充当最终版二阶核心。它当前暴露的是一阶 reverse 接口：

\[
\text{input} \mapsto E,\ g
\]

而不是二阶 directional 接口：

\[
(\text{input}, \Delta) \mapsto H\Delta.
\]

所以真正应该新增的是一个并列的 second-order operator 模块，而不是继续把一阶 gradient evaluator 的职责向上堆叠。

#### D. 面向 Krylov 的二阶预条件

解析 `H v` 补上之后，新的瓶颈就会变成 Krylov 迭代数。

因此最终版必须同步补上更强的 preconditioner，而不能只依赖当前 block-local curvature diagonal。至少要准备：

- Hessian diagonal / near-diagonal derived preconditioner；
- inactive-active / occupied-virtual 分块预条件；
- 必要时的 transported secant 与物理预条件混合策略。

### 6.3 哪些东西不是最终目标

下面这些可以作为开发期 oracle 或 debug 参考，但不应作为最终版主工作流：

- 显式构造 dense \(H_{\mathrm{nr}}\)；
- 对 dense \(H_{\mathrm{nr}}\) 做 `dsyev`；
- 显式构造完整 \(H_{\mathrm{oo}}\) 并把它当成正式运行路径；
- 继续依赖有限差分 HVP 作为 production 二阶算子。

其中“显式 \(H_{\mathrm{oo}}\) 再乘方向”可以保留为小体系校验路径，但只能作为验证工具，不能作为论文和最终性能版本的核心算法。

### 6.4 最终版必须补齐的验证

如果直接做最终版，那么验证也必须同步上到位，至少包括：

1. directional finite-difference check：

\[
H v \approx \frac{g(x+\varepsilon v)-g(x)}{\varepsilon};
\]

2. 双线性对称性检查：

\[
u^{\mathrm T} H v = v^{\mathrm T} H u;
\]

3. 小体系与显式 Hessian 的对照：

\[
H_{\mathrm{nr}} v
\stackrel{?}{=}
P^{\mathrm T} H_{\mathrm{oo}} P v;
\]

4. 进入 TN 后的求解稳定性检查：

- trust ratio 是否稳定；
- negative-curvature handling 是否正常；
- CG 迭代数是否比 FD-HVP 版本更可控。

## 7. 复杂度上到底改掉了什么

### 7.1 旧 dense nonredundant Newton

旧流程包含：

1. 构造 Hessian 相关张量；
2. 显式组装 dense \(H_{\mathrm{nr}}\)；
3. dense 对角化 / dense 线性求解。

因此至少包含：

\[
O(n_{\mathrm{nr}}^2)
\]

的存储，以及大约

\[
O(n_{\mathrm{nr}}^3)
\]

的 reduced Hessian 求解代价。

### 7.2 我们要做的 matrix-free Newton

新流程只需要：

\[
v \mapsto H_{\mathrm{nr}} v.
\]

于是：

- 不再需要存储 dense \(H_{\mathrm{nr}}\)；
- 不再有 \(O(n_{\mathrm{nr}}^3)\) 的对角化；
- 求解代价变成若干次 `H v` 与向量内积。

也就是说，外层求解从 dense direct solver 变成 Krylov iterative solver。

### 7.3 最终版的现实约束

如果只是做到“显式生成 \(H_{\mathrm{oo}}\)，然后再做张量收缩”，那么虽然 reduced Hessian 的瓶颈消失了，`H v` 本身仍可能偏贵。

因此最终版真正要追求的是：

\[
\text{direct analytic } H_{\mathrm{nr}} v
\]

也就是：

- 不显式生成 \(H_{\mathrm{nr}}\)；
- 最好也不把完整 \(H_{\mathrm{oo}}\) 作为正式运行时对象；
- 直接从 exact Hessian 各项公式生成 \(Y_{\mathrm{oo}}\)。

## 8. 与当前 same-spin/unique-spin-string 项目的关系

这两条线并不冲突，反而是互补的：

- same-spin / unique-spin-string：显著降低了矩阵元和梯度构建中的 determinant-pair 冗余；
- nonredundant matrix-free second-order：降低优化器对 dense Hessian 的依赖，减少二阶方向求解成本。

前者主要解决“单次梯度/能量评估”的成本；
后者主要解决“如何更高质量地利用二阶信息而不把 Hessian dense 化”。

如果两者合起来成功，才可能同时做到：

- 单步更便宜；
- 迭代步数更少；
- 总墙钟时间真正下降。

## 9. 直接做最终版的执行清单

这里不把“显式 \(H_{\mathrm{oo}}\) 路线”当正式阶段，只把它看作必要时的小体系校验工具。

### 阶段 A：搭出最终版 operator 的骨架

目标是把当前

\[
\text{FD-HVP}
\]

替换成

\[
v_{\mathrm{nr}}
\mapsto
P^{\mathrm T}\bigl(Y_{\mathrm{oo}}[\Delta_{\mathrm{oo}} = Pv_{\mathrm{nr}}]\bigr),
\]

其中 \(Y_{\mathrm{oo}}\) 来自 direct action，而不是显式 Hessian。

具体要做：

1. 单独实现 oo-space direction container 与 lifting/projection helper；
2. 定义新的 exact second-order operator 接口；
3. 让当前 `nonredundant_truncated_newton` 直接消费这个 operator。

### 阶段 B：把 old Hessian 公式逐项改写成 direct action

目标是：

\[
\Delta_{\mathrm{oo}}
\longmapsto
Y_{\mathrm{oo}}
\]

而不是

\[
\Delta_{\mathrm{oo}}
\longmapsto
H_{\mathrm{oo}}
\longmapsto
Y_{\mathrm{oo}}.
\]

这一步是最终版的主体工作。也就是说：

- old formula 要复用；
- old dataflow 不能照搬。

### 阶段 C：补强二阶预条件

当解析 `H v` 跑通后，立刻补：

- diagonal / block-diagonal physical preconditioner；
- inactive-active / occupied-virtual 分块缩放；
- 必要时与 transported secant 混合。

否则求解框架虽然已经“正确”，但 total wall time 未必足够好。

### 阶段 D：最终验证与 benchmark

最终版必须同时给出：

1. 数值正确性：
   \(H v\) 对有限差分一致；
2. 算子性质：
   对称性与负曲率行为正常；
3. 优化器性能：
   相对当前 FD-TN 的单步与总墙钟收益；
4. 方法学比较：
   相对 old dense exact Hessian 的复杂度与内存优势。

## 10. 文章层面的定位

如果只做“把老版 dense Hessian 搬到 C++”：

- 价值有限；
- 更像重构/移植。

如果做到

\[
\text{nonredundant matrix-free exact second-order optimization}
\]

那么方法学表述就非常清晰：

1. 保留精确 Hessian 的物理信息；
2. 避免 dense reduced Hessian 的组装与对角化；
3. 在 nonredundant 坐标中直接进行 iterative second-order solve；
4. 与当前 unique-spin-string 加速自然兼容。

这才是值得作为下一阶段核心目标的路线。
