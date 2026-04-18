# Nonredundant Matrix-Free Second-Order Optimization: Current Status

## 1. Executive Summary

截至目前，可以把当前工作状态概括为：

\[
\boxed{
\text{方法框架已经做出来了，而且稳定 exact 路径已经在大体系上赢过了 } \texttt{lbfgspp}.
}
\]

更具体地说：

1. 我们已经具备了

\[
\text{nonredundant} + \text{matrix-free} + \text{second-order}
\]

这一优化器的核心算法形态。

2. 当前 `nonredundant_truncated_newton` 已经不是“概念验证”级别的空壳，而是已经包含：
   - nonredundant reduced space；
   - trust-region / truncated-Newton 外层；
   - Krylov / CG 内层；
   - reduced Hessian-vector product 接口；
   - `exact_ctx` 这一 accepted-point second-order operator 落脚点。

3. 当前稳定生产路径 `exact_ctx` fallback 在 `C6H6_full` 上已经达到明确的总 wall-time 优势：

\[
\texttt{nonredundant\_truncated\_newton\ exact\_ctx}
=
12.1247\ \mathrm{s},
\qquad
\texttt{lbfgspp}
=
28.4372\ \mathrm{s}.
\]

也就是说，当前稳定二阶路径已经约快

\[
\frac{28.4372}{12.1247} \approx 2.35\times.
\]

4. 但是，当前每次 \(H v\) 仍然偏贵，same-spin 已不再是主要瓶颈，后续继续压时间仍然要集中在 exact HVP。

5. 此外，当前受环境变量

```bash
XMVB_CPP_ENABLE_EXACT_CTX_STAGE1_ANALYTIC_CORE=1
```

控制的新解析路径，在我本次复测时仍会导致 `F2` 端到端优化器运行出现段错误。因此它还不能算 production-ready。

---

## 2. 我们现在是否已经“做出了” nonredundant matrix-free second-order optimization

答案是：

\[
\boxed{\text{是，已经做出来了；但还没有做成最终高效版本。}}
\]

这句话里，“做出来了”指的是算法组织方式已经成立，而不是说所有热点都已经优化到位。

当前要解的 reduced Newton 问题是

\[
H_{\mathrm{nr}} \Delta x = - g_{\mathrm{nr}},
\]

其中

\[
g_{\mathrm{nr}} = Q^{\mathrm T} g_{\mathrm{packed}},
\qquad
H_{\mathrm{nr}} = Q^{\mathrm T} H_{\mathrm{packed}} Q.
\]

matrix-free 的关键不是显式形成 \(H_{\mathrm{nr}}\)，而是只实现算子作用

\[
v \mapsto H_{\mathrm{nr}} v
=
Q^{\mathrm T}\!\left( H_{\mathrm{packed}} (Q v) \right).
\]

从这个定义看，当前代码已经有了：

1. \(Q\) 与 \(Q^{\mathrm T}\) 两端接口；
2. reduced-space trust-region truncated Newton；
3. Hessian-vector product operator；
4. accepted-point exact context；
5. 独立的 HVP 数值校验工具。

因此，从方法学角度，当前系统已经不是“还没开始做”，也不是“只有一阶优化器”，而是一个已经成型但尚未提速完成的二阶框架。

---

## 3. 代码中已经完成的核心部件

### 3.1 Nonredundant reduced space

当前代码已经在 orbital packed parameter 空间上构造 nonredundant basis，并提供

\[
v_{\mathrm{nr}} \xrightarrow{Q} v_{\mathrm{packed}},
\qquad
w_{\mathrm{packed}} \xrightarrow{Q^{\mathrm T}} w_{\mathrm{nr}}
\]

这两个核心映射。

对应实现主入口位于：

- `src/vb/orbital/nonredundant_orbital_space.hpp`
- `src/vb/orbital/nonredundant_orbital_space.cpp`

### 3.2 Truncated-Newton / trust-region 外层

当前 `nonredundant_truncated_newton` 已经具备：

1. reduced gradient 驱动的外层迭代；
2. trust radius 更新；
3. inner CG / Krylov 解子问题；
4. boundary step / negative curvature 处理；
5. secant-based curvature history 与预条件器接口。

对应主流程位于：

- `src/vb/scf/cpp_vb_scf_optimizer.cpp`

### 3.3 Exact accepted-point HVP landing zone

当前 exact 路径不再只依赖“完整梯度有限差分”，而是已经有 accepted-point second-order context：

- `src/vb/scf/cpp_active_space_second_order_context.hpp`
- `src/vb/scf/exact_orbital_second_order_operator.hpp`
- `src/vb/scf/exact_orbital_second_order_operator.cpp`

### 3.4 新增的诊断工具

为避免“感觉对了但其实没对”，当前已经有两个专门的检查程序：

- `src/tools/check_exact_two_electron_hvp.cpp`
- `src/tools/check_exact_ctx_hvp.cpp`

其中 `check_exact_ctx_hvp` 直接比较当前 operator 给出的解析/半解析 HVP 与有限差分 HVP。

---

## 4. 当前 benchmark 结论

下面只记录本次复测中最关键的 benchmark 结论。

### 4.1 `F2`：小体系上，HVP 数值是对的

运行：

```bash
env OMP_NUM_THREADS=1 XMVB_CPP_ENABLE_EXACT_CTX_STAGE1_ANALYTIC_CORE=1 \
  build/src/check_exact_ctx_hvp src/test_molecule/F2.xmi --step 1e-3
```

得到：

\[
\text{max\_abs\_diff} = 3.42 \times 10^{-7},
\qquad
\text{max\_rel\_diff} = 1.50 \times 10^{-7}.
\]

这说明：

\[
\boxed{\text{当前 exact\_ctx HVP 算子在 F}_2\text{ 上数值上是正确的。}}
\]

### 4.2 `F2`：fallback 路径可收敛

运行：

```bash
env OMP_NUM_THREADS=1 build/src/xmvb-cpp.exe src/test_molecule/F2.xmi \
  --optimizer-backend nonredundant_truncated_newton \
  --nonredundant-truncated-newton-hvp-mode exact_ctx
```

得到：

- `iterations = 4`
- `final_gradient_inf_norm = 1.534298276e-3`
- `total_wall_time_seconds = 0.026494754`

并且单步细分显示：

- `DT_S \approx 0.0057-0.0059 s`
- 端到端仍然是 `4` 个 accepted iterations
- 当前小体系上已经不再需要额外的 accepted-point baseline probe

这说明：当前旧的 `exact_ctx` fallback 框架在小体系上已经具备正常的二阶收敛行为。

### 4.3 `F2`：新解析 core 当前存在稳定性回退

运行：

```bash
env OMP_NUM_THREADS=1 XMVB_CPP_ENABLE_EXACT_CTX_STAGE1_ANALYTIC_CORE=1 \
  build/src/xmvb-cpp.exe src/test_molecule/F2.xmi \
  --optimizer-backend nonredundant_truncated_newton \
  --nonredundant-truncated-newton-hvp-mode exact_ctx
```

本次复测直接退出，返回码为 `139`，即段错误。

因此当前需要非常明确地区分：

1. `check_exact_ctx_hvp` 数值校验通过；
2. 端到端优化器新路径仍存在稳定性问题。

这两件事不能混为一谈。

---

## 5. `C6H6_full` benchmark：当前已经达到什么水平

这里的核心问题已经不再是“新路径能不能快过旧路径”，而是：

\[
\text{当前稳定 exact\_ctx fallback 是否已经成为值得使用的生产优化器？}
\]

答案现在是肯定的。

### 5.1 干净 benchmark

测试条件：

```bash
OMP_NUM_THREADS=1
```

输入：

```bash
test_molecule/C6H6_full.xmi
```

实测结果：

| backend | iterations | final gradient inf-norm | total wall time (s) |
|---|---:|---:|---:|
| `lbfgspp` | 48 | `1.3279e-4` | `28.4372` |
| `nonredundant_truncated_newton --nonredundant-truncated-newton-hvp-mode exact_ctx` | 5 | `7.6697e-4` | `12.1247` |

因此：

\[
\frac{28.4372}{12.1247} \approx 2.35.
\]

也就是说，当前稳定 fallback 二阶路径已经比 `lbfgspp` 快约 `2.35x`。

### 5.2 本轮新增的关键收益来自哪里

本轮最重要的性能修复不是改数学，而是去掉了 fallback `exact_ctx` 里的一个冗余 accepted-point probe：

- 旧实现每个 accepted outer iteration 都会额外再算一次 accepted-point
  `fixed_active_space_adjoint_gradient`，只为了形成有限差分基线；
- 但这个基线在 accepted point 上本来就等于当前 reduced gradient；
- 因此现在直接重用 `current_reduced_gradient_` 作为基线。

这一点把稳定路径进一步从此前约 `15.98 s` 压到了当前的 `12.12 s`。

---

## 6. 当前瓶颈到底在哪里

从 `C6H6_full` 的 objective progress 输出看，same-spin / structure builder 已经不是 exact 二阶优化的主瓶颈。

稳定 fallback 路径里，一个 accepted iteration 的典型分布大致是：

- accepted objective `\approx 0.48-0.56 s`
- `hvp_calls = 3`
- `hvp_wall \approx 1.32-1.35 s`

而单次 fixed-active-adjoint probe 内部，主要时间仍然落在：

- `active_2e \approx 0.25-0.30 s`
- `ao_h1e \approx 0.085-0.10 s`
- `ao_h1e_backprop \approx 0.09-0.12 s`

因此现阶段最准确的判断是：

\[
\boxed{
\text{same-spin 的主账已经结清；当前真正的主瓶颈是 exact HVP.}
}
\]

真正的大头已经转移到：

1. exact active-space two-electron preparation；
2. AO effective one-electron build / backprop；
3. orbital-preparation pullback；
4. inner CG 中重复出现的 HVP 调用。

---

## 7. 为什么说“matrix-free 已经做出来了，而且已经有生产价值”

当前瓶颈并不在 outer TN 框架，而在 \(H v\) 的实际成本。但这已经不再意味着“方法还不值得用”。

当前 `C6H6_full` 的稳定路径特征是：

\[
N_{\mathrm{outer}} = 5,
\qquad
k_{\mathrm{hvp}} \approx 3.
\]

也就是说，outer step 数已经很小；继续提速主要靠降低单次 \(H v\) 的成本，而不是再去讨论框架是否成立。

所以现在更准确的说法应当是：

\[
\boxed{
\text{matrix-free second-order 不仅已经做出来了，而且稳定路径已经具备明确的 wall-time 价值。}
}
\]

---

## 8. 距离“本质降低单步时间”还差多少

如果目标是“继续明显压低单步时间”，那么工作当然还没有结束。

从 `C6H6_full` 的 accepted iteration 看，`hvp_wall` 仍占单步主体。因此若想再出现一轮明显体感上的提速，经验上仍需要把单次 HVP 进一步压到大约

\[
0.20 \sim 0.30\ \mathrm{s}
\]

这个量级。

换句话说：

\[
\boxed{
\text{当前已经跨过了“值不值得用”的门槛，但还没有跨过“exact HVP 已经足够便宜”的门槛。}
}
\]

---

## 9. 当前最重要的未完成事项

### 9.1 稳定性问题

仍需先修复：

- `XMVB_CPP_ENABLE_EXACT_CTX_STAGE1_ANALYTIC_CORE=1` 打开时，`F2` 端到端优化器段错误。

只要这个问题还在，analytic-core 就不能作为默认生产路径。

### 9.2 真正的性能问题

继续降低 HVP 代价时，重点方向应放在：

1. accepted-point orbital normalization / inactive projector / pullback 固定量缓存；
2. exact AO-H1E directional build / backprop 的专门化；
3. exact 2e directional kernel 的 scratch 复用与 support-local / block-local 化；
4. 尽可能减少 inner CG 中的重复 HVP 工作。

---

## 10. 最简明的结论

如果只用一句话总结当前状态，那么最准确的说法是：

\[
\boxed{
\text{nonredundant matrix-free second-order optimization 已经做出来了，而且稳定 exact 路径已经具有明确的总 wall-time 优势。}
}
\]

若再补一句工程判断，则是：

\[
\boxed{
\text{same-spin 的账基本已经算清；下一阶段应把火力集中到 exact HVP 的 active-space / AO-H1E / pullback 热点。}
}
\]
