# Exact Second-Order Benchmark and Optimization Priorities

## 1. Purpose

这份文档回答两个实际问题：

1. 当前 `nonredundant matrix-free second-order optimization` 在总 wall-time 上，是否已经有意义；
2. 如果后续全部投入性能优化，哪些方向最值得做，哪些收益最大。

这里我们只讨论精确积分路径：

\[
\texttt{int = libcint}
\]

并把真正的判据定义为

\[
T_{\mathrm{total}}^{\mathrm{second\ order}}
<
T_{\mathrm{total}}^{\mathrm{LBFGS}}.
\]

也就是说，二阶方法的价值不由“外层步数”单独决定，而由最终总 wall-time 决定。

---

## 2. Current Benchmark: `C6H6_full`, Exact Integrals

测试条件：

```bash
OMP_NUM_THREADS=1
input = test_molecule/C6H6_full.xmi
```

### 2.1 Measured Results

| backend | HVP path | iterations | final gradient inf-norm | total wall time (s) | comment |
|---|---|---:|---:|---:|---|
| `lbfgspp` | N/A | 48 | `1.3279e-4` | `28.4372` | 当前代码实测，收敛 |
| `nonredundant_truncated_newton` | `exact_ctx` fallback | 5 | `7.6697e-4` | `12.1247` | 当前稳定生产路径 |
| `nonredundant_lbfgspp` | N/A | 35 | `6.4631e-4` | `39.4930` | 历史参考值，未在本轮重跑 |
| `nonredundant_truncated_newton` | `exact_ctx` analytic core | N/A | N/A | N/A | `F2` 端到端仍有段错误，暂不作为生产 benchmark |

这里：

- `exact_ctx fallback` 指环境变量 `XMVB_CPP_ENABLE_EXACT_CTX_STAGE1_ANALYTIC_CORE` 未开启时的生产路径；
- `exact_ctx analytic core` 指开启该环境变量后的新路径，但当前只保留数值校验用途。

### 2.2 Direct Interpretation

从总 wall-time 来看，当前稳定的精确积分二阶路径已经不只是“有潜力”，而是已经在这个 benchmark 上明显赢过现有的一阶基线。

相对普通 `lbfgspp`：

\[
\frac{28.4372}{12.1247} \approx 2.35.
\]

也就是说：

- 当前生产可用的 TN fallback 路径，约快 `2.35x`。

相对 `nonredundant_lbfgspp`：

\[
\frac{39.4930}{12.1247} \approx 3.26.
\]

这里的 `39.4930 s` 是同一代码分支上此前记录的参考值，本轮没有重新测。

因此，结论应当明确写成：

\[
\boxed{
\text{当前稳定版 exact TN 在 } \mathrm{C_6H_6\_full} \text{ 上已经具有明确的总 wall-time 优势。}
}
\]

---

## 3. Important Caveat: The New Analytic-Core Path Is Still Not Stable Enough

虽然 `check_exact_ctx_hvp` 在 `F2` 上仍然给出很好的数值一致性，

\[
\text{max\_abs\_diff} = 3.42 \times 10^{-7},
\qquad
\text{max\_rel\_diff} = 1.50 \times 10^{-7},
\]

但 analytic-core 目前还不能直接作为默认生产路径。原因是：

- 在 `F2` 上，开启

```bash
XMVB_CPP_ENABLE_EXACT_CTX_STAGE1_ANALYTIC_CORE=1
```

后，端到端优化器复测时仍然返回 `139`，即段错误；
- 因此 analytic-core 当前仍然只适合作为 HVP 数值校验路径，而不是生产优化路径。

所以当前最准确的状态是：

1. TN 路线本身已经证明有 wall-time 价值；
2. 稳定生产路径是 `exact_ctx fallback`；
3. analytic-core 还没有达到“默认开启”的稳定性标准。

---

## 4. Where the Time Goes Now

根据现有的 objective progress 计时，`C6H6_full` 的精确积分路径里，主时间已经不在 same-spin 结构矩阵，而在 exact HVP 里的 active-space / pullback 相关层。

典型 objective 计时大致为：

- `active_2e \approx 0.25-0.30 s`
- `ao_h1e \approx 0.09-0.10 s`
- `ao_h1e_backprop \approx 0.09-0.12 s`
- `structure_matrices \approx 0.024-0.028 s`
- `eigensolver \approx 0.010 s`

这意味着：

\[
\boxed{
\text{same-spin 已经不再是 exact 二阶优化的主瓶颈。}
}
\]

当前主要瓶颈是：

1. exact two-electron directional contractions；
2. AO effective one-electron directional build/backprop；
3. orbital-preparation backprop / pullback；
4. inner CG 中重复出现的 HVP 调用。

特别要说明的是，本轮已经去掉了 fallback `exact_ctx` 路径里一个多余的 accepted-point probe：

- 旧实现每个 accepted outer iteration 都会额外再算一次
  `evaluate_fixed_active_space_adjoint_gradient(current_parameters)`，
  只为了形成有限差分基线；
- 但在 accepted point 上，这个基线本来就等于当前 reduced gradient；
- 因而现在直接重用 `current_reduced_gradient_` 作为基线。

这一步不改变算法，只去掉了每个 outer step 一次冗余 probe，因此稳定路径的 wall time 才从此前约 `15.98 s` 进一步降到了当前的 `12.12 s`。

---

## 5. The Key Cost Model

当前 truncated-Newton 的总成本可以写成

\[
T_{\mathrm{TN}}
\approx
N_{\mathrm{outer}}
\left(
T_{\mathrm{obj}}
+
k_{\mathrm{hvp}} T_{Hv}
\right),
\]

其中：

- \(N_{\mathrm{outer}}\) 是 accepted outer iterations；
- \(k_{\mathrm{hvp}}\) 是每一步平均 HVP 次数；
- \(T_{Hv}\) 是单次 Hessian-vector product 成本。

而当前精确积分 `C6H6_full` 上的特征是：

- \(N_{\mathrm{outer}}\) 已经很小，约为 `5`；
- 稳定 fallback 路径当前每步通常有 `3` 次 HVP；
- 因此后续所有大收益优化，本质上都要围绕

\[
T_{Hv}
\]

和

\[
k_{\mathrm{hvp}}
\]

展开。

换句话说：

\[
\boxed{
\text{后续真正值得投入的，不是再抠 same-spin，而是继续把 HVP 做便宜。}
}
\]

---

## 6. Current Hotspots in the Code

### 6.1 Exact HVP front-end

核心入口在：

- `src/vb/scf/exact_orbital_second_order_operator.cpp`

当前 `ExactOrbitalSecondOrderOperator::apply_reduced()` 每次调用都会重新做：

1. direction 对应的 orbital normalization / tangent build；
2. orbital-preparation directional propagation；
3. `AoEffectiveOneElectronBuilder` 的 directional build；
4. `apply_exact_packed_active_two_electron_adjoint_hessian_vector(...)`；
5. `AoEffectiveOneElectronBackpropagator`；
6. `ActiveSpaceOrbitalBackpropagator`；
7. accepted-point 固定上游梯度下的解析 orbital pullback directional correction。

### 6.2 Exact 2e directional kernel

真正最值得关注的热点仍然在：

- `src/vb/orbital/active_space_two_electron_utils.cpp`

尤其是

```cpp
apply_exact_packed_active_two_electron_adjoint_hessian_vector(...)
```

accepted-point 常量缓存现在已经到位，当前每次 HVP 真正还会重做的是：

1. direction-dependent `directional_pair_coefficients`
2. `directional_transformed_pair_coefficients`
3. `directional_pair_gradients`
4. 两次 dense-active backprop

accepted-point 相关的

1. `active_pair_gradient_matrix`
2. `accepted_pair_products`
3. `accepted_base_pair_gradients`

已经被 `ExactPackedActiveTwoElectronAdjointCache` 缓存起来，不再在每次 HVP 中重建。

所以这块后续真正值得做的，已经不是“继续补 accepted-point cache”，而是：

1. 减少 direction-dependent scratch allocation；
2. 继续专门化 exact AO-pair kernel 的 directional 路径；
3. 把非积分层的 orbital pullback 固定量也缓存干净。
### 6.3 Orbital backprop / pullback

另一个明显热点在：

- `src/vb/orbital/active_space_orbital_backpropagator.cpp`

当前 `ActiveSpaceOrbitalBackpropagator::backpropagate(...)` 对固定 accepted input 仍会反复执行：

1. sparse orbital normalization；
2. dense orbital expansion；
3. inactive overlap / inverse；
4. projector 与 auxiliary orbital 构造；
5. 从 dense 梯度再回到 sparse parameter 梯度。

对于固定 accepted-point 来说，这里面很多中间量本来应该可以缓存或线性化。

---

## 7. High-Value Optimization Directions

下面按“预期收益”和“是否值得优先投入”来排序。

## 7.1 Priority A: Cache Accepted-Point Exact-2e Invariants

### Idea

把 exact 2e directional kernel 里所有与 accepted point 有关、与方向 \(v\) 无关的对象，在 operator 构造时一次性建好并缓存。

包括但不限于：

1. `active_pairs`
2. `active_pair_gradient_matrix`
3. accepted-point `dense_active_coefficients`
4. accepted-point `accepted_pair_products`
5. `base_pair_gradients`

### Why it matters

这一步现在已经完成。当前代码中：

1. `active_pair_gradient_matrix`
2. `accepted_pair_products`
3. `accepted_base_pair_gradients`

都已经进入 accepted-point cache，不再在每次 `H v` 中重建。

### Expected gain

这一步已经兑现成了稳定路径的实际收益。结合后续去掉的 fallback 基线冗余 probe，当前稳定 `exact_ctx` 路径已经从此前约 `15.98 s` 降到 `12.12 s`。

### Judgment

\[
\boxed{\text{这一步已经完成；后续不应再把“补 accepted-point exact-2e cache”当成主方向。}}
\]

---

## 7.2 Priority B: Make the Exact-2e Directional Kernel Support-Local

### Idea

当前

```cpp
build_mixed_ao_pair_to_active_pair_coefficients(...)
```

和后续 exact AO pair kernel 的作用，仍然是对几乎整个 AO pair space 做全扫。

但 nonredundant reduced direction 本质上只作用在少量 block-local orbital rotations 上，因此方向对应的

\[
\delta C
\]

其实是高度局域的。

因此，可以把 mixed pair coefficient 构造改成：

\[
\text{only touched orbitals / only touched AO-pairs}
\]

的局部更新，而不是每次都遍历整个 dense AO-pair 空间。

### Why it matters

这是当前最有可能带来“本质收益”的 exact-HVP 优化点，因为它不只是 cache 常量，而是在改变 directional kernel 的有效工作量。

### Expected gain

如果实现得好，这是最可能带来大幅收益的一项。对总 TN wall-time 的潜在提升，我会估计为：

\[
15\% \sim 35\%
\]

对于更大的 basis / active space，这个收益可能还会继续放大。

### Judgment

\[
\boxed{
\text{这是当前最值得赌的高收益优化点。}
}
\]

---

## 7.3 Priority C: Replace the Local Orbital Pullback Finite Difference

### Idea

这一步现在也已经完成。当前 `ExactOrbitalSecondOrderOperator::apply_reduced()` 末尾处理的是 accepted-point 固定上游梯度下的解析 directional pullback：

\[
\delta(J^{\mathrm T}) \lambda
\]

而不是先前那种局部双边有限差分近似。

### Why it matters

它的收益不仅在性能，还在稳定性：

1. 去掉局部有限差分，已经减少了 analytic-core HVP 内部的额外工作；
2. 但这还没有自动解决 analytic-core 端到端的稳定性问题，`F2` 上仍然有段错误。

### Expected gain

这一步现在更准确的表述应当是：

- 性能上的局部 FD 已经去掉；
- analytic-core 仍需继续做 accepted-point 线性化与稳定性清理。

### Judgment

\[
\boxed{
\text{这一步已经完成其“去掉局部 FD”的部分，但 analytic-core 仍未 production-ready。}
}
\]

---

## 7.4 Priority D: Build an Accepted-Point Linear Orbital-Backprop Operator

### Idea

对固定 accepted input，`ActiveSpaceOrbitalBackpropagator` 本质上是一个从上游 adjoint 到 orbital parameter gradient 的线性映射。

因此，可以把它改造成：

\[
(\Delta G_{\mathrm{aux}}, \Delta G_D)
\mapsto
\Delta g_{\mathrm{orb}}
\]

的 accepted-point 线性算子，而不是每次 HVP 都重新做：

1. normalization；
2. projector build；
3. inactive overlap inverse；
4. dense-to-sparse gather。

### Why it matters

当前代码里，orbital backprop 被调用：

1. 一次解析 directional backprop；
2. 两次局部 finite-difference pullback。

也就是说，这部分重复次数不止一次。

### Expected gain

如果和 Priority C 联合做，整体收益会明显高于单独做 C。

对总 TN wall-time 的保守估计是：

\[
5\% \sim 15\%
\]

### Judgment

\[
\boxed{
\text{这是一个中高收益、同时能把代码结构做正确的优化点。}
}
\]

---

## 7.5 Priority E: Reduce HVP Calls per Outer Step

### Idea

当前 `C6H6_full` 上，accepted TN step 通常使用 `4` 次 HVP。

因此，如果通过更好的 preconditioner、warm start、CG stopping rule、transported history 使用方式等手段，把平均 HVP 次数从

\[
4 \rightarrow 3
\]

则总收益会非常直接。

### Why it matters

若单步中 HVP 占主要成本，则

\[
25\%
\]

的 HVP 次数减少，可以转化为大约

\[
10\% \sim 20\%
\]

的总 wall-time 降低。

### Expected gain

这是一个：

- 风险中低；
- 不必改太多物理公式；
- 但可能立刻有效的优化方向。

### Judgment

\[
\boxed{
\text{这是最值得并行推进的 optimizer-level 优化。}
}
\]

---

## 8. What Is No Longer Worth Focusing On First

当前不应该再把主要精力放在：

1. same-spin structure builder 微调；
2. eigensolver 微调；
3. structure-basis gather 的边角成本；
4. 再去抠已经只占几十分之一秒的矩阵装配杂项。

原因很简单：这些项已经不在主导项里。

如果目标是继续明显降低单步时间，那么这些地方即使各自优化一倍，总收益也会非常有限。

---

## 9. Recommended Execution Order

如果下一阶段完全以性能为中心，我建议按下面顺序推进。

1. 先继续压稳定 fallback `exact_ctx`：
   避免 outer iteration 上的一切冗余 probe / 冗余构造。

2. 主攻 exact HVP 的非积分固定量缓存：
   orbital normalization、inactive overlap、projector、pullback fixed tensors。

3. 然后做 exact AO-H1E directional 专门化：
   减少通用 builder / backprop 接口在 inner HVP 中的常数。

4. 最后做真正高风险高收益的 exact-2e support-local / block-local directional kernel。

这个顺序的理由是：

- 第 1 步和第 2 步仍然属于低风险、可直接兑现 wall-time 的工程优化；
- 第 3 步开始真正进入 exact HVP 的通用 AO-ERI 路径专门化；
- 第 4 步最可能带来最大收益，但也最像新的算法开发工作。

---

## 10. Bottom Line

当前最重要的结论有三条。

第一，

\[
\boxed{
\text{exact nonredundant matrix-free TN 已经在总 wall-time 上赢过了 L-BFGS。}
}
\]

第二，

\[
\boxed{
\text{analytic-core 路径已经显示出进一步提速潜力，但还没有稳定到可默认开启。}
}
\]

第三，

\[
\boxed{
\text{后续真正值得投入的优化中心，不是 same-spin，而是 } T_{Hv} \text{ 与 } k_{\mathrm{cg}}.
}
\]

如果只用一句工程语言来总结，那么就是：

\[
\boxed{
\text{二阶路线已经证明值；现在要做的是把 exact HVP 做得更便宜、更稳定。}
}
\]
