# Exact Raw-VB Overlap Via A Layered Separator Recurrence

## 1. 目的与定位

本文档总结我们当前对 **exact raw-VB overlap** 的最新算法思路。这里讨论的对象不是
`pfaffian_vbscf` 的 AGP / pairing-state 近似，而是：

- 保持 strict exact raw-VB determinant expansion 的数值定义不变；
- 尝试用图分解与分层动态规划，减少必须显式处理的唯一 exact 子问题数；
- 目标是降低与

```math
N_{\mathrm{det}}^2
```

同阶相关的常数，而不是宣称把 general exact raw-VB 彻底变成多项式算法。

当前最可信的命题是：

> exact raw-VB overlap 也许不能在一般情形下摆脱组合复杂度，但有机会重写成一个
> **width-parameterized layered recurrence**，使真正需要处理的唯一 exact 子问题数
> 小于基线的 unique determinant-pair 数。

---

## 2. 与不同分子的关系

### 2.1 能否迁移到不同分子

可以迁移，但算法收益是 **molecule dependent** 的。

这套方法依赖于给定结构对 `(X,Y)` 的 **metric-aware component graph** 是否具有小的
separator width。更具体地说，收益取决于：

- active orbital overlap matrix 的局域性；
- 两个 raw-VB 结构叠加后的 union graph 组件如何被当前轨道度量耦合；
- 图是否存在小 separator，尤其是当前原型优先处理的 star-like 情形。

因此，这不是“只要换个分子就自动一样有效”的 black-box 加速器。它更像：

- 对所有分子都可以定义同样的 exact recurrence；
- 但只有在图宽

```math
w
```

较小、局域性较强时，才有显著压缩潜力。

### 2.2 当前阶段的现实判断

对任意分子，这条路线都可以做如下两步：

1. 构造 structure-pair 的 metric-aware component graph；
2. 统计 exact layered subproblem count 与 unique determinant-pair count 的比值。

于是每个新分子都可以被快速归类为：

- **promising**: 图宽小，separator recurrence 有压缩空间；
- **neutral**: 图宽中等，只能拿到有限常数收益；
- **unfavorable**: 图宽大，或 graph planning 不稳定，收益很弱。

也就是说，这套方法可以迁移到不同分子，但是否真正有用，需要先做图诊断，而不是先假定有效。

---

## 3. 基线 exact overlap 公式

给定两个 closed-shell raw-VB 结构 `X` 与 `Y`，记它们的 canonical determinant expansions 为

```math
\Phi_X = \sum_{D_L \in \mathcal{D}(X)} c_X(D_L)\, D_L,
```

```math
\Phi_Y = \sum_{D_R \in \mathcal{D}(Y)} c_Y(D_R)\, D_R.
```

则 exact overlap 为

```math
S_{XY}
=
\langle \Phi_X | \Phi_Y \rangle
=
\sum_{D_L \in \mathcal{D}(X)}
\sum_{D_R \in \mathcal{D}(Y)}
c_X(D_L)\, c_Y(D_R)\,
\det S^\alpha(D_L,D_R)\,
\det S^\beta(D_L,D_R).
```

这里：

- `D_L` 与 `D_R` 分别表示左右结构中的 canonical alpha/beta determinant term；
- `S^\alpha(D_L,D_R)` 与 `S^\beta(D_L,D_R)` 分别是 alpha 和 beta occupied orbital overlap submatrices。

基线算法的唯一工作对象是

```math
(D_L, D_R),
```

也就是 determinant-pair。

如果我们把

```math
N_{\mathrm{pair}}^{\mathrm{uniq}}
```

记为 unique determinant-pair 总数，那么当前你关心的唯一评判标准就是：

> 我们是否能把真正需要显式处理的唯一 exact 子问题数，压到
> `N_{\mathrm{pair}}^{\mathrm{uniq}}` 以下。

---

## 4. 图分解与 component ordering

### 4.1 support, union graph, component graph

对一对结构 `(X,Y)`：

1. 构造其 support orbital set；
2. 在 support 上叠加左右配对，得到 union graph；
3. 基于当前 active orbital overlap matrix，构造 metric-aware component graph。

当前原型首先关注 component graph 呈 star-like 的情形：

- 一个 root component；
- 若干 leaf components。

### 4.2 component-ordered support

为了让 exact recurrence 的边界状态显式化，我们将 support orbital 重排为：

```text
root block first, then leaf blocks in merge order.
```

这个重排不改变 exact overlap，只是把轨道排列改成更适合 separator recurrence 的形式。

若记重排后的 support 为

```math
\Omega = \Omega_0 \cup \Omega_1 \cup \cdots \cup \Omega_L,
```

其中：

- `\Omega_0` 是 root component；
- `\Omega_\ell` 是第 `\ell` 个 leaf component；

那么 exact overlap 的组合结构可以重写为：

- 每个 leaf 先形成一个 exact message bundle；
- 然后这些 bundles 在 root 上通过 frontier masks 合并；
- 最后乘以 root remainder 的 exact spin determinants。

---

## 5. 第一层：exact leaf message bundle

固定一对 root determinant terms：

```math
R_L = (\alpha_L^{(0)}, \beta_L^{(0)}), \qquad
R_R = (\alpha_R^{(0)}, \beta_R^{(0)}).
```

对于 leaf `\ell`，我们定义一个 exact message table：

```math
M_\ell^{(R_L,R_R)}
\bigl(
\mu_{\alpha}^{r},
\mu_{\alpha}^{c},
\mu_{\beta}^{r},
\mu_{\beta}^{c}
\bigr),
```

其中四个 masks 分别表示：

- `\mu_{\alpha}^{r}`: leaf 从右 root alpha positions 中选择的行位置；
- `\mu_{\alpha}^{c}`: leaf 从左 root alpha positions 中选择的列位置；
- `\mu_{\beta}^{r}`: leaf 从右 root beta positions 中选择的行位置；
- `\mu_{\beta}^{c}`: leaf 从左 root beta positions 中选择的列位置。

对给定的四元组 mask，`M_\ell` 定义为该 leaf 内部所有 exact determinant contributions 的总和：

```math
M_\ell^{(R_L,R_R)}(\mu)
=
\sum_{(d_L,d_R)\in\mathcal{L}_\ell(\mu)}
w_\ell(d_L,d_R;\mu),
```

其中：

- `\mu` 简记四个 mask 的集合；
- `\mathcal{L}_\ell(\mu)` 表示在 leaf `\ell` 内、并且恰好落到边界条件 `\mu` 的 local determinant-pairs；
- `w_\ell` 是相应的 exact zeroed-block determinant contribution。

关键点在于：

> leaf 内部很多不同的 exact determinant-pairs，只要它们诱导出相同的边界 mask 四元组，就可以在 leaf 内部先精确求和，形成同一个 message entry。

因此，一个 leaf 对后续 root merge 的真正输出不是许多 determinant-pairs，而是一张 exact message table，也就是一个 **leaf message bundle**。

---

## 6. 第二层：merge frontier recurrence

假设一共有 `L` 个 leaf components。定义 merge frontier state：

```math
F_k(U_\alpha^r,U_\alpha^c,U_\beta^r,U_\beta^c),
```

其中：

- `k` 表示已经合并了前 `k` 个 leaves；
- `U_\alpha^r, U_\alpha^c, U_\beta^r, U_\beta^c` 是当前已经被前 `k` 个 leaves 占用的 root masks。

初值为

```math
F_0(0,0,0,0) = 1.
```

递推关系为

```math
F_{k+1}(U')
=
\sum_{\mu \in \mathcal{A}_{k+1}(U')}
F_k(U)\,
M_{k+1}^{(R_L,R_R)}(\mu),
```

这里：

- `U'` 表示加入第 `k+1` 个 leaf 后的新累计 masks；
- `\mathcal{A}_{k+1}(U')` 表示所有与旧 state `U` 相容、并且不会与已占用 root positions 冲突的 message masks。

因此：

> 如果两条不同的 determinant history 在某一步落到同一个 frontier state
> `F_k(U_\alpha^r,U_\alpha^c,U_\beta^r,U_\beta^c)`，它们后续的 exact continuation 完全相同，可以在该 state 上合并。

这一步就是第二层 exact compression。

---

## 7. 第三层：root remainder 与 spin-kernel

当所有 leaves 合并完成后，root 上剩余的位置由补集 masks 给出：

```math
\bar U_\alpha^r = \mathbf{1}_\alpha^r \setminus U_\alpha^r,\qquad
\bar U_\alpha^c = \mathbf{1}_\alpha^c \setminus U_\alpha^c,
```

```math
\bar U_\beta^r = \mathbf{1}_\beta^r \setminus U_\beta^r,\qquad
\bar U_\beta^c = \mathbf{1}_\beta^c \setminus U_\beta^c.
```

这些 remainder 对应的 exact root determinants 为

```math
K_\alpha(\bar U_\alpha^r,\bar U_\alpha^c)
=
\det S^\alpha_{\mathrm{root}}(\bar U_\alpha^r,\bar U_\alpha^c),
```

```math
K_\beta(\bar U_\beta^r,\bar U_\beta^c)
=
\det S^\beta_{\mathrm{root}}(\bar U_\beta^r,\bar U_\beta^c).
```

在当前原型中，我们还会遇到 leaf-local 的 zeroed-root-block determinants。于是可复用的 spin-kernel 子问题主要分成两类：

- root remainder spin determinants；
- zeroed-block spin determinants。

这些 kernel 仍然是 exact determinant 子问题，但已经不是一一对应原始 determinant-pair。

---

## 8. 分层 exact overlap 总式

综合三层之后，固定 root determinant terms `(R_L,R_R)` 的 contribution 可写成

```math
S_{XY}^{(R_L,R_R)}
=
\sum_{U_\alpha^r,U_\alpha^c,U_\beta^r,U_\beta^c}
F_L(U_\alpha^r,U_\alpha^c,U_\beta^r,U_\beta^c)\,
K_\alpha(\bar U_\alpha^r,\bar U_\alpha^c)\,
K_\beta(\bar U_\beta^r,\bar U_\beta^c)\,
\sigma(R_L,R_R,U),
```

其中 `\sigma` 是由 canonicalization parity 与 root-term coefficients 共同给出的符号和系数因子。

于是整体 overlap 为

```math
S_{XY}
=
\sum_{R_L \in \mathcal{R}(X)}
\sum_{R_R \in \mathcal{R}(Y)}
S_{XY}^{(R_L,R_R)}.
```

关键不是这个公式看起来是否更短，而是：

> 在这个分层写法里，真正的唯一 exact 子问题已不再是 determinant-pair，而是：
> leaf message bundles、merge frontier states、以及 spin-kernel 子问题。

---

## 9. 当前最有意义的 exact 计数量

为了避免把内部实现统计误当成结论，当前最有意义的 exact 计数 proxy 是：

```math
N_{\mathrm{layered}}
=
N_{\mathrm{bundle}}
+ N_{\mathrm{merge}}
+ N_{\mathrm{spin\text{-}kernel}},
```

其中：

- `N_{\mathrm{bundle}}` 是 unique exact leaf message bundle 数；
- `N_{\mathrm{merge}}` 是 unique exact merge frontier state 数；
- `N_{\mathrm{spin\text{-}kernel}}` 是 unique root / zeroed-block spin determinant 子问题数。

然后与基线的

```math
N_{\mathrm{pair}}^{\mathrm{uniq}}
```

比较比值：

```math
\rho_{\mathrm{layered}}
=
\frac{N_{\mathrm{layered}}}{N_{\mathrm{pair}}^{\mathrm{uniq}}}.
```

如果

```math
\rho_{\mathrm{layered}} < 1,
```

则说明这套 exact layered recurrence 至少在“唯一 exact 子问题数”上小于 determinant-pair 基线。

---

## 10. 当前实测信号

在 `C6H6_full` 的当前测试中，我们已经观察到：

```math
\rho_{\mathrm{layered}}
\approx 0.62
\quad \text{(input overlap)},
```

```math
\rho_{\mathrm{layered}}
\approx 0.59
\quad \text{(optimized overlap)}.
```

这意味着：

- layered exact subproblem proxy 大约压到原来的 `59\%` 到 `62\%`；
- 对应的理想压缩倍率约为

```math
\frac{1}{0.59} \sim 1.7.
```

这不是数量级的革命性提升，但它第一次说明：

> exact raw-VB 路线并非完全没有结构可用；
> 真正可复用的唯一 exact 子问题数，可能稳定小于 unique determinant-pair baseline。

---

## 11. 与不同 SCF 步之间的复用

### 11.1 当前原型

当前 metric-aware graph planning 依赖当前 active orbital overlap matrix。因此：

- graph decomposition；
- root / leaf 选择；
- width；
- 哪些 structure pair 落入当前可处理情形；

都会随 SCF 轨道变化。

因此，**当前原型下每个 SCF 步都应重新判断 graph planning**。

### 11.2 可以静态复用的部分

尽管如此，仍有许多部分是 run-level static 的：

- 每个 raw-VB structure 的 canonical determinant terms；
- coefficient lookup；
- support orbital set；
- union graph 的纯组合拓扑；
- masks 的枚举空间；
- 若固定 separator plan，则 DP state-space 模板也可复用。

### 11.3 更理想的长期目标

更好的架构应拆成：

1. **static separator plan**

   - 尽量仅依赖结构组合拓扑；
   - 在整个 SCF 过程中尽量固定。

2. **dynamic numeric kernels**

   - 每步更新 leaf message 数值；
   - 每步更新 root / zeroed-block spin determinants；
   - 但不重建整套状态空间。

如果这一步能实现，那么总 SCF 过程中的收益会明显大于单步内的局部压缩收益。

---

## 12. 当前已知局限

1. 当前最成熟的是 **overlap** 路径；Hamiltonian 和 gradient 尚未按同样的 layered exact recurrence 完整落地。

2. 当前最有希望的 exact recurrence 是 **layered** 的，而不是 monolithic exact state key。后者在某些轨道上会膨胀到比 determinant-pair 更大。

3. 当前原型最自然处理的是 star-like component graph。更一般的 graph 需要进一步推广到 tree / separator decomposition。

4. 当前结果只说明“存在真实压缩信号”，还不等于已经实现了最终可用的 overlap evaluator。

---

## 13. 下一步实现顺序

从实现角度，最合理的顺序应是：

1. 先把 exact overlap 的三层缓存真正实现：

   - leaf bundle cache；
   - merge frontier cache；
   - spin-kernel cache。

2. 在 `C6H6_full` 上验证：

   - exact 数值与 determinant-basis reference 严格一致；
   - wall time 是否真的下降。

3. 若 overlap 成功，再把同样的 layered recurrence 推广到：

   - one-electron matrix elements；
   - two-electron matrix elements；
   - 最终的 orbital gradient。

在这之前，不应对 Hamiltonian / gradient 做过早实现。

---

## 14. 一句话总结

我们当前的 exact raw-VB overlap 思路，不是再去寻找一个“万能单态编码”，而是：

```math
\text{determinant-pair sum}
\;\Longrightarrow\;
\text{leaf exact bundles}
\;+\;
\text{merge frontier DP}
\;+\;
\text{reusable spin kernels}.
```

如果这套 layered recurrence 能完全落地，那么它的价值不在于宣称“解决了 general exact VB 的指数问题”，而在于：

> 在 strict exact raw-VB 定义不变的前提下，把真正需要处理的唯一 exact 子问题数压低到 determinant-pair 基线以下，并将复杂度更直接地绑定到 metric-aware separator structure 上。
