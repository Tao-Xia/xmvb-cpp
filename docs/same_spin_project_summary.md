# Same-Spin Project Summary

## 1. 项目目标

本项目要解决的问题是：

1. full determinant space 在很多 VBSCF 体系里接近
   $$
   D_p = (\alpha_{i(p)}, \beta_{j(p)})
   $$
   的笛卡尔积结构；
2. 传统实现仍然在 full determinant pair 空间上工作，很多 same-spin 量被重复计算；
3. 即使已经做了 unique same-spin cache，如果 backward 仍然沿着 full determinant pair 遍历，主复杂度骨架仍然是
   $$
   O(N_{\mathrm{det}}^2),
   $$
   优化收益会被大幅吞掉。

本项目的核心目标有两层：

1. 在 forward 里把 same-spin determinant kernel 变成 unique alpha / unique beta pair 复用；
2. 在 forward / backward 里把能分离到 unique alpha / unique beta 空间的主路径都改成 matrix form，避免再挂在 full determinant pair 外循环上。

这里的 "same-spin / one-electron" 是同一条分离通道，数学上对应
$$
H^\alpha S^\beta + S^\alpha H^\beta,
$$
其中 $H^\sigma$ 包含该自旋块的一电子项和同自旋二电子项。

## 2. 符号

记

$$
N_{\mathrm{det}} = \text{full determinant 数},
$$

$$
N_\alpha = \text{unique alpha determinant 数},
\qquad
N_\beta = \text{unique beta determinant 数}.
$$

第 $p$ 个 full determinant 写成

$$
D_p = (\alpha_{i(p)}, \beta_{j(p)}).
$$

对选中的 state-average 态 $n$，定义 unique-spin 系数矩阵

$$
C^{(n)} \in \mathbb{R}^{N_\alpha \times N_\beta},
\qquad
C^{(n)}_{ij} = c^{(n)}_{(\alpha_i,\beta_j)}.
$$

定义 ordered unique same-spin pair 的标量矩阵

$$
S^\alpha, \; S^\beta \in \mathbb{R}^{N_\alpha \times N_\alpha}, \mathbb{R}^{N_\beta \times N_\beta},
$$

$$
H^{\alpha,\mathrm{reg}}, \; H^{\beta,\mathrm{reg}},
$$

以及 singular partner 通道

$$
H^{\alpha,\mathrm{sing}}, \; H^{\beta,\mathrm{sing}}.
$$

## 3. 原始问题在哪里

### 3.1 full determinant pair 的 same-spin 重复

对两个 full determinants

$$
D_p = (\alpha_{i(p)}, \beta_{j(p)}),
\qquad
D_q = (\alpha_{i(q)}, \beta_{j(q)}),
$$

其 full overlap 精确分解为

$$
S_{pq}
=
S^\alpha_{i(p)i(q)} S^\beta_{j(p)j(q)}.
$$

same-spin / one-electron Hamiltonian 部分精确分解为

$$
H_{pq}^{\mathrm{same}}
=
H^\alpha_{i(p)i(q)} S^\beta_{j(p)j(q)}
+
S^\alpha_{i(p)i(q)} H^\beta_{j(p)j(q)}.
$$

因此 expensive 的 same-spin determinant kernel 本来就应该只按 unique alpha pair 和 unique beta pair 计算一次。

### 3.2 只做 cache 还不够

如果只是在 pair evaluator 内部缓存 unique same-spin payload，而 structure builder / backward 仍然外层完整遍历所有 unordered full determinant pairs，那么：

1. same-spin kernel 计算次数会下降；
2. 但外层主循环仍然是
   $$
   O(N_{\mathrm{det}}^2).
   $$

这就是早期 same-spin cache "忽略了很多重复行列式计算，但总单步时间没有本质下降" 的根本原因。

## 4. 本项目完成的核心改动

### 4.1 unique same-spin pair cache

当前代码会先把 full determinant 列表压缩成 unique alpha / unique beta occupied string：

```text
alpha_det -> alpha_reuse_table
beta_det  -> beta_reuse_table
```

然后只为 ordered unique pairs 构建

$$
N_\alpha^2 + N_\beta^2
$$

个 `SpinDeterminantPairEvaluation`，而不是为每个 full determinant pair 重算 same-spin 子问题。

对 `test/10698_RI.xmi` 这个 400 determinant 的例子，当前观测到

$$
N_{\mathrm{det}} = 400,
\qquad
N_\alpha = 20,
\qquad
N_\beta = 20.
$$

如果按 full determinant unordered pairs 做 two-spin same-spin kernel，总数约为

$$
2 \cdot \frac{N_{\mathrm{det}}(N_{\mathrm{det}}+1)}{2}
=
160400.
$$

而 unique same-spin cache 只需要

$$
N_\alpha^2 + N_\beta^2 = 20^2 + 20^2 = 800.
$$

理论复用倍率为

$$
\rho
=
\frac{160400}{800}
=
200.5.
$$

### 4.2 exact 路径：same-spin phi cache 也被打通

早期 matrix-form backward 主要是围绕 RI 做的，exact 没有完全走进 same-spin phi cache。

现在 exact 和 RI 都会在 regular ordered unique same-spin pair 上预先缓存

$$
\phi^\sigma_{ij},
\qquad
\frac{\partial \phi^\sigma_{ij}}{\partial X},
\qquad
\sigma \in \{\alpha,\beta\}.
$$

这一步完成后，same-spin backward 就不需要再回到 full determinant pair 上重复计算 same-spin `phi`。

### 4.3 exact backward：从 full determinant pair 改成 unique-spin matrix form

这是本项目的关键点。

#### 旧做法

旧 exact backward 先在 full determinant pair 空间重建

$$
W^{(H)}_{pq}, \qquad W^{(S)}_{pq},
$$

再把这些权重散射回 unique alpha / beta pair 通道。即使 same-spin kernel 已缓存，外层依然保留 full determinant pair 的主循环。

#### 新做法

当前实现直接在 unique alpha / beta 空间上做压缩。

对每个 selected state $n$，用 $C^{(n)}$ 直接构造：

$$
W_\alpha^{(H)}
=
\sum_n w_n \, C^{(n)} S^\beta [C^{(n)}]^T,
$$

$$
W_\alpha^{(S)}
=
\sum_n \left(-w_n E_n\right) C^{(n)} S^\beta [C^{(n)}]^T,
$$

$$
T_\alpha^{\mathrm{reg}}
=
\sum_n w_n \, C^{(n)} H^{\beta,\mathrm{reg}} [C^{(n)}]^T,
$$

$$
T_\alpha^{\mathrm{sing}}
=
\sum_n w_n \, C^{(n)} H^{\beta,\mathrm{sing}} [C^{(n)}]^T.
$$

beta 通道完全对称：

$$
W_\beta^{(H)}
=
\sum_n w_n \, [C^{(n)}]^T S^\alpha C^{(n)},
$$

$$
W_\beta^{(S)}
=
\sum_n \left(-w_n E_n\right) [C^{(n)}]^T S^\alpha C^{(n)},
$$

$$
T_\beta^{\mathrm{reg}}
=
\sum_n w_n \, [C^{(n)}]^T H^{\alpha,\mathrm{reg}} C^{(n)},
$$

$$
T_\beta^{\mathrm{sing}}
=
\sum_n w_n \, [C^{(n)}]^T H^{\alpha,\mathrm{sing}} C^{(n)}.
$$

也就是说，exact same-spin backward 的核心已经不再是 full determinant pair scatter，而是 BLAS-3 风格的

$$
C B C^T,
\qquad
C^T A C.
$$

在这一步之后，才对 ordered unique alpha / beta pairs 做最终 AO 梯度散射。

### 4.4 exact forward：structure builder 已经 matrix-form 化

仅仅做 unique same-spin cache 还不够，因为如果 structure builder 仍然完整遍历 unordered full determinant pairs，那么 forward 的主循环骨架仍然是

$$
O(N_{\mathrm{det}}^2).
$$

当前 `build()` 路径在 same-spin cache 可用时，已经直接把 structure-level 矩阵装配写成 unique alpha / unique beta 空间上的矩阵乘法，不再进入 full determinant pair 外循环。

对每个 structure $I$，先构造一个 unique-spin 系数矩阵

$$
C_I \in \mathbb{R}^{N_\alpha \times N_\beta},
\qquad
(C_I)_{ab}
=
\sum_{d:\,(a_d,b_d)=(a,b)} t_{I,d}.
$$

于是：

$$
S_{IJ}
=
\langle C_I,\; S^\alpha C_J [S^\beta]^T \rangle_F,
$$

$$
H^{1e}_{IJ}
=
\langle C_I,\; H^{1e,\alpha} C_J [S^\beta]^T \rangle_F
+
\langle C_I,\; S^\alpha C_J [H^{1e,\beta}]^T \rangle_F,
$$

$$
H^{\mathrm{same}}_{IJ}
=
\langle C_I,\; H^\alpha C_J [S^\beta]^T \rangle_F
+
\langle C_I,\; S^\alpha C_J [H^\beta]^T \rangle_F.
$$

exact 反自旋 forward 也已经进入同一套 structure-level matrix form。
记按 packed active pair $P$ 分组后的 alpha first-order 通道矩阵为

$$
U_\alpha^{(P)} \in \mathbb{R}^{N_\alpha \times N_\alpha},
$$

beta 侧经 exact kernel 作用后的 projected image 为

$$
\widetilde{U}_\beta^{(P)} \in \mathbb{R}^{N_\beta \times N_\beta}.
$$

则反自旋结构矩阵元可写成

$$
H^{\alpha\beta}_{IJ}
=
\sum_P
\left\langle
C_I,\;
U_\alpha^{(P)} C_J [\widetilde{U}_\beta^{(P)}]^T
\right\rangle_F.
$$

因此当前 production `build()` 路径里，exact 反自旋 forward 已经不再保留 full determinant pair 的外层主循环。

需要保留旧 pair loop 的只剩 `build_with_pair_evaluations()` 这类诊断接口，因为它们显式要求 materialize 每个 `FullDeterminantPairEvaluation`。

### 4.5 RI 路径：same-spin matrix-form backward 与表示无关

RI 路径和 exact 的 backward 数学结构是同一套，差别只在 pair payload 的来源。

当前 RI same-spin 路径有两种前向准备方式：

1. 如果 `ActiveSpaceTwoElectronResult` 里已经带有 materialized packed active ERI，就直接复用 packed path；
2. 如果只有 RI factor，则先重建 packed active ERI 一次，再建立 dense active-pair kernel。

因此，RI 的 same-spin cache / backward 并不是单独写了一套新代数，而是把 pair payload 构好以后，继续走同一个 unique-spin matrix-form adjoint。

## 5. 计算标度分析

### 5.1 cache 之前

如果 same-spin forward / backward 都直接挂在 full determinant pair 循环里，则主复杂度骨架是

$$
O(N_{\mathrm{det}}^2).
$$

更准确地说，same-spin kernel 的总工作量随 full unordered pair 数增长：

$$
O\!\left(\frac{N_{\mathrm{det}}(N_{\mathrm{det}}+1)}{2}\right).
$$

### 5.2 只做 unique same-spin cache 之后

如果仅仅把 same-spin determinant kernel 换成 unique pair cache，那么 expensive same-spin kernel 的计算次数下降为

$$
O(N_\alpha^2 + N_\beta^2),
$$

但是 structure builder 和旧 backward 外层仍可能保持

$$
O(N_{\mathrm{det}}^2).
$$

这解释了为什么 "cache 明明省掉了大量重复 same-spin determinant 计算，但总时间看起来没有按比例下降"。

### 5.3 当前 exact/RI forward structure assembly 的标度

在 cache-enabled 的 production `build()` 路径里，structure builder 已经不再沿 full determinant pair 遍历。

same-spin / one-electron 通道的结构层装配主要是若干次

$$
A C_I B^T
$$

和 Frobenius 内积，因此标度可写成

$$
O\!\left(
N_{\mathrm{str}} \left(
N_\alpha^2 N_\beta + N_\alpha N_\beta^2
\right)
\right),
$$

其中 $N_{\mathrm{str}}$ 是 structure 数。

exact 反自旋 forward 在此基础上还要对 packed active pair 通道求和：

$$
O\!\left(
N_{\mathrm{str}} \, N_{\mathrm{pair}}
\left(
N_\alpha^2 N_\beta + N_\alpha N_\beta^2
\right)
\right),
$$

其中

$$
N_{\mathrm{pair}} = \frac{n_{\mathrm{act}}(n_{\mathrm{act}}+1)}{2}.
$$

这已经不再由 $N_{\mathrm{det}}^2$ 控制；它的主依赖变成了 unique spin 维数和 active packed-pair 维数。

### 5.4 当前 exact/RI matrix-form backward 的标度

当前 same-spin / one-electron backward 可以拆成三部分：

#### 1. unique pair payload 构造

same-spin pair cache：

$$
O(N_\alpha^2 + N_\beta^2).
$$

same-spin `phi` cache 也只在 regular ordered unique pairs 上构造一次，因此同阶。

#### 2. selected-state 权重压缩

每个 state 需要做数次

$$
C^{(n)} B [C^{(n)}]^T
\quad \text{或} \quad
[C^{(n)}]^T A C^{(n)}
$$

矩阵乘法，因此代数压缩成本为

$$
O\!\left(
N_s \left(
N_\alpha^2 N_\beta + N_\alpha N_\beta^2
\right)
\right),
$$

其中 $N_s$ 是 selected state 数。

#### 3. unique pair 到 AO 梯度的最终散射

这一部分仍然按 ordered unique pairs 遍历，因此 pair 数是

$$
O(N_\alpha^2 + N_\beta^2).
$$

但每个 pair 内部还要做 cofactor、overlap gradient、same-spin two-electron gradient 的局部更新。
如果把自旋电子数视为常数，那么它对 unique pair 数是线性的，即仍可写成

$$
O(N_\alpha^2 + N_\beta^2).
$$

如果把电子数也显式算进去，更保守地可以写成

$$
O\!\left(N_\alpha^2 f_\alpha + N_\beta^2 f_\beta\right),
$$

其中 $f_\sigma$ 是单个 unique pair 的局部 cofactor / scatter 成本。

### 5.5 目前尚未做到的极致分离

对 same-spin backward 而言，当前主路径已经不再受 full determinant pair 数控制，这是项目的本质改进。

但它还没有化成严格的

$$
O(N_\alpha^2 + N_\beta^2)
$$

总代价，因为 selected-state 系数矩阵 $C^{(n)}$ 把 alpha / beta 空间耦合在一起，权重压缩步骤仍然含有

$$
N_\alpha^2 N_\beta + N_\alpha N_\beta^2
$$

的矩阵乘法成本。

这已经比 full determinant pair 空间好得多，但不是数学意义上的极限分离。

## 6. benchmark 分析

下面给出当前最有代表性的 benchmark。

### 6.1 大复用例子：`test/10698_RI.xmi`

这个输入 deck 本身是 RI deck，即 `$ctrl` 里写的是 `int=RI`。

因此需要区分两种运行方式：

1. `--standard-two-electron-mode ri`
   表示按 deck 语义走原生 RI；
2. `--standard-two-electron-mode exact`
   表示在 RI deck 上强制覆盖到 exact。

下面两组 benchmark 都是在同一个 `10698_RI.xmi` 上跑的，只是 two-electron mode 不同。

#### 几何规模

对这个体系，当前观测到

$$
N_{\mathrm{det}} = 400,
\qquad
N_\alpha = 20,
\qquad
N_\beta = 20.
$$

same-spin cache 构建规模为 800 个 ordered unique same-spin pairs，cache build 时间约为

$$
1.6 \text{ ms} \sim 2.0 \text{ ms}.
$$

旧的

$$
N_\alpha^2 N_\beta^2
$$

反自旋标量 cache 现在默认不再 materialize；production forward/backward 直接复用 same-spin pair cache 里的 projected channels。

#### exact override：稳态单步

命令：

```bash
source scripts/xmvb_cpp_runtime_env.sh
prepare_xmvb_cpp_runtime_env build/src/xmvb-cpp.exe "$(pwd)"
OMP_NUM_THREADS=1 XMVB_CPP_LOG_OBJECTIVE_PROGRESS=1 \
  build/src/xmvb-cpp.exe test/10698_RI.xmi \
  --optimizer-backend lbfgspp \
  --max-iterations 1 \
  --standard-two-electron-mode exact
```

取同一进程内后续调用的稳态数据，可读出：

| stage | time (s) |
| --- | ---: |
| total | 6.378570 |
| active_space | 4.533779 |
| orbital_prepare | 0.025956 |
| ao_h1e | 1.575131 |
| active_h1e | 0.000106 |
| active_2e | 2.872926 |
| structure | 0.042711 |
| eigensolver | 0.010750 |
| active_adjoint | 0.001632 |
| matrix_backprop | 0.000518 |
| active_2e_backprop | 0.006366 |
| ao_h1e_backprop | 1.828812 |
| orbital_backprop | 0.007258 |

same-spin 相关的关键结论是：

1. `same_spin_phi_cache` 只有约 `0.00029 s`；
2. `active_adjoint` 只有约 `0.0016 s`；
3. structure builder 当前已经是 unique-spin matrix-form forward，所以对这个 case 只有约 `0.043 s`，并且不再由 full determinant pair 数主导。

所以 exact 单步里，same-spin 已经完全不是瓶颈。

#### 原生 RI：稳态单步

命令：

```bash
source scripts/xmvb_cpp_runtime_env.sh
prepare_xmvb_cpp_runtime_env build/src/xmvb-cpp.exe "$(pwd)"
OMP_NUM_THREADS=1 XMVB_CPP_LOG_OBJECTIVE_PROGRESS=1 \
  build/src/xmvb-cpp.exe test/10698_RI.xmi \
  --optimizer-backend lbfgspp \
  --max-iterations 1 \
  --standard-two-electron-mode ri
```

同样取稳态数据：

| stage | time (s) |
| --- | ---: |
| total | 0.712549 |
| active_space | 0.356606 |
| orbital_prepare | 0.026364 |
| ao_h1e | 0.215693 |
| active_h1e | 0.000126 |
| active_2e | 0.091091 |
| structure | 0.007399 |
| eigensolver | 0.010544 |
| active_adjoint | 0.001609 |
| matrix_backprop | 0.000463 |
| active_2e_backprop | 0.111635 |
| ao_h1e_backprop | 0.236437 |
| orbital_backprop | 0.006520 |

额外要注意：

1. RI 首次进入同一进程时还有一次 `ao_ri_cache` 冷启动成本，这个例子里约 `5.44 s`；
2. 冷启动后，稳态单步不再承担这一成本。

对 same-spin 本项目而言，RI 的结论和 exact 完全一致：same-spin adjoint 已经降到毫秒级，不再是主瓶颈。

### 6.2 exact deck 的辅助 benchmark：`src/test_molecule/C6H6.xmi`

`src/test_molecule/C6H6.xmi` 本身就是 exact deck，因为 `$ctrl` 写的是

```text
int=libcint
```

命令：

```bash
OMP_NUM_THREADS=1 \
  build/src/benchmark_cpp_orbital_eval src/test_molecule/C6H6.xmi \
  --standard-two-electron-mode exact \
  --repeat 1 --warmup 0
```

这里要注意：`benchmark_cpp_orbital_eval` 调用的是 `CppOrbitalGradientEvaluator::evaluate()`，
因此 `mean_total_dt` 不是优化器 accepted-step 的严格 wall time 口径；但其中各个 stage 的相对占比仍然具有参考价值。

测得：

| metric | value |
| --- | ---: |
| mean_total_dt | 0.768793 s |
| mean_active_space_dt | 0.542181 s |
| mean_ao_h1e_dt | 0.111753 s |
| mean_active_2e_dt | 0.417185 s |
| mean_structure_dt | 0.000073 s |
| mean_active_adjoint_dt | 0.002051 s |
| mean_active_2e_backprop_dt | 0.001582 s |
| mean_ao_h1e_backprop_dt | 0.111109 s |

这个例子里的 unique reuse 不像 `10698_RI.xmi` 那么极端，但 same-spin adjoint 仍然只有约 `2 ms`，说明 same-spin 项目已经把 backward 压到非常低的水平。

### 6.3 benchmark 总结

目前性能图景已经很清楚：

1. same-spin cache build 时间大约是毫秒级；
2. same-spin phi cache、same-spin matrix-form adjoint、以及 structure-level matrix-form forward 都已经是毫秒级；
3. 当前单步主成本不再是 same-spin，而是：
   - exact: `active_2e` forward 和 `ao_h1e_backprop`
   - RI: `ao_h1e_backprop`、`active_2e_backprop`，以及首次冷启动的 `ao_ri_cache`

## 7. 正确性验证

当前已完成的数值验证包括：

### 7.1 exact active-space two-electron gradient

命令：

```bash
OMP_NUM_THREADS=1 \
  build/src/check_cpp_active_space_gradient src/test_molecule/F2.xmi \
  --standard-two-electron-mode exact \
  --component two_electron \
  --count 4 --step 1e-6
```

当前前几项误差量级约为

$$
10^{-8}.
$$

### 7.2 exact orbital gradient

命令：

```bash
OMP_NUM_THREADS=1 \
  build/src/check_cpp_orbital_gradient src/test_molecule/F2.xmi \
  --standard-two-electron-mode exact \
  --count 2 --step 1e-6
```

当前前几项误差量级约为

$$
10^{-9} \sim 10^{-8}.
$$

因此，same-spin exact matrix-form 主路径目前的正确性是成立的。

## 8. 当前项目状态

### 8.1 已经完成的部分

可以认为以下目标已经完成：

1. unique same-spin pair cache 已经打通；
2. exact 和 RI 都已经进入 same-spin phi cache；
3. production `build()` 的 structure builder 已经改成 unique-spin matrix-form forward，exact 反自旋 forward 也不再保留 full determinant pair 外循环；
4. same-spin / one-electron backward 已经从 full determinant pair adjoint 主循环中抽离，改成 unique-spin matrix form；
5. 当前 same-spin 路径已经不再是单步瓶颈。

### 8.2 还没有完全闭环的部分

当前仍然存在一个明确的边角：

1. matrix-form same-spin backward 只支持 regular pair；
2. 如果 singular partner channel 上出现非零 selected-state 权重，当前实现会主动报错，而不是给出一般化的 matrix-form 结果。

换句话说，当前完成的是：

$$
\text{regular same-spin 主路径}
$$

已经完全 matrix-form 化；

而不是

$$
\text{所有 same-spin 数学分支}
$$

都已经无条件 matrix-form 化。

### 8.3 对整个项目的判断

如果问题是：

> same-spin 这个项目是否已经做到足够成熟，可以暂时封板？

我的判断是：

**可以。**

理由是：

1. 它已经完成了最本质的结构性改进；
2. 正确性检查已经通过；
3. benchmark 已经显示 same-spin 不再主导单步时间。

因此，后续如果目标是继续降低当前最快版本的单步时间，优化重点应该转移到：

1. exact `active_space_two_electron_builder.cpp`
2. exact `ao_effective_one_electron_builder.cpp`
3. exact `ao_effective_one_electron_backpropagator.cpp`
4. RI 的 AO 层前后向算子

而不是继续在 same-spin adjoint 上投入主要精力。
