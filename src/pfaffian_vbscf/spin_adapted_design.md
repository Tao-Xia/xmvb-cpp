# Spin-Adapted Pfaffian-VBSCF Design

## 1. 目标与基本判断

我们已经完成了以下 primitive kernel 主干：

- closed-shell 解析矩阵元与梯度；
- general fixed-`M_s` open-shell 解析矩阵元与梯度；
- 生产路径中的矩阵构建与 active-space gradient 接入。

因此，下一步实现 spin-adapted 的正确策略，不是重新推导一套新的
spin-adapted two-electron kernel，而是：

- 保留现有 primitive fixed-`M_s` kernel 作为唯一底层矩阵元引擎；
- 在其外层增加一个 spin-coupling / recoupling 层；
- 通过线性组合把 primitive fixed-`M_s` states 收缩为 pure-`S`
  spin-adapted states。

这一步完成后，方法学主干将变为：

1. closed-shell
2. general fixed-`M_s` open-shell
3. spin-adapted open-shell

后续主线即可转向：

- 性能优化
- 分子体系测试
- selected-VB
- 化学问题和文章组织

---

## 2. 核心设计决策

### 2.1 不做 direct spin-adapted kernel

第一版 spin-adapted 不应尝试直接推导 pure-`S` Pfaffian matrix element 的
专用 closed form。原因是：

- 现有 general fixed-`M_s` primitive kernel 已经正确且可微；
- spin-adapted 的新增内容本质上是自旋耦合线性组合，而不是电子积分核；
- 如果现在去做 direct spin-adapted kernel，会把工作重新推回理论主推导，
  并打断当前“方法完成后转向性能和应用”的节奏。

因此第一版采用：

```math
|\Psi_I^{SA}\rangle = \sum_{a=1}^{N_{\mathrm{prim}}} C_{aI}\,|\phi_a^{(M_s)}\rangle
```

其中：

- `|\phi_a^{(M_s)}\rangle` 是 primitive fixed-`M_s` Pfaffian-VB state；
- `C` 是 primitive-to-spin-adapted coefficient matrix。

### 2.2 保持现有 primitive kernel 不变

现有以下文件应继续作为 primitive engine：

- `src/pfaffian_vbscf/kernel/pf_fixed_ms_open_shell_kernel.cpp`
- `src/pfaffian_vbscf/matrices/pf_matrix_builder.cpp`
- `src/pfaffian_vbscf/scf/pf_active_grad_eval.cpp`

spin-adapted 的实现不应破坏这些文件的 primitive 语义。新增逻辑应位于：

- basis construction 层
- matrix projection 层
- gradient weight lifting 层

---

## 3. 数学框架

### 3.1 primitive 到 spin-adapted 的矩阵投影

设 primitive basis 大小为 `N_p`，spin-adapted basis 大小为 `N_s`。

- primitive overlap matrix: `S^{(p)} \in R^{N_p \times N_p}`
- primitive Hamiltonian matrix: `H^{(p)} \in R^{N_p \times N_p}`
- coefficient matrix: `C \in R^{N_p \times N_s}`

则 spin-adapted matrices 为：

```math
S^{(sa)} = C^T S^{(p)} C
```

```math
H^{(sa)} = C^T H^{(p)} C
```

这一步完全是外层线性代数，不涉及新的轨道积分公式。

### 3.2 generalized eigenproblem

spin-adapted SCF 子空间上求解：

```math
H^{(sa)} d = E\, S^{(sa)} d
```

其中 `d \in R^{N_s}` 是 spin-adapted CI 系数。

primitive 表示下的等价系数为：

```math
c = C d
```

这里 `c` 不是新的物理态定义，而是 recoupled spin-adapted eigenvector
在 primitive subspace 上的展开系数。

### 3.3 梯度 lifting

设在 spin-adapted generalized eigensolver 之后得到：

- `G_H^{(sa)} = \partial E / \partial H^{(sa)}`
- `G_S^{(sa)} = \partial E / \partial S^{(sa)}`

则 primitive matrix 的权重矩阵为：

```math
G_H^{(p)} = C\, G_H^{(sa)}\, C^T
```

```math
G_S^{(p)} = C\, G_S^{(sa)}\, C^T
```

于是对任意底层变量 `x`，有：

```math
\frac{\partial E}{\partial x}
=
\sum_{ab}
G_{H,ab}^{(p)} \frac{\partial H_{ab}^{(p)}}{\partial x}

\;+\;
\sum_{ab}
G_{S,ab}^{(p)} \frac{\partial S_{ab}^{(p)}}{\partial x}
```

这意味着：

- 现有 primitive pairwise gradient kernel 可直接复用；
- 不需要重新推导 spin-adapted Hamiltonian gradient；
- 只需要把 spin-adapted matrix gradient lift 回 primitive pair 权重。

### 3.4 对空间标度的影响

primitive pair kernel 仍然保持：

- one-/two-electron forward: `O(M^4)` 主标度
- adjoint: 与当前 primitive kernel 相同阶

新增的 spin-adapted 代价为：

- `C^T S C`
- `C^T H C`
- `C G C^T`

这些代价只依赖 basis 维度，不依赖活性轨道数 `M`。
因此它们不会改变我们已经建立的轨道标度结论。

换句话说：

- spin-adapted 会增加 basis-level 常数或组合维度代价；
- 不会把轨道积分主标度从 `O(M^4)` 打回去。

---

## 4. spin-adapted 系数如何生成

### 4.1 第一版不手写 Rumer / genealogical 公式

第一版建议不要先手写一套 closed-form spin-coupling coefficient 公式。

更稳妥的通用路线是：

1. 固定一组 open-shell spatial orbitals；
2. 枚举该组轨道上的所有 fixed-`M_s` spin strings；
3. 在这个自旋字符串空间中构造 `\hat S^2` 矩阵；
4. 对 `\hat S^2` 对角化，取目标 `S(S+1)` 本征空间；
5. 由该本征空间向量给出 spin-adapted coupling coefficients。

这样做的优点：

- 完全 general，不依赖某个特定 open-shell 数目的手工公式；
- 与现有 fixed-`M_s` primitive kernel 严格兼容；
- 后续若需要换成 Rumer / GUGA / genealogical coefficients，
  只需替换 coefficient builder，不需改 kernel。

### 4.2 维度为什么可控

若一个结构含有 `n_u` 个 open-shell electrons，在 fixed-`M_s` 下，
自旋字符串数目为：

```math
\binom{n_u}{n_\beta^{(u)}}
```

这只依赖 open-shell electron 数，而不依赖活性轨道数 `M`。

对我们当前关心的 VBSCF 场景，这个维度通常远小于 determinant space，
因此在第一版中完全可接受。

### 4.3 每个“空间结构”上局部做 spin coupling

最自然的做法不是在整个 primitive basis 上一次性构造 `\hat S^2`，
而是对每个相同空间结构的 open-shell pattern 局部构造 coupling space。

也就是：

- 先固定 open-shell spatial orbital pattern；
- 再在该 pattern 上生成 fixed-`M_s` primitive states；
- 再对该局部 primitive 子空间做 `\hat S^2` 投影。

这样可以保持系数矩阵 `C` 的块结构，便于：

- builder 实现
- 数值验证
- 后续性能优化

---

## 5. 推荐的数据结构

### 5.1 primitive basis 继续沿用现有 `PfBasisData`

现有 primitive basis：

- `src/pfaffian_vbscf/types/pf_basis_data.hpp`

不建议改变其语义。它仍然表示：

- 一组 primitive fixed-`M_s` Pf states

### 5.2 新增 spin-adapted state/basis wrapper

建议新增：

```text
src/pfaffian_vbscf/data/pf_spin_adapted_state.hpp
src/pfaffian_vbscf/types/pf_spin_adapted_basis_data.hpp
```

推荐结构如下：

```cpp
struct PfSpinAdaptedState {
  int spin_multiplicity = 1;
  int ms_twice = 0;
  std::vector<int> primitive_state_indices;
  std::vector<double> primitive_coefficients;
};
```

```cpp
struct PfSpinAdaptedBasisData {
  PfBasisData primitive_basis;
  int n_states = 0;
  int spin_multiplicity = 1;
  int ms_twice = 0;
  Matrix primitive_to_adapted_coefficients;  // N_p x N_s
  std::vector<PfSpinAdaptedState> states;
};
```

说明：

- `primitive_basis` 是唯一真正需要 kernel 逐对求值的 basis；
- `primitive_to_adapted_coefficients` 就是前文的 `C`；
- `states` 主要用于可读性、调试和文件输出；
- `ms_twice` 统一记录 `2 M_s`，避免半整数表示问题。

### 5.3 不建议把 spin-adapted 字段直接塞进 `PfState`

原因：

- `PfState` 当前语义非常清楚，就是 primitive state；
- spin-adapted state 是线性组合对象，不是单一 pair/block state；
- 混在一起会让 matrix builder 和 basis factory 变得难以维护。

第一版应尽量保持：

- `PfState` = primitive
- `PfSpinAdaptedState` = linear combination wrapper

---

## 6. 推荐的实现分层

### 6.1 coefficient builder

新增：

```text
src/pfaffian_vbscf/scf/pf_spin_coupling_builder.hpp
src/pfaffian_vbscf/scf/pf_spin_coupling_builder.cpp
```

职责：

- 输入：某一 open-shell spatial pattern 的 primitive fixed-`M_s` states
- 输出：该 pattern 上的 spin-adapted coefficient block

第一版推荐内部算法：

1. 生成 fixed-`M_s` spin-string basis；
2. 构造 `S^2` 矩阵；
3. 对角化得到 pure-`S` eigenvectors；
4. 组装为 coefficient block。

### 6.2 spin-adapted basis factory

新增：

```text
src/pfaffian_vbscf/scf/pf_spin_adapted_basis_factory.hpp
src/pfaffian_vbscf/scf/pf_spin_adapted_basis_factory.cpp
```

职责：

- 读取 raw structure / open-shell orbital pattern；
- 构造 primitive fixed-`M_s` basis；
- 调用 `PfSpinCouplingBuilder`；
- 汇总得到 `PfSpinAdaptedBasisData`。

### 6.3 matrix projection layer

新增：

```text
src/pfaffian_vbscf/matrices/pf_spin_adapted_matrix_projector.hpp
src/pfaffian_vbscf/matrices/pf_spin_adapted_matrix_projector.cpp
```

职责：

- 输入 primitive `S/H` matrices 和 coefficient matrix `C`
- 输出 spin-adapted `S/H`

第一版接口应简单：

```cpp
PfMatrixBuildResult project_spin_adapted_matrices(
    const PfMatrixBuildResult& primitive_mats,
    const Matrix& primitive_to_adapted_coefficients);
```

### 6.4 SCF evaluator 层

为了避免把现有 primitive `PfScfEval` 搅乱，建议新增一个薄包装：

```text
src/pfaffian_vbscf/scf/pf_spin_adapted_scf_eval.hpp
src/pfaffian_vbscf/scf/pf_spin_adapted_scf_eval.cpp
```

逻辑：

1. 先调用现有 `PfMatrixBuilder` 得到 primitive matrices
2. 用 projector 得到 spin-adapted matrices
3. 在 spin-adapted space 做 generalized eigensolve

这样 primitive flow 不需要改语义。

### 6.5 active gradient 层

同理新增薄包装：

```text
src/pfaffian_vbscf/scf/pf_spin_adapted_active_grad_eval.hpp
src/pfaffian_vbscf/scf/pf_spin_adapted_active_grad_eval.cpp
```

核心步骤：

1. primitive matrices
2. spin-adapted projection
3. spin-adapted generalized eigensolve
4. 得到 `G_H^{(sa)}` 与 `G_S^{(sa)}`
5. lift 回 primitive：
   `G_H^{(p)} = C G_H^{(sa)} C^T`
   `G_S^{(p)} = C G_S^{(sa)} C^T`
6. 调用现有 primitive pair adjoint sweep

这一步是第一版最重要的设计点。
它保证：

- gradient 仍然只在 primitive kernel 上求；
- spin-adapted 只是 matrix-weight lifting；
- 代码路径清晰且易验证。

---

## 7. 为什么第一版不直接改现有 `PfActiveGradEval`

理论上可以直接把 spin-adapted 逻辑塞进：

- `src/pfaffian_vbscf/scf/pf_scf_eval.cpp`
- `src/pfaffian_vbscf/scf/pf_active_grad_eval.cpp`

但第一版不建议这样做，原因是：

- 当前 primitive closed-shell / fixed-`M_s` flow 已经稳定；
- spin-adapted 是上层线性组合语义，不是 primitive kernel 语义；
- 先做薄包装最容易调试，也最容易和 primitive path 对照。

等 spin-adapted v1 稳定后，再考虑是否把 wrapper 向现有 evaluator 中合并。

---

## 8. 第一版实现顺序

推荐顺序如下：

1. 新增 `PfSpinAdaptedState` / `PfSpinAdaptedBasisData`
2. 做 `PfSpinCouplingBuilder`
3. 做 `PfSpinAdaptedMatrixProjector`
4. 做 `PfSpinAdaptedScfEval`
5. 做 `PfSpinAdaptedActiveGradEval`
6. 再决定是否接入 orbital optimizer

不要一开始就碰 orbital optimization。
先把：

- 矩阵元
- generalized eigensolver
- active-space gradient

三件事跑通，spin-adapted 主干就已经成立。

---

## 9. 验证策略

### 9.1 局部 algebra 验证

对小型 open-shell pattern：

- 先生成 primitive fixed-`M_s` states；
- 再用 coefficient matrix `C` 人工投影；
- 对比 spin-adapted matrix 是否满足：
  `S^{(sa)} = C^T S^{(p)} C`
  `H^{(sa)} = C^T H^{(p)} C`

### 9.2 与 primitive 再展开回去的一致性

取任意 spin-adapted eigenvector `d`，构造 primitive coefficient `c = C d`。
应验证：

- spin-adapted 能量
- primitive 表示下同一 recoupled 状态的能量

两者一致。

### 9.3 梯度验证

先对 spin-adapted matrix 层做有限差分验证：

- overlap gradient
- one-electron gradient
- two-electron gradient

确认：

```math
\partial E / \partial x
```

与

```math
C\, G^{(sa)} C^T
```

lift 回 primitive 后的 pairwise adjoint 累积一致。

### 9.4 分子测试顺序

建议顺序：

1. `F2_triplet` 这类最小高自旋 open-shell smoke test
2. 更一般 fixed-`M_s` 的人工小体系
3. 真正的 open-shell 分子输入

不要一开始就上大体系。

---

## 10. 与性能优化的关系

spin-adapted v1 做完后，性能优化的重心会很明确：

1. primitive pair kernel 的 `O(M^4)` 常数因子
2. primitive basis 的冗余去重与 pair screening
3. `C^T (·) C` 投影的批量线性代数实现

也就是说，spin-adapted 完成后，后续工作基本不再是“理论能不能做”，
而是“实现够不够快、体系够不够大、测试够不够全”。

---

## 11. 第一版明确不做的事情

以下内容不应纳入 spin-adapted 第一版：

- direct spin-adapted pure-`S` Pfaffian kernel
- 为每种 open-shell 数目手写 Rumer closed-form 系数
- 与 determinant basis 竞争的激进压缩技巧
- orbit optimizer 中的大规模框架重构

这些都应该放在 spin-adapted v1 跑通之后再考虑。

---

## 12. 推荐结论

推荐的 spin-adapted 实现路线是：

- 以现有 general fixed-`M_s` primitive kernel 为唯一底层矩阵元引擎；
- 新增一个外层 spin-coupling coefficient builder；
- 通过 `C^T S C`、`C^T H C` 和 `C G C^T` 完成 spin-adapted forward 与 gradient；
- 先做 wrapper 层，不破坏 primitive evaluator 语义；
- 等 spin-adapted v1 稳定后，再统一接口或继续优化。

这是当前最稳妥、最清晰、也最符合我们总体目标的路线。
