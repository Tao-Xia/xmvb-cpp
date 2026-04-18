# Biorthogonal Exact Selected Gradient Plan

本文档给出当前 `exact biorthogonal selected-space` 路径的梯度公式与实现路线，目标是指导下一步代码工作，而不是重复前向理论。前向投影的严格等价性已经在 `dual_structure_projection_theory.md` 中说明；这里重点回答以下问题：

1. full space 与任意 selected structure subspace 能否使用同一套梯度框架。
2. 当前双正交 exact selected-space 的目标函数到底是什么。
3. 外层 generalized eigensolver 的反传能否直接复用非正交实现。
4. determinant / unique-spin 层需要新增哪些中间量与 contraction 公式。
5. 下一步代码应按什么顺序推进，才能先用 F2 做出可靠验证，再做 block contraction 优化。

## 1. 统一目标函数与统一框架

设 full determinant 空间维数为 $N_{\det}$，structure 投影维数为 $K$。定义固定投影矩阵

$$
T \in \mathbb R^{N_{\det} \times K}.
$$

其中：

1. 全空间时，$T = T_{\mathrm{full}}$，$K = N_{\mathrm{str}}$。
2. 截断子空间时，$T = T_{\mathrm{sel}}$，$K < N_{\mathrm{str}}$。

因此，full space 与 selected subspace 的差别不在理论框架，而只在所选投影矩阵 $T$。两者都属于同一个 projected objective family：

$$
J(T; \theta) = \sum_{n \in \mathrm{sel}} \omega_n E_n(T; \theta),
$$

其中 $\theta$ 表示轨道参数，$\omega_n$ 为 state-average 权重。

所以结论非常明确：

$$
\boxed{
\text{full space 是同一套梯度框架的特例；selected space 不是另一种理论。}
}
$$

但如果 $T_{\mathrm{sel}} \neq T_{\mathrm{full}}$，则优化目标本身不同，因此梯度数值一般不同：

$$
\nabla_\theta J(T_{\mathrm{sel}}; \theta)
\neq
\nabla_\theta J(T_{\mathrm{full}}; \theta).
$$

## 2. 当前 exact selected-space 前向对象

当前代码中的 exact selected-space 前向已经实现了如下对象：

$$
U = S_{\det} T,
\qquad
Y = h_{\mathrm{bi}} T,
$$

并构造物理 structure 矩阵

$$
S_{\mathrm{str}} = U^{\mathsf T} T,
\qquad
H_{\mathrm{str}} = U^{\mathsf T} Y.
$$

然后解对称广义本征问题

$$
H_{\mathrm{str}} c_n
=
E_n S_{\mathrm{str}} c_n.
$$

当前实现位置：

1. `src/vb/biorthogonal_vbscf/biorthogonal_exact_selected_structure.hpp`
2. `src/vb/biorthogonal_vbscf/biorthogonal_exact_selected_structure.cpp`

特别地，当前 forward 已经显式导出了

$$
R = T C,
\qquad
L = U C,
$$

也就是 determinant 层的右系数矩阵与 dual-left 系数矩阵：

$$
R_{I n} = r_I^{(n)},
\qquad
L_{J n} = l_J^{(n)}.
$$

这对应代码里的：

1. `right_determinant_coefficient_matrix`
2. `left_determinant_coefficient_matrix`

## 3. 外层 generalized eigensolver 反传与非正交相同

由于当前 exact selected-space 路径最终仍然在解物理的对称广义本征问题

$$
H_{\mathrm{str}} c_n
=
E_n S_{\mathrm{str}} c_n,
\qquad
c_n^{\mathsf T} S_{\mathrm{str}} c_m = \delta_{nm},
$$

所以外层 adjoint 与非正交实现完全相同：

$$
W_H
=
\sum_{n \in \mathrm{sel}} \omega_n c_n c_n^{\mathsf T},
\qquad
W_S
=
-\sum_{n \in \mathrm{sel}} \omega_n E_n c_n c_n^{\mathsf T}.
$$

于是

$$
\mathrm d J
=
\operatorname{tr}\!\left(W_H^{\mathsf T} \mathrm d H_{\mathrm{str}}\right)
+
\operatorname{tr}\!\left(W_S^{\mathsf T} \mathrm d S_{\mathrm{str}}\right).
$$

这一步与当前非正交代码一致，可直接参考：

1. `src/vb/scf/cpp_active_space_gradient_evaluator.cpp`
2. `build_structure_pair_weight_tables(...)`
3. `determinant_pair_structure_adjoints(...)`

因此，双正交梯度工作**不需要重写外层 eigensolver 反传理论**。

## 4. 真正变化的地方：从 structure 层回传到 determinant 层

由

$$
S_{\mathrm{str}} = U^{\mathsf T} T,
\qquad
H_{\mathrm{str}} = U^{\mathsf T} Y,
$$

并注意到 $T$ 对轨道参数固定，可得

$$
\mathrm d S_{\mathrm{str}} = \mathrm d U^{\mathsf T} T,
$$

$$
\mathrm d H_{\mathrm{str}} = \mathrm d U^{\mathsf T} Y + U^{\mathsf T} \mathrm d Y.
$$

反向传播后得到

$$
\bar Y = U W_H,
\qquad
\bar U = T W_S + Y W_H.
$$

再利用

$$
Y = h_{\mathrm{bi}} T,
\qquad
U = S_{\det} T,
$$

可得 determinant 层 adjoint

$$
\bar h_{\mathrm{bi}} = \bar Y T^{\mathsf T},
\qquad
\bar S_{\det} = \bar U T^{\mathsf T}.
$$

为了写成最适合代码实现的状态求和形式，定义

$$
r_n = T c_n,
\qquad
l_n = U c_n,
\qquad
m_n = Y c_n = h_{\mathrm{bi}} r_n,
\qquad
q_n = m_n - E_n r_n.
$$

则有

$$
\bar h_{\mathrm{bi}}
=
\sum_{n \in \mathrm{sel}} \omega_n\, l_n r_n^{\mathsf T},
$$

$$
\bar S_{\det}
=
\sum_{n \in \mathrm{sel}} \omega_n\, q_n r_n^{\mathsf T}.
$$

这就是双正交 exact selected-space 梯度的核心公式。

## 5. determinant ordered-pair 权重公式

对有序 determinant pair $(J, I)$，定义：

$$
W_H^{\det}(J, I)
=
\sum_{n \in \mathrm{sel}} \omega_n\, l_J^{(n)} r_I^{(n)},
$$

$$
W_S^{\det}(J, I)
=
\sum_{n \in \mathrm{sel}} \omega_n\, q_J^{(n)} r_I^{(n)}.
$$

于是 active-space 目标的微分可以统一写成

$$
\mathrm d J
=
\sum_{J, I}
W_H^{\det}(J, I)\, \mathrm d h_{\mathrm{bi}, JI}
+
\sum_{J, I}
W_S^{\det}(J, I)\, \mathrm d S_{\det, JI}.
$$

这正是双正交版本的 determinant-pair backward 权重。

和非正交相比，变化不在“是否还有 overlap 项”，而在于：

1. 非正交是对称双线性型，权重由同一套系数 $c_d^{(n)}$ 构成。
2. 双正交 exact 是非对称双线性型，Hamiltonian 通道使用 $(L, R)$，overlap 通道使用 $(Q, R)$。

## 6. 为什么当前 matrix-form backward 不能直接复用

当前非正交 matrix-form backward 的核心假设是：每个 selected state 只有一套 unique-spin 系数矩阵

$$
C^{(n)}.
$$

因此 same-spin 与 opposite-spin 通道都压缩成类似

$$
C^{(n)} B [C^{(n)}]^{\mathsf T},
\qquad
[C^{(n)}]^{\mathsf T} A C^{(n)}
$$

的形式。

这对应当前实现：

1. `src/vb/scf/same_spin_matrix_backward.cpp`
2. `src/vb/scf/opposite_spin_matrix_backward.cpp`
3. `src/vb/scf/selected_state_determinant_matrices.hpp`

但双正交 exact backward 需要三套 state bundle：

$$
R^{(n)}, \qquad L^{(n)}, \qquad Q^{(n)}.
$$

因此所有 matrix-form contraction 都必须从单侧 `C/C` 形式推广为双侧形式。

## 7. 双正交 unique-spin contraction 的目标形式

设 $R^{(n)}$、$L^{(n)}$、$Q^{(n)}$ 分别是第 $n$ 个 state 的 unique-spin coefficient matrix。则应有如下目标形式：

### 7.1 same-spin alpha 通道

Hamiltonian 权重：

$$
W_{\alpha, H}
=
\sum_n \omega_n\,
L^{(n)} B_\beta [R^{(n)}]^{\mathsf T}.
$$

Overlap 权重：

$$
W_{\alpha, S}
=
\sum_n \omega_n\,
Q^{(n)} B_\beta [R^{(n)}]^{\mathsf T}.
$$

### 7.2 same-spin beta 通道

Hamiltonian 权重：

$$
W_{\beta, H}
=
\sum_n \omega_n\,
[L^{(n)}]^{\mathsf T} A_\alpha R^{(n)}.
$$

Overlap 权重：

$$
W_{\beta, S}
=
\sum_n \omega_n\,
[Q^{(n)}]^{\mathsf T} A_\alpha R^{(n)}.
$$

### 7.3 opposite-spin 通道

所有当前依赖 `C^{(n)}` 的 opposite-spin block contraction，也都要推广为双侧版本。原则上统一写成

$$
\sum_n \omega_n\, \mathcal B\!\left(L^{(n)}, R^{(n)}\right)
\quad \text{或} \quad
\sum_n \omega_n\, \mathcal B\!\left(Q^{(n)}, R^{(n)}\right),
$$

其中 $\mathcal B(\cdot, \cdot)$ 表示当前 opposite-spin packed-pair / sparse-block contraction 核。

## 8. 新的数据结构建议

建议不要直接复用 `SelectedStateDeterminantMatrices`，而是新增一个双正交专用 bundle，例如：

$$
\texttt{BiorthogonalSelectedStateMatrices}.
$$

每个 state 至少保存：

1. determinant-level 右系数
   $$
   r^{(n)} \in \mathbb R^{N_{\det}}
   $$
2. determinant-level dual-left 系数
   $$
   l^{(n)} \in \mathbb R^{N_{\det}}
   $$
3. determinant-level residual-left 系数
   $$
   q^{(n)} = m^{(n)} - E_n r^{(n)}
   $$
4. unique-spin matrix form
   $$
   R^{(n)},\ L^{(n)},\ Q^{(n)}
   $$
5. 每个 state 的 support block 信息，便于继续沿用局部块 contraction。

其中

$$
m^{(n)} = h_{\mathrm{bi}} r^{(n)}
$$

不能只靠 `L` 替代，因为

$$
L = S_{\det} R,
\qquad
Q = Y C - E R
$$

对应不同的物理通道。

## 9. 推荐的实施顺序

### 9.1 第一步：先做 ordered determinant-pair reference gradient

先不要急着做 matrix-form 优化，先实现一个最小但严格的 reference 版：

1. 由 exact selected-space forward 构造
   $$
   R,\ L,\ Q
   $$
2. 组装有序 pair 权重
   $$
   W_H^{\det}(J, I),\quad W_S^{\det}(J, I)
   $$
3. 对每个 ordered pair $(J, I)$ 调 exact active-space gradient oracle。

这样做的优点是：

1. 数学最直接。
2. 最容易和有限差分比对。
3. 最适合先在 F2 上验证。

现有仓库中已经有非常合适的 exact oracle 接口，接受 `(hamiltonian_weight, overlap_weight)`：

1. `src/vb/exact_separator/component_tree.cpp`
2. `evaluate_rooted_component_tree_active_space_gradient_exact(...)`
3. `evaluate_rooted_component_tree_active_space_gradient_boundary_collapsed(...)`

因此 reference 版最适合作为第一步真值实现。

### 9.2 第二步：在 F2 上做解析梯度与有限差分验证

F2 上至少要验证以下场景：

1. full space
2. selected space 且 $K = 1$
3. selected space 且 $K = 2$
4. selected space 且 $K = 3$

验证对象至少包括：

1. 总能量梯度
2. `active_orbital_overlap_gradient`
3. `active_one_electron_gradient`
4. `packed_active_two_electron_gradient`

目标不是一开始追求快，而是先确认：

$$
\nabla_\theta J_{\mathrm{analytic}}
\approx
\nabla_\theta J_{\mathrm{FD}}.
$$

### 9.3 第三步：再推广到 matrix-form-block-contraction

当 ordered reference 版通过 F2 后，再将 current nonorth matrix-form backward 推广到 biorth 版：

1. same-spin：
   $$
   C B C^{\mathsf T}
   \longrightarrow
   L B R^{\mathsf T},\quad Q B R^{\mathsf T}
   $$
2. opposite-spin：
   所有使用 `C^(n)` 的 dense / sparse contraction 都推广为左-右双侧 contraction。

这一阶段的目标是保留当前非正交实现的优势：

1. `unique-spin-string` 级别压缩
2. `matrix-form-block-contraction`
3. 内存友好的 tile / batch sweep

### 9.4 第四步：接入 orbital gradient 总链路

只要 exact biorth active-space backward 最终产出三块矩阵 adjoint：

1. `active_orbital_overlap_gradient`
2. `active_one_electron_gradient`
3. `packed_active_two_electron_gradient`

则下游 orbital / AO backprop 原则上可直接复用：

1. `src/vb/scf/cpp_orbital_gradient_evaluator.cpp`
2. `src/vb/orbital/active_space_matrix_backpropagator.*`
3. `src/vb/orbital/active_space_orbital_backpropagator.*`
4. `src/vb/orbital/active_space_two_electron_backpropagator.*`
5. `src/vb/orbital/ao_effective_one_electron_backpropagator.*`

也就是说，新的工作边界是：

$$
\boxed{
\text{主要新增的是 exact biorth active-space backward，而不是 AO/orbital 下游链路。}
}
$$

## 10. 建议的代码拆分

建议新增以下层次，而不是把所有逻辑硬塞进现有 nonorth 文件：

1. `biorthogonal_selected_state_matrices.hpp/.cpp`
   用于构造 determinant-level 与 unique-spin-level 的 $R/L/Q$。
2. `biorthogonal_exact_selected_gradient_reference.hpp/.cpp`
   有序 determinant-pair reference gradient，用于 F2 验证。
3. `biorthogonal_same_spin_matrix_backward.hpp/.cpp`
   双侧 same-spin matrix-form contraction。
4. `biorthogonal_opposite_spin_matrix_backward.hpp/.cpp`
   双侧 opposite-spin matrix-form contraction。
5. `biorthogonal_exact_selected_structure_gradient.hpp/.cpp`
   统一封装 selected-space biorth active-space gradient 接口。

这样做的好处是：

1. nonorth 现有逻辑不被大范围扰动。
2. reference 与 optimized 两条路径可以并存，方便回归比较。
3. full space 与 selected space 只通过传入的 $T$ 区分，无需写两套理论。

## 11. 最小闭环验收标准

下一阶段建议的最小闭环不是 benchmark，而是以下四条：

1. F2 的 exact biorth selected-space forward 继续与 nonorth selected-space 严格一致。
2. F2 上 ordered reference gradient 与有限差分一致。
3. matrix-form biorth gradient 与 ordered reference gradient 一致。
4. full space 与 selected space 都通过同一套接口，只改变 $T$ 或 `selected_structure_indices`。

只有在这四条都通过后，再开始谈 C6H6 上的 gradient benchmark，才不会在错误实现上做优化。

## 12. 当前最重要的实现结论

最终要落地的，不是“再造一套 full-space 梯度”和“再造一套 selected-space 梯度”，而是实现下面这个统一模板：

1. 给定固定投影 $T$。
2. 前向构造
   $$
   U = S_{\det} T,\qquad Y = h_{\mathrm{bi}} T.
   $$
3. 解物理广义本征问题
   $$
   H_{\mathrm{str}} c_n = E_n S_{\mathrm{str}} c_n.
   $$
4. 反向构造
   $$
   R = T C,\qquad L = U C,\qquad Q = Y C - R E.
   $$
5. 用
   $$
   (L, R)
   $$
   处理 Hamiltonian 通道，用
   $$
   (Q, R)
   $$
   处理 overlap 通道。
6. 压回 active-space 三块梯度，再复用现有 AO / orbital backprop。

因此，下一步最值得做的具体工作是：

$$
\boxed{
\text{先实现 } R/L/Q \text{ 的 ordered exact reference gradient，并在 F2 上过 finite difference。}
}
$$

这一步通过后，再做 `matrix-form-block-contraction` 的双侧优化，风险最低，路径最清楚。
