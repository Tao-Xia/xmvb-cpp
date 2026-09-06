# Outer-Response 的接受点线性响应预分解

## 1. 问题

当前 `exact_ctx` 路径中的完整轨道 HVP 可以写成

$$
B_k = B_k^{\mathrm{core}} + B_k^{\mathrm{outer}},
$$

其中：

- \(B_k^{\mathrm{core}}\) 表示当前 accepted point \(x_k\) 处的 core / local 二阶项；
- \(B_k^{\mathrm{outer}}\) 表示 relaxed structure / eigen response 所带来的 outer-response 修正。

这里真正需要回答的问题不是“要不要换一个优化器外壳”，而是：

$$
B_k^{\mathrm{outer}}
$$

在固定 accepted point \(x_k\) 之后，是否能够改写成“方向右端项 + 接受点固定线性算子”的形式。若答案为是，那么后续最有价值的工作就不是继续修改 trust-region 外壳，而是把 `outer-response` 中间链路改写为一个可复用的分块线性响应算子。

本文只讨论这一点。

## 2. 记号与维度

记：

- \(n_r\): 非冗余轨道变量维数；
- \(n_a\): 活性轨道数；
- \(n_{\mathrm{str}}\): structure space 维数；
- \(n_{\mathrm{sel}}\): selected states 数目；
- \(n_{\mathrm{det}}\): determinant 数目；
- \(d_{2e}^{\mathrm{pack}}\): 压缩存储后的活性双电子积分维数；
- \(d_{\mathrm{act}} = 2 n_a^2 + d_{2e}^{\mathrm{pack}}\).

在当前实现中，一个 reduced-space 方向写为

$$
p \in \mathbb{R}^{n_r}.
$$

对应的活性空间方向量记为

$$
\delta \eta =
\begin{bmatrix}
\operatorname{vec}(\delta S_{\mathrm{act}}) \\
\operatorname{vec}(\delta h_{\mathrm{act}}) \\
\delta g_{\mathrm{act}}^{(2)}
\end{bmatrix}
\in \mathbb{R}^{d_{\mathrm{act}}},
$$

其中：

- \(\delta S_{\mathrm{act}} \in \mathbb{R}^{n_a \times n_a}\) 是活性轨道重叠矩阵的一阶变化；
- \(\delta h_{\mathrm{act}} \in \mathbb{R}^{n_a \times n_a}\) 是活性单电子矩阵的一阶变化；
- \(\delta g_{\mathrm{act}}^{(2)} \in \mathbb{R}^{d_{2e}^{\mathrm{pack}}}\) 是压缩存储的活性双电子积分一阶变化。

这一步由
[src/vb/scf/exact_orbital_second_order_operator.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/exact_orbital_second_order_operator.cpp#L3373)
中的 `build_active_space_directional_integrals(...)` 完成。因此在 accepted point 固定时，存在一个线性映射

$$
T_k : \mathbb{R}^{n_r} \to \mathbb{R}^{d_{\mathrm{act}}},
\qquad
\delta \eta = T_k p.
$$

这里 \(T_k\) 仍然需要逐方向作用，因为它本身包含 AO 到活性空间的一阶方向收缩；但其系数完全由 accepted point \(x_k\) 决定。

## 3. 投影到 selected-state 结构响应

当前代码不会先构造完整的 structure-space 方向矩阵，再做一次截取；它直接构造 selected-state 所需的投影列块。记

$$
\delta \zeta =
\begin{bmatrix}
\operatorname{vec}(\delta \widetilde H_{\mathrm{sel}}) \\
\operatorname{vec}(\delta \widetilde S_{\mathrm{sel}})
\end{bmatrix}
\in \mathbb{R}^{2 n_{\mathrm{str}} n_{\mathrm{sel}}},
$$

其中

$$
\delta \widetilde H_{\mathrm{sel}},
\delta \widetilde S_{\mathrm{sel}}
\in \mathbb{R}^{n_{\mathrm{str}} \times n_{\mathrm{sel}}}.
$$

它们对应当前 accepted generalized eigensystem 基底中的投影结构响应。由
[src/vb/scf/exact_orbital_second_order_operator.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/exact_orbital_second_order_operator.cpp#L6580)
中的
`build_selected_state_projected_directional_structure_matrices(...)`
可知，在 accepted point 固定时存在一个线性映射

$$
\mathcal A_k : \mathbb{R}^{d_{\mathrm{act}}}
\to
\mathbb{R}^{2 n_{\mathrm{str}} n_{\mathrm{sel}}},
\qquad
\delta \zeta = \mathcal A_k \, \delta \eta.
$$

需要强调两点：

1. \(\mathcal A_k\) 的输入已经不是 AO 级轨道方向，而是更小的活性空间方向量 \(\delta \eta\)；
2. \(\mathcal A_k\) 在数学上是线性的，但在实现上不应该被显式存成一个巨大稠密矩阵，更合理的对象是“接受点固定的分块算子”。

## 4. 广义本征问题的一阶响应

记 accepted point 的 structure-space 广义本征问题为

$$
H_k U_k = S_k U_k E_k,
$$

其中：

- \(U_k \in \mathbb{R}^{n_{\mathrm{str}} \times n_{\mathrm{str}}}\) 为广义本征矢矩阵；
- \(E_k = \operatorname{diag}(\varepsilon_1, \dots, \varepsilon_{n_{\mathrm{str}}})\).

对每个 selected state \(i\)，记

$$
\delta \widetilde h_i \in \mathbb{R}^{n_{\mathrm{str}}},
\qquad
\delta \widetilde s_i \in \mathbb{R}^{n_{\mathrm{str}}}
$$

分别为 \(\delta \widetilde H_{\mathrm{sel}}\) 与 \(\delta \widetilde S_{\mathrm{sel}}\) 的第 \(i\) 列。则当前代码实现的 first-order generalized-eigen response 可写为

$$
\delta E_i
=
e_i^{\mathsf T}
\left(
\delta \widetilde h_i - \varepsilon_i \delta \widetilde s_i
\right),
$$

以及

$$
\delta \omega_i
=
\widehat D_i
\left(
\delta \widetilde h_i - \varepsilon_i \delta \widetilde s_i
\right)
- \frac{1}{2} e_i e_i^{\mathsf T} \delta \widetilde s_i,
$$

$$
\delta c_i = U_k \, \delta \omega_i.
$$

这里：

- \(e_i \in \mathbb{R}^{n_{\mathrm{str}}}\) 是第 \(i\) 个标准基向量；
- \(\widehat D_i\) 是 gap operator 的有界版本；
- 当能隙正常时，\(\widehat D_i\) 的非对角元退化为
  $$
  (\widehat D_i)_{jj} = \frac{1}{\varepsilon_i - \varepsilon_j},
  \qquad j \ne i;
  $$
- 在近简并或简并情况下，代码使用与权重相关的规范处理与安全分母替代。其实现位于
  [src/vb/scf/exact_orbital_second_order_operator.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/exact_orbital_second_order_operator.cpp#L7151)。

因此，对全部 selected states 汇总后，存在一个接受点固定线性映射

$$
\mathcal R_k :
\mathbb{R}^{2 n_{\mathrm{str}} n_{\mathrm{sel}}}
\to
\mathbb{R}^{n_{\mathrm{str}} n_{\mathrm{sel}} + n_{\mathrm{sel}}},
$$

使得

$$
\delta y =
\begin{bmatrix}
\operatorname{vec}(\delta C_{\mathrm{sel}}^{\mathrm{str}}) \\
\delta E_{\mathrm{sel}}
\end{bmatrix}
=
\mathcal R_k \, \delta \zeta.
$$

这里 \(\delta C_{\mathrm{sel}}^{\mathrm{str}} \in \mathbb{R}^{n_{\mathrm{str}} \times n_{\mathrm{sel}}}\) 表示 selected-state 结构系数的一阶响应。

这一节实际上已经说明：当前 `outer-response` 中最“像线性响应求解器”的部分，本质上就是一个 accepted-point 固定的有界线性算子应用，而不是每次方向都重新定义一个新问题。

## 5. 从 selected-state 响应到 determinant 权重

当前代码并不直接用 structure-space 的 \(\delta C_{\mathrm{sel}}^{\mathrm{str}}\) 参与回传，而是先把 selected-state 列转换成 determinant 系数表示，再构造 determinant-pair 权重。

记

$$
\delta Z_{\mathrm{sel}}
\in
\mathbb{R}^{n_{\mathrm{det}} \times n_{\mathrm{sel}}}
$$

为 selected-state determinant coefficient 的一阶响应。由
[src/vb/scf/selected_state_determinant_matrices.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/selected_state_determinant_matrices.cpp#L488)
可知，这一步是一个 accepted-point 固定的线性变换：

$$
\delta Z_{\mathrm{sel}} = \mathcal D_k \, \delta C_{\mathrm{sel}}^{\mathrm{str}}.
$$

随后，对任意 determinant 对 \((m,n)\)，当前代码构造的方向权重为

$$
\delta w_{mn}^{(H)}
=
\sum_{i \in I_k}
\omega_i
\left(
\delta z_{im} z_{in} + z_{im} \delta z_{in}
\right),
$$

$$
\delta w_{mn}^{(S)}
=
- \sum_{i \in I_k}
\omega_i
\left[
\delta E_i \, z_{im} z_{in}
+
\varepsilon_i
\left(
\delta z_{im} z_{in} + z_{im} \delta z_{in}
\right)
\right],
$$

其中：

- \(I_k\) 是 selected-state 指标集合；
- \(\omega_i\) 是 accepted point 处归一化后的 state-average 权重；
- \(z_{im}\) 是 accepted point 处第 \(i\) 个 selected state 对第 \(m\) 个 determinant 的系数；
- \(\delta z_{im}\) 是其一阶响应。

这正对应
[src/vb/scf/selected_state_determinant_matrices.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/selected_state_determinant_matrices.cpp#L681)
中的
`build_directional_determinant_pair_weight_tables_from_coefficients(...)`。

因此，存在一个接受点固定线性映射

$$
\mathcal B_k :
\mathbb{R}^{n_{\mathrm{det}} n_{\mathrm{sel}} + n_{\mathrm{sel}}}
\to
\mathbb{R}^{d_{\mathrm{pair}}},
$$

使得

$$
\delta w = \mathcal B_k
\begin{bmatrix}
\operatorname{vec}(\delta Z_{\mathrm{sel}}) \\
\delta E_{\mathrm{sel}}
\end{bmatrix},
$$

其中 \(d_{\mathrm{pair}}\) 表示 determinant-pair 权重存储的总维数。于是也可以把 \(\mathcal D_k\) 吸收到 \(\mathcal B_k\) 中，写成

$$
\delta w = \widetilde{\mathcal B}_k \, \delta y.
$$

## 6. determinant 权重到活性空间梯度方向

当前 outer-response 对活性空间梯度的贡献分成两部分：

1. 固定 selected states 时，由 \(\delta \eta\) 直接产生的 local outer-response；
2. 由 \((\delta Z_{\mathrm{sel}}, \delta E_{\mathrm{sel}})\) 引起的 directional outer-response。

因此可以写成

$$
\delta g_{\mathrm{act}}^{\mathrm{outer}}
=
\mathcal W_k^{\mathrm{loc}} \, \delta \eta
+
\mathcal W_k^{\mathrm{dir}} \, \delta w.
$$

这里：

- \(\delta g_{\mathrm{act}}^{\mathrm{outer}} \in \mathbb{R}^{d_{\mathrm{act}}}\)；
- \(\mathcal W_k^{\mathrm{loc}}\) 对应
  `build_local_same_spin_matrix_backward_contribution(...)`
  与
  `build_local_opposite_spin_matrix_backward_contribution(...)`
  所实现的活性空间回传；
- \(\mathcal W_k^{\mathrm{dir}}\) 对应
  `build_directional_same_spin_matrix_backward_contribution(...)`
  与
  `build_directional_opposite_spin_matrix_backward_contribution(...)`
  所实现的方向回传。

它们的实现分别位于：

- [src/vb/scf/exact_orbital_second_order_operator.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/exact_orbital_second_order_operator.cpp#L7896)
- [src/vb/scf/same_spin_matrix_backward.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/same_spin_matrix_backward.cpp#L3007)
- [src/vb/scf/opposite_spin_matrix_backward.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/opposite_spin_matrix_backward.cpp#L2289)

将上一节代入，可得

$$
\delta g_{\mathrm{act}}^{\mathrm{outer}}
=
\left(
\mathcal W_k^{\mathrm{loc}}
+
\mathcal W_k^{\mathrm{dir}}
\widetilde{\mathcal B}_k
\mathcal R_k
\mathcal A_k
\right)
\delta \eta.
$$

记

$$
\mathcal K_k
=
\mathcal W_k^{\mathrm{loc}}
+
\mathcal W_k^{\mathrm{dir}}
\widetilde{\mathcal B}_k
\mathcal R_k
\mathcal A_k,
$$

则

$$
\delta g_{\mathrm{act}}^{\mathrm{outer}} = \mathcal K_k \, \delta \eta.
$$

## 7. 回到轨道变量空间

最后，outer-response 的活性空间梯度方向会通过 accepted point 轨道准备链的伴随映射返回到轨道变量空间。这一步由
[src/vb/scf/exact_orbital_second_order_operator.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/exact_orbital_second_order_operator.cpp#L8010)
中的
`build_orbital_value_gradient_from_active_space_gradient_direction(...)`
完成。

记这一伴随映射为

$$
T_k^\sharp : \mathbb{R}^{d_{\mathrm{act}}} \to \mathbb{R}^{n_r}.
$$

则 outer-response HVP 可写成

$$
B_k^{\mathrm{outer}} p
=
T_k^\sharp \, \mathcal K_k \, T_k p.
$$

这就是当前 `exact_ctx` outer-response 的接受点预分解形式。

## 8. 这一分解说明了什么

### 8.1 当前优化器模式并不是主要问题

从这个分解可以看出，`nonredundant_truncated_newton + exact_ctx` 的思路本身没有问题。它本来就在尝试使用

$$
B_k^{\mathrm{core}} + B_k^{\mathrm{outer}}
$$

这个 accepted-point 二阶模型。

当前收敛偏慢，更本质的原因不是“选错了优化器模式”，而是：

1. `outer-response` 的中间链路虽然在数学上是固定线性算子，但在实现上还没有被整理成一个可复用的分块响应算子；
2. 因而每次 HVP 仍要重复做大量中间准备与回传，导致单次 HVP 成本偏高；
3. 单次 HVP 太贵，就会反过来限制内层 Newton / Krylov 求解的有效质量，于是总收敛效率和总时间都被拖慢。

### 8.2 真正值得做的是 accepted-point 预分解，而不是继续修改外壳

这一分解已经明确表明，后续最值得投入的工作不是继续在 trust-region / heuristic 分支上做文章，而是把

$$
\mathcal A_k,\quad
\mathcal R_k,\quad
\widetilde{\mathcal B}_k,\quad
\mathcal W_k^{\mathrm{dir}}
$$

整理成接受点固定的块算子，并尽量把其中昂贵的 accepted-point 部分预分解。

这里“预分解”不等于把 \(\mathcal K_k\) 显式写成一个巨大稠密矩阵，而是：

- 对 \(\mathcal R_k\) 显式缓存 gap operator、规范处理与 selected-state 列块响应；
- 对 \(\mathcal A_k\) 暴露从 \(\delta \eta\) 到 projected structure columns 的块作用接口；
- 对 \(\widetilde{\mathcal B}_k\) 与 \(\mathcal W_k^{\mathrm{dir}}\) 暴露 determinant-pair 权重与 same/opposite-spin backward 的线性块作用接口；
- 使一次 outer-response HVP 更接近“块线性算子作用”，而不是“重新走一遍方向流程”。

### 8.3 并不是所有成本都能消掉

这一分解同时也说明，以下两部分仍然是每个方向都必须付出的：

1. \(T_k p\): 方向轨道变化到活性空间方向积分的构造；
2. \(T_k^\sharp\): 活性空间梯度方向回到轨道空间的伴随回传。

因此，预分解不会把 HVP 变成零成本；它主要压缩的是中间的响应链

$$
\mathcal A_k
\to
\mathcal R_k
\to
\widetilde{\mathcal B}_k
\to
\mathcal W_k^{\mathrm{dir}}.
$$

这也是为什么它既有希望降低单次 HVP 时间，又有希望在相同总时间下允许更高质量的内层 Newton 步。

## 9. 对后续实现的直接启示

如果要继续做真正本质的改进，最合理的顺序是：

1. 先把 `outer-response` 中间链路改写为接受点固定的块线性响应算子；
2. 让一次 HVP 只显式构造方向右端项，而不重复准备中间结构；
3. 在这一基础上，再讨论是否值得继续推进更高层的二阶策略，例如更强的 Newton 子问题求解或重新评估 MFTR。

换句话说，当前首先需要解决的不是“换优化器模式”，而是“让当前二阶模型真正以线性响应算子的方式被使用”。

## 10. 总结

在当前代码语义下，`outer-response` 已经可以严格写成

$$
B_k^{\mathrm{outer}} = T_k^\sharp \, \mathcal K_k \, T_k,
$$

其中

$$
\mathcal K_k
=
\mathcal W_k^{\mathrm{loc}}
+
\mathcal W_k^{\mathrm{dir}}
\widetilde{\mathcal B}_k
\mathcal R_k
\mathcal A_k
$$

完全由 accepted point \(x_k\) 决定。

因此，当前收敛偏慢的主要矛盾不是优化器“模式”错了，而是这个接受点固定的二阶响应算子还没有被整理成真正高效的预分解块算子。后续若想同时改善收敛效率与总时间，这就是最有希望的主线。
