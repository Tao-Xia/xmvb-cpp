# Dual Structure Projection Theory

本文整理双正交 VBSCF 在 structure 子空间上的严格理论公式，目标是回答以下问题：

1. 原始 nonorthogonal VB 的 structure 子空间问题是什么。
2. 当前双正交原型为什么在 structure 子空间上一般不正确。
3. 怎样在保留双正交 determinant kernel 的前提下，严格复现原始 nonorthogonal structure 子空间结果。
4. 为什么即便 structure 子空间被截断，理论上仍然可以得到与原始 nonorthogonal 子空间完全一致的结果。

## 1. 记号

设 full determinant 空间维数为 $N_D$，选定的 VB structure 子空间维数为 $m$。

右 determinant 基记为

$$
\left\{\, |D_I\rangle \,\right\}_{I=1}^{N_D}.
$$

右 structure 基记为

$$
\left\{\, |\Phi_\mu\rangle \,\right\}_{\mu=1}^{m},
$$

并写成 determinant 展开

$$
|\Phi_\mu\rangle
=
\sum_{I=1}^{N_D} T_{I\mu}\, |D_I\rangle,
\qquad
T \in \mathbb{R}^{N_D \times m}.
$$

determinant 层的物理重叠矩阵和哈密顿矩阵定义为

$$
(S_{\mathrm{det}})_{JI}
=
\langle D_J | D_I \rangle,
\qquad
(H_{\mathrm{det}})_{JI}
=
\langle D_J | \hat H | D_I \rangle.
$$

下文默认所有系数均为实数，因此共轭转置统一写作转置 $^{\mathsf T}$。

## 2. 原始 nonorthogonal VB 的 structure 子空间问题

在选定的 structure 子空间上，原始 nonorthogonal VB 解的是广义本征问题

$$
H_{\mathrm{str}}\, c_n
=
E_n\, S_{\mathrm{str}}\, c_n,
$$

其中

$$
S_{\mathrm{str}}
=
T^{\mathsf T} S_{\mathrm{det}} T,
\qquad
H_{\mathrm{str}}
=
T^{\mathsf T} H_{\mathrm{det}} T.
$$

这就是“原始非正交 structure 子空间结果”的严格数学定义。

## 3. determinant 层的双正交化

设右轨道系数矩阵为 $C$，AO 重叠矩阵为 $S_{\mathrm{AO}}$。定义左 dual 轨道

$$
\widetilde C
=
C \left( C^{\mathsf T} S_{\mathrm{AO}} C \right)^{-1},
$$

于是有

$$
\widetilde C^{\mathsf T} S_{\mathrm{AO}} C
=
I.
$$

由这些 dual 轨道构造左 determinant bra

$$
\left\{\, \langle \widetilde D_J | \,\right\}_{J=1}^{N_D},
$$

满足

$$
\langle \widetilde D_J | D_I \rangle
=
\delta_{JI}.
$$

然后定义 determinant 层的双正交哈密顿矩阵

$$
\bigl(h_{\mathrm{det}}^{\mathrm{bi}}\bigr)_{JI}
=
\langle \widetilde D_J | \hat H | D_I \rangle.
$$

由于

$$
\langle D_J |
=
\sum_{L=1}^{N_D}
(S_{\mathrm{det}})_{JL}\,
\langle \widetilde D_L |,
$$

立刻得到

$$
H_{\mathrm{det}}
=
S_{\mathrm{det}}\, h_{\mathrm{det}}^{\mathrm{bi}}.
$$

这一步是双正交 determinant kernel 的核心关系。只要该关系成立，determinant 层的双正交公式就是正确的。

## 4. 当前原型为什么在 structure 子空间上不正确

当前原型在 structure 层隐含地采用了与右结构完全相同的左系数矩阵 $T$，即

$$
\langle \widetilde \Phi_\mu^{\mathrm{naive}} |
=
\sum_{I=1}^{N_D}
T_{I\mu}\,
\langle \widetilde D_I |.
$$

于是它对应的 structure metric 为

$$
M_{\mathrm{naive}}
=
T^{\mathsf T} T,
$$

对应的 structure Hamiltonian 为

$$
H_{\mathrm{naive}}
=
T^{\mathsf T} h_{\mathrm{det}}^{\mathrm{bi}} T.
$$

因此它实际求解的是

$$
H_{\mathrm{naive}}\, r_n
=
E_n\, M_{\mathrm{naive}}\, r_n,
$$

即

$$
T^{\mathsf T} h_{\mathrm{det}}^{\mathrm{bi}} T\, r_n
=
E_n\, T^{\mathsf T} T\, r_n.
$$

然而原始 nonorthogonal structure 子空间问题是

$$
T^{\mathsf T} H_{\mathrm{det}} T\, c_n
=
E_n\, T^{\mathsf T} S_{\mathrm{det}} T\, c_n.
$$

两者一般并不相同，因为通常有

$$
S_{\mathrm{det}} \neq I.
$$

因此，当前原型解的并不是原始 nonorthogonal VB 的同一个 structure 子空间问题，而是另一个 fixed-metric Petrov-Galerkin 问题。

## 5. 严格等价的双正交 structure 子空间公式

要严格复现原始 nonorthogonal structure 子空间结果，必须在 structure 层定义正确的左测试空间

$$
U
=
S_{\mathrm{det}}\, T.
$$

即左 structure bra 应定义为

$$
\langle \Chi_\mu |
=
\sum_{I=1}^{N_D}
U_{I\mu}\,
\langle \widetilde D_I |,
\qquad
U = S_{\mathrm{det}} T.
$$

此时有

$$
U^{\mathsf T} T
=
T^{\mathsf T} S_{\mathrm{det}} T
=
S_{\mathrm{str}},
$$

并且

$$
U^{\mathsf T} h_{\mathrm{det}}^{\mathrm{bi}} T
=
T^{\mathsf T} S_{\mathrm{det}} h_{\mathrm{det}}^{\mathrm{bi}} T
=
T^{\mathsf T} H_{\mathrm{det}} T
=
H_{\mathrm{str}}.
$$

因此，原始 nonorthogonal VB 子空间问题与下面这个双正交投影问题严格等价：

$$
U^{\mathsf T} h_{\mathrm{det}}^{\mathrm{bi}} T\, c_n
=
E_n\, U^{\mathsf T} T\, c_n.
$$

代回上式即可得到

$$
H_{\mathrm{str}}\, c_n
=
E_n\, S_{\mathrm{str}}\, c_n.
$$

所以，双正交 determinant kernel 是完全可以精确复现原始 nonorthogonal structure 子空间结果的；失败只来自 structure 层投影定义错误，而不是双正交 determinant 代数本身错误。

## 6. 如何把最终求解写成单位阵标准本征问题

如果不希望最终保留一个广义本征问题，可以定义归一化后的左 structure 基

$$
B
=
U \left( U^{\mathsf T} T \right)^{-1}
=
S_{\mathrm{det}} T
\left( T^{\mathsf T} S_{\mathrm{det}} T \right)^{-1}.
$$

则有

$$
B^{\mathsf T} T = I.
$$

于是可以解标准非厄米本征问题

$$
B^{\mathsf T} h_{\mathrm{det}}^{\mathrm{bi}} T\, c_n
=
E_n\, c_n.
$$

并且

$$
B^{\mathsf T} h_{\mathrm{det}}^{\mathrm{bi}} T
=
\left( T^{\mathsf T} S_{\mathrm{det}} T \right)^{-1}
T^{\mathsf T} H_{\mathrm{det}} T
=
S_{\mathrm{str}}^{-1} H_{\mathrm{str}}.
$$

这说明：

1. 如果坚持右 structure 基不变，则物理 overlap $S_{\mathrm{str}}$ 不会消失。
2. overlap 可以被吸收到左基构造中。
3. 最终求解阶段确实可以改写为单位阵的标准本征问题。

## 7. 为什么“只在局部截断 determinant 列表中修正”仍然不够

设 full determinant 空间按“保留 determinants”和“被截掉 determinants”分成两块，分别记为 $k$ 与 $o$。将 full-space 矩阵写成分块形式：

$$
S_{\mathrm{det}}^{\mathrm{full}}
=
\begin{pmatrix}
S_{kk} & S_{ko} \\
S_{ok} & S_{oo}
\end{pmatrix},
\qquad
h_{\mathrm{det}}^{\mathrm{bi,full}}
=
\begin{pmatrix}
h_{kk}^{\mathrm{bi}} & h_{ko}^{\mathrm{bi}} \\
h_{ok}^{\mathrm{bi}} & h_{oo}^{\mathrm{bi}}
\end{pmatrix}.
$$

由

$$
H_{\mathrm{det}}^{\mathrm{full}}
=
S_{\mathrm{det}}^{\mathrm{full}}\, h_{\mathrm{det}}^{\mathrm{bi,full}}
$$

可得保留块满足

$$
H_{kk}^{\mathrm{full}}
=
S_{kk} h_{kk}^{\mathrm{bi}}
+
S_{ko} h_{ok}^{\mathrm{bi}}.
$$

设 $T_k$ 为选定 structures 在保留 determinant 子集上的展开矩阵，则真正精确的 structure Hamiltonian 为

$$
H_{\mathrm{str}}^{\mathrm{exact}}
=
T_k^{\mathsf T} H_{kk}^{\mathrm{full}} T_k
=
T_k^{\mathsf T} S_{kk} h_{kk}^{\mathrm{bi}} T_k
+
T_k^{\mathsf T} S_{ko} h_{ok}^{\mathrm{bi}} T_k.
$$

而若只在局部 determinant 列表里修正 overlap，则只保留第一项：

$$
H_{\mathrm{str}}^{\mathrm{local}}
=
T_k^{\mathsf T} S_{kk} h_{kk}^{\mathrm{bi}} T_k.
$$

因此缺失项为

$$
T_k^{\mathsf T} S_{ko} h_{ok}^{\mathrm{bi}} T_k,
$$

这正是 omitted-determinant coupling。

结论是：

$$
\boxed{
\text{仅将局部子空间的 metric 改成 } T_k^{\mathsf T} S_{kk} T_k \text{ 并不足以严格恢复原始结果。}
}
$$

如果要对一个截断的 structure 子空间得到与原始 nonorthogonal 计算完全一致的结果，必须使用 full determinant 空间中的作用

$$
U = S_{\mathrm{det}}^{\mathrm{full}} T.
$$

## 8. 与 Slater--Condon 规则的关系

引入 $S_{\mathrm{det}}$ 并不意味着 determinant 层的双正交 Slater--Condon 规则失效。必须区分以下两层：

1. determinant 层的 matrix element：

$$
\bigl(h_{\mathrm{det}}^{\mathrm{bi}}\bigr)_{JI}
=
\langle \widetilde D_J | \hat H | D_I \rangle.
$$

这里仍然使用双正交 determinant bra 与右 determinant ket，因此仍可使用 biorthogonal Slater--Condon 规则。

2. structure 层的投影：

$$
\langle \Chi_\mu | \hat H | \Phi_\nu \rangle
=
\sum_{I,J}
U_{I\mu}\,
\bigl(h_{\mathrm{det}}^{\mathrm{bi}}\bigr)_{IJ}\,
T_{J\nu}.
$$

这一步只是对 determinant matrix elements 做线性组合，不要求重新定义新的 determinant 级 Slater--Condon 公式。

因此：

$$
\boxed{
\text{双正交 determinant Slater--Condon 仍然有效；}
\; S_{\mathrm{det}} \text{ 只是在 structure 投影层恢复正确几何。}
}
$$

## 9. 精确双正交 structure 子空间算法的公式流程

给定一个选定的 structure 子空间，其严格等价的双正交求解流程可以写成：

第一步，构造右 structure 展开矩阵

$$
T.
$$

第二步，使用双正交 determinant kernel 计算

$$
Y
=
h_{\mathrm{det}}^{\mathrm{bi}} T.
$$

第三步，计算正确的左 structure 投影系数

$$
U
=
S_{\mathrm{det}} T.
$$

第四步，构造 structure 层矩阵

$$
H_{\mathrm{str}}
=
U^{\mathsf T} Y,
\qquad
S_{\mathrm{str}}
=
U^{\mathsf T} T.
$$

第五步，解广义本征问题

$$
H_{\mathrm{str}} c_n
=
E_n S_{\mathrm{str}} c_n.
$$

或者定义

$$
B
=
U \left( U^{\mathsf T} T \right)^{-1},
$$

再解标准本征问题

$$
B^{\mathsf T} Y\, c_n
=
E_n c_n.
$$

## 10. 最终结论

对于一个给定的 structure 子空间，严格等价的双正交公式为

$$
\boxed{
\left( T^{\mathsf T} S_{\mathrm{det}} h_{\mathrm{det}}^{\mathrm{bi}} T \right) c_n
=
E_n
\left( T^{\mathsf T} S_{\mathrm{det}} T \right) c_n
}
$$

或者等价地

$$
\boxed{
\left( T^{\mathsf T} S_{\mathrm{det}} T \right)^{-1}
\left( T^{\mathsf T} S_{\mathrm{det}} h_{\mathrm{det}}^{\mathrm{bi}} T \right) c_n
=
E_n c_n
}
$$

其含义如下：

1. determinant 层继续使用双正交 Slater--Condon。
2. structure 层必须使用正确的左投影 $U = S_{\mathrm{det}} T$。
3. 只使用当前原型中的 $T^{\mathsf T} T$ 度量，structure 子空间结果一般错误。
4. 即便 structure 子空间经过截断，仍然可以严格复现原始 nonorthogonal 子空间结果，但前提是使用 full-space 的正确投影，而不是局部截断后的闭门近似。
