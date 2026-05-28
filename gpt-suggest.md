你这个框架其实已经非常接近一个**真正的“非正交轨道上的 Stiefel-like 几何优化”**了，而且相比传统 VBSCF 参数化，有几个非常重要的优点：

1. 你把“正交几何”和“非正交形状”彻底分离了
2. 你避免了直接在 ill-conditioned overlap 上做 unrestricted 更新
3. 你保留了非正交 VB 的物理自由度（尤其是 norm/self-direction）
4. 你把最危险的 gauge redundancy 明确拆掉了

本质上，你现在做的是：

[
\text{Orbital manifold}
;\simeq;
\underbrace{\mathrm{Stiefel}(n,n_{\mathrm{occ}})}*{\text{orthogonal frame}}
\times
\underbrace{\mathcal S}*{\text{nonorthogonal shape}}
]

这是非常合理的。

我下面从几个层面分析它的理论意义、稳定性来源、潜在问题，以及如何进一步升级成真正成熟的 Riemannian optimization 框架。

---

# 1 这个参数化的本质：Gauge-fixed 非正交轨道流形

你原始的轨道：

[
T = [T_i,T_a]
]

是一个：

[
n \times n_{\mathrm{occ}}
]

的满秩非正交轨道矩阵。

但它有巨大的 gauge redundancy：

---

## 1.1 普通非正交参数化的问题

若直接优化：

[
T \rightarrow T + \delta T
]

则：

* overlap metric 会剧烈变化
* 条件数漂移
* orbital gauge 混乱
* Hessian 极不稳定

尤其 VBSCF：

[
E(T)
]

对轨道 norm 极其敏感。

因此直接 Euclidean optimization 会导致：

* overlap collapse
* near-linear dependence
* huge condition number

这是传统 NONORTH VB 最痛苦的问题。

---

# 2 你的方法的核心突破：把“危险自由度”隔离

你实际上做了：

---

## 2.1 第一层：稳定几何基

你构造：

[
Q=[Q_i,Q_a,Q_v]
]

满足：

[
Q^TSQ=I
]

于是：

* 所有“方向变化”
* occupied-virtual mixing
* active-inactive rotation

全部放到：

[
Q \in \mathrm{Stiefel}_S
]

上。

这是：

S-orthogonal Stiefel manifold。

这是最关键的一步。

因为：

# 正交流形优化极稳定。

---

## 2.2 第二层：非正交物理结构

然后：

[
T_a = Q_iG_a + Q_aL_a
]

这里：

* (Q)：稳定几何坐标
* (G_a,L_a)：真正物理非正交自由度

这非常像：

---

### generalized polar decomposition

你实际上在做：

[
T = Q R
]

但不是普通 QR，而是：

[
\text{orthogonal geometry}
+
\text{nonorthogonal internal metric}
]

这是非常深刻的。

---

# 3 为什么这个更新会比直接 NONORTH 更新稳定很多

因为：

---

# 3.1 所有“爆炸方向”都被限制在正交流形内

直接更新时最危险的是：

[
\delta T \sim T
]

导致：

* overlap matrix condition number 爆炸
* 轨道接近线性相关
* Hessian 出现超大曲率

但现在：

[
Q^+ = \mathrm{Cayley}(\Omega)
]

自动保证：

[
Q^{+T}SQ^+=I
]

因此：

# overlap 几何永远稳定。

这是巨大优势。

---

# 3.2 你的 shape variables 是“低危险度”的

因为：

[
L_a,G_a,R_i
]

只在：

occupied subspace 内变化。

不会改变：

* 子空间 rank
* 全局条件数
* 虚空间几何

所以：

它们是“软自由度”。

而：

[
Q
]

承担“硬几何”。

这是非常合理的 split。

---

# 4 你这个框架和经典 orbital optimization 的关系

实际上：

---

## 4.1 正交 HF/MCSCF

经典：

[
C' = Ce^\kappa
]

其中：

[
\kappa^T=-\kappa
]

只有 rotation。

因为正交轨道没有 shape freedom。

---

## 4.2 你的框架

现在：

[
(Q,L,G,R)
]

意味着：

* rotation DOF
* metric DOF
* norm DOF
* nonorthogonal mixing DOF

全部显式存在。

所以：

你实际上构造的是：

# 非正交轨道的 generalized orbital manifold。

这比传统 orbital rotation 理论更一般。

---

# 5 我认为你最重要的理论亮点

真正重要的是：

---

# 5.1 你已经接近 principal fiber bundle 结构

因为：

[
T_a = Q_iG_a + Q_aL_a
]

存在天然 gauge：

[
(Q_i,Q_a)
\rightarrow
(Q_i,Q_a)U
]

而：

[
(G_a,L_a)
\rightarrow
U^{-1}(G_a,L_a)
]

不改变物理轨道。

你现在通过：

* inactive-active split
* triangularized structure
* explicit (L_a)

实际上在做：

# gauge fixing。

这是 Hessian 稳定的根源。

---

# 6 你现在的 Hessian 结构非常漂亮

你已经推导出：

[
Hv
==

J^T\delta g(Jv)
+
\dot J^T g
]

这其实已经是：

# Riemannian Hessian

标准形式。

---

## 6.1 第一项

[
J^T\delta g(Jv)
]

是 pullback Hessian。

---

## 6.2 第二项

[
\dot J^T g
]

是 Weingarten / connection term。

这说明：

# 你已经不是“经验更新”了。

而是真正的：

[
\text{manifold optimization}
]

---

# 7 你现在最值得警惕的问题

虽然整体方向非常对，但有几个关键问题。

---

# 7.1 (L_a) 的 gauge redundancy 可能仍未完全去除

这是最危险点。

因为：

[
T_a = Q_iG_a + Q_aL_a
]

若：

[
Q_a \rightarrow Q_aU
]

则：

[
L_a \rightarrow U^{-1}L_a
]

物理不变。

因此：

[
L_a
]

内部仍可能有：

* rotational redundancy
* ill-conditioned flat directions

---

## 建议

不要允许一般：

[
L_a \in \mathbb R^{n_a\times n_a}
]

而应限制：

---

### 方案1（推荐）

Cholesky gauge：

[
L_a
===

\text{lower triangular},
\quad
(L_a)_{kk}>0
]

这样：

* gauge 唯一
* metric 唯一
* Hessian 稳定很多

这是最重要的改进。

---

### 方案2

SPD metric：

定义：

[
M_a=L_a^TL_a
]

直接优化 SPD manifold。

这更几何化，但实现复杂。

---

# 7.2 自方向可能导致 scale runaway

你已经意识到：

[
\delta T \sim T
]

是物理自由度。

但：

# scale directions 往往非常 stiff。

可能导致：

* Hessian 巨特征值
* line search 不稳定
* trust radius collapse

---

## 我强烈建议

不要直接优化：

[
L_{kk}
]

而优化：

[
\lambda_k = \log L_{kk}
]

即：

[
L_{kk}=e^{\lambda_k}
]

这样：

* positivity 自动满足
* scale curvature 更平滑
* Newton 更稳定

这是非常经典的 SPD trick。

---

# 8 你真正接近的对象是什么

我觉得你实际上在构造：

---

# Nonorthogonal orbital principal bundle

底空间：

[
\mathrm{Gr}(n,n_{\mathrm{occ}})
]

（occupied subspace）

纤维：

[
\mathrm{GL}(n_i)\times \mathrm{GL}(n_a)
]

而：

[
Q
]

是 horizontal geometry。

[
L,G,R
]

是 fiber coordinates。

---

这其实已经有：

* differential geometry
* gauge theory
* Riemannian optimization

的完整结构了。

这远超过普通 quantum chemistry orbital optimization。

---

# 9 我认为最值得发展的方向

我最推荐你继续发展的不是“再推公式”，而是：

---

# 9.1 真正构造 Riemannian Newton / RTR

因为你已经有：

* tangent space
* retraction
* Hessian-vector product

所以已经满足：

# Riemannian trust-region

全部要求。

---

## 这会带来巨大稳定性

相比：

* DIIS
* augmented Hessian
* quasi-Newton

RTR 在 ill-conditioned NONORTH VB 上可能优势巨大。

---

# 10 一个非常关键的观点

你现在的理论其实说明：

---

# “非正交轨道优化”并不应该直接在轨道系数空间做

而应该：

[
\text{Orthogonal geometry}
+
\text{Nonorthogonal metric}
]

分离。

这是核心思想。

---

# 11 我对这个框架的总体评价

我认为这是：

# 一个非常合理、而且有研究价值的非正交轨道流形优化框架。

尤其重要的是：

* 不是 ad hoc 参数化
* 有明确微分几何结构
* 有严格 tangent/retraction/Hessian
* 与 Riemannian optimization 完全兼容
* 保留 NONORTH VB 物理自由度

我认为下一步最重要的是：

---

## （1）Gauge fixing 完整化

尤其：

[
L_a
]

的规范化。

---

## （2）真正定义 manifold metric

比如：

[
g(\delta\eta_1,\delta\eta_2)
============================

\mathrm{Tr}(\Omega_1^T\Omega_2)
+
\alpha|\delta L|^2
+\cdots
]

这样才能有真正自然梯度。

---

## （3）RTR / Riemannian Newton

这是最可能产生“明显优于传统 VBSCF”的地方。
