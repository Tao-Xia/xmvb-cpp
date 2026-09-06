# same-spin 相对老版 Fortran 的优势总结

## 1. 目的

这份说明文档的目的，是把下面两件事明确区分开：

1. 老版 Fortran `RDM-VBSCF` / `exact` 路线里，same-spin 部分原本就已经做到的事情；
2. 当前 C++ `unique-spin-string` same-spin 项目，相对于老版 Fortran 真正新增的优势。

核心结论先写在前面：

$$
\text{如果以老版 Fortran exact RDM 路线为对照，}
$$

$$
\text{same-spin 的 half-determinant / unique-string 去重本身并不是我们从零发明的。}
$$

但是：

$$
\text{我们把这套思想显式提升为统一的 unique-spin 矩阵代数框架，}
$$

$$
\text{并把它真正打通到了当前 C++ production 的 exact 与 RI 路径。}
$$

因此，当前工作的优势不应表述为“老版 Fortran 完全没有 unique-string”，而应表述为：

$$
\boxed{
\text{显式 unique-spin 表示}
+\text{selected-state 矩阵压缩}
+\text{forward/backward 统一矩阵化}
+\text{exact/RI production 统一化}
}
$$

---

## 2. 老版 Fortran 已经做到的事情

### 2.1 已经有 unique alpha / unique beta half-determinant 去重

老版 Fortran 在 `hes_str_det.F90` 中，会先把 structure expansion 展开成 determinant，
然后把 determinant 再按 alpha / beta half-determinant 去重，得到：

$$
N_{h\alpha} = \text{Nhda}, \qquad
N_{h\beta} = \text{Nhdb}.
$$

代码证据：

- `/export/home/xiatao/new-xmvb/src/vbscf/hes_str_det.F90:55-63`
- `/export/home/xiatao/new-xmvb/src/vbscf/hes_str_det.F90:93-117`

从实现语义上看，这里的 `Nhda` / `Nhdb` 就是 old Fortran exact 路线中的
unique alpha / unique beta strings。

### 2.2 old Fortran 的 exact same-spin forward 已经在 half-determinant pair 上工作

老版 Fortran 在 `hes_hamhd.F90` 中构造：

$$
\mathbf{S}^{\alpha},\quad \mathbf{H}^{\alpha}
\in \mathbb{R}^{N_{h\alpha}\times N_{h\alpha}},
$$

以及

$$
\mathbf{S}^{\beta},\quad \mathbf{H}^{\beta}
\in \mathbb{R}^{N_{h\beta}\times N_{h\beta}}.
$$

其循环对象已经是：

$$
(i,j)\in \{1,\dots,N_{h\alpha}\}^2
\quad \text{或} \quad
(i,j)\in \{1,\dots,N_{h\beta}\}^2,
$$

而不是 full determinant pair。

代码证据：

- `/export/home/xiatao/new-xmvb/src/vbscf/hes_hamhd.F90:87-139`
- `/export/home/xiatao/new-xmvb/src/vbscf/hes_hamhd.F90:238-263`

因此，如果只讨论 old Fortran exact same-spin kernel 本身，它的主循环已经接近

$$
O(N_{h\alpha}^2 + N_{h\beta}^2).
$$

### 2.3 old Fortran 的 RDM / gradient 路线也已经显式生成 half-determinant pairs

在 `rdm_vbscf.F90` 中，Fortran 会显式生成：

$$
\mathrm{DETPAIRA}
=
\{(i,j)\mid 1\le i\le j\le N_{h\alpha}\},
$$

以及必要时的

$$
\mathrm{DETPAIRB}
=
\{(i,j)\mid 1\le i\le j\le N_{h\beta}\}.
$$

代码证据：

- `/export/home/xiatao/new-xmvb/src/vbscf/rdm_vbscf.F90:1776-1811`
- `/export/home/xiatao/new-xmvb/src/vbscf/rdm_vbscf.F90:1814-1825`

随后在 `gradient_rdm.F90` 的 `DenPiTou` 中，same-spin 相关循环实际扫的也是
`DETPAIRA`，而不是 full determinant pair：

- `/export/home/xiatao/new-xmvb/src/vbscf/gradient_rdm.F90:4243-4320`

所以，如果以 old Fortran exact RDM 内核为参照，不能说：

$$
\text{“以前完全没有 unique-string / half-determinant 去重”。}
$$

---

## 3. 我们不能再怎么表述

为了避免论文和汇报中的表述不准确，下面这些说法不应该再使用：

### 3.1 不能说 old Fortran 完全没有 unique-string

更准确的说法应当是：

$$
\text{old Fortran 在 exact RDM / same-spin 路线上已经隐式使用了 half-determinant 去重。}
$$

### 3.2 不能说历史上 same-spin 从 $O(N_{\mathrm{det}}^2)$ 降到 $O(N_\alpha^2+N_\beta^2)$ 这件事完全不存在

更准确地说：

$$
\text{对 old Fortran exact same-spin 内核而言，这种压缩思想本来就已经部分存在。}
$$

### 3.3 不能把我们的贡献只写成“cache”

因为当前 C++ 项目做的已经不是单纯 cache，而是：

$$
\text{显式 representation change} + \text{matrix-form reformulation}.
$$

---

## 4. 我们当前真正新增的优势

下面这些，才是当前 C++ same-spin 项目相对 old Fortran 真正新增、而且值得强调的地方。

### 4.1 把隐式的 half-determinant 组织，提升为显式的 unique-spin 表示

old Fortran 中的 `Nhda` / `Nhdb`、`SA/HA`、`DETPAIRA` 等对象都存在，但它们主要是
特定 exact RDM 实现中的内部数组。

当前 C++ 里，我们把这些内容显式抽象为：

1. determinant 到 unique alpha / beta ID 的映射；
2. structure coefficient matrix
   $$
   \mathbf{C}_I \in \mathbb{R}^{N_\alpha\times N_\beta};
   $$
3. selected-state coefficient matrix
   $$
   \mathbf{C}^{(n)} \in \mathbb{R}^{N_\alpha\times N_\beta}.
   $$

代码证据：

- [src/vb/scf/selected_state_determinant_matrices.hpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/selected_state_determinant_matrices.hpp)
- [src/vb/scf/selected_state_determinant_matrices.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/selected_state_determinant_matrices.cpp#L157)

这意味着 same-spin 去重不再只是“若干个 Fortran 内部数组的实现技巧”，而是被提升成了一个可以直接写理论公式、直接服务 forward/backward 的显式数学表示。

### 4.2 我们把 structure forward 写成了清晰的 unique-spin matrix form

对于 structure pair \( (I,J) \)，当前 C++ exact forward 可以直接写成：

$$
S_{IJ}
=
\left\langle
\mathbf{C}_I,\;
\mathbf{S}^{\alpha}\mathbf{C}_J(\mathbf{S}^{\beta})^{\mathrm T}
\right\rangle_{\mathrm F},
$$

$$
H_{IJ}^{\mathrm{same}}
=
\left\langle
\mathbf{C}_I,\;
\mathbf{H}^{\alpha}\mathbf{C}_J(\mathbf{S}^{\beta})^{\mathrm T}
+
\mathbf{S}^{\alpha}\mathbf{C}_J(\mathbf{H}^{\beta})^{\mathrm T}
\right\rangle_{\mathrm F}.
$$

代码证据：

- [src/vb/matrices/full_structure_builder.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/matrices/full_structure_builder.cpp#L1116)

这件事的意义在于：

$$
\text{same-spin 的数学结构第一次以统一、可写论文的矩阵形式暴露出来。}
$$

old Fortran 虽然内部已经有 `SA/HA/HB/SB`，但没有把整个 structure algebra 提炼成这种统一的 `\mathbf{C}_I` 矩阵收缩框架。

### 4.3 我们把 selected-state backward 从“逐态重复做 RDM”改成了“先压缩，再统一回传”

old Fortran 的 state-average 梯度主逻辑是：

1. 对每个 selected state \(n\)；
2. 单独调用一次 `GRAD_RDM5(...)`；
3. 再用该态权重把结果加到总梯度中。

代码证据：

- `/export/home/xiatao/new-xmvb/src/vbscf/gradient_rdm.F90:266-277`

也就是说，old Fortran 的状态平均逻辑是

$$
\text{for } n=1,\dots,N_{\mathrm{state}}
\quad
\Longrightarrow
\quad
\text{repeat one full state-specific RDM/gradient sweep.}
$$

当前 C++ 的思路则不同。我们先构造每个态的

$$
\mathbf{C}^{(n)},
$$

然后先压缩出 same-spin 权重矩阵，例如

$$
\mathbf{W}^{(H)}_\alpha
=
\sum_n w_n\,\mathbf{C}^{(n)}\mathbf{S}^{\beta}\mathbf{C}^{(n)\mathrm T},
$$

$$
\mathbf{W}^{(S)}_\alpha
=
\sum_n \left(-w_n E_n\right)\mathbf{C}^{(n)}\mathbf{S}^{\beta}\mathbf{C}^{(n)\mathrm T},
$$

beta 侧完全对称：

$$
\mathbf{W}^{(H)}_\beta
=
\sum_n w_n\,\mathbf{C}^{(n)\mathrm T}\mathbf{S}^{\alpha}\mathbf{C}^{(n)},
$$

$$
\mathbf{W}^{(S)}_\beta
=
\sum_n \left(-w_n E_n\right)\mathbf{C}^{(n)\mathrm T}\mathbf{S}^{\alpha}\mathbf{C}^{(n)}.
$$

然后再只做一次 unique pair scatter。

代码证据：

- [src/vb/scf/selected_state_determinant_matrices.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/selected_state_determinant_matrices.cpp#L157)
- [src/vb/scf/same_spin_matrix_backward.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/same_spin_matrix_backward.cpp#L576)
- [src/vb/scf/same_spin_matrix_backward.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/same_spin_matrix_backward.cpp#L835)

这一步是我们相对 old Fortran 在 backward 上最重要的新增优势。

它的本质不是简单 cache，而是：

$$
\boxed{
\text{state-average adjoint compression}
\;\Longrightarrow\;
\text{one shared unique-spin backward sweep}
}
$$

### 4.4 我们把 same-spin cache 做成了当前 determinant backend 的公共基础设施

当前 C++ 中，same-spin cache 不是某一个专用 RDM 子程序内部的私有数组，而是一个统一的上下文对象：

$$
\texttt{SameSpinPairCacheContext}.
$$

它会统一提供：

1. ordered unique alpha / beta pair 的 exact payload；
2. same-spin scalar / phi / inverse-overlap gradient；
3. opposite-spin 需要的 projected channel payload。

代码证据：

- [src/vb/matrices/same_spin_pair_cache.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/matrices/same_spin_pair_cache.cpp#L206)

这意味着：

$$
\text{same-spin 不再只是某条 exact RDM 公式的局部技巧，}
$$

$$
\text{而成为整个 active-space determinant algebra 的公共基础设施。}
$$

### 4.5 我们把 same-spin 框架统一接入了 exact 与 RI 两条 production 路径

old Fortran 的 half-determinant / same-spin 组织，主要体现在 exact RDM-VBSCF 的实现内部。

当前 C++ 项目则把同一套 unique-spin same-spin 表示同时接入：

1. exact structure build；
2. exact backward；
3. RI structure build；
4. RI active-space adjoint。

因此我们真正兑现出来的是：

$$
\text{one representation, two integral routes, same production interface}.
$$

这也是为什么在当前 RI production benchmark 中，same-spin 这条线可以给出非常大的端到端收益。

### 4.6 我们显式利用了 structure / selected-state 的局部 support

当前 C++ 中，每个 structure 或 selected state 不仅有全局
\(\mathbf{C}\) 矩阵，还会提取出局部 support：

$$
A_I = \{a \mid (\mathbf{C}_I)_{ab}\neq 0 \text{ for some } b\},
$$

$$
B_I = \{b \mid (\mathbf{C}_I)_{ab}\neq 0 \text{ for some } a\},
$$

以及局部块

$$
\mathbf{C}_I^{\mathrm{loc}}
=
\mathbf{C}_I[A_I,B_I].
$$

同样，对 selected state \(n\) 也有

$$
\mathbf{C}^{(n)}_{\mathrm{loc}}
=
\mathbf{C}^{(n)}[A_n,B_n].
$$

这使得 exact forward / backward 可以在局部 support 上做 dense contraction，而不是总在整块
\(N_\alpha\times N_\beta\) 矩阵上收缩。

这也是当前 same-spin 项目一个很实际的工程优势。

---

## 5. 标度层面的真实对比

### 5.1 old Fortran exact same-spin 内核

如果只看 old Fortran exact same-spin kernel，那么其工作对象已经是

$$
N_{h\alpha} = N_\alpha, \qquad N_{h\beta} = N_\beta,
$$

因此 same-spin kernel 本身的 pair 数量级已经是

$$
O(N_\alpha^2 + N_\beta^2).
$$

所以不能把我们的优势写成：

$$
\text{“我们第一次把 old Fortran 的 same-spin 从 }O(N_{\det}^2)\text{ 变成 }
O(N_\alpha^2 + N_\beta^2)\text{”。}
$$

### 5.2 当前 C++ 相对 determinant-pair baseline

对当前 C++ baseline 来说，情况完全不同。

如果没有 unique-spin matrix-form，则 structure builder / backward 容易重新掉回

$$
O(N_{\det}^2)
$$

的外层 determinant-pair 拓扑。

因此对当前 C++ 项目而言，我们的 same-spin 确实完成了实质性改进：

$$
\text{把 same-spin 主路径从 full determinant-pair topology 中剥离出来。}
$$

### 5.3 我们相对 old Fortran 真正更强的地方

更准确地说，我们的优势主要体现在下面这个层面：

old Fortran 的 state-average backward 更接近

$$
O\!\left(
N_{\mathrm{state}}
\cdot
\bigl[\text{one state-specific same-spin/RDM sweep}\bigr]
\right),
$$

而我们当前的 same-spin backward 是

$$
O\!\left(
N_{\mathrm{state}}
\left(
N_\alpha^2 N_\beta + N_\alpha N_\beta^2
\right)
+
N_\alpha^2 + N_\beta^2
\right),
$$

其中第一项是 selected-state 压缩，
第二项是压缩完成后只做一次 unique-pair scatter。

因此我们真正新增的算法点是：

$$
\boxed{
\text{selected-state compression}
\;+\;
\text{single shared unique-pair backward}
}
$$

而不是单独的 half-string 去重本身。

---

## 6. 对比表

| 项目 | 老版 Fortran exact/RDM | 当前 C++ unique-spin same-spin | 我们的净优势 |
| --- | --- | --- | --- |
| unique alpha / beta 去重 | 有 | 有 | 不是净新增 |
| same-spin exact forward 在 half-determinant pair 上求值 | 有 | 有 | 不是净新增 |
| `SA/HA/SB/HB` 这种 same-spin 对象 | 有 | 有 | 不是净新增 |
| 显式 `\mathbf{C}_I` structure matrix form | 没有形成统一公开框架 | 有 | 是 |
| 显式 `\mathbf{C}^{(n)}` selected-state matrix form | 没有形成统一公开框架 | 有 | 是 |
| state-average 先压缩再统一回传 | 没有，逐态重复 `GRAD_RDM5` | 有 | 是 |
| support-aware local block contraction | 没有显式统一框架 | 有 | 是 |
| same-spin cache 作为公共基础设施 | 没有 | 有 | 是 |
| exact / RI 统一 same-spin 框架 | 不明显，主要埋在 exact RDM 内核 | 有 | 是 |
| 适合写成清晰理论公式与论文 | 较弱 | 强 | 是 |

---

## 7. 论文与汇报里建议怎么说

如果以后要写文章或做汇报，我建议用下面这种口径：

### 7.1 可以说的

可以说：

$$
\text{我们提出了显式的 unique-spin-string 表示，}
$$

$$
\text{把 structure 和 selected-state determinant algebra 统一写成了矩阵形式。}
$$

可以说：

$$
\text{我们把 same-spin forward/backward 从当前 C++ determinant-pair 主循环中剥离出来，}
$$

$$
\text{并把 same-spin 压缩真正打通到了 exact 与 RI production 路径。}
$$

可以说：

$$
\text{我们新增了 selected-state adjoint compression，}
$$

$$
\text{使 state-average backward 可以先聚合，再做统一 unique-pair 回传。}
$$

### 7.2 不建议说的

不建议说：

$$
\text{“历史上没有 unique-spin / half-determinant 去重”。}
$$

也不建议说：

$$
\text{“old Fortran 的 same-spin 本质上还是 }O(N_{\det}^2)\text{”。}
$$

如果以 old Fortran exact RDM 路线为参照，这两句都不够准确。

---

## 8. 最终结论

最精炼的总结是：

$$
\text{old Fortran 已经有 same-spin 的 half-determinant 去重。}
$$

但是当前 C++ 项目的价值在于：

$$
\boxed{
\text{把这套思想显式提升为统一的 unique-spin 矩阵代数框架，}
}
$$

$$
\boxed{
\text{并进一步完成了 selected-state 压缩、forward/backward 统一矩阵化、}
}
$$

$$
\boxed{
\text{support-aware 局部收缩，以及 exact/RI production 统一接入。}
}
$$

如果只用一句话来概括我们相对 old Fortran 的优势，我建议写成：

$$
\boxed{
\text{我们的优势不在“第一次想到 half-string 去重”，}
\text{而在于把它发展成了可统一、可扩展、可落地到当前 production 路径的}
\text{ exact unique-spin matrix-form framework。}
}
$$
