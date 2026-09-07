# VB 稀疏轨道 TNHVP 梯度更新几何推导

> **历史推导（部分失效）**：其中关于“当前 mixed chart”的描述不再对应
> 代码。internal chart 已删除；现实现只保留物理严格稀疏 (U_p) 坐标。
> 非正交轨道、support 与商空间的基础推导仍可作为历史参考。

## 1. 问题背景

VBSCF 的轨道变量和常规正交 MO 优化不一样：

1. 轨道按物理语义分为 inactive doubly occupied orbitals 和 active orbitals。
2. active orbitals 允许非正交，active-active overlap 是物理自由度的一部分。
3. inactive orbitals 可以通过正交化辅助 frame 简化计算，但物理存储仍是 sparse orbital coefficients。
4. sparse orbital 有固定 support，任意列旋转都会把某个轨道的分量泄露到原 support 之外。

因此，TNHVP 的 reduced gradient / HVP / finite trial step 必须使用同一个几何定义。否则 inner TN 模型预测的 trial point 和实际写回的 sparse orbital point 不一致，trust-region 比值会失真。

本文档只讨论公式和算法目标，不讨论具体代码改动。

## 2. 物理稀疏轨道流形

考虑一个 block \(b\)。令 block-local AO support 为

\[
\Omega_b = \{\mu_1,\ldots,\mu_{N_b}\},
\qquad
S_b \in \mathbb{R}^{N_b\times N_b}.
\]

每个轨道 \(p\) 只有自己的 sparse support \(J_{b,p}\subseteq \Omega_b\)。用注入矩阵

\[
U_{b,p}\in \mathbb{R}^{N_b\times m_{b,p}}
\]

把局部 sparse coefficient \(x_{b,p}\in\mathbb{R}^{m_{b,p}}\) 嵌入 block AO 空间：

\[
c_{b,p}=U_{b,p}x_{b,p}.
\]

所以 sparse support 约束的本质是

\[
\delta c_{b,p}\in \mathrm{range}(U_{b,p}).
\]

如果轨道逐列归一化，则还应满足

\[
x_{b,p}^{\mathsf T} S_{b,p} x_{b,p}=1,
\qquad
S_{b,p}=U_{b,p}^{\mathsf T}S_bU_{b,p}.
\]

它的线性化是

\[
x_{b,p}^{\mathsf T}S_{b,p}\delta x_{b,p}=0.
\]

这里要注意：逐列归一化只是每个 sparse orbital 自己的局部约束；它不能自动保持整个 occupied block 的 overlap/gauge 几何。

## 3. Inactive-active 混合辅助 chart

对一个 block，收集 inactive 和 active 物理轨道：

\[
C_i=[c_{i_1},\ldots,c_{i_{n_i}}],
\qquad
C_a=[c_{a_1},\ldots,c_{a_{n_a}}].
\]

定义 inactive overlap：

\[
M_i=C_i^{\mathsf T}S_bC_i.
\]

构造 inactive orthonormal working frame：

\[
Q_i=C_iM_i^{-1/2},
\qquad
Q_i^{\mathsf T}S_bQ_i=I.
\]

同时保留右侧 inactive shape：

\[
R_i=M_i^{1/2},
\qquad
C_i=Q_iR_i.
\]

把 active 轨道投影到 inactive 正交补：

\[
P_i=Q_iQ_i^{\mathsf T}S_b,
\qquad
T_a=(I-P_i)C_a.
\]

于是

\[
Q_i^{\mathsf T}S_bT_a=0.
\]

active auxiliary overlap 为

\[
A_a=T_a^{\mathsf T}S_bT_a.
\]

再定义

\[
Q_a=T_aA_a^{-1/2},
\qquad
L_a=A_a^{1/2},
\qquad
T_a=Q_aL_a.
\]

所以

\[
Q_a^{\mathsf T}S_bQ_a=I,
\qquad
Q_i^{\mathsf T}S_bQ_a=0.
\]

最后定义 active 在 inactive frame 上的 gauge：

\[
G=Q_i^{\mathsf T}S_bC_a.
\]

于是物理 active 轨道可写成

\[
C_a=Q_iG+Q_aL_a.
\]

这个分解的含义是：

1. \(Q_i,Q_a\) 是可以做正交 frame rotation 的部分。
2. \(R_i\) 表示 inactive 内部 metric shape，通常在一个 finite step 内固定。
3. \(G\) 表示 active 相对 inactive 的 gauge/overlap，通常在一个 finite step 内固定。
4. \(L_a\) 是 active 非正交 shape，是物理自由度，必须允许更新。

这里没有假设原始轨道正交。完整 occupied overlap 是

\[
\begin{bmatrix}
C_i^{\mathsf T}S_bC_i & C_i^{\mathsf T}S_bC_a\\
C_a^{\mathsf T}S_bC_i & C_a^{\mathsf T}S_bC_a
\end{bmatrix}
=
\begin{bmatrix}
R_i^{\mathsf T}R_i & R_i^{\mathsf T}G\\
G^{\mathsf T}R_i & G^{\mathsf T}G+L_a^{\mathsf T}L_a
\end{bmatrix}.
\]

也就是说：

1. inactive-inactive 非正交性在 \(R_i\) 中；
2. inactive-active 非正交性在 \(G\) 和 \(R_i\) 中；
3. active-active 非正交性在 \(G^{\mathsf T}G+L_a^{\mathsf T}L_a\) 中；
4. \(Q_i,Q_a,Q_v\) 只是辅助正交 frame，不是把待优化的物理轨道强行正交化。

## 4. Dense mixed chart 的一阶切向量

令 \(Q_v\) 是 \(S_b\)-orthonormal virtual complement，满足

\[
[Q_i,Q_a,Q_v]^{\mathsf T}S_b[Q_i,Q_a,Q_v]=I.
\]

reduced 方向包含：

\[
K_{ai}\in\mathbb{R}^{n_a\times n_i},
\quad
K_{vi}\in\mathbb{R}^{n_v\times n_i},
\quad
K_{va}\in\mathbb{R}^{n_v\times n_a},
\quad
\Delta L_a\in\mathbb{R}^{n_a\times n_a}.
\]

如果把原始非正交轨道的所有 shape/gauge 变化都写出来，还可以包含

\[
\Delta R_i\in\mathbb{R}^{n_i\times n_i},
\qquad
\Delta G\in\mathbb{R}^{n_i\times n_a}.
\]

其中：

1. \(K\) 是辅助正交 frame rotation，自由度是 inactive-active、inactive-virtual、active-virtual；
2. \(\Delta R_i\) 改变 inactive-inactive overlap；
3. \(\Delta G\) 改变 inactive-active overlap/gauge；
4. \(\Delta L_a\) 改变 active projected nonorthogonal shape。

在接受点的一阶切向量为

\[
\delta C_i
=Q_i\Delta R_i+Q_aK_{ai}R_i+Q_vK_{vi}R_i,
\]

\[
\delta C_a
=Q_i(\Delta G-K_{ai}^{\mathsf T}L_a)
+Q_a(\Delta L_a+K_{ai}G)
+Q_v(K_{va}L_a+K_{vi}G).
\]

如果选择固定 inactive internal shape 和 inactive-active gauge，则取

\[
\Delta R_i=0,
\qquad
\Delta G=0.
\]

这就是当前 mixed-chart 公式使用的 gauge-fixed 子空间。它不是说原始轨道正交，而是把 \(R_i,G\) 当作 accepted-point gauge/shape，在一个 trust-region trial 内固定。

这组公式很重要，因为它同时保持了 VB 轨道意义：

1. inactive-inactive rotation 不出现，它是 inactive closed-shell gauge 冗余。
2. active-active 不作为正交 rotation 出现，而是通过 \(\Delta L_a\) 改变 active 非正交 shape。
3. inactive-active rotation 是真实物理自由度，因为它改变 inactive/active partition。
4. occupied-virtual rotation 是真实 orbital relaxation。

## 5. Dense finite retraction 应保持的几何

令

\[
\Omega=
\begin{bmatrix}
0 & -K_{ai}^{\mathsf T} & -K_{vi}^{\mathsf T}\\
K_{ai} & 0 & -K_{va}^{\mathsf T}\\
K_{vi} & K_{va} & 0
\end{bmatrix}.
\]

用 Cayley update：

\[
[Q_i^+,Q_a^+,Q_v^+]
=
[Q_i,Q_a,Q_v]
\left(I-\frac12\Omega\right)^{-1}
\left(I+\frac12\Omega\right).
\]

然后

\[
C_i^+=Q_i^+(R_i+\Delta R_i),
\]

\[
C_a^+=Q_i^+(G+\Delta G)+Q_a^+(L_a+\Delta L_a).
\]

这个 dense mixed retraction 有几个关键性质：

\[
(Q_i^+)^{\mathsf T}S_bQ_i^+=I,
\qquad
(Q_i^+)^{\mathsf T}S_bQ_a^+=0.
\]

并且

在 gauge-fixed \(\Delta R_i=\Delta G=0\) 情况下，

\[
(C_i^+)^{\mathsf T}S_bC_i^+
=R_i^{\mathsf T}R_i
=C_i^{\mathsf T}S_bC_i,
\]

\[
(C_i^+)^{\mathsf T}S_bC_a^+
=R_i^{\mathsf T}G
=C_i^{\mathsf T}S_bC_a.
\]

也就是说，在这个 chart 里，一个 finite step 默认保持 inactive metric 和 inactive-active overlap/gauge，只让 active shape 通过 \(L_a+\Delta L_a\) 变化。

这正是辅助 active orbitals 能简化计算的原因：inactive-active 正交关系在内部 frame 里是精确定义的。

如果算法决定把 \(R_i\) 或 \(G\) 也作为可优化物理自由度，则上面两个保持关系不应强加；对应的 overlap 变化为

\[
\delta(C_i^{\mathsf T}S_bC_i)
=
\Delta R_i^{\mathsf T}R_i+R_i^{\mathsf T}\Delta R_i,
\]

\[
\delta(C_i^{\mathsf T}S_bC_a)
=
\Delta R_i^{\mathsf T}G+R_i^{\mathsf T}\Delta G.
\]

这时 reduced gradient / HVP / trust norm 必须把 \(\Delta R_i,\Delta G\) 也纳入同一个 sparse tangent map。

## 6. Sparse support 下真正的问题

Dense formula 得到 \(C_i^*,C_a^*\) 后，如果每个轨道都可以写 full block support，那么直接写回即可。

但 sparse 轨道只能写

\[
c_p^+=U_px_p^+.
\]

如果简单做：

1. 从 dense target \(c_p^*\) 截取到 support \(J_p\)；
2. 每个轨道单独归一化 \(x_p^{\mathsf T}S_px_p=1\)；

则只能保证每列自己的 norm，不能保证 block-level 几何：

\[
(C_i^+)^{\mathsf T}S_bC_i^+
\approx
C_i^{\mathsf T}S_bC_i,
\]

\[
(C_i^+)^{\mathsf T}S_bC_a^+
\approx
C_i^{\mathsf T}S_bC_a.
\]

这个近似在小步时可能足够，但大步时会破坏 TN 模型假设。MnF2 的首个坏步就是这种情况：最大 overlap drift 出现在 inactive-active block，而不是 inactive-inactive 或 active-active block。

所以，“inactive-active overlap 被改坏”的准确含义是：

\[
\Delta_{ia}
=
(C_i^+)^{\mathsf T}S_bC_a^+
-
C_i^{\mathsf T}S_bC_a
\]

在 sparse finite retraction 后不再是高阶小量，而变成了和 trust step 同阶的有限误差。

## 7. Reduced gradient 的正确抽象

设 packed sparse 参数为 \(x\)，能量为 \(E(x)\)。对任意 packed 方向 \(\delta x\)，代码里的 packed gradient \(\bar x\) 应满足

\[
\delta E=\bar x^{\mathsf T}\delta x.
\]

如果 reduced 坐标为 \(\theta\)，接受点处的切向量映射为

\[
\delta x = J\delta\theta.
\]

那么 reduced gradient covector 必须是 pullback：

\[
g_\theta=J^{\mathsf T}\bar x.
\]

如果 \(J\) 的列不是正交的，reduced 坐标的 metric 是

\[
M_\theta=J^{\mathsf T}J.
\]

于是 steepest descent vector 不是 \(-g_\theta\)，而是

\[
\delta\theta_{\mathrm{sd}}
=-M_\theta^{-1}g_\theta.
\]

TN trust-region 也不应只用 \(\|\delta\theta\|_2\)，而应使用物理 step norm：

\[
\|\delta x\|^2
=
\delta\theta^{\mathsf T}M_\theta\delta\theta.
\]

这给出一个基本原则：

\[
\boxed{
\text{gradient projection、HVP、trust norm、finite retraction 必须使用同一个 }J。
}
\]

如果 \(J\) 是 dense mixed chart 的切向量，但 finite retraction 实际是 support 截断加逐列归一化，那么 TN 模型在大步时一定会出现 mismatch。

## 8. Dense mixed chart 的梯度分量

先忽略 sparse support，设 coefficient gradients 为

\[
B_i=\frac{\partial E}{\partial C_i},
\qquad
B_a=\frac{\partial E}{\partial C_a},
\]

并且

\[
\delta E=\mathrm{tr}(B_i^{\mathsf T}\delta C_i)
+\mathrm{tr}(B_a^{\mathsf T}\delta C_a).
\]

把第 4 节的一阶切向量代入，可得 dense mixed reduced gradient：

\[
g_{\Delta R_i}
=
Q_i^{\mathsf T}B_i,
\]

\[
g_{\Delta G}
=
Q_i^{\mathsf T}B_a,
\]

\[
g_{K_{ai}}
=
Q_a^{\mathsf T}B_iR_i^{\mathsf T}
+Q_a^{\mathsf T}B_aG^{\mathsf T}
-L_aB_a^{\mathsf T}Q_i,
\]

\[
g_{K_{vi}}
=
Q_v^{\mathsf T}B_iR_i^{\mathsf T}
+Q_v^{\mathsf T}B_aG^{\mathsf T},
\]

\[
g_{K_{va}}
=
Q_v^{\mathsf T}B_aL_a^{\mathsf T},
\]

\[
g_{\Delta L_a}
=
Q_a^{\mathsf T}B_a.
\]

如果采用 gauge-fixed 子空间 \(\Delta R_i=\Delta G=0\)，则 \(g_{\Delta R_i}\) 和 \(g_{\Delta G}\) 不进入 TN reduced vector；它们对应的 packed 梯度分量应被投影掉，而不是误混到 \(K_{ai},K_{vi},K_{va},\Delta L_a\) 里。

如果实现里使用的是 \(S\)-metric gradient \(F\)，即

\[
\delta E=\mathrm{tr}(F_i^{\mathsf T}S_b\delta C_i)
+\mathrm{tr}(F_a^{\mathsf T}S_b\delta C_a),
\]

则上式中的 \(Q^{\mathsf T}B\) 应替换为 \(Q^{\mathsf T}S_bF\)。

这组公式说明了 active 非正交 shape 和 orbital rotation 的梯度应分开处理。特别是 active-active 自由度不应被错误地当成正交 rotation。

## 9. Sparse support 下的最佳 reduced gradient

Sparse 情况下，不能直接使用 dense \(Q_i,Q_a,Q_v\) 全列，因为方向必须落在每个轨道自己的 support 内。

定义 dense mixed chart basis 为 \(B_\theta\)，即第 4 节公式给出的 \(\delta C(\theta)\)。对每个 basis direction，理想的 sparse tangent 应通过 block-level constrained projection 得到：

\[
\delta x(\theta)
=
\arg\min_{\delta x}
\left\|
U\delta x-\delta C(\theta)
\right\|_{S_b}^2
\]

subject to

\[
x_p^{\mathsf T}S_p\delta x_p=0
\quad\text{for every orbital }p,
\]

以及保持 mixed chart 语义的一阶 block 约束：

\[
\delta(C_i^{\mathsf T}S_bC_i)=0,
\]

\[
\delta(C_i^{\mathsf T}S_bC_a)=0
\quad
\text{for pure rotation / fixed }G\text{ directions}.
\]

如果 \(\Delta R_i,\Delta G\) 也作为 reduced variables，则这些 block 约束不能写成 0，而应匹配第 5 节的目标 overlap variation：

\[
\delta(C_i^{\mathsf T}S_bC_i)
=
\Delta R_i^{\mathsf T}R_i+R_i^{\mathsf T}\Delta R_i,
\]

\[
\delta(C_i^{\mathsf T}S_bC_a)
=
\Delta R_i^{\mathsf T}G+R_i^{\mathsf T}\Delta G.
\]

对 active shape 方向，约束应改成让 active auxiliary overlap 跟随

\[
\delta(A_a)=
\delta(L_a^{\mathsf T}L_a)
=
L_a^{\mathsf T}\Delta L_a+\Delta L_a^{\mathsf T}L_a.
\]

这样得到的 sparse tangent map 记为

\[
J_{\mathrm{sparse}}.
\]

那么最佳 reduced gradient 是

\[
\boxed{
g_\theta
=
J_{\mathrm{sparse}}^{\mathsf T}\bar x.
}
\]

对应 metric 是

\[
\boxed{
M_\theta
=
J_{\mathrm{sparse}}^{\mathsf T}J_{\mathrm{sparse}}.
}
\]

如果为了速度不显式构造 \(J_{\mathrm{sparse}}\)，也必须保证代码里的 projection 和 expansion 是这两个线性算子的 adjoint pair：

\[
\langle \bar x,J_{\mathrm{sparse}}\delta\theta\rangle
=
\langle J_{\mathrm{sparse}}^{\mathsf T}\bar x,\delta\theta\rangle.
\]

这比“先投到内部正交 chart，再随便截 support”更重要。只要 adjoint pair 不一致，TNHVP 的梯度和 HVP 就会在数值上不自洽。

## 10. TNHVP 的正确 reduced Hessian action

在 accepted point \(x_k\)，reduced gradient 是

\[
g_\theta(x_k)=J_k^{\mathsf T}\nabla_xE(x_k).
\]

对 reduced 方向 \(p\)，理想的 reduced HVP 是方向导数：

\[
H_\theta p
=
\left.
\frac{d}{d\epsilon}
g_\theta(R_k(\epsilon p))
\right|_{\epsilon=0}.
\]

展开后包含两部分：

\[
H_\theta p
=
J_k^{\mathsf T}H_xJ_kp
+
\left(\frac{dJ}{d\epsilon}[p]\right)^{\mathsf T}
\nabla_xE(x_k).
\]

第一项是常见的 packed HVP pullback。第二项是 chart/retraction 曲率项。远离收敛时第二项可能不小，尤其是 sparse retraction 非线性明显时。

实际 TNHVP 可以有两个层级：

1. 严格层级：
   \[
   H_\theta p
   =
   \frac{g_\theta(R_k(\epsilon p))-g_\theta(x_k)}{\epsilon}
   \]
   其中新点必须重新做 orbital preparation、重新构造同一类 chart、再 pullback gradient。
2. 近似层级：
   \[
   H_\theta p\approx J_k^{\mathsf T}H_xJ_kp
   \]
   但 trust region 必须足够小，并且 finite retraction 的几何 drift 必须受控。

MnF2 这类 open-shell sparse system 的问题是：近似层级在大步时不够可靠，因为 \(J_kp\) 对应的内部 mixed tangent 和实际 sparse retraction 后的 finite displacement 偏离太大。

## 11. Finite retraction 的推荐数学形式

为了让 TNHVP 模型和 sparse support 一致，finite retraction 不应定义成“逐列截断 + 逐列归一化”。更合理的定义是 block-level constrained retraction：

给定 dense mixed target

\[
C_i^*=Q_i^+R_i,
\qquad
C_a^*=Q_i^+G+Q_a^+(L_a+\Delta L_a),
\]

求 sparse trial

\[
X^+
=
\arg\min_X
\left\|
U X-C^*
\right\|_{S_b}^2
\]

subject to

\[
(x_p^+)^{\mathsf T}S_p x_p^+=1,
\]

\[
(C_i^+)^{\mathsf T}S_bC_i^+
=
C_i^{\mathsf T}S_bC_i
\quad
\text{or as close as feasible},
\]

\[
(C_i^+)^{\mathsf T}S_bC_a^+
=
C_i^{\mathsf T}S_bC_a
\quad
\text{or as close as feasible},
\]

and

\[
(T_a^+)^{\mathsf T}S_bT_a^+
\approx
(L_a+\Delta L_a)^{\mathsf T}(L_a+\Delta L_a).
\]

如果这些约束在当前 sparse support 下不可行，正确响应不是强行写回，而是缩小 trust radius，或者改变 support。这样才能保留 orbital meaning。

## 12. 梯度更新策略结论

从公式上看，最合理的 TNHVP 梯度更新方式是：

1. 每个 accepted point 从物理 sparse orbitals \(C_i,C_a\) 重新构造 \(Q_i,Q_a,Q_v,R_i,G,L_a\)。
2. 定义 support-compatible tangent map \(J_{\mathrm{sparse}}\)，它是 dense mixed tangent 到 sparse support tangent 的 constrained projection。
3. 用
   \[
   g_\theta=J_{\mathrm{sparse}}^{\mathsf T}\nabla_xE
   \]
   得到 reduced gradient。
4. 用
   \[
   M_\theta=J_{\mathrm{sparse}}^{\mathsf T}J_{\mathrm{sparse}}
   \]
   定义 trust-region norm 和 preconditioner。
5. HVP 使用同一个 \(J_{\mathrm{sparse}}\) 做 pushforward 和 pullback；如果用 finite-difference HVP，则 perturbed point 必须通过同一个 block-level retraction 产生。
6. accepted step 后不要把旧 reduced gradient 当成新 chart 下的对象直接复用；secant pair 或 transported history 必须经过同一几何下的 vector transport。

最核心的一句话是：

\[
\boxed{
\text{VB sparse TNHVP 的 reduced gradient 必须是“物理 sparse 流形上的梯度”拉回到 mixed chart，}
\text{而不是 dense mixed chart 梯度再事后截断。}
}
\]

## 13. 对当前 MnF2 现象的解释

当前实现已经在内部使用 mixed chart，并且 dense full-support 情况下几何是清楚的。但 sparse finite step 仍然包含逐轨道 support restriction 和 local normalization。

因此大步时会出现：

\[
(C_i^+)^{\mathsf T}S_bC_a^+
-
C_i^{\mathsf T}S_bC_a
\]

过大，导致 TNHVP 预测模型和真实 trial energy 不一致。

这不是简单内存连续性问题，也不像单个 HVP kernel bug。它是几何定义不一致：

1. gradient/HVP 认为自己在内部 mixed chart 上工作；
2. finite trial 实际落在 per-orbital sparse support normalized chart 上；
3. 两者在小步近似一致，在大步或 open-shell near-degenerate system 上明显分离。

所以后续真正值得实现的不是兼容旧轨道空间，而是把 sparse support retraction、gradient pullback、HVP pushforward/pullback 统一到同一个 block-level constrained geometry。

## 14. 代码落实计划

本节给出后续代码修改计划。目标不是引入两套非冗余轨道空间，而是在当前 mixed chart 上把 sparse support 的 tangent、gradient、HVP 和 finite retraction 统一起来。

### 14.1 先固定算法选择

第一版实现采用 gauge-fixed mixed chart：

\[
C_i=Q_iR_i,
\qquad
C_a=Q_iG+Q_aL_a,
\]

优化变量仍为

\[
K_{ai},K_{vi},K_{va},\Delta L_a.
\]

暂时不加入 \(\Delta R_i,\Delta G\)。原因是：

1. inactive-inactive shape \(R_i\) 对 closed-shell inactive 子空间大多是 gauge/metric 代表，不应先作为自由优化变量引入；
2. inactive-active gauge \(G\) 已经由 active projection / auxiliary active orbitals 间接处理，直接优化会增加病态自由度；
3. 当前 MnF2 暴露的问题不是缺少 \(\Delta R_i,\Delta G\)，而是选择固定 \(R_i,G\) 后 sparse finite retraction 没有保持它们。

因此第一版正确性目标是：

\[
\delta(C_i^{\mathsf T}S C_i)=0,
\qquad
\delta(C_i^{\mathsf T}S C_a)=0
\]

对 rotation-like directions 成立；active shape 方向只允许 projected active overlap 按 \(\Delta L_a\) 改变。

### 14.2 新增 block-level sparse tangent 算子

在 `NonredundantOrbitalSpace` 内部新增 block-local 线性算子概念：

\[
J_b:\delta\theta_b\mapsto \delta x_b.
\]

它接收一个 block reduced direction，输出 packed sparse tangent slots。

对每个 block，先由 dense mixed chart 生成目标方向

\[
\delta C_b^{\mathrm{target}}(\delta\theta_b).
\]

然后求 support-compatible tangent：

\[
\delta x_b
=
\arg\min_{\delta x}
\left\|
U_b\delta x-\delta C_b^{\mathrm{target}}
\right\|_{S_b}^2
\]

subject to local norm tangent constraints：

\[
x_p^{\mathsf T}S_p\delta x_p=0.
\]

并加入 gauge-fixed block constraints：

\[
\delta(C_i^{\mathsf T}S_bC_i)=0,
\]

\[
\delta(C_i^{\mathsf T}S_bC_a)=0.
\]

实际实现建议：

1. 每个 block 构造局部 packed sparse slot 向量 \(x_b\)。
2. 构造二次型 \(H_b=U_b^{\mathsf T}S_bU_b\)。
3. 构造约束矩阵 \(A_b\) 和右端 \(r_b(\delta\theta_b)\)。
4. 解 KKT 系统：
   \[
   \begin{bmatrix}
   H_b & A_b^{\mathsf T}\\
   A_b & 0
   \end{bmatrix}
   \begin{bmatrix}
   \delta x_b\\
   \lambda
   \end{bmatrix}
   =
   \begin{bmatrix}
   U_b^{\mathsf T}S_b\delta C_b^{\mathrm{target}}\\
   r_b
   \end{bmatrix}.
   \]
5. 如果约束 rank 不足或不可解，先使用 rank-revealing solver；不要静默退回逐列截断。

对应代码位置：

1. `src/vb/orbital/nonredundant_orbital_space.hpp`
2. `src/vb/orbital/nonredundant_orbital_space.cpp`

建议新增 helper：

1. `build_sparse_mixed_tangent_projection_cache(...)`
2. `apply_sparse_mixed_tangent(...)`
3. `apply_sparse_mixed_tangent_transpose(...)`
4. `apply_sparse_mixed_tangent_metric(...)`

### 14.3 让 projection 和 expansion 成为严格 adjoint pair

当前相关函数是：

1. `accumulate_block_candidate_combination(...)`
2. `project_block_candidate_overlap(...)`
3. `project_impl(...)`
4. `expand_step(...)`
5. `expand_retract_input_tangent(...)`

改造目标：

\[
\langle g_x,Jp\rangle
=
\langle J^{\mathsf T}g_x,p\rangle
\]

必须在数值上成立。

具体计划：

1. `accumulate_block_candidate_combination()` 的 sparse 分支改为调用 `apply_sparse_mixed_tangent()`。
2. `project_block_candidate_overlap()` 的 sparse 分支改为调用 `apply_sparse_mixed_tangent_transpose()`。
3. `build_sparse_mixed_chart_metric_diagonal()` 和 block preconditioner 改为基于 \(J^{\mathsf T}J\)，至少先支持 `apply_sparse_mixed_tangent_metric(p)=J^{\mathsf T}Jp`。
4. 添加 debug-only 或 env-gated adjoint check：
   \[
   |g^{\mathsf T}Jp-(J^{\mathsf T}g)^{\mathsf T}p|
   \]
   要达到 double 精度下合理误差。

### 14.4 Trust-region norm 改成 sparse tangent metric

TN 目前主要使用 reduced Euclidean norm 和 packed/retraction tangent norm 的补丁式限制。正确做法是使用：

\[
\|p\|_J^2=p^{\mathsf T}J^{\mathsf T}Jp.
\]

实现阶段可以分两步：

1. 第一阶段：保留现有 TN solver，只在 trial 前用 \(J\)-norm 做可靠裁剪，并记录日志。
2. 第二阶段：把 `solve_nonredundant_truncated_newton_step()` 的 trust-region boundary 从 Euclidean norm 改为 metric norm。

涉及文件：

1. `src/vb/scf/cpp_vb_scf_optimizer.cpp`
2. `src/vb/orbital/nonredundant_orbital_space.hpp`
3. `src/vb/orbital/nonredundant_orbital_space.cpp`

验收指标：

1. MnF2 首个 `radius=1` 大步不应再出现 inactive-active overlap drift 远大于 accepted step 的情况；
2. rejected trial 数应下降；
3. 241、TiCl、FeCl 不应因为 trust norm 改变而明显变慢。

### 14.5 Sparse finite retraction 改为 block-level constrained retraction

当前 sparse retraction 的问题是：

1. dense target column；
2. 截到每个 orbital support；
3. 每列 local normalization。

这只能保持每个 orbital norm，不能保持 block-level \(R_i,G\)。

新 retraction 应求：

\[
X_b^+
=
\arg\min_X
\left\|
U_bX-C_b^{\mathrm{target}}
\right\|_{S_b}^2
\]

subject to：

\[
(x_p^+)^{\mathsf T}S_p x_p^+=1,
\]

\[
(C_i^+)^{\mathsf T}S_bC_i^+
\approx
C_i^{\mathsf T}S_bC_i,
\]

\[
(C_i^+)^{\mathsf T}S_bC_a^+
\approx
C_i^{\mathsf T}S_bC_a.
\]

实现建议：

1. 先实现一阶一致版本：使用 `x + Jp` 加局部二阶归一化，确保 retraction derivative 正好是 \(J\)。
2. 再实现 finite constrained solve：从 `x + Jp` 初始化，做少量 Newton/KKT correction 保持 norm 和 gauge。
3. 如果 correction 不收敛或几何残差过大，返回 failure，让 optimizer 缩小 trust radius，而不是强行接受 distorted point。

相关函数：

1. `NonredundantOrbitalSpace::retract_step(...)`
2. `NonredundantOrbitalSpace::expand_retract_input_tangent(...)`
3. 当前新增的 `diagnose_retraction_geometry(...)`

### 14.6 HVP 路径统一同一个 \(J\)

TNHVP 的 reduced HVP 应尽量满足：

\[
H_\theta p
=
\frac{g_\theta(R(\epsilon p))-g_\theta(x)}{\epsilon}
\]

或近似为：

\[
J^{\mathsf T}H_xJp.
\]

因此：

1. exact_ctx direct-action HVP 的 pushforward 方向必须用 `apply_sparse_mixed_tangent()`。
2. gradient pullback 必须用 `apply_sparse_mixed_tangent_transpose()`。
3. finite-difference HVP 的 perturbed point 必须由新的 `retract_step()` 生成。
4. 如果 chart 在 perturbed point 重建，必须明确这是 full finite-difference HVP；如果固定 accepted-point \(J\)，就只能宣称是 accepted-point model HVP。

涉及文件：

1. `src/vb/scf/cpp_vb_scf_optimizer.cpp`
2. `src/vb/scf/exact_orbital_second_order_operator.cpp`
3. `src/vb/scf/exact_orbital_second_order_operator.hpp`
4. `src/vb/orbital/nonredundant_orbital_space.cpp`

### 14.7 分阶段测试计划

第一阶段只测线性几何：

1. 对随机 reduced vector \(p\)，检查 `expand_step(p)` 是否满足 sparse support。
2. 检查每列 norm tangent：
   \[
   x_p^{\mathsf T}S_p\delta x_p\approx 0.
   \]
3. 检查 inactive metric tangent：
   \[
   \delta(C_i^{\mathsf T}SC_i)\approx 0.
   \]
4. 检查 inactive-active gauge tangent：
   \[
   \delta(C_i^{\mathsf T}SC_a)\approx 0.
   \]
5. 检查 adjoint：
   \[
   g^{\mathsf T}Jp=(J^{\mathsf T}g)^{\mathsf T}p.
   \]

第二阶段测 finite retraction：

1. 对小 \(p\)，验证
   \[
   \frac{R(\epsilon p)-x}{\epsilon}\to Jp.
   \]
2. 对中等 \(p\)，验证 inactive-active overlap drift 比当前逐列 retraction 小。
3. 对 MnF2 首步，`norm_ovlp_delta` 和 `inactive_active_ovlp_delta` 不应出现 `0.23` 级别的大漂移。

第三阶段测优化表现：

1. `241_VBSCF`：确认 wall-time 不退化，迭代数不明显增加。
2. `MnF2`：比较 rejected trial 数、80 步 projected gradient、总 wall-time。
3. `FeCl`：确认 open-shell sparse 不出现更差 tail stall。
4. `TiCl`：确认小 open-shell 体系不被过强约束拖慢。

### 14.8 推荐提交顺序

建议不要一次性大改。推荐顺序：

1. 提交 1：新增 sparse tangent projection cache 和 adjoint check，只挂诊断，不改变默认优化。
2. 提交 2：把 `project_block_candidate_overlap()` / `accumulate_block_candidate_combination()` 切到新的 adjoint pair，默认只对 sparse mixed blocks 生效。
3. 提交 3：把 trust norm / clipping 改为 \(J^{\mathsf T}J\) metric，先 env-gated。
4. 提交 4：实现一阶一致 sparse retraction，让 `expand_retract_input_tangent()` 与 `retract_step()` 导数一致。
5. 提交 5：实现 finite block-level constrained retraction，替换逐列 local normalization。
6. 提交 6：清理旧的 geometry prescreen 临时开关，把它降级为诊断或删除。

每一步都必须能独立 benchmark，避免把 gradient、HVP、retraction 三个风险同时引入。

## 15. 2026-04-26 执行记录

已完成第一轮代码落地，但 constrained sparse tangent 仍保持 env-gated，尚未默认开启：

1. 新增 block-level constrained sparse tangent KKT solve，目标为
   \[
   \min_{\delta x}\|U_b\delta x-\delta C_b^{\mathrm{target}}\|_{S_b}^2
   \]
   subject to local norm tangent、inactive metric tangent、inactive-active gauge tangent。
2. 新增显式 \(J_{\mathrm{sparse}}\)、\(J_{\mathrm{sparse}}^{\mathsf T}\)、\(J_{\mathrm{sparse}}^{\mathsf T}J_{\mathrm{sparse}}\) 诊断。
3. 新增 `XMVB_CPP_LOG_NONREDUNDANT_TANGENT=1` 轻量诊断。
4. 新增 `XMVB_CPP_LOG_NONREDUNDANT_TANGENT_METRIC=1` 显式 metric 诊断。
5. 新增 `XMVB_CPP_NONREDUNDANT_USE_CONSTRAINED_SPARSE_TANGENT=1`，打开后 sparse mixed block 的 expansion、projection、metric factorization 使用同一个 cached \(J_{\mathrm{sparse}}\)。
6. 在该开关下，`expand_retract_input_tangent()` 和 `retract_step()` 已改为一阶一致：finite retraction 使用 \(x+J_{\mathrm{sparse}}p\) 再做局部归一化，因此 retraction derivative 与 \(J_{\mathrm{sparse}}\) 一致。

MnF2 单核短 benchmark：

1. 默认路径，无诊断，`--max-iterations 2`：
   `SCF iteration wall time = 16.931646 s`，
   `Final total energy = -1348.828137573960`，
   `Final projected |g|_inf = 3.16262090e+00`。
2. 打开 `XMVB_CPP_NONREDUNDANT_USE_CONSTRAINED_SPARSE_TANGENT=1`，无诊断，`--max-iterations 2`：
   `SCF iteration wall time = 12.596179 s`，
   `Final total energy = -1348.827306440504`，
   `Final projected |g|_inf = 1.73063450e+00`。
3. 打开 constrained tangent 和 tangent/metric 诊断时：
   `local_norm_tangent = 0`，
   `inactive_metric_tangent = 0`，
   `inactive_active_gauge_tangent = 0`，
   `raw_adj_abs = 0`，
   `retract_adj_abs = 0`，
   `constrained_adj_abs = 0`。

额外 smoke test：

1. `TiCl.xmi` 单核 2 步：
   默认 `SCF wall = 3.510605 s`，
   constrained `SCF wall = 2.902319 s`；
   final energy 和 projected gradient 相同。
2. `FeCl.xmi` 单核 2 步：
   默认 `SCF wall = 3.312565 s`，
   constrained `SCF wall = 3.125301 s`；
   final energy 和 projected gradient 相同。
3. `241_VBSCF.xmi` 单核完整收敛：
   默认 9 步收敛，
   `SCF wall = 64.006615 s`，
   `Final energy = -230.720590176029`，
   `Final projected |g|_inf = 1.27165154e-04`。
4. `241_VBSCF.xmi` 单核完整收敛，constrained：
   63 步收敛，
   `SCF wall = 402.832111 s`，
   `Final energy = -230.720589509029`，
   `Final projected |g|_inf = 1.56419948e-04`。

当前判断：

1. 公式层面的主要问题确实是 sparse support tangent、gradient pullback 和 finite retraction 没有使用同一个几何对象。
2. MnF2、TiCl、FeCl 上 constrained sparse tangent 的短步行为更好或不差。
3. `241_VBSCF` 上 constrained path 最终能收敛到同一能量，但迭代数从 9 增加到 63、SCF wall-time 从 64 s 增加到 403 s，因此不能默认开启。
4. 下一步不应直接移除 env gate，而应分析 241 的步长/metric 策略：可能需要把 constrained tangent 只用于 sparse open-shell blocks，或先完成 \(J^{\mathsf T}J\) trust-region norm，而不是只替换 expansion/projection/retraction。

## 16. 普适 full nonorthogonal sparse tangent 推导

上一节的结果说明，`constrained sparse tangent` 不是最终答案。它修正了同一个 \(J\) 下 tangent、gradient、retraction 不一致的问题，但它同时强制固定了原始非正交轨道的部分 overlap/shape 自由度。因此更普适的方法应直接优化完整非正交 VB 轨道 tangent，而不是在旧逻辑和 constrained 逻辑之间做体系特判。

### 16.1 基本分解

对一个 block，记物理 occupied VB 轨道为

\[
C=\begin{bmatrix}C_i&C_a\end{bmatrix}.
\]

使用辅助 \(S\)-正交 frame

\[
Q=\begin{bmatrix}Q_i&Q_a&Q_v\end{bmatrix},
\qquad
Q^{\mathsf T}SQ=I.
\]

原始非正交 VB 轨道写成

\[
C_i=Q_iR,
\]

\[
C_a=Q_iG+Q_aL.
\]

因此 occupied overlap 为

\[
S_{ii}=C_i^{\mathsf T}SC_i=R^{\mathsf T}R,
\]

\[
S_{ia}=C_i^{\mathsf T}SC_a=R^{\mathsf T}G,
\]

\[
S_{aa}=C_a^{\mathsf T}SC_a=G^{\mathsf T}G+L^{\mathsf T}L.
\]

当前 constrained 方法等价于只允许 \(Q\) 和 \(L\) 的部分变化，同时令

\[
\delta S_{ii}=0,\qquad \delta S_{ia}=0.
\]

这就是它在 MnF2 上稳定、但在 `241_VBSCF` 上可能删掉有效下降方向的根本原因。

### 16.2 完整一阶变量

设 orthogonal frame 的外部/内部分区旋转为

\[
A=K_{ai},\qquad X=K_{vi},\qquad Y=K_{va}.
\]

采用 gauge-fixed frame tangent：

\[
\delta Q_i=Q_aA+Q_vX,
\]

\[
\delta Q_a=-Q_iA^{\mathsf T}+Q_vY.
\]

完整非正交 shape tangent 还应包含

\[
\Delta R,\qquad \Delta G,\qquad \Delta L.
\]

于是 dense physical tangent 为

\[
\delta C_i
=
Q_i\Delta R
+Q_aAR
+Q_vXR,
\]

\[
\delta C_a
=
Q_i(\Delta G-A^{\mathsf T}L)
+Q_a(\Delta L+AG)
+Q_v(XG+YL).
\]

因此统一变量可以写成

\[
\zeta=(A,X,Y,\Delta R,\Delta G,\Delta L).
\]

当前 constrained 方法只是特殊情况：

\[
\Delta R=0,\qquad \Delta G=0.
\]

旧逐列 retraction 的问题则是：它确实会产生等效的 \(\Delta R,\Delta G\)，但这些量不是优化变量，而是 sparse 截断和归一化的副作用，所以 gradient、HVP、trust model 无法解释它。

### 16.3 用 overlap 变量去掉 gauge 歧义

直接优化 \(\Delta R,\Delta G\) 会有 gauge 冗余。更清晰的变量是使用物理 overlap variation。

定义

\[
M=\delta S_{ii},
\qquad
N=\delta S_{ia}.
\]

其中

\[
M=\Delta R^{\mathsf T}R+R^{\mathsf T}\Delta R,
\]

\[
N=\Delta R^{\mathsf T}G+R^{\mathsf T}\Delta G.
\]

若每个 VB 轨道保持单位范数，则 \(M\) 的对角元应为零；active-active overlap 的对角约束也应由 finite retraction 保持。

一个 canonical gauge choice 是

\[
\Delta R=\frac12 R^{-\mathsf T}M,
\]

因为此时

\[
R^{\mathsf T}\Delta R+\Delta R^{\mathsf T}R=M.
\]

然后

\[
\Delta G=R^{-\mathsf T}(N-\Delta R^{\mathsf T}G).
\]

这样完整变量可写成

\[
\zeta=(A,X,Y,M,N,\Delta L),
\]

其中 constrained 方法就是

\[
M=0,\qquad N=0.
\]

这比直接保留两套逻辑更清晰：是否允许 inactive metric / inactive-active overlap relaxation 由变量空间决定，而不是由 retraction 副作用决定。

### 16.4 Sparse support 上的统一 tangent

设 dense tangent map 为

\[
D_b\zeta=\delta C_b^{\mathrm{target}}(\zeta).
\]

固定 sparse support 后，实际存储 slot tangent \(y_b\) 应通过同一个 constrained projection 得到：

\[
y_b
=
J_b\zeta
=
\arg\min_y
\left\|U_by-D_b\zeta\right\|_{S_b}^2.
\]

约束不是永远等于零，而是由 \(\zeta\) 给出右端：

\[
\delta(C_i^{\mathsf T}S_bC_i)=M,
\]

\[
\delta(C_i^{\mathsf T}S_bC_a)=N,
\]

以及每列单位范数 tangent：

\[
c_p^{\mathsf T}S_b\delta c_p=0.
\]

因此 KKT 系统应为

\[
\begin{bmatrix}
H_b&A_b^{\mathsf T}\\
A_b&0
\end{bmatrix}
\begin{bmatrix}
y_b\\
\lambda
\end{bmatrix}
=
\begin{bmatrix}
U_b^{\mathsf T}S_bD_b\zeta\\
B_b\zeta
\end{bmatrix}.
\]

其中

\[
H_b=U_b^{\mathsf T}S_bU_b.
\]

当前 constrained 实现相当于保留 \(A_b\)，但把 \(B_b\zeta\) 固定为零，并且变量里没有 \(M,N\)。普适实现应保留同一个 KKT 结构，但让 \(B_b\zeta\) 随 \(M,N\) 改变。

### 16.5 统一 gradient、HVP 和 trust norm

一旦定义

\[
y=J_{\mathrm{full}}\zeta,
\]

所有优化量都必须来自同一个 \(J_{\mathrm{full}}\)：

\[
g_\zeta=J_{\mathrm{full}}^{\mathsf T}g_x,
\]

\[
G_\zeta=J_{\mathrm{full}}^{\mathsf T}J_{\mathrm{full}},
\]

\[
H_\zeta p\approx J_{\mathrm{full}}^{\mathsf T}H_xJ_{\mathrm{full}}p.
\]

trust-region 应使用

\[
p^{\mathsf T}G_\zeta p\le \Delta^2,
\]

而不是 reduced Euclidean norm。否则即使 \(J\) 正确，TN solver 仍可能在不同变量方向上给出错误步长尺度。

### 16.6 有限步 retraction

finite retraction 也应使用同一个变量 \(\zeta\)。目标点应解

\[
X_b^+
=
\arg\min_X
\left\|U_bX-C_b^{\mathrm{target}}(\zeta)\right\|_{S_b}^2,
\]

subject to

\[
(c_p^+)^{\mathsf T}S_bc_p^+=1,
\]

\[
(C_i^+)^{\mathsf T}S_bC_i^+
=
S_{ii}^+,
\]

\[
(C_i^+)^{\mathsf T}S_bC_a^+
=
S_{ia}^+.
\]

其中一阶近似可以取

\[
S_{ii}^+=S_{ii}+M,
\qquad
S_{ia}^+=S_{ia}+N.
\]

如果 \(S_{ii}^+\) 不可行、局部 support 不足，或 Newton/KKT correction 不收敛，应让 optimizer 缩小 trust radius，而不是退回逐列归一化。

### 16.7 理论结论

普适方法不是：

1. 对 MnF2 用 constrained tangent；
2. 对 241 用旧 tangent。

普适方法应是：

\[
\zeta=(A,X,Y,M,N,\Delta L),
\qquad
y=J_{\mathrm{full}}\zeta.
\]

然后由 optimizer 自己决定 \(M,N\) 是否应该接近零。若某个体系像 MnF2 一样不需要 inactive-active overlap drift，优化会给出小的 \(N\)。若某个体系像 `241_VBSCF` 一样依赖 overlap relaxation，优化可以使用 \(M,N\) 作为受控变量，而不是依赖 retraction 副作用。

这样才能避免两套逻辑，并且数学解释是统一的。

## 17. full nonorthogonal sparse tangent 代码落实步骤

本节给出从当前代码迁移到 \(J_{\mathrm{full}}\) 的具体步骤。目标不是长期维护“旧 tangent / constrained tangent / full tangent”三套逻辑，而是用 env-gated 阶段化实现保证每一步都可 benchmark，最后收敛到一套统一数学路径。

### 17.1 当前代码可复用部分

当前 `NonredundantOrbitalSpace::initialize_block_mixed_chart_cache()` 已经构造了 full tangent 需要的主要分解量：

1. `block_basis.inactive_working_orbitals` 对应 \(Q_i\)。
2. `block_basis.active_working_orbitals` 对应 \(Q_a\)。
3. `block_basis.internal_virtual_orbitals` 对应 \(Q_v\)。
4. `block_basis.inactive_right_transform` 对应 \(R\)，满足 \(C_i=Q_iR\)。
5. `block_basis.active_inactive_gauge_coefficients` 对应 \(G\)，满足 \(C_a=Q_iG+Q_aL\)。
6. `block_basis.active_shape_matrix` 对应 \(L\)。

当前 constrained tangent 已经提供了可复用骨架：

1. `build_sparse_mixed_chart_block_step(...)` 可扩展为 full dense target step。
2. `solve_sparse_mixed_constrained_local_tangent(...)` 可扩展为带非零约束右端的 KKT solve。
3. `build_sparse_mixed_tangent_matrix(...)` 可扩展为 explicit \(J_{\mathrm{full}}\) matrix builder。
4. `apply_sparse_mixed_tangent(...)` / transpose / metric 可迁移到 full \(J\)。
5. `diagnose_sparse_tangent_geometry(...)` 可扩展为检查 \(M,N\) 约束右端和 adjoint。

### 17.2 新增 full direction layout

不要直接把现有 `BlockDirectionLayout` 强行塞满。建议新增一个 layout helper：

```cpp
struct FullNonorthogonalDirectionLayout {
  int n_inactive = 0;
  int n_active = 0;
  int n_virtual = 0;
  int inactive_active_offset = 0;  // A = K_ai
  int inactive_virtual_offset = 0; // X = K_vi
  int active_virtual_offset = 0;   // Y = K_va
  int inactive_metric_offset = 0;  // M = delta S_ii, symmetric zero diagonal
  int inactive_active_overlap_offset = 0; // N = delta S_ia
  int active_shape_offset = 0;     // Delta L
  int direction_count = 0;
};
```

变量顺序建议固定为

\[
\zeta=(A,X,Y,M,N,\Delta L).
\]

其中

1. \(A\in\mathbb R^{n_a\times n_i}\)。
2. \(X\in\mathbb R^{n_v\times n_i}\)。
3. \(Y\in\mathbb R^{n_v\times n_a}\)。
4. \(M\in\mathbb R^{n_i\times n_i}\)，对称，且对角为零。
5. \(N\in\mathbb R^{n_i\times n_a}\)。
6. \(\Delta L\in\mathbb R^{n_a\times n_a}\)。

`M` 的 packed 形式只存 off-diagonal：

\[
\#M=n_i(n_i-1)/2.
\]

原因是每个 orbital 的单位范数仍由 local norm constraint 保持，因此 inactive-inactive overlap 的对角变化不应作为自由变量。

### 17.3 新增 full dense target step

新增 helper：

```cpp
Eigen::MatrixXd build_full_nonorthogonal_chart_block_step(
    const BlockBasis& block_basis,
    const Eigen::VectorXd& full_coefficients) const;
```

该函数负责把 \(\zeta\) 变成 dense target tangent：

\[
\delta C_i
=
Q_i\Delta R
+Q_aAR
+Q_vXR,
\]

\[
\delta C_a
=
Q_i(\Delta G-A^{\mathsf T}L)
+Q_a(\Delta L+AG)
+Q_v(XG+YL).
\]

其中 \(\Delta R,\Delta G\) 由 \(M,N\) canonical gauge 得到：

\[
\Delta R=\frac12R^{-\mathsf T}M,
\]

\[
\Delta G=R^{-\mathsf T}(N-\Delta R^{\mathsf T}G).
\]

实现上不要显式求逆，使用 Eigen solve：

```cpp
delta_R = 0.5 * R.transpose().triangular_or_ldlt_solve(M);
delta_G = R.transpose().solve(N - delta_R.transpose() * G);
```

`R` 当前是 `inactive_right_transform`，通常是 SPD square root，可以用 `LDLT` 或 `SelfAdjointEigenSolver` 缓存 inverse action。第一版可以直接 `completeOrthogonalDecomposition().solve(...)` 做诊断，性能稳定后再缓存分解。

### 17.4 KKT 右端从零约束改成 \(B\zeta\)

当前 constrained KKT 是

\[
\begin{bmatrix}
H&A^{\mathsf T}\\
A&0
\end{bmatrix}
\begin{bmatrix}
y\\
\lambda
\end{bmatrix}
=
\begin{bmatrix}
U^{\mathsf T}S\delta C^{target}\\
0
\end{bmatrix}.
\]

full tangent 应改成

\[
\begin{bmatrix}
H&A^{\mathsf T}\\
A&0
\end{bmatrix}
\begin{bmatrix}
y\\
\lambda
\end{bmatrix}
=
\begin{bmatrix}
U^{\mathsf T}S\delta C^{target}\\
B\zeta
\end{bmatrix}.
\]

约束右端规则：

1. 每列 local norm tangent：右端为 0。
2. inactive-inactive overlap tangent：右端为 \(M_{pq}\)，其中 diagonal 固定为 0。
3. inactive-active overlap tangent：右端为 \(N_{pa}\)。

建议新增：

```cpp
Eigen::MatrixXd build_full_sparse_tangent_matrix(
    const BlockBasis& block_basis) const;
```

该函数一次构造所有 basis direction 的 KKT RHS，多右端 solve，返回 explicit \(J_{\mathrm{full}}\)。这比逐方向反复分解 KKT 快，也方便 adjoint check。

### 17.5 full \(J\) 的 apply / transpose / metric

新增一组与当前 constrained 函数平行的 helper：

```cpp
Eigen::VectorXd apply_full_sparse_tangent(
    const BlockBasis& block_basis,
    const Eigen::VectorXd& full_coefficients) const;

Eigen::VectorXd apply_full_sparse_tangent_transpose(
    const BlockBasis& block_basis,
    const Eigen::VectorXd& local_slots) const;

Eigen::VectorXd apply_full_sparse_tangent_metric(
    const BlockBasis& block_basis,
    const Eigen::VectorXd& full_coefficients) const;
```

验收条件：

\[
g^{\mathsf T}Jp=(J^{\mathsf T}g)^{\mathsf T}p
\]

应达到 double 精度合理误差。

同时诊断必须检查：

\[
\delta(C_i^{\mathsf T}SC_i)-M\approx 0,
\]

\[
\delta(C_i^{\mathsf T}SC_a)-N\approx 0.
\]

### 17.6 reduced chart 和 whitening

full tangent 增加 \(M,N\) 后，不应直接用 raw Euclidean norm。第一版应仍沿用当前 small-block exact factorization 思路：

\[
G=J_{\mathrm{full}}^{\mathsf T}J_{\mathrm{full}},
\qquad
G=LL^{\mathsf T}.
\]

优化器内部 reduced coordinate 使用 whitened variable：

\[
z=L^{\mathsf T}\zeta.
\]

这样 TN solver 仍可以暂时使用 Euclidean norm，但这个 Euclidean norm 已经等价于 physical tangent norm：

\[
\|z\|^2=\zeta^{\mathsf T}G\zeta.
\]

涉及函数：

1. `maybe_factorize_small_block_candidate_metric(...)`
2. `block_candidate_coefficients_from_reduced_step(...)`
3. `apply_block_candidate_metric(...)`
4. `solve_block_candidate_metric(...)`
5. `apply_inverse_reduced_block_preconditioner(...)`

注意：如果某些 large sparse block 不能 exact factorize，不能简单 fallback 到旧 diagonal。至少要记录 metric diagonal 和 condition diagnostics，否则 \(M,N\) 的尺度可能压倒 rotation variables。

### 17.7 projection / expansion 接入

第一阶段 env-gated：

```text
XMVB_CPP_NONREDUNDANT_USE_FULL_NONORTHOGONAL_TANGENT=1
```

打开后：

1. `project_block_candidate_overlap()` 对 sparse block 调用 \(J_{\mathrm{full}}^{\mathsf T}\)。
2. `accumulate_block_candidate_combination()` 对 sparse block 调用 \(J_{\mathrm{full}}\)。
3. `expand_step()` 自动得到 packed sparse tangent。
4. `project_reduced_gradient()` 自动得到 full reduced gradient。
5. `project_vector()` 使用同一个 \(J_{\mathrm{full}}\) metric solve。

不要在 hot path 中根据分子名、open-shell、活性轨道数做特判。env gate 只用于阶段验证，不作为长期策略。

### 17.8 一阶一致 retraction

先实现一阶一致版本：

\[
x^+=\mathrm{normalize}(x+J_{\mathrm{full}}p).
\]

这一步的目标不是 finite overlap 完全精确，而是保证

\[
\frac{R(\epsilon p)-x}{\epsilon}\to J_{\mathrm{full}}p.
\]

涉及函数：

1. `expand_retract_input_tangent(...)`
2. `retract_step(...)`
3. `diagnose_retraction_geometry(...)`

验收条件：

1. `raw_adj_abs` 接近 0。
2. `retract_adj_abs` 接近 0。
3. finite-difference tangent check 随 \(\epsilon\) 一阶收敛。

### 17.9 finite constrained retraction

第二阶段再做 finite KKT/Newton correction。目标是满足：

\[
(C_i^+)^{\mathsf T}SC_i^+=S_{ii}+M+O(\|p\|^2),
\]

\[
(C_i^+)^{\mathsf T}SC_a^+=S_{ia}+N+O(\|p\|^2).
\]

实现建议：

1. 初值取一阶一致点 \(x+J_{\mathrm{full}}p\)。
2. 每个 block 做少量 Newton/KKT correction。
3. 如果 correction 不收敛，返回 failure。
4. optimizer 收到 failure 后缩小 trust radius，不退回逐列归一化。

这一步完成前，不应把 full tangent 默认开启。

### 17.10 HVP 路径统一

exact_ctx direct-action HVP 中所有 pushforward/pullback 必须使用同一个 \(J_{\mathrm{full}}\)：

1. `expand_step(reduced_direction)`：用于 packed tangent pushforward。
2. `expand_retract_input_tangent(...)`：用于 stored-orbital derivative。
3. `project_reduced_gradient(packed_response)`：用于 pullback。
4. finite-difference perturbed point：由 full retraction 生成。

涉及文件：

1. `src/vb/scf/exact_orbital_second_order_operator.cpp`
2. `src/vb/scf/cpp_vb_scf_optimizer.cpp`
3. `src/vb/orbital/nonredundant_orbital_space.cpp`

验收条件是 HVP secant consistency：

\[
H_\zeta p\approx
\frac{g_\zeta(R(\epsilon p))-g_\zeta(x)}{\epsilon}.
\]

### 17.11 测试顺序

不要直接跑完整 SCF 作为第一测试。推荐顺序：

1. 单 block algebra diagnostic：
   `J` shape、finite values、constraint RHS residual。
2. Adjoint diagnostic：
   \(g^{\mathsf T}Jp=(J^{\mathsf T}g)^{\mathsf T}p\)。
3. Tangent constraint diagnostic：
   \(\delta S_{ii}-M\)、\(\delta S_{ia}-N\)。
4. Retraction derivative diagnostic：
   \((R(\epsilon p)-x)/\epsilon-Jp\)。
5. HVP secant diagnostic。
6. 单核短 SCF：
   `MnF2`、`TiCl`、`FeCl`、`241_VBSCF`。
7. 单核完整收敛：
   同上。
8. 32 核 sbatch wall-time：
   至少 `241_VBSCF` 和 `MnF2`。

### 17.12 推荐提交拆分

建议拆成以下提交，避免一次性把变量、projection、retraction、HVP 全部改掉：

1. 提交 1：新增 full layout 和 matrix unpack/pack helper，不接 hot path。
2. 提交 2：新增 `build_full_nonorthogonal_chart_block_step()`，只做 dense formula diagnostic。
3. 提交 3：新增 full KKT \(J_{\mathrm{full}}\) matrix builder，检查约束 RHS。
4. 提交 4：新增 full adjoint/metric diagnostic。
5. 提交 5：env-gated 接入 projection / expansion。
6. 提交 6：env-gated 一阶一致 retraction。
7. 提交 7：trust norm whitening / metric preconditioner。
8. 提交 8：finite constrained retraction failure-to-shrink。
9. 提交 9：HVP secant consistency 和 exact_ctx 路径清理。
10. 提交 10：多体系完整 benchmark 后，决定是否替换默认路径。

完成提交 10 前，`XMVB_CPP_NONREDUNDANT_USE_FULL_NONORTHOGONAL_TANGENT` 只能作为实验开关，不能作为默认路径。

## 18. full nonorthogonal sparse tangent 当前落地记录

当前代码已经实现到第 6 步的一阶一致实验路径，但默认路径仍不启用。

新增开关：

```text
XMVB_CPP_NONREDUNDANT_USE_FULL_NONORTHOGONAL_TANGENT=1
XMVB_CPP_LOG_NONREDUNDANT_FULL_TANGENT=1
```

已接入内容：

1. sparse block 的 full direction layout：
   \[
   \zeta=(A,X,Y,M,N,\Delta L).
   \]
2. `build_full_nonorthogonal_chart_block_step()` 实现 dense target tangent：
   \[
   \delta C_i=Q_i\Delta R+Q_aAR+Q_vXR,
   \]
   \[
   \delta C_a=Q_i(\Delta G-A^{\mathsf T}L)+Q_a(\Delta L+AG)+Q_v(XG+YL).
   \]
3. `build_full_sparse_tangent_matrix()` 通过带非零约束右端的 KKT 构造 \(J_{\mathrm{full}}\)。
4. full sparse tangent 的 apply / transpose / metric helper 已接入。
5. 打开 full env 后，sparse block 的 `project_block_candidate_overlap()`、`accumulate_block_candidate_combination()`、`expand_step()`、`project_reduced_gradient()` 使用同一个 \(J_{\mathrm{full}}\)。
6. 打开 full env 后，`retract_step()` 使用一阶一致的
   \[
   x^+=\operatorname{normalize}(x+J_{\mathrm{full}}p)
   \]
   形式；finite \(M,N\) constrained Newton correction 尚未实现。
7. full 变量的曲率对角先由旧 mixed 变量映射；\(M,N\) 使用同 block baseline curvature。小 block 仍会走 exact `D^T D` whitening。

已验证的短测试：

1. 默认路径 MnF2 单核 1 步：行为保持原结果，full diagnostic 的 residual/adjoint 为 0 到日志精度。
2. full env MnF2 单核 1 步：完整通过 TN/HVP/retraction，`full_local_norm_tangent`、`full_inactive_metric_residual`、`full_inactive_active_overlap_residual`、`full_adj_abs` 均为 0 到日志精度。
3. full env `241_VBSCF` 单核 1 步：完整通过，full residual/adjoint 为 0 到日志精度。

尚未完成：

1. finite \(M,N\) constrained retraction。
2. HVP secant consistency 的系统测试。
3. `MnF2`、`FeCl`、`TiCl`、`241_VBSCF` 的单核完整收敛 benchmark。
4. 32 核 sbatch wall-time benchmark。
