# RI low-rank update for unique-spin-string pairs

## 1. Scope

本文推导在 RI active-space two-electron 表示下，如何利用相邻
unique-spin-string pair 之间的低秩变化，加速 regular determinant-pair
矩阵元计算。

目标不是只加速 overlap / inverse，而是把双电子项也写成可以随
unique-spin-string-pair 低秩更新的形式。核心结论是：

$$
\text{RI 把四指标积分拆成辅助指标上的二指标矩阵，}
$$

所以 same-spin 与 opposite-spin 的双电子矩阵元都可以通过更新
RI occupied-block feature 来避免重新扫完整四重电子循环。

本文只讨论 regular pair：

$$
\det S \neq 0.
$$

singular pair 的 deleted-minor / SVD 路径不适合直接套 Woodbury 公式，
可以后续单独处理。

## 2. Notation

设 active 轨道数为 \(n\)，某个 spin string 的电子数为 \(m\)。一个 ordered
unique-spin-string pair 记为

$$
P=(L,R),
$$

其中 \(L\) 是 left string，\(R\) 是 right string。用选择矩阵表示 occupied
orbital：

$$
L,R \in \mathbb{R}^{n\times m}.
$$

active-orbital overlap matrix 为 \(\Omega\)。occupied overlap block 为

$$
S_P = R^T \Omega L \in \mathbb{R}^{m\times m}.
$$

regular pair 下定义

$$
s_P = \det S_P,\qquad X_P=S_P^{-1}.
$$

active-space RI 因子使用代码里的 packed active-pair 约定：

$$
(pq|rs) \approx \sum_{\Lambda=1}^{N_{\mathrm{aux}}}
B_{\Lambda,pq} B_{\Lambda,rs}.
$$

等价地，对每个辅助指标 \(\Lambda\)，定义一个对称 active-pair 矩阵

$$
B_\Lambda \in \mathbb{R}^{n\times n},
$$

满足

$$
(pq|rs) \approx \sum_{\Lambda}
(B_\Lambda)_{pq}(B_\Lambda)_{rs}.
$$

对 ordered pair \(P=(L,R)\)，定义 RI occupied block：

$$
A_\Lambda(P)=R^T B_\Lambda L \in \mathbb{R}^{m\times m}.
$$

这里 \(A_\Lambda(P)\) 的行对应 right occupied orbital，列对应 left
occupied orbital，和当前代码中 `h1e_act(occ_R, occ_L)` 的方向一致。

## 3. Regular pair matrix elements in RI form

### 3.1 One-electron term

令

$$
H(P)=R^T h L.
$$

regular same-spin one-electron \(\phi\) 为

$$
\phi^{1e}_P
=\operatorname{Tr}\!\left[H(P)X_P\right].
$$

完整矩阵元为

$$
H^{1e}_P=s_P\phi^{1e}_P.
$$

### 3.2 Same-spin two-electron term

定义

$$
M_\Lambda(P)=A_\Lambda(P)X_P.
$$

用 occupied index 显式写，令 \(a,b\) 表示 right occupied 行，\(i,j\) 表示
left occupied 列：

$$
(A_\Lambda)_{ai}=(B_\Lambda)_{R_aL_i},
\qquad
(X_P)_{ia}=(S_P^{-1})_{ia}.
$$

regular same-spin 的 RI 双电子 \(\phi\) 为

$$
\phi^{ss}_{2e}(P)
=
\frac12
\sum_{\Lambda}
\sum_{a,i,b,j}
(A_\Lambda)_{ai}(A_\Lambda)_{bj}
\left[
(X_P)_{ia}(X_P)_{jb}
-
(X_P)_{ja}(X_P)_{ib}
\right].
$$

第一项是 direct-like contraction：

$$
\sum_{a,i,b,j}
(A_\Lambda)_{ai}(X_P)_{ia}
(A_\Lambda)_{bj}(X_P)_{jb}
=
\left[\operatorname{Tr}(A_\Lambda X_P)\right]^2.
$$

第二项是 exchange-like contraction：

$$
\sum_{a,i,b,j}
(A_\Lambda)_{ai}(A_\Lambda)_{bj}
(X_P)_{ja}(X_P)_{ib}
=
\operatorname{Tr}(A_\Lambda X_P A_\Lambda X_P).
$$

则 regular same-spin 双电子 \(\phi\) 可以写成

$$
\phi^{ss}_{2e}(P)
=
\frac{1}{2}
\sum_{\Lambda=1}^{N_{\mathrm{aux}}}
\left\{
\left[\operatorname{Tr}M_\Lambda(P)\right]^2
-
\operatorname{Tr}\!\left[M_\Lambda(P)^2\right]
\right\}.
$$

完整 same-spin 矩阵元为

$$
H^{ss}_{2e}(P)
=s_P\phi^{ss}_{2e}(P).
$$

这就是 RI 下双电子项可以被低秩更新的原因：双电子项不再显式依赖
\((pq|rs)\) 四指标张量，而是依赖每个 \(\Lambda\) 上的
\(A_\Lambda X\)。

### 3.3 Opposite-spin two-electron term

对一个 alpha pair \(P_\alpha\) 和 beta pair \(P_\beta\)，定义

$$
x_\Lambda(P)=\operatorname{Tr}\!\left[A_\Lambda(P)X_P\right]
=\operatorname{Tr}M_\Lambda(P).
$$

opposite-spin 没有 exchange。把 alpha occupied index 记为 \(a,i\)，beta
occupied index 记为 \(b,j\)，则

$$
\phi^{\alpha\beta}_{2e}(P_\alpha,P_\beta)
=
\sum_\Lambda
\sum_{a,i,b,j}
(A_\Lambda^\alpha)_{ai}(X_\alpha)_{ia}
(A_\Lambda^\beta)_{bj}(X_\beta)_{jb}.
$$

因此它直接分解成两个单自旋 RI feature 的内积：

$$
\phi^{\alpha\beta}_{2e}(P_\alpha,P_\beta)
=
\sum_\Lambda
\operatorname{Tr}(A_\Lambda^\alpha X_\alpha)
\operatorname{Tr}(A_\Lambda^\beta X_\beta).
$$

regular opposite-spin \(\phi\) 为

$$
\phi^{\alpha\beta}_{2e}(P_\alpha,P_\beta)
=
\sum_{\Lambda=1}^{N_{\mathrm{aux}}}
x_\Lambda(P_\alpha)x_\Lambda(P_\beta).
$$

完整 opposite-spin 矩阵元为

$$
H^{\alpha\beta}_{2e}
=
s_{P_\alpha}s_{P_\beta}
\sum_{\Lambda}
x_\Lambda(P_\alpha)x_\Lambda(P_\beta).
$$

如果使用 first-cofactor projection，则可以定义

$$
y_\Lambda(P)=s_Px_\Lambda(P),
$$

于是

$$
H^{\alpha\beta}_{2e}
=
\sum_{\Lambda}
y_\Lambda(P_\alpha)y_\Lambda(P_\beta).
$$

这正对应当前 opposite-spin cache 里“先投影单自旋 pair，再跨自旋收缩”的数学结构。

## 4. Low-rank update between neighboring spin-string pairs

设从 pair \(P=(L,R)\) 移动到邻居

$$
P'=(L',R'),
$$

其中

$$
L'=L+\Delta L,\qquad R'=R+\Delta R.
$$

若只替换一个 occupied orbital，则 \(\Delta L\) 或 \(\Delta R\) 只有一个非零列。
一般地，若一次变化 \(k_L\) 个 left orbital 和 \(k_R\) 个 right orbital，则

$$
\operatorname{rank}\Delta L\le k_L,\qquad
\operatorname{rank}\Delta R\le k_R.
$$

### 4.1 Overlap inverse update

occupied overlap block 变化为

$$
\Delta S
=
R^T\Omega\Delta L
+\Delta R^T\Omega L
+\Delta R^T\Omega\Delta L.
$$

把它写成低秩形式：

$$
S_{P'}=S_P+U_SV_S^T,
$$

其中

$$
U_S,V_S\in\mathbb{R}^{m\times k},
\qquad
k=O(k_L+k_R).
$$

Woodbury 公式给出

$$
K=I_k+V_S^TX_PU_S,
$$

$$
X_{P'}
=
X_P-X_PU_SK^{-1}V_S^TX_P.
$$

因此

$$
\Delta X=X_{P'}-X_P=U_XV_X^T,
$$

可以取

$$
U_X=-X_PU_SK^{-1},
\qquad
V_X^T=V_S^TX_P.
$$

行列式同步更新为

$$
s_{P'}=s_P\det K.
$$

所以 regular pair 的 determinant / inverse 从重新分解的

$$
O(m^3)
$$

变为

$$
O(m^2k+k^3).
$$

### 4.2 RI occupied block update

RI occupied block 变化为

$$
\Delta A_\Lambda
=
R^TB_\Lambda\Delta L
+\Delta R^TB_\Lambda L
+\Delta R^TB_\Lambda\Delta L.
$$

因此

$$
A_\Lambda(P')=A_\Lambda(P)+\Delta A_\Lambda.
$$

如果只替换一个 left orbital，则 \(\Delta A_\Lambda\) 只改变少数列；
如果只替换一个 right orbital，则 \(\Delta A_\Lambda\) 只改变少数行。对每个
\(\Lambda\)，构造 \(\Delta A_\Lambda\) 只需要访问 \(B_\Lambda\) 在
old/new occupied support 上的小块，而不是完整 \(n\times n\) 矩阵。

## 5. Updating RI features

### 5.1 Update of \(M_\Lambda=A_\Lambda X\)

记

$$
M_\Lambda=A_\Lambda X.
$$

更新后

$$
M'_\Lambda
=
(A_\Lambda+\Delta A_\Lambda)(X+\Delta X).
$$

所以

$$
\Delta M_\Lambda
=
A_\Lambda\Delta X
+\Delta A_\Lambda X
+\Delta A_\Lambda\Delta X.
$$

由于 \(\Delta X\) 低秩，\(\Delta A_\Lambda\) 是少行/少列变化，所以
\(\Delta M_\Lambda\) 也可以用低秩或少行列形式表示。实现时不需要形成全局
active-pair vector，也不需要访问 \(N_{\mathrm{unique}}^2\) dense weight。

### 5.2 Opposite-spin feature update

opposite-spin 只需要

$$
x_\Lambda=\operatorname{Tr}M_\Lambda.
$$

所以

$$
x'_\Lambda
=x_\Lambda+\Delta x_\Lambda,
$$

其中

$$
\Delta x_\Lambda
=\operatorname{Tr}\Delta M_\Lambda.
$$

如果缓存的是 first-cofactor RI feature

$$
y_\Lambda=sx_\Lambda,
$$

则

$$
y'_\Lambda=s'x'_\Lambda
=s_P\det K\left(x_\Lambda+\Delta x_\Lambda\right).
$$

也可以写成增量形式：

$$
\Delta y_\Lambda
=
(s'-s)x_\Lambda+s'\Delta x_\Lambda.
$$

于是 full determinant opposite-spin coupling 的更新只需要

$$
H^{\alpha\beta}_{2e}
=
\sum_\Lambda
y_\Lambda^\alpha y_\Lambda^\beta.
$$

如果 alpha 或 beta 一侧发生 low-rank 更新，只更新该侧的 \(y_\Lambda\)，另一侧
可直接复用。

### 5.3 Same-spin two-electron feature update

same-spin 双电子项需要两个 RI scalar：

$$
a_\Lambda=\operatorname{Tr}M_\Lambda,
\qquad
b_\Lambda=\operatorname{Tr}\left(M_\Lambda^2\right).
$$

更新后

$$
a'_\Lambda
=a_\Lambda+\operatorname{Tr}\Delta M_\Lambda.
$$

对 exchange-like scalar：

$$
b'_\Lambda
=
\operatorname{Tr}\left[(M_\Lambda+\Delta M_\Lambda)^2\right].
$$

展开得

$$
b'_\Lambda
=
b_\Lambda
+2\operatorname{Tr}\left(M_\Lambda\Delta M_\Lambda\right)
+\operatorname{Tr}\left[(\Delta M_\Lambda)^2\right].
$$

因此 same-spin 双电子 \(\phi\) 更新为

$$
\phi^{ss}_{2e}(P')
=
\frac{1}{2}
\sum_\Lambda
\left[
(a'_\Lambda)^2-b'_\Lambda
\right].
$$

完整矩阵元同步乘新的 determinant：

$$
H^{ss}_{2e}(P')=s_{P'}\phi^{ss}_{2e}(P').
$$

这说明 RI + low-rank update 确实可以更新 same-spin 双电子主项，而不是只更新
overlap / inverse。

## 6. Equivalent active-density form

也可以用 transition density 写同一个公式。定义

$$
D_P=R X_P L^T.
$$

如果

$$
D_{P'}=D_P+UV^T,
\qquad U,V\in\mathbb{R}^{n\times r},
$$

则 RI projection 为

$$
x_\Lambda(D)=\operatorname{Tr}(B_\Lambda D).
$$

更新为

$$
x_\Lambda(D+UV^T)
=
x_\Lambda(D)
+\operatorname{Tr}(V^TB_\Lambda U).
$$

opposite-spin 直接使用这些 \(x_\Lambda\)。same-spin 的 RI exchange 更新为

$$
z_\Lambda(D)=\operatorname{Tr}(B_\Lambda D B_\Lambda D),
$$

$$
z_\Lambda(D+UV^T)
=
z_\Lambda(D)
+2\operatorname{Tr}(V^TB_\Lambda D B_\Lambda U)
+\operatorname{Tr}\left[(V^TB_\Lambda U)^2\right].
$$

这个形式更理论化，但实现上 occupied-block 形式更合适，因为 \(D_P\) 只在
left/right occupied support 上有效，而代码的 determinant-pair kernel 本来就按
occupied string 工作。

## 7. Cost model

设一次 pair-to-pair 变化的有效秩为 \(k\)。

当前直接 regular pair 计算大致包含：

$$
O(m^3)
$$

的 overlap 分解，以及 same-spin 双电子四重电子循环。如果 RI lookup 每次都做
auxiliary dot，则同自旋双电子代价接近

$$
O(N_{\mathrm{aux}}m^4).
$$

如果已经重构 packed exact active kernel，则四重循环是

$$
O(m^4),
$$

但代价被转移到了 active-space integral preparation 和 packed kernel memory。

RI occupied-block 直接重算的形式是：

$$
A_\Lambda=R^TB_\Lambda L,
$$

每个 pair 大约需要

$$
O(N_{\mathrm{aux}}m^2)
$$

访问 occupied RI blocks，再加上构造 \(A_\Lambda X\) 的矩阵乘法成本。

low-rank update 后，核心变化为：

$$
X\rightarrow X+\Delta X,
\qquad
A_\Lambda\rightarrow A_\Lambda+\Delta A_\Lambda,
$$

其中 \(\Delta X\) rank 为 \(k\)，\(\Delta A_\Lambda\) 只改少量行列。若维护
tile-local 的 \(M_\Lambda=A_\Lambda X\)，每个邻居更新的主成本可以降到近似

$$
O(m^2k+k^3)
+
O(N_{\mathrm{aux}}m^2k),
$$

单替换时 \(k=1\)，即接近

$$
O(m^2)+O(N_{\mathrm{aux}}m^2).
$$

这不是降低 \(N_{\mathrm{aux}}\) 的标度，而是把每个 unique-spin pair 的
四重电子循环或完整 \(A_\Lambda X\) 重算，替换为沿 pair graph 的低秩更新。

## 8. Memory policy

不能为所有 ordered unique-spin pairs 存储

$$
\{M_\Lambda(P)\}_{P,\Lambda},
$$

否则内存为

$$
O(N_{\mathrm{unique}}^2N_{\mathrm{aux}}m^2),
$$

不可接受。

合理策略是 tile-local / traversal-local：

1. 在一个 unique-spin tile 或一个 pair-graph traversal lane 内维护当前
   \(S,X,s,A_\Lambda,M_\Lambda,a_\Lambda,b_\Lambda\)。
2. 沿 Hamming-distance 小的邻接 pair 更新。
3. 当前 tile 的矩阵元或 RI feature 写入上层 contraction 后立即释放。
4. 不建立全局 \(N_{\mathrm{unique}}^2\) dense feature table。

opposite-spin 如果只需要 \(y_\Lambda=sx_\Lambda\)，每个当前 pair 的工作内存是

$$
O(N_{\mathrm{aux}}).
$$

same-spin 若维护 \(M_\Lambda\)，每个 traversal lane 的工作内存是

$$
O(N_{\mathrm{aux}}m^2).
$$

这可以作为 tile-local workspace，但不应成为全局 cache。

## 9. Implementation implications

### 9.1 What becomes faster

在 RI 框架下，low-rank update 可以加速：

$$
S,\quad S^{-1},\quad \det S,
$$

也可以加速：

$$
x_\Lambda=\operatorname{Tr}(A_\Lambda X),
\qquad
\operatorname{Tr}\left[(A_\Lambda X)^2\right].
$$

因此它不只是单电子优化，而是可以进入 same-spin 与 opposite-spin 双电子矩阵元。

### 9.2 What does not become faster automatically

如果仍然把 RI factors 先重构成 packed exact \(GGO\)，然后按
\((pq|rs)\) 直接 lookup 做四重循环，则 low-rank update 无法利用 RI 分解结构。
这种路径只会保留 determinant / inverse 的收益，双电子主项仍然是四重电子循环。

所以该优化要求矩阵元 kernel 原生消费

$$
B_{\Lambda,pq},
$$

而不是消费已经重构好的 packed pair-of-pairs integral。

### 9.3 Traversal requirement

收益依赖 pair graph 的局部性。若从一个 arbitrary pair 跳到另一个 unrelated pair，
有效秩 \(k\) 接近 \(m\)，Woodbury 与 \(\Delta A_\Lambda\) 的优势会消失。

因此需要按 unique string 的邻接关系组织计算，例如：

$$
\text{same left string, right string single replacement}
$$

或

$$
\text{same right string, left string single replacement}.
$$

这种 traversal 对稀疏轨道体系尤其重要，因为常用 VBSCF determinant space 通常由局部
active orbital occupation 变化生成，邻接 pair 的 support 差异较小。

## 10. Recommended next step

先不要直接改 HVP 或 backward。更稳的验证顺序是：

1. 写一个 RI-only regular same-spin pair evaluator prototype，直接用
   \(A_\Lambda=R^TB_\Lambda L\) 和
   \(\phi^{ss}_{2e}=\frac12\sum_\Lambda[(\operatorname{Tr}AX)^2-\operatorname{Tr}((AX)^2)]\)
   复现当前 same-spin pair 标量。
2. 在 prototype 内加入单替换 pair 的 Woodbury 更新，验证
   \(s,X,A_\Lambda,M_\Lambda,a_\Lambda,b_\Lambda\) 更新后的矩阵元与直接重算一致。
3. 再把 opposite-spin first-cofactor projection 改成 RI feature
   \(y_\Lambda=s\operatorname{Tr}(A_\Lambda X)\) 的 tile-local contraction。
4. 最后再考虑 backward / HVP，因为它们需要同步更新 feature 的 adjoint。

这个方向的本质收益来自

$$
\text{RI feature representation}
+
\text{pair-graph locality}
+
\text{tile-local bounded memory}.
$$

缺少其中任一项，low-rank update 都容易退化成只优化 determinant / inverse 的小收益。

## 11. Minimal feasibility validation

这个思路最容易验证的对象不是 HVP，而是当前输入展开后的 unique-spin-string
集合。验证应分成三个独立层次。

### 11.1 Unique-string locality statistic

对两个同电子数 spin string

$$
I=\{i_1,\dots,i_m\},
\qquad
J=\{j_1,\dots,j_m\},
$$

定义 replacement distance：

$$
d(I,J)=m-|I\cap J|.
$$

如果固定 left string，只沿 right string 更新，则 ordered pair

$$
(L,R)\rightarrow(L,R')
$$

的 Woodbury rank 就是

$$
k=d(R,R').
$$

如果固定 right string，只沿 left string 更新，则

$$
k=d(L,L').
$$

因此第一个 diagnostic 只需要打印：

1. alpha unique string 数 \(N_\alpha\)，beta unique string 数 \(N_\beta\)；
2. alpha / beta 的 replacement-distance histogram；
3. 对每个 unique-string 集合做 greedy nearest-neighbor traversal，打印路径边权
   \(d\) 的平均值、最大值、\(d=1\) 比例、\(d\le2\) 比例。

如果 greedy path 的平均 \(d\) 接近 1，说明可以按 row/column sweep 用 rank-1
或 rank-2 update 覆盖大部分 ordered unique-pair 表：

$$
(L_i,R_{\pi_1})\rightarrow(L_i,R_{\pi_2})\rightarrow\cdots
\rightarrow(L_i,R_{\pi_N}).
$$

这一步完全不需要积分，也不需要求能量，只看 unique-spin-string 本身。

### 11.2 Direct RI formula correctness check

第二步只验证 regular pair 公式是否等价当前实现。对随机或前若干个 regular
ordered unique-spin pair \(P=(L,R)\)，直接构造

$$
S=R^T\Omega L,\qquad X=S^{-1},\qquad s=\det S,
$$

以及

$$
A_\Lambda=R^TB_\Lambda L.
$$

然后计算

$$
\phi^{ss}_{2e}
=
\frac12\sum_\Lambda
\left[
(\operatorname{Tr}A_\Lambda X)^2
-
\operatorname{Tr}(A_\Lambda X A_\Lambda X)
\right].
$$

对比当前 `evaluate_same_spin_pair(...)` 给出的 two-electron 部分：

$$
H^{ss}_{2e}
=
H^{ss}_{\mathrm{total}}-H^{ss}_{1e}.
$$

检查误差：

$$
\Delta
=
\left|
s\phi^{ss}_{2e}
-
\left(H^{ss}_{\mathrm{total}}-H^{ss}_{1e}\right)
\right|.
$$

opposite-spin 同理，先对单自旋 pair 计算

$$
y_\Lambda(P)=s_P\operatorname{Tr}(A_\Lambda(P)X_P),
$$

再对 alpha/beta pair 检查

$$
H^{\alpha\beta}_{2e}
=
\sum_\Lambda
y_\Lambda(P_\alpha)y_\Lambda(P_\beta)
$$

是否复现当前 `evaluate_opposite_spin_coulomb_coupling(...)`。

这一步验证的是公式，不涉及 low-rank update。

### 11.3 Rank update correctness check

第三步选 replacement distance 为 1 的邻接 string。若只替换 right string 的第
\(t\) 个 occupied orbital，

$$
R'\leftarrow R+\Delta R,
$$

则 overlap block 只改变一行：

$$
S'=S+e_t w^T,
$$

其中

$$
w_i=\Omega_{r_{\mathrm{new}},L_i}
-\Omega_{r_{\mathrm{old}},L_i}.
$$

Woodbury rank-1 update 为

$$
K=1+w^TXe_t,
$$

$$
X'=X-\frac{Xe_t w^T X}{K},
\qquad
s'=sK.
$$

RI occupied block 也只改变同一行：

$$
A'_\Lambda=A_\Lambda+e_t q_\Lambda^T,
$$

其中

$$
(q_\Lambda)_i
=
(B_\Lambda)_{r_{\mathrm{new}},L_i}
-
(B_\Lambda)_{r_{\mathrm{old}},L_i}.
$$

然后用更新后的

$$
X',\quad s',\quad A'_\Lambda
$$

计算 same-spin / opposite-spin RI feature，并与直接从 \(P'\) 重算的结果比较。
left string 单替换完全类似，只是 row update 变成 column update：

$$
S'=S+w e_t^T,
\qquad
A'_\Lambda=A_\Lambda+q_\Lambda e_t^T.
$$

如果 11.2 和 11.3 都通过，说明算法数学上可行；剩下的问题才是 traversal
组织和 kernel 性能。

### 11.4 Performance go/no-go criterion

这个优化不是必然比当前 packed-GGO lookup 快。若 RI 先重构成 packed active
two-electron kernel，当前 same-spin regular pair 的双电子主循环近似是

$$
O(m^4)
$$

的内存 lookup。RI low-rank update 原生消费 \(B_{\Lambda,pq}\)，same-spin
更新每个邻居大约是

$$
O(N_{\mathrm{aux}}m^2k),
$$

opposite-spin feature 更新可接近

$$
O(N_{\mathrm{aux}}mk).
$$

所以需要实际打印估算比值：

$$
\rho_{ss}
=
\frac{N_{\mathrm{aux}}m^2\bar{k}}{m^4}
=
\frac{N_{\mathrm{aux}}\bar{k}}{m^2},
$$

其中 \(\bar{k}\) 是 traversal 平均 replacement rank。若

$$
\rho_{ss}\ll 1,
$$

same-spin 很可能收益明显；若

$$
\rho_{ss}\gtrsim 1,
$$

same-spin 可能不如 packed-GGO lookup，但 opposite-spin 的 \(O(N_{\mathrm{aux}}m)\)
feature update 仍可能有收益。

因此 prototype diagnostic 应同时输出：

$$
N_{\mathrm{aux}},\quad m_\alpha,\quad m_\beta,\quad
\bar{k}_\alpha,\quad \bar{k}_\beta,\quad
\rho_{ss}^\alpha,\quad \rho_{ss}^\beta.
$$

如果 unique-string locality 很好，但 \(\rho_{ss}\) 不好，说明该方案更适合作为
large-active / no-packed-GGO / opposite-spin-feature 优化，而不应优先替换当前小
active packed-GGO same-spin 热路径。

## 12. Initial validation results

新增诊断工具：

```bash
build/src/check_ri_low_rank_unique_spin_pair <input.xmi>
```

该工具做三件事：

1. 统计 alpha / beta unique spin string 的 replacement distance 和当前排序下的
   positional update rank；
2. 用 RI trace 公式复现当前 regular same-spin / opposite-spin 矩阵元；
3. 对 replacement-distance 为 1 的邻接 string 做 Woodbury 更新，检查
   \(X\)、\(\det S\)、same-spin RI \(\phi_{2e}\) 是否等于直接重算。

### 12.1 F2 smoke check

命令：

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  build/src/check_ri_low_rank_unique_spin_pair \
  src/test_molecule/F2.xmi \
  --max-pair-checks 16 \
  --max-rank-update-checks 8
```

结果：

| item | value |
|---|---:|
| \(N_{\mathrm{aux}}\) | 210 |
| \(n_{\mathrm{act}}\) | 2 |
| alpha / beta unique strings | 2 / 2 |
| same-spin max \(|\Delta\phi_{2e}|\) | 0 |
| opposite-spin max \(|\Delta\phi|\) | \(8.9\times10^{-16}\) |
| rank-update max \(|\Delta X|\) | \(1.4\times10^{-14}\) |
| rank-update max \(|\Delta\phi_{2e}|\) | 0 |

这说明公式和 rank update 基本索引方向正确。

### 12.2 241_VBSCF

Slurm job:

```bash
sbatch -c 32 --partition=6526Y --account=weiwu --time=00:30:00 \
  --job-name=rilr241 \
  --wrap="cd /pool1/home/xiatao/project/xmvb-cpp && \
  env OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  build/src/check_ri_low_rank_unique_spin_pair \
  test/241_VBSCF.xmi \
  --max-pair-checks 128 \
  --max-rank-update-checks 64"
```

Output:

```text
test/ri_low_rank_241..out
```

The missing job id in the filename is from local shell expansion of
`$SLURM_JOB_ID` before submission; later runs should use `--output=%j`.

Key results:

| item | alpha | beta |
|---|---:|---:|
| unique strings | 20 | 20 |
| active spin electrons \(m\) | 3 | 3 |
| greedy mean replacement distance | 1.05 | 1.00 |
| greedy mean positional update rank | 1.11 | 1.00 |
| greedy max positional update rank | 2 | 1 |
| \(\rho_{ss}=N_{\mathrm{aux}}\bar{k}/m^2\) | 79.6 | 72.0 |
| same-spin max \(|\Delta\phi_{2e}|\) | \(1.4\times10^{-14}\) | \(6.6\times10^{-14}\) |
| rank-update mean update rank | 1.70 | 1.70 |
| rank-update max update rank | 3 | 3 |
| rank-update max \(|\Delta X|\) | \(3.0\times10^{-11}\) | \(7.9\times10^{-11}\) |
| rank-update max \(|\Delta\phi_{2e}|\) | \(2.4\times10^{-14}\) | \(1.0\times10^{-13}\) |

opposite-spin RI feature formula also matches:

$$
\max|\Delta\phi^{\alpha\beta}_{2e}|=6.2\times10^{-15}.
$$

Interpretation:

1. unique-string traversal locality is excellent;
2. Woodbury + RI block update is numerically correct;
3. same-spin replacement of the current packed-GGO hot path is not attractive for this
   active size because \(N_{\mathrm{aux}}/m^2\) is too large.

### 12.3 MnF2

Slurm job:

```bash
sbatch -c 32 --partition=6526Y --account=weiwu --time=00:30:00 \
  --job-name=rilrmnf2 \
  --output=test/ri_low_rank_mnf2.%j.out \
  --wrap="cd /pool1/home/xiatao/project/xmvb-cpp && \
  env OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  build/src/check_ri_low_rank_unique_spin_pair \
  test/MnF2.xmi \
  --max-pair-checks 128 \
  --max-rank-update-checks 64"
```

Output:

```text
test/ri_low_rank_mnf2.1943312.out
```

Key results:

| item | alpha | beta |
|---|---:|---:|
| unique strings | 8 | 28 |
| active spin electrons \(m\) | 7 | 2 |
| greedy mean replacement distance | 1.00 | 1.00 |
| greedy mean positional update rank | 1.00 | 1.22 |
| greedy max positional update rank | 1 | 2 |
| \(\rho_{ss}=N_{\mathrm{aux}}\bar{k}/m^2\) | 6.43 | 96.25 |
| same-spin max \(|\Delta\phi_{2e}|\) | \(6.8\times10^{-14}\) | \(1.1\times10^{-10}\) |
| same-spin max determinant-scaled \(|\Delta H_{2e}|\) | \(1.9\times10^{-14}\) | \(2.2\times10^{-16}\) |
| rank-update mean update rank | 3.08 | 1.16 |
| rank-update max update rank | 7 | 2 |
| rank-update max \(|\Delta X|\) | \(6.2\times10^{-11}\) | \(1.1\times10^{-8}\) |
| rank-update max \(|\Delta\phi_{2e}|\) | \(4.6\times10^{-13}\) | \(2.0\times10^{-12}\) |

opposite-spin RI feature formula matches:

$$
\max|\Delta\phi^{\alpha\beta}_{2e}|=2.8\times10^{-13}.
$$

Interpretation:

1. open-shell sparse case also has very good replacement-distance locality;
2. sorted occupied-string order can inflate update rank, especially alpha when one
   replaced orbital changes many sorted positions;
3. determinant-scaled errors stay at numerical-noise level;
4. same-spin packed-GGO replacement is still not an obvious win for beta because \(m=2\).

### 12.4 Current decision

The RI low-rank idea is mathematically validated:

$$
\text{RI trace formula correct}
\quad+\quad
\text{Woodbury update correct}
\quad+\quad
\text{unique-string locality good}.
$$

However, for the tested current VBSCF cases, it should not replace the packed-GGO
same-spin hot path yet. The immediate useful direction is narrower:

1. use RI low-rank features for opposite-spin feature generation, where the update is
   \(O(N_{\mathrm{aux}}mk)\) rather than same-spin \(O(N_{\mathrm{aux}}m^2k)\);
2. keep same-spin packed-GGO path for small active spaces;
3. revisit same-spin RI low-rank only for larger active spaces, disabled packed-GGO,
   or a traversal that also removes sorted-order rank inflation by carrying explicit
   permutation signs.

## 13. Opposite-spin feature-generation benchmark

The diagnostic tool now also benchmarks regular opposite-spin feature generation:

```bash
--max-feature-benchmark-pairs N
```

For each ordered unique-spin pair it compares:

1. **direct RI feature recomputation**

   $$
   y_\Lambda
   =
   s\sum_{ia}
   X_{ia}
   B_{\Lambda,R_aL_i},
   $$

   implemented as RI-factor column axpy operations;

2. **right-string traversal low-rank update**

   for fixed left string, traverse right strings in greedy nearest-neighbor order,
   update \(X\), \(\det S\), and the auxiliary-channel matrices

   $$
   M_\Lambda=A_\Lambda X.
   $$

The low-rank benchmark is mathematically consistent with direct recomputation:
checksums agree to floating-point noise. It is not yet a production algorithm.

### 13.1 Benchmark results

| input | spin | pairs | direct feature time | low-rank feature time | speedup |
|---|---:|---:|---:|---:|---:|
| F2 | alpha | 4 | \(7.4\times10^{-6}\) s | \(1.36\times10^{-4}\) s | 0.055 |
| F2 | beta | 4 | \(4.9\times10^{-6}\) s | \(1.27\times10^{-4}\) s | 0.039 |
| 241_VBSCF | alpha | 400 | \(1.29\times10^{-3}\) s | \(4.24\times10^{-2}\) s | 0.030 |
| 241_VBSCF | beta | 400 | \(1.20\times10^{-3}\) s | \(3.55\times10^{-2}\) s | 0.034 |
| MnF2 | alpha | 64 | \(5.92\times10^{-4}\) s | \(7.39\times10^{-3}\) s | 0.080 |
| MnF2 | beta | 784 | \(1.11\times10^{-3}\) s | \(3.57\times10^{-2}\) s | 0.031 |
| 10698_RI | alpha | 400 | \(1.93\times10^{-3}\) s | \(6.40\times10^{-2}\) s | 0.030 |
| 10698_RI | beta | 400 | \(1.58\times10^{-3}\) s | \(5.64\times10^{-2}\) s | 0.028 |
| C6H6.full | alpha | 400 | \(1.22\times10^{-3}\) s | \(3.79\times10^{-2}\) s | 0.032 |
| C6H6.full | beta | 400 | \(9.89\times10^{-4}\) s | \(3.39\times10^{-2}\) s | 0.029 |

The current prototype is therefore slower than direct RI feature recomputation
on all tested cases.

### 13.2 Why this prototype loses

The direct feature formula is already very cheap:

$$
y_\Lambda
=
\sum_{ia}
c_{ia}B_{\Lambda,p(ia)}.
$$

This is a short sequence of vector axpy operations over the auxiliary dimension,
with only \(m^2\) active occupied pairs.

The prototype low-rank traversal maintains the full matrix

$$
M_\Lambda=A_\Lambda X
\in\mathbb{R}^{m\times m}
$$

for every \(\Lambda\). Updating \(M_\Lambda\) after a right-string change costs
outer-product updates over the full \(m\times m\) block:

$$
O(N_{\mathrm{aux}}m^2k).
$$

For the current small/medium active spaces, this is worse than direct axpy
recomputation:

$$
O(N_{\mathrm{aux}}m^2).
$$

The tested cases have small spin electron counts:

| input | \(m_\alpha\) | \(m_\beta\) |
|---|---:|---:|
| 241_VBSCF / C6H6.full / 10698_RI | 3 | 3 |
| MnF2 | 7 | 2 |

For such sizes, matrix-state maintenance overhead dominates.

### 13.3 What would be needed for a real win

The useful theoretical update for opposite-spin is not the full-\(M_\Lambda\)
prototype. For rank-1 right-string update,

$$
S'=S+e_t w^T,
\qquad
X'=X-cd^T,
\qquad
A'_\Lambda=A_\Lambda+e_tq_\Lambda^T.
$$

The scalar feature can be updated as

$$
x'_\Lambda
=
x_\Lambda
-d^T M_{\Lambda,:,t}
+q_\Lambda^T X_{:,t}
-(q_\Lambda^Tc)d_t,
$$

where

$$
M_\Lambda=A_\Lambda X.
$$

This only needs the selected column \(M_{\Lambda,:,t}\), not the full
\(m\times m\) matrix, if the traversal repeatedly changes a known slot \(t\).
That would reduce the intended update to roughly

$$
O(N_{\mathrm{aux}}mk)
$$

instead of

$$
O(N_{\mathrm{aux}}m^2k).
$$

However, current unique strings are sorted. A one-orbital replacement can move
many positions after sorting; this is why MnF2 alpha showed positional update
rank up to 7 even though replacement distance is 1. To make the scalar-only
feature update fast, we need a traversal-local occupied ordering with explicit
permutation signs:

$$
\text{replace one slot}
\quad\Longrightarrow\quad
\text{rank-1 update in that slot}.
$$

Without that slot-stable representation, the update frequently becomes multi-row
and loses the expected advantage.

### 13.4 Slot-stable scalar prototype

The narrower scalar prototype has now been tested in
`check_ri_low_rank_unique_spin_pair`. It keeps a traversal-local right-string
ordering, updates a single occupied slot when possible, and carries the
permutation sign back to the canonical sorted determinant convention:

$$
y_\Lambda^{\mathrm{canonical}}
=
\operatorname{sgn}(R_{\mathrm{internal}}\to R_{\mathrm{sorted}})
\det(S_{\mathrm{internal}})
x_\Lambda^{\mathrm{internal}}.
$$

For a rank-1 right-slot replacement, the implemented scalar update is

$$
x'_\Lambda
=
x_\Lambda
\mu\left(q_\Lambda^TX_{:,t}-d^TM_{\Lambda,:,t}\right),
\qquad
\mu=(1+d_t)^{-1},
$$

where \(t\) is the replaced right-occupied slot. The selected column is updated
as

$$
M'_{\Lambda,:,t}
=
\mu M_{\Lambda,:,t}+\mu e_t(q_\Lambda^TX_{:,t}).
$$

This avoids storing the full \(M_\Lambda\) block for every auxiliary channel,
but still needs one selected \(M_{\Lambda,:,t}\) column and a feature vector of
length \(N_{\mathrm{aux}}\).

### 13.5 Slot-stable benchmark result

The slot-stable traversal reduces selected-column rebuilds in 20-string cases,
but it still does not beat direct RI axpy recomputation.

| input / spin | rebuilds before | rebuilds after | direct seconds | scalar seconds | direct/scalar |
|---|---:|---:|---:|---:|---:|
| 241 alpha | 300 | 200 | 0.00176 | 0.00255 | 0.69 |
| 241 beta | 260 | 180 | 0.00125 | 0.00245 | 0.51 |
| MnF2 alpha | 56 | 56 | 0.00080 | 0.00104 | 0.77 |
| MnF2 beta | 196 | 196 | 0.00153 | 0.00342 | 0.45 |
| 10698_RI alpha | 300 | 200 | 0.00367 | 0.00441 | 0.83 |
| 10698_RI beta | 260 | 180 | 0.00217 | 0.00347 | 0.62 |

The full-\(M_\Lambda\) low-rank path is much slower, with direct/full-M speedup
only around \(0.03\) to \(0.10\) on the same cases.

The scalar path matches direct RI to roundoff on most cases. MnF2 beta still
shows accumulated absolute checksum drift around \(10^{-8}\), which is another
reason not to move this prototype into production without stronger numerical
controls.

### 13.6 Updated decision

Do not implement the full-\(M_\Lambda\) or slot-stable scalar low-rank feature
paths in production for the current code. They are mathematically valid, but
the direct RI feature formula is already too cheap for the tested active spaces:

$$
y_\Lambda
=
\sum_{ia}
c_{ia}B_{\Lambda,p(ia)}.
$$

The bottleneck should stay focused on tile contraction, avoiding
\(N_{\mathrm{unique}}^2\) dense weight storage, and on the HVP/CG kernel rather
than on this RI low-rank feature-update route.
