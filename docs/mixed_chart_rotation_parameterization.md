# 正交块使用轨道旋转的混合参数化

## 1. 目标

当前 `orbtyp=oeo` 路径已经在 accepted-point cache 中显式构造了

\[
Q_i \in \mathbb{R}^{N \times n_i},
\qquad
Q_i^{\mathrm T} S Q_i = I,
\]

以及辅助活性轨道

\[
T_a \in \mathbb{R}^{N \times n_a},
\qquad
Q_i^{\mathrm T} S T_a = 0.
\]

但外层优化变量仍主要以物理 AO 系数 `C_i, C_a` 的混合形式表达，因此“非活性块已经正交”这一事实并没有被真正用于外层更新语义。这样会带来两个问题：

1. 正交块仍以一般系数混合更新，切空间包含大量几何上冗余的方向。
2. internal chart 与 external chart 的 transport / secant 语义不一致，容易拖慢收敛。

本文档给出一个更本质的 mixed-chart 参数化：  
**所有正交部分使用轨道旋转，只有真正非正交的活性内部自由度保留非正交 mixing。**

---

## 2. 变量与维度

记：

\[
N = \text{AO 基函数数},
\qquad
n_i = \text{非活性双占轨道数},
\qquad
n_a = \text{活性轨道数},
\qquad
n_v = N - n_i - n_a.
\]

在当前 OEO 语义下，能量相关的工作变量可以写成：

\[
Q_i \in \mathbb{R}^{N \times n_i},
\qquad
T_a \in \mathbb{R}^{N \times n_a},
\qquad
P_i = Q_i Q_i^{\mathrm T},
\]

\[
Q_i^{\mathrm T} S Q_i = I,
\qquad
Q_i^{\mathrm T} S T_a = 0.
\]

注意：

\[
T_a^{\mathrm T} S T_a \neq I
\]

一般并不成立，因此活性内部仍然是非正交对象。

为了把“正交部分”与“非正交部分”彻底拆开，引入活性辅助轨道的极分解式表达：

\[
A_{aa} = T_a^{\mathrm T} S T_a \in \mathbb{R}^{n_a \times n_a},
\]

\[
Q_a = T_a A_{aa}^{-1/2} \in \mathbb{R}^{N \times n_a},
\qquad
L_a = A_{aa}^{1/2} \in \mathbb{R}^{n_a \times n_a}.
\]

于是

\[
T_a = Q_a L_a,
\]

并满足

\[
Q_a^{\mathrm T} S Q_a = I,
\qquad
Q_i^{\mathrm T} S Q_a = 0.
\]

再取一个 \(S\)-正交的虚轨道补空间

\[
Q_v \in \mathbb{R}^{N \times n_v},
\]

满足

\[
Q_v^{\mathrm T} S Q_v = I,
\qquad
Q_i^{\mathrm T} S Q_v = 0,
\qquad
Q_a^{\mathrm T} S Q_v = 0.
\]

于是可以得到一个正交 frame

\[
Q = [\,Q_i,\; Q_a,\; Q_v\,] \in \mathbb{R}^{N \times N},
\qquad
Q^{\mathrm T} S Q = I.
\]

这一步是整个 mixed-chart 的关键：  
`Q_i, Q_a, Q_v` 之间的耦合都属于正交子空间之间的旋转；  
只有 \(L_a\) 负责活性内部的非正交形状。

---

## 3. 推荐的混合变量

推荐把工作变量写成

\[
(Q, L_a),
\]

其中：

1. \(Q\) 的更新使用轨道旋转；
2. \(L_a\) 的更新使用一般非正交 mixing。

更具体地，令

\[
\Omega =
\begin{bmatrix}
0              & -K_{ia}^{\mathrm T} & -K_{iv}^{\mathrm T} \\
K_{ia}         & 0                   & -K_{av}^{\mathrm T} \\
K_{iv}         & K_{av}              & 0
\end{bmatrix},
\]

其中

\[
K_{ia} \in \mathbb{R}^{n_a \times n_i},
\qquad
K_{iv} \in \mathbb{R}^{n_v \times n_i},
\qquad
K_{av} \in \mathbb{R}^{n_v \times n_a}.
\]

由于 \(\Omega^{\mathrm T} = -\Omega\)，所以 \(Q\) 的有限步更新可写成

\[
Q' = Q \exp(\Omega),
\]

或在实现上使用 Cayley 型 retraction

\[
Q' = Q
\left(I - \frac{1}{2}\Omega\right)^{-1}
\left(I + \frac{1}{2}\Omega\right).
\]

这部分严格保持

\[
{Q'}^{\mathrm T} S Q' = I.
\]

活性内部的非正交形状则保留为

\[
L_a' = L_a + \Delta L_a,
\qquad
\Delta L_a \in \mathbb{R}^{n_a \times n_a}.
\]

因此新的辅助活性轨道为

\[
T_a' = Q_a' L_a'.
\]

这个表示已经满足用户期望的语义：

1. 正交块之间全部用 rotation；
2. 只有活性内部仍使用一般 mixing。

---

## 3.1 物理轨道与工作变量的关系

mixed-chart 中真正参与能量与梯度传播的是

\[
Q_i,\qquad T_a = Q_a L_a.
\]

而用户最终看到的物理轨道可以写成

\[
C_i = Q_i U_i,
\qquad
C_a = T_a + Q_i K_a,
\]

其中

\[
U_i \in \mathbb{R}^{n_i \times n_i},
\qquad
K_a \in \mathbb{R}^{n_i \times n_a}.
\]

这里：

1. \(U_i\) 只负责非活性代表元 / 定域表示；
2. \(K_a\) 只负责活性轨道的 inactive-null representative；
3. 在固定 \(Q_i\) 与 \(T_a\) 时，\(U_i\) 与 \(K_a\) 不改变 \(P_i\) 和 \(T_a\)，因此不应进入主优化变量。

这也是为什么 active representative reset 更适合被视为 gauge / export 语义，而不是优化过程中的 accepted-point mutation。

---

## 4. 一阶变化公式

### 4.1 正交 frame 的切向形式

对上式取一阶变化，有

\[
\delta Q = Q \Omega,
\qquad
\Omega^{\mathrm T} = -\Omega.
\]

分块写出可得

\[
\delta Q_i = Q_a K_{ia} + Q_v K_{iv},
\]

\[
\delta Q_a = -Q_i K_{ia}^{\mathrm T} + Q_v K_{av},
\]

\[
\delta Q_v = -Q_i K_{iv}^{\mathrm T} - Q_a K_{av}^{\mathrm T}.
\]

### 4.2 辅助活性轨道的变化

由

\[
T_a = Q_a L_a
\]

可得

\[
\delta T_a = \delta Q_a \, L_a + Q_a \, \delta L_a.
\]

代入上面的 \(\delta Q_a\)：

\[
\delta T_a
=
\left(-Q_i K_{ia}^{\mathrm T} + Q_v K_{av}\right)L_a
+ Q_a \Delta L_a.
\]

因此活性变化被清楚拆成三类：

1. `inactive-active` 旋转：
   \[
   -Q_i K_{ia}^{\mathrm T} L_a;
   \]
2. `active-virtual` 旋转：
   \[
   Q_v K_{av} L_a;
   \]
3. `active-active` 非正交形状变化：
   \[
   Q_a \Delta L_a.
   \]

这正是“正交部分 rotation，非正交部分 mixing”的严格数学表达。

---

## 5. 梯度在 mixed-chart 中的分解

设当前公共回传已经给出辅助活性轨道的 AO 系数梯度

\[
G_T = \frac{\partial E}{\partial T_a}
\in \mathbb{R}^{N \times n_a}.
\]

由

\[
T_a = Q_a L_a
\]

得到

\[
\langle G_T, \delta T_a \rangle_S
=
\langle G_T, \delta Q_a L_a \rangle_S
+
\langle G_T, Q_a \Delta L_a \rangle_S.
\]

这里采用 \(S\)-度量 Frobenius 配对

\[
\langle X, Y \rangle_S = \operatorname{tr}(X^{\mathrm T} S Y).
\]

于是可定义对应的 ambient gradient：

\[
G_{Q_a} = G_T L_a^{\mathrm T},
\qquad
G_{L_a} = Q_a^{\mathrm T} S G_T.
\]

再与关于 \(Q_i\) 的 projector-pullback 梯度 \(G_{Q_i}\) 合并，就能把 rotation 梯度写成三个块：

\[
g_{ia}
=
Q_a^{\mathrm T} S G_{Q_i}
-
\left(Q_i^{\mathrm T} S G_{Q_a}\right)^{\mathrm T},
\]

\[
g_{iv}
=
Q_v^{\mathrm T} S G_{Q_i},
\]

\[
g_{av}
=
Q_v^{\mathrm T} S G_{Q_a}.
\]

而活性内部的非正交梯度则直接由

\[
g_{aa} = G_{L_a}
\]

给出，或者进一步按需要再映射到用户选定的 active-shape 参数。

因此 mixed-chart 的 reduced 变量可以统一写成

\[
\theta =
\left(
K_{ia},\;
K_{iv},\;
K_{av},\;
\Delta L_a
\right).
\]

### 5.1 mixed-chart 的 Gram / metric 结构

为了讨论外层信赖域、预条件子以及 reduced gradient 的尺度，必须把

\[
\langle \delta C, \delta \widetilde C \rangle_S
\]

写成 mixed-chart 变量上的显式二次型。

记

\[
A = K_{ia} \in \mathbb{R}^{n_a \times n_i},
\qquad
B = \Delta L_a \in \mathbb{R}^{n_a \times n_a},
\]

\[
X = K_{iv} \in \mathbb{R}^{n_v \times n_i},
\qquad
Y = K_{av} \in \mathbb{R}^{n_v \times n_a},
\]

并记

\[
M_i = U_i U_i^{\mathrm T} + K_a K_a^{\mathrm T}
\in \mathbb{R}^{n_i \times n_i},
\]

\[
N_a = L_a L_a^{\mathrm T}
\in \mathbb{R}^{n_a \times n_a}.
\]

则当前物理轨道的一阶变化可写成

\[
\delta C_i = Q_a A U_i + Q_v X U_i,
\]

\[
\delta C_a
=
-Q_i A^{\mathrm T} L_a
+ Q_a(B + A K_a)
+ Q_v(Y L_a + X K_a).
\]

由于 \(Q_i, Q_a, Q_v\) 彼此 \(S\)-正交，所有不同子空间之间的交叉项都严格为零，因此

\[
\langle \delta C, \delta \widetilde C \rangle_S
=
\langle \delta C_i, \delta \widetilde C_i \rangle_S
+
\langle \delta C_a, \delta \widetilde C_a \rangle_S
\]

可以分解为

\[
\langle \delta C, \delta \widetilde C \rangle_S
=
\operatorname{tr}(A M_i \widetilde A^{\mathrm T})
+
\operatorname{tr}(A^{\mathrm T} N_a \widetilde A)
+
\operatorname{tr}\!\left((B + A K_a)^{\mathrm T}
(\widetilde B + \widetilde A K_a)\right)
\]

\[
\qquad
+
\operatorname{tr}(X M_i \widetilde X^{\mathrm T})
+
\operatorname{tr}\!\left((Y L_a + X K_a)^{\mathrm T}
(\widetilde Y L_a + \widetilde X K_a)\right).
\]

这说明 mixed-chart 的 metric 不是完全对角的，但其结构非常稀疏：

1. \((A, B)\) 构成一个耦合子系统；
2. \((X, Y)\) 构成另一个耦合子系统；
3. 两个子系统之间没有直接耦合；
4. 所有耦合都只通过 \(K_a\) 与 \(L_a\) 出现。

也就是说，真正破坏“近似正交坐标”的不是 rotation 本身，而是物理活性轨道的 gauge/shape 加权。

### 5.2 metric 作用到 reduced 变量上的显式形式

把上面的双线性型对测试方向逐块配对，可得到 metric 作用

\[
(A, B, X, Y)
\mapsto
(G_A, G_B, G_X, G_Y),
\]

其中

\[
G_A = A M_i + N_a A + B K_a^{\mathrm T},
\]

\[
G_B = B + A K_a,
\]

\[
G_X = X M_i + Y L_a K_a^{\mathrm T},
\]

\[
G_Y = X K_a L_a^{\mathrm T} + Y N_a.
\]

这四个式子与当前 `project_dense_full_support_candidate_overlap(...)`
对 `build_dense_full_support_block_step(...)` 的回投影是一致的。

它们直接给出两个重要结论：

1. `inactive-active` 与 `active-shape` 之间确实通过 \(K_a\) 耦合；
2. `inactive-virtual` 与 `active-virtual` 之间也通过 \(K_a, L_a\) 耦合；
3. 因此 large dense OEO block 不能简单当成“原始系数 + 一个统一缩放”来处理；
4. 但它的对角项可以精确解析写出，不必再逐单位向量显式构造。

### 5.3 解析对角项

对单位方向逐一代入上式，可得到解析对角项：

对于 \(A_{rs}\)（其中 \(r\) 是活性行指标，\(s\) 是非活性列指标）：

\[
d_{ia}(r,s)
=
(M_i)_{ss} + (N_a)_{rr}.
\]

对于 \(B_{rt}\)：

\[
d_{aa}(r,t) = 1.
\]

对于 \(X_{us}\)（其中 \(u\) 是虚轨道行指标）：

\[
d_{iv}(u,s) = (M_i)_{ss}.
\]

对于 \(Y_{ur}\)：

\[
d_{av}(u,r) = (N_a)_{rr}.
\]

因此 dense full-support / OEO 路径的 `metric_diagonal` 可以直接由

\[
\operatorname{diag}(M_i),
\qquad
\operatorname{diag}(N_a)
\]

构造，而不必再对每个 reduced 方向显式调用一次
`build_dense_full_support_block_step + project_dense_full_support_candidate_overlap`。

---

## 6. 与当前代码的对应关系

当前实现并不是完全错误，而是“已经做了一半”。

### 6.1 `exact_ctx` 已经拥有的结构

当前 accepted-point cache 已经显式保存：

1. `inactive_orthonormal_orbital_matrix`，即 \(Q_i\)；
2. `auxiliary_orbital_matrix.middleCols(...)`，即 \(T_a\)；
3. 由 block-local occupied span 构造的虚轨道补空间。

因此在数学对象上，mixed-chart 所需的数据大部分已经存在。

### 6.2 `NonredundantOrbitalSpace` 当前做法

当前 `src/vb/orbital/nonredundant_orbital_space.cpp` 的 reduced 方向分成：

1. `inactive_active_coefficients`
2. `active_active_coefficients`
3. `occupied_virtual_coefficients`

并且：

1. `build_block_rotation_generator(...)` 只把 `inactive-active` 和 `occupied-virtual` 放进反对称 generator；
2. `active-active` 仍然保留为显式线性块；
3. `project_dense_full_support_candidate_overlap(...)` 实际上已经隐式实现了上一节给出的 metric 作用公式；
4. `build_dense_full_support_metric_diagonal(...)` 目前仍通过逐单位向量显式抽取对角项，这在理论上是多余的，因为该对角已经可以解析写出；
5. 对 dense full-support / OEO 分支，`retract_step(...)` 仍然故意沿用 legacy 的 one-sided `putDeltaOrb` 语义，而不是在正交 frame 上做真正的 rotation retraction。

这说明当前代码已经部分体现了 mixed-chart 思想，但还没有把正交 frame 提升为真正的一等工作变量。

---

## 7. 为什么这会改善收敛

如果对正交块仍使用一般系数混合，则 reduced step 里会混入很多“只是坐标变化”的坏方向：

1. 非活性块内部的正交约束不能被有限步更新自动保持；
2. internal chart 与 external physical chart 之间必须频繁 transport；
3. secant history 里会混入大量 chart-dependent 信息；
4. line search / trust region 看到的是扭曲后的坐标几何，而不是天然的轨道流形几何。

一旦把正交块统一改成 rotation：

1. 约束在几何上被精确保留；
2. `inactive-inactive` / `inactive-active` / `active-virtual` 的条件数通常更好；
3. HVP 与 preconditioner 可以在更干净的切空间里工作；
4. active-active 的非正交性仍被完整保留，不会破坏 VB 活性轨道的物理语义。

因此它不是“折中技巧”，而是更符合问题本身几何结构的参数化。

---

## 8. 对当前 OEO/TNHVP 路径的最小落地方案

推荐按下面顺序落地，而不是一次性重写所有公共模块。

### 第一步：把 OEO dense full-support 分支改成真正的 mixed-chart

在 `NonredundantOrbitalSpace` 的 dense full-support / OEO 分支中：

1. 用 accepted-point cache 中的 \(Q_i\) 替代物理 `inactive_frame`；
2. 由 \(T_a\) 构造 \(Q_a = T_a A_{aa}^{-1/2}\) 与 \(L_a = A_{aa}^{1/2}\)；
3. 用当前 block virtual builder 构造 \(Q_v\)；
4. reduced 变量改成
   \[
   (K_{ia}, K_{iv}, K_{av}, \Delta L_a);
   \]
5. finite step 使用 \(Q\) 上的 Cayley / exponential retraction，而不是 one-sided `putDeltaOrb`。

这是收益最大、也最符合当前 TiCl/FeCl 收敛现象的改造。

在不进一步改动 finite retraction 之前，一个安全的子步骤是先把
`build_dense_full_support_metric_diagonal(...)` 改成上一节的解析公式。
这一步不改变 mixed-chart 的一阶语义，只是把当前已经存在的
`D^{\mathrm T} D` 对角项显式化并降低 accepted-point 构建开销。

### 第二步：把 active representative 彻底降级为 gauge / export 语义

即：

1. 优化过程中只维护 \(Q_i\) 与 \(T_a\)；
2. \(C_i = Q_i U_i\) 与 \(C_a = T_a + Q_i K_a\) 只在输出或特定可视化时重构；
3. 不再把 representative reset 当作 accepted-point optimizer mutation。

### 第三步：再考虑把 HVP/gradient 完全迁移到 mixed-chart

此时 public pullback 的目标将从物理 AO 系数梯度，逐渐转为：

\[
G_{Q_i},\;
G_{Q_a},\;
G_{L_a}.
\]

然后统一映射到 reduced coordinates

\[
(g_{ia}, g_{iv}, g_{av}, g_{aa}).
\]

---

## 9. 推荐结论

对当前项目，最合理的结论是：

1. **正交部分全部应使用轨道旋转。**
2. **只有活性内部的非正交形状变量保留 mixing。**
3. **`Q_i` 的正交化若只停留在 exact_ctx 内部，而不进入外层更新 chart，则收益只能吃到一半。**
4. **TiCl/FeCl 上看到的收敛退化，本质上说明“内部正交、外部仍混合”的半混合语义还不够干净。**

因此，真正的下一步不是继续堆 accepted-point repair，而是把当前 OEO 路径升级为：

\[
\text{orthogonal frame rotations} + \text{active-shape mixing}.
\]

这才是“正交的部分都使用轨道旋转”的严格落地形式。
