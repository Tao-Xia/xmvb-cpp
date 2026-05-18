# Per-block 旋转更新：稀疏非正交 VBSCF 轨道优化的理论推导

## 1 Block 结构与支撑一致性

`detect_orbital_blocks` 将具有**完全相同的 AO 基函数集合**的轨道分入同一 block。因此：

- 设 block $B$ 的 AO 支撑为 $\{\mu_1, \ldots, \mu_n\}$，则 block 内每个轨道的存储系数恰好都在这 $n$ 个基函数上
- 不同 block 的 AO 支撑互不相交

记号：

- $n$：block AO 支撑大小
- $n_i$：block 内非活性占据轨道数
- $n_a$：block 内活性占据轨道数
- $n_{\text{occ}} = n_i + n_a$，$n_v = n - n_{\text{occ}}$
- $S_B \in \mathbb{R}^{n \times n}$：block AO 重叠矩阵
- $T_i \in \mathbb{R}^{n \times n_i}$，$T_a \in \mathbb{R}^{n \times n_a}$：占据轨道 raw 系数矩阵

## 2 正交工作框架的构造

### 2.1 非活性 S-正交化

$$
M_i = T_i^{\mathrm{T}} S_B T_i, \quad Q_i = T_i M_i^{-1/2}, \quad R_i = M_i^{1/2}.
$$

$Q_i^{\mathrm{T}} S_B Q_i = I$，物理关系 $T_i = Q_i R_i$。

### 2.2 活性轨道分解

非活性投影算符 $P_i = Q_i Q_i^{\mathrm{T}} S_B$。辅助活性轨道

$$
\widetilde{T}_a = (I - P_i) T_a, \quad Q_i^{\mathrm{T}} S_B \widetilde{T}_a = 0.
$$

S-正交化：

$$
\widetilde{M}_a = \widetilde{T}_a^{\mathrm{T}} S_B \widetilde{T}_a, \quad
Q_a = \widetilde{T}_a \widetilde{M}_a^{-1/2}, \quad
L_a = \widetilde{M}_a^{1/2}.
$$

$Q_a^{\mathrm{T}} S_B Q_a = I$，$\widetilde{T}_a = Q_a L_a$。

定义非活性-活性耦合矩阵 $G_a = Q_i^{\mathrm{T}} S_B T_a \in \mathbb{R}^{n_i \times n_a}$。由 $P_i$ 的定义：

$$
T_a = Q_i G_a + Q_a L_a.
$$

### 2.3 虚空间补集

构造 $Q_v \in \mathbb{R}^{n \times n_v}$ 使得

$$
Q = [Q_i, Q_a, Q_v] \in \mathbb{R}^{n \times n}, \quad Q^{\mathrm{T}} S_B Q = I.
$$

## 3 切空间分解

### 3.1 独立变量

反对称生成元

$$
\Omega = \begin{bmatrix}
0 & -A^{\mathrm{T}} & -X^{\mathrm{T}} \\
A & 0 & -Y^{\mathrm{T}} \\
X & Y & 0
\end{bmatrix}, \quad \Omega^{\mathrm{T}} = -\Omega.
$$

六组独立扰动变量

$$
\delta\eta = (A, X, Y, \Delta R_i, \Delta G_a, \Delta L_a),
$$

其中 $A \in \mathbb{R}^{n_a \times n_i}$，$X \in \mathbb{R}^{n_v \times n_i}$，$Y \in \mathbb{R}^{n_v \times n_a}$，$\Delta R_i \in \mathbb{R}^{n_i \times n_i}$，$\Delta G_a \in \mathbb{R}^{n_i \times n_a}$，$\Delta L_a \in \mathbb{R}^{n_a \times n_a}$。

### 3.2 工作框架的一阶扰动

$\delta Q = Q\Omega$：

$$
\delta Q_i = Q_a A + Q_v X,
$$
$$
\delta Q_a = -Q_i A^{\mathrm{T}} + Q_v Y,
$$
$$
\delta Q_v = -Q_i X^{\mathrm{T}} - Q_a Y^{\mathrm{T}}.
$$

### 3.3 物理轨道的完整扰动

活性轨道：

$$
\delta C_a = \delta Q_i \cdot G_a + Q_i \cdot \Delta G_a + \delta Q_a \cdot L_a + Q_a \cdot \Delta L_a,
$$

$$
\boxed{
\delta C_a = Q_i(\Delta G_a - A^{\mathrm{T}} L_a)
+ Q_a(\Delta L_a + A G_a)
+ Q_v(X G_a + Y L_a)
}.
$$

非活性轨道：

$$
\delta C_i = Q_i \Delta R_i + Q_a A R_i + Q_v X R_i.
$$

### 3.4 各变量的几何含义

| 变量 | 维度 | 作用 |
|------|------|------|
| $A$ | $n_a n_i$ | 非活性↔活性子空间旋转 |
| $X$ | $n_v n_i$ | 非活性↔虚空间旋转 |
| $Y$ | $n_v n_a$ | 活性↔虚空间旋转 |
| $\Delta R_i$ | $n_i^2$ | 非活性轨道形状（含对角元：S-范数缩放） |
| $\Delta G_a$ | $n_i n_a$ | 非活性-活性耦合变化 |
| $\Delta L_a$ | $n_a^2$ | 活性轨道非正交形状（含对角元：S-范数缩放） |

**自方向（$\Delta L_a$ 和 $\Delta R_i$ 的对角元）**：对非正交 VBSCF 轨道，S-范数缩放是有效的物理自由度。沿自方向 $\delta T_a = \alpha \cdot T_a[:, k] e_k^{\mathrm{T}}$，活性度量变化 $\delta S^{\text{act}}_{kk} = 2\alpha (T_a[:, k])^{\mathrm{T}} S T_a[:, k] \neq 0$，因此自方向改变能量。

**$\Delta G_a$ 的独立性**：$\delta C_a$ 中 $Q_i$ 分量的系数是 $(\Delta G_a - A^{\mathrm{T}} L_a)$，有 $n_i \times n_a$ 个独立参数。$\Delta G_a$ 允许非活性分量 $Q_i G_a$ 在不旋转 $Q_i$ 的情况下独立变化。

## 4 Cayley 回缩

### 4.1 旋转部分

$$
Q^+ = Q\left(I - \tfrac{1}{2}\Omega\right)^{-1}\left(I + \tfrac{1}{2}\Omega\right).
$$

精确保持 $Q^{+\mathrm{T}} S_B Q^+ = I$。

$\Omega$ 反对称，$I - \frac{1}{2}\Omega$ 的特征值为 $1 \pm \frac{i}{2}\lambda_j$，永远非零，Cayley 变换总是良定的。

### 4.2 形状与耦合部分

$$
L_a^+ = L_a + \Delta L_a, \quad
G_a^+ = G_a + \Delta G_a, \quad
R_i^+ = R_i + \Delta R_i.
$$

### 4.3 物理轨道的恢复

$$
C_a^+ = Q_i^+ G_a^+ + Q_a^+ L_a^+.
$$

Block 内第 $k$ 个活性轨道的新系数是 $C_a^+$ 的第 $k$ 列：

$$
c_a^{+,k} = Q_i^+ G_a^+[:, k] + Q_a^+ L_a^+[:, k].
$$

由于 block 内所有轨道共享相同的 $n$ 个 AO 基函数，$c_a^{+,k} \in \mathbb{R}^n$ 恰好是轨道 $k$ 需要写入的全部存储系数。**无需投影或截断。**

非活性轨道同理：

$$
c_i^{+,k} = Q_i^+ R_i^+[:, k].
$$

### 4.4 回缩的微分分析

Cayley 变换的展开：

$$
\text{Cayley}(t\Omega) = I + t\Omega + \tfrac{1}{2}t^2\Omega^2 + O(t^3).
$$

因此 $Q^+(t) = Q + t \cdot Q\Omega + \tfrac{1}{2}t^2 \cdot Q\Omega^2 + O(t^3)$。

一阶项 $Q\Omega$ 精确对应 §3.3 中的切空间映射。二阶项 $\frac{1}{2}Q\Omega^2$ 是几何拉回项的来源（§5.2）。

### 4.5 重正则化闭环

§4.3 给出回缩后的物理轨道 $(C_i^+, C_a^+)$。在下一步迭代中，需要从这些物理轨道**重新构造正交框架**，即重新执行 §2 的全部步骤：

1. 由 $C_i^+$ 重新计算 $M_i^+ = C_i^{+\mathrm{T}} S_B C_i^+$，正交化得 $Q_i^+$、$R_i^+$
2. 由 $Q_i^+$ 构造新的投影算符 $P_i^+ = Q_i^+ Q_i^{+\mathrm{T}} S_B$，投影得 $\widetilde{T}_a^+ = (I - P_i^+) C_a^+$
3. S-正交化 $\widetilde{T}_a^+$ 得 $Q_a^+$、$L_a^+$
4. 计算新的耦合矩阵 $G_a^+ = Q_i^{+\mathrm{T}} S_B C_a^+$

这一重正则化步骤是框架正确性的关键环节。原因是：虽然切空间在当前点满足 $\Omega_{aa} = 0$（无 gauge 方向），Cayley 变换的二阶项 $\Omega^2$ 会产生有效的子空间内旋转。例如，$(\Omega^2)_{aa} = -AA^{\mathrm{T}} - Y^{\mathrm{T}}Y$ 在有限步回缩后改变了 $Q_a$ 的 intra-active 结构。重正则化将这种 drift 吸收回规范分解 $(Q_i^+, Q_a^+, R_i^+, G_a^+, L_a^+)$，使下一步的切空间自动恢复到无 gauge 的水平截面。

完整迭代循环为：

$$
\underbrace{(T_i, T_a) \xrightarrow{\S 2} (Q, R_i, G_a, L_a)}_{\text{A: canonicalization}}
\;\to\;
\underbrace{\text{在无 gauge 切空间中求解 Newton 方向}}_{\text{B: tangent optimization}}
\;\to\;
\underbrace{(Q^+, R_i^+, G_a^+, L_a^+) \xrightarrow{\S 4} (T_i^+, T_a^+)}_{\text{C: retraction}}
\;\to\;
\underbrace{(T_i^+, T_a^+) \xrightarrow{\S 2} \text{新的}(Q, R_i, G_a, L_a)}_{\text{D: re-canonicalization}}
$$

每步都从物理轨道出发重新分解，不累积框架的历史状态。图表的 gauge-fixed 性质在每步被重置。

## 5 HVP 结构

### 5.1 约化梯度

$$
g = J^{\mathrm{T}} g_{\text{raw}},
$$

其中 $J = \partial\varphi/\partial\eta|_{\eta=0}$ 是回缩在 $\eta = 0$ 处的雅可比。由 §4.4，$J$ 精确对应 §3.3 的扰动公式。

### 5.2 约化 HVP

沿约化方向 $v$：

$$
Hv = J^{\mathrm{T}}\,\delta g_{\text{raw}}(Jv) + \dot{J}(v)^{\mathrm{T}} g_{\text{raw}}.
$$

两项分别为：

**直接项 + 响应项**：$J^{\mathrm{T}} \delta g_{\text{raw}}(Jv)$。原始梯度沿切向量 $Jv$（即物理轨道方向 $\delta C_a$）的方向导数，投影到约化空间。包含活性空间直接方向导数和结构系数弛豫响应（Schur 补）。

**几何拉回项**：$\dot{J}(v)^{\mathrm{T}} g_{\text{raw}}$。由回缩的二阶展开精确推导。

物理轨道回缩为

$$
C(\theta) = Q \cdot \operatorname{Cayley}(\Omega(\theta)) \cdot M(\theta),
$$

其中 $M(\theta) = [R_i + \Delta R_i, G_a + \Delta G_a;\, 0, L_a + \Delta L_a;\, 0, 0]$ 是形状矩阵，$M = M(0)$ 是接受点的分解矩阵。将 $\operatorname{Cayley}(\Omega) = I + \Omega + \frac{1}{2}\Omega^2 + O(\Omega^3)$ 与 $M(\theta) = M + \sum_{\alpha}\theta_{\alpha}M_{\alpha}$ 代入：

$$
C(\theta) = QM + \underbrace{Q\cdot\Omega(\theta)\cdot M + Q\cdot\sum_{\alpha}\theta_{\alpha}M_{\alpha}}_{J\theta}
+ \underbrace{\tfrac{1}{2}Q\cdot\Omega(\theta)^2\cdot M + Q\cdot\Omega(\theta)\cdot\sum_{\alpha}\theta_{\alpha}M_{\alpha}}_{\frac{1}{2}\theta^{\mathrm{T}}H_C\theta} + O(\theta^3).
$$

因此物理 Hessian 为

$$
\frac{\partial^2 C}{\partial\theta_{\alpha}\partial\theta_{\beta}}\Big|_0
= Q \cdot \tfrac{1}{2}\{\Omega_{\alpha}, \Omega_{\beta}\} \cdot M
+ Q \cdot \Omega_{\alpha} \cdot M_{\beta}
+ Q \cdot \Omega_{\beta} \cdot M_{\alpha}.
$$

定义 $g_Q = Q^{\mathrm{T}} g_{\text{raw}}$（$n\times n_{\text{occ}}$，梯度在 $Q$ 基下的表示），则约化空间中的几何 HVP 对方向 $v$ 有

$$
\boxed{
(H^{\text{geom}}v)_{\alpha}=
\operatorname{tr}\!\big(g_Q^{\mathrm{T}}\cdot \tfrac{1}{2}\{\Omega_{\alpha},\Omega_v\}\cdot M\big)
+ \operatorname{tr}\!\big(g_Q^{\mathrm{T}}\cdot\Omega_{\alpha}\cdot M_v\big)
+ \operatorname{tr}\!\big(g_Q^{\mathrm{T}}\cdot\Omega_v\cdot M_{\alpha}\big),
}
$$

其中 $\Omega_v = \sum_{\beta}v_{\beta}\Omega_{\beta}$（仅旋转变量非零），$M_v = \sum_{\beta}v_{\beta}M_{\beta}$（仅形状变量非零）。三项的物理解释：

1. **旋转-旋转项**（$\operatorname{tr}(g_Q^{\mathrm{T}}\cdot\frac{1}{2}\{\Omega_{\alpha},\Omega_v\}\cdot M)$）：Cayley 变换的二阶弯曲，作用于梯度在 $M$ 上的投影。仅当 $\alpha$ 为旋转变量 $(A,X,Y)$ 且 $v$ 包含非零旋转分量时非零。
2. **旋转-形状耦合项**（$\operatorname{tr}(g_Q^{\mathrm{T}}\cdot\Omega_{\alpha}\cdot M_v)$）：旋转 $\alpha$ 与方向 $v$ 的形状变化之间的交叉导数。仅当 $\alpha$ 为旋转变量且 $v$ 包含非零形状分量时非零。
3. **形状-旋转耦合项**（$\operatorname{tr}(g_Q^{\mathrm{T}}\cdot\Omega_v\cdot M_{\alpha})$）：方向 $v$ 的旋转与形状变量 $\alpha$ 的交叉导数。仅当 $\alpha$ 为形状变量 $(\Delta R_i,\Delta G_a,\Delta L_a)$ 且 $v$ 包含非零旋转分量时非零。

**高效计算**：定义中间量 $W = M \cdot g_Q^{\mathrm{T}} \in \mathbb{R}^{n\times n}$ 和 $K = M_v \cdot g_Q^{\mathrm{T}} \in \mathbb{R}^{n\times n}$，令 $Z = \frac{1}{2}(\Omega_v W + W\Omega_v) + K$。则旋转分量的几何 HVP 分量为 $Z$ 反对称部分的块提取：

$$
gA_{\text{geom}}(i,a) = Z_{i,\,n_i+a} - Z_{n_i+a,\,i},
\qquad
gX_{\text{geom}}(i,v) = Z_{i,\,n_i+n_a+v} - Z_{n_i+n_a+v,\,i},
\qquad
gY_{\text{geom}}(a,v) = Z_{n_i+a,\,n_i+n_a+v} - Z_{n_i+n_a+v,\,n_i+a}.
$$

形状分量的几何 HVP 由 $H = g_Q^{\mathrm{T}}\cdot\Omega_v \in \mathbb{R}^{n_{\text{occ}}\times n}$ 的对应子块转置给出：

$$
g\Delta R_i^{\text{geom}} = H_{0:n_i,\,0:n_i}^{\mathrm{T}},
\quad
g\Delta G_a^{\text{geom}} = H_{n_i:n_i+n_a,\,0:n_i}^{\mathrm{T}},
\quad
g\Delta L_a^{\text{geom}} = H_{n_i:n_i+n_a,\,n_i:n_i+n_a}^{\mathrm{T}}.
$$

总计算代价 $O(n^3)$，对典型 block（$n\approx 20$–$40$）开销远小于 HVP 主体。

## 6 单射性证明：图表无 gauge 零模

### 6.1 维度计数

六组独立变量的总维度：

$$
d_B = \underbrace{n_a n_i}_{A} + \underbrace{n_v n_i}_{X} + \underbrace{n_v n_a}_{Y}
+ \underbrace{n_i^2}_{\Delta R_i} + \underbrace{n_i n_a}_{\Delta G_a}
+ \underbrace{n_a^2}_{\Delta L_a}.
$$

Block 内物理轨道系数总数为 $n \cdot n_{\text{occ}} = n(n_i + n_a)$。代入 $n_v = n - n_i - n_a$：

$$
d_B = n_a n_i + (n - n_i - n_a)n_i + (n - n_i - n_a)n_a + n_i^2 + n_i n_a + n_a^2.
$$

展开后每一项恰好相消为：

$$
\boxed{d_B = n(n_i + n_a) = n \cdot n_{\text{occ}}}.
$$

参数维度**精确等于**物理轨道系数总数，不多不少。

### 6.2 单射性证明

若扰动 $\delta\eta = (A, X, Y, \Delta R_i, \Delta G_a, \Delta L_a)$ 产生零物理变化 $\delta C_i = 0$ 且 $\delta C_a = 0$，则由 §3.3：

**从 $\delta C_i = 0$**：

$$
Q_i \Delta R_i + Q_a A R_i + Q_v X R_i = 0.
$$

$Q = [Q_i, Q_a, Q_v]$ 构成 $\mathbb{R}^n$ 的 $S$-正交基（$Q^{\mathrm{T}} S_B Q = I$），因此各分量独立：

- $Q_a$ 分量：$A R_i = 0$。$R_i = M_i^{1/2}$ 对称正定，故 $A = 0$。
- $Q_v$ 分量：$X R_i = 0$。同理 $X = 0$。
- $Q_i$ 分量：$\Delta R_i = 0$。

**从 $\delta C_a = 0$**，代入 $A = 0$：

$$
Q_i(\Delta G_a) + Q_a(\Delta L_a) + Q_v(Y L_a) = 0.
$$

同样由 $Q$ 的正交分解：

- $Q_i$ 分量：$\Delta G_a = 0$。
- $Q_v$ 分量：$Y L_a = 0$。$L_a = \widetilde{M}_a^{1/2}$ 对称正定，故 $Y = 0$。
- $Q_a$ 分量：$\Delta L_a = 0$。

因此 $\delta\eta = 0$，映射 $\delta\eta \mapsto (\delta C_i, \delta C_a)$ 是**单射**。

### 6.3 为什么不需要 Cholesky gauge

一个自然的疑问是：$T_a = Q_i G_a + Q_a L_a$ 在变换 $Q_a \to Q_a U_a$，$L_a \to U_a^{-1} L_a$ 下不变，这是否意味着 $\Delta L_a$ 中包含 gauge 零模？

答案是**否**。关键在于我们的参数化选择了**特定的正交分解** $Q = [Q_i, Q_a, Q_v]$，而非任意的 $(Q_a, L_a)$ 组合。$\Delta L_a$ 是在这个固定分解下的独立扰动变量，不与任何 $Q$ 的旋转耦合：

1. $\Omega$ 只包含**跨子空间**旋转（$A$: inactive↔active，$X$: inactive↔virtual，$Y$: active↔virtual），不包含 intra-active 或 intra-inactive 旋转
2. $\Delta L_a$ 的 $n_a^2$ 个分量全部独立作用在 $Q_a$ 子空间上

GPT 的分析正确指出了 $T_a = Q_i G_a + Q_a L_a$ 整体上的 gauge 冗余。但在我们的切空间参数化中，$\Omega$ 已经排除了子空间内旋转，而 $\Delta L_a$ 作为独立变量覆盖了 $n_a^2$ 个自由度。由于 $d_B = n \cdot n_{\text{occ}}$（精确匹配物理维度）且映射单射，这些自由度全部是物理的，不存在 gauge 零模。

等价地，intra-active 旋转 $Q_a \to Q_a(I + W_a)$ 在我们的参数化中同时需要 $\delta Q_a = Q_a W_a$（来自 $\Omega$）和 $\Delta L_a = -W_a L_a$。但 $\delta Q_a = Q_a W_a$ 不在我们的 $\Omega$ 中（$\Omega$ 无 intra-active 分量），因此这样的 gauge 变换**无法由我们的切变量表示**。切空间本身已经与 gauge 方向正交。

### 6.4 图表的有效域

§6.2 的单射性证明依赖 $R_i$ 和 $L_a$ 的可逆性。图表的参数化

$$
\Phi_B: (Q, R_i, G_a, L_a) \mapsto (T_i, T_a)
$$

定义在开集

$$
\mathcal{U}_B = \{(Q, R_i, G_a, L_a) : R_i \in \mathrm{SPD}(n_i),\; L_a \in \mathrm{SPD}(n_a),\; Q^{\mathrm{T}} S_B Q = I\}
$$

上。边界行为：当 $\lambda_{\min}(R_i) \to 0$ 或 $\lambda_{\min}(L_a) \to 0$ 时，两个活性或非活性轨道趋近线性相关，$M^{-1/2}$ 的条件数发散，图表退化。

这是非正交轨道优化的固有困难——所有参数化都面临此奇异性，并非本框架特有。数值处理：对 $M$ 做特征分解，截断小于阈值的特征值，用正则化的 $M^{-1/2}$ 代替精确值。$Q^{\mathrm{T}}SQ = I$ 在截断精度内成立。Trust-region 机制自动限制步长，防止穿越图表边界。

### 6.5 局部等价性

映射 $\Phi_B: \delta\eta \mapsto (\delta C_i, \delta C_a)$ 在 block 内定义。由于 $d_B = n \cdot n_{\text{occ}}$ 精确匹配物理维度且映射单射，$\Phi_B$ 在 $\delta\eta = 0$ 处的雅可比是 $n \cdot n_{\text{occ}}$ 阶可逆方阵。由逆函数定理，$\Phi_B$ 是局部微分同胚。图表与原始非正交优化问题局部等价。

剩余的 $\frac{1}{2}n_{\text{occ}}(n_{\text{occ}} - 1)$ 个非物理自由度（$Q$ 的整体正交旋转）不出现，因为它们已被参数化的结构排除：$\Omega$ 只含跨子空间旋转，形状变量 $(R_i, G_a, L_a)$ 在固定 $Q$ 分解下独立变化。

### 6.6 Pullback Hessian 与 retraction 的关系

§5.2 的几何拉回项来源于 Cayley 回缩的二阶展开，是 retraction-induced pullback Hessian。注意完整几何拉回包含三部分：Cayley 弯曲项 $\frac{1}{2}\{\Omega_{\alpha},\Omega_v\}$、旋转-形状耦合项 $\Omega_{\alpha}M_v$、以及形状-旋转耦合项 $\Omega_v M_{\alpha}$。不同回缩（exponential、QR、polar、Cayley）产生不同的 $\dot{J}$ 二阶项，因此对应不同的 pullback Hessian。

这并不影响优化收敛性：Riemannian trust-region 理论正是使用 retraction-induced pullback Hessian。所有回缩的一阶行为一致（精确对应切空间映射），二阶项的差异在 trust-region 框架下被步长控制吸收。Cayley 变换作为回缩的优势在于：(1) 精确保持 $Q^{\mathrm{T}} S Q = I$（无需迭代），(2) $I - \frac{1}{2}\Omega$ 总是可逆（Cayley 变换无奇异性），(3) 二阶项有封闭形式，可通过 $Z = \frac{1}{2}(\Omega_v W + W\Omega_v) + K$ 高效计算。

## 7 Cayley 图表的有效域：HAO 限制

### 7.1 数值证据

Cayley 图表回缩在四类体系上进行了测试。结果按轨道类型分类：

| 体系 | 轨道类型 | Cayley ON | Cayley OFF |
|------|---------|-----------|-------------|
| FeCl2 | OEO (全 AO 支撑) | 3 步收敛 | 3 步收敛 |
| 240 | HAO (稀疏 AO 支撑) | **梯度爆炸** | 22 步收敛 |
| 241 | HAO (稀疏 AO 支撑) | 27 步收敛 | 25 步收敛 |
| MnF2 | HAO (稀疏 AO 支撑) | **块重建崩溃** | 201 步收敛 |

对于 OEO 体系（FeCl2），Cayley 图表功能良好。对于 HAO 体系，其表现从轻微劣化（241）到彻底失败（240、MnF2）不等。

### 7.2 数学原因：HAO 系统中 S-正交化产生的度量耦合

Cayley 分解构造 S-正交矩阵 $Q$（$Q^{\mathrm{T}} S Q = I$）并在 Q 基中定义切变量。物理轨道恢复为 $C = Q M$，其中度量矩阵 $M$（$R_i, G_a, L_a$）充当 Q 基与物理空间之间的桥梁。

对于 OEO 体系（全 AO 支撑，$E_p = I$），S-正交化是良好调节的：物理度量 $S$ 是单位矩阵的稠密扰动，且 $Q^{\mathrm{T}}Q \approx I$。Q 基切方向与物理系数变化之间的耦合是弱且平滑的。

对于 HAO 体系（稀疏 AO 支撑），每个轨道的系数仅在一小部分 AO 基函数上非零。S-正交化在块内的不同 HAO 轨道之间产生了**非对角度量耦合**。具体而言：

- 物理轨道 $C_i$ 具有稀疏支撑 $\{\mu_1,\ldots,\mu_{n_i}\}$，但 S-正交化的 $Q_i = C_i M_i^{-1/2}$ 在每个 Q_i 向量中混合了所有 inactive 轨道
- $Q_i^{\mathrm{T}} S Q_i = I$ 在 S 度量下保持正交性，但 $Q_i$ 列现在具有稠密 AO 支撑
- Cayley 旋转 $Q^+ = Q \cdot \mathrm{Cayley}(\Omega)$ 产生横跨整个块 AO 支撑的 Q 基向量，而 HAO 约束要求每个单独的轨道保持其原始稀疏支撑
- 在第 4.5 节的重新正则化过程中，当新的物理轨道 $C^+ = Q^+ M^+$ 计算出来后，HAO 稀疏性通过 `enforce_strict_sparse_orbital_support` 强制施加，该函数**裁剪**了超出每个轨道指定支撑的系数

这种裁剪破坏了 $Q^+$ 的 S-正交性（$Q^{+\mathrm{T}} S Q^+ \neq I$），并为下一次迭代的度量矩阵 $M$ 引入了不连续性。经过多次迭代后，数值度量与代数度量之间的差异不断扩大，导致：

- 切映射 $J$ 不再准确表示一阶物理变化
- 梯度投影 $J^{\mathrm{T}} g$ 失去精度
- Trust-region 模型不一致，导致步长频繁被拒绝，trust radius 崩溃

### 7.3 修复策略

Cayley 图表应**仅对 OEO（全 AO 支撑）体系启用**。对于 HAO 体系，应使用逐轨道物理切基 $U_p$（Cayley OFF 路径）：

```
对于 OEO 体系: 使用 Cayley 图表（S-正交化 + Q 基切变量 + 几何拉回）
对于 HAO 体系: 使用逐轨道 $U_p$ 切基（物理系数空间, 无几何拉回）
```

HAO 的逐轨道 $U_p$ 路径通过直接在原始物理系数空间中参数化切方向来避免度量耦合问题，其中稀疏性约束是自然满足的。该路径的 HVP 一致性已在数值上验证（解析 vs 有限差分误差 < $10^{-8}$）。

$U_p$ 路径中不需要几何拉回，因为回缩是线性的（$C^+ = C + \delta C$ 后逐列归一化），因此 $R''(0) = 0$。修正后的 Cayley 图表的几何拉回公式（§5.2）仍然是**在数学上正确的**，但仅在 OEO 体系上数值有效，这些体系没有会破坏它的稀疏性不连续性。
