# 基于精确结构重叠约束的 DeepVBSCF 技术路线

本文档给出一版更贴合当前 `xmvb-cpp` 内核的数据流与优化目标的 DeepVBSCF 方案。核心思想不是预测原始 AO 双电子积分，也不是单独训练梯度头，而是：

$$
S_{\text{pred}} = S_{\text{exact}}, \qquad
H_{\text{pred}} = H_{\text{exact}}^{(1e)} + H_{\text{ML}}^{(2e)}
$$

即保留**精确结构重叠矩阵**与**精确一电子结构哈密顿矩阵**，仅用神经网络学习**结构哈密顿矩阵中的双电子残差项**。

---

## 1. 目标与边界

当前 C++ VB kernel 的精确路径可概括为：

1. 从几何、基组和 VB 输入构造 AO overlap 与 AO 一/二电子积分。
2. 由当前轨道系数重建 active orbital。
3. 构造 active-space `SSO`、`HHO`、`GGO`。
4. 在 determinant/structure 层累积出结构重叠矩阵 `S` 和结构哈密顿矩阵 `H`。
5. 通过广义本征值问题得到 VBSCF 能量，并对轨道参数反传梯度。

本方案只替换第 3-4 步中的**双电子贡献**，不替换：

- 精确结构重叠 `S`
- 精确一电子贡献 `H^(1e)`
- 广义本征值求解
- 轨道参数化与轨道重建逻辑

因此，模型学习目标不是 AO ERI，也不是 `GGO` 本身，而是：

$$
H_{\text{exact}}^{(2e)} = H_{\text{exact}} - H_{\text{exact}}^{(1e)}
$$

---

## 2. 输入定义

### 2.1 分子级输入

由几何与原子类型构造分子图，使用 `e3nn` 或等变图网络提取原子特征：

- 节点：原子序数 `Z`
- 几何：3D 坐标 `R`
- 输出：带角动量分辨率的等变原子特征

注意：当前内核使用的是 **Cartesian AO** 约定，而不是直接的实球谐 `2l+1` 约定。若网络内部采用 `e3nn` irreps，必须显式处理：

- AO 分量顺序对齐
- Cartesian AO 到球谐/irreps 的固定线性变换
- 反向映射回当前轨道系数定义时的约定一致性

### 2.2 轨道级输入

当前 VB 轨道参数 `C` 不是附加信息，而是模型输入的一部分。利用当前轨道系数，将原子高阶角动量特征重新组合为轨道特征：

$$
f_p = \text{Contract}(V_{\text{atom}}, C_p)
$$

这里的关键不是“学习轨道系数”，而是使用**当前优化迭代中的真实轨道系数**构造可微轨道 embedding，使模型输出对轨道参数保持可导。

### 2.3 结构级输入

每个价键结构表示为一个轨道图：

- 节点：active orbitals
- 节点特征：轨道 embedding、占据信息、局域环境
- 边特征：spin-pair、耦合方式、结构内配对关系

不同价键结构的差异主要体现在：

- 哪些轨道被占据
- alpha/beta 电子如何配对
- 结构展开到 determinant 后的符号与组合关系

因此，模型不应只编码单个 structure，还应编码 structure pair `(I, J)` 的关系。

---

## 3. 模型输出层级

### 3.1 不推荐的目标

以下目标不作为主路线：

- 直接预测 AO 两电子积分 `ggf`
- 直接预测 active-space 双电子积分 `GGO`
- 单独训练轨道梯度头

原因是它们仍然保留了较重的后处理链，或者会破坏能量与梯度的一致性。

### 3.2 推荐目标

模型直接预测结构矩阵中的双电子部分：

$$
H_{\text{ML}}^{(2e)}(I, J)
$$

并构造最终结构哈密顿矩阵：

$$
H_{\text{pred}}(I, J) = H_{\text{exact}}^{(1e)}(I, J) + H_{\text{ML}}^{(2e)}(I, J)
$$

同时保留精确结构重叠：

$$
S_{\text{pred}}(I, J) = S_{\text{exact}}(I, J)
$$

这样做的好处是：

- 避开 AO 二电子积分生成
- 避开 AO -> active-space `GGO` 变换
- 避开 determinant/structure 层双电子精确累积
- 保留广义本征值问题的物理约束

---

## 4. 结构对预测器

`H^{(2e)}` 不是单个 structure 的属性，而是 structure pair 的耦合量。因此预测器应采用 pairwise 设计。

推荐输入：

- 左结构 embedding `g_I`
- 右结构 embedding `g_J`
- 精确结构重叠 `S_{IJ}`
- 轨道级 cross features
- 可选的 determinant 展开统计特征

推荐输出：

- 上三角 `H^{(2e)}_{IJ}`，再做对称化得到完整矩阵

为了增强可学性，可将对角元与非对角元分头建模：

- diagonal head: 学习结构自能修正
- off-diagonal head: 学习 structure coupling

---

## 5. 训练目标

### 5.1 矩阵监督

由当前精确 kernel 生成标签：

$$
H_{\text{label}}^{(2e)} = H_{\text{exact}} - H_{\text{exact}}^{(1e)}
$$

对应损失：

$$
\mathcal{L}_H = \| H_{\text{ML}}^{(2e)} - H_{\text{label}}^{(2e)} \|^2
$$

### 5.2 能量监督

利用预测矩阵解广义本征值问题：

$$
H_{\text{pred}} C = S_{\text{exact}} C E
$$

对选定状态或态平均能量做监督：

$$
\mathcal{L}_E = \| E_{\text{pred}} - E_{\text{exact}} \|^2
$$

### 5.3 约束项

可选附加项：

- 哈密顿矩阵对称性约束
- 对角/非对角分块加权
- 低重叠结构对的鲁棒性加权

综合损失可写为：

$$
\mathcal{L} = \alpha \mathcal{L}_H + \beta \mathcal{L}_E + \gamma \mathcal{L}_{\text{sym}}
$$

---

## 6. 梯度策略

本方案**不单独训练梯度头**。

轨道优化时，定义混合能量：

$$
E_{\text{hybrid}}(\theta) =
E_{\text{ref}}(\theta) +
\lambda_{\text{sel}}\Big(
H_{\text{exact}}^{(1e)}(\theta) + H_{\text{ML}}^{(2e)}(\theta),
S_{\text{exact}}(\theta)
\Big)
$$

其中：

- `\theta` 是当前 VB 轨道参数
- `E_ref` 为精确参考一电子项与核排斥等已知贡献
- `\lambda_sel` 表示所选态或态平均的广义本征值结果

然后直接对 `E_hybrid(\theta)` 自动微分得到轨道梯度。

这意味着：

- `S_exact` 的导数保留精确路径
- `H_exact^(1e)` 的导数保留精确路径
- `H_ML^(2e)` 的导数由模型自动微分给出

所得梯度不是“精确量子化学能量梯度”，而是**混合 surrogate 能量的自洽梯度**。但它与模型输出的能量严格一致，因此适合驱动优化器。

---

## 7. 加速来源

若推理阶段只保留：

- 精确结构重叠 `S`
- 精确一电子结构哈密顿 `H^(1e)`
- ML 双电子结构残差 `H_ML^(2e)`

则可绕开：

- AO 两电子积分生成
- active-space `GGO` 构造
- determinant/structure 双电子精确累积

保留的主要计算为：

- 轨道重建与 structure overlap
- 一电子矩阵构造
- structure pair 模型前向
- 广义本征值求解

因此，本路线的加速点不是“更快地使用双电子积分”，而是**不再显式生成和累积双电子积分的精确贡献**。

---

## 8. 工程落地要求

为了在当前仓库中落地，该方案需要补齐以下能力：

1. 从 runtime 导出 `Z`、坐标、AO 到 atom/shell/component 的映射。
2. 提供一个轻量加载路径，在推理阶段跳过 AO 两电子积分生成。
3. 用现有精确 kernel 生成 `H^(2e)` 标签，作为训练集监督。
4. 在优化阶段保留 exact fallback，便于数值稳定性验证。

---

## 9. 总结

更准确的 DeepVBSCF 主路线应当定义为：

- **精确 `S`**
- **精确 `H^(1e)`**
- **学习结构哈密顿矩阵的双电子残差 `H^(2e)`**
- **通过广义本征值问题端到端训练**
- **通过单一混合能量自动微分获得轨道梯度**

这一路线比“直接预测 AO 双电子积分”更贴近当前 VB kernel 的真实数据流，也比“同时预测能量和梯度头”更容易保持优化中的数值一致性。
