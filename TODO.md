# VBSCF 底层算法重构规格说明书：从稠密余子式到真正的双正交变换

## 1. 架构目标 (Architecture Objective)
当前 VBSCF 底层评估两个非正交 Slater 行列式 $\langle \Psi_L | \hat{H} | \Psi_R \rangle$ 时，使用的是基于一阶、二阶代数余子式的广义 Löwdin 规则。这导致双电子积分收缩存在 $O(N^4)$ 的极其昂贵的四重循环瓶颈。

本次重构的目标是：**引入真正的双正交变换 (Biorthogonalization)**。
利用 SVD 得到的左右奇异向量 $\mathbf{U}$ 和 $\mathbf{V}$ 将活性轨道基底对角化，使得重叠矩阵 $\mathbf{S}$ 变为对角阵 $\mathbf{\Sigma}$。通过 $O(M_{act}^5)$ 的积分前置变换，换取前向能量评估的 $O(N^2)$ 坍缩，并通过跃迁密度矩阵 (TDM) 的反向拉回 (Pull-back) 彻底避免对 SVD 过程求导，实现解析梯度的极速计算。

---

## 2. 数据结构重构 (Phase 1: Data Structures)

修改 `DeterminantOverlapResult`，废弃存储稠密伴随矩阵的方案，改为保存完整的 SVD 分解结果。

**修改要求：**
- 移除：`std::vector<double> first_order_cofactor_matrix;`
- 新增：
  - `Eigen::VectorXd singular_values;` (存储奇异值 $\sigma_i$)
  - `ColumnMajorMatrixXd matrix_U;` (左奇异向量矩阵)
  - `ColumnMajorMatrixXd matrix_V;` (右奇异向量矩阵)
  - `double parity;` (存储 $\det(\mathbf{U}) \cdot \det(\mathbf{V})$，值为 `1.0` 或 `-1.0`)
- `overlap_determinant` 的计算变更为：`parity * singular_values.prod()`。

---

## 3. 核心新模块：双正交积分变换器 (Phase 2: Integral Transformer)

引入一个新的辅助类或函数空间，用于将原始活性空间 (VB basis) 的积分变换到当前的双正交基底 (Biorthogonal basis) 下。

**模块要求：**
1. **单电子积分变换** ($O(N^3)$):
   $$\tilde{\mathbf{h}} = \mathbf{U}^T \mathbf{h}_{active} \mathbf{V}$$
   使用 Eigen 实现：`transformed_h.noalias() = matrix_U.transpose() * active_h * matrix_V;`

2. **双电子积分变换** ($O(M_{act}^5)$):
   将输入的 `packed_active_two_electron_integrals` (具有 8 重对称性) 变换为新的双正交双电子积分 $(\tilde{i}\tilde{j}|\tilde{k}\tilde{l})$。
   公式：
   $$(\tilde{i}\tilde{j}|\tilde{k}\tilde{l}) = \sum_{p,q,r,s} U_{pi} U_{qj} V_{rk} V_{sl} (pq|rs)$$
   *Codex 提示*：为了性能，请勿使用 8 重嵌套的标量 `for` 循环。请将此四指数收缩展平为类似于连续矩阵乘法 (`Eigen::gemm`) 的形式，或提供优化的高速收缩逻辑。

---

## 4. 前向计算重写：能量组装 (Phase 3: Forward Pass - Energy)

彻底重写 `compute_same_spin_biorthogonal_phi` 等函数。废除传入稠密 `inverse_overlap_submatrix` 的四重循环，直接接收双正交变换后的单/双电子积分和奇异值数组。

**重写逻辑 (以满秩 `nullity == 0` 为例)：**
1. **单电子贡献** (单层循环 $O(N)$):
   $$E_{1e} = \sum_{i=0}^{N-1} \frac{\tilde{h}_{ii}}{\sigma_i}$$
2. **双电子贡献** (双重循环 $O(N^2)$):
   直接提取对角双电子积分：$J_{ij} = (\tilde{i}\tilde{i}|\tilde{j}\tilde{j})$, $K_{ij} = (\tilde{i}\tilde{j}|\tilde{j}\tilde{i})$。
   $$E_{2e} = \sum_{i=0}^{N-2} \sum_{j=i+1}^{N-1} \frac{J_{ij} - K_{ij}}{\sigma_i \sigma_j}$$
3. **奇异情况 (`nullity > 0`)**:
   - 如果 `nullity == 1` (仅 $\sigma_p \approx 0$): 能量公式退化为不除以 $\sigma_p$ 的剩余对角项求和。
   - 如果 `nullity == 2` (两个奇异值 $\approx 0$): 直接提取对应的双电子积分项。
4. 总能量乘以整体的行列式系数 `overlap_determinant`。

---

## 5. 反向计算重构：梯度 Pull-back (Phase 4: Backward Pass - Gradients)

废除当前 `accumulate_spin_overlap_gradient` 中基于逆矩阵导数 $(\mathbf{S}^{-1})^T$ 的组装方法。改用**跃迁密度矩阵 (TDM) 拉回机制**。

**重写逻辑：**
1. **组装双正交 TDM (极度稀疏)**：
   在双正交基下，1-RDM ($\tilde{\mathbf{D}}$) 是对角阵，2-RDM ($\tilde{\mathbf{d}}$) 也高度稀疏。
   - 满秩时：$\tilde{D}_{ii} = \frac{\det(\mathbf{S})}{\sigma_i}$。
2. **反向拉回 (Back-Transformation)**：
   将双正交 TDM 变换回原始活性空间 (Active AO/VB basis)：
   $$\mathbf{D}^{original} = \mathbf{U} \tilde{\mathbf{D}} \mathbf{V}^T$$
   $$d^{original}_{pqrs} = \sum_{ijkl} U_{pi} U_{qj} V_{rk} V_{sl} \tilde{d}_{ij,kl}$$
3. **全局累加**：
   将转换回原基底的这对行列式的 $\mathbf{D}^{original}$ 和 $\mathbf{d}^{original}$ 累加到全局的 Total 1-RDM 和 Total 2-RDM 结构中。
   *(注意：梯度现在不在此函数内直接生成，而是由外部利用最终的 Total RDM 统一缩并积分生成。)*

---

## 6. C++ & Eigen 编码规范与性能要求 (Coding Standards)
- 全局禁用动态大小矩阵在深层循环内的堆内存分配 (`new`/`delete`)。使用 `Eigen::Ref` 传递 Workspace 或提前预分配。
- 所有与 Eigen 相关的密集代数操作需显式调用 `.noalias()` 以防临时拷贝。
- 处理近奇异判定时，保留现有的 `linear_dependence_threshold_` 判定机制。