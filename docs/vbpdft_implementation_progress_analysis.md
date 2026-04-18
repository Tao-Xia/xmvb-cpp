# VB-PDFT 实现进度分析报告

## 执行摘要

**当前状态：** 基础设施层 **60-70% 完成**，但距离可用的 VB-PDFT 能量计算还有 **关键缺失组件**

**最大差距：** 缺少 grid + libxc 集成层（约占剩余工作量的 40-50%）

**预计完成时间：** 如果全职投入，约需 **2-3 个月** 完成 Stage 1A（能量计算）

---

## 1. 规划文档分析

### 1.1 总体目标（来自 `vb_pdft.md`）

**Stage 1A 目标：**
- 闭壳层
- 单态特定（state-specific）
- 后 VBSCF（post-VBSCF）
- 仅能量计算
- 使用 translated LDA
- 无轨道响应
- 无解析梯度

**能量公式：**
$$
E_n^{\text{VB-PDFT}} = V_{nn} + E_{\text{one}}[\gamma^{(n)}] + J[\rho^{(n)}] + E_{\text{ot}}[\rho^{(n)}, \Pi^{(n)}]
$$

### 1.2 关键设计约束

**最重要的约束：** 必须使用**物理轨道框架**（physical orbital frame），而不是当前的辅助/投影轨道（auxiliary/projected orbitals）

当前 VBSCF 使用：
$$
T_a = (I - P_{11} S) C_a
$$

但 PDFT 需要：
$$
\rho(\mathbf{r}), \Pi(\mathbf{r}) \text{ 必须从物理轨道 } C_a \text{ 构建}
$$

---

## 2. 已完成的组件

### 2.1 物理轨道框架 ✅

**文件：** `src/vb/pdft/physical_orbital_frame.hpp`

**状态：** 已定义数据结构

```cpp
struct PhysicalOrbitalFrame {
  std::vector<double> normalized_orbital_matrix;        // C
  std::vector<double> inactive_physical_orbital_matrix; // C_inactive
  std::vector<double> active_physical_orbital_matrix;   // C_active
};
```

**完成度：** 100%（数据结构）

**缺失：** 从 `OrbitalPreparationResult` 提取物理轨道的构建函数

### 2.2 选择态 1-RDM 构建器 ✅

**文件：**
- `selected_state_matrix_form_one_rdm_builder.hpp/cpp` (181 行)
- `selected_state_exact_physical_one_rdm_builder.hpp/cpp` (568 行)

**功能：**
1. **Matrix-form 1-RDM**：在当前非正交活性轨道基下的密度矩阵
   - 与 `HHO` 直接收缩
   - 通过重叠度规 `SSO` 解释

2. **Physical 1-RDM**：在物理轨道框架下的密度矩阵
   - 用于实空间密度 $\rho(\mathbf{r})$
   - 已验证电子数守恒

**完成度：** 95%

**验证状态：**
- ✅ 电子数守恒：`Tr(γ * S) = N_active`
- ✅ 与活性空间梯度一致性
- ✅ 对称性检查

### 2.3 选择态 2-RDM / On-top Pair Density 构建器 ✅

**文件：**
- `selected_state_matrix_form_two_rdm_builder.hpp/cpp` (205 行)
- `selected_state_exact_physical_on_top_pair_density_builder.hpp/cpp` (702 行)

**功能：**
1. **Matrix-form 2-RDM**：在活性轨道基下的双粒子密度
   - 与 `GGO` 直接收缩

2. **Physical On-top Pair Density**：物理框架下的 on-top 对密度
   - 避免显式构造完整 4-index 2-RDM
   - 提供逐点评估接口：
     ```cpp
     double evaluate_selected_state_exact_physical_on_top_pair_density(
         const Context& context,
         const std::vector<double>& ao_values);
     ```

**完成度：** 90%

**关键特性：**
- ✅ 支持批量评估（多个格点）
- ✅ 使用 cofactor 方法避免完整 2-RDM
- ✅ 归一化验证

**缺失：** 与实际 grid 集成的测试

### 2.4 测试工具 ✅

**文件：** `src/tools/check_selected_state_exact_physical_on_top_pair_density.cpp`

**功能：** 验证 on-top pair density 构建的正确性

**完成度：** 100%

---

## 3. 缺失的关键组件

### 3.1 Grid 生成器 ❌ **（最大缺口）**

**需要：** `src/vb/pdft/grid_builder.hpp/cpp`

**功能：**
- 原子中心积分格点生成
- Becke 分区
- 径向 + 角度积分
- 块布局优化

**预计工作量：** 1000-1500 行代码，2-3 周

**依赖：**
- 需要选择径向积分方案（Gauss-Chebyshev, Mura-Knowles, etc.）
- 需要选择角度积分方案（Lebedev 格点）
- 需要实现 Becke 权重函数

**参考实现：**
- PySCF 的 `pyscf.dft.gen_grid`
- Libxc 的 grid 示例
- 或者直接调用外部库（如 `libgrpp`）

### 3.2 AO 值在格点上的评估器 ❌

**需要：** `src/vb/pdft/libcint_ao_grid_evaluator.hpp/cpp`

**功能：**
- 使用 `LibcintInput` 数据
- 评估 AO 基函数值 $\chi_\mu(\mathbf{r}_g)$
- （后续）评估 AO 梯度（用于 GGA）

**预计工作量：** 500-800 行代码，1-2 周

**技术细节：**
- 需要调用 `libcint` 的 `int1e_ovlp` 类似接口
- 需要处理不同壳层类型（s, p, d, f）
- 需要批量评估优化

### 3.3 On-top Functional 包装器 ❌

**需要：** `src/vb/pdft/on_top_functional.hpp/cpp`

**功能：**
1. 计算 translated spin polarization：
   $$
   R(\mathbf{r}) = \frac{4\Pi(\mathbf{r})}{\rho(\mathbf{r})^2}
   $$
   $$
   \zeta_t(\mathbf{r}) = \sqrt{\max(0, 1 - R(\mathbf{r}))}
   $$

2. 构造有效自旋密度：
   $$
   \tilde{\rho}_\alpha = \frac{\rho}{2}(1 + \zeta_t), \quad
   \tilde{\rho}_\beta = \frac{\rho}{2}(1 - \zeta_t)
   $$

3. 调用 `libxc` 计算 $\varepsilon_{xc}^{\text{LDA}}$

**预计工作量：** 300-500 行代码，1 周

**依赖：**
- 需要链接 `libxc`
- 需要处理数值稳定性（$\rho \to 0$ 时）

### 3.4 端到端 VB-PDFT 能量评估器 ❌

**需要：** `src/vb/pdft/vb_pdft_energy_evaluator.hpp/cpp`

**功能：**
1. 调用 VBSCF 或接受 VBSCF 结果
2. 构建 $\rho(\mathbf{r})$ 和 $\Pi(\mathbf{r})$
3. 在格点上积分 on-top functional
4. 返回 $E_n^{\text{VB-PDFT}}$

**预计工作量：** 400-600 行代码，1-2 周

**集成挑战：**
- 需要协调所有上述组件
- 需要处理并行化（OpenMP）
- 需要内存管理优化

---

## 4. 完成度评估

### 4.1 按模块分类

| 模块 | 状态 | 完成度 | 代码行数 | 缺失工作量 |
|------|------|--------|----------|-----------|
| 物理轨道框架 | ✅ 定义完成 | 100% | 40 | 构建函数 (~100 行) |
| 1-RDM 构建器 | ✅ 完成 | 95% | 749 | 测试 (~50 行) |
| 2-RDM / On-top 构建器 | ✅ 完成 | 90% | 907 | Grid 集成测试 |
| Grid 生成器 | ❌ 未开始 | 0% | 0 | ~1200 行 |
| AO 格点评估器 | ❌ 未开始 | 0% | 0 | ~650 行 |
| On-top Functional | ❌ 未开始 | 0% | ~350 行 |
| 端到端评估器 | ❌ 未开始 | 0% | 0 | ~500 行 |
| **总计** | | **~35%** | **1696** | **~2850 行** |

### 4.2 按功能分类

| 功能层 | 完成度 | 说明 |
|--------|--------|------|
| **理论基础** | 100% | 文档完善，公式清晰 |
| **数据结构** | 90% | RDM 和 on-top 数据结构完成 |
| **密度构建** | 85% | 1-RDM 和 on-top pair density 完成 |
| **实空间积分** | 0% | Grid + AO 评估完全缺失 |
| **DFT Functional** | 0% | Libxc 集成缺失 |
| **端到端流程** | 0% | 集成层缺失 |

### 4.3 关键路径分析

**当前瓶颈：** Grid 生成 + AO 评估（占剩余工作 65%）

**依赖关系：**
```
Grid 生成器 (必需)
    ↓
AO 格点评估器 (必需)
    ↓
On-top Functional (必需)
    ↓
端到端评估器 (必需)
    ↓
Stage 1A 完成
```

---

## 5. 预计完成时间

### 5.1 乐观估计（全职投入）

| 任务 | 时间 | 累计 |
|------|------|------|
| Grid 生成器 | 2-3 周 | 3 周 |
| AO 格点评估器 | 1-2 周 | 5 周 |
| On-top Functional | 1 周 | 6 周 |
| 端到端评估器 | 1-2 周 | 8 周 |
| 测试 + 调试 | 2 周 | 10 周 |
| **总计** | **~2.5 个月** | |

### 5.2 现实估计（兼职投入）

假设每周投入 20 小时：
- **预计时间：** 4-6 个月

### 5.3 保守估计（考虑意外）

包括：
- 数值稳定性问题
- 性能优化
- 与现有代码集成问题

- **预计时间：** 6-9 个月

---

## 6. 技术风险

### 6.1 高风险项

1. **Grid 生成的数值精度**
   - Becke 分区在分子边界处可能不稳定
   - 需要仔细调试权重函数

2. **On-top pair density 的数值稳定性**
   - $R(\mathbf{r}) = 4\Pi/\rho^2$ 在 $\rho \to 0$ 时发散
   - 需要截断和正则化策略

3. **性能问题**
   - Grid 积分可能很慢（数万个格点）
   - 需要 OpenMP 并行化
   - 可能需要 GPU 加速

### 6.2 中风险项

1. **Libxc 集成**
   - API 可能变化
   - 需要处理不同版本兼容性

2. **内存管理**
   - 大分子的 grid 数据可能很大
   - 需要流式处理

### 6.3 低风险项

1. **RDM 正确性**
   - 已有完善的验证
   - 与梯度一致性已确认

2. **物理轨道提取**
   - 逻辑清晰
   - 实现简单

---

## 7. 优化建议

### 7.1 短期（加速 Stage 1A）

**选项 1：使用外部 Grid 库**
- 使用 PySCF 的 grid 生成器（通过 Python 接口）
- 或者使用 `libgrpp` 等现成库
- **优点：** 节省 2-3 周开发时间
- **缺点：** 增加依赖

**选项 2：简化 Grid 方案**
- 先实现均匀格点（而非 Becke 分区）
- 仅用于概念验证
- **优点：** 实现简单（1 周）
- **缺点：** 精度较低，不适合发表

**选项 3：并行开发**
- Grid 生成器和 AO 评估器可以并行开发
- 使用 mock 数据测试下游组件
- **优点：** 节省 1-2 周
- **缺点：** 需要多人协作

### 7.2 中期（Stage 1B）

1. **性能优化**
   - 实现 grid 缓存
   - 批量 AO 评估
   - OpenMP 并行化

2. **扩展到 GGA**
   - 添加 AO 梯度评估
   - 实现 translated PBE

### 7.3 长期（Stage 2）

1. **解析梯度**
   - 需要 grid 权重的导数
   - 需要 functional 导数
   - 预计额外 3-6 个月

2. **态平均和多态**
   - 扩展到多个态
   - 实现 XMS-PDFT
   - 预计额外 2-4 个月

---

## 8. 资源需求

### 8.1 人力

**最小配置：**
- 1 名全职开发者，2.5-3 个月

**推荐配置：**
- 1 名主开发者（grid + 集成）
- 1 名辅助开发者（functional + 测试）
- 总时间：1.5-2 个月

### 8.2 外部依赖

**必需：**
- `libxc` (v5.0+)：DFT functional
- `libcint`：已有

**可选：**
- `libgrpp` 或 PySCF：Grid 生成
- `Eigen`：已有

### 8.3 计算资源

**开发阶段：**
- 小分子测试（H2, F2）：笔记本即可

**验证阶段：**
- 中等分子（C6H6）：需要多核服务器
- 大分子：可能需要 GPU

---

## 9. 验证计划

### 9.1 单元测试

| 组件 | 测试内容 | 通过标准 |
|------|----------|----------|
| Grid 生成器 | 权重归一化 | $\sum_g w_g = 1$ |
| AO 评估器 | 与解析值对比 | 相对误差 < 1e-10 |
| On-top functional | 与参考实现对比 | 能量差 < 1e-6 |

### 9.2 集成测试

**测试体系：**
1. **H2**：最简单，解析可解
2. **F2**：测试键断裂
3. **C2H2**：测试多中心
4. **C6H6**：测试共轭体系

**验证指标：**
- 能量守恒
- 电子数守恒
- 与 MC-PDFT 对比（如果可能）

### 9.3 性能测试

**目标：**
- H2：< 1 秒
- F2：< 5 秒
- C6H6：< 1 分钟

---

## 10. 发表策略

### 10.1 Stage 1A 完成后

**可发表内容：**
- 方法论文："VB-PDFT: Pair-Density Functional Theory for Valence Bond Wave Functions"
- 重点：
  - 非正交 VBSCF 参考态的 PDFT 扩展
  - 与 MC-PDFT 的对比
  - 键断裂曲线

**目标期刊：**
- J. Chem. Theory Comput.
- J. Phys. Chem. A

### 10.2 Stage 2 完成后

**可发表内容：**
- 应用论文："VB-PDFT for Transition States and Excited States"
- 解析梯度论文："Analytic Gradients for VB-PDFT"

---

## 11. 总结与建议

### 11.1 当前状态

**已完成：**
- ✅ 理论框架清晰
- ✅ RDM 构建器完成
- ✅ On-top pair density 构建器完成
- ✅ 物理轨道框架定义

**缺失：**
- ❌ Grid 生成器（关键）
- ❌ AO 格点评估器（关键）
- ❌ On-top functional 包装器
- ❌ 端到端集成

### 11.2 完成度评估

**总体完成度：** 35-40%

**关键路径完成度：**
- 理论 + 数据结构：90%
- 实空间积分：0%
- 端到端流程：0%

### 11.3 建议

**立即行动：**
1. **决定 Grid 策略**
   - 自己实现 vs 使用外部库
   - 如果自己实现，预留 3 周时间

2. **并行开发**
   - Grid 和 AO 评估器可以并行
   - 使用 mock 数据测试下游

3. **设定里程碑**
   - Milestone 1：Grid 生成器（3 周）
   - Milestone 2：AO 评估器（2 周）
   - Milestone 3：Functional 集成（1 周）
   - Milestone 4：端到端测试（2 周）

**预计时间线：**
- **乐观：** 2.5 个月（全职）
- **现实：** 4-6 个月（兼职）
- **保守：** 6-9 个月（考虑意外）

### 11.4 最终评估

**VB-PDFT 距离完善还差多远？**

**答案：** 距离 **Stage 1A（能量计算）** 还需要 **2.5-6 个月**，主要取决于：
1. Grid 生成器的实现策略
2. 开发投入程度（全职 vs 兼职）
3. 是否遇到数值稳定性问题

**好消息：**
- 理论基础扎实
- 核心 RDM 组件已完成
- 代码质量高，易于扩展

**挑战：**
- Grid 积分是全新模块，需要从零开始
- 数值稳定性需要仔细调试
- 性能优化可能耗时

**建议：** 如果这是优先项目，应该立即启动 Grid 生成器的开发。
