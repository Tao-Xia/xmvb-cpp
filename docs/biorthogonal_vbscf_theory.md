# 双正交价键自洽场方法的数学理论

## 1. 引言与动机

### 1.1 传统非正交 VBSCF 的问题

在传统的价键自洽场（VBSCF）方法中，价键波函数表示为价键结构的线性组合：

$$
|\Psi\rangle = \sum_{I=1}^{N_s} c_I |\Phi_I\rangle
$$

其中 $|\Phi_I\rangle$ 是价键结构，由行列式的线性组合构成：

$$
|\Phi_I\rangle = \sum_{J} T_{JI} |D_J\rangle
$$

由于轨道非正交，结构之间也非正交，需要求解广义本征值问题：

$$
\mathbf{H} \mathbf{c} = E \mathbf{S} \mathbf{c}
$$

其中：
- $H_{IJ} = \langle \Phi_I | \hat{H} | \Phi_J \rangle$ 是结构哈密顿矩阵
- $S_{IJ} = \langle \Phi_I | \Phi_J \rangle$ 是结构重叠矩阵

**问题：** 当选择结构子空间（selected structures）时，截断后的 $\mathbf{S}$ 可能不正定，导致数值不稳定。

### 1.2 双正交方法的核心思想

双正交方法引入**左对偶轨道** $\{\tilde{\phi}_i\}$ 满足双正交条件：

$$
\langle \tilde{\phi}_i | \phi_j \rangle = \delta_{ij}
$$

这样可以构造**双正交行列式基**，使得行列式之间满足：

$$
\langle \tilde{D}_I | D_J \rangle = \delta_{IJ}
$$

从而避免重叠矩阵的病态问题。

---

## 2. 双正交轨道框架

### 2.1 对偶轨道的构造

给定右轨道 $\mathbf{C} = [\phi_1, \phi_2, \ldots, \phi_{n_a}]$（活性空间轨道），其在原子轨道基下的重叠矩阵为：

$$
\mathbf{X} = \mathbf{C}^T \mathbf{S} \mathbf{C}
$$

其中 $\mathbf{S}$ 是原子轨道重叠矩阵。

左对偶轨道定义为：

$$
\tilde{\mathbf{C}} = \mathbf{C} \mathbf{X}^{-1}
$$

**验证双正交性：**

$$
\tilde{\mathbf{C}}^T \mathbf{S} \mathbf{C} = (\mathbf{C} \mathbf{X}^{-1})^T \mathbf{S} \mathbf{C} = \mathbf{X}^{-T} \mathbf{X} = \mathbf{I}
$$

### 2.2 双正交行列式

对于右行列式 $|D_I\rangle$，其占据轨道为 $\{\phi_{i_1}, \phi_{i_2}, \ldots, \phi_{i_{n_e}}\}$。

对应的左对偶行列式 $|\tilde{D}_I\rangle$ 占据对偶轨道 $\{\tilde{\phi}_{i_1}, \tilde{\phi}_{i_2}, \ldots, \tilde{\phi}_{i_{n_e}}\}$。

**关键性质：**

$$
\langle \tilde{D}_I | D_J \rangle = \delta_{IJ}
$$

这是因为行列式重叠可以表示为轨道重叠的行列式，而轨道满足双正交条件。

---

## 3. 双正交哈密顿矩阵元

### 3.1 一般形式

双正交哈密顿矩阵元定义为：

$$
h_{IJ}^{\text{bi}} = \langle \tilde{D}_I | \hat{H} | D_J \rangle
$$

其中哈密顿算符为：

$$
\hat{H} = \sum_{i} h_i + \sum_{i<j} g_{ij}
$$

### 3.2 双正交 Slater-Condon 规则

根据左右行列式之间的激发关系，矩阵元可以分类计算。

#### 3.2.1 零激发（对角元）

当 $|\tilde{D}_I\rangle$ 和 $|D_I\rangle$ 占据相同轨道集合 $\{i_1, i_2, \ldots, i_{n_e}\}$ 时：

$$
h_{II}^{\text{bi}} = \sum_{k=1}^{n_e} h_{i_k i_k}^{LR} + \sum_{k<l} \left[ g_{i_k i_k i_l i_l}^{\text{bi}} - g_{i_k i_l i_l i_k}^{\text{bi}} \right]
$$

其中：
- $h_{ij}^{LR} = \langle \tilde{\phi}_i | \hat{h} | \phi_j \rangle$ 是左右一电子积分
- $g_{ijkl}^{\text{bi}} = \langle \tilde{\phi}_i \phi_k | \hat{g} | \phi_j \phi_l \rangle$ 是双正交双电子积分

#### 3.2.2 单激发

假设 $|\tilde{D}_I\rangle$ 相对于 $|D_J\rangle$ 有单个轨道激发 $i \to a$（相位因子为 $\sigma$）：

$$
h_{IJ}^{\text{bi}} = \sigma \left[ h_{ai}^{LR} + \sum_{k \in \text{occ}_J, k \neq i} \left( g_{iakk}^{\text{bi}} - g_{ikkа}^{\text{bi}} \right) \right]
$$

其中求和遍历 $|D_J\rangle$ 中除 $i$ 外的所有占据轨道。

#### 3.2.3 双激发

对于双激发 $i,j \to a,b$（相位因子为 $\sigma$）：

$$
h_{IJ}^{\text{bi}} = \sigma \left[ g_{iajb}^{\text{bi}} - g_{ibja}^{\text{bi}} \right]
$$

#### 3.2.4 高阶激发

激发秩 $> 2$ 时，矩阵元为零：

$$
h_{IJ}^{\text{bi}} = 0 \quad \text{if } \text{rank}(\text{excitation}) > 2
$$

### 3.3 双正交积分的计算

#### 3.3.1 左右一电子积分

$$
h_{ij}^{LR} = \langle \tilde{\phi}_i | \hat{h} | \phi_j \rangle = \sum_{k} (\mathbf{X}^{-1})_{ki} h_{kj}^{RR}
$$

其中 $h_{kj}^{RR} = \langle \phi_k | \hat{h} | \phi_j \rangle$ 是右右一电子积分。

**代码实现：** `biorthogonal_orbital_integrals.cpp` 第 38-45 行
```cpp
left_right_one_electron = 
    left_dual_from_right_transform * right_right_one_electron;
```

#### 3.3.2 双正交双电子积分

$$
g_{ijkl}^{\text{bi}} = \langle \tilde{\phi}_i \phi_k | \hat{g} | \phi_j \phi_l \rangle = \sum_{m} (\mathbf{X}^{-1})_{mi} g_{mjkl}^{RR}
$$

其中 $g_{mjkl}^{RR} = \langle \phi_m \phi_k | \hat{g} | \phi_j \phi_l \rangle$ 是右右双电子积分。

**代码实现：** `biorthogonal_orbital_integrals.hpp` 第 60-70 行
```cpp
inline double evaluate_biorthogonal_two_electron_integral(
    int left_index, int right_first_index,
    int right_second_index, int right_third_index,
    const BiorthogonalOrbitalIntegrals& orbital_integrals,
    const ActiveSpaceTwoElectronView& right_right_two_electron_view) {
  // 实现 X^{-1} 变换
}
```

---

## 4. 精确选择结构子空间方法

### 4.1 问题设定

给定完整结构空间 $\{|\Phi_1\rangle, \ldots, |\Phi_{N_s}\rangle\}$，选择子集 $\mathcal{I} \subset \{1, \ldots, N_s\}$。

目标：在选择的结构子空间中求解本征值问题，同时保持物理正确性。

### 4.2 完整行列式空间的必要性

每个结构可以展开为行列式的线性组合：

$$
|\Phi_I\rangle = \sum_{J=1}^{N_d} T_{JI} |D_J\rangle
$$

其中 $N_d$ 是完整行列式空间的维数。

**关键矩阵：**

选择结构对应的展开矩阵 $\mathbf{T}_{\text{sel}} \in \mathbb{R}^{N_d \times N_{\text{sel}}}$，其第 $k$ 列对应第 $I_k$ 个选择结构的展开系数。

### 4.3 精确投影公式

#### 4.3.1 行列式空间的作用

定义行列式重叠矩阵（单位矩阵，因为双正交）：

$$
\mathbf{S}_{\text{det}} = \mathbf{I}_{N_d}
$$

定义双正交行列式哈密顿矩阵：

$$
(\mathbf{h}_{\text{det}}^{\text{bi}})_{IJ} = \langle \tilde{D}_I | \hat{H} | D_J \rangle
$$

#### 4.3.2 投影到选择结构

计算选择结构上的作用：

$$
\mathbf{U}_{\text{sel}} = \mathbf{S}_{\text{det}} \mathbf{T}_{\text{sel}} = \mathbf{T}_{\text{sel}}
$$

$$
\mathbf{Y}_{\text{sel}} = \mathbf{h}_{\text{det}}^{\text{bi}} \mathbf{T}_{\text{sel}}
$$

**物理意义：**
- $\mathbf{U}_{\text{sel}}$ 的第 $k$ 列是 $|\Phi_{I_k}\rangle$ 在行列式基下的展开
- $\mathbf{Y}_{\text{sel}}$ 的第 $k$ 列是 $\hat{H}|\Phi_{I_k}\rangle$ 在行列式基下的展开

#### 4.3.3 物理结构矩阵

投影回选择结构空间：

$$
\mathbf{S}_{\text{str}} = \mathbf{U}_{\text{sel}}^T \mathbf{T}_{\text{sel}} = \mathbf{T}_{\text{sel}}^T \mathbf{T}_{\text{sel}}
$$

$$
\mathbf{H}_{\text{str}} = \mathbf{U}_{\text{sel}}^T \mathbf{Y}_{\text{sel}} = \mathbf{T}_{\text{sel}}^T \mathbf{h}_{\text{det}}^{\text{bi}} \mathbf{T}_{\text{sel}}
$$

**关键性质：**

$$
(\mathbf{S}_{\text{str}})_{kl} = \langle \Phi_{I_k} | \Phi_{I_l} \rangle
$$

$$
(\mathbf{H}_{\text{str}})_{kl} = \langle \Phi_{I_k} | \hat{H} | \Phi_{I_l} \rangle
$$

这些是**精确的物理矩阵元**，不是近似！

#### 4.3.4 求解本征值问题

$$
\mathbf{H}_{\text{str}} \mathbf{C} = \mathbf{S}_{\text{str}} \mathbf{C} \mathbf{E}
$$

其中 $\mathbf{C} \in \mathbb{R}^{N_{\text{sel}} \times N_{\text{sel}}}$ 是本征向量矩阵。

### 4.4 为什么必须用完整行列式空间？

**数学原因：**

如果只在选择结构对应的行列式子空间中工作，会丢失结构之间的交叉项。

例如，考虑两个结构：
$$
|\Phi_1\rangle = |D_1\rangle + 0.5|D_2\rangle
$$
$$
|\Phi_2\rangle = 0.3|D_1\rangle + |D_3\rangle
$$

它们的重叠为：
$$
\langle \Phi_1 | \Phi_2 \rangle = \langle D_1|D_1\rangle \cdot 0.3 + \langle D_2|D_3\rangle \cdot 0.5 \cdot 1
$$

如果只保留 $\{|D_1\rangle, |D_2\rangle, |D_3\rangle\}$，可以正确计算。

但如果有第三个结构：
$$
|\Phi_3\rangle = 0.2|D_4\rangle + 0.8|D_5\rangle
$$

而 $|D_4\rangle$ 和 $|D_5\rangle$ 不在前两个结构的行列式支撑中，那么：
$$
\langle \Phi_1 | \Phi_3 \rangle = 0
$$

这是**精确的零**，不是近似。

**结论：** 必须保留完整行列式空间 $\{|D_1\rangle, \ldots, |D_{N_d}\rangle\}$ 才能正确计算所有选择结构之间的矩阵元。

---

## 5. 计算流程与复杂度分析

### 5.1 算法流程

**输入：**
- 完整结构数据：$N_s$ 个结构，$N_d$ 个行列式
- 选择结构索引：$\mathcal{I} = \{I_1, \ldots, I_{N_{\text{sel}}}\}$
- 活性空间轨道：$n_a$ 个轨道

**步骤：**

1. **构造双正交轨道框架** $O(n_a^3)$
   - 计算 $\mathbf{X} = \mathbf{C}^T \mathbf{S} \mathbf{C}$
   - 求逆 $\mathbf{X}^{-1}$
   - 构造 $\tilde{\mathbf{C}} = \mathbf{C} \mathbf{X}^{-1}$

2. **构造选择结构展开** $O(N_d \cdot N_{\text{sel}})$
   - 提取 $\mathbf{T}_{\text{sel}}$

3. **计算行列式作用** $O(N_d^2 \cdot N_{\text{sel}})$ **← 瓶颈**
   - 对每个选择结构 $k = 1, \ldots, N_{\text{sel}}$：
     - 计算 $\mathbf{u}_k = \mathbf{T}_{\text{sel}}[:, k]$（第 $k$ 列）
     - 计算 $\mathbf{y}_k = \mathbf{h}_{\text{det}}^{\text{bi}} \mathbf{u}_k$
   
   这需要计算 $N_d \times N_d$ 的双正交行列式哈密顿矩阵与 $N_{\text{sel}}$ 个向量的乘积。

4. **投影回结构空间** $O(N_d \cdot N_{\text{sel}}^2)$
   - $\mathbf{S}_{\text{str}} = \mathbf{T}_{\text{sel}}^T \mathbf{T}_{\text{sel}}$
   - $\mathbf{H}_{\text{str}} = \mathbf{T}_{\text{sel}}^T \mathbf{Y}_{\text{sel}}$

5. **求解广义本征值问题** $O(N_{\text{sel}}^3)$
   - $\mathbf{H}_{\text{str}} \mathbf{C} = \mathbf{S}_{\text{str}} \mathbf{C} \mathbf{E}$

### 5.2 复杂度分析

**主要瓶颈：** 步骤 3 的行列式作用计算。

#### 5.2.1 朴素方法

直接构造完整的 $\mathbf{h}_{\text{det}}^{\text{bi}} \in \mathbb{R}^{N_d \times N_d}$：

- 时间复杂度：$O(N_d^2 \cdot T_{\text{elem}})$
- 空间复杂度：$O(N_d^2)$

其中 $T_{\text{elem}}$ 是单个矩阵元的计算时间（Slater-Condon 规则）。

**问题：** 对于大体系，$N_d$ 可能达到 $10^4 \sim 10^6$，$N_d^2$ 不可行。

#### 5.2.2 稀疏方法（当前实现）

利用 Slater-Condon 规则，只有激发秩 $\leq 2$ 的行列式对有非零矩阵元。

**稀疏度估计：**

对于 $n_a$ 个活性轨道，$n_e$ 个活性电子：
- 零激发：$N_d$ 个（对角元）
- 单激发：每个行列式约 $n_e \cdot (n_a - n_e)$ 个非零元
- 双激发：每个行列式约 $\binom{n_e}{2} \cdot \binom{n_a - n_e}{2}$ 个非零元

总非零元数：
$$
N_{\text{nnz}} \approx N_d \cdot \left[ 1 + n_e(n_a - n_e) + \binom{n_e}{2}\binom{n_a - n_e}{2} \right]
$$

对于 $n_a = 6, n_e = 3$：
$$
N_{\text{nnz}} \approx N_d \cdot [1 + 9 + 27] = 37 N_d
$$

**当前实现：**

不显式构造 $\mathbf{h}_{\text{det}}^{\text{bi}}$，而是：
1. 将行列式按 unique alpha/beta spin strings 分组
2. 使用 tile-based 方法计算 $\mathbf{Y}_{\text{sel}} = \mathbf{h}_{\text{det}}^{\text{bi}} \mathbf{T}_{\text{sel}}$

**代码位置：** `biorthogonal_exact_selected_structure.cpp` 第 751-841 行

**复杂度：**
- 时间：$O(N_{\text{nnz}} \cdot N_{\text{sel}})$
- 空间：$O(N_d \cdot N_{\text{sel}})$（只存储 $\mathbf{Y}_{\text{sel}}$）

### 5.3 实际性能瓶颈

#### 5.3.1 双电子积分查询

每个非零矩阵元需要查询 $O(n_e)$ 个双电子积分，每次查询需要：
1. 通过 $\mathbf{X}^{-1}$ 变换：$O(n_a)$
2. 查询 $g_{ijkl}^{RR}$：$O(1)$（假设已压缩存储）

总时间：$O(N_{\text{nnz}} \cdot N_{\text{sel}} \cdot n_e \cdot n_a)$

#### 5.3.2 Tile cache miss

当前实现使用 LRU cache 缓存 unique spin pair tiles，但：
- 默认只缓存 4 个 tiles
- 访问模式可能导致频繁 cache miss

**代码位置：** `biorthogonal_spin_pair_tiles.cpp` 第 30-41 行

#### 5.3.3 并行效率

当前使用 OpenMP 并行化选择结构的循环，但：
- 每个线程需要独立的 tile provider（无法共享 cache）
- 动态调度可能导致负载不均

**代码位置：** `biorthogonal_exact_selected_structure.cpp` 第 790-830 行

---

## 6. 与传统方法的对比

### 6.1 传统非正交 VBSCF

**优点：**
- 直接在结构空间工作，复杂度 $O(N_s^2)$
- 不需要行列式展开

**缺点：**
- 选择结构时，$\mathbf{S}_{\text{str}}$ 可能不正定
- 数值不稳定

### 6.2 双正交 VBSCF

**优点：**
- 行列式正交，$\mathbf{S}_{\text{det}} = \mathbf{I}$
- 选择结构的 $\mathbf{S}_{\text{str}}$ 总是正定的（只要 $\mathbf{T}_{\text{sel}}$ 列满秩）
- 数值稳定

**缺点：**
- 必须在完整行列式空间工作，复杂度 $O(N_d^2 \cdot N_{\text{sel}})$
- 计算量大

### 6.3 适用场景

- **小体系**（$N_d < 1000$）：双正交方法可行
- **大体系**（$N_d > 10000$）：需要进一步优化或近似方法

---

## 7. 可能的优化方向

### 7.1 数学层面

#### 7.1.1 近似双正交方法

**思路：** 只在选择结构的行列式支撑上构造双正交基。

**问题：** 可能丢失结构间的交叉项，需要误差估计。

#### 7.1.2 分层双正交

**思路：** 
1. 先在粗粒度结构空间求解
2. 再在精细行列式空间修正

**挑战：** 如何定义合理的分层。

### 7.2 实现层面

#### 7.2.1 预计算与缓存

- 预计算所有非零行列式对的激发秩
- 缓存常用的双电子积分变换结果
- 增大 tile cache size

#### 7.2.2 批量计算

- 批量查询双电子积分，利用 SIMD
- 批量计算矩阵元，减少函数调用开销

#### 7.2.3 改进并行策略

- 使用共享的 tile cache（需要线程安全机制）
- 改进负载均衡策略
- 考虑 GPU 加速

---

## 8. 总结

### 8.1 核心数学要点

1. **双正交条件：** $\langle \tilde{\phi}_i | \phi_j \rangle = \delta_{ij}$ 导致 $\langle \tilde{D}_I | D_J \rangle = \delta_{IJ}$

2. **双正交矩阵元：** $h_{IJ}^{\text{bi}} = \langle \tilde{D}_I | \hat{H} | D_J \rangle$ 通过修改的 Slater-Condon 规则计算

3. **精确投影：** 必须在完整行列式空间中计算 $\mathbf{Y}_{\text{sel}} = \mathbf{h}_{\text{det}}^{\text{bi}} \mathbf{T}_{\text{sel}}$，然后投影回结构空间

4. **物理正确性：** 最终的 $\mathbf{H}_{\text{str}}$ 和 $\mathbf{S}_{\text{str}}$ 是精确的物理矩阵，不是近似

### 8.2 计算瓶颈

1. **行列式空间维数：** $N_d$ 随活性空间指数增长
2. **双电子积分变换：** 每个矩阵元需要多次 $\mathbf{X}^{-1}$ 变换
3. **Cache 效率：** Tile-based 方法的 cache miss rate 较高

### 8.3 优化策略

1. **短期：** 增大 cache、批量计算、改进并行
2. **中期：** 预计算激发秩、优化内存访问模式
3. **长期：** 探索近似双正交方法、GPU 加速

---

## 参考文献

1. 代码实现：`src/vb/biorthogonal_vbscf/`
2. README：`src/vb/biorthogonal_vbscf/README.md`
3. 相关论文：（待补充）

---

## 附录：关键代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| 双正交轨道框架 | `biorthogonal_orbital_frame.cpp` | 全文 |
| 双正交积分 | `biorthogonal_orbital_integrals.cpp` | 38-70 |
| Slater-Condon 规则 | `biorthogonal_determinant_hamiltonian.cpp` | 330-472 |
| 精确选择结构 | `biorthogonal_exact_selected_structure.cpp` | 845-1010 |
| Tile-based 计算 | `biorthogonal_structure_hamiltonian_builder.cpp` | 66-225 |
| Spin pair tiles | `biorthogonal_spin_pair_tiles.cpp` | 全文 |
