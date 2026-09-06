# Davidson 结构矩阵算子设计

## 核心思想

Davidson 需要矩阵自由的 $H_{\mathrm{VB}} x$ 和 $M_{\mathrm{VB}} x$。不应物化 $N\times N$ 稠密矩阵，而应复用正向构建的分块基础设施。

## 矩阵元素结构

每个 VB 结构 $I$ 有一个 `StructureCoefficientBlock`：

- $\mathrm{alpha\_support}_I$：对该结构有贡献的唯一 alpha 自旋串 ID 列表
- $\mathrm{beta\_support}_I$：唯一 beta 自旋串 ID 列表  
- $C_I[a,b]$：局部系数矩阵（`local_coefficients`）

结构矩阵元素为：

$$
H_{\mathrm{VB}}[I,J] = \sum_{\substack{a,a'\in\mathrm{supp}(I,J) \\ b,b'\in\mathrm{supp}(I,J)}} C_I[a,b] \cdot H_{\mathrm{spin}}(a,b;a',b') \cdot C_J[a',b']
$$

其中 $H_{\mathrm{spin}}$ 存储在 `ForwardSpinPairTileProvider` 中，按唯一自旋串对进行分块。

## 代码组织

### 1. 抽象接口 (`structure_matrix_operator.hpp`)

```cpp
class StructureMatrixOperator {
public:
  virtual ~StructureMatrixOperator() = default;
  virtual int dimension() const = 0;
  virtual void apply_hamiltonian(const double* x, double* y) const = 0;
  virtual void apply_overlap(const double* x, double* y) const = 0;
  virtual Eigen::VectorXd precondition_diag() const;  // diag(H) or diag(H - shift*S)
};
```

### 2. 稠密算子（LAPACK 路径）

```cpp
class DenseStructureMatrixOperator : public StructureMatrixOperator {
  // 包装已物化的 H 和 S。apply_* 使用 BLAS dsymv。
  std::vector<double> H_, S_;
  int dim_;
};
```

**用法**：先正向构建→物化 H,S→包装为 Dense→传给 `solve()`（LAPACK dsygvd）。

### 3. 分块算子（Davidson 路径）

```cpp
class TiledStructureMatrixOperator : public StructureMatrixOperator {
  // 引用正向构建基础设施。apply_* 实现矩阵自由的 H*x、S*x。
  const std::vector<StructureCoefficientBlock>& blocks_;
  ForwardSpinPairTileProvider<...>& alpha_provider_;
  ForwardSpinPairTileProvider<...>& beta_provider_;
  const std::vector<double>& det_overlap_cache_;
  int n_structures_, n_unique_alpha_, n_unique_beta_;
  bool close_shell_same_spin_;
};
```

**用法**：获取系数块和分块提供者→构造 Tiled→传给 `solve_davidson()`。

每个结构对的 `apply_hamiltonian` 伪代码：
```
对于每个结构 I（可能并行化）：
  y[I] = 0
  对于每个结构 J（其中 supp(I) ∩ supp(J) 非空——仅相邻结构）：
    H_tile = alpha_provider.entries(alpha_I, alpha_J)
          ⊗ beta_provider.entries(beta_I, beta_J)
    y[I] += C_I · H_tile · C_J^T · x[J]
  加上对角线贡献
```

### 4. 本征求解器 (`generalized_eigensolver.hpp`)

现有内容保留：
```cpp
class GeneralizedEigensolver {
  // 稠密 LAPACK（现有，不变）
  GeneralizedEigenResult solve(
      const std::vector<double>& H, const std::vector<double>& S, int dim) const;

  // 矩阵自由 Davidson（新增，接受任意算子）
  GeneralizedEigenResult solve_davidson(
      const StructureMatrixOperator& op, int n_roots) const;
};
```

### 5. 构建位置

| 文件 | 内容 |
|------|---------|
| `vb/matrices/structure_matrix_operator.hpp` | 抽象接口 + Dense + Tiled 声明 |
| `vb/matrices/structure_matrix_operator.cpp` | Dense + Tiled 实现 |
| `core/linear_algebra/generalized_eigensolver.cpp` | `solve_davidson(operator, n_roots)` |

## 关键设计决策

1. **接口使用原始指针而非 Eigen**：为与 LAPACK 约定和现有 `std::vector<double>` 存储保持一致。
2. **Tiled 算子持有引用**而非副本：它引用正向构建基础设施，生命周期由调用者管理。
3. **预条件器**：`precondition_diag()` 返回近似于 $\mathrm{diag}(H - \lambda S)^{-1}$ 的向量，所有算子实现均相同。
4. **两种方法均可切换**：使用 Tiled 算子时启用 Davidson，否则回退到 LAPACK。
