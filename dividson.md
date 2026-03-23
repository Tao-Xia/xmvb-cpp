# C++ / Eigen 广义 Davidson 对角化算法实现规范 (Implementation Spec)

## 1. 架构目标 (Objective)
实现一个基于 C++17 和 Eigen3 的块 Davidson 算法 (Block-Davidson Algorithm)，用于求解大规模稀疏/隐式广义特征值问题：
$$\mathbf{H} \mathbf{C} = \mathbf{S} \mathbf{C} \mathbf{\Lambda}$$
该求解器专为量子化学 (VBSCF/CI) 设计，针对对角线占优的矩阵，仅求解最低的 `n_roots` 个特征值和特征向量。矩阵 $\mathbf{H}$ 和 $\mathbf{S}$ 不会在内存中完整实例化，而是通过外部传入的 `Matrix-Vector Product (Sigma Vector)` 回调函数动态计算。

## 2. 接口定义 (Interface Definition)

请在一个独立的作用域或类中实现该求解器。核心接口设计如下：

```cpp
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <functional>

namespace xmvb::math {

// 矩阵-向量乘法回调函数定义
// 输入: V (当前搜索子空间矩阵, 大小为 dim x block_size)
// 输出: HV (H * V), SV (S * V)
using SigmaVectorOp = std::function<void(
    const Eigen::MatrixXd& V, 
    Eigen::MatrixXd& HV, 
    Eigen::MatrixXd& SV)>;

class DavidsonSolver {
public:
    struct Config {
        int n_roots = 1;                 // 求解的最低特征值个数
        int max_subspace_size = 40;      // 搜索子空间最大维度 (通常为 n_roots 的 10-20 倍)
        int max_iterations = 100;        // 最大迭代次数
        double tolerance = 1e-8;         // 残差收敛阈值
    };

    struct Result {
        Eigen::VectorXd eigenvalues;     // 大小: n_roots
        Eigen::MatrixXd eigenvectors;    // 大小: dim x n_roots
        bool converged;
        int iterations;
    };

    // 求解器主函数
    // dim: 宏观态空间总维度 K
    // diag_H: 哈密顿矩阵对角线 (用于预条件)
    // diag_S: 重叠矩阵对角线
    static Result solve(
        int dim,
        const Eigen::VectorXd& diag_H,
        const Eigen::VectorXd& diag_S,
        const SigmaVectorOp& compute_sigma,
        const Config& config = Config());
};

} // namespace xmvb::math
```

## 3. 核心算法流程 (Algorithm Steps)

请在 `solve` 函数内部实现以下流程。**必须预先分配好固定大小的工作空间矩阵 (Workspace)，在循环中使用 Eigen 的 `.leftCols()` 视图进行操作，严禁在迭代循环内部执行 `new`/`resize` 等堆内存分配。**

### Step 0: 初始化工作空间 (Initialization)
- 分配矩阵 `V`、`HV`、`SV`，大小均为 `dim x max_subspace_size`。
- 构建初始猜测向量：通常取 `diag_H` 中最小的 `n_roots` 个对角元对应的单位向量。
- 将这些单位向量放入 `V` 的前 `n_roots` 列。
- 调用 `compute_sigma` 计算前 `n_roots` 列的 `HV` 和 `SV`。
- 设定当前子空间大小 `m = n_roots`。

### Step 1: 投影到子空间 (Subspace Projection)
- 提取当前有效子空间：`V_curr = V.leftCols(m)`，`HV_curr = HV.leftCols(m)`，`SV_curr = SV.leftCols(m)`。
- 构建小矩阵 ($m \times m$):
  - $\mathbf{h}_{small} = \mathbf{V}_{curr}^T \mathbf{HV}_{curr}$
  - $\mathbf{s}_{small} = \mathbf{V}_{curr}^T \mathbf{SV}_{curr}$
  - *注意：使用 `.noalias()` 优化矩阵乘法。*

### Step 2: 求解子空间特征值问题 (Subspace Diagonalization)
- 对小矩阵使用 Eigen 原生求解器：`Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(h_small, s_small);`
- 提取最低的 `n_roots` 个特征值 $\lambda_k$ (Ritz values) 和对应的特征向量 $\mathbf{c}_k$ ($m \times n\_roots$)。

### Step 3: 计算 Ritz 向量与残差 (Ritz Vectors & Residuals)
- 近似特征向量 ($dim \times n\_roots$): $\mathbf{X} = \mathbf{V}_{curr} \mathbf{c}$
- 近似 $\mathbf{HX}$ 和 $\mathbf{SX}$: $\mathbf{HX} = \mathbf{HV}_{curr} \mathbf{c}$，$\mathbf{SX} = \mathbf{SV}_{curr} \mathbf{c}$
- 计算残差矩阵 ($dim \times n\_roots$): $\mathbf{R} = \mathbf{HX} - \mathbf{SX} \mathbf{\Lambda}$ (其中 $\mathbf{\Lambda}$ 是 $\lambda_k$ 的对角阵)。
- 检查收敛：如果 $\mathbf{R}$ 的每一列的 L2 范数 (Norm) 均小于 `config.tolerance`，则退出循环，返回成功。

### Step 4: 预条件与新搜索方向 (Preconditioning)
对残差矩阵 $\mathbf{R}$ 的每一列 $k$ (对应根 $k$) 和每一行 $i$，生成修正向量 $\mathbf{T}$：
$$ T_{ik} = \frac{R_{ik}}{\lambda_k S_{ii} - H_{ii}} $$
*注意：为了防止除零，如果分母绝对值 $< 10^{-5}$，则将分母强行置为 $10^{-5}$ (或取符号保留极小值)。*

### Step 5: 子空间折叠 / 重启 (Subspace Collapse / Restart)
- 如果当前子空间大小 $m + n\_roots > max\_subspace\_size$：
  - 将当前最优的近似向量 $\mathbf{X}$ 覆盖写入 $\mathbf{V}$ 的前 `n_roots` 列。
  - 同样将 $\mathbf{HX}$ 和 $\mathbf{SX}$ 覆盖写入 `HV` 和 `SV` 的前 `n_roots` 列。
  - 重置子空间大小 $m = n\_roots$。
  - 直接跳至 Step 7 (跳过当前步的正交化，因为下一轮将直接使用现有的 Ritz 向量)。

### Step 6: 正交化并扩充子空间 (Orthogonalization & Expansion)
- 将新增的修正向量 $\mathbf{T}$ ($dim \times n\_roots$) 对当前的 $\mathbf{V}_{curr}$ 进行 **Modified Gram-Schmidt (MGS)** 正交化，确保新加入的基向量之间，以及与原有基向量之间都严格正交 ($\mathbf{V}^T \mathbf{V} = \mathbf{I}$)。
- 在正交化过程中，剔除范数极小（线性相关）的向量。
- 将正交化后的新向量存入 $\mathbf{V}$ 从列索引 $m$ 开始的位置。
- 对新增的列调用 `compute_sigma` 获取它们对应的 `HV` 和 `SV`。
- 更新子空间大小：$m = m + \text{新加入的列数}$。

### Step 7: 循环迭代 (Iterate)
- 返回 Step 1，直到达到 `config.max_iterations`。

---

## 4. Eigen 性能与代码规范要求 (Eigen C++ Guidelines)

1. **零动态分配 (Zero Dynamic Allocation)**: 
   内部的投影小矩阵 `h_small`, `s_small` 等，应当复用外层预先分配好的最大 `max_subspace_size x max_subspace_size` 的缓存矩阵，通过 `block()` 或 `topLeftCorner()` 视图进行操作。
2. **延迟求值与别名 (Aliasing)**:
   在执行 `V_curr.transpose() * HV_curr` 等矩阵乘法赋值时，必须追加 `.noalias()`，避免 Eigen 产生临时拷贝。
   例: `h_small.topLeftCorner(m, m).noalias() = V.leftCols(m).transpose() * HV.leftCols(m);`
3. **MGS 正交化数值稳定性**:
   在向子空间追加新向量时，务必使用两遍 Modified Gram-Schmidt，或者对残差向量本身先内部正交化，再对子空间正交化，确保基向量的正交性达到机器精度。



> *"Here is the specification for a Generalized Davidson Solver using C++ and Eigen. Please implement the `DavidsonSolver` class exactly as described, paying special attention to the Eigen performance guidelines (Zero dynamic allocation in the loop, using `.leftCols()`, and `.noalias()`)."*
