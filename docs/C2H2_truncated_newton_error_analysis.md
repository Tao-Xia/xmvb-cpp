# C2H2 截断牛顿法错误分析报告

## 错误现象

**文件：** `/export/home/xiatao/project/xmvb-cpp/test_molecule/C2H2.1938430.stderr.log`

**错误信息：**
```
terminate called after throwing an instance of 'std::runtime_error'
terminate called recursively
```

**作业信息：**
- Job ID: 1938430, 1938431
- 优化器：`nonredundant_truncated_newton`
- 线程数：32 (OMP_NUM_THREADS)
- 输入文件：`C2H2.xmi`

## 错误复现

### 多线程环境（原始错误）
```bash
cd /export/home/xiatao/project/xmvb-cpp
./build/src/xmvb-cpp.exe test_molecule/C2H2.xmi \
  --optimizer-backend nonredundant_truncated_newton \
  --max-iterations 2
```

**输出：**
```
terminate called recursively
terminate called recursively
...
terminate called after throwing an instance of 'std::runtime_error'
```

### 单线程环境（清晰错误信息）
```bash
OMP_NUM_THREADS=1 ./build/src/xmvb-cpp.exe test_molecule/C2H2.xmi \
  --optimizer-backend nonredundant_truncated_newton \
  --max-iterations 2
```

**输出：**
```
terminate called after throwing an instance of 'std::runtime_error'
  what():  matrix-form directional structure builder requires nullity == 0
```

## 根本原因

### 1. 错误位置

**文件：** `src/vb/scf/exact_orbital_second_order_operator.cpp`

**行号：** 3518-3522

```cpp
if (pair_evaluation.overlap_result.nullity != 0 ||
    pair_evaluation.overlap_result.overlap_determinant == 0.0) {
  throw std::runtime_error(
      "matrix-form directional structure builder requires nullity == 0");
}
```

**调用栈：**
```
#9  gather_directional_spin_block_local
#10 build_tiled_directional_structure_matrices
#11 gomp_thread_start (OpenMP 线程)
```

### 2. 什么是 nullity？

**定义：** `src/vb/matrices/determinant_types.hpp`

```cpp
/**
 * @brief Number of singular values below the linear-dependence threshold.
 *
 * This matches the `Nlt` quantity in the legacy implementation.
 */
int nullity = 0;
```

**物理含义：**
- `nullity` 是行列式对重叠矩阵的零空间维数
- `nullity = 0`：行列式对线性独立，重叠矩阵满秩
- `nullity > 0`：行列式对线性相关，重叠矩阵奇异

### 3. 为什么会出现 nullity > 0？

对于 C2H2 分子：
- 活性空间：4 个电子，4 个轨道
- 结构数：20 个 VB 结构
- 行列式数：36 个行列式

**可能原因：**

1. **轨道接近线性相关**
   - C2H2 是线性分子，对称性高
   - 某些轨道组合可能接近线性相关
   - 导致行列式重叠矩阵接近奇异

2. **数值精度问题**
   - 重叠矩阵的奇异值接近阈值
   - 被判定为 `nullity > 0`

3. **结构选择问题**
   - 某些 VB 结构在行列式空间中几乎重叠
   - 导致行列式对的重叠矩阵奇异

### 4. 为什么截断牛顿法需要 nullity == 0？

**截断牛顿法（Truncated Newton）** 需要计算 Hessian-vector product (HVP)：

$$
\mathbf{H} \mathbf{v} = \nabla^2 E \cdot \mathbf{v}
$$

在 `matrix-form directional structure builder` 中，需要计算方向导数：

$$
\frac{\partial S_{IJ}}{\partial \mathbf{x}} \cdot \mathbf{v}, \quad
\frac{\partial H_{IJ}}{\partial \mathbf{x}} \cdot \mathbf{v}
$$

这需要：
1. 计算行列式对的重叠矩阵逆：$(\mathbf{S}_{\text{det}})^{-1}$
2. 计算 cofactor 矩阵

**当 `nullity > 0` 时：**
- 重叠矩阵奇异，不可逆
- cofactor 计算失败
- 方向导数无法计算

因此代码抛出异常。

## 为什么多线程会导致 "terminate called recursively"？

### 问题分析

1. **OpenMP 并行区域中抛出异常**
   ```cpp
   #pragma omp parallel if(n_structures > 2)
   {
       // ... 多个线程并行执行
       gather_directional_spin_block_local(...);  // 抛出异常
   }
   ```

2. **多个线程同时抛出异常**
   - 线程 1 抛出异常 → 调用 `std::terminate`
   - 线程 2 也抛出异常 → 再次调用 `std::terminate`
   - ...
   - 导致递归调用 `terminate`

3. **异常信息被覆盖**
   - 第一个异常的 `what()` 信息被后续异常覆盖
   - 只看到 "terminate called recursively"

### 单线程环境的优势

```bash
OMP_NUM_THREADS=1
```

- 只有一个线程
- 异常信息清晰
- 便于调试

## 解决方案

### 方案 1：使用其他优化器（推荐）

**LBFGS 优化器**（不需要 Hessian）：
```bash
./build/src/xmvb-cpp.exe test_molecule/C2H2.xmi \
  --optimizer-backend nonredundant_lbfgspp \
  --max-iterations 2000
```

**验证：**
```bash
cd /export/home/xiatao/project/xmvb-cpp
./build/src/xmvb-cpp.exe test_molecule/C2H2.xmi \
  --optimizer-backend nonredundant_lbfgspp \
  --max-iterations 1
```

**结果：** 成功运行，无错误

### 方案 2：修复 nullity 问题

#### 2.1 增加数值容差

修改 `src/vb/matrices/determinant_overlap_resolver.cpp` 中的奇异值阈值：

```cpp
// 当前阈值（可能太严格）
const double singular_value_threshold = 1.0e-10;

// 放宽阈值
const double singular_value_threshold = 1.0e-12;
```

#### 2.2 使用伪逆代替逆矩阵

修改 `src/vb/scf/exact_orbital_second_order_operator.cpp:3518-3522`：

```cpp
// 原代码：
if (pair_evaluation.overlap_result.nullity != 0 ||
    pair_evaluation.overlap_result.overlap_determinant == 0.0) {
  throw std::runtime_error(
      "matrix-form directional structure builder requires nullity == 0");
}

// 修改为：使用伪逆
if (pair_evaluation.overlap_result.nullity != 0) {
  // 使用 SVD 伪逆处理奇异情况
  result = compute_directional_with_pseudoinverse(...);
  return result;
}
```

#### 2.3 跳过奇异的行列式对

```cpp
if (pair_evaluation.overlap_result.nullity != 0) {
  // 跳过这个行列式对，贡献为零
  result.delta_overlap_determinant = 0.0;
  result.delta_total_hamiltonian = 0.0;
  return result;
}
```

### 方案 3：改进轨道猜测

**问题：** 初始轨道可能导致线性相关

**解决：**
1. 使用更好的初始猜测（HF 轨道）
2. 添加轨道正交化步骤
3. 检查轨道重叠矩阵的条件数

### 方案 4：使用有限差分 HVP

截断牛顿法支持两种 HVP 模式：

```cpp
enum class NonredundantTruncatedNewtonHvpMode {
  FullFiniteDifference,           // 有限差分（不需要 nullity == 0）
  ExactContextDirectAction,       // 精确方向导数（需要 nullity == 0）
};
```

**修改命令行参数：**
```bash
./build/src/xmvb-cpp.exe test_molecule/C2H2.xmi \
  --optimizer-backend nonredundant_truncated_newton \
  --hvp-mode full_fd \
  --max-iterations 2000
```

（需要检查代码是否支持此参数）

## 临时解决方案（立即可用）

### 1. 修改提交脚本

编辑 `test_molecule/vbscf-cpp-tnhvp.sh`，将默认优化器改为 LBFGS：

```bash
# 第 53 行
optimizer_backend="${OPTIMIZER_BACKEND:-lbfgspp}"  # 原来是 truncated_newton
```

或者在提交时指定：

```bash
OPTIMIZER_BACKEND=lbfgspp sbatch -c 32 vbscf-cpp-tnhvp.sh C2H2.xmi
```

### 2. 添加错误处理

在 `vbscf-cpp-tnhvp.sh` 中添加：

```bash
# 第 136 行之前
set +e  # 允许命令失败
"${command[@]}" >"${stdout_log}" 2>"${stderr_log}"
exit_code=$?
set -e

if [[ $exit_code -ne 0 ]]; then
  echo "ERROR: xmvb-cpp failed with exit code $exit_code" >&2
  echo "Check logs:" >&2
  echo "  stdout: ${stdout_log}" >&2
  echo "  stderr: ${stderr_log}" >&2
  
  # 显示实际错误信息
  if [[ -f "${stderr_log}" ]]; then
    echo "Last 20 lines of stderr:" >&2
    tail -20 "${stderr_log}" >&2
  fi
  
  exit $exit_code
fi
```

## 长期解决方案

### 1. 实现鲁棒的 HVP 计算

- 自动检测 `nullity > 0` 的情况
- 自动切换到有限差分模式
- 或者使用伪逆处理奇异情况

### 2. 改进行列式对评估

- 在构造行列式对之前检查线性相关性
- 过滤掉接近奇异的行列式对
- 使用更稳定的数值方法

### 3. 添加诊断信息

在抛出异常前输出：
```cpp
if (pair_evaluation.overlap_result.nullity != 0) {
  std::cerr << "WARNING: Singular determinant pair detected\n";
  std::cerr << "  nullity = " << pair_evaluation.overlap_result.nullity << "\n";
  std::cerr << "  overlap_determinant = " 
            << pair_evaluation.overlap_result.overlap_determinant << "\n";
  std::cerr << "  left_structure = " << left_structure << "\n";
  std::cerr << "  right_structure = " << right_structure << "\n";
  
  throw std::runtime_error(...);
}
```

## 总结

### 问题本质

**截断牛顿法的 matrix-form directional structure builder 要求所有行列式对的重叠矩阵满秩（`nullity == 0`），但 C2H2 分子的某些行列式对存在线性相关性。**

### 立即行动

1. **使用 LBFGS 优化器**（最简单）
   ```bash
   --optimizer-backend nonredundant_lbfgspp
   ```

2. **单线程调试**（如果必须用截断牛顿法）
   ```bash
   OMP_NUM_THREADS=1
   ```

3. **修改提交脚本**
   - 将默认优化器改为 `lbfgspp`
   - 或者添加错误处理

### 后续工作

1. 调查为什么 C2H2 会出现 `nullity > 0`
2. 实现更鲁棒的 HVP 计算
3. 添加自动降级机制（精确 → 有限差分）
4. 改进错误信息（避免 "terminate called recursively"）

---

## 附录：相关代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| 错误抛出 | `exact_orbital_second_order_operator.cpp` | 3518-3522 |
| HVP 计算 | `exact_orbital_second_order_operator.cpp` | 3795-3944 |
| nullity 定义 | `determinant_types.hpp` | 行号待查 |
| 优化器选择 | `cpp_vb_scf_optimizer.cpp` | 待查 |
| 提交脚本 | `test_molecule/vbscf-cpp-tnhvp.sh` | 全文 |
