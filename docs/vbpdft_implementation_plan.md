# VB-PDFT 实现计划（遵循 HPC_CPP_STYLE）

## 总体策略

**原则：**
1. 使用 `Eigen::MatrixXd` 和 `Eigen::VectorXd`，避免 `std::vector<double>` 表示矩阵
2. 列主序存储（column-major）
3. 清晰的数学注释
4. 避免重复代码，提取共享逻辑
5. 无调试打印在热路径

**开发顺序：** 自底向上，每个模块独立测试

---

## Phase 1: Grid 生成器（Week 1-3）

### 1.1 数据结构设计

**文件：** `src/vb/pdft/molecular_grid.hpp`

```cpp
namespace xmvb::vb::pdft {

/**
 * @brief Atom-centered integration grid with Becke partitioning.
 *
 * The grid stores quadrature points and weights for numerical integration
 * of density functionals. Each point carries:
 * - spatial coordinates r_g
 * - integration weight w_g (including Becke partition)
 * - atom assignment for load balancing
 */
struct MolecularGrid {
  /// Grid point coordinates in Cartesian space.
  /// Column-major: n_points x 3 (x, y, z).
  Eigen::MatrixXd points;

  /// Integration weights including Becke partition.
  /// Length: n_points.
  Eigen::VectorXd weights;

  /// Atom index for each grid point (for load balancing).
  /// Length: n_points.
  std::vector<int> atom_assignments;

  /// Number of grid points.
  int n_points() const { return static_cast<int>(points.rows()); }
};

}  // namespace xmvb::vb::pdft
```

### 1.2 Grid 构建器接口

**文件：** `src/vb/pdft/molecular_grid_builder.hpp`

```cpp
namespace xmvb::vb::pdft {

/**
 * @brief Configuration for molecular grid generation.
 */
struct MolecularGridConfig {
  /// Radial grid size per atom (e.g., 50, 75, 100).
  int radial_points = 75;

  /// Angular grid size (Lebedev order, e.g., 302, 590, 974).
  int angular_points = 302;

  /// Becke partition exponent (typically 3).
  int becke_exponent = 3;

  /// Pruning scheme: "none", "sg1", "nwchem".
  std::string pruning_scheme = "sg1";
};

/**
 * @brief Builds atom-centered integration grids with Becke partitioning.
 *
 * The builder generates radial grids using Mura-Knowles transformation
 * and angular grids using Lebedev quadrature, then applies Becke
 * partition weights to ensure smooth integration across atomic regions.
 */
class MolecularGridBuilder {
public:
  explicit MolecularGridBuilder(MolecularGridConfig config);

  /**
   * @brief Builds the molecular grid from atomic coordinates and charges.
   *
   * @param atomic_coords Column-major n_atoms x 3 matrix.
   * @param atomic_charges Length n_atoms vector.
   * @return Molecular grid with Becke-partitioned weights.
   */
  MolecularGrid build(
      const Eigen::MatrixXd& atomic_coords,
      const Eigen::VectorXd& atomic_charges) const;

private:
  MolecularGridConfig config_;
};

}  // namespace xmvb::vb::pdft
```

### 1.3 实现细节

**文件：** `src/vb/pdft/molecular_grid_builder.cpp`

**关键函数：**
1. `build_radial_grid()`: Mura-Knowles 变换
2. `build_angular_grid()`: Lebedev 格点
3. `compute_becke_weights()`: Becke 分区权重
4. `apply_pruning()`: 格点剪枝

**数学注释示例：**
```cpp
// Becke partition weight for atom A at point r:
//   w_A(r) = P_A(r) / sum_B P_B(r)
// where P_A(r) = prod_{B != A} s_k(mu_AB(r))
// and mu_AB(r) = (r_A - r_B) / |R_A - R_B|
```

---

## Phase 2: AO 格点评估器（Week 4-5）

### 2.1 数据结构

**文件：** `src/vb/pdft/ao_grid_values.hpp`

```cpp
namespace xmvb::vb::pdft {

/**
 * @brief AO basis function values evaluated on a grid.
 *
 * Stores chi_mu(r_g) for all basis functions mu and grid points g.
 */
struct AoGridValues {
  /// AO values: column-major n_points x n_basis_functions.
  /// Entry (g, mu) = chi_mu(r_g).
  Eigen::MatrixXd values;

  /// Optional: AO gradient values for GGA functionals.
  /// Column-major n_points x (3 * n_basis_functions).
  /// Entry (g, 3*mu + d) = d chi_mu / d x_d at r_g.
  Eigen::MatrixXd gradients;

  int n_points() const { return static_cast<int>(values.rows()); }
  int n_basis_functions() const { return static_cast<int>(values.cols()); }
};

}  // namespace xmvb::vb::pdft
```

### 2.2 评估器接口

**文件：** `src/vb/pdft/libcint_ao_grid_evaluator.hpp`

```cpp
namespace xmvb::vb::pdft {

/**
 * @brief Evaluates AO basis functions on grid points using libcint.
 */
class LibcintAoGridEvaluator {
public:
  /**
   * @brief Constructs evaluator from libcint input.
   */
  explicit LibcintAoGridEvaluator(const LibcintInput& libcint_input);

  /**
   * @brief Evaluates AO values at grid points.
   *
   * @param grid_points Column-major n_points x 3 matrix.
   * @return AO values at each grid point.
   */
  AoGridValues evaluate_values(const Eigen::MatrixXd& grid_points) const;

  /**
   * @brief Evaluates AO values and gradients (for GGA).
   */
  AoGridValues evaluate_values_and_gradients(
      const Eigen::MatrixXd& grid_points) const;

private:
  LibcintInput libcint_input_;
  int n_basis_functions_;
};

}  // namespace xmvb::vb::pdft
```

---

## Phase 3: 密度评估器（Week 6）

### 3.1 实空间密度构建

**文件：** `src/vb/pdft/real_space_density_builder.hpp`

```cpp
namespace xmvb::vb::pdft {

/**
 * @brief Real-space densities evaluated on a grid.
 */
struct RealSpaceDensities {
  /// Total electron density rho(r_g).
  /// Length: n_points.
  Eigen::VectorXd rho;

  /// On-top pair density Pi(r_g).
  /// Length: n_points.
  Eigen::VectorXd pi;

  int n_points() const { return static_cast<int>(rho.size()); }
};

/**
 * @brief Builds real-space densities from RDMs and AO values.
 *
 * Evaluates:
 *   rho(r) = sum_{mu,nu} gamma_{mu,nu} chi_mu(r) chi_nu(r)
 *   Pi(r) = (1/2) sum_{mu,nu,lambda,kappa} Gamma_{mu,nu,lambda,kappa}
 *           chi_mu(r) chi_nu(r) chi_lambda(r) chi_kappa(r)
 */
class RealSpaceDensityBuilder {
public:
  /**
   * @brief Builds densities from physical 1-RDM and on-top context.
   *
   * @param ao_density_matrix Physical AO 1-RDM (n_basis x n_basis).
   * @param on_top_context On-top pair density contraction context.
   * @param ao_values AO values at grid points.
   * @return Real-space densities at each grid point.
   */
  RealSpaceDensities build(
      const Eigen::MatrixXd& ao_density_matrix,
      const SelectedStateExactPhysicalOnTopPairDensityContext& on_top_context,
      const AoGridValues& ao_values) const;
};

}  // namespace xmvb::vb::pdft
```

### 3.2 实现要点

**关键优化：**
1. 使用 `Eigen::Map` 避免拷贝
2. 批量矩阵乘法：`rho = diag(chi * gamma * chi^T)`
3. On-top 使用已有的逐点评估接口

---

## Phase 4: On-top Functional（Week 7）

### 4.1 Translated 自旋密度

**文件：** `src/vb/pdft/translated_spin_density.hpp`

```cpp
namespace xmvb::vb::pdft {

/**
 * @brief Translated spin densities for on-top functionals.
 */
struct TranslatedSpinDensity {
  /// Translated alpha density: rho * (1 + zeta_t) / 2.
  Eigen::VectorXd rho_alpha;

  /// Translated beta density: rho * (1 - zeta_t) / 2.
  Eigen::VectorXd rho_beta;

  /// On-top ratio: R = 4 * Pi / rho^2.
  Eigen::VectorXd on_top_ratio;

  /// Translated spin polarization: zeta_t = sqrt(max(0, 1 - R)).
  Eigen::VectorXd zeta_translated;
};

/**
 * @brief Computes translated spin densities from rho and Pi.
 *
 * Implements the translation:
 *   R(r) = 4 * Pi(r) / rho(r)^2
 *   zeta_t(r) = sqrt(max(0, 1 - R(r)))
 *   rho_alpha(r) = rho(r) * (1 + zeta_t(r)) / 2
 *   rho_beta(r) = rho(r) * (1 - zeta_t(r)) / 2
 *
 * Numerical stability: applies density threshold to avoid division by zero.
 */
TranslatedSpinDensity compute_translated_spin_density(
    const Eigen::VectorXd& rho,
    const Eigen::VectorXd& pi,
    double density_threshold = 1.0e-12);

}  // namespace xmvb::vb::pdft
```

### 4.2 Libxc 包装器

**文件：** `src/vb/pdft/libxc_functional.hpp`

```cpp
namespace xmvb::vb::pdft {

/**
 * @brief Wrapper for libxc exchange-correlation functionals.
 */
class LibxcFunctional {
public:
  /**
   * @brief Constructs functional from libxc identifier.
   *
   * @param functional_id Libxc functional ID (e.g., XC_LDA_X, XC_LDA_C_VWN).
   */
  explicit LibxcFunctional(int functional_id);

  ~LibxcFunctional();

  /**
   * @brief Evaluates XC energy density at grid points.
   *
   * @param rho_alpha Alpha spin density at each point.
   * @param rho_beta Beta spin density at each point.
   * @return XC energy density eps_xc(r) at each point.
   */
  Eigen::VectorXd evaluate_energy_density(
      const Eigen::VectorXd& rho_alpha,
      const Eigen::VectorXd& rho_beta) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xmvb::vb::pdft
```

---

## Phase 5: 端到端集成（Week 8-9）

### 5.1 VB-PDFT 能量评估器

**文件：** `src/vb/pdft/vb_pdft_energy_evaluator.hpp`

```cpp
namespace xmvb::vb::pdft {

/**
 * @brief Configuration for VB-PDFT energy evaluation.
 */
struct VbPdftConfig {
  /// Grid configuration.
  MolecularGridConfig grid_config;

  /// Libxc functional ID for on-top functional.
  int functional_id = 1;  // XC_LDA_X

  /// Density threshold for numerical stability.
  double density_threshold = 1.0e-12;
};

/**
 * @brief VB-PDFT energy evaluation result.
 */
struct VbPdftEnergyResult {
  /// Nuclear repulsion energy.
  double nuclear_repulsion_energy = 0.0;

  /// One-electron energy: Tr(h * gamma).
  double one_electron_energy = 0.0;

  /// Classical Coulomb energy: (1/2) Tr(gamma * gamma * J).
  double coulomb_energy = 0.0;

  /// On-top functional energy: integral of eps_ot[rho, Pi].
  double on_top_energy = 0.0;

  /// Total VB-PDFT energy.
  double total_energy = 0.0;

  /// Number of electrons from density integration.
  double integrated_electron_count = 0.0;
};

/**
 * @brief Evaluates VB-PDFT energy for a selected VBSCF state.
 *
 * Implements:
 *   E_n^{VB-PDFT} = V_nn + E_one[gamma] + J[rho] + E_ot[rho, Pi]
 */
class VbPdftEnergyEvaluator {
public:
  explicit VbPdftEnergyEvaluator(VbPdftConfig config);

  /**
   * @brief Evaluates VB-PDFT energy from VBSCF result.
   *
   * @param input VBSCF input with molecular geometry.
   * @param vbscf_result Converged VBSCF result.
   * @param state_index Selected state index.
   * @return VB-PDFT energy components.
   */
  VbPdftEnergyResult evaluate(
      const CppVbInput& input,
      const CppVbScfResult& vbscf_result,
      int state_index) const;

private:
  VbPdftConfig config_;
};

}  // namespace xmvb::vb::pdft
```

### 5.2 实现流程

```cpp
VbPdftEnergyResult VbPdftEnergyEvaluator::evaluate(...) const {
  // 1. Build molecular grid
  MolecularGridBuilder grid_builder(config_.grid_config);
  MolecularGrid grid = grid_builder.build(atomic_coords, atomic_charges);

  // 2. Evaluate AO values on grid
  LibcintAoGridEvaluator ao_evaluator(input.libcint_input);
  AoGridValues ao_values = ao_evaluator.evaluate_values(grid.points);

  // 3. Build physical 1-RDM
  SelectedStateExactPhysicalOneRdmBuilder rdm_builder;
  auto rdm_result = rdm_builder.build(input, state_index);

  // 4. Build on-top pair density context
  SelectedStateExactPhysicalOnTopPairDensityBuilder on_top_builder;
  auto on_top_context = on_top_builder.build(input, state_index);

  // 5. Evaluate real-space densities
  RealSpaceDensityBuilder density_builder;
  RealSpaceDensities densities = density_builder.build(
      rdm_result.physical_ao_one_rdm, on_top_context, ao_values);

  // 6. Compute translated spin densities
  TranslatedSpinDensity spin_density = compute_translated_spin_density(
      densities.rho, densities.pi, config_.density_threshold);

  // 7. Evaluate on-top functional
  LibxcFunctional functional(config_.functional_id);
  Eigen::VectorXd eps_xc = functional.evaluate_energy_density(
      spin_density.rho_alpha, spin_density.rho_beta);

  // 8. Integrate on-top energy
  double on_top_energy = grid.weights.dot(eps_xc);

  // 9. Assemble total energy
  VbPdftEnergyResult result;
  result.on_top_energy = on_top_energy;
  result.total_energy = nuclear_repulsion + one_electron + coulomb + on_top_energy;
  return result;
}
```

---

## Phase 6: 测试与验证（Week 10）

### 6.1 单元测试

**文件：** `src/tools/test_vb_pdft_components.cpp`

```cpp
// Test 1: Grid weight normalization
void test_grid_weights() {
  MolecularGridBuilder builder(default_config);
  MolecularGrid grid = builder.build(h2_coords, h2_charges);
  double weight_sum = grid.weights.sum();
  assert(std::abs(weight_sum - 1.0) < 1.0e-10);
}

// Test 2: AO value accuracy
void test_ao_values() {
  // Compare with analytical values for H2
  LibcintAoGridEvaluator evaluator(h2_libcint_input);
  AoGridValues ao_values = evaluator.evaluate_values(test_points);
  // Check against reference
}

// Test 3: Density integration
void test_density_integration() {
  // Verify: integral of rho(r) = N_electrons
  double integrated_electrons = (densities.rho.array() * grid.weights.array()).sum();
  assert(std::abs(integrated_electrons - expected_n_electrons) < 1.0e-6);
}
```

### 6.2 集成测试

**测试体系：**
1. H2 @ R=0.74 Å
2. F2 @ R=1.41 Å
3. C2H2（线性）

**验证指标：**
- 能量守恒
- 电子数守恒
- 与参考值对比（如果有）

---

## 代码风格检查清单

### ✅ 遵循 HPC_CPP_STYLE

- [ ] 所有矩阵使用 `Eigen::MatrixXd`（列主序）
- [ ] 所有向量使用 `Eigen::VectorXd`
- [ ] `std::vector<T>` 仅用于索引列表、稀疏数据
- [ ] 无 `std::cout` / `std::cerr` 在热路径
- [ ] 无环境变量开关
- [ ] 共享逻辑提取到头文件
- [ ] 数学注释清晰
- [ ] 无重复代码

### 示例：正确的矩阵使用

```cpp
// ✅ 正确
Eigen::MatrixXd ao_density_matrix(n_basis, n_basis);
Eigen::VectorXd grid_weights(n_points);

// ❌ 错误
std::vector<double> ao_density_matrix(n_basis * n_basis);  // 不要用于矩阵
std::vector<double> grid_weights(n_points);  // 不要用于向量
```

### 示例：正确的注释

```cpp
// ✅ 正确：解释数学含义
// Becke partition weight for atom A at point r:
//   w_A(r) = P_A(r) / sum_B P_B(r)
// where P_A(r) = prod_{B != A} s_k(mu_AB(r))

// ❌ 错误：重复语法
// Loop over atoms
for (int atom = 0; atom < n_atoms; ++atom) { ... }
```

---

## 时间线总结

| Week | 任务 | 交付物 |
|------|------|--------|
| 1-3 | Grid 生成器 | `molecular_grid_builder.hpp/cpp` |
| 4-5 | AO 评估器 | `libcint_ao_grid_evaluator.hpp/cpp` |
| 6 | 密度构建器 | `real_space_density_builder.hpp/cpp` |
| 7 | On-top Functional | `libxc_functional.hpp/cpp` |
| 8-9 | 端到端集成 | `vb_pdft_energy_evaluator.hpp/cpp` |
| 10 | 测试验证 | `test_vb_pdft_components.cpp` |

**总计：** 10 周（全职）或 20 周（兼职）

---

## 下一步行动

1. **本周：** 实现 `MolecularGrid` 数据结构
2. **Week 1：** 实现 Lebedev 格点生成
3. **Week 2：** 实现 Becke 分区权重
4. **Week 3：** 集成和测试 Grid 生成器

**第一个里程碑：** 3 周后完成 Grid 生成器，能够为 H2 生成正确的格点。
