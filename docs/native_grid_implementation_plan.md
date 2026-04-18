# Native Grid 实现方案（无外部依赖）

## 总体策略

**目标：** 实现高质量、高性能的分子积分格点生成器

**原则：**
1. 纯 C++ 实现，无 Python 依赖
2. 遵循 HPC_CPP_STYLE
3. 数值稳定性优先
4. 性能优化（OpenMP 并行）
5. 易于测试和验证

---

## Phase 1: Lebedev 角度格点（Week 1）

### 1.1 Lebedev 格点数据

**策略：** 硬编码常用阶数的 Lebedev 格点表

**文件：** `src/vb/pdft/lebedev_grid_data.hpp`

```cpp
#pragma once

#include <vector>
#include <Eigen/Core>

namespace xmvb::vb::pdft {

/**
 * @brief Single Lebedev quadrature point on unit sphere.
 */
struct LebedevPoint {
  double x;       ///< Cartesian x coordinate
  double y;       ///< Cartesian y coordinate  
  double z;       ///< Cartesian z coordinate
  double weight;  ///< Quadrature weight (sum to 4*pi)
};

/**
 * @brief Lebedev angular quadrature grid on unit sphere.
 *
 * Lebedev grids provide spherical quadrature rules that exactly
 * integrate spherical harmonics up to a given order. The grids
 * respect octahedral symmetry.
 */
class LebedevGrid {
public:
  /**
   * @brief Returns Lebedev grid for given number of points.
   *
   * Supported sizes: 6, 14, 26, 38, 50, 74, 86, 110, 146, 170,
   *                  194, 230, 266, 302, 350, 434, 590, 770, 974
   *
   * @param n_points Number of angular points (must be supported).
   * @return Vector of Lebedev points with weights summing to 4*pi.
   */
  static std::vector<LebedevPoint> get_grid(int n_points);

  /**
   * @brief Returns the largest supported grid size <= requested.
   */
  static int get_closest_supported_size(int requested_size);

  /**
   * @brief Returns all supported grid sizes.
   */
  static std::vector<int> get_supported_sizes();
};

}  // namespace xmvb::vb::pdft
```

### 1.2 Lebedev 数据生成

**方法：** 使用对称性生成规则

**文件：** `src/vb/pdft/lebedev_grid_data.cpp`

```cpp
namespace xmvb::vb::pdft {

namespace {

// Lebedev grid generation using octahedral symmetry.
// Based on Lebedev & Laikov, Doklady Mathematics (1999).

// Type A: 6 points along axes (±1, 0, 0), (0, ±1, 0), (0, 0, ±1)
void add_type_a_points(double weight, std::vector<LebedevPoint>* points) {
  points->push_back({1.0, 0.0, 0.0, weight});
  points->push_back({-1.0, 0.0, 0.0, weight});
  points->push_back({0.0, 1.0, 0.0, weight});
  points->push_back({0.0, -1.0, 0.0, weight});
  points->push_back({0.0, 0.0, 1.0, weight});
  points->push_back({0.0, 0.0, -1.0, weight});
}

// Type B: 12 points along face diagonals (±a, ±a, 0) and permutations
void add_type_b_points(double a, double weight, 
                       std::vector<LebedevPoint>* points) {
  const double coords[3][2] = {{a, 0.0}, {0.0, a}, {a, a}};
  for (int i = 0; i < 3; ++i) {
    for (int sx = -1; sx <= 1; sx += 2) {
      for (int sy = -1; sy <= 1; sy += 2) {
        double x = sx * coords[i][0];
        double y = sy * coords[i][1];
        double z = coords[i][2];
        // Add all permutations
        points->push_back({x, y, z, weight});
        points->push_back({x, z, y, weight});
        points->push_back({y, x, z, weight});
        points->push_back({y, z, x, weight});
        points->push_back({z, x, y, weight});
        points->push_back({z, y, x, weight});
      }
    }
  }
}

// Type C: 8 points along body diagonals (±a, ±a, ±a)
void add_type_c_points(double a, double weight,
                       std::vector<LebedevPoint>* points) {
  for (int sx = -1; sx <= 1; sx += 2) {
    for (int sy = -1; sy <= 1; sy += 2) {
      for (int sz = -1; sz <= 1; sz += 2) {
        points->push_back({sx * a, sy * a, sz * a, weight});
      }
    }
  }
}

// Lebedev-6 grid (order 3)
std::vector<LebedevPoint> generate_lebedev_6() {
  std::vector<LebedevPoint> points;
  add_type_a_points(4.0 * M_PI / 6.0, &points);
  return points;
}

// Lebedev-14 grid (order 5)
std::vector<LebedevPoint> generate_lebedev_14() {
  std::vector<LebedevPoint> points;
  add_type_a_points(0.06666666666666667 * 4.0 * M_PI, &points);
  add_type_c_points(1.0 / std::sqrt(3.0), 
                    0.07500000000000000 * 4.0 * M_PI, &points);
  return points;
}

// Lebedev-26 grid (order 7)
std::vector<LebedevPoint> generate_lebedev_26() {
  std::vector<LebedevPoint> points;
  add_type_a_points(0.04761904761904762 * 4.0 * M_PI, &points);
  add_type_b_points(1.0 / std::sqrt(2.0),
                    0.03809523809523810 * 4.0 * M_PI, &points);
  add_type_c_points(1.0 / std::sqrt(3.0),
                    0.03214285714285714 * 4.0 * M_PI, &points);
  return points;
}

// ... 继续实现其他阶数 (38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302)

// Lebedev-302 grid (order 29) - 常用于 DFT
std::vector<LebedevPoint> generate_lebedev_302() {
  // 完整参数见 Lebedev & Laikov 表格
  std::vector<LebedevPoint> points;
  // Type A
  add_type_a_points(0.008611732891199902 * 4.0 * M_PI, &points);
  // Type B
  add_type_b_points(1.0 / std::sqrt(2.0),
                    0.01976842567811548 * 4.0 * M_PI, &points);
  // Type C
  add_type_c_points(1.0 / std::sqrt(3.0),
                    0.02074999276256982 * 4.0 * M_PI, &points);
  // ... 更多 Type D, E, F 点
  return points;
}

}  // anonymous namespace

std::vector<LebedevPoint> LebedevGrid::get_grid(int n_points) {
  switch (n_points) {
    case 6: return generate_lebedev_6();
    case 14: return generate_lebedev_14();
    case 26: return generate_lebedev_26();
    // ... 其他阶数
    case 302: return generate_lebedev_302();
    default:
      throw std::invalid_argument(
          "Unsupported Lebedev grid size: " + std::to_string(n_points));
  }
}

std::vector<int> LebedevGrid::get_supported_sizes() {
  return {6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 
          194, 230, 266, 302, 350, 434, 590, 770, 974};
}

int LebedevGrid::get_closest_supported_size(int requested_size) {
  auto sizes = get_supported_sizes();
  auto it = std::lower_bound(sizes.begin(), sizes.end(), requested_size);
  if (it == sizes.end()) {
    return sizes.back();
  }
  return *it;
}

}  // namespace xmvb::vb::pdft
```

**数据来源：**
- Lebedev & Laikov, Doklady Mathematics (1999)
- 或者从 PySCF/Libint2 源码提取表格

---

## Phase 2: 径向格点（Week 1）

### 2.1 径向格点接口

**文件：** `src/vb/pdft/radial_grid.hpp`

```cpp
namespace xmvb::vb::pdft {

/**
 * @brief Radial quadrature point.
 */
struct RadialPoint {
  double r;       ///< Radial distance from nucleus
  double weight;  ///< Quadrature weight
};

/**
 * @brief Radial grid schemes for atomic integration.
 */
enum class RadialGridScheme {
  MuraKnowles,    ///< Mura-Knowles transformation (recommended)
  TreutlerAhlrichs, ///< Treutler-Ahlrichs (M4 mapping)
  BeckeLog,       ///< Becke logarithmic grid
};

/**
 * @brief Generates radial quadrature grids for atomic integration.
 *
 * The radial grid maps a finite interval [0, R_max] to the full
 * semi-infinite range [0, ∞) using various transformation schemes.
 */
class RadialGridBuilder {
public:
  /**
   * @brief Builds radial grid using specified scheme.
   *
   * @param n_points Number of radial points.
   * @param atomic_number Atomic number (for scaling).
   * @param scheme Radial transformation scheme.
   * @return Vector of radial points with weights.
   */
  static std::vector<RadialPoint> build(
      int n_points,
      int atomic_number,
      RadialGridScheme scheme = RadialGridScheme::MuraKnowles);
};

}  // namespace xmvb::vb::pdft
```

### 2.2 Mura-Knowles 变换实现

**文件：** `src/vb/pdft/radial_grid.cpp`

```cpp
namespace xmvb::vb::pdft {

namespace {

// Mura-Knowles transformation:
//   r(x) = R * x^2 / (1 - x)^2
//   dr/dx = 2 * R * x * (1 + x) / (1 - x)^3
// where x ∈ [0, 1) and R is the Bragg-Slater radius.

double bragg_slater_radius(int atomic_number) {
  // Bragg-Slater radii in Bohr
  static const double radii[] = {
    0.0,    // dummy for index 0
    0.35,   // H
    0.35,   // He
    1.45,   // Li
    1.05,   // Be
    0.85,   // B
    0.70,   // C
    0.65,   // N
    0.60,   // O
    0.50,   // F
    0.45,   // Ne
    // ... 继续到 Z=118
  };
  if (atomic_number < 1 || atomic_number > 118) {
    throw std::invalid_argument("Invalid atomic number");
  }
  return radii[atomic_number];
}

std::vector<RadialPoint> build_mura_knowles_grid(
    int n_points,
    int atomic_number) {
  const double R = bragg_slater_radius(atomic_number);
  
  std::vector<RadialPoint> points;
  points.reserve(n_points);
  
  // Use Gauss-Chebyshev quadrature on [0, 1]
  for (int i = 1; i <= n_points; ++i) {
    // Gauss-Chebyshev nodes: x_i = (1 + cos(π(2i-1)/(2n))) / 2
    const double theta = M_PI * (2.0 * i - 1.0) / (2.0 * n_points);
    const double x = 0.5 * (1.0 + std::cos(theta));
    
    // Mura-Knowles transformation
    const double one_minus_x = 1.0 - x;
    const double r = R * x * x / (one_minus_x * one_minus_x);
    
    // Jacobian: dr/dx
    const double dr_dx = 2.0 * R * x * (1.0 + x) / 
                         (one_minus_x * one_minus_x * one_minus_x);
    
    // Gauss-Chebyshev weight: π / n * sin(θ)
    const double gauss_weight = M_PI / n_points * std::sin(theta);
    
    // Combined weight
    const double weight = gauss_weight * dr_dx;
    
    points.push_back({r, weight});
  }
  
  return points;
}

}  // anonymous namespace

std::vector<RadialPoint> RadialGridBuilder::build(
    int n_points,
    int atomic_number,
    RadialGridScheme scheme) {
  switch (scheme) {
    case RadialGridScheme::MuraKnowles:
      return build_mura_knowles_grid(n_points, atomic_number);
    // case RadialGridScheme::TreutlerAhlrichs:
    //   return build_treutler_ahlrichs_grid(n_points, atomic_number);
    default:
      throw std::invalid_argument("Unsupported radial grid scheme");
  }
}

}  // namespace xmvb::vb::pdft
```

---

## Phase 3: Becke 分区权重（Week 2）

### 3.1 Becke 权重接口

**文件：** `src/vb/pdft/becke_partition.hpp`

```cpp
namespace xmvb::vb::pdft {

/**
 * @brief Becke partition weights for molecular integration.
 *
 * The Becke partition ensures smooth integration across atomic
 * regions by constructing weights w_A(r) that sum to 1 at each
 * point and smoothly transition between atoms.
 *
 * Reference: Becke, J. Chem. Phys. 88, 2547 (1988).
 */
class BeckePartition {
public:
  /**
   * @brief Computes Becke partition weights for all atoms.
   *
   * @param grid_points Grid coordinates (n_points x 3, column-major).
   * @param atomic_coords Atomic coordinates (n_atoms x 3, column-major).
   * @param atomic_numbers Atomic numbers (length n_atoms).
   * @param exponent Becke partition exponent (typically 3).
   * @return Partition weights (n_points x n_atoms, column-major).
   *         Each row sums to 1.
   */
  static Eigen::MatrixXd compute_weights(
      const Eigen::MatrixXd& grid_points,
      const Eigen::MatrixXd& atomic_coords,
      const Eigen::VectorXi& atomic_numbers,
      int exponent = 3);

private:
  // Becke cell function: s(mu) = (1 - mu^2)^3 for |mu| <= 1
  static double cell_function(double mu);
  
  // Atomic size adjustment factor
  static double size_adjustment_factor(int Z_A, int Z_B);
};

}  // namespace xmvb::vb::pdft
```

### 3.2 Becke 权重实现

**文件：** `src/vb/pdft/becke_partition.cpp`

```cpp
namespace xmvb::vb::pdft {

namespace {

// Becke cell function: s(mu) = (1 - mu^2)^3 for |mu| <= 1, else 0
double cell_function(double mu) {
  if (std::abs(mu) >= 1.0) {
    return 0.0;
  }
  const double one_minus_mu2 = 1.0 - mu * mu;
  return one_minus_mu2 * one_minus_mu2 * one_minus_mu2;
}

// Iterated cell function: s_k(mu) = s(s(...s(mu)...))
double iterated_cell_function(double mu, int k) {
  double result = mu;
  for (int i = 0; i < k; ++i) {
    result = cell_function(result);
  }
  return result;
}

// Atomic size adjustment (Becke's formula)
double size_adjustment_factor(int Z_A, int Z_B) {
  // Bragg-Slater radii ratio
  const double chi_A = bragg_slater_radius(Z_A);
  const double chi_B = bragg_slater_radius(Z_B);
  const double u_AB = (chi_A - chi_B) / (chi_A + chi_B);
  
  // Adjustment parameter (Becke recommends a = 0.25)
  const double a = 0.25;
  return u_AB / (u_AB * u_AB - 1.0 + 1.0e-12) * 
         (1.0 - u_AB * u_AB) * a;
}

}  // anonymous namespace

Eigen::MatrixXd BeckePartition::compute_weights(
    const Eigen::MatrixXd& grid_points,
    const Eigen::MatrixXd& atomic_coords,
    const Eigen::VectorXi& atomic_numbers,
    int exponent) {
  const int n_points = static_cast<int>(grid_points.rows());
  const int n_atoms = static_cast<int>(atomic_coords.rows());
  
  // Partition weights: w_A(r) = P_A(r) / sum_B P_B(r)
  Eigen::MatrixXd weights(n_points, n_atoms);
  
  #pragma omp parallel for schedule(static)
  for (int g = 0; g < n_points; ++g) {
    const Eigen::Vector3d r = grid_points.row(g);
    
    // Compute P_A(r) for each atom A
    Eigen::VectorXd P(n_atoms);
    for (int A = 0; A < n_atoms; ++A) {
      const Eigen::Vector3d R_A = atomic_coords.row(A);
      const double r_A = (r - R_A).norm();
      
      // P_A(r) = prod_{B != A} s_k(mu_AB(r))
      double product = 1.0;
      for (int B = 0; B < n_atoms; ++B) {
        if (B == A) continue;
        
        const Eigen::Vector3d R_B = atomic_coords.row(B);
        const double r_B = (r - R_B).norm();
        const double R_AB = (R_A - R_B).norm();
        
        // Confocal elliptical coordinate: mu_AB = (r_A - r_B) / R_AB
        double mu_AB = (r_A - r_B) / (R_AB + 1.0e-12);
        
        // Apply size adjustment
        const double nu_AB = size_adjustment_factor(
            atomic_numbers(A), atomic_numbers(B));
        mu_AB = mu_AB + nu_AB * (1.0 - mu_AB * mu_AB);
        
        // Apply iterated cell function
        const double s_k = iterated_cell_function(mu_AB, exponent);
        product *= 0.5 * (1.0 - s_k);
      }
      P(A) = product;
    }
    
    // Normalize: w_A(r) = P_A(r) / sum_B P_B(r)
    const double sum_P = P.sum();
    if (sum_P > 1.0e-15) {
      weights.row(g) = P / sum_P;
    } else {
      // Degenerate case: assign equal weights
      weights.row(g).setConstant(1.0 / n_atoms);
    }
  }
  
  return weights;
}

}  // namespace xmvb::vb::pdft
```

**数值稳定性要点：**
1. 避免除零：`R_AB + 1.0e-12`
2. 处理退化情况：`sum_P < 1.0e-15`
3. OpenMP 并行化

---

## Phase 4: 分子格点构建器（Week 3）

### 4.1 完整的分子格点构建

**文件：** `src/vb/pdft/molecular_grid_builder.cpp`

```cpp
namespace xmvb::vb::pdft {

MolecularGrid MolecularGridBuilder::build(
    const Eigen::MatrixXd& atomic_coords,
    const Eigen::VectorXd& atomic_charges) const {
  const int n_atoms = static_cast<int>(atomic_coords.rows());
  
  // 1. Generate atomic grids
  std::vector<AtomicGrid> atomic_grids;
  atomic_grids.reserve(n_atoms);
  
  for (int atom = 0; atom < n_atoms; ++atom) {
    const int Z = static_cast<int>(atomic_charges(atom));
    const Eigen::Vector3d center = atomic_coords.row(atom);
    
    // Build radial grid
    auto radial_points = RadialGridBuilder::build(
        config_.radial_points, Z, RadialGridScheme::MuraKnowles);
    
    // Build angular grid
    auto angular_points = LebedevGrid::get_grid(config_.angular_points);
    
    // Combine radial and angular
    AtomicGrid atomic_grid = combine_radial_angular(
        radial_points, angular_points, center);
    
    atomic_grids.push_back(std::move(atomic_grid));
  }
  
  // 2. Concatenate all atomic grids
  MolecularGrid raw_grid = concatenate_atomic_grids(atomic_grids);
  
  // 3. Apply Becke partition
  Eigen::VectorXi atomic_numbers = atomic_charges.cast<int>();
  Eigen::MatrixXd becke_weights = BeckePartition::compute_weights(
      raw_grid.points, atomic_coords, atomic_numbers, config_.becke_exponent);
  
  // 4. Multiply atomic weights by Becke weights
  apply_becke_partition(&raw_grid, becke_weights, atomic_grids);
  
  // 5. Optional: apply pruning
  if (config_.pruning_scheme != "none") {
    raw_grid = apply_pruning(raw_grid, config_.pruning_scheme);
  }
  
  return raw_grid;
}

}  // namespace xmvb::vb::pdft
```

---

## 测试与验证（Week 3）

### 测试用例

**文件：** `src/tools/test_molecular_grid.cpp`

```cpp
// Test 1: Lebedev 权重归一化
void test_lebedev_normalization() {
  auto grid = LebedevGrid::get_grid(302);
  double weight_sum = 0.0;
  for (const auto& point : grid) {
    weight_sum += point.weight;
  }
  // Lebedev weights sum to 4*pi
  EXPECT_NEAR(weight_sum, 4.0 * M_PI, 1.0e-10);
}

// Test 2: Becke 权重归一化
void test_becke_partition_h2() {
  Eigen::MatrixXd coords(2, 3);
  coords << 0.0, 0.0, 0.0,
            0.0, 0.0, 1.4;  // H2 at 1.4 Bohr
  Eigen::VectorXi charges(2);
  charges << 1, 1;
  
  Eigen::MatrixXd test_points(100, 3);
  // ... 生成测试点
  
  Eigen::MatrixXd weights = BeckePartition::compute_weights(
      test_points, coords, charges);
  
  // 每个点的权重和应该为 1
  for (int g = 0; g < 100; ++g) {
    double row_sum = weights.row(g).sum();
    EXPECT_NEAR(row_sum, 1.0, 1.0e-10);
  }
}

// Test 3: 完整分子格点
void test_molecular_grid_h2() {
  MolecularGridConfig config;
  config.radial_points = 50;
  config.angular_points = 302;
  
  MolecularGridBuilder builder(config);
  
  Eigen::MatrixXd coords(2, 3);
  coords << 0.0, 0.0, 0.0,
            0.0, 0.0, 1.4;
  Eigen::VectorXd charges(2);
  charges << 1.0, 1.0;
  
  MolecularGrid grid = builder.build(coords, charges);
  
  // 检查格点数量
  EXPECT_GT(grid.n_points(), 1000);
  
  // 检查权重归一化（积分常函数 = 体积）
  double weight_sum = grid.weights.sum();
  // 对于无限空间，这个测试需要调整
}
```

---

## 性能优化

### OpenMP 并行化

```cpp
// Becke 权重计算（已并行）
#pragma omp parallel for schedule(static)
for (int g = 0; g < n_points; ++g) {
  // 计算权重
}

// 原子格点生成（可并行）
#pragma omp parallel for
for (int atom = 0; atom < n_atoms; ++atom) {
  atomic_grids[atom] = build_atomic_grid(atom);
}
```

### 内存优化

```cpp
// 预分配内存
grid.points.resize(total_points, 3);
grid.weights.resize(total_points);

// 避免临时拷贝
grid.points.block(offset, 0, n_atom_points, 3) = atomic_grid.points;
```

---

## 时间线

| Week | 任务 | 交付物 |
|------|------|--------|
| 1 | Lebedev + 径向格点 | `lebedev_grid_data.cpp`, `radial_grid.cpp` |
| 2 | Becke 分区 | `becke_partition.cpp` |
| 3 | 集成 + 测试 | `molecular_grid_builder.cpp`, 测试通过 |

**总计：** 3 周，~1500 行高质量代码

---

## 下一步行动

**本周（Week 1）：**
1. 实现 Lebedev 格点数据（6, 14, 26, 302）
2. 实现 Mura-Knowles 径向格点
3. 单元测试：权重归一化

**Week 2：**
1. 实现 Becke 分区权重
2. 测试 H2 分子的权重归一化

**Week 3：**
1. 集成完整的分子格点构建器
2. 测试 H2, F2, C2H2
3. 性能优化（OpenMP）

**3 周后：** 完成 Grid 生成器，进入 AO 评估器阶段
