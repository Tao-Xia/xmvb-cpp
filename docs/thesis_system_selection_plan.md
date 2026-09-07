# 毕业论文数值实验体系与数据计划

## 1. 目标

毕业论文里的数值部分不应该只是“列出很多体系”，而应该让每个体系都对应一个明确论点。对当前工作，建议把实验目标压缩为四类：

1. 解析 HVP / 非冗余 TN 路线在小体系上是正确的。
2. 在代表性的 closed-shell sparse HAO 体系上，TNHVP 相对一阶基线确实能更快收敛。
3. 在 open-shell sparse 体系上，算法的几何/图表选择会影响鲁棒性，因此需要用难体系和生产体系同时验证。
4. 在更大的 deck 上，矩阵自由 HVP 与分块收缩的工程路线确实有可接受的总时间和扩展性。

因此，体系选择应追求“少而硬”，而不是“多而散”。

## 2. 选择原则

### 2.1 正文主表优先使用可直接命名的分子

`240`、`241`、`10698` 这类内部 deck 编号适合放补充材料或脚注，不适合做正文主角。正文主表优先使用：

- `F2`
- `C6H6` / benzene
- `MnF2`
- `TiCl`
- `FeCl2`

如果必须使用内部 deck，则应同时写明它代表的分子或至少写明它是“internal benchmark deck”。

### 2.2 F2 只做正确性，不做性能主结论

`F2` 活性空间太小，难以体现：

- outer-response 近似的误差影响；
- exact-ctx 内部图表策略差异；
- unique-spin / block contraction 的实际收益；
- TNHVP 相对一阶方法的明显 wall-time 改善。

因此 `F2` 只适合做：

- 梯度/HVP 正确性；
- 小体系收敛 sanity check；
- 附录里的有限差分验证。

### 2.3 正文需要同时覆盖 closed-shell / open-shell 与 HAO / OEO

当前代码路径中，数值行为最容易受以下因素影响：

- closed-shell vs open-shell；
- HAO sparse chart vs OEO chart；
- 小体系 vs 较大 deck；
- benign 体系 vs difficult 体系。

所以正文体系不能只选一种类型。

## 3. 建议的正文主体系

下面这 6 个体系足够支撑正文。若时间非常紧，可以先做前 5 个。

| 体系 | 输入文件 | 类型 | 主要用途 | 优先级 |
| --- | --- | --- | --- | --- |
| `F2` | `src/test_molecule/F2.xmi` | closed-shell, HAO, 2e/2o, 很小 | 正确性/有限差分/小体系 sanity check | 必做 |
| `C6H6` (benzene) | `test/241_VBSCF.xmi` | closed-shell, HAO, 6e/6o, 12 atoms, 120 AO, 175 structures, 400 dets | 正文 closed-shell sparse HAO 主 benchmark；收敛与 wall-time 主表 | 必做 |
| `MnF2` | `test/MnF2.xmi` | open-shell, HAO, 9e/8o, sextet, 124 AO, 216 structures, 224 dets | 几何/图表敏感难体系；验证 open-shell sparse 情况 | 必做 |
| `TiCl` | `test/TiCl.xmi` | open-shell, OEO, 5e/7o, quartet | 小型 OEO 迁移性检查；避免论文只在 HAO 上成立 | 建议做 |
| `FeCl2` | `test/FeCl2.xmi` | open-shell, OEO, 10e/8o, quintet, 136 AO, 420 structures, 448 dets | 生产级 OEO 体系；证明默认策略没有拖坏较大活性空间过渡金属体系 | 必做 |
| `10698_VBSCF` | `test/10698_VBSCF.xmi` | larger closed-shell HAO deck, 22 atoms, 210 AO, 6e/6o, 175 structures, 400 dets | 较大 deck 的性能/阶段耗时/扩展性主 benchmark | 建议做 |

## 4. 建议放入补充材料的体系

这些体系很有价值，但不一定要进入正文主表。

| 体系 | 输入文件 | 用途 |
| --- | --- | --- |
| `240` internal deck | `test/240_tnhvp.xmi` | 历史输入，仅用于物理严格稀疏坐标回归 |
| `FeCl` | `test/FeCl.xmi` | 小型 open-shell OEO 补充例子；可作为 TiCl 的旁证 |
| `C6H6.full` | `src/test_molecule/C6H6.full.xmi` | full-structure benzene exact deck；适合讨论 same-spin / unique-string / exact contraction 工程收益 |
| `10698_RI` | `test/10698_RI.xmi` | 如果论文还要讨论 RI / low-rank 或 unique-spin pair 的工程优化，可单独放补充材料 |

## 5. 建议论文里的表格设计

建议不要把所有数据堆在一个大表里，而是按论点拆成 4 张表。

### 表 1：体系摘要表

每个正文体系给出：

- 分子名
- 输入文件
- orbital type (`HAO` / `OEO`)
- spin multiplicity
- basis set
- active space (`nae` / `nao`)
- AO 数
- selected structures 数
- expanded determinants 数

这张表用于告诉读者：后面的 benchmark 覆盖了哪些物理/组合复杂度。

### 表 2：优化收敛主表

建议在 `6526Y`、固定线程数（建议 32 线程）下，对每个正文体系比较：

- `nonredundant_truncated_newton`
- `nonredundant_lbfgspp`

每行给出：

- final energy
- final projected gradient norm
- iterations
- SCF wall time
- 是否收敛

这张表是正文最重要的一张表，因为它直接支持“TNHVP 是否有数值价值”。

### 表 3：HVP/阶段耗时表

只在 2--3 个代表体系上给出阶段耗时，建议：

- `241_VBSCF`
- `FeCl2`
- `10698_VBSCF`

每行可记录：

- total HVP apply wall time
- active-space integrals
- outer-response
- orbital pullback / fixed-upstream pullback
- active 2e contraction

这张表用于支撑“矩阵自由 HVP 的成本主要花在哪、哪些工程优化真正有意义”。

### 表 4：策略/图表补充表

只放补充材料，建议体系：

- `240_tnhvp`
- `241_VBSCF`
- `MnF2`

比较的开关只保留最少必要项：

- default policy
- physical strict-sparse chart finite-difference consistency

记录：

- first-step energy drop
- total iterations
- SCF wall time
- 是否触发大量 rejected/rescue 步

这张表不是为了正文宣称“某个 chart 普遍更优”，而是为了证明：

1. 图表/accepted-point 模型会影响 TN 轨迹；
2. 默认策略必须靠多体系验证决定。

## 6. 最小可行数据集

如果时间非常紧，先做下面这 5 个体系的主数据：

1. `F2`
2. `241_VBSCF` (benzene)
3. `MnF2`
4. `FeCl2`
5. `10698_VBSCF`

其中：

- `F2` 负责正确性；
- `241_VBSCF` 负责 closed-shell sparse HAO；
- `MnF2` 负责 hard open-shell sparse HAO；
- `FeCl2` 负责生产级 open-shell OEO；
- `10698_VBSCF` 负责较大 deck 性能。

如果还能再补一个，就加 `TiCl`，让 OEO 的 open-shell 小体系也有一个清楚例子。

## 7. 不建议现在投入正文主表的数据

### 7.1 不建议只做 F2 + 一个大体系

这样会缺少：

- closed-shell vs open-shell 对照；
- HAO vs OEO 对照；
- benign vs difficult 对照。

### 7.2 不建议把太多内部 deck 编号直接放正文

`240`、`10698` 这类编号读者无法建立直觉。除非它们承载的是“工程 benchmark deck”这一角色，否则正文最好用化学名。

### 7.3 不建议把所有实验都混在一个章节主表

正确性、收敛、阶段耗时、图表策略是不同问题。混成一个大表会让论文很难读，也会削弱论证力度。

## 8. 建议的运行矩阵

### 8.1 正文主表

对每个正文体系，至少跑：

1. `nonredundant_truncated_newton`
2. `nonredundant_lbfgspp`

运行环境统一为：

- `sbatch`
- partition `6526Y`
- 固定 `OMP_NUM_THREADS=32`
- `OPENBLAS_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`

### 8.2 正确性/附录

对 `F2`，补：

1. 小步有限差分或已有检查程序的 HVP/梯度一致性；
2. TNHVP 与 LBFGS 的最终能量一致性。

### 8.3 补充策略表

只对少数体系做开关 A/B：

1. `240_tnhvp`: internal chart on/off
2. `241_VBSCF`: internal chart on/off
3. `MnF2`: 如需要，可作为 open-shell 对照

## 9. 我建议的最终写法

如果只从论文叙事角度看，正文最干净的体系组合是：

1. `F2`
2. `benzene (241_VBSCF)`
3. `MnF2`
4. `TiCl`
5. `FeCl2`
6. `10698_VBSCF`（可写成 “larger closed-shell HAO benchmark deck”）

这样可以把论文里的数值部分组织成：

- 小体系正确性：`F2`
- 闭壳层主 benchmark：`benzene`
- 开壳层困难 benchmark：`MnF2`
- OEO 迁移性：`TiCl`
- 生产级过渡金属：`FeCl2`
- 大 deck 性能：`10698_VBSCF`

这是当前最稳妥、也最省字的一套方案。
