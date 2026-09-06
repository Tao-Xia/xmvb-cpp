# 毕业论文数据表模板

这个文档不讨论算法细节，只负责把将要计算的体系和最终要落入论文的数据字段固定下来。建议先按这里的表头收集结果，再回填到正文。

## 1. 正文主体系

建议正文主体系固定为：

| 分子/体系 | 输入文件 | 角色 |
| --- | --- | --- |
| `F2` | `src/test_molecule/F2.xmi` | 正确性与 sanity check |
| `benzene` | `test/241_VBSCF.xmi` | closed-shell sparse HAO 主 benchmark |
| `MnF2` | `test/MnF2.xmi` | open-shell sparse HAO 困难体系 |
| `TiCl` | `test/TiCl.xmi` | 小型 open-shell OEO 迁移性检查 |
| `FeCl2` | `test/FeCl2.xmi` | 生产级 open-shell OEO 体系 |
| `10698_VBSCF` | `test/10698_VBSCF.xmi` | larger closed-shell HAO benchmark deck |

如果时间不够，`TiCl` 可以延后；其余 5 个优先。

## 2. 表 1：体系摘要表

这张表建议放在数值实验章节开头。

| 体系 | 轨道类型 | 自旋多重度 | basis | active space | AO 数 | structures 数 | determinants 数 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `F2` |  |  |  |  |  |  |  | correctness only |
| `benzene` |  |  |  |  |  |  |  | `241_VBSCF` |
| `MnF2` |  |  |  |  |  |  |  | difficult open-shell |
| `TiCl` |  |  |  |  |  |  |  | OEO transfer check |
| `FeCl2` |  |  |  |  |  |  |  | production OEO |
| `10698_VBSCF` |  |  |  |  |  |  |  | larger deck |

## 3. 表 2：优化收敛主表

建议正文主表只比较两条主线：

- `nonredundant_truncated_newton`
- `nonredundant_lbfgspp`

统一运行条件：

- `sbatch`
- partition `6526Y`
- `OMP_NUM_THREADS=32`
- `OPENBLAS_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`

| 体系 | backend | 是否收敛 | 迭代数 | SCF wall time / s | 最终能量 / Eh | final projected grad norm | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `F2` | TN |  |  |  |  |  | sanity check |
| `F2` | LBFGS |  |  |  |  |  | sanity check |
| `benzene` | TN |  |  |  |  |  | 主 closed-shell 表项 |
| `benzene` | LBFGS |  |  |  |  |  | 主 closed-shell 表项 |
| `MnF2` | TN |  |  |  |  |  | difficult open-shell |
| `MnF2` | LBFGS |  |  |  |  |  | difficult open-shell |
| `TiCl` | TN |  |  |  |  |  | OEO transfer |
| `TiCl` | LBFGS |  |  |  |  |  | OEO transfer |
| `FeCl2` | TN |  |  |  |  |  | production OEO |
| `FeCl2` | LBFGS |  |  |  |  |  | production OEO |
| `10698_VBSCF` | TN |  |  |  |  |  | larger deck |
| `10698_VBSCF` | LBFGS |  |  |  |  |  | larger deck |

## 4. 表 3：阶段耗时表

这张表不需要所有体系，只要挑 2 到 3 个代表体系即可。建议：

- `benzene`
- `FeCl2`
- `10698_VBSCF`

如果当前代码已经能稳定输出分项耗时，则记录：

| 体系 | total HVP time / s | active integrals / s | outer-response / s | orbital pullback / s | active 2e contraction / s | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| `benzene` |  |  |  |  |  |  |
| `FeCl2` |  |  |  |  |  |  |
| `10698_VBSCF` |  |  |  |  |  |  |

如果当前日志没有完整分项，至少保留：

| 体系 | total HVP applies | total HVP wall time / s | SCF wall time / s | 备注 |
| --- | --- | --- | --- | --- |
| `benzene` |  |  |  |  |
| `FeCl2` |  |  |  |  |
| `10698_VBSCF` |  |  |  |  |

## 5. 表 4：策略/图表补充表

这一张建议只放补充材料，不放正文主表。

| 体系 | 策略 | 是否收敛 | first-step energy drop | 迭代数 | SCF wall time / s | rejected/rescue 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| `240_tnhvp` | default |  |  |  |  |  |
| `240_tnhvp` | force internal inactive chart |  |  |  |  |  |
| `benzene` | default |  |  |  |  |  |
| `benzene` | force internal inactive chart |  |  |  |  |  |
| `MnF2` | default |  |  |  |  |  |
| `MnF2` | force internal inactive chart |  |  |  |  |  |

## 6. 第一批建议计算

如果当前目标是尽快补论文数据，建议先交这一批：

1. `F2`: TN + LBFGS
2. `241_VBSCF`: TN + LBFGS
3. `MnF2`: TN + LBFGS
4. `FeCl2`: TN + LBFGS
5. `10698_VBSCF`: TN + LBFGS

这样先把正文最核心的 5 个体系补齐。`TiCl` 作为第二批补充。

## 7. 数据回填原则

正文里尽量不要直接写内部 deck 编号作为体系名。建议：

- `241_VBSCF` 写成 `benzene`
- `10698_VBSCF` 写成 `larger closed-shell HAO benchmark deck`
- `240_tnhvp` 只在补充材料中写作 `internal benchmark deck`

同一张表内应统一：

- 时间单位统一为秒
- 能量统一为 Hartree
- 梯度统一使用同一种 projected gradient norm
- 所有数据都来自同一分区 `6526Y` 和同一线程设置
