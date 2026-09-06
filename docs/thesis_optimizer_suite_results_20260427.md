# 毕业论文优化器对比结果（2026-04-27）

数据来源：

- 汇总 TSV：[benchmarks/thesis_optimizer_suite_20260427_124732/suite_summary.tsv](/pool1/home/xiatao/project/xmvb-cpp/benchmarks/thesis_optimizer_suite_20260427_124732/suite_summary.tsv)
- 运行目录：[benchmarks/thesis_optimizer_suite_20260427_124732](/pool1/home/xiatao/project/xmvb-cpp/benchmarks/thesis_optimizer_suite_20260427_124732)

统一运行条件：

- partition `6526Y`
- `CPUS_PER_TASK=32`
- `OMP_NUM_THREADS=32`
- `OPENBLAS_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`

需要区分三条路线：

1. `cpp_tnhvp`：`xmvb-cpp` 的 `nonredundant_truncated_newton`
2. `cpp_lbfgs`：`xmvb-cpp` 的 `lbfgspp`
3. `legacy_xmvb`：通过 [xmvb.sh](/pool1/home/xiatao/project/xmvb-cpp/xmvb.sh) 调用的 legacy XMVB 端到端基线

`legacy_xmvb` 不是“强制 legacy LBFGS”或“强制 legacy TNHVP”，而是按 deck 自身关键字运行，因此更适合作为历史生产基线。

## 1. 原始结果表

| 体系 | 类型/角色 | backend | 收敛 | 迭代数 | wall time / s | final energy / Eh | final gradient inf-norm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `F2` | correctness / sanity check | `cpp_tnhvp` | yes | 8 | 0.361937 | -198.751155798123 | 2.07495225e-04 |
| `F2` | correctness / sanity check | `cpp_lbfgs` | yes | 28 | 0.634885 | -198.751155827862 | 1.31653316e-04 |
| `F2` | correctness / sanity check | `legacy_xmvb` | yes | 28 | 2.232 | -198.75115582 |  |
| `benzene` (`241_VBSCF`) | closed-shell sparse HAO 主 benchmark | `cpp_tnhvp` | yes | 9 | 7.589154 | -230.720590176024 | 1.51528928e-04 |
| `benzene` (`241_VBSCF`) | closed-shell sparse HAO 主 benchmark | `cpp_lbfgs` | yes | 74 | 14.464779 | -230.720589594596 | 2.56206771e-04 |
| `benzene` (`241_VBSCF`) | closed-shell sparse HAO 主 benchmark | `legacy_xmvb` | yes | 76 | 22.502 | -230.72058966 |  |
| `MnF2` | difficult open-shell sparse HAO | `cpp_tnhvp` | yes | 132 | 186.600209 | -1348.893351183859 | 1.32436349e-03 |
| `MnF2` | difficult open-shell sparse HAO | `cpp_lbfgs` | yes | 992 | 119.214302 | -1348.893350530760 | 1.61715722e-04 |
| `MnF2` | difficult open-shell sparse HAO | `legacy_xmvb` | yes | 1067 | 344.557 | -1348.89335001 |  |
| `FeCl2` | production open-shell OEO | `cpp_tnhvp` | yes | 3 | 4.485240 | -2181.617636108513 | 3.31620464e-04 |
| `FeCl2` | production open-shell OEO | `cpp_lbfgs` | yes | 1040 | 184.142219 | -2181.617642772074 | 1.11246578e-04 |
| `FeCl2` | production open-shell OEO | `legacy_xmvb` | yes | 1044 | 684.252 | -2181.61764257 |  |
| `10698` (`10698_VBSCF`) | larger closed-shell HAO benchmark deck | `cpp_tnhvp` | yes | 11 | 83.311023 | -422.611823781255 | 4.00343331e-04 |
| `10698` (`10698_VBSCF`) | larger closed-shell HAO benchmark deck | `cpp_lbfgs` | yes | 122 | 96.598535 | -422.611823580310 | 4.25825917e-04 |
| `10698` (`10698_VBSCF`) | larger closed-shell HAO benchmark deck | `legacy_xmvb` | yes | 115 | 169.270 | -422.61182365 |  |
| `240` (`240_tnhvp`) | supplementary internal benchmark deck | `cpp_tnhvp` | yes | 20 | 112.572492 | -343.446301677009 | 6.38437172e-04 |
| `240` (`240_tnhvp`) | supplementary internal benchmark deck | `cpp_lbfgs` | yes | 126 | 272.465908 | -343.446301442804 | 1.30801403e-04 |
| `240` (`240_tnhvp`) | supplementary internal benchmark deck | `legacy_xmvb` | yes | 115 | 1330.044 | -343.44630103 |  |

## 2. 派生比较

下面用 `LBFGS/TNHVP` 和 `legacy/TNHVP` 表示 wall-time 比值。数值大于 `1` 表示 TNHVP 更快。

| 体系 | LBFGS/TNHVP wall-time | legacy/TNHVP wall-time | LBFGS/TNHVP iteration ratio | `E_TN - E_LBFGS` / Eh |
| --- | ---: | ---: | ---: | ---: |
| `F2` | 1.75 | 6.17 | 3.50 | 2.97389988191e-08 |
| `benzene` | 1.91 | 2.97 | 8.22 | -5.81427997304e-07 |
| `MnF2` | 0.64 | 1.85 | 7.52 | -6.53099050396e-07 |
| `FeCl2` | 41.06 | 152.56 | 346.67 | 6.66356072543e-06 |
| `10698` | 1.16 | 2.03 | 11.09 | -2.00945009965e-07 |
| `240` | 2.42 | 11.82 | 6.30 | -2.34205003835e-07 |

## 3. 可直接写进论文的结论

### 3.1 不是“TNHVP 全面更快”

这批数据不能支持“TNHVP 在所有体系上都优于 LBFGS”。

- `benzene`、`FeCl2`、`10698`、`240` 上，TNHVP 的 wall-time 优于 `cpp_lbfgs`
- `F2` 上 TNHVP 也更快，但体系太小，只能当 correctness / sanity check
- `MnF2` 上，TNHVP 的迭代数显著更少，但总 wall-time 反而比 `cpp_lbfgs` 更慢

因此，正文更稳妥的表述应是：

> TNHVP 在多个代表体系上可以显著减少迭代数，并且在若干 closed-shell 或较稳定的 open-shell 体系上带来明显 wall-time 收益；但在困难 open-shell sparse HAO 体系上，其单步二阶代价仍可能抵消迭代数优势。

### 3.2 迭代数收益是真实的

TNHVP 在 6 个体系上都显著减少了迭代数：

- `F2`: `8` vs `28`
- `benzene`: `9` vs `74`
- `MnF2`: `132` vs `992`
- `FeCl2`: `3` vs `1040`
- `10698`: `11` vs `122`
- `240`: `20` vs `126`

所以 TNHVP 的主要优势首先体现在“步数更少”，并不总能自动转化成“总时间更少”。

### 3.3 `FeCl2` 是最强主结果

`FeCl2` 是这批结果里最有说服力的体系：

- TNHVP：`3` 步，`4.485240 s`
- `cpp_lbfgs`：`1040` 步，`184.142219 s`
- `legacy_xmvb`：`1044` 步，`684.252 s`

这个体系上 TNHVP 相对 `cpp_lbfgs` 的 wall-time 加速约为 `41.06x`，相对 `legacy_xmvb` 约为 `152.56x`。

### 3.4 `MnF2` 是必须保留的反例/边界例

`MnF2` 不能删，因为它恰好说明：

- TNHVP 的模型质量并不是唯一问题；
- 难 open-shell sparse HAO 体系上，单次 HVP / outer-response / orbital pullback 成本仍然很重；
- “二阶步数优势”与“总 wall-time 优势”不是同一件事。

论文里保留 `MnF2`，会让结论更可信。

### 3.5 最终能量差异很小，但趋势上 LBFGS/legacy 往往略低

所有体系三路都收敛，最终能量彼此接近。相对 `cpp_lbfgs`，TNHVP 的最终能量差异量级大致在 `1e-7` 到 `1e-6 Eh`，`FeCl2` 上约为 `6.7e-6 Eh`。

这意味着：

- 对论文主结论，三条路线都能到达非常接近的最终能量；
- 但若要强调“最低能量”，则 `cpp_lbfgs` 和 `legacy_xmvb` 往往略占优；
- 若要强调“更快达到合理收敛”，则 `benzene`、`FeCl2`、`10698`、`240` 更适合用来突出 TNHVP。

## 4. 论文表述建议

### 正文主表建议保留

1. `benzene`
2. `MnF2`
3. `FeCl2`
4. `10698`

### 正文里如何使用

- `benzene`：closed-shell sparse HAO 主 benchmark，TNHVP 明显更快
- `MnF2`：困难 open-shell sparse HAO 反例，说明 TNHVP 还不是普适 wall-time 最优
- `FeCl2`：最强性能结果
- `10698`：较大 deck 上仍有 wall-time 收益

### 补充材料建议保留

- `F2`：正确性与 sanity check
- `240`：internal benchmark deck，适合和图表/策略敏感性一起放补充材料

## 5. 后续可直接复用的文件

- 总汇总表：[benchmarks/thesis_optimizer_suite_20260427_124732/suite_summary.tsv](/pool1/home/xiatao/project/xmvb-cpp/benchmarks/thesis_optimizer_suite_20260427_124732/suite_summary.tsv)
- 本文档：[docs/thesis_optimizer_suite_results_20260427.md](/pool1/home/xiatao/project/xmvb-cpp/docs/thesis_optimizer_suite_results_20260427.md)

如果后面要往论文正文里贴表，建议直接从 `suite_summary.tsv` 抽 `cpp_tnhvp` 与 `cpp_lbfgs` 两行，`legacy_xmvb` 放附表或补充材料。
