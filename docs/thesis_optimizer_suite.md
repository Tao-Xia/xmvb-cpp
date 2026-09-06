# 毕业论文优化器对比计算套件

这个套件固定为 6 个体系：

| 标签 | 输入文件 | 论文角色 |
| --- | --- | --- |
| `F2` | `test/F2.xmi` | 小体系正确性与 sanity check |
| `benzene` | `test/241_VBSCF.xmi` | closed-shell sparse HAO 主 benchmark |
| `MnF2` | `test/MnF2.xmi` | difficult open-shell sparse HAO |
| `FeCl2` | `test/FeCl2.xmi` | 生产级 open-shell OEO |
| `10698` | `test/10698_VBSCF.xmi` | larger closed-shell HAO benchmark deck |
| `240` | `test/240_tnhvp.xmi` | 补充材料里的 internal benchmark deck |

每个体系默认提交三路计算：

1. `cpp_tnhvp`
2. `cpp_lbfgs`
3. `legacy_xmvb`

其中前两路是 `xmvb-cpp` 的 `TNHVP` / `LBFGS` 对比。第三路通过 [xmvb.sh](/pool1/home/xiatao/project/xmvb-cpp/xmvb.sh) 跑 legacy XMVB，可作为生产脚本基线。

需要注意：`legacy_xmvb` 不是“强制 legacy LBFGS”或“强制 legacy TNHVP”。它按输入 deck 自身关键字运行，因此更适合作为端到端历史基线，而不是与 `cpp_tnhvp`、`cpp_lbfgs` 做逐选项一一对应。

## 提交脚本

批量提交入口：

```bash
bash scripts/submit_thesis_optimizer_suite.sh
```

默认设置：

- partition: `6526Y`
- `CPUS_PER_TASK=32`
- `OMP_NUM_THREADS=32`
- `OPENBLAS_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `TIME_LIMIT=24:00:00`

只跑子集：

```bash
bash scripts/submit_thesis_optimizer_suite.sh F2 benzene 240
```

给 `xmvb-cpp` 两路统一附加参数：

```bash
bash scripts/submit_thesis_optimizer_suite.sh -- --max-iterations 80
```

只做 dry run：

```bash
DRY_RUN=1 bash scripts/submit_thesis_optimizer_suite.sh MnF2 FeCl2
```

## 结果目录

默认输出目录：

```text
benchmarks/thesis_optimizer_suite_<timestamp>/
```

其中每个体系一个子目录，例如：

```text
benchmarks/thesis_optimizer_suite_<timestamp>/MnF2/
```

每个体系目录下会包含：

- `jobs.tsv`：三路任务的 Slurm job id 与日志位置
- `submit.log`：提交记录
- `cpp_tnhvp/`
- `cpp_lbfgs/`
- `legacy_xmvb/`

顶层目录还会生成：

- `systems.tsv`：本批次包含的体系与目录映射
- `submit_all.log`：整批提交记录

## 汇总脚本

整批任务完成后，用下面的脚本生成总表：

```bash
bash scripts/summarize_thesis_optimizer_suite.sh benchmarks/thesis_optimizer_suite_<timestamp>
```

它会输出：

```text
benchmarks/thesis_optimizer_suite_<timestamp>/suite_summary.tsv
```

这个总表会把 6 个体系的 `cpp_tnhvp`、`cpp_lbfgs`、`legacy_xmvb` 汇总到同一个 TSV 中，便于直接回填论文表格。

## 论文使用建议

- 正文主对比表：优先使用 `cpp_tnhvp` vs `cpp_lbfgs`
- `legacy_xmvb`：作为历史生产基线或附表
- `240`：建议只放补充材料，不放正文主表
