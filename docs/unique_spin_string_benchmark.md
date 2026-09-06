# Unique Spin-String Benchmark Analysis

## 背景

本文档记录 same-spin unique spin-string / unique spin-pair 算法的
benchmark 结果，并分析它在 `exact` 与 `ri` 两条 production 路径中的
实际收益。

这里的目标不是缓存完整的 determinant pair，而是：

1. 先对 alpha / beta spin-string pair 去重。
2. 只计算唯一的 same-spin pair kernel。
3. 再由这些 spin-resolved payload 重建完整的 determinant-pair
   overlap / Hamiltonian / gradient 贡献。

算法说明见 [unique_spin_pair_cache.md](./unique_spin_pair_cache.md)。

## 比较口径

### 算法开关

当前 A/B 对比不需要改代码，直接使用环境变量：

```bash
XMVB_SAME_SPIN_PAIR_CACHE_MB=0
```

在
[src/vb/matrices/same_spin_pair_cache.cpp](../src/vb/matrices/same_spin_pair_cache.cpp)
中，这会让 same-spin pair cache 预算直接变成 0，因此完全关闭 unique
spin-string / unique spin-pair 路径。

未设置该环境变量，或设置为正数时，表示开启该算法。

需要注意，程序最终摘要中的

```text
algorithm = original
```

并不表示 same-spin unique spin-string 算法被关闭。当前这个字段描述的是更高层的
算法标签，不反映 `XMVB_SAME_SPIN_PAIR_CACHE_MB` 的取值。因此做 A/B 时应以
环境变量和 stage 计时结果为准，而不是看这行输出。

### 体系与输入

本次对比统一使用：

- 输入文件：`data/training_xmi/6e6o_full/10698_VBSCF.xmi`
- `OMP_NUM_THREADS=1`
- raw structure 数：`175`
- expanded determinant 数：`400`
- determinant pair 数：`15400`（含对角）

### 比较哪些时间

我们使用两类 benchmark：

1. `benchmark_union_graph_single_step`
   这是更干净的 kernel benchmark，可以单独看 structure build 和单步总时长。
2. `build/src/xmvb-cpp.exe --max-iterations 1`
   这是 production 可执行文件的真实单步 objective + gradient 时间。

对第二类 benchmark，首轮 `call=0` 会包含 RI 初始缓存构建，不适合拿来做
steady-state 对比。因此本文统一使用 `call=2` 作为稳定口径。

对 `benchmark_union_graph_single_step`，总时间近似满足

$$
T_{\text{single-step}}
\approx
T_{\text{prepare-active-space}}
+
T_{\text{structure-build}}
+
T_{\text{eigensolve}}.
$$

对 production `xmvb-cpp.exe`，单次 objective/gradient 的计时满足

$$
T_{\text{objective}}
=
T_{\text{active-space}}
+
T_{\text{matrix-backprop}}
+
T_{\text{active-2e-backprop}}
+
T_{\text{ao-h1e-backprop}}
+
T_{\text{orbital-backprop}},
$$

其中

$$
T_{\text{active-space}}
=
T_{\text{prepared-active-space}}
+
T_{\text{same-spin-phi-cache}}
+
T_{\text{structure}}
+
T_{\text{eigensolver}}
+
T_{\text{adjoint}}.
$$

## Benchmark 命令

### Kernel benchmark

```bash
env OMP_NUM_THREADS=1 build/src/benchmark_union_graph_single_step \
  data/training_xmi/6e6o_full/10698_VBSCF.xmi \
  --standard-two-electron-mode exact

env OMP_NUM_THREADS=1 XMVB_SAME_SPIN_PAIR_CACHE_MB=0 \
  build/src/benchmark_union_graph_single_step \
  data/training_xmi/6e6o_full/10698_VBSCF.xmi \
  --standard-two-electron-mode exact

env OMP_NUM_THREADS=1 build/src/benchmark_union_graph_single_step \
  data/training_xmi/6e6o_full/10698_VBSCF.xmi \
  --standard-two-electron-mode ri

env OMP_NUM_THREADS=1 XMVB_SAME_SPIN_PAIR_CACHE_MB=0 \
  build/src/benchmark_union_graph_single_step \
  data/training_xmi/6e6o_full/10698_VBSCF.xmi \
  --standard-two-electron-mode ri
```

### Production 单步 objective/gradient benchmark

```bash
env OMP_NUM_THREADS=1 XMVB_CPP_LOG_OBJECTIVE_PROGRESS=1 \
  build/src/xmvb-cpp.exe \
  data/training_xmi/6e6o_full/10698_VBSCF.xmi \
  --optimizer-backend lbfgspp \
  --max-iterations 1 \
  --standard-two-electron-mode ri

env OMP_NUM_THREADS=1 XMVB_CPP_LOG_OBJECTIVE_PROGRESS=1 \
  XMVB_SAME_SPIN_PAIR_CACHE_MB=0 \
  build/src/xmvb-cpp.exe \
  data/training_xmi/6e6o_full/10698_VBSCF.xmi \
  --optimizer-backend lbfgspp \
  --max-iterations 1 \
  --standard-two-electron-mode ri
```

## 结果一：`benchmark_union_graph_single_step`

### `exact` 路径

| 指标 | 关算法 (`MB=0`) | 开算法 | 加速比 |
| --- | ---: | ---: | ---: |
| `prepare_active_space_wall_time_seconds` | `3.865149 s` | `3.941489 s` | `0.98x` |
| `cpp_exact_structure_build_wall_time_seconds` | `0.236393 s` | `0.009621 s` | `24.57x` |
| `cpp_exact_eigensolve_wall_time_seconds` | `0.010679 s` | `0.010722 s` | `1.00x` |
| `cpp_exact_single_step_wall_time_seconds` | `4.112221 s` | `3.961832 s` | `1.04x` |

### `ri` 路径

| 指标 | 关算法 (`MB=0`) | 开算法 | 加速比 |
| --- | ---: | ---: | ---: |
| `prepare_active_space_wall_time_seconds` | `0.555282 s` | `0.571320 s` | `0.97x` |
| `cpp_exact_structure_build_wall_time_seconds` | `9.869368 s` | `0.063188 s` | `156.19x` |
| `cpp_exact_eigensolve_wall_time_seconds` | `0.010443 s` | `0.010891 s` | `0.96x` |
| `cpp_exact_single_step_wall_time_seconds` | `10.435093 s` | `0.645399 s` | `16.17x` |

### 对 kernel benchmark 的解释

在 `exact` 路径下，same-spin unique spin-string 算法把 structure build
本身压到了原来的约 `1/25`，但单步总时间只改善了约 `4%`。原因不是算法无效，
而是 `prepare_active_space` 本身就接近 `4 s`，已经占据了绝大部分墙钟时间。

在 `ri` 路径下，`prepare_active_space` 已经被压到 `0.56 s` 左右，这时
same-spin determinant-pair 构建会重新成为主瓶颈。因此一旦开启 unique
spin-string 复用，structure build 从 `9.87 s` 下降到 `0.063 s`，单步总时长
也同步从 `10.44 s` 下降到 `0.65 s`。

这说明该算法并不是只在 micro-benchmark 中有效，而是在 `ri` production
路径下真正决定单步时间。

## 结果二：production `xmvb-cpp.exe` 单步 objective/gradient

以下对比统一取 steady-state 的 `call=2`。

### `ri` 端到端 A/B

| 指标 | 关算法 (`MB=0`) | 开算法 | 加速比 |
| --- | ---: | ---: | ---: |
| `ao_h1e` | `0.515343 s` | `0.444112 s` | `1.16x` |
| `active_2e` | `0.143610 s` | `0.095254 s` | `1.51x` |
| `same_spin_phi_cache + structure` | `10.701026 s` | `0.022797 s` | `469.41x` |
| `active_adjoint` | `20.772451 s` | `0.176858 s` | `117.45x` |
| `active_space` | `32.171525 s` | `0.827661 s` | `38.87x` |
| `active_2e_backprop` | `0.140745 s` | `0.113439 s` | `1.24x` |
| `ao_h1e_backprop` | `0.505917 s` | `0.494185 s` | `1.02x` |
| `total` | `32.829276 s` | `1.443871 s` | `22.74x` |

这里将

$$
T_{\text{same-spin-related}}
=
T_{\text{same-spin-phi-cache}}
+
T_{\text{structure}}
+
T_{\text{adjoint}}
$$

视为 same-spin unique spin-string 直接作用的主要区域。

对应地：

$$
T_{\text{same-spin-related, off}}
=
10.701026 + 20.772451
=
31.473477 \text{ s},
$$

$$
T_{\text{same-spin-related, on}}
=
0.015685 + 0.007112 + 0.176858
=
0.199655 \text{ s}.
$$

即 same-spin 相关部分从总时间占比

$$
\frac{31.473477}{32.829276} \approx 95.9\%
$$

下降到

$$
\frac{0.199655}{1.443871} \approx 13.8\%.
$$

## 核心结论

### 1. 算法目标已经兑现

本次 benchmark 证明，当前实现确实达到了最初目标：

- 不再把 full determinant pair 当作最小复用单位。
- 而是把 alpha / beta unique spin-string pair 当作复用单位。
- 在 full determinant-pair 层面只做重组，不再重复计算相同的 same-spin kernel。

在 `ri` production path 上，这个设计已经带来数量级的实际收益，而不是概念收益。

### 2. forward 和 backward 都已经受益

如果只有前向 structure build 被加速，而反向没有跟上，那么端到端单步时间不可能
从 `32.83 s` 降到 `1.44 s`。

现在的结果表明两部分都吃到了收益：

- 前向：`same_spin_phi_cache + structure`
- 反向：`active_adjoint`

尤其 `active_adjoint` 从 `20.77 s` 降到 `0.177 s`，说明 forward 建好的
same-spin payload 已经实质性复用于 backward。

### 3. 为什么 `exact` 路径下总收益不显著

这不是 same-spin 算法的问题，而是主瓶颈位置不同。

在 `exact` 路径中，

$$
T_{\text{prepare-active-space}} \approx 3.9 \text{ s}
$$

已经接近整个单步时间，因此 same-spin structure build 即使提速 `24.6x`，
总步长也只从 `4.11 s` 下降到 `3.96 s`。

换句话说，`exact` 路径下 same-spin 去重是正确的优化，但不是总墙钟的主导优化。

### 4. 为什么 `ri` 路径下收益巨大

`ri` 路径把 AO / active-space 积分准备阶段压缩到 `0.56 s` 左右后，
same-spin determinant-pair 构建与其 adjoint 立刻暴露成主瓶颈。

于是同一个 unique spin-string 复用策略，在 `ri` 路径下就从“局部优化”
变成了“决定总时间的核心优化”。

这也解释了为什么 `benchmark_union_graph_single_step` 和真实
`xmvb-cpp.exe` 单步时间，都在 `ri` 模式下表现出数量级加速。

### 5. 现在的主瓶颈已经转移

开启该算法后，steady-state `call=2` 中最显著的剩余成本变成：

- `ao_h1e = 0.444112 s`
- `ao_h1e_backprop = 0.494185 s`

两者合计

$$
0.444112 + 0.494185 = 0.938297 \text{ s},
$$

占单步总时间

$$
\frac{0.938297}{1.443871} \approx 65.0\%.
$$

这说明在 same-spin unique spin-string 这条线上，当前实现已经基本做透。
后续如果继续追求单步时间下降，更合理的方向应当是：

- opposite-spin 项；
- AO / 非活性空间侧的剩余 contraction；
- 或者进一步压缩相应的 backward 路径。

## 数值一致性

本次 A/B benchmark 中，算法开关不会改变数值结果。

以 production `ri` 路径一步优化后的总能量为例，开关前后都得到

$$
E_{\text{final}} \approx -422.343596471768.
$$

因此该优化当前可以视为纯性能优化，不引入可见的能量偏差。

## 建议的生产结论

1. 默认应保持 same-spin unique spin-string 算法开启。
2. `XMVB_SAME_SPIN_PAIR_CACHE_MB=0` 只适合做 debug、回归测试和 benchmark A/B。
3. 在 `ri` production path 中，该算法已经是核心优化之一，关闭后单步 wall time
   会从约 `1.44 s` 回退到约 `32.83 s`。
4. 下一阶段如果还要继续压缩时间，优先级应该转向 opposite-spin 与 AO-side
   residual bottleneck，而不是继续深挖 same-spin 去重本身。
