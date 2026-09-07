# TNHVP Outer-Response 近似策略分析

> **已退役（2026-09-07）**：本文记录的 core-only、local-only、
> energy-only、SR1 和 hybrid full-retry 路径均已从生产代码删除。
> `exact_ctx` 现在只表示物理 (U_p) 坐标上的完整解析 Hessian 作用。
> 下文仅保留为历史分析，所列环境变量和运行方式不再有效。

## 1. 目标和当前 exact 路径

TNHVP 在 `exact_ctx` 路径下已经有两种 reduced-space HVP：

\[
B_k^{(H)} p = B_k^{(L)} p + B_k^{(\mathrm{outer})} p .
\]

其中：

- \(B_k^{(L)}\) 是 cheap/core-only HVP；
- \(B_k^{(H)}\) 是 full HVP；
- \(B_k^{(\mathrm{outer})}\) 是 relaxed selected-state / structure response 带来的高保真修正；
- \(p \in \mathbb{R}^{n_r}\) 是 nonredundant reduced-space 方向。

当前 full 路径在
[`src/vb/scf/exact_orbital_second_order_operator.cpp`](../src/vb/scf/exact_orbital_second_order_operator.cpp)
中按下面顺序执行 outer-response：

1. 构造 active-space 方向积分
   \[
   u = T_k p =
   \left(
   \delta S_{\mathrm{act}},
   \delta h_{\mathrm{act}},
   \delta g_{\mathrm{act}}^{(2)}
   \right);
   \]
2. 构造 selected-state projected structure columns；
3. 做 selected-state generalized-eigen 一阶响应；
4. 用方向 selected-state 系数和能量构造 active-space gradient direction；
5. 通过 orbital pullback 回到 reduced space。

已有文档
[`outer_response_linear_response_predecomposition.md`](./outer_response_linear_response_predecomposition.md)
和
[`mftr_outer_response_projection_theory.md`](./mftr_outer_response_projection_theory.md)
已经说明，在 accepted point \(x_k\) 固定时，outer-response 可以写成

\[
B_k^{(\mathrm{outer})}
=
T_k^\sharp K_k^{(\mathrm{outer})} T_k,
\]

其中

\[
K_k^{(\mathrm{outer})}
=
W_k^{(\mathrm{loc})}
+
W_k^{(\mathrm{dir})}
R_{k,\mathrm{sel}}
P_{k,\mathrm{sel}} .
\]

本文只回答一个更具体的问题：

\[
\text{能不能不直接计算 } B_k^{(\mathrm{outer})}p,
\text{而用更便宜的近似替代？}
\]

结论是：可以，但最有希望的路线不是继续逐方向做 full outer-response 的变体，而是用 **缺失曲率的低秩/secant 近似** 或 **解析链路的截断近似** 替代它。

## 2. Exact selected-state 响应公式

对 selected state \(i\)，当前代码的 generalized-eigen 一阶响应可以概括为：

\[
H_k U_k = S_k U_k E_k,
\qquad
E_k = \operatorname{diag}(\varepsilon_1,\ldots,\varepsilon_{n_{\mathrm{str}}}).
\]

令

\[
n_i =
\delta \widetilde h_i
-
\varepsilon_i \delta \widetilde s_i,
\]

其中 \(\delta \widetilde h_i\) 和 \(\delta \widetilde s_i\) 是 transformed directional Hamiltonian / overlap 的 selected column。则

\[
\delta \varepsilon_i = e_i^\mathsf{T} n_i,
\]

并且在非简并情形下，

\[
(\delta \Omega_i)_j =
\begin{cases}
-\dfrac{1}{2}(\delta \widetilde s_i)_i, & j=i,\\[6pt]
\dfrac{(n_i)_j}{\varepsilon_i-\varepsilon_j}, & j \ne i.
\end{cases}
\]

最终

\[
\delta c_i = U_k \delta \Omega_i.
\]

代码中还对近简并 gap 使用 accepted-point 固定的安全分母和 gauge 处理。这个响应公式说明了 outer-response 贵的原因：即使只关心 selected states，每个 selected state 的一阶系数响应仍会遍历全部 structure roots。

后续近似的核心就是替换或截断下面这一项：

\[
R_{k,\mathrm{sel}}
P_{k,\mathrm{sel}} T_k p .
\]

## 3. 近似策略的评价标准

一个可用的 TNHVP outer-response 近似应满足：

1. 单次 HVP 不能接近 full outer-response 成本；
2. 保持 reduced-space 模型尽量对称，否则 CG / trust-region 子问题会变差；
3. 允许不精确，但误差必须能被 trust-ratio 和收敛判据兜住；
4. 在 selected states、active space、nonredundant chart 变化时必须重置；
5. 不能只在 `F2` 上验证，因为 `F2` 太小，outer-response 误差和性能差别都不明显。

建议验证体系至少包括：

- `test/241_VBSCF.xmi`；
- `test/MnF2.xmi`；
- 一个更大的 deck，例如已有 benchmark 中的 `10698_VBSCF`。

## 4. 策略 A：缺失曲率的 L-SR1 / L-BFGS 近似

这是最推荐优先尝试的近似，因为它可以 **完全不额外调用 exact outer-response HVP**。

### 4.1 基本思想

在 accepted points 上，优化器已经会得到真实 relaxed objective 的梯度。设

\[
s_j = x_{j+1} - x_j,
\qquad
y_j = g_{j+1} - g_j .
\]

其中 \(g_j\) 是 accepted point 的 reduced gradient。若在 \(x_j\) 处计算一次 cheap/core-only Hessian action

\[
c_j = B_j^{(L)} s_j,
\]

则

\[
r_j = y_j - c_j
\]

就是 full Hessian secant 中 cheap/core-only 模型解释不了的剩余部分。理想情况下，

\[
r_j
\approx
B_j^{(\mathrm{outer})} s_j
+
O(\|s_j\|^2).
\]

因此可以用历史 secant 对构造一个低秩的

\[
\widehat B_k^{(\mathrm{outer})}
\]

并在 TNHVP 内部使用

\[
\widehat B_k p
=
B_k^{(L)}p
+
\widehat B_k^{(\mathrm{outer})}p .
\]

这条路线的关键优点是：\(r_j\) 来自已经发生的 accepted step 和 gradient difference，不需要为每个内层 HVP 额外走 full outer-response 链路。

### 4.2 L-SR1 公式

令当前近似为 \(M_j \approx B^{(\mathrm{outer})}\)。对 secant pair \((s_j,r_j)\)，定义

\[
v_j = r_j - M_j s_j.
\]

若满足安全条件

\[
\left|v_j^\mathsf{T}s_j\right|
\ge
\tau_{\mathrm{sr1}}
\|v_j\|\,\|s_j\|,
\]

则做 SR1 更新：

\[
M_{j+1}
=
M_j
+
\frac{v_j v_j^\mathsf{T}}{v_j^\mathsf{T}s_j}.
\]

实际实现中应使用 limited-memory 形式，只保留最近 \(m_{\mathrm{sec}}\) 个 pair。HVP 成本为

\[
O(m_{\mathrm{sec}} n_r),
\]

远低于 full outer-response。

### 4.3 L-BFGS 变体

若希望 correction 更稳定并近似正定，可在

\[
r_j^\mathsf{T}s_j > 0
\]

时使用 BFGS：

\[
M_{j+1}
=
M_j
-
\frac{M_js_js_j^\mathsf{T}M_j}{s_j^\mathsf{T}M_js_j}
+
\frac{r_jr_j^\mathsf{T}}{r_j^\mathsf{T}s_j}.
\]

但是 outer-response 本身未必正定。若真实缺失曲率有明显负曲率，SR1 通常比 BFGS 更合适。

### 4.4 初始标量模型

可以令

\[
M_0 = \alpha_k I,
\]

其中

\[
\alpha_k
=
\operatorname{clip}
\left(
\frac{s_{k-1}^\mathsf{T} r_{k-1}}
     {s_{k-1}^\mathsf{T} s_{k-1}},
\alpha_{\min},
\alpha_{\max}
\right).
\]

如果没有可靠 secant，则使用

\[
M_0 = 0.
\]

这会退化为当前 cheap/core-only TNHVP。

### 4.5 安全策略

建议的 safeguard：

1. 如果 selected-state indices、active orbital count、nonredundant dimension 改变，清空 secant history；
2. 如果 trust ratio 连续偏低，缩小或清空 \(\widehat B^{(\mathrm{outer})}\)；
3. 限制 correction 大小：
   \[
   \|\widehat B^{(\mathrm{outer})}p\|
   \le
   \eta
   \max\left(\|B^{(L)}p\|,\|p\|\right);
   \]
4. 对 SR1 denominator 使用严格阈值；
5. 可选地每隔若干 accepted iterations 做一次 single-direction exact outer probe，用来校准而不是每次 HVP 都算。

当前实验实现使用以下环境变量开关：

- `XMVB_CPP_EXACT_CTX_SR1_OUTER_APPROX=1`：启用 SR1 outer-response 近似；
- `XMVB_CPP_EXACT_CTX_SR1_OUTER_HISTORY_SIZE`：历史长度，默认 `6`；
- `XMVB_CPP_EXACT_CTX_SR1_OUTER_MIN_ALIGNMENT`：SR1 denominator 对齐阈值，默认 `1e-8`；
- `XMVB_CPP_EXACT_CTX_SR1_OUTER_MAX_CORRECTION_RATIO`：修正向量相对 core HVP / 方向范数的上限，默认 `0.5`；
- `XMVB_CPP_EXACT_CTX_SR1_OUTER_CLEAR_TRUST_RATIO`：accepted trust ratio 低于该值时清空 SR1 历史，默认 `0.35`。

实现上该路径默认关闭，并且只在 exact_ctx cheap/core-only inner solve 上叠加 SR1 correction。若一次 trial 被拒绝，会清空 SR1 历史和当前 accepted point 的 cheap Krylov cache，避免继续复用已经证明不可靠的近似曲率。

### 4.6 适用性判断

这是最适合作为第一版近似的方案：

- 优点：几乎没有额外 kernel 成本，直接服务 TNHVP；
- 缺点：它是历史 secant 模型，不能保证每个新方向都准确；
- 风险：如果 accepted steps 很少或方向变化很快，低秩模型可能滞后。

### 4.7 初步 241_VBSCF 结果

在 `6526Y` 上对 `test/241_VBSCF.xmi` 做了第一轮 A/B 测试：

| 模式 | Slurm job | accepted iterations | SCF iteration wall time | 现象 |
| --- | ---: | ---: | ---: | --- |
| baseline | `1943681` | 8 | `14.763475 s` | 默认策略，前两步 full outer-response，后续 core-only |
| SR1 初版 | `1943682` | 11 | `21.617981 s` | SR1 correction 过激，早期 trust radius 被打小 |
| SR1 + conservative cap | `1943684` | 10 | `26.459419 s` | 仍触发多次 rejected/full-probe 路径 |
| SR1 + rejection/low-trust clear | `1943685` | 8 | `18.555064 s` | 收敛恢复，但仍慢于 baseline |

这个结果说明：**reduced-space missing-curvature SR1 虽然便宜，但第一版并没有在 241 上改善总时间**。它的主要问题不是单次 HVP 成本，而是历史 secant correction 改变了 trust-region 模型，导致额外 rejected steps、full probes 和半径收缩。

因此，若继续这条线，应先把它定位为诊断/校准模型，而不是默认优化路径。下一步更合理的尝试是：

1. 只在 tail 区域启用 SR1，而不是前两次 full startup 后立刻启用；
2. 只使用 single accepted step 的 rank-1 correction，并在每个 accepted point 后重新筛选；
3. 用 exact outer-response probe 校准 correction 符号和尺度，而不是完全依赖 relaxed gradient secant；
4. 或转向 local-only / energy-only 解析近似，判断 selected-state coefficient response 是否才是需要截断的部分。

## 5. 策略 B：标量或低秩对角修正

这是策略 A 的更便宜版本。

直接使用

\[
\widehat B_k^{(\mathrm{outer})}
=
\alpha_k I
\]

或在某个低维子空间 \(Q_k\) 上使用

\[
\widehat B_k^{(\mathrm{outer})}
=
Q_k \Theta_k Q_k^\mathsf{T}.
\]

若只用最近一步估计标量，

\[
\alpha_k
=
\frac{s_{k-1}^\mathsf{T}r_{k-1}}
       {s_{k-1}^\mathsf{T}s_{k-1}}.
\]

若保留 \(m\) 个正交历史方向

\[
Q_k = [q_1,\ldots,q_m],
\]

则可以拟合

\[
\Theta_k
=
Q_k^\mathsf{T} R_k,
\]

其中 \(R_k\) 是历史 missing-curvature responses 在该子空间上的表示。

这类方法非常便宜，但表达力有限。它适合作为 regularization 或 fallback，不适合作为最终的高精度 outer-response 替代。

## 6. 策略 C：解析链路的 local-only / energy-only 近似

这类方案不依赖历史 secant，而是直接截断 exact outer-response 链路。

Exact active-space outer response 可写成

\[
g_{\mathrm{act}}^{(\mathrm{outer})}
=
W^{(\mathrm{loc})}u
+
W^{(\mathrm{dir})}
\begin{bmatrix}
\delta C_{\mathrm{sel}}\\
\delta E_{\mathrm{sel}}
\end{bmatrix},
\qquad
u=T p.
\]

### 6.1 Local-only outer-response

最简单的解析近似是丢掉 selected-state 系数和能量响应：

\[
\delta C_{\mathrm{sel}} \leftarrow 0,
\qquad
\delta E_{\mathrm{sel}} \leftarrow 0.
\]

于是

\[
\widehat B^{(\mathrm{outer,loc})}p
=
T^\sharp W^{(\mathrm{loc})} T p.
\]

它避免了：

- projected structure matrices；
- generalized-eigen response；
- directional selected-state determinant matrices；
- directional same/opposite-spin backward contribution。

但是它仍需要 \(T p\)，尤其仍可能需要 directional exact two-electron integral
\(\delta g_{\mathrm{act}}^{(2)}\)。因此它不是零成本近似，而是“只保留局部 outer-response”的解析降阶模型。

当前实验实现用环境变量打开：

```bash
XMVB_CPP_EXACT_CTX_OUTER_RESPONSE_APPROX=local_only
```

默认值仍是 full/exact。实现位置是
[`src/vb/scf/exact_orbital_second_order_operator.cpp`](../src/vb/scf/exact_orbital_second_order_operator.cpp)：
在构造完 active-space 方向积分 \(u=T p\) 后，`local_only` 路径直接调用
`build_local_active_space_gradient_direction_from_outer_response(...)`，只保留 local
same-spin / opposite-spin matrix backward contribution，并跳过：

\[
P_{k,\mathrm{sel}}T_kp,
\qquad
R_{k,\mathrm{sel}}P_{k,\mathrm{sel}}T_kp,
\qquad
W^{(\mathrm{dir})}
\begin{bmatrix}
\delta C_{\mathrm{sel}}\\
\delta E_{\mathrm{sel}}
\end{bmatrix}.
\]

也就是说当前实现实际计算的是

\[
\widehat B^{(\mathrm{outer,loc})}p
=
T^\sharp W^{(\mathrm{loc})}T p,
\]

而 full 路径仍计算

\[
B^{(\mathrm{outer})}p
=
T^\sharp
\left(
W^{(\mathrm{loc})}
+
W^{(\mathrm{dir})}R_{k,\mathrm{sel}}P_{k,\mathrm{sel}}
\right)
T p.
\]

### 6.1.1 Local-only 的 241_VBSCF 实测

在 `6526Y` 上用 32 线程测试 `test/241_VBSCF.xmi`。第一组使用默认 hybrid
TNHVP 策略，即前几步允许 full outer-response，之后回到 core-only：

| 模式 | Slurm job | accepted iterations | SCF iteration wall time | weighted full-outer apply | weighted outer stage | 现象 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full/auto | `1943691` | 8 | `15.004738 s` | `0.3367 s` | `0.1750 s` | baseline |
| local-only/auto | `1943692` | 10 | `14.988757 s` | `0.2663 s` | `0.1149 s` | 单次 full outer HVP 便宜约三分之一，但多走了 2 次 accepted iteration 和一次 late full retry |

第二组强制 inner solve 一直包含 outer-response：

| 模式 | Slurm job | accepted iterations | SCF iteration wall time | weighted outer stage | 现象 |
| --- | ---: | ---: | ---: | ---: | --- |
| full/on | `1943694` | 8 | `18.712990 s` | `0.1687 s` | 收敛路径稳定 |
| local-only/on | `1943695` | 17 | `25.586039 s` | 约 `0.106 s` | 单次 HVP 更便宜，但 trust radius 多次收缩/拒绝，整体显著变慢 |

结论：local-only 解析截断 **能按预期降低单次 outer-response apply 成本**，但在
`241_VBSCF` 上缺失的 selected-state coefficient / energy response 对 TNHVP 模型质量很重要。
因此它不适合作为直接替代 full outer-response 的默认优化策略。它更适合做诊断开关，用来量化

\[
W^{(\mathrm{dir})}R_{k,\mathrm{sel}}P_{k,\mathrm{sel}}T_kp
\]

这条链路对收敛的贡献；若要继续做解析截断，应优先尝试保留 selected-state 的能量响应或近简并
root 子空间，而不是完全丢掉 selected-state response。

### 6.2 Energy-only outer-response

保留 selected-state 能量响应，但忽略 selected-state 系数响应：

\[
\delta C_{\mathrm{sel}} \leftarrow 0,
\qquad
\delta E_i =
e_i^\mathsf{T}
\left(
\delta \widetilde h_i
-
\varepsilon_i \delta \widetilde s_i
\right).
\]

此时 determinant pair weights 近似为

\[
\delta w_{mn}^{(H)} \approx 0,
\]

\[
\delta w_{mn}^{(S)}
\approx
-
\sum_{i\in I_{\mathrm{sel}}}
\omega_i
\delta E_i
z_{im}z_{in}.
\]

这比 full eigenvector response 便宜，因为不需要所有

\[
(\varepsilon_i-\varepsilon_j)^{-1}
\]

耦合项。真正省成本的前提是实现中能只构造 selected diagonal 所需的 projected structure entries；如果仍然用当前 full selected-column builder 生成全部 rows，那么性能收益会被抵消。

当前实验实现用环境变量打开：

```bash
XMVB_CPP_EXACT_CTX_OUTER_RESPONSE_APPROX=energy_only
```

实现上仍构造完整 projected directional structure columns，但只读取 selected diagonal 来计算
\(\delta E_i\)，不再计算 \(\delta C_{\mathrm{sel}}\)。active-space backward 使用零
directional selected-state coefficients，因此只保留 same-spin overlap-weight 中的
\(-\omega_i\delta E_i c_i c_i^\mathsf{T}\) 项。

### 6.2.1 Energy-only 的 241_VBSCF 实测

在 `6526Y` 上用 32 线程测试 `test/241_VBSCF.xmi`：

| 模式 | Slurm job | accepted iterations | SCF iteration wall time | outer stage | 现象 |
| --- | ---: | ---: | ---: | ---: | --- |
| full/auto | `1943691` | 8 | `15.004738 s` | `0.1750 s` | baseline |
| energy-only/auto | `1943697` | 12 | `16.165263 s` | `0.1683 s` / `0.1760 s` startup, full retry `0.1550 s` | 比 local-only 更接近 full 成本，但仍多走 4 次 accepted iteration |
| full/on | `1943694` | 8 | `18.712990 s` | `0.1687 s` | 强制 full outer-response |
| energy-only/on | `1943698` | 13 | `21.647043 s` | 约 `0.16 s` | 单次成本没有明显下降，且模型质量仍低于 full |

结论：energy-only 保留的 \(\delta E_i\) 对模型质量有帮助，但不足以替代
\(\delta C_{\mathrm{sel}}\)。更关键的是，当前实现仍完整构造
\(P_{k,\mathrm{sel}}T_kp\)，所以单次 outer-response 成本接近 full。若要让
energy-only 真正便宜，必须新增只计算 selected diagonal 的 projected-structure builder：

\[
\delta \widetilde h_{ii},
\quad
\delta \widetilde s_{ii},
\qquad i\in I_{\mathrm{sel}},
\]

而不是生成所有

\[
\delta \widetilde H_{j i},
\quad
\delta \widetilde S_{j i},
\qquad
j=1,\ldots,n_{\mathrm{str}}.
\]

在当前代码形态下，energy-only 不值得作为优化策略继续推进；它主要确认了
selected-state coefficient response 仍是 241 上 TNHVP 曲率质量的重要组成部分。

### 6.3 适用性判断

local-only 和 energy-only 是 deterministic approximation：

- 优点：不依赖历史步，行为可重复；
- 优点：比 full selected-state eigenvector response 更容易解释；
- 缺点：仍可能保留 \(\delta GGO\) 和 orbital pullback 成本；
- 缺点：忽略 \(\delta C_{\mathrm{sel}}\) 可能显著低估近简并结构响应。

它们适合作为第二优先级实现，尤其适合回答“outer-response 中到底哪部分对收敛最有贡献”。

## 7. 策略 D：截断 selected-root eigenvector response

Full selected-state 响应的贵处在于每个 selected state \(i\) 都遍历所有 root \(j\)：

\[
(\delta \Omega_i)_j
=
\frac{
(\delta \widetilde h_i)_j
-
\varepsilon_i(\delta \widetilde s_i)_j
}
{\varepsilon_i-\varepsilon_j}.
\]

可用 mask \(\chi_{ji}\in\{0,1\}\) 做截断：

\[
(\delta \Omega_i)_j
\approx
\chi_{ji}
\frac{
(\delta \widetilde h_i)_j
-
\varepsilon_i(\delta \widetilde s_i)_j
}
{\varepsilon_i-\varepsilon_j},
\qquad j\ne i.
\]

一个合理的 root 集合是

\[
\mathcal{J}_i
=
\{i\}
\cup
\{j:\ |\varepsilon_i-\varepsilon_j|\le \Gamma\}
\cup
\mathcal{J}_{i,\mathrm{sel}}
\cup
\mathcal{J}_{i,\mathrm{top}}.
\]

其中：

- \(\Gamma\) 是 gap cutoff；
- \(\mathcal{J}_{i,\mathrm{sel}}\) 包含其他 selected roots；
- \(\mathcal{J}_{i,\mathrm{top}}\) 可由历史 coupling magnitude 或 structure weight 选出。

为了稳定近简并项，可以使用 damped denominator：

\[
\frac{1}{\Delta_{ji}}
\quad\longrightarrow\quad
\frac{\Delta_{ji}}{\Delta_{ji}^2+\mu^2},
\qquad
\Delta_{ji}=\varepsilon_i-\varepsilon_j.
\]

注意：damping 本身主要提高稳定性，不一定省时间。只有当 projected structure builder 也能按 \(\mathcal{J}_i\) 的 row subset 构造时，这个近似才会真正便宜。

适用性：

- 优点：保留近简并和 selected-root 的主要响应；
- 优点：比完全忽略 \(\delta C_{\mathrm{sel}}\) 更准确；
- 缺点：需要 row-subset projected structure builder，否则不会明显省成本；
- 风险：large-gap roots 数量多时，单个项小但总和未必可忽略，需要 benchmark 量化。

## 8. 策略 E：偶尔 exact probe 的低秩投影模型

如果允许少量 exact outer-response 采样，但不想每次 HVP 都直接计算它，可以构造低秩投影模型。

选取少量方向

\[
V=[v_1,\ldots,v_q],
\qquad
V^\mathsf{T}V=I,
\]

只在这些方向上计算 exact outer-response：

\[
C = B^{(\mathrm{outer})}V.
\]

投影 Hessian 为

\[
H = V^\mathsf{T}C.
\]

最保守的子空间近似是

\[
\widehat B^{(\mathrm{outer})}
=
V\,\operatorname{sym}(H)\,V^\mathsf{T},
\]

即

\[
\widehat B^{(\mathrm{outer})}p
=
V\,\operatorname{sym}(H)\,(V^\mathsf{T}p).
\]

若 \(B^{(\mathrm{outer})}\) 在采样空间上接近正定，也可以考虑 Nystrom 型近似：

\[
\widehat B^{(\mathrm{outer})}
=
C\,(H+\lambda I)^\dagger C^\mathsf{T}.
\]

但 outer-response 未必正定，所以 Nystrom 版本需要更强 safeguard。

这条路线适合：

- 每隔若干 accepted iterations 采样一次；
- 只采样当前 TNHVP 的少量关键方向，例如 accepted step、gradient direction、负曲率方向；
- 将采样结果和策略 A 的 secant history 合并。

不建议把整个 cheap Krylov basis 都逐列 exact outer probe 一遍，因为那会退回到

\[
q \cdot \operatorname{cost}(B^{(\mathrm{outer})}p)
\]

的成本模型，之前的 MFTR 分析已经说明这很难回本。

## 9. 策略 F：active-space level 的 diagonal / block-diagonal \(K\) 近似

从分解

\[
B^{(\mathrm{outer})}
=
T^\sharp K^{(\mathrm{outer})}T
\]

出发，可以直接近似 active-space operator：

\[
K^{(\mathrm{outer})}
\approx
\widehat K
=
\operatorname{blockdiag}(D_S,D_h,D_g).
\]

于是

\[
\widehat B^{(\mathrm{outer})}p
=
T^\sharp \widehat K T p.
\]

如果能拿到 active-space level 的 secant pair

\[
u_j = T s_j,
\qquad
a_j \approx K^{(\mathrm{outer})}u_j,
\]

则 diagonal \(D\) 可用 ridge regression 拟合：

\[
d_\ell
=
\frac{
\sum_j (a_j)_\ell (u_j)_\ell
}{
\lambda + \sum_j (u_j)_\ell^2
}.
\]

这个模型比 reduced-space SR1 更贴近 outer-response 结构，但当前实现还没有直接暴露 \(a_j\)。如果只从 reduced gradient secant 反推，会多一个 \(T^\sharp\) 的不可逆问题。因此它更适合作为后续重构 active-space block operator 后的方案，不建议作为第一版。

## 10. 不建议优先做的方案

### 10.1 逐列 full outer-response 的 MFTR

若为了构造 projected high-fidelity model 而逐列计算

\[
B^{(\mathrm{outer})}q_1,\ldots,B^{(\mathrm{outer})}q_m,
\]

则成本近似是

\[
m\,C_{\mathrm{outer}}.
\]

这不是“避免直接计算 outer-response”，只是把直接计算做了多次。除非 outer-response 已经被 block 化，否则不建议继续把它作为主要路线。

### 10.2 只做 denominator damping

将

\[
\frac{1}{\varepsilon_i-\varepsilon_j}
\]

改成 damped denominator 可以提高稳定性，但如果仍遍历全部 \(j\)，它不降低复杂度。它应和 root truncation 一起使用，而不是单独作为性能优化。

### 10.3 完全关闭 outer-response 作为最终方案

\[
\widehat B^{(\mathrm{outer})}=0
\]

就是当前 cheap/core-only 模型。它是必要 baseline，但从已有 TNHVP 行为看，某些体系会因此出现模型 mismatch、tail 收敛慢或 retry 变多。它可以作为 fallback，不应视为最终近似策略。

## 11. 推荐落地顺序

### 第一优先级：missing-curvature L-SR1

实现内容：

1. 在 accepted TNHVP step 后形成
   \[
   r_j = (g_{j+1}-g_j) - B_j^{(L)}s_j;
   \]
2. 用 limited-memory SR1 构造 \(\widehat B^{(\mathrm{outer})}\)；
3. 内层 HVP 使用
   \[
   B^{(L)}p+\widehat B^{(\mathrm{outer})}p;
   \]
4. 通过 trust ratio、denominator check、dimension/state reset 做 safeguard。

这是最符合“避免直接计算 outer-response”的方案。

### 第二优先级：local-only / energy-only analytical approximation

实现内容：

1. 增加 outer-response approximation mode；
2. 支持
   \[
   W^{(\mathrm{loc})}T p
   \]
   的 local-only 路径；
3. 可选支持 energy-only \(\delta E_i\)；
4. benchmark 它们相对 full outer-response 的 HVP 误差和 TNHVP 迭代数。

这条路线适合判断 selected-state coefficient response 到底有多重要。
`241_VBSCF` 的结果显示它们不适合作为当前默认优化路径：local-only 虽然单次 HVP
更便宜，但模型太弱；energy-only 保留了更多曲率，但由于仍构造完整 projected
structure columns，成本接近 full，同时收敛仍变差。

### 第三优先级：truncated selected-root response

实现前提：

1. projected structure builder 支持 row subset；
2. generalized-eigen response operator 支持 \(\mathcal{J}_i\) mask；
3. 误差监控能比较 truncated 和 occasional exact probe。

如果没有 row-subset builder，这个方案不会真正便宜。

### 第四优先级：少量 exact probe 的低秩模型

它适合作为策略 A 的校准工具，而不是每轮都做的主路径。

## 12. 总结

最值得优先尝试的近似是：

\[
\boxed{
\widehat B_k
=
B_k^{(L)}
+
M_k^{(\mathrm{SR1})}
}
\]

其中 \(M_k^{(\mathrm{SR1})}\) 用 accepted steps 的 missing-curvature secants 拟合：

\[
r_j
=
(g_{j+1}-g_j)
-
B_j^{(L)}s_j.
\]

它的优势是几乎不增加 HVP kernel 成本，且直接补偿 cheap/core-only 模型缺失的 relaxed response 曲率。

若希望保留更多解析物理结构，则第二条路线曾是：

\[
\widehat B^{(\mathrm{outer})}p
=
T^\sharp
\left(
W^{(\mathrm{loc})}
+
W^{(\mathrm{energy})}
\right)
T p,
\]

即 local-only 或 energy-only outer-response。它比 SR1 更可解释，但仍可能保留 \(\delta GGO\) 和 orbital pullback 成本，因此性能收益需要通过 `241_VBSCF`、`MnF2` 和更大 deck 实测。

当前 `241_VBSCF` 实测后，local-only 和 energy-only 都不能作为默认替代。下一条真正有工程价值的解析近似应转向
**selected-root response truncation with row-subset projected structure builder**，否则只在响应公式后半段截断，不会切掉足够多的成本。

不建议优先做逐列 full outer-response projection。那不是近似，只是把 exact outer-response 的成本搬到了另一个位置。
