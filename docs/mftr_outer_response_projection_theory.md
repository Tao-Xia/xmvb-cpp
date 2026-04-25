# MFTR 在当前 Outer-Response 公式下的可行性判断

## 1. 目标

这里不讨论任何启发式策略，只回答一个更根本的问题：

\[
\text{在当前 exact\_ctx outer-response 公式下，}
\quad
Q^\top B_{\mathrm{outer}} Q
\quad
\text{是否存在比逐列 outer-response HVP 更本质的构造方式？}
\]

其中

\[
B_H = B_L + B_{\mathrm{outer}}
\]

分别表示：

- \(B_L\): 当前 TNHVP 的 cheap/core-only 二阶模型；
- \(B_H\): 含 outer-response 的 full 二阶模型；
- \(B_{\mathrm{outer}}\): high-fidelity 修正项。

若答案为否，则系统化 MFTR 基本没有继续实现的价值；若答案为是，则应把后续工作重心转向 outer-response 的结构化块实现，而不是继续修改 trust-region 外壳。

## 2. 当前代码中的 outer-response 数据流

在 accepted point 固定时，当前代码中的 outer-response 对一个 reduced-space 方向
\[
s \in \mathbb{R}^{n_r}
\]
执行以下链式线性映射：

\[
s
\xrightarrow{\,T\,}
u
\xrightarrow{\,P_{\mathrm{sel}}\,}
x
\xrightarrow{\,R_{\mathrm{sel}}\,}
y
\xrightarrow{\,W\,}
g_{\mathrm{act}}
\xrightarrow{\,T^\ast\,}
B_{\mathrm{outer}} s.
\]

各层含义如下。

### 2.1 轨道方向到活性空间方向量

记
\[
u = T s.
\]

这里 \(u\) 表示 active-space directional data，即当前代码中 outer-response 真正消费的三组量：

\[
u =
\left(
\delta S_{\mathrm{act}},
\delta H_{\mathrm{act}},
\delta G_{\mathrm{act}}
\right).
\]

对应实现入口是
[exact_orbital_second_order_operator.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/exact_orbital_second_order_operator.cpp#L8937)
中的
`build_active_space_directional_integrals(...)`。

若定义

\[
d_S = n_a^2, \qquad
d_H = n_a^2, \qquad
d_G = n_{2e}^{\mathrm{pack}},
\]

则可把 \(u\) 视为一个总维数为

\[
d_{\mathrm{act}} = d_S + d_H + d_G
\]

的向量。

这一层已经说明，outer-response 并不是直接在 reduced 轨道空间里工作的，而是先进入一个更小的 active-space directional space。

### 2.2 活性空间方向量到 selected-state 投影结构响应

代码不会显式构造完整的
\[
U^\top \delta H_{\mathrm{str}} U,
\qquad
U^\top \delta S_{\mathrm{str}} U
\]
的全态矩阵，而是直接构造只服务于 selected states 的投影块：

\[
x = P_{\mathrm{sel}} u
=
\left(
\delta \widetilde H_{\mathrm{sel}},
\delta \widetilde S_{\mathrm{sel}}
\right),
\]

其中

\[
\delta \widetilde H_{\mathrm{sel}},
\delta \widetilde S_{\mathrm{sel}}
\in
\mathbb{R}^{n_{\mathrm{str}} \times n_{\mathrm{sel}}}.
\]

对应实现见
[exact_orbital_second_order_operator.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/exact_orbital_second_order_operator.cpp#L6580)
和
[exact_orbital_second_order_operator.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/exact_orbital_second_order_operator.cpp#L6646)。

因此，当前 outer-response 已经天然把 full structure response 压缩到了
\[
2 n_{\mathrm{str}} n_{\mathrm{sel}}
\]
的中间表示上。

### 2.3 selected-state 广义本征响应

随后代码对每个 selected state \(i\) 构造一阶本征值和本征矢响应：

\[
y = R_{\mathrm{sel}} x
=
\left(
\delta C_{\mathrm{sel}},
\delta E_{\mathrm{sel}}
\right),
\]

其中

\[
\delta C_{\mathrm{sel}} \in \mathbb{R}^{n_{\mathrm{str}} \times n_{\mathrm{sel}}},
\qquad
\delta E_{\mathrm{sel}} \in \mathbb{R}^{n_{\mathrm{sel}}}.
\]

对应实现见
[exact_orbital_second_order_operator.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/exact_orbital_second_order_operator.cpp#L7151)。

这里最重要的一点是：虽然最终只保留 selected-state 列，但每个 selected state 的一阶本征矢响应仍然与全部 structure-space roots 耦合，即
\[
j = 1,\dots,n_{\mathrm{str}}
\]
仍完整出现于分母
\[
(\varepsilon_i - \varepsilon_j)^{-1}
\]
中。这保证了公式正确，但也意味着这里只压缩了输出列数，并没有消掉全态耦合。

### 2.4 selected-state 响应到 active-space 梯度方向

当前代码把 active-space outer-response 进一步拆成两部分：

\[
g_{\mathrm{act}} = W_{\mathrm{loc}} u + W_{\mathrm{dir}} y.
\]

其中：

- \(W_{\mathrm{loc}} u\) 是 accepted selected states 固定时，由
  \((\delta S_{\mathrm{act}}, \delta H_{\mathrm{act}}, \delta G_{\mathrm{act}})\)
  直接产生的 local outer-response；
- \(W_{\mathrm{dir}} y\) 是 selected-state coefficient / energy 响应引起的 directional outer-response。

对应实现见
[exact_orbital_second_order_operator.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/exact_orbital_second_order_operator.cpp#L7896)。

这一步输出的是 active-space gradient direction：

\[
g_{\mathrm{act}}
\in
\mathbb{R}^{d_{\mathrm{act}}}.
\]

注意这里 again 表明，outer-response 的核心高保真修正完全停留在 active-space level，并未立即回到 reduced orbital space。

### 2.5 active-space 梯度方向回拉到 reduced orbital space

最后，代码使用 orbital pullback 和 reduced projection 把
\[
g_{\mathrm{act}}
\]
映射回 reduced-space HVP 输出。对应实现见
[exact_orbital_second_order_operator.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/exact_orbital_second_order_operator.cpp#L8010)
以及后续 reduced projection。

这一步记为

\[
T^\ast g_{\mathrm{act}}.
\]

这里的 \(T^\ast\) 不必强行理解为普通欧氏转置；更准确地说，它是当前代码中与 \(T\) 配对的 accepted-point pullback / adjoint。

## 3. Outer-response 的算子分解

由上面的链式结构，

\[
x = P_{\mathrm{sel}} u,
\qquad
y = R_{\mathrm{sel}} x,
\qquad
g_{\mathrm{act}} = W_{\mathrm{loc}} u + W_{\mathrm{dir}} y.
\]

消去中间变量可得

\[
g_{\mathrm{act}}
=
\left(
W_{\mathrm{loc}} + W_{\mathrm{dir}} R_{\mathrm{sel}} P_{\mathrm{sel}}
\right) u.
\]

记

\[
K_{\mathrm{outer}}
=
W_{\mathrm{loc}} + W_{\mathrm{dir}} R_{\mathrm{sel}} P_{\mathrm{sel}},
\]

则

\[
g_{\mathrm{act}} = K_{\mathrm{outer}} u,
\qquad
u = T s.
\]

因此

\[
B_{\mathrm{outer}} s
=
T^\ast K_{\mathrm{outer}} T s.
\]

也就是说，在当前代码语义下，

\[
\boxed{
B_{\mathrm{outer}} = T^\ast K_{\mathrm{outer}} T
}
\]

是一个严格成立的 accepted-point 算子分解。

这就是判断 MFTR 是否有理论希望的关键。

## 4. 对 MFTR 的直接含义

设 low-fidelity Krylov 子空间为

\[
Q = [q_1,\dots,q_m] \in \mathbb{R}^{n_r \times m},
\qquad
Q^\top Q = I.
\]

则系统化 MFTR 需要的 high-fidelity 修正块为

\[
Q^\top B_{\mathrm{outer}} Q.
\]

由上一节分解，

\[
Q^\top B_{\mathrm{outer}} Q
=
Q^\top T^\ast K_{\mathrm{outer}} T Q.
\]

记

\[
Z = T Q \in \mathbb{R}^{d_{\mathrm{act}} \times m},
\]

则

\[
\boxed{
Q^\top B_{\mathrm{outer}} Q = Z^\ast K_{\mathrm{outer}} Z
}
\]

其中 \(Z^\ast\) 表示与 \(T^\ast\) 配对的 active-space dual pairing。在代码当前的打平存储约定下，可把它理解为与 active-space directional coordinates 一致的双线性配对。

这一步非常关键，因为它表明：

1. MFTR 并不一定要通过逐列构造 reduced-space 向量
   \[
   B_{\mathrm{outer}} q_i
   \]
   才能得到 projected high-fidelity Hessian；
2. 理论上完全可以先把 Krylov block 映射到 active-space directional block \(Z\)，再在 active-space / structure / selected-state level 完成 high-fidelity 投影。

因此，MFTR 在理论上 **不是死路**。

## 5. 当前 projected MFTR 为什么仍然失败

虽然上一节给出了 exact 的结构化表示，但当前实现并没有利用这个结构。

目前实现做的仍然是：

\[
q_i
\longmapsto
B_{\mathrm{outer}} q_i
\]

逐列调用，然后再拼出

\[
Q^\top B_{\mathrm{outer}} Q.
\]

这相当于把 \(T\)、\(P_{\mathrm{sel}}\)、\(R_{\mathrm{sel}}\)、\(W\)、\(T^\ast\) 全部重新执行 \(m\) 次。于是 high-fidelity projection 的代价仍近似为

\[
m \cdot \mathrm{cost}(B_{\mathrm{outer}} q).
\]

因此现在这版 projected MFTR 的失败，不是因为 MFTR 理论不成立，而是因为实现上仍把 outer-response 当成了一个逐方向黑盒 HVP。

## 6. 真正有希望的 MFTR 需要什么

若要让 MFTR 真正转正，至少要做到下面这件事中的一种。

### 6.1 在 active-space level 上做 block 传播

先构造

\[
Z = T Q
\]

的 block 形式，然后一次性传播：

\[
Z
\xrightarrow{\,P_{\mathrm{sel}}\,}
X
\xrightarrow{\,R_{\mathrm{sel}}\,}
Y
\xrightarrow{\,W\,}
G,
\]

其中

\[
X \in \mathbb{R}^{(2 n_{\mathrm{str}} n_{\mathrm{sel}}) \times m},
\qquad
Y \in \mathbb{R}^{(n_{\mathrm{str}} n_{\mathrm{sel}} + n_{\mathrm{sel}}) \times m},
\qquad
G \in \mathbb{R}^{d_{\mathrm{act}} \times m}.
\]

然后直接形成

\[
Q^\top B_{\mathrm{outer}} Q = Z^\ast G.
\]

这条路线的核心是：不再对每一列都执行最终的 orbital pullback 和 reduced projection。

### 6.2 显式或隐式构造 active-space level 的 \(K_{\mathrm{outer}}\)

如果 \(d_{\mathrm{act}}\) 足够小，则存在第二种更激进的可能：

\[
B_{\mathrm{outer}} = T^\ast K_{\mathrm{outer}} T
\]

中的
\[
K_{\mathrm{outer}} \in \mathbb{R}^{d_{\mathrm{act}} \times d_{\mathrm{act}}}
\]
在 accepted point 上也许可以显式或半显式组装。

若这一点成立，则

\[
Q^\top B_{\mathrm{outer}} Q
=
Z^\ast K_{\mathrm{outer}} Z
\]

将退化为一个小维 dense operator 的块二次型计算。

但这条路线是否值得，取决于：

1. \(d_{\mathrm{act}}\) 的实际大小；
2. \(K_{\mathrm{outer}}\) 的组装是否比 block-apply 更便宜；
3. selected-state eigresponse 与 matrix-form backward 是否允许在 basis directions 上高效重用。

从当前代码状态看，这条路线还只是理论可能性，不能直接假定它更优。

## 7. 目前的理论结论

结论可以明确写成三条。

### 7.1 MFTR 不是理论死路

当前 outer-response 可以严格写成

\[
B_{\mathrm{outer}} = T^\ast K_{\mathrm{outer}} T,
\]

因此

\[
Q^\top B_{\mathrm{outer}} Q
\]

确实存在一个 active-space / selected-state 中间层上的结构化表示。

### 7.2 当前实现路线基本没有希望

只要 high-fidelity correction 仍然通过逐列 reduced-space outer-response HVP 来得到，

\[
Q^\top B_{\mathrm{outer}} Q
\]

就不会比“多做若干次 full HVP”本质更便宜。

因此，沿着当前这版 projected MFTR 再加 optimizer 层技巧，成功概率很低。

### 7.3 真正值得做的是 outer-response 的块化，而不是 trust-region 外壳

若后续继续投入 MFTR，正确方向不是再调：

- trust-radius 规则；
- retry / followup 策略；
- mixed-fidelity 接受判据；

而是直接实现：

\[
Z = TQ,
\qquad
X = P_{\mathrm{sel}} Z,
\qquad
Y = R_{\mathrm{sel}} X,
\qquad
G = K_{\mathrm{outer}} Z,
\qquad
Q^\top B_{\mathrm{outer}} Q = Z^\ast G.
\]

只有做到这一步，MFTR 才有现实希望。

## 8. 对后续实现的建议

若要继续推进，推荐按下面顺序做。

1. 不再从 reduced-space outer-response matvec 出发设计 MFTR。
2. 先把 \(T\) 的 block 版本明确下来，也就是 one-shot 生成 \(Z = TQ\)。
3. 再把 `build_selected_state_projected_directional_structure_matrices(...)` 改成真正消费 block \(Z\) 的接口，而不是每列独立跑一遍。
4. 再把 `build_selected_state_generalized_eigen_directional_response(...)` 改成 block 版本，输入为 \(X\)，输出为 \(Y\)。
5. 最后才回到 optimizer 中实现
   \[
   Q^\top B_H Q
   =
   Q^\top B_L Q + Q^\top B_{\mathrm{outer}} Q.
   \]

在这之前，继续在 optimizer 外壳上折腾 MFTR，意义不大。

## 9. 最终判断

\[
\boxed{
\text{MFTR 仍然有希望，但前提是 outer-response 先被改造成结构化 block 算子。}
}
\]

\[
\boxed{
\text{如果 outer-response 继续作为逐方向黑盒 HVP 使用，则当前这条 MFTR 路线基本没有希望。}
}
\]

## 10. MFTR 的理论 ceiling 有多高

上一节说明了 MFTR 是否有理论希望；这一节进一步回答另一个更实际的问题：

\[
\text{即使 outer-response 被块化，MFTR 的理论上限到底有多高？}
\]

这里的“上限”不是指严格数学最优值，而是指在当前代码的数据流和主热点不变的前提下，MFTR 相对现有 TNHVP 所能达到的最佳量级。

### 10.1 当前代码与理想 MFTR 的成本模型

记当前 TNHVP 一次 accepted iteration 的主成本为

\[
C_0 = C_{\mathrm{obj}} + m C_L,
\]

其中

- \(C_{\mathrm{obj}}\): 一次 trial objective evaluation 的成本；
- \(C_L\): 一次 cheap/core HVP 的成本；
- \(m\): cheap Krylov 子空间维数。

若 MFTR 使用一个额外的 high-fidelity 投影修正，则每次 accepted iteration 的成本变为

\[
C_1 = C_{\mathrm{obj}} + m C_L + C_{\Delta,Q},
\]

其中

\[
C_{\Delta,Q}
\approx
\text{构造 }
Q^\top B_{\mathrm{outer}} Q
\text{ 的额外成本。}
\]

若当前 TNHVP 与 MFTR 的 accepted iteration 数分别为 \(N_0\) 与 \(N_1\)，则总 wall time 满足

\[
W_0 \approx N_0 C_0,
\qquad
W_1 \approx N_1 C_1.
\]

因此 MFTR 相对当前代码的总加速比为

\[
S_{\mathrm{tot}}
=
\frac{W_0}{W_1}
=
\frac{N_0}{N_1}
\cdot
\frac{C_0}{C_0 + C_{\Delta,Q}}.
\]

这条式子给出了两个完全不同的门槛：

1. MFTR 必须显著减少 accepted iterations，即 \(N_1 < N_0\)；
2. 即使迭代数减少，若 \(C_{\Delta,Q}\) 仍然过大，则 wall time 依然可能恶化。

### 10.2 若继续逐列 outer-response，ceiling 几乎为零

如果仍然逐列构造

\[
B_{\mathrm{outer}} q_1,\dots,B_{\mathrm{outer}} q_m,
\]

则

\[
C_{\Delta,Q}^{\mathrm{scalar}}
\approx
m \, C_{\mathrm{outer}}.
\]

这意味着 MFTR 相当于在每个 accepted point 上又额外做了一轮“接近 full outer-response 内层求解”的工作。此时哪怕 \(N_1 < N_0\)，也必须有非常夸张的迭代数下降才可能回本，因此这条实现路线基本没有现实希望。

这正是当前 projected MFTR 原型失败的根本原因。

### 10.3 理想 block 实现的 ceiling 来自哪些层

当前文档前半部分把

\[
B_{\mathrm{outer}} = T^\ast K_{\mathrm{outer}} T
\]

分解成了五层：

\[
T,
\quad
P_{\mathrm{sel}},
\quad
R_{\mathrm{sel}},
\quad
W,
\quad
T^\ast.
\]

对 Krylov block \(Q\) 的理想块传播成本可记为

\[
C_{\Delta,Q}^{\mathrm{block}}
=
C_T^{\mathrm{block}}(m)
+
C_P^{\mathrm{block}}(m)
+
C_R^{\mathrm{block}}(m)
+
C_W^{\mathrm{block}}(m)
+
C_{T^\ast}^{\mathrm{block}}(m).
\]

关键是，这五层的可压缩程度并不相同。

#### 10.3.1 \(T\) 与 \(T^\ast\)：只能吃到有限常数改进

由当前代码实现可见：

- \(T\) 对应 `build_active_space_directional_integrals(...)`，见
  [exact_orbital_second_order_operator.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/exact_orbital_second_order_operator.cpp#L3373)；
- \(T^\ast\) 对应 `build_orbital_value_gradient_from_active_space_gradient_direction(...)`，见
  [exact_orbital_second_order_operator.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/exact_orbital_second_order_operator.cpp#L8010)。

这两层内部都仍然包含当前 exact HVP 最硬的热点：

- `active_2e`
- `h1e_fused`

而 [tnhvp_structural_optimization.md](/pool1/home/xiatao/project/xmvb-cpp/docs/tnhvp_structural_optimization.md) 已经说明，这两部分的核心改进空间主要来自中间量删除和访存组织，而不是“多右端项之后标度直接下降一阶”。

因此，对 \(m\) 个 Krylov 向量而言，

\[
C_T^{\mathrm{block}}(m),
\;
C_{T^\ast}^{\mathrm{block}}(m)
=
\Theta(m)
\]

这一点很难改变。块化最多只能把逐列 BLAS-2 / 稀疏散写改成更好的 BLAS-3 / tile streaming，因此这里的理论 ceiling 更像是一个有限常数，而不是 \(m\) 倍级别的共享。

更直白地说：

- 这一层不可能因为 MFTR 而“几乎免费”；
- 它最多只能变成“同样线性于 \(m\)，但常数更小”。

#### 10.3.2 \(P_{\mathrm{sel}}\) 与 \(W\)：这里才有真正的 block 共享机会

由代码可见：

- \(P_{\mathrm{sel}}\) 对应 `build_selected_state_projected_directional_structure_matrices(...)`，见
  [exact_orbital_second_order_operator.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/exact_orbital_second_order_operator.cpp#L6580)；
- \(W\) 对应 `build_active_space_gradient_direction_from_outer_response(...)`，见
  [exact_orbital_second_order_operator.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/exact_orbital_second_order_operator.cpp#L7896)。

这两层都已经建立在 selected-state / unique-spin 压缩对象之上，最大的共享对象包括：

- accepted-point same-spin cache；
- structure coefficient blocks；
- selected-state coefficient matrices；
- opposite-spin packed-pair channel family；
- generalized-eigen gap / gauge regularization 的固定 accepted-point数据。

因此若输入从一个方向变成 \(m\) 个方向块，则这两层的理想 block 成本更接近

\[
C_P^{\mathrm{block}}(m)
\approx
C_{P,\mathrm{traverse}}
+
m\, C_{P,\mathrm{rhs}},
\]

\[
C_W^{\mathrm{block}}(m)
\approx
C_{W,\mathrm{setup}}
+
m\, C_{W,\mathrm{rhs}}.
\]

而不是

\[
m(C_{P,\mathrm{traverse}} + C_{P,\mathrm{rhs}}),
\qquad
m(C_{W,\mathrm{setup}} + C_{W,\mathrm{rhs}}).
\]

这说明：

- 若当前这两层的主要代价来自一次次重复遍历 accepted-point graph / channel / structure support，
  则 block 化的理论共享上限可以接近 \(m\)；
- 若主要代价来自每个方向都必须做的 dense contraction，
  则上限就只剩一个中等常数。

因此，MFTR 的真正 ceiling 主要取决于 \(P_{\mathrm{sel}}\) 与 \(W\) 这两层，而不是 \(R_{\mathrm{sel}}\)。

#### 10.3.3 \(R_{\mathrm{sel}}\)：通常不是决定上限的层

\(R_{\mathrm{sel}}\) 对应
`build_selected_state_generalized_eigen_directional_response(...)`，见
[exact_orbital_second_order_operator.cpp](/pool1/home/xiatao/project/xmvb-cpp/src/vb/scf/exact_orbital_second_order_operator.cpp#L7151)。

这一层的主要工作是：

- 对 selected-state columns 做线性响应；
- 对每个 selected state 遍历所有 structure-space roots。

它当然依然依赖
\[
n_{\mathrm{str}} \times n_{\mathrm{sel}},
\]
但与 `active_2e` / `h1e_fused` 相比，这一层通常不是 outer-response 的最硬热点。因此它影响是否“值得做”，但通常不决定理论 ceiling。

### 10.4 对 \(C_{\Delta,Q}\) 的下界判断

由于 \(T\) 与 \(T^\ast\) 两层仍然包含当前 exact HVP 的最硬热点，并且它们对多右端项至多给出有限常数优化，因此有一个重要结论：

\[
\boxed{
C_{\Delta,Q}^{\mathrm{block}}
\text{ 不可能被压到可忽略。}
}
\]

也就是说，哪怕 MFTR 的 block 实现非常理想，也仍然至少要额外支付“一次 block 版 outer-response 主链”的成本，而不是只多付一个很小的 dense \(m\times m\) 代价。

因此 MFTR 的理论 ceiling 从一开始就不可能是数量级改进。

### 10.5 对总 wall-time 上限的判断

由

\[
S_{\mathrm{tot}}
=
\frac{N_0}{N_1}
\cdot
\frac{C_0}{C_0 + C_{\Delta,Q}}
\]

可知：

1. 即使 high-fidelity projection 完全 block 化，第二项也永远小于 1；
2. 当前 TNHVP 已经是二阶方法，MFTR 最多只能进一步减少 accepted iterations 的常数因子，而不可能把迭代数降到一个数量级以下。

因此总 wall-time 的理论 ceiling 更接近：

- 某些难收敛体系上得到明显但有限的常数因子改善；
- 而不是 \(5\times\)、\(10\times\) 这种数量级跃迁。

更稳妥的表达是：

\[
\boxed{
\text{MFTR 的合理 ceiling 更像是 tens-of-percent 到至多接近 } 2\times,
\text{ 而不是数量级提升。}
}
\]

这个判断来自以下三点共同作用：

1. cheap/core HVP 和 objective evaluation 仍然必须做；
2. \(T\) 与 \(T^\ast\) 这两层不会因为 block 化而消失；
3. 当前 TNHVP 本身已经具备一定二阶收敛性质，MFTR 能改善的主要是 tail / model-mismatch 带来的额外 accepted iterations。

### 10.6 能不能优于当前代码

可以，但前提很严格。

若要优于当前代码，必须同时满足：

1. 通过 block 化把
   \[
   C_{\Delta,Q}
   \]
   压到“当前每步成本的非主导修正项”；
2. 通过更准确的高保真投影 trust-region 子问题，把 accepted iterations 明显减少。

如果只做到第 2 点而做不到第 1 点，则 MFTR 仍可能是负优化；
如果只做到第 1 点而第 2 点收益很小，则 MFTR 也没有意义。

因此，MFTR 是否值得继续，并不取决于 trust-region 理论，而取决于下面这个更具体的问题：

\[
\boxed{
\text{能否把 } P_{\mathrm{sel}} \text{ 与 } W \text{ 做成真正共享 accepted-point 遍历的 block 算子，}
}
\]
\[
\boxed{
\text{同时不让 } T \text{ 与 } T^\ast \text{ 成为新的不可压缩瓶颈。}
}
\]

这就是当前代码语义下，MFTR 的真正理论上限所在。
