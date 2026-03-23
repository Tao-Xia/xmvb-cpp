# 项目需求文档：基于 JAX/Flax (NNX API) 的 Pair-Biased Transformer

## 1. 项目背景与技术栈设定
本项目旨在构建一个深度学习模型，用于加速非正交价键自洽场（VBSCF）计算中的哈密顿矩阵元预测。
模型需要将两个价键结构（左、右结构）之间的积分与拓扑连接图映射为一个全局标量（哈密顿矩阵元 $H_{IJ}$）。为了保证轨道置换不变性，并充分利用稠密张量信息，本项目采用 **无位置编码的 Pair-Biased Transformer** 架构。

**核心技术栈强制要求：**
* **必须使用最新的 `flax.nnx` API** 进行模块定义，绝对禁止使用旧版 `flax.linen` (不允许出现 `setup`, `compact`, `nn.Module`, `model.init()`, `model.apply()` 等旧版写法)。
* 所有模块需继承自 `nnx.Module`，在 `__init__` 中完成层实例化（需传入 `nnx.Rngs`），在 `__call__` 中实现前向传播。

## 2. 数据输入与输出规格

* **输入特征张量 (Input Tensor):**
    * **形状:** `(batch_size, N, N, 6)`，其中 $N$ 为系统的活性轨道数目。
    * **核心物理机制（为什么是 6 个维度）：** 预测两个非正交价键结构间的哈密顿矩阵元 $H_{IJ} = \langle \Phi_L | \hat{H} | \Phi_R \rangle$，本质上是由**能量标度（积分）**和**拓扑图（配对关系）**共同决定的。这 6 个特征通道完美解耦了这两部分信息：
        1. **通道 0: $h_{ij}$ (单电子积分)**。代表电子在轨道 $i$ 和 $j$ 之间的动能与核吸引势能，提供体系最基础的能量标度。对角线 $h_{ii}$ 是轨道的本征能量。
        2. **通道 1: $s_{ij}$ (重叠积分)**。非正交量子化学的核心！它刻画了轨道的空间重叠程度，网络需要隐式地利用它来感受重叠矩阵的代数余子式衰减。
        3. **通道 2: $J_{ij}$ (库仑积分对角项)**。代表轨道 $i$ 和 $j$ 的电荷云之间的经典静电排斥能 $(ii|jj)$。
        4. **通道 3: $K_{ij}$ (交换积分对角项)**。纯量子力学效应 $(ij|ji)$，这是决定自旋配对能量分裂（成键/反键）的最关键物理量。
        5. **通道 4: `left_edge` (左侧结构拓扑, $0/1$)**。一个二值图邻接矩阵。如果轨道 $i$ 和 $j$ 在左侧的价键结构（Bra state $\langle \Phi_L |$）中发生了自旋配对，则该位置为 1，否则为 0。
        6. **通道 5: `right_edge` (右侧结构拓扑, $0/1$)**。同理，代表右侧价键结构（Ket state $| \Phi_R \rangle$）中的自旋配对情况。
      
      *网络学习目标提示：* 通道 4 和 5 叠加在一起，隐式地构建了非正交 VB 理论中的 **Pauling Islands（鲍林岛/重叠循环）**。网络需要通过 Transformer 的注意力机制，结合前 4 个积分通道的能量大小，去“阅读”这些岛屿的连通性，从而推导出跃迁密度矩阵的贡献。

    * **对称性约束:** 该 `(N, N, 6)` 张量在物理上是严格对称的（即 $X_{ij, c} = X_{ji, c}$）。这是网络偏置层必须强制对称的根本原因。

## 3. 模型架构详细规范 (NNX 实现)

请严格按照以下模块结构使用 `flax.nnx` 编写代码：

### 3.1. 节点初始化模块 (Node Initialization)
1. **输入特征提取:** 从输入 `X` `(N, N, 6)` 中提取对角线特征：`node_features = jnp.diagonal(X, axis1=1, axis2=2)`，得到形状为 `(6, N)` 的张量。
2. **转置:** 变为序列格式 `(N, 6)`。
3. **特征映射:** 在 `__init__` 中定义一个 `nnx.Linear` 层，将通道数从 6 投影到 `d_model`。在 `__call__` 中应用该层并接 `jax.nn.gelu` 激活函数。

### 3.2. 边偏置生成模块 (Edge Bias Generation)
该模块将 `(N, N, 6)` 映射为用于干预 Attention 的偏置项。
1. **特征映射:** 定义一个 `nnx.Linear` 层，将输入 `X` 的最后一个维度从 6 映射到 `num_heads`（多头注意力的头数）。输出形状为 `(N, N, num_heads)`。
2. **强制对称性 (Crucial):** 为防止数值漂移，强制偏置张量对称：
   `bias = 0.5 * (bias + jnp.swapaxes(bias, 1, 2))`
3. **维度重排:** 使用 `jnp.transpose` 将其转换为 `(num_heads, N, N)`，以完美对齐 Attention logits 的形状。

### 3.3. Pair-Biased 自注意力层 (Custom Self-Attention)
**关键限制：禁止直接实例化 `nnx.MultiHeadAttention`**，因为其标准 `__call__` 未暴露动态注入 `bias` 的接口。必须显式组装：
1. **初始化 (`__init__`):**
   * 定义 4 个独立的 `nnx.Linear` 层，分别用于 `q_proj`, `k_proj`, `v_proj` 和 `out_proj`。注意传入 `rngs: nnx.Rngs`。
2. **前向传播 (`__call__(self, x, edge_bias, mask=None)`):**
   * 计算 `q, k, v`，将其 Reshape 并 Transpose 为多头格式：`(, num_heads, N, head_dim)`。
   * **调用底层算子：** 直接使用 `nnx.dot_product_attention`。
     ```python
     # 核心计算逻辑示例
     attn_out = nnx.dot_product_attention(
         query=q, key=k, value=v,
         bias=edge_bias,  # 注入 3.2 步生成的对称偏置
         mask=mask        # Padding mask（如有）
     )
     ```
   * 将 `attn_out` 恢复为 `(N, d_model)` 并通过 `out_proj` 层。

### 3.4. Transformer Block 与主网络
1. **Block 结构 (`PairBiasedTransformerBlock`):**
   * Pre-LayerNorm 架构设计。
   * `x = x + attention(nnx.LayerNorm(x), edge_bias)`
   * `x = x + mlp(nnx.LayerNorm(x))` (MLP 结构为 Dense -> GELU -> Dense)。
2. **主网络 (`VBHamiltonianPredictor`):**
   * 堆叠 `L` 层上述 Block。
   * **绝对禁止使用任何位置编码 (Positional Encoding)**，以保持轨道置换等变性。

### 3.5. 置换不变的读出层 (Permutation Invariant Readout)
1. 获取最后一层 Block 的输出 `H_out` `(N, d_model)`。
2. **广延性池化 (Extensive Pooling):** 必须使用 `jnp.sum(H_out, axis=1)` 进行求和池化（若有 Padding mask，请利用 `jnp.where` 将 Padding 节点置零后再求和）。绝对禁止使用平均池化 (`mean`)。
3. 通过两层 `nnx.Linear` (中间夹 `gelu`) 将维度降维至 1。最后一层无激活函数。

## 4. 交付代码要求
1. 提供完整的类定义：`NodeInit`, `EdgeBias`, `PairBiasedAttention`, `TransformerBlock`, `VBHamiltonianPredictor`。
2. 提供一个 `if __name__ == "__main__":` 测试脚本：
   * 初始化 `rngs = nnx.Rngs(0)`。
   * 实例化模型 `model = VBHamiltonianPredictor(d_model=64, num_heads=4, num_layers=3, rngs=rngs)`。
   * 生成随机 `(10, 10, 6)` 输入张量进行一次 Forward 验证，并打印输出形状（应为 `(1,)`）。



