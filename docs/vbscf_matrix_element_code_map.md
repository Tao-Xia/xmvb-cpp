# VBSCF Matrix-Element Code Map

这份文档只做一件事：把当前 `xmvb-cpp` 里一次 `VBSCF` 运行真正走过的主线串起来，尤其是

- 前向矩阵元 `S/H`
- 反向 active-space 梯度
- 轨道梯度回传
- exact-ctx HVP

目标不是讲所有细节，而是回答两个问题：

1. 真正的算法主线在哪里？
2. 哪些文件是在“编排流程”，哪些文件才是“矩阵元核心”？

## 1. 一张总图

```text
run_cpp_vbscf.cpp
  -> load_cpp_vb_input_with_timings(...)
  -> CppVbScfOptimizer::optimize(...)
    -> OrbitalObjective::operator() / evaluate_energy_only(...)
      -> CppOrbitalGradientEvaluator
        -> CppActiveSpaceGradientEvaluator
          -> build_active_space_gradient_forward_context(...)
            -> prepare_timed_active_space_context(...)
              -> ActiveSpaceOrbitalPreparer
              -> AoEffectiveOneElectronBuilder
              -> ActiveSpaceOneElectronBuilder
              -> ActiveSpaceTwoElectronBuilder
            -> FullDeterminantStructureHamiltonianOverlapBuilder
            -> generalized eigensolver
          -> accumulate_active_space_gradient(...)
          -> finalize_active_space_second_order_context(...)
        -> active-space backpropagators
        -> ActiveSpaceOrbitalBackpropagator
      -> sparse orbital gradient / total energy
    -> line search / lbfgs / nonredundant / TN
      -> exact_orbital_second_order_operator (TN / exact_ctx only)
```

## 2. 顶层入口

### 2.1 运行入口

真正的 CLI 入口在 `src/tools/run_cpp_vbscf.cpp`。

- `load_cpp_vb_input_with_timings(...)` 负责把 `.xmi`、积分、轨道初猜、structure 数据装配成 `CppVbInput`
- 然后根据 backend 选择优化器
- 标准路径最终进入 `CppVbScfOptimizer::optimize(...)`

关键位置：

- `src/tools/run_cpp_vbscf.cpp:1866-1945`

从 review 角度看，这一层只是“驱动层”，不是矩阵元算法本体。

### 2.2 优化主循环

优化器入口：

- `src/vb/scf/cpp_vb_scf_optimizer.cpp:4477+`

真正每一步求值走的是内部的 `OrbitalObjective`：

- `src/vb/scf/cpp_vb_scf_optimizer.cpp:1688-1852`

这里有两个重要入口：

- `operator()(parameter_vector, gradient)`：完整能量 + 梯度评估
- `evaluate_energy_only(parameter_vector)`：只算 relaxed energy，给某些 trial-screening 用

所以后面你看优化器时，不要先看几千行 line search / TN 细节，先抓住一点：

- 所有 backend 最终都依赖 `OrbitalObjective`
- `OrbitalObjective` 决定“一个轨道点如何被评估”

## 3. 前向矩阵元主线

这一层的主问题是：

- 从当前轨道参数，如何得到 active-space 输入
- 再如何组装 structure-level 的 `S/H`

### 3.1 `CppVbScfEvaluator`

单步 SCF 前向入口在：

- `src/vb/scf/cpp_vb_scf_evaluator.cpp:91-154`

逻辑很简单：

1. `matrix_evaluator_.prepare_active_space(input)`
2. `matrix_evaluator_.evaluate(input, prepared_active_space)`
3. 广义本征值求解
4. 组合总能量

这层是“前向编排层”，不是重计算热点。

### 3.2 `StructureMatrixEvaluator`

结构矩阵前向总入口在：

- `src/vb/matrices/structure_matrix_evaluator.cpp:26-54`

这层把前向流程拆成两段：

1. `prepare_active_space(input)`
2. `structure_builder_.build(...)`

因此它是前向矩阵元最重要的分界点：

- 上半段：轨道与积分准备
- 下半段：determinant / structure 矩阵元装配

### 3.3 `PreparedActiveSpaceContext`

真正把轨道点变成 active-space 输入的主链在：

- `src/vb/matrices/prepared_active_space_context.cpp:55-140`

按顺序是：

1. `orbital_preparer.prepare(...)`
2. `ao_effective_one_electron_builder.build(...)`
3. `active_space_one_electron_builder.build(...)`
4. `active_space_two_electron_builder.build(...)`
5. `compute_one_electron_reference_energy(...)`

这就是最核心的“积分准备链”。

它产出的 `PreparedActiveSpaceContext` 包含：

- `OrbitalPreparationResult`
- `AoEffectiveOneElectronResult`
- `ActiveSpaceOneElectronResult`
- `ActiveSpaceTwoElectronResult`
- `one_electron_reference_energy`

也就是说，后面所有矩阵元/梯度/HVP，实际上都是围绕这个 context 展开。

### 3.4 四个前向 builder 的职责

这一层建议按“数学对象”理解，不要按文件树理解。

#### `ActiveSpaceOrbitalPreparer`

- 文件：`src/vb/orbital/active_space_orbital_preparer.*`
- 作用：从 sparse orbital parameterization 重建辅助轨道、inactive density、active overlap 等轨道相关量

#### `AoEffectiveOneElectronBuilder`

- 文件：`src/vb/orbital/ao_effective_one_electron_builder.*`
- 作用：从 AO 积分和 inactive density 构建 AO `G11/F11`

#### `ActiveSpaceOneElectronBuilder`

- 文件：`src/vb/orbital/active_space_one_electron_builder.*`
- 作用：把 AO `F11` 投影到 active-space `HHO`

#### `ActiveSpaceTwoElectronBuilder`

- 文件：`src/vb/orbital/active_space_two_electron_builder.*`
- 作用：把 AO 2e 积分变换成 active-space `GGO`

注意：从主链角度看，这四个 builder 是一条线，不是四个平级模块。现在代码上分文件没问题，但 review 时最好把它们当成“一个 preparation pipeline”。

## 4. 真正的矩阵元核心

如果你的问题是“VBSCF 矩阵元算法核心在哪”，答案不是 `cpp_vb_scf_evaluator`，而是：

- `src/vb/matrices/full_structure_builder.*`

### 4.1 结构

主入口是：

- `src/vb/matrices/full_structure_builder.cpp:1044-1235`

前向主构建调用：

- `build_impl(...)`
  - 先构建 / 复用 `SameSpinPairCacheContext`
  - 再进入 `build_tiled_matrix_form_structure_matrices(...)`

### 4.2 当前前向算法主形态

前向结构矩阵不是简单地“determinant pair 双循环硬算”，而是：

1. alpha / beta determinant 去重
2. 构建 same-spin reuse table
3. 用 tiled unique-spin contraction 做 structure-pair 累积

最关键的 forward 说明在：

- `src/vb/matrices/full_structure_builder.cpp:786-790`

和主 build 实现在：

- `src/vb/matrices/full_structure_builder.cpp:1088-1105`
- `src/vb/matrices/full_structure_builder.cpp:1186-1203`

所以从算法视角，这一层真正的前向矩阵元核心是：

- `FullDeterminantPairEvaluator`
- `SameSpinPairCacheContext`
- `build_tiled_matrix_form_structure_matrices(...)`

不是外面的 evaluator 壳子。

### 4.3 为何这里最值得重点 review

因为这一层同时决定：

- `S/H` 前向数值
- active-space adjoint 的很多上游语义
- backward 里 same-spin / opposite-spin 是否能复用 forward cache

所以只看 `cpp_vb_scf_evaluator` 会看不到重点，真正要顺的是：

`PreparedActiveSpaceContext -> FullStructureBuilder`

## 5. active-space 梯度主线

这一层回答的是：

- 对 state-averaged VBSCF 目标函数，`SSO/HHO/GGO` 的 adjoint 怎么来？

入口在：

- `src/vb/scf/cpp_active_space_gradient_evaluator.cpp:1321-1368`

主逻辑只有三步：

1. `build_active_space_gradient_forward_context(...)`
2. `initialize_active_space_gradient_result(...)`
3. `accumulate_active_space_gradient(...)`
4. `finalize_active_space_second_order_context(...)`

这里要注意一个认知点：

- 这层不是在做“轨道梯度”
- 这层是在做“active-space 层的 adjoint”

输出主要是：

- `active_orbital_overlap_gradient`
- `active_one_electron_gradient`
- `packed_active_two_electron_gradient`
- `second_order_context`

也就是，轨道梯度还没到；这里只是把 structure/eigensolver 的响应先压回 `SSO/HHO/GGO`。

## 6. 轨道梯度回传主线

真正把 active-space adjoint 拉回 sparse orbital parameter 的主入口在：

- `src/vb/scf/cpp_orbital_gradient_evaluator.cpp:295-488`

这段代码是当前“链式回传”的主干，顺序非常明确：

1. `CppActiveSpaceGradientEvaluator::evaluate(...)`
2. `active_space_matrix_backpropagator_.backpropagate(...)`
3. `active_space_two_electron_backpropagator_.backpropagate(...)`
4. `ao_effective_one_electron_backpropagator_.backpropagate(...)`
5. `active_space_orbital_backpropagator_.backpropagate(...)`
6. 得到 `sparse_orbital_energy_gradient`

也就是说，当前轨道梯度并不是一个单体公式，而是分层 pullback：

- active-space matrix pullback
- active-space 2e pullback
- AO effective 1e pullback
- orbital-preparation pullback

这也是现在阅读难度高的核心原因之一：数学上是一条链，但代码上拆成了多个 backpropagator。

## 7. exact-ctx HVP 主线

二阶 direct-action 入口是：

- `src/vb/scf/exact_orbital_second_order_operator.hpp:23-259`

这层不是普通前向 / 梯度层，而是：

- 在一个 accepted point 上缓存 relaxed active-space / eigensystem / selected-state adjoint
- 然后对 reduced direction 直接做 `H v`

主接口：

- `apply_reduced(...)`
- `apply_reduced_without_outer_response(...)`
- `apply_reduced_outer_response_only(...)`

从职责上，这一层把二阶项拆成两大块：

1. local/core orbital-integral chain differentiation
2. outer-response through `delta SSO / delta HHO / delta GGO`

所以 exact-ctx 其实是在复用前面整条主链，只是从“标量目标求值”换成了“accepted-point directional differentiation”。

## 8. 为什么现在会“层层封装，看不清”

当前可读性差，主要不是因为函数多，而是三层东西缠在一起了：

### 8.1 数学层和工程层混在一起

比如一个函数里同时出现：

- 数学对象：`SSO/HHO/GGO`
- 工程对象：cache / timed context / snapshot / probe path
- 策略分支：RI / exact / matrix-form / cached / uncached

这样主线就容易被埋掉。

### 8.2 前向编排层太多

现在至少有这些“调度壳”：

- `run_cpp_vbscf`
- `CppVbScfOptimizer`
- `OrbitalObjective`
- `CppOrbitalGradientEvaluator`
- `CppActiveSpaceGradientEvaluator`
- `StructureMatrixEvaluator`

每一层都合理，但层数叠起来后，人很难第一眼看出真正热点。

### 8.3 一个数学阶段被拆到多个 helper

尤其在 backward / exact-ctx 里，一个数学动作常常被拆成：

- forward context build
- initialize result
- accumulate adjoint
- finalize second-order context

这有利于复用，但不利于第一次 review。

## 9. 我建议的 review 顺序

如果目标是“先看懂，再优化”，建议不要按目录树看，而按下面顺序：

1. `src/tools/run_cpp_vbscf.cpp`
2. `src/vb/scf/cpp_vb_scf_optimizer.cpp`
3. `src/vb/scf/cpp_orbital_gradient_evaluator.cpp`
4. `src/vb/scf/cpp_active_space_gradient_evaluator.cpp`
5. `src/vb/matrices/prepared_active_space_context.cpp`
6. `src/vb/matrices/structure_matrix_evaluator.cpp`
7. `src/vb/matrices/full_structure_builder.cpp`
8. 再看各个 `builder` / `backpropagator`
9. 最后看 `src/vb/scf/exact_orbital_second_order_operator.*`

这样看，主链会更清楚：

```text
优化器
  -> 轨道梯度
    -> active-space adjoint
      -> active-space forward
      -> structure matrix builder
```

## 10. 下一步结构整理建议

如果后面继续清代码，我建议按执行顺序动，而不是按模块名动：

### 10.1 先固定一条“主流程文件图”

让每一层只有一个主入口，其他 helper 尽量下沉。

### 10.2 把前向 preparation pipeline 看成一段连续流程

现在它逻辑上是一段，但代码上被拆成多个 builder。后面可以保留文件拆分，但要让主入口更显式。

### 10.3 把“active-space adjoint”与“orbital pullback”边界写得更清楚

这条边界现在是存在的，但从文件跳转上不够直观。

### 10.4 exact-ctx 只在最后接入

不要一开始就从 `exact_orbital_second_order_operator.cpp` 往里读。它复用了太多前面的链路，先看它只会更乱。

---

一句话总结：

当前 `VBSCF` 代码不是没有主线，而是主线被很多“合理但分散的编排层”包住了。  
真正的矩阵元核心，前向在 `PreparedActiveSpaceContext + FullStructureBuilder`，反向在 `CppActiveSpaceGradientEvaluator + CppOrbitalGradientEvaluator`，二阶在 `ExactOrbitalSecondOrderOperator`。
