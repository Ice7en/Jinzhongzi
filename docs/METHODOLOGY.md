# 方法论：M4 双向时序记忆 · DACC 密度自适应通道校准 · M5 轨迹外推一致性

本文档详细说明本仓库三个核心模块的**原理、数学模型与实现机制**，并解释每个设计
决策背后的动机与消融依据。代码位置以 `m4-dacc-m5` 分支为准。

| 模块 | 类型 | 主要代码位置 | 作用 |
| --- | --- | --- | --- |
| **M4** 双向时序记忆 | 网络结构 | `model/temporal_memory_net.py` | 沿完整视频序列建模时序依赖，提 Pd / 降 Fa |
| **DACC** 密度自适应通道校准 | 网络结构 | `model/temporal_frame_net.py` | 通道级密度门控，抑制高密度噪声误检 |
| **M5** 轨迹外推一致性 loss | 训练期 loss | `utils/temporal_frame_loss.py` | 强化小目标轨迹连续性，抬 Pd |

三者叠加后 Score 0.94822 → 0.94965（官方 P0 口径），调优 P0 后 **0.95444**。

---

## 0. 基线 P23 的三个局限

P23（官方 temporal-frame 基线）是一个逐帧 2D U-Net：把完整事件流按 50 时间单位分箱，
每中心箱取前后共 5 箱构成 10 通道（5 箱 × 2 极性）计数帧，对每个 box 独立预测目标 logit。
它对每个 box **独立处理、没有跨 box 的时序上下文**，带来三个问题：

1. **瞬时噪声误检（Fa 高）**：单独一个 box 内一闪而过的噪声簇没有前后一致的运动模式，
   逐帧模型会把它当成目标。
2. **弱小目标漏检（Pd 低）**：高空 UAV 视角下小目标每帧只占极少数像素，单帧证据弱，
   逐帧判定容易漏。
3. **高密度场景通道被噪声主导**：草地/树冠等高频纹理区域产生大量噪声事件，输入通道
   计数帧在这些位置数值很高，干扰解码特征。

M4 针对 1/2，DACC 针对 3，M5 在 M4 的序列训练基础上进一步强化 2。

---

## 1. M4 —— 双向时序记忆（BidirectionalTemporalMemoryNet）

### 1.1 设计动机

事件小目标本质上是**沿轨迹移动**的：真正的目标在相邻时间箱之间存在一致的位置转移，
而瞬时噪声没有。逐帧 U-Net 浪费了这个强先验。M4 在 P23 U-Net 的 bottleneck 上叠一对
ConvGRU，沿完整视频序列正向 + 反向传播低分辨率证据，让每个 box 的预测**看到整个视频**
的前后文。

### 1.2 网络结构

```
输入帧 [B, T, 10, H, W]                    （H=346, W=260，10 通道 = 5箱×2极性）
        │
        │  P23 backbone（TemporalFrameNet，每个 box 共享权重）
        │  encoder0 → encoder1 → encoder2 → encoder3 → context
        │
  bottleneck [B*T, C=96, H/8, W/8]          （width=16 → bottleneck 通道 = 16×6 = 96）
        │
        │  重排为 [B, T, 96, h, w]
        │
        ├──► forward_memory  (ConvGRU，t=0..T−1)  ──► forward_states
        ├──► backward_memory (ConvGRU，t=T−1..0)  ──► backward_states
        │
        memory_features = concat(forward_states, backward_states)   [B, T, 192, h, w]
        │
        memory_projection = Conv2d(192, 96, 1×1)   ★ 零初始化
        │
        residual [B, T, 96, h, w]
        │
        bottleneck + residual ──► decoder2 → decoder1 → decoder0 → head → logits
```

关键点：**时序记忆只挂在 bottleneck 上**。bottleneck 分辨率只有 H/8×W/8、通道 96，
整条双向 GRU 的参数量和计算量都极小，却携带了最抽象、语义最强的特征，是放置时序
上下文的"最佳杠杆点"。

### 1.3 ConvGRU 单元（Cho et al. 的 GRU 公式的卷积版）

每个 box 的低分辨率特征图 $\mathbf{x}_t \in \mathbb{R}^{C \times h \times w}$ 和上一步
隐状态 $\mathbf{h}_{t-1}$ 输入单元：

$$ \mathbf{g}_t = \sigma\Big( W_g * [\mathbf{x}_t ; \mathbf{h}_{t-1}] \Big) =
[\mathbf{r}_t ; \mathbf{z}_t] $$

$$ \tilde{\mathbf{h}}_t = \tanh\Big( W_c * [\mathbf{x}_t ; \mathbf{r}_t \odot
\mathbf{h}_{t-1}] \Big) $$

$$ \mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot
\tilde{\mathbf{h}}_t $$

其中 $*$ 为 3×3 卷积，$[\cdot;\cdot]$ 为通道拼接，$\odot$ 为逐元素乘。$\mathbf{r}_t$
（reset gate）决定丢弃多少旧记忆，$\mathbf{z}_t$（update gate）决定新旧信息各占多少。
代码见 `model/temporal_memory_net.py:67-94`（`ConvGRUCell`）。

### 1.4 前向传播流程（`_memory_residual`，L167-212）

1. 正向：$t=0 \to T{-}1$，逐 box 迭代 `forward_memory`，收集全部隐状态
   `forward_states`。
2. 反向：$t=T{-}1 \to 0$，逐 box 迭代 `backward_memory`，收集 `backward_states`。
3. 拼接：`concat(forward_states, backward_states)`，沿时间维对齐，形状
   `[B, T, 192, h, w]`。
4. 投影：`memory_projection`（1×1 卷积，192→96 通道）压回 bottleneck 通道数。
5. 残差相加：`bottleneck + residual` 再走解码器。

推理端（`utils/temporal_memory_inference.py`）为了省显存做了两段式：先对全部时间箱
编码并**缓存 bottleneck**，然后**一次性**跑完整双向 GRU 得到全部 residual，再重算
skip 特征逐箱解码。这样每个箱只需一次完整前向，峰值显存控制在 4GB 内。

### 1.5 零初始化残差 —— 安全加载 P23 权重的关键

`memory_projection` 的权重和偏置全部初始化为 0（`temporal_memory_net.py:132-133`）。
于是训练一开始：

$$ \text{residual} = \text{Conv2d}_{\text{weight}=0}( \text{memory\_features}) + 0 = 0
$$

预测与纯 P23 **逐位一致**。因此可以先把 P23 checkpoint 原样加载进
`model.base`，再开始训练 GRU——**时序记忆从"零残差"出发，永远不会在首步破坏
baseline 预测**。这也是本项目所有新模块通用的"安全初始化"原则：新结构必须是
恒等/零残差起步，保证与官方权重兼容、且训练收益可归因于新模块本身。

### 1.6 为什么"双向 + 只挂 bottleneck"有效

- **双向**：检测目标是任意时刻出现的。正向 GRU 只看到过去，会漏掉"目标从未来才进入
  视野"的 case；反向补上了未来的上下文，让每个 box 同时利用过去和未来证据。
- **bottleneck 时序一致性天然抗噪声**：真正的目标跨 box 位置连续 → 隐状态逐步累积
  证据 → 置信度抬升（Pd ↑）；瞬时单 bin 噪声没有任何跨时间的一致性 → 隐状态不会为它
  累积 → 抑制（Fa ↓）。5 轮消融里 M4 单独在 P23 上即 **Score +0.0225**，且 Fa -35.6%。

### 1.7 `temporal_memory_base_lr_multiplier=1.0` 为什么是命门

M4 训练把参数分成两组（`train_temporal_memory.py:119-139`）：

| 参数组 | 学习率 | 内容 |
| --- | --- | --- |
| base | `lr × base_lr_multiplier` | P23 backbone 全部参数（含 DACC） |
| memory | `lr × memory_lr_multiplier` | 两个 GRU + 投影 |

官方默认 `base_lr_multiplier=0.1`，在 `lr=0.0001` 下把 base 压到 **0.00001**——
backbone（以及挂靠其下的 DACC）几乎被冻结。消融显示此时 IoU 封顶 ~0.894，Score
~0.936。设 **1.0** 后 base 以与官方 P23 完全相同的速率 0.0001 继续训练，IoU 升到
0.919，Score **+0.012**。这是本仓库最重要的复现参数。

### 1.8 消融证据

| 配置 | Score | Pd | IoU | Fa |
| --- | ---: | ---: | ---: | ---: |
| 官方 P23 baseline（50ep, 官方协议） | 0.93820 | 0.94939 | 0.90670 | 6.22e-06 |
| **M4 + DACC（base_lr×1.0）** | **0.94822** | 0.96073 | 0.91887 | 5.47e-06 |

四项全面超过官方 baseline（Pd +0.0113, IoU +0.0122, Acc +0.0092, Fa -12%）。

---

## 2. DACC —— 密度自适应通道校准（DensityAdaptiveChannelCalibrator）

### 2.1 设计动机

背景噪声随场景密度分布极不均匀：树冠/草地等高频区域事件密度远高于天空。逐帧模型
在这些位置误检多。DACC 的思路是**用"全局密度统计量"调节特征通道的重要性**——高密度
噪声场景下压低易受干扰的通道，从而抑制 Fa。

### 2.2 结构与公式（`model/temporal_frame_net.py:190-240`）

```
density_map = |input| 沿通道求和            # [B, H, W]，原始输入帧的计数帧
        │
        │  density_encoder: Conv2d(1,1,3×3,s=2) → ReLU → AdaptiveAvgPool2d(1)
        ▼
global_density g ∈ [B, 1]                    # 每个样本一个全局标量
        │
        │  gating_network: Linear(1→C/16) → ReLU → Linear(C/16→C) → Sigmoid
        ▼
channel_weights w ∈ [B, C, 1, 1]
        │
features = features ⊙ w                     # 通道级重加权
```

数学上等价于 SE-Block（Squeeze-and-Excitation），但 squeeze 的统计量不是特征图的全局
池化，而是**输入密度**。门控网络为一个两层的 MLP 加 Sigmoid，输出 [0,1] 内的逐通道
权重：

$$ w_c = \sigma\Big( W_2 \cdot \text{ReLU}\big( W_1 \cdot g \big) \Big) $$

### 2.3 安全初始化 Sigmoid(4)≈1.0

`gating_network` 的两层 `Linear` 权重全部零初始化，最后一层偏置置为 4.0
（L219-224）。于是初始时：

$$ w_c = \sigma(0 \cdot g + 4) = \sigma(4) \approx 0.982 \approx 1.0
$$

DACC 起步时近似恒等映射，**不改变任何解码输出**，因此 P23 权重可无损加载。训练中
网络学到有意义的密度门控后，才逐步产生偏离恒等的校准。

### 2.4 为什么必须是"通道级"而非"空间级"

DACC 只做逐通道乘法，**不改变空间结构**，因此不破坏 IoU。对照实验（M3 迭代历史）：
- M3v1 DensityAdaptiveStem：在 encoder0 处加多扩张卷积 + 密度门 → 在浅层破坏细节，
  50ep 退化到 ~0.9305。
- M3v2 DGC：decoder 后做空间注意力 mask 并拼接密度 → 直接改变输出分布，Score 崩到
  0.8645。
- **M3v3 DACC（本仓库）**：通道级 SE 式校准 + 权重乘法 → 5ep 有效 +0.0096。

结论：**空间级干预改动预测分布、浅层干预破坏细节；通道级重加权是唯一兼顾"能学"与
"不破坏 baseline"的形态。**

### 2.5 与 M4 的协作方式

DACC 挂在 `TemporalFrameNet` 内（base 内），作用于 decoder 输出、head 之前。M4 的
`_decode` 在密度校准开启时传入原始输入帧（`base_input`）供 DACC 计算密度
（`temporal_memory_net.py:232-237`）。所以 M4 与 DACC 是**串行**关系：bottleneck 加
时序残差 → 解码 → DACC 密度校准 → head。训练时 DACC 挂靠在 base 参数组下，因此
`base_lr_multiplier=1.0` 也是 DACC 不被饿死的前提。

---

## 3. M5 —— 轨迹外推一致性 loss（trajectory_extrapolation_loss_memory）

### 3.1 设计动机

M4 的 GRU 已经在序列层面累积了时序证据，但仍有**真阳被漏掉**的情况：目标在某个未观测
时间箱上恰好没被模型判出高分。M5 利用"小目标在短时窗内近似匀速直线运动"这个先验，
在**未观测时间步的外推位置上**主动施加高置信正则，把该位置的 logit 拉高——轨迹上被
漏检的位置随之恢复为真阳，Pd 提升。

### 3.2 算法（`utils/temporal_frame_loss.py:774-894`）

对每个训练序列（M4 的 `sequence_length=16` 个连续 box）：

1. **按目标分组**：遍历正事件（`label==1`），按 `target_id`（>0）分组，记录每个事件
   的 `(时间箱 t, x, y)`。
2. **过滤弱目标**：只处理已观测到 **≥ `min_known_points`**（默认 3）个不同时间箱的
   目标——样本太少时线性拟合不可靠。
3. **同箱合并**：同一时间箱内的多个事件点求均值，得到去重后的观测序列
   $\{(t_1, \bar x_1, \bar y_1), \dots\}$。
4. **最小二乘线性拟合**：构造设计矩阵 $A = [\mathbf{t} ; \mathbf{1}] \in
   \mathbb{R}^{n\times2}$，用 `torch.linalg.lstsq` 分别解
   $A \cdot [v_x, c_x]^\top = \mathbf{x}$ 与 $A \cdot [v_y, c_y]^\top = \mathbf{y}$，
   得到速度 $(v_x, v_y)$ 与起点 $(c_x, c_y)$。
5. **外推正则**：对序列内**每一个未被观测的时间箱** $t'$，计算外推位置
   $(p_x, p_y) = (v_x t' + c_x,\; v_y t' + c_y)$；若落在帧内，取该位置 logit，施加
   hinge 损失：
   $$ \mathcal{L}_{\text{traj}} = \frac{1}{N} \sum \text{relu}\big(
   \text{margin\_logit} - \text{logit}(t', p_x, p_y) \big) $$
6. **加权进总 loss**：训练箱 $t \ge \text{warmup}$ 后，
   $\mathcal{L} = \mathcal{L}_{\text{BCE}} + \text{weight} \cdot
   \mathcal{L}_{\text{traj}}$（默认 `weight=0.05`, `warmup_epochs=3`,
   `margin_logit=1.0`）。

`margin_logit=1.0` 对应 $\sigma(1.0)\approx0.73$ 的概率——即要求外推位置达到较高置信
而不是过拟合到 1。

### 3.3 为什么在 M4 上有效、在 P23 上失效

这是本仓库最重要的方法论结论之一：

- **P23 逐 view 训练**：每个训练样本是从完整视频随机采一个 800ms view，跨 view 无连续
  性。在完整视频尺度上，小目标轨迹方向会变，"线性运动"假设不成立——M5 在 P23 上两次
  验证均退化（5ep -0.0046、50ep -0.0097），已放弃。
- **M4 序列训练**：M4 训练数据是单视频内**连续 16 箱 / 800ms 窗口**的完整序列。窗口
  尺度下小目标近似匀速直线运动，线性拟合可靠——M5 在 M4 上首次有效。

**结论：M5 的有效性完全依赖训练数据粒度。它只在"短时窗线性运动假设成立"的序列训练
模式下有效。**

### 3.4 消融与机制拆解

| 配置 | Score @0.7 | Pd | Acc | IoU | Fa |
| --- | ---: | ---: | ---: | ---: | ---: |
| M4 + DACC（pid13） | 0.94822 | 0.96073 | 0.96127 | 0.91887 | 5.47e-06 |
| **M4 + DACC + M5** | **0.94965** | 0.97291 | 0.96736 | 0.91711 | 6.78e-06 |

M5 把 Pd 从 0.9607 抬到 0.9729（**+0.0122**）、Acc +0.0061；代价是 Fa 升 24%
（外推位置也会在个别错误点上点火）。Score 分解：Pd +0.0049 vs Fa −0.0037，净
+0.0014。**M5 的本质是"用灵敏度换冗余"——把 Pd 抬到 0.973，为后续用后处理压 Fa
留出空间**。

### 3.5 超参数语义一览

| 参数 | 默认 | 含义 |
| --- | ---: | --- |
| `weight` | 0.05 | 轨迹 loss 占总 loss 的比例（过大会压制 BCE） |
| `margin_logit` | 1.0 | 外推位置的目标 logit（对应 σ≈0.73 概率） |
| `min_points` | 3 | 线性拟合所需的最少观测时间箱数 |
| `warmup_epochs` | 3 | 前 N 个 epoch 关闭轨迹 loss，先让 base 稳定 |

M5 是**纯训练期 loss，无任何网络结构改动，推理路径完全不变**。训练产物 checkpoint 的
metadata 会记录 `trajectory_extrapolation_enabled`，供复现审计
（`train_temporal_memory.py:399`）。

---

## 4. 三个模块如何协同

完整计算图（训练时）：

```
原始事件序列 ─► 计数帧 [B,T,10,H,W]
                    │
        ┌───────────┴───────────┐
        │ P23 backbone          │  （shared per-box）
        ▼                       ▼
   bottleneck  ──────────►  双向 ConvGRU ──► 零初始化残差
        │                       │
        └────────── bottleneck + residual ──► 解码器
                                                  │
                                            DACC 密度校准  ── 用原始输入密度门控通道
                                                  │
                                              head → logits
                                                  │
                       ┌──────────────────────────┤
                       ▼                          ▼
              逐事件平衡 BCE（主 loss）       M5 轨迹外推 hinge loss
```

- **M4 提供时序上下文**：抬 Pd、压 Fa，是主贡献者（0.93820 → 0.94822 的主要来源）。
- **DACC 提供密度感知**：抑制高密度噪声误检，与 M4 互补（一个管时间、一个管密度）。
- **M5 提供轨迹先验**：在 M4 序列训练的基础上再抬 Pd，把灵敏度预算拉满。

---

## 5. 复现要点核对表

训练与评估涉及的所有关键参数都在命令行显式给出，并满足以下约束：

1. **随机种子固定**：P23 阶段 `TRAIN.seed=37`，M4 阶段 `TRAIN.seed=42`。两个训练脚本
   的 `setup_seed` 会同时固定 `random` / `numpy` / `torch` / CUDA 并关闭 cudnn benchmark；
   数据集视角采样同样使用该种子（`np.random.default_rng(seed)`）。
2. **参数记录**：每次训练自动写出 `config.yaml` 快照（解析后的完整配置）与
   `run_summary.json`（seed、best_loss、best_epoch、checkpoint 路径、`config_overrides`
   命令行覆盖列表）。
3. **checkpoint metadata**：`best_loss_seed42.pt` 内保存模型架构标志
   （`density_calibration_enabled`、`trajectory_extrapolation_enabled` 等），推理端
   `load_temporal_memory_model` 从 checkpoint 还原，避免配置与权重失配。
4. **评估协议**：必须开启 P0/P0c 后处理（`p0_enabled=true`、
   `p0c_high_confidence_recovery_enabled=true`），否则 Score 偏低 0.005-0.015，与官方
   口径不可比。验证与提交必须使用**完全一致**的模型、阈值（0.7）与 P0/P0c 参数。
5. **学习率命门**：`temporal_memory_base_lr_multiplier` 必须为 1.0（或 ≥0.5）。官方默认
   0.1 会饿死 base，IoU 封顶 ~0.894。
6. **数值差异**：不同 CUDA / PyTorch / spconv 版本会带来轻微数值差，属预期范围。
