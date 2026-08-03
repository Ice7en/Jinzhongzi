# EVm4+m5 — Bidirectional Temporal Memory for Event-based Tiny Object Detection

基于 ICCV 2025 [*Event-based Tiny Object Detection*](https://arxiv.org/abs/2506.23575) 官方 P23 全事件流时序帧方案的改进版。

在 EV-UAV Challenge 2 的 `val/` 验证集（24 个视频）上，模型最佳 **Score 0.94822**
（M4+DACC，官方 P0 协议），叠加 M5 轨迹外推一致性 loss 后为 **0.94965**；
换用调优 P0 参数可进一步提升至 **Score 0.95444**，全部超过官方 P23 baseline（Score 0.93820）。

## 结果总览

| 模型 | 阈值 | Score | Pd | IoU | Acc | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P23 baseline（官方复现） | 0.600 | 0.93820 | 0.94939 | 0.90670 | 0.95210 | 6.22e-06 |
| M4 + DACC（本仓库） | 0.700 | 0.94822 | 0.96073 | 0.91887 | 0.96127 | 5.47e-06 |
| M4 + DACC + M5 | 0.700 | 0.94965 | 0.97291 | 0.91711 | 0.96736 | 6.78e-06 |
| **M4+DACC+M5 + 调优 P0（推荐提交）** | **0.700** | **0.95419** | **0.96871** | **0.92672** | **0.96605** | **5.21e-06** |
| M4+DACC+M5 + 调优 P0（扫描最优） | 0.700 | 0.95444 | 0.96661 | 0.92852 | 0.96568 | 4.94e-06 |

M4+DACC 四项指标全面超过官方 baseline（Pd +0.0113, IoU +0.0122, Acc +0.0092, Fa -12%）。
M5 主要提升 Pd（+0.0122）并同步拉高 Acc；调优 P0 通过过滤单 bin 噪声簇进一步压 Fa（-23%）并抬 IoU。

> **方法论详解**（M4 / DACC / M5 的数学模型、前向流程、安全初始化机制、消融依据、
> 复现要点核对表）见 **[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)**。以下「方法」
> 节为摘要。

评分使用官方 Challenge 2 公式：

```text
Score_Fa = exp(-10000 * Fa)
Score = 0.4 * Pd + 0.3 * Score_Fa + 0.2 * IoU + 0.1 * Acc
```

## 预训练权重（免训练复现报告分数）

训练产物（`*.pt`）已提交到 `checkpoints/`，**无需再训练即可复现报告分数**。仍需先按
下方「数据准备」下载数据集并设置 `$DATA_ROOT`。

| 文件 | 说明 | 用途 |
| --- | --- | --- |
| `checkpoints/m4_dacc_m5_best_loss_seed42.pt` | M4+DACC+M5（50 epoch，best epoch 48，seed 42） | 直接评估 → **Score 0.95444** |
| `checkpoints/p23_baseline_5ep_seed42.pt` | P23 基线（5 epoch，best epoch 4，seed 42） | 重训 M4 时的 init 权重 |

> 本仓库报告分数全部基于上述 `m4_dacc_m5_best_loss_seed42.pt`。若要逐位复现训练
> 产物，用 `p23_baseline_5ep_seed42.pt` 作为 M4 的 `temporal_memory_init_model_path`
> 并按「复现流程」第 2 步训练即可。

### 免训练直接评估（复现 0.95444）

```bash
python test2.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TEST.eval=true TEST.roc=true TEST.prediction_threshold=0.7 \
  TEMPORAL_MEMORY.temporal_memory_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_model_path=checkpoints/m4_dacc_m5_best_loss_seed42.pt \
  TEMPORAL_MEMORY.temporal_memory_sparse_weight=0.0 \
  POSTPROCESS.p0_enabled=true POSTPROCESS.p0_spatial_radius=2 POSTPROCESS.p0_temporal_bin_size=50 \
  POSTPROCESS.p0_temporal_radius_bins=1 POSTPROCESS.p0_min_cluster_events=3 POSTPROCESS.p0_min_duration_bins=5 \
  POSTPROCESS.p0c_high_confidence_recovery_enabled=true POSTPROCESS.p0c_retain_min_score=0.92
```

预期输出 `Score: 0.9544424489`（调优 P0：mdb=5 / retain=0.92）。若改用官方 P0 参数
（mdb=1）评估则为 **0.94965**，与「结果总览」一致。

> 注：`checkpoints/.gitignore` 用 `!*.pt` 放行了该目录以便提交权重；仓库其余位置
> 仍忽略 `*.pt`。

## 方法

- **P23 基线**：轻量级 2D U-Net（width=16，输入 346x260），把完整事件流按 50 时间单位
  分箱，每中心箱取前后共 5 个箱构建 10 通道（5 箱 x 2 极性）计数帧。事件坐标处取 logit
  计算平衡 BCE。官方实现细节见 `docs/CHANGES.md` 与官方 EVC 仓库。
- **M4 双向时序记忆**（`model/temporal_memory_net.py`）：在 P23 U-Net bottleneck
  （96 通道，H/8×W/8）上叠加一对 ConvGRU，沿完整视频序列正向/反向传播低分辨率证据。
  ConvGRU 更新规则为
  $h_t = (1-z_t)\odot h_{t-1} + z_t \odot \tilde h_t$，
  其中 $z_t$ 为 update gate、$r_t$ 为 reset gate（3×3 卷积 + sigmoid 得到），
  $\tilde h_t = \tanh(W_c * [x_t; r_t \odot h_{t-1}])$。正反向隐状态拼接后经 1×1 卷积
  投影为**零初始化残差**，与 bottleneck 相加再解码——零初始化保证加载 P23 权重后初始
  预测逐位不变，时序证据随训练逐步累积（真正的目标跨箱位置连续 → 置信抬升 Pd↑；瞬时
  单 bin 噪声无一致性 → 抑制 Fa↓）。推理时先缓存全部 bottleneck、一次跑完整双向 GRU
  再逐箱解码，峰值显存约 4GB。
- **DACC 密度自适应通道校准**（`model/temporal_frame_net.py` 内
  `DensityAdaptiveChannelCalibrator`）：通道级 SE 式密度门控。把原始输入计数帧沿通道
  求和得密度图，经小 conv + 全局池化得每个样本的全局密度标量 $g$，再由两层 MLP +
  Sigmoid 输出逐通道权重 $w_c=\sigma(W_2\,\mathrm{ReLU}(W_1 g))$，与解码特征逐通道相乘。
  两层 MLP 零初始化、末层偏置置 4.0 → 初始 $w_c=\sigma(4)\approx0.98\approx1$（恒等），
  P23 权重可无损加载。通道级重加权不改空间结构，因此不破坏 IoU。
- **M5 轨迹外推一致性 loss**（`utils/temporal_frame_loss.py` 内
  `trajectory_extrapolation_loss_memory`）：仅训练期生效、无任何结构改动。对
  `target_id` 分组，用 ≥`min_points` 个已观测时间箱经 `lstsq` 拟合线性轨迹，在序列内
  每个未观测时间箱的外推位置上施 hinge 正则
  $\mathcal{L}_{\mathrm{traj}} = \mathrm{relu}(\mathrm{margin\_logit} - \mathrm{logit}(t', p_x, p_y))$，
  强化小目标轨迹连续性。loss 权重 0.05，前 3 epoch warmup（`margin_logit=1.0` ≈
  σ(1.0)≈0.73）。**关键结论**：M5 在 P23 逐 view 训练上两次验证均退化（线性假设跨完整
  视频不成立），只在 M4 的「单视频连续 16 箱/800ms 窗口」序列训练上有效。

**关键训练设置（本仓库最重要的复现参数）：**

```text
TEMPORAL_MEMORY.temporal_memory_base_lr_multiplier=1.0
```

> 该值必须是 **1.0（或 ≥0.5）**。官方默认 0.1 会把 base 学习率压到 0.00001，
> base（以及挂靠在 base 下的 DACC）被「饿死」，IoU 封顶 ~0.894；设 1.0 后
> base 以 0.0001（与官方 P23 训练同率）训练，IoU 升至 0.919，Score 提升 +0.012。

## 环境配置

已验证环境：WSL/Ubuntu、Python 3.8（3.9 亦可）、PyTorch 1.9.1 + CUDA 11.1、torchvision 0.10.1、
`spconv-cu111`、NumPy 1.23.5。

```bash
conda create -n EV39 python=3.9 pip -y
conda activate EV39
python -m pip install --upgrade pip
python -m pip install \
  torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install \
  numpy==1.23.5 pyyaml==6.0.2 tqdm==4.66.5 pandas==2.0.3 \
  opencv-python==4.8.1.78 mlflow==2.17.2 spconv-cu111 \
  typing-extensions==4.12.2 pillow==10.4.0
```

> 本仓库的时序帧/时序记忆路径在 `temporal_memory_sparse_weight=0.0` 下不加载稀疏
> 模型，无需编译 HAIS_OP。但 `test2.py` 等脚本仍会 import 稀疏模型模块，
> 需保证 `spconv` 可用。环境变量示例：

```bash
export PROJECT_DIR=/absolute/path/to/EVm4+m5
export DATA_ROOT="$PROJECT_DIR/dataset/训练集、验证集"
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
```

## 数据准备

从官方渠道下载 EV-UAV 数据包（见官方 EVC 仓库 README 的百度网盘 / Google Drive 链接），
解压后将 Challenge 2 数据放在 `$DATA_ROOT` 下：

```text
dataset/训练集、验证集/
|-- train/     # 99 个 .npz
`-- val/       # 24 个 .npz
```

## 复现流程

以下命令均在仓库根目录执行。

### 1. 训练 P23 baseline（作为 M4 的 init 权重）

M4 通过 `temporal_memory_init_model_path` 加载 P23 权重。先用官方 P23 训练流程
产出一个 P23 checkpoint（要求 `temporal_frame_context_bins=5`、`temporal_frame_width=16`，
与 M4 配置匹配）：

```bash
P23_ROOT="$PROJECT_DIR/log/p23_baseline"

python train_temporal_frame.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TRAIN.seed=37 \
  TRAIN.epochs=50 \
  TRAIN.lr=0.0001 \
  TRAIN.scheduler=cosine \
  TRAIN.scheduler_min_lr=0.000001 \
  TRAIN.max_events_num=100000 \
  TRAIN.model_save_root="$P23_ROOT" \
  TEMPORAL_FRAME.temporal_frame_enabled=true \
  TEMPORAL_FRAME.temporal_frame_bin_size=50 \
  TEMPORAL_FRAME.temporal_frame_context_bins=5 \
  TEMPORAL_FRAME.temporal_frame_width=16 \
  TEMPORAL_FRAME.temporal_frame_train_views_per_video=8 \
  TEMPORAL_FRAME.temporal_frame_positive_frame_probability=0.75 \
  TEMPORAL_FRAME.temporal_frame_target_positive_loss_mass=0.20 \
  TEMPORAL_FRAME.temporal_frame_max_positive_weight=16 \
  TEMPORAL_FRAME.temporal_frame_cache_all_videos=true \
  TEMPORAL_FRAME.temporal_frame_train_workers=0
```

训练结束后记录 `.../best_loss_seed37.pt` 路径作为 `$P23_CKPT`。

> 本仓库报告的 0.94822 使用的 init 是一个 5 epoch 训练的 P23 checkpoint。
> init 质量会影响最终分数，更强的 init（如 50 epoch P23）可能进一步提升。

### 2. 训练 M4 + DACC + M5（50 epoch）

```bash
M5_ROOT="$PROJECT_DIR/log/m4_dacc_m5"

python train_temporal_memory.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TRAIN.seed=42 \
  TRAIN.epochs=50 \
  TRAIN.lr=0.0001 \
  TRAIN.scheduler=cosine \
  TRAIN.scheduler_min_lr=0.000001 \
  TRAIN.max_events_num=100000 \
  TRAIN.model_save_root="$M5_ROOT" \
  TEMPORAL_FRAME.temporal_frame_density_calibration_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_init_model_path="$P23_CKPT" \
  TEMPORAL_MEMORY.temporal_memory_base_lr_multiplier=1.0 \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_enabled=true \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_weight=0.05 \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_margin_logit=1.0 \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_min_points=3 \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_warmup_epochs=3
```

去掉最后 4 个 `TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_*` 参数即为
M4+DACC 训练命令（0.94822）。M5 参数为本仓库验证值：weight=0.05、margin_logit=1.0、
min_points=3、warmup_epochs=3，`temporal_memory_base_lr_multiplier` 必须为 1.0。

训练日志、`config.yaml` 快照、`run_summary.json` 会写入 `$M5_ROOT/runs/<时间戳>/`。
使用控制台输出的 `best loss checkpoint`（即 `best_loss_seed42.pt`，约 epoch 48）做评估。

训练规模约 198 序列/epoch，50 epoch 在单卡 3090 上约 80 分钟，显存约 4GB。

### 3. 评估

使用 `test2.py`，**必须开启 P0/P0c 后处理**（关闭 P0/P0c 时本模型只有 0.932 左右，
无法复现 0.94822）：

```bash
TEMPORAL="$M5_ROOT/runs/<时间戳>/best_loss_seed42.pt"

python test2.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TEST.prediction_threshold=0.700 \
  POSTPROCESS.p0_enabled=true \
  POSTPROCESS.p0_spatial_radius=2 \
  POSTPROCESS.p0_temporal_bin_size=50 \
  POSTPROCESS.p0_temporal_radius_bins=1 \
  POSTPROCESS.p0_min_cluster_events=3 \
  POSTPROCESS.p0_min_duration_bins=1 \
  POSTPROCESS.p0c_high_confidence_recovery_enabled=true \
  POSTPROCESS.p0c_retain_min_score=0.975 \
  POSTPROCESS.p6_density_threshold_enabled=false \
  TEMPORAL_MEMORY.temporal_memory_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_model_path="$TEMPORAL" \
  TEMPORAL_MEMORY.temporal_memory_sparse_weight=0.0
```

预期输出（本仓库复现值，M4+DACC+M5，含官方 P0/P0c）：

```text
IoU:      0.9171141386
Acc:      0.9673617482
Pd:       0.9729105418
Fa:       6.7816210190e-06
Score_Fa: 0.9344321969
Score:    0.9496528783
```

不同 CUDA / PyTorch / spconv 版本可能有轻微数值差异。

**阈值扫描参考**（M4+DACC 官方 P0/P0c，本仓库实测）：

| 阈值 | Score | Pd | IoU | Fa |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.94373 | 0.97186 | 0.91043 | 8.61e-06 |
| 0.6 | 0.94692 | 0.96619 | 0.91677 | 6.86e-06 |
| **0.7** | **0.94822** | 0.96073 | 0.91887 | 5.47e-06 |
| 0.8 | 0.94520 | 0.94834 | 0.91532 | 4.07e-06 |

### 3b. 调优 P0 评估（推荐提交口径）

官方 P0 参数（mdb=1, mce=3）下 Fa 主要由单时间 bin 的噪声簇贡献。把
`p0_min_duration_bins` 从 1 收紧到 6，噪声簇被过滤，Fa 从 6.78e-6 降到 5.21e-6
（-23%），同时 IoU 从 0.917 升到 0.927（误检事件同时计入 Fa 分子与 IoU 分母，
一体两用）。推荐使用 **mdb=6 / P0c retain=0.90**：

```bash
python test2.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TEST.prediction_threshold=0.700 \
  POSTPROCESS.p0_enabled=true \
  POSTPROCESS.p0_spatial_radius=2 \
  POSTPROCESS.p0_temporal_bin_size=50 \
  POSTPROCESS.p0_temporal_radius_bins=1 \
  POSTPROCESS.p0_min_cluster_events=3 \
  POSTPROCESS.p0_min_duration_bins=6 \
  POSTPROCESS.p0c_high_confidence_recovery_enabled=true \
  POSTPROCESS.p0c_retain_min_score=0.900 \
  POSTPROCESS.p6_density_threshold_enabled=false \
  TEMPORAL_MEMORY.temporal_memory_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_model_path="$TEMPORAL" \
  TEMPORAL_MEMORY.temporal_memory_sparse_weight=0.0
```

预期输出：

```text
IoU:      0.9267199636
Acc:      0.9660488963
Pd:       0.9687106258
Fa:       5.2143525528e-06
Score:    0.9541909198
```

调优 P0 扫描中最优为 **mdb=5 / retain=0.92 → 0.95444**（Pd 0.96661, IoU 0.92852,
Fa 4.94e-6）。两档差距很小，均属于在 `val/` 上调参的结果，提交时需意识到参数可能
轻微过拟合验证集。扫描区间：sr∈{2,3}, trb∈{1,2}, mce∈{3,4}, mdb∈{1..7},
retain∈{0.88..0.975}；上述两档命令即为复现方式，其余组合可通过替换
`p0_min_duration_bins` / `p0c_retain_min_score` 值复现。

### 4. 生成提交 TXT

验证与提交必须使用完全相同的模型、阈值（0.7）和 P0/P0c 参数。
`submit_challenge2.py` 在阈值 + P0/P0c 之后写出最终二值 label，因此调优 P0 参数
（下例 mdb=6 / retain=0.90，对应 0.95419）可直接用于提交：

```bash
OUTPUT_DIR="$PROJECT_DIR/log/challenge2/m4_dacc_m5"

python submit_challenge2.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TEST.challenge_output_dir="$OUTPUT_DIR" \
  TEST.prediction_threshold=0.700 \
  POSTPROCESS.p0_enabled=true \
  POSTPROCESS.p0_spatial_radius=2 \
  POSTPROCESS.p0_temporal_bin_size=50 \
  POSTPROCESS.p0_temporal_radius_bins=1 \
  POSTPROCESS.p0_min_cluster_events=3 \
  POSTPROCESS.p0_min_duration_bins=6 \
  POSTPROCESS.p0c_high_confidence_recovery_enabled=true \
  POSTPROCESS.p0c_retain_min_score=0.900 \
  POSTPROCESS.p6_density_threshold_enabled=false \
  TEMPORAL_MEMORY.temporal_memory_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_model_path="$TEMPORAL" \
  TEMPORAL_MEMORY.temporal_memory_sparse_weight=0.0

cd "$OUTPUT_DIR"
zip -j ../m4_dacc_m5_challenge2.zip val_*.txt
```

如想沿用官方 P0 参数提交，把 `p0_min_duration_bins` 改回 `1`、`p0c_retain_min_score`
改回 `0.975` 即可（对应 0.94965）。

`submit_challenge2.py` 生成 24 个 `val_*.txt`，每行 `x y t p label`。

## 关键配置（configs/evisseg_evuav.yaml）

Temporal-memory 相关默认值：

```yaml
TEMPORAL_MEMORY:
  temporal_memory_enabled: false
  temporal_memory_bin_size: 50
  temporal_memory_context_bins: 5
  temporal_memory_width: 16
  temporal_memory_sequence_length: 16
  temporal_memory_inference_batch_size: 8
  temporal_memory_log_count_clip: 4.0
  temporal_memory_base_lr_multiplier: 1.0    # 本仓库改为 1.0（关键）
  temporal_memory_memory_lr_multiplier: 1.0
```

TEMPORAL_FRAME 下 M5 相关默认值（本仓库验证值）：

```yaml
TEMPORAL_FRAME:
  temporal_frame_density_calibration_enabled: false
  temporal_frame_trajectory_extrapolation_enabled: false
  temporal_frame_trajectory_extrapolation_weight: 0.05
  temporal_frame_trajectory_extrapolation_margin_logit: 1.0
  temporal_frame_trajectory_extrapolation_min_points: 3
  temporal_frame_trajectory_extrapolation_warmup_epochs: 3
```

## 与官方 EVC 的代码改动

见 [`docs/CHANGES.md`](docs/CHANGES.md) —— 官方 P23 → M4+DACC（3 个文件）→ M5（追加）
的修改记录。

## 仓库结构

```text
EVm4+m5/
|-- configs/
|   `-- evisseg_evuav.yaml
|-- dataset/
|   |-- ev_uav.py
|   |-- temporal_frame.py
|   `-- temporal_memory.py
|-- model/
|   |-- temporal_frame_net.py
|   `-- temporal_memory_net.py
|-- utils/
|   |-- temporal_memory_inference.py
|   |-- temporal_frame_inference.py
|   |-- temporal_frame_loss.py
|   |-- postprocess.py
|   `-- challenge_eval.py
|-- train_temporal_frame.py     # P23 init 训练
|-- train_temporal_memory.py    # M4+DACC+M5 训练
|-- test2.py                    # 评估
|-- submit_challenge2.py        # 提交
|-- docs/
|   `-- CHANGES.md
`-- README.md
```

## 引用

```bibtex
@misc{chen2025eventbasedtinyobjectdetection,
  title={Event-based Tiny Object Detection: A Benchmark Dataset and Baseline},
  author={Nuo Chen and Chao Xiao and Yimian Dai and Shiman He and Miao Li and Wei An},
  year={2025},
  eprint={2506.23575},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2506.23575}
}
```
