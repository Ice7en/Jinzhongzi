# EVm4 — Bidirectional Temporal Memory for Event-based Tiny Object Detection

基于 ICCV 2025 [*Event-based Tiny Object Detection*](https://arxiv.org/abs/2506.23575) 官方 P23 全事件流时序帧方案的改进版。

在 EV-UAV Challenge 2 的 `val/` 验证集（24 个视频）上取得 **Score 0.94822**，
超过官方 P23 baseline（Score 0.93820）。

## 结果总览

| 模型 | 阈值 | Score | Pd | IoU | Acc | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P23 baseline（官方复现） | 0.600 | 0.93820 | 0.94939 | 0.90670 | 0.95210 | 6.22e-06 |
| **M4 + DACC（本仓库）** | **0.700** | **0.94822** | **0.96073** | **0.91887** | **0.96127** | **5.47e-06** |

四项指标全面超过官方 baseline（Pd +0.0113, IoU +0.0122, Acc +0.0092, Fa -12%）。

> **方法论详解**（M4 / DACC 的数学模型、前向流程、安全初始化机制、消融依据、复现要点
> 核对表）见 **[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)**。以下「方法」节为摘要。
> 在此基础上叠加 M5 轨迹外推 loss 的版本见 **`m4-dacc-m5` 分支**（Score 0.94965 /
> 调优 P0 0.95444）。

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
| `checkpoints/m4_dacc_best_loss_seed42.pt` | M4+DACC（50 epoch，best epoch 49，seed 42） | 直接评估 → **Score 0.94822** |

> 本仓库报告分数全部基于上述 checkpoint。叠加 M5 轨迹外推 loss 的 `m4-dacc-m5` 分支
> 另含 `checkpoints/m4_dacc_m5_best_loss_seed42.pt`（Score 0.95444）。

### 免训练直接评估（复现 0.94822）

```bash
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
  TEMPORAL_MEMORY.temporal_memory_model_path=checkpoints/m4_dacc_best_loss_seed42.pt \
  TEMPORAL_MEMORY.temporal_memory_sparse_weight=0.0
```

预期输出 `Score: 0.9482174352`（官方 P0：mdb=1 / retain=0.975）。

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
export PROJECT_DIR=/absolute/path/to/EVm4
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

### 2. 训练 M4 + DACC（50 epoch）

```bash
M4_ROOT="$PROJECT_DIR/log/m4_dacc"

python train_temporal_memory.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TRAIN.seed=42 \
  TRAIN.epochs=50 \
  TRAIN.lr=0.0001 \
  TRAIN.scheduler=cosine \
  TRAIN.scheduler_min_lr=0.000001 \
  TRAIN.max_events_num=100000 \
  TRAIN.model_save_root="$M4_ROOT" \
  TEMPORAL_FRAME.temporal_frame_density_calibration_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_init_model_path="$P23_CKPT" \
  TEMPORAL_MEMORY.temporal_memory_base_lr_multiplier=1.0
```

训练日志、`config.yaml` 快照、`run_summary.json` 会写入 `$M4_ROOT/runs/<时间戳>/`。
使用控制台输出的 `best loss checkpoint`（即 `best_loss_seed42.pt`）做评估。

训练规模约 198 序列/epoch，50 epoch 在单卡 3090 上约 80 分钟，显存约 4GB。

### 3. 评估

使用 `test2.py`，**必须开启 P0/P0c 后处理**（关闭 P0/P0c 时本模型只有 0.932 左右，
无法复现 0.94822）：

```bash
TEMPORAL="$M4_ROOT/runs/<时间戳>/best_loss_seed42.pt"

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

预期输出（本仓库复现值，含 P0/P0c）：

```text
IoU:      0.9188664556
Acc:      0.9612707496
Pd:       0.9607307854
Fa:       5.4721074801e-06
Score:    0.9482174352
```

不同 CUDA / PyTorch / spconv 版本可能有轻微数值差异。

**阈值扫描参考**（memory-only + P0/P0c，本仓库实测）：

| 阈值 | Score | Pd | IoU | Fa |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.94373 | 0.97186 | 0.91043 | 8.61e-06 |
| 0.6 | 0.94692 | 0.96619 | 0.91677 | 6.86e-06 |
| **0.7** | **0.94822** | 0.96073 | 0.91887 | 5.47e-06 |
| 0.8 | 0.94520 | 0.94834 | 0.91532 | 4.07e-06 |

### 4. 生成提交 TXT

验证与提交必须使用完全相同的模型、阈值（0.7）和 P0/P0c 参数：

```bash
OUTPUT_DIR="$PROJECT_DIR/log/challenge2/m4_dacc"

python submit_challenge2.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TEST.challenge_output_dir="$OUTPUT_DIR" \
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

cd "$OUTPUT_DIR"
zip -j ../m4_dacc_challenge2.zip val_*.txt
```

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
注意：YAML 配置文件的默认 seed: 37（e.g. evisseg_evuav.yaml:20），seed=42 是靠命令行 --set TRAIN.seed=42
  覆盖的。如果以后裸跑不带这个 override，会掉回 37，跨实验不可复现。

## 与官方 EVC 的代码改动

见 [`docs/CHANGES.md`](docs/CHANGES.md) —— 3 个文件的修改记录（M4 支持 DACC）。

## 仓库结构

```text
EVm4/
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
|-- train_temporal_memory.py    # M4+DACC 训练
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
