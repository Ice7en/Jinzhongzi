# CHANGES — M4+DACC+M5 的代码改动

相对官方 EVC（`Picasso9jiu/EVC`, evc-main 分支）P23 全事件流时序帧方案的改动。
第一轮（M4+DACC）共 **3 个文件**：让 `BidirectionalTemporalMemoryNet`（M4）贯通
`DensityAdaptiveChannelCalibrator`（DACC / M3v3），使 M4+DACC 可联合训练，
并在加载/推理时正确还原 DACC 状态。第二轮（M5 轨迹外推一致性 loss）追加
**2 个文件**（见文末「第二轮：M5」），纯训练期 loss，无结构改动。

`docs/CHANGES.md` 里的 baseline 特指「官方 P23 temporal-frame 方案」。本仓库的
M4、DACC、M5 模块本身即新增代码（`model/temporal_memory_net.py`、
`model/temporal_frame_net.py`、`utils/temporal_frame_loss.py`），此处只记录
为了叠加三者所需的对既有文件的最小改动。

---

## 1. `model/temporal_memory_net.py`

`BidirectionalTemporalMemoryNet` 包住 `TemporalFrameNet` 作为 base。

- **`__init__`**（L112-L118）：新增参数 `density_calibration_enabled=False`，
  构造 base 时透传：`self.base = TemporalFrameNet(..., density_calibration_enabled=...)`。
- **`_decode`**（L217-L226）：新增 `base_input=None` 参数；当
  `self.base.density_calibration_enabled` 时，对 decoder 输出做
  `decoded0 = self.base.density_calibrator(decoded0, base_input)`。
  无 base_input 时抛错提示（DACC 需要原始输入帧计算密度）。
- **`decode_with_residual`**（L239）与 **`forward`**（L281）：
  调用 `_decode` 时传入 `base_input=frames[:, :self.input_channels]`，
  即原始输入帧，供 DACC 的密度编码器使用。

## 2. `train_temporal_memory.py`

- **`load_p23_base_weights`**（L82-L107）：新增参数
  `density_calibration_enabled=False`；加载 P23 权重时
  `strict=not bool(density_calibration_enabled)`。
  原因：DACC 是 base 内新加的层，P23 init 权重里没有对应参数，
  必须放宽 strict 加载，让 DACC 走自身的 Sigmoid(4)≈1.0 安全初始化。
- **模型构造**（L199-L212）：从配置读取
  `temporal_frame_density_calibration_enabled`，构造 base 与 M4 时都传入。
- **checkpoint 记录**（L296-L299）：saved dict 写入
  `density_calibration_enabled`，供推理时还原。

## 3. `utils/temporal_memory_inference.py`

- **`load_temporal_memory_model`**（L80-L106）：从 checkpoint 读取
  `saved.get('density_calibration_enabled', False)`，用该值构造
  `BidirectionalTemporalMemoryNet(..., density_calibration_enabled=...)`，
  并以 `strict=True` 加载权重（train 时已保存完整状态）。
  checkpoint 中无该字段（旧模型）时按 `False` 处理，保持向后兼容。

---

## 运行时开关

上述改动全部由配置开关驱动，不改动默认行为：

```yaml
TEMPORAL_FRAME.temporal_frame_density_calibration_enabled: false   # 关闭时行为等同官方 P23
TEMPORAL_MEMORY.temporal_memory_enabled: true                      # 开启 M4
```

复现本仓库 0.94822 结果时，训练需开启
`TEMPORAL_FRAME.temporal_frame_density_calibration_enabled=true`，
并设置 `TEMPORAL_MEMORY.temporal_memory_base_lr_multiplier=1.0`（见 README「关键训练设置」）。

---

## 第二轮：M5 轨迹外推一致性 loss（M4+DACC 之上）

相对第一轮（M4+DACC）追加 **2 个文件**。M5 为纯训练期 loss，无任何网络结构改动，
推理路径不变。

### 1. `utils/temporal_frame_loss.py`

- **新增 `trajectory_extrapolation_loss_memory`**（L774-L894）：M4 专用变体。
  对每个 batch 内的正事件按 `target_id` 分组，仅当某目标已观测到
  `≥ min_known_points`（默认 3）个事件点时，用 `lstsq` 拟合线性轨迹
  （x-y-t 三通道），在尚未观测到的时间步上把外推位置投影回帧坐标，
  对该位置施加 `relu(margin_logit − logit)` 正则（`margin_logit` 默认 1.0）。
  强化小目标轨迹的时间连续性。
  - 对照组 `trajectory_extrapolation_loss`（L533）与
    `trajectory_extrapolation_loss_p23`（L626）为 P23 路径的既有实现，
    P23 上验证退化（-0.0046），本仓库只在 M4 路径启用 memory 变体。
  - 各 `TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_*` 配置键在
    `configs/evisseg_evuav.yaml`（L228-L232）已存在，无需新增。

### 2. `train_temporal_memory.py`

- **import**（L24）：`trajectory_extrapolation_loss_memory`。
- **配置读取**（L211-L238）：读 `temporal_frame_trajectory_extrapolation_enabled`、
  `weight`（默认 0.05）、`margin_logit`（默认 1.0）、`min_points`（默认 3）、
  `warmup_epochs`（默认 3），并做合法性校验（weight>0、min_points≥2、warmup≥0）。
- **loss 接线**（L346-L360）：`epoch ≥ warmup_epochs` 后每 batch 调用
  `trajectory_extrapolation_loss_memory(logit_maps, event_time_indices, event_x,
  event_y, labels, target_ids, ...)`，`loss = base_loss + trajectory_weight * trajectory_loss`。
  warmup 期 `trajectory_loss = event_logits.sum() * 0.0`（占位，backward 无梯度贡献）。
  需要 dataset 提供 `target_ids`（EV-UAV 数据已含）。
- **checkpoint 记录**（L399）：saved dict 写入 `trajectory_extrapolation_enabled`，
  供复现审计。

### 运行时开关

```yaml
TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_enabled: false   # 关闭时行为等同第一轮 M4+DACC
```

复现本仓库 0.94965 结果时，训练在 M4+DACC 命令基础上追加：
`TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_enabled=true`、
`..._weight=0.05`、`..._margin_logit=1.0`、`..._min_points=3`、
`..._warmup_epochs=3`（完整命令见 README「复现流程」第 2 步）。
