# CHANGES — M4 支持 DACC 的代码改动

相对官方 EVC（`Picasso9jiu/EVC`, evc-main 分支）P23 全事件流时序帧方案的改动。
共 **3 个文件**：让 `BidirectionalTemporalMemoryNet`（M4）贯通
`DensityAdaptiveChannelCalibrator`（DACC / M3v3），使 M4+DACC 可联合训练，
并在加载/推理时正确还原 DACC 状态。

`docs/CHANGES.md` 里的 baseline 特指「官方 P23 temporal-frame 方案」。本仓库的
M4、DACC 模块本身即新增代码（`model/temporal_memory_net.py`、
`model/temporal_frame_net.py`），此处只记录为了让二者叠加所需的对既有文件的最小改动。

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
