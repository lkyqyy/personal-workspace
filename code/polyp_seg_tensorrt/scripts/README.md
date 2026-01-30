# scripts 脚本说明

脚本用于：PyTorch 模型导出 → ONNX → TensorRT engine，以及验证集准备、精度对比与锚点校验。

---

## 1. 典型使用顺序

| 步骤 | 脚本 | 说明 |
|------|------|------|
| 0 | 训练 | 得到 `train/snapshots/best.pth`（或自定义 .pt） |
| 1 | `export_pt_to_onnx.py` | .pt → .onnx |
| 2 | `export_onnx_to_engine.py` | .onnx → .engine（TensorRT） |
| 3 | `prepare_val_for_cpp.py` | 把验证集拷/链到 `data/images`、`data/masks` |
| 4 | `eval_val_set.py` | 在 data 上跑 PyTorch，输出 mean Dice/IoU，与 C++ 对比 |
| 5 | （可选）`generate_anchor_data.py` | 导出 input/output bin，供 C++ 逐元素对比 |

---

## 2. 各脚本简介与常用参数

### export_pt_to_onnx.py

- **作用**：将 PyTorch UNet 分割模型 (.pt) 导出为 ONNX，供 TensorRT 等使用。
- **依赖**：与 `train/models/unet.py` 结构一致；可选 `onnx-simplifier`。
- **常用**：
  - `--pt`：输入 .pt 路径（默认 `models/unet_polyp_seg.pth`）
  - `--onnx`：输出 .onnx 路径（默认 `models/unet_polyp_seg.onnx`）
  - `--height/--width`：输入分辨率，需与 `train/config.py` 的 IMG_SIZE 一致（默认 256）
  - `--sigmoid`：输出端包 Sigmoid（与 C++ 端一致时建议加）
  - `--dynamic`：动态 batch/H/W（默认）；固定用 `--static`
  - `--simplify`：导出后用 onnxsim 精简

### export_onnx_to_engine.py

- **作用**：ONNX 转 TensorRT engine，供 C++ 推理。
- **依赖**：TensorRT Python 包（随 CUDA 或 pip 安装）。
- **常用**：
  - `--onnx`：输入 .onnx（默认 `models/unet_polyp_seg.onnx`）
  - `--engine`：输出 .engine；不传则按 fp16/fp32 自动命名
  - `--fp16`：启用 FP16；不加则 FP32
  - `--min_shapes/--opt_shapes/--max_shapes`：动态时必填，格式 `N,H,W`（如 `1,256,256` / `4,256,256` / `8,256,256`）
  - `--workspace`：builder 工作空间（GB）

### prepare_val_for_cpp.py

- **作用**：把验证集准备到 `data/images` 与 `data/masks`，供 C++ 精度校验；与 `generate_anchor_data` 使用同一套 val 划分。
- **常用**：
  - `--data-root`：训练数据根（默认 `train/data`）
  - `--out-dir`：输出目录（默认 `data`），下建 images/、masks/
  - `--symlink`：用符号链接代替拷贝

### eval_val_set.py

- **作用**：在 `data/images` + `data/masks` 上跑 PyTorch 推理，输出 mean Dice / mean IoU，便于与 C++ TensorRT 结果对比。
- **前置**：先运行 `prepare_val_for_cpp.py`。
- **常用**：
  - `--data-dir`：含 images/、masks/ 的目录（默认 `data`）
  - `--pt`：.pt 权重（默认 `train/snapshots/best.pth`）
  - `--match-training`：在 256×256 空间算 Dice（与训练验证一致）

### generate_anchor_data.py

- **作用**：生成 Binary Validation 锚点数据：导出 `input_tensor.bin`、`output_prob.bin` 及元信息，供 C++ TensorRT 推理后逐元素对比，用于隔离「引擎正确性」与「前后处理正确性」。
- **常用**：
  - `--data-root`：数据根，用于取验证集（默认 `train/data`）
  - `--image-dir`：可选，直接指定图像目录（不区分 train/val，慎用）
  - `--num`：选取图像数（默认 1；C++ 锚点校验仅支持 batch=1）
  - `--pt`：.pt 路径（与导出 ONNX 一致）
  - `--out-dir`：输出目录（默认 `models/anchor`），写入 input_tensor.bin、output_prob.bin、anchor_meta.txt、anchor_images.txt
  - `--sigmoid`：与导出 ONNX 时一致则加
  - `--seed`：随机选图种子

---

## 3. 路径与约定

- 验证集：有 `images/val/` 则用该目录；否则从 `images/*.jpg` 按 seed 做 80/20 划分，后 20% 为 val。
- Mask：与 `train.dataset` 约定一致（如 `masks/` 下与图像同名的 .png）。
- 归一化：推理时 mean/std 需与 `train/config.py`（IMG_MEAN、IMG_STD）一致；C++ 侧通道建议统一为 RGB。

---

## 4. 快速命令示例

```bash
# 导出 ONNX（带 sigmoid）
python3 scripts/export_pt_to_onnx.py --sigmoid

# ONNX → engine（FP16，动态 1~8 batch）
python3 scripts/export_onnx_to_engine.py --fp16

# 准备验证集到 data/
python3 scripts/prepare_val_for_cpp.py

# PyTorch 在 data 上评估
python3 scripts/eval_val_set.py

# 生成 1 张图的锚点数据到 models/anchor
python3 scripts/generate_anchor_data.py --sigmoid
```
