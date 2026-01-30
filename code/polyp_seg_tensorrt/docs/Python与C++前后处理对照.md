# Python 与 C++ 前后处理对照清单

用于排查 PyTorch 验证集指标与 C++ TensorRT 指标差异（如 Dice 差 ~0.03）。  
对照脚本：`scripts/eval_val_set.py`（默认在原图空间评估）、C++ `Preprocessor` / `Postprocessor` / `Metrics`。

---

## 1. 预处理

| 项 | Python (eval_val_set + dataset) | C++ (Preprocessor) | 是否一致 |
|---|--------------------------------|--------------------|----------|
| **输入尺寸** | `IMG_SIZE = 256` | `Preprocessor(256, 256)` | ✅ 一致 |
| **Letterbox 缩放** | `scale = 256 / max(w,h)`，`new_w = int(round(w*scale))`，`new_h = int(round(h*scale))` | `scale = h_/max(raw_w,raw_h)`，`new_w = round(raw_w*scale)`，`new_h = round(raw_h*scale)` | ✅ 一致 |
| **padding 偏移** | `x0 = (256 - new_w) // 2`，`y0 = (256 - new_h) // 2` | `x0 = (w_ - new_w)/2`，`y0 = (h_ - new_h)/2` | ✅ 一致 |
| **缩放插值** | PIL `Image.resize((new_w,new_h), Image.BILINEAR)` | C++ `resizePilBilinear`（首末像素对齐，与 PIL 一致） | ✅ 已对齐 |
| **通道顺序** | PIL `RGB` → `(H,W,3)` → `(3,H,W)`，即模型输入为 RGB | `cv::imread` 得 BGR → `cvtColor(BGR2RGB)` → buffer 为 RGB | ✅ 一致（均为 RGB 进模型） |
| **归一化** | `(x/255 - mean) / std`，`mean=[0.485,0.456,0.406]`，`std=[0.229,0.224,0.225]` | 同上，`mean_/std_` 与 config 一致 | ✅ 一致 |
| **Buffer 布局** | NCHW，`(1,3,256,256)` | CHW 写入 `buffer[c*area + i]`，即 NCHW | ✅ 一致 |

**结论（预处理）**：逻辑一致。C++ 已改用 **PIL 风格 bilinear**（首末像素对齐），与 Python 插值一致，input.bin MAE 应进一步缩小。

---

## 2. 后处理

| 项 | Python (eval_val_set.postprocess) | C++ (Postprocessor) | 是否一致 |
|---|-----------------------------------|---------------------|----------|
| **二值化阈值** | `logit[0,0] > 0.5` → 255 | `cv::threshold(..., 0.5, 255, BINARY)` | ✅ 一致 |
| **裁掉 padding** | `pred[y0:y0+new_h, x0:x0+new_w]` | `prob_mat(Rect(x0,y0,new_w,new_h))` | ✅ 一致 |
| **还原原图尺寸** | `Image.fromarray(crop).resize((raw_w,raw_h), Image.NEAREST)` | `cv::resize(mask, ..., INTER_NEAREST)` | ✅ 一致（均为最近邻） |

**结论（后处理）**：逻辑一致，无发现差异。

---

## 3. 指标计算（Dice / IoU）

| 项 | Python (eval_val_set.dice / iou) | C++ (Metrics::calculateDice / IOU) | 是否一致 |
|---|----------------------------------|------------------------------------|----------|
| **前景判定** | `pred > 127`，`gt > 127` | `maskPred > 127`，`maskGT > 127` | ✅ 一致 |
| **Dice 公式** | `2*inter / (sum_p + sum_g)`，空为 1.0 | 同上 | ✅ 一致 |
| **IoU 公式** | `inter / (sum_p + sum_g - inter)`，空为 1.0 | 同上 | ✅ 一致 |

**结论（指标）**：一致。

---

## 4. 评估空间差异（易被忽略）

| 模式 | Python | C++ (main.cpp Step 2) |
|------|--------|------------------------|
| **默认** | `postprocess(prob, ...)` 后在原图尺寸上与 GT 算 Dice | `process_with_info` 得到原图尺寸 mask，与 GT 算 Dice | ✅ 一致（均为原图空间） |
| **match_training** | `--match-training` 时在 256×256 上算 Dice（GT 也 letterbox 到 256×256） | 当前 C++ 未提供该模式 | 仅 Python 有 |

对比 Py vs C++ 时，**不要**开 `--match-training`，否则 Python 在 256×256 上算、C++ 在原图空间算，不可比。

---

## 5. 小结与建议

- **前后处理与指标**：未发现逻辑不一致；PIL vs OpenCV 的 resize 可能带来张量级 ~1e-3 MAE，属正常。
- **若 Dice 仍差 ~0.03**：  
  1. 确认两边用**同一批图**（同一 `data/images` + `data/masks`）。  
  2. **图片顺序**：Python 为 `sorted(*.jpg) + sorted(*.png)`，C++ 为 `cv::glob` 未排序；均值不受影响，但逐张对比时应对 C++ 的 `image_paths` 做 `std::sort` 与 Python 对齐。  
  3. 抽 1～2 张图，在 Python 和 C++ 各保存**预处理后输入**（或归一化后的统计）和**二值化前的 prob**，逐像素比对，看差异出在预处理阶段还是引擎阶段。

**可选后续**：在 C++ 侧对同一张图打印或写出 `ImageInfo`（raw_w, raw_h, new_w, new_h, x0, y0），与 Python `_letterbox_params` 输出对比，进一步确认 letterbox 完全对齐。

---

## 6. 单图对比调试（定位差异阶段）

对验证集**前五张图**（或指定张数）在 Python 和 C++ 各跑一遍，保存相同产物后逐项对比：

1. **Python**（在项目根下执行，默认前 5 张）：
   ```bash
   python3 scripts/debug_single_image.py [--num 5] [--out debug_py]
   ```
   产物在 `debug_py/img_0` .. `debug_py/img_4`：各子目录下 `input.bin`、`prob.bin`、`mask.png`、`letterbox.txt`、`input_stats.txt`。脚本会打印对应的 C++ 命令。

2. **C++**（自动取验证集前 5 张，与 Python 顺序一致）：
   ```bash
   ./bin/PolypSegmentationTRT --first-5
   ```
   会从 `data/images` 自动 glob 并排序后取前 5 张，产物在 `debug_cpp/img_0` .. `debug_cpp/img_4`，与 `debug_py/img_*` 一一对应。也可手动传入路径：`./bin/PolypSegmentationTRT 图1 图2 ... 图5`。

3. **自动对比**（推荐）：
   ```bash
   python3 scripts/compare_debug_outputs.py [--py debug_py] [--cpp debug_cpp] [--num 5]
   ```
   会逐张输出：letterbox 是否一致、input.bin MAE/max、prob.bin MAE/max、mask 像素一致率，并在最后给出简要结论（差异在预处理 / 引擎 / 后处理）。

4. **手动对比**（可选）：
   - `letterbox.txt`：应完全一致（raw_w raw_h new_w new_h x0 y0）。
   - `input_stats.txt`：逐通道 min/max/mean，若差异大则问题在预处理。
   - `input.bin`：逐元素 diff（或 MAE），若差异大则问题在预处理/插值。
   - `prob.bin`：逐元素 diff（或 MAE），若差异大则问题在引擎。
   - `mask.png`：视觉或像素一致率，若 input/prob 都接近而 mask 差则问题在后处理。

---

## 7. 验证集对比结果（实测）

**环境**：验证集 704 张（`data/images` + `data/masks`），ONNX 含 Sigmoid（`check_onnx_has_sigmoid.py` 确认）。

| 阶段 | Python (PyTorch) | C++ (TensorRT) | Dice 差值 | IoU 差值 |
|------|------------------|----------------|-----------|----------|
| C++ 用 `cv::INTER_LINEAR` | Mean Dice 0.8184，Mean IoU 0.7430 | 0.7889，0.7095 | -0.0295 | -0.0335 |
| C++ 改用 PIL 风格 bilinear | 同上 | 0.8028，0.7253 | -0.0156 | -0.0177 |

**提升量化**（C++ 改用 PIL 风格 bilinear 后）：

| 指标 | 改前 (INTER_LINEAR) | 改后 (PIL bilinear) | 绝对提升 | 相对提升 | 与 Python 差值缩小 |
|------|---------------------|----------------------|----------|----------|--------------------|
| Mean Dice | 0.7889 | 0.8028 | +0.0139 | 约 +1.8% | -0.0295 → -0.0156（约 47%） |
| Mean IoU  | 0.7095 | 0.7253 | +0.0158 | 约 +2.2% | -0.0335 → -0.0177（约 47%） |

**结论**：

- ONNX 输出端含 Sigmoid 后，C++ 与 Python 的 prob 尺度一致，prob.bin MAE 从 ~9 降到 ~0.002。
- C++ 预处理改用 **PIL 风格 bilinear**（首末像素对齐）后，验证集 Dice/IoU 差值由 ~0.03 降到 **≤0.02**，部署对齐良好。
- 复现命令：`python3 scripts/compare_val_set.py --data-dir data --pt train/snapshots/best.pth --bin bin/PolypSegmentationTRT`。
