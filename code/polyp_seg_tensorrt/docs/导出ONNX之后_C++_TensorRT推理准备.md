# 导出 ONNX 之后：C++ TensorRT 高性能推理的准备工作

**前置**：.pt 已训练完毕并完成 ONNX 导出（参见 [导出脚本_写代码逻辑](./导出脚本_写代码逻辑.md)）。本文档说明在动手写 C++ TensorRT 推理前，需要完成的**准备工作**，以及 C++ 推理打通后的**精度-性能平衡验证**——目的不仅是“跑通”，而是确保在 C++ 环境下榨干硬件性能且精度不掉点。

文档按五个维度组织：模型层、数据层、环境层、工程细节、精度-性能验证。

---

## 一、模型层：ONNX 的“终极体检”

ONNX 是中间产物，其质量直接决定 TensorRT 生成 Engine 的优劣。在进入 TensorRT 构建前，需完成以下准备。

**静态 vs 动态 Shape**  
先明确生产环境输入尺寸。若能固定（如全部 resize 到 352×352），则导出**静态 ONNX**，TensorRT 在构建时可做更深度的算子融合与 Kernel 选择，性能最优。若必须支持多分辨率或动态 batch，则准备好 **min / opt / max** 三组尺寸，供 TensorRT 的 profile 使用。

**ONNX 算子简化**  
必须运行 **onnx-simplifier**。PyTorch 导出常带冗余的 Shape、Gather 等节点，在 TensorRT 动态推理中会带来额外开销；简化后可消除这类节点，图更干净、解析与优化更稳定。导出脚本若已支持 `--simplify`，构建 C++ 前确认已对当前 ONNX 执行过简化。

**输入输出节点名称**  
用 **Netron** 打开 ONNX，记录 Input 和 Output 的 **Name**（字符串）。C++ 中绑定显存、设置 binding 时，必须用这些字符串；写死或猜错会导致运行时报错或结果错乱。

---

## 二、数据层：定义“金标准”对齐数据

用于避免“C++ 推理结果与 Python 不一致”的排查困境。提前准备好对齐数据，可快速区分问题来自推理引擎还是预处理/后处理。

**导出权重统计量**  
记录训练时使用的 **Mean 和 Std**（建议精确到小数点后 6 位）。C++ 预处理中的归一化必须与训练完全一致，否则必然掉点；先有精确数值，再在代码或配置中写死/加载。

**生成“锚点”数据（Binary Validation）**  
挑选 5～10 张典型息肉图像，在 **Python 环境**下导出两份二进制文件：

- **input_tensor.bin**：预处理后、送入模型前的 float32 数据（shape 与 ONNX 输入一致，如 NCHW）。
- **output_prob.bin**：模型输出、未做阈值处理的原始 float32 概率图。

用途：在 C++ 中先加载 `input_tensor.bin` 做一次推理，将输出与 `output_prob.bin` 逐元素对比。若一致，说明推理引擎与 Python 一致；若不一致，再排查 C++ 预处理（resize、归一化、通道顺序等）。这一步是隔离“引擎正确性”与“前后处理正确性”的关键。

**脚本与格式**：运行 `scripts/generate_anchor_data.py`（需与导出 ONNX 相同的 Python 环境）。**默认仅从验证集选取**（与 `train.dataset.create_dataloaders` 的划分一致：有 `images/val` 则用该目录，否则对 `images/*.jpg` 按 80/20 划分取后 20%），避免数据泄露；默认 8 张、权重 `train/snapshots/best.pth`、输出 `models/anchor/`。若 ONNX 导出时用了 `--sigmoid`，本脚本需加 `--sigmoid` 以保证输出为概率。二进制格式：文件前 4 个 int32 为 shape (N,C,H,W)，随后为 C 序 float32 数据；另生成 `anchor_meta.txt`（N/H/W/mean/std）与 `anchor_images.txt`（所用图像文件名列表）。

---

## 三、环境层：构建高性能脚手架

高性能推理不单是“调通 TensorRT API”，还涉及显存管理与多线程/异步设计。

**TensorRT 运行时**  
确保 **CUDA、cuDNN、TensorRT** 版本匹配。推荐 **TensorRT 8.6+**，对 `enqueueV3` 与动态形状的支持更简洁稳定，便于后续维护。

**内存/显存池**  
根据模型**最大输入尺寸**预计算所需显存（含输入/输出 buffer 与中间激活）。在类似 **memory_manager.cpp** 的模块中，实现 **Pinned Memory（锁页内存）** 的申请与复用逻辑。高性能推理严禁在推理循环内 `malloc`/`new`，应一次性申请、循环内复用，避免 CPU-GPU 拷贝与分配带来的抖动。

**CUDA Stream 规划**  
设计好异步流：准备好 `cudaStream_t`，实现“CPU 预处理下一张图的同时，GPU 推理当前张图”的流水。这是把吞吐做上去的前提，需要在写 InferenceEngine 前就定好 stream 与 buffer 的对应关系。

---

## 四、工程细节：预防性“避坑”逻辑

在写 InferenceEngine 主逻辑前，先想清楚以下三点，可减少返工。

**Resize 对齐策略**  
确认训练时 **PyTorch `F.interpolate`** 使用的对齐方式（如 `align_corners=False`）。C++ 侧若用 OpenCV，需选对插值方式（如 `INTER_LINEAR`），或自写 CUDA 线性插值时，采样点公式与 PyTorch 完全一致，否则边界和细节易掉点。导出文档中已强调 Up 里显式写死 `align_corners=False`，C++ 端需与之对应。

**后处理阈值**  
针对当前模型（如 Dice 0.8194），可基于验证集画 **PR 曲线**，看 0.5 是否仍是最优阈值。C++ 中不要写死 0.5，建议预留从**配置文件**读取的 `threshold` 参数，便于后续调参与 A/B 对比。

**错误处理（Logger）**  
TensorRT 内部错误**不会**抛出 C++ 异常，只能通过 **Logger** 回调获取。实现一个继承 **`nvinfer1::ILogger`** 的类，在回调中打印或落盘；否则构建或运行时报错时难以定位，排查成本高。

---

## 五、精度-性能平衡验证（Accuracy-Performance Trade-off Validation）

锚点数据只能验证“代码跑得对不对”（逐元素对比）；**精度测量**则验证“量化（FP16/INT8）后的模型还能不能用”。这一步是整个工程开发的分水岭：工业界称之为 **Accuracy-Performance Trade-off Validation**。针对 Dice 0.8194 级别的模型，需从以下三个维度构建精度防线，并辅以防损锦囊与推荐步骤。

### 5.1 FP32 vs FP16 双轨验证

TensorRT 最明显的性能提升来自 **FP16**（半精度），但半精度表示范围窄，容易在 U-Net 深层特征图或 Sigmoid 附近的极小数值上产生截断误差。

**操作准备**  
用 `scripts/export_onnx_to_engine.py` 编译两个 Engine：`--fp32` 与 `--fp16`（输出如 `*_fp32.engine`、`*_fp16.engine`）。

**指标对撞**  
用**验证集**图片分别跑这两个 Engine，在 C++ 侧计算 Dice（与 Python 训练时同一指标定义）。  
- 若 **FP32 的 Dice 几乎等于 Python 的 0.8194**（误差建议在 ±0.001 以内），说明转换与算子映射正确。  
- 若 **FP16 的 Dice 掉到 0.80 以下**（掉点 > 1%），说明该模型对数值精度敏感，不能简单全图 FP16，需考虑在 TensorRT 的 `IBuilderConfig` 中**对部分层强制 FP32**（或混合精度策略）。

### 5.2 对抗“像素偏移”：Resize 的二次确认

息肉分割属于**稠密预测**（Dense Prediction），U-Net 中大量 `F.interpolate` 在 C++ 中若处理不当，会导致 Mask 出现 1～2 像素位移。在 256×256 或 352×352 分辨率下，边缘处 1 像素偏移就可能导致 Dice 明显波动。

**精度风险**  
C++ 预处理/后处理中的 resize、坐标变换若与 PyTorch 的 `align_corners=False` 或 half_pixel 逻辑不一致，会出现整体向左上或右下的系统性偏移。

**准备工作**  
- 指标不只看 **Dice**，同时看 **IoU** 和**边缘 F-measure**，便于发现边缘错位。  
- **对齐手段**：对比 Python 输出的 Mask 轮廓与 C++ 输出的 Mask 轮廓（叠加或差分）。若 C++ Mask 整体有固定方向位移，优先检查 C++ 预处理/后处理中的 half_pixel、resize 插值方式是否与训练一致。

### 5.3 自动化指标测试脚本（Validation Pipeline）

不能只靠手动对比几张图。需要写一个 **C++ 验证程序**，循环读取验证集目录，对每张图推理并累加 Dice（及 IoU、边缘 F-measure），最后输出平均指标。

**代码结构建议**（如 `examples/test_val_dataset.cpp` 或等价模块）：

- 遍历验证集图像路径；对每张图：读原图、读对应 GT mask；调用 `InferenceEngine::predict(img, mask_pred)`；用 `mask_pred`（建议仍为 float 概率图）与 `mask_gt` 计算当前图的 Dice（及 IoU 等）；累加。  
- 循环结束后输出 `Final TensorRT Dice: xxx`（及 IoU、边缘 F-measure）。  
- 这样可对 FP32/FP16 两个 Engine 分别跑一遍，得到可复现的指标对比。

### 5.4 防止精度损失的锦囊妙计

- **不要过早做 uint8 转换**  
  在计算 Dice 等指标时，尽量用模型输出的 **float 概率图**与 GT 对比（或先用概率图算 Dice 再 threshold 可视化）。过早将概率图 threshold 成 0/255 的 uint8 会掩盖模型层面的微小精度抖动，不利于排查 FP16/量化问题。

- **检查 Batch Normalization**  
  导出 ONNX 时必须处于 **`model.eval()`** 模式。若漏掉，TensorRT 推理会使用错误的均值和方差，精度可能雪崩。导出脚本与文档中已强调，上线前再确认一次。

### 5.5 推荐步骤（先 FP32 再 FP16）

1. **先跑 FP32 验证**  
   确保 C++ 推理逻辑能复现 Python 的 0.8194（误差在 ±0.001 以内）。若达不到，先排查预处理、后处理、指标计算方式，再考虑量化。

2. **再跑 FP16 验证**  
   观察 Dice（及 IoU、边缘 F-measure）变化。若掉点严重，再通过 TensorRT 的 `IBuilderConfig` 对敏感层锁定 FP32，或采用混合精度策略。

---

## 小结

| 维度           | 核心准备项 / 验证项 |
|----------------|---------------------|
| 模型层         | 静态/动态 Shape 决策；onnx-simplifier；Netron 记录输入输出 Name |
| 数据层         | 训练用 Mean/Std 精确记录；Python 导出 input_tensor.bin / output_prob.bin 做二值校验 |
| 环境层         | CUDA/cuDNN/TensorRT 版本匹配；显存与 Pinned Memory 池；CUDA Stream 与流水设计 |
| 工程细节       | Resize 与 PyTorch 对齐；阈值可配置；实现 ILogger 以捕获 TensorRT 报错 |
| 精度-性能验证  | FP32/FP16 双轨 Dice 对撞；Dice + IoU + 边缘 F-measure；自动化 Validation Pipeline；指标用 float 概率图、确认 BN eval |

完成上述准备并打通 C++ 推理后，按“先 FP32 再 FP16”的顺序做精度-性能平衡验证，可以少踩坑、更容易做到“性能榨干、精度不掉点”。
