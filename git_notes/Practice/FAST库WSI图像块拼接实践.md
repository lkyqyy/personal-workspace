# FAST 库 WSI Patch 拼接实践

> 文档目的：
> 本文档用于算法/工程背景下理解 FAST 库处理全切片图像(Whole-Slide Image, WSI)的 patch 分割结果拼接技术，重点回答以下问题：
>
> * WSI 图像处理面临哪些核心挑战？
> * FAST 库如何通过 PatchGenerator 和 PatchStitcher 实现 patch 生成与拼接？
> * 如何通过 patch 重叠策略减少边缘伪影？
> * 在实际应用中需要注意哪些技术要点和最佳实践？
>
> 本文档旨在为 WSI patch-wise 分割任务提供工业级实现方案参考。

---

## 目录

- [1. WSI 图像处理面临哪些核心挑战](#1-wsi-图像处理面临哪些核心挑战)
- [2. FAST 库如何通过 PatchGenerator 和 PatchStitcher 实现 patch 生成与拼接](#2-fast-库如何通过-patchgenerator-和-patchstitcher-实现-patch-生成与拼接)
- [3. 如何通过 patch 重叠策略减少边缘伪影](#3-如何通过-patch-重叠策略减少边缘伪影)
- [4. 在实际应用中需要注意哪些技术要点和最佳实践](#4-在实际应用中需要注意哪些技术要点和最佳实践)

---

## 1. WSI 图像处理面临哪些核心挑战

WSI 图像尺寸通常极大(可达 200,000 × 100,000 像素)，无法直接加载到内存或 GPU 显存中。因此，WSI 通常以图像金字塔(Image Pyramid)形式存储，每个层级包含多个 tile(patch)。在处理这类图像时，需要采用 patch-wise 的处理策略：首先将 WSI 按固定尺寸切分为多个 patch，逐个送入模型推理；然后将各 patch 的分割结果重新拼接为完整的 mask 图像；最后通过 patch 重叠与融合策略，消除拼接边界处的伪影。

WSI 图像处理面临的核心挑战包括：内存限制(无法将整张 WSI 加载到内存或显存中)、计算效率(需要处理大量 patch，计算量大)、边界伪影(patch 边界处的预测往往不稳定，导致拼接结果出现明显边界)、上下文信息丢失(每个 patch 只能看到局部区域，缺乏全局上下文信息)、坐标管理(需要准确管理每个 patch 在全图中的位置，以及重叠区域的处理)。FAST 库提供了完整的 pipeline 支持上述流程，包括 PatchGenerator、PatchStitcher 等组件，能够高效处理大规模 WSI 图像。

---

## 2. FAST 库如何通过 PatchGenerator 和 PatchStitcher 实现 patch 生成与拼接

PatchGenerator 负责将 WSI 按指定尺寸和层级切分为多个 patch。关键参数包括 patch 尺寸(宽度和高度，如 256×256、512×512)、图像金字塔层级(通过 `level` 参数指定，0 为最高分辨率，或通过 `magnification` 指定放大倍数，如 40X、20X、10X)，以及重叠比例(通过 `overlapPercent` 或 `overlap` 参数控制 patch 之间的重叠区域，用于减少边缘伪影)。PatchGenerator 会自动管理 patch 的生成位置、尺寸和重叠信息，确保每个 patch 都有准确的元数据，便于后续拼接。

基本使用示例：

```python
import fast

# 加载 WSI
importer = fast.WholeSlideImageImporter.create(
    fast.Config.getTestDataPath() + "/WSI/CMU-1.svs"
)

# 组织分割(可选)：仅从组织区域生成 patch，跳过玻璃背景
tissueSegmentation = fast.TissueSegmentation.create().connect(importer)

# 生成 256×256 的 patch，在 20X 放大倍数下，重叠 10%
patchGenerator = fast.PatchGenerator.create(
    256, 256,
    magnification=20,
    overlapPercent=0.1
).connect(0, importer).connect(1, tissueSegmentation)
```

PatchStitcher 将 patch-wise 的分割结果拼接回完整的图像金字塔。它会自动处理重叠区域，生成与原始 WSI 尺寸匹配的分割结果。PatchStitcher 内部会根据 patch 的元数据(位置、尺寸、重叠信息)确定每个 patch 在全图中的位置；处理重叠区域的融合(通常采用加权平均或中心区域优先策略)；生成与原始 WSI 相同层级的图像金字塔，便于后续可视化或导出。

基本使用示例：

```python
# 假设 segmentation 是分割网络的输出(每个 patch 的分割结果)
segmentation = fast.SegmentationNetwork.create(
    model_path + '/high_res_nuclei_unet.onnx',
    scaleFactor=1./255.
).connect(patchGenerator)

# 将 patch 分割结果拼接为完整 mask
stitcher = fast.PatchStitcher.create().connect(segmentation)
```

以下示例展示了从 WSI 加载、patch 生成、分割推理到结果拼接的完整流程：

```python
import fast

# 1. 下载并加载分割模型
model = fast.DataHub().download('nuclei-segmentation-model')

# 2. 加载 WSI
importer = fast.WholeSlideImageImporter.create(
    fast.Config.getTestDataPath() + "/WSI/CMU-1.svs"
)

# 3. 组织分割(可选，用于过滤背景区域)
tissueSegmentation = fast.TissueSegmentation.create().connect(importer)

# 4. 生成 patch(256×256，20X 放大倍数，10% 重叠)
generator = fast.PatchGenerator.create(
    256, 256,
    magnification=20,
    overlapPercent=0.1
).connect(0, importer).connect(1, tissueSegmentation)

# 5. 对每个 patch 进行分割
segmentation = fast.SegmentationNetwork.create(
    model.paths[0] + '/high_res_nuclei_unet.onnx',
    scaleFactor=1./255.
).connect(generator)

# 6. 拼接 patch 分割结果为完整 mask
stitcher = fast.PatchStitcher.create().connect(segmentation)

# 7. 可视化：将拼接结果叠加在原始 WSI 上
fast.display2D(imagePyramid=importer, segmentation=stitcher)
```

拼接后的分割结果可以导出为 TIFF 格式的图像金字塔，便于后续分析或可视化：

```python
# 等待所有 patch 处理完成
finished = fast.RunUntilFinished.create().connect(stitcher)

# 导出为 TIFF 图像金字塔
exporter = fast.TIFFImagePyramidExporter.create('segmented-nuclei.tiff')\
    .connect(finished)\
    .run()
```

---

## 3. 如何通过 patch 重叠策略减少边缘伪影

在 patch-wise 分割中，模型在 patch 边缘区域的预测往往不稳定，原因包括：边缘区域缺乏足够的上下文信息；目标可能被 patch 边界截断；模型在训练时主要关注 patch 中心区域的特征。因此需要通过重叠策略来减少边缘伪影。

当设置 `overlapPercent=0.1` 时，每个 patch 的四周会有约 10% 的重叠区域。例如，对于 256×256 的 patch，重叠区域约为 25.6 像素(四舍五入为 26 像素)。这意味着相邻 patch 之间有 26 像素的公共区域，在拼接时可以通过加权融合等方式处理这些重叠区域，从而减少边界处的预测不一致问题。

重叠比例选择：10% 重叠适用于大多数场景，在计算效率与拼接质量之间取得平衡；20-30% 重叠适用于对边界精度要求较高的任务，但会增加计算量；0% 重叠仅适用于快速预览或对精度要求不高的场景。

PatchStitcher 内部会自动处理重叠区域，通常采用以下策略之一：中心区域优先(重叠区域使用中心 patch 的预测结果)、加权平均(根据像素到 patch 中心的距离进行加权融合)、最大值/平均值(对重叠区域的预测值取最大值或平均值)。这些策略能够有效减少边界处的伪影，提升拼接结果的连续性和准确性。

---

## 4. 在实际应用中需要注意哪些技术要点和最佳实践

在实际应用中，WSI 图像中往往包含大量空白背景(玻璃区域)，这些区域不包含组织信息，无需进行分割处理。通过 TissueSegmentation 可以减少计算量(只对组织区域生成 patch，跳过背景区域)、提升模型性能(避免模型在背景区域产生误检)、优化存储(导出的分割结果仅包含组织区域，文件更小)。

放大倍数与层级选择：高分辨率(40X 或 level=0)适用于需要精细分割的任务(如细胞核分割)，但计算量大；中等分辨率(20X 或 level=1-2)适用于大多数分割任务，在精度与效率之间平衡；低分辨率(10X 或更高层级)适用于快速预览或粗分割任务。选择合适的放大倍数需要在精度要求和计算资源之间进行权衡。

性能优化建议：如果模型支持 batch 推理，可以在 PatchGenerator 后添加 batch 处理组件；对于大规模 WSI，可以考虑并行处理多个 patch；确保分割模型在 GPU 上运行，充分利用硬件加速；对于超大 WSI，注意控制同时处理的 patch 数量，避免内存溢出。

该实现方案与论文中描述的 patch 分割后处理逻辑高度一致：Patch 生成对应论文中的滑动窗口裁剪策略，通过 PatchGenerator 实现；重叠处理对应论文中的 overlap 策略，通过 `overlapPercent` 参数控制；结果融合对应论文中的概率图融合或加权平均，由 PatchStitcher 内部实现；边缘平滑通过重叠区域的加权融合，实现与论文中高斯加权或中心区域优先类似的平滑效果。相比手动实现 patch 拼接，FAST 库的方案具有以下优势：自动化处理(无需手动管理 patch 坐标、重叠区域计算等细节)、内存高效(基于图像金字塔的存储方式，支持流式处理大规模 WSI)、工业级稳定(经过实际项目验证，可直接用于生产环境)。
