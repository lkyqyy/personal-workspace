# 个人技术笔记

记录日常开发中遇到的问题和解决方案，以及相关论文阅读笔记和项目经验总结。笔记按主题分类存放在 Papers（论文阅读）、Practice（项目实践）和 Figures（图片资源）目录中。

## 最近更新

<!-- 按时间倒序，最新在前，格式：YYYYMMDD 问题描述 -->
- 20260110 [An Effective Pipeline for Whole-Slide Image Glomerulus Segmentation逐点精读](Papers/An_Effective_Pipeline_for_Whole-Slide_Image_Glomerulus_Segmentation_逐点精读.md) - Cap, 2024 - 全切片图像肾小球分割、重叠补丁拼接、有效流程
- 20260110 [Ultra-High Resolution Image Segmentation via Locality-Aware Contextual Correlation逐点精读](Papers/Ultra-High_Resolution_Image_Segmentation_Locality-Aware_Contextual_Correlation_逐点精读.md) - Li et al., 2021 - 超高分辨率分割、局部感知上下文相关性、上下文语义精化
- 20260110 [Ultra-High Resolution Image Segmentation逐点精读](Papers/Ultra-High_Resolution_Image_Segmentation_逐点精读.md) - Liu et al., 2021 - 超高分辨率分割、局部感知上下文融合、交替局部增强
- 20260110 [CascadePSP逐点精读](Papers/CascadePSP_逐点精读.md) - 类别无关高分辨率分割、全局局部精炼
- 20260110 [MNet-SAt逐点精读](Papers/MNet-SAt_逐点精读.md) - 结肠镜息肉分割、多尺度网络、空间增强注意力
- 20260110 [TPCP-Net逐点精读](Papers/TPCP-Net逐点精读_BiomedicalSignalProcessing.md) - 息肉分割、信息增强网络
- 20260110 [µ-Net逐点精读](Papers/µ-Net_Framework_with_Explainable_AI_逐点精读.md) - 结直肠息肉分割、可解释AI
- 20260110 [Annotation-Efficient Polyp Segmentation逐点精读](Papers/Annotation-Efficient_Polyp_Segmentation_via_Active_Learning_逐点精读.md) - 主动学习、标注高效息肉分割
- 20260110 [HFRF-Net逐点精读](Papers/HFRF-Net逐点精读.md) - 双任务层次特征细化融合网络、内窥镜工具和息肉分割
- 20260109 [Vahadane: 病理图像颜色归一化逐点精读](Papers/Vahadane病理图像颜色归一化逐点精读.md) - 结构保持、稀疏非负矩阵分解
- 20260109 [UNet++逐点精读](Papers/UNet++逐点精读_2018.md) - 嵌套U-Net、密集跳跃连接
- 20260109 [U-Net逐点精读](Papers/U-Net逐点精读_MICCAI2015.md) - U形架构、跳跃连接、生物医学图像分割
- 20260109 [Cell Detection with Star-convex Polygons逐点精读](Papers/StarDist_Detailed_Reading_MICCAI2018.md) - 星形凸多边形、实例分割
- 20260109 [DeepLabv3+逐点精读](Papers/DeepLabv3+逐点精读_2018.md) - 编码器-解码器、空洞可分离卷积
- 20260101 [眼底图像视网膜血管分割](Practice/Fundus_Retinal_Vessel_Segmentation.md) - 血管分割、多层空间注意力、多分辨率融合
- 20260101 [眼底图像质量分级预测](Practice/Fundus_Image_Quality_Grading.md) - 图像质量评估、锐度感知、多色域融合
- 20260101 [眼底图像糖尿病视网膜病变分级预测](Practice/Fundus_Diabetic_Retinopathy_Grading.md) - 病变分级、特征自适应过滤

## 分类索引

### 论文阅读笔记

#### 逐点精读

##### 息肉分割

- [MNet-SAt逐点精读](Papers/MNet-SAt_逐点精读.md) - Singh Raghaw et al., 2024 - 结肠镜息肉分割、多尺度网络、空间增强注意力
- [TPCP-Net逐点精读](Papers/TPCP-Net逐点精读_BiomedicalSignalProcessing.md) - 2025 - 息肉分割、信息增强网络
- [µ-Net逐点精读](Papers/µ-Net_Framework_with_Explainable_AI_逐点精读.md) - Emon et al., 2025 - 结直肠息肉分割、可解释AI
- [Annotation-Efficient Polyp Segmentation逐点精读](Papers/Annotation-Efficient_Polyp_Segmentation_via_Active_Learning_逐点精读.md) - Huang et al., 2024 - 主动学习、标注高效息肉分割
- [HFRF-Net逐点精读](Papers/HFRF-Net逐点精读.md) - 双任务层次特征细化融合网络、内窥镜工具和息肉分割

##### 图像分割基础架构

- [Ultra-High Resolution Image Segmentation via Locality-Aware Contextual Correlation逐点精读](Papers/Ultra-High_Resolution_Image_Segmentation_Locality-Aware_Contextual_Correlation_逐点精读.md) - Li et al., 2021 - 超高分辨率分割、局部感知上下文相关性、上下文语义精化
- [Ultra-High Resolution Image Segmentation逐点精读](Papers/Ultra-High_Resolution_Image_Segmentation_逐点精读.md) - Liu et al., 2021 - 超高分辨率分割、局部感知上下文融合、交替局部增强
- [CascadePSP逐点精读](Papers/CascadePSP_逐点精读.md) - Cheng et al., 2020 - 类别无关超高分辨率分割、级联全局局部精炼
- [U-Net逐点精读](Papers/U-Net逐点精读_MICCAI2015.md) - Ronneberger et al., 2015 - U形架构、跳跃连接、生物医学图像分割
- [DeepLabv3+逐点精读](Papers/DeepLabv3+逐点精读_2018.md) - Chen et al., 2018 - 编码器-解码器、空洞可分离卷积
- [UNet++逐点精读](Papers/UNet++逐点精读_2018.md) - Zhou et al., 2018 - 嵌套U-Net、密集跳跃连接

##### 其他

- [Cell Detection with Star-convex Polygons逐点精读](Papers/StarDist_Detailed_Reading_MICCAI2018.md) - Schmidt et al., 2018 - 星形凸多边形、实例分割
- [Vahadane: 病理图像颜色归一化逐点精读](Papers/Vahadane病理图像颜色归一化逐点精读.md) - Vahadane et al., 2016 - 结构保持、稀疏非负矩阵分解

### 项目实践

- [数字病理图像免疫组化癌区肿瘤细胞检测（ER）](Practice/Digital_Pathology_IHC_ER_Cell_Detection.md) - ER表达检测、大尺寸图像推理、TensorRT加速
- [眼底图像视网膜血管分割](Practice/Fundus_Retinal_Vessel_Segmentation.md) - 血管分割、多层空间注意力、多分辨率融合
- [眼底图像糖尿病视网膜病变分级预测](Practice/Fundus_Diabetic_Retinopathy_Grading.md) - 病变分级、特征自适应过滤
- [眼底图像质量分级预测](Practice/Fundus_Image_Quality_Grading.md) - 图像质量评估、锐度感知、多色域融合
- [大尺寸医学图像分割与后处理](Practice/Large_Scale_Medical_Image_Segmentation_Postprocessing.md) - 滑窗分割、概率图融合、后处理策略

## 使用说明

1. **文件命名**: 使用下划线命名，格式为 `主题_描述.md`
2. **索引更新**: 在对应目录的 README.md 中添加索引链接
3. **时间记录**: 按时间倒序更新主 README.md 的"最近更新"部分
4. **图片管理**: 图片按主题分类存放在 `Figures/` 对应子目录

## 快速导航

- [论文阅读笔记](Papers/README.md) - 深度学习、医学图像处理论文逐点精读
- [项目实践笔记](Practice/README.md) - 实际项目经验总结和技术方案
- [图片资源](Figures/README.md) - 论文和项目相关的图片资源

---
