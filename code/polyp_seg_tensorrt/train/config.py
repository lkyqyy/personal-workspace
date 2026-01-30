"""
训练配置：路径相对当前项目根（polyp_seg_tensorrt）。
"""
from pathlib import Path

# 项目根目录 = 本仓库根（polyp_seg_tensorrt）
REPO_ROOT = Path(__file__).resolve().parent.parent
# 训练包目录（train）
PROJECT_ROOT = Path(__file__).resolve().parent

# 数据根目录：train/data；支持两种结构：
# 1) images/{train,val,test}/*.jpg, masks/{train,val,test}/*.png
# 2) 扁平：images/*.jpg, masks/*.png（按比例 8:2 划分 train/val）
DATA_ROOT = REPO_ROOT / "train" / "data"

# 模型保存目录；最佳权重与导出模型统一命名：unet 息肉分割
MODEL_NAME = "unet_polyp_seg"
SAVE_DIR = PROJECT_ROOT / "snapshots"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 图像与通道
IMG_SIZE = 256
IN_CHANNELS = 3
NUM_CLASSES = 1

# 训练超参
BATCH_SIZE = 8
NUM_EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = "cuda"
NUM_WORKERS = 4
SEED = 42
EARLY_STOP_PATIENCE = 15

# 与 inference 预处理一致：ImageNet 归一化
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
