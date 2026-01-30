#!/usr/bin/env python3
"""
在与 C++ 相同的验证集（data/images + data/masks）上跑 PyTorch 推理，输出 mean Dice / mean IoU，
便于与 C++ TensorRT 结果直接对比。
需先运行: python3 scripts/prepare_val_for_cpp.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import config as train_config
from train.dataset import normalize_image, resize_keep_ratio

# 复用 export_pt_to_onnx 的模型与加载
from scripts.export_pt_to_onnx import UNet, UNetWithSigmoid, load_checkpoint

IMG_SIZE = 256
MEAN = train_config.IMG_MEAN
STD = train_config.IMG_STD


def _letterbox_params(w: int, h: int) -> tuple[int, int, int, int, int, int]:
    """与 C++ Preprocessor 一致：raw_w, raw_h, new_w, new_h, x0, y0。"""
    scale = IMG_SIZE / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    x0 = (IMG_SIZE - new_w) // 2
    y0 = (IMG_SIZE - new_h) // 2
    return w, h, new_w, new_h, x0, y0


def preprocess(img: Image.Image) -> tuple[torch.Tensor, tuple[int, int, int, int, int, int]]:
    """Letterbox + 归一化，返回 (1,3,256,256) 与 letterbox 参数供后处理裁边。"""
    w, h = img.size
    raw_w, raw_h, new_w, new_h, x0, y0 = _letterbox_params(w, h)
    img_resized = resize_keep_ratio(img, IMG_SIZE, is_mask=False)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = normalize_image(arr, MEAN, STD)
    t = torch.from_numpy(arr).float().unsqueeze(0)
    return t, (raw_w, raw_h, new_w, new_h, x0, y0)


def postprocess(logit: np.ndarray, raw_w: int, raw_h: int, new_w: int, new_h: int, x0: int, y0: int) -> np.ndarray:
    """裁掉 padding、二值化、缩回原图尺寸，与 C++ Postprocessor 一致。"""
    # logit: (1,1,H,W) float [0,1]
    pred = (logit[0, 0] > 0.5).astype(np.uint8) * 255
    crop = pred[y0 : y0 + new_h, x0 : x0 + new_w]
    out = np.array(Image.fromarray(crop).resize((raw_w, raw_h), Image.NEAREST), dtype=np.uint8)
    return out


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """与 C++ Metrics::calculateDice 一致：>127 为前景。"""
    p = (pred > 127).astype(np.uint8)
    g = (gt > 127).astype(np.uint8)
    inter = np.sum(p & g)
    s = np.sum(p) + np.sum(g)
    if s == 0:
        return 1.0
    return float(2 * inter) / s


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """与 C++ Metrics::calculateIOU 一致。"""
    p = (pred > 127).astype(np.uint8)
    g = (gt > 127).astype(np.uint8)
    inter = np.sum(p & g)
    u = np.sum(p) + np.sum(g) - inter
    if u == 0:
        return 1.0
    return float(inter) / u


def main() -> None:
    p = argparse.ArgumentParser(description="在 data/images+masks 上评估 PyTorch，输出 mean Dice/IoU 与 C++ 对比")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data", help="含 images/ 与 masks/ 的目录")
    p.add_argument("--pt", type=Path, default=REPO_ROOT / "train" / "snapshots" / "best.pth", help=".pt 权重路径")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--match-training", action="store_true", help="在 256×256 空间算 Dice，与训练时验证一致，可复现 0.8194")
    args = p.parse_args()

    images_dir = args.data_dir / "images"
    masks_dir = args.data_dir / "masks"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        print(f"错误: 需要 {images_dir} 与 {masks_dir}，请先运行 python3 scripts/prepare_val_for_cpp.py", file=sys.stderr)
        sys.exit(1)

    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not image_paths:
        print("错误: data/images 下无 jpg/png", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)
    unet, _ = load_checkpoint(
        args.pt, device,
        n_channels=train_config.IN_CHANNELS,
        n_classes=train_config.NUM_CLASSES,
        base_channels=64,
    )
    model = UNetWithSigmoid(unet).to(device).eval()

    total_dice = 0.0
    total_iou = 0.0
    count = 0
    match_training = args.match_training
    with torch.no_grad():
        for img_path in image_paths:
            mask_path = masks_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            raw_w, raw_h = img.size[0], img.size[1]

            x, (rw, rh, nw, nh, x0, y0) = preprocess(img)
            x = x.to(device)
            out = model(x)
            prob = out.cpu().numpy()

            if match_training:
                # 与训练一致：在 256×256 空间算 Dice（GT 也 letterbox 到 256×256）
                pred_256 = (prob[0, 0] > 0.5).astype(np.uint8) * 255
                mask_pil = Image.open(mask_path).convert("L")
                gt_256_pil = resize_keep_ratio(mask_pil, IMG_SIZE, is_mask=True)
                gt_256 = np.array(gt_256_pil, dtype=np.uint8)
                total_dice += dice(pred_256, gt_256)
                total_iou += iou(pred_256, gt_256)
            else:
                pred = postprocess(prob, rw, rh, nw, nh, x0, y0)
                gt = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
                total_dice += dice(pred, gt)
                total_iou += iou(pred, gt)
            count += 1

    if count == 0:
        print("未找到有效 (img, mask) 对", file=sys.stderr)
        sys.exit(1)
    mean_dice = total_dice / count
    mean_iou = total_iou / count
    print(f"[Python] 验证集: {count} 张 (与 C++ 相同 data/images+masks)")
    if match_training:
        print(" -> 评估空间: 256×256（与训练时验证一致）")
    print(f" -> Mean Dice: {mean_dice:.4f}")
    print(f" -> Mean IoU:  {mean_iou:.4f}")


if __name__ == "__main__":
    main()
