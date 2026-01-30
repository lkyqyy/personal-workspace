#!/usr/bin/env python3
"""
将验证集准备到 data/images 与 data/masks，供 C++ 精度校验使用。
与 generate_anchor_data 使用同一套验证集划分，避免数据泄露。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import config as train_config
from scripts.generate_anchor_data import get_val_image_paths, get_mask_path


def main() -> None:
    p = argparse.ArgumentParser(description="将验证集准备到 data/images 与 data/masks")
    p.add_argument("--data-root", type=Path, default=REPO_ROOT / "train" / "data", help="训练数据根目录")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data", help="输出目录，下建 images/ 与 masks/")
    p.add_argument("--symlink", action="store_true", help="使用符号链接而非拷贝，节省磁盘")
    args = p.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    if not data_root.is_dir():
        print(f"错误: 数据根目录不存在 {data_root}", file=sys.stderr)
        sys.exit(1)

    val_paths = get_val_image_paths(data_root, train_config.SEED)
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for img_path in val_paths:
        mask_path = get_mask_path(img_path, data_root)
        if not mask_path.exists():
            print(f"跳过（无 GT）: {img_path.name}", file=sys.stderr)
            continue
        dst_img = images_dir / img_path.name
        dst_mask = masks_dir / (img_path.stem + ".png")
        if args.symlink:
            if not dst_img.exists():
                dst_img.symlink_to(img_path.resolve())
            if not dst_mask.exists():
                dst_mask.symlink_to(mask_path.resolve())
        else:
            shutil.copy2(img_path, dst_img)
            shutil.copy2(mask_path, dst_mask)

    print(f"已准备 {len(val_paths)} 张验证集到 {out_dir}/images 与 {out_dir}/masks")
    print("C++ 精度校验：在项目根运行 ./bin/PolypSegmentationTRT，程序会遍历 data/images 下图片并输出 mean Dice / mean IoU。")


if __name__ == "__main__":
    main()
