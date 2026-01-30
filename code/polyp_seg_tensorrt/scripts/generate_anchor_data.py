#!/usr/bin/env python3
"""
生成 Binary Validation 锚点数据：在 Python 下导出 input_tensor.bin 与 output_prob.bin，
供 C++ TensorRT 推理后逐元素对比，用于隔离「引擎正确性」与「前后处理正确性」。

参见 docs/导出ONNX之后_C++_TensorRT推理准备.md 第二节「数据层」。
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# 项目根目录，保证可导入 train 与 scripts
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import config as train_config
from train.dataset import normalize_image, resize_keep_ratio


def _load_export_module():
    """动态加载 scripts/export_pt_to_onnx.py，复用 load_checkpoint、UNet、UNetWithSigmoid。"""
    script_path = REPO_ROOT / "scripts" / "export_pt_to_onnx.py"
    spec = importlib.util.spec_from_file_location("export_pt_to_onnx", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def preprocess_image(img_path: Path, size: int, mean: list[float], std: list[float]) -> np.ndarray:
    """
    与 train.dataset.PolypWliDataset 推理路径一致：等比缩放+padding → [0,1] → CHW → 归一化。
    返回 float32，shape (1, 3, size, size)，便于后续 stack。
    """
    img = Image.open(img_path).convert("RGB")
    img = resize_keep_ratio(img, size, is_mask=False)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = normalize_image(img, mean, std)
    img = np.ascontiguousarray(img)
    return img[np.newaxis, ...]


def get_val_image_paths(data_root: Path, split_seed: int) -> list[Path]:
    """
    获取验证集图像路径，与 train.dataset.create_dataloaders 的划分逻辑一致，避免数据泄露。
    - 若存在 images/val 目录则直接使用该目录下 *.jpg；
    - 若为扁平结构（仅 images/*.jpg），则用 split_seed 做 80/20 划分，返回后 20% 作为 val。
    """
    data_root = Path(data_root)
    val_dir = data_root / "images" / "val"
    if val_dir.exists():
        paths = sorted(val_dir.glob("*.jpg"))
    else:
        img_dir = data_root / "images"
        if not img_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {img_dir}")
        all_paths = sorted(img_dir.glob("*.jpg"))
        if len(all_paths) == 0:
            raise FileNotFoundError(f"未找到 jpg 图像: {img_dir}")
        rng = np.random.default_rng(split_seed)
        idx = rng.permutation(len(all_paths))
        n_train = int(len(all_paths) * 0.8)
        paths = [all_paths[i] for i in idx[n_train:]]
    if len(paths) == 0:
        raise RuntimeError("验证集为空，无法生成锚点数据")
    return paths


def get_mask_path(img_path: Path, data_root: Path) -> Path:
    """
    根据图像路径得到对应 GT mask 路径，与 train.dataset 约定一致。
    - 有 images/val 时：masks/val/<stem>.png
    - 扁平结构：masks/<stem>.png
    """
    data_root = Path(data_root)
    try:
        rel = img_path.resolve().relative_to(data_root.resolve())
    except ValueError:
        return data_root / "masks" / (img_path.stem + ".png")
    parts = rel.parts
    if len(parts) >= 3 and parts[0] == "images" and parts[1] == "val":
        return data_root / "masks" / "val" / (img_path.stem + ".png")
    return data_root / "masks" / (img_path.stem + ".png")


def pick_paths_from_list(path_list: list[Path], num: int, seed: int | None) -> list[Path]:
    """从路径列表中选取 num 条；seed 固定时可复现。"""
    if seed is not None:
        rng = random.Random(seed)
        path_list = list(path_list)
        rng.shuffle(path_list)
    return path_list[:num]


def collect_image_paths(image_dir: Path, num: int, seed: int | None) -> list[Path]:
    """从 image_dir 下收集 num 张 jpg 路径；若不足则全部使用。seed 固定时可复现。"""
    paths = sorted(image_dir.glob("*.jpg"))
    if len(paths) == 0:
        raise FileNotFoundError(f"未找到 jpg 图像: {image_dir}")
    return pick_paths_from_list(paths, num, seed)


def save_bin_with_shape(path: Path, data: np.ndarray) -> None:
    """
    保存为 C++ 可读格式：前 4 个 int32 为 shape (N,C,H,W)，随后为 float32 数据（C 序）。
    """
    data = np.ascontiguousarray(data.astype(np.float32))
    shape = np.array(data.shape, dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(shape.tobytes())
        f.write(data.tobytes())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="生成 Binary Validation 锚点数据：input_tensor.bin / output_prob.bin"
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "train" / "data",
        help="数据根目录（与 train/config DATA_ROOT 一致）；仅当未指定 --image-dir 时用于取验证集",
    )
    p.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="可选：直接指定图像目录，从该目录取图（不区分 train/val，慎用以防泄露）；不传则默认仅从验证集取",
    )
    p.add_argument(
        "--num",
        type=int,
        default=1,
        help="选取图像数量；锚点校验时 C++ predictRaw 仅支持 batch=1，默认 1；多张仅用于离线对比时可加大",
    )
    p.add_argument(
        "--pt",
        type=Path,
        default=REPO_ROOT / "train" / "snapshots" / "best.pth",
        help=".pt 权重路径（与导出 ONNX 所用一致）",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "models" / "anchor",
        help="输出目录，将写入 input_tensor.bin、output_prob.bin、anchor_meta.txt、anchor_images.txt",
    )
    p.add_argument(
        "--sigmoid",
        action="store_true",
        help="输出端包 Sigmoid，与导出 ONNX 时 --sigmoid 一致，输出为概率；不加则输出 logits",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机选取图像时的种子，便于复现（默认 42）",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="推理设备 (cuda/cpu)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pt_path = args.pt
    out_dir = args.out_dir

    if not pt_path.is_file():
        print(f"错误: 权重文件不存在 {pt_path}", file=sys.stderr)
        sys.exit(1)

    size = train_config.IMG_SIZE
    mean = train_config.IMG_MEAN
    std = train_config.IMG_STD
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 选取图像路径：默认仅从验证集取，避免数据泄露
    if args.image_dir is not None:
        if not args.image_dir.is_dir():
            print(f"错误: 图像目录不存在 {args.image_dir}", file=sys.stderr)
            sys.exit(1)
        image_paths = collect_image_paths(args.image_dir, args.num, args.seed)
        print("注意: 已指定 --image-dir，未做 train/val 区分，请确认无数据泄露风险。", file=sys.stderr)
    else:
        if not args.data_root.is_dir():
            print(f"错误: 数据根目录不存在 {args.data_root}", file=sys.stderr)
            sys.exit(1)
        val_paths = get_val_image_paths(args.data_root, train_config.SEED)
        image_paths = pick_paths_from_list(val_paths, args.num, args.seed)
        print(f"从验证集选取 {len(image_paths)} 张图像（共 {len(val_paths)} 张 val，与训练划分一致）")

    # 预处理并堆叠为 (N, 3, H, W)
    tensors = [preprocess_image(p, size, mean, std) for p in image_paths]
    input_np = np.concatenate(tensors, axis=0)
    input_tensor = torch.from_numpy(input_np).to(device)

    # 加载模型并推理
    export_mod = _load_export_module()
    model, _ = export_mod.load_checkpoint(
        pt_path, device, n_channels=3, n_classes=1, base_channels=64
    )
    if args.sigmoid:
        model = export_mod.UNetWithSigmoid(model)
        model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    output_np = output.cpu().numpy().astype(np.float32)

    # 保存二进制与元信息
    out_dir.mkdir(parents=True, exist_ok=True)
    save_bin_with_shape(out_dir / "input_tensor.bin", input_np)
    save_bin_with_shape(out_dir / "output_prob.bin", output_np)

    meta_path = out_dir / "anchor_meta.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"N={input_np.shape[0]}\n")
        f.write(f"C=3\n")
        f.write(f"H={size}\n")
        f.write(f"W={size}\n")
        f.write(f"mean={mean[0]:.6f},{mean[1]:.6f},{mean[2]:.6f}\n")
        f.write(f"std={std[0]:.6f},{std[1]:.6f},{std[2]:.6f}\n")
    list_path = out_dir / "anchor_images.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in image_paths:
            f.write(f"{p.name}\n")

    print(f"已写入: {out_dir / 'input_tensor.bin'}, {out_dir / 'output_prob.bin'}")
    print(f"元信息: {meta_path}, 图像列表: {list_path}")
    print("C++ 侧：先读 4 个 int32 得到 shape，再读 N*C*H*W 个 float32 与 output 对比。")


if __name__ == "__main__":
    main()
