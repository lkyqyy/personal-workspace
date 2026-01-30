#!/usr/bin/env python3
"""
单图对比调试：对同一张验证集图在 Python 下跑一遍，保存预处理输入、模型输出 prob、最终 mask，
便于与 C++ debug_cpp/ 产物逐项对比，定位差异在预处理、引擎还是后处理。

用法:
  python3 scripts/debug_single_image.py [--image path] [--num 2] [--out debug_py]
  # 再用 C++ 对同一张图: ./bin/PolypSegmentationTRT <同一张图路径>
  # 对比 debug_py/ 与 debug_cpp/ 下 input.bin / prob.bin / mask.png
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
    """Letterbox + 归一化，返回 (1,3,256,256) 与 letterbox 参数。"""
    w, h = img.size
    raw_w, raw_h, new_w, new_h, x0, y0 = _letterbox_params(w, h)
    img_resized = resize_keep_ratio(img, IMG_SIZE, is_mask=False)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = normalize_image(arr, MEAN, STD)
    t = torch.from_numpy(arr).float().unsqueeze(0)
    return t, (raw_w, raw_h, new_w, new_h, x0, y0)


def postprocess(logit: np.ndarray, raw_w: int, raw_h: int, new_w: int, new_h: int, x0: int, y0: int) -> np.ndarray:
    """裁掉 padding、二值化、缩回原图尺寸。"""
    pred = (logit[0, 0] > 0.5).astype(np.uint8) * 255
    crop = pred[y0 : y0 + new_h, x0 : x0 + new_w]
    out = np.array(Image.fromarray(crop).resize((raw_w, raw_h), Image.NEAREST), dtype=np.uint8)
    return out


def save_bin_with_shape(path: Path, data: np.ndarray) -> None:
    """与 C++ / generate_anchor_data 一致：4×int32 shape + NCHW float32。"""
    data = np.ascontiguousarray(data.astype(np.float32))
    shape = np.array(data.shape, dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(shape.tobytes())
        f.write(data.tobytes())


def run_one(img_path: Path, out_dir: Path, model, device: torch.device) -> None:
    """对单张图预处理 → 推理 → 后处理，并保存 input.bin / prob.bin / mask.png / letterbox.txt。"""
    img = Image.open(img_path).convert("RGB")
    x, (raw_w, raw_h, new_w, new_h, x0, y0) = preprocess(img)
    x = x.to(device)
    with torch.no_grad():
        out = model(x)
    prob = out.cpu().numpy()
    mask = postprocess(prob, raw_w, raw_h, new_w, new_h, x0, y0)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_bin_with_shape(out_dir / "input.bin", x.cpu().numpy())
    save_bin_with_shape(out_dir / "prob.bin", prob)
    Image.fromarray(mask).save(out_dir / "mask.png")
    with open(out_dir / "letterbox.txt", "w") as f:
        f.write(f"{raw_w} {raw_h} {new_w} {new_h} {x0} {y0}\n")

    # 归一化后输入的统计，便于快速对比
    inp = x.cpu().numpy()
    with open(out_dir / "input_stats.txt", "w") as f:
        for c in range(3):
            ch = inp[0, c]
            f.write(f"ch{c} min={ch.min():.6f} max={ch.max():.6f} mean={ch.mean():.6f}\n")


def main() -> None:
    p = argparse.ArgumentParser(description="单图调试：保存预处理/模型输出/最终 mask 供与 C++ 对比")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data", help="含 images/ 与 masks/ 的目录")
    p.add_argument("--image", type=Path, default=None, help="指定一张图；不指定则用 --num 从 data/images 取前 N 张")
    p.add_argument("--num", type=int, default=5, help="未指定 --image 时，取验证集前 num 张（默认 5）")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "debug_py", help="输出目录")
    p.add_argument("--pt", type=Path, default=REPO_ROOT / "train" / "snapshots" / "best.pth", help=".pth 权重")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    images_dir = args.data_dir / "images"
    if not images_dir.is_dir():
        print(f"错误: 需要目录 {images_dir}", file=sys.stderr)
        sys.exit(1)
    if args.image is not None:
        paths = [args.image]
        if not paths[0].is_file():
            print(f"错误: 文件不存在 {paths[0]}", file=sys.stderr)
            sys.exit(1)
    else:
        all_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
        paths = all_paths[: args.num]
        if not paths:
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

    for i, img_path in enumerate(paths):
        sub = args.out if len(paths) == 1 else args.out / f"img_{i}"
        run_one(img_path, sub, model, device)
        print(f"[Python] {img_path.name} -> {sub}")
        print(f"  input.bin, prob.bin, mask.png, letterbox.txt, input_stats.txt")

    print("\nC++ 对比：对前 N 张图运行（与上面顺序一致）")
    if len(paths) > 1:
        paths_str = " ".join(str(p) for p in paths)
        print(f"  ./bin/PolypSegmentationTRT {paths_str}")
        print("  产物在 debug_cpp/img_0 .. img_{} 下，与 debug_py/img_* 逐项对比。".format(len(paths) - 1))
    else:
        print(f"  ./bin/PolypSegmentationTRT {paths[0]}")
        print("  产物在 debug_cpp/ 下，可与 debug_py/ 逐项对比。")


if __name__ == "__main__":
    main()
