#!/usr/bin/env python3
"""
在完整验证集（data/images + data/masks）上分别跑 Python 与 C++，对比 Mean Dice / Mean IoU 差异。
需在项目根目录执行；C++ 会先跑 Anchor 校验再跑验证集。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_metrics(text: str) -> tuple[float | None, float | None]:
    """从输出中解析 Mean Dice 与 Mean IoU。"""
    dice = None
    iou = None
    for line in text.splitlines():
        m = re.search(r"Mean Dice:\s*([\d.]+)", line, re.IGNORECASE)
        if m:
            dice = float(m.group(1))
        m = re.search(r"Mean IoU:\s*([\d.]+)", line, re.IGNORECASE)
        if m:
            iou = float(m.group(1))
    return dice, iou


def main() -> None:
    p = argparse.ArgumentParser(description="验证集上对比 Python 与 C++ 的 Mean Dice / Mean IoU")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data", help="含 images/ 与 masks/ 的目录")
    p.add_argument("--pt", type=Path, default=REPO_ROOT / "train" / "snapshots" / "best.pth", help="Python 使用的 .pth")
    p.add_argument("--bin", type=Path, default=REPO_ROOT / "bin" / "PolypSegmentationTRT", help="C++ 可执行文件")
    args = p.parse_args()

    if not args.bin.is_file():
        print(f"错误: C++ 可执行文件不存在: {args.bin}", file=sys.stderr)
        print("  请先编译: cd build && make", file=sys.stderr)
        sys.exit(1)

    images_dir = args.data_dir / "images"
    masks_dir = args.data_dir / "masks"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        print(f"错误: 需要 {images_dir} 与 {masks_dir}，请先运行 prepare_val_for_cpp.py", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("验证集对比：Python (PyTorch) vs C++ (TensorRT)")
    print("=" * 60)

    # Python
    print("\n[1/2] 运行 Python 验证...")
    cmd_py = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_val_set.py"),
        "--data-dir", str(args.data_dir),
        "--pt", str(args.pt),
    ]
    try:
        out_py = subprocess.run(
            cmd_py,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        py_stdout = out_py.stdout or ""
        if out_py.returncode != 0:
            print(out_py.stderr or py_stdout, file=sys.stderr)
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print("Python 运行超时", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Python 运行失败: {e}", file=sys.stderr)
        sys.exit(1)
    py_dice, py_iou = parse_metrics(py_stdout)
    print(py_stdout.strip())

    # C++
    print("\n[2/2] 运行 C++ 验证（含 Anchor 校验）...")
    try:
        out_cpp = subprocess.run(
            [str(args.bin)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        cpp_stdout = out_cpp.stdout or ""
        if out_cpp.returncode != 0:
            print(out_cpp.stderr or cpp_stdout, file=sys.stderr)
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print("C++ 运行超时", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"C++ 运行失败: {e}", file=sys.stderr)
        sys.exit(1)
    # 只打印 Step 2 相关行，避免刷屏
    for line in cpp_stdout.splitlines():
        if "Step 2" in line or "Validation set" in line or "Mean Dice" in line or "Mean IoU" in line:
            print(line)
    cpp_dice, cpp_iou = parse_metrics(cpp_stdout)

    # 对比
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"                Python (PyTorch)   C++ (TensorRT)   差值")
    print("-" * 60)
    if py_dice is not None and cpp_dice is not None:
        print(f"  Mean Dice      {py_dice:.4f}             {cpp_dice:.4f}           {cpp_dice - py_dice:+.4f}")
    else:
        print(f"  Mean Dice      {py_dice}             {cpp_dice}           (解析失败)")
    if py_iou is not None and cpp_iou is not None:
        print(f"  Mean IoU       {py_iou:.4f}             {cpp_iou:.4f}           {cpp_iou - py_iou:+.4f}")
    else:
        print(f"  Mean IoU       {py_iou}             {cpp_iou}           (解析失败)")
    print("-" * 60)
    if py_dice is not None and cpp_dice is not None:
        diff_dice = abs(cpp_dice - py_dice)
        if diff_dice <= 0.02:
            print("  结论: 差异 ≤ 0.02，部署对齐良好。")
        elif diff_dice <= 0.05:
            print("  结论: 差异在 0.02～0.05，可接受；若需进一步对齐可检查前后处理。")
        else:
            print("  结论: 差异 > 0.05，建议检查前后处理或引擎（如 sigmoid 是否一致）。")


if __name__ == "__main__":
    main()
