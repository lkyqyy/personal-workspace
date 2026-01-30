#!/usr/bin/env python3
"""
对比 debug_py/ 与 debug_cpp/ 下同名产物，输出逐项差异与简要结论。
用于定位 Py vs C++ 差异在预处理、引擎还是后处理。

用法:
  python3 scripts/compare_debug_outputs.py [--py debug_py] [--cpp debug_cpp] [--num 5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_bin_with_shape(path: Path) -> tuple[np.ndarray, tuple[int, ...]]:
    """与 generate_anchor_data / C++ 一致：前 4 个 int32 为 shape，随后 float32。"""
    data = np.fromfile(path, dtype=np.int32, count=4)
    shape = tuple(int(x) for x in data)
    n = int(np.prod(shape))
    arr = np.fromfile(path, dtype=np.float32, offset=16, count=n).reshape(shape)
    return arr, shape


def compare_letterbox(py_path: Path, cpp_path: Path) -> tuple[bool, str]:
    """对比 letterbox.txt，应完全一致。"""
    if not py_path.exists() or not cpp_path.exists():
        return False, "文件缺失"
    py_line = py_path.read_text().strip()
    cpp_line = cpp_path.read_text().strip()
    ok = py_line == cpp_line
    return ok, "一致" if ok else f"Py: {py_line} | C++: {cpp_line}"


def compare_input_stats(py_path: Path, cpp_path: Path) -> tuple[bool, str]:
    """对比 input_stats.txt，允许小幅数值差异（PIL vs OpenCV resize）。"""
    if not py_path.exists() or not cpp_path.exists():
        return False, "文件缺失"
    py_lines = py_path.read_text().strip().splitlines()
    cpp_lines = cpp_path.read_text().strip().splitlines()
    if len(py_lines) != len(cpp_lines):
        return False, f"行数不同 {len(py_lines)} vs {len(cpp_lines)}"
    diffs = []
    for pl, cl in zip(py_lines, cpp_lines):
        if pl != cl:
            diffs.append(f"  {pl}\n  {cl}")
    if not diffs:
        return True, "一致"
    # 粗略判断：若仅小数位略有差异可视为可接受
    return False, "\n".join(diffs[:3]) + ("\n  ..." if len(diffs) > 3 else "")


def compare_bin_mae(py_path: Path, cpp_path: Path) -> tuple[float, float, int, str]:
    """对比 bin 文件，返回 (MAE, max_abs_diff, num_elements, status)。"""
    if not py_path.exists():
        return -1.0, -1.0, 0, "Py 缺失"
    if not cpp_path.exists():
        return -1.0, -1.0, 0, "C++ 缺失"
    try:
        py_arr, _ = load_bin_with_shape(py_path)
        cpp_arr, _ = load_bin_with_shape(cpp_path)
    except Exception as e:
        return -1.0, -1.0, 0, str(e)
    if py_arr.shape != cpp_arr.shape:
        return -1.0, -1.0, int(py_arr.size), f"shape 不同 {py_arr.shape} vs {cpp_arr.shape}"
    diff = np.abs(py_arr.astype(np.float64) - cpp_arr.astype(np.float64))
    mae = float(np.mean(diff))
    max_diff = float(np.max(diff))
    return mae, max_diff, int(py_arr.size), "ok"


def compare_mask_pixels(py_path: Path, cpp_path: Path) -> tuple[float, str]:
    """对比 mask.png：前景判定 >127，计算像素一致率 [0,1]。"""
    if not py_path.exists():
        return -1.0, "Py 缺失"
    if not cpp_path.exists():
        return -1.0, "C++ 缺失"
    py_img = np.array(Image.open(py_path).convert("L"))
    cpp_img = np.array(Image.open(cpp_path).convert("L"))
    if py_img.shape != cpp_img.shape:
        return -1.0, f"shape 不同 {py_img.shape} vs {cpp_img.shape}"
    py_bin = (py_img > 127).astype(np.uint8)
    cpp_bin = (cpp_img > 127).astype(np.uint8)
    same = np.sum(py_bin == cpp_bin)
    total = py_bin.size
    return same / total, "ok"


def run_compare(py_root: Path, cpp_root: Path, num: int) -> None:
    """对 img_0 .. img_{num-1} 逐项对比并打印表格与结论。"""
    print("对比目录: Python {}  vs  C++ {}".format(py_root, cpp_root))
    print("")

    # 支持单图模式：无 img_0 子目录时直接比较 py_root/cpp_root 下文件
    def dir_for(i: int) -> tuple[Path, Path]:
        py_dir = py_root / f"img_{i}" if (py_root / "img_0").exists() else py_root
        cpp_dir = cpp_root / f"img_{i}" if (cpp_root / "img_0").exists() else cpp_root
        return py_dir, cpp_dir

    n = num
    if not (py_root / "img_0").exists() and not (cpp_root / "img_0").exists():
        n = 1

    results = []
    for i in range(n):
        py_dir, cpp_dir = dir_for(i)
        label = f"img_{i}" if n > 1 else "single"
        lb_ok, lb_msg = compare_letterbox(py_dir / "letterbox.txt", cpp_dir / "letterbox.txt")
        st_ok, st_msg = compare_input_stats(py_dir / "input_stats.txt", cpp_dir / "input_stats.txt")
        in_mae, in_max, in_n, in_status = compare_bin_mae(py_dir / "input.bin", cpp_dir / "input.bin")
        pr_mae, pr_max, pr_n, pr_status = compare_bin_mae(py_dir / "prob.bin", cpp_dir / "prob.bin")
        mask_agree, mask_status = compare_mask_pixels(py_dir / "mask.png", cpp_dir / "mask.png")

        results.append({
            "label": label,
            "letterbox": lb_ok,
            "input_mae": in_mae,
            "input_max": in_max,
            "prob_mae": pr_mae,
            "prob_max": pr_max,
            "mask_agree": mask_agree,
            "input_stats_ok": st_ok,
        })
        print("[{}] letterbox: {}  |  input.bin MAE: {:.6f}  max: {:.6f}  |  prob.bin MAE: {:.6f}  max: {:.6f}  |  mask 一致率: {:.2%}".format(
            label,
            "ok" if lb_ok else "diff",
            in_mae if in_mae >= 0 else float("nan"),
            in_max if in_max >= 0 else float("nan"),
            pr_mae if pr_mae >= 0 else float("nan"),
            pr_max if pr_max >= 0 else float("nan"),
            mask_agree if mask_agree >= 0 else float("nan"),
        ))
        if not lb_ok and lb_msg != "文件缺失":
            print("     letterbox: {}".format(lb_msg.replace("\n", " ")))
        if not st_ok and st_msg != "文件缺失":
            print("     input_stats: {}".format(st_msg.split("\n")[0][:80]))

    # 汇总与结论
    print("")
    print("--- 简要结论 ---")
    avg_in_mae = np.nanmean([r["input_mae"] for r in results if r["input_mae"] >= 0])
    avg_pr_mae = np.nanmean([r["prob_mae"] for r in results if r["prob_mae"] >= 0])
    avg_mask = np.nanmean([r["mask_agree"] for r in results if r["mask_agree"] >= 0])

    if not np.isnan(avg_in_mae):
        if avg_in_mae > 0.01:
            print("  预处理：input.bin MAE 较大 ({:.4f})，可能为 PIL vs OpenCV resize 或归一化差异。".format(avg_in_mae))
        else:
            print("  预处理：input.bin MAE 较小 ({:.6f})，前后处理逻辑基本对齐。".format(avg_in_mae))
    if not np.isnan(avg_pr_mae):
        if avg_pr_mae > 0.01:
            print("  引擎：prob.bin MAE 较大 ({:.4f})，TRT 与 PyTorch 数值差异明显。".format(avg_pr_mae))
        else:
            print("  引擎：prob.bin MAE 可接受 ({:.6f})。".format(avg_pr_mae))
    if not np.isnan(avg_mask):
        if avg_mask < 0.99:
            print("  后处理/结果：mask 一致率 {:.2%}，若 input/prob 接近而 mask 差，重点检查裁边与 resize。".format(avg_mask))
        else:
            print("  后处理/结果：mask 一致率 {:.2%}，对齐良好。".format(avg_mask))


def main() -> None:
    p = argparse.ArgumentParser(description="对比 debug_py 与 debug_cpp 产物，分析差异阶段")
    p.add_argument("--py", type=Path, default=REPO_ROOT / "debug_py", help="Python 输出目录")
    p.add_argument("--cpp", type=Path, default=REPO_ROOT / "debug_cpp", help="C++ 输出目录")
    p.add_argument("--num", type=int, default=5, help="对比前 num 张（img_0 .. img_{num-1}）")
    args = p.parse_args()

    if not args.py.is_dir():
        print("错误: Python 目录不存在:", args.py, file=sys.stderr)
        sys.exit(1)
    if not args.cpp.is_dir():
        print("错误: C++ 目录不存在:", args.cpp, file=sys.stderr)
        sys.exit(1)

    run_compare(args.py, args.cpp, args.num)


if __name__ == "__main__":
    main()
