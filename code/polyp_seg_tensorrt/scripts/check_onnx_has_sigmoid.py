#!/usr/bin/env python3
"""
检验导出的 ONNX 是否在输出端包含 Sigmoid：从输出张量反向查找，若最后一层为 Sigmoid 则视为已加。
也可用一次推理看输出范围：若在 [0,1] 则多为概率（含 sigmoid），否则多为 logits。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def check_onnx_graph(onnx_path: Path) -> bool:
    """
    从 ONNX 输出反向查找，若输出由 Sigmoid 产生（或 Identity(Sigmoid(...))）则返回 True。
    依赖: pip install onnx
    """
    try:
        import onnx
    except ImportError:
        print("需要安装 onnx: pip install onnx", file=sys.stderr)
        return False

    model = onnx.load(str(onnx_path))
    graph = model.graph
    # 张量名 -> 产生该张量的 node
    name_to_node: dict[str, object] = {}
    for node in graph.node:
        for out in node.output:
            name_to_node[out] = node
    # 张量名 -> 该张量作为唯一输入时的 node（用于反向）
    name_to_consumer: dict[str, list] = {}
    for node in graph.node:
        for inp in node.input:
            name_to_consumer.setdefault(inp, []).append(node)

    def producer_op_type(tensor_name: str) -> str | None:
        if tensor_name not in name_to_node:
            return None
        node = name_to_node[tensor_name]
        return getattr(node, "op_type", None)

    has_sigmoid = False
    for out in graph.output:
        name = out.name
        op = producer_op_type(name)
        if op == "Sigmoid":
            has_sigmoid = True
            break
        if op == "Identity" and name in name_to_node:
            node = name_to_node[name]
            if len(node.input) >= 1:
                op_in = producer_op_type(node.input[0])
                if op_in == "Sigmoid":
                    has_sigmoid = True
                    break
    return has_sigmoid


def check_onnx_inference_range(onnx_path: Path) -> tuple[float, float]:
    """
    用 ONNX Runtime 跑一次推理（随机输入），返回输出 min/max。
    若在 [0,1] 则多为概率（含 sigmoid），否则多为 logits）。
    动态维度用 1,3,256,256 作为输入 shape。
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("需要安装 onnxruntime: pip install onnxruntime", file=sys.stderr)
        return float("nan"), float("nan")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = list(inp.shape)
    for i, d in enumerate(shape):
        if isinstance(d, str) or d is None or d <= 0:
            # 常见 NCHW：N=1, C=3, H=W=256
            shape[i] = [1, 3, 256, 256][i] if i < 4 else 1
    x = np.random.randn(*shape).astype(np.float32) * 0.5
    try:
        out = sess.run(None, {inp.name: x})[0]
        return float(np.min(out)), float(np.max(out))
    except Exception as e:
        print(f"  推理失败（可忽略）: {e}", file=sys.stderr)
        return float("nan"), float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description="检验 ONNX 输出端是否含 Sigmoid")
    p.add_argument("--onnx", type=Path, default=REPO_ROOT / "models" / "unet_polyp_seg.onnx", help="ONNX 路径")
    p.add_argument("--infer", action="store_true", help="额外做一次推理，打印输出 min/max 以辅助判断")
    args = p.parse_args()

    if not args.onnx.is_file():
        print(f"文件不存在: {args.onnx}", file=sys.stderr)
        sys.exit(1)

    has_sigmoid = check_onnx_graph(args.onnx)
    print(f"ONNX: {args.onnx}")
    print(f"输出端含 Sigmoid（图结构）: {'是' if has_sigmoid else '否'}")

    if args.infer:
        lo, hi = check_onnx_inference_range(args.onnx)
        if not np.isnan(lo):
            print(f"一次推理输出范围: min={lo:.4f}, max={hi:.4f}")
            if 0 <= lo and hi <= 1:
                print("  -> 输出在 [0,1]，可视为概率（与含 Sigmoid 一致）")
            else:
                print("  -> 输出不在 [0,1]，多为 logits（未含 Sigmoid）")

    sys.exit(0 if has_sigmoid else 1)


if __name__ == "__main__":
    main()
