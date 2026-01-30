#!/usr/bin/env python3
"""
将 ONNX 模型转换为 TensorRT engine，供本地推理使用。

动态 ONNX（导出时用了 --dynamic）：必须提供 min/opt/max shape，与 trtexec 的
--minShapes/--optShapes/--maxShapes 一致，默认 1,256,256 / 4,256,256 / 8,256,256（N,H,W）。
静态 ONNX：可将 --min_shapes/--opt_shapes/--max_shapes 设为同一值。

依赖: pip install tensorrt（或随 CUDA 安装的 TensorRT Python 包）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_shape(s: str) -> tuple[int, int, int]:
    """解析 'N,H,W' 为 (N, H, W)，与 trtexec --minShapes/--optShapes/--maxShapes 一致。"""
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError("shape 需为 N,H,W 格式，例如 4,256,256")
    return (parts[0], parts[1], parts[2])


def parse_args() -> argparse.Namespace:
    """解析命令行参数。动态 ONNX 必须提供 min/opt/max，与 trtexec 一致。"""
    p = argparse.ArgumentParser(description="ONNX 转 TensorRT engine")
    p.add_argument("--onnx", type=str, default="models/unet_polyp_seg.onnx", help="输入的 .onnx 路径")
    p.add_argument("--engine", type=str, default=None, help="输出的 .engine 路径；不传则根据 --fp16/--fp32 自动命名")
    p.add_argument("--fp16", action="store_true", help="启用 FP16，加速且多数 GPU 支持")
    p.add_argument("--fp32", action="store_true", help="显式使用 FP32（默认即 FP32；与 --fp16 二选一）")
    p.add_argument("--workspace", type=int, default=1, help="builder 工作空间大小（GB）")
    p.add_argument("--verbose", action="store_true", help="打印 TensorRT 详细日志")
    # 动态 shape：与 trtexec --minShapes/--optShapes/--maxShapes 对应，格式 N,H,W（通道 3 固定）
    p.add_argument(
        "--min_shapes",
        type=str,
        default="1,256,256",
        help="动态输入最小 shape：N,H,W（与导出时 --dynamic 对应）",
    )
    p.add_argument(
        "--opt_shapes",
        type=str,
        default="4,256,256",
        help="动态输入最优 shape：N,H,W，TensorRT 按此优化",
    )
    p.add_argument(
        "--max_shapes",
        type=str,
        default="8,256,256",
        help="动态输入最大 shape：N,H,W",
    )
    return p.parse_args()


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool = False,
    workspace_gb: int = 1,
    verbose: bool = False,
    min_shapes: tuple[int, int, int] | None = None,
    opt_shapes: tuple[int, int, int] | None = None,
    max_shapes: tuple[int, int, int] | None = None,
) -> None:
    """
    从 ONNX 构建 TensorRT engine 并序列化到文件。

    动态 ONNX 必须提供 min/opt/max shapes，与 trtexec 的 --minShapes/--optShapes/--maxShapes 一致，
    格式为 (N, H, W)，脚本内补通道 3 得到 (N, 3, H, W)。

    Args:
        onnx_path: ONNX 模型路径
        engine_path: 输出 engine 路径
        fp16: 是否启用 FP16
        workspace_gb: builder 工作空间（GB）
        verbose: 是否输出详细日志
        min_shapes: 动态输入最小 (N, H, W)，None 表示静态/不设 profile
        opt_shapes: 动态输入最优 (N, H, W)
        max_shapes: 动态输入最大 (N, H, W)
    """
    import tensorrt as trt

    log_level = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
    # explicit batch，与 ONNX (N,C,H,W) 一致
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i), file=sys.stderr)
            raise RuntimeError("ONNX 解析失败")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30)
    )
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("已启用 FP16")
    elif fp16:
        print("当前平台未检测到 FP16 加速，使用 FP32", file=sys.stderr)
    else:
        print("使用 FP32")

    # 动态 ONNX：添加 optimization profile，与 trtexec --minShapes/--optShapes/--maxShapes 一致
    if min_shapes is not None and opt_shapes is not None and max_shapes is not None:
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        # 输入 shape (N, 3, H, W)
        n_min, h_min, w_min = min_shapes
        n_opt, h_opt, w_opt = opt_shapes
        n_max, h_max, w_max = max_shapes
        profile.set_shape(
            input_name,
            (n_min, 3, h_min, w_min),
            (n_opt, 3, h_opt, w_opt),
            (n_max, 3, h_max, w_max),
        )
        config.add_optimization_profile(profile)
        print(f"动态 shape profile: min={min_shapes} opt={opt_shapes} max={max_shapes} (N,H,W)")

    # TRT 8.5+ 推荐 build_serialized_network；旧版用 build_engine + serialize
    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("构建 engine 失败")
        serialized = engine.serialize()

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"Engine 已保存: {engine_path}")


def main() -> None:
    args = parse_args()
    onnx_path = Path(args.onnx)

    if not onnx_path.is_file():
        print(f"错误: 未找到 ONNX 文件 {onnx_path}", file=sys.stderr)
        sys.exit(1)

    # 精度：默认 FP32；--fp16 与 --fp32 二选一
    if args.fp16 and args.fp32:
        print("错误: --fp16 与 --fp32 不可同时指定", file=sys.stderr)
        sys.exit(1)
    use_fp16 = args.fp16

    if args.engine is not None:
        engine_path = Path(args.engine)
    else:
        stem = onnx_path.stem
        suffix = "_fp16" if use_fp16 else "_fp32"
        engine_path = onnx_path.parent / f"{stem}{suffix}.engine"

    min_shapes = _parse_shape(args.min_shapes)
    opt_shapes = _parse_shape(args.opt_shapes)
    max_shapes = _parse_shape(args.max_shapes)

    build_engine(
        onnx_path,
        engine_path,
        fp16=use_fp16,
        workspace_gb=args.workspace,
        verbose=args.verbose,
        min_shapes=min_shapes,
        opt_shapes=opt_shapes,
        max_shapes=max_shapes,
    )


if __name__ == "__main__":
    main()
