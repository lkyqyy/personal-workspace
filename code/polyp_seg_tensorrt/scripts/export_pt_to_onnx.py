#!/usr/bin/env python3
"""
将 PyTorch UNet 分割模型 (.pt) 导出为 ONNX，供 TensorRT 等推理引擎使用。

"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 对应 docs/导出脚本_写代码逻辑.md 第 1 步：模型定义（与训练完全一致）
# 与 train/models/unet.py 结构、属性名完全一致，保证加载 .pt 权重时 key 匹配。
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2，与 train/models/unet.py 一致，属性名为 block。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """下采样: MaxPool2d(2) + DoubleConv，与 train 一致用 pool + conv。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.conv(x)


class Up(nn.Module):
    """上采样 + concat skip + DoubleConv，与 train 一致：F.interpolate + cat([skip, x]) + conv。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """U-Net 分割网络，与 train/models/unet.py 结构、通道数完全一致。"""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        b = base_channels
        self.inc = DoubleConv(in_channels, b)
        self.down1 = Down(b, b * 2)
        self.down2 = Down(b * 2, b * 4)
        self.down3 = Down(b * 4, b * 8)
        self.down4 = Down(b * 8, b * 8)
        self.bottleneck = DoubleConv(b * 8, b * 16)
        self.up1 = Up(b * 16 + b * 8, b * 8)
        self.up2 = Up(b * 8 + b * 4, b * 4)
        self.up3 = Up(b * 4 + b * 2, b * 2)
        self.up4 = Up(b * 2 + b, b)
        self.outc = OutConv(b, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_center = self.bottleneck(x5)
        x = self.up1(x_center, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x).float()


class UNetWithSigmoid(nn.Module):
    """UNet 输出后接 Sigmoid，便于 TensorRT 对 Sigmoid 做图内优化；仅用于导出，不改变原 .pt 权重。"""

    def __init__(self, unet: UNet) -> None:
        super().__init__()
        self.unet = unet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.unet(x))


# 对应 docs/导出脚本_写代码逻辑.md 第 2 步：加载 .pt（checkpoint 格式兼容）
def load_checkpoint(
    pt_path: Path,
    device: torch.device,
    n_channels: int = 3,
    n_classes: int = 1,
    base_channels: int = 64,
) -> tuple[nn.Module, dict]:
    """
    加载 .pt 权重到 UNet。支持两种格式：
    - 直接 state_dict: torch.save(model.state_dict(), path)
    - 含 'state_dict' / 'model_state_dict' 的字典

    模型构造与 train/models/unet.py 一致（in_channels, num_classes, base_channels）。

    Returns:
        (model, extra): model 已加载权重；extra 为 checkpoint 中除 state_dict 外的字段（若有）。
    """
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict")
        if state_dict is not None:
            extra = {k: v for k, v in ckpt.items() if k not in ("state_dict", "model_state_dict")}
        else:
            state_dict = ckpt
            extra = {}
    else:
        state_dict = ckpt
        extra = {}

    model = UNet(in_channels=n_channels, num_classes=n_classes, base_channels=base_channels)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, extra


# 对应 docs/导出脚本_写代码逻辑.md 第 4 步：导出 ONNX（固定/动态、常量折叠）
def export_onnx(
    model: nn.Module,
    onnx_path: Path,
    height: int,
    width: int,
    batch_size: int = 1,
    opset_version: int = 14,
    dynamic: bool = False,
    device: torch.device | None = None,
) -> None:
    """
    导出 ONNX。固定输入形状 (batch, 3, H, W)，输出 (batch, 1, H, W)。
    推荐：固定分辨率 + 多 batch（batch_size>1），便于 TensorRT 选 kernel；已开启 do_constant_folding=True。
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # 导出前必须 eval，影响 BatchNorm/Dropout 等

    if dynamic:
        # 动态 batch/高/宽；非必要不加，固定分辨率+多 batch 更利于 TensorRT
        dummy = torch.randn(1, 3, height, width, device=device)
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )
    else:
        # 推荐：固定分辨率 + 多 batch
        dummy = torch.randn(batch_size, 3, height, width, device=device)
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=opset_version,
            do_constant_folding=True,
        )
    print(f"ONNX 已保存: {onnx_path}")


# 对应 docs/导出脚本_写代码逻辑.md 第 3 步：命令行参数
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PyTorch UNet .pt 导出为 ONNX")
    p.add_argument("--pt", type=str, default="/mnt/d/lk/code/polyp_seg_tensorrt/models/unet_polyp_seg.pth", help="输入的 .pt 权重路径")
    p.add_argument("--onnx", type=str, default="models/unet_polyp_seg.onnx", help="输出的 .onnx 路径")
    p.add_argument("--height", type=int, default=256, help="输入高度（与 train/config.py IMG_SIZE 一致）")
    p.add_argument("--width", type=int, default=256, help="输入宽度")
    p.add_argument("--batch", type=int, default=4, help="固定分辨率时的 batch 大小，多 batch 推理；与训练 BATCH_SIZE 无关，按部署需求设")
    p.add_argument("--opset", type=int, default=14, help="ONNX opset 版本，建议 14 以兼容 TensorRT")
    p.add_argument("--bilinear", action="store_true", help="若训练时 UNet 使用 bilinear 上采样则加上")
    p.add_argument("--sigmoid", action="store_true", help="在输出端包一层 Sigmoid 再导出，便于 TensorRT 图内优化")
    p.add_argument("--simplify", action="store_true", help="导出后调用 onnxsim 精简 ONNX（需 pip install onnx-simplifier）")
    p.add_argument("--dynamic", action="store_true", default=True, help="启用动态 batch/高/宽（默认）；推理时 N/H/W 可变，与 export_onnx_to_engine 的 min/opt/max 一致")
    p.add_argument("--static", action="store_true", dest="static_shape", help="固定 batch/高/宽，不使用动态轴；与 --dynamic 互斥")
    p.add_argument("--device", type=str, default="cuda", help="导出时使用的设备 (cuda/cpu)")
    return p.parse_args()


# 对应 docs/导出脚本_写代码逻辑.md 第 6 步：主流程串联与提示
def main() -> None:
    args = parse_args()
    pt_path = Path(args.pt)
    onnx_path = Path(args.onnx)

    if not pt_path.is_file():
        print(f"错误: 未找到权重文件 {pt_path}", file=sys.stderr)
        sys.exit(1)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(
        pt_path, device, n_channels=3, n_classes=1, base_channels=64
    )
    if args.sigmoid:
        model = UNetWithSigmoid(model)
        model.eval()

    # 默认动态；传 --static 时改为固定 batch/高/宽
    dynamic = not getattr(args, "static_shape", False)
    export_onnx(
        model,
        onnx_path,
        height=args.height,
        width=args.width,
        batch_size=args.batch,
        opset_version=args.opset,
        dynamic=dynamic,
        device=device,
    )

    if args.simplify:
        _run_onnx_simplifier(onnx_path)

    # 预处理提醒：C++ 侧 mean/std 与 PyTorch Normalize 一致，通道建议在 C++ 中统一为 RGB
    print("提示: 推理时归一化 mean/std 需与训练时 transforms.Normalize 一致；通道顺序建议在 C++ 中转为 RGB 再输入。")


# 对应 docs/导出脚本_写代码逻辑.md 第 5 步：可选 ONNX 简化（onnxsim）
def _run_onnx_simplifier(onnx_path: Path) -> None:
    """调用 onnxsim 精简 ONNX；校验通过则覆盖原文件，否则写入 xxx_simplified.onnx。"""
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print("未安装 onnx-simplifier，跳过 simplify。可执行: pip install onnx-simplifier", file=sys.stderr)
        return
    model = onnx.load(str(onnx_path))
    model_simp, check = simplify(model)
    out_sim = onnx_path.parent / (onnx_path.stem + "_simplified.onnx")
    if check:
        onnx.save(model_simp, str(onnx_path))
        print(f"ONNX 已简化并覆盖: {onnx_path}")
    else:
        onnx.save(model_simp, str(out_sim))
        print("onnxsim 校验未通过，保留未简化版本。简化结果已写入:", out_sim, file=sys.stderr)


if __name__ == "__main__":
    main()
