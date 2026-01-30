"""
Polyp WLI 数据集：与 inference 预处理对齐（等比缩放+padding、ImageNet 归一化）。
"""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter

from . import config


def resize_keep_ratio(img: Image.Image, size: int, is_mask: bool = False) -> Image.Image:
    """等比缩放 + padding 到 size x size。"""
    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError(f"非法图像尺寸: {img.size}")
    scale = size / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    img_resized = img.resize((new_w, new_h), resample)
    if is_mask:
        background = Image.new("L", (size, size), 0)
    else:
        background = Image.new("RGB", (size, size), (0, 0, 0))
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    background.paste(img_resized, (x0, y0))
    return background


def random_flip(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if np.random.rand() > 0.5:
        img, mask = np.fliplr(img), np.fliplr(mask)
    if np.random.rand() > 0.5:
        img, mask = np.flipud(img), np.flipud(mask)
    return img, mask


def random_rotate(img: np.ndarray, mask: np.ndarray, angle_range: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    angle = np.random.uniform(-angle_range, angle_range)
    img_pil = Image.fromarray((img * 255).astype(np.uint8))
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    img_rotated = img_pil.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    mask_rotated = mask_pil.rotate(angle, resample=Image.NEAREST, fillcolor=0)
    img = np.array(img_rotated, dtype=np.float32) / 255.0
    mask = (np.array(mask_rotated, dtype=np.uint8) > 127).astype(np.float32)
    return img, mask


def random_brightness_contrast(img: np.ndarray, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)) -> np.ndarray:
    img = img * np.random.uniform(*brightness_range)
    img = (img - 0.5) * np.random.uniform(*contrast_range) + 0.5
    return np.clip(img, 0.0, 1.0)


def elastic_transform(img: np.ndarray, mask: np.ndarray, alpha: float = 100, sigma: float = 10) -> tuple[np.ndarray, np.ndarray]:
    if np.random.rand() > 0.5:
        return img, mask
    h, w = img.shape[:2]
    dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    if len(img.shape) == 3:
        img_transformed = np.zeros_like(img)
        for i in range(img.shape[2]):
            img_transformed[:, :, i] = map_coordinates(img[:, :, i], indices, order=1, mode='reflect').reshape(h, w)
    else:
        img_transformed = map_coordinates(img, indices, order=1, mode='reflect').reshape(h, w)
    mask_transformed = map_coordinates(mask, indices, order=0, mode='reflect').reshape(h, w)
    mask_transformed = (mask_transformed > 0.5).astype(np.float32)
    return img_transformed, mask_transformed


def apply_augmentation(img: np.ndarray, mask: np.ndarray, split: str) -> tuple[np.ndarray, np.ndarray]:
    if split != "train":
        return img, mask
    img, mask = random_flip(img, mask)
    if np.random.rand() > 0.5:
        img, mask = random_rotate(img, mask, angle_range=15.0)
    if np.random.rand() > 0.5:
        img = random_brightness_contrast(img)
    img, mask = elastic_transform(img, mask)
    return img, mask


def normalize_image(img: np.ndarray, mean: list, std: list) -> np.ndarray:
    mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(std, dtype=np.float32).reshape(3, 1, 1)
    return (img - mean) / std


class PolypWliDataset(Dataset):
    """
    支持两种目录结构：
    1) DATA_ROOT/images/{split}/*.jpg, DATA_ROOT/masks/{split}/*.png
    2) 扁平：DATA_ROOT/images/*.jpg, DATA_ROOT/masks/*.png；img_paths 传入时按比例划分好的路径列表
    """

    def __init__(self, split: str, augment: bool = False, img_paths: list[Path] | None = None):
        assert split in {"train", "val", "test"}
        self.split = split
        self.augment = augment
        root = Path(config.DATA_ROOT)
        if img_paths is not None:
            self.img_paths = img_paths
            self.mask_dir = root / "masks"
        else:
            self.img_dir = root / "images" / split
            self.mask_dir = root / "masks" / split
            if not self.img_dir.exists() or not self.mask_dir.exists():
                raise FileNotFoundError(f"找不到图像或掩码目录: {self.img_dir}, {self.mask_dir}")
            self.img_paths = sorted(self.img_dir.glob("*.jpg"))
        if len(self.img_paths) == 0:
            raise RuntimeError(f"没有找到 jpg 图像")

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        stem = Path(img_path).stem
        mask_path = self.mask_dir / f"{stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"掩码不存在: {mask_path}")
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        img = resize_keep_ratio(img, config.IMG_SIZE, is_mask=False)
        mask = resize_keep_ratio(mask, config.IMG_SIZE, is_mask=True)
        img = np.array(img, dtype=np.float32) / 255.0
        mask = (np.array(mask, dtype=np.uint8) > 127).astype(np.float32)
        if self.augment:
            img, mask = apply_augmentation(img, mask, self.split)
        img = np.transpose(img, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)
        img = normalize_image(img, config.IMG_MEAN, config.IMG_STD)
        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask)
        return torch.from_numpy(img).float(), torch.from_numpy(mask).float()


def create_dataloaders():
    root = Path(config.DATA_ROOT)
    img_all_dir = root / "images"
    has_split_dirs = (root / "images" / "train").exists()
    if has_split_dirs:
        train_ds = PolypWliDataset(split="train", augment=True)
        val_ds = PolypWliDataset(split="val", augment=False)
    else:
        if not img_all_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {img_all_dir}")
        all_paths = sorted(img_all_dir.glob("*.jpg"))
        if len(all_paths) == 0:
            raise RuntimeError(f"{img_all_dir} 下没有 jpg 图像")
        rng = np.random.default_rng(config.SEED)
        idx = rng.permutation(len(all_paths))
        n_train = int(len(all_paths) * 0.8)
        train_paths = [all_paths[i] for i in idx[:n_train]]
        val_paths = [all_paths[i] for i in idx[n_train:]]
        train_ds = PolypWliDataset(split="train", augment=True, img_paths=train_paths)
        val_ds = PolypWliDataset(split="val", augment=False, img_paths=val_paths)
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    return train_loader, val_loader
