"""
训练入口：在项目根下运行。例如: python train/train.py 或 PYTHONPATH=. python train/train.py
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import torch
import random
import numpy as np
from tqdm import tqdm

from train import config
from train.models.unet import UNet
from train.dataset import create_dataloaders
from train.utils import losses, metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = total_dice = total_iou = 0.0
    num_batches = 0
    for img, mask in tqdm(train_loader, desc="Train"):
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                pred = model(img)
                loss = criterion(pred, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(img)
            loss = criterion(pred, mask)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        total_dice += metrics.dice_score(pred, mask).item()
        total_iou += metrics.iou_score(pred, mask).item()
        num_batches += 1
    n = max(num_batches, 1)
    return {'loss': total_loss / n, 'dice': total_dice / n, 'iou': total_iou / n}


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = total_dice = total_iou = total_hd95 = 0.0
    num_batches = 0
    with torch.no_grad():
        for img, mask in tqdm(val_loader, desc="Val"):
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            loss = criterion(pred, mask)
            total_loss += loss.item()
            total_dice += metrics.dice_score(pred, mask).item()
            total_iou += metrics.iou_score(pred, mask).item()
            total_hd95 += metrics.hd95(pred, mask).item()
            num_batches += 1
    n = max(num_batches, 1)
    return {'loss': total_loss / n, 'dice': total_dice / n, 'iou': total_iou / n, 'hd95': total_hd95 / n}


def save_checkpoint(model, optimizer, epoch, metrics_dict, save_dir, scaler=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics_dict,
    }
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    latest_path = save_dir / 'latest.pth'
    torch.save(checkpoint, latest_path)
    if 'dice' in metrics_dict:
        best_path = save_dir / f"{config.MODEL_NAME}.pth"
        if not best_path.exists():
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型: Dice = {metrics_dict['dice']:.4f}")
        else:
            best_ckpt = torch.load(best_path, weights_only=False)
            if metrics_dict['dice'] > best_ckpt['metrics']['dice']:
                torch.save(checkpoint, best_path)
                print(f"更新最佳模型: Dice {best_ckpt['metrics']['dice']:.4f} -> {metrics_dict['dice']:.4f}")


def load_checkpoint(checkpoint_path, model, optimizer, device, scaler=None):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None, None
    print(f"加载检查点: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scaler is not None and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    epoch = ckpt['epoch']
    metrics_dict = ckpt.get('metrics', {})
    print(f"已恢复: Epoch {epoch}, Dice: {metrics_dict.get('dice', 0):.4f}")
    return epoch, metrics_dict


class EarlyStopping:
    def __init__(self, patience=10, mode='max', min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (score > self.best_score + self.min_delta) if self.mode == 'max' else (score < self.best_score - self.min_delta):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def main():
    set_seed(config.SEED)
    model = UNet(in_channels=config.IN_CHANNELS, num_classes=config.NUM_CLASSES)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    criterion = losses.bce_dice_loss
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    early_stopping = EarlyStopping(patience=config.EARLY_STOP_PATIENCE, mode='max')

    train_loader, val_loader = create_dataloaders()
    latest_checkpoint = Path(config.SAVE_DIR) / 'latest.pth'
    start_epoch = 1
    if latest_checkpoint.exists():
        loaded_epoch, loaded_metrics = load_checkpoint(latest_checkpoint, model, optimizer, device, scaler)
        if loaded_epoch is not None:
            start_epoch = loaded_epoch + 1
            if loaded_metrics and 'dice' in loaded_metrics:
                early_stopping.best_score = loaded_metrics['dice']

    print(f"开始训练，共 {config.NUM_EPOCHS} 个 epoch，设备: {device}")
    print(f"训练/验证样本: {len(train_loader.dataset)} / {len(val_loader.dataset)}")
    if start_epoch > 1:
        print(f"从 Epoch {start_epoch} 继续")

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS} " + "-" * 40)
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}")
        val_metrics = validate(model, val_loader, criterion, device)
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}, HD95: {val_metrics['hd95']:.2f}")
        save_checkpoint(model, optimizer, epoch, val_metrics, config.SAVE_DIR, scaler)
        if early_stopping(val_metrics['dice']):
            print(f"\n早停触发，最佳 Dice: {early_stopping.best_score:.4f}")
            break
    print("\n训练完成！")


if __name__ == "__main__":
    main()
