import torch
import torch.nn.functional as F


def dice_loss(pred, target, eps=1e-6):
    """
    Dice Loss 计算
    pred: logits [B, 1, H, W], target: mask [B, 1, H, W] 0/1
    """
    pred_prob = torch.sigmoid(pred)
    pred_flat = pred_prob.view(pred_prob.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def bce_dice_loss(pred, target, bce_weight=0.5, eps=1e-6):
    """BCE + Dice 组合 Loss"""
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target, eps)
    return bce_weight * bce + (1 - bce_weight) * dice
