import torch
import numpy as np
from scipy.ndimage import distance_transform_edt


def dice_score(pred, target, threshold: float = 0.5, eps: float = 1e-6):
    prob = torch.sigmoid(pred)
    pred_bin = (prob > threshold).float()
    pred_flat = pred_bin.view(pred_bin.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


def iou_score(pred, target, threshold: float = 0.5, eps: float = 1e-6):
    prob = torch.sigmoid(pred)
    pred_bin = (prob > threshold).float()
    pred_flat = pred_bin.view(pred_bin.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def hd95(pred, target, threshold: float = 0.5):
    prob = torch.sigmoid(pred)
    pred_bin = (prob > threshold).float()
    pred_np = pred_bin.squeeze(1).cpu().numpy().astype(np.uint8)
    target_np = target.squeeze(1).cpu().numpy().astype(np.uint8)
    batch_size = pred_np.shape[0]
    hd95_values = []
    for i in range(batch_size):
        pred_mask = pred_np[i]
        target_mask = target_np[i]
        if pred_mask.sum() == 0 and target_mask.sum() == 0:
            hd95_values.append(0.0)
            continue
        if pred_mask.sum() == 0 or target_mask.sum() == 0:
            h, w = pred_mask.shape
            hd95_values.append(np.sqrt(h**2 + w**2))
            continue
        dist_target = distance_transform_edt(1 - target_mask)
        pred_boundary_dist = dist_target[pred_mask > 0]
        dist_pred = distance_transform_edt(1 - pred_mask)
        target_boundary_dist = dist_pred[target_mask > 0]
        hd95_pred = np.percentile(pred_boundary_dist, 95) if len(pred_boundary_dist) > 0 else 0.0
        hd95_target = np.percentile(target_boundary_dist, 95) if len(target_boundary_dist) > 0 else 0.0
        hd95_values.append(max(hd95_pred, hd95_target))
    return torch.tensor(np.mean(hd95_values), dtype=torch.float32)
