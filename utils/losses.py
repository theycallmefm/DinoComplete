import torch
import torch.nn.functional as F
import torch.nn as nn

def calc_metrics(x_gt, x_pred, loss_weights, metrics):
    pred_tsdf = x_pred
    target_tsdf = x_gt

    criterion_l1 = nn.SmoothL1Loss(reduction='none')
    loss_l1_ori = criterion_l1(pred_tsdf, target_tsdf)

    target_mask = (target_tsdf <= 1e-10).float()
    pred_mask = (pred_tsdf <= 1e-10).float()
    false_pos_mask = pred_mask * (1 - target_mask)
    false_neg_mask = (1 - pred_mask) * target_mask
    correct_mask = 1.0 - false_pos_mask - false_neg_mask

    loss_l1 = torch.mean(
        false_neg_mask * loss_l1_ori * 5 +
        false_pos_mask * loss_l1_ori * 3 +
        correct_mask * loss_l1_ori
    )
    metrics['l1_loss'] += loss_l1.detach().cpu().item() * target_tsdf.size(0)

    loss = 0.0

    if 'l1' in loss_weights:
        loss = loss + loss_weights['l1'] * loss_l1

    # IoU
    iou_sum = 0
    for idx, p in enumerate(pred_tsdf):
        pred_bin = (p[0] <= 0).float()
        gt_bin = (target_tsdf[idx][0] <= 0).float()
        intersection = torch.sum((pred_bin + gt_bin) == 2)
        union = torch.sum((pred_bin + gt_bin) >= 1)
        iou = (intersection / (union + 1e-8)).item()
        iou_sum += iou
    metrics['iou'] += iou_sum

    metrics['total_loss'] += loss.detach().cpu().item()
    return loss