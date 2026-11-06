import torch
import torch.nn.functional as F


def teacher_valid_mask(teacher: torch.Tensor, eps=1e-6) -> torch.Tensor:
    t_norm = torch.linalg.norm(teacher.float(), dim=1, keepdim=True)
    mask = torch.isfinite(t_norm) & (t_norm > float(eps))
    return mask.float()


def masked_cosine_distill(student, teacher, mask, eps=1e-6):
    s = torch.nan_to_num(student.float(), nan=0.0, posinf=0.0, neginf=0.0)
    t = torch.nan_to_num(teacher.float(), nan=0.0, posinf=0.0, neginf=0.0)

    s = F.normalize(s, p=2, dim=1, eps=eps)
    t = F.normalize(t, p=2, dim=1, eps=eps)

    cos = (s * t).sum(dim=1, keepdim=True)
    cos = torch.nan_to_num(cos, nan=0.0, posinf=0.0, neginf=0.0)

    return ((1.0 - cos) * mask).sum() / mask.sum().clamp_min(1.0)


def masked_mse(student, teacher, mask):
    s = torch.nan_to_num(student.float(), nan=0.0, posinf=0.0, neginf=0.0)
    t = torch.nan_to_num(teacher.float(), nan=0.0, posinf=0.0, neginf=0.0)

    loss_map = ((s - t) ** 2).mean(dim=1, keepdim=True)
    return (loss_map * mask).sum() / mask.sum().clamp_min(1.0)


def masked_logmag_l1(student, teacher, mask, eps=1e-6):
    s_mag = torch.linalg.norm(student.float(), dim=1, keepdim=True)
    t_mag = torch.linalg.norm(teacher.float(), dim=1, keepdim=True)

    loss_map = torch.abs(torch.log(s_mag + eps) - torch.log(t_mag + eps))
    return (loss_map * mask).sum() / mask.sum().clamp_min(1.0)


def zero_mask_bce(mask_logits, target_mask):
    if mask_logits is None:
        raise RuntimeError("w_zero > 0 but model output has no 'mask_logits'.")

    with torch.no_grad():
        pos = target_mask.sum()
        neg = target_mask.numel() - pos
        pos_weight = (neg / (pos + 1e-6)).clamp(1.0, 100.0)

    return F.binary_cross_entropy_with_logits(
        mask_logits.float(),
        target_mask.float(),
        pos_weight=pos_weight,
    )


@torch.no_grad()
def masked_cosine_similarity(student, teacher, mask, eps=1e-6):
    s = torch.nan_to_num(student.float(), nan=0.0, posinf=0.0, neginf=0.0)
    t = torch.nan_to_num(teacher.float(), nan=0.0, posinf=0.0, neginf=0.0)

    s = F.normalize(s, p=2, dim=1, eps=eps)
    t = F.normalize(t, p=2, dim=1, eps=eps)

    cos = (s * t).sum(dim=1, keepdim=True)
    cos = torch.nan_to_num(cos, nan=0.0, posinf=0.0, neginf=0.0)

    return (cos * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def magnitude_stats(student, teacher, mask):
    s_mag = torch.linalg.norm(student.float(), dim=1, keepdim=True)
    t_mag = torch.linalg.norm(teacher.float(), dim=1, keepdim=True)

    denom = mask.sum().clamp_min(1.0)

    mag_err = (torch.abs(s_mag - t_mag) * mask).sum() / denom
    mag_ratio = ((s_mag / (t_mag + 1e-6)) * mask).sum() / denom

    return mag_err, mag_ratio