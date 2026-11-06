import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils.config import cfg
from data.dataset import DinoTSDFDataset
import model.student as model_module

from utils.losses import (
    teacher_valid_mask,
    masked_cosine_distill,
    masked_mse,
    masked_logmag_l1,
    zero_mask_bce,
    masked_cosine_similarity,
    magnitude_stats,
)

from utils.ddp_utils import setup_ddp, cleanup_ddp, is_rank0, reduce_mean_stats
from utils.train_utils import snapshot_config, save_checkpoint


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def make_loader(list_file, train):
    dataset = DinoTSDFDataset(cfg=cfg, train=train, list_file=list_file)

    sampler = DistributedSampler(
        dataset,
        shuffle=train,
        drop_last=False,
    )

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        drop_last=False,
    )

    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(**loader_kwargs)


def build_model(device):
    model_name = cfg.model_name

    if not hasattr(model_module, model_name):
        raise ValueError(f"Model '{model_name}' not found in model.py")

    model_cls = getattr(model_module, model_name)
    config = vars(cfg)

    config.setdefault("model_args", getattr(cfg, "model_args", {}))

    return model_cls(config).to(device)


def forward_model(model, x):
    batch = {"input_tsdf": x} if cfg.model_expects_batch_dict else x
    out = model(batch)

    if isinstance(out, dict):
        pred = out.get("dino_proj", None)
        if pred is None:
            pred = out.get("pred", None)

        if pred is None:
            raise KeyError("Model output dict must contain 'dino_proj' or 'pred'.")

        return pred, out

    return out, {}


def compute_losses(pred, teacher, out):
    mask = teacher_valid_mask(teacher, eps=cfg.teacher_mask_eps)

    cos_loss = masked_cosine_distill(pred, teacher, mask) if cfg.w_cos > 0 else pred.new_tensor(0.0)
    mse_loss = masked_mse(pred, teacher, mask) if cfg.w_mse > 0 else pred.new_tensor(0.0)
    mag_loss = masked_logmag_l1(pred, teacher, mask) if cfg.w_mag > 0 else pred.new_tensor(0.0)

    if cfg.w_zero > 0:
        zero_loss = zero_mask_bce(out.get("mask_logits", None), mask)
    else:
        zero_loss = pred.new_tensor(0.0)

    loss = (
        cfg.w_cos * cos_loss
        + cfg.w_mse * mse_loss
        + cfg.w_mag * mag_loss
        + cfg.w_zero * zero_loss
    )

    with torch.no_grad():
        cos_sim = masked_cosine_similarity(pred, teacher, mask)
        mag_err, mag_ratio = magnitude_stats(pred, teacher, mask)
        occ = mask.mean()

    return loss, {
        "loss": loss.detach(),
        "cos_loss": cos_loss.detach(),
        "mse": mse_loss.detach(),
        "mag_loss": mag_loss.detach(),
        "zero": zero_loss.detach(),
        "cos_sim": cos_sim.detach(),
        "mag_err": mag_err.detach(),
        "mag_ratio": mag_ratio.detach(),
        "occ": occ.detach(),
    }


def run_epoch(model, loader, optimizer, scheduler, scaler, device, train, epoch):
    model.train(train)

    phase = "train" if train else "val"
    rank0 = is_rank0()

    stats = {}
    pbar = tqdm(loader, desc=phase, ncols=120, disable=not rank0)

    for batch in pbar:
        x = batch["input_tsdf"].to(device, non_blocking=True)
        teacher = batch["teacher"].to(device, non_blocking=True)

        if teacher.ndim == 4:
            teacher = teacher.unsqueeze(0)

        batch_size = x.shape[0]

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=cfg.amp):
                pred, out = forward_model(model, x)
                loss, loss_stats = compute_losses(pred, teacher, out)

        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

        for k, v in loss_stats.items():
            value = float(v.item())
            if k not in stats:
                stats[k] = [0.0, 0.0]
            stats[k][0] += value * batch_size
            stats[k][1] += batch_size

        if rank0:
            shown = {k: f"{stats[k][0] / max(stats[k][1], 1):.4f}" for k in stats}
            pbar.set_postfix(shown)

    return reduce_mean_stats(stats, device)


def main():
    device, rank, local_rank, world_size = setup_ddp()
    rank0 = rank == 0

    if rank0:
        print(f"[DDP] world_size={world_size}")

    exp_dir = os.path.join(cfg.save_root, cfg.experiment_name)

    if rank0:
        os.makedirs(exp_dir, exist_ok=True)
        print(f"[Save] {exp_dir}")
        snapshot_config(cfg, model_module, exp_dir)

    train_loader = make_loader(cfg.train_list, train=True)
    val_loader = make_loader(cfg.val_list, train=False)
    val_novel_loader = make_loader(cfg.val_novel_list, train=False)

    model = build_model(device)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
    )

    if rank0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[Model] {cfg.model_name} | params={n_params / 1e6:.2f}M")

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        epochs=cfg.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.10,
        div_factor=25.0,
        final_div_factor=1e4,
        anneal_strategy="cos",
    )

    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    use_wandb = bool(cfg.wandb_enabled) and WANDB_AVAILABLE and rank0

    if cfg.wandb_enabled and not WANDB_AVAILABLE and rank0:
        print("[WARN] wandb_enabled=True but wandb is not installed.")

    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.experiment_name,
            config=vars(cfg),
        )

    best_val = float("inf")
    best_val_novel = float("inf")
    best_val_path = None
    best_val_novel_path = None

    for epoch in range(1, cfg.num_epochs + 1):
        train_loader.sampler.set_epoch(epoch)

        if rank0:
            print(f"\n==== Epoch {epoch}/{cfg.num_epochs} ====")

        t0 = time.time()

        train_stats = run_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, train=True, epoch=epoch
        )

        val_stats = run_epoch(
            model, val_loader, None, None, scaler, device, train=False, epoch=epoch
        )

        val_novel_stats = run_epoch(
            model, val_novel_loader, None, None, scaler, device, train=False, epoch=epoch
        )

        torch.distributed.barrier()

        if rank0:
            dt = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]

            print(f"[LR] {lr_now:.6e} | [Time] {dt:.1f}s")
            print(f"[Train] {train_stats}")
            print(f"[Val] {val_stats}")
            print(f"[Val Novel] {val_novel_stats}")

            if use_wandb:
                log_dict = {
                    "epoch": epoch,
                    "lr": lr_now,
                    "time/epoch_sec": dt,
                }

                for k, v in train_stats.items():
                    log_dict[f"train/{k}"] = v
                for k, v in val_stats.items():
                    log_dict[f"val/{k}"] = v
                for k, v in val_novel_stats.items():
                    log_dict[f"val_novel/{k}"] = v

                wandb.log(log_dict)

            if val_stats["loss"] < best_val:
                best_val = val_stats["loss"]

                if best_val_path and os.path.exists(best_val_path):
                    os.remove(best_val_path)

                best_val_path = os.path.join(
                    exp_dir,
                    f"best_val_epoch_{epoch:04d}_{best_val:.6f}.pt",
                )

                save_checkpoint(
                    best_val_path,
                    model.module,
                    optimizer,
                    scheduler,
                    epoch,
                    cfg,
                    extra={"best_val": best_val},
                )

                print(f"[BEST VAL] saved: {best_val_path}")

            if val_novel_stats["loss"] < best_val_novel:
                best_val_novel = val_novel_stats["loss"]

                if best_val_novel_path and os.path.exists(best_val_novel_path):
                    os.remove(best_val_novel_path)

                best_val_novel_path = os.path.join(
                    exp_dir,
                    f"best_val_novel_epoch_{epoch:04d}_{best_val_novel:.6f}.pt",
                )

                save_checkpoint(
                    best_val_novel_path,
                    model.module,
                    optimizer,
                    scheduler,
                    epoch,
                    cfg,
                    extra={"best_val_novel": best_val_novel},
                )

                print(f"[BEST NOVEL] saved: {best_val_novel_path}")

            if epoch % cfg.checkpoint_interval == 0:
                ckpt_path = os.path.join(exp_dir, f"checkpoint_epoch_{epoch:04d}.pt")

                save_checkpoint(
                    ckpt_path,
                    model.module,
                    optimizer,
                    scheduler,
                    epoch,
                    cfg,
                )

                print(f"[CKPT] saved: {ckpt_path}")

        torch.distributed.barrier()

    if use_wandb:
        wandb.finish()

    cleanup_ddp()

    if rank0:
        print("Done.")


if __name__ == "__main__":
    main()