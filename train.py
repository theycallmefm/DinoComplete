import torch
import time
import os
import torch.nn.functional as F
from collections import defaultdict
from utils.losses import calc_metrics
from tqdm import tqdm
import wandb
import torch.distributed as dist


def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def is_main_process():
    return get_rank() == 0

def reduce_metrics_across_gpus(metrics, epoch_samples, device):
    if not is_dist():
        return metrics, epoch_samples

    # reduce epoch_samples
    sample_tensor = torch.tensor([epoch_samples], dtype=torch.float64, device=device)
    dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)
    epoch_samples_total = int(sample_tensor.item())

    # reduce each metric
    reduced = {}
    for k, v in metrics.items():
        t = torch.tensor([v], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        reduced[k] = t.item()

    return reduced, epoch_samples_total

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append(f"{k}: {metrics[k] / epoch_samples:.6f}")
    if is_main_process():
        print(f"{phase}: {', '.join(outputs)}")

def save_checkpoint(epoch, model, optimizer, scheduler, checkpoint_path, extra_state=None):
    raw_model = model.module if hasattr(model, "module") else model

    save_dict = {
        'epoch': epoch,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }

    if extra_state is not None:
        save_dict.update(extra_state)

    torch.save(save_dict, checkpoint_path)

def save_last_checkpoint(epoch, model, optimizer, scheduler, output_path, model_stage, patch_res,
                         previous_last_path=None, save_epoch=None):
    if not is_main_process():
        return previous_last_path

    # delete previous rolling last checkpoint only if it is NOT a periodic checkpoint
    if previous_last_path and os.path.exists(previous_last_path):
        prev_name = os.path.basename(previous_last_path)
        try:
            prev_epoch = int(prev_name.split("_epoch_")[1].split(".pt")[0])
        except Exception:
            prev_epoch = None

        is_periodic = (prev_epoch is not None and save_epoch is not None and ((prev_epoch + 1) % save_epoch == 0))
        if not is_periodic:
            print(f"Removing previous last checkpoint: {previous_last_path}")
            os.remove(previous_last_path)

    checkpoint_name = f"{model_stage}_epoch_{epoch}.pt"
    checkpoint_path = os.path.join(output_path, checkpoint_name)

    save_checkpoint(
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_path=checkpoint_path,
    )

    return checkpoint_path

def train_with_dino(args, device, model, dataloaders, start_epoch):
    optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.5)

    use_amp = (device.type == "cuda")
    amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    last_checkpoint_path = ""
    phases = ['train', 'val_trained', 'val_novel']
    for epoch in range(start_epoch, args.num_epochs):
        torch.cuda.empty_cache()
        if is_main_process():
            print(f"Epoch {epoch}/{args.num_epochs - 1}\n----------")
        since = time.time()

        if "train" in dataloaders and hasattr(dataloaders["train"], "sampler") and hasattr(dataloaders["train"].sampler, "set_epoch"):
            dataloaders["train"].sampler.set_epoch(epoch)

        epoch_phase_metrics = {}

        for phase in phases:
            is_train = (phase == 'train')
            model.train() if is_train else model.eval()
            metrics = defaultdict(float)
            epoch_samples = 0
            loader = dataloaders[phase]
            pbar = tqdm(loader, desc=f"{phase} Epoch {epoch}",ncols=120, leave=False) if is_main_process() else loader

            for input_dict in pbar:
                x_partial = input_dict["input1"].float().to(device, non_blocking=True).contiguous() 
                x_gt = input_dict["gt"].float().to(device, non_blocking=True).contiguous()               

                if is_train:
                    optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(is_train):
                    with torch.autocast(
                        device_type="cuda",
                        dtype=amp_dtype,
                        enabled=use_amp
                    ):
                        output_dict = model(x_partial)

                        loss = calc_metrics(
                            x_gt,
                            output_dict["output"],
                            args.loss_weights,
                            metrics,
                        )

                    if torch.isnan(loss).any() and is_main_process():
                        print("NaN detected in loss!", name)

                    if is_train:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()

                epoch_samples += output_dict["output"].size(0)

                if is_main_process() and hasattr(pbar, "set_postfix"):
                    postfix_dict = {
                        "iou": f"{metrics['iou'] / max(epoch_samples, 1):.4f}",
                        "l1": f"{metrics['l1_loss'] / max(epoch_samples, 1):.4f}",
                        "amp": use_amp,
                    }
                    pbar.set_postfix(postfix_dict)

            metrics, epoch_samples = reduce_metrics_across_gpus(metrics, epoch_samples, device)
            print_metrics(metrics, epoch_samples, phase)

            epoch_iou = metrics['iou'] / max(epoch_samples, 1)
            epoch_l1 = metrics['l1_loss'] / max(epoch_samples, 1)

            epoch_phase_metrics[phase] = {
                "iou": epoch_iou,
                "l1_loss": epoch_l1,
            }

            # new: DDP-safe gather for category metrics
            if is_main_process():
                log_payload = {
                    f"iou/{phase}": epoch_iou,
                    f"l1_loss/{phase}": epoch_l1,
                    "epoch": epoch,
                    "amp_enabled": int(use_amp),
                }

                for k in metrics:
                    log_payload[f"{k}/{phase}"] = metrics[k] / max(epoch_samples, 1)
                wandb.log(log_payload, step=epoch)

        scheduler.step()

        # Save last checkpoint
        # -----------------------------
        last_checkpoint_path = save_last_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            output_path=args.output_path,
            model_stage=args.model_stage,
            patch_res=args.patch_res,
            previous_last_path=last_checkpoint_path,
            save_epoch=args.save_epoch
        )

        # -----------------------------
        # Save periodic checkpoint
        # -----------------------------
        if (epoch + 1) % args.save_epoch == 0 and is_main_process():
            model_name = f"{args.model_stage}_res_{args.patch_res}_epoch_{epoch}.pt"
            model_path = os.path.join(args.output_path, model_name)

            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=model_path
            )

            wandb.save(model_path)

        if is_main_process():
            print(f"Epoch time: {time.time() - since:.2f}s")