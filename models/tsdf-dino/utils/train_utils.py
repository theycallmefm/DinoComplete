import os
import shutil
import yaml
import torch


def snapshot_config(cfg, model_module, exp_dir):
    os.makedirs(exp_dir, exist_ok=True)

    if hasattr(cfg, "config_path") and os.path.exists(cfg.config_path):
        shutil.copyfile(
            cfg.config_path,
            os.path.join(exp_dir, "config_distillation.yaml"),
        )

    with open(os.path.join(exp_dir, "config_distillation.yaml"), "w") as f:
        yaml.safe_dump(vars(cfg), f, sort_keys=False)

    try:
        shutil.copyfile(
            os.path.abspath(model_module.__file__),
            os.path.join(exp_dir, "model.py"),
        )
    except Exception as e:
        print(f"[WARN] could not snapshot model.py: {e}")

    try:
        shutil.copyfile(
            os.path.abspath(__import__("__main__").__file__),
            os.path.join(exp_dir, "train_clean.py"),
        )
    except Exception as e:
        print(f"[WARN] could not snapshot training script: {e}")


def save_checkpoint(path, model, optimizer, scheduler, epoch, cfg, extra=None):
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "config": vars(cfg),
    }

    if extra:
        payload.update(extra)

    torch.save(payload, path)