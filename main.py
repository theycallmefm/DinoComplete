import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.cuda\.amp\.custom_fwd.*deprecated.*",
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.cuda\.amp\.custom_bwd.*deprecated.*",
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*You are using `torch\.load` with `weights_only=False`.*",
)

import os
import random
import yaml
import argparse
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import inspect
import sys
import textwrap
import types
import data.dataset as dataset
from models import shape_completion
from utils.builder import build_hybrid_model_from_config, configure_fine_tuning
from train import train_with_dino

# ---------------------------------- CONFIG HANDLING ----------------------------------
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return argparse.Namespace(**config)


# ---------------------------------- UTILS ----------------------------------
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def is_main_process():
    return get_rank() == 0

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def save_experiment_code_full(args, model_instance, dataset_instance):
    checkpoint_dir = args.output_path
    os.makedirs(checkpoint_dir, exist_ok=True)

    collected_code = []
    seen = set()

    def collect_source(obj):
        if obj in seen or not hasattr(obj, "__module__") or not hasattr(obj, "__name__"):
            return
        seen.add(obj)
        try:
            source = inspect.getsource(obj)
            collected_code.append(textwrap.dedent(source).strip())
        except Exception as e:
            print(f"Could not retrieve source for {obj}: {e}")

    def recursively_collect_class_and_submodules(instance):
        collect_source(instance.__class__)
        for attr in vars(instance).values():
            if isinstance(attr, nn.Module):
                collect_source(attr.__class__)
            elif isinstance(attr, (types.FunctionType, types.MethodType)):
                collect_source(attr)
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, nn.Module):
                        collect_source(item.__class__)

    recursively_collect_class_and_submodules(model_instance)
    recursively_collect_class_and_submodules(dataset_instance)

    code_path = os.path.join(checkpoint_dir, "experiment_code.py")
    with open(code_path, "w") as f:
        f.write("import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n")
        for block in collected_code:
            f.write(block + "\n\n")
    print(f"Saved experiment code to {code_path}")

    with open(args.config, "r") as f:
        original_config = yaml.safe_load(f)

    config_path = os.path.join(checkpoint_dir, "config.yaml")
    with open(config_path, "w") as yaml_file:
        yaml.dump(original_config, yaml_file, sort_keys=False)
    print(f"Config saved to {config_path}")

    if is_main_process():
        wandb.save(os.path.join(args.output_path, "experiment_code.py"))
        wandb.save(os.path.join(args.output_path, "config.yaml"))

def setup_ddp(rank, world_size, master_port="8"):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    if is_dist():
        dist.destroy_process_group()

def build_dataloaders(args, g, distributed):
    shapenet_dataset_cls = getattr(dataset, args.shapenet_dataset_class)
    scannet_dataset_cls = getattr(dataset, args.scannet_dataset_class)

    if args.dataset == "shapenet":
        train_set = shapenet_dataset_cls(args, args.train_file, train=True)
        val_trained_set = shapenet_dataset_cls(args, args.val_trained_file)
        val_novel_set = shapenet_dataset_cls(args, args.val_novel_file)
    elif args.dataset == "scannet":
        train_set = scannet_dataset_cls(args, args.train_file, use_bbox=True)
        val_trained_set = scannet_dataset_cls(args, args.val_trained_file, use_bbox=True)
        val_novel_set = scannet_dataset_cls(args, args.val_novel_file, use_bbox=True)
    else:
        raise ValueError("Invalid dataset specified. Use 'shapenet' or 'scannet'.")

    train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    val_trained_sampler = DistributedSampler(val_trained_set, shuffle=False) if distributed else None
    val_novel_sampler = DistributedSampler(val_novel_set, shuffle=False) if distributed else None

    dataloaders = {
        "train": DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            worker_init_fn=seed_worker,
            generator=g,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=2 if args.num_workers > 0 else None,
        ),
        "val_trained": DataLoader(
            val_trained_set,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_trained_sampler,
            worker_init_fn=seed_worker,
            generator=g,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=2 if args.num_workers > 0 else None,
        ),
        "val_novel": DataLoader(
            val_novel_set,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_novel_sampler,
            worker_init_fn=seed_worker,
            generator=g,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=2 if args.num_workers > 0 else None,
        )
    }

    return dataloaders, train_set

def main_worker(local_rank, args):
    distributed = len(args.gpu) > 1
    rank = local_rank
    world_size = len(args.gpu)

    if distributed:
        setup_ddp(rank, world_size, args.master_port)

    device_id = args.gpu[local_rank]
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    manual_seed = args.seed if hasattr(args, "seed") else 42
    random.seed(manual_seed + rank)
    np.random.seed(manual_seed + rank)
    torch.manual_seed(manual_seed + rank)
    torch.cuda.manual_seed(manual_seed + rank)

    g = torch.Generator()
    g.manual_seed(manual_seed + rank)

    torch.backends.cudnn.benchmark = True

    if is_main_process():
        run_name = os.path.basename(os.path.normpath(os.path.expanduser(os.path.expandvars(args.output_path))))
        run = wandb.init(
            project="dino-complete",
            name=run_name,
            config=vars(args),
            tags=[args.dataset, args.model_class],
            dir=args.output_path,
        )
        print("W&B:", {"entity": run.entity, "project": run.project, "id": run.id, "url": run.url})

    print(f"[rank {rank}] Using device: {device}")

    dataloaders, train_set = build_dataloaders(args, g, distributed)

    model_refine = build_hybrid_model_from_config(
        args,
        eval=False,
        device=device,
        verbose=is_main_process()
    )[0]

    if args.load_model is not None:
        model_path = os.path.join(args.load_model)
        ckpt = torch.load(model_path, map_location=device)
        model_refine.load_state_dict(ckpt["model_state_dict"]) #, strict=False)
        if args.model_stage != "fine_tune":
            start_epoch = ckpt.get("epoch", 0)
            if is_main_process():
                print(f"Loaded model {args.load_model} from epoch {start_epoch}")
        else:
            configure_fine_tuning(
                model_refine,
                getattr(args, "fine_tuned_modules", None),
                verbose=is_main_process() if "is_main_process" in globals() else True
            )
            start_epoch = 0
        

    else:
        start_epoch = 0

    model_refine = model_refine.to(device)

    if distributed:
        for p in model_refine.parameters():
            if p.is_floating_point():
                p.data = p.data.contiguous()
        model_refine = DDP(
            model_refine,
            device_ids=[device_id],
            output_device=device_id,
            find_unused_parameters=False,#true
        )

    if is_main_process():
        raw_model = model_refine.module if hasattr(model_refine, "module") else model_refine
        print("Model initialized with", sum(p.numel() for p in raw_model.parameters() if p.requires_grad) / 1e6, "M trainable parameters")
        save_experiment_code_full(args, raw_model, train_set)
        print("LOSS WEIGHTS:", args.loss_weights)
        #print("CUBLAS_WORKSPACE_CONFIG =", os.environ.get("CUBLAS_WORKSPACE_CONFIG", "Not set"))

    

    train_with_dino(args, device, model_refine, dataloaders, start_epoch)

    if distributed:
        cleanup_ddp()

def main(args):
    if not os.path.exists(args.data_path):
        raise FileNotFoundError("Invalid data_path")
    if not os.path.exists(args.train_file):
        raise FileNotFoundError("Invalid train_file")
    if not os.path.exists(args.val_trained_file):
        raise FileNotFoundError("Invalid val_trained_file")
    if not os.path.exists(args.val_novel_file):
        raise FileNotFoundError("Invalid val_novel_file")
    os.makedirs(args.output_path, exist_ok=True)

    # gpu is a YAML list, e.g. gpu: [0] or gpu: [0, 2]
    if not isinstance(args.gpu, list):
        raise ValueError(f"Expected args.gpu to be a list, got {type(args.gpu)}")
    args.gpu = [int(x) for x in args.gpu]

    if len(args.gpu) == 1:
        main_worker(0, args)
    else:
        mp.spawn(main_worker, nprocs=len(args.gpu), args=(args,), join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args_cmd = parser.parse_args()
    args = load_config(args_cmd.config)

    os.makedirs(args.output_path, exist_ok=True)
    if args.save_log:
        log_file_path = os.path.join(args.output_path, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        sys.stdout = Tee(sys.stdout, open(log_file_path, "w"))
        sys.stderr = sys.stdout

    main(args)