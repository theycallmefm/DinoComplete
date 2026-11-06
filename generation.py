import os
import sys
import signal
import faulthandler

# dump python stack traces on: kill -USR1 <pid>
faulthandler.enable()
faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
import argparse
import subprocess
import numpy as np
import torch
from torch.utils.data import DataLoader
import trimesh
from skimage import measure
import yaml
import shutil
from tqdm import tqdm

import data.dataset as dataset

from utils.builder import build_hybrid_model_from_config


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def merge_pretrained_and_eval_args(eval_args):
    train_config_path = os.path.join(eval_args.model_path, "config.yaml")

    if not os.path.isfile(train_config_path):
        raise FileNotFoundError(f"Missing pretrained config: {train_config_path}")

    train_cfg = load_yaml(train_config_path)
    eval_cfg = vars(eval_args)

    # evaluation.yaml overrides pretrained config.yaml
    merged = {**train_cfg, **eval_cfg}

    # builder expects this
    merged["output_path"] = eval_args.model_path

    return argparse.Namespace(**merged)

# -------------------------
# YAML args
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to config.yaml")
    cli = parser.parse_args()

    with open(cli.config, "r") as f:
        config_dict = yaml.safe_load(f)

    # defaults if missing
    config_dict.setdefault("do_generation", True)
    config_dict.setdefault("do_evaluation", False)
    config_dict.setdefault("evaluation_root", "../PatchComplete/data")

    return argparse.Namespace(**config_dict)


# -------------------------
# Mesh + save utils
# -------------------------
def extract_and_save_mesh(tsdf, save_path, level=0.0):
    try:
        tsdf_min = tsdf.min()
        tsdf_max = tsdf.max()
        level = (tsdf_min + tsdf_max) / 2.0
        print("Marching cubes level:", level)

        verts, faces, normals, _ = measure.marching_cubes(tsdf, level=level)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)
        mesh.export(save_path)
        print(f"Saved mesh: {save_path}")
    except Exception as e:
        print(f"Marching cubes failed at {save_path} with error: {e}")


def save_prediction(data_to_save, name, output_path, dataset, save_mesh=False):
    if dataset == "shapenet":
        parts = name.split("/")[-3:-1]
    else:
        parts = [name.split("/")[-2]]

    folder_path = os.path.join(output_path, *parts)
    os.makedirs(folder_path, exist_ok=True)

    file_name = name.split("/")[-1][:-4] + "_pred.npz"
    np.savez(os.path.join(folder_path, file_name), data=data_to_save)

    if save_mesh:
        mesh_name = name.split("/")[-1][:-4] + "_pred_mesh.obj"
        extract_and_save_mesh(
            data_to_save["output"].squeeze(0),
            os.path.join(folder_path, mesh_name),
        )


# -------------------------
# Generation
# -------------------------
def generate(model, dataloader, output_path, dataset, save_mesh=False):
    model.eval()
    total = len(dataloader.dataset)
    seen = 0

    pbar = tqdm(dataloader, desc=f"Generating samples",ncols=120, leave=False) 
    with torch.no_grad():
        for batch in pbar:
            batch["input"], names = batch["input1"].to(model._device), batch["name"]
            batch_size = len(names)
            seen += batch_size

            
            outputs = model(batch["input"])

            batch_size = len(names)
            for i in range(batch_size):
                data_to_save = {
                    k: (v[i].cpu().numpy() if isinstance(v, torch.Tensor) else v[i])
                    for k, v in outputs.items()
                }
                save_prediction(data_to_save, names[i], output_path, dataset, save_mesh)


def get_dataloader(args):
    test_file = args.test_file

    if args.dataset == "shapenet":
        base_set = dataset.ShapeNetDataset(args, test_file)

    elif args.dataset == "scannet":
        base_set = dataset.ScannetDataset(args, test_file)
    else:
        raise ValueError("Invalid dataset")

    return DataLoader(base_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


# -------------------------
# Evaluation
# -------------------------
def run_evaluation(args, pred_path: str):
    """
    Runs: python evaluation/evaluation.py --dataset <dataset> --pred_path <pred_path> --root <evaluation_root>
    but with cwd=repo_root/evaluation to make relative paths + chamfer compile work.
    """
    repo_root = os.path.abspath(os.getcwd())  # assume you run generation.py from repo root
    eval_dir = os.path.join(repo_root, "evaluation")
    eval_py = os.path.join(eval_dir, "evaluation.py")

    if not os.path.isfile(eval_py):
        raise FileNotFoundError(f"Missing evaluation script at: {eval_py}")

    cmd = [
        sys.executable if "sys" in globals() else "python",
        "evaluation.py",
        "--dataset", str(args.dataset),
        "--pred_path", os.path.abspath(pred_path),
        "--root", os.path.abspath(args.evaluation_root),
        "--test_file", "../"+ str(args.test_file)
    ]

    print("\n=== Running evaluation ===")
    print("CWD:", eval_dir)
    print("CMD:", " ".join(cmd))

    # Run from evaluation/ so chamfer_distance relative sources resolve
    subprocess.run(cmd, cwd=eval_dir, check=True)
    print("=== Evaluation done ===\n")


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()

    # output folder: <model_path>/test/<dataset>
    #args.output_path = os.path.join(args.model_path, "test", args.dataset)
    
    ckpt_stem = os.path.splitext(args.model_name)[0]
    args.output_base = os.path.join(args.model_path, "test_runs", ckpt_stem)
    args.output_path = os.path.join(args.output_base, args.dataset)
    os.makedirs(args.output_path, exist_ok=True)

    # device
    args.gpu = [int(g) for g in str(args.gpu).split(",")]
    device = torch.device(f"cuda:{args.gpu[0]}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load model using your hybrid loader
    # merge pretrained training config + evaluation config
    model_args = merge_pretrained_and_eval_args(args)

    # keep generation output paths from evaluation.yaml logic
    model_args.output_base = args.output_base
    model_args.output_path = args.output_path

    model_ckpt_path = os.path.join(args.model_path, args.model_name)

    model, model_args = build_hybrid_model_from_config(
        model_args,
        eval=True,
        ckpt_path=model_ckpt_path,
        device=device,
        verbose=True,
    )
    model = model.to(device)
    print(f"Loaded model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    # generation
    if getattr(args, "do_generation", True):
        test_loader = get_dataloader(args)
        print("Loaded test data.")
        print("Generating predictions...")
        generate(
            model,
            test_loader,
            args.output_path,
            args.dataset,
            save_mesh=getattr(args, "save_mesh", False),
        )
        print("Generation done.")
    else:
        print("Skipping generation (do_generation: false).")

    # evaluation
    if getattr(args, "do_evaluation", False):
        # Evaluate the same folder you just generated into
        #pred_base = os.path.join(args.model_path, "test")  # NOTE: not ".../test/scannet"
        pred_base = args.output_base
        run_evaluation(args, pred_path=pred_base)
        # -------------------------
        # Optional cleanup
        # -------------------------
        if getattr(args, "cleanup_test_after_eval", False):
            #test_dir = os.path.join(args.model_path, "test")
            test_dir = args.output_base
            if os.path.isdir(test_dir):
                print(f"🧹 Cleaning up test folder: {test_dir}")
                shutil.rmtree(test_dir)
            else:
                print(f"⚠️ Test folder not found, nothing to clean: {test_dir}")
    else:
        print("Skipping evaluation (do_evaluation: false).")


if __name__ == "__main__":
    main()
