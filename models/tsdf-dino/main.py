import os
import yaml
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config yaml",
    )

    parser.add_argument(
        "--train_script",
        type=str,
        default="train_dino.py",
        help="Training script",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    gpus = cfg.get("gpu", None)

    if gpus is None or len(gpus) == 0:
        raise ValueError(
            "Config must contain:\n"
            "gpu:\n"
            "- 0\n"
            "- 1"
        )

    gpu_str = ",".join(str(g) for g in gpus)
    nproc = len(gpus)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_str

    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc}",
        args.train_script,
        "--config",
        args.config,
    ]

    print(f"[Launcher] config={args.config}")
    print(f"[Launcher] CUDA_VISIBLE_DEVICES={gpu_str}")
    print(f"[Launcher] nproc_per_node={nproc}")
    print("[Launcher] command:", " ".join(cmd))

    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()