import os
import torch
import torch.distributed as dist


def setup_ddp():
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    return device, rank, local_rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def is_rank0():
    return dist.get_rank() == 0


def reduce_mean_stats(stats, device):
    keys = list(stats.keys())

    sums = torch.tensor(
        [stats[k][0] for k in keys],
        device=device,
        dtype=torch.float64,
    )
    counts = torch.tensor(
        [stats[k][1] for k in keys],
        device=device,
        dtype=torch.float64,
    )

    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    return {
        k: (sums[i] / counts[i].clamp_min(1.0)).item()
        for i, k in enumerate(keys)
    }