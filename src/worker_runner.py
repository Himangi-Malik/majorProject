from gradient_sync import parameter_server, ring, tree
from models import (
    ann_small,
    ann_medium,
    ann_large,
    cnn_small,
    cnn_medium,
    cnn_large,
    rnn_small,
    rnn_medium,
    rnn_large,
)

import torch
import time
from pathlib import Path

from metrics.rank_metrics import RankMetrics
from metrics.step_metrics import StepMetrics
from data.fashion_mnist import get_dataloader


def get_algo_module(algo_tag: str):
    return {
        "ring": ring,
        "tree": tree,
        "parameter_server": parameter_server,
    }[algo_tag]


def get_model_module(model_tag: str, model_size: str = "medium"):
    return {
        "ann": {"small": ann_small, "medium": ann_medium, "large": ann_large},
        "cnn": {"small": cnn_small, "medium": cnn_medium, "large": cnn_large},
        "rnn": {"small": rnn_small, "medium": rnn_medium, "large": rnn_large},
    }[model_tag][model_size]


def _compute_grad_norm(grad_tensor: torch.Tensor) -> float:
    """Compute L2 norm of gradient tensor."""
    if isinstance(grad_tensor, torch.Tensor):
        return torch.norm(grad_tensor).item()
    return 0.0


def _snapshot_endpoint_bytes(comm_ctx) -> tuple[int, int]:
    bytes_sent = 0
    bytes_received = 0

    for value in comm_ctx.values():
        bytes_sent += getattr(value, "bytes_sent", 0)
        bytes_received += getattr(value, "bytes_received", 0)

    return bytes_sent, bytes_received


def run_worker(config):
    algo_module = get_algo_module(config["algo"])
    model_module = get_model_module(config["model"], config.get("model_size", "medium"))

    rank = config["rank"]
    world_size = config["world_size"]

    epochs = int(config.get("epochs", 1))
    steps_per_epoch = int(config.get("steps_per_epoch", 1))

    output_dir = Path(
        config.get("benchmark_results_dir", "benchmark_results")
    )

    metrics = RankMetrics(rank, world_size, output_dir)

    print(
        f"[rank {rank}] start "
        f"algo={config['algo']} "
        f"model={config['model']} "
        f"model_size={config.get('model_size', 'medium')} "
        f"epochs={epochs} "
        f"steps={steps_per_epoch}",
        flush=True,
    )

    comm_ctx = None

    setup_start = time.perf_counter()

    try:
        # ---------------- Setup ----------------
        comm_ctx = algo_module.setup(config)
        state = model_module.build_model(config)

        train_loader = get_dataloader(
            batch_size=config["batch_size"],
            rank=config["rank"],
            world_size=config["world_size"]
        )

        train_iterator = iter(train_loader)

        setup_time = time.perf_counter() - setup_start

        print(
            f"[rank {rank}] setup complete "
            f"({setup_time:.4f}s)",
            flush=True,
        )

        # ---------------- Training Loop ----------------
        for epoch in range(epochs):
            train_loader.sampler.set_epoch(epoch)
            for step in range(steps_per_epoch):

                step_config = {
                    **config,
                    "current_epoch": epoch,
                    "step": step,
                }
                # ---------- Get next batch ----------
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)

                # ---------- Compute ----------
                compute_start = time.perf_counter()
                iteration_start = compute_start
                comm_before = _snapshot_endpoint_bytes(comm_ctx)

                local_grad = model_module.train_step(
                    state,
                    batch,
                    step_config,
                )

                compute_time = time.perf_counter() - compute_start
                loss = local_grad.get("loss", 0.0)

                # ---------- Synchronization ----------
                start = time.perf_counter()

                synced_grad = algo_module.average(
                    local_grad,
                    comm_ctx,
                    config,
                )

                sync_time = time.perf_counter() - start

                if isinstance(synced_grad, dict):
                    grad_tensor = synced_grad["gradients"]
                else:
                    grad_tensor = synced_grad

                grad_norm = _compute_grad_norm(grad_tensor)

                # ---------- Optimizer ----------
                start = time.perf_counter()

                model_module.apply_synced_gradients(
                    state,
                    grad_tensor,
                )

                optim_time = time.perf_counter() - start
                iteration_time = time.perf_counter() - iteration_start
                comm_after = _snapshot_endpoint_bytes(comm_ctx)
                bytes_sent = comm_after[0] - comm_before[0]
                bytes_received = comm_after[1] - comm_before[1]

                step_metrics = StepMetrics(
                    epoch=epoch,
                    step=step,
                    is_warmup=(epoch == 0 and step == 0),

                    compute_time=compute_time,
                    sync_time=sync_time,
                    optim_time=optim_time,
                    iteration_time=iteration_time,

                    loss=float(loss)
                    if isinstance(loss, torch.Tensor)
                    else loss,

                    grad_norm=grad_norm,

                    bytes_sent=bytes_sent,
                    bytes_received=bytes_received,
                )

                metrics.add_step(step_metrics)
            print(
                    f"[rank {rank}] "
                    f"epoch={epoch} "
                    f"step={step} "
                    f"compute={compute_time:.4f}s "
                    f"sync={sync_time:.4f}s "
                    f"optim={optim_time:.4f}s "
                    f"total={iteration_time:.4f}s "
                    f"loss={loss:.6f} "
                    f"grad_norm={grad_norm:.6f}",
                    flush=True,
                )

    finally:
        if comm_ctx is not None:
            algo_module.teardown(comm_ctx)

        metrics_path = metrics.save()
        print(
            f"[rank {rank}] metrics saved to {metrics_path}",
            flush=True,
        )

        print(
            f"[rank {rank}] teardown complete",
            flush=True,
        )

        print(
            f"[rank {rank}] metrics: {metrics_path}",
            flush=True,
        )