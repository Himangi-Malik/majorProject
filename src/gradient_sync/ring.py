""" Ring gradient synchronization placeholder module.
    setup = open/init resources
    average = do communication work
    teardown = close/free resources """

import torch


def setup_distributed(config: dict) -> dict:
    left_endpoint = config.get("left_endpoint")
    right_endpoint = config.get("right_endpoint")

    if left_endpoint is None or right_endpoint is None:
        raise ValueError(
            "distributed ring setup requires left_endpoint and right_endpoint in config"
        )

    print(
        f"[ring.setup] rank={config['rank']} ",
        flush=True,
    )
    return {
        "rank": config["rank"],
        "world_size": config["world_size"],
        "left_endpoint": left_endpoint,
        "right_endpoint": right_endpoint,
    }


def setup(config: dict) -> dict:
    return setup_distributed(config)


def _normalize_tensor_grad(grad_tensor):
    """Return gradient as a float32 tensor."""

    if grad_tensor is None:
        raise ValueError("local_grad must contain a 'gradients' field")

    if isinstance(grad_tensor, dict):
        grad_tensor = grad_tensor.get("gradients")

    if not isinstance(grad_tensor, torch.Tensor):
        grad_tensor = torch.as_tensor(
            grad_tensor,
            dtype=torch.float32,
        )
    elif grad_tensor.dtype != torch.float32:
        grad_tensor = grad_tensor.to(dtype=torch.float32)

    if grad_tensor.ndim == 0:
        grad_tensor = grad_tensor.unsqueeze(0)

    return grad_tensor


def _tensor_summary(tensor: torch.Tensor) -> str:
    flat = tensor.detach().flatten()
    sample = flat[: min(4, flat.numel())].tolist()
    return f"shape={tuple(tensor.shape)} dtype={tensor.dtype} sample={sample}"


def _chunk_sizes(total_elems: int, world_size: int) -> list:
    """Split total_elems into world_size near-equal chunk sizes."""
    base, remainder = divmod(total_elems, world_size)
    return [base + 1 if i < remainder else base for i in range(world_size)]


def _exchange(send_tensor, left_endpoint, right_endpoint, rank: int):
    """Send to the right neighbor and receive from the left neighbor, using
    parity-based ordering so ranks don't deadlock on a simultaneous send."""
    if rank % 2 == 0:
        right_endpoint.send(send_tensor)
        recv_tensor = left_endpoint.recv()
    else:
        recv_tensor = left_endpoint.recv()
        right_endpoint.send(send_tensor)
    return recv_tensor


def average(local_grad, comm_ctx, config: dict):
    grad_tensor = _normalize_tensor_grad(local_grad.get("gradients"))

    if comm_ctx is None:
        raise ValueError("ring average requires comm_ctx from ring.setup")

    world_size = int(comm_ctx.get("world_size", config.get("world_size", 1)))
    left_endpoint = comm_ctx.get("left_endpoint")
    right_endpoint = comm_ctx.get("right_endpoint")
    rank = int(comm_ctx.get("rank", config.get("rank", 0)))
    log_cycles = bool(config.get("ring_cycle_logs", False))

    if world_size <= 1:
        averaged_tensor = grad_tensor
    else:
        if left_endpoint is None or right_endpoint is None:
            raise ValueError("ring average requires left_endpoint and right_endpoint in comm_ctx")

        # Chunked ring all-reduce: split the tensor into world_size chunks so
        # each of the 2*(world_size-1) hops moves ~1/world_size of the data,
        # instead of shipping the full tensor at every hop.
        sizes = _chunk_sizes(grad_tensor.numel(), world_size)
        chunks = [c.clone() for c in grad_tensor.split(sizes)]

        # Phase 1: scatter-reduce. After world_size-1 steps, chunk[(rank+1) % world_size]
        # holds the fully summed value for that chunk.
        for step in range(world_size - 1):
            send_idx = (rank - step) % world_size
            recv_idx = (rank - step - 1) % world_size
            recv_chunk = _exchange(chunks[send_idx], left_endpoint, right_endpoint, rank)
            chunks[recv_idx].add_(_normalize_tensor_grad(recv_chunk))
            if log_cycles:
                print(f"[ring.average] rank={rank} scatter_reduce step={step + 1}/{world_size - 1}", flush=True)

        # Phase 2: all-gather. Propagate each fully-summed chunk around the ring
        # so every rank ends up with every chunk.
        for step in range(world_size - 1):
            send_idx = (rank - step + 1) % world_size
            recv_idx = (rank - step) % world_size
            recv_chunk = _exchange(chunks[send_idx], left_endpoint, right_endpoint, rank)
            chunks[recv_idx] = _normalize_tensor_grad(recv_chunk)
            if log_cycles:
                print(f"[ring.average] rank={rank} all_gather step={step + 1}/{world_size - 1}", flush=True)

        averaged_tensor = torch.cat(chunks) / float(world_size)

    if log_cycles:
        print(
            f"[ring.average] rank={rank} final_avg {_tensor_summary(averaged_tensor)}",
            flush=True,
        )

    return {
        **local_grad,
        "gradients": averaged_tensor,
    }


def teardown(comm_ctx) -> None:
    if comm_ctx is None:
        return

    for key in ("left_endpoint", "right_endpoint"):
        endpoint = comm_ctx.get(key)
        if endpoint is None:
            continue
        try:
            endpoint.close()
        except OSError as error:
            rank = comm_ctx.get("rank", "unknown")
            print(f"[ring.teardown] warning: rank={rank} failed to close {key}: {error}", flush=True)