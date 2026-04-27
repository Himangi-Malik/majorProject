""" Ring gradient synchronization placeholder module.
    setup = open/init resources
    average = do communication work
    teardown = close/free resources """

import torch

class _SocketEndpointPlaceholder:
    def __init__(self, direction: str):
        self.direction = direction

    def send(self, _payload):
        raise NotImplementedError(f"socket stream send not implemented yet ({self.direction})")

    def recv(self):
        raise NotImplementedError(f"socket stream recv not implemented yet ({self.direction})")

    def close(self):
        return None


def setup_distributed(config: dict) -> dict:
    left_endpoint = config.get("left_endpoint")
    right_endpoint = config.get("right_endpoint")

    if left_endpoint is None or right_endpoint is None:
        raise ValueError(
            "distributed ring setup requires left_endpoint and right_endpoint in config"
        )

    print(
        f"[ring.setup] rank={config['rank']} mode=distributed transport=socket placeholder_endpoints=false",
        flush=True,
    )
    return {
        "mode": "distributed",
        "rank": config["rank"],
        "world_size": config["world_size"],
        "left_endpoint": left_endpoint,
        "right_endpoint": right_endpoint,
        "left_endpoint_info": config.get("left_endpoint_info"),
        "right_endpoint_info": config.get("right_endpoint_info"),
        "transport": "socket",
    }

def setup_local(config: dict) -> dict:
     return {
            "mode": "local",
            "rank": config["rank"],
            "world_size": config["world_size"],
            "left_endpoint": config.get("left_conn"),
            "right_endpoint": config.get("right_conn"),
            "left_endpoint_info": config.get("left_endpoint_info"),
            "right_endpoint_info": config.get("right_endpoint_info"),
            "transport": "pipe",
        }


def setup(config: dict) -> dict:
    mode = config["mode"]

    if mode == "local":
        return setup_local(config)

    elif mode == "distributed":
        return setup_distributed(config)

    raise ValueError("mode must be one of: local, distributed")


def _normalize_tensor_grad(grad_tensor):
    if grad_tensor is None:
        raise ValueError("local_grad must contain a 'gradients' field")

    if not isinstance(grad_tensor, torch.Tensor):
        grad_tensor = torch.as_tensor(grad_tensor, dtype=torch.float32)
    else:
        grad_tensor = grad_tensor.detach().clone().to(dtype=torch.float32)

    if grad_tensor.ndim == 0:
        grad_tensor = grad_tensor.unsqueeze(0)

    return grad_tensor


def _tensor_summary(tensor: torch.Tensor) -> str:
    flat = tensor.detach().flatten()
    sample = flat[: min(4, flat.numel())].tolist()
    return f"shape={tuple(tensor.shape)} dtype={tensor.dtype} sample={sample}"


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

        # Tiny, slow ring pass: circulate tensors and accumulate.
        send_buf = grad_tensor.clone()
        running_sum = grad_tensor.clone()

        for _ in range(world_size - 1):
            if rank % 2 == 0:
                right_endpoint.send(send_buf)
                recv_buf = left_endpoint.recv()
            else:
                recv_buf = left_endpoint.recv()
                right_endpoint.send(send_buf)

            recv_tensor = _normalize_tensor_grad(recv_buf)
            running_sum = running_sum + recv_tensor
            send_buf = recv_tensor

            if log_cycles:
                cycle_avg = running_sum / float(_ + 2)
                print(
                    f"[ring.average] rank={rank} cycle={_ + 1}/{world_size - 1} running_avg {_tensor_summary(cycle_avg)}",
                    flush=True,
                )

        averaged_tensor = running_sum / float(world_size)

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