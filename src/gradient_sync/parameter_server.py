"""Parameter-server gradient synchronization module.
    Centralized aggregation: all workers send gradients to server (rank 0),
    server averages and broadcasts back to all workers.
    setup = initialize server and client connections
    average = perform parameter server aggregation
    teardown = close connections """

import torch


def setup_distributed(config: dict) -> dict:
    """Setup parameter server for distributed mode using socket endpoints."""
    rank = config["rank"]
    world_size = config["world_size"]
    
    # Rank 0 is the parameter server
    is_server = (rank == 0)
    
    if is_server:
        # Server role: needs connections to all clients
        client_endpoints = {}
        for client_rank in range(1, world_size):
            endpoint_key = f"client_{client_rank}_endpoint"
            if endpoint_key in config:
                client_endpoints[endpoint_key] = config[endpoint_key]
    else:
        # Client role: needs connection to server
        server_endpoint = config.get("server_endpoint")
        if server_endpoint is None:
            raise ValueError("parameter_server client requires server_endpoint in config")
        client_endpoints = {"server_endpoint": server_endpoint}
    
    print(
        f"[ps.setup] rank={rank} role={'server' if is_server else 'client'}",
        flush=True,
    )
    
    return {
        "rank": rank,
        "world_size": world_size,
        "is_server": is_server,
        **client_endpoints,
    }


def setup(config: dict) -> dict:
    """Initialize parameter server context."""
    mode = config.get("mode", "distributed")

    if mode != "distributed":
        raise ValueError("mode must be distributed")

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

    return (
        f"shape={tuple(tensor.shape)} "
        f"dtype={tensor.dtype} "
        f"norm={flat.norm().item():.6f} "
        f"max={flat.abs().max().item():.6f} "
        f"nonzero={torch.count_nonzero(flat).item()}/{flat.numel()}"
    )


def average(local_grad, comm_ctx, config: dict):
    """Perform parameter server based gradient aggregation."""
    grad_tensor = _normalize_tensor_grad(local_grad.get("gradients"))

    if comm_ctx is None:
        raise ValueError("parameter_server average requires comm_ctx from parameter_server.setup")
    
    rank = int(comm_ctx.get("rank", config.get("rank", 0)))
    world_size = int(comm_ctx.get("world_size", config.get("world_size", 1)))
    is_server = comm_ctx.get("is_server", False)
    
    if world_size <= 1:
        return {**local_grad, "gradients": grad_tensor}
    
    log_steps = bool(config.get("ps_step_logs", False))
    
    if is_server:
        # Server role: collect from all clients, average, broadcast back
        accumulated = grad_tensor.clone()
        num_clients_received = 0
        
        # Receive from all clients (ranks 1 to world_size-1)
        for client_rank in range(1, world_size):
            endpoint_key = f"client_{client_rank}_endpoint"
            endpoint = comm_ctx.get(endpoint_key)
            
            if endpoint is not None:
                try:
                    client_grad = endpoint.recv()
                    client_tensor = _normalize_tensor_grad(client_grad)
                    accumulated.add_(client_tensor)
                    num_clients_received += 1
                    if log_steps:
                        print(f"[ps.average] rank={rank} server_recv_from_client={client_rank}", flush=True)
                except Exception as e:
                    print(f"[ps.average] rank={rank} server error receiving from client {client_rank}: {e}", flush=True)
        
        # Average
        averaged_tensor = accumulated / float(world_size)
        
        # Broadcast to all clients
        for client_rank in range(1, world_size):
            endpoint_key = f"client_{client_rank}_endpoint"
            endpoint = comm_ctx.get(endpoint_key)
            
            if endpoint is not None:
                try:
                    endpoint.send(averaged_tensor)
                    if log_steps:
                        print(f"[ps.average] rank={rank} server_send_to_client={client_rank}", flush=True)
                except Exception as e:
                    print(f"[ps.average] rank={rank} server error sending to client {client_rank}: {e}", flush=True)
    else:
        # Client role: send local gradient to server, wait for averaged gradient
        server_endpoint = comm_ctx.get("server_endpoint")
        
        if server_endpoint is None:
            raise ValueError("parameter_server client missing server connection")
        
        try:
            server_endpoint.send(grad_tensor)
            if log_steps:
                print(f"[ps.average] rank={rank} client_send_to_server", flush=True)
            
            # Receive averaged gradient from server
            server_grad = server_endpoint.recv()
            averaged_tensor = _normalize_tensor_grad(server_grad)
            if log_steps:
                print(f"[ps.average] rank={rank} client_recv_from_server", flush=True)
        except Exception as e:
            print(f"[ps.average] rank={rank} client error communicating with server: {e}", flush=True)
            averaged_tensor = grad_tensor
    
    if log_steps:
        print(
            f"[ps.average] rank={rank} final_avg {_tensor_summary(averaged_tensor)}",
            flush=True,
        )

    return {
        **local_grad,
        "gradients": averaged_tensor,
    }


def teardown(comm_ctx) -> None:
    """Close all parameter server connections."""
    if comm_ctx is None:
        return
    
    rank = comm_ctx.get("rank", "unknown")
    is_server = comm_ctx.get("is_server", False)
    world_size = comm_ctx.get("world_size", 1)
    
    if is_server:
        # Close all client connections
        for client_rank in range(1, world_size):
            endpoint_key = f"client_{client_rank}_endpoint"
            endpoint = comm_ctx.get(endpoint_key)
            if endpoint is None:
                continue
            try:
                endpoint.close()
            except OSError as error:
                print(f"[ps.teardown] warning: rank={rank} failed to close {endpoint_key}: {error}", flush=True)
    else:
        # Close server connection
        endpoint_key = "server_endpoint"
        endpoint = comm_ctx.get(endpoint_key)
        if endpoint is None:
            return
        try:
            endpoint.close()
        except OSError as error:
            print(f"[ps.teardown] warning: rank={rank} failed to close {endpoint_key}: {error}", flush=True)
    
    # Close listener if it exists
    listener = comm_ctx.get("_listener")
    if listener is not None:
        try:
            listener.close()
        except OSError as error:
            print(f"[ps.teardown] warning: rank={rank} failed to close listener: {error}", flush=True)
