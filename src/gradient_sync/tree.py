"""Tree gradient synchronization module.
    Binary tree aggregation: reduce phase (leaves → root), broadcast phase (root → leaves).
    setup = initialize tree structure and endpoints
    average = perform tree aggregation
    teardown = close connections """

import torch

def _build_binary_tree(rank: int, world_size: int) -> dict:
    """Compute parent, left_child, right_child for binary tree topology."""
    if world_size <= 1:
        return {"parent": None, "left_child": None, "right_child": None}
    
    parent = (rank - 1) // 2 if rank > 0 else None
    left_child = 2 * rank + 1 if 2 * rank + 1 < world_size else None
    right_child = 2 * rank + 2 if 2 * rank + 2 < world_size else None
    
    return {"parent": parent, "left_child": left_child, "right_child": right_child}


def setup_distributed(config: dict) -> dict:
    """Setup tree for distributed mode using socket endpoints created in dist_launcher."""
    rank = config["rank"]
    world_size = config["world_size"]
    tree_structure = _build_binary_tree(rank, world_size)
    
    endpoints = {}
    for child_type in ["left_child", "right_child", "parent"]:
        endpoint_key = f"{child_type}_endpoint"
        if tree_structure[child_type] is not None:
            endpoints[endpoint_key] = config.get(endpoint_key)
    
    print(
        f"[tree.setup] rank={rank} mode=distributed tree={tree_structure}",
        flush=True,
    )
    
    return {
        "rank": rank,
        "world_size": world_size,
        "tree_structure": tree_structure,
        **endpoints,
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
    """Return a concise tensor summary."""
    flat = tensor.detach().flatten()
    sample = flat[:min(4, flat.numel())].tolist()
    return f"shape={tuple(tensor.shape)} dtype={tensor.dtype} sample={sample}"


def average(local_grad, comm_ctx, config: dict):
    """Perform tree-based gradient aggregation."""
    grad_tensor = _normalize_tensor_grad(local_grad.get("gradients"))
    
    if comm_ctx is None:
        raise ValueError("tree average requires comm_ctx from tree.setup")
    
    rank = int(comm_ctx.get("rank", config.get("rank", 0)))
    world_size = int(comm_ctx.get("world_size", config.get("world_size", 1)))
    tree_structure = comm_ctx.get("tree_structure", {})
    
    if world_size <= 1:
        return {**local_grad, "gradients": grad_tensor}
    
    log_phases = bool(config.get("tree_phase_logs", False))
    
    # Phase 1: Reduce (accumulate from leaves to root)
    accumulated = grad_tensor.clone()
    num_children_processed = 0
    
    left_child = tree_structure.get("left_child")
    right_child = tree_structure.get("right_child")
    
    # Receive from children
    if left_child is not None:
        left_endpoint = comm_ctx.get("left_child_endpoint")
        if left_endpoint is not None:
            try:
                left_grad = left_endpoint.recv()
                left_tensor = _normalize_tensor_grad(left_grad)
                accumulated.add_(left_tensor)
                num_children_processed += 1
                if log_phases:
                    print(f"[tree.average] rank={rank} reduce_from_left_child={left_child}", flush=True)
            except Exception as e:
                print(f"[tree.average] rank={rank} error receiving from left_child: {e}", flush=True)
    
    if right_child is not None:
        right_endpoint = comm_ctx.get("right_child_endpoint")
        if right_endpoint is not None:
            try:
                right_grad = right_endpoint.recv()
                right_tensor = _normalize_tensor_grad(right_grad)
                accumulated.add_(right_tensor)
                num_children_processed += 1
                if log_phases:
                    print(f"[tree.average] rank={rank} reduce_from_right_child={right_child}", flush=True)
            except Exception as e:
                print(f"[tree.average] rank={rank} error receiving from right_child: {e}", flush=True)
    
    # Send to parent
    parent = tree_structure.get("parent")
    if parent is not None:
        parent_endpoint = comm_ctx.get("parent_endpoint")
        if parent_endpoint is not None:
            try:
                parent_endpoint.send(accumulated)
                if log_phases:
                    print(f"[tree.average] rank={rank} reduce_send_to_parent={parent}", flush=True)
            except Exception as e:
                print(f"[tree.average] rank={rank} error sending to parent: {e}", flush=True)
        
        # Wait for broadcast from parent
        try:
            broadcast_data = parent_endpoint.recv()
            averaged_tensor = _normalize_tensor_grad(broadcast_data)
            if log_phases:
                print(f"[tree.average] rank={rank} broadcast_recv_from_parent={parent}", flush=True)

            if left_child is not None:
                left_endpoint = comm_ctx.get("left_child_endpoint")
                if left_endpoint is not None:
                    left_endpoint.send(averaged_tensor)

            if right_child is not None:
                right_endpoint = comm_ctx.get("right_child_endpoint")
                if right_endpoint is not None:
                    right_endpoint.send(averaged_tensor)
        except Exception as e:
            print(f"[tree.average] rank={rank} error receiving broadcast from parent: {e}", flush=True)
            averaged_tensor = accumulated / float(world_size)
    else:
        # Root node: average and broadcast to children
        averaged_tensor = accumulated / float(world_size)
        
        if left_child is not None:
            left_endpoint = comm_ctx.get("left_child_endpoint")
            if left_endpoint is not None:
                try:
                    left_endpoint.send(averaged_tensor)
                    if log_phases:
                        print(f"[tree.average] rank={rank} broadcast_send_to_left_child={left_child}", flush=True)
                except Exception as e:
                    print(f"[tree.average] rank={rank} error sending to left_child: {e}", flush=True)

        if right_child is not None:
            right_endpoint = comm_ctx.get("right_child_endpoint")
            if right_endpoint is not None:
                try:
                    right_endpoint.send(averaged_tensor)
                    if log_phases:
                        print(f"[tree.average] rank={rank} broadcast_send_to_right_child={right_child}", flush=True)
                except Exception as e:
                    print(f"[tree.average] rank={rank} error sending to right_child: {e}", flush=True)
    
    if log_phases:
        print(
            f"[tree.average] rank={rank} final_avg {_tensor_summary(averaged_tensor)}",
            flush=True,
        )

    return {
        **local_grad,
        "gradients": averaged_tensor,
    }


def teardown(comm_ctx) -> None:
    """Close all tree connections."""
    if comm_ctx is None:
        return
    
    rank = comm_ctx.get("rank", "unknown")
    
    # Close all endpoint connections
    for endpoint_key in ["parent_endpoint", "parent_conn", "left_child_endpoint", 
                          "left_child_conn", "right_child_endpoint", "right_child_conn"]:
        endpoint = comm_ctx.get(endpoint_key)
        if endpoint is None:
            continue
        try:
            endpoint.close()
        except OSError as error:
            print(f"[tree.teardown] warning: rank={rank} failed to close {endpoint_key}: {error}", flush=True)
    
    # Close listener if it exists
    listener = comm_ctx.get("_listener")
    if listener is not None:
        try:
            listener.close()
        except OSError as error:
            print(f"[tree.teardown] warning: rank={rank} failed to close listener: {error}", flush=True)
