"""Minimal ANN model module for the first implementation slice."""
"""for chahat, have not verified the implementation yet,  please check."""
import torch

def build_model(config: dict):
    # Tiny 1D linear model represented as tensors.
    # This state object is passed into train_step and mutated each call.
    return {
        "w": torch.tensor(0.0, dtype=torch.float32),
        "b": torch.tensor(0.0, dtype=torch.float32),
        "lr": float(config["lr"]),
    }


def train_step(model, config: dict):
    # One synthetic sample: x=1.0, y=1.0.
    x = torch.tensor(1.0, dtype=torch.float32)
    y = torch.tensor(1.0, dtype=torch.float32)

    # Forward pass from current model state.
    w = model["w"]
    b = model["b"]
    pred = w * x + b
    error = pred - y

    # Backward pass for the tiny linear model.
    grad_w = error * x
    grad_b = error
    grad_vector = torch.stack([grad_w, grad_b])

    # Local SGD update; runner will send gradients to sync algo separately.
    lr = model.get("lr", float(config["lr"]))
    model["w"] = w - lr * grad_w
    model["b"] = b - lr * grad_b

    # Unified payload contract consumed by the runner/algo modules.
    return {
        "rank": config.get("rank", 0),
        "gradients": grad_vector,
        "loss": float(0.5 * (error ** 2)),
    }
