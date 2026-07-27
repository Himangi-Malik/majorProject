"""Medium ANN model module for distributed training benchmarks."""
import torch
import torch.nn as nn
import torch.optim as optim


class SimpleANN(nn.Module):
    """Medium fully connected network for Fashion-MNIST benchmark runs."""

    def __init__(self, input_dim: int = 28 * 28, hidden_dim: int = 128, output_dim: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def _flatten_gradients(model):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.detach().clone().flatten())
    return torch.cat(grads) if grads else torch.tensor([0.0], dtype=torch.float32)


def build_model(config: dict):
    """Build and initialize medium ANN model state."""
    # CRITICAL: Set fixed seed on ALL ranks before model creation to ensure
    # identical parameter initialization. This is required for valid distributed SGD.
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleANN(
        input_dim=int(config.get("ann_input_dim", 28 * 28)),
        hidden_dim=int(config.get("ann_hidden_dim", 128)),
        output_dim=int(config.get("ann_output_dim", 10)),
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=float(config.get("lr", 0.01)))
    criterion = nn.CrossEntropyLoss()

    return {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "device": device,
        "lr": float(config.get("lr", 0.01)),
    }


def train_step(state, batch, config: dict):
    """Execute one Fashion-MNIST training step """
    model = state["model"]
    criterion = state["criterion"]
    device = state["device"]

    x, y = batch

    x = x.view(x.size(0), -1).to(device)
    y = y.to(device)

    model.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    grad_vector = _flatten_gradients(model)

    return {
        "rank": config.get("rank", 0),
        "gradients": grad_vector,
        "loss": float(loss.detach()),
    }


def apply_synced_gradients(state, averaged_grad):
    """Write averaged gradients back into the model and step the optimizer."""
    model = state["model"]
    optimizer = state["optimizer"]

    pointer = 0
    for param in model.parameters():
        numel = param.numel()
        param.grad = averaged_grad[pointer:pointer + numel].view_as(param)
        pointer += numel

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
