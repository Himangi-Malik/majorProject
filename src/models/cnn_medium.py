"""Medium CNN model module for distributed training."""
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Medium CNN for benchmark purposes."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def _flatten_gradients(model):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.detach().clone().flatten())
    return torch.cat(grads) if grads else torch.tensor([0.0], dtype=torch.float32)

def build_model(config: dict):
    """Build and initialize medium CNN model."""
    # CRITICAL: Set fixed seed on ALL ranks before model creation to ensure
    # identical parameter initialization. This is required for valid distributed SGD.
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    lr = float(config.get("lr", 0.01))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    return {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "device": device,
        "lr": lr,
    }


def train_step(state, batch, config: dict):
    """Execute one training step """
    model = state["model"]
    criterion = state["criterion"]
    device = state["device"]

    x, y = batch
    x = x.to(device)
    y = y.to(device)

    # Forward pass
    logits = model(x)
    loss = criterion(logits, y)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Collect gradients into a single vector
    grad_vector = _flatten_gradients(model)

    return {
        "rank": config.get("rank", 0),
        "gradients": grad_vector,
        "loss": float(loss.detach()),
    }


def apply_synced_gradients(state, averaged_grad):
    """Apply synchronized averaged gradients back into the model."""
    model = state["model"]
    optimizer = state["optimizer"]

    pointer = 0
    for param in model.parameters():
        numel = param.numel()
        param.grad = averaged_grad[pointer:pointer + numel].view_as(param)
        pointer += numel

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
