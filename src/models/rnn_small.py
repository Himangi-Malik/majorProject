"""Small RNN model module for distributed training."""
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    """Small LSTM classifier for FashionMNIST."""

    def __init__(
        self,
        input_size=28,
        hidden_size=16,
        output_size=10,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, 28, 28)
        _, (hidden, _) = self.lstm(x)

        # hidden shape: (1, batch_size, hidden_size)
        x = hidden.squeeze(0)

        x = self.fc(x)

        return x

def _flatten_gradients(model):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.detach().clone().flatten())
    return torch.cat(grads) if grads else torch.tensor([0.0], dtype=torch.float32)

def build_model(config: dict):
    """Build and initialize small RNN model."""
    # CRITICAL: Set fixed seed on ALL ranks before model creation to ensure
    # identical parameter initialization. This is required for valid distributed SGD.
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleRNN(
        input_size=28,
        hidden_size=int(config.get("rnn_hidden_size", 16)),
        output_size=10,
    ).to(device)

    lr = float(config.get("lr", 0.01))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    return {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "lr": lr,
        "device": device,
    }


def train_step(state, batch, config: dict):
    """Execute one training step """
    model = state["model"]
    criterion = state["criterion"]
    device = state["device"]

    x, y = batch

    # Remove channel dimension:
    # (batch, 1, 28, 28) -> (batch, 28, 28)
    x = x.squeeze(1).to(device)
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
