import torch
import torch.nn as nn

torch.manual_seed(0)

model = nn.Linear(1, 1, bias=False)

x = torch.tensor([[2.0]])
target = torch.tensor([[1.0]])

def grad_hook(grad):
    print("Hook triggered!")
    print("Gradient inside hook:", grad)

# Register hook on weight
model.weight.register_hook(grad_hook)

# Forward
output = model(x)
loss = (output - target).pow(2).mean()

# Backward
loss.backward()

print("Gradient stored in .grad:", model.weight.grad)