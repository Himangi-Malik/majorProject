import torch
import torch.nn as nn

torch.manual_seed(0)
model = nn.Linear(1, 1, bias=False)

x = torch.tensor([[2.0]])   # input
target = torch.tensor([[1.0]])

output = model(x)
loss = (output - target).pow(2).mean()

print("Weight before backward:", model.weight)
print("Loss:", loss)

# Backward
loss.backward()

print("Gradient after backward:", model.weight.grad)

