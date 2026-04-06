import torch
import torch.nn as nn
import torch.optim as optim

# 1. Model (torch.nn)
model = nn.Linear(1, 1)

# 2. Loss function
loss_fn = nn.MSELoss()

# 3. Optimizer (torch.optim)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training data
x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[3.0], [5.0], [7.0]])

# Training loop
for epoch in range(100):
    # Forward pass
    y_pred = model(x)
    print(y_pred)
    # Compute loss
    loss = loss_fn(y_pred, y)

    # Backward pass (torch.autograd)
    optimizer.zero_grad()   # clear old gradients
    loss.backward()         # compute new gradients

    # Update weights
    optimizer.step()
print (y)
