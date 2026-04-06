import multiprocessing as mp
import torch
import torch.nn as nn
import os

def worker(rank, left_conn, right_conn, world_size):
    torch.manual_seed(0)

    # Simple model: y = wx
    model = nn.Linear(1, 1, bias=False)

    # Fake data (different per rank)
    x = torch.tensor([[rank + 1.0]])
    y = torch.tensor([[2.0]])

    # Forward
    output = model(x)
    loss = (output - y).pow(2).mean()

    # Backward
    loss.backward()

    print(f"Initial weight: {model.weight.data}")
    print(f"Local gradient: {model.weight.grad}")

    # ---- Ring All-Reduce on gradient ----
    grad = model.weight.grad.clone()
    total = grad.clone()
    send_tensor = grad.clone()

    for _ in range(world_size - 1):
        right_conn.send(send_tensor)
        recv_tensor = left_conn.recv()

        total += recv_tensor
        send_tensor = recv_tensor.clone()

    avg_grad = total / world_size

    # Replace local gradient with averaged gradient
    model.weight.grad = avg_grad.clone()

    print(f"Averaged gradient: {model.weight.grad}")

    left_conn.close()
    right_conn.close()


if __name__ == "__main__":
    world_size = 3
    processes = []

    pipes = [mp.Pipe() for _ in range(world_size)]

    for rank in range(world_size):
        left_conn = pipes[rank][0]
        right_conn = pipes[(rank + 1) % world_size][1]

        p = mp.Process(
            target=worker,
            args=(rank, left_conn, right_conn, world_size)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()