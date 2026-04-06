import multiprocessing as mp
import torch
import torch.nn as nn

def worker(rank, left_conn, right_conn, world_size):
    torch.manual_seed(0)

    # Simple model: y = wx
    model = nn.Linear(1, 1, bias=False)

    # Fake data (different per rank)
    x = torch.tensor([[rank + 1.0]])
    y = torch.tensor([[2.0]])

    print(f"[Rank {rank}] Initial weight: {model.weight}")

    # ---- Define ring all-reduce ----
    def ring_all_reduce(grad):
        total = grad.clone()
        send_tensor = grad.clone()

        for _ in range(world_size - 1):
            right_conn.send(send_tensor)
            recv_tensor = left_conn.recv()

            total += recv_tensor
            send_tensor = recv_tensor.clone()

        avg = total / world_size
        return avg

    # ---- Register hook on parameter ----
    def gradient_hook(grad):
        print(f"[Rank {rank}] Local gradient inside hook: {grad}")

        avg_grad = ring_all_reduce(grad)

        print(f"[Rank {rank}] Averaged gradient inside hook: {avg_grad}")

        return avg_grad  # This replaces param.grad automatically

    model.weight.register_hook(gradient_hook)

    # Forward
    output = model(x)
    loss = (output - y).pow(2).mean()

    # Backward triggers hook automatically
    loss.backward()

    # Now gradient is already averaged
    print(f"[Rank {rank}] Final stored gradient: {model.weight.grad}")

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