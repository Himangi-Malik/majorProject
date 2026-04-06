import multiprocessing as mp
import torch
import os

def worker(rank, left_conn, right_conn, world_size):
    # Each process creates its own tensor
    tensor = torch.tensor([rank + 1.0])
    print(f"Rank {rank}, PID {os.getpid()}, initial tensor: {tensor}")

    total = tensor.clone()
    send_tensor = tensor.clone()

    for _ in range(world_size - 1):
        # Send tensor
        right_conn.send(send_tensor)

        # Receive tensor
        recv_tensor = left_conn.recv()

        total += recv_tensor
        send_tensor = recv_tensor.clone()

    avg = total / world_size

    print(f"Rank {rank}, final averaged tensor: {avg}")

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