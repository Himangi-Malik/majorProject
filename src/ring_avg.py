import multiprocessing as mp

def worker(rank, left_conn, right_conn, world_size):
    local_value =  (rank + 1) * 10
    print(f"Rank {rank}, local value: {local_value}")

    total = local_value

    # Ring pass: circulate values world_size - 1 times
    send_value = local_value

    for _ in range(world_size - 1):
        # Send to right neighbor
        right_conn.send(send_value)

        # Receive from left neighbor
        recv_value = left_conn.recv()

        total += recv_value

        # Forward what we received in next round
        send_value = recv_value

    avg = total / world_size
    print(f"Rank {rank} final average: {avg}")

    left_conn.close()
    right_conn.close()


if __name__ == "__main__":
    world_size = 3
    processes = []

    # Create ring pipes
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

   
