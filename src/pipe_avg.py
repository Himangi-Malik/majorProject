import multiprocessing as mp
import os

def worker(rank, conn, local_value):
    print(f"Process {rank} started with value {local_value}")
    print(f"Rank {rank} PID: {os.getpid()}")

    if rank == 0:
        # Send first
        conn.send(local_value)
        other_value = conn.recv()
    else:
        # Receive first
        other_value = conn.recv()
        conn.send(local_value)

    avg = (local_value + other_value) / 2
    print(f"Process {rank} computed average: {avg}")

    conn.close()


if __name__ == "__main__":
    # Create a communication pipe
    conn1, conn2 = mp.Pipe()

    # Assign different values to each process
    p0 = mp.Process(target=worker, args=(0, conn1, 10))
    p1 = mp.Process(target=worker, args=(1, conn2, 20))

    p0.start()
    p1.start()

    p0.join()
    p1.join()
