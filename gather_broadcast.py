import multiprocessing as mp

def worker(rank,conn):
    print(f"Rank {rank}")
    local_val= (rank+1)*10 + 20
    conn.send(local_val)

    avg = conn.recv()
    print(f"{avg}")
    conn.close()


if __name__ == "__main__":
    num = 3
    processes = []
    parent_conns = []

    for rank in range(num):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(
            target = worker,
            args= (rank,child_conn)
        )
        p.start()
        processes.append(p)
        parent_conns.append(parent_conn)
    
    sum = 0
    for conn in parent_conns:
        sum =+ conn.recv()
    avg = sum/num
    print(f"\nMain computed global average: {avg}\n")

    for conn in parent_conns:
        conn.send(avg)

    for p in processes:
        p.join()