import multiprocessing as mp

def worker(rank):
    print(f"I am process {rank}")

if __name__ == "__main__":
    processes = []
    for rank in range(2):
        p = mp.Process(target=worker, args=(rank,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
