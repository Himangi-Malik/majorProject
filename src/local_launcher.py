import multiprocessing as mp

from worker_runner import run_worker

def build_local_topology(algo, pipes, rank, world_size):
    if algo == "ring":
        return {
            # receive from left neighbor (r-1 → r)
            "left_conn": pipes[(rank - 1) % world_size][0],

            # send to right neighbor (r → r+1)
            "right_conn": pipes[rank][1],

            "left_endpoint_info": {
                "peer_rank": (rank - 1) % world_size,
                "direction": "left",
                "transport": "pipe",
            },
            "right_endpoint_info": {
                "peer_rank": (rank + 1) % world_size,
                "direction": "right",
                "transport": "pipe",
            },
        }

    if algo == "tree":
        return {}

    if algo == "parameter_server":
        return {}

    raise ValueError("Unknown algo")


def launch_local(config):
    world_size = config["world_size"]
    algo = config["algo"]

    pipes = [mp.Pipe() for _ in range(world_size)]
    processes = []

    for rank in range(world_size):
        topo = build_local_topology(algo, pipes, rank, world_size)
        print(
            f"rank {rank} left={topo.get('left_endpoint_info')}, right={topo.get('right_endpoint_info')}",
            flush=True,
        )

        worker_config = {
            **config,
            "rank": rank,
            **topo,
        }

        process = mp.Process(target=run_worker, args=(worker_config,))
        process.start()
        processes.append(process)

    for left_conn, right_conn in pipes:
        left_conn.close()
        right_conn.close()

    print("[parent rank 0] waiting for child processes", flush=True)

    for process in processes:
        process.join()

    print("[parent rank 0] local launch complete", flush=True)
