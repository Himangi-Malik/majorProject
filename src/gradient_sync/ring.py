""" Ring gradient synchronization placeholder module.
    setup = open/init resources
    average = do communication work
    teardown = close/free resources """


def _setup_worker(config: dict) -> dict:
    return {
        "mode": "worker",
        "rank": config["rank"],
        "world_size": config["world_size"],
        "transport": "socket",
    }


def setup(config: dict):
    mode = config["mode"]

    print(f"[ring.setup] mode={mode}, rank={config.get('rank')}, world_size={config.get('world_size')}", flush=True)
    if mode == "worker":
        return _setup_worker(config)

    #raise ValueError("mode must be one of: local, worker")


def average(local_grad, comm_ctx, config: dict):
    raise NotImplementedError("Ring average is not implemented yet.")


def teardown(comm_ctx) -> None:
    if comm_ctx is None:
        return

    for key in ("left_conn", "right_conn"):
        conn = comm_ctx.get(key)
        if conn is None:
            continue
        try:
            conn.close()
        except OSError:
            pass