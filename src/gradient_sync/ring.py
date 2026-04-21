"""Ring gradient synchronization placeholder module.
    setup = open/init resources
    average = do communication work
    teardown = close/free resources"""


def setup(config: dict):
    raise NotImplementedError("Ring setup is not implemented yet.")


def average(local_grad, comm_ctx, config: dict):
    raise NotImplementedError("Ring average is not implemented yet.")


def teardown(comm_ctx) -> None:
    raise NotImplementedError("Ring teardown is not implemented yet.")
