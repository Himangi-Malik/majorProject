"""Parameter-server gradient synchronization placeholder module."""


def setup(config: dict):
    raise NotImplementedError("Parameter-server setup is not implemented yet.")


def average(local_grad, comm_ctx, config: dict):
    raise NotImplementedError("Parameter-server average is not implemented yet.")


def teardown(comm_ctx) -> None:
    raise NotImplementedError("Parameter-server teardown is not implemented yet.")
