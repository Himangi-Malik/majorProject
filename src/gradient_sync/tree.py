"""Tree gradient synchronization placeholder module."""


def setup(config: dict):
    raise NotImplementedError("Tree setup is not implemented yet.")


def average(local_grad, comm_ctx, config: dict):
    raise NotImplementedError("Tree average is not implemented yet.")


def teardown(comm_ctx) -> None:
    raise NotImplementedError("Tree teardown is not implemented yet.")
