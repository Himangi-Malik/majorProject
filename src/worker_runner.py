from gradient_sync import parameter_server, ring, tree
from models import ann_model, cnn_model, rnn_model
import torch


def get_algo_module(algo_tag: str):
    return {
        "ring": ring,
        "tree": tree,
        "parameter_server": parameter_server,
    }[algo_tag]


def get_model_module(model_tag: str):
    return {
        "ann": ann_model,
        "cnn": cnn_model,
        "rnn": rnn_model,
    }[model_tag]


def _summarize_grad(grad_obj: dict) -> str:
    grad_tensor = grad_obj.get("gradients")
    if isinstance(grad_tensor, torch.Tensor):
        flat = grad_tensor.detach().flatten()
        sample = flat[: min(4, flat.numel())].tolist()
        return f"shape={tuple(grad_tensor.shape)} dtype={grad_tensor.dtype} sample={sample}"
    return f"type={type(grad_tensor).__name__}"


def run_worker(config):
    algo_module = get_algo_module(config["algo"])
    model_module = get_model_module(config["model"])

    print(f"[rank {config['rank']}] start mode={config['mode']} algo={config['algo']}", flush=True)

    comm_ctx = None

    try:
        # Step 1: communication setup first.
        comm_ctx = algo_module.setup(config)

        # Step 2: build model.
        try:
            model = model_module.build_model(config)
        except NotImplementedError as error:
            print(f"[rank {config['rank']}] warning: build_model placeholder hit: {error}", flush=True)
            model = {"model_tag": config["model"], "placeholder": True}

        # Step 3: run one tiny train step to get gradients.
        try:
            local_grad = model_module.train_step(model, config)
        except NotImplementedError as error:
            print(f"[rank {config['rank']}] warning: train_step placeholder hit: {error}", flush=True)
            local_grad = {"rank": config["rank"], "gradients": torch.tensor([0.0], dtype=torch.float32)}

        # Step 4: pass gradients to algorithm module.
        try:
            synced_grad = algo_module.average(local_grad, comm_ctx, config)
        except NotImplementedError as error:
            print(f"[rank {config['rank']}] warning: average placeholder hit: {error}", flush=True)
            synced_grad = local_grad

        loss = synced_grad.get("loss") if isinstance(synced_grad, dict) else None
        print(f"[rank {config['rank']}] gradients {_summarize_grad(synced_grad)} loss={loss}", flush=True)
    finally:
        # Step 5: teardown last.
        algo_module.teardown(comm_ctx)
        print(f"[rank {config['rank']}] teardown done", flush=True)
