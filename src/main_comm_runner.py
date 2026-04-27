import argparse
import json

from dist_launcher import launch_distributed
from local_launcher import launch_local
from worker_runner import get_algo_module, get_model_module, run_worker

CONFIG_PATH = "config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal tag-driven communication runner")
    parser.add_argument("--rank", type=int, required=True, help="Rank id")
    return parser.parse_args()


def load_json_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def build_runtime_config(args: argparse.Namespace) -> dict:
    config = load_json_config(CONFIG_PATH)
    config["rank"] = args.rank
    return config


def validate_config(config: dict) -> None:
    required_keys = ["mode", "algo", "model", "lr", "world_size", "ip_list"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")

    if config["mode"] not in {"local", "distributed"}:
        raise ValueError("mode must be one of: local, distributed")

    if config["algo"] not in {"ring", "tree", "parameter_server"}:
        raise ValueError("algo must be one of: ring, tree, parameter_server")

    if config["model"] not in {"ann", "cnn", "rnn"}:
        raise ValueError("model must be one of: ann, cnn, rnn")

    if config["world_size"] != len(config["ip_list"]):
        raise ValueError("world_size must match number of values in ip_list")

    if config["mode"] == "distributed":
        if not 0 <= config["rank"] < config["world_size"]:
            raise ValueError("rank must be in range [0, world_size - 1]")


def main() -> None:
    args = parse_args()
    config = build_runtime_config(args)
    validate_config(config)

    algo_module = get_algo_module(config["algo"])
    model_module = get_model_module(config["model"])

    print(
        f"[runner] mode={config['mode']} algo={config['algo']} model={config['model']} "
        f"world_size={config['world_size']} rank={config['rank']}",
        flush=True,
    )
    print(f"[runner] selected algorithm module: {algo_module.__name__}", flush=True)
    print(f"[runner] selected model module: {model_module.__name__}", flush=True)

    if config["mode"] == "local":
        if config["rank"] == 0:
            launch_local(config)
        return
        
    if config["mode"] == "distributed":
        launch_distributed(config)
        return

    run_worker(config)


if __name__ == "__main__":
    main()