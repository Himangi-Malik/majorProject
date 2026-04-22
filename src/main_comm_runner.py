import argparse
import json

from gradient_sync import parameter_server, ring, tree
from models import ann_model, cnn_model, rnn_model

CONFIG_PATH = "config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal tag-driven communication runner")
    parser.add_argument("--rank", type=int, required=True, help="Rank id")
    return parser.parse_args()


def load_json_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_runtime_config(args: argparse.Namespace) -> dict:
    config = load_json_config(CONFIG_PATH)
    config["rank"] = args.rank
    return config


def validate_config(config: dict) -> None:
    required_keys = ["mode", "algo", "model", "lr", "world_size", "ip_list"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")

    if config["mode"] not in {"local", "worker"}:
        raise ValueError("mode must be one of: local, worker")
    if config["algo"] not in {"ring", "tree", "parameter_server"}:
        raise ValueError("algo must be one of: ring, tree, parameter_server")
    if config["model"] not in {"ann", "cnn", "rnn"}:
        raise ValueError("model must be one of: ann, cnn, rnn")

    if config["world_size"] != len(config["ip_list"]):
        raise ValueError("world_size must match number of values in ip_list")

    if config["mode"] == "worker":
        if "rank" not in config:
            raise ValueError("worker mode requires rank")
        if not 0 <= config["rank"] < config["world_size"]:
            raise ValueError("rank must be in range [0, world_size - 1]")


def get_algo_module(algo_tag: str):
    algo_modules = {
        "ring": ring,
        "tree": tree,
        "parameter_server": parameter_server,
    }
    return algo_modules[algo_tag]


def get_model_module(model_tag: str):
    model_modules = {
        "ann": ann_model,
        "cnn": cnn_model,
        "rnn": rnn_model,
    }
    return model_modules[model_tag]


def main() -> None:
    args = parse_args()
    config = build_runtime_config(args)
    validate_config(config)

    algo_module = get_algo_module(config["algo"])
    algo_module.setup(config)
    model_module = get_model_module(config["model"])

    print("Final runtime config:")
    print(json.dumps(config, indent=2))
    print(f"Selected algorithm module: {algo_module.__name__}")
    print(f"Selected model module: {model_module.__name__}")
    print(f"Execution mode: {config['mode']}")

    if config["mode"] == "local":
        print("Local mode selected; internal os pipelines will be tested without actual communication.")
        return

if __name__ == "__main__":
    main()
