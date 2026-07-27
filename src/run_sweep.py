"""
Benchmark sweep controller.

Responsibilities:
1. Define the benchmark search space.
2. Generate every experiment configuration.
3. (Later) Launch distributed workers.
4. (Later) Collect benchmark results.
"""

from itertools import product
from pathlib import Path
import json
import subprocess
import sys

from metrics.figures import generate_all_figures
from metrics.rank_metrics import RankMetrics
from metrics.summary import SummaryGenerator
from metrics.tables import generate_all_tables

# ==========================================================
# Benchmark Search Space
# ==========================================================

ALGORITHMS = [
    "ring",
]

MODELS = [
    "ann",
]

MODEL_SIZES = [
    "small",
    "medium",
    "large",
]

WORLD_SIZES = [
    8
]

BATCH_SIZES = [
    32
]


# ==========================================================
# Default Configuration
# ==========================================================
EPOCHS = 5
STEPS_PER_EPOCH = 10
LEARNING_RATE = 0.001

DATASET = "fashion_mnist"

BASE_PORT = 5000
CONNECT_TIMEOUT = 10.0

RESULTS_DIR = "benchmark_results"

# ==========================================================
# Experiment Generator
# ==========================================================

def generate_experiments():
    """
    Generate every benchmark configuration.

    Yields:
        dict: One experiment configuration.
    """

    for algo, model, model_size, world_size, batch_size in product(
        ALGORITHMS,
        MODELS,
        MODEL_SIZES,
        WORLD_SIZES,
        BATCH_SIZES,
    ):

        yield {

            "algo": algo,
            "model": model,
            "model_size": model_size,

            "world_size": world_size,
            "batch_size": batch_size,

            "epochs": EPOCHS,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "lr": LEARNING_RATE,

            "dataset": DATASET,

            "base_port": BASE_PORT,
            "connect_timeout": CONNECT_TIMEOUT,
            "ip_list": ["127.0.0.1"] * world_size,
        }


def prepare_experiment(experiment_number, config):

    experiment_name = f"experiment_{experiment_number:03d}"

    experiment_dir = (
        Path(RESULTS_DIR)
        / experiment_name
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    config["benchmark_results_dir"] = str(experiment_dir)
    config_path = experiment_dir / "config.json"

    with open(config_path, "w", encoding="utf-8") as file_handle:
        json.dump(
            config,
            file_handle,
            indent=4,
        )

    return experiment_dir

def run_experiment(experiment_dir: Path):
    """
    Run one benchmark experiment.

    Returns:
        bool: True if all workers completed successfully.
    """

    config_path = experiment_dir / "config.json"

    with open(config_path, "r", encoding="utf-8") as file_handle:
        config = json.load(file_handle)

    world_size = config["world_size"]

    print(f"\nRunning {experiment_dir.name}")
    print(f"Workers: {world_size}")

    processes = []

    # --------------------------------------------------
    # Launch workers
    # --------------------------------------------------

    for rank in range(world_size):

        cmd = [
            sys.executable,
            "benchmark_worker.py",
            "--config",
            str(config_path),
            "--rank",
            str(rank),
        ]

        proc = subprocess.Popen(cmd)

        processes.append((rank, proc))

    # --------------------------------------------------
    # Wait for completion
    # --------------------------------------------------

    failed_ranks = []

    for rank, proc in processes:

        try:
            proc.wait(timeout=3600)

            if proc.returncode != 0:
                failed_ranks.append(rank)

        except subprocess.TimeoutExpired:

            proc.kill()
            failed_ranks.append(rank)

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    if failed_ranks:

        print(
            f"{experiment_dir.name} failed "
            f"(ranks: {failed_ranks})"
        )

        return False

    print(f"{experiment_dir.name} completed successfully")

    return True
# ==========================================================
# Main
# ==========================================================

def main():

    print("Running benchmark sweep...\n")

    total = 0
    successful = 0

    for total, experiment in enumerate(
        generate_experiments(),
        start=1,
    ):

        experiment_dir = prepare_experiment(
            total,
            experiment,
        )

        if run_experiment(experiment_dir):
            config_path = experiment_dir / "config.json"

            with open(config_path, "r") as file_handle:
                    config = json.load(file_handle)

            rank_metrics = []

            for rank in range(config["world_size"]):

                metrics_path = (
                    experiment_dir
                    / f"rank_{rank}_metrics.json"
                )

                rank_metrics.append(
                    RankMetrics.load(metrics_path)
                )

            summary = SummaryGenerator(rank_metrics)

            summary.save_summary(
                experiment_dir / "summary.json"
            )

            successful += 1 

    print("\n===================================")
    print(f"Experiments : {total}")
    print(f"Succeeded  : {successful}")
    print(f"Failed     : {total - successful}")
    print("===================================")

    print("\nGenerating paper tables and figures...")
    generate_all_tables()
    generate_all_figures()
    print("Done.")

if __name__ == "__main__":
    main()