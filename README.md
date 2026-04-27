# Distributed Training Skeleton

This project is a minimal distributed training skeleton.

Current design:
- Main entrypoint: src/main_comm_runner.py
- Shared config file: src/config.json
- Model modules: src/models
- Gradient sync modules: src/gradient_sync

## Current behavior

- The runner loads configuration from src/config.json.
- Each machine passes only rank through CLI.
- The runner validates config and selects model and gradient modules.

## Run

From project root:

python src/main_comm_runner.py --rank 0

Use a different rank on each machine.

## Notes

- Keep src/config.json identical across machines.
- Ensure world_size matches the number of IPs in ip_list.
- Ensure each machine uses a unique rank in range 0 to world_size - 1.
