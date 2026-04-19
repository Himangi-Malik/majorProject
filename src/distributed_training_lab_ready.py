import argparse
import pickle
import socket
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

class ANNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.fc(x)


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 26 * 26, 10),
        )

    def forward(self, x):
        return self.net(x)


class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=8, hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)


def send_bytes(sock, payload):
    sock.sendall(len(payload).to_bytes(4, "big") + payload)


def recv_exact(sock, num_bytes):
    data = b""
    while len(data) < num_bytes:
        chunk = sock.recv(num_bytes - len(data))
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        data += chunk
    return data


def recv_bytes(sock):
    length = int.from_bytes(recv_exact(sock, 4), "big")
    return recv_exact(sock, length)


def send_obj(sock, obj):
    send_bytes(sock, pickle.dumps(obj))


def recv_obj(sock):
    return pickle.loads(recv_bytes(sock))


def send_tensor(sock, tensor):
    send_obj(sock, tensor.detach().cpu())


def recv_tensor(sock):
    return recv_obj(sock)


def connect_with_retry(sock, host, port, retry_delay=0.1, max_wait=10.0):
    deadline = time.monotonic() + max_wait
    while True:
        try:
            sock.connect((host, port))
            return
        except (ConnectionRefusedError, OSError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(retry_delay)


def setup_ring_connections(rank, world_size, ip_list, base_port):
    right_rank = (rank + 1) % world_size

    my_port = base_port + rank
    right_port = base_port + right_rank

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", my_port))
    server.listen(1)

    right_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connect_with_retry(right_sock, ip_list[right_rank], right_port)

    left_sock, left_addr = server.accept()
    print(f"[Rank {rank}] left neighbor connected from {left_addr}")
    return left_sock, right_sock, server


def build_model(model_type):
    if model_type == "ann":
        return ANNModel(), nn.MSELoss()
    if model_type == "cnn":
        return CNNModel(), nn.CrossEntropyLoss()
    if model_type == "rnn":
        return RNNModel(), nn.MSELoss()
    raise ValueError(f"Unknown model type: {model_type}")


def create_batch(model_type, rank, batch_size):
    torch.manual_seed(1000 + rank)

    if model_type == "ann":
        x = torch.randn(batch_size, 10)
        y = torch.randn(batch_size, 1)
        return x, y

    if model_type == "cnn":
        x = torch.randn(batch_size, 1, 28, 28)
        y = torch.randint(0, 10, (batch_size,))
        return x, y

    if model_type == "rnn":
        x = torch.randn(batch_size, 5, 8)
        y = torch.randn(batch_size, 1)
        return x, y

    raise ValueError(f"Unknown model type: {model_type}")


def flatten_gradients(model):
    return torch.cat([p.grad.view(-1) for p in model.parameters()])


def set_flattened_gradients(model, flat_grad):
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param.grad = flat_grad[offset : offset + numel].view_as(param).clone()
        offset += numel


def train_step(model, optimizer, criterion, x, y):
    t0 = time.time()

    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)

    fwd_time = time.time() - t0

    t1 = time.time()
    loss.backward()
    bwd_time = time.time() - t1

    grads = flatten_gradients(model)
    return loss.item(), grads, fwd_time, bwd_time


def ring_average_gradients(local_grad, left_sock, right_sock, world_size):
    total = local_grad.clone()
    send_buf = local_grad.clone()

    for _ in range(world_size - 1):
        send_tensor(right_sock, send_buf)
        recv_buf = recv_tensor(left_sock)
        total += recv_buf
        send_buf = recv_buf

    return total / world_size


def get_local_ip():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return probe.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        probe.close()


def run_training(config):
    rank = config["rank"]
    world_size = config["world_size"]
    ip_list = config["ip_list"]
    model_type = config["model"]
    steps = config["steps"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    base_port = config["base_port"]

    torch.manual_seed(42 + rank)
    model, criterion = build_model(model_type)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    left_sock, right_sock, server = setup_ring_connections(rank, world_size, ip_list, base_port)

    try:
        for step in range(1, steps + 1):
            x, y = create_batch(model_type, rank, batch_size)
            loss, local_grad, fwd, bwd = train_step(model, optimizer, criterion, x, y)

            comm_t0 = time.time()
            avg_grad = ring_average_gradients(local_grad, left_sock, right_sock, world_size)
            comm = time.time() - comm_t0

            set_flattened_gradients(model, avg_grad)
            optimizer.step()

            print(
                f"[Rank {rank}] step={step} loss={loss:.6f} "
                f"fwd={fwd:.6f}s bwd={bwd:.6f}s comm={comm:.6f}s "
                f"local_norm={local_grad.norm().item():.6f} "
                f"avg_norm={avg_grad.norm().item():.6f}"
            )
    finally:
        left_sock.close()
        right_sock.close()
        server.close()


def run_coordinator(args):
    if args.world_size < 1:
        raise ValueError("--world-size must be >= 1")

    host_ip = get_local_ip()
    required_workers = args.world_size - 1
    session_id = str(int(time.time()))

    control_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    control_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    control_server.bind(("0.0.0.0", args.control_port))
    control_server.listen(required_workers if required_workers > 0 else 1)
    control_server.settimeout(0.4)

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    discovered = {}
    try:
        print(
            f"[Coordinator] local_ip={host_ip} waiting for {required_workers} workers "
            f"(session={session_id})"
        )

        deadline = time.monotonic() + args.discovery_timeout
        while len(discovered) < required_workers:
            announce = {
                "kind": "discover",
                "session_id": session_id,
                "world_size": args.world_size,
                "control_port": args.control_port,
            }
            udp.sendto(pickle.dumps(announce), ("255.255.255.255", args.discovery_port))

            interval_deadline = time.monotonic() + args.discovery_interval
            while time.monotonic() < interval_deadline and len(discovered) < required_workers:
                try:
                    conn, addr = control_server.accept()
                except socket.timeout:
                    continue

                worker_ip = addr[0]
                try:
                    hello = recv_obj(conn)
                    if hello.get("kind") != "join":
                        conn.close()
                        continue
                except Exception:
                    conn.close()
                    continue

                if worker_ip in discovered:
                    conn.close()
                    continue

                discovered[worker_ip] = conn
                print(f"[Coordinator] worker joined: {worker_ip}")

            if time.monotonic() > deadline and len(discovered) < required_workers:
                raise TimeoutError(
                    f"Discovery timeout: needed={required_workers}, found={len(discovered)}"
                )

        worker_ips = sorted(discovered.keys())
        ip_list = [host_ip] + worker_ips
        print(f"[Coordinator] ring order: {ip_list}")

        for idx, worker_ip in enumerate(worker_ips, start=1):
            payload = {
                "kind": "start",
                "rank": idx,
                "world_size": args.world_size,
                "ip_list": ip_list,
                "model": args.model,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "base_port": args.base_port,
            }
            send_obj(discovered[worker_ip], payload)
            discovered[worker_ip].close()

        local_config = {
            "rank": 0,
            "world_size": args.world_size,
            "ip_list": ip_list,
            "model": args.model,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "base_port": args.base_port,
        }
        run_training(local_config)
    finally:
        for conn in discovered.values():
            try:
                conn.close()
            except OSError:
                pass
        udp.close()
        control_server.close()


def run_worker(args):
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp.bind(("0.0.0.0", args.discovery_port))

    print(f"[Worker] listening for coordinator broadcast on UDP {args.discovery_port}")

    try:
        while True:
            data, addr = udp.recvfrom(4096)
            try:
                msg = pickle.loads(data)
            except Exception:
                continue

            if msg.get("kind") != "discover":
                continue

            control_port = int(msg["control_port"])
            coordinator_ip = addr[0]
            print(f"[Worker] discovered coordinator at {coordinator_ip}:{control_port}")

            control = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                connect_with_retry(
                    control,
                    coordinator_ip,
                    control_port,
                    retry_delay=0.2,
                    max_wait=args.discovery_timeout,
                )
                send_obj(control, {"kind": "join", "hostname": socket.gethostname()})
                start_payload = recv_obj(control)
            finally:
                control.close()

            if start_payload.get("kind") != "start":
                continue

            run_training(start_payload)
            break
    finally:
        udp.close()


def run_local_processes(args):
    if args.world_size is None or args.world_size < 1:
        raise ValueError("--world-size must be >= 1 in local mode")

    ip_list = ["127.0.0.1"] * args.world_size
    ips_csv = ",".join(ip_list)
    script_path = __file__

    children = []
    try:
        for rank in range(1, args.world_size):
            cmd = [
                sys.executable,
                script_path,
                "--mode",
                "manual",
                "--rank",
                str(rank),
                "--world-size",
                str(args.world_size),
                "--ips",
                ips_csv,
                "--model",
                args.model,
                "--steps",
                str(args.steps),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--base-port",
                str(args.base_port),
            ]
            children.append(subprocess.Popen(cmd))

        local_config = {
            "rank": 0,
            "world_size": args.world_size,
            "ip_list": ip_list,
            "model": args.model,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "base_port": args.base_port,
        }
        run_training(local_config)

        failures = []
        for idx, child in enumerate(children, start=1):
            code = child.wait()
            if code != 0:
                failures.append((idx, code))

        if failures:
            raise RuntimeError(f"Child ranks failed: {failures}")
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lab-ready distributed training: local process test + LAN auto-discovery"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["local", "coordinator", "worker", "manual"],
        help="local: single-PC process test, coordinator/worker: LAN auto mode, manual: explicit rank+ips",
    )

    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--ips", type=str, default="")

    parser.add_argument("--model", type=str, default="ann", choices=["ann", "cnn", "rnn"])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--base-port", type=int, default=5000)

    parser.add_argument("--discovery-port", type=int, default=39000)
    parser.add_argument("--control-port", type=int, default=39001)
    parser.add_argument("--discovery-timeout", type=float, default=60.0)
    parser.add_argument("--discovery-interval", type=float, default=1.0)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "local":
        run_local_processes(args)
        return

    if args.mode == "coordinator":
        if args.world_size is None:
            raise ValueError("--world-size is required in coordinator mode")
        run_coordinator(args)
        return

    if args.mode == "worker":
        run_worker(args)
        return

    if args.rank is None or args.world_size is None:
        raise ValueError("--rank and --world-size are required in manual mode")

    ip_list = [ip.strip() for ip in args.ips.split(",") if ip.strip()]
    if len(ip_list) != args.world_size:
        raise ValueError(
            f"Expected {args.world_size} IPs in --ips, got {len(ip_list)}: {ip_list}"
        )

    config = {
        "rank": args.rank,
        "world_size": args.world_size,
        "ip_list": ip_list,
        "model": args.model,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "base_port": args.base_port,
    }
    run_training(config)


if __name__ == "__main__":
    main()
