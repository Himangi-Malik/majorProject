import argparse
import pickle
import socket
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


def send_tensor(sock, tensor):
    payload = pickle.dumps(tensor.detach().cpu())
    sock.sendall(len(payload).to_bytes(4, "big") + payload)


def recv_exact(sock, num_bytes):
    data = b""
    while len(data) < num_bytes:
        chunk = sock.recv(num_bytes - len(data))
        if not chunk:
            raise ConnectionError("Socket closed while receiving data")
        data += chunk
    return data


def recv_tensor(sock):
    length = int.from_bytes(recv_exact(sock, 4), "big")
    payload = recv_exact(sock, length)
    return pickle.loads(payload)


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ring-based gradient averaging with selectable ANN/CNN/RNN training"
    )
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument(
        "--ips",
        type=str,
        required=True,
        help="Comma-separated IP list ordered by rank. Example: 192.168.1.10,192.168.1.11",
    )
    parser.add_argument("--model", type=str, default="ann", choices=["ann", "cnn", "rnn"])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--base-port", type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()

    ip_list = [ip.strip() for ip in args.ips.split(",") if ip.strip()]
    if len(ip_list) != args.world_size:
        raise ValueError(
            f"Expected {args.world_size} IPs in --ips, got {len(ip_list)}: {ip_list}"
        )

    torch.manual_seed(42 + args.rank)
    model, criterion = build_model(args.model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    left_sock, right_sock, server = setup_ring_connections(
        args.rank, args.world_size, ip_list, args.base_port
    )

    try:
        for step in range(1, args.steps + 1):
            x, y = create_batch(args.model, args.rank, args.batch_size)
            loss, local_grad, fwd, bwd = train_step(model, optimizer, criterion, x, y)

            comm_t0 = time.time()
            avg_grad = ring_average_gradients(
                local_grad, left_sock, right_sock, args.world_size
            )
            comm = time.time() - comm_t0

            set_flattened_gradients(model, avg_grad)
            optimizer.step()

            print(
                f"[Rank {args.rank}] step={step} loss={loss:.6f} "
                f"fwd={fwd:.6f}s bwd={bwd:.6f}s comm={comm:.6f}s "
                f"local_norm={local_grad.norm().item():.6f} "
                f"avg_norm={avg_grad.norm().item():.6f}"
            )
    finally:
        left_sock.close()
        right_sock.close()
        server.close()


if __name__ == "__main__":
    main()
