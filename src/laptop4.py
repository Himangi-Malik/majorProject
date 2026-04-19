import torch
import torch.nn as nn
import socket
import pickle
import sys
import time


def send_tensor(sock, tensor):
    data = pickle.dumps(tensor)
    sock.sendall(len(data).to_bytes(4, 'big') + data)


def recv_tensor(sock):
    length = int.from_bytes(sock.recv(4), 'big')
    data = b''
    while len(data) < length:
        chunk = sock.recv(min(4096, length - len(data)))
        data += chunk
    return pickle.loads(data)


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


def worker(rank, world_size, ip_list, base_port=5000):
    torch.manual_seed(0)

    #left_rank = (rank - 1 + world_size) % world_size
    right_rank = (rank + 1) % world_size

    my_port = base_port + rank
    right_port = base_port + right_rank

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", my_port))
    server.listen(1)

    right_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    right_ip = ip_list[right_rank]
    connect_with_retry(right_sock, right_ip, right_port)

    left_sock, _ = server.accept()

    model = nn.Linear(1, 1, bias=False)

    x = torch.tensor([[rank + 1.0]])
    y = torch.tensor([[2.0]])

    output = model(x)
    loss = (output - y).pow(2).mean()

    loss.backward()

    grad = model.weight.grad.clone()
    total = grad.clone()
    send_data = grad.clone()

    for _ in range(world_size - 1):
        send_tensor(right_sock, send_data)

        recv_data = recv_tensor(left_sock)

        total += recv_data
        send_data = recv_data.clone()

    avg_grad = total / world_size
    model.weight.grad = avg_grad

    left_sock.close()
    right_sock.close()
    server.close()


if __name__ == "__main__":
    rank = int(sys.argv[1])
    world_size = 4

    ip_list = [f"192.168.1.{i}" for i in range(1, world_size + 1)]

    worker(rank, world_size, ip_list)