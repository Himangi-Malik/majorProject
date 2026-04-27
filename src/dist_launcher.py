import json
import pickle
import socket
import time

from worker_runner import run_worker

CONFIG_PATH = "config.json"


def load_json_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def get_left_right_neighbor_ip(left_peer_rank, right_peer_rank):
    config = load_json_config(CONFIG_PATH)
    ip_list = config["ip_list"]
    left_ip = ip_list[left_peer_rank]
    right_ip = ip_list[right_peer_rank]
    return left_ip, right_ip


def create_socket():
    return socket.socket(socket.AF_INET, socket.SOCK_STREAM)


class SocketEndpoint:
    def __init__(self, conn: socket.socket, listener: socket.socket | None = None):
        self._conn = conn
        self._listener = listener

    def send(self, payload):
        raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        header = len(raw).to_bytes(4, byteorder="big")
        self._conn.sendall(header + raw)

    def recv(self):
        data = bytearray()
        while len(data) < 4:
            chunk = self._conn.recv(4 - len(data))
            if not chunk:
                raise ConnectionError("socket closed while receiving payload")
            data.extend(chunk)
        header = bytes(data)
        size = int.from_bytes(header, byteorder="big")
        data = bytearray()
        while len(data) < size:
            chunk = self._conn.recv(size - len(data))
            if not chunk:
                raise ConnectionError("socket closed while receiving payload")
            data.extend(chunk)
        raw = bytes(data)
        return pickle.loads(raw)

    def close(self):
        try:
            self._conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._conn.close()

        if self._listener is not None:
            self._listener.close()


def build_distributed_topology(algo, rank):
    if algo == "ring":
        config = load_json_config(CONFIG_PATH)
        local_ip = config["ip_list"][rank]
        base_port = int(config.get("base_port", 5000))
        left_ip = config["ip_list"][(rank - 1) % config["world_size"]]
        right_ip = config["ip_list"][(rank + 1) % config["world_size"]]
        listener = create_socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((local_ip, base_port))
        listener.listen(1)

        right_conn = create_socket()
        while True:
            try:
                right_conn.connect((right_ip, base_port))
                break
            except OSError:
                time.sleep(0.2)

        left_conn, _ = listener.accept()
        return {
            "left_endpoint": SocketEndpoint(left_conn, listener=listener),
            "right_endpoint": SocketEndpoint(right_conn),
            "left_endpoint_info": {
                "peer_rank": (rank - 1) % config["world_size"],
                "direction": "left",
                "transport": "socket",
            },
            "right_endpoint_info": {
                "peer_rank": (rank + 1) % config["world_size"],
                "direction": "right",
                "transport": "socket",
            },
        }

    if algo == "tree":
        return {}

    if algo == "parameter_server":
        return {}

    raise ValueError("Unknown algo")

def launch_distributed(config):
    topo = build_distributed_topology(
        config["algo"],
        config["rank"],
    )

    worker_config = {
        **config,
        **topo,
    }
    run_worker(worker_config)