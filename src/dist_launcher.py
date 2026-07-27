import json
import socket
import time

import torch

from worker_runner import run_worker

def _set_nodelay(sock):
    # Disable Nagle's algorithm: this protocol is a tight send/recv round-trip
    # per hop, and Nagle interacting with delayed ACKs adds tens of ms of
    # stall per hop that has nothing to do with the algorithm being measured.
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


def create_socket():
    return _set_nodelay(socket.socket(socket.AF_INET, socket.SOCK_STREAM))


class SocketEndpoint:
    def __init__(
        self,
        conn: socket.socket,
        listener: socket.socket | None = None,
        *,
        rank: int | None = None,
        direction: str | None = None,
    ):
        self._conn = conn
        self._listener = listener
        self._rank = rank
        self._direction = direction
        self.bytes_sent = 0
        self.bytes_received = 0

    def _recv_exact(self, nbytes):
        data = bytearray(nbytes)
        view = memoryview(data)
        received = 0
        while received < nbytes:
            n = self._conn.recv_into(view[received:], nbytes - received)
            if n == 0:
                raise ConnectionError("socket closed while receiving payload")
            received += n
        return data

    def send(self, tensor):
        """Send a 1-D float32 tensor as a raw length-prefixed buffer (no pickle)."""
        array = tensor.detach().contiguous().numpy()
        header = array.nbytes.to_bytes(4, byteorder="big")
        self._conn.sendall(header)
        self._conn.sendall(array.data)
        self.bytes_sent += 4 + array.nbytes

    def recv(self):
        """Receive a 1-D float32 tensor as a raw length-prefixed buffer (no pickle)."""
        size = int.from_bytes(self._recv_exact(4), byteorder="big")
        payload = self._recv_exact(size)
        self.bytes_received += 4 + size
        return torch.frombuffer(payload, dtype=torch.float32).clone()

    def close(self):
        try:
            self._conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._conn.close()
        if self._listener is not None:
            self._listener.close()


def _get_binary_tree_structure(rank: int, world_size: int) -> dict:
    """Calculate parent, left_child, right_child for binary tree topology."""
    if world_size <= 1:
        return {"parent": None, "left_child": None, "right_child": None}
    
    parent = (rank - 1) // 2 if rank > 0 else None
    left_child = 2 * rank + 1 if 2 * rank + 1 < world_size else None
    right_child = 2 * rank + 2 if 2 * rank + 2 < world_size else None
    
    return {"parent": parent, "left_child": left_child, "right_child": right_child}


def build_tree_topology(config):
    """Setup binary tree topology for distributed tree aggregation."""
    rank = config["rank"]
    world_size = config["world_size"]
    base_port = int(config.get("base_port", 5000))
    local_ip = config["ip_list"][rank]
    
    tree_struct = _get_binary_tree_structure(rank, world_size)
    endpoints = {}
    
    # Bind listener for incoming connections from children
    local_port = base_port + rank
    listener = create_socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print(f"[dist_launcher] rank={rank} tree_ports local={local_ip}:{local_port}", flush=True)
    print(f"[dist_launcher] rank={rank} binding {local_ip}:{local_port}", flush=True)
    listener.bind((local_ip, local_port))
    listener.listen(2)  # Up to 2 children
    
    # Connect to parent if not root
    if tree_struct["parent"] is not None:
        parent_rank = tree_struct["parent"]
        parent_ip = config["ip_list"][parent_rank]
        parent_port = base_port + parent_rank
        parent_conn = create_socket()
        
        connect_timeout = float(config.get("connect_timeout", 10.0))
        start_time = time.time()
        while True:
            try:
                print(
                    f"[dist_launcher] rank={rank} attempting connect to parent={parent_rank} {parent_ip}:{parent_port}",
                    flush=True,
                )
                parent_conn.connect((parent_ip, parent_port))
                print(f"[dist_launcher] rank={rank} connected to parent at {parent_ip}:{parent_port}", flush=True)
                break
            except OSError as error:
                if time.time() - start_time > connect_timeout:
                    raise ConnectionError(
                        f"rank={rank} failed to connect to parent {parent_ip}:{parent_port} after {connect_timeout}s: {error}"
                    )
                print(
                    f"[dist_launcher] rank={rank} connect failed to parent: {error}; retrying",
                    flush=True,
                )
                time.sleep(0.2)
        
        endpoints["parent_endpoint"] = SocketEndpoint(parent_conn, rank=rank, direction="parent")
    
    # Accept connections from children
    num_children = 0
    if tree_struct["left_child"] is not None:
        print(f"[dist_launcher] rank={rank} waiting to accept left_child connection on {local_ip}:{local_port}", flush=True)
        left_child_conn, _ = listener.accept()
        _set_nodelay(left_child_conn)
        print(f"[dist_launcher] rank={rank} accepted connection from left_child", flush=True)
        endpoints["left_child_endpoint"] = SocketEndpoint(left_child_conn, rank=rank, direction="left_child")
        num_children += 1
    
    if tree_struct["right_child"] is not None:
        print(f"[dist_launcher] rank={rank} waiting to accept right_child connection on {local_ip}:{local_port}", flush=True)
        right_child_conn, _ = listener.accept()
        _set_nodelay(right_child_conn)
        print(f"[dist_launcher] rank={rank} accepted connection from right_child", flush=True)
        endpoints["right_child_endpoint"] = SocketEndpoint(right_child_conn, rank=rank, direction="right_child")
        num_children += 1
    
    # Close listener after accepting all expected children
    if num_children == 0:
        listener.close()
    else:
        # Keep listener open in a separate endpoint for cleanup
        endpoints["_listener"] = listener
    
    return endpoints


def build_parameter_server_topology(config):
    """Setup parameter server topology: rank 0 is server, others are clients."""
    rank = config["rank"]
    world_size = config["world_size"]
    base_port = int(config.get("base_port", 5000))
    local_ip = config["ip_list"][rank]
    endpoints = {}
    
    if rank == 0:
        # Server role: accept connections from all clients
        server_port = base_port
        server_socket = create_socket()
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print(f"[dist_launcher] rank={rank} ps_server binding on {local_ip}:{server_port}", flush=True)
        server_socket.bind((local_ip, server_port))
        server_socket.listen(world_size - 1)
        print(f"[dist_launcher] rank={rank} ps_server listening on {local_ip}:{server_port}", flush=True)
        
        # Accept connections from all clients
        for client_rank in range(1, world_size):
            print(f"[dist_launcher] rank={rank} ps_server waiting for client {client_rank}", flush=True)
            client_conn, client_addr = server_socket.accept()
            _set_nodelay(client_conn)
            print(f"[dist_launcher] rank={rank} ps_server accepted client {client_rank} from {client_addr}", flush=True)
            endpoints[f"client_{client_rank}_endpoint"] = SocketEndpoint(
                client_conn, rank=rank, direction=f"client_{client_rank}"
            )
        
        # Store listener for cleanup
        endpoints["_listener"] = server_socket
    else:
        # Client role: connect to server
        server_rank = 0
        server_ip = config["ip_list"][server_rank]
        server_port = base_port
        server_conn = create_socket()
        
        connect_timeout = float(config.get("connect_timeout", 10.0))
        start_time = time.time()
        while True:
            try:
                print(
                    f"[dist_launcher] rank={rank} ps_client attempting connect to server {server_ip}:{server_port}",
                    flush=True,
                )
                server_conn.connect((server_ip, server_port))
                print(f"[dist_launcher] rank={rank} ps_client connected to server at {server_ip}:{server_port}", flush=True)
                break
            except OSError as error:
                if time.time() - start_time > connect_timeout:
                    raise ConnectionError(
                        f"rank={rank} failed to connect to server {server_ip}:{server_port} after {connect_timeout}s: {error}"
                    )
                print(
                    f"[dist_launcher] rank={rank} ps_client connect failed: {error}; retrying",
                    flush=True,
                )
                time.sleep(0.2)
        
        endpoints["server_endpoint"] = SocketEndpoint(server_conn, rank=rank, direction="server")

    return endpoints


def build_ring_topology(config):
    """Setup ring topology for distributed ring aggregation."""
    rank = config["rank"]
    local_ip = config["ip_list"][rank]
    base_port = int(config.get("base_port", 5000))
    local_port = base_port + rank
    left_ip = config["ip_list"][(rank - 1) % config["world_size"]]
    right_ip = config["ip_list"][(rank + 1) % config["world_size"]]
    listener = create_socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print(
        f"[dist_launcher] rank={rank} ring_ports local={local_ip}:{local_port} left={left_ip}:{base_port + ((rank - 1) % config['world_size'])} right={right_ip}:{base_port + ((rank + 1) % config['world_size'])}",
        flush=True,
    )
    print(f"[dist_launcher] rank={rank} binding {local_ip}:{local_port}", flush=True)
    listener.bind((local_ip, local_port))
    print(f"[dist_launcher] rank={rank} listening on {local_ip}:{local_port}", flush=True)
    listener.listen(1)

    right_conn = create_socket()
    right_port = base_port + ((rank + 1) % config["world_size"])
    connect_timeout = float(config.get("connect_timeout", 10.0))
    start_time = time.time()
    while True:
        try:
            print(
                f"[dist_launcher] rank={rank} attempting connect to right_peer={(rank + 1) % config['world_size']} {right_ip}:{right_port}",
                flush=True,
            )
            right_conn.connect((right_ip, right_port))
            print(f"[dist_launcher] rank={rank} connected to {right_ip}:{right_port}", flush=True)
            break
        except OSError as error:
            if time.time() - start_time > connect_timeout:
                raise ConnectionError(
                    f"rank={rank} failed to connect to {right_ip}:{right_port} after {connect_timeout}s: {error}"
                )
            print(
                f"[dist_launcher] rank={rank} connect failed to {right_ip}:{right_port}: {error}; retrying",
                flush=True,
            )
            time.sleep(0.2)

    print(f"[dist_launcher] rank={rank} waiting to accept left connection on {local_ip}:{local_port}", flush=True)
    left_conn, _ = listener.accept()
    _set_nodelay(left_conn)
    print(f"[dist_launcher] rank={rank} accepted connection from left peer", flush=True)
    return {
        "left_endpoint": SocketEndpoint(left_conn, listener=listener, rank=rank, direction="left"),
        "right_endpoint": SocketEndpoint(right_conn, rank=rank, direction="right"),
    }


def build_distributed_topology(config):

    algo = config["algo"]

    if algo == "ring":
        return build_ring_topology(config)

    if algo == "tree":
        return build_tree_topology(config)

    if algo == "parameter_server":
        return build_parameter_server_topology(config)

    raise ValueError(f"Unknown algorithm: {algo}")


def launch_distributed(config):
    topo = build_distributed_topology(config)

    worker_config = {
        **config,
        **topo,
    }
    run_worker(worker_config)