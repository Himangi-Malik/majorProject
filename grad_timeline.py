import threading
import time

# bilkul bhi lauda bhi nhi smjh aaya ki kya hua mere saath 

# Simulated all-reduce (communication)
def all_reduce(layer_name):
    print(f"[{time.time():.2f}]   -> All-reduce START for {layer_name}")
    time.sleep(1)  # simulate communication cost
    print(f"[{time.time():.2f}]   -> All-reduce DONE  for {layer_name}")

# Hook function
def gradient_ready_hook(layer_name):
    print(f"[{time.time():.2f}] Hook fired for {layer_name}")
    
    # Launch communication in a separate thread
    thread = threading.Thread(target=all_reduce, args=(layer_name,))
    thread.start()
    return thread

def backward_with_overlap():
    print("=== BACKWARD WITH DDP-STYLE HOOKS ===")
    start = time.time()

    layers = ["L3", "L2", "L1"]
    threads = []

    for layer in layers:
        print(f"[{time.time():.2f}] Computing gradient for {layer}")
        time.sleep(1)  # simulate gradient computation

        # gradient finished -> hook fires immediately
        t = gradient_ready_hook(layer)
        threads.append(t)

    # Wait for all communication to finish
    for t in threads:
        t.join()

    print(f"Total time: {time.time() - start:.2f} seconds\n")


def backward_without_overlap():
    print("=== BACKWARD WITHOUT OVERLAP (NAIVE) ===")
    start = time.time()

    layers = ["L3", "L2", "L1"]

    # First compute all gradients
    for layer in layers:
        print(f"[{time.time():.2f}] Computing gradient for {layer}")
        time.sleep(1)

    # Then communicate everything
    for layer in layers:
        all_reduce(layer)

    print(f"Total time: {time.time() - start:.2f} seconds\n")


if __name__ == "__main__":
    backward_without_overlap()
    backward_with_overlap()