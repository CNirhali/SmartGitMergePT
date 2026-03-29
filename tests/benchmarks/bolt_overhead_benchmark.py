import time
import hashlib
from typing import Dict, Tuple

def benchmark_overhead():
    print("--- Benchmarking Overhead ---")

    # 1. set(tuple) overhead
    lines = [f"line {i}" for i in range(100)]
    line_tuple = tuple(sorted(lines))

    start = time.perf_counter()
    iterations = 100000
    for _ in range(iterations):
        s = set(line_tuple)
    end = time.perf_counter()
    print(f"set(tuple) overhead (100 lines): {((end - start) / iterations) * 1e6:.4f} us")

    # 2. setdefault overhead
    data = {}
    start = time.perf_counter()
    for _ in range(iterations):
        cache = data.setdefault('sorted_lines', {})
    end = time.perf_counter()
    print(f"dict.setdefault overhead: {((end - start) / iterations) * 1e6:.4f} us")

    # 3. Direct access overhead (for comparison)
    data = {'sorted_lines': {}}
    start = time.perf_counter()
    for _ in range(iterations):
        cache = data['sorted_lines']
    end = time.perf_counter()
    print(f"Direct dict access overhead: {((end - start) / iterations) * 1e6:.4f} us")

if __name__ == "__main__":
    benchmark_overhead()
