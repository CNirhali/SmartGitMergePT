import time
import psutil
import os

def benchmark_psutil():
    process = psutil.Process()

    # Warm up
    process.cpu_percent(interval=None)
    process.memory_info().rss

    n = 10000

    start = time.time()
    for _ in range(n):
        process.cpu_percent(interval=None)
    end = time.time()
    print(f"cpu_percent(interval=None): { (end - start) / n * 1e6 :.2f} us per call")

    start = time.time()
    for _ in range(n):
        process.memory_info().rss
    end = time.time()
    print(f"memory_info().rss: { (end - start) / n * 1e6 :.2f} us per call")

    start = time.time()
    for _ in range(n):
        psutil.Process()
    end = time.time()
    print(f"psutil.Process() initialization: { (end - start) / n * 1e6 :.2f} us per call")

if __name__ == "__main__":
    benchmark_psutil()
