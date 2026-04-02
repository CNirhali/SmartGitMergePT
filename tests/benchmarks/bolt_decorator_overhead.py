import time
import sys
import os
import psutil

# Add src to path
sys.path.append(os.path.abspath('src'))

# Mock GuardrailsManager if needed, but PerformanceOptimizer seems independent enough
from optimizer import PerformanceOptimizer, CacheConfig, PerformanceConfig

def benchmark_decorator():
    cache_config = CacheConfig()
    perf_config = PerformanceConfig()
    optimizer = PerformanceOptimizer(cache_config, perf_config)

    @optimizer.resource_optimized()
    def fast_func():
        return 1

    # Warm up
    for _ in range(100):
        fast_func()

    start = time.perf_counter()
    iterations = 5000
    for _ in range(iterations):
        fast_func()
    end = time.perf_counter()

    avg_overhead = (end - start) / iterations * 1e6
    print(f"Average @resource_optimized overhead: {avg_overhead:.2f} us")

    @optimizer.cached_function()
    def cached_func(x):
        return x

    start = time.perf_counter()
    for i in range(iterations):
        cached_func(i % 10) # 90% cache hit
    end = time.perf_counter()

    avg_overhead = (end - start) / iterations * 1e6
    print(f"Average @cached_function overhead: {avg_overhead:.2f} us")

if __name__ == "__main__":
    benchmark_decorator()
