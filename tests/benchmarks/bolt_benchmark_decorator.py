import time
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from optimizer import PerformanceOptimizer, CacheConfig, PerformanceConfig

def benchmark_decorator_overhead():
    cache_config = CacheConfig()
    perf_config = PerformanceConfig()
    optimizer = PerformanceOptimizer(cache_config, perf_config)

    @optimizer.resource_optimized()
    def fast_function():
        return True

    def baseline_function():
        return True

    n = 1000

    # Warm up
    for _ in range(100):
        fast_function()
        baseline_function()

    start = time.time()
    for _ in range(n):
        baseline_function()
    end = time.time()
    baseline_time = (end - start) / n * 1e6
    print(f"Baseline function: {baseline_time:.2f} us per call")

    start = time.time()
    for _ in range(n):
        fast_function()
    end = time.time()
    decorated_time = (end - start) / n * 1e6
    print(f"Resource-optimized function: {decorated_time:.2f} us per call")
    print(f"Overhead: {decorated_time - baseline_time:.2f} us per call")

    # Measure components
    import psutil
    proc = psutil.Process()

    # Measure validate_input overhead
    from guardrails import GuardrailsManager
    gm = GuardrailsManager()

    start = time.time()
    for _ in range(n):
        gm.validate_input(True)
    end = time.time()
    validate_input_overhead = (end - start) / n * 1e6
    print(f"validate_input(True) overhead: {validate_input_overhead:.2f} us")

    # Measure monitor.record_request
    from guardrails import PerformanceMonitor
    pm = PerformanceMonitor()
    start = time.time()
    for _ in range(n):
        pm.record_request(0.001, True)
    end = time.time()
    record_request_overhead = (end - start) / n * 1e6
    print(f"record_request overhead: {record_request_overhead:.2f} us")

if __name__ == "__main__":
    benchmark_decorator_overhead()
