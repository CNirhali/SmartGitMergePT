import time
import sys
import os
import difflib
from typing import Dict

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from predictor import ConflictPredictor

def benchmark_semantic_similarity():
    print("--- Benchmarking _semantic_similarity ---")
    predictor = ConflictPredictor()

    # Ensure no caching by bypassing @lru_cache if possible, or using unique strings
    # ConflictPredictor._semantic_similarity is decorated with @functools.lru_cache(maxsize=1024)

    # Test case: One is a substring of another
    s1 = "A" * 5000
    s2 = "A" * 6000

    # First call will be slow and cached
    predictor._semantic_similarity(s1, s2)

    # To measure SequenceMatcher without lru_cache interference for DIFFERENT pairs:
    start = time.perf_counter()
    iterations = 20
    for i in range(iterations):
        # Unique strings each time to bypass lru_cache
        p1 = s1 + str(i)
        p2 = s2 + str(i)
        predictor._semantic_similarity(p1, p2)
    end = time.perf_counter()

    avg_time = (end - start) / iterations
    print(f"Substring case (new pairs) avg time: {avg_time:.6f}s")

    # Test case: Not substrings but similar (ratio > 0.7)
    start = time.perf_counter()
    for i in range(iterations):
        p3 = s1 + "B" * 500 + str(i)
        p4 = s1 + "C" * 500 + str(i)
        predictor._semantic_similarity(p3, p4)
    end = time.perf_counter()

    avg_time = (end - start) / iterations
    print(f"Similar non-substring (new pairs, string) avg time: {avg_time:.6f}s")

    # Test case: Tuple of lines (the new optimized path)
    t1 = tuple(s1.splitlines())
    t2 = tuple(s2.splitlines())
    start = time.perf_counter()
    for i in range(iterations):
        p3 = t1 + (f"B {i}",)
        p4 = t1 + (f"C {i}",)
        predictor._semantic_similarity(p3, p4)
    end = time.perf_counter()
    avg_time = (end - start) / iterations
    print(f"Similar non-substring (new pairs, tuple) avg time: {avg_time:.6f}s")

def benchmark_lazy_sorted_lines():
    print("\n--- Benchmarking _get_lazy_sorted_lines ---")
    predictor = ConflictPredictor()

    data = {
        'lines': {f'file_{i}.txt': {f'line {j}' for j in range(100)} for i in range(100)},
        'sorted_lines': {}
    }

    # Subsequent access (cache hits)
    # We want to measure the overhead of:
    # if 'sorted_lines' not in data: ...
    # if file not in data['sorted_lines']: ...
    # return data['sorted_lines'][file]

    # Pre-populate
    for i in range(10):
        predictor._get_lazy_sorted_lines(data, f'file_{i}.txt')

    start = time.perf_counter()
    iterations = 1000000
    for _ in range(iterations):
        # Hot loop: mostly returns already calculated values
        predictor._get_lazy_sorted_lines(data, 'file_0.txt')
    end = time.perf_counter()

    avg_time = (end - start) / iterations
    print(f"Cached access (hot loop) avg time: {avg_time:.8f}s")

if __name__ == "__main__":
    benchmark_semantic_similarity()
    benchmark_lazy_sorted_lines()
