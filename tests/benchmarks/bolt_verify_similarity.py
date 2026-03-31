import sys
import os
import difflib
import time
import random
import string
sys.path.append(os.path.abspath('src'))
from predictor import ConflictPredictor

def generate_random_lines(n, line_length=50):
    lines = []
    for _ in range(n):
        lines.append(''.join(random.choices(string.ascii_letters + string.digits, k=line_length)))
    return sorted(list(set(lines)))

def benchmark():
    predictor = ConflictPredictor()
    num_lines = 1000
    lines_a = generate_random_lines(num_lines)
    lines_b = sorted(list(set(lines_a + generate_random_lines(100))))

    tuple_a = tuple(lines_a)
    tuple_b = tuple(lines_b)

    # Pre-clear cache if any
    predictor._cached_similarity_ratio.cache_clear()

    print(f"\nBenchmarking with {num_lines} lines...")

    # Original (SequenceMatcher ratio)
    start = time.perf_counter()
    for _ in range(100):
        seq = difflib.SequenceMatcher(None, tuple_a, tuple_b)
        _ = seq.ratio() > 0.7
    t_orig = time.perf_counter() - start
    print(f"Original O(N^2) (100 iterations): {t_orig:.4f}s")

    # Optimized (set intersection)
    start = time.perf_counter()
    for _ in range(100):
        _ = predictor._semantic_similarity(tuple_a, tuple_b)
    t_opt = time.perf_counter() - start
    print(f"Optimized O(N) (100 iterations): {t_opt:.4f}s")

    print(f"Speedup: {t_orig / t_opt:.1f}x")

if __name__ == "__main__":
    benchmark()
