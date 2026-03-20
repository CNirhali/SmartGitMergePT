import os
import sys
import time
import hashlib
import difflib

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from predictor import ConflictPredictor

def benchmark_semantic():
    predictor = ConflictPredictor()
    # Generate two strings that are similar but not identical
    # and have enough overlap to pass quick_ratio
    s1 = "import os\nimport sys\ndef hello():\n    print('hello world')\n" * 100
    s2 = "import os\nimport sys\ndef hello_world():\n    print('hello world!')\n" * 100

    print("Benchmarking _semantic_similarity...")

    start = time.perf_counter()
    for _ in range(100):
        predictor._semantic_similarity(s1, s2)
    end = time.perf_counter()
    print(f"Original time (100 iterations): {end - start:.4f}s")

    # Test real_quick_ratio impact
    seq = difflib.SequenceMatcher(None, s1, s2)

    start = time.perf_counter()
    for _ in range(1000):
        r1 = seq.quick_ratio()
    end = time.perf_counter()
    print(f"quick_ratio (1000 iterations): {end - start:.4f}s")

    start = time.perf_counter()
    for _ in range(1000):
        r2 = seq.real_quick_ratio()
    end = time.perf_counter()
    print(f"real_quick_ratio (1000 iterations): {end - start:.4f}s")

    start = time.perf_counter()
    for _ in range(100):
        r3 = seq.ratio()
    end = time.perf_counter()
    print(f"ratio (100 iterations): {end - start:.4f}s")

def benchmark_metadata_extraction():
    predictor = ConflictPredictor()
    # Generate a large diff string
    diff_lines = []
    for i in range(1000):
        diff_lines.append(f"diff --git a/file_{i}.txt b/file_{i}.txt")
        diff_lines.append("--- a/file.txt")
        diff_lines.append("+++ b/file.txt")
        for j in range(10):
            diff_lines.append(f"+added line {j}")
            diff_lines.append(f"-removed line {j}")
    diff_str = "\n".join(diff_lines)

    print("\nBenchmarking _get_diff_metadata...")
    start = time.perf_counter()
    for _ in range(10):
        predictor._get_diff_metadata(diff_str)
    end = time.perf_counter()
    print(f"Original time (10 iterations): {end - start:.4f}s")

if __name__ == "__main__":
    benchmark_semantic()
    benchmark_metadata_extraction()
