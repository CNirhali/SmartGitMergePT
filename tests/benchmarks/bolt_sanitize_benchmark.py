import time
import sys
import os
import html
import re

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from guardrails import InputValidator

validator = InputValidator()

# A typical large log message that has colons but no HTML and no dangerous characters
safe_large_text = "Log entry: User 'admin' performed action 'login' at 2023-10-27 10:00:00. Details: success. " * 2000

def benchmark_current():
    print(f"Benchmarking _sanitize_html with SAFE text (len={len(safe_large_text)})")

    start = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        validator._sanitize_html(safe_large_text)
    end = time.perf_counter()
    avg = (end - start) / iterations
    print(f"Current implementation avg time: {avg:.6f}s")

if __name__ == "__main__":
    benchmark_current()
