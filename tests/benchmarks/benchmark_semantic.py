import time
import random
import string
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from predictor import ConflictPredictor

def generate_random_string(length):
    return ''.join(random.choices(string.ascii_letters + string.digits + "\n", k=length))

def main():
    predictor = ConflictPredictor()

    # Test 1: Highly different lengths (should trigger early rejection)
    s1 = generate_random_string(1000)
    s2 = generate_random_string(3000)

    print("Testing highly different lengths with ConflictPredictor._semantic_similarity...")
    start = time.perf_counter()
    for _ in range(1000):
        predictor._semantic_similarity(s1, s2)
    end = time.perf_counter()
    print(f"Time taken (1000 iterations): {end - start:.4f}s")

    # Test 2: Similar strings (should pass early rejection)
    s3 = generate_random_string(1000)
    s4 = s3[:900] + generate_random_string(100)

    print("\nTesting similar strings with ConflictPredictor._semantic_similarity...")
    start = time.perf_counter()
    for _ in range(100):
        predictor._semantic_similarity(s3, s4)
    end = time.perf_counter()
    print(f"Time taken (100 iterations): {end - start:.4f}s")

if __name__ == "__main__":
    main()
