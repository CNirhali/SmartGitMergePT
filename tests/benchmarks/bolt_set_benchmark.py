import difflib
import time
import random
import string

def benchmark():
    num_lines = 200
    lines = [''.join(random.choices(string.ascii_letters, k=50)) for _ in range(num_lines)]

    set_a = set(random.sample(lines, 150))
    set_b = set(random.sample(lines, 150))

    tuple_a = tuple(sorted(list(set_a)))
    tuple_b = tuple(sorted(list(set_b)))

    print(f"Benchmarking with {num_lines} lines...")

    # SequenceMatcher
    start = time.perf_counter()
    for _ in range(1000):
        seq = difflib.SequenceMatcher(None, tuple_a, tuple_b)
        _ = seq.ratio()
    end = time.perf_counter()
    print(f"SequenceMatcher: {end - start:.4f}s")

    # Set intersection
    start = time.perf_counter()
    for _ in range(1000):
        intersection = set_a & set_b
        ratio = 2.0 * len(intersection) / (len(set_a) + len(set_b))
    end = time.perf_counter()
    print(f"Set intersection: {end - start:.4f}s")

    # Verify results
    seq = difflib.SequenceMatcher(None, tuple_a, tuple_b)
    ratio_seq = seq.ratio()
    ratio_set = 2.0 * len(set_a & set_b) / (len(set_a) + len(set_b))
    print(f"Verification: SequenceMatcher ratio={ratio_seq:.4f}, Set ratio={ratio_set:.4f}")

if __name__ == "__main__":
    benchmark()
