import difflib
import time
import random
import string

def generate_random_lines(n, line_length=50):
    lines = []
    for _ in range(n):
        lines.append(''.join(random.choices(string.ascii_letters + string.digits, k=line_length)))
    return lines

def benchmark():
    num_lines = 100
    lines_a = generate_random_lines(num_lines)
    lines_b = list(lines_a)
    # Introduce some changes
    for _ in range(10):
        idx = random.randint(0, num_lines - 1)
        lines_b[idx] = generate_random_lines(1)[0]

    str_a = "\n".join(lines_a)
    str_b = "\n".join(lines_b)

    tuple_a = tuple(lines_a)
    tuple_b = tuple(lines_b)

    print(f"Benchmarking with {num_lines} lines...")

    # String comparison
    start = time.perf_counter()
    for _ in range(100):
        seq = difflib.SequenceMatcher(None, str_a, str_b)
        _ = seq.ratio()
    end = time.perf_counter()
    print(f"String comparison (character-level): {end - start:.4f}s")

    # Tuple comparison
    start = time.perf_counter()
    for _ in range(100):
        seq = difflib.SequenceMatcher(None, tuple_a, tuple_b)
        _ = seq.ratio()
    end = time.perf_counter()
    print(f"Tuple comparison (line-level): {end - start:.4f}s")

if __name__ == "__main__":
    benchmark()
