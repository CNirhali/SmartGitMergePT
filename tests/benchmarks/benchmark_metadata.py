import time
import random
import string

def generate_large_diff(num_files=1000, lines_per_file=20):
    lines = []
    for i in range(num_files):
        filename = f"file_{i}.txt"
        lines.append(f"diff --git a/{filename} b/{filename}")
        lines.append(f"--- a/{filename}")
        lines.append(f"+++ b/{filename}")
        for j in range(lines_per_file):
            content = ''.join(random.choices(string.ascii_letters, k=50))
            lines.append(f"+{content}")
            lines.append(f"-{content}")
    return "\n".join(lines)

def original_get_diff_metadata(diff):
    files = set()
    lines_by_file = {}
    current_lines = None
    for line in diff.splitlines():
        if not line: continue
        c0 = line[0]
        if c0 == '+' or c0 == '-':
            if current_lines is not None:
                if line.startswith('+++') or line.startswith('---'):
                    continue
                current_lines.add(line[1:])
        elif c0 == 'd':
            if line.startswith('diff --git'):
                b_idx = line.find(' b/')
                if b_idx != -1:
                    current_file = line[b_idx + 3:]
                    files.add(current_file)
                    current_lines = set()
                    lines_by_file[current_file] = current_lines
                else:
                    current_lines = None
    return files, lines_by_file

def optimized_get_diff_metadata(diff):
    files = set()
    lines_by_file = {}
    current_lines = None
    # Use split('\n') instead of splitlines() for potentially less overhead
    for line in diff.split('\n'):
        if not line: continue
        c0 = line[0]
        if c0 == '+' or c0 == '-':
            if current_lines is not None:
                # Optimized header check: check 2nd and 3rd chars
                # If it's a diff line it must have at least 1 char.
                # We check if 2nd and 3rd match the first (+ or -)
                if len(line) >= 3 and line[1] == c0 and line[2] == c0:
                    continue
                current_lines.add(line[1:])
        elif c0 == 'd':
            if line.startswith('diff --git'):
                b_idx = line.find(' b/')
                if b_idx != -1:
                    current_file = line[b_idx + 3:]
                    files.add(current_file)
                    current_lines = set()
                    lines_by_file[current_file] = current_lines
                else:
                    current_lines = None
    return files, lines_by_file

if __name__ == "__main__":
    diff = generate_large_diff(2000, 20) # ~84,000 lines
    print(f"Diff size: {len(diff)/1024/1024:.2f} MB")

    # Warm up
    original_get_diff_metadata(diff[:1000])
    optimized_get_diff_metadata(diff[:1000])

    start = time.perf_counter()
    for _ in range(5):
        original_get_diff_metadata(diff)
    end = time.perf_counter()
    print(f"Original (5 iterations): {end - start:.4f}s (avg {(end-start)/5:.4f}s)")

    start = time.perf_counter()
    for _ in range(5):
        optimized_get_diff_metadata(diff)
    end = time.perf_counter()
    print(f"Optimized (5 iterations): {end - start:.4f}s (avg {(end-start)/5:.4f}s)")
