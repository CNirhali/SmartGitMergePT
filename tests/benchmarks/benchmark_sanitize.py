import timeit
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from guardrails import InputValidator

def benchmark():
    validator = InputValidator()

    # 1KB, 100KB, 1MB safe strings
    sizes = {"1KB": 1024, "100KB": 1024 * 100, "1MB": 1024 * 1024}

    print(f"{'Scenario':<40} | {'Size':<10} | {'Avg Time (ms)':<15}")
    print("-" * 70)

    for label, size in sizes.items():
        # Case 1: No colons, No tags (Pure fast-path candidate)
        text_no_colon = "A" * size

        # Case 2: Has colons, but safe (Currently hits lower())
        text_with_colon = "A" * (size // 2) + ": content " + "A" * (size // 2)

        # Case 3: Has tags (Hits full regex)
        text_with_tags = "A" * (size // 2) + "<div>" + "A" * (size // 2)

        for case_name, text in [
            ("Safe (No Colons)", text_no_colon),
            ("Safe (With Colons)", text_with_colon),
            ("Unsafe (With Tags)", text_with_tags)
        ]:
            # Number of iterations adjusted by size to keep benchmark time reasonable
            number = 1000 if size < 100000 else 100
            if size > 1000000: number = 10

            timer = timeit.Timer(lambda: validator._sanitize_html(text))
            total_time = timer.timeit(number=number)
            avg_ms = (total_time / number) * 1000

            print(f"{case_name:<40} | {label:<10} | {avg_ms:15.4f}")

if __name__ == "__main__":
    benchmark()
