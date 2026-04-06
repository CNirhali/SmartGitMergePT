import timeit
import sys
import os
sys.path.insert(0, 'src')
from guardrails import InputValidator

def benchmark_baseline():
    print("⚡ Bolt: Benchmarking InputValidator Baseline ⚡")
    validator = InputValidator()

    # Typical input strings (no encoding)
    test_strings = [
        "feature/user-authentication",
        "bugfix/issue-123-login-error",
        "main",
        "a" * 100,
        "some normal text with spaces and punctuation.",
    ]

    # URLs (no encoding)
    test_urls = [
        "http://example.com",
        "https://github.com/user/repo",
        "https://api.openai.com/v1/chat/completions",
    ]

    # HTML-like strings (no tags, no encoding)
    test_html = [
        "This is a normal string with a colon: like this.",
        "another:string:with:colons",
    ]

    iterations = 50000

    print(f"\nBenchmarking with {iterations} iterations...")

    # validate_string
    t_str = timeit.timeit(lambda: [validator.validate_string(s) for s in test_strings], number=iterations)
    print(f"validate_string (baseline): {t_str:.4f}s")

    # validate_url
    t_url = timeit.timeit(lambda: [validator.validate_url(u) for u in test_urls], number=iterations)
    print(f"validate_url (baseline):    {t_url:.4f}s")

    # _sanitize_html
    t_html = timeit.timeit(lambda: [validator._sanitize_html(h) for h in test_html], number=iterations)
    print(f"_sanitize_html (baseline):  {t_html:.4f}s")

if __name__ == "__main__":
    benchmark_baseline()
