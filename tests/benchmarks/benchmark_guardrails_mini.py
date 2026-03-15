
import timeit
import re
import html
from src.guardrails import InputValidator

def benchmark():
    validator = InputValidator()

    # Test data: a mix of safe and unsafe strings
    test_strings = [
        "This is a perfectly safe string.",
        "Short safe",
        "Another safe string with some numbers 1234567890",
        "Safe string with symbols !@#$%^&*()_+",
        "password = secret", # unsafe
        "../../etc/passwd", # unsafe
        "SELECT * FROM users", # unsafe
        "normal string",
        "another one",
        "yet another safe one",
    ] * 100 # 1000 strings total

    def run_validation():
        for s in test_strings:
            validator.validate_string(s)

    # Measure
    timer = timeit.Timer(run_validation)
    number = 100
    total_time = timer.timeit(number=number)

    print(f"Total time for {number * len(test_strings)} validations: {total_time:.4f} seconds")
    print(f"Average time per validation: {total_time / (number * len(test_strings)) * 1000000:.4f} microseconds")

if __name__ == "__main__":
    benchmark()
