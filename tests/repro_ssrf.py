import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
from guardrails import InputValidator

validator = InputValidator()

urls = [
    "http://127.0.0.1",
    "http://127.0.0.1.",
    "http://%31%32%37%2e%30%2e%30%2e%31",
    "http://[::1]",
    "http://[::ffff:7f00:1]",
    "http://127.1",
    "http://127.1.",
    "http://localhost%2e",
    "http://%6c%6f%63%61%6c%68%6f%73%74",
]

print(f"{'URL':<45} | {'Valid':<5} | {'Message'}")
print("-" * 80)
for url in urls:
    is_valid, msg = validator.validate_url(url)
    print(f"{url:<45} | {int(is_valid):<5} | {msg}")
