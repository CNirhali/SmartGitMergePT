from src.guardrails import InputValidator
import ipaddress

def test_ssrf():
    validator = InputValidator()

    # Test cases that should be blocked even with allow_local=True
    blocked_cases = [
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/admin",
        "http://172.16.0.1/",
        "http://192.168.1.1/",
    ]

    # Test cases that should be allowed with allow_local=True
    allowed_cases = [
        "http://localhost:11434/v1/chat/completions",
        "http://127.0.0.1:11434/v1/chat/completions",
        "http://[::1]:11434/v1/chat/completions",
        "http://google.com",
        "https://api.openai.com/v1/chat/completions",
    ]

    print("Testing with allow_local=True:")
    for url in blocked_cases:
        is_valid, msg = validator.validate_url(url, allow_local=True)
        if is_valid:
            print(f"❌ FAIL: {url} was ALLOWED but should be BLOCKED")
        else:
            print(f"✅ PASS: {url} was BLOCKED: {msg}")

    for url in allowed_cases:
        is_valid, msg = validator.validate_url(url, allow_local=True)
        if not is_valid:
            print(f"❌ FAIL: {url} was BLOCKED but should be ALLOWED: {msg}")
        else:
            print(f"✅ PASS: {url} was ALLOWED")

if __name__ == "__main__":
    test_ssrf()
