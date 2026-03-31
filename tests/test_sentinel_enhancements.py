import sys
import os
import ipaddress
sys.path.append(os.path.join(os.getcwd(), 'src'))
from guardrails import InputValidator

def test_sensitive_patterns():
    validator = InputValidator()
    test_cases = [
        ("Authorization: Bearer mytoken123", False),
        ("Authorization: Basic dXNlcjpwYXNz", False),
        ("AKIAIOSFODNN7EXAMPLE", False),
        ("aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", False),
        ("normal text", True),
    ]

    for text, expected_valid in test_cases:
        is_valid, _ = validator.validate_string(text)
        assert is_valid == expected_valid, f"Failed for: {text}"
    print("test_sensitive_patterns passed")

def test_transition_ssrf():
    validator = InputValidator()

    # 6to4 addresses
    # 2002:7f00:0001:: (127.0.0.1)
    assert validator._is_internal_ip(ipaddress.IPv6Address('2002:7f00:1::')) == True
    # 2002:0a00:0001:: (10.0.0.1)
    assert validator._is_internal_ip(ipaddress.IPv6Address('2002:0a00:1::')) == True
    # 2002:0808:0808:: (8.8.8.8) - Global
    assert validator._is_internal_ip(ipaddress.IPv6Address('2002:0808:0808::')) == False

    # Teredo addresses
    # 2001:0000:xxxx:xxxx:xxxx:xxxx:80ff:ffff (127.0.0.0 XOR FFFF:FFFF = 7F00:0000 -> 80ff:ffff)
    # Wait, Teredo IPv4 is last 32 bits XORed with 0xFFFFFFFF
    # 127.0.0.1 -> 0x7f000001 XOR 0xffffffff = 0x80ffffff
    assert validator._is_internal_ip(ipaddress.IPv6Address('2001:0000:0000:0000:0000:0000:80ff:fffe')) == True # 127.0.0.1
    # 8.8.8.8 -> 0x08080808 XOR 0xffffffff = 0xf7f7f7f7
    assert validator._is_internal_ip(ipaddress.IPv6Address('2001:0000:0000:0000:0000:0000:f7f7:f7f7')) == False

    print("test_transition_ssrf passed")

if __name__ == "__main__":
    try:
        test_sensitive_patterns()
        test_transition_ssrf()
        print("All manual tests passed!")
    except Exception as e:
        print(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
