import sys
import os
import socket
import ipaddress
sys.path.append(os.path.join(os.getcwd(), 'src'))
from guardrails import InputValidator

validator = InputValidator()

test_urls = [
    "http://017700000001", # Octal integer
    "http://0x7f.1",       # Mixed hex and shorthand
    "http://0177.1",       # Mixed octal and shorthand
    "http://[2002:7f00:0001::]", # 6to4 loopback
    "http://[2001:0000:4136:e378:8000:63bf:80ff:fefe]", # Teredo
    "http://127.0.0.1%00.evil.com", # Percent-encoded null byte
    "http://[::127.0.0.1]", # IPv4-compatible IPv6 (loopback)
    "http://[::ffff:127.0.0.1]", # IPv4-mapped IPv6 (loopback)
    "http://[fe80::1%eth0]", # IPv6 with Scope ID
]

print(f"{'URL':<45} | {'Valid':<5} | {'Message'}")
print("-" * 100)
for url in test_urls:
    try:
        is_valid, msg = validator.validate_url(url)
        print(f"{url:<45} | {int(is_valid):<5} | {msg}")
    except Exception as e:
        print(f"{url:<45} | ERROR | {str(e)}")
