import pytest
import ipaddress
import socket
from src.guardrails import InputValidator

def test_is_internal_ip_ipv4_mapped():
    validator = InputValidator()

    # IPv4-mapped IPv6 loopback
    ip = ipaddress.ip_address('::ffff:127.0.0.1')
    assert validator._is_internal_ip(ip) is True

    # IPv4-mapped IPv6 private
    ip = ipaddress.ip_address('::ffff:192.168.1.1')
    assert validator._is_internal_ip(ip) is True

    # IPv4-mapped IPv6 link-local
    ip = ipaddress.ip_address('::ffff:169.254.169.254')
    assert validator._is_internal_ip(ip) is True

    # IPv4-mapped IPv6 public
    ip = ipaddress.ip_address('::ffff:8.8.8.8')
    assert validator._is_internal_ip(ip) is False

def test_validate_url_dangerous_protocols_obfuscated():
    validator = InputValidator()

    # Obfuscated protocols
    dangerous_urls = [
        "java script:alert(1)",
        "v b s c r i p t:msgbox(1)",
        "d a t a:text/html,<script>alert(1)</script>",
        "f i l e:///etc/passwd",
        "g o p h e r://localhost:70",
        "p h p://filter/read=convert.base64-encode/resource=config.php",
        "j a r:http://example.com/out.jar!/test.txt",
        "d i c t://localhost:11211/stat",
        "l d a p://localhost:389/o=University%20of%20Michigan,c=US"
    ]

    for url in dangerous_urls:
        valid, msg = validator.validate_url(url)
        assert valid is False, f"URL should be blocked: {url}"
        assert "Dangerous URL protocol detected" in msg

def test_sanitize_html_expanded_fastpath():
    validator = InputValidator()

    # Text with a colon but no dangerous characters from protocols
    # "Hello: World" contains 'h' and 'l' which are now in dangerous_chars (from gopher, php, jar, ldap)
    # So it will now trigger the regex, which is fine and safe.

    # Let's test something that truly doesn't contain any dangerous chars
    safe_text = "Tst: 123" # No j, v, d, f, g, p, h, l
    assert validator._sanitize_html(safe_text) == "Tst: 123"

    # Text with dangerous protocol obfuscated
    dangerous_text = "j a v a s c r i p t :alert(1)"
    sanitized = validator._sanitize_html(dangerous_text)
    assert "javascript" not in sanitized.lower()

    # Test file protocol in HTML context
    file_text = "f i l e :/etc/passwd"
    sanitized = validator._sanitize_html(file_text)
    assert "file" not in sanitized.lower()

def test_validate_url_standard_dangerous_protocols():
    validator = InputValidator()

    # Test newly added protocols in the standard set
    assert validator.validate_url("php://filter")[0] is False
    assert validator.validate_url("jar:http://example.com/out.jar!/")[0] is False

def test_validate_url_integer_ips():
    validator = InputValidator()

    # Integer-based IPs (loopback)
    assert validator.validate_url("http://2130706433")[0] is False # Decimal 127.0.0.1
    assert validator.validate_url("http://0x7f000001")[0] is False # Hex 127.0.0.1
    assert validator.validate_url("http://017700000001")[0] is False # Octal 127.0.0.1

    # Public integer IP
    assert validator.validate_url("http://134744072")[0] is True # 8.8.8.8

def test_is_internal_ip_ipv4_compatible():
    validator = InputValidator()

    # IPv4-compatible IPv6 loopback (::7f00:1)
    ip = ipaddress.ip_address('::127.0.0.1')
    assert validator._is_internal_ip(ip) is True

    # IPv4-compatible IPv6 private (::c0a8:101)
    ip = ipaddress.ip_address('::192.168.1.1')
    assert validator._is_internal_ip(ip) is True

    # IPv4-compatible IPv6 public (::808:808)
    ip = ipaddress.ip_address('::8.8.8.8')
    assert validator._is_internal_ip(ip) is False

def test_validate_url_null_bytes_and_scope_id():
    validator = InputValidator()

    # Percent-encoded null byte in hostname
    assert validator.validate_url("http://127.0.0.1%00.evil.com")[0] is False

    # IPv6 with Scope ID
    assert validator.validate_url("http://[fe80::1%eth0]")[0] is False # fe80::1 is link-local
