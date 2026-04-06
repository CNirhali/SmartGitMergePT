import pytest
import ipaddress
import socket
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.guardrails import InputValidator

def test_guardrails_secrets_and_dos():
    validator = InputValidator()

    # 1. Test Secret Patterns (Expanded)
    # GitHub PAT (classic)
    is_valid, msg = validator.validate_string("ghp_1234567890abcdefghijklmnopqrstuvwxyz1234")
    assert is_valid is False
    assert "Sensitive data" in msg

    # GitHub Fine-grained PAT
    is_valid, msg = validator.validate_string("github_pat_1234567890abcdefghij12_1234567890abcdefghijklmnopqrstuvwxyz1234567890abcdefghijklm")
    assert is_valid is False
    assert "Sensitive data" in msg

    # Private Key Header
    is_valid, msg = validator.validate_string("-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA75")
    assert is_valid is False
    assert "Sensitive data" in msg

    # 2. Test URL length limit (DoS protection)
    long_url = "http://example.com/" + "a" * 2040 # Over 2048 chars
    is_valid, msg = validator.validate_url(long_url)
    assert is_valid is False
    assert "URL too long" in msg

def test_ssrf_transition_mechanisms():
    validator = InputValidator()

    # 3. Test IPv6 Transition Mechanism SSRF Bypasses (IPv6 encapsulating internal IPv4)

    # 6to4 (2002::/16) - Encapsulates 127.0.0.1
    url_6to4 = "http://[2002:7f00:0001::]"
    is_valid, msg = validator.validate_url(url_6to4)
    assert is_valid is False
    assert "Internal IP" in msg

    # Teredo (2001:0::/32) - Encapsulates 127.0.0.1 (XORed)
    # 127.0.0.1 XOR 255.255.255.255 = 128.255.254.254 (80 ff fe fe)
    url_teredo = "http://[2001:0000:4136:e378:8000:63bf:80ff:fefe]"
    is_valid, msg = validator.validate_url(url_teredo)
    assert is_valid is False
    assert "Internal IP" in msg

    # IPv4-compatible IPv6 loopback
    url_compat = "http://[::127.0.0.1]"
    is_valid, msg = validator.validate_url(url_compat)
    assert is_valid is False
    assert "Internal IP" in msg

    # IPv4-mapped IPv6 loopback
    url_mapped = "http://[::ffff:127.0.0.1]"
    is_valid, msg = validator.validate_url(url_mapped)
    assert is_valid is False
    assert "Internal IP" in msg

def test_ssrf_advanced_cases():
    validator = InputValidator()

    # 4. Miscellaneous advanced bypasses

    # Percent-encoded null byte
    url_null = "http://127.0.0.1%00.evil.com"
    is_valid, msg = validator.validate_url(url_null)
    assert is_valid is False
    assert "Null byte" in msg or "Internal IP" in msg

    # IPv6 Scope ID
    url_scope = "http://[fe80::1%eth0]"
    is_valid, msg = validator.validate_url(url_scope)
    assert is_valid is False
    assert "Internal IP" in msg

    # Carrier-Grade NAT (Shared Address Space - non-global)
    url_cgnat = "http://100.64.0.1"
    is_valid, msg = validator.validate_url(url_cgnat)
    assert is_valid is False
    assert "Internal IP" in msg
