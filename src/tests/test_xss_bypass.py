import pytest
from guardrails import InputValidator

def test_html_sanitization_bypass():
    validator = InputValidator()

    # Test nested tags bypass
    # <scr<script>ipt> should be fully sanitized
    nested_tag = "<scr<script>ipt>alert(1)</script>"
    is_valid, sanitized = validator.validate_string(nested_tag)
    assert "alert(1)" not in sanitized
    assert "<script" not in sanitized.lower()

    # Test nested protocol bypass
    # javajavascript:script:alert(1) should be fully sanitized
    nested_protocol = "javajavascript:script:alert(1)"
    is_valid, sanitized = validator.validate_string(nested_protocol)
    assert "javascript:" not in sanitized.lower()

    # Test other dangerous protocols
    # vbscript: and data: should also be blocked in sanitization
    vb_script = "vbscript:alert(1)"
    is_valid, sanitized = validator.validate_string(vb_script)
    assert "vbscript:" not in sanitized.lower()

    data_url = "data:text/html,<script>alert(1)</script>"
    is_valid, sanitized = validator.validate_string(data_url)
    assert "data:" not in sanitized.lower()
