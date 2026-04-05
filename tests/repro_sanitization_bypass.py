import html
import re
from urllib.parse import unquote
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from guardrails import InputValidator

validator = InputValidator()

# Payload: javascript:alert(1) with all chars percent-encoded
# j=%6a, a=%61, v=%76, s=%73, c=%63, r=%72, i=%69, p=%70, t=%74
payload = "%6a%61%76%61%73%63%72%69%70%74:alert(1)"

print(f"Testing payload: {payload}")
print(f"Unquoted payload: {unquote(payload)}")

# Test _sanitize_html
# We'll use validate_string which calls _sanitize_html
is_valid, sanitized = validator.validate_string(payload)
print(f"Is valid string: {is_valid}")
print(f"Sanitized result: {sanitized}")

if not is_valid and "Dangerous protocol" in sanitized:
    print("✅ Bypass failed in validate_string (Caught encoded protocol).")
else:
    print("❌ Bypass successful in validate_string!")

# Test validate_url
is_valid_url, result = validator.validate_url(payload)
print(f"Is valid URL: {is_valid_url}")
print(f"Result: {result}")
if not is_valid_url:
    print("✅ Bypass failed in validate_url.")
else:
    print("❌ Bypass successful in validate_url!")
