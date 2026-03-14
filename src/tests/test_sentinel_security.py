import pytest
import time
from guardrails import RateLimiter, InputValidator

def test_rate_limiter_memory_exhaustion():
    # Current implementation has no limit on number of keys
    # We want to ensure that it doesn't grow indefinitely
    limiter = RateLimiter(max_requests=10, window_seconds=60)

    # Simulate many unique keys (e.g., from a botnet or IP spoofing)
    # If the limit is 1000 (default), this should trigger eviction
    for i in range(2000):
        limiter.is_allowed(f"key_{i}")

    # If we implement a limit of 1000, then len(limiter.requests) should be <= 1000
    # Currently this will fail if we expect a limit
    assert hasattr(limiter, 'max_tracked_keys'), "RateLimiter should have max_tracked_keys attribute"
    assert len(limiter.requests) <= limiter.max_tracked_keys

def test_input_validator_null_byte():
    validator = InputValidator()
    # Null bytes can sometimes bypass security checks if not handled correctly
    malicious_string = "safe_string\0malicious_content"
    is_valid, result = validator.validate_string(malicious_string)

    # We want this to be invalid
    assert not is_valid, "Input with null bytes should be rejected"
    assert "Null byte" in result

def test_validate_input_circular_reference():
    from guardrails import GuardrailsManager
    gm = GuardrailsManager()

    # Dictionary circular reference
    a = {}
    a['b'] = a
    is_valid, result = gm.validate_input(a)
    assert not is_valid
    assert "Circular reference" in result

    # List circular reference
    L = []
    L.append(L)
    is_valid, result = gm.validate_input(L)
    assert not is_valid
    assert "Circular reference" in result

def test_validate_input_deep_nesting():
    from guardrails import GuardrailsManager
    gm = GuardrailsManager()

    # Create deeply nested list
    deep = "final"
    for _ in range(25): # Limit is 20
        deep = [deep]

    is_valid, result = gm.validate_input(deep)
    assert not is_valid
    assert "too deep" in result

def test_validate_input_dag_allowed():
    from guardrails import GuardrailsManager
    gm = GuardrailsManager()

    # Directed Acyclic Graph (DAG) - shared reference but NOT circular
    shared = {"key": "value"}
    dag = {
        "part1": shared,
        "part2": shared
    }

    is_valid, result = gm.validate_input(dag)
    assert is_valid, f"DAG should be valid, but got error: {result}"
    assert result["part1"] == shared
    assert result["part2"] == shared
