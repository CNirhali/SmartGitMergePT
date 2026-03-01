"""
Tests for the Guardrails System
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from guardrails import (
    GuardrailsManager, SecurityLevel, PerformanceLevel,
    InputValidator, RateLimiter, DataEncryption,
    PerformanceMonitor, SecurityMonitor, ErrorHandler,
    with_guardrails, secure_function, resource_guard
)

class TestInputValidator:
    """Test input validation functionality"""
    
    def setup_method(self):
        self.validator = InputValidator()
    
    def test_validate_string_normal(self):
        """Test normal string validation"""
        is_valid, result = self.validator.validate_string("normal string")
        assert is_valid
        assert result == "normal string"
    
    def test_validate_string_too_long(self):
        """Test string length validation"""
        long_string = "a" * 1001
        is_valid, result = self.validator.validate_string(long_string, max_length=1000)
        assert not is_valid
        assert "too long" in result
    
    def test_validate_string_sensitive_data(self):
        """Test sensitive data detection"""
        sensitive_string = "password=secret123"
        is_valid, result = self.validator.validate_string(sensitive_string)
        assert not is_valid
        assert "Sensitive data" in result
    
    def test_validate_string_path_traversal(self):
        """Test path traversal detection"""
        malicious_string = "../../../etc/passwd"
        is_valid, result = self.validator.validate_string(malicious_string)
        assert not is_valid
        assert "Path traversal" in result
    
    def test_validate_string_sql_injection(self):
        """Test SQL injection detection"""
        malicious_string = "'; DROP TABLE users; --"
        is_valid, result = self.validator.validate_string(malicious_string)
        assert not is_valid
        assert "SQL injection" in result
    
    def test_validate_path_safe(self):
        """Test safe path validation"""
        base_path = "/safe/directory"
        safe_path = "/safe/directory/file.txt"
        is_valid, result = self.validator.validate_path(safe_path, base_path)
        assert is_valid
        assert result == safe_path
    
    def test_validate_path_traversal(self):
        """Test path traversal prevention"""
        base_path = "/safe/directory"
        malicious_path = "/safe/directory/../../../etc/passwd"
        is_valid, result = self.validator.validate_path(malicious_path, base_path)
        assert not is_valid
        assert "Path traversal" in result
    
    def test_validate_url_safe(self):
        """Test safe URL validation"""
        safe_url = "https://example.com"
        is_valid, result = self.validator.validate_url(safe_url)
        assert is_valid
        assert result == safe_url
    
    def test_validate_url_dangerous_protocol(self):
        """Test dangerous protocol detection"""
        dangerous_url = "javascript:alert('xss')"
        is_valid, result = self.validator.validate_url(dangerous_url)
        assert not is_valid
        assert "Dangerous" in result

class TestRateLimiter:
    """Test rate limiting functionality"""
    
    def setup_method(self):
        self.limiter = RateLimiter(max_requests=3, window_seconds=1)
    
    def test_rate_limit_allowed(self):
        """Test that requests within limit are allowed"""
        assert self.limiter.is_allowed("test_key")
        assert self.limiter.is_allowed("test_key")
        assert self.limiter.is_allowed("test_key")
    
    def test_rate_limit_exceeded(self):
        """Test that requests beyond limit are blocked"""
        assert self.limiter.is_allowed("test_key")
        assert self.limiter.is_allowed("test_key")
        assert self.limiter.is_allowed("test_key")
        assert not self.limiter.is_allowed("test_key")  # 4th request should be blocked
    
    def test_rate_limit_window_expiry(self):
        """Test that rate limit resets after window expires"""
        # Make 3 requests
        for _ in range(3):
            assert self.limiter.is_allowed("test_key")
        
        # 4th should be blocked
        assert not self.limiter.is_allowed("test_key")
        
        # Wait for window to expire
        time.sleep(1.1)
        
        # Should be allowed again
        assert self.limiter.is_allowed("test_key")

class TestDataEncryption:
    """Test data encryption functionality"""
    
    def setup_method(self):
        self.encryption = DataEncryption()
    
    def test_encrypt_decrypt(self):
        """Test encryption and decryption"""
        original_data = "sensitive information"
        encrypted = self.encryption.encrypt(original_data)
        decrypted = self.encryption.decrypt(encrypted)
        
        assert decrypted == original_data
        assert encrypted != original_data
    
    def test_encrypt_with_password(self):
        """Test encryption with password"""
        password = "my_secret_password"
        encryption_with_password = DataEncryption(password)
        
        original_data = "sensitive information"
        encrypted = encryption_with_password.encrypt(original_data)
        decrypted = encryption_with_password.decrypt(encrypted)
        
        assert decrypted == original_data
    
    def test_encrypt_empty_string(self):
        """Test encryption of empty string"""
        original_data = ""
        encrypted = self.encryption.encrypt(original_data)
        decrypted = self.encryption.decrypt(encrypted)
        
        assert decrypted == original_data

class TestPerformanceMonitor:
    """Test performance monitoring functionality"""
    
    def setup_method(self):
        self.monitor = PerformanceMonitor()
    
    def test_record_request(self):
        """Test recording request metrics"""
        self.monitor.record_request(1.5, success=True)
        self.monitor.record_request(2.0, success=False)
        
        metrics = self.monitor.get_performance_metrics()
        assert metrics.response_time > 0
        assert metrics.error_rate > 0
    
    def test_is_performance_healthy(self):
        """Test performance health check"""
        is_healthy, message = self.monitor.is_performance_healthy()
        assert isinstance(is_healthy, bool)
        assert isinstance(message, str)
    
    def test_performance_metrics_structure(self):
        """Test performance metrics structure"""
        metrics = self.monitor.get_performance_metrics()
        
        assert hasattr(metrics, 'timestamp')
        assert hasattr(metrics, 'cpu_usage')
        assert hasattr(metrics, 'memory_usage')
        assert hasattr(metrics, 'disk_usage')
        assert hasattr(metrics, 'response_time')
        assert hasattr(metrics, 'throughput')
        assert hasattr(metrics, 'error_rate')

class TestSecurityMonitor:
    """Test security monitoring functionality"""
    
    def setup_method(self):
        self.monitor = SecurityMonitor()
    
    def test_record_security_event(self):
        """Test recording security events"""
        self.monitor.record_security_event(
            "test_event",
            SecurityLevel.MEDIUM,
            "Test security event",
            user_id="test_user",
            ip_address="192.168.1.1"
        )
        
        summary = self.monitor.get_security_summary()
        assert summary['total_events_24h'] > 0
    
    def test_ip_blocking(self):
        """Test IP blocking functionality"""
        # Record failed attempts
        for _ in range(5):
            self.monitor.record_security_event(
                "failed_login",
                SecurityLevel.MEDIUM,
                "Failed login attempt",
                ip_address="192.168.1.100"
            )
        
        # IP should be blocked after 5 failed attempts
        assert self.monitor.is_ip_blocked("192.168.1.100")
    
    def test_security_summary(self):
        """Test security summary structure"""
        summary = self.monitor.get_security_summary()
        
        assert 'total_events_24h' in summary
        assert 'critical_events' in summary
        assert 'blocked_ips' in summary
        assert 'suspicious_ips' in summary
        assert 'failed_attempts' in summary

class TestErrorHandler:
    """Test error handling functionality"""
    
    def setup_method(self):
        self.handler = ErrorHandler()
    
    def test_handle_error(self):
        """Test error handling"""
        error = ValueError("Test error")
        result = self.handler.handle_error(error, "test_context")
        
        # Should return False for unhandled errors
        assert result == False
    
    def test_register_recovery_strategy(self):
        """Test recovery strategy registration"""
        def recovery_strategy(error, context):
            return True
        
        self.handler.register_recovery_strategy("ValueError", recovery_strategy)
        
        error = ValueError("Test error")
        result = self.handler.handle_error(error, "test_context")
        
        # Should return True for handled errors
        assert result == True
    
    def test_error_summary(self):
        """Test error summary"""
        error = ValueError("Test error")
        self.handler.handle_error(error, "test_context")
        
        summary = self.handler.get_error_summary()
        assert summary['total_errors'] > 0
        assert 'ValueError' in summary['error_types']

class TestGuardrailsManager:
    """Test main guardrails manager"""
    
    def setup_method(self):
        self.manager = GuardrailsManager()
    
    def test_validate_input_string(self):
        """Test input validation"""
        is_valid, result = self.manager.validate_input("test string")
        assert is_valid
        assert result == "test string"
    
    def test_validate_input_dict(self):
        """Test dictionary input validation"""
        test_dict = {"key": "value", "number": 123}
        is_valid, result = self.manager.validate_input(test_dict)
        assert is_valid
        assert result == test_dict
    
    def test_validate_input_list(self):
        """Test list input validation"""
        test_list = ["item1", "item2", "item3"]
        is_valid, result = self.manager.validate_input(test_list)
        assert is_valid
        assert result == test_list
    
    def test_check_rate_limit(self):
        """Test rate limiting"""
        assert self.manager.check_rate_limit("test_key")
        assert self.manager.check_rate_limit("test_key")
        assert self.manager.check_rate_limit("test_key")
        # Should still be allowed for API rate limiter (1000 requests)
    
    def test_encrypt_decrypt_sensitive_data(self):
        """Test sensitive data encryption"""
        sensitive_data = "secret_password"
        encrypted = self.manager.encrypt_sensitive_data(sensitive_data)
        decrypted = self.manager.decrypt_sensitive_data(encrypted)
        
        assert decrypted == sensitive_data
    
    def test_record_security_event(self):
        """Test security event recording"""
        self.manager.record_security_event(
            "test_event",
            SecurityLevel.MEDIUM,
            "Test event",
            user_id="test_user"
        )
        
        # Should not raise any exceptions
    
    def test_handle_error(self):
        """Test error handling"""
        error = ValueError("Test error")
        result = self.manager.handle_error(error, "test_context")
        
        # Should return False for unhandled errors
        assert result == False
    
    def test_get_health_status(self):
        """Test health status"""
        status = self.manager.get_health_status()
        
        assert 'overall_health' in status
        assert 'performance' in status
        assert 'security' in status
        assert 'errors' in status

class TestDecorators:
    """Test guardrails decorators"""
    
    def setup_method(self):
        self.manager = GuardrailsManager()
    
    def test_with_guardrails_decorator(self):
        """Test with_guardrails decorator"""
        @with_guardrails(self.manager)
        def test_function(arg1, arg2):
            return arg1 + arg2
        
        result = test_function("hello", "world")
        assert result == "helloworld"
    
    def test_secure_function_decorator(self):
        """Test secure_function decorator"""
        @secure_function(self.manager)
        def test_secure_function(data):
            return f"processed: {data}"
        
        result = test_secure_function("test_data")
        assert result == "processed: test_data"
    
    def test_resource_guard_context_manager(self):
        """Test resource_guard context manager"""
        with resource_guard(self.manager, "test_resource"):
            # Should not raise any exceptions
            pass

class TestIntegration:
    """Integration tests for guardrails system"""
    
    def setup_method(self):
        self.manager = GuardrailsManager()
    
    def test_full_workflow(self):
        """Test complete guardrails workflow"""
        # 1. Validate input
        is_valid, data = self.manager.validate_input("safe_input")
        assert is_valid
        
        # 2. Check rate limit
        assert self.manager.check_rate_limit("test_key")
        
        # 3. Record security event
        self.manager.record_security_event(
            "workflow_event",
            SecurityLevel.INFO,
            "Workflow completed",
            user_id="test_user"
        )
        
        # 4. Handle potential error
        try:
            raise ValueError("Test error")
        except Exception as e:
            result = self.manager.handle_error(e, "workflow")
            assert result == False
        
        # 5. Get health status
        status = self.manager.get_health_status()
        assert isinstance(status, dict)
    
    @pytest.mark.asyncio
    async def test_async_operations(self):
        """Test async operations with guardrails"""
        async def async_function():
            return "async_result"
        
        # Test with async context
        result = await async_function()
        assert result == "async_result"

if __name__ == "__main__":
    pytest.main([__file__]) 