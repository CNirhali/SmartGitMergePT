"""
Guardrails System for SmartGitMergePT
Provides security, performance, error handling, and monitoring capabilities
"""

import asyncio
import functools
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse
import weakref

import psutil
import requests
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PerformanceLevel(Enum):
    OPTIMIZED = "optimized"
    NORMAL = "normal"
    DEGRADED = "degraded"
    CRITICAL = "critical"

@dataclass
class SecurityEvent:
    timestamp: datetime
    event_type: str
    severity: SecurityLevel
    description: str
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    response_time: float
    throughput: float
    error_rate: float

class RateLimiter:
    """Rate limiting implementation with sliding window"""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)
    
    def is_allowed(self, key: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        while self.requests[key] and self.requests[key][0] < window_start:
            self.requests[key].popleft()
        
        # Check if under limit
        if len(self.requests[key]) < self.max_requests:
            self.requests[key].append(now)
            return True
        
        return False

class InputValidator:
    """Comprehensive input validation and sanitization"""
    
    def __init__(self):
        self.sensitive_patterns = [
            r'password\s*[:=]\s*\S+',
            r'api_key\s*[:=]\s*\S+',
            r'token\s*[:=]\s*\S+',
            r'secret\s*[:=]\s*\S+',
            r'private_key\s*[:=]\s*\S+',
        ]
        
        self.path_traversal_patterns = [
            r'\.\./',
            r'\.\.\\',
            r'%2e%2e%2f',
            r'%2e%2e%5c',
        ]
        
        self.sql_injection_patterns = [
            r'(\b(union|select|insert|update|delete|drop|create|alter)\b)',
            r'(\b(or|and)\b\s+\d+\s*[=<>])',
            r'(\b(union|select)\b.*\bfrom\b)',
        ]
    
    def validate_string(self, value: str, max_length: int = 1000, allow_html: bool = False) -> Tuple[bool, str]:
        """Validate and sanitize string input"""
        if not isinstance(value, str):
            return False, "Value must be a string"
        
        if len(value) > max_length:
            return False, f"String too long (max {max_length} characters)"
        
        # Check for sensitive data
        for pattern in self.sensitive_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False, "Sensitive data detected in input"
        
        # Check for path traversal
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False, "Path traversal attempt detected"
        
        # Check for SQL injection
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False, "SQL injection attempt detected"
        
        # Sanitize HTML if not allowed
        if not allow_html:
            value = self._sanitize_html(value)
        
        return True, value
    
    def validate_path(self, path: str, base_path: str) -> Tuple[bool, str]:
        """Validate file path for security"""
        try:
            # Normalize paths
            normalized_path = os.path.normpath(path)
            normalized_base = os.path.normpath(base_path)
            
            # Check if path is within base directory
            if os.path.commonpath([os.path.abspath(normalized_base), os.path.abspath(normalized_path)]) != os.path.abspath(normalized_base):
                return False, "Path traversal attempt detected"
            
            # Check for dangerous file extensions
            dangerous_extensions = {'.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js'}
            if any(normalized_path.lower().endswith(ext) for ext in dangerous_extensions):
                return False, "Dangerous file extension detected"
            
            return True, normalized_path
        except Exception as e:
            return False, f"Path validation error: {str(e)}"
    
    def validate_url(self, url: str) -> Tuple[bool, str]:
        """Validate URL for security"""
        try:
            parsed = urlparse(url)
            
            # Check for dangerous protocols
            dangerous_protocols = {'file', 'javascript', 'data', 'vbscript'}
            if parsed.scheme.lower() in dangerous_protocols:
                return False, "Dangerous URL protocol detected"
            
            # Check for localhost/private IPs if not allowed
            if parsed.hostname in {'localhost', '127.0.0.1', '::1'}:
                return False, "Local URL not allowed"
            
            return True, url
        except Exception as e:
            return False, f"URL validation error: {str(e)}"
    
    def _sanitize_html(self, text: str) -> str:
        """Basic HTML sanitization"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove script tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        # Remove javascript: URLs
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        return text

class DataEncryption:
    """Data encryption and decryption utilities"""
    
    def __init__(self, key: Optional[str] = None):
        if key:
            self.key = self._derive_key(key)
        else:
            self.key = Fernet.generate_key()
        
        self.cipher = Fernet(self.key)
    
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password"""
        salt = b'smartgit_salt_2024'  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data"""
        try:
            encrypted = self.cipher.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

class PerformanceMonitor:
    """Performance monitoring and optimization"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.start_time = time.time()
        self.request_times = deque(maxlen=100)
        self.error_counts = defaultdict(int)
        self.cache_hits = 0
        self.cache_misses = 0
    
    def record_request(self, duration: float, success: bool = True):
        """Record request performance metrics"""
        self.request_times.append(duration)
        
        if not success:
            self.error_counts[datetime.now().strftime('%Y-%m-%d %H:%M')] += 1
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        avg_response_time = sum(self.request_times) / len(self.request_times) if self.request_times else 0
        throughput = len(self.request_times) / 60 if self.request_times else 0  # requests per minute
        
        # Calculate error rate
        total_requests = len(self.request_times)
        total_errors = sum(self.error_counts.values())
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            response_time=avg_response_time,
            throughput=throughput,
            error_rate=error_rate
        )
    
    def is_performance_healthy(self) -> Tuple[bool, str]:
        """Check if system performance is healthy"""
        metrics = self.get_performance_metrics()
        
        if metrics.cpu_usage > 90:
            return False, f"High CPU usage: {metrics.cpu_usage}%"
        
        if metrics.memory_usage > 90:
            return False, f"High memory usage: {metrics.memory_usage}%"
        
        if metrics.disk_usage > 95:
            return False, f"High disk usage: {metrics.disk_usage}%"
        
        if metrics.error_rate > 10:
            return False, f"High error rate: {metrics.error_rate}%"
        
        return True, "Performance is healthy"

class SecurityMonitor:
    """Security monitoring and threat detection"""
    
    def __init__(self):
        self.security_events: List[SecurityEvent] = []
        self.suspicious_ips: Set[str] = set()
        self.failed_attempts: Dict[str, int] = defaultdict(int)
        self.blocked_ips: Set[str] = set()
        self.rate_limiter = RateLimiter(max_requests=100, window_seconds=3600)
    
    def record_security_event(self, event_type: str, severity: SecurityLevel, 
                            description: str, user_id: Optional[str] = None,
                            ip_address: Optional[str] = None, details: Dict[str, Any] = None):
        """Record a security event"""
        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            description=description,
            user_id=user_id,
            ip_address=ip_address,
            details=details or {}
        )
        
        self.security_events.append(event)
        
        # Check for suspicious patterns
        if ip_address:
            self._check_suspicious_activity(ip_address, event)
        
        # Log critical events
        if severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            logger.warning(f"SECURITY ALERT: {event_type} - {description}")
    
    def _check_suspicious_activity(self, ip_address: str, event: SecurityEvent):
        """Check for suspicious activity patterns"""
        # Track failed attempts
        if "failed" in event.event_type.lower():
            self.failed_attempts[ip_address] += 1
            
            # Block IP after 5 failed attempts
            if self.failed_attempts[ip_address] >= 5:
                self.blocked_ips.add(ip_address)
                logger.warning(f"IP {ip_address} blocked due to multiple failed attempts")
        
        # Check rate limiting
        if not self.rate_limiter.is_allowed(ip_address):
            self.suspicious_ips.add(ip_address)
            logger.warning(f"Rate limit exceeded for IP {ip_address}")
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP is blocked"""
        return ip_address in self.blocked_ips
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary"""
        recent_events = [e for e in self.security_events 
                        if e.timestamp > datetime.now() - timedelta(hours=24)]
        
        return {
            "total_events_24h": len(recent_events),
            "critical_events": len([e for e in recent_events if e.severity == SecurityLevel.CRITICAL]),
            "blocked_ips": len(self.blocked_ips),
            "suspicious_ips": len(self.suspicious_ips),
            "failed_attempts": sum(self.failed_attempts.values())
        }

class ErrorHandler:
    """Comprehensive error handling and recovery"""
    
    def __init__(self):
        self.error_counts = defaultdict(int)
        self.error_history = deque(maxlen=1000)
        self.recovery_strategies: Dict[str, Callable] = {}
    
    def handle_error(self, error: Exception, context: str = "", 
                    recovery_strategy: Optional[Callable] = None) -> bool:
        """Handle an error with optional recovery strategy"""
        error_type = type(error).__name__
        self.error_counts[error_type] += 1
        
        error_info = {
            "timestamp": datetime.now(),
            "error_type": error_type,
            "error_message": str(error),
            "context": context,
            "traceback": traceback.format_exc()
        }
        
        self.error_history.append(error_info)
        
        # Log error
        logger.error(f"Error in {context}: {error_type} - {error}")
        
        # Determine which recovery strategy to use
        strategy = recovery_strategy or self.recovery_strategies.get(error_type)

        # Try recovery strategy
        if strategy:
            try:
                return strategy(error, context)
            except Exception as recovery_error:
                logger.error(f"Recovery strategy failed: {recovery_error}")
                return False
        
        return False
    
    def register_recovery_strategy(self, error_type: str, strategy: Callable):
        """Register a recovery strategy for a specific error type"""
        self.recovery_strategies[error_type] = strategy
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary"""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_types": dict(self.error_counts),
            "recent_errors": list(self.error_history)[-10:]
        }

class GuardrailsManager:
    """Main guardrails manager that coordinates all security and performance features"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.input_validator = InputValidator()
        self.performance_monitor = PerformanceMonitor()
        self.security_monitor = SecurityMonitor()
        self.error_handler = ErrorHandler()
        self.encryption = DataEncryption()
        
        # Initialize rate limiters
        self.api_rate_limiter = RateLimiter(max_requests=1000, window_seconds=3600)
        self.auth_rate_limiter = RateLimiter(max_requests=5, window_seconds=300)
        
        # Setup monitoring
        self._setup_monitoring()
    
    def _setup_monitoring(self):
        """Setup monitoring and recovery strategies"""
        # Register recovery strategies
        self.error_handler.register_recovery_strategy("ConnectionError", self._recover_connection)
        self.error_handler.register_recovery_strategy("TimeoutError", self._recover_timeout)
        self.error_handler.register_recovery_strategy("PermissionError", self._recover_permission)
    
    def validate_input(self, data: Any, input_type: str = "general") -> Tuple[bool, Any]:
        """Validate input data"""
        if isinstance(data, str):
            return self.input_validator.validate_string(data)
        elif isinstance(data, dict):
            return self._validate_dict(data)
        elif isinstance(data, list):
            return self._validate_list(data)
        else:
            return True, data
    
    def _validate_dict(self, data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Validate dictionary input"""
        validated_data = {}
        
        for key, value in data.items():
            # Validate key
            is_valid_key, clean_key = self.input_validator.validate_string(key, max_length=100)
            if not is_valid_key:
                return False, f"Invalid key: {key}"
            
            # Validate value
            is_valid_value, clean_value = self.validate_input(value)
            if not is_valid_value:
                return False, f"Invalid value for key {key}: {clean_value}"
            
            validated_data[clean_key] = clean_value
        
        return True, validated_data
    
    def _validate_list(self, data: List[Any]) -> Tuple[bool, List[Any]]:
        """Validate list input"""
        validated_data = []
        
        for i, item in enumerate(data):
            is_valid, clean_item = self.validate_input(item)
            if not is_valid:
                return False, f"Invalid item at index {i}: {clean_item}"
            validated_data.append(clean_item)
        
        return True, validated_data
    
    def check_rate_limit(self, key: str, limiter_type: str = "api") -> bool:
        """Check rate limiting"""
        if limiter_type == "auth":
            return self.auth_rate_limiter.is_allowed(key)
        else:
            return self.api_rate_limiter.is_allowed(key)
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.encryption.encrypt(data)
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.encryption.decrypt(encrypted_data)
    
    def record_security_event(self, event_type: str, severity: SecurityLevel,
                            description: str, **kwargs):
        """Record security event"""
        self.security_monitor.record_security_event(
            event_type, severity, description, **kwargs
        )
    
    def handle_error(self, error: Exception, context: str = "") -> bool:
        """Handle error with recovery"""
        return self.error_handler.handle_error(error, context)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        perf_healthy, perf_message = self.performance_monitor.is_performance_healthy()
        security_summary = self.security_monitor.get_security_summary()
        error_summary = self.error_handler.get_error_summary()
        
        return {
            "overall_health": perf_healthy and security_summary["critical_events"] == 0,
            "performance": {
                "healthy": perf_healthy,
                "message": perf_message,
                "metrics": self.performance_monitor.get_performance_metrics().__dict__
            },
            "security": security_summary,
            "errors": error_summary
        }
    
    def _recover_connection(self, error: Exception, context: str) -> bool:
        """Recovery strategy for connection errors"""
        logger.info(f"Attempting connection recovery for {context}")
        time.sleep(1)  # Wait before retry
        return True
    
    def _recover_timeout(self, error: Exception, context: str) -> bool:
        """Recovery strategy for timeout errors"""
        logger.info(f"Attempting timeout recovery for {context}")
        time.sleep(2)  # Wait longer for timeouts
        return True
    
    def _recover_permission(self, error: Exception, context: str) -> bool:
        """Recovery strategy for permission errors"""
        logger.warning(f"Permission error in {context}, cannot auto-recover")
        return False

# Decorators for easy guardrails integration
def with_guardrails(guardrails: GuardrailsManager, input_validation: bool = True,
                   rate_limiting: bool = True, error_handling: bool = True):
    """Decorator to add guardrails to functions"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Input validation
            if input_validation:
                for arg in args:
                    is_valid, validated_arg = guardrails.validate_input(arg)
                    if not is_valid:
                        raise ValueError(f"Invalid input: {validated_arg}")
                
                for key, value in kwargs.items():
                    is_valid, validated_value = guardrails.validate_input(value)
                    if not is_valid:
                        raise ValueError(f"Invalid input for {key}: {validated_value}")
            
            # Rate limiting
            if rate_limiting:
                # Use function name as rate limit key
                if not guardrails.check_rate_limit(func.__name__):
                    raise Exception("Rate limit exceeded")
            
            # Execute function with error handling
            if error_handling:
                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    # Record performance metrics
                    guardrails.performance_monitor.record_request(duration, success=True)
                    
                    return result
                except Exception as e:
                    guardrails.performance_monitor.record_request(0, success=False)
                    guardrails.handle_error(e, f"Function {func.__name__}")
                    raise
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

def secure_function(guardrails: GuardrailsManager):
    """Decorator for functions that handle sensitive data"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Record security event
            guardrails.record_security_event(
                "function_call",
                SecurityLevel.MEDIUM,
                f"Calling secure function: {func.__name__}",
                details={"function": func.__name__}
            )
            
            # Execute with full guardrails
            return with_guardrails(guardrails, True, True, True)(func)(*args, **kwargs)
        
        return wrapper
    return decorator

# Context manager for resource management
@contextmanager
def resource_guard(guardrails: GuardrailsManager, resource_name: str):
    """Context manager for resource management with guardrails"""
    start_time = time.time()
    
    try:
        yield
    except Exception as e:
        guardrails.handle_error(e, f"Resource: {resource_name}")
        raise
    finally:
        duration = time.time() - start_time
        guardrails.performance_monitor.record_request(duration, success=True) 