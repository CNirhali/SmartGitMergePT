"""
Guardrails System for SmartGitMergePT
Provides security, performance, error handling, and monitoring capabilities
"""

import asyncio
import functools
import hashlib
import hmac
import html
import ipaddress
import json
import logging
import os
import re
import secrets
import socket
import time
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict, deque, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse, unquote
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
    """Rate limiting implementation with sliding window and memory protection"""
    
    def __init__(self, max_requests: int, window_seconds: int, max_tracked_keys: int = 1000):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_tracked_keys = max_tracked_keys
        self.requests = OrderedDict()
    
    def is_allowed(self, key: str) -> bool:
        now = time.monotonic()
        window_start = now - self.window_seconds
        
        # Get or create deque for this key
        if key in self.requests:
            # Move to end to maintain LRU order (most recently used)
            self.requests.move_to_end(key)
            request_queue = self.requests[key]
        else:
            # Enforce max tracked keys limit to prevent DoS via memory exhaustion
            if len(self.requests) >= self.max_tracked_keys:
                self.requests.popitem(last=False)  # Remove least recently used

            request_queue = deque()
            self.requests[key] = request_queue

        # Clean old requests from the window
        while request_queue and request_queue[0] < window_start:
            request_queue.popleft()
        
        # Check if under limit
        if len(request_queue) < self.max_requests:
            request_queue.append(now)
            return True
        
        return False

class InputValidator:
    """Comprehensive input validation and sanitization"""
    
    def __init__(self):
        # BOLT: Pre-compile and combine patterns for performance
        # Using a single combined regex reduces re.search calls from N to 1
        sensitive_patterns = [
            r'password\s*[:=]\s*\S+',
            r'api_key\s*[:=]\s*\S+',
            r'token\s*[:=]\s*\S+',
            r'secret\s*[:=]\s*\S+',
            r'private_key\s*[:=]\s*\S+',
        ]
        self._sensitive_re = re.compile('|'.join(sensitive_patterns), re.IGNORECASE)
        
        path_traversal_patterns = [
            r'\.\./',
            r'\.\.\\',
            r'%2e%2e%2f',
            r'%2e%2e%5c',
        ]
        self._path_traversal_re = re.compile('|'.join(path_traversal_patterns), re.IGNORECASE)
        
        sql_injection_patterns = [
            r'(\b(union|select|insert|update|delete|drop|create|alter)\b)',
            r'(\b(or|and)\b\s+\d+\s*[=<>])',
            r'(\b(union|select)\b.*\bfrom\b)',
        ]
        self._sql_injection_re = re.compile('|'.join(sql_injection_patterns), re.IGNORECASE)

        # BOLT: Combined security regex for faster all-in-one check
        self._combined_security_re = re.compile('|'.join(
            sensitive_patterns + path_traversal_patterns + sql_injection_patterns
        ), re.IGNORECASE)

        # BOLT: Pre-compile HTML patterns
        # 🛡️ Sentinel: Use innermost matching for tags to support recursive sanitization
        self._html_tag_re = re.compile(r'<[^<>]*+>')
        self._script_tag_re = re.compile(r'<script[^<>]*>.*?</script>', re.IGNORECASE | re.DOTALL)
        # 🛡️ Sentinel: Support optional whitespace within/after dangerous protocols to prevent bypasses (e.g. j a v a s c r i p t : )
        # BOLT: Using a combination of character sets and atomic groups (emulated via lookahead) if supported,
        # but standard possessive quantifiers (\s*+) are available in Python 3.11+.
        # Expanded with file, gopher, php, jar, dict, and ldap to prevent SSRF and other URI-based attacks.
        self._dangerous_protocol_re = re.compile(
            r'(j\s*+a\s*+v\s*+a\s*+s\s*+c\s*+r\s*+i\s*+p\s*+t|v\s*+b\s*+s\s*+c\s*+r\s*+i\s*+p\s*+t|d\s*+a\s*+t\s*+a|'
            r'f\s*+i\s*+l\s*+e|g\s*+o\s*+p\s*+h\s*+e\s*+r|p\s*+h\s*+p|j\s*+a\s*+r|'
            r'd\s*+i\s*+c\s*+t|l\s*+d\s*+a\s*+p)\s*+:',
            re.IGNORECASE
        )
    
    def validate_string(self, value: str, max_length: int = 1000, allow_html: bool = False) -> Tuple[bool, str]:
        """Validate and sanitize string input"""
        if not isinstance(value, str):
            return False, "Value must be a string"
        
        if len(value) > max_length:
            return False, f"String too long (max {max_length} characters)"

        # Check for null bytes to prevent injection/bypass
        if '\0' in value:
            return False, "Null byte detected in input"
        
        # BOLT: Use pre-compiled combined regex for high-performance security check
        if self._combined_security_re.search(value):
            # If hit, do individual checks to return specific error messages
            if self._sensitive_re.search(value):
                return False, "Sensitive data detected in input"
            if self._path_traversal_re.search(value):
                return False, "Path traversal attempt detected"
            if self._sql_injection_re.search(value):
                return False, "SQL injection attempt detected"
        
        # Sanitize HTML if not allowed
        if not allow_html:
            value = self._sanitize_html(value)
        
        return True, value
    
    def validate_path(self, path: str, base_path: str) -> Tuple[bool, str]:
        """Validate file path for security"""
        try:
            # Normalize paths and resolve symlinks using realpath
            normalized_path = os.path.realpath(path)
            normalized_base = os.path.realpath(base_path)
            
            # Check if resolved path is within base directory
            if os.path.commonpath([normalized_base, normalized_path]) != normalized_base:
                return False, "Path traversal attempt detected"
            
            # Check for dangerous file extensions
            dangerous_extensions = {'.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js'}
            if any(normalized_path.lower().endswith(ext) for ext in dangerous_extensions):
                return False, "Dangerous file extension detected"
            
            return True, normalized_path
        except Exception as e:
            return False, f"Path validation error: {str(e)}"
    
    def validate_url(self, url: str) -> Tuple[bool, str]:
        """Validate URL for security (SSRF and protocol bypass protection)"""
        try:
            if not isinstance(url, str):
                return False, "URL must be a string"

            # 🛡️ Sentinel: Strip whitespace and check for null bytes to prevent bypasses
            url = url.strip()
            if '\0' in url:
                return False, "Null byte detected in URL"

            # 🛡️ Sentinel: Use pre-compiled regex for robust protocol detection (e.g. j a v a s c r i p t :)
            if self._dangerous_protocol_re.search(url):
                return False, "Dangerous URL protocol detected"

            parsed = urlparse(url)
            
            # Check for dangerous protocols via parsed scheme as a second layer
            dangerous_protocols = {'file', 'javascript', 'data', 'vbscript', 'gopher', 'dict', 'ldap', 'ftp', 'tftp', 'php', 'jar'}
            if parsed.scheme.lower() in dangerous_protocols:
                return False, f"Dangerous URL protocol detected: {parsed.scheme}"

            hostname = parsed.hostname
            if not hostname:
                return False, "URL must have a hostname"
            
            # Normalize hostname: unquote and strip trailing dots (e.g. localhost.)
            normalized_hostname = unquote(hostname).lower().rstrip('.')

            # Check for literal loopback/private hostnames
            if normalized_hostname in {'localhost', 'loopback', 'localhost.localdomain'}:
                return False, "Local URL not allowed"

            # Handle IP-based hostnames for robust SSRF protection
            try:
                # Remove brackets from IPv6 literal if present
                # 🛡️ Sentinel: Use normalized_hostname for IP checks to prevent bypasses
                ip_str = normalized_hostname.strip('[]')
                ip = ipaddress.ip_address(ip_str)
                if self._is_internal_ip(ip):
                    return False, f"Internal IP not allowed: {ip}"

            except ValueError:
                # 🛡️ Sentinel: Check for integer-based IPv4 (decimal, hex, octal) for robust SSRF protection
                try:
                    # If it looks like a number (including 0x prefix), try parsing as integer
                    if normalized_hostname.isdigit() or normalized_hostname.startswith('0x'):
                        ip_int = int(normalized_hostname, 0)
                        if 0 <= ip_int <= 0xFFFFFFFF:
                            ip = ipaddress.IPv4Address(ip_int)
                            if self._is_internal_ip(ip):
                                return False, f"Internal IP (integer) not allowed: {ip}"
                except (ValueError, OverflowError):
                    pass

                # Hostname is not a valid IP address, check for shorthand IP notation (e.g. 127.1)
                try:
                    # 🛡️ Sentinel: Use normalized_hostname for shorthand IP checks
                    packed_ip = socket.inet_aton(normalized_hostname)
                    ip_str = socket.inet_ntoa(packed_ip)
                    ip = ipaddress.ip_address(ip_str)
                    if self._is_internal_ip(ip):
                        return False, f"Internal IP (shorthand) not allowed: {ip}"
                except (socket.error, ValueError):
                    # Hostname is truly not an IP or invalid IP
                    pass
            
            return True, url
        except Exception as e:
            return False, f"URL validation error: {str(e)}"

    def _is_internal_ip(self, ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address]) -> bool:
        """Check if an IP address belongs to an internal or reserved range, including IPv4-mapped IPv6"""
        # 🛡️ Sentinel: Check for IPv4-mapped IPv6 address (e.g., ::ffff:127.0.0.1)
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            return self._is_internal_ip(ip.ipv4_mapped)

        return (
            ip.is_loopback or
            ip.is_private or
            ip.is_link_local or
            ip.is_multicast or
            ip.is_reserved or
            ip.is_unspecified
        )
    
    def _sanitize_html(self, text: str) -> str:
        """Robust recursive HTML sanitization to prevent nested tag/protocol bypasses"""
        # BOLT: O(N) fast-path check to avoid expensive regex substitutions if safe
        if '<' not in text:
            # Check for dangerous protocols only if a colon is present
            if ':' not in text:
                return html.escape(text)

            # BOLT: Faster O(N) check for potential dangerous protocol start characters
            # Only hit regex if text contains characters from dangerous protocols (ignoring case)
            # Expanded set: j, v, d, f, g, p, l (from javascript, vbscript, data, file, gopher, php, jar, dict, ldap)
            # This avoids expensive regex engine overhead for most strings with colons.
            # 🛡️ Sentinel: We must still proceed to regex if we want to strip tags later,
            # but since we are inside `if '<' not in text:`, we know there are no tags.
            # BOLT: Using a fast-path regex for protocol start characters is ~10-20x faster
            # than creating a full lowercase copy and iterating with 'any()'.
            # 🛡️ Sentinel: 'h' is omitted to avoid triggering on all 'http' URLs while maintaining security.
            if not re.search(r'[jvdgfplJVDGFPL]', text):
                return html.escape(text)

            # 🛡️ Sentinel: Use the pre-compiled regex for fast-path check as well
            if not self._dangerous_protocol_re.search(text):
                # If no tags and no dangerous protocols, we only need html.escape for safety
                return html.escape(text)

        # 🛡️ Sentinel: Recursive sanitization to handle bypasses like <scr<script>ipt>
        max_iterations = 5
        for _ in range(max_iterations):
            original_text = text
            # BOLT: Use pre-compiled regex for better performance
            # 🛡️ Sentinel: Remove script tags first to capture content
            text = self._script_tag_re.sub('', text)
            # Remove common HTML tag patterns
            text = self._html_tag_re.sub('', text)
            # Remove dangerous protocol URLs
            text = self._dangerous_protocol_re.sub('', text)

            if text == original_text:
                break

        # BOLT: Check if any potentially dangerous characters remain that need escaping
        # If the string was already sanitized or is safe, html.escape might be redundant
        # but is kept for defense-in-depth as per security guidelines.
        return html.escape(text)

def ensure_private_file(path: Union[str, Path]):
    """Ensure a file exists and has restrictive (0o600) permissions"""
    path = Path(path)
    if not path.exists():
        path.touch()
    os.chmod(path, 0o600)

def ensure_private_dir(path: Union[str, Path]):
    """Ensure a directory exists and has restrictive (0o700) permissions"""
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)

class DataEncryption:
    """Data encryption and decryption utilities with versioned random salts"""
    
    def __init__(self, key: Optional[str] = None):
        self.password = key
        self.legacy_salt = b'smartgit_salt_2024'
        if not key:
            self.static_key = Fernet.generate_key()
            self.cipher = Fernet(self.static_key)
    
    def _derive_key(self, password: str, salt: bytes, iterations: int = 600000) -> bytes:
        """Derive encryption key from password and salt"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data with version prefix and random salt for password keys"""
        try:
            if self.password:
                salt = os.urandom(16)
                key = self._derive_key(self.password, salt)
                cipher = Fernet(key)
                # Decode Fernet token to raw bytes to avoid double base64 encoding
                raw_encrypted = base64.urlsafe_b64decode(cipher.encrypt(data.encode()))
                return base64.urlsafe_b64encode(b'\x01' + salt + raw_encrypted).decode()
            return self.cipher.encrypt(data.encode()).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data, supporting versioned salts and legacy formats"""
        try:
            if self.password:
                decoded = base64.urlsafe_b64decode(encrypted_data.encode())
                if len(decoded) > 17 and decoded[0] == 1:  # Version 1: [v1][salt][data]
                    salt, raw_data = decoded[1:17], decoded[17:]
                    key = self._derive_key(self.password, salt)
                    return Fernet(key).decrypt(base64.urlsafe_b64encode(raw_data)).decode()
                # Legacy fallback: try old format (static salt, iterations=100k)
                key = self._derive_key(self.password, self.legacy_salt, iterations=100000)
                try: return Fernet(key).decrypt(decoded).decode()
                except: return Fernet(key).decrypt(base64.urlsafe_b64decode(decoded)).decode()
            try: return self.cipher.decrypt(encrypted_data.encode()).decode()
            except: return self.cipher.decrypt(base64.urlsafe_b64decode(encrypted_data.encode())).decode()
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
    """Security monitoring and threat detection with resource limits"""
    
    def __init__(self, max_tracked_ips: int = 1000):
        # 🛡️ Sentinel: Use bounded deque to prevent memory exhaustion DoS
        self.security_events = deque(maxlen=1000)
        # 🛡️ Sentinel: Use bounded OrderedDict for IP tracking to prevent memory exhaustion
        self.max_tracked_ips = max_tracked_ips
        self.suspicious_ips = OrderedDict()
        self.failed_attempts = OrderedDict()
        self.blocked_ips = OrderedDict()
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
        """Check for suspicious activity patterns with memory protection"""
        # Track failed attempts
        if "failed" in event.event_type.lower():
            # Bounded update for failed_attempts
            if ip_address in self.failed_attempts:
                self.failed_attempts.move_to_end(ip_address)
                self.failed_attempts[ip_address] += 1
            else:
                if len(self.failed_attempts) >= self.max_tracked_ips:
                    self.failed_attempts.popitem(last=False)
                self.failed_attempts[ip_address] = 1
            
            # Block IP after 5 failed attempts
            if self.failed_attempts[ip_address] >= 5:
                if ip_address in self.blocked_ips:
                    self.blocked_ips.move_to_end(ip_address)
                else:
                    if len(self.blocked_ips) >= self.max_tracked_ips:
                        self.blocked_ips.popitem(last=False)
                    self.blocked_ips[ip_address] = True
                logger.warning(f"IP {ip_address} blocked due to multiple failed attempts")
        
        # Check rate limiting
        if not self.rate_limiter.is_allowed(ip_address):
            if ip_address in self.suspicious_ips:
                self.suspicious_ips.move_to_end(ip_address)
            else:
                if len(self.suspicious_ips) >= self.max_tracked_ips:
                    self.suspicious_ips.popitem(last=False)
                self.suspicious_ips[ip_address] = True
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
        """Get error summary with sensitive information (tracebacks) removed"""
        recent_errors = []
        for error in list(self.error_history)[-10:]:
            # Create a shallow copy and remove the traceback for the summary report
            safe_error = error.copy()
            if "traceback" in safe_error:
                del safe_error["traceback"]
            recent_errors.append(safe_error)

        return {
            "total_errors": sum(self.error_counts.values()),
            "error_types": dict(self.error_counts),
            "recent_errors": recent_errors
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
    
    def validate_input(self, data: Any, input_type: str = "general",
                      _visited: Optional[Set[int]] = None, _depth: int = 0) -> Tuple[bool, Any]:
        """Validate input data with recursion and depth protection"""
        if _depth > 20:
            return False, "Input nesting too deep (max 20)"

        # Check for circular references in collection types
        is_collection = isinstance(data, (dict, list, tuple, set))
        if is_collection:
            if _visited is None:
                _visited = set()

            if id(data) in _visited:
                return False, "Circular reference detected in input"

            _visited.add(id(data))

        try:
            if isinstance(data, str):
                return self.input_validator.validate_string(data)
            elif isinstance(data, dict):
                return self._validate_dict(data, _visited, _depth + 1)
            elif isinstance(data, list):
                return self._validate_list(data, _visited, _depth + 1)
            elif isinstance(data, tuple):
                return self._validate_tuple(data, _visited, _depth + 1)
            elif isinstance(data, set):
                return self._validate_set(data, _visited, _depth + 1)
            else:
                return True, data
        finally:
            # Remove from visited set after processing to allow Directed Acyclic Graphs (DAGs)
            if is_collection and _visited is not None:
                _visited.remove(id(data))
    
    def _validate_dict(self, data: Dict[str, Any], visited: Set[int], depth: int) -> Tuple[bool, Dict[str, Any]]:
        """Validate dictionary input"""
        validated_data = {}
        
        for key, value in data.items():
            # Validate key
            is_valid_key, clean_key = self.input_validator.validate_string(key, max_length=100)
            if not is_valid_key:
                return False, f"Invalid key: {key}"
            
            # Validate value
            is_valid_value, clean_value = self.validate_input(value, _visited=visited, _depth=depth)
            if not is_valid_value:
                return False, f"Invalid value for key {key}: {clean_value}"
            
            validated_data[clean_key] = clean_value
        
        return True, validated_data
    
    def _validate_list(self, data: List[Any], visited: Set[int], depth: int) -> Tuple[bool, List[Any]]:
        """Validate list input"""
        validated_data = []
        
        for i, item in enumerate(data):
            is_valid, clean_item = self.validate_input(item, _visited=visited, _depth=depth)
            if not is_valid:
                return False, f"Invalid item at index {i}: {clean_item}"
            validated_data.append(clean_item)
        
        return True, validated_data

    def _validate_tuple(self, data: Tuple[Any, ...], visited: Set[int], depth: int) -> Tuple[bool, Tuple[Any, ...]]:
        """Validate tuple input"""
        validated_data = []

        for i, item in enumerate(data):
            is_valid, clean_item = self.validate_input(item, _visited=visited, _depth=depth)
            if not is_valid:
                return False, f"Invalid item in tuple at index {i}: {clean_item}"
            validated_data.append(clean_item)

        return True, tuple(validated_data)

    def _validate_set(self, data: Set[Any], visited: Set[int], depth: int) -> Tuple[bool, Set[Any]]:
        """Validate set input"""
        validated_data = set()

        for item in data:
            is_valid, clean_item = self.validate_input(item, _visited=visited, _depth=depth)
            if not is_valid:
                return False, f"Invalid item in set: {clean_item}"
            validated_data.add(clean_item)

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
                new_args = []
                for arg in args:
                    is_valid, validated_arg = guardrails.validate_input(arg)
                    if not is_valid:
                        raise ValueError(f"Invalid input: {validated_arg}")
                    new_args.append(validated_arg)
                args = tuple(new_args)
                
                new_kwargs = {}
                for key, value in kwargs.items():
                    is_valid, validated_value = guardrails.validate_input(value)
                    if not is_valid:
                        raise ValueError(f"Invalid input for {key}: {validated_value}")
                    new_kwargs[key] = validated_value
                kwargs = new_kwargs
            
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