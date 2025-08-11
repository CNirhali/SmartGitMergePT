# Guardrails and Optimization Systems for SmartGitMergePT

This document provides comprehensive documentation for the guardrails and optimization systems that have been integrated into the SmartGitMergePT project to enhance security, performance, and reliability.

## Table of Contents

1. [Overview](#overview)
2. [Guardrails System](#guardrails-system)
3. [Optimization System](#optimization-system)
4. [Monitoring System](#monitoring-system)
5. [Integration Guide](#integration-guide)
6. [Configuration](#configuration)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)

## Overview

The SmartGitMergePT project now includes three major systems for enhanced security, performance, and monitoring:

- **Guardrails System**: Comprehensive security, input validation, and error handling
- **Optimization System**: Caching, async operations, and performance improvements
- **Monitoring System**: Health checks, metrics collection, and alerting

## Guardrails System

### Features

The guardrails system provides:

- **Input Validation**: Comprehensive validation and sanitization of all inputs
- **Rate Limiting**: Sliding window rate limiting for API and authentication
- **Data Encryption**: Automatic encryption of sensitive data
- **Security Monitoring**: Real-time security event tracking and threat detection
- **Error Handling**: Comprehensive error handling with recovery strategies
- **Performance Monitoring**: System performance tracking and optimization

### Components

#### InputValidator
Validates and sanitizes input data to prevent security vulnerabilities:

```python
from guardrails import InputValidator

validator = InputValidator()

# Validate string input
is_valid, result = validator.validate_string("safe_input")
if is_valid:
    print(f"Valid input: {result}")

# Validate file path
is_valid, path = validator.validate_path("/safe/file.txt", "/safe")
if is_valid:
    print(f"Safe path: {path}")

# Validate URL
is_valid, url = validator.validate_url("https://example.com")
if is_valid:
    print(f"Safe URL: {url}")
```

#### RateLimiter
Implements sliding window rate limiting:

```python
from guardrails import RateLimiter

# API rate limiter: 1000 requests per hour
api_limiter = RateLimiter(max_requests=1000, window_seconds=3600)

# Auth rate limiter: 5 attempts per 5 minutes
auth_limiter = RateLimiter(max_requests=5, window_seconds=300)

if api_limiter.is_allowed("user123"):
    # Process request
    pass
else:
    # Rate limit exceeded
    pass
```

#### DataEncryption
Provides automatic encryption of sensitive data:

```python
from guardrails import DataEncryption

encryption = DataEncryption()

# Encrypt sensitive data
encrypted = encryption.encrypt("sensitive_password")
decrypted = encryption.decrypt(encrypted)

assert decrypted == "sensitive_password"
```

#### SecurityMonitor
Monitors security events and detects threats:

```python
from guardrails import SecurityMonitor, SecurityLevel

monitor = SecurityMonitor()

# Record security event
monitor.record_security_event(
    "failed_login",
    SecurityLevel.WARNING,
    "Failed login attempt",
    user_id="user123",
    ip_address="192.168.1.100"
)

# Check if IP is blocked
if monitor.is_ip_blocked("192.168.1.100"):
    print("IP is blocked")

# Get security summary
summary = monitor.get_security_summary()
print(f"Critical events: {summary['critical_events']}")
```

#### ErrorHandler
Provides comprehensive error handling with recovery strategies:

```python
from guardrails import ErrorHandler

handler = ErrorHandler()

# Register recovery strategy
def recover_connection(error, context):
    print(f"Attempting to recover connection in {context}")
    return True

handler.register_recovery_strategy("ConnectionError", recover_connection)

# Handle error
try:
    # Some operation that might fail
    raise ConnectionError("Connection lost")
except Exception as e:
    success = handler.handle_error(e, "database_operation")
    if success:
        print("Error recovered successfully")
```

#### GuardrailsManager
Main coordinator for all guardrails functionality:

```python
from guardrails import GuardrailsManager, SecurityLevel

manager = GuardrailsManager()

# Validate input
is_valid, data = manager.validate_input({"key": "value"})
if is_valid:
    print("Input is valid")

# Check rate limit
if manager.check_rate_limit("user123"):
    print("Request allowed")

# Encrypt sensitive data
encrypted = manager.encrypt_sensitive_data("secret")
decrypted = manager.decrypt_sensitive_data(encrypted)

# Record security event
manager.record_security_event(
    "api_call",
    SecurityLevel.INFO,
    "API endpoint accessed",
    user_id="user123"
)

# Get health status
status = manager.get_health_status()
print(f"Overall health: {status['overall_health']}")
```

### Decorators

The guardrails system provides convenient decorators for easy integration:

```python
from guardrails import with_guardrails, secure_function, resource_guard

# Add guardrails to any function
@with_guardrails(manager)
def process_data(data):
    return data.upper()

# Secure function with full protection
@secure_function(manager)
def handle_sensitive_data(data):
    return process_sensitive_data(data)

# Resource management with guardrails
with resource_guard(manager, "database_connection"):
    # Database operations
    pass
```

## Optimization System

### Features

The optimization system provides:

- **Smart Caching**: Multiple caching strategies with compression and persistence
- **Async Operations**: Thread and process pool management for concurrent operations
- **Resource Management**: Memory and CPU monitoring with automatic optimization
- **Performance Tracking**: Detailed performance metrics and optimization statistics

### Components

#### SmartCache
Intelligent caching with multiple strategies:

```python
from optimizer import SmartCache, CacheConfig, CacheStrategy

# TTL cache with compression
config = CacheConfig(
    strategy=CacheStrategy.TTL,
    max_size=1000,
    ttl_seconds=3600,
    enable_compression=True
)
cache = SmartCache(config)

# Store and retrieve data
cache.set("key", "value")
value = cache.get("key")

# Get cache statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2f}%")
```

#### AsyncTaskManager
Manages concurrent operations:

```python
from optimizer import AsyncTaskManager
import asyncio

manager = AsyncTaskManager(max_workers=4, max_processes=2)

# Run function in thread pool
async def example():
    result = await manager.run_in_thread(slow_function, arg1, arg2)
    return result

# Run function in process pool
result = await manager.run_in_process(cpu_intensive_function, arg1, arg2)

# Run multiple tasks concurrently
tasks = [task1, task2, task3]
results = await manager.run_concurrent(tasks)
```

#### ResourceManager
Monitors and optimizes system resources:

```python
from optimizer import ResourceManager

manager = ResourceManager(memory_limit_mb=512, cpu_limit_percent=80)

# Check resource usage
memory_healthy, memory_usage = manager.check_memory_usage()
cpu_healthy, cpu_usage = manager.check_cpu_usage()

if not memory_healthy or not cpu_healthy:
    manager.optimize_memory()

# Get resource statistics
stats = manager.get_resource_stats()
print(f"Memory usage: {stats['memory']['usage_mb']:.1f} MB")
print(f"CPU usage: {stats['cpu']['usage_percent']:.1f}%")
```

#### PerformanceOptimizer
Main optimization coordinator:

```python
from optimizer import PerformanceOptimizer, CacheConfig, PerformanceConfig

cache_config = CacheConfig()
perf_config = PerformanceConfig()
optimizer = PerformanceOptimizer(cache_config, perf_config)

# Cached function decorator
@optimizer.cached_function(ttl_seconds=3600)
def expensive_function(x, y):
    # Expensive computation
    return x + y

# Async function decorator
@optimizer.async_function()
async def async_function(x, y):
    await asyncio.sleep(0.1)
    return x * y

# Resource optimized decorator
@optimizer.resource_optimized()
def resource_intensive_function(x, y):
    return x ** y

# Get performance statistics
stats = optimizer.get_performance_stats()
print(f"Cache hit rate: {stats['cache']['hit_rate']:.2f}%")

# Optimize system
optimizer.optimize_system()
```

### Context Managers

The optimization system provides context managers for performance monitoring:

```python
from optimizer import performance_monitor, async_performance_monitor

# Monitor performance of synchronous operations
with performance_monitor(optimizer, "database_query"):
    # Database operations
    pass

# Monitor performance of asynchronous operations
async with async_performance_monitor(optimizer, "api_call"):
    # API operations
    pass
```

### Utility Functions

```python
from optimizer import compress_data, decompress_data, batch_process, parallel_map

# Compress data for storage/transmission
compressed = compress_data(large_data)
decompressed = decompress_data(compressed)

# Process items in batches
items = list(range(1000))
batches = batch_process(items, batch_size=100)

# Parallel processing
def process_item(item):
    return item * 2

results = parallel_map(process_item, items, max_workers=4)
```

## Monitoring System

### Features

The monitoring system provides:

- **Health Checks**: System, database, and network health monitoring
- **Metrics Collection**: Real-time system metrics collection
- **Alerting**: Configurable alerting with multiple levels
- **Live Monitoring**: Real-time monitoring with rich interface

### Components

#### Health Checkers

```python
from monitor import SystemHealthChecker, DatabaseHealthChecker, NetworkHealthChecker

# System health checker
system_checker = SystemHealthChecker()
result = await system_checker.check()
print(f"System status: {result.status}")

# Database health checker
db_checker = DatabaseHealthChecker("/path/to/database.db")
result = await db_checker.check()
print(f"Database status: {result.status}")

# Network health checker
network_checker = NetworkHealthChecker([
    "https://api.github.com",
    "https://httpbin.org/get"
])
result = await network_checker.check()
print(f"Network status: {result.status}")
```

#### MetricsCollector

```python
from monitor import MetricsCollector

collector = MetricsCollector()

# Start metrics collection
collector.start_collection()

# Add custom metric
collector.add_metric("custom_metric", 42.5, datetime.now())

# Get metrics
metrics = collector.get_metrics("cpu_usage", minutes=60)
latest = collector.get_latest_metric("memory_usage")
stats = collector.get_metric_stats("disk_usage", minutes=60)

# Stop collection
collector.stop_collection()
```

#### AlertManager

```python
from monitor import AlertManager, AlertLevel

manager = AlertManager()

# Register alert handler
def email_handler(alert):
    if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
        send_email(alert.message)

manager.register_handler(AlertLevel.ERROR, email_handler)

# Add alerts
manager.add_alert(
    AlertLevel.WARNING,
    "High memory usage detected",
    "system_monitor",
    {"memory_usage": 85.5}
)

# Get alerts
recent_alerts = manager.get_alerts(hours=24)
critical_alerts = manager.get_alerts(level=AlertLevel.CRITICAL)
```

#### SystemMonitor

```python
from monitor import SystemMonitor, create_monitor

# Create monitor
monitor = create_monitor("/path/to/repo")

# Start monitoring
monitor.start_monitoring()

# Run health checks
results = await monitor.run_health_checks()

# Get overall health
health = monitor.get_overall_health()
print(f"Overall health: {health}")

# Get status report
report = monitor.get_status_report()
print(f"Recent alerts: {report['recent_alerts']}")

# Display live status
monitor.display_status()

# Stop monitoring
monitor.stop_monitoring()
```

## Integration Guide

### Basic Integration

To integrate the guardrails and optimization systems into your application:

```python
from guardrails import GuardrailsManager
from optimizer import PerformanceOptimizer, CacheConfig, PerformanceConfig
from monitor import create_monitor

# Initialize systems
guardrails = GuardrailsManager()
cache_config = CacheConfig()
perf_config = PerformanceConfig()
optimizer = PerformanceOptimizer(cache_config, perf_config)
monitor = create_monitor("/path/to/repo")

# Use guardrails for input validation
is_valid, data = guardrails.validate_input(user_input)
if is_valid:
    # Process data
    pass

# Use optimization for expensive operations
@optimizer.cached_function()
def expensive_operation(data):
    # Expensive computation
    return result

# Use monitoring for health checks
health_status = await monitor.run_health_checks()
```

### Advanced Integration

For advanced integration with decorators and context managers:

```python
# Secure function with full protection
@secure_function(guardrails)
def process_user_data(user_data):
    # Validate input
    is_valid, validated_data = guardrails.validate_input(user_data)
    if not is_valid:
        raise ValueError("Invalid input")
    
    # Process with optimization
    @optimizer.cached_function()
    def process_data(data):
        return expensive_processing(data)
    
    result = process_data(validated_data)
    
    # Record security event
    guardrails.record_security_event(
        "data_processed",
        SecurityLevel.INFO,
        "User data processed successfully",
        user_id=user_data.get("user_id")
    )
    
    return result

# Resource management with monitoring
with resource_guard(guardrails, "database_operation"):
    with performance_monitor(optimizer, "query_execution"):
        # Database operations
        result = execute_query(query)
        
        # Monitor health
        health = await monitor.run_health_checks()
        if health['database'].status == HealthStatus.CRITICAL:
            raise Exception("Database health critical")
        
        return result
```

## Configuration

### Guardrails Configuration

```python
# Custom guardrails configuration
guardrails_config = {
    "rate_limiting": {
        "api_requests_per_hour": 1000,
        "auth_attempts_per_5min": 5
    },
    "security": {
        "enable_encryption": True,
        "encryption_key": "your-secret-key"
    },
    "validation": {
        "max_string_length": 1000,
        "enable_html_sanitization": True
    }
}

guardrails = GuardrailsManager(guardrails_config)
```

### Optimization Configuration

```python
# Custom optimization configuration
cache_config = CacheConfig(
    strategy=CacheStrategy.LRU,
    max_size=2000,
    ttl_seconds=7200,
    enable_compression=True,
    enable_persistence=True,
    cache_dir="/path/to/cache"
)

perf_config = PerformanceConfig(
    max_workers=8,
    max_processes=4,
    enable_async=True,
    enable_caching=True,
    memory_limit_mb=1024,
    cpu_limit_percent=90
)

optimizer = PerformanceOptimizer(cache_config, perf_config)
```

### Monitoring Configuration

```python
# Custom monitoring configuration
monitor_config = {
    "health_checks": {
        "system_check_interval": 60,
        "database_check_interval": 300,
        "network_check_interval": 120
    },
    "metrics": {
        "collection_interval": 30,
        "retention_days": 30
    },
    "alerts": {
        "enable_email_alerts": True,
        "enable_slack_alerts": False,
        "critical_threshold": 90
    }
}

monitor = SystemMonitor("/path/to/repo", monitor_config)
```

## Testing

### Running Tests

The project includes comprehensive test suites for all systems:

```bash
# Run all tests
pytest src/tests/

# Run specific test suites
pytest src/tests/test_guardrails.py
pytest src/tests/test_optimizer.py
pytest src/tests/test_monitor.py

# Run with coverage
pytest --cov=src src/tests/
```

### Test Structure

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Error Handling Tests**: Test error scenarios and recovery
- **Performance Tests**: Test performance under load

### Example Test

```python
import pytest
from guardrails import GuardrailsManager

class TestGuardrailsIntegration:
    def setup_method(self):
        self.manager = GuardrailsManager()
    
    def test_input_validation_workflow(self):
        # Test complete input validation workflow
        test_data = {"user_id": "123", "name": "John"}
        
        is_valid, validated_data = self.manager.validate_input(test_data)
        assert is_valid
        assert validated_data == test_data
    
    def test_rate_limiting_workflow(self):
        # Test rate limiting workflow
        for i in range(10):
            allowed = self.manager.check_rate_limit("test_user")
            if i < 5:  # First 5 should be allowed
                assert allowed
            else:  # Next 5 should be blocked
                assert not allowed
```

## Troubleshooting

### Common Issues

#### Guardrails Issues

**Issue**: Input validation failing unexpectedly
```python
# Check validation configuration
validator = InputValidator()
is_valid, result = validator.validate_string(input_data)
print(f"Validation result: {result}")
```

**Issue**: Rate limiting too restrictive
```python
# Adjust rate limiting configuration
limiter = RateLimiter(max_requests=100, window_seconds=60)
# Increase max_requests or decrease window_seconds
```

#### Optimization Issues

**Issue**: Cache not working as expected
```python
# Check cache configuration and statistics
stats = cache.get_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2f}%")
print(f"Cache size: {stats['size']}")
```

**Issue**: Memory usage too high
```python
# Check resource usage and optimize
memory_healthy, usage = resource_manager.check_memory_usage()
if not memory_healthy:
    resource_manager.optimize_memory()
```

#### Monitoring Issues

**Issue**: Health checks failing
```python
# Check individual health checkers
system_result = await system_checker.check()
print(f"System health: {system_result.status}")
print(f"System message: {system_result.message}")
```

**Issue**: Metrics not being collected
```python
# Check metrics collection status
collector.start_collection()
time.sleep(1)
latest_metric = collector.get_latest_metric("cpu_usage")
if latest_metric is None:
    print("Metrics collection not working")
```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug mode for specific components
guardrails.logger.setLevel(logging.DEBUG)
optimizer.logger.setLevel(logging.DEBUG)
monitor.logger.setLevel(logging.DEBUG)
```

### Performance Tuning

#### Cache Optimization

```python
# Optimize cache for your use case
cache_config = CacheConfig(
    strategy=CacheStrategy.LRU,  # Use LRU for frequently accessed data
    max_size=5000,  # Increase size for more data
    ttl_seconds=1800,  # Reduce TTL for fresher data
    enable_compression=True  # Enable for large data
)
```

#### Resource Management

```python
# Adjust resource limits based on your system
perf_config = PerformanceConfig(
    max_workers=8,  # Increase for more concurrent operations
    max_processes=4,  # Increase for CPU-intensive tasks
    memory_limit_mb=2048,  # Increase for memory-intensive operations
    cpu_limit_percent=90  # Adjust based on system capacity
)
```

#### Monitoring Configuration

```python
# Adjust monitoring intervals based on your needs
monitor_config = {
    "health_checks": {
        "system_check_interval": 30,  # More frequent checks
        "database_check_interval": 60,  # Less frequent for database
        "network_check_interval": 30   # More frequent for network
    },
    "metrics": {
        "collection_interval": 15,  # More frequent metrics
        "retention_days": 7  # Shorter retention for performance
    }
}
```

## Conclusion

The guardrails and optimization systems provide comprehensive security, performance, and monitoring capabilities for the SmartGitMergePT project. By following this documentation and using the provided examples, you can effectively integrate these systems into your applications to achieve:

- **Enhanced Security**: Input validation, rate limiting, and threat detection
- **Improved Performance**: Caching, async operations, and resource optimization
- **Better Monitoring**: Health checks, metrics collection, and alerting
- **Increased Reliability**: Error handling, recovery strategies, and system resilience

For additional support or questions, please refer to the test suites and examples provided in the codebase. 