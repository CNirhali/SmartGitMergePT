"""
Optimization System for SmartGitMergePT
Provides caching, async operations, resource management, and performance improvements
"""

import asyncio
import functools
import gc
import hashlib
import json
import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import weakref

import psutil
from cachetools import TTLCache, LRUCache
import aiofiles
import aiohttp

# Configure logging
logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    LRU = "lru"
    TTL = "ttl"
    NONE = "none"

class OptimizationLevel(Enum):
    MINIMAL = "minimal"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

@dataclass
class CacheConfig:
    """Configuration for caching strategies"""
    strategy: CacheStrategy = CacheStrategy.TTL
    max_size: int = 1000
    ttl_seconds: int = 3600  # 1 hour
    enable_compression: bool = True
    enable_persistence: bool = False
    cache_dir: str = ".cache"

@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations"""
    max_workers: int = 4
    max_processes: int = 2
    enable_async: bool = True
    enable_caching: bool = True
    enable_compression: bool = True
    memory_limit_mb: int = 512
    cpu_limit_percent: int = 80

class SmartCache:
    """Intelligent caching system with multiple strategies"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize cache based on strategy
        if config.strategy == CacheStrategy.LRU:
            self.cache = LRUCache(maxsize=config.max_size)
        elif config.strategy == CacheStrategy.TTL:
            self.cache = TTLCache(maxsize=config.max_size, ttl=config.ttl_seconds)
        else:
            self.cache = {}
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Load persistent cache if enabled
        if config.enable_persistence:
            self._load_persistent_cache()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            # Try memory cache first
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
            
            # Try persistent cache if enabled
            if self.config.enable_persistence:
                value = self._load_from_disk(key)
                if value is not None:
                    self.hits += 1
                    self.cache[key] = value
                    return value
            
            self.misses += 1
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any) -> bool:
        """Set value in cache"""
        try:
            # Compress if enabled
            if self.config.enable_compression:
                value = self._compress_value(value)
            
            # Store in memory cache
            self.cache[key] = value
            
            # Store in persistent cache if enabled
            if self.config.enable_persistence:
                self._save_to_disk(key, value)
            
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            if key in self.cache:
                del self.cache[key]
            
            if self.config.enable_persistence:
                self._delete_from_disk(key)
            
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        if self.config.enable_persistence:
            self._clear_disk_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "evictions": self.evictions
        }
    
    def _compress_value(self, value: Any) -> Any:
        """Compress cache value"""
        if isinstance(value, str) and len(value) > 1000:
            # Simple compression for large strings
            return f"COMPRESSED:{hashlib.md5(value.encode()).hexdigest()}"
        return value
    
    def _load_from_disk(self, key: str) -> Optional[Any]:
        """Load value from disk cache"""
        try:
            cache_file = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.cache"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Load from disk error: {e}")
        return None
    
    def _save_to_disk(self, key: str, value: Any):
        """Save value to disk cache"""
        try:
            cache_file = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.cache"
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.error(f"Save to disk error: {e}")
    
    def _delete_from_disk(self, key: str):
        """Delete value from disk cache"""
        try:
            cache_file = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.cache"
            if cache_file.exists():
                cache_file.unlink()
        except Exception as e:
            logger.error(f"Delete from disk error: {e}")
    
    def _clear_disk_cache(self):
        """Clear disk cache"""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
        except Exception as e:
            logger.error(f"Clear disk cache error: {e}")
    
    def _load_persistent_cache(self):
        """Load persistent cache on startup"""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                with open(cache_file, 'rb') as f:
                    # We can't reconstruct the key, so we'll use filename as key
                    key = cache_file.stem
                    self.cache[key] = pickle.load(f)
        except Exception as e:
            logger.error(f"Load persistent cache error: {e}")

class AsyncTaskManager:
    """Manages async operations and concurrency"""
    
    def __init__(self, max_workers: int = 4, max_processes: int = 2):
        self.max_workers = max_workers
        self.max_processes = max_processes
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_processes)
        self.active_tasks: Set[asyncio.Task] = set()
    
    async def run_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """Run function in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, func, *args, **kwargs)
    
    async def run_in_process(self, func: Callable, *args, **kwargs) -> Any:
        """Run function in process pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.process_pool, func, *args, **kwargs)
    
    async def run_concurrent(self, tasks: List[Callable], *args, **kwargs) -> List[Any]:
        """Run multiple tasks concurrently"""
        async def run_task(task):
            if asyncio.iscoroutinefunction(task):
                return await task(*args, **kwargs)
            else:
                return await self.run_in_thread(task, *args, **kwargs)
        
        results = await asyncio.gather(*[run_task(task) for task in tasks])
        return results
    
    def cleanup(self):
        """Cleanup resources"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

class ResourceManager:
    """Manages system resources and memory"""
    
    def __init__(self, memory_limit_mb: int = 512, cpu_limit_percent: int = 80):
        self.memory_limit_mb = memory_limit_mb
        self.cpu_limit_percent = cpu_limit_percent
        self.memory_usage = []
        self.cpu_usage = []
        self.gc_stats = defaultdict(int)
    
    def check_memory_usage(self) -> Tuple[bool, float]:
        """Check current memory usage"""
        memory = psutil.virtual_memory()
        usage_mb = memory.used / (1024 * 1024)
        is_healthy = usage_mb < self.memory_limit_mb
        
        self.memory_usage.append({
            'timestamp': datetime.now(),
            'usage_mb': usage_mb,
            'percent': memory.percent
        })
        
        # Keep only last 100 entries
        if len(self.memory_usage) > 100:
            self.memory_usage.pop(0)
        
        return is_healthy, usage_mb
    
    def check_cpu_usage(self) -> Tuple[bool, float]:
        """Check current CPU usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        is_healthy = cpu_percent < self.cpu_limit_percent
        
        self.cpu_usage.append({
            'timestamp': datetime.now(),
            'usage_percent': cpu_percent
        })
        
        # Keep only last 100 entries
        if len(self.cpu_usage) > 100:
            self.cpu_usage.pop(0)
        
        return is_healthy, cpu_percent
    
    def optimize_memory(self):
        """Perform memory optimization"""
        # Force garbage collection
        collected = gc.collect()
        self.gc_stats['collections'] += 1
        self.gc_stats['objects_collected'] += collected
        
        # Clear memory usage history if too large
        if len(self.memory_usage) > 50:
            self.memory_usage = self.memory_usage[-25:]
        
        if len(self.cpu_usage) > 50:
            self.cpu_usage = self.cpu_usage[-25:]
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics"""
        memory_healthy, memory_usage = self.check_memory_usage()
        cpu_healthy, cpu_usage = self.check_cpu_usage()
        
        return {
            'memory': {
                'healthy': memory_healthy,
                'usage_mb': memory_usage,
                'limit_mb': self.memory_limit_mb,
                'history': self.memory_usage[-10:] if self.memory_usage else []
            },
            'cpu': {
                'healthy': cpu_healthy,
                'usage_percent': cpu_usage,
                'limit_percent': self.cpu_limit_percent,
                'history': self.cpu_usage[-10:] if self.cpu_usage else []
            },
            'gc': dict(self.gc_stats)
        }

class PerformanceOptimizer:
    """Main performance optimization system"""
    
    def __init__(self, cache_config: CacheConfig, perf_config: PerformanceConfig):
        self.cache_config = cache_config
        self.perf_config = perf_config
        
        # Initialize components
        self.cache = SmartCache(cache_config)
        self.async_manager = AsyncTaskManager(
            max_workers=perf_config.max_workers,
            max_processes=perf_config.max_processes
        )
        self.resource_manager = ResourceManager(
            memory_limit_mb=perf_config.memory_limit_mb,
            cpu_limit_percent=perf_config.cpu_limit_percent
        )
        
        # Performance tracking
        self.function_timings = defaultdict(list)
        self.optimization_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'async_operations': 0,
            'memory_optimizations': 0
        }
    
    def cached_function(self, ttl_seconds: int = 3600):
        """Decorator for caching function results"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key
                cache_key = self._create_cache_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    self.optimization_stats['cache_hits'] += 1
                    return cached_result
                
                self.optimization_stats['cache_misses'] += 1
                
                # Execute function and cache result
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record timing
                self.function_timings[func.__name__].append(duration)
                
                # Cache result
                self.cache.set(cache_key, result)
                
                return result
            
            return wrapper
        return decorator
    
    def async_function(self):
        """Decorator for async function optimization"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                self.optimization_stats['async_operations'] += 1
                
                start_time = time.time()
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record timing
                self.function_timings[func.__name__].append(duration)
                
                return result
            
            return wrapper
        return decorator
    
    def resource_optimized(self):
        """Decorator for resource-optimized functions"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Check resource usage before execution
                memory_healthy, _ = self.resource_manager.check_memory_usage()
                cpu_healthy, _ = self.resource_manager.check_cpu_usage()
                
                # Optimize if needed
                if not memory_healthy or not cpu_healthy:
                    self.resource_manager.optimize_memory()
                    self.optimization_stats['memory_optimizations'] += 1
                
                # Execute function
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record timing
                self.function_timings[func.__name__].append(duration)
                
                return result
            
            return wrapper
        return decorator
    
    def _create_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Create cache key from function name and arguments"""
        # Create a hash of the function call
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        cache_stats = self.cache.get_stats()
        resource_stats = self.resource_manager.get_resource_stats()
        
        # Calculate function performance
        function_stats = {}
        for func_name, timings in self.function_timings.items():
            if timings:
                function_stats[func_name] = {
                    'avg_time': sum(timings) / len(timings),
                    'min_time': min(timings),
                    'max_time': max(timings),
                    'call_count': len(timings)
                }
        
        return {
            'cache': cache_stats,
            'resources': resource_stats,
            'functions': function_stats,
            'optimizations': dict(self.optimization_stats)
        }
    
    def optimize_system(self):
        """Perform system-wide optimization"""
        # Memory optimization
        self.resource_manager.optimize_memory()
        
        # Cache cleanup
        if self.cache_config.strategy == CacheStrategy.TTL:
            # TTL cache auto-cleans, but we can force cleanup
            pass
        
        # Log optimization results
        stats = self.get_performance_stats()
        logger.info(f"System optimization completed. Cache hit rate: {stats['cache']['hit_rate']:.2f}%")
    
    def cleanup(self):
        """Cleanup all resources"""
        self.async_manager.cleanup()
        self.cache.clear()

# Context managers for optimization
@contextmanager
def performance_monitor(optimizer: PerformanceOptimizer, operation_name: str):
    """Context manager for performance monitoring"""
    start_time = time.time()
    start_memory = psutil.virtual_memory().used
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        end_memory = psutil.virtual_memory().used
        memory_delta = end_memory - start_memory
        
        logger.info(f"Operation '{operation_name}' completed in {duration:.3f}s, "
                   f"memory delta: {memory_delta / (1024*1024):.2f}MB")

@asynccontextmanager
async def async_performance_monitor(optimizer: PerformanceOptimizer, operation_name: str):
    """Async context manager for performance monitoring"""
    start_time = time.time()
    start_memory = psutil.virtual_memory().used
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        end_memory = psutil.virtual_memory().used
        memory_delta = end_memory - start_memory
        
        logger.info(f"Async operation '{operation_name}' completed in {duration:.3f}s, "
                   f"memory delta: {memory_delta / (1024*1024):.2f}MB")

# Utility functions for optimization
def optimize_imports():
    """Optimize Python imports for better startup time"""
    import importlib
    import sys
    
    # Lazy import common modules
    def lazy_import(module_name: str):
        def import_module():
            return importlib.import_module(module_name)
        return import_module
    
    # Replace heavy imports with lazy versions
    sys.modules['lazy_cv'] = lazy_import('cv2')
    sys.modules['lazy_np'] = lazy_import('numpy')
    sys.modules['lazy_pd'] = lazy_import('pandas')

def compress_data(data: Any) -> bytes:
    """Compress data for storage/transmission"""
    import gzip
    import pickle
    
    serialized = pickle.dumps(data)
    compressed = gzip.compress(serialized)
    return compressed

def decompress_data(compressed_data: bytes) -> Any:
    """Decompress data"""
    import gzip
    import pickle
    
    decompressed = gzip.decompress(compressed_data)
    return pickle.loads(decompressed)

def batch_process(items: List[Any], batch_size: int = 100) -> List[List[Any]]:
    """Split items into batches for processing"""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

def parallel_map(func: Callable, items: List[Any], max_workers: int = 4) -> List[Any]:
    """Parallel map function with ThreadPoolExecutor"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, items))

# Performance profiling decorator
def profile_function(func: Callable) -> Callable:
    """Decorator to profile function performance"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import cProfile
        import pstats
        import io
        
        # Create profiler
        pr = cProfile.Profile()
        pr.enable()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Stop profiling
        pr.disable()
        
        # Get stats
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions
        
        # Log results
        logger.debug(f"Profile for {func.__name__}:\n{s.getvalue()}")
        
        return result
    
    return wrapper 