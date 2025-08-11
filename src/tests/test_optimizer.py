"""
Tests for the Optimization System
"""

import pytest
import asyncio
import time
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from optimizer import (
    PerformanceOptimizer, CacheConfig, PerformanceConfig,
    SmartCache, AsyncTaskManager, ResourceManager,
    CacheStrategy, OptimizationLevel,
    performance_monitor, async_performance_monitor,
    compress_data, decompress_data, batch_process, parallel_map
)

class TestCacheConfig:
    """Test cache configuration"""
    
    def test_default_config(self):
        """Test default cache configuration"""
        config = CacheConfig()
        assert config.strategy == CacheStrategy.TTL
        assert config.max_size == 1000
        assert config.ttl_seconds == 3600
        assert config.enable_compression == True
        assert config.enable_persistence == False
        assert config.cache_dir == ".cache"

class TestPerformanceConfig:
    """Test performance configuration"""
    
    def test_default_config(self):
        """Test default performance configuration"""
        config = PerformanceConfig()
        assert config.max_workers == 4
        assert config.max_processes == 2
        assert config.enable_async == True
        assert config.enable_caching == True
        assert config.enable_compression == True
        assert config.memory_limit_mb == 512
        assert config.cpu_limit_percent == 80

class TestSmartCache:
    """Test smart caching functionality"""
    
    def setup_method(self):
        self.cache_config = CacheConfig()
        self.cache = SmartCache(self.cache_config)
    
    def test_cache_set_get(self):
        """Test basic cache set and get operations"""
        key = "test_key"
        value = "test_value"
        
        # Set value
        success = self.cache.set(key, value)
        assert success
        
        # Get value
        retrieved = self.cache.get(key)
        assert retrieved == value
    
    def test_cache_miss(self):
        """Test cache miss behavior"""
        retrieved = self.cache.get("nonexistent_key")
        assert retrieved is None
    
    def test_cache_stats(self):
        """Test cache statistics"""
        # Add some data
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.get("key1")  # Hit
        self.cache.get("nonexistent")  # Miss
        
        stats = self.cache.get_stats()
        assert stats['hits'] > 0
        assert stats['misses'] > 0
        assert stats['size'] > 0
        assert 'hit_rate' in stats
    
    def test_cache_clear(self):
        """Test cache clearing"""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        assert self.cache.get("key1") == "value1"
        
        self.cache.clear()
        assert self.cache.get("key1") is None
    
    def test_cache_delete(self):
        """Test cache deletion"""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        # Delete one key
        success = self.cache.delete("key1")
        assert success
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") == "value2"
    
    def test_cache_compression(self):
        """Test cache compression"""
        # Create a large string
        large_string = "x" * 2000
        self.cache.set("large_key", large_string)
        
        # Should be compressed
        retrieved = self.cache.get("large_key")
        assert retrieved == large_string
    
    def test_persistent_cache(self):
        """Test persistent cache functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(
                enable_persistence=True,
                cache_dir=temp_dir
            )
            cache = SmartCache(config)
            
            # Set value
            cache.set("persistent_key", "persistent_value")
            
            # Create new cache instance (simulating restart)
            new_cache = SmartCache(config)
            
            # Should be able to retrieve the value
            retrieved = new_cache.get("persistent_key")
            assert retrieved == "persistent_value"

class TestAsyncTaskManager:
    """Test async task manager"""
    
    def setup_method(self):
        self.manager = AsyncTaskManager(max_workers=2, max_processes=1)
    
    @pytest.mark.asyncio
    async def test_run_in_thread(self):
        """Test running function in thread pool"""
        def test_function(x, y):
            return x + y
        
        result = await self.manager.run_in_thread(test_function, 5, 3)
        assert result == 8
    
    @pytest.mark.asyncio
    async def test_run_in_process(self):
        """Test running function in process pool"""
        def test_function(x, y):
            return x * y
        
        result = await self.manager.run_in_process(test_function, 4, 6)
        assert result == 24
    
    @pytest.mark.asyncio
    async def test_run_concurrent(self):
        """Test running multiple tasks concurrently"""
        def task1():
            time.sleep(0.1)
            return "result1"
        
        def task2():
            time.sleep(0.1)
            return "result2"
        
        tasks = [task1, task2]
        results = await self.manager.run_concurrent(tasks)
        
        assert len(results) == 2
        assert "result1" in results
        assert "result2" in results
    
    def test_cleanup(self):
        """Test cleanup functionality"""
        # Should not raise any exceptions
        self.manager.cleanup()

class TestResourceManager:
    """Test resource manager"""
    
    def setup_method(self):
        self.manager = ResourceManager(memory_limit_mb=1024, cpu_limit_percent=50)
    
    def test_check_memory_usage(self):
        """Test memory usage checking"""
        is_healthy, usage_mb = self.manager.check_memory_usage()
        
        assert isinstance(is_healthy, bool)
        assert isinstance(usage_mb, float)
        assert usage_mb > 0
    
    def test_check_cpu_usage(self):
        """Test CPU usage checking"""
        is_healthy, usage_percent = self.manager.check_cpu_usage()
        
        assert isinstance(is_healthy, bool)
        assert isinstance(usage_percent, float)
        assert 0 <= usage_percent <= 100
    
    def test_optimize_memory(self):
        """Test memory optimization"""
        # Should not raise any exceptions
        self.manager.optimize_memory()
    
    def test_get_resource_stats(self):
        """Test resource statistics"""
        stats = self.manager.get_resource_stats()
        
        assert 'memory' in stats
        assert 'cpu' in stats
        assert 'gc' in stats
        
        assert 'healthy' in stats['memory']
        assert 'usage_mb' in stats['memory']
        assert 'limit_mb' in stats['memory']
        
        assert 'healthy' in stats['cpu']
        assert 'usage_percent' in stats['cpu']
        assert 'limit_percent' in stats['cpu']

class TestPerformanceOptimizer:
    """Test performance optimizer"""
    
    def setup_method(self):
        cache_config = CacheConfig()
        perf_config = PerformanceConfig()
        self.optimizer = PerformanceOptimizer(cache_config, perf_config)
    
    def test_cached_function_decorator(self):
        """Test cached function decorator"""
        @self.optimizer.cached_function(ttl_seconds=3600)
        def test_function(x, y):
            return x + y
        
        # First call should cache
        result1 = test_function(5, 3)
        assert result1 == 8
        
        # Second call should use cache
        result2 = test_function(5, 3)
        assert result2 == 8
    
    def test_async_function_decorator(self):
        """Test async function decorator"""
        @self.optimizer.async_function()
        async def test_async_function(x, y):
            await asyncio.sleep(0.01)  # Simulate async work
            return x * y
        
        # Test async function
        result = asyncio.run(test_async_function(4, 6))
        assert result == 24
    
    def test_resource_optimized_decorator(self):
        """Test resource optimized decorator"""
        @self.optimizer.resource_optimized()
        def test_resource_function(x, y):
            return x ** y
        
        result = test_resource_function(2, 8)
        assert result == 256
    
    def test_create_cache_key(self):
        """Test cache key creation"""
        args = (1, 2, 3)
        kwargs = {'a': 1, 'b': 2}
        
        key = self.optimizer._create_cache_key("test_function", args, kwargs)
        assert isinstance(key, str)
        assert len(key) > 0
    
    def test_get_performance_stats(self):
        """Test performance statistics"""
        stats = self.optimizer.get_performance_stats()
        
        assert 'cache' in stats
        assert 'resources' in stats
        assert 'functions' in stats
        assert 'optimizations' in stats
    
    def test_optimize_system(self):
        """Test system optimization"""
        # Should not raise any exceptions
        self.optimizer.optimize_system()
    
    def test_cleanup(self):
        """Test cleanup functionality"""
        # Should not raise any exceptions
        self.optimizer.cleanup()

class TestContextManagers:
    """Test context managers"""
    
    def setup_method(self):
        cache_config = CacheConfig()
        perf_config = PerformanceConfig()
        self.optimizer = PerformanceOptimizer(cache_config, perf_config)
    
    def test_performance_monitor(self):
        """Test performance monitor context manager"""
        with performance_monitor(self.optimizer, "test_operation"):
            # Simulate some work
            time.sleep(0.01)
            pass
    
    @pytest.mark.asyncio
    async def test_async_performance_monitor(self):
        """Test async performance monitor context manager"""
        async with async_performance_monitor(self.optimizer, "test_async_operation"):
            # Simulate some async work
            await asyncio.sleep(0.01)
            pass

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_compress_decompress_data(self):
        """Test data compression and decompression"""
        original_data = {"key": "value", "number": 123, "list": [1, 2, 3]}
        
        compressed = compress_data(original_data)
        decompressed = decompress_data(compressed)
        
        assert decompressed == original_data
        assert compressed != original_data
    
    def test_batch_process(self):
        """Test batch processing"""
        items = list(range(25))
        batches = batch_process(items, batch_size=10)
        
        assert len(batches) == 3
        assert len(batches[0]) == 10
        assert len(batches[1]) == 10
        assert len(batches[2]) == 5
    
    def test_parallel_map(self):
        """Test parallel mapping"""
        def square(x):
            return x ** 2
        
        items = [1, 2, 3, 4, 5]
        results = parallel_map(square, items, max_workers=2)
        
        assert results == [1, 4, 9, 16, 25]

class TestIntegration:
    """Integration tests for optimization system"""
    
    def setup_method(self):
        cache_config = CacheConfig()
        perf_config = PerformanceConfig()
        self.optimizer = PerformanceOptimizer(cache_config, perf_config)
    
    def test_full_optimization_workflow(self):
        """Test complete optimization workflow"""
        # 1. Use cached function
        @self.optimizer.cached_function()
        def expensive_function(x):
            time.sleep(0.01)  # Simulate expensive operation
            return x * 2
        
        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        
        # Second call (should use cache)
        result2 = expensive_function(5)
        assert result2 == 10
        
        # 2. Use resource optimized function
        @self.optimizer.resource_optimized()
        def resource_intensive_function(x, y):
            return x ** y
        
        result3 = resource_intensive_function(2, 10)
        assert result3 == 1024
        
        # 3. Get performance stats
        stats = self.optimizer.get_performance_stats()
        assert 'cache' in stats
        assert 'functions' in stats
        
        # 4. Optimize system
        self.optimizer.optimize_system()
        
        # 5. Cleanup
        self.optimizer.cleanup()
    
    @pytest.mark.asyncio
    async def test_async_workflow(self):
        """Test async optimization workflow"""
        # 1. Use async function decorator
        @self.optimizer.async_function()
        async def async_expensive_function(x):
            await asyncio.sleep(0.01)  # Simulate async work
            return x * 3
        
        # Call async function
        result = await async_expensive_function(7)
        assert result == 21
        
        # 2. Use async task manager
        def sync_function(x):
            return x * 4
        
        result2 = await self.optimizer.async_manager.run_in_thread(sync_function, 6)
        assert result2 == 24
        
        # 3. Use concurrent execution
        def task1():
            return "task1_result"
        
        def task2():
            return "task2_result"
        
        results = await self.optimizer.async_manager.run_concurrent([task1, task2])
        assert len(results) == 2
        assert "task1_result" in results
        assert "task2_result" in results

class TestErrorHandling:
    """Test error handling in optimization system"""
    
    def setup_method(self):
        cache_config = CacheConfig()
        perf_config = PerformanceConfig()
        self.optimizer = PerformanceOptimizer(cache_config, perf_config)
    
    def test_cache_error_handling(self):
        """Test cache error handling"""
        # Test with invalid cache key
        result = self.optimizer.cache.get(None)
        assert result is None
    
    def test_async_error_handling(self):
        """Test async error handling"""
        @self.optimizer.async_function()
        async def error_function():
            raise ValueError("Test error")
        
        # Should handle the error gracefully
        with pytest.raises(ValueError):
            asyncio.run(error_function())
    
    def test_resource_error_handling(self):
        """Test resource error handling"""
        @self.optimizer.resource_optimized()
        def error_function():
            raise RuntimeError("Resource error")
        
        # Should handle the error gracefully
        with pytest.raises(RuntimeError):
            error_function()

if __name__ == "__main__":
    pytest.main([__file__]) 