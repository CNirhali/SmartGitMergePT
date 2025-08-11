"""
Tests for the Monitoring System
"""

import pytest
import asyncio
import time
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from monitor import (
    SystemMonitor, SystemHealthChecker, DatabaseHealthChecker, NetworkHealthChecker,
    MetricsCollector, AlertManager, HealthStatus, AlertLevel,
    create_monitor, run_health_checks, get_system_status
)

class TestHealthStatus:
    """Test health status enum"""
    
    def test_health_status_values(self):
        """Test health status enum values"""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.WARNING.value == "warning"
        assert HealthStatus.CRITICAL.value == "critical"
        assert HealthStatus.UNKNOWN.value == "unknown"

class TestAlertLevel:
    """Test alert level enum"""
    
    def test_alert_level_values(self):
        """Test alert level enum values"""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"

class TestSystemHealthChecker:
    """Test system health checker"""
    
    def setup_method(self):
        self.checker = SystemHealthChecker()
    
    @pytest.mark.asyncio
    async def test_check_system_health(self):
        """Test system health check"""
        result = await self.checker.check()
        
        assert result.name == "system"
        assert isinstance(result.status, HealthStatus)
        assert isinstance(result.message, str)
        assert isinstance(result.timestamp, datetime)
        assert isinstance(result.duration_ms, float)
        assert result.duration_ms > 0
        
        # Check details structure
        assert 'cpu_percent' in result.details
        assert 'memory_percent' in result.details
        assert 'disk_percent' in result.details
        assert 'memory_available_mb' in result.details
        assert 'disk_free_gb' in result.details
    
    @pytest.mark.asyncio
    async def test_check_system_health_error_handling(self):
        """Test system health check error handling"""
        with patch('psutil.cpu_percent', side_effect=Exception("CPU error")):
            result = await self.checker.check()
            
            assert result.status == HealthStatus.CRITICAL
            assert "failed" in result.message.lower()
            assert 'error' in result.details

class TestDatabaseHealthChecker:
    """Test database health checker"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.checker = DatabaseHealthChecker(self.db_path)
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_check_database_not_exists(self):
        """Test database health check when database doesn't exist"""
        result = await self.checker.check()
        
        assert result.name == "database"
        assert result.status == HealthStatus.WARNING
        assert "not found" in result.message.lower()
        assert self.db_path in result.details['db_path']
    
    @pytest.mark.asyncio
    async def test_check_database_exists(self):
        """Test database health check when database exists"""
        # Create a test database
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)")
        cursor.execute("INSERT INTO test_table (name) VALUES ('test')")
        conn.commit()
        conn.close()
        
        result = await self.checker.check()
        
        assert result.name == "database"
        assert result.status == HealthStatus.HEALTHY
        assert "healthy" in result.message.lower()
        assert result.details['tables_count'] > 0
        assert result.details['db_size_mb'] > 0
    
    @pytest.mark.asyncio
    async def test_check_database_large_size(self):
        """Test database health check with large database"""
        # Create a large database
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE large_table (id INTEGER PRIMARY KEY, data TEXT)")
        
        # Insert large amount of data
        large_data = "x" * 1000
        for i in range(1000):
            cursor.execute("INSERT INTO large_table (data) VALUES (?)", (large_data,))
        
        conn.commit()
        conn.close()
        
        result = await self.checker.check()
        
        # Should be warning due to large size
        assert result.status == HealthStatus.WARNING
        assert "large" in result.message.lower()

class TestNetworkHealthChecker:
    """Test network health checker"""
    
    def setup_method(self):
        self.checker = NetworkHealthChecker()
    
    @pytest.mark.asyncio
    async def test_check_network_health(self):
        """Test network health check"""
        result = await self.checker.check()
        
        assert result.name == "network"
        assert isinstance(result.status, HealthStatus)
        assert isinstance(result.message, str)
        assert isinstance(result.timestamp, datetime)
        assert isinstance(result.duration_ms, float)
        assert result.duration_ms > 0
        
        # Check details structure
        assert 'success_rate' in result.details
        assert 'endpoints' in result.details
        assert isinstance(result.details['endpoints'], list)
    
    @pytest.mark.asyncio
    async def test_check_network_health_with_custom_endpoints(self):
        """Test network health check with custom endpoints"""
        custom_endpoints = ["https://httpbin.org/status/200"]
        checker = NetworkHealthChecker(custom_endpoints)
        
        result = await checker.check()
        
        assert result.name == "network"
        assert len(result.details['endpoints']) == 1
        assert result.details['endpoints'][0]['endpoint'] == custom_endpoints[0]

class TestMetricsCollector:
    """Test metrics collector"""
    
    def setup_method(self):
        self.collector = MetricsCollector()
    
    def test_add_metric(self):
        """Test adding metrics"""
        timestamp = datetime.now()
        self.collector.add_metric("test_metric", 42.5, timestamp, {"tag": "value"})
        
        metric = self.collector.get_latest_metric("test_metric")
        assert metric is not None
        assert metric.name == "test_metric"
        assert metric.value == 42.5
        assert metric.timestamp == timestamp
        assert metric.tags == {"tag": "value"}
    
    def test_get_metrics_time_range(self):
        """Test getting metrics within time range"""
        now = datetime.now()
        
        # Add metrics with different timestamps
        self.collector.add_metric("test_metric", 1, now - timedelta(minutes=30))
        self.collector.add_metric("test_metric", 2, now - timedelta(minutes=15))
        self.collector.add_metric("test_metric", 3, now)
        
        # Get metrics from last 20 minutes
        metrics = self.collector.get_metrics("test_metric", minutes=20)
        assert len(metrics) == 2  # Should get 2 metrics within 20 minutes
    
    def test_get_metric_stats(self):
        """Test getting metric statistics"""
        now = datetime.now()
        
        # Add multiple metrics
        for i in range(5):
            self.collector.add_metric("test_metric", i, now)
        
        stats = self.collector.get_metric_stats("test_metric", minutes=60)
        
        assert stats['min'] == 0
        assert stats['max'] == 4
        assert stats['avg'] == 2.0
        assert stats['count'] == 5
    
    def test_get_latest_metric_nonexistent(self):
        """Test getting latest metric for nonexistent metric"""
        metric = self.collector.get_latest_metric("nonexistent")
        assert metric is None
    
    def test_collect_system_metrics(self):
        """Test system metrics collection"""
        # This should not raise any exceptions
        self.collector._collect_system_metrics()
        
        # Should have collected some metrics
        cpu_metric = self.collector.get_latest_metric("cpu_usage")
        memory_metric = self.collector.get_latest_metric("memory_usage")
        
        # At least one should exist
        assert cpu_metric is not None or memory_metric is not None

class TestAlertManager:
    """Test alert manager"""
    
    def setup_method(self):
        self.manager = AlertManager()
    
    def test_add_alert(self):
        """Test adding alerts"""
        self.manager.add_alert(
            AlertLevel.WARNING,
            "Test warning message",
            "test_source",
            {"detail": "test_detail"}
        )
        
        alerts = self.manager.get_alerts()
        assert len(alerts) == 1
        
        alert = alerts[0]
        assert alert.level == AlertLevel.WARNING
        assert alert.message == "Test warning message"
        assert alert.source == "test_source"
        assert alert.details == {"detail": "test_detail"}
    
    def test_get_alerts_with_level_filter(self):
        """Test getting alerts with level filter"""
        # Add different level alerts
        self.manager.add_alert(AlertLevel.INFO, "Info message", "source1")
        self.manager.add_alert(AlertLevel.WARNING, "Warning message", "source2")
        self.manager.add_alert(AlertLevel.ERROR, "Error message", "source3")
        
        # Get only warning alerts
        warning_alerts = self.manager.get_alerts(level=AlertLevel.WARNING)
        assert len(warning_alerts) == 1
        assert warning_alerts[0].level == AlertLevel.WARNING
    
    def test_get_alerts_with_time_filter(self):
        """Test getting alerts with time filter"""
        # Add an alert
        self.manager.add_alert(AlertLevel.INFO, "Test message", "source")
        
        # Get alerts from last hour
        alerts = self.manager.get_alerts(hours=1)
        assert len(alerts) == 1
        
        # Get alerts from last minute (should be empty)
        alerts = self.manager.get_alerts(hours=1/60)
        assert len(alerts) == 0
    
    def test_clear_old_alerts(self):
        """Test clearing old alerts"""
        # Add an alert
        self.manager.add_alert(AlertLevel.INFO, "Test message", "source")
        
        # Clear alerts older than 1 day
        self.manager.clear_old_alerts(days=1)
        
        # Should still have the alert (it's recent)
        alerts = self.manager.get_alerts()
        assert len(alerts) == 1
    
    def test_register_handler(self):
        """Test registering alert handlers"""
        handler_called = False
        handler_message = None
        
        def test_handler(alert):
            nonlocal handler_called, handler_message
            handler_called = True
            handler_message = alert.message
        
        self.manager.register_handler(AlertLevel.INFO, test_handler)
        
        # Add an alert
        self.manager.add_alert(AlertLevel.INFO, "Test message", "source")
        
        # Handler should have been called
        assert handler_called
        assert handler_message == "Test message"

class TestSystemMonitor:
    """Test system monitor"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = SystemMonitor(self.temp_dir)
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test monitor initialization"""
        assert self.monitor.repo_path == self.temp_dir
        assert len(self.monitor.health_checkers) > 0
        assert self.monitor.metrics_collector is not None
        assert self.monitor.alert_manager is not None
    
    @pytest.mark.asyncio
    async def test_run_health_checks(self):
        """Test running health checks"""
        results = await self.monitor.run_health_checks()
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check that all health checkers returned results
        for checker in self.monitor.health_checkers:
            assert checker.name in results
    
    def test_get_overall_health(self):
        """Test getting overall health status"""
        health = self.monitor.get_overall_health()
        assert isinstance(health, HealthStatus)
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring"""
        self.monitor.start_monitoring()
        # Should not raise any exceptions
        
        self.monitor.stop_monitoring()
        # Should not raise any exceptions
    
    def test_get_status_report(self):
        """Test getting status report"""
        report = self.monitor.get_status_report()
        
        assert 'overall_health' in report
        assert 'last_check_time' in report
        assert 'health_checks' in report
        assert 'latest_metrics' in report
        assert 'recent_alerts' in report
        assert 'alert_summary' in report
    
    def test_display_status(self):
        """Test displaying status"""
        # Should not raise any exceptions
        self.monitor.display_status()

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_create_monitor(self):
        """Test creating monitor"""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = create_monitor(temp_dir)
            assert isinstance(monitor, SystemMonitor)
            assert monitor.repo_path == temp_dir
    
    @pytest.mark.asyncio
    async def test_run_health_checks(self):
        """Test running health checks utility function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = await run_health_checks(temp_dir)
            assert isinstance(results, dict)
            assert len(results) > 0
    
    def test_get_system_status(self):
        """Test getting system status utility function"""
        with tempfile.TemporaryDirectory() as temp_dir:
            status = get_system_status(temp_dir)
            assert isinstance(status, dict)
            assert 'overall_health' in status

class TestIntegration:
    """Integration tests for monitoring system"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = SystemMonitor(self.temp_dir)
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_full_monitoring_workflow(self):
        """Test complete monitoring workflow"""
        # 1. Start monitoring
        self.monitor.start_monitoring()
        
        # 2. Run health checks
        results = await self.monitor.run_health_checks()
        assert len(results) > 0
        
        # 3. Get overall health
        health = self.monitor.get_overall_health()
        assert isinstance(health, HealthStatus)
        
        # 4. Get status report
        report = self.monitor.get_status_report()
        assert isinstance(report, dict)
        
        # 5. Add some alerts
        self.monitor.alert_manager.add_alert(
            AlertLevel.INFO,
            "Test alert",
            "test_source"
        )
        
        # 6. Stop monitoring
        self.monitor.stop_monitoring()
    
    def test_metrics_collection_workflow(self):
        """Test metrics collection workflow"""
        # 1. Start metrics collection
        self.monitor.metrics_collector.start_collection()
        
        # 2. Wait a bit for collection
        time.sleep(0.1)
        
        # 3. Check that metrics were collected
        cpu_metric = self.monitor.metrics_collector.get_latest_metric("cpu_usage")
        memory_metric = self.monitor.metrics_collector.get_latest_metric("memory_usage")
        
        # At least one should exist
        assert cpu_metric is not None or memory_metric is not None
        
        # 4. Stop collection
        self.monitor.metrics_collector.stop_collection()
    
    def test_alert_workflow(self):
        """Test alert workflow"""
        # 1. Register a test handler
        alerts_received = []
        
        def test_handler(alert):
            alerts_received.append(alert)
        
        self.monitor.alert_manager.register_handler(AlertLevel.INFO, test_handler)
        
        # 2. Add some alerts
        self.monitor.alert_manager.add_alert(AlertLevel.INFO, "Alert 1", "source1")
        self.monitor.alert_manager.add_alert(AlertLevel.WARNING, "Alert 2", "source2")
        
        # 3. Check that handler was called
        assert len(alerts_received) == 1  # Only INFO level alerts
        assert alerts_received[0].message == "Alert 1"
        
        # 4. Get alerts
        all_alerts = self.monitor.alert_manager.get_alerts()
        assert len(all_alerts) == 2

class TestErrorHandling:
    """Test error handling in monitoring system"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = SystemMonitor(self.temp_dir)
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_health_checker_error_handling(self):
        """Test error handling in health checkers"""
        # Create a health checker that raises an exception
        class ErrorHealthChecker:
            @property
            def name(self):
                return "error_checker"
            
            async def check(self):
                raise Exception("Test error")
        
        # Add the error health checker
        self.monitor.health_checkers.append(ErrorHealthChecker())
        
        # Run health checks
        results = await self.monitor.run_health_checks()
        
        # Should handle the error gracefully
        assert "error_checker" in results
        assert results["error_checker"].status == HealthStatus.CRITICAL
        assert "failed" in results["error_checker"].message.lower()
    
    def test_metrics_collection_error_handling(self):
        """Test error handling in metrics collection"""
        # Mock psutil to raise an exception
        with patch('psutil.cpu_percent', side_effect=Exception("CPU error")):
            # Should not raise an exception
            self.monitor.metrics_collector._collect_system_metrics()
    
    def test_alert_manager_error_handling(self):
        """Test error handling in alert manager"""
        # Create a handler that raises an exception
        def error_handler(alert):
            raise Exception("Handler error")
        
        self.monitor.alert_manager.register_handler(AlertLevel.INFO, error_handler)
        
        # Add an alert - should not raise an exception
        self.monitor.alert_manager.add_alert(AlertLevel.INFO, "Test", "source")

if __name__ == "__main__":
    pytest.main([__file__]) 