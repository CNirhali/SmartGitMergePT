"""
Monitoring and Health Check System for SmartGitMergePT
Provides metrics collection, health checks, alerting, and system status monitoring
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import threading

import psutil
import requests
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout

# Configure logging
logger = logging.getLogger(__name__)
console = Console()

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

@dataclass
class Alert:
    level: AlertLevel
    message: str
    timestamp: datetime
    source: str
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Metric:
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)

class HealthChecker(ABC):
    """Abstract base class for health checks"""
    
    @abstractmethod
    async def check(self) -> HealthCheck:
        """Perform health check"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Health check name"""
        pass

class SystemHealthChecker(HealthChecker):
    """System-level health checks"""
    
    def __init__(self):
        self.name = "system"
    
    async def check(self) -> HealthCheck:
        start_time = time.time()
        
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Check memory usage
            memory = psutil.virtual_memory()
            
            # Check disk usage
            disk = psutil.disk_usage('/')
            
            # Determine status
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 95:
                status = HealthStatus.CRITICAL
                message = f"System resources critical: CPU {cpu_percent}%, Memory {memory.percent}%, Disk {disk.percent}%"
            elif cpu_percent > 80 or memory.percent > 80 or disk.percent > 85:
                status = HealthStatus.WARNING
                message = f"System resources warning: CPU {cpu_percent}%, Memory {memory.percent}%, Disk {disk.percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"System healthy: CPU {cpu_percent}%, Memory {memory.percent}%, Disk {disk.percent}%"
            
            duration = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name=self.name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                details={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': disk.percent,
                    'memory_available_mb': memory.available / (1024 * 1024),
                    'disk_free_gb': disk.free / (1024 * 1024 * 1024)
                },
                duration_ms=duration
            )
        
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return HealthCheck(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"System health check failed: {str(e)}",
                timestamp=datetime.now(),
                details={'error': str(e)},
                duration_ms=duration
            )

class DatabaseHealthChecker(HealthChecker):
    """Database health checks"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.name = "database"
    
    async def check(self) -> HealthCheck:
        start_time = time.time()
        
        try:
            import sqlite3
            
            # Check if database file exists
            if not os.path.exists(self.db_path):
                return HealthCheck(
                    name=self.name,
                    status=HealthStatus.WARNING,
                    message="Database file not found",
                    timestamp=datetime.now(),
                    details={'db_path': self.db_path},
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            # Test database connection
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            # Check database size
            db_size = os.path.getsize(self.db_path) / (1024 * 1024)  # MB
            
            conn.close()
            
            if db_size > 100:  # 100MB limit
                status = HealthStatus.WARNING
                message = f"Database size large: {db_size:.2f}MB"
            else:
                status = HealthStatus.HEALTHY
                message = f"Database healthy: {len(tables)} tables, {db_size:.2f}MB"
            
            duration = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name=self.name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                details={
                    'tables_count': len(tables),
                    'db_size_mb': db_size,
                    'tables': [table[0] for table in tables]
                },
                duration_ms=duration
            )
        
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return HealthCheck(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Database health check failed: {str(e)}",
                timestamp=datetime.now(),
                details={'error': str(e)},
                duration_ms=duration
            )

class NetworkHealthChecker(HealthChecker):
    """Network connectivity health checks"""
    
    def __init__(self, endpoints: List[str] = None):
        self.endpoints = endpoints or [
            "https://api.github.com",
            "https://httpbin.org/get",
            "https://www.google.com"
        ]
        self.name = "network"
    
    async def check(self) -> HealthCheck:
        start_time = time.time()
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                results = []
                
                for endpoint in self.endpoints:
                    try:
                        start = time.time()
                        async with session.get(endpoint, timeout=5) as response:
                            duration = time.time() - start
                            results.append({
                                'endpoint': endpoint,
                                'status': response.status,
                                'duration': duration,
                                'success': response.status < 400
                            })
                    except Exception as e:
                        results.append({
                            'endpoint': endpoint,
                            'status': 'error',
                            'duration': 0,
                            'success': False,
                            'error': str(e)
                        })
                
                # Calculate overall status
                successful = sum(1 for r in results if r['success'])
                total = len(results)
                success_rate = successful / total if total > 0 else 0
                
                if success_rate == 1.0:
                    status = HealthStatus.HEALTHY
                    message = f"Network healthy: {successful}/{total} endpoints reachable"
                elif success_rate >= 0.5:
                    status = HealthStatus.WARNING
                    message = f"Network warning: {successful}/{total} endpoints reachable"
                else:
                    status = HealthStatus.CRITICAL
                    message = f"Network critical: {successful}/{total} endpoints reachable"
                
                duration = (time.time() - start_time) * 1000
                
                return HealthCheck(
                    name=self.name,
                    status=status,
                    message=message,
                    timestamp=datetime.now(),
                    details={
                        'success_rate': success_rate,
                        'endpoints': results
                    },
                    duration_ms=duration
                )
        
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return HealthCheck(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Network health check failed: {str(e)}",
                timestamp=datetime.now(),
                details={'error': str(e)},
                duration_ms=duration
            )

class MetricsCollector:
    """Collects and stores system metrics"""
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics))
        self.collection_interval = 60  # seconds
        self.is_collecting = False
        self.collection_thread = None
    
    def start_collection(self):
        """Start metrics collection"""
        if not self.is_collecting:
            self.is_collecting = True
            self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
            self.collection_thread.start()
            logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join()
            logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Main collection loop"""
        while self.is_collecting:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(10)  # Wait before retry
    
    def _collect_system_metrics(self):
        """Collect system metrics"""
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self.add_metric("cpu_usage", cpu_percent, timestamp)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.add_metric("memory_usage", memory.percent, timestamp)
        self.add_metric("memory_available_mb", memory.available / (1024 * 1024), timestamp)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.add_metric("disk_usage", disk.percent, timestamp)
        self.add_metric("disk_free_gb", disk.free / (1024 * 1024 * 1024), timestamp)
        
        # Network metrics
        net_io = psutil.net_io_counters()
        self.add_metric("network_bytes_sent", net_io.bytes_sent, timestamp)
        self.add_metric("network_bytes_recv", net_io.bytes_recv, timestamp)
        
        # Process metrics
        process = psutil.Process()
        self.add_metric("process_memory_mb", process.memory_info().rss / (1024 * 1024), timestamp)
        self.add_metric("process_cpu_percent", process.cpu_percent(), timestamp)
    
    def add_metric(self, name: str, value: float, timestamp: datetime, tags: Dict[str, str] = None):
        """Add a metric"""
        metric = Metric(
            name=name,
            value=value,
            timestamp=timestamp,
            tags=tags or {}
        )
        self.metrics[name].append(metric)
    
    def get_metrics(self, name: str, minutes: int = 60) -> List[Metric]:
        """Get metrics for a specific name within time range"""
        if name not in self.metrics:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.metrics[name] if m.timestamp > cutoff_time]
    
    def get_latest_metric(self, name: str) -> Optional[Metric]:
        """Get the latest metric for a name"""
        if name in self.metrics and self.metrics[name]:
            return self.metrics[name][-1]
        return None
    
    def get_metric_stats(self, name: str, minutes: int = 60) -> Dict[str, float]:
        """Get statistics for a metric"""
        metrics = self.get_metrics(name, minutes)
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        return {
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'count': len(values)
        }

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_handlers: Dict[AlertLevel, List[Callable]] = defaultdict(list)
        self.alert_filters: Dict[str, Callable] = {}
    
    def add_alert(self, level: AlertLevel, message: str, source: str, details: Dict[str, Any] = None):
        """Add an alert"""
        alert = Alert(
            level=level,
            message=message,
            timestamp=datetime.now(),
            source=source,
            details=details or {}
        )
        
        self.alerts.append(alert)
        
        # Call alert handlers
        for handler in self.alert_handlers[level]:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
        # Log critical alerts
        if level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
            logger.error(f"ALERT [{level.value.upper()}] {source}: {message}")
    
    def register_handler(self, level: AlertLevel, handler: Callable):
        """Register an alert handler"""
        self.alert_handlers[level].append(handler)
    
    def get_alerts(self, level: Optional[AlertLevel] = None, hours: int = 24) -> List[Alert]:
        """Get alerts within time range"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return alerts
    
    def clear_old_alerts(self, days: int = 7):
        """Clear old alerts"""
        cutoff_time = datetime.now() - timedelta(days=days)
        self.alerts = [a for a in self.alerts if a.timestamp > cutoff_time]

class SystemMonitor:
    """Main system monitoring and health check system"""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.db_path = os.path.join(repo_path, ".smartgit_tracker.db")
        
        # Initialize components
        self.health_checkers: List[HealthChecker] = [
            SystemHealthChecker(),
            DatabaseHealthChecker(self.db_path),
            NetworkHealthChecker()
        ]
        
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        
        # Health check results
        self.health_results: Dict[str, HealthCheck] = {}
        self.last_check_time = None
        
        # Setup default alert handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default alert handlers"""
        def log_handler(alert: Alert):
            logger.info(f"Alert: {alert.level.value} - {alert.message}")
        
        def console_handler(alert: Alert):
            if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
                console.print(f"[red]ALERT: {alert.message}[/red]")
            elif alert.level == AlertLevel.WARNING:
                console.print(f"[yellow]WARNING: {alert.message}[/yellow]")
            else:
                console.print(f"[blue]INFO: {alert.message}[/blue]")
        
        # Register handlers
        for level in AlertLevel:
            self.alert_manager.register_handler(level, log_handler)
            self.alert_manager.register_handler(level, console_handler)
    
    async def run_health_checks(self) -> Dict[str, HealthCheck]:
        """Run all health checks"""
        results = {}
        
        for checker in self.health_checkers:
            try:
                result = await checker.check()
                results[checker.name] = result
                
                # Generate alerts based on health check results
                if result.status == HealthStatus.CRITICAL:
                    self.alert_manager.add_alert(
                        AlertLevel.CRITICAL,
                        result.message,
                        f"health_check.{checker.name}",
                        result.details
                    )
                elif result.status == HealthStatus.WARNING:
                    self.alert_manager.add_alert(
                        AlertLevel.WARNING,
                        result.message,
                        f"health_check.{checker.name}",
                        result.details
                    )
                
            except Exception as e:
                logger.error(f"Health check failed for {checker.name}: {e}")
                results[checker.name] = HealthCheck(
                    name=checker.name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}",
                    timestamp=datetime.now(),
                    details={'error': str(e)}
                )
        
        self.health_results = results
        self.last_check_time = datetime.now()
        
        return results
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status"""
        if not self.health_results:
            return HealthStatus.UNKNOWN
        
        # Check if any critical issues
        if any(r.status == HealthStatus.CRITICAL for r in self.health_results.values()):
            return HealthStatus.CRITICAL
        
        # Check if any warnings
        if any(r.status == HealthStatus.WARNING for r in self.health_results.values()):
            return HealthStatus.WARNING
        
        return HealthStatus.HEALTHY
    
    def start_monitoring(self):
        """Start monitoring system"""
        self.metrics_collector.start_collection()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring system"""
        self.metrics_collector.stop_collection()
        logger.info("System monitoring stopped")
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report"""
        overall_health = self.get_overall_health()
        
        # Get latest metrics
        latest_metrics = {}
        for metric_name in ['cpu_usage', 'memory_usage', 'disk_usage']:
            metric = self.metrics_collector.get_latest_metric(metric_name)
            if metric:
                latest_metrics[metric_name] = metric.value
        
        # Get recent alerts
        recent_alerts = self.alert_manager.get_alerts(hours=1)
        
        return {
            'overall_health': overall_health.value,
            'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
            'health_checks': {
                name: {
                    'status': result.status.value,
                    'message': result.message,
                    'duration_ms': result.duration_ms
                }
                for name, result in self.health_results.items()
            },
            'latest_metrics': latest_metrics,
            'recent_alerts': len(recent_alerts),
            'alert_summary': {
                level.value: len(self.alert_manager.get_alerts(level=level, hours=24))
                for level in AlertLevel
            }
        }
    
    def display_status(self):
        """Display system status in rich format"""
        status = self.get_status_report()
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="health", size=10),
            Layout(name="metrics", size=8),
            Layout(name="alerts", size=6)
        )
        
        # Header
        health_icon = "🟢" if status['overall_health'] == 'healthy' else "🔴"
        layout["header"].update(Panel(
            f"{health_icon} SmartGitMergePT System Status - {status['overall_health'].upper()}",
            style="bold blue"
        ))
        
        # Health checks table
        health_table = Table(title="Health Checks")
        health_table.add_column("Service", style="cyan")
        health_table.add_column("Status", style="magenta")
        health_table.add_column("Message", style="green")
        health_table.add_column("Duration (ms)", style="yellow")
        
        for name, check in status['health_checks'].items():
            status_icon = "🟢" if check['status'] == 'healthy' else "🔴"
            health_table.add_row(
                name,
                f"{status_icon} {check['status']}",
                check['message'][:50] + "..." if len(check['message']) > 50 else check['message'],
                f"{check['duration_ms']:.1f}"
            )
        
        layout["health"].update(health_table)
        
        # Metrics
        metrics_text = ""
        for metric_name, value in status['latest_metrics'].items():
            metrics_text += f"{metric_name}: {value:.2f}\n"
        
        layout["metrics"].update(Panel(
            f"Latest Metrics:\n{metrics_text}",
            title="System Metrics"
        ))
        
        # Alerts
        alerts_text = f"Recent Alerts: {status['recent_alerts']}\n"
        for level, count in status['alert_summary'].items():
            alerts_text += f"{level}: {count}\n"
        
        layout["alerts"].update(Panel(
            alerts_text,
            title="Alert Summary"
        ))
        
        console.print(layout)

# Live monitoring display
class LiveMonitor:
    """Live monitoring display with rich interface"""
    
    def __init__(self, monitor: SystemMonitor):
        self.monitor = monitor
        self.is_running = False
    
    async def start_live_display(self, update_interval: int = 30):
        """Start live monitoring display"""
        self.is_running = True
        
        with Live(self.monitor.display_status, refresh_per_second=1) as live:
            while self.is_running:
                # Run health checks
                await self.monitor.run_health_checks()
                
                # Update display
                live.update(self.monitor.display_status)
                
                # Wait for next update
                await asyncio.sleep(update_interval)
    
    def stop_live_display(self):
        """Stop live monitoring display"""
        self.is_running = False

# Utility functions
def create_monitor(repo_path: str) -> SystemMonitor:
    """Create a system monitor instance"""
    return SystemMonitor(repo_path)

async def run_health_checks(repo_path: str) -> Dict[str, HealthCheck]:
    """Run health checks for a repository"""
    monitor = create_monitor(repo_path)
    return await monitor.run_health_checks()

def get_system_status(repo_path: str) -> Dict[str, Any]:
    """Get system status for a repository"""
    monitor = create_monitor(repo_path)
    return monitor.get_status_report()

# Command line interface
def main():
    """Main monitoring interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SmartGitMergePT System Monitor')
    parser.add_argument('--repo-path', default='.', help='Repository path')
    parser.add_argument('--live', action='store_true', help='Start live monitoring')
    parser.add_argument('--check', action='store_true', help='Run health checks')
    parser.add_argument('--status', action='store_true', help='Show system status')
    
    args = parser.parse_args()
    
    monitor = create_monitor(args.repo_path)
    
    if args.live:
        console.print("Starting live monitoring... Press Ctrl+C to stop")
        try:
            asyncio.run(LiveMonitor(monitor).start_live_display())
        except KeyboardInterrupt:
            console.print("\nMonitoring stopped")
    
    elif args.check:
        console.print("Running health checks...")
        results = asyncio.run(monitor.run_health_checks())
        monitor.display_status()
    
    elif args.status:
        monitor.display_status()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 