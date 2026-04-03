import argparse
import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor
from git_utils import GitUtils
from predictor import ConflictPredictor
from llm_resolver import resolve_conflict_with_mistral
from guardrails import InputValidator

# Import guardrails and optimization systems
try:
    from guardrails import GuardrailsManager, with_guardrails, secure_function, resource_guard
    from optimizer import PerformanceOptimizer, CacheConfig, PerformanceConfig, performance_monitor
    from monitor import SystemMonitor, create_monitor
    GUARDRAILS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Guardrails and optimization modules not available: {e}")
    GUARDRAILS_AVAILABLE = False

# Import new agentic AI tracking modules
try:
    from agentic_tracker import AgenticTracker
    from ai_validator import AIValidator
    from config_manager import ConfigManager
    AGENTIC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Agentic AI modules not available: {e}")
    AGENTIC_AVAILABLE = False
    # Define stubs for type hints and to avoid NameError when modules are missing
    class ConfigManager:
        def __init__(self, *args, **kwargs): pass
    class AgenticTracker:
        def __init__(self, *args, **kwargs): pass
    class AIValidator:
        def __init__(self, *args, **kwargs): pass

# Import code review modules
try:
    from code_reviewer import MistralCodeReviewer, ReviewConfig, run_pre_commit_review
    from hook_installer import GitHookInstaller
    CODE_REVIEW_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Code review modules not available: {e}")
    CODE_REVIEW_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
guardrails_manager = None
performance_optimizer = None
system_monitor = None

def main():
    parser = argparse.ArgumentParser(description='SmartGitMergePT: LLM-based Git merge conflict resolver and predictor with Agentic AI tracking')
    subparsers = parser.add_subparsers(dest='command')

    # Original commands
    subparsers.add_parser('predict', help='Predict potential merge conflicts')
    subparsers.add_parser('detect', help='Detect actual merge conflicts')
    subparsers.add_parser('resolve', help='Resolve merge conflicts using LLM')
    subparsers.add_parser('dashboard', help='Show team conflict dashboard')

    # New agentic AI tracking commands
    if AGENTIC_AVAILABLE:
        subparsers.add_parser('track', help='Start agentic AI tracking for a developer')
        subparsers.add_parser('stop-track', help='Stop agentic AI tracking')
        subparsers.add_parser('validate', help='Validate tracked session data with AI')
        subparsers.add_parser('estimates', help='Generate project estimates from tracked data')
        subparsers.add_parser('stats', help='Show developer work statistics')
        subparsers.add_parser('config', help='Manage agentic AI tracking configuration')
        subparsers.add_parser('demo', help='Run comprehensive agentic AI demo')

    # New guardrails and optimization commands
    if GUARDRAILS_AVAILABLE:
        subparsers.add_parser('monitor', help='Start system monitoring')
        subparsers.add_parser('health', help='Run health checks')
        subparsers.add_parser('optimize', help='Run system optimization')
        subparsers.add_parser('guardrails', help='Show guardrails status')
        subparsers.add_parser('performance', help='Show performance metrics')

    # New code review commands
    if CODE_REVIEW_AVAILABLE:
        subparsers.add_parser('review', help='Review staged changes with Mistral')
        subparsers.add_parser('review-file', help='Review a specific file')
        subparsers.add_parser('install-hook', help='Install pre-commit hook for code review')
        subparsers.add_parser('uninstall-hook', help='Remove pre-commit hook')
        subparsers.add_parser('test-hook', help='Test the installed pre-commit hook')
        subparsers.add_parser('check-ollama', help='Check Ollama status for code review')

    args = parser.parse_args()
    repo_path = "demo/conflict_scenarios/demo-repo"
    
    # Initialize guardrails and optimization systems
    global guardrails_manager, performance_optimizer, system_monitor
    if GUARDRAILS_AVAILABLE:
        guardrails_manager = GuardrailsManager()
        cache_config = CacheConfig()
        perf_config = PerformanceConfig()
        performance_optimizer = PerformanceOptimizer(cache_config, perf_config)
        system_monitor = create_monitor(repo_path)
        logger.info("✅ Guardrails and optimization systems initialized")
    
    git_utils = GitUtils(repo_path)
    predictor = ConflictPredictor(repo_path)

    if args.command == 'predict':
        branches = git_utils.list_branches()
        predictions = predictor.predict_conflicts(branches)
        if not predictions:
            print("No likely conflicts detected between branches.")
        else:
            print("Predicted conflicts:")
            for pred in predictions:
                print(f"Branches: {pred['branches']}, Files: {pred['files']}")

    elif args.command == 'detect':
        branches = git_utils.list_branches()
        # BOLT: Optimization - Use ConflictPredictor to pre-filter branch pairs.
        # Only perform the expensive 'git merge-tree' simulation for pairs that
        # are predicted to have at least one modified file in common.
        # This reduces the number of git calls from O(N^2) to O(Pairs with overlap).
        predictions = predictor.predict_conflicts(branches)
        # BOLT: Normalize candidate branch pairs as sorted tuples for robust hash-based lookups.
        candidate_pairs = [tuple(sorted(p['branches'])) for p in predictions]

        # BOLT: Only submit candidate branch pairs to the ThreadPoolExecutor to minimize task overhead.
        results = {}
        if candidate_pairs:
            def check_pair(pair):
                branch_a, branch_b = pair
                ok, msg = git_utils.simulate_merge(branch_a, branch_b)
                return (pair, ok, msg)

            with ThreadPoolExecutor() as executor:
                for pair, ok, msg in executor.map(check_pair, candidate_pairs):
                    results[pair] = (ok, msg)

        # To maintain original output behavior (listing "No conflict" for all),
        # we iterate through all possible pairs and use simulation results if available.
        for i, branch_a in enumerate(branches):
            for branch_b in branches[i+1:]:
                # BOLT: Check if this pair was simulated. ConflictPredictor already uses sorted pairs.
                pair_key = (branch_a, branch_b) if branch_a < branch_b else (branch_b, branch_a)
                if pair_key in results:
                    ok, msg = results[pair_key]
                    if not ok:
                        print(f"Conflict detected merging {branch_a} into {branch_b}: {msg}")
                    else:
                        print(f"No conflict merging {branch_a} into {branch_b}.")
                else:
                    print(f"No conflict merging {branch_a} into {branch_b}. (No shared files modified)")

    elif args.command == 'resolve':
        print("Paste the merge conflict block (end with EOF / Ctrl-D):")
        import sys
        conflict_block = sys.stdin.read()
        resolved = resolve_conflict_with_mistral(conflict_block)
        print("\nResolved block:\n")
        print(resolved)

    elif args.command == 'dashboard':
        branches = git_utils.list_branches()
        predictions = predictor.predict_conflicts(branches)
        print("=== Team Conflict Dashboard ===")
        print(f"Branches: {branches}")
        if not predictions:
            print("No likely conflicts detected.")
        else:
            print("Predicted conflicts:")
            for pred in predictions:
                print(f"Branches: {pred['branches']}, Files: {pred['files']}")

    # New agentic AI tracking commands
    elif args.command == 'track' and AGENTIC_AVAILABLE:
        track_developer(repo_path)

    elif args.command == 'stop-track' and AGENTIC_AVAILABLE:
        stop_tracking(repo_path)

    elif args.command == 'validate' and AGENTIC_AVAILABLE:
        validate_session_data(repo_path)

    elif args.command == 'estimates' and AGENTIC_AVAILABLE:
        generate_project_estimates(repo_path)

    elif args.command == 'stats' and AGENTIC_AVAILABLE:
        show_developer_stats(repo_path)

    elif args.command == 'config' and AGENTIC_AVAILABLE:
        manage_configuration(repo_path)

    elif args.command == 'demo' and AGENTIC_AVAILABLE:
        run_agentic_demo(repo_path)

    # New guardrails and optimization commands
    elif args.command == 'monitor' and GUARDRAILS_AVAILABLE:
        start_system_monitoring(repo_path)

    elif args.command == 'health' and GUARDRAILS_AVAILABLE:
        run_health_checks(repo_path)

    elif args.command == 'optimize' and GUARDRAILS_AVAILABLE:
        run_system_optimization(repo_path)

    elif args.command == 'guardrails' and GUARDRAILS_AVAILABLE:
        show_guardrails_status()

    elif args.command == 'performance' and GUARDRAILS_AVAILABLE:
        show_performance_metrics()

    # New code review commands
    elif args.command == 'review' and CODE_REVIEW_AVAILABLE:
        review_staged_changes(repo_path)

    elif args.command == 'review-file' and CODE_REVIEW_AVAILABLE:
        review_specific_file(repo_path)

    elif args.command == 'install-hook' and CODE_REVIEW_AVAILABLE:
        install_pre_commit_hook(repo_path)

    elif args.command == 'uninstall-hook' and CODE_REVIEW_AVAILABLE:
        uninstall_pre_commit_hook(repo_path)

    elif args.command == 'test-hook' and CODE_REVIEW_AVAILABLE:
        test_pre_commit_hook(repo_path)

    elif args.command == 'check-ollama' and CODE_REVIEW_AVAILABLE:
        check_ollama_status()

    else:
        parser.print_help()

def track_developer(repo_path: str):
    """Start tracking a developer's work session"""
    print("🤖 Starting Agentic AI Developer Tracking...")
    
    # Get developer info
    developer_id = input("Enter developer ID: ").strip()
    if not developer_id:
        developer_id = "alice_dev_001"
        print(f"Using default developer ID: {developer_id}")
    
    # Get branch info
    branch_name = input("Enter branch name: ").strip()
    if not branch_name:
        branch_name = "feature/user-authentication"
        print(f"Using default branch: {branch_name}")
    
    # Initialize tracker
    tracker = AgenticTracker(repo_path)
    
    try:
        # Start tracking
        tracker.start_tracking(developer_id, branch_name)
        
        print(f"✅ Started tracking {developer_id} on branch {branch_name}")
        print("📊 Tracking in progress... Press Ctrl+C to stop")
        
        # Keep tracking until interrupted
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  Stopping tracking...")
            tracker.stop_tracking()
            print("✅ Tracking stopped")
    
    except Exception as e:
        print(f"❌ Error starting tracking: {e}")

def stop_tracking(repo_path: str):
    """Stop current tracking session"""
    print("⏹️  Stopping Agentic AI tracking...")
    
    tracker = AgenticTracker(repo_path)
    tracker.stop_tracking()
    print("✅ Tracking stopped")

def validate_session_data(repo_path: str):
    """Validate session data using AI"""
    print("🔍 Validating session data with AI...")
    
    # Check for OpenAI API key
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("❌ OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        return
    
    validator = AIValidator(openai_api_key, repo_path)
    
    # Get session data from database
    tracker = AgenticTracker(repo_path)
    
    # For demo purposes, create sample session data
    session_data = {
        'developer_id': 'alice_dev_001',
        'branch_name': 'feature/user-authentication',
        'start_time': '2024-01-15T09:00:00',
        'end_time': '2024-01-15T12:00:00',
        'duration_hours': 3.0,
        'activity_type': 'coding',
        'confidence_score': 0.85,
        'screen_activity': True,
        'face_detected': True,
        'validation_status': 'pending'
    }
    
    # Validate session
    validation_result = validator.validate_work_session(session_data)
    
    print(f"✅ Validation Status: {'VALID' if validation_result.is_valid else 'INVALID'}")
    print(f"🎯 Confidence Score: {validation_result.confidence_score:.2f}")
    print(f"📊 Validation Level: {validation_result.validation_level.value.upper()}")
    
    if validation_result.issues:
        print("⚠️  Issues Found:")
        for issue in validation_result.issues:
            print(f"   • {issue}")
    
    if validation_result.recommendations:
        print("💡 Recommendations:")
        for rec in validation_result.recommendations:
            print(f"   • {rec}")

def generate_project_estimates(repo_path: str):
    """Generate project estimates from tracked data"""
    print("📊 Generating project estimates...")
    
    tracker = AgenticTracker(repo_path)
    
    # Get estimates for all branches
    git_utils = GitUtils(repo_path)
    branches = git_utils.list_branches()
    
    print("📈 Project Estimates:")
    print("=" * 50)
    
    for branch in branches:
        estimates = tracker.get_project_estimates(branch)
        print(f"\n🌿 Branch: {estimates['branch_name']}")
        print(f"⏱️  Total Hours: {estimates['total_hours_spent']:.2f}")
        print(f"👥 Developers: {estimates['developers_working']}")
        print(f"🎯 Avg Confidence: {estimates['avg_confidence']:.2f}")
        print(f"📊 Data Quality: {estimates['data_quality'].upper()}")
        print(f"⏳ Est. Completion: {estimates['estimated_completion_hours']:.2f} hours")

def show_developer_stats(repo_path: str):
    """Show developer work statistics"""
    print("📊 Developer Work Statistics...")
    
    tracker = AgenticTracker(repo_path)
    
    # Get developer ID
    developer_id = input("Enter developer ID (or press Enter for default): ").strip()
    if not developer_id:
        developer_id = "alice_dev_001"
    
    # Get time period
    days = input("Enter number of days (or press Enter for 7): ").strip()
    if not days:
        days = 7
    else:
        days = int(days)
    
    stats = tracker.get_developer_stats(developer_id, days)
    
    print(f"\n👤 Developer: {stats['developer_id']}")
    print(f"📅 Period: Last {stats['period_days']} days")
    print(f"⏱️  Total Hours: {stats['total_hours']:.2f}")
    print(f"📅 Sessions: {stats['sessions']}")
    print(f"🌿 Branches: {', '.join(stats['branches'])}")
    print(f"🎯 Avg Confidence: {stats['avg_confidence']:.2f}")
    print(f"✅ Validation Rate: {stats['validation_rate']:.1%}")

def manage_configuration(repo_path: str):
    """Manage agentic AI tracking configuration"""
    print("🔧 Agentic AI Configuration Manager")
    
    config_manager = ConfigManager(repo_path)
    
    while True:
        print("\n" + "=" * 50)
        print("Configuration Options:")
        print("1. Show current configuration")
        print("2. Create privacy-compliant config")
        print("3. Create performance-optimized config")
        print("4. Create high-accuracy config")
        print("5. Configure LLM settings")
        print("6. Export configuration")
        print("7. Import configuration")
        print("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == '1':
            config_manager.print_current_config()
        elif choice == '2':
            config_manager.create_privacy_compliant_config()
        elif choice == '3':
            config_manager.create_performance_optimized_config()
        elif choice == '4':
            config_manager.create_high_accuracy_config()
        elif choice == '5':
            configure_llm_settings(config_manager)
        elif choice == '6':
            filepath = input("Enter export filepath: ").strip()
            if filepath:
                config_manager.export_config(filepath)
        elif choice == '7':
            filepath = input("Enter import filepath: ").strip()
            if filepath:
                config_manager.import_config(filepath)
        elif choice == '8':
            print("✅ Configuration management completed")
            break
        else:
            print("❌ Invalid choice. Please try again.")

def configure_llm_settings(config_manager: ConfigManager):
    """Configure LLM settings interactively"""
    print("\n🤖 LLM Configuration")
    print("=" * 30)
    
    current_config = config_manager.get_llm_config()
    print(f"Current provider: {current_config.get('provider', 'ollama')}")
    print(f"Current Ollama endpoint: {current_config.get('ollama_endpoint', 'http://localhost:11434/v1/chat/completions')}")
    print(f"Current Ollama model: {current_config.get('ollama_model', 'mistral')}")
    print(f"Current OpenAI model: {current_config.get('openai_model', 'gpt-3.5-turbo')}")
    print(f"Current Mistral endpoint: {current_config.get('mistral_endpoint', 'http://localhost:11434/v1/chat/completions')}")
    print(f"Current Mistral model: {current_config.get('mistral_model', 'mistral-7b-instruct')}")
    
    print("\nLLM Provider Options:")
    print("1. Ollama (local - recommended)")
    print("2. OpenAI (cloud-based)")
    print("3. Mistral (local - legacy)")
    
    provider_choice = input("\nSelect LLM provider (1, 2, or 3): ").strip()
    
    if provider_choice == '1':
        # Ollama configuration
        provider = 'ollama'
        endpoint = input("Enter Ollama endpoint (default: http://localhost:11434/v1/chat/completions): ").strip()
        if not endpoint:
            endpoint = 'http://localhost:11434/v1/chat/completions'
        
        # 🛡️ Sentinel: Validate URL to prevent SSRF (allow local for Ollama)
        is_valid, msg = InputValidator().validate_url(endpoint, allow_local=True)
        if not is_valid:
            print(f"❌ Invalid endpoint: {msg}")
            return

        model = input("Enter Ollama model name (default: mistral): ").strip()
        if not model:
            model = 'mistral'
        
        config_manager.update_llm_config(
            provider=provider,
            ollama_endpoint=endpoint,
            ollama_model=model
        )
        print("✅ Ollama configuration updated")
        print("⚠️  Make sure Ollama is running: ollama serve")
        print("⚠️  Make sure model is pulled: ollama pull mistral")
        
    elif provider_choice == '2':
        # OpenAI configuration
        provider = 'openai'
        model = input("Enter OpenAI model (default: gpt-3.5-turbo): ").strip()
        if not model:
            model = 'gpt-3.5-turbo'
        
        config_manager.update_llm_config(
            provider=provider,
            openai_model=model
        )
        print("✅ OpenAI configuration updated")
        print("⚠️  Remember to set OPENAI_API_KEY environment variable")
        
    elif provider_choice == '3':
        # Mistral configuration
        provider = 'mistral'
        endpoint = input("Enter Mistral endpoint (default: http://localhost:11434/v1/chat/completions): ").strip()
        if not endpoint:
            endpoint = 'http://localhost:11434/v1/chat/completions'
        
        # 🛡️ Sentinel: Validate URL to prevent SSRF (allow local for legacy Mistral)
        is_valid, msg = InputValidator().validate_url(endpoint, allow_local=True)
        if not is_valid:
            print(f"❌ Invalid endpoint: {msg}")
            return

        model = input("Enter Mistral model name (default: mistral-7b-instruct): ").strip()
        if not model:
            model = 'mistral-7b-instruct'
        
        config_manager.update_llm_config(
            provider=provider,
            mistral_endpoint=endpoint,
            mistral_model=model
        )
        print("✅ Mistral configuration updated")
        print("⚠️  Make sure your local Mistral server is running")
        
    else:
        print("❌ Invalid choice. LLM configuration not changed.")
        return
    
    # Show updated configuration
    print("\nUpdated LLM Configuration:")
    updated_config = config_manager.get_llm_config()
    for key, value in updated_config.items():
        print(f"  {key}: {value}")
    
    # Test connection if local provider
    if provider in ['ollama', 'mistral']:
        test_connection = input(f"\nTest {provider.title()} connection? (y/n): ").strip().lower()
        if test_connection == 'y':
            if provider == 'ollama':
                test_ollama_connection(endpoint, model)
            else:
                test_mistral_connection(endpoint, model)

def test_ollama_connection(endpoint: str, model: str):
    """Test connection to local Ollama server with SSRF protection"""
    # 🛡️ Sentinel: Validate URL to prevent SSRF (allow local)
    is_valid, msg = InputValidator().validate_url(endpoint, allow_local=True)
    if not is_valid:
        print(f"❌ Connection test failed: {msg}")
        return

    print(f"🧪 Testing Ollama connection to {endpoint}...")
    
    try:
        import requests
        import json
        
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Hello, this is a test message."}
            ],
            "max_tokens": 10,
            "temperature": 0.1
        }
        
        response = requests.post(endpoint, headers=headers, data=json.dumps(data), timeout=30)
        response.raise_for_status()
        
        result = response.json()
        if 'choices' in result or 'message' in result or 'content' in result:
            print("✅ Ollama connection successful!")
            print("💡 Make sure you have the model pulled: ollama pull mistral")
        else:
            print("⚠️  Connection successful but unexpected response format")
            print(f"Response: {result}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Make sure Ollama is running:")
        print("   1. Install Ollama: https://ollama.ai/")
        print("   2. Start server: ollama serve")
        print("   3. Pull model: ollama pull mistral")
    except Exception as e:
        print(f"❌ Connection test failed: {e}")

def test_mistral_connection(endpoint: str, model: str):
    """Test connection to local Mistral server with SSRF protection"""
    # 🛡️ Sentinel: Validate URL to prevent SSRF (allow local)
    is_valid, msg = InputValidator().validate_url(endpoint, allow_local=True)
    if not is_valid:
        print(f"❌ Connection test failed: {msg}")
        return

    print(f"🧪 Testing Mistral connection to {endpoint}...")
    
    try:
        import requests
        import json
        
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Hello, this is a test message."}
            ],
            "max_tokens": 10,
            "temperature": 0.1
        }
        
        response = requests.post(endpoint, headers=headers, data=json.dumps(data), timeout=30)
        response.raise_for_status()
        
        result = response.json()
        if 'choices' in result or 'message' in result or 'content' in result:
            print("✅ Mistral connection successful!")
        else:
            print("⚠️  Connection successful but unexpected response format")
            print(f"Response: {result}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Make sure your Mistral server is running.")
        print("   Common endpoints:")
        print("   - Ollama: http://localhost:11434/v1/chat/completions")
        print("   - LMStudio: http://localhost:1234/v1/chat/completions")
        print("   - Custom: Your custom endpoint")
    except Exception as e:
        print(f"❌ Connection test failed: {e}")

def run_agentic_demo(repo_path: str):
    """Run the comprehensive agentic AI demo"""
    print("🎬 Running Agentic AI Demo...")
    
    try:
        from agentic_demo import AgenticDemo
        
        # Check for OpenAI API key
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            print("⚠️  Warning: No OpenAI API key found. AI validation will be limited.")
            print("   Set OPENAI_API_KEY environment variable for full AI validation.")
        
        # Create and run demo
        demo = AgenticDemo(repo_path, openai_api_key)
        results = demo.run_comprehensive_demo()
        
        print("✅ Agentic AI demo completed successfully!")
        
    except ImportError:
        print("❌ Agentic demo module not available")
    except Exception as e:
        print(f"❌ Demo failed: {e}")

# Guardrails and optimization functions
def start_system_monitoring(repo_path: str):
    """Start system monitoring"""
    print("🔍 Starting system monitoring...")
    
    try:
        import asyncio
        from monitor import LiveMonitor
        
        monitor = create_monitor(repo_path)
        monitor.start_monitoring()
        
        print("✅ System monitoring started")
        print("📊 Press Ctrl+C to stop monitoring")
        
        # Start live monitoring
        live_monitor = LiveMonitor(monitor)
        asyncio.run(live_monitor.start_live_display())
        
    except KeyboardInterrupt:
        print("\n⏹️  Stopping monitoring...")
        monitor.stop_monitoring()
        print("✅ Monitoring stopped")
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")

def run_health_checks(repo_path: str):
    """Run health checks"""
    print("🏥 Running health checks...")
    
    try:
        import asyncio
        
        monitor = create_monitor(repo_path)
        results = asyncio.run(monitor.run_health_checks())
        
        print("✅ Health checks completed")
        monitor.display_status()
        
    except Exception as e:
        print(f"❌ Health checks failed: {e}")

def run_system_optimization(repo_path: str):
    """Run system optimization"""
    print("⚡ Running system optimization...")
    
    try:
        global performance_optimizer
        
        if performance_optimizer:
            performance_optimizer.optimize_system()
            stats = performance_optimizer.get_performance_stats()
            
            print("✅ System optimization completed")
            print(f"📊 Cache hit rate: {stats['cache']['hit_rate']:.2f}%")
            print(f"💾 Memory optimizations: {stats['optimizations']['memory_optimizations']}")
            
        else:
            print("❌ Performance optimizer not available")
            
    except Exception as e:
        print(f"❌ Optimization failed: {e}")

def show_guardrails_status():
    """Show guardrails status"""
    print("🛡️  Guardrails Status")
    
    try:
        global guardrails_manager
        
        if guardrails_manager:
            health_status = guardrails_manager.get_health_status()
            
            print(f"Overall Health: {'✅ Healthy' if health_status['overall_health'] else '❌ Unhealthy'}")
            print(f"Performance: {'✅ Healthy' if health_status['performance']['healthy'] else '❌ Unhealthy'}")
            print(f"Security Events (24h): {health_status['security']['total_events_24h']}")
            print(f"Critical Events: {health_status['security']['critical_events']}")
            print(f"Total Errors: {health_status['errors']['total_errors']}")
            
        else:
            print("❌ Guardrails manager not available")
            
    except Exception as e:
        print(f"❌ Failed to get guardrails status: {e}")

def show_performance_metrics():
    """Show performance metrics"""
    print("📊 Performance Metrics")
    
    try:
        global performance_optimizer
        
        if performance_optimizer:
            stats = performance_optimizer.get_performance_stats()
            
            print(f"Cache Hit Rate: {stats['cache']['hit_rate']:.2f}%")
            print(f"Cache Size: {stats['cache']['size']} items")
            print(f"Memory Usage: {stats['resources']['memory']['usage_mb']:.1f} MB")
            print(f"CPU Usage: {stats['resources']['cpu']['usage_percent']:.1f}%")
            print(f"Async Operations: {stats['optimizations']['async_operations']}")
            print(f"Memory Optimizations: {stats['optimizations']['memory_optimizations']}")
            
        else:
            print("❌ Performance optimizer not available")
            
    except Exception as e:
        print(f"❌ Failed to get performance metrics: {e}")

# Code Review Functions
def review_staged_changes(repo_path: str):
    """Review staged changes with Mistral"""
    print("🔍 Reviewing staged changes with Mistral...")
    
    try:
        success = run_pre_commit_review(repo_path)
        if success:
            print("✅ Code review completed successfully")
        else:
            print("❌ Code review found issues")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Code review failed: {e}")
        sys.exit(1)

def review_specific_file(repo_path: str):
    """Review a specific file"""
    print("🔍 Reviewing specific file with Mistral...")
    
    try:
        file_path = input("Enter file path to review: ").strip()
        if not file_path:
            print("❌ No file path provided")
            return
        
        reviewer = MistralCodeReviewer(repo_path)
        result = reviewer.review_file(file_path)
        
        print("\n" + reviewer.generate_report([result]))
        
    except Exception as e:
        print(f"❌ File review failed: {e}")

def install_pre_commit_hook(repo_path: str):
    """Install pre-commit hook for code review"""
    print("🔧 Installing pre-commit hook for Mistral code review...")
    
    try:
        installer = GitHookInstaller(repo_path)
        
        if installer.install_pre_commit_hook():
            print("✅ Pre-commit hook installed successfully")
            print("🔍 Code review will now run automatically before each commit")
        else:
            print("❌ Failed to install pre-commit hook")
            
    except Exception as e:
        print(f"❌ Hook installation failed: {e}")

def uninstall_pre_commit_hook(repo_path: str):
    """Remove pre-commit hook"""
    print("🔧 Removing pre-commit hook...")
    
    try:
        installer = GitHookInstaller(repo_path)
        
        if installer.uninstall_pre_commit_hook():
            print("✅ Pre-commit hook removed successfully")
        else:
            print("❌ Failed to remove pre-commit hook")
            
    except Exception as e:
        print(f"❌ Hook removal failed: {e}")

def test_pre_commit_hook(repo_path: str):
    """Test the installed pre-commit hook"""
    print("🧪 Testing pre-commit hook...")
    
    try:
        installer = GitHookInstaller(repo_path)
        
        if installer.test_hook():
            print("✅ Pre-commit hook test completed")
        else:
            print("❌ Pre-commit hook test failed")
            
    except Exception as e:
        print(f"❌ Hook test failed: {e}")

def check_ollama_status():
    """Check Ollama status for code review"""
    print("🔍 Checking Ollama status for code review...")
    
    try:
        installer = GitHookInstaller()
        
        if installer.check_ollama_status():
            print("✅ Ollama is ready for code review")
        else:
            print("❌ Ollama is not ready")
            print("Please ensure Ollama is running and Mistral model is available")
            
    except Exception as e:
        print(f"❌ Ollama status check failed: {e}")

if __name__ == '__main__':
    main() 