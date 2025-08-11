#!/usr/bin/env python3
"""
Demo script for Mistral Code Review System
Shows how the code review system works with various code examples
"""

import os
import tempfile
import subprocess
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from code_reviewer import MistralCodeReviewer, ReviewConfig
    from hook_installer import GitHookInstaller
    CODE_REVIEW_AVAILABLE = True
except ImportError as e:
    print(f"❌ Code review modules not available: {e}")
    CODE_REVIEW_AVAILABLE = False

def create_test_files():
    """Create test files with various code quality issues"""
    test_files = {}
    
    # Good code example
    good_code = '''
def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number efficiently.
    
    Args:
        n: The position in the Fibonacci sequence
        
    Returns:
        The nth Fibonacci number
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b

def main():
    """Main function to demonstrate Fibonacci calculation."""
    try:
        result = calculate_fibonacci(10)
        print(f"Fibonacci(10) = {result}")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
'''
    
    # Bad code example with issues
    bad_code = '''
def bad_function():
    password = "hardcoded_secret_123"  # Security issue
    x = 1 + 1  # Simple operation
    return x

# Missing docstring
def another_function():
    pass

def inefficient_fibonacci(n):
    if n <= 1:
        return n
    return inefficient_fibonacci(n-1) + inefficient_fibonacci(n-2)  # Performance issue

# SQL injection vulnerability
def get_user_data(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"  # Security issue
    return execute_query(query)

# Global variable abuse
global_var = 0

def modify_global():
    global global_var
    global_var += 1
    return global_var
'''
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(good_code)
        test_files['good_code.py'] = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(bad_code)
        test_files['bad_code.py'] = f.name
    
    return test_files

def setup_demo_repo():
    """Set up a demo Git repository"""
    temp_dir = tempfile.mkdtemp()
    
    # Initialize git repository
    subprocess.run(["git", "init"], cwd=temp_dir, check=True)
    
    # Create test files
    test_files = create_test_files()
    
    # Copy files to repo
    for filename, temp_path in test_files.items():
        dest_path = os.path.join(temp_dir, filename)
        with open(temp_path, 'r') as src, open(dest_path, 'w') as dst:
            dst.write(src.read())
        os.unlink(temp_path)  # Clean up temp file
    
    return temp_dir, test_files

def demo_code_review():
    """Demonstrate code review functionality"""
    print("🔍 MISTRAL CODE REVIEW SYSTEM DEMO")
    print("=" * 50)
    
    if not CODE_REVIEW_AVAILABLE:
        print("❌ Code review system not available")
        return
    
    # Set up demo repository
    print("📁 Setting up demo repository...")
    repo_path, test_files = setup_demo_repo()
    
    try:
        # Initialize code reviewer
        print("🤖 Initializing Mistral Code Reviewer...")
        reviewer = MistralCodeReviewer(repo_path)
        
        # Check Ollama status
        print("🔍 Checking Ollama status...")
        installer = GitHookInstaller(repo_path)
        ollama_ready = installer.check_ollama_status()
        
        if not ollama_ready:
            print("⚠️  Ollama not ready. Running demo with mock responses...")
            # For demo purposes, we'll show the structure without actual LLM calls
            demo_without_ollama(repo_path, test_files)
            return
        
        # Review good code
        print("\n📝 Reviewing good code example...")
        good_result = reviewer.review_file("good_code.py")
        print(f"Quality Score: {good_result.code_quality_score:.2f}")
        print(f"Issues Found: {len(good_result.issues)}")
        print(f"Suggestions: {len(good_result.suggestions)}")
        
        # Review bad code
        print("\n📝 Reviewing bad code example...")
        bad_result = reviewer.review_file("bad_code.py")
        print(f"Quality Score: {bad_result.code_quality_score:.2f}")
        print(f"Issues Found: {len(bad_result.issues)}")
        print(f"Security Concerns: {len(bad_result.security_concerns)}")
        print(f"Performance Issues: {len(bad_result.performance_issues)}")
        
        # Generate comprehensive report
        print("\n📊 Generating comprehensive report...")
        results = [good_result, bad_result]
        report = reviewer.generate_report(results)
        print(report)
        
        # Demo pre-commit hook
        print("\n🔧 Demonstrating pre-commit hook...")
        demo_pre_commit_hook(repo_path, test_files)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(repo_path)

def demo_without_ollama(repo_path, test_files):
    """Demo without Ollama using mock responses"""
    print("\n🎭 Running demo with mock responses...")
    
    # Show what the system would do
    print("\n📝 Good Code Analysis (Mock):")
    print("  ✅ Quality Score: 0.85")
    print("  ✅ Issues Found: 0")
    print("  ✅ Suggestions: 2")
    print("  ✅ Security Concerns: 0")
    print("  ✅ Performance Issues: 0")
    
    print("\n📝 Bad Code Analysis (Mock):")
    print("  ❌ Quality Score: 0.45")
    print("  ❌ Issues Found: 6")
    print("  ❌ Suggestions: 4")
    print("  ❌ Security Concerns: 2")
    print("  ❌ Performance Issues: 1")
    
    print("\n🚨 Issues Detected:")
    print("  • Hardcoded password (Security)")
    print("  • SQL injection vulnerability (Security)")
    print("  • Inefficient recursive Fibonacci (Performance)")
    print("  • Missing docstrings (Quality)")
    print("  • Global variable abuse (Quality)")
    print("  • Poor function naming (Quality)")
    
    print("\n💡 Suggestions:")
    print("  • Use environment variables for secrets")
    print("  • Use parameterized queries")
    print("  • Implement iterative Fibonacci")
    print("  • Add comprehensive docstrings")
    print("  • Avoid global variables")

def demo_pre_commit_hook(repo_path, test_files):
    """Demonstrate pre-commit hook functionality"""
    print("🔧 Setting up pre-commit hook...")
    
    # Stage the bad code file
    bad_file = os.path.join(repo_path, "bad_code.py")
    subprocess.run(["git", "add", bad_file], cwd=repo_path, check=True)
    
    # Try to commit (this should trigger the hook)
    print("📝 Attempting to commit bad code...")
    try:
        result = subprocess.run(
            ["git", "commit", "-m", "Add bad code example"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ Commit succeeded (hook may not be blocking)")
        else:
            print("❌ Commit blocked by code review hook")
            print("Output:", result.stdout)
            print("Errors:", result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⚠️  Commit timed out")
    except Exception as e:
        print(f"❌ Commit failed: {e}")

def demo_configuration():
    """Demonstrate configuration options"""
    print("\n⚙️  Configuration Examples:")
    
    # Security-focused config
    security_config = ReviewConfig(
        enable_security_scanning=True,
        enable_performance_analysis=False,
        block_on_high_severity=True,
        min_quality_score=0.8
    )
    print("🔒 Security-focused configuration:")
    print(f"  • Security scanning: {security_config.enable_security_scanning}")
    print(f"  • Performance analysis: {security_config.enable_performance_analysis}")
    print(f"  • Block on high severity: {security_config.block_on_high_severity}")
    print(f"  • Min quality score: {security_config.min_quality_score}")
    
    # Performance-focused config
    perf_config = ReviewConfig(
        enable_security_scanning=False,
        enable_performance_analysis=True,
        block_on_high_severity=True,
        min_quality_score=0.7
    )
    print("\n⚡ Performance-focused configuration:")
    print(f"  • Security scanning: {perf_config.enable_security_scanning}")
    print(f"  • Performance analysis: {perf_config.enable_performance_analysis}")
    print(f"  • Block on high severity: {perf_config.block_on_high_severity}")
    print(f"  • Min quality score: {perf_config.min_quality_score}")
    
    # Development config
    dev_config = ReviewConfig(
        block_on_high_severity=True,
        block_on_medium_severity=False,
        min_quality_score=0.6
    )
    print("\n🚀 Development configuration:")
    print(f"  • Block on high severity: {dev_config.block_on_high_severity}")
    print(f"  • Block on medium severity: {dev_config.block_on_medium_severity}")
    print(f"  • Min quality score: {dev_config.min_quality_score}")

def main():
    """Main demo function"""
    print("🚀 SmartGitMergePT - Mistral Code Review Demo")
    print("=" * 60)
    
    # Run main demo
    demo_code_review()
    
    # Show configuration examples
    demo_configuration()
    
    print("\n" + "=" * 60)
    print("✅ Demo completed!")
    print("\n📚 Next steps:")
    print("  1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
    print("  2. Start Ollama: ollama serve")
    print("  3. Pull Mistral: ollama pull mistral")
    print("  4. Install hook: python src/main.py install-hook")
    print("  5. Test review: python src/main.py review")
    print("\n📖 For more information, see CODE_REVIEW_GUIDE.md")

if __name__ == "__main__":
    main()
