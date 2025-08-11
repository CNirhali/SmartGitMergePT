# 🔍 Mistral Code Review System Guide

A comprehensive guide to using the AI-powered code review system that automatically analyzes your code before committing using the Mistral LLM.

## 🚀 Overview

The Mistral Code Review System provides automated, intelligent code analysis that runs before each commit to ensure code quality, security, and performance. It uses the Mistral LLM via Ollama to provide human-like code review insights.

### Key Features

- **🤖 AI-Powered Analysis**: Uses Mistral LLM for intelligent code review
- **🔒 Security Scanning**: Detects vulnerabilities, hardcoded secrets, and security issues
- **⚡ Performance Analysis**: Identifies performance bottlenecks and optimization opportunities
- **📊 Quality Assessment**: Provides quality scores and improvement suggestions
- **🚫 Commit Blocking**: Automatically blocks commits with critical issues
- **📋 Comprehensive Reports**: Detailed reports with actionable insights
- **🔧 Pre-commit Hooks**: Automatic integration with Git workflow

## 🛠️ Installation & Setup

### Prerequisites

1. **Ollama**: Local LLM server for Mistral
2. **Python 3.8+**: Required for the code review system
3. **Git**: For pre-commit hook integration

### Step 1: Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve

# Pull Mistral model
ollama pull mistral
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install Pre-commit Hook

```bash
# Install the pre-commit hook
python src/main.py install-hook

# Test the installation
python src/main.py test-hook
```

## 🎮 Usage

### Basic Commands

#### Review Staged Changes
Review all files that are staged for commit:

```bash
python src/main.py review
```

#### Review Specific File
Review a single file:

```bash
python src/main.py review-file
# Enter file path when prompted
```

#### Install Pre-commit Hook
Set up automatic code review before each commit:

```bash
python src/main.py install-hook
```

#### Test Pre-commit Hook
Test the installed hook:

```bash
python src/main.py test-hook
```

#### Check Ollama Status
Verify Ollama is running and Mistral is available:

```bash
python src/main.py check-ollama
```

### Advanced Usage

#### Custom Configuration

```python
from src.code_reviewer import MistralCodeReviewer, ReviewConfig

# Create custom configuration
config = ReviewConfig(
    enable_security_scanning=True,
    enable_performance_analysis=True,
    enable_code_quality_check=True,
    block_on_high_severity=True,
    block_on_medium_severity=False,
    min_quality_score=0.8,
    max_issues_per_file=5,
    ignore_patterns=["*.tmp", "*.log", "node_modules/*"]
)

# Initialize reviewer with custom config
reviewer = MistralCodeReviewer(repo_path, config)
```

#### Programmatic Review

```python
from src.code_reviewer import MistralCodeReviewer

# Initialize reviewer
reviewer = MistralCodeReviewer(".")

# Review staged changes
results = reviewer.review_staged_changes()

# Generate report
report = reviewer.generate_report(results)
print(report)

# Check if commit should be blocked
should_block, reasons = reviewer.should_block_commit(results)
if should_block:
    print("Commit blocked:", reasons)
```

#### Using Decorators

```python
from src.code_reviewer import with_code_review

@with_code_review(".")
def deploy_to_production():
    """This function will only execute if code review passes"""
    print("Deploying to production...")

# The function will automatically run code review before execution
deploy_to_production()
```

## 📊 Understanding Review Results

### Review Report Structure

The code review system generates comprehensive reports with the following sections:

#### Summary Statistics
- **Total Issues**: Number of issues found across all files
- **Security Concerns**: Security-related issues detected
- **Performance Issues**: Performance-related problems
- **Average Quality Score**: Overall code quality (0.0-1.0)

#### Per-File Analysis
For each reviewed file:
- **Quality Score**: Code quality rating (0.0-1.0)
- **Overall Score**: Combined assessment score
- **Issues Count**: Number of issues found
- **Suggestions Count**: Number of improvement suggestions
- **Security Issues**: Security-related concerns
- **Performance Issues**: Performance-related problems

#### Issue Categories

1. **Security Issues**:
   - Hardcoded secrets and passwords
   - SQL injection vulnerabilities
   - XSS vulnerabilities
   - Insecure file operations
   - Missing input validation

2. **Performance Issues**:
   - Inefficient algorithms
   - Memory leaks
   - CPU-intensive operations
   - I/O bottlenecks
   - Unnecessary computations

3. **Code Quality Issues**:
   - Missing documentation
   - Poor naming conventions
   - Code duplication
   - Complex functions
   - Inconsistent formatting

4. **Bug Detection**:
   - Potential null pointer exceptions
   - Logic errors
   - Edge cases
   - Type mismatches
   - Resource leaks

### Quality Scoring

- **0.9-1.0**: Excellent code quality
- **0.8-0.9**: Good code quality
- **0.7-0.8**: Acceptable code quality
- **0.6-0.7**: Needs improvement
- **0.0-0.6**: Poor code quality

## ⚙️ Configuration Options

### ReviewConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_security_scanning` | bool | True | Enable security vulnerability detection |
| `enable_performance_analysis` | bool | True | Enable performance analysis |
| `enable_code_quality_check` | bool | True | Enable code quality assessment |
| `enable_dependency_scanning` | bool | True | Enable dependency analysis |
| `enable_secret_detection` | bool | True | Enable secret/hardcoded credential detection |
| `block_on_high_severity` | bool | True | Block commits with high severity issues |
| `block_on_medium_severity` | bool | False | Block commits with medium severity issues |
| `min_quality_score` | float | 0.7 | Minimum quality score required |
| `max_issues_per_file` | int | 10 | Maximum issues allowed per file |
| `ignore_patterns` | List[str] | See below | File patterns to ignore |

### Default Ignore Patterns

```python
ignore_patterns = [
    "*.pyc", "*.pyo", "__pycache__", "*.log", "*.tmp",
    "node_modules", ".git", ".env", "*.key", "*.pem"
]
```

### Custom Rules

You can add custom review rules:

```python
config = ReviewConfig(
    custom_rules=[
        {
            "name": "no_print_statements",
            "pattern": r"print\(",
            "severity": "medium",
            "message": "Use logging instead of print statements"
        },
        {
            "name": "require_docstrings",
            "pattern": r"def [^:]+:\s*(?!\s*\"\"\")",
            "severity": "low",
            "message": "Functions should have docstrings"
        }
    ]
)
```

## 🔧 Integration Examples

### CI/CD Pipeline Integration

```yaml
# GitHub Actions example
name: Code Review
on: [push, pull_request]

jobs:
  code-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          curl -fsSL https://ollama.ai/install.sh | sh
          ollama pull mistral
      - name: Run code review
        run: python src/main.py review
```

### IDE Integration

#### VS Code Extension

Create a VS Code extension that runs code review on save:

```json
{
  "name": "Mistral Code Review",
  "version": "1.0.0",
  "engines": {
    "vscode": "^1.60.0"
  },
  "activationEvents": [
    "onCommand:mistral.reviewFile"
  ],
  "main": "./extension.js",
  "contributes": {
    "commands": [
      {
        "command": "mistral.reviewFile",
        "title": "Review Current File with Mistral"
      }
    ]
  }
}
```

### Webhook Integration

```python
from flask import Flask, request
from src.code_reviewer import MistralCodeReviewer

app = Flask(__name__)

@app.route('/webhook/code-review', methods=['POST'])
def code_review_webhook():
    data = request.json
    
    # Extract repository information
    repo_path = data['repository']['path']
    
    # Run code review
    reviewer = MistralCodeReviewer(repo_path)
    results = reviewer.review_staged_changes()
    
    # Generate report
    report = reviewer.generate_report(results)
    
    # Send to Slack/Teams/etc.
    send_notification(report)
    
    return {'status': 'success'}
```

## 🚨 Troubleshooting

### Common Issues

#### Ollama Connection Issues

**Problem**: "Ollama is not responding"
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama if not running
ollama serve

# Check available models
ollama list

# Pull Mistral if not available
ollama pull mistral
```

**Problem**: "Mistral model not found"
```bash
# List available models
ollama list

# Pull Mistral model
ollama pull mistral

# Test connection
curl http://localhost:11434/v1/tags
```

#### Pre-commit Hook Issues

**Problem**: Hook not executing
```bash
# Check if hook exists
ls -la .git/hooks/pre-commit

# Make hook executable
chmod +x .git/hooks/pre-commit

# Test hook manually
.git/hooks/pre-commit
```

**Problem**: Hook failing silently
```bash
# Check hook logs
tail -f .git/hooks/pre-commit.log

# Test with verbose output
python src/main.py review --verbose
```

#### Performance Issues

**Problem**: Reviews taking too long
```bash
# Check Ollama performance
ollama ps

# Use smaller model for faster reviews
ollama pull mistral:7b

# Adjust configuration for faster reviews
config = ReviewConfig(
    max_issues_per_file=5,  # Reduce analysis depth
    min_quality_score=0.6   # Lower threshold
)
```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run review with debug output
reviewer = MistralCodeReviewer(".")
results = reviewer.review_staged_changes()
```

## 📈 Best Practices

### 1. Gradual Integration

Start with non-blocking reviews and gradually increase strictness:

```python
# Phase 1: Non-blocking reviews
config = ReviewConfig(block_on_high_severity=False)

# Phase 2: Block on high severity
config = ReviewConfig(block_on_high_severity=True)

# Phase 3: Block on medium severity
config = ReviewConfig(block_on_medium_severity=True)
```

### 2. Customize for Your Project

Adjust configuration based on your project needs:

```python
# For security-critical projects
config = ReviewConfig(
    enable_security_scanning=True,
    block_on_high_severity=True,
    min_quality_score=0.8
)

# For performance-critical projects
config = ReviewConfig(
    enable_performance_analysis=True,
    block_on_high_severity=True
)

# For rapid development
config = ReviewConfig(
    block_on_high_severity=True,
    block_on_medium_severity=False,
    min_quality_score=0.6
)
```

### 3. Regular Model Updates

Keep your Mistral model updated:

```bash
# Update Mistral model
ollama pull mistral

# Check for updates
ollama list
```

### 4. Monitor Review Performance

Track review effectiveness:

```python
# Collect review statistics
stats = {
    'total_reviews': 0,
    'blocked_commits': 0,
    'average_quality_score': 0.0,
    'common_issues': []
}

# Analyze trends over time
def analyze_review_trends():
    # Implementation for trend analysis
    pass
```

## 🔮 Future Enhancements

### Planned Features

1. **Multi-Model Support**: Support for other LLMs (GPT-4, Claude, etc.)
2. **Language-Specific Rules**: Specialized rules for different programming languages
3. **Team Learning**: Learn from team's coding patterns and preferences
4. **Integration APIs**: REST APIs for external tool integration
5. **Advanced Analytics**: Detailed analytics and trend analysis
6. **Custom Model Training**: Train models on your codebase

### Contributing

To contribute to the code review system:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📞 Support

For questions and support:

- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration options
- Test with the provided examples

---

**Note**: The Mistral Code Review System is designed to assist developers but should not replace human code review entirely. Always use it as part of a comprehensive code quality strategy.
