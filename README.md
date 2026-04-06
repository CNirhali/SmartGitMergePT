# SmartGitMergePT with Agentic AI Tracking

An intelligent Git merge conflict resolver and predictor with **Agentic AI Developer Tracking** for automated project management and time estimation.

## 🚀 Features

### Core Git Features
- **Conflict Prediction**: Predict potential merge conflicts between branches
- **Conflict Detection**: Detect actual merge conflicts in real-time
- **LLM Resolution**: Resolve conflicts using AI-powered analysis
- **Team Dashboard**: Visualize team conflict patterns

### 🎯 Agentic AI Tracking System
- **Automated Data Acquisition**: Screen recording, webcam monitoring, activity tracking
- **AI-Powered Validation**: Cross-check and verify work details using AI
- **Developer Tracking**: Monitor Alice working on feature branch A (and other developers)
- **Project Estimates**: Automated time estimation and project completion predictions
- **Privacy Controls**: Configurable privacy settings and data retention
- **Real-time Monitoring**: Continuous tracking with validation

### 🛡️ Guardrails & Security System
- **Input Validation**: Comprehensive validation and sanitization of all inputs
- **Rate Limiting**: Sliding window rate limiting for API and authentication
- **Data Encryption**: Automatic encryption of sensitive data
- **Security Monitoring**: Real-time security event tracking and threat detection
- **Error Handling**: Comprehensive error handling with recovery strategies
- **Threat Detection**: Automatic detection and blocking of suspicious activities

### ⚡ Performance Optimization System
- **Smart Caching**: Multiple caching strategies with compression and persistence
- **Async Operations**: Thread and process pool management for concurrent operations
- **Resource Management**: Memory and CPU monitoring with automatic optimization
- **Performance Tracking**: Detailed performance metrics and optimization statistics
- **Batch Processing**: Efficient batch processing for large datasets
- **Parallel Computing**: Multi-threaded and multi-process execution

### 📊 Monitoring & Health System
- **Health Checks**: System, database, and network health monitoring
- **Metrics Collection**: Real-time system metrics collection
- **Alerting**: Configurable alerting with multiple levels
- **Live Monitoring**: Real-time monitoring with rich interface
- **Performance Analytics**: Detailed performance analysis and reporting

### 🔍 Mistral Code Review System
- **AI-Powered Review**: Automatic code review using Mistral LLM
- **Pre-commit Hooks**: Automatic review before each commit
- **Security Scanning**: Detection of vulnerabilities and security issues
- **Performance Analysis**: Identification of performance bottlenecks
- **Code Quality Assessment**: Quality scoring and improvement suggestions
- **Commit Blocking**: Automatic blocking of commits with critical issues
- **Comprehensive Reports**: Detailed review reports with actionable insights

## 🏗️ Architecture

```
SmartGitMergePT/
├── src/
│   ├── agentic_tracker.py      # Main tracking system
│   ├── ai_validator.py         # AI validation engine
│   ├── config_manager.py       # Configuration management
│   ├── agentic_demo.py         # Comprehensive demo
│   ├── main.py                 # CLI interface
│   ├── git_utils.py            # Git operations
│   ├── predictor.py            # Conflict prediction
│   ├── llm_resolver.py         # Conflict resolution
│   ├── dashboard.py            # Team dashboard
│   ├── guardrails.py           # Security and guardrails system
│   ├── optimizer.py            # Performance optimization system
│   ├── monitor.py              # Health monitoring system
│   ├── code_reviewer.py        # Mistral-powered code review system
│   └── hook_installer.py       # Git hook installer for code review
├── demo/                       # Demo scenarios
├── requirements.txt            # Dependencies
└── README.md                  # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Git
- Webcam (for face recognition)
- OpenAI API key (for AI validation) OR Ollama (for local LLM)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd SmartGitMergePT
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up LLM (choose one)**:
   
   **Option A: Local Ollama (Recommended)**
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama server
   ollama serve
   
   # Pull Mistral model
   ollama pull mistral
   ```
   
   **Option B: OpenAI (Cloud-based)**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

4. **Configure tracking** (optional):
```bash
python src/main.py config
```

5. **Initialize project** (optional):
```bash
python install.py
```

## 🎮 Usage

### Basic Git Operations

**Predict conflicts**:
```bash
python src/main.py predict
```

**Detect conflicts**:
```bash
python src/main.py detect
```

**Resolve conflicts**:
```bash
python src/main.py resolve
```

**Show dashboard**:
```bash
python src/main.py dashboard
```

### Agentic AI Tracking

**Start tracking a developer**:
```bash
python src/main.py track
```

**Stop tracking**:
```bash
python src/main.py stop-track
```

**Validate session data**:
```bash
python src/main.py validate
```

**Generate project estimates**:
```bash
python src/main.py estimates
```

**Show developer statistics**:
```bash
python src/main.py stats
```

**Manage configuration**:
```bash
python src/main.py config
```

**Run comprehensive demo**:
```bash
python src/main.py demo
```

### 🛡️ Guardrails & Security

**Start system monitoring**:
```bash
python src/main.py monitor
```

**Run health checks**:
```bash
python src/main.py health
```

**Show guardrails status**:
```bash
python src/main.py guardrails
```

### ⚡ Performance & Optimization

**Run system optimization**:
```bash
python src/main.py optimize
```

**Show performance metrics**:
```bash
python src/main.py performance
```

### 🔍 Code Review & Quality

**Review staged changes**:
```bash
python src/main.py review
```

**Review specific file**:
```bash
python src/main.py review-file
```

**Install pre-commit hook**:
```bash
python src/main.py install-hook
```

**Test pre-commit hook**:
```bash
python src/main.py test-hook
```

**Check Ollama status**:
```bash
python src/main.py check-ollama
```

## 🤖 Agentic AI Tracking System

### How It Works

The agentic AI tracking system autonomously monitors developer work and provides intelligent insights:

1. **Data Acquisition**:
   - Screen recording and analysis
   - Webcam monitoring with face recognition
   - Keyboard and mouse activity tracking
   - Git activity correlation

2. **AI Validation**:
   - Cross-reference with git commits
   - Temporal consistency checks
   - Activity pattern validation
   - Anomaly detection
   - AI-powered content analysis

3. **Project Intelligence**:
   - Automated time estimation
   - Project completion predictions
   - Developer productivity analysis
   - Data quality assessment

### Example: Alice Working on Feature Branch A

```python
# Alice starts working on feature/user-authentication
tracker = AgenticTracker(repo_path)
tracker.start_tracking("alice_dev_001", "feature/user-authentication")

# System automatically:
# - Records screen activity
# - Monitors webcam for face detection
# - Tracks keyboard/mouse activity
# - Correlates with git commits
# - Validates work patterns with AI

# Generate estimates
estimates = tracker.get_project_estimates("feature/user-authentication")
print(f"Estimated completion: {estimates['estimated_completion_hours']:.2f} hours")
```

### Privacy and Security

The system includes comprehensive privacy controls:

- **Data Encryption**: Optional encryption of sensitive data
- **Face Blurring**: Automatic face blurring in stored images
- **Data Retention**: Configurable retention periods
- **Anonymization**: Optional data anonymization
- **Local Storage**: All data stored locally by default

## 📊 Configuration Options

### LLM Configuration
- **Provider**: `ollama` (local), `openai` (cloud), or `mistral` (legacy)
- **Ollama Endpoint**: `http://localhost:11434/v1/chat/completions`
- **Ollama Model**: `mistral` (default), `llama2`, `codellama`, etc.
- **Validation Confidence Thresholds**: Configurable confidence levels

### Tracking Configuration
- Screenshot interval (15-300 seconds)
- Webcam monitoring frequency
- Activity confidence thresholds
- Validation intervals

### Privacy Configuration
- Data retention periods
- Encryption settings
- Face blurring options
- Anonymization controls

### AI Validation Configuration
- Validation confidence thresholds
- Cross-reference settings
- Anomaly detection sensitivity
- Temporal consistency checks

### Preset Configurations

**Privacy-Compliant**:
```bash
python src/main.py config
# Choose option 2: Create privacy-compliant config
```

**Performance-Optimized**:
```bash
python src/main.py config
# Choose option 3: Create performance-optimized config
```

**High-Accuracy**:
```bash
python src/main.py config
# Choose option 4: Create high-accuracy config
```

**Configure LLM**:
```bash
python src/main.py config
# Choose option 5: Configure LLM settings
# Select Ollama (option 1) for local LLM
```

## 🔧 Advanced Usage

### Custom Configuration

Create custom tracking configurations:

```python
from src.config_manager import ConfigManager

config_manager = ConfigManager(repo_path)

# Update LLM settings for Ollama
config_manager.update_llm_config(
    provider='ollama',
    ollama_endpoint='http://localhost:11434/v1/chat/completions',
    ollama_model='mistral'
)

# Update tracking settings
config_manager.update_tracking_config(
    screenshot_interval=60,
    webcam_interval=30,
    activity_threshold=0.8
)

# Update privacy settings
config_manager.update_privacy_config(
    store_screenshots=False,
    encrypt_data=True,
    data_retention_days=7
)
```

### Guardrails & Security Integration

```python
from src.guardrails import GuardrailsManager, with_guardrails, secure_function
from src.optimizer import PerformanceOptimizer, CacheConfig, PerformanceConfig

# Initialize guardrails and optimization
guardrails = GuardrailsManager()
cache_config = CacheConfig()
perf_config = PerformanceConfig()
optimizer = PerformanceOptimizer(cache_config, perf_config)

# Secure function with full protection
@secure_function(guardrails)
@optimizer.cached_function(ttl_seconds=3600)
def process_sensitive_data(data):
    # Input validation, caching, and security monitoring
    return processed_data

# Use guardrails for input validation
is_valid, validated_data = guardrails.validate_input(user_input)
if is_valid:
    # Process validated data
    pass
```

### Performance Optimization

```python
from src.optimizer import SmartCache, AsyncTaskManager, ResourceManager

# Smart caching
cache = SmartCache(CacheConfig(strategy='ttl', max_size=1000))
cache.set("key", "value")
value = cache.get("key")

# Async operations
async_manager = AsyncTaskManager(max_workers=4)
result = await async_manager.run_in_thread(expensive_function, args)

# Resource management
resource_manager = ResourceManager(memory_limit_mb=512)
memory_healthy, usage = resource_manager.check_memory_usage()
if not memory_healthy:
    resource_manager.optimize_memory()
```

### Health Monitoring

```python
from src.monitor import SystemMonitor, create_monitor

# Create and start monitoring
monitor = create_monitor(repo_path)
monitor.start_monitoring()

# Run health checks
results = await monitor.run_health_checks()
health = monitor.get_overall_health()

# Display live status
monitor.display_status()
```

**Code Review Integration**

```python
from src.code_reviewer import MistralCodeReviewer, ReviewConfig, with_code_review

# Initialize code reviewer
config = ReviewConfig(
    enable_security_scanning=True,
    min_quality_score=0.8,
    block_on_high_severity=True
)
reviewer = MistralCodeReviewer(repo_path, config)

# Review staged changes
results = reviewer.review_staged_changes()
report = reviewer.generate_report(results)
print(report)

# Use decorator for automatic review
@with_code_review(repo_path, config)
def deploy_to_production():
    # This function will only execute if code review passes
    pass
```

### Direct API Usage

```python
from src.agentic_tracker import AgenticTracker
from src.ai_validator import AIValidator

# Initialize tracking
tracker = AgenticTracker(repo_path)

# Register developer
tracker.register_developer("alice_dev_001", "Alice Johnson", "alice_face.jpg")

# Start tracking
tracker.start_tracking("alice_dev_001", "feature/user-authentication")

# Get statistics
stats = tracker.get_developer_stats("alice_dev_001", days=7)
print(f"Total hours: {stats['total_hours']:.2f}")

# Get project estimates
estimates = tracker.get_project_estimates("feature/user-authentication")
print(f"Estimated completion: {estimates['estimated_completion_hours']:.2f} hours")
```

### AI Validation

```python
from src.ai_validator import AIValidator

validator = AIValidator(openai_api_key, repo_path)

# Validate session data
validation_result = validator.validate_work_session(session_data)

if validation_result.is_valid:
    print("✅ Session validated successfully")
else:
    print("❌ Validation issues found:")
    for issue in validation_result.issues:
        print(f"   • {issue}")
```

## 📈 Demo Scenarios

### Alice's Work Session Demo

Run the comprehensive demo showing Alice working on feature branch A:

```bash
python src/main.py demo
```

This demo includes:
- Alice registration with face recognition
- Simulated work session with multiple activities
- AI validation of session data
- Project estimate generation
- Developer statistics analysis

### Custom Demo

```python
from src.agentic_demo import AgenticDemo

demo = AgenticDemo(repo_path, openai_api_key)
results = demo.run_comprehensive_demo()

print(f"Session duration: {results['session_data']['duration_minutes']:.1f} minutes")
print(f"Validation passed: {results['validation_result'].is_valid}")
```

## 🔍 Troubleshooting

### Common Issues

**Ollama not detected**:
- Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
- Start server: `ollama serve`
- Pull model: `ollama pull mistral`
- Test connection: `curl http://localhost:11434/v1/chat/completions`

**Webcam not detected**:
- Check webcam permissions
- Ensure webcam is not in use by other applications
- Try different webcam index in configuration

**AI validation fails**:
- For Ollama: Check if `ollama serve` is running
- For OpenAI: Verify API key is set
- Check internet connectivity
- Review API usage limits

**Tracking stops unexpectedly**:
- Check system resources
- Verify database permissions
- Review log files in tracking_data/

### Guardrails & Security Issues

**Input validation failing**:
- Check validation patterns in guardrails configuration
- Review input data for sensitive information
- Adjust validation thresholds as needed

**Rate limiting too restrictive**:
- Increase rate limit thresholds in configuration
- Review usage patterns and adjust accordingly
- Check for legitimate high-volume usage

**Security alerts too frequent**:
- Review security event thresholds
- Adjust alert sensitivity in configuration
- Check for false positives in threat detection

### Performance & Optimization Issues

**Cache not working effectively**:
- Check cache hit rates and statistics
- Adjust cache size and TTL settings
- Review cache strategy (LRU vs TTL)

**Memory usage too high**:
- Enable automatic memory optimization
- Review resource limits in configuration
- Check for memory leaks in long-running processes

**Async operations failing**:
- Check thread and process pool sizes
- Review async function implementations
- Monitor system resource usage during async operations

### Monitoring & Health Issues

**Health checks failing**:
- Check individual health checker configurations
- Review system resource availability
- Verify network connectivity for network checks

**Metrics collection issues**:
- Check metrics collection intervals
- Review system permissions for metrics access
- Verify storage space for metrics data

**Alerts not triggering**:
- Check alert configuration and thresholds
- Review alert handler implementations
- Verify alert delivery mechanisms

### Code Review Issues

**Mistral not responding**:
- Check if Ollama is running: `ollama serve`
- Verify Mistral model is available: `ollama list`
- Pull Mistral model if needed: `ollama pull mistral`
- Test connection: `python src/main.py check-ollama`

**Pre-commit hook not working**:
- Verify hook is installed: `ls -la .git/hooks/pre-commit`
- Check hook permissions: `chmod +x .git/hooks/pre-commit`
- Test hook manually: `python src/main.py test-hook`
- Review hook logs for errors

**Code review too strict/lenient**:
- Adjust quality thresholds in ReviewConfig
- Modify blocking criteria (high/medium severity)
- Update ignore patterns for specific file types
- Customize review rules and patterns

**Review taking too long**:
- Check Ollama performance and model size
- Reduce content length in prompts
- Optimize file filtering and ignore patterns
- Consider using smaller/faster models

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Data Recovery

Session data is stored in SQLite database:
```bash
sqlite3 .smartgit_tracker.db
.tables
SELECT * FROM work_sessions;
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for AI validation capabilities
- OpenCV for computer vision features
- MediaPipe for face detection
- GitPython for Git integration
- Cryptography library for encryption features
- Rich library for beautiful terminal interfaces
- Cachetools for intelligent caching
- Psutil for system monitoring
- Ollama for local Mistral LLM access
- Requests for API communication

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the demo scenarios

---

**Note**: This system includes advanced tracking capabilities. Ensure compliance with local privacy laws and obtain necessary consent before deploying in production environments.