import os
import json
import subprocess
import tempfile
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import difflib
from pathlib import Path

# Import guardrails and optimization if available
try:
    from guardrails import with_guardrails, secure_function
    from optimizer import cached_function
    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class CodeReviewResult:
    """Result of a code review"""
    file_path: str
    issues: List[Dict[str, Any]]
    suggestions: List[Dict[str, Any]]
    security_concerns: List[Dict[str, Any]]
    performance_issues: List[Dict[str, Any]]
    code_quality_score: float  # 0.0 to 1.0
    overall_score: float  # 0.0 to 1.0
    should_block_commit: bool
    review_summary: str
    timestamp: datetime

@dataclass
class ReviewConfig:
    """Configuration for code review settings"""
    enable_security_scanning: bool = True
    enable_performance_analysis: bool = True
    enable_code_quality_check: bool = True
    enable_dependency_scanning: bool = True
    enable_secret_detection: bool = True
    block_on_high_severity: bool = True
    block_on_medium_severity: bool = False
    min_quality_score: float = 0.7
    max_issues_per_file: int = 10
    ignore_patterns: List[str] = None
    custom_rules: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.ignore_patterns is None:
            self.ignore_patterns = [
                "*.pyc", "*.pyo", "__pycache__", "*.log", "*.tmp",
                "node_modules", ".git", ".env", "*.key", "*.pem"
            ]
        if self.custom_rules is None:
            self.custom_rules = []

class MistralCodeReviewer:
    """
    Mistral-powered code reviewer that analyzes code before committing
    """
    
    def __init__(self, repo_path: str, config: Optional[ReviewConfig] = None):
        self.repo_path = repo_path
        self.config = config or ReviewConfig()
        self.ollama_endpoint = "http://localhost:11434/v1/chat/completions"
        self.model = "mistral"
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        
    def review_staged_changes(self) -> List[CodeReviewResult]:
        """Review all staged changes before commit"""
        try:
            # Get staged files
            staged_files = self._get_staged_files()
            if not staged_files:
                logger.info("No staged files to review")
                return []
            
            results = []
            for file_path in staged_files:
                if self._should_ignore_file(file_path):
                    logger.info(f"Ignoring file: {file_path}")
                    continue
                    
                result = self._review_file(file_path)
                results.append(result)
                
            return results
            
        except Exception as e:
            logger.error(f"Error during code review: {e}")
            return []
    
    def review_file(self, file_path: str) -> CodeReviewResult:
        """Review a specific file"""
        return self._review_file(file_path)
    
    def review_diff(self, diff_content: str, file_path: str) -> CodeReviewResult:
        """Review a diff content"""
        return self._review_diff_content(diff_content, file_path)
    
    def _get_staged_files(self) -> List[str]:
        """Get list of staged files"""
        try:
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return [f.strip() for f in result.stdout.splitlines() if f.strip()]
        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting staged files: {e}")
            return []
    
    def _should_ignore_file(self, file_path: str) -> bool:
        """Check if file should be ignored based on patterns"""
        for pattern in self.config.ignore_patterns:
            if pattern.startswith("*."):
                if file_path.endswith(pattern[1:]):
                    return True
            elif pattern in file_path:
                return True
        return False
    
    def _review_file(self, file_path: str) -> CodeReviewResult:
        """Review a single file"""
        try:
            # Get file content and validate path for security
            abs_repo_path = os.path.abspath(self.repo_path)
            full_path = os.path.abspath(os.path.join(abs_repo_path, file_path))

            # Prevent path traversal
            if os.path.commonpath([abs_repo_path, full_path]) != abs_repo_path:
                logger.warning(f"Path traversal attempt detected: {file_path}")
                return self._create_empty_result(file_path, "Error: Path traversal attempt detected")

            if not os.path.exists(full_path):
                return self._create_empty_result(file_path, "File not found")
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Get diff for staged changes
            diff_content = self._get_file_diff(file_path)
            
            # Analyze with Mistral
            analysis = self._analyze_with_mistral(content, diff_content, file_path)
            
            return self._create_review_result(file_path, analysis)
            
        except Exception as e:
            logger.error(f"Error reviewing file {file_path}: {e}")
            return self._create_empty_result(file_path, f"Error: {str(e)}")
    
    def _review_diff_content(self, diff_content: str, file_path: str) -> CodeReviewResult:
        """Review diff content directly"""
        try:
            analysis = self._analyze_with_mistral("", diff_content, file_path)
            return self._create_review_result(file_path, analysis)
        except Exception as e:
            logger.error(f"Error reviewing diff for {file_path}: {e}")
            return self._create_empty_result(file_path, f"Error: {str(e)}")
    
    def _get_file_diff(self, file_path: str) -> str:
        """Get diff for staged changes in a file"""
        try:
            result = subprocess.run(
                ["git", "diff", "--cached", file_path],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError:
            return ""
    
    def _analyze_with_mistral(self, content: str, diff_content: str, file_path: str) -> Dict[str, Any]:
        """Analyze code using Mistral LLM"""
        try:
            # Prepare prompt for code review
            prompt = self._create_review_prompt(content, diff_content, file_path)
            
            # Call Mistral via Ollama
            response = self._call_mistral(prompt)
            
            # Parse response
            return self._parse_mistral_response(response, file_path)
            
        except Exception as e:
            logger.error(f"Error analyzing with Mistral: {e}")
            return self._create_fallback_analysis(file_path)
    
    def _create_review_prompt(self, content: str, diff_content: str, file_path: str) -> str:
        """Create a comprehensive review prompt for Mistral"""
        file_extension = Path(file_path).suffix.lower()
        
        prompt = f"""You are an expert code reviewer. Analyze the following code and provide a comprehensive review.

File: {file_path}
File Type: {file_extension}

Code Content:
{content[:2000]}{'...' if len(content) > 2000 else ''}

Staged Changes (Diff):
{diff_content[:2000]}{'...' if len(diff_content) > 2000 else ''}

Please provide a detailed analysis in the following JSON format:
{{
    "issues": [
        {{
            "type": "bug|security|performance|style|maintainability",
            "severity": "high|medium|low",
            "line": <line_number>,
            "description": "<detailed description>",
            "suggestion": "<how to fix>"
        }}
    ],
    "suggestions": [
        {{
            "type": "improvement|optimization|refactoring",
            "description": "<suggestion description>",
            "priority": "high|medium|low"
        }}
    ],
    "security_concerns": [
        {{
            "type": "vulnerability|exposure|injection",
            "severity": "critical|high|medium|low",
            "description": "<security issue description>",
            "mitigation": "<how to fix>"
        }}
    ],
    "performance_issues": [
        {{
            "type": "complexity|memory|cpu|io",
            "severity": "high|medium|low",
            "description": "<performance issue>",
            "optimization": "<optimization suggestion>"
        }}
    ],
    "code_quality_score": <0.0-1.0>,
    "overall_score": <0.0-1.0>,
    "should_block_commit": <true|false>,
    "summary": "<brief summary of findings>"
}}

Focus on:
1. Security vulnerabilities (SQL injection, XSS, hardcoded secrets, etc.)
2. Performance issues (complexity, memory leaks, inefficient algorithms)
3. Code quality (readability, maintainability, best practices)
4. Potential bugs and edge cases
5. Style and consistency issues

Be thorough but practical. Only block commits for critical issues."""
        
        return prompt
    
    def _call_mistral(self, prompt: str) -> str:
        """Call Mistral via Ollama API"""
        try:
            import requests
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 4000
                }
            }
            
            response = requests.post(
                self.ollama_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                logger.error(f"Mistral API error: {response.status_code} - {response.text}")
                return ""
                
        except ImportError:
            logger.error("requests library not available")
            return ""
        except Exception as e:
            logger.error(f"Error calling Mistral: {e}")
            return ""
    
    def _parse_mistral_response(self, response: str, file_path: str) -> Dict[str, Any]:
        """Parse Mistral's JSON response"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                return self._create_fallback_analysis(file_path)
            
            json_str = response[json_start:json_end]
            analysis = json.loads(json_str)
            
            # Validate and sanitize the analysis
            return self._validate_analysis(analysis)
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Mistral response: {e}")
            return self._create_fallback_analysis(file_path)
        except Exception as e:
            logger.error(f"Error processing Mistral response: {e}")
            return self._create_fallback_analysis(file_path)
    
    def _validate_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize analysis results"""
        validated = {
            "issues": analysis.get("issues", []),
            "suggestions": analysis.get("suggestions", []),
            "security_concerns": analysis.get("security_concerns", []),
            "performance_issues": analysis.get("performance_issues", []),
            "code_quality_score": max(0.0, min(1.0, float(analysis.get("code_quality_score", 0.5)))),
            "overall_score": max(0.0, min(1.0, float(analysis.get("overall_score", 0.5)))),
            "should_block_commit": bool(analysis.get("should_block_commit", False)),
            "summary": str(analysis.get("summary", "Analysis completed"))
        }
        
        # Validate lists
        for key in ["issues", "suggestions", "security_concerns", "performance_issues"]:
            if not isinstance(validated[key], list):
                validated[key] = []
        
        return validated
    
    def _create_fallback_analysis(self, file_path: str) -> Dict[str, Any]:
        """Create fallback analysis when Mistral fails"""
        return {
            "issues": [],
            "suggestions": [],
            "security_concerns": [],
            "performance_issues": [],
            "code_quality_score": 0.5,
            "overall_score": 0.5,
            "should_block_commit": False,
            "summary": f"Analysis failed for {file_path}, using fallback"
        }
    
    def _create_review_result(self, file_path: str, analysis: Dict[str, Any]) -> CodeReviewResult:
        """Create CodeReviewResult from analysis"""
        return CodeReviewResult(
            file_path=file_path,
            issues=analysis.get("issues", []),
            suggestions=analysis.get("suggestions", []),
            security_concerns=analysis.get("security_concerns", []),
            performance_issues=analysis.get("performance_issues", []),
            code_quality_score=analysis.get("code_quality_score", 0.5),
            overall_score=analysis.get("overall_score", 0.5),
            should_block_commit=analysis.get("should_block_commit", False),
            review_summary=analysis.get("summary", ""),
            timestamp=datetime.now()
        )
    
    def _create_empty_result(self, file_path: str, reason: str) -> CodeReviewResult:
        """Create empty review result"""
        return CodeReviewResult(
            file_path=file_path,
            issues=[],
            suggestions=[],
            security_concerns=[],
            performance_issues=[],
            code_quality_score=0.0,
            overall_score=0.0,
            should_block_commit=False,
            review_summary=f"Review skipped: {reason}",
            timestamp=datetime.now()
        )
    
    def should_block_commit(self, results: List[CodeReviewResult]) -> Tuple[bool, List[str]]:
        """Determine if commit should be blocked based on review results"""
        blocking_reasons = []
        
        for result in results:
            # Check for high severity security issues
            if self.config.block_on_high_severity:
                high_security = [issue for issue in result.security_concerns 
                               if issue.get("severity") in ["critical", "high"]]
                if high_security:
                    blocking_reasons.append(f"High severity security issues in {result.file_path}")
            
            # Check for high severity issues
            if self.config.block_on_high_severity:
                high_issues = [issue for issue in result.issues 
                             if issue.get("severity") == "high"]
                if high_issues:
                    blocking_reasons.append(f"High severity issues in {result.file_path}")
            
            # Check quality score
            if result.code_quality_score < self.config.min_quality_score:
                blocking_reasons.append(f"Low code quality score ({result.code_quality_score:.2f}) in {result.file_path}")
            
            # Check issue count
            if len(result.issues) > self.config.max_issues_per_file:
                blocking_reasons.append(f"Too many issues ({len(result.issues)}) in {result.file_path}")
        
        should_block = len(blocking_reasons) > 0
        return should_block, blocking_reasons
    
    def generate_report(self, results: List[CodeReviewResult]) -> str:
        """Generate a human-readable report"""
        if not results:
            return "No files reviewed."
        
        report = []
        report.append("=" * 80)
        report.append("🔍 MISTRAL CODE REVIEW REPORT")
        report.append("=" * 80)
        report.append(f"Review Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Files Reviewed: {len(results)}")
        report.append("")
        
        # Summary statistics
        total_issues = sum(len(r.issues) for r in results)
        total_security = sum(len(r.security_concerns) for r in results)
        total_performance = sum(len(r.performance_issues) for r in results)
        avg_quality = sum(r.code_quality_score for r in results) / len(results)
        
        report.append("📊 SUMMARY STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Issues: {total_issues}")
        report.append(f"Security Concerns: {total_security}")
        report.append(f"Performance Issues: {total_performance}")
        report.append(f"Average Quality Score: {avg_quality:.2f}")
        report.append("")
        
        # Per-file results
        for result in results:
            report.append(f"📁 {result.file_path}")
            report.append("-" * 40)
            report.append(f"Quality Score: {result.code_quality_score:.2f}")
            report.append(f"Overall Score: {result.overall_score:.2f}")
            report.append(f"Issues: {len(result.issues)}")
            report.append(f"Suggestions: {len(result.suggestions)}")
            report.append(f"Security: {len(result.security_concerns)}")
            report.append(f"Performance: {len(result.performance_issues)}")
            
            if result.issues:
                report.append("")
                report.append("🚨 ISSUES:")
                for issue in result.issues[:5]:  # Show first 5 issues
                    severity = issue.get("severity", "unknown").upper()
                    line = issue.get("line", "N/A")
                    desc = issue.get("description", "No description")
                    report.append(f"  [{severity}] Line {line}: {desc}")
            
            if result.security_concerns:
                report.append("")
                report.append("🔒 SECURITY CONCERNS:")
                for concern in result.security_concerns[:3]:  # Show first 3
                    severity = concern.get("severity", "unknown").upper()
                    desc = concern.get("description", "No description")
                    report.append(f"  [{severity}] {desc}")
            
            if result.suggestions:
                report.append("")
                report.append("💡 SUGGESTIONS:")
                for suggestion in result.suggestions[:3]:  # Show first 3
                    priority = suggestion.get("priority", "medium").upper()
                    desc = suggestion.get("description", "No description")
                    report.append(f"  [{priority}] {desc}")
            
            report.append("")
            report.append(f"Summary: {result.review_summary}")
            report.append("")
        
        # Overall recommendation
        should_block, reasons = self.should_block_commit(results)
        report.append("🎯 OVERALL RECOMMENDATION")
        report.append("-" * 40)
        
        if should_block:
            report.append("❌ COMMIT BLOCKED")
            report.append("Reasons:")
            for reason in reasons:
                report.append(f"  • {reason}")
        else:
            report.append("✅ COMMIT APPROVED")
            if total_issues > 0:
                report.append("Note: Some issues found but not blocking")
        
        report.append("=" * 80)
        
        return "\n".join(report)

# Decorator for automatic code review
def with_code_review(repo_path: str = ".", config: Optional[ReviewConfig] = None):
    """Decorator to automatically review code before function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Review staged changes before executing function
            reviewer = MistralCodeReviewer(repo_path, config)
            results = reviewer.review_staged_changes()
            
            if results:
                print(reviewer.generate_report(results))
                
                should_block, reasons = reviewer.should_block_commit(results)
                if should_block:
                    print("\n❌ Execution blocked due to code review issues!")
                    for reason in reasons:
                        print(f"  • {reason}")
                    return None
            
            # Execute function if review passes
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Pre-commit hook function
def run_pre_commit_review(repo_path: str = ".", config: Optional[ReviewConfig] = None) -> bool:
    """Run code review as a pre-commit hook"""
    try:
        reviewer = MistralCodeReviewer(repo_path, config)
        results = reviewer.review_staged_changes()
        
        if not results:
            print("✅ No files to review")
            return True
        
        # Generate and display report
        report = reviewer.generate_report(results)
        print(report)
        
        # Check if commit should be blocked
        should_block, reasons = reviewer.should_block_commit(results)
        
        if should_block:
            print("\n❌ COMMIT BLOCKED by code review!")
            print("Please fix the issues above before committing.")
            return False
        else:
            print("\n✅ Code review passed - commit can proceed")
            return True
            
    except Exception as e:
        print(f"❌ Error during code review: {e}")
        return False

if __name__ == "__main__":
    # Test the code reviewer
    import sys
    
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    else:
        repo_path = "."
    
    success = run_pre_commit_review(repo_path)
    sys.exit(0 if success else 1)
