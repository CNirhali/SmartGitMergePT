import os
import tempfile
import pytest
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.code_reviewer import (
    MistralCodeReviewer, 
    ReviewConfig, 
    CodeReviewResult,
    run_pre_commit_review,
    with_code_review
)

class TestReviewConfig:
    """Test ReviewConfig class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ReviewConfig()
        
        assert config.enable_security_scanning == True
        assert config.enable_performance_analysis == True
        assert config.enable_code_quality_check == True
        assert config.block_on_high_severity == True
        assert config.block_on_medium_severity == False
        assert config.min_quality_score == 0.7
        assert config.max_issues_per_file == 10
        assert "*.pyc" in config.ignore_patterns
        assert "*.log" in config.ignore_patterns
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = ReviewConfig(
            enable_security_scanning=False,
            min_quality_score=0.5,
            max_issues_per_file=5
        )
        
        assert config.enable_security_scanning == False
        assert config.min_quality_score == 0.5
        assert config.max_issues_per_file == 5

class TestMistralCodeReviewer:
    """Test MistralCodeReviewer class"""
    
    @pytest.fixture
    def temp_repo(self):
        """Create a temporary repository for testing"""
        temp_dir = tempfile.mkdtemp()
        
        # Initialize git repository
        os.system(f"cd {temp_dir} && git init")
        
        # Create some test files
        test_file = os.path.join(temp_dir, "test.py")
        with open(test_file, 'w') as f:
            f.write('''
def bad_function():
    password = "hardcoded_secret_123"  # Security issue
    x = 1 + 1  # Simple operation
    return x

# Missing docstring
def another_function():
    pass
''')
        
        yield temp_dir
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def reviewer(self, temp_repo):
        """Create a code reviewer instance"""
        return MistralCodeReviewer(temp_repo)
    
    def test_init(self, temp_repo):
        """Test reviewer initialization"""
        reviewer = MistralCodeReviewer(temp_repo)
        
        assert reviewer.repo_path == temp_repo
        assert reviewer.ollama_endpoint == "http://localhost:11434/v1/chat/completions"
        assert reviewer.model == "mistral"
        assert isinstance(reviewer.config, ReviewConfig)
    
    def test_should_ignore_file(self, reviewer):
        """Test file ignore patterns"""
        # Should ignore
        assert reviewer._should_ignore_file("file.pyc") == True
        assert reviewer._should_ignore_file("__pycache__/module.py") == True
        assert reviewer._should_ignore_file("logs/app.log") == True
        
        # Should not ignore
        assert reviewer._should_ignore_file("src/main.py") == False
        assert reviewer._should_ignore_file("test.py") == False
    
    @patch('subprocess.run')
    def test_get_staged_files(self, mock_run, reviewer):
        """Test getting staged files"""
        mock_run.return_value.stdout = "file1.py\nfile2.py\n"
        mock_run.return_value.returncode = 0
        
        files = reviewer._get_staged_files()
        
        assert files == ["file1.py", "file2.py"]
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_get_staged_files_error(self, mock_run, reviewer):
        """Test getting staged files with error"""
        mock_run.side_effect = Exception("Git error")
        
        files = reviewer._get_staged_files()
        
        assert files == []
    
    def test_create_review_prompt(self, reviewer):
        """Test prompt creation"""
        content = "def test(): pass"
        diff_content = "+ def test(): pass"
        file_path = "test.py"
        
        prompt = reviewer._create_review_prompt(content, diff_content, file_path)
        
        assert "test.py" in prompt
        assert "def test(): pass" in prompt
        assert "JSON format" in prompt
        assert "security" in prompt.lower()
        assert "performance" in prompt.lower()
    
    @patch('requests.post')
    def test_call_mistral_success(self, mock_post, reviewer):
        """Test successful Mistral API call"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"issues": [], "summary": "test"}'}}]
        }
        mock_post.return_value = mock_response
        
        response = reviewer._call_mistral("test prompt")
        
        assert response == '{"issues": [], "summary": "test"}'
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_call_mistral_error(self, mock_post, reviewer):
        """Test Mistral API call with error"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        response = reviewer._call_mistral("test prompt")
        
        assert response == ""
    
    def test_parse_mistral_response_valid(self, reviewer):
        """Test parsing valid Mistral response"""
        response = '{"issues": [], "suggestions": [], "code_quality_score": 0.8, "summary": "test"}'
        
        result = reviewer._parse_mistral_response(response, "test.py")
        
        assert result["code_quality_score"] == 0.8
        assert result["summary"] == "test"
        assert isinstance(result["issues"], list)
    
    def test_parse_mistral_response_invalid(self, reviewer):
        """Test parsing invalid Mistral response"""
        response = "invalid json"
        
        result = reviewer._parse_mistral_response(response, "test.py")
        
        assert result["code_quality_score"] == 0.5
        assert "fallback" in result["summary"]
    
    def test_validate_analysis(self, reviewer):
        """Test analysis validation"""
        analysis = {
            "issues": [{"type": "bug", "severity": "high"}],
            "code_quality_score": 1.5,  # Should be clamped to 1.0
            "overall_score": -0.5,      # Should be clamped to 0.0
            "should_block_commit": True
        }
        
        validated = reviewer._validate_analysis(analysis)
        
        assert validated["code_quality_score"] == 1.0
        assert validated["overall_score"] == 0.0
        assert validated["should_block_commit"] == True
        assert len(validated["issues"]) == 1
    
    def test_create_review_result(self, reviewer):
        """Test creating review result"""
        analysis = {
            "issues": [{"type": "bug", "severity": "high"}],
            "suggestions": [{"type": "improvement"}],
            "security_concerns": [{"type": "vulnerability"}],
            "performance_issues": [{"type": "complexity"}],
            "code_quality_score": 0.8,
            "overall_score": 0.7,
            "should_block_commit": False,
            "summary": "Test summary"
        }
        
        result = reviewer._create_review_result("test.py", analysis)
        
        assert isinstance(result, CodeReviewResult)
        assert result.file_path == "test.py"
        assert result.code_quality_score == 0.8
        assert result.overall_score == 0.7
        assert result.should_block_commit == False
        assert result.review_summary == "Test summary"
        assert len(result.issues) == 1
        assert len(result.suggestions) == 1
    
    def test_should_block_commit_no_issues(self, reviewer):
        """Test commit blocking with no issues"""
        results = [
            CodeReviewResult(
                file_path="test.py",
                issues=[],
                suggestions=[],
                security_concerns=[],
                performance_issues=[],
                code_quality_score=0.9,
                overall_score=0.9,
                should_block_commit=False,
                review_summary="Good code",
                timestamp=None
            )
        ]
        
        should_block, reasons = reviewer.should_block_commit(results)
        
        assert should_block == False
        assert len(reasons) == 0
    
    def test_should_block_commit_with_issues(self, reviewer):
        """Test commit blocking with issues"""
        results = [
            CodeReviewResult(
                file_path="test.py",
                issues=[{"severity": "high"}],
                suggestions=[],
                security_concerns=[{"severity": "critical"}],
                performance_issues=[],
                code_quality_score=0.5,  # Below threshold
                overall_score=0.5,
                should_block_commit=True,
                review_summary="Bad code",
                timestamp=None
            )
        ]
        
        should_block, reasons = reviewer.should_block_commit(results)
        
        assert should_block == True
        assert len(reasons) > 0
        assert any("security" in reason.lower() for reason in reasons)
        assert any("quality" in reason.lower() for reason in reasons)
    
    def test_generate_report(self, reviewer):
        """Test report generation"""
        results = [
            CodeReviewResult(
                file_path="test.py",
                issues=[{"severity": "high", "description": "Test issue"}],
                suggestions=[{"description": "Test suggestion"}],
                security_concerns=[{"severity": "critical", "description": "Test security"}],
                performance_issues=[],
                code_quality_score=0.8,
                overall_score=0.7,
                should_block_commit=False,
                review_summary="Test summary",
                timestamp=None
            )
        ]
        
        report = reviewer.generate_report(results)
        
        assert "MISTRAL CODE REVIEW REPORT" in report
        assert "test.py" in report
        assert "Test issue" in report
        assert "Test suggestion" in report
        assert "Test security" in report
        assert "COMMIT APPROVED" in report

class TestCodeReviewIntegration:
    """Integration tests for code review"""
    
    @patch('src.code_reviewer.MistralCodeReviewer._call_mistral')
    def test_review_file_integration(self, mock_call_mistral):
        """Test full file review integration"""
        # Mock Mistral response
        mock_response = json.dumps({
            "issues": [{"type": "bug", "severity": "medium", "description": "Test issue"}],
            "suggestions": [{"type": "improvement", "description": "Test suggestion"}],
            "security_concerns": [],
            "performance_issues": [],
            "code_quality_score": 0.8,
            "overall_score": 0.7,
            "should_block_commit": False,
            "summary": "Good code with minor issues"
        })
        mock_call_mistral.return_value = mock_response
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('def test(): pass')
            temp_file = f.name
        
        try:
            reviewer = MistralCodeReviewer(".")
            result = reviewer.review_file(temp_file)
            
            assert isinstance(result, CodeReviewResult)
            assert result.code_quality_score == 0.8
            assert len(result.issues) == 1
            assert len(result.suggestions) == 1
            assert result.should_block_commit == False
            
        finally:
            os.unlink(temp_file)
    
    @patch('src.code_reviewer.MistralCodeReviewer._call_mistral')
    def test_review_staged_changes_integration(self, mock_call_mistral):
        """Test staged changes review integration"""
        # Mock Mistral response
        mock_response = json.dumps({
            "issues": [],
            "suggestions": [],
            "security_concerns": [],
            "performance_issues": [],
            "code_quality_score": 0.9,
            "overall_score": 0.9,
            "should_block_commit": False,
            "summary": "Excellent code"
        })
        mock_call_mistral.return_value = mock_response
        
        # Mock git staged files
        with patch.object(MistralCodeReviewer, '_get_staged_files') as mock_staged:
            mock_staged.return_value = ["test.py"]
            
            reviewer = MistralCodeReviewer(".")
            results = reviewer.review_staged_changes()
            
            assert len(results) == 1
            assert results[0].code_quality_score == 0.9
            assert results[0].should_block_commit == False

class TestDecorators:
    """Test code review decorators"""
    
    @patch('src.code_reviewer.run_pre_commit_review')
    def test_with_code_review_decorator(self, mock_review):
        """Test with_code_review decorator"""
        mock_review.return_value = True
        
        @with_code_review(".")
        def test_function():
            return "success"
        
        result = test_function()
        
        assert result == "success"
        mock_review.assert_called_once()
    
    @patch('src.code_reviewer.run_pre_commit_review')
    def test_with_code_review_decorator_blocked(self, mock_review):
        """Test with_code_review decorator with blocked commit"""
        mock_review.return_value = False
        
        @with_code_review(".")
        def test_function():
            return "success"
        
        result = test_function()
        
        assert result is None  # Function should not execute

class TestPreCommitHook:
    """Test pre-commit hook functionality"""
    
    @patch('src.code_reviewer.MistralCodeReviewer._call_mistral')
    def test_run_pre_commit_review_success(self, mock_call_mistral):
        """Test successful pre-commit review"""
        # Mock Mistral response
        mock_response = json.dumps({
            "issues": [],
            "suggestions": [],
            "security_concerns": [],
            "performance_issues": [],
            "code_quality_score": 0.9,
            "overall_score": 0.9,
            "should_block_commit": False,
            "summary": "Excellent code"
        })
        mock_call_mistral.return_value = mock_response
        
        # Mock git staged files
        with patch.object(MistralCodeReviewer, '_get_staged_files') as mock_staged:
            mock_staged.return_value = ["test.py"]
            
            success = run_pre_commit_review(".")
            
            assert success == True
    
    @patch('src.code_reviewer.MistralCodeReviewer._call_mistral')
    def test_run_pre_commit_review_blocked(self, mock_call_mistral):
        """Test pre-commit review that blocks commit"""
        # Mock Mistral response with issues
        mock_response = json.dumps({
            "issues": [{"severity": "high", "description": "Critical bug"}],
            "suggestions": [],
            "security_concerns": [],
            "performance_issues": [],
            "code_quality_score": 0.3,
            "overall_score": 0.3,
            "should_block_commit": True,
            "summary": "Critical issues found"
        })
        mock_call_mistral.return_value = mock_response
        
        # Mock git staged files
        with patch.object(MistralCodeReviewer, '_get_staged_files') as mock_staged:
            mock_staged.return_value = ["test.py"]
            
            success = run_pre_commit_review(".")
            
            assert success == False

if __name__ == "__main__":
    pytest.main([__file__])
