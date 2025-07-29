import openai
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
import os
from dataclasses import dataclass
from enum import Enum
import requests
from src.config_manager import ConfigManager

class ValidationLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    is_valid: bool
    confidence_score: float
    validation_level: ValidationLevel
    issues: List[str]
    recommendations: List[str]
    timestamp: datetime

class AIValidator:
    """
    AI-powered validation system for cross-checking developer work data
    """
    
    def __init__(self, openai_api_key: str, repo_path: str):
        self.openai_api_key = openai_api_key
        self.repo_path = repo_path
        self.db_path = os.path.join(repo_path, ".smartgit_tracker.db")
        self._setup_logging()
        # Load LLM config
        self.config_manager = ConfigManager(repo_path)
        self.llm_config = self.config_manager.get_llm_config()
        self.provider = self.llm_config.get('provider', 'ollama')
        self.ollama_endpoint = self.llm_config.get('ollama_endpoint', 'http://localhost:11434/v1/chat/completions')
        self.mistral_endpoint = self.llm_config.get('mistral_endpoint', 'http://localhost:11434/v1/chat/completions')
        self.openai_model = self.llm_config.get('openai_model', 'gpt-3.5-turbo')
        self.ollama_model = self.llm_config.get('ollama_model', 'mistral')
        self.mistral_model = self.llm_config.get('mistral_model', 'mistral-7b-instruct')
        # Only set openai.api_key if using openai
        if self.provider == 'openai':
            import openai
            openai.api_key = openai_api_key
    
    def _setup_logging(self):
        """Setup logging for validation"""
        self.logger = logging.getLogger('AIValidator')
    
    def validate_work_session(self, session_data: Dict) -> ValidationResult:
        """
        Comprehensive validation of a work session using AI
        """
        validation_checks = []
        
        # 1. Temporal consistency check
        temporal_check = self._validate_temporal_consistency(session_data)
        validation_checks.append(temporal_check)
        
        # 2. Activity pattern validation
        activity_check = self._validate_activity_patterns(session_data)
        validation_checks.append(activity_check)
        
        # 3. Cross-reference with git activity
        git_check = self._validate_git_activity(session_data)
        validation_checks.append(git_check)
        
        # 4. AI-powered content analysis
        ai_check = self._ai_validate_session(session_data)
        validation_checks.append(ai_check)
        
        # 5. Anomaly detection
        anomaly_check = self._detect_anomalies(session_data)
        validation_checks.append(anomaly_check)
        
        # Aggregate results
        return self._aggregate_validation_results(validation_checks)
    
    def _validate_temporal_consistency(self, session_data: Dict) -> Dict:
        """Validate temporal consistency of work session"""
        issues = []
        recommendations = []
        
        start_time = datetime.fromisoformat(session_data['start_time'])
        end_time = datetime.fromisoformat(session_data['end_time']) if session_data['end_time'] else datetime.now()
        
        # Check session duration
        duration = (end_time - start_time).total_seconds() / 3600  # hours
        
        if duration > 12:
            issues.append("Session duration exceeds 12 hours - potential data error")
            recommendations.append("Review session boundaries for accuracy")
        
        if duration < 0.1:  # Less than 6 minutes
            issues.append("Session duration too short - may be incomplete")
            recommendations.append("Verify session start/end times")
        
        # Check for overlapping sessions
        overlapping_sessions = self._check_overlapping_sessions(session_data)
        if overlapping_sessions:
            issues.append(f"Found {len(overlapping_sessions)} overlapping sessions")
            recommendations.append("Review session timing for conflicts")
        
        confidence = 1.0 - (len(issues) * 0.2)
        
        return {
            'type': 'temporal_consistency',
            'is_valid': len(issues) == 0,
            'confidence': max(confidence, 0.0),
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _validate_activity_patterns(self, session_data: Dict) -> Dict:
        """Validate activity patterns for consistency"""
        issues = []
        recommendations = []
        
        # Check for suspicious patterns
        screen_activity = session_data.get('screen_activity', False)
        face_detected = session_data.get('face_detected', False)
        confidence_score = session_data.get('confidence_score', 0.0)
        
        # High confidence but no face detected
        if confidence_score > 0.8 and not face_detected:
            issues.append("High confidence score but no face detected")
            recommendations.append("Verify face detection system")
        
        # Low confidence but face detected
        if confidence_score < 0.3 and face_detected:
            issues.append("Face detected but low confidence score")
            recommendations.append("Check activity detection algorithms")
        
        # No screen activity but high confidence
        if not screen_activity and confidence_score > 0.7:
            issues.append("High confidence without screen activity")
            recommendations.append("Review screen activity detection")
        
        confidence = 1.0 - (len(issues) * 0.25)
        
        return {
            'type': 'activity_patterns',
            'is_valid': len(issues) == 0,
            'confidence': max(confidence, 0.0),
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _validate_git_activity(self, session_data: Dict) -> Dict:
        """Cross-reference with git activity"""
        issues = []
        recommendations = []
        
        developer_id = session_data['developer_id']
        branch_name = session_data['branch_name']
        start_time = datetime.fromisoformat(session_data['start_time'])
        end_time = datetime.fromisoformat(session_data['end_time']) if session_data['end_time'] else datetime.now()
        
        # Get git commits during session
        git_commits = self._get_git_commits_in_period(developer_id, start_time, end_time)
        
        if not git_commits:
            issues.append("No git activity detected during work session")
            recommendations.append("Verify developer was working on correct branch")
        else:
            # Check if commits match the branch
            branch_commits = [c for c in git_commits if c['branch'] == branch_name]
            if not branch_commits:
                issues.append("Git commits don't match tracked branch")
                recommendations.append("Verify branch tracking accuracy")
        
        confidence = 1.0 - (len(issues) * 0.3)
        
        return {
            'type': 'git_activity',
            'is_valid': len(issues) == 0,
            'confidence': max(confidence, 0.0),
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _ai_validate_session(self, session_data: Dict) -> Dict:
        """Use LLM to validate session data (OpenAI, Ollama, or local Mistral)"""
        try:
            analysis_prompt = self._create_analysis_prompt(session_data)
            if self.provider == 'openai':
                import openai
                response = openai.ChatCompletion.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": "You are an AI expert in validating developer work sessions. Analyze the provided data for consistency, anomalies, and credibility."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    max_tokens=500,
                    temperature=0.3
                )
                ai_response = response.choices[0].message.content
            elif self.provider == 'ollama':
                ai_response = self._call_ollama_llm(analysis_prompt)
            elif self.provider == 'mistral':
                ai_response = self._call_mistral_llm(analysis_prompt)
            else:
                raise ValueError(f"Unknown LLM provider: {self.provider}")
            return self._parse_ai_validation_response(ai_response)
        except Exception as e:
            self.logger.error(f"AI validation failed: {e}")
            return {
                'type': 'ai_validation',
                'is_valid': True,  # Default to valid if AI fails
                'confidence': 0.5,
                'issues': ["AI validation unavailable"],
                'recommendations': ["Check AI service connectivity"]
            }
    
    def _create_analysis_prompt(self, session_data: Dict) -> str:
        """Create prompt for AI analysis"""
        return f"""
        Analyze this developer work session for validity and consistency:
        
        Developer: {session_data['developer_id']}
        Branch: {session_data['branch_name']}
        Duration: {session_data.get('duration_hours', 'unknown')} hours
        Activity Type: {session_data['activity_type']}
        Confidence Score: {session_data['confidence_score']}
        Screen Activity: {session_data['screen_activity']}
        Face Detected: {session_data['face_detected']}
        Validation Status: {session_data['validation_status']}
        
        Please analyze for:
        1. Data consistency
        2. Suspicious patterns
        3. Credibility indicators
        4. Potential issues
        
        Respond with a JSON format:
        {{
            "is_valid": boolean,
            "confidence": float (0-1),
            "issues": [list of issues],
            "recommendations": [list of recommendations],
            "risk_level": "low|medium|high"
        }}
        """
    
    def _parse_ai_validation_response(self, response: str) -> Dict:
        """Parse AI validation response"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            
            result = json.loads(json_str)
            
            return {
                'type': 'ai_validation',
                'is_valid': result.get('is_valid', True),
                'confidence': result.get('confidence', 0.5),
                'issues': result.get('issues', []),
                'recommendations': result.get('recommendations', [])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to parse AI response: {e}")
            return {
                'type': 'ai_validation',
                'is_valid': True,
                'confidence': 0.5,
                'issues': ["AI response parsing failed"],
                'recommendations': ["Review AI validation system"]
            }
    
    def _detect_anomalies(self, session_data: Dict) -> Dict:
        """Detect anomalies in work patterns"""
        issues = []
        recommendations = []
        
        # Get historical data for comparison
        historical_sessions = self._get_historical_sessions(session_data['developer_id'])
        
        if historical_sessions:
            # Compare with historical patterns
            avg_duration = sum(s['duration_hours'] for s in historical_sessions) / len(historical_sessions)
            current_duration = session_data.get('duration_hours', 0)
            
            # Check for unusual duration
            if current_duration > avg_duration * 2:
                issues.append("Session duration significantly longer than average")
                recommendations.append("Verify session boundaries")
            
            if current_duration < avg_duration * 0.3:
                issues.append("Session duration significantly shorter than average")
                recommendations.append("Check for incomplete session")
            
            # Check for unusual activity patterns
            avg_confidence = sum(s['confidence_score'] for s in historical_sessions) / len(historical_sessions)
            current_confidence = session_data['confidence_score']
            
            if abs(current_confidence - avg_confidence) > 0.3:
                issues.append("Confidence score deviates significantly from historical average")
                recommendations.append("Review activity detection accuracy")
        
        confidence = 1.0 - (len(issues) * 0.2)
        
        return {
            'type': 'anomaly_detection',
            'is_valid': len(issues) == 0,
            'confidence': max(confidence, 0.0),
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _aggregate_validation_results(self, validation_checks: List[Dict]) -> ValidationResult:
        """Aggregate multiple validation results"""
        all_issues = []
        all_recommendations = []
        total_confidence = 0.0
        valid_checks = 0
        
        for check in validation_checks:
            all_issues.extend(check['issues'])
            all_recommendations.extend(check['recommendations'])
            total_confidence += check['confidence']
            if check['is_valid']:
                valid_checks += 1
        
        # Calculate overall confidence
        avg_confidence = total_confidence / len(validation_checks) if validation_checks else 0.0
        
        # Determine validation level
        if avg_confidence >= 0.8 and len(all_issues) == 0:
            validation_level = ValidationLevel.HIGH
        elif avg_confidence >= 0.6 and len(all_issues) <= 2:
            validation_level = ValidationLevel.MEDIUM
        elif avg_confidence >= 0.4:
            validation_level = ValidationLevel.LOW
        else:
            validation_level = ValidationLevel.CRITICAL
        
        # Overall validity
        is_valid = valid_checks >= len(validation_checks) * 0.7  # 70% of checks must pass
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_score=avg_confidence,
            validation_level=validation_level,
            issues=all_issues,
            recommendations=all_recommendations,
            timestamp=datetime.now()
        )
    
    def _check_overlapping_sessions(self, session_data: Dict) -> List[Dict]:
        """Check for overlapping sessions with the same developer"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_time = datetime.fromisoformat(session_data['start_time'])
        end_time = datetime.fromisoformat(session_data['end_time']) if session_data['end_time'] else datetime.now()
        developer_id = session_data['developer_id']
        
        cursor.execute('''
            SELECT id, start_time, end_time FROM work_sessions 
            WHERE developer_id = ? AND id != ? AND
            ((start_time BETWEEN ? AND ?) OR (end_time BETWEEN ? AND ?) OR
             (start_time <= ? AND end_time >= ?))
        ''', (developer_id, session_data.get('id', 0), start_time, end_time, start_time, end_time, start_time, end_time))
        
        overlapping = cursor.fetchall()
        conn.close()
        
        return [{'id': row[0], 'start_time': row[1], 'end_time': row[2]} for row in overlapping]
    
    def _get_git_commits_in_period(self, developer_id: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get git commits for a developer in a time period"""
        # This would integrate with git utilities to get actual commit data
        # For now, return mock data
        return [
            {'hash': 'abc123', 'branch': 'feature/a', 'timestamp': start_time + timedelta(hours=1)},
            {'hash': 'def456', 'branch': 'feature/a', 'timestamp': start_time + timedelta(hours=2)}
        ]
    
    def _get_historical_sessions(self, developer_id: str, days: int = 30) -> List[Dict]:
        """Get historical sessions for a developer"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT start_time, end_time, confidence_score, activity_type
            FROM work_sessions 
            WHERE developer_id = ? AND start_time >= ?
        ''', (developer_id, start_date))
        
        results = cursor.fetchall()
        conn.close()
        
        sessions = []
        for row in results:
            start_time = datetime.fromisoformat(row[0])
            end_time = datetime.fromisoformat(row[1]) if row[1] else datetime.now()
            duration_hours = (end_time - start_time).total_seconds() / 3600
            
            sessions.append({
                'duration_hours': duration_hours,
                'confidence_score': row[2],
                'activity_type': row[3]
            })
        
        return sessions
    
    def validate_project_estimates(self, estimates: Dict) -> ValidationResult:
        """Validate project estimates for credibility"""
        issues = []
        recommendations = []
        
        # Check for unrealistic estimates
        if estimates.get('estimated_completion_hours', 0) > 1000:
            issues.append("Estimated completion time exceeds 1000 hours")
            recommendations.append("Review estimation methodology")
        
        if estimates.get('total_hours_spent', 0) < 1:
            issues.append("Very low hours spent - may indicate tracking issues")
            recommendations.append("Verify tracking system accuracy")
        
        # Check data quality
        data_quality = estimates.get('data_quality', 'low')
        if data_quality == 'low':
            issues.append("Low data quality for estimates")
            recommendations.append("Improve tracking accuracy")
        
        confidence = 1.0 - (len(issues) * 0.25)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence_score=max(confidence, 0.0),
            validation_level=ValidationLevel.HIGH if confidence > 0.8 else ValidationLevel.MEDIUM,
            issues=issues,
            recommendations=recommendations,
            timestamp=datetime.now()
        ) 

    def _call_ollama_llm(self, prompt: str) -> str:
        """Call local Ollama LLM HTTP API"""
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": "You are an AI expert in validating developer work sessions. Analyze the provided data for consistency, anomalies, and credibility."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.3
        }
        try:
            resp = requests.post(self.ollama_endpoint, headers=headers, data=json.dumps(data), timeout=60)
            resp.raise_for_status()
            result = resp.json()
            # Ollama API compatibility
            if 'choices' in result and result['choices']:
                return result['choices'][0].get('message', {}).get('content', '')
            elif 'message' in result:
                return result['message']['content']
            elif 'content' in result:
                return result['content']
            else:
                return str(result)
        except Exception as e:
            self.logger.error(f"Ollama LLM call failed: {e}")
            return "{\"is_valid\": true, \"confidence\": 0.5, \"issues\": [\"Ollama LLM call failed\"], \"recommendations\": [\"Check local Ollama server\"]}"

    def _call_mistral_llm(self, prompt: str) -> str:
        """Call local Mistral LLM HTTP API"""
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.mistral_model,
            "messages": [
                {"role": "system", "content": "You are an AI expert in validating developer work sessions. Analyze the provided data for consistency, anomalies, and credibility."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.3
        }
        try:
            resp = requests.post(self.mistral_endpoint, headers=headers, data=json.dumps(data), timeout=60)
            resp.raise_for_status()
            result = resp.json()
            # Mistral/Ollama/LMStudio API compatibility
            if 'choices' in result and result['choices']:
                return result['choices'][0].get('message', {}).get('content', '')
            elif 'message' in result:
                return result['message']['content']
            elif 'content' in result:
                return result['content']
            else:
                return str(result)
        except Exception as e:
            self.logger.error(f"Mistral LLM call failed: {e}")
            return "{\"is_valid\": true, \"confidence\": 0.5, \"issues\": [\"Mistral LLM call failed\"], \"recommendations\": [\"Check local LLM server\"]}" 