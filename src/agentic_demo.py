#!/usr/bin/env python3
"""
Agentic AI Demo: Alice working on feature branch A
Demonstrates automated developer tracking, data validation, and project estimates
"""

import os
import sys
import time
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentic_tracker import AgenticTracker
from ai_validator import AIValidator, ValidationLevel
from git_utils import GitUtils

class AgenticDemo:
    """
    Demo class for showcasing the agentic AI tracking system
    """
    
    def __init__(self, repo_path: str, openai_api_key: str = None):
        self.repo_path = repo_path
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        # Initialize components
        self.tracker = AgenticTracker(repo_path)
        self.validator = AIValidator(self.openai_api_key, repo_path) if self.openai_api_key else None
        self.git_utils = GitUtils(repo_path)
        
        # Demo state
        self.demo_data = {
            'alice_id': 'alice_dev_001',
            'alice_name': 'Alice Johnson',
            'feature_branch': 'feature/user-authentication',
            'project_name': 'SmartGitMergePT'
        }
    
    def setup_demo_environment(self):
        """Setup demo environment with sample data"""
        print("🚀 Setting up Agentic AI Demo Environment...")
        
        # Create demo directories
        tracking_dir = os.path.join(self.repo_path, "tracking_data")
        os.makedirs(tracking_dir, exist_ok=True)
        os.makedirs(os.path.join(tracking_dir, "screenshots"), exist_ok=True)
        os.makedirs(os.path.join(tracking_dir, "webcam"), exist_ok=True)
        
        # Register Alice as a developer (mock face data)
        self._register_alice()
        
        print("✅ Demo environment setup complete!")
    
    def _register_alice(self):
        """Register Alice as a developer with mock face data"""
        try:
            # Create a mock face image for Alice
            face_image_path = self._create_mock_face_image()
            
            # Register Alice
            self.tracker.register_developer(
                developer_id=self.demo_data['alice_id'],
                name=self.demo_data['alice_name'],
                face_image_path=face_image_path
            )
            
            print(f"👤 Registered developer: {self.demo_data['alice_name']}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not register Alice (mock mode): {e}")
    
    def _create_mock_face_image(self):
        """Create a mock face image for demo purposes"""
        # In a real scenario, this would be an actual photo
        # For demo purposes, we'll create a placeholder
        face_dir = os.path.join(self.repo_path, "demo_faces")
        os.makedirs(face_dir, exist_ok=True)
        
        face_path = os.path.join(face_dir, "alice_face.jpg")
        
        # Create a simple placeholder file
        with open(face_path, 'w') as f:
            f.write("# Mock face image for demo purposes")
        
        return face_path
    
    def simulate_alice_work_session(self, duration_minutes: int = 30):
        """Simulate Alice working on the feature branch"""
        print(f"\n👩‍💻 Simulating Alice's work session on {self.demo_data['feature_branch']}...")
        print(f"⏱️  Duration: {duration_minutes} minutes")
        
        # Start tracking Alice
        self.tracker.start_tracking(
            developer_id=self.demo_data['alice_id'],
            branch_name=self.demo_data['feature_branch']
        )
        
        # Simulate work session
        start_time = datetime.now()
        print(f"🕐 Session started at: {start_time.strftime('%H:%M:%S')}")
        
        # Simulate different activities during the session
        activities = [
            ("Coding authentication logic", 0.8),
            ("Writing unit tests", 0.7),
            ("Code review", 0.6),
            ("Documentation", 0.5),
            ("Debugging", 0.9),
            ("Git commits", 0.8)
        ]
        
        activity_duration = duration_minutes // len(activities)
        
        for i, (activity, confidence) in enumerate(activities):
            print(f"  📝 {activity} (confidence: {confidence:.1f})")
            
            # Update tracker with simulated activity
            self._simulate_activity(confidence)
            
            # Wait for activity duration
            time.sleep(activity_duration * 60 / len(activities))  # Convert to seconds
        
        # Stop tracking
        self.tracker.stop_tracking()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        print(f"🕐 Session ended at: {end_time.strftime('%H:%M:%S')}")
        print(f"⏱️  Actual duration: {duration:.1f} minutes")
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'duration_minutes': duration,
            'activities': activities
        }
    
    def _simulate_activity(self, confidence: float):
        """Simulate activity with given confidence level"""
        # In a real scenario, this would be actual screen/webcam monitoring
        # For demo purposes, we'll simulate the data
        
        if hasattr(self.tracker, 'current_session') and self.tracker.current_session:
            self.tracker.current_session.confidence_score = confidence
            self.tracker.current_session.screen_activity = confidence > 0.3
            self.tracker.current_session.face_detected = confidence > 0.5
            self.tracker.current_session.keyboard_activity = confidence > 0.4
            self.tracker.current_session.mouse_activity = confidence > 0.4
    
    def validate_session_data(self, session_data: dict):
        """Validate the session data using AI"""
        print("\n🔍 Validating session data with AI...")
        
        if not self.validator:
            print("⚠️  AI validator not available (no OpenAI API key)")
            return None
        
        # Prepare session data for validation
        validation_data = {
            'developer_id': self.demo_data['alice_id'],
            'branch_name': self.demo_data['feature_branch'],
            'start_time': session_data['start_time'].isoformat(),
            'end_time': session_data['end_time'].isoformat(),
            'duration_hours': session_data['duration_minutes'] / 60,
            'activity_type': 'coding',
            'confidence_score': 0.75,  # Average confidence
            'screen_activity': True,
            'face_detected': True,
            'validation_status': 'pending'
        }
        
        # Perform AI validation
        validation_result = self.validator.validate_work_session(validation_data)
        
        # Display validation results
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
        
        return validation_result
    
    def generate_project_estimates(self):
        """Generate project estimates based on tracked data"""
        print("\n📊 Generating project estimates...")
        
        # Get project estimates
        estimates = self.tracker.get_project_estimates(self.demo_data['feature_branch'])
        
        print(f"📈 Project: {self.demo_data['project_name']}")
        print(f"🌿 Branch: {estimates['branch_name']}")
        print(f"⏱️  Total Hours Spent: {estimates['total_hours_spent']:.2f}")
        print(f"👥 Developers Working: {estimates['developers_working']}")
        print(f"🎯 Average Confidence: {estimates['avg_confidence']:.2f}")
        print(f"📊 Data Quality: {estimates['data_quality'].upper()}")
        print(f"⏳ Estimated Completion: {estimates['estimated_completion_hours']:.2f} hours")
        
        # Validate estimates if AI validator is available
        if self.validator:
            estimate_validation = self.validator.validate_project_estimates(estimates)
            print(f"\n🔍 Estimate Validation: {'VALID' if estimate_validation.is_valid else 'INVALID'}")
            print(f"🎯 Estimate Confidence: {estimate_validation.confidence_score:.2f}")
        
        return estimates
    
    def get_alice_stats(self, days: int = 7):
        """Get Alice's work statistics"""
        print(f"\n📊 Alice's Work Statistics (Last {days} days)...")
        
        stats = self.tracker.get_developer_stats(self.demo_data['alice_id'], days)
        
        print(f"👤 Developer: {stats['developer_id']}")
        print(f"⏱️  Total Hours: {stats['total_hours']:.2f}")
        print(f"📅 Sessions: {stats['sessions']}")
        print(f"🌿 Branches Worked On: {', '.join(stats['branches'])}")
        print(f"🎯 Average Confidence: {stats['avg_confidence']:.2f}")
        print(f"✅ Validation Rate: {stats['validation_rate']:.1%}")
        
        return stats
    
    def run_comprehensive_demo(self):
        """Run the complete agentic AI demo"""
        print("=" * 60)
        print("🤖 AGENTIC AI DEVELOPER TRACKING DEMO")
        print("=" * 60)
        print("Scenario: Alice working on feature branch A")
        print("Features: Screen recording, webcam monitoring, AI validation")
        print("=" * 60)
        
        # Setup environment
        self.setup_demo_environment()
        
        # Simulate Alice's work session
        session_data = self.simulate_alice_work_session(duration_minutes=15)  # Shorter for demo
        
        # Validate session data
        validation_result = self.validate_session_data(session_data)
        
        # Generate project estimates
        estimates = self.generate_project_estimates()
        
        # Get Alice's stats
        stats = self.get_alice_stats()
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 DEMO SUMMARY")
        print("=" * 60)
        print(f"✅ Session tracked: {session_data['duration_minutes']:.1f} minutes")
        print(f"✅ AI validation: {'PASSED' if validation_result and validation_result.is_valid else 'FAILED'}")
        print(f"✅ Project estimates generated")
        print(f"✅ Developer statistics collected")
        print("=" * 60)
        
        return {
            'session_data': session_data,
            'validation_result': validation_result,
            'estimates': estimates,
            'stats': stats
        }

def main():
    """Main demo function"""
    # Get repository path
    repo_path = os.getcwd()
    
    # Check for OpenAI API key
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("⚠️  Warning: No OpenAI API key found. AI validation will be limited.")
        print("   Set OPENAI_API_KEY environment variable for full AI validation.")
    
    # Create and run demo
    demo = AgenticDemo(repo_path, openai_api_key)
    
    try:
        results = demo.run_comprehensive_demo()
        
        # Save results to file
        results_file = os.path.join(repo_path, "demo_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'session_duration_minutes': results['session_data']['duration_minutes'],
                'validation_passed': results['validation_result'].is_valid if results['validation_result'] else False,
                'total_hours_spent': results['estimates']['total_hours_spent'],
                'estimated_completion_hours': results['estimates']['estimated_completion_hours']
            }, f, indent=2)
        
        print(f"\n💾 Demo results saved to: {results_file}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 