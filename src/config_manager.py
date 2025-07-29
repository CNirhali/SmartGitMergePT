import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class TrackingConfig:
    """Configuration for tracking settings"""
    screenshot_interval: int = 30  # seconds
    webcam_interval: int = 10      # seconds
    activity_threshold: float = 0.6  # confidence threshold
    idle_timeout: int = 300        # 5 minutes
    validation_interval: int = 300  # 5 minutes
    screen_recording: bool = True
    webcam_monitoring: bool = True
    keyboard_tracking: bool = True
    mouse_tracking: bool = True

@dataclass
class PrivacyConfig:
    """Configuration for privacy settings"""
    store_screenshots: bool = True
    store_webcam_images: bool = True
    encrypt_data: bool = False
    data_retention_days: int = 30
    anonymize_data: bool = False
    blur_faces: bool = True
    mask_sensitive_content: bool = True

@dataclass
class AIValidationConfig:
    """Configuration for AI validation"""
    enable_ai_validation: bool = True
    validation_confidence_threshold: float = 0.7
    cross_reference_git: bool = True
    anomaly_detection: bool = True
    temporal_consistency_check: bool = True
    activity_pattern_validation: bool = True

@dataclass
class NotificationConfig:
    """Configuration for notifications"""
    enable_notifications: bool = True
    notify_on_session_start: bool = True
    notify_on_session_end: bool = True
    notify_on_validation_failure: bool = True
    notify_on_anomaly_detection: bool = True
    email_notifications: bool = False
    slack_notifications: bool = False

@dataclass
class LLMConfig:
    """Configuration for LLM provider"""
    provider: str = "ollama"  # 'openai', 'ollama', or 'mistral'
    ollama_endpoint: str = "http://localhost:11434/v1/chat/completions"
    mistral_endpoint: str = "http://localhost:11434/v1/chat/completions"
    openai_model: str = "gpt-3.5-turbo"
    ollama_model: str = "mistral"
    mistral_model: str = "mistral-7b-instruct"

class ConfigManager:
    """
    Manages configuration for the agentic AI tracking system
    """
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.config_file = os.path.join(repo_path, ".smartgit_config.json")
        
        # Initialize default configurations
        self.tracking = TrackingConfig()
        self.privacy = PrivacyConfig()
        self.ai_validation = AIValidationConfig()
        self.notifications = NotificationConfig()
        self.llm = LLMConfig()
        
        # Load existing configuration if available
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Update configurations with loaded data
                if 'tracking' in config_data:
                    self._update_config(self.tracking, config_data['tracking'])
                if 'privacy' in config_data:
                    self._update_config(self.privacy, config_data['privacy'])
                if 'ai_validation' in config_data:
                    self._update_config(self.ai_validation, config_data['ai_validation'])
                if 'notifications' in config_data:
                    self._update_config(self.notifications, config_data['notifications'])
                if 'llm' in config_data:
                    self._update_config(self.llm, config_data['llm'])
                
                print(f"✅ Configuration loaded from {self.config_file}")
                
            except Exception as e:
                print(f"⚠️  Warning: Could not load configuration: {e}")
                print("   Using default configuration")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            config_data = {
                'tracking': asdict(self.tracking),
                'privacy': asdict(self.privacy),
                'ai_validation': asdict(self.ai_validation),
                'notifications': asdict(self.notifications),
                'llm': asdict(self.llm)
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"✅ Configuration saved to {self.config_file}")
            
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
    
    def _update_config(self, config_obj, data: Dict[str, Any]):
        """Update configuration object with new data"""
        for key, value in data.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
    
    def get_tracking_config(self) -> Dict[str, Any]:
        """Get tracking configuration as dictionary"""
        return asdict(self.tracking)
    
    def get_privacy_config(self) -> Dict[str, Any]:
        """Get privacy configuration as dictionary"""
        return asdict(self.privacy)
    
    def get_ai_validation_config(self) -> Dict[str, Any]:
        """Get AI validation configuration as dictionary"""
        return asdict(self.ai_validation)
    
    def get_notification_config(self) -> Dict[str, Any]:
        """Get notification configuration as dictionary"""
        return asdict(self.notifications)
    
    def get_llm_config(self) -> dict:
        return asdict(self.llm)
    
    def update_tracking_config(self, **kwargs):
        """Update tracking configuration"""
        for key, value in kwargs.items():
            if hasattr(self.tracking, key):
                setattr(self.tracking, key, value)
        self.save_config()
    
    def update_privacy_config(self, **kwargs):
        """Update privacy configuration"""
        for key, value in kwargs.items():
            if hasattr(self.privacy, key):
                setattr(self.privacy, key, value)
        self.save_config()
    
    def update_ai_validation_config(self, **kwargs):
        """Update AI validation configuration"""
        for key, value in kwargs.items():
            if hasattr(self.ai_validation, key):
                setattr(self.ai_validation, key, value)
        self.save_config()
    
    def update_notification_config(self, **kwargs):
        """Update notification configuration"""
        for key, value in kwargs.items():
            if hasattr(self.notifications, key):
                setattr(self.notifications, key, value)
        self.save_config()
    
    def update_llm_config(self, **kwargs):
        """Update LLM configuration"""
        for key, value in kwargs.items():
            if hasattr(self.llm, key):
                setattr(self.llm, key, value)
        self.save_config()
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration and return issues"""
        issues = []
        warnings = []
        
        # Validate tracking configuration
        if self.tracking.screenshot_interval < 5:
            issues.append("Screenshot interval too short (< 5 seconds)")
        if self.tracking.webcam_interval < 5:
            issues.append("Webcam interval too short (< 5 seconds)")
        if self.tracking.activity_threshold < 0 or self.tracking.activity_threshold > 1:
            issues.append("Activity threshold must be between 0 and 1")
        
        # Validate privacy configuration
        if self.privacy.data_retention_days < 1:
            issues.append("Data retention days must be at least 1")
        if self.privacy.data_retention_days > 365:
            warnings.append("Data retention set to more than 1 year")
        
        # Validate AI validation configuration
        if self.ai_validation.validation_confidence_threshold < 0 or self.ai_validation.validation_confidence_threshold > 1:
            issues.append("Validation confidence threshold must be between 0 and 1")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    def create_privacy_compliant_config(self):
        """Create a privacy-compliant configuration"""
        self.privacy.store_screenshots = False
        self.privacy.store_webcam_images = False
        self.privacy.encrypt_data = True
        self.privacy.data_retention_days = 7
        self.privacy.anonymize_data = True
        self.privacy.blur_faces = True
        self.privacy.mask_sensitive_content = True
        
        self.save_config()
        print("✅ Privacy-compliant configuration created")
    
    def create_performance_optimized_config(self):
        """Create a performance-optimized configuration"""
        self.tracking.screenshot_interval = 60  # 1 minute
        self.tracking.webcam_interval = 30      # 30 seconds
        self.tracking.validation_interval = 600  # 10 minutes
        
        self.privacy.store_screenshots = False
        self.privacy.store_webcam_images = False
        
        self.ai_validation.enable_ai_validation = False
        
        self.save_config()
        print("✅ Performance-optimized configuration created")
    
    def create_high_accuracy_config(self):
        """Create a high-accuracy configuration"""
        self.tracking.screenshot_interval = 15  # 15 seconds
        self.tracking.webcam_interval = 5       # 5 seconds
        self.tracking.validation_interval = 120  # 2 minutes
        self.tracking.activity_threshold = 0.8   # Higher threshold
        
        self.privacy.store_screenshots = True
        self.privacy.store_webcam_images = True
        self.privacy.encrypt_data = True
        
        self.ai_validation.enable_ai_validation = True
        self.ai_validation.validation_confidence_threshold = 0.9
        self.ai_validation.cross_reference_git = True
        self.ai_validation.anomaly_detection = True
        
        self.save_config()
        print("✅ High-accuracy configuration created")
    
    def print_current_config(self):
        """Print current configuration"""
        print("\n" + "=" * 60)
        print("🔧 CURRENT CONFIGURATION")
        print("=" * 60)
        
        print("\n📊 TRACKING CONFIGURATION:")
        for key, value in asdict(self.tracking).items():
            print(f"   {key}: {value}")
        
        print("\n🔒 PRIVACY CONFIGURATION:")
        for key, value in asdict(self.privacy).items():
            print(f"   {key}: {value}")
        
        print("\n🤖 AI VALIDATION CONFIGURATION:")
        for key, value in asdict(self.ai_validation).items():
            print(f"   {key}: {value}")
        
        print("\n🔔 NOTIFICATION CONFIGURATION:")
        for key, value in asdict(self.notifications).items():
            print(f"   {key}: {value}")
        
        print("\n🤖 LLM CONFIGURATION:")
        for key, value in asdict(self.llm).items():
            print(f"   {key}: {value}")
        
        # Validate configuration
        validation = self.validate_config()
        if validation['issues']:
            print("\n❌ CONFIGURATION ISSUES:")
            for issue in validation['issues']:
                print(f"   • {issue}")
        
        if validation['warnings']:
            print("\n⚠️  CONFIGURATION WARNINGS:")
            for warning in validation['warnings']:
                print(f"   • {warning}")
        
        print("=" * 60)
    
    def export_config(self, filepath: str):
        """Export configuration to file"""
        try:
            config_data = {
                'tracking': asdict(self.tracking),
                'privacy': asdict(self.privacy),
                'ai_validation': asdict(self.ai_validation),
                'notifications': asdict(self.notifications),
                'llm': asdict(self.llm)
            }
            
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"✅ Configuration exported to {filepath}")
            
        except Exception as e:
            print(f"❌ Error exporting configuration: {e}")
    
    def import_config(self, filepath: str):
        """Import configuration from file"""
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            
            # Update configurations with imported data
            if 'tracking' in config_data:
                self._update_config(self.tracking, config_data['tracking'])
            if 'privacy' in config_data:
                self._update_config(self.privacy, config_data['privacy'])
            if 'ai_validation' in config_data:
                self._update_config(self.ai_validation, config_data['ai_validation'])
            if 'notifications' in config_data:
                self._update_config(self.notifications, config_data['notifications'])
            if 'llm' in config_data:
                self._update_config(self.llm, config_data['llm'])
            
            self.save_config()
            print(f"✅ Configuration imported from {filepath}")
            
        except Exception as e:
            print(f"❌ Error importing configuration: {e}")

def main():
    """Demo function for configuration management"""
    repo_path = os.getcwd()
    config_manager = ConfigManager(repo_path)
    
    print("🔧 Agentic AI Tracking Configuration Manager")
    print("=" * 50)
    
    # Show current configuration
    config_manager.print_current_config()
    
    # Create different configuration presets
    print("\n🎯 Creating configuration presets...")
    
    # Privacy-compliant config
    config_manager.create_privacy_compliant_config()
    print("✅ Privacy-compliant configuration created")
    
    # Performance-optimized config
    config_manager.create_performance_optimized_config()
    print("✅ Performance-optimized configuration created")
    
    # High-accuracy config
    config_manager.create_high_accuracy_config()
    print("✅ High-accuracy configuration created")
    
    # Export configurations
    config_manager.export_config("privacy_config.json")
    config_manager.export_config("performance_config.json")
    config_manager.export_config("accuracy_config.json")
    
    print("\n✅ Configuration management demo completed!")

if __name__ == "__main__":
    main() 