import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Ensure src is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

class TestLazyTracker(unittest.TestCase):
    def test_import_without_dependencies(self):
        """Test that AgenticTracker can be imported and initialized even if CV dependencies are missing."""
        # Mock sys.modules to simulate missing dependencies
        missing_modules = ['cv2', 'numpy', 'pyautogui', 'face_recognition', 'mediapipe']

        with patch.dict(sys.modules, {mod: None for mod in missing_modules}):
            from agentic_tracker import AgenticTracker, ActivityType

            tracker = AgenticTracker(repo_path=".")
            self.assertIsNone(tracker.face_cascade)
            self.assertIsNone(tracker.face_detection)
            self.assertFalse(tracker._ensure_cv_initialized())

    def test_lazy_initialization_failure(self):
        """Test that methods handle missing dependencies gracefully."""
        missing_modules = ['cv2', 'numpy', 'pyautogui', 'face_recognition', 'mediapipe']

        with patch.dict(sys.modules, {mod: None for mod in missing_modules}):
            from agentic_tracker import AgenticTracker

            tracker = AgenticTracker(repo_path=".")

            # These should not raise ImportError but handle it gracefully
            self.assertFalse(tracker._detect_faces(None))
            self.assertEqual(tracker._analyze_screen_content(None), 0.0)
            self.assertIsNone(tracker._identify_developer(None))
            self.assertEqual(tracker._save_webcam_image(None), "")

if __name__ == '__main__':
    unittest.main()
