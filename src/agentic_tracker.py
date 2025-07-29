import cv2
import numpy as np
import pyautogui
import psutil
import face_recognition
import mediapipe as mp
import sqlite3
import json
import logging
import threading
import time
import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
from dataclasses import dataclass
from enum import Enum
import tempfile
from pathlib import Path

class ActivityType(Enum):
    CODING = "coding"
    MEETING = "meeting"
    BREAK = "break"
    IDLE = "idle"
    UNKNOWN = "unknown"

@dataclass
class WorkSession:
    developer_id: str
    branch_name: str
    start_time: datetime
    end_time: Optional[datetime]
    activity_type: ActivityType
    confidence_score: float
    screen_activity: bool
    face_detected: bool
    keyboard_activity: bool
    mouse_activity: bool
    validation_status: str = "pending"

class AgenticTracker:
    """
    Agentic AI system for automated developer work tracking and validation
    """
    
    def __init__(self, repo_path: str, config: Dict = None):
        self.repo_path = repo_path
        self.config = config or self._default_config()
        self.db_path = os.path.join(repo_path, ".smartgit_tracker.db")
        self._setup_database()
        self._setup_logging()
        
        # Initialize tracking components
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.7)
        
        # Tracking state
        self.current_session: Optional[WorkSession] = None
        self.is_tracking = False
        self.tracking_thread = None
        
        # Known faces database
        self.known_faces = {}
        self._load_known_faces()
    
    def _default_config(self) -> Dict:
        return {
            "screenshot_interval": 30,  # seconds
            "webcam_interval": 10,      # seconds
            "activity_threshold": 0.6,  # confidence threshold
            "idle_timeout": 300,        # 5 minutes
            "validation_interval": 300, # 5 minutes
            "screen_recording": True,
            "webcam_monitoring": True,
            "keyboard_tracking": True,
            "mouse_tracking": True
        }
    
    def _setup_database(self):
        """Initialize SQLite database for tracking data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS work_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                developer_id TEXT NOT NULL,
                branch_name TEXT NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                activity_type TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                screen_activity BOOLEAN,
                face_detected BOOLEAN,
                keyboard_activity BOOLEAN,
                mouse_activity BOOLEAN,
                validation_status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS known_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                developer_id TEXT UNIQUE NOT NULL,
                face_encoding BLOB NOT NULL,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp TIMESTAMP,
                activity_type TEXT,
                confidence_score REAL,
                screenshot_path TEXT,
                webcam_image_path TEXT,
                FOREIGN KEY (session_id) REFERENCES work_sessions (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.repo_path, 'tracker.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AgenticTracker')
    
    def _load_known_faces(self):
        """Load known face encodings from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT developer_id, face_encoding, name FROM known_faces')
        
        for row in cursor.fetchall():
            developer_id, face_encoding, name = row
            self.known_faces[developer_id] = {
                'encoding': np.frombuffer(face_encoding, dtype=np.float64),
                'name': name
            }
        
        conn.close()
    
    def register_developer(self, developer_id: str, name: str, face_image_path: str):
        """Register a new developer with their face encoding"""
        # Load and encode face
        image = face_recognition.load_image_file(face_image_path)
        face_encodings = face_recognition.face_encodings(image)
        
        if not face_encodings:
            raise ValueError("No face detected in the provided image")
        
        face_encoding = face_encodings[0]
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO known_faces (developer_id, face_encoding, name) VALUES (?, ?, ?)',
            (developer_id, face_encoding.tobytes(), name)
        )
        conn.commit()
        conn.close()
        
        # Update in-memory cache
        self.known_faces[developer_id] = {
            'encoding': face_encoding,
            'name': name
        }
        
        self.logger.info(f"Registered developer: {name} ({developer_id})")
    
    def start_tracking(self, developer_id: str, branch_name: str):
        """Start tracking a developer's work session"""
        if self.is_tracking:
            self.logger.warning("Tracking already in progress")
            return
        
        self.current_session = WorkSession(
            developer_id=developer_id,
            branch_name=branch_name,
            start_time=datetime.now(),
            end_time=None,
            activity_type=ActivityType.UNKNOWN,
            confidence_score=0.0,
            screen_activity=False,
            face_detected=False,
            keyboard_activity=False,
            mouse_activity=False
        )
        
        self.is_tracking = True
        self.tracking_thread = threading.Thread(target=self._tracking_loop)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
        
        self.logger.info(f"Started tracking {developer_id} on branch {branch_name}")
    
    def stop_tracking(self):
        """Stop the current tracking session"""
        if not self.is_tracking:
            return
        
        self.is_tracking = False
        if self.tracking_thread:
            self.tracking_thread.join()
        
        if self.current_session:
            self.current_session.end_time = datetime.now()
            self._save_session()
        
        self.logger.info("Stopped tracking session")
    
    def _tracking_loop(self):
        """Main tracking loop that runs in background thread"""
        last_screenshot = 0
        last_webcam = 0
        last_validation = 0
        
        while self.is_tracking:
            current_time = time.time()
            
            # Take screenshots periodically
            if current_time - last_screenshot >= self.config["screenshot_interval"]:
                self._capture_screen_activity()
                last_screenshot = current_time
            
            # Capture webcam images
            if current_time - last_webcam >= self.config["webcam_interval"]:
                self._capture_webcam_activity()
                last_webcam = current_time
            
            # Validate session periodically
            if current_time - last_validation >= self.config["validation_interval"]:
                self._validate_session()
                last_validation = current_time
            
            time.sleep(1)
    
    def _capture_screen_activity(self):
        """Capture and analyze screen activity"""
        try:
            # Take screenshot
            screenshot = pyautogui.screenshot()
            screenshot_path = self._save_screenshot(screenshot)
            
            # Analyze screen content for coding activity
            activity_score = self._analyze_screen_content(screenshot)
            
            # Update session
            if self.current_session:
                self.current_session.screen_activity = activity_score > 0.3
                self.current_session.confidence_score = max(
                    self.current_session.confidence_score, 
                    activity_score
                )
                
                # Determine activity type
                if activity_score > 0.7:
                    self.current_session.activity_type = ActivityType.CODING
                elif activity_score > 0.4:
                    self.current_session.activity_type = ActivityType.MEETING
                else:
                    self.current_session.activity_type = ActivityType.IDLE
            
            self.logger.debug(f"Screen activity captured: {activity_score:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error capturing screen activity: {e}")
    
    def _capture_webcam_activity(self):
        """Capture and analyze webcam activity"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.logger.warning("Could not open webcam")
                return
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return
            
            # Save webcam image
            webcam_path = self._save_webcam_image(frame)
            
            # Detect faces
            face_detected = self._detect_faces(frame)
            
            # Identify developer if face detected
            if face_detected and self.known_faces:
                developer_identified = self._identify_developer(frame)
                if developer_identified:
                    self.logger.info(f"Developer identified: {developer_identified}")
            
            # Update session
            if self.current_session:
                self.current_session.face_detected = face_detected
            
            self.logger.debug(f"Webcam activity captured: face_detected={face_detected}")
            
        except Exception as e:
            self.logger.error(f"Error capturing webcam activity: {e}")
    
    def _detect_faces(self, frame) -> bool:
        """Detect faces in webcam frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        return len(results.detections) > 0 if results.detections else False
    
    def _identify_developer(self, frame) -> Optional[str]:
        """Identify developer from face recognition"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for face_encoding in face_encodings:
            for developer_id, face_data in self.known_faces.items():
                matches = face_recognition.compare_faces(
                    [face_data['encoding']], face_encoding, tolerance=0.6
                )
                if matches[0]:
                    return developer_id
        
        return None
    
    def _analyze_screen_content(self, screenshot) -> float:
        """Analyze screenshot for coding activity"""
        # Convert to numpy array
        img_array = np.array(screenshot)
        
        # Simple heuristics for coding activity
        # This could be enhanced with OCR and more sophisticated analysis
        
        # Check for common IDE/editor indicators
        # (This is a simplified version - in practice, you'd use OCR)
        activity_score = 0.0
        
        # Check if mouse/keyboard are active
        if self._check_input_activity():
            activity_score += 0.3
        
        # Check for active windows (simplified)
        if self._check_active_windows():
            activity_score += 0.4
        
        return min(activity_score, 1.0)
    
    def _check_input_activity(self) -> bool:
        """Check for keyboard and mouse activity"""
        # This is a simplified check - in practice, you'd use more sophisticated methods
        return True  # Placeholder
    
    def _check_active_windows(self) -> bool:
        """Check for active development windows"""
        # This is a simplified check - in practice, you'd analyze window titles
        return True  # Placeholder
    
    def _save_screenshot(self, screenshot) -> str:
        """Save screenshot to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        filepath = os.path.join(self.repo_path, "tracking_data", "screenshots", filename)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        screenshot.save(filepath)
        
        return filepath
    
    def _save_webcam_image(self, frame) -> str:
        """Save webcam image to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"webcam_{timestamp}.png"
        filepath = os.path.join(self.repo_path, "tracking_data", "webcam", filename)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        cv2.imwrite(filepath, frame)
        
        return filepath
    
    def _validate_session(self):
        """AI-powered session validation"""
        if not self.current_session:
            return
        
        # Calculate validation score based on multiple factors
        validation_score = 0.0
        validation_factors = []
        
        # Face detection factor
        if self.current_session.face_detected:
            validation_score += 0.3
            validation_factors.append("face_detected")
        
        # Screen activity factor
        if self.current_session.screen_activity:
            validation_score += 0.3
            validation_factors.append("screen_activity")
        
        # Activity type factor
        if self.current_session.activity_type == ActivityType.CODING:
            validation_score += 0.4
            validation_factors.append("coding_activity")
        
        # Set validation status
        if validation_score >= 0.7:
            self.current_session.validation_status = "validated"
        elif validation_score >= 0.4:
            self.current_session.validation_status = "partial"
        else:
            self.current_session.validation_status = "invalid"
        
        self.logger.info(f"Session validation: {self.current_session.validation_status} "
                        f"(score: {validation_score:.2f}, factors: {validation_factors})")
    
    def _save_session(self):
        """Save current session to database"""
        if not self.current_session:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO work_sessions 
            (developer_id, branch_name, start_time, end_time, activity_type, 
             confidence_score, screen_activity, face_detected, keyboard_activity, 
             mouse_activity, validation_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.current_session.developer_id,
            self.current_session.branch_name,
            self.current_session.start_time,
            self.current_session.end_time,
            self.current_session.activity_type.value,
            self.current_session.confidence_score,
            self.current_session.screen_activity,
            self.current_session.face_detected,
            self.current_session.keyboard_activity,
            self.current_session.mouse_activity,
            self.current_session.validation_status
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Saved session for {self.current_session.developer_id}")
    
    def get_developer_stats(self, developer_id: str, days: int = 7) -> Dict:
        """Get statistics for a developer"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT 
                branch_name,
                SUM(strftime('%s', end_time) - strftime('%s', start_time)) as total_seconds,
                COUNT(*) as session_count,
                AVG(confidence_score) as avg_confidence,
                validation_status
            FROM work_sessions 
            WHERE developer_id = ? AND start_time >= ?
            GROUP BY branch_name, validation_status
        ''', (developer_id, start_date))
        
        results = cursor.fetchall()
        conn.close()
        
        stats = {
            'developer_id': developer_id,
            'period_days': days,
            'total_hours': sum(row[1] or 0 for row in results) / 3600,
            'sessions': len(results),
            'branches': list(set(row[0] for row in results)),
            'avg_confidence': sum(row[3] or 0 for row in results) / len(results) if results else 0,
            'validation_rate': len([r for r in results if r[4] == 'validated']) / len(results) if results else 0
        }
        
        return stats
    
    def get_project_estimates(self, branch_name: str) -> Dict:
        """Get project estimates based on tracked data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                developer_id,
                SUM(strftime('%s', end_time) - strftime('%s', start_time)) as total_seconds,
                AVG(confidence_score) as avg_confidence
            FROM work_sessions 
            WHERE branch_name = ? AND validation_status = 'validated'
            GROUP BY developer_id
        ''', (branch_name,))
        
        results = cursor.fetchall()
        conn.close()
        
        total_hours = sum(row[1] or 0 for row in results) / 3600
        avg_confidence = sum(row[2] or 0 for row in results) / len(results) if results else 0
        
        # Simple estimation model (could be enhanced with ML)
        estimated_completion_hours = total_hours * (1.0 / avg_confidence) if avg_confidence > 0 else 0
        
        return {
            'branch_name': branch_name,
            'total_hours_spent': total_hours,
            'avg_confidence': avg_confidence,
            'estimated_completion_hours': estimated_completion_hours,
            'developers_working': len(results),
            'data_quality': 'high' if avg_confidence > 0.7 else 'medium' if avg_confidence > 0.4 else 'low'
        } 