import cv2
import numpy as np
from typing import Tuple, Optional

class MoodDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Pre-trained emotion detection model path would go here
        self.emotions = ['angry', 'happy', 'sad', 'neutral', 'surprised']
        
    def detect_mood_from_image(self, image_path: str) -> Tuple[str, float]:
        """
        Analyzes an image to detect the dominant mood/emotion.
        Returns tuple of (mood, confidence_score)
        """
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f'Could not load image from {image_path}')
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return ('neutral', 0.0)
                
            # Process the first detected face
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Normalize and prepare face for emotion detection
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi / 255.0
            
            # Here we would normally feed the processed face into an emotion detection model
            # For demo purposes, returning mock results
            mock_mood = self._mock_emotion_detection(face_roi)
            return mock_mood
            
        except Exception as e:
            print(f'Error detecting mood: {str(e)}')
            return ('neutral', 0.0)
            
    def detect_mood_from_webcam(self) -> Optional[Tuple[str, float]]:
        """
        Captures and analyzes mood from webcam feed in real-time.
        Returns tuple of (mood, confidence_score)
        """
        cap = cv2.VideoCapture(0)
        
        try:
            ret, frame = cap.read()
            if not ret:
                return None
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return ('neutral', 0.0)
                
            # Process the first detected face
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Normalize and prepare face for emotion detection
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi / 255.0
            
            mock_mood = self._mock_emotion_detection(face_roi)
            return mock_mood
            
        except Exception as e:
            print(f'Error detecting mood from webcam: {str(e)}')
            return None
            
        finally:
            cap.release()
            
    def _mock_emotion_detection(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """
        Mock emotion detection - to be replaced with actual ML model.
        """
        # Simulate emotion detection with random selection
        confidence = np.random.random()
        mood = np.random.choice(self.emotions)
        return (mood, confidence)

if __name__ == '__main__':
    detector = MoodDetector()
    
    # Test image detection
    result = detector.detect_mood_from_image('test.jpg')
    print(f'Detected mood from image: {result}')
    
    # Test webcam detection
    result = detector.detect_mood_from_webcam()
    print(f'Detected mood from webcam: {result}')