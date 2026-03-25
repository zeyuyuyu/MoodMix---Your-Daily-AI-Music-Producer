import cv2
import numpy as np
from transformers import pipeline
from typing import Dict, Union, Tuple

class MoodDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.sentiment_analyzer = pipeline('sentiment-analysis')
        
        # Emotion mapping to musical characteristics
        self.emotion_music_map = {
            'happy': {'tempo': 'upbeat', 'mode': 'major', 'energy': 'high'},
            'sad': {'tempo': 'slow', 'mode': 'minor', 'energy': 'low'},
            'neutral': {'tempo': 'moderate', 'mode': 'major', 'energy': 'medium'},
            'angry': {'tempo': 'fast', 'mode': 'minor', 'energy': 'high'}
        }

    def detect_facial_emotion(self, image: np.ndarray) -> Dict[str, float]:
        """Detect emotion from facial expression in image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return {'neutral': 1.0}
            
        # For demo, using simple pixel intensity as proxy for emotion
        face_x, face_y, face_w, face_h = faces[0]
        face_roi = gray[face_y:face_y+face_h, face_x:face_x+face_w]
        avg_intensity = np.mean(face_roi)
        
        # Simple mapping of intensity to emotions
        if avg_intensity > 150:
            return {'happy': 0.8, 'neutral': 0.2}
        elif avg_intensity < 100:
            return {'sad': 0.7, 'neutral': 0.3}
        else:
            return {'neutral': 0.6, 'happy': 0.2, 'sad': 0.2}

    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze emotion from text input."""
        result = self.sentiment_analyzer(text)
        sentiment = result[0]
        
        if sentiment['label'] == 'POSITIVE':
            return {'happy': sentiment['score'], 'neutral': 1 - sentiment['score']}
        else:
            return {'sad': sentiment['score'], 'neutral': 1 - sentiment['score']}

    def combine_emotions(self, facial: Dict[str, float], textual: Dict[str, float]) -> Dict[str, float]:
        """Combine emotions from different modalities with weights."""
        combined = {}
        # Weight facial expressions more heavily (0.6) than text (0.4)
        for emotion, score in facial.items():
            combined[emotion] = score * 0.6
        for emotion, score in textual.items():
            if emotion in combined:
                combined[emotion] += score * 0.4
            else:
                combined[emotion] = score * 0.4
        return combined

    def get_musical_parameters(self, emotions: Dict[str, float]) -> Dict[str, str]:
        """Convert emotions to musical parameters."""
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        return self.emotion_music_map.get(dominant_emotion, self.emotion_music_map['neutral'])

    def analyze_mood(self, image: np.ndarray, text: str = '') -> Tuple[Dict[str, float], Dict[str, str]]:
        """Main method to analyze mood from both image and text."""
        facial_emotions = self.detect_facial_emotion(image)
        text_emotions = self.analyze_text_sentiment(text) if text else {'neutral': 1.0}
        
        combined_emotions = self.combine_emotions(facial_emotions, text_emotions)
        musical_params = self.get_musical_parameters(combined_emotions)
        
        return combined_emotions, musical_params