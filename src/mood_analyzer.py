import numpy as np
from textblob import TextBlob

class MoodAnalyzer:
    def __init__(self):
        self.mood_mappings = {
            'happy': {
                'tempo_range': (120, 160),
                'key_preference': ['C major', 'G major', 'D major'],
                'rhythm_complexity': 0.7,
                'chord_brightness': 0.8
            },
            'sad': {
                'tempo_range': (60, 90),
                'key_preference': ['A minor', 'D minor', 'E minor'],
                'rhythm_complexity': 0.3,
                'chord_brightness': 0.2
            },
            'energetic': {
                'tempo_range': (140, 180),
                'key_preference': ['E major', 'A major', 'B major'],
                'rhythm_complexity': 0.9,
                'chord_brightness': 0.9
            },
            'calm': {
                'tempo_range': (70, 100),
                'key_preference': ['F major', 'Bb major', 'G minor'],
                'rhythm_complexity': 0.4,
                'chord_brightness': 0.5
            }
        }

    def analyze_text(self, text):
        """Analyze text input to determine musical parameters."""
        blob = TextBlob(text)
        
        # Get sentiment polarity (-1 to 1) and subjectivity (0 to 1)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine base mood
        if polarity > 0.3:
            base_mood = 'happy' if subjectivity < 0.5 else 'energetic'
        elif polarity < -0.3:
            base_mood = 'sad'
        else:
            base_mood = 'calm'
            
        return self.generate_music_parameters(base_mood, polarity, subjectivity)

    def generate_music_parameters(self, base_mood, polarity, subjectivity):
        """Generate specific music parameters based on mood analysis."""
        params = {}
        base_params = self.mood_mappings[base_mood]
        
        # Calculate tempo
        tempo_min, tempo_max = base_params['tempo_range']
        tempo_range = tempo_max - tempo_min
        intensity = (polarity + 1) / 2  # Convert to 0-1 range
        params['tempo'] = int(tempo_min + (tempo_range * intensity))
        
        # Select key
        params['key'] = np.random.choice(base_params['key_preference'])
        
        # Calculate rhythm complexity
        params['rhythm_complexity'] = base_params['rhythm_complexity'] * (1 + subjectivity) / 2
        
        # Calculate chord complexity
        params['chord_brightness'] = base_params['chord_brightness'] * (1 + polarity) / 2
        
        # Additional parameters
        params['volume_dynamics'] = 0.4 + (subjectivity * 0.6)
        params['note_density'] = 0.3 + (abs(polarity) * 0.7)
        
        return params

    def get_mood_description(self, params):
        """Generate human-readable description of the musical mood."""
        descriptions = []
        
        if params['tempo'] > 140:
            descriptions.append('upbeat')
        elif params['tempo'] < 90:
            descriptions.append('relaxed')
        
        if params['chord_brightness'] > 0.7:
            descriptions.append('bright')
        elif params['chord_brightness'] < 0.3:
            descriptions.append('dark')
            
        if params['rhythm_complexity'] > 0.7:
            descriptions.append('complex')
        elif params['rhythm_complexity'] < 0.3:
            descriptions.append('simple')
            
        return ' and '.join(descriptions) + ' musical atmosphere'