import numpy as np
import math
from collections import deque, Counter

class EmotionModel:
    """
    Infers emotion from face landmarks.
    Simplified "Gesture" Mode for easy activation.
    """
    def __init__(self, mode='heuristic'):
        self.mode = mode
        self.emotions = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Angry']
        
        # Reduced smoothing for faster response to gestures
        self.history_len = 5
        self.history = deque(maxlen=self.history_len)

    def predict(self, landmarks):
        """
        Predict emotion from NormalizedLandmarkList.
        """
        if not landmarks:
            return "Neutral"
            
        # 1. Get raw frame prediction
        if self.mode == 'heuristic':
            raw_label = self._heuristic_predict(landmarks)
        else:
            raw_label = "Neutral"

        # 2. Smooth prediction
        self.history.append(raw_label)
        
        # Effective Smoothing: Voting
        counts = Counter(self.history)
        smoothed_label = counts.most_common(1)[0][0]
        
        return smoothed_label

    def _dist(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def _heuristic_predict(self, lm):
        """
        Simplified geometric heuristics.
        Designed for easy "caricature" gestures.
        """
        # Landmarks:
        # Lips: 13 (upper inner), 14 (lower inner)
        # Mouth Corners: 61 (left), 291 (right)
        # Eye Corners (Outer): 33 (left), 263 (right) - used for scaling
        # Eyebrows Inner: 66 (left), 296 (right)
        # Eyebrows Outer: 105 (left), 334 (right)
        # Eye Center/Pupil approx: 159 (left upper lid), 386 (right upper lid)
        
        # 0. Normalization Scale (Inter-ocular distance)
        face_scale = self._dist(lm[33], lm[263])
        if face_scale == 0: face_scale = 0.01
        
        # 1. Mouth Metrics
        mouth_open = self._dist(lm[13], lm[14]) / face_scale
        mouth_width = self._dist(lm[61], lm[291]) / face_scale
        
        # Frown/Smile Curve
        avg_corner_y = (lm[61].y + lm[291].y) / 2
        lip_center_y = lm[13].y
        
        # Positive if corners are HIGHER (smaller y) than upper lip (smile)
        smile_curve = lip_center_y - avg_corner_y 
        
        # Positive if corners are LOWER (larger y) than lower lip approx (frown)
        # Using lm[14] (lower lip) as reference for frown might be too strict.
        # Let's use the mid-point of the mouth or upper lip.
        # Simple Frown: Corners are significantly lower than 0-level of mouth.
        # Let's use lip_center_y (upper lip) as the anchor.
        # If avg_corner_y is > lip_center_y, corners are down.
        # "Frown Strength" = avg_corner_y - lip_center_y
        frown_strength = avg_corner_y - lip_center_y
        
        # 2. Eyebrow Metrics
        # Average Brow Height (Dist from eye top to brow)
        l_brow_dist = self._dist(lm[66], lm[159]) / face_scale
        r_brow_dist = self._dist(lm[296], lm[386]) / face_scale
        avg_brow_lift = (l_brow_dist + r_brow_dist) / 2
        
        # --- Simpler Decision Tree ---
        
        # 1. Surprise: Mouth Open
        if mouth_open > 0.3: 
            return "Surprise"
            
        # 2. Happy: Wide Mouth OR Smile Curve
        # Relaxed threshold for easy activation
        if mouth_width > 0.5 or smile_curve > 0.01: 
            return "Happy"

        # 3. Angry: Eyebrows LOW
        # Removed "brow squeeze" requirement. Just look "intense" or lower brows.
        # Normal lift is ~0.15-0.20. Low is < 0.12 or so.
        # Let's make it easy: < 0.14
        if avg_brow_lift < 0.14:
            return "Angry"
            
        # 4. Sad: Mouth Corners Down (Frown)
        # If corners are clearly below the upper lip level.
        # Threshold 0.02 means corners are slightly down.
        if frown_strength > 0.02: 
             return "Sad"
             
        # Fallback
        return "Neutral"
