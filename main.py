import cv2
import mediapipe as mp
import numpy as np
import sys

# Import our modules
from face_detector import FaceDetector
from emotion_model import EmotionModel
from overlay_utils import OverlayUtils

def main():
    # Initialize modules
    detector = FaceDetector()
    emotion_model = EmotionModel()
    overlay = OverlayUtils()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Face Emotion & Persona Overlay...")
    print("Press 'q' or 'ESC' to quit.")
    print("Press 's' to save a screenshot.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Detect faces
        results = detector.detect(frame)
        
        # Default status
        current_emotion = "Neutral"

        if results and results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                # 1. Predict Emotion
                # face_landmarks is now a list of NormalizedLandmark objects directly
                current_emotion = emotion_model.predict(face_landmarks)
                
                # 2. Calculate Bounding Box from landmarks
                x_min, x_max = w, 0
                y_min, y_max = h, 0
                
                # Iterating directly over the list of landmarks
                for lm in face_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x < x_min: x_min = x
                    if x > x_max: x_max = x
                    if y < y_min: y_min = y
                    if y > y_max: y_max = y
                
                # Add some padding
                pad = 20
                x_min = max(0, x_min - pad)
                y_min = max(0, y_min - pad)
                x_max = min(w, x_max + pad)
                y_max = min(h, y_max + pad)
                
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                
                # 3. Draw Overlay
                # Color based on emotion?
                color = (0, 255, 0) # Green default
                if current_emotion == "Happy": color = (0, 255, 255) # Yellow
                elif current_emotion == "Surprise": color = (255, 0, 255) # Magenta
                
                overlay.draw_neon_rect(frame, bbox, color=color)
                
                # Draw Label above box
                overlay.draw_text_with_bg(frame, f"STATUS: {current_emotion.upper()}", 
                                        (x_min, y_min - 10), 
                                        color=color)

        # Draw Global HUD
        overlay.draw_hud(frame, current_emotion)

        # Display
        cv2.imshow('Cyberpunk Persona HUD', frame)

        # Controls
        key = cv2.waitKey(5) & 0xFF
        if key == 27 or key == ord('q'): # ESC or q
            break
        elif key == ord('s'):
            overlay.save_screenshot(frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
