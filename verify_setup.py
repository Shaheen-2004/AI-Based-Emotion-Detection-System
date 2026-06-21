try:
    from face_detector import FaceDetector
    from emotion_model import EmotionModel
    from overlay_utils import OverlayUtils
    import cv2
    import mediapipe
    
    print("Imports success.")
    
    fd = FaceDetector()
    print("FaceDetector initialized.")
    
    em = EmotionModel()
    print("EmotionModel initialized.")
    
    ou = OverlayUtils()
    print("OverlayUtils initialized.")
    
    print("Verification complete.")
except Exception as e:
    print(f"Verification failed: {e}")
