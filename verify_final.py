import cv2
import numpy as np
import os
from face_detector import FaceDetector

try:
    print("Initializing FaceDetector...")
    if not os.path.exists('assets/face_landmarker.task'):
        raise FileNotFoundError("Model file not found in assets/")
        
    fd = FaceDetector()
    print("FaceDetector initialized successfully.")
    
    # Create dummy black image
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    print("Testing detection on dummy image...")
    result = fd.detect(dummy_img)
    print("Detection call successful (result likely empty for black image).")
    
    print("Verification Passed!")
except Exception as e:
    print(f"Verification Failed: {e}")
