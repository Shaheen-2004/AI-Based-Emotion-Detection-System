import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceDetector:
    """
    Wrapper around MediaPipe Face Mesh (Tasks API) for face detection and landmark extraction.
    """
    def __init__(self, 
                 model_path='assets/face_landmarker.task',
                 num_faces=1, 
                 min_face_detection_confidence=0.5, 
                 min_face_presence_confidence=0.5,
                 min_tracking_confidence=0.5):
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence)
            
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, image):
        """
        Processes an image (BGR) and returns the face mesh results.
        
        Args:
            image: A BGR image/frame from OpenCV.
            
        Returns:
            The results object from FaceLandmarker. 
            Contains `face_landmarks` (list of list of NormalizedLandmark) if faces are detected.
        """
        if image is None:
            return None
        
        # Tasks API requires mp.Image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        try:
            detection_result = self.detector.detect(mp_image)
            return detection_result
        except Exception as e:
            # Handle empty frames or errors
            # print(f"Detection error: {e}")
            return None
