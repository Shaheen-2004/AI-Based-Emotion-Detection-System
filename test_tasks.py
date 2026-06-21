try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    print("Tasks API imports successful.")
    
    FaceLandmarker = vision.FaceLandmarker
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions
    BaseOptions = python.BaseOptions
    
    print("FaceLandmarker classes available.")
    
except Exception as e:
    print(f"Tasks API failed: {e}")
