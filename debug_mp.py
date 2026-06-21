import sys
try:
    import mediapipe
    print(f"MediaPipe found at: {mediapipe.__file__}")
    print(f"Dir: {dir(mediapipe)}")
    import mediapipe.python.solutions as solutions
    print("Direct import of solutions worked.")
except Exception as e:
    print(f"Error: {e}")

try:
    import google.protobuf
    print(f"Protobuf version: {google.protobuf.__version__}")
except Exception as e:
    print(f"Protobuf error: {e}")
