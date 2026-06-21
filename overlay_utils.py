import cv2
import numpy as np
import time
import os
import random
from datetime import datetime

class OverlayUtils:
    """
    Handles all the visual overlays: HUD, Neon effects, Text, etc.
    """
    def __init__(self):
        self.screenshot_dir = "screenshots"
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)
            
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # FPS calculation
        self.prev_time = 0
        self.curr_time = 0
        self.fps = 0

    def draw_neon_rect(self, image, bbox, color=(0, 255, 0), thickness=2):
        """
        Draws a rectangle with a neon glow effect.
        bbox: (x, y, w, h)
        """
        x, y, w, h = bbox
        
        # Draw the main rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        
        # Draw glow effect (multiple blurred rectangles) for a "cyberpunk" feel
        # Note: Blurring the whole image is too slow, so we just draw thicker, semi-transparent lines if performance allows.
        # For a simple glow, we can overlay a blurred copy, but let's stick to efficient layered drawing.
        
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness + 4)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    def draw_text_with_bg(self, image, text, pos, font_scale=0.6, color=(255, 255, 255), bg_color=(0, 0, 0)):
        """
        Draws text with a background box for better visibility.
        """
        x, y = pos
        (text_w, text_h), baseline = cv2.getTextSize(text, self.font, font_scale, 1)
        
        # Background rectangle
        cv2.rectangle(image, (x, y - text_h - 4), (x + text_w, y + baseline), bg_color, -1)
        
        # Text
        cv2.putText(image, text, (x, y), self.font, font_scale, color, 1, cv2.LINE_AA)

    def draw_hud(self, image, emotion_label, confidence=None, mode="Heuristic"):
        """
        Draws the main HUD elements: FPS, Status, Scanlines.
        """
        h, w = image.shape[:2]
        
        # 1. Scanlines (simple horizontal lines)
        # We can implement a moving scanline if needed, but static for now for performance
        # or a simple subtle grid
        
        # 2. FPS Counter
        self.curr_time = time.time()
        fps_val = 1 / (self.curr_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = self.curr_time
        # Smoothing FPS display slightly
        self.fps = 0.9 * self.fps + 0.1 * fps_val
        
        self.draw_text_with_bg(image, f"FPS: {int(self.fps)}", (20, 40), font_scale=0.7, color=(0, 255, 0))
        
        # 3. Status Panel
        status_text = f"Mode: {mode}"
        self.draw_text_with_bg(image, status_text, (20, h - 30), font_scale=0.6, color=(0, 255, 255))
        
        # 4. Glitch Text / Header
        # Randomly offset the header slightly for a glitch effect
        header_x = w // 2 - 100
        if random.random() < 0.05:
            header_x += random.randint(-5, 5)
            
        cv2.putText(image, "NOVA INTERFACE //", (header_x, 40), self.font, 0.8, (255, 0, 255), 2, cv2.LINE_AA)
        

    def save_screenshot(self, image):
        """
        Saves the current frame to the screenshots directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.screenshot_dir}/screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, image)
        print(f"Screenshot saved: {filename}")
        
        # Optional: play sound (Windows only as per original project reqs)
        try:
            import winsound
            winsound.PlaySound("SystemShutter", winsound.SND_ALIAS)
        except:
            pass
            
        return filename
