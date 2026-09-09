"""
generate_demo_photo.py
----------------------
Generates a realistic 60-student classroom auditorium group photo (1920x1080)
with 60 distinct face targets arranged in tiered rows (6 rows x 10 students).
"""

import os
import math
import numpy as np
from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_PHOTO = os.path.join(DATA_DIR, "demo_classroom_60.jpg")

os.makedirs(DATA_DIR, exist_ok=True)

def generate_classroom_photo():
    width = 1920
    height = 1080
    
    # Background: modern classroom / auditorium hall background
    img = Image.new("RGB", (width, height), color=(240, 243, 248))
    draw = ImageDraw.Draw(img)
    
    # Draw soft classroom wall gradient & floor
    draw.rectangle([0, 0, width, 700], fill=(225, 232, 240))  # wall
    draw.rectangle([0, 700, width, height], fill=(180, 190, 205))  # floor
    
    # Draw auditorium desks/rows (6 tiered rows x 10 cols = 60 students)
    rows = 6
    cols = 10
    
    start_y = 180
    row_gap = 135
    col_gap = 165
    margin_x = 220
    
    skin_tones = [
        (255, 224, 189), (241, 194, 125), (255, 205, 148), (224, 172, 105),
        (198, 134, 66), (141, 85, 36), (112, 65, 20), (255, 219, 172)
    ]
    hair_colors = [
        (20, 20, 20), (75, 45, 20), (160, 100, 40), (210, 160, 70), (40, 30, 20)
    ]
    shirt_colors = [
        (79, 70, 229), (16, 185, 129), (245, 158, 11), (239, 68, 68), (14, 165, 233),
        (139, 92, 246), (236, 72, 153), (107, 114, 128), (59, 130, 246), (5, 150, 105)
    ]
    
    for r in range(rows):
        row_y = start_y + r * row_gap
        scale = 0.75 + (r * 0.06)  # perspective scale
        face_radius = int(32 * scale)
        
        # Desks row line
        draw.line([margin_x - 40, row_y + face_radius + 60, width - margin_x + 40, row_y + face_radius + 60], fill=(140, 150, 170), width=4)
        
        for c in range(cols):
            idx = r * cols + c + 1
            cx = margin_x + c * col_gap + (10 if r % 2 == 1 else 0)
            cy = row_y
            
            skin = skin_tones[idx % len(skin_tones)]
            hair = hair_colors[idx % len(hair_colors)]
            shirt = shirt_colors[idx % len(shirt_colors)]
            
            # Draw Shoulders / Shirt
            draw.ellipse([cx - int(face_radius * 1.8), cy + int(face_radius * 0.8),
                          cx + int(face_radius * 1.8), cy + int(face_radius * 2.8)], fill=shirt)
            
            # Draw Neck
            draw.rectangle([cx - int(face_radius * 0.3), cy + int(face_radius * 0.7),
                            cx + int(face_radius * 0.3), cy + int(face_radius * 1.2)], fill=skin)
            
            # Draw Hair (back)
            draw.ellipse([cx - face_radius - 4, cy - face_radius - 8,
                          cx + face_radius + 4, cy + face_radius], fill=hair)
            
            # Draw Face Head
            draw.ellipse([cx - face_radius, cy - face_radius,
                          cx + face_radius, cy + face_radius + 4], fill=skin)
            
            # Draw Hair (top/bangs)
            draw.arc([cx - face_radius - 2, cy - face_radius - 6,
                      cx + face_radius + 2, cy], start=180, end=360, fill=hair, width=int(face_radius * 0.5))
            
            # Draw Eyes
            eye_off_x = int(face_radius * 0.35)
            eye_y = cy - int(face_radius * 0.1)
            eye_r = max(2, int(face_radius * 0.12))
            draw.ellipse([cx - eye_off_x - eye_r, eye_y - eye_r, cx - eye_off_x + eye_r, eye_y + eye_r], fill=(30, 30, 30))
            draw.ellipse([cx + eye_off_x - eye_r, eye_y - eye_r, cx + eye_off_x + eye_r, eye_y + eye_r], fill=(30, 30, 30))
            
            # Draw Smile
            mouth_y = cy + int(face_radius * 0.4)
            draw.arc([cx - int(face_radius * 0.3), mouth_y - 3, cx + int(face_radius * 0.3), mouth_y + 6], start=0, end=180, fill=(180, 50, 50), width=2)
            
    img.save(OUTPUT_PHOTO, "JPEG", quality=95)
    print(f"Generated 60-student demo group photo at {OUTPUT_PHOTO}")
    return OUTPUT_PHOTO

if __name__ == "__main__":
    generate_classroom_photo()
