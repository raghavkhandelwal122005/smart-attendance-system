"""
generate_demo_photo.py
----------------------
Generates a 1920x1080 demo classroom photo with 60 drawn student faces
arranged in 6 tiered rows × 10 columns.

OUTPUT_PHOTO defaults to backend/data/demo_classroom_60.jpg locally,
or to a caller-supplied path (e.g. /tmp/demo_classroom_60.jpg on Vercel).
"""

import os
import math
import numpy as np
from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(__file__)
_DEFAULT_OUTPUT = os.path.join(BASE_DIR, "data", "demo_classroom_60.jpg")


def generate_classroom_photo(output_path: str = _DEFAULT_OUTPUT) -> str:
    """
    Generate the demo classroom photo and save to output_path.
    Creates parent directories as needed (safe for /tmp on Vercel).
    Returns the final path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    width, height = 1920, 1080

    img = Image.new("RGB", (width, height), color=(240, 243, 248))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, width, 700], fill=(225, 232, 240))   # wall
    draw.rectangle([0, 700, width, height], fill=(180, 190, 205))  # floor

    rows, cols = 6, 10
    start_y, row_gap, col_gap, margin_x = 180, 135, 165, 220

    skin_tones  = [(255,224,189),(241,194,125),(255,205,148),(224,172,105),
                   (198,134,66),(141,85,36),(112,65,20),(255,219,172)]
    hair_colors = [(20,20,20),(75,45,20),(160,100,40),(210,160,70),(40,30,20)]
    shirt_colors = [(79,70,229),(16,185,129),(245,158,11),(239,68,68),(14,165,233),
                    (139,92,246),(236,72,153),(107,114,128),(59,130,246),(5,150,105)]

    for r in range(rows):
        row_y = start_y + r * row_gap
        scale = 0.75 + r * 0.06
        fr = int(32 * scale)

        draw.line(
            [margin_x - 40, row_y + fr + 60, width - margin_x + 40, row_y + fr + 60],
            fill=(140, 150, 170), width=4,
        )

        for c in range(cols):
            idx = r * cols + c + 1
            cx = margin_x + c * col_gap + (10 if r % 2 == 1 else 0)
            cy = row_y

            skin  = skin_tones[idx % len(skin_tones)]
            hair  = hair_colors[idx % len(hair_colors)]
            shirt = shirt_colors[idx % len(shirt_colors)]

            draw.ellipse([cx - int(fr*1.8), cy + int(fr*0.8),
                          cx + int(fr*1.8), cy + int(fr*2.8)], fill=shirt)
            draw.rectangle([cx - int(fr*0.3), cy + int(fr*0.7),
                             cx + int(fr*0.3), cy + int(fr*1.2)], fill=skin)
            draw.ellipse([cx-fr-4, cy-fr-8, cx+fr+4, cy+fr], fill=hair)
            draw.ellipse([cx-fr, cy-fr, cx+fr, cy+fr+4], fill=skin)
            draw.arc([cx-fr-2, cy-fr-6, cx+fr+2, cy],
                     start=180, end=360, fill=hair, width=int(fr*0.5))

            eox = int(fr * 0.35)
            ey  = cy - int(fr * 0.1)
            er  = max(2, int(fr * 0.12))
            draw.ellipse([cx-eox-er, ey-er, cx-eox+er, ey+er], fill=(30,30,30))
            draw.ellipse([cx+eox-er, ey-er, cx+eox+er, ey+er], fill=(30,30,30))

            my = cy + int(fr * 0.4)
            draw.arc([cx-int(fr*0.3), my-3, cx+int(fr*0.3), my+6],
                     start=0, end=180, fill=(180,50,50), width=2)

    img.save(output_path, "JPEG", quality=95)
    print(f"[DEMO] Generated 60-student demo photo -> {output_path}")
    return output_path


if __name__ == "__main__":
    generate_classroom_photo()
