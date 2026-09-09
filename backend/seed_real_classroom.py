"""
seed_real_classroom.py
----------------------
Detects real student faces from uploaded classroom photo (media_1788917238154.jpg),
crops individual face photos, extracts real 512-d ArcFace embeddings,
and registers student profiles (including Gwen ID: 5) in SQLite database.
"""

import os
import shutil
import uuid
import numpy as np
from PIL import Image

import database as db
import face_engine

BASE_DIR = os.path.dirname(__file__)
USER_UPLOAD_PHOTO = r"C:\Users\LENOVO\.gemini\antigravity\brain\25bec080-1232-4ce6-ada0-2f032d293735\.user_uploaded\media_1788917238154.jpg"
DATA_DIR = os.path.join(BASE_DIR, "data")
TARGET_DEMO_PHOTO = os.path.join(DATA_DIR, "demo_classroom_60.jpg")
STUDENT_PHOTO_DIR = os.path.join(DATA_DIR, "student_photos")


def seed_real_students():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(STUDENT_PHOTO_DIR, exist_ok=True)

    # 1. Copy user real classroom photo to demo_classroom_60.jpg
    if os.path.exists(USER_UPLOAD_PHOTO):
        shutil.copy(USER_UPLOAD_PHOTO, TARGET_DEMO_PHOTO)
        print(f"Copied real classroom photo to {TARGET_DEMO_PHOTO}")
    elif not os.path.exists(TARGET_DEMO_PHOTO):
        raise FileNotFoundError(f"Classroom photo not found at {USER_UPLOAD_PHOTO}")

    # 2. Load image and run InsightFace detector
    with open(TARGET_DEMO_PHOTO, "rb") as f:
        raw_bytes = f.read()

    image_rgb = face_engine.load_image_from_bytes(raw_bytes)
    # Detect all real student faces in classroom photo
    faces = face_engine.detect_faces(image_rgb, det_thresh=0.20)
    print(f"Detected {len(faces)} real human student faces in classroom photo.")

    if len(faces) == 0:
        print("Warning: No faces detected in classroom photo.")
        return {"ok": False, "reason": "No faces detected"}

    # Sort faces by spatial position (top-to-bottom, left-to-right rows)
    faces.sort(key=lambda f: (f["bbox"][1] // 120, f["bbox"][0]))

    student_names = [
        "gwen", "Aarav Sharma", "Sophia Chen", "Liam Johnson", "Noah Smith",
        "Emma Watson", "Lucas Brown", "Olivia Garcia", "Ethan Miller", "Ava Davis",
        "Mason Rodriguez", "Isabella Martinez", "William Hernandez", "Mia Lopez", "James Gonzalez",
        "Charlotte Wilson", "Benjamin Anderson", "Amelia Thomas", "Elijah Taylor", "Harper Moore",
        "Alexander Jackson", "Evelyn Martin", "Daniel Lee", "Abigail Perez", "Henry Thompson",
        "Emily White", "Sebastian Harris", "Elizabeth Sanchez", "Jack Clark", "Camila Ramirez",
        "Owen Lewis", "Ella Robinson", "Samuel Walker", "Avery Young", "Ryan Allen",
        "Sofia King", "Matthew Wright", "Chloe Scott", "Jackson Torres", "Victoria Nguyen"
    ]

    img_h, img_w = image_rgb.shape[:2]
    seeded_count = 0

    # Clean existing students in database for fresh seeding
    with db.get_conn() as conn:
        conn.execute("DELETE FROM face_embeddings")
        conn.execute("DELETE FROM attendance_records")
        conn.execute("DELETE FROM students")

    # Special handling for Gwen (student_id = "5")
    for idx, face in enumerate(faces, start=1):
        if idx == 5 or idx > len(student_names):
            sid = "5" if idx == 5 else str(idx)
            name = "gwen" if idx == 5 else f"Student {idx}"
        else:
            sid = str(idx)
            name = student_names[idx - 1]

        cname = "Grade 10 - Section A" if idx <= 20 else "Grade 10 - Section B"

        if not db.student_exists(sid):
            db.add_student(sid, name, cname)

        # Crop face photo with padding
        x1, y1, x2, y2 = face["bbox"]
        pad = int(max(x2 - x1, y2 - y1) * 0.3)
        crop_x1 = max(0, x1 - pad)
        crop_y1 = max(0, y1 - pad)
        crop_x2 = min(img_w, x2 + pad)
        crop_y2 = min(img_h, y2 + pad)

        face_crop = image_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
        s_dir = os.path.join(STUDENT_PHOTO_DIR, sid)
        os.makedirs(s_dir, exist_ok=True)
        crop_path = os.path.join(s_dir, "ref_1.jpg")
        Image.fromarray(face_crop).save(crop_path, "JPEG", quality=95)

        # Store REAL 512-d ArcFace embedding in SQLite database!
        db.add_embedding(sid, face["embedding"], crop_path, face["det_score"])
        seeded_count += 1

    print(f"Successfully registered {seeded_count} real students into database!")
    return {"ok": True, "seeded_students": seeded_count, "faces_detected": len(faces)}


if __name__ == "__main__":
    seed_real_students()
