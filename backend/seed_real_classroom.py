"""
seed_real_classroom.py
----------------------
Triggered by POST /api/demo/seed.
Detects real student faces from demo_classroom_60.jpg (generated on-the-fly if absent),
crops face thumbnails, stores real 512-d ArcFace embeddings in the database,
and persists face crops via the storage layer.

Guards:
- Does NOT wipe existing students if the database is already populated.
- Writes demo photo to /tmp on Vercel (writable) rather than backend/data/.
"""

import os
import io
import numpy as np
from PIL import Image

import database as db
import face_engine
import storage

BASE_DIR = os.path.dirname(__file__)
IS_VERCEL = bool(os.environ.get("VERCEL"))

# Demo photo: use /tmp on Vercel (writable), local data/ dir otherwise
_LOCAL_DEMO_PHOTO = os.path.join(BASE_DIR, "data", "demo_classroom_60.jpg")
_TMP_DEMO_PHOTO   = "/tmp/demo_classroom_60.jpg"
TARGET_DEMO_PHOTO = _TMP_DEMO_PHOTO if IS_VERCEL else _LOCAL_DEMO_PHOTO


def _ensure_demo_photo() -> str:
    """
    Return a path to demo_classroom_60.jpg, generating it if necessary.
    On Vercel we always write to /tmp; locally we prefer backend/data/.
    """
    if os.path.exists(TARGET_DEMO_PHOTO):
        return TARGET_DEMO_PHOTO

    # Try the other location (e.g. local data/ exists but /tmp doesn't yet)
    alt = _LOCAL_DEMO_PHOTO if IS_VERCEL else _TMP_DEMO_PHOTO
    if os.path.exists(alt):
        import shutil
        os.makedirs(os.path.dirname(TARGET_DEMO_PHOTO), exist_ok=True)
        shutil.copy2(alt, TARGET_DEMO_PHOTO)
        return TARGET_DEMO_PHOTO

    # Generate fresh
    import generate_demo_photo
    return generate_demo_photo.generate_classroom_photo(output_path=TARGET_DEMO_PHOTO)


def seed_real_students():
    try:
        # Guard: do not wipe an already-populated database
        stats = db.get_stats()
        if stats["total_students"] > 0:
            return {
                "ok": True,
                "message": "Database already has students — skipping re-seed to protect existing data.",
                "total_students": stats["total_students"],
            }

        demo_photo_path = _ensure_demo_photo()

        with open(demo_photo_path, "rb") as f:
            raw_bytes = f.read()

        image_rgb = face_engine.load_image_from_bytes(raw_bytes)
        faces = face_engine.detect_faces(image_rgb, det_thresh=0.20)
        print(f"[SEED] Detected {len(faces)} faces in demo classroom photo.")

        if len(faces) == 0:
            return {"ok": False, "reason": "No faces detected in demo classroom photo."}

        # Sort spatially: top-to-bottom rows, left-to-right within each row
        faces.sort(key=lambda f: (f["bbox"][1] // 120, f["bbox"][0]))

        student_names = [
            "gwen", "Aarav Sharma", "Sophia Chen", "Liam Johnson", "Noah Smith",
            "Emma Watson", "Lucas Brown", "Olivia Garcia", "Ethan Miller", "Ava Davis",
            "Mason Rodriguez", "Isabella Martinez", "William Hernandez", "Mia Lopez", "James Gonzalez",
            "Charlotte Wilson", "Benjamin Anderson", "Amelia Thomas", "Elijah Taylor", "Harper Moore",
            "Alexander Jackson", "Evelyn Martin", "Daniel Lee", "Abigail Perez", "Henry Thompson",
            "Emily White", "Sebastian Harris", "Elizabeth Sanchez", "Jack Clark", "Camila Ramirez",
            "Owen Lewis", "Ella Robinson", "Samuel Walker", "Avery Young", "Ryan Allen",
            "Sofia King", "Matthew Wright", "Chloe Scott", "Jackson Torres", "Victoria Nguyen",
        ]

        img_h, img_w = image_rgb.shape[:2]
        seeded_count = 0

        # Clear existing data first (safe here because we confirmed count == 0 above)
        with db.get_conn() as conn:
            conn.execute("DELETE FROM face_embeddings")
            conn.execute("DELETE FROM attendance_records")
            conn.execute("DELETE FROM students")

        for idx, face in enumerate(faces, start=1):
            if idx > len(student_names):
                name = f"Student {idx}"
            else:
                name = student_names[idx - 1]
            sid   = str(idx)
            cname = "Grade 10 - Section A" if idx <= 20 else "Grade 10 - Section B"

            if not db.student_exists(sid):
                db.add_student(sid, name, cname)

            # Crop face with padding
            x1, y1, x2, y2 = face["bbox"]
            pad     = int(max(x2 - x1, y2 - y1) * 0.3)
            crop_x1 = max(0, x1 - pad)
            crop_y1 = max(0, y1 - pad)
            crop_x2 = min(img_w, x2 + pad)
            crop_y2 = min(img_h, y2 + pad)

            face_crop = image_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_buf  = io.BytesIO()
            Image.fromarray(face_crop).save(crop_buf, format="JPEG", quality=95)

            saved_path = storage.save_photo(crop_buf.getvalue(), f"{sid}/ref_1.jpg", kind="student")
            db.add_embedding(sid, face["embedding"], saved_path, face["det_score"])
            seeded_count += 1

        print(f"[SEED] Registered {seeded_count} students with real ArcFace embeddings.")
        return {
            "ok": True,
            "seeded_students": seeded_count,
            "faces_detected": len(faces),
        }

    except Exception as err:
        import traceback
        print(f"[SEED ERROR] {err}\n{traceback.format_exc()}")
        return {"ok": False, "error": str(err)}


if __name__ == "__main__":
    seed_real_students()
