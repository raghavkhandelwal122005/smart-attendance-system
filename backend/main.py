"""
main.py
-------
FastAPI application for Smart Attendance System:
  - Face Database Management & Embeddings
  - Multi-Student Face Recognition with Global Ranked Assignment
  - Attendance Verification, Finalization & CSV Export
"""

import os
import io
import csv
import uuid
from datetime import date
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import database as db
import face_engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
STUDENT_PHOTO_DIR = os.path.join(BASE_DIR, "data", "student_photos")
SESSION_PHOTO_DIR = os.path.join(BASE_DIR, "data", "session_photos")
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

DEFAULT_THRESHOLD = 0.35  # Cosine similarity threshold for ResNet50 ArcFace (buffalo_l)

os.makedirs(STUDENT_PHOTO_DIR, exist_ok=True)
os.makedirs(SESSION_PHOTO_DIR, exist_ok=True)

app = FastAPI(title="Smart Attendance System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    db.init_db()
    db.seed_demo_data()
    # Warm up InsightFace detector & recognizer on startup
    face_engine.get_face_app()


@app.get("/api/stats")
def get_dashboard_stats():
    return db.get_stats()


@app.post("/api/demo/seed")
def seed_demo():
    return db.seed_demo_data()


@app.get("/api/demo/classroom-photo")
def get_demo_classroom_photo():
    demo_path = os.path.join(BASE_DIR, "data", "demo_classroom_60.jpg")
    if not os.path.exists(demo_path):
        import generate_demo_photo
        generate_demo_photo.generate_classroom_photo()
    with open(demo_path, "rb") as f:
        data = f.read()
    return StreamingResponse(
        io.BytesIO(data),
        media_type="image/jpeg",
        headers={"Content-Disposition": "inline; filename=demo_classroom_60.jpg"}
    )




# ---------------------------------------------------------------------------
# Students / Face Database
# ---------------------------------------------------------------------------

@app.post("/api/students")
async def register_student(
    student_id: str = Form(...),
    name: str = Form(...),
    class_name: str = Form(""),
    photos: List[UploadFile] = File(...),
):
    if db.student_exists(student_id):
        raise HTTPException(400, f"Student ID '{student_id}' already exists.")
    if len(photos) < 1:
        raise HTTPException(400, "At least one photo is required.")

    db.add_student(student_id, name, class_name)

    student_dir = os.path.join(STUDENT_PHOTO_DIR, student_id)
    os.makedirs(student_dir, exist_ok=True)

    saved = 0
    skipped = []
    for photo in photos:
        raw = await photo.read()
        try:
            image_rgb = face_engine.load_image_from_bytes(raw)
        except Exception:
            skipped.append({"file": photo.filename, "reason": "unreadable image"})
            continue

        faces = face_engine.detect_faces(image_rgb, det_thresh=0.25)
        if len(faces) == 0:
            skipped.append({"file": photo.filename, "reason": "no face detected"})
            continue
        if len(faces) > 1:
            # Registration photo: select largest face
            faces.sort(
                key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
                reverse=True,
            )

        face = faces[0]
        filename = f"{uuid.uuid4().hex}.jpg"
        path = os.path.join(student_dir, filename)
        from PIL import Image
        Image.fromarray(image_rgb).save(path, "JPEG", quality=90)

        db.add_embedding(student_id, face["embedding"], path, face["det_score"])
        saved += 1

    if saved == 0:
        db.delete_student(student_id)
        raise HTTPException(
            400,
            f"Could not extract a face from any uploaded photo. Details: {skipped}",
        )

    return {
        "student_id": student_id,
        "name": name,
        "class_name": class_name,
        "photos_saved": saved,
        "photos_skipped": skipped,
    }


@app.get("/api/students")
def get_students():
    return db.list_students()


@app.delete("/api/students/{student_id}")
def remove_student(student_id: str):
    if not db.student_exists(student_id):
        raise HTTPException(404, "Student not found")
    db.delete_student(student_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Sessions & Recognition
# ---------------------------------------------------------------------------

class SessionCreate(BaseModel):
    class_name: str
    session_date: Optional[str] = None


@app.post("/api/sessions")
def create_session(payload: SessionCreate):
    session_date = payload.session_date or date.today().isoformat()
    session_id = db.create_session(payload.class_name, session_date)
    return {"id": session_id, "class_name": payload.class_name, "session_date": session_date}


@app.get("/api/sessions")
def get_sessions():
    return db.list_sessions()


@app.get("/api/sessions/{session_id}/records")
def get_records(session_id: int):
    return db.get_session_records(session_id)


@app.post("/api/sessions/{session_id}/recognize")
async def recognize(
    session_id: int,
    image: UploadFile = File(...),
    threshold: float = Query(DEFAULT_THRESHOLD, ge=0.0, le=1.0),
):
    raw = await image.read()
    try:
        image_rgb = face_engine.load_image_from_bytes(raw)
    except Exception:
        raise HTTPException(400, "Could not decode uploaded image.")

    filename = f"{session_id}_{uuid.uuid4().hex}.jpg"
    photo_path = os.path.join(SESSION_PHOTO_DIR, filename)
    from PIL import Image
    Image.fromarray(image_rgb).save(photo_path, "JPEG", quality=90)

    # High resolution face detection with det_thresh=0.25 to catch every student face
    faces = face_engine.detect_faces(image_rgb, det_thresh=0.25)
    gallery = db.get_gallery()

    # Vectorized matrix similarity matching for enterprise scaling (1,000s of vectors)
    face_embeddings_list = [f["embedding"] for f in faces]
    face_sim_maps = face_engine.compute_student_similarities_vectorized(face_embeddings_list, gallery)

    all_pairs = []
    for face_idx, sim_map in enumerate(face_sim_maps):
        for sid, info in sim_map.items():
            all_pairs.append((info["max_score"], face_idx, sid, info["name"]))

    # Sort candidates by similarity score descending
    all_pairs.sort(key=lambda p: p[0], reverse=True)

    assigned_faces = {}     # face_idx -> (student_id, name, score)
    claimed_students = set()

    for score, face_idx, sid, name in all_pairs:
        if score < threshold:
            break
        if face_idx not in assigned_faces and sid not in claimed_students:
            assigned_faces[face_idx] = (sid, name, score)
            claimed_students.add(sid)

    results = []
    for face_idx, face in enumerate(faces):
        if face_idx in assigned_faces:
            sid, name, score = assigned_faces[face_idx]
            status = "present"
        else:
            # Face wasn't matched to any database student above threshold
            sim_map = face_sim_maps[face_idx]
            best_unmatched_score = max([info["max_score"] for info in sim_map.values()], default=0.0)
            sid, name = None, None
            score = best_unmatched_score
            status = "unknown"

        record_id = db.add_attendance_record(
            session_id=session_id,
            student_id=sid,
            name=name,
            status=status,
            confidence=round(score, 4),
            bbox=face["bbox"],
            source_photo=photo_path,
            verified=0,
        )
        results.append(
            {
                "record_id": record_id,
                "bbox": face["bbox"],
                "det_score": round(face["det_score"], 4),
                "student_id": sid,
                "name": name,
                "confidence": round(score, 4),
                "status": status,
            }
        )

    return {
        "session_id": session_id,
        "faces_detected": len(faces),
        "students_matched": len(claimed_students),
        "image_width": image_rgb.shape[1],
        "image_height": image_rgb.shape[0],
        "results": results,
        "photo_url": f"/api/photo?path={filename}&kind=session",
    }


@app.get("/api/photo")
def get_photo(path: str, kind: str = "session"):
    base = SESSION_PHOTO_DIR if kind == "session" else STUDENT_PHOTO_DIR
    full = os.path.join(base, path)
    if not os.path.abspath(full).startswith(os.path.abspath(base)):
        raise HTTPException(400, "Invalid path")
    if not os.path.exists(full):
        raise HTTPException(404, "Not found")
    with open(full, "rb") as f:
        data = f.read()
    return StreamingResponse(io.BytesIO(data), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Teacher Verification & CSV Export
# ---------------------------------------------------------------------------

class RecordUpdate(BaseModel):
    student_id: Optional[str] = None
    name: Optional[str] = None
    status: str = "present"


@app.put("/api/sessions/{session_id}/records/{record_id}")
def update_record(session_id: int, record_id: int, payload: RecordUpdate):
    db.update_record(
        record_id,
        student_id=payload.student_id,
        name=payload.name,
        status=payload.status,
        verified=1,
    )
    return {"ok": True}


@app.delete("/api/sessions/{session_id}/records/{record_id}")
def remove_record(session_id: int, record_id: int):
    db.delete_record(record_id)
    return {"ok": True}


@app.post("/api/sessions/{session_id}/finalize")
def finalize_session(session_id: int, class_name: str = Query(...)):
    db.mark_all_students_absent_if_missing(session_id, class_name)
    return {"ok": True, "records": db.get_session_records(session_id)}


@app.get("/api/sessions/{session_id}/export")
def export_csv(session_id: int):
    records = db.get_session_records(session_id)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["student_id", "name", "status", "confidence", "verified"])
    for r in records:
        writer.writerow(
            [r["student_id"], r["name"], r["status"], r["confidence"], r["verified"]]
        )
    buf.seek(0)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=attendance_session_{session_id}.csv"},
    )


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
