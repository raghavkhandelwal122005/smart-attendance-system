# Smart Attendance System (Face Recognition)

A self-contained, working attendance system that:

1. **Student Face Database** — register each student with their Student ID, name, class, and 2-5 photos (upload or webcam).
2. **Face Embedding** — every photo is run through a face detector, and the detected face is converted into a 512-d numerical embedding using an ArcFace-family recognition model (via [InsightFace](https://github.com/deepinsight/insightface)).
3. **Vector Storage** — embeddings are stored in SQLite as JSON arrays. On recognition, all embeddings are loaded into memory and compared with fast vectorized cosine similarity (fine for classroom/school scale — hundreds to a few thousand faces).
4. **Attendance Recognition** — capture or upload a classroom photo; every face in it is detected, embedded, and matched against the student database.
5. **Attendance Generation** — matches above the confidence threshold are marked "present" automatically; the teacher reviews the annotated photo, can correct any wrong/missed match or mark an unknown face as a specific student, and finalizes the session (everyone else in that class who wasn't matched is marked "absent"). Results can be exported to CSV.

## How it works technically

- **Detection + Embedding**: `backend/face_engine.py` wraps InsightFace's `buffalo_sc` model pack (SCRFD detector + a MobileFaceNet-style ArcFace recognizer). It's pure pip-installable (no compiler needed) and runs on CPU. Model weights (~15 MB) auto-download on first run from InsightFace's GitHub releases.
- **Storage**: `backend/database.py` — plain SQLite (`students`, `face_embeddings`, `sessions`, `attendance_records` tables).
- **Matching**: for every detected face in a classroom photo, cosine similarity is computed against every stored student embedding (`face_engine.best_match`). A greedy assignment prevents two faces in the same photo from claiming the same student. Anything below the threshold is left as "unknown" for teacher review.
- **API**: `backend/main.py` — FastAPI app exposing REST endpoints and serving the frontend.
- **Frontend**: plain HTML/CSS/JS (`frontend/`) — no build step. Supports both webcam capture and file upload for both registration and attendance photos, draws bounding boxes + names on the classroom photo, and lets the teacher fix any result inline.

## Setup

```bash
cd attendance_system
pip install -r requirements.txt
cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in a browser (webcam capture requires either `localhost` or HTTPS).

The first run will download the InsightFace model pack automatically (~15 MB, needs internet access once).

## Using it

1. **Register Student** tab — enter Student ID, name, class, and add 2-5 photos (different angles/lighting improve accuracy). Submit.
2. **Take Attendance** tab — enter the class name and date, click "Start Session", capture or upload a classroom photo, adjust the confidence threshold if needed, and click "Run Recognition". Detected faces are boxed and labeled on the photo; use the dropdown next to each result to correct a wrong match, or "Remove" for false-positive detections (e.g. a poster of a face).
3. Click **Finalize Session** — every registered student in that class who wasn't matched is automatically marked absent, and you can **Export CSV**.
4. **Records** tab — browse past sessions and their per-student results.
5. **Student Database** tab — view/delete registered students.

## Tuning accuracy

- **Confidence threshold** (default 0.45, adjustable per-session in the UI): raise it to reduce false positives (wrong student matched), lower it if real students aren't being matched. In testing, correct matches scored ~0.90-0.98 cosine similarity while non-matches scored well under 0.2, so 0.4-0.5 is a safe default — but lighting, camera angle, and photo quality all affect this, so try it on your own data.
- **Registration photo quality**: use normal, uncropped photos (not tightly cropped to the jawline) so the detector has enough context around the face. Multiple photos per student (different lighting/angle) noticeably improve match reliability.
- **Detector input size**: `face_engine.py`'s `det_size=(640, 640)` balances speed vs. the ability to find small/far faces in a large classroom photo. Increase it (e.g. `(1024, 1024)`) if faces at the back of a large classroom photo are being missed, at the cost of slower processing.

## Scaling beyond a single classroom

This uses SQLite + brute-force cosine similarity, which comfortably handles a school of a few thousand students in-memory. If you outgrow that (many thousands of students, sub-millisecond lookups needed), swap `database.get_gallery()` + `face_engine.best_match()` for a dedicated vector database (e.g. Chroma, Qdrant, Milvus, or pgvector) — the rest of the system (detection, embedding, API, UI) doesn't need to change.

## Privacy note

This system stores biometric data (face photos and embeddings) tied to real students. Before deploying it in a real school, make sure you have appropriate consent and comply with your local student-data privacy regulations (e.g. FERPA in the US, or equivalent local laws).
