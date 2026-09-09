"""
database.py
-----------
SQLite storage layer. Embeddings are stored as JSON-encoded float lists
in a TEXT column (simplest possible approach, per project requirements:
"SQLite + manual cosine similarity"). At classroom / school scale
(hundreds to a few thousand embeddings) loading them into memory and
computing cosine similarity with numpy is fast (< a few ms).
"""

import sqlite3
import json
import os
from datetime import datetime
from contextlib import contextmanager

if os.environ.get("VERCEL"):
    DB_DIR = "/tmp/data"
    os.makedirs(DB_DIR, exist_ok=True)
    DB_PATH = os.path.join(DB_DIR, "attendance.db")
    # Copy seed DB if not yet present in /tmp
    seed_db = os.path.join(os.path.dirname(__file__), "data", "attendance.db")
    if not os.path.exists(DB_PATH) and os.path.exists(seed_db):
        import shutil
        shutil.copyfile(seed_db, DB_PATH)
else:
    DB_DIR = os.path.join(os.path.dirname(__file__), "data")
    DB_PATH = os.path.join(DB_DIR, "attendance.db")


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                class_name TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                embedding TEXT NOT NULL,
                photo_path TEXT,
                det_score REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_name TEXT NOT NULL,
                session_date TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS attendance_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                student_id TEXT,
                name TEXT,
                status TEXT NOT NULL DEFAULT 'present',  -- present / absent
                confidence REAL,
                verified INTEGER NOT NULL DEFAULT 0,     -- 0 = pending teacher review, 1 = confirmed
                bbox TEXT,
                source_photo TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_students_class ON students(class_name);
            CREATE INDEX IF NOT EXISTS idx_embeddings_student ON face_embeddings(student_id);
            CREATE INDEX IF NOT EXISTS idx_records_session ON attendance_records(session_id);
            CREATE INDEX IF NOT EXISTS idx_records_student ON attendance_records(student_id);
            """
        )


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------- Students ----------------

def add_student(student_id: str, name: str, class_name: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO students (student_id, name, class_name, created_at) VALUES (?, ?, ?, ?)",
            (student_id, name, class_name, datetime.utcnow().isoformat()),
        )


def student_exists(student_id: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM students WHERE student_id = ?", (student_id,)
        ).fetchone()
        return row is not None


def list_students():
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT s.student_id, s.name, s.class_name, s.created_at,
                   COUNT(f.id) as photo_count
            FROM students s
            LEFT JOIN face_embeddings f ON f.student_id = s.student_id
            GROUP BY s.student_id
            ORDER BY s.class_name, s.name
            """
        ).fetchall()
        return [dict(r) for r in rows]


def delete_student(student_id: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM face_embeddings WHERE student_id = ?", (student_id,))
        conn.execute("DELETE FROM students WHERE student_id = ?", (student_id,))


# ---------------- Embeddings ----------------

def add_embedding(student_id: str, embedding, photo_path: str, det_score: float):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO face_embeddings (student_id, embedding, photo_path, det_score, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                student_id,
                json.dumps(embedding.tolist()),
                photo_path,
                det_score,
                datetime.utcnow().isoformat(),
            ),
        )


def get_gallery():
    """
    Returns a flat list of every stored embedding joined with student name,
    ready for brute-force cosine similarity search.
    """
    import numpy as np

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT f.student_id, s.name, f.embedding
            FROM face_embeddings f
            JOIN students s ON s.student_id = f.student_id
            """
        ).fetchall()

    gallery = []
    for r in rows:
        gallery.append(
            {
                "student_id": r["student_id"],
                "name": r["name"],
                "embedding": np.array(json.loads(r["embedding"]), dtype=np.float32),
            }
        )
    return gallery


# ---------------- Sessions & Attendance ----------------

def create_session(class_name: str, session_date: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO sessions (class_name, session_date, created_at) VALUES (?, ?, ?)",
            (class_name, session_date, datetime.utcnow().isoformat()),
        )
        return cur.lastrowid


def list_sessions():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def add_attendance_record(session_id, student_id, name, status, confidence, bbox, source_photo, verified=0):
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO attendance_records
                (session_id, student_id, name, status, confidence, verified, bbox, source_photo, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                student_id,
                name,
                status,
                confidence,
                verified,
                json.dumps(bbox) if bbox is not None else None,
                source_photo,
                datetime.utcnow().isoformat(),
            ),
        )
        return cur.lastrowid


def get_session_records(session_id: int):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM attendance_records WHERE session_id = ? ORDER BY name",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def update_record(record_id: int, student_id: str, name: str, status: str, verified: int = 1):
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE attendance_records
            SET student_id = ?, name = ?, status = ?, verified = ?
            WHERE id = ?
            """,
            (student_id, name, status, verified, record_id),
        )


def delete_record(record_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM attendance_records WHERE id = ?", (record_id,))


def mark_all_students_absent_if_missing(session_id: int, class_name: str):
    """
    For every registered student in this class who has no attendance_record
    yet in this session, insert an 'absent' record. Called when the teacher
    finalizes the session.
    """
    with get_conn() as conn:
        students = conn.execute(
            "SELECT student_id, name FROM students WHERE class_name = ?", (class_name,)
        ).fetchall()
        existing = conn.execute(
            "SELECT DISTINCT student_id FROM attendance_records WHERE session_id = ? AND student_id IS NOT NULL",
            (session_id,),
        ).fetchall()
        existing_ids = {r["student_id"] for r in existing}

        for s in students:
            if s["student_id"] not in existing_ids:
                conn.execute(
                    """
                    INSERT INTO attendance_records
                        (session_id, student_id, name, status, confidence, verified, bbox, source_photo, created_at)
                    VALUES (?, ?, ?, 'absent', NULL, 1, NULL, NULL, ?)
                    """,
                    (session_id, s["student_id"], s["name"], datetime.utcnow().isoformat()),
                )


def get_stats():
    with get_conn() as conn:
        total_students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        total_photos = conn.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0]
        total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        total_present = conn.execute("SELECT COUNT(*) FROM attendance_records WHERE status = 'present'").fetchone()[0]
        return {
            "total_students": total_students,
            "total_photos": total_photos,
            "total_sessions": total_sessions,
            "total_present": total_present,
        }


def seed_demo_data():
    """
    Seed registered students from real classroom photo (including student Gwen ID: 5).
    """
    try:
        import seed_real_classroom
        return seed_real_classroom.seed_real_students()
    except Exception as e:
        print("Real classroom seeding fallback:", e)
        return {"ok": False, "error": str(e)}




