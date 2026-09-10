"""
database.py
-----------
Unified database layer. Uses PostgreSQL in production (DATABASE_URL / POSTGRES_URL)
and SQLite for local development. Never falls back to SQLite on Vercel — raises
a clear error if DATABASE_URL is absent in production so misconfiguration is obvious.

Key design decisions:
- PostgreSQL DDL: each statement issued separately (psycopg2 rejects multi-statement execute)
- Seeding: only loads database_seed.py when table is empty; never wipes existing data
- Row access: always returns plain dicts so callers work identically for PG and SQLite
"""

import os
import json
import sqlite3
from datetime import datetime
from contextlib import contextmanager

# ---------------------------------------------------------------------------
# Database connection configuration
# ---------------------------------------------------------------------------

_RAW_DB_URL = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
IS_POSTGRES = bool(_RAW_DB_URL)

if IS_POSTGRES:
    # Heroku / Vercel Postgres use the legacy postgres:// scheme; psycopg2 needs postgresql://
    DATABASE_URL = (
        _RAW_DB_URL.replace("postgres://", "postgresql://", 1)
        if _RAW_DB_URL.startswith("postgres://")
        else _RAW_DB_URL
    )
    DB_PATH = None  # not used in production
else:
    if os.environ.get("VERCEL"):
        # Vercel environment without DATABASE_URL — refuse to run silently on ephemeral SQLite
        raise RuntimeError(
            "DATABASE_URL / POSTGRES_URL environment variable is not set. "
            "Set it in your Vercel project settings to a PostgreSQL connection string "
            "(e.g. from Vercel Postgres, Neon, or Supabase). "
            "The application cannot persist data without a real database on Vercel."
        )
    DATABASE_URL = None
    DB_DIR = os.path.join(os.path.dirname(__file__), "data")
    DB_PATH = os.path.join(DB_DIR, "attendance.db")


# ---------------------------------------------------------------------------
# Connection context manager
# ---------------------------------------------------------------------------

class _PgWrapper:
    """
    Thin wrapper around psycopg2 connection + RealDictCursor that exposes the
    same execute / fetchone / fetchall / lastrowid interface as the SQLite conn,
    and translates '?' placeholders to '%s' for psycopg2.
    """

    def __init__(self, conn, cur):
        self._conn = conn
        self._cur = cur

    def execute(self, query: str, params=None):
        pg_query = query.replace("?", "%s")
        self._cur.execute(pg_query, params)
        return self

    def executemany(self, query: str, seq_of_params):
        pg_query = query.replace("?", "%s")
        self._cur.executemany(pg_query, seq_of_params)
        return self

    def fetchone(self):
        row = self._cur.fetchone()
        return dict(row) if row is not None else None

    def fetchall(self):
        rows = self._cur.fetchall()
        return [dict(r) for r in rows] if rows else []

    @property
    def lastrowid(self):
        return getattr(self._cur, "lastrowid", None)


@contextmanager
def get_conn():
    if IS_POSTGRES:
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        wrapper = _PgWrapper(conn, cur)
        try:
            yield wrapper
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()
    else:
        os.makedirs(DB_DIR, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def _exec(conn, sql: str, params=None):
    """Helper: execute one statement, normalising dict access."""
    if params is not None:
        conn.execute(sql, params)
    else:
        conn.execute(sql)


def init_db():
    """
    Create tables if they don't exist, then seed demo students from database_seed.py
    if the students table is empty.  Never wipes existing data.
    """
    with get_conn() as conn:
        if IS_POSTGRES:
            # PostgreSQL: issue each DDL statement individually (psycopg2 rejects batched DDL)
            _exec(conn, """
                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    name       TEXT NOT NULL,
                    class_name TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
            """)
            _exec(conn, """
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id         SERIAL PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    embedding  TEXT NOT NULL,
                    photo_path TEXT,
                    det_score  REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
                )
            """)
            _exec(conn, """
                CREATE TABLE IF NOT EXISTS sessions (
                    id           SERIAL PRIMARY KEY,
                    class_name   TEXT NOT NULL,
                    session_date TEXT NOT NULL,
                    created_at   TEXT NOT NULL
                )
            """)
            _exec(conn, """
                CREATE TABLE IF NOT EXISTS attendance_records (
                    id           SERIAL PRIMARY KEY,
                    session_id   INTEGER NOT NULL,
                    student_id   TEXT,
                    name         TEXT,
                    status       TEXT NOT NULL DEFAULT 'present',
                    confidence   REAL,
                    verified     INTEGER NOT NULL DEFAULT 0,
                    bbox         TEXT,
                    source_photo TEXT,
                    created_at   TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)
            _exec(conn, "CREATE INDEX IF NOT EXISTS idx_students_class   ON students(class_name)")
            _exec(conn, "CREATE INDEX IF NOT EXISTS idx_embeddings_student ON face_embeddings(student_id)")
            _exec(conn, "CREATE INDEX IF NOT EXISTS idx_records_session  ON attendance_records(session_id)")
            _exec(conn, "CREATE INDEX IF NOT EXISTS idx_records_student  ON attendance_records(student_id)")
        else:
            # SQLite: executescript is fine for batched DDL
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    name       TEXT NOT NULL,
                    class_name TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    embedding  TEXT NOT NULL,
                    photo_path TEXT,
                    det_score  REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    class_name   TEXT NOT NULL,
                    session_date TEXT NOT NULL,
                    created_at   TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attendance_records (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id   INTEGER NOT NULL,
                    student_id   TEXT,
                    name         TEXT,
                    status       TEXT NOT NULL DEFAULT 'present',
                    confidence   REAL,
                    verified     INTEGER NOT NULL DEFAULT 0,
                    bbox         TEXT,
                    source_photo TEXT,
                    created_at   TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_students_class    ON students(class_name);
                CREATE INDEX IF NOT EXISTS idx_embeddings_student ON face_embeddings(student_id);
                CREATE INDEX IF NOT EXISTS idx_records_session   ON attendance_records(session_id);
                CREATE INDEX IF NOT EXISTS idx_records_student   ON attendance_records(student_id);
            """)

        # Seed pre-computed embeddings only when table is empty (never wipe existing data)
        row = conn.execute("SELECT COUNT(*) AS count FROM students").fetchone()
        student_count = row["count"] if isinstance(row, dict) else row[0]

        if student_count == 0:
            _seed_from_module(conn)


def _seed_from_module(conn):
    """Insert SEED_STUDENTS and SEED_EMBEDDINGS from database_seed.py."""
    try:
        import database_seed
        now = datetime.utcnow().isoformat()

        for s in database_seed.SEED_STUDENTS:
            if IS_POSTGRES:
                conn.execute(
                    "INSERT INTO students (student_id, name, class_name, created_at) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT (student_id) DO NOTHING",
                    (s["student_id"], s["name"], s.get("class_name", ""), s.get("created_at", now)),
                )
            else:
                conn.execute(
                    "INSERT OR IGNORE INTO students (student_id, name, class_name, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (s["student_id"], s["name"], s.get("class_name", ""), s.get("created_at", now)),
                )

        for e in database_seed.SEED_EMBEDDINGS:
            # Normalize photo_path — strip absolute local paths from seed file
            photo_path = e.get("photo_path", "") or ""
            if os.path.isabs(photo_path):
                photo_path = ""  # don't store Windows absolute paths in production DB

            conn.execute(
                "INSERT INTO face_embeddings "
                "(student_id, embedding, photo_path, det_score, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    e["student_id"],
                    e["embedding"],
                    photo_path,
                    e.get("det_score", 0.95),
                    e.get("created_at", datetime.utcnow().isoformat()),
                ),
            )

        print(f"[DB INIT] Seeded {len(database_seed.SEED_STUDENTS)} students from database_seed.py")
    except Exception as err:
        print(f"[DB INIT] Seed error (non-fatal): {err}")


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------

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
                   COUNT(f.id) AS photo_count
            FROM students s
            LEFT JOIN face_embeddings f ON f.student_id = s.student_id
            GROUP BY s.student_id, s.name, s.class_name, s.created_at
            ORDER BY s.class_name, s.name
            """
        ).fetchall()
        return [dict(r) for r in rows]


def delete_student(student_id: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM face_embeddings WHERE student_id = ?", (student_id,))
        conn.execute("DELETE FROM students WHERE student_id = ?", (student_id,))


# ---------------------------------------------------------------------------
# Face embeddings
# ---------------------------------------------------------------------------

def add_embedding(student_id: str, embedding, photo_path: str, det_score: float):
    emb_json = json.dumps(embedding.tolist()) if hasattr(embedding, "tolist") else json.dumps(embedding)
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO face_embeddings "
            "(student_id, embedding, photo_path, det_score, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (student_id, emb_json, photo_path, det_score, datetime.utcnow().isoformat()),
        )


def get_gallery():
    import numpy as np
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT f.student_id, s.name, f.embedding
            FROM face_embeddings f
            JOIN students s ON s.student_id = f.student_id
            """
        ).fetchall()

    return [
        {
            "student_id": r["student_id"],
            "name": r["name"],
            "embedding": np.array(json.loads(r["embedding"]), dtype=np.float32),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Sessions & attendance
# ---------------------------------------------------------------------------

def create_session(class_name: str, session_date: str) -> int:
    with get_conn() as conn:
        if IS_POSTGRES:
            row = conn.execute(
                "INSERT INTO sessions (class_name, session_date, created_at) "
                "VALUES (?, ?, ?) RETURNING id",
                (class_name, session_date, datetime.utcnow().isoformat()),
            ).fetchone()
            return row["id"]
        else:
            cur = conn.execute(
                "INSERT INTO sessions (class_name, session_date, created_at) VALUES (?, ?, ?)",
                (class_name, session_date, datetime.utcnow().isoformat()),
            )
            return cur.lastrowid


def list_sessions():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM sessions ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def add_attendance_record(
    session_id, student_id, name, status, confidence, bbox, source_photo, verified=0
):
    bbox_json = json.dumps(bbox) if bbox is not None else None
    with get_conn() as conn:
        if IS_POSTGRES:
            row = conn.execute(
                """
                INSERT INTO attendance_records
                    (session_id, student_id, name, status, confidence,
                     verified, bbox, source_photo, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
                """,
                (session_id, student_id, name, status, confidence,
                 verified, bbox_json, source_photo, datetime.utcnow().isoformat()),
            ).fetchone()
            return row["id"]
        else:
            cur = conn.execute(
                """
                INSERT INTO attendance_records
                    (session_id, student_id, name, status, confidence,
                     verified, bbox, source_photo, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, student_id, name, status, confidence,
                 verified, bbox_json, source_photo, datetime.utcnow().isoformat()),
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
            "UPDATE attendance_records "
            "SET student_id = ?, name = ?, status = ?, verified = ? "
            "WHERE id = ?",
            (student_id, name, status, verified, record_id),
        )


def delete_record(record_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM attendance_records WHERE id = ?", (record_id,))


def mark_all_students_absent_if_missing(session_id: int, class_name: str):
    """Insert an 'absent' record for every class student not yet in this session."""
    with get_conn() as conn:
        students = conn.execute(
            "SELECT student_id, name FROM students WHERE class_name = ?", (class_name,)
        ).fetchall()
        existing = conn.execute(
            "SELECT DISTINCT student_id FROM attendance_records "
            "WHERE session_id = ? AND student_id IS NOT NULL",
            (session_id,),
        ).fetchall()
        existing_ids = {dict(r)["student_id"] for r in existing}

        now = datetime.utcnow().isoformat()
        for s in students:
            s = dict(s)
            if s["student_id"] not in existing_ids:
                conn.execute(
                    """
                    INSERT INTO attendance_records
                        (session_id, student_id, name, status, confidence,
                         verified, bbox, source_photo, created_at)
                    VALUES (?, ?, ?, 'absent', NULL, 1, NULL, NULL, ?)
                    """,
                    (session_id, s["student_id"], s["name"], now),
                )


def get_stats():
    with get_conn() as conn:
        def _count(sql):
            row = conn.execute(sql).fetchone()
            return row["count"] if isinstance(row, dict) else row[0]

        return {
            "total_students": _count("SELECT COUNT(*) AS count FROM students"),
            "total_photos":   _count("SELECT COUNT(*) AS count FROM face_embeddings"),
            "total_sessions": _count("SELECT COUNT(*) AS count FROM sessions"),
            "total_present":  _count("SELECT COUNT(*) AS count FROM attendance_records WHERE status = 'present'"),
        }


def seed_demo_data():
    """
    Trigger from /api/demo/seed endpoint. Runs the real-classroom seeder
    (reads demo_classroom_60.jpg, detects faces, inserts real ArcFace embeddings).
    Skips if students already exist in the database.
    """
    try:
        import seed_real_classroom
        return seed_real_classroom.seed_real_students()
    except Exception as e:
        print(f"[DB] seed_demo_data error: {e}")
        return {"ok": False, "error": str(e)}
