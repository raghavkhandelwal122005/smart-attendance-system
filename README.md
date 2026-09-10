# Smart Attendance System (AI Face Recognition)

A production-ready attendance system powered by FastAPI, InsightFace ArcFace, PostgreSQL / SQLite, and Object Storage:

1. **Student Face Database** — register students with ID, name, class, and reference photos (upload or webcam).
2. **Face Embedding** — face detector extracts 512-d ArcFace numerical embeddings via InsightFace.
3. **Dual Database Engine** — supports PostgreSQL (for Vercel serverless / cloud deployment via `DATABASE_URL` / `POSTGRES_URL`) and SQLite (for local offline dev).
4. **Persistent Object Storage** — supports Cloudinary / Vercel Blob / S3 for image persistence across serverless executions, with local file fallback.
5. **Attendance Recognition** — capture or upload a classroom group photo; every face is detected, embedded, and matched against student embeddings using matrix cosine similarity.
6. **Teacher Verification & CSV Export** — review annotated photos, correct matches, finalize session, and export CSV reports.

## Architecture & Production Deployment on Vercel

On Vercel's serverless environment, local filesystem files are ephemeral. The project uses:

- **Database**: PostgreSQL (Vercel Postgres, Neon, Supabase, Render Postgres) set via `DATABASE_URL` or `POSTGRES_URL`.
- **Storage**: Persistent Object Storage (Cloudinary / Vercel Blob) set via `CLOUDINARY_URL` or `BLOB_READ_WRITE_TOKEN`.
- **API Routing**: `vercel.json` routes `/api/*` to `api/index.py`. The frontend uses relative API paths (`/api/...`) by default.

### Environment Variables

Configure these in your Vercel project settings:

```env
DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
CLOUDINARY_URL=cloudinary://API_KEY:API_SECRET@CLOUD_NAME
```

## Local Setup

```bash
cd attendance_system
pip install -r requirements.txt
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.

## Privacy Note

This system processes biometric data (face photos and embeddings). Before deploying in an institutional setting, ensure compliance with applicable student-data privacy laws (e.g. FERPA or local regulations).
