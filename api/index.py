import os
import sys

# Set InsightFace model cache directory to writable /tmp on Vercel BEFORE any backend imports.
# Without this, InsightFace tries to write model weights to ~/.insightface which is read-only.
if not os.environ.get("INSIGHTFACE_HOME"):
    os.environ["INSIGHTFACE_HOME"] = "/tmp/.insightface"

# Add backend directory to sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from main import app  # noqa: F401 — Vercel uses this as the ASGI handler
