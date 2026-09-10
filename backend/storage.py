"""
storage.py
----------
Persistent object storage for student reference photos and session photos.
Priority:
  1. Cloudinary  — set CLOUDINARY_URL  (cloudinary://key:secret@cloud_name)
  2. Vercel Blob — set BLOB_READ_WRITE_TOKEN
  3. Local disk  — fallback for local development (and /tmp on Vercel without cloud storage)
"""

import os
import io

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IS_VERCEL = bool(os.environ.get("VERCEL"))


def _local_dir(kind: str) -> str:
    """Return the local base directory for the given storage kind."""
    if IS_VERCEL:
        base = f"/tmp/{kind}_photos"
    else:
        base = os.path.join(BASE_DIR, "data", f"{kind}_photos")
    os.makedirs(base, exist_ok=True)
    return base


def is_cloud_storage_enabled() -> bool:
    return bool(
        os.environ.get("CLOUDINARY_URL")
        or os.environ.get("CLOUDINARY_CLOUD_NAME")
        or os.environ.get("BLOB_READ_WRITE_TOKEN")
    )


# ---------------------------------------------------------------------------
# Save a photo — returns a persistent URL (cloud) or relative filename (local)
# ---------------------------------------------------------------------------

def save_photo(image_bytes: bytes, filename: str, kind: str = "session") -> str:
    """
    Persist image_bytes. Returns:
      - A full https:// URL when cloud storage is configured and upload succeeds.
      - The relative filename string when falling back to local/tmp disk.
    """

    # 1. Cloudinary
    cloudinary_url = os.environ.get("CLOUDINARY_URL") or os.environ.get("CLOUDINARY_CLOUD_NAME")
    if cloudinary_url:
        result = _save_cloudinary(image_bytes, filename, kind)
        if result:
            return result

    # 2. Vercel Blob
    blob_token = os.environ.get("BLOB_READ_WRITE_TOKEN")
    if blob_token:
        result = _save_vercel_blob(image_bytes, filename, kind, blob_token)
        if result:
            return result

    # 3. Local / /tmp fallback
    return _save_local(image_bytes, filename, kind)


def _save_cloudinary(image_bytes: bytes, filename: str, kind: str):
    try:
        import cloudinary  # noqa: F401
        import cloudinary.uploader

        folder = f"smart_attendance/{kind}"
        # Use filename without extension as the public_id (Cloudinary adds extension)
        public_id = os.path.splitext(filename.replace("/", "__"))[0]

        res = cloudinary.uploader.upload(
            image_bytes,
            folder=folder,
            public_id=public_id,
            overwrite=True,
            resource_type="image",
        )
        url = res.get("secure_url") or res.get("url")
        if url:
            return url
    except Exception as err:
        print(f"[STORAGE] Cloudinary upload failed ({err}), falling back.")
    return None


def _save_vercel_blob(image_bytes: bytes, filename: str, kind: str, token: str):
    """
    Upload via the Vercel Blob REST API.
    Docs: https://vercel.com/docs/storage/vercel-blob/using-blob-sdk#upload-a-file
    The correct endpoint is PUT https://blob.vercel-storage.com/<path>
    with Authorization: Bearer <token> and x-api-version: 7 headers.
    """
    try:
        import requests

        # Sanitize filename for use in the URL path
        safe_name = filename.replace("\\", "/")
        blob_path = f"{kind}/{safe_name}"
        url = f"https://blob.vercel-storage.com/{blob_path}"

        headers = {
            "Authorization": f"Bearer {token}",
            "x-api-version": "7",
            "content-type": "image/jpeg",
        }
        resp = requests.put(url, headers=headers, data=image_bytes, timeout=15)
        if resp.status_code in (200, 201):
            data = resp.json()
            return data.get("url") or url
        else:
            print(f"[STORAGE] Vercel Blob returned {resp.status_code}: {resp.text[:200]}")
    except Exception as err:
        print(f"[STORAGE] Vercel Blob upload failed ({err}), falling back.")
    return None


def _save_local(image_bytes: bytes, filename: str, kind: str) -> str:
    from PIL import Image

    target_dir = _local_dir(kind)

    # Support sub-paths like "student_id/filename.jpg"
    safe_filename = filename.replace("\\", "/")
    full_path = os.path.join(target_dir, safe_filename)
    # Security: ensure the resolved path stays inside target_dir
    if not os.path.abspath(full_path).startswith(os.path.abspath(target_dir)):
        raise ValueError(f"Unsafe filename rejected: {filename!r}")

    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.save(full_path, "JPEG", quality=90)
    return safe_filename  # relative reference stored in DB


# ---------------------------------------------------------------------------
# Resolve a stored photo reference for serving
# ---------------------------------------------------------------------------

def resolve_photo_location(photo_ref: str, kind: str = "session") -> dict:
    """
    Returns:
      {"exists": True,  "is_url": True,  "url": "https://..."}   — redirect to cloud
      {"exists": True,  "is_url": False, "local_path": "/abs/path"} — stream from disk
      {"exists": False}
    """
    if not photo_ref:
        return {"exists": False}

    if photo_ref.startswith("http://") or photo_ref.startswith("https://"):
        return {"exists": True, "is_url": True, "url": photo_ref}

    target_dir = _local_dir(kind)
    safe_ref = photo_ref.replace("\\", "/")
    full_path = os.path.join(target_dir, safe_ref)

    if not os.path.abspath(full_path).startswith(os.path.abspath(target_dir)):
        return {"exists": False, "error": "Invalid path"}

    if os.path.exists(full_path):
        return {"exists": True, "is_url": False, "local_path": full_path}

    return {"exists": False}
