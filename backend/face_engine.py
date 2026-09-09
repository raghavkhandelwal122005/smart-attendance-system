"""
face_engine.py
--------------
InsightFace ArcFace Flagship Engine (buffalo_l: ResNet50 w600k_r50 + SCRFD 10G)
for high-accuracy face detection, 5-point alignment, and 512-d feature embedding.
"""

import io
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from insightface.app import FaceAnalysis

_APP = None


def get_face_app():
    """Lazily load and cache the ultra-fast InsightFace buffalo_sc model pack (1-second CPU speed)."""
    global _APP
    if _APP is None:
        _APP = FaceAnalysis(
            name="buffalo_sc",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        _APP.prepare(ctx_id=-1, det_size=(640, 640))
        if hasattr(_APP, "det_model") and _APP.det_model is not None:
            _APP.det_model.det_thresh = 0.20
    return _APP


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes into an RGB numpy array (H, W, 3)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(img)


def enhance_contrast_if_needed(image_rgb: np.ndarray) -> np.ndarray:
    """Apply adaptive CLAHE contrast enhancement if an image is dark or low contrast."""
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    avg_brightness = np.mean(l)
    if avg_brightness < 80 or np.std(l) < 35:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return image_rgb


def detect_faces(image_rgb: np.ndarray, det_thresh: float = 0.20):
    """
    Accelerated SCRFD 10G face detection + 5-point alignment + ResNet50 embedding extraction.
    Fast CPU viewport scaling (1024px) for sub-second recognition on classroom photos.
    """
    import time
    t0 = time.time()

    app = get_face_app()

    if hasattr(app, "det_model") and app.det_model is not None:
        app.det_model.det_thresh = det_thresh

    h, w = image_rgb.shape[:2]
    max_dim = max(h, w)
    if max_dim > 1024:
        scale = 1024.0 / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        det_img_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        scale_x = w / float(new_w)
        scale_y = h / float(new_h)
    else:
        det_img_rgb = image_rgb
        scale_x = 1.0
        scale_y = 1.0

    bgr = det_img_rgb[:, :, ::-1]
    faces = app.get(bgr)

    # Fallback retry with contrast enhancement if no faces found
    if len(faces) == 0 and det_img_rgb is image_rgb:
        enhanced_rgb = enhance_contrast_if_needed(image_rgb)
        bgr = enhanced_rgb[:, :, ::-1]
        faces = app.get(bgr)

    results = []
    for f in faces:
        if float(f.det_score) < det_thresh:
            continue
        bbox_det = f.bbox
        bbox = [
            int(bbox_det[0] * scale_x),
            int(bbox_det[1] * scale_y),
            int(bbox_det[2] * scale_x),
            int(bbox_det[3] * scale_y),
        ]
        embedding = f.normed_embedding.astype(np.float32)
        results.append(
            {
                "bbox": bbox,
                "det_score": float(f.det_score),
                "embedding": embedding,
            }
        )

    t1 = time.time()
    print(f"[FACE_ENGINE] Detected {len(results)} faces in {(t1-t0)*1000:.1f}ms")
    return results


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalized 512-d vectors."""
    return float(np.dot(a, b))


def compute_student_similarities(query_embedding: np.ndarray, gallery: list):
    """
    gallery: list of {"student_id": str, "name": str, "embedding": np.ndarray}
    Returns a dict mapping student_id -> {"name": str, "max_score": float}
    Takes the maximum cosine similarity across all stored face vectors for each student.
    """
    if not gallery:
        return {}
    student_scores = {}
    for entry in gallery:
        sid = entry["student_id"]
        score = cosine_similarity(query_embedding, entry["embedding"])
        if sid not in student_scores or score > student_scores[sid]["max_score"]:
            student_scores[sid] = {"name": entry["name"], "max_score": score}
    return student_scores


def compute_student_similarities_vectorized(query_embeddings: list, gallery: list):
    """
    Vectorized matrix BLAS dot-product search for 1,000s of student embeddings.
    M query faces x N gallery vectors matrix multiplication in a single pass.
    """
    if not query_embeddings or not gallery:
        return [{} for _ in query_embeddings]

    Q = np.vstack([q.reshape(1, -1) for q in query_embeddings])  # Shape (M, 512)
    G = np.vstack([entry["embedding"].reshape(1, -1) for entry in gallery])  # Shape (N, 512)

    # Compute dot products (M, N)
    S = np.dot(Q, G.T)

    results = []
    for face_idx in range(len(query_embeddings)):
        scores_row = S[face_idx]
        student_scores = {}
        for g_idx, entry in enumerate(gallery):
            sid = entry["student_id"]
            score = float(scores_row[g_idx])
            if sid not in student_scores or score > student_scores[sid]["max_score"]:
                student_scores[sid] = {"name": entry["name"], "max_score": score}
        results.append(student_scores)
    return results

