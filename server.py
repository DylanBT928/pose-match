import io
import math
import threading
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

# ---- Model & constants ----
try:
    cv2.setNumThreads(0)
except Exception:
    pass

MODEL = YOLO("yolov8n-pose.pt")
IMG_SIZE = 320
CONF = 0.15

# 17-keypoint skeleton (COCO order)
SKELETON = [
    (5,7),(7,9),(6,8),(8,10),
    (11,13),(13,15),(12,14),(14,16),
    (5,6),(11,12),(5,11),(6,12)
]

app = Flask(__name__)
CORS(app)

# In-memory target state
state_lock = threading.Lock()
target_norm = None

# ---------- Keypoint helpers ----------
def _to_np(x):
    try: return x.detach().cpu().numpy()
    except Exception:
        try: return x.cpu().numpy()
        except Exception: return np.array(x)

def best_keypoints(pred):
    """Pick the best person by either box conf or keypoint conf."""
    kobj = getattr(pred, "keypoints", None)
    if kobj is None or len(kobj) == 0:
        return None
    kxy_all = kobj.xy
    if kxy_all is None or len(kxy_all) == 0:
        return None

    confs = None
    boxes = getattr(pred, "boxes", None)
    if boxes is not None and getattr(boxes, "conf", None) is not None:
        confs = _to_np(boxes.conf)
    if (confs is None or len(confs) == 0) and getattr(kobj, "conf", None) is not None:
        try:
            confs = np.nanmean(_to_np(kobj.conf), axis=1)
        except Exception:
            confs = None

    idx = int(np.nanargmax(confs)) if (isinstance(confs, np.ndarray) and np.size(confs) > 0) else 0
    kp = _to_np(kxy_all[idx])
    # Expect (17,2)
    if kp.ndim != 2 or kp.shape[1] != 2:
        return None
    return kp

def detect_keypoints_rgb(img_rgb: np.ndarray):
    """Returns absolute pixel keypoints (17,2) or None."""
    h, w = img_rgb.shape[:2]
    if h == 0 or w == 0:
        return None
    scale = IMG_SIZE / max(h, w)
    if scale < 1.0:
        small = cv2.resize(img_rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        small, scale = img_rgb, 1.0
    pred = MODEL.predict(small, imgsz=IMG_SIZE, conf=CONF, verbose=False)[0]
    kp = best_keypoints(pred)
    if kp is None:
        return None
    return kp / scale

def normalize_keypoints(pts):
    """Center on mid-hips, scale by shoulder-hip average distance."""
    if pts is None or not np.isfinite(pts).all():
        return None
    # hips: 11,12; shoulders: 5,6
    if not (np.isfinite(pts[11]).all() and np.isfinite(pts[12]).all()):
        return None
    center = (pts[11] + pts[12]) / 2.0
    p = pts - center
    if not (np.isfinite(pts[5]).all() and np.isfinite(pts[6]).all()):
        return None
    scale = 0.5 * (np.linalg.norm(pts[5] - pts[11]) + np.linalg.norm(pts[6] - pts[12]))
    if not np.isfinite(scale) or scale < 1e-6:
        return None
    return p / scale

def pose_score(a, b):
    """Return 0-100 similarity score."""
    a = a.reshape(-1); b = b.reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    cos = (a @ b) / denom if denom > 0 else 0.0
    cos = float(np.clip(cos, -1, 1))
    l2 = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-6))
    s = 0.6 * ((cos + 1) / 2) + 0.4 * (1 / (1 + l2))
    return float(np.clip(100 * s, 0, 100))

def read_frame_rgb(file_storage):
    """Decode uploaded image -> RGB np array + (w,h)."""
    data = np.frombuffer(file_storage.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, 0, 0
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return rgb, w, h

def kp_to_list(kp):
    return None if kp is None else [[float(x), float(y)] for x, y in kp]

# ---------- Routes ----------
@app.post("/infer")
def infer():
    """
    POST form-data:
      - frame: image blob (UN-MIRRORED)
      - mirror: 'true'/'false' (ignored since frame is unmirrored)
    Returns: { keypoints: [[x,y]..] | null, score: float|null, message: str }
    """
    file = request.files.get("frame", None)
    if file is None:
        return jsonify(error="no frame"), 400

    img_rgb, w, h = read_frame_rgb(file)
    if img_rgb is None:
        return jsonify(keypoints=None, score=None, message="Bad image")

    kp = detect_keypoints_rgb(img_rgb)
    live_norm = normalize_keypoints(kp) if kp is not None else None

    with state_lock:
        tgt = None if target_norm is None else target_norm.copy()

    score = None
    if (tgt is not None) and (live_norm is not None):
        score = pose_score(live_norm, tgt)

    return jsonify(
        keypoints=kp_to_list(kp),
        score=score,
        message=("OK" if kp is not None else "No person detected")
    )

@app.post("/set_target_from_frame")
def set_target_from_frame():
    """
    POST form-data: frame (UN-MIRRORED)
    Sets global target_norm. Returns {ok, message, keypoints, width, height}
    """
    global target_norm
    file = request.files.get("frame", None)
    if file is None:
        return jsonify(ok=False, message="no frame"), 400

    img_rgb, w, h = read_frame_rgb(file)
    if img_rgb is None:
        return jsonify(ok=False, message="Bad image"), 400

    kp = detect_keypoints_rgb(img_rgb)
    if kp is None:
        return jsonify(ok=False, message="No person detected", keypoints=None, width=w, height=h)

    norm = normalize_keypoints(kp)
    if norm is None:
        return jsonify(ok=False, message="Pose not stable", keypoints=None, width=w, height=h)

    with state_lock:
        target_norm = norm.astype(np.float32)

    return jsonify(ok=True, message="Target set", keypoints=kp_to_list(kp), width=w, height=h)

@app.post("/set_target_from_upload")
def set_target_from_upload():
    """
    POST form-data: image
    Works for uploaded (non-mirrored) files.
    """
    global target_norm
    file = request.files.get("image", None)
    if file is None:
        return jsonify(ok=False, message="no image"), 400

    img_rgb, w, h = read_frame_rgb(file)
    if img_rgb is None:
        return jsonify(ok=False, message="Bad image"), 400

    kp = detect_keypoints_rgb(img_rgb)
    if kp is None:
        return jsonify(ok=False, message="No person detected", keypoints=None, width=w, height=h)

    norm = normalize_keypoints(kp)
    if norm is None:
        return jsonify(ok=False, message="Pose not stable", keypoints=None, width=w, height=h)

    with state_lock:
        target_norm = norm.astype(np.float32)

    return jsonify(ok=True, message="Target set", keypoints=kp_to_list(kp), width=w, height=h)

@app.post("/clear_target")
def clear_target():
    global target_norm
    with state_lock:
        target_norm = None
    return jsonify(ok=True, message="Target cleared")

if __name__ == "__main__":
    # Run on localhost:5000
    app.run(host="127.0.0.1", port=5000, threaded=True)
