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

MODEL = YOLO("yolo11n-pose.pt")
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

# ---------- Helpers ----------
def _to_np(x):
    try: return x.detach().cpu().numpy()
    except Exception:
        try: return x.cpu().numpy()
        except Exception: return np.array(x)

def read_frame_rgb(file_storage):
    data = np.frombuffer(file_storage.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None: return None, 0, 0
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return rgb, w, h

def best_keypoints(pred):
    kobj = getattr(pred, "keypoints", None)
    if kobj is None or len(kobj) == 0: return None
    kxy_all = kobj.xy
    if kxy_all is None or len(kxy_all) == 0: return None

    confs = None
    boxes = getattr(pred, "boxes", None)
    if boxes is not None and getattr(boxes, "conf", None) is not None:
        confs = _to_np(boxes.conf)
    if (confs is None or len(confs) == 0) and getattr(kobj, "conf", None) is not None:
        try: confs = np.nanmean(_to_np(kobj.conf), axis=1)
        except Exception: confs = None

    idx = int(np.nanargmax(confs)) if (isinstance(confs, np.ndarray) and np.size(confs) > 0) else 0
    kp = _to_np(kxy_all[idx])
    if kp.ndim != 2 or kp.shape[1] != 2: return None
    return kp

def detect_keypoints_rgb(img_rgb: np.ndarray):
    h, w = img_rgb.shape[:2]
    if h == 0 or w == 0: return None
    scale = IMG_SIZE / max(h, w)
    if scale < 1.0:
        small = cv2.resize(img_rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        small, scale = img_rgb, 1.0
    pred = MODEL.predict(small, imgsz=IMG_SIZE, conf=CONF, verbose=False)[0]
    kp = best_keypoints(pred)
    if kp is None: return None
    return kp / scale

def normalize_keypoints(pts):
    if pts is None or not np.isfinite(pts).all(): return None
    # hips: 11,12; shoulders: 5,6
    if not (np.isfinite(pts[11]).all() and np.isfinite(pts[12]).all()): return None
    center = (pts[11] + pts[12]) / 2.0
    p = pts - center
    if not (np.isfinite(pts[5]).all() and np.isfinite(pts[6]).all()): return None
    scale = 0.5 * (np.linalg.norm(pts[5] - pts[11]) + np.linalg.norm(pts[6] - pts[12]))
    if not np.isfinite(scale) or scale < 1e-6: return None
    return p / scale

def pose_score(a, b):
    a = a.reshape(-1); b = b.reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    cos = (a @ b) / denom if denom > 0 else 0.0
    cos = float(np.clip(cos, -1, 1))
    l2 = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-6))
    s = 0.6 * ((cos + 1) / 2) + 0.4 * (1 / (1 + l2))
    return float(np.clip(100 * s, 0, 100))

def keypoint_accuracy_scores(live_norm, target_norm):
    if live_norm is None or target_norm is None: return None
    if live_norm.shape != target_norm.shape: return None
    distances = np.linalg.norm(live_norm - target_norm, axis=1)
    k = 1.0
    scores = 100.0 * np.exp(-k * distances)
    scores = np.clip(scores, 0, 100)
    return [float(s) for s in scores]

def kp_to_list(kp):  # legacy (unused for filtered)
    return None if kp is None else [[float(x), float(y)] for x, y in kp]

def filter_keypoints(kp_np, conf_np=None, w=None, h=None, conf_thresh=0.3, pad=0):
    """
    Return a list of [x,y] or None for each joint.
    Hides (0,0)/edge points, out-of-bounds, or low-confidence joints.
    """
    out = []
    for i in range(kp_np.shape[0]):
        x, y = float(kp_np[i, 0]), float(kp_np[i, 1])
        if not np.isfinite(x) or not np.isfinite(y):
            out.append(None); continue
        if conf_np is not None:
            c = float(np.squeeze(conf_np[i]))
            if not np.isfinite(c) or c < conf_thresh:
                out.append(None); continue
        if w is not None and h is not None:
            if x < -pad or y < -pad or x > (w + pad) or y > (h + pad):
                out.append(None); continue
            if x <= 1 or y <= 1 or x >= (w - 1) or y >= (h - 1):
                out.append(None); continue
        out.append([x, y])
    return out

def pick_person_index(result):
    try:
        if hasattr(result, "boxes") and getattr(result.boxes, "conf", None) is not None:
            confs = _to_np(result.boxes.conf)
            if isinstance(confs, np.ndarray) and confs.size > 0:
                return int(np.nanargmax(confs))
    except Exception:
        pass
    try:
        kobj = getattr(result, "keypoints", None)
        if kobj is not None and getattr(kobj, "conf", None) is not None:
            kconfs = _to_np(kobj.conf)
            means = np.nanmean(kconfs, axis=1)
            return int(np.nanargmax(means))
    except Exception:
        pass
    return 0

def encode_image_to_base64(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    import base64
    return base64.b64encode(buffer).decode('utf-8')

# ---------- Routes ----------
@app.route("/infer", methods=["POST"])
def infer():
    file = request.files.get("frame", None)
    if file is None: return jsonify(error="no frame"), 400

    img_rgb, w, h = read_frame_rgb(file)
    if img_rgb is None: return jsonify(keypoints=None, score=None, message="Bad image")

    kp = detect_keypoints_rgb(img_rgb)
    live_norm = normalize_keypoints(kp) if kp is not None else None

    with state_lock:
        tgt = None if target_norm is None else target_norm.copy()

    score = None
    if (tgt is not None) and (live_norm is not None):
        score = pose_score(live_norm, tgt)

    # Filter for display (no per-kp conf available here)
    kplist = None
    if kp is not None:
        kplist = filter_keypoints(kp, None, w, h, conf_thresh=0.0, pad=0)

    return jsonify(keypoints=kplist, score=score, message=("OK" if kp is not None else "No person detected"))

@app.route("/set_target_from_frame", methods=["POST"])
def set_target_from_frame():
    global target_norm
    file = request.files.get("frame", None)
    if file is None: return jsonify(ok=False, message="no frame"), 400

    img_rgb, w, h = read_frame_rgb(file)
    if img_rgb is None: return jsonify(ok=False, message="Bad image"), 400

    kp = detect_keypoints_rgb(img_rgb)
    if kp is None: return jsonify(ok=False, message="No person detected", keypoints=None, width=w, height=h)

    norm = normalize_keypoints(kp)
    if norm is None: return jsonify(ok=False, message="Pose not stable", keypoints=None, width=w, height=h)

    with state_lock:
        target_norm = norm.astype(np.float32)

    # Return filtered points for drawing
    return jsonify(ok=True, message="Target set", keypoints=filter_keypoints(kp, None, w, h, conf_thresh=0.0), width=w, height=h)

@app.route("/set_target_from_upload", methods=["POST"])
def set_target_from_upload():
    global target_norm
    file = request.files.get("image", None)
    if file is None: return jsonify(ok=False, message="no image"), 400

    img_rgb, w, h = read_frame_rgb(file)
    if img_rgb is None: return jsonify(ok=False, message="Bad image"), 400

    kp = detect_keypoints_rgb(img_rgb)
    if kp is None: return jsonify(ok=False, message="No person detected", keypoints=None, width=w, height=h)

    norm = normalize_keypoints(kp)
    if norm is None: return jsonify(ok=False, message="Pose not stable", keypoints=None, width=w, height=h)

    with state_lock:
        target_norm = norm.astype(np.float32)

    return jsonify(ok=True, message="Target set", keypoints=filter_keypoints(kp, None, w, h, conf_thresh=0.0), width=w, height=h)

@app.route("/clear_target", methods=["POST"])
def clear_target():
    global target_norm
    with state_lock:
        target_norm = None
    return jsonify(ok=True, message="Target cleared")

def _collect_people(result, w, h, conf_thresh=0.3):
    """Return (all_keypoints_filtered, raw_np_list)"""
    all_kp_vis = []
    all_kp_raw = []
    if hasattr(result, 'keypoints') and result.keypoints is not None and getattr(result.keypoints, 'xy', None) is not None:
        kobj = result.keypoints
        num = len(kobj.xy)
        for i in range(num):
            kp = _to_np(kobj.xy[i])              # (17,2)
            conf = None
            try:
                if getattr(kobj, 'conf', None) is not None:
                    conf = _to_np(kobj.conf[i])  # (17,) or (17,1)
            except Exception:
                conf = None
            all_kp_raw.append(kp)
            all_kp_vis.append(filter_keypoints(kp, conf, w, h, conf_thresh=conf_thresh, pad=0))
    return all_kp_vis, all_kp_raw

@app.route("/set_target_multiperson_frame", methods=["POST"])
def set_target_multiperson_frame():
    global target_norm
    file = request.files.get("frame", None)
    if file is None: return jsonify(ok=False, message="no frame"), 400

    img_rgb, w, h = read_frame_rgb(file)
    if img_rgb is None: return jsonify(ok=False, message="Bad image"), 400

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    results = MODEL.predict(img_bgr, imgsz=IMG_SIZE, conf=CONF, verbose=False)
    if not results: return jsonify(ok=False, message="No detection", all_keypoints=None, num_people=0, width=w, height=h)

    result = results[0]
    all_kp_vis, all_kp_raw = _collect_people(result, w, h, conf_thresh=0.3)
    num_people = len(all_kp_vis)
    if num_people == 0: return jsonify(ok=False, message="No people detected", all_keypoints=None, num_people=0, width=w, height=h)

    idx = pick_person_index(result)
    chosen_kp = all_kp_raw[idx]
    norm = normalize_keypoints(chosen_kp)
    if norm is None:
        return jsonify(ok=False, message="Pose not stable for target", all_keypoints=all_kp_vis, num_people=num_people, width=w, height=h, target_index=idx)

    with state_lock:
        target_norm = norm.astype(np.float32)

    return jsonify(ok=True, message=f"Target set from person {idx+1}", all_keypoints=all_kp_vis, num_people=num_people, width=w, height=h, target_index=idx)

@app.route("/set_target_multiperson_upload", methods=["POST"])
def set_target_multiperson_upload():
    global target_norm
    file = request.files.get("image", None)
    if file is None: return jsonify(ok=False, message="no image"), 400

    img_rgb, w, h = read_frame_rgb(file)
    if img_rgb is None: return jsonify(ok=False, message="Bad image"), 400

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    results = MODEL.predict(img_bgr, imgsz=IMG_SIZE, conf=CONF, verbose=False)
    if not results: return jsonify(ok=False, message="No detection", all_keypoints=None, num_people=0, width=w, height=h)

    result = results[0]
    all_kp_vis, all_kp_raw = _collect_people(result, w, h, conf_thresh=0.3)
    num_people = len(all_kp_vis)
    if num_people == 0: return jsonify(ok=False, message="No people detected", all_keypoints=None, num_people=0, width=w, height=h)

    idx = pick_person_index(result)
    chosen_kp = all_kp_raw[idx]
    norm = normalize_keypoints(chosen_kp)
    if norm is None:
        return jsonify(ok=False, message="Pose not stable for target", all_keypoints=all_kp_vis, num_people=num_people, width=w, height=h, target_index=idx)

    with state_lock:
        target_norm = norm.astype(np.float32)

    return jsonify(ok=True, message=f"Target set from image (person {idx+1})", all_keypoints=all_kp_vis, num_people=num_people, width=w, height=h, target_index=idx)

@app.route("/infer_multiperson", methods=["POST"])
def infer_multiperson():
    try:
        file = request.files.get("frame", None)
        if file is None: return jsonify(error="no frame"), 400

        img_rgb, w, h = read_frame_rgb(file)
        if img_rgb is None: return jsonify(all_keypoints=None, num_people=0, message="Bad image")

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        results = MODEL.predict(img_bgr, imgsz=IMG_SIZE, conf=CONF, verbose=False)
        if not results or len(results) == 0: return jsonify(all_keypoints=None, num_people=0, message="No detection")

        result = results[0]
        all_kp_vis, all_kp_raw = _collect_people(result, w, h, conf_thresh=0.3)
        num_people = len(all_kp_vis)

        all_scores = []
        all_keypoint_scores = []
        with state_lock:
            tgt = None if target_norm is None else target_norm.copy()

        if tgt is not None:
            for kp_raw in all_kp_raw:
                live_norm = normalize_keypoints(kp_raw)
                if live_norm is not None:
                    all_scores.append(pose_score(live_norm, tgt))
                    all_keypoint_scores.append(keypoint_accuracy_scores(live_norm, tgt))
                else:
                    all_scores.append(None)
                    all_keypoint_scores.append(None)
        else:
            all_scores = [None]*num_people
            all_keypoint_scores = [None]*num_people

        return jsonify(
            all_keypoints=all_kp_vis if all_kp_vis else None,
            all_scores=all_scores if any(s is not None for s in all_scores) else None,
            all_keypoint_scores=all_keypoint_scores if any(s is not None for s in all_keypoint_scores) else None,
            num_people=num_people,
            message=(f"Detected {num_people} person(s)" if num_people > 0 else "No person detected")
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(all_keypoints=None, num_people=0, message=f"Server error: {str(e)}"), 500

@app.route("/visualize_single", methods=["POST"])
def visualize_single():
    file = request.files.get("frame", None)
    if file is None: return jsonify(error="no frame"), 400

    img_rgb, w, h = read_frame_rgb(file)
    if img_rgb is None: return jsonify(image_base64=None, keypoints=None, message="Bad image")

    kp = detect_keypoints_rgb(img_rgb)
    # simple overlay using YOLO's plot for demo
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    result = MODEL.predict(img_bgr, imgsz=IMG_SIZE, conf=CONF, verbose=False)[0]
    annotated = result.plot(line_width=2)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    img_base64 = encode_image_to_base64(annotated_rgb)

    vis = filter_keypoints(kp, None, w, h, conf_thresh=0.0) if kp is not None else None
    return jsonify(image_base64=img_base64, keypoints=vis, message=("OK" if kp is not None else "No person detected"))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)
