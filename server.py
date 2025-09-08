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

# Skeleton colors for visualization (RGB tuples)
SKELETON_COLORS = [
    (255, 0, 0),    # (5,7) - left shoulder to left elbow - red
    (255, 128, 0),  # (7,9) - left elbow to left wrist - orange
    (0, 255, 0),    # (6,8) - right shoulder to right elbow - green
    (0, 255, 128),  # (8,10) - right elbow to right wrist - light green
    (255, 0, 255),  # (11,13) - left hip to left knee - magenta
    (255, 128, 255),# (13,15) - left knee to left ankle - light magenta
    (0, 0, 255),    # (12,14) - right hip to right knee - blue
    (128, 128, 255),# (14,16) - right knee to right ankle - light blue
    (255, 255, 0),  # (5,6) - shoulders - yellow
    (128, 255, 255),# (11,12) - hips - cyan
    (255, 0, 128),  # (5,11) - left shoulder to left hip - pink
    (0, 128, 255)   # (6,12) - right shoulder to right hip - light blue
]

# Keypoint colors (for the joints themselves)
KEYPOINT_COLORS = [
    (255, 0, 0),    # 0: nose - red
    (255, 85, 0),   # 1: left eye - orange
    (255, 170, 0),  # 2: right eye - yellow-orange
    (255, 255, 0),  # 3: left ear - yellow
    (170, 255, 0),  # 4: right ear - yellow-green
    (85, 255, 0),   # 5: left shoulder - green
    (0, 255, 0),    # 6: right shoulder - bright green
    (0, 255, 85),   # 7: left elbow - green-cyan
    (0, 255, 170),  # 8: right elbow - cyan-green
    (0, 255, 255),  # 9: left wrist - cyan
    (0, 170, 255),  # 10: right wrist - cyan-blue
    (0, 85, 255),   # 11: left hip - blue
    (0, 0, 255),    # 12: right hip - bright blue
    (85, 0, 255),   # 13: left knee - blue-purple
    (170, 0, 255),  # 14: right knee - purple-blue
    (255, 0, 255),  # 15: left ankle - magenta
    (255, 0, 170)   # 16: right ankle - magenta-red
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

def draw_skeleton_on_image(img_rgb, keypoints, thickness=3, keypoint_radius=5):
    """
    Draw colorful skeleton on image similar to the reference image.
    
    Args:
        img_rgb: Input image as RGB numpy array
        keypoints: (17,2) array of keypoint coordinates
        thickness: Line thickness for skeleton
        keypoint_radius: Radius for keypoint circles
    
    Returns:
        Image with skeleton drawn
    """
    if keypoints is None or len(keypoints) != 17:
        return img_rgb
    
    # Create a copy to avoid modifying original
    img_with_skeleton = img_rgb.copy()
    
    # Draw skeleton connections
    for i, (start_idx, end_idx) in enumerate(SKELETON):
        if (start_idx < len(keypoints) and end_idx < len(keypoints) and
            np.isfinite(keypoints[start_idx]).all() and np.isfinite(keypoints[end_idx]).all()):
            
            start_point = tuple(map(int, keypoints[start_idx]))
            end_point = tuple(map(int, keypoints[end_idx]))
            color = SKELETON_COLORS[i % len(SKELETON_COLORS)]
            
            # Convert RGB to BGR for OpenCV
            cv2_color = (color[2], color[1], color[0])
            cv2.line(img_with_skeleton, start_point, end_point, cv2_color, thickness)
    
    # Draw keypoints as circles
    for i, (x, y) in enumerate(keypoints):
        if np.isfinite([x, y]).all():
            center = (int(x), int(y))
            color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
            cv2_color = (color[2], color[1], color[0])  # RGB to BGR
            cv2.circle(img_with_skeleton, center, keypoint_radius, cv2_color, -1)
            # Add a black border for better visibility
            cv2.circle(img_with_skeleton, center, keypoint_radius, (0, 0, 0), 1)
    
    return img_with_skeleton

def encode_image_to_base64(img_rgb):
    """Convert RGB image to base64 encoded JPEG string."""
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    # Convert to base64
    import base64
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

# ---------- Routes ----------
@app.route("/infer", methods=["POST"])
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

@app.route("/set_target_from_frame", methods=["POST"])
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

@app.route("/set_target_from_upload", methods=["POST"])
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

@app.route("/clear_target", methods=["POST"])
def clear_target():
    global target_norm
    with state_lock:
        target_norm = None
    return jsonify(ok=True, message="Target cleared")

@app.route("/visualize", methods=["POST"])
def visualize():
    """
    POST form-data:
      - frame: image blob
    Returns: { image_base64: str, keypoints: [[x,y]..] | null, message: str, num_people: int }
    """
    file = request.files.get("frame", None)
    if file is None:
        return jsonify(error="no frame"), 400

    img_rgb, w, h = read_frame_rgb(file)
    if img_rgb is None:
        return jsonify(image_base64=None, keypoints=None, message="Bad image", num_people=0)

    # Use YOLO's built-in multi-person pose estimation
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    results = MODEL.predict(img_bgr, imgsz=IMG_SIZE, conf=CONF, verbose=False)
    
    if not results or len(results) == 0:
        return jsonify(image_base64=None, keypoints=None, message="No detection", num_people=0)
    
    result = results[0]
    
    # Get number of people detected
    num_people = 0
    all_keypoints = []
    
    if hasattr(result, 'keypoints') and result.keypoints is not None:
        kobj = result.keypoints
        if hasattr(kobj, 'xy') and kobj.xy is not None:
            num_people = len(kobj.xy)
            # Convert all keypoints to list format
            for i in range(num_people):
                kp = _to_np(kobj.xy[i])
                if kp.ndim == 2 and kp.shape[1] == 2:
                    all_keypoints.append(kp_to_list(kp))
    
    # Use YOLO's built-in visualization (handles multiple people automatically)
    annotated_img_bgr = result.plot(
        boxes=True,      # Show bounding boxes
        labels=True,     # Show labels
        conf=True,       # Show confidence scores
        line_width=2     # Skeleton line thickness
    )
    
    # Convert back to RGB and then to base64
    annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
    img_base64 = encode_image_to_base64(annotated_img_rgb)

    return jsonify(
        image_base64=img_base64,
        keypoints=all_keypoints if all_keypoints else None,
        num_people=num_people,
        message=(f"Detected {num_people} person(s)" if num_people > 0 else "No person detected")
    )

@app.route("/infer_multiperson", methods=["POST"])
def infer_multiperson():
    """
    POST form-data:
      - frame: image blob (UN-MIRRORED)
      - mirror: 'true'/'false' (ignored since frame is unmirrored)
    Returns: { all_keypoints: [[[x,y]..], [[x,y]..], ...], num_people: int, message: str }
    Multi-person version of /infer for real-time webcam use.
    """
    try:
        file = request.files.get("frame", None)
        if file is None:
            return jsonify(error="no frame"), 400

        img_rgb, w, h = read_frame_rgb(file)
        if img_rgb is None:
            return jsonify(all_keypoints=None, num_people=0, message="Bad image")

        # Use YOLO's multi-person detection (no visualization, just keypoints)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        results = MODEL.predict(img_bgr, imgsz=IMG_SIZE, conf=CONF, verbose=False)
        
        if not results or len(results) == 0:
            return jsonify(all_keypoints=None, num_people=0, message="No detection")
        
        result = results[0]
        
        # Get all keypoints for all people
        num_people = 0
        all_keypoints = []
        
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            kobj = result.keypoints
            if hasattr(kobj, 'xy') and kobj.xy is not None:
                num_people = len(kobj.xy)
                # Convert all keypoints to list format
                for i in range(num_people):
                    try:
                        kp = _to_np(kobj.xy[i])
                        if kp.ndim == 2 and kp.shape[1] == 2:
                            all_keypoints.append(kp_to_list(kp))
                    except Exception as e:
                        print(f"Error processing person {i}: {e}")
                        continue

        return jsonify(
            all_keypoints=all_keypoints if all_keypoints else None,
            num_people=num_people,
            message=(f"Detected {num_people} person(s)" if num_people > 0 else "No person detected")
        )
    
    except Exception as e:
        print(f"Error in infer_multiperson: {e}")
        import traceback
        traceback.print_exc()
        return jsonify(
            all_keypoints=None, 
            num_people=0, 
            message=f"Server error: {str(e)}"
        ), 500

@app.route("/visualize_single", methods=["POST"])
def visualize_single():
    """
    POST form-data:
      - frame: image blob
    Returns: { image_base64: str, keypoints: [[x,y]..] | null, message: str }
    Legacy endpoint that uses custom single-person visualization.
    """
    file = request.files.get("frame", None)
    if file is None:
        return jsonify(error="no frame"), 400

    img_rgb, w, h = read_frame_rgb(file)
    if img_rgb is None:
        return jsonify(image_base64=None, keypoints=None, message="Bad image")

    kp = detect_keypoints_rgb(img_rgb)
    
    # Draw skeleton on image (single person only)
    img_with_skeleton = draw_skeleton_on_image(img_rgb, kp)
    
    # Convert to base64
    img_base64 = encode_image_to_base64(img_with_skeleton)

    return jsonify(
        image_base64=img_base64,
        keypoints=kp_to_list(kp),
        message=("OK" if kp is not None else "No person detected")
    )

if __name__ == "__main__":
    # Run on localhost:5000
    app.run(host="127.0.0.1", port=5000, threaded=True)
