import os
import cv2
import numpy as np
import imageio.v2 as imageio
from PIL import Image

INPUT_VIDEO = "videos/video.mp4"
OUTPUT_DIR = "outputs"

VEHICLES = [
    (0, (0, 215, 170, 90)),
    (90, (343, 237, 63, 42)),
    (360, (204, 230, 136, 66)),
    (450, (319, 244, 69, 35)),
]

PROCESS_WIDTH = 960
SEARCH_MARGIN = 0.4
LOST_LIMIT = 8
MIN_VISIBLE = 0.4
RESEED_EVERY = 5
MIN_POINTS = 15
MAX_POINTS = 80
FB_THRESHOLD = 1.0
TRAIL_FADE = 0.9
MAX_FEATURES = 1500
RATIO_THRESHOLD = 0.75
RANSAC_THRESHOLD = 3.0
MIN_MATCHES = 4
GIF_STEP = 4
GIF_FRAME_MS = 80
GIF_WIDTH = 480
MAX_GIF_FRAMES = 70

METHODS = ["lk", "orb", "sift"]
GIF_METHOD = "lk"

feature_params = dict(maxCorners=MAX_POINTS, qualityLevel=0.05, minDistance=5, blockSize=7)
lk_params = dict(
    winSize=(15, 15),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)


def box_mask(shape, box, margin=0.0):
    x, y, w, h = box
    x0 = int(max(0, x - w * margin))
    y0 = int(max(0, y - h * margin))
    x1 = int(min(shape[1], x + w * (1 + margin)))
    y1 = int(min(shape[0], y + h * (1 + margin)))
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 255
    return mask


def inside_box(points, box, margin):
    x, y, w, h = box
    return (
        (points[:, 0] >= x - w * margin) & (points[:, 0] <= x + w * (1 + margin))
        & (points[:, 1] >= y - h * margin) & (points[:, 1] <= y + h * (1 + margin))
    )


def seed_points(gray, box, existing):
    mask = box_mask(gray.shape, box)
    for px, py in existing.reshape(-1, 2):
        cv2.circle(mask, (int(px), int(py)), 6, 0, -1)
    found = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
    if found is None:
        return existing
    return np.concatenate([existing, found], axis=0)[:MAX_POINTS]


def track_lk(prev_gray, gray, p0):
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **lk_params)
    fb_error = np.linalg.norm(p0 - back, axis=2).reshape(-1)
    good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb_error < FB_THRESHOLD)
    return p0[good].reshape(-1, 2), p1[good].reshape(-1, 2)


def create_features(method):
    if method == "orb":
        detector = cv2.ORB_create(nfeatures=MAX_FEATURES, fastThreshold=10)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        detector = cv2.SIFT_create(nfeatures=MAX_FEATURES)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    return detector, matcher


def match_features(matcher, prev_kp, prev_des, kp, des):
    empty = np.zeros((0, 2), dtype=np.float32)
    if prev_des is None or des is None or len(prev_des) < 2 or len(des) < 2:
        return empty, empty
    src, dst = [], []
    for pair in matcher.knnMatch(prev_des, des, k=2):
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < RATIO_THRESHOLD * second.distance:
            src.append(prev_kp[best.queryIdx].pt)
            dst.append(kp[best.trainIdx].pt)
    if not src:
        return empty, empty
    return np.float32(src), np.float32(dst)


def keep_inside(kp, des, box, margin):
    if des is None or len(kp) == 0:
        return [], None
    pts = np.float32([k.pt for k in kp])
    idx = np.where(inside_box(pts, box, margin))[0]
    if len(idx) == 0:
        return [], None
    return [kp[i] for i in idx], des[idx]


def new_track(method, number, box, gray, detector):
    track = {
        "id": number,
        "box": [float(v) for v in box],
        "lost": 0,
        "active": True,
        "pts": np.zeros((0, 1, 2), dtype=np.float32),
        "kp": [],
        "des": None,
    }
    if method == "lk":
        track["pts"] = seed_points(gray, box, track["pts"])
    else:
        kp, des = detector.detectAndCompute(gray, box_mask(gray.shape, box))
        track["kp"], track["des"] = kp, des
    return track


def update_track(track, method, prev_gray, gray, detector, matcher, frame_index):
    empty = np.zeros((0, 2), dtype=np.float32)
    old, new = empty, empty
    kp, des = [], None

    if method == "lk":
        if len(track["pts"]) > 0:
            old, new = track_lk(prev_gray, gray, track["pts"])
    else:
        mask = box_mask(gray.shape, track["box"], SEARCH_MARGIN)
        kp, des = detector.detectAndCompute(gray, mask)
        old, new = match_features(matcher, track["kp"], track["des"], kp, des)

    motion = None
    if len(old) >= MIN_MATCHES:
        motion, inliers = cv2.estimateAffinePartial2D(
            old, new, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD
        )
        if motion is not None:
            keep = inliers.ravel() == 1
            old, new = old[keep], new[keep]
            if len(old) < MIN_MATCHES:
                motion = None

    shift = (0.0, 0.0)
    if motion is None:
        track["lost"] += 1
        old, new = empty, empty
    else:
        track["lost"] = 0
        x, y, w, h = track["box"]
        cx, cy = x + w / 2, y + h / 2
        ncx = motion[0, 0] * cx + motion[0, 1] * cy + motion[0, 2]
        ncy = motion[1, 0] * cx + motion[1, 1] * cy + motion[1, 2]
        scale = float(np.clip(np.hypot(motion[0, 0], motion[1, 0]), 0.9, 1.1))
        w, h = w * scale, h * scale
        track["box"] = [ncx - w / 2, ncy - h / 2, w, h]
        shift = (ncx - cx, ncy - cy)

        if method == "lk":
            pts = new[inside_box(new, track["box"], 0.1)].reshape(-1, 1, 2)
            if frame_index % RESEED_EVERY == 0 or len(pts) < MIN_POINTS:
                pts = seed_points(gray, track["box"], pts)
            track["pts"] = pts.astype(np.float32)
        else:
            track["kp"], track["des"] = keep_inside(kp, des, track["box"], 0.1)

    x, y, w, h = track["box"]
    height, width = gray.shape
    visible_w = max(0.0, min(x + w, width) - max(x, 0.0))
    visible_h = max(0.0, min(y + h, height) - max(y, 0.0))
    visible = (visible_w * visible_h) / max(w * h, 1.0)
    if track["lost"] >= LOST_LIMIT or w < 10 or h < 8 or visible < MIN_VISIBLE:
        track["active"] = False
    return old, new, shift


def run(method):
    out_video = f"{OUTPUT_DIR}/tracked_{method}.mp4"
    out_gif = f"{OUTPUT_DIR}/tracking_demo.gif"
    make_gif = len(METHODS) == 1 or method == GIF_METHOD

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print("Error: Could not open video:", INPUT_VIDEO)
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    width = PROCESS_WIDTH
    height = int(src_h * PROCESS_WIDTH / src_w)
    height -= height % 2
    gif_size = (GIF_WIDTH, int(height * GIF_WIDTH / width))

    try:
        out = imageio.get_writer(
            out_video,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
            ffmpeg_params=["-crf", "26", "-preset", "fast"],
        )
    except Exception as error:
        print("Error: could not start the video writer:", error)
        print("Run: py -m pip install imageio-ffmpeg")
        cap.release()
        return False

    detector, matcher = (None, None) if method == "lk" else create_features(method)
    tracks = []
    prev_gray = None
    trails = np.zeros((height, width, 3), dtype=np.uint8)

    gif_frames = []
    frame_index = -1
    print(f"[{method.upper()}] Tracking started...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        trails = (trails * TRAIL_FADE).astype(np.uint8)
        results = []

        for track in tracks:
            if track["active"]:
                old, new, shift = update_track(
                    track, method, prev_gray, gray, detector, matcher, frame_index
                )
                results.append((track, old, new, shift))

        for number, (start, box) in enumerate(VEHICLES, start=1):
            if start == frame_index:
                track = new_track(method, number, box, gray, detector)
                tracks.append(track)
                results.append((track, np.zeros((0, 2)), np.zeros((0, 2)), (0.0, 0.0)))

        for track, old, new, shift in results:
            for (xo, yo), (xn, yn) in zip(old, new):
                cv2.line(trails, (int(xo), int(yo)), (int(xn), int(yn)), (0, 255, 0), 2)
                cv2.circle(frame, (int(xn), int(yn)), 3, (0, 0, 255), -1)

        frame = cv2.add(frame, trails)

        for track, old, new, shift in results:
            if not track["active"]:
                continue
            x, y, w, h = [int(v) for v in track["box"]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
            speed = float(np.hypot(shift[0], shift[1]))
            cx, cy = x + w // 2, y + h // 2
            tip = (int(cx + shift[0] * 5), int(cy + shift[1] * 5))
            cv2.arrowedLine(frame, (cx, cy), tip, (0, 255, 255), 2, tipLength=0.4)
            cv2.putText(
                frame, f"V{track['id']} {speed:.1f} px/f", (x, max(y - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.append_data(rgb)

        if make_gif and frame_index % GIF_STEP == 0 and len(gif_frames) < MAX_GIF_FRAMES:
            gif_frames.append(cv2.resize(rgb, gif_size))

        prev_gray = gray

    cap.release()
    out.close()

    if make_gif:
        if gif_frames:
            pil_frames = [Image.fromarray(f) for f in gif_frames]
            pil_frames[0].save(
                out_gif,
                save_all=True,
                append_images=pil_frames[1:],
                duration=GIF_FRAME_MS,
                loop=0,
            )
            print(f"[{method.upper()}] GIF saved:", out_gif)
        else:
            print(f"[{method.upper()}] No GIF frames were generated.")

    print(f"[{method.upper()}] Video saved:", out_video)
    print(f"[{method.upper()}] Total frames processed:", frame_index + 1)
    return True


for method in METHODS:
    if method not in ("lk", "orb", "sift"):
        raise SystemExit("METHODS can only contain 'lk', 'orb' and 'sift'")

os.makedirs(OUTPUT_DIR, exist_ok=True)

for method in METHODS:
    if not run(method):
        raise SystemExit

print("Tracking completed!")