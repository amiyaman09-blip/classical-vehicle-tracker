import os
import cv2
import numpy as np
import imageio.v2 as imageio
from PIL import Image

INPUT_VIDEO = "videos/traffic.mp4"
OUTPUT_DIR = "outputs"

PROCESS_WIDTH = 960
RESEED_EVERY = 5
FB_THRESHOLD = 1.0
TRAIL_FADE = 0.9
MIN_AREA_RATIO = 0.0008
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

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
feature_params = dict(maxCorners=100, qualityLevel=0.2, minDistance=7, blockSize=7)
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)


def vehicle_mask(subtractor, frame):
    fg = subtractor.apply(frame)
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    return fg


def find_boxes(fg, min_area):
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]


def seed_points(gray, boxes, existing):
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for x, y, w, h in boxes:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    for px, py in existing.reshape(-1, 2):
        cv2.circle(mask, (int(px), int(py)), 8, 0, -1)
    found = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
    if found is None:
        return existing
    return np.concatenate([existing, found], axis=0)


def track_lk(prev_gray, gray, p0):
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **lk_params)
    fb_error = np.linalg.norm(p0 - back, axis=2).reshape(-1)
    good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb_error < FB_THRESHOLD)
    return p0[good].reshape(-1, 2), p1[good].reshape(-1, 2)


def create_features(method):
    if method == "orb":
        detector = cv2.ORB_create(nfeatures=MAX_FEATURES)
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


def ransac_filter(src, dst, boxes):
    keep = np.zeros(len(dst), dtype=bool)
    for x, y, w, h in boxes:
        inside = (
            (dst[:, 0] >= x) & (dst[:, 0] <= x + w)
            & (dst[:, 1] >= y) & (dst[:, 1] <= y + h)
        )
        idx = np.where(inside)[0]
        if len(idx) < MIN_MATCHES:
            continue
        _, inliers = cv2.estimateAffinePartial2D(
            src[idx], dst[idx], method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_THRESHOLD,
        )
        if inliers is None:
            continue
        keep[idx[inliers.ravel() == 1]] = True
    return src[keep], dst[keep]


def run(method):
    make_gif = len(METHODS) == 1 or method == GIF_METHOD
    out_video = f"{OUTPUT_DIR}/tracked_{method}.mp4"
    out_gif = f"{OUTPUT_DIR}/tracking_demo.gif"

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
    min_area = MIN_AREA_RATIO * width * height

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

    def read_frame():
        ok, frame = cap.read()
        if not ok:
            return None
        return cv2.resize(frame, (width, height))

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=40, detectShadows=True
    )

    first = read_frame()
    if first is None:
        print("Error: Could not read the first frame.")
        cap.release()
        out.close()
        return False

    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    vehicle_mask(subtractor, first)

    p0 = np.zeros((0, 1, 2), dtype=np.float32)
    detector, matcher = (None, None) if method == "lk" else create_features(method)
    prev_kp, prev_des = None, None
    trails = np.zeros_like(first)

    gif_frames = []
    frame_count = 0
    print(f"[{method.upper()}] Tracking started...")

    while True:
        frame = read_frame()
        if frame is None:
            break
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fg = vehicle_mask(subtractor, frame)
        boxes = find_boxes(fg, min_area)
        fg_zone = cv2.dilate(fg, kernel, iterations=3)

        trails = (trails * TRAIL_FADE).astype(np.uint8)
        old_pts = np.zeros((0, 2), dtype=np.float32)
        new_pts = np.zeros((0, 2), dtype=np.float32)

        if method == "lk":
            if len(p0) > 0:
                old_pts, new_pts = track_lk(prev_gray, gray, p0)
                keep = np.zeros(len(new_pts), dtype=bool)
                for i, (x, y) in enumerate(new_pts):
                    xi, yi = int(x), int(y)
                    keep[i] = 0 <= xi < width and 0 <= yi < height and fg_zone[yi, xi] > 0
                old_pts, new_pts = old_pts[keep], new_pts[keep]
            p0 = new_pts.reshape(-1, 1, 2).astype(np.float32)
            if frame_count % RESEED_EVERY == 0 or len(p0) < 20:
                p0 = seed_points(gray, boxes, p0)
        else:
            kp, des = detector.detectAndCompute(gray, fg_zone)
            src, dst = match_features(matcher, prev_kp, prev_des, kp, des)
            if len(dst) > 0:
                old_pts, new_pts = ransac_filter(src, dst, boxes)
            prev_kp, prev_des = kp, des

        for (xo, yo), (xn, yn) in zip(old_pts, new_pts):
            cv2.line(trails, (int(xo), int(yo)), (int(xn), int(yn)), (0, 255, 0), 2)
            cv2.circle(frame, (int(xn), int(yn)), 3, (0, 0, 255), -1)

        frame = cv2.add(frame, trails)

        for x, y, w, h in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
            if len(new_pts) == 0:
                continue
            in_box = (
                (new_pts[:, 0] >= x) & (new_pts[:, 0] <= x + w)
                & (new_pts[:, 1] >= y) & (new_pts[:, 1] <= y + h)
            )
            if in_box.sum() < 3:
                continue
            shift = np.median(new_pts[in_box] - old_pts[in_box], axis=0)
            speed = float(np.hypot(shift[0], shift[1]))
            cx, cy = x + w // 2, y + h // 2
            tip = (int(cx + shift[0] * 5), int(cy + shift[1] * 5))
            cv2.arrowedLine(frame, (cx, cy), tip, (0, 255, 255), 2, tipLength=0.4)
            cv2.putText(
                frame, f"{speed:.1f} px/f", (x, max(y - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.append_data(rgb)

        if make_gif and frame_count % GIF_STEP == 0 and len(gif_frames) < MAX_GIF_FRAMES:
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
    print(f"[{method.upper()}] Total frames processed:", frame_count)
    return True


os.makedirs(OUTPUT_DIR, exist_ok=True)

for method in METHODS:
    if not run(method):
        raise SystemExit

print("Tracking completed!")