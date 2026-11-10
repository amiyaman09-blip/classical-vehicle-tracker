# Classical Vehicle Tracker

A vehicle motion tracking project built using classical computer vision techniques in Python and OpenCV, without relying on deep learning.

The project implements three tracking methods: **Lucas-Kanade (LK) Optical Flow, ORB, and SIFT**. It explores vehicle detection and motion tracking using both regular traffic footage and dashcam footage involving additional camera movement.

## Tracking Demonstrations

### 1. Regular Traffic Footage

The first implementation detects moving vehicles using background subtraction and tracks their movement across consecutive frames.

![Regular Traffic Tracking Demo](outputs/tracking_demo.gif)

### 2. Dashcam Footage

A second implementation was developed specifically for dashcam footage, where the camera itself is moving. It uses individually initialized vehicle bounding boxes and feature-based motion estimation to track vehicles across frames.

![Dashcam Tracking Demo](output2/tracking_demo.gif)

Both implementations support Lucas-Kanade, ORB, and SIFT, with a configurable option to choose which method's output is used to generate the animated GIF.

## What It Does

### Regular Traffic Tracker

For every frame of the input video, the tracker:

1. Detects moving regions using MOG2 background subtraction.
2. Applies morphological opening and closing to clean the foreground mask.
3. Identifies moving objects using contour detection and draws bounding boxes.
4. Detects feature points within the vehicle regions:
   - `lk`: Shi-Tomasi corner features.
   - `orb`: ORB keypoints with binary descriptors.
   - `sift`: SIFT keypoints with floating-point descriptors.
5. Tracks feature points between consecutive frames using the selected method.
6. Filters unreliable points using forward-backward validation, Lowe's ratio test, and RANSAC.
7. Estimates vehicle displacement using the median movement of tracked points.
8. Visualizes movement through bounding boxes, directional arrows, pixel-per-frame labels, and fading green trails.

### Dashcam Tracker

The second implementation handles footage where the camera itself is moving.

It:

1. Initializes individual vehicle tracks using specified starting frames and bounding boxes.
2. Detects feature points within each vehicle's search region.
3. Uses LK optical flow or ORB/SIFT feature matching to estimate movement.
4. Applies RANSAC-based affine motion estimation to filter unreliable matches.
5. Updates vehicle bounding boxes based on estimated movement and scale changes.
6. Maintains individual vehicle IDs and removes tracks that are lost or no longer sufficiently visible.
7. Displays bounding boxes, movement trails, directional arrows, and displacement estimates.

## How It Works

| Stage | Technique | Purpose |
|---|---|---|
| Vehicle detection | MOG2 background subtraction | Identify moving regions |
| Feature selection (LK) | Shi-Tomasi corners | Find suitable points to track |
| Feature selection (ORB/SIFT) | Keypoint and descriptor detection | Identify distinctive features |
| Motion tracking (LK) | Pyramidal Lucas-Kanade | Follow points between frames |
| Feature matching | BFMatcher with Hamming or L2 distance | Match corresponding features |
| Match filtering | Lowe's ratio test | Remove ambiguous matches |
| Motion estimation | RANSAC and affine transformation | Estimate reliable movement |
| Displacement | Median point displacement / affine transformation | Estimate vehicle movement |

## Tracking Methods

### 1. Lucas-Kanade (LK)

Uses sparse optical flow to track feature points between consecutive frames. It assumes approximately constant pixel intensity within a local neighbourhood.

- Uses Shi-Tomasi corner detection.
- Uses pyramidal optical flow to handle larger movements.
- Uses forward-backward validation to remove unreliable points.

### 2. ORB

ORB stands for Oriented FAST and Rotated BRIEF.

- Detects keypoints and generates binary descriptors.
- Uses Hamming distance for descriptor matching.
- Is relatively fast and suitable for real-time applications.

### 3. SIFT

SIFT stands for Scale-Invariant Feature Transform.

- Detects distinctive keypoints using scale-space analysis.
- Generates floating-point descriptors.
- Uses L2 distance for matching.
- Is more robust to scale and rotation changes, although computationally more expensive.

## Regular Traffic vs Dashcam Implementation

| Aspect | Regular Traffic | Dashcam |
|---|---|---|
| Camera | Relatively stationary | Moving camera |
| Vehicle initialization | Background subtraction | Specified starting frames and bounding boxes |
| Tracking methods | LK, ORB, SIFT | LK, ORB, SIFT |
| Motion estimation | Median feature displacement | RANSAC-based affine transformation |
| Vehicle IDs | No persistent IDs | Individual vehicle IDs |
| Main challenge | Vehicle detection and feature tracking | Tracking under camera movement |

## Project Structure

```text
classical-vehicle-tracker/
│
├── vehicle_tracker.py
├── vehicle_tracker2.py
├── README.md
├── requirements.txt
│
├── videos/
│   └── traffic.mp4
│
├── outputs/
│   ├── tracked_lk.mp4
│   ├── tracked_orb.mp4
│   ├── tracked_sift.mp4
│   └── tracking_demo.gif
│
├── video/
│   └── video.mp4
│
└── output2/
    ├── tracked_lk.mp4
    ├── tracked_orb.mp4
    ├── tracked_sift.mp4
    └── tracking_demo.gif
```

## Setup

Requires Python 3.9 or newer. SIFT requires OpenCV 4.4 or newer.

Install the required libraries:

```bash
py -m pip install opencv-python numpy imageio imageio-ffmpeg pillow
```

## Usage

### Regular Traffic Tracker

Place the input video inside the `videos` folder and run:

```bash
py vehicle_tracker.py
```

### Dashcam Tracker

Place the dashcam input video inside the `video` folder and run:

```bash
py vehicle_tracker2.py
```

Make sure the input path in the dashcam script matches the folder structure:

```python
INPUT_VIDEO = "video/video.mp4"
```

Both scripts generate their respective outputs in separate folders.

## GIF Configuration

Both implementations allow the user to configure which tracking method generates the animated GIF.

The following variables are used:

```python
METHODS = ["lk", "orb", "sift"]
GIF_METHOD = "lk"
```

- `METHODS` specifies which tracking methods will run.
- `GIF_METHOD` specifies which method's output will be converted into an animated GIF.

For example, setting `GIF_METHOD = "orb"` generates the GIF using ORB, while all three methods can still generate their respective processed videos.

## Outputs

### Regular Traffic

| File | Description |
|---|---|
| `tracked_lk.mp4` | Lucas-Kanade tracking output |
| `tracked_orb.mp4` | ORB tracking output |
| `tracked_sift.mp4` | SIFT tracking output |
| `tracking_demo.gif` | Animated tracking demonstration |

### Dashcam

| File | Description |
|---|---|
| `tracked_lk.mp4` | Lucas-Kanade dashcam tracking output |
| `tracked_orb.mp4` | ORB dashcam tracking output |
| `tracked_sift.mp4` | SIFT dashcam tracking output |
| `tracking_demo.gif` | Animated dashcam tracking demonstration |

## Important Parameters

| Parameter | Purpose |
|---|---|
| `PROCESS_WIDTH` | Controls processing resolution |
| `MAX_FEATURES` | Maximum number of ORB/SIFT features |
| `FB_THRESHOLD` | Maximum forward-backward error for LK |
| `RATIO_THRESHOLD` | Lowe's ratio test threshold |
| `RANSAC_THRESHOLD` | RANSAC reprojection error threshold |
| `RESEED_EVERY` | Frequency of adding new LK feature points |
| `TRAIL_FADE` | Controls how quickly movement trails fade |
| `GIF_METHOD` | Selects the tracking method used for GIF generation |

## Limitations

- Background subtraction can mistake other moving objects for vehicles.
- Dense traffic can cause overlapping or merged bounding boxes.
- Small or distant vehicles have fewer distinguishable features, making matching less reliable.
- Feature points can be lost when vehicles move significantly between frames.
- Camera movement in dashcam footage makes reliable motion estimation more challenging.
- Dashcam vehicle tracking currently requires manually specified initial bounding boxes and starting frames.
- Displacement is measured in pixels per frame rather than real-world speed. Camera calibration would be required to estimate actual vehicle speed.

## Areas for Improvement

- Implement automatic vehicle detection and initialization in dashcam footage.
- Improve camera motion compensation.
- Improve tracking when vehicles overlap or move close together.
- Improve detection of distant vehicles.
- Explore more robust tracking techniques under changing lighting and road conditions.

## Tools and Libraries

- Python
- OpenCV
- NumPy
- ImageIO
- Pillow

## Key Learning Outcomes

This project helped develop a practical understanding of classical computer vision techniques, including feature detection, descriptor matching, sparse optical flow, background subtraction, motion estimation, and outlier rejection.

Implementing three different tracking methods and testing them on both regular traffic and dashcam footage provided an opportunity to compare their performance and understand the challenges involved in vehicle motion tracking.
