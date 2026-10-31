# Classical Vehicle Tracker

A vehicle motion tracker built with classical computer vision (no deep learning). Moving vehicles are detected with background subtraction, then tracked frame by frame using three classical approaches that can be compared side by side:

- Lucas-Kanade sparse optical flow
- ORB feature matching with Hamming distance
- SIFT feature matching with L2 distance

The two feature-matching trackers use Lowe's ratio test and RANSAC to remove bad matches.

## Demo

![Vehicle tracking demo](outputs/tracking_demo.gif)

The demo shows the Lucas-Kanade tracker. Blue boxes are detected vehicles, red dots are tracked points, green trails show recent point motion, and each yellow arrow with its label shows that vehicle's displacement in pixels per frame. The ORB and SIFT trackers draw the same overlay.

## Result videos

The full annotated video for each method is saved in the `outputs` folder. GitHub cannot preview MP4 files in its file viewer, so these links open or download the raw video file:

- [Lucas-Kanade optical flow](outputs/tracked_lk.mp4?raw=true)
- [ORB feature matching](outputs/tracked_orb.mp4?raw=true)
- [SIFT feature matching](outputs/tracked_sift.mp4?raw=true)

## How it works

All three trackers share the same detection stage, then differ in how they follow points between frames.

**1. Vehicle detection (shared).** A MOG2 background subtractor finds moving pixels. Morphological opening and closing clean up the mask, and contours above a minimum area become vehicle bounding boxes.

**2. Point tracking (one of three methods).**

| Method | Keypoints | Correspondence | Outlier removal |
|---|---|---|---|
| `lk` | Shi-Tomasi corners inside vehicle boxes | Pyramidal Lucas-Kanade optical flow (15 x 15 window, 2 pyramid levels) | Forward-backward error check (1.0 px) |
| `orb` | ORB keypoints in moving regions | Brute-force matching, Hamming distance | Lowe's ratio test (0.75), then RANSAC per vehicle |
| `sift` | SIFT keypoints in moving regions | Brute-force matching, L2 norm | Lowe's ratio test (0.75), then RANSAC per vehicle |

For the feature-matching methods, keypoints are detected only inside the dilated moving-region mask, so the static background is ignored. Descriptors from the previous frame are matched to the current frame. A match is kept only if its best distance is clearly smaller than the second-best distance (Lowe's ratio test). Then, for each vehicle box, RANSAC fits a similarity transform with `estimateAffinePartial2D` and keeps only the inlier matches.

**3. Displacement.** For each vehicle, the displacement is the median motion of the surviving points inside its box, drawn as an arrow and labelled in pixels per frame.

## Project structure

```
classical-vehicle-tracker/
    vehicle_tracker.py
    README.md
    requirements.txt
    videos/
        traffic.mp4
    outputs/
        tracked_lk.mp4
        tracked_orb.mp4
        tracked_sift.mp4
        tracking_demo.gif
```

## Setup

Requires Python 3.9 or newer. SIFT needs OpenCV 4.4 or newer.

```
py -m pip install -r requirements.txt
```

`requirements.txt` contains `opencv-python`, `numpy`, `imageio`, `imageio-ffmpeg` and `pillow`. The `imageio-ffmpeg` package bundles the encoder used to write H.264 video, so ffmpeg does not need to be installed separately.

## Usage

Place the input video at `videos/traffic.mp4`, then run:

```
py vehicle_tracker.py
```

This runs the three methods one after another. Each one writes an annotated video to the `outputs` folder (`tracked_lk.mp4`, `tracked_orb.mp4` and `tracked_sift.mp4`). A single GIF, `tracking_demo.gif`, is made from the Lucas-Kanade run. To run fewer methods, edit the `METHODS` list near the top of `vehicle_tracker.py`, for example `METHODS = ["lk"]` for Lucas-Kanade only. Frames are processed at 960 px wide for speed, whatever the source resolution. On the test machine the full 699-frame clip took about 52 s for Lucas-Kanade, 74 s for ORB and 148 s for SIFT, so SIFT is the slowest of the three. Times on other machines will differ.

## Configuration

Settings are constants at the top of `vehicle_tracker.py`:

| Constant | Default | Effect |
|---|---|---|
| `PROCESS_WIDTH` | 960 | Width frames are resized to before processing |
| `RESEED_EVERY` | 5 | Lucas-Kanade only: frames between adding new points |
| `FB_THRESHOLD` | 1.0 | Lucas-Kanade only: maximum forward-backward error (pixels) |
| `MAX_FEATURES` | 1500 | ORB and SIFT: maximum keypoints per frame |
| `RATIO_THRESHOLD` | 0.75 | ORB and SIFT: Lowe's ratio test cutoff (lower is stricter) |
| `RANSAC_THRESHOLD` | 3.0 | ORB and SIFT: RANSAC inlier distance (pixels) |
| `MIN_MATCHES` | 4 | ORB and SIFT: matches needed in a box before RANSAC runs |
| `MIN_AREA_RATIO` | 0.0008 | Smallest blob (fraction of the frame) counted as a vehicle |
| `TRAIL_FADE` | 0.9 | How quickly trails fade (lower fades faster) |
| `METHODS` | `["lk", "orb", "sift"]` | Which trackers to run |
| `GIF_METHOD` | `"lk"` | Which method the demo GIF is made from |
| `GIF_STEP` / `GIF_WIDTH` / `MAX_GIF_FRAMES` | 4 / 480 / 70 | GIF frame sampling, width and length |

If vehicles are missed, lower `MIN_AREA_RATIO` or the `varThreshold` value in the MOG2 call. If one vehicle is split into two boxes, raise the closing `iterations` in `vehicle_mask`.

## Limitations

- Assumes a static camera. Camera shake or panning makes the whole background look like motion.
- Boxes are detected independently in each frame, so vehicles do not keep a persistent ID across frames.
- In dense traffic, MOG2 can split one large vehicle into two boxes or merge two close vehicles into one.
- Vehicles that stop for a long time are gradually absorbed into the background and stop being detected.
- Small or distant vehicles have few distinctive keypoints, so feature matching is less reliable on them than on large nearby ones.
- Displacement is in pixels per frame, not real-world speed. Converting to km/h would need camera calibration.

## Tools

Python, OpenCV, NumPy, imageio, Pillow.
