# Classical Vehicle Tracker

A vehicle motion tracker built with classical computer vision (no deep learning). It detects moving vehicles in a traffic video using background subtraction, then tracks their motion frame by frame using Lucas-Kanade sparse optical flow.

![Vehicle tracking demo](outputs/tracking_demo.gif)

## What it does

For every frame of the input video, the tracker:

1. Detects moving vehicles with MOG2 background subtraction and draws a bounding box around each one.
2. Picks trackable corner features (Shi-Tomasi) inside those boxes only, so the background is not tracked.
3. Tracks the features from the previous frame to the current one with pyramidal Lucas-Kanade optical flow.
4. Rejects unreliable tracks using a forward-backward consistency check.
5. Estimates each vehicle's displacement as the median motion of the points inside its box, drawn as an arrow with a label in pixels per frame.
6. Draws fading green trails showing recent point motion.

The result is saved as an annotated video and a short GIF.

## How it works

| Stage | Technique | Purpose |
|---|---|---|
| Vehicle detection | MOG2 background subtractor, morphological open and close, contour area filter | Find moving blobs and turn them into bounding boxes |
| Feature selection | Shi-Tomasi corners (`goodFeaturesToTrack`) restricted to the boxes | Choose points worth tracking on vehicles |
| Motion tracking | Pyramidal Lucas-Kanade optical flow | Follow each point between consecutive frames |
| Outlier rejection | Forward-backward error check (threshold 1.0 px) | Drop points whose flow is not reversible |
| Displacement | Median of per-point motion inside each box | A robust per-vehicle motion estimate |

Lucas-Kanade assumes that pixel intensity stays roughly constant and that motion is small within a local window (15 x 15 here, with 2 pyramid levels to handle larger motion).

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

Requires Python 3.9 or newer.

```
py -m pip install opencv-python numpy imageio imageio-ffmpeg pillow
```

`imageio-ffmpeg` bundles the ffmpeg encoder used to write the H.264 output video, so ffmpeg does not need to be installed separately.

## Usage

Place the input video at `videos/traffic.mp4`, then run:

```
py vehicle_tracker.py
```

When it finishes, the results are in the `outputs` folder:

- `tracked_lk.mp4`, `tracked_orb.mp4` and `tracked_sift.mp4`: the annotated videos for each method
- `tracking_demo.gif`: a short preview of the Lucas-Kanade run, used in this README

The script processes frames at 960 px wide for speed, regardless of the source resolution.

## Outputs

Everything the script produces is saved in the [`outputs`](outputs/) folder:

| File | What it is |
|---|---|
| [`tracking_demo.gif`](outputs/tracking_demo.gif) | About 5.6 s preview of the Lucas-Kanade run, shown at the top of this page |
| [`tracked_lk.mp4`](outputs/tracked_lk.mp4?raw=true) | Full annotated video from the Lucas-Kanade tracker |
| [`tracked_orb.mp4`](outputs/tracked_orb.mp4?raw=true) | Full annotated video from the ORB tracker |
| [`tracked_sift.mp4`](outputs/tracked_sift.mp4?raw=true) | Full annotated video from the SIFT tracker |

GitHub cannot preview MP4 files in its file viewer, so the video links open or download the raw file. The GIF is made from the method set in `GIF_METHOD` (Lucas-Kanade by default).

## Configuration

Settings are constants at the top of `vehicle_tracker.py`:

| Constant | Default | Effect |
|---|---|---|
| `PROCESS_WIDTH` | 960 | Width frames are resized to before processing |
| `RESEED_EVERY` | 5 | Frames between adding new feature points |
| `FB_THRESHOLD` | 1.0 | Maximum forward-backward error (pixels) to keep a point |
| `TRAIL_FADE` | 0.9 | How quickly the green trails fade (lower fades faster) |
| `MIN_AREA_RATIO` | 0.0008 | Smallest blob (as a fraction of the frame) counted as a vehicle |
| `GIF_STEP` / `GIF_WIDTH` / `MAX_GIF_FRAMES` | 4 / 480 / 70 | GIF frame sampling, width, and length |

If vehicles are missed, lower `MIN_AREA_RATIO` or the `varThreshold` value in the MOG2 call. If one vehicle is split into two boxes, raise the closing `iterations` in `vehicle_mask`.

## Limitations

- Assumes a static camera. Camera shake or panning makes the whole background look like motion.
- Boxes are detected independently in each frame, so vehicles do not keep a persistent ID across frames.
- In dense traffic, MOG2 can split one large vehicle into two boxes or merge two close vehicles into one.
- Vehicles that stop for a long time are gradually absorbed into the background and stop being detected.
- Displacement is reported in pixels per frame, not real-world speed. Converting to km/h would need camera calibration.

## Related techniques

The demo uses Lucas-Kanade optical flow for tracking. The script also includes ORB and SIFT feature-matching trackers, which use Hamming or L2 distance, Lowe's ratio test, and RANSAC to filter bad matches. Which trackers run is set by the `METHODS` list near the top of `vehicle_tracker.py`.

## Tools

Python, OpenCV, NumPy, imageio, Pillow.
