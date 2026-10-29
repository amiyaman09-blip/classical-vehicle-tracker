# Classical Vehicle Tracking & Feature Extraction

## 1. Project Overview

This project implements a classical computer vision-based vehicle motion tracker using Python and OpenCV. It processes traffic videos frame by frame to detect moving regions, track visual features, and estimate displacement without relying on deep learning models.

Three tracking approaches are implemented and compared:

- Lucas–Kanade (LK) sparse optical flow
- ORB feature matching
- SIFT feature matching

## 2. Objectives

- Process traffic videos using OpenCV.
- Detect moving regions using background subtraction.
- Track feature points across consecutive frames.
- Estimate movement using optical flow and feature matching.
- Filter unreliable correspondences using forward-backward validation, Lowe's ratio test, and RANSAC.
- Generate processed videos and an animated GIF to demonstrate the tracking results.

## 3. Technologies Used

- **Python** — Main programming language.
- **OpenCV** — Video processing, background subtraction, feature detection, optical flow, and motion estimation.
- **NumPy** — Array operations and coordinate calculations.
- **ImageIO** — Writing processed video output.
- **Pillow (PIL)** — Creating the animated GIF.

## 4. Tracking Methods

### 4.1 Lucas–Kanade Optical Flow

- Detects feature points using Shi–Tomasi corner detection.
- Tracks points between consecutive frames using `cv2.calcOpticalFlowPyrLK()`.
- Uses forward-backward validation to reject unreliable points.
- Draws movement trails and estimates displacement in pixels per frame.

### 4.2 ORB Feature Matching

- Detects and describes visual features using ORB.
- Matches binary descriptors using Hamming distance.
- Applies Lowe's ratio test to remove ambiguous matches.
- Uses RANSAC to filter inconsistent correspondences.

### 4.3 SIFT Feature Matching

- Detects distinctive keypoints and generates gradient-based descriptors.
- Matches descriptors using the L2 distance metric.
- Applies Lowe's ratio test and RANSAC-based verification.
- Estimates feature displacement between consecutive frames.

## 5. Vehicle Detection and Motion Estimation

The program uses the MOG2 background subtractor to identify moving regions. Morphological opening and closing help reduce noise in the foreground mask.

Contours are used to create bounding boxes around moving regions. Feature points within these regions are used to estimate displacement, which is visualized through motion trails, arrows, and pixel-per-frame labels.

**Note:** Background subtraction may detect moving objects other than vehicles. The reported displacement is image-space motion, not real-world speed in kilometres per hour.

## 6. Results and Visual Demonstrations

The following animated GIF demonstrates the Lucas–Kanade tracking output.

### 6.1 Lucas–Kanade Optical Flow

![Lucas–Kanade Optical Flow Tracking](outputs/tracking_demo_lk.gif)

Lucas–Kanade tracks selected feature points between consecutive frames and visualizes their movement using points and trails.

## 7. Output Files

| File | Description |
|---|---|
| `tracked_lk.mp4` | Processed video using Lucas–Kanade |
| `tracked_orb.mp4` | Processed video using ORB |
| `tracked_sift.mp4` | Processed video using SIFT |
| `tracking_demo_lk.gif` | Lucas–Kanade animated demonstration |

## 8. Project Structure

```text
classical-vehicle-tracker/
├── videos/
│   └── traffic.mp4
├── outputs/
│   ├── tracked_lk.mp4
│   ├── tracked_orb.mp4
│   ├── tracked_sift.mp4
│   └── tracking_demo_lk.gif
├── vehicle_tracker.py
└── README.md
```

## 9. Installation and Execution

### Step 1: Install Dependencies

Run the following command in the VS Code terminal:

```bash
py -m pip install opencv-python numpy imageio imageio-ffmpeg Pillow
```

### Step 2: Add the Input Video

Place the traffic video at:

```text
videos/traffic.mp4
```

### Step 3: Run the Program

Execute the following command from the project root directory:

```bash
py vehicle_tracker.py
```

The script processes the video using all three methods and saves the processed videos and the Lucas–Kanade animated GIF in the `outputs/` directory.

## 10. Limitations

- Background subtraction may produce false detections because of shadows, lighting changes, or background movement.
- Feature tracking can lose points when objects are occluded or move rapidly.
- Bounding boxes are based on moving regions rather than a dedicated vehicle classification model.
- Motion estimates are expressed in pixels per frame and are not calibrated real-world velocities.
- The methods do not currently maintain persistent identities for individual vehicles.

## 11. Future Improvements

- Introduce persistent vehicle IDs across frames.
- Improve separation of overlapping vehicles.
- Use camera calibration to estimate real-world displacement and speed.
- Compare the accuracy and computational efficiency of LK, ORB, and SIFT.
- Improve robustness under occlusion and changing illumination.

## 12. Conclusion

This project demonstrates how classical computer vision techniques can be used to track motion in traffic videos. Comparing Lucas–Kanade optical flow with ORB and SIFT feature matching provides practical insight into feature correspondence, motion estimation, and visual tracking.

The generated videos and animated GIF provide visual evidence of the implemented tracking approaches.