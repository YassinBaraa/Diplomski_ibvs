# IBVS Pipeline

Image-Based Visual Servoing for UAV branch perching. Consumes detection pipeline output and computes pixel error to drive the UAV toward the locked branch point.

## What It Does

During the detection pipeline warmup, uses `best_candidate` as a soft visual target. After warmup it switches to model-free KLT optical flow tracking of FAST+Harris features near the locked `final_point` — no segmentation model runs here. The feature centroid drives the control law. Pixel error relative to frame center is yielded each frame for the UDP client.

`IBVSPipeline.run()` is a generator consumed by `UDP_client/main_record.py`.

---

## Pipeline Flow

```
DetectionPipelineSource.read()
  → frame, point (final_point xy or best_candidate xy)

FASTHarrisExtractor.extract(ctx)
  → FAST keypoints filtered by Harris response
  → post-warmup: radius-filtered around ctx.point

KLTTracker
  Pre-lock:  accumulate detection point, visualize features
  .lock()    — first post-warmup frame with enough features
  .update()  — every subsequent frame
    → LK optical flow on tracked features
    → re-extraction around last centroid if too many features drop out
    → estimated_point = locked_final_point + mean feature displacement
    → unlock() if estimated_point lost for 5 consecutive frames
       → pipeline reverts to detection mode and re-locks on new final_point

PointController.update_ctx(ctx)
  → error    = estimated_point - frame_center   (px)
  → velocity = -gain * error

yield IBVSContext
```

---

## Directory Structure

```
ibvs/
├── main.py
├── test_record.py
├── config/
│   └── __init__.py
├── sources/
│   ├── FrameSource.py
│   ├── MP4Source.py
│   ├── DetectionPipelineSource.py   # Wraps detection_pipeline generator (branch mode)
│   └── ArucoSource.py               # ArUco marker point source (aruco mode)
├── feature_extraction/
│   ├── FASTHarrisExtractor.py       # FAST detection + Harris corner filtering
│   ├── FeatureSelector.py           # Abstract interface
│   └── AprilTagExtractor.py         # AprilTag extractor (testing only)
├── trackers/
│   └── KLTTracker.py                # KLT optical flow with re-extraction and auto-unlock
├── controller/
│   └── PointController.py           # Proportional error → velocity command
└── pipeline/
    ├── IBVSContext.py               # Per-frame data container
    └── IBVSPipeline.py              # Wires source, extractor, tracker, controller
```

---

## Detection Modes

The perch/land point can come from either the branch segmentation pipeline
(`DetectionPipelineSource`) or direct ArUco marker detection
(`ArucoSource.py`). Both produce the same `(frame, point)` contract plus
`reference_frame` / `warmup_complete`, so KLT tracking,
feature extraction, and the controller are identical either way — switching
modes is a one-line change of `DETECTION_MODE` in
`UDP_client/pipeline_factory.py` (`"branch"` or `"aruco"`), shared by both
`UDP_client/main.py` and `main_record.py`, not a code change here.

ArUco is purely a detector choice — `ArucoSource` wraps whatever camera
`pipeline_factory.py`'s `SOURCE_TYPE` selects (DSJ / Nicla / Pi camera / MP4). By
default it tries every predefined ArUco dictionary during warmup and locks
onto whichever one actually finds the tag — there's no way to tell which of
the ~17 families a given printed/generated marker uses just by looking at it,
and pinning the wrong one means silent, permanent non-detection. This only
costs anything during warmup; detection stops entirely once locked. Set
`ARUCO_DICTIONARY` to a specific name (e.g. `"DICT_4X4_50"`) once you know
your tag's dictionary to skip the scan. It also doesn't filter by marker ID,
since this is a single-tag perch/land setup, not multi-tag identification.
Needs `cv2.aruco`, which requires **opencv-contrib-python** (plain
`opencv-python` does not include it) — install it if using `"aruco"` mode.

---

## Tracker Details

`KLTTracker` maintains:
- `_locked_pts` — feature positions at lock time (trimmed as features drop)
- `_locked_final_point` — branch point position at lock time
- `_last_centroid` — centroid of currently tracked features

`estimated_point` each frame = `_locked_final_point + mean(current_pts - locked_pts)`

When features drop below `min(min_features, 30% of tracked)`:
1. Re-extract FAST+Harris around `_last_centroid`
2. Shift `_locked_final_point` by centroid delta
3. Reset `_locked_pts` to new features
4. `estimated_point = _locked_final_point` (no displacement available after reextract)

After 5 consecutive frames with no `estimated_point`, the tracker unlocks and detection takes over to find a new `final_point`.

---

## IBVSContext Fields

| Field | Type | Description |
|-------|------|-------------|
| `frame` | `np.ndarray` | BGR image |
| `point` | `np.ndarray \| None` | detection anchor point (pre-lock only) |
| `extracted_features` | `np.ndarray [N,2]` | currently tracked feature positions |
| `estimated_point` | `np.ndarray \| None` | KLT-estimated branch position |
| `reference_frame` | `np.ndarray \| None` | frame when final_point was locked |
| `warmup_complete` | `bool` | whether detection pipeline warmup is done |
| `debug` | `dict` | controller output, velocity command, control error |

---

## Visualization

| Element | Meaning |
|---------|---------|
| Green dots | KLT-tracked FAST+Harris features |
| Blue circle | Detection anchor point |
| Red circle | KLT-estimated branch position |
| Gray crosshair | Frame center |
| Orange arrow | Velocity command direction and magnitude |

---

## Installation

```bash
pip install opencv-python numpy pyyaml
# detection_pipeline deps also required when using DetectionPipelineSource
```

## Running

```bash
python main.py          # full stack via DetectionPipelineSource
python test_record.py   # record IBVS output to video
```

## Configuration

`config/default_config.yaml` — key parameters:

| Key | Default | Description |
|-----|---------|-------------|
| `feature_extraction.max_features` | `80` | Max FAST+Harris features per frame |
| `feature_extraction.point_focus_radius` | `80` | Pixel radius for feature filtering post-warmup |
| `controller.main_gain` | `0.5` | Proportional gain (pixels → velocity) |
| `controller.min_features` | `8` | Min tracked features before re-extraction |
