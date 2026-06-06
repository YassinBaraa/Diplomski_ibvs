# IBVS Pipeline

Image-Based Visual Servoing for UAV branch perching. Consumes the detection pipeline output and computes frame-center error to drive the UAV toward the target point.

## What It Does

During the detection pipeline's 20s warmup, uses `best_candidate` as a soft visual target. After warmup, switches to model-free KLT optical flow tracking of FAST+Harris features near the locked `final_point` — no segmentation model needed. The feature centroid is used as the control target. Pixel error relative to frame center is computed each frame and yielded for the UDP client.

`main()` is a generator — it runs the full pipeline, shows visualization, and yields a dict each frame.

---

## Pipeline Flow

```
DetectionPipelineSource.read()
  → frame, point (final_point or best_candidate xy)

FASTHarrisExtractor.extract(ctx)
  → FAST+Harris features near ctx.point (radius-filtered post-warmup)

KLTTracker
  .lock(reference_frame, features)   — first post-warmup frame
  .update(frame, anchor_point)       — every subsequent frame
  → tracked feature positions [N, 2]

PointController.update_ctx(ctx)
  → error  = feature_centroid - frame_center
  → velocity = -gain * error

yield {"error_x", "error_y", "distance_mm"}
```

---

## Directory Structure

```
ibvs/
├── main.py                              # Generator entry point
├── config/
│   ├── __init__.py                      # Config loader
│   └── default_config.yaml
├── sources/
│   ├── FrameSource.py                   # Abstract interface
│   ├── MP4Source.py                     # Video file source
│   └── DetectionPipelineSource.py       # Wraps detection_pipeline yield
├── feature_extraction/
│   ├── FASTHarrisExtractor.py           # FAST detection + Harris filtering
│   ├── AprilTagExtractor.py             # AprilTag (testing only)
│   └── FeatureSelector.py
├── trackers/
│   └── KLTTracker.py                    # KLT optical flow tracker
├── controller/
│   └── PointController.py               # Proportional error → velocity
└── pipeline/
    ├── IBVSContext.py                    # Per-frame data container
    └── IBVSPipeline.py                  # Wires extractor, tracker, controller
```

---

## Yield Output (per frame)

```python
{
    "error_x": float | None,   # pixels, positive = target right of center
    "error_y": float | None,   # pixels, positive = target below center
    "distance_mm": float | None  # ToF reading from detection pipeline
}
```

---

## Visualization

| Element | Meaning |
|---------|---------|
| Green dots | KLT-tracked FAST+Harris features |
| Blue circle | Detection anchor point (final_point) |
| Red circle | Estimated branch position (feature centroid) |
| Gray crosshair | Frame center |
| Orange arrow | Velocity command direction and magnitude |

---

## Usage

```bash
# With detection pipeline (default)
python main.py

# With test video
# set source.type: "mp4" in config/default_config.yaml
python main.py

# Press 'q' to quit
```

## Configuration

`config/default_config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `source.type` | `detection` | `detection` or `mp4` |
| `feature_extraction.max_features` | `80` | Max FAST+Harris features |
| `feature_extraction.point_focus_radius` | `80` | Pixel radius around anchor for feature filtering |
| `controller.main_gain` | `0.5` | Proportional gain |
| `controller.min_features` | `8` | Min tracked features before re-extraction |

## Dependencies

```bash
pip install opencv-python numpy pyyaml
# detection_pipeline deps also needed when source.type = "detection"
```
