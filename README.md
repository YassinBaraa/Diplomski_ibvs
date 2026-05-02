# IBVS Pipeline

Image-Based Visual Servoing pipeline for UAV control.

## Quick Start

**Run with Detection Pipeline:**
```bash
python main.py --source=detection
```

**Run with Test Video:**
```bash
python main.py --source=video --video-path=/path/to/video.mp4
```

## Optional Arguments

- `--max-frames=N` - Process only N frames (default: all)
- `--print-every=N` - Print info every N frames (default: 1)
- `--no-display` - Hide OpenCV window


## Project Structure

- `pipeline/` - Core components (IBVSContext, IBVSPipeline)
- `feature_extraction/` - Feature extractors (AprilTag, FAST+Harris)
- `sources/` - Data sources (MP4Source, DetectionPipelineSource)
- `controller/` - Control law (PointController)
- `main.py` - Main entry point
- `tests/` - Test scripts

## Architecture

- **Primary Source**: Detection pipeline point
- **Validation**: Feature extraction points check
- **Fallback**: Use feature centroid if detection point is None
- **Control**: Error-based proportional velocity scaling
- **Output**: All results stored in `ctx.debug`

## Output Example

```
--- Frame 0 ---
Point Source: detection
Status: ok
Control Error (px): [10.50, -2.30]
Velocity Command: [5.25, -1.15]

--- Frame 5 ---
Point Source: fallback
Status: ok
Control Error (px): [2.10, 0.80]
Velocity Command: [1.05, 0.40]
```

## References

- IBVS theory: https://visp-doc.inria.fr/doxygen/visp-daily/tutorial-ibvs.html
- Camera specs: https://joy-it.net/files/files/Produkte/rb-camera_JT/rb-camera_JT_Datasheet_2021-02-09.pdf
- AprilTag: https://pyimagesearch.com/2020/11/02/apriltag-with-python/
- Harris corners: https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html
