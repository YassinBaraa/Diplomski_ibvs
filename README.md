# Diplomski IBVS

IBVS (Image-Based Visual Servoing) pipeline for UAV control. Core non-ROS implementation.

## Architecture

- `pipeline/`: Core orchestration
  - `IBVSPipeline.py`: frame loop + feature extraction
  - `IBVSContext.py`: frame/point/feature context container
- `sources/`: Input adapters (abstract `FrameSource`)
  - `MP4Source.py`: video file input
  - `DetectionPipelineSource.py`: detection pipeline context adapter
- `feature_extraction/`: Visual feature detection
  - `FASTHarrisExtractor.py`: FAST + Harris corner ranking
  - `AprilTagExtractor.py`: optional AprilTag marker centers
  - `FeatureSelector.py`: extraction interface
- `controller/`: UAV velocity computation (Phase 2)
  - `SimplePointController.py`: Proportional 2D centering control law
  - `FeatureValidator.py`: Feature consistency robustness checks
- `output/`: Command publishing backends
  - `PrintPublisher.py`: Console output for debugging
  - `FilePublisher.py`: CSV logging for post-analysis
- `config/`: Configuration management
  - `default_config.yaml`: IBVS parameters + controller gains
- `main.py`: Entry point (matches detection pipeline structure)
- `tests/`: Pre-merge validation tests

## Control Law (Phase 2)

**Proportional 2D Centering:**
```
Error: e_n = [u_n - w/2, v_n - h/2]  (pixels from image center)
Command: [v_x, v_y] = -k * e_n  (body-frame velocity, k ≈ 0.5-1.0)
Effect: Negative feedback drives detected point toward image center
```

**Feature Validation:**
- Tracks feature motion across frames
- Flags invalid frames if features jump (> threshold) or disappear (> 50%)
- Can modulate gain or freeze control on unreliable frames

## Quick Start

### Run IBVS with MP4 source

```bash
python3 main.py
```

Configure via `config/default_config.yaml` or environment:

```bash
export IBVS_VIDEO_PATH=/path/to/video.mp4
export IBVS_MAX_FEATURES=80
export IBVS_CONTROLLER_MAIN_GAIN=0.5
export IBVS_OUTPUT_PUBLISHER_TYPE=print
python3 main.py
```

### Tests

Run pre-merge validation (controller + feature extraction tests):
```bash
python3 -m pytest tests/ -v
```

Or classic unittest:
```bash
python3 -m unittest discover
```

Optional: Full detection+IBVS integration test:
```bash
export IBVS_RUN_DETECTION_INTEGRATION=1
export IBVS_DETECTION_REPO=../detection_pipeline
export IBVS_DETECTION_VIDEO=/path/to/video.mp4
export IBVS_DETECTION_MODEL=/path/to/model.pt
python3 -m unittest discover
```

## Configuration

See `config/default_config.yaml` for all tunable parameters:

**Source:**
- `source.type`: "mp4" or "detection"
- `source.video_path`: path to video file

**Feature Extraction:**
- `feature_extraction.max_features`: number of features to extract (default: 80)
- `feature_extraction.fast_threshold`: FAST keypoint threshold (default: 20)
- `feature_extraction.point_focus_radius`: optional ROI radius around target (default: null)

**Controller (Phase 2):**
- `controller.main_gain`: proportional gain k (default: 0.5, range: 0.1-2.0)
- `controller.max_feature_motion_px`: max feature motion per frame for consistency (default: 10 px)
- `controller.validation_enabled`: enable robustness checks (default: true)

**Output (Phase 2):**
- `output.publisher_type`: "print", "file", or "ros" (Phase 3+) (default: "print")
- `output.file_path`: CSV output path if publisher_type="file"
- `output.verbose`: verbose console output (default: true)

**Visualization:**
- `visualization.enabled`: show live frame with point + features
- `visualization.display_window_name`: window title

## Notes

- Core is framework-agnostic (non-ROS for simplicity and testing)
- ROS adapter layer planned as separate module (Phase 3)
- Detection pipeline integration via context adapter (simple, no duplication)
- FAST+Harris is primary feature extraction method
- Proportional control law (Phase 2); depth control deferred to Phase 3+

## References

- IBVS theory: https://visp-doc.inria.fr/doxygen/visp-daily/tutorial-ibvs.html
- Camera specs: https://joy-it.net/files/files/Produkte/rb-camera_JT/rb-camera_JT_Datasheet_2021-02-09.pdf
- AprilTag: https://pyimagesearch.com/2020/11/02/apriltag-with-python/
- Harris corners: https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html
