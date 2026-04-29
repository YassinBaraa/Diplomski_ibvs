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
- `config/`: Configuration management
  - `default_config.yaml`: IBVS parameters
- `main.py`: Entry point (matches detection pipeline structure)
- `tests/`: Pre-merge validation tests

## Quick Start

### Run IBVS with MP4 source

```bash
python3 main.py
```

Configure via `config/default_config.yaml` or environment:

```bash
export IBVS_VIDEO_PATH=/path/to/video.mp4
export IBVS_MAX_FEATURES=80
python3 main.py
```

### Tests

Run pre-merge validation:
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
- Source type (mp4 / detection)
- Feature extraction thresholds (FAST, Harris)
- Visualization toggles

## Notes

- Core is framework-agnostic (non-ROS for simplicity and testing)
- ROS adapter layer planned as separate module
- Detection pipeline integration via context adapter (simple, no duplication)
- FAST+Harris is primary feature extraction method

## References

- IBVS theory: https://visp-doc.inria.fr/doxygen/visp-daily/tutorial-ibvs.html
- Camera specs: https://joy-it.net/files/files/Produkte/rb-camera_JT/rb-camera_JT_Datasheet_2021-02-09.pdf
- AprilTag: https://pyimagesearch.com/2020/11/02/apriltag-with-python/
- Harris corners: https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html
