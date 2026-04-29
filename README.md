# Diplomski IBVS

Initial IBVS workspace for UAV control development.

## Current implementation
- `pipeline/IBVSContext.py`: frame/point/feature context container
- `pipeline/IBVSPipeline.py`: frame loop + feature extraction orchestration
- `sources/FrameSource.py`: source interface
- `sources/MP4Source.py`: MP4-only IBVS input source
- `sources/DetectionPipelineSource.py`: adapter from detection pipeline context to IBVS source contract
- `feature_extraction/FASTHarrisExtractor.py`: FAST keypoints ranked by Harris response
- `feature_extraction/AprilTagExtractor.py`: optional AprilTag center extraction
- `feature_extraction/FeatureSelector.py`: extraction interface
- `main.py`: simple local entry with IBVS helper + imported detection-pipeline main hook

## Run

### IBVS-only with MP4 source (simple)
```bash
python3 -c "from main import run_ibvs_with_mp4; run_ibvs_with_mp4('/absolute/path/to/video.mp4', max_frames=100)"
```

### Detection pipeline main (imported, not duplicated)
```bash
python3 -c "from main import run_detection_pipeline_main; run_detection_pipeline_main()"
```

## Tests

### 1) IBVS-only MP4 test (pre-merge path)
```bash
python3 -m unittest tests.test_ibvs_mp4_source -v
```

### 2) Full detection+IBVS integration test
```bash
export IBVS_RUN_DETECTION_INTEGRATION=1
export IBVS_DETECTION_REPO=../detection_pipeline
export IBVS_DETECTION_VIDEO=/absolute/path/to/video.mp4
export IBVS_DETECTION_MODEL=/absolute/path/to/best_small.pt
python3 -m unittest tests.test_integration_detection_pipeline -v
```

This integration test is intentionally opt-in because it depends on detection assets and heavier dependencies, and it feeds detection context into IBVS via `DetectionPipelineSource`.

## Notes
- FAST+Harris is the default feature extraction direction for IBVS control.
- AprilTag extraction is optional and requires the `apriltag` Python package.

## References
- https://visp-doc.inria.fr/doxygen/visp-daily/tutorial-ibvs.html
- https://joy-it.net/files/files/Produkte/rb-camera_JT/rb-camera_JT_Datasheet_2021-02-09.pdf
- https://pyimagesearch.com/2020/11/02/apriltag-with-python/
- https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html
