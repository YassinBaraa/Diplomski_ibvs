# Diplomski IBVS

Initial IBVS workspace for UAV control development.

## Current implementation
- `pipeline/IBVSContext.py`: frame/point/feature context container
- `pipeline/IBVSPipeline.py`: frame loop + feature extraction orchestration
- `feature_extraction/FASTHarrisExtractor.py`: FAST keypoints ranked by Harris response
- `feature_extraction/AprilTagExtractor.py`: optional AprilTag center extraction
- `feature_extraction/FeatureSelector.py`: extraction interface

## Notes
- FAST+Harris is the default feature extraction direction for IBVS control.
- AprilTag extraction is optional and requires the `apriltag` Python package.

## References
- https://visp-doc.inria.fr/doxygen/visp-daily/tutorial-ibvs.html
- https://joy-it.net/files/files/Produkte/rb-camera_JT/rb-camera_JT_Datasheet_2021-02-09.pdf
- https://pyimagesearch.com/2020/11/02/apriltag-with-python/
- https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html
