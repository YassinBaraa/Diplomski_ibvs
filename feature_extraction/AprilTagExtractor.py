import cv2
import numpy as np
from feature_extraction.FeatureSelector import FeatureSelector
from pipeline.IBVSContext import IBVSContext

try:
    import apriltag
except ImportError:  # pragma: no cover - depends on optional system package
    apriltag = None


class AprilTagExtractor(FeatureSelector):
    def __init__(self):
        if apriltag is None:
            raise ImportError(
                "apriltag is not installed. Install it before using AprilTagExtractor."
            )
        self.detector = apriltag.Detector()

    def extract(self, ctx: IBVSContext) -> np.ndarray:
        gray = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)

        if len(detections) == 0:
            return np.empty((0, 2), dtype=np.float32)

        centers = [tag.center for tag in detections]
        return np.asarray(centers, dtype=np.float32)
