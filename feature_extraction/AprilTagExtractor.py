import cv2
import numpy as np
from feature_extraction.FeatureSelector import FeatureSelector
from pipeline.IBVSContext import IBVSContext
import apriltag


class AprilTagExtractor(FeatureSelector):
    def __init__(self, detector_options: dict | None = None):
        if apriltag is None:
            raise ImportError("apriltag is not installed. Install it before using AprilTagExtractor.")
        if detector_options is None:
            opts = apriltag.DetectorOptions()
        else:
            opts = apriltag.DetectorOptions(**detector_options)
        self.detector = apriltag.Detector(opts)

    def extract(self, ctx: IBVSContext) -> np.ndarray:
        gray = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        detections = self.detector.detect(gray)

        if len(detections) == 0:
            return np.empty((0, 2), dtype=np.float32)

        centers = [tag.center for tag in detections]
        return np.asarray(centers, dtype=np.float32)
