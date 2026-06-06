import cv2
import numpy as np
from feature_extraction.FeatureSelector import FeatureSelector
from pipeline.IBVSContext import IBVSContext

class FASTHarrisExtractor(FeatureSelector):
    def __init__(
        self,
        max_features: int = 80,
        fast_threshold: int = 20,
        harris_block_size: int = 2,
        harris_ksize: int = 3,
        harris_k: float = 0.04,
        point_focus_radius: float | None = None,
    ):
        self.max_features = max_features
        self.fast_detector = cv2.FastFeatureDetector_create(
            threshold=fast_threshold, nonmaxSuppression=True
        )
        self.harris_block_size = harris_block_size
        self.harris_ksize = harris_ksize
        self.harris_k = harris_k
        self.point_focus_radius = point_focus_radius

    def extract(self, ctx: IBVSContext) -> np.ndarray:
        if ctx.frame is None:
            return np.empty((0, 2), dtype=np.float32)

        gray = cv2.cvtColor(ctx.frame, cv2.COLOR_BGR2GRAY)
        keypoints = self.fast_detector.detect(gray, None)

        if not keypoints:
            return np.empty((0, 2), dtype=np.float32)

        gray_f32 = np.float32(gray)
        harris_response = cv2.cornerHarris(
            gray_f32,
            self.harris_block_size,
            self.harris_ksize,
            self.harris_k,
        )

        candidates: list[tuple[float, float, float]] = []
        target_xy = None
        if ctx.point is not None and len(ctx.point) >= 2:
            target_xy = np.asarray(ctx.point[:2], dtype=np.float32)

        for kp in keypoints:
            x, y = kp.pt
            xi, yi = int(round(x)), int(round(y))
            if yi < 0 or yi >= harris_response.shape[0] or xi < 0 or xi >= harris_response.shape[1]:
                continue

            if ctx.warmup_complete and target_xy is not None and self.point_focus_radius is not None:
                if np.linalg.norm(np.asarray([x, y], dtype=np.float32) - target_xy) > self.point_focus_radius:
                    continue

            score = float(harris_response[yi, xi])
            candidates.append((x, y, score))

        if not candidates:
            return np.empty((0, 2), dtype=np.float32)

        candidates.sort(key=lambda v: v[2], reverse=True)
        top = candidates[: self.max_features]
        points = np.asarray([[x, y] for x, y, _ in top], dtype=np.float32)
        return points
