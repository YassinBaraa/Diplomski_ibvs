import cv2
import numpy as np

_LK_PARAMS = dict(
    winSize=(21, 21), #search window size
    maxLevel=3, # pyramid levels of zoom
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01), # stop search condition
)

MIN_SURVIVAL_RATIO = 0.3


class KLTTracker:
    def __init__(self, feature_extractor, min_features=8):
        self.feature_extractor = feature_extractor
        self.min_features = int(min_features)

        self._locked = False
        self._prev_gray = None    # grayscale of previous frame
        self._tracked_pts = None  # [N,1,2] float32 — current tracked positions

    def lock(self, reference_frame, features):
        # Initialise tracker from the frame where final_point was locked.
        # reference_frame ensures KLT starts from the same moment the features were extracted.
        self._prev_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        self._tracked_pts = features.reshape(-1, 1, 2).astype(np.float32)
        self._locked = True

    def update(self, frame, anchor_point):
        # Track features from prev frame to current frame using KLT.
        # anchor_point is the last known branch position, used for re-extraction if needed.
        # Returns tracked points [M,2] and the updated grayscale frame.

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not self._locked or self._tracked_pts is None:
            self._prev_gray = curr_gray
            return np.empty((0, 2), dtype=np.float32)

        # Run KLT: propagate tracked points from prev to curr
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, curr_gray, self._tracked_pts, None, **_LK_PARAMS
        )
        good = next_pts[status.reshape(-1) == 1]  # keep only successful tracks

        # Re-extract if too few features survived
        min_required = max(self.min_features, int(len(self._tracked_pts) * MIN_SURVIVAL_RATIO))
        if len(good) < min_required:
            good = self._reextract(frame, anchor_point)

        self._tracked_pts = good if len(good) > 0 else self._tracked_pts
        self._prev_gray = curr_gray

        return good.reshape(-1, 2) if len(good) > 0 else np.empty((0, 2), dtype=np.float32)

    def _reextract(self, frame, anchor_point):
        # Re-run FAST+Harris near anchor_point when too many features are lost.
        # Reuses the FASTHarrisExtractor with a mock ctx carrying the anchor as ctx.point.

        from pipeline.IBVSContext import IBVSContext
        ctx = IBVSContext(frame=frame)
        ctx.point = np.asarray(anchor_point, dtype=np.float32)
        ctx.warmup_complete = True  # ensure radius filter is active

        new_pts = self.feature_extractor.extract(ctx)
        return new_pts.reshape(-1, 1, 2).astype(np.float32) if len(new_pts) > 0 else np.empty((0, 1, 2), dtype=np.float32)
