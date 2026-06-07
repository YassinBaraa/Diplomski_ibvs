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
        self._prev_gray = None       # grayscale of previous frame
        self._tracked_pts = None     # [N,1,2] float32 — current tracked positions
        self._locked_pts = None      # [N,2] float32 — feature positions at lock time
        self._locked_final_point = None  # final_point pixel coords at lock time
        self._last_centroid = None   # last known KLT centroid, used as re-extraction anchor

    def lock(self, reference_frame, features, final_point):
        # Store locked feature positions and the final_point so we can track
        # where the final_point has moved to each frame via feature displacement.
        self._prev_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        pts = features.reshape(-1, 2).astype(np.float32)
        self._tracked_pts = pts.reshape(-1, 1, 2)
        self._locked_pts = pts.copy()
        self._locked_final_point = np.asarray(final_point, dtype=np.float32)
        self._last_centroid = pts.mean(axis=0).astype(np.float32)
        self._locked = True

    def update(self, frame):
        # Track features and estimate where final_point has moved using mean feature displacement.
        # Returns (tracked_points [M,2], estimated_final_point [2]).

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not self._locked or self._tracked_pts is None:
            self._prev_gray = curr_gray
            return np.empty((0, 2), dtype=np.float32), None

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, curr_gray, self._tracked_pts, None, **_LK_PARAMS
        )
        mask = status.reshape(-1) == 1
        good = next_pts[mask]

        min_required = max(self.min_features, int(len(self._tracked_pts) * MIN_SURVIVAL_RATIO))
        reextracted = False
        if len(good) < min_required:
            good = self._reextract(frame, self._last_centroid)
            mask = None  # no per-point correspondence after re-extraction
            reextracted = True

        if len(good) > 0:
            self._tracked_pts = good
            new_centroid = good.reshape(-1, 2).mean(axis=0).astype(np.float32)
            if reextracted:
                # Anchor resets: treat current centroid as the new origin for displacement.
                # Carry forward the last estimated final_point as the new locked reference.
                if self._locked_final_point is not None:
                    old_centroid = self._last_centroid if self._last_centroid is not None else new_centroid
                    self._locked_final_point = self._locked_final_point + (new_centroid - old_centroid)
                self._locked_pts = good.reshape(-1, 2).copy()
            self._last_centroid = new_centroid
        self._prev_gray = curr_gray

        good_2d = good.reshape(-1, 2) if len(good) > 0 else np.empty((0, 2), dtype=np.float32)

        # Compute estimated final_point as locked position + mean displacement of survived features.
        estimated = None
        if mask is not None and self._locked_pts is not None and len(good_2d) > 0:
            locked_survived = self._locked_pts[mask]
            if len(locked_survived) > 0:
                mean_disp = (good_2d - locked_survived).mean(axis=0)
                estimated = self._locked_final_point + mean_disp

        return good_2d, estimated

    def _reextract(self, frame, anchor_point):
        # Re-run FAST+Harris near anchor_point when too many features are lost.
        # anchor_point is always the last KLT centroid, not the model output.

        from pipeline.IBVSContext import IBVSContext
        ctx = IBVSContext(frame=frame)
        ctx.point = np.asarray(anchor_point, dtype=np.float32) if anchor_point is not None else None
        ctx.warmup_complete = anchor_point is not None  # radius filter only if anchor is known

        new_pts = self.feature_extractor.extract(ctx)
        return new_pts.reshape(-1, 1, 2).astype(np.float32) if len(new_pts) > 0 else np.empty((0, 1, 2), dtype=np.float32)
