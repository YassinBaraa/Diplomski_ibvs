import cv2
import numpy as np
import types

_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

MIN_SURVIVAL_RATIO = 0.3


class KLTTracker:
    def __init__(self, feature_extractor, min_features=8):
        self.feature_extractor = feature_extractor
        self.min_features = int(min_features)

        self._locked = False
        self._prev_gray = None
        self._tracked_pts = None
        self._locked_pts = None
        self._locked_final_point = None
        self._last_centroid = None

    def lock(self, reference_frame, features, final_point):
        self._prev_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        pts = features.reshape(-1, 2).astype(np.float32)
        self._tracked_pts = pts.reshape(-1, 1, 2)
        self._locked_pts = pts.copy()
        self._locked_final_point = np.asarray(final_point, dtype=np.float32)
        self._last_centroid = pts.mean(axis=0).astype(np.float32)
        self._locked = True
        print(f"[KLTTracker] Locked on {len(pts)} features, final_point={final_point}")

    def unlock(self):
        """Drop all tracking state so IBVSPipeline reverts to detection mode."""
        self._locked = False
        self._tracked_pts = None
        self._locked_pts = None
        self._locked_final_point = None
        self._last_centroid = None
        self._prev_gray = None
        print("[KLTTracker] Unlocked — reverting to detection mode")

    def update(self, frame):
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
            print(f"[KLTTracker] Only {len(good)}/{len(self._tracked_pts)} features survived "
                  f"(need {min_required}) — re-extracting around last centroid")
            good = self._reextract(frame, self._last_centroid)
            mask = None
            reextracted = True

        if len(good) == 0:
            print("[KLTTracker] Re-extraction found no features — signalling lost")
            self._prev_gray = curr_gray
            return np.empty((0, 2), dtype=np.float32), None

        self._tracked_pts = good
        new_centroid = good.reshape(-1, 2).mean(axis=0).astype(np.float32)
        if reextracted:
            self._locked_pts = good.reshape(-1, 2).copy()
            print(f"[KLTTracker] Re-extracted {len(good)} features, centroid={new_centroid}")
        else:
            self._locked_pts = self._locked_pts[mask]
        self._last_centroid = new_centroid
        self._prev_gray = curr_gray

        good_2d = good.reshape(-1, 2)

        # Use the current centroid of tracked features directly as the branch position.
        # This is a direct measurement and doesn't accumulate drift over time.
        estimated = new_centroid.copy() if len(good_2d) > 0 else None

        return good_2d, estimated

    def _reextract(self, frame, anchor_point):
        ctx = types.SimpleNamespace(
            frame=frame,
            point=np.asarray(anchor_point, dtype=np.float32) if anchor_point is not None else None,
            warmup_complete=anchor_point is not None,
        )
        new_pts = self.feature_extractor.extract(ctx)
        return new_pts.reshape(-1, 1, 2).astype(np.float32) if len(new_pts) > 0 else np.empty((0, 1, 2), dtype=np.float32)