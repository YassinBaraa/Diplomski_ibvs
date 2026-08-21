import cv2
import numpy as np
import types

_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

MIN_SURVIVAL_RATIO = 0.3
_FB_ERROR_THRESHOLD = 2.0  # px; forward-backward round-trip error above this is a bad track
_STALL_MOTION_PX = 1.5     # px/frame; centroid motion below this counts as "not moving"
_STALL_FRAMES_LIMIT = 15   # consecutive stalled frames before treating the lock as bogus


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
        self._velocity = np.zeros(2, dtype=np.float32)  # last-seen centroid displacement per frame
        self._stall_streak = 0  # consecutive frames with ~zero centroid motion

    def lock(self, reference_frame, features, final_point):
        self._prev_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        pts = features.reshape(-1, 2).astype(np.float32)
        self._tracked_pts = pts.reshape(-1, 1, 2)
        self._locked_pts = pts.copy()
        self._locked_final_point = np.asarray(final_point, dtype=np.float32)
        self._last_centroid = pts.mean(axis=0).astype(np.float32)
        self._velocity = np.zeros(2, dtype=np.float32)
        self._stall_streak = 0
        self._locked = True
        print(f"[KLTTracker] Locked on {len(pts)} features, final_point={final_point}")

    def unlock(self):
        """Drop all tracking state so IBVSPipeline reverts to detection mode."""
        self._locked = False
        self._tracked_pts = None
        self._locked_pts = None
        self._locked_final_point = None
        self._last_centroid = None
        self._velocity = np.zeros(2, dtype=np.float32)
        self._stall_streak = 0
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
        fwd_mask = status.reshape(-1) == 1

        # Forward-backward consistency check: track next_pts back into prev_gray and
        # discard any point that doesn't round-trip near its origin. This throws out
        # spurious matches (e.g. onto background clutter) that forward-only status==1
        # accepts, so fewer good tracks are lost to noise and re-extraction (which
        # searches near the stale last-known position, not the true new one) triggers
        # less often.
        mask = np.zeros_like(fwd_mask)
        if np.any(fwd_mask):
            back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
                curr_gray, self._prev_gray, next_pts, None, **_LK_PARAMS
            )
            back_ok = back_status.reshape(-1) == 1
            fb_error = np.linalg.norm(
                self._tracked_pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1
            )
            mask = fwd_mask & back_ok & (fb_error < _FB_ERROR_THRESHOLD)

        good = next_pts[mask]

        min_required = max(self.min_features, int(len(self._tracked_pts) * MIN_SURVIVAL_RATIO))
        reextracted = False
        if len(good) < min_required:
            predicted_centroid = self._last_centroid + self._velocity
            print(f"[KLTTracker] Only {len(good)}/{len(self._tracked_pts)} features survived "
                  f"(need {min_required}) — re-extracting around predicted centroid "
                  f"{predicted_centroid} (last={self._last_centroid}, vel={self._velocity})")
            good = self._reextract(frame, predicted_centroid)
            mask = None
            reextracted = True

        if len(good) == 0:
            print("[KLTTracker] Re-extraction found no features — signalling lost")
            self._prev_gray = curr_gray
            return np.empty((0, 2), dtype=np.float32), None

        self._tracked_pts = good
        new_centroid = good.reshape(-1, 2).mean(axis=0).astype(np.float32)
        centroid_motion = float(np.linalg.norm(new_centroid - self._last_centroid))
        if reextracted:
            self._locked_pts = good.reshape(-1, 2).copy()
            print(f"[KLTTracker] Re-extracted {len(good)} features, centroid={new_centroid}")
            # Re-extraction has no optical-flow displacement to measure velocity from --
            # keep the last known velocity rather than zeroing it, so a subsequent
            # re-extraction still predicts forward instead of collapsing to a static anchor.
        else:
            self._locked_pts = self._locked_pts[mask]
            self._velocity = new_centroid - self._last_centroid
        self._last_centroid = new_centroid
        self._prev_gray = curr_gray

        # Stale-lock guard: a real target being approached/panned shows SOME parallax
        # frame to frame. If the centroid sits still for many consecutive frames, the
        # tracked features most likely re-extracted onto static background clutter
        # (e.g. a fold/shadow in a plain backdrop) rather than the actual target --
        # FASTHarrisExtractor has no notion of "target" vs. "background", it just
        # picks the best-scoring corners near the search anchor. Report lost instead
        # of silently repeating a frozen, wrong pixel for seconds at a time (this
        # produced exact, unchanging target_point runs of 8-11s in real flight bags).
        if centroid_motion < _STALL_MOTION_PX:
            self._stall_streak += 1
        else:
            self._stall_streak = 0
        if self._stall_streak >= _STALL_FRAMES_LIMIT:
            print(f"[KLTTracker] Centroid stalled for {self._stall_streak} frames "
                  f"(motion={centroid_motion:.2f}px < {_STALL_MOTION_PX}px) — likely locked "
                  f"onto static background, not the target. Signalling lost.")
            return np.empty((0, 2), dtype=np.float32), None

        good_2d = good.reshape(-1, 2)

        # Use the current centroid of tracked features directly as the branch position.
        # This is a direct measurement and doesn't accumulate drift over time.
        # Internal tracking state (_last_centroid etc.) stays float32 for numerical
        # accuracy; only the point handed to the rest of the pipeline is rounded to int.
        estimated = np.round(new_centroid).astype(int) if len(good_2d) > 0 else None

        return good_2d, estimated

    def _reextract(self, frame, anchor_point):
        ctx = types.SimpleNamespace(
            frame=frame,
            point=np.asarray(anchor_point, dtype=np.float32) if anchor_point is not None else None,
            warmup_complete=anchor_point is not None,
        )
        new_pts = self.feature_extractor.extract(ctx)
        return new_pts.reshape(-1, 1, 2).astype(np.float32) if len(new_pts) > 0 else np.empty((0, 1, 2), dtype=np.float32)