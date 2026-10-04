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
_STALL_MOTION_PX = 1.5     # px/frame; point motion below this counts as "not moving"
_STALL_FRAMES_LIMIT = 15   # consecutive stalled frames (after a re-extraction) before treating the lock as bogus
_MIN_VOTE_FEATURES = 3     # fewer surviving features than this cannot vote for the point -> target lost

# Candidate confirmation (replaces the old 50-frame warmup): follow the candidate's
# features over a few consecutive detections and lock once they survive.
_CONFIRM_FRAMES = 3
_CONFIRM_RADIUS_PX = 40    # detection point must stay this close to the features' own estimate
_CONFIRM_SURVIVAL = 0.6    # share of features that must survive forward-backward tracking

# Recovery by ORB descriptor voting
_REF_RADIUS_PX = 100       # descriptors are stored for keypoints this close to the point
_MATCH_RATIO = 0.8         # Lowe ratio test
_MIN_MATCHES = 6
_VOTE_INLIER_PX = 15       # votes (keypoint - stored offset) within this of the median are inliers
_COAST_DAMPING = 0.7       # per-frame shrink of the coasted velocity


class KLTTracker:
    """KLT tracker whose output is the perch POINT, not the feature centroid.

    Every tracked feature stores its offset to the point, so the point is
    median(feature - offset) over the survivors. Losing some features therefore
    does not move the point, and re-extracted features get fresh offsets.

    States: "idle" (nothing tracked), "tracking", "lost" (the target vanished, the
    stored descriptors are kept so it can be found again).
    """

    def __init__(self, feature_extractor, min_features=8, recovery=True):
        self.feature_extractor = feature_extractor
        self.min_features = int(min_features)
        self.recovery = bool(recovery)

        self._orb = cv2.ORB_create(nfeatures=1000) if self.recovery else None
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING) if self.recovery else None
        self.unlock()

    # ---------------------------------------------------------------- state

    @property
    def state(self):
        if self._locked:
            return "tracking"
        return "lost" if self._point is not None else "idle"

    @property
    def point(self):
        return None if self._point is None else self._point.copy()

    @property
    def points(self):
        """Features currently shown/tracked (candidate features while idle)."""
        if self._tracked_pts is not None:
            return self._tracked_pts.reshape(-1, 2)
        if self._cand is not None:
            return self._cand["pts"].reshape(-1, 2)
        return np.empty((0, 2), dtype=np.float32)

    def unlock(self):
        """Forget everything (back to idle)."""
        self._locked = False
        self._prev_gray = None
        self._tracked_pts = None
        self._offsets = None
        self._point = None
        self._velocity = np.zeros(2, dtype=np.float32)
        self._stall_streak = 0
        self._reextracted = False
        self._ref_kp_offsets = None
        self._ref_desc = None
        self._cand = None
        self._lost_point = None

    def _start_tracking(self, gray, pts, offsets, point):
        self._prev_gray = gray
        self._tracked_pts = pts.reshape(-1, 1, 2).astype(np.float32)
        self._offsets = offsets.reshape(-1, 2).astype(np.float32)
        self._point = np.asarray(point, dtype=np.float32)
        self._velocity = np.zeros(2, dtype=np.float32)
        self._stall_streak = 0
        self._reextracted = False
        self._cand = None
        self._lost_point = None
        self._locked = True
        self._make_reference(gray, self._point)

    # ------------------------------------------------------------- features

    def _extract_around(self, frame, point):
        ctx = types.SimpleNamespace(frame=frame, point=np.asarray(point, dtype=np.float32))
        new_pts = self.feature_extractor.extract(ctx)
        return new_pts.reshape(-1, 2).astype(np.float32)

    def _make_reference(self, gray, point):
        """Store ORB descriptors around the point, with each keypoint's offset to it."""
        self._ref_desc = None
        self._ref_kp_offsets = None
        if not self.recovery:
            return
        mask = np.zeros_like(gray)
        cv2.circle(mask, (int(round(point[0])), int(round(point[1]))), _REF_RADIUS_PX, 255, -1)
        kps, desc = self._orb.detectAndCompute(gray, mask)
        if desc is None or len(kps) < _MIN_MATCHES:
            return
        self._ref_desc = desc
        self._ref_kp_offsets = np.array([kp.pt for kp in kps], dtype=np.float32) - point

    def _klt(self, prev_gray, gray, pts):
        """Forward-backward checked LK. Returns (next_pts[N,1,2], ok_mask[N])."""
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **_LK_PARAMS)
        ok = status.reshape(-1) == 1
        if np.any(ok):
            back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, next_pts, None, **_LK_PARAMS)
            fb_error = np.linalg.norm(pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1)
            ok = ok & (back_status.reshape(-1) == 1) & (fb_error < _FB_ERROR_THRESHOLD)
        return next_pts, ok

    # ------------------------------------------------------------ detection

    def observe(self, frame, point):
        """Idle mode: follow one detection candidate over a few frames and lock when
        its features hold up. `point=None` (no candidate this frame) clears it.
        Returns True on the frame the lock happens. Never starts a second lock while one is active."""
        if self._locked:
            return False
        if point is None:
            self._cand = None
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        point = np.asarray(point, dtype=np.float32)
        c = self._cand
        if c is not None:
            next_pts, ok = self._klt(c["gray"], gray, c["pts"])
            n_ok = int(ok.sum())
            if n_ok >= max(self.min_features, int(len(c["pts"]) * _CONFIRM_SURVIVAL)):
                pts = next_pts[ok]
                offsets = c["offsets"][ok]
                est = np.median(pts.reshape(-1, 2) - offsets, axis=0)
                if np.linalg.norm(est - point) <= _CONFIRM_RADIUS_PX:
                    c.update(pts=pts, offsets=offsets, gray=gray, hits=c["hits"] + 1)
                    if c["hits"] >= _CONFIRM_FRAMES:
                        self._start_tracking(gray, pts, offsets, est)
                        print(f"[KLTTracker] Locked on {len(pts)} features, point={est}")
                        return True
                    return False
            c = None

        pts = self._extract_around(frame, point)
        if len(pts) >= self.min_features:
            self._cand = dict(pts=pts.reshape(-1, 1, 2), offsets=pts - point, gray=gray, hits=1)
        else:
            self._cand = None
        return False

    def lock_at(self, frame, point):
        """Lock immediately on a trusted detection point (used to re-bind after a loss)."""
        if self._locked:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pts = self._extract_around(frame, point)
        if len(pts) < self.min_features:
            return False
        self._start_tracking(gray, pts, pts - np.asarray(point, dtype=np.float32), point)
        print(f"[KLTTracker] Re-bound to detection at {point} with {len(pts)} features")
        return True

    # ------------------------------------------------------------- tracking

    def update(self, frame):
        """Returns (tracked_features[N,2], point_or_None). None means the target was lost."""
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not self._locked or self._tracked_pts is None:
            return np.empty((0, 2), dtype=np.float32), None

        next_pts, ok = self._klt(self._prev_gray, curr_gray, self._tracked_pts)
        n_prev = len(self._tracked_pts)
        pts = next_pts[ok]
        offsets = self._offsets[ok]

        if len(pts) < _MIN_VOTE_FEATURES:
            print(f"[KLTTracker] Only {len(pts)}/{n_prev} features survived — target lost")
            self._lose()
            return np.empty((0, 2), dtype=np.float32), None

        # Each survivor votes for the point through its stored offset
        est = np.median(pts.reshape(-1, 2) - offsets, axis=0).astype(np.float32)

        min_required = max(self.min_features, int(n_prev * MIN_SURVIVAL_RATIO))
        if len(pts) < min_required:
            new_pts = self._extract_around(frame, est)
            if len(new_pts) > 0:
                print(f"[KLTTracker] Only {len(pts)}/{n_prev} features survived (need {min_required}) — "
                      f"re-extracted {len(new_pts)} around {est}")
                pts, offsets = new_pts.reshape(-1, 1, 2), new_pts - est
                self._reextracted = True
                self._make_reference(curr_gray, est)

        motion = float(np.linalg.norm(est - self._point))
        self._velocity = est - self._point
        self._point = est
        self._tracked_pts = pts.reshape(-1, 1, 2).astype(np.float32)
        self._offsets = offsets.reshape(-1, 2).astype(np.float32)
        self._prev_gray = curr_gray

        # Stale-lock guard: after a re-extraction the features may have landed on static
        # background clutter, which would then report a frozen, wrong point forever.
        # Only counted after a re-extraction, so a UAV that is genuinely holding still
        # on a healthy lock is not declared lost.
        if self._reextracted and motion < _STALL_MOTION_PX:
            self._stall_streak += 1
        else:
            self._stall_streak = 0
        if self._stall_streak >= _STALL_FRAMES_LIMIT:
            print(f"[KLTTracker] Point stalled for {self._stall_streak} frames after re-extraction — "
                  f"likely locked onto static background. Dropping the target.")
            self.unlock()
            return np.empty((0, 2), dtype=np.float32), None

        return pts.reshape(-1, 2), np.round(est).astype(int)

    # ------------------------------------------------------------- loss

    def _lose(self):
        """Target vanished. With recovery on, keep the descriptors and last point so it can be found again."""
        if not self.recovery or self._ref_desc is None:
            self.unlock()
            return
        self._locked = False
        self._tracked_pts = None
        self._prev_gray = None
        self._lost_point = self._point.copy()  # coasting position, starts at the last good point

    @property
    def last_point(self):
        """Last good point of a lost target (for gating re-binding)."""
        return None if self.state != "lost" else self._point.copy()

    def coast(self):
        """Next coasted position of a lost target: last velocity, damped each call."""
        self._lost_point = self._lost_point + self._velocity
        self._velocity = self._velocity * _COAST_DAMPING
        return np.round(self._lost_point).astype(int)

    def recover(self, frame):
        """Find a lost target again by matching the stored ORB descriptors.
        Each match votes for the point with (keypoint - stored offset). Returns True if re-locked."""
        if self.state != "lost" or self._ref_desc is None:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kps, desc = self._orb.detectAndCompute(gray, None)
        if desc is None or len(kps) < 2:
            return False

        pairs = self._matcher.knnMatch(self._ref_desc, desc, k=2)
        votes = []
        for p in pairs:
            if len(p) == 2 and p[0].distance < _MATCH_RATIO * p[1].distance:
                m = p[0]
                votes.append(np.array(kps[m.trainIdx].pt, dtype=np.float32) - self._ref_kp_offsets[m.queryIdx])
        if len(votes) < _MIN_MATCHES:
            return False
        votes = np.array(votes)
        # Mode of the votes (the vote with most neighbours), not their median: a few outlier
        # matches drag a median far off, while a real re-sighting piles votes at one spot.
        neighbours = (np.linalg.norm(votes[:, None] - votes[None], axis=2) <= _VOTE_INLIER_PX).sum(axis=1)
        inliers = np.linalg.norm(votes - votes[int(np.argmax(neighbours))], axis=1) <= _VOTE_INLIER_PX
        if inliers.sum() < _MIN_MATCHES:
            return False

        point = np.median(votes[inliers], axis=0)
        pts = self._extract_around(frame, point)
        if len(pts) < self.min_features:
            return False
        self._start_tracking(gray, pts, pts - point, point)
        print(f"[KLTTracker] Recovered by descriptor voting ({int(inliers.sum())} matches) at {point}")
        return True
