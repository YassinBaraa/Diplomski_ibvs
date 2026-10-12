"""The chosen perch point, remembered so it is always found again at the same spot on the branch.

In the 2026-10-12 flight the target point jumped 12 times by 80-400 px: every re-acquisition
(new lock after a drop, re-bind to a detection, ORB recovery) took whatever the detector or the
descriptor vote offered at that moment. Here the point is stored once, at the first lock, as ORB
keypoints + descriptors on and along the branch around it, and found again only through them:
a similarity (shift, rotation, scale) fitted to the matched keypoints carries the stored point
into the current frame. The detector never picks a new point; the operator does, with 'r'.

The branch grows ~30x during an approach (6 -> 165 px), more than ORB matches across, so the
model holds several views at different scales. A new view is added only while the tracked point
and the model agree, with its point taken from the model, so the stored spot does not drift.
"""
import cv2
import numpy as np

MIN_INLIERS = 5        # matched keypoints that must agree on the similarity
RATIO = 0.8            # Lowe ratio test
INLIER_PX = 4.0        # RANSAC reprojection threshold
MAX_VIEWS = 8          # the first view (the anchor) is always kept, then the newest
NEW_VIEW_SCALE = 1.5   # add a view when the target is this much bigger/smaller than the closest one
MAX_MATCH_SCALE = 4.0  # a fit scaling a view by more than this (or less than its inverse) is a false match


def branch_strip(shape, point, direction, width, half_len=None, half_wid=None):
    """uint8 mask (255 inside) of a strip along the branch through `point`."""
    d = np.asarray(direction, dtype=np.float64)
    n = np.array([-d[1], d[0]])
    L = half_len if half_len is not None else max(10.0 * width, 100.0)
    W = half_wid if half_wid is not None else max(2.0 * width, 10.0)
    p = np.asarray(point, dtype=np.float64)
    corners = np.array([p + L * d + W * n, p + L * d - W * n, p - L * d - W * n, p - L * d + W * n])
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(corners).astype(np.int32), 255)
    return mask


class TargetModel:

    def __init__(self):
        self._orb = cv2.ORB_create(nfeatures=1500, fastThreshold=10)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.reset()

    def reset(self):
        """Forget the target (operator reset)."""
        self.views = []   # dicts: xy [N,2], desc [N,32], point [2], direction [2], width

    @property
    def active(self):
        return bool(self.views)

    def add_view(self, frame, point, direction, width, region):
        """Store the ORB keypoints inside `region` (uint8 mask) with the point. False if too few."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kps, desc = self._orb.detectAndCompute(gray, region)
        if desc is None or len(kps) < 2 * MIN_INLIERS:
            return False
        self.views.append(dict(xy=np.float32([k.pt for k in kps]), desc=desc,
                               point=np.asarray(point, dtype=np.float32),
                               direction=np.asarray(direction, dtype=np.float64), width=float(width)))
        if len(self.views) > MAX_VIEWS:
            del self.views[1]
        return True

    def locate(self, frame, search=None):
        """Find the stored point in `frame` (search: optional uint8 mask). Returns a dict with
        point, direction, width (expected now), scale (vs the matching view), inliers, view
        index -- or None."""
        if not self.views:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kps, desc = self._orb.detectAndCompute(gray, search)
        if desc is None or len(kps) < MIN_INLIERS:
            return None
        xy = np.float32([k.pt for k in kps])
        best = None
        for i, v in enumerate(self.views):
            good = [p[0] for p in self._matcher.knnMatch(v["desc"], desc, k=2)
                    if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
            if len(good) < MIN_INLIERS:
                continue
            src = v["xy"][[m.queryIdx for m in good]]
            dst = xy[[m.trainIdx for m in good]]
            M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=INLIER_PX)
            if M is None or inl is None or int(inl.sum()) < MIN_INLIERS:
                continue
            n_inl = int(inl.sum())
            if best is not None and n_inl <= best["inliers"]:
                continue
            A = M[:, :2]
            scale = float(np.sqrt(abs(np.linalg.det(A))))
            if not 1.0 / MAX_MATCH_SCALE <= scale <= MAX_MATCH_SCALE:
                continue
            d = A @ v["direction"]
            best = dict(point=(A @ v["point"] + M[:, 2]).astype(np.float32), direction=d / np.linalg.norm(d),
                        width=v["width"] * scale, scale=scale, inliers=n_inl, view=i)
        return best

    def needs_view(self, width):
        """True if no stored view is within NEW_VIEW_SCALE of this branch width."""
        if not width or width <= 0:
            return False
        return all(max(width / v["width"], v["width"] / width) >= NEW_VIEW_SCALE for v in self.views)
