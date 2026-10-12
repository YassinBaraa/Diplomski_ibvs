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
# A stick looks alike along its length: in the 2026-10-12 10:30 flight one view matched 16 and 15
# times at two spots 180 px apart, alternately. A rival placement of the point this far away
# with this share of the best fit's matches makes the result ambiguous.
RIVAL_MIN_SEP = 3.0    # [widths], at least RIVAL_MIN_SEP_PX
RIVAL_MIN_SEP_PX = 20.0
RIVAL_SHARE = 0.6


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


# Where a background line (wire, ceiling beam) crosses behind the branch, the strongest corners of
# the whole strip appear -- and they belong to neither: being far behind, the crossing slides along
# the branch as the UAV moves. In the 2026-10-12 16:11 and 17:13 flights the KLT features and the
# stored views sat on such crossings and the perch point slid 70-200 px along the stick with them.
CROSS_CONTRAST = 0.4   # both sides of the branch darker than the plain background by this share of the branch's contrast
CROSS_MARGIN = 1.5     # [widths], at least CROSS_MARGIN_PX: kept clear on each side of a crossing
CROSS_MARGIN_PX = 8.0


def crossing_spans(gray, point, direction, width, half_len):
    """[(t0, t1)]: stretches along the branch (px from `point`, along `direction`) where the
    background is dark on both sides of it, i.e. something crosses behind it. gray: uint8 image."""
    d = np.asarray(direction, dtype=np.float32)
    d = d / max(float(np.linalg.norm(d)), 1e-6)
    n = np.array([-d[1], d[0]], dtype=np.float32)
    p = np.asarray(point, dtype=np.float32)
    w = float(width)
    t = np.arange(-half_len, half_len + 1.0, 1.0, dtype=np.float32)
    band = np.arange(0.5 * w + 3.0, 0.5 * w + 3.0 + max(4.0, 0.6 * w), 1.0, dtype=np.float32)
    body = np.linspace(-0.25 * w, 0.25 * w, 3, dtype=np.float32)
    h, wd = gray.shape[:2]

    def sample(s):   # [len(t), len(s)], +inf outside the image (never dark)
        X = p[0] + t[:, None] * d[0] + s[None, :] * n[0]
        Y = p[1] + t[:, None] * d[1] + s[None, :] * n[1]
        v = cv2.remap(gray, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        v[(X < 0) | (Y < 0) | (X > wd - 1) | (Y > h - 1)] = np.inf
        return v

    q = (len(band) - 1) // 4   # a thin wire darkens only part of the band: its darker quarter counts
    left = np.partition(sample(-band), q, axis=1)[:, q]
    right = np.partition(sample(band), q, axis=1)[:, q]
    sides = np.concatenate([left, right])
    sides = sides[np.isfinite(sides)]
    body_v = sample(body)
    body_v = body_v[np.isfinite(body_v)]
    if len(sides) < 10 or not len(body_v):
        return []
    bg, branch = float(np.percentile(sides, 75)), float(np.median(body_v))
    if bg - branch < 15.0:
        return []
    thr = bg - CROSS_CONTRAST * (bg - branch)
    # A slanted line darkens the two sides a few px apart along the branch
    k = np.ones(2 * max(3, int(0.5 * w)) + 1, dtype=np.float32)
    dark = (np.convolve((left < thr).astype(np.float32), k, "same") > 0) & \
           (np.convolve((right < thr).astype(np.float32), k, "same") > 0)
    spans, start = [], None
    for i, on in enumerate(dark):
        if on and start is None:
            start = i
        elif not on and start is not None:
            spans.append((float(t[start]), float(t[i - 1])))
            start = None
    if start is not None:
        spans.append((float(t[start]), float(t[-1])))
    return spans


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
        index, rival (matches of the best other placement, see RIVAL_*) and ambiguous -- or None."""
        if not self.views:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kps, desc = self._orb.detectAndCompute(gray, search)
        if desc is None or len(kps) < MIN_INLIERS:
            return None
        xy = np.float32([k.pt for k in kps])
        fits = []
        for i, v in enumerate(self.views):
            good = [p[0] for p in self._matcher.knnMatch(v["desc"], desc, k=2)
                    if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
            src = v["xy"][[m.queryIdx for m in good]]
            dst = xy[[m.trainIdx for m in good]]
            for _ in range(2):   # the best fit, then the best fit of the matches it left out
                if len(src) < MIN_INLIERS:
                    break
                M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=INLIER_PX)
                if M is None or inl is None or int(inl.sum()) < MIN_INLIERS:
                    break
                inl = inl.reshape(-1).astype(bool)
                A = M[:, :2]
                scale = float(np.sqrt(abs(np.linalg.det(A))))
                if 1.0 / MAX_MATCH_SCALE <= scale <= MAX_MATCH_SCALE:
                    d = A @ v["direction"]
                    fits.append(dict(point=(A @ v["point"] + M[:, 2]).astype(np.float32), direction=d / np.linalg.norm(d),
                                     width=v["width"] * scale, scale=scale, inliers=int(inl.sum()), view=i))
                src, dst = src[~inl], dst[~inl]
        if not fits:
            return None
        best = max(fits, key=lambda f: f["inliers"])
        sep = max(RIVAL_MIN_SEP * best["width"], RIVAL_MIN_SEP_PX)
        best["rival"] = max((f["inliers"] for f in fits if np.linalg.norm(f["point"] - best["point"]) > sep), default=0)
        best["ambiguous"] = best["rival"] >= RIVAL_SHARE * best["inliers"]
        return best

    def needs_view(self, width):
        """True if no stored view is within NEW_VIEW_SCALE of this branch width."""
        if not width or width <= 0:
            return False
        return all(max(width / v["width"], v["width"] / width) >= NEW_VIEW_SCALE for v in self.views)
