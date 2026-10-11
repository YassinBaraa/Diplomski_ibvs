"""Apparent width of the tracked branch, measured from the image edges on every frame.

This is the UDP packet's "size" in branch mode. The segmentation mask can't provide it:
the detector does not run while KLT tracks, and the mask is too coarse to measure growth
anyway (one mask cell is 8 x 4.5 px at 1280x720, ~10 % of a 20 px branch). So the mask is
used once, when the tracker locks, to learn the branch's direction, rough width and whether
it is darker or brighter than its background. From then on, every frame, short intensity
profiles are sampled across the branch at several places around the tracked point; on each,
the two branch edges are found from the gradient and placed to sub-pixel where the profile is
half-way between branch and background (blur does not move that point), and the width is the
median of their distances. Only the edges define the sent width; the mask's own width is just
a search hint.
"""
import cv2
import numpy as np

N_PROFILES = 9         # cross-sections, spaced along the branch around the tracked point
MIN_VALID = 5          # cross-sections that must show both edges
STEP = 0.5             # sample spacing along a cross-section, px
BORDER_MARGIN_PX = 3   # a cross-section this close to the image edge is not used
MIN_WIDTH_PX = 3.0     # narrower than this, the two edges blur together
MAX_SPREAD = 0.2       # cross-sections must agree: median abs deviation / median width
MIN_CONTRAST = 6.0     # gray levels between the branch and the background on each side
# Accepted width ratio to the last accepted width (widens while frames are omitted). Real changes
# reached ~18 %/frame in a hand-held 2026-10-08 clip (a few % in a slow climb); blurred frames
# there jumped +42 % for 2 frames and back.
JUMP = 1.3

_SMOOTH = cv2.getGaussianKernel(9, 1.0 / STEP).T  # 1 px sigma along each cross-section


def _local_extrema(g, thr):
    """Indices of local minima below -thr and local maxima above thr (strongest 4 of each)."""
    i = np.arange(1, len(g) - 1)
    mins = i[(g[i] <= g[i - 1]) & (g[i] < g[i + 1]) & (g[i] < -thr)]
    maxs = i[(g[i] >= g[i - 1]) & (g[i] > g[i + 1]) & (g[i] > thr)]
    return mins[np.argsort(g[mins])[:4]], maxs[np.argsort(-g[maxs])[:4]]


def _crossing(p, i0, level, radius):
    """Sub-sample position where profile p crosses `level`, nearest to sample i0, or None."""
    j = np.arange(max(i0 - radius, 0), min(i0 + radius, len(p) - 1))
    s = p[j] - level
    hit = j[(s == 0) | (s * (p[j + 1] - level) < 0)]
    if len(hit) == 0:
        return None
    k = hit[np.argmin(np.abs(hit - i0))]
    den = p[k + 1] - p[k]
    return float(k) if den == 0 else k + (level - p[k]) / den


class BranchWidthTracker:

    def __init__(self):
        self.reset()

    def reset(self):
        """Forget the branch (tracker went back to detection)."""
        self.seeded = False
        self._dir = None          # unit vector along the branch, (x, y)
        self._width = None        # last accepted width, px (before the first: the mask's rough width)
        self._dark = True         # branch darker than its background
        self._confirmed = False   # _width was measured on the edges, not only taken from the mask
        self._missed = 0          # frames since the last accepted width
        self.last_edges = []      # [(left_xy, right_xy)] found on the last measured frame, for drawing

    def seed(self, frame, point, mask, width_hint=None):
        """Learn direction, rough width and polarity of the branch from its segmentation mask
        around `point`. Returns False if the mask has too little branch there."""
        h, w = mask.shape[:2]
        r = int(max(3.0 * (width_hint or 10.0), 40.0))
        x, y = int(round(point[0])), int(round(point[1]))
        x0, y0, x1, y1 = max(x - r, 0), max(y - r, 0), min(x + r + 1, w), min(y + r + 1, h)
        m = mask[y0:y1, x0:x1].astype(bool)
        ys, xs = np.nonzero(m)
        if len(xs) < 20 or m.all():
            return False
        pts = np.column_stack([xs, ys]).astype(np.float64)
        evals, evecs = np.linalg.eigh(np.cov((pts - pts.mean(axis=0)).T))
        d = evecs[:, np.argmax(evals)]
        gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        self._dark = float(gray[m].mean()) < float(gray[~m].mean())
        self._dir = d / np.linalg.norm(d)
        # no width at the point (not on the skeleton): mask area / length across the window
        self._width = float(width_hint) if width_hint else len(xs) / (2.0 * r)
        self._confirmed, self._missed, self.seeded = False, 0, True
        return True

    def measure(self, frame, point):
        """(width, reason, raw): the branch width at `point` on this frame, or None and why it is
        left out; raw is the measured width even when it is not accepted (None if not measured)."""
        if not self.seeded:
            return None, "branch not seeded (no mask at lock)", None
        w_prev, d = self._width, self._dir
        n = np.array([-d[1], d[0]])
        c = np.asarray(point, dtype=np.float64)

        # Sample points: N_PROFILES cross-sections along the branch, each +-(1.5 widths + 6 px)
        spacing = float(np.clip(0.5 * w_prev, 3.0, 20.0))
        ts = (np.arange(N_PROFILES) - (N_PROFILES - 1) / 2.0) * spacing
        ss = np.arange(-(1.5 * w_prev + 6.0), 1.5 * w_prev + 6.0 + STEP / 2, STEP)
        X = c[0] + ts[:, None] * d[0] + ss[None, :] * n[0]
        Y = c[1] + ts[:, None] * d[1] + ss[None, :] * n[1]
        H, W = frame.shape[:2]
        m = BORDER_MARGIN_PX
        inside = (X.min(1) >= m) & (Y.min(1) >= m) & (X.max(1) <= W - 1 - m) & (Y.max(1) <= H - 1 - m)
        if inside.sum() < MIN_VALID:
            self._missed += 1
            return None, f"cross-section within {m}px of the image edge", None

        bx0, by0 = int(np.floor(X[inside].min())), int(np.floor(Y[inside].min()))
        bx1, by1 = int(np.ceil(X[inside].max())) + 1, int(np.ceil(Y[inside].max())) + 1
        gray = cv2.cvtColor(frame[by0:by1 + 1, bx0:bx1 + 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        prof = cv2.remap(gray, (X[inside] - bx0).astype(np.float32), (Y[inside] - by0).astype(np.float32),
                         cv2.INTER_LINEAR)
        prof = cv2.filter2D(prof, -1, _SMOOTH, borderType=cv2.BORDER_REPLICATE)
        g = np.gradient(prof, axis=1) / STEP          # gray levels per px, along +n
        if not self._dark:
            g = -g                                    # entering the branch is always a fall
        lo_p, hi_p = (0.4, 2.5) if self._confirmed else (0.33, 3.0)

        widths, axis_pts, self.last_edges = [], [], []
        for k, t in zip(np.flatnonzero(inside), range(int(inside.sum()))):
            gk = g[t]
            falls, rises = _local_extrema(gk, max(3.0, 0.25 * float(np.abs(gk).max())))
            best = None
            for li in falls:
                for ri in rises:
                    if ri <= li:
                        continue
                    wk = (ri - li) * STEP
                    if not lo_p <= wk / w_prev <= hi_p:
                        continue
                    score = abs(np.log(wk / w_prev)) + abs(0.5 * (ss[li] + ss[ri])) / max(w_prev, 10.0)
                    if best is None or score < best[0]:
                        best = (score, li, ri)
            if best is None:
                continue
            # Each edge is where the profile is half-way between the branch and the background
            # next to it: a symmetric blur (motion, focus) widens the ramp but does not move that
            # point, while the steepest point wanders inside the ramp.
            pk, li, ri = prof[t], best[1], best[2]
            o = max((ri - li) // 2, 2)
            body = pk[li + (ri - li) // 4: ri - (ri - li) // 4 + 1]
            out_l, out_r = pk[max(li - 2 * o, 0): max(li - o, 0) + 1], pk[ri + o: ri + 2 * o + 1]
            if len(body) == 0 or len(out_l) == 0 or len(out_r) == 0:
                continue
            v_in, v_l, v_r = float(np.median(body)), float(np.median(out_l)), float(np.median(out_r))
            if min(abs(v_in - v_l), abs(v_in - v_r)) < MIN_CONTRAST:
                continue
            xl, xr = _crossing(pk, li, 0.5 * (v_in + v_l), o), _crossing(pk, ri, 0.5 * (v_in + v_r), o)
            if xl is None or xr is None or xr <= xl:
                continue
            left, right = ss[0] + xl * STEP, ss[0] + xr * STEP
            widths.append(right - left)
            axis_pts.append(c + ts[k] * d + 0.5 * (left + right) * n)
            self.last_edges.append((c + ts[k] * d + left * n, c + ts[k] * d + right * n))

        if len(widths) < MIN_VALID:
            self._missed += 1
            return None, f"edges found on only {len(widths)}/{N_PROFILES} cross-sections", None
        widths = np.asarray(widths)
        med = float(np.median(widths))

        # Branch axis through the cross-section centres: corrects for a direction that was off
        # (width measured along the old normal is too wide by 1/cos of the error)
        a = np.asarray(axis_pts)
        evals, evecs = np.linalg.eigh(np.cov((a - a.mean(axis=0)).T))
        d_new = evecs[:, np.argmax(evals)]
        d_new = d_new if np.dot(d_new, d) >= 0 else -d_new
        raw = med * abs(float(np.dot(n, [-d_new[1], d_new[0]])))

        spread = float(np.median(np.abs(widths - med))) / med
        if spread > MAX_SPREAD:
            self._missed += 1
            return None, f"cross-sections disagree ({100 * spread:.0f}% spread)", raw
        if raw < MIN_WIDTH_PX:
            self._missed += 1
            return None, f"branch narrower than {MIN_WIDTH_PX:.0f}px", raw
        k = 1.0 + self._missed / 10.0
        lo, hi = ((1.0 / JUMP) ** k, JUMP ** k) if self._confirmed else (0.5, 2.0)
        if not lo <= raw / w_prev <= hi:
            self._missed += 1
            return None, f"width jumped {w_prev:.1f} -> {raw:.1f}px", raw

        self._width, self._dir, self._confirmed, self._missed = raw, d_new / np.linalg.norm(d_new), True, 0
        return raw, None, raw
