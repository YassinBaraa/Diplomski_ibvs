"""Apparent width of the tracked branch, measured from the image edges on every frame.

This is the UDP packet's "size" in branch mode. The segmentation mask can't provide it:
the detector does not run while KLT tracks, and the mask is too coarse to measure growth
anyway (one mask cell is 8 x 4.5 px at 1280x720, ~10 % of a 20 px branch). So the mask is
used once, when the tracker locks, to learn the branch's direction, rough width and whether
it is darker or brighter than its background. From then on, every frame, short intensity
profiles are sampled across the branch at several places around the tracked point; on each,
the two branch edges are found from the gradient and placed to sub-pixel where the profile is
half-way between branch and background (blur does not move that point), and the width is the
mean of the middle half of their distances. Cross-sections where a wire or beam crosses behind
the branch are left out. Only the edges define the sent width; the mask's own width is just a
search hint.
"""
import cv2
import numpy as np

from trackers.TargetModel import crossing_spans

# 17 cross-sections over the same stretch as the earlier 9 (+-2 widths, 48-160 px) and the mean of
# their middle half: at 6-8 px (far away) the width moved ~2 % frame to frame and ~3 % around its
# 1 s trend in the 2026-10-12 flights, which is what the UAV's time-to-contact fit sees.
N_PROFILES = 17        # cross-sections, spaced along the branch around the tracked point
MIN_VALID = 8          # cross-sections that must show both edges
STEP = 0.5             # sample spacing along a cross-section, px
BORDER_MARGIN_PX = 3   # samples this close to the image edge are not used (a branch edge there = touching it)
MIN_WIDTH_PX = 3.0     # narrower than this, the two edges blur together
MAX_SPREAD = 0.2       # cross-sections must agree: median abs deviation / median width
MIN_CONTRAST = 6.0     # gray levels between the branch and the background on each side
# Accepted width ratio to the last accepted width (widens while frames are omitted). Real changes
# reached ~18 %/frame in a hand-held 2026-10-08 clip (a few % in a slow climb); blurred frames
# there jumped +42 % for 2 frames and back.
JUMP = 1.3
# The first width is not checked against the mask's: a thin stick came out 14.8 px in the mask
# and 5.4 px (steady) on its edges in the 2026-10-11 flight. It is taken once two consecutive
# frames agree within this.
CONFIRM_AGREE = 0.15
MIN_SPACING_PX = 3.0   # cross-sections at least this far apart (a crossing wire or knot spoils only a few)
MAX_SPACING_PX = 10.0
CROSSING_GAP_PX = 2.0  # cross-sections this close to a background crossing are left out as well
# Near the branch's end half the cross-sections lie past it (no edges) and the rest on the tapering
# tip (2026-10-13 10:19 flight: size on 12 % of packets). When the edges are found only on one side,
# the cross-sections move that way, a quarter of their stretch per frame, up to half of it.
WINDOW_STEP = 0.25
WINDOW_MAX = 0.5
# A wrong expected width or direction makes every cross-section miss the branch, and both are
# updated only from successful frames, so they never recover: in the 2026-10-12 16:11 flight a weak
# correction set the expected width to 6.7 px on a 16-49 px stick and size stayed off for 100-170
# frames. After this many failed frames in a row, both are re-measured from the image at the point.
REORIENT_AFTER = 3
REORIENT_MIN_DEG = 10.0  # only re-orient when the image disagrees by more than this

_SMOOTH = cv2.getGaussianKernel(9, 1.0 / STEP).T  # 1 px sigma along each cross-section


def _local_extrema(g, thr):
    """Indices of local minima below -thr and local maxima above thr (strongest 4 of each)."""
    prev, mid, nxt = g[:-2], g[1:-1], g[2:]
    mins = np.flatnonzero((mid <= prev) & (mid < nxt) & (mid < -thr)) + 1
    maxs = np.flatnonzero((mid >= prev) & (mid > nxt) & (mid > thr)) + 1
    return mins[np.argsort(g[mins])[:4]], maxs[np.argsort(-g[maxs])[:4]]


def _image_direction(frame, point, width):
    """Direction along the dominant edges around `point` (Gaussian-weighted structure tensor of
    the image gradient), or None if there is no clearly oriented structure there."""
    r = int(np.clip(4.0 * width, 16, 100))
    h, w = frame.shape[:2]
    x, y = int(round(point[0])), int(round(point[1]))
    x0, y0, x1, y1 = max(x - r, 0), max(y - r, 0), min(x + r + 1, w), min(y + r + 1, h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    g = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    wgt = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * (r / 2.0) ** 2))
    jxx, jyy, jxy = float((wgt * gx * gx).sum()), float((wgt * gy * gy).sum()), float((wgt * gx * gy).sum())
    if jxx + jyy <= 0 or np.hypot(jxx - jyy, 2.0 * jxy) / (jxx + jyy) < 0.3:
        return None
    theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)          # dominant gradient direction
    return np.array([-np.sin(theta), np.cos(theta)])        # along the edges


def _image_width(frame, point, direction, dark, width):
    """Width [px] of the dark (or bright) band across `direction` nearest to `point`, from one
    wide intensity profile (half-way between band and background), or None."""
    d = np.asarray(direction, dtype=float)
    n = np.array([-d[1], d[0]])
    R = float(np.clip(8.0 * width, 40.0, 160.0))
    ss = np.arange(-R, R + STEP / 2, STEP)
    xy = np.asarray(point, dtype=float)[None, :] + ss[:, None] * n
    h, w = frame.shape[:2]
    keep = (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] <= w - 1) & (xy[:, 1] <= h - 1)
    if keep.sum() < 20:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p = cv2.remap(gray, xy[keep, 0].astype(np.float32).reshape(1, -1), xy[keep, 1].astype(np.float32).reshape(1, -1),
                  cv2.INTER_LINEAR).astype(np.float32).ravel()
    p = cv2.GaussianBlur(p.reshape(1, -1), (0, 0), 2.0).ravel()
    if not dark:
        p = -p
    bg = float(np.percentile(p, 80))                   # background: the bright majority
    c = int(np.argmin(np.abs(ss[keep])))               # sample at the point
    lo = int(np.argmin(p[max(c - int(4 * width / STEP), 0):c + int(4 * width / STEP) + 1])) + max(c - int(4 * width / STEP), 0)
    level = 0.5 * (p[lo] + bg)
    if bg - p[lo] < MIN_CONTRAST:
        return None
    a = lo
    while a > 0 and p[a - 1] < level:
        a -= 1
    b = lo
    while b < len(p) - 1 and p[b + 1] < level:
        b += 1
    if a == 0 or b == len(p) - 1:                     # band runs off the profile: no clean edges
        return None
    return (b - a) * STEP


def _med(a):
    """Median of a short 1-D array (np.median costs ~50 us of overhead per call, 3 calls per cross-section)."""
    a = np.sort(a)
    n = len(a)
    return 0.5 * float(a[(n - 1) // 2] + a[n // 2])


def _crossing(p, i0, level, radius):
    """Sub-sample position where profile p crosses `level`, nearest to sample i0, or None."""
    a, b = max(i0 - radius, 0), min(i0 + radius, len(p) - 1)
    s = p[a:b + 1] - level
    hit = np.flatnonzero((s[:-1] == 0) | (s[:-1] * s[1:] < 0)) + a
    if len(hit) == 0:
        return None
    k = hit[np.argmin(np.abs(hit - i0))]
    den = p[k + 1] - p[k]
    return float(k) if den == 0 else k + (level - p[k]) / den


def _crossings(frame, point, direction, width, half_len):
    """crossing_spans() around `point`, computed on a small crop only."""
    reach = 0.5 * width + 3.0 + max(4.0, 0.6 * width) + 2.0
    d = np.asarray(direction, dtype=np.float64)
    n = np.array([-d[1], d[0]])
    c = np.asarray(point, dtype=np.float64)
    corners = np.array([c + a * half_len * d + b * reach * n for a in (-1, 1) for b in (-1, 1)])
    H, W = frame.shape[:2]
    x0, y0 = np.clip(np.floor(corners.min(axis=0)).astype(int), 0, [W - 1, H - 1])
    x1, y1 = np.clip(np.ceil(corners.max(axis=0)).astype(int) + 1, 1, [W, H])
    if x1 - x0 < 4 or y1 - y0 < 4:
        return []
    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return crossing_spans(gray, c - [x0, y0], d, width, half_len)


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
        self._pending = None      # unconfirmed width of the previous frame
        self._missed = 0          # frames since the last accepted width
        self._centre_t = 0.0      # [px] along the branch from the point to the middle of the cross-sections
        self.last_edges = []      # [(left_xy, right_xy)] found on the last measured frame, for drawing
        self.axis_point = None    # the tracked point moved sideways onto the branch centreline (last accepted frame)
        # for the overlay: every cross-section of the last measurement, {centre, edges or None}, and its normal/half-length
        self.last_profiles, self.last_normal, self.last_half = [], None, 0.0

    @property
    def direction(self):
        return None if self._dir is None else self._dir.copy()

    @property
    def width(self):
        return self._width

    @property
    def dark(self):
        """The branch is darker than its background."""
        return self._dark

    @property
    def confirmed(self):
        """The width was measured on the edges (not only the mask's or the TargetModel's guess)."""
        return self._confirmed

    def expect(self, direction, width, dark=None):
        """The branch was found again (TargetModel) at this direction and width: search there and
        re-confirm the width over 2 frames, without comparing it to the one before the loss."""
        d = np.asarray(direction, dtype=np.float64)
        self._dir, self._width = d / np.linalg.norm(d), float(width)
        if dark is not None:
            self._dark = bool(dark)
        self._confirmed, self._pending, self._missed, self.seeded = False, None, 0, True
        self._centre_t = 0.0

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
        self._confirmed, self._pending, self._missed, self.seeded = False, None, 0, True
        self._centre_t = 0.0
        return True

    def _follow_edges(self, found_t, half_span):
        """After a failed frame: move the cross-sections toward where edges were found, if that is
        clearly one side of them (the other side being past the branch's end)."""
        if len(found_t) < 2:
            return
        lean = float(np.mean(found_t)) - self._centre_t
        if abs(lean) > 0.3 * half_span:
            self._centre_t += float(np.sign(lean)) * WINDOW_STEP * 2 * half_span

    def measure(self, frame, point):
        """(width, reason, raw): the branch width at `point` on this frame, or None and why it is
        left out; raw is the measured width even when it is not accepted (None if not measured)."""
        self.axis_point = None
        self.last_profiles = []
        if not self.seeded:
            return None, "branch not seeded (no mask at lock)", None
        w_prev, d = self._width, self._dir
        c = np.asarray(point, dtype=np.float64)
        if self._missed >= REORIENT_AFTER and self._missed % REORIENT_AFTER == 0:
            d_img = _image_direction(frame, c, w_prev)
            if d_img is not None:
                d_img = d_img if np.dot(d_img, d) >= 0 else -d_img
                turn = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(d_img, d)))))))
                if turn > REORIENT_MIN_DEG:
                    print(f"[BranchWidth] direction re-oriented from the image by {turn:.0f} deg "
                          f"after {self._missed} failed frames")
                    self._dir = d = d_img
            w_img = _image_width(frame, c, d, self._dark, w_prev)
            if w_img is not None and w_img >= MIN_WIDTH_PX and max(w_img / w_prev, w_prev / w_img) > 1.3:
                print(f"[BranchWidth] expected width re-measured from the image: {w_prev:.1f} -> {w_img:.1f}px "
                      f"after {self._missed} failed frames")
                self._width = w_prev = w_img
                self._confirmed, self._pending = False, None   # re-confirmed over 2 agreeing frames
        n = np.array([-d[1], d[0]])

        # Sample points: N_PROFILES cross-sections along the branch, each +-(1.5 widths + 6 px)
        spacing = float(np.clip(0.25 * w_prev, MIN_SPACING_PX, MAX_SPACING_PX))
        half_span = (N_PROFILES - 1) / 2.0 * spacing
        self._centre_t = float(np.clip(self._centre_t, -WINDOW_MAX * 2 * half_span, WINDOW_MAX * 2 * half_span))
        ts = (np.arange(N_PROFILES) - (N_PROFILES - 1) / 2.0) * spacing + self._centre_t
        ss = np.arange(-(1.5 * w_prev + 6.0), 1.5 * w_prev + 6.0 + STEP / 2, STEP)
        X = c[0] + ts[:, None] * d[0] + ss[None, :] * n[0]
        Y = c[1] + ts[:, None] * d[1] + ss[None, :] * n[1]
        H, W = frame.shape[:2]
        m = BORDER_MARGIN_PX
        # Samples within m px of the image edge are not used. A cross-section is cut there rather
        # than dropped (dropping whole ones near the edge cost 145 frames of size in a replay):
        # the branch only counts as touching the edge if its own edges get that close.
        ok_s = (X >= m) & (Y >= m) & (X <= W - 1 - m) & (Y <= H - 1 - m)
        inside = ok_s.sum(axis=1) * STEP >= w_prev + 8.0   # room for the branch and some background
        if inside.sum() < MIN_VALID:
            self._missed += 1
            return None, f"cross-section within {m}px of the image edge", None

        # Where a wire or beam crosses behind the branch the background beside it is dark and the
        # edges there are wrong: those cross-sections are left out, unless too few would remain.
        on_crossing = np.zeros(N_PROFILES, dtype=bool)
        for t0, t1 in _crossings(frame, c, d, w_prev, float(np.abs(ts).max())):
            on_crossing |= (ts >= t0 - CROSSING_GAP_PX) & (ts <= t1 + CROSSING_GAP_PX)
        if (inside & ~on_crossing).sum() >= MIN_VALID:
            inside &= ~on_crossing

        Xc, Yc = np.clip(X, 0, W - 1), np.clip(Y, 0, H - 1)
        bx0, by0 = int(np.floor(Xc[inside].min())), int(np.floor(Yc[inside].min()))
        bx1, by1 = int(np.ceil(Xc[inside].max())) + 1, int(np.ceil(Yc[inside].max())) + 1
        gray = cv2.cvtColor(frame[by0:by1 + 1, bx0:bx1 + 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        prof = cv2.remap(gray, (Xc[inside] - bx0).astype(np.float32), (Yc[inside] - by0).astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        ok_s = ok_s[inside]
        prof = cv2.filter2D(prof, -1, _SMOOTH, borderType=cv2.BORDER_REPLICATE)
        g = np.gradient(prof, axis=1) / STEP          # gray levels per px, along +n
        if not self._dark:
            g = -g                                    # entering the branch is always a fall
        lo_p, hi_p = (0.4, 2.5) if self._confirmed else (0.2, 3.0)

        widths, axis_pts, mids, self.last_edges, found_t = [], [], [], [], []
        self.last_profiles = [dict(centre=c + t * d, edges=None) for t in ts]
        self.last_normal, self.last_half = n, float(ss[-1])
        for k, t in zip(np.flatnonzero(inside), range(int(inside.sum()))):
            gk = g[t]
            v = np.flatnonzero(ok_s[t])
            v0, v1 = int(v[0]), int(v[-1])          # usable samples (a straight cut is one stretch)
            falls, rises = _local_extrema(gk[v0:v1 + 1], max(3.0, 0.25 * float(np.abs(gk[v0:v1 + 1]).max())))
            falls, rises = falls + v0, rises + v0
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
            out_l, out_r = pk[max(li - 2 * o, v0): max(li - o, v0) + 1], pk[ri + o: min(ri + 2 * o, v1) + 1]
            if li - o < v0 or ri + o > v1:   # no background beyond an edge: it is at the image edge
                continue
            if len(body) == 0 or len(out_l) == 0 or len(out_r) == 0:
                continue
            v_in, v_l, v_r = _med(body), _med(out_l), _med(out_r)
            if min(abs(v_in - v_l), abs(v_in - v_r)) < MIN_CONTRAST:
                continue
            xl, xr = _crossing(pk, li, 0.5 * (v_in + v_l), o), _crossing(pk, ri, 0.5 * (v_in + v_r), o)
            if xl is None or xr is None or xr <= xl or xl < v0 or xr > v1:
                continue
            left, right = ss[0] + xl * STEP, ss[0] + xr * STEP
            widths.append(right - left)
            found_t.append(ts[k])
            axis_pts.append(c + ts[k] * d + 0.5 * (left + right) * n)
            mids.append(0.5 * (left + right))
            self.last_edges.append((c + ts[k] * d + left * n, c + ts[k] * d + right * n))
            self.last_profiles[k]["edges"] = self.last_edges[-1]

        if len(widths) < MIN_VALID:
            self._missed += 1
            self._follow_edges(found_t, half_span)
            return None, f"edges found on only {len(widths)}/{N_PROFILES} cross-sections", None
        widths = np.sort(np.asarray(widths))
        med = float(np.median(widths))
        q = len(widths) // 4
        mid_mean = float(np.mean(widths[q:len(widths) - q]))   # mean of the middle half

        # Branch axis through the cross-section centres: corrects for a direction that was off
        # (width measured along the old normal is too wide by 1/cos of the error)
        a = np.asarray(axis_pts)
        evals, evecs = np.linalg.eigh(np.cov((a - a.mean(axis=0)).T))
        d_new = evecs[:, np.argmax(evals)]
        d_new = d_new if np.dot(d_new, d) >= 0 else -d_new
        raw = mid_mean * abs(float(np.dot(n, [-d_new[1], d_new[0]])))

        spread = float(np.median(np.abs(widths - med))) / med
        if spread > MAX_SPREAD:
            self._missed += 1
            self._follow_edges(found_t, half_span)
            return None, f"cross-sections disagree ({100 * spread:.0f}% spread)", raw
        if raw < MIN_WIDTH_PX:
            self._missed += 1
            return None, f"branch narrower than {MIN_WIDTH_PX:.0f}px", raw
        if not self._confirmed:
            # The next search works at this scale; the width is sent once the next frame agrees
            pending, self._pending, self._width = self._pending, raw, raw
            if pending is None or abs(raw / pending - 1.0) > CONFIRM_AGREE:
                self._missed += 1
                return None, "first width not confirmed yet (needs 2 agreeing frames)", raw
        else:
            k = 1.0 + self._missed / 10.0
            if not (1.0 / JUMP) ** k <= raw / w_prev <= JUMP ** k:
                self._missed += 1
                return None, f"width jumped {w_prev:.1f} -> {raw:.1f}px", raw

        self._width, self._dir, self._confirmed, self._missed = raw, d_new / np.linalg.norm(d_new), True, 0
        # centreline at the tracked point: its sideways offset is the median of the cross-section centres
        self.axis_point = c + float(np.median(mids)) * n
        return raw, None, raw
