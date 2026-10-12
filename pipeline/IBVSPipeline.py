from pipeline.IBVSContext import IBVSContext
from trackers.TargetModel import MIN_INLIERS, CROSS_MARGIN, CROSS_MARGIN_PX, branch_strip, crossing_spans
import cv2
import logging
import numpy as np
import time

logger = logging.getLogger(__name__)

# A detection counts as the tracked target only if it lies within its own size (or this many
# px, the tracker's confirmation radius) of the tracked point: only then is its size sent with
# that point, and with trust_detections a detection farther away re-locks the tracker.
SIZE_MATCH_MIN_PX = 40.0


def _match_gate(size):
    return max(size or 0.0, SIZE_MATCH_MIN_PX)


MODEL_CHECK_EVERY = 5    # tracking frames between checks of the KLT point against the TargetModel
MIN_SNAP_PX = 8.0        # the KLT point is put back on the stored target when off by more than max(this, 1 width)
# Finding a lost target again: in 2026-10-12 replays the right re-finds had 10-22 matches at a size
# within ~x1.3 of the last tracked one, the wrong ones (a fork below the stored spot) 5-8 matches
# and x0.4-2.6. A re-find needs both; otherwise nothing is sent until one does (or the operator's
# reset) -- the point never moves to a doubtful spot.
REFIND_MIN_INLIERS = 10
REFIND_SCALE = 1.6
VIEW_AFTER_CHECKS = 3    # agreeing checks in a row (since the last re-find / put-back) before adding a view


def _view_region(shape, point, direction, width):
    """Where a TargetModel view takes its keypoints: a strip along the branch, +-10 widths along it
    and +-2 across. In 2026-10-12 replays a wider strip (+-6 across) matched background and put
    the point off the branch, a longer one (+-30 along) stopped matching after the first loss.
    (Leaving out the crossings, as for the KLT features, left too few keypoints to find it again.)"""
    return branch_strip(shape, point, direction, width)


class IBVSPipeline:
    """Detection -> candidate confirmation -> KLT tracking -> (loss) coast + recovery.

    recovery (dict, all optional):
      enabled            keep descriptors of the tracked target and search for it after a loss
      coast_seconds      after a loss keep sending the last point, moving with its damped velocity,
                         for this long (0 turns the coast off)
      give_up_seconds    forget the lost target after this long and go back to plain detection
      rebind_radius_px   a detection this close (grows with time lost) to the last point is
                         taken to be the lost target

    trust_detections: the source's detections are the target itself (ArUco: ID-checked), so
      the tracker locks on the first one, and re-locks on one that is away from the tracked
      point. KLT only bridges the frames without a detection. In the 2026-10-09 flight the
      tracker followed the ceiling beam for ~20 s while the tag was in view, because detections
      were ignored during tracking.

    size_tracker: measures the sent size itself on every tracking frame (branch mode:
      BranchWidthTracker), instead of taking it from the detection, which only exists while
      searching. It is seeded from the detection's mask when the tracker locks and forgotten
      when the tracker goes back to detection. It also keeps the point on the branch centreline
      and restricts new KLT features to a strip along the branch.

    target_model: remembers the first locked point (TargetModel) so it is always found again at
      the same spot: checked every MODEL_CHECK_EVERY frames while tracking (the point is put back
      if it drifted off), and after a loss the target is only searched through it -- no new point
      is ever chosen until reset_target() (the operator's 'r').
    """

    def __init__(self, source, feature_extractor, tracker=None, controller=None, raw_source=None,
                 recovery=None, trust_detections=False, size_tracker=None, target_model=None):
        self.source = source
        self.feature_extractor = feature_extractor
        self.tracker = tracker
        self.controller = controller
        self.trust_detections = trust_detections
        self.size_tracker = size_tracker
        self.target_model = target_model
        self._since_check = 0
        self._strip_done = False   # features moved onto the branch strip since the last lock
        self._pending_snap = None  # offset to the stored target seen at the last check, awaiting a 2nd
        self._capture_wait = 0     # frames until the next try to store the target
        # for the overlay: this frame's events (LOCKED, LOST, PUT BACK, ...) and where the stored
        # target was last seen relative to the tracked point
        self._events = []
        self.model_offset, self.model_seen_t = None, None
        self._agree = 0            # checks in a row that agreed with the stored target
        self._since_cleanup = 0    # tracking frames since tracked features were last checked for crossings
        self._last_width = None    # branch width while last tracked, and when (for re-finds)
        self._last_width_t = None
        if tracker is not None and size_tracker is not None:
            tracker.feature_region = self._branch_region
            tracker.feature_filter = self._avoid_crossings
        rec = recovery or {}
        self.recovery_enabled = bool(rec.get("enabled", True))
        self.coast_seconds = float(rec.get("coast_seconds", 0.0))
        self.give_up_seconds = float(rec.get("give_up_seconds", 5.0))
        self.rebind_radius_px = float(rec.get("rebind_radius_px", 120.0))
        self._lost_t0 = None
        # Read frames straight from the camera while KLT is tracking, instead of
        # through `source` (branch mode's DetectionPipelineSource, backed by
        # the full Hailo detect + mask/skeleton/scoring postprocessing chain).
        # While tracking, that chain's output is never read -- only the raw frame
        # is -- so running it anyway cost ~145ms/frame for nothing (measured:
        # detect ~77ms + postprocess ~68ms out of a ~170ms frame). The chain runs
        # again while idle or while looking for a lost target. None for ArUco
        # mode / no raw camera available, where `source` is already cheap and
        # this optimization does not apply.
        self.raw_source = raw_source

    def run(self):
        frame_n = 0
        while True:
            t_frame0 = time.monotonic()
            tracking = self.tracker is not None and self.tracker.state == "tracking"
            t_read0 = time.monotonic()
            if tracking and self.raw_source is not None:
                read_result = self.raw_source.read()
                ret, frame = read_result[0], read_result[1]
                point = None
                # A raw camera's metadata (DSJSource: grab time) comes third
                info = read_result[2] if len(read_result) > 2 and isinstance(read_result[2], dict) else {}
                info = {**info, "omit": "detector not run (KLT-only frame)"}
            else:
                read_result = self.source.read()
                ret, frame, point = read_result[:3]
                # Optional fourth value: {"t_frame": grab time, "size": detection's apparent size px}
                info = read_result[3] if len(read_result) > 3 and read_result[3] else {}
            t_read1 = time.monotonic()
            if not ret or frame is None:
                print("[IBVSPipeline] Source exhausted — stopping")
                break

            frame_n += 1
            ctx = IBVSContext(frame=frame, t_frame=info.get("t_frame"))
            ctx.extracted_features = np.empty((0, 2), dtype=np.float32)

            was_tracking = self.tracker is not None and self.tracker.state == "tracking"
            if self.tracker is not None:
                try:
                    self._track(ctx, frame, point, frame_n, info)
                except Exception as e:
                    logger.error(f"Tracker error: {e}", exc_info=True)
                    self.tracker.unlock()
                    self._lost_t0 = None
            elif point is not None:
                ctx.point = np.asarray(point, dtype=np.float32)

            if self.size_tracker is not None and self.tracker is not None:
                if self.target_model is not None:
                    self._check_against_model(ctx, frame, frame_n)
                ctx.target_size, why, raw = self._tracked_size(ctx, frame, point, info, was_tracking)
                self._hold_on_branch(ctx, frame)
            else:
                ctx.target_size, why = self._size_at_sent_point(ctx.estimated_point, point, info)
                raw = info.get("size_raw")
            if ctx.estimated_point is not None:
                raw_txt = "-" if raw is None else f"{raw:.1f}"
                result = f"sent {ctx.target_size:.1f}" if why is None else f"omitted: {why}"
                print(f"[size] frame {frame_n}: detected={'yes' if point is not None else 'no'} "
                      f"gain={info.get('gain') or '-'} raw={raw_txt} -> {result}")

            t_ctrl0 = time.monotonic()
            if self.controller is not None:
                try:
                    self.controller.update_ctx(ctx)
                    ctrl = ctx.debug.get("controller", {})
                    print(f"[IBVSPipeline] Frame {frame_n}: controller — "
                          f"source={ctrl.get('point_source','none')}, "
                          f"error={ctx.debug.get('control_error_px')}, "
                          f"vel={ctx.debug.get('velocity_command')}")
                except Exception as e:
                    logger.error(f"Controller error: {e}", exc_info=True)
                    ctx.debug["velocity_command"] = np.zeros(2, dtype=np.float32)
            t_ctrl1 = time.monotonic()

            print(f"[IBVSPipeline] Frame {frame_n}: TIMING — "
                  f"source_read={1000*(t_read1-t_read0):.0f}ms "
                  f"tracking_and_extraction={1000*(t_ctrl0-t_read1):.0f}ms "
                  f"controller={1000*(t_ctrl1-t_ctrl0):.0f}ms "
                  f"frame_total={1000*(t_ctrl1-t_frame0):.0f}ms")

            ctx.debug["events"], self._events = self._events, []
            yield ctx

    def _event(self, name, detail=""):
        self._events.append((name, detail))

    def reset_target(self):
        """Operator reset ('r'): forget the target; the next confirmed detection becomes the new one."""
        if self.tracker is not None:
            self.tracker.unlock()
        if self.target_model is not None:
            self.target_model.reset()
        if self.size_tracker is not None:
            self.size_tracker.reset()
        self._lost_t0, self._strip_done, self._since_check, self._pending_snap = None, False, 0, None
        self._agree, self._last_width, self._last_width_t = 0, None, None
        self.model_offset = None
        self._event("RESET", "target dropped")

    def _branch_region(self, shape, point):
        """Where new KLT features may come from: a strip along the tracked branch, so they move with
        it (around a thin stick, a circle is mostly ceiling, pipe and wire, which move differently)."""
        st = self.size_tracker
        if not st.seeded or not st.width:
            return None
        w = st.width
        return branch_strip(shape, point, st.direction, w, half_len=max(10.0 * w, 100.0), half_wid=max(1.5 * w, 6.0))

    def _bad_features(self, frame, point, pts):
        """[N] bool: features where a background line crosses behind the branch, or off the branch
        strip. Crossings give the strongest corners near the point, but slide along the branch as the
        UAV moves (16:11 and 17:13 flights: the point slid 70-200 px with them). None if unknown."""
        st = self.size_tracker
        if not st.seeded or not st.width or not len(pts):
            return None
        w = st.width
        d = np.asarray(st.direction, dtype=np.float32)
        d = d / max(float(np.linalg.norm(d)), 1e-6)
        rel = np.asarray(pts, dtype=np.float32).reshape(-1, 2) - np.asarray(point, dtype=np.float32)
        t, across = rel @ d, np.abs(rel @ np.array([-d[1], d[0]], dtype=np.float32))
        bad = (across > max(1.5 * w, 6.0)) | (np.abs(t) > max(10.0 * w, 100.0))
        m = max(CROSS_MARGIN * w, CROSS_MARGIN_PX)
        for t0, t1 in crossing_spans(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), point, d, w, max(10.0 * w, 100.0)):
            bad |= (t >= t0 - m) & (t <= t1 + m)
        return bad

    def _avoid_crossings(self, frame, point, pts):
        """KLT feature_filter: new features minus bad ones (_bad_features), unless too few remain."""
        bad = self._bad_features(frame, point, pts)
        if bad is None or (~bad).sum() < self.tracker.min_features:
            return pts
        return pts[~bad]

    def _clean_features(self, frame):
        """Every MODEL_CHECK_EVERY tracking frames: drop tracked features that drifted onto a crossing
        or off the branch; when that leaves fewer than twice the minimum, take fresh ones (point kept).
        Not every frame, and never down to the bare minimum: both made the loop 2x slower and lost the
        target close to the branch (17:36 flight)."""
        tr = self.tracker
        self._since_cleanup += 1
        if self._since_cleanup < MODEL_CHECK_EVERY:
            return
        self._since_cleanup = 0
        bad = self._bad_features(frame, tr.point, tr.points)
        if bad is None or not bad.any():
            return
        if (~bad).sum() >= 2 * tr.min_features:
            tr.keep_features(~bad)
        else:
            tr.refresh_features(frame)

    def _tracked_size(self, ctx, frame, point, info, was_tracking):
        """(size, reason, raw) from size_tracker, measured at the sent point on this frame."""
        st, state = self.size_tracker, self.tracker.state
        if state == "idle":
            st.reset()
        if state != "tracking" or ctx.estimated_point is None:
            return None, "no point sent" if ctx.estimated_point is None else "target lost", None
        # Locked on a new target this frame, or not seeded yet: learn the branch from this frame's
        # detection mask, if that detection is the target. A target found again through the
        # TargetModel was already re-seeded by it (expect()).
        mask = info.get("mask")
        new_target = not was_tracking and not (self.target_model is not None and self.target_model.active)
        if (new_target or not st.seeded) and mask is not None and point is not None and \
                np.linalg.norm(np.asarray(point, dtype=np.float32) - ctx.estimated_point) \
                <= _match_gate(info.get("size_raw")):
            st.seed(frame, ctx.estimated_point, mask, info.get("size_raw"))
        return st.measure(frame, ctx.estimated_point)

    def _hold_on_branch(self, ctx, frame):
        """After a width was measured: keep the point on the branch centreline (sideways only),
        move the features onto the branch strip once, and store the target the first time."""
        st, tr = self.size_tracker, self.tracker
        ap = st.axis_point
        if tr.state != "tracking" or ctx.estimated_point is None or ap is None:
            return
        self._last_width, self._last_width_t = st.width, time.monotonic()
        delta = np.asarray(ap, dtype=np.float32) - tr.point
        if 0.5 < float(np.linalg.norm(delta)) <= max(st.width, 4.0):
            tr.shift(delta)
            ctx.estimated_point = np.round(tr.point).astype(int)
        if not self._strip_done:
            self._strip_done = tr.refresh_features(frame)
            if self._strip_done:
                ctx.extracted_features = tr.points
        else:
            self._clean_features(frame)
            ctx.extracted_features = tr.points
        if self.target_model is not None and not self.target_model.active:
            self._capture_wait -= 1
            if self._capture_wait <= 0:
                self._capture_wait = MODEL_CHECK_EVERY
                w, d, p = st.width, st.direction, tr.point
                if self.target_model.add_view(frame, p, d, w, _view_region(frame.shape, p, d, w)):
                    print(f"[IBVSPipeline] Target stored at {np.round(p).astype(int)} (width {w:.1f}px)")
                    self._event("TARGET STORED", f"{w:.1f} px")

    def _check_against_model(self, ctx, frame, frame_n):
        """Every MODEL_CHECK_EVERY tracking frames: find the stored target near the KLT point. Off by
        more than a branch width on two checks in a row (same offset) -> put the point back there.
        Otherwise add a view if the target's scale is new."""
        tm, tr, st = self.target_model, self.tracker, self.size_tracker
        if not tm.active or tr.state != "tracking" or ctx.estimated_point is None:
            return
        self._since_check += 1
        if self._since_check < MODEL_CHECK_EVERY:
            return
        self._since_check = 0
        est = tr.point
        w = st.width if st.seeded and st.width else 10.0
        r = int(max(12.0 * w, 150.0))
        h, wf = frame.shape[:2]
        x, y = int(est[0]), int(est[1])
        search = np.zeros((h, wf), dtype=np.uint8)
        search[max(y - r, 0):y + r, max(x - r, 0):x + r] = 255
        res = tm.locate(frame, search)
        if res is None:
            return
        offset = res["point"] - est
        gap, tol = float(np.linalg.norm(offset)), max(w, MIN_SNAP_PX)
        if gap > tol:
            # Only a strong, unambiguous, same-scale match may move the point: in the 2026-10-12 16:11 flight two
            # corrections from 7 matches moved it and turned the width measurement along the stick
            # (size then stayed off for 100-170 frames). A rejected one also clears a half-confirmed one.
            if res["inliers"] < REFIND_MIN_INLIERS or res["ambiguous"] or \
                    (st.confirmed and max(res["width"] / st.width, st.width / res["width"]) > REFIND_SCALE):
                self._pending_snap = None
                return
            self.model_offset, self.model_seen_t = offset, time.monotonic()
            # The second offset may differ by half the first: a point sliding along the stick (KLT
            # features slide along its edges) changes the offset every check, and was never put back
            # (16:11 flight: 70-130 px below the stored spot for seconds).
            prev = self._pending_snap
            confirmed = prev is not None and \
                float(np.linalg.norm(offset - prev)) <= max(tol, 0.5 * float(np.linalg.norm(prev)))
            self._pending_snap = offset
            if not confirmed:
                return
            self._pending_snap, self._agree = None, 0
            print(f"[IBVSPipeline] Frame {frame_n}: point {gap:.0f}px off the stored target "
                  f"({res['inliers']} matches) — put back")
            self._event("PUT BACK", f"{gap:.0f} px · {res['inliers']} matches")
            self.model_offset = np.zeros(2, dtype=np.float32)
            if not st.confirmed:   # otherwise keep the measured direction and width: only the position is corrected
                st.expect(res["direction"], res["width"])
            tr.unlock()
            if tr.lock_at(frame, res["point"]):
                ctx.estimated_point = np.round(tr.point).astype(int)
                ctx.extracted_features = tr.points
                self._strip_done = True
            else:
                ctx.estimated_point = None   # searched for again from the next frame
            return
        self.model_offset, self.model_seen_t = offset, time.monotonic()
        self._pending_snap = None
        self._agree += 1
        d, wv = (st.direction, st.width) if st.confirmed else (res["direction"], res["width"])
        if self._agree >= VIEW_AFTER_CHECKS and tm.needs_view(wv) and \
                tm.add_view(frame, res["point"], d, wv, _view_region(frame.shape, res["point"], d, wv)):
            print(f"[IBVSPipeline] Frame {frame_n}: target view added at width {wv:.1f}px ({len(tm.views)} views)")
            self._event("VIEW ADDED", f"{wv:.1f} px · {len(tm.views)} views")

    def _reacquire(self, ctx, frame, point, info, frame_n):
        """Target lost but stored: look only for the stored target. The detector merely confirms
        that the spot found is on a branch; it never supplies a new point."""
        res = self.target_model.locate(frame)
        why = None
        if res is not None and res["inliers"] < REFIND_MIN_INLIERS:
            why = f"only {res['inliers']} matches"
        elif res is not None and self._last_width:
            ratio = max(res["width"] / self._last_width, self._last_width / res["width"])
            if ratio > REFIND_SCALE:
                why = f"implausible size change x{ratio:.1f} since last tracked"
        if res is not None and why is None and not self._on_a_branch(res, info, frame.shape):
            why = "off every detected branch"
        if res is None or why is not None:
            print(f"[IBVSPipeline] Frame {frame_n}: searching for the stored target"
                  + (f" (a match was ignored: {why})" if why else ""))
            return
        if self.size_tracker is not None:
            self.size_tracker.expect(res["direction"], res["width"])
        if self.tracker.lock_at(frame, res["point"]):
            ctx.estimated_point = np.round(self.tracker.point).astype(int)
            ctx.extracted_features = self.tracker.points
            self._strip_done, self._since_check, self._pending_snap, self._agree = True, 0, None, 0
            print(f"[IBVSPipeline] Frame {frame_n}: stored target found again "
                  f"({res['inliers']} matches, x{res['scale']:.2f}) — tracking")
            self._event("RE-FOUND", f"{res['inliers']} matches")
            self.model_offset, self.model_seen_t = np.zeros(2, dtype=np.float32), time.monotonic()

    @staticmethod
    def _on_a_branch(res, info, shape):
        """True if the found point lies on (within 2 widths of) a branch the detector segmented this
        frame. If the detector saw no branch at all, only a strong match counts."""
        x, y = (int(v) for v in np.round(res["point"]))
        h, w = shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return False
        bm = info.get("branch_mask")
        if bm is None or not np.any(bm):
            return res["inliers"] >= 2 * MIN_INLIERS
        r = int(max(2.0 * res["width"], 10.0))
        return bool(np.any(bm[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1]))

    @staticmethod
    def _size_at_sent_point(estimated_point, point, info):
        """(size, reason): this frame's detection size, but only if that detection is the target
        being sent; otherwise None and the reason. The sent point is KLT-tracked or coasted, not
        the detection itself, so it may have drifted off it -- a size that belongs to something
        else would corrupt the UAV's time-to-contact."""
        if estimated_point is None:
            return None, "no point sent"
        if point is None:
            return None, info.get("omit") or "no detection this frame"
        size = info.get("size")
        if not size:
            return None, info.get("omit") or "no size measured"
        gap = float(np.linalg.norm(np.asarray(point, dtype=np.float32) - np.asarray(estimated_point, dtype=np.float32)))
        gate = _match_gate(size)
        if gap > gate:
            return None, f"detection {gap:.0f}px from the sent point (> {gate:.0f}px)"
        return size, None

    # Only estimated_point is ever sent (UDP):
    #   idle      nothing sent (point = live candidate, for display only, not yet trusted)
    #   tracking  estimated_point = the tracked perch point
    #   lost      nothing sent unless coast_seconds > 0 (then the coasted point), or the point
    #             once it is found again. An unrelated detection is never sent.
    def _track(self, ctx, frame, point, frame_n, info):
        tr = self.tracker
        state = tr.state

        if state == "tracking":
            tracked, est = tr.update(frame)
            ctx.extracted_features = tracked
            if est is not None:
                gap = None if point is None else float(np.linalg.norm(np.asarray(point, dtype=np.float32) - est))
                if not (self.trust_detections and gap is not None and gap > _match_gate(info.get("size_raw"))):
                    ctx.estimated_point = est
                    print(f"[IBVSPipeline] Frame {frame_n}: KLT tracking — {len(tracked)} features, point={est}")
                    return
                print(f"[IBVSPipeline] Frame {frame_n}: tracked point {gap:.0f}px off the detection — "
                      f"re-locking on the detection")
                self._event("RE-LOCKED", f"{gap:.0f} px off the detection")
                tr.unlock()
            if tr.state == "lost":
                self._lost_t0 = time.monotonic()
                print(f"[IBVSPipeline] Frame {frame_n}: target LOST — coasting/searching")
                self._event("LOST")
                state = "lost"
            else:
                if est is None and self.target_model is not None and self.target_model.active:
                    print(f"[IBVSPipeline] Frame {frame_n}: target LOST — searching for the stored target")
                    self._event("LOST", "searching for the stored target")
                    self.model_offset = None
                state = "idle"

        if state == "lost":
            elapsed = time.monotonic() - (self._lost_t0 or time.monotonic())
            found = tr.recover(frame)
            if not found and point is not None:
                gate = self.rebind_radius_px * (1.0 + elapsed)
                if np.linalg.norm(np.asarray(point, dtype=np.float32) - tr.last_point) <= gate:
                    found = tr.lock_at(frame, point)
            if found:
                self._lost_t0 = None
                ctx.estimated_point = np.round(tr.point).astype(int)
                ctx.extracted_features = tr.points
            elif elapsed > self.give_up_seconds:
                print(f"[IBVSPipeline] Frame {frame_n}: lost for {elapsed:.1f}s — giving up, back to detection")
                self._event("GAVE UP", f"lost {elapsed:.1f} s")
                tr.unlock()
                self._lost_t0 = None
            elif self.coast_seconds > 0 and elapsed <= self.coast_seconds:
                ctx.estimated_point = tr.coast()
            return

        # idle: detection mode
        if point is not None:
            ctx.point = np.asarray(point, dtype=np.float32)
        if self.target_model is not None and self.target_model.active:
            self._reacquire(ctx, frame, point, info, frame_n)   # a stored target is never replaced here
            return
        print(f"[IBVSPipeline] Frame {frame_n}: DETECTION mode — detection_point={point}")
        # A trusted detection is locked on at once; others are confirmed over a few frames first
        locked = (point is not None and tr.lock_at(frame, point)) if self.trust_detections \
            else tr.observe(frame, point)
        if locked:
            ctx.estimated_point = np.round(tr.point).astype(int)
            self._strip_done, self._since_check, self._capture_wait = False, 0, 0
            print(f"[IBVSPipeline] Frame {frame_n}: LOCKED — switching to KLT tracking")
            self._event("LOCKED")
        ctx.extracted_features = tr.points
