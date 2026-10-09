from pipeline.IBVSContext import IBVSContext
import logging
import numpy as np
import time

logger = logging.getLogger(__name__)

# A detection's size is sent with the tracked point only if the detection lies within its
# own size (or this many px, the tracker's confirmation radius) of that point.
SIZE_MATCH_MIN_PX = 40.0


class IBVSPipeline:
    """Detection -> candidate confirmation -> KLT tracking -> (loss) coast + recovery.

    recovery (dict, all optional):
      enabled            keep descriptors of the tracked target and search for it after a loss
      coast_seconds      after a loss keep sending the last point, moving with its damped velocity,
                         for this long (0 turns the coast off)
      give_up_seconds    forget the lost target after this long and go back to plain detection
      rebind_radius_px   a detection this close (grows with time lost) to the last point is
                         taken to be the lost target
    """

    def __init__(self, source, feature_extractor, tracker=None, controller=None, raw_source=None,
                 recovery=None):
        self.source = source
        self.feature_extractor = feature_extractor
        self.tracker = tracker
        self.controller = controller
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

            if self.tracker is not None:
                try:
                    self._track(ctx, frame, point, frame_n)
                except Exception as e:
                    logger.error(f"Tracker error: {e}", exc_info=True)
                    self.tracker.unlock()
                    self._lost_t0 = None
            elif point is not None:
                ctx.point = np.asarray(point, dtype=np.float32)

            ctx.target_size, why = self._size_at_sent_point(ctx.estimated_point, point, info)
            if ctx.estimated_point is not None:
                raw = info.get("size_raw")
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

            yield ctx

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
        gate = max(size, SIZE_MATCH_MIN_PX)
        if gap > gate:
            return None, f"detection {gap:.0f}px from the sent point (> {gate:.0f}px)"
        return size, None

    # Only estimated_point is ever sent (UDP):
    #   idle      nothing sent (point = live candidate, for display only, not yet trusted)
    #   tracking  estimated_point = the tracked perch point
    #   lost      nothing sent unless coast_seconds > 0 (then the coasted point), or the point
    #             once it is found again. An unrelated detection is never sent.
    def _track(self, ctx, frame, point, frame_n):
        tr = self.tracker
        state = tr.state

        if state == "tracking":
            tracked, est = tr.update(frame)
            ctx.extracted_features = tracked
            if est is not None:
                ctx.estimated_point = est
                print(f"[IBVSPipeline] Frame {frame_n}: KLT tracking — {len(tracked)} features, point={est}")
                return
            if tr.state == "lost":
                self._lost_t0 = time.monotonic()
                print(f"[IBVSPipeline] Frame {frame_n}: target LOST — coasting/searching")
                state = "lost"
            else:
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
                tr.unlock()
                self._lost_t0 = None
            elif self.coast_seconds > 0 and elapsed <= self.coast_seconds:
                ctx.estimated_point = tr.coast()
            return

        # idle: detection mode
        print(f"[IBVSPipeline] Frame {frame_n}: DETECTION mode — detection_point={point}")
        if point is not None:
            ctx.point = np.asarray(point, dtype=np.float32)
        if tr.observe(frame, point):
            ctx.estimated_point = np.round(tr.point).astype(int)
            print(f"[IBVSPipeline] Frame {frame_n}: LOCKED — switching to KLT tracking")
        ctx.extracted_features = tr.points
