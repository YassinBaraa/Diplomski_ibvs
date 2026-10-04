import numpy as np


class PointController:
    def __init__(self, gain=0.5):
        self.gain = float(gain)

    def update_ctx(self, ctx):
        h, w = ctx.frame.shape[:2]
        image_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

        # Post-lock: only use KLT-estimated point. Pre-lock: fall back to detection point.
        if ctx.estimated_point is not None:
            target = ctx.estimated_point
            point_source = "klt_centroid"
        elif ctx.point is not None:
            # Still pre-lock (candidate not confirmed yet)
            target = np.asarray(ctx.point, dtype=np.float32)
            point_source = "detection_prelocked"
        else:
            ctx.debug["control_error_px"] = None
            ctx.debug["velocity_command"] = np.zeros(2, dtype=np.float32)
            ctx.debug["controller"] = {"status": "no_target"}
            return ctx

        # Error = target - center; velocity drives target toward center
        error = (target - image_center).astype(np.float32)
        velocity = (-self.gain * error).astype(np.float32)

        ctx.debug["control_error_px"] = error
        ctx.debug["velocity_command"] = velocity
        ctx.debug["controller"] = {"status": "ok", "point_source": point_source}
        return ctx
