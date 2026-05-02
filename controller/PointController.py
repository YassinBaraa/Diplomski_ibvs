import numpy as np


class PointController:
	def __init__(
		self,
		gain=0.5,
		validation_enabled=True,
		max_feature_motion_px=50.0,
		fallback_enabled=True,
		fallback_gain_scale=0.5,
	):
		self.gain = float(gain)
		self.validation_enabled = bool(validation_enabled)
		self.max_feature_motion_px = float(max_feature_motion_px)
		self.fallback_enabled = bool(fallback_enabled)
		self.fallback_gain_scale = float(fallback_gain_scale)
		self._prev_feature_centroid = None

	def _validate_features(self, features):
		if features is None or len(features) == 0:
			return False, "no_features"

		centroid = np.mean(features, axis=0).astype(np.float32)

		if self._prev_feature_centroid is None:
			self._prev_feature_centroid = centroid
			return True, "ok"

		motion = float(np.linalg.norm(centroid - self._prev_feature_centroid))
		self._prev_feature_centroid = centroid

		if motion > self.max_feature_motion_px:
			return False, f"feature_jump:{motion:.2f}px"

		return True, "ok"

	def update_ctx(self, ctx):
		h, w = ctx.frame.shape[:2]
		features = ctx.extracted_features

		if self.validation_enabled:
			valid, reason = self._validate_features(features)
		else:
			valid, reason = True, "validation_disabled"

		target = None
		point_source = "none"
		effective_gain = self.gain

		if ctx.point is not None:
			target = np.asarray(ctx.point, dtype=np.float32).reshape(2)
			point_source = "detection"
		elif self.fallback_enabled and features is not None and len(features) > 0 and valid:
			target = np.mean(features, axis=0).astype(np.float32).reshape(2)
			point_source = "feature_fallback"
			effective_gain = self.gain * self.fallback_gain_scale

		if target is None:
			ctx.debug["controller"] = {
				"status": "no_target",
				"gain": self.gain,
				"effective_gain": 0.0,
				"point_source": point_source,
				"feature_validation": {"valid": valid, "reason": reason},
			}
			ctx.debug["control_error_px"] = None
			ctx.debug["velocity_command"] = np.array([0.0, 0.0], dtype=np.float32)
			return ctx

		image_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
		error = target - image_center
		velocity = -effective_gain * error

		ctx.debug["controller"] = {
			"status": "ok",
			"gain": self.gain,
			"effective_gain": float(effective_gain),
			"point_source": point_source,
			"feature_validation": {"valid": valid, "reason": reason},
		}
		ctx.debug["control_error_px"] = error.astype(np.float32)
		ctx.debug["velocity_command"] = velocity.astype(np.float32)
		return ctx
