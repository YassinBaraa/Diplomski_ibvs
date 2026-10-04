"""ArUco-marker point source for the IBVS pipeline.

Alternative to DetectionPipelineSource: instead of running the branch
segmentation pipeline, it detects a known ArUco marker directly on every
frame and exposes the same (frame, point) return value as
DetectionPipelineSource. IBVSPipeline reads only that contract, so KLT
tracking / feature extraction / the controller are unaffected by which mode
produced the point.

ArUco detection is essentially exact and false-positive-free
(checksum/ID-verified): every frame is detected fresh and handed straight to
IBVSPipeline, which confirms the point over a few frames and locks KLT on it.

Deliberately has no imports from the `sources` or `pipeline` packages:
ibvs/ and detection_pipeline/ each define top-level packages with those
same names, and a caller that needs a raw camera source from
detection_pipeline (NiclaSource/MP4Source) alongside this class runs into
a name collision on sys.modules['sources']. Keeping this file
self-contained lets callers load it by file path (see
UDP_client/main_record.py) regardless of which `sources` package is
currently active on sys.path.
"""
import cv2
import numpy as np

# "auto" tries every dictionary below per frame and uses whichever one actually
# finds a marker. A generated/printed tag can come from any of these families
# and there's no way to tell which from the image alone -- guessing wrong means
# detectMarkers() silently returns zero detections forever, which is exactly
# the "never locks" symptom this exists to avoid. Pass a specific name (e.g.
# "DICT_4X4_50") instead once you know your tag's dictionary, to skip the scan
# and detect on every frame at minimal cost.
DEFAULT_DICTIONARY = "auto"

_CANDIDATE_DICTIONARIES = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
]


def _make_single_detector(dictionary_name):
    """Returns a detect(gray) -> (corners, ids) callable for one predefined
    dictionary, compatible with both the pre- and post-4.7 OpenCV cv2.aruco APIs."""
    if hasattr(cv2.aruco, "ArucoDetector"):
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        return lambda gray: detector.detectMarkers(gray)[:2]
    else:
        dictionary = cv2.aruco.Dictionary_get(getattr(cv2.aruco, dictionary_name))
        params = cv2.aruco.DetectorParameters_create()
        return lambda gray: cv2.aruco.detectMarkers(gray, dictionary, parameters=params)[:2]


def _make_marker_detector(dictionary_name):
    if dictionary_name != "auto":
        return _make_single_detector(dictionary_name)

    detectors = [(name, _make_single_detector(name)) for name in _CANDIDATE_DICTIONARIES]

    def detect_any(gray):
        for name, detect in detectors:
            corners, ids = detect(gray)
            if ids is not None and len(corners) > 0:
                return corners, ids
        return [], None

    return detect_any


class ArucoSource:
    def __init__(self, frame_source, dictionary: str = DEFAULT_DICTIONARY):
        self.frame_source = frame_source
        self._detect = _make_marker_detector(dictionary)


    def _detect_center(self, frame):
        # Any marker from the configured dictionary counts -- this is a single-tag
        # perch/land setup, not a multi-tag identification task, so there's no ID to match.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = self._detect(gray)
        if ids is None or len(corners) == 0:
            return None
        return corners[0][0].mean(axis=0)

    def read(self):
        read_result = self.frame_source.read()
        ret, frame = read_result[0], read_result[1]

        if not ret or frame is None:
            return False, None, None

        center = self._detect_center(frame)
        point = np.round(center).astype(int) if center is not None else None
        return True, frame, point

    def release(self):
        self.frame_source.release()
