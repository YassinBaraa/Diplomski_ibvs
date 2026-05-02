import cv2
from sources.FrameSource import FrameSource

class MP4Source(FrameSource):
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

    def read(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None, None
        return True, frame, None

    def release(self):
        self.cap.release()