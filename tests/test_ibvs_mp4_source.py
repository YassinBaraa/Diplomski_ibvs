import os
import tempfile
import unittest

import cv2
import numpy as np

from feature_extraction.FASTHarrisExtractor import FASTHarrisExtractor
from pipeline.IBVSPipeline import IBVSPipeline
from sources.MP4Source import MP4Source


class TestIBVSMP4Source(unittest.TestCase):
    def _create_test_video(self, path: str, width: int = 160, height: int = 120, frames: int = 12):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
        if not writer.isOpened():
            self.skipTest("OpenCV VideoWriter could not open mp4v codec in this environment")

        for i in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            x = 20 + i
            y = 20 + i // 2
            cv2.rectangle(frame, (x, y), (x + 40, y + 30), (255, 255, 255), 2)
            cv2.line(frame, (0, height // 2), (width - 1, height // 2), (180, 180, 180), 1)
            writer.write(frame)
        writer.release()

    def test_pipeline_runs_with_mp4_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "ibvs_test.mp4")
            self._create_test_video(video_path)

            source = MP4Source(video_path)
            extractor = FASTHarrisExtractor(max_features=25)
            pipeline = IBVSPipeline(source=source, feature_extractor=extractor)

            contexts = []
            for idx, ctx in enumerate(pipeline.run()):
                contexts.append(ctx)
                if idx >= 2:
                    break
            source.release()

        self.assertGreater(len(contexts), 0)
        for ctx in contexts:
            self.assertIsNotNone(ctx.frame)
            self.assertIsNotNone(ctx.extracted_features)
            self.assertEqual(ctx.extracted_features.shape[1], 2)


if __name__ == "__main__":
    unittest.main()
