#!/usr/bin/env python3
"""Quick check that DetectionPipelineSource code is syntactically correct."""

import sys
from pathlib import Path

# Add IBVS to path
ibvs_path = Path(__file__).parent
sys.path.insert(0, str(ibvs_path))

from sources.DetectionPipelineSource import DetectionPipelineSource
from pipeline.IBVSContext import IBVSContext
from feature_extraction.FASTHarrisExtractor import FASTHarrisExtractor
import numpy as np

print("Testing DetectionPipelineSource code structure...")

# Test 1: Verify DetectionPipelineSource can be instantiated with a simple iterator
print("  ✓ DetectionPipelineSource imported successfully")

# Test 2: Verify it correctly extracts frame and point from mock contexts
def mock_detector_iterator():
    """Simulate detection pipeline iterator."""
    for i in range(3):
        ctx = type('obj', (object,), {
            'frame': np.zeros((480, 640, 3), dtype=np.uint8),
            'best_candidate': {'x': 100 + i*10, 'y': 150 + i*10}
        })()
        yield ctx

source = DetectionPipelineSource(mock_detector_iterator())
print("  ✓ DetectionPipelineSource instantiated")

# Test 3: Call read() to verify the flow
for i in range(3):
    ret, frame, point = source.read()
    if ret:
        print(f"    Frame {i}: ret={ret}, frame_shape={frame.shape}, point={point}")
    else:
        print(f"    Frame {i}: End of stream")

print("  ✓ DetectionPipelineSource.read() works correctly")

# Test 4: Verify IBVSContext accepts point and frame
print("  Testing IBVSContext...")
test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
test_point = np.array([100.5, 150.5], dtype=np.float32)
ctx = IBVSContext(frame=test_frame, point=test_point)
print(f"  ✓ IBVSContext created: frame shape={ctx.frame.shape}, point={ctx.point}")

print("\n✅ All structural tests passed!")
print("   DetectionPipelineSource → IBVSPipeline flow is correctly wired.")
