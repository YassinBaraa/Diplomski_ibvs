# IBVS - Restructured for Easy ROS Integration

## New Structure: Everything in `ibvs_ros/`

The entire IBVS package is now self-contained in `ibvs_ros/` directory, making it easy to import into any ROS workspace.

```
ibvs/                          ← catkin package root
├── ibvs_ros/                  ← Main Python package (ALL CODE HERE)
│   ├── __init__.py
│   ├── ibvs_node.py           ← ROS node entry point
│   ├── config/
│   │   └── default_config.yaml
│   ├── controller/
│   │   ├── __init__.py
│   │   └── PointController.py
│   ├── feature_extraction/
│   │   ├── __init__.py
│   │   ├── AprilTagExtractor.py
│   │   ├── FASTHarrisExtractor.py
│   │   └── FeatureSelector.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── IBVSContext.py
│   │   └── IBVSPipeline.py
│   ├── sources/
│   │   ├── __init__.py
│   │   ├── DetectionPipelineSource.py
│   │   ├── FrameSource.py
│   │   ├── MP4Source.py
│   │   └── apriltag_test.mp4
│   ├── ros_config/
│   │   ├── __init__.py
│   │   └── ibvs_ros.yaml
│   └── ros_launch/
│       └── ibvs.launch
├── model/                     ← Model weights directory
├── tests/                     ← Unit tests
├── CMakeLists.txt
├── package.xml
├── main.py                    ← Standalone (non-ROS) entry point
├── setup.py
├── requirements.txt
└── README.md
```

## How to Use in Your ROS Workspace

### Option 1: Direct Workspace Integration

```bash
# Copy entire ibvs folder to your catkin workspace
cp -r /path/to/ibvs ~/catkin_ws/src/

# Build
cd ~/catkin_ws
catkin build ibvs

# Run
roslaunch ibvs ibvs.launch
```

### Option 2: Using as External Dependency

```bash
# In your package that uses IBVS:
from ibvs_ros.controller import PointController
from ibvs_ros.pipeline import IBVSContext
from ibvs_ros.feature_extraction import AprilTagExtractor
```

## What Changed

✅ **All code now in `ibvs_ros/`**
- `controller/` → `ibvs_ros/controller/`
- `feature_extraction/` → `ibvs_ros/feature_extraction/`
- `pipeline/` → `ibvs_ros/pipeline/`
- `sources/` → `ibvs_ros/sources/`

✅ **All imports updated to relative paths**
- Old: `from controller.PointController import PointController`
- New: `from .controller.PointController import PointController`

✅ **ROS-specific files organized**
- Launch files: `ibvs_ros/ros_launch/`
- ROS config: `ibvs_ros/ros_config/`
- Core config: `ibvs_ros/config/`

✅ **Self-contained package**
- Can import `ibvs_ros` from anywhere after `sys.path` adjustment
- No dependencies on root-level folders
- All 16 Python modules in one place

## Standalone Mode Still Works

```bash
cd ibvs/
python main.py --source=detection --max-frames=50
```

## Verification

All imports have been tested:
```bash
✓ from ibvs_ros.controller.PointController import PointController
✓ from ibvs_ros.pipeline.IBVSContext import IBVSContext
✓ from ibvs_ros.feature_extraction.AprilTagExtractor import AprilTagExtractor
✓ from ibvs_ros.sources.MP4Source import MP4Source
```

## ROS Topics Published

All topics under `/ibvs/` namespace:
- `/ibvs/cmd_vel` - Control velocity (Twist)
- `/ibvs/error` - Control error in pixels (Vector3Stamped)
- `/ibvs/extracted_points` - Frame with features (Image)
- `/ibvs/debug_frame` - Debug visualization (Image)
- `/ibvs/diagnostics` - Error, velocity, num_features, point_source (Float32MultiArray)
