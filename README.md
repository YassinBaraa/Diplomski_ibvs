# IBVS Pipeline

Image-Based Visual Servoing pipeline for UAV control. Both standalone and ROS 1 (Noetic) compatible versions.

## Quick Start

### ROS Mode (Recommended)

All code is contained in `ibvs_ros/` - just copy the entire package to your catkin workspace.

**Start detection pipeline (in separate terminal):**
```bash
roslaunch detection_pipeline_ros detection.launch
```

**Start IBVS node:**
```bash
roslaunch ibvs ibvs.launch
```

**View topics:**
```bash
rostopic list | grep ibvs
rostopic echo /ibvs/cmd_vel
rostopic echo /ibvs/error
```

### Standalone Mode (Non-ROS)

**Run with Detection Pipeline:**
```bash
python main.py --source=detection
```

**Run with Test Video:**
```bash
python main.py --source=video --video-path=/path/to/video.mp4
```

## Optional Arguments (Standalone)

- `--max-frames=N` - Process only N frames (default: all)
- `--print-every=N` - Print info every N frames (default: 1)
- `--no-display` - Hide OpenCV window

## Project Structure

```
ibvs/
├── ibvs_ros/                   # Main IBVS ROS package (self-contained)
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
│   │   └── ibvs_ros.yaml       # ROS parameters
│   ├── ros_launch/
│   │   └── ibvs.launch         # ROS launch file
│   ├── __init__.py
│   └── ibvs_node.py            # ROS node entry point
├── model/                       # Model files (placeholder)
├── tests/                       # Unit and integration tests
├── CMakeLists.txt              # Catkin build config
├── package.xml                 # ROS package descriptor
├── main.py                     # Standalone (non-ROS) entry point
├── requirements.txt            # Python dependencies
├── setup.py                    # Setup script
└── README.md
```

## Architecture

- **Primary Source**: Detection pipeline point
- **Validation**: Feature extraction points check
- **Fallback**: Use feature centroid if detection point is None
- **Control**: Error-based proportional velocity scaling
- **Output**: Results published to ROS topics (ROS mode) or stored in `ctx.debug` (standalone)

## ROS Topics

### Subscriptions
- `/detection/frame` (sensor_msgs/Image) - Input frame from detection pipeline
- `/detection/best_candidate` (geometry_msgs/PointStamped) - Detection point (x, y pixels)

### Publications
All topics under `/ibvs/` namespace:

| Topic | Type | Content |
|-------|------|---------|
| `/ibvs/cmd_vel` | geometry_msgs/Twist | Control velocity command |
| `/ibvs/error` | geometry_msgs/Vector3Stamped | Control error in pixels |
| `/ibvs/extracted_points` | sensor_msgs/Image | Frame with extracted features overlaid |
| `/ibvs/debug_frame` | sensor_msgs/Image | Frame with detection point + features |
| `/ibvs/diagnostics` | std_msgs/Float32MultiArray | [error_x, error_y, vel_x, vel_y, num_features, point_source] |

## ROS Parameters

Edit `config/ibvs_ros.yaml` or pass as arguments:

```yaml
controller_gain: 0.5           # Proportional gain
max_feature_motion_px: 50.0    # Feature validation threshold
fallback_enabled: true         # Use centroid if no detection point
fallback_gain_scale: 0.5       # Fallback velocity scaling
enable_visualization: false    # Debug OpenCV window
target_x: null                 # Custom target (null = image center)
target_y: null
```

## Standalone Output Example

```
--- Frame 0 ---
Point Source: detection
Status: ok
Control Error (px): [10.50, -2.30]
Velocity Command: [5.25, -1.15]

--- Frame 5 ---
Point Source: fallback
Status: ok
Control Error (px): [2.10, 0.80]
Velocity Command: [1.05, 0.40]
```

### ROS Setup (Noetic)

#### Prerequisites
```bash
# Install ROS dependencies
sudo apt install ros-noetic-cv-bridge ros-noetic-sensor-msgs ros-noetic-geometry-msgs python3-catkin-tools
```

#### Building the Package

```bash
# Navigate to catkin workspace
cd ~/catkin_ws/src

# Clone or place IBVS package here (should be at ~/catkin_ws/src/ibvs)
# If it's in another location, create a symlink:
# ln -s /path/to/ibvs ~/catkin_ws/src/ibvs

# Build
cd ~/catkin_ws
catkin build ibvs

# Source setup
source devel/setup.bash
```

#### Running ROS Version

**Terminal 1 - Start detection pipeline:**
```bash
roslaunch detection_pipeline_ros detection.launch
```

**Terminal 2 - Start IBVS node:**
```bash
roslaunch ibvs ibvs.launch
```

**Terminal 3 - Monitor topics:**
```bash
# List all IBVS topics
rostopic list | grep ibvs

# Echo specific topics
rostopic echo /ibvs/cmd_vel
rostopic echo /ibvs/error
rostopic echo /ibvs/diagnostics

# Plot error over time (requires rqt)
rqt_plot /ibvs/error/vector/x /ibvs/error/vector/y
```

#### Configuration
Edit `ibvs_ros/config/ibvs_ros.yaml` to change:
- Proportional gain
- Feature validation thresholds
- Fallback behavior
- Target point (for custom tracking)

Override at runtime:
```bash
roslaunch ibvs ibvs.launch controller_gain:=0.8 fallback_enabled:=false
```

## References

- IBVS theory: https://visp-doc.inria.fr/doxygen/visp-daily/tutorial-ibvs.html
- Camera specs: https://joy-it.net/files/files/Produkte/rb-camera_JT/rb-camera_JT_Datasheet_2021-02-09.pdf
- AprilTag: https://pyimagesearch.com/2020/11/02/apriltag-with-python/
- Harris corners: https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html
