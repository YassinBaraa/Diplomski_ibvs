# IBVS Package Structure

Restructured to match the detection pipeline package organization.

## Directory Layout

```
ibvs/
├── config/                          # Core configuration (shared)
│   ├── __init__.py
│   └── default_config.yaml
├── controller/                      # Control algorithms
│   ├── __init__.py
│   └── PointController.py
├── feature_extraction/              # Feature detection modules
│   ├── __init__.py
│   ├── AprilTagExtractor.py
│   ├── FASTHarrisExtractor.py
│   └── FeatureSelector.py
├── ibvs_ros/                        # ROS-specific package
│   ├── config/
│   │   └── ibvs_ros.yaml           # ROS runtime parameters
│   ├── launch/
│   │   └── ibvs.launch             # ROS node launcher
│   ├── __init__.py
│   └── ibvs_node.py                # ROS node entry point
├── model/                           # Model weights directory (placeholder)
├── pipeline/                        # Core pipeline orchestration
│   ├── __init__.py
│   ├── IBVSContext.py              # Data context container
│   └── IBVSPipeline.py             # Pipeline orchestrator
├── sources/                         # Data source adapters
│   ├── __init__.py
│   ├── DetectionPipelineSource.py  # Detection pipeline wrapper
│   ├── FrameSource.py              # Abstract base class
│   └── MP4Source.py                # Video file adapter
├── tests/                           # Unit and integration tests
├── CMakeLists.txt                  # Catkin build system
├── package.xml                     # ROS package manifest
├── main.py                         # Standalone (non-ROS) entry point
├── requirements.txt                # Python dependencies
├── setup.py                        # Python setuptools config
└── README.md
```

## Comparison with Detection Pipeline

| Aspect | IBVS | Detection Pipeline |
|--------|------|-------------------|
| **ROS Module** | `ibvs_ros/` | `detection_pipeline_ros/` |
| **Node File** | `ibvs_ros/ibvs_node.py` | `detection_pipeline_ros/detection_node.py` |
| **Config** | `ibvs_ros/config/` | `detection_pipeline_ros/config/` |
| **Launch** | `ibvs_ros/launch/` | `detection_pipeline_ros/launch/` |
| **Root Config** | `config/` | `config/` |
| **Models** | `model/` | `model/` |
| **Standalone** | `main.py` | N/A |

## Key Differences

### IBVS-Specific
- **Dual Mode**: Can run as standalone (non-ROS) or ROS node
- `main.py` - Standalone entry point using CLI arguments
- `controller/` - IBVS control law implementation

### Detection Pipeline Pattern
- **ROS-first**: Designed primarily for ROS
- `detection_pipeline_ros/` - All ROS code in single folder
- Organized by function: `detectors/`, `trackers/`, `postprocessing/`, etc.

### IBVS Improvements
- Adopted ROS package structure (`<package>_ros/`)
- Modular organization: core logic separate from ROS wrapper
- Can be used as library outside ROS

## Migration Steps Completed

✅ Created `ibvs_ros/` folder for ROS-specific code
✅ Moved ROS node to `ibvs_ros/ibvs_node.py`
✅ Moved launch files to `ibvs_ros/launch/`
✅ Moved ROS config to `ibvs_ros/config/`
✅ Created `ibvs_ros/__init__.py`
✅ Created `model/` directory (for future weights)
✅ Added `requirements.txt` and `setup.py`
✅ Updated `CMakeLists.txt` paths
✅ Validated Python syntax

## File Paths in ROS Context

When deployed in catkin workspace (`~/catkin_ws/src/ibvs`):

```bash
# Find/reference from launch files:
$(find ibvs)/ibvs_ros/launch/
$(find ibvs)/ibvs_ros/config/

# Node executable:
pkg="ibvs" type="ibvs_node.py"

# Python imports:
from ibvs.controller import PointController
from ibvs.feature_extraction import AprilTagExtractor
from ibvs.pipeline import IBVSContext
```
