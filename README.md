# PFE — TurtleBot4 MC-SLAM 

Autonomous SLAM pipeline for TurtleBot4 using ROS2, featuring camera calibration, LiDAR odometry, camera loop closure, and real-time mapping.

---

## Workspace Structure

```
PFE_ws/src/
├── camera_frontend/
├── camera_info_saver/
├── lidar_frontend/
├── slam_common/
├── slam_core/
└── slam_msgs/
```

---

## Build

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
cd ~/PFE/PFE_ws
colcon build --symlink-install
source install/setup.bash
```

---

## Usage

### 1. Simulation 

Launch the Gazebo simulation if you are not running on a physical robot:

```bash
ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py
```

---

### 2. Camera Calibration

**Terminal 1** — Start the camera info saver:

```bash
ros2 run camera_info_saver camera_info_saver --camera-name oakd_rgb --save-path ~/.ros/camera_info/oakd_rgb.yaml --namespace /camera --ros-args --remap /camera/set_camera_info:=/camera/set_camera_info

```

**Terminal 2** — Run the calibrator (use a 5×8 checkerboard, 30mm squares):

```bash
 ros2 run camera_calibration cameracalibrator --size 5x8 --square 0.030 --ros-args -r image:=/oakd/rgb/preview/image_raw -p camera:=/oakd/rgb/preview

```

---

### 3. Run SLAM

```bash
ros2 launch my_slam slam.launch.py
```

---

### 4. Teleoperate the Robot

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args -p stamped:=true
```

---

##  Requirements

- ROS2 Jazzy
- TurtleBot4 packages
- `colcon`, `teleop_twist_keyboard`, `camera_calibration`