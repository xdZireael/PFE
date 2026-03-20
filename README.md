# PFE
Turtlebot4 Slam Solutions

source /opt/ros/$ROS_DISTRO/setup.bash
cd ~/PFE/PFE_ws
rm -rf build/ install/ log/
colcon build --symlink-install --packages-select my_slam
source install/setup.bash

# Launch Simulation if needed
ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py

# Calibrate Camera
ros2 run camera_info_saver camera_info_saver   --camera-name oakd_rgb   --save-path ~/.ros/camera_info/oakd_rgb.yaml   --namespace /camera   --ros-args --remap /camera/set_camera_info:=/camera/set_camera_info

# In another terminal run the calibrator
 ros2 run camera_calibration cameracalibrator   --size 5x8 --square 0.030   --ros-args   -r image:=/oakd/rgb/preview/image_raw   -p camera:=/oakd/rgb/preview

# Run SLAM
ros2 launch my_slam slam.launch.py

# Teleoperate the robot
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p stamped:=true

