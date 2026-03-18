# PFE
Turtlebot4 Slam Solutions

cd ~/PFE/PFE_ws
rm -rf build/ install/ log/
colcon build --symlink-install --packages-select my_slam
source install/setup.bash

# Launch Simulation if needed
ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py

# Calibrate Camera
ros2 run camera_info_saver camera_info_saver \
  --camera-name oakd_rgb \
  --save-path ~/.ros/camera_info/oakd_rgb.yaml \
  --namespace /oakd/rgb/preview

# In another terminal run the calibrator
ros2 run camera_calibration cameracalibrator   --size 12x18 --square 0.015   --no-service-check   --ros-args -r image:=/oakd/rgb/preview/image_raw -p camera:=/oakd/rgb/preview
