# PFE
Turtlebot4 Slam Solutions

cd ~/PFE/PFE_ws
rm -rf build/ install/ log/
colcon build --symlink-install --packages-select my_slam
source install/setup.bash

# Launch Simulation if needed
ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py