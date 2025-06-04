# Navigate to your repository root (adjust the path as needed)
cd ~/TelloDrone_CV-Nav

# Initialize and update submodules
# (this fetches the CoppeliaSim ROS2 Interface submodule located in ros2_ws/src/sim_ros2_interface)
git submodule update --init --recursive

# Checkout the desired branch/tag in the submodule using the -C flag
# (replace with the version of CoppeliaSim you have installed, as seen in the Help/About section)
git -C ros2_ws/src/sim_ros2_interface checkout coppeliasim-v4.10.0-rev0

# (Optional) Install xsltproc, a dependency required for the build process
sudo apt install xsltproc

# (Optional) Install xmlschema to remove warnings
pip3 install xmlschema

# Set the CoppeliaSim root directory (replace with your actual installation path)
export COPPELIASIM_ROOT_DIR=~/path/to/coppeliaSim/folder

# Increase stack size to prevent potential compilation freeze/crash
ulimit -s unlimited

# Navigate to the ROS2 workspace directory and build the workspace
cd ros2_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash

# ---------------------------------------------------------------------------
# The plugin is now ready to be used.
#
# Next steps:
# 1. Navigate to your CoppeliaSim installation folder (replace with your actual installation path):
cd ~/path/to/coppeliaSim/folder
#
# (Optional) Make CoppeliaSim executable:
chmod +x coppeliaSim
#
# 2. Start CoppeliaSim:
./coppeliaSim
#
# 3. Once CoppeliaSim is running, load the ROS2 plugin using Lua or Python:
#    (For example, in Lua:)
#       simROS2 = require('simROS2')
#
# 4. Upon successful ROS2 Interface load, you should see the node /sim_ros2_interface by running:
ros2 node list
# ---------------------------------------------------------------------------
