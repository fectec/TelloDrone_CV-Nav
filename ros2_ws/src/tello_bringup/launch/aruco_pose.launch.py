import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_bringup = get_package_share_directory('tello_bringup')
    shared_param_file = os.path.join(pkg_bringup, 'config', 'aruco_pose_config.yaml')

    tello_downward_camera_node = Node(
        package='tello_vision',
        executable='tello_downward_camera',
        name='tello_downward_camera',
        parameters=[shared_param_file],
        output='screen'
    )

    aruco_detector_node = Node(
        package='tello_vision',
        executable='aruco_detector',
        name='aruco_detector',
        parameters=[shared_param_file],
        output='screen'
    )

    return LaunchDescription([
        tello_downward_camera_node,
        aruco_detector_node
    ])