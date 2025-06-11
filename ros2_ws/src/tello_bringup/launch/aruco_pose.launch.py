import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_bringup = get_package_share_directory('tello_bringup')
    shared_param_file = os.path.join(pkg_bringup, 'config', 'aruco_pose_config.yaml')

    downward_camera_node = Node(
        package='tello_vision',
        executable='downward_camera',
        name='downward_camera',
        parameters=[shared_param_file],
        output='screen'
    )

    pose_estimator = Node(
        package='tello_vision',
        executable='pose_estimator',
        name='pose_estimator',
        parameters=[shared_param_file],
        output='screen'
    )

    return LaunchDescription([
        downward_camera_node,
        pose_estimator
    ])