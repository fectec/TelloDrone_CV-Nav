#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_bringup = get_package_share_directory('tello_bringup')
    shared_param_file = os.path.join(pkg_bringup, 'config', 'pose_trajectory_config.yaml')
    
    pose_estimator = Node(
        package='tello_vision',
        executable='pose_estimator',
        name='pose_estimator',
        parameters=[shared_param_file],
        output='screen'
    )
    
    waypoints_publisher = Node(
        package='tello_control',
        executable='waypoints_publisher',
        name='waypoints_publisher',
        parameters=[shared_param_file],
        output='screen'
    )
    
    pose_controller = Node(
        package='tello_control',
        executable='pose_controller',
        name='pose_controller',
        parameters=[shared_param_file],
        output='screen'
    )
    
    return LaunchDescription([
        pose_estimator,
        waypoints_publisher,
        pose_controller
    ])