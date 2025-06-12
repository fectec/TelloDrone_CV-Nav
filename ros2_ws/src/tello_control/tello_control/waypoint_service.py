#!/usr/bin/env python3

import sys
import math

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from geometry_msgs.msg import Pose
from std_srvs.srv import Trigger

class WaypointService(Node):
    """
    Service that provides waypoints one by one upon request.
    """
    def __init__(self):
        super().__init__('waypoint_service')
        
        # Declare parameters
        self.declare_parameter('service_name', 'get_next_waypoint')
        self.declare_parameter('frame_id', 'charuco_board')
        
        # Custom waypoints (x, y, z, yaw) - flat list
        self.declare_parameter('custom_waypoints', [
            0.0, 0.0, 0.0, 0.0
        ])
        
        # Retrieve parameters
        self.service_name = self.get_parameter('service_name').value
        self.frame_id = self.get_parameter('frame_id').value
        self.custom_waypoints = self.get_parameter('custom_waypoints').value
        
        # Register parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        # Validate initial parameters
        init_params = [
            Parameter('service_name', Parameter.Type.STRING, self.service_name),
            Parameter('frame_id', Parameter.Type.STRING, self.frame_id),
            Parameter('custom_waypoints', Parameter.Type.DOUBLE_ARRAY, self.custom_waypoints),
        ]
        
        result = self.parameter_callback(init_params)
        if not result.successful:
            raise RuntimeError(f"Parameter validation failed: {result.reason}")
        
        # Create service
        self.waypoint_service = self.create_service(
            Trigger,
            self.service_name,
            self.get_next_waypoint_callback
        )
        
        # State
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.generate_waypoints()
        
        self.get_logger().info(f"WaypointService Start.")
    
    def get_next_waypoint_callback(self, request, response):
        """Service callback to get the next waypoint."""
        if not self.waypoints:
            response.success = False
            response.message = "No waypoints available."
            return response
        
        if self.current_waypoint_idx >= len(self.waypoints):
            response.success = False
            response.message = "Mission complete - no more waypoints."
            self.get_logger().info("Mission complete - all waypoints served.")
            return response
        
        # Get current waypoint
        waypoint = self.waypoints[self.current_waypoint_idx]
        
        # Create response message with waypoint data
        response.success = True
        response.message = (
            f"{waypoint.position.x},{waypoint.position.y},{waypoint.position.z},"
            f"{waypoint.orientation.x},{waypoint.orientation.y},"
            f"{waypoint.orientation.z},{waypoint.orientation.w}"
        )
        
        self.get_logger().info(
            f"Served waypoint {self.current_waypoint_idx + 1}/{len(self.waypoints)}: "
            f"({waypoint.position.x:.2f}, {waypoint.position.y:.2f}, {waypoint.position.z:.2f})."
        )
        
        # Advance to next waypoint
        self.current_waypoint_idx += 1
        
        return response
    
    def generate_waypoints(self):
        """Generate waypoints from custom parameter list."""
        self.waypoints = []
        
        if len(self.custom_waypoints) % 4 != 0:
            self.get_logger().error("Custom waypoints must have 4 values per point (x,y,z,yaw).")
            return
        
        for i in range(0, len(self.custom_waypoints), 4):
            pose = Pose()
            pose.position.x = self.custom_waypoints[i]
            pose.position.y = self.custom_waypoints[i+1]
            pose.position.z = self.custom_waypoints[i+2]
            pose.orientation = self.yaw_to_quaternion(self.custom_waypoints[i+3])
            self.waypoints.append(pose)
        
        self.get_logger().info(f"Generated {len(self.waypoints)} waypoints.")
    
    def yaw_to_quaternion(self, yaw):
        """Convert yaw angle to quaternion."""
        # Assuming roll=0, pitch=0
        qw = math.cos(yaw / 2.0)
        qx = 0.0
        qy = 0.0
        qz = math.sin(yaw / 2.0)
        
        # Return quaternion
        q = Pose().orientation
        q.x = qx
        q.y = qy
        q.z = qz
        q.w = qw
        return q
    
    def parameter_callback(self, params):
        """Handle parameter updates."""
        for param in params:
            name = param.name
            value = param.value
            
            if name in ['service_name', 'frame_id']:
                if not isinstance(value, str):
                    return SetParametersResult(successful=False, reason=f"{name} must be a string.")
                setattr(self, name, value)
                self.get_logger().info(f"Updated {name}: {value}")
            
            elif name == 'custom_waypoints':
                if not isinstance(value, list):
                    return SetParametersResult(successful=False, reason="custom_waypoints must be a list.")
                self.custom_waypoints = value
                # Reset state for new waypoints
                self.current_waypoint_idx = 0
                self.generate_waypoints()
                self.get_logger().info(f"Updated custom_waypoints: {len(self.waypoints)} waypoints.")
        
        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = WaypointService()
    except Exception as e:
        print(f"[FATAL] WaypointService failed to initialize: {e}", file=sys.stderr)
        rclpy.shutdown()
        return
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted with Ctrl+C.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()